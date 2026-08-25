"""
learning/estep.py
=================
E-step coordinator: infer q(S^(k), H^(k)) at scale k given current parameters.

Algorithm (for scale k)
------------------------
Input:  data at scale k, ScaleParams Θ_k, MultiScaleDBNGraph G_k,
        PosteriorSummaries from parent scale k+1 (or None for k=K).

1.  Compute emission log-likelihoods log p(x_t | S_t=m, h̄_t) for all t, m.
    - Coarsest scale:  ZINB count + NB duration (Eq. 220-289).
    - Finer scales:    Dir-Mult allocation score (Eq. 300-412).
    At the first inner iteration h̄_t = 0 (or from previous outer iteration).

2.  Build log_init and log_trans from RegimeParams and optional parent-γ.
    (Eq. 177-198 for the transition model.)  These do not depend on h̄, so they
    are built once per E-step, outside the inner loop.

3.  Run RegimePosterior.forward_backward() → γ, ξ, log_Z.

4.  If r > 0:
    a.  Build Laplace pseudo-observations (z_t, R_t) for the factor
        using the coarsest-scale Hessian (or prior-only for finer scales).
    b.  Run FactorPosterior.smoother() → m_H, Σ_H.
    c.  Update emission log-likelihoods with new h̄ = m_H.
    d.  Repeat 3-c until max_inner_iter or ELBO convergence.

5.  Return PosteriorSummaries(γ, ξ, m_H, Σ_H, log_Z).

Implementation note (vectorisation)
-----------------------------------
Every emission builder here is fully vectorised over (epoch, regime, label).
The previous per-(t, m, ℓ) Python loops dominated the wall-clock of a fit;
the arrays involved are small ((T, M, L) doubles) so materialising them is far
cheaper than looping.

Implementation note (finer-scale likelihood)
--------------------------------------------
The finer-scale emission term is the Dirichlet-Multinomial allocation
log-likelihood

    Σ_ℓ [ log Γ(X_{t,ℓ} + κ ω_{t,ℓ}(m)) - log Γ(κ ω_{t,ℓ}(m)) ],

with ω the block softmax of the allocation scores, evaluated at the
regime-averaged score for the other children of the block (a mean-field
treatment of a denominator that couples the whole block).  Two properties that
the earlier unnormalised score Σ_ℓ X_{t,ℓ} η_{t,ℓ}(m) lacked matter here.  It
is bounded, so log_Z, the ELBO, val_loss and the finer-scale structure scores
mean something; and it carries the overdispersion of a finite κ, without which
the score overstates the evidence by about an order of magnitude and spurious
lagged edges clear the BIC penalty.  The data-only constant
log Γ(κ) - log Γ(N + κ) plus the multinomial coefficient is dropped, since it
cancels in every comparison.

Implementation note (provenance and scorability)
------------------------------------------------
Two row sets travel together and neither folds into the other.  `recorded` is
provenance: the epochs the record holds, whose counts make up a block sum and
whose scores compete in the block softmax.  `scored` is the subset a likelihood
term sums, which is `recorded` eroded forward by the declared lag budget, since
an epoch whose lagged feature reaches into a gap reads a value nobody wrote
down.  A recorded epoch outside `scored` keeps its count in its block and
contributes no term of its own, so dropping it from the sum leaves every
sibling's allocation where it was.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.special import digamma, gammaln

from ..core.blocks import BlockIndex
from ..core.graph import MultiScaleDBNGraph
from ..core.mask import declared_rows
from ..core.scales import ScaleHierarchy
from ..core.variables import MultiScaleData
from ..inference.posterior import (
    FactorPosterior,
    PosteriorSummaries,
    RegimePosterior,
)
from ..model.emissions import _sigmoid, nb_log_pmf, zinb_log_pmf
from ..model.params import FineCountParams, FineDurParams, ScaleParams

LagKey = Tuple[int, int]

# Guard rails for the log-link: exp() of anything larger overflows and
# poisons the whole likelihood with inf/nan.
_LOG_CLIP = 20.0


def rows_of(rows: Optional[np.ndarray], n_epochs: int) -> np.ndarray:
    """The epochs a term may run over, all of them when no mask is given."""
    if rows is None:
        return np.ones(n_epochs, dtype=bool)
    return np.asarray(rows, dtype=bool)


def recorded_and_scored(
    recorded: Optional[np.ndarray],
    scored: Optional[np.ndarray],
    n_epochs: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    The recorded epochs and the scored ones, each filled in when left out.

    A block sum runs over what the record holds.  A likelihood term runs over
    the rows the search declared scorable, which is the recorded set eroded
    forward by the declared lag budget.  The second is contained in the first,
    so a caller handing them over crossed is passing a dilated row set where
    provenance is meant, and the containment check below is what catches it.

    Args:
        recorded: (n_epochs,) bool provenance, `None` for a whole record.
        scored: (n_epochs,) bool scorable rows, `None` when every recorded
            epoch is scorable.
        n_epochs: how many epochs the scale holds.

    Returns:
        (recorded, scored), both (n_epochs,) bool.

    Raises:
        ValueError: when an epoch is scored that the record does not hold.
    """
    kept = rows_of(recorded, n_epochs)
    rows = kept if scored is None else np.asarray(scored, dtype=bool)
    crossed = int(np.count_nonzero(rows & ~kept))
    if crossed:
        raise ValueError(
            f"{crossed} epoch(s) would be scored while the record does not hold "
            "them; the scored rows are the recorded ones eroded forward by the "
            "declared lag budget and can only be a subset of them"
        )
    return kept, rows


# ---------------------------------------------------------------------------
# Lagged-parent features
# ---------------------------------------------------------------------------


def lag_feature(
    arr: np.ndarray,  # (T, L) counts or durations
    key: LagKey,
    parent_sets: dict,  # {dst_ell: {(src_ell, d), ...}}
    L: int,
) -> np.ndarray:
    """
    Design matrix of one lagged edge: (T, L).

    `out[t, dst] = log(1 + arr[t-d, src])` when `(src, d) ∈ P_dst` and
    `t ≥ d`, else 0.
    """
    src_ell, d = key
    T = arr.shape[0]
    active = np.array(
        [1.0 if key in parent_sets.get(dst, ()) else 0.0 for dst in range(L)]
    )
    if d <= 0 or d >= T or not active.any():
        return np.zeros((T, L))
    col = np.zeros(T)
    col[d:] = np.log1p(arr[:-d, src_ell].astype(float))
    return np.outer(col, active)


def lag_features(
    arr: np.ndarray,
    keys,
    parent_sets: dict,
    L: int,
) -> Dict[LagKey, np.ndarray]:
    """Design matrices for a set of lagged edges: {key: (T, L)}."""
    return {key: lag_feature(arr, key, parent_sets, L) for key in keys}


def lag_contrib(
    arr: np.ndarray,
    coeff: Dict[LagKey, np.ndarray],  # {(src, d): (L,)}
    parent_sets: dict,
    L: int,
    feats: Optional[Dict[LagKey, np.ndarray]] = None,
) -> np.ndarray:
    """
    Total lagged contribution Σ_{(ℓ',d)∈P_ℓ} coef[ℓ] log(1+arr_{t-d}^{ℓ'}): (T, L).
    """
    out = np.zeros((arr.shape[0], L))
    for key, coef in coeff.items():
        f = feats[key] if feats is not None else lag_feature(arr, key, parent_sets, L)
        out += f * coef
    return out


# Backwards-compatible per-epoch helpers (kept for the public API / tests;
# the hot paths use the vectorised functions above).


def _agg_lag_X(
    t: int,
    X: np.ndarray,  # (T, L)
    parent_sets: dict,  # {dst_ell: {(src_ell, d), ...}}
    betaLag: dict,  # {(src_ell, d): (L,)}
    L: int,
) -> np.ndarray:
    """Compute Σ_{(ℓ',d) ∈ P_ℓ} β_{ℓ'd}[ℓ] log(1 + X_{t-d}^{ℓ'}) for all ℓ."""
    result = np.zeros(L)
    for dst_ell in range(L):
        for src_ell, d in parent_sets.get(dst_ell, set()):
            t_lag = t - d
            if t_lag >= 0:
                coef = betaLag.get((src_ell, d), np.zeros(L))
                result[dst_ell] += coef[dst_ell] * np.log1p(float(X[t_lag, src_ell]))
    return result


def _agg_lag_Y(
    t: int,
    Y: np.ndarray,  # (T, L)
    parent_sets: dict,
    alphaLag: dict,  # {(src_ell, d): (L,)}
    L: int,
) -> np.ndarray:
    result = np.zeros(L)
    for dst_ell in range(L):
        for src_ell, d in parent_sets.get(dst_ell, set()):
            t_lag = t - d
            if t_lag >= 0:
                coef = alphaLag.get((src_ell, d), np.zeros(L))
                result[dst_ell] += coef[dst_ell] * np.log1p(float(Y[t_lag, src_ell]))
    return result


# ---------------------------------------------------------------------------
# Coarsest-scale emission log-likelihood
# ---------------------------------------------------------------------------


def coarse_linear_predictors(
    X_K: np.ndarray,  # (T, L)
    h_bar: np.ndarray,  # (T, r)
    ctx: np.ndarray,  # (T, d_ctx)
    params: ScaleParams,
    lag_x: np.ndarray,  # (T, L)
    lag_y: np.ndarray,  # (T, L)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Linear predictors of the coarsest-scale emission, shape (T, M, L) each.

    Returns
    -------
    log_mu   : count log-mean
    logit_pi : zero-inflation logit
    log_nu   : duration log-mean
    """
    cp = params.coarse_count
    dp = params.coarse_dur
    r = params.r

    base_mu = cp.beta0 + lag_x
    base_pi = np.broadcast_to(cp.gamma0, X_K.shape).copy()
    base_nu = dp.alpha0 + lag_y + dp.alphaX * np.log1p(X_K.astype(float))

    if r > 0:
        base_mu = base_mu + h_bar @ cp.betaH
        base_pi = base_pi + h_bar @ cp.gammaH
        base_nu = base_nu + h_bar @ dp.alphaH
    if ctx.shape[1] > 0:
        if cp.betaCtx.shape[0] > 0:
            base_mu = base_mu + ctx @ cp.betaCtx
        if cp.gammaCtx.shape[0] > 0:
            base_pi = base_pi + ctx @ cp.gammaCtx
        if dp.alphaCtx.shape[0] > 0:
            base_nu = base_nu + ctx @ dp.alphaCtx

    log_mu = base_mu[:, None, :] + cp.betaS[None, :, :]
    logit_pi = base_pi[:, None, :] + cp.gammaS[None, :, :]
    log_nu = base_nu[:, None, :] + dp.alphaS[None, :, :]
    return log_mu, logit_pi, log_nu


def coarse_ll_tml(
    X_K: np.ndarray,
    Y_K: np.ndarray,
    h_bar: np.ndarray,
    ctx: np.ndarray,
    params: ScaleParams,
    lag_x: np.ndarray,
    lag_y: np.ndarray,
) -> np.ndarray:
    """Per-label coarsest-scale emission log-likelihood: (T, M, L)."""
    cp = params.coarse_count
    dp = params.coarse_dur

    log_mu, logit_pi, log_nu = coarse_linear_predictors(
        X_K, h_bar, ctx, params, lag_x, lag_y
    )
    mu = np.exp(np.clip(log_mu, -_LOG_CLIP, _LOG_CLIP))
    pi = _sigmoid(logit_pi)
    nu = np.maximum(np.exp(np.clip(log_nu, -_LOG_CLIP, _LOG_CLIP)), 1.0)

    x = X_K[:, None, :].astype(float)
    y = Y_K[:, None, :].astype(float)

    ll = zinb_log_pmf(x, pi, cp.r_disp, mu)
    active = X_K[:, None, :] > 0
    ll = ll + np.where(active, nb_log_pmf(y, dp.s_disp, nu), 0.0)
    return ll


def build_log_like_coarsest(
    X_K: np.ndarray,  # (T, L)
    Y_K: np.ndarray,  # (T, L)
    h_bar: np.ndarray,  # (T, r)
    ctx: np.ndarray,  # (T, d_ctx)
    params: ScaleParams,  # is_coarsest == True
    lag_x: Optional[np.ndarray] = None,
    lag_y: Optional[np.ndarray] = None,
    scored: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute log p(X_t, Y_t | S_t=m, H_t=h̄_t) for all t, m.

    Each epoch carries its own term here and no block sum couples one to the
    next, so provenance has nothing to say and `scored` alone fixes the rows.
    An epoch outside it gets a row of zeros, which is the emission-equals-one
    rule: `forward_backward` adds the row into log_alpha, log_beta and log_xi,
    so a zero row leaves the chain bridging the gap on its transitions alone.

    Returns
    -------
    log_like : (T, M)
    """
    L = X_K.shape[1]
    if lag_x is None:
        lag_x = lag_contrib(X_K, params.coarse_count.betaLag, params.parent_sets, L)
    if lag_y is None:
        lag_y = lag_contrib(Y_K, params.coarse_dur.alphaLag, params.parent_sets, L)
    ll = coarse_ll_tml(X_K, Y_K, h_bar, ctx, params, lag_x, lag_y).sum(axis=2)
    return np.where(rows_of(scored, X_K.shape[0])[:, None], ll, 0.0)


def expected_emission_ll_coarse(
    X_K: np.ndarray,
    Y_K: np.ndarray,
    h_bar: np.ndarray,  # (T, r)
    gamma: np.ndarray,  # (T, M)
    params: ScaleParams,
    ctx: np.ndarray,
    lag_x: np.ndarray,
    lag_y: np.ndarray,
) -> np.ndarray:
    """Σ_m γ_t(m) log p(X_t, Y_t | m, h̄_t) for every epoch: (T,)."""
    ll_tm = coarse_ll_tml(X_K, Y_K, h_bar, ctx, params, lag_x, lag_y).sum(axis=2)
    return np.einsum("tm,tm->t", gamma, ll_tm)


# ---------------------------------------------------------------------------
# Finer-scale allocation log-likelihood
# ---------------------------------------------------------------------------


def fine_scores(
    X_k: np.ndarray,  # (H_k, L)
    X_par: np.ndarray,  # (H_{k+1}, L)
    Y_par: np.ndarray,  # (H_{k+1}, L)
    h_bar: np.ndarray,  # (H_k, r)
    ctx: np.ndarray,  # (H_k, d_ctx)
    params: ScaleParams,
    block: BlockIndex,
    lag_x: np.ndarray,  # (H_k, L)
    lag_y: np.ndarray,  # (H_k, L)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Allocation scores η (counts) and ξ (durations): (T, M, L) each.
    """
    fp = params.fine_count
    fdp = params.fine_dur
    r = params.r

    log_x_par = np.log1p(block.broadcast(X_par).astype(float))  # (T, L)
    log_y_par = np.log1p(block.broadcast(Y_par).astype(float))  # (T, L)

    # What earlier children of this block have already taken.  The children
    # compete for a fixed parent total, and without this term the only way the
    # structure search can express that competition is a lagged self edge.
    dep_x = np.log1p(block.exclusive_cumsum(X_k.astype(float)))  # (T, L)

    base_eta = (
        fp.beta0
        + fp.betaP * log_x_par
        + fp.betaD * log_y_par
        + fp.betaDep * dep_x
        + lag_x
    )
    base_xi = (
        fdp.alpha0
        + fdp.alphaX * np.log1p(X_k.astype(float))
        + fdp.alphaP * log_y_par
        + fdp.alphaDep * dep_x
        + lag_y
    )
    if r > 0:
        base_eta = base_eta + h_bar @ fp.betaH
        base_xi = base_xi + h_bar @ fdp.alphaH
    if ctx.shape[1] > 0:
        if fp.betaCtx.shape[0] > 0:
            base_eta = base_eta + ctx @ fp.betaCtx
        if fdp.alphaCtx.shape[0] > 0:
            base_xi = base_xi + ctx @ fdp.alphaCtx

    eta = base_eta[:, None, :] + fp.betaS[None, :, :]
    xi = base_xi[:, None, :] + fdp.alphaS[None, :, :]
    return eta, xi


def block_softmax(
    scores: np.ndarray,  # (T, L) or (T, M, L)
    block: BlockIndex,
    mask: Optional[np.ndarray] = None,  # (T, L) bool - participating epochs
) -> np.ndarray:
    """Allocation probabilities ω: softmax of `scores` within each block."""
    s = scores if mask is None else np.where(mask, scores, -np.inf)
    logZ = block.broadcast(block.logsumexp(s))
    ok = np.isfinite(logZ) & np.isfinite(s)
    # Avoid (-inf) − (-inf) inside the exponent for empty/masked blocks.
    return np.where(ok, np.exp(np.where(ok, s, 0.0) - np.where(ok, logZ, 0.0)), 0.0)


def dirmult_ll(x: np.ndarray, omega: np.ndarray, conc: float) -> np.ndarray:
    """
    Per-epoch Dirichlet-Multinomial contribution log Γ(x + κω) − log Γ(κω).

    The block-constant log Γ(κ) − log Γ(N + κ) and the multinomial coefficient
    are dropped: they depend on the data only, so they shift every score by the
    same amount.  As κ → ∞ this tends to x·log ω, the multinomial limit.
    """
    a = np.maximum(conc * omega, 1e-10)
    return gammaln(x + a) - gammaln(a)


def dirmult_block_ll(
    x: np.ndarray,  # (T, L) counts
    omega: np.ndarray,  # (T, L) allocation probabilities
    conc: float,
    block: BlockIndex,
) -> float:
    """
    Complete Dirichlet-Multinomial log-likelihood of one allocation channel.

    Unlike :func:`dirmult_ll` this keeps the block-constant term
    log Γ(κ) − log Γ(N + κ), which the per-epoch version drops.  The constant
    depends on the concentration, so estimating κ requires it.
    """
    a = np.maximum(conc * omega, 1e-10)
    per_epoch = gammaln(x + a) - gammaln(a)
    totals = block.sum(x)  # (n_blocks, L)
    const = gammaln(conc) - gammaln(totals + conc)
    return float(per_epoch.sum() + const.sum())


def fit_concentration(
    x: np.ndarray,
    omega: np.ndarray,
    block: BlockIndex,
    lo: float = 0.5,
    hi: float = 2000.0,
    n_iter: int = 40,
) -> float:
    """
    Maximise the Dirichlet-Multinomial log-likelihood over the concentration.

    Golden-section search on log κ.  The likelihood is unimodal in κ for fixed
    allocation probabilities, and the search needs no derivative, which keeps it
    stable when a block holds a single count.
    """
    inv_phi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = np.log(lo), np.log(hi)
    c, d = b - inv_phi * (b - a), a + inv_phi * (b - a)
    fc = dirmult_block_ll(x, omega, float(np.exp(c)), block)
    fd = dirmult_block_ll(x, omega, float(np.exp(d)), block)
    for _ in range(n_iter):
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - inv_phi * (b - a)
            fc = dirmult_block_ll(x, omega, float(np.exp(c)), block)
        else:
            a, c, fc = c, d, fd
            d = a + inv_phi * (b - a)
            fd = dirmult_block_ll(x, omega, float(np.exp(d)), block)
        if b - a < 1e-3:
            break
    return float(np.exp(0.5 * (a + b)))


def dirmult_grad(
    x: np.ndarray,  # (T, L)
    omega: np.ndarray,  # (T, L)
    conc: float,
    block: BlockIndex,
) -> np.ndarray:
    """
    d/dη of Σ_c Σ_{t∈ch(c)} log Γ(x_t + κω_t) − log Γ(κω_t):  (T, L).

        ∂/∂η_s = κ ω_s [ g_s − Σ_{t∈ch(c)} ω_t g_t ],
        g_t = ψ(x_t + κω_t) − ψ(κω_t)

    (the softmax Jacobian applied to the Dir-Mult score).
    """
    a = np.maximum(conc * omega, 1e-10)
    g = digamma(x + a) - digamma(a)
    weighted = block.broadcast(block.sum(omega * g))
    return conc * omega * (g - weighted)


def leave_one_out_omega(
    scores_m: np.ndarray,  # (T, M, L)  per-regime scores
    scores_bar: np.ndarray,  # (T, L)     regime-averaged scores
    block: BlockIndex,
    mask: Optional[np.ndarray] = None,  # (T, L) bool
) -> np.ndarray:
    """
    Allocation probability of each epoch *under each regime*: (T, M, L).

    Child t is scored with η_t(m) while its block-mates keep their
    regime-averaged score η̄ - the mean-field treatment of a softmax whose
    denominator couples the whole block.
    """
    s_bar = scores_bar if mask is None else np.where(mask, scores_bar, -np.inf)
    mx = block.broadcast(block.max(s_bar))
    mx = np.where(np.isfinite(mx), mx, 0.0)
    e_bar = np.where(np.isfinite(s_bar), np.exp(s_bar - mx), 0.0)  # (T, L)
    A = block.broadcast(block.sum(e_bar))  # (T, L)
    others = np.maximum(A - e_bar, 0.0)[:, None, :]  # (T, 1, L)
    e_m = np.exp(np.clip(scores_m - mx[:, None, :], -_LOG_CLIP, _LOG_CLIP))
    if mask is not None:
        e_m = np.where(mask[:, None, :], e_m, 0.0)
    denom = others + e_m
    return np.where(denom > 0.0, e_m / np.maximum(denom, 1e-300), 0.0)


def build_log_like_fine(
    X_k: np.ndarray,  # (H_k, L)
    Y_k: np.ndarray,  # (H_k, L)
    X_par: np.ndarray,  # (H_{k+1}, L)
    Y_par: np.ndarray,  # (H_{k+1}, L)
    h_bar: np.ndarray,  # (H_k, r)
    ctx: np.ndarray,  # (H_k, d_ctx)
    params: ScaleParams,  # finer scale
    hier: ScaleHierarchy,
    k: int,  # finer scale index (1 <= k < K)
    gamma: Optional[np.ndarray] = None,  # (H_k, M) for the normaliser
    block: Optional[BlockIndex] = None,
    lag_x: Optional[np.ndarray] = None,
    lag_y: Optional[np.ndarray] = None,
    recorded: Optional[np.ndarray] = None,
    scored: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Emission log-likelihood of the finer-scale allocation.

    For each epoch t at scale k and regime m this is the Dirichlet-Multinomial
    allocation log-likelihood of Eq. 300-412,

        Σ_ℓ [ log Γ(X_{t,ℓ} + κ ω_{t,ℓ}(m)) − log Γ(κ ω_{t,ℓ}(m)) ]
      + Σ_ℓ [ log Γ(Y_{t,ℓ} + τ ρ_{t,ℓ}(m)) − log Γ(τ ρ_{t,ℓ}(m)) ],

    up to an additive data-only constant, with ω/ρ the block softmaxes of the
    allocation scores.  Keeping the finite concentration κ (rather than the
    multinomial limit) matters: the allocation is genuinely overdispersed, and
    a likelihood that ignores that is overconfident by roughly an order of
    magnitude - enough to make spurious lagged edges look significant.

    An epoch outside `recorded` holds nothing, so it leaves the block
    competition and its count leaves the block sum.  An epoch that is recorded
    and outside `scored` was written down and cannot be summed, so it keeps its
    place in the block competition and its count in the block sum while its own
    term drops.  Either way the count the term reads is zero and Dir-Mult
    returns zero on it, so the row comes out at zero and the
    emission-equals-one rule holds.

    Returns
    -------
    log_like : (H_k, M)
    """
    L = X_k.shape[1]
    M = params.M
    if block is None:
        block = BlockIndex.from_hierarchy(hier, k)
    if lag_x is None:
        lag_x = lag_contrib(X_k, params.fine_count.betaLag, params.parent_sets, L)
    if lag_y is None:
        lag_y = lag_contrib(Y_k, params.fine_dur.alphaLag, params.parent_sets, L)
    if gamma is None:
        gamma = np.full((X_k.shape[0], M), 1.0 / M)

    eta, xi = fine_scores(X_k, X_par, Y_par, h_bar, ctx, params, block, lag_x, lag_y)
    kept, rows = recorded_and_scored(recorded, scored, X_k.shape[0])
    seen = np.broadcast_to(kept[:, None], X_k.shape)
    summed = np.broadcast_to(rows[:, None], X_k.shape)
    active = (X_k > 0) & seen
    eta_bar = np.einsum("tm,tml->tl", gamma, eta)
    xi_bar = np.einsum("tm,tml->tl", gamma, xi)

    omega = leave_one_out_omega(eta, eta_bar, block, mask=seen)
    rho = leave_one_out_omega(xi, xi_bar, block, mask=active)

    x = np.where(summed, X_k, 0).astype(float)[:, None, :]
    y = np.where(active & summed, Y_k, 0.0).astype(float)[:, None, :]
    ll = dirmult_ll(x, omega, params.fine_count.kappa).sum(axis=2)
    ll = ll + dirmult_ll(y, rho, params.fine_dur.tau).sum(axis=2)
    return ll


# ---------------------------------------------------------------------------
# E-step coordinator
# ---------------------------------------------------------------------------


class EStep:
    """
    E-step: infer q(S^(k), H^(k)) at scale k.

    Alternates RegimePosterior.forward_backward() and FactorPosterior.smoother()
    until the log-Z change is below `tol` or `max_inner_iter` is reached.
    """

    def __init__(
        self,
        max_inner_iter: int = 10,
        tol: float = 1e-4,
        use_laplace: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        max_inner_iter : maximum alternating iterations between γ and H̄ updates
        tol            : *relative* convergence threshold on log_Z
        use_laplace    : if True, use Laplace pseudo-obs for coarsest-scale H
        """
        self.max_inner_iter = max_inner_iter
        self.tol = tol
        self.use_laplace = use_laplace

    # ------------------------------------------------------------------

    def _budget(self, k: int) -> Optional[int]:
        """The declared lag budget at scale k, or None when none was set."""
        budget = getattr(self, "lag_budget", None)
        return None if budget is None else int(budget[k - 1])

    def run(
        self,
        k: int,  # 1-indexed scale
        data: MultiScaleData,
        params: ScaleParams,
        graph: MultiScaleDBNGraph,  # not used in E-step but kept for API symmetry
        parent_summaries: Optional[PosteriorSummaries] = None,  # k+1 posteriors
    ) -> PosteriorSummaries:
        """
        Run the E-step at scale k.

        Parameters
        ----------
        k                : scale index (1 = finest, K = coarsest)
        data             : full multiscale observed data
        params           : ScaleParams for scale k
        graph            : structural graph (not used in E-step)
        parent_summaries : PosteriorSummaries from scale k+1 (or None if k=K)

        Returns
        -------
        PosteriorSummaries for scale k
        """
        K = data.K
        hier = data.hierarchy
        X_k = data.get_X(k)  # (H_k, L)
        Y_k = data.get_Y(k)  # (H_k, L)
        ctx_k = data.get_C(k)  # (H_k, d_ctx)

        T = hier.n_epochs(k)
        L = params.L
        M = params.M
        r = params.r
        is_coarsest = params.is_coarsest

        # Parent-scale γ for cross-scale coupling, expanded to fine epochs:
        # build_log_trans indexes parent_gamma[t] with fine-scale t, so map
        # each fine epoch to its coarse parent.
        if parent_summaries is not None and k < K:
            parent_gamma = parent_summaries.gamma[hier.parent_maps[k - 1]]
        else:
            parent_gamma = None

        # ---- Terms that do not depend on h̄ or γ: build once ---------------
        log_init, log_trans = RegimePosterior.build_log_trans(
            params.regime, ctx_k, parent_gamma=parent_gamma
        )
        if is_coarsest:
            lag_x = lag_contrib(X_k, params.coarse_count.betaLag, params.parent_sets, L)
            lag_y = lag_contrib(Y_k, params.coarse_dur.alphaLag, params.parent_sets, L)
            block = None
            X_par = Y_par = None
        else:
            lag_x = lag_contrib(X_k, params.fine_count.betaLag, params.parent_sets, L)
            lag_y = lag_contrib(Y_k, params.fine_dur.alphaLag, params.parent_sets, L)
            block = BlockIndex.from_hierarchy(hier, k)
            X_par = data.get_X(k + 1)
            Y_par = data.get_Y(k + 1)

        # An epoch whose lagged feature reaches into a gap reads a value nobody
        # recorded, so the mask is eroded forward before any term runs over it.
        # The erosion uses the budget the search declared, which is the same
        # for every model compared on this data.  The uneroded mask travels
        # alongside, since the block sums of a finer scale are over what the
        # record holds and dilating them would move a recorded sibling.
        recorded = data.get_observed(k)
        rows = declared_rows(recorded, self._budget(k), "the E-step")

        # Initialise H estimate
        h_bar = np.zeros((T, r)) if r > 0 else np.empty((T, 0))
        Sigma_H = np.tile(np.eye(r) if r > 0 else np.empty((0, 0)), (T, 1, 1))

        prev_log_Z = -np.inf
        gamma = np.full((T, M), 1.0 / M)
        xi = None
        log_Z = 0.0

        for iteration in range(self.max_inner_iter):
            # ---- Step A: compute emission log-likelihoods ---------------
            if is_coarsest:
                log_like = build_log_like_coarsest(
                    X_k,
                    Y_k,
                    h_bar,
                    ctx_k,
                    params,
                    lag_x=lag_x,
                    lag_y=lag_y,
                    scored=rows,
                )
            else:
                log_like = build_log_like_fine(
                    X_k,
                    Y_k,
                    X_par,
                    Y_par,
                    h_bar,
                    ctx_k,
                    params,
                    hier,
                    k,
                    gamma=gamma,
                    block=block,
                    lag_x=lag_x,
                    lag_y=lag_y,
                    recorded=recorded,
                    scored=rows,
                )

            # ---- Step B: forward-backward → γ, ξ, log_Z -----------------
            gamma, xi, log_Z = RegimePosterior.forward_backward(
                log_like, log_init, log_trans
            )

            # ---- Step C: update H estimate (Kalman smoother) ------------
            if r > 0:
                if self.use_laplace and is_coarsest:
                    # A masked epoch contributes no curvature and no gradient,
                    # so its pseudo-observation carries no information and the
                    # smoother bridges it on the transition alone.
                    pseudo_obs, pseudo_R = FactorPosterior.compute_pseudo_obs_coarse(
                        X_k,
                        Y_k,
                        h_bar,
                        np.where(rows[:, None], gamma, 0.0),
                        params,
                        ctx_k,
                        lag_x,
                        lag_y,
                    )
                    h_bar, Sigma_H = FactorPosterior.smoother(
                        gamma, params.factor, ctx_k, pseudo_obs, pseudo_R
                    )
                else:
                    # Prior-only smoother (no direct observation of H)
                    h_bar, Sigma_H = FactorPosterior.smoother(
                        gamma, params.factor, ctx_k
                    )

            # ---- Convergence check --------------------------------------
            # With r=0 at the coarsest scale nothing feeds back into the
            # emissions, so one pass is exact.  At a finer scale the block
            # normaliser depends on gamma even when r=0, so the two updates
            # still have to be iterated to a fixed point.
            if r == 0 and is_coarsest:
                break
            delta = abs(log_Z - prev_log_Z)
            if iteration > 0 and delta < self.tol * (1.0 + abs(log_Z)):
                break
            prev_log_Z = log_Z

        if r == 0:
            h_bar = np.empty((T, 0))
            Sigma_H = np.empty((T, 0, 0))

        return PosteriorSummaries(
            gamma=gamma,
            xi=xi,
            m_H=h_bar,
            Sigma_H=Sigma_H,
            log_Z=float(log_Z),
        )
