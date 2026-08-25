"""
learning/mstep.py
=================
M-step: update all Θ_k given posterior summaries q^new.

Sub-blocks
----------
update_regime_transitions  - softmax MLE via gradient ascent (Adam)
update_factor_params       - closed-form Baum-Welch-style LGSSM update
update_coarse_count_params - ZINB regression via Adam on expected log-likelihood
update_coarse_dur_params   - NB regression via Adam
update_allocation_scores   - DirMult count/duration score regression via Adam
MStep.run                  - dispatches all sub-blocks for one scale

Mathematical references
-----------------------
* M-step definition:  Eq. 690-695 of the manuscript.
* Regime transition update:  softmax MLE, Eq. 4.1.
* LGSSM closed form:  regime-switching Baum-Welch (standard textbook result).
* ZINB / NB gradient:  Eq. 4.3-4.4.
* Allocation score gradient:  Eq. 4.5.

Design note
-----------
Every parameter update is independent of the others given the posteriors,
so the sub-blocks can be called in any order.  Adam is used as a unified
optimizer for all gradient-based updates; closed-form updates are used where
they exist (LGSSM blocks).

Implementation notes
--------------------
* All gradients are accumulated with array operations over (epoch, regime,
  label) instead of Python loops - same mathematics, ~50x less wall-clock.
* Adam's per-step displacement is ≈ `lr` regardless of gradient scale, so a
  full M-step can move a parameter by at most `n_steps × lr`.  With the old
  default (30 × 1e-3 = 0.03) parameters were effectively frozen at their
  initial values, which is why fitted models never improved on their
  initialisation; the defaults below are set so an M-step can actually move.
* The allocation gradient is the multinomial gradient `x_t − N_c ω_t`
  (count-weighted).  The previous `x_t/N_c − ω_t` down-weighted busy parent
  blocks by their own total.
"""

from __future__ import annotations

from copy import copy, deepcopy
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.special import digamma, gammaln

from ..core.blocks import BlockIndex
from ..core.graph import MultiScaleDBNGraph
from ..core.mask import declared_rows
from ..core.scales import ScaleHierarchy
from ..core.variables import MultiScaleData
from ..inference.posterior import PosteriorSummaries, logsumexp, regime_offsets
from ..model.emissions import _sigmoid, nb_log_pmf, zinb_log_pmf
from ..model.params import (
    CoarseCountParams,
    CoarseDurParams,
    FactorParams,
    FineCountParams,
    FineDurParams,
    RegimeParams,
    ScaleParams,
)
from .estep import (
    _LOG_CLIP,
    _agg_lag_X,
    _agg_lag_Y,
    block_softmax,
    coarse_linear_predictors,
    coarse_ll_tml,
    dirmult_block_ll,
    dirmult_grad,
    dirmult_ll,
    fine_scores,
    fit_concentration,
    lag_contrib,
    lag_features,
    recorded_and_scored,
    rows_of,
)

# ---------------------------------------------------------------------------
# Adam optimizer (stateless, functional style)
# ---------------------------------------------------------------------------


class _Adam:
    """Minimal Adam optimizer for one parameter array."""

    def __init__(
        self,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m: Optional[np.ndarray] = None
        self.v: Optional[np.ndarray] = None
        self.t: int = 0

    def step(self, theta: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if self.m is None:
            self.m = np.zeros_like(theta)
            self.v = np.zeros_like(theta)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        return theta + self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def _softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    z = logits - np.max(logits, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# Block objectives, used to keep every update monotone
# ---------------------------------------------------------------------------


def coarse_emission_ll(
    q: PosteriorSummaries,
    X_K: np.ndarray,
    Y_K: np.ndarray,
    ctx: np.ndarray,
    params: ScaleParams,
    scored: Optional[np.ndarray] = None,
) -> float:
    """Expected coarsest-scale emission log-likelihood under q, over `scored`."""
    L = params.L
    lag_x = lag_contrib(X_K, params.coarse_count.betaLag, params.parent_sets, L)
    lag_y = lag_contrib(Y_K, params.coarse_dur.alphaLag, params.parent_sets, L)
    ll = coarse_ll_tml(X_K, Y_K, q.m_H, ctx, params, lag_x, lag_y).sum(axis=2)
    weight = np.where(rows_of(scored, X_K.shape[0])[:, None], q.gamma, 0.0)
    return float(np.einsum("tm,tm->", weight, ll))


def allocation_ll(
    q: PosteriorSummaries,
    X_k: np.ndarray,
    Y_k: np.ndarray,
    X_par: np.ndarray,
    Y_par: np.ndarray,
    ctx: np.ndarray,
    params: ScaleParams,
    block: BlockIndex,
    recorded: Optional[np.ndarray] = None,
    scored: Optional[np.ndarray] = None,
) -> float:
    """
    Expected finer-scale allocation log-likelihood under q, over `scored`.

    The block softmaxes read `recorded`, so the allocation a scored epoch is
    measured against is the one the whole recorded block competed for.  That
    is what makes the sum additive over disjoint row sets: taking a row out of
    `scored` removes its own term and moves no other.
    """
    L = params.L
    lag_x = lag_contrib(X_k, params.fine_count.betaLag, params.parent_sets, L)
    lag_y = lag_contrib(Y_k, params.fine_dur.alphaLag, params.parent_sets, L)
    eta, xi = fine_scores(X_k, X_par, Y_par, q.m_H, ctx, params, block, lag_x, lag_y)
    kept, rows = recorded_and_scored(recorded, scored, X_k.shape[0])
    seen = np.broadcast_to(kept[:, None], X_k.shape)
    summed = np.broadcast_to(rows[:, None], X_k.shape)
    active = (X_k > 0) & seen
    omega_x = block_softmax(np.einsum("tm,tml->tl", q.gamma, eta), block, mask=seen)
    omega_y = block_softmax(np.einsum("tm,tml->tl", q.gamma, xi), block, mask=active)
    x = np.where(summed, X_k, 0).astype(float)
    y = np.where(active & summed, Y_k, 0.0).astype(float)
    return float(
        dirmult_ll(x, omega_x, params.fine_count.kappa).sum()
        + dirmult_ll(y, omega_y, params.fine_dur.tau).sum()
    )


def regime_ll(
    q: PosteriorSummaries,
    ctx: np.ndarray,
    params: RegimeParams,
    parent_gamma: Optional[np.ndarray] = None,
) -> float:
    """Expected log-probability of the regime path under q."""
    T = q.gamma.shape[0]
    ctx_term, offset = regime_offsets(params, ctx, parent_gamma)
    log_p0 = params.init + ctx_term[0]
    log_p0 = log_p0 - logsumexp(log_p0)
    total = float(np.dot(q.gamma[0], log_p0))
    if T > 1:
        logits = (params.bias[:, None] + params.trans)[None, :, :] + offset[1:][
            :, :, None
        ]  # (T-1, M_next, M_prev)
        log_A = logits - logsumexp(logits, axis=1, keepdims=True)
        expected = np.transpose(q.xi[: T - 1], (0, 2, 1))  # (T-1, M_next, M_prev)
        total += float(np.sum(expected * log_A))
    return total


# ---------------------------------------------------------------------------
# 4.1  Regime transition update
# ---------------------------------------------------------------------------


def update_regime_transitions(
    q: PosteriorSummaries,
    ctx: np.ndarray,  # (T, d_ctx)
    params: RegimeParams,
    parent_gamma: Optional[np.ndarray] = None,  # (T, M_parent)
    n_steps: int = 30,
    lr: float = 5e-2,
) -> RegimeParams:
    """
    Update regime-transition parameters (init, bias, trans, cross, ctx weights)
    via gradient ascent on E_q[log p(S | ctx, parent)].

    The gradient of the softmax MLE is the usual "expected minus predicted"
    residual, with expectations supplied by γ_t(m) and ξ_t(m, m'):

        ∂/∂logit_t(m | m') = ξ_{t-1}(m', m) − γ_{t-1}(m') p_t(m | m').

    Returns a new RegimeParams (does not modify in place).
    """
    p = deepcopy(params)
    M = p.M
    T = q.gamma.shape[0]
    gamma = q.gamma  # (T, M)
    xi = q.xi  # (T-1, M, M) - xi[t, m_prev, m_next]
    d_ctx = ctx.shape[1]

    has_parent = parent_gamma is not None and p.cross is not None

    opt_bias = _Adam(lr=lr)
    opt_trans = _Adam(lr=lr)
    opt_cross = _Adam(lr=lr) if has_parent else None
    opt_ctx = _Adam(lr=lr) if d_ctx > 0 else None
    opt_init = _Adam(lr=lr)

    # expected[t, m_next, m_prev] = ξ_t(m_prev, m_next)
    expected = np.transpose(xi[: max(T - 1, 0)], (0, 2, 1)) if T > 1 else None

    for _ in range(n_steps):
        ctx_term, offset = regime_offsets(p, ctx, parent_gamma)  # (T, M)

        # ---- initial distribution: E[S_0=m] − softmax(init + b·ctx_0) ----
        p0 = _softmax(p.init + ctx_term[0])
        d0 = gamma[0] - p0  # (M,)
        grad_init = d0
        grad_ctx_w = np.outer(d0, ctx[0]) if d_ctx > 0 else None

        grad_bias = np.zeros_like(p.bias)
        grad_trans = np.zeros_like(p.trans)
        grad_cross = np.zeros_like(p.cross) if has_parent else None

        if T > 1:
            # logits[t, m_next, m_prev] for t = 1 ... T-1
            logits = (p.bias[:, None] + p.trans)[None, :, :] + offset[1:][
                :, :, None
            ]  # (T-1, M, M)
            probs = _softmax(logits, axis=1)
            pred = probs * gamma[:-1][:, None, :]  # γ_{t-1}(m') p
            diff = expected - pred  # (T-1, M, M)

            grad_bias = grad_bias + diff.sum(axis=(0, 2))
            grad_trans = grad_trans + diff.sum(axis=0)
            if d_ctx > 0:
                grad_ctx_w = grad_ctx_w + np.einsum("tmp,ti->mi", diff, ctx[1:])
            if has_parent:
                grad_cross = grad_cross + np.einsum(
                    "tmp,tn->mn", diff, parent_gamma[1:]
                )

        p.init = opt_init.step(p.init, grad_init)
        p.bias = opt_bias.step(p.bias, grad_bias)
        p.trans = opt_trans.step(p.trans, grad_trans)
        if opt_ctx is not None:
            p.ctx = opt_ctx.step(p.ctx, grad_ctx_w)
        if opt_cross is not None:
            p.cross = opt_cross.step(p.cross, grad_cross)

    return p


# ---------------------------------------------------------------------------
# 4.2  Factor (LGSSM) closed-form update
# ---------------------------------------------------------------------------


def update_factor_params(
    q: PosteriorSummaries,
    ctx: np.ndarray,  # (T, d_ctx)
    params: FactorParams,
    reg: float = 1e-4,  # regularisation for PD matrices
) -> FactorParams:
    """
    Closed-form Baum-Welch update for the regime-switching LGSSM parameters
    (F[m], G[m], Σ[m]) and the initial distribution (μ_0, Σ_0).

    Uses the expected sufficient statistics:
        γ_t(m)                  = q(S_t=m)
        E[H_t]    = m_H[t]
        E[H_t H_t^T] = Σ_H[t] + m_H[t] m_H[t]^T   =: P_t
        E[H_t H_{t-1}^T]  (approximated via smoother cross-covariance, which
                           is set to γ_t(m) P_f[t-1] F_bar^T here - rough approx)

    Returns a new FactorParams.
    """
    p = deepcopy(params)
    r = p.r
    if r == 0:
        return p

    T = q.gamma.shape[0]
    M = p.M
    d_ctx = p.d_ctx

    m_H = q.m_H  # (T, r)
    Sigma_H = q.Sigma_H  # (T, r, r)
    P_t = q.expected_hh()  # (T, r, r)  = Σ_H + m_H m_H^T

    eps = reg * np.eye(r)

    # Approximate cross-covariance E[H_t H_{t-1}^T] ≈ m_H[t] m_H[t-1]^T
    # (ignores posterior correlation; improve in future with stored P_cross)
    P_cross = np.einsum("ti,tj->tij", m_H[1:], m_H[:-1])  # (T-1, r, r)

    # ---- initial distribution (t=0 only), shared by all regimes ----------
    p.mu0 = m_H[0].copy()
    p.Sigma0 = _make_sym(Sigma_H[0] + eps)

    # Update per regime m
    for m in range(M):
        gam_m = q.gamma[:, m]  # (T,)  γ_t(m)

        # ---- F[m] and G[m] update ----------------------------------------
        # Model: H_t = F H_{t-1} + G ctx_t + ε,  ε ~ N(0, Σ)
        gam_t1_m = gam_m[1:]  # (T-1,)  γ_t(m), t≥1
        sum_gam1 = max(float(gam_t1_m.sum()), 1e-6)

        A = np.einsum("t,tij->ij", gam_t1_m, P_cross)  # (r, r)
        B = np.einsum("t,tij->ij", gam_t1_m, P_t[:-1])  # (r, r)

        if d_ctx > 0:
            C = np.einsum("t,ti,tj->ij", gam_t1_m, m_H[1:], ctx[1:])  # (r, d_ctx)
            D = np.einsum("t,ti,tj->ij", gam_t1_m, ctx[1:], ctx[1:])  # (d, d)
            B_ctx = np.einsum("t,ti,tj->ij", gam_t1_m, m_H[:-1], ctx[1:])

            lhs = np.block([[B + eps, B_ctx], [B_ctx.T, D + reg * np.eye(d_ctx)]])
            rhs = np.hstack([A, C])  # (r, r+d)
            try:
                FG = np.linalg.solve(lhs.T, rhs.T).T  # (r, r+d)
                p.F[m] = FG[:, :r]
                p.G[m] = FG[:, r:]
            except np.linalg.LinAlgError:
                pass  # keep existing values if singular
        else:
            try:
                p.F[m] = np.linalg.solve((B + eps).T, A.T).T
            except np.linalg.LinAlgError:
                pass
            p.G[m] = np.zeros((r, 0))

        # ---- Σ[m] update ------------------------------------------------
        # Σ_m = Σ_t γ_t(m) E[(H_t − F H_{t-1} − G ctx_t)(…)^T] / Σ_t γ_t(m)
        F_m, G_m = p.F[m], p.G[m]
        mu_pred = m_H[:-1] @ F_m.T
        if d_ctx > 0:
            mu_pred = mu_pred + ctx[1:] @ G_m.T
        resid = m_H[1:] - mu_pred  # (T-1, r)
        S_hat = np.einsum(
            "t,tij->ij",
            gam_t1_m,
            Sigma_H[1:] + np.einsum("ti,tj->tij", resid, resid),
        )
        p.Sigma[m] = _make_sym(S_hat / sum_gam1) + eps

    return p


# ---------------------------------------------------------------------------
# 4.3  ZINB count regression
# ---------------------------------------------------------------------------


def update_coarse_count_params(
    q: PosteriorSummaries,
    X_K: np.ndarray,  # (T, L)
    ctx: np.ndarray,  # (T, d_ctx)
    params: ScaleParams,
    n_steps: int = 30,
    lr: float = 5e-2,
    scored: Optional[np.ndarray] = None,
) -> CoarseCountParams:
    """
    Update ZINB emission parameters (β, γ, r_disp) via Adam gradient ascent
    on E_q[Σ_t log p(X_t^(K) | S_t, H_t)].

    Every expectation below is weighted by γ, so the scorable-row indicator
    enters as a factor on it and an epoch outside it moves no parameter.

    Returns a new CoarseCountParams.
    """
    cp = deepcopy(params.coarse_count)
    T, L = X_K.shape
    M = params.M
    r = params.r
    gamma = np.where(rows_of(scored, T)[:, None], q.gamma, 0.0)  # (T, M)
    h_bar = q.m_H  # (T, r)
    d_ctx = ctx.shape[1]

    # Lagged design matrices (fixed across steps; only the coefficients move)
    feats = lag_features(X_K, list(cp.betaLag), params.parent_sets, L)

    opt_b0 = _Adam(lr=lr)
    opt_bS = _Adam(lr=lr)
    opt_bH = _Adam(lr=lr)
    opt_bCtx = _Adam(lr=lr)
    opt_g0 = _Adam(lr=lr)
    opt_gS = _Adam(lr=lr)
    opt_gH = _Adam(lr=lr)
    opt_gCtx = _Adam(lr=lr)
    opt_r = _Adam(lr=lr * 0.1)  # dispersion needs smaller LR
    opt_bLag: Dict[Tuple[int, int], _Adam] = {key: _Adam(lr=lr) for key in cp.betaLag}

    x = X_K[:, None, :].astype(float)  # (T, 1, L)
    zero_mask = x == 0

    for _ in range(n_steps):
        lag_x = np.zeros((T, L))
        for key, coef in cp.betaLag.items():
            lag_x += feats[key] * coef

        base_mu = cp.beta0 + lag_x
        base_pi = np.broadcast_to(cp.gamma0, (T, L)).copy()
        if r > 0:
            base_mu = base_mu + h_bar @ cp.betaH
            base_pi = base_pi + h_bar @ cp.gammaH
        if d_ctx > 0 and cp.betaCtx.shape[0] > 0:
            base_mu = base_mu + ctx @ cp.betaCtx
        if d_ctx > 0 and cp.gammaCtx.shape[0] > 0:
            base_pi = base_pi + ctx @ cp.gammaCtx

        log_mu = base_mu[:, None, :] + cp.betaS[None, :, :]  # (T, M, L)
        logit_pi = base_pi[:, None, :] + cp.gammaS[None, :, :]  # (T, M, L)

        mu = np.exp(np.clip(log_mu, -_LOG_CLIP, _LOG_CLIP))
        pi = _sigmoid(logit_pi)
        rv = np.maximum(cp.r_disp, 1e-4)  # (L,)
        nb_p = rv / (rv + mu)

        # --- NB gradient w.r.t. log μ:  x − (r+x) μ/(r+μ) ---
        dnb_dlogmu = x - (rv + x) * (mu / (rv + mu))

        nb0 = rv * np.log(nb_p + 1e-15)  # log NB(0; r, μ)
        log_p0 = np.logaddexp(np.log(pi + 1e-15), np.log1p(-pi + 1e-15) + nb0)
        dnb0_dlogmu = -rv * mu / (rv + mu)
        zero_share = np.exp(np.log1p(-pi + 1e-15) + nb0 - log_p0)

        grad_logmu = np.where(zero_mask, zero_share * dnb0_dlogmu, dnb_dlogmu)

        pi_deriv = pi * (1.0 - pi)
        grad_logitpi = np.where(
            zero_mask,
            pi_deriv * (1.0 - np.exp(nb0 - log_p0)) / (pi + 1e-15),
            -pi_deriv / (1.0 - pi + 1e-15),
        )

        grad_r_pos = (
            digamma(x + rv) - digamma(rv) + np.log(nb_p + 1e-15) - x / (rv + mu)
        )
        grad_r = np.where(
            zero_mask,
            zero_share * (np.log(nb_p + 1e-15) - rv / (rv + mu)),
            grad_r_pos,
        )

        # --- accumulate, weighting each (t, m) by γ_t(m) ---
        Gmu = np.einsum("tm,tml->tl", gamma, grad_logmu)
        Gpi = np.einsum("tm,tml->tl", gamma, grad_logitpi)

        gb0 = Gmu.sum(axis=0)
        gbS = np.einsum("tm,tml->ml", gamma, grad_logmu)
        gg0 = Gpi.sum(axis=0)
        ggS = np.einsum("tm,tml->ml", gamma, grad_logitpi)
        gr = np.einsum("tm,tml->l", gamma, grad_r)

        cp.beta0 = opt_b0.step(cp.beta0, gb0)
        cp.betaS = opt_bS.step(cp.betaS, gbS)
        if r > 0:
            cp.betaH = opt_bH.step(cp.betaH, h_bar.T @ Gmu)
            cp.gammaH = opt_gH.step(cp.gammaH, h_bar.T @ Gpi)
        if cp.betaCtx.shape[0] > 0:
            cp.betaCtx = opt_bCtx.step(cp.betaCtx, ctx.T @ Gmu)
        cp.gamma0 = opt_g0.step(cp.gamma0, gg0)
        cp.gammaS = opt_gS.step(cp.gammaS, ggS)
        if cp.gammaCtx.shape[0] > 0:
            cp.gammaCtx = opt_gCtx.step(cp.gammaCtx, ctx.T @ Gpi)
        cp.r_disp = np.maximum(opt_r.step(cp.r_disp, gr), 0.05)
        for key in cp.betaLag:
            cp.betaLag[key] = opt_bLag[key].step(
                cp.betaLag[key], (Gmu * feats[key]).sum(axis=0)
            )

    return cp


# ---------------------------------------------------------------------------
# 4.4  NB duration regression
# ---------------------------------------------------------------------------


def update_coarse_dur_params(
    q: PosteriorSummaries,
    X_K: np.ndarray,  # (T, L)
    Y_K: np.ndarray,  # (T, L)
    ctx: np.ndarray,  # (T, d_ctx)
    params: ScaleParams,
    n_steps: int = 30,
    lr: float = 5e-2,
    scored: Optional[np.ndarray] = None,
) -> CoarseDurParams:
    """
    Update NB duration parameters (α, s_disp) via Adam.
    Only applied to scorable epochs where X_t > 0.
    """
    dp = deepcopy(params.coarse_dur)
    T, L = Y_K.shape
    M = params.M
    r = params.r
    gamma = np.where(rows_of(scored, T)[:, None], q.gamma, 0.0)
    h_bar = q.m_H
    d_ctx = ctx.shape[1]

    feats = lag_features(Y_K, list(dp.alphaLag), params.parent_sets, L)

    opt_a0 = _Adam(lr=lr)
    opt_aS = _Adam(lr=lr)
    opt_aH = _Adam(lr=lr)
    opt_aCtx = _Adam(lr=lr)
    opt_aX = _Adam(lr=lr)
    opt_s = _Adam(lr=lr * 0.1)
    opt_aLag: Dict[Tuple[int, int], _Adam] = {key: _Adam(lr=lr) for key in dp.alphaLag}

    active = (X_K > 0)[:, None, :]  # (T, 1, L)
    y = Y_K[:, None, :].astype(float)
    log1p_X = np.log1p(X_K.astype(float))  # (T, L)

    for _ in range(n_steps):
        lag_y = np.zeros((T, L))
        for key, coef in dp.alphaLag.items():
            lag_y += feats[key] * coef

        base_nu = dp.alpha0 + lag_y + dp.alphaX * log1p_X
        if r > 0:
            base_nu = base_nu + h_bar @ dp.alphaH
        if d_ctx > 0 and dp.alphaCtx.shape[0] > 0:
            base_nu = base_nu + ctx @ dp.alphaCtx

        log_nu = base_nu[:, None, :] + dp.alphaS[None, :, :]  # (T, M, L)
        nu = np.maximum(np.exp(np.clip(log_nu, -_LOG_CLIP, _LOG_CLIP)), 1.0)
        s_ell = np.maximum(dp.s_disp, 0.05)  # (L,)
        nb_p = s_ell / (s_ell + nu)

        # ∂ log NB / ∂ log ν  and  ∂ log NB / ∂ s
        grad_lognu = np.where(active, y - (s_ell + y) * nu / (s_ell + nu), 0.0)
        grad_s = np.where(
            active,
            digamma(y + s_ell)
            - digamma(s_ell)
            + np.log(nb_p + 1e-15)
            - y / (s_ell + nu),
            0.0,
        )

        Gnu = np.einsum("tm,tml->tl", gamma, grad_lognu)  # (T, L)

        dp.alpha0 = opt_a0.step(dp.alpha0, Gnu.sum(axis=0))
        dp.alphaS = opt_aS.step(dp.alphaS, np.einsum("tm,tml->ml", gamma, grad_lognu))
        if r > 0:
            dp.alphaH = opt_aH.step(dp.alphaH, h_bar.T @ Gnu)
        if dp.alphaCtx.shape[0] > 0:
            dp.alphaCtx = opt_aCtx.step(dp.alphaCtx, ctx.T @ Gnu)
        dp.alphaX = opt_aX.step(dp.alphaX, (Gnu * log1p_X).sum(axis=0))
        dp.s_disp = np.maximum(
            opt_s.step(dp.s_disp, np.einsum("tm,tml->l", gamma, grad_s)), 0.05
        )
        for key in dp.alphaLag:
            dp.alphaLag[key] = opt_aLag[key].step(
                dp.alphaLag[key], (Gnu * feats[key]).sum(axis=0)
            )

    return dp


# ---------------------------------------------------------------------------
# 4.5  Allocation score update (finer scales)
# ---------------------------------------------------------------------------


def update_allocation_scores(
    q: PosteriorSummaries,
    X_k: np.ndarray,  # (H_k, L)  fine scale
    Y_k: np.ndarray,  # (H_k, L)
    X_par: np.ndarray,  # (H_{k+1}, L)
    Y_par: np.ndarray,  # (H_{k+1}, L)
    ctx: np.ndarray,  # (H_k, d_ctx)
    params: ScaleParams,  # finer scale
    hier: ScaleHierarchy,
    k: int,
    n_steps: int = 20,
    lr: float = 5e-2,
    fit_conc: bool = True,
    recorded: Optional[np.ndarray] = None,
    scored: Optional[np.ndarray] = None,
) -> Tuple[FineCountParams, FineDurParams]:
    """
    Update Dir-Mult allocation score parameters (η-weights, ξ-weights)
    via Adam on the expected allocation log-likelihood

        Σ_c Σ_{t∈ch(c)} Σ_m γ_t(m) Σ_ℓ X_{t,ℓ} [η_{t,ℓ,m} − logZ_{c,ℓ}],

    whose gradient w.r.t. η is the multinomial residual γ_t(m)(X_t − N_c ω_t).

    The concentrations κ and τ are estimated alongside the scores when
    `fit_conc` is set.  They control how much the realised allocation may
    depart from its expectation, so leaving them at an assumed value makes every
    structure score that uses this likelihood over- or under-confident.

    An epoch outside `recorded` leaves the block competition and carries a zero
    count.  An epoch that is recorded and outside `scored` stays in the
    competition, since its score still moves what its siblings may claim, and
    carries a zero count, which drops its own term from the objective and
    leaves the gradient of that term at zero.

    Returns (new FineCountParams, new FineDurParams).
    """
    fp = deepcopy(params.fine_count)
    fdp = deepcopy(params.fine_dur)
    T, L = X_k.shape
    M = params.M
    r = params.r
    gamma = q.gamma  # (T, M)
    h_bar = q.m_H  # (T, r)
    d_ctx = ctx.shape[1]

    block = BlockIndex.from_hierarchy(hier, k)
    feats_x = lag_features(X_k, list(fp.betaLag), params.parent_sets, L)
    feats_y = lag_features(Y_k, list(fdp.alphaLag), params.parent_sets, L)

    opt_fb0 = _Adam(lr=lr)
    opt_fbS = _Adam(lr=lr)
    opt_fbH = _Adam(lr=lr)
    opt_fbCtx = _Adam(lr=lr)
    opt_fbP = _Adam(lr=lr)
    opt_fbD = _Adam(lr=lr)
    opt_fbDep = _Adam(lr=lr)
    opt_fbLag = {key: _Adam(lr=lr) for key in fp.betaLag}

    opt_fa0 = _Adam(lr=lr)
    opt_faS = _Adam(lr=lr)
    opt_faH = _Adam(lr=lr)
    opt_faCtx = _Adam(lr=lr)
    opt_faX = _Adam(lr=lr)
    opt_faP = _Adam(lr=lr)
    opt_faDep = _Adam(lr=lr)
    opt_faLag = {key: _Adam(lr=lr) for key in fdp.alphaLag}

    x = X_k.astype(float)
    y = Y_k.astype(float)
    kept, rows = recorded_and_scored(recorded, scored, T)
    seen = np.broadcast_to(kept[:, None], X_k.shape)
    summed = np.broadcast_to(rows[:, None], X_k.shape)
    active = (X_k > 0) & seen
    # The counts the Dirichlet-Multinomial term scores.  `log1p_X` and `dep_x`
    # below keep reading the whole `x`, since both are features of the record
    # and the forward dilation is what removes a row whose feature reads a gap.
    x_scored = np.where(summed, x, 0.0)
    y_scored = np.where(active & summed, y, 0.0)
    N_x = block.broadcast(X_par).astype(float)  # (T, L) parent totals
    N_y = block.broadcast(Y_par).astype(float)
    log_x_par = np.log1p(N_x)
    log_y_par = np.log1p(N_y)
    log1p_X = np.log1p(x)
    dep_x = np.log1p(block.exclusive_cumsum(x))  # (T, L) earlier siblings

    # View of the scale params that tracks the in-progress fp / fdp objects.
    params_now = copy(params)
    params_now.fine_count = fp
    params_now.fine_dur = fdp

    for _ in range(n_steps):
        lag_x = lag_contrib(X_k, fp.betaLag, params.parent_sets, L, feats=feats_x)
        lag_y = lag_contrib(Y_k, fdp.alphaLag, params.parent_sets, L, feats=feats_y)

        eta, xi = fine_scores(
            X_k, X_par, Y_par, h_bar, ctx, params_now, block, lag_x, lag_y
        )  # (T, M, L) each

        # Dir-Mult gradient at the regime-averaged score η̄ = Σ_m γ_m η_m.
        # Because the likelihood depends on the parameters only through η̄,
        # ∂/∂(regime-m block) picks up a factor γ_t(m).
        omega_x = block_softmax(np.einsum("tm,tml->tl", gamma, eta), block, mask=seen)
        omega_y = block_softmax(np.einsum("tm,tml->tl", gamma, xi), block, mask=active)
        Ge = dirmult_grad(x_scored, omega_x, fp.kappa, block)  # (T, L)
        Gx = dirmult_grad(y_scored, omega_y, fdp.tau, block)  # (T, L)
        grad_eta = gamma[:, :, None] * Ge[:, None, :]  # (T, M, L)
        grad_xi = gamma[:, :, None] * Gx[:, None, :]

        fp.beta0 = opt_fb0.step(fp.beta0, Ge.sum(axis=0))
        fp.betaS = opt_fbS.step(fp.betaS, grad_eta.sum(axis=0))
        if r > 0:
            fp.betaH = opt_fbH.step(fp.betaH, h_bar.T @ Ge)
        if fp.betaCtx.shape[0] > 0:
            fp.betaCtx = opt_fbCtx.step(fp.betaCtx, ctx.T @ Ge)
        fp.betaP = opt_fbP.step(fp.betaP, (Ge * log_x_par).sum(axis=0))
        fp.betaD = opt_fbD.step(fp.betaD, (Ge * log_y_par).sum(axis=0))
        fp.betaDep = opt_fbDep.step(fp.betaDep, (Ge * dep_x).sum(axis=0))
        for key in fp.betaLag:
            fp.betaLag[key] = opt_fbLag[key].step(
                fp.betaLag[key], (Ge * feats_x[key]).sum(axis=0)
            )

        fdp.alpha0 = opt_fa0.step(fdp.alpha0, Gx.sum(axis=0))
        fdp.alphaS = opt_faS.step(fdp.alphaS, grad_xi.sum(axis=0))
        if r > 0:
            fdp.alphaH = opt_faH.step(fdp.alphaH, h_bar.T @ Gx)
        if fdp.alphaCtx.shape[0] > 0:
            fdp.alphaCtx = opt_faCtx.step(fdp.alphaCtx, ctx.T @ Gx)
        fdp.alphaX = opt_faX.step(fdp.alphaX, (Gx * log1p_X).sum(axis=0))
        fdp.alphaP = opt_faP.step(fdp.alphaP, (Gx * log_y_par).sum(axis=0))
        fdp.alphaDep = opt_faDep.step(fdp.alphaDep, (Gx * dep_x).sum(axis=0))
        for key in fdp.alphaLag:
            fdp.alphaLag[key] = opt_faLag[key].step(
                fdp.alphaLag[key], (Gx * feats_y[key]).sum(axis=0)
            )

    if fit_conc:
        lag_x = lag_contrib(X_k, fp.betaLag, params.parent_sets, L, feats=feats_x)
        lag_y = lag_contrib(Y_k, fdp.alphaLag, params.parent_sets, L, feats=feats_y)
        eta, xi = fine_scores(
            X_k, X_par, Y_par, h_bar, ctx, params_now, block, lag_x, lag_y
        )
        omega_x = block_softmax(np.einsum("tm,tml->tl", gamma, eta), block, mask=seen)
        omega_y = block_softmax(np.einsum("tm,tml->tl", gamma, xi), block, mask=active)
        fp.kappa = fit_concentration(x_scored, omega_x, block)
        fdp.tau = fit_concentration(y_scored, omega_y, block)

    return fp, fdp


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _make_sym(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)


# ---------------------------------------------------------------------------
# 4.6  MStep: dispatches all sub-blocks
# ---------------------------------------------------------------------------


class MStep:
    """
    M-step coordinator: update all Θ_k given q^new.

    A block update is applied only when it does not lower that block's own
    expected log-likelihood, since a fixed number of Adam steps can overshoot.
    Without the check the monitored objective of the outer loop can decrease,
    which makes a converged fit hard to recognise.

    Usage::
        mstep = MStep()
        new_params = mstep.run(k, q, params, data, hier)
    """

    def __init__(
        self,
        n_steps_regime: int = 30,
        n_steps_emission: int = 30,
        lr: float = 5e-2,
        monotone: bool = True,
        fit_concentration: bool = True,
        regime_update: str = "gradient",
        regime_estimator: str = "soft-bayes",
        regime_ess: float = 5.0,
    ) -> None:
        if regime_update not in ("gradient", "table", "table+gradient"):
            raise ValueError(
                f"unknown regime_update {regime_update!r}, expected gradient, "
                f"table or table+gradient"
            )
        self.n_steps_regime = n_steps_regime
        self.n_steps_emission = n_steps_emission
        self.lr = lr
        self.monotone = monotone
        self.fit_concentration = fit_concentration
        self.regime_update = regime_update
        self.regime_estimator = regime_estimator
        self.regime_ess = regime_ess

    # ------------------------------------------------------------------
    def _regime_candidates(
        self,
        q: PosteriorSummaries,
        ctx: np.ndarray,
        params: RegimeParams,
        parent_gamma: Optional[np.ndarray],
    ) -> List[RegimeParams]:
        """Every regime proposal this configuration produces, best last."""
        from .param_estimation import update_regime_table

        out: List[RegimeParams] = []
        table = None
        if self.regime_update in ("table", "table+gradient"):
            table = update_regime_table(
                q,
                params,
                method=self.regime_estimator,
                equivalent_sample_size=self.regime_ess,
            )
            out.append(table)
        if self.regime_update in ("gradient", "table+gradient"):
            out.append(
                update_regime_transitions(
                    q,
                    ctx,
                    params,
                    parent_gamma=parent_gamma,
                    n_steps=self.n_steps_regime,
                    lr=self.lr,
                )
            )
        if table is not None and self.regime_update == "table+gradient":
            out.append(
                update_regime_transitions(
                    q,
                    ctx,
                    table,
                    parent_gamma=parent_gamma,
                    n_steps=self.n_steps_regime,
                    lr=self.lr,
                )
            )
        return out

    def _budget(self, k: int) -> Optional[int]:
        """The declared lag budget at scale k, or None when none was set."""
        budget = getattr(self, "lag_budget", None)
        return None if budget is None else int(budget[k - 1])

    def run(
        self,
        k: int,
        q: PosteriorSummaries,
        params: ScaleParams,
        data: MultiScaleData,
        hier: Optional[ScaleHierarchy] = None,
        parent_summaries: Optional[PosteriorSummaries] = None,
    ) -> ScaleParams:
        """
        Update all parameters at scale k.

        Parameters
        ----------
        k                : 1-indexed scale
        q                : PosteriorSummaries from the E-step at this scale
        params           : current ScaleParams for scale k
        data             : full multiscale data
        hier             : ScaleHierarchy (optional, defaults to data.hierarchy)
        parent_summaries : PosteriorSummaries from scale k+1 (for cross-scale coupling)

        Returns
        -------
        New ScaleParams with updated parameters.
        """
        if hier is None:
            hier = data.hierarchy

        ctx_k = data.get_C(k)
        new_p = deepcopy(params)
        # The same row set the E-step scored, so the guard below compares two
        # values of one objective.  The uneroded mask travels with it, since a
        # finer-scale block sum is over what the record holds.
        recorded = data.get_observed(k)
        rows = declared_rows(recorded, self._budget(k), "the M-step")

        # ---- 4.1  Regime transitions ------------------------------------
        parent_gamma = None
        if parent_summaries is not None and k < data.K:
            parent_gamma = parent_summaries.gamma[hier.parent_maps[k - 1]]
        candidates = self._regime_candidates(q, ctx_k, params.regime, parent_gamma)
        candidate = max(candidates, key=lambda c: regime_ll(q, ctx_k, c, parent_gamma))
        if not self.monotone or regime_ll(
            q, ctx_k, candidate, parent_gamma
        ) >= regime_ll(q, ctx_k, params.regime, parent_gamma):
            new_p.regime = candidate

        # ---- 4.2  Latent factor ----------------------------------------
        if params.r > 0:
            new_p.factor = update_factor_params(q, ctx_k, params.factor)

        # ---- 4.3 / 4.4  Coarsest-scale emissions -----------------------
        if params.is_coarsest:
            X_K = data.get_X(k)
            Y_K = data.get_Y(k)
            before = (
                coarse_emission_ll(q, X_K, Y_K, ctx_k, params, scored=rows)
                if self.monotone
                else 0.0
            )
            trial = deepcopy(params)
            trial.coarse_count = update_coarse_count_params(
                q,
                X_K,
                ctx_k,
                params,
                n_steps=self.n_steps_emission,
                lr=self.lr,
                scored=rows,
            )
            trial.coarse_dur = update_coarse_dur_params(
                q,
                X_K,
                Y_K,
                ctx_k,
                params,
                n_steps=self.n_steps_emission,
                lr=self.lr,
                scored=rows,
            )
            if (
                not self.monotone
                or coarse_emission_ll(q, X_K, Y_K, ctx_k, trial, scored=rows) >= before
            ):
                new_p.coarse_count = trial.coarse_count
                new_p.coarse_dur = trial.coarse_dur

        # ---- 4.5  Allocation scores (finer scales) ----------------------
        else:
            X_k = data.get_X(k)
            Y_k = data.get_Y(k)
            X_par = data.get_X(k + 1)
            Y_par = data.get_Y(k + 1)
            block = BlockIndex.from_hierarchy(hier, k)
            before = (
                allocation_ll(
                    q,
                    X_k,
                    Y_k,
                    X_par,
                    Y_par,
                    ctx_k,
                    params,
                    block,
                    recorded=recorded,
                    scored=rows,
                )
                if self.monotone
                else 0.0
            )
            trial = deepcopy(params)
            trial.fine_count, trial.fine_dur = update_allocation_scores(
                q,
                X_k,
                Y_k,
                X_par,
                Y_par,
                ctx_k,
                params,
                hier,
                k,
                n_steps=self.n_steps_emission,
                lr=self.lr,
                fit_conc=self.fit_concentration,
                recorded=recorded,
                scored=rows,
            )
            if (
                not self.monotone
                or allocation_ll(
                    q,
                    X_k,
                    Y_k,
                    X_par,
                    Y_par,
                    ctx_k,
                    trial,
                    block,
                    recorded=recorded,
                    scored=rows,
                )
                >= before
            ):
                new_p.fine_count = trial.fine_count
                new_p.fine_dur = trial.fine_dur

        return new_p
