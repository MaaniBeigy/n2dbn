"""
model/forecast.py
=================
The h-step predictive mean, walked down the scale ladder from the origin.

How a forecast is produced
--------------------------
The fit runs on history up to the origin alone, so the posterior at its last
epoch conditions on nothing after the origin and no separate filter is needed.
From that epoch the regime chain and the latent factor are rolled forward under
their own transition parameters, with the context of the future epochs supplied
by the calendar and therefore known in advance.  The coarsest scale evaluates
its own emission mean at the rolled latents, and every finer scale spreads its
parent's forecast over the children of each block by the allocation weights of
that block.

Why the result is coherent without a repair step
------------------------------------------------
The weights of one block are a softmax over the children of that block, so they
sum to one.  Summing the children of a block therefore returns the parent's
forecast exactly, for every block, every label and every scale of the chain.
The coherence column of a results table reads that as a zero, and the zero is a
property of the construction and not of the estimate.

What the mean path leaves out
-----------------------------
The allocation scores omit the parent-count term and the depletion term.  The
parent-count term is constant across the children of a block, so the softmax
cancels it exactly.  The depletion term reads the counts earlier siblings
already took, which a mean forecast has not drawn, so what this module reports
is the marginal share.

A drawn path was written and removed.  It ran the generative model forward so
that a lagged term read a realized count, and it measured worse than the mean
path on published traffic, at 3.93 against 0.78 in mean absolute error, because
each epoch's lag reads the drawn previous epoch and the noise of a draw compounds
through the log of a count.  It also built on the frozen allocation formulation,
which nothing new may depend on.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from ..core.scales import ScaleHierarchy
from ..evaluation.coherence import _MAX_ETA
from .params import FactorParams, RegimeParams, ScaleParams


def transition_matrix(
    regime: RegimeParams, ctx_t: np.ndarray, parent_gamma: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    One epoch's regime transition matrix, column `prev` giving `P(S_t | prev)`.

    Args:
        regime: the chain's parameters.
        ctx_t: `(d_ctx,)` context of the epoch being entered.
        parent_gamma: `(parent_M,)` posterior over the parent regime, mixed into
            the cross-scale block; absent at the coarsest scale.

    Returns:
        An `(M, M)` column-stochastic array.
    """
    logits = regime.bias[:, None] + regime.trans
    if parent_gamma is not None and regime.cross is not None:
        logits = logits + (regime.cross @ np.asarray(parent_gamma, float))[:, None]
    if regime.ctx.shape[1] > 0 and np.size(ctx_t) > 0:
        logits = logits + (regime.ctx @ np.asarray(ctx_t, float))[:, None]
    logits = np.clip(logits, -_MAX_ETA, _MAX_ETA)
    weights = np.exp(logits - logits.max(axis=0, keepdims=True))
    return weights / np.maximum(weights.sum(axis=0, keepdims=True), 1e-300)


def roll_regime(
    regime: RegimeParams,
    origin: np.ndarray,
    ctx_future: np.ndarray,
    parent_gamma: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    The regime posterior carried forward over the forecast epochs.

    Args:
        regime: the chain's parameters.
        origin: `(M,)` posterior at the last epoch before the origin.
        ctx_future: `(h, d_ctx)` context of the forecast epochs.
        parent_gamma: `(h, parent_M)` parent posterior over the same epochs.

    Returns:
        An `(h, M)` array, one row per forecast epoch.
    """
    h = ctx_future.shape[0]
    out = np.zeros((h, regime.M))
    current = np.asarray(origin, dtype=float)
    for j in range(h):
        parent = None if parent_gamma is None else parent_gamma[j]
        current = transition_matrix(regime, ctx_future[j], parent) @ current
        out[j] = current
    return out


def roll_factor(
    factor: FactorParams,
    origin: np.ndarray,
    gamma_future: np.ndarray,
    ctx_future: np.ndarray,
) -> np.ndarray:
    """
    The latent factor mean carried forward, mixed over the regime posterior.

    Args:
        factor: the state-space parameters.
        origin: `(r,)` posterior mean at the last epoch before the origin.
        gamma_future: `(h, M)` regime posterior over the forecast epochs.
        ctx_future: `(h, d_ctx)` context of the same epochs.

    Returns:
        An `(h, r)` array, empty in its second axis when the scale carries no
        factor.
    """
    h = gamma_future.shape[0]
    if factor.r == 0:
        return np.zeros((h, 0))
    out = np.zeros((h, factor.r))
    current = np.asarray(origin, dtype=float).ravel()
    for j in range(h):
        step = np.zeros(factor.r)
        for s in range(factor.M):
            moved = factor.F[s] @ current
            if factor.G[s].shape[1] > 0 and np.size(ctx_future[j]) > 0:
                moved = moved + factor.G[s] @ np.asarray(ctx_future[j], float)
            step = step + gamma_future[j, s] * moved
        current = step
        out[j] = current
    return out


def coarse_mixture_mean(
    params: ScaleParams,
    gamma: np.ndarray,
    factor: np.ndarray,
    ctx: np.ndarray,
    lag: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    The zero-inflated negative binomial mean averaged over the regime posterior.

    The count is drawn under one regime at a time, so its expectation under a
    posterior over regimes is the average of the per-regime means.  Averaging
    the linear predictor first and exponentiating once is a different quantity,
    and the two diverge by the exponential of the spread of the regime block,
    which the fits on this data drive past twenty in log space.

    Args:
        params: a scale whose emission is the coarsest-style pair.
        gamma: `(h, M)` regime posterior.
        factor: `(h, r)` factor mean, empty when the scale carries none.
        ctx: `(h, d)` context.
        lag: `(h, L)` lagged block, added inside the exponential.  The linear
            predictor is one sum, so it is bounded once here; bounding the lag
            separately and multiplying the two exponentials lets their product
            reach the square of the bound, which is what a fitted lag
            coefficient of three and a half on a log count drives it to.

    Returns:
        An `(h, L)` array of expected counts.

    Raises:
        ValueError: the scale carries an allocation and has no mean of its own.
    """
    if not params.is_coarsest:
        raise ValueError(
            f"scale {params.k} carries an allocation and has no mean of its own"
        )
    count = params.coarse_count
    h = gamma.shape[0]
    base_mu = count.beta0[None, :] + np.zeros((h, count.L))
    base_pi = count.gamma0[None, :] + np.zeros((h, count.L))
    if count.betaH.size and factor.shape[1]:
        base_mu = base_mu + factor @ count.betaH
        base_pi = base_pi + factor @ count.gammaH
    if count.betaCtx.size and ctx.shape[1]:
        base_mu = base_mu + ctx @ count.betaCtx
        base_pi = base_pi + ctx @ count.gammaCtx
    if lag is not None:
        base_mu = base_mu + lag

    out = np.zeros((h, count.L))
    for s in range(gamma.shape[1]):
        eta = base_mu + (count.betaS[s][None, :] if count.betaS.size else 0.0)
        logit = base_pi + (count.gammaS[s][None, :] if count.gammaS.size else 0.0)
        mu = np.exp(np.clip(eta, -_MAX_ETA, _MAX_ETA))
        pi = 1.0 / (1.0 + np.exp(-np.clip(logit, -_MAX_ETA, _MAX_ETA)))
        out = out + gamma[:, s : s + 1] * (1.0 - pi) * mu
    return out


def lag_offset(
    params: ScaleParams, history: np.ndarray, step: int, pending: np.ndarray
) -> np.ndarray:
    """
    The lagged block of the coarsest predictor at one forecast epoch.

    Args:
        params: the coarsest scale's parameters.
        history: `(T, L)` observed counts up to the origin.
        step: which forecast epoch this is, counting from zero.
        pending: `(step, L)` forecasts already produced, read where a lag
            reaches past the end of the observed history.

    Returns:
        An `(L,)` array to add to the log mean.
    """
    count = params.coarse_count
    out = np.zeros(count.L)
    for (src, lag), coef in count.betaLag.items():
        index = step - lag
        if index >= 0:
            value = pending[index, src]
        elif history.shape[0] + index >= 0:
            value = float(history[history.shape[0] + index, src])
        else:
            continue
        out = out + coef * np.log1p(max(value, 0.0))
    return out


def lag_block(
    beta_lag, history: np.ndarray, pending: Optional[np.ndarray], shape
) -> np.ndarray:
    """
    The lagged contribution at every epoch of a forecast window.

    Args:
        beta_lag: the `{(source, lag): (L,)}` coefficients of one scale.
        history: `(T, L)` observed counts up to the origin.
        pending: `(h, L)` counts already forecast, or `None` before any are.
        shape: the `(h, L)` shape to return.

    Returns:
        An `(h, L)` array to add to the linear predictor.
    """
    h, L = shape
    out = np.zeros((h, L))
    known = np.asarray(history, dtype=float)
    ahead = None if pending is None else np.asarray(pending, dtype=float)
    for (src, lag), coef in beta_lag.items():
        values = np.zeros(h)
        for j in range(h):
            index = j - lag
            if index >= 0:
                values[j] = 0.0 if ahead is None else ahead[index, src]
            elif known.shape[0] + index >= 0:
                values[j] = known[known.shape[0] + index, src]
        out = out + np.log1p(np.maximum(values, 0.0))[:, None] * coef[None, :]
    return out


def forecast_coarse(
    params: ScaleParams,
    gamma_future: np.ndarray,
    factor_future: np.ndarray,
    ctx_future: np.ndarray,
    history: np.ndarray,
) -> np.ndarray:
    """
    The coarsest scale's predicted counts over the forecast epochs.

    Args:
        params: the coarsest scale's parameters.
        gamma_future: `(h, M)` rolled regime posterior.
        factor_future: `(h, r)` rolled factor mean.
        ctx_future: `(h, d_ctx)` context of the forecast epochs.
        history: `(T, L)` observed counts up to the origin, read by the lagged
            block.

    Returns:
        An `(h, L)` array of expected counts.
    """
    h = gamma_future.shape[0]
    out = np.zeros((h, params.coarse_count.L))
    for j in range(h):
        offset = lag_offset(params, history, j, out[:j])
        row = coarse_mixture_mean(
            params,
            gamma_future[j : j + 1],
            factor_future[j : j + 1],
            ctx_future[j : j + 1],
            offset[None, :],
        )
        out[j] = row[0]
    return out


def allocation_weights(
    params: ScaleParams,
    gamma_future: np.ndarray,
    factor_future: np.ndarray,
    ctx_future: np.ndarray,
    blocks: np.ndarray,
    history: Optional[np.ndarray] = None,
    pending: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    The share each forecast epoch takes of its block, summing to one per block.

    The lagged block is read the way the coarsest scale reads it: from the
    observed record where the lag reaches back into it, and from the forecast
    already produced where it does not.  Leaving it out would drop the
    coefficients the structure search fitted at this scale, which is where the
    dependence on recent epochs lives.

    Args:
        params: a finer scale's parameters.
        gamma_future: `(h, M)` rolled regime posterior.
        factor_future: `(h, r)` rolled factor mean.
        ctx_future: `(h, d_ctx)` context of the forecast epochs.
        blocks: `(h,)` parent epoch of every forecast epoch.
        history: `(T, L)` observed counts at this scale up to the origin;
            absent leaves the lagged block out.
        pending: `(h, L)` counts already forecast at this scale, read where a
            lag falls inside the forecast window.

    Returns:
        An `(h, L)` array of weights.
    """
    count = params.fine_count
    eta = count.beta0[None, :] + np.zeros((gamma_future.shape[0], count.L))
    if count.betaS.size:
        eta = eta + gamma_future @ count.betaS
    if count.betaH.size and factor_future.shape[1]:
        eta = eta + factor_future @ count.betaH
    if count.betaCtx.size and ctx_future.shape[1]:
        eta = eta + ctx_future @ count.betaCtx
    if history is not None and count.betaLag:
        eta = eta + lag_block(count.betaLag, history, pending, eta.shape)
    eta = np.clip(eta, -_MAX_ETA, _MAX_ETA)

    weights = np.zeros_like(eta)
    for parent in np.unique(blocks):
        rows = np.flatnonzero(blocks == parent)
        block = np.exp(eta[rows] - eta[rows].max(axis=0, keepdims=True))
        weights[rows] = block / np.maximum(block.sum(axis=0, keepdims=True), 1e-300)
    return weights


def forecast_ladder(
    params_list: Sequence[ScaleParams],
    origins_gamma: Sequence[np.ndarray],
    origins_factor: Sequence[np.ndarray],
    ctx_future: Sequence[np.ndarray],
    hierarchy: ScaleHierarchy,
    history_coarsest: np.ndarray,
    history_fine: Optional[Sequence[np.ndarray]] = None,
) -> List[np.ndarray]:
    """
    The predicted counts of every scale over one forecast window, finest first.

    Args:
        params_list: one entry per scale, finest first.
        origins_gamma: `(M_k,)` regime posterior at each scale's last observed
            epoch, finest first.
        origins_factor: `(r_k,)` factor mean at the same epochs, finest first.
        ctx_future: one `(H_k, d_k)` context array per scale of the forecast
            window, finest first.
        hierarchy: the forecast window's own chain.
        history_coarsest: `(T_K, L)` observed coarsest counts up to the origin.
        history_fine: one observed `(T_k, L)` array per scale up to the origin,
            read by each finer scale's lagged block; absent leaves it out.

    Returns:
        One `(H_k, L)` array per scale, finest first, whose block sums agree
        across every adjacent pair.
    """
    K = hierarchy.K
    gamma: List[np.ndarray] = [None] * K  # type: ignore[list-item]
    factor: List[np.ndarray] = [None] * K  # type: ignore[list-item]
    out: List[np.ndarray] = [None] * K  # type: ignore[list-item]

    for ki in range(K - 1, -1, -1):
        params = params_list[ki]
        if ki == K - 1:
            parent_gamma = None
        else:
            parent_gamma = gamma[ki + 1][hierarchy.parent_maps[ki]]
        gamma[ki] = roll_regime(
            params.regime, origins_gamma[ki], ctx_future[ki], parent_gamma
        )
        factor[ki] = roll_factor(
            params.factor, origins_factor[ki], gamma[ki], ctx_future[ki]
        )
        if ki == K - 1:
            out[ki] = forecast_coarse(
                params, gamma[ki], factor[ki], ctx_future[ki], history_coarsest
            )
            continue
        blocks = np.asarray(hierarchy.parent_maps[ki])
        past = None if history_fine is None else history_fine[ki]
        shares = allocation_weights(
            params, gamma[ki], factor[ki], ctx_future[ki], blocks, past, None
        )
        out[ki] = shares * out[ki + 1][blocks]
        # A lag reaching inside the window needs the epochs it points at, so
        # the pass above supplies them and this one reads them back.
        if past is not None and params.fine_count.betaLag:
            shares = allocation_weights(
                params, gamma[ki], factor[ki], ctx_future[ki], blocks, past, out[ki]
            )
            out[ki] = shares * out[ki + 1][blocks]
    return out
