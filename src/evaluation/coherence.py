"""
src/evaluation/coherence.py
===========================
Whether a fitted model's own predictions agree across the scales it was fitted at.

`MultiScaleData.check_reconciliation` reads the observed arrays, and those are
built by aggregation, so it returns zero for every method that was ever scored
and says nothing about any model.  What a claim about disagreeing granularities
needs is the other quantity: each scale's own fitted mean, predicted without
consulting the other scale, aggregated and compared.

Write $\\hat\\mu_t^{(k,\\ell)}$ for the mean scale $k$'s own model assigns to
epoch $t$ and label $\\ell$ with no scale-$k+1$ quantity supplied to it.  The
coherence gap of one adjacent pair is

    gap_k = mean over coarse epochs c and labels l of
            | sum over children t of c of mu_hat[t, l] - mu_hat[c, l] |.

Two facts make this the right measure of the claim.

*It is exactly zero for a nested model, and provably so.*  A finer scale of the
nested model is an allocation of its parent total, so its unconditional mean is
the parent's mean times the block shares, and the shares of one block sum to
one.  Summing the children returns the parent's own mean by the tower property,
for every block and every label, whatever the shares are.  `coherence_gap`
walks that recursion and does not assert the result, so a defect in the
allocation would show up here as a positive number.

*It is free to be positive for anything else.*  Two models fitted at two
granularities carry two unrelated intercepts and two unrelated lag blocks, and
nothing constrains the sum of one to the other.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from ..core.variables import MultiScaleData
from ..model.params import ScaleParams

#: The predictor is exponentiated, so it is held inside the range a float can
#: represent before it is.
_MAX_ETA = 30.0


def _regime_mix(betaS: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """The regime block at the posterior over states, epoch by epoch."""
    if betaS.size == 0 or gamma.size == 0:
        return 0.0
    return np.asarray(gamma) @ np.asarray(betaS)


def _factor_mix(betaH: np.ndarray, m_H: np.ndarray) -> np.ndarray:
    """The factor block at the posterior mean, epoch by epoch."""
    if betaH.size == 0 or m_H is None or np.size(m_H) == 0:
        return 0.0
    return np.asarray(m_H) @ np.asarray(betaH)


def _context(betaCtx: np.ndarray, C: np.ndarray) -> np.ndarray:
    """The context block, epoch by epoch."""
    if betaCtx.size == 0 or C.size == 0:
        return 0.0
    return np.asarray(C, dtype=float) @ np.asarray(betaCtx)


def coarse_mean(
    params: ScaleParams, gamma: np.ndarray, m_H: np.ndarray, C: np.ndarray
) -> np.ndarray:
    """
    The zero-inflated negative binomial mean of one coarsest-style scale.

    The lag block is left out, so what comes back is the marginal rate the scale
    carries before any realized history is supplied to it.  That is the quantity
    the coherence gap compares, since a lagged term evaluated on observed
    history would import the other scale's data through the back door.

    Args:
        params: a scale whose emission is the coarsest-style pair.
        gamma: (T, M) regime posterior at that scale.
        m_H: (T, r) factor posterior mean, empty when the scale carries none.
        C: (T, d) context at that scale.

    Returns:
        (T, L) expected counts.

    Raises:
        ValueError: the scale carries an allocation emission and not an
            emission of its own.
    """
    if not params.is_coarsest:
        raise ValueError(
            f"scale {params.k} carries an allocation and has no mean of its own; "
            "coarse_mean is for a scale fitted as its own model"
        )
    count = params.coarse_count
    eta = count.beta0[None, :] + np.zeros((gamma.shape[0], count.L))
    eta = eta + _regime_mix(count.betaS, gamma)
    eta = eta + _factor_mix(count.betaH, m_H)
    eta = eta + _context(count.betaCtx, C)
    mu = np.exp(np.clip(eta, -_MAX_ETA, _MAX_ETA))

    logit = count.gamma0[None, :] + np.zeros_like(mu)
    logit = logit + _regime_mix(count.gammaS, gamma)
    logit = logit + _factor_mix(count.gammaH, m_H)
    logit = logit + _context(count.gammaCtx, C)
    pi = 1.0 / (1.0 + np.exp(-np.clip(logit, -_MAX_ETA, _MAX_ETA)))
    return (1.0 - pi) * mu


def block_shares(
    params: ScaleParams, gamma: np.ndarray, m_H: np.ndarray, C: np.ndarray, blocks
) -> np.ndarray:
    """
    The allocation weights one finer scale gives the children of each block.

    The weights of one block sum to one, which is what makes the summed child
    mean return the parent mean.

    Args:
        params: a scale whose emission is the allocation pair.
        gamma: (T, M) regime posterior at that scale.
        m_H: (T, r) factor posterior mean.
        C: (T, d) context at that scale.
        blocks: (T,) the parent epoch of every child epoch.

    Returns:
        (T, L) weights, summing to one over the children of each block.

    Raises:
        ValueError: the scale carries its own emission and not an allocation.
    """
    if params.is_coarsest:
        raise ValueError(
            f"scale {params.k} carries its own emission and allocates nothing"
        )
    count = params.fine_count
    eta = count.beta0[None, :] + np.zeros((gamma.shape[0], count.L))
    eta = eta + _regime_mix(count.betaS, gamma)
    eta = eta + _factor_mix(count.betaH, m_H)
    eta = eta + _context(count.betaCtx, C)
    eta = np.clip(eta, -_MAX_ETA, _MAX_ETA)

    blocks = np.asarray(blocks)
    weights = np.zeros_like(eta)
    for parent in np.unique(blocks):
        rows = np.flatnonzero(blocks == parent)
        block = np.exp(eta[rows] - eta[rows].max(axis=0, keepdims=True))
        weights[rows] = block / np.maximum(block.sum(axis=0, keepdims=True), 1e-300)
    return weights


def predicted_means(
    data: MultiScaleData,
    params_list: Sequence[ScaleParams],
    summaries: Sequence,
) -> List[np.ndarray]:
    """
    Each scale's own predicted count mean, in the order scale 1 first.

    A scale fitted as its own model reports its emission mean.  A scale that
    allocates reports its parent's mean spread over the block by its own
    weights, which is the only mean an allocation has.

    Args:
        data: the arrays and the hierarchy the scales were fitted on.
        params_list: one entry per scale, scale 1 first.
        summaries: one posterior summary per scale, scale 1 first.

    Returns:
        One (T_k, L) array per scale.
    """
    K = data.K
    means: List[np.ndarray] = [None] * K  # type: ignore[list-item]
    for ki in range(K - 1, -1, -1):
        k = ki + 1
        params, summary = params_list[ki], summaries[ki]
        gamma = np.asarray(summary.gamma)
        m_H = np.asarray(summary.m_H) if summary.m_H is not None else np.zeros((0, 0))
        C = data.get_C(k)
        if params.is_coarsest:
            means[ki] = coarse_mean(params, gamma, m_H, C)
            continue
        blocks = np.asarray(data.hierarchy.parent_maps[ki])
        shares = block_shares(params, gamma, m_H, C, blocks)
        means[ki] = shares * means[ki + 1][blocks]
    return means


def coherence_gap(
    data: MultiScaleData, means: Sequence[np.ndarray]
) -> Dict[str, float]:
    """
    How far each scale's own prediction is from the scale above it, per pair.

    Args:
        data: the hierarchy the scales were fitted on.
        means: one (T_k, L) predicted mean per scale, scale 1 first.

    Returns:
        `gap_k{k}` for every adjacent pair and `gap` for their mean, in
        episodes per coarse epoch and label.
    """
    out: Dict[str, float] = {}
    per_pair: List[float] = []
    for k in range(1, data.K):
        blocks = np.asarray(data.hierarchy.parent_maps[k - 1])
        parent_epochs = data.hierarchy.n_epochs(k + 1)
        rolled = np.zeros_like(means[k])
        np.add.at(rolled, blocks, means[k - 1])
        gap = float(np.mean(np.abs(rolled[:parent_epochs] - means[k][:parent_epochs])))
        out[f"gap_k{k}"] = gap
        per_pair.append(gap)
    out["gap"] = float(np.mean(per_pair)) if per_pair else 0.0
    return out
