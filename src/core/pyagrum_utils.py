"""
core/pyagrum_utils.py
=====================
Utilities for converting between N²DBN parameter objects and pyAgrum CPTs.

These functions bridge the gap between the numpy-based `RegimeParams`
representation (softmax logits) and pyAgrum's `Potential`-based CPTs.

All functions that touch pyAgrum objects are guarded by a lazy import;
`extract_transition_matrix` is pure numpy and works without pyAgrum.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.special import softmax

from ..model.params import RegimeParams

# ---------------------------------------------------------------------------
# Pure-numpy helpers (no pyAgrum required)
# ---------------------------------------------------------------------------


def extract_transition_matrix(
    regime_params: RegimeParams,
    ctx_t: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute the (M, M) transition matrix from `RegimeParams`.

    Parameters
    ----------
    regime_params : RegimeParams
    ctx_t : (d_ctx,) context vector, or None for zero context.

    Returns
    -------
    P : (M, M) array where `P[m_next, m_prev]` = P(S_t=m_next | S_{t-1}=m_prev).
        Each column sums to 1.
    """
    M = regime_params.M
    d_ctx = regime_params.d_ctx
    if ctx_t is None:
        ctx_t = np.zeros(d_ctx)
    P = np.zeros((M, M))
    for prev_m in range(M):
        logits = regime_params.logit_transition(prev_m, parent_n=None, ctx_t=ctx_t)
        P[:, prev_m] = softmax(logits)
    return P


def extract_init_distribution(
    regime_params: RegimeParams,
    ctx_t: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute the initial regime distribution from `RegimeParams`.

    Returns
    -------
    pi0 : (M,) array summing to 1.
    """
    d_ctx = regime_params.d_ctx
    if ctx_t is None:
        ctx_t = np.zeros(d_ctx)
    logits = regime_params.logit_init(ctx_t)
    return softmax(logits)


# ---------------------------------------------------------------------------
# pyAgrum CPT functions
# ---------------------------------------------------------------------------


def fill_regime_cpt(
    bn: "pyagrum.BayesNet",
    regime_params: RegimeParams,
    ctx_t: Optional[np.ndarray] = None,
) -> None:
    """
    Fill the regime-transition CPT in a pyAgrum BN built by
    :meth:`MultiScaleDBNGraph.to_pyagrum_bn`.

    Sets:
    * `bn.cpt("S_t_prev")` ← initial distribution `softmax(init)`.
    * `bn.cpt("S_t")`      ← conditional `P(S_t | S_t_prev)`.

    Parameters
    ----------
    bn : pyAgrum.BayesNet
        Must contain nodes named `S_t` and `S_t_prev`.
    regime_params : RegimeParams
    ctx_t : optional context vector (defaults to zero).
    """
    from .graph import _require_pyagrum

    _require_pyagrum("fill_regime_cpt")

    pi0 = extract_init_distribution(regime_params, ctx_t)
    bn.cpt("S_t_prev").fillWith(pi0.tolist())

    P = extract_transition_matrix(regime_params, ctx_t)
    # pyAgrum CPT for S_t with parent S_t_prev has shape (M_St, M_St_prev)
    # where dim 0 = S_t, dim 1 = S_t_prev.
    # P[m_next, m_prev] matches this layout directly.
    bn.cpt("S_t").fillWith(P.flatten().tolist())


def regime_cpt_to_params(
    bn: "pyagrum.BayesNet",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract (init_probs, trans_matrix) from a pyAgrum BN's CPTs.

    Parameters
    ----------
    bn : pyAgrum.BayesNet
        Must contain nodes `S_t` and `S_t_prev`.

    Returns
    -------
    init_probs : (M,) initial distribution.
    trans_matrix : (M, M) where `trans_matrix[m_next, m_prev]` =
        P(S_t = m_next | S_{t-1} = m_prev).
    """
    from .graph import _require_pyagrum

    _require_pyagrum("regime_cpt_to_params")

    init_probs = np.array(bn.cpt("S_t_prev").toarray(), dtype=np.float64)
    trans_matrix = np.array(bn.cpt("S_t").toarray(), dtype=np.float64)
    return init_probs, trans_matrix
