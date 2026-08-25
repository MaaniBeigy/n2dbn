"""
learning/param_estimation.py
============================
Tabular estimation of the regime transition table, as an alternative to the
gradient step that the M-step takes by default.

What this adds
--------------
The regime block of the M-step maximizes a softmax likelihood by gradient
ascent, so a fixed step budget bounds how far one sweep can move.  When the
regime chain carries no context and no cross-scale term the same likelihood has
a closed form, the normalized expected transition counts, and there is no reason
to walk to it.  Even with those terms present the closed form is a proposal that
costs one pass over the posterior, and the monotonicity check of the M-step
keeps it only when it raises the block objective, so offering it is free.

Estimators
----------
`soft-mle`
    Expected counts from the forward-backward posterior, normalized.  This is
    the M-step of expectation maximization for a Markov chain, which is why no
    separate expectation-maximization entry appears below.
`soft-bayes`
    The same counts with a BDeu Dirichlet prior added, with pseudo-count
    `equivalent_sample_size / (M * M_prev)` per cell.  A regime that the
    posterior never visits then still receives a proper distribution instead of
    a division by zero.
`pgmpy-mle`, `pgmpy-bayes`
    The same two estimators as implemented by pgmpy, applied to the hard
    assignment `argmax_m gamma_t(m)` of the posterior.  They exist so that the
    cost of discarding the posterior uncertainty can be measured rather than
    assumed: on a chain whose posterior is nearly deterministic the four agree,
    and they separate as the posterior flattens.

Mapping a table onto the parameters
-----------------------------------
The model writes the transition logit as

    bias[m] + trans[m, m'] + cross[m, n] + ctx[m] . c_t,

so a table `P(m | m')` maps onto it by `trans[m, m'] = log P(m | m')` with
`bias = 0`, leaving `cross` and `ctx` untouched.  The map is exact when the
chain has neither a context nor a cross-scale term, and is a proposal otherwise.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Optional, Tuple

import numpy as np

from ..inference.posterior import PosteriorSummaries
from ..model.params import RegimeParams

REGIME_ESTIMATORS = ("soft-mle", "soft-bayes", "pgmpy-mle", "pgmpy-bayes")

_LOG_FLOOR = 1e-12


# ---------------------------------------------------------------------------
# Counts
# ---------------------------------------------------------------------------


def expected_regime_counts(q: PosteriorSummaries) -> Tuple[np.ndarray, np.ndarray]:
    """
    Expected initial and transition counts under the posterior.

    Returns:
        `(init_counts, trans_counts)` of shapes `(M,)` and `(M_prev, M)`, where
        `trans_counts[m', m] = sum_t xi_t(m', m)`.
    """
    gamma = np.asarray(q.gamma, dtype=float)
    M = gamma.shape[1]
    init = gamma[0].astype(float).copy()
    if gamma.shape[0] < 2 or q.xi is None or np.size(q.xi) == 0:
        return init, np.zeros((M, M))
    xi = np.asarray(q.xi, dtype=float)[: gamma.shape[0] - 1]
    return init, xi.sum(axis=0)


def hard_regime_frame(q: PosteriorSummaries):
    """
    Consecutive regime pairs of the maximum-a-posteriori path, as a data frame.

    Columns are `S_prev` and `S_t`, both categorical over the full alphabet, so
    a regime the path never enters still appears as a state of the estimated
    conditional probability table rather than vanishing from it.
    """
    import pandas as pd

    gamma = np.asarray(q.gamma, dtype=float)
    path = gamma.argmax(axis=1)
    states = list(range(gamma.shape[1]))
    dtype = pd.CategoricalDtype(categories=states, ordered=False)
    return pd.DataFrame(
        {
            "S_prev": pd.Series(path[:-1], dtype=dtype),
            "S_t": pd.Series(path[1:], dtype=dtype),
        }
    )


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------


def _normalize_rows(counts: np.ndarray) -> np.ndarray:
    """Row-normalize, sending an all-zero row to the uniform distribution."""
    total = counts.sum(axis=1, keepdims=True)
    uniform = np.full_like(counts, 1.0 / counts.shape[1])
    return np.where(total > _LOG_FLOOR, counts / np.maximum(total, _LOG_FLOOR), uniform)


def _pgmpy_transition_table(q: PosteriorSummaries, method: str, ess: float):
    """Transition table from pgmpy, fitted to the hard-assigned path."""
    from pgmpy.estimators import BayesianEstimator, MaximumLikelihoodEstimator
    from pgmpy.models import DiscreteBayesianNetwork

    frame = hard_regime_frame(q)
    M = int(np.asarray(q.gamma).shape[1])
    net = DiscreteBayesianNetwork([("S_prev", "S_t")])
    if method == "pgmpy-mle":
        cpds = MaximumLikelihoodEstimator(net, frame).get_parameters()
    else:
        cpds = BayesianEstimator(net, frame).get_parameters(
            prior_type="BDeu", equivalent_sample_size=ess
        )
    cpd = next(c for c in cpds if c.variable == "S_t")
    # pgmpy lays the table out as (child state, parent state).
    values = np.asarray(cpd.get_values(), dtype=float).reshape(M, -1)
    return _normalize_rows(values.T)


def estimate_transition_table(
    q: PosteriorSummaries,
    method: str = "soft-bayes",
    equivalent_sample_size: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initial distribution and transition table of the regime chain.

    Args:
        q: posterior summaries of one scale.
        method: one of `REGIME_ESTIMATORS`.
        equivalent_sample_size: total prior mass of the Dirichlet prior, split
            evenly over the cells, used by the two Bayesian estimators.

    Returns:
        `(init, trans)` of shapes `(M,)` and `(M_prev, M)`, both normalized.

    Raises:
        ValueError: the method name is unknown.
    """
    if method not in REGIME_ESTIMATORS:
        raise ValueError(
            f"unknown estimator {method!r}, expected one of {list(REGIME_ESTIMATORS)}"
        )
    init_counts, trans_counts = expected_regime_counts(q)
    M = init_counts.shape[0]

    if method.startswith("pgmpy"):
        init_prior = equivalent_sample_size / M
        init = (init_counts + init_prior) / (init_counts.sum() + equivalent_sample_size)
        return init, _pgmpy_transition_table(q, method, equivalent_sample_size)

    if method == "soft-bayes":
        cell = equivalent_sample_size / (M * M)
        trans = _normalize_rows(trans_counts + cell)
        init = (init_counts + equivalent_sample_size / M) / (
            init_counts.sum() + equivalent_sample_size
        )
        return init, trans

    trans = _normalize_rows(trans_counts)
    total = init_counts.sum()
    init = init_counts / total if total > _LOG_FLOOR else np.full(M, 1.0 / M)
    return init, trans


# ---------------------------------------------------------------------------
# Mapping onto RegimeParams
# ---------------------------------------------------------------------------


def regime_params_from_table(
    params: RegimeParams,
    init: np.ndarray,
    trans: np.ndarray,
) -> RegimeParams:
    """Copy of `params` whose `init` and `trans` blocks hold the log table."""
    out = deepcopy(params)
    out.init = np.log(np.maximum(np.asarray(init, dtype=float), _LOG_FLOOR))
    out.bias = np.zeros_like(out.bias)
    out.trans = np.log(np.maximum(np.asarray(trans, dtype=float), _LOG_FLOOR)).T
    return out


def update_regime_table(
    q: PosteriorSummaries,
    params: RegimeParams,
    method: str = "soft-bayes",
    equivalent_sample_size: float = 5.0,
) -> RegimeParams:
    """Regime parameters whose transition block is the estimated table."""
    init, trans = estimate_transition_table(
        q, method=method, equivalent_sample_size=equivalent_sample_size
    )
    return regime_params_from_table(params, init, trans)
