"""
inference/bound.py
==================
One objective, and the two operators that carry information across scales.

The allocation formulation monitors the summed per-scale log evidence less an
edge penalty, which its own code calls an approximation: each term is conditional
on the parent's posterior treated as a fixed input, the finer terms come from a
mean field, and no entropy appears anywhere.  What it bounds is not settled.

Here there is one quantity.  The start term is quadratic in the predictor once a
Polya-Gamma variate is introduced, so the bound is

    E_q[ (x - 1/2) eta - (w/2) eta^2 ] + E_q[ log p(marks) ] + H[q] + const

and every piece of it is computable in closed form.  The quadratic form is what
makes a scale's pseudo-emission a segment sum, which is the upward message.

The two operators
-----------------
A scale-k quantity reaches a fine slot by broadcast, `B_k`, and a fine
per-slot contribution reaches a scale-k epoch by segment sum, `B_k` transposed.
They are adjoint,

    <B_k v, x> = <v, B_k^T x>,

so the upward message is not a new object to invent.  It is the same operator the
downward broadcast already uses, read the other way, which is why running the
pass in both directions needs only a schedule.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

_TINY = 1e-12


# ---------------------------------------------------------------------------
# The two adjoint operators
# ---------------------------------------------------------------------------


def broadcast(values: np.ndarray, ancestors: np.ndarray) -> np.ndarray:
    """A scale-k array read at every finest slot, which is `B_k`."""
    return np.asarray(values)[np.asarray(ancestors)]


def segment_sum(fine: np.ndarray, ancestors: np.ndarray, n_epochs: int) -> np.ndarray:
    """A finest-scale array summed into its scale-k epochs, which is `B_k` transposed."""
    fine = np.asarray(fine, dtype=float)
    ancestors = np.asarray(ancestors)
    shape = (n_epochs,) + fine.shape[1:]
    out = np.zeros(shape, dtype=float)
    np.add.at(out, ancestors, fine)
    return out


def adjoint_residual(
    values: np.ndarray, fine: np.ndarray, ancestors: np.ndarray, n_epochs: int
) -> float:
    """
    How far the two operators are from being adjoint, which should be rounding.

    Equality of `<B v, x>` and `<v, B^T x>` is the whole reason an upward message
    needs no new machinery, so it is measured here.
    """
    left = float(np.sum(broadcast(values, ancestors) * np.asarray(fine)))
    right = float(np.sum(np.asarray(values) * segment_sum(fine, ancestors, n_epochs)))
    return abs(left - right)


# ---------------------------------------------------------------------------
# The Polya-Gamma variate and the quadratic bound it gives
# ---------------------------------------------------------------------------


def pg_mean(theta: np.ndarray) -> np.ndarray:
    """
    Mean of the Polya-Gamma variate at `theta`, which is tanh(theta/2)/(2 theta).

    The value at zero is one quarter, taken as the limit.
    """
    theta = np.abs(np.asarray(theta, dtype=float))
    out = np.full_like(theta, 0.25)
    big = theta > _TINY
    out[big] = np.tanh(theta[big] / 2.0) / (2.0 * theta[big])
    return out


def quadratic_start_bound(
    x: np.ndarray,
    eta: np.ndarray,
    anchor: np.ndarray,
    at_risk: Optional[np.ndarray] = None,
) -> float:
    """
    The quadratic lower bound on the Bernoulli start term, at a given anchor.

    The bound touches the exact term when the anchor equals the predictor and
    lies below it everywhere else, which is what makes maximizing it safe.
    """
    x = (np.asarray(x) > 0).astype(float)
    eta = np.asarray(eta, dtype=float)
    anchor = np.abs(np.asarray(anchor, dtype=float))
    w = pg_mean(anchor)
    # The value the exact term takes at the anchor, plus the quadratic that
    # touches it there.
    const = -np.logaddexp(0.0, anchor) + anchor / 2.0 + w * anchor**2 / 2.0
    terms = (x - 0.5) * eta - w * eta**2 / 2.0 + const
    if at_risk is not None:
        terms = terms[at_risk]
    return float(terms.sum())


def exact_start_term(
    x: np.ndarray, eta: np.ndarray, at_risk: Optional[np.ndarray] = None
) -> float:
    """The Bernoulli term the quadratic bounds, for comparison."""
    x = (np.asarray(x) > 0).astype(float)
    eta = np.asarray(eta, dtype=float)
    terms = x * eta - np.logaddexp(0.0, eta)
    if at_risk is not None:
        terms = terms[at_risk]
    return float(terms.sum())


def pg_update(eta_mean: np.ndarray, eta_var: Optional[np.ndarray] = None) -> np.ndarray:
    """
    The variate's anchor and mean under a posterior over the predictor.

    The anchor is the root of the second moment, so a predictor the posterior is
    uncertain about is anchored further from zero and damped harder, which is the
    variance term a mean-only design would drop.
    """
    second = np.asarray(eta_mean, dtype=float) ** 2
    if eta_var is not None:
        second = second + np.asarray(eta_var, dtype=float)
    return pg_mean(np.sqrt(np.maximum(second, 0.0)))


# ---------------------------------------------------------------------------
# The exact start term's own rate and curvature
# ---------------------------------------------------------------------------


def start_rate(eta: np.ndarray) -> np.ndarray:
    """The Bernoulli start probability at a predictor, which is its logistic."""
    return np.exp(-np.logaddexp(0.0, -np.asarray(eta, dtype=float)))


def start_curvature(eta: np.ndarray) -> np.ndarray:
    """
    Curvature of the exact start term, which is the rate times its complement.

    `pg_mean` is the curvature of the quadratic that lies under the same term,
    and the two part company as the rate leaves one half.  At a start
    probability of 1.45e-3 the quadratic carries 0.0763 where the term itself
    carries 0.00145, a factor of 52.5, since the bound decays as one over twice
    the predictor while the term decays as its exponential.  A quadratic that
    steep is still a lower bound and is still safe to maximize.  It also moves
    one fifty-second of the way to the maximizer per step and prices a column at
    one fifty-second of the evidence the column carries, so a number read as
    nats is read against this curvature.
    """
    eta = np.asarray(eta, dtype=float)
    return np.exp(-np.logaddexp(0.0, -eta) - np.logaddexp(0.0, eta))


def fisher_response(x: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """
    The response a Fisher-scoring step puts against a column.

    `start_curvature` supplies the step's weights and this its right-hand side,
    so the solve that maximized a quadratic at fixed weights takes a Newton step
    on the exact term once it is handed these two.  At a predictor the step has
    already reached, the weighted part cancels and what a column meets is the
    exact score, x - sigma(eta), which is what makes a candidate's residual the
    score statistic's numerator.
    """
    x = (np.asarray(x) > 0).astype(float)
    eta = np.asarray(eta, dtype=float)
    return x - start_rate(eta) + start_curvature(eta) * eta


# ---------------------------------------------------------------------------
# The upward message
# ---------------------------------------------------------------------------


def upward_message(
    x: np.ndarray,
    eta_rest: np.ndarray,
    own_effect: np.ndarray,
    weights: np.ndarray,
    ancestors: np.ndarray,
    n_epochs: int,
    at_risk: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    A scale's per-state pseudo-emission, which is a segment sum and nothing more.

    `own_effect` is what the scale contributes to the predictor under each of its
    states, shape (M, L); `eta_rest` is what every other part contributes, shape
    (T, L).  The quadratic form separates the two, so the state-dependent part of
    each slot's contribution is a plain array that the segment sum reduces to one
    row per coarse epoch.

    Returns (n_epochs, M).
    """
    x = (np.asarray(x) > 0).astype(float)
    eta_rest = np.asarray(eta_rest, dtype=float)
    own_effect = np.asarray(own_effect, dtype=float)
    weights = np.asarray(weights, dtype=float)
    keep = np.ones(x.shape, dtype=bool) if at_risk is None else np.asarray(at_risk)

    linear = (x - 0.5 - weights * eta_rest) * keep  # (T, L)
    quad = weights * keep  # (T, L)
    per_slot = np.einsum("tl,ml->tm", linear, own_effect) - 0.5 * np.einsum(
        "tl,ml->tm", quad, own_effect**2
    )
    return segment_sum(per_slot, ancestors, n_epochs)


# ---------------------------------------------------------------------------
# The objective
# ---------------------------------------------------------------------------


def chain_entropy(gamma: np.ndarray, xi: Optional[np.ndarray] = None) -> float:
    """
    Entropy of a chain posterior, exact when the pair marginals are given.

    With `xi` the value is the chain's own entropy,

        H = -sum_t sum_{m,m'} xi_t(m,m') log xi_t(m,m')
            + sum_{t interior} sum_m gamma_t(m) log gamma_t(m),

    which is what the factorization of a Markov chain gives.  Without `xi` it is
    the entropy of the independent approximation, larger than the truth whenever
    the chain is persistent, and a caller records which one it used.
    """
    gamma = np.asarray(gamma, dtype=float)
    marginal = -float(np.sum(gamma * np.log(np.clip(gamma, _TINY, None))))
    if xi is None:
        return marginal
    xi = np.asarray(xi, dtype=float)
    pair = -float(np.sum(xi * np.log(np.clip(xi, _TINY, None))))
    interior = -float(np.sum(gamma[1:-1] * np.log(np.clip(gamma[1:-1], _TINY, None))))
    return pair - interior


def gaussian_entropy(covariances: np.ndarray) -> float:
    """Entropy of the factor posterior, one Gaussian per epoch."""
    covariances = np.asarray(covariances, dtype=float)
    if covariances.size == 0 or covariances.shape[-1] == 0:
        return 0.0
    r = covariances.shape[-1]
    sign, logdet = np.linalg.slogdet(covariances)
    keep = sign > 0
    return float(np.sum(0.5 * (r * (1.0 + np.log(2.0 * np.pi)) + logdet[keep])))


def evidence_bound(
    start_term: float,
    mark_term: float,
    entropy: float,
    penalty: float = 0.0,
) -> float:
    """
    The one objective: the two data terms, the posterior entropy, less the price.

    Every piece is named, so a reported number says what it contains, which the
    summed per-scale log evidence it replaces could not.
    """
    return float(start_term + mark_term + entropy - penalty)
