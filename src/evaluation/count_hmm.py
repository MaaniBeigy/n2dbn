"""
evaluation/count_hmm.py
=======================
A Poisson hidden Markov model of a count series, and how two fitted regime sets
are compared.

Two rows of the nonstationarity grid wait on one dependency.  A14 asks whether a
series moves between a small number of levels it holds for a while, and B5 asks
whether the fine scale changes regime where its parent does.  Both are read off
one fit: a hidden state per epoch, one Poisson rate per state, a full transition
matrix and an initial distribution, fitted by Baum-Welch.

Every pass runs in log space.  A forward recursion carried on probabilities
underflows inside a few hundred epochs of a count series, and the panel holds
1344 of them at the hour scale, so the forward, the backward and the Viterbi
recursions are sums of logarithms taken with `logsumexp`.  The rates start from
the quantiles of the series with a seeded jitter, the iteration cap and the
convergence tolerance are declared, and the states are ordered by rate at the
end of the fit, so one series fitted twice carries the same labels.

How many regimes a series holds is a choice, so it is selected by BIC over a
declared range of state counts, and a count needing more epochs than the series
holds is never tried.  What a caller gets is the fitted model, the decoded path,
the criterion at every count, the stationary distribution of the transition
matrix, the dwell time each state implies and the dwell times the decoded path
shows.

Two fits are compared on the total variation between their stationary
distributions and on the rate gaps of a Hungarian matching of their states,
which pairs two state sets of different sizes.  B5 is the adjusted Rand index
between a fine decoded path and its parent's path broadcast down to the fine
epochs, which is one number for how far the two scales change together.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import gammaln, logsumexp

#: The state counts the criterion chooses between.
STATE_COUNTS: Tuple[int, ...] = (1, 2, 3, 4)

#: How many epochs one state needs before a fit is allowed to claim it.
MIN_EPOCHS_PER_STATE = 10

#: A regime is a level held across cycles, so the summary scalar reports one
#: regime for a series carrying fewer calendar cycles than this.
MIN_CYCLES = 2.0

#: The declared iteration cap of Baum-Welch and the log-likelihood gain it stops
#: below.
MAX_ITERATIONS = 100
TOLERANCE = 1e-6

#: Kept out of a logarithm of a rate that a series of zeros drives to zero.
RATE_FLOOR = 1e-9

#: The self-transition an initial fit starts from, since a regime is a level held.
HOLD = 0.9


@dataclass(frozen=True)
class PoissonHMM:
    """One fitted model, with the log-likelihood of every EM iteration."""

    rates: np.ndarray
    transition: np.ndarray
    initial: np.ndarray
    log_likelihood: float
    history: Tuple[float, ...]

    @property
    def n_states(self) -> int:
        return int(self.rates.size)


@dataclass(frozen=True)
class ForwardBackward:
    """One E-step: the two log passes, the state posterior and the pair counts."""

    log_alpha: np.ndarray
    log_beta: np.ndarray
    gamma: np.ndarray
    xi_sum: np.ndarray
    log_likelihood: float


def _counts(values: Sequence[float]) -> np.ndarray:
    """
    A count series as a float vector, refusing what a Poisson rate cannot hold.

    Raises:
        ValueError: when the series holds a non-finite or a negative entry.
    """
    x = np.asarray(values, dtype=float).ravel()
    if not np.all(np.isfinite(x)):
        raise ValueError(f"a count series is finite; got {x[~np.isfinite(x)][0]}")
    if x.size and float(x.min()) < 0.0:
        raise ValueError(f"a count series is non-negative; got {float(x.min())}")
    return x


def _log(p: np.ndarray) -> np.ndarray:
    """The logarithm of a probability array, a zero going to minus infinity."""
    with np.errstate(divide="ignore"):
        return np.log(np.asarray(p, dtype=float))


def _shapes(
    log_initial: np.ndarray, log_transition: np.ndarray, log_emission: np.ndarray
) -> Tuple[int, int]:
    """
    The epoch and state counts of one parameter set.

    Raises:
        ValueError: when the three shapes disagree on the state count, when the
            set carries no state, or when the series holds no epoch.
    """
    a = np.asarray(log_initial, dtype=float)
    p = np.asarray(log_transition, dtype=float)
    e = np.asarray(log_emission, dtype=float)
    if e.ndim != 2 or p.shape != (a.size, a.size) or e.shape[1] != a.size:
        raise ValueError(
            f"one parameter set holds one state count; got initial {a.shape}, "
            f"transition {p.shape} and emission {e.shape}"
        )
    if a.size == 0:
        raise ValueError("a pass over a series needs at least one state; got 0")
    if e.shape[0] == 0:
        raise ValueError("a pass over a series needs at least one epoch; got 0")
    return int(e.shape[0]), int(a.size)


def log_emissions(values: Sequence[float], rates: Sequence[float]) -> np.ndarray:
    """
    The log Poisson probability of every epoch under every state's rate.

    Raises:
        ValueError: when no rate is given, which leaves the epochs unexplained,
            or when a rate is one a Poisson cannot hold.
    """
    x = _counts(values)
    lam = np.asarray(rates, dtype=float).ravel()
    if lam.size == 0:
        raise ValueError("a Poisson emission holds at least one rate; got 0")
    if not np.all(np.isfinite(lam)):
        raise ValueError(f"a Poisson rate is finite; got {lam[~np.isfinite(lam)][0]}")
    if float(lam.min()) < 0.0:
        raise ValueError(f"a Poisson rate is non-negative; got {float(lam.min())}")
    lam = np.maximum(lam, RATE_FLOOR)
    return x[:, None] * np.log(lam)[None, :] - lam[None, :] - gammaln(x + 1.0)[:, None]


def initial_parameters(
    values: Sequence[float], n_states: int, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rates from the quantiles of the series, a sticky transition and a flat start.

    Raises:
        ValueError: when the state count is below one, or when the series holds
            fewer epochs than states.
    """
    if n_states < 1:
        raise ValueError(f"a fit holds at least one state; got {n_states}")
    x = _counts(values)
    if x.size < n_states:
        raise ValueError(
            f"a {n_states}-state fit needs {n_states} epochs; got {x.size}"
        )
    rng = np.random.default_rng(seed)
    quantiles = np.quantile(x, (np.arange(n_states) + 0.5) / n_states)
    # Two equal quantiles hand two states one rate, and every later E-step
    # splits such a pair evenly forever, so the seeded jitter separates them.
    jitter = 0.1 * (float(x.std()) + 1.0) * rng.standard_normal(n_states)
    rates = np.maximum(quantiles + jitter, RATE_FLOOR)
    transition = np.full(
        (n_states, n_states), (1.0 - HOLD) / max(n_states - 1, 1), dtype=float
    )
    np.fill_diagonal(transition, HOLD)
    return rates, transition, np.full(n_states, 1.0 / n_states)


def _log_sum(matrix: np.ndarray, axis: int) -> np.ndarray:
    """
    The log of a sum of exponentials along one axis, shifted by its largest term.

    The general routine costs more in its own argument handling than the sums of
    a forward pass do, and a pass over one series makes two of those calls per
    epoch.  A column whose every term is -inf carries no path and stays -inf.
    """
    top = np.max(matrix, axis=axis, keepdims=True)
    top = np.where(np.isfinite(top), top, 0.0)
    with np.errstate(divide="ignore"):
        total = np.log(np.exp(matrix - top).sum(axis=axis))
    return np.squeeze(top, axis=axis) + total


def forward_backward(
    log_initial: np.ndarray, log_transition: np.ndarray, log_emission: np.ndarray
) -> ForwardBackward:
    """
    The log-space forward and backward passes, and what an M-step reads off them.

    Raises:
        ValueError: when the three parameter shapes disagree, when the series
            holds no epoch, or when the parameters give the series no path of
            positive probability, which leaves every posterior undefined.
    """
    n, k = _shapes(log_initial, log_transition, log_emission)
    a = np.asarray(log_initial, dtype=float)
    p = np.asarray(log_transition, dtype=float)
    e = np.asarray(log_emission, dtype=float)
    log_alpha = np.empty((n, k), dtype=float)
    log_alpha[0] = a + e[0]
    for t in range(1, n):
        log_alpha[t] = _log_sum(log_alpha[t - 1][:, None] + p, axis=0) + e[t]
    log_beta = np.zeros((n, k), dtype=float)
    for t in range(n - 2, -1, -1):
        log_beta[t] = _log_sum(p + (e[t + 1] + log_beta[t + 1])[None, :], axis=1)
    total = float(logsumexp(log_alpha[-1]))
    if not math.isfinite(total):
        raise ValueError(
            f"a posterior needs a series the parameters can produce; got total "
            f"log probability {total}"
        )
    log_xi = (
        log_alpha[:-1, :, None]
        + p[None, :, :]
        + (e[1:] + log_beta[1:])[:, None, :]
        - total
    )
    return ForwardBackward(
        log_alpha=log_alpha,
        log_beta=log_beta,
        gamma=np.exp(log_alpha + log_beta - total),
        xi_sum=np.exp(log_xi).sum(axis=0),
        log_likelihood=total,
    )


def m_step(
    values: Sequence[float], posterior: ForwardBackward
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The rates, transition and initial distribution one E-step implies.

    A state the posterior never occupies leaves no pair count to normalize, and
    the uniform row it gets is the only proper distribution left for it.

    Raises:
        ValueError: when the posterior does not hold one row per epoch.
    """
    x = _counts(values)
    if posterior.gamma.shape[0] != x.size:
        raise ValueError(
            f"an M-step reads one posterior row per epoch; got "
            f"{posterior.gamma.shape[0]} for {x.size} epochs"
        )
    weight = posterior.gamma.sum(axis=0)
    rates = np.maximum(
        posterior.gamma.T @ x / np.maximum(weight, RATE_FLOOR), RATE_FLOOR
    )
    leaving = posterior.xi_sum.sum(axis=1)
    transition = np.where(
        leaving[:, None] > 0.0,
        posterior.xi_sum / np.maximum(leaving, RATE_FLOOR)[:, None],
        1.0 / weight.size,
    )
    return rates, transition, posterior.gamma[0]


def _order_by_rate(
    rates: np.ndarray, transition: np.ndarray, initial: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Relabel the states by ascending rate, which fixes what state zero means."""
    order = np.argsort(rates)
    return rates[order], transition[np.ix_(order, order)], initial[order]


def fit_poisson_hmm(
    values: Sequence[float],
    n_states: int,
    seed: int = 0,
    max_iterations: int = MAX_ITERATIONS,
    tolerance: float = TOLERANCE,
) -> PoissonHMM:
    """
    Fit a Poisson hidden Markov model by Baum-Welch, with states ordered by rate.

    Args:
        values: one participant's count series at one scale.
        n_states: how many regimes the fit holds.
        seed: what the quantile initialization is jittered with.
        max_iterations: the declared cap on EM iterations.
        tolerance: the log-likelihood gain EM stops below.

    Returns:
        The fitted model, whose history holds the log-likelihood of every EM
        iteration in the order they were reached and whose reported
        log-likelihood is the one of the parameters it carries, which stands
        one M-step above the last iteration when the cap binds.

    Raises:
        ValueError: when the cap or the tolerance cannot work, or when the
            series is shorter than the state count.
    """
    if max_iterations < 1:
        raise ValueError(f"an iteration cap is at least one; got {max_iterations}")
    if tolerance <= 0.0:
        raise ValueError(f"a convergence tolerance is positive; got {tolerance}")
    x = _counts(values)
    rates, transition, initial = initial_parameters(x, n_states, seed)
    history: List[float] = []
    for _ in range(max_iterations):
        state = forward_backward(
            _log(initial), _log(transition), log_emissions(x, rates)
        )
        history.append(state.log_likelihood)
        if len(history) > 1 and history[-1] - history[-2] <= tolerance:
            break
        rates, transition, initial = m_step(x, state)
    rates, transition, initial = _order_by_rate(rates, transition, initial)
    total = forward_backward(
        _log(initial), _log(transition), log_emissions(x, rates)
    ).log_likelihood
    return PoissonHMM(rates, transition, initial, total, tuple(history))


def viterbi(
    log_initial: np.ndarray, log_transition: np.ndarray, log_emission: np.ndarray
) -> np.ndarray:
    """
    The most probable state path, decoded in log space.

    Raises:
        ValueError: when the three parameter shapes disagree, when the series
            holds no epoch, or when the parameters give the series no path of
            positive probability, which leaves every path equally impossible.
    """
    n, k = _shapes(log_initial, log_transition, log_emission)
    a = np.asarray(log_initial, dtype=float)
    p = np.asarray(log_transition, dtype=float)
    e = np.asarray(log_emission, dtype=float)
    score = a + e[0]
    origin = np.zeros((n, k), dtype=np.int64)
    for t in range(1, n):
        step = score[:, None] + p
        origin[t] = np.argmax(step, axis=0)
        score = step[origin[t], np.arange(k)] + e[t]
    best = float(np.max(score))
    if not math.isfinite(best):
        raise ValueError(
            f"a decoding needs a path the parameters can take; got best path "
            f"log probability {best}"
        )
    path = np.zeros(n, dtype=np.int64)
    path[-1] = int(np.argmax(score))
    for t in range(n - 1, 0, -1):
        path[t - 1] = origin[t, path[t]]
    return path


def decode(values: Sequence[float], model: PoissonHMM) -> np.ndarray:
    """The Viterbi path of one series under one fitted model."""
    return viterbi(
        _log(model.initial), _log(model.transition), log_emissions(values, model.rates)
    )


def bic(log_likelihood: float, n_states: int, n_epochs: int) -> float:
    """
    The Bayesian criterion of one fit, where the smaller value is the better fit.

    A k-state model holds k rates, k(k-1) free transition probabilities and k-1
    free initial probabilities.

    Raises:
        ValueError: when no epoch is given, since a criterion prices a fit
            against what it was fitted on, or when the state count is below one,
            which frees a negative number of parameters.
    """
    if n_states < 1:
        raise ValueError(f"a criterion prices at least one state; got {n_states}")
    if n_epochs < 1:
        raise ValueError(f"a criterion prices at least one epoch; got {n_epochs}")
    parameters = n_states * n_states + n_states - 1
    return float(-2.0 * log_likelihood + parameters * math.log(n_epochs))


def admissible_states(
    n_epochs: int, state_counts: Sequence[int] = STATE_COUNTS
) -> Tuple[int, ...]:
    """
    The declared state counts a series of this length carries enough epochs for.

    One state is always admissible, since a series always holds one level.

    Raises:
        ValueError: when the series holds no epoch.
    """
    if n_epochs < 1:
        raise ValueError(f"a fit needs at least one epoch; got {n_epochs}")
    allowed = tuple(
        int(k) for k in state_counts if k >= 1 and k * MIN_EPOCHS_PER_STATE <= n_epochs
    )
    return allowed or (1,)


def select_states(
    values: Sequence[float], state_counts: Sequence[int] = STATE_COUNTS, seed: int = 0
) -> Tuple[PoissonHMM, Tuple[Tuple[int, float], ...]]:
    """
    Fit every admissible state count and keep the one the criterion prices lowest.

    Returns:
        The chosen model, and the criterion at every count that was tried.
    """
    x = _counts(values)
    fits = [
        fit_poisson_hmm(x, k, seed) for k in admissible_states(x.size, state_counts)
    ]
    scores = tuple(
        (fit.n_states, bic(fit.log_likelihood, fit.n_states, x.size)) for fit in fits
    )
    return fits[int(np.argmin([score for _, score in scores]))], scores


def _check_transition(transition: np.ndarray) -> np.ndarray:
    """
    One transition matrix as a float array.

    Raises:
        ValueError: when the matrix is not a square, filled set of non-negative
            rows summing to one.
    """
    p = np.asarray(transition, dtype=float)
    if p.ndim != 2 or p.shape[0] != p.shape[1] or p.shape[0] == 0:
        raise ValueError(f"a transition matrix is square and filled; got {p.shape}")
    if float(p.min()) < 0.0 or not np.allclose(p.sum(axis=1), 1.0):
        raise ValueError(
            f"a transition matrix holds non-negative rows summing to one; got row "
            f"sums {p.sum(axis=1)} and smallest entry {float(p.min())}"
        )
    return p


def stationary_distribution(transition: np.ndarray) -> np.ndarray:
    """
    The normalized left eigenvector of eigenvalue one.

    Raises:
        ValueError: when the matrix is not a proper transition matrix.
    """
    p = _check_transition(transition)
    eigenvalues, eigenvectors = np.linalg.eig(p.T)
    vector = np.abs(np.real(eigenvectors[:, int(np.argmin(np.abs(eigenvalues - 1.0)))]))
    return vector / float(vector.sum())


def expected_dwell(transition: np.ndarray) -> np.ndarray:
    """
    How many epochs each state is expected to hold, one over its leaving chance.

    A state that never leaves dwells for an infinite time, which is what an
    absorbing regime means.

    Raises:
        ValueError: when the matrix is not a proper transition matrix.
    """
    hold = np.diag(_check_transition(transition))
    with np.errstate(divide="ignore"):
        return np.where(hold >= 1.0, np.inf, 1.0 / (1.0 - hold))


def empirical_dwell(path: Sequence[int], n_states: int) -> np.ndarray:
    """
    The mean run length of each state in a decoded path, zero where it is absent.

    Raises:
        ValueError: when the state count is below one, or when the path holds a
            state outside it.
    """
    if n_states < 1:
        raise ValueError(f"a path holds at least one state; got {n_states}")
    z = np.asarray(path, dtype=np.int64).ravel()
    totals = np.zeros(n_states, dtype=float)
    counts = np.zeros(n_states, dtype=float)
    if z.size == 0:
        return totals
    if int(z.min()) < 0 or int(z.max()) >= n_states:
        raise ValueError(
            f"a decoded state is inside the state set; got {int(z.min())} to "
            f"{int(z.max())} for {n_states} states"
        )
    for run in np.split(z, np.nonzero(np.diff(z))[0] + 1):
        totals[run[0]] += float(run.size)
        counts[run[0]] += 1.0
    return np.where(counts > 0.0, totals / np.maximum(counts, 1.0), 0.0)


def entropy(distribution: Sequence[float]) -> float:
    """
    The Shannon entropy of a distribution, in nats.

    Raises:
        ValueError: when no state is given, when the mass is not finite, or
            when it is negative or sums to nothing.
    """
    p = np.asarray(distribution, dtype=float).ravel()
    if p.size == 0:
        raise ValueError("an entropy is taken over at least one state; got 0")
    if not np.all(np.isfinite(p)):
        raise ValueError(f"a distribution is finite; got {p[~np.isfinite(p)][0]}")
    total = float(p.sum())
    if float(p.min()) < 0.0 or total <= 0.0:
        raise ValueError(
            f"a distribution is non-negative and carries mass; got sum {total} "
            f"and smallest entry {float(p.min())}"
        )
    mass = p[p > 0.0] / total
    return float(-np.sum(mass * np.log(mass)))


@dataclass(frozen=True)
class RegimeFit:
    """What the regime row reports on one series."""

    model: PoissonHMM
    path: np.ndarray
    bic_by_state: Tuple[Tuple[int, float], ...]
    stationary: np.ndarray
    expected_dwell: np.ndarray
    empirical_dwell: np.ndarray
    entropy: float

    @property
    def n_states(self) -> int:
        return self.model.n_states


def fit_regimes(
    values: Sequence[float], state_counts: Sequence[int] = STATE_COUNTS, seed: int = 0
) -> RegimeFit:
    """
    Fit the regime structure of one series and report everything the row reads.

    Raises:
        ValueError: when the series holds no epoch, which carries no regime.
    """
    x = _counts(values)
    model, scores = select_states(x, state_counts, seed)
    path = decode(x, model)
    stationary = stationary_distribution(model.transition)
    return RegimeFit(
        model=model,
        path=path,
        bic_by_state=scores,
        stationary=stationary,
        expected_dwell=expected_dwell(model.transition),
        empirical_dwell=empirical_dwell(path, model.n_states),
        entropy=entropy(stationary),
    )


# ---------------------------------------------------------------------------
# How two fits are compared
# ---------------------------------------------------------------------------


def total_variation(p: Sequence[float], q: Sequence[float]) -> float:
    """
    Half the summed absolute gap between two stationary distributions.

    States are labelled by ascending rate, so two fits of different state counts
    are compared by giving the shorter one no mass on the states it lacks.

    Raises:
        ValueError: when neither side holds a state.
    """
    a = np.asarray(p, dtype=float).ravel()
    b = np.asarray(q, dtype=float).ravel()
    width = max(a.size, b.size)
    if width == 0:
        raise ValueError("a total variation is taken over at least one state; got 0")
    left = np.zeros(width, dtype=float)
    right = np.zeros(width, dtype=float)
    left[: a.size] = a
    right[: b.size] = b
    return float(0.5 * np.abs(left - right).sum())


def match_rates(
    a: Sequence[float], b: Sequence[float]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pair two fitted rate sets by the assignment minimizing the summed rate gap.

    Returns:
        The matched index pairs, and the absolute rate gap of each pair.  The
        larger set leaves its unmatched states out.

    Raises:
        ValueError: when either side holds no state.
    """
    left = np.asarray(a, dtype=float).ravel()
    right = np.asarray(b, dtype=float).ravel()
    if left.size == 0 or right.size == 0:
        raise ValueError(
            f"a rate matching needs a state on each side; got {left.size} and "
            f"{right.size}"
        )
    cost = np.abs(left[:, None] - right[None, :])
    rows, columns = linear_sum_assignment(cost)
    return np.column_stack([rows, columns]), cost[rows, columns]


def _choose_two(counts: np.ndarray) -> np.ndarray:
    """How many pairs each group of this size holds."""
    return counts * (counts - 1.0) / 2.0


def adjusted_rand(a: Sequence[int], b: Sequence[int]) -> float:
    """
    The adjusted Rand index of two labelings of the same epochs.

    Two labelings agreeing up to a relabelling score one, two independent ones
    score near zero, and two labelings each holding a single cluster agree
    trivially and score one.

    Raises:
        ValueError: when the two labelings are of different lengths, or when
            they hold no epoch.
    """
    left = np.asarray(a, dtype=np.int64).ravel()
    right = np.asarray(b, dtype=np.int64).ravel()
    if left.size != right.size:
        raise ValueError(
            f"an adjusted Rand index is taken on one epoch set; got {left.size} "
            f"and {right.size}"
        )
    if left.size == 0:
        raise ValueError("an adjusted Rand index needs at least one epoch; got 0")
    _, rows = np.unique(left, return_inverse=True)
    _, columns = np.unique(right, return_inverse=True)
    table = np.zeros((int(rows.max()) + 1, int(columns.max()) + 1), dtype=float)
    np.add.at(table, (rows, columns), 1.0)
    pairs = float(_choose_two(table).sum())
    by_row = float(_choose_two(table.sum(axis=1)).sum())
    by_column = float(_choose_two(table.sum(axis=0)).sum())
    total = float(_choose_two(np.array([float(left.size)]))[0])
    expected = by_row * by_column / total if total > 0.0 else 0.0
    largest = 0.5 * (by_row + by_column)
    if largest == expected:
        return 1.0
    return float((pairs - expected) / (largest - expected))


def cross_scale_coupling(
    fine_path: Sequence[int], coarse_path: Sequence[int], parent: Sequence[int]
) -> float:
    """
    The adjusted Rand index between a fine path and its parent's path.

    The coarse path is broadcast down to the fine epochs through the parent map,
    so the coupling reads whether the fine scale changes regime where its parent
    does.

    Raises:
        ValueError: when the parent map does not hold one entry per fine epoch,
            or when it points outside the coarse path.
    """
    fine = np.asarray(fine_path, dtype=np.int64).ravel()
    coarse = np.asarray(coarse_path, dtype=np.int64).ravel()
    index = np.asarray(parent, dtype=np.int64).ravel()
    if index.size != fine.size:
        raise ValueError(
            f"a parent map holds one entry per fine epoch; got {index.size} for "
            f"{fine.size} epochs"
        )
    if index.size and (int(index.min()) < 0 or int(index.max()) >= coarse.size):
        raise ValueError(
            f"a parent epoch is inside the coarse path; got {int(index.min())} to "
            f"{int(index.max())} for {coarse.size} epochs"
        )
    return adjusted_rand(fine, coarse[index])


def regime_coupling(values: Sequence[float], period: float) -> float:
    """
    How far one series' regime path follows the path of the scale above it.

    The series is folded into parent blocks of one calendar period and the block
    totals are the parent's own count series, so both paths are decoded from
    counts by the same fit and the adjusted Rand index between them is the
    coupling B5 reports.  A series carrying fewer than `MIN_CYCLES` parent blocks
    holds no coupling to read and scores zero.

    The parent series is admitted `MIN_EPOCHS_PER_STATE` epochs per state like
    any other, so a series holding under twenty parent blocks leaves the parent
    at one regime.  The coupling is then one where the fine path also holds one
    regime and zero where it holds more, which is a coarser reading of the same
    question and is what the day scale of an eight-week window supports.

    Args:
        values: one participant's count series at one scale.
        period: how many epochs of that scale one parent block holds.

    Raises:
        ValueError: when the parent block is under two epochs or undefined,
            since a block of one epoch is the series itself.
    """
    if not math.isfinite(period) or period < 2.0:
        raise ValueError(f"a parent block holds at least two epochs; got {period}")
    x = _counts(values)
    step = int(period)
    n_parents = x.size // step
    if n_parents < MIN_CYCLES:
        return 0.0
    fine = x[: n_parents * step]
    coarse = fine.reshape(n_parents, step).sum(axis=1)
    parents = np.repeat(np.arange(n_parents, dtype=np.int64), step)
    return cross_scale_coupling(
        fit_regimes(fine).path, fit_regimes(coarse).path, parents
    )


@dataclass(frozen=True)
class RegimeMatch:
    """The five figures the regime row reports on one pair of series."""

    state_difference: int
    stationary_tv: float
    rate_gap: float
    entropy_difference: float
    dwell_difference: float

    def as_row(self) -> Dict[str, float]:
        return {
            "state_difference": self.state_difference,
            "stationary_tv": self.stationary_tv,
            "rate_gap": self.rate_gap,
            "entropy_difference": self.entropy_difference,
            "dwell_difference": self.dwell_difference,
        }


def compare_regimes(
    real: Sequence[float],
    synthetic: Sequence[float],
    state_counts: Sequence[int] = STATE_COUNTS,
    seed: int = 0,
) -> RegimeMatch:
    """Fit both series and compare the two regime sets they carry."""
    left = fit_regimes(real, state_counts, seed)
    right = fit_regimes(synthetic, state_counts, seed)
    return RegimeMatch(
        state_difference=abs(left.n_states - right.n_states),
        stationary_tv=total_variation(left.stationary, right.stationary),
        rate_gap=float(match_rates(left.model.rates, right.model.rates)[1].mean()),
        entropy_difference=abs(left.entropy - right.entropy),
        dwell_difference=abs(
            float(left.empirical_dwell.mean()) - float(right.empirical_dwell.mean())
        ),
    )


def regime_entropy(values: Sequence[float], period: float) -> float:
    """
    The entropy of the fitted stationary distribution, in nats.

    A regime is a level held across cycles, so a series carrying fewer than
    `MIN_CYCLES` calendar cycles is reported as one regime and scores zero.  A
    one-state fit scores zero as well, since one regime carries no choice.

    Args:
        values: one participant's count series at one scale.
        period: the calendar period of that scale, in epochs.

    Raises:
        ValueError: when the calendar period is not positive.
    """
    if period <= 0.0:
        raise ValueError(f"a calendar period is positive; got {period}")
    x = _counts(values)
    if x.size < MIN_CYCLES * period:
        return 0.0
    return float(fit_regimes(x).entropy)
