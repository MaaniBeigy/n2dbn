"""
evaluation/nonstationarity.py
=============================
The enumerated grid on which a real and a synthetic log are compared.

Nothing here is declared in advance about the log.  Periods are discovered,
trend shape is read with a statistic that admits a rise and a fall, and each
magnitude is estimated per participant so a synthetic log handing every
participant the cohort-mean profile fails on the spread.  A cell is one form,
one series and one scale, and it carries its magnitude, its denominator and the
clause that decided whether it is tested or descriptive.

Three rules make the panel evidence.

The inclusion rule partitions the grid on the real log alone, before a synthetic
log exists, so a cell cannot move into the tested set once a result is visible.
It has two clauses about what a cell can support and three about what it holds.
A cell that cannot see the perturbation its row exists to catch is underpowered,
and a cell whose tolerance is narrower than the spread of its own statistic
cannot claim equivalence at this roster size, since it would fail on two draws of
one process.
The equivalence claim is a conjunction of per-cell equivalences, so each two
one-sided test runs at its own level with no correction and the claim holds only
when every tested cell passes; the difference direction is controlled inside
each form family instead.  And the injection calibration reports the smallest
effect each check catches, which is what separates a silent check from an
underpowered one.

Nineteen of the twenty-one forms are computed, here and in the modules each row
waits on: `segmentation` for the three changepoint rows, `trend_shape`,
`seasonality_drift`, `persistence`, `count_hmm`, `placement`, `omnibus` and
`factor_model` for the rest.  The two that remain are declared with what each
waits on, since a grid that quietly omits a row reads as coverage it does not
have.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import signal, stats

MARGINAL = "A"
JOINT = "B"

IMPLEMENTED = "implemented"
DECLARED = "declared"

TESTED = "tested"
DESCRIPTIVE = "descriptive"

#: At least this many episodes and carrier participants for a cell to be tested.
MIN_EPISODES = 200
MIN_CARRIERS = 20
MIN_EPOCHS = 30
MIN_CYCLES = 2

#: Standard errors of its own statistic a cell's tolerance has to exceed before
#: it can claim equivalence.  The contrast is between two groups of participants,
#: so the difference of their means carries about sqrt(2) times the spread of one
#: of them, and a one-sided t test at 0.05 on a few dozen participants needs
#: about 1.7 standard errors, so the bar is their product.  A tolerance under it
#: is narrower than the noise it is meant to rule out, and the cell fails
#: whatever the two logs hold.
#:
#: The spread this prices is the one between participants.  A comparison that
#: paired each participant with a draw taken at that participant's own fitted
#: level was measured on the panel and cancelled none of it: the two standard
#: errors stood at a median ratio of 0.98 over the 224 cells carrying a
#: tolerance, and no cell this clause drops would have been tested on the paired
#: difference.
PRECISION_FACTOR = 2.4


@dataclass(frozen=True)
class GridForm:
    """One row of the catalogue, and whether this module computes it."""

    id: str
    layer: str
    form: str
    status: str
    needs: str = ""


#: The catalogue.  A declared row names what it waits on, so the omission is
#: readable beside the rows that run.
GRID: Tuple[GridForm, ...] = (
    GridForm("A01", MARGINAL, "monotone trend", IMPLEMENTED),
    GridForm("A02", MARGINAL, "trend shape, form-free", IMPLEMENTED),
    GridForm("A03", MARGINAL, "stochastic against deterministic level", IMPLEMENTED),
    GridForm("A04", MARGINAL, "scale-localized drift", IMPLEMENTED),
    GridForm("A05", MARGINAL, "period discovery", IMPLEMENTED),
    GridForm("A06", MARGINAL, "seasonal profile shape", IMPLEMENTED),
    GridForm("A07", MARGINAL, "evolving seasonality", IMPLEMENTED),
    GridForm("A08", MARGINAL, "level changepoints", IMPLEMENTED),
    GridForm("A09", MARGINAL, "dispersion changepoints", IMPLEMENTED),
    GridForm("A10", MARGINAL, "slope changepoints", IMPLEMENTED),
    GridForm("A11", MARGINAL, "persistence changepoints", IMPLEMENTED),
    GridForm("A12", MARGINAL, "zero rate and intermittency", IMPLEMENTED),
    GridForm("A13", MARGINAL, "dispersion and family drift", IMPLEMENTED),
    GridForm("A14", MARGINAL, "regime structure", IMPLEMENTED),
    GridForm("A15", MARGINAL, "within-epoch placement", IMPLEMENTED),
    GridForm("A16", MARGINAL, "omnibus, form-free", IMPLEMENTED),
    GridForm("B1", JOINT, "common changepoint", IMPLEMENTED),
    GridForm("B2", JOINT, "common factor", IMPLEMENTED),
    GridForm("B3", JOINT, "same-epoch dependence drift", IMPLEMENTED),
    GridForm("B4", JOINT, "lagged-structure drift", IMPLEMENTED),
    GridForm("B5", JOINT, "cross-scale regime coupling", IMPLEMENTED),
)


def implemented_forms() -> Tuple[str, ...]:
    """The rows this module computes."""
    return tuple(row.id for row in GRID if row.status == IMPLEMENTED)


# ---------------------------------------------------------------------------
# A01, monotone trend
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MannKendall:
    """The tie-corrected trend statistic of one series."""

    s: float
    z: float
    p_value: float


def mann_kendall(values: Sequence[float]) -> MannKendall:
    """The tie-corrected Mann-Kendall statistic and its normal approximation."""
    x = np.asarray(values, dtype=float)
    n = x.size
    if n < 3:
        return MannKendall(0.0, 0.0, 1.0)
    # Only the pairs with i < j: the full matrix is antisymmetric and sums to
    # zero whatever the series does.
    s = float(np.triu(np.sign(x[None, :] - x[:, None]), k=1).sum())
    _, ties = np.unique(x, return_counts=True)
    correction = float(np.sum(ties * (ties - 1) * (2 * ties + 5)))
    variance = (n * (n - 1) * (2 * n + 5) - correction) / 18.0
    if variance <= 0.0:
        return MannKendall(s, 0.0, 1.0)
    z = (s - np.sign(s)) / math.sqrt(variance)
    return MannKendall(s, float(z), float(2.0 * stats.norm.sf(abs(z))))


def theil_sen(values: Sequence[float]) -> float:
    """The median pairwise slope of one series against its own index."""
    x = np.asarray(values, dtype=float)
    n = x.size
    if n < 2:
        return 0.0
    i, j = np.triu_indices(n, k=1)
    return float(np.median((x[j] - x[i]) / (j - i)))


# ---------------------------------------------------------------------------
# A04, scale-localized drift
# ---------------------------------------------------------------------------


def haar_detail_energy(values: Sequence[float], levels: int) -> np.ndarray:
    """
    The energy the block decomposition puts in each level's detail.

    A level shift is a large detail coefficient and folding is a low-pass
    filter, so one scan compares trend, changepoints and band ownership at once.
    """
    x = np.asarray(values, dtype=float)
    out = np.zeros(levels, dtype=float)
    for level in range(levels):
        if x.size < 2:
            break
        usable = x[: x.size - (x.size % 2)]
        pairs = usable.reshape(-1, 2)
        detail = (pairs[:, 0] - pairs[:, 1]) / 2.0
        out[level] = float(detail @ detail)
        x = pairs.mean(axis=1)
    return out


def detail_profile_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """The relative distance between two log detail-energy profiles."""
    floor = 1e-12
    la = np.log(np.asarray(a, dtype=float) + floor)
    lb = np.log(np.asarray(b, dtype=float) + floor)
    scale = float(np.sqrt(la @ la)) or 1.0
    return float(np.sqrt(np.sum((la - lb) ** 2)) / scale)


# ---------------------------------------------------------------------------
# A05 and A06, period discovery and the profile at each period
# ---------------------------------------------------------------------------


def periodogram(values: Sequence[float], periods: Sequence[float]) -> np.ndarray:
    """Normalized Lomb-Scargle power at each period, on the detrended series."""
    x = np.asarray(values, dtype=float)
    t = np.arange(x.size, dtype=float)
    if x.size < 2:
        return np.zeros(len(periods), dtype=float)
    slope, intercept = np.polyfit(t, x, 1)
    residual = x - (slope * t + intercept)
    # A series with no contrast leaves a residual of rounding error, and the
    # normalization below would turn that into power of order one.
    if float(residual.var()) <= 1e-12 * max(float(x.var()), 1.0):
        return np.zeros(len(periods), dtype=float)
    freqs = 2.0 * np.pi / np.asarray(periods, dtype=float)
    # A single frequency comes back zero-dimensional, and every caller indexes
    # the result by candidate.
    return np.atleast_1d(signal.lombscargle(t, residual, freqs, normalize=True))


def period_significance(
    values: Sequence[float],
    periods: Sequence[float],
    n_permutations: int = 199,
    seed: int = 0,
) -> Tuple[float, float]:
    """
    The largest normalized power over a candidate grid, and how often a reshuffle
    reaches it.

    The exceedance formula of an iterated Fisher test assumes the periodogram
    ordinates are independent, which holds on the Fourier grid of a series and
    not on a declared candidate grid.  A reshuffle of the same values keeps the
    marginal, destroys the ordering and needs no such assumption.
    """
    x = np.asarray(values, dtype=float)
    power = periodogram(x, periods)
    if power.size == 0:
        return 0.0, 1.0
    stat = float(power.max())
    if stat <= 0.0:
        return 0.0, 1.0
    rng = np.random.default_rng(seed)
    null = np.array(
        [
            float(periodogram(rng.permutation(x), periods).max())
            for _ in range(n_permutations)
        ]
    )
    return stat, float((1 + np.sum(null >= stat)) / (1 + n_permutations))


def harmonic(values: Sequence[float], period: float) -> Tuple[float, float]:
    """The amplitude and phase of one harmonic fitted to a series."""
    x = np.asarray(values, dtype=float)
    t = np.arange(x.size, dtype=float)
    design = np.column_stack(
        [
            np.ones(x.size),
            np.cos(2.0 * np.pi * t / period),
            np.sin(2.0 * np.pi * t / period),
        ]
    )
    beta, *_ = np.linalg.lstsq(design, x, rcond=None)
    return float(math.hypot(beta[1], beta[2])), float(math.atan2(beta[2], beta[1]))


def significant_periods(
    values: Sequence[float],
    periods: Sequence[float],
    alpha: float = 0.05,
    n_permutations: int = 199,
    seed: int = 0,
) -> Tuple[float, ...]:
    """
    The periods whose power survives an iterated reshuffle null.

    Each accepted period is removed by fitting and subtracting its harmonic, and
    the test is taken again on what is left.  The iteration examines each
    candidate at most once, so the level is divided by the size of the grid.
    """
    x = np.asarray(values, dtype=float).copy()
    if not len(periods):
        return ()
    level = alpha / len(periods)
    found: List[float] = []
    for _ in range(len(periods)):
        power = periodogram(x, periods)
        if period_significance(x, periods, n_permutations, seed)[1] > level:
            break
        best = float(periods[int(np.argmax(power))])
        if best in found:
            break
        found.append(best)
        amplitude, phase = harmonic(x, best)
        t = np.arange(x.size, dtype=float)
        x = x - amplitude * np.cos(2.0 * np.pi * t / best - phase)
    return tuple(found)


def circular_distance(a: float, b: float) -> float:
    """The shortest angular distance between two phases."""
    return float(abs((a - b + math.pi) % (2.0 * math.pi) - math.pi))


def jaccard(a: Sequence[float], b: Sequence[float]) -> float:
    """Overlap of two discovered sets, one when both are empty."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


# ---------------------------------------------------------------------------
# A12 and A13, intermittency and dispersion
# ---------------------------------------------------------------------------


def _windows(n: int, window: int) -> List[slice]:
    """Consecutive non-overlapping windows that fit inside a series."""
    if window < 2:
        raise ValueError(f"a window holds at least two epochs; got {window}")
    return [slice(lo, lo + window) for lo in range(0, n - window + 1, window)]


def zero_rate(values: Sequence[float], window: int) -> np.ndarray:
    """The share of zero epochs inside each window."""
    x = np.asarray(values, dtype=float)
    return np.array([float((x[w] == 0.0).mean()) for w in _windows(x.size, window)])


def run_lengths(values: Sequence[float], zero: bool = True) -> np.ndarray:
    """Lengths of the maximal runs of zero or of nonzero epochs."""
    flags = (np.asarray(values, dtype=float) == 0.0) == bool(zero)
    edges = np.diff(np.concatenate(([0], flags.astype(np.int8), [0])))
    return np.nonzero(edges == -1)[0] - np.nonzero(edges == 1)[0]


def variance_mean_ratio(values: Sequence[float], window: int) -> np.ndarray:
    """The variance-to-mean ratio inside each window, zero where the mean is zero."""
    x = np.asarray(values, dtype=float)
    out = []
    for w in _windows(x.size, window):
        block = x[w]
        mean = float(block.mean())
        out.append(float(block.var() / mean) if mean > 0.0 else 0.0)
    return np.array(out, dtype=float)


# ---------------------------------------------------------------------------
# B3, same-epoch dependence drift
# ---------------------------------------------------------------------------


def windowed_correlation(matrix: np.ndarray, window: int) -> np.ndarray:
    """
    The correlation matrix of each window, centred inside that window.

    Centring inside the window is what stops a level change from arriving as a
    change in dependence.
    """
    values = np.asarray(matrix, dtype=float)
    out = []
    for w in _windows(values.shape[0], window):
        block = values[w] - values[w].mean(axis=0)
        spread = np.sqrt(np.sum(block * block, axis=0))
        spread = np.where(spread > 0.0, spread, np.inf)
        out.append((block.T @ block) / np.outer(spread, spread))
    return np.array(out, dtype=float)


def lagged_correlation(matrix: np.ndarray, window: int, lag: int = 1) -> np.ndarray:
    """
    The lagged cross-correlation matrix of each window, centred inside that window.

    Entry (i, j) holds column i at an epoch against column j at the epoch `lag`
    later, so the matrix is asymmetric and its diagonal carries each column's own
    autocorrelation.  B3 reads the same-epoch cross-section and this is what B4
    reads beside it, so a dependence that moves across the lag shows up here and
    leaves the same-epoch matrix where it was.

    Raises:
        ValueError: when the lag is under one epoch, or the window holds no pair
            at that lag.
    """
    if lag < 1:
        raise ValueError(
            f"a lagged correlation needs a lag of one epoch or more; got {lag}"
        )
    if window <= lag:
        raise ValueError(f"a window of {window} epochs holds no pair at lag {lag}")
    values = np.asarray(matrix, dtype=float)
    out = []
    for w in _windows(values.shape[0], window):
        block = values[w]
        past = block[:-lag] - block[:-lag].mean(axis=0)
        future = block[lag:] - block[lag:].mean(axis=0)
        left = np.sqrt(np.sum(past * past, axis=0))
        right = np.sqrt(np.sum(future * future, axis=0))
        left = np.where(left > 0.0, left, np.inf)
        right = np.where(right > 0.0, right, np.inf)
        out.append((past.T @ future) / np.outer(left, right))
    return np.array(out, dtype=float)


def dependence_distance(a: np.ndarray, b: np.ndarray) -> float:
    """The root mean squared difference between two correlation trajectories."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(a.shape[0], b.shape[0])
    if n == 0:
        return 0.0
    return float(np.sqrt(np.mean((a[:n] - b[:n]) ** 2)))


# ---------------------------------------------------------------------------
# How a cell is compared
# ---------------------------------------------------------------------------


def sign_flip_p(
    differences: Sequence[float], n_flips: int = 999, seed: int = 0
) -> float:
    """
    The paired sign-flip p-value of a per-participant difference.

    The participant is the sampling unit, so the null flips the sign of each
    participant's whole difference and nothing inside it.
    """
    d = np.asarray(differences, dtype=float)
    if d.size == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    stat = abs(float(d.mean()))
    signs = rng.choice((-1.0, 1.0), size=(n_flips, d.size))
    null = np.abs((signs * d).mean(axis=1))
    return float((1 + np.sum(null >= stat)) / (1 + n_flips))


@dataclass(frozen=True)
class Equivalence:
    """A two one-sided test of one cell against its declared tolerance."""

    difference: float
    tolerance: float
    p_lower: float
    p_upper: float

    @property
    def p_value(self) -> float:
        return max(self.p_lower, self.p_upper)

    def equivalent(self, alpha: float = 0.05) -> bool:
        return self.p_value <= alpha


def tost(differences: Sequence[float], tolerance: float) -> Equivalence:
    """
    Test whether a per-participant difference sits inside a declared tolerance.

    Raises:
        ValueError: when the tolerance is not positive, which would make every
            cell fail for a reason unrelated to the log.
    """
    if tolerance <= 0.0:
        raise ValueError(f"an equivalence tolerance is positive; got {tolerance}")
    d = np.asarray(differences, dtype=float)
    n = d.size
    if n < 2:
        return Equivalence(float(d.mean()) if n else 0.0, tolerance, 1.0, 1.0)
    mean = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(n))
    if se <= 0.0:
        edge = 0.0 if abs(mean) < tolerance else 1.0
        return Equivalence(mean, tolerance, edge, edge)
    dof = n - 1
    return Equivalence(
        difference=mean,
        tolerance=tolerance,
        p_lower=float(stats.t.sf((mean + tolerance) / se, dof)),
        p_upper=float(stats.t.cdf((mean - tolerance) / se, dof)),
    )


def group_tost(
    real: Sequence[float],
    synthetic: Sequence[float],
    tolerance: float,
    synthetic_units: Optional[int] = None,
) -> Equivalence:
    """
    Test whether two groups' means sit inside a declared tolerance of each other.

    The contrast is between a group of real participants and a group of drawn
    ones, so the two samples are independent and the standard error is Welch's,
    which asks neither group for the other's spread.  A group of one carries no
    spread of its own and the pair reports no equivalence.

    `synthetic_units` is how many independent units stand behind `synthetic`,
    when it holds several draws over the same ones.  `panel_compare.compare`
    pools R replicates of one panel into a single drawn group, so five draws
    over the same 51 windows and masks arrive as 255 values; charging Welch
    `vb / 255` understates the standard error and the error runs in the
    direction that makes a cell easier to pass.  Counting the panel rather than
    the draw is the conservative reading, and it is what the sampling unit
    actually is.

    Raises:
        ValueError: when the tolerance is not positive, which would make every
            cell fail for a reason unrelated to the log; or when
            `synthetic_units` exceeds the values offered.
    """
    if tolerance <= 0.0:
        raise ValueError(f"an equivalence tolerance is positive; got {tolerance}")
    a = np.asarray(real, dtype=float)
    b = np.asarray(synthetic, dtype=float)
    units = int(b.size if synthetic_units is None else synthetic_units)
    if units > b.size:
        raise ValueError(
            f"{units} independent units claimed behind {b.size} synthetic values"
        )
    mean = float(a.mean()) - float(b.mean()) if a.size and b.size else 0.0
    if a.size < 2 or b.size < 2 or units < 2:
        return Equivalence(mean, tolerance, 1.0, 1.0)
    va, vb = float(a.var(ddof=1)) / a.size, float(b.var(ddof=1)) / units
    se = math.sqrt(va + vb)
    if se <= 0.0:
        edge = 0.0 if abs(mean) < tolerance else 1.0
        return Equivalence(mean, tolerance, edge, edge)
    dof = (va + vb) ** 2 / (va**2 / (a.size - 1) + vb**2 / (units - 1))
    return Equivalence(
        difference=mean,
        tolerance=tolerance,
        p_lower=float(stats.t.sf((mean + tolerance) / se, dof)),
        p_upper=float(stats.t.cdf((mean - tolerance) / se, dof)),
    )


def hedges_g(real: Sequence[float], synthetic: Sequence[float]) -> float:
    """
    The independent-groups standardized mean difference, real minus synthetic.

    The two groups are pooled for the denominator and the small-sample factor of
    Hedges is applied, so this is the $g$ of two groups and never the paired
    $g_z$.  Two groups that between them carry no spread report zero, since a
    standardized difference has no unit there.
    """
    a = np.asarray(real, dtype=float)
    b = np.asarray(synthetic, dtype=float)
    if a.size < 2 or b.size < 2:
        return 0.0
    dof = a.size + b.size - 2
    pooled = math.sqrt(
        ((a.size - 1) * float(a.var(ddof=1)) + (b.size - 1) * float(b.var(ddof=1)))
        / dof
    )
    if pooled <= 0.0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled * (1.0 - 3.0 / (4.0 * dof - 1.0)))


def group_permutation_p(
    real: Sequence[float],
    synthetic: Sequence[float],
    n_draws: int = 999,
    seed: int = 0,
) -> float:
    """
    The two-sided p-value of a group-label permutation over participants.

    The participant is the sampling unit, so the null moves whole participants
    between the two groups and nothing inside one.

    Raises:
        ValueError: when either group is empty, or no draw is asked for.
    """
    a = np.asarray(real, dtype=float)
    b = np.asarray(synthetic, dtype=float)
    if a.size == 0 or b.size == 0:
        raise ValueError(
            f"a two-group null needs both groups; got {a.size} and {b.size}"
        )
    if n_draws < 1:
        raise ValueError(f"a permutation null needs at least one draw; got {n_draws}")
    pool = np.concatenate([a, b])
    observed = abs(float(a.mean()) - float(b.mean()))
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_draws):
        moved = rng.permutation(pool)
        if (
            abs(float(moved[: a.size].mean()) - float(moved[a.size :].mean()))
            >= observed
        ):
            hits += 1
    return (1 + hits) / (1 + n_draws)


def benjamini_hochberg(p_values: Sequence[float], q: float = 0.05) -> np.ndarray:
    """Which p-values of one form family are rejected at a false discovery rate."""
    p = np.asarray(p_values, dtype=float)
    n = p.size
    out = np.zeros(n, dtype=bool)
    if n == 0:
        return out
    order = np.argsort(p)
    thresholds = q * (np.arange(1, n + 1) / n)
    passing = np.nonzero(p[order] <= thresholds)[0]
    if passing.size:
        out[order[: passing[-1] + 1]] = True
    return out


# ---------------------------------------------------------------------------
# The inclusion rule
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Denominator:
    """What one cell has to work with, measured on the real log alone."""

    episodes: int
    carriers: int
    epochs_per_participant: int
    cycles: float


@dataclass(frozen=True)
class Inclusion:
    """Whether a cell is tested, and which clause decided it."""

    status: str
    clause: str

    @property
    def tested(self) -> bool:
        return self.status == TESTED


def inclusion(
    denominator: Denominator,
    minimum_detectable_effect: float,
    bound: float,
    needs_period: bool = False,
    tolerance: Optional[float] = None,
    standard_error: Optional[float] = None,
) -> Inclusion:
    """
    Partition one cell into tested or descriptive, on the real log alone.

    Args:
        tolerance: what the cell forgives, in the units of its own statistic.
        standard_error: the spread of that statistic between participants,
            divided by the square root of their count, which is what a group mean
            carries.  Given both, a cell whose tolerance is narrower than
            `PRECISION_FACTOR` standard errors is descriptive: it would fail on
            the spread of two groups drawn from one population and report nothing
            about the logs.

    Raises:
        ValueError: when the per-form bound is not positive, since a cell could
            then never clear the power clause for a readable reason.
    """
    if bound <= 0.0:
        raise ValueError(f"a per-form bound is positive; got {bound}")
    if denominator.episodes < MIN_EPISODES or denominator.carriers < MIN_CARRIERS:
        return Inclusion(DESCRIPTIVE, "denominator")
    if denominator.epochs_per_participant < MIN_EPOCHS:
        return Inclusion(DESCRIPTIVE, "epochs")
    if needs_period and denominator.cycles < MIN_CYCLES:
        return Inclusion(DESCRIPTIVE, "definedness")
    if minimum_detectable_effect > bound:
        return Inclusion(DESCRIPTIVE, "power")
    if tolerance is not None and standard_error is not None:
        if tolerance <= PRECISION_FACTOR * standard_error:
            return Inclusion(DESCRIPTIVE, "precision")
    return Inclusion(TESTED, "")


# ---------------------------------------------------------------------------
# The injection calibration
# ---------------------------------------------------------------------------

#: `slope` tilts the whole series and `slope_break` tilts its second half alone.
#: The two are different perturbations: a segmentation on a linear cost fits a
#: line inside each segment, so a tilt of the whole series moves every segment
#: equally and buys no cut, and only a break in the slope does.
INJECTIONS = (
    "level_step",
    "slope",
    "slope_break",
    "dispersion",
    "phase",
    "zero_rate",
    "correlation",
)


def inject(
    values: Sequence[float],
    kind: str,
    size: float,
    period: float = 24.0,
    seed: int = 0,
    blocks: Optional[Sequence[Tuple[int, int]]] = None,
) -> np.ndarray:
    """
    Put a known perturbation of a stated size into a series.

    Args:
        blocks: the half-open range of each participant, so the perturbation
            lands inside every one of them the way a real one would.

    Raises:
        ValueError: when the perturbation is not one of `INJECTIONS`.
    """
    if kind not in INJECTIONS:
        raise ValueError(f"injection {kind!r}; expected one of {list(INJECTIONS)}")
    if blocks is not None:
        out = np.asarray(values, dtype=float).copy()
        for start, stop in blocks:
            out[start:stop] = inject(out[start:stop], kind, size, period, seed)
        return out
    x = np.asarray(values, dtype=float).copy()
    n = x.size
    t = np.arange(n, dtype=float)
    rng = np.random.default_rng(seed)
    level = float(np.mean(x)) or 1.0
    if kind == "level_step":
        x[n // 2 :] += size * level
    elif kind == "slope":
        x += size * level * (t / max(n - 1, 1))
    elif kind == "slope_break":
        half = n // 2
        x[half:] += size * level * ((t[half:] - t[half]) / max(n - 1, 1))
    elif kind == "dispersion":
        x = level + (x - level) * (1.0 + size * (t / max(n - 1, 1)))
    elif kind == "phase":
        x = np.interp((t - size * period) % n, t, x)
    elif kind == "zero_rate":
        x = np.where(rng.random(n) < size * (t / max(n - 1, 1)), 0.0, x)
    else:
        noise = rng.normal(size=n)
        x[n // 2 :] = (1.0 - size) * x[n // 2 :] + size * level * noise[n // 2 :]
    return x


def minimum_detectable(
    values: Sequence[float],
    kind: str,
    detector: Callable[[np.ndarray, np.ndarray], float],
    sizes: Sequence[float],
    alpha: float = 0.05,
    period: float = 24.0,
    seed: int = 0,
    blocks: Optional[Sequence[Tuple[int, int]]] = None,
) -> float:
    """
    The smallest injected size at which the detector is active.

    Returns:
        The first size whose p-value clears the level, or infinity when no size
        in the grid does, which is what a check that never is active reports.
    """
    base = np.asarray(values, dtype=float)
    for size in sorted(sizes):
        moved = inject(base, kind, size, period=period, seed=seed, blocks=blocks)
        if detector(base, moved) <= alpha:
            return float(size)
    return math.inf


# ---------------------------------------------------------------------------
# The negative controls
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NegativeControl:
    """One control, and the rows it has to break."""

    name: str
    must_fail: Tuple[str, ...]
    must_pass: Tuple[str, ...]


CONTROLS: Tuple[NegativeControl, ...] = (
    NegativeControl("no regime", ("A08", "A09", "A11", "A14", "B1", "B5"), ()),
    NegativeControl("calendar nodes removed", ("A05", "A06", "A07"), ()),
    NegativeControl(
        "label-shuffled cross-section",
        ("B3", "B4"),
        tuple(row.id for row in GRID if row.layer == MARGINAL),
    ),
    NegativeControl(
        "within-participant time shuffle",
        tuple(row.id for row in GRID if row.layer == MARGINAL),
        (),
    ),
)


def shuffle_within_participant(
    values: Sequence[float], who: Sequence[int], seed: int = 0
) -> np.ndarray:
    """Reorder each participant's epochs, which keeps the marginal histogram."""
    values = np.asarray(values, dtype=float)
    who = np.asarray(who, dtype=np.int64)
    rng = np.random.default_rng(seed)
    out = values.copy()
    for unit in np.unique(who):
        rows = np.nonzero(who == unit)[0]
        out[rows] = values[rng.permutation(rows)]
    return out


def shuffle_labels(matrix: np.ndarray, seed: int = 0) -> np.ndarray:
    """
    Move each label's own column by its own offset, which breaks the cross-section.

    The order inside a column is kept, so every marginal series arrives with its
    histogram, its trend and its autocorrelation intact and only the alignment
    between labels is gone.  Reordering a column instead would break every Layer
    A form as well, and the control is stated to leave those alone.
    """
    values = np.asarray(matrix, dtype=float)
    n = values.shape[0]
    rng = np.random.default_rng(seed)
    out = values.copy()
    for j in range(values.shape[1]):
        out[:, j] = np.roll(values[:, j], int(rng.integers(1, max(n, 2))))
    return out


def control_verdict(
    control: NegativeControl,
    failed: Sequence[str],
    computed: Optional[Sequence[str]] = None,
    baseline: Sequence[str] = (),
) -> Tuple[bool, Dict[str, List[str]]]:
    """
    Whether one control broke the rows it names and left the others alone.

    A control has to show that a row can fail for its own reason, so it is
    graded on what it broke and on nothing else.  A row nobody tested cannot
    break.  A row already failing before the control was applied has been shown
    to fail by whatever broke it, and the control neither adds to that nor is
    answerable for it.  Both are reported apart from the rows the control was
    actually put to.

    Args:
        failed: the rows that fail under the control.
        computed: the rows the run tested at all; None takes every named row as
            tested.
        baseline: the rows already failing before the control was applied.

    Returns:
        Whether the control did its job, and the rows it missed, the rows it
        broke by accident, the rows nobody tested and the rows that were already
        failing.
    """
    already = set(baseline)
    broke = set(failed) - already
    named = set(control.must_fail)
    tested = named if computed is None else named & set(computed)
    reachable = tested - already
    missed = sorted(reachable - broke)
    spurious = sorted(broke & set(control.must_pass))
    return not missed and not spurious, {
        "missed": missed,
        "spurious": spurious,
        "untested": sorted(named - tested),
        "already_failing": sorted(tested & already),
    }


# ---------------------------------------------------------------------------
# What a cell reports
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """One form on one series at one scale, with what decided its standing."""

    form: str
    series: str
    scale: str
    magnitude: float
    spread: float
    p_value: float
    inclusion: Inclusion
    denominator: Denominator
    #: The independent-groups standardized mean difference, real minus drawn.
    effect: float = 0.0
    #: The group-label permutation null's two-sided p-value.  This is the
    #: difference test the equivalence test cannot stand in for: `p_value` small
    #: says "close enough", this small says "not the same".  1.0 where it was
    #: not computed.
    permutation_p: float = 1.0

    def as_row(self) -> Dict[str, object]:
        return {
            "form": self.form,
            "series": self.series,
            "scale": self.scale,
            "magnitude": self.magnitude,
            "spread": self.spread,
            "effect": self.effect,
            "p_value": self.p_value,
            "permutation_p": self.permutation_p,
            "status": self.inclusion.status,
            "clause": self.inclusion.clause,
            "episodes": self.denominator.episodes,
            "carriers": self.denominator.carriers,
        }


def panel_verdict(
    cells: Sequence[Cell], alpha: float = 0.05
) -> Tuple[bool, List[Cell]]:
    """
    Whether every tested cell passes, and which ones do not.

    The claim is a conjunction of per-cell equivalences, so each cell runs at its
    own level with no correction and one failure decides the verdict.
    """
    failing = [c for c in cells if c.inclusion.tested and c.p_value > alpha]
    return not failing, failing
