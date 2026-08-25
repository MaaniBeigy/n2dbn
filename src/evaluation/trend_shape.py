"""
evaluation/trend_shape.py
=========================
The shape of a trend, read without declaring its form, and whether the level it
moves around is stochastic.

Two rows of the nonstationarity grid wait here.  A02 asks how a series departs
from a straight line without naming the departure in advance, and A03 asks
whether the level wanders because a shock never dies or because a deterministic
path carries it.  Both are read on one series, at one scale, inside one
participant.

A penalized regression spline answers A02.  A cubic B-spline basis on a declared
number of uniform interior knots holds more coefficients than a trend needs, and
a second difference penalty on those coefficients buys the smoothness back, so
the fitted curve follows a rise and a fall of any shape and pays for every
wiggle it keeps.  Generalized cross validation picks the price over a declared
grid, so the series chooses its own smoothness.  The effective degrees of
freedom, the trace of the hat matrix, is what the curve spends; a straight line
spends two, and the F ratio of the drop in residual sum of squares per extra
degree of freedom is the test that the curve spends more.

A pair of tests answers A03, since either one alone leaves the reading open.  An
augmented Dickey-Fuller regression takes a unit root as its null and a KPSS
partial-sum statistic takes stationarity as its null.  A series that rejects the
first and holds the second is stationary; one that holds the first and rejects
the second at both a stationary level and a stationary trend carries a unit
root; one that holds the first, rejects a stationary level and holds a
stationary trend sits on a deterministic path.  The two tests can also disagree,
and that verdict is reported as its own class, since the series is then
informative about neither reading.

The scalar a grid cell reads is the fitted spline's mean squared second
derivative in calendar-period time, divided by the squared level of the series.
It is how far the trend bends over one calendar period, it is zero for a
straight line, and it is finite for every series the caller can hand it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import stats
from scipy.interpolate import BSpline

#: The spline is cubic, so its second derivative is piecewise linear and the
#: curvature integral below is exact under two-point Gauss-Legendre.
DEGREE = 3

#: Interior knots of the basis.  The count is declared, and the penalty is what
#: decides how smooth the fitted curve comes out.
KNOTS = 6

#: The smoothing prices generalized cross validation chooses among.
SMOOTHING_GRID: Tuple[float, ...] = tuple(
    float(10.0**power) for power in np.arange(-3.0, 9.5, 0.5)
)

#: What a straight line spends, which is the null of the shape test.
LINE_EDF = 2.0

#: A fit within this of a straight line has no extra degrees to test.
EDF_TOLERANCE = 1e-6

#: The share of a series a residual falls to before it is rounding.  It is
#: applied to norms, so a comparison of two sums of squares carries its square.
_FLOOR = 1e-10

#: Asymptotic quantiles of the Dickey-Fuller tau statistic for the
#: constant-only regression.  Source: Fuller (1976), Table 8.5.2, the n equals
#: infinity row of the tau-mu block; the 1, 5 and 10 percent entries agree with
#: MacKinnon (1991) at -3.4336, -2.8621 and -2.5671.
ADF_TAU_QUANTILES: Tuple[float, ...] = (
    0.01,
    0.025,
    0.05,
    0.10,
    0.50,
    0.90,
    0.95,
    0.975,
    0.99,
)
ADF_TAU_CRITICAL: Tuple[float, ...] = (
    -3.43,
    -3.12,
    -2.86,
    -2.57,
    -1.57,
    -0.44,
    -0.07,
    0.23,
    0.60,
)

#: Upper-tail critical values of the KPSS statistic.  Source: Kwiatkowski,
#: Phillips, Schmidt and Shin (1992), Table 1: the eta-mu row for a stationary
#: level and the eta-tau row for a stationary trend.
KPSS_P_VALUES: Tuple[float, ...] = (0.10, 0.05, 0.025, 0.01)
KPSS_LEVEL_CRITICAL: Tuple[float, ...] = (0.347, 0.463, 0.574, 0.739)
KPSS_TREND_CRITICAL: Tuple[float, ...] = (0.119, 0.146, 0.176, 0.216)

STATIONARY = "stationary"
UNIT_ROOT = "unit root"
TREND_STATIONARY = "trend stationary"
UNDETERMINED = "undetermined"
CLASSIFICATIONS: Tuple[str, ...] = (
    STATIONARY,
    UNIT_ROOT,
    TREND_STATIONARY,
    UNDETERMINED,
)


def _finite(values: Sequence[float], what: str) -> np.ndarray:
    """The series as a float array, refusing an entry no method can read."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"{what} reads one series laid flat; got {x.ndim} dimensions")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{what} reads finite values; got {x[~np.isfinite(x)][0]}")
    return x


def _no_residual(residual: float, series: float) -> bool:
    """
    Whether a residual sum of squares has fallen to rounding beside its series.

    Both arguments are sums of squares, so the share of the norm enters squared.
    Reading the residual against the series it came from is what keeps every
    verdict below unchanged when the whole series is rescaled.
    """
    return residual <= _FLOOR * _FLOOR * series


# ---------------------------------------------------------------------------
# A02, the penalized spline
# ---------------------------------------------------------------------------


def epochs_needed(n_knots: int = KNOTS) -> int:
    """How many epochs a basis on this many interior knots is fitted on."""
    if n_knots < 1:
        raise ValueError(f"a basis holds at least one interior knot; got {n_knots}")
    return 2 * (n_knots + DEGREE + 1)


def spline_knots(n: int, n_knots: int = KNOTS) -> np.ndarray:
    """
    The knot vector of a cubic basis on n epochs, uniformly spaced throughout.

    The spacing runs on past both ends of the series, which is what makes a
    second difference of the coefficients equal to a second derivative of the
    curve, so the penalty has a straight line in its null space exactly.

    Raises:
        ValueError: when the series is too short to span, since the knots then
            collapse onto one point and the basis has no support to stand on.
    """
    if n < 2:
        raise ValueError(f"a knot vector spans at least two epochs; got {n}")
    span = float(n - 1)
    # The interior knots carry the ends of the series exactly.  One linspace over
    # the padded range puts them at rounded positions, and the first knot of the
    # basis then sits a rounding error inside epoch zero, which the design matrix
    # reads as an epoch out of bounds.
    interior = np.linspace(0.0, span, n_knots + 2)
    step = span / (n_knots + 1)
    left = interior[0] - step * np.arange(DEGREE, 0, -1)
    right = interior[-1] + step * np.arange(1, DEGREE + 1)
    return np.concatenate([left, interior, right])


def spline_basis(n: int, n_knots: int = KNOTS) -> Tuple[np.ndarray, np.ndarray]:
    """The knot vector and the B-spline design matrix on epochs 0 to n - 1."""
    knots = spline_knots(n, n_knots)
    epochs = np.arange(n, dtype=float)
    return knots, BSpline.design_matrix(epochs, knots, DEGREE).toarray()


def _difference_penalty(size: int) -> np.ndarray:
    """The second difference penalty matrix on a coefficient vector."""
    differences = np.diff(np.eye(size), n=2, axis=0)
    return differences.T @ differences


@dataclass(frozen=True)
class SplineFit:
    """One penalized fit, and what it spent to get there."""

    fitted: np.ndarray
    coefficients: np.ndarray
    knots: np.ndarray
    smoothing: float
    edf: float
    rss: float
    gcv: float


def fit_spline(
    values: Sequence[float], smoothing: float, n_knots: int = KNOTS
) -> SplineFit:
    """
    Fit a cubic spline by penalized least squares at one smoothing price.

    Args:
        values: the series, at one scale and inside one participant.
        smoothing: the price of a second difference in the coefficients.
        n_knots: interior knots of the basis.

    Returns:
        The fitted values, the coefficients, the effective degrees of freedom
        and the generalized cross validation score at this price.

    Raises:
        ValueError: when the price is negative or is not itself finite, when the
            series is shorter than the basis is fitted on, or when an entry is
            not finite.
    """
    # A price that is not a number compares false against zero and would carry a
    # fit of nothing but nan all the way out to the grid.
    if not np.isfinite(smoothing) or smoothing < 0.0:
        raise ValueError(
            f"a smoothing price is finite and not negative; got {smoothing}"
        )
    needed = epochs_needed(n_knots)
    x = _finite(values, "a penalized spline")
    n = x.size
    if n < needed:
        raise ValueError(
            f"a spline on {n_knots} interior knots is fitted on {needed} epochs; "
            f"got {n}"
        )
    knots, basis = spline_basis(n, n_knots)
    normal = basis.T @ basis + float(smoothing) * _difference_penalty(basis.shape[1])
    coefficients = np.linalg.solve(normal, basis.T @ x)
    edf = float(np.sum(basis.T * np.linalg.solve(normal, basis.T)))
    fitted = basis @ coefficients
    residual = x - fitted
    rss = float(residual @ residual)
    return SplineFit(
        fitted=fitted,
        coefficients=coefficients,
        knots=knots,
        smoothing=float(smoothing),
        edf=edf,
        rss=rss,
        gcv=float(n * rss / (n - edf) ** 2),
    )


def select_smoothing(
    values: Sequence[float],
    smoothings: Sequence[float] = SMOOTHING_GRID,
    n_knots: int = KNOTS,
) -> SplineFit:
    """
    The fit whose generalized cross validation score is lowest on a declared grid.

    Raises:
        ValueError: when the grid is empty, since nothing could then be chosen.
    """
    if not len(smoothings):
        raise ValueError(f"a smoothing grid holds at least one price; got {smoothings}")
    fits = [fit_spline(values, price, n_knots) for price in smoothings]
    return min(fits, key=lambda fit: fit.gcv)


def curvature_energy(fit: SplineFit, period: float) -> float:
    """
    The mean squared second derivative of a fitted spline, in calendar-period time.

    One unit of the axis is one calendar period, so an hour-scale series and a
    day-scale series carry the same unit and the number reads as how far the
    trend bends over one period.

    Raises:
        ValueError: when the period is not positive, or is not itself finite.
    """
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError(f"a calendar period is positive and finite; got {period}")
    curve = BSpline(fit.knots, fit.coefficients, DEGREE).derivative(2)
    lo = float(fit.knots[DEGREE])
    hi = float(fit.knots[-DEGREE - 1])
    edges = np.unique(np.clip(fit.knots, lo, hi))
    nodes, weights = np.polynomial.legendre.leggauss(2)
    half = np.diff(edges)[:, None] / 2.0
    middle = ((edges[:-1] + edges[1:]) / 2.0)[:, None]
    integral = float(np.sum(half * weights * curve(middle + half * nodes) ** 2))
    return float(period**4 * integral / (hi - lo))


@dataclass(frozen=True)
class EdfTest:
    """The shape test: what the curve spent, and whether a line explains it."""

    edf: float
    statistic: float
    p_value: float
    rss_line: float
    rss_spline: float
    smoothing: float


def edf_test(
    values: Sequence[float],
    smoothings: Sequence[float] = SMOOTHING_GRID,
    n_knots: int = KNOTS,
) -> EdfTest:
    """
    Test a fitted spline against a straight line on the degrees of freedom it spent.

    Returns:
        The effective degrees of freedom, the F ratio of the drop in residual
        sum of squares per extra degree, and its p-value.
    """
    x = _finite(values, "a shape test")
    fit = select_smoothing(x, smoothings, n_knots)
    n = x.size
    design = np.column_stack([np.ones(n), np.arange(n, dtype=float)])
    line = design @ np.linalg.lstsq(design, x, rcond=None)[0]
    rss_line = float((x - line) @ (x - line))
    extra = fit.edf - LINE_EDF
    # A curve that reproduces the series leaves no residual to take a ratio on,
    # and a curve already at two degrees has no extra degree to test.
    if _no_residual(fit.rss, float(x @ x)) or extra <= EDF_TOLERANCE:
        return EdfTest(fit.edf, 0.0, 1.0, rss_line, fit.rss, fit.smoothing)
    denominator = n - fit.edf
    statistic = max(0.0, (rss_line - fit.rss) / extra / (fit.rss / denominator))
    return EdfTest(
        edf=fit.edf,
        statistic=statistic,
        p_value=float(stats.f.sf(statistic, extra, denominator)),
        rss_line=rss_line,
        rss_spline=fit.rss,
        smoothing=fit.smoothing,
    )


def spline_curvature(values: Sequence[float], period: float) -> float:
    """
    How far the fitted trend bends over one calendar period, per squared level.

    Args:
        values: one participant's series at one scale.
        period: the calendar period of that scale in epochs, 24 at the hour
            scale and 7 at the day scale.

    Returns:
        The mean squared second derivative of the selected spline in
        calendar-period time, divided by the squared mean absolute level.  A
        straight line, a constant series and a series shorter than the basis is
        fitted on all read 0.0.

    Raises:
        ValueError: when the period is not positive or is not itself finite, or
            when an entry is not finite.
    """
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError(f"a calendar period is positive and finite; got {period}")
    x = _finite(values, "a spline curvature")
    if x.size < epochs_needed():
        return 0.0
    level = float(np.mean(np.abs(x)))
    if level <= 0.0:
        return 0.0
    # The curve is fitted on the series divided by its level, so the ratio is
    # taken before anything is squared.  Squaring the level of a series held in
    # small units underflows to zero and the division then reads nan.
    scaled = x / level
    if float(scaled.var()) <= 0.0:
        return 0.0
    return curvature_energy(select_smoothing(scaled), period)


# ---------------------------------------------------------------------------
# A03, the unit-root pair
# ---------------------------------------------------------------------------


def lag_rule(n: int) -> int:
    """
    The declared lag order and Newey-West bandwidth at a series length.

    Raises:
        ValueError: when the length is negative, since the fourth root is then
            complex and the rule has nothing to round.
    """
    if n < 0:
        raise ValueError(f"a series length is not negative; got {n}")
    return int(4.0 * (n / 100.0) ** 0.25)


@dataclass(frozen=True)
class LevelTest:
    """One test of the level, with the lag order it was taken at."""

    statistic: float
    p_value: float
    lags: int


def adf_test(values: Sequence[float], lags: Optional[int] = None) -> LevelTest:
    """
    The augmented Dickey-Fuller t-ratio of the constant-only regression.

    The null is a unit root, so a small p-value is evidence of a level that
    returns.  The p-value is interpolated in the declared tau table and is held
    inside its range, so 0.01 reads as at most 0.01.

    Args:
        values: one participant's series at one scale.
        lags: how many lagged differences the regression holds; None takes
            `lag_rule`.

    Raises:
        ValueError: when the lag order is negative, when the series is shorter
            than the regression needs, when the series is constant, or when the
            regression fits exactly and leaves no residual to scale the ratio.
    """
    x = _finite(values, "an augmented Dickey-Fuller test")
    n = x.size
    order = lag_rule(n) if lags is None else int(lags)
    if order < 0:
        raise ValueError(f"a lag order is not negative; got {lags}")
    rows = n - order - 1
    if rows <= order + 2:
        raise ValueError(
            f"a Dickey-Fuller regression at lag order {order} needs "
            f"{2 * order + 4} epochs; got {n}"
        )
    if float(x.var()) <= 0.0:
        raise ValueError(
            f"a Dickey-Fuller regression needs a varying series; got a constant "
            f"at {x[0]}"
        )
    # The t-ratio is unchanged when the series is shifted and rescaled.  The
    # design holds a column of ones beside a column of the series itself, so a
    # series carried in small units or around a distant level hands the solver
    # columns of different sizes and the ratio comes back at 1e20 or at nan.
    # Centring and scaling first puts every column of the design at one size.
    centred = x - x.mean()
    x = centred / float(np.max(np.abs(centred)))
    difference = np.diff(x)
    columns = [np.ones(rows), x[order : n - 1]]
    for lag in range(1, order + 1):
        columns.append(difference[order - lag : n - 1 - lag])
    design = np.column_stack(columns)
    response = difference[order : n - 1]
    beta = np.linalg.lstsq(design, response, rcond=None)[0]
    residual = response - design @ beta
    rss = float(residual @ residual)
    if _no_residual(rss, float(response @ response)):
        raise ValueError(
            f"a Dickey-Fuller t-ratio needs a residual; the regression on {n} "
            f"epochs leaves a residual sum of squares of {rss}"
        )
    variance = rss / (rows - order - 2)
    standard_error = float(np.sqrt(variance * np.linalg.pinv(design.T @ design)[1, 1]))
    statistic = float(beta[1] / standard_error)
    return LevelTest(
        statistic=statistic,
        p_value=float(np.interp(statistic, ADF_TAU_CRITICAL, ADF_TAU_QUANTILES)),
        lags=order,
    )


def kpss_test(
    values: Sequence[float], lags: Optional[int] = None, trend: bool = False
) -> LevelTest:
    """
    The KPSS partial-sum statistic against a stationary level or a stationary trend.

    The null is stationarity, so a large statistic is evidence of a unit root.
    The long-run variance is the Bartlett-weighted Newey-West sum, and the
    p-value is interpolated in the declared table and held inside its range, so
    0.10 reads as at least 0.10.

    Args:
        values: one participant's series at one scale.
        lags: the Newey-West bandwidth; None takes `lag_rule`.
        trend: fit a level and a slope; the default fits a level alone.

    Raises:
        ValueError: when the bandwidth is negative, when the series is shorter
            than the statistic needs, or when an entry is not finite.
    """
    x = _finite(values, "a KPSS test")
    n = x.size
    order = lag_rule(n) if lags is None else int(lags)
    if order < 0:
        raise ValueError(f"a Newey-West bandwidth is not negative; got {lags}")
    needed = max(3, order + 2)
    if n < needed:
        raise ValueError(
            f"a KPSS statistic at bandwidth {order} needs {needed} epochs; got {n}"
        )
    design = np.column_stack([np.ones(n), np.arange(n, dtype=float)])
    if not trend:
        design = design[:, :1]
    residual = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    # A residual of exactly zero has no shape to read, and the declared value is
    # the smallest the statistic takes, which is no evidence against the null.
    # The long-run variance below is a sum of squares of that same residual, so
    # the reading is settled here, where the series is still there to divide by.
    if _no_residual(float(residual @ residual), float(x @ x)):
        return LevelTest(0.0, KPSS_P_VALUES[0], order)
    variance = float(residual @ residual) / n
    for lag in range(1, order + 1):
        covariance = float(residual[lag:] @ residual[:-lag]) / n
        variance += 2.0 * (1.0 - lag / (order + 1.0)) * covariance
    partial = np.cumsum(residual)
    statistic = float(partial @ partial) / (n * n * variance)
    table = KPSS_TREND_CRITICAL if trend else KPSS_LEVEL_CRITICAL
    return LevelTest(
        statistic=statistic,
        p_value=float(np.interp(statistic, table, KPSS_P_VALUES)),
        lags=order,
    )


@dataclass(frozen=True)
class UnitRootPair:
    """Both tests of the level, and the class the two of them give together."""

    adf: LevelTest
    kpss_level: LevelTest
    kpss_trend: LevelTest
    classification: str

    def as_row(self) -> Dict[str, object]:
        return {
            "adf_statistic": self.adf.statistic,
            "adf_p_value": self.adf.p_value,
            "kpss_level": self.kpss_level.statistic,
            "kpss_level_p_value": self.kpss_level.p_value,
            "kpss_trend": self.kpss_trend.statistic,
            "kpss_trend_p_value": self.kpss_trend.p_value,
            "lags": self.adf.lags,
            "classification": self.classification,
        }


def unit_root_pair(
    values: Sequence[float], lags: Optional[int] = None, alpha: float = 0.05
) -> UnitRootPair:
    """
    Both level tests on one series, and the class their two verdicts give.

    A rejected unit root with a level that holds is `STATIONARY`.  A unit root
    that holds with a rejected level and a rejected trend is `UNIT_ROOT`.  A
    unit root that holds with a rejected level and a trend that holds is
    `TREND_STATIONARY`.  Every other pattern is `UNDETERMINED`, which covers a
    series informative about neither null and a series whose two tests
    contradict each other.

    Raises:
        ValueError: when the level is outside (0, 1), and for whatever the two
            tests refuse.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"a test level is inside (0, 1); got {alpha}")
    adf = adf_test(values, lags)
    level = kpss_test(values, lags, trend=False)
    slope = kpss_test(values, lags, trend=True)
    returns = adf.p_value <= alpha
    level_moves = level.p_value <= alpha
    slope_moves = slope.p_value <= alpha
    if returns and not level_moves:
        verdict = STATIONARY
    elif level_moves and not slope_moves and not returns:
        verdict = TREND_STATIONARY
    elif level_moves and slope_moves and not returns:
        verdict = UNIT_ROOT
    else:
        verdict = UNDETERMINED
    return UnitRootPair(adf, level, slope, verdict)
