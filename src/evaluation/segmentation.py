"""
evaluation/segmentation.py
==========================
Penalized segmentation of a count series, and how two segmentations are compared.

Three rows of the nonstationarity grid wait on one dependency.  A08 asks where
the level of a series changes, A09 where its dispersion changes and A10 where
its slope changes, and all three are the same optimization under a different
segment cost: choose the cut set minimizing the summed cost plus a price per
cut.  Optimal partitioning solves it exactly, and prefix sums make each segment
cost an O(1) read, so one pass over a series is quadratic in its epochs with one
vector operation per epoch.

The cost decides which change the segmentation is sensitive to.  A Poisson cost
holds the rate constant inside a segment, so a cut buys a change in level.  A
Gaussian cost with the variance free holds mean and spread constant, so a cut
buys a change in either and the level term is already spent by A08.  A linear
cost holds an intercept and a slope constant, so a cut buys a change in trend.

One penalty is a choice, so the price is swept.  The path reports the cut set at
every penalty in a declared grid, which is what makes the reported count a curve
and not a single number chosen after seeing it.

Two segmentations are compared on six figures: the difference in how many cuts
they hold, the F1 of the cuts that match inside a margin, the covering of one
partition by the other, the Hausdorff distance between the cut sets, and the two
Kolmogorov-Smirnov statistics between their jump magnitudes and their segment
lengths.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from scipy.special import gammaln

#: The three costs, and the change each one buys a cut for.
POISSON = "poisson"
VARIANCE = "variance"
SLOPE = "slope"
COSTS: Tuple[str, ...] = (POISSON, VARIANCE, SLOPE)

#: Free parameters of each cost, which sets how short a segment may be.
COST_PARAMETERS: Dict[str, int] = {POISSON: 1, VARIANCE: 2, SLOPE: 3}

#: Kept out of a logarithm of a residual sum that a flat segment drives to zero.
_FLOOR = 1e-9

#: The smallest share of the whole series' variance one segment may claim.  A
#: count series holds long runs of zeros, and a segment covering one of them has
#: a residual of zero and a log-likelihood the price can never outweigh, so an
#: unfloored cost buys a cut around every run.
VARIANCE_FLOOR = 0.01


def _residual_variance(x: np.ndarray, cost: str) -> float:
    """
    The variance the floor is read off, which is what the cost cannot explain.

    A linear cost fits a line inside every segment, so it absorbs a trend over
    the whole series exactly and its optimal partition is invariant to one.  A
    floor read off the raw variance would move when a trend is added, and the
    cuts would move with it, so the linear cost reads its floor off the series
    with a global line taken out.
    """
    if x.size < 2:
        return 0.0
    if cost != SLOPE:
        return float(x.var())
    t = np.arange(x.size, dtype=float)
    slope, intercept = np.polyfit(t, x, 1)
    return float(np.var(x - (slope * t + intercept)))


def _prefix(values: np.ndarray, cost: str = VARIANCE) -> Dict[str, np.ndarray]:
    """Running sums a segment cost is read off in constant time."""
    x = np.asarray(values, dtype=float)
    t = np.arange(x.size, dtype=float)
    zero = np.zeros(1)
    return {
        "floor": np.array([max(_FLOOR, VARIANCE_FLOOR * _residual_variance(x, cost))]),
        "n": np.concatenate([zero, np.cumsum(np.ones_like(x))]),
        "x": np.concatenate([zero, np.cumsum(x)]),
        "xx": np.concatenate([zero, np.cumsum(x * x)]),
        "t": np.concatenate([zero, np.cumsum(t)]),
        "tt": np.concatenate([zero, np.cumsum(t * t)]),
        "tx": np.concatenate([zero, np.cumsum(t * x)]),
        "logfact": np.concatenate([zero, np.cumsum(_log_factorial(x))]),
    }


def _log_factorial(x: np.ndarray) -> np.ndarray:
    """The log factorial of a non-negative series, zero where it is negative."""
    x = np.asarray(x, dtype=float)
    return np.where(x >= 0.0, gammaln(np.maximum(x, 0.0) + 1.0), 0.0)


def segment_cost(
    prefix: Dict[str, np.ndarray], cost: str, starts: np.ndarray, stop: int
) -> np.ndarray:
    """
    Twice the negative maximized log-likelihood of every segment [start, stop).

    Args:
        prefix: the running sums of the series.
        cost: one of `COSTS`.
        starts: the candidate opening epochs, as one vector.
        stop: the epoch the segment ends before.

    Raises:
        ValueError: when the cost is not one of `COSTS`.
    """
    if cost not in COSTS:
        raise ValueError(f"segment cost {cost!r}; expected one of {list(COSTS)}")
    lo = np.asarray(starts, dtype=np.int64)
    n = prefix["n"][stop] - prefix["n"][lo]
    total = prefix["x"][stop] - prefix["x"][lo]
    if cost == POISSON:
        rate = np.maximum(total / n, _FLOOR)
        constant = prefix["logfact"][stop] - prefix["logfact"][lo]
        return -2.0 * (total * np.log(rate) - n * rate - constant)
    squares = prefix["xx"][stop] - prefix["xx"][lo]
    if cost == VARIANCE:
        residual = squares - total * total / n
    else:
        t_sum = prefix["t"][stop] - prefix["t"][lo]
        tt = prefix["tt"][stop] - prefix["tt"][lo]
        tx = prefix["tx"][stop] - prefix["tx"][lo]
        spread = np.maximum(tt - t_sum * t_sum / n, _FLOOR)
        covariance = tx - t_sum * total / n
        residual = squares - total * total / n - covariance * covariance / spread
    return n * np.log(np.maximum(residual / n, prefix["floor"][0]))


def segment(
    values: Sequence[float],
    cost: str = POISSON,
    penalty: float = 10.0,
    min_size: int = 0,
) -> Tuple[int, ...]:
    """
    The cut set minimizing the summed segment cost plus one price per cut.

    Args:
        values: the series, at one scale and inside one participant.
        cost: one of `COSTS`.
        penalty: what a cut costs, in the same units as the segment cost.
        min_size: the shortest admissible segment; zero takes the shortest the
            cost can be read on, which is one epoch past its free parameters.

    Returns:
        The interior cut points, each the first epoch of a new segment.

    Raises:
        ValueError: when the penalty is negative, since a cut would then pay.
    """
    if penalty < 0.0:
        raise ValueError(f"a cut price is not negative; got {penalty}")
    x = np.asarray(values, dtype=float)
    n = x.size
    floor = int(min_size) if min_size > 0 else COST_PARAMETERS[cost] + 1
    if n < 2 * floor:
        return ()
    prefix = _prefix(x, cost)
    best = np.full(n + 1, np.inf)
    best[0] = -float(penalty)
    origin = np.zeros(n + 1, dtype=np.int64)
    for stop in range(floor, n + 1):
        # Opening at zero is always admissible, so every reachable stop has at
        # least one finite predecessor and the minimum below is always defined.
        starts = np.arange(0, stop - floor + 1, dtype=np.int64)
        total = best[starts] + segment_cost(prefix, cost, starts, stop) + penalty
        pick = int(np.argmin(total))
        best[stop] = float(total[pick])
        origin[stop] = int(starts[pick])
    cuts: List[int] = []
    at = n
    while at > 0:
        at = int(origin[at])
        if at > 0:
            cuts.append(at)
    return tuple(sorted(cuts))


def penalty_path(
    values: Sequence[float],
    cost: str = POISSON,
    penalties: Sequence[float] = (5.0, 10.0, 20.0, 40.0, 80.0),
    min_size: int = 0,
) -> Tuple[Tuple[float, Tuple[int, ...]], ...]:
    """The cut set at every price in a declared grid, cheapest price first."""
    return tuple(
        (float(price), segment(values, cost, price, min_size))
        for price in sorted(float(p) for p in penalties)
    )


# ---------------------------------------------------------------------------
# How two cut sets are compared
# ---------------------------------------------------------------------------


def annotation_error(a: Sequence[int], b: Sequence[int]) -> int:
    """How many cuts one segmentation holds beyond the other."""
    return abs(len(a) - len(b))


def f1_margin(
    a: Sequence[int], b: Sequence[int], margin: int = 3
) -> Tuple[float, float, float]:
    """
    Precision, recall and F1 of the cuts matching inside a margin.

    Each cut is used once, so two cuts of one set never both claim the same cut
    of the other.
    """
    left, right = list(a), list(b)
    if not left and not right:
        return 1.0, 1.0, 1.0
    if not left or not right:
        return 0.0, 0.0, 0.0
    taken = set()
    hits = 0
    for cut in left:
        near = [
            (abs(cut - other), j)
            for j, other in enumerate(right)
            if j not in taken and abs(cut - other) <= margin
        ]
        if near:
            taken.add(min(near)[1])
            hits += 1
    precision = hits / len(left)
    recall = hits / len(right)
    if precision + recall == 0.0:
        return 0.0, 0.0, 0.0
    return precision, recall, 2.0 * precision * recall / (precision + recall)


def _partition(cuts: Sequence[int], n: int) -> List[Tuple[int, int]]:
    """The half-open segments a cut set makes of [0, n)."""
    edges = [0] + [int(c) for c in cuts] + [int(n)]
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def covering(a: Sequence[int], b: Sequence[int], n: int) -> float:
    """
    How well one partition of [0, n) is covered by the other, weighted by length.

    Raises:
        ValueError: when the series is empty, which no partition covers.
    """
    if n <= 0:
        raise ValueError(f"a partition covers at least one epoch; got {n}")
    left = _partition(a, n)
    right = _partition(b, n)
    total = 0.0
    for start, stop in left:
        best = 0.0
        for other_start, other_stop in right:
            overlap = max(0, min(stop, other_stop) - max(start, other_start))
            union = max(stop, other_stop) - min(start, other_start)
            best = max(best, overlap / union)
        total += (stop - start) * best
    return total / n


def hausdorff(a: Sequence[int], b: Sequence[int], n: int) -> float:
    """
    The largest distance from a cut of one set to the nearest cut of the other.

    An empty set against a non-empty one is the span itself, which is the
    distance a cut can be wrong by inside the series.
    """
    left = np.asarray(list(a), dtype=float)
    right = np.asarray(list(b), dtype=float)
    if left.size == 0 and right.size == 0:
        return 0.0
    if left.size == 0 or right.size == 0:
        return float(n)
    gaps = np.abs(left[:, None] - right[None, :])
    return float(max(gaps.min(axis=1).max(), gaps.min(axis=0).max()))


def jump_magnitudes(values: Sequence[float], cuts: Sequence[int]) -> np.ndarray:
    """The change in segment mean at every cut."""
    x = np.asarray(values, dtype=float)
    parts = _partition(cuts, x.size)
    means = np.array(
        [x[start:stop].mean() if stop > start else 0.0 for start, stop in parts]
    )
    return np.abs(np.diff(means)) if means.size > 1 else np.zeros(0)


def segment_lengths(cuts: Sequence[int], n: int) -> np.ndarray:
    """The length of every segment a cut set makes."""
    return np.array([stop - start for start, stop in _partition(cuts, n)], dtype=float)


def ks_statistic(a: Sequence[float], b: Sequence[float]) -> float:
    """
    The largest gap between two empirical distributions.

    Two empty samples agree and score zero; one empty sample against a filled one
    is the largest gap there is.
    """
    left = np.sort(np.asarray(list(a), dtype=float))
    right = np.sort(np.asarray(list(b), dtype=float))
    if left.size == 0 and right.size == 0:
        return 0.0
    if left.size == 0 or right.size == 0:
        return 1.0
    grid = np.concatenate([left, right])
    cdf_left = np.searchsorted(left, grid, side="right") / left.size
    cdf_right = np.searchsorted(right, grid, side="right") / right.size
    return float(np.max(np.abs(cdf_left - cdf_right)))


@dataclass(frozen=True)
class ChangepointMatch:
    """The six figures a changepoint row reports on one pair of series."""

    annotation_error: int
    f1_margin: float
    covering: float
    hausdorff: float
    jump_ks: float
    segment_ks: float

    def as_row(self) -> Dict[str, float]:
        return {
            "annotation_error": self.annotation_error,
            "f1_margin": self.f1_margin,
            "covering": self.covering,
            "hausdorff": self.hausdorff,
            "jump_ks": self.jump_ks,
            "segment_ks": self.segment_ks,
        }


def match_changepoints(
    real: Sequence[float],
    synthetic: Sequence[float],
    cost: str = POISSON,
    penalty: float = 10.0,
    margin: int = 3,
    min_size: int = 0,
) -> ChangepointMatch:
    """
    Segment two series under one cost and one price, and compare the two cut sets.

    Raises:
        ValueError: when the two series are of different lengths, since the
            covering and the Hausdorff distance are both stated on one span.
    """
    left = np.asarray(real, dtype=float)
    right = np.asarray(synthetic, dtype=float)
    if left.size != right.size:
        raise ValueError(
            f"a changepoint match is taken on one span; these are {left.size} and "
            f"{right.size} epochs"
        )
    a = segment(left, cost, penalty, min_size)
    b = segment(right, cost, penalty, min_size)
    return ChangepointMatch(
        annotation_error=annotation_error(a, b),
        f1_margin=f1_margin(a, b, margin)[2],
        covering=covering(a, b, left.size),
        hausdorff=hausdorff(a, b, left.size),
        jump_ks=ks_statistic(jump_magnitudes(left, a), jump_magnitudes(right, b)),
        segment_ks=ks_statistic(
            segment_lengths(a, left.size), segment_lengths(b, right.size)
        ),
    )


def cut_rate(
    values: Sequence[float], cost: str, penalty: float, min_size: int = 0
) -> float:
    """Cuts per epoch, which is the magnitude a changepoint cell carries."""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return 0.0
    return len(segment(x, cost, penalty, min_size)) / x.size
