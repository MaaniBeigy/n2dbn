"""
evaluation/omnibus.py
=====================
The form-free omnibus and the common changepoint, which are two rows of the grid.

A16 asks whether the distribution of a series moved at all, with no form named
in advance.  The maximum mean discrepancy under a Gaussian kernel answers it:
the kernel is characteristic, so the population discrepancy is zero exactly when
the two distributions agree, and every difference a declared form would name
sits inside it.  The estimator is the unbiased one, whose expectation is zero
under the null, and the bandwidth is the median pairwise distance of the pooled
sample, which puts the kernel on the scale the points occupy.  The reference
distribution comes from permuting the pooled sample under a seeded generator, so
the p-value stands without an asymptotic argument.

The sample is a delay embedding, so one point is a short window of the series
and the discrepancy reads the joint distribution of consecutive epochs.  A
comparison of the values alone returns one answer for every ordering of them,
and the omnibus is here to see an ordering move as well.

B1 asks whether several series share one changepoint.  It is optimal
partitioning again, over a multivariate Gaussian cost: a segment costs its
length times the log determinant of its covariance, so a cut is bought by what
the panel does jointly and a change in one series alone pays for all of them.
The diagonal cost holds the series independent inside a segment and prices a
change in the level or the spread of any one of them.  The full cost carries the
whole covariance and prices a change in their dependence as well, on segments
longer than the series count.  Both determinants are ridged, so a short or flat
segment stays finite and a cut there buys nothing.

A caller gets the cut set at one price, the cut set at every price in a declared
grid, and the share of the achievable cost drop the priced cut set buys, which
is the magnitude B1 reports.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: The two covariance costs a common cut is bought under.
DIAGONAL = "diagonal"
FULL = "full"
COVARIANCES: Tuple[str, ...] = (DIAGONAL, FULL)

#: Kept inside every logarithm of a variance a flat segment drives to zero.
_FLOOR = 1e-9

#: The smallest share of the panel's mean variance one segment may claim.  A
#: segment covering a run of equal epochs has a determinant of zero and a cost
#: no price can outweigh, so an unridged cost buys a cut around every such run.
VARIANCE_FLOOR = 0.01

#: The window the omnibus reads, in epochs.  Three consecutive epochs carry a
#: level, a step and a turn, which is what separates two orderings of one
#: histogram.
EMBED_LENGTH = 3

#: The prices the common cut is swept over, in the units of the segment cost.
PRICES: Tuple[float, ...] = (5.0, 10.0, 20.0, 40.0, 80.0)


# ---------------------------------------------------------------------------
# A16, the omnibus under a characteristic kernel
# ---------------------------------------------------------------------------


def _samples(values: np.ndarray) -> np.ndarray:
    """
    One sample as points by coordinates, taking a series as one coordinate.

    Raises:
        ValueError: when the sample carries more than two axes, or holds a
            point that is not finite, which stands at no distance from anything.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim > 2:
        raise ValueError(f"a sample holds one or two axes; got {x.ndim}")
    stray = x[~np.isfinite(x)]
    if stray.size:
        raise ValueError(f"a sample holds finite points; got {stray[0]}")
    return x.reshape(-1, 1) if x.ndim == 1 else x


def _pair(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Two samples as point matrices, checked to share their coordinates.

    Raises:
        ValueError: when the two samples carry a different number of
            coordinates, which leaves no distance between their points.
    """
    x, y = _samples(a), _samples(b)
    if x.shape[1] != y.shape[1]:
        raise ValueError(
            f"two samples share their coordinates; got {x.shape[1]} and {y.shape[1]}"
        )
    return x, y


def _square_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The squared distance from every point of one sample to every point of another."""
    difference = a[:, None, :] - b[None, :, :]
    return np.einsum("ijk,ijk->ij", difference, difference)


def median_bandwidth(a: np.ndarray, b: np.ndarray) -> float:
    """
    The median distance between two points of the pooled sample, one when it is zero.

    A pool holding one distinct point has a median distance of zero, and a
    kernel of zero bandwidth is undefined, so the fallback is one.
    """
    x, y = _pair(a, b)
    pooled = np.vstack([x, y])
    if pooled.shape[0] < 2:
        return 1.0
    squares = _square_distances(pooled, pooled)
    off = squares[np.triu_indices(pooled.shape[0], k=1)]
    median = float(np.median(np.sqrt(off)))
    return median if median > 0.0 else 1.0


def gaussian_kernel(a: np.ndarray, b: np.ndarray, bandwidth: float) -> np.ndarray:
    """
    The Gaussian kernel between every point of one sample and every point of another.

    Raises:
        ValueError: when the bandwidth is not positive, which has no kernel.
    """
    if bandwidth <= 0.0:
        raise ValueError(f"a kernel bandwidth is positive; got {bandwidth}")
    x, y = _pair(a, b)
    return np.exp(-_square_distances(x, y) / (2.0 * bandwidth * bandwidth))


def mmd2_unbiased(
    a: np.ndarray, b: np.ndarray, bandwidth: Optional[float] = None
) -> float:
    """
    The unbiased estimate of the squared maximum mean discrepancy of two samples.

    Both within-sample means drop their diagonal, so the expectation is zero
    when the two samples come from one distribution and the estimate takes small
    negative values there.

    Args:
        a: one sample as points by coordinates.
        b: the other sample.
        bandwidth: the kernel bandwidth; None takes the median heuristic.

    Returns:
        The estimate, and 0.0 when either sample holds fewer than two points,
        since a within-sample mean off the diagonal then has no term.
    """
    x, y = _pair(a, b)
    n, m = x.shape[0], y.shape[0]
    if n < 2 or m < 2:
        return 0.0
    scale = median_bandwidth(x, y) if bandwidth is None else float(bandwidth)
    kxx = gaussian_kernel(x, x, scale)
    kyy = gaussian_kernel(y, y, scale)
    kxy = gaussian_kernel(x, y, scale)
    within_x = (float(kxx.sum()) - float(np.trace(kxx))) / (n * (n - 1))
    within_y = (float(kyy.sum()) - float(np.trace(kyy))) / (m * (m - 1))
    return float(within_x + within_y - 2.0 * float(kxy.mean()))


def mmd_permutation_p(
    a: np.ndarray, b: np.ndarray, n_permutations: int = 199, seed: int = 0
) -> Tuple[float, float]:
    """
    The discrepancy of two samples, and how often a reshuffle of the pool reaches it.

    The bandwidth is read once on the observed pool and held across the
    permutations, so the null moves the labels and leaves the geometry alone.

    Returns:
        The statistic and its p-value; a sample of fewer than two points scores
        zero at a p-value of one.

    Raises:
        ValueError: when no permutation is asked for, which leaves no null.
    """
    if n_permutations < 1:
        raise ValueError(
            f"a permutation null holds at least one draw; got {n_permutations}"
        )
    x, y = _pair(a, b)
    n, m = x.shape[0], y.shape[0]
    if n < 2 or m < 2:
        return 0.0, 1.0
    scale = median_bandwidth(x, y)
    statistic = mmd2_unbiased(x, y, scale)
    pooled = np.vstack([x, y])
    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations, dtype=float)
    for draw in range(n_permutations):
        order = rng.permutation(n + m)
        null[draw] = mmd2_unbiased(pooled[order[:n]], pooled[order[n:]], scale)
    return statistic, float((1 + np.sum(null >= statistic)) / (1 + n_permutations))


def delay_embed(values: Sequence[float], length: int = EMBED_LENGTH) -> np.ndarray:
    """
    The consecutive windows of a series, one window per row.

    Returns:
        The windows, and a sample of no points when the series is shorter than
        one window.

    Raises:
        ValueError: when the window holds no epoch, or the series carries more
            than one axis, whose consecutive windows are undefined.
    """
    if length < 1:
        raise ValueError(f"a delay window holds at least one epoch; got {length}")
    x = np.asarray(values, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"a series carries one axis; got {x.ndim}")
    if x.size < length:
        return np.zeros((0, length), dtype=float)
    index = np.arange(length)[None, :] + np.arange(x.size - length + 1)[:, None]
    return x[index]


def window_mmd(values: Sequence[float], period: float) -> float:
    """
    How far the distribution of one participant's short windows moved over the record.

    The series is split into its first and its second half, both halves are
    delay-embedded at `EMBED_LENGTH`, and the statistic is the unbiased squared
    discrepancy between the two window samples.

    Args:
        values: one participant's series at one scale.
        period: the calendar period of that scale, carried for the grid
            interface; the split is taken on the record itself.

    Returns:
        The discrepancy, and 0.0 for a constant series or one too short to fill
        both halves with two windows each.
    """
    x = np.asarray(values, dtype=float)
    if x.size < 2 * (EMBED_LENGTH + 1):
        return 0.0
    if float(x.max()) == float(x.min()):
        return 0.0
    half = x.size // 2
    return mmd2_unbiased(delay_embed(x[:half]), delay_embed(x[half:]))


# ---------------------------------------------------------------------------
# B1, the changepoint several series share
# ---------------------------------------------------------------------------


def _check_cost(kind: str) -> None:
    """
    Refuse a covariance cost this module does not carry.

    Raises:
        ValueError: when the cost is not one of `COVARIANCES`.
    """
    if kind not in COVARIANCES:
        raise ValueError(
            f"covariance cost {kind!r}; expected one of {list(COVARIANCES)}"
        )


def _panel(matrix: np.ndarray) -> np.ndarray:
    """
    A panel as epochs by series, taking a single series as one column.

    Raises:
        ValueError: when the panel carries no series or more than two axes, or
            holds an epoch that is not finite, which no covariance reads.
    """
    values = np.asarray(matrix, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError(f"a panel is epochs by series; got shape {values.shape}")
    stray = values[~np.isfinite(values)]
    if stray.size:
        raise ValueError(f"a panel holds finite epochs; got {stray[0]}")
    return values


def min_segment(kind: str, n_series: int) -> int:
    """
    The shortest segment a covariance cost can be read on.

    A diagonal cost estimates one mean and one variance per series, which two
    epochs define.  A full cost estimates the whole covariance, which a segment
    of no more epochs than the series count leaves singular.

    Raises:
        ValueError: when the cost is not one of `COVARIANCES`.
    """
    _check_cost(kind)
    return 2 if kind == DIAGONAL else int(n_series) + 1


def _prefix(values: np.ndarray, kind: str) -> Dict[str, np.ndarray]:
    """Running sums a multivariate segment cost is read off in constant time."""
    n, d = values.shape
    spread = float(np.mean(values.var(axis=0)))
    # The last term holds the ridge above the rounding error in the scatter,
    # which a constant panel of large counts otherwise sits below.
    ridge = max(_FLOOR, VARIANCE_FLOOR * spread, _FLOOR * float(np.mean(values**2)))
    zero = np.zeros((1, d))
    prefix = {
        "floor": np.array([ridge]),
        "n": np.arange(n + 1, dtype=float),
        "x": np.concatenate([zero, values]).cumsum(axis=0),
        "xx": np.concatenate([zero, values * values]).cumsum(axis=0),
    }
    if kind == FULL:
        outer = values[:, :, None] * values[:, None, :]
        prefix["outer"] = np.concatenate([np.zeros((1, d, d)), outer]).cumsum(axis=0)
    return prefix


def common_segment_cost(
    prefix: Dict[str, np.ndarray], kind: str, starts: np.ndarray, stop: int
) -> np.ndarray:
    """
    The length of every segment [start, stop) times the log determinant it carries.

    The additive constants of the Gaussian log-likelihood are the same for every
    partition of one panel, so the cost carries the determinant term alone.

    Args:
        prefix: the running sums of the panel.
        kind: one of `COVARIANCES`.
        starts: the candidate opening epochs, as one vector.
        stop: the epoch the segment ends before.

    Raises:
        ValueError: when the cost is not one of `COVARIANCES`, or a segment
            falls outside the panel or holds no epoch, which has no covariance.
    """
    _check_cost(kind)
    lo = np.asarray(starts, dtype=np.int64)
    epochs = int(prefix["n"][-1])
    if not 1 <= int(stop) <= epochs:
        raise ValueError(
            f"a segment ends inside the panel of {epochs} epochs; got stop {stop}"
        )
    outside = lo[(lo < 0) | (lo >= int(stop))]
    if outside.size:
        raise ValueError(
            f"a segment opens inside [0, {int(stop)}); got start {int(outside[0])}"
        )
    length = (prefix["n"][stop] - prefix["n"][lo])[:, None]
    total = prefix["x"][stop] - prefix["x"][lo]
    floor = float(prefix["floor"][0])
    if kind == DIAGONAL:
        squares = prefix["xx"][stop] - prefix["xx"][lo]
        variance = (squares - total * total / length) / length
        logdet = np.sum(np.log(np.maximum(variance, floor)), axis=1)
        return length[:, 0] * logdet
    scatter = prefix["outer"][stop] - prefix["outer"][lo]
    centred = scatter - total[:, :, None] * total[:, None, :] / length[:, :, None]
    covariance = centred / length[:, :, None] + floor * np.eye(total.shape[1])
    return length[:, 0] * np.linalg.slogdet(covariance)[1]


def segment_common(
    matrix: np.ndarray,
    kind: str = DIAGONAL,
    penalty: float = 10.0,
    min_size: int = 0,
) -> Tuple[int, ...]:
    """
    The cut set several series share, minimizing the summed cost plus one price per cut.

    Every series is cut at the same epoch, so a cut is bought by what the panel
    does jointly, and a change in one series has to outweigh the price on its
    own.

    Args:
        matrix: epochs by series, at one scale and inside one participant.
        kind: one of `COVARIANCES`.
        penalty: what a cut costs, in the units of the segment cost.
        min_size: the shortest admissible segment; zero takes `min_segment`.

    Returns:
        The interior cut points, each the first epoch of a new segment.

    Raises:
        ValueError: when the penalty is negative, since a cut would then pay.
    """
    if penalty < 0.0:
        raise ValueError(f"a cut price is not negative; got {penalty}")
    values = _panel(matrix)
    n = values.shape[0]
    floor = int(min_size) if min_size > 0 else min_segment(kind, values.shape[1])
    if n < 2 * floor:
        return ()
    prefix = _prefix(values, kind)
    best = np.full(n + 1, np.inf)
    best[0] = -float(penalty)
    origin = np.zeros(n + 1, dtype=np.int64)
    for stop in range(floor, n + 1):
        # Opening at zero is always admissible, so every reachable stop has at
        # least one finite predecessor and the minimum below is always defined.
        starts = np.arange(0, stop - floor + 1, dtype=np.int64)
        total = best[starts] + common_segment_cost(prefix, kind, starts, stop) + penalty
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


def partition_cost(
    matrix: np.ndarray, cuts: Sequence[int], kind: str = DIAGONAL
) -> float:
    """
    The summed cost of the segments a cut set makes, with no price for the cuts.

    Returns:
        The sum, and 0.0 for a panel of no epochs, which holds no segment.
    """
    values = _panel(matrix)
    n = values.shape[0]
    if n == 0:
        return 0.0
    prefix = _prefix(values, kind)
    edges = [0] + sorted(int(c) for c in cuts) + [n]
    return float(
        sum(
            float(common_segment_cost(prefix, kind, np.array([lo]), hi)[0])
            for lo, hi in zip(edges[:-1], edges[1:])
            if hi > lo
        )
    )


def common_penalty_path(
    matrix: np.ndarray,
    kind: str = DIAGONAL,
    penalties: Sequence[float] = PRICES,
    min_size: int = 0,
) -> Tuple[Tuple[float, Tuple[int, ...]], ...]:
    """The common cut set at every price in a declared grid, cheapest price first."""
    return tuple(
        (float(price), segment_common(matrix, kind, price, min_size))
        for price in sorted(float(p) for p in penalties)
    )


def cost_drop_fraction(
    matrix: np.ndarray,
    kind: str = DIAGONAL,
    penalty: float = 10.0,
    min_size: int = 0,
) -> float:
    """
    The share of the achievable cost drop the priced common cut set buys.

    The panel held as one segment is the dearest partition and the unpriced
    segmentation is the cheapest, so the distance between the two is the largest
    drop any admissible cut set buys.  What the priced cut set takes of it is a
    share in [0, 1], which is the magnitude the common changepoint row reports.

    Returns:
        The share, and 0.0 when no cut set drops the cost at all, which is what
        a panel with no joint structure to segment reports.
    """
    values = _panel(matrix)
    whole = partition_cost(values, (), kind)
    free = partition_cost(values, segment_common(values, kind, 0.0, min_size), kind)
    if whole - free <= 0.0:
        return 0.0
    priced = partition_cost(
        values, segment_common(values, kind, penalty, min_size), kind
    )
    return float((whole - priced) / (whole - free))
