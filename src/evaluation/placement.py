"""
evaluation/placement.py
=======================
Where inside its parent epoch an event sits, and how two logs are compared on it.

Row A15 reads the parent-epoch profile the ladder supplies.  An epoch total says
how many events a day held.  Where inside that day each event sat is a second
quantity, and the totals at every scale of the ladder are blind to it: a log
placing every episode at nine in the morning and a log spreading the same
episodes across the whole day carry the same daily count.

A position wraps, so every summary here is taken on the angle two pi times the
position.  The resultant length of the count-weighted unit vectors says how
tightly the events cluster inside the block, the circular mean says where the
cluster sits, and the circular variance is one minus the resultant.  A Rayleigh
test reads the same resultant as evidence against an even spread through the
block, and Mardia's series gives its p-value.

Two logs are compared on their binned profiles.  The total variation distance
and a chi-square statistic price the disagreement in mass, and the circular
distance between the two mean angles prices a placement that holds its shape and
moves in time.

`placement_concentration` is the one scalar the grid reads per participant per
series: the resultant length of the within-parent positions, zero when the
events are spread evenly through the parent block and one when they all sit at
one position.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

#: How many equal bins of the parent block a profile holds unless a caller
#: declares another number.
DEFAULT_BINS = 12

#: Below this resultant length the mean angle is the angle of rounding error,
#: since weight spread evenly around the block cancels to nothing.
RESULTANT_FLOOR = 1e-12

#: At this statistic the exponential of its negative is already zero at double
#: precision, so the correction series cannot lift the p-value off the floor,
#: and its fourth power overflows before it gets the chance.
Z_FLOOR = 750.0


def _finite_row(values: Sequence[float], role: str) -> np.ndarray:
    """One row of finite floats, refusing every other shape and every gap."""
    v = np.asarray(values, dtype=float)
    if v.ndim != 1:
        raise ValueError(f"{role} runs along one axis; got shape {v.shape}")
    gap = ~np.isfinite(v)
    if gap.any():
        raise ValueError(f"{role} is finite; got {float(v[gap][0])}")
    return v


def _weights(counts: Sequence[float]) -> np.ndarray:
    """The counts as a weight vector, refusing what cannot weight an angle."""
    w = _finite_row(counts, "a count")
    if w.size and float(w.min()) < 0.0:
        raise ValueError(f"a count is not negative; got {float(w.min())}")
    with np.errstate(over="ignore"):
        total = float(w.sum())
    if not math.isfinite(total):
        raise ValueError(f"the counts sum to a finite weight; got {total}")
    return w


def epoch_blocks(n_epochs: int, period: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    The parent block of every epoch of a series, and where inside it each sits.

    Args:
        n_epochs: how many epochs the series holds.
        period: how many epochs of that scale one parent block spans, which is
            24 at the hour scale and 7 at the day scale.

    Returns:
        The parent index of every epoch, and its position inside the parent in
        [0, 1).

    Raises:
        ValueError: when the epoch count is negative, or the period is not a
            finite number of at least one epoch.  A block shorter than one epoch
            carries a parent index no integer holds.
    """
    if n_epochs < 0:
        raise ValueError(f"a series holds no negative count of epochs; got {n_epochs}")
    if not math.isfinite(period) or period < 1.0:
        raise ValueError(f"a parent block spans at least one whole epoch; got {period}")
    t = np.arange(int(n_epochs), dtype=float)
    # The final block of a window is often truncated, and its positions stay on
    # the scale of a whole block so a half-recorded day is read on the same axis.
    return np.floor(t / float(period)).astype(np.int64), np.mod(t, period) / period


def within_parent_positions(parents: Sequence[int]) -> np.ndarray:
    """
    Where inside its parent each slot sits, as a share of the parent in [0, 1).

    Args:
        parents: the parent block of every slot, non-decreasing, as the ladder
            supplies it.

    Returns:
        One position per slot, the rank of the slot inside its parent divided by
        how many slots that parent holds.  An empty map gives an empty result.

    Raises:
        ValueError: when the map is not one row, or when it steps backwards,
            since a parent block is a run of consecutive slots.
    """
    ids = np.asarray(parents, dtype=np.int64)
    if ids.ndim != 1:
        raise ValueError(f"a parent map runs along one axis; got shape {ids.shape}")
    gaps = np.diff(ids)
    if gaps.size and int(gaps.min()) < 0:
        at = int(np.argmin(gaps))
        raise ValueError(
            f"a parent map is non-decreasing; slot {at + 1} drops from "
            f"{int(ids[at])} to {int(ids[at + 1])}"
        )
    _, first, sizes = np.unique(ids, return_index=True, return_counts=True)
    rank = np.arange(ids.size, dtype=float) - np.repeat(first, sizes).astype(float)
    return rank / np.repeat(sizes, sizes).astype(float)


@dataclass(frozen=True)
class PositionProfile:
    """The within-parent density of one series, and the weight that built it."""

    density: np.ndarray
    weight: float
    defined: bool


def position_profile(
    counts: Sequence[float], parents: Sequence[int], n_bins: int = DEFAULT_BINS
) -> PositionProfile:
    """
    The density of within-parent positions, weighted by the counts.

    Args:
        counts: how many events every slot holds.
        parents: the parent block of every slot, non-decreasing.
        n_bins: how many equal bins of the parent block the density holds.

    Returns:
        The density over `n_bins` bins summing to one, the total count behind
        it, and whether any event defined it.  A series holding no events gives
        the uniform density, a weight of zero and `defined` False.

    Raises:
        ValueError: when the bin count is below two, when the counts and the
            parent map are of different lengths, or when a count is negative.
    """
    if n_bins < 2:
        raise ValueError(f"a position profile holds at least two bins; got {n_bins}")
    w = _weights(counts)
    ids = np.asarray(parents, dtype=np.int64)
    if w.size != ids.size:
        raise ValueError(
            f"a count runs beside its parent; these are {w.size} counts and "
            f"{ids.size} parents"
        )
    total = float(w.sum())
    if total <= 0.0:
        return PositionProfile(np.full(int(n_bins), 1.0 / int(n_bins)), 0.0, False)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    mass, _ = np.histogram(within_parent_positions(ids), bins=edges, weights=w)
    return PositionProfile(mass / total, total, True)


@dataclass(frozen=True)
class CircularStats:
    """The three circular figures of a weighted set of within-parent positions."""

    mean: float
    variance: float
    resultant: float
    defined: bool


def circular_stats(
    positions: Sequence[float], weights: Optional[Sequence[float]] = None
) -> CircularStats:
    """
    The circular mean, variance and resultant length of within-parent positions.

    Args:
        positions: the within-parent positions, each a share of the block.
        weights: how many events sit at each position; one each when omitted.

    Returns:
        The mean angle in [0, 2 pi), the circular variance, the resultant length
        in [0, 1], and whether the mean angle is defined.  No weight at all, and
        weight spread evenly around the block, both give a resultant of zero, a
        variance of one, a mean of zero and `defined` False.

    Raises:
        ValueError: when a position is not finite, when the positions and the
            weights are of different lengths, or when a weight is negative.
    """
    p = _finite_row(positions, "a position")
    w = np.ones(p.size) if weights is None else _weights(weights)
    if w.size != p.size:
        raise ValueError(
            f"a weight runs beside its position; these are {w.size} weights and "
            f"{p.size} positions"
        )
    total = float(w.sum())
    if total <= 0.0:
        return CircularStats(0.0, 1.0, 0.0, False)
    angles = 2.0 * math.pi * p
    cosine = float(np.sum(w * np.cos(angles)) / total)
    sine = float(np.sum(w * np.sin(angles)) / total)
    resultant = min(1.0, math.hypot(cosine, sine))
    if resultant < RESULTANT_FLOOR:
        return CircularStats(0.0, 1.0, 0.0, False)
    return CircularStats(
        math.atan2(sine, cosine) % (2.0 * math.pi), 1.0 - resultant, resultant, True
    )


@dataclass(frozen=True)
class Rayleigh:
    """The uniformity test of one set of within-parent positions."""

    resultant: float
    z: float
    p_value: float


def rayleigh_test(
    positions: Sequence[float], weights: Optional[Sequence[float]] = None
) -> Rayleigh:
    """
    Test whether within-parent positions are spread evenly through the parent.

    The statistic is the event count times the squared resultant length, and the
    p-value is Mardia's second-order series for it, held inside [0, 1] since the
    series overshoots at the ends.  Fewer than two events carry no evidence
    against an even spread, so the p-value is one.

    Args:
        positions: the within-parent positions, each a share of the block.
        weights: how many events sit at each position; one each when omitted.

    Raises:
        ValueError: when a position is not finite, when the positions and the
            weights are of different lengths, or when a weight is negative.
    """
    p = np.asarray(positions, dtype=float)
    w = np.ones(p.size) if weights is None else _weights(weights)
    stats = circular_stats(p, w)
    n = float(w.sum())
    if n < 2.0:
        return Rayleigh(stats.resultant, 0.0, 1.0)
    z = n * stats.resultant * stats.resultant
    if z >= Z_FLOOR:
        return Rayleigh(stats.resultant, z, 0.0)
    series = (
        1.0
        + (2.0 * z - z * z) / (4.0 * n)
        - (24.0 * z - 132.0 * z**2 + 76.0 * z**3 - 9.0 * z**4) / (288.0 * n * n)
    )
    return Rayleigh(stats.resultant, z, min(1.0, max(0.0, math.exp(-z) * series)))


# ---------------------------------------------------------------------------
# How two placements are compared
# ---------------------------------------------------------------------------


def _profile_pair(
    a: Sequence[float], b: Sequence[float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Two profiles as arrays, refusing a pair no single binning covers."""
    left = _finite_row(a, "a profile bin")
    right = _finite_row(b, "a profile bin")
    if left.size != right.size:
        raise ValueError(
            f"two profiles share one binning; these hold {left.size} and "
            f"{right.size} bins"
        )
    if left.size == 0:
        raise ValueError("two profiles hold at least one bin; both hold none")
    low = min(float(left.min()), float(right.min()))
    if low < 0.0:
        raise ValueError(f"a profile bin holds no negative mass; got {low}")
    with np.errstate(over="ignore"):
        mass = float(left.sum()) + float(right.sum())
    if not math.isfinite(mass):
        raise ValueError(f"two profiles hold a finite mass between them; got {mass}")
    return left, right


def total_variation(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Half the summed gap between two position profiles, bin by bin.

    Two profiles that each sum to one give a value in [0, 1], zero when they
    agree in every bin and one when they share no bin.
    """
    left, right = _profile_pair(a, b)
    return float(0.5 * np.sum(np.abs(left - right)))


def chi_square(a: Sequence[float], b: Sequence[float]) -> float:
    """
    The symmetric chi-square divergence between two position profiles.

    Every bin contributes its squared gap over the mass the two profiles hold
    there together, so two profiles that each sum to one give a value in [0, 2].
    Bins holding no mass in either profile contribute nothing, which is what
    keeps a profile with empty bins out of a division by zero.
    """
    left, right = _profile_pair(a, b)
    total = left + right
    active = total > 0.0
    gap = left[active] - right[active]
    return float(np.sum(gap * (gap / total[active])))


def circular_mean_distance(a: float, b: float) -> float:
    """
    The shortest distance between two mean angles, in [0, pi].

    Raises:
        ValueError: when either angle is not a finite number of radians.
    """
    left = float(a)
    right = float(b)
    for angle in (left, right):
        if not math.isfinite(angle):
            raise ValueError(f"a mean angle is finite; got {angle}")
    return float(abs((left - right + math.pi) % (2.0 * math.pi) - math.pi))


def placement_concentration(values: Sequence[float], period: float) -> float:
    """
    How tightly one series places its events inside its parent block.

    Args:
        values: one participant's counts at one scale.
        period: how many epochs of that scale one parent block spans.

    Returns:
        The resultant length of the count-weighted within-parent positions,
        between 0 when the events are spread evenly through the block and 1 when
        they all sit at one position.  A series holding no events gives 0.0.

    Raises:
        ValueError: when the period is not a finite number of at least one
            epoch, or when a count is negative, is not finite, or is one of a
            set whose total overflows.
    """
    w = _weights(values)
    _, positions = epoch_blocks(w.size, period)
    return circular_stats(positions, w).resultant
