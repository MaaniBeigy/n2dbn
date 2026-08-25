"""
src/evaluation/heterogeneity.py
===============================
Whether a drawn log reproduces the heterogeneity of the log it was drawn from.

The measure is the coefficient of quartile variation, $CQV=(Q_3-Q_1)/(Q_3+Q_1)$,
a unitless measure of relative dispersion read off the quartiles.  It is stable
to skew and to extreme values, where the coefficient of variation moves with the
mean and the standard deviation.

Two decisions are stated here because the data forced them and a reader of a
result has to know which one it was read under.

*The totals are weekly.*  A daily count of one label is zero on most days, so
its first quartile is zero, and the measure is then either undefined at zero over
zero or pinned at one, its maximum, where it carries nothing.  Measured on the
observed panel, CQV was defined on 1 to 42 participants of 51 depending on the
label; on weekly totals it is defined on 9 to 47.

*The interval is a percentile bootstrap over participants.*  The accelerated
interval is undefined on this data: quartiles of tied small integers do not move
when one observation is dropped, so the jackknife variance is zero and the
acceleration has no value.  The participant is the sampling unit, so the
resampling moves whole participants and the interval is the percentile one.

A label whose quartiles are both zero has no CQV, which is reported as a share of
the roster and never as a value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from pycvcqv import cqv

#: Bootstrap draws and the level of the reported interval.
REPLICATES = 999
CONF_LEVEL = 0.95

#: A participant needs this many epochs before a quartile means anything.
MIN_EPOCHS = 4

#: How far the share of the roster with no defined CQV may differ between the
#: two panels before they are called incomparable.  The measure is read over the
#: participants that define it, so two panels that define it for very different
#: fractions of the roster are not describing the same population however well
#: their intervals overlap: `walk` stood at 0.078 real against 0.788 drawn.
SPARSE_TOLERANCE = 0.15


def _quartiles(x: np.ndarray) -> Tuple[float, float]:
    """The first and third quartiles, on the convention the measure is read at."""
    return float(np.percentile(x, 25)), float(np.percentile(x, 75))


def participant_cqv(values: Sequence[float]) -> Optional[float]:
    """
    One participant's coefficient of quartile variation, or nothing where it has none.

    Returns:
        The measure, or `None` when both quartiles are zero and it is undefined,
        which is the sparse case the roster reports as a share.

    Raises:
        ValueError: when the series holds fewer than `MIN_EPOCHS` epochs, or a
            value that is not finite or is below zero, since a count and a
            duration are both non-negative.
    """
    x = np.asarray(values, dtype=float).ravel()
    if x.size < MIN_EPOCHS:
        raise ValueError(f"a quartile needs at least {MIN_EPOCHS} epochs; got {x.size}")
    if not np.all(np.isfinite(x)):
        raise ValueError("a series of counts or durations holds finite values only")
    if np.any(x < 0.0):
        raise ValueError(f"a count and a duration are at or above zero; got {x.min()}")
    # Both quartiles zero is the undefined case, and it is decided here rather
    # than left to the library's warning.  Reaching it through an exception
    # handler made every other failure inside `cqv` look like sparsity, which is
    # how a whole missing label read as "no participant defines a CQV".
    low, high = _quartiles(x)
    if low == 0.0 and high == 0.0:
        return None
    # `cqv` raises a `Warning` when $Q_3+Q_1$ is zero.  On a non-negative series
    # that is exactly the case decided above, so the call cannot warn here and
    # the handler that used to stand around it is gone.
    return float(cqv(data=x))


def is_pinned(values: Sequence[float]) -> bool:
    """
    Whether a participant's CQV sits at its maximum because its lower quartile is zero.

    `pycvcqv` returns exactly 1.0 whenever $Q_1=0$ and $Q_3>0$, which is
    arithmetically right and carries no dispersion: it says a quarter of this
    participant's weeks are empty, not that its spread is maximal.  On the
    observed roster that is 13 of 51 for `gym`, 23 of 51 for `nap` and 8 of 51
    for `mindfulness`, so a median taken over these is a median of the
    degeneracy.  The weekly-totals decision of 2026-08-22 moved the problem from
    undefined to pinned; it did not remove it.
    """
    x = np.asarray(values, dtype=float).ravel()
    low, high = _quartiles(x)
    return low == 0.0 and high > 0.0


@dataclass(frozen=True)
class Heterogeneity:
    """One label's population CQV, its interval and what the roster could not define."""

    label: str
    cqv: float
    lower: float
    upper: float
    n_defined: int
    n_participants: int
    replicates: int
    seed: int
    #: Participants whose CQV is pinned at 1.0 by a zero lower quartile.  These
    #: are defined but carry no dispersion, so the interval is read without them
    #: and their share is reported beside it.
    n_pinned: int = 0

    @property
    def undefined_sparse(self) -> float:
        """The share of the roster whose quartiles are both zero."""
        if self.n_participants == 0:
            return 0.0
        return 1.0 - self.n_defined / self.n_participants

    @property
    def pinned_share(self) -> float:
        """The share of the roster whose lower quartile is zero."""
        if self.n_participants == 0:
            return 0.0
        return self.n_pinned / self.n_participants

    def as_row(self) -> dict:
        """The record a run stores, so a verdict can be re-read from the file."""
        return {
            "label": self.label,
            "cqv": self.cqv,
            "lower": self.lower,
            "upper": self.upper,
            "n_defined": self.n_defined,
            "n_pinned": self.n_pinned,
            "n_participants": self.n_participants,
            "undefined_sparse": self.undefined_sparse,
            "pinned_share": self.pinned_share,
        }

    def __str__(self) -> str:
        return (
            f"{self.label}: CQV {self.cqv:.3f} [{self.lower:.3f}, {self.upper:.3f}] "
            f"on {self.n_defined} of {self.n_participants}, "
            f"undefined_sparse {self.undefined_sparse:.2f}, "
            f"pinned {self.pinned_share:.2f}"
        )


def population_cqv(
    blocks: Sequence[Sequence[float]],
    label: str = "",
    replicates: int = REPLICATES,
    conf_level: float = CONF_LEVEL,
    seed: int = 0,
) -> Heterogeneity:
    """
    The median participant CQV of one label, with a percentile bootstrap interval.

    The participant is the sampling unit, so a bootstrap draw is a resample of
    participants and never of the epochs inside one.  A draw that defines the
    measure for nobody contributes no median and is skipped, so the interval is
    read over the draws that carry one.

    Args:
        blocks: one array of weekly totals per participant.

    Raises:
        ValueError: when no participant is offered, when the draw count is not
            positive, when the level lies outside the open unit interval, or when
            no participant of the roster defines the measure at all.
    """
    if not blocks:
        raise ValueError("a population CQV needs at least one participant; got 0")
    if replicates < 1:
        raise ValueError(f"a bootstrap needs at least one draw; got {replicates}")
    if not 0.0 < conf_level < 1.0:
        raise ValueError(f"a confidence level lies inside (0, 1); got {conf_level}")
    points = [participant_cqv(block) for block in blocks]
    pinned = [is_pinned(block) for block in blocks]
    n_defined = sum(1 for p in points if p is not None)
    if n_defined == 0:
        raise ValueError(
            f"no participant of {len(blocks)} defines a CQV for {label!r}, so the "
            "label is undefined_sparse on the whole roster"
        )
    # The measure is read over participants that carry an interior statistic.  A
    # pinned participant is defined and contributes nothing but its own
    # sparsity, so it is counted and set aside rather than averaged in.
    interior = np.array(
        [p for p, flat in zip(points, pinned) if p is not None and not flat],
        dtype=float,
    )
    if interior.size == 0:
        raise ValueError(
            f"every one of the {n_defined} participants defining a CQV for "
            f"{label!r} is pinned at 1.0 by a zero lower quartile, so the label "
            "carries no dispersion to compare"
        )
    rng = np.random.default_rng(seed)
    drawn: List[float] = []
    for _ in range(replicates):
        taken = rng.choice(interior, size=interior.size, replace=True)
        drawn.append(float(np.median(taken)))
    tail = (1.0 - conf_level) / 2.0
    return Heterogeneity(
        label=label,
        cqv=float(np.median(interior)),
        lower=float(np.quantile(drawn, tail)),
        upper=float(np.quantile(drawn, 1.0 - tail)),
        n_defined=int(n_defined),
        n_participants=len(blocks),
        replicates=int(replicates),
        seed=int(seed),
        n_pinned=int(sum(pinned)),
    )


def comparable(
    real: Heterogeneity,
    synthetic: Heterogeneity,
    sparse_tolerance: float = SPARSE_TOLERANCE,
) -> bool:
    """
    Whether the two panels describe the same heterogeneity.

    Two conditions, not one.  The intervals have to overlap, and the two panels
    have to define the measure for comparable shares of their rosters: the CQV
    is read over the participants that carry one, so a drawn panel defining it
    for 21 percent of the roster where the real one defines it for 92 has not
    reproduced the heterogeneity however neatly the two intervals sit on top of
    each other.  The overlap alone was the whole verdict, which is why every
    label came back `comparable, separation 0.0`.
    """
    overlap = real.lower <= synthetic.upper and synthetic.lower <= real.upper
    gap = abs(real.undefined_sparse - synthetic.undefined_sparse)
    return overlap and gap <= sparse_tolerance


def sparse_gap(real: Heterogeneity, synthetic: Heterogeneity) -> float:
    """How far apart the two panels' undefined shares stand."""
    return abs(real.undefined_sparse - synthetic.undefined_sparse)


def separation(real: Heterogeneity, synthetic: Heterogeneity) -> float:
    """
    The gap between two intervals, in CQV units, and zero where they overlap.

    A number beside the verdict says how far apart a failing pair stands, which
    an overlap alone never reports.  This reads the intervals only; a pair that
    overlaps here can still be incomparable on `sparse_gap`, and reporting a
    negative overlap as a separation would hide that.
    """
    gap = max(real.lower - synthetic.upper, synthetic.lower - real.upper)
    return float(max(gap, 0.0))


def weekly_blocks(cohort, scale: int, label: int) -> Tuple[np.ndarray, ...]:
    """
    Every participant's totals of one label at one scale of the ladder.

    Args:
        scale: the calendar scale number the panel names, which is one-based.
            The array comes back through `MultiScaleData.get_X`, which is where
            that convention lives, so this reads the scale the panel reader
            reads and no arithmetic here can disagree with it.

    Raises:
        ValueError: when the ladder holds no such scale, or the axis no such
            label, since either would otherwise be read off the end of a list.
    """
    if scale < 1:
        raise ValueError(f"a calendar scale is numbered from one; got {scale}")
    blocks = []
    for name in cohort.subjects:
        data = cohort.open(name)
        if scale > len(data.X):
            raise ValueError(
                f"the ladder holds {len(data.X)} scales and scale {scale} was asked for"
            )
        counts = np.asarray(data.get_X(scale), dtype=float)
        if not 0 <= label < counts.shape[1]:
            raise ValueError(
                f"label {label} is outside the {counts.shape[1]} the axis carries"
            )
        blocks.append(counts[:, label])
    return tuple(blocks)


def compare_heterogeneity(
    real_blocks: Sequence[Sequence[float]],
    synthetic_blocks: Sequence[Sequence[float]],
    label: str = "",
    replicates: int = REPLICATES,
    conf_level: float = CONF_LEVEL,
    seed: int = 0,
) -> Tuple[Heterogeneity, Heterogeneity, bool]:
    """One label's real and drawn heterogeneity, and whether the two intervals meet."""
    left = population_cqv(real_blocks, label, replicates, conf_level, seed)
    right = population_cqv(synthetic_blocks, label, replicates, conf_level, seed + 1)
    return left, right, comparable(left, right)
