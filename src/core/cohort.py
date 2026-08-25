"""
core/cohort.py
==============
The participant axis, held where the axis has behavior.

A field on `MultiScaleData` would be read by no method of that container and
would say nothing about the memory the axis creates: one subject's finest counts
and durations are 126 MB at a one-minute slot, so thirty of them resident is
3.8 GB.  The axis is a container of its own because it owns the two halves of
that fact.  What every subject shares is held once: the time ladder, the label
axis, the calendar, the ancestor maps and the calendar blocks.  What differs is
held only where it is cheap: the (S, H_1) observation mask, 3.9 MB at the
benchmark's shape.  Everything expensive arrives through a reader, one subject
at a time.  Whether it is dropped before the next one opens is the caller's,
which `stream` says plainly, since a loop variable bound to the record just
visited keeps it alive while the reader builds the next.

The mean over participants lives here and never in the objective.  A fit sums
per-subject log-likelihood terms, because the log of a product of independent
records is a sum.  A population count is a mean, because a total over
participants changes meaning when the roster changes size.  Its denominator is
the per-epoch participant count taken from the masks, so an epoch nobody
observed is NaN and never a fabricated zero, and a subject absent at one epoch
lowers that epoch's denominator alone.

`check_ladder` holds per subject, which is what makes the mean reconcile.  A
coarse epoch is fully observed or fully unobserved for each subject, so the
participant set is constant inside it and a parent's denominator equals each of
its children's.  Under staggered enrollment a window opens mid-week and leaves a
half-covered coarsest epoch; `whole_epochs` rounds that epoch to unobserved and
reports the slots it cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np

from ..model.intensity import FineSources, ancestor_maps
from .calendar import SlotCalendar
from .mask import check_ladder, reduce_by
from .scales import ScaleHierarchy
from .typed_graph import COUNT, QUANTITIES
from .variables import MultiScaleData


def whole_epochs(
    masks: np.ndarray, hierarchy: ScaleHierarchy
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Masks with every partly covered coarsest epoch rounded to unobserved, and the cost.

    A window that opens mid-week leaves a coarsest epoch holding both covered and
    uncovered slots, and `check_ladder` refuses one, because the epoch's stored
    count is a sum over the covered children alone and no reader of a parent
    total can tell such a sum from a whole one.  Rounding the partial epoch down
    is the one resolution that leaves every stored total whole.  It throws away
    recorded slots, so the count of them comes back with the masks.

    Args:
        masks: (S, H_1) bool, one row per subject.
        hierarchy: the ladder the rounding is taken on.

    Returns:
        The rounded masks and the observed slots each subject lost.
    """
    flags = np.asarray(masks, dtype=bool)
    who = ancestor_maps(hierarchy.parent_maps[:-1], hierarchy.n_epochs(1))[-1]
    size = hierarchy.n_epochs(hierarchy.K)
    out = np.stack([row & reduce_by(row, who, size)[who] for row in flags])
    return out, (flags & ~out).sum(axis=1)


@dataclass
class Cohort:
    """
    S subjects on one time ladder and one label axis.

    Attributes
    ----------
    subjects         : the roster, sorted, which fixes the accumulation order and
                       makes a floating-point sum over subjects reproducible
    hierarchy        : the one time ladder every subject is laid on
    label_names      : the one label axis every subject carries in full
    reader           : produces one subject's arrays on demand
    masks            : (S, H_1) bool in `subjects` order, the only per-subject
                       array held resident
    calendar         : the clock behind the finest slots, shared
    calendar_columns : the encoded calendar block of each scale, shared, so a
                       per-subject source object references these and carries
                       only that subject's counts and durations
    subject_calendar : whether each subject brings its own exogenous block in
                       `MultiScaleData.C`.  An exogenous quantity a study
                       delivers on its own schedule differs from subject to
                       subject, so it rides with the subject and the shared
                       block cannot hold it
    """

    subjects: Tuple[str, ...]
    hierarchy: ScaleHierarchy
    label_names: List[str]
    reader: Callable[[str], MultiScaleData]
    masks: np.ndarray
    calendar: Optional[SlotCalendar] = None
    calendar_columns: Optional[List[np.ndarray]] = None
    subject_calendar: bool = False
    _ancestors: Optional[List[np.ndarray]] = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        """
        Check the roster and the mask block, and check the ladder of every subject.

        Raises:
            ValueError: on an empty roster, a repeated id, a roster out of order,
                a mask block that is not (S, H_1) bool, a calendar block set that
                does not cover the ladder, or a subject whose mask half-covers a
                coarse epoch.
        """
        self.subjects = tuple(str(name) for name in self.subjects)
        if not self.subjects:
            raise ValueError(
                "a cohort of no subjects has an empty denominator at every epoch, "
                "so no population quantity it reports is defined"
            )
        if len(set(self.subjects)) != len(self.subjects):
            raise ValueError(
                f"the roster repeats an id: {sorted(set(self.subjects))} distinct "
                f"of {len(self.subjects)}, so one mask row stands for two subjects"
            )
        if list(self.subjects) != sorted(self.subjects):
            raise ValueError(
                "the roster is out of order; sorting it here would permute the "
                "roster away from the mask rows, so the caller sorts both together"
            )
        self.masks = np.asarray(self.masks)
        if self.masks.dtype != np.bool_:
            raise ValueError(
                f"an observation mask block is boolean; this one has dtype "
                f"{self.masks.dtype}, which every reader would take for a flag "
                "wherever it is non-zero"
            )
        expected = (len(self.subjects), self.hierarchy.n_epochs(1))
        if self.masks.shape != expected:
            raise ValueError(
                f"a mask block carries one row per subject and one flag per finest "
                f"slot, so {expected}; this one is {self.masks.shape}"
            )
        if self.calendar_columns is None:
            self.calendar_columns = [
                np.zeros((self.hierarchy.n_epochs(k), 0)) for k in range(1, self.K + 1)
            ]
        elif len(self.calendar_columns) != self.K:
            raise ValueError(
                f"the cohort carries {len(self.calendar_columns)} calendar blocks "
                f"for a ladder of {self.K} scales"
            )
        for name, row in zip(self.subjects, self.masks):
            try:
                check_ladder(row, self.hierarchy.parent_maps)
            except ValueError as exc:
                raise ValueError(f"subject {name!r}: {exc}") from exc

    # ------------------------------------------------------------------
    @property
    def S(self) -> int:
        return len(self.subjects)

    @property
    def K(self) -> int:
        return self.hierarchy.K

    @property
    def L(self) -> int:
        return len(self.label_names)

    @property
    def n_slots(self) -> int:
        return self.hierarchy.n_epochs(1)

    @property
    def ancestors(self) -> List[np.ndarray]:
        """The scale-k epoch of every finest slot, derived once for the cohort."""
        if self._ancestors is None:
            self._ancestors = ancestor_maps(
                self.hierarchy.parent_maps[:-1], self.n_slots
            )
        return self._ancestors

    # ------------------------------------------------------------------
    def open(self, subject: str) -> MultiScaleData:
        """
        One subject's arrays through the reader, checked against the shared axes.

        The reader states what the subject's own record covers and the cohort
        states the window it is watched over.  A slot needs both, so the two
        masks meet in an intersection here; taking either one alone would call a
        slot recorded that one of the two says nobody watched.

        Raises:
            ValueError: when the id is off the roster, or the arrays handed back
                sit on some other ladder or label axis than the cohort's.
        """
        if subject not in self.subjects:
            raise ValueError(
                f"subject {subject!r} is not one of the {self.S} on the roster"
            )
        data = self.reader(subject)
        if list(data.hierarchy.H) != list(self.hierarchy.H):
            raise ValueError(
                f"subject {subject!r} came back on a ladder of {list(data.hierarchy.H)} "
                f"epochs where the cohort shares {list(self.hierarchy.H)}"
            )
        if list(data.label_names) != list(self.label_names):
            raise ValueError(
                f"subject {subject!r} came back on {len(data.label_names)} labels "
                f"where the cohort shares {self.L}, so one column index names two "
                "different labels"
            )
        row = self.masks[self.subjects.index(subject)]
        return replace(
            data, observed=row if data.observed is None else data.observed & row
        )

    def stream(self) -> Iterator[Tuple[str, MultiScaleData]]:
        """
        Every subject in roster order, opened as it is reached.

        Nothing here keeps a record once it is yielded.  The caller's own binding
        does: the loop variable still names the subject just visited while this
        generator resumes and the reader builds the next, so two records are
        resident at that moment.  A caller that has to hold one at a time opens
        the subjects itself, inside a frame whose locals die on return, which is
        what `PooledFit._one_subject` is.
        """
        for name in self.subjects:
            yield name, self.open(name)

    def sources(self, data: MultiScaleData) -> FineSources:
        """One subject's arrays as fine sources, over the maps the cohort shares."""
        return FineSources(
            counts=list(data.X),
            durations=list(data.Y),
            calendar=list(data.C) if self.subject_calendar else self.calendar_columns,
            ancestors=self.ancestors,
        )

    # ------------------------------------------------------------------
    def observed_at(self, k: int = 1) -> np.ndarray:
        """
        (S, H_k) bool, each subject's mask AND-reduced into the epochs of scale k.

        Derived from the finest masks and never stored, so a coarse flag cannot
        contradict the fine one it came from.
        """
        if k == 1:
            return self.masks
        who = self.ancestors[k - 1]
        size = self.hierarchy.n_epochs(k)
        return np.stack([reduce_by(row, who, size) for row in self.masks])

    def participants(self, k: int = 1) -> np.ndarray:
        """(H_k,) int, how many subjects were observed at each epoch of scale k."""
        return self.observed_at(k).sum(axis=0)

    def population(self, k: int = 1, quantity: str = COUNT) -> np.ndarray:
        """
        The per-participant mean of one quantity at scale k, streamed over the cohort.

        This is a reporting operator and no term of any objective.  The
        denominator is the participant count of the epoch itself, so a subject
        absent there lowers that epoch alone, and an epoch nobody observed comes
        back NaN, which no later reader can mistake for a recorded zero.

        Raises:
            ValueError: when the quantity is neither the count nor the duration.
        """
        if quantity not in QUANTITIES:
            raise ValueError(
                f"quantity {quantity!r}; expected one of {list(QUANTITIES)}"
            )
        seen = self.observed_at(k)
        total = np.zeros((self.hierarchy.n_epochs(k), self.L))
        for index, (_, data) in enumerate(self.stream()):
            arrays = data.X if quantity == COUNT else data.Y
            total += np.asarray(arrays[k - 1], dtype=float) * seen[index][:, None]
        denom = seen.sum(axis=0).astype(float)[:, None]
        out = np.full(total.shape, np.nan)
        return np.divide(total, denom, out=out, where=denom > 0)

    def reconcile(self) -> Dict[str, Dict[str, float]]:
        """
        Every subject's reconciliation error, taken on that subject's own arrays.

        The identity is that one record's coarse total is the sum of its own fine
        ones.  A pooled array of the cohort satisfies no such thing and would only
        show that summation commutes, so the check runs over the stream and one
        subject at a time.
        """
        return {name: data.check_reconciliation() for name, data in self.stream()}
