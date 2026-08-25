"""
data/lifesnaps_ladder.py
========================
The epoch ladder of the observed log, built on participant time.

Entry is staggered over 224 days, so a shared calendar grid would leave each
participant about three-quarters structurally absent.  The ladder rides on each
participant's own clock instead: day 0 is the date of that participant's first
episode, and the four scales are the minute, the hour, the day and the week.
Each participant is padded up to a whole number of weeks and the padded records
are laid end to end, so no epoch at any scale holds slots from two participants
and every parent map is non-decreasing, which is what a lagged coarse read needs
to stay in the past.

The calendar enters as exogenous context read off the episode timestamps, and
`notification_received` joins it because the study delivers it.  Both sit in one
exogenous block, which is what `FineSources.calendar` holds.

A day carrying no episode is absent, which is the rule the cohort filter of
`lifesnaps` already applies when it divides observed days by the calendar span.
The mask says so at every scale, and the padding days beyond a participant's own
span are absent for the same reason.

The finest grid is large: 51 participants padded to whole weeks give 4060 days,
which is 5.85 million minute slots.  Counts, durations and calendar indices are
whole numbers that `float32` holds exactly, and `participants` limits how much of
the cohort a run reads.
"""

from __future__ import annotations

import collections
import hashlib
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..evaluation.missingness import MissingnessAudit, audit, slot_design
from ..learning.fine_design import ScaleLadder
from ..model.intensity import FineSources
from .label_axis import (
    EXOGENOUS_NAMES,
    SCHEDULED_LABEL,
    CollapseReport,
    LabelAxis,
    collapse_labels,
    label_axis,
)
from .lifesnaps import (
    MIN_COVERAGE,
    MIN_OBSERVED_DAYS,
    RAW_CSV,
    CohortReport,
    LifeSnapsEvent,
    read_lifesnaps,
)

#: The four scales, finest first, and how many minutes one epoch of each holds.
LADDER_UNITS: Tuple[str, ...] = ("minute", "hour", "day", "week")
LADDER_MINUTES: Tuple[int, ...] = (1, 60, 1440, 10080)

MINUTES_PER_DAY = 1440
DAYS_PER_WEEK = 7

#: The gap mechanism the record is audited against.  An absent day is a day the
#: device recorded nothing, and whether the calendar predicts them is what the
#: audit answers.
DECLARED_MECHANISM = "ignorable_given_calendar"


@dataclass(frozen=True)
class ParticipantBlock:
    """Where one participant's own record sits inside the concatenated one."""

    unit: str
    wave: str
    day_zero: date
    span_days: int
    padded_days: int
    observed_days: int
    offsets: Tuple[int, ...]

    @property
    def n_slots(self) -> int:
        return self.padded_days * MINUTES_PER_DAY

    def n_epochs(self, scale: int) -> int:
        """How many epochs of one scale this participant contributes."""
        return self.n_slots // LADDER_MINUTES[scale - 1]


@dataclass(frozen=True)
class LadderRecord:
    """The observed log as the arrays a fit reads, with the mask and the axis."""

    sources: FineSources
    observed: np.ndarray
    blocks: Tuple[ParticipantBlock, ...]
    axis: LabelAxis
    provenance: Dict[str, object]

    @property
    def K(self) -> int:
        return len(LADDER_UNITS)

    @property
    def n_slots(self) -> int:
        return int(self.sources.n_slots)

    def n_epochs(self, scale: int) -> int:
        return int(self.sources.counts[scale - 1].shape[0])

    def ladder(self) -> ScaleLadder:
        """The nesting, with the mask reduced into every scale."""
        return ScaleLadder.from_sources(self.sources, self.observed)

    def scale_of(self, unit: str) -> int:
        """The scale index of a named unit."""
        if unit not in LADDER_UNITS:
            raise KeyError(f"{unit!r} is not one of {list(LADDER_UNITS)}")
        return LADDER_UNITS.index(unit) + 1

    def observed_at(self, scale: int) -> np.ndarray:
        """The mask at one scale, where an epoch is observed when all of it is."""
        who = self.sources.ancestors[scale - 1]
        count = self.n_epochs(scale)
        size = np.bincount(who, minlength=count)
        seen = np.bincount(who, weights=self.observed.astype(float), minlength=count)
        return seen >= size

    def participant_index(self, scale: int) -> np.ndarray:
        """Which participant each epoch of one scale belongs to."""
        out = np.zeros(self.n_epochs(scale), dtype=np.int64)
        for i, block in enumerate(self.blocks):
            start = block.offsets[scale - 1]
            out[start : start + block.n_epochs(scale)] = i
        return out

    def exogenous_at(self, scale: int) -> np.ndarray:
        """The exogenous block of one scale, columns in `LabelAxis.exogenous` order."""
        return self.sources.calendar[scale - 1]


def dataset_digest(path: str = RAW_CSV) -> str:
    """The SHA-256 of the raw file, so a stored result names the bytes it read."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _day_zero(events: Sequence[LifeSnapsEvent]) -> date:
    """The calendar date of a participant's own day 0."""
    first = min(events, key=lambda e: (e.day_index, e.start_min))
    return first.start.date() - timedelta(days=first.day_index)


def _padded_days(span: int) -> int:
    """A participant's span rounded up to a whole number of weeks."""
    return -(-span // DAYS_PER_WEEK) * DAYS_PER_WEEK


def _calendar_rows(block: ParticipantBlock, scale: int) -> np.ndarray:
    """The five calendar columns of one participant at one scale."""
    n = block.n_epochs(scale)
    minutes = np.arange(n, dtype=np.int64) * LADDER_MINUTES[scale - 1]
    day = minutes // MINUTES_PER_DAY
    dates = np.datetime64(block.day_zero, "D") + day.astype("timedelta64[D]")
    out = np.empty((n, 5), dtype=np.float32)
    out[:, 0] = (minutes % MINUTES_PER_DAY) // 60
    out[:, 1] = (block.day_zero.weekday() + day) % DAYS_PER_WEEK
    out[:, 2] = day
    out[:, 3] = day // DAYS_PER_WEEK
    out[:, 4] = dates.astype("datetime64[M]").astype(np.int64) % 12
    return out


def build_ladder(
    participants: Optional[int] = None, path: str = RAW_CSV
) -> Tuple[LadderRecord, CohortReport, CollapseReport]:
    """
    Read the observed log, collapse its three-column artifact and build its ladder.

    Args:
        participants: how many participants of the kept cohort to carry, taken
            in the order the file lists them; None carries all of them.

    Returns:
        The record, the cohort report of the declared rules and the report of
        the collapse.
    """
    raw, cohort = read_lifesnaps(path)
    events, collapse = collapse_labels(raw)
    axis = label_axis(events)

    order: List[str] = []
    by_unit: Dict[str, List[LifeSnapsEvent]] = collections.defaultdict(list)
    for event in events:
        if event.unit not in by_unit:
            order.append(event.unit)
        by_unit[event.unit].append(event)
    if participants is not None:
        order = order[:participants]

    blocks: List[ParticipantBlock] = []
    running = [0] * len(LADDER_UNITS)
    for unit in order:
        own = by_unit[unit]
        seen = {event.day_index for event in own}
        padded = _padded_days(max(seen) + 1)
        block = ParticipantBlock(
            unit=unit,
            wave=own[0].wave,
            day_zero=_day_zero(own),
            span_days=max(seen) + 1,
            padded_days=padded,
            observed_days=len(seen),
            offsets=tuple(running),
        )
        blocks.append(block)
        for k, minutes in enumerate(LADDER_MINUTES):
            running[k] += block.n_slots // minutes

    n_epochs = list(running)
    n_slots = n_epochs[0]
    ancestors: List[np.ndarray] = []
    for k, minutes in enumerate(LADDER_MINUTES):
        who = np.empty(n_slots, dtype=np.int64)
        for block in blocks:
            lo = block.offsets[0]
            local = np.arange(block.n_slots, dtype=np.int64)
            who[lo : lo + block.n_slots] = block.offsets[k] + local // minutes
        ancestors.append(who)

    counts = [np.zeros((n, axis.L), dtype=np.float32) for n in n_epochs]
    durations = [np.zeros((n, axis.L), dtype=np.float32) for n in n_epochs]
    exogenous = [
        np.zeros((n, len(EXOGENOUS_NAMES)), dtype=np.float32) for n in n_epochs
    ]
    observed = np.zeros(n_slots, dtype=bool)
    scheduled = axis.exogenous_index(SCHEDULED_LABEL)

    for block in blocks:
        own = by_unit[block.unit]
        for day in sorted({event.day_index for event in own}):
            lo = block.offsets[0] + day * MINUTES_PER_DAY
            observed[lo : lo + MINUTES_PER_DAY] = True
        for event in own:
            slot = (
                block.offsets[0] + event.day_index * MINUTES_PER_DAY + event.start_min
            )
            target = (
                None
                if event.label == SCHEDULED_LABEL
                else axis.target_index(event.label)
            )
            for k in range(len(LADDER_UNITS)):
                epoch = int(ancestors[k][slot])
                if target is None:
                    exogenous[k][epoch, scheduled] += 1.0
                else:
                    counts[k][epoch, target] += 1.0
                    durations[k][epoch, target] += event.duration_s
        for k in range(len(LADDER_UNITS)):
            start = block.offsets[k]
            rows = _calendar_rows(block, k + 1)
            exogenous[k][start : start + rows.shape[0], :5] = rows

    record = LadderRecord(
        sources=FineSources(
            counts=counts,
            durations=durations,
            calendar=exogenous,
            ancestors=ancestors,
        ),
        observed=observed,
        blocks=tuple(blocks),
        axis=axis,
        provenance=ladder_provenance(path, blocks, axis, cohort, collapse, n_epochs),
    )
    return record, cohort, collapse


def ladder_provenance(
    path: str,
    blocks: Sequence[ParticipantBlock],
    axis: LabelAxis,
    cohort: CohortReport,
    collapse: CollapseReport,
    n_epochs: Sequence[int],
) -> Dict[str, object]:
    """The record of what was read and which rules produced it."""
    return {
        "csv": path,
        "csv_sha256": dataset_digest(path),
        "min_coverage": MIN_COVERAGE,
        "min_observed_days": MIN_OBSERVED_DAYS,
        "participants_raw": cohort.participants_raw,
        "participants_kept": cohort.participants_kept,
        "participants_read": len(blocks),
        "episodes_kept": cohort.n_rows_kept,
        "collapsed_into": collapse.into,
        "collapsed_group": list(collapse.group),
        "collapsed_episodes": collapse.episodes,
        "collapse_agreement": collapse.agreement.share_in_agreement,
        "axis_digest": axis.digest(),
        "targets": list(axis.targets),
        "exogenous": list(axis.exogenous),
        "units": list(LADDER_UNITS),
        "n_epochs": [int(n) for n in n_epochs],
    }


def _first_slot_of(ancestor: np.ndarray, n_epochs: int) -> np.ndarray:
    """The first finest slot of each epoch of one scale."""
    out = np.zeros(n_epochs, dtype=np.int64)
    out[ancestor[::-1]] = np.arange(ancestor.size, dtype=np.int64)[::-1]
    return out


def missingness_audit(
    record: LadderRecord,
    unit: str = "day",
    n_permutations: int = 199,
    seed: int = 0,
) -> MissingnessAudit:
    """
    Judge the record's absent epochs against the mechanism the ladder declares.

    The mask is a day-level object, so the audit runs on the day grid and reads
    the day and week exogenous blocks as its context.
    """
    scale = record.scale_of(unit)
    mask = record.observed_at(scale)
    first = _first_slot_of(record.sources.ancestors[scale - 1], mask.size)
    ancestors = [
        np.arange(mask.size, dtype=np.int64),
        record.sources.ancestors[scale][first],
    ]
    blocks = [record.exogenous_at(scale), record.exogenous_at(scale + 1)]
    design = slot_design(blocks, ancestors)
    return audit(
        DECLARED_MECHANISM,
        mask,
        design,
        labels=[f"{unit} coverage"],
        n_permutations=n_permutations,
        seed=seed,
    )
