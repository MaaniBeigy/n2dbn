"""
data/lifesnaps_panel.py
=======================
The balanced panel of the observed log, as the cohort a pooled fit streams.

A pooled fit lays every subject on one ladder, so the window has to be the same
length for all of them.  The window here is each participant's **first eight
complete calendar weeks from their own first Monday**: 80,640 minute slots,
1344 hours, 56 days and 8 weeks.  Every one of the 51 kept participants clears
it, at a median coverage of 1.00 and a minimum of 0.88.

Starting at a Monday is what lets the weekday index be shared. In participant
time an hour index, a day index and a week index are the same function of the
epoch for everybody; a weekday index is only if every window opens on the same
weekday. Aligning the window buys that, and it also makes a week epoch a real
calendar week, which is what a weekly rhythm means. `month_index` stays out:
eight weeks is under two cycles of it, and the plan already reports monthly
periodicity descriptively for that reason.

Each periodic clock node carries a cosine and a sine at its own period beside its
index. An index reaches a predictor as a level, so the log-odds it buys are
linear in the hour of the day and monotone across it, and no daily peak is
reachable. A cosine and a sine at one period span every phase of one cycle, so
the pair is what carries a profile with a peak in it. `day_index` and
`week_index` are unbounded and have no period to take a harmonic of.

`notification_received` cannot be shared. The study delivers it, so it is
exogenous, and how much of it a participant received is that participant's own
number. It rides in each subject's own block, which is what `subject_calendar`
on the cohort says.

A day the record does not cover is absent, which is the rule the cohort filter
of `lifesnaps` already applies. Inside the window that leaves the mask whole
weeks at every scale, which is what a cohort ladder needs.

Durations are stored in finest slots, one minute each. `intensity.running_mask`
reads a stored duration as a span of slots, so storing seconds put a label
inside an episode for sixty times as long as it lasted: on the observed log the
`sleep` column was at risk in 0.05% of its slots and its count model was fitted
on almost nothing. An episode occupies at least one slot, so a one-second
instant stores one.

The observation schedule is exogenous. A study decides which days it watches,
and `PanelSchedule` carries that decision so a log drawn from a fit can be laid
on the same window and the same mask as the record the fit read, without the
reader inferring a mask from whichever days a draw happened to fill.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..core.cohort import Cohort, whole_epochs
from ..core.scales import ScaleHierarchy
from ..core.variables import MultiScaleData
from .label_axis import (
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
from .lifesnaps_ladder import DAYS_PER_WEEK, MINUTES_PER_DAY, _day_zero, dataset_digest

#: Seconds in one finest slot of the panel ladder.
SECONDS_PER_SLOT = 60

#: The window every subject is laid on, and the branching that makes its ladder.
PANEL_WEEKS = 8
PANEL_BRANCHING: Tuple[int, ...] = (60, 24, DAYS_PER_WEEK)

#: The clock nodes the panel shares, in the column order of the exogenous block.
#: The scheduled label follows them and is the subject's own.
PANEL_NODES: Tuple[str, ...] = (
    "hour_index",
    "hour_cos",
    "hour_sin",
    "weekday_index",
    "weekday_cos",
    "weekday_sin",
    "day_index",
    "week_index",
)
PANEL_EXOGENOUS: Tuple[str, ...] = PANEL_NODES + (SCHEDULED_LABEL,)


@dataclass(frozen=True)
class PanelSubject:
    """One participant's window inside its own record."""

    unit: str
    wave: str
    offset_days: int
    observed_days: int
    window_days: int

    @property
    def coverage(self) -> float:
        return self.observed_days / self.window_days


@dataclass(frozen=True)
class PanelSchedule:
    """
    Which days each subject is watched over, and where its window opens.

    A study decides what it watches, so the schedule is exogenous and a log
    drawn from a fit inherits it. Handing it back to the reader is what makes a
    drawn log's panel the same window, the same roster and the same mask as the
    record the fit read.

    Attributes
    ----------
    weeks     : the window length the schedule was taken over
    offsets   : days from each subject's own day 0 to the first slot of its
                window
    days      : the raw day indices of each subject that the panel observes
    day_zero  : the calendar date of each subject's own day 0, which a writer
                needs to stamp a drawn episode
    axis      : the vocabulary the study records, which a drawn log inherits.
                A label a draw happens to leave empty is still a column of the
                panel, so the axis is declared and never inferred from the rows
                one log holds
    """

    weeks: int
    offsets: Mapping[str, int]
    days: Mapping[str, Tuple[int, ...]]
    day_zero: Mapping[str, date]
    axis: Optional[LabelAxis] = None


@dataclass(frozen=True)
class PanelReport:
    """What the window did to the roster, so a run reports it without recomputing."""

    weeks: int
    subjects: Tuple[PanelSubject, ...]
    dropped: Tuple[Tuple[str, int, int], ...]
    rounded_slots: Tuple[int, ...] = ()
    schedule: Optional[PanelSchedule] = None

    @property
    def n_days(self) -> int:
        return self.weeks * DAYS_PER_WEEK

    def summary(self) -> str:
        coverage = sorted(s.coverage for s in self.subjects)
        lines = [
            f"balanced panel: {self.weeks} weeks from each participant's first "
            f"Monday, {len(self.subjects)} subjects",
            f"  coverage min {coverage[0]:.2f} median "
            f"{coverage[len(coverage) // 2]:.2f} max {coverage[-1]:.2f}",
        ]
        lost = sum(self.rounded_slots)
        if lost:
            lines.append(
                f"  {lost} observed minute slots dropped by rounding a partly "
                f"covered week down, across "
                f"{sum(1 for n in self.rounded_slots if n)} subjects"
            )
        if self.dropped:
            lines.append("  dropped for a window longer than their record:")
            lines.extend(
                f"    {unit}  span {span}  needs {need}"
                for unit, span, need in self.dropped
            )
        return "\n".join(lines)


def panel_hierarchy(weeks: int = PANEL_WEEKS) -> ScaleHierarchy:
    """The shared four-scale ladder of one panel window."""
    return ScaleHierarchy.from_uniform_branching(
        weeks * DAYS_PER_WEEK * MINUTES_PER_DAY, list(PANEL_BRANCHING)
    )


def shared_calendar(weeks: int = PANEL_WEEKS) -> List[np.ndarray]:
    """
    The clock nodes of every scale, which every subject reads the same way.

    Each column takes the value at the epoch's first minute, so a node is
    constant inside a parent block exactly where the invariance argument says it
    carries nothing at that scale.
    """
    out: List[np.ndarray] = []
    for minutes in (1, 60, MINUTES_PER_DAY, MINUTES_PER_DAY * DAYS_PER_WEEK):
        n = weeks * DAYS_PER_WEEK * MINUTES_PER_DAY // minutes
        start = np.arange(n, dtype=np.int64) * minutes
        day = start // MINUTES_PER_DAY
        hour = (start % MINUTES_PER_DAY) // 60
        weekday = day % DAYS_PER_WEEK
        block = np.empty((n, len(PANEL_NODES)), dtype=np.float32)
        block[:, 0] = hour
        block[:, 1] = np.cos(2.0 * np.pi * hour / 24.0)
        block[:, 2] = np.sin(2.0 * np.pi * hour / 24.0)
        block[:, 3] = weekday
        block[:, 4] = np.cos(2.0 * np.pi * weekday / DAYS_PER_WEEK)
        block[:, 5] = np.sin(2.0 * np.pi * weekday / DAYS_PER_WEEK)
        block[:, 6] = day
        block[:, 7] = day // DAYS_PER_WEEK
        out.append(block)
    return out


def panel_axis(axis: LabelAxis) -> LabelAxis:
    """The axis restricted to the exogenous sources the panel carries."""
    return LabelAxis(
        targets=axis.targets, exogenous=PANEL_EXOGENOUS, grouped=axis.grouped
    )


def _first_monday_offset(events: Sequence[LifeSnapsEvent]) -> int:
    """Days from a participant's own day 0 to their first Monday."""
    return (7 - _day_zero(events).weekday()) % 7


def duration_slots(event: LifeSnapsEvent) -> int:
    """An episode's span in finest slots, which is one minute on this ladder."""
    return max(1, int(round(event.duration_s / SECONDS_PER_SLOT)))


def _subject_arrays(
    own: Sequence[LifeSnapsEvent],
    offset: int,
    axis: LabelAxis,
    weeks: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    """One subject's counts, durations and receipt column, per scale."""
    n_days = weeks * DAYS_PER_WEEK
    sizes = [
        n_days * MINUTES_PER_DAY,
        n_days * 24,
        n_days,
        weeks,
    ]
    minutes = (1, 60, MINUTES_PER_DAY, MINUTES_PER_DAY * DAYS_PER_WEEK)
    counts = [np.zeros((n, axis.L), dtype=np.int64) for n in sizes]
    durations = [np.zeros((n, axis.L), dtype=np.int64) for n in sizes]
    received = [np.zeros((n, 1), dtype=np.float32) for n in sizes]
    for event in own:
        day = event.day_index - offset
        if not 0 <= day < n_days:
            continue
        slot = day * MINUTES_PER_DAY + event.start_min
        span = duration_slots(event)
        for k, step in enumerate(minutes):
            epoch = slot // step
            if event.label == SCHEDULED_LABEL:
                received[k][epoch, 0] += 1.0
            else:
                column = axis.target_index(event.label)
                counts[k][epoch, column] += 1
                durations[k][epoch, column] += span
    return counts, durations, received


def balanced_panel(
    participants: Optional[int] = None,
    weeks: int = PANEL_WEEKS,
    path: str = RAW_CSV,
    schedule: Optional[PanelSchedule] = None,
) -> Tuple[Cohort, LabelAxis, PanelReport, CohortReport, CollapseReport]:
    """
    Read a log and lay its balanced panel out as a cohort.

    Args:
        participants: how many of the kept cohort to carry, in the order the
            file lists them; None carries all of them.
        weeks: the window length, in complete calendar weeks from each
            participant's own first Monday.
        schedule: a declared observation schedule, which names the roster, the
            window of each subject and the days it is watched over. The cohort
            filter and the first-Monday rule already ran on the record that
            produced it, so they do not run again here.

    Returns:
        The cohort, the axis its exogenous block is keyed on, the panel report
        and the two reports of the rules that produced the log.

    Raises:
        ValueError: when a schedule was taken over a different window, when the
            log holds no rows for a subject the schedule names, or when no
            record is long enough to carry the window.
    """
    if schedule is not None and int(schedule.weeks) != int(weeks):
        raise ValueError(
            f"the schedule was taken over {schedule.weeks} weeks and this panel "
            f"is {weeks}, so its day indices name other epochs"
        )
    filters = (0.0, 0) if schedule is not None else (MIN_COVERAGE, MIN_OBSERVED_DAYS)
    raw, cohort_report = read_lifesnaps(path, *filters)
    events, collapse = collapse_labels(raw)
    declared = None if schedule is None else schedule.axis
    axis = panel_axis(label_axis(events)) if declared is None else declared

    order: List[str] = []
    by_unit: Dict[str, List[LifeSnapsEvent]] = collections.defaultdict(list)
    for event in events:
        if event.unit not in by_unit:
            order.append(event.unit)
        by_unit[event.unit].append(event)
    if participants is not None:
        order = order[:participants]
    if schedule is not None:
        absent = sorted(set(schedule.offsets) - set(by_unit))
        if absent:
            raise ValueError(
                f"the schedule names {len(absent)} subject(s) the log holds no row "
                f"for, first {absent[0]!r}, so their window would be empty"
            )
        order = [unit for unit in order if unit in schedule.offsets]

    n_days = weeks * DAYS_PER_WEEK
    subjects: List[PanelSubject] = []
    dropped: List[Tuple[str, int, int]] = []
    offsets: Dict[str, int] = {}
    watched: Dict[str, Tuple[int, ...]] = {}
    for unit in order:
        own = by_unit[unit]
        seen = {event.day_index for event in own}
        span = max(seen) + 1
        if schedule is None:
            offset = _first_monday_offset(own)
            if offset + n_days > span:
                dropped.append((unit, span, offset + n_days))
                continue
            days = tuple(sorted(seen & set(range(offset, offset + n_days))))
        else:
            offset = int(schedule.offsets[unit])
            days = tuple(schedule.days[unit])
        offsets[unit] = offset
        watched[unit] = days
        subjects.append(
            PanelSubject(
                unit=unit,
                wave=own[0].wave,
                offset_days=offset,
                # What the record covers inside the window, which the mask then
                # rounds down; the schedule states the mask and this states the
                # record, so the field means one thing on either path.
                observed_days=len(seen & set(range(offset, offset + n_days))),
                window_days=n_days,
            )
        )

    if not offsets:
        raise ValueError(
            f"a window of {weeks} weeks opening on a Monday is longer than every "
            f"record read, so the panel holds no subject"
        )
    roster = tuple(sorted(offsets))
    hierarchy = panel_hierarchy(weeks)
    shared = shared_calendar(weeks)
    masks = np.zeros((len(roster), n_days * MINUTES_PER_DAY), dtype=bool)
    for row, unit in enumerate(roster):
        offset = offsets[unit]
        for day in watched[unit]:
            local = day - offset
            if 0 <= local < n_days:
                lo = local * MINUTES_PER_DAY
                masks[row, lo : lo + MINUTES_PER_DAY] = True

    # A coarsest epoch holding both covered and uncovered slots stores a total
    # over its covered children alone, and no reader of that total can tell it
    # from a whole one, so a partly covered week rounds down.
    masks, rounded = whole_epochs(masks, hierarchy)

    def reader(unit: str) -> MultiScaleData:
        counts, durations, received = _subject_arrays(
            by_unit[unit], offsets[unit], axis, weeks
        )
        return MultiScaleData(
            hierarchy=hierarchy,
            label_names=list(axis.targets),
            X=counts,
            Y=durations,
            C=[np.concatenate([a, b], axis=1) for a, b in zip(shared, received)],
        )

    cohort = Cohort(
        subjects=roster,
        hierarchy=hierarchy,
        label_names=list(axis.targets),
        reader=reader,
        masks=masks,
        calendar_columns=shared,
        subject_calendar=True,
    )
    kept = tuple(sorted(set(roster)))
    report = PanelReport(
        weeks=weeks,
        subjects=tuple(s for s in subjects),
        dropped=tuple(dropped),
        rounded_slots=tuple(int(n) for n in rounded),
        schedule=PanelSchedule(
            weeks=weeks,
            offsets={unit: offsets[unit] for unit in kept},
            # The mask is what a later reader has to reproduce, and rounding a
            # partly covered week down already moved it away from the days the
            # log lists, so the schedule states the rounded days.
            days={
                unit: tuple(
                    offsets[unit] + local
                    for local in range(n_days)
                    if masks[row, local * MINUTES_PER_DAY]
                )
                for row, unit in enumerate(roster)
            },
            day_zero={unit: _day_zero(by_unit[unit]) for unit in kept},
            axis=axis,
        ),
    )
    return cohort, axis, report, cohort_report, collapse


def panel_provenance(
    cohort: Cohort,
    axis: LabelAxis,
    report: PanelReport,
    path: str = RAW_CSV,
) -> Dict[str, object]:
    """The record of which window, which subjects and which bytes a fit read."""
    return {
        "csv": path,
        "csv_sha256": dataset_digest(path),
        "window_weeks": report.weeks,
        "window_days": report.n_days,
        "n_subjects": cohort.S,
        "dropped": [list(row) for row in report.dropped],
        "coverage_mean": float(
            np.mean([s.coverage for s in report.subjects]) if report.subjects else 0.0
        ),
        "axis_digest": axis.digest(),
        "targets": list(axis.targets),
        "exogenous": list(axis.exogenous),
        "n_epochs": [cohort.hierarchy.n_epochs(k) for k in range(1, cohort.K + 1)],
    }


def offset_days(report: PanelReport) -> Dict[str, int]:
    """Where each subject's window opens inside its own record."""
    return {s.unit: s.offset_days for s in report.subjects}


def wave_of(report: PanelReport) -> Dict[str, str]:
    """Which recruitment wave each subject belongs to, which R7 reads as a block."""
    return {s.unit: s.wave for s in report.subjects}
