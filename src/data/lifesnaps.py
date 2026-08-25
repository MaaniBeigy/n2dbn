"""
data/lifesnaps.py
=================
Read the observed LifeSnaps event log into the project's event-log schema.

The raw file is `data/observed_lifesnaps_event_log.csv`: 34615 episodes, 60
participants, 25 labels, with `person_id`, `event`, `day_index`, `start_min`,
`duration_min`, `start_datetime` and `end_datetime`.  Four properties of it
need a declared rule before any fit reads it, and this module holds all four so
that the rule is data and not a constant buried in an experiment.  Applying
them leaves 30409 episodes, 51 participants and nine labels.

Two clocks
----------
Entry is staggered.  First-event dates span 224 days across two waves, and
`day_index` counts days since *that participant's own* first event.  A shared
calendar grid would leave each participant about three-quarters structurally
absent, so the epoch ladder rides on participant time and the calendar enters
as exogenous context read off `start_datetime`.  `wave` is carried beside it,
because the spring and autumn cohorts differ in season and daylight.

Which calendar node is identified where follows from the block-constant
invariance of the child kernel: a covariate that is constant across the
children of a parent epoch contributes exactly zero to that kernel, so a
calendar node is identified at the finest scale at which it still varies
inside its parent block.  `CALENDAR_NODES` records that scale for each.

Duration lives in seconds, epochs in minutes
--------------------------------------------
28081 of 34615 rows carry `duration_min` zero.  They are instants, and on a
minute lattice the one-slot floor would invent 28081 minutes that were never
observed.  Durations are therefore counted in **seconds** while epochs are
counted in minutes.  The two units are independent: aggregation needs only that
duration sums over children, which it does in any unit.

Every real duration is a whole number of minutes, hence a multiple of 60
seconds.  An injected duration is 1 or 2 seconds.  The two sets never collide,
so the injection is auditable after the fact and invertible.  The same rule is
applied to a generated log, so a real-versus-synthetic duration comparison on
these labels is not biased by the injection.

Labels
------
Seventeen of the 25 raw labels are exercise modalities, one of them with a
single event, and they merge into `gym`.  After the participant filter what
survives is nine labels, four of them instantaneous, every one carrying at
least 100 episodes across at least 20 participants.
"""

from __future__ import annotations

import collections
import csv
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Sequence, Set, Tuple

# ---------------------------------------------------------------------------
# The declared rules.  Every number here is measured on the shipped file, so a
# change to one of them is a change to what the cohort is.

RAW_CSV = "data/observed_lifesnaps_event_log.csv"

#: Exercise modalities that merge into one label.  Walking is locomotion and
#: keeps its own label; mindfulness is not physical exercise and keeps its own.
GYM_MODALITIES: Set[str] = {
    "bike",
    "workout",
    "sport",
    "aerobic_workout",
    "run",
    "swim",
    "yoga_pilates",
    "weights",
    "circuit_training",
    "treadmill",
    "hike",
    "martial_arts",
    "elliptical",
    "interval_workout",
    "tennis",
    "bootcamp",
    "spinning",
}
GYM_LABEL = "gym"

#: Labels whose episodes are instants.  They keep a duration channel so that
#: `Y >= X` holds and the channel is well posed.  That channel is excluded from
#: duration scoring, since the count is what the model is asked to predict.
INSTANT_LABELS: Set[str] = {
    "notification_received",
    "notification_read",
    "mood_survey",
    "step_goal_survey",
}

#: Seconds given to an instant so its duration is nonzero and predictable.  A
#: real duration is a multiple of 60, so 1 and 2 can never be mistaken for one.
INJECTED_SECONDS: Tuple[int, ...] = (1, 2)

#: A participant is kept when it clears both bars.  The coverage distribution
#: is discontinuous at 0.90.  43 participants sit at 0.95 or above, 8 more
#: between 0.90 and 0.95, and the next one down is 0.86, so the cut falls in a
#: gap.  The day bar is eight complete weeks,
#: because the week scale is the coarsest the ladder carries and a participant
#: contributing fewer than eight of its epochs cannot inform it.
MIN_COVERAGE = 0.90
MIN_OBSERVED_DAYS = 56  # eight complete weeks

#: Calendar nodes, with the scale at which each is identified.  A node is
#: identified at the finest scale where it still varies inside its parent
#: block, which is the block-constant invariance argument applied to the
#: calendar.  `period` is None for an unbounded index.
CALENDAR_NODES: Dict[str, Dict[str, object]] = {
    "hour_index": {"period": 24, "identified_at": "hour", "varies_within": "day"},
    # An index enters a predictor as a level, so the log-odds it buys are linear
    # in the hour and monotone across the day, and no daily peak is reachable.
    # A cosine and a sine at the same period span every phase of one cycle, so
    # the pair is what carries a profile with a peak in it.
    "hour_cos": {"period": 24, "identified_at": "hour", "varies_within": "day"},
    "hour_sin": {"period": 24, "identified_at": "hour", "varies_within": "day"},
    "weekday_index": {"period": 7, "identified_at": "day", "varies_within": "week"},
    "weekday_cos": {"period": 7, "identified_at": "day", "varies_within": "week"},
    "weekday_sin": {"period": 7, "identified_at": "day", "varies_within": "week"},
    "day_index": {"period": None, "identified_at": "day", "varies_within": "week"},
    "week_index": {"period": None, "identified_at": "week", "varies_within": "horizon"},
    "month_index": {"period": 12, "identified_at": "day", "varies_within": "horizon"},
}

_RAW_FORMAT = "%m/%d/%Y %H:%M"


# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LifeSnapsEvent:
    """One episode, after the declared rules have been applied."""

    unit: str
    label: str
    day_index: int
    start_min: int
    duration_s: int
    start: datetime
    wave: str
    duration_source: str  # "observed", "instant" or "repaired"

    @property
    def injected(self) -> bool:
        return self.duration_source != "observed"

    @property
    def hour_index(self) -> int:
        return self.start.hour

    @property
    def weekday_index(self) -> int:
        return self.start.weekday()

    @property
    def hour_cos(self) -> float:
        return math.cos(2.0 * math.pi * self.hour_index / 24.0)

    @property
    def hour_sin(self) -> float:
        return math.sin(2.0 * math.pi * self.hour_index / 24.0)

    @property
    def weekday_cos(self) -> float:
        return math.cos(2.0 * math.pi * self.weekday_index / 7.0)

    @property
    def weekday_sin(self) -> float:
        return math.sin(2.0 * math.pi * self.weekday_index / 7.0)

    @property
    def week_index(self) -> int:
        return self.day_index // 7

    @property
    def month_index(self) -> int:
        return self.start.month - 1


@dataclass(frozen=True)
class CohortReport:
    """What the declared rules did, so a run can report it without recomputing."""

    n_rows_raw: int
    n_rows_kept: int
    participants_raw: int
    participants_kept: int
    dropped: Tuple[Tuple[str, float, int], ...]
    merged_into_gym: int
    injected_instants: int
    repaired_zeros: int
    labels: Tuple[Tuple[str, int, int, bool], ...]

    def summary(self) -> str:
        lines = [
            f"rows {self.n_rows_raw} -> {self.n_rows_kept}",
            f"participants {self.participants_raw} -> {self.participants_kept}",
            f"merged into {GYM_LABEL}: {self.merged_into_gym} episodes",
            f"instants given {INJECTED_SECONDS} s: {self.injected_instants}",
            f"zero durations repaired on non-instant labels: {self.repaired_zeros}",
            "dropped participants (coverage, observed days):",
        ]
        lines.extend(f"  {unit}  {cov:.2f}  {days}" for unit, cov, days in self.dropped)
        lines.append("labels (episodes, participants, instantaneous):")
        lines.extend(
            f"  {name:<24}{n:>7}{p:>5}   {inst}" for name, n, p, inst in self.labels
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------


def _wave(first: datetime) -> str:
    """Which recruitment wave a participant belongs to."""
    return "spring" if first.month <= 6 else "autumn"


def _injected_seconds(unit: str, day_index: int, start_min: int, label: str) -> int:
    """
    A deterministic duration for one instant.

    The value depends only on the episode's own identity, so it is stable
    across runs and across machines without carrying a generator, and it is
    reproduced exactly when the same episode is read again.
    """
    key = f"{unit}|{label}|{day_index}|{start_min}"
    return INJECTED_SECONDS[hash_key(key) % len(INJECTED_SECONDS)]


def hash_key(key: str) -> int:
    """
    A stable hash.

    `hash` is randomized per process, so it cannot decide a stored value.  This
    is the FNV-1a 64-bit hash, which is short enough to read and does not carry
    a dependency.
    """
    h = 0xCBF29CE484222325
    for byte in key.encode("utf-8"):
        h ^= byte
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def read_lifesnaps(
    path: str = RAW_CSV,
    min_coverage: float = MIN_COVERAGE,
    min_observed_days: int = MIN_OBSERVED_DAYS,
) -> Tuple[List[LifeSnapsEvent], CohortReport]:
    """
    Read the raw log and apply the four declared rules.

    Args:
        path: the raw CSV.
        min_coverage: observed days divided by the participant's own calendar
            span, below which the participant is dropped.
        min_observed_days: days a kept participant must carry.

    Returns:
        The kept episodes and a report of what the rules did.

    Raises:
        ValueError: when the file is missing a column the rules need.
    """
    with open(path, newline="") as handle:
        raw = list(csv.DictReader(handle))

    required = {
        "person_id",
        "event",
        "day_index",
        "start_min",
        "duration_min",
        "start_datetime",
    }
    missing = required - set(raw[0] if raw else {})
    if missing:
        raise ValueError(
            f"{path} is missing {sorted(missing)}, which the declared rules read"
        )

    days: Dict[str, Set[int]] = collections.defaultdict(set)
    first: Dict[str, datetime] = {}
    for row in raw:
        unit = row["person_id"]
        days[unit].add(int(row["day_index"]))
        when = datetime.strptime(row["start_datetime"].strip(), _RAW_FORMAT)
        if unit not in first or when < first[unit]:
            first[unit] = when

    keep: Set[str] = set()
    dropped: List[Tuple[str, float, int]] = []
    for unit, seen in days.items():
        span = max(seen) - min(seen) + 1
        coverage = len(seen) / span
        if coverage >= min_coverage and len(seen) >= min_observed_days:
            keep.add(unit)
        else:
            dropped.append((unit, coverage, len(seen)))
    dropped.sort(key=lambda item: item[1])

    events: List[LifeSnapsEvent] = []
    merged = 0
    instants = 0
    repaired = 0
    for row in raw:
        unit = row["person_id"]
        if unit not in keep:
            continue
        label = row["event"]
        if label in GYM_MODALITIES:
            label = GYM_LABEL
            merged += 1
        day_index = int(row["day_index"])
        start_min = int(row["start_min"])
        minutes = int(row["duration_min"])
        if minutes > 0:
            duration_s, source = minutes * 60, "observed"
        else:
            duration_s = _injected_seconds(unit, day_index, start_min, label)
            if label in INSTANT_LABELS:
                source = "instant"
                instants += 1
            else:
                # A zero on a label that has real episodes is a recording
                # error, not an instant.  It is given the same floor so that
                # `Y >= X` holds, and it is counted separately so the two
                # reasons never merge into one number.
                source = "repaired"
                repaired += 1
        when = datetime.strptime(row["start_datetime"].strip(), _RAW_FORMAT)
        events.append(
            LifeSnapsEvent(
                unit=unit,
                label=label,
                day_index=day_index,
                start_min=start_min,
                duration_s=duration_s,
                start=when,
                wave=_wave(first[unit]),
                duration_source=source,
            )
        )

    per_label = collections.defaultdict(lambda: [0, set()])
    for event in events:
        per_label[event.label][0] += 1
        per_label[event.label][1].add(event.unit)
    labels = tuple(
        (name, count, len(units), name in INSTANT_LABELS)
        for name, (count, units) in sorted(per_label.items(), key=lambda kv: -kv[1][0])
    )

    report = CohortReport(
        n_rows_raw=len(raw),
        n_rows_kept=len(events),
        participants_raw=len(days),
        participants_kept=len(keep),
        dropped=tuple(dropped),
        merged_into_gym=merged,
        injected_instants=instants,
        repaired_zeros=repaired,
        labels=labels,
    )
    return events, report


def calendar_columns(events: Sequence[LifeSnapsEvent]) -> Dict[str, List[int]]:
    """
    The calendar node values, one entry per episode, in the order given.

    These are exogenous: they are a known function of the clock, they have no
    parents, and they may therefore enter a predictor at lag zero without
    threatening acyclicity.  Every node of `CALENDAR_NODES` names a property of
    an episode, so the vocabulary and the columns cannot drift apart.
    """
    return {name: [getattr(event, name) for event in events] for name in CALENDAR_NODES}


def duration_is_scored(label: str) -> bool:
    """
    Whether the duration channel of a label is a modelling target.

    An instant carries a duration so that the channel is well posed.  The count
    is what the model is asked to predict there, so the duration of an instant
    is excluded from the duration panel and from the parameter-recovery tables.
    """
    return label not in INSTANT_LABELS
