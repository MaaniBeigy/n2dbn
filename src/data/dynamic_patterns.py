"""
data/dynamic_patterns.py
========================
Ingest the dynamic_patterns benchmark: its event log and its two manifests.

The log lists every eventuality of every subject as one row carrying a label, a
start and an end.  Events, routines and context episodes are written the same
way and nothing marks which is which, so every label enters as a column of the
same array and a context episode is a label in its own right.

Two things separate this from a single-subject log.

The log holds 30 subjects whose arrays share one label axis and have to share one
time axis before a fit can compare them, and each subject's own last episode ends
at its own hour, so the horizon is measured once over the whole file and handed
to every subject.  A subject whose log stops a week short of that horizon is
zero-filled to reach it, and those slots hold no record of anything, so the fill
comes back marked unobserved and a likelihood that reads the mask leaves it out.

A subject also records episodes of one label that the finest scale cannot hold
apart.  Two of them occasionally start in the same minute, and more often a
second one starts while the first is still running.  Two starts in one slot enter
X as one Bernoulli success and their marks add into a single duration no episode
had.  A start inside a running episode of its own label sits outside the risk
set, so `start_loglik` drops it while `mark_loglik` keeps its mark, and the two
halves of the likelihood then read different episode sets.

The two readings of this log want opposite things there.  Checking the arrays
against the resolved manifest wants every episode the log holds, since
`realized_count` counts them all.  Fitting the likelihood wants one start per
label per slot and one mark per start.  A default would hand one reading the
other's arrays in silence, so `build_subject_data` takes its policy as an
argument with no default.  `VERBATIM` lays every episode down as it was read.
`IDENTIFIABLE` folds the duplicate starts and then the same-label overlaps.
Either way the counts that come back carry the collisions the arrays still hold,
so a caller reading verbatim arrays is handed the size of what is hidden in them.

Under `VERBATIM` the day scale reproduces the resolved manifest cell for cell.  X
at the day scale counts the episodes that began that day and Y under the
initiation reading charges each episode's whole length to the day it began, which
is what `realized_count` and `realized_total_minutes` hold.  Every fold under
`IDENTIFIABLE` takes one episode off the day it began, so the day-scale count gap
against the manifest is the number of folds exactly.

The declared manifest is the structural truth.  Six carry-over rules each name a
source label, a source quantity, a lag in days and a target quantity, which is
one relation apiece at the day scale.  The rules leave the shape open.  A rule
acts when its source crosses a threshold, and a fit may express that through any
term of the source's basis, so what is declared is the relation and the sign of
its coefficient, and the threshold travels on the rule for whoever scores a fit.
`relation_terms` is the comparison: a search that came back with the log term of
the right source, target and lag and a search that came back with one of its
bands both found the declared relation, since the rule is the relation and the
basis term is how it was expressed.

The resolved manifest is the per-subject, per-day, per-label record of what the
declarations resolved to and what the solver realized.  It is the arithmetic
truth the arrays are checked against.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from itertools import groupby
from typing import (
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

from ..core.calendar import SlotCalendar, require_naive
from ..core.durations import aggregate
from ..core.scales import ScaleHierarchy
from ..core.typed_graph import COUNT, DURATION, OBSERVED, TypedEdge, TypedGraph
from ..core.variables import MultiScaleData
from .event_log import ObservedEvent, build_multiscale_data

# The grid the benchmark was measured on.  2026-02-16 is a Monday and is the
# earliest start in the log, so a branching to 7 days gives Monday-anchored
# weeks.  Every timestamp in the log carries second 0, so a one-minute slot
# loses nothing and a duration in slots is the duration in minutes exactly.
BENCHMARK_ORIGIN = datetime(2026, 2, 16)
BENCHMARK_BRANCHING: Tuple[int, ...] = (60, 24, 7)
BENCHMARK_SLOT_MINUTES = 1

# With that branching the ladder is minute, hour, day, week.
DAY_SCALE = 3

# The two readings of one log.  Neither is the default; a caller names one.
VERBATIM = "verbatim"
IDENTIFIABLE = "identifiable"
EPISODE_POLICIES = (VERBATIM, IDENTIFIABLE)

# What a declared source quantity reads on the model's arrays.  Both duration
# readings map to Y, whose day-scale value is the total minutes of the episodes
# that began that day; that equals the longest of them exactly when at most one
# episode of the source begins per day, which is the case for every source the
# manifest reads as a maximum.  Measured on the shipped resolved manifest: sleep
# is the one source declared as a maximum, its 2520 cells hold at most one
# episode a day, and the total and the maximum agree on every one of them, so a
# maximum-duration column would be the column Y already holds.
SOURCE_QUANTITIES = {
    "count": COUNT,
    "total_duration": DURATION,
    "max_duration": DURATION,
}

# What a declared target quantity is a density over.
TARGET_QUANTITIES = {"episodes": COUNT, "duration": DURATION}

DIRECTIONS = ("increasing", "decreasing")


# ---------------------------------------------------------------------------
# The log
# ---------------------------------------------------------------------------


def read_subject_event_log(
    path: str,
    subject_col: str = "person_id",
    label_col: str = "label",
    start_col: str = "start_iso",
    end_col: str = "end_iso",
    delimiter: str = "\t",
) -> Dict[str, List[ObservedEvent]]:
    """
    Read a multi-subject event log into episodes grouped by subject.

    Args:
        path: the delimited file to read.
        subject_col: column holding the subject the episode belongs to.
        label_col: column holding the label.
        start_col: column holding the naive ISO start.
        end_col: column holding the naive ISO end.
        delimiter: field separator of the file.

    Returns:
        One list of episodes per subject, each in file order, the subjects in
        the order they first appear.

    Raises:
        ValueError: if a named column is absent, a timestamp is unreadable or
            timezone-aware, a row ends before it starts, or a subject is blank.
    """
    by_subject: Dict[str, List[ObservedEvent]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        fields = list(reader.fieldnames or [])
        for col in (subject_col, label_col, start_col, end_col):
            if col not in fields:
                raise ValueError(
                    f"column '{col}' is missing from {path}; the file has {fields}"
                )
        # Row 1 is the header, so the first record is row 2 of the file.
        for row_no, row in enumerate(reader, start=2):
            subject = (row[subject_col] or "").strip()
            if not subject:
                raise ValueError(f"row {row_no} of {path} names no subject")
            try:
                start = datetime.fromisoformat(row[start_col])
                end = datetime.fromisoformat(row[end_col])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"row {row_no} of {path} carries an unreadable timestamp: {exc}"
                ) from exc
            if end < start:
                raise ValueError(
                    f"row {row_no} of {path} ends at {end.isoformat()} before "
                    f"its start {start.isoformat()}"
                )
            require_naive(start, f"row {row_no} of {path} start")
            require_naive(end, f"row {row_no} of {path} end")
            # The reader builds a fresh string per cell, so interning collapses
            # the 59 labels of 175894 rows to one object each.
            event = ObservedEvent(
                label=sys.intern(row[label_col]), start=start, end=end
            )
            by_subject.setdefault(sys.intern(subject), []).append(event)
    return by_subject


def subject_persona(person_id: str) -> str:
    """The persona a subject id names, which is the id without its trailing number."""
    stem, _, tail = person_id.rpartition("_")
    return stem if stem and tail.isdigit() else person_id


# ---------------------------------------------------------------------------
# Episode policies
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeCounts:
    """Episodes read for one subject, what the policy folded, and what collides."""

    read: int
    duplicate_starts: int = 0
    overlaps: int = 0
    stacked_starts: int = 0
    shadowed_starts: int = 0

    @property
    def kept(self) -> int:
        """Episodes that reached the arrays."""
        return self.read - self.duplicate_starts - self.overlaps


def merge_duplicate_starts(
    events: Sequence[ObservedEvent],
) -> Tuple[List[ObservedEvent], int]:
    """
    Fold episodes of one label sharing a start into one running to the latest end.

    Returns:
        The kept episodes in first-appearance order, and how many were dropped.
    """
    kept: Dict[Tuple[str, datetime], ObservedEvent] = {}
    order: List[Tuple[str, datetime]] = []
    for e in events:
        key = (e.label, e.start)
        seen = kept.get(key)
        if seen is None:
            kept[key] = e
            order.append(key)
        elif e.end > seen.end:
            kept[key] = ObservedEvent(label=e.label, start=e.start, end=e.end)
    return [kept[k] for k in order], len(events) - len(order)


def merge_same_label_overlap(
    events: Sequence[ObservedEvent],
) -> Tuple[List[ObservedEvent], int]:
    """
    Fold overlapping episodes of one label into their union.

    Returns:
        The kept episodes sorted by start then label, and how many were dropped.
    """
    by_label: Dict[str, List[ObservedEvent]] = {}
    for e in events:
        by_label.setdefault(e.label, []).append(e)

    out: List[ObservedEvent] = []
    for label, items in by_label.items():
        items.sort(key=lambda e: (e.start, e.end))
        current = items[0]
        for e in items[1:]:
            # An episode beginning exactly at another's end is left alone: an
            # episode at slot t of length Y marks t+1 to t+Y-1 as running and
            # leaves t+Y at risk, so that start is already inside the risk set.
            if e.start < current.end:
                if e.end > current.end:
                    current = ObservedEvent(label=label, start=current.start, end=e.end)
            else:
                out.append(current)
                current = e
        out.append(current)
    out.sort(key=lambda e: (e.start, e.label))
    return out, len(events) - len(out)


def _placement(
    event: ObservedEvent, slot_minutes: int, origin: datetime
) -> Tuple[int, int]:
    """The finest slot an episode starts in, and the slots it goes on to occupy."""
    slot_seconds = float(slot_minutes) * 60.0
    start = int((event.start - origin).total_seconds() // slot_seconds)
    n_occ = max(1, math.ceil((event.end - event.start).total_seconds() / slot_seconds))
    return start, n_occ


def count_collisions(
    events: Sequence[ObservedEvent],
    slot_minutes: int = BENCHMARK_SLOT_MINUTES,
    origin: datetime = BENCHMARK_ORIGIN,
) -> Tuple[int, int]:
    """
    Count the starts the finest scale cannot hold apart, on the grid it lays them on.

    A start is stacked when an earlier start of its label falls in the same slot,
    and a slot is shadowed when an episode of its label is still running in it.

    The two figures count different things because the arrays do.  Stacking
    costs episodes, since one slot holds one count however many started there.
    Shadowing costs slots, since `start_loglik` masks a shadowed slot's whole
    Bernoulli term whatever it holds, so the second figure equals
    `((X > 0) & running_mask(X, Y)).sum()` on the arrays this grid produces.

    Args:
        events: the episodes as they would reach the arrays.
        slot_minutes: wall-clock length of one finest slot.
        origin: first slot boundary.

    Returns:
        Stacked starts, then shadowed slots.
    """
    # The reach below sums the group's durations because that is what the
    # arrays hold: `build_multiscale_data` adds every episode's occupancy into
    # one Y cell, and `running_mask` then runs to t + Y - 1.  Measuring the
    # longest member alone would report a shorter shadow than the arrays cast.
    by_label: Dict[str, List[Tuple[int, int]]] = {}
    for e in events:
        by_label.setdefault(e.label, []).append(_placement(e, slot_minutes, origin))

    stacked = shadowed = 0
    for spans in by_label.values():
        spans.sort()
        # The last slot still occupied by an episode of this label whose start
        # came strictly earlier, which is what shadows the starts after it.
        reach = -1
        for slot, group in groupby(spans, key=lambda span: span[0]):
            here = list(group)
            stacked += len(here) - 1
            if slot <= reach:
                shadowed += 1
            reach = max(reach, slot + sum(n for _, n in here) - 1)
    return stacked, shadowed


# ---------------------------------------------------------------------------
# The shared grid
# ---------------------------------------------------------------------------


def shared_horizon_slots(
    by_subject: Mapping[str, Sequence[ObservedEvent]],
    slot_minutes: int = BENCHMARK_SLOT_MINUTES,
    branching: Sequence[int] = BENCHMARK_BRANCHING,
    origin: datetime = BENCHMARK_ORIGIN,
) -> int:
    """
    Finest slots reaching every subject's last episode, padded to whole epochs.

    Raises:
        ValueError: if no subject holds an episode.
    """
    reach = 0
    for events in by_subject.values():
        for e in events:
            start, n_occ = _placement(e, slot_minutes, origin)
            reach = max(reach, start + n_occ)
    if reach <= 0:
        raise ValueError("an empty log fixes no horizon for the subjects to share")
    span = _coarsest_span(branching)
    return -(-reach // span) * span


def build_subject_data(
    events: Sequence[ObservedEvent],
    labels: Sequence[str],
    policy: str,
    slot_minutes: int = BENCHMARK_SLOT_MINUTES,
    branching: Sequence[int] = BENCHMARK_BRANCHING,
    origin: datetime = BENCHMARK_ORIGIN,
    horizon_slots: Optional[int] = None,
    study_slots: Optional[int] = None,
    observed: Optional[Sequence[bool]] = None,
) -> Tuple[MultiScaleData, SlotCalendar, EpisodeCounts]:
    """
    Lay one subject's episodes on the shared grid under the policy the caller names.

    `VERBATIM` lays every episode down as it was read and lets a slot hold two
    starts of one label.  `IDENTIFIABLE` folds the duplicate starts and then the
    same-label overlaps, and rejects a slot that still holds two starts of one
    label, which is what a slot wider than the gap between two starts leaves.

    Pooling several subjects is the same call under `VERBATIM` on their
    concatenated log.  30 subjects begin one label in the same minute, so a
    pooled slot carries up to 30 starts and every one past the first is stacked.

    Args:
        events: the subject's episodes.
        labels: the shared label axis, which every subject is given in full.
        policy: `VERBATIM` or `IDENTIFIABLE`.
        slot_minutes: wall-clock length of one finest slot.
        branching: branching factors, finest pair first.
        origin: first slot boundary, shared by every subject.
        horizon_slots: finest slots the arrays run to, zero-filled to reach it
            and marked unobserved over the fill.
        observed: optional flag per finest slot of this subject's record, which
            the fill up to the shared horizon extends.

    Returns:
        The arrays, the calendar behind them, and the episode counts, whose
        stacked and shadowed figures are what the arrays handed back still hide.

    Raises:
        ValueError: if the policy is not one of `EPISODE_POLICIES`.
    """
    if policy not in EPISODE_POLICIES:
        raise ValueError(
            f"episode policy {policy!r}; expected one of {list(EPISODE_POLICIES)}"
        )
    counts = EpisodeCounts(read=len(events))
    kept = list(events)
    if policy == IDENTIFIABLE:
        kept, duplicates = merge_duplicate_starts(kept)
        kept, overlaps = merge_same_label_overlap(kept)
        counts = replace(counts, duplicate_starts=duplicates, overlaps=overlaps)
    stacked, shadowed = count_collisions(kept, slot_minutes, origin)
    counts = replace(counts, stacked_starts=stacked, shadowed_starts=shadowed)

    # Every slot past this subject's last episode is a slot the record does
    # not cover.  `build_multiscale_data` pads up to whole coarsest epochs and
    # counts that pad as observed, which every stored result depends on, so the
    # subject's own reach is supplied here where there is no stored result to
    # preserve and the fabricated absence would otherwise read as a recorded
    # non-start.

    data, calendar, _ = build_multiscale_data(
        kept,
        slot_minutes=slot_minutes,
        branching=list(branching),
        labels=list(labels),
        origin=origin,
        observed=observed,
        finest_binary=(policy == IDENTIFIABLE),
    )
    if horizon_slots is not None:
        data = _pad_to_horizon(data, int(horizon_slots), branching, calendar)
    if study_slots is not None:
        watched = np.arange(data.get_X(1).shape[0]) < int(study_slots)
        data = replace(data, observed=watched)
    return data, calendar, counts


def build_population(
    by_subject: Mapping[str, Sequence[ObservedEvent]],
    labels: Sequence[str],
    policy: str = VERBATIM,
    slot_minutes: int = BENCHMARK_SLOT_MINUTES,
    branching: Sequence[int] = BENCHMARK_BRANCHING,
    origin: datetime = BENCHMARK_ORIGIN,
) -> Iterator[Tuple[str, MultiScaleData, EpisodeCounts]]:
    """
    Every subject on one grid, watched over one window, yielded one at a time.

    The horizon and the study window are properties of the run, so both are
    measured once over the whole log and handed to every subject.  Building a
    subject alone and forgetting either one fills its tail with zeros the log
    never recorded, which a fit reads as recorded non-starts.

    Yielding rather than returning a list keeps one subject's finest arrays
    resident at a time, which at a one-minute slot is the difference between
    126 MB and the whole cohort.
    """
    horizon = shared_horizon_slots(by_subject, slot_minutes, branching, origin)
    study = study_period_slots(by_subject, slot_minutes, branching, origin)
    for person in sorted(by_subject):
        data, _, counts = build_subject_data(
            by_subject[person],
            labels,
            policy,
            slot_minutes=slot_minutes,
            branching=branching,
            origin=origin,
            horizon_slots=horizon,
            study_slots=study,
        )
        yield person, data, counts


def study_period_slots(
    by_subject: Mapping[str, Sequence[ObservedEvent]],
    slot_minutes: int = BENCHMARK_SLOT_MINUTES,
    branching: Sequence[int] = BENCHMARK_BRANCHING,
    origin: datetime = BENCHMARK_ORIGIN,
) -> int:
    """
    Finest slots the run itself covers, rounded up to whole coarsest epochs.

    Every subject was watched over one window, so a slot inside it holding no
    episode is a fact about that subject and a slot past it is a slot nobody
    watched.  The window is measured from episode starts, since the initiation
    reading charges an episode's whole length to the slot it began in, so a
    start inside the window carries its mark whatever it overruns.

    Raises:
        ValueError: if no subject holds an episode.
    """
    last = -1
    for events in by_subject.values():
        for event in events:
            last = max(last, _placement(event, slot_minutes, origin)[0])
    if last < 0:
        raise ValueError("an empty log covers no study period")
    span = _coarsest_span(branching)
    return -(-(last + 1) // span) * span


def _coarsest_span(branching: Sequence[int]) -> int:
    """Finest slots inside one coarsest epoch."""
    span = 1
    for b in branching:
        span *= int(b)
    return span


def _pad_to_horizon(
    data: MultiScaleData,
    horizon_slots: int,
    branching: Sequence[int],
    calendar: SlotCalendar,
) -> MultiScaleData:
    """Extend the finest arrays with unobserved slots and re-sum the ladder above."""
    X1, Y1 = data.get_X(1), data.get_Y(1)
    T = int(X1.shape[0])
    if horizon_slots == T:
        return data
    if horizon_slots < T:
        raise ValueError(
            f"a horizon of {horizon_slots} slots ends before this subject's last "
            f"episode, which reaches slot {T}"
        )
    span = _coarsest_span(branching)
    if horizon_slots % span:
        raise ValueError(
            f"a horizon of {horizon_slots} slots is not a whole number of "
            f"coarsest epochs of {span} slots"
        )
    hierarchy = ScaleHierarchy.from_uniform_branching(horizon_slots, list(branching))
    maps = [np.asarray(pm) for pm in hierarchy.parent_maps[:-1]]
    pad = np.zeros((horizon_slots - T, int(X1.shape[1])), dtype=X1.dtype)
    # The fill records nothing about the subject, so a zero there is an absence
    # nobody watched and reading it as a non-start would credit the model for
    # predicting silence out of a stretch the log never covered.  Both ends of
    # the fill land on a coarsest epoch boundary, so whole epochs go unobserved
    # and `mask.check_ladder` stays satisfied.
    watched = data.observed
    if watched is None:
        watched = np.ones(T, dtype=bool)
    return MultiScaleData(
        hierarchy=hierarchy,
        label_names=list(data.label_names),
        X=aggregate(np.vstack([X1, pad]), maps),
        Y=aggregate(np.vstack([Y1, pad]), maps),
        C=[
            np.zeros((hierarchy.n_epochs(k), 0), dtype=np.float64)
            for k in range(1, hierarchy.K + 1)
        ],
        calendar=calendar,
        observed=np.concatenate([watched, np.zeros(horizon_slots - T, dtype=bool)]),
    )


# ---------------------------------------------------------------------------
# The declared manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CarryOverRule:
    """
    One declared lagged dependence, as the manifest states it.

    The amount is in the unit of the target quantity, so a count target carries
    episodes and a duration target carries minutes.
    """

    source: str
    target: str
    source_quantity: str
    target_quantity: str
    lag_days: int
    amount: float
    direction: str
    source_threshold: Optional[float] = None
    personas: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.source_quantity not in SOURCE_QUANTITIES:
            raise ValueError(
                f"source quantity {self.source_quantity!r}; expected one of "
                f"{sorted(SOURCE_QUANTITIES)}"
            )
        if self.target_quantity not in TARGET_QUANTITIES:
            raise ValueError(
                f"target quantity {self.target_quantity!r}; expected one of "
                f"{sorted(TARGET_QUANTITIES)}"
            )
        if self.direction not in DIRECTIONS:
            raise ValueError(
                f"direction {self.direction!r}; expected one of {list(DIRECTIONS)}"
            )
        if self.lag_days < 1:
            raise ValueError(
                f"a lag of {self.lag_days} days reads the day being predicted, "
                "which an observed source cannot do"
            )

    @property
    def sign(self) -> int:
        """+1 when the rule raises its target, -1 when it lowers it."""
        return -1 if self.direction == "decreasing" else 1

    def applies_to(self, persona: Optional[str]) -> bool:
        """Whether the rule acts on a subject of the given persona."""
        return not self.personas or persona is None or persona in self.personas

    def typed_edge(self, src_label: int, src_scale: int = DAY_SCALE) -> TypedEdge:
        """The edge key the rule declares, read at the given scale."""
        return TypedEdge(
            src_kind=OBSERVED,
            src_scale=int(src_scale),
            src_label=int(src_label),
            src_quantity=SOURCE_QUANTITIES[self.source_quantity],
            lag=int(self.lag_days),
        )


@dataclass(frozen=True)
class DeclaredLabel:
    """One label as the manifest declares it, with the patterns acting on it."""

    name: str
    kind: str
    category: str
    stationary: bool
    personas: Tuple[str, ...] = ()
    patterns: Tuple[Mapping[str, object], ...] = ()

    @property
    def modes(self) -> Tuple[str, ...]:
        """The mode of each declared pattern, in declaration order."""
        return tuple(str(p.get("mode", "")) for p in self.patterns)


@dataclass(frozen=True)
class DeclaredEdge:
    """One carry-over rule placed on a label axis as a typed edge."""

    rule: CarryOverRule
    edge: TypedEdge
    dst_label: int
    dst_quantity: str


def relation_terms(
    parents: Iterable[TypedEdge], declared: TypedEdge
) -> List[TypedEdge]:
    """
    The basis terms of one declared relation that a parent set holds, sorted.

    A rule declares a relation: one source read at one scale and one quantity,
    one lag and one target.  The basis term is how a fit expressed that relation,
    the log term where the source acts monotonically and a band where it steps,
    so any term of the right relation is that rule recovered and the comparison
    reads every field of the edge but the basis.
    """
    return sorted(e for e in parents if e.relation == declared.relation)


@dataclass(frozen=True)
class DeclaredTruth:
    """Every declared label, and the distinct carry-over rules over them."""

    labels: Dict[str, DeclaredLabel]
    rules: Tuple[CarryOverRule, ...]

    @property
    def stationary_labels(self) -> List[str]:
        """The labels declared stationary, sorted, which are the control group."""
        return sorted(name for name, d in self.labels.items() if d.stationary)

    def typed_edges(
        self,
        label_names: Sequence[str],
        src_scale: int = DAY_SCALE,
        persona: Optional[str] = None,
    ) -> Tuple[List[DeclaredEdge], List[CarryOverRule]]:
        """
        The rules as typed edges on a label axis, and the rules it cannot carry.

        A label axis read off the log has a column only for a label with at
        least one episode, so a rule naming a label that was declared and never
        realized comes back in the second list and is unrecoverable by any fit.
        """
        index = {name: i for i, name in enumerate(label_names)}
        mapped: List[DeclaredEdge] = []
        off_axis: List[CarryOverRule] = []
        for rule in self.rules:
            if not rule.applies_to(persona):
                continue
            if rule.source not in index or rule.target not in index:
                off_axis.append(rule)
                continue
            mapped.append(
                DeclaredEdge(
                    rule=rule,
                    edge=rule.typed_edge(index[rule.source], src_scale),
                    dst_label=index[rule.target],
                    dst_quantity=TARGET_QUANTITIES[rule.target_quantity],
                )
            )
        return mapped, off_axis

    def truth_graph(
        self,
        label_names: Sequence[str],
        src_scale: int = DAY_SCALE,
        persona: Optional[str] = None,
    ) -> TypedGraph:
        """The declared edges as a graph, which a fitted graph is scored against."""
        graph = TypedGraph(label_names=list(label_names))
        mapped, _ = self.typed_edges(label_names, src_scale, persona)
        for declared in mapped:
            graph.add_edge(declared.edge, declared.dst_label, declared.dst_quantity)
        return graph


def read_declared_manifest(path: str) -> DeclaredTruth:
    """
    Read the declared manifest, one entry per label.

    A carry-over rule is stated twice, once on its source and once on its
    target, so the rules are collected by identity and ordered by source,
    target, quantities, lag and personas.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    labels: Dict[str, DeclaredLabel] = {}
    rules: Dict[tuple, CarryOverRule] = {}
    for name in sorted(raw):
        entry = raw[name]
        labels[name] = DeclaredLabel(
            name=name,
            kind=str(entry["kind"]),
            category=str(entry["category"]),
            stationary=bool(entry["stationary"]),
            personas=tuple(entry.get("personas") or ()),
            patterns=tuple(entry.get("patterns") or ()),
        )
        for side in ("edges_out", "edges_in"):
            for spec in entry.get(side) or ():
                rule = _carry_over_rule(spec)
                rules[_rule_key(rule)] = rule
    return DeclaredTruth(labels=labels, rules=tuple(rules[k] for k in sorted(rules)))


def _carry_over_rule(spec: Mapping[str, object]) -> CarryOverRule:
    """One carry-over rule from its manifest entry."""
    threshold = spec.get("source_threshold")
    return CarryOverRule(
        source=str(spec["source"]),
        target=str(spec["target"]),
        source_quantity=str(spec["source_quantity"]),
        target_quantity=str(spec["target_quantity"]),
        lag_days=int(spec["lag_days"]),
        amount=float(spec["amount"]),
        direction=str(spec["direction"]),
        source_threshold=None if threshold is None else float(threshold),
        personas=tuple(spec.get("personas") or ()),
    )


def _rule_key(rule: CarryOverRule) -> tuple:
    """What makes two stated rules the same rule."""
    return (
        rule.source,
        rule.target,
        rule.source_quantity,
        rule.target_quantity,
        rule.lag_days,
        rule.personas,
    )


# ---------------------------------------------------------------------------
# The resolved manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedCell:
    """One subject-day-label cell: what was asked of the solver, and what it did."""

    person_id: str
    persona_id: str
    day_index: int
    label: str
    resolved_count: int
    realized_count: int
    realized_total_minutes: int
    realized_max_minutes: int


def iter_resolved_cells(path: str) -> Iterator[ResolvedCell]:
    """Stream the resolved manifest, one JSON object per line."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield _resolved_cell(json.loads(line))


def read_resolved_manifest(path: str) -> List[ResolvedCell]:
    """The whole resolved manifest, in file order."""
    return list(iter_resolved_cells(path))


def _resolved_cell(record: Mapping[str, object]) -> ResolvedCell:
    """One resolved cell from its JSON record."""
    return ResolvedCell(
        person_id=str(record["person_id"]),
        persona_id=str(record["persona_id"]),
        day_index=int(record["day_index"]),
        label=str(record["label"]),
        resolved_count=int(record["resolved_count"]),
        realized_count=int(record["realized_count"]),
        realized_total_minutes=int(record["realized_total_minutes"]),
        realized_max_minutes=int(record["realized_max_minutes"]),
    )


@dataclass(frozen=True)
class CellAgreement:
    """How far one subject's day-scale arrays sit from the resolved cells."""

    n_cells: int
    count_error: int
    duration_error: int

    @property
    def exact(self) -> bool:
        """Whether both totals are zero, which is what the benchmark asks for."""
        return self.count_error == 0 and self.duration_error == 0


def compare_day_scale(
    data: MultiScaleData,
    cells: Iterable[ResolvedCell],
    day_scale: int = DAY_SCALE,
) -> CellAgreement:
    """
    Total absolute gap between one subject's day-scale arrays and its cells.

    Raises:
        ValueError: if the named scale does not span a day, if a cell names a
            label the arrays do not carry, or if a cell falls past the horizon.
    """
    if data.calendar is not None:
        span = data.calendar.epoch_span(day_scale)
        if span != timedelta(days=1):
            raise ValueError(
                f"scale {day_scale} spans {span}, so its epochs are not days and "
                "a per-day cell has nothing to compare against"
            )
    index = {name: i for i, name in enumerate(data.label_names)}
    X, Y = data.get_X(day_scale), data.get_Y(day_scale)

    n_cells = count_error = duration_error = 0
    for cell in cells:
        if cell.label not in index:
            raise ValueError(
                f"cell ({cell.person_id}, day {cell.day_index}) names label "
                f"{cell.label!r}, which the arrays do not carry"
            )
        if not 0 <= cell.day_index < X.shape[0]:
            raise ValueError(
                f"cell ({cell.person_id}, {cell.label}) sits on day "
                f"{cell.day_index}, past the horizon of {X.shape[0]} days"
            )
        ell = index[cell.label]
        n_cells += 1
        count_error += abs(int(X[cell.day_index, ell]) - cell.realized_count)
        duration_error += abs(int(Y[cell.day_index, ell]) - cell.realized_total_minutes)
    return CellAgreement(
        n_cells=n_cells,
        count_error=count_error,
        duration_error=duration_error,
    )
