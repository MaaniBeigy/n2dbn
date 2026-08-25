"""
data/label_axis.py
==================
The label axis of the observed log, after one act logged three times is collapsed.

Three columns record one act.  `notification_read` equals
`mood_survey + step_goal_survey` exactly on 97 percent of the participant-days
on which any of the three occurs, over 99 percent of each survey episode falls
within one minute of a read, and the day-scale correlation of the read count
with the survey sum is 0.99.  A learner scored on the log as shipped is handed
that identity as an edge and reports it as a finding, so the three become one
vertex here and the identity stops being expressible.

`notification_received` is delivered by the study on a schedule the participant
does not set.  It is a source and never a target, so it joins the calendar
nodes in the exogenous block.  Nothing can point into it and its own value may
be read at lag zero, which is the same permission a clock node carries.

Both facts live on the axis, so a run that builds its graph from
`LabelAxis.graph` cannot express a within-group edge and has no index to put the
scheduled label on.  The axis is what every rung of the falsification
ladder counts its denominator against.
"""

from __future__ import annotations

import collections
import dataclasses
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

from ..core.typed_graph import CALENDAR, LEVEL, TypedEdge, TypedGraph
from .lifesnaps import CALENDAR_NODES, INSTANT_LABELS, LifeSnapsEvent

#: The three columns that record one act, and the vertex they become.
COLLAPSE_GROUP: Tuple[str, ...] = (
    "notification_read",
    "mood_survey",
    "step_goal_survey",
)
COLLAPSED_LABEL = "survey_response"

#: Delivered by the study, so it is exogenous and never a learnable target.
SCHEDULED_LABEL = "notification_received"

#: Vertices whose episodes are instants: the raw instant labels of the log, plus
#: the vertex the three instant columns become.
INSTANT_VERTICES = frozenset(INSTANT_LABELS | {COLLAPSED_LABEL})

#: The exogenous block, calendar nodes first and the scheduled label last.
EXOGENOUS_NAMES: Tuple[str, ...] = tuple(CALENDAR_NODES) + (SCHEDULED_LABEL,)


@dataclass(frozen=True)
class LabelAxis:
    """Which labels a learner may target, which it may only read, and how they map."""

    targets: Tuple[str, ...]
    exogenous: Tuple[str, ...]
    grouped: Mapping[str, str]

    @property
    def L(self) -> int:
        return len(self.targets)

    def axis_name(self, raw: str) -> str:
        """The vertex a raw label becomes."""
        if raw not in self.grouped:
            raise KeyError(f"{raw!r} is not a label of this log")
        return self.grouped[raw]

    def target_index(self, raw: str) -> int:
        """The target index of a raw label, which the scheduled label has none of."""
        name = self.axis_name(raw)
        if name not in self.targets:
            raise KeyError(f"{raw!r} is exogenous and is never a target")
        return self.targets.index(name)

    def exogenous_index(self, name: str) -> int:
        """The column of an exogenous source in the calendar block."""
        if name not in self.exogenous:
            raise KeyError(f"{name!r} is not an exogenous source")
        return self.exogenous.index(name)

    def is_target(self, raw: str) -> bool:
        return self.axis_name(raw) in self.targets

    def expressible(self, src_raw: str, dst_raw: str) -> bool:
        """Whether an edge between two raw labels survives the collapse as a relation."""
        if not self.is_target(dst_raw):
            return False
        if not self.is_target(src_raw):
            return True
        return self.axis_name(src_raw) != self.axis_name(dst_raw)

    def is_instant(self, name: str) -> bool:
        """Whether a vertex carries instants, whose duration channel is unscored."""
        members = [raw for raw, axis in self.grouped.items() if axis == name]
        return bool(members) and all(raw in INSTANT_VERTICES for raw in members)

    def graph(self) -> TypedGraph:
        """An empty typed graph over this axis."""
        return TypedGraph(
            label_names=list(self.targets), calendar_names=list(self.exogenous)
        )

    def scheduled_edge(self, scale: int, lag: int) -> TypedEdge:
        """The scheduled label read as an exogenous level at one scale and lag."""
        return TypedEdge(
            src_kind=CALENDAR,
            src_scale=scale,
            src_label=self.exogenous_index(SCHEDULED_LABEL),
            src_quantity=LEVEL,
            lag=lag,
        )

    def digest(self) -> str:
        """A stable digest of the axis, so a stored result names the axis it used."""
        payload = "|".join(
            [
                ",".join(self.targets),
                ",".join(self.exogenous),
                ",".join(f"{raw}={axis}" for raw, axis in sorted(self.grouped.items())),
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Agreement:
    """What the three columns do to each other, measured on the log being read."""

    active_days: int
    days_in_agreement: int
    max_residual: int
    coincident: Tuple[Tuple[str, int, int], ...]
    day_correlation: float

    @property
    def share_in_agreement(self) -> float:
        return self.days_in_agreement / self.active_days if self.active_days else 0.0

    def summary(self) -> str:
        lines = [
            f"participant-days with any of {list(COLLAPSE_GROUP)}: {self.active_days}",
            f"  read == mood + step on {self.days_in_agreement} "
            f"({self.share_in_agreement:.3f}), largest residual {self.max_residual}",
            f"  day-scale correlation of read with the sum: {self.day_correlation:.3f}",
        ]
        lines.extend(
            f"  {name} within one minute of a read: {hit}/{total} "
            f"({hit / total:.3f})"
            for name, hit, total in self.coincident
        )
        return "\n".join(lines)


@dataclass(frozen=True)
class CollapseReport:
    """What the collapse did, so a run reports it without recomputing."""

    into: str
    group: Tuple[str, ...]
    episodes: int
    agreement: Agreement
    axis_digest: str

    def summary(self) -> str:
        return "\n".join(
            [
                f"collapsed {list(self.group)} into {self.into}: "
                f"{self.episodes} episodes",
                f"axis digest {self.axis_digest}",
                self.agreement.summary(),
            ]
        )


def _minute_of(event: LifeSnapsEvent) -> int:
    """The episode's start as a minute of participant time."""
    return event.day_index * 1440 + event.start_min


def _correlation(a: Sequence[float], b: Sequence[float]) -> float:
    """Pearson correlation of two sequences, zero when either is constant."""
    n = len(a)
    if n < 2:
        return 0.0
    mean_a, mean_b = sum(a) / n, sum(b) / n
    da = [x - mean_a for x in a]
    db = [y - mean_b for y in b]
    var_a = sum(x * x for x in da)
    var_b = sum(y * y for y in db)
    if var_a <= 0.0 or var_b <= 0.0:
        return 0.0
    return sum(x * y for x, y in zip(da, db)) / (var_a * var_b) ** 0.5


def agreement(events: Sequence[LifeSnapsEvent]) -> Agreement:
    """Measure the identity the collapse rests on, on the log being read."""
    per_day: Dict[Tuple[str, int], collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    starts: Dict[Tuple[str, str], List[int]] = collections.defaultdict(list)
    for event in events:
        per_day[(event.unit, event.day_index)][event.label] += 1
        starts[(event.unit, event.label)].append(_minute_of(event))

    active = in_agreement = 0
    largest = 0
    read_counts: List[float] = []
    sum_counts: List[float] = []
    for counts in per_day.values():
        read = counts[COLLAPSE_GROUP[0]]
        total = counts[COLLAPSE_GROUP[1]] + counts[COLLAPSE_GROUP[2]]
        if read == 0 and total == 0:
            continue
        active += 1
        in_agreement += read == total
        largest = max(largest, abs(read - total))
        read_counts.append(float(read))
        sum_counts.append(float(total))

    reads_by_unit = {
        unit: sorted(times)
        for (unit, label), times in starts.items()
        if label == COLLAPSE_GROUP[0]
    }
    coincident: List[Tuple[str, int, int]] = []
    for name in COLLAPSE_GROUP[1:]:
        hit = total = 0
        for (unit, label), times in starts.items():
            if label != name:
                continue
            reads = reads_by_unit.get(unit, [])
            for when in times:
                total += 1
                hit += any(abs(when - read) <= 1 for read in reads)
        coincident.append((name, hit, total))

    return Agreement(
        active_days=active,
        days_in_agreement=in_agreement,
        max_residual=largest,
        coincident=tuple(coincident),
        day_correlation=_correlation(read_counts, sum_counts),
    )


def label_axis(events: Sequence[LifeSnapsEvent]) -> LabelAxis:
    """The axis the labels present in a log induce, with the collapse applied."""
    present = {event.label for event in events}
    grouped: Dict[str, str] = {}
    for raw in sorted(present):
        grouped[raw] = COLLAPSED_LABEL if raw in COLLAPSE_GROUP else raw
    # A log read after the collapse carries the vertex and none of its members,
    # and the axis still has to answer for the members that made it.
    if COLLAPSED_LABEL in grouped.values():
        for raw in COLLAPSE_GROUP:
            grouped[raw] = COLLAPSED_LABEL
    targets = tuple(
        sorted({name for name in grouped.values() if name != SCHEDULED_LABEL})
    )
    # Every vertex names itself, so the axis of a log read before the collapse
    # and the axis of the same log read after it are the same axis.
    for name in targets:
        grouped[name] = name
    exogenous = tuple(
        name for name in EXOGENOUS_NAMES if name != SCHEDULED_LABEL or name in present
    )
    return LabelAxis(targets=targets, exogenous=exogenous, grouped=grouped)


def collapse_labels(
    events: Sequence[LifeSnapsEvent],
) -> Tuple[List[LifeSnapsEvent], CollapseReport]:
    """Rewrite the three columns into one vertex and report what that did."""
    measured = agreement(events)
    axis = label_axis(events)
    collapsed = 0
    out: List[LifeSnapsEvent] = []
    for event in events:
        if event.label in COLLAPSE_GROUP:
            collapsed += 1
            out.append(dataclasses.replace(event, label=COLLAPSED_LABEL))
        else:
            out.append(event)
    report = CollapseReport(
        into=COLLAPSED_LABEL,
        group=COLLAPSE_GROUP,
        episodes=collapsed,
        agreement=measured,
        axis_digest=axis.digest(),
    )
    return out, report
