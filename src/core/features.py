"""
core/features.py
================
Calendar quantities as nodes of the graph.

A calendar node is a variable whose value is a known function of an epoch's
wall-clock time: the hour of the day, the day of the week, the index of the week
within the horizon.  It is exogenous, so it has no parents, it is never a target
and it needs no inference, and it is always observed, so it enters a fit as a
column of the context block of its own scale.

Each node has one home scale, the coarsest scale whose epochs its value is
constant across.  The hour of the day lives at an hour scale, the weekday and
the horizon index at a day scale.  A node reaches a finer target by broadcast,
so placing it once is enough.

Each node also has one declared encoding.  A threshold or a bin edge that is
fitted by accident is worse than one that is written down, so the encoding is
part of the node.  The estimator does not choose it.  A one-hot node drops
its first level, which the intercept then absorbs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from .calendar import SlotCalendar

# The encodings a node may declare.  An ordinal node is one column carrying the
# index itself, which is what a trend coefficient is read from; a binned or
# one-hot node is a set of contrasts, which is what a conditional table is read
# from.  Both readings of the horizon index are wanted and neither substitutes
# for the other.
ENCODINGS = ("onehot", "bins", "ordinal")


@dataclass(frozen=True)
class CalendarNode:
    """One exogenous calendar variable, with its encoding declared."""

    name: str
    value: Callable[[datetime], int]
    encoding: str = "onehot"
    levels: Tuple[int, ...] = ()
    edges: Tuple[int, ...] = ()
    period: str = "day"

    def __post_init__(self) -> None:
        if self.encoding not in ENCODINGS:
            raise ValueError(
                f"{self.name!r} declares encoding {self.encoding!r}; "
                f"expected one of {list(ENCODINGS)}"
            )
        if self.encoding == "onehot" and len(self.levels) < 2:
            raise ValueError(
                f"{self.name!r} is one-hot over {list(self.levels)}, which needs "
                "at least two levels to carry a contrast"
            )
        if self.encoding == "bins":
            if len(self.edges) < 3:
                raise ValueError(
                    f"{self.name!r} is binned on edges {list(self.edges)}, which "
                    "needs at least two bins, so at least three edges"
                )
            if list(self.edges) != sorted(set(self.edges)):
                raise ValueError(
                    f"{self.name!r} has bin edges {list(self.edges)}, which are "
                    "not strictly increasing"
                )

    @property
    def width(self) -> int:
        """Columns the node contributes, with the reference level left out."""
        if self.encoding == "ordinal":
            return 1
        if self.encoding == "onehot":
            return len(self.levels) - 1
        return len(self.edges) - 2

    @property
    def columns(self) -> List[str]:
        if self.encoding == "ordinal":
            return [self.name]
        if self.encoding == "onehot":
            return [f"{self.name}={v}" for v in self.levels[1:]]
        pairs = list(zip(self.edges, self.edges[1:]))[1:]
        return [f"{self.name}[{lo},{hi})" for lo, hi in pairs]

    def encode(self, whens: Sequence[datetime]) -> np.ndarray:
        """The node's design matrix over the given epoch starts."""
        raw = np.asarray([self.value(w) for w in whens], dtype=float)
        if self.encoding == "ordinal":
            return raw[:, None]
        if self.encoding == "onehot":
            return np.stack([(raw == v).astype(float) for v in self.levels[1:]], axis=1)
        # The first bin is the reference, as the first level of a one-hot is.
        out = np.zeros((raw.size, self.width), dtype=float)
        for j, (lo, hi) in enumerate(list(zip(self.edges, self.edges[1:]))[1:]):
            out[:, j] = ((raw >= lo) & (raw < hi)).astype(float)
        return out


# ---------------------------------------------------------------------------
# The nodes themselves
# ---------------------------------------------------------------------------


def hour_of_day(edges: Sequence[int] = (0, 7, 10, 12, 14, 17, 21, 24)) -> CalendarNode:
    """Hour of the day, binned, with the first bin as the reference."""
    return CalendarNode(
        name="hour_of_day",
        value=lambda w: w.hour,
        encoding="bins",
        edges=tuple(int(e) for e in edges),
        period="day",
    )


def day_of_week() -> CalendarNode:
    """Day of the week as seven levels, Monday first and Monday the reference."""
    return CalendarNode(
        name="day_of_week",
        value=lambda w: w.weekday(),
        encoding="onehot",
        levels=tuple(range(7)),
        period="week",
    )


def weekend() -> CalendarNode:
    """Saturday and Sunday against the rest of the week."""
    return CalendarNode(
        name="weekend",
        value=lambda w: int(w.weekday() >= 5),
        encoding="onehot",
        levels=(0, 1),
        period="week",
    )


def month_of_year() -> CalendarNode:
    """Month as twelve levels, January the reference."""
    return CalendarNode(
        name="month_of_year",
        value=lambda w: w.month - 1,
        encoding="onehot",
        levels=tuple(range(12)),
        period="year",
    )


def quarter_of_year() -> CalendarNode:
    """Calendar quarter as four levels, the first quarter the reference."""
    return CalendarNode(
        name="quarter_of_year",
        value=lambda w: (w.month - 1) // 3,
        encoding="onehot",
        levels=(0, 1, 2, 3),
        period="year",
    )


def _horizon_value(origin: datetime, unit: str) -> Callable[[datetime], int]:
    days = {"day": 1, "week": 7}
    if unit not in days:
        raise ValueError(f"horizon unit must be 'day' or 'week'; got {unit!r}")
    step = days[unit]
    return lambda w: (w - origin).days // step


def horizon_trend(origin: datetime, unit: str = "week") -> CalendarNode:
    """The horizon index as one ordinal column, which a trend is read from."""
    return CalendarNode(
        name=f"{unit}_index",
        value=_horizon_value(origin, unit),
        encoding="ordinal",
        period="horizon",
    )


def horizon_bins(
    origin: datetime, n_units: int, n_bins: int = 3, unit: str = "week"
) -> CalendarNode:
    """The horizon index cut into equal bins, which a conditional is read from."""
    if n_bins < 2:
        raise ValueError(f"a binned horizon needs at least two bins; got {n_bins}")
    edges = [round(j * n_units / n_bins) for j in range(n_bins)] + [n_units + 1]
    return CalendarNode(
        name=f"{unit}_bin",
        value=_horizon_value(origin, unit),
        encoding="bins",
        edges=tuple(edges),
        period="horizon",
    )


# ---------------------------------------------------------------------------
# Placing a node on the ladder
# ---------------------------------------------------------------------------


def home_scale(cal: SlotCalendar, node: CalendarNode, counts: Sequence[int]) -> int:
    """
    The coarsest scale whose epochs the node's value never changes inside.

    A node is constant across an epoch when it reads the same at the epoch's
    first slot and at its last, and the check runs over the horizon actually
    given, so the answer belongs to this ladder.
    """
    home = 1
    for k in range(1, len(counts) + 1):
        if all(
            node.value(cal.start_of(k, t)) == node.value(cal.end_of(k, t) - cal.slot)
            for t in range(counts[k - 1])
        ):
            home = k
        else:
            break
    return home


def node_matrix(
    cal: SlotCalendar, node: CalendarNode, k: int, n_epochs: int
) -> np.ndarray:
    """The node's design matrix over the first `n_epochs` of scale k."""
    return node.encode([cal.start_of(k, t) for t in range(n_epochs)])


@dataclass(frozen=True)
class CalendarBlocks:
    """Per-scale context matrices and the column names behind them."""

    blocks: List[np.ndarray] = field(default_factory=list)
    names: Dict[int, List[str]] = field(default_factory=dict)
    homes: Dict[str, int] = field(default_factory=dict)

    @property
    def widths(self) -> List[int]:
        return [int(b.shape[1]) for b in self.blocks]


def calendar_blocks(
    cal: SlotCalendar, nodes: Sequence[CalendarNode], counts: Sequence[int]
) -> CalendarBlocks:
    """
    Context blocks for every scale, each node placed once at its home scale.

    A node placed at every scale would be the same column repeated, which the
    likelihood cannot separate, so each one is placed where its value stops
    varying and the scales below it read it through the nesting instead.
    """
    K = len(counts)
    per_scale: Dict[int, List[CalendarNode]] = {k: [] for k in range(1, K + 1)}
    homes: Dict[str, int] = {}
    for node in nodes:
        k = home_scale(cal, node, counts)
        homes[node.name] = k
        per_scale[k].append(node)

    blocks: List[np.ndarray] = []
    names: Dict[int, List[str]] = {}
    for k in range(1, K + 1):
        here = per_scale[k]
        names[k] = [name for node in here for name in node.columns]
        if here:
            blocks.append(
                np.concatenate(
                    [node_matrix(cal, node, k, counts[k - 1]) for node in here], axis=1
                )
            )
        else:
            blocks.append(np.zeros((counts[k - 1], 0), dtype=float))
    return CalendarBlocks(blocks=blocks, names=names, homes=homes)
