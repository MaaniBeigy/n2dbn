"""
learning/calendar_arms.py
=========================
The three-way calendar-node ablation, declared before it is run.

A calendar node is a deterministic function of the epoch, so it reaches the
child kernel only through the part of the design that varies inside a parent
block.  Its home scale is the finest scale whose parent epoch spans a full
period: an hour index varies inside a day, a weekday index inside a week.
Placing a node above its home scale buys nothing and pays the price anyway.

The three arms follow from that.  `absent` carries no calendar node, `home`
carries each at the scale that identifies it, and `every` carries each at every
scale.  The second beating the first answers whether calendar nodes make a
better nested model.  The third beating the second would say the home-scale
argument is wrong, which is the falsification the arm exists for.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from ..core.typed_graph import CALENDAR, LEVEL, TypedEdge
from ..data.label_axis import LabelAxis
from ..data.lifesnaps import CALENDAR_NODES
from ..data.lifesnaps_ladder import LADDER_UNITS

ABSENT = "absent"
HOME = "home"
EVERY = "every"
ARMS: Tuple[str, ...] = (ABSENT, HOME, EVERY)


def home_scale(node: str, units: Sequence[str] = LADDER_UNITS) -> int:
    """The scale index that identifies one calendar node."""
    if node not in CALENDAR_NODES:
        raise KeyError(f"{node!r} is not a calendar node")
    unit = str(CALENDAR_NODES[node]["identified_at"])
    if unit not in units:
        raise KeyError(f"{node!r} is identified at {unit!r}, which this ladder lacks")
    return list(units).index(unit) + 1


def placement(
    arm: str, units: Sequence[str] = LADDER_UNITS
) -> Dict[str, Tuple[int, ...]]:
    """
    Which scales each calendar node is offered at, under one arm.

    Raises:
        ValueError: when the arm is not one of `ARMS`.
    """
    if arm not in ARMS:
        raise ValueError(f"ablation arm {arm!r}; expected one of {list(ARMS)}")
    if arm == ABSENT:
        return {node: () for node in CALENDAR_NODES}
    if arm == HOME:
        return {node: (home_scale(node, units),) for node in CALENDAR_NODES}
    return {node: tuple(range(1, len(units) + 1)) for node in CALENDAR_NODES}


def calendar_candidates(
    axis: LabelAxis, arm: str, units: Sequence[str] = LADDER_UNITS
) -> List[TypedEdge]:
    """
    The calendar edges one arm offers to the search, all at lag zero.

    A calendar node has no parents, so reading it at the epoch being predicted
    reveals nothing about that epoch and lag zero stays admissible.
    """
    edges: List[TypedEdge] = []
    for node, scales in placement(arm, units).items():
        if node not in axis.exogenous:
            continue
        column = axis.exogenous_index(node)
        for scale in scales:
            edges.append(
                TypedEdge(
                    src_kind=CALENDAR,
                    src_scale=scale,
                    src_label=column,
                    src_quantity=LEVEL,
                    lag=0,
                )
            )
    return sorted(edges)
