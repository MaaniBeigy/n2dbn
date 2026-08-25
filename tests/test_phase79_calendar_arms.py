"""
Tests for the three-way calendar-node ablation.

The arms are declared before the run, so what is pinned here is the placement
each arm gives, that a home-scale placement matches the scale the invariance
argument identifies the node at, and that every offered edge is a lag-zero
exogenous edge a target can accept.
"""

from __future__ import annotations

import pytest

from src.core.typed_graph import CALENDAR, COUNT, LEVEL
from src.data.label_axis import collapse_labels, label_axis
from src.data.lifesnaps import CALENDAR_NODES, read_lifesnaps
from src.data.lifesnaps_ladder import LADDER_UNITS
from src.learning.calendar_arms import (
    ABSENT,
    ARMS,
    EVERY,
    HOME,
    calendar_candidates,
    home_scale,
    placement,
)


@pytest.fixture(scope="module")
def axis():
    events, _ = read_lifesnaps()
    return label_axis(collapse_labels(events)[0])


def test_the_three_arms_are_absent_home_and_every():
    assert ARMS == (ABSENT, HOME, EVERY)


def test_a_home_scale_is_the_finest_scale_the_node_varies_inside_its_parent_at():
    assert home_scale("hour_index") == LADDER_UNITS.index("hour") + 1
    assert home_scale("weekday_index") == LADDER_UNITS.index("day") + 1
    assert home_scale("week_index") == LADDER_UNITS.index("week") + 1
    for node, record in CALENDAR_NODES.items():
        assert LADDER_UNITS[home_scale(node) - 1] == record["identified_at"]


def test_a_node_the_ladder_cannot_identify_is_refused():
    with pytest.raises(KeyError, match="not a calendar node"):
        home_scale("phase_of_moon")
    with pytest.raises(KeyError, match="which this ladder lacks"):
        home_scale("hour_index", units=("day", "week"))


def test_the_absent_arm_offers_nothing(axis):
    assert placement(ABSENT) == {node: () for node in CALENDAR_NODES}
    assert calendar_candidates(axis, ABSENT) == []


def test_the_home_arm_offers_each_node_exactly_once(axis):
    edges = calendar_candidates(axis, HOME)
    assert len(edges) == len(CALENDAR_NODES)
    assert len({e.src_label for e in edges}) == len(CALENDAR_NODES)
    for node, scales in placement(HOME).items():
        assert scales == (home_scale(node),)


def test_the_every_arm_offers_each_node_at_every_scale(axis):
    edges = calendar_candidates(axis, EVERY)
    assert len(edges) == len(CALENDAR_NODES) * len(LADDER_UNITS)
    assert set(placement(EVERY)["hour_index"]) == set(range(1, len(LADDER_UNITS) + 1))
    home = set(calendar_candidates(axis, HOME))
    assert home.issubset(set(edges))


def test_every_offered_edge_is_an_exogenous_level_at_lag_zero(axis):
    for edge in calendar_candidates(axis, EVERY):
        assert edge.src_kind == CALENDAR
        assert edge.src_quantity == LEVEL
        assert edge.lag == 0
        assert edge.is_exogenous


def test_an_offered_edge_is_one_a_target_accepts(axis):
    graph = axis.graph()
    for edge in calendar_candidates(axis, HOME):
        graph.add_edge(edge, dst_label=0, dst_quantity=COUNT)
    assert graph.n_edges(0, COUNT) == len(CALENDAR_NODES)
    assert graph.is_acyclic()


def test_an_arm_that_was_not_declared_is_refused(axis):
    with pytest.raises(ValueError, match="ablation arm"):
        placement("some other way")
    with pytest.raises(ValueError, match="ablation arm"):
        calendar_candidates(axis, "some other way")


def test_a_node_the_axis_does_not_carry_is_left_out():
    events, _ = read_lifesnaps()
    axis = label_axis(collapse_labels(events)[0])
    trimmed = type(axis)(
        targets=axis.targets,
        exogenous=("hour_index",),
        grouped=axis.grouped,
    )
    assert len(calendar_candidates(trimmed, HOME)) == 1
