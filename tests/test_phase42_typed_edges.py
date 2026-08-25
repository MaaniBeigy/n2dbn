"""
tests/test_phase42_typed_edges.py
=================================
Edges typed by source scale, source quantity and target quantity.

Three properties are asserted rather than assumed.  All four count-and-duration
pairings are expressible, since a duration source feeding a count target is the
shape the older type could not carry.  Lag zero is admitted from a calendar
source at any scale and from an observed source at the finest one alone, and a
label that reads itself in the epoch being drawn is refused, so what is left of
acyclicity is a question about the order the labels are drawn in.  And iteration
is sorted, so a result cannot depend on how the interpreter hashed a string this
run.
"""

from __future__ import annotations

import pytest

from src.core.typed_graph import (
    CALENDAR,
    COUNT,
    DURATION,
    LEVEL,
    OBSERVED,
    QUANTITIES,
    TypedEdge,
    TypedGraph,
)

LABELS = ["sleep", "walk", "work"]
NODES = ["weekend", "hour_of_day", "week_index"]


def _graph() -> TypedGraph:
    return TypedGraph(label_names=list(LABELS), calendar_names=list(NODES))


def _observed(label=0, scale=1, quantity=COUNT, lag=1) -> TypedEdge:
    return TypedEdge(OBSERVED, scale, label, quantity, lag)


def _calendar(label=0, scale=1, lag=0) -> TypedEdge:
    return TypedEdge(CALENDAR, scale, label, LEVEL, lag)


# ---------------------------------------------------------------------------
# All four pairings, which is the point of the type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("src_quantity", QUANTITIES)
@pytest.mark.parametrize("dst_quantity", QUANTITIES)
def test_every_quantity_pairing_is_expressible(src_quantity, dst_quantity) -> None:
    g = _graph()
    edge = _observed(label=0, quantity=src_quantity, lag=1)
    g.add_edge(edge, dst_label=1, dst_quantity=dst_quantity)
    assert edge in g.get_parents(1, dst_quantity)
    assert g.n_edges() == 1


def test_a_long_sleep_can_drive_a_walking_count() -> None:
    """The relation the older type had no shape for, at a day-scale source."""
    g = _graph()
    edge = TypedEdge(OBSERVED, 3, LABELS.index("sleep"), DURATION, 1)
    g.add_edge(edge, dst_label=LABELS.index("walk"), dst_quantity=COUNT)
    got, ell, quantity = next(iter(g.iter_edges()))
    assert got.src_quantity == DURATION and quantity == COUNT
    assert got.src_scale == 3 and ell == LABELS.index("walk")


def test_a_coarse_source_can_feed_a_target() -> None:
    g = _graph()
    for scale in (1, 2, 4):
        g.add_edge(_observed(label=2, scale=scale), dst_label=0, dst_quantity=COUNT)
    assert g.n_edges(dst_label=0, dst_quantity=COUNT) == 3
    assert {e.src_scale for e, _, _ in g.iter_edges()} == {1, 2, 4}


def test_the_same_source_at_two_scales_is_two_edges() -> None:
    """A resolution is part of the identity, not a detail of one edge."""
    g = _graph()
    g.add_edge(_observed(scale=1), 1, COUNT)
    g.add_edge(_observed(scale=2), 1, COUNT)
    assert g.n_edges() == 2


# ---------------------------------------------------------------------------
# Lag zero, and why it is safe exactly once
# ---------------------------------------------------------------------------


def test_a_calendar_source_may_act_on_the_epoch_it_belongs_to() -> None:
    g = _graph()
    g.add_edge(_calendar(label=NODES.index("weekend"), scale=3), 1, COUNT)
    assert g.n_edges() == 1
    assert g.is_acyclic()


def test_an_observed_source_may_act_at_lag_zero_at_the_finest_scale() -> None:
    """One epoch is one slot there, so the draw order says what is already known."""
    edge = TypedEdge(OBSERVED, 1, 0, COUNT, 0)
    assert edge.is_same_epoch
    g = _graph()
    g.add_edge(edge, dst_label=1, dst_quantity=COUNT)
    assert g.is_acyclic()
    assert g.label_order() == [0, 2, 1]


@pytest.mark.parametrize("scale", [2, 3, 4])
def test_an_observed_source_may_not_act_at_lag_zero_above_the_finest_scale(
    scale,
) -> None:
    """A coarse epoch read at lag zero holds the slots after the one predicted."""
    with pytest.raises(ValueError, match="same-epoch read of an observed source"):
        TypedEdge(OBSERVED, scale, 0, COUNT, 0)


def test_a_label_may_not_read_itself_in_the_epoch_being_drawn() -> None:
    g = _graph()
    with pytest.raises(ValueError, match="into itself reads"):
        g.add_edge(TypedEdge(OBSERVED, 1, 1, COUNT, 0), 1, COUNT)
    with pytest.raises(ValueError, match="into itself reads"):
        g.set_parents(1, COUNT, [TypedEdge(OBSERVED, 1, 1, DURATION, 0)])


def test_a_calendar_source_may_also_be_lagged() -> None:
    """Whether yesterday was a workday is a legitimate question."""
    g = _graph()
    g.add_edge(_calendar(lag=1), 1, COUNT)
    assert next(iter(g.iter_edges()))[0].lag == 1


def test_a_calendar_node_is_never_a_target() -> None:
    """It has no parents, which is what makes a lag-zero edge from it safe."""
    g = _graph()
    assert set(g._parents) == {
        (ell, quantity) for ell in range(len(LABELS)) for quantity in QUANTITIES
    }
    with pytest.raises(ValueError, match="target label"):
        g.add_edge(_observed(), dst_label=len(LABELS), dst_quantity=COUNT)


def test_a_graph_of_lagged_and_calendar_edges_is_acyclic() -> None:
    g = _graph()
    for ell in range(len(LABELS)):
        g.add_edge(_calendar(label=1), ell, COUNT)
        for src in range(len(LABELS)):
            g.add_edge(_observed(label=src, lag=1), ell, COUNT)
    assert g.is_acyclic()
    g.assert_acyclic()


def test_two_labels_reading_each_other_in_one_epoch_leave_no_draw_order() -> None:
    """The one shape the same-epoch edges can take that no draw order survives."""
    g = _graph()
    g.add_edge(TypedEdge(OBSERVED, 1, 0, COUNT, 0), 1, COUNT)
    g._parents[(0, COUNT)].add(TypedEdge(OBSERVED, 1, 1, COUNT, 0))
    assert not g.is_acyclic()
    with pytest.raises(ValueError, match="close a cycle over labels"):
        g.assert_acyclic()


# ---------------------------------------------------------------------------
# What an edge refuses to be
# ---------------------------------------------------------------------------


def test_an_unknown_source_kind_is_refused() -> None:
    with pytest.raises(ValueError, match="source kind"):
        TypedEdge("guessed", 1, 0, COUNT, 1)


def test_a_calendar_source_carries_a_level_and_not_a_count() -> None:
    with pytest.raises(ValueError, match="carries 'level'"):
        TypedEdge(CALENDAR, 1, 0, COUNT, 0)


def test_an_observed_source_carries_a_count_or_a_duration() -> None:
    with pytest.raises(ValueError, match="source quantity"):
        TypedEdge(OBSERVED, 1, 0, LEVEL, 1)


def test_a_scale_index_starts_at_one() -> None:
    with pytest.raises(ValueError, match="not a scale index"):
        TypedEdge(OBSERVED, 0, 0, COUNT, 1)


def test_a_negative_label_or_lag_is_refused() -> None:
    with pytest.raises(ValueError, match="is negative"):
        TypedEdge(OBSERVED, 1, -1, COUNT, 1)
    with pytest.raises(ValueError, match="lag -2 is negative"):
        TypedEdge(OBSERVED, 1, 0, COUNT, -2)


def test_a_source_outside_its_own_namespace_is_refused() -> None:
    """An observed index and a calendar index are counted separately."""
    g = TypedGraph(label_names=["a", "b"], calendar_names=["weekend"])
    with pytest.raises(ValueError, match="observed sources"):
        g.add_edge(_observed(label=5), 0, COUNT)
    with pytest.raises(ValueError, match="calendar sources"):
        g.add_edge(_calendar(label=3), 0, COUNT)


def test_an_unknown_target_quantity_is_refused() -> None:
    g = _graph()
    with pytest.raises(ValueError, match="target quantity"):
        g.add_edge(_observed(), 0, "intensity")


# ---------------------------------------------------------------------------
# Determinism, counting and distance
# ---------------------------------------------------------------------------


def test_iteration_is_sorted_and_so_does_not_depend_on_hashing() -> None:
    g = _graph()
    edges = [
        _observed(label=2, scale=2, quantity=DURATION, lag=3),
        _observed(label=0, scale=1, quantity=COUNT, lag=1),
        _calendar(label=2, scale=3, lag=0),
    ]
    for edge in edges:
        g.add_edge(edge, 1, COUNT)
    seen = [e for e, _, _ in g.iter_edges()]
    assert seen == sorted(seen)
    assert list(g.iter_edges()) == list(g.iter_edges())


def test_removing_and_replacing_a_parent_set() -> None:
    g = _graph()
    edge = _observed()
    g.add_edge(edge, 0, COUNT)
    g.remove_edge(edge, 0, COUNT)
    assert g.n_edges() == 0
    g.remove_edge(edge, 0, COUNT)

    g.set_parents(0, DURATION, [_observed(label=1), _observed(label=2)])
    assert g.n_edges(dst_label=0) == 2
    assert g.n_edges(dst_quantity=COUNT) == 0


def test_set_parents_checks_its_sources() -> None:
    g = TypedGraph(label_names=["a"], calendar_names=[])
    with pytest.raises(ValueError, match="observed sources"):
        g.set_parents(0, COUNT, [_observed(label=4)])


def test_the_distance_to_an_identical_graph_is_zero() -> None:
    a, b = _graph(), _graph()
    for g in (a, b):
        g.add_edge(_observed(label=1, lag=2), 0, COUNT)
        g.add_edge(_calendar(label=0, scale=3), 2, DURATION)
    assert a.structural_hamming_distance(b) == 0
    assert a.edge_set() == b.edge_set()


def test_the_distance_counts_both_directions() -> None:
    a, b = _graph(), _graph()
    a.add_edge(_observed(label=1), 0, COUNT)
    b.add_edge(_observed(label=2), 0, COUNT)
    assert a.structural_hamming_distance(b) == 2

    b.add_edge(_observed(label=1), 0, COUNT)
    assert a.structural_hamming_distance(b) == 1


def test_a_pairing_difference_is_a_distance_of_two() -> None:
    """The same source into the same label is a different edge per quantity."""
    a, b = _graph(), _graph()
    a.add_edge(_observed(label=1, quantity=COUNT), 0, COUNT)
    b.add_edge(_observed(label=1, quantity=DURATION), 0, COUNT)
    assert a.structural_hamming_distance(b) == 2


def test_the_graph_reports_its_own_shape() -> None:
    g = _graph()
    assert g.L == 3 and g.n_calendar == 3
    assert g.n_edges() == 0
    assert g.is_acyclic()


# ---------------------------------------------------------------------------
# The basis term a source offers
# ---------------------------------------------------------------------------


def test_a_basis_term_is_the_log_term_or_one_band() -> None:
    """Three edges name no band, so nothing says which contrast is meant."""
    with pytest.raises(ValueError, match="the log term or one band"):
        TypedEdge(
            src_kind=OBSERVED,
            src_scale=2,
            src_label=0,
            src_quantity=COUNT,
            lag=1,
            basis=(1.0, 2.0, 3.0),
        )


def test_a_band_runs_from_its_lower_edge_upward() -> None:
    with pytest.raises(ValueError, match="from its lower edge upward"):
        TypedEdge(
            src_kind=OBSERVED,
            src_scale=2,
            src_label=0,
            src_quantity=COUNT,
            lag=1,
            basis=(3.0, 1.0),
        )


def test_a_calendar_source_carries_its_level_whole() -> None:
    """A level is already the contrast, so banding it names no new column."""
    with pytest.raises(ValueError, match="carries its level whole"):
        TypedEdge(
            src_kind=CALENDAR,
            src_scale=2,
            src_label=0,
            src_quantity=LEVEL,
            lag=0,
            basis=(1.0, 2.0),
        )
