"""
tests/test_phase98_same_epoch.py
================================
The fine half of the bidirectional walk: a finest-scale source, a same-epoch
read and a fitted regime chain.

Four claims are pinned.  A same-epoch read of another label is expressible at
the finest scale and refused above it, and what keeps it predictable is the
order `TypedGraph.label_order` puts the labels in, which the draw follows slot
by slot.  A refused move blocks its own candidate and leaves the one ranked
behind it alone, which is what lets the finest scale be a source at all.  The
arm offers the same-epoch family, never offers a label its own draw, and fits a
chain the search runs against.  And a fitted chain crosses into a draw as a
trajectory of its own, so a drawn log carries the switching the fit found and
never the posterior of the participant it was fitted on.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.cohort import Cohort
from src.core.scales import ScaleHierarchy
from src.core.typed_graph import (
    COUNT,
    DURATION,
    FINEST_SCALE,
    OBSERVED,
    TypedEdge,
    TypedGraph,
)
from src.core.variables import MultiScaleData
from src.data.label_axis import LabelAxis
from src.evaluation.round_trip import draw_offsets, regime_offset
from src.inference.bound import segment_sum
from src.learning.arm1 import (
    MAX_REFUSED,
    REGIME_SCALE,
    REGIME_STATES,
    candidate_pool,
    fit_arm1,
    regime_chains,
    target_pool,
)
from src.learning.calendar_arms import HOME
from src.learning.fine_sem import RegimeChain
from src.learning.relation_price import RelationPrice, grouped_search
from src.model.intensity import FineParams, empty_sources, sample

LABELS = ("sleep", "walk", "work")
NODES = ("hour_index",)

HOURS, DAYS, WEEKS = 24, 28, 4
T = HOURS * DAYS
DAY_OF = np.repeat(np.arange(DAYS), HOURS).astype(np.int32)
WEEK_OF = np.repeat(np.arange(WEEKS), 7).astype(np.int32)


def graph_of() -> TypedGraph:
    return TypedGraph(label_names=list(LABELS), calendar_names=list(NODES))


def same_epoch(src: int, quantity: str = COUNT) -> TypedEdge:
    return TypedEdge(OBSERVED, FINEST_SCALE, src, quantity, 0)


# ---------------------------------------------------------------------------
# The order the labels are drawn in
# ---------------------------------------------------------------------------


def test_a_graph_with_no_same_epoch_edge_keeps_the_declared_label_order():
    g = graph_of()
    g.add_edge(TypedEdge(OBSERVED, 1, 2, COUNT, 1), 0, COUNT)
    assert g.label_order() == [0, 1, 2]
    assert g.same_epoch_children() == {0: set(), 1: set(), 2: set()}


def test_the_order_draws_every_same_epoch_source_before_the_label_reading_it():
    g = graph_of()
    g.add_edge(same_epoch(2), 1, COUNT)
    g.add_edge(same_epoch(1), 0, DURATION)
    order = g.label_order()
    assert order.index(2) < order.index(1) < order.index(0)
    assert g.same_epoch_children() == {0: set(), 1: {0}, 2: {1}}


def test_two_edges_of_one_pair_are_counted_as_one_arc():
    """A count read and a duration read of one source are one ordering claim."""
    g = graph_of()
    g.add_edge(same_epoch(2, COUNT), 0, COUNT)
    g.add_edge(same_epoch(2, DURATION), 0, COUNT)
    g.add_edge(same_epoch(2, COUNT), 0, DURATION)
    assert g.label_order() == [1, 2, 0]


def test_a_cycle_over_three_labels_leaves_no_label_to_draw_first():
    g = graph_of()
    g.add_edge(same_epoch(0), 1, COUNT)
    g.add_edge(same_epoch(1), 2, COUNT)
    g._parents[(0, COUNT)].add(same_epoch(2))
    assert not g.is_acyclic()
    with pytest.raises(ValueError, match=r"close a cycle over labels \[0, 1, 2\]"):
        g.label_order()


# ---------------------------------------------------------------------------
# Which further edge the search may still hold
# ---------------------------------------------------------------------------


def test_a_lagged_edge_never_closes_a_cycle():
    g = graph_of()
    g.add_edge(same_epoch(0), 1, COUNT)
    assert not g.closes_cycle(TypedEdge(OBSERVED, 1, 1, COUNT, 1), 0)


def test_a_label_reading_itself_in_the_epoch_being_drawn_closes_a_cycle():
    assert graph_of().closes_cycle(same_epoch(1), 1)


def test_the_edge_that_would_reverse_an_accepted_one_closes_a_cycle():
    g = graph_of()
    g.add_edge(same_epoch(0), 1, COUNT)
    assert g.closes_cycle(same_epoch(1), 0)
    assert not g.closes_cycle(same_epoch(2), 0)


def test_the_walk_over_two_routes_to_one_label_visits_it_once():
    """The second arrival at a label finds it already seen and stops there."""
    g = TypedGraph(label_names=["a", "b", "c", "d"])
    g.add_edge(same_epoch(0), 1, COUNT)
    g.add_edge(same_epoch(0), 2, COUNT)
    g.add_edge(same_epoch(1), 2, COUNT)
    assert not g.closes_cycle(same_epoch(3), 0)
    assert g.closes_cycle(same_epoch(2), 0)


# ---------------------------------------------------------------------------
# The draw follows that order
# ---------------------------------------------------------------------------


def sources_of(n_slots: int = 40, n_labels: int = 3):
    ancestors = [np.arange(n_slots, dtype=np.int64)]
    calendar = [np.zeros((n_slots, 1))]
    return empty_sources(ancestors, calendar, n_labels)


def test_a_same_epoch_parent_is_written_before_the_label_that_reads_it():
    """Walking never starts without sleep in the same slot, at a large weight."""
    graph = graph_of()
    edge = same_epoch(0)
    graph.add_edge(edge, 1, COUNT)
    params = FineParams(n_labels=3)
    params.base[COUNT][:] = [0.0, -30.0, -30.0]
    params.base[DURATION][:] = -20.0
    params.coef[(1, COUNT)][edge] = 100.0
    X1, _ = sample(params, graph, sources_of(), np.random.default_rng(0))
    assert 0 < X1[:, 0].sum() < X1.shape[0]
    assert np.array_equal(X1[:, 1] > 0, X1[:, 0] > 0)


def test_the_draw_refuses_a_graph_whose_labels_have_no_order():
    graph = graph_of()
    graph.add_edge(same_epoch(0), 1, COUNT)
    graph._parents[(0, COUNT)].add(same_epoch(1))
    with pytest.raises(ValueError, match="close a cycle over labels"):
        sample(FineParams(n_labels=3), graph, sources_of(), np.random.default_rng(0))


# ---------------------------------------------------------------------------
# A refused move blocks itself and nothing else
# ---------------------------------------------------------------------------


def price_of(n: int = 3) -> RelationPrice:
    return RelationPrice(relation=0.0, representation=0.0, n_relations=n)


def toy_candidates(n: int = 3):
    return [(TypedEdge(OBSERVED, 2, src, COUNT, 1), 9) for src in range(n)]


def falling_gains(_selected):
    return np.array([3.0, 2.0, 1.0])


def test_one_refusal_ends_the_round_at_the_declared_bound():
    """The bound of one is the round that stops at its first refusal."""
    report = grouped_search(
        falling_gains,
        lambda selected: -1.0 if selected else 0.0,
        toy_candidates(),
        price_of(),
        max_refused=1,
    )
    assert report.accepted == []
    assert report.rejected == [0]


def test_a_refused_candidate_is_blocked_and_the_next_one_is_still_reached():
    def apply_fn(selected):
        return -1.0 if 0 in selected else float(len(selected))

    report = grouped_search(
        falling_gains, apply_fn, toy_candidates(), price_of(), max_refused=3
    )
    assert report.rejected == [0]
    assert report.accepted == [1, 2]


def test_the_refusal_bound_stops_a_round_that_keeps_refusing():
    report = grouped_search(
        falling_gains,
        lambda selected: -1.0 if selected else 0.0,
        toy_candidates(),
        price_of(),
        max_refused=2,
    )
    assert report.rejected == [0, 1]
    assert report.accepted == []


def test_a_candidate_the_caller_cannot_hold_is_never_proposed():
    report = grouped_search(
        falling_gains,
        lambda selected: float(len(selected)),
        toy_candidates(),
        price_of(),
        admits=[False, True, False],
    )
    assert report.accepted == [1]


# ---------------------------------------------------------------------------
# What the arm offers
# ---------------------------------------------------------------------------


def axis() -> LabelAxis:
    return LabelAxis(
        targets=LABELS, exogenous=NODES, grouped={name: name for name in LABELS}
    )


def test_the_pool_offers_a_same_epoch_read_of_every_label_at_the_finest_scale():
    pool = candidate_pool(axis(), HOME, (1, 2), max_lag=2)
    zero = [e for e in pool if e.is_same_epoch]
    assert len(zero) == len(LABELS) * 2
    assert {e.src_scale for e in zero} == {FINEST_SCALE}
    assert {e.src_quantity for e in zero} == {COUNT, DURATION}


def test_a_pool_that_does_not_search_the_finest_scale_offers_no_same_epoch_read():
    pool = candidate_pool(axis(), HOME, (2, 3), max_lag=2)
    assert not [e for e in pool if e.is_same_epoch]


def test_no_target_is_offered_the_same_epoch_read_of_its_own_draw():
    pool = candidate_pool(axis(), HOME, (1, 2), max_lag=1)
    for target in range(len(LABELS)):
        own = target_pool(pool, target)
        assert not [e for e in own if e.is_same_epoch and e.src_label == target]
        assert len(own) == len(pool) - 2


def test_the_declared_regime_and_refusal_settings():
    assert (REGIME_SCALE, REGIME_STATES, MAX_REFUSED) == (2, 2, 4)


# ---------------------------------------------------------------------------
# The cohort the arm is fitted on
# ---------------------------------------------------------------------------


def hierarchy() -> ScaleHierarchy:
    return ScaleHierarchy(K=3, H=[T, DAYS, WEEKS], parent_maps=[DAY_OF, WEEK_OF, None])


def blocks_of():
    hour = np.tile(np.arange(24, dtype=float), DAYS)[:, None]
    return [hour, np.zeros((DAYS, 1)), np.zeros((WEEKS, 1))]


def subject(seed: int) -> MultiScaleData:
    """A record whose walking follows its sleeping inside the same slot."""
    rng = np.random.default_rng(seed)
    X1 = np.zeros((T, len(LABELS)), dtype=np.int64)
    X1[:, 0] = rng.random(T) < 0.25
    X1[:, 1] = (X1[:, 0] > 0) & (rng.random(T) < 0.8)
    X1[:, 2] = rng.random(T) < 0.1
    Y1 = (X1 > 0).astype(np.int64)
    day_x = segment_sum(X1.astype(float), DAY_OF, DAYS).astype(np.int64)
    day_y = segment_sum(Y1.astype(float), DAY_OF, DAYS).astype(np.int64)
    return MultiScaleData(
        hierarchy=hierarchy(),
        label_names=list(LABELS),
        X=[
            X1,
            day_x,
            segment_sum(day_x.astype(float), WEEK_OF, WEEKS).astype(np.int64),
        ],
        Y=[
            Y1,
            day_y,
            segment_sum(day_y.astype(float), WEEK_OF, WEEKS).astype(np.int64),
        ],
        C=blocks_of(),
    )


def cohort_of(n: int = 4) -> Cohort:
    names = tuple(f"p{i}" for i in range(n))
    seeds = dict(zip(names, range(n)))
    return Cohort(
        subjects=names,
        hierarchy=hierarchy(),
        label_names=list(LABELS),
        reader=lambda name: subject(seeds[name]),
        masks=np.ones((n, T), dtype=bool),
        subject_calendar=True,
    )


def test_an_arm_declaring_one_state_carries_no_chain():
    assert regime_chains(cohort_of(), axis(), REGIME_SCALE, 1) == []


def test_the_chain_the_arm_builds_runs_on_the_grid_of_its_own_scale():
    chain = regime_chains(cohort_of(), axis(), 2, 3)[0]
    assert (chain.scale, chain.n_states) == (2, 3)
    assert chain.effect.shape == (3, len(LABELS))
    assert chain.gamma.shape == (DAYS, 3)


@pytest.fixture(scope="module")
def fitted():
    return fit_arm1(
        cohort_of(),
        axis(),
        HOME,
        scales=(1, 2),
        max_lag=1,
        coefficient_rounds=3,
        price=1.0,
        regime_scale=2,
        regime_states=2,
    )


def test_the_arm_buys_the_same_epoch_relation_the_world_holds(fitted):
    sleep, walk = LABELS.index("sleep"), LABELS.index("walk")
    pairs = {
        (edge.src_label, target)
        for edge, target in fitted.accepted
        if edge.is_same_epoch
    }
    # A same-epoch association explains the record equally well either way
    # round, so the target the search reaches first takes the orientation and
    # the reverse edge is then the one that would close a cycle.
    assert {(walk, sleep), (sleep, walk)} & pairs
    assert not {(walk, sleep), (sleep, walk)} <= pairs


def test_the_arm_reads_the_finest_scale_as_a_source(fitted):
    lagged = [
        edge
        for edge, _ in fitted.accepted
        if not edge.is_exogenous and edge.src_scale == 1 and edge.lag > 0
    ]
    assert lagged


def test_the_fitted_graph_keeps_an_order_to_draw_its_labels_in(fitted):
    assert fitted.graph.is_acyclic()
    assert sorted(fitted.graph.label_order()) == list(range(len(LABELS)))


def test_the_arm_comes_back_with_the_chain_it_fitted(fitted):
    chain = fitted.chains[0]
    assert len(fitted.chains) == 1
    assert chain.scale == 2
    assert np.allclose(chain.effect[0], 0.0)
    assert np.all(np.isfinite(chain.effect))
    assert np.allclose(np.exp(chain.log_trans).sum(axis=0), 1.0)


def test_an_arm_declaring_one_state_fits_no_chain():
    fit = fit_arm1(
        cohort_of(2),
        axis(),
        HOME,
        scales=(2,),
        max_lag=1,
        coefficient_rounds=2,
        price=1e9,
        regime_states=1,
    )
    assert fit.chains == []
    assert fit.accepted == []


# ---------------------------------------------------------------------------
# What a drawn log carries of the chain
# ---------------------------------------------------------------------------


def chain_of(persistence: float = 0.9) -> RegimeChain:
    chain = RegimeChain.uniform(
        scale=1, n_states=2, n_labels=3, n_epochs=40, persistence=persistence
    )
    chain.effect[1, :] = 2.0
    return chain


def test_a_drawn_path_holds_one_state_per_epoch_and_is_seeded():
    chain = chain_of()
    first = chain.draw(40, np.random.default_rng(0))
    again = chain.draw(40, np.random.default_rng(0))
    assert first.shape == (40,)
    assert set(np.unique(first)) <= {0, 1}
    assert np.array_equal(first, again)


def test_a_chain_over_no_epoch_draws_no_state():
    assert chain_of().draw(0, np.random.default_rng(0)).size == 0


def test_a_persistent_chain_switches_less_than_a_forgetful_one():
    steady = chain_of(0.99).draw(400, np.random.default_rng(1))
    restless = chain_of(0.5).draw(400, np.random.default_rng(1))
    assert np.diff(steady).astype(bool).sum() < np.diff(restless).astype(bool).sum()


def test_a_drawn_path_reaches_every_slot_of_the_epoch_it_belongs_to():
    chain = RegimeChain.uniform(scale=2, n_states=2, n_labels=3, n_epochs=DAYS)
    chain.effect[1, :] = 1.5
    ancestors = [np.arange(T, dtype=np.int64), DAY_OF.astype(np.int64)]
    added = chain.path_contribution(np.tile([0, 1], DAYS // 2), ancestors)
    assert added.shape == (T, 3)
    assert np.allclose(added[:HOURS], 0.0)
    assert np.allclose(added[HOURS : 2 * HOURS], 1.5)


def test_a_drawn_regime_path_adds_its_own_effect_at_every_slot():
    added = regime_offset([chain_of()], sources_of(), np.random.default_rng(0))
    assert added.shape == (40, 3)
    assert set(np.unique(added)) <= {0.0, 2.0}
    assert added.any()


def test_the_count_offset_carries_the_drawn_path_and_the_duration_one_does_not():
    graph, params, sources = graph_of(), FineParams(n_labels=3), sources_of()
    bare = draw_offsets(graph, params, sources, None, 0, None, {1: 1})
    with_chain = draw_offsets(
        graph,
        params,
        sources,
        None,
        0,
        None,
        {1: 1},
        [chain_of()],
        np.random.default_rng(0),
    )
    assert np.allclose(with_chain[DURATION], bare[DURATION])
    assert not np.allclose(with_chain[COUNT], bare[COUNT])
    assert set(np.unique(with_chain[COUNT] - bare[COUNT])) <= {0.0, 2.0}


def test_a_chain_with_no_generator_to_draw_from_is_refused():
    with pytest.raises(ValueError, match="no generator to draw"):
        draw_offsets(
            graph_of(),
            FineParams(n_labels=3),
            sources_of(),
            None,
            0,
            None,
            {1: 1},
            [chain_of()],
            None,
        )
