"""
Tests for the nested price and the two strata a search reports.

The claim the price exists for is that a label's own autocorrelation stops
taking the slots a cross-label relation was competing for, and
`test_a_self_relation_no_longer_takes_the_cross_label_budget` measures exactly
that against the flat price it replaces.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.typed_graph import CALENDAR, COUNT, LEVEL, OBSERVED, TypedEdge
from src.learning.fine_sstep import greedy_search
from src.learning.relation_price import (
    CROSS,
    SELF,
    RelationKey,
    RelationPrice,
    candidate_relations,
    grouped_search,
    relation_of,
)


def edge(src_label: int, lag: int = 1, scale: int = 1) -> TypedEdge:
    return TypedEdge(OBSERVED, scale, src_label, COUNT, lag)


def fixed(gains):
    """A search whose gains never move and whose objective always rises."""
    values = np.asarray(gains, dtype=float)

    def gain_fn(selected):
        return values

    def apply_fn(selected):
        return float(values[list(selected)].sum())

    return gain_fn, apply_fn


# ---------------------------------------------------------------------------
# The key and the price


def test_a_relation_is_the_ordered_pair_and_the_representations_collapse_into_it():
    representations = [
        edge(0, lag=1),
        edge(0, lag=3),
        edge(0, lag=1, scale=2),
        TypedEdge(OBSERVED, 1, 0, "duration", 2),
    ]
    keys = {relation_of(e, 4) for e in representations}
    assert keys == {RelationKey(OBSERVED, 0, 4)}
    assert candidate_relations([(e, 4) for e in representations]) == [
        RelationKey(OBSERVED, 0, 4)
    ]


def test_a_label_reading_its_own_history_is_the_self_stratum():
    assert relation_of(edge(2), 2).is_self
    assert relation_of(edge(2), 2).stratum == SELF
    assert not relation_of(edge(3), 2).is_self
    assert relation_of(edge(3), 2).stratum == CROSS
    # A calendar node sharing an index with its target is still exogenous.
    assert not relation_of(TypedEdge(CALENDAR, 1, 2, LEVEL, 0), 2).is_self


def test_a_refinement_never_costs_more_than_the_relation_it_sits_in():
    with pytest.raises(ValueError, match="at least as much"):
        RelationPrice(relation=1.0, representation=2.0, n_relations=4)
    with pytest.raises(ValueError, match="at least one relation"):
        RelationPrice(relation=2.0, representation=1.0, n_relations=0)


def test_the_multiplicity_price_rises_with_each_purchase_and_exhausts_the_field():
    price = RelationPrice(relation=5.0, representation=0.5, n_relations=4)
    values = [price.multiplicity(j) for j in (1, 2, 3, 4)]
    assert values == sorted(values, reverse=True)
    assert values[0] == pytest.approx(np.log(4.0))
    assert values[3] == pytest.approx(np.log(0.25))
    assert price.multiplicity(5) == float("inf")
    with pytest.raises(ValueError, match="ranked from one"):
        price.multiplicity(0)


def test_the_first_representation_pays_for_the_relation_and_the_rest_do_not():
    price = RelationPrice(relation=5.0, representation=0.5, n_relations=4)
    key = RelationKey(OBSERVED, 0, 1)
    assert price.of(key, {}) == pytest.approx(5.0 + np.log(4.0))
    assert price.of(key, {key: 1}) == 0.5
    assert price.of(key, {key: 7}) == 0.5


def test_a_self_relation_pays_the_refinement_price_alone():
    price = RelationPrice(relation=5.0, representation=0.5, n_relations=4)
    own = RelationKey(OBSERVED, 2, 2)
    assert price.of(own, {}) == 0.5
    # A self relation already bought does not advance the multiplicity either.
    other = RelationKey(OBSERVED, 0, 2)
    assert price.of(other, {own: 3}) == pytest.approx(5.0 + np.log(4.0))


# ---------------------------------------------------------------------------
# The search


def test_the_search_buys_a_relation_once_then_refines_it_cheaply():
    candidates = [(edge(0, lag=1), 9), (edge(0, lag=2), 9), (edge(1, lag=1), 9)]
    gains = [10.0, 2.0, 3.0]
    price = RelationPrice(relation=5.0, representation=0.5, n_relations=2)
    report = grouped_search(*fixed(gains), candidates, price, max_edges=3)

    assert report.accepted == [0, 1]
    assert report.relation_count(CROSS) == 1
    assert report.by_stratum() == {CROSS: 2, SELF: 0}
    assert report.prices[0] == pytest.approx(5.0 + np.log(2.0))
    assert report.prices[1] == 0.5
    # The second relation's gain of 3.0 never clears its own relation price.
    assert 2 not in report.accepted


def test_a_self_relation_no_longer_takes_the_cross_label_budget():
    # One label's own history offers two strong lags; a cross-label relation
    # offers one weaker representation, and the in-degree bound holds two slots.
    candidates = [(edge(4, lag=1), 4), (edge(4, lag=2), 4), (edge(1, lag=1), 4)]
    gains = [12.0, 11.0, 9.0]
    flat = greedy_search(*fixed(gains), len(candidates), price=1.0, max_edges=2)
    assert flat.accepted == [0, 1]

    price = RelationPrice(relation=5.0, representation=0.5, n_relations=2)
    report = grouped_search(*fixed(gains), candidates, price, max_edges=2, max_self=2)
    assert set(report.accepted) == {0, 1, 2}
    assert report.by_stratum() == {CROSS: 1, SELF: 2}


def test_the_guard_stops_a_move_whose_realized_objective_falls():
    candidates = [(edge(0), 5), (edge(1), 5)]
    gains = np.array([10.0, 9.0])

    def gain_fn(selected):
        return gains

    def apply_fn(selected):
        return -1.0 if len(selected) > 1 else float(gains[list(selected)].sum())

    price = RelationPrice(relation=1.0, representation=0.5, n_relations=2)
    report = grouped_search(gain_fn, apply_fn, candidates, price, max_edges=2)
    assert report.accepted == [0]
    assert report.rejected == [1]


def test_a_field_of_candidates_none_of_which_clears_its_price_buys_nothing():
    candidates = [(edge(0), 5), (edge(1), 5)]
    price = RelationPrice(relation=50.0, representation=40.0, n_relations=2)
    report = grouped_search(*fixed([1.0, 2.0]), candidates, price, max_edges=2)
    assert report.accepted == []
    assert report.relations == {}
    assert report.by_stratum() == {CROSS: 0, SELF: 0}


def test_a_bound_of_zero_in_both_strata_returns_before_scoring():
    candidates = [(edge(0), 1)]
    price = RelationPrice(relation=1.0, representation=0.5, n_relations=1)
    report = grouped_search(*fixed([10.0]), candidates, price, max_edges=0, max_self=0)
    assert report.accepted == []
    assert len(report.trace) == 1
