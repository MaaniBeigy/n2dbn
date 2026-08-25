"""
Tests for the grouped price inside the pooled search.

A label's own history is declared structure, and on a real log it outscores every
cross-label relation by orders of magnitude.  Under one in-degree bound the self
lags take every slot and the cross relation is never reached.  What is pinned
here is that the two strata carry their own bounds over the pooled sum, and that
both searches read the same moves.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.cohort import Cohort
from src.core.scales import ScaleHierarchy
from src.core.typed_graph import COUNT, DURATION, OBSERVED, TypedEdge, TypedGraph
from src.core.variables import MultiScaleData
from src.inference.bound import segment_sum
from src.learning.pooled import PooledFit
from src.learning.relation_price import CROSS, SELF, RelationPrice
from src.model.intensity import FineParams

HOURS, DAYS, WEEKS = 24, 56, 8
T = HOURS * DAYS
LABELS = ["walking", "sleep", "gym"]
L = len(LABELS)
DAY_OF = np.repeat(np.arange(DAYS), HOURS).astype(np.int32)
WEEK_OF = np.repeat(np.arange(WEEKS), 7).astype(np.int32)
BUDGET = {2: 2, 3: 2}


def hierarchy() -> ScaleHierarchy:
    return ScaleHierarchy(K=3, H=[T, DAYS, WEEKS], parent_maps=[DAY_OF, WEEK_OF, None])


def arrays(seed: int):
    """A target a persistent level of its own drives, and a second label as well."""
    rng = np.random.default_rng(seed)
    X1 = np.zeros((T, L), dtype=np.int64)
    for ell in (1, 2):
        X1[:, ell] = rng.random(T) < 0.08 + 0.04 * ell
    sleep = segment_sum(X1[:, 1].astype(float), DAY_OF, DAYS)
    previous = np.concatenate([[0.0], sleep[:-1]])
    # A persistent level makes the target's own day count autocorrelated, so its
    # own lag is a real relation and the strongest one on offer.  It is drawn and
    # never fed back, so nothing here runs away.
    level = np.zeros(DAYS)
    for day in range(1, DAYS):
        level[day] = 0.8 * level[day - 1] + 0.6 * rng.normal()
    odds = -3.0 + 1.1 * level + 0.14 * previous
    X1[:, 0] = rng.random(T) < 1.0 / (1.0 + np.exp(-odds[DAY_OF]))
    return X1, (X1 > 0).astype(np.int64)


def subject(seed: int) -> MultiScaleData:
    X1, Y1 = arrays(seed)
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
        C=[np.zeros((T, 1)), np.zeros((DAYS, 1)), np.zeros((WEEKS, 1))],
    )


def cohort(n: int) -> Cohort:
    names = tuple(f"p{i:02d}" for i in range(n))
    seeds = dict(zip(names, range(n)))
    return Cohort(
        subjects=names,
        hierarchy=hierarchy(),
        label_names=list(LABELS),
        reader=lambda name: subject(seeds[name]),
        masks=np.ones((n, T), dtype=bool),
    )


def pool():
    return [
        TypedEdge(OBSERVED, k, src, quantity, lag)
        for k in (2, 3)
        for src in range(L)
        for quantity in (COUNT, DURATION)
        for lag in (1, 2)
    ]


def fitted(n: int = 6, rounds: int = 2):
    """A cohort and a fit that has taken its coefficient rounds."""
    graph = TypedGraph(label_names=list(LABELS))
    params = FineParams(n_labels=L)
    params.base[COUNT][:] = -2.0
    candidates = pool()
    fit = PooledFit(
        graph,
        params,
        [],
        BUDGET,
        targets=[0],
        candidates={0: list(candidates)},
    )
    people = cohort(n)
    for _ in range(rounds):
        fit.solve(fit.accumulate(people, step=False))
    return people, fit, candidates


def relations(edges, chosen, target: int = 0):
    """The (kind, label) of each accepted edge, which names its stratum."""
    return [
        (edges[i].src_kind, edges[i].src_label, edges[i].src_scale, edges[i].lag)
        for i in chosen
    ]


def test_the_self_stratum_carries_its_own_bound_over_the_pooled_sum():
    people, fit, edges = fitted()
    price = RelationPrice(relation=2.0, representation=0.2, n_relations=6)
    grouped = fit.search_grouped(people, price, max_edges=2, max_self=2)[0]
    tally = grouped.by_stratum()
    assert tally[SELF] == 2
    assert tally[CROSS] == 2
    # The world holds two relations: the target's own persistence and sleep.
    assert {label for _, label, _, _ in relations(edges, grouped.accepted)} == {0, 1}
    assert len(grouped.accepted) == 4


def test_the_same_bound_read_flat_holds_fewer_cross_relations():
    people, fit, edges = fitted()
    flat = fit.search(people, price=2.0, max_edges=2)[0]
    picked = relations(edges, flat.accepted)
    assert len(picked) == 2
    cross = sum(1 for _, label, _, _ in picked if label != 0)
    assert cross == 1

    other, grouped_fit, grouped_edges = fitted()
    price = RelationPrice(relation=2.0, representation=0.2, n_relations=6)
    grouped = grouped_fit.search_grouped(other, price, max_edges=2, max_self=2)[0]
    assert grouped.by_stratum()[CROSS] > cross


def test_the_two_searches_read_the_same_moves():
    people, fit, edges = fitted()
    first = fit._moves(people, 0, fit.accumulate(people, step=False))
    assert first[0] == edges
    assert callable(first[1]) and callable(first[2])
    # Nothing is selected yet, so the realized objective is the opening one.
    assert first[2]([]) == pytest.approx(first[2]([]))


def test_a_grouped_round_reports_a_relation_count_per_stratum():
    people, fit, _ = fitted()
    price = RelationPrice(relation=2.0, representation=0.2, n_relations=6)
    grouped = fit.search_grouped(people, price, max_edges=3, max_self=1)[0]
    assert grouped.relation_count(SELF) <= 1
    assert sum(grouped.by_stratum().values()) == len(grouped.accepted)
    assert len(grouped.gains) == len(grouped.accepted)
    assert len(grouped.prices) == len(grouped.accepted)


def test_a_price_no_candidate_clears_buys_nothing():
    people, fit, _ = fitted()
    price = RelationPrice(relation=1e9, representation=1e9, n_relations=6)
    grouped = fit.search_grouped(people, price, max_edges=3)[0]
    assert grouped.accepted == []
    assert grouped.by_stratum() == {CROSS: 0, SELF: 0}


def test_every_target_the_fit_carries_gets_its_own_round():
    graph = TypedGraph(label_names=list(LABELS))
    params = FineParams(n_labels=L)
    params.base[COUNT][:] = -2.0
    edges = pool()
    fit = PooledFit(
        graph, params, [], BUDGET, candidates={t: list(edges) for t in range(L)}
    )
    people = cohort(4)
    fit.solve(fit.accumulate(people, step=False))
    price = RelationPrice(relation=2.0, representation=0.2, n_relations=6)
    reports = fit.search_grouped(people, price, max_edges=2)
    assert sorted(reports) == list(range(L))
    assert all(hasattr(report, "by_stratum") for report in reports.values())
