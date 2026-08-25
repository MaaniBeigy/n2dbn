"""
Exact and evolutionary parent-set search.

Greedy forward-backward accepts one edge at a time, so a pair of edges that
only pays jointly is unreachable from the empty set.  These tests pin the two
searches that are not bound by that, on a score built so the joint pair is the
only optimum, plus the dispatch and budget logic that chooses between them.

1.  `n_subsets` counts every subset within the in-degree bound
2.  `n_subsets` saturates when the bound exceeds the candidate count
3.  `ExactSearch` finds a pair that greedy cannot reach
4.  `ExactSearch` returns the empty set when nothing scores above it
5.  `ExactSearch` respects the in-degree bound
6.  `GeneticSearch` finds the same pair
7.  `GeneticSearch` is deterministic under a seed
8.  `GeneticSearch` returns the empty set with no candidates
9.  `GeneticSearch` respects the in-degree bound
10. `GeneticSearch` mutation reaches drop, add and swap
11. `GeneticSearch` crossover respects the in-degree bound
12. `SStep(search='exact')` dispatches to the exact search
13. `SStep(search='genetic')` dispatches to the evolutionary search
14. `SStep(search='auto')` enumerates when the space is small
15. `SStep(search='auto')` falls back to evolution when the space is large
"""

import numpy as np
import pytest

from src.core.graph import MultiScaleDBNGraph
from src.learning.sstep import ExactSearch, GeneticSearch, SStep, n_subsets

from .test_phase09_coverage import (  # noqa: F401
    mini_ds,
    mini_gen,
    mini_graph,
    mini_qs,
)

CANDS = [(0, 1), (1, 1), (2, 1), (3, 1)]
PAIR = {(0, 1), (1, 1)}


def pair_score(parent_set):
    """Only the joint pair pays; either edge alone is a loss."""
    return 5.0 if set(parent_set) == PAIR else -1.0 * len(parent_set)


# ---------------------------------------------------------------------------
# 1-2.  n_subsets
# ---------------------------------------------------------------------------


def test_n_subsets_counts_bounded_subsets():
    """Sum of binomials up to the bound, empty set included."""
    assert n_subsets(4, 2) == 1 + 4 + 6
    assert n_subsets(11, 3) == 1 + 11 + 55 + 165


def test_n_subsets_saturates_at_candidate_count():
    """A bound above the candidate count counts the full power set."""
    assert n_subsets(3, 10) == 2**3


# ---------------------------------------------------------------------------
# 3-5.  ExactSearch
# ---------------------------------------------------------------------------


def test_exact_search_finds_joint_pair():
    """Enumeration reaches an optimum no single-edge step can approach."""
    assert ExactSearch(max_indegree=2).search(pair_score, CANDS) == PAIR


def test_exact_search_returns_empty_when_nothing_pays():
    """Every non-empty set scores below the empty one, so none is accepted."""
    assert (
        ExactSearch(max_indegree=3).search(lambda ps: -float(len(ps)), CANDS) == set()
    )


def test_exact_search_respects_indegree():
    """A score rewarding size cannot push the result past the bound."""
    best = ExactSearch(max_indegree=2).search(lambda ps: float(len(ps)), CANDS)
    assert len(best) == 2


# ---------------------------------------------------------------------------
# 6-11.  GeneticSearch
# ---------------------------------------------------------------------------


def test_genetic_search_finds_joint_pair():
    """Crossover and mutation both put the two edges in one individual."""
    best = GeneticSearch(max_indegree=2, seed=0).search(set(), pair_score, CANDS)
    assert best == PAIR


def test_genetic_search_is_deterministic():
    """Two runs at one seed give one answer."""
    kw = dict(max_indegree=3, population=10, generations=4, seed=7)
    a = GeneticSearch(**kw).search(set(), pair_score, CANDS)
    b = GeneticSearch(**kw).search(set(), pair_score, CANDS)
    assert a == b


def test_genetic_search_no_candidates():
    """Nothing to select from returns the empty set."""
    assert GeneticSearch().search(set(), pair_score, []) == set()


def test_genetic_search_respects_indegree():
    """The bound holds even when the score rewards every extra edge."""
    best = GeneticSearch(max_indegree=2, seed=1).search(
        {(0, 1)}, lambda ps: float(len(ps)), CANDS
    )
    assert len(best) <= 2


def test_genetic_mutation_reaches_every_move():
    """Drop, add and swap are all reachable across seeds."""
    searcher = GeneticSearch(max_indegree=3)
    sizes, swapped = set(), False
    for seed in range(40):
        rng = np.random.default_rng(seed)
        out = searcher._mutate(rng, {(0, 1), (1, 1)}, CANDS)
        sizes.add(len(out))
        if len(out) == 2 and out != {(0, 1), (1, 1)}:
            swapped = True
    assert sizes == {1, 2, 3} and swapped
    # A full set has no room to add, and an empty one has nothing to drop.
    full = searcher._mutate(np.random.default_rng(0), set(CANDS[:3]), CANDS[:3])
    assert len(full) <= 3
    assert searcher._mutate(np.random.default_rng(0), set(), []) == set()


def test_genetic_crossover_respects_indegree():
    """A union wider than the bound is cut back to it."""
    searcher = GeneticSearch(max_indegree=2)
    child = searcher._crossover(
        np.random.default_rng(0), set(CANDS[:2]), set(CANDS[2:]), CANDS
    )
    assert len(child) <= 2
    assert searcher._crossover(np.random.default_rng(0), set(), set(), CANDS) == set()


# ---------------------------------------------------------------------------
# 12-15.  SStep dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("search", ["exact", "genetic", "auto"])
def test_sstep_search_dispatch(mini_ds, mini_graph, mini_qs, search):  # noqa: F811
    """Every search returns a graph whose parent sets honor the bound."""
    q_coarse, _ = mini_qs
    K = mini_ds.data.K
    sstep = SStep(
        lam=1.0, max_indegree=1, max_lag=1, search=search, population=8, generations=3
    )
    graph, params = sstep.run(
        k=K,
        q=q_coarse,
        params=mini_ds.params[K - 1],
        graph=mini_graph,
        data=mini_ds.data,
    )
    assert isinstance(graph, MultiScaleDBNGraph)
    for ell in range(params.L):
        assert len(params.parent_sets.get(ell, set())) <= 1


def test_sstep_auto_falls_back_to_genetic(mini_ds, mini_graph, mini_qs):  # noqa: F811
    """A budget of zero puts every label past the enumeration limit."""
    q_coarse, _ = mini_qs
    K = mini_ds.data.K
    sstep = SStep(
        lam=1.0,
        max_indegree=1,
        max_lag=1,
        search="auto",
        exact_budget=0,
        population=8,
        generations=3,
    )
    graph, _ = sstep.run(
        k=K,
        q=q_coarse,
        params=mini_ds.params[K - 1],
        graph=mini_graph,
        data=mini_ds.data,
    )
    assert isinstance(graph, MultiScaleDBNGraph)


# ---------------------------------------------------------------------------
# A parent set scored at its own level
# ---------------------------------------------------------------------------


def test_a_free_intercept_scores_the_empty_set_at_its_own_level(mini_ds, mini_qs):
    """Both hypotheses of the comparison carry a level, so both may fit one."""
    from src.learning.sstep import LocalScore

    K = mini_ds.data.K
    params = mini_ds.params[K - 1]
    args = (
        mini_ds.data.get_X(K),
        mini_ds.data.get_Y(K),
        mini_qs[0],
        mini_ds.data.get_C(K),
        params,
    )
    fixed = LocalScore.compute_coarsest(set(), 0, *args, lam=0.0, free_intercept=False)
    free = LocalScore.compute_coarsest(set(), 0, *args, lam=0.0, free_intercept=True)
    assert free >= fixed - 1e-6, "fitting a level cannot lower the maximum"


def test_a_free_intercept_still_returns_the_candidate_coefficients(mini_ds, mini_qs):
    from src.learning.sstep import LocalScore

    K = mini_ds.data.K
    params = mini_ds.params[K - 1]
    args = (
        mini_ds.data.get_X(K),
        mini_ds.data.get_Y(K),
        mini_qs[0],
        mini_ds.data.get_C(K),
        params,
    )
    scored = LocalScore.compute_coarsest(
        {(1, 1)}, 0, *args, lam=0.0, free_intercept=True
    )
    assert np.isfinite(scored)


def test_the_level_is_not_charged_as_a_parent(mini_ds, mini_qs):
    # The intercept is free under both hypotheses, so it carries no penalty and
    # the penalised score of the empty set is its unpenalised score.
    from src.learning.sstep import LocalScore

    K = mini_ds.data.K
    params = mini_ds.params[K - 1]
    args = (
        mini_ds.data.get_X(K),
        mini_ds.data.get_Y(K),
        mini_qs[0],
        mini_ds.data.get_C(K),
        params,
    )
    unpenalised = LocalScore.compute_coarsest(
        set(), 0, *args, lam=0.0, free_intercept=True
    )
    penalised = LocalScore.compute_coarsest(
        set(), 0, *args, lam=5.0, free_intercept=True
    )
    assert penalised == pytest.approx(unpenalised)


# ---------------------------------------------------------------------------
# The two floors an edge has to clear
# ---------------------------------------------------------------------------


def _floor_scorer(mini_ds, mini_qs, lam):
    from src.learning.sstep import _CoarseLabelScorer

    K = mini_ds.data.K
    return _CoarseLabelScorer(
        0,
        mini_ds.data.get_X(K),
        mini_ds.data.get_Y(K),
        mini_qs[0],
        mini_ds.data.get_C(K),
        mini_ds.params[K - 1],
        lam,
        [(1, 1)],
    )


def test_the_penalty_floor_is_zero_because_the_gain_already_carries_it(
    mini_ds, mini_qs
):
    # A diagnostic that rebuilds the penalty can disagree with the search, which
    # is what sent one change the wrong way, so the gain is reported as the
    # search sees it and the bar is zero.
    from src.learning.sstep import candidate_floors

    got = candidate_floors(
        _floor_scorer(mini_ds, mini_qs, 1.0),
        (1, 1),
        set(),
        np.random.default_rng(0),
        n_perm=8,
    )
    assert got["penalty_floor"] == 0.0


def test_an_edge_the_penalty_rejects_is_named_as_such(mini_ds, mini_qs):
    from src.learning.sstep import candidate_floors

    got = candidate_floors(
        _floor_scorer(mini_ds, mini_qs, 1e6),
        (1, 1),
        set(),
        np.random.default_rng(0),
        n_perm=8,
    )
    assert got["gain"] < 0.0
    assert got["binding"] == "penalty"


def test_the_binding_floor_is_the_higher_of_the_two(mini_ds, mini_qs):
    from src.learning.sstep import candidate_floors

    got = candidate_floors(
        _floor_scorer(mini_ds, mini_qs, 0.0),
        (1, 1),
        set(),
        np.random.default_rng(0),
        n_perm=8,
    )
    if got["binding"] == "none":
        assert got["gain"] > got["shift_floor"] and got["gain"] > 0.0
    elif got["binding"] == "confirmation":
        assert 0.0 < got["gain"] <= got["shift_floor"]
    else:
        assert got["gain"] <= 0.0


def test_an_edge_a_shifted_copy_of_itself_matches_is_named_confirmation():
    """The middle branch: it clears the penalty and not its own shifted null."""
    from src.learning.sstep import candidate_floors

    class _Columns(dict):
        def set(self, key, value):
            self[key] = value

    class _Stub:
        """A scorer whose real edge gains less than any shifted copy of it."""

        def __init__(self):
            col = _Columns({(1, 1): np.zeros((32, 1))})
            self.feat_x, self.feat_y = col, _Columns(dict(col))
            self.b_init, self.a_init = {}, {}

        def score(self, parent_set):
            if not parent_set:
                return 0.0
            return 1.0 if parent_set == {(1, 1)} else 5.0

    got = candidate_floors(_Stub(), (1, 1), set(), np.random.default_rng(0), n_perm=4)
    assert got["gain"] == pytest.approx(1.0)
    assert got["shift_floor"] == pytest.approx(5.0)
    assert got["binding"] == "confirmation"


def test_an_edge_that_beats_its_own_shifted_copies_clears_both_floors():
    """The remaining branch: nothing binds, which is what an accepted edge looks like."""
    from src.learning.sstep import candidate_floors

    class _Columns(dict):
        def set(self, key, value):
            self[key] = value

    class _Stub:
        """A scorer whose real edge gains more than any shifted copy of it."""

        def __init__(self):
            col = _Columns({(1, 1): np.zeros((32, 1))})
            self.feat_x, self.feat_y = col, _Columns(dict(col))
            self.b_init, self.a_init = {}, {}

        def score(self, parent_set):
            if not parent_set:
                return 0.0
            return 9.0 if parent_set == {(1, 1)} else 2.0

    got = candidate_floors(_Stub(), (1, 1), set(), np.random.default_rng(0), n_perm=4)
    assert got["gain"] == pytest.approx(9.0)
    assert got["binding"] == "none"
