"""
Tests for the coherence gap and for the sorting baselines that grade a benchmark.

Two claims of the manuscript rest on these modules.  The first is that fitting
one granularity at a time gives answers that disagree, which needs a quantity
that is zero for the nested model and free to be positive otherwise; the stored
`RecErr` is not that quantity, since it reads the observed arrays and is
arithmetic.  The second is that the recovery comparison is not won by a
regularity of the simulation, which needs the sorting baselines to be scored by
the same code as every other method.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.graph import MultiScaleDBNGraph
from src.core.scales import ScaleHierarchy
from src.core.variables import MultiScaleData
from src.data.initialize import initialize_from_data
from src.evaluation.baselines import SORTNREGRESS, PerScaleDBN, SortnRegressStructure
from src.evaluation.coherence import (
    block_shares,
    coarse_mean,
    coherence_gap,
    predicted_means,
)
from src.evaluation.sortnregress import (
    r2_coefficients,
    r2_sort_regress,
    random_sort_regress,
    sort_regress,
    var_sort_regress,
)

L = 2
FINE, COARSE = 6, 3
PARENTS = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)


class _Summary:
    """The two posterior fields `predicted_means` reads."""

    def __init__(self, gamma, m_H):
        self.gamma = gamma
        self.m_H = m_H


def _hierarchy() -> ScaleHierarchy:
    return ScaleHierarchy(K=2, H=[FINE, COARSE], parent_maps=[PARENTS, None])


def _data(fine=None, coarse=None) -> MultiScaleData:
    """A two-scale record whose coarse arrays are the sums of its fine ones."""
    if fine is None:
        rng = np.random.default_rng(0)
        fine = rng.integers(0, 3, size=(FINE, L))
    fine = np.asarray(fine, dtype=np.int64)
    if coarse is None:
        coarse = np.zeros((COARSE, L), dtype=np.int64)
        np.add.at(coarse, PARENTS, fine)
    return MultiScaleData(
        hierarchy=_hierarchy(),
        label_names=[f"l{i}" for i in range(L)],
        X=[fine, np.asarray(coarse, dtype=np.int64)],
        Y=[fine, np.asarray(coarse, dtype=np.int64)],
        C=[np.zeros((FINE, 0)), np.zeros((COARSE, 0))],
    )


def _params(data: MultiScaleData, M: int = 1, r: int = 0, every_coarse: bool = False):
    return initialize_from_data(
        data,
        M_k=[M, M],
        r_k=[r, r],
        max_lag=[1, 1],
        d_ctx=[0, 0],
        seed=0,
        every_scale_coarse=every_coarse,
    )


def _summaries(M: int = 1, r: int = 0):
    return [
        _Summary(np.full((FINE, M), 1.0 / M), np.zeros((FINE, r))),
        _Summary(np.full((COARSE, M), 1.0 / M), np.zeros((COARSE, r))),
    ]


# ---------------------------------------------------------------- the mean


def test_the_coarse_mean_is_the_zero_inflated_product():
    data = _data()
    params = _params(data)[1]
    params.coarse_count.beta0[:] = np.log([4.0, 9.0])
    params.coarse_count.gamma0[:] = [0.0, -30.0]
    mean = coarse_mean(
        params, np.ones((COARSE, 1)), np.zeros((COARSE, 0)), np.zeros((COARSE, 0))
    )
    # A logit of zero halves the rate; a logit far below it leaves the rate whole.
    assert mean[0, 0] == pytest.approx(2.0, rel=1e-6)
    assert mean[0, 1] == pytest.approx(9.0, rel=1e-6)


def test_a_scale_that_allocates_has_no_mean_of_its_own():
    data = _data()
    with pytest.raises(ValueError, match="no mean of its own"):
        coarse_mean(
            _params(data)[0],
            np.ones((FINE, 1)),
            np.zeros((FINE, 0)),
            np.zeros((FINE, 0)),
        )


def test_a_scale_with_its_own_emission_allocates_nothing():
    data = _data()
    with pytest.raises(ValueError, match="allocates nothing"):
        block_shares(
            _params(data)[1],
            np.ones((COARSE, 1)),
            np.zeros((COARSE, 0)),
            np.zeros((COARSE, 0)),
            np.arange(COARSE),
        )


def test_the_weights_of_one_block_sum_to_one():
    data = _data()
    shares = block_shares(
        _params(data)[0],
        np.ones((FINE, 1)),
        np.zeros((FINE, 0)),
        np.zeros((FINE, 0)),
        PARENTS,
    )
    for parent in np.unique(PARENTS):
        assert shares[PARENTS == parent].sum(axis=0) == pytest.approx(np.ones(L))


def test_the_regime_and_factor_blocks_reach_the_mean():
    data = _data()
    params = _params(data, M=2, r=1)[1]
    params.coarse_count.beta0[:] = 0.0
    params.coarse_count.gamma0[:] = -30.0
    params.coarse_count.betaS[:] = [[np.log(2.0), 0.0], [np.log(8.0), 0.0]]
    params.coarse_count.betaH[:] = [[0.0, np.log(3.0)]]
    mean = coarse_mean(
        params,
        np.tile([1.0, 0.0], (COARSE, 1)),
        np.ones((COARSE, 1)),
        np.zeros((COARSE, 0)),
    )
    assert mean[0, 0] == pytest.approx(2.0, rel=1e-6)
    assert mean[0, 1] == pytest.approx(3.0, rel=1e-6)


def test_a_context_block_reaches_the_mean():
    data = _data()
    params = _params(data)[1]
    params.coarse_count.beta0[:] = 0.0
    params.coarse_count.gamma0[:] = -30.0
    params.coarse_count.betaCtx = np.array([[np.log(5.0), 0.0]])
    mean = coarse_mean(
        params, np.ones((COARSE, 1)), np.zeros((COARSE, 0)), np.ones((COARSE, 1))
    )
    assert mean[0, 0] == pytest.approx(5.0, rel=1e-6)


# -------------------------------------------------------------- the gap


def test_a_nested_model_closes_the_gap_to_machine_zero():
    data = _data()
    means = predicted_means(data, _params(data), _summaries())
    assert coherence_gap(data, means)["gap"] == pytest.approx(0.0, abs=1e-10)


def test_the_gap_is_positive_when_each_scale_carries_its_own_emission():
    data = _data()
    params = _params(data, every_coarse=True)
    # The fine scale claims its own rate in each of six epochs and the coarse
    # scale claims its own in each of three, and nothing ties the two claims.
    for scale in params:
        scale.coarse_count.beta0[:] = np.log(7.0)
        scale.coarse_count.gamma0[:] = -30.0
    means = predicted_means(data, params, _summaries())
    assert coherence_gap(data, means)["gap"] == pytest.approx(7.0, rel=1e-6)


def test_a_single_scale_ladder_has_no_pair_to_compare():
    hierarchy = ScaleHierarchy(K=1, H=[COARSE], parent_maps=[None])
    data = MultiScaleData(
        hierarchy=hierarchy,
        label_names=["a", "b"],
        X=[np.zeros((COARSE, L), dtype=np.int64)],
        Y=[np.zeros((COARSE, L), dtype=np.int64)],
        C=[np.zeros((COARSE, 0))],
    )
    assert coherence_gap(data, [np.ones((COARSE, L))]) == {"gap": 0.0}


def test_every_adjacent_pair_is_reported_on_its_own():
    data = _data()
    gaps = coherence_gap(data, predicted_means(data, _params(data), _summaries()))
    assert set(gaps) == {"gap_k1", "gap"}


def test_a_summary_with_no_factor_is_read_without_one():
    data = _data()
    summaries = [
        _Summary(np.ones((FINE, 1)), None),
        _Summary(np.ones((COARSE, 1)), None),
    ]
    means = predicted_means(data, _params(data), summaries)
    assert coherence_gap(data, means)["gap"] == pytest.approx(0.0, abs=1e-10)


# ------------------------------------------------------- the sorting baselines


def _chain(seed: int = 0, T: int = 600) -> np.ndarray:
    """Label 0 drives 1 drives 2 at lag one, with the variance growing along it."""
    rng = np.random.default_rng(seed)
    X = np.zeros((T, 3))
    for t in range(1, T):
        X[t, 0] = 0.5 * X[t - 1, 0] + rng.normal()
        X[t, 1] = 1.5 * X[t - 1, 0] + rng.normal()
        X[t, 2] = 1.5 * X[t - 1, 1] + rng.normal()
    return X


def test_a_variance_sort_recovers_a_chain_whose_variance_grows_along_it():
    edges = var_sort_regress(_chain())
    assert (0, 1, 1) in edges and (1, 1, 2) in edges


def test_the_scale_free_sort_reads_the_same_data():
    X = _chain()
    assert r2_sort_regress(X) and r2_sort_regress(X) == r2_sort_regress(X)


def test_a_random_order_is_the_floor_the_other_two_are_read_against():
    X = _chain()
    first = random_sort_regress(X, rng=np.random.default_rng(1))
    assert first == random_sort_regress(X, rng=np.random.default_rng(1))


def test_the_r2_of_a_chain_is_between_zero_and_one():
    r2 = r2_coefficients(_chain())
    assert r2.shape == (3,) and np.all(r2 >= 0.0) and np.all(r2 <= 1.0)


def test_a_single_column_explains_nothing():
    assert r2_coefficients(np.arange(20.0).reshape(-1, 1)) == pytest.approx([0.0])


def test_a_constant_column_falls_back_to_a_regression_per_column():
    X = np.column_stack([np.ones(50), np.arange(50.0), np.arange(50.0) * 2])
    r2 = r2_coefficients(X)
    assert r2[0] == 0.0
    assert r2[1] == pytest.approx(1.0, abs=1e-6)


def test_a_score_vector_of_the_wrong_length_is_refused():
    with pytest.raises(ValueError, match="one each is needed"):
        sort_regress(_chain(), np.array([0.0, 1.0]))


def test_a_series_shorter_than_its_lag_budget_selects_nothing():
    assert sort_regress(np.zeros((2, 3)), np.arange(3.0), max_lag=3) == set()


def test_a_constant_target_is_left_alone():
    X = np.column_stack([np.zeros(60), _chain(T=60)[:, 0]])
    edges = sort_regress(X, np.array([0.0, 1.0]))
    assert all(dst != 0 for _, _, dst in edges)


def test_a_longer_lag_budget_reaches_a_longer_lag():
    rng = np.random.default_rng(3)
    T = 600
    X = np.zeros((T, 2))
    for t in range(3, T):
        X[t, 0] = rng.normal()
        X[t, 1] = 2.0 * X[t - 3, 0] + 0.1 * rng.normal()
    assert (0, 3, 1) in sort_regress(X, np.var(X, axis=0), max_lag=3)


def test_a_scale_with_no_regime_block_is_read_at_its_intercept():
    data = _data()
    params = _params(data)[1]
    params.coarse_count.beta0[:] = np.log(3.0)
    params.coarse_count.gamma0[:] = -30.0
    params.coarse_count.betaS = np.zeros((0, L))
    params.coarse_count.gammaS = np.zeros((0, L))
    mean = coarse_mean(
        params, np.zeros((COARSE, 0)), np.zeros((COARSE, 0)), np.zeros((COARSE, 0))
    )
    assert mean[0, 0] == pytest.approx(3.0, rel=1e-6)


def test_two_identical_columns_fall_back_to_a_regression_per_column():
    # The correlation matrix is finite and singular, which the closed form
    # cannot invert.
    base = _chain(T=200)[:, 0]
    X = np.column_stack([base, base, base * 2.0 + 1.0])
    r2 = r2_coefficients(X)
    assert r2.shape == (3,) and np.all(np.isfinite(r2))


def test_a_target_whose_offered_lags_are_all_constant_buys_no_edge():
    # The response varies only in its last epoch, so every lagged column offered
    # to it is constant, the unpenalized fit returns zero weights, and there is
    # nothing for the Lasso to select.
    X = np.zeros((60, 2))
    X[-1, 1] = 1.0
    assert sort_regress(X, np.array([0.0, 1.0])) == set()


# --------------------------------------------------- the per-granularity arm


def test_the_per_scale_arm_gives_every_scale_its_own_emission():
    data = _data()
    params = PerScaleDBN.initial_params(_params(data, every_coarse=True))
    assert all(p.is_coarsest for p in params)


def test_the_nested_arm_leaves_every_finer_scale_allocating():
    data = _data()
    params = _params(data)
    assert [p.is_coarsest for p in params] == [False, True]


def test_a_scale_carrying_no_emission_is_refused():
    data = _data()
    params = _params(data)
    params[0].fine_count = None
    with pytest.raises(ValueError, match="carries no emission"):
        PerScaleDBN.initial_params(params)


def test_the_per_scale_arm_scores_itself_the_way_it_was_fitted():
    arm = PerScaleDBN()
    assert arm.couple_scales is False


def test_the_per_scale_arm_fits_and_predicts_every_scale_on_its_own():
    data = _data()
    graph = MultiScaleDBNGraph(hierarchy=data.hierarchy, label_names=data.label_names)
    arm = PerScaleDBN(max_outer_iter=1, do_structure=False)
    params, learned, _ = arm.fit(data, _params(data, every_coarse=True), graph)
    assert all(p.is_coarsest for p in params)
    summaries = arm.predict(data, params, learned)
    assert len(summaries) == data.K
    # The arm the gap exists to measure is free to disagree with itself.
    assert coherence_gap(data, predicted_means(data, params, summaries))["gap"] >= 0.0


# ------------------------------------------------- the sorting comparison row


def _sorting_data(seed: int = 0, T: int = 200):
    """A two-scale record whose fine series carry a chain the sorter can find."""
    chain = _chain(seed=seed, T=T)
    fine = np.abs(np.rint(chain[:, :L])).astype(np.int64)
    parents = np.repeat(np.arange(T // 4), 4).astype(np.int32)
    hierarchy = ScaleHierarchy(K=2, H=[T, T // 4], parent_maps=[parents, None])
    coarse = np.zeros((T // 4, L), dtype=np.int64)
    np.add.at(coarse, parents, fine)
    return MultiScaleData(
        hierarchy=hierarchy,
        label_names=[f"l{i}" for i in range(L)],
        X=[fine, coarse],
        Y=[fine, coarse],
        C=[np.zeros((T, 0)), np.zeros((T // 4, 0))],
    )


def _graph(data: MultiScaleData) -> MultiScaleDBNGraph:
    return MultiScaleDBNGraph(hierarchy=data.hierarchy, label_names=data.label_names)


@pytest.mark.parametrize("name", SORTNREGRESS)
def test_each_ordering_returns_a_graph_over_the_same_labels(name):
    data = _sorting_data()
    arm = SortnRegressStructure(algorithm=name)
    arm.set_search_space(max_lag={1: 1, 2: 1}, max_indegree={1: 0, 2: 0})
    learned = arm.learn_graph(data, _graph(data))
    for k in (1, 2):
        for dst in range(L):
            assert all(0 <= src < L for src, _lag in learned.get_parents(k, dst))


def test_an_ordering_the_comparison_does_not_carry_is_refused():
    with pytest.raises(ValueError, match="not a sorting baseline"):
        SortnRegressStructure(algorithm="hc-bic")


def test_the_in_degree_budget_of_the_scenario_is_respected():
    data = _sorting_data()
    arm = SortnRegressStructure(algorithm="var-sortnregress")
    arm.set_search_space(max_lag={1: 2, 2: 2}, max_indegree={1: 1, 2: 1})
    learned = arm.learn_graph(data, _graph(data))
    for k in (1, 2):
        assert all(len(learned.get_parents(k, dst)) <= 1 for dst in range(L))


def test_the_random_ordering_is_fixed_by_its_seed():
    data = _sorting_data()
    graphs = []
    for _ in range(2):
        arm = SortnRegressStructure(algorithm="random-sortnregress", seed=7)
        arm.set_search_space(max_lag={1: 1, 2: 1}, max_indegree={1: 0, 2: 0})
        learned = arm.learn_graph(data, _graph(data))
        graphs.append(
            {
                (k, dst): set(learned.get_parents(k, dst))
                for k in (1, 2)
                for dst in range(L)
            }
        )
    assert graphs[0] == graphs[1]


def test_the_row_runs_with_the_structure_search_switched_off():
    arm = SortnRegressStructure(algorithm="r2-sortnregress")
    assert arm._sem_kwargs["do_structure"] is False


def test_a_sorting_row_is_requested_by_name_and_never_by_all():
    from experiments.exp5_algorithm_comparison import resolve_algorithms
    from src.learning.classical import ALGORITHMS

    assert resolve_algorithms("var-sortnregress,r2-sortnregress") == [
        "var-sortnregress",
        "r2-sortnregress",
    ]
    # Widening `all` would change what a stored run of it contains.
    assert not set(resolve_algorithms("all")) & set(SORTNREGRESS)
    assert resolve_algorithms("all") == [a for a in ALGORITHMS if a != "empty"]
