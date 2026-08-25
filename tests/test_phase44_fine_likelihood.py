"""
tests/test_phase44_fine_likelihood.py
=====================================
One likelihood at the finest scale, and the properties that make it one.

Reconciliation is an identity here rather than an invariant, so it is asserted
for random parameters and a random graph instead of for a fitted model.  No
feature reads the epoch it predicts.  A drawn trace is a valid event log because
the risk set is part of the density and not a repair applied afterward.  And the
sampler and the density agree, which is what makes the simulator an executable
statement of the model rather than a second model.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import nbinom

from src.core.durations import aggregate
from src.core.typed_graph import (
    CALENDAR,
    COUNT,
    DURATION,
    OBSERVED,
    TypedEdge,
    TypedGraph,
)
from src.model.intensity import (
    FineParams,
    FineSources,
    ancestor_maps,
    assert_predictable,
    edge_feature,
    empty_sources,
    linear_predictor,
    mark_loglik,
    risk_set,
    running_mask,
    sample,
    start_loglik,
    total_loglik,
)

T_SLOTS = 1800
LABELS = ["walk", "sleep"]
PARENT_MAPS = [np.repeat(np.arange(180), 10), np.repeat(np.arange(18), 10)]


def _sources(n_labels: int = 2) -> FineSources:
    anc = ancestor_maps(PARENT_MAPS, T_SLOTS)
    calendar = [
        np.zeros((T_SLOTS, 0)),
        np.zeros((180, 0)),
        np.tile(np.array([[0.0], [1.0]]), (9, 1)),
    ]
    return empty_sources(anc, calendar, n_labels)


def _params(n_labels: int = 2, rate: float = -3.0, mean_len: float = 4.0):
    params = FineParams(n_labels=n_labels)
    params.base[COUNT][:] = rate
    params.base[DURATION][:] = np.log(mean_len)
    params.mark_dispersion = np.full(n_labels, 3.0)
    return params


def _graph() -> TypedGraph:
    return TypedGraph(label_names=list(LABELS), calendar_names=["weekend"])


# ---------------------------------------------------------------------------
# Reconciliation, which is now an identity
# ---------------------------------------------------------------------------


def test_coarse_arrays_are_block_sums_for_arbitrary_parameters() -> None:
    """
    Asserted over random parameters and a random graph, not over a fitted model.

    Reconciliation cannot fail here, and that is the claim: it holds for a badly
    parameterized model exactly as it holds for a good one.
    """
    rng = np.random.default_rng(4)
    for trial in range(5):
        sources, graph = _sources(), _graph()
        params = _params(rate=float(rng.uniform(-5, -2)))
        for ell in range(2):
            edge = TypedEdge(
                OBSERVED, int(rng.integers(1, 4)), int(rng.integers(0, 2)), COUNT, 1
            )
            graph.add_edge(edge, ell, COUNT)
            params.coef[(ell, COUNT)][edge] = float(rng.normal(0, 1.5))
        X1, Y1 = sample(params, graph, sources, np.random.default_rng(trial))
        for k, expected in enumerate(aggregate(X1, PARENT_MAPS)):
            assert np.array_equal(sources.counts[k], expected)
        for k, expected in enumerate(aggregate(Y1, PARENT_MAPS)):
            assert np.array_equal(sources.durations[k], expected)


def test_reconciliation_holds_with_no_edges_at_all() -> None:
    sources, graph, params = _sources(), _graph(), _params()
    X1, _ = sample(params, graph, sources, np.random.default_rng(0))
    assert np.array_equal(sources.counts[2], aggregate(X1, PARENT_MAPS)[2])
    assert sources.counts[2].sum() == X1.sum()


# ---------------------------------------------------------------------------
# Nothing reads the epoch it predicts
# ---------------------------------------------------------------------------


def test_a_coarse_feature_reads_only_completed_epochs() -> None:
    """A change inside the target's own epoch leaves the feature untouched."""
    sources = _sources()
    edge = TypedEdge(OBSERVED, 2, 0, COUNT, 1)
    before = edge_feature(edge, sources).copy()
    sources.counts[1][17, 0] += 5  # the epoch holding slots 170 to 179
    after = edge_feature(edge, sources)
    changed = np.nonzero(before != after)[0]
    assert changed.min() >= 180, "a slot inside epoch 17 saw its own epoch"
    assert set(sources.ancestors[1][changed].tolist()) == {18}


def test_an_observed_edge_at_lag_zero_is_refused_above_the_finest_scale() -> None:
    """A scale-2 epoch read at lag zero holds the slots after the one predicted."""
    with pytest.raises(ValueError, match="same-epoch read of an observed source"):
        TypedEdge(OBSERVED, 2, 0, COUNT, 0)


def test_a_non_monotone_ancestor_map_is_caught() -> None:
    """`ScaleHierarchy.validate` never runs in production, so this checks here."""
    sources = _sources()
    sources.ancestors[1] = sources.ancestors[1][::-1].copy()
    with pytest.raises(ValueError, match="not non-decreasing"):
        assert_predictable(TypedEdge(OBSERVED, 2, 0, COUNT, 1), sources)


def test_a_calendar_edge_at_lag_zero_is_predictable() -> None:
    sources = _sources()
    assert_predictable(TypedEdge(CALENDAR, 3, 0, "level", 0), sources)


def test_the_vectorized_feature_matches_a_per_slot_reference() -> None:
    """The invariant every vectorization in this project has to satisfy."""
    from src.model.intensity import _feature_at

    sources, graph, params = _sources(), _graph(), _params()
    sample(params, graph, sources, np.random.default_rng(1))
    for edge in (
        TypedEdge(OBSERVED, 1, 0, COUNT, 3),
        TypedEdge(OBSERVED, 2, 1, DURATION, 1),
        TypedEdge(OBSERVED, 3, 0, COUNT, 2),
        TypedEdge(CALENDAR, 3, 0, "level", 0),
    ):
        column = edge_feature(edge, sources)
        for t in (0, 1, 9, 10, 500, T_SLOTS - 1):
            assert column[t] == pytest.approx(_feature_at(edge, sources, t), abs=1e-12)


# ---------------------------------------------------------------------------
# A drawn trace is a valid event log
# ---------------------------------------------------------------------------


def test_a_draw_holds_no_same_label_overlap() -> None:
    sources, graph, params = _sources(), _graph(), _params(mean_len=12.0)
    X1, Y1 = sample(params, graph, sources, np.random.default_rng(2))
    assert X1.sum() > 20
    assert not ((X1 > 0) & running_mask(X1, Y1)).any()
    assert set(np.unique(X1)) <= {0, 1}


def test_every_drawn_episode_has_a_positive_length() -> None:
    sources, graph, params = _sources(), _graph(), _params()
    X1, Y1 = sample(params, graph, sources, np.random.default_rng(3))
    assert np.all(Y1[X1 > 0] >= 1)
    assert np.all(Y1[X1 == 0] == 0)


def test_the_risk_set_removes_a_measurable_share_of_the_grid() -> None:
    """With long episodes most of the grid is not at risk, which is the point."""
    sources, graph, params = _sources(), _graph(), _params(rate=-2.0, mean_len=25.0)
    X1, Y1 = sample(params, graph, sources, np.random.default_rng(5))
    at_risk = risk_set(X1, Y1)
    assert 0.2 < 1.0 - at_risk.mean() < 0.95


# ---------------------------------------------------------------------------
# The density is a density, and the sampler draws from it
# ---------------------------------------------------------------------------


def test_the_start_term_matches_the_bernoulli_it_claims_to_be() -> None:
    X1 = np.array([[1, 0], [0, 1]], dtype=np.int64)
    eta = np.array([[0.5, -1.0], [2.0, 0.25]])
    p = 1.0 / (1.0 + np.exp(-eta))
    expected = np.log(np.where(X1 > 0, p, 1.0 - p)).sum()
    assert start_loglik(X1, eta) == pytest.approx(expected, abs=1e-12)


def test_the_start_term_skips_slots_that_are_not_at_risk() -> None:
    X1 = np.array([[1], [0], [0]], dtype=np.int64)
    Y1 = np.array([[3], [0], [0]], dtype=np.int64)
    eta = np.full((3, 1), 0.4)
    at_risk = risk_set(X1, Y1)
    assert at_risk.ravel().tolist() == [True, False, False]
    assert start_loglik(X1, eta, at_risk) == pytest.approx(
        start_loglik(X1[:1], eta[:1]), abs=1e-12
    )
    assert start_loglik(X1, eta, at_risk) > start_loglik(X1, eta)


def test_the_mark_term_matches_a_shifted_negative_binomial() -> None:
    X1 = np.array([[1], [1]], dtype=np.int64)
    Y1 = np.array([[3], [7]], dtype=np.int64)
    log_nu = np.array([[np.log(4.0)], [np.log(4.0)]])
    r = np.array([3.0])
    expected = nbinom.logpmf([2, 6], 3.0, 3.0 / (3.0 + 4.0)).sum()
    assert mark_loglik(X1, Y1, log_nu, r) == pytest.approx(expected, abs=1e-10)


def test_a_record_with_no_start_carries_no_mark_term() -> None:
    X1 = np.zeros((4, 2), dtype=np.int64)
    assert mark_loglik(X1, X1, np.zeros((4, 2)), np.ones(2)) == 0.0


def test_the_sampler_and_the_density_agree_on_the_start_rate() -> None:
    """
    The empirical rate over the risk set is the probability the density states.

    This is the check that the simulator is the same model as the likelihood
    rather than a second one that resembles it.
    """
    sources, graph = _sources(), _graph()
    params = _params(rate=-2.5, mean_len=3.0)
    X1, Y1 = sample(params, graph, sources, np.random.default_rng(11))
    at_risk = risk_set(X1, Y1)
    observed = X1[at_risk].mean()
    stated = 1.0 / (1.0 + np.exp(2.5))
    assert observed == pytest.approx(stated, rel=0.15), f"{observed} against {stated}"


def test_the_total_is_the_two_terms_and_nothing_else() -> None:
    sources, graph, params = _sources(), _graph(), _params()
    X1, Y1 = sample(params, graph, sources, np.random.default_rng(6))
    eta = linear_predictor(params, graph, sources, COUNT)
    log_nu = linear_predictor(params, graph, sources, DURATION)
    at_risk = risk_set(X1, Y1)
    expected = start_loglik(X1, eta, at_risk) + mark_loglik(
        X1, Y1, log_nu, params.mark_dispersion
    )
    assert total_loglik(X1, Y1, params, graph, sources) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# What the older formulation could not express
# ---------------------------------------------------------------------------


def test_a_duration_source_drives_a_count_target() -> None:
    """A day-scale sleep duration acting on walking, which had no shape before."""
    edge = TypedEdge(OBSERVED, 3, LABELS.index("sleep"), DURATION, 1)
    walk = LABELS.index("walk")

    def draw(weight: float) -> int:
        sources, graph = _sources(), _graph()
        params = _params(rate=-4.0, mean_len=30.0)
        graph.add_edge(edge, walk, COUNT)
        params.coef[(walk, COUNT)][edge] = weight
        X1, _ = sample(params, graph, sources, np.random.default_rng(9))
        return int(X1[:, walk].sum())

    assert draw(1.0) > 2 * draw(0.0) + 5


def test_a_calendar_source_drives_a_target_at_lag_zero() -> None:
    edge = TypedEdge(CALENDAR, 3, 0, "level", 0)

    def draw(weight: float):
        sources, graph = _sources(), _graph()
        params = _params(rate=-4.0)
        graph.add_edge(edge, 0, COUNT)
        params.coef[(0, COUNT)][edge] = weight
        X1, _ = sample(params, graph, sources, np.random.default_rng(8))
        on = sources.calendar[2][sources.ancestors[2], 0] == 1.0
        return int(X1[on, 0].sum()), int(X1[~on, 0].sum())

    high, low = draw(2.0)
    assert high > low


def test_an_unknown_target_quantity_is_refused() -> None:
    sources, graph, params = _sources(), _graph(), _params()
    with pytest.raises(ValueError, match="target quantity"):
        linear_predictor(params, graph, sources, "intensity")


def test_the_predictor_takes_a_latent_offset() -> None:
    """A regime or factor enters as one array once its path is known."""
    sources, graph, params = _sources(), _graph(), _params()
    flat = linear_predictor(params, graph, sources, COUNT)
    offset = np.full((T_SLOTS, 2), 0.75)
    lifted = linear_predictor(params, graph, sources, COUNT, offsets=offset)
    assert np.allclose(lifted - flat, 0.75)


def test_sources_report_their_own_shape() -> None:
    sources = _sources()
    assert sources.n_slots == T_SLOTS and sources.n_labels == 2 and sources.K == 3
    assert sources.quantity(2, COUNT).shape == (180, 2)
    assert sources.quantity(2, DURATION).shape == (180, 2)


def test_the_predictor_sums_its_edges() -> None:
    """The vectorized predictor equals the terms it claims to add."""
    sources, graph = _sources(), _graph()
    params = _params()
    edges = {
        TypedEdge(OBSERVED, 1, 1, COUNT, 2): 0.7,
        TypedEdge(OBSERVED, 3, 0, DURATION, 1): -0.4,
        TypedEdge(CALENDAR, 3, 0, "level", 0): 1.1,
    }
    sample(params, graph, sources, np.random.default_rng(12))
    for edge, weight in edges.items():
        graph.add_edge(edge, 0, COUNT)
        params.coef[(0, COUNT)][edge] = weight
    # A parent carrying a zero coefficient contributes nothing and is skipped.
    idle = TypedEdge(OBSERVED, 2, 1, COUNT, 1)
    graph.add_edge(idle, 0, COUNT)
    params.coef[(0, COUNT)][idle] = 0.0

    got = linear_predictor(params, graph, sources, COUNT)
    expected = np.full(T_SLOTS, params.base[COUNT][0])
    for edge, weight in edges.items():
        expected = expected + weight * edge_feature(edge, sources)
    assert np.allclose(got[:, 0], expected)
    assert np.allclose(got[:, 1], params.base[COUNT][1])


def test_the_sampler_takes_a_latent_offset_on_both_predictors() -> None:
    def draw(count_offset: float, duration_offset: float):
        sources, graph, params = _sources(), _graph(), _params(rate=-4.0)
        offsets = {
            COUNT: np.full((T_SLOTS, 2), count_offset),
            DURATION: np.full((T_SLOTS, 2), duration_offset),
        }
        X1, Y1 = sample(
            params, graph, sources, np.random.default_rng(13), offsets=offsets
        )
        mean_len = Y1[X1 > 0].mean() if X1.any() else 0.0
        return int(X1.sum()), float(mean_len)

    quiet_n, quiet_len = draw(0.0, 0.0)
    loud_n, loud_len = draw(2.0, 1.0)
    assert loud_n > quiet_n
    assert loud_len > quiet_len


def test_a_parent_can_lengthen_an_episode() -> None:
    """An edge into the duration target, which is the second predictor."""
    edge = TypedEdge(OBSERVED, 3, 1, COUNT, 1)

    def draw(weight: float) -> float:
        sources, graph, params = _sources(), _graph(), _params(rate=-3.0)
        graph.add_edge(edge, 0, DURATION)
        params.coef[(0, DURATION)][edge] = weight
        X1, Y1 = sample(params, graph, sources, np.random.default_rng(14))
        return float(Y1[X1[:, 0] > 0, 0].mean())

    assert draw(0.8) > draw(0.0) * 1.3
