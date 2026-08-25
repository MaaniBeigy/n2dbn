"""
The published structure learners and the lagged design they search.

Two kinds of assertion appear here.  The mechanical ones check that every
algorithm returns parent sets inside the declared search space and inside the
in-degree budget, which is what makes the comparison in experiment 5 a fair one.
The recovery ones check that on a series built with two known lagged edges and
nothing else, the algorithms find those edges, which is what makes a failure on
the benchmark grid attributable to the grid rather than to the wiring.
"""

from __future__ import annotations

import builtins
import importlib

import numpy as np
import pytest

from src.learning import classical as C

pgmpy = pytest.importorskip("pgmpy")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _series(T=1500, L=4, seed=0):
    """
    Binary series whose only dependence is `(1, 1) -> 0` and `(0, 2) -> 2`.

    Labels 1 and 3 are independent noise, so an algorithm that reports a parent
    for either of them is reporting structure the data does not contain.
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((T, L), dtype=np.int64)
    for t in range(2, T):
        X[t, 0] = rng.random() < 0.10 + 0.60 * (X[t - 1, 1] > 0)
        X[t, 1] = rng.random() < 0.25
        X[t, 2] = rng.random() < 0.10 + 0.60 * (X[t - 2, 0] > 0)
        X[t, 3] = rng.random() < 0.25
    return X


TRUE_EDGES = {0: {(1, 1)}, 1: set(), 2: {(0, 2)}, 3: set()}


@pytest.fixture(scope="module")
def series():
    return _series()


@pytest.fixture(scope="module")
def short_series():
    """Small enough that the slowest algorithm still finishes in seconds."""
    return _series(T=300)


@pytest.fixture(scope="module")
def tiny_series():
    """Two labels and one lag, because `mmhc` costs grow steeply in node count."""
    rng = np.random.default_rng(2)
    T = 200
    X = np.zeros((T, 2), dtype=np.int64)
    for t in range(1, T):
        X[t, 0] = rng.random() < 0.10 + 0.60 * (X[t - 1, 1] > 0)
        X[t, 1] = rng.random() < 0.25
    return X


# ---------------------------------------------------------------------------
# Encodings and the design
# ---------------------------------------------------------------------------


def test_binary_encoding_marks_presence():
    X = np.array([[0, 3], [7, 0]])
    assert C.encode_counts(X, "binary").tolist() == [[0, 1], [1, 0]]


def test_clip_encoding_collapses_the_tail():
    X = np.array([[0, 9], [2, 4]])
    assert C.encode_counts(X, "clip", n_bins=3).tolist() == [[0, 2], [2, 2]]


def test_quantile_encoding_stays_inside_the_alphabet():
    rng = np.random.default_rng(1)
    X = rng.poisson(3.0, size=(400, 2))
    codes = C.encode_counts(X, "quantile", n_bins=4)
    assert codes.min() >= 0 and codes.max() <= 3


def test_unknown_encoding_is_rejected():
    with pytest.raises(ValueError, match="unknown encoding"):
        C.encode_counts(np.zeros((3, 1)), "sqrt")


def test_lagged_frame_shape_and_names(series):
    lf = C.build_lagged_frame(series, max_lag=2)
    assert lf.frame.shape == (series.shape[0] - 2, series.shape[1] * 3)
    assert lf.current == {0: "x0_t", 1: "x1_t", 2: "x2_t", 3: "x3_t"}
    assert lf.lagged[(1, 2)] == "x1_lag2"
    assert len(lf.allowed_edges) == len(lf.lagged) * len(lf.current)


def test_lagged_columns_hold_the_shifted_series(series):
    lf = C.build_lagged_frame(series, max_lag=2)
    got = np.asarray(lf.frame["x0_lag1"]).astype(int)
    want = (series[1:-1, 0] > 0).astype(int)
    assert np.array_equal(got, want)


def test_a_series_shorter_than_the_lag_is_rejected():
    with pytest.raises(ValueError, match="cannot support a lag"):
        C.build_lagged_frame(np.zeros((2, 2), dtype=int), max_lag=3)


def test_parent_sets_ignore_edges_outside_the_space(series):
    lf = C.build_lagged_frame(series, max_lag=1)
    got = lf.parent_sets(
        [("x1_lag1", "x0_t"), ("x0_t", "x1_t"), ("junk", "x0_t")], n_labels=4
    )
    assert got == {0: {(1, 1)}, 1: set(), 2: set(), 3: set()}


def test_constant_columns_are_dropped():
    X = np.zeros((50, 3), dtype=np.int64)
    X[:, 0] = np.arange(50) % 2
    lf = C._drop_constant(C.build_lagged_frame(X, max_lag=1))
    assert lf.current == {0: "x0_t"}
    assert set(lf.lagged) == {(0, 1)}


def test_dropping_a_middle_column_keeps_the_label_indices():
    """
    A positional list would report label 2's parents under index 1 here.

    Label 1 is constant, so its columns go, and what is left has to stay keyed by
    the label it belongs to rather than by its position in the surviving set.
    """
    X = np.zeros((60, 3), dtype=np.int64)
    X[:, 0] = np.arange(60) % 2
    X[:, 2] = (np.arange(60) // 2) % 2
    lf = C._drop_constant(C.build_lagged_frame(X, max_lag=1))
    assert lf.current == {0: "x0_t", 2: "x2_t"}
    got = lf.parent_sets([("x0_lag1", "x2_t")], n_labels=3)
    assert got == {0: set(), 1: set(), 2: {(0, 1)}}


def test_a_constant_middle_label_does_not_shift_the_others():
    """The same alignment, end to end through an algorithm."""
    X = _series(T=800, seed=11)
    X[:, 1] = 0
    got = C.learn_structure("exhaustive", X, max_lag=2, max_indegree=2)
    assert set(got) == {0, 1, 2, 3}
    assert got[1] == set()
    assert (0, 2) in got[2]


def test_dropping_is_a_no_op_when_every_column_varies(series):
    lf = C.build_lagged_frame(series, max_lag=1)
    assert C._drop_constant(lf) is lf


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------


def _found(algorithm, X, **kw):
    return C.learn_structure(algorithm, X, max_lag=2, max_indegree=2, **kw)


@pytest.mark.parametrize(
    "algorithm",
    ["hc-bic", "hc-bdeu", "hc-k2", "pc", "exhaustive", "dynotears", "granger"],
)
def test_consistent_criteria_recover_the_graph_exactly(algorithm, series):
    assert _found(algorithm, series) == TRUE_EDGES


@pytest.mark.parametrize("algorithm", ["hc-aic", "ges"])
def test_liberal_criteria_find_the_true_edges_and_add_to_them(algorithm, series):
    """AIC and the equivalence-class search pay too little per parameter here."""
    found = _found(algorithm, series)
    assert all(TRUE_EDGES[ell] <= found[ell] for ell in TRUE_EDGES)
    assert sum(len(p) for p in found.values()) > 2


@pytest.mark.parametrize("algorithm", ["pc", "ges", "tree"])
def test_global_algorithms_stay_inside_the_search_space(algorithm, series):
    found = _found(algorithm, series)
    assert set(found) == {0, 1, 2, 3}
    for parents in found.values():
        assert all(1 <= lag <= 2 and 0 <= src < 4 for src, lag in parents)


@pytest.mark.parametrize("algorithm", [a for a in C.ALGORITHMS if a != "mmhc"])
def test_every_algorithm_respects_the_indegree_budget(algorithm, short_series):
    found = C.learn_structure(algorithm, short_series, max_lag=2, max_indegree=1)
    assert set(found) == {0, 1, 2, 3}
    assert all(len(p) <= 1 for p in found.values())


def test_mmhc_stays_inside_the_search_space_and_the_budget(tiny_series):
    """
    Recovery is not asserted here, and the reason is worth stating.

    `mmhc` takes no edge whitelist, so its lagged edges are whatever survives
    orienting an undirected skeleton, and that orientation is settled by tie
    breaking over sets of variable names.  String hashing is randomized per
    process, so the same call on the same data returns the true edge under some
    values of `PYTHONHASHSEED` and the empty set under others.  What holds for
    every value is that the answer sits inside the declared space.
    """
    found = C.learn_structure("mmhc", tiny_series, max_lag=1, max_indegree=1)
    assert set(found) == {0, 1}
    assert all(len(p) <= 1 for p in found.values())
    assert all(key == (1, 1) or key == (0, 1) for p in found.values() for key in p)


def test_the_empty_algorithm_reports_nothing(series):
    assert _found("empty", series) == {ell: set() for ell in range(4)}


def test_a_zero_indegree_budget_reports_nothing(series):
    got = C.learn_structure("hc-bic", series, max_lag=2, max_indegree=0)
    assert got == {ell: set() for ell in range(4)}


def test_an_all_constant_series_reports_nothing():
    X = np.zeros((60, 3), dtype=np.int64)
    assert C.learn_structure("hc-bic", X, max_lag=1, max_indegree=2) == {
        0: set(),
        1: set(),
        2: set(),
    }


def test_a_label_with_no_variation_keeps_an_empty_parent_set():
    X = _series(T=400, seed=5)
    X[:, 3] = 0
    got = C.learn_structure("hc-bic", X, max_lag=1, max_indegree=2)
    assert got[3] == set()


def test_an_unknown_algorithm_is_rejected(series):
    with pytest.raises(ValueError, match="unknown algorithm"):
        C.learn_structure("notears-plus", series, max_lag=1, max_indegree=1)


def test_quantile_encoding_runs_end_to_end():
    rng = np.random.default_rng(3)
    X = rng.poisson(2.0, size=(600, 3))
    got = C.learn_structure(
        "hc-bic", X, max_lag=1, max_indegree=1, encoding="quantile", n_bins=3
    )
    assert set(got) == {0, 1, 2}


# ---------------------------------------------------------------------------
# Trimming and local scores
# ---------------------------------------------------------------------------


def test_trimming_keeps_the_higher_scoring_parent(series):
    lf = C.build_lagged_frame(series, max_lag=2)
    over_full = {0: {(1, 1), (3, 1), (3, 2)}, 1: set(), 2: set(), 3: set()}
    trimmed = C._trim_to_indegree(lf, over_full, max_indegree=1)
    assert trimmed[0] == {(1, 1)}


def test_trimming_is_a_no_op_inside_the_budget(series):
    lf = C.build_lagged_frame(series, max_lag=2)
    parents = {0: {(1, 1)}, 1: set(), 2: set(), 3: set()}
    assert C._trim_to_indegree(lf, parents, max_indegree=2) is parents


def test_trimming_drops_a_key_the_frame_no_longer_holds(series):
    lf = C._drop_constant(C.build_lagged_frame(series, max_lag=2))
    parents = {0: {(1, 1), (3, 1)}, 1: set(), 2: set(), 3: set()}
    trimmed = C._trim_to_indegree(lf, parents, max_indegree=1)
    assert len(trimmed[0]) <= 1


def test_greedy_selection_stops_when_the_score_stops_rising():
    def score(_target, parents):
        return float(len(parents)) if len(parents) <= 1 else -1.0

    chosen = C._greedy_by_score(score, "y", ["a", "b", "c"], max_indegree=3)
    assert len(chosen) == 1


def test_greedy_selection_stops_when_the_pool_runs_out():
    def score(_target, parents):
        return float(len(parents))

    chosen = C._greedy_by_score(score, "y", ["a", "b"], max_indegree=5)
    assert chosen == {"a", "b"}


# ---------------------------------------------------------------------------
# The continuous-optimization and time-series learners
# ---------------------------------------------------------------------------


def test_the_design_is_centered_and_scaled(series):
    W, Y, keys = C._design(series, max_lag=2)
    assert W.shape == (series.shape[0] - 2, series.shape[1] * 2)
    assert Y.shape == (series.shape[0] - 2, series.shape[1])
    assert np.allclose(W.mean(axis=0), 0.0, atol=1e-10)
    assert np.allclose(W.std(axis=0), 1.0, atol=1e-10)
    assert keys[0] == (0, 1) and keys[1] == (0, 2)


def test_a_constant_design_column_does_not_divide_by_zero():
    X = np.zeros((40, 2), dtype=np.int64)
    X[:, 0] = np.arange(40) % 3
    W, _, _ = C._design(X, max_lag=1)
    assert np.all(np.isfinite(W))


def test_fista_matches_the_ridgeless_least_squares_solution_without_a_penalty():
    rng = np.random.default_rng(0)
    W = rng.normal(size=(300, 4))
    truth = np.array([[1.0], [0.0], [-2.0], [0.5]])
    Y = W @ truth + 0.01 * rng.normal(size=(300, 1))
    got = C._fista(W, Y, lam=0.0, max_iter=5000, tol=1e-12)
    want, *_ = np.linalg.lstsq(W, Y, rcond=None)
    assert np.allclose(got, want, atol=1e-3)


def test_fista_sets_small_coefficients_to_exactly_zero():
    rng = np.random.default_rng(0)
    W = rng.normal(size=(400, 3))
    Y = (W @ np.array([[2.0], [0.0], [0.0]])) + 0.05 * rng.normal(size=(400, 1))
    got = C._fista(W, Y, lam=0.2, max_iter=2000, tol=1e-12)
    assert got[1, 0] == 0.0 and got[2, 0] == 0.0 and abs(got[0, 0]) > 1.0


def test_dynotears_drops_every_edge_under_a_large_penalty(series):
    got = C._learn_dynotears([series], max_lag=2, max_indegree=2, lam=10.0)
    assert all(not p for p in got.values())


def test_dynotears_falls_back_to_numpy_for_an_unusable_device(series, monkeypatch):
    monkeypatch.setattr(C, "_torch_device_available", lambda _d: False)
    assert C._use_torch("cuda") is False
    assert C._learn_dynotears([series], 2, 2, device="cuda") == TRUE_EDGES


def test_the_default_device_stays_on_numpy():
    """`cpu` means numpy here, so the default path never imports torch."""
    assert C._use_torch("cpu") is False


def test_granger_reports_nothing_on_white_noise():
    rng = np.random.default_rng(7)
    X = rng.poisson(1.0, size=(800, 3))
    got = C._learn_granger([X], max_lag=2, max_indegree=2)
    assert all(not p for p in got.values())


def test_residual_sum_of_squares_is_zero_for_an_exact_fit():
    A = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    y = A @ np.array([2.0, -1.0])
    assert C._residual_sum_of_squares(A, y) == pytest.approx(0.0, abs=1e-20)


# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------


def test_available_algorithms_lists_everything_when_pgmpy_is_installed():
    assert C.available_algorithms() == list(C.ALGORITHMS)


def _block_import(monkeypatch, blocked):
    real = builtins.__import__

    def fake(name, *args, **kwargs):
        if name == blocked or name.startswith(blocked + "."):
            raise ImportError(f"blocked {name}")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake)


def test_available_algorithms_drops_the_pgmpy_entries_without_pgmpy(monkeypatch):
    _block_import(monkeypatch, "pgmpy")
    monkeypatch.delitem(importlib.sys.modules, "pgmpy", raising=False)
    assert C.available_algorithms() == list(C.NATIVE_ALGORITHMS)


def test_a_pgmpy_algorithm_is_refused_without_pgmpy(monkeypatch, series):
    monkeypatch.setattr(C, "_pgmpy_available", lambda: False)
    with pytest.raises(ValueError, match="needs pgmpy"):
        C.learn_structure("hc-bic", series, max_lag=1, max_indegree=1)


def test_a_native_algorithm_still_runs_without_pgmpy(monkeypatch, series):
    monkeypatch.setattr(C, "_pgmpy_available", lambda: False)
    assert C.learn_structure("granger", series, max_lag=2, max_indegree=2)


def test_the_progress_bar_setting_is_restored():
    before = pgmpy.config.get_show_progress()
    with C._quiet_pgmpy():
        assert pgmpy.config.get_show_progress() is False
    assert pgmpy.config.get_show_progress() is before


def test_torch_device_availability_reports_false_without_torch(monkeypatch):
    _block_import(monkeypatch, "torch")
    monkeypatch.delitem(importlib.sys.modules, "torch", raising=False)
    assert C._torch_device_available("cuda") is False


# ---------------------------------------------------------------------------
# Every scale at once
# ---------------------------------------------------------------------------


def test_multiscale_learning_covers_every_scale():
    from src.synthetic.families import make_generator
    from src.synthetic.scenarios import ScenarioGrid

    cfg = next(s for s in ScenarioGrid.sanity_check() if s.name == "sanity_L3_M2_r0_F2")
    ds = make_generator(cfg, seed=0).generate(seed=0)
    found = C.learn_multiscale_structure(
        "granger",
        ds.data,
        max_lag={k: cfg.max_lag[k - 1] for k in (1, 2, 3)},
        max_indegree={k: max(cfg.max_indegree[k - 1], 1) for k in (1, 2, 3)},
    )
    assert set(found) == {1, 2, 3}
    assert all(set(found[k]) == {0, 1, 2} for k in found)


def test_multiscale_learning_defaults_a_missing_scale_to_no_edges():
    from src.synthetic.families import make_generator
    from src.synthetic.scenarios import ScenarioGrid

    cfg = next(s for s in ScenarioGrid.sanity_check() if s.name == "sanity_L3_M2_r0_F2")
    ds = make_generator(cfg, seed=0).generate(seed=0)
    found = C.learn_multiscale_structure(
        "granger", ds.data, max_lag={}, max_indegree={}
    )
    assert all(
        not parents for per_scale in found.values() for parents in per_scale.values()
    )
