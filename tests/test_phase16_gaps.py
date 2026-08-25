"""
tests/test_phase16_gaps.py
==========================
The last uncovered lines: the calibrated edge penalty, the
optional-dependency error paths, the reference lag aggregators, and the
protocol helpers that only a parallel or partial run reaches.

The import guards themselves (`except ImportError` at module scope in
`core/graph.py`, `learning/pyagrum_scoring.py` and `utils/visualization.py`)
stay uncovered, since they run only when the package is absent.
"""

from __future__ import annotations

import csv
import os
from copy import deepcopy

import numpy as np
import pytest

from src.core.graph import MultiScaleDBNGraph
from src.evaluation.baselines import SingleScaleDBN
from src.evaluation.metrics import _normalised_mutual_info
from src.evaluation.protocol import EvaluationProtocol, RunResult, _replicate_worker
from src.inference.posterior import PosteriorSummaries, _sigmoid
from src.learning.estep import (
    EStep,
    _agg_lag_X,
    _agg_lag_Y,
    build_log_like_coarsest,
)
from src.learning.sstep import SStep, _LagColumns, _null_threshold
from src.learning.structural_em import TopDownSEM
from src.model.latent_factor import LatentFactor
from src.model.params import FactorParams
from src.synthetic.families import make_generator
from src.synthetic.scenarios import ScenarioConfig, ScenarioGrid


def _cfg(**over) -> ScenarioConfig:
    base = dict(
        name="gap",
        K=2,
        H_finest=120,
        branching=[10],
        L=3,
        M_k=[2, 2],
        r_k=[0, 0],
        max_lag=[2, 1],
        max_indegree=[0, 1],
        zi_level="medium",
        family=2,
        seed=5,
        n_replicates=1,
        edge_strength=10.0,
        regime_strength=3.0,
        finest_rate=0.2,
        finest_binary=True,
        slot_minutes=1,
    )
    base.update(over)
    return ScenarioConfig(**base)


@pytest.fixture(scope="module")
def dataset():
    cfg = _cfg()
    gen = make_generator(cfg, seed=5)
    ds = gen.generate(seed=5)
    graph = MultiScaleDBNGraph(hierarchy=ds.hierarchy, label_names=gen.label_names)
    for ki, p in enumerate(ds.params):
        for dst, parents in p.parent_sets.items():
            for src, lag in parents:
                graph.add_edge(ki + 1, src, lag, dst)
    return cfg, ds, graph


# ---------------------------------------------------------------------------
# Calibrated edge penalty
# ---------------------------------------------------------------------------


class _StubScorer:
    """Scorer whose gain depends only on how many parents a set holds."""

    def __init__(self, n_rows: int, gain: float = 1.0) -> None:
        arr = np.arange(n_rows, dtype=float).reshape(n_rows, 1)
        self.feat_x = _LagColumns(np.tile(arr, (1, 2)))
        self.feat_y = _LagColumns(np.tile(arr, (1, 2)))
        self.b_init: dict = {}
        self.a_init: dict = {}
        self.lam = 0.0
        self._gain = gain

    def score(self, parent_set) -> float:
        return self._gain * len(parent_set)


def test_null_threshold_is_zero_without_candidates():
    scorer = _StubScorer(n_rows=32)
    rng = np.random.default_rng(0)
    assert _null_threshold(scorer, [], rng, n_perm=4, quantile=0.95) == 0.0


def test_null_threshold_is_zero_on_a_series_too_short_to_shift():
    """A circular shift needs enough rows for the shifted copy to differ."""
    scorer = _StubScorer(n_rows=4)
    rng = np.random.default_rng(0)
    keys = [(0, 1), (1, 1)]
    assert _null_threshold(scorer, keys, rng, n_perm=4, quantile=0.95) == 0.0


def test_null_threshold_returns_the_quantile_of_the_shifted_gains():
    scorer = _StubScorer(n_rows=64, gain=2.5)
    rng = np.random.default_rng(0)
    keys = [(0, 1), (1, 1)]
    floor = _null_threshold(scorer, keys, rng, n_perm=8, quantile=0.95)
    assert floor == pytest.approx(2.5)


def test_lag_columns_accept_an_injected_column():
    cols = _LagColumns(np.arange(12.0).reshape(6, 2))
    injected = np.full((6, 1), 3.0)
    cols.set((-1, 0), injected)
    assert np.array_equal(cols[(-1, 0)], injected)


def test_calibrated_penalty_never_falls_below_the_bic_penalty(dataset):
    """The calibrated floor raises the penalty, so it cannot select more."""
    cfg, ds, graph = dataset
    K = cfg.K
    estep = EStep(max_inner_iter=2)
    q = estep.run(k=K, data=ds.data, params=ds.params[K - 1], graph=graph)

    bic = SStep(lam=1.0, max_lag=2, max_indegree=2, penalty="bic", max_iter=5)
    cal = SStep(
        lam=1.0, max_lag=2, max_indegree=2, penalty="calibrated", n_perm=4, max_iter=5
    )
    g_bic, _ = bic.run(K, q, deepcopy(ds.params[K - 1]), graph, ds.data)
    g_cal, _ = cal.run(K, q, deepcopy(ds.params[K - 1]), graph, ds.data)
    assert g_cal.n_edges(K) <= g_bic.n_edges(K)


# ---------------------------------------------------------------------------
# Beam search boundaries
# ---------------------------------------------------------------------------


def test_beam_search_stops_at_the_indegree_bound(dataset):
    """With max_indegree reached, no beam entry can be extended."""
    cfg, ds, graph = dataset
    K = cfg.K
    estep = EStep(max_inner_iter=2)
    q = estep.run(k=K, data=ds.data, params=ds.params[K - 1], graph=graph)
    step = SStep(
        lam=0.0, penalty="fixed", search="beam", beam_width=2, max_indegree=1, max_lag=2
    )
    new_graph, new_params = step.run(K, q, deepcopy(ds.params[K - 1]), graph, ds.data)
    assert all(len(p) <= 1 for p in new_params.parent_sets.values())


def test_beam_search_with_no_candidates_returns_the_empty_set(dataset):
    """max_lag below 1 leaves the candidate list empty, so the beam is empty."""
    cfg, ds, graph = dataset
    K = cfg.K
    estep = EStep(max_inner_iter=2)
    q = estep.run(k=K, data=ds.data, params=ds.params[K - 1], graph=graph)
    step = SStep(
        lam=0.0, penalty="fixed", search="beam", beam_width=2, max_indegree=2, max_lag=0
    )
    _, new_params = step.run(K, q, deepcopy(ds.params[K - 1]), graph, ds.data)
    assert all(len(p) == 0 for p in new_params.parent_sets.values())


# ---------------------------------------------------------------------------
# Reference implementations and small numerical helpers
# ---------------------------------------------------------------------------


def test_reference_lag_aggregators_match_the_vectorized_contribution(dataset):
    cfg, ds, _ = dataset
    K = cfg.K
    X = ds.data.get_X(K)
    Y = ds.data.get_Y(K)
    L = X.shape[1]
    parent_sets = {0: {(1, 1)}, 1: {(2, 2), (0, 1)}}
    betaLag = {
        (1, 1): np.full(L, 0.3),
        (2, 2): np.full(L, -0.2),
        (0, 1): np.full(L, 0.1),
    }
    for t in (0, 1, 5):
        got_x = _agg_lag_X(t, X, parent_sets, betaLag, L)
        got_y = _agg_lag_Y(t, Y, parent_sets, betaLag, L)
        want_x = np.zeros(L)
        want_y = np.zeros(L)
        for dst, parents in parent_sets.items():
            for src, d in parents:
                if t - d >= 0:
                    want_x[dst] += betaLag[(src, d)][dst] * np.log1p(X[t - d, src])
                    want_y[dst] += betaLag[(src, d)][dst] * np.log1p(Y[t - d, src])
        assert np.allclose(got_x, want_x)
        assert np.allclose(got_y, want_y)


def test_coarsest_log_likelihood_builds_its_own_lag_features(dataset):
    """Passing no precomputed features gives the same result as passing them."""
    cfg, ds, _ = dataset
    K = cfg.K
    params = ds.params[K - 1]
    X, Y = ds.data.get_X(K), ds.data.get_Y(K)
    ctx = ds.data.get_C(K)
    h_bar = np.zeros((X.shape[0], params.factor.r))
    from src.learning.estep import lag_contrib

    lag_x = lag_contrib(X, params.coarse_count.betaLag, params.parent_sets, params.L)
    lag_y = lag_contrib(Y, params.coarse_dur.alphaLag, params.parent_sets, params.L)
    with_feats = build_log_like_coarsest(X, Y, h_bar, ctx, params, lag_x, lag_y)
    without = build_log_like_coarsest(X, Y, h_bar, ctx, params)
    assert np.allclose(with_feats, without)


def test_sigmoid_is_stable_at_extreme_arguments():
    x = np.array([-1e6, -50.0, 0.0, 50.0, 1e6])
    p = _sigmoid(x)
    assert np.all(np.isfinite(p))
    assert p[0] == pytest.approx(0.0)
    assert p[2] == pytest.approx(0.5)
    assert p[-1] == pytest.approx(1.0)


def test_factor_posterior_moments_are_empty_without_a_factor():
    q = PosteriorSummaries(
        gamma=np.ones((6, 1)),
        xi=np.ones((5, 1, 1)),
        m_H=np.zeros((6, 0)),
        Sigma_H=np.zeros((6, 0, 0)),
        log_Z=0.0,
    )
    assert q.h_bar.shape == (6, 0)
    assert q.expected_hh().shape == (6, 0, 0)
    assert q.expected_h_prev_h(np.zeros((5, 0, 0))).shape == (5, 0, 0)


def test_factor_posterior_second_moment_adds_the_outer_product():
    m = np.array([[1.0, 2.0], [3.0, 4.0]])
    S = np.stack([np.eye(2), 2.0 * np.eye(2)])
    q = PosteriorSummaries(
        gamma=np.ones((2, 1)),
        xi=np.ones((1, 1, 1)),
        m_H=m,
        Sigma_H=S,
        log_Z=0.0,
    )
    hh = q.expected_hh()
    assert np.allclose(hh[0], np.eye(2) + np.outer(m[0], m[0]))
    cross = q.expected_h_prev_h(np.zeros((1, 2, 2)))
    assert np.allclose(cross[0], np.outer(m[1], m[0]))


def test_normalized_mutual_information_of_empty_partitions_is_zero():
    assert (
        _normalised_mutual_info(np.array([], dtype=int), np.array([], dtype=int)) == 0.0
    )


def test_factor_trajectory_starts_from_a_supplied_previous_state():
    """A warm start replaces the initial draw at the first epoch."""
    p = FactorParams(
        M=1,
        r=2,
        d_ctx=0,
        F=np.stack([np.eye(2)]),
        G=np.zeros((1, 2, 0)),
        Sigma=np.stack([1e-12 * np.eye(2)]),
        mu0=np.zeros(2),
        Sigma0=1e-12 * np.eye(2),
    )
    factor = LatentFactor(p)
    rng = np.random.default_rng(0)
    regimes = np.zeros(4, dtype=int)
    ctx = np.zeros((4, 0))
    warm = factor.sample_trajectory(4, regimes, ctx, rng, h_prev=np.full(2, 5.0))
    cold = factor.sample_trajectory(4, regimes, ctx, rng)
    assert warm[1] == pytest.approx(np.full(2, 5.0), abs=1e-4)
    assert cold[1] == pytest.approx(np.zeros(2), abs=1e-4)


# ---------------------------------------------------------------------------
# Baselines and the protocol
# ---------------------------------------------------------------------------


def test_single_scale_baseline_fits_without_cross_scale_coupling(dataset):
    cfg, ds, graph = dataset
    baseline = SingleScaleDBN(max_outer_iter=1, verbose=0)
    params, learned, history = baseline.fit(
        ds.data, [deepcopy(p) for p in ds.params], deepcopy(graph)
    )
    assert len(params) == cfg.K
    assert len(history.elbo) >= 1
    assert baseline._sem_kwargs["couple_scales"] is False


def test_replicate_worker_caps_verbosity_and_returns_results():
    cfg = _cfg(H_finest=60, branching=[10], name="worker")
    grid = ScenarioGrid(scenarios=[cfg])
    protocol = EvaluationProtocol(
        methods={"n2dbn": TopDownSEM(max_outer_iter=1, verbose=0)},
        n_replicates=1,
        verbose=3,
    )
    results = _replicate_worker(protocol, grid.scenarios[0], 0)
    assert protocol.verbose == 1
    assert len(results) == 1


def test_protocol_run_accepts_an_initializer_and_writes_partial_rows(tmp_path):
    cfg = _cfg(H_finest=60, branching=[10], name="init")
    grid = ScenarioGrid(scenarios=[cfg])
    calls = []

    def init_params_fn(scenario, train_data):
        calls.append(scenario.name)
        gen = make_generator(scenario, seed=scenario.seed)
        ds = gen.generate(seed=scenario.seed)
        graph = MultiScaleDBNGraph(
            hierarchy=train_data.hierarchy, label_names=gen.label_names
        )
        return [deepcopy(p) for p in ds.params], graph

    out = tmp_path / "partial.csv"
    protocol = EvaluationProtocol(
        methods={"n2dbn": TopDownSEM(max_outer_iter=1, verbose=0)},
        n_replicates=1,
        verbose=0,
    )
    results = protocol.run(grid, init_params_fn=init_params_fn, partial_csv=str(out))
    assert calls == ["init"]
    assert out.exists()
    with open(out, newline="") as fh:
        assert len(list(csv.DictReader(fh))) == len(results)


def test_save_csv_of_an_empty_result_list_writes_nothing(tmp_path):
    out = tmp_path / "none.csv"
    EvaluationProtocol.save_csv([], str(out))
    assert not os.path.exists(out)


def test_summary_table_picks_its_own_columns_and_marks_missing_cells():
    results = [
        RunResult(
            scenario_name="s",
            method_name="m",
            replicate=0,
            metrics={"F1_mean": 0.5},
            wall_time_s=1.0,
            n_outer_iters=1,
        ),
        RunResult(
            scenario_name="s",
            method_name="m",
            replicate=1,
            metrics={"SHD_total": 3.0},
            wall_time_s=1.0,
            n_outer_iters=1,
        ),
    ]
    table = EvaluationProtocol.summary_table(results)
    assert "F1_mean" in table and "SHD_total" in table
    assert "N/A" not in table

    partial = EvaluationProtocol.summary_table(results, metric_keys=["AccS_k1"])
    assert "N/A" in partial


# ---------------------------------------------------------------------------
# Optional-dependency guards, exercised by hiding the dependency
# ---------------------------------------------------------------------------


def test_graph_export_names_the_missing_dependency(dataset, monkeypatch):
    cfg, ds, graph = dataset
    import src.core.graph as graph_module

    monkeypatch.setattr(graph_module, "_PYAGRUM_AVAILABLE", False)
    with pytest.raises(ImportError, match="pyAgrum"):
        graph.to_pyagrum_bn(1, M=2)


def test_pyagrum_plots_name_the_missing_dependency(dataset, monkeypatch):
    cfg, ds, graph = dataset
    import src.utils.visualization as viz

    monkeypatch.setattr(viz, "_PYAGRUM_AVAILABLE", False)
    with pytest.raises(ImportError, match="pyAgrum"):
        viz.plot_graph_pyagrum(graph, scale=1, M=2)
    with pytest.raises(ImportError, match="pyAgrum"):
        viz.show_regime_cpt(graph, ds.params[0].regime, scale=1)


def test_pyagrum_scorer_names_the_missing_dependency(monkeypatch):
    import src.learning.pyagrum_scoring as scoring

    monkeypatch.setattr(scoring, "_PYAGRUM_AVAILABLE", False)
    with pytest.raises(ImportError, match="pyAgrum"):
        scoring.PyAgrumScore(np.zeros((10, 2)), ["a", "b"])


def test_graph_structure_plot_names_the_missing_dependency(dataset, monkeypatch):
    cfg, ds, graph = dataset
    import builtins

    import src.utils.visualization as viz

    real_import = builtins.__import__

    def no_networkx(name, *args, **kwargs):
        if name == "networkx":
            raise ImportError("networkx is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_networkx)
    with pytest.raises(ImportError, match="networkx"):
        viz.plot_graph_structure(graph, scale=1)


# ---------------------------------------------------------------------------
# Remaining branches
# ---------------------------------------------------------------------------


def test_discretization_of_a_constant_column_is_a_single_bin():
    scoring = pytest.importorskip("src.learning.pyagrum_scoring")
    binned = scoring._discretize_column(np.full(20, 3.0), n_bins=5)
    assert np.all(binned == 0)


def test_unknown_score_type_names_the_available_ones():
    pytest.importorskip("pyagrum")
    from src.learning.pyagrum_scoring import PyAgrumScore

    rng = np.random.default_rng(0)
    X = rng.integers(0, 4, size=(60, 2)).astype(float)
    with pytest.raises(ValueError, match="Unknown score_type"):
        PyAgrumScore(X, ["a", "b"], max_lag=1, score_type="nonsense")


def test_pyagrum_hill_climbing_returns_lagged_parents_only():
    pytest.importorskip("pyagrum")
    from src.learning.pyagrum_scoring import PyAgrumScore

    rng = np.random.default_rng(1)
    base = rng.integers(0, 4, size=80)
    X = np.stack([base, np.roll(base, 1)], axis=1).astype(float)
    scorer = PyAgrumScore(X, ["a", "b"], max_lag=1, n_bins=4)
    parent_sets = scorer.learn_initial_structure(max_indegree=2)
    assert set(parent_sets) == {0, 1}
    assert all(lag >= 1 for parents in parent_sets.values() for _, lag in parents)


def test_empty_graph_factory_has_no_edges(dataset):
    cfg, ds, graph = dataset
    empty = MultiScaleDBNGraph.empty(ds.hierarchy, graph.label_names)
    assert all(empty.n_edges(k) == 0 for k in range(1, cfg.K + 1))


def test_factor_update_keeps_its_parameters_when_the_solve_is_singular(monkeypatch):
    """A singular normal equation leaves the previous parameters in place."""
    from src.learning import mstep as mstep_module
    from src.learning.mstep import update_factor_params

    p = FactorParams(
        M=1,
        r=2,
        d_ctx=1,
        F=np.stack([np.eye(2)]),
        G=np.zeros((1, 2, 1)),
        Sigma=np.stack([np.eye(2)]),
        mu0=np.zeros(2),
        Sigma0=np.eye(2),
    )
    q = PosteriorSummaries(
        gamma=np.ones((6, 1)),
        xi=np.ones((5, 1, 1)),
        m_H=np.ones((6, 2)),
        Sigma_H=np.stack([np.eye(2)] * 6),
        log_Z=0.0,
    )

    def always_singular(*args, **kwargs):
        raise np.linalg.LinAlgError("singular")

    monkeypatch.setattr(mstep_module.np.linalg, "solve", always_singular)
    out = update_factor_params(q, np.ones((6, 1)), p)
    assert np.allclose(out.F[0], np.eye(2))


def test_generator_accepts_a_context_function():
    cfg = _cfg(name="ctxfn", H_finest=60, branching=[10])
    gen = make_generator(cfg, seed=3)
    seen = []

    def context_fn(k, T, rng):
        seen.append((k, T))
        d = gen.params_list[k - 1].factor.d_ctx
        return np.zeros((T, d))

    ds = gen.generate(seed=3, context_fn=context_fn)
    assert sorted(k for k, _ in seen) == list(range(1, cfg.K + 1))
    assert all(T == ds.hierarchy.n_epochs(k) for k, T in seen)
    assert ds.data.check_reconciliation()["RecErr_X"] == 0.0


def test_omitted_context_scenario_generates_a_context_effect():
    cfg = _cfg(name="omit", family=4, misspec_type="omit_context")
    gen = make_generator(cfg, seed=7)
    coarse = gen.params_list[cfg.K - 1]
    assert np.any(coarse.coarse_count.betaCtx != 0.0)


def test_family_four_dispatch_forwards_the_misspecification():
    """A grid that varies misspec_type has to reach the factory through
    make_generator, which is the only entry point the protocol uses."""

    def coarse(misspec_type):
        cfg = _cfg(name=f"m_{misspec_type}", family=4, misspec_type=misspec_type)
        return make_generator(cfg, seed=5).params_list[cfg.K - 1].coarse_count

    reference = coarse(None)
    tail = coarse("heavier_tail")
    context = coarse("omit_context")

    assert np.all(tail.r_disp == 50.0)
    assert not np.all(reference.r_disp == 50.0)
    assert np.abs(context.betaCtx).max() > np.abs(reference.betaCtx).max()
    assert np.allclose(tail.betaCtx, reference.betaCtx)


def test_beam_search_skips_a_parent_set_at_the_indegree_bound():
    """A depth budget above the in-degree bound leaves nothing to extend."""
    from src.learning.sstep import BeamSearch

    calls = []

    def score_fn(parent_set):
        calls.append(frozenset(parent_set))
        return float(len(parent_set))

    searcher = BeamSearch(beam_width=2, max_indegree=1, max_depth=3)
    best = searcher.search(score_fn, [(0, 1), (1, 1)])
    assert len(best) == 1
    assert max(len(c) for c in calls) == 1


def test_parallel_run_writes_partial_rows(tmp_path):
    cfg = _cfg(H_finest=60, branching=[10], name="par")
    grid = ScenarioGrid(scenarios=[cfg])
    out = tmp_path / "parallel.csv"
    protocol = EvaluationProtocol(
        methods={"n2dbn": TopDownSEM(max_outer_iter=1, verbose=0)},
        n_replicates=2,
        verbose=1,
        n_jobs=2,
    )
    results = protocol.run(grid, partial_csv=str(out))
    assert len(results) == 2
    with open(out, newline="") as fh:
        assert len(list(csv.DictReader(fh))) == 2


def test_parallel_and_serial_runs_agree():
    """Replicates are seeded from the scenario, so worker count cannot matter."""
    cfg = _cfg(H_finest=60, branching=[10], name="determinism", n_replicates=2)
    grid = ScenarioGrid(scenarios=[cfg])

    def run(n_jobs):
        protocol = EvaluationProtocol(
            methods={"n2dbn": TopDownSEM(max_outer_iter=2, verbose=0)},
            n_replicates=2,
            verbose=0,
            n_jobs=n_jobs,
        )
        rows = protocol.run(grid)
        return {(r.scenario_name, r.replicate): r.metrics["F1_mean"] for r in rows}

    serial, parallel = run(1), run(2)
    assert serial.keys() == parallel.keys()
    for key in serial:
        assert serial[key] == pytest.approx(parallel[key])
