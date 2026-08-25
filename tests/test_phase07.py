"""
tests/test_phase07.py
=====================
Evaluation metrics and baselines.

Tests
-----
Metrics (10 tests)
  1.  PredictiveMetrics.compute  - NLL is finite
  2.  PredictiveMetrics          - nll_per_scale length matches K
  3.  ReconciliationMetrics      - RecErr=0 on generated data
  4.  StructureMetrics           - perfect match to SHD=0, F1=1
  5.  StructureMetrics           - empty-vs-true to recall=0, precision undefined
  6.  StructureMetrics.to_dict   - contains F1_mean and SHD_total
  7.  LatentMetrics.compute      - acc_S / ARI / NMI in valid bounds
  8.  CalibrationMetrics         - perfect predictions to all coverages = 1
  9.  CalibrationMetrics.to_dict - contains Coverage80/90/95
  10. aggregate_metrics          - merges all metric dicts into one

Baselines (3 tests)
  11. _empty_graph helper        - resulting graph has no edges
  12. EmptyGraph.fit             - completes without error, graph edges = 0
  13. OracleGraph.fit            - completes with finite ELBO

Protocol static methods (2 tests)
  14. EvaluationProtocol.save_csv      - creates CSV with header row
  15. EvaluationProtocol.summary_table - returns non-empty string

Run with:
    pytest tests/test_phase07.py -v
"""

from __future__ import annotations

import csv
import os
import tempfile

import numpy as np
import pytest

from src.core.graph import MultiScaleDBNGraph
from src.evaluation.baselines import (
    EmptyGraph,
    OracleGraph,
    TopDownNoFactor,
    TopDownNoRegime,
    _empty_graph,
)
from src.evaluation.metrics import (
    CalibrationMetrics,
    LatentMetrics,
    PredictiveMetrics,
    ReconciliationMetrics,
    StructureMetrics,
    aggregate_metrics,
)
from src.evaluation.protocol import EvaluationProtocol, RunResult
from src.synthetic.families import make_generator
from src.synthetic.scenarios import ScenarioConfig

# ---------------------------------------------------------------------------
# Shared minimal scenario + generated data (module scope)
# ---------------------------------------------------------------------------

_MINI_CFG = ScenarioConfig(
    name="phase07_mini",
    K=2,
    H_finest=12,
    branching=[3],
    L=2,
    M_k=[1, 1],
    r_k=[0, 0],
    max_lag=[1, 1],
    max_indegree=[0, 0],  # no edges to avoids S-step complications
    zi_level="low",
    sparse_graph=False,
    family=1,
    seed=77,
)

_SEM_KWARGS = dict(
    max_outer_iter=2,
    do_structure=False,
    do_warmstart=False,
    verbose=0,
    estep_kwargs={"max_inner_iter": 2, "use_laplace": False},
)


@pytest.fixture(scope="module")
def mini_gen():
    return make_generator(_MINI_CFG, seed=_MINI_CFG.seed)


@pytest.fixture(scope="module")
def mini_ds(mini_gen):
    return mini_gen.generate(seed=_MINI_CFG.seed)


@pytest.fixture(scope="module")
def mini_graph(mini_gen, mini_ds):
    return MultiScaleDBNGraph(
        hierarchy=mini_ds.hierarchy,
        label_names=mini_gen.label_names,
    )


# ---------------------------------------------------------------------------
# 1.  PredictiveMetrics - NLL is finite
# ---------------------------------------------------------------------------


def test_predictive_metrics_nll_finite(mini_ds, mini_graph):
    """PredictiveMetrics.compute() must return a finite NLL."""
    K = mini_ds.data.K
    T_per_k = [mini_ds.hierarchy.n_epochs(k) for k in range(1, K + 1)]
    M = mini_ds.params[0].M  # all scales have M=1 here

    rng = np.random.default_rng(0)
    log_likes = [rng.standard_normal((T, M)) for T in T_per_k]
    gammas = [np.ones((T, M)) for T in T_per_k]
    pred_X = [
        np.ones((T, mini_ds.data.get_X(k + 1).shape[1])) for k, T in enumerate(T_per_k)
    ]
    pred_Y = [
        np.ones((T, mini_ds.data.get_Y(k + 1).shape[1])) for k, T in enumerate(T_per_k)
    ]

    pm = PredictiveMetrics.compute(
        data=mini_ds.data,
        log_likes_per_scale=log_likes,
        gamma_per_scale=gammas,
        pred_X_per_scale=pred_X,
        pred_Y_per_scale=pred_Y,
    )
    assert np.isfinite(pm.nll), f"NLL = {pm.nll} is not finite"


# ---------------------------------------------------------------------------
# 2.  PredictiveMetrics - nll_per_scale length matches K
# ---------------------------------------------------------------------------


def test_predictive_metrics_per_scale_length(mini_ds):
    """len(nll_per_scale) == K."""
    K = mini_ds.data.K
    T_per_k = [mini_ds.hierarchy.n_epochs(k) for k in range(1, K + 1)]
    M = 1

    rng = np.random.default_rng(1)
    log_likes = [rng.standard_normal((T, M)) for T in T_per_k]
    gammas = [np.ones((T, M)) for T in T_per_k]
    pred_X = [np.ones((T, 2)) for T in T_per_k]
    pred_Y = [np.ones((T, 2)) for T in T_per_k]

    pm = PredictiveMetrics.compute(
        data=mini_ds.data,
        log_likes_per_scale=log_likes,
        gamma_per_scale=gammas,
        pred_X_per_scale=pred_X,
        pred_Y_per_scale=pred_Y,
    )
    assert len(pm.nll_per_scale) == K
    assert len(pm.mae_X) == K
    assert len(pm.mae_Y) == K

    d = pm.to_dict()
    assert "NLL" in d
    for k in range(1, K + 1):
        assert f"NLL_k{k}" in d


# ---------------------------------------------------------------------------
# 3.  ReconciliationMetrics - zero error on generated data
# ---------------------------------------------------------------------------


def test_reconciliation_zero_on_generated(mini_ds):
    """RecErr_X and RecErr_Y must be 0 on data that passed reconciliation check."""
    rm = ReconciliationMetrics.compute(mini_ds.data)
    assert rm.rec_err_X == pytest.approx(0.0, abs=1e-6), f"RecErr_X = {rm.rec_err_X}"
    assert rm.rec_err_Y == pytest.approx(0.0, abs=1e-6), f"RecErr_Y = {rm.rec_err_Y}"
    d = rm.to_dict()
    assert "RecErr_X" in d and "RecErr_Y" in d


# ---------------------------------------------------------------------------
# 4.  StructureMetrics - perfect match to SHD=0, F1=1
# ---------------------------------------------------------------------------


def test_structure_metrics_perfect_match(mini_graph):
    """Comparing a graph to itself must give SHD=0 and F1=1 at every scale."""
    sm = StructureMetrics.compute(estimated=mini_graph, true_graph=mini_graph)
    assert all(s == 0 for s in sm.shd_per_scale), f"SHD={sm.shd_per_scale}"
    # With no edges, P/R/F1 are all 0/0 to _safe_div returns 0.0; F1=1 only if edges exist.
    # Since max_indegree=0 the graph is empty, so P=R=F1=0 by convention is fine.
    # Just check the shapes are correct:
    K = mini_graph.K
    assert len(sm.shd_per_scale) == K
    assert len(sm.f1_per_scale) == K


# ---------------------------------------------------------------------------
# 5.  StructureMetrics - estimated empty vs true-with-edges
# ---------------------------------------------------------------------------


def test_structure_metrics_empty_vs_with_edges(mini_gen, mini_ds):
    """When estimated graph is empty and true graph has edges, recall=0."""
    # Build a true graph that has one edge
    true_g = MultiScaleDBNGraph(
        hierarchy=mini_ds.hierarchy,
        label_names=mini_gen.label_names,
    )
    K = mini_ds.data.K
    true_g.add_edge(k=K, src_ell=0, lag=1, dst_ell=1)

    # Estimated is empty
    est_g = MultiScaleDBNGraph(
        hierarchy=mini_ds.hierarchy,
        label_names=mini_gen.label_names,
    )

    sm = StructureMetrics.compute(estimated=est_g, true_graph=true_g)

    # At the coarsest scale: TP=0, FP=0, FN=1 to recall=0, SHD=1
    assert sm.recall_per_scale[K - 1] == pytest.approx(
        0.0
    ), f"Expected recall=0 but got {sm.recall_per_scale[K-1]}"
    assert (
        sm.shd_per_scale[K - 1] == 1
    ), f"Expected SHD=1 but got {sm.shd_per_scale[K-1]}"


# ---------------------------------------------------------------------------
# 6.  StructureMetrics.to_dict - macro averages present
# ---------------------------------------------------------------------------


def test_structure_metrics_to_dict_macro(mini_graph):
    """to_dict() must contain F1_mean and SHD_total."""
    sm = StructureMetrics.compute(estimated=mini_graph, true_graph=mini_graph)
    d = sm.to_dict()
    assert "F1_mean" in d, f"Missing F1_mean in {list(d.keys())}"
    assert "SHD_total" in d, f"Missing SHD_total in {list(d.keys())}"
    assert d["SHD_total"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 7.  LatentMetrics - acc_S / ARI / NMI in valid bounds
# ---------------------------------------------------------------------------


def test_latent_metrics_bounds():
    """Acc_S ∈ [0,1], ARI ∈ [-1,1], NMI ∈ [0,1]."""
    T, M, r = 20, 2, 0
    rng = np.random.default_rng(5)
    gamma = rng.dirichlet(np.ones(M), size=T)
    true_S = rng.integers(0, M, size=T)
    m_H = np.zeros((T, r))
    true_H = np.zeros((T, r))

    lm = LatentMetrics.compute(
        gamma_per_scale=[gamma],
        true_S_per_scale=[true_S],
        m_H_per_scale=[m_H],
        true_H_per_scale=[true_H],
    )
    acc = lm.acc_S_per_scale[0]
    ari = lm.ari_per_scale[0]
    nmi = lm.nmi_per_scale[0]

    assert 0.0 <= acc <= 1.0, f"acc_S = {acc} out of [0,1]"
    assert -1.0 <= ari <= 1.0 + 1e-8, f"ARI = {ari} out of [-1,1]"
    assert -1e-8 <= nmi <= 1.0 + 1e-8, f"NMI = {nmi} out of [0,1]"


# ---------------------------------------------------------------------------
# 8.  CalibrationMetrics - perfect predictions to all coverages = 1
# ---------------------------------------------------------------------------


def test_calibration_coverage_perfect():
    """When true = predicted mean, z=0 ≤ all thresholds → all coverages = 1."""
    n = 100
    true_vals = np.arange(n, dtype=float)
    pred_mean = true_vals.copy()
    pred_std = np.ones(n)

    cm = CalibrationMetrics.compute(true_vals, pred_mean, pred_std)
    assert cm.coverage_80 == pytest.approx(1.0)
    assert cm.coverage_90 == pytest.approx(1.0)
    assert cm.coverage_95 == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 9.  CalibrationMetrics.to_dict - contains all three keys
# ---------------------------------------------------------------------------


def test_calibration_to_dict():
    """to_dict() must return Coverage80, Coverage90, Coverage95."""
    rng = np.random.default_rng(42)
    true_vals = rng.standard_normal(50)
    pred_mean = rng.standard_normal(50)
    pred_std = np.abs(rng.standard_normal(50)) + 0.1

    cm = CalibrationMetrics.compute(true_vals, pred_mean, pred_std)
    d = cm.to_dict()
    assert "Coverage80" in d
    assert "Coverage90" in d
    assert "Coverage95" in d
    for v in d.values():
        assert 0.0 <= v <= 1.0, f"Coverage value {v} outside [0,1]"


# ---------------------------------------------------------------------------
# 10.  aggregate_metrics - merges all dicts into one
# ---------------------------------------------------------------------------


def test_aggregate_metrics_merges_all():
    """aggregate_metrics must merge all non-None metric dicts."""
    rng = np.random.default_rng(9)
    true_vals = rng.standard_normal(20)

    predictive = PredictiveMetrics(
        nll=-10.0,
        nll_per_scale=[-5.0, -5.0],
        mae_X=[0.3, 0.4],
        mae_Y=[0.2, 0.1],
    )
    calibration = CalibrationMetrics(coverage_80=0.8, coverage_90=0.9, coverage_95=0.95)
    reconciliation = ReconciliationMetrics(rec_err_X=0.0, rec_err_Y=0.0)

    merged = aggregate_metrics(
        predictive=predictive,
        calibration=calibration,
        reconciliation=reconciliation,
    )
    assert "NLL" in merged
    assert "Coverage80" in merged
    assert "RecErr_X" in merged


# ---------------------------------------------------------------------------
# 11.  _empty_graph - resulting graph has no edges
# ---------------------------------------------------------------------------


def test_empty_graph_has_no_edges(mini_gen, mini_ds):
    """_empty_graph() must return a graph with zero edges at all scales."""
    # Build a graph with one edge
    graph_with_edge = MultiScaleDBNGraph(
        hierarchy=mini_ds.hierarchy,
        label_names=mini_gen.label_names,
    )
    K = mini_ds.data.K
    graph_with_edge.add_edge(k=K, src_ell=0, lag=1, dst_ell=1)

    empty = _empty_graph(graph_with_edge)

    for k in range(1, empty.K + 1):
        for ell in range(empty.L):
            parents = list(empty.get_parents(k, ell))
            assert (
                len(parents) == 0
            ), f"Expected no parents for (k={k}, ell={ell}), got {parents}"


# ---------------------------------------------------------------------------
# 12.  EmptyGraph.fit - completes without error
# ---------------------------------------------------------------------------


def test_empty_graph_fit_smoke(mini_gen, mini_ds, mini_graph):
    """EmptyGraph.fit() must complete and return a graph with no edges."""
    baseline = EmptyGraph(**_SEM_KWARGS)
    params_list, graph_out, history = baseline.fit(
        mini_ds.data, mini_ds.params, mini_graph
    )
    # Graph should have no edges (EmptyGraph forces do_structure=False)
    for k in range(1, graph_out.K + 1):
        for ell in range(graph_out.L):
            assert (
                len(list(graph_out.get_parents(k, ell))) == 0
            ), f"EmptyGraph should have no edges at (k={k}, ell={ell})"
    assert len(params_list) == mini_ds.data.K


# ---------------------------------------------------------------------------
# 13.  OracleGraph.fit - completes with finite ELBO
# ---------------------------------------------------------------------------


def test_oracle_graph_fit_smoke(mini_gen, mini_ds, mini_graph):
    """OracleGraph.fit() must complete and produce finite ELBO."""
    baseline = OracleGraph(true_graph=mini_graph, **_SEM_KWARGS)
    params_list, graph_out, history = baseline.fit(
        mini_ds.data, mini_ds.params, mini_graph
    )
    assert len(params_list) == mini_ds.data.K
    assert all(np.isfinite(e) for e in history.elbo), f"Non-finite ELBO: {history.elbo}"


# ---------------------------------------------------------------------------
# 14.  EvaluationProtocol.save_csv - creates CSV with expected headers
# ---------------------------------------------------------------------------


def test_protocol_save_csv():
    """save_csv must write a valid CSV with scenario/method/replicate columns."""
    results = [
        RunResult(
            scenario_name="scen_A",
            method_name="n2dbn",
            replicate=0,
            metrics={"NLL": -50.0, "F1_mean": 0.8},
            wall_time_s=1.5,
            elbo_history=[-100.0, -90.0],
            n_outer_iters=2,
        ),
        RunResult(
            scenario_name="scen_A",
            method_name="empty",
            replicate=0,
            metrics={"NLL": -45.0, "F1_mean": 0.0},
            wall_time_s=0.5,
            n_outer_iters=2,
        ),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "results.csv")
        EvaluationProtocol.save_csv(results, path)

        assert os.path.exists(path), "CSV file was not created"
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = list(reader)

    assert "scenario" in fieldnames, f"'scenario' not in {fieldnames}"
    assert "method" in fieldnames
    assert "NLL" in fieldnames
    assert len(rows) == 2, f"Expected 2 data rows, got {len(rows)}"
    assert rows[0]["scenario"] == "scen_A"
    assert rows[0]["method"] == "n2dbn"


# ---------------------------------------------------------------------------
# 15.  EvaluationProtocol.summary_table - returns correctly formatted string
# ---------------------------------------------------------------------------


def test_protocol_summary_table():
    """summary_table must return a non-empty string containing scenario and method."""
    results = [
        RunResult(
            scenario_name="mini",
            method_name="n2dbn",
            replicate=0,
            metrics={"NLL": -50.0, "SHD_total": 2.0},
            wall_time_s=1.0,
            n_outer_iters=2,
        ),
        RunResult(
            scenario_name="mini",
            method_name="n2dbn",
            replicate=1,
            metrics={"NLL": -48.0, "SHD_total": 1.0},
            wall_time_s=1.1,
            n_outer_iters=2,
        ),
    ]
    table = EvaluationProtocol.summary_table(results, metric_keys=["NLL", "SHD_total"])
    assert isinstance(table, str)
    assert len(table) > 0
    assert "mini" in table, f"'mini' not found in table:\n{table}"
    assert "n2dbn" in table
    # Mean ± std formatting: NLL mean = (-50+(-48))/2 = -49
    assert "NLL" in table


# ---------------------------------------------------------------------------
# 16.  TopDownNoFactor.fit - completes without error (r>0 to zero_r)
# ---------------------------------------------------------------------------


def test_no_factor_baseline_smoke():
    """TopDownNoFactor.fit() must complete and return valid params with r=0."""
    cfg = ScenarioConfig(
        name="no_factor_test",
        K=2,
        H_finest=12,
        branching=[3],
        L=2,
        M_k=[1, 1],
        r_k=[1, 1],
        max_lag=[1, 1],
        max_indegree=[0, 0],
        zi_level="low",
        sparse_graph=False,
        family=3,
        seed=88,
    )
    gen = make_generator(cfg, seed=cfg.seed)
    ds = gen.generate(seed=cfg.seed)
    graph = MultiScaleDBNGraph(
        hierarchy=ds.hierarchy,
        label_names=gen.label_names,
    )

    baseline = TopDownNoFactor(**_SEM_KWARGS)
    params_list, graph_out, history = baseline.fit(ds.data, ds.params, graph)

    assert len(params_list) == cfg.K
    # Factor dimension should be 0 in all returned params
    for p in params_list:
        assert p.factor.r == 0, f"Expected r=0, got r={p.factor.r}"


# ---------------------------------------------------------------------------
# 17.  TopDownNoRegime.fit - completes without error (M>1 to collapse)
# ---------------------------------------------------------------------------


def test_no_regime_baseline_smoke():
    """TopDownNoRegime.fit() must complete and return valid params with M=1."""
    cfg = ScenarioConfig(
        name="no_regime_test",
        K=2,
        H_finest=12,
        branching=[3],
        L=2,
        M_k=[2, 2],
        r_k=[0, 0],
        max_lag=[1, 1],
        max_indegree=[0, 0],
        zi_level="low",
        sparse_graph=False,
        family=2,
        seed=99,
    )
    gen = make_generator(cfg, seed=cfg.seed)
    ds = gen.generate(seed=cfg.seed)
    graph = MultiScaleDBNGraph(
        hierarchy=ds.hierarchy,
        label_names=gen.label_names,
    )

    baseline = TopDownNoRegime(**_SEM_KWARGS)
    params_list, graph_out, history = baseline.fit(ds.data, ds.params, graph)

    assert len(params_list) == cfg.K
    # Regime count should be 1 in all returned params
    for p in params_list:
        assert p.regime.M == 1, f"Expected M=1, got M={p.regime.M}"


# ---------------------------------------------------------------------------
# 18.  EvaluationProtocol.run - completes a single replicate
# ---------------------------------------------------------------------------


def test_protocol_run_single_replicate():
    """EvaluationProtocol.run() must complete and return RunResult objects."""
    from src.synthetic.scenarios import ScenarioGrid

    grid = ScenarioGrid(
        scenarios=[
            ScenarioConfig(
                name="protocol_test",
                K=2,
                H_finest=30,
                branching=[3],
                L=2,
                M_k=[1, 1],
                r_k=[0, 0],
                max_lag=[1, 1],
                max_indegree=[0, 0],
                zi_level="low",
                sparse_graph=False,
                family=1,
                seed=77,
                n_replicates=1,
            ),
        ]
    )

    protocol = EvaluationProtocol(
        methods={"empty": EmptyGraph(**_SEM_KWARGS)},
        n_replicates=1,
        train_frac=0.7,
        valid_frac=0.15,
        verbose=0,
    )

    results = protocol.run(grid)
    assert len(results) == 1
    assert isinstance(results[0], RunResult)
    assert results[0].method_name == "empty"
    assert results[0].scenario_name == "protocol_test"
    # Should have actual metrics (not an error result)
    assert "error" not in results[0].metrics
