"""
tests/test_phase13_utils.py
===========================
Coverage for the reporting layer and the remaining defensive branches.

The experiment-tracking hooks of RunLogger are exercised against stub modules
rather than a real MLflow or Weights and Biases install, so the calls the logger
makes are checked without adding an optional dependency to the test run.
"""

from __future__ import annotations

import json
import os
import sys
import types
from copy import deepcopy

import numpy as np
import pytest

from src.core.blocks import BlockIndex
from src.core.calendar import SlotCalendar
from src.core.graph import MultiScaleDBNGraph
from src.core.scales import ScaleHierarchy
from src.core.variables import MultiScaleData
from src.inference.posterior import RegimePosterior, logsumexp
from src.learning.estep import EStep, block_softmax, dirmult_ll, fit_concentration
from src.synthetic.families import make_generator
from src.synthetic.scenarios import ScenarioConfig
from src.utils import logging as run_logging
from src.utils import visualization as viz

# ---------------------------------------------------------------------------
# Stub trackers
# ---------------------------------------------------------------------------


class _StubTracker(types.ModuleType):
    """Records the calls a tracking backend would receive."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.metrics = []
        self.metric_dicts = []
        self.params = []
        self.logs = []
        self.configs = []
        self.config = types.SimpleNamespace(update=self.configs.append)

    # mlflow surface
    def log_metric(self, name, value, step=0):
        self.metrics.append((name, value, step))

    def log_metrics(self, values, step=0):
        self.metric_dicts.append((dict(values), step))

    def log_params(self, params):
        self.params.append(dict(params))

    # wandb surface
    def log(self, values, step=None):
        self.logs.append((dict(values), step))


@pytest.fixture
def stub_trackers(monkeypatch):
    mlflow = _StubTracker("mlflow")
    wandb = _StubTracker("wandb")
    monkeypatch.setitem(sys.modules, "mlflow", mlflow)
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    return mlflow, wandb


# ---------------------------------------------------------------------------
# 1.  RunLogger with tracking backends
# ---------------------------------------------------------------------------


def test_run_logger_forwards_to_trackers(tmp_path, stub_trackers):
    mlflow, wandb = stub_trackers
    path = os.path.join(tmp_path, "run.jsonl")
    with run_logging.RunLogger(
        path, tags={"grid": "sanity"}, use_mlflow=True, use_wandb=True
    ) as log:
        log.log_config({"lam_edge": 1.0, "nested": {"a": 1}})
        log.log_scalar("F1_mean", 0.75, step=2)
        log.log_scalars({"AccS_k1": 0.5, "SHD_total": 4.0}, step=3)
        log.log_artifact(path, name="run")
        log.log_text("a note")

    assert ("F1_mean", 0.75, 2) in mlflow.metrics
    assert mlflow.metric_dicts[0][1] == 3
    assert mlflow.params[0]["lam_edge"] == "1.0"
    assert any(values.get("F1_mean") == 0.75 for values, _ in wandb.logs)
    assert wandb.configs and "lam_edge" in wandb.configs[0]

    records = run_logging.load_log(path)
    events = [r["event"] for r in records]
    assert events[0] == "run_start" and events[-1] == "run_end"
    assert {"config", "scalar", "scalars", "artifact", "text"} <= set(events)


def test_run_logger_without_optional_backends(tmp_path, monkeypatch):
    """A missing tracker is not an error, the JSONL file is still written."""
    monkeypatch.setitem(sys.modules, "mlflow", None)
    real_import = __import__

    def _no_trackers(name, *args, **kwargs):
        if name in ("mlflow", "wandb"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _no_trackers)
    path = os.path.join(tmp_path, "run.jsonl")
    with run_logging.RunLogger(path, use_mlflow=True, use_wandb=True) as log:
        log.log_scalar("x", 1.0)
    assert len(run_logging.load_log(path)) == 3


def test_extract_scalars_selects_one_series(tmp_path):
    path = os.path.join(tmp_path, "run.jsonl")
    with run_logging.RunLogger(path) as log:
        for step, value in enumerate([0.1, 0.2, 0.3]):
            log.log_scalar("elbo", value, step=step)
        log.log_scalar("other", 9.0, step=0)
    records = run_logging.extract_scalars(run_logging.load_log(path), "elbo")
    assert [r["step"] for r in records] == [0, 1, 2]
    assert [r["value"] for r in records] == [0.1, 0.2, 0.3]


# ---------------------------------------------------------------------------
# 2.  Visualization branches
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_dataset():
    cfg = ScenarioConfig(
        name="viz",
        K=2,
        H_finest=120,
        branching=[10],
        L=3,
        M_k=[2, 2],
        r_k=[1, 1],
        max_lag=[1, 1],
        max_indegree=[0, 1],
        zi_level="medium",
        family=3,
        seed=5,
        n_replicates=1,
        finest_rate=0.2,
        finest_binary=True,
        slot_minutes=1,
    )
    gen = make_generator(cfg, seed=5)
    ds = gen.generate(seed=5)
    graph = MultiScaleDBNGraph(hierarchy=ds.hierarchy, label_names=gen.label_names)
    for ki, p in enumerate(ds.params):
        for dst, parents in p.parent_sets.items():
            for src, lag in parents:
                graph.add_edge(ki + 1, src, lag, dst)
    return cfg, ds, graph


def test_plot_graph_structure_with_edges(small_dataset):
    cfg, ds, graph = small_dataset
    fig = viz.plot_graph_structure(graph, scale=2)
    assert fig is not None


def test_plot_graph_structure_custom_labels(small_dataset):
    cfg, ds, graph = small_dataset
    fig = viz.plot_graph_structure(graph, scale=2, label_names=["a", "b", "c"])
    assert fig is not None


def test_require_mpl_raises_when_absent(monkeypatch):
    monkeypatch.setattr(viz, "_MPL_AVAILABLE", False)
    with pytest.raises(ImportError, match="matplotlib"):
        viz._require_mpl("plot_x")


def test_plot_multiscale_counts_with_names(small_dataset):
    cfg, ds, _ = small_dataset
    fig = viz.plot_multiscale_counts(ds.data, label_names=["a", "b", "c"])
    assert fig is not None


# ---------------------------------------------------------------------------
# 3.  Remaining defensive branches
# ---------------------------------------------------------------------------


def test_logsumexp_keepdims_and_full_reduction():
    a = np.array([[0.0, 1.0], [2.0, 3.0]])
    assert logsumexp(a, axis=1, keepdims=True).shape == (2, 1)
    assert logsumexp(a) == pytest.approx(float(np.log(np.exp(a).sum())))
    assert np.isneginf(logsumexp(np.array([-np.inf, -np.inf])))


def test_forward_backward_single_epoch():
    gamma, xi, log_Z = RegimePosterior.forward_backward(
        np.array([[0.1, 0.2]]), np.zeros(2), np.empty((0, 2, 2))
    )
    assert gamma.shape == (1, 2)
    assert xi.shape == (1, 2, 2)
    assert np.isfinite(log_Z)


def test_build_log_trans_single_epoch(small_dataset):
    cfg, ds, _ = small_dataset
    params = ds.params[cfg.K - 1].regime
    ctx = ds.data.get_C(cfg.K)[:1]
    log_init, log_trans = RegimePosterior.build_log_trans(params, ctx)
    assert log_init.shape == (params.M,)
    assert log_trans.shape == (0, params.M, params.M)


def test_regime_offsets_rejects_a_mismatched_parent_posterior(small_dataset):
    from src.inference.posterior import regime_offsets

    cfg, ds, _ = small_dataset
    params = ds.params[0].regime
    if params.cross is None:
        pytest.skip("no cross-scale block at this scale")
    ctx = ds.data.get_C(1)
    wrong = np.ones((ctx.shape[0], params.cross.shape[1] + 1))
    with pytest.raises(ValueError, match="regimes"):
        regime_offsets(params, ctx, wrong)


def test_scale_hierarchy_repr_and_validate():
    hier = ScaleHierarchy.from_uniform_branching(120, [10])
    hier.validate()
    text = repr(hier)
    assert "K=2" in text and "finest" in text and "coarsest" in text


def test_check_reconciliation_reports_a_violation(small_dataset):
    cfg, ds, _ = small_dataset
    broken = deepcopy(ds.data)
    broken.X[0][0, 0] += 5  # break the child sum of the first block
    errs = broken.check_reconciliation()
    assert errs["RecErr_X"] == 5.0


def test_multiscale_data_accessors(small_dataset):
    cfg, ds, _ = small_dataset
    assert ds.data.K == cfg.K
    assert ds.data.L == cfg.L
    assert ds.data.get_C(1).shape[0] == ds.hierarchy.n_epochs(1)
    assert repr(ds.data)


def test_dirmult_reduces_to_zero_without_counts():
    bi = BlockIndex.from_parent_map(np.repeat(np.arange(4), 3), 4)
    omega = block_softmax(np.zeros((12, 1)), bi)
    assert float(dirmult_ll(np.zeros((12, 1)), omega, 8.0).sum()) == pytest.approx(0.0)


def test_fit_concentration_is_bounded():
    bi = BlockIndex.from_parent_map(np.repeat(np.arange(20), 4), 20)
    x = np.tile(np.array([[3.0], [0.0], [0.0], [0.0]]), (20, 1))
    omega = block_softmax(np.zeros((80, 1)), bi)
    conc = fit_concentration(x, omega, bi, lo=0.5, hi=100.0)
    assert 0.5 <= conc <= 100.0


def test_estep_uses_the_prior_smoother_at_finer_scales(small_dataset):
    """With r>0 below the coarsest scale the factor follows the prior only."""
    cfg, ds, graph = small_dataset
    q = EStep(use_laplace=True).run(1, ds.data, ds.params[0], graph)
    assert q.m_H.shape == (ds.hierarchy.n_epochs(1), ds.params[0].r)
    assert np.all(np.isfinite(q.m_H))
    assert np.all(np.isfinite(q.Sigma_H))
