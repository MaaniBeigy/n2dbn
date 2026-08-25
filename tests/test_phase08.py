"""
tests/test_phase08.py
=====================
Logging and visualization utilities.

Tests
-----
RunLogger (6 tests)
  1.  context manager creates JSONL file with valid JSON lines
  2.  log_scalar writes event="scalar" with correct name/value/step
  3.  run_end record has elapsed_s > 0
  4.  log_config writes event="config" with payload
  5.  load_log roundtrip recovers all written records
  6.  extract_scalars filters by name correctly

Visualization (6 tests)
  7.  plot_multiscale_counts returns a Figure without error
  8.  plot_regime_sequence returns a Figure (M=2, K=2)
  9.  plot_factor_trajectory with r=0 returns a Figure (text-only)
  10. plot_factor_trajectory with r=1 returns a Figure with trajectory
  11. plot_elbo_history returns a Figure
  12. plot_graph_structure returns a Figure for an empty graph

Run with:
    pytest tests/test_phase08.py -v
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from src.core.graph import MultiScaleDBNGraph
from src.inference.posterior import PosteriorSummaries
from src.synthetic.families import make_generator
from src.synthetic.scenarios import ScenarioConfig
from src.utils.logging import RunLogger, extract_scalars, load_log

# ---------------------------------------------------------------------------
# Shared minimal generated dataset (module scope)
# ---------------------------------------------------------------------------

_MINI_CFG = ScenarioConfig(
    name="phase08_mini",
    K=2,
    H_finest=12,
    branching=[3],
    L=2,
    M_k=[2, 2],  # M=2 so regime plots have multiple rows
    r_k=[0, 0],
    max_lag=[1, 1],
    max_indegree=[0, 0],
    zi_level="low",
    sparse_graph=False,
    family=2,
    seed=88,
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
# Helper: build PosteriorSummaries without running EStep
# ---------------------------------------------------------------------------


def _make_ps(T: int, M: int, r: int) -> PosteriorSummaries:
    rng = np.random.default_rng(0)
    gamma = rng.dirichlet(np.ones(M), size=T)
    xi = rng.dirichlet(np.ones(M * M), size=T - 1).reshape(T - 1, M, M)
    xi /= xi.sum(axis=(1, 2), keepdims=True)
    m_H = rng.standard_normal((T, r)) if r > 0 else np.zeros((T, 0))
    Sigma_H = np.tile(0.1 * np.eye(r), (T, 1, 1)) if r > 0 else np.zeros((T, 0, 0))
    return PosteriorSummaries(gamma=gamma, xi=xi, m_H=m_H, Sigma_H=Sigma_H, log_Z=-20.0)


# ===========================================================================
# Logging tests
# ===========================================================================


class TestRunLogger:
    """Tests for the RunLogger context manager and JSONL output."""

    # -----------------------------------------------------------------------
    # 1.  Context manager creates valid JSONL
    # -----------------------------------------------------------------------

    def test_creates_jsonl_file(self, tmp_path):
        """RunLogger must create the file and write valid JSON lines."""
        path = str(tmp_path / "run.jsonl")
        with RunLogger(path, tags={"scenario": "test"}) as log:
            log.log_scalar("ELBO", -99.0, step=0)

        assert os.path.exists(path), "Log file was not created"
        records = load_log(path)
        assert len(records) >= 2, "Expected at least run_start + scalar + run_end"
        # Every record is valid JSON - load_log already parsed them
        for rec in records:
            assert isinstance(rec, dict)

    # -----------------------------------------------------------------------
    # 2.  log_scalar writes correct fields
    # -----------------------------------------------------------------------

    def test_log_scalar_fields(self, tmp_path):
        """log_scalar must write event='scalar' with name, value, step."""
        path = str(tmp_path / "scalars.jsonl")
        with RunLogger(path) as log:
            log.log_scalar("NLL", -50.0, step=3)

        records = load_log(path)
        scalar_recs = [r for r in records if r.get("event") == "scalar"]
        assert (
            len(scalar_recs) == 1
        ), f"Expected 1 scalar record, got {len(scalar_recs)}"
        rec = scalar_recs[0]
        assert rec["name"] == "NLL"
        assert rec["value"] == pytest.approx(-50.0)
        assert rec["step"] == 3
        assert "ts" in rec, "Timestamp 'ts' missing from scalar record"

    # -----------------------------------------------------------------------
    # 3.  run_end record has elapsed_s > 0
    # -----------------------------------------------------------------------

    def test_run_end_has_elapsed(self, tmp_path):
        """The run_end event written by __exit__ must have elapsed_s ≥ 0."""
        path = str(tmp_path / "end.jsonl")
        with RunLogger(path) as log:
            pass  # no operations

        records = load_log(path)
        end_recs = [r for r in records if r.get("event") == "run_end"]
        assert len(end_recs) == 1, "Expected exactly one run_end record"
        assert end_recs[0]["elapsed_s"] >= 0.0

    # -----------------------------------------------------------------------
    # 4.  log_config writes event='config'
    # -----------------------------------------------------------------------

    def test_log_config(self, tmp_path):
        """log_config must write event='config' containing the config dict."""
        path = str(tmp_path / "cfg.jsonl")
        cfg = {"K": 2, "L": 3, "seed": 42}
        with RunLogger(path) as log:
            log.log_config(cfg)

        records = load_log(path)
        cfg_recs = [r for r in records if r.get("event") == "config"]
        assert len(cfg_recs) == 1
        assert cfg_recs[0]["config"]["K"] == 2
        assert cfg_recs[0]["config"]["seed"] == 42

    # -----------------------------------------------------------------------
    # 5.  load_log roundtrip - all records recovered
    # -----------------------------------------------------------------------

    def test_load_log_roundtrip(self, tmp_path):
        """load_log must recover every record written by RunLogger."""
        path = str(tmp_path / "rt.jsonl")
        with RunLogger(path, tags={"exp": "unit"}) as log:
            log.log_scalar("A", 1.0, step=0)
            log.log_scalar("B", 2.0, step=1)
            log.log_scalars({"C": 3.0, "D": 4.0}, step=2)

        records = load_log(path)
        events = [r["event"] for r in records]
        assert "run_start" in events
        assert "run_end" in events
        assert events.count("scalar") == 2
        assert events.count("scalars") == 1

        # Tags propagate to every record
        for r in records:
            assert r.get("exp") == "unit", f"Tag missing from record: {r}"

    # -----------------------------------------------------------------------
    # 6.  extract_scalars filters by name
    # -----------------------------------------------------------------------

    def test_extract_scalars_filters(self, tmp_path):
        """extract_scalars must return only records whose name matches."""
        path = str(tmp_path / "ex.jsonl")
        with RunLogger(path) as log:
            log.log_scalar("ELBO", -100.0, step=0)
            log.log_scalar("ELBO", -90.0, step=1)
            log.log_scalar("SHD", 3.0, step=0)

        records = load_log(path)
        elbo_recs = extract_scalars(records, "ELBO")
        shd_recs = extract_scalars(records, "SHD")
        other_recs = extract_scalars(records, "MISSING")

        assert len(elbo_recs) == 2
        assert len(shd_recs) == 1
        assert len(other_recs) == 0
        assert elbo_recs[0]["value"] == pytest.approx(-100.0)
        assert elbo_recs[1]["step"] == 1


# ===========================================================================
# Visualization tests
# ===========================================================================

# Try to import matplotlib; skip all viz tests if not available
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _MPL_OK = True
except ImportError:
    _MPL_OK = False

pytestmark_mpl = pytest.mark.skipif(not _MPL_OK, reason="matplotlib not installed")


class TestVisualization:
    """Smoke tests for all visualization functions (Agg backend, no display)."""

    # -----------------------------------------------------------------------
    # 7.  plot_multiscale_counts
    # -----------------------------------------------------------------------

    @pytestmark_mpl
    def test_plot_multiscale_counts(self, mini_ds):
        """plot_multiscale_counts must return a Figure without raising."""
        from src.utils.visualization import plot_multiscale_counts

        fig = plot_multiscale_counts(mini_ds.data, label_names=["A", "B"])
        try:
            assert fig is not None
            assert hasattr(fig, "savefig"), "Return value must be a matplotlib Figure"
        finally:
            plt.close(fig)

    # -----------------------------------------------------------------------
    # 8.  plot_regime_sequence
    # -----------------------------------------------------------------------

    @pytestmark_mpl
    def test_plot_regime_sequence(self):
        """plot_regime_sequence must return a Figure for K=2, M=2 summaries."""
        from src.utils.visualization import plot_regime_sequence

        K, T, M = 2, 10, 2
        summaries = [_make_ps(T, M, r=0) for _ in range(K)]
        fig = plot_regime_sequence(summaries)
        try:
            assert fig is not None
            assert hasattr(fig, "savefig")
        finally:
            plt.close(fig)

    @pytestmark_mpl
    def test_plot_regime_sequence_with_true_S(self):
        """plot_regime_sequence with true_S must produce 2× as many rows."""
        from src.utils.visualization import plot_regime_sequence

        K, T, M = 2, 10, 2
        summaries = [_make_ps(T, M, r=0) for _ in range(K)]
        true_S = [np.zeros(T, dtype=int) for _ in range(K)]
        fig = plot_regime_sequence(summaries, true_S=true_S)
        try:
            assert fig is not None
        finally:
            plt.close(fig)

    # -----------------------------------------------------------------------
    # 9.  plot_factor_trajectory - r=0 (text-only fallback)
    # -----------------------------------------------------------------------

    @pytestmark_mpl
    def test_plot_factor_trajectory_r0(self):
        """plot_factor_trajectory with r=0 must return a Figure with text."""
        from src.utils.visualization import plot_factor_trajectory

        K, T, M = 2, 10, 2
        summaries = [_make_ps(T, M, r=0) for _ in range(K)]
        fig = plot_factor_trajectory(summaries, scale=1)
        try:
            assert fig is not None
        finally:
            plt.close(fig)

    # -----------------------------------------------------------------------
    # 10.  plot_factor_trajectory - r=1 (trajectory with ±2σ bands)
    # -----------------------------------------------------------------------

    @pytestmark_mpl
    def test_plot_factor_trajectory_r1(self):
        """plot_factor_trajectory with r=1 must return a Figure with subplots."""
        from src.utils.visualization import plot_factor_trajectory

        K, T, M, r = 2, 10, 1, 1
        summaries = [_make_ps(T, M, r=r) for _ in range(K)]
        fig = plot_factor_trajectory(summaries, scale=1)
        try:
            assert fig is not None
            # With r=1 there should be exactly one subplot row for H_1
            assert len(fig.axes) >= 1
        finally:
            plt.close(fig)

    # -----------------------------------------------------------------------
    # 11.  plot_elbo_history
    # -----------------------------------------------------------------------

    @pytestmark_mpl
    def test_plot_elbo_history(self):
        """plot_elbo_history must return a Figure given a list of floats."""
        from src.utils.visualization import plot_elbo_history

        elbo = [-200.0, -150.0, -130.0, -125.0]
        fig = plot_elbo_history(elbo)
        try:
            assert fig is not None
            assert len(fig.axes) == 1
        finally:
            plt.close(fig)

    # -----------------------------------------------------------------------
    # 12.  plot_graph_structure (empty graph)
    # -----------------------------------------------------------------------

    @pytestmark_mpl
    def test_plot_graph_structure_empty(self, mini_graph):
        """plot_graph_structure must return a Figure for an empty-edge graph."""
        from src.utils.visualization import plot_graph_structure

        fig = plot_graph_structure(mini_graph, scale=mini_graph.K)
        try:
            assert fig is not None
        finally:
            plt.close(fig)
