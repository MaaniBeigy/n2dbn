"""
tests/test_phase23_temporal_viz.py
==================================
Timeline and snapshot figures of events and their lagged relations, and the
orchestrator that renders one stored experiment run.
"""

from __future__ import annotations

import builtins
import importlib
import json
import sys
from contextlib import contextmanager

import numpy as np
import pytest

from src.synthetic.families import make_generator
from src.synthetic.scenarios import ScenarioConfig, ScenarioGrid
from src.utils import temporal_viz
from src.utils.temporal_viz import (
    edge_occurrences,
    experiment_plots,
    plot_event_timeline,
    plot_snapshot_sequence,
    plot_teneto_slices,
    to_temporal_graph,
)


@contextmanager
def _without(*packages: str):
    """Reload the caller's modules with `packages` unimportable, then restore."""
    real_import = builtins.__import__
    saved = {
        name: sys.modules.pop(name)
        for name in list(sys.modules)
        if any(name == p or name.startswith(p + ".") for p in packages)
    }

    def blocked(name, *args, **kwargs):
        if any(name == p or name.startswith(p + ".") for p in packages):
            raise ImportError(f"{name} is unavailable")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = blocked
    try:
        yield
    finally:
        builtins.__import__ = real_import
        sys.modules.update(saved)


def _starts() -> np.ndarray:
    rng = np.random.default_rng(7)
    starts = (rng.random((24, 4)) < 0.4).astype(float)
    starts[3, 0] = 2.0  # a count above one, for the dot-size branch
    return starts


def _cfg() -> ScenarioConfig:
    return ScenarioConfig(
        name="viz",
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
        seed=4,
        n_replicates=1,
        edge_strength=10.0,
        regime_strength=3.0,
        finest_rate=0.2,
        finest_binary=True,
        slot_minutes=1,
    )


# ---------------------------------------------------------------------------
# Edge instances
# ---------------------------------------------------------------------------


def test_edge_occurrences_finds_the_aligned_pairs():
    starts = np.zeros((6, 2))
    starts[1, 0] = 1.0
    starts[2, 1] = 1.0
    starts[4, 0] = 1.0
    starts[5, 1] = 1.0
    occ = edge_occurrences(starts, [(0, 1, 1)])
    assert occ == [(0, 1, 1, 2), (0, 4, 1, 5)]


def test_edge_occurrences_respects_the_window_and_the_lag():
    starts = np.ones((6, 2))
    occ = edge_occurrences(starts, [(0, 3, 1)], t0=0, t1=4)
    assert occ == [(0, 0, 1, 3)]
    assert edge_occurrences(starts, [(0, 3, 1)], t0=5, t1=4) == []


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------


def test_timeline_draws_events_and_both_edge_families():
    fig = plot_event_timeline(
        _starts(),
        ["a", "b", "c", "d"],
        edges=[(0, 1, 1)],
        edges_true=[(2, 1, 3)],
        stamps=[f"{t:02d}" for t in range(24)],
    )
    ax = fig.axes[0]
    assert len(ax.get_legend().get_texts()) == 2
    assert [t.get_text() for t in ax.get_yticklabels()] == ["a", "b", "c", "d"]


def test_timeline_without_edges_has_no_legend():
    fig = plot_event_timeline(_starts())
    assert fig.axes[0].get_legend() is None


# ---------------------------------------------------------------------------
# networkx-temporal
# ---------------------------------------------------------------------------


def test_to_temporal_graph_places_edges_in_their_target_epoch():
    starts = np.zeros((4, 2))
    starts[0, 0] = 1.0
    starts[1, 1] = 1.0
    TG = to_temporal_graph(starts, [(0, 1, 1)], label_names=["u", "v"])
    assert TG[0].has_node("u")
    assert TG[1].has_edge("u", "v")
    assert not TG[2].nodes


def test_snapshot_sequence_has_one_panel_per_epoch():
    fig = plot_snapshot_sequence(_starts(), [(0, 1, 1)], t0=1, n_panels=3)
    assert len(fig.axes) == 3
    assert fig.axes[0].get_title() == "t = 1"


def test_snapshot_sequence_titles_from_stamps():
    fig = plot_snapshot_sequence(
        _starts(), [(0, 1, 1)], t0=0, n_panels=2, stamps=["01 00:00", "01 00:01"]
    )
    assert fig.axes[1].get_title() == "t = 01 00:01"


# ---------------------------------------------------------------------------
# teneto
# ---------------------------------------------------------------------------


def test_teneto_slices_draws_the_contact_sequence():
    fig = plot_teneto_slices(_starts(), 0, 12, ["a", "b", "c", "d"])
    assert fig.axes[0].get_title() == "Temporal network"


def test_teneto_slices_with_no_contacts():
    starts = np.zeros((6, 3))
    starts[2, 1] = 1.0  # one active label, so no pair ever co-fires
    fig = plot_teneto_slices(starts)
    assert fig is not None


# ---------------------------------------------------------------------------
# Stored-run orchestrator
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def stored_run(tmp_path_factory):
    cfg = _cfg()
    path = tmp_path_factory.mktemp("run") / "results.json"
    records = [
        {
            "scenario": cfg.name,
            "method": "n2dbn",
            "replicate": 0,
            "learned_edges": {"1": [[0, 1, 0]], "2": [[0, 1, 1]]},
        },
        {
            "scenario": cfg.name,
            "method": "empty_graph",
            "replicate": 0,
            "learned_edges": {"1": [], "2": []},
        },
    ]
    path.write_text(json.dumps(records))
    return cfg, str(path)


def test_experiment_plots_writes_every_figure(stored_run, tmp_path):
    cfg, json_path = stored_run
    grid = ScenarioGrid(scenarios=[cfg])
    paths = experiment_plots(json_path, str(tmp_path), grid=grid, window=20)
    names = [p.rsplit("/", 1)[1] for p in paths]
    assert names == [
        "viz_rep0_timeline_k1.png",
        "viz_rep0_timeline_k2.png",
        "viz_rep0_snapshots_k2.png",
        "viz_rep0_teneto_k2.png",
    ]
    import os

    assert all(os.path.exists(p) for p in paths)


def test_experiment_plots_without_teneto_skips_the_slice_plot(
    stored_run, tmp_path, monkeypatch
):
    cfg, json_path = stored_run

    def _no_teneto(*args, **kwargs):
        raise ImportError("teneto is unavailable")

    monkeypatch.setattr(temporal_viz, "plot_teneto_slices", _no_teneto)
    grid = ScenarioGrid(scenarios=[cfg])
    paths = experiment_plots(json_path, str(tmp_path), grid=grid, window=20)
    assert len(paths) == 3
    assert not any("teneto" in p for p in paths)


def test_experiment_plots_without_a_matching_record_raises(stored_run, tmp_path):
    cfg, json_path = stored_run
    with pytest.raises(ValueError, match="no record"):
        experiment_plots(
            json_path, str(tmp_path), grid=ScenarioGrid(scenarios=[cfg]), replicate=9
        )


def test_experiment_plots_with_an_unknown_scenario_raises(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(
        json.dumps(
            [
                {
                    "scenario": "nowhere",
                    "method": "n2dbn",
                    "replicate": 0,
                    "learned_edges": {"1": [[0, 1, 0]]},
                }
            ]
        )
    )
    with pytest.raises(ValueError, match="unknown scenario"):
        experiment_plots(str(path), str(tmp_path), grid=ScenarioGrid(scenarios=[]))


def test_experiment_plots_falls_back_to_the_named_grids(stored_run, tmp_path):
    _, json_path = stored_run
    with pytest.raises(ValueError, match="unknown scenario"):
        experiment_plots(json_path, str(tmp_path))


# ---------------------------------------------------------------------------
# Optional-dependency guards
# ---------------------------------------------------------------------------


def test_the_module_reports_its_optional_packages_as_absent():
    module = "src.utils.temporal_viz"
    try:
        with _without("matplotlib", "networkx_temporal"):
            reloaded = importlib.reload(importlib.import_module(module))
            assert reloaded._MPL_AVAILABLE is False
            assert reloaded._TX_AVAILABLE is False
            with pytest.raises(ImportError, match="matplotlib"):
                reloaded._require_mpl("plot_event_timeline")
            with pytest.raises(ImportError, match="networkx-temporal"):
                reloaded._require_tx("to_temporal_graph")
    finally:
        restored = importlib.reload(importlib.import_module(module))
        assert restored._MPL_AVAILABLE is True
        assert restored._TX_AVAILABLE is True


def test_the_teneto_guard_names_the_missing_package():
    with _without("teneto"):
        with pytest.raises(ImportError, match="teneto is required"):
            plot_teneto_slices(_starts())
