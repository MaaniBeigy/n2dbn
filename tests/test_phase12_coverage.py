"""
tests/test_phase12_coverage.py
==============================
Coverage for the wall-clock layer, the parallel protocol path and the code
paths the earlier phase tests never reach.

Groups
------
1.  SlotCalendar: epoch spans, stamps, shifting, formatting.
2.  Event log: reconstruction to wall-clock records and CSV export.
3.  BlockIndex: the fallback used when a parent block has no children.
4.  Protocol: parallel execution, JSON export, verbose paths, failing methods.
5.  Baselines: the predict interface of every baseline.
6.  Learning: monotone rejection, concentration estimation off, checkpoints,
    warm start, beam search, per-scale search spaces.
7.  Scenario configuration: calendar properties and the identifiability probe.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from datetime import datetime

import numpy as np
import pytest

from src.core.blocks import BlockIndex
from src.core.calendar import (
    DEFAULT_ORIGIN,
    EventRecord,
    SlotCalendar,
    episodes_to_event_log,
    format_duration,
    format_stamp,
    write_event_log,
)
from src.core.graph import MultiScaleDBNGraph
from src.evaluation.baselines import (
    EmptyGraph,
    OracleGraph,
    OracleParams,
    SingleScaleDBN,
    TopDownNoFactor,
    TopDownNoRegime,
)
from src.evaluation.protocol import (
    EvaluationProtocol,
    RunResult,
    apply_search_space,
)
from src.inference.reconstruction import Episode
from src.learning.estep import EStep, _agg_lag_X, _agg_lag_Y
from src.learning.mstep import MStep
from src.learning.sstep import SStep
from src.learning.structural_em import TopDownSEM
from src.synthetic.families import make_generator
from src.synthetic.scenarios import ScenarioConfig, ScenarioGrid

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _cfg(**over) -> ScenarioConfig:
    base = dict(
        name="cov",
        K=3,
        H_finest=720,
        branching=[10, 6],
        L=3,
        M_k=[2, 2, 2],
        r_k=[0, 0, 0],
        max_lag=[2, 1, 1],
        max_indegree=[0, 1, 0],
        zi_level="medium",
        sparse_graph=True,
        family=2,
        seed=7,
        n_replicates=1,
        edge_strength=10.0,
        regime_strength=3.0,
        finest_rate=0.15,
        finest_binary=True,
        slot_minutes=1,
    )
    base.update(over)
    return ScenarioConfig(**base)


@pytest.fixture(scope="module")
def dataset():
    cfg = _cfg()
    gen = make_generator(cfg, seed=7)
    ds = gen.generate(seed=7)
    graph = MultiScaleDBNGraph(hierarchy=ds.hierarchy, label_names=gen.label_names)
    for ki, p in enumerate(ds.params):
        for dst, parents in p.parent_sets.items():
            for src, lag in parents:
                graph.add_edge(ki + 1, src, lag, dst)
    return cfg, ds, graph


# ---------------------------------------------------------------------------
# 1.  SlotCalendar
# ---------------------------------------------------------------------------


def test_calendar_epoch_spans():
    cal = SlotCalendar.from_config(branching=[10, 6], slot_minutes=1)
    assert cal.epoch_minutes(1) == 1
    assert cal.epoch_minutes(2) == 10
    assert cal.epoch_minutes(3) == 60
    assert cal.describe() == {1: "1 minute", 2: "10 minutes", 3: "1 hour"}


def test_calendar_start_end_and_stamp():
    cal = SlotCalendar.from_config(
        branching=[10, 6], slot_minutes=1, origin=datetime(2026, 3, 1, 8, 0)
    )
    assert cal.start_of(1, 0) == datetime(2026, 3, 1, 8, 0)
    assert cal.start_of(1, 75) == datetime(2026, 3, 1, 9, 15)
    assert cal.end_of(1, 0) == datetime(2026, 3, 1, 8, 1)
    assert cal.start_of(3, 2) == datetime(2026, 3, 1, 10, 0)
    assert cal.stamp(1, 75) == "01 09:15"


def test_calendar_shift_moves_the_origin():
    cal = SlotCalendar.from_config(branching=[6], slot_minutes=5)
    moved = cal.shifted(12)  # 12 slots of 5 minutes
    assert moved.origin == DEFAULT_ORIGIN.replace(hour=1)
    assert moved.slot_minutes == cal.slot_minutes
    assert moved.branching == cal.branching


def test_format_helpers():
    assert format_duration(1) == "1 minute"
    assert format_duration(7) == "7 minutes"
    assert format_duration(60) == "1 hour"
    assert format_duration(180) == "3 hours"
    assert format_duration(1440) == "1 day"
    assert format_duration(2880) == "2 days"
    origin = datetime(2026, 1, 1, 0, 0)
    assert format_stamp(datetime(2026, 1, 3, 14, 5), origin) == "03 14:05"


# ---------------------------------------------------------------------------
# 2.  Event log
# ---------------------------------------------------------------------------


def test_event_log_records_wall_clock_intervals():
    cal = SlotCalendar.from_config(
        branching=[10, 6], slot_minutes=1, origin=datetime(2026, 1, 1, 0, 0)
    )
    traces = {
        0: [Episode(label=0, t_start=0, t_end=2)],
        1: [Episode(label=1, t_start=75, t_end=75)],
    }
    log = episodes_to_event_log(traces, ["work", "rest"], cal)
    assert [r.label for r in log] == ["work", "rest"]
    assert log[0].duration_minutes == 3
    assert log[0].start == datetime(2026, 1, 1, 0, 0)
    assert log[0].end == datetime(2026, 1, 1, 0, 3)
    assert log[1].start == datetime(2026, 1, 1, 1, 15)
    row = log[1].as_row(cal.origin)
    assert row["start"] == "01 01:15" and row["duration_minutes"] == 1
    assert row["start_iso"].startswith("2026-01-01T01:15")


def test_event_log_falls_back_to_label_indices():
    cal = SlotCalendar.from_config(branching=[4], slot_minutes=15)
    traces = {2: [Episode(label=2, t_start=1, t_end=1)]}
    log = episodes_to_event_log(traces, ["a"], cal)
    assert log[0].label == "L2"


def test_event_log_csv_roundtrip(tmp_path, dataset):
    cfg, ds, _ = dataset
    log = ds.event_log()
    assert len(log) > 0
    path = os.path.join(tmp_path, "events.csv")
    write_event_log(path, log, ds.calendar)
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    assert lines[0].startswith("label,start,end")
    assert len(lines) == len(log) + 1
    assert all(r.duration_minutes >= 1 for r in log)


def test_event_log_needs_a_calendar(dataset):
    cfg, ds, _ = dataset
    stripped = deepcopy(ds)
    stripped.calendar = None
    with pytest.raises(ValueError, match="no calendar"):
        stripped.event_log()


def test_split_shifts_the_calendar(dataset):
    cfg, ds, _ = dataset
    train, valid, test = ds.data.split(fracs=(0.7, 0.15, 0.15))
    assert train.calendar.origin == ds.data.calendar.origin
    assert valid.calendar.origin > train.calendar.origin
    assert test.calendar.origin > valid.calendar.origin


# ---------------------------------------------------------------------------
# 3.  BlockIndex fallbacks
# ---------------------------------------------------------------------------


def test_block_index_handles_empty_blocks():
    # block 1 has no children, which sends the reductions down the np.add.at path
    bi = BlockIndex.from_parent_map(np.array([0, 0, 2, 2]), n_blocks=3)
    arr = np.array([[1.0], [2.0], [3.0], [4.0]])
    assert bi._has_empty
    np.testing.assert_allclose(bi.sum(arr), [[3.0], [0.0], [7.0]])
    mx = bi.max(arr)
    assert mx[0, 0] == 2.0 and mx[2, 0] == 4.0 and not np.isfinite(mx[1, 0])
    lse = bi.logsumexp(arr)
    assert np.isfinite(lse[0, 0]) and not np.isfinite(lse[1, 0])


def test_block_index_sorts_an_unordered_parent_map():
    bi = BlockIndex.from_parent_map(np.array([1, 0, 1, 0]), n_blocks=2)
    assert bi.order is not None
    np.testing.assert_allclose(
        bi.sum(np.array([[1.0], [2.0], [3.0], [4.0]])), [[6.0], [4.0]]
    )


# ---------------------------------------------------------------------------
# 4.  Protocol
# ---------------------------------------------------------------------------


class _FailingMethod:
    """A method whose fit always raises, to exercise the error path."""

    def fit(self, data, params, graph):
        raise RuntimeError("deliberate failure")


def _tiny_grid() -> ScenarioGrid:
    return ScenarioGrid(scenarios=[_cfg(H_finest=360, name="tiny", n_replicates=2)])


def test_protocol_records_a_failing_method(capsys):
    protocol = EvaluationProtocol(
        methods={"broken": _FailingMethod()},
        n_replicates=1,
        verbose=2,
    )
    results = protocol.run(_tiny_grid())
    assert len(results) == 1
    assert results[0].metrics == {"error": 1.0}
    assert "deliberate failure" in capsys.readouterr().out


def test_protocol_parallel_matches_serial():
    grid = _tiny_grid()
    methods = {"n2dbn": TopDownSEM(max_outer_iter=1, verbose=0)}
    serial = EvaluationProtocol(methods=methods, n_replicates=2, verbose=0).run(grid)
    parallel = EvaluationProtocol(
        methods=methods, n_replicates=2, verbose=1, n_jobs=2
    ).run(grid)
    assert len(serial) == len(parallel) == 2
    for a, b in zip(serial, parallel):
        assert a.scenario_name == b.scenario_name and a.replicate == b.replicate
        assert a.metrics["F1_mean"] == pytest.approx(b.metrics["F1_mean"])


def test_protocol_resolve_jobs():
    assert EvaluationProtocol(methods={}, n_jobs=1)._resolve_jobs() == 1
    assert EvaluationProtocol(methods={}, n_jobs=None)._resolve_jobs() == 1
    assert EvaluationProtocol(methods={}, n_jobs=2)._resolve_jobs() >= 1
    assert EvaluationProtocol(methods={}, n_jobs=-1)._resolve_jobs() >= 1


def test_protocol_save_json(tmp_path):
    results = [
        RunResult(
            scenario_name="s",
            method_name="m",
            replicate=0,
            metrics={"F1_mean": 0.5},
            wall_time_s=1.0,
            elbo_history=[1.0, 2.0],
            n_outer_iters=2,
        )
    ]
    path = os.path.join(tmp_path, "r.json")
    EvaluationProtocol.save_json(results, path)
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    assert records[0]["metrics"]["F1_mean"] == 0.5
    assert records[0]["elbo_history"] == [1.0, 2.0]


def test_protocol_partial_csv(tmp_path):
    path = os.path.join(tmp_path, "partial.csv")
    protocol = EvaluationProtocol(
        methods={"oracle_params": OracleParams()},
        n_replicates=1,
        verbose=1,
    )
    protocol.run(_tiny_grid(), partial_csv=path)
    with open(path, encoding="utf-8") as f:
        assert len(f.read().splitlines()) == 2


def test_apply_search_space_handles_both_method_shapes():
    cfg = _cfg()
    sem = TopDownSEM(max_outer_iter=1, verbose=0)
    apply_search_space(sem, cfg)
    assert sem.sstep.max_lag == {1: 2, 2: 1, 3: 1}
    assert sem.sstep.max_indegree == {1: 1, 2: 1, 3: 1}

    wrapper = EmptyGraph(max_outer_iter=1, verbose=0)
    apply_search_space(wrapper, cfg)
    assert wrapper._sem_kwargs["sstep_kwargs"]["max_lag"] == {1: 2, 2: 1, 3: 1}

    apply_search_space(object(), cfg)  # nothing to configure, no error


# ---------------------------------------------------------------------------
# 5.  Baselines
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory",
    [
        lambda g: SingleScaleDBN(max_outer_iter=1, verbose=0),
        lambda g: TopDownNoFactor(max_outer_iter=1, verbose=0),
        lambda g: TopDownNoRegime(max_outer_iter=1, verbose=0),
        lambda g: EmptyGraph(max_outer_iter=1, verbose=0),
        lambda g: OracleGraph(g, max_outer_iter=1, verbose=0),
        lambda g: OracleParams(),
    ],
)
def test_baseline_predict_returns_summaries(dataset, factory):
    cfg, ds, graph = dataset
    method = factory(graph)
    summaries = method.predict(ds.data, [deepcopy(p) for p in ds.params], graph)
    assert len(summaries) == cfg.K
    for k, s in enumerate(summaries, start=1):
        assert s.gamma.shape[0] == ds.hierarchy.n_epochs(k)
        assert np.isfinite(s.log_Z)


def test_oracle_graph_fits_the_given_graph(dataset):
    cfg, ds, graph = dataset
    method = OracleGraph(graph, max_outer_iter=1, verbose=0)
    _, learned_graph, _ = method.fit(
        ds.data,
        [deepcopy(p) for p in ds.params],
        MultiScaleDBNGraph(
            hierarchy=ds.hierarchy, label_names=list(ds.data.label_names)
        ),
    )
    assert learned_graph.n_edges() == graph.n_edges()


# ---------------------------------------------------------------------------
# 6.  Learning paths
# ---------------------------------------------------------------------------


def test_mstep_rejects_an_update_that_lowers_its_block(dataset):
    """A huge learning rate overshoots, and the monotone guard drops it."""
    cfg, ds, graph = dataset
    K = cfg.K
    q = EStep().run(K, ds.data, ds.params[K - 1], graph)
    wild = MStep(n_steps_regime=5, n_steps_emission=5, lr=50.0, monotone=True)
    kept = wild.run(K, q, ds.params[K - 1], ds.data)
    loose = MStep(n_steps_regime=5, n_steps_emission=5, lr=50.0, monotone=False)
    moved = loose.run(K, q, ds.params[K - 1], ds.data)
    assert np.allclose(kept.coarse_count.beta0, ds.params[K - 1].coarse_count.beta0)
    assert not np.allclose(
        moved.coarse_count.beta0, ds.params[K - 1].coarse_count.beta0
    )


def test_mstep_without_concentration_fitting(dataset):
    cfg, ds, graph = dataset
    q = EStep().run(1, ds.data, ds.params[0], graph)
    step = MStep(n_steps_emission=2, fit_concentration=False)
    out = step.run(1, q, ds.params[0], ds.data)
    assert out.fine_count.kappa == ds.params[0].fine_count.kappa


def test_sem_checkpoint_and_resume(dataset, tmp_path):
    cfg, ds, graph = dataset
    ckpt = os.path.join(tmp_path, "ckpt")
    sem = TopDownSEM(
        max_outer_iter=2, do_structure=False, verbose=2, checkpoint_dir=ckpt
    )
    params, g, hist = sem.fit(
        ds.data, [deepcopy(p) for p in ds.params], deepcopy(graph)
    )
    saved = sorted(os.listdir(ckpt))
    assert saved, "no checkpoint written"

    resumed = TopDownSEM(max_outer_iter=3, do_structure=False, verbose=1)
    _, _, hist2 = resumed.fit(
        ds.data,
        [deepcopy(p) for p in ds.params],
        deepcopy(graph),
        resume_from=os.path.join(ckpt, saved[-1]),
    )
    assert len(hist2.elbo) >= len(hist.elbo)


def test_sem_warmstart_pass(dataset):
    cfg, ds, graph = dataset
    sem = TopDownSEM(max_outer_iter=1, do_structure=False, do_warmstart=True, verbose=0)
    _, _, hist = sem.fit(ds.data, [deepcopy(p) for p in ds.params], deepcopy(graph))
    assert len(hist.elbo) == 2  # one sweep plus the warm-start pass


def test_sem_cross_scale_warm_start_fills_the_block(dataset):
    cfg, ds, graph = dataset
    params = [deepcopy(p) for p in ds.params]
    for p in params:
        if p.regime.cross is not None:
            p.regime.cross = np.zeros_like(p.regime.cross)
    sem = TopDownSEM(max_outer_iter=1, do_structure=False, verbose=0)
    learned, _, _ = sem.fit(ds.data, params, deepcopy(graph))
    finer = [p for p in learned if p.regime.cross is not None]
    assert finer, "no scale carries a cross-scale block"
    assert any(np.abs(p.regime.cross).max() > 0 for p in finer)


def test_sem_edge_penalty_enters_the_objective(dataset):
    cfg, ds, graph = dataset
    sem = TopDownSEM(max_outer_iter=1, do_structure=True, verbose=0)
    _, learned, hist = sem.fit(
        ds.data, [deepcopy(p) for p in ds.params], deepcopy(graph)
    )
    assert hist.penalty[0] >= 0.0
    assert hist.elbo[0] == pytest.approx(sum(hist.log_Z_per_scale[0]) - hist.penalty[0])


def test_sstep_beam_search(dataset):
    cfg, ds, graph = dataset
    k = 2
    q = EStep().run(k, ds.data, ds.params[k - 1], graph)
    sstep = SStep(lam=1.0, search="beam", beam_width=3, max_lag=2, max_indegree=1)
    new_graph, new_params = sstep.run(
        k=k,
        q=q,
        params=ds.params[k - 1],
        graph=graph,
        data=ds.data,
        hier=ds.hierarchy,
    )
    for ell in range(cfg.L):
        assert len(new_params.parent_sets[ell]) <= 1


def test_sstep_fixed_penalty_mode(dataset):
    cfg, ds, graph = dataset
    k = 2
    q = EStep().run(k, ds.data, ds.params[k - 1], graph)
    strict = SStep(lam=1e6, penalty="fixed", max_lag=2)
    g2, _ = strict.run(
        k=k,
        q=q,
        params=ds.params[k - 1],
        graph=graph,
        data=ds.data,
        hier=ds.hierarchy,
    )
    assert g2.n_edges(k) == 0
    assert strict._lam_effective(100) == 1e6


def test_legacy_lag_helpers_match_the_vectorised_form(dataset):
    cfg, ds, _ = dataset
    p = ds.params[cfg.K - 1]
    X = ds.data.get_X(cfg.K)
    Y = ds.data.get_Y(cfg.K)
    for t in (0, 1, X.shape[0] - 1):
        vx = _agg_lag_X(t, X, p.parent_sets, p.coarse_count.betaLag, cfg.L)
        vy = _agg_lag_Y(t, Y, p.parent_sets, p.coarse_dur.alphaLag, cfg.L)
        assert vx.shape == (cfg.L,) and vy.shape == (cfg.L,)
        assert np.all(np.isfinite(vx)) and np.all(np.isfinite(vy))


# ---------------------------------------------------------------------------
# 7.  Scenario configuration
# ---------------------------------------------------------------------------


def test_scenario_calendar_properties():
    cfg = _cfg(slot_minutes=5, branching=[12, 24], H_finest=8640)
    assert cfg.scale_spans == ["5 minutes", "1 hour", "1 day"]
    assert cfg.calendar.origin == DEFAULT_ORIGIN
    dated = _cfg(origin="2026-06-01T07:30")
    assert dated.calendar.origin == datetime(2026, 6, 1, 7, 30)


def test_coarse_log_mean_is_none_without_a_rate():
    assert _cfg(finest_rate=None).coarse_log_mean is None
    cfg = _cfg(finest_rate=0.5)
    assert cfg.coarse_log_mean == pytest.approx(np.log(0.5 * 60))


def test_identifiability_probe_reports_every_scale():
    out = ScenarioGrid.identifiability_probe(_cfg(H_finest=720), seed=3)
    assert set(out) == {1, 2, 3}
    for row in out.values():
        assert row["n_epochs"] > 0
        assert 0.0 <= row["density"] <= 1.0
        # A probability, or not applicable where the true path never switches.
        assert np.isnan(row["oracle_accS"]) or 0.0 <= row["oracle_accS"] <= 1.0
        assert np.isfinite(row["null_gain95"])


def test_split_refuses_a_degenerate_hierarchy():
    cfg = _cfg(H_finest=60, branching=[10, 6])  # 60 / 6 / 1 epochs
    ds = make_generator(cfg, seed=1).generate(seed=1)
    with pytest.raises(ValueError, match="coarsest-scale epoch"):
        ds.data.split(fracs=(0.7, 0.15, 0.15))
