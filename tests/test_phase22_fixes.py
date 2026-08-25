"""
tests/test_phase22_fixes.py
===========================
The monotone outer guard of the SEM loop and the per-edge shift confirmation
of the structure step.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from src.core.graph import MultiScaleDBNGraph
from src.learning.estep import EStep
from src.learning.sstep import (
    SStep,
    _block_roll,
    _confirm_edges,
    _LagColumns,
    _shift_gains,
)
from src.learning.structural_em import TopDownSEM
from src.synthetic.families import make_generator
from src.synthetic.scenarios import ScenarioConfig


def _cfg(**over) -> ScenarioConfig:
    base = dict(
        name="fx",
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
    base.update(over)
    return ScenarioConfig(**base)


@pytest.fixture(scope="module")
def dataset():
    cfg = _cfg()
    gen = make_generator(cfg, seed=4)
    ds = gen.generate(seed=4)
    graph = MultiScaleDBNGraph(hierarchy=ds.hierarchy, label_names=gen.label_names)
    for ki, p in enumerate(ds.params):
        for dst, parents in p.parent_sets.items():
            for src, lag in parents:
                graph.add_edge(ki + 1, src, lag, dst)
    return cfg, ds, graph


# ---------------------------------------------------------------------------
# Fakes for the outer guard, so a sweep's objective is chosen directly
# ---------------------------------------------------------------------------


class _FakeSummary:
    def __init__(self, log_Z: float) -> None:
        self.log_Z = log_Z


class _FakeEStep:
    """E-step stub returning a scripted log evidence per sweep."""

    def __init__(self, values) -> None:
        self.values = list(values)
        self.i = 0

    def run(self, k, data, params, graph, parent_summaries=None):
        v = self.values[min(self.i, len(self.values) - 1)]
        self.i += 1
        return _FakeSummary(v)


class _FakeMStep:
    """M-step stub tagging the parameters with the sweep that produced them."""

    def __init__(self) -> None:
        self.lr = 0.05
        self.calls = 0

    def run(self, k, q, params, data, hier=None, parent_summaries=None):
        self.calls += 1
        return {"sweep": self.calls}


class _FakeGraph:
    def n_edges(self, k=None) -> int:
        return 0


class _FakeData:
    K = 1
    hierarchy = None


def _guarded_sem(values, **kw):
    sem = TopDownSEM(
        do_structure=False,
        couple_scales=False,
        verbose=0,
        **kw,
    )
    sem.estep = _FakeEStep(values)
    sem.mstep = _FakeMStep()
    return sem


def test_a_decreasing_run_stops_at_the_best_sweep():
    sem = _guarded_sem([100.0, 90.0, 85.0, 80.0], max_outer_iter=10, outer_tol=1e-4)
    params, _, hist = sem.fit(_FakeData(), [{"sweep": 0}], _FakeGraph())
    assert hist.elbo == [100.0]
    assert hist.rejected == [90.0, 85.0, 80.0]
    # The returned parameters are the ones of the best sweep, not the last.
    assert params[0] == {"sweep": 1}


def test_a_rejected_sweep_halves_the_learning_rate():
    sem = _guarded_sem([100.0, 90.0, 85.0, 80.0], max_outer_iter=10, outer_tol=1e-4)
    sem.fit(_FakeData(), [{"sweep": 0}], _FakeGraph())
    # Two backtracks damped the rate twice; the third rejection stopped the fit.
    assert sem.mstep.lr == pytest.approx(0.05 / 4)


def test_the_damped_learning_rate_does_not_leak_into_the_next_fit():
    sem = _guarded_sem([100.0, 90.0, 85.0, 80.0], max_outer_iter=10, outer_tol=1e-4)
    sem.fit(_FakeData(), [{"sweep": 0}], _FakeGraph())
    assert sem.mstep.lr < 0.05
    sem.estep = _FakeEStep([50.0, 50.0, 50.0, 50.0])
    sem.fit(_FakeData(), [{"sweep": 0}], _FakeGraph())
    assert sem.mstep.lr == pytest.approx(0.05)


def test_a_sub_tolerance_wiggle_converges_at_the_best_sweep():
    sem = _guarded_sem([100.0, 100.001, 100.0009], max_outer_iter=10, outer_tol=1e-4)
    params, _, hist = sem.fit(_FakeData(), [{"sweep": 0}], _FakeGraph())
    assert hist.elbo == [100.0, 100.001]
    assert hist.rejected == [100.0009]
    assert params[0] == {"sweep": 2}


def test_the_guard_off_records_every_sweep():
    sem = _guarded_sem(
        [100.0, 90.0, 95.0],
        max_outer_iter=3,
        outer_tol=0.0,
        monotone_outer=False,
    )
    params, _, hist = sem.fit(_FakeData(), [{"sweep": 0}], _FakeGraph())
    assert hist.elbo == [100.0, 90.0, 95.0]
    assert hist.rejected == []
    assert params[0] == {"sweep": 3}


def test_a_worsening_warmstart_pass_is_rejected():
    sem = _guarded_sem([100.0, 100.0, 100.0], max_outer_iter=3, do_warmstart=True)
    sem._warmstart_pass = lambda data, params_list, graph, summaries, hier: (
        [{"sweep": 99}],
        _FakeGraph(),
        [_FakeSummary(60.0)],
    )
    params, _, hist = sem.fit(_FakeData(), [{"sweep": 0}], _FakeGraph())
    assert 60.0 in hist.rejected
    assert params[0] != {"sweep": 99}


def test_an_improving_warmstart_pass_is_recorded():
    sem = _guarded_sem([100.0, 100.0, 100.0], max_outer_iter=3, do_warmstart=True)
    sem._warmstart_pass = lambda data, params_list, graph, summaries, hier: (
        [{"sweep": 99}],
        _FakeGraph(),
        [_FakeSummary(200.0)],
    )
    params, _, hist = sem.fit(_FakeData(), [{"sweep": 0}], _FakeGraph())
    assert hist.elbo[-1] == 200.0
    assert params[0] == {"sweep": 99}


def test_the_guard_narrates_a_rejection_and_the_stop(capsys):
    sem = _guarded_sem([100.0, 90.0, 85.0, 80.0], max_outer_iter=10, outer_tol=1e-4)
    sem.verbose = 1
    sem.fit(_FakeData(), [{"sweep": 0}], _FakeGraph())
    out = capsys.readouterr().out
    assert "sweep rejected" in out
    assert "Stopped at the best sweep" in out


def test_the_guard_narrates_convergence_at_the_best_sweep(capsys):
    sem = _guarded_sem([100.0, 100.001, 100.0009], max_outer_iter=10, outer_tol=1e-4)
    sem.verbose = 1
    sem.fit(_FakeData(), [{"sweep": 0}], _FakeGraph())
    out = capsys.readouterr().out
    assert "Converged at the best sweep" in out


def test_the_recorded_trace_is_monotone_on_data(dataset):
    cfg, ds, graph = dataset
    sem = TopDownSEM(max_outer_iter=6, outer_tol=0.0, lam_edge=2.0, verbose=0)
    _, _, hist = sem.fit(ds.data, [deepcopy(p) for p in ds.params], deepcopy(graph))
    trace = np.asarray(hist.elbo)
    scale = 1.0 + abs(trace[-1])
    assert np.all(np.diff(trace) >= -1e-8 * scale)


# ---------------------------------------------------------------------------
# Per-edge shift confirmation
# ---------------------------------------------------------------------------


class _ConfirmScorer:
    """Scorer with a score table for real sets and a flat gain per fake key."""

    def __init__(self, n_rows: int, table, fake_gain: float) -> None:
        arr = np.arange(n_rows, dtype=float).reshape(n_rows, 1)
        self.feat_x = _LagColumns(np.tile(arr, (1, 4)))
        self.feat_y = _LagColumns(np.tile(arr, (1, 4)))
        self.b_init: dict = {}
        self.a_init: dict = {}
        self.table = {frozenset(k): v for k, v in table.items()}
        self.fake_gain = fake_gain

    def score(self, parent_set) -> float:
        real = frozenset(e for e in parent_set if e[0] != -1)
        n_fake = len(parent_set) - len(real)
        return self.table[real] + self.fake_gain * n_fake


def test_confirm_keeps_an_edge_that_beats_its_shifted_null():
    scorer = _ConfirmScorer(64, {(): 0.0, ((0, 1),): 10.0}, fake_gain=1.0)
    kept = _confirm_edges(
        scorer.score, scorer, {(0, 1)}, np.random.default_rng(0), 8, 0.95
    )
    assert kept == {(0, 1)}


def test_confirm_drops_an_edge_its_null_matches():
    scorer = _ConfirmScorer(64, {(): 0.0, ((0, 1),): 1.0}, fake_gain=1.0)
    kept = _confirm_edges(
        scorer.score, scorer, {(0, 1)}, np.random.default_rng(0), 8, 0.95
    )
    assert kept == set()


def test_confirm_rechecks_the_survivors_after_a_drop():
    table = {
        (): 0.0,
        ((0, 1),): 6.0,
        ((1, 1),): 6.0,
        ((0, 1), (1, 1)): 8.0,
    }
    scorer = _ConfirmScorer(64, table, fake_gain=5.0)
    kept = _confirm_edges(
        scorer.score,
        scorer,
        {(0, 1), (1, 1)},
        np.random.default_rng(0),
        8,
        0.95,
    )
    # Each edge adds 2.0 given the other, below the 5.0 null, so one drops;
    # alone the survivor adds 6.0, above the null, so it stays.
    assert len(kept) == 1


def test_confirm_returns_an_empty_set_unchanged():
    scorer = _ConfirmScorer(64, {(): 0.0}, fake_gain=1.0)
    assert (
        _confirm_edges(scorer.score, scorer, set(), np.random.default_rng(0), 4, 0.95)
        == set()
    )


def test_confirm_skips_a_series_too_short_to_shift():
    scorer = _ConfirmScorer(4, {(): 0.0, ((0, 1),): 0.1}, fake_gain=5.0)
    kept = _confirm_edges(
        scorer.score, scorer, {(0, 1)}, np.random.default_rng(0), 4, 0.95
    )
    assert kept == {(0, 1)}


def test_block_roll_permutes_inside_each_block_only():
    col = np.arange(12, dtype=float).reshape(12, 1)
    block_ids = np.repeat([0, 1, 2], 4)
    rolled = _block_roll(col, block_ids, np.random.default_rng(0))
    for b in range(3):
        rows = slice(4 * b, 4 * (b + 1))
        assert sorted(rolled[rows, 0]) == sorted(col[rows, 0])
    assert not np.array_equal(rolled, col)


def test_block_roll_leaves_singleton_blocks_alone():
    col = np.arange(3, dtype=float).reshape(3, 1)
    block_ids = np.array([0, 1, 2])
    rolled = _block_roll(col, block_ids, np.random.default_rng(0))
    assert np.array_equal(rolled, col)


def test_shift_gains_supports_both_null_families():
    scorer = _ConfirmScorer(64, {(): 0.0}, fake_gain=2.0)
    rng = np.random.default_rng(0)
    plain = _shift_gains(scorer, (0, 1), set(), rng, 5)
    block_ids = np.repeat(np.arange(8), 8)
    rolled = _shift_gains(scorer, (0, 1), set(), rng, 5, None, block_ids)
    assert plain.shape == (5,) and rolled.shape == (5,)
    assert np.allclose(plain, 2.0) and np.allclose(rolled, 2.0)


def test_the_protocol_stores_the_learned_graph_and_the_guard_count(dataset, tmp_path):
    import json

    from src.evaluation.protocol import EvaluationProtocol
    from src.synthetic.scenarios import ScenarioGrid

    cfg, _, _ = dataset
    protocol = EvaluationProtocol(
        methods={"n2dbn": TopDownSEM(max_outer_iter=2, verbose=0)},
        n_replicates=1,
        verbose=0,
    )
    results = protocol.run(ScenarioGrid(scenarios=[cfg]))
    r = results[0]
    assert set(r.learned_edges) == {"1", "2"}
    assert all(len(e) == 3 for edges in r.learned_edges.values() for e in edges)
    assert r.n_rejected_sweeps >= 0

    csv_path = tmp_path / "r.csv"
    json_path = tmp_path / "r.json"
    protocol.save_csv(results, str(csv_path))
    protocol.save_json(results, str(json_path))
    assert "n_rejected_sweeps" in csv_path.read_text().splitlines()[0]
    rec = json.loads(json_path.read_text())[0]
    assert rec["learned_edges"] == r.learned_edges
    assert rec["n_rejected_sweeps"] == r.n_rejected_sweeps


def test_sstep_confirmation_returns_a_subset_of_the_plain_search(dataset):
    cfg, ds, graph = dataset
    estep = EStep()
    q2 = estep.run(2, ds.data, ds.params[1], graph, None)
    q1 = estep.run(1, ds.data, ds.params[0], graph, q2)

    plain = SStep(lam=0.05, max_indegree=2, max_lag=2, seed=3)
    confirmed = SStep(
        lam=0.05,
        max_indegree=2,
        max_lag=2,
        confirm_edges=True,
        n_perm=8,
        seed=3,
    )
    _, p_plain = plain.run(1, q1, deepcopy(ds.params[0]), graph, ds.data)
    _, p_conf = confirmed.run(1, q1, deepcopy(ds.params[0]), graph, ds.data)
    for ell in range(cfg.L):
        assert p_conf.parent_sets[ell] <= p_plain.parent_sets[ell]
