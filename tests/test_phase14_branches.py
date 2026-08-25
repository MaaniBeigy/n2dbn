"""
tests/test_phase14_branches.py
==============================
The last reachable branches: guard clauses, degenerate inputs and the verbose
paths of the outer loop.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from src.core.graph import MultiScaleDBNGraph
from src.core.scales import ScaleHierarchy
from src.core.variables import LatentTrajectories, MultiScaleData
from src.evaluation.metrics import ReconstructionMetrics, _safe_div
from src.inference.reconstruction import Episode, EpisodeReconstructor
from src.learning.estep import EStep, _agg_lag_X, _agg_lag_Y
from src.learning.sstep import SStep, _maximise
from src.learning.structural_em import TopDownSEM
from src.model.allocation import redistribute_excess
from src.synthetic.families import make_generator
from src.synthetic.scenarios import ScenarioConfig


def _cfg(**over) -> ScenarioConfig:
    base = dict(
        name="br",
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
# Episodes
# ---------------------------------------------------------------------------


def test_episode_overlap_and_repr():
    a = Episode(label=0, t_start=0, t_end=4)
    b = Episode(label=0, t_start=3, t_end=6)
    c = Episode(label=0, t_start=5, t_end=6)
    assert a.overlaps(b) and b.overlaps(a)
    assert not a.overlaps(c)
    assert "dur=5" in repr(a)


def test_reconstruction_metrics_to_dict():
    m = ReconstructionMetrics(start_f1=0.5, iou=0.25)
    assert m.to_dict() == {"StartF1": 0.5, "IoU": 0.25}


def test_safe_div_guards_zero():
    assert _safe_div(1.0, 0.0) == 0.0
    assert _safe_div(2.0, 4.0) == 0.5


def test_reconstructor_projects_exclusive_labels():
    rec = EpisodeReconstructor(["a"], exclusivity_set={0})
    X1 = np.zeros((10, 1), dtype=np.int64)
    Y1 = np.zeros((10, 1), dtype=np.int64)
    X1[0, 0] = 1
    Y1[0, 0] = 6  # would run into the next episode
    X1[3, 0] = 1
    Y1[3, 0] = 2
    traces = rec.reconstruct(X1, Y1)
    first, second = traces[0]
    assert first.t_end < second.t_start


# ---------------------------------------------------------------------------
# Lagged features and allocation guards
# ---------------------------------------------------------------------------


def test_lag_helpers_skip_epochs_before_the_lag(dataset):
    cfg, ds, _ = dataset
    p = ds.params[cfg.K - 1]
    X = ds.data.get_X(cfg.K)
    Y = ds.data.get_Y(cfg.K)
    # epoch 0 has no history, so every lagged term is dropped
    assert np.allclose(
        _agg_lag_X(0, X, p.parent_sets, p.coarse_count.betaLag, cfg.L), 0
    )
    assert np.allclose(_agg_lag_Y(0, Y, p.parent_sets, p.coarse_dur.alphaLag, cfg.L), 0)


def test_redistribute_excess_rejects_an_infeasible_total():
    with pytest.raises(ValueError, match="exceeds the capacity"):
        redistribute_excess(
            np.array([5, 0]),
            np.array([1, 1]),
            np.array([1.0, 1.0]),
            np.random.default_rng(0),
        )


def test_redistribute_excess_falls_back_to_room_when_scores_vanish():
    out = redistribute_excess(
        np.array([3, 0, 0]),
        np.array([1, 1, 1]),
        np.array([0.0, 0.0, 0.0]),
        np.random.default_rng(0),
    )
    assert out.sum() == 3 and out.max() <= 1


def test_redistribute_excess_is_a_no_op_within_capacity():
    x = np.array([1, 0, 2])
    out = redistribute_excess(
        x, np.array([2, 2, 2]), np.ones(3), np.random.default_rng(0)
    )
    np.testing.assert_array_equal(out, x)


# ---------------------------------------------------------------------------
# Search helpers
# ---------------------------------------------------------------------------


def test_maximise_returns_immediately_without_parameters():
    x, f = _maximise(lambda b: 1.0, lambda b: np.zeros(0), np.zeros(0))
    assert x.size == 0 and f == 1.0


def test_maximise_stops_on_a_vanishing_gradient():
    calls = {"n": 0}

    def obj(b):
        calls["n"] += 1
        return -float(b[0] ** 2)

    x, f = _maximise(obj, lambda b: np.zeros(1), np.array([0.0]))
    assert x[0] == 0.0 and f == 0.0


def test_sstep_per_scale_settings_accept_a_sequence(dataset):
    cfg, ds, graph = dataset
    step = SStep(max_lag=[2, 1], max_indegree=[1, 1])
    assert step._per_scale(step.max_lag, 1) == 2
    assert step._per_scale(step.max_indegree, 2) == 1
    assert step._per_scale(3, 2) == 3


# ---------------------------------------------------------------------------
# Outer loop, verbose paths
# ---------------------------------------------------------------------------


def test_sem_verbose_reports_every_scale(dataset, capsys):
    cfg, ds, graph = dataset
    sem = TopDownSEM(max_outer_iter=1, do_structure=True, verbose=2)
    sem.fit(ds.data, [deepcopy(p) for p in ds.params], deepcopy(graph))
    out = capsys.readouterr().out
    assert "scale k=" in out and "|E_k|=" in out


def test_sem_reports_convergence(dataset, capsys):
    cfg, ds, graph = dataset
    sem = TopDownSEM(
        max_outer_iter=8,
        do_structure=False,
        verbose=1,
        mstep_kwargs=dict(n_steps_regime=0, n_steps_emission=0),
    )
    _, _, hist = sem.fit(ds.data, [deepcopy(p) for p in ds.params], deepcopy(graph))
    assert "Converged" in capsys.readouterr().out
    assert len(hist.elbo) < 8


def test_sem_warmstart_announces_itself(dataset, capsys):
    cfg, ds, graph = dataset
    sem = TopDownSEM(max_outer_iter=1, do_structure=False, do_warmstart=True, verbose=1)
    sem.fit(ds.data, [deepcopy(p) for p in ds.params], deepcopy(graph))
    assert "warm-start" in capsys.readouterr().out.lower()


def test_cross_scale_warm_start_skips_the_coarsest_scale(dataset):
    """The coarsest scale has no parent, so its block stays None."""
    cfg, ds, graph = dataset
    sem = TopDownSEM(max_outer_iter=1, do_structure=False, verbose=0)
    learned, _, _ = sem.fit(ds.data, [deepcopy(p) for p in ds.params], deepcopy(graph))
    assert learned[cfg.K - 1].regime.cross is None


def test_edge_penalty_is_zero_without_structure_learning(dataset):
    cfg, ds, graph = dataset
    sem = TopDownSEM(max_outer_iter=1, do_structure=False, verbose=0)
    _, _, hist = sem.fit(ds.data, [deepcopy(p) for p in ds.params], deepcopy(graph))
    assert hist.penalty == [0.0]


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------


def test_latent_trajectories_accessors(dataset):
    cfg, ds, _ = dataset
    assert ds.latents.K == cfg.K
    assert ds.latents.get_S(1).shape[0] == ds.hierarchy.n_epochs(1)
    assert ds.latents.get_H(1).shape[0] == ds.hierarchy.n_epochs(1)


def test_reconciliation_skips_empty_blocks():
    hier = ScaleHierarchy.from_explicit_maps(
        H=[3, 2], parent_maps=[np.array([0, 0, 0]), None]
    )
    data = MultiScaleData(
        hierarchy=hier,
        label_names=["a"],
        X=[np.array([[1], [1], [1]]), np.array([[3], [0]])],
        Y=[np.array([[1], [1], [1]]), np.array([[3], [0]])],
        C=[np.zeros((3, 0)), np.zeros((2, 0))],
    )
    errs = data.check_reconciliation()
    assert errs["RecErr_X"] == 0.0 and errs["RecErr_Y"] == 0.0
