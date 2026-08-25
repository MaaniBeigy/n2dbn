"""
tests/test_phase15_final.py
===========================
The last reachable lines: numerical helpers, fallback branches and the
degenerate configurations that the benchmark grids never produce.

What stays uncovered after this file is the set of import guards for optional
dependencies (matplotlib, networkx, pyAgrum), which can only run when those
packages are absent.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from src.core.graph import MultiScaleDBNGraph
from src.evaluation.baselines import TopDownNoRegime, _collapse_regimes
from src.inference.posterior import FactorPosterior, _safe_inv, _safe_solve
from src.learning.estep import EStep, _agg_lag_X, _agg_lag_Y, coarse_linear_predictors
from src.learning.sstep import _maximise
from src.learning.structural_em import TopDownSEM
from src.model.allocation import allocate_durations_block
from src.model.latent_factor import LatentFactor
from src.synthetic.families import make_generator
from src.synthetic.scenarios import ScenarioConfig, ScenarioGrid


def _cfg(**over) -> ScenarioConfig:
    base = dict(
        name="fin",
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
        seed=9,
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
    gen = make_generator(cfg, seed=9)
    ds = gen.generate(seed=9)
    graph = MultiScaleDBNGraph(hierarchy=ds.hierarchy, label_names=gen.label_names)
    for ki, p in enumerate(ds.params):
        for dst, parents in p.parent_sets.items():
            for src, lag in parents:
                graph.add_edge(ki + 1, src, lag, dst)
    return cfg, ds, graph


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------


def test_safe_solve_and_inverse_survive_a_singular_matrix():
    singular = np.zeros((2, 2))
    x = _safe_solve(singular, np.array([1.0, 2.0]))
    assert np.all(np.isfinite(x))
    inv = _safe_inv(singular)
    assert np.all(np.isfinite(inv))


def test_factor_smoother_returns_empty_arrays_without_a_factor(dataset):
    cfg, ds, _ = dataset
    params = ds.params[0].factor
    gamma = np.ones((5, params.M)) / params.M
    m_H, Sigma_H = FactorPosterior.smoother(gamma, params, np.zeros((5, 0)))
    assert m_H.shape == (5, 0) and Sigma_H.shape == (5, 0, 0)


def test_pseudo_obs_returns_empty_arrays_without_a_factor(dataset):
    cfg, ds, _ = dataset
    K = cfg.K
    X, Y = ds.data.get_X(K), ds.data.get_Y(K)
    T, L = X.shape
    z, R = FactorPosterior.compute_pseudo_obs_coarse(
        X,
        Y,
        np.empty((T, 0)),
        np.ones((T, ds.params[K - 1].M)),
        ds.params[K - 1],
        ds.data.get_C(K),
        np.zeros((T, L)),
        np.zeros((T, L)),
    )
    assert z.shape == (T, 0) and R.shape == (T, 0, 0)


def test_latent_factor_sampler_without_a_factor(dataset):
    cfg, ds, _ = dataset
    factor = ds.params[0].factor
    traj = LatentFactor(factor).sample_trajectory(
        6,
        np.zeros(6, dtype=int),
        np.zeros((6, factor.d_ctx)),
        np.random.default_rng(0),
    )
    assert traj.shape == (6, factor.r)


# ---------------------------------------------------------------------------
# Fallback branches
# ---------------------------------------------------------------------------


def test_lag_helpers_accumulate_when_history_exists(dataset):
    """The lagged term is added once the epoch index passes the lag."""
    cfg, ds, _ = dataset
    p = ds.params[0]  # scale 1 carries the parent sets here
    X, Y = ds.data.get_X(1), ds.data.get_Y(1)
    if not any(p.parent_sets.values()):
        pytest.skip("no lagged parents at this scale")
    seen = False
    for t in range(2, 40):
        vx = _agg_lag_X(t, X, p.parent_sets, p.fine_count.betaLag, cfg.L)
        vy = _agg_lag_Y(t, Y, p.parent_sets, p.fine_dur.alphaLag, cfg.L)
        seen = seen or bool(np.any(vx != 0)) or bool(np.any(vy != 0))
    assert seen, "no epoch produced a lagged contribution"


def test_coarse_predictors_without_context(dataset):
    cfg, ds, _ = dataset
    K = cfg.K
    params = deepcopy(ds.params[K - 1])
    X, Y = ds.data.get_X(K), ds.data.get_Y(K)
    T, L = X.shape
    zero = np.zeros((T, L))
    log_mu, logit_pi, log_nu = coarse_linear_predictors(
        X, np.empty((T, 0)), np.zeros((T, 0)), params, zero, zero
    )
    assert log_mu.shape == (T, params.M, L)
    assert np.all(np.isfinite(logit_pi)) and np.all(np.isfinite(log_nu))


def test_duration_allocation_below_the_floor():
    """A parent with fewer units than active children splits proportionally."""
    from src.model.params import FineDurParams

    p = FineDurParams(
        L=1,
        alpha0=np.zeros(1),
        alphaS=np.zeros((1, 1)),
        alphaH=np.zeros((0, 1)),
        alphaCtx=np.zeros((0, 1)),
        alphaX=np.zeros(1),
        alphaP=np.zeros(1),
        alphaLag={},
        tau=8.0,
    )
    out = allocate_durations_block(
        c=0,
        children=[0, 1, 2],
        ell=0,
        Y_parent=2,
        X_fine_children=np.array([2, 2, 2]),
        s_seq=np.zeros(3, dtype=int),
        H_traj=np.zeros((3, 0)),
        ctx=np.zeros((3, 0)),
        X_fine=np.ones((3, 1), dtype=np.int64),
        Y_fine=np.zeros((3, 1), dtype=np.int64),
        p=p,
        parent_sets_ell=set(),
        rng=np.random.default_rng(0),
        floor_per_active=True,
    )
    assert int(out.sum()) == 2


def test_duration_allocation_without_active_children():
    from src.model.params import FineDurParams

    p = FineDurParams(
        L=1,
        alpha0=np.zeros(1),
        alphaS=np.zeros((1, 1)),
        alphaH=np.zeros((0, 1)),
        alphaCtx=np.zeros((0, 1)),
        alphaX=np.zeros(1),
        alphaP=np.zeros(1),
        alphaLag={},
        tau=8.0,
    )
    out = allocate_durations_block(
        c=0,
        children=[0, 1],
        ell=0,
        Y_parent=5,
        X_fine_children=np.array([0, 0]),
        s_seq=np.zeros(2, dtype=int),
        H_traj=np.zeros((2, 0)),
        ctx=np.zeros((2, 0)),
        X_fine=np.zeros((2, 1), dtype=np.int64),
        Y_fine=np.zeros((2, 1), dtype=np.int64),
        p=p,
        parent_sets_ell=set(),
        rng=np.random.default_rng(0),
    )
    assert int(out.sum()) == 0


def test_maximise_gives_up_after_a_failed_line_search():
    """A step that never improves ends the search instead of looping."""
    calls = {"n": 0}

    def obj(b):
        calls["n"] += 1
        return 0.0 if calls["n"] == 1 else -1.0

    x, f = _maximise(obj, lambda b: np.ones(1), np.array([0.0]), n_iter=3)
    assert f == 0.0 and x[0] == 0.0


def test_collapse_regimes_is_a_no_op_when_already_single(dataset):
    cfg, ds, _ = dataset
    single = _collapse_regimes(deepcopy(ds.params[0]))
    again = _collapse_regimes(single)
    assert again.regime.M == 1


def test_no_regime_baseline_on_single_regime_data():
    cfg = _cfg(M_k=[1, 1], family=1)
    gen = make_generator(cfg, seed=2)
    ds = gen.generate(seed=2)
    graph = MultiScaleDBNGraph(hierarchy=ds.hierarchy, label_names=gen.label_names)
    params, _, hist = TopDownNoRegime(max_outer_iter=1, verbose=0).fit(
        ds.data, [deepcopy(p) for p in ds.params], graph
    )
    assert all(p.M == 1 for p in params)
    assert np.isfinite(hist.elbo[-1])


def test_cross_scale_warm_start_ignores_a_missing_block(dataset):
    cfg, ds, graph = dataset
    sem = TopDownSEM(max_outer_iter=1, verbose=0)
    coarsest = deepcopy(ds.params[cfg.K - 1])
    assert coarsest.regime.cross is None
    q = EStep().run(cfg.K, ds.data, coarsest, graph)
    sem._init_cross_scale(coarsest, q, q, ds.hierarchy, 1)  # returns without acting
    assert coarsest.regime.cross is None


# ---------------------------------------------------------------------------
# Scenario helpers
# ---------------------------------------------------------------------------


def test_scenario_grid_filter_and_iteration():
    grid = ScenarioGrid.sanity_check()
    subset = grid.filter(family=2)
    assert 0 < len(subset) < len(grid)
    assert all(cfg.family == 2 for cfg in subset)
    assert len(list(iter(grid))) == len(grid)


def test_short_id_encodes_the_configuration():
    cfg = _cfg(K=2, L=3, M_k=[2, 2], r_k=[0, 0], zi_level="high", seed=11)
    assert cfg.short_id() == "F2_L3_K2_M22_r00_zih_s11"


def test_family_four_misspecification_variants():
    for misspec in (None, "heavier_tail", "omit_context"):
        cfg = _cfg(family=4, misspec_type=misspec, name=f"fiv_{misspec}")
        ds = make_generator(cfg, seed=3).generate(seed=3)
        assert ds.data.check_reconciliation()["RecErr_X"] == 0.0
