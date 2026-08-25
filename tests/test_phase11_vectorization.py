"""
tests/test_phase11_vectorization.py
===================================
Regression tests for the vectorised learning code and the statistical fixes
that came with it.

Groups
------
1.  BlockIndex segment reductions match explicit per-block loops.
2.  Vectorised E-step pieces reproduce the original per-epoch implementations.
3.  The finer-scale allocation likelihood is the Dir-Mult one and is *bounded*
    (it reduces to the multinomial limit as the concentration grows).
4.  Analytic Dir-Mult gradients match finite differences.
5.  The S-step can add an edge at all (profiled coefficients) and refuses one
    under a large penalty.
6.  SEM convergence uses a relative tolerance, so a converged fit stops early.
7.  Structure metrics score an empty-vs-empty scale as perfect.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from src.core.blocks import BlockIndex
from src.core.graph import MultiScaleDBNGraph
from src.evaluation.baselines import OracleParams
from src.evaluation.metrics import StructureMetrics
from src.inference.posterior import RegimePosterior, logsumexp
from src.learning.estep import (
    EStep,
    _agg_lag_X,
    block_softmax,
    build_log_like_coarsest,
    build_log_like_fine,
    dirmult_grad,
    dirmult_ll,
    fine_scores,
    lag_contrib,
)
from src.learning.sstep import SStep
from src.learning.structural_em import TopDownSEM, TrainingHistory
from src.synthetic.families import make_generator
from src.synthetic.scenarios import ScenarioConfig, ScenarioGrid

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _cfg(**over) -> ScenarioConfig:
    base = dict(
        name="vec",
        K=3,
        H_finest=216,
        branching=[6, 6],
        L=3,
        M_k=[2, 2, 2],
        r_k=[0, 0, 0],
        max_lag=[2, 2, 1],
        max_indegree=[1, 1, 0],
        zi_level="medium",
        sparse_graph=True,
        family=2,
        seed=5,
        n_replicates=1,
        edge_strength=3.0,
        finest_rate=1.0,
    )
    base.update(over)
    return ScenarioConfig(**base)


def _dataset(cfg=None, seed=5):
    cfg = cfg or _cfg()
    gen = make_generator(cfg, seed=seed)
    ds = gen.generate(seed=seed)
    graph = MultiScaleDBNGraph(hierarchy=ds.hierarchy, label_names=gen.label_names)
    for ki, p in enumerate(ds.params):
        for dst, parents in p.parent_sets.items():
            for src, lag in parents:
                graph.add_edge(ki + 1, src, lag, dst)
    return cfg, ds, graph


@pytest.fixture(scope="module")
def dataset():
    return _dataset()


# ---------------------------------------------------------------------------
# 1. BlockIndex
# ---------------------------------------------------------------------------


def test_block_index_matches_loop(dataset):
    cfg, ds, _ = dataset
    hier = ds.hierarchy
    bi = BlockIndex.from_hierarchy(hier, 1)
    X = ds.data.get_X(1).astype(float)

    loop_sum = np.stack(
        [X[hier.children(1, c)].sum(axis=0) for c in range(hier.n_epochs(2))]
    )
    assert np.allclose(bi.sum(X), loop_sum)

    loop_max = np.stack(
        [X[hier.children(1, c)].max(axis=0) for c in range(hier.n_epochs(2))]
    )
    assert np.allclose(bi.max(X), loop_max)

    loop_lse = np.stack(
        [logsumexp(X[hier.children(1, c)], axis=0) for c in range(hier.n_epochs(2))]
    )
    assert np.allclose(bi.logsumexp(X), loop_lse)

    # broadcast is the inverse of the parent map
    back = bi.broadcast(bi.sum(X))
    assert back.shape == X.shape
    assert np.allclose(back[0], loop_sum[hier.parent(1, 0)])


def test_block_softmax_sums_to_one_per_block(dataset):
    cfg, ds, _ = dataset
    bi = BlockIndex.from_hierarchy(ds.hierarchy, 1)
    rng = np.random.default_rng(0)
    scores = rng.standard_normal((ds.hierarchy.n_epochs(1), cfg.L))
    omega = block_softmax(scores, bi)
    assert np.allclose(bi.sum(omega), 1.0)


# ---------------------------------------------------------------------------
# 2. Vectorised E-step pieces
# ---------------------------------------------------------------------------


def test_lag_contrib_matches_per_epoch(dataset):
    cfg, ds, _ = dataset
    p = ds.params[cfg.K - 1]
    X = ds.data.get_X(cfg.K)
    L = cfg.L
    ref = np.stack(
        [
            _agg_lag_X(t, X, p.parent_sets, p.coarse_count.betaLag, L)
            for t in range(X.shape[0])
        ]
    )
    fast = lag_contrib(X, p.coarse_count.betaLag, p.parent_sets, L)
    assert np.allclose(fast, ref)


def test_build_log_trans_matches_per_epoch(dataset):
    cfg, ds, _ = dataset
    p = ds.params[cfg.K - 1].regime
    ctx = ds.data.get_C(cfg.K)
    log_init, log_trans = RegimePosterior.build_log_trans(p, ctx)

    for t in (1, ctx.shape[0] // 2, ctx.shape[0] - 1):
        for prev in range(p.M):
            logits = p.logit_transition(prev, None, ctx[t])
            expected = logits - logsumexp(logits)
            assert np.allclose(log_trans[t - 1][:, prev], expected)
    # rows normalise
    assert np.allclose(logsumexp(log_trans, axis=1), 0.0, atol=1e-10)


def test_forward_backward_single_regime_fast_path():
    rng = np.random.default_rng(1)
    T = 25
    log_like = rng.standard_normal((T, 1))
    log_init = np.zeros(1)
    log_trans = np.zeros((T - 1, 1, 1))
    gamma, xi, log_Z = RegimePosterior.forward_backward(log_like, log_init, log_trans)
    assert np.allclose(gamma, 1.0)
    assert np.allclose(xi, 1.0)
    assert log_Z == pytest.approx(float(log_like.sum()))


def test_estep_r0_coarsest_single_pass(dataset):
    """With r=0 at the coarsest scale the inner loop cannot change anything."""
    cfg, ds, graph = dataset
    a = EStep(max_inner_iter=1).run(cfg.K, ds.data, ds.params[cfg.K - 1], graph)
    b = EStep(max_inner_iter=10).run(cfg.K, ds.data, ds.params[cfg.K - 1], graph)
    assert a.log_Z == pytest.approx(b.log_Z)
    assert np.allclose(a.gamma, b.gamma)


# ---------------------------------------------------------------------------
# 3. Finer-scale allocation likelihood
# ---------------------------------------------------------------------------


def test_fine_log_like_is_bounded_under_scaling(dataset):
    """
    The old score Σ_t X_t η_t grew without bound when the scores were inflated;
    a normalised allocation likelihood must not.
    """
    cfg, ds, _ = dataset
    k = 1
    p = deepcopy(ds.params[k - 1])
    args = (
        ds.data.get_X(k),
        ds.data.get_Y(k),
        ds.data.get_X(k + 1),
        ds.data.get_Y(k + 1),
        np.empty((ds.hierarchy.n_epochs(k), 0)),
        ds.data.get_C(k),
    )
    base = build_log_like_fine(*args, p, ds.hierarchy, k).sum()

    p_big = deepcopy(p)
    p_big.fine_count.beta0 = p_big.fine_count.beta0 + 25.0
    p_big.fine_dur.alpha0 = p_big.fine_dur.alpha0 + 25.0
    shifted = build_log_like_fine(*args, p_big, ds.hierarchy, k).sum()

    # A constant shift of every score cancels inside the block softmax: this is
    # exactly what the missing normaliser used to break.
    assert shifted == pytest.approx(base, rel=1e-6)
    assert np.isfinite(base)

    # And it behaves like a likelihood: the generating parameters beat noise.
    rng = np.random.default_rng(7)
    p_bad = deepcopy(p)
    p_bad.fine_count.betaS = p_bad.fine_count.betaS + rng.normal(
        0, 3.0, p_bad.fine_count.betaS.shape
    )
    p_bad.fine_count.betaP = p_bad.fine_count.betaP + rng.normal(
        0, 3.0, p_bad.fine_count.betaP.shape
    )
    worse = build_log_like_fine(*args, p_bad, ds.hierarchy, k).sum()
    assert worse < base


def test_dirmult_reduces_to_multinomial_limit(dataset):
    """log Γ(x+κω) − log Γ(κω) → x log ω  as κ → ∞ (up to a constant)."""
    cfg, ds, _ = dataset
    k = 1
    bi = BlockIndex.from_hierarchy(ds.hierarchy, k)
    X = ds.data.get_X(k).astype(float)
    rng = np.random.default_rng(2)
    scores = rng.standard_normal(X.shape)
    omega = block_softmax(scores, bi)

    big = 1e7
    approx = dirmult_ll(X, omega, big)
    exact = X * (np.log(big) + np.log(np.maximum(omega, 1e-300)))
    mask = omega > 1e-6
    assert np.allclose(approx[mask], exact[mask], rtol=1e-5, atol=1e-3)


# ---------------------------------------------------------------------------
# 4. Gradients
# ---------------------------------------------------------------------------


def test_dirmult_grad_matches_finite_difference(dataset):
    cfg, ds, _ = dataset
    k = 1
    bi = BlockIndex.from_hierarchy(ds.hierarchy, k)
    X = ds.data.get_X(k).astype(float)
    rng = np.random.default_rng(3)
    scores = rng.standard_normal(X.shape) * 0.5
    conc = 8.0

    def total(s):
        return float(dirmult_ll(X, block_softmax(s, bi), conc).sum())

    grad = dirmult_grad(X, block_softmax(scores, bi), conc, bi)
    eps = 1e-5
    for t, ell in [(0, 0), (7, 1), (30, 2)]:
        bump = scores.copy()
        bump[t, ell] += eps
        drop = scores.copy()
        drop[t, ell] -= eps
        fd = (total(bump) - total(drop)) / (2 * eps)
        assert fd == pytest.approx(float(grad[t, ell]), rel=1e-3, abs=1e-5)


# ---------------------------------------------------------------------------
# 5. S-step can actually add edges
# ---------------------------------------------------------------------------


def _empty_params(ds, cfg):
    """Ground-truth parameters with every lagged edge stripped out."""
    out = []
    for p in ds.params:
        p = deepcopy(p)
        p.parent_sets = {ell: set() for ell in range(cfg.L)}
        if p.is_coarsest:
            p.coarse_count.betaLag = {}
            p.coarse_dur.alphaLag = {}
        else:
            p.fine_count.betaLag = {}
            p.fine_dur.alphaLag = {}
        out.append(p)
    return out


def test_sstep_can_add_edge_from_empty_graph():
    """
    Regression: with a zero coefficient every candidate scored exactly −λ, so
    the search could only ever delete edges - structure recovery was impossible
    by construction.  Profiling the coefficient fixes that.
    """
    cfg, ds, _ = _dataset(_cfg(edge_strength=6.0, L=3))
    params = _empty_params(ds, cfg)
    empty_graph = MultiScaleDBNGraph(
        hierarchy=ds.hierarchy, label_names=[f"L{i}" for i in range(cfg.L)]
    )
    k = 2
    q = EStep().run(k, ds.data, params[k - 1], empty_graph)
    sstep = SStep(lam=0.0, max_lag=cfg.max_lag[k - 1], max_indegree=2)
    new_graph, new_params = sstep.run(
        k=k,
        q=q,
        params=params[k - 1],
        graph=empty_graph,
        data=ds.data,
        hier=ds.hierarchy,
    )
    assert new_graph.n_edges(k) > 0, "S-step never adds an edge"
    # and the coefficient it fitted is written back, not left at zero
    for ell, pset in new_params.parent_sets.items():
        for key in pset:
            assert key in new_params.fine_count.betaLag
    assert any(
        abs(float(v[ell])) > 0
        for ell, pset in new_params.parent_sets.items()
        for key, v in new_params.fine_count.betaLag.items()
        if key in pset
    )


def test_sstep_high_penalty_keeps_graph_empty():
    cfg, ds, _ = _dataset(_cfg(edge_strength=6.0, L=3))
    params = _empty_params(ds, cfg)
    empty_graph = MultiScaleDBNGraph(
        hierarchy=ds.hierarchy, label_names=[f"L{i}" for i in range(cfg.L)]
    )
    k = 2
    q = EStep().run(k, ds.data, params[k - 1], empty_graph)
    sstep = SStep(lam=1e6, penalty="fixed", max_lag=cfg.max_lag[k - 1])
    new_graph, _ = sstep.run(
        k=k,
        q=q,
        params=params[k - 1],
        graph=empty_graph,
        data=ds.data,
        hier=ds.hierarchy,
    )
    assert new_graph.n_edges(k) == 0


def test_sstep_per_scale_search_space(dataset):
    """max_lag / max_indegree may be given per scale."""
    cfg, ds, graph = dataset
    sstep = SStep(lam=1.0, max_lag={1: 1, 2: 2, 3: 1}, max_indegree={1: 1, 2: 1, 3: 1})
    k = 2
    q = EStep().run(k, ds.data, ds.params[k - 1], graph)
    new_graph, new_params = sstep.run(
        k=k,
        q=q,
        params=ds.params[k - 1],
        graph=graph,
        data=ds.data,
        hier=ds.hierarchy,
    )
    for ell in range(cfg.L):
        pset = new_params.parent_sets[ell]
        assert len(pset) <= 1
        assert all(d <= 2 for _, d in pset)


# ---------------------------------------------------------------------------
# 6. Convergence
# ---------------------------------------------------------------------------


def test_convergence_is_relative():
    hist = TrainingHistory()
    hist.elbo = [-5000.0, -4000.0, -4000.05, -4000.08]
    # absolute deltas are 0.05 / 0.03 - far above the old 1e-3 threshold,
    # but negligible relative to |ELBO| ≈ 4000
    assert hist.converged(tol=1e-4, n_check=2)
    hist.elbo = [-5000.0, -4000.0, -3000.0, -2000.0]
    assert not hist.converged(tol=1e-4, n_check=2)


def test_sem_stops_early_when_converged(dataset):
    """A fit whose ELBO has stopped moving must exit the outer loop."""
    cfg, ds, graph = dataset
    sem = TopDownSEM(
        max_outer_iter=25,
        do_structure=False,
        verbose=0,
        mstep_kwargs=dict(n_steps_regime=0, n_steps_emission=0),
    )
    _, _, hist = sem.fit(ds.data, [deepcopy(p) for p in ds.params], deepcopy(graph))
    assert len(hist.elbo) < 25, "outer loop always exhausts max_outer_iter"


# ---------------------------------------------------------------------------
# 7. Metrics + oracle baseline + probe
# ---------------------------------------------------------------------------


def test_structure_metrics_empty_scale_is_perfect(dataset):
    cfg, ds, graph = dataset
    empty = MultiScaleDBNGraph(
        hierarchy=ds.hierarchy, label_names=[f"L{i}" for i in range(cfg.L)]
    )
    m = StructureMetrics.compute(empty, empty).to_dict()
    assert m["F1_mean"] == pytest.approx(1.0)
    assert m["SHD_total"] == 0.0


def test_oracle_params_baseline_is_a_noop(dataset):
    cfg, ds, graph = dataset
    params = [deepcopy(p) for p in ds.params]
    learned, learned_graph, hist = OracleParams().fit(ds.data, params, graph)
    assert learned is params
    assert learned_graph is graph
    assert len(hist.elbo) == 1 and np.isfinite(hist.elbo[0])


def test_identifiability_probe_reports_signal_and_ceiling():
    cfg = _cfg(edge_strength=6.0)
    out = ScenarioGrid.identifiability_probe(cfg, seed=5)
    assert set(out) == {1, 2, 3}
    for k, row in out.items():
        assert row["n_epochs"] > 0
        assert 0.0 <= row["density"] <= 1.0
        # A probability, or not applicable where the true path never switches.
        assert np.isnan(row["oracle_accS"]) or 0.0 <= row["oracle_accS"] <= 1.0
        assert row["lam_bic"] > 0
        assert np.isfinite(row["null_gain95"])
