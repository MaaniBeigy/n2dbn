"""
tests/test_phase09_coverage.py
==============================
TODO-5: Coverage gap tests for core and learning modules.

Targets uncovered branches in:
- structural_em.py: warmstart pass, verbose output, compute_elbo
- mstep.py: no-context regime init, cross-scale regime coupling, lagged alloc
- sstep.py: beam search dispatch, fine-scale lagged scoring
- graph.py: set_parents, n_edges (all scales), is_acyclic, copy, __repr__

Run with:
    pytest tests/test_phase09_coverage.py -v
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from src.core.graph import MultiScaleDBNGraph
from src.core.scales import ScaleHierarchy
from src.inference.posterior import PosteriorSummaries
from src.learning.estep import EStep
from src.learning.mstep import (
    MStep,
    update_allocation_scores,
    update_regime_transitions,
)
from src.learning.sstep import LocalScore, SStep
from src.learning.structural_em import TopDownSEM
from src.synthetic.families import make_generator
from src.synthetic.scenarios import ScenarioConfig

# ---------------------------------------------------------------------------
# Scenario configs (tiny for speed)
# ---------------------------------------------------------------------------

_CFG_MINI = ScenarioConfig(
    name="cov_mini",
    K=2,
    H_finest=12,
    branching=[3],
    L=2,
    M_k=[1, 1],
    r_k=[0, 0],
    max_lag=[1, 1],
    max_indegree=[1, 1],
    zi_level="low",
    sparse_graph=False,
    family=1,
    seed=90,
)

_CFG_M2 = ScenarioConfig(
    name="cov_m2",
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
    seed=91,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mini_ds():
    gen = make_generator(_CFG_MINI, seed=_CFG_MINI.seed)
    return gen.generate(seed=_CFG_MINI.seed)


@pytest.fixture(scope="module")
def mini_gen():
    return make_generator(_CFG_MINI, seed=_CFG_MINI.seed)


@pytest.fixture(scope="module")
def mini_graph(mini_ds, mini_gen):
    return MultiScaleDBNGraph(
        hierarchy=mini_ds.hierarchy,
        label_names=mini_gen.label_names,
    )


@pytest.fixture(scope="module")
def mini_qs(mini_ds, mini_graph):
    K = mini_ds.data.K
    estep = EStep(max_inner_iter=2, use_laplace=False)
    q_coarse = estep.run(
        k=K, data=mini_ds.data, params=mini_ds.params[K - 1], graph=mini_graph
    )
    q_fine = estep.run(
        k=1,
        data=mini_ds.data,
        params=mini_ds.params[0],
        graph=mini_graph,
        parent_summaries=q_coarse,
    )
    return q_coarse, q_fine


@pytest.fixture(scope="module")
def m2_ds():
    gen = make_generator(_CFG_M2, seed=_CFG_M2.seed)
    return gen.generate(seed=_CFG_M2.seed)


@pytest.fixture(scope="module")
def m2_gen():
    return make_generator(_CFG_M2, seed=_CFG_M2.seed)


@pytest.fixture(scope="module")
def m2_graph(m2_ds, m2_gen):
    return MultiScaleDBNGraph(
        hierarchy=m2_ds.hierarchy,
        label_names=m2_gen.label_names,
    )


@pytest.fixture(scope="module")
def m2_q_coarse(m2_ds, m2_graph):
    K = m2_ds.data.K
    estep = EStep(max_inner_iter=2, use_laplace=False)
    return estep.run(k=K, data=m2_ds.data, params=m2_ds.params[K - 1], graph=m2_graph)


# ===========================================================================
# structural_em.py coverage
# ===========================================================================


def test_sem_warmstart_pass():
    """TopDownSEM(do_warmstart=True) runs the _warmstart_pass (lines 299-328)."""
    cfg = ScenarioConfig(
        name="cov_warmstart",
        K=2,
        H_finest=12,
        branching=[3],
        L=2,
        M_k=[1, 1],
        r_k=[0, 0],
        max_lag=[1, 1],
        max_indegree=[0, 0],
        zi_level="low",
        sparse_graph=False,
        family=1,
        seed=92,
    )
    gen = make_generator(cfg, seed=cfg.seed)
    ds = gen.generate(seed=cfg.seed)
    graph = MultiScaleDBNGraph(hierarchy=ds.hierarchy, label_names=gen.label_names)

    sem = TopDownSEM(
        max_outer_iter=1,
        do_structure=False,
        do_warmstart=True,
        verbose=0,
        estep_kwargs={"max_inner_iter": 2, "use_laplace": False},
    )
    params_list, graph_out, history = sem.fit(ds.data, ds.params, graph)

    # warmstart adds one extra record to history
    assert (
        len(history.elbo) == 2
    ), f"Expected 2 ELBO entries (1 outer + 1 warmstart), got {len(history.elbo)}"
    assert all(np.isfinite(e) for e in history.elbo)


def test_sem_verbose_output(capsys):
    """verbose=2 produces print output covering lines 219-277."""
    cfg = ScenarioConfig(
        name="cov_verbose",
        K=2,
        H_finest=12,
        branching=[3],
        L=2,
        M_k=[1, 1],
        r_k=[0, 0],
        max_lag=[1, 1],
        max_indegree=[0, 0],
        zi_level="low",
        sparse_graph=False,
        family=1,
        seed=93,
    )
    gen = make_generator(cfg, seed=cfg.seed)
    ds = gen.generate(seed=cfg.seed)
    graph = MultiScaleDBNGraph(hierarchy=ds.hierarchy, label_names=gen.label_names)

    sem = TopDownSEM(
        max_outer_iter=1,
        do_structure=False,
        do_warmstart=False,
        verbose=2,
        estep_kwargs={"max_inner_iter": 2, "use_laplace": False},
    )
    sem.fit(ds.data, ds.params, graph)

    captured = capsys.readouterr()
    assert "[SEM] Outer iteration" in captured.out
    assert "E-step" in captured.out
    assert "M-step" in captured.out
    assert "(skipped)" in captured.out  # do_structure=False path


def test_sem_compute_elbo(mini_ds, mini_graph):
    """compute_elbo() returns finite float (line 342)."""
    K = mini_ds.data.K
    estep = EStep(max_inner_iter=2, use_laplace=False)
    q_K = estep.run(
        k=K, data=mini_ds.data, params=mini_ds.params[K - 1], graph=mini_graph
    )
    q_1 = estep.run(
        k=1,
        data=mini_ds.data,
        params=mini_ds.params[0],
        graph=mini_graph,
        parent_summaries=q_K,
    )

    sem = TopDownSEM(max_outer_iter=1, verbose=0)
    summaries = [q_1, q_K]  # index 0 = scale 1 (finest), index 1 = scale 2 (coarsest)
    elbo = sem.compute_elbo(summaries)

    assert np.isfinite(elbo), f"compute_elbo returned non-finite: {elbo}"
    assert isinstance(elbo, float)


# ===========================================================================
# mstep.py coverage
# ===========================================================================


def test_mstep_no_context_regime_init():
    """d_ctx=0 hits the no-context initial distribution path (line 143)."""
    cfg = ScenarioConfig(
        name="cov_noctx",
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
        family=1,
        seed=94,
    )
    gen = make_generator(cfg, seed=cfg.seed)
    ds = gen.generate(seed=cfg.seed)
    K = ds.data.K
    params = ds.params[K - 1]
    graph = MultiScaleDBNGraph(hierarchy=ds.hierarchy, label_names=gen.label_names)

    # Ensure d_ctx=0 by zeroing out context
    estep = EStep(max_inner_iter=2, use_laplace=False)
    q = estep.run(k=K, data=ds.data, params=params, graph=graph)

    # Create zero-context array
    T = q.gamma.shape[0]
    ctx_zero = np.zeros((T, 0))

    new_regime = update_regime_transitions(
        q, ctx_zero, params.regime, n_steps=5, lr=1e-2
    )
    assert np.all(np.isfinite(new_regime.bias))
    assert np.all(np.isfinite(new_regime.init))


def test_mstep_cross_scale_regime_coupling(m2_ds, m2_q_coarse):
    """Provide parent_gamma to hit cross-scale coupling (lines 154, 162-170)."""
    K = m2_ds.data.K
    params = m2_ds.params[K - 1]
    ctx = m2_ds.data.get_C(K)
    T = m2_q_coarse.gamma.shape[0]
    M = params.M

    # Create a synthetic parent_gamma (soft assignment over M_parent regimes)
    rng = np.random.RandomState(95)
    parent_gamma_raw = rng.dirichlet(np.ones(M), size=T)  # (T, M)

    new_regime = update_regime_transitions(
        m2_q_coarse,
        ctx,
        params.regime,
        parent_gamma=parent_gamma_raw,
        n_steps=10,
        lr=1e-2,
    )

    assert np.all(np.isfinite(new_regime.bias)), "bias must be finite with parent_gamma"
    assert np.all(
        np.isfinite(new_regime.trans)
    ), "trans must be finite with parent_gamma"


def test_mstep_alloc_with_lagged_parents(mini_ds, mini_qs):
    """Inject parent_sets to hit lagged allocation lines 670-673, 734-737."""
    _, q_fine = mini_qs
    k = 1
    params_fine = deepcopy(mini_ds.params[k - 1])
    # Inject a lagged parent edge: label 0 is a lag-1 parent of label 1
    params_fine.parent_sets[1] = {(0, 1)}

    X_k = mini_ds.data.get_X(k)
    Y_k = mini_ds.data.get_Y(k)
    X_par = mini_ds.data.get_X(k + 1)
    Y_par = mini_ds.data.get_Y(k + 1)
    ctx = mini_ds.data.get_C(k)

    new_fp, new_fdp = update_allocation_scores(
        q_fine,
        X_k,
        Y_k,
        X_par,
        Y_par,
        ctx,
        params_fine,
        mini_ds.hierarchy,
        k=k,
        n_steps=5,
    )

    assert np.all(np.isfinite(new_fp.beta0)), "Fine count beta0 must be finite"
    assert np.all(np.isfinite(new_fdp.alpha0)), "Fine dur alpha0 must be finite"


# ===========================================================================
# sstep.py coverage
# ===========================================================================


def test_sstep_beam_search_dispatch(mini_ds, mini_graph, mini_qs):
    """SStep(search='beam') dispatches to BeamSearch (lines 472-477)."""
    q_coarse, _ = mini_qs
    K = mini_ds.data.K

    sstep = SStep(
        lam=1.0,
        max_indegree=1,
        max_lag=1,
        search="beam",
        beam_width=3,
    )
    new_graph, new_params = sstep.run(
        k=K,
        q=q_coarse,
        params=mini_ds.params[K - 1],
        graph=mini_graph,
        data=mini_ds.data,
    )

    assert isinstance(new_graph, MultiScaleDBNGraph)
    # Parent sets should respect max_indegree
    L = new_params.L
    for ell in range(L):
        ps = new_params.parent_sets.get(ell, set())
        assert len(ps) <= 1, f"Label {ell} has {len(ps)} parents, max_indegree=1"


def test_sstep_fine_lagged_scoring(mini_ds, mini_graph, mini_qs):
    """Fine-scale scoring with a lagged parent set (lines 281-283 of sstep.py)."""
    _, q_fine = mini_qs
    k = 1
    params_fine = mini_ds.params[k - 1]
    X_k = mini_ds.data.get_X(k)
    Y_k = mini_ds.data.get_Y(k)
    X_par = mini_ds.data.get_X(k + 1)
    Y_par = mini_ds.data.get_Y(k + 1)
    ctx = mini_ds.data.get_C(k)

    # Score with a lagged parent
    parent_set_with_lag = {(0, 1)}
    score_lag = LocalScore.compute_fine(
        parent_set_with_lag,
        1,
        X_k,
        Y_k,
        X_par,
        Y_par,
        q_fine,
        ctx,
        params_fine,
        mini_ds.hierarchy,
        k,
        lam=1.0,
    )

    # Score without parents
    score_empty = LocalScore.compute_fine(
        set(),
        1,
        X_k,
        Y_k,
        X_par,
        Y_par,
        q_fine,
        ctx,
        params_fine,
        mini_ds.hierarchy,
        k,
        lam=1.0,
    )

    assert np.isfinite(score_lag), f"Fine-scale lagged score not finite: {score_lag}"
    assert np.isfinite(score_empty), f"Fine-scale empty score not finite: {score_empty}"
    # The lagged score includes a penalty, so it should differ from empty
    assert score_lag != score_empty


# ===========================================================================
# graph.py coverage
# ===========================================================================


def test_graph_set_parents(mini_ds, mini_gen):
    """set_parents() replaces parent set for (k, ell) - line 110."""
    graph = MultiScaleDBNGraph(
        hierarchy=mini_ds.hierarchy,
        label_names=mini_gen.label_names,
    )
    assert graph.get_parents(1, 0) == set()

    new_parents = {(1, 1), (0, 2)}
    graph.set_parents(1, 0, new_parents)
    assert graph.get_parents(1, 0) == new_parents


def test_graph_total_edges(mini_ds, mini_gen):
    """n_edges() without scale arg sums across all scales - line 124."""
    graph = MultiScaleDBNGraph(
        hierarchy=mini_ds.hierarchy,
        label_names=mini_gen.label_names,
    )
    assert graph.n_edges() == 0

    graph.add_edge(k=1, src_ell=0, lag=1, dst_ell=1)
    graph.add_edge(k=2, src_ell=1, lag=1, dst_ell=0)
    assert graph.n_edges() == 2
    assert graph.n_edges(k=1) == 1
    assert graph.n_edges(k=2) == 1


def test_graph_is_acyclic(mini_ds, mini_gen):
    """is_acyclic() returns True for lagged-edge graphs - line 145."""
    graph = MultiScaleDBNGraph(
        hierarchy=mini_ds.hierarchy,
        label_names=mini_gen.label_names,
    )
    assert graph.is_acyclic() is True

    # Still True even with edges
    graph.add_edge(k=1, src_ell=0, lag=1, dst_ell=0)  # self-loop with lag
    graph.add_edge(k=1, src_ell=0, lag=1, dst_ell=1)
    assert graph.is_acyclic() is True


def test_graph_copy_and_repr(mini_ds, mini_gen):
    """copy() deep-copies and __repr__() returns informative string - lines 214, 224-225."""
    graph = MultiScaleDBNGraph(
        hierarchy=mini_ds.hierarchy,
        label_names=mini_gen.label_names,
    )
    graph.add_edge(k=1, src_ell=0, lag=1, dst_ell=1)

    # Test copy
    g2 = graph.copy()
    assert g2.n_edges() == 1
    assert g2.get_parents(1, 1) == {(0, 1)}
    # Verify deep copy: mutating g2 doesn't affect original
    g2.add_edge(k=1, src_ell=1, lag=1, dst_ell=0)
    assert g2.n_edges() == 2
    assert graph.n_edges() == 1  # original unchanged

    # Test __repr__
    r = repr(graph)
    assert "MultiScaleDBNGraph" in r
    assert "K=2" in r
    assert "L=2" in r
    assert "total_edges=1" in r
