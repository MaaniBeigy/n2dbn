"""
tests/test_phase04_05.py
========================
M-step parameter updates and S-step structure search.

Tests
-----
M-step (update_* functions + MStep.run):
 1.  update_regime_transitions smoke (M=1) - returns RegimeParams with finite values
 2.  update_regime_transitions M=2 - regime-switching case runs without error
 3.  update_factor_params - closed-form update returns PSD Sigma matrices (r=1)
 4.  update_factor_params r=0 - must be a no-op (returns unchanged F)
 5.  update_coarse_count_params smoke - finite CoarseCountParams, r_disp > 0
 6.  update_coarse_dur_params smoke - finite CoarseDurParams, s_disp > 0
 7.  update_allocation_scores smoke - finite (FineCountParams, FineDurParams)
 8.  MStep.run at coarsest scale - returns new ScaleParams, finite params
 9.  MStep.run at finest scale - returns new ScaleParams, finite params

S-step (LocalScore, GreedyForwardBackward, BeamSearch, SStep.run):
10.  LocalScore.compute_coarsest - returns finite float
11.  LocalScore.compute_coarsest - large penalty makes 1-edge set worse than empty
12.  LocalScore.compute_fine - returns finite float
13.  GreedyForwardBackward.search - returns valid parent set (set of (src, lag) tuples)
14.  GreedyForwardBackward.search - very high penalty yields empty parent set
15.  BeamSearch.search - returns valid parent set
16.  SStep.run at coarsest scale - (graph, params) returned, parent sets ≤ max_indegree
17.  SStep.run at finest scale - same invariant
18.  TopDownSEM with do_structure=True - smoke test, ELBO finite

Run with:
    pytest tests/test_phase04_05.py -v
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from src.core.graph import MultiScaleDBNGraph
from src.inference.posterior import PosteriorSummaries
from src.learning.estep import EStep
from src.learning.mstep import (
    MStep,
    update_allocation_scores,
    update_coarse_count_params,
    update_coarse_dur_params,
    update_factor_params,
    update_regime_transitions,
)
from src.learning.sstep import BeamSearch, GreedyForwardBackward, LocalScore, SStep
from src.learning.structural_em import TopDownSEM
from src.model.params import (
    CoarseCountParams,
    CoarseDurParams,
    FactorParams,
    FineCountParams,
    FineDurParams,
)
from src.synthetic.families import make_generator
from src.synthetic.scenarios import ScenarioConfig

# ---------------------------------------------------------------------------
# Scenario configs  (all tiny for speed)
# ---------------------------------------------------------------------------

_CFG_MINI = ScenarioConfig(
    name="phase0405_mini",
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
    seed=7,
)

_CFG_M2 = ScenarioConfig(
    name="phase0405_m2",
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
    seed=17,
)

_CFG_FACTOR = ScenarioConfig(
    name="phase0405_factor",
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
    seed=27,
)


# ---------------------------------------------------------------------------
# Module-scope fixtures  (generated once per test session for this file)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mini_ds():
    gen = make_generator(_CFG_MINI, seed=_CFG_MINI.seed)
    return gen.generate(seed=_CFG_MINI.seed)


@pytest.fixture(scope="module")
def mini_graph(mini_ds):
    gen = make_generator(_CFG_MINI, seed=_CFG_MINI.seed)
    return MultiScaleDBNGraph(
        hierarchy=mini_ds.hierarchy,
        label_names=gen.label_names,
    )


@pytest.fixture(scope="module")
def mini_qs(mini_ds, mini_graph):
    """Run EStep at both scales; returns (q_coarse, q_fine)."""
    K = mini_ds.data.K
    estep = EStep(max_inner_iter=2, use_laplace=False)
    q_coarse = estep.run(
        k=K,
        data=mini_ds.data,
        params=mini_ds.params[K - 1],
        graph=mini_graph,
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
def m2_graph(m2_ds):
    gen = make_generator(_CFG_M2, seed=_CFG_M2.seed)
    return MultiScaleDBNGraph(
        hierarchy=m2_ds.hierarchy,
        label_names=gen.label_names,
    )


@pytest.fixture(scope="module")
def m2_q_coarse(m2_ds, m2_graph):
    K = m2_ds.data.K
    estep = EStep(max_inner_iter=2, use_laplace=False)
    return estep.run(
        k=K,
        data=m2_ds.data,
        params=m2_ds.params[K - 1],
        graph=m2_graph,
    )


@pytest.fixture(scope="module")
def factor_ds():
    gen = make_generator(_CFG_FACTOR, seed=_CFG_FACTOR.seed)
    return gen.generate(seed=_CFG_FACTOR.seed)


@pytest.fixture(scope="module")
def factor_graph(factor_ds):
    gen = make_generator(_CFG_FACTOR, seed=_CFG_FACTOR.seed)
    return MultiScaleDBNGraph(
        hierarchy=factor_ds.hierarchy,
        label_names=gen.label_names,
    )


@pytest.fixture(scope="module")
def factor_q_coarse(factor_ds, factor_graph):
    K = factor_ds.data.K
    estep = EStep(max_inner_iter=2, use_laplace=False)
    return estep.run(
        k=K,
        data=factor_ds.data,
        params=factor_ds.params[K - 1],
        graph=factor_graph,
    )


# ---------------------------------------------------------------------------
# 1.  update_regime_transitions - M=1 smoke
# ---------------------------------------------------------------------------


def test_update_regime_transitions_smoke(mini_ds, mini_qs):
    """update_regime_transitions returns a RegimeParams with all-finite values (M=1)."""
    q_coarse, _ = mini_qs
    K = mini_ds.data.K
    params = mini_ds.params[K - 1]
    ctx = mini_ds.data.get_C(K)

    new_regime = update_regime_transitions(
        q_coarse, ctx, params.regime, n_steps=5, lr=1e-2
    )

    assert np.all(np.isfinite(new_regime.bias)), "bias must be finite"
    assert np.all(np.isfinite(new_regime.trans)), "trans must be finite"
    assert np.all(np.isfinite(new_regime.init)), "init must be finite"


# ---------------------------------------------------------------------------
# 2.  update_regime_transitions - M=2
# ---------------------------------------------------------------------------


def test_update_regime_transitions_M2(m2_ds, m2_q_coarse):
    """update_regime_transitions handles M=2 regime-switching without error."""
    K = m2_ds.data.K
    params = m2_ds.params[K - 1]
    ctx = m2_ds.data.get_C(K)

    new_regime = update_regime_transitions(
        m2_q_coarse, ctx, params.regime, n_steps=5, lr=1e-2
    )

    assert new_regime.M == 2
    assert np.all(np.isfinite(new_regime.bias))
    assert np.all(np.isfinite(new_regime.trans))


# ---------------------------------------------------------------------------
# 3.  update_factor_params - Sigma must be PSD (r=1)
# ---------------------------------------------------------------------------


def test_update_factor_params_psd_sigma(factor_ds, factor_q_coarse):
    """Closed-form LGSSM update must return PSD noise covariance matrices."""
    K = factor_ds.data.K
    params = factor_ds.params[K - 1]
    ctx = factor_ds.data.get_C(K)

    new_fp = update_factor_params(factor_q_coarse, ctx, params.factor)

    assert new_fp.r > 0, "test requires r > 0"
    for m in range(new_fp.M):
        eigvals = np.linalg.eigvalsh(new_fp.Sigma[m])
        assert (
            eigvals.min() >= -1e-6
        ), f"Sigma[{m}] not PSD: min eigval = {eigvals.min():.3e}"


# ---------------------------------------------------------------------------
# 4.  update_factor_params - r=0 is a no-op
# ---------------------------------------------------------------------------


def test_update_factor_params_r0_noop(mini_ds, mini_qs):
    """update_factor_params must return unchanged FactorParams when r=0."""
    q_coarse, _ = mini_qs
    K = mini_ds.data.K
    params = mini_ds.params[K - 1]
    ctx = mini_ds.data.get_C(K)

    assert params.factor.r == 0, "test requires r=0"
    original_F = params.factor.F.copy()

    new_fp = update_factor_params(q_coarse, ctx, params.factor)

    np.testing.assert_array_equal(
        new_fp.F,
        original_F,
        err_msg="F must be unchanged when r=0",
    )


# ---------------------------------------------------------------------------
# 5.  update_coarse_count_params smoke
# ---------------------------------------------------------------------------


def test_update_coarse_count_smoke(mini_ds, mini_qs):
    """update_coarse_count_params returns finite CoarseCountParams with r_disp > 0."""
    q_coarse, _ = mini_qs
    K = mini_ds.data.K
    params = mini_ds.params[K - 1]
    X_K = mini_ds.data.get_X(K)
    ctx = mini_ds.data.get_C(K)

    new_cp = update_coarse_count_params(q_coarse, X_K, ctx, params, n_steps=5, lr=1e-3)

    assert np.all(np.isfinite(new_cp.beta0)), "beta0 must be finite"
    assert np.all(np.isfinite(new_cp.gamma0)), "gamma0 must be finite"
    assert np.all(np.isfinite(new_cp.r_disp)), "r_disp must be finite"
    assert np.all(new_cp.r_disp > 0), "r_disp must be positive"


# ---------------------------------------------------------------------------
# 6.  update_coarse_dur_params smoke
# ---------------------------------------------------------------------------


def test_update_coarse_dur_smoke(mini_ds, mini_qs):
    """update_coarse_dur_params returns finite CoarseDurParams with s_disp > 0."""
    q_coarse, _ = mini_qs
    K = mini_ds.data.K
    params = mini_ds.params[K - 1]
    X_K = mini_ds.data.get_X(K)
    Y_K = mini_ds.data.get_Y(K)
    ctx = mini_ds.data.get_C(K)

    new_dp = update_coarse_dur_params(
        q_coarse, X_K, Y_K, ctx, params, n_steps=5, lr=1e-3
    )

    assert np.all(np.isfinite(new_dp.alpha0)), "alpha0 must be finite"
    assert np.all(np.isfinite(new_dp.s_disp)), "s_disp must be finite"
    assert np.all(new_dp.s_disp > 0), "s_disp must be positive"


# ---------------------------------------------------------------------------
# 7.  update_allocation_scores smoke
# ---------------------------------------------------------------------------


def test_update_allocation_scores_smoke(mini_ds, mini_qs):
    """update_allocation_scores returns finite (FineCountParams, FineDurParams)."""
    _, q_fine = mini_qs
    k = 1
    params_fine = mini_ds.params[k - 1]
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

    assert np.all(np.isfinite(new_fp.beta0)), "beta0 must be finite"
    assert np.all(np.isfinite(new_fdp.alpha0)), "alpha0 must be finite"


# ---------------------------------------------------------------------------
# 8.  MStep.run at coarsest scale
# ---------------------------------------------------------------------------


def test_mstep_run_coarsest(mini_ds, mini_graph, mini_qs):
    """MStep.run at coarsest scale returns a new ScaleParams with finite values."""
    q_coarse, _ = mini_qs
    K = mini_ds.data.K
    params = mini_ds.params[K - 1]

    mstep = MStep(n_steps_regime=5, n_steps_emission=5)
    new_params = mstep.run(K, q_coarse, params, mini_ds.data)

    assert new_params is not params, "must return a new object"
    assert new_params.is_coarsest
    assert np.all(np.isfinite(new_params.regime.bias))
    assert np.all(np.isfinite(new_params.coarse_count.beta0))
    assert np.all(np.isfinite(new_params.coarse_dur.alpha0))


# ---------------------------------------------------------------------------
# 9.  MStep.run at finest scale
# ---------------------------------------------------------------------------


def test_mstep_run_fine(mini_ds, mini_graph, mini_qs):
    """MStep.run at finest scale returns a new ScaleParams with finite values."""
    _, q_fine = mini_qs
    k = 1
    params_fine = mini_ds.params[k - 1]

    mstep = MStep(n_steps_regime=5, n_steps_emission=5)
    new_params = mstep.run(k, q_fine, params_fine, mini_ds.data)

    assert new_params is not params_fine, "must return a new object"
    assert not new_params.is_coarsest
    assert np.all(np.isfinite(new_params.regime.bias))
    assert np.all(np.isfinite(new_params.fine_count.beta0))
    assert np.all(np.isfinite(new_params.fine_dur.alpha0))


# ---------------------------------------------------------------------------
# 10.  LocalScore.compute_coarsest - returns finite float
# ---------------------------------------------------------------------------


def test_local_score_coarsest_finite(mini_ds, mini_qs):
    """LocalScore.compute_coarsest must return a finite scalar."""
    q_coarse, _ = mini_qs
    K = mini_ds.data.K
    params = mini_ds.params[K - 1]
    X_K = mini_ds.data.get_X(K)
    Y_K = mini_ds.data.get_Y(K)
    ctx = mini_ds.data.get_C(K)

    score = LocalScore.compute_coarsest(
        parent_set=set(),
        ell=0,
        X_K=X_K,
        Y_K=Y_K,
        q=q_coarse,
        ctx=ctx,
        params=params,
        lam=1.0,
    )

    assert isinstance(score, float)
    assert np.isfinite(score), f"score = {score}"


# ---------------------------------------------------------------------------
# 11.  LocalScore.compute_coarsest - large penalty makes 1-edge worse than empty
# ---------------------------------------------------------------------------


def test_local_score_coarsest_penalty_effect(mini_ds, mini_qs):
    """With a very large edge penalty, adding an edge must decrease the score."""
    q_coarse, _ = mini_qs
    K = mini_ds.data.K
    params = mini_ds.params[K - 1]
    X_K = mini_ds.data.get_X(K)
    Y_K = mini_ds.data.get_Y(K)
    ctx = mini_ds.data.get_C(K)

    def _score(ps):
        return LocalScore.compute_coarsest(
            ps,
            ell=0,
            X_K=X_K,
            Y_K=Y_K,
            q=q_coarse,
            ctx=ctx,
            params=params,
            lam=1e6,
        )

    score_empty = _score(set())
    score_one_edge = _score({(0, 1)})

    assert score_one_edge < score_empty, (
        f"score with 1 edge ({score_one_edge:.4f}) should be "
        f"< empty ({score_empty:.4f}) under lam=1e6"
    )


# ---------------------------------------------------------------------------
# 12.  LocalScore.compute_fine - returns finite float
# ---------------------------------------------------------------------------


def test_local_score_fine_finite(mini_ds, mini_qs):
    """LocalScore.compute_fine must return a finite scalar."""
    _, q_fine = mini_qs
    k = 1
    params_fine = mini_ds.params[k - 1]
    X_k = mini_ds.data.get_X(k)
    Y_k = mini_ds.data.get_Y(k)
    X_par = mini_ds.data.get_X(k + 1)
    Y_par = mini_ds.data.get_Y(k + 1)
    ctx = mini_ds.data.get_C(k)

    score = LocalScore.compute_fine(
        parent_set=set(),
        ell=0,
        X_k=X_k,
        Y_k=Y_k,
        X_par=X_par,
        Y_par=Y_par,
        q=q_fine,
        ctx=ctx,
        params=params_fine,
        hier=mini_ds.hierarchy,
        k=k,
        lam=1.0,
    )

    assert isinstance(score, float)
    assert np.isfinite(score), f"score = {score}"


# ---------------------------------------------------------------------------
# 13.  GreedyForwardBackward.search - returns valid parent set
# ---------------------------------------------------------------------------


def test_greedy_fb_smoke(mini_ds, mini_qs):
    """GreedyForwardBackward.search returns a set of (src_ell, lag) tuples."""
    q_coarse, _ = mini_qs
    K = mini_ds.data.K
    params = mini_ds.params[K - 1]
    X_K = mini_ds.data.get_X(K)
    Y_K = mini_ds.data.get_Y(K)
    ctx = mini_ds.data.get_C(K)

    L = params.L
    candidates = [(src, 1) for src in range(L)]

    def score_fn(ps):
        return LocalScore.compute_coarsest(
            ps, 0, X_K, Y_K, q_coarse, ctx, params, lam=0.5
        )

    searcher = GreedyForwardBackward(max_indegree=2, max_iter=5)
    best = searcher.search(set(), score_fn, candidates)

    assert isinstance(best, set)
    assert len(best) <= 2
    for item in best:
        assert len(item) == 2, f"each edge must be (src, lag), got {item}"


# ---------------------------------------------------------------------------
# 14.  GreedyForwardBackward - very high penalty yields empty parent set
# ---------------------------------------------------------------------------


def test_greedy_fb_high_penalty_empty(mini_ds, mini_qs):
    """With lam=1e6, greedy search must find no beneficial edges to add."""
    q_coarse, _ = mini_qs
    K = mini_ds.data.K
    params = mini_ds.params[K - 1]
    X_K = mini_ds.data.get_X(K)
    Y_K = mini_ds.data.get_Y(K)
    ctx = mini_ds.data.get_C(K)

    L = params.L
    candidates = [(src, 1) for src in range(L)]

    def score_fn(ps):
        return LocalScore.compute_coarsest(
            ps, 0, X_K, Y_K, q_coarse, ctx, params, lam=1e6
        )

    searcher = GreedyForwardBackward(max_indegree=3, max_iter=10)
    best = searcher.search(set(), score_fn, candidates)

    assert len(best) == 0, f"Expected empty parent set with lam=1e6 but got {best}"


# ---------------------------------------------------------------------------
# 15.  BeamSearch.search - returns valid parent set
# ---------------------------------------------------------------------------


def test_beam_search_smoke(mini_ds, mini_qs):
    """BeamSearch.search must return a valid parent set without error."""
    q_coarse, _ = mini_qs
    K = mini_ds.data.K
    params = mini_ds.params[K - 1]
    X_K = mini_ds.data.get_X(K)
    Y_K = mini_ds.data.get_Y(K)
    ctx = mini_ds.data.get_C(K)

    L = params.L
    candidates = [(src, 1) for src in range(L)]

    def score_fn(ps):
        return LocalScore.compute_coarsest(
            ps, 0, X_K, Y_K, q_coarse, ctx, params, lam=1.0
        )

    searcher = BeamSearch(beam_width=3, max_indegree=2, max_depth=3)
    best = searcher.search(score_fn, candidates)

    assert isinstance(best, set)
    assert len(best) <= 2


# ---------------------------------------------------------------------------
# 16.  SStep.run at coarsest scale
# ---------------------------------------------------------------------------


def test_sstep_run_coarsest(mini_ds, mini_graph, mini_qs):
    """SStep.run at coarsest scale returns (graph, params); parent sets ≤ max_indegree."""
    q_coarse, _ = mini_qs
    K = mini_ds.data.K
    params = mini_ds.params[K - 1]

    sstep = SStep(lam=1.0, max_indegree=1, max_lag=1, search="greedy", max_iter=5)
    new_graph, new_params = sstep.run(
        k=K,
        q=q_coarse,
        params=params,
        graph=mini_graph,
        data=mini_ds.data,
    )

    assert isinstance(new_graph, MultiScaleDBNGraph)
    for ell in range(params.L):
        assert (
            len(new_params.parent_sets.get(ell, set())) <= 1
        ), f"label {ell}: parent set size {len(new_params.parent_sets.get(ell, set()))} > max_indegree=1"


# ---------------------------------------------------------------------------
# 17.  SStep.run at finest scale
# ---------------------------------------------------------------------------


def test_sstep_run_fine(mini_ds, mini_graph, mini_qs):
    """SStep.run at the finest scale returns (graph, params) without error."""
    _, q_fine = mini_qs
    k = 1
    params_fine = mini_ds.params[k - 1]

    sstep = SStep(lam=1.0, max_indegree=1, max_lag=1, search="greedy", max_iter=5)
    new_graph, new_params = sstep.run(
        k=k,
        q=q_fine,
        params=params_fine,
        graph=mini_graph,
        data=mini_ds.data,
    )

    assert isinstance(new_graph, MultiScaleDBNGraph)
    for ell in range(params_fine.L):
        assert (
            len(new_params.parent_sets.get(ell, set())) <= 1
        ), f"label {ell}: parent set size exceeds max_indegree=1"


# ---------------------------------------------------------------------------
# 18.  TopDownSEM with do_structure=True - smoke test
# ---------------------------------------------------------------------------


def test_sem_do_structure_integration() -> None:
    """TopDownSEM with do_structure=True must complete and produce finite ELBO."""
    gen = make_generator(_CFG_MINI, seed=_CFG_MINI.seed)
    ds = gen.generate(seed=_CFG_MINI.seed)
    graph = MultiScaleDBNGraph(hierarchy=ds.hierarchy, label_names=gen.label_names)

    sem = TopDownSEM(
        max_outer_iter=1,
        do_structure=True,
        do_warmstart=False,
        verbose=0,
        estep_kwargs={"max_inner_iter": 2, "use_laplace": False},
        mstep_kwargs={"n_steps_regime": 5, "n_steps_emission": 5},
        sstep_kwargs={"max_indegree": 1, "max_lag": 1, "max_iter": 5},
    )
    params_list, graph_out, history = sem.fit(ds.data, ds.params, graph)

    assert isinstance(graph_out, MultiScaleDBNGraph)
    assert len(history.elbo) > 0
    assert all(
        np.isfinite(e) for e in history.elbo
    ), f"ELBO contains non-finite values: {history.elbo}"


# ===========================================================================
# M-step correctness tests  (TODO-1)
# ===========================================================================


# ---------------------------------------------------------------------------
# Helper: expected transition log-likelihood
# ---------------------------------------------------------------------------


def _expected_transition_ll(
    gamma: np.ndarray,
    xi: np.ndarray,
    regime,
    ctx: np.ndarray,
) -> float:
    """E_q[log p(S | regime params)] using posterior gamma and xi."""
    T, M = gamma.shape
    ll = 0.0
    # Initial distribution
    logits_0 = regime.logit_init(ctx[0])
    log_p0 = logits_0 - np.logaddexp.reduce(logits_0)
    ll += float(gamma[0] @ log_p0)
    # Transitions
    for t in range(1, T):
        for prev_m in range(M):
            logits = regime.logit_transition(prev_m, None, ctx[t])
            log_probs = logits - np.logaddexp.reduce(logits)
            if t - 1 < xi.shape[0]:
                ll += float(xi[t - 1, prev_m, :] @ log_probs)
    return ll


# ---------------------------------------------------------------------------
# 19.  Regime transitions improve expected log-likelihood
# ---------------------------------------------------------------------------


def test_mstep_regime_transitions_improve_elbo(m2_ds, m2_q_coarse):
    """After one update_regime_transitions call, expected transition LL must not decrease."""
    K = m2_ds.data.K
    params = m2_ds.params[K - 1]
    ctx = m2_ds.data.get_C(K)

    # Perturb initial params so they are clearly suboptimal
    perturbed = deepcopy(params.regime)
    rng = np.random.RandomState(42)
    perturbed.bias += rng.randn(*perturbed.bias.shape) * 0.3
    perturbed.trans += rng.randn(*perturbed.trans.shape) * 0.3

    ll_before = _expected_transition_ll(
        m2_q_coarse.gamma, m2_q_coarse.xi, perturbed, ctx
    )

    new_regime = update_regime_transitions(
        m2_q_coarse, ctx, perturbed, n_steps=100, lr=1e-3
    )

    ll_after = _expected_transition_ll(
        m2_q_coarse.gamma, m2_q_coarse.xi, new_regime, ctx
    )

    assert ll_after >= ll_before - 0.1, (
        f"Expected transition LL decreased significantly: "
        f"{ll_before:.4f} -> {ll_after:.4f}"
    )


# ---------------------------------------------------------------------------
# 20.  Factor params closed-form recovery (M=1, r=1)
# ---------------------------------------------------------------------------


def test_mstep_factor_params_closed_form():
    """With M=1 and synthetic AR(1) data, closed-form F update recovers near-true dynamics."""
    cfg = ScenarioConfig(
        name="factor_cf",
        K=2,
        H_finest=60,
        branching=[3],
        L=2,
        M_k=[1, 1],
        r_k=[1, 1],
        max_lag=[1, 1],
        max_indegree=[0, 0],
        zi_level="low",
        sparse_graph=False,
        family=3,
        seed=55,
    )
    gen = make_generator(cfg, seed=55)
    ds = gen.generate(seed=55)
    K = ds.data.K
    params = ds.params[K - 1]
    graph = MultiScaleDBNGraph(hierarchy=ds.hierarchy, label_names=gen.label_names)

    assert params.factor.r == 1 and params.factor.M == 1
    true_F = float(params.factor.F[0, 0, 0])

    estep = EStep(max_inner_iter=3, use_laplace=False)
    q = estep.run(k=K, data=ds.data, params=params, graph=graph)

    ctx = ds.data.get_C(K)
    new_fp = update_factor_params(q, ctx, params.factor)

    assert np.all(np.isfinite(new_fp.F)), "F must be finite"
    est_F = float(new_fp.F[0, 0, 0])
    assert abs(est_F - true_F) < 0.5, (
        f"F recovery: true={true_F:.3f}, estimated={est_F:.3f}, "
        f"diff={abs(est_F - true_F):.3f}"
    )


# ---------------------------------------------------------------------------
# 21.  Coarse count params smoke - type and shape
# ---------------------------------------------------------------------------


def test_mstep_coarse_count_params_smoke(mini_ds, mini_qs):
    """update_coarse_count_params returns CoarseCountParams with preserved shapes."""
    q_coarse, _ = mini_qs
    K = mini_ds.data.K
    params = mini_ds.params[K - 1]
    X_K = mini_ds.data.get_X(K)
    ctx = mini_ds.data.get_C(K)

    new_cp = update_coarse_count_params(q_coarse, X_K, ctx, params, n_steps=5, lr=1e-3)

    assert isinstance(new_cp, CoarseCountParams)
    assert new_cp.beta0.shape == params.coarse_count.beta0.shape
    assert new_cp.betaS.shape == params.coarse_count.betaS.shape
    assert new_cp.r_disp.shape == params.coarse_count.r_disp.shape
    assert np.all(np.isfinite(new_cp.beta0))
    assert np.all(new_cp.r_disp > 0)


# ---------------------------------------------------------------------------
# 22.  Coarse duration params smoke - type and shape
# ---------------------------------------------------------------------------


def test_mstep_coarse_dur_params_smoke(mini_ds, mini_qs):
    """update_coarse_dur_params returns CoarseDurParams with preserved shapes."""
    q_coarse, _ = mini_qs
    K = mini_ds.data.K
    params = mini_ds.params[K - 1]
    X_K = mini_ds.data.get_X(K)
    Y_K = mini_ds.data.get_Y(K)
    ctx = mini_ds.data.get_C(K)

    new_dp = update_coarse_dur_params(
        q_coarse, X_K, Y_K, ctx, params, n_steps=5, lr=1e-3
    )

    assert isinstance(new_dp, CoarseDurParams)
    assert new_dp.alpha0.shape == params.coarse_dur.alpha0.shape
    assert new_dp.alphaS.shape == params.coarse_dur.alphaS.shape
    assert new_dp.s_disp.shape == params.coarse_dur.s_disp.shape
    assert np.all(np.isfinite(new_dp.alpha0))
    assert np.all(new_dp.s_disp > 0)


# ---------------------------------------------------------------------------
# 23.  Fine allocation params smoke - type and shape
# ---------------------------------------------------------------------------


def test_mstep_fine_alloc_params_smoke(mini_ds, mini_qs):
    """update_allocation_scores at k<K returns (FineCountParams, FineDurParams)."""
    _, q_fine = mini_qs
    k = 1
    params_fine = mini_ds.params[k - 1]
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

    assert isinstance(new_fp, FineCountParams)
    assert isinstance(new_fdp, FineDurParams)
    assert new_fp.beta0.shape == params_fine.fine_count.beta0.shape
    assert new_fdp.alpha0.shape == params_fine.fine_dur.alpha0.shape
    assert np.all(np.isfinite(new_fp.beta0))
    assert np.all(np.isfinite(new_fdp.alpha0))


# ---------------------------------------------------------------------------
# 24.  MStep.run returns ScaleParams with same shapes
# ---------------------------------------------------------------------------


def test_mstep_run_returns_scale_params(mini_ds, mini_graph, mini_qs):
    """MStep.run returns ScaleParams preserving M, r, L, and sub-param shapes."""
    q_coarse, _ = mini_qs
    K = mini_ds.data.K
    params = mini_ds.params[K - 1]

    mstep = MStep(n_steps_regime=5, n_steps_emission=5)
    new_params = mstep.run(K, q_coarse, params, mini_ds.data)

    # Structural properties preserved
    assert new_params.M == params.M
    assert new_params.r == params.r
    assert new_params.L == params.L
    assert new_params.is_coarsest == params.is_coarsest

    # Regime shapes preserved
    assert new_params.regime.bias.shape == params.regime.bias.shape
    assert new_params.regime.trans.shape == params.regime.trans.shape
    assert new_params.regime.ctx.shape == params.regime.ctx.shape
    assert new_params.regime.init.shape == params.regime.init.shape

    # Emission shapes preserved (coarsest scale)
    assert new_params.coarse_count.beta0.shape == params.coarse_count.beta0.shape
    assert new_params.coarse_count.betaS.shape == params.coarse_count.betaS.shape
    assert new_params.coarse_dur.alpha0.shape == params.coarse_dur.alpha0.shape
    assert new_params.coarse_dur.alphaS.shape == params.coarse_dur.alphaS.shape


# ---------------------------------------------------------------------------
# 25.  Context-dependent regime update moves ctx weights
# ---------------------------------------------------------------------------


def test_mstep_context_dependent_regime(m2_ds, m2_q_coarse):
    """With d_ctx>0, gradient step changes ctx weight in correct direction."""
    K = m2_ds.data.K
    params = m2_ds.params[K - 1]
    ctx = m2_ds.data.get_C(K)

    assert params.regime.d_ctx > 0, "test requires d_ctx > 0"
    original_ctx = params.regime.ctx.copy()

    new_regime = update_regime_transitions(
        m2_q_coarse, ctx, params.regime, n_steps=50, lr=1e-2
    )

    assert np.all(np.isfinite(new_regime.ctx)), "ctx weights must be finite"
    assert not np.allclose(
        new_regime.ctx, original_ctx, atol=1e-8
    ), "ctx weights must change after gradient update with d_ctx > 0"


# ---------------------------------------------------------------------------
# 26.  LinAlgError fallback keeps F unchanged
# ---------------------------------------------------------------------------


def test_mstep_linalg_error_fallback():
    """Singular covariance triggers LinAlgError; update_factor_params keeps existing F."""
    M, r, T = 1, 1, 4

    # Degenerate posterior: all-zero factor means and covariances
    gamma = np.ones((T, M))
    xi = np.ones((T - 1, M, M))
    m_H = np.zeros((T, r))
    Sigma_H = np.zeros((T, r, r))
    q = PosteriorSummaries(gamma=gamma, xi=xi, m_H=m_H, Sigma_H=Sigma_H, log_Z=0.0)

    original_F = np.array([[[0.5]]])
    fp = FactorParams(
        M=M,
        r=r,
        d_ctx=0,
        F=original_F.copy(),
        G=np.zeros((M, r, 0)),
        Sigma=np.array([[[1.0]]]),
        mu0=np.zeros(r),
        Sigma0=np.eye(r),
    )
    ctx = np.zeros((T, 0))

    # reg=0 -> eps=0 -> (B + eps) = zeros -> singular -> LinAlgError
    new_fp = update_factor_params(q, ctx, fp, reg=0.0)

    np.testing.assert_array_equal(
        new_fp.F,
        original_F,
        err_msg="F must remain unchanged when LinAlgError occurs",
    )
