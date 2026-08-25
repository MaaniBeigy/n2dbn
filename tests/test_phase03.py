"""
tests/test_phase03.py
=====================
Posterior inference, the E-step.

Tests
-----
1.  RegimePosterior.forward_backward
        - γ rows sum to 1 (probability simplex at each t)
        - ξ slices sum to 1 (joint marginals at each t)
        - log_Z is finite
        - T=1 edge-case handled
2.  RegimePosterior.build_log_trans
        - exp(log_trans[t]) columns sum to 1
3.  FactorPosterior.smoother
        - returns (T, r) mean and (T, r, r) covariances
        - covariances are positive semi-definite
        - r=0 returns empty arrays
4.  PosteriorSummaries helpers
        - expected_hh() is PSD
        - elbo_entropy() ≥ 0
5.  EStep.run (integration test on minimal Family-I dataset)
        - returns PosteriorSummaries with correct shapes
        - γ rows sum to 1
        - log_Z is finite

Run with:
    pytest tests/test_phase03.py -v
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from src.core.graph import MultiScaleDBNGraph
from src.inference.posterior import (
    FactorPosterior,
    PosteriorSummaries,
    RegimePosterior,
)
from src.learning.estep import EStep
from src.model.params import FactorParams, RegimeParams
from src.synthetic.families import make_generator
from src.synthetic.scenarios import ScenarioConfig

# ---------------------------------------------------------------------------
# Minimal dataset fixture  (module scope = generated once for the whole file)
# ---------------------------------------------------------------------------

_MINI_CFG = ScenarioConfig(
    name="phase03_mini",
    K=2,
    H_finest=12,
    branching=[3],
    L=2,
    M_k=[1, 1],  # stationary (Family I)
    r_k=[0, 0],
    max_lag=[1, 1],
    max_indegree=[1, 1],
    zi_level="medium",
    sparse_graph=False,
    family=1,
    seed=42,
)


@pytest.fixture(scope="module")
def mini_dataset():
    gen = make_generator(_MINI_CFG, seed=42)
    return gen


@pytest.fixture(scope="module")
def mini_ds(mini_dataset):
    return mini_dataset.generate(seed=42)


@pytest.fixture(scope="module")
def mini_graph(mini_dataset, mini_ds):
    return MultiScaleDBNGraph(
        hierarchy=mini_ds.hierarchy,
        label_names=mini_dataset.label_names,
    )


# ---------------------------------------------------------------------------
# Helpers: build minimal params without the families factory
# ---------------------------------------------------------------------------


def _regime_params(M: int, d_ctx: int = 0) -> RegimeParams:
    rng = np.random.default_rng(0)
    return RegimeParams(
        M=M,
        parent_M=0,
        d_ctx=d_ctx,
        init=rng.standard_normal(M),
        bias=rng.standard_normal(M),
        trans=0.5 * rng.standard_normal((M, M)),
        cross=None,
        ctx=np.zeros((M, d_ctx)),
    )


def _factor_params(M: int = 2, r: int = 2) -> FactorParams:
    F = np.tile(0.8 * np.eye(r), (M, 1, 1))
    G = np.zeros((M, r, 0))
    Sigma = np.tile(0.1 * np.eye(r), (M, 1, 1))
    return FactorParams(
        M=M,
        r=r,
        d_ctx=0,
        F=F,
        G=G,
        Sigma=Sigma,
        mu0=np.zeros(r),
        Sigma0=np.eye(r),
    )


# ---------------------------------------------------------------------------
# 1.  RegimePosterior.forward_backward
# ---------------------------------------------------------------------------


class TestForwardBackward:
    """Mathematical properties of the exact HMM forward-backward."""

    @pytest.fixture
    def fb_inputs(self):
        """Return (log_like, log_init, log_trans, T, M) for a standard setup."""
        T, M = 15, 3
        p = _regime_params(M)
        ctx = np.zeros((T, 0))
        log_init, log_trans = RegimePosterior.build_log_trans(p, ctx)
        log_like = np.zeros((T, M))  # uninformative likelihoods
        return log_like, log_init, log_trans, T, M

    def test_gamma_sums_to_one(self, fb_inputs):
        log_like, log_init, log_trans, T, M = fb_inputs
        gamma, xi, log_Z = RegimePosterior.forward_backward(
            log_like, log_init, log_trans
        )
        np.testing.assert_allclose(
            gamma.sum(axis=1),
            np.ones(T),
            atol=1e-8,
            err_msg="γ rows must sum to 1 (probability simplex)",
        )

    def test_xi_slices_sum_to_one(self, fb_inputs):
        log_like, log_init, log_trans, T, M = fb_inputs
        gamma, xi, log_Z = RegimePosterior.forward_backward(
            log_like, log_init, log_trans
        )
        # xi[t] is (M, M) joint marginal over (S_t, S_{t+1})
        xi_sums = xi.sum(axis=(1, 2))
        np.testing.assert_allclose(
            xi_sums,
            np.ones(T - 1),
            atol=1e-8,
            err_msg="ξ slices must sum to 1 (joint marginals)",
        )

    def test_log_Z_is_finite(self, fb_inputs):
        log_like, log_init, log_trans, T, M = fb_inputs
        _, _, log_Z = RegimePosterior.forward_backward(log_like, log_init, log_trans)
        assert np.isfinite(log_Z), f"log_Z = {log_Z} is not finite"

    def test_T1_edge_case(self):
        """T=1: no xi (length-0 dimension), single gamma row."""
        M = 2
        p = _regime_params(M)
        ctx = np.zeros((1, 0))
        log_init, log_trans = RegimePosterior.build_log_trans(p, ctx)
        log_like = np.zeros((1, M))

        gamma, xi, log_Z = RegimePosterior.forward_backward(
            log_like, log_init, log_trans
        )
        assert gamma.shape == (1, M)
        np.testing.assert_allclose(gamma.sum(axis=1), np.ones(1), atol=1e-8)
        assert np.isfinite(log_Z)

    def test_informative_likelihood_shifts_gamma(self):
        """Strong emission evidence must concentrate γ on the signalled regime."""
        T, M = 10, 3
        p = _regime_params(M)
        ctx = np.zeros((T, 0))
        log_init, log_trans = RegimePosterior.build_log_trans(p, ctx)

        # Force all mass onto regime 0 via strong positive evidence
        log_like = np.full((T, M), -50.0)
        log_like[:, 0] = 0.0

        gamma, _, _ = RegimePosterior.forward_backward(log_like, log_init, log_trans)
        # γ_t(0) should be ≈ 1 for all t
        assert (
            gamma[:, 0].min() > 0.9
        ), f"Expected γ[:,0] ≈ 1 but min={gamma[:,0].min():.4f}"


# ---------------------------------------------------------------------------
# 2.  RegimePosterior.build_log_trans
# ---------------------------------------------------------------------------


def test_build_log_trans_columns_sum_to_one():
    """exp(log_trans[t]) must have each column summing to 1."""
    M, T = 3, 10
    p = _regime_params(M)
    ctx = np.zeros((T, 0))
    _, log_trans = RegimePosterior.build_log_trans(p, ctx)

    for t in range(T - 1):
        col_sums = np.exp(log_trans[t]).sum(axis=0)
        np.testing.assert_allclose(
            col_sums,
            np.ones(M),
            atol=1e-8,
            err_msg=f"exp(log_trans[{t}]) columns must sum to 1",
        )


# ---------------------------------------------------------------------------
# 3.  FactorPosterior.smoother
# ---------------------------------------------------------------------------


class TestFactorSmoother:
    """Shape and numerical properties of the Kalman smoother."""

    def test_shapes(self):
        T, M, r = 20, 2, 3
        gamma = np.ones((T, M)) / M
        fp = _factor_params(M=M, r=r)
        ctx = np.zeros((T, 0))
        m_H, Sigma_H = FactorPosterior.smoother(gamma, fp, ctx)
        assert m_H.shape == (T, r), f"mean shape {m_H.shape}"
        assert Sigma_H.shape == (T, r, r), f"cov shape {Sigma_H.shape}"

    def test_covariances_psd(self):
        """All smoothed covariances must be positive semi-definite."""
        T, M, r = 10, 2, 2
        gamma = np.ones((T, M)) / M
        fp = _factor_params(M=M, r=r)
        ctx = np.zeros((T, 0))
        _, Sigma_H = FactorPosterior.smoother(gamma, fp, ctx)
        for t in range(T):
            eigvals = np.linalg.eigvalsh(Sigma_H[t])
            assert (
                eigvals.min() >= -1e-6
            ), f"Sigma_H[{t}] not PSD: min eigval={eigvals.min():.3e}"

    def test_r0_returns_empty(self):
        """When r=0, smoother returns (T, 0) and (T, 0, 0)."""
        T, M = 15, 2
        gamma = np.ones((T, M)) / M
        fp = FactorParams(
            M=M,
            r=0,
            d_ctx=0,
            F=np.zeros((M, 0, 0)),
            G=np.zeros((M, 0, 0)),
            Sigma=np.zeros((M, 0, 0)),
            mu0=np.zeros(0),
            Sigma0=np.zeros((0, 0)),
        )
        ctx = np.zeros((T, 0))
        m_H, Sigma_H = FactorPosterior.smoother(gamma, fp, ctx)
        assert m_H.shape == (T, 0)
        assert Sigma_H.shape == (T, 0, 0)

    def test_with_pseudo_obs_runs(self):
        """Smoother must run without error when pseudo-observations are supplied."""
        T, M, r = 8, 2, 2
        rng = np.random.default_rng(7)
        gamma = rng.dirichlet(np.ones(M), size=T)
        fp = _factor_params(M=M, r=r)
        ctx = np.zeros((T, 0))

        pseudo_obs = rng.standard_normal((T, r))
        pseudo_R = np.tile(0.5 * np.eye(r), (T, 1, 1))

        m_H, Sigma_H = FactorPosterior.smoother(
            gamma, fp, ctx, pseudo_obs=pseudo_obs, pseudo_R=pseudo_R
        )
        assert m_H.shape == (T, r)
        assert np.all(np.isfinite(m_H))


# ---------------------------------------------------------------------------
# 4.  PosteriorSummaries helpers
# ---------------------------------------------------------------------------


def _make_summaries(T: int = 8, M: int = 2, r: int = 2) -> PosteriorSummaries:
    rng = np.random.default_rng(3)
    gamma = rng.dirichlet(np.ones(M), size=T)
    # Build a valid xi: (T-1, M, M), each slice summing to 1
    xi_raw = rng.dirichlet(np.ones(M * M), size=T - 1).reshape(T - 1, M, M)
    xi = xi_raw / xi_raw.sum(axis=(1, 2), keepdims=True)
    m_H = rng.standard_normal((T, r))
    Sigma_H = np.tile(np.eye(r), (T, 1, 1))
    return PosteriorSummaries(gamma=gamma, xi=xi, m_H=m_H, Sigma_H=Sigma_H, log_Z=-40.0)


def test_expected_hh_is_psd():
    """E[H_t H_t^T] = Σ_H[t] + m_H[t] m_H[t]^T must be PSD for all t."""
    ps = _make_summaries(T=5, M=2, r=3)
    EHH = ps.expected_hh()
    assert EHH.shape == (5, 3, 3)
    for t in range(5):
        eigvals = np.linalg.eigvalsh(EHH[t])
        assert (
            eigvals.min() >= -1e-10
        ), f"E[HH^T][{t}] not PSD: min eigval={eigvals.min():.3e}"


def test_elbo_entropy_nonneg():
    """Variational entropy H[q] must be ≥ 0."""
    ps = _make_summaries(T=10, M=3, r=2)
    ent = ps.elbo_entropy()
    assert ent >= -1e-8, f"Entropy = {ent:.6f} < 0"


def test_posterior_summaries_repr():
    """__repr__ must not raise and contain useful info."""
    ps = _make_summaries(T=6, M=2, r=1)
    r = repr(ps)
    assert "T=6" in r
    assert "M=2" in r


def test_s_bar_equals_gamma():
    """s_bar property must return the same array as gamma."""
    ps = _make_summaries(T=5, M=2, r=0)
    np.testing.assert_array_equal(ps.s_bar, ps.gamma)


# ---------------------------------------------------------------------------
# 5.  EStep.run - integration test on minimal Family-I dataset
# ---------------------------------------------------------------------------


class TestEStepRun:
    """Integration tests: EStep.run on a real (minimal) generated dataset."""

    def test_shapes_coarsest_scale(self, mini_ds, mini_graph):
        """EStep at the coarsest scale returns the right output shapes."""
        K = mini_ds.data.K
        estep = EStep(max_inner_iter=2, use_laplace=False)
        q = estep.run(
            k=K,
            data=mini_ds.data,
            params=mini_ds.params[K - 1],
            graph=mini_graph,
        )
        T = mini_ds.hierarchy.n_epochs(K)
        M = mini_ds.params[K - 1].M
        r = mini_ds.params[K - 1].r

        assert q.gamma.shape == (T, M), f"gamma shape {q.gamma.shape}"
        assert q.m_H.shape == (T, r), f"m_H shape {q.m_H.shape}"
        assert q.Sigma_H.shape == (T, r, r), f"Sigma_H shape {q.Sigma_H.shape}"

    def test_shapes_finest_scale(self, mini_ds, mini_graph):
        """EStep at the finest scale returns the right output shapes."""
        K = mini_ds.data.K
        # First run coarsest to get parent summaries
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
        T = mini_ds.hierarchy.n_epochs(1)
        M = mini_ds.params[0].M
        r = mini_ds.params[0].r

        assert q_fine.gamma.shape == (T, M)
        assert q_fine.m_H.shape == (T, r)

    def test_gamma_rows_sum_to_one(self, mini_ds, mini_graph):
        K = mini_ds.data.K
        estep = EStep(max_inner_iter=2, use_laplace=False)
        q = estep.run(
            k=K,
            data=mini_ds.data,
            params=mini_ds.params[K - 1],
            graph=mini_graph,
        )
        np.testing.assert_allclose(
            q.gamma.sum(axis=1), 1.0, atol=1e-8, err_msg="EStep γ rows must sum to 1"
        )

    def test_log_Z_is_finite(self, mini_ds, mini_graph):
        K = mini_ds.data.K
        estep = EStep(max_inner_iter=2, use_laplace=False)
        q = estep.run(
            k=K,
            data=mini_ds.data,
            params=mini_ds.params[K - 1],
            graph=mini_graph,
        )
        assert np.isfinite(q.log_Z), f"log_Z = {q.log_Z}"

    def test_repr_works(self, mini_ds, mini_graph):
        K = mini_ds.data.K
        estep = EStep(max_inner_iter=1, use_laplace=False)
        q = estep.run(
            k=K,
            data=mini_ds.data,
            params=mini_ds.params[K - 1],
            graph=mini_graph,
        )
        r_str = repr(q)
        assert "log_Z" in r_str


# ---------------------------------------------------------------------------
# TODO-4 gap-filling: EpisodeReconstructor
# ---------------------------------------------------------------------------


def test_episode_reconstruct_basic():
    """EpisodeReconstructor.reconstruct produces correct episode traces."""
    from src.inference.reconstruction import EpisodeReconstructor

    # T=5 slots, L=2 labels
    # Label 0: starts at t=0 (duration 3 to [0,2]) and t=4 (duration 1 to [4,4])
    # Label 1: starts at t=2 (duration 2 to [2,3])
    X1 = np.zeros((5, 2), dtype=int)
    Y1 = np.zeros((5, 2), dtype=int)
    X1[0, 0] = 1
    Y1[0, 0] = 3
    X1[4, 0] = 1
    Y1[4, 0] = 1
    X1[2, 1] = 1
    Y1[2, 1] = 2

    rec = EpisodeReconstructor(label_names=["A", "B"])
    traces = rec.reconstruct(X1, Y1)

    assert len(traces[0]) == 2
    assert traces[0][0].t_start == 0 and traces[0][0].t_end == 2
    assert traces[0][1].t_start == 4 and traces[0][1].t_end == 4
    assert len(traces[1]) == 1
    assert traces[1][0].t_start == 2 and traces[1][0].t_end == 3


def test_episode_exclusivity_projection():
    """_project_exclusive truncates overlapping episodes to enforce at-most-one."""
    from src.inference.reconstruction import EpisodeReconstructor

    # Two overlapping episodes for label 0: [0,5] and [3,8]
    X1 = np.zeros((10, 1), dtype=int)
    Y1 = np.zeros((10, 1), dtype=int)
    X1[0, 0] = 1
    Y1[0, 0] = 6  # episode [0, 5]
    X1[3, 0] = 1
    Y1[3, 0] = 6  # episode [3, 8]

    rec = EpisodeReconstructor(label_names=["A"], exclusivity_set={0})
    traces = rec.reconstruct(X1, Y1)
    eps = traces[0]

    assert len(eps) == 2, "Both episodes must be present after projection"
    # First must end before second starts (no overlap)
    assert (
        eps[0].t_end < eps[1].t_start
    ), f"Overlap remains: eps[0].t_end={eps[0].t_end} >= eps[1].t_start={eps[1].t_start}"


def test_episode_iou():
    """IoU = 1.0 when pred == true, IoU = 0.0 when disjoint."""
    from src.inference.reconstruction import EpisodeReconstructor

    H1 = 10
    X1 = np.zeros((H1, 1), dtype=int)
    Y1 = np.zeros((H1, 1), dtype=int)
    X1[2, 0] = 1
    Y1[2, 0] = 3  # episode [2, 4]

    rec = EpisodeReconstructor(label_names=["A"])
    traces = rec.reconstruct(X1, Y1)

    # Self-IoU must be 1.0
    ious_self = rec.iou(traces, traces, H1)
    assert ious_self[0] == pytest.approx(1.0), f"Self-IoU = {ious_self[0]}"

    # Disjoint episode [7, 8] to IoU = 0.0
    X1b = np.zeros((H1, 1), dtype=int)
    Y1b = np.zeros((H1, 1), dtype=int)
    X1b[7, 0] = 1
    Y1b[7, 0] = 2
    traces_b = rec.reconstruct(X1b, Y1b)
    ious_disj = rec.iou(traces, traces_b, H1)
    assert ious_disj[0] == pytest.approx(0.0), f"Disjoint IoU = {ious_disj[0]}"


def test_episode_start_f1_perfect():
    """start_detection_f1 = 1.0 when predicted starts exactly match true starts."""
    from src.inference.reconstruction import EpisodeReconstructor

    H1, L = 20, 2
    X1 = np.zeros((H1, L), dtype=int)
    Y1 = np.zeros((H1, L), dtype=int)
    X1[1, 0] = 1
    Y1[1, 0] = 2
    X1[10, 0] = 1
    Y1[10, 0] = 3
    X1[5, 1] = 1
    Y1[5, 1] = 4

    rec = EpisodeReconstructor(label_names=["A", "B"])
    traces = rec.reconstruct(X1, Y1)
    results = rec.start_detection_f1(traces, traces)

    for ell in range(L):
        assert results[ell]["f1"] == pytest.approx(
            1.0
        ), f"label {ell}: f1={results[ell]['f1']}"
        assert results[ell]["precision"] == pytest.approx(1.0)
        assert results[ell]["recall"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# TODO-4 gap-filling: MultiScaleDBNGraph SHD and networkx export
# ---------------------------------------------------------------------------


def test_graph_shd_self(mini_graph):
    """SHD of a graph against itself must be 0."""
    assert mini_graph.structural_hamming_distance(mini_graph) == 0


def test_graph_shd_one_edge(mini_graph):
    """Adding one edge to a copy must increase SHD by exactly 1."""
    g2 = mini_graph.copy()
    g2.add_edge(k=1, src_ell=0, lag=1, dst_ell=1)
    shd = mini_graph.structural_hamming_distance(g2)
    assert shd == 1, f"Expected SHD=1 after adding one edge, got {shd}"


def test_graph_networkx_export(mini_graph):
    """to_networkx returns a DiGraph with at least 2*L + 2 nodes."""
    import networkx as nx

    G = mini_graph.to_networkx(k=1)
    assert isinstance(G, nx.DiGraph)
    # At minimum: X_{ell}[t] and Y_{ell}[t] for L=2 labels + S[t] + H[t]
    assert G.number_of_nodes() >= 2 * mini_graph.L + 2


# ---------------------------------------------------------------------------
# TODO-4 gap-filling: Laplace pseudo-observations path
# ---------------------------------------------------------------------------


def test_laplace_pseudo_obs_coarse():
    """use_laplace=True with r>0 invokes compute_pseudo_obs_coarse and returns PSD Sigma_H."""
    from src.synthetic.families import make_generator as mk
    from src.synthetic.scenarios import ScenarioConfig as Cfg

    # Family III with r=1 so the Laplace path in EStep activates
    cfg = Cfg(
        name="laplace_test",
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
        seed=5,
    )
    gen = mk(cfg, seed=5)
    ds = gen.generate(seed=5)
    K = ds.data.K
    T = ds.hierarchy.n_epochs(K)
    r = ds.params[K - 1].r
    assert r == 1, "Family III must have r=1"

    graph = MultiScaleDBNGraph(hierarchy=ds.hierarchy, label_names=gen.label_names)
    estep = EStep(max_inner_iter=2, use_laplace=True)
    q = estep.run(k=K, data=ds.data, params=ds.params[K - 1], graph=graph)

    assert q.m_H.shape == (T, r)
    assert q.Sigma_H.shape == (T, r, r)
    assert np.all(np.isfinite(q.m_H)), "m_H contains non-finite values"
    for t in range(T):
        eigvals = np.linalg.eigvalsh(q.Sigma_H[t])
        assert (
            eigvals.min() >= -1e-6
        ), f"Sigma_H[{t}] not PSD: min eigval={eigvals.min():.3e}"


# ---------------------------------------------------------------------------
# TODO-4 gap-filling: EStep with non-empty parent sets (lag aggregation)
# ---------------------------------------------------------------------------


def test_estep_with_lagged_parents(mini_ds, mini_graph):
    """EStep.run with non-empty params.parent_sets exercises _agg_lag_X/_Y branches."""
    K = mini_ds.data.K

    # Deep-copy the coarsest params and inject a lag-1 parent edge for label 1
    params_with_edge = copy.deepcopy(mini_ds.params[K - 1])
    params_with_edge.parent_sets[1] = {(0, 1)}  # label 1 depends on label 0 at lag 1

    estep = EStep(max_inner_iter=2, use_laplace=False)
    q = estep.run(
        k=K,
        data=mini_ds.data,
        params=params_with_edge,
        graph=mini_graph,
    )

    T = mini_ds.hierarchy.n_epochs(K)
    M = mini_ds.params[K - 1].M
    assert q.gamma.shape == (T, M)
    np.testing.assert_allclose(
        q.gamma.sum(axis=1),
        1.0,
        atol=1e-8,
        err_msg="γ rows must sum to 1 even with lagged parents",
    )
    assert np.isfinite(q.log_Z), f"log_Z={q.log_Z} is not finite"
