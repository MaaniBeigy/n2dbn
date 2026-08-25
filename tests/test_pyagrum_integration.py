"""
tests/test_pyagrum_integration.py
=================================
Tests for pyAgrum integration (graph conversion, CPT export, scoring, visualisation).

All tests are guarded by `pytest.importorskip("pyagrum")` so the file
is safely skipped when pyAgrum is not installed.
"""

import numpy as np
import pytest

gum = pytest.importorskip("pyagrum")

from src.core.graph import MultiScaleDBNGraph
from src.core.pyagrum_utils import (
    extract_init_distribution,
    extract_transition_matrix,
    fill_regime_cpt,
    regime_cpt_to_params,
)
from src.core.scales import ScaleHierarchy
from src.learning.pyagrum_scoring import PyAgrumScore
from src.model.params import RegimeParams
from src.utils.visualization import plot_graph_pyagrum, show_regime_cpt

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def hier_K1():
    """Single-scale hierarchy with H_finest = 20."""
    return ScaleHierarchy.from_uniform_branching(20, branching_factors=[])


@pytest.fixture
def hier_K2():
    """Two-scale hierarchy with branching factor 4."""
    return ScaleHierarchy.from_uniform_branching(20, branching_factors=[4])


@pytest.fixture
def labels3():
    return ["a", "b", "c"]


@pytest.fixture
def regime_params_M2():
    """RegimeParams with M=2, d_ctx=0."""
    return RegimeParams(
        M=2,
        parent_M=0,
        d_ctx=0,
        init=np.array([0.0, 1.0]),
        bias=np.array([0.0, 0.5]),
        trans=np.array([[1.0, 0.1], [0.1, 1.0]]),
        cross=None,
        ctx=np.empty((2, 0)),
    )


@pytest.fixture
def regime_params_M3_ctx():
    """RegimeParams with M=3, d_ctx=2."""
    return RegimeParams(
        M=3,
        parent_M=0,
        d_ctx=2,
        init=np.array([0.0, 1.0, -1.0]),
        bias=np.array([0.0, 0.5, -0.5]),
        trans=np.array([[1.0, 0.1, 0.1], [0.1, 1.0, 0.1], [0.1, 0.1, 1.0]]),
        cross=None,
        ctx=np.array([[0.3, -0.2], [0.1, 0.4], [-0.1, 0.0]]),
    )


# ===================================================================
# F.1 - Round-trip conversion tests
# ===================================================================


class TestGraphConversion:
    def test_to_pyagrum_bn_empty_graph(self, hier_K1, labels3):
        g = MultiScaleDBNGraph(hierarchy=hier_K1, label_names=labels3)
        bn = g.to_pyagrum_bn(1, M=2)

        assert isinstance(bn, gum.BayesNet)
        # Should have S_t, S_t_prev, 3 X_{label}_t, 3 Y_{label}_t = 8 nodes
        assert bn.size() == 8
        # Arcs: S_t_prev -> S_t, plus S_t -> each X and Y = 1 + 6 = 7
        assert bn.dag().sizeArcs() == 7
        # No lagged nodes
        assert "X_a_t_lag1" not in bn.names()

    def test_to_pyagrum_bn_with_edges(self, hier_K1, labels3):
        g = MultiScaleDBNGraph(hierarchy=hier_K1, label_names=labels3)
        g.add_edge(1, 0, 1, 1)  # a[t-1] -> b[t]
        g.add_edge(1, 2, 2, 0)  # c[t-2] -> a[t]

        bn = g.to_pyagrum_bn(1, M=2)

        # 8 base nodes + 2 lagged nodes
        assert bn.size() == 10
        # 7 base arcs + 2 lagged arcs = 9
        assert bn.dag().sizeArcs() == 9
        assert "X_a_t_lag1" in bn.names()
        assert "X_c_t_lag2" in bn.names()

    def test_update_from_pyagrum_bn_roundtrip(self, hier_K1, labels3):
        g = MultiScaleDBNGraph(hierarchy=hier_K1, label_names=labels3)
        g.add_edge(1, 0, 1, 1)
        g.add_edge(1, 2, 2, 0)
        g.add_edge(1, 1, 1, 2)

        bn = g.to_pyagrum_bn(1, M=2)

        g2 = MultiScaleDBNGraph(hierarchy=hier_K1, label_names=labels3)
        g2.update_from_pyagrum_bn(bn, 1)

        assert g.n_edges(1) == g2.n_edges(1)
        for ell in range(3):
            assert g.get_parents(1, ell) == g2.get_parents(1, ell)

    def test_to_pyagrum_2tbn_alias(self, hier_K1, labels3):
        g = MultiScaleDBNGraph(hierarchy=hier_K1, label_names=labels3)
        g.add_edge(1, 0, 1, 1)

        bn1 = g.to_pyagrum_bn(1, M=3)
        bn2 = g.to_pyagrum_2tbn(1, M=3)

        assert bn1.size() == bn2.size()
        assert bn1.dag().sizeArcs() == bn2.dag().sizeArcs()

    def test_to_pyagrum_bn_multiple_scales(self, hier_K2, labels3):
        g = MultiScaleDBNGraph(hierarchy=hier_K2, label_names=labels3)
        g.add_edge(1, 0, 1, 1)  # edge at k=1
        g.add_edge(2, 1, 1, 2)  # edge at k=2

        bn1 = g.to_pyagrum_bn(1, M=2)
        bn2 = g.to_pyagrum_bn(2, M=2)

        # k=1 has 1 edge, k=2 has 1 edge - independent
        assert "X_a_t_lag1" in bn1.names()
        assert "X_a_t_lag1" not in bn2.names()
        assert "X_b_t_lag1" in bn2.names()


# ===================================================================
# F.2 - CPT export tests
# ===================================================================


class TestCPTExport:
    def test_extract_transition_matrix_shape(self, regime_params_M2):
        P = extract_transition_matrix(regime_params_M2)
        assert P.shape == (2, 2)
        np.testing.assert_allclose(P.sum(axis=0), [1.0, 1.0], atol=1e-10)

    def test_extract_init_distribution(self, regime_params_M2):
        pi0 = extract_init_distribution(regime_params_M2)
        assert pi0.shape == (2,)
        assert abs(pi0.sum() - 1.0) < 1e-10
        # init = [0.0, 1.0] to softmax to second element larger
        assert pi0[1] > pi0[0]

    def test_fill_regime_cpt_rows_sum_to_one(self, hier_K1, labels3, regime_params_M2):
        g = MultiScaleDBNGraph(hierarchy=hier_K1, label_names=labels3)
        bn = g.to_pyagrum_bn(1, M=2)
        fill_regime_cpt(bn, regime_params_M2)

        cpt_array = bn.cpt("S_t").toarray()
        # Each column should sum to 1 (conditional on S_t_prev)
        np.testing.assert_allclose(cpt_array.sum(axis=0), [1.0, 1.0], atol=1e-10)

    def test_fill_regime_cpt_matches_softmax(self, hier_K1, labels3, regime_params_M2):
        g = MultiScaleDBNGraph(hierarchy=hier_K1, label_names=labels3)
        bn = g.to_pyagrum_bn(1, M=2)
        fill_regime_cpt(bn, regime_params_M2)

        P_expected = extract_transition_matrix(regime_params_M2)
        P_from_bn = bn.cpt("S_t").toarray()
        np.testing.assert_allclose(P_from_bn, P_expected, atol=1e-10)

    def test_regime_cpt_to_params_roundtrip(self, hier_K1, labels3, regime_params_M2):
        g = MultiScaleDBNGraph(hierarchy=hier_K1, label_names=labels3)
        bn = g.to_pyagrum_bn(1, M=2)
        fill_regime_cpt(bn, regime_params_M2)

        pi0_back, P_back = regime_cpt_to_params(bn)
        pi0_orig = extract_init_distribution(regime_params_M2)
        P_orig = extract_transition_matrix(regime_params_M2)

        np.testing.assert_allclose(pi0_back, pi0_orig, atol=1e-10)
        np.testing.assert_allclose(P_back, P_orig, atol=1e-10)

    def test_fill_regime_cpt_with_context(self, hier_K1, labels3, regime_params_M3_ctx):
        g = MultiScaleDBNGraph(hierarchy=hier_K1, label_names=labels3)
        bn_no_ctx = g.to_pyagrum_bn(1, M=3)
        fill_regime_cpt(bn_no_ctx, regime_params_M3_ctx, ctx_t=None)
        P_no_ctx = bn_no_ctx.cpt("S_t").toarray().copy()

        bn_ctx = g.to_pyagrum_bn(1, M=3)
        ctx = np.array([1.0, -0.5])
        fill_regime_cpt(bn_ctx, regime_params_M3_ctx, ctx_t=ctx)
        P_ctx = bn_ctx.cpt("S_t").toarray()

        # Context should change the CPT
        assert not np.allclose(P_no_ctx, P_ctx)
        # Both should still be valid probability distributions
        np.testing.assert_allclose(P_ctx.sum(axis=0), np.ones(3), atol=1e-10)


# ===================================================================
# F.3 - Scoring backend tests
# ===================================================================


class TestPyAgrumScoring:
    @pytest.fixture
    def correlated_data(self):
        """Generate T=200, L=3 data where b[t] ~ a[t-1]."""
        rng = np.random.default_rng(42)
        T, L = 200, 3
        X = np.zeros((T, L))
        X[:, 0] = rng.integers(0, 10, T)
        for t in range(1, T):
            X[t, 1] = np.clip(X[t - 1, 0] + rng.integers(-2, 3), 0, 10)
        X[:, 2] = rng.integers(0, 10, T)
        return X, ["a", "b", "c"]

    def test_pyagrum_score_finite(self, correlated_data):
        X, labels = correlated_data
        scorer = PyAgrumScore(X, labels, max_lag=2, n_bins=5, score_type="BDeu")
        s = scorer.score_parent_set(1, set())
        assert np.isfinite(s)

    def test_pyagrum_score_empty_vs_parent(self, correlated_data):
        X, labels = correlated_data
        scorer = PyAgrumScore(X, labels, max_lag=2, n_bins=5, score_type="BDeu")
        s_empty = scorer.score_parent_set(1, set())
        s_a_lag1 = scorer.score_parent_set(1, {(0, 1)})
        # With 200 samples, BDeu should detect the correlation
        assert s_a_lag1 > s_empty

    def test_learn_initial_structure_smoke(self, correlated_data):
        X, labels = correlated_data
        scorer = PyAgrumScore(X, labels, max_lag=2, n_bins=5, score_type="BDeu")
        ps = scorer.learn_initial_structure(max_indegree=3)
        # Should return a valid parent_sets dict
        assert isinstance(ps, dict)
        assert set(ps.keys()) == {0, 1, 2}
        for ell, parents in ps.items():
            assert isinstance(parents, set)

    def test_sstep_pyagrum_backend_smoke(self):
        """SStep with scoring_backend='pyagrum_bdeu' runs without error."""
        from src.core.variables import MultiScaleData
        from src.inference.posterior import PosteriorSummaries
        from src.learning.sstep import SStep
        from src.model.params import (
            CoarseCountParams,
            CoarseDurParams,
            FactorParams,
            RegimeParams,
            ScaleParams,
        )

        rng = np.random.default_rng(42)
        T, L, M, r = 30, 3, 2, 0

        hier = ScaleHierarchy.from_uniform_branching(T, branching_factors=[])
        X = rng.integers(0, 10, (T, L)).astype(float)
        Y = rng.integers(0, 5, (T, L)).astype(float)
        data = MultiScaleData(
            hierarchy=hier,
            label_names=[f"l{i}" for i in range(L)],
            X=[X],
            Y=[Y],
            C=[np.zeros((T, 0))],
        )

        rp = RegimeParams(
            M=M,
            parent_M=0,
            d_ctx=0,
            init=np.zeros(M),
            bias=np.zeros(M),
            trans=np.zeros((M, M)),
            cross=None,
            ctx=np.empty((M, 0)),
        )
        fp = FactorParams(
            M=M,
            r=r,
            d_ctx=0,
            F=np.empty((M, 0, 0)),
            G=np.empty((M, 0, 0)),
            Sigma=np.empty((M, 0, 0)),
            mu0=np.empty(0),
            Sigma0=np.empty((0, 0)),
        )
        cc = CoarseCountParams(
            L=L,
            beta0=np.zeros(L),
            betaS=np.zeros((M, L)),
            betaH=np.empty((0, L)),
            betaCtx=np.empty((0, L)),
            betaLag={},
            r_disp=np.ones(L),
            gamma0=np.zeros(L),
            gammaS=np.zeros((M, L)),
            gammaH=np.empty((0, L)),
            gammaCtx=np.empty((0, L)),
        )
        cd = CoarseDurParams(
            L=L,
            alpha0=np.zeros(L),
            alphaS=np.zeros((M, L)),
            alphaH=np.empty((0, L)),
            alphaCtx=np.empty((0, L)),
            alphaX=np.zeros(L),
            alphaLag={},
            s_disp=np.ones(L),
        )

        params = ScaleParams(k=1, regime=rp, factor=fp, coarse_count=cc, coarse_dur=cd)

        gamma = np.ones((T, M)) / M
        xi = np.ones((T - 1, M, M)) / (M * M)
        q = PosteriorSummaries(
            gamma=gamma,
            xi=xi,
            m_H=np.zeros((T, 0)),
            Sigma_H=np.zeros((T, 0, 0)),
            log_Z=-100.0,
        )

        graph = MultiScaleDBNGraph(
            hierarchy=hier, label_names=[f"l{i}" for i in range(L)]
        )

        sstep = SStep(lam=1.0, max_lag=1, scoring_backend="pyagrum_bdeu", n_bins=3)
        new_graph, new_params = sstep.run(1, q, params, graph, data)
        assert isinstance(new_graph, MultiScaleDBNGraph)


# ===================================================================
# F.4 - Visualization tests
# ===================================================================


class TestPyAgrumVisualization:
    def test_plot_graph_pyagrum_smoke(self, hier_K1, labels3):
        g = MultiScaleDBNGraph(hierarchy=hier_K1, label_names=labels3)
        g.add_edge(1, 0, 1, 1)
        dot = plot_graph_pyagrum(g, scale=1, M=2)
        assert isinstance(dot, str)
        assert "digraph" in dot

    def test_show_regime_cpt_smoke(self, hier_K1, labels3, regime_params_M2):
        g = MultiScaleDBNGraph(hierarchy=hier_K1, label_names=labels3)
        s = show_regime_cpt(g, regime_params_M2, scale=1)
        assert isinstance(s, str)
        assert "S_t" in s


# ===================================================================
# F.5 - Validation tests
# ===================================================================


class TestValidation:
    def test_validate_pyagrum_consistent(self, hier_K1, labels3):
        g = MultiScaleDBNGraph(hierarchy=hier_K1, label_names=labels3)
        g.add_edge(1, 0, 1, 1)
        g.add_edge(1, 2, 2, 0)
        assert g.validate_pyagrum(1, M=2)

    def test_validate_pyagrum_empty_graph(self, hier_K1, labels3):
        g = MultiScaleDBNGraph(hierarchy=hier_K1, label_names=labels3)
        assert g.validate_pyagrum(1, M=2)


# ===================================================================
# F.6 - Backward compatibility
# ===================================================================


class TestBackwardCompatibility:
    def test_existing_graph_api_unchanged(self, hier_K1, labels3):
        """All dict-based methods still work identically."""
        g = MultiScaleDBNGraph(hierarchy=hier_K1, label_names=labels3)
        g.add_edge(1, 0, 1, 1)
        g.add_edge(1, 2, 2, 0)

        assert g.n_edges(1) == 2
        assert g.get_parents(1, 1) == {(0, 1)}
        assert g.get_parents(1, 0) == {(2, 2)}
        assert g.is_acyclic()

        g.remove_edge(1, 0, 1, 1)
        assert g.n_edges(1) == 1

        g2 = g.copy()
        assert g2.n_edges(1) == g.n_edges(1)

        assert g.structural_hamming_distance(g2) == 0

        nx_g = g.to_networkx(1)
        assert nx_g is not None

        r = repr(g)
        assert "MultiScaleDBNGraph" in r
