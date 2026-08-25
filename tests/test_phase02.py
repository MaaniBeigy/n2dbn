"""
tests/test_phase02.py
=====================
Emission log-likelihoods and allocation sampling.

Tests
-----
1.  NB PMF normalization      - Σ_x P(X=x|r,μ) ≈ 1
2.  ZINB PMF normalization     - Σ_x P(X=x|π,r,μ) ≈ 1
3.  ZINB zero-inflation        - P(X=0) strictly increasing in π
4.  ZINB reduces to NB        - when π ≈ 0, ZINB ≈ NB everywhere
5.  nb_sample shape            - output shape and non-negative
6.  zinb_sample shape          - output shape and non-negative
7.  RegimeChain transition_matrix - each column sums to 1
8.  RegimeChain log_transition_prob - finite and ≤ 0 for sampled sequence
9.  RegimeChain sample_sequence - shape (T,), values in [0, M)
10. LatentFactor sample_trajectory - shape (T, r)
11. LatentFactor r=0 no-op     - returns (T, 0)
12. LatentFactor log_prob       - finite scalar for sampled trajectory
13. DirMult sample sum          - Σ_i x_i == n
14. DirMult n=0 case            - all zeros returned

Run with:
    pytest tests/test_phase02.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from src.model.allocation import dirichlet_multinomial_sample
from src.model.emissions import nb_log_pmf, nb_sample, zinb_log_pmf, zinb_sample
from src.model.latent_factor import LatentFactor
from src.model.latent_regime import RegimeChain
from src.model.params import FactorParams, RegimeParams

# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _regime_params(M: int, d_ctx: int = 0) -> RegimeParams:
    """Minimal RegimeParams with no cross-scale coupling."""
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


def _factor_params(M: int = 2, r: int = 2, d_ctx: int = 0) -> FactorParams:
    """Minimal FactorParams: stable AR(1) with identity dynamics."""
    F = np.tile(0.8 * np.eye(r), (M, 1, 1))
    G = np.zeros((M, r, max(d_ctx, 1)))[:, :, :d_ctx]  # (M, r, d_ctx)
    Sigma = np.tile(0.1 * np.eye(r), (M, 1, 1))
    return FactorParams(
        M=M,
        r=r,
        d_ctx=d_ctx,
        F=F,
        G=G,
        Sigma=Sigma,
        mu0=np.zeros(r),
        Sigma0=np.eye(r),
    )


# ---------------------------------------------------------------------------
# 1.  NB PMF normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("r,mu", [(1.0, 5.0), (5.0, 10.0), (0.5, 2.0)])
def test_nb_pmf_sums_to_one(r: float, mu: float) -> None:
    """NB PMF must sum to ≈ 1 over x = 0..199."""
    x = np.arange(200)
    log_p = nb_log_pmf(
        np.full_like(x, float(x[0])) * 0 + x, np.full(200, r), np.full(200, mu)
    )
    total = np.exp(log_p).sum()
    assert abs(total - 1.0) < 0.02, f"NB sum={total:.4f} (r={r}, mu={mu})"


# ---------------------------------------------------------------------------
# 2.  ZINB PMF normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pi,r,mu",
    [
        (0.3, 2.0, 5.0),
        (0.0, 1.0, 3.0),
        (0.8, 0.5, 1.0),
    ],
)
def test_zinb_pmf_sums_to_one(pi: float, r: float, mu: float) -> None:
    """ZINB PMF must sum to ≈ 1."""
    x = np.arange(200)
    log_p = zinb_log_pmf(x, np.full(200, pi), np.full(200, r), np.full(200, mu))
    total = np.exp(log_p).sum()
    assert abs(total - 1.0) < 0.02, f"ZINB sum={total:.4f} (pi={pi}, r={r}, mu={mu})"


# ---------------------------------------------------------------------------
# 3.  ZINB zero-inflation property
# ---------------------------------------------------------------------------


def test_zinb_zero_inflation_increases_with_pi() -> None:
    """P(X=0) must be strictly increasing in π."""
    r, mu = 2.0, 5.0
    x0 = np.array([0])
    r_arr = np.array([r])
    mu_arr = np.array([mu])

    p0_low = float(np.exp(zinb_log_pmf(x0, np.array([0.1]), r_arr, mu_arr))[0])
    p0_mid = float(np.exp(zinb_log_pmf(x0, np.array([0.5]), r_arr, mu_arr))[0])
    p0_high = float(np.exp(zinb_log_pmf(x0, np.array([0.9]), r_arr, mu_arr))[0])

    assert (
        p0_low < p0_mid < p0_high
    ), f"P(X=0) should increase with pi: {p0_low:.4f}, {p0_mid:.4f}, {p0_high:.4f}"


# ---------------------------------------------------------------------------
# 4.  ZINB reduces to NB when π to 0
# ---------------------------------------------------------------------------


def test_zinb_reduces_to_nb_when_pi_zero() -> None:
    """ZINB(π≈0, r, μ) ≈ NB(r, μ) pointwise."""
    x = np.arange(50)
    r_arr = np.full(50, 2.0)
    mu_arr = np.full(50, 5.0)
    pi_zero = np.full(50, 1e-10)

    log_zinb = zinb_log_pmf(x, pi_zero, r_arr, mu_arr)
    log_nb = nb_log_pmf(x, r_arr, mu_arr)

    np.testing.assert_allclose(
        log_zinb, log_nb, atol=1e-5, err_msg="ZINB(pi to 0) should equal NB"
    )


# ---------------------------------------------------------------------------
# 5-6.  Sample shapes
# ---------------------------------------------------------------------------


def test_nb_sample_shape_and_nonneg() -> None:
    rng = np.random.default_rng(42)
    shape = (3, 4)
    s = nb_sample(np.ones(shape), 5.0 * np.ones(shape), rng)
    assert s.shape == shape
    assert (s >= 0).all()


def test_zinb_sample_shape_and_nonneg() -> None:
    rng = np.random.default_rng(42)
    n = 7
    pi = 0.3 * np.ones(n)
    r = 2.0 * np.ones(n)
    mu = 4.0 * np.ones(n)
    s = zinb_sample(pi, r, mu, rng)
    assert s.shape == (n,)
    assert (s >= 0).all()


# ---------------------------------------------------------------------------
# 7.  RegimeChain transition matrix columns sum to 1
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("M", [2, 3, 5])
def test_transition_matrix_columns_sum_to_one(M: int) -> None:
    """Each column of T_mat (axis=0) is a valid probability simplex."""
    p = _regime_params(M=M, d_ctx=0)
    chain = RegimeChain(p)
    ctx_t = np.zeros(0)
    T_mat = chain.transition_matrix(ctx_t)

    assert T_mat.shape == (M, M)
    np.testing.assert_allclose(
        T_mat.sum(axis=0),
        np.ones(M),
        atol=1e-10,
        err_msg="Transition matrix columns must sum to 1",
    )


# ---------------------------------------------------------------------------
# 8.  RegimeChain log_transition_prob is finite and ≤ 0
# ---------------------------------------------------------------------------


def test_log_transition_prob_finite_and_nonpositive() -> None:
    rng = np.random.default_rng(7)
    M = 3
    p = _regime_params(M=M, d_ctx=0)
    chain = RegimeChain(p)
    T = 20
    ctx = np.zeros((T, 0))
    S = chain.sample_sequence(T, ctx, parent_regimes=None, rng=rng)
    log_p = chain.log_transition_prob(S, ctx, parent_regimes=None)
    assert np.isfinite(log_p), f"log p(S) = {log_p} is not finite"
    assert log_p <= 1e-8, f"log p(S) = {log_p} > 0"


# ---------------------------------------------------------------------------
# 9.  RegimeChain sample_sequence shape and range
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("M,T", [(2, 15), (4, 30), (1, 5)])
def test_sample_sequence_shape_and_range(M: int, T: int) -> None:
    rng = np.random.default_rng(99)
    p = _regime_params(M=M, d_ctx=0)
    chain = RegimeChain(p)
    ctx = np.zeros((T, 0))
    S = chain.sample_sequence(T, ctx, parent_regimes=None, rng=rng)
    assert S.shape == (T,), f"Expected shape ({T},), got {S.shape}"
    assert (
        int(S.min()) >= 0 and int(S.max()) < M
    ), f"Values out of range [0, {M}): min={S.min()}, max={S.max()}"


# ---------------------------------------------------------------------------
# 10.  LatentFactor sample_trajectory shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("M,r,T", [(2, 3, 20), (3, 1, 10)])
def test_factor_trajectory_shape(M: int, r: int, T: int) -> None:
    rng = np.random.default_rng(0)
    fp = _factor_params(M=M, r=r, d_ctx=0)
    lf = LatentFactor(fp)
    regimes = rng.integers(0, M, size=T)
    ctx = np.zeros((T, 0))
    H = lf.sample_trajectory(T, regimes, ctx, rng)
    assert H.shape == (T, r), f"Expected ({T}, {r}), got {H.shape}"


# ---------------------------------------------------------------------------
# 11.  LatentFactor r=0 returns (T, 0) - no-op
# ---------------------------------------------------------------------------


def test_factor_trajectory_r0_noop() -> None:
    """When r=0, sample_trajectory must return (T, 0) without error."""
    rng = np.random.default_rng(0)
    M, T = 2, 10
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
    lf = LatentFactor(fp)
    regimes = rng.integers(0, M, size=T)
    ctx = np.zeros((T, 0))
    H = lf.sample_trajectory(T, regimes, ctx, rng)
    assert H.shape == (T, 0), f"Expected ({T}, 0), got {H.shape}"


# ---------------------------------------------------------------------------
# 12.  LatentFactor log_prob is finite
# ---------------------------------------------------------------------------


def test_factor_log_prob_finite() -> None:
    rng = np.random.default_rng(5)
    M, r, T = 2, 2, 15
    fp = _factor_params(M=M, r=r, d_ctx=0)
    lf = LatentFactor(fp)
    regimes = rng.integers(0, M, size=T)
    ctx = np.zeros((T, 0))
    H = lf.sample_trajectory(T, regimes, ctx, rng)
    log_p = lf.log_prob(H, regimes, ctx)
    assert np.isfinite(log_p), f"log p(H) = {log_p} is not finite"


def test_factor_log_prob_r0_is_zero() -> None:
    """log_prob must return 0.0 when r=0 (no factor)."""
    M, r, T = 2, 0, 10
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
    lf = LatentFactor(fp)
    H = np.empty((T, 0))
    regimes = np.zeros(T, dtype=int)
    ctx = np.zeros((T, 0))
    assert lf.log_prob(H, regimes, ctx) == 0.0


# ---------------------------------------------------------------------------
# 13-14.  DirMult allocation
# ---------------------------------------------------------------------------


def test_dirmult_sample_sum_equals_n() -> None:
    """Output must sum to n for various n and number of categories."""
    rng = np.random.default_rng(10)
    for n, K in [(20, 5), (1, 3), (100, 10), (7, 2)]:
        scores = rng.standard_normal(K)
        x = dirichlet_multinomial_sample(n, scores, concentration=10.0, rng=rng)
        assert int(x.sum()) == n, f"n={n}, K={K}: sum={x.sum()}"
        assert (x >= 0).all()


def test_dirmult_sample_zero_n_returns_zeros() -> None:
    """n=0 must return the zero vector."""
    rng = np.random.default_rng(10)
    K = 5
    x = dirichlet_multinomial_sample(0, np.zeros(K), concentration=10.0, rng=rng)
    assert x.shape == (K,)
    assert (x == 0).all()
