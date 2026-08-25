"""
tests/test_phase10_edge_cases.py
================================
TODO-6: Edge-case coverage tests for evaluation metrics, visualization,
and emission utilities.

Tests
-----
Metrics (3 tests)
  1. test_latent_metrics_factor_correlation - r>0 factor correlation path
  2. test_reconstruction_metrics_compute    - xfail: API mismatch in compute()
  3. test_best_perm_accuracy_large_M        - M>8 greedy matching path

Visualization (2 tests)
  4. test_plot_episode_comparison           - plot_episode_comparison()
  5. test_plot_factor_trajectory_with_true  - true_H overlay in trajectory plot

Emissions (2 tests)
  6. test_softplus_large_input             - x>30 branch in _softplus
  7. test_aggregate_lagged_features        - aggregate_lagged() with parents

Run with:
    pytest tests/test_phase10_edge_cases.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# 1. LatentMetrics - factor correlation with r > 0
# ---------------------------------------------------------------------------


def test_latent_metrics_factor_correlation():
    """With r>0, LatentMetrics.compute must return finite factor_corr in [0,1]."""
    from src.evaluation.metrics import LatentMetrics

    T, M, r = 50, 2, 2
    rng = np.random.default_rng(42)

    # Construct H_true with two independent random walks
    H_true = np.cumsum(rng.standard_normal((T, r)), axis=0)

    # H_hat is a noisy version of H_true (should have high correlation)
    H_hat = H_true + 0.1 * rng.standard_normal((T, r))

    gamma = rng.dirichlet(np.ones(M), size=T)
    true_S = rng.integers(0, M, size=T)

    lm = LatentMetrics.compute(
        gamma_per_scale=[gamma],
        true_S_per_scale=[true_S],
        m_H_per_scale=[H_hat],
        true_H_per_scale=[H_true],
    )

    fc = lm.factor_corr[0]
    assert np.isfinite(fc), f"factor_corr = {fc} is not finite"
    assert 0.0 <= fc <= 1.0 + 1e-8, f"factor_corr = {fc} outside [0,1]"
    # Since H_hat ≈ H_true + small noise, correlation should be high
    assert fc > 0.8, f"Expected high correlation, got {fc}"

    # Also check to_dict has the key
    d = lm.to_dict()
    assert "FactCorr_k1" in d


# ---------------------------------------------------------------------------
# 2. ReconstructionMetrics.compute - broken API (xfail)
# ---------------------------------------------------------------------------


def test_reconstruction_metrics_compute():
    """ReconstructionMetrics.compute() should produce valid start_f1 and iou."""
    from src.evaluation.metrics import ReconstructionMetrics

    T, L = 20, 2
    rng = np.random.default_rng(7)

    true_X1 = np.zeros((T, L), dtype=int)
    true_Y1 = np.zeros((T, L), dtype=int)
    true_X1[2, 0] = 1
    true_Y1[2, 0] = 3
    true_X1[10, 1] = 1
    true_Y1[10, 1] = 2

    # Predicted = true (perfect reconstruction)
    pred_X1 = true_X1.astype(float)
    pred_Y1 = true_Y1.astype(float)

    rm = ReconstructionMetrics.compute(true_X1, true_Y1, pred_X1, pred_Y1)
    assert 0.0 <= rm.start_f1 <= 1.0
    assert 0.0 <= rm.iou <= 1.0


# ---------------------------------------------------------------------------
# 3. _best_perm_accuracy - M > 8 triggers greedy path
# ---------------------------------------------------------------------------


def test_best_perm_accuracy_large_M():
    """With M>8, _best_perm_accuracy must use greedy matching (not enumeration)."""
    from src.evaluation.metrics import _best_perm_accuracy

    M = 10
    T = 100
    rng = np.random.default_rng(99)

    S_true = rng.integers(0, M, size=T)
    # Perfect prediction - greedy should find accuracy = 1.0
    S_pred = S_true.copy()
    acc = _best_perm_accuracy(S_pred, S_true, M)
    assert acc == pytest.approx(1.0), f"Perfect match should give acc=1.0, got {acc}"

    # Random prediction - accuracy should be low but finite
    S_rand = rng.integers(0, M, size=T)
    acc_rand = _best_perm_accuracy(S_rand, S_true, M)
    assert 0.0 <= acc_rand <= 1.0, f"acc = {acc_rand} outside [0,1]"
    assert np.isfinite(acc_rand)


# ---------------------------------------------------------------------------
# 4. plot_episode_comparison
# ---------------------------------------------------------------------------


def test_plot_episode_comparison():
    """plot_episode_comparison() returns a Figure without error."""
    from src.utils.visualization import plot_episode_comparison

    T, L = 30, 2
    true_X1 = np.zeros((T, L), dtype=int)
    true_Y1 = np.zeros((T, L), dtype=int)
    true_X1[3, 0] = 1
    true_Y1[3, 0] = 4
    true_X1[15, 0] = 1
    true_Y1[15, 0] = 2
    true_X1[5, 1] = 1
    true_Y1[5, 1] = 3

    pred_X1 = true_X1.astype(float)
    pred_Y1 = true_Y1.astype(float)

    fig = plot_episode_comparison(
        true_X1, true_Y1, pred_X1, pred_Y1, label=0, max_slots=30
    )

    import matplotlib.pyplot as plt

    assert isinstance(fig, plt.Figure)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. plot_factor_trajectory with true_H overlay
# ---------------------------------------------------------------------------


def test_plot_factor_trajectory_with_true():
    """plot_factor_trajectory with true_H kwarg overlays the ground truth."""
    from src.inference.posterior import PosteriorSummaries
    from src.utils.visualization import plot_factor_trajectory

    T, M, r = 20, 2, 1
    rng = np.random.default_rng(11)

    summary = PosteriorSummaries(
        gamma=rng.dirichlet(np.ones(M), size=T),
        xi=rng.dirichlet(np.ones(M), size=(T - 1, M)),
        m_H=rng.standard_normal((T, r)),
        Sigma_H=np.tile(0.1 * np.eye(r), (T, 1, 1)),
        log_Z=-50.0,
    )

    true_H = [rng.standard_normal((T, r))]  # one scale

    fig = plot_factor_trajectory(
        summaries=[summary],
        true_H=true_H,
        scale=1,
    )

    import matplotlib.pyplot as plt

    assert isinstance(fig, plt.Figure)
    # With r=1, there should be 1 subplot with 2 lines (posterior + true)
    ax = fig.axes[0]
    assert len(ax.lines) >= 2, "Expected at least 2 lines (posterior + true_H)"
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. _softplus - large input branch (x > 30)
# ---------------------------------------------------------------------------


def test_softplus_large_input():
    """For x > 30, _softplus should return x directly (identity branch)."""
    from src.model.emissions import _softplus

    # Values well above 30 - identity branch
    x_large = np.array([31.0, 50.0, 100.0, 500.0])
    result = _softplus(x_large)
    np.testing.assert_array_almost_equal(result, x_large)

    # Values below 30 - log1p(exp(x)) branch
    x_small = np.array([0.0, 1.0, -5.0])
    result_small = _softplus(x_small)
    expected = np.log1p(np.exp(x_small))
    np.testing.assert_array_almost_equal(result_small, expected)

    # Boundary near 30 - both branches should be close
    x_boundary = np.array([29.9, 30.1])
    result_boundary = _softplus(x_boundary)
    assert all(np.isfinite(result_boundary))
    # At x=30, log(1+exp(30)) ≈ 30.0 (within machine precision)
    assert abs(result_boundary[0] - 29.9) < 0.01


# ---------------------------------------------------------------------------
# 7. aggregate_lagged - lagged count features
# ---------------------------------------------------------------------------


def test_aggregate_lagged_features():
    """aggregate_lagged must accumulate lagged log(1+X) contributions."""
    from src.model.emissions import aggregate_lagged

    T, L = 10, 3
    rng = np.random.default_rng(33)
    X_arr = rng.integers(0, 5, size=(T, L))

    # Label 1 has parent (src_ell=0, lag=1), label 2 has parent (src_ell=1, lag=2)
    parent_sets = {
        1: {(0, 1)},
        2: {(1, 2)},
    }

    # Coefficients: each parent contributes a fixed weight
    coeff = {
        (0, 1): np.array([0.0, 0.5, 0.0]),  # only affects label 1
        (1, 2): np.array([0.0, 0.0, 0.3]),  # only affects label 2
    }

    # At t=0: lag=1 to t_lag=-1 < 0 to no contribution
    result_t0 = aggregate_lagged(
        t=0, X_arr=X_arr, parent_sets=parent_sets, coeff=coeff, L=L
    )
    assert result_t0[0] == 0.0  # label 0 has no parents
    assert result_t0[1] == 0.0  # t-1 < 0

    # At t=3: label 1 gets 0.5 * log(1 + X[2, 0])
    result_t3 = aggregate_lagged(
        t=3, X_arr=X_arr, parent_sets=parent_sets, coeff=coeff, L=L
    )
    expected_label1 = 0.5 * np.log1p(float(X_arr[2, 0]))
    assert result_t3[1] == pytest.approx(expected_label1)

    # At t=3: label 2 gets 0.3 * log(1 + X[1, 1])
    expected_label2 = 0.3 * np.log1p(float(X_arr[1, 1]))
    assert result_t3[2] == pytest.approx(expected_label2)

    # Label 0 always zero
    assert result_t3[0] == 0.0
