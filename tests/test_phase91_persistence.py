"""
Tests for the persistence trajectory row A11 waits on.

What is pinned here is that the rolling autocorrelation follows a persistence
that moves and holds still where the persistence holds still, that the local
periodogram puts a window's power at that window's own rhythm and agrees with
the discrete transform, that the CUSUM finds the window where the spectrum turns
and finds nothing in a record of one spectrum or in a straight line, and that
every degenerate series returns a defined value while an undefined one is
refused.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.persistence import (
    WINDOW_PERIODS,
    SpectralChange,
    local_periodogram,
    persistence_drift,
    persistence_rmse,
    persistence_variance,
    rolling_autocorrelation,
    spectral_cusum,
    window_plan,
)

#: The grid the local periodogram is read on, in cycles per epoch.  It holds the
#: two rhythms the switching fixture carries, at 1/20 and at 1/5.
FREQUENCIES = (0.02, 0.05, 0.1, 0.2, 0.25)


def ar_series(early: float, late: float, seed: int = 0, n: int = 960) -> np.ndarray:
    """An autoregressive series whose coefficient turns at the halfway epoch."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=n)
    phi = np.where(np.arange(n) < n // 2, early, late)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi[t] * x[t - 1] + noise[t]
    return x


def two_rhythm_series(seed: int = 3, n: int = 800) -> np.ndarray:
    """A rhythm of period 20 for the first half and of period 5 for the second."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    slow = np.sin(2.0 * np.pi * t / 20.0)
    fast = np.sin(2.0 * np.pi * t / 5.0)
    return np.where(t < n // 2, slow, fast) + 0.2 * rng.normal(size=n)


def one_rhythm_series(seed: int = 4, n: int = 800) -> np.ndarray:
    """The same rhythm of period 20 across the whole record."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    return np.sin(2.0 * np.pi * t / 20.0) + 0.2 * rng.normal(size=n)


def test_the_window_plan_spans_two_periods_and_opens_one_period_apart():
    assert WINDOW_PERIODS == 2.0
    assert window_plan(24.0) == (48, 24, 24)
    assert window_plan(7.0) == (14, 7, 7)
    # The shortest period still declares a window carrying a lag.
    assert window_plan(1.0) == (2, 1, 1)


def test_a_period_of_no_epochs_is_refused():
    with pytest.raises(ValueError, match="positive and finite"):
        window_plan(0.0)
    with pytest.raises(ValueError, match="positive and finite"):
        window_plan(float("nan"))
    with pytest.raises(ValueError, match="positive and finite"):
        persistence_drift(np.zeros(100), -7.0)


def test_the_lag_one_trajectory_follows_the_persistence_down():
    trajectory = rolling_autocorrelation(ar_series(0.85, 0.05), 96, 48, 6)
    lag_one = trajectory[:, 0]
    assert lag_one[0] > 0.5
    assert lag_one[-1] < 0.2
    assert lag_one[:8].mean() - lag_one[-8:].mean() > 0.4


def test_a_steady_series_holds_one_persistence_across_the_record():
    steady = rolling_autocorrelation(ar_series(0.5, 0.5, seed=1), 96, 48, 6)
    moving = rolling_autocorrelation(ar_series(0.85, 0.05, seed=1), 96, 48, 6)
    assert steady[:, 0].std() < 0.2
    assert moving[:, 0].std() > 2.0 * steady[:, 0].std()


def test_the_higher_lags_decay_the_way_the_persistence_says_they_do():
    first = rolling_autocorrelation(ar_series(0.8, 0.8, seed=2), 96, 48, 6)[0]
    assert first[0] > first[2] > first[5]
    # Lag k of an autoregressive series of one coefficient decays as that
    # coefficient to the k, so lag 2 sits near the square of lag 1.
    assert abs(first[1] - first[0] ** 2) < 0.15


def test_the_trajectory_of_a_constant_series_is_zero_at_every_lag():
    trajectory = rolling_autocorrelation(np.full(200, 3.0), 48, 24, 24)
    assert trajectory.shape == (7, 24)
    assert np.all(trajectory == 0.0)


def test_a_series_too_short_for_two_windows_carries_no_trajectory():
    assert rolling_autocorrelation(np.arange(60.0), 48, 24, 24).shape == (0, 24)
    assert rolling_autocorrelation(np.zeros(0), 48, 24, 24).shape == (0, 24)


def test_a_window_the_largest_lag_reaches_is_refused():
    with pytest.raises(ValueError, match="carries no lag"):
        rolling_autocorrelation(np.arange(100.0), 24, 12, 24)
    with pytest.raises(ValueError, match="at least one epoch"):
        rolling_autocorrelation(np.arange(100.0), 24, 12, 0)


def test_a_window_too_narrow_to_open_names_the_window_and_not_the_lag():
    # A width of zero is a bad width, and the message names the width.
    with pytest.raises(ValueError, match="at least two epochs"):
        rolling_autocorrelation(np.arange(100.0), 0, 24, 24)
    with pytest.raises(ValueError, match="at least one epoch"):
        rolling_autocorrelation(np.arange(100.0), 48, 0, 24)


def test_a_window_of_one_epoch_and_a_step_of_none_are_refused():
    with pytest.raises(ValueError, match="at least two epochs"):
        local_periodogram(np.arange(100.0), 1, 12, FREQUENCIES)
    with pytest.raises(ValueError, match="at least one epoch"):
        local_periodogram(np.arange(100.0), 24, 0, FREQUENCIES)


def test_the_local_periodogram_puts_its_power_at_the_rhythm_of_its_own_window():
    power = local_periodogram(two_rhythm_series(), 40, 40, FREQUENCIES)
    assert power.shape == (20, 5)
    assert FREQUENCIES[int(np.argmax(power[0]))] == 0.05
    assert FREQUENCIES[int(np.argmax(power[-1]))] == 0.2


def test_a_ramp_inside_a_window_leaves_the_low_frequencies_alone():
    epochs = np.arange(200.0)
    power = local_periodogram(
        epochs + np.sin(2.0 * np.pi * epochs / 5.0), 40, 20, FREQUENCIES
    )
    assert FREQUENCIES[int(np.argmax(power[0]))] == 0.2
    assert power[0, 0] < 0.01 * power[0, 3]


def test_a_flat_window_carries_no_power_at_any_frequency():
    power = local_periodogram(np.full(400, 2.0), 40, 40, FREQUENCIES)
    assert power.shape == (10, 5)
    assert np.all(power == 0.0)


def test_a_frequency_grid_that_cannot_be_read_is_refused():
    with pytest.raises(ValueError, match="at least one frequency"):
        local_periodogram(np.arange(200.0), 40, 20, [])
    with pytest.raises(ValueError, match="positive, in cycles per epoch"):
        local_periodogram(np.arange(200.0), 40, 20, [0.0, 0.2])
    with pytest.raises(ValueError, match="finite values only"):
        local_periodogram(np.arange(200.0), 40, 20, [0.1, float("nan")])
    with pytest.raises(ValueError, match="finite values only"):
        local_periodogram(np.arange(200.0), 40, 20, [float("inf")])


def test_the_periodogram_agrees_with_the_discrete_transform_on_its_own_grid():
    # An independent reading of the same window: the squared modulus of the
    # discrete transform over the window length, at three Fourier frequencies.
    width = 64
    epochs = np.arange(width, dtype=float)
    block = 2.0 * np.sin(2.0 * np.pi * 5.0 * epochs / width) + 0.5 * np.cos(
        2.0 * np.pi * 11.0 * epochs / width
    )
    power = local_periodogram(
        np.concatenate([block, block]),
        width,
        width,
        [5.0 / width, 11.0 / width, 17.0 / width],
    )
    detrended = block - np.polyval(np.polyfit(epochs, block, 1), epochs)
    reference = np.abs(np.fft.rfft(detrended)) ** 2 / width
    assert power.shape == (2, 3)
    assert np.allclose(power[0], reference[[5, 11, 17]], rtol=1e-8, atol=1e-12)
    # The record repeats one window, so the two rows carry one spectrum.
    assert np.allclose(power[0], power[1])


def test_a_straight_line_carries_no_power_and_manufactures_no_change():
    # Detrending a straight line leaves float error, and normalizing that error
    # into a density would hand the CUSUM a spectral shape that is not there.
    for n in (400, 1000, 1200, 2000):
        for series in (
            np.arange(float(n)),
            1000.0 * np.arange(float(n)) + 5.0,
            0.001 * np.arange(float(n)),
        ):
            power = local_periodogram(series, 40, 40, FREQUENCIES)
            assert np.all(power == 0.0)
            change = spectral_cusum(power, n_permutations=199, seed=0)
            assert change.statistic == 0.0
            assert change.p_value == 1.0


def test_the_cusum_finds_the_window_where_the_spectrum_turns():
    power = local_periodogram(two_rhythm_series(), 40, 40, FREQUENCIES)
    change = spectral_cusum(power, n_permutations=199, seed=0)
    assert change.p_value <= 0.05
    # The rhythm turns at epoch 400, which is the tenth of twenty windows.
    assert abs(change.window - 9) <= 1
    assert change.statistic > 0.5


def test_a_record_of_one_rhythm_gives_the_cusum_nothing_to_find():
    change = spectral_cusum(
        local_periodogram(one_rhythm_series(), 40, 40, FREQUENCIES), 199, 0
    )
    turning = spectral_cusum(
        local_periodogram(two_rhythm_series(), 40, 40, FREQUENCIES), 199, 0
    )
    assert isinstance(change, SpectralChange)
    assert change.p_value > 0.05
    assert turning.statistic > 10.0 * change.statistic


def test_a_constant_record_holds_one_uniform_density_and_no_change():
    power = local_periodogram(np.full(400, 5.0), 40, 40, FREQUENCIES)
    change = spectral_cusum(power, n_permutations=19, seed=0)
    assert change.statistic == 0.0
    assert change.p_value == 1.0
    assert change.as_row() == {"statistic": 0.0, "window": 0.0, "p_value": 1.0}


def test_a_periodogram_of_fewer_than_two_windows_reports_no_change():
    change = spectral_cusum(np.array([[1.0, 2.0, 3.0]]))
    assert (change.statistic, change.window, change.p_value) == (0.0, 0, 1.0)


def test_a_periodogram_of_the_wrong_shape_is_refused():
    with pytest.raises(ValueError, match="n_windows, n_frequencies"):
        spectral_cusum(np.zeros(10))
    with pytest.raises(ValueError, match="n_windows, n_frequencies"):
        spectral_cusum(np.zeros((10, 0)))


def test_a_price_drawn_from_no_reshuffle_at_all_is_refused():
    power = local_periodogram(two_rhythm_series(), 40, 40, FREQUENCIES)
    for count in (0, -3):
        with pytest.raises(ValueError, match="at least one reshuffle"):
            spectral_cusum(power, n_permutations=count)


def test_the_permutation_price_is_the_same_under_one_seed_and_close_under_another():
    power = local_periodogram(one_rhythm_series(), 40, 40, FREQUENCIES)
    first = spectral_cusum(power, n_permutations=99, seed=7)
    again = spectral_cusum(power, n_permutations=99, seed=7)
    other = spectral_cusum(power, n_permutations=99, seed=8)
    assert first.p_value == again.p_value
    assert first.statistic == other.statistic
    assert abs(first.p_value - other.p_value) < 0.2


def test_the_variance_of_the_trajectory_rises_with_the_persistence_that_moves():
    steady = rolling_autocorrelation(ar_series(0.5, 0.5, seed=5), 96, 48, 6)
    moving = rolling_autocorrelation(ar_series(0.9, 0.0, seed=5), 96, 48, 6)
    assert persistence_variance(moving) > 4.0 * persistence_variance(steady)
    assert persistence_variance(np.zeros((0, 4))) == 0.0
    assert persistence_variance(np.full((1, 4), 0.7)) == 0.0


def test_a_trajectory_of_the_wrong_shape_is_refused():
    with pytest.raises(ValueError, match="n_windows, n_lags"):
        persistence_variance(np.zeros(8))
    with pytest.raises(ValueError, match="n_windows, n_lags"):
        persistence_variance(np.zeros((8, 0)))


def test_the_rmse_separates_a_second_draw_of_one_world_from_a_different_world():
    real = rolling_autocorrelation(ar_series(0.85, 0.05, seed=10), 96, 48, 6)
    twin = rolling_autocorrelation(ar_series(0.85, 0.05, seed=11), 96, 48, 6)
    flat = rolling_autocorrelation(ar_series(0.05, 0.05, seed=11), 96, 48, 6)
    assert persistence_rmse(real, real) == 0.0
    assert persistence_rmse(real, twin) < 0.25
    assert persistence_rmse(real, flat) > 0.35


def test_the_shorter_record_sets_how_many_windows_the_rmse_compares():
    long_run = rolling_autocorrelation(ar_series(0.8, 0.8, seed=13), 96, 48, 6)
    assert persistence_rmse(long_run, long_run[:5]) == 0.0
    assert persistence_rmse(long_run, np.zeros((0, 6))) == 0.0


def test_two_trajectories_of_different_lags_are_refused():
    with pytest.raises(ValueError, match="the same lags"):
        persistence_rmse(np.zeros((5, 4)), np.zeros((5, 6)))
    with pytest.raises(ValueError, match="the same lags"):
        persistence_rmse(np.zeros((5, 4)), np.zeros(4))


def test_the_drift_scalar_is_larger_where_the_persistence_moves():
    moving = persistence_drift(ar_series(0.9, 0.0, seed=5, n=480), 24.0)
    steady = persistence_drift(ar_series(0.5, 0.5, seed=5, n=480), 24.0)
    assert moving > 0.3
    assert steady < 0.2
    assert moving > 2.0 * steady


def test_the_drift_scalar_reads_the_day_scale_on_its_own_period():
    moving = persistence_drift(ar_series(0.9, 0.0, seed=1, n=280), 7.0)
    steady = persistence_drift(ar_series(0.5, 0.5, seed=1, n=280), 7.0)
    assert moving > 0.3
    # A window of fourteen epochs leaves the estimate noisy, and the day scale
    # separates the two worlds by less than the hour scale does.
    assert moving > 1.5 * steady


def test_the_drift_scalar_is_the_deviation_of_the_lag_one_trajectory():
    series = ar_series(0.9, 0.2, seed=6, n=600)
    width, step, max_lag = window_plan(24.0)
    trajectory = rolling_autocorrelation(series, width, step, max_lag)
    assert trajectory.shape[0] > 2
    assert persistence_drift(series, 24.0) == pytest.approx(
        float(trajectory[:, 0].std()), abs=1e-15
    )


def test_a_value_that_is_not_defined_is_refused_by_every_reading():
    broken = np.concatenate([np.zeros(200), [np.nan], np.zeros(199)])
    with pytest.raises(ValueError, match="finite values only"):
        persistence_drift(broken, 24.0)
    with pytest.raises(ValueError, match="finite values only"):
        rolling_autocorrelation(broken, 48, 24, 24)
    with pytest.raises(ValueError, match="finite values only"):
        local_periodogram(np.full(400, np.inf), 40, 40, FREQUENCIES)
    # An all-undefined periodogram used to normalize to a uniform density and
    # come back as a record of one spectrum.
    with pytest.raises(ValueError, match="finite values only"):
        spectral_cusum(np.full((5, 3), np.nan))
    with pytest.raises(ValueError, match="finite values only"):
        persistence_variance(np.full((5, 4), np.nan))
    with pytest.raises(ValueError, match="finite values only"):
        persistence_rmse(np.full((5, 4), np.nan), np.zeros((5, 4)))
    with pytest.raises(ValueError, match="finite values only"):
        persistence_rmse(np.zeros((5, 4)), np.full((5, 4), -np.inf))


def test_the_drift_scalar_is_a_finite_zero_on_every_degenerate_series():
    for series in (
        np.zeros(0),
        np.zeros(1),
        np.array([5.0]),
        np.full(3, 2.0),
        np.zeros(400),
        np.full(400, 7.0),
        np.arange(400.0),
    ):
        for period in (24.0, 7.0, 1.0):
            value = persistence_drift(series, period)
            assert np.isfinite(value)
            assert value == 0.0
