"""
Tests for the complex demodulation the evolving-seasonality row waits on.

What is pinned here is that the demodulation reads back the amplitude and the
phase of a rhythm it was pointed at and reads near nothing at a period the
series does not carry, that the amplitude trajectory rises exactly when the
rhythm grows, that the unwrapped phase slides at the offset between the rhythm
and the carrier, and that the two distances separate a pair of logs sharing one
evolving rhythm from a pair where one log holds its rhythm still.  A drift of
zero is pinned to the series that hold one level, and the drift a series of pure
noise reports is pinned to the spread that noise actually carries.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.seasonality_drift import (
    AMPLITUDE_FLOOR,
    AmplitudeTrend,
    Demodulation,
    SeasonalityMatch,
    amplitude_distance,
    amplitude_drift,
    amplitude_trajectory,
    amplitude_trend,
    complex_demodulate,
    demodulation,
    lowpass_window,
    match_seasonality,
    phase_distance,
    phase_slope,
    phase_trajectory,
    unwrapped_phase,
)

DAY = 24.0
WEEK = 7.0


def pure_rhythm(
    n: int = 240, period: float = DAY, amplitude: float = 3.0, shift: float = 0.0
) -> np.ndarray:
    """One rhythm on a flat level, with no noise at all."""
    t = np.arange(float(n))
    return 10.0 + amplitude * np.cos(2.0 * np.pi * (t - shift) / period)


def noisy_rhythm(
    n: int = 240,
    period: float = DAY,
    amplitude: float = 3.0,
    seed: int = 0,
    noise: float = 0.2,
) -> np.ndarray:
    """One rhythm of a fixed size, under measurement noise."""
    rng = np.random.default_rng(seed)
    return pure_rhythm(n, period, amplitude) + rng.normal(0.0, noise, n)


def evolving_rhythm(
    first: float = 1.0, last: float = 4.0, n: int = 240, seed: int = 1
) -> np.ndarray:
    """One rhythm whose amplitude runs from `first` to `last` across the series."""
    rng = np.random.default_rng(seed)
    t = np.arange(float(n))
    gain = np.linspace(first, last, n)
    return 10.0 + gain * np.cos(2.0 * np.pi * t / DAY) + rng.normal(0.0, 0.2, n)


def test_an_even_window_removes_the_image_and_leaves_the_amplitude_flat():
    read = demodulation(pure_rhythm(), DAY)
    assert read.window == 24
    assert read.start == 12
    assert read.amplitude.size == 240 - 24
    # The boxcar of one whole period nulls the image at twice the carrier
    # exactly, so a rhythm of a fixed size reads that size at every epoch.
    assert read.amplitude == pytest.approx(3.0, abs=1e-9)


def test_an_odd_window_removes_the_image_and_leaves_the_amplitude_flat():
    read = demodulation(pure_rhythm(140, WEEK, 2.0), WEEK)
    assert read.window == 7
    assert read.start == 3
    assert read.amplitude.size == 140 - 6
    assert read.amplitude == pytest.approx(2.0, abs=1e-9)


def test_a_rhythm_the_series_does_not_carry_is_not_read_at_the_carrier():
    week_only = pure_rhythm(280, WEEK, 2.0)
    assert demodulation(week_only, WEEK).amplitude == pytest.approx(2.0, abs=1e-9)
    assert demodulation(week_only, DAY).amplitude.max() < 0.4


def test_a_constant_series_has_no_amplitude_at_any_epoch_and_does_not_drift():
    read = demodulation(np.full(120, 4.0), DAY)
    assert read.defined
    assert np.all(read.amplitude == 0.0)
    assert amplitude_drift(np.full(120, 4.0), DAY) == 0.0


def test_a_growing_rhythm_rises_and_a_fading_one_falls_by_what_was_built_in():
    # The trajectory opens half a window in, so its first quarter averages a
    # gain of about 1.48 and its last quarter about 3.52.
    assert amplitude_drift(evolving_rhythm(1.0, 4.0), DAY) == pytest.approx(1.37, 0.15)
    assert amplitude_drift(evolving_rhythm(4.0, 1.0), DAY) == pytest.approx(-0.58, 0.15)


def test_a_rhythm_that_holds_its_size_drifts_by_almost_nothing():
    for seed in range(4):
        assert abs(amplitude_drift(noisy_rhythm(seed=seed), DAY)) < 0.06


def test_a_window_stride_clears_the_trend_a_filtered_trajectory_invents():
    steady = demodulation(noisy_rhythm(seed=0), DAY)
    # Neighbouring epochs of a moving average share most of their window, so
    # every epoch tested reads the wobble of a steady rhythm as a trend.
    assert amplitude_trend(steady.amplitude).p_value < 0.05
    assert amplitude_trend(steady.amplitude, steady.window).p_value > 0.1
    growing = demodulation(evolving_rhythm(1.0, 4.0), DAY)
    trend = amplitude_trend(growing.amplitude, growing.window)
    assert trend.z > 3.0
    assert trend.p_value < 0.001
    assert trend.tau == pytest.approx(1.0)


def test_a_fading_rhythm_carries_the_trend_with_the_opposite_sign():
    fading = demodulation(evolving_rhythm(4.0, 1.0), DAY)
    trend = amplitude_trend(fading.amplitude, fading.window)
    assert trend.s < 0.0
    assert trend.z < -3.0
    assert trend.tau == pytest.approx(-1.0)


def test_a_trajectory_of_one_repeated_value_has_no_trend_to_report():
    trend = amplitude_trend(np.full(20, 2.0))
    assert isinstance(trend, AmplitudeTrend)
    assert (trend.s, trend.tau, trend.z, trend.p_value) == (0.0, 0.0, 0.0, 1.0)


def test_a_trajectory_of_fewer_than_three_tested_points_leaves_the_trend_undecided():
    assert amplitude_trend([1.0, 2.0]).p_value == 1.0
    # Ten rising epochs read at a stride of five leave two points to compare.
    assert amplitude_trend(np.arange(10.0), stride=5).s == 0.0
    assert amplitude_trend(np.arange(10.0)).s == 45.0


def test_a_trend_stride_below_one_epoch_is_refused():
    with pytest.raises(ValueError, match="at least one epoch; got 0"):
        amplitude_trend(np.arange(10.0), stride=0)


def test_a_rhythm_slower_than_the_carrier_slides_its_phase_backwards():
    for period in (22.0, 26.0):
        offset = demodulation(
            noisy_rhythm(480, period, noise=0.05), DAY
        ).frequency_offset
        assert offset == pytest.approx(2.0 * np.pi * (1.0 / period - 1.0 / DAY), 0.02)
    assert demodulation(noisy_rhythm(480, DAY, noise=0.05), DAY).frequency_offset == (
        pytest.approx(0.0, abs=1e-3)
    )


def test_the_unwrapped_phase_climbs_past_the_branch_the_wrapped_one_folds_at():
    read = demodulation(noisy_rhythm(480, 26.0, noise=0.05), DAY)
    assert np.abs(np.diff(read.phase)).max() > np.pi
    assert np.abs(np.diff(read.unwrapped)).max() < 0.5
    span = float(read.unwrapped[-1] - read.unwrapped[0])
    assert span == pytest.approx(
        read.frequency_offset * (read.unwrapped.size - 1), 0.05
    )


def test_the_phase_trajectory_reads_where_in_the_day_the_rhythm_peaks():
    # A peak six epochs later on a period of 24 is a quarter turn of phase.
    assert phase_trajectory(complex_demodulate(pure_rhythm(), DAY)) == pytest.approx(
        0.0, abs=1e-9
    )
    late = complex_demodulate(pure_rhythm(shift=6.0), DAY)
    assert phase_trajectory(late) == pytest.approx(-np.pi / 2.0, abs=1e-9)
    assert amplitude_trajectory(late) == pytest.approx(3.0, abs=1e-9)


def test_the_phase_slope_of_a_single_point_is_zero():
    assert phase_slope([0.7]) == 0.0
    assert phase_slope(unwrapped_phase(np.zeros(0))) == 0.0
    assert phase_slope([0.0, 0.5, 1.0, 1.5]) == pytest.approx(0.5)


def test_two_logs_sharing_one_evolving_rhythm_match_and_a_steady_log_does_not():
    same = match_seasonality(
        evolving_rhythm(1.0, 4.0, seed=1), evolving_rhythm(1.0, 4.0, seed=11), DAY
    )
    assert isinstance(same, SeasonalityMatch)
    assert same.defined
    assert same.amplitude_distance < 0.1
    assert same.phase_distance < 0.1
    assert same.drift_gap < 0.25
    apart = match_seasonality(
        evolving_rhythm(1.0, 4.0, seed=1), noisy_rhythm(seed=11), DAY
    )
    assert apart.amplitude_distance > 3.0 * same.amplitude_distance
    assert apart.drift_gap > 1.0
    assert set(apart.as_row()) == {
        "amplitude_distance",
        "phase_distance",
        "drift_gap",
        "offset_gap",
        "defined",
    }


def test_a_pair_whose_rhythms_slide_apart_is_split_by_the_phase_distance():
    apart = match_seasonality(pure_rhythm(480), pure_rhythm(480, shift=6.0), DAY)
    assert apart.phase_distance == pytest.approx(np.pi / 2.0, abs=1e-6)
    assert apart.amplitude_distance == pytest.approx(0.0, abs=1e-9)
    assert apart.offset_gap == pytest.approx(0.0, abs=1e-9)


def test_a_pair_too_short_to_read_reports_agreement_under_an_undefined_flag():
    match = match_seasonality(np.zeros(10), pure_rhythm(10), DAY)
    assert not match.defined
    assert (match.amplitude_distance, match.phase_distance) == (0.0, 0.0)
    assert (match.drift_gap, match.offset_gap) == (0.0, 0.0)


def test_the_amplitude_distance_is_the_gap_in_units_of_the_amplitude_itself():
    full = demodulation(pure_rhythm(), DAY).amplitude
    half = demodulation(pure_rhythm(amplitude=1.5), DAY).amplitude
    assert amplitude_distance(full, full) == 0.0
    # A gap of 1.5 against a mean amplitude of 2.25.
    assert amplitude_distance(full, half) == pytest.approx(2.0 / 3.0, abs=1e-6)
    # The shared span decides, so a shorter trajectory is compared on its own.
    assert amplitude_distance(full, half[:50]) == pytest.approx(2.0 / 3.0, abs=1e-6)
    assert amplitude_distance(np.zeros(30), np.zeros(30)) == 0.0
    assert amplitude_distance(np.zeros(0), full) == 0.0


def test_the_phase_distance_takes_the_shorter_way_round_the_circle():
    late = np.full(20, 2.0 * np.pi - 0.1)
    assert phase_distance(np.full(20, 0.1), late) == pytest.approx(0.2)
    assert phase_distance(np.full(20, 0.1), np.full(20, 0.1)) == 0.0
    assert phase_distance(np.zeros(0), np.zeros(0)) == 0.0
    # Half a turn apart is the furthest two phases get.
    assert phase_distance(np.zeros(20), np.full(20, np.pi)) == pytest.approx(np.pi)


def test_a_series_shorter_than_two_periods_reads_as_flat_and_undefined():
    read = demodulation(noisy_rhythm(30), DAY)
    assert isinstance(read, Demodulation)
    assert not read.defined
    assert read.amplitude.size == 30
    assert np.all(read.amplitude == 0.0)
    assert np.all(read.phase == 0.0)
    assert np.all(read.demodulate == 0.0)
    assert read.frequency_offset == 0.0
    assert amplitude_drift(noisy_rhythm(30), DAY) == 0.0


def test_an_empty_series_is_undefined_and_drifts_by_nothing():
    read = demodulation(np.zeros(0), DAY)
    assert not read.defined
    assert read.amplitude.size == 0
    assert amplitude_drift([], DAY) == 0.0


def test_a_trajectory_of_fewer_than_four_epochs_drifts_by_nothing():
    # Three epochs at a period of 1.5 are two whole periods, so the series is
    # read and its trajectory is one epoch long, which holds no quarter.
    assert demodulation([1.0, 2.0, 3.0], 1.5).defined
    assert amplitude_drift([1.0, 2.0, 3.0], 1.5) == 0.0


def test_a_rhythm_growing_out_of_silence_reports_a_large_finite_drift():
    t = np.arange(240.0)
    gain = np.where(t < 120.0, 0.0, np.linspace(0.0, 4.0, 240))
    silent_first = 10.0 + gain * np.cos(2.0 * np.pi * t / DAY)
    read = demodulation(silent_first, DAY)
    quarter = read.amplitude.size // 4
    # The first quarter of the trajectory is exactly silent, so the floor on
    # the denominator is the only thing holding the ratio finite, and the drift
    # is the last quarter measured against that floor.
    assert read.amplitude[:quarter].max() < 1e-12
    drift = amplitude_drift(silent_first, DAY)
    assert drift == pytest.approx(
        read.amplitude[-quarter:].mean() / (AMPLITUDE_FLOOR * read.amplitude.max())
    )
    assert drift == pytest.approx(886.33, abs=0.01)


def test_a_series_with_no_rhythm_under_it_drifts_by_what_its_noise_holds():
    # A drift is stated against the amplitude the trajectory opens with, so a
    # series of pure noise reports the wander of its own noise and reaches a
    # few tenths in either direction.  Only a series holding one level reads
    # exactly zero.
    drifts = [
        amplitude_drift(np.random.default_rng(seed).normal(0.0, 1.0, 480), DAY)
        for seed in range(20)
    ]
    assert max(drifts) == pytest.approx(0.530, abs=0.01)
    assert min(drifts) == pytest.approx(-0.271, abs=0.01)


def test_a_series_no_longer_than_the_window_is_refused_by_the_demodulate():
    with pytest.raises(ValueError, match="more epochs than its window; got 24"):
        complex_demodulate(np.zeros(24), DAY)


def test_a_series_holding_a_non_finite_value_is_refused():
    with pytest.raises(ValueError, match="is finite; got nan"):
        demodulation([1.0, np.nan, 3.0], DAY)


def test_a_trajectory_holding_a_non_finite_value_is_refused_by_every_reader():
    bad = np.array([1.0, np.nan, 2.0, 3.0])
    good = np.ones(4)
    readers = [
        lambda: amplitude_trajectory(bad),
        lambda: phase_trajectory(bad),
        lambda: unwrapped_phase(bad),
        lambda: phase_slope(bad),
        lambda: amplitude_trend(bad),
        lambda: amplitude_distance(bad, good),
        lambda: amplitude_distance(good, bad),
        lambda: phase_distance(bad, good),
        lambda: phase_distance(good, bad),
    ]
    for reader in readers:
        with pytest.raises(ValueError, match="is finite; got"):
            reader()
    # An infinity names itself the same way a nan does.
    with pytest.raises(ValueError, match="is finite; got inf"):
        phase_slope([0.0, np.inf])
    with pytest.raises(ValueError, match="is finite; got inf"):
        amplitude_trend([1.0, 2.0, np.inf])


def test_a_period_at_or_below_one_epoch_is_refused():
    for period in (1.0, 0.0, -3.0, np.inf):
        with pytest.raises(ValueError, match="above one epoch"):
            amplitude_drift(pure_rhythm(), period)


def test_a_low_pass_window_holding_less_than_one_period_is_refused():
    with pytest.raises(ValueError, match="at least one whole period; got 0"):
        complex_demodulate(pure_rhythm(), DAY, cycles=0)


def test_the_low_pass_window_spans_a_whole_number_of_periods():
    assert lowpass_window(DAY) == 24
    assert lowpass_window(WEEK) == 7
    assert lowpass_window(DAY, 2) == 48
    # A period between one and two epochs still needs two epochs to average.
    assert lowpass_window(1.2) == 2
