"""
Tests for the within-parent placement row of the nonstationarity grid.

What is pinned here is that the circular figures find a placement the epoch
totals are blind to, that they report no concentration where the events are
scattered through the block, that a position at the end of a block sits beside
one at its start, that the profile comparisons separate two logs placing their
events the same way from two that place them twelve hours apart, that the
circular figures and the Rayleigh p-value carry the values their definitions
give, and that every degenerate series returns one finite scalar.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.evaluation.placement import (
    DEFAULT_BINS,
    CircularStats,
    PositionProfile,
    Rayleigh,
    chi_square,
    circular_mean_distance,
    circular_stats,
    epoch_blocks,
    placement_concentration,
    position_profile,
    rayleigh_test,
    total_variation,
    within_parent_positions,
)

HOURS = 24.0
DAYS = 30


def clustered_day(hour: float, spread: float, seed: int = 0) -> np.ndarray:
    """Hourly counts over whole days, each day's eight events near one hour."""
    rng = np.random.default_rng(seed)
    counts = np.zeros(DAYS * int(HOURS))
    for day in range(DAYS):
        for value in rng.normal(hour, spread, 8):
            counts[day * int(HOURS) + int(np.floor(value)) % int(HOURS)] += 1.0
    return counts


def scattered_day(seed: int = 1) -> np.ndarray:
    """Hourly counts over whole days, each day's eight events at any hour."""
    rng = np.random.default_rng(seed)
    counts = np.zeros(DAYS * int(HOURS))
    for day in range(DAYS):
        for hour in rng.integers(0, int(HOURS), 8):
            counts[day * int(HOURS) + int(hour)] += 1.0
    return counts


def hour_of(mean_angle: float) -> float:
    """The hour of the day one mean angle points at."""
    return mean_angle / (2.0 * math.pi) * HOURS


def test_the_circular_mean_lands_on_the_hour_the_events_sit_at():
    counts = clustered_day(9.5, 0.5)
    _, positions = epoch_blocks(counts.size, HOURS)
    stats = circular_stats(positions, counts)
    assert isinstance(stats, CircularStats)
    assert stats.defined
    assert hour_of(stats.mean) == pytest.approx(9.0, abs=0.5)
    assert stats.resultant > 0.9
    assert stats.variance < 0.1


def test_events_scattered_through_the_day_leave_almost_no_concentration():
    assert placement_concentration(clustered_day(9.5, 0.5), HOURS) > 0.9
    assert placement_concentration(scattered_day(), HOURS) < 0.15


def test_a_constant_series_spreads_its_events_evenly_and_declares_no_mean():
    flat = np.full(DAYS * int(HOURS), 3.0)
    assert placement_concentration(flat, HOURS) == pytest.approx(0.0, abs=1e-12)
    _, positions = epoch_blocks(flat.size, HOURS)
    stats = circular_stats(positions, flat)
    assert not stats.defined
    assert stats.mean == 0.0
    assert stats.variance == 1.0


def test_the_concentration_rises_as_the_placement_tightens():
    values = [
        placement_concentration(clustered_day(9.5, spread), HOURS)
        for spread in (6.0, 3.0, 1.0, 0.25)
    ]
    assert values == sorted(values)
    assert values[0] < 0.5 < values[-1]


def test_the_rayleigh_test_rejects_a_clustered_day_and_passes_a_scattered_one():
    _, positions = epoch_blocks(DAYS * int(HOURS), HOURS)
    clustered = rayleigh_test(positions, clustered_day(9.5, 1.0))
    scattered = rayleigh_test(positions, scattered_day())
    assert isinstance(clustered, Rayleigh)
    assert clustered.p_value < 1e-6
    assert clustered.z > scattered.z
    assert scattered.p_value > 0.05


def test_a_position_at_the_end_of_a_block_sits_beside_one_at_its_start():
    stats = circular_stats([0.98, 0.02])
    assert stats.resultant > 0.99
    assert circular_mean_distance(stats.mean, 0.0) < 0.01
    # Midnight and noon are the width of a day apart, which is the widest gap.
    assert circular_mean_distance(0.0, math.pi) == pytest.approx(math.pi)


def test_two_logs_placing_events_the_same_way_agree_and_a_shifted_one_does_not():
    parents, positions = epoch_blocks(DAYS * int(HOURS), HOURS)
    morning = clustered_day(9.5, 1.0, seed=0)
    second_morning = clustered_day(9.5, 1.0, seed=5)
    evening = clustered_day(21.5, 1.0, seed=6)
    profiles = [
        position_profile(counts, parents, int(HOURS))
        for counts in (morning, second_morning, evening)
    ]
    agree = total_variation(profiles[0].density, profiles[1].density)
    disagree = total_variation(profiles[0].density, profiles[2].density)
    assert agree < 0.2 < 0.9 < disagree
    assert chi_square(profiles[0].density, profiles[1].density) < 0.5
    assert chi_square(profiles[0].density, profiles[2].density) > 1.5
    means = [circular_stats(positions, counts).mean for counts in (morning, evening)]
    assert circular_mean_distance(means[0], means[0]) == 0.0
    assert circular_mean_distance(means[0], means[1]) > 2.8


def test_the_two_logs_that_disagree_on_placement_hold_the_same_daily_total():
    morning = clustered_day(9.5, 1.0, seed=0)
    evening = clustered_day(21.5, 1.0, seed=6)
    daily = [
        counts.reshape(DAYS, int(HOURS)).sum(axis=1) for counts in (morning, evening)
    ]
    assert daily[0].tolist() == daily[1].tolist()
    assert placement_concentration(morning, HOURS) > 0.5
    assert placement_concentration(evening, HOURS) > 0.5


def test_a_series_with_no_events_gives_a_uniform_profile_declared_undefined():
    parents, positions = epoch_blocks(48, HOURS)
    empty = np.zeros(48)
    profile = position_profile(empty, parents, 6)
    assert isinstance(profile, PositionProfile)
    assert profile.density.tolist() == [1.0 / 6.0] * 6
    assert profile.weight == 0.0
    assert not profile.defined
    assert placement_concentration(empty, HOURS) == 0.0
    assert rayleigh_test(positions, empty).p_value == 1.0


def test_one_event_concentrates_fully_and_carries_no_evidence_of_clustering():
    counts = np.zeros(int(HOURS))
    counts[9] = 1.0
    _, positions = epoch_blocks(counts.size, HOURS)
    test = rayleigh_test(positions, counts)
    assert placement_concentration(counts, HOURS) == pytest.approx(1.0)
    assert test.resultant == pytest.approx(1.0)
    assert test.z == 0.0
    assert test.p_value == 1.0


def test_the_profile_puts_its_mass_in_the_bin_the_events_sit_in():
    parents, _ = epoch_blocks(DAYS * int(HOURS), HOURS)
    profile = position_profile(clustered_day(9.5, 0.5), parents, int(HOURS))
    assert profile.defined
    assert profile.weight == float(DAYS * 8)
    assert profile.density.sum() == pytest.approx(1.0)
    assert int(np.argmax(profile.density)) == 9
    assert profile.density[9] > 0.5
    assert profile.density[18:].sum() == 0.0


def test_the_default_binning_splits_the_parent_block_into_twelve():
    parents, _ = epoch_blocks(DAYS * int(HOURS), HOURS)
    profile = position_profile(clustered_day(9.5, 0.5), parents)
    assert DEFAULT_BINS == 12
    assert profile.density.size == 12
    # Twelve bins of a day are two hours each, so nine in the morning is bin four.
    assert int(np.argmax(profile.density)) == 4


def test_the_hour_scale_blocks_number_the_days_and_place_nine_in_the_morning():
    parents, positions = epoch_blocks(48, HOURS)
    assert parents[:24].tolist() == [0] * 24
    assert parents[24:].tolist() == [1] * 24
    assert positions[9] == pytest.approx(0.375)
    assert positions[33] == pytest.approx(0.375)


def test_a_truncated_final_block_keeps_the_scale_of_a_whole_block():
    parents, positions = epoch_blocks(30, HOURS)
    assert int(parents[29]) == 1
    assert positions[24] == 0.0
    assert positions[29] == pytest.approx(5.0 / 24.0)
    # A parent map read on its own knows only the six slots the block holds.
    assert within_parent_positions(parents)[29] == pytest.approx(5.0 / 6.0)


def test_the_within_parent_position_is_the_rank_inside_its_own_parent():
    positions = within_parent_positions([0, 0, 0, 0, 1, 1])
    assert positions.tolist() == [0.0, 0.25, 0.5, 0.75, 0.0, 0.5]
    assert within_parent_positions([]).size == 0
    assert within_parent_positions([7]).tolist() == [0.0]


def test_a_parent_map_that_steps_backwards_is_refused():
    with pytest.raises(ValueError, match="non-decreasing"):
        within_parent_positions([0, 0, 1, 0])


def test_a_profile_of_fewer_than_two_bins_is_refused():
    with pytest.raises(ValueError, match="at least two bins"):
        position_profile(np.ones(4), [0, 0, 0, 0], 1)


def test_a_negative_count_and_a_count_that_is_not_finite_are_refused():
    with pytest.raises(ValueError, match="not negative"):
        placement_concentration([1.0, -2.0], 2.0)
    with pytest.raises(ValueError, match="is finite"):
        placement_concentration([1.0, np.nan], 2.0)


def test_a_parent_block_of_no_epochs_and_a_series_of_negative_length_are_refused():
    with pytest.raises(ValueError, match="at least one whole epoch"):
        placement_concentration([1.0, 2.0], 0.0)
    with pytest.raises(ValueError, match="at least one whole epoch"):
        epoch_blocks(10, float("inf"))
    with pytest.raises(ValueError, match="no negative count"):
        epoch_blocks(-1, HOURS)


def test_arrays_that_do_not_line_up_are_refused():
    with pytest.raises(ValueError, match="beside its parent"):
        position_profile(np.ones(4), [0, 0, 0])
    with pytest.raises(ValueError, match="beside its position"):
        circular_stats([0.1, 0.2], [1.0])
    with pytest.raises(ValueError, match="one binning"):
        total_variation([0.5, 0.5], [1.0])
    with pytest.raises(ValueError, match="at least one bin"):
        chi_square([], [])


def test_the_distances_are_zero_between_equal_profiles_and_largest_between_disjoint():
    left = np.array([0.5, 0.5, 0.0, 0.0])
    right = np.array([0.0, 0.0, 0.5, 0.5])
    assert total_variation(left, left) == 0.0
    assert chi_square(left, left) == 0.0
    assert total_variation(left, right) == pytest.approx(1.0)
    assert chi_square(left, right) == pytest.approx(2.0)


def test_the_chi_square_grows_as_two_profiles_move_apart():
    left = np.array([0.4, 0.4, 0.1, 0.1])
    near = np.array([0.35, 0.45, 0.1, 0.1])
    far = np.array([0.1, 0.1, 0.4, 0.4])
    assert 0.0 < chi_square(left, near) < chi_square(left, far)


def test_the_rayleigh_p_value_stays_inside_the_unit_interval_on_every_draw():
    rng = np.random.default_rng(11)
    for trial in range(40):
        positions = rng.beta(1.0, 1.0 + trial, size=25)
        weights = rng.integers(0, 4, size=25).astype(float)
        for test in (rayleigh_test(positions), rayleigh_test(positions, weights)):
            assert math.isfinite(test.p_value)
            assert 0.0 <= test.p_value <= 1.0


def test_the_summary_scalar_is_one_finite_share_on_every_degenerate_series():
    series = [
        np.zeros(0),
        np.zeros(int(HOURS)),
        np.full(int(HOURS), 5.0),
        np.array([3.0]),
        np.full(30, 2.0),
    ]
    for counts in series:
        value = placement_concentration(counts, HOURS)
        assert math.isfinite(value)
        assert 0.0 <= value <= 1.0
    assert placement_concentration(np.zeros(0), HOURS) == 0.0
    assert placement_concentration(np.array([3.0]), HOURS) == pytest.approx(1.0)


def test_the_circular_figures_carry_their_analytic_values():
    # Two positions a quarter of the block apart resolve to the angle between
    # them, at the length the cosine of half their separation gives.
    quarter = circular_stats([0.0, 0.25])
    assert quarter.mean == pytest.approx(math.pi / 4.0)
    assert quarter.resultant == pytest.approx(math.sqrt(0.5))
    assert quarter.variance == pytest.approx(1.0 - math.sqrt(0.5))
    # Two positions half a block apart cancel exactly.
    antipodal = circular_stats([0.0, 0.5])
    assert antipodal.resultant == pytest.approx(0.0, abs=1e-15)
    assert antipodal.variance == 1.0
    assert not antipodal.defined
    # Three at the start against one half a block away leave half the weight.
    weighted = circular_stats([0.0, 0.5], [3.0, 1.0])
    assert weighted.resultant == pytest.approx(0.5)
    assert weighted.mean == pytest.approx(0.0, abs=1e-12)


def two_sided(n: float, z: float) -> Rayleigh:
    """The Rayleigh test of n events whose resultant carries the statistic z."""
    offset = math.acos(math.sqrt(z / n)) / (2.0 * math.pi)
    return rayleigh_test([offset, 1.0 - offset], [n / 2.0, n / 2.0])


def test_the_rayleigh_p_value_lands_on_the_published_critical_points():
    # The statistics that sit at the five percent point of the Rayleigh test,
    # as tabulated for these event counts.  Mardia's series has to price them
    # there, which a series carrying the wrong correction term does not.
    for n, z in ((10.0, 2.945), (20.0, 2.971), (50.0, 2.981), (100.0, 2.988)):
        test = two_sided(n, z)
        assert test.z == pytest.approx(z)
        assert test.p_value == pytest.approx(0.05, abs=0.002)
    # The correction dies away with the event count, leaving exp(-z).
    assert two_sided(1.0e6, 3.0).p_value == pytest.approx(math.exp(-3.0), rel=1e-4)


def test_a_statistic_past_the_reach_of_the_series_reads_a_p_value_of_zero():
    test = rayleigh_test([0.3, 0.3], [400.0, 400.0])
    assert test.z == pytest.approx(800.0)
    assert test.p_value == 0.0
    # The fourth power of the statistic overflows a float well before this.
    assert rayleigh_test([0.3, 0.3], [1e160, 1e160]).p_value == 0.0


def test_an_argument_of_more_than_one_axis_is_refused():
    with pytest.raises(ValueError, match="a count runs along one axis"):
        placement_concentration(np.ones((4, 1)), HOURS)
    with pytest.raises(ValueError, match="a position runs along one axis"):
        circular_stats(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="a parent map runs along one axis"):
        within_parent_positions(np.zeros((2, 2), dtype=np.int64))
    with pytest.raises(ValueError, match="a profile bin runs along one axis"):
        total_variation(np.ones((2, 2)), np.ones((2, 2)))


def test_a_position_and_a_mean_angle_that_are_not_finite_are_refused():
    with pytest.raises(ValueError, match="a position is finite"):
        circular_stats([0.1, float("nan")])
    with pytest.raises(ValueError, match="a position is finite"):
        rayleigh_test([0.1, float("inf")])
    with pytest.raises(ValueError, match="a mean angle is finite"):
        circular_mean_distance(float("nan"), 0.0)
    with pytest.raises(ValueError, match="a mean angle is finite"):
        circular_mean_distance(0.0, float("inf"))


def test_a_profile_bin_that_is_not_finite_or_holds_negative_mass_is_refused():
    with pytest.raises(ValueError, match="a profile bin is finite"):
        chi_square([0.5, float("nan")], [0.5, 0.5])
    with pytest.raises(ValueError, match="a profile bin is finite"):
        total_variation([0.5, 0.5], [0.5, float("inf")])
    with pytest.raises(ValueError, match="no negative mass"):
        chi_square([1.0, -1.0], [-1.0, 1.0])
    with pytest.raises(ValueError, match="no negative mass"):
        total_variation([0.5, 0.5], [1.5, -0.5])


def test_a_total_that_overflows_its_own_type_is_refused():
    with pytest.raises(ValueError, match="finite weight"):
        placement_concentration(np.full(4, 1e308), HOURS)
    with pytest.raises(ValueError, match="finite mass"):
        total_variation([1e308, 0.0], [0.0, 1e308])
    with pytest.raises(ValueError, match="finite mass"):
        chi_square([1e308, 1e308], [0.0, 0.0])
    # Inside that bound the divergence stays on the scale of the mass itself.
    assert chi_square([1e200, 0.0], [0.0, 1e200]) == pytest.approx(2e200)


def test_a_parent_block_shorter_than_one_epoch_is_refused():
    with pytest.raises(ValueError, match="at least one whole epoch"):
        epoch_blocks(3, 0.5)
    with pytest.raises(ValueError, match="at least one whole epoch"):
        placement_concentration([1.0, 2.0], 1e-300)


def test_the_day_scale_reads_a_week_the_same_way_the_hour_scale_reads_a_day():
    week = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 6.0] * 8)
    even = np.full(56, 4.0)
    assert placement_concentration(week, 7.0) > 0.85
    assert placement_concentration(even, 7.0) == pytest.approx(0.0, abs=1e-12)
