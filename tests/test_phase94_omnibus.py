"""
Tests for the characteristic-kernel omnibus and the common changepoint.

Two rows of the grid wait on what is pinned here: that the discrepancy sits at
zero on two draws of one distribution and finds any two that differ, that the
delay embedding is what makes an ordering visible to it, that a common cut is
bought by what a panel does jointly, and that the full covariance cost prices a
change in dependence the diagonal cost holds fixed.  The degenerate cases are
pinned beside them, since a constant series and an empty panel both reach these
functions from the grid.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.omnibus import (
    DIAGONAL,
    EMBED_LENGTH,
    FULL,
    PRICES,
    _prefix,
    common_penalty_path,
    common_segment_cost,
    cost_drop_fraction,
    delay_embed,
    gaussian_kernel,
    median_bandwidth,
    min_segment,
    mmd2_unbiased,
    mmd_permutation_p,
    partition_cost,
    segment_common,
    window_mmd,
)


def draw(seed: int, n: int = 240, mean: float = 0.0, spread: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(mean, spread, n)


def autocorrelated(phi: float = 0.9, n: int = 240, seed: int = 5) -> np.ndarray:
    """A series whose neighbouring epochs agree, so its ordering carries structure."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + noise[t]
    return x


def common_step(
    delta: float, seed: int = 2, n: int = 200, width: int = 4
) -> np.ndarray:
    """A panel whose every series steps by the same amount at the midpoint."""
    rng = np.random.default_rng(seed)
    panel = rng.normal(0.0, 1.0, (n, width))
    panel[n // 2 :, :] += delta
    return panel


def lone_step(delta: float, seed: int = 3, n: int = 200, width: int = 4) -> np.ndarray:
    """A panel where one series of four steps and the other three hold."""
    rng = np.random.default_rng(seed)
    panel = rng.normal(0.0, 1.0, (n, width))
    panel[n // 2 :, 0] += delta
    return panel


def flipped_dependence(seed: int = 4, n: int = 200) -> np.ndarray:
    """Two series whose correlation turns from plus to minus at the midpoint.

    Each series is standard normal over the whole record, so every marginal
    moment holds across the turn.
    """
    rng = np.random.default_rng(seed)
    first = rng.normal(0.0, 1.0, n)
    noise = rng.normal(0.0, 1.0, n)
    rho = np.where(np.arange(n) < n // 2, 0.9, -0.9)
    return np.column_stack([first, rho * first + np.sqrt(1.0 - 0.81) * noise])


def null_panel(seed: int = 9, n: int = 200, width: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, (n, width))


# ---------------------------------------------------------------------------
# A16
# ---------------------------------------------------------------------------


def test_the_bandwidth_is_the_median_distance_between_two_points_of_the_pool():
    # The pooled distances are 2, 4 and 2.
    assert median_bandwidth(np.array([[0.0], [2.0]]), np.array([[4.0]])) == 2.0


def test_the_bandwidth_is_the_median_of_an_even_number_of_pooled_distances():
    # The pool 0, 1, 4, 10 carries six distances: 1, 3, 4, 6, 9 and 10.  Their
    # median is 5, the mean of the two central ones.  The root of the median
    # squared distance is 5.0990, so an even pool separates the two.
    pooled = median_bandwidth(np.array([[0.0], [1.0], [4.0]]), np.array([[10.0]]))
    assert pooled == pytest.approx(5.0)


def test_a_pool_of_one_distinct_point_takes_a_bandwidth_of_one():
    assert median_bandwidth(np.zeros((3, 1)), np.zeros((2, 1))) == 1.0
    assert median_bandwidth(np.array([[1.0]]), np.zeros((0, 1))) == 1.0


def test_a_bandwidth_of_zero_has_no_kernel():
    with pytest.raises(ValueError, match="bandwidth is positive"):
        gaussian_kernel(np.zeros((2, 1)), np.zeros((2, 1)), 0.0)


def test_the_kernel_is_one_between_a_point_and_itself_and_falls_away_with_distance():
    points = np.array([[0.0], [1.0], [3.0]])
    kernel = gaussian_kernel(points, points, 1.0)
    assert np.diag(kernel) == pytest.approx([1.0, 1.0, 1.0])
    assert 1.0 > kernel[0, 1] > kernel[0, 2] > 0.0


def test_the_estimator_is_the_unbiased_one_and_drops_both_diagonals():
    a = np.array([[0.0], [1.0]])
    b = np.array([[3.0], [5.0]])
    width = 2.0

    def kernel(p: float, q: float) -> float:
        return float(np.exp(-((p - q) ** 2) / (2.0 * width * width)))

    # Each within-sample mean runs over its one off-diagonal pair, and the cross
    # term over all four pairs.
    hand = (
        kernel(0.0, 1.0)
        + kernel(3.0, 5.0)
        - 0.5
        * (kernel(0.0, 3.0) + kernel(0.0, 5.0) + kernel(1.0, 3.0) + kernel(1.0, 5.0))
    )
    assert mmd2_unbiased(a, b, width) == pytest.approx(hand)
    # The V-statistic keeps the two diagonals and lands elsewhere, which is what
    # separates the two estimators on a sample this small.
    biased = float(
        gaussian_kernel(a, a, width).mean()
        + gaussian_kernel(b, b, width).mean()
        - 2.0 * gaussian_kernel(a, b, width).mean()
    )
    assert abs(biased - hand) > 0.1


def test_two_draws_of_one_distribution_leave_the_discrepancy_at_zero():
    a = delay_embed(draw(0), EMBED_LENGTH)
    b = delay_embed(draw(1), EMBED_LENGTH)
    statistic, p_value = mmd_permutation_p(a, b, n_permutations=99, seed=0)
    assert abs(statistic) < 0.02
    assert p_value > 0.05


def test_the_discrepancy_finds_two_distributions_that_differ():
    a = delay_embed(draw(0), EMBED_LENGTH)
    b = delay_embed(draw(1, mean=1.5), EMBED_LENGTH)
    statistic, p_value = mmd_permutation_p(a, b, n_permutations=99, seed=0)
    assert statistic > 0.2
    assert p_value <= 0.05


def test_the_delay_embedding_is_what_makes_an_ordering_visible():
    ordered = autocorrelated()
    reordered = np.random.default_rng(6).permutation(ordered)
    values = mmd_permutation_p(
        delay_embed(ordered, 1), delay_embed(reordered, 1), 99, seed=0
    )
    windows = mmd_permutation_p(
        delay_embed(ordered, EMBED_LENGTH),
        delay_embed(reordered, EMBED_LENGTH),
        99,
        seed=0,
    )
    # One histogram carried in two orders: the values alone agree.
    assert abs(values[0]) < 0.01
    assert values[1] > 0.5
    assert windows[0] > 0.02
    assert windows[1] <= 0.05


def test_a_sample_of_one_point_carries_no_unbiased_discrepancy():
    lone = np.array([[0.0]])
    pair = np.array([[1.0], [2.0]])
    assert mmd2_unbiased(lone, pair) == 0.0
    assert mmd_permutation_p(lone, pair, 9, seed=0) == (0.0, 1.0)


def test_two_samples_of_different_coordinates_are_refused():
    with pytest.raises(ValueError, match="share their coordinates"):
        mmd2_unbiased(np.zeros((4, 2)), np.zeros((4, 3)))


def test_a_sample_of_three_axes_is_refused():
    with pytest.raises(ValueError, match="one or two axes"):
        mmd2_unbiased(np.zeros((2, 2, 2)), np.zeros((4, 2)))


def test_a_permutation_null_of_no_draws_is_refused():
    with pytest.raises(ValueError, match="at least one draw"):
        mmd_permutation_p(np.zeros((4, 1)), np.ones((4, 1)), n_permutations=0)


def test_the_delay_embedding_holds_the_consecutive_windows_of_the_series():
    assert delay_embed([1.0, 2.0, 3.0, 4.0], 2).tolist() == [
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
    ]
    assert delay_embed([1.0, 2.0, 3.0]).tolist() == [[1.0, 2.0, 3.0]]


def test_a_series_shorter_than_one_window_carries_no_point_to_compare():
    short = delay_embed([1.0, 2.0], 3)
    assert short.shape == (0, 3)
    assert mmd2_unbiased(short, delay_embed(draw(0), 3)) == 0.0


def test_a_sample_that_is_not_finite_is_refused():
    with pytest.raises(ValueError, match="finite points"):
        mmd2_unbiased(np.array([[0.0], [np.nan]]), np.zeros((2, 1)))
    with pytest.raises(ValueError, match="finite points"):
        window_mmd(np.concatenate([np.arange(19.0), [np.inf]]), 24.0)


def test_a_series_of_more_than_one_axis_carries_no_delay_window():
    with pytest.raises(ValueError, match="one axis"):
        delay_embed(np.zeros((6, 2)))


def test_a_delay_window_of_no_epoch_is_refused():
    with pytest.raises(ValueError, match="at least one epoch"):
        delay_embed([1.0, 2.0, 3.0], 0)


def test_the_omnibus_finds_a_move_that_leaves_the_mean_where_it_was():
    moved = np.concatenate([draw(11, 120), draw(12, 120, spread=3.0)])
    halves = (moved[:120].mean(), moved[120:].mean())
    assert abs(halves[0] - halves[1]) < 0.3
    assert window_mmd(moved, 24.0) > 0.1
    assert window_mmd(draw(13), 24.0) < 0.05


def test_a_constant_series_and_a_series_too_short_to_embed_score_zero():
    assert window_mmd(np.full(240, 3.0), 24.0) == 0.0
    assert window_mmd(np.arange(2 * EMBED_LENGTH + 1, dtype=float), 7.0) == 0.0
    assert window_mmd(np.zeros(0), 7.0) == 0.0


# ---------------------------------------------------------------------------
# B1
# ---------------------------------------------------------------------------


def test_the_common_cut_lands_where_every_series_steps_together():
    assert segment_common(common_step(1.0), DIAGONAL, penalty=60.0) == (100,)


def test_a_step_in_one_series_of_four_does_not_pay_for_the_whole_panel():
    price = 60.0
    assert segment_common(common_step(1.0), DIAGONAL, price) == (100,)
    assert segment_common(lone_step(1.0), DIAGONAL, price) == ()


def test_the_full_cost_finds_a_change_in_dependence_the_diagonal_cost_holds_fixed():
    panel = flipped_dependence()
    cuts = segment_common(panel, FULL, penalty=20.0)
    assert cuts and all(abs(cut - 100) <= 5 for cut in cuts)
    assert segment_common(panel, DIAGONAL, penalty=20.0) == ()


def test_a_panel_with_no_joint_structure_buys_no_cut_under_either_cost():
    assert segment_common(null_panel(), DIAGONAL, penalty=60.0) == ()
    assert segment_common(null_panel(), FULL, penalty=60.0) == ()


def test_a_negative_price_is_refused():
    with pytest.raises(ValueError, match="not negative"):
        segment_common(common_step(1.0), DIAGONAL, penalty=-1.0)


def test_an_unknown_covariance_cost_is_refused():
    with pytest.raises(ValueError, match="expected one of"):
        min_segment("median", 3)
    with pytest.raises(ValueError, match="expected one of"):
        common_segment_cost(_prefix(null_panel(), DIAGONAL), "median", np.array([0]), 8)


def test_the_shortest_segment_grows_with_the_series_count_only_under_the_full_cost():
    assert min_segment(DIAGONAL, 3) == 2
    assert min_segment(DIAGONAL, 9) == 2
    assert min_segment(FULL, 3) == 4


def test_a_segment_shorter_than_the_series_count_keeps_a_finite_cost():
    panel = np.random.default_rng(1).normal(0.0, 1.0, (10, 3))
    prefix = _prefix(panel, FULL)
    cost = float(common_segment_cost(prefix, FULL, np.array([0]), 2)[0])
    floor = float(prefix["floor"][0])
    assert np.isfinite(cost)
    # Two epochs leave the covariance of three series singular, and the ridge is
    # what holds its determinant above the floor.
    assert cost >= 2.0 * 3.0 * np.log(floor)


def test_a_declared_shortest_segment_moves_where_a_cut_may_fall():
    assert segment_common(common_step(1.0), DIAGONAL, 60.0, min_size=50) == (100,)
    assert segment_common(common_step(1.0), DIAGONAL, 60.0, min_size=150) == ()


def test_a_constant_panel_costs_the_same_however_it_is_cut():
    panel = np.full((60, 3), 7.0)
    assert partition_cost(panel, (20, 40)) == pytest.approx(partition_cost(panel, ()))
    assert segment_common(panel, DIAGONAL, penalty=5.0) == ()
    assert cost_drop_fraction(panel, DIAGONAL, penalty=5.0) == 0.0
    assert cost_drop_fraction(panel, FULL, penalty=5.0) == 0.0


def test_the_cut_set_a_price_buys_costs_less_than_holding_the_panel_whole():
    panel = common_step(2.0)
    cuts = segment_common(panel, DIAGONAL, penalty=60.0)
    assert partition_cost(panel, cuts) < partition_cost(panel, ())


def test_the_price_path_buys_fewer_cuts_as_the_price_rises():
    path = common_penalty_path(common_step(1.0), DIAGONAL)
    prices = [price for price, _ in path]
    counts = [len(cuts) for _, cuts in path]
    assert prices == sorted(float(p) for p in PRICES)
    assert counts == sorted(counts, reverse=True)
    assert counts[0] > counts[-1]


def test_the_drop_share_rises_with_the_common_step_and_is_zero_without_one():
    price = 60.0
    small = cost_drop_fraction(common_step(1.0), DIAGONAL, price)
    large = cost_drop_fraction(common_step(3.0), DIAGONAL, price)
    assert 0.0 < small < large <= 1.0
    assert cost_drop_fraction(null_panel(), DIAGONAL, price) == 0.0


def test_a_panel_of_no_epochs_holds_no_cost_and_no_drop():
    empty = np.zeros((0, 2))
    assert partition_cost(empty, ()) == 0.0
    assert segment_common(empty, DIAGONAL) == ()
    assert cost_drop_fraction(empty, DIAGONAL) == 0.0


def test_a_segment_outside_the_panel_is_refused():
    panel = null_panel(n=20)
    prefix = _prefix(panel, DIAGONAL)
    with pytest.raises(ValueError, match="ends inside the panel"):
        common_segment_cost(prefix, DIAGONAL, np.array([0]), 40)
    with pytest.raises(ValueError, match="opens inside"):
        common_segment_cost(prefix, DIAGONAL, np.array([9]), 9)
    # A cut off the end and a cut before the start both reach the same guard,
    # and an unguarded cost reads the second one off the far end of the panel.
    with pytest.raises(ValueError, match="ends inside the panel"):
        partition_cost(panel, (40,))
    with pytest.raises(ValueError, match="opens inside"):
        partition_cost(panel, (-3,))


def test_a_panel_that_is_not_finite_is_refused():
    panel = null_panel(n=20)
    panel[5, 1] = np.nan
    with pytest.raises(ValueError, match="finite epochs"):
        segment_common(panel, DIAGONAL)
    with pytest.raises(ValueError, match="finite epochs"):
        cost_drop_fraction(panel, DIAGONAL)


def test_a_panel_of_no_series_and_a_panel_of_three_axes_are_refused():
    with pytest.raises(ValueError, match="epochs by series"):
        segment_common(np.zeros((5, 0)), DIAGONAL)
    with pytest.raises(ValueError, match="epochs by series"):
        segment_common(np.zeros((5, 2, 2)), DIAGONAL)


def test_one_series_is_read_as_a_panel_of_one_column():
    rng = np.random.default_rng(14)
    series = np.concatenate([rng.normal(0.0, 1.0, 50), rng.normal(4.0, 1.0, 50)])
    assert segment_common(series, DIAGONAL, penalty=20.0) == (50,)
