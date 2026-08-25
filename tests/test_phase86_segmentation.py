"""
Tests for the penalized segmentation the three changepoint rows share.

Three rows of the grid wait on one dependency, so what is pinned here is that
each cost buys a cut for the change it names and for no other, that the price
sweep is monotone in what it buys, and that the six figures a changepoint row
reports separate two segmentations that agree from two that do not.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.segmentation import (
    COST_PARAMETERS,
    COSTS,
    POISSON,
    SLOPE,
    VARIANCE,
    ChangepointMatch,
    _log_factorial,
    _prefix,
    _residual_variance,
    annotation_error,
    covering,
    cut_rate,
    f1_margin,
    hausdorff,
    jump_magnitudes,
    ks_statistic,
    match_changepoints,
    penalty_path,
    segment,
    segment_cost,
    segment_lengths,
)


def level_series(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.concatenate(
        [rng.poisson(2, 60), rng.poisson(9, 60), rng.poisson(3, 60)]
    ).astype(float)


def spread_series(seed: int = 1) -> np.ndarray:
    """Counts whose mean holds at five and whose variance rises from 4 to 43."""
    rng = np.random.default_rng(seed)
    quiet = rng.poisson(5.0, 80)
    loud = rng.negative_binomial(1.0, 1.0 / 6.0, 80)
    return np.concatenate([quiet, loud]).astype(float)


def slope_series(seed: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(120.0)
    return np.where(t < 60, 0.5 * t, 30.0 - 0.5 * (t - 60)) + rng.normal(0, 1, 120)


def test_the_poisson_cost_puts_its_cuts_where_the_rate_changes():
    cuts = segment(level_series(), POISSON, penalty=10.0)
    assert cuts == (60, 120)


def test_the_variance_cost_cuts_where_the_spread_changes_and_the_level_does_not():
    cuts = segment(spread_series(), VARIANCE, penalty=50.0)
    assert any(abs(c - 80) <= 5 for c in cuts)
    # The rate holds at five across the change, so a level cost has nothing to buy.
    assert segment(spread_series(), POISSON, penalty=50.0) == ()


def test_the_slope_cost_cuts_where_the_trend_turns():
    cuts = segment(slope_series(), SLOPE, penalty=20.0)
    assert cuts and all(abs(c - 60) <= 3 for c in cuts)


def test_the_linear_cost_keeps_its_cuts_when_the_whole_series_is_tilted():
    """A line fitted inside every segment absorbs a global trend exactly."""
    x = slope_series()
    t = np.arange(x.size, dtype=float)
    cuts = segment(x, SLOPE, penalty=20.0)
    assert cuts
    for tilt in (0.5, 3.0, -2.0):
        assert segment(x + tilt * t, SLOPE, penalty=20.0) == cuts


def test_the_variance_floor_of_the_linear_cost_ignores_a_global_trend():
    x = slope_series()
    t = np.arange(x.size, dtype=float)
    assert _residual_variance(x, SLOPE) == pytest.approx(
        _residual_variance(x + 4.0 * t, SLOPE)
    )
    # The other costs have no line to absorb it, so their floor moves with it.
    assert _residual_variance(x, VARIANCE) < _residual_variance(x + 4.0 * t, VARIANCE)


def test_a_series_too_short_to_fit_a_line_has_no_residual_variance():
    assert _residual_variance(np.zeros(0), SLOPE) == 0.0
    assert _residual_variance(np.array([3.0]), VARIANCE) == 0.0


def test_a_flat_series_buys_no_cut_under_any_cost():
    flat = np.full(120, 4.0)
    for cost in COSTS:
        assert segment(flat, cost, penalty=5.0) == ()


def test_a_price_no_segment_can_pay_leaves_the_series_whole():
    assert segment(level_series(), POISSON, penalty=1e6) == ()


def test_a_series_shorter_than_two_segments_is_never_cut():
    assert segment([1.0, 2.0, 3.0], POISSON, penalty=0.1) == ()


def test_a_negative_price_is_refused():
    with pytest.raises(ValueError, match="not negative"):
        segment(level_series(), POISSON, penalty=-1.0)


def test_an_unknown_cost_is_refused():
    with pytest.raises(ValueError, match="expected one of"):
        segment_cost(_prefix(level_series()), "median", np.array([0]), 10)


def test_the_declared_shortest_segment_is_one_epoch_past_the_free_parameters():
    assert COST_PARAMETERS == {POISSON: 1, VARIANCE: 2, SLOPE: 3}
    # A declared floor overrides it, and a long one forbids every cut.
    assert segment(level_series(), POISSON, penalty=10.0, min_size=100) == ()


def test_the_price_path_buys_fewer_cuts_as_the_price_rises():
    path = penalty_path(level_series(), POISSON, (2.0, 10.0, 40.0, 400.0))
    prices = [price for price, _ in path]
    counts = [len(cuts) for _, cuts in path]
    assert prices == sorted(prices)
    assert counts == sorted(counts, reverse=True)
    assert counts[-1] == 0


def test_the_log_factorial_ignores_a_negative_entry():
    assert _log_factorial(np.array([-1.0, 0.0, 3.0])) == pytest.approx(
        [0.0, 0.0, float(np.log(6.0))]
    )


def test_a_poisson_segment_of_no_events_costs_nothing_to_hold():
    prefix = _prefix(np.zeros(20))
    assert segment_cost(prefix, POISSON, np.array([0]), 20)[0] == pytest.approx(
        0.0, abs=1e-6
    )


def test_two_identical_cut_sets_agree_on_every_figure():
    a = (20, 60)
    assert annotation_error(a, a) == 0
    assert f1_margin(a, a) == (1.0, 1.0, 1.0)
    assert covering(a, a, 100) == pytest.approx(1.0)
    assert hausdorff(a, a, 100) == 0.0


def test_two_empty_cut_sets_match_and_an_empty_one_against_a_filled_one_does_not():
    assert f1_margin((), ()) == (1.0, 1.0, 1.0)
    assert hausdorff((), (), 50) == 0.0
    assert f1_margin((), (10,)) == (0.0, 0.0, 0.0)
    assert hausdorff((), (10,), 50) == 50.0


def test_a_cut_inside_the_margin_matches_and_one_outside_it_does_not():
    assert f1_margin((20,), (22,), margin=3)[2] == 1.0
    assert f1_margin((20,), (30,), margin=3)[2] == 0.0


def test_one_cut_of_the_other_set_is_claimed_once():
    precision, recall, _ = f1_margin((20, 21), (20,), margin=3)
    assert precision == 0.5
    assert recall == 1.0


def test_two_cut_sets_that_match_nothing_score_zero_rather_than_dividing_by_zero():
    assert f1_margin((0,), (99,), margin=1) == (0.0, 0.0, 0.0)


def test_the_covering_of_a_partition_by_a_coarser_one_falls_with_the_disagreement():
    close = covering((50,), (52,), 100)
    far = covering((50,), (90,), 100)
    assert 1.0 > close > far > 0.0


def test_a_partition_of_no_epochs_is_refused():
    with pytest.raises(ValueError, match="at least one epoch"):
        covering((), (), 0)


def test_the_jump_magnitude_is_the_change_in_segment_mean():
    values = np.concatenate([np.zeros(10), np.full(10, 4.0)])
    assert jump_magnitudes(values, (10,)).tolist() == [4.0]
    assert jump_magnitudes(values, ()).size == 0


def test_the_segment_lengths_cover_the_series():
    lengths = segment_lengths((10, 30), 50)
    assert lengths.tolist() == [10.0, 20.0, 20.0]
    assert lengths.sum() == 50.0


def test_the_ks_statistic_is_zero_on_one_sample_and_one_on_an_empty_pair():
    assert ks_statistic([1.0, 2.0], [1.0, 2.0]) == 0.0
    assert ks_statistic([], []) == 0.0
    assert ks_statistic([], [1.0]) == 1.0
    assert ks_statistic([0.0, 0.0], [5.0, 5.0]) == 1.0


def test_a_match_against_a_second_draw_of_the_same_world_agrees():
    match = match_changepoints(level_series(0), level_series(3), POISSON, 10.0)
    assert isinstance(match, ChangepointMatch)
    assert match.annotation_error == 0
    assert match.f1_margin == 1.0
    assert match.covering > 0.95
    assert match.hausdorff <= 3.0


def test_a_match_against_a_flat_series_disagrees_on_every_figure():
    match = match_changepoints(level_series(), np.full(180, 4.0), POISSON, 10.0)
    assert match.annotation_error == 2
    assert match.f1_margin == 0.0
    assert match.covering < 0.5
    assert match.hausdorff == 180.0
    assert match.jump_ks == 1.0
    assert match.segment_ks == 1.0
    assert set(match.as_row()) == {
        "annotation_error",
        "f1_margin",
        "covering",
        "hausdorff",
        "jump_ks",
        "segment_ks",
    }


def test_two_series_of_different_lengths_are_refused():
    with pytest.raises(ValueError, match="one span"):
        match_changepoints(np.zeros(10), np.zeros(11))


def test_the_cut_rate_is_cuts_per_epoch_and_an_empty_series_has_none():
    assert cut_rate(level_series(), POISSON, 10.0) == pytest.approx(2 / 180)
    assert cut_rate(np.zeros(0), POISSON, 10.0) == 0.0
