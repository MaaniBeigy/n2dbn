"""
Tests for the trend shape and the unit-root pair the two level rows share.

What is pinned here is that the penalized spline follows a trend of any shape
and charges for it, that the shape test rejects a straight line where the trend
bends and holds one where it does not, that the summary scalar is zero on a
line and invariant to the level of the series, and that the two level tests
separate a stationary series, a unit root and a deterministic trend, and report
their disagreement as its own class.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.trend_shape import (
    DEGREE,
    KPSS_P_VALUES,
    SMOOTHING_GRID,
    STATIONARY,
    TREND_STATIONARY,
    UNDETERMINED,
    UNIT_ROOT,
    adf_test,
    curvature_energy,
    edf_test,
    epochs_needed,
    fit_spline,
    kpss_test,
    lag_rule,
    select_smoothing,
    spline_basis,
    spline_curvature,
    spline_knots,
    unit_root_pair,
)


def curved_series(seed: int = 0) -> np.ndarray:
    """A trend that rises once and falls once, under light noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(200.0)
    return 10.0 + 6.0 * np.sin(2.0 * np.pi * t / 200.0) + rng.normal(0.0, 0.5, 200)


def linear_series(seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(200.0)
    return 20.0 + 0.2 * t + rng.normal(0.0, 1.0, 200)


def white_noise(seed: int = 0, n: int = 300) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, 1.0, n)


def random_walk(seed: int = 0, n: int = 300) -> np.ndarray:
    return np.cumsum(np.random.default_rng(seed).normal(0.0, 1.0, n))


def trending_series(seed: int = 0, n: int = 300) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 0.1 * np.arange(float(n)) + rng.normal(0.0, 1.0, n)


def persistent_series(phi: float, n: int, seed: int) -> np.ndarray:
    """An autoregressive series at one persistence, built one epoch at a time."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal(0.0, 1.0)
    return x


# ---------------------------------------------------------------------------
# The penalized spline
# ---------------------------------------------------------------------------


def test_the_basis_stands_on_uniform_knots_and_sums_to_one_at_every_epoch():
    knots = spline_knots(200)
    _, basis = spline_basis(200)
    assert float(np.ptp(np.diff(knots))) < 1e-9
    assert basis.shape == (200, epochs_needed() // 2)
    assert np.allclose(basis.sum(axis=1), 1.0)


def test_a_heavy_price_leaves_the_least_squares_straight_line():
    t = np.arange(200.0)
    x = linear_series()
    fit = fit_spline(x, 1e12)
    line = np.polyval(np.polyfit(t, x, 1), t)
    assert fit.edf == pytest.approx(2.0, abs=1e-3)
    assert float(np.abs(fit.fitted - line).max()) < 1e-2


def test_a_larger_price_spends_fewer_degrees_of_freedom_and_fits_worse():
    x = curved_series()
    fits = [fit_spline(x, price) for price in (1e-3, 1e0, 1e3, 1e6)]
    edf = [fit.edf for fit in fits]
    rss = [fit.rss for fit in fits]
    assert edf == sorted(edf, reverse=True)
    assert rss == sorted(rss)
    assert edf[0] > 6.0
    assert edf[-1] == pytest.approx(2.0, abs=1e-3)


def test_the_selected_price_follows_the_truth_better_than_either_end_of_the_grid():
    rng = np.random.default_rng(7)
    t = np.arange(200.0)
    truth = 10.0 + 6.0 * np.sin(2.0 * np.pi * t / 200.0) + 0.02 * t
    noisy = truth + rng.normal(0.0, 1.5, 200)
    selected = select_smoothing(noisy)

    def error(fit):
        return float(np.sqrt(np.mean((fit.fitted - truth) ** 2)))

    assert error(selected) < error(fit_spline(noisy, SMOOTHING_GRID[0]))
    assert error(selected) < error(fit_spline(noisy, SMOOTHING_GRID[-1]))
    assert selected.gcv == min(fit_spline(noisy, p).gcv for p in SMOOTHING_GRID)


def test_a_negative_smoothing_price_is_refused():
    with pytest.raises(ValueError, match="not negative"):
        fit_spline(curved_series(), -1.0)


def test_a_series_shorter_than_the_basis_is_refused():
    assert epochs_needed(6) == 20
    with pytest.raises(ValueError, match="is fitted on 20 epochs; got 19"):
        fit_spline(np.arange(19.0), 1.0)


def test_a_basis_with_no_interior_knot_is_refused():
    with pytest.raises(ValueError, match="at least one interior knot"):
        epochs_needed(0)


def test_an_empty_smoothing_grid_is_refused():
    with pytest.raises(ValueError, match="at least one price"):
        select_smoothing(curved_series(), ())


def test_a_series_holding_a_value_that_is_not_finite_is_refused():
    x = curved_series()
    x[10] = np.nan
    with pytest.raises(ValueError, match="finite values"):
        spline_curvature(x, 24.0)


def test_the_curvature_energy_of_a_fitted_quadratic_is_its_squared_bend():
    t = np.arange(200.0)
    fit = select_smoothing(5.0 + 0.001 * t * t)
    # The second derivative of the quadratic is 0.002 per epoch squared, and one
    # unit of the axis is seven epochs.
    assert curvature_energy(fit, 7.0) == pytest.approx(0.002**2 * 7.0**4, rel=0.05)


def test_a_calendar_period_that_is_not_positive_is_refused():
    with pytest.raises(ValueError, match="period is positive"):
        curvature_energy(select_smoothing(curved_series()), 0.0)
    with pytest.raises(ValueError, match="period is positive"):
        spline_curvature(curved_series(), -7.0)


# ---------------------------------------------------------------------------
# The shape test
# ---------------------------------------------------------------------------


def test_the_shape_test_rejects_a_line_on_a_trend_that_rises_and_falls():
    result = edf_test(curved_series())
    assert result.edf > 4.0
    assert result.statistic > 100.0
    assert result.p_value < 1e-6
    assert result.rss_spline < 0.05 * result.rss_line


def test_the_shape_test_holds_a_line_on_a_noisy_straight_line():
    result = edf_test(linear_series())
    assert result.edf == pytest.approx(2.0, abs=1e-3)
    assert result.statistic == 0.0
    assert result.p_value == 1.0


def test_the_shape_test_holds_a_line_on_a_series_the_spline_reproduces():
    exact = edf_test(3.0 + 0.2 * np.arange(200.0))
    constant = edf_test(np.full(200, 4.0))
    assert exact.statistic == 0.0 and exact.p_value == 1.0
    assert constant.statistic == 0.0 and constant.p_value == 1.0
    assert exact.rss_spline < 1e-12 and constant.rss_spline < 1e-12


def test_a_bend_the_grid_can_only_pay_for_once_still_reports_its_price():
    result = edf_test(curved_series(), smoothings=(1.0,))
    assert result.smoothing == 1.0
    assert result.edf > 4.0
    assert result.p_value < 1e-6


# ---------------------------------------------------------------------------
# The summary scalar
# ---------------------------------------------------------------------------


def test_the_curvature_is_zero_on_a_line_a_constant_a_short_and_an_empty_series():
    assert spline_curvature(3.0 + 0.2 * np.arange(200.0), 24.0) == pytest.approx(
        0.0, abs=1e-12
    )
    assert spline_curvature(np.full(50, 4.0), 7.0) == 0.0
    assert spline_curvature(np.zeros(50), 7.0) == 0.0
    assert spline_curvature(np.arange(19.0), 7.0) == 0.0
    assert spline_curvature([], 7.0) == 0.0


def test_the_curvature_separates_a_bending_trend_from_a_straight_one():
    assert spline_curvature(curved_series(), 24.0) > 100.0 * spline_curvature(
        linear_series(), 24.0
    )
    assert spline_curvature(curved_series(), 24.0) > 0.01


def test_the_curvature_is_unchanged_when_the_whole_series_is_rescaled():
    x = curved_series()
    assert spline_curvature(3.0 * x, 24.0) == pytest.approx(
        spline_curvature(x, 24.0), rel=1e-9
    )


def test_the_curvature_is_stated_per_calendar_period():
    x = curved_series()
    assert spline_curvature(x, 24.0) == pytest.approx(
        spline_curvature(x, 7.0) * (24.0 / 7.0) ** 4, rel=1e-9
    )


# ---------------------------------------------------------------------------
# The unit-root pair
# ---------------------------------------------------------------------------


def test_the_lag_rule_grows_with_the_length_of_the_series():
    assert lag_rule(100) == 4
    assert lag_rule(1600) == 8
    assert lag_rule(300) >= lag_rule(100)


def test_the_dickey_fuller_test_rejects_a_unit_root_on_noise_and_holds_it_on_a_walk():
    noise = adf_test(white_noise())
    walk = adf_test(random_walk())
    assert noise.statistic < -3.43
    assert noise.p_value <= 0.01
    assert walk.statistic > -2.57
    assert walk.p_value > 0.10


def test_the_dickey_fuller_p_value_is_held_inside_its_table():
    explosive = adf_test(persistent_series(1.05, 200, 0))
    assert explosive.p_value == 0.99
    assert adf_test(white_noise()).p_value == 0.01


def test_the_declared_lag_order_is_the_one_the_regression_holds():
    assert adf_test(white_noise(), lags=0).lags == 0
    assert adf_test(white_noise(), lags=9).lags == 9
    assert adf_test(white_noise()).lags == lag_rule(300)


def test_a_negative_lag_order_is_refused():
    with pytest.raises(ValueError, match="lag order is not negative"):
        adf_test(white_noise(), lags=-1)


def test_a_series_shorter_than_the_dickey_fuller_regression_is_refused():
    with pytest.raises(ValueError, match="needs 8 epochs; got 7"):
        adf_test(white_noise(n=7), lags=2)


def test_a_constant_series_has_no_dickey_fuller_regression():
    with pytest.raises(ValueError, match="a constant at 4.0"):
        adf_test(np.full(60, 4.0))


def test_a_series_the_regression_fits_exactly_has_no_dickey_fuller_ratio():
    with pytest.raises(ValueError, match="needs a residual"):
        adf_test(np.arange(60.0))


def test_the_kpss_statistic_is_small_on_noise_and_large_on_a_walk():
    noise = kpss_test(white_noise())
    walk = kpss_test(random_walk())
    assert noise.statistic < 0.347
    assert noise.p_value == 0.10
    assert walk.statistic > 0.739
    assert walk.p_value == 0.01


def test_the_trend_form_holds_a_deterministic_path_the_level_form_rejects():
    x = trending_series()
    assert kpss_test(x, trend=False).p_value <= 0.01
    assert kpss_test(x, trend=True).p_value >= 0.10


def test_a_series_with_nothing_left_around_its_fit_reports_the_smallest_statistic():
    constant = kpss_test(np.full(60, 4.0))
    line = kpss_test(np.arange(60.0), trend=True)
    assert constant.statistic == 0.0 and constant.p_value == KPSS_P_VALUES[0]
    assert line.statistic == 0.0 and line.p_value == KPSS_P_VALUES[0]


def test_a_negative_bandwidth_is_refused():
    with pytest.raises(ValueError, match="bandwidth is not negative"):
        kpss_test(white_noise(), lags=-1)


def test_a_series_shorter_than_the_kpss_statistic_is_refused():
    with pytest.raises(ValueError, match="needs 3 epochs; got 2"):
        kpss_test(np.zeros(2))


def test_noise_reads_as_stationary_and_a_walk_carries_a_unit_root():
    noise = unit_root_pair(white_noise())
    walk = unit_root_pair(random_walk())
    assert noise.classification == STATIONARY
    assert walk.classification == UNIT_ROOT
    assert noise.adf.p_value <= 0.05 < noise.kpss_level.p_value
    assert walk.adf.p_value > 0.05 >= walk.kpss_level.p_value


def test_a_deterministic_trend_reads_as_stationary_around_that_trend():
    verdict = unit_root_pair(trending_series())
    assert verdict.classification == TREND_STATIONARY
    assert verdict.kpss_level.p_value <= 0.05 < verdict.kpss_trend.p_value


def test_a_short_persistent_series_that_rejects_neither_null_is_undetermined():
    verdict = unit_root_pair(persistent_series(0.7, 40, 0))
    assert verdict.classification == UNDETERMINED
    assert verdict.adf.p_value > 0.05
    assert verdict.kpss_level.p_value > 0.05


def test_two_tests_that_contradict_each_other_report_the_disagreement():
    verdict = unit_root_pair(persistent_series(0.95, 300, 1))
    assert verdict.adf.p_value <= 0.05
    assert verdict.kpss_level.p_value <= 0.05
    assert verdict.classification == UNDETERMINED


def test_the_reported_row_carries_both_statistics_and_the_class():
    row = unit_root_pair(white_noise()).as_row()
    assert set(row) == {
        "adf_statistic",
        "adf_p_value",
        "kpss_level",
        "kpss_level_p_value",
        "kpss_trend",
        "kpss_trend_p_value",
        "lags",
        "classification",
    }
    assert row["classification"] == STATIONARY
    assert row["lags"] == lag_rule(300)


def test_a_test_level_outside_the_unit_interval_is_refused():
    with pytest.raises(ValueError, match="inside"):
        unit_root_pair(white_noise(), alpha=0.0)


# ---------------------------------------------------------------------------
# What the basis has to hold at every length, in every unit
# ---------------------------------------------------------------------------


def test_the_basis_carries_the_first_and_the_last_epoch_at_every_length():
    """
    The knots are laid by padding a span, and a span divided and multiplied back
    lands a rounding error away from where it started.  A first knot of the basis
    a rounding error inside epoch zero puts that epoch outside the support, and
    the design matrix refuses the series.  This pins the two boundary knots on
    the ends of the series exactly, at every length the basis is fitted on.
    """
    for n in range(epochs_needed(), 301):
        knots, basis = spline_basis(n)
        assert knots[DEGREE] == 0.0
        assert knots[-DEGREE - 1] == float(n - 1)
        assert np.allclose(basis.sum(axis=1), 1.0)
        assert float(np.ptp(np.diff(knots))) < 1e-9


def test_the_curvature_reads_every_length_the_basis_is_fitted_on():
    lengths = range(epochs_needed(), 121)
    read = {
        n: spline_curvature(3.0 + np.sin(np.arange(float(n)) / 5.0), 7.0)
        for n in lengths
    }
    assert all(isinstance(value, float) for value in read.values())
    assert all(np.isfinite(value) for value in read.values())
    assert all(value > 0.0 for value in read.values())


def test_a_knot_vector_needs_two_epochs_to_span():
    for n in (-1, 0, 1):
        with pytest.raises(ValueError, match="at least two epochs"):
            spline_knots(n)
        with pytest.raises(ValueError, match="at least two epochs"):
            spline_basis(n)


def test_a_smoothing_price_that_is_not_a_number_is_refused():
    for price in (np.nan, np.inf):
        with pytest.raises(ValueError, match="not negative"):
            fit_spline(curved_series(), price)


def test_a_calendar_period_that_is_not_a_number_is_refused():
    for period in (np.nan, np.inf):
        with pytest.raises(ValueError, match="period is positive"):
            spline_curvature(curved_series(), period)
        with pytest.raises(ValueError, match="period is positive"):
            curvature_energy(select_smoothing(curved_series()), period)


def test_a_series_that_is_not_one_series_is_refused():
    grid = np.arange(200.0).reshape(10, 20)
    for read in (
        lambda x: spline_curvature(x, 7.0),
        edf_test,
        lambda x: fit_spline(x, 1.0),
        adf_test,
        kpss_test,
        unit_root_pair,
    ):
        with pytest.raises(ValueError, match="one series laid flat"):
            read(grid)
    with pytest.raises(ValueError, match="one series laid flat"):
        spline_curvature(5.0, 7.0)


# ---------------------------------------------------------------------------
# The units a series is carried in, which no statistic here reads
# ---------------------------------------------------------------------------
#
# Every scalar below is a ratio of the series to itself, so multiplying the whole
# series by a constant leaves it where it was.  A degeneracy guard written
# against an absolute size breaks that silently: the series stops being read and
# the declared value of a series with no shape is reported in its place.

UNITS = (1e6, 1e3, 1.0, 1e-3, 1e-6, 1e-9, 1e-14, 1e-40, 1e-150)


def test_the_curvature_holds_its_value_in_every_unit_the_series_can_be_carried_in():
    """
    The curve is fitted on the series divided by its level, so the ratio is taken
    before anything is squared.  Squaring the level of a series carried in small
    enough units underflows to zero, and the summary scalar then reads nan or
    zero on a trend that bends.  The scales below reach past that point.
    """
    x = curved_series()
    read = [spline_curvature(scale * x, 24.0) for scale in UNITS + (1e-200, 1e-300)]
    assert all(isinstance(value, float) and np.isfinite(value) for value in read)
    assert max(read) - min(read) < 1e-9 * read[0]


def test_the_shape_test_holds_its_ratio_in_every_unit():
    x = curved_series()
    read = [edf_test(scale * x).statistic for scale in UNITS]
    assert read[0] > 100.0
    assert max(read) - min(read) < 1e-9 * read[0]


def test_the_kpss_statistic_holds_its_value_in_every_unit():
    x = random_walk()
    read = [kpss_test(scale * x).statistic for scale in UNITS]
    assert read[0] > 0.739
    assert max(read) - min(read) < 1e-9 * read[0]


def test_the_dickey_fuller_ratio_holds_its_value_in_every_unit_and_at_every_level():
    x = white_noise()
    scaled = [adf_test(scale * x).statistic for scale in UNITS]
    shifted = [adf_test(x + offset).statistic for offset in (0.0, 1e3, 1e6, -1e6)]
    assert scaled[0] < -3.43
    assert max(scaled) - min(scaled) < 1e-9 * abs(scaled[0])
    assert max(shifted) - min(shifted) < 1e-9 * abs(shifted[0])


def test_a_negative_series_length_has_no_lag_rule():
    with pytest.raises(ValueError, match="length is not negative"):
        lag_rule(-1)


# ---------------------------------------------------------------------------
# Both level statistics against a published implementation
# ---------------------------------------------------------------------------
#
# The three values on each row were taken from statsmodels 0.14.6, at the lag
# order and bandwidth `lag_rule` declares: `adfuller(x, maxlag=5, regression='c',
# autolag=None)` and `kpss(x, regression='c' or 'ct', nlags=5)`.  They are
# written out here so the arithmetic is pinned to something outside this file.

PUBLISHED = {
    "white_noise": (-6.703009789996, 0.278999848079, 0.029640311244),
    "random_walk": (-0.950134610925, 1.449752203411, 0.967472331504),
    "trending_series": (-0.282951594031, 5.085000260237, 0.029640311244),
}


def test_both_level_statistics_match_a_published_implementation():
    for name, (adf, level, slope) in PUBLISHED.items():
        x = {
            "white_noise": white_noise,
            "random_walk": random_walk,
            "trending_series": trending_series,
        }[name]()
        assert lag_rule(x.size) == 5
        assert adf_test(x).statistic == pytest.approx(adf, abs=1e-9)
        assert kpss_test(x, trend=False).statistic == pytest.approx(level, abs=1e-9)
        assert kpss_test(x, trend=True).statistic == pytest.approx(slope, abs=1e-9)
