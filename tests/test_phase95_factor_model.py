"""
Tests for the approximate factor model the common factor row waits on.

What is pinned here is that principal components find a shared trajectory where
one was injected and find none where the columns were drawn apart, that the
Bai-Ng criteria recover the number of trajectories injected, that the two
comparison figures separate two panels of one factor world from two panels of
different ones, and that the summary scalar is high only when the blocks of a
series move together.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.evaluation.factor_model import (
    CRITERIA,
    IC_P1,
    IC_P2,
    IC_P3,
    bai_ng_criteria,
    compare_factors,
    factor_correlation,
    factor_share,
    principal_angles,
    principal_components,
    standardize,
)


def factor_panel(
    n_factors: int = 1,
    epochs: int = 200,
    width: int = 20,
    noise: float = 0.4,
    world: int = 0,
    draw: int = 0,
) -> tuple:
    """A panel whose columns share `n_factors` trajectories under column noise.

    Every column loads on every factor with a magnitude between 0.7 and 1.3 and
    a sign of its own, so each factor stands well clear of the column noise and
    the loading space is well spread.
    """
    shared = np.random.default_rng(world)
    factors = shared.normal(0.0, 1.0, (epochs, n_factors))
    magnitude = shared.uniform(0.7, 1.3, (width, n_factors))
    loadings = shared.choice((-1.0, 1.0), (width, n_factors)) * magnitude
    drawn = np.random.default_rng(1000 + draw)
    panel = factors @ loadings.T + drawn.normal(0.0, noise, (epochs, width))
    return panel, factors


def noise_panel(epochs: int = 200, width: int = 20, seed: int = 3) -> np.ndarray:
    """A panel whose columns were drawn apart, so nothing is shared."""
    return np.random.default_rng(seed).normal(0.0, 1.0, (epochs, width))


def daily_series(days: int = 40, width: int = 24, seed: int = 5) -> np.ndarray:
    """One shape every block carries, each block at its own strength."""
    rng = np.random.default_rng(seed)
    shape = np.sin(np.arange(width) * 2.0 * np.pi / width) + 2.0
    strength = rng.uniform(0.5, 1.5, days)
    blocks = strength[:, None] * shape[None, :]
    return (blocks + rng.normal(0.0, 0.05, (days, width))).ravel()


def test_the_first_factor_of_a_shared_world_is_the_trajectory_that_was_injected():
    panel, factors = factor_panel(1, world=1)
    fit = principal_components(panel, 1)
    assert fit.n_factors == 1
    assert factor_correlation(fit.factors[:, 0], factors[:, 0]) > 0.98
    assert fit.explained > 0.6


def test_columns_drawn_apart_carry_no_common_factor():
    panel = noise_panel()
    assert bai_ng_criteria(panel).selected == {IC_P1: 0, IC_P2: 0, IC_P3: 0}
    assert principal_components(panel, 1).explained < 0.2


def test_the_criteria_recover_how_many_trajectories_were_injected():
    panel, _ = factor_panel(3, world=2)
    result = bai_ng_criteria(panel, max_factors=8)
    assert result.selected == {IC_P1: 3, IC_P2: 3, IC_P3: 3}
    # The count a criterion reports has to be the count its own curve bottoms
    # out at, which is what ties the selection to the sweep it is read off.
    for name, curve in zip(CRITERIA, (result.ic_p1, result.ic_p2, result.ic_p3)):
        assert curve[result.selected[name]] == curve.min()


def test_the_common_component_and_the_residual_add_back_to_the_panel():
    panel, _ = factor_panel(2, epochs=80, width=10, world=3)
    fit = principal_components(panel, 2)
    assert fit.common + fit.idiosyncratic == pytest.approx(standardize(panel))
    # The residual of a factor model is orthogonal to the factors it keeps.
    assert fit.factors.T @ fit.idiosyncratic == pytest.approx(
        np.zeros((2, 10)), abs=1e-8
    )


def test_the_factors_are_orthogonal_and_carry_the_identity_as_their_moment():
    panel, _ = factor_panel(3, epochs=120, width=12, world=4)
    fit = principal_components(panel, 3)
    moment = fit.factors.T @ fit.factors / fit.factors.shape[0]
    assert moment == pytest.approx(np.eye(3), abs=1e-8)


def test_a_constant_column_is_left_at_zero_and_takes_no_loading():
    panel, _ = factor_panel(1, epochs=60, width=5, world=5)
    padded = np.column_stack([panel, np.full(60, 4.0)])
    assert np.all(standardize(padded)[:, -1] == 0.0)
    fit = principal_components(padded, 2)
    assert fit.loadings[-1] == pytest.approx(np.zeros(2))
    assert fit.variance_shares[-1] == pytest.approx(0.0)
    # The column carries nothing, so the shares of the rest are the shares they
    # hold on the panel that never had it.
    base = principal_components(panel, 2)
    assert fit.variance_shares[:2] == pytest.approx(base.variance_shares[:2])


def test_a_panel_of_one_column_is_wholly_explained_by_its_own_factor():
    column = np.random.default_rng(6).normal(0.0, 1.0, (50, 1))
    fit = principal_components(column, 1)
    assert fit.explained == pytest.approx(1.0)
    assert factor_correlation(fit.factors[:, 0], column[:, 0]) == pytest.approx(1.0)


def test_a_factor_count_the_panel_cannot_carry_is_refused():
    panel, _ = factor_panel(1, epochs=40, width=4, world=7)
    with pytest.raises(ValueError, match="carries 1 to 4 factors"):
        principal_components(panel, 5)
    with pytest.raises(ValueError, match="got 0"):
        principal_components(panel, 0)


def test_a_panel_wider_than_it_is_long_still_carries_its_shared_trajectory():
    panel, factors = factor_panel(1, epochs=6, width=30, world=8)
    fit = principal_components(panel, 1)
    assert factor_correlation(fit.factors[:, 0], factors[:, 0]) > 0.9
    assert fit.singular_values.size == 6
    with pytest.raises(ValueError, match="carries 1 to 6 factors"):
        principal_components(panel, 7)


def test_the_sweep_stops_at_the_rank_the_panel_carries():
    panel, _ = factor_panel(1, epochs=40, width=4, world=9)
    result = bai_ng_criteria(panel, max_factors=8)
    assert result.counts.tolist() == [0, 1, 2, 3, 4]
    # The cap is the rank, so asking beyond it answers the sweep the panel
    # admits and does not shift the criteria of the counts it shares.
    inside = bai_ng_criteria(panel, max_factors=4)
    assert result.ic_p2 == pytest.approx(inside.ic_p2)
    assert result.selected == inside.selected


def test_a_sweep_reaching_no_factor_at_all_is_refused():
    with pytest.raises(ValueError, match="at least one factor; got 0"):
        bai_ng_criteria(noise_panel(20, 4), max_factors=0)


def test_the_zero_factor_residual_is_the_whole_variance_and_the_sweep_eats_it():
    panel, _ = factor_panel(2, epochs=100, width=8, world=10)
    result = bai_ng_criteria(panel, max_factors=5)
    assert result.residual_variance[0] == pytest.approx(1.0)
    assert np.all(np.diff(result.residual_variance) < 0.0)


def test_the_three_criteria_are_the_declared_bai_ng_penalties():
    epochs, width = 60, 10
    panel, _ = factor_panel(2, epochs=epochs, width=width, world=11)
    result = bai_ng_criteria(panel, max_factors=4)
    k = result.counts
    log_v = np.log(result.residual_variance)
    ratio = (epochs + width) / (epochs * width)
    smaller = min(epochs, width)
    assert result.ic_p1 == pytest.approx(
        log_v + k * ratio * math.log(epochs * width / (epochs + width))
    )
    assert result.ic_p2 == pytest.approx(log_v + k * ratio * math.log(smaller))
    assert result.ic_p3 == pytest.approx(log_v + k * math.log(smaller) / smaller)


def test_the_fit_term_is_the_residual_of_a_refit_at_that_many_factors():
    """V(k) is the mean squared idiosyncratic residual of the k-factor fit.

    The penalties are read off a stated formula, so the fit term is the half of
    the criterion a mistake could hide in.  It is checked here against a refit
    that never touches the singular values the sweep is built from.
    """
    panel, _ = factor_panel(2, epochs=60, width=10, world=21)
    result = bai_ng_criteria(panel, max_factors=4)
    z = standardize(panel)
    assert result.residual_variance[0] == pytest.approx(float((z**2).sum()) / z.size)
    for k in result.counts[1:]:
        refit = principal_components(panel, int(k))
        assert result.residual_variance[k] == pytest.approx(
            float((refit.idiosyncratic**2).sum()) / z.size
        )


def test_the_variance_share_is_the_share_of_the_panel_the_fit_takes_off():
    panel, _ = factor_panel(3, epochs=90, width=12, world=22)
    fit = principal_components(panel, 3)
    z = standardize(panel)
    left_over = float((fit.idiosyncratic**2).sum()) / float((z**2).sum())
    assert fit.explained == pytest.approx(1.0 - left_over)
    # The loadings are the projection of the panel on the factors, which is the
    # normalization the fit is stated on.
    assert fit.loadings == pytest.approx(z.T @ fit.factors / z.shape[0])


def test_a_factor_count_that_is_not_a_whole_number_is_refused():
    panel = noise_panel(30, 5)
    with pytest.raises(ValueError, match="a whole number; got 1.5"):
        principal_components(panel, 1.5)
    with pytest.raises(ValueError, match="a whole number; got inf"):
        bai_ng_criteria(panel, max_factors=math.inf)
    with pytest.raises(ValueError, match="a whole number; got nan"):
        bai_ng_criteria(panel, max_factors=math.nan)


def test_a_panel_too_small_for_the_penalties_to_price_a_factor_is_refused():
    # ln(NT / (N + T)) is the IC_p1 price of a factor, and it is zero at two by
    # two and negative below, so the sweep would run to its own top.
    with pytest.raises(ValueError, match=r"\(T - 1\)\(N - 1\) above one"):
        bai_ng_criteria(np.array([[1.0, 2.0, 3.0]]))
    with pytest.raises(ValueError, match=r"got shape \(5, 1\)"):
        bai_ng_criteria(np.arange(5.0).reshape(5, 1))
    with pytest.raises(ValueError, match=r"got shape \(2, 2\)"):
        bai_ng_criteria(np.array([[1.0, 2.0], [3.0, 5.0]]))
    # One column more and the price is positive, so the sweep is readable.
    wider = bai_ng_criteria(np.array([[1.0, 2.0, 4.0], [3.0, 5.0, 6.0]]))
    assert wider.counts.tolist() == [0, 1, 2]


def test_a_panel_of_one_value_everywhere_explains_nothing_and_selects_nothing():
    flat = np.full((30, 4), 2.0)
    fit = principal_components(flat, 2)
    assert fit.explained == 0.0
    assert np.all(fit.variance_shares == 0.0)
    assert np.all(fit.common == 0.0)
    assert bai_ng_criteria(flat, max_factors=4).selected == {
        IC_P1: 0,
        IC_P2: 0,
        IC_P3: 0,
    }


def test_a_panel_that_is_not_a_matrix_of_finite_numbers_is_refused():
    with pytest.raises(ValueError, match="two-dimensional; got 1"):
        standardize(np.zeros(10))
    with pytest.raises(ValueError, match="an epoch and a column"):
        standardize(np.zeros((0, 4)))
    with pytest.raises(ValueError, match="an epoch and a column"):
        standardize(np.zeros((4, 0)))
    with pytest.raises(ValueError, match="is finite; got nan"):
        standardize(np.array([[1.0, np.nan], [2.0, 3.0]]))


def test_a_space_meets_itself_at_no_angle_and_a_foreign_one_at_a_right_angle():
    basis = np.eye(6)[:, :2]
    assert principal_angles(basis, basis) == pytest.approx(np.zeros(2), abs=1e-8)
    assert principal_angles(basis, np.eye(6)[:, 2:4]) == pytest.approx(
        np.full(2, math.pi / 2)
    )
    # A shared direction and a foreign one give one of each angle.
    mixed = principal_angles(basis, np.eye(6)[:, 1:3])
    assert mixed[0] == pytest.approx(0.0, abs=1e-8)
    assert mixed[1] == pytest.approx(math.pi / 2)


def test_the_angles_do_not_move_when_a_space_is_handed_another_basis():
    rng = np.random.default_rng(12)
    a = rng.normal(0.0, 1.0, (15, 3))
    b = rng.normal(0.0, 1.0, (15, 3))
    rotation, _ = np.linalg.qr(rng.normal(0.0, 1.0, (3, 3)))
    assert principal_angles(a @ rotation, b) == pytest.approx(principal_angles(a, b))


def test_two_panels_of_one_world_span_one_space_and_two_worlds_do_not():
    left, _ = factor_panel(2, world=13, draw=0)
    right, _ = factor_panel(2, world=13, draw=1)
    stranger, _ = factor_panel(2, world=14, draw=2)
    together = principal_angles(
        principal_components(left, 2).loadings,
        principal_components(right, 2).loadings,
    )
    apart = principal_angles(
        principal_components(left, 2).loadings,
        principal_components(stranger, 2).loadings,
    )
    assert float(together.max()) < 0.2
    assert float(apart.min()) > 0.5


def test_two_loading_spaces_that_do_not_line_up_are_refused():
    with pytest.raises(ValueError, match="one series count; got 6 and 4"):
        principal_angles(np.eye(6)[:, :2], np.eye(4)[:, :2])
    with pytest.raises(ValueError, match="two-dimensional; got 1 and 2"):
        principal_angles(np.zeros(6), np.eye(6)[:, :2])


def test_a_loading_space_of_rank_zero_carries_no_angle():
    with pytest.raises(ValueError, match="rank zero"):
        principal_angles(np.zeros((6, 2)), np.eye(6)[:, :2])
    with pytest.raises(ValueError, match="ranks 0 and 2"):
        principal_angles(np.zeros((6, 0)), np.eye(6)[:, :2])


def test_a_loading_space_holding_a_non_finite_entry_is_refused():
    """A non-finite loading is named, and named before anything reads it.

    An undefined entry sends the decomposition into a failure of its own, and an
    infinite one drives every finite direction to a zero singular value, which
    reads as a space of rank zero and misnames what went wrong.
    """
    undefined = np.array([[np.nan], [1.0], [2.0], [3.0], [4.0], [5.0]])
    endless = np.array([[np.inf], [1.0], [2.0], [3.0], [4.0], [5.0]])
    with pytest.raises(ValueError, match="a loading space is finite; got nan"):
        principal_angles(undefined, np.eye(6)[:, :1])
    with pytest.raises(ValueError, match="a loading space is finite; got inf"):
        principal_angles(np.eye(6)[:, :1], endless)


def test_a_flipped_trajectory_matches_the_one_it_was_flipped_from():
    trajectory = np.random.default_rng(15).normal(0.0, 1.0, 100)
    assert factor_correlation(trajectory, -trajectory) == pytest.approx(1.0)
    assert factor_correlation(trajectory, trajectory) == pytest.approx(1.0)
    stranger = np.random.default_rng(16).normal(0.0, 1.0, 100)
    assert factor_correlation(trajectory, stranger) < 0.3


def test_a_trajectory_holding_one_value_correlates_with_nothing():
    assert factor_correlation(np.full(20, 3.0), np.arange(20.0)) == 0.0


def test_two_trajectories_of_different_lengths_or_of_one_epoch_are_refused():
    with pytest.raises(ValueError, match="one length; got 10 and 11"):
        factor_correlation(np.zeros(10), np.zeros(11))
    with pytest.raises(ValueError, match="needs two epochs; got 1"):
        factor_correlation(np.zeros(1), np.zeros(1))
    with pytest.raises(ValueError, match="a correlation needs two epochs; got 0"):
        factor_correlation(np.zeros(0), np.zeros(0))


def test_a_trajectory_holding_a_non_finite_entry_is_refused():
    """A non-finite entry is named, where the arithmetic would carry it through.

    Centring an undefined entry leaves every entry undefined, and the ratio of
    two undefined figures is one more, so the figure returned would be nan.
    """
    with pytest.raises(ValueError, match="a trajectory is finite; got nan"):
        factor_correlation(np.array([np.nan, 1.0, 2.0, 3.0]), np.arange(4.0))
    with pytest.raises(ValueError, match="a trajectory is finite; got inf"):
        factor_correlation(np.arange(4.0), np.array([np.inf, 1.0, 2.0, 3.0]))


def test_a_correlation_survives_a_trajectory_large_enough_to_overflow_its_norm():
    """The squared norm of a 1e200 trajectory overflows; the correlation must not.

    Taking the scale out before the norm keeps the figure the one the same
    trajectory scores at a readable size.
    """
    small = np.array([1.0, 2.0, 3.0, 5.0, 8.0])
    huge = small * 1e200
    partner = np.array([2.0, 1.0, 4.0, 4.0, 9.0])
    scored = factor_correlation(huge, partner)
    assert scored == pytest.approx(factor_correlation(small, partner))
    assert scored == pytest.approx(abs(np.corrcoef(small, partner)[0, 1]))
    assert 0.0 < scored < 1.0


def test_a_match_against_a_second_draw_of_one_world_agrees_on_every_figure():
    left, _ = factor_panel(1, world=17, draw=0)
    right, _ = factor_panel(1, world=17, draw=1)
    match = compare_factors(left, right)
    assert match.count_difference == 0
    assert match.mean_angle < 0.15
    assert match.first_correlation > 0.9
    assert match.share_difference < 0.05
    # The row carries the four figures under their own names, and the four are
    # far enough apart here that a crossed wiring would show.
    assert match.as_row() == {
        "count_difference": match.count_difference,
        "mean_angle": match.mean_angle,
        "first_correlation": match.first_correlation,
        "share_difference": match.share_difference,
    }
    assert len({round(v, 6) for v in match.as_row().values()}) == 4


def test_a_match_against_a_panel_of_no_shared_trajectory_disagrees():
    left, _ = factor_panel(1, world=18)
    match = compare_factors(left, noise_panel(200, 20, seed=19))
    assert match.count_difference == 1
    assert match.mean_angle > 1.0
    assert match.first_correlation < 0.3
    assert match.share_difference > 0.3


def test_a_match_on_two_panels_that_do_not_line_up_is_refused():
    with pytest.raises(ValueError, match="expected one of"):
        compare_factors(noise_panel(50, 5), noise_panel(50, 5), criterion="IC_p4")
    with pytest.raises(ValueError, match="one panel shape"):
        compare_factors(noise_panel(50, 5), noise_panel(50, 6))


def test_the_summary_scalar_is_high_when_the_blocks_move_together():
    together = factor_share(daily_series(), 24.0)
    apart = factor_share(np.random.default_rng(20).normal(0.0, 1.0, 960), 24.0)
    assert together > 0.9
    assert apart < 0.3
    # Blocks of seven cut the same series at a phase that walks, so the blocks
    # no longer share one shape and the leading component holds less of them.
    assert factor_share(daily_series(), 7.0) < together


def test_the_summary_scalar_is_zero_where_no_block_departs_from_the_average():
    shape = np.sin(np.arange(24) * 2.0 * np.pi / 24.0)
    assert factor_share(np.tile(shape, 5), 24.0) == 0.0
    assert factor_share(np.full(240, 3.0), 24.0) == 0.0


def test_the_summary_scalar_is_zero_on_a_series_holding_fewer_than_two_blocks():
    assert factor_share(np.zeros(0), 24.0) == 0.0
    assert factor_share(np.arange(30.0), 24.0) == 0.0


def test_the_summary_scalar_is_exactly_one_where_every_block_is_one_shape_scaled():
    """Blocks that are one shape at their own strengths sit at rank one.

    Standardizing by position divides that strength out, so every column becomes
    the same standardized strength up to the sign of the shape at that position,
    and the leading component holds all of it.
    """
    shape = np.sin(np.arange(24) * 2.0 * np.pi / 24.0) + 2.0
    strength = np.linspace(0.5, 1.5, 30)
    assert factor_share((strength[:, None] * shape[None, :]).ravel(), 24.0) == 1.0
    # Two blocks put every column at one of two values, so the panel is at rank
    # one whatever the blocks hold, and that is all two cycles can say.
    assert factor_share(np.arange(48.0), 24.0) == 1.0


def test_the_summary_scalar_reads_the_period_it_is_handed():
    series = daily_series()
    # A period is rounded to whole epochs, so a period just off an hour is the
    # hour, and the whole blocks of it are what the panel is laid out from.
    assert factor_share(series, 24.4) == factor_share(series, 24.0)
    # The tail past the last whole block is dropped, so a series of five blocks
    # and a remainder reads as the five.
    assert factor_share(series[: 24 * 5 + 7], 24.0) == pytest.approx(
        factor_share(series[: 24 * 5], 24.0)
    )


def test_a_calendar_period_that_is_undefined_or_under_two_epochs_is_refused():
    with pytest.raises(ValueError, match="two epochs; got period 1.0"):
        factor_share(np.arange(100.0), 1.0)
    with pytest.raises(ValueError, match="a calendar period is finite; got nan"):
        factor_share(np.arange(100.0), math.nan)
    with pytest.raises(ValueError, match="a calendar period is finite; got inf"):
        factor_share(np.arange(100.0), math.inf)


def test_a_series_holding_a_non_finite_entry_is_refused_at_every_length():
    """The same bad data is refused whether or not it fills two blocks.

    A series too short to lay out returns zero, and a non-finite entry has to be
    caught before that, so a broken series never scores zero on its length.
    """
    with pytest.raises(ValueError, match="a series is finite; got nan"):
        factor_share(np.full(240, np.nan), 24.0)
    with pytest.raises(ValueError, match="a series is finite; got nan"):
        factor_share(np.full(30, np.nan), 24.0)
    with pytest.raises(ValueError, match="a series is finite; got inf"):
        factor_share(np.full(30, np.inf), 24.0)
