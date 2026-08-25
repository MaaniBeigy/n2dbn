"""
Tests for the participant axis as a drawn level.

Two limits pin the mechanism.  A spread taken to infinity returns the free fit
the cohort had before, and a spread taken to zero pools every participant into
the cohort, so what the estimated spread does is choose between them on the
record.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.learning.partial_pooling import (
    LevelPrior,
    fit_levels,
    penalized_solve,
    shrinkage,
    variance_update,
)

WIDTH = 4  # the pooled intercept and three deviation contrasts


def normal_equations(seed: int = 0, n_other: int = 2, level_sd: float = 1.0):
    """
    A curvature and a score from a record whose levels are drawn at `level_sd`.

    The level block is the first `WIDTH` columns, the intercept and three
    deviation contrasts, and the rest is whatever else the design carries.
    """
    rng = np.random.default_rng(seed)
    size = WIDTH + n_other
    design = rng.normal(size=(200, size))
    truth = np.zeros(size)
    truth[0] = 1.0
    truth[1:WIDTH] = rng.normal(scale=level_sd, size=WIDTH - 1)
    truth[WIDTH:] = rng.normal(size=n_other)
    response = design @ truth + rng.normal(scale=0.5, size=200)
    return design.T @ design, design.T @ response


def test_a_prior_names_its_free_levels_and_leaves_the_intercept_alone():
    prior = LevelPrior(variance=2.0, width=WIDTH)
    assert prior.n_levels == WIDTH - 1
    precision = prior.precision(6)
    assert precision.tolist() == [0.0, 0.5, 0.5, 0.5, 0.0, 0.0]


def test_a_prior_refuses_a_spread_or_a_width_it_cannot_use():
    with pytest.raises(ValueError, match="variance is positive"):
        LevelPrior(variance=0.0, width=WIDTH)
    with pytest.raises(ValueError, match="at least one column"):
        LevelPrior(variance=1.0, width=0)


def test_a_wide_prior_returns_the_free_fit():
    A, b = normal_equations()
    free = np.linalg.solve(A, b)
    fitted, _ = penalized_solve(A, b, LevelPrior(1e12, WIDTH))
    assert np.allclose(fitted, free, atol=1e-6)


def test_a_tight_prior_pools_every_participant_into_the_cohort():
    A, b = normal_equations()
    fitted, _ = penalized_solve(A, b, LevelPrior(1e-10, WIDTH))
    assert np.allclose(fitted[1:WIDTH], 0.0, atol=1e-6)
    # The pooled intercept is never shrunk, since the coding already centers it.
    assert abs(fitted[0]) > 0.0


def test_a_level_block_wider_than_the_design_is_refused():
    A, b = normal_equations(n_other=0)
    with pytest.raises(ValueError, match="does not fit a design"):
        penalized_solve(A, b, LevelPrior(1.0, WIDTH + 3))


def test_the_variance_update_is_the_levels_plus_their_posterior_spread():
    A, b = normal_equations(seed=1)
    prior = LevelPrior(1.5, WIDTH)
    beta, covariance = penalized_solve(A, b, prior)
    expected = (
        beta[1:WIDTH] @ beta[1:WIDTH] + np.trace(covariance[1:WIDTH, 1:WIDTH])
    ) / prior.n_levels
    assert variance_update(beta, covariance, prior) == pytest.approx(expected)


def test_a_roster_of_one_subject_has_no_spread_to_estimate():
    prior = LevelPrior(1.0, width=1)
    with pytest.raises(ValueError, match="no deviation"):
        variance_update(np.zeros(3), np.eye(3), prior)


def test_the_spread_follows_how_far_apart_the_record_puts_the_levels():
    _, spread_weak, _ = fit_levels(*normal_equations(seed=2, level_sd=0.05), WIDTH)
    _, spread_strong, _ = fit_levels(*normal_equations(seed=2, level_sd=3.0), WIDTH)
    assert spread_strong > 20.0 * spread_weak


def test_a_record_that_separates_no_participant_pools_them_all():
    A, b = normal_equations(seed=6, level_sd=1e-6)
    _, spread, _ = fit_levels(A, b, WIDTH)
    assert spread <= 1e-3


def test_the_fit_settles_and_reports_how_many_alternations_it_took():
    A, b = normal_equations(seed=3, level_sd=2.0)
    beta, spread, taken = fit_levels(A, b, WIDTH, n_iter=200)
    assert 1 <= taken < 200
    # The value it stopped at is a fixed point of the update it stopped on.
    _, covariance = penalized_solve(A, b, LevelPrior(spread, WIDTH))
    assert variance_update(
        beta, covariance, LevelPrior(spread, WIDTH)
    ) == pytest.approx(spread, rel=1e-6)


def test_a_capped_iteration_returns_the_value_it_reached():
    A, b = normal_equations(seed=4, level_sd=2.0)
    _, _, taken = fit_levels(A, b, WIDTH, n_iter=2, tol=0.0)
    assert taken == 2


def test_shrinkage_runs_from_a_pooled_level_to_a_free_one():
    A, _ = normal_equations(seed=5)
    pooled = shrinkage(A, WIDTH, variance=1e-10)
    free = shrinkage(A, WIDTH, variance=1e12)
    assert pooled.size == WIDTH - 1
    assert np.all(pooled > 0.99)
    assert np.all(free < 0.01)
