"""
Tests for the four screens, and for the two properties that make them safe.

The pooled pre-score keeps a relation visible at one scale alone, which is the
escape hatch a pooled ordering needs before it may be used at all, and the
superset bound is above every gain any set of columns can reach, which is what
makes closing a target sound.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.learning.fine_sstep import BaseFit, candidate_gains
from src.learning.screens import (
    ScreenReport,
    SweepCache,
    closes,
    log_bayes_factor,
    pooled_pre_score,
    saturated_bound,
    screen_agreement,
    screen_by_bayes_factor,
)


def test_the_bayes_factor_charges_a_candidate_for_the_curvature_it_uses():
    gains = np.array([4.0, 4.0])
    curvature = np.array([1.0, 100.0])
    values = log_bayes_factor(gains, curvature, prior_variance=1.0)
    assert values[0] > values[1]
    assert values[0] == pytest.approx(4.0 - 0.5 * np.log(2.0))
    # A candidate the data say nothing about keeps its gain whole.
    assert log_bayes_factor(gains, np.zeros(2))[0] == pytest.approx(4.0)


def test_the_bayes_factor_refuses_arguments_it_cannot_read():
    with pytest.raises(ValueError, match="prior variance is positive"):
        log_bayes_factor(np.ones(2), np.ones(2), prior_variance=0.0)
    with pytest.raises(ValueError, match="do not describe one candidate field"):
        log_bayes_factor(np.ones(2), np.ones(3))


def test_the_screen_keeps_what_clears_the_threshold_and_counts_the_rest():
    gains = np.array([10.0, 0.1, 6.0, 0.0])
    curvature = np.ones(4)
    keep, report = screen_by_bayes_factor(gains, curvature, threshold=1.0)
    assert keep.tolist() == [True, False, True, False]
    assert (report.considered, report.kept, report.dropped) == (4, 2, 2)
    assert report.share_kept == 0.5
    assert report.name == "bayes_factor"


def test_an_empty_screen_reports_that_it_kept_everything():
    assert ScreenReport("nothing").share_kept == 1.0


def test_a_relation_visible_at_one_scale_alone_survives_the_pooled_ordering():
    # Candidate 1 shows at one resolution and nowhere else, so its pooled score
    # sits under the bar and its own-scale score does not.
    per_scale = [np.array([5.0, 0.0, 0.0]), np.array([5.0, 0.0, 0.0])]
    own_scale = np.array([5.0, 6.0, 0.2])
    keep, report = pooled_pre_score(per_scale, own_scale, bar=1.0)
    assert keep.tolist() == [True, True, False]
    assert report.kept == 2


def test_the_pooled_pre_score_refuses_a_field_it_cannot_pool():
    with pytest.raises(ValueError, match="at least one scale"):
        pooled_pre_score([], np.ones(3), bar=1.0)
    with pytest.raises(ValueError, match="pooled field"):
        pooled_pre_score([np.ones(3)], np.ones(4), bar=1.0)


def test_no_set_of_columns_reaches_past_the_saturated_bound():
    rng = np.random.default_rng(0)
    n, p = 40, 5
    weights = rng.uniform(0.5, 2.0, size=n)
    residual = rng.normal(size=n)
    columns = rng.normal(size=(n, p))
    bound = saturated_bound(residual, weights)

    base = BaseFit(A=np.eye(1) * 1e6, b=np.zeros(1))
    cross = np.zeros((p, 1))
    diagonal = np.einsum("tp,t,tp->p", columns, weights, columns)
    targets = columns.T @ residual
    gains = candidate_gains(base, cross, diagonal, targets)
    assert float(gains.max()) <= bound

    # The whole set at once is still under it, which is what "superset" means.
    gram = columns.T @ (columns * weights[:, None])
    whole = 0.5 * targets @ np.linalg.solve(gram, targets)
    assert whole <= bound + 1e-9


def test_the_bound_refuses_a_weight_that_would_make_it_no_bound():
    with pytest.raises(ValueError, match="non-negative"):
        saturated_bound(np.ones(3), np.array([1.0, -1.0, 1.0]))


def test_a_target_closes_when_nothing_it_could_buy_clears_its_price():
    assert closes(bound=2.0, cheapest_price=5.0)
    assert not closes(bound=9.0, cheapest_price=5.0)


def test_a_parent_set_reached_twice_inside_one_sweep_is_computed_once():
    cache = SweepCache()
    calls = []

    def compute():
        calls.append(1)
        return 3.5

    assert cache.get(1, 0, "count", [2, 7], compute) == 3.5
    # The same set in a different order is the same set.
    assert cache.get(1, 0, "count", [7, 2], compute) == 3.5
    assert len(calls) == 1
    assert (cache.hits, cache.misses) == (1, 1)
    assert cache.share_hit == 0.5

    # A different target is a different key.
    cache.get(1, 1, "count", [2, 7], compute)
    assert len(calls) == 2

    cache.clear()
    assert cache.values == {}
    assert cache.share_hit == 0.0


def test_the_agreement_names_what_a_screened_run_missed():
    assert screen_agreement([1, 2], [1, 2, 5, 3]) == [3, 5]
    assert screen_agreement([1, 2, 3], [1, 2]) == []
