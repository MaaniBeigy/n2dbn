"""
tests/test_phase36_sufficiency.py
=================================
Whether a scale can support a structural claim, judged before anything is fitted.

A coarse scale holds a small fraction of the epochs the finest one does, and
below some count no estimator recovers a graph there. Searching anyway returns
an empty graph, which reads as an answer when it is a shrug. These tests pin the
verdict to the two things that decide it, the epochs a scale has and the number
of candidate edges it must choose between, and pin the reported reason to the
one that actually fired.
"""

from __future__ import annotations

import pytest

from src.evaluation.sufficiency import (
    ABSOLUTE_MIN_EPOCHS,
    MIN_EPOCHS_PER_EDGE,
    RECALL_CURVE,
    epochs_for_recall,
    expected_recall,
    horizon_for_recall,
    ledger,
    min_epochs_per_subject,
    n_candidate_edges,
    pooled_design,
    report,
    scale_verdict,
    subjects_for_recall,
    supported_scales,
)

# ---------------------------------------------------------------------------
# The candidate count
# ---------------------------------------------------------------------------


def test_every_label_is_a_candidate_at_every_lag():
    assert n_candidate_edges(5, 2) == 10


def test_a_scale_with_no_lag_budget_has_nothing_to_choose_between():
    assert n_candidate_edges(5, 0) == 0


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------


def test_a_long_scale_against_a_small_candidate_set_expects_high_recall():
    got = scale_verdict(k=3, n_epochs=336, n_labels=5, max_lag=1)
    assert got.supported
    assert got.n_candidates == 5
    assert got.epochs_per_candidate == pytest.approx(67.2)
    assert got.expected_recall == pytest.approx(0.75)


def test_a_scale_with_almost_no_epochs_is_refused_outright():
    # Under the floor the candidate count cannot rescue it, so the reason names
    # the epochs rather than the ratio.
    got = scale_verdict(k=3, n_epochs=8, n_labels=1, max_lag=1)
    assert not got.supported
    assert "fewer than" in got.reason


def test_a_scale_below_where_recovery_begins_is_refused():
    # The sweep recovered nothing at 1.6 epochs per candidate and still returned
    # edges, which is the one regime where a graph is worse than no graph.
    got = scale_verdict(k=3, n_epochs=12, n_labels=5, max_lag=1)
    assert got.epochs_per_candidate < 3.0
    assert not got.supported
    assert "returned edges there anyway" in got.reason


def test_the_same_epochs_buy_less_recall_against_a_larger_candidate_set():
    """What decides it is epochs against choices, not epochs alone."""
    many = scale_verdict(k=2, n_epochs=100, n_labels=5, max_lag=3)
    few = scale_verdict(k=2, n_epochs=100, n_labels=2, max_lag=1)
    assert many.expected_recall < few.expected_recall


# ---------------------------------------------------------------------------
# The measured curve
# ---------------------------------------------------------------------------


def test_recall_never_falls_as_epochs_are_added():
    xs = [0.5, 1.6, 3.4, 6.8, 13.4, 23.6, 47.0, 200.0]
    got = [expected_recall(x) for x in xs]
    assert all(b >= a for a, b in zip(got, got[1:]))


def test_the_curve_is_read_off_at_its_measured_points():
    for x, y in RECALL_CURVE:
        assert expected_recall(x) == pytest.approx(y)


def test_a_count_between_two_measurements_lands_between_them():
    got = expected_recall(10.0)
    assert 0.250 < got < 0.375


def test_a_count_beyond_the_sweep_claims_no_more_than_the_sweep_saw():
    # Extrapolating past the measured range would be inventing a number.
    assert expected_recall(10_000.0) == pytest.approx(0.750)


def test_a_count_below_the_sweep_claims_nothing():
    assert expected_recall(0.1) == pytest.approx(0.0)


def test_the_thresholds_are_parameters_rather_than_constants():
    strict = scale_verdict(
        k=3, n_epochs=100, n_labels=5, max_lag=1, min_epochs_per_edge=50.0
    )
    loose = scale_verdict(
        k=3, n_epochs=100, n_labels=5, max_lag=1, min_epochs_per_edge=5.0
    )
    assert not strict.supported
    assert loose.supported


def test_the_shipped_numbers_are_the_measured_ones():
    # These came from a sweep, so a change to them is a change to a measurement.
    assert MIN_EPOCHS_PER_EDGE > 0
    assert ABSOLUTE_MIN_EPOCHS >= 3
    assert RECALL_CURVE[0][1] == 0.0


def test_a_verdict_says_which_scale_what_it_had_and_what_to_expect():
    text = str(scale_verdict(k=3, n_epochs=336, n_labels=5, max_lag=1))
    assert "scale 3" in text and "336 epochs" in text
    assert "expect recall about 0.75" in text


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------


def test_the_ledger_judges_every_scale_finest_first():
    got = ledger([20160, 2016, 336], n_labels=5, max_lag=[2, 2, 1])
    assert [v.k for v in got] == [1, 2, 3]
    assert [v.n_epochs for v in got] == [20160, 2016, 336]
    assert all(v.supported for v in got)


def test_a_coarsest_scale_of_one_day_expects_far_less_than_one_of_two_weeks():
    short = ledger([1440, 144, 24], n_labels=5, max_lag=[2, 2, 1])[-1]
    long = ledger([20160, 2016, 336], n_labels=5, max_lag=[2, 2, 1])[-1]
    assert short.expected_recall < long.expected_recall / 2


def test_a_hierarchy_below_the_floor_is_refused_at_that_scale_alone():
    got = ledger([720, 72, 12], n_labels=5, max_lag=[2, 2, 1])
    assert supported_scales(got) == {1: True, 2: True, 3: False}


def test_a_lag_bound_missing_for_a_scale_is_an_error_rather_than_a_guess():
    with pytest.raises(ValueError, match="scale count mismatch"):
        ledger([20160, 2016, 336], n_labels=5, max_lag=[2, 2])


def test_the_report_names_the_scales_that_cannot_answer():
    text = report(ledger([720, 72, 12], n_labels=5, max_lag=[2, 2, 1]))
    assert "no structural claim is supported at scale 3" in text
    assert "without evidence" in text


def test_a_report_with_nothing_to_refuse_makes_no_such_claim():
    text = report(ledger([20160, 2016, 336], n_labels=5, max_lag=[2, 2, 1]))
    assert "no structural claim" not in text


# ---------------------------------------------------------------------------
# The inverse, which is what a study design reads
# ---------------------------------------------------------------------------


def test_the_inverse_returns_the_curve_it_came_from():
    for x, y in RECALL_CURVE:
        got = epochs_for_recall(y, n_labels=5, max_lag=1)
        assert got == pytest.approx(x * 5)


def test_more_recall_costs_more_epochs():
    low = epochs_for_recall(0.25, n_labels=5, max_lag=1)
    high = epochs_for_recall(0.625, n_labels=5, max_lag=1)
    assert high > low


def test_recall_beyond_the_sweep_cannot_be_promised():
    # Claiming a number the sweep never reached would be inventing it.
    assert epochs_for_recall(0.95, n_labels=5, max_lag=1) == float("inf")


def test_a_scale_with_no_candidates_needs_no_epochs():
    assert epochs_for_recall(0.5, n_labels=5, max_lag=0) == 0.0


def test_a_coarser_scale_costs_more_of_the_series_it_is_built_from():
    """One coarse epoch is many fine ones, so the same count costs far more."""
    hour = horizon_for_recall(0.375, n_labels=5, max_lag=1, epochs_below=60)
    week = horizon_for_recall(0.375, n_labels=5, max_lag=1, epochs_below=10080)
    assert week == pytest.approx(hour * 168)


def test_a_week_scale_claim_costs_more_than_a_year_of_minutes():
    # The number that decides whether an n-of-1 study can answer at all.
    minutes = horizon_for_recall(0.375, n_labels=5, max_lag=1, epochs_below=10080)
    assert minutes / 1440 > 365


def test_a_target_below_the_curve_costs_the_least_the_sweep_measured():
    """A target no measurement is needed for still returns the smallest count."""
    got = epochs_for_recall(-0.1, n_labels=5, max_lag=1)
    assert got == pytest.approx(RECALL_CURVE[0][0] * 5)


# ---------------------------------------------------------------------------
# Pooling across subjects
# ---------------------------------------------------------------------------


def test_a_subject_shorter_than_its_own_lag_carries_no_transition():
    """Pooling adds subjects and cannot buy a subject the epochs it lacks."""
    assert min_epochs_per_subject(1) == 3
    assert subjects_for_recall(0.375, 5, 1, epochs_per_subject=2) == float("inf")


def test_subjects_and_epochs_trade_against_each_other():
    few_long = subjects_for_recall(0.375, 5, 1, epochs_per_subject=52)
    many_short = subjects_for_recall(0.375, 5, 1, epochs_per_subject=4)
    assert many_short > few_long > 0


def test_more_recall_needs_more_subjects_at_the_same_length():
    low = subjects_for_recall(0.375, 5, 1, epochs_per_subject=13)
    high = subjects_for_recall(0.750, 5, 1, epochs_per_subject=13)
    assert high > low


def test_recall_beyond_the_sweep_is_unreachable_by_pooling_too():
    assert subjects_for_recall(0.95, 5, 1, epochs_per_subject=52) == float("inf")


def test_a_target_needing_nothing_needs_no_subjects():
    assert subjects_for_recall(0.0, 5, 0, epochs_per_subject=52) == 0.0


def test_the_design_table_rounds_up_because_half_a_subject_collects_nothing():
    rows = pooled_design(0.375, 5, 1, [13])
    assert rows[0]["subjects"] == float(int(rows[0]["subjects"]))
    assert rows[0]["total_epochs"] >= epochs_for_recall(0.375, 5, 1)


def test_the_design_table_marks_the_shapes_that_cannot_work():
    rows = pooled_design(0.375, 5, 1, [2, 13])
    assert rows[0]["subjects"] == float("inf")
    assert rows[1]["subjects"] < float("inf")
