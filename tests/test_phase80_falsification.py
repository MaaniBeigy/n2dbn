"""
Tests for the rungs of the falsification ladder.

The load-bearing pair is R5.  A synthetic pair that co-moves only on a diurnal
rhythm beats the unrepaired shift null and does not beat the repaired one, and
the repaired null holds its nominal level while the unrepaired one runs far above
it.  Both are measured over twenty seeds here, so the property is the null's
calibration.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.anchors import AnchorKey, anchor_set
from src.data.label_axis import collapse_labels, label_axis
from src.data.lifesnaps import read_lifesnaps
from src.evaluation.falsification import (
    DEFAULT_ALPHA,
    EdgeRecord,
    anchor_recall,
    e_value,
    edge_records,
    expected_false_selections,
    future_lag_control,
    invariance,
    lag1_autocorrelation,
    period_shifts,
    protocol_negatives,
    risk_ratio,
    roll_within_blocks,
    sham_match,
    sham_rung,
    sham_series,
    shift_null,
    stability_selection,
)

PERIOD = 24
DAYS = 84
SPAN = DAYS * PERIOD
BLOCKS = [(0, SPAN), (SPAN, 2 * SPAN)]


@pytest.fixture(scope="module")
def anchors():
    events, _ = read_lifesnaps()
    return anchor_set(label_axis(collapse_labels(events)[0]))


def diurnal_pair(seed: int):
    """Two series that share nothing but the clock."""
    rng = np.random.default_rng(seed)
    t = np.arange(2 * SPAN)
    rhythm = np.sin(2.0 * np.pi * (t % SPAN) / PERIOD)
    return (
        rhythm + 0.3 * rng.normal(size=t.size),
        rhythm + 0.3 * rng.normal(size=t.size),
    )


# ---------------------------------------------------------------------------
# R1 and R3
# ---------------------------------------------------------------------------


def test_r1_passes_on_an_empty_accepted_set_and_names_the_structural_share(anchors):
    rung = protocol_negatives([], anchors)
    assert rung.passed
    assert rung.denominator == 6
    assert "72 of the 78 declared negatives are structural" in rung.note
    assert "passes" in str(rung)


def test_r1_fails_when_a_forbidden_relation_is_accepted(anchors):
    forbidden = anchors.of_sign("negative", True)[0].key
    rung = protocol_negatives([forbidden], anchors)
    assert not rung.passed
    assert rung.hits == 1
    assert rung.rate == pytest.approx(1 / 6)
    assert "FAILS" in str(rung)


def test_r3_prints_its_denominator_beside_its_rate(anchors):
    positives = [a.key for a in anchors.of_sign("positive", True)]
    rung = anchor_recall(positives[:2], anchors)
    assert rung.hits == 2
    assert rung.denominator == 6
    assert rung.passed
    assert "n=6" in rung.note
    assert "0.204" in rung.note


def test_r3_fails_when_no_positive_is_recovered(anchors):
    assert not anchor_recall([], anchors).passed


def test_a_rung_with_no_denominator_reports_a_rate_of_zero():
    rung = sham_rung(0, 0)
    assert rung.rate == 0.0
    assert rung.passed


# ---------------------------------------------------------------------------
# R2
# ---------------------------------------------------------------------------


def test_the_sham_label_matches_the_marginal_and_the_heterogeneity_exactly():
    rng = np.random.default_rng(0)
    who = np.repeat(np.arange(4), 60)
    reference = np.concatenate(
        [rng.poisson(2.0 + 3.0 * unit, 60).astype(float) for unit in range(4)]
    )
    sham = sham_series(reference, who, np.random.default_rng(1))
    match = sham_match(sham, reference, who)
    assert match.marginal_distance == 0.0
    assert match.heterogeneity == 0.0
    assert match.matched(tolerance=0.15)
    assert np.array_equal(np.sort(sham), np.sort(reference))


def test_the_sham_label_reproduces_the_persistence_it_was_asked_for():
    who = np.zeros(400, dtype=np.int64)
    rng = np.random.default_rng(2)
    reference = np.cumsum(rng.normal(size=400))
    sham = sham_series(reference, who, np.random.default_rng(3), autocorrelation=0.0)
    assert abs(lag1_autocorrelation(sham, who)) < 0.2
    persistent = sham_series(
        reference, who, np.random.default_rng(3), autocorrelation=0.9
    )
    assert lag1_autocorrelation(persistent, who) > 0.5


def test_a_constant_series_has_no_persistence():
    who = np.zeros(10, dtype=np.int64)
    assert lag1_autocorrelation(np.ones(10), who) == 0.0
    assert lag1_autocorrelation(np.ones(2), np.zeros(2, dtype=np.int64)) == 0.0


def test_every_edge_touching_the_sham_is_a_false_positive():
    rung = sham_rung(3, 40)
    assert not rung.passed
    assert rung.rate == pytest.approx(0.075)
    assert "connected to nothing" in rung.note


# ---------------------------------------------------------------------------
# R4
# ---------------------------------------------------------------------------


def test_a_future_value_of_a_parent_is_silent_when_nothing_slow_drives_both():
    rng = np.random.default_rng(4)
    n = 500
    x = rng.normal(size=n)
    y = 0.8 * np.concatenate(([0.0], x[:-1])) + rng.normal(size=n)
    control = future_lag_control(
        y, np.concatenate(([0.0], x[:-1])), np.concatenate((x[1:], [0.0]))
    )
    assert control.passed()
    assert abs(control.coefficient) < 0.2


def test_a_slow_confounder_shows_up_on_the_future_value():
    rng = np.random.default_rng(5)
    n = 500
    slow = np.repeat(rng.normal(size=n // 25), 25)
    x = slow + 0.2 * rng.normal(size=n)
    y = slow + 0.2 * rng.normal(size=n)
    control = future_lag_control(
        y, np.concatenate(([0.0], x[:-1])), np.concatenate((x[1:], [0.0]))
    )
    assert not control.passed()
    assert control.p_value < 1e-6


def test_the_control_reads_the_context_columns_it_is_given():
    rng = np.random.default_rng(6)
    n = 300
    context = rng.normal(size=(n, 2))
    x = context[:, 0] + 0.1 * rng.normal(size=n)
    y = context[:, 0] + 0.1 * rng.normal(size=n)
    with_context = future_lag_control(
        y,
        np.concatenate(([0.0], x[:-1])),
        np.concatenate((x[1:], [0.0])),
        context=context,
    )
    without = future_lag_control(
        y, np.concatenate(([0.0], x[:-1])), np.concatenate((x[1:], [0.0]))
    )
    assert with_context.p_value > without.p_value


# ---------------------------------------------------------------------------
# R5, the repaired shift placebo
# ---------------------------------------------------------------------------


def test_a_purely_diurnal_pair_beats_the_unrepaired_null_and_not_the_repaired_one():
    y, x = diurnal_pair(0)
    unrepaired = shift_null(y, x, BLOCKS, period=1, n_shifts=99, seed=0)
    repaired = shift_null(y, x, BLOCKS, period=PERIOD, n_shifts=99, seed=0)
    assert unrepaired.statistic == repaired.statistic
    assert unrepaired.beats_null()
    assert not repaired.beats_null()
    assert unrepaired.p_value < repaired.p_value


def test_the_repaired_null_holds_its_level_where_the_unrepaired_one_does_not():
    unrepaired = []
    repaired = []
    for seed in range(20):
        y, x = diurnal_pair(seed)
        unrepaired.append(shift_null(y, x, BLOCKS, 1, 99, seed).p_value)
        repaired.append(shift_null(y, x, BLOCKS, PERIOD, 99, seed).p_value)
    assert all(a < b for a, b in zip(unrepaired, repaired))
    # The pair is connected by nothing but the clock, so a null at its nominal
    # level calls it a relation about one time in twenty.
    assert sum(p <= DEFAULT_ALPHA for p in repaired) <= 2
    assert sum(p <= DEFAULT_ALPHA for p in unrepaired) >= 10


def test_a_relation_carried_by_something_other_than_the_rhythm_survives_the_repair():
    rng = np.random.default_rng(7)
    n = 2 * SPAN
    t = np.arange(n)
    rhythm = np.sin(2.0 * np.pi * (t % SPAN) / PERIOD)
    driver = rng.normal(size=n)
    response = rhythm + 1.5 * np.concatenate(([0.0], driver[:-1]))
    lagged = np.concatenate(([0.0], driver[:-1]))
    assert shift_null(response, lagged, BLOCKS, PERIOD, 99, 7).beats_null()


def test_a_roll_never_moves_an_epoch_across_a_participant_boundary():
    values = np.concatenate([np.zeros(10), np.ones(10)])
    rolled = roll_within_blocks(values, [(0, 10), (10, 20)], 3)
    assert set(rolled[:10]) == {0.0}
    assert set(rolled[10:]) == {1.0}
    assert not np.array_equal(np.roll(values, 3)[:10], rolled[:10])


def test_an_empty_block_is_left_alone():
    values = np.arange(4.0)
    assert np.array_equal(roll_within_blocks(values, [(2, 2)], 1), values)


def test_a_block_that_is_not_a_whole_number_of_periods_is_refused():
    with pytest.raises(ValueError, match="whole number"):
        shift_null(np.zeros(10), np.zeros(10), [(0, 10)], period=4)


def test_a_record_too_short_to_shift_returns_an_empty_null():
    values = np.arange(24.0)
    report = shift_null(values, values, [(0, 24)], period=24, n_shifts=8)
    assert report.n_shifts == 0
    assert report.p_value == 1.0
    assert report.null == ()


def test_a_constant_parent_contributes_no_null_draw():
    n = 4 * PERIOD
    report = shift_null(
        np.arange(float(n)), np.ones(n), [(0, n)], period=PERIOD, n_shifts=8
    )
    assert report.n_shifts == 0


def test_a_period_below_one_epoch_is_not_a_period():
    with pytest.raises(ValueError, match="at least one epoch"):
        period_shifts(10, 0, 4, np.random.default_rng(0))


def test_every_offset_is_a_whole_number_of_periods():
    offsets = period_shifts(10, 7, 5, np.random.default_rng(0))
    assert len(offsets) == 5
    assert all(offset % 7 == 0 and 0 < offset < 70 for offset in offsets)


# ---------------------------------------------------------------------------
# R7
# ---------------------------------------------------------------------------


def test_an_invariant_parent_set_is_found_when_the_relation_holds_everywhere():
    rng = np.random.default_rng(8)
    n = 400
    env = np.repeat(np.arange(2), n // 2)
    x = rng.normal(size=n) + env
    design = np.column_stack([x, rng.normal(size=n)])
    y = 1.5 * x + rng.normal(size=n) * 0.3
    report = invariance(y, design, env, max_size=2)
    assert not report.model_rejected
    assert 0 in report.intersection or (0,) in report.accepted_sets


def test_a_sign_that_flips_with_the_environment_identifies_no_parent():
    rng = np.random.default_rng(9)
    n = 400
    env = np.repeat(np.arange(2), n // 2)
    x = rng.normal(size=n)
    y = np.where(env == 0, 3.0 * x, -3.0 * x) + 0.05 * rng.normal(size=n)
    report = invariance(y, x[:, None], env, max_size=1)
    # The two environments leave the same residual distribution under the empty
    # parent set, so that set is accepted and the intersection is empty, which
    # is the null this rung was expected to deliver.
    assert () in report.accepted_sets
    assert report.intersection == ()


def test_no_set_is_invariant_when_the_environment_shifts_the_response():
    rng = np.random.default_rng(9)
    n = 400
    env = np.repeat(np.arange(2), n // 2)
    x = rng.normal(size=n)
    y = 2.0 * x + 5.0 * env + 0.05 * rng.normal(size=n)
    report = invariance(y, x[:, None], env, max_size=1)
    assert report.model_rejected
    assert report.accepted_sets == ()
    assert report.intersection == ()
    assert "the deliverable is the null" in str(report)


def test_an_environment_with_one_row_is_skipped():
    rng = np.random.default_rng(10)
    n = 51
    env = np.concatenate([np.zeros(n - 1), [1]])
    x = rng.normal(size=n)
    report = invariance(x + 0.1 * rng.normal(size=n), x[:, None], env, max_size=1)
    assert not report.model_rejected
    assert "invariant sets" in str(report)


# ---------------------------------------------------------------------------
# R9 and R10
# ---------------------------------------------------------------------------


def test_a_candidate_selected_everywhere_clears_the_stability_threshold():
    def select(units):
        return [0, 1] if len(units) % 2 == 0 else [0]

    report = stability_selection(select, range(10), n_candidates=4, n_pairs=8)
    assert 0 in report.selected
    assert report.frequency[0] == 1.0
    assert report.frequency[3] == 0.0
    assert report.expected_false > 0.0
    assert "complementary pairs" in str(report)


def test_the_bound_needs_a_threshold_above_one_half():
    with pytest.raises(ValueError, match="threshold in"):
        expected_false_selections(4.0, 20, 0.5)
    assert expected_false_selections(4.0, 0, 0.75) == 0.0
    assert expected_false_selections(4.0, 20, 0.75) == pytest.approx(1.6)


def test_the_bound_falls_as_the_threshold_rises():
    assert expected_false_selections(5.0, 30, 0.9) < expected_false_selections(
        5.0, 30, 0.6
    )


def test_the_e_value_of_no_association_is_one_and_rises_with_the_ratio():
    assert e_value(1.0) == pytest.approx(1.0)
    assert e_value(2.0) == pytest.approx(2.0 + 2.0**0.5)
    assert e_value(0.5) == pytest.approx(e_value(2.0))
    with pytest.raises(ValueError, match="is positive"):
        e_value(0.0)


def test_a_standardized_effect_becomes_a_risk_ratio():
    assert risk_ratio(0.0, 1.0) == pytest.approx(1.0)
    assert risk_ratio(1.0, 1.0) > 1.0
    with pytest.raises(ValueError, match="positive spread"):
        risk_ratio(1.0, 0.0)


def test_every_reported_edge_carries_its_three_numbers():
    keys = [AnchorKey("hour", 1, "a", "b"), AnchorKey("day", 2, "b", "a")]
    stability = stability_selection(
        lambda units: [0], range(6), n_candidates=2, n_pairs=4
    )
    controls = [
        future_lag_control(np.arange(20.0), np.arange(20.0), np.zeros(20)),
        future_lag_control(np.arange(20.0), np.arange(20.0), np.zeros(20)),
    ]
    records = edge_records(keys, stability, [2.0, 0.5], controls)
    assert [isinstance(r, EdgeRecord) for r in records] == [True, True]
    assert records[0].inclusion == 1.0
    assert records[1].inclusion == 0.0
    assert records[0].e_value == pytest.approx(records[1].e_value)
    row = records[0].as_row()
    assert set(row) == {
        "scale",
        "lag",
        "source",
        "target",
        "inclusion",
        "e_value",
        "future_lag_p",
    }


def test_a_mismatched_bundle_of_edge_evidence_is_refused():
    stability = stability_selection(
        lambda units: [0], range(4), n_candidates=1, n_pairs=2
    )
    with pytest.raises(ValueError, match="against"):
        edge_records([AnchorKey("day", 1, "a", "b")], stability, [1.0, 2.0], [])
