"""
tests/test_phase58_missingness.py
=================================
The missingness declaration, and the two places a gap still reaches the fit.

A record states the mechanism its gaps are assumed to have, and the tests here
fix what the record can refuse.  Gaps that fall at night are clumped and follow
the clock, so the ignorable-given-calendar path stands and a claim of complete
randomness does not.  Gaps that swallow whole days are clumped and orthogonal
to every calendar column, so the ignorable path is refused for them.  Gaps
scattered one slot at a time match a reshuffle of their own rate, so a complete
randomness claim stands.

The rest fixes the arithmetic the fine formulation leaves behind.  A start
hidden inside a gap corrupts the risk set after it, so the mask is dilated
forward to the first recorded start or to a cap read off the longest observed
episode.  A partly recorded block is a defective regressor, so coverage scaling
recovers its rate and a floor removes the finest slots whose parent reads a
block that is too empty to read.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from src.core.calendar import SlotCalendar
from src.core.features import calendar_blocks, day_of_week, hour_of_day
from src.evaluation.missingness import (
    DEFAULT_ALPHA,
    MECHANISMS,
    FeatureExposure,
    audit,
    block_coverage,
    check,
    coverage_scaled,
    dilate_after_gaps,
    duration_cap,
    feature_exposure,
    gap_runs,
    gap_structure,
    predictability,
    readable_slots,
    slot_design,
    structure_dependence,
)

DAY = 24
N_DAYS = 14
N_SLOTS = DAY * N_DAYS
PERMUTATIONS = 99


# ---------------------------------------------------------------------------
# Records the tests are read against
# ---------------------------------------------------------------------------


def clock_design(n_slots: int = N_SLOTS) -> np.ndarray:
    """Night, morning and a horizon trend, at the finest slots."""
    hour = np.arange(n_slots) % DAY
    return np.column_stack(
        [
            (hour < 6).astype(float),
            ((hour >= 6) & (hour < 12)).astype(float),
            np.linspace(0.0, 1.0, n_slots),
        ]
    )


def night_gaps(n_slots: int = N_SLOTS) -> np.ndarray:
    """A device that sleeps from midnight to six, every day."""
    hour = np.arange(n_slots) % DAY
    return ~(hour < 6)


def whole_day_gaps(n_slots: int = N_SLOTS) -> np.ndarray:
    """Two whole days lost, which covers every hour of the clock equally."""
    observed = np.ones(n_slots, dtype=bool)
    observed[2 * DAY : 3 * DAY] = False
    observed[8 * DAY : 9 * DAY] = False
    return observed


def scattered_gaps(n_slots: int = N_SLOTS, n_missing: int = 34) -> np.ndarray:
    """Single slots lost at random positions, at the same rate as one whole day."""
    rng = np.random.default_rng(11)
    observed = np.ones(n_slots, dtype=bool)
    observed[rng.choice(n_slots, size=n_missing, replace=False)] = False
    return observed


# ---------------------------------------------------------------------------
# The design the gaps are tested against
# ---------------------------------------------------------------------------


def test_slot_design_broadcasts_every_scale_to_the_finest_grid():
    blocks = [np.arange(6.0).reshape(6, 1), np.array([[1.0, 2.0], [3.0, 4.0]])]
    ancestors = [np.arange(6), np.arange(6) // 3]
    design = slot_design(blocks, ancestors)
    assert design.shape == (6, 3)
    assert np.allclose(design[:, 0], np.arange(6.0))
    assert np.allclose(design[:3, 1], 1.0)
    assert np.allclose(design[3:, 2], 4.0)


def test_slot_design_refuses_no_block_and_a_map_it_cannot_pair():
    with pytest.raises(ValueError, match="at least one scale"):
        slot_design([], [])
    with pytest.raises(ValueError, match="ancestor maps"):
        slot_design([np.zeros((4, 1))], [np.arange(4), np.arange(4)])


def test_a_calendar_ladder_builds_a_design_the_gaps_can_be_read_against():
    cal = SlotCalendar.from_config(
        branching=[DAY, 7], slot_minutes=60, origin=datetime(2026, 1, 5)
    )
    counts = [N_SLOTS, N_DAYS, 2]
    blocks = calendar_blocks(cal, [hour_of_day(), day_of_week()], counts)
    ancestors = [
        np.arange(N_SLOTS),
        np.arange(N_SLOTS) // DAY,
        np.arange(N_SLOTS) // (DAY * 7),
    ]
    design = slot_design(blocks.blocks, ancestors)
    assert design.shape[0] == N_SLOTS
    verdict = check(
        "ignorable_given_calendar",
        night_gaps(),
        design,
        n_permutations=PERMUTATIONS,
    )
    assert verdict.accepted
    assert verdict.predictability.r2 > 0.5


# ---------------------------------------------------------------------------
# The shared checks
# ---------------------------------------------------------------------------


def test_a_zero_one_indicator_is_refused_as_a_mask():
    with pytest.raises(ValueError, match="boolean mask"):
        gap_structure(np.ones(20, dtype=np.int64))


def test_a_gap_indicator_is_one_column_at_a_time():
    with pytest.raises(ValueError, match="one entry per finest slot"):
        gap_structure(np.ones((20, 2), dtype=bool))


def test_a_record_with_no_gap_and_a_record_with_no_observation_are_refused():
    with pytest.raises(ValueError, match="no gap"):
        gap_structure(np.ones(20, dtype=bool))
    with pytest.raises(ValueError, match="no slot"):
        gap_structure(np.zeros(20, dtype=bool))


def test_the_level_lies_inside_the_unit_interval():
    observed = scattered_gaps()
    with pytest.raises(ValueError, match=r"lies in \(0, 1\)"):
        gap_structure(observed, alpha=0.0)
    with pytest.raises(ValueError, match=r"lies in \(0, 1\)"):
        gap_structure(observed, alpha=1.0)


def test_too_few_permutations_cannot_reach_the_declared_level():
    with pytest.raises(ValueError, match="never reaches the level"):
        gap_structure(scattered_gaps(), alpha=DEFAULT_ALPHA, n_permutations=5)


def test_gap_runs_counts_maximal_spells():
    gap = np.array([True, True, False, True, False, False, True])
    assert list(gap_runs(gap)) == [2, 1, 1]
    assert list(gap_runs(np.zeros(4, dtype=bool))) == []


# ---------------------------------------------------------------------------
# Do the gaps clump
# ---------------------------------------------------------------------------


def test_whole_day_gaps_clump_beyond_a_reshuffle_of_their_rate():
    structure = gap_structure(whole_day_gaps(), n_permutations=PERMUTATIONS)
    assert structure.n_missing == 2 * DAY
    assert structure.longest_gap == DAY
    assert structure.mean_gap == pytest.approx(DAY)
    assert structure.mean_gap > structure.mean_gap_null
    assert structure.clustered
    assert structure.missing_fraction == pytest.approx(2.0 / N_DAYS)


def test_scattered_gaps_match_a_reshuffle_of_their_rate():
    structure = gap_structure(scattered_gaps(), n_permutations=PERMUTATIONS)
    assert not structure.clustered
    assert structure.mean_gap < 2.0
    assert structure.p_value > DEFAULT_ALPHA


# ---------------------------------------------------------------------------
# Do the calendar and context columns predict the gaps
# ---------------------------------------------------------------------------


def test_night_gaps_are_predicted_by_the_clock():
    result = predictability(
        night_gaps(), clock_design(), n_permutations=PERMUTATIONS, seed=3
    )
    assert result.predictable
    assert result.r2 > 0.9
    assert result.r2 > result.r2_null
    # One night is six slots, so the null permutes blocks of six and every
    # draw carries the same run lengths the record has.
    assert result.block == 6
    assert result.n_columns == 3


def test_whole_day_gaps_are_orthogonal_to_every_clock_column():
    result = predictability(
        whole_day_gaps(), clock_design(), n_permutations=PERMUTATIONS, seed=3
    )
    assert not result.predictable
    assert result.r2 < 0.1


def test_predictability_refuses_a_design_it_cannot_read():
    observed = whole_day_gaps()
    with pytest.raises(ValueError, match="one row per finest slot"):
        predictability(observed, clock_design(N_SLOTS - 1))
    with pytest.raises(ValueError, match="no columns"):
        predictability(observed, np.zeros((N_SLOTS, 0)))


def test_a_series_too_short_for_the_folds_is_refused():
    observed = np.ones(8, dtype=bool)
    observed[2:4] = False
    with pytest.raises(ValueError, match="nothing to hold out"):
        predictability(observed, clock_design(8), n_permutations=PERMUTATIONS)
    with pytest.raises(ValueError, match="nothing to hold out"):
        predictability(
            observed, clock_design(8), n_permutations=PERMUTATIONS, n_folds=1
        )


# ---------------------------------------------------------------------------
# The declaration
# ---------------------------------------------------------------------------


def test_an_unknown_mechanism_is_refused():
    with pytest.raises(ValueError, match="declared mechanism"):
        check("missing_at_random", whole_day_gaps(), clock_design())
    assert MECHANISMS == ("mcar", "ignorable_given_calendar", "nonignorable")


def test_a_record_with_no_gap_never_exercises_the_mechanism():
    verdict = check("mcar", np.ones(N_SLOTS, dtype=bool), clock_design(), label="sleep")
    assert verdict.accepted
    assert verdict.structure is None
    assert verdict.predictability is None
    assert "no gap" in str(verdict)


def test_complete_randomness_is_refused_when_the_gaps_clump_and_follow_the_clock():
    verdict = check(
        "mcar", night_gaps(), clock_design(), n_permutations=PERMUTATIONS, seed=3
    )
    assert not verdict.accepted
    assert "clump" in verdict.reason
    assert "predict the gaps" in verdict.reason
    assert "REFUSED" in str(verdict)


def test_complete_randomness_stands_on_scattered_unrelated_gaps():
    verdict = check(
        "mcar", scattered_gaps(), clock_design(), n_permutations=PERMUTATIONS, seed=3
    )
    assert verdict.accepted
    assert "exchangeable in time" in verdict.reason


def test_the_ignorable_path_stands_when_the_clock_explains_the_gaps():
    verdict = check(
        "ignorable_given_calendar",
        night_gaps(),
        clock_design(),
        n_permutations=PERMUTATIONS,
        seed=3,
    )
    assert verdict.accepted
    assert "covers the gap process" in verdict.reason


def test_the_ignorable_path_is_refused_when_the_clumping_is_unexplained():
    verdict = check(
        "ignorable_given_calendar",
        whole_day_gaps(),
        clock_design(),
        n_permutations=PERMUTATIONS,
        seed=3,
    )
    assert not verdict.accepted
    assert "outside the conditioning set" in verdict.reason


def test_the_ignorable_path_stands_on_gaps_that_stay_exchangeable():
    verdict = check(
        "ignorable_given_calendar",
        scattered_gaps(),
        clock_design(),
        n_permutations=PERMUTATIONS,
        seed=3,
    )
    assert verdict.accepted
    assert "exchangeable in time" in verdict.reason


def test_an_informative_gap_process_is_accepted_and_keeps_the_ignorable_path_closed():
    verdict = check(
        "nonignorable",
        whole_day_gaps(),
        clock_design(),
        n_permutations=PERMUTATIONS,
        seed=3,
    )
    assert verdict.accepted
    assert "stays closed" in verdict.reason
    assert verdict.structure.clustered


# ---------------------------------------------------------------------------
# The audit over labels
# ---------------------------------------------------------------------------


def test_one_declaration_is_judged_against_every_label():
    mask = np.column_stack([night_gaps(), whole_day_gaps()])
    result = audit(
        "ignorable_given_calendar",
        mask,
        clock_design(),
        labels=["sleep", "walk"],
        n_permutations=PERMUTATIONS,
        seed=3,
    )
    assert [v.label for v in result.verdicts] == ["sleep", "walk"]
    assert result.verdicts[0].accepted
    assert not result.verdicts[1].accepted
    assert not result.accepted
    text = result.report()
    assert "missingness declared 'ignorable_given_calendar'" in text
    assert "does not stand for walk" in text


def test_an_accepted_audit_names_no_refusal_and_a_single_column_is_one_label():
    result = audit(
        "nonignorable",
        night_gaps(),
        clock_design(),
        n_permutations=PERMUTATIONS,
        seed=3,
    )
    assert result.accepted
    assert [v.label for v in result.verdicts] == ["L0"]
    assert "does not stand" not in result.report()


def test_the_audit_refuses_a_mask_it_cannot_index_and_names_it_cannot_pair():
    with pytest.raises(ValueError, match="indexed by slot and label"):
        audit("mcar", np.ones((4, 2, 2), dtype=bool), clock_design(4))
    with pytest.raises(ValueError, match="label names against"):
        audit(
            "mcar",
            np.ones((N_SLOTS, 2), dtype=bool),
            clock_design(),
            labels=["only_one"],
        )


# ---------------------------------------------------------------------------
# The risk set after a gap
# ---------------------------------------------------------------------------


def test_the_cap_is_read_off_the_longest_episode_the_label_was_seen_to_have():
    starts = np.array([[True, False], [False, False], [True, False]])
    marks = np.array([[3, 0], [0, 0], [5, 0]])
    assert list(duration_cap(starts, marks)) == [4, 0]


def test_the_cap_refuses_marks_that_are_not_shaped_like_the_starts():
    with pytest.raises(ValueError, match="indexed by slot and label"):
        duration_cap(np.ones((3, 2), dtype=bool), np.zeros((3, 3)))


def test_slots_after_a_gap_leave_the_risk_set_until_a_start_proves_it_open():
    observed = np.ones((10, 1), dtype=bool)
    observed[2:4, 0] = False
    starts = np.zeros((10, 1), dtype=bool)
    starts[7, 0] = True
    out = dilate_after_gaps(observed, starts, 6)
    # The gap ends at slot 3, so slots 4 to 6 are ambiguous and slot 7 records
    # a start, which proves the label was at risk there and stops the dilation.
    assert list(out[:, 0]) == [
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
    ]


def test_the_cap_bounds_the_dilation_and_a_gap_at_the_end_has_nothing_after_it():
    observed = np.ones((8, 1), dtype=bool)
    observed[1, 0] = False
    observed[6:, 0] = False
    starts = np.zeros((8, 1), dtype=bool)
    out = dilate_after_gaps(observed, starts, np.array([2]))
    assert list(out[:, 0]) == [True, False, False, False, True, True, False, False]


def test_a_cap_of_zero_leaves_the_mask_alone():
    observed = np.ones((6, 2), dtype=bool)
    observed[2, 0] = False
    starts = np.zeros((6, 2), dtype=bool)
    assert np.array_equal(dilate_after_gaps(observed, starts, 0), observed)


def test_one_flag_per_slot_covers_every_label_at_that_slot():
    observed = np.ones(6, dtype=bool)
    observed[2] = False
    starts = np.zeros((6, 2), dtype=bool)
    starts[5, 0] = True
    out = dilate_after_gaps(observed, starts, 4)
    assert list(out[:, 0]) == [True, True, False, False, False, True]
    assert list(out[:, 1]) == [True, True, False, False, False, False]


def test_the_dilation_refuses_a_shape_or_a_cap_it_cannot_use():
    observed = np.ones((6, 2), dtype=bool)
    starts = np.zeros((6, 2), dtype=bool)
    with pytest.raises(ValueError, match="a start indicator is indexed"):
        dilate_after_gaps(np.ones(6, dtype=bool), np.zeros(6, dtype=bool), 1)
    with pytest.raises(ValueError, match="both are"):
        dilate_after_gaps(observed, np.zeros((6, 3), dtype=bool), 1)
    with pytest.raises(ValueError, match="one per label"):
        dilate_after_gaps(observed, starts, np.array([1, 2, 3]))
    with pytest.raises(ValueError, match="starts at zero"):
        dilate_after_gaps(observed, starts, -1)


# ---------------------------------------------------------------------------
# Coverage of a coarse block
# ---------------------------------------------------------------------------


def test_coverage_is_the_recorded_share_of_each_epoch():
    observed = np.array([True, True, False, False, True, True, True, True])
    ancestors = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    assert list(block_coverage(observed, ancestors, 2)) == [0.5, 1.0]


def test_coverage_reads_each_label_on_its_own_and_an_empty_epoch_reads_zero():
    observed = np.array([[True, False], [True, True]])
    ancestors = np.array([0, 0])
    cov = block_coverage(observed, ancestors, 3)
    assert cov.shape == (3, 2)
    assert np.allclose(cov[0], [1.0, 0.5])
    assert np.allclose(cov[2], 0.0)


def test_coverage_refuses_an_ancestor_map_of_another_length():
    with pytest.raises(ValueError, match="one entry per finest slot"):
        block_coverage(np.ones(4, dtype=bool), np.array([0, 0]), 1)


def test_scaling_raises_an_observed_sum_to_the_whole_block():
    # A block of four slots holding one event each, half of it recorded.
    assert coverage_scaled(np.array([2.0]), np.array([0.5]))[0] == pytest.approx(4.0)
    assert coverage_scaled(np.array([0.0]), np.array([0.0]))[0] == 0.0
    with pytest.raises(ValueError, match="indexed by epoch and label"):
        coverage_scaled(np.zeros((2, 2)), np.zeros(2))


def test_a_slot_reading_an_empty_block_or_a_block_before_the_record_is_dropped():
    coverage = np.array([0.0, 0.4, 1.0])
    ancestors = np.array([0, 0, 1, 1, 2, 2])
    lag = 1
    good = readable_slots(coverage, ancestors, lag)
    # Slots in epoch 0 read epoch -1, which the record does not hold; slots in
    # epoch 1 read epoch 0, which holds nothing that was recorded.
    assert list(good) == [False, False, False, False, True, True]
    assert list(readable_slots(coverage, ancestors, lag, floor=0.5)) == [
        False,
        False,
        False,
        False,
        False,
        False,
    ]


def test_readable_slots_refuses_a_lag_a_floor_or_a_map_it_cannot_use():
    coverage = np.array([1.0, 1.0])
    ancestors = np.array([0, 1])
    with pytest.raises(ValueError, match="starts at zero"):
        readable_slots(coverage, ancestors, -1)
    with pytest.raises(ValueError, match=r"lies in \[0, 1\)"):
        readable_slots(coverage, ancestors, 1, floor=1.0)
    with pytest.raises(ValueError, match="coverage column holds"):
        readable_slots(coverage, np.array([0, 2]), 1)


def test_the_exposure_counts_the_rows_that_stand_on_the_coverage_rules():
    coverage = np.array([0.0, 0.4, 1.0])
    ancestors = np.array([0, 0, 1, 1, 2, 2])
    exposure = feature_exposure(coverage, ancestors, 1)
    assert exposure.n_slots == 6
    assert exposure.n_dropped == 4
    assert exposure.n_scaled == 2
    assert exposure.dropped_fraction == pytest.approx(2.0 / 3.0)
    assert exposure.scaled_fraction == pytest.approx(1.0 / 3.0)
    assert "2 kept on a scaled block" in str(exposure)


def test_an_empty_column_carries_no_exposure():
    empty = feature_exposure(np.array([1.0]), np.zeros(0, dtype=np.int64), 0)
    assert empty == FeatureExposure(0, 0, 0)
    assert empty.dropped_fraction == 0.0
    assert empty.scaled_fraction == 0.0


def test_structure_dependence_counts_what_a_second_coverage_rule_keeps():
    kept = structure_dependence({("a", 1), ("b", 2)}, {("a", 1), ("c", 3)})
    assert kept["baseline"] == 2.0
    assert kept["shared"] == 1.0
    assert kept["kept"] == pytest.approx(0.5)
    assert kept["changed"] == 2.0
    assert structure_dependence([], [])["kept"] == 0.0
