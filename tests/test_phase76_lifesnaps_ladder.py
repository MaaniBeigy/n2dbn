"""
Tests for the epoch ladder built on participant time.

What is pinned here is that the ladder never lets one participant's epochs reach
another's, that a lagged coarse read stays in the past because every parent map
is non-decreasing, that the counts reconcile up the ladder, and that a day the
record does not cover is absent at every scale.
"""

from __future__ import annotations

import hashlib
from datetime import date

import numpy as np
import pytest

from src.data.label_axis import SCHEDULED_LABEL
from src.data.lifesnaps import RAW_CSV
from src.data.lifesnaps_ladder import (
    DAYS_PER_WEEK,
    LADDER_MINUTES,
    LADDER_UNITS,
    MINUTES_PER_DAY,
    ParticipantBlock,
    _calendar_rows,
    _day_zero,
    _first_slot_of,
    _padded_days,
    build_ladder,
    dataset_digest,
    missingness_audit,
)

N_READ = 4


@pytest.fixture(scope="module")
def built():
    return build_ladder(participants=N_READ)


@pytest.fixture(scope="module")
def record(built):
    return built[0]


def test_the_ladder_reads_the_participants_it_was_asked_for(built):
    record, cohort, collapse = built
    assert len(record.blocks) == N_READ
    assert cohort.participants_kept == 51
    assert collapse.episodes > 0
    assert record.K == len(LADDER_UNITS) == 4


def test_every_parent_map_is_non_decreasing(record):
    for k in range(1, record.K + 1):
        who = record.sources.ancestors[k - 1]
        assert np.all(np.diff(who) >= 0)
        # Surjective onto the epochs the scale declares, so no epoch is empty
        # and the nesting the ladder verifies is the one the arrays hold.
        assert int(who.max()) + 1 == record.n_epochs(k)


def test_no_epoch_at_any_scale_holds_two_participants(record):
    for k in range(1, record.K + 1):
        who = record.participant_index(k)
        assert np.all(np.diff(who) >= 0)
        assert len(np.unique(who)) == len(record.blocks)
    for k in range(1, record.K):
        parent = record.sources.ancestors[k]
        fine_owner = record.participant_index(k)
        first = _first_slot_of(record.sources.ancestors[k - 1], record.n_epochs(k))
        coarse_owner = record.participant_index(k + 1)[parent[first]]
        assert np.array_equal(fine_owner, coarse_owner)


def test_the_ladder_builds_and_its_nesting_verifies(record):
    ladder = record.ladder()
    assert ladder.K == record.K
    assert ladder.n_epochs == [record.n_epochs(k) for k in range(1, 5)]
    assert ladder.n_slots == record.n_slots


def test_counts_reconcile_up_the_ladder(record):
    for k in range(1, record.K):
        parent = record.sources.ancestors[k]
        first = _first_slot_of(record.sources.ancestors[k - 1], record.n_epochs(k))
        into = parent[first]
        for array in (record.sources.counts, record.sources.durations):
            folded = np.zeros_like(array[k])
            np.add.at(folded, into, array[k - 1])
            assert np.allclose(folded, array[k])


def test_a_day_the_record_does_not_cover_is_absent_at_every_scale(record):
    day = record.observed_at(3)
    minute = record.observed_at(1)
    assert day.sum() < day.size
    assert minute.sum() == day.sum() * MINUTES_PER_DAY
    for block in record.blocks:
        start = block.offsets[2]
        own = day[start : start + block.padded_days]
        assert int(own.sum()) == block.observed_days
        # The padding beyond a participant's own span is absent for the same
        # reason a day with no episode is.
        assert not own[block.span_days :].any()


def test_a_padded_participant_holds_whole_weeks(record):
    for block in record.blocks:
        assert block.padded_days % DAYS_PER_WEEK == 0
        assert block.padded_days >= block.span_days
        assert block.padded_days - block.span_days < DAYS_PER_WEEK
        assert block.n_slots == block.padded_days * MINUTES_PER_DAY
        for k, minutes in enumerate(LADDER_MINUTES, start=1):
            assert block.n_epochs(k) == block.n_slots // minutes


def test_the_scheduled_label_is_a_source_and_never_a_column_of_the_counts(record):
    assert SCHEDULED_LABEL not in record.axis.targets
    column = record.axis.exogenous_index(SCHEDULED_LABEL)
    received = record.exogenous_at(4)[:, column]
    assert received.sum() > 0
    assert record.sources.counts[3].shape[1] == len(record.axis.targets)


def test_the_hour_index_is_block_constant_at_the_day_scale(record):
    day_block = record.exogenous_at(3)
    assert np.all(day_block[:, 0] == 0.0)
    hour_block = record.exogenous_at(2)
    for block in record.blocks:
        start = block.offsets[1]
        own = hour_block[start : start + block.n_epochs(2), 0]
        assert np.array_equal(own[:48], np.tile(np.arange(24, dtype=float), 2))


def test_the_weekday_index_turns_over_every_seven_days(record):
    day_block = record.exogenous_at(3)
    for block in record.blocks:
        start = block.offsets[2]
        own = day_block[start : start + block.padded_days, 1]
        assert own[0] == float(block.day_zero.weekday())
        assert np.array_equal(own[DAYS_PER_WEEK:], own[:-DAYS_PER_WEEK])


def test_the_provenance_names_the_bytes_and_the_rules(record):
    provenance = record.provenance
    assert provenance["csv"] == RAW_CSV
    assert provenance["csv_sha256"] == dataset_digest(RAW_CSV)
    assert len(provenance["csv_sha256"]) == 64
    assert provenance["participants_read"] == N_READ
    assert provenance["participants_kept"] == 51
    assert provenance["axis_digest"] == record.axis.digest()
    assert provenance["units"] == list(LADDER_UNITS)
    assert provenance["n_epochs"] == [record.n_epochs(k) for k in range(1, 5)]
    assert provenance["collapsed_group"] == [
        "notification_read",
        "mood_survey",
        "step_goal_survey",
    ]


def test_the_digest_is_the_hash_of_the_file(tmp_path):
    path = tmp_path / "small.csv"
    path.write_bytes(b"person_id,event\n0,walk\n")
    assert (
        dataset_digest(str(path))
        == hashlib.sha256(b"person_id,event\n0,walk\n").hexdigest()
    )


def test_a_scale_is_named_by_its_unit(record):
    assert [record.scale_of(u) for u in LADDER_UNITS] == [1, 2, 3, 4]
    with pytest.raises(KeyError, match="not one of"):
        record.scale_of("fortnight")


def test_the_gap_audit_reads_the_day_grid_and_reports_a_verdict(record):
    report = missingness_audit(record, n_permutations=49)
    assert len(report.verdicts) == 1
    verdict = report.verdicts[0]
    assert verdict.label == "day coverage"
    assert verdict.declared == "ignorable_given_calendar"
    assert "clump" in str(verdict) or "predict" in str(verdict)
    assert report.accepted in (True, False)


def test_a_span_is_padded_up_to_whole_weeks():
    assert _padded_days(1) == 7
    assert _padded_days(7) == 7
    assert _padded_days(8) == 14
    assert _padded_days(70) == 70


def test_day_zero_is_the_date_of_the_participants_own_first_episode(record):
    for block in record.blocks:
        assert isinstance(block.day_zero, date)
    assert record.blocks[0].day_zero == date(2021, 5, 24)


def test_the_first_slot_of_each_epoch_is_the_earliest_one():
    ancestor = np.array([0, 0, 0, 1, 1, 2], dtype=np.int64)
    assert np.array_equal(_first_slot_of(ancestor, 3), np.array([0, 3, 5]))


def test_the_calendar_rows_of_one_block_cover_its_own_epochs():
    block = ParticipantBlock(
        unit="x",
        wave="spring",
        day_zero=date(2021, 12, 30),
        span_days=8,
        padded_days=14,
        observed_days=8,
        offsets=(0, 0, 0, 0),
    )
    rows = _calendar_rows(block, 3)
    assert rows.shape == (14, 5)
    assert np.array_equal(rows[:, 2], np.arange(14, dtype=float))
    assert np.array_equal(rows[:, 3], np.arange(14, dtype=float) // 7)
    # 2021-12-30 is a Thursday, and the month index rolls over into January.
    assert rows[0, 1] == 3.0
    assert rows[0, 4] == 11.0
    assert rows[2, 4] == 0.0


def test_a_larger_read_carries_every_participant_the_smaller_one_did(record):
    wider, _, _ = build_ladder(participants=N_READ + 1)
    assert [b.unit for b in wider.blocks[:N_READ]] == [b.unit for b in record.blocks]
    assert wider.n_slots > record.n_slots
