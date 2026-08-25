"""
Tests for the declared rules of the observed LifeSnaps log.

Each test pins one of the four rules of `src/data/lifesnaps.py`, so a change to
a rule fails here and has to be made deliberately.
"""

from __future__ import annotations

import collections

import pytest

from src.data.lifesnaps import (
    CALENDAR_NODES,
    GYM_LABEL,
    GYM_MODALITIES,
    INJECTED_SECONDS,
    INSTANT_LABELS,
    calendar_columns,
    duration_is_scored,
    hash_key,
    read_lifesnaps,
)


def test_the_cohort_filter_keeps_only_high_coverage_participants():
    events, report = read_lifesnaps()
    assert report.participants_raw == 60
    assert report.participants_kept == 51
    assert len(report.dropped) == 9
    # Every dropped participant is dropped for coverage, and every one of them
    # sits below the gap in the coverage distribution.
    assert max(coverage for _, coverage, _ in report.dropped) < 0.90
    assert {event.unit for event in events}.isdisjoint(
        {unit for unit, _, _ in report.dropped}
    )


def test_every_exercise_modality_becomes_gym():
    events, report = read_lifesnaps()
    labels = {event.label for event in events}
    assert GYM_LABEL in labels
    assert labels.isdisjoint(GYM_MODALITIES)
    assert report.merged_into_gym == sum(
        1 for event in events if event.label == GYM_LABEL
    )


def test_an_injected_duration_can_never_be_mistaken_for_a_real_one():
    events, _ = read_lifesnaps()
    injected = {event.duration_s for event in events if event.injected}
    real = {event.duration_s for event in events if not event.injected}
    assert injected == set(INJECTED_SECONDS)
    # A real duration is a whole number of minutes, so the two sets are
    # disjoint and the injection stays auditable after the fact.
    assert all(seconds % 60 == 0 for seconds in real)
    assert not (injected & real)


def test_every_episode_carries_a_positive_duration():
    events, _ = read_lifesnaps()
    assert all(event.duration_s >= 1 for event in events)


def test_an_instant_and_a_repaired_zero_are_counted_apart():
    events, _ = read_lifesnaps()
    for event in events:
        if event.duration_source == "instant":
            assert event.label in INSTANT_LABELS
        if event.duration_source == "repaired":
            assert event.label not in INSTANT_LABELS
        assert duration_is_scored(event.label) == (event.label not in INSTANT_LABELS)


def test_the_injected_duration_is_a_function_of_the_episode_alone():
    first, _ = read_lifesnaps()
    second, _ = read_lifesnaps()
    assert [event.duration_s for event in first] == [
        event.duration_s for event in second
    ]
    # Both values are used, so the channel carries a distribution and a
    # statistic over it is defined.
    counts = collections.Counter(event.duration_s for event in first if event.injected)
    assert set(counts) == set(INJECTED_SECONDS)
    assert min(counts.values()) > 0


def test_the_stable_hash_does_not_depend_on_the_interpreter_seed():
    # FNV-1a of the empty string is its offset basis.
    assert hash_key("") == 0xCBF29CE484222325
    assert hash_key("a") != hash_key("b")


def test_the_calendar_columns_span_their_declared_periods():
    events, _ = read_lifesnaps()
    columns = calendar_columns(events)
    assert set(columns) == set(CALENDAR_NODES)
    assert set(columns["weekday_index"]) == set(range(7))
    assert set(columns["hour_index"]) == set(range(24))
    assert min(columns["day_index"]) == 0
    assert all(
        week == day // 7
        for week, day in zip(columns["week_index"], columns["day_index"])
    )


def test_both_recruitment_waves_are_present():
    events, _ = read_lifesnaps()
    assert {event.wave for event in events} == {"spring", "autumn"}


def test_the_report_says_what_every_rule_did():
    _, report = read_lifesnaps()
    text = report.summary()
    assert f"rows {report.n_rows_raw} -> {report.n_rows_kept}" in text
    assert f"merged into {GYM_LABEL}: {report.merged_into_gym} episodes" in text
    assert f"instants given {INJECTED_SECONDS} s:" in text
    # One line per dropped participant and one per surviving label, so the
    # report accounts for every row the rules moved.
    assert sum(1 for line in text.splitlines() if line.startswith("  ")) == len(
        report.dropped
    ) + len(report.labels)


def test_a_log_missing_a_column_the_rules_read_is_refused(tmp_path):
    path = tmp_path / "short.csv"
    path.write_text("person_id,event,day_index,start_min\np1,walk,0,10\n")
    with pytest.raises(ValueError, match="duration_min.*start_datetime"):
        read_lifesnaps(str(path))
