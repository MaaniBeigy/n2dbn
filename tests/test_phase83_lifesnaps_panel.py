"""
Tests for the balanced panel a pooled fit streams.

A pooled fit lays every subject on one ladder, so what is pinned here is that
the window is the same length for all of them, that it opens on a Monday so the
weekday index can be shared, that the exogenous quantity the study delivers
rides with the subject, and that every coarsest epoch is whole.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from src.core.mask import check_ladder
from src.data.label_axis import SCHEDULED_LABEL, collapse_labels, label_axis
from src.data.lifesnaps import read_lifesnaps
from src.data.lifesnaps_panel import (
    PANEL_BRANCHING,
    PANEL_EXOGENOUS,
    PANEL_NODES,
    PANEL_WEEKS,
    _first_monday_offset,
    balanced_panel,
    offset_days,
    panel_axis,
    panel_hierarchy,
    panel_provenance,
    shared_calendar,
    wave_of,
)

DAYS = PANEL_WEEKS * 7
SLOTS = DAYS * 1440


@pytest.fixture(scope="module")
def built():
    return balanced_panel()


@pytest.fixture(scope="module")
def cohort(built):
    return built[0]


def test_the_window_is_eight_complete_weeks_on_one_ladder(cohort):
    hierarchy = panel_hierarchy()
    assert PANEL_BRANCHING == (60, 24, 7)
    assert [hierarchy.n_epochs(k) for k in range(1, 5)] == [SLOTS, DAYS * 24, DAYS, 8]
    assert cohort.hierarchy.H == hierarchy.H
    assert cohort.masks.shape == (cohort.S, SLOTS)


def test_every_kept_participant_clears_the_window(built):
    cohort, _, report, cohort_report, _ = built
    assert cohort_report.participants_kept == 51
    assert cohort.S == 51
    assert report.dropped == ()
    assert len(report.subjects) == 51


def test_the_window_opens_on_a_monday_for_every_subject(built):
    _, _, report, _, _ = built
    shared = shared_calendar()
    day_block = shared[2]
    weekday = PANEL_NODES.index("weekday_index")
    assert day_block[0, weekday] == 0.0
    assert np.array_equal(day_block[7:, weekday], day_block[:-7, weekday])
    events, _ = read_lifesnaps()
    own = {}
    for event in events:
        own.setdefault(event.unit, []).append(event)
    for subject in report.subjects:
        offset = subject.offset_days
        assert 0 <= offset < 7
        assert offset == _first_monday_offset(own[subject.unit])
        first = own[subject.unit][0]
        opened = first.start.date().toordinal() - first.day_index + offset
        # A Monday is ordinal 1 modulo 7 in the proleptic Gregorian calendar.
        assert opened % 7 == 1


def test_the_shared_clock_nodes_are_the_same_function_of_the_epoch(built):
    shared = shared_calendar()
    assert PANEL_NODES == (
        "hour_index",
        "hour_cos",
        "hour_sin",
        "weekday_index",
        "weekday_cos",
        "weekday_sin",
        "day_index",
        "week_index",
    )
    width = len(PANEL_NODES)
    assert [block.shape for block in shared] == [
        (SLOTS, width),
        (DAYS * 24, width),
        (DAYS, width),
        (8, width),
    ]
    # An hour index is constant inside a day, so it carries nothing at the day
    # scale and above, which is the invariance argument applied to the clock.
    # Its two harmonics sit at the same hour and are constant with it.
    assert np.all(shared[2][:, 0] == 0.0)
    assert np.all(shared[3][:, 0] == 0.0)
    assert np.all(shared[2][:, 1] == 1.0)
    assert np.all(shared[2][:, 2] == 0.0)
    assert np.array_equal(shared[2][:, 6], np.arange(DAYS, dtype=np.float32))
    assert np.array_equal(shared[3][:, 7], np.arange(8, dtype=np.float32))
    assert np.array_equal(
        shared[1][:48, 0], np.tile(np.arange(24, dtype=np.float32), 2)
    )
    # The harmonics close on the clock they are built from, so hour 0 and hour
    # 23 sit next to each other where the raw index puts them furthest apart.
    hours = shared[1][:24]
    assert np.allclose(hours[:, 1] ** 2 + hours[:, 2] ** 2, 1.0, atol=1e-6)
    assert abs(hours[0, 1] - hours[23, 1]) < abs(hours[0, 1] - hours[12, 1])
    days = shared[2][:7]
    assert np.allclose(days[:, 4] ** 2 + days[:, 5] ** 2, 1.0, atol=1e-6)


def test_the_axis_of_the_panel_carries_no_month_index(built):
    _, axis, _, _, _ = built
    assert axis.exogenous == PANEL_EXOGENOUS
    assert "month_index" not in axis.exogenous
    assert axis.exogenous[-1] == SCHEDULED_LABEL
    events, _ = read_lifesnaps()
    full = label_axis(collapse_labels(events)[0])
    assert panel_axis(full).targets == full.targets


def test_the_scheduled_label_rides_with_the_subject(cohort):
    assert cohort.subject_calendar
    blocks = []
    for name in cohort.subjects[:4]:
        sources = cohort.sources(cohort.open(name))
        clocks = len(PANEL_NODES)
        assert sources.calendar[2].shape == (DAYS, len(PANEL_EXOGENOUS))
        blocks.append(sources.calendar[2][:, clocks].copy())
        # The clock columns are the shared ones, read the same way by all.
        assert np.array_equal(sources.calendar[2][:, :clocks], shared_calendar()[2])
    assert any(not np.array_equal(blocks[0], other) for other in blocks[1:])
    assert sum(float(block.sum()) for block in blocks) > 0.0


def test_a_cohort_that_shares_its_calendar_reads_the_shared_block(cohort):
    from dataclasses import replace

    shared_only = replace(cohort, subject_calendar=False)
    sources = shared_only.sources(shared_only.open(shared_only.subjects[0]))
    assert sources.calendar[2].shape == (DAYS, len(PANEL_NODES))


def test_every_coarsest_epoch_is_whole_and_the_rounding_is_reported(built):
    cohort, _, report, _, _ = built
    for row in cohort.masks:
        check_ladder(row, cohort.hierarchy.parent_maps)
    per_week = cohort.masks.reshape(cohort.S, 8, -1)
    assert set(np.unique(per_week.sum(axis=2))) <= {0, per_week.shape[2]}
    assert sum(report.rounded_slots) > 0
    assert len(report.rounded_slots) == cohort.S


def test_the_counts_reconcile_up_the_shared_ladder(cohort):
    data = cohort.open(cohort.subjects[0])
    for k in range(cohort.K - 1):
        parent = cohort.hierarchy.parent_maps[k]
        for array in (data.X, data.Y):
            folded = np.zeros_like(array[k + 1])
            np.add.at(folded, parent, array[k])
            assert np.array_equal(folded, array[k + 1])


def test_a_subject_carries_only_what_falls_inside_its_own_window(built):
    cohort, axis, report, _, _ = built
    offsets = offset_days(report)
    events, _ = read_lifesnaps()
    events, _ = collapse_labels(events)
    name = cohort.subjects[0]
    offset = offsets[name]
    inside = [
        e
        for e in events
        if e.unit == name
        and 0 <= e.day_index - offset < DAYS
        and e.label != SCHEDULED_LABEL
    ]
    data = cohort.open(name)
    assert int(data.X[3].sum()) == len(inside)


def test_a_window_longer_than_a_record_drops_the_subject_and_says_so():
    cohort, _, report, _, _ = balanced_panel(participants=12, weeks=10)
    assert len(report.dropped) == 8
    assert cohort.S == 12 - len(report.dropped)
    for unit, span, need in report.dropped:
        assert need > span
    text = report.summary()
    assert "10 weeks" in text
    assert "longer than their record" in text
    assert "coverage min" in text


def test_a_window_no_record_clears_leaves_no_panel():
    with pytest.raises(ValueError, match="holds no subject"):
        balanced_panel(participants=12, weeks=30)


def test_the_summary_reports_the_rounding_it_did(built):
    _, _, report, _, _ = built
    text = report.summary()
    assert "51 subjects" in text
    assert "rounding a partly covered week down" in text


def test_a_subject_report_carries_its_own_coverage(built):
    _, _, report, _, _ = built
    assert all(s.window_days == DAYS for s in report.subjects)
    for subject in report.subjects:
        assert 0.0 < subject.coverage <= 1.0
        assert subject.wave in ("spring", "autumn")
    assert set(wave_of(report).values()) <= {"spring", "autumn"}
    assert min(s.coverage for s in report.subjects) >= 0.85


def test_the_provenance_names_the_window_and_the_bytes(built):
    cohort, axis, report, _, _ = built
    record = panel_provenance(cohort, axis, report)
    assert record["window_weeks"] == PANEL_WEEKS
    assert record["window_days"] == DAYS
    assert record["n_subjects"] == 51
    assert record["dropped"] == []
    assert record["exogenous"] == list(PANEL_EXOGENOUS)
    assert record["n_epochs"] == [SLOTS, DAYS * 24, DAYS, 8]
    assert len(record["csv_sha256"]) == 64
    assert record["coverage_mean"] > 0.9


def test_a_smaller_read_carries_a_prefix_of_the_roster(cohort):
    small, _, _, _, _ = balanced_panel(participants=4)
    assert small.S <= 4
    assert set(small.subjects).issubset(set(cohort.subjects))
    assert small.hierarchy.H == cohort.hierarchy.H


def test_the_schedule_names_the_window_the_roster_and_the_axis(built):
    cohort, axis, report, _, _ = built
    schedule = report.schedule
    assert schedule.weeks == PANEL_WEEKS
    assert sorted(schedule.offsets) == list(cohort.subjects)
    assert schedule.axis is not None and schedule.axis.digest() == axis.digest()
    for row, unit in enumerate(cohort.subjects):
        watched = cohort.masks[row].reshape(-1, 1440).any(axis=1)
        assert len(schedule.days[unit]) == int(watched.sum())
        assert min(schedule.days[unit]) >= schedule.offsets[unit]


def test_a_declared_schedule_reproduces_the_panel_it_was_taken_from(built):
    cohort, axis, report, _, _ = built
    again, again_axis, again_report, _, _ = balanced_panel(schedule=report.schedule)
    assert again.subjects == cohort.subjects
    assert np.array_equal(again.masks, cohort.masks)
    assert again_axis.digest() == axis.digest()
    assert again_report.dropped == ()


def test_a_schedule_taken_over_another_window_is_refused(built):
    _, _, report, _, _ = built
    with pytest.raises(ValueError, match="other epochs"):
        balanced_panel(weeks=4, schedule=report.schedule)


def test_a_schedule_naming_a_subject_the_log_lacks_is_refused(built):
    _, _, report, _, _ = built
    schedule = dataclasses.replace(
        report.schedule,
        offsets={**dict(report.schedule.offsets), "nobody": 0},
    )
    with pytest.raises(ValueError, match="holds no row"):
        balanced_panel(schedule=schedule)


def test_a_duration_is_stored_in_finest_slots_and_never_in_seconds(built):
    cohort, axis, _, _, _ = built
    events, _ = collapse_labels(read_lifesnaps()[0])
    longest = max(event.duration_s for event in events if event.label == "sleep")
    data = cohort.open(cohort.subjects[0])
    column = axis.target_index("sleep")
    assert int(np.max(data.Y[0][:, column])) <= longest // 60 + 1
    # An episode occupies at least one slot, so an instant of one second is one.
    survey = axis.target_index("survey_response")
    instants = data.Y[0][data.X[0][:, survey] == 1, survey]
    assert instants.size and instants.min() >= 1


def test_each_periodic_clock_node_carries_a_harmonic_pair():
    shared = shared_calendar()
    hour = shared[1]
    cos = PANEL_NODES.index("hour_cos")
    sin = PANEL_NODES.index("hour_sin")
    index = PANEL_NODES.index("hour_index")
    angle = 2.0 * np.pi * hour[:, index] / 24.0
    assert np.allclose(hour[:, cos], np.cos(angle), atol=1e-6)
    assert np.allclose(hour[:, sin], np.sin(angle), atol=1e-6)
    # An index is monotone across the day and a harmonic pair is not, which is
    # what lets a profile carry a peak.
    assert np.all(np.diff(hour[:24, index]) > 0)
    assert np.any(np.diff(hour[:24, cos]) > 0) and np.any(np.diff(hour[:24, cos]) < 0)


def test_a_harmonic_is_constant_inside_the_block_its_node_is_identified_in():
    shared = shared_calendar()
    day = shared[2]
    for name in ("hour_cos", "hour_sin"):
        column = day[:, PANEL_NODES.index(name)]
        assert np.allclose(column, column[0])
    # A weekday harmonic varies inside a week, so it is identified at the day.
    weekday = day[:, PANEL_NODES.index("weekday_cos")]
    assert not np.allclose(weekday, weekday[0])
