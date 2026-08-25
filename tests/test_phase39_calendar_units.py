"""
tests/test_phase39_calendar_units.py
====================================
The scale ladder names real units, from a second to a week.

One ladder per named unit as the finest scale.  Each asserts that an epoch's
span equals the difference of consecutive epoch starts at every scale, that the
finest slot round-trips through the log reader, and that every emitted episode
carries a well-formed interval at the resolution the slot implies.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from src.core.calendar import (
    DEFAULT_ORIGIN,
    EventRecord,
    SlotCalendar,
    format_duration,
    require_naive,
)
from src.core.scales import ScaleHierarchy
from src.data.event_log import read_event_log

ORIGIN = datetime(2026, 3, 1, 0, 0)

# (name, slot, branching) with the finest scale at the named unit.  Each ladder
# climbs to a week so the coarse end is exercised as well as the fine one.
LADDERS = [
    ("second", timedelta(seconds=1), (60, 60, 24, 7)),
    ("15 seconds", timedelta(seconds=15), (4, 60, 24, 7)),
    ("minute", timedelta(minutes=1), (60, 24, 7)),
    ("10 minutes", timedelta(minutes=10), (6, 24, 7)),
    ("hour", timedelta(hours=1), (24, 7)),
    ("day", timedelta(days=1), (7,)),
    ("week", timedelta(weeks=1), ()),
]

IDS = [name for name, _, _ in LADDERS]


@pytest.fixture(params=LADDERS, ids=IDS)
def ladder(request) -> SlotCalendar:
    _, slot, branching = request.param
    return SlotCalendar(origin=ORIGIN, slot=slot, branching=branching)


def test_a_span_equals_the_difference_of_consecutive_starts(
    ladder: SlotCalendar,
) -> None:
    for k in range(1, len(ladder.branching) + 2):
        for t in (0, 1, 5):
            gap = ladder.start_of(k, t + 1) - ladder.start_of(k, t)
            assert gap == ladder.epoch_span(k)


def test_an_epoch_ends_where_the_next_one_starts(ladder: SlotCalendar) -> None:
    for k in range(1, len(ladder.branching) + 2):
        assert ladder.end_of(k, 3) == ladder.start_of(k, 4)


def test_a_parent_epoch_spans_exactly_its_children(ladder: SlotCalendar) -> None:
    for k in range(1, len(ladder.branching) + 1):
        assert (
            ladder.epoch_span(k + 1) == ladder.epoch_span(k) * ladder.branching[k - 1]
        )


def test_the_coarsest_scale_of_every_ladder_is_a_week(ladder: SlotCalendar) -> None:
    coarsest = len(ladder.branching) + 1
    assert ladder.epoch_span(coarsest) == timedelta(weeks=1)
    assert format_duration(ladder.epoch_span(coarsest)) == "1 week"


def test_describe_names_every_scale(ladder: SlotCalendar) -> None:
    described = ladder.describe()
    assert len(described) == len(ladder.branching) + 1
    assert all(isinstance(v, str) and v for v in described.values())


def test_shifting_moves_the_origin_by_whole_slots(ladder: SlotCalendar) -> None:
    moved = ladder.shifted(7)
    assert moved.origin - ladder.origin == ladder.slot * 7
    assert moved.slot == ladder.slot and moved.branching == ladder.branching


def test_an_episode_carries_a_well_formed_interval(ladder: SlotCalendar) -> None:
    """Both endpoints are datetimes at the slot's own resolution, end after start."""
    record = EventRecord(
        label="L0",
        start=ladder.start_of(1, 4),
        end=ladder.end_of(1, 6),
        duration=ladder.slot * 3,
        slot_start=4,
        slot_end=6,
    )
    row = record.as_row(ladder.origin, ladder.timespec())
    assert isinstance(record.start, datetime) and isinstance(record.end, datetime)
    assert record.end > record.start
    assert record.duration == ladder.slot * 3
    # The emitted stamp keeps the whole interval, so it parses back exactly.
    assert datetime.fromisoformat(str(row["start_iso"])) == record.start
    assert datetime.fromisoformat(str(row["end_iso"])) == record.end


# ---------------------------------------------------------------------------
# Sub-minute slots, which the minute-only calendar could not express
# ---------------------------------------------------------------------------


def test_a_sub_minute_slot_is_expressible() -> None:
    cal = SlotCalendar(origin=ORIGIN, slot=timedelta(seconds=1), branching=(60,))
    assert cal.start_of(1, 90) == ORIGIN + timedelta(seconds=90)
    assert cal.epoch_span(2) == timedelta(minutes=1)


def test_slot_minutes_raises_rather_than_rounding() -> None:
    cal = SlotCalendar(origin=ORIGIN, slot=timedelta(seconds=30), branching=())
    with pytest.raises(ValueError, match="whole number of minutes"):
        cal.slot_minutes


def test_epoch_minutes_raises_on_a_sub_minute_scale() -> None:
    cal = SlotCalendar(origin=ORIGIN, slot=timedelta(seconds=1), branching=(60,))
    with pytest.raises(ValueError, match="whole number of minutes"):
        cal.epoch_minutes(1)
    assert cal.epoch_minutes(2) == 1


def test_slot_minutes_stays_available_on_a_whole_minute_slot() -> None:
    cal = SlotCalendar.from_config(branching=[10, 6], slot_minutes=5)
    assert cal.slot_minutes == 5
    assert cal.epoch_minutes(3) == 300


def test_from_config_refuses_a_fractional_minute() -> None:
    with pytest.raises(ValueError, match="whole number of minutes"):
        SlotCalendar.from_config(branching=[2], slot_minutes=0.5)


def test_from_config_refuses_both_slot_and_slot_minutes() -> None:
    with pytest.raises(ValueError, match="not both"):
        SlotCalendar.from_config(
            branching=[2], slot_minutes=1, slot=timedelta(seconds=30)
        )


def test_from_config_takes_a_timedelta_slot() -> None:
    cal = SlotCalendar.from_config(branching=[60], slot=timedelta(seconds=1))
    assert cal.slot == timedelta(seconds=1)
    assert cal.epoch_span(2) == timedelta(minutes=1)


def test_the_emitted_precision_follows_the_slot() -> None:
    assert SlotCalendar(slot=timedelta(minutes=1)).timespec() == "minutes"
    assert SlotCalendar(slot=timedelta(seconds=30)).timespec() == "seconds"
    assert SlotCalendar(slot=timedelta(microseconds=500)).timespec() == "microseconds"


def test_a_sub_minute_duration_is_not_truncated_to_zero() -> None:
    cal = SlotCalendar(origin=ORIGIN, slot=timedelta(seconds=30), branching=(2,))
    record = EventRecord(
        label="L0",
        start=cal.start_of(1, 0),
        end=cal.end_of(1, 0),
        duration=cal.slot,
        slot_start=0,
        slot_end=0,
    )
    assert record.duration_minutes == 0.5
    row = record.as_row(cal.origin, cal.timespec())
    assert row["end_iso"] == "2026-03-01T00:00:30"


# ---------------------------------------------------------------------------
# The clock is uniform and naive
# ---------------------------------------------------------------------------


def test_a_timezone_aware_origin_is_refused() -> None:
    from datetime import timezone

    aware = datetime(2026, 3, 1, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="uniform clock"):
        SlotCalendar(origin=aware, slot=timedelta(minutes=1), branching=())


def test_require_naive_returns_a_naive_datetime_unchanged() -> None:
    assert require_naive(ORIGIN, "origin") is ORIGIN


def test_the_reader_refuses_a_timezone_aware_timestamp(tmp_path) -> None:
    path = tmp_path / "events.csv"
    path.write_text(
        "label,start_iso,end_iso\n"
        "walk,2026-03-01T09:00:00+01:00,2026-03-01T09:30:00+01:00\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="uniform clock"):
        read_event_log(str(path))


def test_the_reader_takes_a_naive_timestamp(tmp_path) -> None:
    path = tmp_path / "events.csv"
    path.write_text(
        "label,start_iso,end_iso\n" "walk,2026-03-01T09:00:00,2026-03-01T09:30:00\n",
        encoding="utf-8",
    )
    events = read_event_log(str(path))
    assert len(events) == 1 and events[0].label == "walk"


def test_a_non_positive_slot_is_refused() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        SlotCalendar(origin=ORIGIN, slot=timedelta(0), branching=())


# ---------------------------------------------------------------------------
# Month, quarter and year, whose epochs differ in length
# ---------------------------------------------------------------------------


def _cal(origin, slot, branching, units) -> SlotCalendar:
    return SlotCalendar(origin=origin, slot=slot, branching=branching, units=units)


CALENDAR_LADDERS = [
    (
        "day to year",
        datetime(2026, 1, 1),
        timedelta(days=1),
        (),
        ("month", "year"),
        365,
    ),
    (
        "day to quarter",
        datetime(2026, 1, 1),
        timedelta(days=1),
        (),
        ("month", "quarter"),
        365,
    ),
    ("leap year", datetime(2028, 1, 1), timedelta(days=1), (), ("month", "year"), 366),
    (
        "partial first month",
        datetime(2026, 3, 15),
        timedelta(days=1),
        (),
        ("month", "quarter"),
        200,
    ),
    (
        "hour to month",
        datetime(2026, 1, 1),
        timedelta(hours=1),
        (24,),
        ("month",),
        24 * 90,
    ),
    (
        "minute to quarter",
        datetime(2026, 1, 1),
        timedelta(minutes=1),
        (60, 24),
        ("month", "quarter"),
        60 * 24 * 120,
    ),
]


@pytest.fixture(params=CALENDAR_LADDERS, ids=[row[0] for row in CALENDAR_LADDERS])
def calendar_ladder(request):
    _, origin, slot, branching, units, n_finest = request.param
    return _cal(origin, slot, branching, units), n_finest


def test_a_calendar_ladder_builds_a_valid_hierarchy(calendar_ladder) -> None:
    cal, n_finest = calendar_ladder
    counts, maps = cal.hierarchy_maps(n_finest)
    hier = ScaleHierarchy.from_explicit_maps(
        counts, [np.asarray(m, dtype=np.int32) for m in maps] + [None]
    )
    hier.validate()
    assert len(counts) == cal.K and counts[0] == n_finest


def test_every_child_epoch_lies_inside_its_parent(calendar_ladder) -> None:
    """
    The nesting partitions wall-clock time, not only indices.

    Containment is asserted on the observed part of an epoch, because a calendar
    epoch can open before the origin and only the recorded part of it nests.
    """
    cal, n_finest = calendar_ladder
    counts, maps = cal.hierarchy_maps(n_finest)
    for k, pm in enumerate(maps, start=1):
        for t in (0, 1, len(pm) // 2, len(pm) - 1):
            parent = pm[t]
            child_start = max(cal.start_of(k, t), cal.origin)
            parent_start = max(cal.start_of(k + 1, parent), cal.origin)
            assert child_start >= parent_start
            assert cal.end_of(k, t) <= cal.end_of(k + 1, parent)


def test_a_calendar_scale_has_no_single_span(calendar_ladder) -> None:
    cal, _ = calendar_ladder
    for k in range(cal.n_fixed_scales + 1, cal.K + 1):
        with pytest.raises(ValueError, match="differ in length"):
            cal.epoch_span(k)


def test_month_lengths_follow_the_calendar() -> None:
    cal = _cal(datetime(2026, 1, 1), timedelta(days=1), (), ("month",))
    lengths = [cal.span_of(2, t).days for t in range(12)]
    assert lengths == [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def test_february_is_longer_in_a_leap_year() -> None:
    cal = _cal(datetime(2028, 1, 1), timedelta(days=1), (), ("month",))
    assert cal.span_of(2, 1).days == 29
    assert cal.start_of(2, 2) == datetime(2028, 3, 1)


def test_quarters_start_in_january_april_july_and_october() -> None:
    cal = _cal(datetime(2026, 1, 1), timedelta(days=1), (), ("quarter",))
    starts = [cal.start_of(2, t) for t in range(5)]
    assert starts == [
        datetime(2026, 1, 1),
        datetime(2026, 4, 1),
        datetime(2026, 7, 1),
        datetime(2026, 10, 1),
        datetime(2027, 1, 1),
    ]
    assert [cal.span_of(2, t).days for t in range(4)] == [90, 91, 92, 92]


def test_a_leap_year_lengthens_the_first_quarter() -> None:
    cal = _cal(datetime(2028, 1, 1), timedelta(days=1), (), ("quarter",))
    assert [cal.span_of(2, t).days for t in range(4)] == [91, 91, 92, 92]


def test_four_quarters_partition_a_year() -> None:
    cal = _cal(datetime(2026, 1, 1), timedelta(days=1), (), ("quarter", "year"))
    _, maps = cal.hierarchy_maps(730)
    assert maps[1][:8] == [0, 0, 0, 0, 1, 1, 1, 1]
    assert cal.start_of(3, 1) == cal.end_of(2, 3)


def test_a_quarter_may_open_before_the_origin() -> None:
    """A log starting mid-quarter observes only part of the quarter holding it."""
    cal = _cal(datetime(2026, 2, 1), timedelta(days=1), (), ("month", "quarter"))
    assert cal.start_of(3, 0) == datetime(2026, 1, 1)
    _, maps = cal.hierarchy_maps(300)
    assert min(maps[1]) == 0, "a parent index may not go negative"


def test_a_year_scale_counts_from_the_origin_year() -> None:
    cal = _cal(datetime(2026, 1, 1), timedelta(days=1), (), ("month", "year"))
    assert cal.start_of(3, 0) == datetime(2026, 1, 1)
    assert cal.start_of(3, 1) == datetime(2027, 1, 1)
    assert cal.span_of(3, 0).days == 365


def test_epoch_index_of_finds_the_containing_epoch() -> None:
    cal = _cal(datetime(2026, 1, 1), timedelta(days=1), (), ("month",))
    assert cal.epoch_index_of(1, datetime(2026, 2, 3)) == 33
    assert cal.epoch_index_of(2, datetime(2026, 2, 3)) == 1
    assert cal.epoch_index_of(2, datetime(2026, 12, 31)) == 11


def test_describe_names_a_calendar_unit() -> None:
    cal = _cal(datetime(2026, 1, 1), timedelta(hours=1), (24,), ("month", "quarter"))
    assert cal.describe() == {
        1: "1 hour",
        2: "1 day",
        3: "1 month",
        4: "1 quarter",
    }


def test_shifting_a_calendar_ladder_by_whole_days_re_anchors_it() -> None:
    cal = _cal(datetime(2026, 1, 1), timedelta(days=1), (), ("month",))
    moved = cal.shifted(31)
    assert moved.origin == datetime(2026, 2, 1)
    assert moved.start_of(2, 0) == datetime(2026, 2, 1)
    assert moved.units == cal.units


def test_shifting_a_calendar_ladder_off_midnight_is_refused() -> None:
    cal = _cal(datetime(2026, 1, 1), timedelta(hours=1), (24,), ("month",))
    with pytest.raises(ValueError, match="not at midnight"):
        cal.shifted(5)


def test_an_unknown_calendar_unit_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown calendar unit"):
        _cal(datetime(2026, 1, 1), timedelta(days=1), (), ("fortnight",))


@pytest.mark.parametrize(
    "units", [("quarter", "month"), ("month", "month"), ("year", "quarter")]
)
def test_calendar_units_must_go_from_finer_to_coarser(units) -> None:
    with pytest.raises(ValueError, match="finer to coarser"):
        _cal(datetime(2026, 1, 1), timedelta(days=1), (), units)


def test_a_scale_that_does_not_divide_a_day_cannot_nest_in_a_month() -> None:
    with pytest.raises(ValueError, match="does not divide one day"):
        _cal(datetime(2026, 1, 1), timedelta(weeks=1), (), ("month",))


def test_a_calendar_ladder_needs_an_origin_at_midnight() -> None:
    with pytest.raises(ValueError, match="not at midnight"):
        _cal(datetime(2026, 1, 1, 6, 30), timedelta(hours=1), (24,), ("month",))


def test_from_config_takes_calendar_units() -> None:
    cal = SlotCalendar.from_config(
        branching=[24],
        slot=timedelta(hours=1),
        origin=datetime(2026, 1, 1),
        units=["month", "quarter"],
    )
    assert cal.units == ("month", "quarter") and cal.K == 4


def test_a_fixed_ladder_reports_no_calendar_unit() -> None:
    cal = SlotCalendar.from_config(branching=[10, 6], slot_minutes=1)
    assert cal.unit_of(1) is None and cal.unit_of(3) is None
    assert cal.K == 3 and cal.n_fixed_scales == 3


def test_parent_map_refuses_the_coarsest_scale() -> None:
    cal = _cal(datetime(2026, 1, 1), timedelta(days=1), (), ("month",))
    with pytest.raises(ValueError, match="has no parent"):
        cal.parent_map(2, 10)


def test_hierarchy_maps_of_a_single_scale_ladder() -> None:
    cal = SlotCalendar.from_config(branching=[], slot_minutes=1)
    counts, maps = cal.hierarchy_maps(7)
    assert counts == [7] and maps == []


# ---------------------------------------------------------------------------
# format_duration names every unit the ladder can reach
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "span,expected",
    [
        (timedelta(seconds=1), "1 second"),
        (timedelta(seconds=30), "30 seconds"),
        (timedelta(minutes=1), "1 minute"),
        (timedelta(minutes=10), "10 minutes"),
        (timedelta(hours=1), "1 hour"),
        (timedelta(hours=3), "3 hours"),
        (timedelta(days=1), "1 day"),
        (timedelta(days=2), "2 days"),
        (timedelta(weeks=1), "1 week"),
        (timedelta(weeks=3), "3 weeks"),
        (timedelta(microseconds=1500), "0.0015 seconds"),
    ],
)
def test_format_duration_names_the_plainest_unit(span, expected) -> None:
    assert format_duration(span) == expected


def test_format_duration_still_takes_a_whole_number_of_minutes() -> None:
    assert format_duration(1) == "1 minute"
    assert format_duration(1440) == "1 day"


def test_the_default_origin_is_naive() -> None:
    assert DEFAULT_ORIGIN.tzinfo is None
