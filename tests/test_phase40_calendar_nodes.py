"""
tests/test_phase40_calendar_nodes.py
====================================
Calendar quantities as nodes, placed once at the scale they are constant on.

Two properties carry the design.  A node's home scale is the coarsest one whose
epochs its value never changes inside, which is checked against the clock rather
than declared.  And a node is placed exactly once, because the same column at
two scales is one column repeated and the likelihood cannot separate it.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from src.core.calendar import SlotCalendar
from src.core.features import (
    CalendarNode,
    calendar_blocks,
    day_of_week,
    home_scale,
    horizon_bins,
    horizon_trend,
    hour_of_day,
    month_of_year,
    node_matrix,
    quarter_of_year,
    weekend,
)

# A Monday, so a weekday index of 0 is the first epoch of the horizon.
MONDAY = datetime(2026, 1, 5)


def _minute_ladder(weeks: int = 12):
    """minute / hour / day / week, the ladder the study runs on."""
    cal = SlotCalendar(origin=MONDAY, slot=timedelta(minutes=1), branching=(60, 24, 7))
    counts = [weeks * 7 * 1440, weeks * 7 * 24, weeks * 7, weeks]
    return cal, counts


# ---------------------------------------------------------------------------
# Home scale
# ---------------------------------------------------------------------------


def test_each_node_lands_on_the_scale_it_is_constant_on() -> None:
    cal, counts = _minute_ladder()
    expected = {
        hour_of_day(): 2,
        day_of_week(): 3,
        weekend(): 3,
        horizon_trend(cal.origin, "week"): 4,
    }
    for node, scale in expected.items():
        assert home_scale(cal, node, counts) == scale, node.name


def test_a_node_is_constant_across_every_epoch_of_its_home_scale() -> None:
    """This is the defining property, so it is asserted and not assumed."""
    cal, counts = _minute_ladder(weeks=3)
    for node in (hour_of_day(), day_of_week(), weekend()):
        k = home_scale(cal, node, counts)
        for t in range(counts[k - 1]):
            first = node.value(cal.start_of(k, t))
            last = node.value(cal.end_of(k, t) - cal.slot)
            assert first == last


def test_a_node_does_vary_inside_the_scale_above_its_home() -> None:
    cal, counts = _minute_ladder(weeks=3)
    for node in (hour_of_day(), day_of_week()):
        k = home_scale(cal, node, counts)
        assert k < len(counts)
        varies = any(
            node.value(cal.start_of(k + 1, t))
            != node.value(cal.end_of(k + 1, t) - cal.slot)
            for t in range(counts[k])
        )
        assert varies, f"{node.name} is constant above its home scale {k}"


def test_the_home_scale_follows_the_ladder_not_the_node() -> None:
    """On a ten-minute ladder with no hour scale, the hour node sits lower."""
    cal = SlotCalendar(origin=MONDAY, slot=timedelta(minutes=10), branching=(6, 24))
    counts = [7 * 144, 7 * 24, 7]
    assert home_scale(cal, hour_of_day(), counts) == 2
    assert home_scale(cal, weekend(), counts) == 3


def test_a_node_that_never_varies_stays_at_the_coarsest_scale() -> None:
    cal = SlotCalendar(origin=MONDAY, slot=timedelta(days=1), branching=(7,))
    counts = [7, 1]
    assert home_scale(cal, month_of_year(), counts) == 2


# ---------------------------------------------------------------------------
# Encodings, which are declared rather than fitted
# ---------------------------------------------------------------------------


def test_hour_of_day_carries_the_declared_windows() -> None:
    node = hour_of_day()
    assert node.width == 6
    assert node.columns[0] == "hour_of_day[7,10)"
    assert "hour_of_day[12,14)" in node.columns
    assert "hour_of_day[17,21)" in node.columns
    whens = [datetime(2026, 1, 5, h) for h in range(24)]
    design = node.encode(whens)
    assert design.shape == (24, 6)
    # Every hour falls in exactly one bin, and the reference bin in none.
    assert set(design.sum(axis=1)) == {0.0, 1.0}
    assert design[8, node.columns.index("hour_of_day[7,10)")] == 1.0
    assert design[3].sum() == 0.0, "hours before 07 are the reference"


def test_a_one_hot_node_drops_its_first_level() -> None:
    node = day_of_week()
    assert node.width == 6 and len(node.levels) == 7
    design = node.encode([MONDAY + timedelta(days=d) for d in range(7)])
    assert design[0].sum() == 0.0, "Monday is the reference"
    assert design[1:].sum() == 6.0


def test_weekend_marks_saturday_and_sunday() -> None:
    design = weekend().encode([MONDAY + timedelta(days=d) for d in range(7)])
    assert design[:, 0].tolist() == [0, 0, 0, 0, 0, 1, 1]


def test_quarter_and_month_nodes_read_the_calendar() -> None:
    whens = [datetime(2026, m, 15) for m in range(1, 13)]
    q = quarter_of_year().encode(whens)
    assert q.shape == (12, 3)
    assert q[0].sum() == 0.0, "the first quarter is the reference"
    assert q[3, 0] == 1.0 and q[6, 1] == 1.0 and q[9, 2] == 1.0
    assert month_of_year().encode(whens).shape == (12, 11)


def test_the_horizon_index_has_both_readings() -> None:
    trend = horizon_trend(MONDAY, "week")
    bins = horizon_bins(MONDAY, 12, 3, "week")
    whens = [MONDAY + timedelta(weeks=w) for w in range(12)]
    assert trend.encode(whens).ravel().tolist() == list(range(12))
    binned = bins.encode(whens)
    assert binned.shape == (12, 2)
    assert binned[0].sum() == 0.0, "the first third is the reference"
    assert binned[5, 0] == 1.0 and binned[11, 1] == 1.0


def test_a_horizon_index_can_count_days() -> None:
    node = horizon_trend(MONDAY, "day")
    assert node.name == "day_index"
    assert node.encode(
        [MONDAY + timedelta(days=d) for d in range(4)]
    ).ravel().tolist() == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# What a node refuses to be
# ---------------------------------------------------------------------------


def test_an_unknown_encoding_is_refused() -> None:
    with pytest.raises(ValueError, match="expected one of"):
        CalendarNode(name="x", value=lambda w: 0, encoding="spline")


def test_a_one_hot_node_needs_two_levels() -> None:
    with pytest.raises(ValueError, match="at least two levels"):
        CalendarNode(name="x", value=lambda w: 0, encoding="onehot", levels=(1,))


def test_a_binned_node_needs_two_bins() -> None:
    with pytest.raises(ValueError, match="at least two bins"):
        CalendarNode(name="x", value=lambda w: 0, encoding="bins", edges=(0, 5))


def test_bin_edges_must_increase() -> None:
    with pytest.raises(ValueError, match="not strictly increasing"):
        CalendarNode(name="x", value=lambda w: 0, encoding="bins", edges=(0, 9, 5, 24))


def test_a_horizon_unit_must_be_a_day_or_a_week() -> None:
    with pytest.raises(ValueError, match="'day' or 'week'"):
        horizon_trend(MONDAY, "fortnight")


def test_a_binned_horizon_needs_two_bins() -> None:
    with pytest.raises(ValueError, match="at least two bins"):
        horizon_bins(MONDAY, 12, 1, "week")


# ---------------------------------------------------------------------------
# Placing the whole set on a ladder
# ---------------------------------------------------------------------------


def test_every_node_is_placed_exactly_once() -> None:
    cal, counts = _minute_ladder()
    nodes = [
        hour_of_day(),
        day_of_week(),
        weekend(),
        horizon_trend(cal.origin, "week"),
        horizon_bins(cal.origin, 12, 3, "week"),
    ]
    cb = calendar_blocks(cal, nodes, counts)
    placed = [name for k in sorted(cb.names) for name in cb.names[k]]
    assert len(placed) == len(set(placed))
    assert len(placed) == sum(node.width for node in nodes)


def test_the_blocks_line_up_with_the_epoch_counts() -> None:
    cal, counts = _minute_ladder()
    cb = calendar_blocks(cal, [hour_of_day(), weekend()], counts)
    assert [b.shape[0] for b in cb.blocks] == counts
    assert cb.widths == [0, 6, 1, 0]
    assert cb.names[1] == [] and cb.names[4] == []


def test_a_scale_with_no_node_gets_a_zero_width_block() -> None:
    cal, counts = _minute_ladder()
    cb = calendar_blocks(cal, [weekend()], counts)
    assert cb.blocks[0].shape == (counts[0], 0)
    assert cb.blocks[2].shape == (counts[2], 1)


def test_node_matrix_can_be_read_at_a_finer_scale_than_the_home() -> None:
    """A node reaches a finer target by broadcast, which is why it is placed once."""
    cal, counts = _minute_ladder(weeks=1)
    fine = node_matrix(cal, weekend(), 2, counts[1])
    coarse = node_matrix(cal, weekend(), 3, counts[2])
    assert fine.shape == (counts[1], 1)
    assert np.array_equal(fine.ravel(), np.repeat(coarse.ravel(), 24))


def test_the_placement_holds_on_a_calendar_unit_ladder() -> None:
    """day / month / quarter / year, where the coarse scales are irregular."""
    cal = SlotCalendar(
        origin=datetime(2026, 1, 1),
        slot=timedelta(days=1),
        units=("month", "quarter", "year"),
    )
    counts, _ = cal.hierarchy_maps(365)
    cb = calendar_blocks(cal, [weekend(), month_of_year(), quarter_of_year()], counts)
    assert cb.homes["weekend"] == 1
    assert cb.homes["month_of_year"] == 2
    assert cb.homes["quarter_of_year"] == 3
    assert cb.widths == [1, 11, 3, 0]
