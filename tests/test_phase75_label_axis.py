"""
Tests for the collapse of the three-column artifact and the axis it leaves.

The load-bearing one is that no within-group edge can be expressed after the
collapse, and that nothing can point into the scheduled label.  The rest pin the
measurement the collapse rests on, so a change to the log fails here.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime

import pytest

from src.core.typed_graph import CALENDAR, COUNT, LEVEL, TypedEdge
from src.data.label_axis import (
    COLLAPSE_GROUP,
    COLLAPSED_LABEL,
    EXOGENOUS_NAMES,
    SCHEDULED_LABEL,
    agreement,
    collapse_labels,
    label_axis,
)
from src.data.lifesnaps import LifeSnapsEvent, read_lifesnaps

WHEN = datetime(2021, 5, 24, 8, 0)


def event(unit: str, label: str, day: int, minute: int) -> LifeSnapsEvent:
    return LifeSnapsEvent(
        unit=unit,
        label=label,
        day_index=day,
        start_min=minute,
        duration_s=1,
        start=WHEN,
        wave="spring",
        duration_source="instant",
    )


@pytest.fixture(scope="module")
def observed():
    events, _ = read_lifesnaps()
    return events


def test_no_within_group_edge_can_be_expressed(observed):
    collapsed, _ = collapse_labels(observed)
    axis = label_axis(collapsed)
    for source in COLLAPSE_GROUP:
        for target in COLLAPSE_GROUP:
            assert not axis.expressible(source, target)
    # The three raw columns are one vertex, so there is one index between them
    # and a graph over the axis has nowhere to put such an edge.
    assert len({axis.target_index(name) for name in COLLAPSE_GROUP}) == 1


def test_nothing_can_point_into_the_scheduled_label(observed):
    collapsed, _ = collapse_labels(observed)
    axis = label_axis(collapsed)
    assert SCHEDULED_LABEL not in axis.targets
    assert SCHEDULED_LABEL in axis.exogenous
    for source in axis.targets:
        assert not axis.expressible(source, SCHEDULED_LABEL)
    with pytest.raises(KeyError, match="never a target"):
        axis.target_index(SCHEDULED_LABEL)
    graph = axis.graph()
    with pytest.raises(ValueError, match="not one of"):
        graph.add_edge(
            TypedEdge(
                src_kind="observed",
                src_scale=1,
                src_label=0,
                src_quantity=COUNT,
                lag=1,
            ),
            dst_label=graph.L,
            dst_quantity=COUNT,
        )


def test_the_scheduled_label_reaches_a_target_as_an_exogenous_level(observed):
    collapsed, _ = collapse_labels(observed)
    axis = label_axis(collapsed)
    edge = axis.scheduled_edge(scale=2, lag=1)
    assert edge.src_kind == CALENDAR
    assert edge.src_quantity == LEVEL
    assert edge.is_exogenous
    graph = axis.graph()
    graph.add_edge(
        edge, dst_label=axis.target_index(COLLAPSED_LABEL), dst_quantity=COUNT
    )
    assert graph.n_edges() == 1
    # Its value is fixed by the study's own schedule, so reading it at the epoch
    # being predicted is admissible the way a clock node is.
    assert axis.scheduled_edge(scale=2, lag=0).lag == 0


def test_the_identity_the_collapse_rests_on_is_measured_on_the_log(observed):
    measured = agreement(observed)
    assert measured.active_days == 2055
    assert measured.days_in_agreement == 1996
    assert measured.share_in_agreement == pytest.approx(0.971, abs=0.001)
    assert measured.max_residual == 3
    assert measured.day_correlation > 0.98
    for name, hit, total in measured.coincident:
        assert name in COLLAPSE_GROUP
        assert hit / total > 0.99


def test_the_collapse_moves_every_episode_of_the_group_and_no_other(observed):
    collapsed, report = collapse_labels(observed)
    assert report.into == COLLAPSED_LABEL
    assert len(collapsed) == len(observed)
    before = sum(1 for e in observed if e.label in COLLAPSE_GROUP)
    assert report.episodes == before
    assert sum(1 for e in collapsed if e.label == COLLAPSED_LABEL) == before
    assert not any(e.label in COLLAPSE_GROUP for e in collapsed)
    unchanged = [
        (a.unit, a.day_index, a.start_min)
        for a, b in zip(observed, collapsed)
        if a.label not in COLLAPSE_GROUP and a == b
    ]
    assert len(unchanged) == len(observed) - before


def test_the_axis_is_the_same_before_and_after_the_collapse(observed):
    collapsed, report = collapse_labels(observed)
    assert label_axis(collapsed).digest() == report.axis_digest
    assert label_axis(observed).digest() == report.axis_digest


def test_a_collapsed_vertex_carries_instants_and_a_real_label_does_not(observed):
    axis = label_axis(collapse_labels(observed)[0])
    assert axis.is_instant(COLLAPSED_LABEL)
    assert not axis.is_instant("sleep")
    assert not axis.is_instant("nothing here")


def test_the_axis_refuses_a_label_the_log_does_not_carry(observed):
    axis = label_axis(collapse_labels(observed)[0])
    with pytest.raises(KeyError, match="not a label"):
        axis.axis_name("bicycle_polo")
    with pytest.raises(KeyError, match="not an exogenous source"):
        axis.exogenous_index("bicycle_polo")


def test_a_log_without_the_scheduled_label_carries_only_the_calendar():
    events = [event("a", "walk", 0, 10), event("a", "mood_survey", 0, 20)]
    axis = label_axis(events)
    assert axis.exogenous == tuple(n for n in EXOGENOUS_NAMES if n != SCHEDULED_LABEL)
    assert axis.targets == ("survey_response", "walk")
    with pytest.raises(KeyError):
        axis.scheduled_edge(scale=1, lag=1)


def test_a_day_holding_none_of_the_three_is_not_a_denominator():
    events = [event("a", "walk", d, 10) for d in range(5)]
    measured = agreement(events)
    assert measured.active_days == 0
    assert measured.share_in_agreement == 0.0
    assert measured.day_correlation == 0.0


def test_the_summary_names_what_the_collapse_did():
    events = [
        event("a", "notification_read", 0, 10),
        event("a", "mood_survey", 0, 10),
        event("a", "step_goal_survey", 1, 30),
        event("a", "notification_read", 1, 30),
        event("a", "walk", 2, 60),
    ]
    _, report = collapse_labels(events)
    text = report.summary()
    assert COLLAPSED_LABEL in text
    assert "2 episodes" not in text
    assert "4 episodes" in text
    assert "axis digest" in text
    assert "within one minute of a read" in text


def test_an_event_keeps_everything_but_its_label():
    original = event("a", "mood_survey", 3, 45)
    collapsed, _ = collapse_labels([original])
    assert collapsed[0] == dataclasses.replace(original, label=COLLAPSED_LABEL)
