"""
tests/test_phase51_dynamic_patterns_adapter.py
==============================================
The dynamic_patterns adapter, and the truth it is checked against.

Five properties carry the module.  A subject's verbatim day-scale arrays
reproduce the resolved manifest cell for cell, which is what makes the manifest
usable as an arithmetic truth.  Subjects built on one shared horizon come out the
same shape, which is what makes them comparable.  The zero-fill that gets them
there is marked unobserved and a mask handed to the ingest survives it, so a
stretch the log never covered is never read as a recorded silence.  The episode
policy is named at every call and the counts that come back carry the collisions
the arrays hold, so neither reading of the log is handed the other's arrays in
silence.  A carry-over rule becomes one typed edge at the day scale, and a rule
naming a label the log never realized comes back unmapped, which is what keeps an
unrecoverable edge out of a recall figure.

The benchmark tests read the real files.  The policy test picks the subject the
log collides most on and the earliest whole weeks holding both of its collisions,
so the two policies are compared where the collisions are.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta
from itertools import islice
from pathlib import Path

import numpy as np
import pytest

from src.core.typed_graph import COUNT, DURATION, OBSERVED, TypedEdge, TypedGraph
from src.data.dynamic_patterns import (
    BENCHMARK_BRANCHING,
    BENCHMARK_ORIGIN,
    DAY_SCALE,
    IDENTIFIABLE,
    VERBATIM,
    CarryOverRule,
    CellAgreement,
    EpisodeCounts,
    ResolvedCell,
    build_population,
    build_subject_data,
    compare_day_scale,
    count_collisions,
    iter_resolved_cells,
    merge_duplicate_starts,
    merge_same_label_overlap,
    read_declared_manifest,
    read_resolved_manifest,
    read_subject_event_log,
    relation_terms,
    shared_horizon_slots,
    study_period_slots,
    subject_persona,
)
from src.data.event_log import ObservedEvent, build_multiscale_data
from src.model.intensity import risk_set

WEEK_SLOTS = 7 * 24 * 60
HEADER = ("person_id", "label", "start_iso", "end_iso")


def _stamp(day: int, hour: int = 0, minute: int = 0) -> str:
    when = BENCHMARK_ORIGIN + timedelta(days=day, hours=hour, minutes=minute)
    return when.isoformat(timespec="seconds")


def _episode(label: str, day: int, hour: int, minute: int, length: int):
    start = BENCHMARK_ORIGIN + timedelta(days=day, hours=hour, minutes=minute)
    return ObservedEvent(
        label=label, start=start, end=start + timedelta(minutes=length)
    )


def _write_log(path, rows, delimiter="\t", header=HEADER) -> str:
    lines = [delimiter.join(header)] + [delimiter.join(r) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _two_subject_log(path) -> str:
    rows = [
        ("ann_0000", "walk", _stamp(0, 8), _stamp(0, 8, 30)),
        ("ann_0000", "sleep", _stamp(0, 22), _stamp(1, 6)),
        ("ann_0000", "walk", _stamp(1, 9), _stamp(1, 9, 20)),
        ("bob_0001", "walk", _stamp(0, 7), _stamp(0, 7, 45)),
        ("bob_0001", "sleep", _stamp(8, 23), _stamp(9, 7)),
    ]
    return _write_log(path, rows)


# ---------------------------------------------------------------------------
# Reading the log
# ---------------------------------------------------------------------------


def test_the_log_comes_back_grouped_by_subject(tmp_path) -> None:
    by_subject = read_subject_event_log(_two_subject_log(tmp_path / "log.tsv"))
    assert list(by_subject) == ["ann_0000", "bob_0001"]
    assert [e.label for e in by_subject["ann_0000"]] == ["walk", "sleep", "walk"]
    first = by_subject["bob_0001"][0]
    assert first.start == datetime(2026, 2, 16, 7, 0)
    assert first.end == datetime(2026, 2, 16, 7, 45)


def test_a_missing_column_is_named(tmp_path) -> None:
    path = _write_log(
        tmp_path / "log.tsv",
        [("ann_0000", "walk", _stamp(0, 8))],
        header=("person_id", "label", "start_iso"),
    )
    with pytest.raises(ValueError, match="column 'end_iso' is missing"):
        read_subject_event_log(path)


def test_an_unreadable_timestamp_names_its_row(tmp_path) -> None:
    path = _write_log(tmp_path / "log.tsv", [("ann_0000", "walk", "yesterday", "now")])
    with pytest.raises(ValueError, match="row 2 .* unreadable timestamp"):
        read_subject_event_log(path)


def test_an_episode_that_ends_before_it_starts_is_rejected(tmp_path) -> None:
    path = _write_log(
        tmp_path / "log.tsv", [("ann_0000", "walk", _stamp(1, 8), _stamp(0, 8))]
    )
    with pytest.raises(ValueError, match="row 2 .* ends at .* before its start"):
        read_subject_event_log(path)


def test_a_blank_subject_is_rejected(tmp_path) -> None:
    path = _write_log(
        tmp_path / "log.tsv", [("  ", "walk", _stamp(0, 8), _stamp(0, 9))]
    )
    with pytest.raises(ValueError, match="row 2 .* names no subject"):
        read_subject_event_log(path)


def test_a_timezone_aware_timestamp_is_rejected(tmp_path) -> None:
    path = _write_log(
        tmp_path / "log.tsv",
        [
            (
                "ann_0000",
                "walk",
                "2026-02-16T08:00:00+01:00",
                "2026-02-16T09:00:00+01:00",
            )
        ],
    )
    with pytest.raises(ValueError, match="timezone-aware"):
        read_subject_event_log(path)


def test_the_column_names_and_the_delimiter_are_arguments(tmp_path) -> None:
    path = _write_log(
        tmp_path / "log.csv",
        [("ann_0000", "walk", _stamp(0, 8), _stamp(0, 9))],
        delimiter=",",
        header=("who", "what", "from", "to"),
    )
    by_subject = read_subject_event_log(
        path,
        subject_col="who",
        label_col="what",
        start_col="from",
        end_col="to",
        delimiter=",",
    )
    assert by_subject["ann_0000"][0].label == "walk"


@pytest.mark.parametrize(
    "person_id,expected",
    [("diligent_anxious_active_0000", "diligent_anxious_active"), ("solo", "solo")],
)
def test_the_persona_is_the_subject_id_without_its_number(person_id, expected) -> None:
    assert subject_persona(person_id) == expected


# ---------------------------------------------------------------------------
# The episode policies
# ---------------------------------------------------------------------------


def test_two_starts_in_one_minute_fold_into_the_longer_one() -> None:
    events = [
        _episode("hungry", 0, 9, 0, 10),
        _episode("hungry", 0, 9, 0, 25),
        _episode("hungry", 0, 11, 0, 5),
    ]
    kept, dropped = merge_duplicate_starts(events)
    assert dropped == 1
    assert [(e.start.hour, (e.end - e.start).seconds // 60) for e in kept] == [
        (9, 25),
        (11, 5),
    ]


def test_a_shorter_duplicate_leaves_the_kept_end_alone() -> None:
    events = [_episode("hungry", 0, 9, 0, 25), _episode("hungry", 0, 9, 0, 10)]
    kept, dropped = merge_duplicate_starts(events)
    assert dropped == 1
    assert kept[0].end - kept[0].start == timedelta(minutes=25)


def test_a_log_with_no_duplicate_start_is_returned_unchanged() -> None:
    events = [_episode("walk", 0, 8, 0, 30), _episode("sleep", 0, 8, 0, 30)]
    kept, dropped = merge_duplicate_starts(events)
    assert dropped == 0 and kept == events


def test_overlapping_episodes_of_one_label_fold_into_their_union() -> None:
    events = [_episode("walk", 0, 8, 0, 30), _episode("walk", 0, 8, 10, 40)]
    kept, dropped = merge_same_label_overlap(events)
    assert dropped == 1
    assert kept[0].start == BENCHMARK_ORIGIN + timedelta(hours=8)
    assert kept[0].end == BENCHMARK_ORIGIN + timedelta(hours=8, minutes=50)


def test_an_episode_beginning_at_another_s_end_stays_separate() -> None:
    """The risk set leaves slot t+Y free, so that start needs no merge."""
    events = [_episode("walk", 0, 8, 0, 30), _episode("walk", 0, 8, 30, 10)]
    kept, dropped = merge_same_label_overlap(events)
    assert dropped == 0 and len(kept) == 2


def test_a_contained_episode_folds_away() -> None:
    events = [_episode("walk", 0, 8, 0, 60), _episode("walk", 0, 8, 10, 5)]
    kept, dropped = merge_same_label_overlap(events)
    assert dropped == 1
    assert kept[0].end - kept[0].start == timedelta(minutes=60)


def test_two_labels_never_merge_into_each_other() -> None:
    events = [_episode("walk", 0, 8, 0, 60), _episode("sleep", 0, 8, 10, 60)]
    kept, dropped = merge_same_label_overlap(events)
    assert dropped == 0
    assert [e.label for e in kept] == ["walk", "sleep"]


def test_episode_counts_report_what_survived() -> None:
    counts = EpisodeCounts(read=100, duplicate_starts=3, overlaps=7)
    assert counts.kept == 90


def test_two_starts_in_one_slot_are_counted_as_stacked() -> None:
    events = [_episode("hungry", 0, 4, 11, 55), _episode("hungry", 0, 4, 11, 34)]
    assert count_collisions(events) == (1, 0)


def test_a_start_inside_a_running_episode_is_counted_as_shadowed() -> None:
    events = [_episode("walk", 0, 8, 0, 30), _episode("walk", 0, 8, 10, 5)]
    assert count_collisions(events) == (0, 1)


def test_a_start_at_the_end_of_a_running_episode_collides_with_nothing() -> None:
    """Slot t+Y is at risk again, so a start there is one the likelihood keeps."""
    events = [_episode("walk", 0, 8, 0, 30), _episode("walk", 0, 8, 30, 10)]
    assert count_collisions(events) == (0, 0)


def test_a_stacked_pair_inside_a_running_episode_is_counted_in_both_figures() -> None:
    """Stacking costs an episode and shadowing costs the slot that holds them."""
    events = [
        _episode("walk", 0, 8, 0, 60),
        _episode("walk", 0, 8, 20, 5),
        _episode("walk", 0, 8, 20, 9),
    ]
    assert count_collisions(events) == (1, 1)


def test_two_labels_sharing_a_slot_collide_with_nothing() -> None:
    events = [_episode("walk", 0, 8, 0, 30), _episode("sleep", 0, 8, 0, 30)]
    assert count_collisions(events) == (0, 0)


def test_a_wider_slot_stacks_starts_a_minute_apart() -> None:
    events = [_episode("walk", 0, 8, 0, 5), _episode("walk", 0, 8, 7, 5)]
    assert count_collisions(events) == (0, 0)
    assert count_collisions(events, slot_minutes=15) == (1, 0)


# ---------------------------------------------------------------------------
# The shared grid
# ---------------------------------------------------------------------------


def test_the_horizon_covers_every_subject_and_lands_on_whole_weeks(tmp_path) -> None:
    by_subject = read_subject_event_log(_two_subject_log(tmp_path / "log.tsv"))
    assert shared_horizon_slots(by_subject) == 2 * WEEK_SLOTS
    assert shared_horizon_slots({"ann_0000": by_subject["ann_0000"]}) == WEEK_SLOTS


def test_a_log_with_no_episode_fixes_no_horizon() -> None:
    with pytest.raises(ValueError, match="fixes no horizon"):
        shared_horizon_slots({"ann_0000": []})


def test_every_subject_gets_the_same_shape_on_the_shared_horizon(tmp_path) -> None:
    by_subject = read_subject_event_log(_two_subject_log(tmp_path / "log.tsv"))
    labels = ["sleep", "walk"]
    horizon = shared_horizon_slots(by_subject)
    shapes = set()
    for events in by_subject.values():
        data, calendar, _ = build_subject_data(
            events, labels, IDENTIFIABLE, horizon_slots=horizon
        )
        shapes.add(data.get_X(1).shape)
        assert calendar.origin == BENCHMARK_ORIGIN
        assert calendar.epoch_span(DAY_SCALE) == timedelta(days=1)
    assert shapes == {(2 * WEEK_SLOTS, 2)}


def test_padding_leaves_the_ladder_reconciled(tmp_path) -> None:
    by_subject = read_subject_event_log(_two_subject_log(tmp_path / "log.tsv"))
    data, _, _ = build_subject_data(
        by_subject["ann_0000"],
        ["sleep", "walk"],
        VERBATIM,
        horizon_slots=2 * WEEK_SLOTS,
    )
    assert data.check_reconciliation() == {"RecErr_X": 0.0, "RecErr_Y": 0.0}
    X_day, Y_day = data.get_X(DAY_SCALE), data.get_Y(DAY_SCALE)
    assert X_day.shape == (14, 2)
    # Day 0 opens one walk of 30 minutes and one sleep of 8 hours, and the sleep
    # is charged whole to the day it began.
    assert list(X_day[0]) == [1, 1] and list(Y_day[0]) == [480, 30]
    assert list(X_day[1]) == [0, 1] and list(Y_day[1]) == [0, 20]
    assert int(X_day[2:].sum()) == 0


def test_a_horizon_shorter_than_the_log_is_rejected(tmp_path) -> None:
    by_subject = read_subject_event_log(_two_subject_log(tmp_path / "log.tsv"))
    with pytest.raises(ValueError, match="ends before this subject's last episode"):
        build_subject_data(
            by_subject["bob_0001"],
            ["sleep", "walk"],
            VERBATIM,
            horizon_slots=WEEK_SLOTS,
        )


def test_a_horizon_that_is_not_whole_weeks_is_rejected(tmp_path) -> None:
    by_subject = read_subject_event_log(_two_subject_log(tmp_path / "log.tsv"))
    with pytest.raises(ValueError, match="not a whole number of coarsest epochs"):
        build_subject_data(
            by_subject["ann_0000"],
            ["sleep", "walk"],
            VERBATIM,
            horizon_slots=WEEK_SLOTS + 60,
        )


def test_a_subject_already_on_the_horizon_is_left_alone(tmp_path) -> None:
    by_subject = read_subject_event_log(_two_subject_log(tmp_path / "log.tsv"))
    plain, _, _ = build_subject_data(
        by_subject["ann_0000"], ["sleep", "walk"], VERBATIM
    )
    padded, _, _ = build_subject_data(
        by_subject["ann_0000"], ["sleep", "walk"], VERBATIM, horizon_slots=WEEK_SLOTS
    )
    assert plain.get_X(1).shape == padded.get_X(1).shape == (WEEK_SLOTS, 2)
    assert np.array_equal(plain.get_Y(DAY_SCALE), padded.get_Y(DAY_SCALE))
    assert plain.observed is None and padded.observed is None


# ---------------------------------------------------------------------------
# The observation mask
# ---------------------------------------------------------------------------


def test_the_fill_up_to_the_shared_horizon_is_marked_unobserved(tmp_path) -> None:
    """A zero the log never recorded is an absence nobody watched."""
    by_subject = read_subject_event_log(_two_subject_log(tmp_path / "log.tsv"))
    plain, _, _ = build_subject_data(
        by_subject["ann_0000"], ["sleep", "walk"], VERBATIM
    )
    padded, _, _ = build_subject_data(
        by_subject["ann_0000"],
        ["sleep", "walk"],
        VERBATIM,
        horizon_slots=2 * WEEK_SLOTS,
    )
    assert plain.observed is None
    assert padded.observed.shape == (2 * WEEK_SLOTS,)
    assert padded.observed[:WEEK_SLOTS].all()
    assert not padded.observed[WEEK_SLOTS:].any()
    # The fill runs from one coarsest epoch boundary to another, so whole weeks
    # go unobserved and the ladder above them stays readable.
    assert list(padded.get_observed(DAY_SCALE)) == [True] * 7 + [False] * 7
    assert padded.check_reconciliation() == {"RecErr_X": 0.0, "RecErr_Y": 0.0}


def test_the_arrays_of_a_filled_subject_are_the_ones_it_had_before(tmp_path) -> None:
    """The mask is the only thing the fill adds; every count and duration holds."""
    by_subject = read_subject_event_log(_two_subject_log(tmp_path / "log.tsv"))
    plain, _, plain_counts = build_subject_data(
        by_subject["ann_0000"], ["sleep", "walk"], VERBATIM
    )
    padded, _, padded_counts = build_subject_data(
        by_subject["ann_0000"],
        ["sleep", "walk"],
        VERBATIM,
        horizon_slots=2 * WEEK_SLOTS,
    )
    assert plain_counts == padded_counts
    assert np.array_equal(padded.get_X(1)[:WEEK_SLOTS], plain.get_X(1))
    assert np.array_equal(padded.get_Y(1)[:WEEK_SLOTS], plain.get_Y(1))
    assert int(padded.get_X(1)[WEEK_SLOTS:].sum()) == 0
    assert np.array_equal(padded.get_X(DAY_SCALE)[:7], plain.get_X(DAY_SCALE))
    assert np.array_equal(padded.get_Y(DAY_SCALE)[:7], plain.get_Y(DAY_SCALE))


def test_a_mask_handed_to_the_padding_survives_it(tmp_path) -> None:
    """A recorder that went down on day 6 is still down after the fill."""
    by_subject = read_subject_event_log(_two_subject_log(tmp_path / "log.tsv"))
    watched = np.ones(WEEK_SLOTS, dtype=bool)
    watched[6 * 24 * 60 :] = False
    data, _, _ = build_subject_data(
        by_subject["ann_0000"],
        ["sleep", "walk"],
        VERBATIM,
        horizon_slots=2 * WEEK_SLOTS,
        observed=watched,
    )
    assert list(data.get_observed(DAY_SCALE)) == [True] * 6 + [False] * 8


def test_a_mask_reaches_the_container_when_nothing_is_filled(tmp_path) -> None:
    by_subject = read_subject_event_log(_two_subject_log(tmp_path / "log.tsv"))
    watched = np.ones(WEEK_SLOTS, dtype=bool)
    watched[3 * 24 * 60 :] = False
    data, _, _ = build_subject_data(
        by_subject["ann_0000"], ["sleep", "walk"], VERBATIM, observed=watched
    )
    assert np.array_equal(data.observed, watched)


def test_the_reader_counts_the_pad_to_a_whole_coarsest_epoch_as_watched() -> None:
    """That pad shares its epoch with recorded slots, which the ladder cannot split."""
    events = [_episode("walk", 0, 0, 0, 1)]
    data, _, _ = build_multiscale_data(
        events,
        slot_minutes=1,
        branching=[2],
        labels=["walk"],
        origin=BENCHMARK_ORIGIN,
        observed=np.array([False]),
    )
    assert list(data.observed) == [False, True]


def test_the_reader_leaves_an_unmasked_dataset_unmasked() -> None:
    events = [_episode("walk", 0, 0, 0, 1)]
    data, _, _ = build_multiscale_data(
        events, slot_minutes=1, branching=[2], labels=["walk"], origin=BENCHMARK_ORIGIN
    )
    assert data.observed is None
    assert data.get_observed(1).all()


@pytest.mark.parametrize(
    "mask,message",
    [
        (np.zeros(2), "boolean array"),
        (np.ones(5, dtype=bool), "at most 2 of them"),
        (np.ones((2, 1), dtype=bool), "at most 2 of them"),
    ],
)
def test_a_mask_that_is_not_one_flag_per_slot_is_rejected(mask, message) -> None:
    with pytest.raises(ValueError, match=message):
        build_multiscale_data(
            [_episode("walk", 0, 0, 0, 1)],
            slot_minutes=1,
            branching=[2],
            labels=["walk"],
            origin=BENCHMARK_ORIGIN,
            observed=mask,
        )


def _colliding_day() -> list:
    """One label, one day: a duplicate start, and a third start inside them both."""
    return [
        _episode("walk", 0, 8, 0, 30),
        _episode("walk", 0, 8, 0, 45),
        _episode("walk", 0, 8, 20, 60),
    ]


def test_the_verbatim_policy_hands_back_every_episode_and_the_collisions() -> None:
    data, _, counts = build_subject_data(_colliding_day(), ["walk"], VERBATIM)
    assert (counts.read, counts.kept) == (3, 3)
    assert (counts.stacked_starts, counts.shadowed_starts) == (1, 1)
    # Three episodes, three starts and 135 minutes at the day scale, which is
    # what the resolved manifest of such a day would hold.
    assert int(data.get_X(DAY_SCALE)[0, 0]) == 3
    assert int(data.get_Y(DAY_SCALE)[0, 0]) == 135


def test_the_verbatim_finest_scale_is_what_the_collision_counts_say_it_is() -> None:
    """The stacked start folds into one slot, the shadowed one leaves the risk set."""
    data, _, counts = build_subject_data(_colliding_day(), ["walk"], VERBATIM)
    X1, Y1 = data.get_X(1), data.get_Y(1)
    slot = 8 * 60
    assert int(X1[slot, 0]) == 2 and int(Y1[slot, 0]) == 75
    assert counts.stacked_starts == 1
    assert not bool(risk_set(X1, Y1)[slot + 20, 0])
    assert counts.shadowed_starts == 1


def test_the_identifiable_policy_folds_the_collisions_away() -> None:
    data, _, counts = build_subject_data(_colliding_day(), ["walk"], IDENTIFIABLE)
    assert (counts.read, counts.duplicate_starts, counts.overlaps) == (3, 1, 1)
    assert counts.kept == 1
    assert (counts.stacked_starts, counts.shadowed_starts) == (0, 0)
    assert int(data.get_X(DAY_SCALE)[0, 0]) == 1
    assert int(data.get_Y(DAY_SCALE)[0, 0]) == 80
    X1, Y1 = data.get_X(1), data.get_Y(1)
    assert not (X1 > 0)[~risk_set(X1, Y1)].any()


def test_an_unnamed_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="episode policy 'as_read'"):
        build_subject_data(_colliding_day(), ["walk"], "as_read")


def test_a_slot_too_wide_to_keep_two_starts_apart_is_rejected() -> None:
    """The merges leave starts seven minutes apart alone, so a wide slot stacks them."""
    events = [_episode("walk", 0, 8, 0, 5), _episode("walk", 0, 8, 7, 5)]
    _, _, counts = build_subject_data(events, ["walk"], IDENTIFIABLE)
    assert (counts.kept, counts.stacked_starts) == (2, 0)
    with pytest.raises(ValueError, match="starts in finest slot"):
        build_subject_data(
            events, ["walk"], IDENTIFIABLE, slot_minutes=15, branching=(4, 24, 7)
        )


def test_pooling_several_subjects_reads_the_log_verbatim(tmp_path) -> None:
    """Two subjects begin one label in one minute, so a pooled slot holds two starts."""
    rows = [
        ("ann_0000", "walk", _stamp(0, 8), _stamp(0, 8, 30)),
        ("bob_0001", "walk", _stamp(0, 8), _stamp(0, 8, 20)),
    ]
    by_subject = read_subject_event_log(_write_log(tmp_path / "log.tsv", rows))
    pooled = [e for events in by_subject.values() for e in events]
    data, _, counts = build_subject_data(pooled, ["walk"], VERBATIM)
    assert counts.stacked_starts == 1
    assert int(data.get_X(DAY_SCALE)[0, 0]) == 2
    assert int(data.get_Y(DAY_SCALE)[0, 0]) == 50


# ---------------------------------------------------------------------------
# The declared manifest
# ---------------------------------------------------------------------------


def _rule(**over) -> dict:
    spec = {
        "source": "sleep",
        "target": "walking",
        "source_quantity": "max_duration",
        "target_quantity": "episodes",
        "lag_days": 1,
        "unit": "count",
        "amount": 2.0,
        "direction": "increasing",
        "source_threshold": 420,
        "personas": None,
    }
    spec.update(over)
    return spec


def _declared(path, rules=None) -> str:
    rules = _rule() if rules is None else rules
    payload = {
        "sleep": {
            "kind": "event",
            "category": "sleep",
            "stationary": False,
            "personas": ["busy"],
            "patterns": [{"mode": "trend", "details": {"target": "placement"}}],
            "edges_out": [rules],
            "edges_in": [],
        },
        "walking": {
            "kind": "event",
            "category": "sports",
            "stationary": False,
            "personas": ["busy"],
            "patterns": [{"mode": "seasonality", "details": {"scale": "weekday"}}],
            "edges_out": [],
            "edges_in": [rules],
        },
        "alone": {
            "kind": "context",
            "category": "social_context",
            "stationary": True,
            "personas": ["busy"],
            "patterns": [],
            "edges_out": [],
            "edges_in": [],
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_a_rule_stated_on_both_ends_is_read_once(tmp_path) -> None:
    truth = read_declared_manifest(_declared(tmp_path / "declared.json"))
    assert len(truth.rules) == 1
    rule = truth.rules[0]
    assert (rule.source, rule.target, rule.lag_days) == ("sleep", "walking", 1)
    assert rule.source_threshold == 420.0 and rule.amount == 2.0


def test_the_declared_labels_carry_their_kind_and_their_patterns(tmp_path) -> None:
    truth = read_declared_manifest(_declared(tmp_path / "declared.json"))
    assert truth.stationary_labels == ["alone"]
    assert truth.labels["alone"].kind == "context"
    assert truth.labels["sleep"].modes == ("trend",)
    assert truth.labels["alone"].modes == ()
    assert truth.labels["walking"].personas == ("busy",)


def test_a_rule_becomes_one_typed_edge_at_the_day_scale(tmp_path) -> None:
    truth = read_declared_manifest(_declared(tmp_path / "declared.json"))
    mapped, off_axis = truth.typed_edges(["alone", "sleep", "walking"])
    assert off_axis == []
    assert len(mapped) == 1
    declared = mapped[0]
    assert declared.edge == TypedEdge(
        src_kind=OBSERVED,
        src_scale=DAY_SCALE,
        src_label=1,
        src_quantity=DURATION,
        lag=1,
    )
    assert (declared.dst_label, declared.dst_quantity) == (2, COUNT)
    assert declared.rule.sign == 1


def test_a_duration_target_from_a_count_source_maps_to_the_other_pairing(
    tmp_path,
) -> None:
    path = _declared(
        tmp_path / "declared.json",
        _rule(
            source="sleep",
            target="walking",
            source_quantity="count",
            target_quantity="duration",
            direction="decreasing",
            source_threshold=None,
        ),
    )
    mapped, _ = read_declared_manifest(path).typed_edges(["sleep", "walking"])
    assert mapped[0].edge.src_quantity == COUNT
    assert mapped[0].dst_quantity == DURATION
    assert mapped[0].rule.sign == -1
    assert mapped[0].rule.source_threshold is None


def test_a_rule_naming_a_label_the_axis_lacks_comes_back_unmapped(tmp_path) -> None:
    """A declared label with no episode has no column, so its rule is unrecoverable."""
    truth = read_declared_manifest(_declared(tmp_path / "declared.json"))
    mapped, off_axis = truth.typed_edges(["sleep", "alone"])
    assert mapped == []
    assert [r.target for r in off_axis] == ["walking"]


def test_a_persona_rule_is_dropped_for_another_persona(tmp_path) -> None:
    path = _declared(tmp_path / "declared.json", _rule(personas=["busy"]))
    truth = read_declared_manifest(path)
    axis = ["sleep", "walking"]
    assert len(truth.typed_edges(axis, persona="busy")[0]) == 1
    assert truth.typed_edges(axis, persona="calm")[0] == []
    assert len(truth.typed_edges(axis)[0]) == 1


def test_a_rule_for_every_persona_survives_any_filter(tmp_path) -> None:
    truth = read_declared_manifest(_declared(tmp_path / "declared.json"))
    assert truth.rules[0].personas == ()
    assert len(truth.typed_edges(["sleep", "walking"], persona="calm")[0]) == 1


def test_any_basis_term_of_the_declared_relation_is_the_rule_recovered(
    tmp_path,
) -> None:
    """The rule is the relation; the basis term is how the fit expressed it."""
    truth = read_declared_manifest(_declared(tmp_path / "declared.json"))
    mapped, _ = truth.typed_edges(["sleep", "walking"])
    declared = mapped[0].edge
    band = replace(declared, basis=(420.0, np.inf))
    assert relation_terms([band], declared) == [band]
    assert relation_terms([declared, band], declared) == sorted([declared, band])
    assert relation_terms([], declared) == []


def test_a_term_of_another_relation_is_not_the_declared_rule(tmp_path) -> None:
    """A band is read off the same source, lag and quantity or it is another edge."""
    truth = read_declared_manifest(_declared(tmp_path / "declared.json"))
    declared = truth.typed_edges(["sleep", "walking"])[0][0].edge
    band = (420.0, np.inf)
    others = [
        replace(declared, lag=2, basis=band),
        replace(declared, src_quantity=COUNT, basis=band),
        replace(declared, src_label=1, basis=band),
        replace(declared, src_scale=DAY_SCALE + 1, basis=band),
    ]
    assert relation_terms(others, declared) == []


def test_the_truth_graph_scores_against_a_fitted_graph(tmp_path) -> None:
    truth = read_declared_manifest(_declared(tmp_path / "declared.json"))
    axis = ["sleep", "walking"]
    declared = truth.truth_graph(axis)
    assert declared.n_edges() == 1
    assert declared.structural_hamming_distance(declared) == 0
    assert declared.structural_hamming_distance(TypedGraph(label_names=axis)) == 1


@pytest.mark.parametrize(
    "over,message",
    [
        ({"source_quantity": "vibes"}, "source quantity"),
        ({"target_quantity": "vibes"}, "target quantity"),
        ({"direction": "sideways"}, "direction"),
        ({"lag_days": 0}, "reads the day being predicted"),
    ],
)
def test_a_rule_the_model_cannot_express_is_rejected(tmp_path, over, message) -> None:
    path = _declared(tmp_path / "declared.json", _rule(**over))
    with pytest.raises(ValueError, match=message):
        read_declared_manifest(path)


def test_a_rule_built_by_hand_is_checked_the_same_way() -> None:
    rule = CarryOverRule(
        source="a",
        target="b",
        source_quantity="total_duration",
        target_quantity="duration",
        lag_days=7,
        amount=30.0,
        direction="decreasing",
    )
    assert rule.typed_edge(0, DAY_SCALE).lag == 7
    assert rule.applies_to(None) and rule.applies_to("anyone")


# ---------------------------------------------------------------------------
# The resolved manifest
# ---------------------------------------------------------------------------


def _cell(day: int, label: str, count: int, minutes: int, longest: int) -> dict:
    return {
        "person_id": "ann_0000",
        "persona_id": "ann",
        "date": "2026-02-16",
        "day_index": day,
        "label": label,
        "resolved_count": count,
        "realized_count": count,
        "realized_total_minutes": minutes,
        "realized_max_minutes": longest,
    }


def _resolved(path, records) -> str:
    # A trailing blank line is what a line-wise writer leaves behind.
    path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n\n", encoding="utf-8"
    )
    return str(path)


def _ann_cells(path) -> str:
    return _resolved(
        path,
        [
            _cell(0, "sleep", 1, 480, 480),
            _cell(0, "walk", 1, 30, 30),
            _cell(1, "walk", 1, 20, 20),
            _cell(1, "sleep", 0, 0, 0),
        ],
    )


def test_the_resolved_cells_come_back_in_file_order(tmp_path) -> None:
    cells = read_resolved_manifest(_ann_cells(tmp_path / "resolved.jsonl"))
    assert len(cells) == 4
    assert [(c.day_index, c.label) for c in cells[:2]] == [(0, "sleep"), (0, "walk")]
    assert cells[0] == ResolvedCell(
        person_id="ann_0000",
        persona_id="ann",
        day_index=0,
        label="sleep",
        resolved_count=1,
        realized_count=1,
        realized_total_minutes=480,
        realized_max_minutes=480,
    )


def test_the_day_arrays_reproduce_the_cells(tmp_path) -> None:
    by_subject = read_subject_event_log(_two_subject_log(tmp_path / "log.tsv"))
    data, _, _ = build_subject_data(
        by_subject["ann_0000"], ["sleep", "walk"], VERBATIM, horizon_slots=WEEK_SLOTS
    )
    cells = read_resolved_manifest(_ann_cells(tmp_path / "resolved.jsonl"))
    agreement = compare_day_scale(data, cells)
    assert agreement == CellAgreement(n_cells=4, count_error=0, duration_error=0)
    assert agreement.exact


def test_a_gap_against_the_cells_is_reported_as_its_size(tmp_path) -> None:
    by_subject = read_subject_event_log(_two_subject_log(tmp_path / "log.tsv"))
    data, _, _ = build_subject_data(
        by_subject["ann_0000"], ["sleep", "walk"], VERBATIM, horizon_slots=WEEK_SLOTS
    )
    path = _resolved(tmp_path / "resolved.jsonl", [_cell(0, "walk", 3, 100, 50)])
    agreement = compare_day_scale(data, read_resolved_manifest(path))
    assert (agreement.count_error, agreement.duration_error) == (2, 70)
    assert not agreement.exact


def test_a_cell_naming_an_unknown_label_is_rejected(tmp_path) -> None:
    by_subject = read_subject_event_log(_two_subject_log(tmp_path / "log.tsv"))
    data, _, _ = build_subject_data(by_subject["ann_0000"], ["sleep", "walk"], VERBATIM)
    stream = iter_resolved_cells(
        _resolved(tmp_path / "resolved.jsonl", [_cell(0, "swimming", 1, 30, 30)])
    )
    with pytest.raises(ValueError, match="which the arrays do not carry"):
        compare_day_scale(data, stream)


def test_a_cell_past_the_horizon_is_rejected(tmp_path) -> None:
    by_subject = read_subject_event_log(_two_subject_log(tmp_path / "log.tsv"))
    data, _, _ = build_subject_data(by_subject["ann_0000"], ["sleep", "walk"], VERBATIM)
    cells = read_resolved_manifest(
        _resolved(tmp_path / "resolved.jsonl", [_cell(90, "walk", 1, 30, 30)])
    )
    with pytest.raises(ValueError, match="past the horizon of 7 days"):
        compare_day_scale(data, cells)


def test_a_scale_that_is_not_a_day_is_rejected(tmp_path) -> None:
    by_subject = read_subject_event_log(_two_subject_log(tmp_path / "log.tsv"))
    data, _, _ = build_subject_data(by_subject["ann_0000"], ["sleep", "walk"], VERBATIM)
    cells = read_resolved_manifest(_ann_cells(tmp_path / "resolved.jsonl"))
    with pytest.raises(ValueError, match="epochs are not days"):
        compare_day_scale(data, cells, day_scale=2)


def test_arrays_without_a_calendar_are_compared_as_given(tmp_path) -> None:
    by_subject = read_subject_event_log(_two_subject_log(tmp_path / "log.tsv"))
    data, _, _ = build_subject_data(by_subject["ann_0000"], ["sleep", "walk"], VERBATIM)
    data.calendar = None
    cells = read_resolved_manifest(_ann_cells(tmp_path / "resolved.jsonl"))
    assert compare_day_scale(data, cells).exact


# ---------------------------------------------------------------------------
# The benchmark itself, on the subject and the weeks its collisions are in
# ---------------------------------------------------------------------------

BENCHMARK = Path(__file__).resolve().parents[1] / "data" / "dynamic_patterns"
MAX_LOG_ROWS = 3000
BENCHMARK_WEEKS = 12


def _window(events, weeks: int) -> list:
    """The episodes of a subject beginning inside the first whole weeks of the log."""
    end = BENCHMARK_ORIGIN + timedelta(weeks=weeks)
    return [e for e in events if e.start < end]


def _most_stacked_subject(by_subject) -> str:
    """The subject whose log packs the most starts of one label into one minute."""
    stacked = {s: count_collisions(evs)[0] for s, evs in by_subject.items()}
    return max(sorted(stacked), key=lambda s: stacked[s])


def _weeks_holding_both_collisions(events) -> int:
    """The fewest whole weeks whose episodes stack a start and shadow another."""
    for weeks in range(1, BENCHMARK_WEEKS + 1):
        stacked, shadowed = count_collisions(_window(events, weeks))
        if stacked and shadowed:
            return weeks
    return 0


@pytest.mark.skipif(
    not BENCHMARK.is_dir(), reason="the dynamic_patterns benchmark is not in the tree"
)
def test_the_two_policies_part_ways_where_the_benchmark_collides() -> None:
    by_subject = read_subject_event_log(str(BENCHMARK / "event_log.tsv"))
    subject = _most_stacked_subject(by_subject)
    weeks = _weeks_holding_both_collisions(by_subject[subject])
    assert weeks, f"{subject} holds no window where the finest scale loses a start"

    window = _window(by_subject[subject], weeks)
    cells = [
        c
        for c in iter_resolved_cells(str(BENCHMARK / "manifest_resolved.jsonl"))
        if c.person_id == subject and c.day_index < 7 * weeks
    ]
    assert cells, f"the resolved manifest says nothing about {subject}"
    labels = sorted({e.label for e in window} | {c.label for c in cells})
    horizon = shared_horizon_slots({subject: window})

    as_read, _, read_counts = build_subject_data(
        window, labels, VERBATIM, horizon_slots=horizon
    )
    assert read_counts.read == read_counts.kept == len(window)
    assert read_counts.stacked_starts > 0 and read_counts.shadowed_starts > 0
    # The day scale carries the collisions and reproduces the manifest anyway,
    # which is what the verbatim reading is for.
    assert compare_day_scale(as_read, cells) == CellAgreement(
        n_cells=len(cells), count_error=0, duration_error=0
    )
    X1, Y1 = as_read.get_X(1), as_read.get_Y(1)
    assert int(X1.max()) > 1
    assert bool(((X1 > 0) & ~risk_set(X1, Y1)).any())

    merged, _, merged_counts = build_subject_data(
        window, labels, IDENTIFIABLE, horizon_slots=horizon
    )
    folds = merged_counts.read - merged_counts.kept
    assert folds > 0
    assert (merged_counts.stacked_starts, merged_counts.shadowed_starts) == (0, 0)
    X2, Y2 = merged.get_X(1), merged.get_Y(1)
    assert int(X2.max()) == 1
    assert not ((X2 > 0) & ~risk_set(X2, Y2)).any()
    # Every fold takes one episode off the day it began, so the price of an
    # identifiable finest scale is that many counts against the manifest.
    priced = compare_day_scale(merged, cells)
    assert priced.count_error == folds
    assert priced.duration_error > 0


@pytest.mark.skipif(
    not BENCHMARK.is_dir(), reason="the dynamic_patterns benchmark is not in the tree"
)
def test_a_benchmark_subject_short_of_the_horizon_is_masked_over_its_fill() -> None:
    """The subjects whose logs stop early are the ones the shared horizon fills."""
    by_subject = read_subject_event_log(str(BENCHMARK / "event_log.tsv"))
    horizon = shared_horizon_slots(by_subject)
    own = {s: shared_horizon_slots({s: evs}) for s, evs in sorted(by_subject.items())}
    short = [s for s, reach in own.items() if reach < horizon]
    assert short, "no subject stops short of the horizon, so nothing is filled"

    subject = short[0]
    events = by_subject[subject]
    data, _, _ = build_subject_data(
        events, sorted({e.label for e in events}), VERBATIM, horizon_slots=horizon
    )
    reach = own[subject]
    assert int((~data.get_observed(1)).sum()) == horizon - reach
    assert data.observed[:reach].all()
    assert not data.observed[reach:].any()
    assert int(data.get_X(1)[reach:].sum()) == 0
    assert data.check_reconciliation() == {"RecErr_X": 0.0, "RecErr_Y": 0.0}


@pytest.mark.skipif(
    not BENCHMARK.is_dir(), reason="the dynamic_patterns benchmark is not in the tree"
)
def test_every_declared_rule_of_the_benchmark_reaches_the_label_axis(
    tmp_path,
) -> None:
    """A rule whose source or target never occurs is unrecoverable, so none does."""
    truth = read_declared_manifest(str(BENCHMARK / "manifest_declared.json"))
    assert len(truth.rules) == 6
    assert len(truth.stationary_labels) == 43

    named = {r.source for r in truth.rules} | {r.target for r in truth.rules}
    with open(BENCHMARK / "event_log.tsv", encoding="utf-8") as f:
        realized = {line.split("\t")[1] for line in islice(f, 1, None)}
    assert named <= realized

    axis = sorted(realized)
    mapped, off_axis = truth.typed_edges(axis)
    assert len(mapped) == 6
    assert off_axis == []
    assert {d.edge.src_scale for d in mapped} == {DAY_SCALE}
    assert {d.edge.lag for d in mapped} == {1, 7}


@pytest.mark.skipif(
    not BENCHMARK.is_dir(), reason="the dynamic_patterns benchmark is not in the tree"
)
def test_the_source_declared_as_a_maximum_holds_one_episode_a_day() -> None:
    """Its total is its longest on every cell, so the duration column reads both."""
    truth = read_declared_manifest(str(BENCHMARK / "manifest_declared.json"))
    maxima = [r for r in truth.rules if r.source_quantity == "max_duration"]
    assert [r.source for r in maxima] == ["sleep"]
    cells = [
        c
        for c in iter_resolved_cells(str(BENCHMARK / "manifest_resolved.jsonl"))
        if c.label == "sleep"
    ]
    assert len(cells) == 2520, "30 subjects over 84 days"
    assert max(c.realized_count for c in cells) == 1
    assert all(c.realized_total_minutes == c.realized_max_minutes for c in cells)
    # The rule fires on the same cells under either reading, so the duration
    # column carries the declared maximum and no third source quantity is needed.
    threshold = maxima[0].source_threshold
    above = [c for c in cells if c.realized_total_minutes > threshold]
    assert threshold == 390
    assert len(above) == len([c for c in cells if c.realized_max_minutes > threshold])
    assert len(above) == 1086


# ---------------------------------------------------------------------------
# The run's window, measured once and handed to every subject
# ---------------------------------------------------------------------------


def test_the_study_window_is_measured_from_starts_and_rounded_to_whole_epochs(
    tmp_path,
) -> None:
    """An episode overruns the window it began in, and its mark travels with
    the start, so the window is read off starts."""
    by_subject = read_subject_event_log(_two_subject_log(tmp_path / "log.tsv"))
    window = study_period_slots(
        by_subject,
        slot_minutes=1,
        branching=BENCHMARK_BRANCHING,
        origin=BENCHMARK_ORIGIN,
    )
    assert window % WEEK_SLOTS == 0
    last_start = max(
        int((e.start - BENCHMARK_ORIGIN).total_seconds() // 60)
        for events in by_subject.values()
        for e in events
    )
    assert last_start < window


def test_an_empty_log_covers_no_window() -> None:
    with pytest.raises(ValueError, match="covers no study period"):
        study_period_slots({})


def test_the_population_hands_every_subject_one_window(tmp_path) -> None:
    by_subject = read_subject_event_log(_two_subject_log(tmp_path / "log.tsv"))
    built = dict(
        (person, (data, counts))
        for person, data, counts in build_population(
            by_subject,
            ["sleep", "walk"],
            VERBATIM,
            slot_minutes=1,
            branching=BENCHMARK_BRANCHING,
            origin=BENCHMARK_ORIGIN,
        )
    )
    assert sorted(built) == sorted(by_subject)
    shapes = {data.get_X(1).shape for data, _ in built.values()}
    masks = {int(data.get_observed(1).sum()) for data, _ in built.values()}
    # One grid and one window, whoever the subject is.
    assert len(shapes) == 1
    assert len(masks) == 1


@pytest.mark.skipif(
    not BENCHMARK.is_dir(), reason="the dynamic_patterns benchmark is not in the tree"
)
def test_the_benchmark_window_leaves_every_coarse_epoch_whole() -> None:
    """A coarse epoch holding both watched and unwatched slots is refused, so
    the window has to land on a coarsest boundary."""
    from src.core.mask import check_ladder

    by_subject = read_subject_event_log(str(BENCHMARK / "event_log.tsv"))
    labels = sorted({e.label for v in by_subject.values() for e in v})
    seen = 0
    for _person, data, _counts in build_population(by_subject, labels):
        check_ladder(data.get_observed(1), data.hierarchy.parent_maps)
        observed = data.get_observed(1)
        assert observed[:120960].all()
        assert not observed[120960:].any()
        seen += 1
        if seen == 3:  # three subjects prove the window is shared
            break
    assert seen == 3
