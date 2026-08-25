"""
Tests for the frozen anchor set of the observed log.

The set is data, so the digest is pinned here.  Changing an anchor, its sign or
its reason changes the digest and fails this test, which is what freezing the
set before the first scoring run means.
"""

from __future__ import annotations

import json

import pytest

from src.core.typed_graph import CALENDAR, COUNT, LEVEL, OBSERVED, TypedEdge
from src.data.anchors import (
    ANCHOR_READ,
    ANCHOR_SOURCE,
    ASSERTABLE,
    MAX_LAG,
    NEGATIVE,
    NEGATIVE_SCALES,
    POSITIVE,
    POSITIVE_SCALES,
    STRUCTURAL,
    AnchorKey,
    anchor_set,
    key_of,
    score,
)
from src.data.label_axis import SCHEDULED_LABEL, collapse_labels, label_axis
from src.data.lifesnaps import read_lifesnaps
from src.data.lifesnaps_ladder import LADDER_UNITS

#: The digest of the set as declared.  A change to any anchor changes it.
FROZEN_DIGEST = "b55cbb587182610252920c278d18e30a2cbea27ac0177bf5e59d058f80193697"


@pytest.fixture(scope="module")
def axis():
    events, _ = read_lifesnaps()
    return label_axis(collapse_labels(events)[0])


@pytest.fixture(scope="module")
def anchors(axis):
    return anchor_set(axis)


def test_the_frozen_set_hashes_to_what_it_hashed_to(anchors):
    assert anchors.digest() == FROZEN_DIGEST


def test_the_same_relation_is_a_positive_and_a_negative_at_different_scales(anchors):
    positives = {a.key.scale for a in anchors.of_sign(POSITIVE)}
    negatives = {a.key.scale for a in anchors.of_sign(NEGATIVE, True)}
    assert positives == set(POSITIVE_SCALES)
    assert negatives == set(NEGATIVE_SCALES)
    assert not positives & negatives
    relations = {(a.key.source, a.key.target) for a in anchors.anchors if a.assertable}
    assert len(relations) == 1


def test_every_anchor_carries_a_reason(anchors):
    for anchor in anchors.anchors:
        assert len(anchor.justification) > 30
        assert anchor.standing in (ASSERTABLE, STRUCTURAL)
        assert anchor.sign in (POSITIVE, NEGATIVE)


def test_the_collapse_turns_most_of_one_negative_family_structural(anchors, axis):
    structural = [a for a in anchors.anchors if not a.assertable]
    assert len(structural) == len(axis.targets) * len(LADDER_UNITS) * MAX_LAG
    for anchor in structural:
        assert anchor.key.target == SCHEDULED_LABEL
        assert anchor.sign == NEGATIVE
    assert len(anchors.of_sign(POSITIVE, True)) == 6
    assert len(anchors.of_sign(NEGATIVE, True)) == 6


def test_an_anchor_names_the_vertex_the_read_column_became(anchors, axis):
    read = axis.axis_name(ANCHOR_READ)
    assert read != ANCHOR_READ
    for anchor in anchors.of_sign(POSITIVE):
        assert anchor.key.source == ANCHOR_SOURCE
        assert anchor.key.target == read


def test_an_accepted_edge_is_keyed_the_way_an_anchor_is(axis):
    edge = axis.scheduled_edge(scale=2, lag=1)
    key = key_of(edge, axis.target_index(ANCHOR_READ), axis)
    assert key == AnchorKey("hour", 1, SCHEDULED_LABEL, axis.axis_name(ANCHOR_READ))
    observed_edge = TypedEdge(
        src_kind=OBSERVED, src_scale=3, src_label=0, src_quantity=COUNT, lag=2
    )
    assert key_of(observed_edge, 1, axis) == AnchorKey(
        "day", 2, axis.targets[0], axis.targets[1]
    )


def test_a_run_that_accepts_nothing_scores_zero_on_both_signs(anchors):
    tallies = score([], anchors)
    assert tallies[POSITIVE].hits == 0
    assert tallies[POSITIVE].denominator == 6
    assert tallies[NEGATIVE].hits == 0
    assert tallies[NEGATIVE].structural == 72
    assert tallies[POSITIVE].rate == 0.0
    assert "structural" in str(tallies[NEGATIVE])


def test_a_run_that_accepts_every_positive_scores_all_of_them(anchors):
    keys = [a.key for a in anchors.of_sign(POSITIVE, True)]
    tallies = score(keys, anchors)
    assert tallies[POSITIVE].hits == 6
    assert tallies[POSITIVE].rate == 1.0
    assert tallies[POSITIVE].missed == ()
    assert tallies[NEGATIVE].hits == 0


def test_a_run_that_states_a_lagged_day_edge_is_caught(anchors, axis):
    read = axis.axis_name(ANCHOR_READ)
    tallies = score([AnchorKey("day", 1, SCHEDULED_LABEL, read)], anchors)
    assert tallies[NEGATIVE].hits == 1
    assert len(tallies[NEGATIVE].missed) == 5


def test_the_set_is_written_as_data_with_its_digest(anchors, tmp_path):
    path = tmp_path / "frozen" / "anchors.json"
    record = anchors.freeze(str(path))
    stored = json.loads(path.read_text())
    assert stored == record
    assert stored["digest"] == FROZEN_DIGEST
    assert stored["n_anchors"] == len(anchors) == 84
    assert stored["n_assertable"] == 12
    assert len(stored["anchors"]) == 84
    assert set(stored["anchors"][0]) == {
        "scale",
        "lag",
        "source",
        "target",
        "sign",
        "standing",
        "justification",
    }


def test_a_shorter_lag_grid_gives_a_different_set(axis, anchors):
    shorter = anchor_set(axis, max_lag=1)
    assert len(shorter) == len(anchors) // MAX_LAG
    assert shorter.digest() != anchors.digest()
    assert shorter.axis_digest == anchors.axis_digest


def test_an_anchor_key_reads_back_as_a_row(anchors):
    key = anchors.of_sign(POSITIVE)[0].key
    assert key.as_row() == (key.scale, key.lag, key.source, key.target)


def test_an_exogenous_edge_and_an_observed_edge_read_their_own_name_table(axis):
    calendar_edge = TypedEdge(
        src_kind=CALENDAR, src_scale=1, src_label=0, src_quantity=LEVEL, lag=0
    )
    assert key_of(calendar_edge, 0, axis).source == axis.exogenous[0]
    observed_edge = TypedEdge(
        src_kind=OBSERVED, src_scale=1, src_label=0, src_quantity=COUNT, lag=1
    )
    assert key_of(observed_edge, 0, axis).source == axis.targets[0]
