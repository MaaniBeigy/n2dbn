"""
Tests for the observational arm's fit and the columns its rungs read.

What is pinned here is the candidate space each ablation arm offers, that an
accepted edge reads back as the key the frozen anchors are keyed on, that a
lagged read never crosses a participant boundary, and that the sham label
reproduces the marginal it was matched to while being caused by nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.cohort import Cohort
from src.core.scales import ScaleHierarchy
from src.core.typed_graph import CALENDAR, COUNT, LEVEL, OBSERVED, TypedEdge
from src.core.variables import MultiScaleData
from src.data.anchors import AnchorKey
from src.data.label_axis import SCHEDULED_LABEL, LabelAxis
from src.inference.bound import segment_sum
from src.learning.arm1 import (
    MAX_LAG,
    SEARCH_SCALES,
    SHAM_LABEL,
    Arm1Fit,
    candidate_pool,
    edge_series,
    environments,
    epoch_series,
    exogenous_index,
    fit_arm1,
    future_of,
    sham_cohort,
    touches_sham,
)
from src.learning.calendar_arms import ABSENT, EVERY, HOME

HOURS, DAYS, WEEKS = 24, 28, 4
T = HOURS * DAYS
TARGETS = ("survey_response", "walk")
EXOGENOUS = ("hour_index", SCHEDULED_LABEL)
DAY_OF = np.repeat(np.arange(DAYS), HOURS).astype(np.int32)
WEEK_OF = np.repeat(np.arange(WEEKS), 7).astype(np.int32)
UNITS = ("minute", "hour", "day", "week")
SCALES = (2, 3)


def axis() -> LabelAxis:
    return LabelAxis(
        targets=TARGETS,
        exogenous=EXOGENOUS,
        grouped={name: name for name in TARGETS},
    )


def hierarchy() -> ScaleHierarchy:
    return ScaleHierarchy(K=3, H=[T, DAYS, WEEKS], parent_maps=[DAY_OF, WEEK_OF, None])


def blocks(seed: int):
    """One subject's exogenous block, whose receipt column is its own."""
    rng = np.random.default_rng(100 + seed)
    hour = np.tile(np.arange(24, dtype=float), DAYS)
    received = (rng.random(T) < 0.2).astype(float)
    finest = np.column_stack([hour, received])
    day = np.column_stack([np.zeros(DAYS), segment_sum(received, DAY_OF, DAYS)])
    week = np.column_stack([np.zeros(WEEKS), segment_sum(day[:, 1], WEEK_OF, WEEKS)])
    return [finest, day, week], received


def subject(seed: int) -> MultiScaleData:
    rng = np.random.default_rng(seed)
    context, received = blocks(seed)
    X1 = np.zeros((T, len(TARGETS)), dtype=np.int64)
    lagged = np.concatenate([[0.0], received[:-1]])
    odds = -3.0 + 2.2 * lagged
    X1[:, 0] = rng.random(T) < 1.0 / (1.0 + np.exp(-odds))
    X1[:, 1] = rng.random(T) < 0.1
    Y1 = (X1 > 0).astype(np.int64)
    day_x = segment_sum(X1.astype(float), DAY_OF, DAYS).astype(np.int64)
    day_y = segment_sum(Y1.astype(float), DAY_OF, DAYS).astype(np.int64)
    return MultiScaleData(
        hierarchy=hierarchy(),
        label_names=list(TARGETS),
        X=[
            X1,
            day_x,
            segment_sum(day_x.astype(float), WEEK_OF, WEEKS).astype(np.int64),
        ],
        Y=[
            Y1,
            day_y,
            segment_sum(day_y.astype(float), WEEK_OF, WEEKS).astype(np.int64),
        ],
        C=context,
    )


def cohort(n: int = 5) -> Cohort:
    names = tuple(f"p{i}" for i in range(n))
    seeds = dict(zip(names, range(n)))
    return Cohort(
        subjects=names,
        hierarchy=hierarchy(),
        label_names=list(TARGETS),
        reader=lambda name: subject(seeds[name]),
        masks=np.ones((n, T), dtype=bool),
        subject_calendar=True,
    )


# ---------------------------------------------------------------------------
# The candidate space
# ---------------------------------------------------------------------------


def test_the_pool_holds_every_family_the_search_may_propose():
    pool = candidate_pool(axis(), HOME, SCALES, max_lag=2)
    observed = [e for e in pool if not e.is_exogenous]
    assert len(observed) == len(SCALES) * len(TARGETS) * 2 * 2
    scheduled = [
        e
        for e in pool
        if e.is_exogenous and e.src_label == EXOGENOUS.index(SCHEDULED_LABEL)
    ]
    # An exogenous source may be read at the epoch being predicted.
    assert {e.lag for e in scheduled} == {0, 1, 2}
    assert all(e.src_quantity == LEVEL for e in scheduled)
    clock = [e for e in pool if e.is_exogenous and e.src_label == 0]
    assert len(clock) == 1
    assert clock[0].src_scale == 2 and clock[0].lag == 0


def test_the_arms_offer_what_they_declare():
    absent = candidate_pool(axis(), ABSENT, SCALES, max_lag=1)
    home = candidate_pool(axis(), HOME, SCALES, max_lag=1)
    every = candidate_pool(axis(), EVERY, SCALES, max_lag=1)
    assert len(absent) < len(home) < len(every)
    assert not [e for e in absent if e.is_exogenous and e.src_label == 0]


def test_a_pool_over_an_axis_with_no_scheduled_label_leaves_it_out():
    bare = LabelAxis(
        targets=TARGETS, exogenous=("hour_index",), grouped={n: n for n in TARGETS}
    )
    pool = candidate_pool(bare, HOME, SCALES, max_lag=1)
    assert all(e.src_label == 0 for e in pool if e.is_exogenous)


def test_the_exogenous_index_names_every_column_at_every_scale():
    index = exogenous_index(axis(), 3)
    assert index.widths == {1: 2, 2: 2, 3: 2}
    assert index.column_name(2, 1) == SCHEDULED_LABEL
    with pytest.raises(ValueError, match="no calendar column"):
        index.column_name(2, 5)


def test_the_declared_scales_and_lags_are_what_the_anchors_declare():
    assert SEARCH_SCALES == (1, 2, 3, 4)
    assert MAX_LAG == 3


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted():
    return fit_arm1(cohort(), axis(), HOME, scales=SCALES, max_lag=2, price=2.0)


def test_the_fit_buys_the_relation_the_world_holds(fitted):
    names = [(e.src_kind, e.src_label, e.lag, t) for e, t in fitted.accepted]
    assert names
    # Receipt at the hour scale drives the survey response, which is target 0.
    assert any(
        kind == CALENDAR and label == 1 and dst == 0 for kind, label, _, dst in names
    )
    assert fitted.price == 2.0
    assert fitted.arm == HOME


def test_an_accepted_edge_reads_back_as_an_anchor_key(fitted):
    keys = fitted.keys(axis(), UNITS)
    assert len(keys) == len(fitted.accepted)
    assert all(isinstance(key, AnchorKey) for key in keys)
    assert {key.scale for key in keys} <= {"hour", "day"}
    assert any(key.source == SCHEDULED_LABEL for key in keys)


def test_the_fit_reports_its_strata_and_its_objective(fitted):
    tally = fitted.by_stratum()
    assert sum(tally.values()) == len(fitted.accepted)
    assert np.isfinite(fitted.objective)
    assert isinstance(fitted.rejected(), dict)


def test_a_measured_price_comes_off_the_cohorts_own_null():
    fit = fit_arm1(cohort(3), axis(), HOME, scales=SCALES, max_lag=1, quantile=0.9)
    assert fit.price >= 0.0
    assert fit.null_max >= fit.null_median
    assert len(fit.pool) == len(candidate_pool(axis(), HOME, SCALES, max_lag=1))


def test_a_fit_with_an_empty_report_carries_an_empty_key_list():
    fit = Arm1Fit(
        graph=None,
        params=None,
        pool=[],
        reports={},
        price=1.0,
        null_median=0.0,
        null_max=0.0,
        arm=HOME,
    )
    assert fit.keys(axis(), UNITS) == []
    assert fit.by_stratum() == {}
    assert fit.rejected() == {}
    assert fit.objective == 0


# ---------------------------------------------------------------------------
# The columns the rungs read
# ---------------------------------------------------------------------------


def test_a_series_is_laid_end_to_end_over_the_roster():
    people = cohort(3)
    values, who = epoch_series(people, 2, 0)
    assert values.size == 3 * DAYS
    assert np.array_equal(np.unique(who), np.arange(3))
    exog, _ = epoch_series(people, 2, 1, exogenous=True)
    assert exog.size == values.size
    assert exog.sum() > 0


def test_a_lagged_read_never_crosses_a_participant_boundary():
    people = cohort(3)
    edge = TypedEdge(OBSERVED, 2, 0, COUNT, 1)
    response, parent, who = edge_series(people, edge, 0)
    raw, _ = epoch_series(people, 2, 0)
    for unit in range(3):
        rows = np.nonzero(who == unit)[0]
        assert parent[rows[0]] == 0.0
        assert np.array_equal(parent[rows[1:]], raw[rows[:-1]])
    assert np.array_equal(response, raw)


def test_an_exogenous_source_at_lag_zero_is_read_where_it_stands():
    people = cohort(2)
    edge = TypedEdge(CALENDAR, 2, 1, LEVEL, 0)
    _, parent, _ = edge_series(people, edge, 0)
    raw, _ = epoch_series(people, 2, 1, exogenous=True)
    assert np.array_equal(parent, raw)


def test_the_future_of_a_column_stops_at_the_end_of_each_subject():
    who = np.repeat(np.arange(2), 4)
    values = np.arange(8.0)
    ahead = future_of(values, who)
    assert np.array_equal(ahead, [1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 7.0, 0.0])
    assert np.all(future_of(values, who, lead=9) == 0.0)


def test_the_environments_are_the_wave_and_the_half_of_the_window():
    people = cohort(2)
    waves = {"p0": "spring", "p1": "autumn"}
    env = environments(people, waves, 2)
    assert env.size == 2 * DAYS
    assert set(np.unique(env)) == {0, 1, 2, 3}
    # A participant is never an environment on its own, so both halves of one
    # participant carry different labels.
    assert env[0] != env[DAYS - 1]
    assert set(np.unique(environments(people, {}, 2))) == {0, 1}


# ---------------------------------------------------------------------------
# R2, the sham label
# ---------------------------------------------------------------------------


def test_the_sham_label_reproduces_the_marginal_it_was_matched_to():
    people = cohort(3)
    wider, wider_axis = sham_cohort(people, axis(), reference="walk")
    assert wider_axis.targets == TARGETS + (SHAM_LABEL,)
    assert wider.L == len(TARGETS) + 1
    for name in wider.subjects:
        data = wider.open(name)
        reference = data.X[0][:, TARGETS.index("walk")]
        sham = data.X[0][:, -1]
        assert np.array_equal(np.sort(sham), np.sort(reference))
        # The counts still add up the ladder, so the extra column is a label
        # like any other.
        folded = np.zeros_like(data.X[1])
        np.add.at(folded, DAY_OF, data.X[0])
        assert np.array_equal(folded, data.X[1])


def test_a_sham_column_is_ordered_differently_from_the_label_it_copies():
    people = cohort(2)
    wider, _ = sham_cohort(people, axis(), reference="walk")
    data = wider.open(wider.subjects[0])
    assert not np.array_equal(data.X[0][:, -1], data.X[0][:, TARGETS.index("walk")])


def test_every_edge_touching_the_sham_is_counted_against_what_was_offered():
    people = cohort(3)
    wider, wider_axis = sham_cohort(people, axis(), reference="walk")
    fit = fit_arm1(wider, wider_axis, HOME, scales=SCALES, max_lag=1, price=2.0)
    hits, offered = touches_sham(fit, wider_axis)
    assert 0 <= hits <= len(fit.accepted)
    assert offered > len(fit.pool)
    sham = wider_axis.target_index(SHAM_LABEL)
    counted = sum(
        1
        for edge, target in fit.accepted
        if target == sham or (not edge.is_exogenous and edge.src_label == sham)
    )
    assert hits == counted
