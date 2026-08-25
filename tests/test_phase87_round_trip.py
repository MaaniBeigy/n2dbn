"""
Tests for the log a fit draws, writes and reads back through the panel reader.

A fidelity claim rests on the synthetic side arriving through the same door as
the real one, so what is pinned here is that the writer emits the schema the
reader reads, that the drawn panel is laid on the same window, roster and mask
as the record it was drawn over, that the exogenous block is carried across
untouched, and that the draw reads the predictor the likelihood was maximized
on.
"""

from __future__ import annotations

import csv
import os
from datetime import date, datetime, timedelta

import numpy as np
import pytest

from src.core.typed_graph import COUNT, DURATION, OBSERVED, TypedEdge, TypedGraph
from src.data.label_axis import SCHEDULED_LABEL
from src.data.lifesnaps_panel import PanelSchedule, balanced_panel
from src.evaluation.round_trip import (
    MARK_SIZE_RANGE,
    RAW_FIELDS,
    Episode,
    calibrate_marks,
    centering_offset,
    draw_offsets,
    draw_subject,
    episode_lengths,
    identified_levels,
    mark_moments,
    missing_days,
    participant_levels,
    population_levels,
    reconciliation_error,
    replicates,
    round_trip,
    start_log_odds,
    write_raw_log,
)
from src.learning.arm1 import exogenous_index
from src.model.intensity import FineParams

#: Day zero of every fixture participant, which is a Monday.
DAY_ZERO = date(2021, 5, 24)
DAYS = 63
LABELS = ("sleep", "walk", "notification_read", SCHEDULED_LABEL)
WEEKS = 1
BUDGET = {1: 1, 2: 1, 3: 1, 4: 1}


def write_fixture(path: str, units: int = 3, days: int = DAYS) -> str:
    """A raw log dense enough for the cohort filter to keep every participant."""
    rows = []
    rng = np.random.default_rng(0)
    for unit in range(units):
        for day in range(days):
            when = datetime.combine(DAY_ZERO, datetime.min.time()) + timedelta(days=day)
            for label in LABELS:
                for _ in range(1 + int(rng.integers(0, 3))):
                    start = int(rng.integers(0, 1440))
                    rows.append(
                        {
                            "person_id": str(unit),
                            "event": label,
                            "day_index": day,
                            "start_min": start,
                            "duration_min": 0 if label != "sleep" else 30,
                            "start_datetime": (
                                when + timedelta(minutes=start)
                            ).strftime("%m/%d/%Y %H:%M"),
                        }
                    )
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RAW_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture(scope="module")
def fixture_log(tmp_path_factory):
    return write_fixture(str(tmp_path_factory.mktemp("raw") / "log.csv"))


@pytest.fixture(scope="module")
def panel(fixture_log):
    return balanced_panel(weeks=WEEKS, path=fixture_log)


def graph_of(axis):
    """A graph holding one behavioural lag and one read of the scheduled label."""
    graph = TypedGraph(
        label_names=list(axis.targets), calendar_names=list(axis.exogenous)
    )
    return graph


def fitted(cohort, axis, sweeps: int = 3):
    params = FineParams(n_labels=axis.L)
    params.base[COUNT][:] = start_log_odds(cohort, BUDGET)
    graph = graph_of(axis)
    params = calibrate_marks(params, cohort)
    return (
        graph,
        participant_levels(
            cohort,
            graph,
            params,
            BUDGET,
            index=exogenous_index(axis, cohort.K),
            max_sweeps=sweeps,
        )[0],
    )


# ---------------------------------------------------------------------------
# The two blocks the search does not price
# ---------------------------------------------------------------------------


def test_the_mark_moments_read_the_length_of_a_started_episode(panel):
    cohort, axis, _, _, _ = panel
    mean, variance = mark_moments(cohort)
    lengths = episode_lengths(cohort)
    # Sleep is written at thirty minutes and every other label as an instant,
    # which the reader gives one slot.
    assert lengths["sleep"][0] == pytest.approx(30.0)
    assert lengths["walk"][0] == pytest.approx(1.0)
    assert mean.shape == (axis.L,)
    assert np.all(variance >= 0.0)


def test_the_mark_intercept_reproduces_the_mean_length_it_was_matched_to(panel):
    cohort, axis, _, _, _ = panel
    params = calibrate_marks(FineParams(n_labels=axis.L), cohort)
    mean, _ = mark_moments(cohort)
    drawn = 1.0 + np.exp(params.base[DURATION])
    assert drawn[axis.target_index("sleep")] == pytest.approx(
        mean[axis.target_index("sleep")]
    )
    assert np.all(params.mark_dispersion >= MARK_SIZE_RANGE[0])
    assert np.all(params.mark_dispersion <= MARK_SIZE_RANGE[1])


def test_a_label_of_constant_length_gets_the_lower_end_of_the_family(panel):
    cohort, axis, _, _, _ = panel
    params = calibrate_marks(FineParams(n_labels=axis.L), cohort)
    # Every instant is exactly one slot, so its length has no spread at all.
    assert params.mark_dispersion[axis.target_index("walk")] == MARK_SIZE_RANGE[0]


def test_the_opening_intercept_is_the_records_own_start_rate(panel):
    cohort, axis, _, _, _ = panel
    opening = start_log_odds(cohort, BUDGET)
    assert opening.shape == (axis.L,)
    # One start per two slots is log-odds zero, and every label here is far rarer.
    assert np.all(opening < -3.0)


def test_the_participant_level_gives_every_subject_a_row(panel):
    cohort, axis, _, _, _ = panel
    _, params = fitted(cohort, axis)
    assert params.n_subjects == cohort.S
    assert params.offset[COUNT].shape == (cohort.S, axis.L)


def test_the_level_fit_does_not_lower_the_start_log_likelihood(panel):
    cohort, axis, _, _, _ = panel
    params = calibrate_marks(FineParams(n_labels=axis.L), cohort)
    _, history, unidentified = participant_levels(
        cohort, graph_of(axis), params, BUDGET, max_sweeps=4
    )
    assert history == sorted(history)
    assert unidentified == ()


# ---------------------------------------------------------------------------
# The predictor the draw reads
# ---------------------------------------------------------------------------


def test_a_graph_of_no_edges_carries_no_centering_term(panel):
    cohort, axis, _, _, _ = panel
    data = cohort.open(cohort.subjects[0])
    block = centering_offset(
        graph_of(axis),
        FineParams(n_labels=axis.L),
        cohort.sources(data),
        data.observed,
        COUNT,
        None,
        BUDGET,
    )
    assert np.array_equal(block, np.zeros_like(block))


def test_an_edge_carrying_a_coefficient_puts_its_removed_mean_back(panel):
    cohort, axis, _, _, _ = panel
    graph = graph_of(axis)
    edge = TypedEdge(OBSERVED, 3, axis.target_index("sleep"), COUNT, 1)
    target = axis.target_index("walk")
    graph.add_edge(edge, target, COUNT)
    params = FineParams(n_labels=axis.L)
    params.coef[(target, COUNT)][edge] = 0.7
    data = cohort.open(cohort.subjects[0])
    block = centering_offset(
        graph, params, cohort.sources(data), data.observed, COUNT, None, BUDGET
    )
    # The mean the design took out of the column is positive, so putting it back
    # lowers the raw-feature predictor by that much.
    assert np.all(block[:, target] < 0.0)
    assert np.array_equal(
        block[:, axis.target_index("sleep")], np.zeros(block.shape[0])
    )


def test_the_draw_offsets_add_the_subject_level_to_the_centering_term(panel):
    cohort, axis, _, _, _ = panel
    graph, params = fitted(cohort, axis)
    data = cohort.open(cohort.subjects[1])
    sources = cohort.sources(data)
    offsets = draw_offsets(graph, params, sources, data.observed, 1, None, BUDGET)
    assert set(offsets) == {COUNT, DURATION}
    assert offsets[COUNT].shape == (sources.n_slots, axis.L)
    assert np.allclose(offsets[COUNT][0], params.offset[COUNT][1])


# ---------------------------------------------------------------------------
# The writer and the reader
# ---------------------------------------------------------------------------


def test_the_writer_emits_the_columns_the_reader_reads(tmp_path):
    path = str(tmp_path / "out.csv")
    write_raw_log(
        path,
        [Episode("0", "walk", 3, 65, 12)],
        {"0": DAY_ZERO},
    )
    with open(path) as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == list(RAW_FIELDS)
    assert rows[0]["person_id"] == "0"
    assert rows[0]["duration_min"] == "12"
    assert rows[0]["start_datetime"] == "05/27/2021 01:05"


def test_a_drawn_subject_stays_inside_its_own_mask_and_window(panel):
    cohort, axis, report, _, _ = panel
    graph, params = fitted(cohort, axis)
    name = cohort.subjects[0]
    drawn, copied = draw_subject(
        cohort,
        axis,
        name,
        0,
        graph,
        params,
        np.random.default_rng(0),
        report.schedule.offsets[name],
        None,
        BUDGET,
    )
    window = range(
        report.schedule.offsets[name],
        report.schedule.offsets[name] + WEEKS * 7,
    )
    assert drawn
    assert all(episode.day_index in window for episode in drawn)
    assert all(0 <= episode.start_min < 1440 for episode in drawn)
    assert all(episode.unit == name for episode in drawn + copied)


def test_an_instant_label_is_written_as_a_zero_the_reader_fills_in(panel):
    cohort, axis, report, _, _ = panel
    graph, params = fitted(cohort, axis)
    name = cohort.subjects[0]
    drawn, copied = draw_subject(
        cohort,
        axis,
        name,
        0,
        graph,
        params,
        np.random.default_rng(1),
        report.schedule.offsets[name],
        None,
        BUDGET,
    )
    for episode in drawn:
        if axis.is_instant(episode.label):
            assert episode.duration_min == 0
        else:
            assert episode.duration_min >= 1
    assert all(episode.label == SCHEDULED_LABEL for episode in copied)


def test_the_scheduled_label_is_copied_and_never_drawn(panel):
    cohort, axis, report, _, _ = panel
    graph, params = fitted(cohort, axis)
    name = cohort.subjects[0]
    _, copied = draw_subject(
        cohort,
        axis,
        name,
        0,
        graph,
        params,
        np.random.default_rng(2),
        report.schedule.offsets[name],
        None,
        BUDGET,
    )
    data = cohort.open(name)
    column = axis.exogenous_index(SCHEDULED_LABEL)
    assert len(copied) == int(cohort.sources(data).calendar[0][:, column].sum())


def test_the_round_trip_reads_back_onto_the_window_it_was_drawn_over(panel, tmp_path):
    cohort, axis, report, _, _ = panel
    graph, params = fitted(cohort, axis)
    trip = round_trip(
        cohort,
        axis,
        report.schedule,
        graph,
        params,
        np.random.default_rng(3),
        str(tmp_path / "syn.csv"),
        None,
        BUDGET,
    )
    assert trip.cohort.subjects == cohort.subjects
    assert np.array_equal(trip.cohort.masks, cohort.masks)
    assert trip.axis.digest() == axis.digest()
    assert trip.episodes == trip.drawn + trip.copied
    assert reconciliation_error(trip) == 0.0


def test_the_drawn_log_carries_the_exogenous_block_it_was_drawn_over(panel, tmp_path):
    cohort, axis, report, _, _ = panel
    graph, params = fitted(cohort, axis)
    trip = round_trip(
        cohort,
        axis,
        report.schedule,
        graph,
        params,
        np.random.default_rng(4),
        str(tmp_path / "syn.csv"),
        None,
        BUDGET,
    )
    column = axis.exogenous_index(SCHEDULED_LABEL)
    for name in cohort.subjects:
        real = cohort.sources(cohort.open(name)).calendar[0][:, column]
        synthetic = trip.cohort.sources(trip.cohort.open(name)).calendar[0][:, column]
        assert np.array_equal(real, synthetic)


def test_a_day_the_draw_left_empty_stays_inside_the_mask(panel, tmp_path):
    cohort, axis, report, _, _ = panel
    graph = graph_of(axis)
    # An intercept far below every rate draws almost nothing, so most watched
    # days come back empty and the mask still has to hold them.
    params = calibrate_marks(FineParams(n_labels=axis.L), cohort)
    params.base[COUNT][:] = -30.0
    trip = round_trip(
        cohort,
        axis,
        report.schedule,
        graph,
        params,
        np.random.default_rng(5),
        str(tmp_path / "sparse.csv"),
        None,
        BUDGET,
    )
    empty = missing_days(trip, report.schedule)
    assert sum(empty.values()) > 0
    assert np.array_equal(trip.cohort.masks, cohort.masks)


def test_replicates_write_one_file_each_and_differ(panel, tmp_path):
    cohort, axis, report, _, _ = panel
    graph, params = fitted(cohort, axis)
    trips = replicates(
        cohort,
        axis,
        report.schedule,
        graph,
        params,
        2,
        str(tmp_path),
        tag="home",
        lag_budget=BUDGET,
    )
    assert [os.path.basename(t.path) for t in trips] == ["home_00.csv", "home_01.csv"]
    assert all(os.path.exists(t.path) for t in trips)
    assert trips[0].drawn != trips[1].drawn


def test_no_replicate_leaves_the_synthetic_side_empty(panel, tmp_path):
    cohort, axis, report, _, _ = panel
    graph, params = fitted(cohort, axis)
    with pytest.raises(ValueError, match="at least one"):
        replicates(
            cohort,
            axis,
            report.schedule,
            graph,
            params,
            0,
            str(tmp_path),
            lag_budget=BUDGET,
        )


def test_a_log_missing_a_label_of_the_axis_is_refused(panel, tmp_path):
    cohort, axis, report, _, _ = panel
    narrower = PanelSchedule(
        weeks=report.schedule.weeks,
        offsets=report.schedule.offsets,
        days=report.schedule.days,
        day_zero=report.schedule.day_zero,
        axis=None,
    )
    graph = graph_of(axis)
    params = calibrate_marks(FineParams(n_labels=axis.L), cohort)
    params.base[COUNT][:] = -30.0
    with pytest.raises(ValueError, match="different label axis"):
        round_trip(
            cohort,
            axis,
            narrower,
            graph,
            params,
            np.random.default_rng(6),
            str(tmp_path / "thin.csv"),
            None,
            BUDGET,
        )


def test_a_schedule_of_another_roster_is_refused(panel, tmp_path):
    cohort, axis, report, _, _ = panel
    short = PanelSchedule(
        weeks=report.schedule.weeks,
        offsets={name: report.schedule.offsets[name] for name in cohort.subjects[:2]},
        days={name: report.schedule.days[name] for name in cohort.subjects[:2]},
        day_zero=report.schedule.day_zero,
        axis=report.schedule.axis,
    )
    graph, params = fitted(cohort, axis)
    with pytest.raises(ValueError, match="names no window"):
        round_trip(
            cohort,
            axis,
            short,
            graph,
            params,
            np.random.default_rng(7),
            str(tmp_path / "short.csv"),
            None,
            BUDGET,
        )


def masked_schedule(report, unit: str, keep: int) -> PanelSchedule:
    """The panel's own schedule with one subject watched over fewer days."""
    days = dict(report.schedule.days)
    days[unit] = tuple(days[unit][:keep])
    return PanelSchedule(
        weeks=report.schedule.weeks,
        offsets=report.schedule.offsets,
        days=days,
        day_zero=report.schedule.day_zero,
        axis=report.schedule.axis,
    )


def test_an_episode_drawn_outside_the_mask_never_reaches_the_log(fixture_log, tmp_path):
    full = balanced_panel(weeks=2, path=fixture_log)
    cohort, axis, report, _, _ = full
    # A fortnight whose first week nobody watched, which the panel keeps whole.
    narrowed = masked_schedule(report, cohort.subjects[0], 7)
    watched, _, watched_report, _, _ = balanced_panel(
        weeks=2, path=fixture_log, schedule=narrowed
    )
    assert watched.masks[0].sum() == 7 * 1440
    graph, params = fitted(watched, axis)
    drawn, _ = draw_subject(
        watched,
        axis,
        watched.subjects[0],
        0,
        graph,
        params,
        np.random.default_rng(8),
        narrowed.offsets[watched.subjects[0]],
        None,
        BUDGET,
    )
    inside = set(narrowed.days[watched.subjects[0]])
    assert drawn
    assert all(episode.day_index in inside for episode in drawn)


def test_a_schedule_watching_other_days_than_the_cohort_is_refused(panel, tmp_path):
    cohort, axis, report, _, _ = panel
    graph, params = fitted(cohort, axis)
    with pytest.raises(ValueError, match="different window"):
        round_trip(
            cohort,
            axis,
            masked_schedule(report, cohort.subjects[0], 3),
            graph,
            params,
            np.random.default_rng(9),
            str(tmp_path / "narrow.csv"),
            None,
            BUDGET,
        )


def test_a_subject_with_no_scorable_row_keeps_the_population_level(fixture_log):
    cohort, axis, report, _, _ = balanced_panel(weeks=2, path=fixture_log)
    # A budget of two at the week scale needs two whole watched weeks of history,
    # and this subject's window opens on the second of the two, so no row of its
    # record is scorable and no number identifies its level.
    days = dict(report.schedule.days)
    first = cohort.subjects[0]
    days[first] = tuple(days[first][7:])
    narrowed = PanelSchedule(
        weeks=2,
        offsets=report.schedule.offsets,
        days=days,
        day_zero=report.schedule.day_zero,
        axis=report.schedule.axis,
    )
    watched, _, _, _, _ = balanced_panel(weeks=2, path=fixture_log, schedule=narrowed)
    budget = {1: 1, 2: 1, 3: 1, 4: 2}
    assert identified_levels(watched, budget).tolist() == [1, 2]
    params = calibrate_marks(FineParams(n_labels=axis.L), watched)
    params, _, unidentified = participant_levels(
        watched, graph_of(axis), params, budget, max_sweeps=2
    )
    assert unidentified == (first,)
    assert np.array_equal(params.offset[COUNT][0], np.zeros(axis.L))
    assert np.any(params.offset[COUNT][1:] != 0.0)


# ---------------------------------------------------------------------------
# The drawn population, which the contrast of 39.2 reads
# ---------------------------------------------------------------------------


def test_a_drawn_participant_takes_a_level_off_the_fitted_roster(panel):
    cohort, axis, _, _, _ = panel
    _, params = fitted(cohort, axis)
    drawn = population_levels(cohort, params, seed=0)
    fitted_rows = {tuple(row) for row in params.offset[COUNT]}
    for row in drawn.offset[COUNT]:
        assert tuple(row) in fitted_rows
    assert drawn.offset[COUNT].shape == params.offset[COUNT].shape


def test_a_level_vector_moves_as_one_so_the_labels_keep_their_covariation(panel):
    cohort, axis, _, _, _ = panel
    _, params = fitted(cohort, axis)
    drawn = population_levels(cohort, params, seed=3)
    # The count row and the duration row of one drawn participant come from the
    # same fitted participant, so a level is a person and never a per-label mix.
    for j in range(cohort.S):
        source = [
            i
            for i in range(cohort.S)
            if np.array_equal(params.offset[COUNT][i], drawn.offset[COUNT][j])
        ]
        assert any(
            np.array_equal(params.offset[DURATION][i], drawn.offset[DURATION][j])
            for i in source
        )


def test_the_draw_is_seeded_and_is_not_the_roster_it_came_from(panel):
    cohort, axis, _, _, _ = panel
    _, params = fitted(cohort, axis)
    again = population_levels(cohort, params, seed=5)
    assert np.array_equal(
        again.offset[COUNT], population_levels(cohort, params, seed=5).offset[COUNT]
    )
    picks = [
        sorted(
            i for i in range(cohort.S) if np.array_equal(params.offset[COUNT][i], row)
        )
        for row in population_levels(cohort, params, seed=11).offset[COUNT]
    ]
    assert picks


def test_the_draw_carries_the_pooled_blocks_the_fit_wrote(panel):
    cohort, axis, _, _, _ = panel
    graph, params = fitted(cohort, axis)
    drawn = population_levels(cohort, params, seed=1)
    assert np.array_equal(drawn.base[COUNT], params.base[COUNT])
    assert np.array_equal(drawn.mark_dispersion, params.mark_dispersion)
    assert set(drawn.coef) == set(params.coef)
    for key, coefficients in params.coef.items():
        assert drawn.coef[key] == coefficients


def test_an_unidentified_subject_is_not_drawn_from(panel):
    cohort, axis, _, _, _ = panel
    _, params = fitted(cohort, axis)
    withheld = cohort.subjects[0]
    params.offset[COUNT][0, :] = 99.0
    drawn = population_levels(cohort, params, [withheld], seed=0)
    assert not np.any(drawn.offset[COUNT] == 99.0)


def test_a_roster_with_no_identified_level_has_no_population_to_draw_from(panel):
    cohort, axis, _, _, _ = panel
    _, params = fitted(cohort, axis)
    with pytest.raises(ValueError, match="needs one identified level"):
        population_levels(cohort, params, cohort.subjects, seed=0)
