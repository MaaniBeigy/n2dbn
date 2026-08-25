"""
tests/test_phase64_fidelity.py
==============================
The posterior-predictive check: cells, replicates, p-values and the verdict.

Every case here runs on a fourteen-day ladder of hourly slots over two labels,
so a whole check of forty replicates costs a fraction of a second and the cells
can be read by hand.  What is asserted is that a cell holds exactly the episodes
that began inside it, that a replicate is drawn on the fit's own grid and
reconciles, that the p-value counts the replicates at or beyond the observation,
and that a model drawing at the wrong rate is named.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from src.core.calendar import SlotCalendar
from src.core.durations import aggregate
from src.core.features import (
    CalendarNode,
    calendar_blocks,
    horizon_trend,
    hour_of_day,
    weekend,
)
from src.core.typed_graph import CALENDAR, COUNT, DURATION, LEVEL, TypedEdge, TypedGraph
from src.evaluation.fidelity import (
    EPISODES,
    MIN_EPISODES,
    CellGrid,
    CellKey,
    _bin_index,
    _bin_names,
    _node_cells,
    check_reconciles,
    posterior_predictive,
    replicate,
)
from src.model.intensity import (
    FineParams,
    ancestor_maps,
    empty_sources,
    running_mask,
    sample,
)

LABELS = ("walk", "rest")
N_DAYS = 14
T_SLOTS = N_DAYS * 24
PARENT_MAPS = [
    np.repeat(np.arange(N_DAYS), 24),
    np.repeat(np.arange(N_DAYS // 7), 7),
]
# 2026-02-16 is a Monday, so a day index of 5 is a Saturday.
ORIGIN = datetime(2026, 2, 16)
SATURDAY_AFTERNOON = 5 * 24 + 15


def _calendar() -> SlotCalendar:
    return SlotCalendar(origin=ORIGIN, slot=timedelta(minutes=60), branching=(24, 7))


def _sources(blocks=None):
    """Empty sources on the shared ladder, with whatever calendar columns are given."""
    anc = ancestor_maps(PARENT_MAPS, T_SLOTS)
    if blocks is None:
        blocks = [
            np.zeros((T_SLOTS, 0)),
            np.zeros((N_DAYS, 0)),
            np.zeros((N_DAYS // 7, 0)),
        ]
    return empty_sources(anc, blocks, len(LABELS))


def _params(rate: float = -2.0, mean_length: float = 2.0) -> FineParams:
    params = FineParams(n_labels=len(LABELS))
    params.base[COUNT][:] = rate
    params.base[DURATION][:] = np.log(mean_length)
    params.mark_dispersion = np.full(len(LABELS), 4.0)
    return params


def _graph(calendar_names=()) -> TypedGraph:
    return TypedGraph(label_names=list(LABELS), calendar_names=list(calendar_names))


def _log(episodes):
    """Finest arrays holding the given (slot, label, length) episodes."""
    X1 = np.zeros((T_SLOTS, len(LABELS)), dtype=np.int64)
    Y1 = np.zeros((T_SLOTS, len(LABELS)), dtype=np.int64)
    for slot, ell, length in episodes:
        X1[slot, ell] += 1
        Y1[slot, ell] += length
    return X1, Y1


def _laid_out(X1, Y1, blocks=None):
    """The arrays, their ladder, and a grid over them."""
    sources = _sources(blocks)
    sources.counts = aggregate(X1, PARENT_MAPS)
    sources.durations = aggregate(Y1, PARENT_MAPS)
    return sources


def _values(grid, X1, Y1, sources):
    return dict(
        zip(grid.keys, grid.summarize(X1, Y1, sources.counts, sources.durations))
    )


def _drawn(seed: int, rate: float = -2.0):
    """One log drawn from the model, with the sources it filled."""
    sources = _sources()
    params = _params(rate)
    X1, Y1 = sample(params, _graph(), sources, np.random.default_rng(seed))
    return X1, Y1, sources, params


def _cell(report, key):
    """One cell of a report, by the key that names it."""
    return next(c for c in report.cells if c.key == key)


def _lopsided(rate: float, seed: int, replicates: int):
    """A record holding forty walk episodes and no rest, against a model drawing both."""
    X1, Y1 = _log([(8 * i, 0, 2) for i in range(40)])
    sources = _laid_out(X1, Y1)
    grid = CellGrid.build(_calendar(), sources, LABELS)
    return posterior_predictive(
        X1,
        Y1,
        _params(rate),
        _graph(),
        sources,
        grid,
        np.random.default_rng(seed),
        replicates,
    )


# ---------------------------------------------------------------------------
# The cells
# ---------------------------------------------------------------------------


def test_every_key_names_one_value_and_no_key_repeats() -> None:
    X1, Y1, sources, _ = _drawn(0)
    grid = CellGrid.build(_calendar(), sources, LABELS)
    values = grid.summarize(X1, Y1, sources.counts, sources.durations)
    assert len(grid.keys) == values.size
    assert len(set(grid.keys)) == len(grid.keys)
    assert grid.support.shape == (values.size,)


def test_the_families_are_the_splits_the_study_names() -> None:
    X1, Y1, sources, _ = _drawn(1)
    grid = CellGrid.build(_calendar(), sources, LABELS)
    families = {f.name: f for f in grid.families}
    assert sorted(families) == ["hour_of_day", "total", "week_index", "weekend"]
    assert families["hour_of_day"].cells == (
        "[0,7)",
        "[7,10)",
        "[10,12)",
        "[12,14)",
        "[14,17)",
        "[17,21)",
        "[21,24)",
    )
    assert families["weekend"].cells == ("0", "1")
    assert families["week_index"].cells == ("0", "1")
    # Each node sits at the coarsest scale its value is constant across.
    assert (families["hour_of_day"].scale, families["weekend"].scale) == (1, 2)
    assert families["week_index"].scale == 3
    assert grid.day_scale == 2


def test_the_total_cell_holds_the_whole_log_at_every_scale() -> None:
    X1, Y1 = _log([(3, 0, 4), (200, 0, 9), (57, 1, 2)])
    sources = _laid_out(X1, Y1)
    grid = CellGrid.build(_calendar(), sources, LABELS)
    values = _values(grid, X1, Y1, sources)
    for k in (1, 2, 3):
        for ell, label in enumerate(LABELS):
            assert values[CellKey("total", f"k{k}", label, COUNT)] == X1[:, ell].sum()
            assert (
                values[CellKey("total", f"k{k}", label, DURATION)] == Y1[:, ell].sum()
            )


def test_a_calendar_cell_holds_the_episodes_that_began_in_it() -> None:
    X1, Y1 = _log([(SATURDAY_AFTERNOON, 0, 3)])
    sources = _laid_out(X1, Y1)
    grid = CellGrid.build(_calendar(), sources, LABELS)
    values = _values(grid, X1, Y1, sources)

    assert values[CellKey("weekend", "1", "walk", COUNT)] == 1
    assert values[CellKey("weekend", "0", "walk", COUNT)] == 0
    assert values[CellKey("hour_of_day", "[14,17)", "walk", COUNT)] == 1
    assert values[CellKey("hour_of_day", "[14,17)", "walk", DURATION)] == 3
    assert values[CellKey("hour_of_day", "[12,14)", "walk", COUNT)] == 0
    assert values[CellKey("week_index", "0", "walk", COUNT)] == 1
    assert values[CellKey("week_index", "1", "walk", COUNT)] == 0
    assert values[CellKey("weekend", "1", "rest", COUNT)] == 0


def test_the_length_bins_count_the_episodes_of_each_label() -> None:
    X1, Y1 = _log([(3, 0, 2), (40, 0, 7), (80, 0, 200), (57, 1, 2)])
    sources = _laid_out(X1, Y1)
    grid = CellGrid.build(_calendar(), sources, LABELS)
    values = _values(grid, X1, Y1, sources)

    assert values[CellKey("episode_length", "[1,5)", "walk", EPISODES)] == 1
    assert values[CellKey("episode_length", "[5,15)", "walk", EPISODES)] == 1
    assert values[CellKey("episode_length", "[120,240)", "walk", EPISODES)] == 1
    assert values[CellKey("episode_length", "[240,inf)", "walk", EPISODES)] == 0
    assert values[CellKey("episode_length", "[1,5)", "rest", EPISODES)] == 1


def test_the_per_day_bins_count_the_days_holding_each_number_of_episodes() -> None:
    X1, Y1 = _log([(3, 0, 2), (5, 0, 1), (40, 0, 1)])
    sources = _laid_out(X1, Y1)
    grid = CellGrid.build(_calendar(), sources, LABELS)
    values = _values(grid, X1, Y1, sources)

    # Day 0 holds two walking episodes, day 1 holds one, the other twelve none.
    assert values[CellKey("episodes_per_day", "[2,3)", "walk", "days")] == 1
    assert values[CellKey("episodes_per_day", "[1,2)", "walk", "days")] == 1
    assert values[CellKey("episodes_per_day", "[0,1)", "walk", "days")] == 12
    assert values[CellKey("episodes_per_day", "[0,1)", "rest", "days")] == N_DAYS


def test_two_starts_in_one_slot_are_two_episodes_and_one_recorded_length() -> None:
    """The collision the finest scale cannot hold apart, reported as it stands."""
    X1, Y1 = _log([(3, 0, 20), (3, 0, 30)])
    sources = _laid_out(X1, Y1)
    grid = CellGrid.build(_calendar(), sources, LABELS)
    values = _values(grid, X1, Y1, sources)

    assert values[CellKey("total", "k1", "walk", COUNT)] == 2
    assert values[CellKey("episode_length", "[30,60)", "walk", EPISODES)] == 1
    assert values[CellKey("episode_length", "[15,30)", "walk", EPISODES)] == 0


def test_an_unobserved_slot_reaches_no_cell() -> None:
    X1, Y1 = _log([(SATURDAY_AFTERNOON, 0, 3)])
    sources = _laid_out(X1, Y1)
    watched = np.ones(T_SLOTS, dtype=bool)
    watched[5 * 24 : 6 * 24] = False
    grid = CellGrid.build(_calendar(), sources, LABELS, observed=watched)
    values = _values(grid, X1, Y1, sources)

    assert values[CellKey("total", "k1", "walk", COUNT)] == 0
    assert values[CellKey("weekend", "1", "walk", COUNT)] == 0
    assert values[CellKey("hour_of_day", "[14,17)", "walk", COUNT)] == 0
    assert values[CellKey("episode_length", "[1,5)", "walk", EPISODES)] == 0
    # The unobserved day leaves the per-day family as well.
    assert values[CellKey("episodes_per_day", "[0,1)", "walk", "days")] == N_DAYS - 1


def test_an_unobserved_week_leaves_the_coarse_cells_too() -> None:
    X1, Y1 = _log([(8 * 24 + 3, 0, 2)])
    sources = _laid_out(X1, Y1)
    watched = np.ones(T_SLOTS, dtype=bool)
    watched[7 * 24 :] = False
    grid = CellGrid.build(_calendar(), sources, LABELS, observed=watched)
    values = _values(grid, X1, Y1, sources)

    assert values[CellKey("week_index", "1", "walk", COUNT)] == 0
    assert values[CellKey("total", "k3", "walk", COUNT)] == 0
    assert values[CellKey("episodes_per_day", "[0,1)", "walk", "days")] == 7


# ---------------------------------------------------------------------------
# What the grid refuses
# ---------------------------------------------------------------------------


def test_a_label_axis_of_another_width_is_refused() -> None:
    sources = _sources()
    with pytest.raises(ValueError, match="another label"):
        CellGrid.build(_calendar(), sources, ["walk"])


def test_a_non_boolean_mask_is_refused() -> None:
    sources = _sources()
    with pytest.raises(ValueError, match="boolean array"):
        CellGrid.build(_calendar(), sources, LABELS, observed=np.ones(T_SLOTS, int))


def test_a_mask_of_another_length_is_refused() -> None:
    sources = _sources()
    with pytest.raises(ValueError, match="one flag per finest slot"):
        CellGrid.build(_calendar(), sources, LABELS, observed=np.ones(7, dtype=bool))


def test_a_ladder_with_no_day_scale_is_refused() -> None:
    calendar = SlotCalendar(origin=ORIGIN, slot=timedelta(minutes=60), branching=(5,))
    anc = ancestor_maps([np.repeat(np.arange(4), 5)], 20)
    sources = empty_sources(anc, [np.zeros((20, 0)), np.zeros((4, 0))], len(LABELS))
    with pytest.raises(ValueError, match="spans one day"):
        CellGrid.build(calendar, sources, LABELS)


def test_a_value_under_the_first_bin_edge_belongs_to_no_bin() -> None:
    with pytest.raises(ValueError, match="below the first bin edge"):
        _bin_index(np.array([0, 4]), (1, 5, 15))


def test_the_last_bin_is_open_above() -> None:
    assert _bin_names((1, 5, 15)) == ("[1,5)", "[5,15)", "[15,inf)")
    assert list(_bin_index(np.array([1, 5, 99]), (1, 5, 15))) == [0, 1, 2]


def test_a_node_reading_past_its_top_bin_edge_is_refused() -> None:
    node = hour_of_day(edges=(0, 7, 10, 12))
    with pytest.raises(ValueError, match="top bin edge"):
        _node_cells(node, np.array([0, 9, 23]))


def test_a_node_reading_an_undeclared_level_is_refused() -> None:
    node = CalendarNode(
        name="phase", value=lambda w: 0, encoding="onehot", levels=(0, 1)
    )
    with pytest.raises(ValueError, match="declared levels"):
        _node_cells(node, np.array([0, 1, 2]))


def test_an_ordinal_node_makes_one_cell_per_value_it_takes() -> None:
    node = horizon_trend(ORIGIN, "week")
    groups, cells = _node_cells(node, np.array([0, 0, 2, 5]))
    assert cells == ("0", "2", "5")
    assert list(groups) == [0, 0, 1, 2]


def test_a_one_hot_node_keeps_the_level_the_design_drops() -> None:
    groups, cells = _node_cells(weekend(), np.array([0, 1, 0]))
    assert cells == ("0", "1")
    assert list(groups) == [0, 1, 0]


# ---------------------------------------------------------------------------
# The identity every replicate satisfies
# ---------------------------------------------------------------------------


def test_a_drawn_replicate_reconciles_and_holds_no_same_label_overlap() -> None:
    sources = _sources()
    params = _params(rate=-1.5, mean_length=6.0)
    X1, Y1, filled = replicate(params, _graph(), sources, np.random.default_rng(4))
    assert X1.sum() > 20
    assert not ((X1 > 0) & running_mask(X1, Y1)).any()
    for k, expected in enumerate(aggregate(X1, PARENT_MAPS)):
        assert np.array_equal(filled.counts[k], expected)


def test_a_replicate_leaves_the_arrays_the_fit_read_alone() -> None:
    X1, Y1, sources, params = _drawn(5)
    before = [c.copy() for c in sources.counts]
    replicate(params, _graph(), sources, np.random.default_rng(6))
    for stored, expected in zip(sources.counts, before):
        assert np.array_equal(stored, expected)


def test_a_coarse_count_that_is_not_the_sum_is_refused() -> None:
    X1, Y1 = _log([(3, 0, 2)])
    sources = _laid_out(X1, Y1)
    sources.counts[1][0, 0] += 1
    with pytest.raises(ValueError, match="scale 2 count array"):
        check_reconciles(X1, Y1, sources.counts, sources.durations, sources.ancestors)


def test_a_coarse_duration_that_is_not_the_sum_is_refused() -> None:
    X1, Y1 = _log([(3, 0, 2)])
    sources = _laid_out(X1, Y1)
    sources.durations[2][0, 0] -= 1
    with pytest.raises(ValueError, match="scale 3 duration array"):
        check_reconciles(X1, Y1, sources.counts, sources.durations, sources.ancestors)


def test_the_observation_has_to_reconcile_before_anything_is_drawn() -> None:
    X1, Y1, sources, params = _drawn(7)
    sources.counts[1][0, 0] += 5
    grid = CellGrid.build(_calendar(), sources, LABELS)
    with pytest.raises(ValueError, match="disagrees with the sum"):
        posterior_predictive(
            X1, Y1, params, _graph(), sources, grid, np.random.default_rng(8), 3
        )


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------


def test_a_replicate_reads_the_calendar_the_fit_read() -> None:
    """A coefficient on one hour bin puts the replicate's episodes in that cell."""
    calendar = _calendar()
    nodes = [hour_of_day(), weekend(), horizon_trend(ORIGIN, "week")]
    blocks = calendar_blocks(calendar, nodes, [T_SLOTS, N_DAYS, N_DAYS // 7])
    sources = _sources(blocks.blocks)
    names = [name for k in sorted(blocks.names) for name in blocks.names[k]]
    graph = _graph(names)
    params = _params(rate=-6.0)
    # Offset 3 of the scale-1 block is the hour_of_day bin [14,17).
    edge = TypedEdge(CALENDAR, 1, 3, LEVEL, 0)
    graph.add_edge(edge, 0, COUNT)
    params.coef[(0, COUNT)][edge] = 6.0

    X1, Y1, filled = replicate(params, graph, sources, np.random.default_rng(3))
    grid = CellGrid.build(calendar, sources, LABELS)
    values = _values(grid, X1, Y1, filled)
    inside = values[CellKey("hour_of_day", "[14,17)", "walk", COUNT)]
    assert inside >= 10
    assert inside > 3 * (values[CellKey("total", "k1", "walk", COUNT)] - inside)


def test_a_silent_model_puts_the_observation_at_every_replicate() -> None:
    """
    Every value is zero on both sides, which fixes what the two p-values mean.

    The observation is at the replicates and never beyond them, so the fraction
    at or beyond is one and the mid-p, which splits the ties, is one half.
    """
    sources = _sources()
    params = _params(rate=-40.0)
    X1, Y1 = _log([])
    grid = CellGrid.build(_calendar(), sources, LABELS)
    report = posterior_predictive(
        X1, Y1, params, _graph(), sources, grid, np.random.default_rng(9), 5
    )
    assert all(
        cell.observed == 0.0 for cell in report.cells if cell.key.cell != "[0,1)"
    )
    for cell in report.cells:
        assert cell.p_value == 1.0
        assert cell.p_mid == 0.5
        assert not cell.outside

    # The only cells carrying anything are the days holding no episode.
    summary = report.calibration()
    assert summary.n_cells == len(LABELS)
    assert summary.outside_share == 0.0
    assert summary.n_sparse == len(report.cells) - len(LABELS)
    assert report.misfits() == []

    empty = report.calibration(min_episodes=float(T_SLOTS))
    assert empty.n_cells == 0
    assert np.isnan(empty.outside_share)

    # The days holding no episode are the same number in every member, so no
    # family holds a comparison and the verdict has nothing to name.
    call = report.verdict()
    assert call.worst is None
    assert np.isnan(call.p_value)
    assert not call.broken
    assert "the verdict is empty" in str(call)
    assert report.verdict(min_episodes=float(T_SLOTS)).n_families == 0


def test_a_check_against_the_model_that_drew_the_log_names_few_misfits() -> None:
    X1, Y1, sources, params = _drawn(11, rate=-1.0)
    grid = CellGrid.build(_calendar(), sources, LABELS)
    report = posterior_predictive(
        X1, Y1, params, _graph(), sources, grid, np.random.default_rng(12), 40
    )
    summary = report.calibration()
    assert summary.n_cells > 20
    assert summary.nominal == pytest.approx(0.05)
    assert summary.outside_share < 0.2
    assert summary.n_misfits <= 2
    # The observation sits inside the interval of the cells that carry the log.
    total = [c for c in report.cells if c.key.family == "total"]
    assert all(c.low <= c.observed <= c.high for c in total)


def test_a_model_drawing_at_the_wrong_rate_is_named() -> None:
    X1, Y1, sources, _ = _drawn(13, rate=-1.0)
    quiet = _params(rate=-4.0)
    grid = CellGrid.build(_calendar(), sources, LABELS)
    report = posterior_predictive(
        X1, Y1, quiet, _graph(), sources, grid, np.random.default_rng(14), 30
    )
    named = {(c.key.family, c.key.cell, c.key.label) for c in report.misfits()}
    for label in LABELS:
        assert ("total", "k1", label) in named
    busiest = report.misfits()[0]
    # Every replicate is under the observation, so nothing is at or beyond it.
    assert busiest.p_value == 0.0
    assert busiest.observed > busiest.high
    assert report.calibration().outside_share > 0.5


def test_the_misfits_are_named_busiest_first_and_the_rest_are_counted() -> None:
    X1, Y1, sources, _ = _drawn(15, rate=-1.0)
    grid = CellGrid.build(_calendar(), sources, LABELS)
    report = posterior_predictive(
        X1,
        Y1,
        _params(rate=-4.0),
        _graph(),
        sources,
        grid,
        np.random.default_rng(16),
        20,
    )
    worst = report.misfits()
    assert len(worst) > 3
    assert [c.support for c in worst] == sorted(
        (c.support for c in worst), reverse=True
    )
    text = report.report(named=2)
    assert "posterior predictive check over 20 replicates" in text
    assert str(worst[0].key) in text
    assert f"and {len(worst) - 2} further misfits" in text
    assert "against a nominal 0.050" in text


def test_a_sparse_cell_is_dropped_from_the_calibration() -> None:
    X1, Y1, sources, params = _drawn(17, rate=-3.0)
    grid = CellGrid.build(_calendar(), sources, LABELS)
    report = posterior_predictive(
        X1, Y1, params, _graph(), sources, grid, np.random.default_rng(18), 10
    )
    loose = report.calibration(min_episodes=0.0)
    tight = report.calibration(min_episodes=MIN_EPISODES)
    assert loose.n_cells > tight.n_cells
    assert tight.n_sparse > loose.n_sparse
    assert all(c.support >= MIN_EPISODES for c in report.misfits())


def test_the_floor_reads_the_busier_of_the_two_sides() -> None:
    report = _lopsided(rate=-3.5, seed=25, replicates=30)
    walk = _cell(report, CellKey("total", "k1", "walk", COUNT))
    rest = _cell(report, CellKey("total", "k1", "rest", COUNT))

    assert walk.observed == 40.0
    assert walk.observed > walk.replicate_mean
    assert walk.support == walk.observed
    assert rest.observed == 0.0
    assert rest.support == pytest.approx(rest.replicate_mean)
    assert rest.support >= MIN_EPISODES


def test_a_cell_the_record_leaves_empty_and_the_model_fills_is_named() -> None:
    """A floor read off the record alone drops the cells over-generation shows in."""
    report = _lopsided(rate=-3.5, seed=25, replicates=30)
    rest = _cell(report, CellKey("total", "k1", "rest", COUNT))

    # Every replicate sits above the observation, so nothing is under it.
    assert rest.p_value == 1.0
    assert rest.outside
    assert rest in report.misfits()
    assert rest in [c for c in report.cells if c.support >= MIN_EPISODES]
    assert report.calibration().n_misfits >= 2


def test_a_label_the_model_never_draws_leaves_the_verdict_to_the_misfit_list() -> None:
    """One member alone is busy there, so the members share no count to compare."""
    report = _lopsided(rate=-40.0, seed=26, replicates=10)
    walk = _cell(report, CellKey("total", "k1", "walk", COUNT))

    assert walk.support == 40.0
    assert walk.shared_support < MIN_EPISODES
    assert walk in report.misfits()
    assert all((f.family, f.label) != ("total", "walk") for f in report.families())


def test_one_replicate_gives_no_spread_to_place_the_observation_in() -> None:
    X1, Y1, sources, params = _drawn(19)
    grid = CellGrid.build(_calendar(), sources, LABELS)
    with pytest.raises(ValueError, match="at least two"):
        posterior_predictive(
            X1, Y1, params, _graph(), sources, grid, np.random.default_rng(20), 1
        )


def test_arrays_of_another_shape_are_refused() -> None:
    X1, Y1, sources, params = _drawn(21)
    grid = CellGrid.build(_calendar(), sources, LABELS)
    with pytest.raises(ValueError, match="where the grid holds"):
        posterior_predictive(
            X1[:10],
            Y1[:10],
            params,
            _graph(),
            sources,
            grid,
            np.random.default_rng(22),
            3,
        )


def test_a_subject_offset_reaches_the_replicate() -> None:
    """The offset the likelihood was evaluated with is the one the draw uses."""
    X1, Y1, sources, params = _drawn(23, rate=-4.0)
    grid = CellGrid.build(_calendar(), sources, LABELS)
    lifted = {COUNT: np.full((T_SLOTS, len(LABELS)), 3.0)}
    report = posterior_predictive(
        X1, Y1, params, _graph(), sources, grid, np.random.default_rng(24), 8, lifted
    )
    total = [c for c in report.cells if c.key == CellKey("total", "k1", "walk", COUNT)]
    assert total[0].replicate_mean > 4.0 * max(total[0].observed, 1.0)
