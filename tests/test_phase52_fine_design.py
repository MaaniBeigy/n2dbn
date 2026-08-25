"""
tests/test_phase52_fine_design.py
=================================
The design builder, and the four equalities that make one design serve two steps.

A built column is what `intensity.edge_feature` returns, so the graph the
likelihood evaluates and the design the solve fits hold the same numbers.  The
coarse centering is what `fine_mstep.block_center` removes at the finest grid,
so the constraint that separates two resolutions is enforced at the resolution
it is cheap at.  The resolution-aware Gram of the built columns is the matrix
the parameter step forms, so a gain the search scores is a gain the solve
realizes.  And the design's own predictor is `linear_predictor` once the
coefficients and the centering offset are written back, so a fitted design can
be read as a model.

A masked record adds a fifth.  A block mean is an average over what was
recorded, so two records that agree at every observed slot and disagree inside a
gap fit one set of coefficients and one predictor at the slots a fit scores.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from src.core.calendar import SlotCalendar
from src.core.features import (
    calendar_blocks,
    day_of_week,
    horizon_trend,
    hour_of_day,
    weekend,
)
from src.core.mask import dilate_forward, scorable_rows
from src.core.typed_graph import (
    CALENDAR,
    COUNT,
    DURATION,
    LEVEL,
    OBSERVED,
    TypedEdge,
    TypedGraph,
    basis_label,
    read_basis,
)
from src.inference.bound import broadcast, pg_mean, segment_sum
from src.learning.fine_design import (
    CalendarIndex,
    DesignColumn,
    FineDesign,
    ScaleLadder,
    _source_vector,
    build_design,
    center_at_scale,
    edge_columns,
    scale_response,
    scale_weights,
)
from src.learning.fine_mstep import (
    DesignBlock,
    ascend,
    block_center,
    normal_equations,
    regime_block,
    solve_coefficients,
)
from src.learning.fine_sstep import (
    BaseFit,
    basis_terms,
    quantile_edges,
    resolution_aware_gram,
)
from src.model.intensity import (
    FineParams,
    FineSources,
    ancestor_maps,
    edge_feature,
    linear_predictor,
    risk_set,
    usable_slots,
)

# An hourly slot over six weeks, so the ladder is hour, day, week and every
# scale holds enough epochs for a centered column to say something.
ORIGIN = datetime(2026, 2, 16)
T = 6 * 7 * 24
LABELS = ["walking", "sleep", "snacking"]
NODES = [hour_of_day(), day_of_week(), weekend(), horizon_trend(ORIGIN, "week")]


def _calendar() -> SlotCalendar:
    return SlotCalendar(origin=ORIGIN, slot=timedelta(hours=1), branching=(24, 7))


def _aggregate(X1, Y1):
    """One record's finest starts and durations summed into every scale."""
    cal = _calendar()
    counts, maps = cal.hierarchy_maps(T)
    ancestors = ancestor_maps(maps, T)
    per_scale = [
        (
            segment_sum(X1.astype(float), a, n).astype(np.int64),
            segment_sum(Y1.astype(float), a, n).astype(np.int64),
        )
        for a, n in zip(ancestors, counts)
    ]
    blocks = calendar_blocks(cal, NODES, counts)
    sources = FineSources(
        counts=[x for x, _ in per_scale],
        durations=[y for _, y in per_scale],
        calendar=blocks.blocks,
        ancestors=ancestors,
    )
    return cal, sources, blocks


def _sources(seed: int = 0):
    rng = np.random.default_rng(seed)
    X1 = (rng.random((T, len(LABELS))) < 0.09).astype(np.int64)
    Y1 = np.where(X1 > 0, rng.integers(1, 4, size=X1.shape), 0).astype(np.int64)
    cal, sources, blocks = _aggregate(X1, Y1)
    return cal, sources, blocks, X1, Y1


EDGES = [
    TypedEdge(CALENDAR, 1, 2, LEVEL, 0),
    TypedEdge(CALENDAR, 2, 0, LEVEL, 0),
    TypedEdge(CALENDAR, 3, 0, LEVEL, 0),
    TypedEdge(OBSERVED, 1, 0, COUNT, 3),
    TypedEdge(OBSERVED, 2, 1, COUNT, 1),
    TypedEdge(OBSERVED, 2, 1, DURATION, 1),
]


def _graph(cal_names, edges=EDGES, target: int = 0, quantity: str = COUNT):
    graph = TypedGraph(label_names=list(LABELS), calendar_names=list(cal_names))
    graph.set_parents(target, quantity, edges)
    return graph


def _design(seed: int = 0, edges=EDGES, latent=()):
    cal, sources, blocks, X1, Y1 = _sources(seed)
    ladder = ScaleLadder.from_sources(sources)
    names = [n for k in sorted(blocks.names) for n in blocks.names[k]]
    graph = _graph(names, edges)
    index = CalendarIndex.place(NODES, cal, sources, ladder)
    design = build_design(graph, 0, COUNT, sources, ladder, index, latent)
    return design, graph, sources, ladder, index, X1, Y1


# ---------------------------------------------------------------------------
# The ladder
# ---------------------------------------------------------------------------


def test_the_ladder_reads_the_nesting_off_the_source_arrays() -> None:
    _, sources, _, _, _ = _sources()
    ladder = ScaleLadder.from_sources(sources)
    assert ladder.K == 3
    assert ladder.n_slots == T
    assert ladder.n_epochs == [T, 42, 6]


def test_every_scale_is_centered_over_the_whole_record() -> None:
    """The intercept is what centering separates a column from, at every scale."""
    _, sources, _, _, _ = _sources()
    ladder = ScaleLadder.from_sources(sources)
    rng = np.random.default_rng(11)
    for scale in (1, 2, 3):
        column = center_at_scale(
            rng.normal(size=ladder.n_epochs[scale - 1]), ladder, scale, 0
        )
        assert float(column @ ladder.sizes[scale - 1]) == pytest.approx(0.0)


def test_the_sizes_are_the_slots_each_epoch_holds() -> None:
    _, sources, _, _, _ = _sources()
    ladder = ScaleLadder.from_sources(sources)
    assert np.allclose(ladder.sizes[0], 1.0)
    assert np.allclose(ladder.sizes[1], 24.0)
    assert np.allclose(ladder.sizes[2], 168.0)


def _bare_sources(ancestors, counts, n_labels: int = 1) -> FineSources:
    return FineSources(
        counts=[np.zeros((n, n_labels), dtype=np.int64) for n in counts],
        durations=[np.zeros((n, n_labels), dtype=np.int64) for n in counts],
        calendar=[np.zeros((n, 0)) for n in counts],
        ancestors=[np.asarray(a, dtype=np.int64) for a in ancestors],
    )


def test_the_ladder_rejects_a_count_its_map_does_not_reach() -> None:
    sources = _bare_sources([np.arange(4), np.array([0, 0, 1, 1])], [4, 3])
    with pytest.raises(ValueError, match="reaches 2"):
        ScaleLadder.from_sources(sources)


def test_the_ladder_rejects_a_ladder_that_does_not_nest() -> None:
    """An epoch holding slots from two parents has no parent to be centered in."""
    sources = _bare_sources(
        [np.arange(4), np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])], [4, 2, 2]
    )
    with pytest.raises(ValueError, match="does not nest"):
        ScaleLadder.from_sources(sources)


def test_an_irregular_nesting_is_accepted() -> None:
    sources = _bare_sources(
        [np.arange(9), np.array([0, 0, 0, 1, 1, 2, 2, 2, 2]), np.zeros(9, dtype=int)],
        [9, 3, 1],
    )
    ladder = ScaleLadder.from_sources(sources)
    assert np.allclose(ladder.sizes[1], [3.0, 2.0, 4.0])


# ---------------------------------------------------------------------------
# The centering, computed where it is cheap
# ---------------------------------------------------------------------------


def test_coarse_centering_equals_the_finest_grid_constraint() -> None:
    """
    The equality the whole coarse form rests on.

    `block_center` over the whole horizon removes a slot-weighted mean from a
    broadcast column.  The same number comes off the coarse vector for one term
    per epoch.
    """
    _, sources, _, _, _ = _sources()
    ladder = ScaleLadder.from_sources(sources)
    rng = np.random.default_rng(1)
    for scale in (1, 2, 3):
        v = rng.normal(size=ladder.n_epochs[scale - 1])
        coarse = center_at_scale(v, ladder, scale, 0)
        fine = broadcast(v, ladder.ancestors[scale - 1])[:, None]
        expected = block_center(fine, np.zeros(T, dtype=int), 1)
        assert np.allclose(
            broadcast(coarse, ladder.ancestors[scale - 1]), expected[:, 0]
        )


def test_centering_keeps_what_a_parent_epoch_mean_would_have_taken() -> None:
    """
    The between-parent contrast the design has no coefficient for, and keeps.

    A day-scale column that is constant inside each of its weeks is annihilated
    by a mean taken inside the week and survives a mean taken over the record,
    which is the difference the benchmark's lagged sleep band is scored on.
    """
    _, sources, _, _, _ = _sources()
    ladder = ScaleLadder.from_sources(sources)
    weekly = np.repeat(np.array([2.0, -1.0, 0.5, 4.0, -3.0, 1.0]), 7)
    kept = center_at_scale(weekly, ladder, 2, 0)
    assert np.std(kept) > 1.0
    inside = block_center(weekly[:, None], np.repeat(np.arange(6), 7), 6)
    assert np.allclose(inside, 0.0)


def test_a_centered_column_is_still_a_column_of_its_own_scale() -> None:
    _, sources, _, _, _ = _sources()
    ladder = ScaleLadder.from_sources(sources)
    rng = np.random.default_rng(2)
    centered = center_at_scale(rng.normal(size=42), ladder, 2, 0)
    assert centered.shape == (42,)
    fine = broadcast(centered, ladder.ancestors[1])
    for epoch in range(42):
        assert len(set(np.round(fine[ladder.ancestors[1] == epoch], 12))) == 1


def test_centering_at_the_coarsest_scale_removes_the_grand_mean() -> None:
    """That is what makes the coarsest columns orthogonal to the intercept."""
    _, sources, _, _, _ = _sources()
    ladder = ScaleLadder.from_sources(sources)
    rng = np.random.default_rng(3)
    centered = center_at_scale(rng.normal(size=6), ladder, 3, 0)
    assert float(broadcast(centered, ladder.ancestors[2]).sum()) == pytest.approx(
        0.0, abs=1e-10
    )


def test_centering_takes_a_matrix_of_columns_at_once() -> None:
    _, sources, _, _, _ = _sources()
    ladder = ScaleLadder.from_sources(sources)
    rng = np.random.default_rng(4)
    wide = rng.normal(size=(42, 3))
    got = center_at_scale(wide, ladder, 2, 0)
    assert got.shape == (42, 3)
    for j in range(3):
        assert np.allclose(got[:, j], center_at_scale(wide[:, j], ladder, 2, 0))


def test_centering_refuses_a_scale_off_the_ladder() -> None:
    _, sources, _, _, _ = _sources()
    ladder = ScaleLadder.from_sources(sources)
    with pytest.raises(ValueError, match="not one of the 3 scales"):
        center_at_scale(np.zeros(6), ladder, 4, 0)


def test_centering_refuses_a_column_of_the_wrong_length() -> None:
    _, sources, _, _, _ = _sources()
    ladder = ScaleLadder.from_sources(sources)
    with pytest.raises(ValueError, match="42 entries"):
        center_at_scale(np.zeros(7), ladder, 2, 0)


def test_centering_refuses_an_array_of_three_dimensions() -> None:
    _, sources, _, _, _ = _sources()
    ladder = ScaleLadder.from_sources(sources)
    with pytest.raises(ValueError, match="neither 1-D nor 2-D"):
        center_at_scale(np.zeros((42, 2, 2)), ladder, 2, 0)


# ---------------------------------------------------------------------------
# Where a calendar node's columns sit
# ---------------------------------------------------------------------------


def test_the_index_places_every_node_at_its_home_scale() -> None:
    cal, sources, blocks, _, _ = _sources()
    ladder = ScaleLadder.from_sources(sources)
    index = CalendarIndex.place(NODES, cal, sources, ladder)
    assert index.widths == {1: 6, 2: 7, 3: 1}
    assert index.placed[(1, 0)][0] == "hour_of_day"
    assert index.placed[(2, 0)] == ("day_of_week", "day_of_week=1")
    assert index.placed[(2, 6)] == ("weekend", "weekend=1")
    assert index.placed[(3, 0)] == ("week_index", "week_index")


def test_the_index_agrees_with_the_blocks_the_sources_were_built_from() -> None:
    """`placed[(k, j)]` is the name `calendar_blocks` gives column j of scale k."""
    cal, sources, blocks, _, _ = _sources()
    ladder = ScaleLadder.from_sources(sources)
    index = CalendarIndex.place(NODES, cal, sources, ladder)
    for k, names in blocks.names.items():
        for j, name in enumerate(names):
            assert index.column_name(k, j) == name


def test_the_index_rejects_a_node_set_the_sources_do_not_match() -> None:
    cal, sources, _, _, _ = _sources()
    ladder = ScaleLadder.from_sources(sources)
    with pytest.raises(ValueError, match="could not be named"):
        CalendarIndex.place(NODES[:2], cal, sources, ladder)


def test_the_index_rejects_a_block_list_of_the_wrong_length() -> None:
    cal, sources, _, _, _ = _sources()
    ladder = ScaleLadder.from_sources(sources)
    sources.calendar = sources.calendar[:2]
    with pytest.raises(ValueError, match="2 calendar blocks"):
        CalendarIndex.place(NODES, cal, sources, ladder)


def test_the_index_rejects_two_nodes_of_one_scale_placed_in_the_wrong_order() -> None:
    """
    A permutation inside one scale keeps every column count and moves every value.

    day_of_week and weekend are both day-scale nodes, so swapping them still
    places seven columns at scale 2.  The offsets they claim then hold each
    other's values, and a fitted weekend coefficient would be a weekday one.
    """
    cal, sources, _, _, _ = _sources()
    ladder = ScaleLadder.from_sources(sources)
    swapped = [NODES[0], NODES[2], NODES[1], NODES[3]]
    with pytest.raises(ValueError, match="does not hold 'weekend' at offset 0"):
        CalendarIndex.place(swapped, cal, sources, ladder)


def test_the_index_rejects_a_calendar_block_holding_other_numbers() -> None:
    """The count is right and the values are not, which the count cannot see."""
    cal, sources, _, _, _ = _sources()
    ladder = ScaleLadder.from_sources(sources)
    sources.calendar[1] = np.roll(sources.calendar[1], 3, axis=0)
    with pytest.raises(ValueError, match="does not hold 'day_of_week' at offset 0"):
        CalendarIndex.place(NODES, cal, sources, ladder)


def test_naming_a_column_nothing_was_placed_at_is_refused() -> None:
    cal, sources, _, _, _ = _sources()
    ladder = ScaleLadder.from_sources(sources)
    index = CalendarIndex.place(NODES, cal, sources, ladder)
    with pytest.raises(ValueError, match="scale 3 offset 4"):
        index.column_name(3, 4)


# ---------------------------------------------------------------------------
# A built column is the edge feature
# ---------------------------------------------------------------------------


def test_a_built_column_is_the_feature_the_likelihood_reads() -> None:
    """
    The first equality: what the builder holds broadcasts to `edge_feature`.

    Centering moves the column, so the check restores the mean that came off and
    compares against the value the likelihood would have read.
    """
    design, _, sources, ladder, _, _, _ = _design()
    for column in design.columns:
        raw = column.values + column.removed
        assert np.allclose(
            broadcast(raw, ladder.ancestors[column.scale - 1]),
            edge_feature(column.edge, sources),
        )


def test_a_lagged_column_reads_nothing_before_the_record_opens() -> None:
    design, _, _, _, _, _, _ = _design()
    lagged = [c for c in design.columns if c.edge.lag == 3][0]
    raw = lagged.values + lagged.removed
    assert np.allclose(raw[:3], 0.0)


def test_a_calendar_node_reaches_the_design_as_a_lag_zero_edge() -> None:
    design, _, sources, ladder, _, _, _ = _design()
    week = [c for c in design.columns if c.edge.src_kind == CALENDAR and c.scale == 3]
    assert len(week) == 1
    assert week[0].name == "week_index k3 lag0"
    assert np.allclose(
        week[0].values, center_at_scale(sources.calendar[2][:, 0], ladder, 3, 0)
    )
    # The horizon is the parent of the coarsest scale, so what centering took is
    # one number and the intercept absorbs it.
    assert np.allclose(week[0].removed, week[0].removed[0])


def test_a_band_of_a_source_is_a_column_beside_that_source_s_log_term() -> None:
    """
    Two terms of one source are two columns, each read the way the likelihood does.

    The band is lagged and centered by the lines the log term is, under the
    budget the search declared, so the two are scored over one row set.
    """
    cal, sources, blocks, _, _ = _sources()
    ladder = ScaleLadder.from_sources(sources)
    index = CalendarIndex.place(NODES, cal, sources, ladder)
    names = [n for k in sorted(blocks.names) for n in blocks.names[k]]
    day = np.asarray(sources.durations[1][:, 1], dtype=float)
    band = basis_terms(quantile_edges(day, 4))[-1]
    plain = TypedEdge(OBSERVED, 2, 1, DURATION, 1)
    banded = TypedEdge(OBSERVED, 2, 1, DURATION, 1, band)
    budget = {2: 2}

    graph = _graph(names, [plain, banded])
    columns = edge_columns([plain, banded], graph, sources, ladder, index, budget)
    assert [c.name for c in columns] == [
        "sleep.duration:log k2 lag1",
        f"sleep.duration:{basis_label(band)} k2 lag1",
    ]

    lagged = np.concatenate([[0.0], read_basis(day, band)[:-1]])
    assert set(np.unique(lagged)) == {0.0, 1.0}, "the band splits this source"
    assert np.allclose(columns[1].values + columns[1].removed, lagged)
    assert np.allclose(columns[1].values, center_at_scale(lagged, ladder, 2, 2))
    assert np.allclose(
        broadcast(lagged, ladder.ancestors[1]), edge_feature(banded, sources)
    )


def test_every_column_carries_the_edge_it_was_built_from() -> None:
    design, graph, _, _, _, _, _ = _design()
    assert {c.edge for c in design.columns} == graph.get_parents(0, COUNT)


def test_a_column_at_a_scale_the_ladder_does_not_have_is_refused() -> None:
    cal, sources, blocks, _, _ = _sources()
    ladder = ScaleLadder.from_sources(sources)
    index = CalendarIndex.place(NODES, cal, sources, ladder)
    names = [n for k in sorted(blocks.names) for n in blocks.names[k]]
    graph = _graph(names, [])
    off = TypedEdge(OBSERVED, 9, 1, COUNT, 1)
    with pytest.raises(ValueError, match="not one of the 3 scales of this ladder"):
        edge_columns([off], graph, sources, ladder, index)


def test_a_calendar_column_nothing_was_placed_at_is_refused() -> None:
    """The graph counts calendar nodes; the design has to find the column itself."""
    cal, sources, blocks, _, _ = _sources()
    ladder = ScaleLadder.from_sources(sources)
    index = CalendarIndex.place(NODES, cal, sources, ladder)
    names = [n for k in sorted(blocks.names) for n in blocks.names[k]]
    graph = _graph(names, [])
    with pytest.raises(ValueError, match="no calendar column sits at scale 3 offset 5"):
        edge_columns(
            [TypedEdge(CALENDAR, 3, 5, LEVEL, 0)], graph, sources, ladder, index
        )


# ---------------------------------------------------------------------------
# The blocks
# ---------------------------------------------------------------------------


def test_the_blocks_run_intercept_then_coarsest_to_finest() -> None:
    design, _, _, _, _, _, _ = _design()
    assert [b.name for b in design.blocks] == [
        "intercept",
        "scale 3",
        "scale 2",
        "scale 1",
    ]
    assert [b.width for b in design.blocks] == [1, 1, 3, 2]
    assert design.width == 7


def test_a_scale_with_no_edge_gets_no_block() -> None:
    only_day = [e for e in EDGES if e.src_scale == 2]
    design, _, _, _, _, _, _ = _design(edges=only_day)
    assert [b.name for b in design.blocks] == ["intercept", "scale 2"]


def test_the_columns_of_one_scale_share_a_block() -> None:
    """
    Centering separates a scale from its parent and leaves a scale to itself.

    Two day-scale columns stay correlated after it, so they are solved together
    and the pair the constraint cannot separate never crosses a block boundary.
    """
    design, _, _, _, _, _, _ = _design()
    day = [c for c in design.columns if c.scale == 2]
    stacked = np.stack([c.values for c in day], axis=1)
    gram = stacked.T @ stacked
    off = gram - np.diag(np.diag(gram))
    assert np.abs(off).max() > 1e-6, "the day columns are correlated"
    block = [b for b in design.blocks if b.name == "scale 2"][0]
    assert block.width == len(day)


def test_a_block_of_two_scales_stays_record_mean_zero() -> None:
    """
    A centered column has no mean over the record, whatever scale it varies at.

    So a design holding an hour column and a day column together survives
    `block_center` over the horizon untouched, which is the constraint that
    holds across a block that mixes resolutions and separates each of them from
    the intercept.  Inside a week the same columns carry a mean of their own,
    which is the between-week contrast the design keeps.
    """
    design, _, _, ladder, _, _, _ = _design()
    mixed = [c for c in design.columns if c.scale in (1, 2)]
    stacked = np.stack([c.at_finest(ladder) for c in mixed], axis=1)
    horizon = np.zeros(ladder.n_slots, dtype=np.int64)
    assert np.allclose(block_center(stacked, horizon, 1), stacked, atol=1e-10)
    assert np.allclose(stacked.mean(axis=0), 0.0, atol=1e-10)
    inside = np.stack(
        [stacked[ladder.ancestors[2] == e].mean(axis=0) for e in range(6)]
    )
    assert np.abs(inside).max() > 1e-6


def test_a_latent_block_lands_after_the_edge_blocks() -> None:
    rng = np.random.default_rng(5)
    gamma = rng.dirichlet([2.0, 2.0], size=42)
    _, sources, _, _, _ = _sources()
    latent = regime_block(gamma, ancestor_maps([np.repeat(np.arange(42), 24)], T)[1])
    design, _, _, _, _, _, _ = _design(latent=[latent])
    assert design.blocks[-1] is latent
    assert design.width == 8
    assert design.names[-1] == "regime[0]"
    assert design.names[:2] == ["intercept", "week_index k3 lag0"]


# ---------------------------------------------------------------------------
# The two steps read one design
# ---------------------------------------------------------------------------


def _weighted(seed: int = 6, latent=()):
    design, graph, sources, ladder, index, X1, Y1 = _design(latent=latent)
    rng = np.random.default_rng(seed)
    weights = pg_mean(rng.normal(size=T))
    at_risk = risk_set(X1, Y1)[:, 0]
    return design, ladder, X1[:, 0], weights, at_risk


def _tuesday_regime(strength: float = 0.95) -> DesignBlock:
    """A two-state regime whose second state is on during Tuesdays."""
    p = np.where(np.arange(42) % 7 == 1, strength, 1.0 - strength)
    gamma = np.stack([1.0 - p, p], axis=1)
    return regime_block(gamma, ancestor_maps([np.repeat(np.arange(42), 24)], T)[1])


def test_the_resolution_aware_gram_is_the_parameter_step_matrix() -> None:
    """
    The third equality: the search scores the matrix the solve forms.

    Every entry is computed at the finer of its two home grids, and the answer
    is the weighted cross-product the normal equations build over the risk set.
    """
    design, ladder, x, weights, at_risk = _weighted()
    A, _ = normal_equations(x, design.blocks, weights, at_risk, ridge=0.0)
    aware = resolution_aware_gram(
        design.scaled(), ladder.ancestors, scale_weights(weights, ladder, at_risk)
    )
    assert np.allclose(aware, A, atol=1e-8)


def test_the_scaled_response_is_the_parameter_step_target() -> None:
    """One per-slot response, reduced at each column's own scale."""
    design, ladder, x, weights, at_risk = _weighted()
    _, b = normal_equations(x, design.blocks, weights, at_risk, ridge=0.0)
    response = scale_response((x > 0).astype(float) - 0.5, ladder, at_risk)
    got = [float(response[c.scale - 1] @ c.values) for c in design.columns]
    assert np.allclose([float(response[-1].sum())] + got, b, atol=1e-8)


def test_the_weights_of_a_slot_off_the_risk_set_are_dropped() -> None:
    _, sources, _, _, _ = _sources()
    ladder = ScaleLadder.from_sources(sources)
    weights = np.ones(T)
    at_risk = np.zeros(T, dtype=bool)
    at_risk[:24] = True
    per_scale = scale_weights(weights, ladder, at_risk)
    assert per_scale[1][0] == 24.0
    assert np.allclose(per_scale[1][1:], 0.0)
    assert np.allclose(scale_weights(weights, ladder)[1], 24.0)


def test_the_response_of_a_slot_off_the_risk_set_is_dropped() -> None:
    _, sources, _, _, _ = _sources()
    ladder = ScaleLadder.from_sources(sources)
    r = np.full(T, 0.5)
    at_risk = np.zeros(T, dtype=bool)
    at_risk[:24] = True
    assert scale_response(r, ladder, at_risk)[1][0] == pytest.approx(12.0)
    assert np.allclose(scale_response(r, ladder, at_risk)[1][1:], 0.0)
    assert np.allclose(scale_response(r, ladder)[1], 12.0)


def test_an_integer_risk_set_is_refused_by_the_reduction() -> None:
    """
    A 0/1 integer mask is elementwise the boolean one and means something else.

    `normal_equations` selects rows with it, so an integer array selects rows 0
    and 1 over and over while the reduction multiplies by it, and the search and
    the solve would fit two different problems with nothing raised.
    """
    design, ladder, x, weights, at_risk = _weighted()
    as_int = at_risk.astype(np.int64)
    solved, _ = normal_equations(x, design.blocks, weights, as_int, ridge=0.0)
    honest, _ = normal_equations(x, design.blocks, weights, at_risk, ridge=0.0)
    assert not np.allclose(solved, honest), "the integer mask moved the solve"
    with pytest.raises(ValueError, match="boolean mask"):
        scale_weights(weights, ladder, as_int)
    with pytest.raises(ValueError, match="boolean mask"):
        scale_response(x, ladder, as_int)


def test_a_risk_set_of_the_wrong_length_is_refused() -> None:
    _, sources, _, _, _ = _sources()
    ladder = ScaleLadder.from_sources(sources)
    with pytest.raises(ValueError, match="one entry per finest slot"):
        scale_weights(np.ones(T), ladder, np.ones((T, 3), dtype=bool))


def test_the_search_system_is_the_solve_system() -> None:
    design, ladder, x, weights, at_risk = _weighted()
    A, b = normal_equations(x, design.blocks, weights, at_risk, ridge=0.0)
    assert np.allclose(design.search_gram(weights, at_risk), A, atol=1e-8)
    r = (x > 0).astype(float) - 0.5
    assert np.allclose(design.search_target(r, at_risk), b, atol=1e-8)


def test_the_search_system_covers_a_latent_block_too() -> None:
    """
    The third equality once a latent variable brings blocks of its own.

    A posterior column has no coarse-grid form, so the resolution-aware Gram of
    `scaled()` stops at the edge coefficients while the solve goes on to invert
    a wider matrix.  The two are one matrix again when the latent block enters
    at the finest grid, covariance included.
    """
    design, ladder, x, weights, at_risk = _weighted(latent=[_tuesday_regime()])
    A, b = normal_equations(x, design.blocks, weights, at_risk, ridge=0.0)
    assert A.shape == (8, 8)
    assert len(design.scaled()) == 7
    assert np.allclose(design.search_gram(weights, at_risk), A, atol=1e-8)
    r = (x > 0).astype(float) - 0.5
    assert np.allclose(design.search_target(r, at_risk), b, atol=1e-8)


def test_a_gain_scored_over_a_latent_block_is_the_gain_the_solve_realizes() -> None:
    """
    What a search matrix short of the latent block costs, in the units of a move.

    The candidate is the Tuesday contrast and the regime is on during Tuesdays,
    so a matrix that stops before the posterior column credits the candidate
    with the share of the response the regime already carries.
    """
    latent = [_tuesday_regime()]
    candidate = TypedEdge(CALENDAR, 2, 0, LEVEL, 0)
    base, _, _, _, _, X1, Y1 = _design(
        edges=[e for e in EDGES if e != candidate], latent=latent
    )
    full, _, _, ladder, _, _, _ = _design(latent=latent)
    weights = pg_mean(np.random.default_rng(6).normal(size=T))
    at_risk = risk_set(X1, Y1)[:, 0]
    x = X1[:, 0]
    r = (x > 0).astype(float) - 0.5

    def searched(design: FineDesign) -> float:
        return BaseFit(
            design.search_gram(weights, at_risk), design.search_target(r, at_risk)
        ).value

    def solved(design: FineDesign) -> float:
        return BaseFit(*normal_equations(x, design.blocks, weights, at_risk, 0.0)).value

    assert searched(full) - searched(base) == pytest.approx(
        solved(full) - solved(base), abs=1e-8
    )


def test_the_intercept_enters_the_search_as_a_coarsest_scale_column() -> None:
    design, _, _, ladder, _, _, _ = _design()
    scaled = design.scaled()
    assert scaled[0].name == "intercept"
    assert scaled[0].scale == ladder.K
    assert np.allclose(scaled[0].values, 1.0)
    assert [c.name for c in scaled[1:]] == [c.name for c in design.columns]


# ---------------------------------------------------------------------------
# A fitted design reads back as a model
# ---------------------------------------------------------------------------


def test_a_fitted_coefficient_names_the_edge_it_belongs_to() -> None:
    design, graph, sources, ladder, index, X1, Y1 = _design()
    weights = pg_mean(np.zeros(T))
    at_risk = risk_set(X1, Y1)[:, 0]
    beta = solve_coefficients(X1[:, 0], design.blocks, weights, at_risk)
    fitted = design.coefficients(beta)
    assert set(fitted) == graph.get_parents(0, COUNT)
    for j, column in enumerate(design.columns):
        assert fitted[column.edge] == pytest.approx(beta[j + 1])
    assert design.intercept(beta) == pytest.approx(beta[0])


def test_the_design_predictor_is_the_likelihood_predictor() -> None:
    """
    The fourth equality: a fitted design is a model the likelihood can evaluate.

    Centering took a parent mean out of every column, so the coefficients go
    into `FineParams` and what centering removed goes into the offset argument.
    """
    design, graph, sources, ladder, index, X1, Y1 = _design()
    weights = pg_mean(np.zeros(T))
    at_risk = risk_set(X1, Y1)[:, 0]
    beta = solve_coefficients(X1[:, 0], design.blocks, weights, at_risk)

    params = design.write_into(FineParams(n_labels=len(LABELS)), beta)
    offsets = np.zeros((T, len(LABELS)))
    offsets[:, 0] = design.offset(beta)
    eta = linear_predictor(params, graph, sources, COUNT, offsets)
    assert np.allclose(eta[:, 0], design.predictor(beta), atol=1e-9)


def test_the_offset_carries_a_latent_block_too() -> None:
    rng = np.random.default_rng(7)
    gamma = rng.dirichlet([2.0, 2.0], size=42)
    latent = regime_block(gamma, ancestor_maps([np.repeat(np.arange(42), 24)], T)[1])
    design, graph, sources, ladder, index, X1, Y1 = _design(latent=[latent])
    beta = rng.normal(size=design.width)

    params = design.write_into(FineParams(n_labels=len(LABELS)), beta)
    offsets = np.zeros((T, len(LABELS)))
    offsets[:, 0] = design.offset(beta)
    eta = linear_predictor(params, graph, sources, COUNT, offsets)
    assert np.allclose(eta[:, 0], design.predictor(beta), atol=1e-9)


def test_a_design_with_no_edge_is_the_intercept_alone() -> None:
    design, graph, sources, ladder, index, X1, Y1 = _design(edges=[])
    assert design.names == ["intercept"]
    assert design.width == 1
    beta = np.array([0.3])
    assert np.allclose(design.predictor(beta), 0.3)
    assert np.allclose(design.offset(beta), 0.0)
    assert design.coefficients(beta) == {}


# ---------------------------------------------------------------------------
# Ascent over the built blocks
# ---------------------------------------------------------------------------


def test_the_bound_never_falls_under_a_shuffled_block_order() -> None:
    """
    The property `fine_mstep.ascend` claims, checked on blocks the builder made.

    Each step maximizes the same bound over one block with the rest held, so a
    schedule is free and a block boundary the builder chose cannot cost the fit.
    """
    rng = np.random.default_rng(8)
    gamma = rng.dirichlet([2.0, 2.0], size=42)
    latent = regime_block(gamma, ancestor_maps([np.repeat(np.arange(42), 24)], T)[1])
    design, _, _, _, _, X1, Y1 = _design(latent=[latent])
    at_risk = risk_set(X1, Y1)[:, 0]
    for _ in range(6):
        order = rng.integers(0, len(design.blocks), size=12).tolist()
        _, trace = ascend(
            X1[:, 0], design.blocks, np.zeros(design.width), order, at_risk
        )
        drops = [b - a for a, b in zip(trace, trace[1:]) if b < a - 1e-9]
        assert drops == [], f"the bound fell by {drops} under {order}"


def test_ascent_over_the_built_blocks_improves_on_where_it_started() -> None:
    design, _, _, _, _, X1, Y1 = _design()
    at_risk = risk_set(X1, Y1)[:, 0]
    order = list(range(len(design.blocks))) * 4
    _, trace = ascend(X1[:, 0], design.blocks, np.zeros(design.width), order, at_risk)
    assert trace[-1] > trace[0]


def test_a_column_holds_its_own_finest_form() -> None:
    design, _, _, ladder, _, _, _ = _design()
    column = design.columns[0]
    assert isinstance(column, DesignColumn)
    assert np.allclose(
        column.at_finest(ladder),
        broadcast(column.values, ladder.ancestors[column.scale - 1]),
    )
    assert isinstance(design, FineDesign)
    assert isinstance(design.blocks[0], DesignBlock)


# ---------------------------------------------------------------------------
# A mean over what the record holds
# ---------------------------------------------------------------------------

# One day of the second week goes unrecorded, which leaves the week partly
# observed and every other week whole.
MASKED_DAY = 9
GAP = slice(MASKED_DAY * 24, (MASKED_DAY + 1) * 24)
MASKED_WEEK = 1
# The largest lag the candidate space allows at each scale, which fixes the row
# set both records below are scored on.
GAP_BUDGET = {1: 3, 2: 1}


def _gap_mask() -> np.ndarray:
    observed = np.ones(T, dtype=bool)
    observed[GAP] = False
    return observed


def _two_records():
    """
    Two records that agree at every observed slot and disagree inside the gap.

    Inside the gap one of them holds nothing and the other holds episodes of a
    single slot, so neither runs past the gap's last slot and the two share one
    risk set.
    """
    _, _, _, X1, Y1 = _sources()
    quiet_x, quiet_y = X1.copy(), Y1.copy()
    quiet_x[GAP] = 0
    quiet_y[GAP] = 0
    busy_x, busy_y = quiet_x.copy(), quiet_y.copy()
    busy_x[GAP.start : GAP.stop : 2] = 1
    busy_y[GAP.start : GAP.stop : 2] = 1
    return (quiet_x, quiet_y), (busy_x, busy_y)


def _gap_design(X1, Y1):
    """One record's design on the masked ladder, with the sources behind it."""
    cal, sources, blocks = _aggregate(X1, Y1)
    ladder = ScaleLadder.from_sources(sources, _gap_mask())
    names = [n for k in sorted(blocks.names) for n in blocks.names[k]]
    index = CalendarIndex.place(NODES, cal, sources, ladder)
    design = build_design(_graph(names), 0, COUNT, sources, ladder, index)
    return design, sources


def test_the_ladder_reduces_the_mask_into_every_scale() -> None:
    _, sources, _, _, _ = _sources()
    ladder = ScaleLadder.from_sources(sources, _gap_mask())
    assert np.array_equal(
        np.flatnonzero(~ladder.observed[0]), np.arange(GAP.start, GAP.stop)
    )
    assert np.array_equal(np.flatnonzero(~ladder.observed[1]), [MASKED_DAY])
    assert np.array_equal(np.flatnonzero(~ladder.observed[2]), [MASKED_WEEK])
    assert all(seen.all() for seen in ScaleLadder.from_sources(sources).observed)


def test_the_ladder_refuses_a_mask_of_the_wrong_length() -> None:
    _, sources, _, _, _ = _sources()
    with pytest.raises(ValueError, match="one flag per finest slot"):
        ScaleLadder.from_sources(sources, np.ones(42, dtype=bool))


def test_an_unobserved_day_carries_no_weight_in_the_record_mean() -> None:
    _, sources, _, _, _ = _sources()
    ladder = ScaleLadder.from_sources(sources, _gap_mask())
    v = np.random.default_rng(9).normal(size=42)
    got = center_at_scale(v, ladder, 2, 0)
    kept = np.arange(42) != MASKED_DAY
    assert np.allclose(got, v - v[kept].mean())


def test_a_day_that_reads_over_the_gap_carries_no_weight_either() -> None:
    """
    The day after the gap is observed and the value it reads at lag one is not.

    Its entry is a number nobody recorded, so it leaves the mean with the gap
    day itself, and what comes off the column is a mean of recorded numbers.
    """
    _, sources, _, _, _ = _sources()
    ladder = ScaleLadder.from_sources(sources, _gap_mask())
    v = np.random.default_rng(10).normal(size=42)
    got = center_at_scale(v, ladder, 2, 1)
    dropped = {MASKED_DAY, MASKED_DAY + 1}
    kept = np.array([d for d in range(42) if d not in dropped])
    assert np.allclose(got, v - v[kept].mean())


def test_a_record_with_nothing_recorded_keeps_its_column() -> None:
    """
    A mean over nothing is undefined, so the entries stand as they came in.

    No row of such a record survives the dilation, so the choice reaches no
    fitted number.
    """
    _, sources, _, _, _ = _sources()
    ladder = ScaleLadder.from_sources(sources, np.zeros(T, dtype=bool))
    v = np.random.default_rng(11).normal(size=42)
    assert np.allclose(center_at_scale(v, ladder, 2, 0), v)


def test_a_fit_at_observed_slots_reads_nothing_from_inside_the_gap() -> None:
    """
    The claim the whole weighting is for, in the units of a fitted model.

    The two records are one record at every slot anybody wrote down.  They are
    scored on one row set and one response, and a mean weighted by the slots the
    record covers is what makes them one fit.  The coefficients agree and so
    does the predictor everywhere the fit reads it.  The day after the gap holds
    the two records apart, since its lagged column reads a source only one of
    them carries, and that day sits outside the row set.
    """
    weights = pg_mean(np.zeros(T))
    fitted = []
    for X1, Y1 in _two_records():
        design, sources = _gap_design(X1, Y1)
        rows = scorable_rows(_gap_mask(), sources.ancestors, GAP_BUDGET)
        at_risk = usable_slots(X1, Y1, rows)[:, 0]
        beta = solve_coefficients(X1[:, 0], design.blocks, weights, at_risk)
        fitted.append((rows, at_risk, X1[:, 0], beta, design.predictor(beta)))

    (rows, at_risk, x, beta, eta), (other_rows, other_risk, y, gamma, zeta) = fitted
    assert np.array_equal(
        np.flatnonzero(~rows), np.arange(GAP.start, GAP.stop + 24)
    ), "the gap day and the day that reads it leave the score"
    assert np.array_equal(rows, other_rows)
    assert np.array_equal(at_risk, other_risk)
    assert np.array_equal(x[at_risk], y[at_risk])
    assert np.allclose(beta, gamma, atol=1e-12)
    assert np.allclose(eta[at_risk], zeta[at_risk], atol=1e-12)
    after = slice(GAP.stop, GAP.stop + 24)
    assert not np.allclose(eta[after], zeta[after])


def test_a_record_with_no_gap_weights_every_epoch_by_its_slot_count() -> None:
    """
    The mask reaches centering as a weight, so a record with no gap is untouched.

    Every stored result was fitted with the slot count as that weight, and a
    ladder carrying no gap holds it entry for entry at every lag.
    """
    _, sources, _, _, _ = _sources()
    ladder = ScaleLadder.from_sources(sources)
    for k in range(ladder.K):
        for lag in (0, 1, 3):
            kept = dilate_forward(ladder.observed[k], lag)
            assert np.array_equal(ladder.sizes[k] * kept, ladder.sizes[k])


def test_an_unmasked_fit_holds_the_coefficients_of_the_record_centering() -> None:
    """The pinned solve of the design as it is built, one mean per column."""
    design, _, _, _, _, X1, Y1 = _design()
    beta = solve_coefficients(
        X1[:, 0], design.blocks, pg_mean(np.zeros(T)), risk_set(X1, Y1)[:, 0]
    )
    assert np.allclose(
        beta,
        [
            -1.5909005739300043,
            0.005505418922767713,
            -0.10825285726180612,
            0.08593715746230888,
            -0.08595712473224487,
            -0.12163333103254124,
            -0.08184034847851246,
        ],
        atol=1e-12,
    )


# ---------------------------------------------------------------------------
# One centering convention for the whole candidate set
# ---------------------------------------------------------------------------


def test_the_ladder_refuses_a_mask_that_is_not_boolean() -> None:
    """A 0/1 integer mask reads as a mask and indexes as row numbers."""
    _cal, sources, _blocks, _X, _Y = _sources()
    with pytest.raises(ValueError, match="boolean array"):
        ScaleLadder.from_sources(
            sources, observed=np.ones(sources.n_slots, dtype=np.int64)
        )


def test_an_edge_reading_past_its_declared_budget_is_refused() -> None:
    """Its rows and the mean removed from it would come from two conventions."""
    cal, sources, _blocks, _X, _Y = _sources()
    ladder = ScaleLadder.from_sources(sources)
    index = CalendarIndex.place(NODES, cal, sources, ladder)
    graph = TypedGraph(label_names=["target", "source"])
    far = TypedEdge(
        src_kind=OBSERVED, src_scale=2, src_label=1, src_quantity=COUNT, lag=3
    )
    with pytest.raises(ValueError, match="declared budget there is 1"):
        edge_columns([far], graph, sources, ladder, index, {2: 1})


def test_two_candidates_of_different_lags_share_one_centering() -> None:
    """Keyed to their own lags the winner can change on the convention alone."""
    cal, sources, _blocks, _X, _Y = _sources()
    observed = np.ones(sources.n_slots, dtype=bool)
    day = np.asarray(sources.ancestors[1])
    observed[(day >= 3) & (day <= 5)] = False
    ladder = ScaleLadder.from_sources(sources, observed=observed)
    index = CalendarIndex.place(NODES, cal, sources, ladder)
    graph = TypedGraph(label_names=["target", "source"])

    near = TypedEdge(
        src_kind=OBSERVED, src_scale=2, src_label=1, src_quantity=COUNT, lag=1
    )
    far = TypedEdge(
        src_kind=OBSERVED, src_scale=2, src_label=1, src_quantity=COUNT, lag=2
    )
    budget = {2: 2}
    columns = edge_columns([near, far], graph, sources, ladder, index, budget)
    # Both columns had the same weight set removed, which is the budget's.
    for column, edge in zip(columns, (near, far)):
        raw = np.zeros(ladder.n_epochs[1])
        reach = np.arange(raw.size) - edge.lag
        inside = reach >= 0
        source, _ = _source_vector(edge, graph, sources, index)
        raw[inside] = source[reach[inside]]
        assert np.allclose(column.values, center_at_scale(raw, ladder, 2, budget[2]))
