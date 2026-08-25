"""
tests/test_phase61_subject_offset.py
====================================
Per-subject variation as a level, and the four things that make it a model.

Identification.  S levels take S numbers together with the pooled intercept, and
the row-weighted deviation coding is what fixes which S.  A pooled intercept
beside S subject indicators is S+1 columns for S levels, so the population
intercept moves by any constant as long as every subject moves back by the same
one, and two runs of the same data report two different populations.  The coding
here has no such direction, which is shown twice, once on the columns themselves
and once on two starts of the same fit landing on one population intercept.

Absorption.  A level is constant over a subject's whole record, and every column
the design carries has weighted mean zero over that record, because
`center_at_scale` removes a parent-epoch mean and the parent of the coarsest
scale is the horizon.  The two are orthogonal under the counting measure of the
declared rows, measured here on the benchmark's own walking count trend, which
the manifest declares as three episodes per week over weeks 1 to 12.  The
response carries 19.9 percent of its variance between subjects and the trend
column carries none, which is the whole of what the level block is for.

Streaming.  Every quantity the fit accumulates is a sum over rows, so a cohort
of identical records carries an exact multiple of one record's matrices and
every deviation comes back at exactly zero.  The level columns are never laid
down, and the totals they are formed from reproduce the block that would have
been.

The regime path.  Per-subject variation here is a level and no chain runs.  A
chain of M states holds M distinct levels per label, so a roster of S distinct
levels needs M = S states and a transition matrix per subject that never leaves
its state.  Granted its best possible posterior, a two-state chain over three
distinct subjects still leaves one pair sharing a level, which is measured.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

from src.core.cohort import Cohort
from src.core.scales import ScaleHierarchy
from src.core.typed_graph import (
    CALENDAR,
    COUNT,
    DURATION,
    LEVEL,
    OBSERVED,
    TypedEdge,
    TypedGraph,
)
from src.core.variables import MultiScaleData
from src.data.dynamic_patterns import (
    BENCHMARK_ORIGIN,
    VERBATIM,
    build_population,
    read_subject_event_log,
)
from src.inference.bound import pg_update, segment_sum
from src.learning.fine_design import (
    CalendarIndex,
    ScaleLadder,
    build_design,
    center_at_scale,
)
from src.learning.fine_mstep import _RIDGE, DesignBlock, normal_equations, stack_design
from src.learning.fine_sem import fit_rows
from src.learning.subject_offset import (
    LevelCoding,
    LevelSweep,
    SubjectLevelFit,
    level_blocks,
    level_normal_equations,
    level_rows,
    subject_deviations,
    subject_offsets,
)
from src.model.intensity import (
    FineParams,
    linear_predictor,
    start_loglik,
    usable_slots,
)

# An hourly slot over four weeks, so the ladder is hour and day.  Two labels: a
# target driven by yesterday's count of the other one.
HOURS = 24
DAYS = 28
T = HOURS * DAYS
LABELS = ["walking", "sleep"]
DAY_OF = np.repeat(np.arange(DAYS), HOURS).astype(np.int64)

# The day-scale lag the candidate space allows, declared once and handed to
# every subject, which fixes the row set for every model the sum compares.
BUDGET = {2: 1}
EDGE = TypedEdge(OBSERVED, 2, 1, COUNT, 1)
EMPTY = CalendarIndex(placed={}, widths={})


def _hierarchy() -> ScaleHierarchy:
    return ScaleHierarchy(K=2, H=[T, DAYS], parent_maps=[DAY_OF, None])


def _subject(seed: int, level: float, rate: float = 0.2) -> MultiScaleData:
    """One record whose target sits at the cohort level plus its own offset."""
    rng = np.random.default_rng(seed)
    X1 = np.zeros((T, 2), dtype=np.int64)
    X1[:, 1] = rng.random(T) < rate
    total = segment_sum(X1[:, 1].astype(float), DAY_OF, DAYS)
    previous = np.concatenate([[0.0], total[:-1]])
    odds = -3.0 + level + 0.4 * np.log1p(previous)[DAY_OF]
    X1[:, 0] = rng.random(T) < 1.0 / (1.0 + np.exp(-odds))
    Y1 = (X1 > 0).astype(np.int64)
    return MultiScaleData(
        hierarchy=_hierarchy(),
        label_names=list(LABELS),
        X=[X1, segment_sum(X1.astype(float), DAY_OF, DAYS).astype(np.int64)],
        Y=[Y1, segment_sum(Y1.astype(float), DAY_OF, DAYS).astype(np.int64)],
        C=[np.zeros((T, 1)), np.zeros((DAYS, 1))],
    )


def _cohort(levels, seeds=None, rates=None) -> Cohort:
    names = tuple(f"p{i:02d}" for i in range(len(levels)))
    seeds = list(range(len(levels))) if seeds is None else list(seeds)
    rates = [0.2] * len(levels) if rates is None else list(rates)
    table = {n: (s, v, r) for n, s, v, r in zip(names, seeds, levels, rates)}
    return Cohort(
        subjects=names,
        hierarchy=_hierarchy(),
        label_names=list(LABELS),
        reader=lambda name: _subject(*table[name]),
        masks=np.ones((len(levels), T), dtype=bool),
    )


def _graph() -> TypedGraph:
    graph = TypedGraph(label_names=list(LABELS))
    graph.add_edge(EDGE, 0, COUNT)
    return graph


def _params(cohort: Cohort) -> FineParams:
    params = FineParams(n_labels=2, n_subjects=cohort.S)
    params.base[COUNT][:] = -2.0
    return params


def _stream_solve(
    cohort, graph, target, row_of, budget=None, index=EMPTY, sweeps=60, start=None
):
    """
    Accumulate and solve one target over a cohort, with every subject's row named.

    The predictor is read off the design's own columns, so nothing is written
    back into a parameter object and the only difference between two runs of this
    is the constant row each subject carries.  A row of `[1.0]` is the pooled
    intercept, a row of the deviation coding is the participant axis, and a row
    of `[1.0, state]` is the best a chain of two states could ever do.
    """
    budget = BUDGET if budget is None else budget
    beta = None if start is None else np.asarray(start, dtype=float)
    last = None
    residual = np.zeros(cohort.S)
    for _ in range(sweeps):
        A = b = None
        total = 0.0
        for i, (_name, data) in enumerate(cohort.stream()):
            X1, Y1 = data.get_X(1), data.get_Y(1)
            sources = cohort.sources(data)
            at_risk = usable_slots(X1, Y1, fit_rows(data.observed, budget, sources))
            ladder = ScaleLadder.from_sources(sources, data.observed)
            design = build_design(
                graph, target, COUNT, sources, ladder, index, budget=budget
            )
            rest = design.blocks[1:]
            means = stack_design(rest)[0] if rest else np.zeros((ladder.n_slots, 0))
            row = np.asarray(row_of(i), dtype=float)
            eta = (
                np.zeros(ladder.n_slots)
                if beta is None
                else float(row @ beta[: row.size]) + means @ beta[row.size :]
            )
            rows = at_risk[:, target]
            starts = (X1[:, target] > 0).astype(float)
            residual[i] = float((starts - 1.0 / (1.0 + np.exp(-eta)))[rows].mean())
            total += start_loglik(X1[:, target], eta, rows)
            one, two = level_normal_equations(
                X1[:, target], rest, pg_update(eta), row, rows
            )
            A, b = (one, two) if A is None else (A + one, b + two)
        if last is not None and abs(total - last) < 1e-12 * (1.0 + abs(last)):
            break
        last = total
        beta = np.linalg.solve(A + _RIDGE * np.eye(A.shape[0]), b)
    return beta, residual, total


# ---------------------------------------------------------------------------
# The coding
# ---------------------------------------------------------------------------


def test_the_level_block_holds_an_intercept_and_S_minus_one_contrasts() -> None:
    coding = subject_deviations([10.0, 20.0, 30.0])
    assert coding.shape == (3, 3)
    assert np.array_equal(coding[:, 0], np.ones(3))
    # Column 1 + r is subject r against the last subject on the roster.
    assert np.array_equal(coding[:2, 1:], np.eye(2))
    assert np.allclose(coding[2, 1:], [-10.0 / 30.0, -20.0 / 30.0])
    assert np.linalg.matrix_rank(coding) == 3


def test_the_coding_puts_the_constraint_on_the_design() -> None:
    """`sum_s n_s a_s = 0` holds for every coefficient vector, so nothing
    downstream remembers it."""
    counts = np.array([37.0, 11.0, 52.0, 4.0])
    rng = np.random.default_rng(0)
    for _ in range(5):
        beta = rng.normal(size=4)
        offsets = subject_offsets(beta, counts)
        assert float(counts @ offsets) == pytest.approx(0.0, abs=1e-12)
        # The level a slot carries is the intercept plus that subject's offset.
        assert np.allclose(subject_deviations(counts) @ beta, beta[0] + offsets)


def test_one_subject_is_the_intercept_it_always_was() -> None:
    assert np.array_equal(subject_deviations([12.0]), np.ones((1, 1)))
    assert np.array_equal(subject_offsets(np.array([-1.7]), [12.0]), np.zeros(1))


def test_a_level_per_epoch_is_refused() -> None:
    """A level free at every epoch is exactly collinear with the subject's own
    coarsest-scale column, so the coding cannot hold one."""
    with pytest.raises(ValueError, match="one row count per subject"):
        subject_deviations(np.ones((3, 12)))
    with pytest.raises(ValueError, match="one row count per subject"):
        subject_deviations([])


def test_a_subject_that_brings_no_rows_is_refused() -> None:
    with pytest.raises(ValueError, match="brings 0 rows"):
        subject_deviations([40.0, 0.0, 12.0])


def test_the_contrasts_and_the_row_counts_have_to_agree() -> None:
    with pytest.raises(ValueError, match="carries 3 coefficients"):
        subject_offsets(np.zeros(2), [1.0, 2.0, 3.0])


def test_a_roster_position_off_the_row_counts_is_refused() -> None:
    counts = np.array([5.0, 6.0, 7.0])
    assert LevelCoding(1, counts).S == 3
    assert np.allclose(LevelCoding(1, counts).row, [1.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="not one of the 3"):
        LevelCoding(3, counts)


# ---------------------------------------------------------------------------
# Identification
# ---------------------------------------------------------------------------


def test_the_level_columns_are_independent_and_S_plus_one_columns_are_not() -> None:
    """
    A pooled intercept beside S indicators leaves the population intercept free.

    Adding a constant to the intercept and taking it off every indicator changes
    no predictor and no likelihood, so the population one subject is measured
    against is whatever the solver happened to reach.
    """
    counts = np.array([120.0, 90.0, 75.0])
    who = np.repeat(np.arange(3), 40)

    coding = subject_deviations(counts)
    deviation = coding[who]
    assert np.linalg.matrix_rank(deviation) == 3

    naive = np.concatenate([np.ones((who.size, 1)), np.eye(3)[who]], axis=1)
    assert naive.shape[1] == 4
    assert np.linalg.matrix_rank(naive) == 3
    null = np.array([1.0, -1.0, -1.0, -1.0])
    assert np.allclose(naive @ null, 0.0)
    beta = np.array([-2.0, 0.4, 0.0, -0.4])
    assert np.allclose(naive @ beta, naive @ (beta + 7.5 * null))


def test_two_starts_reach_one_population_intercept() -> None:
    """The weights are re-formed at the current level every sweep, so a start
    that is wrong by orders of magnitude is a different path to the same fit."""
    cohort = _cohort([0.8, 0.2, -0.4, -0.6])
    rows = level_rows(cohort, BUDGET)

    reached = []
    for base, offsets, weight in [
        (-2.0, [0.0, 0.0, 0.0, 0.0], 0.0),
        (3.0, [-2.0, 1.5, 0.7, -0.2], -1.3),
    ]:
        params = FineParams(n_labels=2, n_subjects=cohort.S)
        params.base[COUNT][0] = base
        params.offset[COUNT][:, 0] = offsets
        params.coef[(0, COUNT)][EDGE] = weight
        fit = SubjectLevelFit(
            _graph(), params, BUDGET, targets=[0], max_sweeps=400, tol=1e-13
        )
        _counts, history = fit.fit(cohort, rows)
        reached.append((params, history))

    (first, one), (second, two) = reached
    assert abs(first.base[COUNT][0] - second.base[COUNT][0]) < 1e-5
    assert np.abs(first.offset[COUNT][:, 0] - second.offset[COUNT][:, 0]).max() < 1e-5
    assert abs(first.coef[(0, COUNT)][EDGE] - second.coef[(0, COUNT)][EDGE]) < 1e-5
    assert one[-1] == pytest.approx(two[-1], abs=1e-8)
    # The offsets the two agree on are the constrained ones, not a shifted pair.
    assert float(rows[:, 0] @ first.offset[COUNT][:, 0]) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# The widened design, formed and unformed
# ---------------------------------------------------------------------------


def _one_design(cohort: Cohort, graph: TypedGraph, target: int = 0):
    """One subject's design, its risk set and a set of weights to form it under."""
    data = cohort.open(cohort.subjects[1])
    sources = cohort.sources(data)
    ladder = ScaleLadder.from_sources(sources, data.observed)
    design = build_design(graph, target, COUNT, sources, ladder, EMPTY, budget=BUDGET)
    X1, Y1 = data.get_X(1), data.get_Y(1)
    at_risk = usable_slots(X1, Y1, fit_rows(data.observed, BUDGET, sources))
    weights = pg_update(np.full(T, -2.0))
    return design, X1[:, target], weights, at_risk[:, target]


def test_the_unformed_level_block_is_the_formed_one() -> None:
    """The block is never laid down in the fit, and the totals it is formed from
    give back the matrices of the design that would have been."""
    cohort = _cohort([0.3, 0.0, -0.3])
    design, x, weights, at_risk = _one_design(cohort, _graph())
    coding = LevelCoding(1, level_rows(cohort, BUDGET)[:, 0])

    formed = level_blocks(design, coding)
    assert [block.name for block in formed] == ["level", "scale 2"]
    assert formed[0].width == 3
    assert np.array_equal(formed[0].columns[0], coding.row)
    assert np.array_equal(formed[0].columns[-1], coding.row)

    A, b = level_normal_equations(x, design.blocks[1:], weights, coding.row, at_risk)
    reference = normal_equations(x, formed, weights, at_risk, ridge=0.0)
    assert np.allclose(A, reference[0], rtol=1e-10, atol=1e-8)
    assert np.allclose(b, reference[1], rtol=1e-10, atol=1e-8)


def test_a_level_row_of_one_is_the_pooled_intercept() -> None:
    cohort = _cohort([0.3, 0.0, -0.3])
    design, x, weights, at_risk = _one_design(cohort, _graph())
    A, b = level_normal_equations(
        x, design.blocks[1:], weights, np.array([1.0]), at_risk
    )
    reference = normal_equations(x, design.blocks, weights, at_risk, ridge=0.0)
    assert np.allclose(A, reference[0], rtol=1e-10, atol=1e-8)
    assert np.allclose(b, reference[1], rtol=1e-10, atol=1e-8)


def test_a_target_with_no_parents_is_the_level_block_alone() -> None:
    cohort = _cohort([0.3, 0.0, -0.3])
    design, x, weights, at_risk = _one_design(cohort, _graph(), target=1)
    assert design.blocks[1:] == []
    coding = LevelCoding(0, level_rows(cohort, BUDGET)[:, 1])
    A, b = level_normal_equations(x, [], weights, coding.row, at_risk)
    assert A.shape == (3, 3)
    assert b.shape == (3,)
    reference = normal_equations(
        x, level_blocks(design, coding), weights, at_risk, ridge=0.0
    )
    assert np.allclose(A, reference[0], rtol=1e-10, atol=1e-8)


def test_the_row_set_is_the_declared_budget_and_never_the_graph() -> None:
    """Row counts read off the graph would move every time an edge landed, and
    the weights of the constraint would then depend on the model being scored."""
    cohort = _cohort([0.3, 0.0, -0.3])
    empty = TypedGraph(label_names=list(LABELS))
    counts = level_rows(cohort, BUDGET)
    assert counts.shape == (3, 2)
    assert (counts > 0).all()
    data = cohort.open("p00")
    sources = cohort.sources(data)
    at_risk = usable_slots(
        data.get_X(1), data.get_Y(1), fit_rows(data.observed, BUDGET, sources)
    )
    assert np.array_equal(counts[0], at_risk.sum(axis=0))
    # The graph the counts were taken beside says nothing about them.
    assert np.array_equal(counts, level_rows(cohort, BUDGET))
    assert empty.get_parents(0, COUNT) == set()


# ---------------------------------------------------------------------------
# Zero offsets reproduce the pooled fit
# ---------------------------------------------------------------------------


def test_no_participant_axis_leaves_the_predictor_where_it_was() -> None:
    """A roster of no subjects gives a (0, L) offset and every fit that exists
    reads the numbers it always read."""
    params = FineParams(n_labels=2)
    assert params.n_subjects == 0
    assert params.offset[COUNT].shape == (0, 2)
    assert params.offset[DURATION].shape == (0, 2)
    assert params.level(COUNT) is params.base[COUNT]
    assert params.level(COUNT, 0) is params.base[COUNT]

    cohort = _cohort([0.0])
    data = cohort.open("p00")
    sources = cohort.sources(data)
    params.base[COUNT][:] = [-2.0, -1.5]
    plain = linear_predictor(params, _graph(), sources, COUNT)

    axis = FineParams(n_labels=2, n_subjects=1)
    axis.base[COUNT][:] = [-2.0, -1.5]
    assert np.array_equal(axis.level(COUNT, 0), params.base[COUNT])
    assert np.array_equal(
        linear_predictor(axis, _graph(), sources, COUNT, axis.offset[COUNT][0]), plain
    )


def test_identical_records_leave_every_deviation_at_exactly_zero() -> None:
    """
    Four copies of one record carry four times one record's matrices.

    Every deviation column then contributes the same total to its own subject
    and to the reference, and the two cancel to the last bit, so the level block
    decouples and the pooled intercept and the edge coefficient are the ones a
    single intercept would have reached.
    """
    cohort = _cohort([0.5] * 4, seeds=[7] * 4)
    rows = level_rows(cohort, BUDGET)
    assert len(set(rows[:, 0].tolist())) == 1

    params = _params(cohort)
    fit = SubjectLevelFit(_graph(), params, BUDGET, targets=[0], max_sweeps=1)
    sweep = fit.accumulate(cohort, rows)
    beta = fit.solve(sweep, rows)[0]
    assert beta[1:4].tolist() == [0.0, 0.0, 0.0]
    assert params.offset[COUNT][:, 0].tolist() == [0.0, 0.0, 0.0, 0.0]

    one = _cohort([0.5], seeds=[7])
    single = FineParams(n_labels=2, n_subjects=1)
    single.base[COUNT][:] = -2.0
    alone = SubjectLevelFit(_graph(), single, BUDGET, targets=[0], max_sweeps=1)
    counts = level_rows(one, BUDGET)
    solo = alone.accumulate(one, counts)
    assert np.allclose(sweep.A[0][0, 0], 4.0 * solo.A[0][0, 0], rtol=1e-12)
    assert np.allclose(sweep.b[0][0], 4.0 * solo.b[0][0], rtol=1e-12)
    assert sweep.loglik == pytest.approx(4.0 * solo.loglik, rel=1e-12)

    # The deviation columns carry nothing, so the rest of the system is the
    # pooled one and the same start reaches the same intercept and coefficient.
    pooled, _residual, _value = _stream_solve(
        cohort, _graph(), 0, lambda i: [1.0], sweeps=1, start=np.array([-2.0, 0.0])
    )
    assert beta[0] == pytest.approx(pooled[0], rel=1e-12)
    assert beta[4] == pytest.approx(pooled[1], rel=1e-12)


# ---------------------------------------------------------------------------
# The stream
# ---------------------------------------------------------------------------


def test_a_roster_the_parameters_do_not_match_is_refused() -> None:
    cohort = _cohort([0.0, 0.0, 0.0])
    params = FineParams(n_labels=2, n_subjects=2)
    fit = SubjectLevelFit(_graph(), params, BUDGET)
    assert fit.targets == [0, 1]
    assert fit.index.placed == {}
    with pytest.raises(ValueError, match="2 offset rows for a cohort of 3"):
        fit.accumulate(cohort, level_rows(cohort, BUDGET))


def test_a_target_that_came_back_over_two_parent_sets_is_refused() -> None:
    sweep = LevelSweep()
    sweep.add(0, np.eye(2), np.zeros(2), (EDGE,))
    sweep.add(0, np.eye(2), np.zeros(2), (EDGE,))
    assert np.allclose(sweep.A[0], 2.0 * np.eye(2))
    with pytest.raises(ValueError, match="the cohort holds one graph"):
        sweep.add(0, np.eye(2), np.zeros(2), ())


def test_the_log_likelihood_cannot_fall() -> None:
    """Each solve maximizes a tangent lower bound that touches the exact
    log-likelihood at the coefficients the pass ran at."""
    cohort = _cohort([0.9, 0.3, -0.4, -0.8])
    params = _params(cohort)
    fit = SubjectLevelFit(_graph(), params, BUDGET, targets=[0], max_sweeps=2)
    _rows, history = fit.fit(cohort)
    assert len(history) == 2
    assert history[1] > history[0]

    params = _params(cohort)
    fit = SubjectLevelFit(_graph(), params, BUDGET, targets=[0], max_sweeps=40)
    rows, history = fit.fit(cohort)
    assert len(history) < 40
    assert all(b >= a - 1e-9 for a, b in zip(history, history[1:]))
    assert float(rows[:, 0] @ params.offset[COUNT][:, 0]) == pytest.approx(
        0.0, abs=1e-6
    )


def test_a_subject_that_never_exercises_the_label_takes_the_lowest_level() -> None:
    """A structural zero has a level the ridge keeps finite, and it stays inside
    the constraint with the rest of the roster."""
    cohort = _cohort([0.6, 0.0, -0.6, -40.0])
    params = _params(cohort)
    fit = SubjectLevelFit(_graph(), params, BUDGET, targets=[0], max_sweeps=40)
    rows, _history = fit.fit(cohort)

    offsets = params.offset[COUNT][:, 0]
    assert np.isfinite(offsets).all()
    assert int(offsets.argmin()) == 3
    assert offsets[3] < offsets[:3].min() - 1.0
    assert float(rows[:, 0] @ offsets) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# The regime path
# ---------------------------------------------------------------------------


def test_a_chain_of_two_states_cannot_hold_three_levels() -> None:
    """
    A chain of M states holds M distinct levels per label; a roster holds S.

    The chain is granted its best possible posterior here, one state pinned over
    each subject's whole record, and every assignment of three subjects to two
    states is tried.  The best of them still leaves a pair sharing one level,
    which the level block does not.
    """
    cohort = _cohort([0.9, 0.0, -0.9], seeds=[11, 12, 13])
    graph = _graph()
    rows = level_rows(cohort, BUDGET)

    _beta, level_residual, level_value = _stream_solve(
        cohort, graph, 0, lambda i: LevelCoding(i, rows[:, 0]).row
    )
    assert np.abs(level_residual).max() < 1e-5

    best = None
    for assignment in itertools.product([0, 1], repeat=3):
        if len(set(assignment)) < 2:
            continue
        _b, residual, value = _stream_solve(
            cohort, graph, 0, lambda i, a=assignment: np.array([1.0, float(a[i])])
        )
        reach = float(np.abs(residual).max())
        best = (reach, value) if best is None or reach < best[0] else best
    assert best[0] > 1e-2
    assert level_value > best[1]

    _p, pooled_residual, pooled_value = _stream_solve(
        cohort, graph, 0, lambda i: np.array([1.0])
    )
    assert float(np.abs(pooled_residual).max()) > best[0]
    assert level_value > pooled_value


def test_the_level_costs_S_minus_one_free_numbers_per_label() -> None:
    cohort = _cohort([0.4, 0.0, -0.4])
    params = _params(cohort)
    fit = SubjectLevelFit(_graph(), params, BUDGET, targets=[0], max_sweeps=1)
    rows = level_rows(cohort, BUDGET)
    beta = fit.solve(fit.accumulate(cohort, rows), rows)[0]

    # One intercept, S-1 deviations and one coefficient per edge.
    assert beta.size == cohort.S + 1
    assert params.offset[COUNT].shape == (cohort.S, 2)
    assert params.offset[DURATION].shape == (cohort.S, 2)
    # The S offsets carry S-1 numbers, since the constraint fixes the last.
    assert subject_offsets(np.zeros(cohort.S), rows[:, 0]).tolist() == [0.0, 0.0, 0.0]
    assert np.linalg.matrix_rank(subject_deviations(rows[:, 0])[:, 1:]) == cohort.S - 1


# ---------------------------------------------------------------------------
# The benchmark's own walking count trend
# ---------------------------------------------------------------------------

BENCHMARK = Path(__file__).resolve().parents[1] / "data" / "dynamic_patterns"
BENCH_LABELS = ["sleep", "walking"]
BENCH_HOURS, BENCH_DAYS, BENCH_WEEKS = 2184, 91, 13
BENCH_BUDGET = {2: 1, 3: 0}
# The declared trend, three walking episodes per week over weeks 1 to 12, reaches
# the design as a week-scale calendar column read at lag zero.  The declared
# carry-over reads yesterday's sleep duration.
TREND = TypedEdge(CALENDAR, 3, 0, LEVEL, 0)
CARRY = TypedEdge(OBSERVED, 2, 0, DURATION, 1)
WEEK_INDEX = np.arange(BENCH_WEEKS, dtype=float)[:, None]


@pytest.fixture(scope="module")
def benchmark():
    """The benchmark's 30 subjects on an hour slot, sleep and walking only."""
    if not BENCHMARK.is_dir():
        pytest.skip("the dynamic_patterns benchmark is not in the tree")
    by_subject = read_subject_event_log(str(BENCHMARK / "event_log.tsv"))
    axis = sorted({e.label for events in by_subject.values() for e in events})
    keep = [axis.index(name) for name in BENCH_LABELS]
    records = {}
    for person, data, _counts in build_population(
        by_subject,
        axis,
        VERBATIM,
        slot_minutes=60,
        branching=(24, 7),
        origin=BENCHMARK_ORIGIN,
    ):
        records[person] = MultiScaleData(
            hierarchy=data.hierarchy,
            label_names=list(BENCH_LABELS),
            X=[array[:, keep].copy() for array in data.X],
            Y=[array[:, keep].copy() for array in data.Y],
            C=list(data.C),
            observed=data.observed.copy(),
        )
    names = tuple(sorted(records))
    return Cohort(
        subjects=names,
        hierarchy=records[names[0]].hierarchy,
        label_names=list(BENCH_LABELS),
        reader=lambda name: records[name],
        masks=np.stack([records[name].observed for name in names]),
        calendar_columns=[
            np.zeros((BENCH_HOURS, 0)),
            np.zeros((BENCH_DAYS, 0)),
            WEEK_INDEX,
        ],
    )


def _between_share(rows: np.ndarray) -> float:
    """How much of a column's variance sits between subjects and not inside one."""
    grand = rows.mean()
    total = float(((rows - grand) ** 2).sum())
    return float(((rows.mean(axis=1) - grand) ** 2).sum() * rows.shape[1] / total)


def _bench_index() -> CalendarIndex:
    return CalendarIndex(
        placed={(3, 0): ("week", "week_index")}, widths={1: 0, 2: 0, 3: 1}
    )


def _bench_graph() -> TypedGraph:
    graph = TypedGraph(label_names=list(BENCH_LABELS), calendar_names=["week"])
    graph.add_edge(TREND, 1, COUNT)
    graph.add_edge(CARRY, 1, COUNT)
    return graph


def test_the_benchmark_cohort_is_thirty_subjects_on_one_shared_calendar(
    benchmark,
) -> None:
    assert benchmark.S == 30
    assert list(benchmark.hierarchy.H) == [BENCH_HOURS, BENCH_DAYS, BENCH_WEEKS]
    # One window, twelve whole weeks of it, and week 13 is horizon padding.
    assert len({row.tobytes() for row in benchmark.masks}) == 1
    assert int(benchmark.masks[0].sum()) == 2016
    assert benchmark.masks[0][:2016].all()
    assert not benchmark.masks[0][2016:].any()


def test_the_week_trend_column_is_orthogonal_to_every_subject_constant(
    benchmark,
) -> None:
    """
    Centering leaves every column with weighted mean zero over each record.

    The parent of the coarsest scale is the whole horizon, so what comes off the
    week column is the mean over the weeks the record covers, and the inner
    product of what is left with any per-subject constant is zero under the
    counting measure of the declared rows.
    """
    data = benchmark.open(benchmark.subjects[0])
    sources = benchmark.sources(data)
    ladder = ScaleLadder.from_sources(sources, data.observed)
    centered = center_at_scale(WEEK_INDEX[:, 0], ladder, 3, 0)
    # Weeks 1 to 12 are the observed ones, so 5.5 is the mean that comes off.
    assert np.allclose(centered[:12], np.arange(12) - 5.5)

    at_finest = centered[benchmark.ancestors[2]]
    for name in benchmark.subjects:
        record = benchmark.open(name)
        rows = fit_rows(record.observed, BENCH_BUDGET, sources)
        assert float(at_finest[rows].sum()) == 0.0


def test_centering_takes_the_whole_between_subject_share_of_a_data_column(
    benchmark,
) -> None:
    """The response is what the level block is for; the feature is what it must
    leave alone."""
    data = benchmark.open(benchmark.subjects[0])
    ladder = ScaleLadder.from_sources(benchmark.sources(data), data.observed)

    lagged, response = [], []
    for name in benchmark.subjects:
        record = benchmark.open(name)
        sleep = np.asarray(record.Y[1][:, 0], dtype=float)
        column = np.zeros_like(sleep)
        column[1:] = np.log1p(sleep[:-1])
        lagged.append(column)
        response.append(np.asarray(record.X[1][:, 1], dtype=float))
    lagged = np.stack(lagged)
    response = np.stack(response)

    watched = 84
    assert _between_share(lagged[:, :watched]) > 0.1
    centered = np.stack([center_at_scale(row, ladder, 2, 1) for row in lagged])
    assert _between_share(centered[:, :watched]) < 1e-25
    # The walking daily count carries a fifth of its variance between subjects.
    assert round(_between_share(response[:, :watched]), 3) == 0.199


def test_the_walking_trend_survives_the_participant_axis(benchmark) -> None:
    """
    The declared trend is where the pooled fit left it once the levels are free.

    The level block and the trend column are orthogonal by construction, so the
    29 free numbers buy the between-subject spread and take nothing off the
    week-scale slope the manifest declares.
    """
    graph = _bench_graph()
    index = _bench_index()
    rows = level_rows(benchmark, BENCH_BUDGET)

    params = FineParams(n_labels=2, n_subjects=benchmark.S)
    fit = SubjectLevelFit(
        graph, params, BENCH_BUDGET, targets=[1], index=index, max_sweeps=120, tol=1e-11
    )
    _counts, history = fit.fit(benchmark, rows)
    assert all(b >= a - 1e-6 for a, b in zip(history, history[1:]))

    pooled, _residual, pooled_value = _stream_solve(
        benchmark, graph, 1, lambda i: np.array([1.0]), budget=BENCH_BUDGET, index=index
    )
    order = [
        c.edge
        for c in build_design(
            graph,
            1,
            COUNT,
            benchmark.sources(benchmark.open(benchmark.subjects[0])),
            ScaleLadder.from_sources(
                benchmark.sources(benchmark.open(benchmark.subjects[0])),
                benchmark.masks[0],
            ),
            index,
            budget=BENCH_BUDGET,
        ).columns
    ]
    pooled_trend = float(pooled[1 + order.index(TREND)])
    trend = params.coef[(1, COUNT)][TREND]

    # The declared direction is increasing and the slope is where it was.
    assert trend > 0.0
    assert pooled_trend > 0.0
    assert abs(trend - pooled_trend) / pooled_trend < 0.05

    # The offsets carry a real between-subject spread, so the two are not equal
    # because the axis is idle.
    assert params.offset[COUNT][:, 1].std() > 0.4
    assert float(rows[:, 1] @ params.offset[COUNT][:, 1]) == pytest.approx(
        0.0, abs=1e-4
    )

    # And the axis pays for itself under a price of half a nat per free number
    # per log row, charged on the pooled row set.
    price = (benchmark.S - 1) * 0.5 * np.log(float(rows[:, 1].sum()))
    assert history[-1] - pooled_value > price


def test_a_level_per_week_would_be_collinear_with_the_trend(benchmark) -> None:
    """
    Which is why the coding holds one number per subject and no more.

    Over one subject's rows the week trend column is exactly a combination of
    that subject's own week indicators, so the two could not be told apart; a
    single constant per subject leaves the whole column standing.
    """
    data = benchmark.open(benchmark.subjects[0])
    sources = benchmark.sources(data)
    ladder = ScaleLadder.from_sources(sources, data.observed)
    rows = fit_rows(data.observed, BENCH_BUDGET, sources)
    column = center_at_scale(WEEK_INDEX[:, 0], ladder, 3, 0)[benchmark.ancestors[2]]

    weekly = np.eye(BENCH_WEEKS)[benchmark.ancestors[2]][rows]
    fitted = weekly @ np.linalg.lstsq(weekly, column[rows], rcond=None)[0]
    assert np.abs(fitted - column[rows]).max() < 1e-9

    constant = np.ones((int(rows.sum()), 1))
    left = (
        column[rows] - constant @ np.linalg.lstsq(constant, column[rows], rcond=None)[0]
    )
    assert np.abs(left).max() == pytest.approx(np.abs(column[rows]).max())


def test_the_level_block_of_the_benchmark_is_twenty_nine_free_numbers(
    benchmark,
) -> None:
    """The cost the participant axis charges, in the currency a price reads."""
    rows = level_rows(benchmark, BENCH_BUDGET)
    coding = subject_deviations(rows[:, 1])
    assert coding.shape == (30, 30)
    assert np.linalg.matrix_rank(coding[:, 1:]) == 29

    params = FineParams(n_labels=60, n_subjects=30)
    assert params.offset[COUNT].shape == (30, 60)
    assert params.offset[DURATION].shape == (30, 60)
    assert sum(a.size for a in params.offset.values()) == 3600
    # A subject's slots carry the pooled intercept plus that subject's row.
    params.base[COUNT][:] = -3.0
    params.offset[COUNT][7, :] = 0.5
    assert np.allclose(params.level(COUNT, 7), -2.5)
    assert np.allclose(params.level(COUNT, 8), -3.0)


def test_the_level_block_never_widens_the_posterior_covariance(benchmark) -> None:
    """
    A formed level block would carry the (T, P, P) covariance with it.

    At the benchmark's shape that is 30 columns where the design has 3, so the
    array `stack_design` allocates grows by two orders of magnitude and the fit
    stops streaming.  The unformed path allocates none of it.
    """
    data = benchmark.open(benchmark.subjects[0])
    sources = benchmark.sources(data)
    ladder = ScaleLadder.from_sources(sources, data.observed)
    design = build_design(
        _bench_graph(), 1, COUNT, sources, ladder, _bench_index(), budget=BENCH_BUDGET
    )
    rest = design.blocks[1:]
    assert sum(block.width for block in rest) == 2
    assert all(isinstance(block, DesignBlock) for block in rest)

    coding = LevelCoding(0, level_rows(benchmark, BENCH_BUDGET)[:, 1])
    formed = level_blocks(design, coding)
    assert sum(block.width for block in formed) == 32
    # The formed block is a view of one row and holds no storage of its own.
    assert formed[0].columns.base is not None
    assert formed[0].columns.shape == (ladder.n_slots, 30)
