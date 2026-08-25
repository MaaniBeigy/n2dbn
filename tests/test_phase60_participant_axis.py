"""
tests/test_phase60_participant_axis.py
======================================
The participant axis as a container, and the pooled fit as a stream.

Four properties carry this phase.

Residency.  A cohort holds the ladder, the label axis, the calendar blocks and
the (S, H_1) mask, and nothing else per subject.  Every subject's counts and
durations arrive through the reader and are dropped before the next one opens,
which the tests below watch through weak references to the arrays themselves.

Additivity.  Every statistic the pooled fit accumulates is a sum over rows, so a
cohort of S copies of one record carries S times the statistic one copy carries,
and a cohort of one reproduces the single-subject fit number for number.  The
quantities that are not sums are charged once: the ridge, the edge price and
every inverse.

The participant mean.  A population count is a mean and its denominator is the
participant count of the epoch itself, so an epoch nobody observed is NaN, a
subject absent at one epoch lowers that epoch alone, and the mean reconciles up
the ladder because `check_ladder` holds for every subject.

Reconciliation.  The identity is that one record's coarse total is the sum of
its own fine ones, so it is checked over the stream and one subject at a time.
A pooled array satisfies nothing but the commutativity of summation.
"""

from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest

from src.core.cohort import Cohort, whole_epochs
from src.core.scales import ScaleHierarchy
from src.core.typed_graph import COUNT, DURATION, OBSERVED, TypedEdge, TypedGraph
from src.core.variables import MultiScaleData
from src.inference.bound import fisher_response, segment_sum, start_curvature
from src.learning.fine_design import (
    CalendarIndex,
    ScaleLadder,
    build_design,
    edge_columns,
)
from src.learning.fine_mstep import (
    _RIDGE,
    normal_equations,
    regime_block,
    stack_design,
)
from src.learning.fine_sem import FineSEM, RegimeChain, fit_rows, update_transitions
from src.learning.fine_sstep import BaseFit, candidate_gains
from src.learning.pooled import (
    PooledCandidates,
    PooledChain,
    PooledDesign,
    PooledFit,
    _candidate_moments,
)
from src.model.intensity import FineParams, linear_predictor, usable_slots

# An hourly slot over four weeks, so the ladder is hour and day and a day-scale
# column has 28 distinct values.  Three labels: a target, a source that drives
# it a day later, and a third that is independent of both.
HOURS = 24
DAYS = 28
T = HOURS * DAYS
LABELS = ["walking", "sleep", "snacking"]
DAY_OF = np.repeat(np.arange(DAYS), HOURS).astype(np.int32)

# The day-scale lag this world's candidate space allows, declared once and
# handed to every subject, which is what fixes the row set for every model the
# pooled sum compares.
BUDGET = {2: 2}

TRUE_EDGE = TypedEdge(OBSERVED, 2, 1, COUNT, 1)
CANDIDATES = [
    TRUE_EDGE,
    TypedEdge(OBSERVED, 2, 2, COUNT, 1),
    TypedEdge(OBSERVED, 2, 2, COUNT, 2),
]


def _hierarchy() -> ScaleHierarchy:
    return ScaleHierarchy(K=2, H=[T, DAYS], parent_maps=[DAY_OF, None])


def _arrays(seed: int):
    """One record: the source, an independent third label, and the driven target."""
    rng = np.random.default_rng(seed)
    X1 = np.zeros((T, 3), dtype=np.int64)
    X1[:, 1] = rng.random(T) < 0.25
    X1[:, 2] = rng.random(T) < 0.15
    total = segment_sum(X1[:, 1].astype(float), DAY_OF, DAYS)
    previous = np.concatenate([[0.0], total[:-1]])
    odds = -3.5 + 0.55 * previous[DAY_OF]
    X1[:, 0] = rng.random(T) < 1.0 / (1.0 + np.exp(-odds))
    return X1, (X1 > 0).astype(np.int64)


def _subject(seed: int, observed=None) -> MultiScaleData:
    X1, Y1 = _arrays(seed)
    day_x = segment_sum(X1.astype(float), DAY_OF, DAYS).astype(np.int64)
    day_y = segment_sum(Y1.astype(float), DAY_OF, DAYS).astype(np.int64)
    return MultiScaleData(
        hierarchy=_hierarchy(),
        label_names=list(LABELS),
        X=[X1, day_x],
        Y=[Y1, day_y],
        C=[np.zeros((T, 1)), np.zeros((DAYS, 1))],
        observed=observed,
    )


class _Watcher:
    """
    A reader that remembers, weakly, every finest array it handed out.

    The weak reference is taken on the array and not on the container, since
    `Cohort.open` hands back a container of its own that shares the arrays.  What
    the memory argument is about is the arrays.
    """

    def __init__(self, seeds):
        self.seeds = dict(seeds)
        self.made = []
        self.resident = []

    def __call__(self, name: str) -> MultiScaleData:
        self.resident.append(self.alive)
        data = _subject(self.seeds[name])
        self.made.append(weakref.ref(data.X[0]))
        return data

    @property
    def alive(self) -> int:
        gc.collect()
        return sum(ref() is not None for ref in self.made)


def _cohort(n: int, seeds=None, masks=None, reader=None) -> Cohort:
    names = tuple(f"p{i:02d}" for i in range(n))
    table = dict(zip(names, list(seeds) if seeds is not None else [0] * n))
    return Cohort(
        subjects=names,
        hierarchy=_hierarchy(),
        label_names=list(LABELS),
        reader=reader if reader is not None else (lambda name: _subject(table[name])),
        masks=np.ones((n, T), dtype=bool) if masks is None else masks,
    )


def _model(n_chains: int = 0):
    """An empty graph, a flat intercept and as many day-scale chains as asked."""
    graph = TypedGraph(label_names=list(LABELS))
    params = FineParams(n_labels=3)
    params.base[COUNT][:] = -2.0
    chains = [
        RegimeChain.uniform(scale=2, n_states=2, n_labels=3, n_epochs=DAYS)
        for _ in range(n_chains)
    ]
    for chain in chains:
        chain.effect = np.array([[0.0, 0.0, 0.0], [0.6, -0.4, 0.2]])
    return graph, params, chains


# ---------------------------------------------------------------------------
# The container
# ---------------------------------------------------------------------------


def test_the_roster_and_the_mask_block_fix_the_shape() -> None:
    cohort = _cohort(3)
    assert cohort.S == 3
    assert cohort.K == 2
    assert cohort.L == 3
    assert cohort.n_slots == T
    assert cohort.masks.shape == (3, T)
    # The ancestor maps are the cohort's, derived once and handed on unchanged.
    assert cohort.ancestors is cohort.ancestors
    assert np.array_equal(cohort.ancestors[1], DAY_OF)


def test_an_empty_roster_is_refused() -> None:
    with pytest.raises(ValueError, match="empty denominator"):
        Cohort(
            subjects=(),
            hierarchy=_hierarchy(),
            label_names=list(LABELS),
            reader=_subject,
            masks=np.ones((0, T), dtype=bool),
        )


def test_a_repeated_id_is_refused() -> None:
    with pytest.raises(ValueError, match="repeats an id"):
        _cohort(2, seeds=[0, 0]).__class__(
            subjects=("p00", "p00"),
            hierarchy=_hierarchy(),
            label_names=list(LABELS),
            reader=_subject,
            masks=np.ones((2, T), dtype=bool),
        )


def test_a_roster_out_of_order_is_refused() -> None:
    """Sorting here would permute the roster away from the mask rows."""
    with pytest.raises(ValueError, match="out of order"):
        Cohort(
            subjects=("p01", "p00"),
            hierarchy=_hierarchy(),
            label_names=list(LABELS),
            reader=_subject,
            masks=np.ones((2, T), dtype=bool),
        )


def test_a_mask_block_that_is_not_boolean_is_refused() -> None:
    with pytest.raises(ValueError, match="dtype"):
        _cohort(1, masks=np.ones((1, T), dtype=np.int64))


def test_a_mask_block_of_the_wrong_shape_is_refused() -> None:
    with pytest.raises(ValueError, match="one row per subject"):
        _cohort(2, masks=np.ones((2, T - 1), dtype=bool))


def test_a_subject_whose_mask_half_covers_a_day_is_refused_by_name() -> None:
    masks = np.ones((2, T), dtype=bool)
    masks[1, :5] = False
    with pytest.raises(ValueError, match="subject 'p01'"):
        _cohort(2, masks=masks)


def test_the_calendar_blocks_default_to_none_at_every_scale() -> None:
    cohort = _cohort(1)
    assert [block.shape for block in cohort.calendar_columns] == [(T, 0), (DAYS, 0)]


def test_a_calendar_block_set_that_misses_a_scale_is_refused() -> None:
    with pytest.raises(ValueError, match="calendar blocks"):
        Cohort(
            subjects=("p00",),
            hierarchy=_hierarchy(),
            label_names=list(LABELS),
            reader=_subject,
            masks=np.ones((1, T), dtype=bool),
            calendar_columns=[np.zeros((T, 0))],
        )


def test_opening_an_id_off_the_roster_is_refused() -> None:
    with pytest.raises(ValueError, match="not one of the 1"):
        _cohort(1).open("nobody")


def test_a_subject_on_another_ladder_is_refused() -> None:
    other = ScaleHierarchy(K=2, H=[T // 2, DAYS], parent_maps=[DAY_OF[: T // 2], None])

    def reader(name):
        data = _subject(0)
        data.hierarchy = other
        return data

    with pytest.raises(ValueError, match="came back on a ladder"):
        _cohort(1, reader=reader).open("p00")


def test_a_subject_on_another_label_axis_is_refused() -> None:
    def reader(name):
        data = _subject(0)
        data.label_names = ["walking"]
        return data

    with pytest.raises(ValueError, match="one column index names two"):
        _cohort(1, reader=reader).open("p00")


def test_opening_intersects_the_record_with_the_window() -> None:
    """
    Two different facts, so the window cannot replace the subject's own record.

    The reader says what the record covers and the cohort says what the run
    watched.  A slot needs both, so what comes back is their intersection.
    """
    own = np.ones(T, dtype=bool)
    own[HOURS * 3 : HOURS * 5] = False
    window = np.ones((1, T), dtype=bool)
    window[0, HOURS * 4 :] = False
    cohort = _cohort(1, masks=window, reader=lambda name: _subject(0, observed=own))
    got = cohort.open("p00").observed
    assert np.array_equal(got, own & window[0])
    assert int(got.sum()) == HOURS * 3


def test_opening_a_record_with_no_mask_of_its_own_takes_the_window() -> None:
    window = np.ones((1, T), dtype=bool)
    window[0, HOURS * 4 :] = False
    cohort = _cohort(1, masks=window)
    assert np.array_equal(cohort.open("p00").observed, window[0])


def test_the_sources_reference_the_shared_maps_and_blocks() -> None:
    """A per-subject source object carries only that subject's counts and durations."""
    cohort = _cohort(1)
    data = cohort.open("p00")
    sources = cohort.sources(data)
    assert sources.ancestors is cohort.ancestors
    assert sources.calendar is cohort.calendar_columns
    assert sources.counts[0] is data.X[0]


# ---------------------------------------------------------------------------
# Staggered enrollment, where a window opens mid-week
# ---------------------------------------------------------------------------


def test_a_partly_covered_coarsest_epoch_is_rounded_and_the_loss_reported() -> None:
    masks = np.ones((2, T), dtype=bool)
    masks[0, : HOURS * 2 + 7] = False  # opens seven hours into day two
    rounded, lost = whole_epochs(masks, _hierarchy())
    assert int(lost[0]) == HOURS - 7
    assert int(lost[1]) == 0
    assert not rounded[0, : HOURS * 3].any()
    assert rounded[0, HOURS * 3 :].all()
    # And what comes back is what the cohort accepts.
    assert _cohort(2, masks=rounded).participants(2)[2] == 1


def test_rounding_a_mask_of_whole_epochs_costs_nothing() -> None:
    masks = np.ones((1, T), dtype=bool)
    masks[0, : HOURS * 3] = False
    rounded, lost = whole_epochs(masks, _hierarchy())
    assert np.array_equal(rounded, masks)
    assert int(lost[0]) == 0


# ---------------------------------------------------------------------------
# Residency: one subject's finest arrays at a time
# ---------------------------------------------------------------------------


def test_a_stream_holds_one_subject_at_a_time() -> None:
    watcher = _Watcher({f"p{i:02d}": i for i in range(4)})
    cohort = _cohort(4, reader=watcher)
    inside = [watcher.alive for _, _ in cohort.stream()]
    assert inside == [1, 1, 1, 1]
    assert watcher.alive == 0


def test_a_pooled_sweep_never_holds_the_roster() -> None:
    """
    The assertion that fails the moment the container materializes the cohort.

    `resident` is how many earlier subjects' finest arrays were still alive when
    each subject was opened.  Holding the roster would make it 0, 1, 2, 3.  A
    loop variable naming the record just visited makes it 0, 1, 1, 1, which is
    the shape `stream` leaves and the reason the pooled pass opens the subjects
    inside a frame of its own.
    """
    watcher = _Watcher({f"p{i:02d}": i for i in range(4)})
    cohort = _cohort(4, reader=watcher)
    graph, params, chains = _model(1)
    fit = PooledFit(graph, params, chains, BUDGET, targets=[0])
    fit.accumulate(cohort, step=True)
    assert watcher.resident == [0, 0, 0, 0]
    assert watcher.alive == 0
    assert len(watcher.made) == 4


def test_a_loop_over_the_stream_holds_the_record_it_just_visited() -> None:
    """
    What the pooled pass would cost if it read the cohort the obvious way.

    The generator keeps nothing once it has yielded, and the caller's own
    binding keeps everything, so the reader builds the next subject's finest
    arrays while the previous subject's are still reachable.
    """
    watcher = _Watcher({f"p{i:02d}": i for i in range(4)})
    for _name, _data in _cohort(4, reader=watcher).stream():
        pass

    assert watcher.resident == [0, 1, 1, 1]


def test_the_cohort_stores_one_array_per_subject_and_it_is_the_mask() -> None:
    cohort = _cohort(4, seeds=[0, 1, 2, 3])
    held = [
        value
        for value in vars(cohort).values()
        if isinstance(value, np.ndarray) and value.ndim == 2
    ]
    assert held == [cohort.masks]
    assert cohort.masks.nbytes == 4 * T


# ---------------------------------------------------------------------------
# The participant mean and its denominator
# ---------------------------------------------------------------------------


def _windows() -> np.ndarray:
    """Three subjects on three windows, and two days nobody watched."""
    masks = np.zeros((3, T), dtype=bool)
    masks[0, : HOURS * 7] = True
    masks[1, HOURS * 4 : HOURS * 26] = True
    masks[2, HOURS * 4 : HOURS * 26] = True
    return masks


def test_the_denominator_is_the_participant_count_of_the_epoch() -> None:
    cohort = _cohort(3, seeds=[0, 1, 2], masks=_windows())
    counts = cohort.participants(2)
    assert list(counts[:4]) == [1, 1, 1, 1]
    assert list(counts[4:7]) == [3, 3, 3]
    assert list(counts[7:26]) == [2] * 19
    assert list(counts[26:]) == [0, 0]
    assert cohort.participants(1).shape == (T,)


def test_the_mean_is_over_the_subjects_observed_at_that_epoch() -> None:
    masks = _windows()
    cohort = _cohort(3, seeds=[0, 1, 2], masks=masks)
    mean = cohort.population(2)
    day_counts = [
        segment_sum(_subject(seed).get_X(1).astype(float), DAY_OF, DAYS)
        for seed in (0, 1, 2)
    ]
    # A day only subject zero watched is subject zero's own count.
    assert np.allclose(mean[0], day_counts[0][0])
    # A day all three watched is the mean of three.
    assert np.allclose(mean[5], sum(d[5] for d in day_counts) / 3.0)
    # A day the late pair watched is the mean of two, and subject zero's own
    # count on that day changes nothing.
    assert np.allclose(mean[10], (day_counts[1][10] + day_counts[2][10]) / 2.0)


def test_an_epoch_nobody_watched_is_not_a_number() -> None:
    """A fabricated zero there would read as a recorded absence."""
    cohort = _cohort(3, seeds=[0, 1, 2], masks=_windows())
    mean = cohort.population(2)
    assert np.isnan(mean[26]).all()
    assert np.isnan(mean[27]).all()
    assert not np.isnan(mean[:26]).any()


def test_the_population_mean_reconciles_up_the_ladder() -> None:
    """
    It reconciles because `check_ladder` holds for every subject.

    A coarse epoch is fully observed or fully unobserved per subject, so the
    participant set is constant inside it and the denominator of a parent is the
    denominator of each of its children.
    """
    cohort = _cohort(3, seeds=[0, 1, 2], masks=_windows())
    fine = cohort.population(1)
    coarse = cohort.population(2)
    watched = cohort.participants(2) > 0
    summed = segment_sum(fine[DAY_OF < DAYS], DAY_OF, DAYS)
    assert np.allclose(summed[watched], coarse[watched])


def test_the_mean_reads_the_duration_too() -> None:
    cohort = _cohort(2, seeds=[0, 1])
    mean = cohort.population(2, DURATION)
    day_y = [
        segment_sum(_subject(seed).get_Y(1).astype(float), DAY_OF, DAYS)
        for seed in (0, 1)
    ]
    assert np.allclose(mean, (day_y[0] + day_y[1]) / 2.0)


def test_the_mean_refuses_a_quantity_that_is_neither() -> None:
    with pytest.raises(ValueError, match="expected one of"):
        _cohort(1).population(2, "level")


# ---------------------------------------------------------------------------
# Reconciliation, per subject and over the stream
# ---------------------------------------------------------------------------


def test_reconciliation_is_reported_once_per_subject() -> None:
    cohort = _cohort(3, seeds=[0, 1, 2])
    report = cohort.reconcile()
    assert sorted(report) == ["p00", "p01", "p02"]
    assert all(v == 0.0 for errs in report.values() for v in errs.values())


def test_one_subject_breaking_the_identity_is_named_and_the_others_are_not() -> None:
    def reader(name):
        data = _subject(int(name[1:]))
        if name == "p01":
            data.X[1][3, 0] += 7
        return data

    report = _cohort(3, seeds=[0, 1, 2], reader=reader).reconcile()
    assert report["p01"]["RecErr_X"] == 7.0
    assert report["p00"]["RecErr_X"] == 0.0
    assert report["p02"]["RecErr_X"] == 0.0


# ---------------------------------------------------------------------------
# A cohort of one is the single-subject fit, number for number
# ---------------------------------------------------------------------------


def _reference(cohort: Cohort, chains, params, graph):
    """
    The single-subject quantities, derived from the shipping calls alone.

    The weights are the start term's own curvature and the response is the
    Fisher-scoring one, which is the step the pooled solve takes.
    """
    data = cohort.open("p00")
    sources = cohort.sources(data)
    X1, Y1 = data.get_X(1), data.get_Y(1)
    offset = sum(
        (c.contribution(sources.ancestors) for c in chains),
        np.zeros((T, len(LABELS))),
    )
    eta = linear_predictor(params, graph, sources, COUNT, offset)
    weights = start_curvature(eta)
    response = fisher_response(X1, eta)
    at_risk = usable_slots(X1, Y1, fit_rows(data.observed, BUDGET, sources))
    ladder = ScaleLadder.from_sources(sources, data.observed)
    latent = [
        regime_block(c.gamma, sources.ancestors[c.scale - 1], f"regime k{c.scale}")
        for c in chains
    ]
    design = build_design(
        graph,
        0,
        COUNT,
        sources,
        ladder,
        CalendarIndex(placed={}, widths={}),
        latent,
        BUDGET,
    )
    return design, response[:, 0], weights[:, 0], at_risk[:, 0]


def test_a_cohort_of_one_reproduces_the_single_subject_latent_fit() -> None:
    """Same bound at every step, same posterior, same transitions."""
    cohort = _cohort(1)
    graph, params, chains = _model(1)
    fit = PooledFit(graph, params, chains, BUDGET, targets=[0])
    stats, history = fit.fit(cohort)

    alone_graph, alone_params, alone_chains = _model(1)
    data = cohort.open("p00")
    sources = cohort.sources(data)
    _, fitted, expected = FineSEM(lag_budget=BUDGET).fit(
        data.get_X(1),
        data.get_Y(1),
        alone_params,
        alone_graph,
        sources,
        alone_chains,
        [T, DAYS],
        observed=data.observed,
    )
    assert history.values == expected.values
    assert np.allclose(fit._posteriors["p00"][0].gamma, fitted[0].gamma)
    assert np.allclose(fit._posteriors["p00"][0].xi, fitted[0].xi)
    assert np.allclose(fit.chains[0].log_trans, fitted[0].log_trans)
    assert np.allclose(fit.chains[0].log_init, fitted[0].log_init)
    assert stats.subjects == 1


def test_a_cohort_of_one_reproduces_the_single_subject_statistics() -> None:
    cohort = _cohort(1)
    graph, params, chains = _model(1)
    fit = PooledFit(graph, params, chains, BUDGET, targets=[0])
    stats, _ = fit.fit(cohort)

    design, response, weights, at_risk = _reference(
        cohort, fit._posteriors["p00"], params, graph
    )
    # The matrix the pooled solve inverts is the matrix the search scores, and
    # it is the one the finest-grid solve forms.
    A, _ = normal_equations(response, design.blocks, weights, at_risk, ridge=0.0)
    means, _ = stack_design(design.blocks)
    b = means[at_risk].T @ response[at_risk]
    got = stats.designs[0]
    assert np.allclose(got.A, A)
    assert np.allclose(got.b, b)
    assert np.allclose(
        fit.solve(stats)[0],
        np.linalg.solve(A + _RIDGE * np.eye(A.shape[0]), b),
    )


def test_the_shared_transition_update_is_the_single_subject_one() -> None:
    cohort = _cohort(1)
    graph, params, chains = _model(1)
    fit = PooledFit(graph, params, chains, BUDGET, targets=[0])
    stats = fit.accumulate(cohort, step=True)
    fit.set_transitions(stats)

    alone = fit._posteriors["p00"][0]
    expected = update_transitions(
        RegimeChain(
            scale=2,
            n_states=2,
            effect=alone.effect,
            log_init=alone.log_init,
            log_trans=alone.log_trans,
            gamma=alone.gamma,
            xi=alone.xi,
        )
    )
    assert np.allclose(fit.chains[0].log_trans, expected.log_trans)
    assert np.allclose(fit.chains[0].log_init, expected.log_init)


# ---------------------------------------------------------------------------
# S copies: every statistic S times over, and one graph
# ---------------------------------------------------------------------------


def _accumulated(n: int, chains: int = 1, candidates=None):
    cohort = _cohort(n)
    graph, params, chain_list = _model(chains)
    fit = PooledFit(
        graph, params, chain_list, BUDGET, targets=[0], candidates=candidates
    )
    return fit, cohort, fit.accumulate(cohort, step=True)


def test_every_accumulated_statistic_is_the_roster_times_over() -> None:
    _, _, one = _accumulated(1, candidates={0: CANDIDATES})
    _, _, four = _accumulated(4, candidates={0: CANDIDATES})
    assert four.subjects == 4
    assert four.rows == 4 * one.rows
    assert four.bound == pytest.approx(4.0 * one.bound)
    for name in ("A", "b"):
        np.testing.assert_allclose(
            getattr(four.designs[0], name), 4.0 * getattr(one.designs[0], name)
        )
    for name in ("cross", "diagonal", "target"):
        np.testing.assert_allclose(
            getattr(four.candidates[0], name), 4.0 * getattr(one.candidates[0], name)
        )
    np.testing.assert_allclose(four.chains[0].pairs, 4.0 * one.chains[0].pairs)
    np.testing.assert_allclose(four.chains[0].first, 4.0 * one.chains[0].first)


def test_the_pooled_solve_and_the_transitions_are_unmoved_by_copying() -> None:
    """
    S copies carry S times the evidence and the same answer.

    The ridge and the initial distribution are the two places a roster could
    leak into a maximizer.  The ridge is charged once, so it does not grow with
    S; the initial distribution is a mean over subjects, so it does not either.
    """
    one_fit, _, one = _accumulated(1)
    four_fit, _, four = _accumulated(4)
    one_fit.set_transitions(one)
    four_fit.set_transitions(four)
    np.testing.assert_allclose(
        four_fit.solve(four)[0], one_fit.solve(one)[0], rtol=1e-6
    )
    np.testing.assert_allclose(
        four_fit.chains[0].log_trans, one_fit.chains[0].log_trans
    )
    np.testing.assert_allclose(four_fit.chains[0].log_init, one_fit.chains[0].log_init)


def test_a_candidate_gain_is_the_roster_times_the_one_subject_gain() -> None:
    _, _, one = _accumulated(1, chains=0, candidates={0: CANDIDATES})
    _, _, four = _accumulated(4, chains=0, candidates={0: CANDIDATES})

    def gains(stats):
        design, cand = stats.designs[0], stats.candidates[0]
        return candidate_gains(
            BaseFit(design.A, design.b), cand.cross, cand.diagonal, cand.target
        )

    np.testing.assert_allclose(gains(four), 4.0 * gains(one), rtol=1e-8)


def _searched(n: int, price: float = 20.0):
    cohort = _cohort(n)
    graph, params, chains = _model(0)
    fit = PooledFit(
        graph, params, chains, BUDGET, targets=[0], candidates={0: CANDIDATES}
    )
    fit.solve(fit.accumulate(cohort, step=False))
    reports = fit.search(cohort, price=price, max_edges=3)
    return graph, reports[0]


def test_the_search_over_copies_finds_the_graph_one_copy_finds() -> None:
    alone, one = _searched(1)
    copied, four = _searched(4)
    assert one.accepted == [0]
    assert four.accepted == [0]
    assert alone.edge_set() == copied.edge_set()
    assert sorted(alone.get_parents(0, COUNT)) == [TRUE_EDGE]
    # The price is charged once against the cohort, so four copies score the
    # same edge four times as high and buy the same graph.
    assert four.gains[0] == pytest.approx(4.0 * one.gains[0])


def test_the_pooled_solve_writes_the_coefficient_of_every_edge_it_fitted() -> None:
    """The fitted vector reaches the likelihood keyed by the edge it belongs to."""
    cohort = _cohort(2, seeds=[0, 1])
    graph, params, chains = _model(0)
    graph.add_edge(TRUE_EDGE, 0, COUNT)
    fit = PooledFit(graph, params, chains, BUDGET, targets=[0])
    stats = fit.accumulate(cohort, step=False)
    assert stats.designs[0].edges == (TRUE_EDGE,)
    beta = fit.solve(stats)[0]
    assert params.base[COUNT][0] == pytest.approx(beta[0])
    assert params.coef[(0, COUNT)][TRUE_EDGE] == pytest.approx(beta[1])
    # A driven target buys a positive coefficient on the day it is driven from.
    assert beta[1] > 0.0


def test_the_pooled_loop_stops_when_the_summed_bound_settles() -> None:
    cohort = _cohort(2, seeds=[0, 1])
    graph, params, chains = _model(1)
    fit = PooledFit(graph, params, chains, BUDGET, targets=[0], max_sweeps=20, tol=1e-2)
    _, history = fit.fit(cohort)
    assert len(history.values) < 21
    assert history.steps[0] == "start"


def test_the_search_leaves_the_graph_it_accepted_and_nothing_else() -> None:
    cohort = _cohort(2, seeds=[0, 1])
    graph, params, chains = _model(0)
    fit = PooledFit(
        graph, params, chains, BUDGET, targets=[0], candidates={0: CANDIDATES}
    )
    before = float(params.base[COUNT][0])
    fit.search(cohort, price=1e9, max_edges=3)
    assert graph.n_edges() == 0
    assert float(params.base[COUNT][0]) == before


# ---------------------------------------------------------------------------
# What the accumulators refuse, and the corners of the stream
# ---------------------------------------------------------------------------


def test_two_designs_over_different_parent_sets_cannot_be_added() -> None:
    """One cohort holds one graph, and its columns sit in one order."""
    left = PooledDesign(edges=(TRUE_EDGE,), A=np.eye(2), b=np.zeros(2))
    right = PooledDesign(edges=(), A=np.eye(1), b=np.zeros(1))
    with pytest.raises(ValueError, match="one graph"):
        left.add(right)


def test_the_candidate_and_chain_accumulators_add_entry_by_entry() -> None:
    cand = PooledCandidates(
        cross=np.ones((2, 3)), diagonal=np.ones(2), target=np.ones(2)
    )
    cand.add(
        PooledCandidates(
            cross=np.full((2, 3), 2.0), diagonal=np.full(2, 3.0), target=np.full(2, 4.0)
        )
    )
    assert np.allclose(cand.cross, 3.0)
    assert np.allclose(cand.diagonal, 4.0)
    assert np.allclose(cand.target, 5.0)

    chain = PooledChain(pairs=np.ones((2, 2)), first=np.ones(2))
    chain.add(PooledChain(pairs=np.ones((2, 2)), first=np.full(2, 2.0)))
    assert np.allclose(chain.pairs, 2.0)
    assert np.allclose(chain.first, 3.0)


def test_a_target_with_no_candidates_accumulates_a_design_and_no_more() -> None:
    cohort = _cohort(2, seeds=[0, 1])
    graph, params, chains = _model(0)
    fit = PooledFit(
        graph, params, chains, BUDGET, targets=[0, 1], candidates={0: CANDIDATES}
    )
    stats = fit.accumulate(cohort, step=False)
    assert sorted(stats.designs) == [0, 1]
    assert sorted(stats.candidates) == [0]
    assert stats.chains == []
    # Restricting the stream to one target forms one design.
    assert sorted(fit.accumulate(cohort, step=False, targets=[1]).designs) == [1]


def test_the_candidate_moments_carry_the_latent_columns_too() -> None:
    """
    A latent block has no coarse-grid form, so a candidate meets it at the finest
    grid, which is where `search_gram` puts it too.
    """
    cohort = _cohort(1)
    graph, params, chains = _model(1)
    fit = PooledFit(
        graph, params, chains, BUDGET, targets=[0], candidates={0: CANDIDATES}
    )
    stats = fit.accumulate(cohort, step=True)
    design, x, weights, at_risk = _reference(
        cohort, fit._posteriors["p00"], params, graph
    )
    columns = design.scaled()
    assert stats.candidates[0].cross.shape == (
        len(CANDIDATES),
        len(columns) + design.latent[0].width,
    )
    without = _model(0)
    bare = PooledFit(
        without[0], without[1], [], BUDGET, targets=[0], candidates={0: CANDIDATES}
    )
    flat = bare.accumulate(cohort, step=False)
    assert flat.candidates[0].cross.shape == (len(CANDIDATES), 1)


def _single_epoch_cohort():
    """One record whose coarsest scale holds a single epoch, so it has no pair."""
    hierarchy = ScaleHierarchy(
        K=2, H=[HOURS, 1], parent_maps=[np.zeros(HOURS, dtype=np.int32), None]
    )
    data = MultiScaleData(
        hierarchy=hierarchy,
        label_names=["a"],
        X=[np.zeros((HOURS, 1), dtype=np.int64), np.zeros((1, 1), dtype=np.int64)],
        Y=[np.zeros((HOURS, 1), dtype=np.int64), np.zeros((1, 1), dtype=np.int64)],
        C=[np.zeros((HOURS, 1)), np.zeros((1, 1))],
    )
    return Cohort(
        subjects=("p00",),
        hierarchy=hierarchy,
        label_names=["a"],
        reader=lambda name: data,
        masks=np.ones((1, HOURS), dtype=bool),
    )


def test_a_chain_with_no_transition_to_count_is_left_alone() -> None:
    """
    A chain carrying no pair marginal has no maximizer, so its parameter stands.

    This is the case `update_transitions` returns on, restated over the summed
    counts: a pair count of zero everywhere is a chain nobody counted a
    transition for.
    """
    cohort = _single_epoch_cohort()
    graph = TypedGraph(label_names=["a"])
    chain = RegimeChain.uniform(scale=2, n_states=2, n_labels=1, n_epochs=1)
    before = chain.log_trans.copy()
    fit = PooledFit(graph, FineParams(n_labels=1), [chain], {2: 0}, targets=[0])
    stats = fit.accumulate(cohort, step=False)
    fit.set_transitions(stats)
    assert stats.chains[0].pairs.shape == (2, 2)
    assert float(stats.chains[0].pairs.sum()) == 0.0
    assert np.allclose(fit.chains[0].log_trans, before)


def test_a_degenerate_chain_still_lands_where_the_single_subject_step_lands() -> None:
    """
    A one-epoch posterior carries a flat pair marginal, and both steps take it.

    The pooled maximizer is the single-subject expression over the summed
    counts, so the two agree even where the count is the flat one a chain with
    one epoch produces.
    """
    cohort = _single_epoch_cohort()
    graph = TypedGraph(label_names=["a"])
    chain = RegimeChain.uniform(scale=2, n_states=2, n_labels=1, n_epochs=1)
    fit = PooledFit(graph, FineParams(n_labels=1), [chain], {2: 0}, targets=[0])
    fit.set_transitions(fit.accumulate(cohort, step=True))
    alone = fit._posteriors["p00"][0]
    expected = update_transitions(
        RegimeChain(
            scale=2,
            n_states=2,
            effect=alone.effect,
            log_init=alone.log_init,
            log_trans=alone.log_trans,
            gamma=alone.gamma,
            xi=alone.xi,
        )
    )
    assert np.allclose(fit.chains[0].log_trans, expected.log_trans)
    assert np.allclose(fit.chains[0].log_init, expected.log_init)


def test_the_declared_budget_erodes_every_subject_the_same_way() -> None:
    """
    One budget covers the cohort, since one graph over one label axis is one
    candidate space.

    Two subjects with different gaps lose different rows, and each loses exactly
    what the declared budget takes.  A per-subject budget would let two models
    compared on the pooled sum run over different row totals.
    """
    masks = np.ones((2, T), dtype=bool)
    masks[0, HOURS * 5 : HOURS * 6] = False
    masks[1, HOURS * 5 : HOURS * 7] = False
    cohort = _cohort(2, seeds=[0, 1], masks=masks)

    def rows(budget, edges):
        graph, params, chains = _model(0)
        for edge in edges:
            graph.add_edge(edge, 0, COUNT)
        fit = PooledFit(graph, params, chains, budget, targets=[0])
        assert fit.lag_budget == budget
        return fit.accumulate(cohort, step=False).rows

    # The pooled count is the sum of what each subject's own mask leaves once
    # the declared budget has eroded it.
    expected = 0
    for _, data in cohort.stream():
        sources = cohort.sources(data)
        seen = fit_rows(data.observed, {2: 1}, sources)
        expected += int(usable_slots(data.get_X(1), data.get_Y(1), seen).sum())
    bare = rows({2: 1}, [])
    assert bare == expected

    # Two models of different reach are summed over one row set, since the
    # erosion reads the declared budget and never the graph in hand.
    assert rows({2: 1}, [TRUE_EDGE]) == bare
    wide = rows({2: 2}, [])
    assert wide < bare
    assert rows({2: 2}, CANDIDATES) == wide


def test_the_candidate_moments_are_the_finest_grid_products() -> None:
    """
    A candidate against the intercept, against itself and against the response.

    The three arrays are formed at each column's own resolution, and what they
    have to equal is the finest-grid product over the rows a fit scores.
    """
    cohort = _cohort(1)
    graph, params, chains = _model(0)
    design, x, weights, at_risk = _reference(cohort, [], params, graph)
    sources = cohort.sources(cohort.open("p00"))
    columns = edge_columns(
        CANDIDATES,
        graph,
        sources,
        design.ladder,
        CalendarIndex(placed={}, widths={}),
        BUDGET,
    )
    got = _candidate_moments(design, columns, x, weights, at_risk)

    keep = at_risk.astype(float)
    response = ((x > 0).astype(float) - 0.5) * keep
    for c, column in enumerate(columns):
        fine = column.at_finest(design.ladder)
        assert got.cross[c, 0] == pytest.approx(float((weights * keep * fine).sum()))
        assert got.diagonal[c] == pytest.approx(
            float((weights * keep * fine * fine).sum())
        )
        assert got.target[c] == pytest.approx(float((response * fine).sum()))
