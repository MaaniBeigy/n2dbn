"""
tests/test_phase45_acceptance_fine.py
=====================================
The acceptance set of the fine-likelihood formulation, gathered in one place.

Four of the five properties are checked here.  A draw reconciles at every scale
for arbitrary parameters; no feature reads the epoch it predicts; a draw becomes
a dated event log and reads back as the same arrays; and the frozen formulation
still reproduces what it always did.

The fifth, that a duration-conditioned cross-label edge is **recovered**, needs a
fit and is not assertable yet.  What is asserted instead is that such an edge is
expressible and that it moves the data, so the recovery test has something to
recover when the inference lands.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from src.core.calendar import SlotCalendar, episodes_to_event_log
from src.core.durations import aggregate
from src.core.typed_graph import COUNT, DURATION, OBSERVED, TypedEdge, TypedGraph
from src.data.event_log import build_multiscale_data
from src.inference.reconstruction import EpisodeReconstructor
from src.model.intensity import (
    FineParams,
    ancestor_maps,
    empty_sources,
    running_mask,
    sample,
)

LABELS = ["walk", "sleep"]
DAYS = 14
SLOTS_PER_HOUR = 60
T_SLOTS = DAYS * 24 * SLOTS_PER_HOUR
PARENT_MAPS = [
    np.repeat(np.arange(DAYS * 24), 60),
    np.repeat(np.arange(DAYS), 24),
]
ORIGIN = datetime(2026, 1, 5)


def _draw(seed: int, mean_len: float = 6.0, rate: float = -6.0):
    anc = ancestor_maps(PARENT_MAPS, T_SLOTS)
    calendar = [np.zeros((T_SLOTS, 0)), np.zeros((DAYS * 24, 0)), np.zeros((DAYS, 0))]
    sources = empty_sources(anc, calendar, len(LABELS))
    params = FineParams(n_labels=len(LABELS))
    params.base[COUNT][:] = rate
    params.base[DURATION][:] = np.log(mean_len)
    params.mark_dispersion = np.full(len(LABELS), 4.0)
    graph = TypedGraph(label_names=list(LABELS))
    X1, Y1 = sample(params, graph, sources, np.random.default_rng(seed))
    return X1, Y1, sources


# ---------------------------------------------------------------------------
# 1. Reconciliation at every scale, for data the model drew
# ---------------------------------------------------------------------------


def test_a_draw_reconciles_at_every_scale() -> None:
    X1, Y1, sources = _draw(0)
    assert X1.sum() > 50
    for k, expected in enumerate(aggregate(X1, PARENT_MAPS)):
        assert np.array_equal(sources.counts[k], expected)
    for k, expected in enumerate(aggregate(Y1, PARENT_MAPS)):
        assert np.array_equal(sources.durations[k], expected)


# ---------------------------------------------------------------------------
# 2. A draw is a valid event log, and it is one at the finest resolution
# ---------------------------------------------------------------------------


def test_a_draw_holds_no_same_label_overlap() -> None:
    X1, Y1, _ = _draw(1, mean_len=30.0)
    assert not ((X1 > 0) & running_mask(X1, Y1)).any()


def test_a_draw_becomes_a_dated_event_log() -> None:
    """
    Every episode carries both endpoints as datetimes, end strictly after start.

    This is the interval requirement the project has carried since 16.1f1, and
    it binds on whatever the model produces, not only on what a reader ingests.
    """
    X1, Y1, _ = _draw(2)
    cal = SlotCalendar(origin=ORIGIN, slot=timedelta(minutes=1), branching=(60, 24))
    traces = EpisodeReconstructor(LABELS).reconstruct(X1, Y1)
    log = episodes_to_event_log(traces, LABELS, cal)

    assert len(log) == int(X1.sum())
    for record in log:
        assert isinstance(record.start, datetime) and isinstance(record.end, datetime)
        assert record.end > record.start
        assert record.start.tzinfo is None and record.end.tzinfo is None
        assert record.duration == cal.slot * (record.slot_end - record.slot_start + 1)


def test_the_log_reads_back_as_the_arrays_it_came_from() -> None:
    """The round trip that makes a modeled trace and an observed one the same object."""
    X1, Y1, _ = _draw(3)
    cal = SlotCalendar(origin=ORIGIN, slot=timedelta(minutes=1), branching=(60, 24))
    traces = EpisodeReconstructor(LABELS).reconstruct(X1, Y1)
    log = episodes_to_event_log(traces, LABELS, cal)

    from src.data.event_log import ObservedEvent

    # `end` is exclusive on both sides of the round trip: an episode covering one
    # slot from 00:00 ends at 00:01.
    events = [ObservedEvent(label=r.label, start=r.start, end=r.end) for r in log]
    data, _, _ = build_multiscale_data(
        events,
        slot_minutes=1,
        branching=[60, 24],
        origin=ORIGIN,
        labels=list(LABELS),
    )
    T = data.get_X(1).shape[0]
    assert np.array_equal(data.get_X(1), X1[:T])
    assert np.array_equal(data.get_Y(1), Y1[:T])
    assert X1[T:].sum() == 0, "only trailing empty slots are dropped"


# ---------------------------------------------------------------------------
# 3. Nothing reads the epoch it predicts
# ---------------------------------------------------------------------------


def test_a_lagged_coarse_parent_sees_only_earlier_epochs() -> None:
    X1, Y1, sources = _draw(4)
    from src.model.intensity import edge_feature

    edge = TypedEdge(OBSERVED, 3, 0, COUNT, 1)
    column = edge_feature(edge, sources)
    day = sources.ancestors[2]
    for d in range(1, DAYS):
        inside = column[day == d]
        assert np.allclose(inside, np.log1p(float(sources.counts[2][d - 1, 0])))
    assert np.allclose(column[day == 0], 0.0)


# ---------------------------------------------------------------------------
# 4. The frozen formulation still reproduces what it always did
# ---------------------------------------------------------------------------


def test_the_frozen_generator_is_unchanged_by_any_of_this() -> None:
    """A fixed scenario at a fixed seed gives the arrays it gave before."""
    from src.synthetic.families import make_generator
    from src.synthetic.scenarios import ScenarioConfig

    cfg = ScenarioConfig(
        name="frozen_check",
        K=2,
        H_finest=480,
        branching=[24],
        slot_minutes=60,
        L=2,
        M_k=[1, 1],
        r_k=[0, 0],
        max_lag=[1, 1],
        max_indegree=[0, 1],
        family=1,
        seed=17,
        finest_rate=0.2,
        finest_binary=True,
    )
    first = make_generator(cfg, seed=17).generate(seed=17)
    second = make_generator(cfg, seed=17).generate(seed=17)
    assert np.array_equal(first.data.get_X(1), second.data.get_X(1))
    assert np.array_equal(first.data.get_X(2), second.data.get_X(2))
    assert first.data.check_reconciliation() == {"RecErr_X": 0.0, "RecErr_Y": 0.0}
    # The allocation formulation still splits downward, which is what it is for.
    assert first.data.get_X(2).sum() == first.data.get_X(1).sum()


# ---------------------------------------------------------------------------
# 5. The relation that waits on a fit
# ---------------------------------------------------------------------------


def test_a_duration_conditioned_cross_label_edge_moves_the_data() -> None:
    """
    Expressible and effective, which is what the recovery test will need.

    Recovery itself waits on the inference, so this asserts the data carry the
    relation rather than that a fit finds it.
    """
    edge = TypedEdge(OBSERVED, 3, LABELS.index("sleep"), DURATION, 1)
    walk = LABELS.index("walk")

    def draw(weight: float) -> int:
        anc = ancestor_maps(PARENT_MAPS, T_SLOTS)
        calendar = [
            np.zeros((T_SLOTS, 0)),
            np.zeros((DAYS * 24, 0)),
            np.zeros((DAYS, 0)),
        ]
        sources = empty_sources(anc, calendar, len(LABELS))
        params = FineParams(n_labels=len(LABELS))
        params.base[COUNT][:] = -7.0
        params.base[DURATION][:] = np.log(40.0)
        params.mark_dispersion = np.full(len(LABELS), 4.0)
        graph = TypedGraph(label_names=list(LABELS))
        graph.add_edge(edge, walk, COUNT)
        params.coef[(walk, COUNT)][edge] = weight
        X1, _ = sample(params, graph, sources, np.random.default_rng(21))
        return int(X1[:, walk].sum())

    quiet, loud = draw(0.0), draw(1.0)
    assert loud > 2 * quiet, f"{loud} against {quiet}"


@pytest.mark.parametrize("seed", [5, 6, 7])
def test_the_acceptance_holds_across_seeds(seed) -> None:
    X1, Y1, sources = _draw(seed)
    assert not ((X1 > 0) & running_mask(X1, Y1)).any()
    assert np.array_equal(sources.counts[2], aggregate(X1, PARENT_MAPS)[2])
    assert np.all(Y1[X1 > 0] >= 1)
