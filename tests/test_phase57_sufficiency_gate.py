"""
tests/test_phase57_sufficiency_gate.py
======================================
The pre-fit sufficiency ledger, wired into the fit path as a per-scale gate.

The ledger judges each scale on the epochs a fit will see against the candidate
edges that scale must choose between, and a refusal marks the one regime the
sweep behind the curve found where a search recovers nothing and returns edges
anyway.  These tests pin four things.  The ledger lands on the training history
of every fit that searches, so a reader who sees an empty scale can tell an
absence of structure from an absence of epochs.  The gate is off unless asked
for, so no stored result moves on its own.  When it is on, a refused scale is
emptied in both the graph and the parameters before its first E-step and its
search never runs, while its E-step and M-step do, so the scales below keep
their coupling.
And the gate reads the same lag budget the search will enumerate, so the two
never disagree about how many candidates a scale has.

The fine formulation types a candidate by its source scale, so the last block
pins that its ladder feeds the ledger the same epoch counts under the same
1-based scale index the columns carry.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np
import pytest

from src.core.calendar import SlotCalendar
from src.core.graph import MultiScaleDBNGraph
from src.core.scales import ScaleHierarchy
from src.core.variables import MultiScaleData
from src.evaluation.sufficiency import (
    data_ledger,
    ledger,
    refused_scales,
    supported_scales,
)
from src.inference.bound import segment_sum
from src.learning.fine_design import ScaleLadder
from src.learning.fine_sstep import ScaledColumn
from src.learning.structural_em import TopDownSEM
from src.model.intensity import FineSources, ancestor_maps
from src.synthetic.families import make_generator
from src.synthetic.scenarios import ScenarioConfig

# A two-scale ladder whose coarse scale holds 12 epochs against 6 candidates,
# which is 2.0 each and under the count at which the sweep recovered anything.
# The finest scale holds 120 against the same 6 and is comfortably above it, so
# one dataset carries both verdicts and the gate has something to leave alone.
LABELS = 3


def _cfg() -> ScenarioConfig:
    return ScenarioConfig(
        name="gate",
        K=2,
        H_finest=120,
        branching=[10],
        L=LABELS,
        M_k=[2, 2],
        r_k=[0, 0],
        max_lag=[2, 1],
        max_indegree=[0, 1],
        zi_level="medium",
        family=2,
        seed=4,
        n_replicates=1,
        edge_strength=10.0,
        regime_strength=3.0,
        finest_rate=0.2,
        finest_binary=True,
        slot_minutes=1,
    )


@pytest.fixture(scope="module")
def dataset():
    cfg = _cfg()
    gen = make_generator(cfg, seed=4)
    ds = gen.generate(seed=4)
    graph = MultiScaleDBNGraph(hierarchy=ds.hierarchy, label_names=gen.label_names)
    for ki, p in enumerate(ds.params):
        for dst, parents in p.parent_sets.items():
            for src, lag in parents:
                graph.add_edge(ki + 1, src, lag, dst)
    return ds, graph


def _start(ds, graph) -> Tuple[list, MultiScaleDBNGraph]:
    """A fresh copy of the starting parameters and graph."""
    return [deepcopy(p) for p in ds.params], deepcopy(graph)


def _searched(sem: TopDownSEM, data, params, graph) -> Tuple[List[int], tuple]:
    """
    The scales the structure step actually ran at, and what the fit returned.

    Watching the call is what separates a scale that was searched and came back
    empty from one that was never searched.
    """
    seen: List[int] = []
    original = sem.sstep.run

    def spy(**kwargs):
        seen.append(int(kwargs["k"]))
        return original(**kwargs)

    sem.sstep.run = spy
    return seen, sem.fit(data, params, graph)


# ---------------------------------------------------------------------------
# The ledger of a dataset
# ---------------------------------------------------------------------------


def test_the_ledger_reads_the_epochs_the_dataset_itself_holds(dataset):
    ds, _graph = dataset
    got = data_ledger(ds.data, [2, 2])
    assert [v.k for v in got] == [1, 2]
    assert [v.n_epochs for v in got] == [120, 12]
    assert [v.n_candidates for v in got] == [6, 6]


def test_a_dataset_carries_one_verdict_per_scale_and_they_can_disagree(dataset):
    ds, _graph = dataset
    got = data_ledger(ds.data, [2, 2])
    assert supported_scales(got) == {1: True, 2: False}
    assert refused_scales(got) == {2}


def test_a_wider_lag_budget_refuses_a_scale_a_narrower_one_supports(dataset):
    """The candidate count is what the epochs are judged against."""
    ds, _graph = dataset
    assert refused_scales(data_ledger(ds.data, [2, 1])) == set()
    assert refused_scales(data_ledger(ds.data, [2, 2])) == {2}


# ---------------------------------------------------------------------------
# The ledger travels with every fit that searches
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("require_support", [False, True])
def test_the_ledger_lands_on_the_history_whether_or_not_it_acts(
    dataset, require_support
):
    ds, graph = dataset
    sem = TopDownSEM(max_outer_iter=1, require_support=require_support, verbose=0)
    _params, _learned, history = sem.fit(*(ds.data,) + _start(ds, graph))
    assert [v.k for v in history.support] == [1, 2]
    assert refused_scales(history.support) == {2}


def test_the_refusal_is_announced_even_when_the_gate_is_off(dataset, capsys):
    """The gate off still owes the reader the arithmetic, so the refusal prints."""
    ds, graph = dataset
    sem = TopDownSEM(max_outer_iter=1, verbose=1)
    sem.fit(*(ds.data,) + _start(ds, graph))
    out = capsys.readouterr().out
    assert "no structural claim is supported at scale 2" in out


def test_a_ladder_the_ledger_supports_is_announced_by_nothing(dataset, capsys):
    ds, graph = dataset
    sem = TopDownSEM(
        max_outer_iter=1, verbose=1, sstep_kwargs=dict(max_lag={1: 2, 2: 1})
    )
    sem.fit(*(ds.data,) + _start(ds, graph))
    assert "no structural claim" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# The gate is off by default
# ---------------------------------------------------------------------------


def test_the_default_fit_searches_the_refused_scale_exactly_as_before(dataset):
    ds, graph = dataset
    sem = TopDownSEM(max_outer_iter=1, do_structure=True, verbose=0)
    seen, (_params, _learned, history) = _searched(sem, ds.data, *_start(ds, graph))
    assert refused_scales(history.support) == {2}
    assert sorted(set(seen)) == [1, 2]


# ---------------------------------------------------------------------------
# The gate, when it is asked for
# ---------------------------------------------------------------------------


def test_a_refused_scale_is_emptied_and_never_searched(dataset):
    ds, graph = dataset
    params, start = _start(ds, graph)
    assert start.n_edges(2) > 0
    sem = TopDownSEM(
        max_outer_iter=1, do_structure=True, require_support=True, verbose=0
    )
    seen, (fitted, learned, _history) = _searched(sem, ds.data, params, start)
    assert sorted(set(seen)) == [1]
    assert learned.n_edges(2) == 0
    assert all(not parents for parents in fitted[1].parent_sets.values())


def test_the_refused_scale_is_emptied_before_its_first_e_step(dataset):
    """
    Clearing the parents up front changes what every later step reads.

    A gated fit that starts from a graph with edges at the refused scale scores
    that scale exactly as an ungated fit that started from an empty one, which
    is only true if the clearing happened before anything read the graph.
    """
    ds, graph = dataset
    params, start = _start(ds, graph)
    gated = TopDownSEM(
        max_outer_iter=1, do_structure=True, require_support=True, verbose=0
    )
    _f, _l, gated_history = gated.fit(ds.data, params, start)

    params, start = _start(ds, graph)
    for ell in range(LABELS):
        start.set_parents(2, ell, set())
        params[1].parent_sets[ell] = set()
    ungated = TopDownSEM(max_outer_iter=1, do_structure=True, verbose=0)
    _f, _l, ungated_history = ungated.fit(ds.data, params, start)

    assert gated_history.log_Z_per_scale[0][1] == ungated_history.log_Z_per_scale[0][1]


def test_a_refused_scale_is_still_fitted_and_still_scores(dataset):
    """Only the search is refused, so the parameters and the evidence stay."""
    ds, graph = dataset
    params, start = _start(ds, graph)
    sem = TopDownSEM(
        max_outer_iter=1, do_structure=True, require_support=True, verbose=0
    )
    fitted, _learned, history = sem.fit(ds.data, params, start)
    coarse_log_Z = history.log_Z_per_scale[-1][1]
    assert np.isfinite(coarse_log_Z) and coarse_log_Z != 0.0
    assert not np.allclose(
        fitted[1].coarse_count.beta0, ds.params[1].coarse_count.beta0
    )


def test_the_structure_first_order_refuses_the_same_scale(dataset):
    ds, graph = dataset
    sem = TopDownSEM(
        max_outer_iter=1,
        do_structure=True,
        structure_first=True,
        require_support=True,
        verbose=0,
    )
    seen, (_f, learned, _h) = _searched(sem, ds.data, *_start(ds, graph))
    assert sorted(set(seen)) == [1]
    assert learned.n_edges(2) == 0


def test_a_fit_that_searches_nothing_is_judged_on_nothing(dataset):
    """
    A fixed graph is the caller's claim, so the ledger has nothing to qualify.

    With the structure step off the fit proposes no graph of its own, the
    starting parents survive at the refused scale, and the ledger stays empty,
    which is the view the edge penalty already takes of the same fit.
    """
    ds, graph = dataset
    params, start = _start(ds, graph)
    before = start.n_edges(2)
    sem = TopDownSEM(
        max_outer_iter=1, do_structure=False, require_support=True, verbose=0
    )
    fitted, learned, history = sem.fit(ds.data, params, start)
    assert learned.n_edges(2) == before
    assert any(fitted[1].parent_sets.values())
    assert history.support == []


def test_the_gate_reads_the_lag_budget_the_search_will_enumerate(dataset):
    """
    A per-scale lag budget narrows the candidate set, and the gate follows it.

    The coarse scale holds the same 12 epochs either way; against 3 candidates
    that is 4.0 each and above where recovery begins, so the same fit searches
    a scale it refused under a budget of two.
    """
    ds, graph = dataset
    sem = TopDownSEM(
        max_outer_iter=1,
        do_structure=True,
        require_support=True,
        sstep_kwargs=dict(max_lag={1: 2, 2: 1}),
        verbose=0,
    )
    seen, (_f, _l, history) = _searched(sem, ds.data, *_start(ds, graph))
    assert [v.n_candidates for v in history.support] == [6, 3]
    assert refused_scales(history.support) == set()
    assert sorted(set(seen)) == [1, 2]


# ---------------------------------------------------------------------------
# The fine formulation reads the same ledger
# ---------------------------------------------------------------------------

FINE_ORIGIN = datetime(2026, 2, 16)
FINE_SLOTS = 6 * 7 * 24  # hourly slots over six weeks
FINE_BRANCHING = (24, 7)


def _fine_sources() -> FineSources:
    """An hour, day, week ladder over six weeks, built the way the fine path is."""
    calendar = SlotCalendar(
        origin=FINE_ORIGIN, slot=timedelta(hours=1), branching=FINE_BRANCHING
    )
    counts, maps = calendar.hierarchy_maps(FINE_SLOTS)
    ancestors = ancestor_maps(maps, FINE_SLOTS)
    rng = np.random.default_rng(0)
    X1 = (rng.random((FINE_SLOTS, LABELS)) < 0.09).astype(np.int64)
    Y1 = np.where(X1 > 0, 1, 0).astype(np.int64)
    per_scale = [
        (
            segment_sum(X1.astype(float), a, n).astype(np.int64),
            segment_sum(Y1.astype(float), a, n).astype(np.int64),
        )
        for a, n in zip(ancestors, counts)
    ]
    return FineSources(
        counts=[x for x, _ in per_scale],
        durations=[y for _, y in per_scale],
        calendar={},
        ancestors=ancestors,
    )


def test_the_fine_ladder_feeds_the_ledger_the_counts_the_coarse_path_does():
    """One ledger governs both formulations, on the same 1-based scale index."""
    ladder = ScaleLadder.from_sources(_fine_sources())
    lags = [2] * ladder.K
    fine = ledger(ladder.n_epochs, LABELS, lags)

    hierarchy = ScaleHierarchy.from_uniform_branching(FINE_SLOTS, list(FINE_BRANCHING))
    coarse = data_ledger(
        MultiScaleData.allocate(hierarchy, [f"L{i}" for i in range(LABELS)]), lags
    )
    assert [v.n_epochs for v in fine] == ladder.n_epochs
    assert [(v.k, v.n_epochs, v.supported) for v in fine] == [
        (v.k, v.n_epochs, v.supported) for v in coarse
    ]


def test_a_candidate_column_carries_the_scale_the_ledger_indexes_by():
    """A typed candidate names its source scale, which is the key the gate uses."""
    ladder = ScaleLadder.from_sources(_fine_sources())
    verdicts = ledger(ladder.n_epochs, LABELS, [2] * ladder.K)
    column = ScaledColumn(
        name="week lag 1", scale=3, values=np.zeros(ladder.n_epochs[2])
    )
    assert refused_scales(verdicts) == {3}
    assert column.scale in refused_scales(verdicts)
