"""
tests/test_phase64_fidelity_verdict.py
======================================
The verdict the posterior-predictive check reads over families of cells.

One share over every cell answers a question about the average cell, so a model
whose morning window sits in the evening passes it.  The cases here run on a
twenty-eight-day ladder of hourly slots over eight labels, which is forty-eight
families, and break one family of one label.  What is asserted is that the share
stays where a correct model leaves it while the verdict names the broken family,
that the verdict's floor and its scaling read every member the same way, so the
observation's rank is uniform over the members, and that a family every member
scores the same leaves the maximum.
"""

from __future__ import annotations

import copy
from datetime import datetime, timedelta

import numpy as np
import pytest

from src.core.calendar import SlotCalendar
from src.core.features import calendar_blocks, horizon_trend, hour_of_day, weekend
from src.core.typed_graph import CALENDAR, COUNT, DURATION, LEVEL, TypedEdge, TypedGraph
from src.evaluation.fidelity import (
    MIN_EPISODES,
    CellGrid,
    FidelityReport,
    posterior_predictive,
    replicate,
)
from src.model.intensity import FineParams, ancestor_maps, empty_sources

LABELS = tuple(f"label{i}" for i in range(8))
N_DAYS = 28
T_SLOTS = N_DAYS * 24
ORIGIN = datetime(2026, 2, 16)
PARENT_MAPS = [
    np.repeat(np.arange(N_DAYS), 24),
    np.repeat(np.arange(N_DAYS // 7), 7),
]
# The bins of `features.hour_of_day` the design carries, the first one dropped
# as the reference level.  Both named bins span three hours, so trading their
# coefficients moves the shape of a day and leaves the rate over it alone.
MORNING, LATE_EVENING = 0, 5
N_REPLICATES = 20


def _calendar() -> SlotCalendar:
    return SlotCalendar(origin=ORIGIN, slot=timedelta(minutes=60), branching=(24, 7))


def _fit():
    """A model with a morning window and a weekend lift, and its empty sources."""
    nodes = [hour_of_day(), weekend(), horizon_trend(ORIGIN, "week")]
    blocks = calendar_blocks(_calendar(), nodes, [T_SLOTS, N_DAYS, N_DAYS // 7])
    sources = empty_sources(
        ancestor_maps(PARENT_MAPS, T_SLOTS), list(blocks.blocks), len(LABELS)
    )
    graph = TypedGraph(
        label_names=list(LABELS),
        calendar_names=[n for k in sorted(blocks.names) for n in blocks.names[k]],
    )
    params = FineParams(n_labels=len(LABELS))
    params.base[COUNT][:] = -2.6
    params.base[DURATION][:] = np.log(2.0)
    params.mark_dispersion = np.full(len(LABELS), 4.0)
    hours = [TypedEdge(CALENDAR, 1, j, LEVEL, 0) for j in range(6)]
    off = TypedEdge(CALENDAR, 2, 0, LEVEL, 0)
    for ell in range(len(LABELS)):
        for j, edge in enumerate(hours + [off]):
            graph.add_edge(edge, ell, COUNT)
            params.coef[(ell, COUNT)][edge] = 0.5
        for j, edge in enumerate(hours):
            params.coef[(ell, COUNT)][edge] = 1.5 if j == MORNING else -0.3
    return params, graph, sources, hours


def _moved_window(params, hours):
    """The same model with one label's morning window traded for its late evening."""
    broken = copy.deepcopy(params)
    coef = broken.coef[(0, COUNT)]
    coef[hours[MORNING]], coef[hours[LATE_EVENING]] = (
        coef[hours[LATE_EVENING]],
        coef[hours[MORNING]],
    )
    return broken


def _record(params, graph, sources, seed: int):
    """One log drawn from the model, with the sources it filled."""
    return replicate(params, graph, sources, np.random.default_rng(seed))


def _check(X1, Y1, filled, params, graph, seed: int, replicates=N_REPLICATES):
    grid = CellGrid.build(_calendar(), filled, LABELS)
    return posterior_predictive(
        X1,
        Y1,
        params,
        graph,
        filled,
        grid,
        np.random.default_rng(seed),
        n_replicates=replicates,
        interval=0.90,
    )


@pytest.fixture(scope="module")
def arms():
    """One record, checked against the model that drew it and against a broken one."""
    params, graph, sources, hours = _fit()
    X1, Y1, filled = _record(params, graph, sources, 1)
    return {
        "params": params,
        "graph": graph,
        "record": (X1, Y1, filled),
        "held": _check(X1, Y1, filled, params, graph, 31),
        "moved": _check(X1, Y1, filled, _moved_window(params, hours), graph, 31),
    }


# ---------------------------------------------------------------------------
# What one share over every cell cannot see
# ---------------------------------------------------------------------------


def test_a_break_in_one_family_leaves_the_share_where_it_was(arms) -> None:
    """The broken cells are a dozen among hundreds, so the share barely moves."""
    held = arms["held"].calibration()
    moved = arms["moved"].calibration()
    assert held.n_cells > 200
    assert abs(moved.outside_share - held.outside_share) < 0.05


def test_the_verdict_names_the_family_the_break_is_in(arms) -> None:
    call = arms["moved"].verdict()
    assert (call.worst.family, call.worst.label) == ("hour_of_day", "label0")
    assert call.broken
    # Every member the observation is compared with stands closer in, so the
    # share of members at or beyond it is the smallest the grid of 1 + R holds.
    assert call.p_value == pytest.approx(1.0 / (N_REPLICATES + 1))
    assert call.worst.discrepancy > 3.0 * call.worst.high
    assert "hour_of_day" in str(call)


def test_the_verdict_holds_still_on_the_model_that_drew_the_log(arms) -> None:
    call = arms["held"].verdict()
    assert call.n_families == len(LABELS) * 6
    assert call.n_cells > 200
    assert not call.broken
    assert call.p_value > call.nominal


def test_the_families_are_ranked_by_how_far_they_stand_out(arms) -> None:
    ranked = arms["moved"].families()
    assert (ranked[0].family, ranked[0].label) == ("hour_of_day", "label0")
    assert [f.standardized for f in ranked] == sorted(
        (f.standardized for f in ranked), reverse=True
    )
    assert ranked[0].n_cells >= 8
    assert str(ranked[0]).startswith("label0 hour_of_day over")


def test_the_report_carries_the_verdict_beside_the_share(arms) -> None:
    text = arms["moved"].report()
    assert "posterior predictive check over 20 replicates" in text
    assert "family-wise p" in text
    assert "label0 hour_of_day" in text


# ---------------------------------------------------------------------------
# The size the verdict keeps
# ---------------------------------------------------------------------------


def test_every_member_taken_as_the_observation_gives_the_uniform_grid(arms) -> None:
    """
    The floor and both scalings read the members symmetrically, so their ranks
    are a permutation of 1 to 1 + R and the observation's share is uniform.
    """
    report = arms["held"]
    p_values = sorted(
        FidelityReport(
            cells=report.cells,
            members=np.roll(report.members, k, axis=0),
            interval=report.interval,
        )
        .verdict()
        .p_value
        for k in range(N_REPLICATES + 1)
    )
    grid = (np.arange(N_REPLICATES + 1) + 1) / float(N_REPLICATES + 1)
    assert p_values == pytest.approx(list(grid))


def test_a_family_every_member_scores_the_same_leaves_the_maximum(arms) -> None:
    """
    A family the members agree on cell for cell holds no comparison.

    Its discrepancy is one number for every member, so keeping it would tie the
    members that reach no further anywhere else and cost the ranks their spread.
    """
    report = arms["held"]
    stuck = np.array(report.members, copy=True)
    columns = [
        j
        for j, cell in enumerate(report.cells)
        if cell.key.family == "episodes_per_day"
    ]
    stuck[:, columns] = 3.0
    frozen = FidelityReport(
        cells=report.cells, members=stuck, interval=report.interval
    ).families()
    assert all(f.family != "episodes_per_day" for f in frozen)
    assert len(frozen) == len(report.families()) - len(LABELS)
