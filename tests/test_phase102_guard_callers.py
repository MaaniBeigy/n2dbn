"""
Tests for what each caller of the conditioning guard does when the solve refuses.

`test_phase99_solve_guard` pins the arithmetic of `solve_informed` itself.  What
is pinned here is the other half, which the 2026-08-23 run needed and did not
have: a caller handed a refusal has to keep the coefficients it came in with and
say so, and a search handed a refused move has to put the move back.  Every
branch below was reachable on that run and none of them had a test.

The guard is driven by replacing `solve_informed` in the module under test.  The
alternative, building a cohort whose weighted Gram underflows, measures the
underflow and not the caller.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.typed_graph import COUNT, OBSERVED, TypedEdge, TypedGraph
from src.data.label_axis import LabelAxis
from src.evaluation.heterogeneity import Heterogeneity, participant_cqv
from src.evaluation.round_trip import panel_counts, stationary_law
from src.learning.fine_sem import RegimeChain
from src.learning.pooled import PooledFit
from src.model.intensity import FineParams
from tests.test_phase84_pooled_grouped import BUDGET, LABELS, L, cohort, pool


def _refusing(monkeypatch, module, verdicts):
    """Replace `solve_informed` in one module with a scripted sequence of verdicts.

    Args:
        verdicts: one `(informed, conditioned)` pair per call, the last repeating.
    """
    calls = []
    original = module.solve_informed

    def scripted(A, b, ridge):
        beta, informed, conditioned = original(A, b, ridge)
        wanted = verdicts[min(len(calls), len(verdicts) - 1)]
        calls.append(wanted)
        mask = (
            np.ones(len(beta), dtype=bool) if wanted[0] else np.zeros_like(beta, bool)
        )
        return beta, mask, wanted[1]

    monkeypatch.setattr(module, "solve_informed", scripted)
    return calls


def _fit(rounds: int = 1, move_rounds: int = 1):
    graph = TypedGraph(label_names=list(LABELS))
    params = FineParams(n_labels=L)
    params.base[COUNT][:] = -2.0
    candidates = list(pool())
    fit = PooledFit(
        graph,
        params,
        [],
        BUDGET,
        targets=[0],
        candidates={0: candidates},
        move_rounds=move_rounds,
    )
    people = cohort(4)
    for _ in range(rounds):
        fit.solve(fit.accumulate(people, step=False))
    return people, fit, candidates


# ---------------------------------------------------------------------------
# PooledFit.solve


def test_a_refused_target_keeps_the_coefficients_it_came_in_with(monkeypatch):
    import src.learning.pooled as pooled

    people, fit, _ = _fit()
    before = float(fit.params.base[COUNT][0])
    _refusing(monkeypatch, pooled, [(True, False)])
    out = fit.solve(fit.accumulate(people, step=False))
    assert out == {}
    assert float(fit.params.base[COUNT][0]) == before
    assert "<solve refused: ill-conditioned>" in fit.refused[0]


def test_a_held_column_is_named_even_where_the_block_still_solves(monkeypatch):
    import src.learning.pooled as pooled

    people, fit, _ = _fit()
    _refusing(monkeypatch, pooled, [(False, True)])
    out = fit.solve(fit.accumulate(people, step=False))
    assert 0 in out
    assert fit.refused[0] and "<solve refused" not in " ".join(fit.refused[0])
    assert "intercept" in fit.refused[0]


def test_nothing_is_refused_when_every_column_is_informed():
    people, fit, _ = _fit()
    fit.solve(fit.accumulate(people, step=False))
    assert fit.refused == {}


# ---------------------------------------------------------------------------
# The widened solve and the move


def test_a_widened_solve_with_no_information_returns_nothing(monkeypatch):
    import src.learning.pooled as pooled

    people, fit, _ = _fit()
    stats = fit.accumulate(people, step=False)
    _refusing(monkeypatch, pooled, [(True, False)])
    assert fit._widened(stats, 0, 0) is None


def test_a_move_whose_base_solve_refuses_is_not_a_move(monkeypatch):
    import src.learning.pooled as pooled

    people, fit, candidates = _fit()
    stats = fit.accumulate(people, step=False)
    _refusing(monkeypatch, pooled, [(True, False)])
    gained = fit._move(people, 0, candidates, [0], {"stats": stats})
    assert gained == float("-inf")
    assert not fit.graph.get_parents(0, COUNT)


def test_a_move_whose_widened_solve_refuses_is_put_back(monkeypatch):
    import src.learning.pooled as pooled

    people, fit, candidates = _fit()
    stats = fit.accumulate(people, step=False)
    # The base solve stands; the widened one, which writes the bought edge's
    # coefficient, is the one that refuses.
    _refusing(monkeypatch, pooled, [(True, True), (True, False)])
    gained = fit._move(people, 0, candidates, [0], {"stats": stats})
    assert gained == float("-inf")
    assert not fit.graph.get_parents(0, COUNT)


def test_a_move_refused_on_a_later_round_removes_the_edge_and_restores(monkeypatch):
    import src.learning.pooled as pooled

    people, fit, candidates = _fit(move_rounds=2)
    stats = fit.accumulate(people, step=False)
    before = float(fit.params.base[COUNT][0])
    # Base solve, widened solve, then the second coefficient round refuses.
    _refusing(monkeypatch, pooled, [(True, True), (True, True), (True, False)])
    gained = fit._move(people, 0, candidates, [0], {"stats": stats})
    assert gained == float("-inf")
    assert not fit.graph.get_parents(0, COUNT)
    assert float(fit.params.base[COUNT][0]) == before


# ---------------------------------------------------------------------------
# The same guard on the draw path


def test_a_refused_subject_level_block_keeps_its_levels(monkeypatch):
    import src.learning.subject_offset as subject_offset
    from src.learning.subject_offset import SubjectLevelFit

    people = cohort(4)
    graph = TypedGraph(label_names=list(LABELS))
    params = FineParams(n_labels=L, n_subjects=people.S)
    params.base[COUNT][:] = -2.0
    fit = SubjectLevelFit(graph, params, BUDGET)
    before = float(fit.params.base[COUNT][0])
    rows = np.ones((people.S, L))
    _refusing(monkeypatch, subject_offset, [(True, False)])
    fit.solve(fit.accumulate(people, rows), rows)
    assert float(fit.params.base[COUNT][0]) == before


# ---------------------------------------------------------------------------
# The stationary law of a chain, and the panel's own counts


def _chain(log_trans, log_init=(0.5, 0.5)) -> RegimeChain:
    return RegimeChain(
        scale=2,
        n_states=2,
        effect=np.zeros((2, L)),
        log_init=np.log(np.asarray(log_init, dtype=float)),
        log_trans=np.asarray(log_trans, dtype=float),
    )


def test_a_chain_whose_transitions_vanish_keeps_the_law_it_reached():
    law = stationary_law(_chain(np.full((2, 2), -np.inf)))
    assert law.shape == (2,) and np.all(np.isfinite(law))


def test_a_law_that_does_not_settle_returns_the_last_iterate():
    # A pure swap from an off-centre start oscillates forever, so the loop runs
    # out and returns where it stopped instead of claiming a fixed point it
    # never reached.  From the uniform start the same chain is already there.
    swap = np.log([[1e-300, 1.0], [1.0, 1e-300]])
    law = stationary_law(_chain(swap, log_init=(0.9, 0.1)))
    assert law.shape == (2,)
    assert float(law.sum()) == pytest.approx(1.0)
    assert float(np.max(law)) == pytest.approx(0.9)
    assert stationary_law(_chain(swap))[0] == pytest.approx(0.5)


def test_the_panel_counts_only_the_episodes_inside_the_mask():
    people = cohort(3)
    axis = LabelAxis(
        targets=tuple(LABELS), exogenous=(), grouped={n: n for n in LABELS}
    )
    counts = panel_counts(people, axis)
    assert set(counts) == set(LABELS)
    assert all(n >= 0 for n in counts.values())
    assert sum(counts.values()) > 0


# ---------------------------------------------------------------------------
# The dispersion measure's two unreached branches


def test_a_participant_whose_weeks_are_all_equal_has_no_dispersion():
    assert participant_cqv([5.0, 5.0, 5.0, 5.0]) == 0.0


def test_a_series_the_measure_would_warn_on_is_refused_before_the_library_sees_it():
    # `cqv` warns when the two quartiles sum to zero.  On a non-negative series
    # that is both of them being zero, which is decided here and returns None,
    # and a series that could reach it another way is refused as a count.
    assert participant_cqv([0.0, 0.0, 0.0, 0.0]) is None
    with pytest.raises(ValueError, match="at or above zero"):
        participant_cqv([-2.0, -1.0, 1.0, 2.0])


def test_an_empty_roster_reports_no_pinned_share():
    empty = Heterogeneity(
        label="gym",
        cqv=0.0,
        lower=0.0,
        upper=0.0,
        n_defined=0,
        n_participants=0,
        replicates=1,
        seed=0,
    )
    assert empty.pinned_share == 0.0
    assert empty.undefined_sparse == 0.0


def test_the_record_a_run_stores_carries_every_field_a_verdict_reads():
    row = Heterogeneity(
        label="nap",
        cqv=0.333,
        lower=0.135,
        upper=0.385,
        n_defined=40,
        n_participants=51,
        replicates=999,
        seed=0,
        n_pinned=23,
    ).as_row()
    assert row["label"] == "nap" and row["cqv"] == 0.333
    assert row["n_defined"] == 40 and row["n_participants"] == 51
