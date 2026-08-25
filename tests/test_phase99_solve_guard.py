"""
Tests for the guard that refuses a solve with no information in it.

The 2026-08-23 `exp16` run drew two of its four legs from a fit whose
`survey_response` intercept was +5.4e10.  The chain was: the search bought a
finest-scale self column into a target whose curvature had underflowed, so the
weighted Gram was exactly zero while the unweighted score was not, and
`solve(A + 1e-8 I, b)` returned `b * 1e8`.  Nothing downstream could see it --
the drawn log held zero `survey_response` episodes and the 21-row grid read the
same verdicts it read on the healthy run.

These tests pin the three places the arithmetic was allowed to run:
`solve_informed` behind `PooledFit.solve`, `candidate_gains`, and the NaN that
`realized < current` lets past.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.typed_graph import COUNT, OBSERVED, TypedEdge
from src.learning.fine_mstep import _RIDGE, solve_informed
from src.learning.fine_sstep import BaseFit, candidate_gains
from src.learning.relation_price import RelationPrice, grouped_search


def edge(src_label: int, lag: int = 1, scale: int = 1) -> TypedEdge:
    return TypedEdge(OBSERVED, scale, src_label, COUNT, lag)


# ---------------------------------------------------------------------------
# The solve


def test_a_well_conditioned_solve_is_the_one_it_replaces():
    A = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 0.5], [0.0, 0.5, 2.0]])
    b = np.array([1.0, -2.0, 0.5])
    beta, informed, conditioned = solve_informed(A, b)
    assert informed.all() and conditioned
    expected = np.linalg.solve(A + _RIDGE * np.eye(3), b)
    np.testing.assert_allclose(beta, expected, rtol=1e-10)


def test_a_column_with_no_curvature_is_held_at_zero_not_amplified():
    # Column 1 is the defect's shape exactly: the weighted Gram row and column
    # are zero because every weight underflowed, and `b` is a raw count that
    # never saw a weight.
    A = np.array([[80640.0, 0.0], [0.0, 0.0]])
    b = np.array([-1200.0, 543.0])

    unguarded = np.linalg.solve(A + _RIDGE * np.eye(2), b)
    assert unguarded[1] > 1e10  # what the run actually did

    beta, informed, conditioned = solve_informed(A, b)
    assert informed.tolist() == [True, False]
    assert conditioned
    assert beta[1] == 0.0
    # The informed column is solved as if the empty one were never offered, so
    # one dead column does not cost the target the columns it has evidence for.
    assert beta[0] == pytest.approx(-1200.0 / 80640.0, rel=1e-9)


def test_a_solve_with_nothing_informed_returns_zeros_and_refuses():
    beta, informed, conditioned = solve_informed(
        np.zeros((3, 3)), np.array([5.0, -2.0, 1.0])
    )
    assert not informed.any()
    assert not conditioned
    np.testing.assert_array_equal(beta, np.zeros(3))


def test_the_threshold_is_relative_so_a_genuinely_weak_column_still_solves():
    # Ten at-risk rows at the start rate against 80640 -- small, but real, and
    # the fit is entitled to it.
    A = np.diag([80640.0, 10 * 1.45e-3])
    b = np.array([-1200.0, 0.5])
    beta, informed, conditioned = solve_informed(A, b)
    assert informed.all() and conditioned
    assert beta[1] != 0.0


def test_a_block_that_stays_ill_conditioned_is_refused_whole():
    # Two columns that are the same column: no diagonal is empty, so the column
    # test passes them both, and the block is still singular.
    A = np.array([[100.0, 100.0], [100.0, 100.0]])
    b = np.array([3.0, 3.0])
    beta, informed, conditioned = solve_informed(A, b)
    assert informed.all()
    assert not conditioned
    np.testing.assert_array_equal(beta, np.zeros(2))


def test_conditioning_is_judged_after_the_empty_columns_are_removed():
    # The empty column alone makes the full matrix singular.  Judged on the
    # remaining block the target is perfectly well determined, and refusing it
    # wholesale would throw away a fit that is there.
    A = np.array([[80640.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 9.0]])
    b = np.array([-1200.0, 4.0, 2.0])
    beta, informed, conditioned = solve_informed(A, b)
    assert informed.tolist() == [True, False, True]
    assert conditioned
    assert beta[1] == 0.0
    assert beta[0] != 0.0 and beta[2] != 0.0


# ---------------------------------------------------------------------------
# The candidate score


def base_fit() -> BaseFit:
    return BaseFit(A=np.array([[100.0, 0.0], [0.0, 50.0]]), b=np.array([10.0, -5.0]))


def test_a_supported_candidate_scores_the_schur_complement_it_always_did():
    base = base_fit()
    cross = np.array([[2.0, 1.0]])
    diagonal = np.array([9.0])
    targets = np.array([3.0])

    projected = cross @ base.inverse
    residual = targets - projected @ base.b
    schur = diagonal - np.einsum("cp,cp->c", projected, cross)
    expected = 0.5 * residual**2 / schur

    np.testing.assert_allclose(
        candidate_gains(base, cross, diagonal, targets), expected, rtol=1e-10
    )


def test_a_candidate_with_no_support_scores_zero_rather_than_first():
    base = base_fit()
    # No weighted support at all: it meets nothing and it meets itself nowhere,
    # yet its unweighted score is a real count.
    cross = np.array([[0.0, 0.0]])
    diagonal = np.array([0.0])
    targets = np.array([7.0])
    gains = candidate_gains(base, cross, diagonal, targets)
    assert gains.tolist() == [0.0]


def test_a_candidate_collinear_with_the_base_scores_zero():
    base = base_fit()
    # Its diagonal is real, but the base already spans it, so the complement is
    # zero to rounding and the flooring divide would have ranked it first.
    cross = np.array([[100.0, 0.0]])
    diagonal = np.array([100.0])
    targets = np.array([12.0])
    gains = candidate_gains(base, cross, diagonal, targets)
    assert gains.tolist() == [0.0]


def test_an_empty_candidate_no_longer_outranks_a_real_one():
    base = base_fit()
    cross = np.array([[2.0, 1.0], [0.0, 0.0]])
    diagonal = np.array([9.0, 0.0])
    targets = np.array([3.0, 7.0])
    gains = candidate_gains(base, cross, diagonal, targets)
    assert gains[0] > gains[1]


# ---------------------------------------------------------------------------
# The move guard


def test_a_move_whose_objective_is_not_a_number_is_refused():
    price = RelationPrice(relation=0.5, representation=0.1, n_relations=4)
    candidates = [(edge(0), 3), (edge(1), 3)]
    gains = np.array([10.0, 9.0])

    def gain_fn(selected):
        return gains

    def apply_fn(selected):
        # The first move taken returns NaN, which is what a singular solve
        # hands back through the objective.
        return float("nan")

    report = grouped_search(
        gain_fn, apply_fn, candidates, price, max_edges=4, max_self=2
    )
    assert report.accepted == []
    assert report.rejected == [0]


def test_a_move_that_raises_the_objective_is_still_taken():
    price = RelationPrice(relation=0.5, representation=0.1, n_relations=4)
    candidates = [(edge(0), 3), (edge(1), 3)]
    gains = np.array([10.0, 9.0])

    def gain_fn(selected):
        return gains

    def apply_fn(selected):
        return float(gains[list(selected)].sum())

    report = grouped_search(
        gain_fn, apply_fn, candidates, price, max_edges=4, max_self=2
    )
    assert report.accepted == [0, 1]
