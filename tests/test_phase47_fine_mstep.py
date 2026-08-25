"""
tests/test_phase47_fine_mstep.py
================================
The parameter step, and the two things that make it correct rather than closed.

The posterior variance is in the solve, and dropping it is measured rather than
argued about: the coefficients a mean-only design returns score lower on the very
objective they are supposed to maximize.  And a scale's columns are centered
inside their parent block, which takes the cross-resolution Gram from a large
number to zero, so an effect is attributed to the resolution it varies at.

Order-free ascent is checked here too.  Each step maximizes one block with the
rest held, so the bound cannot fall whatever order the blocks are visited in, and
the orders converge to the same place.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.inference.bound import exact_start_term, pg_mean
from src.learning.fine_mstep import (
    DesignBlock,
    anchor_from,
    ascend,
    block_center,
    bound_value,
    normal_equations,
    quadratic_value,
    regime_block,
    solve_coefficients,
    stack_design,
    update_block,
)

T = 2000
DAY = np.repeat(np.arange(20), 100)
HOUR = np.repeat(np.arange(200), 10)


def _problem(seed: int = 0, n_states: int = 3):
    rng = np.random.default_rng(seed)
    x = (rng.random(T) < 0.2).astype(np.int64)
    gamma = rng.dirichlet([2.0] * n_states, size=20)
    blocks = [
        DesignBlock("base", np.ones((T, 1))),
        regime_block(gamma, DAY),
    ]
    return x, blocks


# ---------------------------------------------------------------------------
# Centering, which is what keeps the resolutions apart
# ---------------------------------------------------------------------------


def test_centering_makes_two_resolutions_orthogonal() -> None:
    """
    The nested-basis problem, measured before and after.

    A day column is reproducible from the hour basis, so the likelihood is flat
    along transfers between them until the finer column is centered.
    """
    rng = np.random.default_rng(1)
    day = rng.normal(size=(20, 1))[DAY]
    hour = rng.normal(size=(200, 1))[HOUR]
    assert abs(float((day * hour).sum())) > 1.0
    centered = block_center(hour, DAY, 20)
    assert abs(float((day * centered).sum())) < 1e-9


def test_a_centered_column_has_no_mean_in_any_parent_block() -> None:
    rng = np.random.default_rng(2)
    hour = rng.normal(size=(200, 2))[HOUR]
    centered = block_center(hour, DAY, 20)
    for c in range(20):
        assert np.allclose(centered[DAY == c].mean(axis=0), 0.0, atol=1e-12)


def test_centering_an_already_centered_column_changes_nothing() -> None:
    rng = np.random.default_rng(3)
    once = block_center(rng.normal(size=(200, 1))[HOUR], DAY, 20)
    assert np.allclose(block_center(once, DAY, 20), once)


def test_centering_removes_a_constant_column_entirely() -> None:
    """A column constant inside every parent block carries no within-block signal."""
    assert np.allclose(block_center(np.ones((T, 1)), DAY, 20), 0.0)


# ---------------------------------------------------------------------------
# The solve
# ---------------------------------------------------------------------------


def test_the_solve_is_the_maximizer_at_its_anchor() -> None:
    x, blocks = _problem()
    weights = pg_mean(np.zeros(T))
    beta = solve_coefficients(x, blocks, weights)
    best = quadratic_value(x, blocks, beta, weights)
    rng = np.random.default_rng(4)
    for _ in range(20):
        other = beta + rng.normal(0, 0.2, size=beta.size)
        assert quadratic_value(x, blocks, other, weights) < best


def test_the_normal_matrix_is_positive_definite() -> None:
    x, blocks = _problem()
    A, _ = normal_equations(x, blocks, pg_mean(np.zeros(T)))
    assert np.allclose(A, A.T)
    assert np.all(np.linalg.eigvalsh(A) > 0)


def test_coordinate_passes_reach_the_joint_solve() -> None:
    """
    One pass is not the joint solve, and that is what coordinate ascent is.

    The blocks are correlated, so a single sweep leaves a gap and repeated
    sweeps close it geometrically.  Measured here rather than assumed, because a
    test asserting one pass suffices would pass only on orthogonal blocks.
    """
    x, blocks = _problem()
    weights = pg_mean(np.zeros(T))
    target = quadratic_value(x, blocks, solve_coefficients(x, blocks, weights), weights)

    beta = np.zeros(sum(b.width for b in blocks))
    gaps = []
    for _ in range(6):
        for which in range(len(blocks)):
            beta = update_block(x, blocks, beta, which, weights)
        gaps.append(target - quadratic_value(x, blocks, beta, weights))
    assert gaps[0] > 1e-4, "one pass does not land on the joint solve"
    assert all(b < a for a, b in zip(gaps, gaps[1:])), "the gap closes every pass"
    assert gaps[-1] < 1e-4


def test_the_solve_respects_the_risk_set() -> None:
    x, blocks = _problem()
    weights = pg_mean(np.zeros(T))
    at_risk = np.zeros(T, dtype=bool)
    at_risk[: T // 2] = True
    beta = solve_coefficients(x, blocks, weights, at_risk)
    half = [
        DesignBlock(
            b.name,
            b.columns[: T // 2],
            None if b.covariance is None else b.covariance[: T // 2],
        )
        for b in blocks
    ]
    same = solve_coefficients(x[: T // 2], half, weights[: T // 2])
    assert np.allclose(beta, same, atol=1e-8)


# ---------------------------------------------------------------------------
# The posterior variance, which is the term a mean-only design drops
# ---------------------------------------------------------------------------


def test_dropping_the_variance_returns_worse_coefficients() -> None:
    """
    Measured, not argued: the mean-only solve scores lower on the real objective.

    This is why the parameter step carries `sum_t w_t V_t` and why a design of
    posterior means alone can lower the bound it is maximizing.
    """
    x, blocks = _problem(5)
    weights = pg_mean(np.zeros(T))
    correct = solve_coefficients(x, blocks, weights, with_variance=True)
    mean_only = solve_coefficients(x, blocks, weights, with_variance=False)
    assert not np.allclose(correct, mean_only)
    assert quadratic_value(x, blocks, mean_only, weights) < quadratic_value(
        x, blocks, correct, weights
    )


def test_the_variance_term_only_matters_where_the_posterior_is_unsure() -> None:
    """With a certain posterior the two solves are the same, which locates it."""
    rng = np.random.default_rng(6)
    x = (rng.random(T) < 0.2).astype(np.int64)
    hard = np.zeros((20, 3))
    hard[np.arange(20), rng.integers(0, 3, size=20)] = 1.0
    blocks = [DesignBlock("base", np.ones((T, 1))), regime_block(hard, DAY)]
    weights = pg_mean(np.zeros(T))
    assert np.allclose(
        solve_coefficients(x, blocks, weights, with_variance=True),
        solve_coefficients(x, blocks, weights, with_variance=False),
        atol=1e-8,
    )


def test_a_regime_block_carries_the_multinomial_covariance() -> None:
    gamma = np.array([[0.2, 0.3, 0.5], [0.1, 0.1, 0.8]])
    block = regime_block(gamma, np.array([0, 0, 1]))
    assert block.width == 2, "the first state is the reference"
    g = gamma[0, 1:]
    assert np.allclose(block.covariance[0], np.diag(g) - np.outer(g, g))
    assert np.allclose(block.columns[2], gamma[1, 1:])


def test_a_certain_regime_has_no_covariance() -> None:
    block = regime_block(np.array([[0.0, 1.0]]), np.array([0, 0]))
    assert np.allclose(block.covariance, 0.0)


# ---------------------------------------------------------------------------
# The bound the step is monotone in
# ---------------------------------------------------------------------------


def test_the_bound_at_its_own_anchor_is_the_exact_term() -> None:
    """With no inferred column the bound is tight, which links back to A10."""
    rng = np.random.default_rng(7)
    x = (rng.random(T) < 0.3).astype(np.int64)
    blocks = [DesignBlock("known", rng.normal(size=(T, 3)))]
    beta = rng.normal(size=3)
    eta, anchor = anchor_from(blocks, beta)
    assert bound_value(x, blocks, beta, anchor) == pytest.approx(
        exact_start_term(x, eta), abs=1e-9
    )


def test_an_inferred_column_makes_the_bound_strictly_lower() -> None:
    """Posterior uncertainty costs the bound, which is the honest direction."""
    x, blocks = _problem(8)
    beta = np.array([0.5, 0.4, -0.3])
    eta, anchor = anchor_from(blocks, beta)
    with_var = bound_value(x, blocks, beta, anchor)
    known = [DesignBlock(b.name, b.columns) for b in blocks]
    _, anchor_known = anchor_from(known, beta)
    assert with_var < bound_value(x, known, beta, anchor_known)


# ---------------------------------------------------------------------------
# Order-free ascent
# ---------------------------------------------------------------------------


ORDERS = [
    [0, 1] * 8,
    [1, 0] * 8,
    [0, 0, 1, 1] * 4,
    [1, 1, 0, 0] * 4,
    [0, 1, 1, 0] * 4,
]


@pytest.mark.parametrize("order", ORDERS)
def test_the_bound_never_falls_whatever_the_order(order) -> None:
    x, blocks = _problem(9)
    width = sum(b.width for b in blocks)
    _, trace = ascend(x, blocks, np.zeros(width), order=order)
    drops = [b - a for a, b in zip(trace, trace[1:]) if b < a - 1e-9]
    assert drops == [], f"the bound fell by {drops}"


def test_a_shuffled_order_never_falls_either() -> None:
    x, blocks = _problem(10)
    width = sum(b.width for b in blocks)
    rng = np.random.default_rng(11)
    for _ in range(10):
        order = rng.integers(0, len(blocks), size=12).tolist()
        _, trace = ascend(x, blocks, np.zeros(width), order=order)
        assert all(b >= a - 1e-9 for a, b in zip(trace, trace[1:]))


def test_the_orders_converge_to_the_same_place() -> None:
    """
    Order-free means the schedule is free, not that a short run is finished.

    At sixteen steps the orders sit 5.36 nats apart, because one of them spends
    half its steps on a block that is already optimal.  Given enough steps they
    agree, which is the claim; the short-run spread is convergence, not a
    dependence on the order.
    """
    x, blocks = _problem(12)
    width = sum(b.width for b in blocks)
    spreads = []
    for n in (16, 40, 80):
        finals = [
            ascend(x, blocks, np.zeros(width), order=(o * n)[:n])[1][-1]
            for o in ([0, 1], [1, 0], [0, 0, 1, 1], [1, 1, 0, 0], [0, 1, 1, 0])
        ]
        spreads.append(max(finals) - min(finals))
    assert spreads[0] > 1.0, "a short run has not converged, whichever order it used"
    assert spreads[-1] < 1e-3, f"the orders disagree at convergence: {spreads}"
    assert all(b < a for a, b in zip(spreads, spreads[1:]))


def test_ascent_improves_on_where_it_started() -> None:
    x, blocks = _problem(13)
    width = sum(b.width for b in blocks)
    start = np.zeros(width)
    _, anchor = anchor_from(blocks, start)
    before = bound_value(x, blocks, start, anchor)
    _, trace = ascend(x, blocks, start, order=[0, 1] * 6)
    assert trace[-1] > before


# ---------------------------------------------------------------------------
# What the design refuses
# ---------------------------------------------------------------------------


def test_an_empty_design_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one block"):
        stack_design([])


def test_a_one_dimensional_column_set_is_refused() -> None:
    with pytest.raises(ValueError, match="not 2-D"):
        DesignBlock("flat", np.ones(10))


def test_a_covariance_of_the_wrong_shape_is_refused() -> None:
    with pytest.raises(ValueError, match="expected"):
        DesignBlock("bad", np.ones((10, 2)), np.zeros((10, 3, 3)))


def test_a_block_with_no_covariance_contributes_none() -> None:
    _, cov = stack_design([DesignBlock("known", np.ones((5, 2)))])
    assert np.allclose(cov, 0.0) and cov.shape == (5, 2, 2)
