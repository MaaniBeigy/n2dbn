"""
tests/test_phase46_bound.py
===========================
One objective, and the adjoint pair that carries it across scales.

Three claims are checked rather than asserted in prose.  The broadcast and the
segment sum are adjoint to machine precision, which is why an upward message
needs no new machinery.  The quadratic form touches the exact start term at its
anchor and lies below it everywhere else, which is what makes maximizing it safe.
And a scale's pseudo-emission is exactly the state-dependent part of that
quadratic, reduced by the same segment sum.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.inference.bound import (
    adjoint_residual,
    broadcast,
    chain_entropy,
    evidence_bound,
    exact_start_term,
    gaussian_entropy,
    pg_mean,
    pg_update,
    quadratic_start_bound,
    segment_sum,
    upward_message,
)

T, L, M, EPOCHS = 400, 3, 4, 20
ANCESTORS = np.repeat(np.arange(EPOCHS), T // EPOCHS)


# ---------------------------------------------------------------------------
# The two operators are adjoint
# ---------------------------------------------------------------------------


def test_the_broadcast_and_the_segment_sum_are_adjoint() -> None:
    rng = np.random.default_rng(0)
    for shape in [(EPOCHS,), (EPOCHS, L), (EPOCHS, M)]:
        values = rng.normal(size=shape)
        fine = rng.normal(size=(T,) + shape[1:])
        assert adjoint_residual(values, fine, ANCESTORS, EPOCHS) < 1e-9


def test_the_adjoint_holds_on_an_irregular_nesting() -> None:
    """Blocks of unequal size, which a calendar ladder produces."""
    ancestors = np.array([0, 0, 0, 1, 2, 2, 3, 3, 3, 3])
    rng = np.random.default_rng(1)
    values, fine = rng.normal(size=(4, 2)), rng.normal(size=(10, 2))
    assert adjoint_residual(values, fine, ancestors, 4) < 1e-12


def test_a_broadcast_followed_by_a_segment_sum_weights_by_block_size() -> None:
    values = np.arange(EPOCHS, dtype=float)
    back = segment_sum(broadcast(values, ANCESTORS), ANCESTORS, EPOCHS)
    assert np.allclose(back, values * (T // EPOCHS))


# ---------------------------------------------------------------------------
# The quadratic form
# ---------------------------------------------------------------------------


def test_the_variate_mean_is_a_quarter_at_zero() -> None:
    assert pg_mean(np.zeros(3)).tolist() == [0.25, 0.25, 0.25]


def test_the_variate_mean_matches_its_formula_and_is_even() -> None:
    theta = np.array([0.3, 1.0, 4.0])
    assert np.allclose(pg_mean(theta), np.tanh(theta / 2) / (2 * theta))
    assert np.allclose(pg_mean(theta), pg_mean(-theta))


def test_the_variate_mean_falls_as_the_anchor_grows() -> None:
    values = pg_mean(np.array([0.0, 0.5, 1.0, 2.0, 8.0]))
    assert np.all(np.diff(values) < 0)


def test_the_bound_touches_the_exact_term_at_its_anchor() -> None:
    rng = np.random.default_rng(2)
    x = (rng.random((T, L)) < 0.2).astype(np.int64)
    eta = rng.normal(0, 1.5, size=(T, L))
    assert quadratic_start_bound(x, eta, eta) == pytest.approx(
        exact_start_term(x, eta), abs=1e-9
    )


@pytest.mark.parametrize("shift", [-3.0, -0.7, 0.4, 2.5])
def test_the_bound_lies_below_the_exact_term_away_from_its_anchor(shift) -> None:
    rng = np.random.default_rng(3)
    x = (rng.random((T, L)) < 0.2).astype(np.int64)
    anchor = rng.normal(0, 1.0, size=(T, L))
    eta = anchor + shift
    assert quadratic_start_bound(x, eta, anchor) < exact_start_term(x, eta)


def test_re_anchoring_tightens_the_bound() -> None:
    """The step the inference takes, and it moves the bound the right way."""
    rng = np.random.default_rng(4)
    x = (rng.random((T, L)) < 0.25).astype(np.int64)
    eta = rng.normal(0, 2.0, size=(T, L))
    anchor = np.zeros_like(eta)
    exact = exact_start_term(x, eta)
    seen = []
    for _ in range(4):
        value = quadratic_start_bound(x, eta, anchor)
        assert value <= exact + 1e-9, "the quadratic is a lower bound at every anchor"
        seen.append(value)
        anchor = np.abs(eta)
    assert seen[0] < exact - 1.0, "a zero anchor is a loose bound"
    assert all(
        b >= a - 1e-9 for a, b in zip(seen, seen[1:])
    ), "re-anchoring never loses"
    # At a fixed predictor one re-anchor is enough, so the rest are flat at the
    # exact value rather than creeping toward it.
    assert seen[-1] == pytest.approx(exact, abs=1e-9)
    assert seen[1] == pytest.approx(seen[-1], abs=1e-9)


def test_the_bound_respects_the_risk_set() -> None:
    rng = np.random.default_rng(5)
    x = (rng.random((T, L)) < 0.2).astype(np.int64)
    eta = rng.normal(size=(T, L))
    at_risk = rng.random((T, L)) < 0.6
    assert quadratic_start_bound(x, eta, eta, at_risk) == pytest.approx(
        exact_start_term(x, eta, at_risk), abs=1e-9
    )


def test_posterior_variance_damps_the_variate() -> None:
    """A predictor the posterior is unsure of is anchored out and damped harder."""
    mean = np.full((3, 1), 0.5)
    certain = pg_update(mean)
    uncertain = pg_update(mean, np.full((3, 1), 4.0))
    assert np.all(uncertain < certain)


# ---------------------------------------------------------------------------
# The upward message is that quadratic, reduced
# ---------------------------------------------------------------------------


def _setup(seed: int = 6):
    rng = np.random.default_rng(seed)
    x = (rng.random((T, L)) < 0.2).astype(np.int64)
    eta_rest = rng.normal(0, 0.8, size=(T, L))
    own = rng.normal(0, 0.6, size=(M, L))
    anchor = np.abs(rng.normal(0, 1.0, size=(T, L)))
    return x, eta_rest, own, anchor, pg_mean(anchor)


def test_the_message_matches_a_per_epoch_reference() -> None:
    x, eta_rest, own, _, weights = _setup()
    got = upward_message(x, eta_rest, own, weights, ANCESTORS, EPOCHS)
    expected = np.zeros((EPOCHS, M))
    xb = (x > 0).astype(float)
    for t in range(T):
        c = ANCESTORS[t]
        for m in range(M):
            for ell in range(L):
                expected[c, m] += (
                    xb[t, ell] - 0.5 - weights[t, ell] * eta_rest[t, ell]
                ) * own[m, ell] - 0.5 * weights[t, ell] * own[m, ell] ** 2
    assert np.allclose(got, expected)
    assert got.shape == (EPOCHS, M)


def test_the_message_is_the_state_dependent_part_of_the_bound() -> None:
    """
    A difference of messages equals the difference the bound itself would show.

    This is what makes the pseudo-emission the right thing to hand a chain: the
    state it prefers is the state the objective prefers.
    """
    x, eta_rest, own, anchor, weights = _setup(7)
    got = upward_message(x, eta_rest, own, weights, ANCESTORS, EPOCHS)
    for m, mp in [(0, 1), (2, 3), (0, 3)]:
        here = quadratic_start_bound(x, eta_rest + own[m], anchor)
        there = quadratic_start_bound(x, eta_rest + own[mp], anchor)
        assert here - there == pytest.approx(
            float(got[:, m].sum() - got[:, mp].sum()), abs=1e-8
        )


def test_the_message_respects_the_risk_set() -> None:
    x, eta_rest, own, _, weights = _setup(8)
    at_risk = np.zeros((T, L), dtype=bool)
    at_risk[: T // 2] = True
    got = upward_message(x, eta_rest, own, weights, ANCESTORS, EPOCHS, at_risk)
    assert np.allclose(got[EPOCHS // 2 :], 0.0)
    assert not np.allclose(got[: EPOCHS // 2], 0.0)


# ---------------------------------------------------------------------------
# The entropy the older objective omitted
# ---------------------------------------------------------------------------


def test_an_independent_chain_has_the_entropy_of_its_marginals() -> None:
    gamma = np.array([[0.3, 0.7], [0.5, 0.5], [0.2, 0.8]])
    xi = np.stack([np.outer(gamma[t], gamma[t + 1]) for t in range(2)])
    assert chain_entropy(gamma, xi) == pytest.approx(chain_entropy(gamma), abs=1e-9)


def test_a_persistent_chain_has_less_entropy_than_its_marginals_suggest() -> None:
    """The term the summed per-scale evidence left out, and its sign."""
    gamma = np.full((6, 2), 0.5)
    xi = np.tile(np.array([[0.49, 0.01], [0.01, 0.49]]), (5, 1, 1))
    assert chain_entropy(gamma, xi) < chain_entropy(gamma)


def test_a_certain_chain_has_no_entropy() -> None:
    gamma = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    xi = np.stack([np.array([[1.0, 0.0], [0.0, 0.0]])] * 2)
    assert chain_entropy(gamma, xi) == pytest.approx(0.0, abs=1e-9)
    assert chain_entropy(gamma) == pytest.approx(0.0, abs=1e-9)


def test_a_two_epoch_chain_is_the_entropy_of_its_pair() -> None:
    gamma = np.array([[0.4, 0.6], [0.5, 0.5]])
    xi = np.array([[[0.3, 0.1], [0.2, 0.4]]])
    expected = -float(np.sum(xi * np.log(xi)))
    assert chain_entropy(gamma, xi) == pytest.approx(expected, abs=1e-12)


def test_the_factor_entropy_matches_the_gaussian_formula() -> None:
    cov = np.stack([np.eye(2)] * 5)
    expected = 5 * 0.5 * 2 * (1.0 + np.log(2.0 * np.pi))
    assert gaussian_entropy(cov) == pytest.approx(expected, abs=1e-9)


def test_a_model_with_no_factor_carries_no_factor_entropy() -> None:
    assert gaussian_entropy(np.zeros((5, 0, 0))) == 0.0
    assert gaussian_entropy(np.zeros((0,))) == 0.0


# ---------------------------------------------------------------------------
# The objective names its parts
# ---------------------------------------------------------------------------


def test_the_objective_is_its_parts_and_nothing_else() -> None:
    assert evidence_bound(-100.0, -20.0, 5.0, 3.0) == pytest.approx(-118.0)
    assert evidence_bound(-100.0, -20.0, 5.0) == pytest.approx(-115.0)


def test_the_penalty_only_lowers_the_objective() -> None:
    without = evidence_bound(-10.0, -1.0, 2.0, 0.0)
    with_price = evidence_bound(-10.0, -1.0, 2.0, 4.0)
    assert with_price == without - 4.0
