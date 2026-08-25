"""
tests/test_phase48_fine_sstep.py
================================
The structure step at full candidate size.

Three claims carry it and each is measured.  A coarse feature lives on a coarse
grid, so the weighted Gram computed at each pair's own resolution equals the
finest-grid answer while touching a small fraction of the terms.  A candidate's
gain is the exact Schur complement of the quadratic, so scoring every candidate
costs one computation rather than one refit each.  And the search keeps a guard,
because a structure move changes the anchor and the posterior and is therefore
not a coordinate-ascent step on the fixed bound.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.typed_graph import LOG_TERM, read_basis
from src.learning.fine_sstep import (
    BaseFit,
    ScaledColumn,
    SearchReport,
    basis_columns,
    basis_terms,
    candidate_gains,
    greedy_search,
    naive_gram,
    quantile_edges,
    reduction_cost,
    resolution_aware_gram,
    shifted_null,
    weights_by_scale,
)
from src.model.intensity import ancestor_maps

# Twelve weeks of minutes, which is the horizon the study runs at.
T = 12 * 7 * 1440
PARENT_MAPS = [
    np.repeat(np.arange(T // 60), 60),
    np.repeat(np.arange(T // 1440), 24),
    np.repeat(np.arange(T // 10080), 7),
]
ANCESTORS = ancestor_maps(PARENT_MAPS, T)
COUNTS = [T, T // 60, T // 1440, T // 10080]


def _columns(seed: int = 0):
    rng = np.random.default_rng(seed)
    return [
        ScaledColumn("minute", 1, rng.normal(size=COUNTS[0])),
        ScaledColumn("hour", 2, rng.normal(size=COUNTS[1])),
        ScaledColumn("day", 3, rng.normal(size=COUNTS[2])),
        ScaledColumn("week", 4, rng.normal(size=COUNTS[3])),
    ]


# ---------------------------------------------------------------------------
# The reduction is exact, and it is what makes the sweep affordable
# ---------------------------------------------------------------------------


def test_the_reduction_equals_the_finest_grid_answer() -> None:
    rng = np.random.default_rng(1)
    weights = rng.random(T)
    columns = _columns()
    per_scale = weights_by_scale(weights, ANCESTORS, COUNTS)
    aware = resolution_aware_gram(columns, ANCESTORS, per_scale)
    assert np.allclose(aware, naive_gram(columns, ANCESTORS, weights), atol=1e-8)


def test_the_reduction_is_symmetric_and_positive_semidefinite() -> None:
    rng = np.random.default_rng(2)
    per_scale = weights_by_scale(rng.random(T), ANCESTORS, COUNTS)
    aware = resolution_aware_gram(_columns(), ANCESTORS, per_scale)
    assert np.allclose(aware, aware.T)
    assert np.min(np.linalg.eigvalsh(aware)) > -1e-6


def test_a_coarse_candidate_set_costs_a_fraction_of_the_finest_grid() -> None:
    """
    The ratio the whole affordability argument rests on, reported not assumed.

    A silent regression here is a silent regression in what the search can
    afford to consider.
    """
    coarse = [c for c in _columns() if c.scale >= 3]
    aware, naive = reduction_cost(coarse, T)
    assert naive / aware > 1000, f"{naive} against {aware}"


def test_one_finest_column_is_what_costs() -> None:
    """
    Almost all the work is the one pair where both columns vary at every slot.

    Measured: dropping the single minute-scale column takes the term count from
    about 125 thousand to about 6 thousand, and the finest self-pair alone is
    120960 of what it removes.  That is why a candidate set can be large while a
    sweep stays affordable.
    """
    everything = reduction_cost(_columns(), T)[0]
    without_finest = reduction_cost([c for c in _columns() if c.scale >= 2], T)[0]
    assert everything > 15 * without_finest
    assert everything - without_finest > 0.9 * everything
    # The finest self-pair by itself accounts for most of the difference.
    assert T > 0.9 * (everything - without_finest)


def test_the_reduction_handles_an_irregular_nesting() -> None:
    ancestors = [np.arange(10), np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 3])]
    counts = [10, 4]
    rng = np.random.default_rng(3)
    weights = rng.random(10)
    columns = [
        ScaledColumn("fine", 1, rng.normal(size=10)),
        ScaledColumn("coarse", 2, rng.normal(size=4)),
    ]
    per_scale = weights_by_scale(weights, ancestors, counts)
    assert np.allclose(
        resolution_aware_gram(columns, ancestors, per_scale),
        naive_gram(columns, ancestors, weights),
        atol=1e-10,
    )


def test_a_one_dimensional_column_is_required() -> None:
    with pytest.raises(ValueError, match="not one column"):
        ScaledColumn("wide", 1, np.ones((4, 2)))


# ---------------------------------------------------------------------------
# The gain is exact, not a proxy
# ---------------------------------------------------------------------------


def _fit_problem(seed: int = 4, n_base: int = 3, n_cand: int = 6, n: int = 800):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, n_base))
    cand = rng.normal(size=(n, n_cand))
    weights = rng.random(n)
    target = rng.normal(size=n)
    fit = BaseFit(base.T @ (base * weights[:, None]), base.T @ target)
    cross = cand.T @ (base * weights[:, None])
    diagonal = np.einsum("tc,tc->c", cand, cand * weights[:, None])
    targets = cand.T @ target
    return base, cand, weights, target, fit, cross, diagonal, targets


def test_the_gain_matches_a_full_refit() -> None:
    base, cand, weights, target, fit, cross, diagonal, targets = _fit_problem()
    got = candidate_gains(fit, cross, diagonal, targets)
    for c in range(cand.shape[1]):
        joined = np.concatenate([base, cand[:, [c]]], axis=1)
        A = joined.T @ (joined * weights[:, None])
        b = joined.T @ target
        refit = 0.5 * b @ np.linalg.solve(A + 1e-8 * np.eye(A.shape[0]), b)
        assert got[c] == pytest.approx(refit - fit.value, abs=1e-8)


def test_every_gain_is_non_negative() -> None:
    """Adding a column cannot lower the maximum of a concave quadratic."""
    *_, fit, cross, diagonal, targets = _fit_problem(5)
    assert np.all(candidate_gains(fit, cross, diagonal, targets) >= -1e-9)


def test_a_candidate_already_in_the_base_gains_nothing() -> None:
    base, _, weights, target, fit, *_ = _fit_problem(6)
    duplicate = base[:, [0]]
    cross = duplicate.T @ (base * weights[:, None])
    diagonal = np.einsum("tc,tc->c", duplicate, duplicate * weights[:, None])
    gain = candidate_gains(fit, cross, diagonal, duplicate.T @ target)
    assert gain[0] == pytest.approx(0.0, abs=1e-6)


def test_the_gains_come_back_in_one_computation() -> None:
    *_, fit, cross, diagonal, targets = _fit_problem(7, n_cand=200)
    gains = candidate_gains(fit, cross, diagonal, targets)
    assert gains.shape == (200,)


# ---------------------------------------------------------------------------
# The basis, declared once per quantity
# ---------------------------------------------------------------------------


def test_the_basis_cuts_at_the_quantiles_of_its_own_split() -> None:
    values = np.arange(100, dtype=float)
    edges = quantile_edges(values, n_bins=4)
    assert edges[0] == -np.inf and edges[-1] == np.inf
    assert np.allclose(edges[1:-1], [24.75, 49.5, 74.25])


def test_the_basis_carries_a_log_term_and_the_contrasts() -> None:
    values = np.arange(1, 101, dtype=float)
    columns = basis_columns(values, quantile_edges(values, 4), "sleep", scale=3)
    assert columns[0].name == "sleep:log"
    assert np.allclose(columns[0].values, np.log1p(values))
    assert len(columns) == 4, "three contrasts against the reference bin"
    stacked = np.stack([c.values for c in columns[1:]], axis=1)
    assert set(stacked.sum(axis=1)) <= {0.0, 1.0}
    assert stacked[0].sum() == 0.0, "the first bin is the reference"


def test_a_threshold_relation_lands_in_one_contrast() -> None:
    """A long sleep is a step between bins, which is how it becomes discoverable."""
    values = np.concatenate([np.full(75, 300.0), np.full(25, 500.0)])
    columns = basis_columns(values, quantile_edges(values, 4), "sleep", 3)
    top = columns[-1].values
    assert np.allclose(top[values > 400.0], 1.0)
    assert np.allclose(top[values < 400.0], 0.0)


def test_a_column_of_the_basis_is_its_term_read_off_the_source() -> None:
    """One list of terms serves the enumeration and the columns built from it."""
    values = np.concatenate([np.arange(1, 51, dtype=float), np.full(30, 500.0)])
    edges = quantile_edges(values, 4)
    columns = basis_columns(values, edges, "sleep", 3)
    terms = basis_terms(edges)
    assert len(columns) == len(terms)
    for column, term in zip(columns, terms):
        assert np.allclose(column.values, read_basis(values, term))


def test_a_basis_needs_at_least_two_bins() -> None:
    with pytest.raises(ValueError, match="at least two bins"):
        quantile_edges(np.arange(10.0), n_bins=1)


def test_a_constant_column_offers_the_log_term_alone() -> None:
    """Its one band would hold every row, which the intercept already carries."""
    edges = quantile_edges(np.full(50, 7.0), n_bins=4)
    assert np.array_equal(edges, [-np.inf, np.inf])
    assert basis_terms(edges) == [LOG_TERM]


def test_a_cut_with_nothing_below_it_is_dropped() -> None:
    """Three quartiles of a sparse column sit on its floor and only one cuts."""
    values = np.concatenate([np.zeros(70), np.full(30, 12.0)])
    edges = quantile_edges(values, n_bins=4)
    assert np.array_equal(edges, [-np.inf, 12.0, np.inf])
    columns = basis_columns(values, edges, "walking", 3)
    assert [c.name for c in columns] == ["walking:log", "walking:[12,inf)"]
    assert np.allclose(columns[1].values, values > 0.0)


# ---------------------------------------------------------------------------
# The null, which is a real column rather than a fabricated edge
# ---------------------------------------------------------------------------


def test_a_shifted_null_keeps_the_column_it_came_from() -> None:
    rng = np.random.default_rng(8)
    values = rng.normal(size=200)
    nulls = shifted_null(values, rng, 16)
    assert len(nulls) == 16
    for null in nulls:
        assert np.allclose(np.sort(null), np.sort(values))
        assert not np.allclose(null, values)


def test_a_shifted_null_of_a_tiny_column_is_the_column() -> None:
    values = np.array([1.0, 2.0])
    assert len(shifted_null(values, np.random.default_rng(9), 3)) == 3


# ---------------------------------------------------------------------------
# The guard the parameter step no longer needs
# ---------------------------------------------------------------------------


def test_the_search_takes_candidates_while_they_clear_the_price() -> None:
    gains = np.array([5.0, 3.0, 0.2, 4.0])
    report = greedy_search(
        gain_fn=lambda chosen: gains,
        apply_fn=lambda chosen: float(gains[chosen].sum()),
        n_candidates=4,
        price=1.0,
    )
    assert report.accepted == [0, 3, 1]
    assert report.rejected == []
    assert report.trace == [0.0, 5.0, 9.0, 12.0]


def test_the_search_stops_when_nothing_clears_the_price() -> None:
    gains = np.array([0.5, 0.4])
    report = greedy_search(
        gain_fn=lambda chosen: gains,
        apply_fn=lambda chosen: float(gains[chosen].sum()),
        n_candidates=2,
        price=1.0,
    )
    assert report.accepted == []


def test_the_guard_rejects_a_move_whose_objective_falls() -> None:
    """
    The score is a promise the realized objective does not have to keep.

    A structure move changes the anchor and the posterior, so the rank-one gain
    is computed under conditions the move itself invalidates.  This is the one
    step the monotonicity of the parameter step does not cover, and the guard is
    what stands in for it.
    """
    gains = np.array([9.0, 8.0])

    def realized(chosen):
        # The second candidate scores well and actually lowers the objective.
        return {(): 0.0, (0,): 4.0, (0, 1): 1.0}[tuple(sorted(chosen))]

    report = greedy_search(
        gain_fn=lambda chosen: gains,
        apply_fn=realized,
        n_candidates=2,
        price=1.0,
    )
    assert report.accepted == [0]
    assert report.rejected == [1]
    assert report.trace == [0.0, 4.0], "the rejected move leaves no trace entry"


def test_the_search_respects_its_edge_budget() -> None:
    gains = np.full(10, 5.0)
    report = greedy_search(
        gain_fn=lambda chosen: gains,
        apply_fn=lambda chosen: float(len(chosen)),
        n_candidates=10,
        price=1.0,
        max_edges=3,
    )
    assert len(report.accepted) == 3


def test_a_search_that_accepts_nothing_reports_its_starting_value() -> None:
    report = greedy_search(
        gain_fn=lambda chosen: np.full(3, -np.inf),
        apply_fn=lambda chosen: -7.0,
        n_candidates=3,
        price=0.0,
    )
    assert report.accepted == [] and report.trace == [-7.0]
    assert isinstance(report, SearchReport)
