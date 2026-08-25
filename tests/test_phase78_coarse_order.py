"""
Tests for the coarse ordering of the fine search.

The load-bearing one is the identity at full depth: the ordered search returns
exactly what an unordered greedy search returns, so depth is a speed setting and
never a recall risk.  The rest pin what the ordering is measured on, what the
merge rule does, and the failure mode the ordering is stated to have.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.learning.coarse_order import (
    AssociationTensor,
    association,
    association_tensor,
    drop_self,
    lagged,
    ordered_search,
    partial_out,
    recall_against_depth,
    regime_tensor,
    visit_order,
    within_participant_center,
)
from src.learning.fine_sstep import greedy_search

N = 12


def fixed(gains):
    """A search whose gain per candidate never moves and whose objective adds up."""
    values = np.asarray(gains, dtype=float)
    return (lambda selected: values), (
        lambda selected: float(sum(values[i] for i in selected))
    )


def test_the_ordered_search_at_full_depth_is_the_unordered_search():
    rng = np.random.default_rng(0)
    gains = rng.uniform(0.0, 5.0, N)
    priority = rng.uniform(0.0, 1.0, N)
    gain_fn, apply_fn = fixed(gains)
    flat = greedy_search(gain_fn, apply_fn, N, price=1.0, max_edges=4)
    full = ordered_search(gain_fn, apply_fn, priority, price=1.0, max_edges=4)
    assert full.accepted == flat.accepted
    assert full.gains == flat.gains
    assert full.trace == flat.trace
    assert full.depth == N
    assert not full.exhausted


def test_a_reversed_priority_still_reaches_the_same_set_at_full_depth():
    rng = np.random.default_rng(1)
    gains = rng.uniform(0.0, 5.0, N)
    gain_fn, apply_fn = fixed(gains)
    forward = ordered_search(gain_fn, apply_fn, gains, price=1.0, max_edges=5)
    reverse = ordered_search(gain_fn, apply_fn, -gains, price=1.0, max_edges=5)
    assert forward.accepted == reverse.accepted


def test_a_shallow_depth_reaches_only_what_the_ordering_puts_on_top():
    gains = np.arange(N, dtype=float)
    priority = np.arange(N, dtype=float)[::-1]
    gain_fn, apply_fn = fixed(gains)
    shallow = ordered_search(
        gain_fn, apply_fn, priority, price=0.5, max_edges=2, depth=3
    )
    assert shallow.accepted == [2, 1]
    assert shallow.depth == 3
    assert shallow.share_visited == pytest.approx(0.25)


def test_an_evaluation_ceiling_stops_the_run_and_says_so():
    gains = np.arange(N, dtype=float)
    gain_fn, apply_fn = fixed(gains)
    report = ordered_search(
        gain_fn, apply_fn, gains, price=0.5, max_edges=5, evaluations=20
    )
    assert report.exhausted
    assert report.evaluations <= 20
    assert len(report.accepted) < 5


def test_a_depth_beyond_the_space_is_the_whole_space():
    gains = np.arange(N, dtype=float)
    gain_fn, apply_fn = fixed(gains)
    report = ordered_search(
        gain_fn, apply_fn, gains, price=0.5, max_edges=1, depth=N + 50
    )
    assert report.depth == N
    assert report.share_visited == 1.0


def test_a_search_with_nothing_left_to_visit_stops():
    gains = np.array([5.0, 4.0])
    gain_fn, apply_fn = fixed(gains)
    report = ordered_search(gain_fn, apply_fn, gains, price=0.5, max_edges=8)
    assert report.accepted == [0, 1]


def test_a_candidate_that_does_not_clear_its_price_is_never_taken():
    gains = np.array([0.4, 0.2, 0.1])
    gain_fn, apply_fn = fixed(gains)
    report = ordered_search(gain_fn, apply_fn, gains, price=1.0, max_edges=3)
    assert report.accepted == []
    assert report.trace == [0.0]


def test_a_candidate_with_no_finite_gain_is_never_taken():
    gains = np.array([-np.inf, -np.inf])
    gain_fn, apply_fn = fixed(gains)
    assert ordered_search(gain_fn, apply_fn, [1.0, 0.5], price=-np.inf).accepted == []


def test_a_move_whose_realized_objective_falls_is_turned_away():
    gains = np.array([5.0, 4.0, 3.0])

    def gain_fn(selected):
        return gains

    def apply_fn(selected):
        return -10.0 if len(selected) == 2 else float(sum(gains[i] for i in selected))

    report = ordered_search(gain_fn, apply_fn, gains, price=0.5, max_edges=3)
    assert report.accepted == [0]
    assert report.rejected == [1]


def test_the_depth_curve_rises_and_closes_at_full_depth():
    priority = np.arange(N, dtype=float)[::-1]
    wanted = [0, 5, 11]
    curve = recall_against_depth(priority, wanted, [2, 6, 9, N])
    assert [row[0] for row in curve] == [2, 6, 9, N]
    assert [row[2] for row in curve] == [1, 2, 2, 3]
    assert curve[-1][1] == 1.0
    assert all(a[2] <= b[2] for a, b in zip(curve, curve[1:]))


def test_the_default_depth_curve_is_the_whole_space():
    assert recall_against_depth([3.0, 1.0, 2.0], [1]) == [(3, 1.0, 1)]
    assert recall_against_depth([], []) == [(0, 0.0, 0)]


def test_the_visit_order_breaks_ties_by_index():
    assert visit_order([1.0, 1.0, 2.0]) == [2, 0, 1]


# ---------------------------------------------------------------------------
# What the ordering is measured on
# ---------------------------------------------------------------------------


def test_centering_removes_each_participants_own_mean():
    who = np.array([0, 0, 0, 1, 1, 1])
    values = np.array([[1.0], [2.0], [3.0], [10.0], [11.0], [12.0]])
    out = within_participant_center(values, who)
    assert np.allclose(out[:3, 0], [-1.0, 0.0, 1.0])
    assert np.allclose(out[3:, 0], [-1.0, 0.0, 1.0])


def test_a_weight_of_zero_drops_an_epoch_from_the_mean_and_the_column():
    who = np.zeros(4, dtype=np.int64)
    values = np.array([[1.0], [1.0], [1.0], [100.0]])
    out = within_participant_center(values, who, np.array([1.0, 1.0, 1.0, 0.0]))
    assert np.allclose(out[:3, 0], 0.0)
    assert out[3, 0] == 0.0


def test_partialling_out_nothing_returns_the_columns():
    values = np.arange(6.0).reshape(3, 2)
    assert np.array_equal(partial_out(values, np.zeros((3, 0))), values)


def test_a_shared_weekday_rhythm_does_not_rank_above_a_real_relation():
    rng = np.random.default_rng(0)
    n, weekday = 140, 7
    who = np.repeat(np.arange(2), n // 2)
    rhythm = np.sin(2.0 * np.pi * np.arange(n) / weekday)
    driver = rng.normal(size=n)
    related = driver + 0.2 * rng.normal(size=n)
    rhythm_only = rhythm + 0.2 * rng.normal(size=n)
    sources = np.column_stack([driver, rhythm_only])
    targets = np.column_stack([related, rhythm + 0.2 * rng.normal(size=n)])
    context = np.column_stack([rhythm, np.arange(n, dtype=float) % weekday])
    matrix = association(sources, targets, context, who)
    assert matrix[0, 0] > matrix[1, 1]
    bare = association(sources, targets, np.zeros((n, 0)), who)
    assert bare[1, 1] > matrix[1, 1]


def test_the_max_merge_lifts_a_relation_visible_at_one_coarse_scale_alone():
    tensor = AssociationTensor(
        per_scale={2: np.array([[0.1, 0.0]]), 3: np.array([[0.9, 0.0]])},
        source_names=("a",),
        target_names=("a", "b"),
    )
    assert tensor.merged(1, "max")[0, 0] == pytest.approx(0.9)
    assert tensor.merged(1, "mean")[0, 0] == pytest.approx(0.5)
    assert tensor.merged(2, "max")[0, 0] == pytest.approx(0.9)


def test_a_merge_needs_a_coarser_scale_and_a_declared_rule():
    tensor = AssociationTensor(
        per_scale={2: np.zeros((1, 1))}, source_names=("a",), target_names=("a",)
    )
    with pytest.raises(ValueError, match="merge rule"):
        tensor.merged(1, "median")
    with pytest.raises(ValueError, match="no coarser scale"):
        tensor.merged(2)


def test_a_lagged_column_never_reaches_back_into_another_participant():
    who = np.array([0, 0, 0, 1, 1, 1])
    values = np.arange(6.0).reshape(6, 1)
    out = lagged(values, who, lag=1)
    assert np.array_equal(out[:, 0], [0.0, 0.0, 1.0, 0.0, 3.0, 4.0])
    assert np.array_equal(lagged(values, who, lag=0)[:, 0], values[:, 0])
    assert np.all(lagged(values, who, lag=5) == 0.0)


def test_the_self_pair_is_dropped_because_it_is_one_by_construction():
    priority = np.array([[1.0, 0.4], [0.3, 1.0]])
    out = drop_self(priority, ("a", "b"), ("a", "b"))
    assert np.array_equal(out, np.array([[0.0, 0.4], [0.3, 0.0]]))
    assert priority[0, 0] == 1.0


def test_the_tensor_ranks_the_sources_beyond_the_calendar_block():
    rng = np.random.default_rng(2)
    n, labels, context = 60, 2, 3
    counts = [np.zeros((n, labels)), rng.uniform(0, 3, (n, labels))]
    driver = rng.normal(size=n)
    counts[1][:, 0] = driver + 0.1 * rng.normal(size=n)
    exogenous = [
        np.zeros((n, context + 1)),
        np.column_stack([rng.normal(size=(n, context)), driver]),
    ]
    tensor = association_tensor(
        counts=counts,
        exogenous=exogenous,
        participants=[np.zeros(n, dtype=np.int64)] * 2,
        observed=[np.ones(n, dtype=bool)] * 2,
        source_names=("x", "y", "received"),
        target_names=("x", "y"),
        n_context=context,
        scales=[2],
    )
    matrix = tensor.per_scale[2]
    assert matrix.shape == (labels + 1, labels)
    assert matrix[2, 0] > 0.9
    assert matrix[2, 1] < 0.5


def test_a_relation_active_inside_one_regime_is_lifted_by_the_merge():
    rng = np.random.default_rng(3)
    n = 200
    counts = np.zeros((n, 2))
    counts[:, 0] = rng.normal(size=n)
    inside = np.zeros(n, dtype=bool)
    inside[: n // 2] = True
    counts[inside, 1] = counts[inside, 0] + 0.1 * rng.normal(size=inside.sum())
    counts[~inside, 1] = rng.normal(size=(~inside).sum())
    gamma = np.column_stack([inside.astype(float), (~inside).astype(float)])
    tensor = regime_tensor(
        counts=[np.zeros((n, 2)), counts],
        exogenous=[np.zeros((n, 1)), np.zeros((n, 1))],
        participants=[np.zeros(n, dtype=np.int64)] * 2,
        observed=[np.ones(n, dtype=bool)] * 2,
        responsibilities=gamma,
        source_names=("a", "b"),
        target_names=("a", "b"),
        n_context=1,
        scale=2,
    )
    assert tensor.per_scale[1][0, 1] > 0.9
    assert tensor.per_scale[2][0, 1] < 0.5
    assert tensor.merged(0, "max")[0, 1] > 0.9


def test_a_sign_alternating_relation_sinks_and_is_still_reached_at_full_depth():
    rng = np.random.default_rng(4)
    n = 240
    who = np.zeros(n, dtype=np.int64)
    driver = rng.normal(size=n)
    # The sign flips inside every coarse block, so the aggregate cancels.
    flip = np.where((np.arange(n) // 10) % 2 == 0, 1.0, -1.0)
    alternating = flip * driver + 0.05 * rng.normal(size=n)
    steady = driver + 0.05 * rng.normal(size=n)
    sources = np.column_stack([driver, rng.normal(size=n)])
    targets = np.column_stack([alternating, steady])
    matrix = association(sources, targets, np.zeros((n, 0)), who)
    assert matrix[0, 0] < 0.2
    assert matrix[0, 1] > 0.9

    priority = np.array([matrix[0, 0], matrix[0, 1]])
    gains = np.array([9.0, 3.0])
    gain_fn, apply_fn = fixed(gains)
    shallow = ordered_search(
        gain_fn, apply_fn, priority, price=0.5, max_edges=1, depth=1
    )
    assert shallow.accepted == [1]
    full = ordered_search(gain_fn, apply_fn, priority, price=0.5, max_edges=1)
    assert full.accepted == [0]
