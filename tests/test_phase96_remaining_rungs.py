"""
Tests for the three rungs the ladder was still missing.

R6 is exact by construction, so its property is that a reconciled log leaves a
residual of exactly zero and a single altered epoch is located by its own
residual.  R8 carries the load: two unrelated autocorrelated series clear a
plain reshuffle five times in twenty and clear the block permutation twice, and
the block null is the wider of the two on every one of those seeds, which is the
whole reason the block exists.  R11 is measured over participants, so a gain
that one participant carries alone does not reach the level the same mean gain
reaches when every participant carries it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.evaluation.falsification import (
    DEFAULT_ALPHA,
    CrossResolution,
    FoldingResidual,
    HeldOut,
    PermutationBaseline,
    block_permutation,
    consistency_ladder,
    folding_residual,
    held_out_gain,
    lag1_autocorrelation,
    paired_gain,
    permuted_score,
    profile_residual,
)

WIDTH = 6
EPOCHS = 12


def ladder_log(seed: int, epochs: int = EPOCHS, width: int = WIDTH):
    """A fine array, the coarse totals a reconciled log stores, and the map."""
    rng = np.random.default_rng(seed)
    fine = rng.poisson(1.5, epochs * width).astype(float)
    parents = np.repeat(np.arange(epochs), width)
    coarse = np.array([fine[parents == e].sum() for e in range(epochs)])
    return fine, coarse, parents


def ar1(rng: np.random.Generator, n: int, rho: float) -> np.ndarray:
    """One autoregressive draw of the given lag-1 correlation."""
    out = np.empty(n)
    out[0] = rng.normal()
    for i in range(1, n):
        out[i] = rho * out[i - 1] + math.sqrt(1.0 - rho * rho) * rng.normal()
    return out


def two_series(seed: int, coupling: float, units: int = 6, span: int = 120):
    """Two persistent series per participant, coupled by the given amount."""
    rng = np.random.default_rng(seed)
    who = np.repeat(np.arange(units), span)
    x = np.concatenate([ar1(rng, span, 0.9) for _ in range(units)])
    z = np.concatenate([ar1(rng, span, 0.9) for _ in range(units)])
    y = coupling * x + math.sqrt(1.0 - coupling * coupling) * z
    return x, y, who


def abs_corr(a: np.ndarray, b: np.ndarray) -> float:
    """The absolute correlation of two columns."""
    return abs(float(np.corrcoef(a, b)[0, 1]))


def per_slot_likelihood(design, y, train, test) -> np.ndarray:
    """A least-squares fit on the training slots, scored per held-out slot."""
    beta, *_ = np.linalg.lstsq(design[train], y[train], rcond=None)
    residual = y[train] - design[train] @ beta
    sigma2 = float(residual @ residual) / int(train.sum())
    error = y[test] - design[test] @ beta
    return -0.5 * np.log(2.0 * np.pi * sigma2) - 0.5 * error**2 / sigma2


def two_models(seed: int, coupling: float, units: int = 12, span: int = 60):
    """Per-slot held-out likelihoods of a model with the parent and one without."""
    rng = np.random.default_rng(seed)
    who = np.repeat(np.arange(units), span)
    x = rng.normal(size=who.size)
    y = coupling * x + rng.normal(size=who.size)
    train = np.arange(who.size) % 2 == 0
    test = ~train
    full = per_slot_likelihood(np.column_stack([np.ones(who.size), x]), y, train, test)
    base = per_slot_likelihood(np.ones((who.size, 1)), y, train, test)
    return full, base, who[test]


# ---------------------------------------------------------------------------
# R6, cross-resolution consistency
# ---------------------------------------------------------------------------


def test_a_reconciled_log_folds_onto_its_coarse_scale_with_no_residual():
    fine, coarse, parents = ladder_log(0)
    report = folding_residual(fine, coarse, parents)
    assert report.residual == 0.0
    assert report.n_epochs == EPOCHS
    assert CrossResolution((report,)).passed


def test_a_reconciled_ladder_of_non_integer_totals_still_folds():
    # Counts fold exactly.  A scale carrying a non-integer aggregate is summed
    # in one order by the fold and in another by whoever built the coarse array,
    # so a reconciled log lands a few epsilons apart and the rung has to say it
    # folds.  Two hundred seeds, none of which is allowed to fail.
    for seed in range(200):
        rng = np.random.default_rng(seed)
        parents = np.repeat(np.arange(EPOCHS), 24)
        fine = rng.normal(size=parents.size) * 1000.0
        coarse = np.array([fine[parents == k].sum() for k in range(EPOCHS)])
        report = folding_residual(fine, coarse, parents)
        assert report.folds
        assert CrossResolution((report,)).passed
    assert report.residual > 0.0
    assert report.tolerance > report.residual


def test_a_departure_larger_than_the_rounding_error_does_not_fold():
    parents = np.repeat(np.arange(4), 3)
    fine = np.full(12, 0.1)
    coarse = np.full(4, 0.30000000000000004)
    assert folding_residual(fine, coarse, parents).folds
    moved = coarse.copy()
    moved[2] += 1e-9
    report = folding_residual(fine, moved, parents)
    assert not report.folds
    assert report.epoch == 2


def test_a_rounding_allowance_below_zero_is_refused():
    with pytest.raises(ValueError, match="at or above zero"):
        FoldingResidual(residual=0.0, epoch=0, n_epochs=1, tolerance=-1.0)
    with pytest.raises(ValueError, match="at or above zero"):
        FoldingResidual(residual=0.0, epoch=0, n_epochs=1, tolerance=float("nan"))


def test_a_single_altered_epoch_is_located_by_its_own_residual():
    fine, coarse, parents = ladder_log(1)
    coarse[7] += 3.0
    report = folding_residual(fine, coarse, parents)
    assert report.residual == 3.0
    assert report.epoch == 7
    assert not CrossResolution((report,)).passed


def test_the_ladder_reports_one_residual_per_adjacent_pair_finest_first():
    fine, mid, fine_map = ladder_log(2)
    mid_map = np.repeat(np.arange(4), 3)
    top = np.array([mid[mid_map == e].sum() for e in range(4)])
    rungs = consistency_ladder([fine, mid, top], [fine_map, mid_map])
    assert [r.n_epochs for r in rungs] == [EPOCHS, 4]
    assert CrossResolution(tuple(rungs)).passed
    # Break only the coarsest pair; the finer one is untouched and stays exact.
    top[2] += 1.0
    rungs = consistency_ladder([fine, mid, top], [fine_map, mid_map])
    assert rungs[0].residual == 0.0
    assert rungs[1].residual == 1.0
    assert rungs[1].epoch == 2
    assert not CrossResolution(tuple(rungs)).passed


def test_a_ladder_of_one_scale_has_no_pair_to_check():
    assert consistency_ladder([np.arange(4.0)], []) == []
    assert CrossResolution(()).passed


def test_a_ladder_with_no_array_or_a_miscounted_map_is_refused():
    with pytest.raises(ValueError, match="at least one array"):
        consistency_ladder([], [])
    with pytest.raises(ValueError, match="adjacent pairs"):
        consistency_ladder([np.arange(4.0), np.arange(2.0)], [])


def test_a_parent_map_that_runs_backwards_is_refused():
    with pytest.raises(ValueError, match="non-decreasing"):
        folding_residual(np.arange(4.0), np.zeros(2), [0, 1, 0, 1])


def test_a_parent_map_of_the_wrong_length_or_reach_is_refused():
    with pytest.raises(ValueError, match="against 4"):
        folding_residual(np.arange(4.0), np.zeros(2), [0, 0, 1])
    with pytest.raises(ValueError, match="outside the 2"):
        folding_residual(np.arange(4.0), np.zeros(2), [0, 0, 1, 2])
    with pytest.raises(ValueError, match="outside the 2"):
        folding_residual(np.arange(4.0), np.zeros(2), [-1, 0, 1, 1])


def test_an_empty_log_folds_onto_an_empty_scale():
    empty = np.zeros(0)
    report = folding_residual(empty, empty, np.zeros(0, dtype=np.int64))
    assert report.residual == 0.0
    assert report.epoch == -1
    assert report.n_epochs == 0
    assert CrossResolution((report,)).passed
    assert profile_residual(empty, empty, np.zeros(0, dtype=np.int64), np.sum) == 0.0


def test_a_fine_array_that_is_absent_leaves_the_whole_coarse_total_standing():
    report = folding_residual(np.zeros(0), np.array([2.0, 5.0]), [])
    assert report.residual == 5.0
    assert report.epoch == 1


def test_a_mean_profile_folds_and_a_variance_profile_does_not():
    fine, totals, parents = ladder_log(3)
    means = np.array([fine[parents == e].mean() for e in range(EPOCHS)])
    variances = np.array([fine[parents == e].var() for e in range(EPOCHS)])
    assert profile_residual(fine, totals, parents, np.sum) == 0.0
    assert profile_residual(fine, means, parents, np.mean) == 0.0
    # The coarse scale records its children's average and never their spread, so
    # the variance read off the same children stands away from that profile.
    gap = profile_residual(fine, means, parents, np.var)
    assert gap == pytest.approx(float(np.abs(variances - means).max()))
    assert gap > 0.5


def test_a_profile_needs_a_child_in_every_epoch_and_a_finite_value():
    with pytest.raises(ValueError, match="no fine slot"):
        profile_residual(np.arange(6.0), np.zeros(3), [0, 0, 0, 2, 2, 2], np.sum)
    with pytest.raises(ValueError, match="returned nan"):
        profile_residual(
            np.arange(6.0), np.zeros(2), np.repeat([0, 1], 3), lambda a: np.nan
        )


def test_an_array_that_is_not_finite_is_named_and_never_folded_to_a_nan():
    # A NaN residual compares false against zero, so an unguarded rung reports a
    # gap no reader can act on and CrossResolution calls the pair broken without
    # naming the slot that carries it.
    with pytest.raises(ValueError, match="the fine array slot 1 holds nan"):
        folding_residual(np.array([1.0, np.nan]), np.array([1.0]), [0, 0])
    with pytest.raises(ValueError, match="the coarse array slot 0 holds inf"):
        folding_residual(np.array([1.0, 1.0]), np.array([np.inf]), [0, 0])
    with pytest.raises(ValueError, match="the coarse array slot 0 holds nan"):
        profile_residual(
            np.arange(6.0), np.array([np.nan, 1.0]), np.repeat([0, 1], 3), np.sum
        )


def test_a_folding_residual_holds_a_gap_and_an_epoch_the_scale_has_got():
    with pytest.raises(ValueError, match="gap at or above zero"):
        FoldingResidual(residual=-1.0, epoch=0, n_epochs=1)
    with pytest.raises(ValueError, match="gap at or above zero"):
        FoldingResidual(residual=float("nan"), epoch=0, n_epochs=1)
    with pytest.raises(ValueError, match="epoch 9 is outside the 2"):
        FoldingResidual(residual=1.0, epoch=9, n_epochs=2)
    # The empty coarse scale is the one report that names no epoch.
    assert FoldingResidual(residual=0.0, epoch=-1, n_epochs=0).epoch == -1


# ---------------------------------------------------------------------------
# R8, the permutation baseline
# ---------------------------------------------------------------------------


def test_a_real_relation_clears_the_permuted_baseline():
    x, y, who = two_series(0, coupling=0.7)
    observed = abs_corr(y, x)
    null = [abs_corr(y, block_permutation(x, who, 30, s)) for s in range(99)]
    report = permuted_score(observed, null)
    assert report.n_at_or_above == 0
    assert report.p_value == pytest.approx(1 / 100)
    assert report.passed
    assert observed > report.null_quantile
    assert "1/100" in str(report)
    assert "passes" in str(report)


def test_the_block_null_holds_its_level_where_a_plain_reshuffle_does_not():
    blocked = 0
    plain = 0
    wider = 0
    for seed in range(20):
        x, y, who = two_series(100 + seed, coupling=0.0)
        observed = abs_corr(y, x)
        kept = permuted_score(
            observed, [abs_corr(y, block_permutation(x, who, 40, s)) for s in range(59)]
        )
        # A block of one epoch is the plain reshuffle.  It destroys the
        # persistence the two series carry, so its null sits narrower than the
        # correlation two persistent series reach on their own.
        broken = permuted_score(
            observed, [abs_corr(y, block_permutation(x, who, 1, s)) for s in range(59)]
        )
        blocked += kept.passed
        plain += broken.passed
        wider += kept.null_quantile > broken.null_quantile
    assert wider == 20
    assert plain >= 5
    assert blocked <= 3


def test_a_block_permutation_keeps_the_persistence_a_reshuffle_destroys():
    x, _, who = two_series(3, coupling=0.0)
    assert lag1_autocorrelation(x, who) > 0.8
    assert lag1_autocorrelation(block_permutation(x, who, 40, 0), who) > 0.7
    assert abs(lag1_autocorrelation(block_permutation(x, who, 1, 0), who)) < 0.2


def test_a_block_permutation_leaves_every_participant_its_own_values():
    values = np.arange(20.0)
    who = np.repeat([0, 1], 10)
    moved = block_permutation(values, who, 5, seed=5)
    assert np.array_equal(np.sort(moved[:10]), values[:10])
    assert np.array_equal(np.sort(moved[10:]), values[10:])
    assert not np.array_equal(moved, values)
    # Every five-slot window of the result is a run of the input, so whole
    # blocks moved and no block was taken apart.
    assert [np.diff(moved[i : i + 5]).tolist() for i in range(0, 20, 5)] == [
        [1.0] * 4
    ] * 4


def test_a_block_longer_than_the_participant_leaves_its_order_alone():
    values = np.arange(6.0)
    who = np.repeat([0, 1], 3)
    assert np.array_equal(block_permutation(values, who, 10, seed=0), values)


def test_an_empty_log_permutes_to_an_empty_log():
    moved = block_permutation(np.zeros(0), np.zeros(0, dtype=np.int64), 4, seed=0)
    assert np.array_equal(moved, np.zeros(0))


def test_a_block_below_one_epoch_or_a_mismatched_roster_is_refused():
    with pytest.raises(ValueError, match="at least one epoch long"):
        block_permutation(np.zeros(4), np.zeros(4, dtype=np.int64), 0)
    with pytest.raises(ValueError, match="participant labels"):
        block_permutation(np.zeros(4), np.zeros(3, dtype=np.int64), 2)


def test_the_baseline_prints_its_numerator_and_its_denominator():
    null = [0.1, 0.5, 0.9, 0.2]
    inside = permuted_score(0.5, null, alpha=0.25)
    assert inside.n_at_or_above == 2
    assert inside.p_value == pytest.approx(3 / 5)
    assert inside.null_median == pytest.approx(0.35)
    assert inside.null_quantile == pytest.approx(0.6)
    assert not inside.passed
    assert "3/5" in str(inside)
    assert "FAILS" in str(inside)
    above = permuted_score(0.95, null, alpha=0.25)
    assert above.n_at_or_above == 0
    assert above.alpha == 0.25
    assert above.passed


def test_a_baseline_with_no_null_draw_or_an_impossible_level_is_refused():
    with pytest.raises(ValueError, match="at least one null score"):
        permuted_score(1.0, [])
    with pytest.raises(ValueError, match=r"inside \(0, 1\)"):
        permuted_score(1.0, [0.0], alpha=0.0)
    with pytest.raises(ValueError, match=r"inside \(0, 1\)"):
        permuted_score(1.0, [0.0], alpha=1.0)


def test_a_baseline_whose_draws_cannot_reach_the_level_is_refused():
    # Three draws put the smallest attainable p-value at one quarter, so no
    # observed score whatever can reach a level of five percent.
    with pytest.raises(ValueError, match="never reaches the level"):
        permuted_score(1.0, [0.0, 0.5, 0.9])


def test_the_verdict_is_the_p_value_and_never_the_interpolated_quantile():
    null = list(np.linspace(0.0, 0.9, 19))
    beats_every_draw = permuted_score(0.95, null)
    assert beats_every_draw.n_at_or_above == 0
    assert beats_every_draw.p_value == pytest.approx(1 / 20)
    assert beats_every_draw.passed
    # The upper quantile interpolates between the two largest draws, so a score
    # above it still leaves a draw at or above it and a p-value of two in twenty.
    between = permuted_score(0.86, null)
    assert between.observed > between.null_quantile
    assert between.n_at_or_above == 1
    assert between.p_value == pytest.approx(2 / 20)
    assert not between.passed


def test_a_score_that_is_not_a_number_never_reaches_the_smallest_p_value():
    # nan and inf compare false against every draw, so an unguarded baseline
    # counts nothing at or above them and hands back 1/(1+n), the most
    # significant number the permutation can reach, on an observation that is
    # not a number and a null it was never read against.
    null = list(np.linspace(0.0, 0.5, 99))
    with pytest.raises(ValueError, match="an observed score is a finite number"):
        permuted_score(float("nan"), null)
    with pytest.raises(ValueError, match="an observed score is a finite number"):
        permuted_score(float("inf"), null)
    null[3] = np.nan
    with pytest.raises(ValueError, match="a null score slot 3 holds nan"):
        permuted_score(0.9, null)
    null[3] = np.inf
    with pytest.raises(ValueError, match="a null score slot 3 holds inf"):
        permuted_score(0.9, null)


def test_a_baseline_cannot_be_built_to_print_a_count_it_never_took():
    with pytest.raises(ValueError, match="4 of 3 null draws"):
        PermutationBaseline(0.5, 0.5, 0.4, 0.6, 4, 3, DEFAULT_ALPHA)
    with pytest.raises(ValueError, match="does not count 1 of 3"):
        PermutationBaseline(0.5, 0.001, 0.4, 0.6, 1, 3, DEFAULT_ALPHA)
    with pytest.raises(ValueError, match=r"inside \(0, 1\)"):
        PermutationBaseline(0.5, 0.5, 0.4, 0.6, 1, 3, 5.0)
    with pytest.raises(ValueError, match="finite scores"):
        PermutationBaseline(0.5, 0.5, 0.4, float("nan"), 1, 3, DEFAULT_ALPHA)


# ---------------------------------------------------------------------------
# R11, the held-out likelihood
# ---------------------------------------------------------------------------


def test_the_sign_convention_is_positive_when_the_full_model_predicts_better():
    assert held_out_gain(-90.0, -100.0, 10) == pytest.approx(1.0)
    assert held_out_gain(-100.0, -90.0, 10) == pytest.approx(-1.0)
    assert HeldOut(-90.0, -100.0, 10, held_out_gain(-90.0, -100.0, 10)).passed
    assert not HeldOut(-100.0, -90.0, 10, held_out_gain(-100.0, -90.0, 10)).passed
    with pytest.raises(ValueError, match="at least one slot"):
        held_out_gain(-90.0, -100.0, 0)


def test_a_parent_that_predicts_wins_the_held_out_comparison():
    full, base, who = two_models(0, coupling=0.8)
    kept, dropped, slots = float(full.sum()), float(base.sum()), int(full.size)
    report = HeldOut(kept, dropped, slots, held_out_gain(kept, dropped, slots))
    assert report.passed
    assert report.gain > 0.1
    gains, p_value = paired_gain(full, base, who)
    assert gains.size == 12
    assert int((gains > 0.0).sum()) == 12
    assert p_value <= DEFAULT_ALPHA


def test_the_held_out_rung_does_not_find_prediction_where_no_parent_predicts():
    reached = 0
    per_slot = []
    for seed in range(20):
        full, base, who = two_models(300 + seed, coupling=0.0)
        per_slot.append(
            held_out_gain(float(full.sum()), float(base.sum()), int(full.size))
        )
        reached += paired_gain(full, base, who, seed=seed)[1] <= DEFAULT_ALPHA
    assert reached <= 2
    assert float(np.mean(per_slot)) < 0.005


def test_one_participant_never_reaches_the_paired_level():
    who = np.zeros(40, dtype=np.int64)
    gains, p_value = paired_gain(
        np.full(40, -1.0), np.full(40, -5.0), who, n_flips=199, seed=0
    )
    assert gains.size == 1
    assert gains[0] == pytest.approx(4.0)
    assert p_value > DEFAULT_ALPHA


def test_the_paired_test_reads_participants_and_never_slots():
    who = np.repeat(np.arange(10), 20)
    lopsided = np.full(200, -0.02)
    lopsided[:20] = 5.0
    gains, p_value = paired_gain(lopsided, np.zeros(200), who)
    assert gains[0] == pytest.approx(5.0)
    assert float(gains.mean()) > 0.4
    assert p_value > DEFAULT_ALPHA
    # A gain of the same mean that every participant carries is what the sign
    # flip over participants can call a gain.
    even = np.full(200, float(gains.mean()))
    _, p_shared = paired_gain(even, np.zeros(200), who)
    assert p_shared <= DEFAULT_ALPHA


def test_a_paired_test_needs_a_participant_and_arrays_that_match():
    with pytest.raises(ValueError, match="participant labels"):
        paired_gain(np.zeros(4), np.zeros(4), np.zeros(3, dtype=np.int64))
    with pytest.raises(ValueError, match="at least one participant"):
        paired_gain(np.zeros(0), np.zeros(0), np.zeros(0, dtype=np.int64))


def test_a_likelihood_that_is_not_finite_buys_no_gain_at_all():
    # A base model that gave a held-out slot no probability at all carries a log
    # likelihood of -inf, and an unguarded difference hands the full model an
    # infinite win on one slot.
    with pytest.raises(ValueError, match="is a finite number"):
        held_out_gain(-90.0, float("-inf"), 10)
    with pytest.raises(ValueError, match="is a finite number"):
        held_out_gain(float("nan"), -100.0, 10)
    who = np.array([0, 1], dtype=np.int64)
    with pytest.raises(ValueError, match="full model's held-out likelihood slot 0"):
        paired_gain(np.array([np.nan, 1.0]), np.zeros(2), who)
    with pytest.raises(ValueError, match="base model's held-out likelihood slot 1"):
        paired_gain(np.ones(2), np.array([0.0, -np.inf]), who)


def test_a_held_out_report_cannot_carry_a_gain_its_own_numbers_deny():
    with pytest.raises(ValueError, match="a gain of 5.0 is not the -1.0"):
        HeldOut(-100.0, -90.0, 10, 5.0)
    with pytest.raises(ValueError, match="at least one slot"):
        HeldOut(-100.0, -90.0, 0, 0.0)


def test_a_sign_flip_test_needs_a_flip_to_take():
    # No flip leaves a reference distribution of nothing and a p-value of one,
    # which reads as a gain the test refused instead of a test never taken.
    with pytest.raises(ValueError, match="at least one flip"):
        paired_gain(np.ones(4), np.zeros(4), np.repeat([0, 1], 2), n_flips=0)
