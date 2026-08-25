"""
Tests for the Poisson hidden Markov model the two regime rows share.

What is pinned here is that the fit finds the regimes a series was generated
with and finds one level in a series that holds one level, that every pass
survives a series long enough to underflow a fit carried on probabilities, that
the criterion keeps the state count the series pays for, and that the
cross-scale coupling is one when a fine path follows its parent and near zero
when it runs free of it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.evaluation.count_hmm import (
    HOLD,
    MIN_EPOCHS_PER_STATE,
    RATE_FLOOR,
    STATE_COUNTS,
    TOLERANCE,
    ForwardBackward,
    PoissonHMM,
    _log,
    adjusted_rand,
    admissible_states,
    bic,
    compare_regimes,
    cross_scale_coupling,
    decode,
    empirical_dwell,
    entropy,
    expected_dwell,
    fit_poisson_hmm,
    fit_regimes,
    forward_backward,
    initial_parameters,
    log_emissions,
    m_step,
    match_rates,
    regime_coupling,
    regime_entropy,
    select_states,
    stationary_distribution,
    total_variation,
    viterbi,
)


def two_regime(seed: int = 0, n_blocks: int = 10, block: int = 40):
    """Counts alternating between rate one and rate nine every `block` epochs."""
    rng = np.random.default_rng(seed)
    truth = np.repeat(np.arange(n_blocks) % 2, block)
    return rng.poisson(np.where(truth == 1, 9.0, 1.0)).astype(float), truth


def one_regime(seed: int = 11, n: int = 400) -> np.ndarray:
    """Counts drawn at one rate throughout, which carries no regime to find."""
    return np.random.default_rng(seed).poisson(4.0, n).astype(float)


def crafted_posterior(gamma: np.ndarray, xi_sum: np.ndarray) -> ForwardBackward:
    """A posterior written by hand, so an M-step can be read on known weights."""
    empty = np.zeros_like(gamma)
    return ForwardBackward(empty, empty, gamma, xi_sum, 0.0)


# ---------------------------------------------------------------------------
# The emission and the two passes
# ---------------------------------------------------------------------------


def test_the_emission_is_the_poisson_probability_of_the_count_under_the_rate():
    assert float(np.exp(log_emissions([2.0], [3.0])[0, 0])) == pytest.approx(
        9.0 * math.exp(-3.0) / 2.0
    )
    # A rate of zero is floored, so a series of zeros stays readable.
    assert float(np.exp(log_emissions([0.0], [0.0])[0, 0])) == pytest.approx(1.0)


def test_the_forward_pass_of_a_one_state_model_is_the_summed_emission():
    emission = log_emissions([0.0, 3.0, 2.0, 7.0], [2.0])
    state = forward_backward(_log(np.array([1.0])), _log(np.array([[1.0]])), emission)
    assert state.log_likelihood == pytest.approx(float(emission.sum()))
    assert state.gamma.ravel().tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0])
    # Four epochs carry three transitions, and one state takes all of them.
    assert state.xi_sum == pytest.approx(np.array([[3.0]]))


def test_a_pass_over_one_epoch_counts_no_transition_and_still_weighs_the_states():
    emission = log_emissions([3.0], [1.0, 5.0])
    state = forward_backward(_log(np.full(2, 0.5)), _log(np.eye(2)), emission)
    assert state.xi_sum.tolist() == [[0.0, 0.0], [0.0, 0.0]]
    assert float(state.gamma.sum()) == pytest.approx(1.0)
    assert state.gamma[0, 1] > state.gamma[0, 0]


def test_the_backward_pass_makes_the_posterior_agree_with_the_forward_one():
    values, _ = two_regime(seed=2, n_blocks=4, block=20)
    model = fit_poisson_hmm(values, 2)
    state = forward_backward(
        _log(model.initial), _log(model.transition), log_emissions(values, model.rates)
    )
    assert state.gamma.sum(axis=1) == pytest.approx(np.ones(values.size))
    assert float(state.xi_sum.sum()) == pytest.approx(values.size - 1.0)


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------


def test_a_series_of_three_thousand_epochs_keeps_a_finite_and_climbing_likelihood():
    values, _ = two_regime(seed=7, n_blocks=60, block=50)
    model = fit_poisson_hmm(values, 2, max_iterations=15)
    history = np.array(model.history)
    assert values.size == 3000
    assert np.all(np.isfinite(history))
    assert np.all(np.diff(history) >= -1e-8)
    # The joint probability of this series sits below the smallest positive
    # double, so a forward pass carried on probabilities returns zero.
    assert model.log_likelihood < math.log(float(np.finfo(float).tiny))
    assert model.log_likelihood == pytest.approx(history[-1])


def test_the_declared_cap_stops_a_fit_the_tolerance_has_not_stopped_yet():
    values, _ = two_regime()
    capped = fit_poisson_hmm(values, 2, max_iterations=2)
    free = fit_poisson_hmm(values, 2)
    assert len(capped.history) == 2
    assert capped.history[1] - capped.history[0] > TOLERANCE
    # The cap leaves EM one M-step past its last reported iteration, and the
    # fit it was stopped short of stands higher.
    assert capped.log_likelihood > capped.history[-1]
    assert free.log_likelihood > capped.log_likelihood


def test_the_fit_recovers_the_two_rates_the_series_was_generated_from():
    values, _ = two_regime()
    model = fit_poisson_hmm(values, 2)
    assert isinstance(model, PoissonHMM)
    assert model.n_states == 2
    assert model.rates[0] == pytest.approx(1.0, abs=0.4)
    assert model.rates[1] == pytest.approx(9.0, abs=0.9)
    assert len(model.history) < 100


def test_the_states_come_back_ordered_by_rate_so_two_fits_carry_one_labelling():
    model = fit_poisson_hmm(two_regime()[0], 3)
    assert np.all(np.diff(model.rates) > 0.0)
    assert float(model.transition.sum()) == pytest.approx(3.0)
    assert float(model.initial.sum()) == pytest.approx(1.0)


def test_the_fitted_transition_holds_a_regime_for_about_as_long_as_it_was_built_to():
    values, _ = two_regime(block=40)
    dwell = expected_dwell(fit_poisson_hmm(values, 2).transition)
    assert np.all(dwell > 15.0)
    assert np.all(dwell < 90.0)


def test_one_seed_gives_one_fit_and_a_second_seed_finds_the_same_regimes():
    values, _ = two_regime()
    first = fit_poisson_hmm(values, 2, seed=0)
    again = fit_poisson_hmm(values, 2, seed=0)
    other = fit_poisson_hmm(values, 2, seed=5)
    assert first.rates.tolist() == again.rates.tolist()
    assert first.rates.tolist() == pytest.approx(other.rates.tolist(), abs=0.05)


def test_the_initialization_reads_the_quantiles_and_starts_from_a_held_state():
    rates, transition, initial = initial_parameters(np.arange(100.0), 2, seed=0)
    assert rates[0] == pytest.approx(25.0, abs=10.0)
    assert rates[1] == pytest.approx(75.0, abs=10.0)
    assert transition[0, 0] == HOLD
    assert initial.tolist() == [0.5, 0.5]
    # A constant series gives one quantile twice, and the jitter separates them.
    assert len(set(initial_parameters(np.full(50, 3.0), 2)[0].tolist())) == 2


# ---------------------------------------------------------------------------
# The decoding
# ---------------------------------------------------------------------------


def test_the_decoded_path_puts_each_epoch_in_the_regime_that_generated_it():
    values, truth = two_regime()
    path = decode(values, fit_poisson_hmm(values, 2))
    assert float(np.mean(path == truth)) > 0.95


def test_the_decoding_keeps_a_lone_deviant_epoch_inside_its_sticky_regime():
    values = np.full(21, 1.0)
    values[10] = 5.0
    emission = log_emissions(values, [1.0, 9.0])
    start = _log(np.full(2, 0.5))
    sticky = viterbi(start, _log(np.array([[0.99, 0.01], [0.01, 0.99]])), emission)
    loose = viterbi(start, _log(np.full((2, 2), 0.5)), emission)
    assert sticky.tolist() == [0] * 21
    assert loose[10] == 1
    assert loose.tolist().count(1) == 1


# ---------------------------------------------------------------------------
# How many regimes
# ---------------------------------------------------------------------------


def test_the_criterion_picks_the_number_of_regimes_the_series_was_built_with():
    values, _ = two_regime()
    model, scores = select_states(values, (1, 2, 3))
    prices = dict(scores)
    assert model.n_states == 2
    assert prices[2] < prices[1]
    assert prices[2] < prices[3]


def test_a_series_holding_one_rate_throughout_is_fitted_with_one_regime():
    assert select_states(one_regime(), (1, 2, 3))[0].n_states == 1


def test_the_criterion_prices_a_state_count_by_the_parameters_it_frees():
    # A three-state model holds nine rates, transitions and starts against three.
    assert bic(-100.0, 3, 100) - bic(-100.0, 1, 100) == pytest.approx(
        10.0 * math.log(100.0)
    )
    assert bic(-90.0, 2, 50) < bic(-100.0, 2, 50)


def test_a_state_count_the_series_is_too_short_for_is_never_tried():
    assert MIN_EPOCHS_PER_STATE == 10
    assert admissible_states(400, STATE_COUNTS) == (1, 2, 3, 4)
    assert admissible_states(25, STATE_COUNTS) == (1, 2)
    # One state is always admissible, since a series always holds one level.
    assert admissible_states(4, (2, 3)) == (1,)


# ---------------------------------------------------------------------------
# What the chain implies
# ---------------------------------------------------------------------------


def test_the_stationary_distribution_is_the_row_the_chain_leaves_unchanged():
    chain = np.array([[0.9, 0.1], [0.2, 0.8]])
    resting = stationary_distribution(chain)
    assert resting == pytest.approx([2.0 / 3.0, 1.0 / 3.0])
    assert resting @ chain == pytest.approx(resting)


def test_the_expected_dwell_is_one_over_the_leaving_chance_and_endless_if_absorbing():
    assert expected_dwell(np.array([[0.9, 0.1], [0.2, 0.8]])) == pytest.approx(
        [10.0, 5.0]
    )
    assert expected_dwell(np.eye(2)).tolist() == [math.inf, math.inf]


def test_the_empirical_dwell_is_the_mean_run_length_and_an_absent_state_has_none():
    assert empirical_dwell([0, 0, 0, 1, 1, 0], 3).tolist() == [2.0, 2.0, 0.0]
    assert empirical_dwell([1, 1, 1, 1], 2).tolist() == [0.0, 4.0]


def test_the_decoded_dwell_agrees_with_the_dwell_the_fitted_chain_implies():
    values, _ = two_regime(block=40)
    fit = fit_regimes(values, (1, 2, 3))
    assert fit.n_states == 2
    assert fit.empirical_dwell == pytest.approx(fit.expected_dwell, rel=0.5)


def test_the_entropy_of_a_balanced_pair_is_log_two_and_of_one_regime_is_zero():
    assert entropy([0.5, 0.5]) == pytest.approx(math.log(2.0))
    assert entropy([1.0]) == 0.0
    assert entropy([1.0, 0.0]) == 0.0
    # Unnormalized mass names the same distribution.
    assert entropy([2.0, 2.0]) == pytest.approx(math.log(2.0))


def test_the_record_reports_the_criterion_at_every_count_it_tried():
    values, _ = two_regime()
    fit = fit_regimes(values, (1, 2, 3))
    assert [count for count, _ in fit.bic_by_state] == [1, 2, 3]
    assert fit.path.size == values.size
    assert float(fit.stationary.sum()) == pytest.approx(1.0)
    assert fit.entropy == pytest.approx(math.log(2.0), abs=0.15)


# ---------------------------------------------------------------------------
# The M-step
# ---------------------------------------------------------------------------


def test_the_m_step_puts_each_rate_at_the_mean_of_the_epochs_its_state_holds():
    gamma = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    posterior = crafted_posterior(gamma, np.array([[1.0, 1.0], [0.0, 1.0]]))
    rates, transition, initial = m_step([1.0, 3.0, 8.0, 10.0], posterior)
    assert rates == pytest.approx([2.0, 9.0])
    assert transition == pytest.approx(np.array([[0.5, 0.5], [0.0, 1.0]]))
    assert initial.tolist() == [1.0, 0.0]


def test_a_state_the_posterior_never_occupies_gets_a_uniform_row():
    gamma = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    posterior = crafted_posterior(gamma, np.array([[2.0, 0.0], [0.0, 0.0]]))
    rates, transition, _ = m_step([1.0, 1.0, 1.0], posterior)
    assert transition[0].tolist() == [1.0, 0.0]
    assert transition[1].tolist() == [0.5, 0.5]
    assert rates[1] == RATE_FLOOR


# ---------------------------------------------------------------------------
# How two fits are compared
# ---------------------------------------------------------------------------


def test_the_total_variation_is_zero_on_one_distribution_and_one_on_disjoint_ones():
    assert total_variation([0.5, 0.5], [0.5, 0.5]) == 0.0
    assert total_variation([1.0, 0.0], [0.0, 1.0]) == 1.0
    # A two-state fit against a three-state one holds no mass on the third.
    assert total_variation([0.5, 0.5], [0.5, 0.25, 0.25]) == pytest.approx(0.25)


def test_the_rate_matching_pairs_each_state_with_its_nearest_in_any_order():
    pairs, gaps = match_rates([1.0, 9.0], [8.6, 1.2])
    assert pairs.tolist() == [[0, 1], [1, 0]]
    assert gaps == pytest.approx([0.2, 0.4])
    # The larger set leaves its unmatched state out.
    assert match_rates([1.0, 9.0], [1.1])[1] == pytest.approx([0.1])


def test_the_adjusted_rand_index_is_one_on_a_relabelling_and_near_zero_on_noise():
    labels = [0, 0, 1, 1, 2, 2]
    assert adjusted_rand(labels, [2, 2, 0, 0, 1, 1]) == pytest.approx(1.0)
    assert adjusted_rand(labels, [0, 1, 2, 0, 1, 2]) < 0.0
    rng = np.random.default_rng(3)
    independent = [
        adjusted_rand(rng.integers(0, 3, 300), rng.integers(0, 3, 300))
        for _ in range(20)
    ]
    assert abs(float(np.mean(independent))) < 0.02


def test_two_labelings_holding_one_cluster_each_agree_trivially():
    assert adjusted_rand([0, 0, 0], [0, 0, 0]) == 1.0
    assert adjusted_rand([0, 0, 0], [1, 1, 1]) == 1.0
    # One epoch holds no pair to agree or disagree on.
    assert adjusted_rand([4], [7]) == 1.0
    assert adjusted_rand([0, 0, 0], [0, 1, 2]) == 0.0


def test_the_coupling_is_one_when_the_fine_path_follows_its_parent():
    parent = np.repeat(np.arange(10), 4)
    coarse = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0])
    assert cross_scale_coupling(coarse[parent], coarse, parent) == pytest.approx(1.0)


def test_a_fine_path_running_free_of_its_parent_carries_no_coupling():
    parent = np.repeat(np.arange(50), 4)
    coarse = np.tile([0, 1], 25)
    rng = np.random.default_rng(5)
    loose = rng.integers(0, 2, parent.size)
    assert cross_scale_coupling(loose, coarse, parent) < 0.05
    # A fine path holding one regime throughout follows nothing either.
    assert cross_scale_coupling(np.zeros(200, dtype=int), coarse, parent) == 0.0


def test_a_second_draw_of_one_world_matches_and_a_flat_synthetic_log_does_not():
    values, _ = two_regime(seed=0)
    close = compare_regimes(values, two_regime(seed=1)[0], (1, 2, 3))
    assert close.state_difference == 0
    assert close.stationary_tv < 0.1
    assert close.rate_gap < 1.0
    assert close.entropy_difference < 0.1
    far = compare_regimes(values, one_regime(), (1, 2, 3))
    assert far.state_difference == 1
    assert far.entropy_difference > 0.5
    assert far.rate_gap > close.rate_gap
    assert far.dwell_difference > close.dwell_difference
    assert set(far.as_row()) == {
        "state_difference",
        "stationary_tv",
        "rate_gap",
        "entropy_difference",
        "dwell_difference",
    }


# ---------------------------------------------------------------------------
# The summary scalar
# ---------------------------------------------------------------------------


def test_the_summary_scalar_separates_a_two_regime_series_from_a_flat_one():
    values, _ = two_regime()
    assert regime_entropy(values, 24.0) == pytest.approx(math.log(2.0), abs=0.15)
    assert regime_entropy(one_regime(), 24.0) == 0.0


def test_the_summary_scalar_is_finite_on_every_series_a_participant_can_hand_it():
    for series in (
        np.zeros(0),
        np.zeros(400),
        np.full(400, 7.0),
        np.arange(20.0),
        one_regime(seed=2, n=56),
    ):
        value = regime_entropy(series, 24.0)
        assert math.isfinite(value)
        assert value == 0.0


def test_a_series_shorter_than_two_calendar_cycles_is_reported_as_one_regime():
    values, _ = two_regime(n_blocks=2, block=10)
    assert regime_entropy(values, 24.0) == 0.0
    # The same series at the day scale carries twenty epochs of a seven-epoch
    # cycle, which is enough to read.
    assert regime_entropy(values, 7.0) > 0.0


# ---------------------------------------------------------------------------
# The guards
# ---------------------------------------------------------------------------


def test_a_series_that_is_not_a_count_series_is_refused():
    with pytest.raises(ValueError, match="non-negative"):
        fit_poisson_hmm([1.0, -2.0], 1)
    with pytest.raises(ValueError, match="finite"):
        fit_poisson_hmm([1.0, np.nan], 1)


def test_a_series_shorter_than_the_state_count_is_refused():
    with pytest.raises(ValueError, match="needs 3 epochs"):
        fit_poisson_hmm([1.0, 2.0], 3)
    with pytest.raises(ValueError, match="at least one state"):
        initial_parameters([1.0, 2.0], 0)


def test_an_iteration_cap_or_a_tolerance_that_cannot_work_is_refused():
    with pytest.raises(ValueError, match="iteration cap"):
        fit_poisson_hmm([1.0, 2.0], 1, max_iterations=0)
    with pytest.raises(ValueError, match="tolerance is positive"):
        fit_poisson_hmm([1.0, 2.0], 1, tolerance=0.0)


def test_a_parameter_set_carrying_no_state_is_refused():
    with pytest.raises(ValueError, match="at least one state"):
        forward_backward(np.zeros(0), np.zeros((0, 0)), np.zeros((3, 0)))
    with pytest.raises(ValueError, match="at least one state"):
        viterbi(np.zeros(0), np.zeros((0, 0)), np.zeros((3, 0)))


def test_parameters_the_series_cannot_have_come_from_are_refused():
    emission = log_emissions([1.0, 2.0], [1.0, 5.0])
    even_start = _log(np.full(2, 0.5))
    even_chain = _log(np.full((2, 2), 0.5))
    dead_start = _log(np.zeros(2))
    dead_chain = _log(np.zeros((2, 2)))
    # A start carrying no mass, and a chain leaving every state nowhere to go,
    # each drive the probability of the series to zero, and every posterior and
    # every path read off that zero is undefined.
    with pytest.raises(ValueError, match="log probability -inf"):
        forward_backward(dead_start, even_chain, emission)
    with pytest.raises(ValueError, match="log probability -inf"):
        forward_backward(even_start, dead_chain, emission)
    with pytest.raises(ValueError, match="log probability -inf"):
        viterbi(dead_start, even_chain, emission)
    with pytest.raises(ValueError, match="log probability -inf"):
        viterbi(even_start, dead_chain, emission)


def test_a_poisson_rate_no_count_can_come_from_is_refused():
    with pytest.raises(ValueError, match="rate is finite"):
        log_emissions([1.0], [np.nan])
    with pytest.raises(ValueError, match="rate is finite"):
        log_emissions([1.0], [np.inf])
    with pytest.raises(ValueError, match="rate is non-negative"):
        log_emissions([1.0], [-5.0])


def test_a_pass_over_no_epoch_or_over_disagreeing_shapes_is_refused():
    with pytest.raises(ValueError, match="at least one epoch"):
        forward_backward(np.zeros(2), np.zeros((2, 2)), np.zeros((0, 2)))
    with pytest.raises(ValueError, match="one state count"):
        viterbi(np.zeros(2), np.zeros((2, 2)), np.zeros((5, 3)))
    with pytest.raises(ValueError, match="at least one rate"):
        log_emissions([1.0], [])
    with pytest.raises(ValueError, match="one posterior row per epoch"):
        m_step([1.0, 2.0], crafted_posterior(np.ones((3, 1)), np.zeros((1, 1))))


def test_a_matrix_that_is_not_a_transition_matrix_is_refused():
    with pytest.raises(ValueError, match="square and filled"):
        stationary_distribution(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="summing to one"):
        expected_dwell(np.array([[0.5, 0.2], [0.1, 0.9]]))


def test_a_reading_with_no_state_or_no_epoch_behind_it_is_refused():
    with pytest.raises(ValueError, match="at least one state"):
        entropy([])
    with pytest.raises(ValueError, match="carries mass"):
        entropy([0.0, 0.0])
    with pytest.raises(ValueError, match="carries mass"):
        entropy([-1.0, 2.0])
    # A non-finite entry passes both readings above, so it is named on its own.
    with pytest.raises(ValueError, match="distribution is finite"):
        entropy([np.nan, 1.0])
    with pytest.raises(ValueError, match="distribution is finite"):
        entropy([np.inf, 1.0])
    with pytest.raises(ValueError, match="at least one epoch"):
        bic(-10.0, 2, 0)
    with pytest.raises(ValueError, match="at least one state"):
        bic(-10.0, 0, 100)
    with pytest.raises(ValueError, match="at least one epoch"):
        admissible_states(0)
    with pytest.raises(ValueError, match="at least one state"):
        total_variation([], [])


def test_a_comparison_missing_a_side_or_an_epoch_set_is_refused():
    with pytest.raises(ValueError, match="state on each side"):
        match_rates([], [1.0])
    with pytest.raises(ValueError, match="one epoch set"):
        adjusted_rand([0, 1], [0])
    with pytest.raises(ValueError, match="at least one epoch"):
        adjusted_rand([], [])
    with pytest.raises(ValueError, match="one entry per fine epoch"):
        cross_scale_coupling([0, 1], [0], [0])
    with pytest.raises(ValueError, match="inside the coarse path"):
        cross_scale_coupling([0, 1], [0], [0, 5])


def test_a_decoded_state_outside_the_state_set_is_refused():
    with pytest.raises(ValueError, match="at least one state"):
        empirical_dwell([0], 0)
    with pytest.raises(ValueError, match="inside the state set"):
        empirical_dwell([0, 2], 2)
    assert empirical_dwell([], 3).tolist() == [0.0, 0.0, 0.0]


def test_the_summary_scalar_reads_a_series_shorter_than_any_multi_state_fit():
    # A period this small lets these reach the fit, which then holds fewer
    # epochs than a two-state count needs and reports the one level it carries.
    for series in (np.array([3.0]), np.zeros(2), np.full(9, 2.0)):
        value = regime_entropy(series, 0.5)
        assert isinstance(value, float)
        assert math.isfinite(value)
        assert value == 0.0


def test_a_calendar_period_that_names_no_cycle_is_refused():
    with pytest.raises(ValueError, match="period is positive"):
        regime_entropy([1.0, 2.0], 0.0)


def test_a_regime_that_turns_at_a_parent_boundary_couples_across_the_scales():
    # Twenty-four parent blocks of ten epochs, the level held inside each block,
    # so the fine path and the folded path turn at the same epoch.  The parent
    # series needs `MIN_EPOCHS_PER_STATE` epochs per state of its own, which is
    # what puts the floor of twenty parent blocks under a two-state coupling.
    rng = np.random.default_rng(7)
    levels = [2.0] * 12 + [40.0] * 12
    fine = np.concatenate([rng.poisson(rate, 10) for rate in levels]).astype(float)
    assert regime_coupling(fine, 10.0) > 0.9


def test_a_series_with_no_regime_leaves_both_paths_flat_and_couples_at_one():
    rng = np.random.default_rng(8)
    flat = rng.poisson(5.0, 120).astype(float)
    coupling = regime_coupling(flat, 10.0)
    # One regime at both scales is one cluster at both, which the adjusted Rand
    # index scores at one, and the row reads that as a coupled pair.
    assert coupling == pytest.approx(1.0)


def test_a_series_holding_under_two_parent_blocks_reports_no_coupling():
    assert regime_coupling(np.arange(19.0), 10.0) == 0.0
    assert regime_coupling(np.zeros(0), 10.0) == 0.0


def test_a_parent_block_of_under_two_epochs_is_refused():
    with pytest.raises(ValueError, match="at least two epochs"):
        regime_coupling(np.arange(50.0), 1.0)
    with pytest.raises(ValueError, match="at least two epochs"):
        regime_coupling(np.arange(50.0), float("inf"))
