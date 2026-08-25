"""
tests/test_phase50_fine_sem.py
==============================
The latent-state fit, and the anchor that decides whether it works.

A scale-k regime enters the one fine predictor additively, so its pseudo-emission
is the segment sum of the per-slot contributions and `forward_backward` runs on
it unchanged.  That is the upward message doing the work it was built for.

The anchor is the part that had to be measured.  A quadratic anchored at the
predictor without the chain's own effect ranks the states by their quadratic
penalty, so the state with the smallest effect wins everywhere and the chain
recovers nothing.  Anchored at the predictor's second moment under the current
posterior, the same data give exact recovery.  Both numbers are pinned here.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.typed_graph import COUNT, DURATION, TypedGraph
from src.inference.bound import pg_mean, segment_sum, upward_message
from src.learning.fine_sem import (
    FineSEM,
    FitHistory,
    RegimeChain,
    latent_step,
    update_transitions,
)
from src.model.intensity import (
    FineParams,
    ancestor_maps,
    empty_sources,
    linear_predictor,
    risk_set,
    sample,
)

DAYS = 120
SLOTS_PER_DAY = 288
T = DAYS * SLOTS_PER_DAY
PARENT_MAPS = [np.repeat(np.arange(DAYS), SLOTS_PER_DAY)]
ANCESTORS = ancestor_maps(PARENT_MAPS, T)
COUNTS = [T, DAYS]
EFFECT = np.array([[0.0], [1.6]])


def _world(seed: int = 0, persistence: float = 0.9):
    """A two-state day regime, quiet days and busy days, and the log it draws."""
    rng = np.random.default_rng(seed)
    states = np.zeros(DAYS, dtype=int)
    for d in range(1, DAYS):
        states[d] = states[d - 1] if rng.random() < persistence else 1 - states[d - 1]
    sources = empty_sources(ANCESTORS, [np.zeros((T, 0)), np.zeros((DAYS, 0))], 1)
    params = FineParams(n_labels=1)
    params.base[COUNT][:] = -4.0
    params.base[DURATION][:] = np.log(3.0)
    params.mark_dispersion = np.array([5.0])
    graph = TypedGraph(label_names=["a"])
    X1, Y1 = sample(
        params, graph, sources, rng, offsets={COUNT: EFFECT[states][ANCESTORS[1]]}
    )
    return states, X1, Y1, params, graph, sources


def _chain() -> RegimeChain:
    chain = RegimeChain.uniform(scale=2, n_states=2, n_labels=1, n_epochs=DAYS)
    chain.effect = EFFECT.copy()
    return chain


def _accuracy(gamma: np.ndarray, states: np.ndarray) -> float:
    got = gamma.argmax(axis=1)
    return float(max((got == states).mean(), (1 - got == states).mean()))


# ---------------------------------------------------------------------------
# The fit recovers the path
# ---------------------------------------------------------------------------


def test_the_latent_path_is_recovered() -> None:
    states, X1, Y1, params, graph, sources = _world()
    _, chains, _ = FineSEM(max_sweeps=10).fit(
        X1, Y1, params, graph, sources, [_chain()], COUNTS
    )
    assert _accuracy(chains[0].gamma, states) == 1.0


def test_the_data_actually_carry_the_regime() -> None:
    """The recovery means nothing if the busy days are not busier."""
    states, X1, _, _, _, _ = _world()
    per_day = segment_sum(X1.astype(float), ANCESTORS[1], DAYS)[:, 0]
    assert per_day[states == 1].mean() > 3 * per_day[states == 0].mean()


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_recovery_holds_across_draws(seed) -> None:
    states, X1, Y1, params, graph, sources = _world(seed)
    _, chains, _ = FineSEM(max_sweeps=10).fit(
        X1, Y1, params, graph, sources, [_chain()], COUNTS
    )
    assert _accuracy(chains[0].gamma, states) > 0.95


def test_the_bound_never_falls() -> None:
    _, X1, Y1, params, graph, sources = _world()
    _, _, history = FineSEM(max_sweeps=10).fit(
        X1, Y1, params, graph, sources, [_chain()], COUNTS
    )
    assert history.monotone, history.values
    assert history.values[-1] > history.values[0]
    assert len(history.values) == len(history.steps)


def test_the_schedule_does_not_change_the_answer() -> None:
    states, X1, Y1, params, graph, sources = _world()
    reached = []
    for schedule in ("coarse_to_fine", "fine_to_coarse"):
        _, chains, history = FineSEM(max_sweeps=10, schedule=schedule).fit(
            X1, Y1, params, graph, sources, [_chain()], COUNTS
        )
        reached.append((history.values[-1], _accuracy(chains[0].gamma, states)))
    assert reached[0][1] == reached[1][1] == 1.0
    assert abs(reached[0][0] - reached[1][0]) < 1e-6


def test_two_chains_at_two_scales_run_together() -> None:
    """A ladder with an hour chain and a day chain, on one bound."""
    hour_maps = [np.repeat(np.arange(T // 12), 12), np.repeat(np.arange(DAYS), 24)]
    ancestors = ancestor_maps(hour_maps, T)
    counts = [T, T // 12, DAYS]
    rng = np.random.default_rng(21)
    states = np.zeros(DAYS, dtype=int)
    for d in range(1, DAYS):
        states[d] = states[d - 1] if rng.random() < 0.9 else 1 - states[d - 1]

    sources = empty_sources(
        ancestors, [np.zeros((T, 0)), np.zeros((T // 12, 0)), np.zeros((DAYS, 0))], 1
    )
    params = FineParams(n_labels=1)
    params.base[COUNT][:] = -4.0
    params.base[DURATION][:] = np.log(3.0)
    params.mark_dispersion = np.array([5.0])
    graph = TypedGraph(label_names=["a"])
    X1, Y1 = sample(
        params, graph, sources, rng, offsets={COUNT: EFFECT[states][ancestors[2]]}
    )

    chains = [
        RegimeChain.uniform(2, 2, 1, counts[1]),
        RegimeChain.uniform(3, 2, 1, counts[2]),
    ]
    chains[1].effect = EFFECT.copy()
    _, out, history = FineSEM(max_sweeps=3).fit(
        X1, Y1, params, graph, sources, chains, counts
    )
    assert history.monotone, history.values
    assert {c.scale for c in out} == {2, 3}
    day_chain = [c for c in out if c.scale == 3][0]
    assert _accuracy(day_chain.gamma, states) > 0.9


# ---------------------------------------------------------------------------
# The anchor, which is what makes the latent step work
# ---------------------------------------------------------------------------


def test_an_anchor_without_the_chain_effect_ranks_the_states_wrongly() -> None:
    """
    The bug this loop was built to find, pinned as a number.

    Anchored at the predictor with the chain's effect left out, the quadratic
    charges each state its own penalty and never credits the effect, so the state
    with the smallest effect wins on every epoch.  The message still separates
    busy days from quiet ones; the ranking is what breaks.
    """
    states, X1, Y1, params, graph, sources = _world()
    chain = _chain()
    eta_rest = linear_predictor(params, graph, sources, COUNT)
    at_risk = risk_set(X1, Y1)

    naive = upward_message(
        X1,
        eta_rest,
        chain.effect,
        pg_mean(np.abs(eta_rest)),
        ANCESTORS[1],
        DAYS,
        at_risk,
    )
    assert _accuracy(naive, states) < 0.7
    picks_the_small_effect = float((naive.argmax(axis=1) == 0).mean())
    assert picks_the_small_effect > 0.9, "the smallest effect wins almost everywhere"

    gap = naive[:, 1] - naive[:, 0]
    assert gap[states == 1].mean() > gap[states == 0].mean() + 10.0


def test_the_second_moment_anchor_recovers_the_ranking() -> None:
    states, X1, Y1, params, graph, sources = _world()
    chain = _chain()
    eta_rest = linear_predictor(params, graph, sources, COUNT)
    mean, variance = chain.moments(sources.ancestors)
    weights = pg_mean(np.sqrt((eta_rest + mean) ** 2 + variance))
    corrected = upward_message(
        X1, eta_rest, chain.effect, weights, ANCESTORS[1], DAYS, risk_set(X1, Y1)
    )
    # The message alone is already almost exact; the chain prior closes the rest,
    # which `test_the_latent_path_is_recovered` takes to 1.0.
    assert _accuracy(corrected, states) > 0.95


def test_a_uniform_chain_starts_undecided_and_carries_its_spread() -> None:
    chain = _chain()
    assert np.allclose(chain.gamma, 0.5)
    mean, variance = chain.moments(ANCESTORS)
    assert np.allclose(mean, 0.8)
    assert np.allclose(variance, 0.64)


def test_a_decided_chain_carries_no_spread() -> None:
    chain = _chain()
    chain.gamma = np.tile(np.array([1.0, 0.0]), (DAYS, 1))
    mean, variance = chain.moments(ANCESTORS)
    assert np.allclose(mean, 0.0) and np.allclose(variance, 0.0)


# ---------------------------------------------------------------------------
# The transition step
# ---------------------------------------------------------------------------


def test_the_transitions_come_back_from_the_posterior() -> None:
    states, X1, Y1, params, graph, sources = _world(persistence=0.9)
    _, chains, _ = FineSEM(max_sweeps=10).fit(
        X1, Y1, params, graph, sources, [_chain()], COUNTS
    )
    stay = np.exp(np.diag(chains[0].log_trans))
    assert np.all(stay > 0.75), stay
    assert np.allclose(np.exp(chains[0].log_trans).sum(axis=0), 1.0, atol=1e-6)


def test_a_chain_with_no_pair_marginal_is_left_alone() -> None:
    chain = _chain()
    before = chain.log_trans.copy()
    assert np.allclose(update_transitions(chain).log_trans, before)


def test_the_latent_step_returns_a_normalized_posterior() -> None:
    _, X1, Y1, params, graph, sources = _world()
    chain = _chain()
    eta_rest = linear_predictor(params, graph, sources, COUNT)
    gamma, xi, log_z = latent_step(
        chain,
        X1,
        eta_rest,
        pg_mean(np.abs(eta_rest)),
        sources.ancestors,
        DAYS,
        risk_set(X1, Y1),
    )
    assert gamma.shape == (DAYS, 2) and xi.shape == (DAYS - 1, 2, 2)
    assert np.allclose(gamma.sum(axis=1), 1.0)
    assert np.allclose(xi.sum(axis=(1, 2)), 1.0)
    assert np.isfinite(log_z)


def test_the_history_reports_what_it_did() -> None:
    _, X1, Y1, params, graph, sources = _world()
    _, _, history = FineSEM(max_sweeps=2).fit(
        X1, Y1, params, graph, sources, [_chain()], COUNTS
    )
    assert isinstance(history, FitHistory)
    assert history.steps[0] == "start"
    assert all(step.startswith("latent k=") for step in history.steps[1:])


def test_a_single_epoch_chain_has_no_pairs() -> None:
    """A chain of one epoch has no transition, so its pair marginal is empty."""
    chain = RegimeChain.uniform(scale=2, n_states=2, n_labels=1, n_epochs=1)
    assert chain.xi.shape == (0, 2, 2)
    assert update_transitions(chain) is chain


def test_an_explicit_schedule_is_taken_as_given() -> None:
    """The order is an argument, so a caller may name it outright."""
    _, X1, Y1, params, graph, sources = _world()
    chains = [_chain(), _chain()]
    chains[1].effect = np.zeros((2, 1))
    _, _, history = FineSEM(max_sweeps=2, schedule=[1, 0]).fit(
        X1, Y1, params, graph, sources, chains, COUNTS
    )
    assert history.monotone, history.values
    assert history.steps[1:3] == ["latent k=2", "latent k=2"]
