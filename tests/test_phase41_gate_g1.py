"""
tests/test_phase41_gate_g1.py
=============================
Gate G1: the calendar nodes are recovered before anything is rewritten.

The nodes are installed at the coarsest scale, where the emission is absolute
and a block-constant feature is not annihilated by an allocation softmax, and a
declared week trend and a declared weekend effect are read back from a fit.  A
failure here is a fault in the node construction rather than in the formulation
that replaces the emission later, which is why the check runs at this point.

The gate tests for **bias** and reports variance separately.  One subject over
twelve weeks holds 24 weekend days, so a single fit cannot resolve a weekend
effect at all, and a gate built on one fit would measure the horizon instead of
the machinery.  Averaging the recovered coefficient over independent replicates
separates the two: the mean answers whether the nodes are built correctly, and
the spread is the power statement.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta

import numpy as np
import pytest

from src.core.calendar import SlotCalendar
from src.core.features import calendar_blocks, horizon_trend, weekend
from src.core.graph import MultiScaleDBNGraph
from src.learning.estep import EStep
from src.learning.mstep import MStep
from src.synthetic.families import make_generator
from src.synthetic.scenarios import ScenarioConfig

WEEKS = 12
DAYS = WEEKS * 7
MONDAY = datetime(2026, 1, 5)
SEEDS = tuple(range(1, 17))

# What the data are built to carry, in log-mean units on the coarsest count.
# One label rises over the horizon and one falls, and both are higher at a
# weekend, so a recovered sign is a claim and not a coin toss.
TRUE_TREND = np.array([0.06, -0.05])
TRUE_WEEKEND = np.array([0.40, 0.25])
TRUE_CTX = np.stack([TRUE_TREND, TRUE_WEEKEND])


def _blocks(cal: SlotCalendar):
    counts = [DAYS * 24, DAYS]
    nodes = [horizon_trend(cal.origin, "week"), weekend()]
    return calendar_blocks(cal, nodes, counts), counts


def _scenario(seed: int) -> ScenarioConfig:
    return ScenarioConfig(
        name="g1",
        K=2,
        H_finest=DAYS * 24,
        branching=[24],
        slot_minutes=60,
        origin=MONDAY.isoformat(),
        L=2,
        M_k=[1, 1],
        r_k=[0, 0],
        max_lag=[1, 1],
        max_indegree=[0, 0],
        context_dim=[0, 2],
        family=1,
        seed=seed,
        finest_rate=0.25,
        # The cap is applied to the coarsest count before any allocation, so it
        # truncates exactly the tail a weekend effect lives in.  This gate is
        # about the nodes, so it is measured where that artifact is absent.
        finest_binary=False,
    )


def _generate(seed: int):
    """One dataset carrying the declared calendar effects at the coarsest scale."""
    cal = SlotCalendar(origin=MONDAY, slot=timedelta(hours=1), branching=(24,))
    blocks, _ = _blocks(cal)
    gen = make_generator(_scenario(seed), seed=seed)
    gen.params_list[-1].coarse_count.betaCtx[:] = TRUE_CTX
    # Only the count channel carries the declared effect, so a recovered
    # coefficient cannot be an echo of the duration or zero-inflation blocks.
    gen.params_list[-1].coarse_dur.alphaCtx[:] = 0.0
    gen.params_list[-1].coarse_count.gammaCtx[:] = 0.0
    ds = gen.generate(seed=seed, context_fn=lambda k, T, rng: blocks.blocks[k - 1])
    return gen, ds


def _recover(seed: int) -> np.ndarray:
    """The coarsest-scale context coefficients a fit reads back."""
    gen, ds = _generate(seed)
    graph = MultiScaleDBNGraph(hierarchy=ds.hierarchy, label_names=gen.label_names)
    params = [deepcopy(p) for p in ds.params]
    # The declared effect is what the fit has to find, so it does not start there.
    params[-1].coarse_count.betaCtx[:] = 0.0
    estep, mstep = EStep(), MStep()
    for _ in range(15):
        q = estep.run(2, ds.data, params[-1], graph, None)
        params[-1] = mstep.run(2, q, params[-1], ds.data, ds.hierarchy)
    return params[-1].coarse_count.betaCtx


@pytest.fixture(scope="module")
def replicates() -> np.ndarray:
    """Recovered coefficients over independent replicates, shape (n, 2, L)."""
    return np.array([_recover(seed) for seed in SEEDS])


# ---------------------------------------------------------------------------
# The gate: are the nodes built correctly
# ---------------------------------------------------------------------------


def test_the_nodes_land_on_the_coarsest_scale() -> None:
    """Both are constant within a day, so the whole block sits at scale 2."""
    cal = SlotCalendar(origin=MONDAY, slot=timedelta(hours=1), branching=(24,))
    blocks, _ = _blocks(cal)
    assert blocks.homes == {"week_index": 2, "weekend": 2}
    assert blocks.widths == [0, 2]
    assert blocks.names[2] == ["week_index", "weekend=1"]


def test_the_week_trend_is_recovered_without_bias(replicates) -> None:
    mean = replicates[:, 0].mean(axis=0)
    assert np.sign(mean).tolist() == np.sign(TRUE_TREND).tolist()
    assert np.allclose(mean, TRUE_TREND, atol=0.02), f"{mean} against {TRUE_TREND}"


def test_the_weekend_effect_is_recovered_without_bias(replicates) -> None:
    mean = replicates[:, 1].mean(axis=0)
    assert np.all(mean > 0.0)
    assert np.allclose(mean, TRUE_WEEKEND, atol=0.10), f"{mean} against {TRUE_WEEKEND}"


def test_the_two_effects_are_not_read_for_each_other(replicates) -> None:
    """A weekend effect booked as a trend, or the reverse, would pass a looser test."""
    trend, week_end = replicates[:, 0].mean(axis=0), replicates[:, 1].mean(axis=0)
    assert abs(trend[0]) < 0.5 * week_end[0]
    assert week_end.min() > 4.0 * abs(trend).max()


def test_the_generator_placed_the_effects_it_was_given() -> None:
    """The recovery means nothing if the data do not carry the effects."""
    _, ds = _generate(SEEDS[0])
    X = ds.data.get_X(2).astype(float)
    is_weekend = ds.data.get_C(2)[:, 1] == 1.0
    assert X[is_weekend].mean(axis=0).min() > X[~is_weekend].mean(axis=0).min()
    assert ds.data.get_C(2)[:, 0].max() == WEEKS - 1


# ---------------------------------------------------------------------------
# What the gate measures beside itself
# ---------------------------------------------------------------------------


def test_one_subject_over_twelve_weeks_cannot_resolve_the_weekend_effect(
    replicates,
) -> None:
    """
    The spread across replicates is the size of the effect being estimated.

    Twelve weeks hold 24 weekend days, and at this event rate and dispersion the
    coefficient's own spread is comparable to the coefficient, so a single fit
    reports a number that is mostly its own noise.  Pooling subjects is what
    makes such a claim, and this is the measurement behind that.
    """
    spread = replicates[:, 1].std(axis=0)
    assert spread.min() > 0.12
    assert spread.max() > 0.5 * TRUE_WEEKEND.min()
