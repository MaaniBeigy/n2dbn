"""
tests/test_phase49_gate_g2.py
=============================
Gate G2: the fine process reproduces coarse variance without a coarse parameter.

The coarsest scale no longer carries a free dispersion, so aggregate variance and
autocorrelation have to come out of the finest-scale mechanism or the formulation
is wrong.  This is the named failure mode of the rewrite, and it is placed here
because the tempting repair, a dispersion parameter on the coarse count, would
reintroduce a second likelihood and undo the argument silently.

What is measured is **range**, against two targets read from real generated data
rather than chosen.  The low end is CalendarBench, whose daily counts are
under-dispersed and often deterministic because its allocator fixes a day's
episode count before placing anything: over 230 label-and-person series the Fano
factor runs 0.000 to 1.714 with a median of 0.000.  The high end is the frozen
ZINB the project modeled with until now, at 4.6 to 8.8.

A plain Bernoulli process cannot reach the high end, which is the honest half of
this result.  A coarse random effect can, and it is the device the plan named:
the autoregressive factor on the coarse grid, entering the one predictor.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.durations import aggregate
from src.core.typed_graph import COUNT, DURATION, TypedGraph
from src.model.intensity import FineParams, ancestor_maps, empty_sources, sample

DAYS = 84
T = DAYS * 1440
PARENT_MAPS = [np.repeat(np.arange(DAYS * 24), 60), np.repeat(np.arange(DAYS), 24)]
ANCESTORS = ancestor_maps(PARENT_MAPS, T)

# Read off the two sources rather than chosen: CalendarBench's daily counts over
# 230 label-and-person series, and the frozen ZINB at three labels.
CALENDAR_BENCH_FANO = (0.000, 1.714)
FROZEN_ZINB_FANO = (4.58, 8.79)


def _daily_counts(offset: np.ndarray, base: float, seed: int, mean_len: float = 5.0):
    rng = np.random.default_rng(seed)
    sources = empty_sources(
        ANCESTORS,
        [np.zeros((T, 0)), np.zeros((DAYS * 24, 0)), np.zeros((DAYS, 0))],
        1,
    )
    params = FineParams(n_labels=1)
    params.base[COUNT][:] = base
    params.base[DURATION][:] = np.log(mean_len)
    params.mark_dispersion = np.array([5.0])
    X, _ = sample(
        params,
        TypedGraph(label_names=["a"]),
        sources,
        rng,
        offsets={COUNT: offset},
    )
    return aggregate(X, PARENT_MAPS)[2][:, 0].astype(float)


def _fano(counts: np.ndarray) -> float:
    return float(counts.var() / max(counts.mean(), 1e-9))


# ---------------------------------------------------------------------------
# The honest half: a plain process is under-dispersed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("base", [-6.0, -5.0, -4.0])
def test_a_homogeneous_fine_process_is_under_dispersed(base) -> None:
    """
    A sum of Bernoulli draws has less variance than its mean, necessarily.

    So the finest scale on its own cannot reach the overdispersion the project
    has always modeled, and saying otherwise would be the failure this gate is
    for.
    """
    counts = _daily_counts(np.zeros((T, 1)), base, seed=0)
    assert counts.mean() > 1.0
    assert _fano(counts) < 1.0


def test_the_plain_process_covers_the_lower_target() -> None:
    """CalendarBench's own range is under-dispersed, and that end needs nothing."""
    counts = _daily_counts(np.zeros((T, 1)), -6.0, seed=1)
    assert CALENDAR_BENCH_FANO[0] <= _fano(counts) <= CALENDAR_BENCH_FANO[1]


def test_a_concentrated_hour_pushes_toward_a_deterministic_count() -> None:
    """
    A pinned routine is a label whose count barely varies, which the low target
    is full of.  Concentrating the intensity into one hour and letting the risk
    set block a second start reaches it from the fine mechanism alone.
    """
    hour_of_day = ANCESTORS[1] % 24
    fanos = []
    for peak in (2.0, 4.0, 6.0):
        offset = np.where(hour_of_day == 8, peak, -peak)[:, None].astype(float)
        fanos.append(_fano(_daily_counts(offset, -4.0, seed=2, mean_len=50.0)))
    assert fanos[-1] < 0.4
    assert all(b < a for a, b in zip(fanos, fanos[1:])), fanos


# ---------------------------------------------------------------------------
# The gate: a coarse random effect reaches the upper target
# ---------------------------------------------------------------------------


def _with_day_effect(sd: float, seed: int = 3) -> float:
    rng = np.random.default_rng(seed)
    offset = rng.normal(0.0, sd, size=(DAYS, 1))[ANCESTORS[2]]
    return _fano(_daily_counts(offset, -5.0, seed=seed))


def test_a_coarse_random_effect_reaches_the_frozen_zinb_range() -> None:
    """
    The gate itself.  The device is the coarse factor, not a coarse dispersion.

    A day-scale random effect of moderate size takes the aggregate Fano factor
    through the range the frozen ZINB occupied, so the variance the old coarse
    parameter supplied is available from the fine mechanism.
    """
    reached = [_with_day_effect(sd) for sd in (0.3, 0.6, 1.0)]
    assert reached[0] > 1.0, "a small coarse effect already crosses into overdispersion"
    assert max(reached) > FROZEN_ZINB_FANO[1], f"{reached} against {FROZEN_ZINB_FANO}"


def test_the_dispersion_rises_with_the_coarse_effect() -> None:
    """A knob, not an accident: more coarse variation gives more coarse variance."""
    reached = [_with_day_effect(sd) for sd in (0.3, 0.6, 1.0, 1.5)]
    assert all(b > a for a, b in zip(reached, reached[1:])), reached


def test_the_range_spans_both_targets() -> None:
    """
    One mechanism covers a Fano factor from a quarter to past sixty.

    Both ends of what the two sources show sit inside that, so the coarse scale
    losing its own dispersion parameter costs no reachable behavior.
    """
    hour_of_day = ANCESTORS[1] % 24
    lowest = _fano(
        _daily_counts(
            np.where(hour_of_day == 8, 6.0, -6.0)[:, None].astype(float),
            -4.0,
            seed=4,
            mean_len=50.0,
        )
    )
    highest = _with_day_effect(2.0, seed=5)
    assert lowest < CALENDAR_BENCH_FANO[0] + 0.4
    assert highest > FROZEN_ZINB_FANO[1]
    assert highest / max(lowest, 1e-9) > 100


# ---------------------------------------------------------------------------
# Autocorrelation, which the gate names beside variance
# ---------------------------------------------------------------------------


def _autoregressive_days(phi: float, sd: float = 0.8, seed: int = 6):
    rng = np.random.default_rng(seed)
    effect = np.zeros(DAYS)
    for d in range(1, DAYS):
        effect[d] = phi * effect[d - 1] + rng.normal(0.0, sd * np.sqrt(1.0 - phi**2))
    counts = _daily_counts(effect[ANCESTORS[2]][:, None], -6.0, seed=seed)
    return _fano(counts), float(np.corrcoef(counts[:-1], counts[1:])[0, 1])


def test_the_coarse_counts_inherit_the_coarse_correlation() -> None:
    """
    Variance alone is not the whole claim; the correlation has to come through.

    An autoregressive effect on the day grid shows up as autocorrelated daily
    counts, so the coarse process the model implies has the right shape and not
    only the right spread.
    """
    quiet = _autoregressive_days(0.0)
    persistent = _autoregressive_days(0.8)
    assert abs(quiet[1]) < 0.25
    assert persistent[1] > 0.5
    assert persistent[1] > quiet[1] + 0.4


def test_a_process_with_no_coarse_effect_leaves_uncorrelated_days() -> None:
    counts = _daily_counts(np.zeros((T, 1)), -5.0, seed=7)
    assert abs(float(np.corrcoef(counts[:-1], counts[1:])[0, 1])) < 0.3


# ---------------------------------------------------------------------------
# What the gate refuses
# ---------------------------------------------------------------------------


def test_the_coarse_count_has_no_parameter_of_its_own() -> None:
    """
    The repair this gate exists to refuse, asserted as an absence.

    `FineParams` carries a dispersion for the duration mark and nothing for a
    coarse count, because a coarse count is a deterministic sum and giving it a
    density would count the same data twice.
    """
    params = FineParams(n_labels=2)
    names = set(vars(params))
    assert "mark_dispersion" in names
    assert not any("coarse" in name for name in names)
    assert not any("count_dispersion" in name for name in names)
