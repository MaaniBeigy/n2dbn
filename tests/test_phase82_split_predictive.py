"""
Tests for the split predictive check that replaces the plug-in one.

`posterior_predictive` compares a record to replicates drawn from a fit that read
that same record, so the replicates are pulled toward the observation.  On a log
whose rate rises halfway through, a constant-rate model reads as calibrated under
the plug-in check and is named under the split check, and that gap is what these
tests pin.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from src.core.calendar import SlotCalendar
from src.core.durations import aggregate
from src.core.typed_graph import COUNT, DURATION, TypedGraph
from src.evaluation.fidelity import (
    MIN_EPISODES,
    CellGrid,
    chronological_split,
    posterior_predictive,
    split_predictive,
)
from src.model.intensity import FineParams, ancestor_maps, empty_sources, sample

LABELS = ("walk", "rest")
N_DAYS = 28
T_SLOTS = N_DAYS * 24
PARENT_MAPS = [
    np.repeat(np.arange(N_DAYS), 24),
    np.repeat(np.arange(N_DAYS // 7), 7),
]
ORIGIN = datetime(2026, 2, 16)
BLOCKS = [(0, N_DAYS // 2), (N_DAYS // 2, N_DAYS)]
DAY_SCALE = 2


def calendar() -> SlotCalendar:
    return SlotCalendar(origin=ORIGIN, slot=timedelta(minutes=60), branching=(24, 7))


def sources_of(X1, Y1):
    anc = ancestor_maps(PARENT_MAPS, T_SLOTS)
    blocks = [
        np.zeros((T_SLOTS, 0)),
        np.zeros((N_DAYS, 0)),
        np.zeros((N_DAYS // 7, 0)),
    ]
    sources = empty_sources(anc, blocks, len(LABELS))
    sources.counts = aggregate(X1, PARENT_MAPS)
    sources.durations = aggregate(Y1, PARENT_MAPS)
    return sources


def params_at(rate: float, mean_length: float = 2.0) -> FineParams:
    params = FineParams(n_labels=len(LABELS))
    params.base[COUNT][:] = rate
    params.base[DURATION][:] = np.log(mean_length)
    params.mark_dispersion = np.full(len(LABELS), 4.0)
    return params


LOW_RATE = -3.0
HIGH_RATE = -0.5


def drawn_at(rate: float, seed: int):
    """One log drawn from the model itself, so the families agree by construction."""
    X1, Y1 = sample(
        params_at(rate),
        TypedGraph(label_names=list(LABELS)),
        sources_of(
            np.zeros((T_SLOTS, len(LABELS)), dtype=np.int64),
            np.zeros((T_SLOTS, len(LABELS)), dtype=np.int64),
        ),
        np.random.default_rng(seed),
    )
    return X1, Y1


def rising_log(seed: int = 0):
    """A record whose rate rises in its last quarter and is drawn at both rates."""
    low_X, low_Y = drawn_at(LOW_RATE, seed)
    high_X, high_Y = drawn_at(HIGH_RATE, seed + 1)
    cut = 3 * T_SLOTS // 4
    X1 = np.concatenate([low_X[:cut], high_X[cut:]])
    Y1 = np.concatenate([low_Y[:cut], high_Y[cut:]])
    return X1, Y1


def fit_rate(X1, train) -> float:
    """The log odds of a start, read off the slots the fit is given."""
    starts = float(X1[train].sum())
    slots = float(train.sum() * X1.shape[1])
    share = min(max(starts / slots, 1.0 / slots), 1.0 - 1.0 / slots)
    return float(np.log(share / (1.0 - share)))


def refit_from(X1):
    def refit(train):
        return (
            params_at(fit_rate(X1, train)),
            TypedGraph(label_names=list(LABELS)),
            None,
        )

    return refit


def outside_share(report) -> float:
    """The share of cells with enough episodes behind them that sit off the spread."""
    backed = [c for c in report.cells if c.support >= MIN_EPISODES]
    return sum(c.outside for c in backed) / len(backed) if backed else 0.0


# ---------------------------------------------------------------------------
# The split itself
# ---------------------------------------------------------------------------


def test_the_split_holds_out_the_last_share_of_every_block():
    anc = ancestor_maps(PARENT_MAPS, T_SLOTS)
    held = chronological_split(anc, DAY_SCALE, BLOCKS, fraction=0.5)
    assert held.sum() == T_SLOTS // 2
    # The held-out span of each block is its own last half, so no block is
    # entirely training and none entirely held out.
    for start, stop in BLOCKS:
        own = held[start * 24 : stop * 24]
        assert own[: own.size // 2].sum() == 0
        assert own[own.size // 2 :].all()


def test_a_split_is_taken_at_whole_epochs_of_the_scale_it_names():
    anc = ancestor_maps(PARENT_MAPS, T_SLOTS)
    held = chronological_split(anc, DAY_SCALE, BLOCKS, fraction=0.25)
    per_day = held.reshape(N_DAYS, 24)
    assert set(per_day.sum(axis=1)) <= {0, 24}


def test_a_split_that_empties_one_side_is_refused():
    anc = ancestor_maps(PARENT_MAPS, T_SLOTS)
    with pytest.raises(ValueError, match="strictly inside"):
        chronological_split(anc, DAY_SCALE, BLOCKS, fraction=0.0)
    with pytest.raises(ValueError, match="strictly inside"):
        chronological_split(anc, DAY_SCALE, BLOCKS, fraction=1.0)
    with pytest.raises(ValueError, match="leaves one side empty"):
        chronological_split(anc, DAY_SCALE, [(0, 2)], fraction=0.1)


# ---------------------------------------------------------------------------
# What the split check sees that the plug-in check does not
# ---------------------------------------------------------------------------


def test_the_plug_in_check_reads_a_rising_record_as_calibrated():
    X1, Y1 = rising_log()
    sources = sources_of(X1, Y1)
    everything = np.ones(T_SLOTS, dtype=bool)
    grid = CellGrid.build(calendar(), sources, LABELS, observed=everything)
    report = posterior_predictive(
        X1,
        Y1,
        params_at(fit_rate(X1, everything)),
        TypedGraph(label_names=list(LABELS)),
        sources,
        grid,
        np.random.default_rng(0),
        n_replicates=60,
    )
    assert outside_share(report) < 0.2


def test_the_split_check_names_the_record_the_plug_in_check_passes():
    X1, Y1 = rising_log()
    sources = sources_of(X1, Y1)
    anc = ancestor_maps(PARENT_MAPS, T_SLOTS)
    held = chronological_split(anc, DAY_SCALE, BLOCKS, fraction=0.25)
    report = split_predictive(
        X1,
        Y1,
        refit_from(X1),
        calendar(),
        LABELS,
        sources,
        observed=np.ones(T_SLOTS, dtype=bool),
        held=held,
        rng=np.random.default_rng(0),
        n_replicates=60,
    )
    assert outside_share(report) > 0.5


def test_the_split_check_passes_a_record_whose_rate_never_moves():
    X1, Y1 = drawn_at(LOW_RATE, 11)
    sources = sources_of(X1, Y1)
    anc = ancestor_maps(PARENT_MAPS, T_SLOTS)
    held = chronological_split(anc, DAY_SCALE, BLOCKS, fraction=0.25)
    report = split_predictive(
        X1,
        Y1,
        refit_from(X1),
        calendar(),
        LABELS,
        sources,
        observed=np.ones(T_SLOTS, dtype=bool),
        held=held,
        rng=np.random.default_rng(2),
        n_replicates=60,
    )
    assert outside_share(report) < 0.2


def test_the_fit_never_reads_a_held_out_slot():
    X1, Y1 = rising_log()
    sources = sources_of(X1, Y1)
    anc = ancestor_maps(PARENT_MAPS, T_SLOTS)
    held = chronological_split(anc, DAY_SCALE, BLOCKS, fraction=0.25)
    seen = {}

    def refit(train):
        seen["train"] = train.copy()
        return (
            params_at(fit_rate(X1, train)),
            TypedGraph(label_names=list(LABELS)),
            None,
        )

    split_predictive(
        X1,
        Y1,
        refit,
        calendar(),
        LABELS,
        sources,
        observed=np.ones(T_SLOTS, dtype=bool),
        held=held,
        rng=np.random.default_rng(0),
        n_replicates=4,
    )
    assert not (seen["train"] & held).any()
    assert (seen["train"] | held).all()


def test_a_split_that_leaves_no_training_or_no_held_out_slot_is_refused():
    X1, Y1 = rising_log()
    sources = sources_of(X1, Y1)
    everything = np.ones(T_SLOTS, dtype=bool)
    with pytest.raises(ValueError, match="no training slot"):
        split_predictive(
            X1,
            Y1,
            refit_from(X1),
            calendar(),
            LABELS,
            sources,
            observed=everything,
            held=everything,
            rng=np.random.default_rng(0),
            n_replicates=4,
        )
    with pytest.raises(ValueError, match="no observed slot to score"):
        split_predictive(
            X1,
            Y1,
            refit_from(X1),
            calendar(),
            LABELS,
            sources,
            observed=everything,
            held=np.zeros(T_SLOTS, dtype=bool),
            rng=np.random.default_rng(0),
            n_replicates=4,
        )
