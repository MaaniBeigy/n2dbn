"""
Tests for the two summaries of the lag axis and the carrier they give a regime.

Both read the strict past, which is what lets an edge carrying one be read at
the epoch it predicts, and neither is constant across the children of a parent
block, which is what keeps a coefficient on it identified at its own scale.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.projection import BlockProjector
from src.model.recency import (
    geometric_lag,
    recency,
    recency_column,
    regime_interaction,
)


def test_recency_counts_from_the_last_event_and_resets_at_one():
    events = np.array([0, 1, 0, 0, 1, 0])
    assert recency(events).tolist() == [1.0, 2.0, 1.0, 2.0, 3.0, 1.0]


def test_recency_before_the_first_event_counts_to_the_edge_of_the_record():
    assert recency(np.zeros(4)).tolist() == [1.0, 2.0, 3.0, 4.0]


def test_recency_reads_the_strict_past_alone():
    # Two series that agree before slot 3 give the same value at slot 3, so the
    # column at an epoch never reads that epoch's own event.
    left = np.array([0, 1, 0, 1, 0])
    right = np.array([0, 1, 0, 0, 0])
    assert recency(left)[3] == recency(right)[3]


def test_the_recency_column_is_the_log_of_the_count():
    events = np.array([0, 1, 0, 0])
    assert np.allclose(recency_column(events), np.log1p(recency(events)))


def test_the_geometric_lag_matches_the_sum_it_is_a_recursion_for():
    rng = np.random.default_rng(0)
    series = rng.poisson(2.0, size=9).astype(float)
    rho = 0.7
    fast = geometric_lag(series, rho)
    slow = np.zeros_like(series)
    for t in range(series.size):
        slow[t] = (1.0 - rho) * sum(
            rho ** (d - 1) * series[t - d] for d in range(1, t + 1)
        )
    assert np.allclose(fast, slow)


def test_a_geometric_lag_of_zero_decay_reads_the_previous_epoch():
    series = np.array([3.0, 5.0, 7.0])
    assert geometric_lag(series, 0.0).tolist() == [0.0, 3.0, 5.0]


def test_both_summaries_refuse_an_input_they_cannot_read():
    with pytest.raises(ValueError, match="one series"):
        recency(np.zeros((3, 2)))
    with pytest.raises(ValueError, match="one series"):
        geometric_lag(np.zeros((3, 2)), 0.5)
    with pytest.raises(ValueError, match=r"in \[0, 1\)"):
        geometric_lag(np.zeros(3), 1.0)
    with pytest.raises(ValueError, match=r"in \[0, 1\)"):
        geometric_lag(np.zeros(3), -0.1)


@pytest.mark.parametrize("block_size", [2, 3, 4, 6])
def test_neither_summary_is_block_constant_at_an_interior_scale(block_size):
    rng = np.random.default_rng(block_size)
    events = (rng.random(24) < 0.25).astype(float)
    projector = BlockProjector.from_parent_map(
        np.arange(24) // block_size, 24 // block_size
    )
    assert not projector.is_block_constant(recency_column(events))
    assert not projector.is_block_constant(geometric_lag(events, 0.5))


def test_a_regime_rides_on_the_carrier_and_keeps_its_within_block_contrast():
    rng = np.random.default_rng(1)
    events = (rng.random(12) < 0.3).astype(float)
    carrier = recency_column(events)
    gamma = np.zeros((12, 2))
    gamma[:6, 0] = 1.0
    gamma[6:, 1] = 1.0

    columns = regime_interaction(carrier, gamma)
    assert columns.shape == (12, 2)
    assert np.allclose(columns.sum(axis=1), carrier)

    projector = BlockProjector.from_parent_map(np.arange(12) // 3, 4)
    # A regime that does not switch inside a block still varies there, because
    # the carrier does; an additive level would not.
    assert not projector.is_block_constant(columns[:, 0])


def test_an_additive_regime_level_averages_away_inside_a_block():
    gamma = np.zeros((12, 2))
    gamma[:6, 0] = 1.0
    gamma[6:, 1] = 1.0
    projector = BlockProjector.from_parent_map(np.arange(12) // 3, 4)
    # The regime path holds one state across every block, so the level it would
    # add is block constant and contributes nothing at this scale.
    assert projector.is_block_constant(gamma[:, 0])


def test_a_carrier_and_a_posterior_of_different_lengths_are_refused():
    with pytest.raises(ValueError, match="different number"):
        regime_interaction(np.zeros(5), np.zeros((4, 2)))
    with pytest.raises(ValueError, match="different number"):
        regime_interaction(np.zeros(5), np.zeros(5))
