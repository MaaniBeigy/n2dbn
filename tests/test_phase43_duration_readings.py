"""
tests/test_phase43_duration_readings.py
=======================================
Two readings of a duration, both exact, and not the same array.

Initiation charges an episode's whole length to the epoch holding its start;
occupancy spreads it over the epochs it covers.  Both sum up the ladder, so
neither is more reconciled than the other, and the difference is large enough
that a reported number has to say which one it is.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.durations import (
    INITIATION,
    OCCUPANCY,
    aggregate,
    duration_ladder,
    epoch_capacity,
    occupancy_finest,
)

# Ten finest slots per epoch, four epochs.
PARENT_MAPS = [np.repeat(np.arange(4), 10)]


def _one_episode(t: int, length: int, n_slots: int = 40, ell: int = 0):
    X = np.zeros((n_slots, 1), dtype=np.int64)
    Y = np.zeros((n_slots, 1), dtype=np.int64)
    X[t, ell] = 1
    Y[t, ell] = length
    return X, Y


# ---------------------------------------------------------------------------
# The two readings differ, and the difference is the point
# ---------------------------------------------------------------------------


def test_an_episode_spanning_epochs_books_differently_under_each_reading() -> None:
    """A 25-slot episode starting at slot 8 crosses three epochs."""
    X, Y = _one_episode(t=8, length=25)
    initiation = duration_ladder(X, Y, PARENT_MAPS, INITIATION)[1].ravel()
    occupancy = duration_ladder(X, Y, PARENT_MAPS, OCCUPANCY)[1].ravel()
    assert initiation.tolist() == [25, 0, 0, 0]
    assert occupancy.tolist() == [2, 10, 10, 3]
    assert initiation.sum() == occupancy.sum() == 25


def test_initiation_can_exceed_the_epoch_it_is_booked_to() -> None:
    """A hundred-slot episode in a ten-slot epoch, which occupancy cannot do."""
    X, Y = _one_episode(t=0, length=100, n_slots=40)
    initiation = duration_ladder(X, Y, PARENT_MAPS, INITIATION)[1].ravel()
    occupancy = duration_ladder(X, Y, PARENT_MAPS, OCCUPANCY)[1].ravel()
    capacity = epoch_capacity(PARENT_MAPS, 2)
    assert initiation[0] == 100 > capacity[0]
    assert np.all(occupancy <= capacity)


def test_occupancy_stays_within_capacity_on_a_log_with_no_overlap() -> None:
    """
    The bound holds exactly when no episode of a label overlaps another.

    That is the property a valid event log has, and it is what the risk set of
    the fine likelihood enforces: a label already running cannot start.  With
    overlap the slot count exceeds one and the bound goes, which is the honest
    statement rather than a bound that quietly assumes its own precondition.
    """
    rng = np.random.default_rng(3)
    X = np.zeros((40, 2), dtype=np.int64)
    Y = np.zeros((40, 2), dtype=np.int64)
    for ell in range(2):
        t = int(rng.integers(0, 4))
        while t < 40:
            length = int(rng.integers(1, 9))
            X[t, ell] = 1
            Y[t, ell] = length
            t += length + int(rng.integers(0, 4))
    fine = occupancy_finest(X, Y)
    assert fine.max() == 1, "the constructed log holds no same-label overlap"
    occupancy = duration_ladder(X, Y, PARENT_MAPS, OCCUPANCY)[1]
    assert np.all(occupancy <= epoch_capacity(PARENT_MAPS, 2)[:, None])


def test_overlapping_episodes_of_one_label_break_the_capacity_bound() -> None:
    """Recorded so the bound is never quoted without its precondition."""
    X = np.zeros((40, 1), dtype=np.int64)
    Y = np.zeros((40, 1), dtype=np.int64)
    X[0, 0], Y[0, 0] = 1, 8
    X[2, 0], Y[2, 0] = 1, 8
    assert occupancy_finest(X, Y).max() == 2
    occupancy = duration_ladder(X, Y, PARENT_MAPS, OCCUPANCY)[1]
    assert occupancy[0, 0] > epoch_capacity(PARENT_MAPS, 2)[0]


def test_the_two_readings_disagree_on_most_blocks() -> None:
    """Measured, because a caller choosing at random would usually be wrong."""
    rng = np.random.default_rng(11)
    X = (rng.random((200, 2)) < 0.1).astype(np.int64)
    Y = X * rng.integers(1, 40, size=(200, 2))
    maps = [np.repeat(np.arange(20), 10)]
    initiation = duration_ladder(X, Y, maps, INITIATION)[1]
    occupancy = duration_ladder(X, Y, maps, OCCUPANCY)[1]
    differing = int((initiation != occupancy).any(axis=1).sum())
    assert differing >= 15, f"only {differing} of 20 blocks differ"
    assert np.abs(initiation - occupancy).max() > 20


# ---------------------------------------------------------------------------
# Both reconcile, which is why neither is a correction of the other
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reading", [INITIATION, OCCUPANCY])
def test_a_reading_sums_exactly_up_the_ladder(reading) -> None:
    rng = np.random.default_rng(5)
    X = (rng.random((360, 3)) < 0.12).astype(np.int64)
    Y = X * rng.integers(1, 25, size=(360, 3))
    maps = [np.repeat(np.arange(60), 6), np.repeat(np.arange(12), 5)]
    ladder = duration_ladder(X, Y, maps, reading)
    assert [a.shape[0] for a in ladder] == [360, 60, 12]
    for k, pm in enumerate(maps):
        rolled = np.zeros_like(ladder[k + 1])
        np.add.at(rolled, pm, ladder[k])
        assert np.array_equal(rolled, ladder[k + 1])
    assert ladder[0].sum() == ladder[-1].sum()


def test_the_totals_agree_when_nothing_runs_past_the_horizon() -> None:
    X, Y = _one_episode(t=8, length=25)
    a = duration_ladder(X, Y, PARENT_MAPS, INITIATION)[-1]
    b = duration_ladder(X, Y, PARENT_MAPS, OCCUPANCY)[-1]
    assert a.sum() == b.sum() == 25


def test_an_overrun_is_the_only_thing_that_separates_the_totals() -> None:
    """
    Initiation books a length the horizon never saw; occupancy books what fits.

    The gap is exactly the overhang, so the two readings differ in total only at
    the end of the record, and a total reported without its reading is wrong by
    that much.
    """
    X, Y = _one_episode(t=36, length=12, n_slots=40)
    a = duration_ladder(X, Y, PARENT_MAPS, INITIATION)[-1].sum()
    b = duration_ladder(X, Y, PARENT_MAPS, OCCUPANCY)[-1].sum()
    assert a == 12 and b == 4
    assert a - b == 12 - (40 - 36)


# ---------------------------------------------------------------------------
# Edges of the finest reading
# ---------------------------------------------------------------------------


def test_an_episode_running_past_the_horizon_is_clipped() -> None:
    X, Y = _one_episode(t=35, length=50, n_slots=40)
    occupancy = occupancy_finest(X, Y)
    assert occupancy[:, 0].sum() == 5
    assert occupancy[35:, 0].tolist() == [1, 1, 1, 1, 1]


def test_a_zero_length_record_occupies_nothing() -> None:
    X = np.zeros((10, 1), dtype=np.int64)
    Y = np.zeros((10, 1), dtype=np.int64)
    X[2, 0] = 1
    assert occupancy_finest(X, Y).sum() == 0


def test_a_slot_with_no_start_occupies_nothing() -> None:
    X = np.zeros((10, 2), dtype=np.int64)
    Y = np.zeros((10, 2), dtype=np.int64)
    assert occupancy_finest(X, Y).sum() == 0


def test_concurrent_labels_are_counted_separately() -> None:
    X = np.zeros((10, 2), dtype=np.int64)
    Y = np.zeros((10, 2), dtype=np.int64)
    X[0, 0] = X[0, 1] = 1
    Y[0, 0], Y[0, 1] = 4, 7
    occupancy = occupancy_finest(X, Y)
    assert occupancy[:, 0].sum() == 4 and occupancy[:, 1].sum() == 7


def test_mismatched_arrays_are_refused() -> None:
    with pytest.raises(ValueError, match="they must match"):
        occupancy_finest(np.zeros((4, 1)), np.zeros((5, 1)))


def test_an_unknown_reading_is_refused() -> None:
    X, Y = _one_episode(t=0, length=3)
    with pytest.raises(ValueError, match="expected one of"):
        duration_ladder(X, Y, PARENT_MAPS, "average")


def test_a_parent_map_of_the_wrong_length_is_refused() -> None:
    with pytest.raises(ValueError, match="against 40 epochs"):
        aggregate(np.zeros((40, 1)), [np.repeat(np.arange(3), 10)])


def test_counts_use_the_same_aggregation_as_durations() -> None:
    """Summation is the one operator up the ladder, for counts too."""
    rng = np.random.default_rng(2)
    X = (rng.random((40, 2)) < 0.3).astype(np.int64)
    ladder = aggregate(X, PARENT_MAPS)
    assert ladder[1].sum() == X.sum()
    assert ladder[1].shape == (4, 2)
