"""
core/durations.py
=================
The two readings of a duration, and which one a reported number is.

An episode's length can be booked two ways, and they are not the same array.

Under **initiation** the whole length is charged to the epoch containing the
episode's start.  An eight-hour sleep beginning at 22:00 books 480 minutes to
that day and none to the next, so a coarse duration reads "time taken by
episodes that began here" and can exceed the epoch's own length.

Under **occupancy** the length is spread over the epochs the episode actually
covers.  The same sleep books 120 minutes to one day and 360 to the next, so a
coarse duration reads "time spent here" and is bounded by the epoch.

Both sum exactly up the ladder, so both reconcile, and neither is a correction of
the other.  A per-episode claim needs initiation, because that is where an
episode's own length survives as one number.  A claim about how much of Saturday
went to walking needs occupancy.  Occupancy is a deterministic function of the
initiation pair, so it is derived here for reporting and is never given a
likelihood term of its own; a density over it would count the same data twice.

Every reported duration says which reading it is.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

# The two readings.  A reported duration carries one of these names.
INITIATION = "initiation"
OCCUPANCY = "occupancy"
READINGS = (INITIATION, OCCUPANCY)


def occupancy_finest(X1: np.ndarray, Y1: np.ndarray) -> np.ndarray:
    """
    Slots of each finest epoch spent inside an episode of each label.

    An episode recorded at slot t with length Y1[t] covers slots t to t+Y1[t]-1,
    clipped at the end of the horizon.  That reading is exact while at most one
    episode of a label begins per epoch, which is what the storage can express:
    a count above one shares a single total, so the individual lengths are not
    recoverable and the total is laid down from the first start.
    """
    if X1.shape != Y1.shape:
        raise ValueError(f"X1 is {X1.shape} and Y1 is {Y1.shape}; they must match")
    T, L = X1.shape
    out = np.zeros((T, L), dtype=np.int64)
    starts = np.argwhere(X1 > 0)
    for t, ell in starts:
        length = int(Y1[t, ell])
        if length <= 0:
            continue
        out[t : min(t + length, T), ell] += 1
    return out


def aggregate(fine: np.ndarray, parent_maps: Sequence[np.ndarray]) -> List[np.ndarray]:
    """
    A finest-scale array summed up the ladder, finest first.

    Summation is the only aggregation up the time ladder, so this is the whole
    cross-scale mechanism for counts and for either duration reading.
    """
    out = [np.asarray(fine)]
    for pm in parent_maps:
        pm = np.asarray(pm)
        if pm.size != out[-1].shape[0]:
            raise ValueError(
                f"parent map of length {pm.size} against {out[-1].shape[0]} epochs"
            )
        parent = np.zeros((int(pm.max()) + 1, out[-1].shape[1]), dtype=out[-1].dtype)
        np.add.at(parent, pm, out[-1])
        out.append(parent)
    return out


def duration_ladder(
    X1: np.ndarray,
    Y1: np.ndarray,
    parent_maps: Sequence[np.ndarray],
    reading: str = INITIATION,
) -> List[np.ndarray]:
    """Durations at every scale under the named reading, finest first."""
    if reading not in READINGS:
        raise ValueError(f"reading {reading!r}; expected one of {list(READINGS)}")
    fine = np.asarray(Y1) if reading == INITIATION else occupancy_finest(X1, Y1)
    return aggregate(fine, parent_maps)


def epoch_capacity(parent_maps: Sequence[np.ndarray], k: int) -> np.ndarray:
    """Finest slots inside each epoch of scale k, which occupancy cannot exceed."""
    counts = np.ones(len(np.asarray(parent_maps[0])), dtype=np.int64)[:, None]
    ladder = aggregate(counts, parent_maps)
    return ladder[k - 1].ravel()
