"""
data/event_log.py
=================
Build MultiScaleData from an observed event log.

An observed log lists episodes as (label, start, end) triples on a wall clock.
The model consumes counts and durations on a nested epoch grid, so the reader
lays the episodes on a finest-scale slot grid of `slot_minutes` and aggregates
upward through the branching factors.

Finest-scale semantics
----------------------
Scale 1 is an event-start process: X^(1)[t, l] counts the episodes of label l
whose start falls in slot t, and Y^(1)[t, l] holds their wall-clock duration
rounded up to whole slots.  Duration is not the count of grid slots an episode
overlaps: a 13-minute episode on a 15-minute grid contributes 1 even when it
straddles a boundary, because Y is the duration the model predicts.  Every
coarser scale is the exact block sum of the scale below it, so RecErr_X and
RecErr_Y are 0 by construction.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..core.calendar import SlotCalendar, require_naive
from ..core.scales import ScaleHierarchy
from ..core.variables import MultiScaleData

# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObservedEvent:
    """One observed episode with wall-clock start and end."""

    label: str
    start: datetime
    end: datetime


# ---------------------------------------------------------------------------


def read_event_log(
    path: str,
    label_col: str = "label",
    start_col: str = "start_iso",
    end_col: str = "end_iso",
) -> List[ObservedEvent]:
    """
    Read an event-log CSV with ISO timestamps into ObservedEvent records.

    Raises
    ------
    ValueError
        If a named column is absent, if a timestamp cell is unreadable or
        missing, or if a row ends before it starts.
    """
    events: List[ObservedEvent] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        for col in (label_col, start_col, end_col):
            if col not in fields:
                raise ValueError(
                    f"column '{col}' is missing from {path}; the file has {fields}"
                )
        # Row 1 is the header, so the first record is row 2 of the file.
        for row_no, row in enumerate(reader, start=2):
            try:
                start = datetime.fromisoformat(row[start_col])
                end = datetime.fromisoformat(row[end_col])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"row {row_no} of {path} carries an unreadable timestamp: {exc}"
                ) from exc
            if end < start:
                raise ValueError(
                    f"row {row_no} of {path} ends at {end.isoformat()} before "
                    f"its start {start.isoformat()}"
                )
            # Mixing offsets, or an offset with a naive origin, mis-dates a log
            # silently rather than raising, so the conversion is the caller's.
            require_naive(start, f"row {row_no} of {path} start")
            require_naive(end, f"row {row_no} of {path} end")
            events.append(ObservedEvent(label=row[label_col], start=start, end=end))
    return events


# ---------------------------------------------------------------------------


def build_multiscale_data(
    events: Sequence[ObservedEvent],
    slot_minutes: int,
    branching: Sequence[int],
    labels: Optional[Sequence[str]] = None,
    origin: Optional[datetime] = None,
    finest_binary: bool = True,
    context: Optional[Sequence[np.ndarray]] = None,
    observed: Optional[Sequence[bool]] = None,
) -> Tuple[MultiScaleData, SlotCalendar, List[str]]:
    """
    Lay observed events on a slot grid and aggregate them into MultiScaleData.

    Parameters
    ----------
    events        : observed episodes; the EventRecord of core.calendar also
                    fits, since only label, start and end are read.
    slot_minutes  : wall-clock length of one finest-scale slot.
    branching     : branching factors, finest pair first; length K-1.
    labels        : label order to use; defaults to the sorted labels observed.
    origin        : first slot boundary; defaults to the earliest start floored
                    to the slot grid of its own day.
    finest_binary : if True, two starts of one label in one slot are rejected.
    context       : optional per-scale context arrays; the default is (H_k, 0).
    observed      : optional bool flag per finest slot of the record, from the
                    origin forward and at most as long as the padded horizon;
                    the default is a window that was watched throughout.

    Returns
    -------
    (data, calendar, label_names)

    Raises
    ------
    ValueError
        If the log is empty, carries a label outside an explicit label list,
        holds an episode that ends before it starts or starts before the
        origin, packs two starts of one label into one slot while
        finest_binary holds, supplies context of the wrong shape, or supplies
        a mask that is not one boolean flag per finest slot.
    """
    if not events:
        raise ValueError("an empty event log fixes neither an origin nor a horizon")

    label_names = _resolve_labels(events, labels)
    index = {name: i for i, name in enumerate(label_names)}
    slot_seconds = float(slot_minutes) * 60.0

    if origin is None:
        origin = _floor_to_grid(min(e.start for e in events), slot_minutes)

    placed: List[Tuple[int, int, int]] = []  # (slot, slots occupied, label index)
    for e in events:
        # max(1, ...) below would silently turn a reversed episode into a
        # one-slot one, so reject it here.
        if e.end < e.start:
            raise ValueError(
                f"event '{e.label}' ends at {e.end.isoformat()} before its "
                f"start {e.start.isoformat()}"
            )
        offset = (e.start - origin).total_seconds()
        if offset < 0:
            raise ValueError(
                f"event '{e.label}' starts at {e.start.isoformat()}, before the "
                f"origin {origin.isoformat()}"
            )
        # An event of zero length still occupies the slot it starts in.
        n_occ = max(1, math.ceil((e.end - e.start).total_seconds() / slot_seconds))
        placed.append((int(offset // slot_seconds), n_occ, index[e.label]))

    span = 1
    for b in branching:
        span *= int(b)
    n_slots = max(t + n for t, n, _ in placed)
    T_finest = -(-n_slots // span) * span  # pad up to whole coarsest epochs

    L = len(label_names)
    X = np.zeros((T_finest, L), dtype=np.int64)
    Y = np.zeros((T_finest, L), dtype=np.int64)
    for t, n_occ, ell in placed:
        X[t, ell] += 1
        Y[t, ell] += n_occ

    if finest_binary:
        clashes = np.argwhere(X > 1)
        if clashes.size:
            t, ell = int(clashes[0, 0]), int(clashes[0, 1])
            raise ValueError(
                f"label '{label_names[ell]}' has {int(X[t, ell])} starts in "
                f"finest slot {t}, but the finest scale is an event-start "
                f"process; rebuild with a shorter slot_minutes"
            )

    hierarchy = ScaleHierarchy.from_uniform_branching(T_finest, list(branching))
    X_scales = [X]
    Y_scales = [Y]
    for k in range(1, hierarchy.K):
        pm = hierarchy.parent_maps[k - 1]
        H_parent = hierarchy.n_epochs(k + 1)
        X_parent = np.zeros((H_parent, L), dtype=np.int64)
        Y_parent = np.zeros((H_parent, L), dtype=np.int64)
        np.add.at(X_parent, pm, X_scales[k - 1])
        np.add.at(Y_parent, pm, Y_scales[k - 1])
        X_scales.append(X_parent)
        Y_scales.append(Y_parent)

    if context is None:
        C = [
            np.zeros((hierarchy.n_epochs(k), 0), dtype=np.float64)
            for k in range(1, hierarchy.K + 1)
        ]
    else:
        C = [np.asarray(c, dtype=np.float64) for c in context]
        _check_context(C, hierarchy)

    calendar = SlotCalendar.from_config(branching, slot_minutes, origin)
    data = MultiScaleData(
        hierarchy=hierarchy,
        label_names=list(label_names),
        X=X_scales,
        Y=Y_scales,
        C=C,
        calendar=calendar,
        observed=_resolve_mask(observed, T_finest),
    )
    return data, calendar, list(label_names)


# ---------------------------------------------------------------------------


def _resolve_labels(
    events: Sequence[ObservedEvent],
    labels: Optional[Sequence[str]],
) -> List[str]:
    """Fix the label order, rejecting a label outside an explicit list."""
    if labels is None:
        return sorted({e.label for e in events})
    names = [str(name) for name in labels]
    known = set(names)
    for e in events:
        if e.label not in known:
            raise ValueError(
                f"label '{e.label}' is not in the supplied label list {names}"
            )
    return names


def _resolve_mask(
    observed: Optional[Sequence[bool]], T_finest: int
) -> Optional[np.ndarray]:
    """One flag per finest slot, with the pad up to whole coarsest epochs observed."""
    if observed is None:
        return None
    flags = np.asarray(observed)
    if flags.dtype != np.bool_:
        raise ValueError(
            f"an observation mask is a boolean array; this one has dtype "
            f"{flags.dtype}, which every reader would take for a flag wherever "
            "it is non-zero"
        )
    if flags.ndim != 1 or flags.size > T_finest:
        raise ValueError(
            f"an observation mask carries one flag per finest slot, so at most "
            f"{T_finest} of them once the horizon is padded to whole coarsest "
            f"epochs; this one has shape {flags.shape}"
        )
    # The slots between the last episode and the end of the coarsest epoch it
    # falls in count as watched.  They sit inside an epoch that also holds
    # recorded slots, so calling them absent would leave that epoch part
    # observed and part not, which `mask.check_ladder` refuses because its
    # stored total is then a sum over the observed children alone with nothing
    # in the container saying so.  The alternatives are to drop the epoch, which
    # discards recorded episodes, or to widen the mask to cover it, which is
    # what happens here and what every result this project has stored assumes.
    out = np.ones(int(T_finest), dtype=bool)
    out[: flags.size] = flags
    return out


def _check_context(C: Sequence[np.ndarray], hierarchy: ScaleHierarchy) -> None:
    """Reject context arrays that miss the padded epoch count of their scale."""
    if len(C) != hierarchy.K:
        raise ValueError(
            f"context supplies {len(C)} arrays but the hierarchy has "
            f"{hierarchy.K} scales"
        )
    for k, c in enumerate(C, start=1):
        H_k = hierarchy.n_epochs(k)
        if c.ndim != 2 or c.shape[0] != H_k:
            raise ValueError(
                f"context at scale {k} has shape {c.shape}, but the horizon "
                f"padded to whole coarsest epochs needs ({H_k}, d_{k})"
            )


def _floor_to_grid(when: datetime, slot_minutes: int) -> datetime:
    """Floor a timestamp to the slot grid anchored at midnight of its own day."""
    midnight = when.replace(hour=0, minute=0, second=0, microsecond=0)
    n = int((when - midnight).total_seconds() // (float(slot_minutes) * 60.0))
    return midnight + timedelta(minutes=slot_minutes * n)


def fitted_trace_round_trip(
    params_list,
    hierarchy,
    label_names: Sequence[str],
    calendar,
    slot_minutes: int,
    branching: Sequence[int],
    seed: int = 0,
    finest_binary: bool = True,
) -> Dict[str, float]:
    """
    Sample a fitted model and check its counts and durations survive an event log.

    A multiscale summary that cannot be turned back into dated episodes answers
    a narrower question than the one asked.  The round trip is checked at
    ingestion elsewhere; this checks it for the arrays a fit produces, by
    sampling from the fitted parameters, writing the sample as episodes with
    wall-clock start and end times, and reading it back.

    Args:
        params_list: fitted parameters, index 0 = scale 1.
        hierarchy: the scale hierarchy the fit used.
        label_names: label order of the fitted model.
        calendar: slot calendar giving epochs a wall-clock origin.
        slot_minutes: wall-clock length of one finest slot.
        branching: branching factors, finest pair first.
        seed: fixes the sample.
        finest_binary: whether an epoch holds at most one start per label.

    Returns:
        The largest absolute count and duration discrepancy of the round trip,
        and the number of episodes it passed through.  A start booked zero
        duration units occupies one slot once it is written as an episode with a
        start and an end, so the duration it is compared against is the sampled
        one raised to at least one wherever a start occurred.
    """
    from ..synthetic.generator import N2DBNGenerator

    gen = N2DBNGenerator(
        params_list=params_list,
        hierarchy=hierarchy,
        label_names=list(label_names),
        calendar=calendar,
        finest_binary=finest_binary,
    )
    sampled = gen.generate(seed=seed)
    records = sampled.event_log()
    rebuilt, _, _ = build_multiscale_data(
        records,
        slot_minutes=slot_minutes,
        branching=list(branching),
        labels=list(label_names),
        origin=calendar.origin,
        finest_binary=finest_binary,
    )
    X1, Y1 = sampled.data.get_X(1), sampled.data.get_Y(1)
    n = min(rebuilt.get_X(1).shape[0], X1.shape[0])
    expected_Y = np.maximum(Y1[:n], (X1[:n] > 0).astype(Y1.dtype))
    dx = np.abs(rebuilt.get_X(1)[:n] - X1[:n]).max()
    dy = np.abs(rebuilt.get_Y(1)[:n] - expected_Y).max()
    return {
        "count_error": float(dx),
        "duration_error": float(dy),
        "n_episodes": float(len(records)),
    }
