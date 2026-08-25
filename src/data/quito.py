"""
data/quito.py
=============
The QuitoBench application-traffic panels, read onto their nested ladders.

Why this benchmark and not an aggregate one
-------------------------------------------
The model is defined on event counts at a fine resolution with every coarser
scale a sum of them, so a benchmark tests it only when the file itself holds
observations at the fine resolution.  QuitoBench publishes production traffic at
ten minutes and at one hour over thousands of consecutive epochs per series,
which populates every scale of the ladders below from real records.

Which channels are read, and why four of five
---------------------------------------------
Each series carries five anonymised variates.  `ind_2` to `ind_5` are integer at
every observed row, so they are counts and their block sums are the counts of
the coarser epoch.  `ind_1` is continuous, at two tenths of one percent integer,
so it is a rate or a latency; adding six or twenty four of them does not produce
the value of the coarser epoch, and a model whose coarse scales are sums cannot
carry it.  It is dropped, and every arm compared reads the same four channels.

The zero encoding
-----------------
The hourly split holds no exact zero in any count channel and holds `1e-5`
instead, so `1e-5` is the encoding of zero there.  The ten-minute split holds
both, so `1e-5` there is read as zero as well and that reading is a decision
this module makes rather than one the published data documents.

The two panels do not share series
----------------------------------
The hourly and ten-minute splits carry disjoint series identifiers, so a
ten-minute scale cannot be placed under an hourly series.  Each panel is fitted
on its own ladder.
"""

from __future__ import annotations

import os
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..core.scales import ScaleHierarchy
from ..core.variables import MultiScaleData

#: The four integer variates, which are the counts the ladder sums.
COUNT_CHANNELS: Tuple[str, ...] = ("ind_2", "ind_3", "ind_4", "ind_5")

#: The continuous variate, named so a reader sees it was considered.
RATE_CHANNEL = "ind_1"

#: Values at or below this are read as zero.
ZERO_LEVEL = 1e-4

#: The benchmark's declared train and test boundary, in UTC.
CUTOFF = "2023-07-28 00:00:00"


class Panel:
    """One published split, with the ladder and the calendar columns it reads."""

    def __init__(
        self,
        name: str,
        filename: str,
        length: int,
        branching: Sequence[int],
        season: Sequence[int],
        harmonics: Sequence[int],
        trend: Sequence[bool],
        block: int,
    ) -> None:
        self.name = name
        self.filename = filename
        self.length = length
        self.branching = tuple(branching)
        self.season = tuple(season)
        self.harmonics = tuple(harmonics)
        self.trend = tuple(trend)
        #: Finest epochs in one coarsest epoch of the ladder.
        self.block = block

    @property
    def K(self) -> int:
        """Scales the ladder holds."""
        return len(self.branching) + 1


#: Ten minutes, hour, day.  The ten-minute scale is what makes this panel the
#: working one: it supplies the finest observations the upward pass reads.
MIN_PANEL = Panel(
    name="min",
    filename="test_min.parquet",
    length=5904,
    branching=(6, 24),
    season=(6, 24, 1),
    harmonics=(2, 3, 0),
    trend=(False, False, True),
    block=144,
)

#: Hour, day, week.
HOUR_PANEL = Panel(
    name="hour",
    filename="test_hour.parquet",
    length=15356,
    branching=(24, 7),
    season=(24, 7, 1),
    harmonics=(3, 2, 0),
    trend=(False, False, True),
    block=168,
)

PANELS: Dict[str, Panel] = {"min": MIN_PANEL, "hour": HOUR_PANEL}


def read_panel(
    panel: Panel, root: str = "data/quito"
) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    """
    One split, as a shared timestamp grid and one count array per series.

    Args:
        panel: which split to read.
        root: directory holding the cached parquet files.

    Returns:
        The shared `(T,)` array of timestamps, and a mapping from series id to
        its `(T, 4)` integer counts in the order of `COUNT_CHANNELS`.

    Raises:
        FileNotFoundError: the file is absent.
        ValueError: the series do not share one timestamp grid.
    """
    path = os.path.join(root, panel.filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} is absent; the QuitoBench splits are fetched once from the "
            "published dataset and cached outside version control"
        )
    import pandas as pd
    import pyarrow.parquet as pq

    handle = pq.ParquetFile(path)
    columns = ["item_id", "date_time", *COUNT_CHANNELS]
    parts: Dict[int, List] = {}
    for group in range(handle.metadata.num_row_groups):
        frame = handle.read_row_group(group, columns=columns).to_pandas()
        for item, rows in frame.groupby("item_id"):
            parts.setdefault(int(item), []).append(rows)

    grid = None
    series: Dict[int, np.ndarray] = {}
    for item, chunks in parts.items():
        rows = pd.concat(chunks).sort_values("date_time")
        stamps = rows["date_time"].to_numpy()
        if grid is None:
            grid = stamps
        elif stamps.shape != grid.shape or not np.array_equal(stamps, grid):
            raise ValueError(
                f"series {item} carries {stamps.size} timestamps against "
                f"{grid.size} for the first series read, and this loader "
                "returns one grid shared by every series"
            )
        values = rows[list(COUNT_CHANNELS)].to_numpy()
        series[item] = np.rint(np.where(values > ZERO_LEVEL, values, 0.0)).astype(
            np.int64
        )
    return grid, series


def four_channel_ids(series: Dict[int, np.ndarray]) -> List[int]:
    """
    The series whose four count channels all fire at least once.

    A channel that never fires leaves the cross-channel graph one node short, so
    the panel is confined to the series that supply all four.

    Args:
        series: the mapping `read_panel` returned.

    Returns:
        The sorted ids.
    """
    return sorted(k for k, v in series.items() if (v > 0).any(axis=0).all())


def split_at_cutoff(grid: np.ndarray, cutoff: str = CUTOFF) -> int:
    """
    The index of the first epoch at or after the declared boundary.

    Args:
        grid: the shared timestamp array.
        cutoff: the boundary, as a timestamp string.

    Returns:
        The number of epochs before it.
    """
    import pandas as pd

    return int((np.asarray(grid) < pd.Timestamp(cutoff).to_datetime64()).sum())


def whole_blocks_before(n_epochs: int, block: int, blocks: int = 0) -> Tuple[int, int]:
    """
    The slice of a block ending at `n_epochs` that holds whole coarsest epochs.

    The cut takes from the front, so the window ends at the boundary and every
    coarsest epoch inside it is complete.

    Args:
        n_epochs: epochs available before the boundary.
        block: finest epochs in one coarsest epoch.
        blocks: coarsest epochs wanted; zero takes as many as fit.

    Returns:
        The start and end index of the retained window.

    Raises:
        ValueError: fewer whole blocks are available than were asked for.
    """
    available = n_epochs // block
    wanted = blocks or available
    if wanted > available or wanted == 0:
        raise ValueError(
            f"{available} whole blocks of {block} epochs are available and "
            f"{wanted} were asked for"
        )
    return n_epochs - wanted * block, n_epochs


def seasonal_context(
    n_epochs: int, period: int, harmonics: int, offset: int = 0
) -> np.ndarray:
    """
    Sine and cosine pairs of a fixed cycle, evaluated at consecutive epochs.

    Args:
        n_epochs: rows to return.
        period: epochs in one cycle.
        harmonics: pairs to return; zero returns a width-zero matrix.
        offset: epoch index the first row stands at, so a forecast window keeps
            the phase of the window before it.

    Returns:
        An `(n_epochs, 2 * harmonics)` array.
    """
    t = np.arange(offset, offset + n_epochs, dtype=float)
    cols: List[np.ndarray] = []
    for h in range(1, harmonics + 1):
        angle = 2.0 * np.pi * h * t / float(period)
        cols.append(np.sin(angle))
        cols.append(np.cos(angle))
    if not cols:
        return np.zeros((n_epochs, 0))
    return np.column_stack(cols)


def trend_context(n_epochs: int, offset: int, total: int) -> np.ndarray:
    """
    One centered column running from -1 to 1 over `total` epochs.

    Args:
        n_epochs: rows to return.
        offset: epoch index the first row stands at.
        total: epochs the full span holds, which fixes the scaling so a forecast
            epoch continues the line the training epochs sat on.

    Returns:
        An `(n_epochs, 1)` array.
    """
    t = np.arange(offset, offset + n_epochs, dtype=float)
    half = max(total - 1, 1) / 2.0
    return ((t - half) / half)[:, None]


def context_ladder(
    n_epochs: Sequence[int],
    offsets: Sequence[int],
    totals: Sequence[int],
    season: Sequence[int],
    harmonics: Sequence[int],
    trend: Sequence[bool],
) -> List[np.ndarray]:
    """
    One context matrix per scale, finest first, of calendar columns.

    Args:
        n_epochs: rows wanted at each scale.
        offsets: epoch index each scale's first row stands at.
        totals: epochs the full span holds at each scale.
        season: cycle length per scale.
        harmonics: harmonic pairs per scale.
        trend: whether each scale carries the trend column.

    Returns:
        One `(n_epochs[k], d_k)` array per scale.
    """
    out: List[np.ndarray] = []
    for rows, offset, total, per, harm, carries in zip(
        n_epochs, offsets, totals, season, harmonics, trend
    ):
        columns = [seasonal_context(rows, per, harm, offset)]
        if carries:
            columns.append(trend_context(rows, offset, total))
        out.append(np.column_stack(columns))
    return out


def aggregate_up(finest: np.ndarray, hierarchy: ScaleHierarchy) -> List[np.ndarray]:
    """
    The finest array and every block sum above it, finest first.

    Args:
        finest: `(H_1, L)` counts.
        hierarchy: the chain to sum along.

    Returns:
        One `(H_k, L)` array per scale.
    """
    out = [np.asarray(finest, dtype=np.int64)]
    for ki in range(hierarchy.K - 1):
        parent = np.asarray(hierarchy.parent_maps[ki])
        up = np.zeros((hierarchy.H[ki + 1], out[-1].shape[1]), dtype=np.int64)
        np.add.at(up, parent, out[-1])
        out.append(up)
    return out


def collapse(counts: np.ndarray, unit: int) -> np.ndarray:
    """
    Counts summed into blocks of `unit` finest epochs.

    Args:
        counts: `(T, L)` finest-scale counts, T a multiple of `unit`.
        unit: finest epochs per returned row.

    Returns:
        A `(T // unit, L)` array.
    """
    counts = np.asarray(counts)
    return counts.reshape(-1, unit, counts.shape[1]).sum(axis=1)


def window_to_multiscale(
    counts: np.ndarray,
    panel: Panel,
    offset: int = 0,
    total_finest: int = 0,
) -> MultiScaleData:
    """
    One window of counts, on the panel's ladder, with its calendar columns.

    Args:
        counts: `(T, L)` counts at the panel's finest scale, T already a whole
            number of coarsest blocks.
        panel: the ladder and calendar columns to build.
        offset: finest epoch index the window starts at, so a forecast window
            continues the phase and the trend of the window before it.
        total_finest: finest epochs the full span holds; zero reads the window
            as the whole span.

    Returns:
        The data object, with durations identically zero.
    """
    counts = np.asarray(counts, dtype=np.int64)
    hierarchy = ScaleHierarchy.from_uniform_branching(
        counts.shape[0], list(panel.branching)
    )
    X = aggregate_up(counts, hierarchy)
    span = total_finest or counts.shape[0]
    step = [1] + [int(np.prod(panel.branching[:i])) for i in range(1, hierarchy.K)]
    C = context_ladder(
        n_epochs=hierarchy.H,
        offsets=[offset // s for s in step],
        totals=[max(span // s, 1) for s in step],
        season=panel.season,
        harmonics=panel.harmonics,
        trend=panel.trend,
    )
    return MultiScaleData(
        hierarchy=hierarchy,
        label_names=list(COUNT_CHANNELS),
        X=X,
        Y=[np.zeros_like(a) for a in X],
        C=C,
    )
