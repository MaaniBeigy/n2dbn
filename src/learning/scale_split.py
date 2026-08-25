"""
learning/scale_split.py
=======================
Every column as the band each resolution identifies, and the level none does.

A column held at scale k is constant inside a scale-k epoch, so under the
scale-(k+1) blocks it splits into a block mean and a set of within-block
contrasts.  The contrasts are what the scale-k pass identifies.  The block mean
is constant on every scale-(k+1) block, which makes it a scale-(k+1) column, and
the same split applies to it again one rung up.

Running that to the top of the ladder writes one column as a sum of bands, one
per scale from its own upward, plus the record mean.  The bands are orthogonal
under the counting measure and they add back to the column exactly, so nothing
is discarded and each piece has a resolution that can see it.  This is the
routing the two directions of the pass need.  The ascent carries a band's
sufficient statistics to the scale that owns it and the descent carries the
coarser bands down as the shadow a finer scale cannot estimate.

The parent of the coarsest scale is the whole record, so the top of the ladder
takes the same split with one block, and the level that comes out is the one
direction the intercept already carries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from ..core.projection import BlockProjector
from .fine_design import ScaleLadder
from .fine_sstep import ScaledColumn


def parent_map(ladder: ScaleLadder, scale: int) -> np.ndarray:
    """
    The scale-(k+1) epoch of every scale-k epoch, read through the finest slots.

    A scale-k epoch holds slots of exactly one scale-(k+1) parent, which
    `ScaleLadder` verifies, so every slot of an epoch writes the same parent and
    the order of the writes does not matter.

    Raises:
        ValueError: when the scale has no parent on this ladder.
    """
    if not 1 <= scale < ladder.K:
        raise ValueError(
            f"scale {scale} has no parent among the {ladder.K} scales of this ladder"
        )
    out = np.zeros(ladder.n_epochs[scale - 1], dtype=np.int64)
    out[ladder.ancestors[scale - 1]] = ladder.ancestors[scale]
    return out


def projector_at(ladder: ScaleLadder, scale: int) -> BlockProjector:
    """
    The blocks a scale-k column sits in, which are its scale-(k+1) parents.

    The coarsest scale has the whole record as its parent, so it comes back as
    one block over every epoch.
    """
    if scale == ladder.K:
        n_epochs = ladder.n_epochs[scale - 1]
        return BlockProjector.from_parent_map(np.zeros(n_epochs, dtype=np.int64), 1)
    return BlockProjector.from_parent_map(
        parent_map(ladder, scale), ladder.n_epochs[scale]
    )


@dataclass(frozen=True)
class SplitColumn:
    """One column as the shadow it shares with its parent and the detail it owns."""

    shadow: ScaledColumn
    detail: ScaledColumn

    @property
    def scale(self) -> int:
        return self.detail.scale


def split_column(column: ScaledColumn, ladder: ScaleLadder) -> SplitColumn:
    """
    One column's block mean, held one scale up, and its within-block contrasts.

    The shadow of a scale-k column is a scale-(k+1) column, since it takes one
    value per parent epoch.  The coarsest scale has the whole record as its
    parent, so its shadow is one number carrying the record mean and its scale
    index names the horizon, one rung above the ladder.
    """
    projector = projector_at(ladder, column.scale)
    shadow_values = projector.mean(column.values)
    detail_values = column.values - projector.expand(shadow_values)
    return SplitColumn(
        shadow=ScaledColumn(f"{column.name} shadow", column.scale + 1, shadow_values),
        detail=ScaledColumn(f"{column.name} detail", column.scale, detail_values),
    )


def band_decomposition(
    column: ScaledColumn, ladder: ScaleLadder
) -> Tuple[Dict[int, ScaledColumn], float]:
    """
    One column as the band each scale identifies, and the level none of them does.

    Returns:
        The bands keyed by the scale that identifies each, from the column's own
        scale to the coarsest, and the record mean the intercept carries.
    """
    bands: Dict[int, ScaledColumn] = {}
    current = column
    for scale in range(column.scale, ladder.K + 1):
        piece = split_column(current, ladder)
        bands[scale] = piece.detail
        current = piece.shadow
    return bands, float(current.values[0])


def reconstruct(
    bands: Dict[int, ScaledColumn], level: float, ladder: ScaleLadder
) -> np.ndarray:
    """The bands and the level summed at the finest grid, which is the column back."""
    out = np.full(ladder.n_slots, level, dtype=float)
    for band in bands.values():
        out = out + band.at_finest(ladder.ancestors)
    return out


def route(
    columns: List[ScaledColumn], ladder: ScaleLadder
) -> Dict[int, List[ScaledColumn]]:
    """
    Every column's bands gathered by the scale whose pass identifies them.

    A scale-k entry holds the contrasts that vary inside a scale-(k+1) parent
    block, which is what the scale-k likelihood can tell apart.  A column
    contributes to every scale from its own upward and to no scale below it.
    """
    out: Dict[int, List[ScaledColumn]] = {k: [] for k in range(1, ladder.K + 1)}
    for column in columns:
        bands, _ = band_decomposition(column, ladder)
        for scale, band in bands.items():
            out[scale].append(band)
    return out
