"""
core/projection.py
==================
The block-constant projector, and the two components a resolution can identify.

A child kernel is normalized over the children of one parent epoch, so adding a
constant to every child of a block leaves the kernel where it was.  The
coordinates on a block therefore split orthogonally into the block-constant
direction and the within-block contrasts, and only the second of the two is
identified at that scale.  The first acts on the data through the parent total
and is identified above.

`BlockProjector` is that split.  `mean` gives the one number per block, `expand`
puts a per-block number back on the fine grid, `shadow` is their composition and
`detail` is the remainder.  The two components are orthogonal under the counting
measure, they sum to what came in, and `shadow` is idempotent, which is what
makes the decomposition a projection and not a convention.

`ScaleBroadcast` is the operator the coarse-to-fine message carries.  A
parent-scale coefficient is stated in the parent's feature units and at the
parent's lag, so it reaches the child rescaled by the ratio of feature standard
deviations and by the ratio of lags.  The image of the broadcast is block
constant, so a parent effect and a child shadow are the same object read at two
scales, and `to_parent` inverts `to_child` exactly.

`invariance_residual` measures the invariance.  Any density normalized over a
block is unchanged when a block-constant vector is added to its predictor, and a
density that moves is one the account does not describe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np

from .blocks import BlockIndex


@dataclass(frozen=True)
class BlockProjector:
    """The orthogonal split of a fine array into its block mean and its contrasts."""

    block: BlockIndex

    @classmethod
    def from_parent_map(cls, parent_map: np.ndarray, n_blocks: int) -> "BlockProjector":
        """The projector of a fine grid onto the blocks a parent map declares."""
        return cls(BlockIndex.from_parent_map(parent_map, n_blocks))

    @property
    def n_blocks(self) -> int:
        return self.block.n_blocks

    @property
    def size(self) -> int:
        return int(self.block.parent.size)

    # ------------------------------------------------------------------
    def mean(self, values: np.ndarray) -> np.ndarray:
        """The average over the children of each block, as (n_blocks, ...)."""
        values = np.asarray(values, dtype=float)
        self._check(values)
        totals = self.block.sum(values)
        sizes = np.maximum(self.block.sizes, 1)
        return totals / sizes.reshape((-1,) + (1,) * (values.ndim - 1))

    def expand(self, per_block: np.ndarray) -> np.ndarray:
        """One number per block read at every child of it, which is the fine grid."""
        per_block = np.asarray(per_block, dtype=float)
        if per_block.shape[0] != self.n_blocks:
            raise ValueError(
                f"a per-block array carries one entry per block, so {self.n_blocks} "
                f"of them; got {per_block.shape[0]}"
            )
        return self.block.broadcast(per_block)

    def shadow(self, values: np.ndarray) -> np.ndarray:
        """The block-constant component, which this scale cannot identify."""
        return self.expand(self.mean(values))

    def detail(self, values: np.ndarray) -> np.ndarray:
        """The within-block contrasts, which this scale identifies."""
        return np.asarray(values, dtype=float) - self.shadow(values)

    def split(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """The shadow and the detail, which sum to what came in."""
        low = self.shadow(values)
        return low, np.asarray(values, dtype=float) - low

    def is_block_constant(self, values: np.ndarray, tol: float = 1e-12) -> bool:
        """Whether a fine array lies in the block-constant subspace."""
        return bool(np.max(np.abs(self.detail(values)), initial=0.0) <= tol)

    def _check(self, values: np.ndarray) -> None:
        if values.ndim == 0 or values.shape[0] != self.size:
            shape = () if values.ndim == 0 else values.shape
            raise ValueError(
                f"a fine array carries one row per epoch, so {self.size} of them; "
                f"got {shape}"
            )


@dataclass(frozen=True)
class ScaleBroadcast:
    """
    The parent-to-child operator of the coarse-to-fine message.

    Attributes
    ----------
    projector : the blocks of the child scale under its parents
    sd_child  : the standard deviation of the relation's feature at the child
    sd_parent : the same at the parent
    lag_child : the lag at which the relation is active at the child scale
    lag_parent: the same at the parent scale, which the ratio converts
    """

    projector: BlockProjector
    sd_child: float = 1.0
    sd_parent: float = 1.0
    lag_child: int = 1
    lag_parent: int = 1

    def __post_init__(self) -> None:
        if self.sd_parent <= 0.0 or self.sd_child <= 0.0:
            raise ValueError(
                f"a feature's standard deviation is positive; got child "
                f"{self.sd_child} and parent {self.sd_parent}"
            )
        if self.lag_child < 1 or self.lag_parent < 1:
            raise ValueError(
                f"a lag is at least one epoch; got child {self.lag_child} and "
                f"parent {self.lag_parent}"
            )

    @property
    def factor(self) -> float:
        """The units the parent coefficient is restated in at the child scale."""
        return (self.sd_child / self.sd_parent) * (self.lag_parent / self.lag_child)

    def to_child(self, parent: np.ndarray) -> np.ndarray:
        """A parent-scale coefficient read at every child epoch."""
        return self.factor * self.projector.expand(parent)

    def to_parent(self, child: np.ndarray) -> np.ndarray:
        """The child-scale shadow read back as the parent coefficient it came from."""
        return self.projector.mean(child) / self.factor


def invariance_residual(
    log_density: Callable[[np.ndarray], float],
    values: np.ndarray,
    projector: BlockProjector,
    offsets: np.ndarray,
) -> float:
    """
    How far a block-normalized density moves when its predictor gains a shadow.

    Args:
        log_density: the density as a function of the fine predictor.
        offsets: one constant per block, which `expand` puts on the fine grid.

    Returns:
        The absolute change, which the block-constant invariance puts at zero.
    """
    values = np.asarray(values, dtype=float)
    base = float(log_density(values))
    moved = float(log_density(values + projector.expand(offsets)))
    return abs(moved - base)
