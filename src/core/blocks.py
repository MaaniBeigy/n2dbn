"""
core/blocks.py
==============
Parent-block index for vectorised finer-scale operations.

Every finer scale k < K groups its epochs into blocks: the children of one
parent epoch c at scale k+1.  Count/duration allocation, the Dir-Mult
normaliser and the allocation gradients are all *segment* operations over
those blocks.  Doing them with a Python `for c in range(H_coarse)` loop is
what made the E/M/S-steps quadratic in wall-clock; `BlockIndex` replaces
that with `np.add.reduceat` style segment reductions.

Usage::

    bi = BlockIndex.from_hierarchy(hier, k)
    tot   = bi.sum(X_k)              # (n_blocks, L)  Σ_{t∈ch(c)} X_t
    lse   = bi.logsumexp(eta)        # (n_blocks, ...) log Σ_{t∈ch(c)} e^{η_t}
    back  = bi.broadcast(lse)        # (T, ...)        value of each epoch's block
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class BlockIndex:
    """
    Segment index mapping fine epochs to parent blocks.

    Attributes
    ----------
    parent   : (T,) int - block id of each fine epoch
    n_blocks : number of parent blocks
    order    : (T,) int or None - permutation sorting epochs by block
               (None when `parent` is already non-decreasing, the common case)
    starts   : (n_blocks,) int - offset of each block in the sorted order
    sizes    : (n_blocks,) int - number of children per block
    """

    parent: np.ndarray
    n_blocks: int
    order: Optional[np.ndarray]
    starts: np.ndarray
    sizes: np.ndarray

    # ------------------------------------------------------------------
    @classmethod
    def from_parent_map(cls, parent_map: np.ndarray, n_blocks: int) -> "BlockIndex":
        parent = np.asarray(parent_map, dtype=np.int64)
        sizes = np.bincount(parent, minlength=n_blocks).astype(np.int64)
        sorted_already = bool(np.all(np.diff(parent) >= 0))
        order = None if sorted_already else np.argsort(parent, kind="stable")
        starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
        return cls(
            parent=parent,
            n_blocks=n_blocks,
            order=order,
            starts=starts,
            sizes=sizes,
        )

    @classmethod
    def from_hierarchy(cls, hier, k: int) -> "BlockIndex":
        """Block index of scale-k epochs under their scale-(k+1) parents."""
        return cls.from_parent_map(hier.parent_maps[k - 1], hier.n_epochs(k + 1))

    # ------------------------------------------------------------------
    @property
    def _has_empty(self) -> bool:
        return bool(np.any(self.sizes == 0))

    def _sorted(self, arr: np.ndarray) -> np.ndarray:
        return arr if self.order is None else arr[self.order]

    # ------------------------------------------------------------------
    def sum(self, arr: np.ndarray) -> np.ndarray:
        """Segment sum along axis 0.  `arr` is (T, ...) to (n_blocks, ...)."""
        if self._has_empty:
            out = np.zeros((self.n_blocks,) + arr.shape[1:], dtype=arr.dtype)
            np.add.at(out, self.parent, arr)
            return out
        return np.add.reduceat(self._sorted(arr), self.starts, axis=0)

    def max(self, arr: np.ndarray) -> np.ndarray:
        """Segment max along axis 0 (empty blocks give -inf)."""
        if self._has_empty:
            out = np.full((self.n_blocks,) + arr.shape[1:], -np.inf, dtype=float)
            np.maximum.at(out, self.parent, arr)
            return out
        return np.maximum.reduceat(self._sorted(arr), self.starts, axis=0)

    def logsumexp(self, arr: np.ndarray) -> np.ndarray:
        """Numerically stable segment log-sum-exp along axis 0."""
        mx = self.max(arr)
        mx_safe = np.where(np.isfinite(mx), mx, 0.0)
        shifted = np.exp(arr - mx_safe[self.parent])
        total = self.sum(shifted)
        with np.errstate(divide="ignore"):
            out = np.log(total) + mx_safe
        return np.where(total > 0.0, out, -np.inf)

    def broadcast(self, block_arr: np.ndarray) -> np.ndarray:
        """Expand a per-block array (n_blocks, ...) back to (T, ...)."""
        return block_arr[self.parent]

    def exclusive_cumsum(self, arr: np.ndarray) -> np.ndarray:
        """
        Running total of the *earlier* epochs of each block: (T, ...).

        Epoch t receives the sum over the epochs of its own block that precede
        it, and zero for the first epoch of a block.  This is what the children
        of a block have already taken of their parent's total, which the
        allocation competes over and which no other term of the score carries.
        """
        order = np.arange(self.parent.size) if self.order is None else self.order
        ordered = arr[order]
        running = np.cumsum(ordered, axis=0) - ordered
        offset = np.zeros_like(running)
        take = self.starts[self.sizes > 0]
        offset[order] = running - np.repeat(
            running[take], self.sizes[self.sizes > 0], axis=0
        )
        return offset
