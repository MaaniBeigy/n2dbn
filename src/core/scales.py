"""
core/scales.py
==============
Nested temporal hierarchy for the N²DBN model.

Convention
----------
* Scale indices are **1-based**: k=1 is the finest, k=K is the coarsest.
* Internal arrays use 0-based epoch indices.
* parent_maps[k-1][t] = parent epoch of t at scale k+1  (None for k=K).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class ScaleHierarchy:
    """
    Encodes the nested temporal hierarchy over K scales.

    Attributes
    ----------
    K : int
        Number of temporal scales.
    H : list[int]
        H[k-1] = number of epochs at scale k.
    parent_maps : list[ndarray | None]
        parent_maps[k-1][t] = parent epoch index at scale k+1, for k < K.
        parent_maps[K-1] = None  (coarsest has no parent).
    """

    K: int
    H: List[int]
    parent_maps: List[Optional[np.ndarray]]
    # Pre-computed child sets: (k, c) -> sorted list of t in scale-k under parent c
    _children: Dict[tuple, List[int]] = field(
        default_factory=dict, init=False, repr=False
    )

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        assert len(self.H) == self.K, "H must have exactly K entries"
        assert len(self.parent_maps) == self.K, "parent_maps must have K entries"
        assert (
            self.parent_maps[self.K - 1] is None
        ), "parent_maps[K-1] must be None (coarsest has no parent)"
        self._build_children()

    def _build_children(self) -> None:
        """Pre-compute child-set dict: _children[(k, c)] = sorted epoch list."""
        self._children = {}
        for ki in range(self.K - 1):  # ki = k-1 to scale k = ki+1
            pm = self.parent_maps[ki]
            H_parent = self.H[ki + 1]
            buckets: Dict[int, List[int]] = {c: [] for c in range(H_parent)}
            for t, c in enumerate(pm):
                buckets[int(c)].append(t)
            for c, ch in buckets.items():
                self._children[(ki + 1, c)] = sorted(ch)  # key: (scale-k, parent-c)

    # ------------------------------------------------------------------
    # Core accessors  (1-indexed scales)
    # ------------------------------------------------------------------

    def n_epochs(self, k: int) -> int:
        """Number of epochs at scale k."""
        return self.H[k - 1]

    def parent(self, k: int, t: int) -> int:
        """
        Parent epoch of t at scale k in scale k+1.

        Parameters
        ----------
        k : int   Fine scale (must satisfy 1 <= k < K).
        t : int   Epoch index at scale k.
        """
        assert 1 <= k < self.K, f"Scale {k} has no parent (K={self.K})"
        return int(self.parent_maps[k - 1][t])

    def children(self, k: int, c: int) -> List[int]:
        """
        Child epochs of parent c (at scale k+1) in scale k.

        Parameters
        ----------
        k : int   Fine scale (must satisfy 1 <= k < K).
        c : int   Parent epoch index at scale k+1.
        """
        assert 1 <= k < self.K, f"Scale {k} has no children (K={self.K})"
        return self._children[(k, c)]

    def ancestor(self, k: int, t: int, target_k: int) -> int:
        """
        Ancestor of epoch t (at scale k) in scale target_k >= k.
        """
        assert target_k >= k
        cur = t
        for s in range(k, target_k):
            cur = self.parent(s, cur)
        return cur

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_uniform_branching(
        cls,
        H_finest: int,
        branching_factors: Sequence[int],
    ) -> "ScaleHierarchy":
        """
        Build from finest-scale epoch count and a list of branching factors.

        Parameters
        ----------
        H_finest : int
            Number of epochs at scale k=1.
        branching_factors : sequence[int]
            branching_factors[i] = children per parent between scale i+1 and i+2.
            Length must equal K-1.  Each factor must divide the scale above it.

        Examples
        --------
        >>> # 3-scale hierarchy: day / hour / minute
        >>> # H_1 = 1440 min, 60 min/hour, 24 hours/day
        >>> hier = ScaleHierarchy.from_uniform_branching(1440, [60, 24])
        >>> hier.n_epochs(1), hier.n_epochs(2), hier.n_epochs(3)
        (1440, 24, 1)
        """
        bfs = list(branching_factors)
        K = len(bfs) + 1
        H: List[int] = [H_finest]
        for bf in bfs:
            prev = H[-1]
            assert prev % bf == 0, f"H={prev} not divisible by branching factor {bf}"
            H.append(prev // bf)

        parent_maps: List[Optional[np.ndarray]] = []
        for ki, bf in enumerate(bfs):
            pm = np.arange(H[ki], dtype=np.int32) // bf
            parent_maps.append(pm)
        parent_maps.append(None)
        return cls(K=K, H=H, parent_maps=parent_maps)

    @classmethod
    def from_explicit_maps(
        cls,
        H: List[int],
        parent_maps: List[Optional[np.ndarray]],
    ) -> "ScaleHierarchy":
        """Direct construction - useful for irregular nestings."""
        return cls(K=len(H), H=H, parent_maps=parent_maps)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """
        Check internal consistency.  Raises AssertionError if anything is wrong.
        """
        for ki in range(self.K - 1):
            pm = self.parent_maps[ki]
            assert len(pm) == self.H[ki], (
                f"parent_map length mismatch at scale {ki+1}: "
                f"len={len(pm)}, expected H_{ki+1}={self.H[ki]}"
            )
            assert int(pm.min()) >= 0, f"Negative parent index at scale {ki+1}"
            assert (
                int(pm.max()) < self.H[ki + 1]
            ), f"Parent index {pm.max()} >= H_{ki+2}={self.H[ki+1]} at scale {ki+1}"
            unique = set(pm.tolist())
            assert len(unique) == self.H[ki + 1], (
                f"Parent map not surjective at scale {ki+1}: "
                f"found {len(unique)} distinct parents, expected {self.H[ki+1]}"
            )

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        lines = [f"ScaleHierarchy  K={self.K}"]
        for k in range(1, self.K + 1):
            tag = " (finest)" if k == 1 else (" (coarsest)" if k == self.K else "")
            lines.append(f"  k={k}: H={self.n_epochs(k)}{tag}")
        return "\n".join(lines)
