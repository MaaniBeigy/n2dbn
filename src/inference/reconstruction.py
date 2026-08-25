"""
inference/reconstruction.py
============================
Convert finest-scale (X^(1), Y^(1)) arrays into event episode traces.

Episode representation
----------------------
Under the atomic-resolution assumption:
    X_t^{(1,ℓ)} ∈ {0,1}  -  1 if a new episode of label ℓ starts at slot t

Episode:  e_t^{(ℓ)} = (ℓ, t_start, t_end)
    where t_end = t_start + L_t^{(ℓ)} - 1  and  L_t^{(ℓ)} = Y_t^{(1,ℓ)}

Trace for label ℓ:  ordered list of (ℓ, t_start, t_end) tuples.

Exclusivity projection
----------------------
For labels in ℒ_ex (at most one active episode at a time):
    If episode j and j+1 overlap, truncate episode j to end just before j+1 starts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Episode:
    """A single reconstructed event episode."""

    label: int  # 0-based label index
    t_start: int  # finest-scale slot (0-based)
    t_end: int  # finest-scale slot (inclusive)

    @property
    def duration(self) -> int:
        """Length in slots."""
        return max(0, self.t_end - self.t_start + 1)

    def overlaps(self, other: "Episode") -> bool:
        return self.t_start <= other.t_end and other.t_start <= self.t_end

    def __repr__(self) -> str:
        return f"Episode(ℓ={self.label}, [{self.t_start}, {self.t_end}], dur={self.duration})"


# ---------------------------------------------------------------------------


class EpisodeReconstructor:
    """
    Converts (X^(1), Y^(1)) at the finest scale into episode traces.

    Parameters
    ----------
    label_names    : list of label strings (for display).
    exclusivity_set: set of label indices for which at most one episode
                     may be active at any time  (optional; default empty).
    """

    def __init__(
        self,
        label_names: List[str],
        exclusivity_set: Optional[Set[int]] = None,
    ) -> None:
        self.label_names = label_names
        self.L = len(label_names)
        self.exclusivity_set = exclusivity_set or set()

    # ------------------------------------------------------------------
    def reconstruct(
        self,
        X1: np.ndarray,  # (H_1, L)  int  - finest-scale counts ∈ {0,1}
        Y1: np.ndarray,  # (H_1, L)  int  - slot durations
    ) -> Dict[int, List[Episode]]:
        """
        Reconstruct episode traces for all labels.

        Parameters
        ----------
        X1 : (H_1, L) array of 0/1 start indicators.
        Y1 : (H_1, L) array of episode lengths in slots.

        Returns
        -------
        traces : dict {label_idx: [Episode, ...] sorted by t_start}
        """
        H1 = X1.shape[0]
        traces: Dict[int, List[Episode]] = {}

        for ell in range(self.L):
            starts = np.where(X1[:, ell] == 1)[0]  # start slots
            episodes: List[Episode] = []

            for t in starts:
                # A slot that starts an episode is occupied by it, so the
                # shortest episode the trace can hold is one slot.  The
                # allocation model may book zero duration units to a start when
                # its parent epoch has fewer units than active children.
                L_t = max(int(Y1[t, ell]), 1)
                t_end = t + L_t - 1
                t_end = min(t_end, H1 - 1)  # clip to horizon
                episodes.append(Episode(label=ell, t_start=int(t), t_end=t_end))

            # Apply exclusivity projection if needed
            if ell in self.exclusivity_set and len(episodes) > 1:
                episodes = self._project_exclusive(episodes)

            traces[ell] = sorted(episodes, key=lambda e: e.t_start)

        return traces

    # ------------------------------------------------------------------
    def _project_exclusive(self, episodes: List[Episode]) -> List[Episode]:
        """
        Deterministic truncation to enforce at-most-one-active constraint.

        For consecutive episodes (b_j, b_{j+1}), if they overlap:
            t_end_j  ←  min(t_end_j,  b_{j+1} - 1)

        The last episode is left at its original end.
        """
        eps = sorted(episodes, key=lambda e: e.t_start)
        projected = []
        for j in range(len(eps)):
            if j < len(eps) - 1:
                new_end = min(eps[j].t_end, eps[j + 1].t_start - 1)
            else:
                new_end = eps[j].t_end
            new_end = max(eps[j].t_start, new_end)  # ensure non-negative duration
            projected.append(
                Episode(
                    label=eps[j].label,
                    t_start=eps[j].t_start,
                    t_end=new_end,
                )
            )
        return projected

    # ------------------------------------------------------------------
    def occupancy(
        self,
        traces: Dict[int, List[Episode]],
        H1: int,
    ) -> np.ndarray:
        """
        Build (H_1, L) binary occupancy array: A_s^{(ℓ)} > 0.

        A_s[ell] = 1 iff slot s is occupied by at least one episode of label ell.
        """
        A = np.zeros((H1, self.L), dtype=np.int8)
        for ell, eps in traces.items():
            for e in eps:
                A[e.t_start : e.t_end + 1, ell] = 1
        return A

    # ------------------------------------------------------------------
    def iou(
        self,
        traces_pred: Dict[int, List[Episode]],
        traces_true: Dict[int, List[Episode]],
        H1: int,
    ) -> Dict[int, float]:
        """
        Slot-level Intersection-over-Union per label.

        IoU^{(ℓ)} = |{s : A_pred_s=1 ∧ A_true_s=1}| / |{s : A_pred_s=1 ∨ A_true_s=1}|
        """
        A_pred = self.occupancy(traces_pred, H1)
        A_true = self.occupancy(traces_true, H1)
        ious: Dict[int, float] = {}

        for ell in range(self.L):
            inter = int((A_pred[:, ell] & A_true[:, ell]).sum())
            union = int((A_pred[:, ell] | A_true[:, ell]).sum())
            ious[ell] = inter / union if union > 0 else 1.0

        return ious

    # ------------------------------------------------------------------
    def start_detection_f1(
        self,
        traces_pred: Dict[int, List[Episode]],
        traces_true: Dict[int, List[Episode]],
    ) -> Dict[int, float]:
        """
        Precision / recall / F1 on start-slot detection, per label.

        Returns dict {ell: {'precision', 'recall', 'f1'}}.
        """
        results: Dict[int, dict] = {}
        for ell in range(self.L):
            pred_starts = {e.t_start for e in traces_pred.get(ell, [])}
            true_starts = {e.t_start for e in traces_true.get(ell, [])}
            tp = len(pred_starts & true_starts)
            fp = len(pred_starts - true_starts)
            fn = len(true_starts - pred_starts)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            results[ell] = {"precision": prec, "recall": rec, "f1": f1}
        return results
