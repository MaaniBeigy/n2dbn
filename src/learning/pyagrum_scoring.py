"""
learning/pyagrum_scoring.py
===========================
Optional pyAgrum-based scoring backend for the S-step.

Provides BIC / BDeu / K2 scores via `pyAgrum.BNLearner` as an
alternative to the custom `LocalScore` in :mod:`learning.sstep`.

Usage
-----
These scores operate on **discretised** count data and are therefore
approximations relative to the exact expected log-likelihood used by
`LocalScore`.  They are most useful as:

* Initialisation heuristic - find a starting graph, then refine with
  the custom score.
* Sanity check - does BDeu agree with our custom scoring direction?
* Research comparison - BDeu vs. penalised likelihood.
"""

from __future__ import annotations

import csv
import os
import tempfile
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

LagKey = Tuple[int, int]  # (src_label_idx, lag)

# Lazy import guard
try:
    import pyagrum as gum

    _PYAGRUM_AVAILABLE = True
except ImportError:
    _PYAGRUM_AVAILABLE = False


def _require_pyagrum(fn: str) -> None:
    if not _PYAGRUM_AVAILABLE:
        raise ImportError(f"{fn}() requires pyAgrum. Install: pip install pyAgrum")


# ---------------------------------------------------------------------------
# Discretisation helpers
# ---------------------------------------------------------------------------


def _discretize_column(col: np.ndarray, n_bins: int) -> np.ndarray:
    """Bin a 1-D array of non-negative counts into `n_bins` integer labels."""
    if col.max() == col.min():
        return np.zeros(len(col), dtype=int)
    edges = np.linspace(col.min(), col.max() + 1, n_bins + 1)
    return np.clip(np.digitize(col, edges) - 1, 0, n_bins - 1)


def build_lagged_table(
    X_k: np.ndarray,
    label_names: List[str],
    max_lag: int = 2,
    n_bins: int = 5,
) -> Tuple[str, List[str], List[str]]:
    """
    Build a discretised CSV file with current and lagged columns.

    Parameters
    ----------
    X_k : (T, L) count array.
    label_names : L label names.
    max_lag : maximum lag to include.
    n_bins : number of bins for discretisation.

    Returns
    -------
    csv_path : path to a temporary CSV file (caller must delete).
    current_names : column names for current-time variables.
    lagged_names  : column names for lagged variables  (`X_{label}_lag{d}`).
    """
    T, L = X_k.shape
    current_names = [f"X_{label_names[ell]}" for ell in range(L)]
    lagged_names: List[str] = []

    # Discretise current columns
    cols = {}
    for ell in range(L):
        cols[current_names[ell]] = _discretize_column(X_k[:, ell], n_bins)

    # Discretise lagged columns
    for d in range(1, max_lag + 1):
        for ell in range(L):
            lag_name = f"X_{label_names[ell]}_lag{d}"
            lagged_names.append(lag_name)
            lagged = np.zeros(T, dtype=float)
            lagged[d:] = X_k[:-d, ell] if d < T else 0.0
            cols[lag_name] = _discretize_column(lagged, n_bins)

    # Trim first max_lag rows (no valid lagged data)
    start = max_lag
    all_names = current_names + lagged_names
    tmpf = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline=""
    )
    writer = csv.writer(tmpf)
    writer.writerow(all_names)
    for t in range(start, T):
        writer.writerow([int(cols[n][t]) for n in all_names])
    tmpf.close()
    return tmpf.name, current_names, lagged_names


# ---------------------------------------------------------------------------
# Scoring class
# ---------------------------------------------------------------------------


class PyAgrumScore:
    """
    pyAgrum-based score evaluator for structure search.

    Wraps `pyAgrum.BNLearner.score(target, parents)` after discretising
    the time-series count data into a CSV table with lagged columns.
    """

    def __init__(
        self,
        X_k: np.ndarray,
        label_names: List[str],
        max_lag: int = 2,
        n_bins: int = 5,
        score_type: str = "BDeu",
    ) -> None:
        _require_pyagrum("PyAgrumScore")
        self.label_names = label_names
        self.max_lag = max_lag
        self.L = len(label_names)

        csv_path, self._current, self._lagged = build_lagged_table(
            X_k, label_names, max_lag, n_bins
        )
        self._learner = gum.BNLearner(csv_path)
        os.unlink(csv_path)  # BNLearner loads data immediately

        score_setters = {
            "BDeu": self._learner.useScoreBDeu,
            "BIC": self._learner.useScoreBIC,
            "AIC": self._learner.useScoreAIC,
            "K2": self._learner.useScoreK2,
            "BD": self._learner.useScoreBD,
        }
        setter = score_setters.get(score_type)
        if setter is None:
            raise ValueError(
                f"Unknown score_type={score_type!r}. "
                f"Choose from {list(score_setters)}"
            )
        setter()

    def _lag_col_name(self, src_ell: int, lag: int) -> str:
        return f"X_{self.label_names[src_ell]}_lag{lag}"

    def _target_col_name(self, ell: int) -> str:
        return f"X_{self.label_names[ell]}"

    def score_parent_set(
        self,
        ell: int,
        parent_set: Set[LagKey],
    ) -> float:
        """
        Compute the pyAgrum score for label `ell` given `parent_set`.

        Parameters
        ----------
        ell : target label index.
        parent_set : set of `(src_ell, lag)` tuples.

        Returns
        -------
        score : float (higher is better, consistent with LocalScore).
        """
        target = self._target_col_name(ell)
        parents = [self._lag_col_name(s, d) for s, d in parent_set]
        return self._learner.score(target, parents)

    def learn_initial_structure(
        self,
        max_indegree: int = 3,
    ) -> Dict[int, Set[LagKey]]:
        """
        Use pyAgrum's greedy hill-climbing to learn an initial graph.

        Only lagged-to-current arcs are kept (same-time arcs are discarded
        to match N²DBN constraints).

        Returns
        -------
        parent_sets : `{ell: {(src_ell, lag), ...}}` compatible with
            `ScaleParams.parent_sets`.
        """
        import re

        self._learner.useGreedyHillClimbing()
        self._learner.setMaxIndegree(max_indegree)

        bn = self._learner.learnBN()

        lag_pattern = re.compile(r"^X_(.+)_lag(\d+)$")
        cur_pattern = re.compile(r"^X_(.+)$")
        name_to_idx = {n: i for i, n in enumerate(self.label_names)}

        parent_sets: Dict[int, Set[LagKey]] = {ell: set() for ell in range(self.L)}

        for arc in bn.arcs():
            tail = bn.variable(arc[0]).name()
            head = bn.variable(arc[1]).name()
            tail_m = lag_pattern.match(tail)
            head_m = cur_pattern.match(head)
            if tail_m and head_m:
                src_label = tail_m.group(1)
                lag = int(tail_m.group(2))
                dst_label = head_m.group(1)
                if src_label in name_to_idx and dst_label in name_to_idx:
                    parent_sets[name_to_idx[dst_label]].add(
                        (name_to_idx[src_label], lag)
                    )

        return parent_sets
