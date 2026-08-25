"""
learning/coarse_order.py
========================
Coarse association orders the fine search, and drops nothing.

A lagged relation whose lag is shorter than a coarse epoch does not vanish under
aggregation.  It reappears as same-epoch co-movement at that coarser scale, so
the coarse association matrix is a cheap prior over where fine lagged edges
live.  The prior decides the order candidates are visited in and never which
ones exist: at full depth the accepted set is the set an unordered greedy search
reaches, so depth is a speed setting and never a recall risk.

The coarse statistic is a same-epoch correlation, which is symmetric, so both
directions of a label pair share one priority and the fine gain is what
separates them.  It is measured within participants and with the calendar
columns partialled out, so the ordering does not simply rank shared weekday
rhythm.

One failure mode is stated here.  A relation whose sign alternates inside a
coarse block cancels under aggregation and sinks to the bottom of the ordering.
It is still reached at full depth, which is the whole reason this is an
ordering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: How several coarse scales become one priority.  `max` lets a relation visible
#: at any coarse scale rise, which is what the measured rank agreement supports.
MERGE_RULES = ("max", "mean")

_RIDGE = 1e-8


def within_participant_center(
    values: np.ndarray, who: np.ndarray, weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """Each column with its own participant's weighted mean removed."""
    values = np.asarray(values, dtype=float)
    who = np.asarray(who, dtype=np.int64)
    n_units = int(who.max(initial=-1)) + 1
    w = np.ones(values.shape[0]) if weights is None else np.asarray(weights, float)
    mass = np.bincount(who, weights=w, minlength=n_units)
    mass = np.where(mass > 0.0, mass, 1.0)
    out = np.empty_like(values)
    for j in range(values.shape[1]):
        total = np.bincount(who, weights=w * values[:, j], minlength=n_units)
        out[:, j] = values[:, j] - (total / mass)[who]
    # The square root, so a cross-product of two centered columns weights each
    # epoch once and the correlation below is the weighted one.
    return out * np.sqrt(w)[:, None]


def partial_out(values: np.ndarray, context: np.ndarray) -> np.ndarray:
    """The part of each column no combination of the context columns explains."""
    values = np.asarray(values, dtype=float)
    context = np.asarray(context, dtype=float)
    if context.shape[1] == 0:
        return values.copy()
    gram = context.T @ context
    gram = gram + _RIDGE * np.eye(gram.shape[0]) * max(float(np.trace(gram)), 1.0)
    coef = np.linalg.solve(gram, context.T @ values)
    return values - context @ coef


def _correlate(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Column-by-column correlation of two residual matrices."""
    norm_a = np.sqrt(np.sum(a * a, axis=0))
    norm_b = np.sqrt(np.sum(b * b, axis=0))
    norm_a = np.where(norm_a > 0.0, norm_a, np.inf)
    norm_b = np.where(norm_b > 0.0, norm_b, np.inf)
    return (a.T @ b) / np.outer(norm_a, norm_b)


def association(
    sources: np.ndarray,
    targets: np.ndarray,
    context: np.ndarray,
    who: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Same-epoch association of every source with every target at one coarse scale.

    Args:
        weights: per-epoch weight, which the observation mask supplies and a
            coarse regime's responsibility supplies for `regime_tensor`.

    Returns:
        The (n_sources, n_targets) matrix of absolute residual correlations.
    """
    src = within_participant_center(sources, who, weights)
    dst = within_participant_center(targets, who, weights)
    ctx = within_participant_center(context, who, weights)
    return np.abs(_correlate(partial_out(src, ctx), partial_out(dst, ctx)))


@dataclass(frozen=True)
class AssociationTensor:
    """One association matrix per coarse scale, and the merge that reads them."""

    per_scale: Dict[int, np.ndarray]
    source_names: Tuple[str, ...]
    target_names: Tuple[str, ...]

    def merged(self, fine_scale: int, rule: str = "max") -> np.ndarray:
        """One priority per (source, target) for a search running at one fine scale."""
        if rule not in MERGE_RULES:
            raise ValueError(
                f"merge rule {rule!r}; expected one of {list(MERGE_RULES)}"
            )
        above = [m for k, m in sorted(self.per_scale.items()) if k > fine_scale]
        if not above:
            raise ValueError(
                f"scale {fine_scale} has no coarser scale in this tensor to read"
            )
        stacked = np.stack(above)
        return stacked.max(axis=0) if rule == "max" else stacked.mean(axis=0)


def association_tensor(
    counts: Sequence[np.ndarray],
    exogenous: Sequence[np.ndarray],
    participants: Sequence[np.ndarray],
    observed: Sequence[np.ndarray],
    source_names: Sequence[str],
    target_names: Sequence[str],
    n_context: int,
    scales: Optional[Sequence[int]] = None,
) -> AssociationTensor:
    """
    The coarse association of every ordered pair, at every scale asked for.

    The sources are the target labels followed by the exogenous columns beyond
    the first `n_context`, which are the ones a search may read as a parent.  The
    first `n_context` exogenous columns are the calendar, and every column is
    partialled out on them.
    """
    wanted = list(scales) if scales is not None else list(range(2, len(counts) + 1))
    per_scale: Dict[int, np.ndarray] = {}
    for k in wanted:
        block = np.asarray(exogenous[k - 1], dtype=float)
        src = np.concatenate(
            [np.asarray(counts[k - 1], dtype=float), block[:, n_context:]], axis=1
        )
        per_scale[k] = association(
            sources=src,
            targets=np.asarray(counts[k - 1], dtype=float),
            context=block[:, :n_context],
            who=participants[k - 1],
            weights=np.asarray(observed[k - 1], dtype=float),
        )
    return AssociationTensor(
        per_scale=per_scale,
        source_names=tuple(source_names),
        target_names=tuple(target_names),
    )


def regime_tensor(
    counts: Sequence[np.ndarray],
    exogenous: Sequence[np.ndarray],
    participants: Sequence[np.ndarray],
    observed: Sequence[np.ndarray],
    responsibilities: np.ndarray,
    source_names: Sequence[str],
    target_names: Sequence[str],
    n_context: int,
    scale: int,
) -> AssociationTensor:
    """
    One association matrix per coarse regime, merged the same way scales are.

    A relation active only inside one regime carries its whole signal there, so
    the `max` merge lifts it and the search visits it first inside that regime's
    fine epochs.
    """
    gamma = np.asarray(responsibilities, dtype=float)
    mask = np.asarray(observed[scale - 1], dtype=float)
    per_regime: Dict[int, np.ndarray] = {}
    block = np.asarray(exogenous[scale - 1], dtype=float)
    src = np.concatenate(
        [np.asarray(counts[scale - 1], dtype=float), block[:, n_context:]], axis=1
    )
    for r in range(gamma.shape[1]):
        per_regime[r + 1] = association(
            sources=src,
            targets=np.asarray(counts[scale - 1], dtype=float),
            context=block[:, :n_context],
            who=participants[scale - 1],
            weights=mask * gamma[:, r],
        )
    return AssociationTensor(
        per_scale=per_regime,
        source_names=tuple(source_names),
        target_names=tuple(target_names),
    )


# ---------------------------------------------------------------------------
# The ordering, and the search that reads it
# ---------------------------------------------------------------------------


def lagged(values: np.ndarray, participants: Sequence[int], lag: int = 1) -> np.ndarray:
    """
    Each column stepped back by a lag inside its own participant block.

    An epoch before its participant's record opens reads zero, which is the
    convention every lagged feature in the project already uses.
    """
    values = np.asarray(values, dtype=float)
    who = np.asarray(participants, dtype=np.int64)
    out = np.zeros_like(values)
    for unit in np.unique(who):
        rows = np.nonzero(who == unit)[0]
        if rows.size > lag:
            out[rows[lag:]] = values[rows[:-lag]] if lag else values[rows]
    return out


def drop_self(
    priority: np.ndarray, source_names: Sequence[str], target_names: Sequence[str]
) -> np.ndarray:
    """
    The priority with every self pair removed.

    A label's same-epoch correlation with itself is one by construction, so a
    self relation would sit at the top of every ordering and say nothing.  Self
    relations carry their own stratum and their own price.
    """
    out = np.array(priority, dtype=float, copy=True)
    for i, name in enumerate(source_names):
        for j, other in enumerate(target_names):
            if name == other:
                out[i, j] = 0.0
    return out


def visit_order(priorities: Sequence[float]) -> List[int]:
    """Candidate indices from the highest priority down, ties by index."""
    values = np.asarray(priorities, dtype=float)
    return sorted(range(values.size), key=lambda i: (-values[i], i))


@dataclass
class OrderedReport:
    """What an ordered search accepted, and how much of the space it reached."""

    accepted: List[int] = field(default_factory=list)
    gains: List[float] = field(default_factory=list)
    rejected: List[int] = field(default_factory=list)
    trace: List[float] = field(default_factory=list)
    depth: int = 0
    n_candidates: int = 0
    evaluations: int = 0
    exhausted: bool = False

    @property
    def share_visited(self) -> float:
        return self.depth / self.n_candidates if self.n_candidates else 0.0


def ordered_search(
    gain_fn: Callable[[List[int]], Sequence[float]],
    apply_fn: Callable[[List[int]], float],
    priorities: Sequence[float],
    price: float,
    max_edges: int = 8,
    depth: Optional[int] = None,
    evaluations: Optional[int] = None,
) -> OrderedReport:
    """
    Greedy search that visits candidates in coarse-priority order.

    Args:
        depth: how many of the highest-priority candidates each round may look
            at; None looks at all of them, which is the exact run.
        evaluations: a ceiling on candidate evaluations across all rounds; None
            is no ceiling.

    Returns:
        The accepted set, the depth actually reached and whether a ceiling
        stopped the run.  At full depth and with no evaluation ceiling this is
        the set an unordered greedy search reaches.
    """
    values = np.asarray(priorities, dtype=float)
    n = int(values.size)
    reach = n if depth is None else min(int(depth), n)
    order = visit_order(values)

    report = OrderedReport(depth=reach, n_candidates=n)
    selected: List[int] = []
    current = apply_fn(selected)
    report.trace.append(current)

    while len(selected) < max_edges:
        visible = [i for i in order[:reach] if i not in selected]
        if not visible:
            break
        if evaluations is not None and report.evaluations + len(visible) > evaluations:
            report.exhausted = True
            break
        report.evaluations += len(visible)
        gains = np.array(gain_fn(selected), dtype=float, copy=True)
        scored = np.full(n, -np.inf)
        scored[visible] = gains[visible]
        best = int(np.argmax(scored))
        if not np.isfinite(scored[best]) or scored[best] <= price:
            break
        proposed = selected + [best]
        realized = apply_fn(proposed)
        if realized < current:
            report.rejected.append(best)
            break
        selected = proposed
        current = realized
        report.accepted.append(best)
        report.gains.append(float(scored[best]))
        report.trace.append(current)
    return report


def recall_against_depth(
    priorities: Sequence[float],
    wanted: Sequence[int],
    depths: Optional[Sequence[int]] = None,
) -> List[Tuple[int, float, int]]:
    """
    How much of a wanted set the ordering reaches at each depth.

    Returns:
        One row per depth: the depth, the share of the space it covers and how
        many of the wanted candidates sit inside it.
    """
    values = np.asarray(priorities, dtype=float)
    n = int(values.size)
    order = visit_order(values)
    grid = list(depths) if depths is not None else [n]
    target = set(int(i) for i in wanted)
    rows: List[Tuple[int, float, int]] = []
    for d in grid:
        reach = min(int(d), n)
        rows.append((reach, reach / n if n else 0.0, len(target & set(order[:reach]))))
    return rows
