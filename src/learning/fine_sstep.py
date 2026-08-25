"""
learning/fine_sstep.py
======================
The structure step at full candidate size.

Once an edge is typed by source scale, source quantity, lag and basis term, the
candidate set is on the order of a hundred thousand.  That is
the intended size: a benchmark whose point is that hundreds of relations are
discoverable cannot have a candidate set of a few.  Three things keep a sweep of
that size affordable and honest.

**A coarse feature lives on a coarse grid.**  A day-scale column over twelve
weeks takes 84 distinct values, not 120960, so the cross-product of two
day-scale candidates is an 84 by 84 reduction against pre-aggregated weights
and never a pass over the finest grid.  In general a pair reduces to the finer
of their two home grids, through the same segment sum the upward message uses.
Only pairs that both vary at the finest resolution cost finest-grid work.

**The score is a rank-one update.**  Under the quadratic form, adding a
candidate to a fitted base set has a closed-form gain, so every candidate is
scored in one matrix computation, where a refit each would be the cost.  The
gain opens each candidate against the base columns and each candidate against
itself, so `cross_and_diagonal` forms those two blocks alone.  The full Gram of
base and candidates together also carries a candidate-by-candidate block of
C (C - 1) / 2 entries, which is 778128 at the benchmark's 1248 candidates per
target and which nothing reads.

**The search keeps its guard.**  A structure move changes the predictor, so it
changes the optimal anchor and the optimal posterior, and it is therefore not a
coordinate-ascent step on the fixed bound.  The monotonicity that covers the
parameter step does not reach here, and this is the one place the project's old
guard is kept.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.typed_graph import LOG_TERM, BasisTerm, basis_label, read_basis
from ..inference.bound import segment_sum

_RIDGE = 1e-8
# The share of its own information a candidate must keep once the fitted base is
# projected out.  Below this the Schur complement is the difference of two
# nearly equal sums plus whatever the ridge in `BaseFit.inverse` leaked into it,
# so its value is arithmetic rather than evidence.
_RETAINED = 1e-8


@dataclass(frozen=True)
class ScaledColumn:
    """One candidate feature, held at the resolution it varies at."""

    name: str
    scale: int
    values: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", np.asarray(self.values, dtype=float))
        if self.values.ndim != 1:
            raise ValueError(f"{self.name!r} is {self.values.shape}, not one column")

    def at_finest(self, ancestors: Sequence[np.ndarray]) -> np.ndarray:
        return self.values[np.asarray(ancestors[self.scale - 1])]


def weights_by_scale(
    weights: np.ndarray, ancestors: Sequence[np.ndarray], n_epochs: Sequence[int]
) -> List[np.ndarray]:
    """The per-slot weights summed into every scale, which the reduction needs."""
    out = []
    for k, count in enumerate(n_epochs):
        out.append(segment_sum(np.asarray(weights, dtype=float), ancestors[k], count))
    return out


def _pair_product(
    a: ScaledColumn,
    b: ScaledColumn,
    ancestors: Sequence[np.ndarray],
    per_scale_weights: Sequence[np.ndarray],
) -> float:
    """
    One weighted cross-product, computed at the finer of the two home grids.

    The coarser column is constant across a finer epoch, so it is read once per
    finer epoch, and the weights are already summed to
    that grid.
    """
    fine, coarse = (a, b) if a.scale <= b.scale else (b, a)
    k = fine.scale
    grid = np.arange(fine.values.size)
    if coarse.scale == k:
        lifted = coarse.values
    else:
        # Which coarse epoch each fine epoch belongs to, read off the first slot
        # of that fine epoch, which the nesting makes well defined.
        first_slot = _first_slot(ancestors[k - 1], fine.values.size)
        lifted = coarse.values[np.asarray(ancestors[coarse.scale - 1])[first_slot]]
    return float(np.sum(per_scale_weights[k - 1][grid] * fine.values * lifted))


def _first_slot(ancestor: np.ndarray, n_epochs: int) -> np.ndarray:
    """
    The first finest slot of each epoch, which a non-decreasing map gives.

    An epoch nothing maps to keeps the zero it came in with, which is the value
    a scan over the map leaves there too.
    """
    ancestor = np.asarray(ancestor)
    out = np.zeros(n_epochs, dtype=np.int64)
    # Written back to front, so where several slots share an epoch the earliest
    # of them lands last and stays.  A scan forward over the map keeps the first
    # slot it sees and this keeps the same one, over a map of any shape.
    out[ancestor[::-1]] = np.arange(ancestor.size - 1, -1, -1, dtype=np.int64)
    return out


def resolution_aware_gram(
    columns: Sequence[ScaledColumn],
    ancestors: Sequence[np.ndarray],
    per_scale_weights: Sequence[np.ndarray],
) -> np.ndarray:
    """The weighted Gram matrix, every entry computed at its own resolution."""
    n = len(columns)
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            value = _pair_product(columns[i], columns[j], ancestors, per_scale_weights)
            out[i, j] = out[j, i] = value
    return out


def cross_and_diagonal(
    candidates: Sequence[ScaledColumn],
    base: Sequence[ScaledColumn],
    ancestors: Sequence[np.ndarray],
    per_scale_weights: Sequence[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    The two blocks of the Gram a rank-one gain reads, each entry at its own resolution.

    Every entry is the same weighted cross-product `resolution_aware_gram` puts
    at that position, so the pair of blocks is that matrix restricted to what is
    read.  The candidate-by-candidate block is quadratic in the candidate count
    and no gain opens it, so at 1248 candidates against a base of 9 this is 12480
    products where the full matrix is 790,653.

    Returns:
        (C, P), each candidate against each base column, and (C,), each
        candidate against itself.
    """
    cross = np.zeros((len(candidates), len(base)))
    diagonal = np.zeros(len(candidates))
    for i, candidate in enumerate(candidates):
        for j, column in enumerate(base):
            cross[i, j] = _pair_product(candidate, column, ancestors, per_scale_weights)
        diagonal[i] = _pair_product(candidate, candidate, ancestors, per_scale_weights)
    return cross, diagonal


def naive_gram(
    columns: Sequence[ScaledColumn],
    ancestors: Sequence[np.ndarray],
    weights: np.ndarray,
) -> np.ndarray:
    """The same matrix computed on the finest grid, which the reduction must match."""
    fine = np.stack([c.at_finest(ancestors) for c in columns], axis=1)
    w = np.asarray(weights, dtype=float)
    return fine.T @ (fine * w[:, None])


def reduction_cost(columns: Sequence[ScaledColumn], n_slots: int) -> Tuple[int, int]:
    """
    Terms the resolution-aware reduction touches, against the finest-grid count.

    Reported, since the whole argument for the sweep being
    affordable is this ratio and a silent regression in it is a silent regression
    in what the search can consider.
    """
    n = len(columns)
    aware = 0
    for i in range(n):
        for j in range(i, n):
            aware += min(columns[i].values.size, columns[j].values.size)
    return aware, n * (n + 1) // 2 * n_slots


# ---------------------------------------------------------------------------
# Scoring a candidate against a fitted base set
# ---------------------------------------------------------------------------


@dataclass
class BaseFit:
    """A fitted base set, held as what a rank-one update needs."""

    A: np.ndarray
    b: np.ndarray
    inverse: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.A = np.asarray(self.A, dtype=float)
        self.b = np.asarray(self.b, dtype=float)
        self.inverse = np.linalg.inv(self.A + _RIDGE * np.eye(self.A.shape[0]))

    @property
    def value(self) -> float:
        """The quadratic's maximum over the base coefficients."""
        return float(0.5 * self.b @ self.inverse @ self.b)


def candidate_gains(
    base: BaseFit,
    cross: np.ndarray,
    diagonal: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """
    The gain of adding each candidate, in one computation.

    `cross` is (C, P), the weighted product of each candidate with each base
    column; `diagonal` is (C,), each candidate with itself; `targets` is (C,),
    each candidate with the response.  The gain is the Schur complement form of
    the quadratic's maximum, so it is exact.

    What the number means is decided by the weights the three arrays were
    summed under.  Under the exact curvature of the start term and the
    Fisher-scoring response, which is what `learning.pooled` supplies, the
    residual is the candidate's own score at the fitted base and the Schur
    complement is that base's Fisher information orthogonalized against it, so
    the gain is the score statistic in nats and sits beside the likelihood ratio
    of a refit.  Under the Polya-Gamma curvature the same expression is the
    exact rise of the bound, which at a start rate of 1.45e-3 is one
    fifty-second of that.

    A candidate that keeps less than `_RETAINED` of its own information once the
    base is projected out scores zero.  Either it carries no weighted support at
    all, or the base already spans it, and in both cases the complement is
    arithmetic rather than evidence.  Flooring the complement instead divides a
    finite residual by the floor, which ranks exactly those candidates first --
    an empty finest-scale self column bought this way is what put a 1e10
    intercept into the 2026-08-23 `exp16` legs.
    """
    cross = np.atleast_2d(np.asarray(cross, dtype=float))
    diagonal = np.asarray(diagonal, dtype=float)
    targets = np.asarray(targets, dtype=float)

    projected = cross @ base.inverse
    residual = targets - projected @ base.b
    schur = diagonal - np.einsum("cp,cp->c", projected, cross)
    retained = schur > _RETAINED * np.maximum(diagonal, 1.0)
    gains = np.zeros(schur.shape[0])
    np.divide(0.5 * residual**2, schur, out=gains, where=retained)
    return gains


# ---------------------------------------------------------------------------
# The basis a candidate is expanded through
# ---------------------------------------------------------------------------


def quantile_edges(values: np.ndarray, n_bins: int = 4) -> np.ndarray:
    """
    Bin edges at the empirical quantiles of a training split.

    The same basis is applied to every candidate of a quantity, so no particular
    relation is anticipated.  A threshold is then expressible as the step between
    adjacent bin contrasts, and it is expressed well where a cut falls near it.
    Where the truth is monotone the log term carries it and the contrasts do not
    survive the price.

    A cut point that leaves the bin below it empty is dropped, so every bin of
    the edges that come back holds rows of its own.  An empty bin makes the bands
    above it sum to the intercept the design already carries, and where it leaves
    a single band that band is the intercept itself.  A price read at a quantile
    of the candidate count counts such a column as one the search could buy, so
    it is never offered as one.  A source whose quartiles all sit at its own floor
    therefore carries the log term alone.
    """
    if n_bins < 2:
        raise ValueError(f"a basis needs at least two bins; got {n_bins}")
    values = np.asarray(values, dtype=float)
    inner = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
    kept: List[float] = []
    lower = -np.inf
    for cut in np.unique(inner):
        if np.any((values >= lower) & (values < cut)):
            kept.append(float(cut))
            lower = float(cut)
    return np.concatenate([[-np.inf], kept, [np.inf]])


def basis_terms(edges: np.ndarray) -> List[BasisTerm]:
    """
    The log term and one band per bin above the reference, as the terms of a basis.

    A candidate enumeration reads this list and `basis_columns` builds one column
    per entry of it, so an edge the search proposes is an edge the design builds.
    """
    edges = np.asarray(edges, dtype=float)
    return [LOG_TERM] + [
        (float(edges[j]), float(edges[j + 1])) for j in range(1, len(edges) - 1)
    ]


def basis_columns(
    values: np.ndarray, edges: np.ndarray, name: str, scale: int
) -> List[ScaledColumn]:
    """The log term and the bin contrasts, with the first bin as the reference."""
    values = np.asarray(values, dtype=float)
    return [
        ScaledColumn(f"{name}:{basis_label(term)}", scale, read_basis(values, term))
        for term in basis_terms(edges)
    ]


def shifted_null(
    values: np.ndarray, rng: np.random.Generator, n: int
) -> List[np.ndarray]:
    """
    Shifted copies of a candidate's own column, which is the null it must beat.

    A rolled copy keeps the column's marginal distribution and its serial
    dependence and destroys only its alignment with the response, so what it
    measures is whether the alignment is worth anything.  It is a real column.
    The older sentinel was a fabricated edge.
    """
    values = np.asarray(values, dtype=float)
    size = values.size
    if size < 3:
        return [values.copy() for _ in range(n)]
    offsets = rng.choice(np.arange(1, size), size=min(n, size - 1), replace=False)
    return [np.roll(values, int(k)) for k in offsets]


# ---------------------------------------------------------------------------
# The search, with the guard the parameter step no longer needs
# ---------------------------------------------------------------------------


@dataclass
class SearchReport:
    """What a search accepted, and what its guard turned away."""

    accepted: List[int] = field(default_factory=list)
    gains: List[float] = field(default_factory=list)
    rejected: List[int] = field(default_factory=list)
    trace: List[float] = field(default_factory=list)


def greedy_search(
    gain_fn,
    apply_fn,
    n_candidates: int,
    price: float,
    max_edges: int = 8,
) -> SearchReport:
    """
    Add the best-scoring candidate while it clears the price, guarding each move.

    `gain_fn(selected)` returns a gain per candidate given what is already in;
    `apply_fn(selected)` returns the objective actually reached by that set.  The
    two differ because a structure move changes the anchor and the posterior, so
    the rank-one gain is a score and not a promise.  A move whose realized
    objective falls is rejected and the search stops, which is the guard the
    monotonicity proof of the parameter step does not cover.
    """
    report = SearchReport()
    selected: List[int] = []
    current = apply_fn(selected)
    report.trace.append(current)

    while len(selected) < max_edges:
        # A copy, because the caller's array is theirs and masking in place would
        # make an already-selected candidate unscoreable on every later round.
        gains = np.array(gain_fn(selected), dtype=float, copy=True)
        gains[selected] = -np.inf
        best = int(np.argmax(gains))
        if not np.isfinite(gains[best]) or gains[best] <= price:
            break
        proposed = selected + [best]
        realized = apply_fn(proposed)
        if realized < current:
            report.rejected.append(best)
            break
        selected = proposed
        current = realized
        report.accepted.append(best)
        report.gains.append(float(gains[best]))
        report.trace.append(current)
    return report
