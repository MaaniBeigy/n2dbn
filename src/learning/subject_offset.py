"""
learning/subject_offset.py
==========================
Per-subject variation as a level, coded on the design and fitted by streaming.

A subject differs from the cohort in how much of a label it exercises.  Every
persona split the benchmark declares is a split in which labels a subject
exercises at all, and every declared nonstationarity is calendar locked and
reaches the design as a named calendar column at its home scale.  The
participant axis therefore needs one number per subject per label per quantity,
a level, and it needs nothing that varies in time.

The level reaches the likelihood through the channel a regime chain already
uses.  `FineParams.offset` holds an (S, L) array per quantity and one row of it
goes to `linear_predictor` as its `offsets` argument, where a (L,) row
broadcasts against the (T_1, L) tile of `base`.  The likelihood gains no term.

No chain runs here.  A cohort-wide switch would assert a state the manifest
never declares, and on a roster sharing one exact calendar it would be near
collinear with the calendar block, which leaves the ridge to decide the split.
A chain of M states holds M distinct levels per label, so expressing S subjects
takes M = S states and an absorbing transition matrix per subject, which is a
level written the long way.

Estimation widens the design's first block from the intercept to the participant
axis.  Column 0 stays the pooled intercept and columns 1 to S-1 carry the
row-weighted deviation contrasts `1_r - (n_r / n_last) 1_last`, which put the
constraint `sum_s n_s a_s = 0` on the design itself.  S levels then cost S
numbers together with the intercept and the solve has one maximizer.  S+1
columns for S levels leave the population intercept free to move by any constant
as long as every subject moves back by the same one.  The constraint lives in
the coding, so nothing downstream remembers it, which is the stance
`fine_mstep.block_center` takes for the resolution constraint.

The row counts `n_{s,l}` are the rows subject s brings to target l under the
declared lag budget, so they are fixed by the data and the candidate space and
never by the graph in hand.  Counts read off the graph would move every time an
edge landed, and the weights of the constraint would then be a function of the
model being scored.

Absorption is prevented by geometry the design already applies.
`fine_design.center_at_scale` removes a parent-epoch mean from every column, and
the parent of the coarsest scale is the whole horizon, so every column has
weighted mean zero over each subject's own record and its inner product with any
per-subject constant is zero under the counting measure of the declared rows.
The level is constant over the subject's whole record, so it has no shape a
trend could occupy.  A level that varied by epoch would be exactly collinear
with that subject's own coarsest-scale column, and `LevelCoding` refuses to hold
one.

The level block is never laid down inside the fit.  Its columns are constant
down one subject's whole record, so every entry they contribute to the normal
equations is a row total times that constant, and `level_normal_equations` forms
them from the totals.  Materialized, the block would widen the (T, P, P)
posterior covariance `fine_mstep.stack_design` allocates from a handful of
columns to S of them, which is a gigabyte per target at the benchmark's shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..core.cohort import Cohort
from ..core.typed_graph import COUNT, TypedEdge, TypedGraph
from ..inference.bound import pg_update
from ..model.intensity import (
    FineParams,
    linear_predictor,
    start_loglik,
    usable_slots,
)
from .fine_design import CalendarIndex, FineDesign, ScaleLadder, build_design
from .fine_mstep import _RIDGE, DesignBlock, solve_informed, stack_design
from .fine_sem import fit_rows

# ---------------------------------------------------------------------------
# The coding
# ---------------------------------------------------------------------------


def subject_deviations(n_rows: np.ndarray) -> np.ndarray:
    """
    The constant row every subject's slots carry across the level block.

    Column 0 is the pooled intercept and column 1 + r is the deviation contrast
    of subject r against the last subject on the roster, weighted by the rows
    the two of them bring.  Read down, row s of the result is what every slot of
    subject s carries, so `row_s @ beta` is the level in force there.

    Args:
        n_rows: (S,) the rows each subject contributes to one target under the
            declared lag budget.

    Returns:
        (S, S), the intercept column and the S-1 deviation columns.

    Raises:
        ValueError: when the counts are not one positive number per subject.
    """
    counts = np.asarray(n_rows, dtype=float)
    if counts.ndim != 1 or counts.size == 0:
        raise ValueError(
            f"a level coding carries one row count per subject, so a non-empty "
            f"1-D array; this one is {counts.shape}.  A count per subject per "
            "epoch would give a subject one level per epoch, which is exactly "
            "collinear with that subject's own coarsest-scale column"
        )
    empty = np.flatnonzero(counts <= 0.0)
    if empty.size:
        raise ValueError(
            f"subject {int(empty[0])} brings {counts[empty[0]]:g} rows to this "
            "target, so no row of the data carries its level and the constraint "
            "would weight a number nothing identifies"
        )
    size = counts.size
    out = np.zeros((size, size))
    out[:, 0] = 1.0
    out[: size - 1, 1:] = np.eye(size - 1)
    out[size - 1, 1:] = -counts[: size - 1] / counts[size - 1]
    return out


def subject_offsets(beta_level: np.ndarray, n_rows: np.ndarray) -> np.ndarray:
    """
    The (S,) level deviation of every subject, the last one read off the constraint.

    The deviation columns are what carry the offsets, so this is the level block
    applied to the fitted contrasts with the intercept column left out.

    Raises:
        ValueError: when the coefficients and the row counts disagree on S.
    """
    coding = subject_deviations(n_rows)
    beta = np.asarray(beta_level, dtype=float)
    if beta.shape != (coding.shape[0],):
        raise ValueError(
            f"a level block over {coding.shape[0]} subjects carries "
            f"{coding.shape[0]} coefficients; got {beta.shape}"
        )
    return coding[:, 1:] @ beta[1:]


@dataclass(frozen=True)
class LevelCoding:
    """
    Which subject one design is for, and the rows every subject brings to its target.

    Attributes
    ----------
    subject : the roster position of the subject this design is built for
    rows    : (S,) the rows each subject contributes to this one target
    """

    subject: int
    rows: np.ndarray

    def __post_init__(self) -> None:
        """
        Raises:
            ValueError: when the row counts are malformed, or the roster position
                names no subject among them.
        """
        subject_deviations(self.rows)
        if not 0 <= int(self.subject) < self.S:
            raise ValueError(
                f"subject {int(self.subject)} is not one of the {self.S} the row "
                "counts name"
            )

    @property
    def S(self) -> int:
        return int(np.asarray(self.rows).shape[0])

    @property
    def row(self) -> np.ndarray:
        """What every slot of this subject carries across the level block."""
        return subject_deviations(self.rows)[int(self.subject)]


# ---------------------------------------------------------------------------
# The widened design
# ---------------------------------------------------------------------------


def level_blocks(design: FineDesign, coding: LevelCoding) -> List[DesignBlock]:
    """
    One target's blocks with the intercept replaced by the participant axis.

    This is the design the widened solve is a solve of, laid down in full.  The
    fit reaches the same `A` and `b` through `level_normal_equations`, which
    never forms the level columns.
    """
    return [
        DesignBlock(
            "level", np.broadcast_to(coding.row, (design.ladder.n_slots, coding.S))
        )
    ] + list(design.blocks[1:])


def level_normal_equations(
    x: np.ndarray,
    blocks: Sequence[DesignBlock],
    weights: np.ndarray,
    level: np.ndarray,
    at_risk: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    One subject's `A` and `b` for a leveled design, with the level block unformed.

    Every level column is constant over this subject's whole record, so what it
    contributes is a row total times that constant, and both the (T, S) block and
    the (T, P, P) covariance it would widen stay unallocated.  The ridge is left
    off, since it belongs to the pooled sum and is charged there once.

    Args:
        x: (T_1,) the target's counts, read as a start indicator.
        blocks: every block but the level one, in the order they stack after it.
        weights: (T_1,) the quadratic's per-slot weights.
        level: (S,) the constant this subject's slots carry.
        at_risk: (T_1,) bool, the rows the term sums over.

    Returns:
        `A` and `b` of width S + P, both additive over subjects.
    """
    constant = np.asarray(level, dtype=float)
    starts = (np.asarray(x) > 0).astype(float)
    w = np.asarray(weights, dtype=float)
    keep = (
        np.ones(starts.shape[0], dtype=bool) if at_risk is None else np.asarray(at_risk)
    )
    if blocks:
        means, cov = stack_design(blocks)
        m, c = means[keep], cov[keep]
    else:
        m = np.zeros((int(keep.sum()), 0))
        c = np.zeros((m.shape[0], 0, 0))

    wk = w[keep]
    residual = starts[keep] - 0.5
    width, other = constant.size, m.shape[1]
    A = np.zeros((width + other, width + other))
    A[:width, :width] = float(wk.sum()) * np.outer(constant, constant)
    A[:width, width:] = np.outer(constant, m.T @ wk)
    A[width:, :width] = A[:width, width:].T
    A[width:, width:] = m.T @ (m * wk[:, None]) + np.einsum("t,tpq->pq", wk, c)
    b = np.concatenate([constant * float(residual.sum()), m.T @ residual])
    return A, b


# ---------------------------------------------------------------------------
# The streamed fit
# ---------------------------------------------------------------------------


def level_rows(cohort: Cohort, lag_budget: Mapping[int, int]) -> np.ndarray:
    """
    (S, L) the rows every subject brings to every target under the declared budget.

    One pass over the risk sets before any design is built, because the coding of
    one subject's level block is a statement about the whole cohort.
    """
    budget = dict(lag_budget)
    out = np.zeros((cohort.S, cohort.L), dtype=np.int64)
    for index, (_name, data) in enumerate(cohort.stream()):
        sources = cohort.sources(data)
        rows = fit_rows(data.observed, budget, sources)
        out[index] = usable_slots(data.get_X(1), data.get_Y(1), rows).sum(axis=0)
    return out


@dataclass
class LevelSweep:
    """One pass over the cohort, as the summed solve and the value it ran at."""

    A: Dict[int, np.ndarray] = field(default_factory=dict)
    b: Dict[int, np.ndarray] = field(default_factory=dict)
    edges: Dict[int, Tuple[TypedEdge, ...]] = field(default_factory=dict)
    loglik: float = 0.0

    def add(
        self,
        target: int,
        A: np.ndarray,
        b: np.ndarray,
        edges: Tuple[TypedEdge, ...],
    ) -> None:
        """
        Add one subject's contribution to one target.

        Raises:
            ValueError: when two subjects came back under different parent sets,
                since the summed matrix would then hold one subject's column in
                the position of another's.
        """
        if target not in self.A:
            self.A[target], self.b[target], self.edges[target] = A, b, edges
            return
        if self.edges[target] != edges:
            raise ValueError(
                f"target {target} came back over {len(edges)} edges from one "
                f"subject and {len(self.edges[target])} from another; the cohort "
                "holds one graph and its columns sit in one order"
            )
        self.A[target] += A
        self.b[target] += b


class SubjectLevelFit:
    """
    One graph over a cohort, with each subject's level free and no chain.

    The cohort shares the graph, one coefficient per edge and one intercept per
    label.  What a subject holds of its own is the level, one number per label,
    and the roster's levels are tied by the constraint the coding carries.

    Attributes
    ----------
    graph      : the one typed graph every subject is scored under
    params     : the coefficients, the pooled intercepts and the offsets
    lag_budget : the largest lag the search may propose at each source scale,
                 declared before the first subject is opened
    targets    : the target labels a design is formed for
    """

    def __init__(
        self,
        graph: TypedGraph,
        params: FineParams,
        lag_budget: Mapping[int, int],
        targets: Optional[Sequence[int]] = None,
        index: Optional[CalendarIndex] = None,
        max_sweeps: int = 8,
        tol: float = 1e-7,
    ) -> None:
        self.graph = graph
        self.params = params
        self.lag_budget = dict(lag_budget)
        self.targets = list(range(graph.L) if targets is None else targets)
        self.index = index if index is not None else CalendarIndex(placed={}, widths={})
        self.max_sweeps = int(max_sweeps)
        self.tol = float(tol)

    # ------------------------------------------------------------------
    def accumulate(self, cohort: Cohort, rows: np.ndarray) -> LevelSweep:
        """
        One stream over the cohort, adding the normal equations of every target.

        Args:
            cohort: the roster and the reader the subjects arrive through.
            rows: (S, L) the declared row counts the level coding is weighted by.

        Returns:
            The summed matrices and the exact start log-likelihood of the targets
            at the coefficients the pass ran at.

        Raises:
            ValueError: when the parameters carry a different roster size than
                the cohort.
        """
        if self.params.n_subjects != cohort.S:
            raise ValueError(
                f"the parameters hold {self.params.n_subjects} offset rows for a "
                f"cohort of {cohort.S}, so one row would stand for two subjects"
            )
        sweep = LevelSweep()
        for index, (_name, data) in enumerate(cohort.stream()):
            X1, Y1 = data.get_X(1), data.get_Y(1)
            sources = cohort.sources(data)
            at_risk = usable_slots(
                X1, Y1, fit_rows(data.observed, self.lag_budget, sources)
            )
            ladder = ScaleLadder.from_sources(sources, data.observed)
            # The subject's own level travels the offsets argument, which is the
            # channel a regime chain already uses, so the weights this pass forms
            # are the weights at that subject's level.
            raw = linear_predictor(
                self.params,
                self.graph,
                sources,
                COUNT,
                self.params.offset[COUNT][index],
            )
            for ell in self.targets:
                design = build_design(
                    self.graph,
                    ell,
                    COUNT,
                    sources,
                    ladder,
                    self.index,
                    budget=self.lag_budget,
                )
                # Centering took a parent mean out of every edge column, so the
                # predictor the design fits reaches the likelihood through the
                # correction `FineDesign.offset` holds and the anchor sits where
                # the solve will read it.
                held = np.array(
                    [
                        self.params.coef[(ell, COUNT)].get(c.edge, 0.0)
                        for c in design.columns
                    ]
                )
                eta = raw[:, ell] + design.offset(np.concatenate([[0.0], held]))
                sweep.loglik += start_loglik(X1[:, ell], eta, at_risk[:, ell])
                coding = LevelCoding(index, rows[:, ell])
                A, b = level_normal_equations(
                    X1[:, ell],
                    design.blocks[1:],
                    pg_update(eta),
                    coding.row,
                    at_risk[:, ell],
                )
                sweep.add(ell, A, b, tuple(c.edge for c in design.columns))
        return sweep

    def solve(self, sweep: LevelSweep, rows: np.ndarray) -> Dict[int, np.ndarray]:
        """
        One coefficient vector per target, with the ridge charged once.

        Charged per subject the ridge would grow with the roster and shrink the
        very coefficients a larger cohort carries more evidence for.

        A subject with no at-risk row for this label leaves an empty column, and
        the same asymmetry the pooled solve has -- weighted `A`, unweighted `b`
        -- would hand it `b * 1e8` as its offset.  Held at zero it sits at the
        population level, which is what no evidence of its own means.  A target
        whose block is still ill-conditioned once those are out keeps the levels
        it came in with.
        """
        width = self.params.n_subjects
        out: Dict[int, np.ndarray] = {}
        for target in sorted(sweep.A):
            A = sweep.A[target]
            beta, _, conditioned = solve_informed(A, sweep.b[target], _RIDGE)
            if not conditioned:
                continue
            self.params.base[COUNT][target] = float(beta[0])
            self.params.offset[COUNT][:, target] = subject_offsets(
                beta[:width], rows[:, target]
            )
            for j, edge in enumerate(sweep.edges[target]):
                self.params.coef[(target, COUNT)][edge] = float(beta[width + j])
            out[target] = beta
        return out

    def fit(
        self, cohort: Cohort, rows: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Alternate the pooled solve and the reweighting until the fit settles.

        Each solve maximizes a tangent lower bound on the start log-likelihood
        that touches it at the coefficients the pass ran at, so the exact
        log-likelihood recorded here cannot fall.

        Returns:
            The declared row counts the coding was weighted by, and the exact
            start log-likelihood of the targets before every solve.
        """
        counts = (
            level_rows(cohort, self.lag_budget) if rows is None else np.asarray(rows)
        )
        history: List[float] = []
        for _ in range(self.max_sweeps):
            sweep = self.accumulate(cohort, counts)
            history.append(sweep.loglik)
            if len(history) > 1 and abs(history[-1] - history[-2]) < self.tol * (
                1.0 + abs(history[-2])
            ):
                break
            self.solve(sweep, counts)
        return counts, history
