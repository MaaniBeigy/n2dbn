"""
learning/fine_mstep.py
======================
The parameter step of the fine likelihood, as one weighted normal equation.

Under the quadratic form the start term is concave in the coefficients, so the
step is a solve.  The allocation formulation needed a guarded gradient ascent
with a learning rate, because a fixed number of Adam steps can overshoot; here
the maximizer is closed form and cannot.

Two things make the solve correct as well as closed form.

The posterior variance is in it.  The objective carries `E_q[eta^2]`, which is
`(E_q eta)^2 + Var_q(eta)`, so a design built from posterior means alone drops
`sum_t w_t V_t` and under-penalizes a large coefficient on a column the posterior
is unsure about.  A step that drops it can lower the very bound it is maximizing,
which is measured.

And a scale's columns are centered inside their parent block.  The broadcast
bases of two scales are nested, so a day-scale column is reproducible from the
hour-scale basis and the likelihood is flat along transfers between them.
Centering a scale-k column within its scale-(k+1) parent makes the two
orthogonal, so an effect is attributed to the resolution it actually varies at.
Without the constraint the fit keeps whichever resolution it reached first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..inference.bound import pg_mean, segment_sum

_RIDGE = 1e-8
# The share of the largest diagonal a column must carry to be solved for at all.
# Below this its curvature has underflowed and the ridge, not the data, would
# decide its coefficient.
_INFORMED = 1e-12
# The reciprocal condition number the remaining block must clear.  Below this the
# solve keeps six of sixteen digits, so what it returns is arithmetic.
_CONDITIONED = 1e-10


def solve_informed(
    A: np.ndarray, b: np.ndarray, ridge: float = _RIDGE
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    The solve, with columns carrying no information held at zero.

    `A` is a weighted Gram and `b` an unweighted score, so a column whose weight
    has underflowed leaves its diagonal at zero while its entry of `b` stays
    finite.  The ridge alone then returns that coefficient as `b / ridge`, which
    is 1e8 times a count, and the predictor it builds saturates every slot the
    column touches.  Such a column carries no evidence about its own
    coefficient, so the coefficient it gets is zero.

    Refusing the column rather than the whole target is deliberate: one dead
    column should not freeze the coefficients every other column of that target
    has evidence for.  What is refused wholesale is a *remaining* block that is
    still ill-conditioned once the dead columns are out, because there no
    subset of the columns can be trusted separately.

    Returns the vector, the mask of columns that were solved for, and whether
    the block cleared its conditioning floor.  A caller that gets `False` should
    keep the coefficients it already had; a caller that gets a `False` in the
    mask can say which column it refused rather than reporting the zero as a
    fit.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    diagonal = np.diag(A)
    scale = float(diagonal.max()) if diagonal.size else 0.0
    informed = diagonal > _INFORMED * max(scale, 1.0)
    beta = np.zeros(b.shape[0])
    if not informed.any():
        return beta, informed, False
    block = A[np.ix_(informed, informed)]
    spectrum = np.linalg.eigvalsh(block)
    if spectrum[0] <= _CONDITIONED * max(float(spectrum[-1]), 1.0):
        return beta, informed, False
    beta[informed] = np.linalg.solve(
        block + ridge * np.eye(block.shape[0]), b[informed]
    )
    return beta, informed, True


def block_center(
    columns: np.ndarray, parent_ancestors: np.ndarray, n_parent_epochs: int
) -> np.ndarray:
    """
    Columns with the mean over each parent block removed.

    This is the constraint that keeps the resolutions apart.  It is applied to
    the design, so nothing downstream has to
    remember it.
    """
    columns = np.asarray(columns, dtype=float)
    sizes = segment_sum(
        np.ones((columns.shape[0], 1)), parent_ancestors, n_parent_epochs
    )
    totals = segment_sum(columns, parent_ancestors, n_parent_epochs)
    means = totals / np.maximum(sizes, 1.0)
    return columns - means[parent_ancestors]


@dataclass
class DesignBlock:
    """
    One group of columns of the predictor, with the posterior's own uncertainty.

    `covariance` is the posterior covariance of the columns at each slot, shape
    (T, P, P), and is None when the columns are known.  A calendar node and a
    lagged count are known; a regime indicator and a factor are inferred.
    """

    name: str
    columns: np.ndarray
    covariance: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.columns = np.asarray(self.columns, dtype=float)
        if self.columns.ndim != 2:
            raise ValueError(f"{self.name!r} columns are {self.columns.shape}, not 2-D")
        if self.covariance is not None:
            self.covariance = np.asarray(self.covariance, dtype=float)
            expected = (self.columns.shape[0], self.width, self.width)
            if self.covariance.shape != expected:
                raise ValueError(
                    f"{self.name!r} covariance is {self.covariance.shape}, "
                    f"expected {expected}"
                )

    @property
    def width(self) -> int:
        return int(self.columns.shape[1])


def stack_design(blocks: Sequence[DesignBlock]) -> Tuple[np.ndarray, np.ndarray]:
    """
    The blocks as one mean design and one block-diagonal posterior covariance.

    Cross-block covariance is not represented, which is the same independence the
    variational family already assumes between a regime chain and a factor.
    """
    if not blocks:
        raise ValueError("a design needs at least one block")
    means = np.concatenate([b.columns for b in blocks], axis=1)
    T, P = means.shape
    cov = np.zeros((T, P, P))
    start = 0
    for block in blocks:
        stop = start + block.width
        if block.covariance is not None:
            cov[:, start:stop, start:stop] = block.covariance
        start = stop
    return means, cov


def normal_equations(
    x: np.ndarray,
    blocks: Sequence[DesignBlock],
    weights: np.ndarray,
    at_risk: Optional[np.ndarray] = None,
    ridge: float = _RIDGE,
    with_variance: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    The `A` and `b` of the weighted solve, with the variance term unless refused.

    `with_variance=False` exists to measure what dropping it costs and is not a
    mode anything should fit in.
    """
    means, cov = stack_design(blocks)
    x = (np.asarray(x) > 0).astype(float)
    w = np.asarray(weights, dtype=float)
    keep = np.ones(x.shape[0], dtype=bool) if at_risk is None else np.asarray(at_risk)

    m = means[keep]
    wk = w[keep]
    b = m.T @ (x[keep] - 0.5)
    A = m.T @ (m * wk[:, None])
    if with_variance:
        A = A + np.einsum("t,tpq->pq", wk, cov[keep])
    return A + ridge * np.eye(A.shape[0]), b


def solve_coefficients(
    x: np.ndarray,
    blocks: Sequence[DesignBlock],
    weights: np.ndarray,
    at_risk: Optional[np.ndarray] = None,
    ridge: float = _RIDGE,
    with_variance: bool = True,
) -> np.ndarray:
    """The coefficient vector that maximizes the quadratic at a fixed anchor."""
    A, b = normal_equations(x, blocks, weights, at_risk, ridge, with_variance)
    return np.linalg.solve(A, b)


def quadratic_value(
    x: np.ndarray,
    blocks: Sequence[DesignBlock],
    beta: np.ndarray,
    weights: np.ndarray,
    at_risk: Optional[np.ndarray] = None,
) -> float:
    """
    The part of the bound the step maximizes, at a fixed anchor and posterior.

    The anchor's own constant is left out, since it does not depend on the
    coefficients and every comparison here is between coefficient vectors.
    """
    means, cov = stack_design(blocks)
    x = (np.asarray(x) > 0).astype(float)
    w = np.asarray(weights, dtype=float)
    keep = np.ones(x.shape[0], dtype=bool) if at_risk is None else np.asarray(at_risk)
    beta = np.asarray(beta, dtype=float)

    eta = means[keep] @ beta
    var = np.einsum("p,tpq,q->t", beta, cov[keep], beta)
    return float(np.sum((x[keep] - 0.5) * eta - 0.5 * w[keep] * (eta**2 + var)))


def bound_value(
    x: np.ndarray,
    blocks: Sequence[DesignBlock],
    beta: np.ndarray,
    anchor: np.ndarray,
    at_risk: Optional[np.ndarray] = None,
) -> float:
    """
    The whole bound at a given anchor, constant included.

    `quadratic_value` compares two coefficient vectors under one anchor and drops
    the anchor's own constant because it cancels.  Across a re-anchoring it does
    not cancel, and comparing anchor-free values across anchors compares two
    different functions, which reads as the objective falling when it has not.
    This is the quantity that is monotone under both moves.
    """
    anchor = np.abs(np.asarray(anchor, dtype=float))
    weights = pg_mean(anchor)
    keep = (
        np.ones(anchor.shape[0], dtype=bool) if at_risk is None else np.asarray(at_risk)
    )
    const = -np.logaddexp(0.0, anchor) + anchor / 2.0 + weights * anchor**2 / 2.0
    return quadratic_value(x, blocks, beta, weights, at_risk) + float(const[keep].sum())


def anchor_from(
    blocks: Sequence[DesignBlock], beta: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """The predictor's mean and the anchor its second moment implies."""
    means, cov = stack_design(blocks)
    beta = np.asarray(beta, dtype=float)
    eta = means @ beta
    var = np.einsum("p,tpq,q->t", beta, cov, beta)
    return eta, np.sqrt(np.maximum(eta**2 + var, 0.0))


def update_block(
    x: np.ndarray,
    blocks: Sequence[DesignBlock],
    beta: np.ndarray,
    which: int,
    weights: np.ndarray,
    at_risk: Optional[np.ndarray] = None,
    ridge: float = _RIDGE,
) -> np.ndarray:
    """
    One block's coefficients, maximized with the others held where they are.

    Coordinate ascent on a concave quadratic, so the value cannot fall whichever
    block is chosen and whatever order the blocks are visited in.
    """
    beta = np.asarray(beta, dtype=float).copy()
    widths = [b.width for b in blocks]
    starts = np.cumsum([0] + widths)
    lo, hi = int(starts[which]), int(starts[which + 1])

    means, cov = stack_design(blocks)
    x = (np.asarray(x) > 0).astype(float)
    w = np.asarray(weights, dtype=float)
    keep = np.ones(x.shape[0], dtype=bool) if at_risk is None else np.asarray(at_risk)

    other = np.ones(beta.size, dtype=bool)
    other[lo:hi] = False
    rest = means[keep][:, other] @ beta[other]

    m = means[keep][:, lo:hi]
    wk = w[keep]
    A = m.T @ (m * wk[:, None]) + np.einsum("t,tpq->pq", wk, cov[keep][:, lo:hi, lo:hi])
    A = A + ridge * np.eye(hi - lo)
    b = m.T @ (x[keep] - 0.5 - wk * rest)
    beta[lo:hi] = np.linalg.solve(A, b)
    return beta


def regime_block(
    gamma: np.ndarray, ancestors: np.ndarray, name: str = "regime"
) -> DesignBlock:
    """
    A regime chain's columns and their posterior covariance, at the fine grid.

    The first state is the reference, so a chain of M states contributes M-1
    columns and the intercept absorbs the rest.  The covariance is the
    multinomial one the posterior implies, `diag(g) - g g'`, which is what a
    design of posterior means alone would throw away.
    """
    gamma = np.asarray(gamma, dtype=float)
    fine = gamma[np.asarray(ancestors)][:, 1:]
    cov = np.einsum("tp,pq->tpq", fine, np.eye(fine.shape[1])) - np.einsum(
        "tp,tq->tpq", fine, fine
    )
    return DesignBlock(name=name, columns=fine, covariance=cov)


def ascend(
    x: np.ndarray,
    blocks: Sequence[DesignBlock],
    beta: np.ndarray,
    order: Sequence[int],
    at_risk: Optional[np.ndarray] = None,
    ridge: float = _RIDGE,
) -> Tuple[np.ndarray, List[float]]:
    """
    One sweep of coordinate ascent in the given order, re-anchoring as it goes.

    The order is an argument because it does not matter: each step maximizes the
    same bound over one factor with the rest fixed, so ascent is non-decreasing
    in any sequential order, and up, down or interleaved are all schedules of one
    guarantee.
    """
    beta = np.asarray(beta, dtype=float).copy()
    trace: List[float] = []
    for which in order:
        # Re-anchor at the current coefficients, which is where the bound is
        # tightest, then maximize over one block with that anchor held.
        _, anchor = anchor_from(blocks, beta)
        beta = update_block(x, blocks, beta, which, pg_mean(anchor), at_risk, ridge)
        _, anchor = anchor_from(blocks, beta)
        trace.append(bound_value(x, blocks, beta, anchor, at_risk))
    return beta, trace
