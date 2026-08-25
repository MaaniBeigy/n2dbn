"""
src/evaluation/sortnregress.py
==============================
The baseline that recovers a graph by sorting the variables and regressing.

Reisach et al. (2021) showed that a benchmark built from an additive noise model
with independently drawn parameters often orders its variables by marginal
variance along the causal order, and that a baseline exploiting only that order
matches the continuous learners it is compared against.  A benchmark on which
this baseline scores well grades a method on a regularity of the simulation.
The scale-free companion sorts by the share of a variable's variance the others
explain, which standardizing does not move.

`evaluation.metrics.varsortability` and `r2_sortability` report how strong the
regularity is.  What those two cannot say is whether it is strong enough to be
exploited, and that is what this module measures: the same metric code that
scores every other method scores these three rows, and their distance from the
random ordering is how much of a result the ordering alone can buy.

Two decisions the temporal setting forces, stated because they change the number.

*A predecessor is read at a lag.*  The static baseline regresses a variable on
its predecessors in the same observation.  Here a parent acts through a lag, so
each target is regressed on the lagged columns of the labels ahead of it in the
order, and the recovered edge carries the lag its coefficient sits at.

*A label keeps its own past.*  The order cannot place a label before itself, so
a self relation is unreachable by sorting alone.  The published learners of the
same comparison are all given their own history, and withholding it here would
report a handicap where the question is what the ordering is worth, so the
target's own lags are always offered.  Recall is split by edge kind wherever this is reported.
"""

from __future__ import annotations

from typing import Optional, Sequence, Set, Tuple

import numpy as np
from sklearn.linear_model import LassoLarsIC, LinearRegression

#: An edge as the rest of the project keys it: source label, lag, target label.
EdgeKey = Tuple[int, int, int]


def r2_coefficients(X: np.ndarray) -> np.ndarray:
    """
    The share of each column's variance the other columns explain.

    Read off the inverse correlation matrix, which is the closed form, and by
    one regression per column when that matrix is singular.

    Args:
        X: (T, L) series, one column per label.

    Returns:
        (L,) coefficients of determination.
    """
    X = np.asarray(X, dtype=float)
    if X.shape[1] < 2:
        return np.zeros(X.shape[1])
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.corrcoef(X, rowvar=False)
    if not np.all(np.isfinite(corr)):
        return _r2_by_regression(X)
    try:
        return 1.0 - 1.0 / np.diag(np.linalg.inv(corr))
    except np.linalg.LinAlgError:
        return _r2_by_regression(X)


def _r2_by_regression(X: np.ndarray) -> np.ndarray:
    """One regression per column, for a correlation matrix that cannot be inverted."""
    L = X.shape[1]
    out = np.zeros(L)
    model = LinearRegression()
    for j in range(L):
        others = np.arange(L) != j
        if X[:, j].std() == 0.0:
            continue
        model.fit(X[:, others], X[:, j])
        out[j] = float(model.score(X[:, others], X[:, j]))
    return out


def _lagged(X: np.ndarray, columns: Sequence[int], max_lag: int) -> np.ndarray:
    """The design of `columns` at lags 1 to `max_lag`, rows aligned to the target."""
    T = X.shape[0]
    blocks = [X[max_lag - d : T - d, columns] for d in range(1, max_lag + 1)]
    return np.column_stack(blocks) if blocks else np.zeros((T - max_lag, 0))


def sort_regress(X: np.ndarray, scores: np.ndarray, max_lag: int = 1) -> Set[EdgeKey]:
    """
    Order the labels by `scores`, then regress each on the lags of its predecessors.

    The weighted Lasso of the reference implementation is kept: an unpenalized
    fit supplies per-column weights, the Lasso runs on the weighted design under
    its own information criterion, and a surviving coefficient is an edge.

    Args:
        X: (T, L) series, one column per label.
        scores: (L,) the quantity the order is read off, increasing.
        max_lag: how many lags of each predecessor are offered.

    Returns:
        The selected edges as (source, lag, target).

    Raises:
        ValueError: `scores` does not carry one entry per column of `X`.
    """
    X = np.asarray(X, dtype=float)
    scores = np.asarray(scores, dtype=float).ravel()
    L = X.shape[1]
    if scores.size != L:
        raise ValueError(
            f"scores carries {scores.size} entries for {L} labels; one each is needed"
        )
    if X.shape[0] <= max_lag:
        return set()

    order = list(np.argsort(scores, kind="stable"))
    edges: Set[EdgeKey] = set()
    for position, target in enumerate(order):
        # The target's own past is always offered; see the module docstring.
        sources = order[:position] + [int(target)]
        design = _lagged(X, sources, max_lag)
        response = X[max_lag:, target]
        if design.shape[1] == 0 or response.std() == 0.0:
            continue
        weights = np.abs(LinearRegression().fit(design, response).coef_)
        if not np.any(weights > 0.0):
            continue
        lasso = LassoLarsIC(criterion="bic").fit(design * weights, response)
        coef = lasso.coef_ * weights
        for column, value in enumerate(coef):
            if value != 0.0:
                lag, index = divmod(column, len(sources))
                edges.add((int(sources[index]), lag + 1, int(target)))
    return edges


def var_sort_regress(X: np.ndarray, max_lag: int = 1) -> Set[EdgeKey]:
    """Sort the labels by marginal variance, then regress."""
    return sort_regress(X, np.var(np.asarray(X, dtype=float), axis=0), max_lag)


def r2_sort_regress(X: np.ndarray, max_lag: int = 1) -> Set[EdgeKey]:
    """Sort the labels by the variance the others explain, then regress."""
    return sort_regress(X, r2_coefficients(X), max_lag)


def random_sort_regress(
    X: np.ndarray, max_lag: int = 1, rng: Optional[np.random.Generator] = None
) -> Set[EdgeKey]:
    """Sort the labels at random, then regress, which is the floor the other two beat."""
    rng = rng or np.random.default_rng(0)
    return sort_regress(X, rng.permutation(np.asarray(X).shape[1]), max_lag)
