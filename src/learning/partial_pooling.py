"""
learning/partial_pooling.py
===========================
The participant axis as a drawn level, with the spread of the draw estimated.

A cohort shares one graph and one set of coefficients and differs in how much of
each label every participant exercises.  `subject_offset` codes that difference
as S levels under a row-weighted sum-to-zero constraint and fits them freely.
Fitted freely they carry the whole of a short record, and a week scale holding
nine epochs per participant then has as many level numbers as it has data.

Here the levels are draws from `N(0, sigma^2)` and the fit is penalized by the
prior precision.  A participant with few rows is pulled toward the cohort and a
participant with many keeps its own level, which is what lets a shared relation
borrow strength across the roster.  The pooled intercept stays free, since the
constraint already fixes where the levels are centered and a prior on the
intercept would move the population level toward zero.

The spread is estimated from the record.  Holding the coefficients, the EM
update for the variance of a normal random effect is the mean of the squared
levels plus the mean posterior variance of them, and iterating it to a fixed
point is what selects how much pooling the record supports.  A record that
separates its participants sharply drives the variance up and the fit back to
the free one; a record that does not drives it down and the levels toward zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

_RIDGE = 1e-10

#: Below this the levels are pooled, since a spread the record cannot separate
#: from zero is a spread that buys nothing and the update crawls toward it.
POOLED_FLOOR = 1e-8


@dataclass(frozen=True)
class LevelPrior:
    """
    The normal prior the subject deviations are drawn from.

    Attributes
    ----------
    variance : the spread of a level around the cohort, on the predictor's scale
    width    : columns the level block occupies, the intercept and S-1 contrasts
    """

    variance: float
    width: int

    def __post_init__(self) -> None:
        if self.variance <= 0.0:
            raise ValueError(f"a prior variance is positive; got {self.variance}")
        if self.width < 1:
            raise ValueError(
                f"a level block holds the intercept and the contrasts, so at "
                f"least one column; got {self.width}"
            )

    @property
    def n_levels(self) -> int:
        """The deviation contrasts, which is every level column but the intercept."""
        return self.width - 1

    def precision(self, size: int) -> np.ndarray:
        """The prior precision as a diagonal over the whole design, intercept free."""
        out = np.zeros(size)
        out[1 : self.width] = 1.0 / self.variance
        return out


def penalized_solve(
    A: np.ndarray, b: np.ndarray, prior: LevelPrior
) -> Tuple[np.ndarray, np.ndarray]:
    """
    The leveled normal equations under the prior, as the fit and its covariance.

    Args:
        A: (P, P) the summed curvature, the level block first.
        b: (P,) the summed score.

    Returns:
        The coefficients and the posterior covariance the variance update reads.

    Raises:
        ValueError: when the level block is wider than the design.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if prior.width > A.shape[0]:
        raise ValueError(
            f"a level block of {prior.width} columns does not fit a design of "
            f"{A.shape[0]}"
        )
    covariance = np.linalg.inv(
        A + np.diag(prior.precision(A.shape[0])) + _RIDGE * np.eye(A.shape[0])
    )
    return covariance @ b, covariance


def variance_update(
    beta: np.ndarray, covariance: np.ndarray, prior: LevelPrior
) -> float:
    """
    One EM step for the spread of the levels, holding everything else.

    The expected complete-data sum of squares is the fitted levels plus their
    posterior variance, and dividing by the number of free levels gives the
    maximizer.

    Raises:
        ValueError: when the block holds no free level to estimate a spread from.
    """
    if prior.n_levels < 1:
        raise ValueError(
            "a roster of one subject has no deviation to estimate a spread from"
        )
    keep = slice(1, prior.width)
    levels = np.asarray(beta, dtype=float)[keep]
    spread = float(np.trace(np.asarray(covariance, dtype=float)[keep, keep]))
    return float((levels @ levels + spread) / prior.n_levels)


def fit_levels(
    A: np.ndarray,
    b: np.ndarray,
    width: int,
    variance: float = 1.0,
    n_iter: int = 200,
    tol: float = 1e-6,
    floor: float = POOLED_FLOOR,
) -> Tuple[np.ndarray, float, int]:
    """
    The coefficients and the spread the record supports, by alternating the two.

    The tolerance is relative, since a spread is a scale and an absolute step
    means different things at 1e-4 and at 10.

    Args:
        width: columns the level block occupies, the intercept and S-1 contrasts.
        variance: where the spread starts, which the iteration moves off.
        n_iter: the most alternations taken before the value in hand is returned.
        floor: the spread at which the levels are declared pooled.

    Returns:
        The coefficients, the fitted spread and the alternations taken.
    """
    current = float(variance)
    beta = np.zeros(np.asarray(A).shape[0])
    taken = 0
    for taken in range(1, n_iter + 1):
        prior = LevelPrior(current, width)
        beta, covariance = penalized_solve(A, b, prior)
        moved = variance_update(beta, covariance, prior)
        settled = abs(moved - current) <= tol * current
        current = max(moved, floor)
        if settled or current <= floor:
            break
    return beta, current, taken


def shrinkage(A: np.ndarray, width: int, variance: float) -> np.ndarray:
    """
    How far each level is pulled toward the cohort, one number per contrast in [0, 1].

    A contrast whose own curvature dominates the prior precision keeps its free
    value and comes back near zero; one the record barely informs comes back
    near one.
    """
    own = np.diag(np.asarray(A, dtype=float))[1:width]
    return 1.0 / (1.0 + variance * own)
