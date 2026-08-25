"""
learning/screens.py
===================
Four ways to visit fewer candidates, none of which changes what is accepted.

A sweep at the benchmark's candidate size scores on the order of a hundred
thousand columns per target, and the four screens here decide which of them are
worth opening.  Each one records what it set aside, so a screened run and an
unscreened run can be compared on the accepted set and any difference between
them is attributable.

**A Bayes factor at the posterior mean.**  Holding the latent path at
`E_q[U]`, a candidate's score statistic and its curvature give the log Bayes
factor of adding it in closed form, so no refit is needed to rank candidates
against a stated prior on the coefficient.

**A pooled pre-score with an escape hatch.**  A relation that shows at several
resolutions is worth opening first, and the pooled score orders the field.  A
relation that shows at one resolution alone would be lost to a pooled ordering,
so any candidate clearing the bar at its own scale survives the pre-score
whatever the pooled value says.

**A bound no superset can beat.**  The quadratic's maximum over any set of
columns is at most the value it reaches with the response fully explained, which
is a weighted total of the residual and costs one pass.  A target whose bound
sits under the cheapest purchase has nothing left to buy and closes.  The bound
is loose by construction; what it is is sound.

**A cache over the sweep.**  A parent set reached twice inside one sweep has one
value, so it is computed once and keyed on the scale, the target and the set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, List, Sequence, Tuple

import numpy as np

_TINY = 1e-12


def log_bayes_factor(
    gains: np.ndarray, curvature: np.ndarray, prior_variance: float = 1.0
) -> np.ndarray:
    """
    The log Bayes factor of adding each candidate, at the latent posterior mean.

    Under a normal prior of the given variance on the coefficient and the
    quadratic form the search already uses, the evidence ratio is the candidate's
    gain less half the log of how much the data sharpen the prior.

    Args:
        gains: the rank-one gain of each candidate, in nats.
        curvature: the Schur complement each gain was formed against.

    Raises:
        ValueError: when the prior variance is not positive, or the two arrays
            disagree on how many candidates there are.
    """
    if prior_variance <= 0.0:
        raise ValueError(f"a prior variance is positive; got {prior_variance}")
    gains = np.asarray(gains, dtype=float)
    curvature = np.maximum(np.asarray(curvature, dtype=float), 0.0)
    if gains.shape != curvature.shape:
        raise ValueError(
            f"{gains.shape} gains meet {curvature.shape} curvatures, so the two "
            "do not describe one candidate field"
        )
    return gains - 0.5 * np.log1p(prior_variance * curvature)


@dataclass
class ScreenReport:
    """What one screen set aside, held so a screened run can be audited."""

    name: str
    considered: int = 0
    kept: int = 0

    @property
    def dropped(self) -> int:
        return self.considered - self.kept

    @property
    def share_kept(self) -> float:
        return self.kept / self.considered if self.considered else 1.0


def screen_by_bayes_factor(
    gains: np.ndarray,
    curvature: np.ndarray,
    prior_variance: float = 1.0,
    threshold: float = 0.0,
) -> Tuple[np.ndarray, ScreenReport]:
    """The candidates whose log Bayes factor clears the threshold, and the count."""
    values = log_bayes_factor(gains, curvature, prior_variance)
    keep = values >= threshold
    return keep, ScreenReport("bayes_factor", int(keep.size), int(keep.sum()))


def pooled_pre_score(
    per_scale: Sequence[np.ndarray], own_scale: np.ndarray, bar: float
) -> Tuple[np.ndarray, ScreenReport]:
    """
    Candidates ordered by their pooled score, with a single-scale relation kept.

    Args:
        per_scale: one score array per scale, all over the same candidate field.
        own_scale: each candidate's score at the resolution it sits at, which is
            the escape hatch a relation visible at one scale alone leaves by.
        bar: the score a candidate needs, pooled or at its own scale.

    Raises:
        ValueError: when the per-scale arrays do not describe one field.
    """
    if not per_scale:
        raise ValueError("a pooled pre-score needs at least one scale to pool over")
    stacked = np.stack([np.asarray(s, dtype=float) for s in per_scale], axis=0)
    own = np.asarray(own_scale, dtype=float)
    if own.shape != stacked.shape[1:]:
        raise ValueError(
            f"the own-scale scores are {own.shape} where the pooled field is "
            f"{stacked.shape[1:]}"
        )
    keep = (stacked.mean(axis=0) >= bar) | (own >= bar)
    return keep, ScreenReport("pooled_pre_score", int(keep.size), int(keep.sum()))


def saturated_bound(residual: np.ndarray, weights: np.ndarray) -> float:
    """
    The most any set of columns can raise the quadratic, which no superset beats.

    The maximum over a column set is the response projected onto that set, and a
    projection never exceeds the response itself, so the weighted total of the
    residual bounds every set at once.

    Raises:
        ValueError: when a weight is negative, since the bound is then not one.
    """
    residual = np.asarray(residual, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if np.any(weights < 0.0):
        raise ValueError("a quadratic's weights are non-negative; one of these is not")
    return float(0.5 * np.sum(residual**2 / np.maximum(weights, _TINY)))


def closes(bound: float, cheapest_price: float) -> bool:
    """Whether nothing a target could still buy would clear its cheapest price."""
    return bound <= cheapest_price


@dataclass
class SweepCache:
    """One sweep's values, keyed on the scale, the target and the parent set."""

    values: Dict[Tuple[int, int, str, FrozenSet[int]], float] = field(
        default_factory=dict
    )
    hits: int = 0
    misses: int = 0

    def get(
        self,
        scale: int,
        target: int,
        quantity: str,
        parents: Sequence[int],
        compute: Callable[[], float],
    ) -> float:
        """The value of one parent set, computed once inside a sweep."""
        key = (scale, target, quantity, frozenset(parents))
        if key in self.values:
            self.hits += 1
            return self.values[key]
        self.misses += 1
        value = float(compute())
        self.values[key] = value
        return value

    def clear(self) -> None:
        """Drop everything, which is what the end of a sweep does."""
        self.values.clear()
        self.hits = 0
        self.misses = 0

    @property
    def share_hit(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


def screen_agreement(screened: Sequence[int], unscreened: Sequence[int]) -> List[int]:
    """What an unscreened run accepted and a screened run did not, in sorted order."""
    return sorted(set(unscreened) - set(screened))
