"""
evaluation/falsification.py
===========================
The rungs of the observational arm's falsification ladder.

The observed log carries no interventions, so nothing here recovers a causal
graph.  What a rung produces is a stated attempt to kill an edge, together with
the denominator it was taken against and what it cannot see.  An edge that
survives every rung is an association that survived a battery, which is the only
thing this arm claims.

Eleven rungs live here and each is separable.

R1 counts the accepted edges that state a relation the study protocol forbids.
R2 injects a label matched on marginal, autocorrelation and participant
heterogeneity and connected to nothing, so every edge touching it is a false
positive by construction, and it is the only rung that measures that rate inside
the behavioural block.
R3 counts the resolution-stratified positives, with n printed beside the rate.
R4 adds a future value of an accepted parent to its own fit.  Temporal
precedence makes the exclusion restriction hold, so a nonzero coefficient
exposes a confounder slower than the lag; a null is no evidence and never no
confounding.
R5 is the shift placebo, repaired.  A roll by an arbitrary offset breaks the
diurnal alignment of a pair that co-moves only on the clock, and the pair then
beats its own null for a reason that has nothing to do with the relation.  A
roll locked to the calendar period keeps the rhythm and moves only the
alignment, and a roll blocked at participant boundaries never moves one
participant's epochs into another's.
R6 folds a fine array into its parent epochs and reads it against the coarse
array beside it.  A reconciled log makes a coarse total the sum of its children,
so the residual is zero there and a departure names the epoch it falls in.  A
statistic derived from those arrays carries no such guarantee, and its residual
is reported and left standing.
R7 is the invariance rung, and it is expected to return nothing.  A participant
effect is a direct edge into every target, so the participant is an invalid
environment; wave and day-index blocks are what remain.
R8 is the permutation baseline.  A permutation that moves whole blocks inside
each participant keeps every marginal and the short-range autocorrelation and
destroys the relations between series, so the score it leaves is what the search
reaches on no structure at all.
R9 is complementary-pairs stability selection with its bound on the expected
number of low-probability selections.
R10 asks how strong an unmeasured confounder would have to be to explain one
edge away.
R11 is the held-out likelihood.  It asks what the accepted structure buys on
slots the fit never read, per slot and per participant, and its sign-flip test is
taken over participants because that is the sampling unit.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

from ..data.anchors import NEGATIVE, POSITIVE, AnchorKey, AnchorSet
from ..inference.bound import segment_sum

DEFAULT_ALPHA = 0.05
_RIDGE = 1e-10


@dataclass(frozen=True)
class Rung:
    """One attempt to kill an edge, with the denominator it was taken against."""

    name: str
    hits: int
    denominator: int
    passed: bool
    note: str

    @property
    def rate(self) -> float:
        return self.hits / self.denominator if self.denominator else 0.0

    def __str__(self) -> str:
        return (
            f"{self.name}: {self.hits}/{self.denominator} ({self.rate:.3f}) "
            f"{'passes' if self.passed else 'FAILS'}; {self.note}"
        )


# ---------------------------------------------------------------------------
# R1 and R3, which read the frozen anchors
# ---------------------------------------------------------------------------


def protocol_negatives(accepted: Iterable[AnchorKey], anchors: AnchorSet) -> Rung:
    """R1: how many accepted edges state a relation the protocol forbids."""
    declared = anchors.of_sign(NEGATIVE)
    assertable = [a for a in declared if a.assertable]
    seen = set(accepted)
    hits = sum(1 for a in assertable if a.key in seen)
    structural = len(declared) - len(assertable)
    return Rung(
        name="R1 protocol negatives",
        hits=hits,
        denominator=len(assertable),
        passed=hits == 0,
        note=(
            f"{structural} of the {len(declared)} declared negatives are "
            "structural, since the axis cannot express an edge into the "
            "scheduled label, and they are not a test"
        ),
    )


def anchor_recall(accepted: Iterable[AnchorKey], anchors: AnchorSet) -> Rung:
    """R3: how many of the resolution-stratified positives the accepted set holds."""
    positives = [a for a in anchors.of_sign(POSITIVE) if a.assertable]
    seen = set(accepted)
    hits = sum(1 for a in positives if a.key in seen)
    n = len(positives)
    se = math.sqrt(0.25 / n) if n else 0.0
    return Rung(
        name="R3 anchor recall",
        hits=hits,
        denominator=n,
        passed=hits > 0,
        note=(
            f"n={n}, binomial standard error at one half is {se:.3f}; this is a "
            "gate with its denominator printed and never a performance number"
        ),
    )


# ---------------------------------------------------------------------------
# R2, the sham label
# ---------------------------------------------------------------------------


def lag1_autocorrelation(values: np.ndarray, who: np.ndarray) -> float:
    """The within-participant lag-1 autocorrelation of one series."""
    values = np.asarray(values, dtype=float)
    who = np.asarray(who, dtype=np.int64)
    num = den = 0.0
    for unit in np.unique(who):
        own = values[who == unit]
        if own.size < 3 or own.std() <= 0.0:
            continue
        centered = own - own.mean()
        num += float(centered[:-1] @ centered[1:])
        den += float(centered @ centered)
    return num / den if den > 0.0 else 0.0


def sham_series(
    reference: np.ndarray,
    who: np.ndarray,
    rng: np.random.Generator,
    autocorrelation: Optional[float] = None,
) -> np.ndarray:
    """
    A label matched on marginal, autocorrelation and heterogeneity, caused by nothing.

    Each participant's own values are re-ordered onto an autoregressive draw of
    the requested lag-1 correlation, so the participant keeps exactly the values
    it had, the persistence is reproduced and the ordering carries no relation to
    any other label.
    """
    reference = np.asarray(reference, dtype=float)
    who = np.asarray(who, dtype=np.int64)
    rho = (
        lag1_autocorrelation(reference, who)
        if autocorrelation is None
        else float(autocorrelation)
    )
    rho = float(np.clip(rho, -0.99, 0.99))
    out = np.empty_like(reference)
    for unit in np.unique(who):
        rows = np.nonzero(who == unit)[0]
        latent = np.empty(rows.size)
        latent[0] = rng.normal()
        for i in range(1, rows.size):
            latent[i] = rho * latent[i - 1] + math.sqrt(1.0 - rho * rho) * rng.normal()
        own = np.sort(reference[rows])
        out[rows] = own[np.argsort(np.argsort(latent))]
    return out


@dataclass(frozen=True)
class ShamMatch:
    """The three statistics the sham label is matched on, side by side."""

    marginal_distance: float
    autocorrelation: float
    autocorrelation_reference: float
    heterogeneity: float
    heterogeneity_reference: float

    def matched(self, tolerance: float = 0.1) -> bool:
        return (
            self.marginal_distance <= 1e-9
            and abs(self.autocorrelation - self.autocorrelation_reference) <= tolerance
            and abs(self.heterogeneity - self.heterogeneity_reference) <= 1e-9
        )


def sham_match(sham: np.ndarray, reference: np.ndarray, who: np.ndarray) -> ShamMatch:
    """How well the sham reproduces the label it was matched to."""
    sham = np.asarray(sham, dtype=float)
    reference = np.asarray(reference, dtype=float)
    who = np.asarray(who, dtype=np.int64)
    distance = float(np.abs(np.sort(sham) - np.sort(reference)).max(initial=0.0))
    units = np.unique(who)
    means_a = np.array([sham[who == u].mean() for u in units])
    means_b = np.array([reference[who == u].mean() for u in units])
    return ShamMatch(
        marginal_distance=distance,
        autocorrelation=lag1_autocorrelation(sham, who),
        autocorrelation_reference=lag1_autocorrelation(reference, who),
        heterogeneity=float(np.abs(means_a - means_b).max(initial=0.0)),
        heterogeneity_reference=0.0,
    )


def sham_rung(hits: int, candidates: int) -> Rung:
    """R2: every accepted edge touching the sham label is a false positive."""
    return Rung(
        name="R2 sham label",
        hits=int(hits),
        denominator=int(candidates),
        passed=hits == 0,
        note=(
            "the sham is connected to nothing by construction, so this is the "
            "false-positive rate inside the behavioural block"
        ),
    )


# ---------------------------------------------------------------------------
# R4, the future-lag negative control
# ---------------------------------------------------------------------------


def _least_squares(design: np.ndarray, response: np.ndarray):
    """Coefficients, their standard errors and the residual degrees of freedom."""
    design = np.asarray(design, dtype=float)
    response = np.asarray(response, dtype=float)
    n, d = design.shape
    gram = design.T @ design
    gram = gram + _RIDGE * np.eye(d) * max(float(np.trace(gram)), 1.0)
    inverse = np.linalg.inv(gram)
    beta = inverse @ (design.T @ response)
    residual = response - design @ beta
    dof = max(n - d, 1)
    sigma2 = float(residual @ residual) / dof
    se = np.sqrt(np.maximum(np.diag(inverse) * sigma2, 0.0))
    return beta, se, dof


@dataclass(frozen=True)
class FutureLag:
    """R4: what a future value of an accepted parent does inside its own fit."""

    coefficient: float
    standard_error: float
    p_value: float

    def passed(self, alpha: float = DEFAULT_ALPHA) -> bool:
        return self.p_value > alpha


def future_lag_control(
    response: np.ndarray,
    parent: np.ndarray,
    future: np.ndarray,
    context: Optional[np.ndarray] = None,
) -> FutureLag:
    """
    Add a future value of an accepted parent to its own fit and test its coefficient.

    Temporal precedence makes the exclusion restriction hold, so a coefficient
    away from zero says a confounder slower than the lag is driving both.
    """
    columns = [np.ones(len(response)), np.asarray(parent, dtype=float)]
    if context is not None:
        columns.extend(np.asarray(context, dtype=float).T)
    columns.append(np.asarray(future, dtype=float))
    design = np.column_stack(columns)
    beta, se, dof = _least_squares(design, response)
    t = beta[-1] / se[-1] if se[-1] > 0.0 else 0.0
    return FutureLag(
        coefficient=float(beta[-1]),
        standard_error=float(se[-1]),
        p_value=float(2.0 * stats.t.sf(abs(t), dof)),
    )


# ---------------------------------------------------------------------------
# R5, the repaired shift placebo
# ---------------------------------------------------------------------------


def period_shifts(
    n_periods: int, period: int, n_shifts: int, rng: np.random.Generator
) -> List[int]:
    """
    Offsets that are whole multiples of the calendar period and move every block.

    Args:
        n_periods: how many whole periods the shortest block holds, so an offset
            stays below it and no block is handed back its own arrangement.
    """
    if period < 1:
        raise ValueError(f"a calendar period is at least one epoch; got {period}")
    if n_periods < 2:
        return []
    draw = min(n_shifts, n_periods - 1)
    return [
        int(period * m)
        for m in rng.choice(np.arange(1, n_periods), size=draw, replace=False)
    ]


def roll_within_blocks(
    values: np.ndarray, blocks: Sequence[Tuple[int, int]], offset: int
) -> np.ndarray:
    """Roll each block by the same offset, so no block borrows another's epochs."""
    out = np.asarray(values, dtype=float).copy()
    for start, stop in blocks:
        span = stop - start
        if span > 0:
            out[start:stop] = np.roll(out[start:stop], offset % span)
    return out


def _abs_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """The absolute correlation of two columns, zero when either never moves."""
    if a.std() <= 0.0 or b.std() <= 0.0:
        return 0.0
    return abs(float(np.corrcoef(a, b)[0, 1]))


@dataclass(frozen=True)
class ShiftNull:
    """R5: the pair's own statistic against copies of itself moved in time."""

    statistic: float
    null: Tuple[float, ...]
    p_value: float
    n_shifts: int

    def beats_null(self, alpha: float = DEFAULT_ALPHA) -> bool:
        """Whether the alignment beats what the same columns give when moved."""
        return self.p_value <= alpha


def shift_null(
    response: np.ndarray,
    parent: np.ndarray,
    blocks: Sequence[Tuple[int, int]],
    period: int,
    n_shifts: int = 32,
    seed: int = 0,
) -> ShiftNull:
    """
    Score a pair against copies of the parent rolled inside each participant block.

    Args:
        blocks: the half-open epoch range of each participant.
        period: the calendar period of the scale, so a roll keeps the rhythm and
            moves only the alignment.  One epoch is the unrepaired null.

    Raises:
        ValueError: when a block span is not a whole number of periods, which
            would let the roll inside that block land off the rhythm.
    """
    for start, stop in blocks:
        if (stop - start) % period:
            raise ValueError(
                f"the block [{start}, {stop}) is not a whole number of "
                f"{period}-epoch periods, so a roll inside it moves the rhythm"
            )
    response = np.asarray(response, dtype=float)
    parent = np.asarray(parent, dtype=float)
    rng = np.random.default_rng(seed)
    stat = _abs_correlation(response, parent)
    shortest = min((stop - start) for start, stop in blocks)
    offsets = period_shifts(shortest // period, period, n_shifts, rng)
    null = []
    for offset in offsets:
        moved = roll_within_blocks(parent, blocks, offset)
        if moved.std() > 0.0:
            null.append(_abs_correlation(response, moved))
    if not null:
        return ShiftNull(statistic=stat, null=(), p_value=1.0, n_shifts=0)
    beaten = sum(1 for value in null if value >= stat)
    return ShiftNull(
        statistic=stat,
        null=tuple(null),
        p_value=(1 + beaten) / (1 + len(null)),
        n_shifts=len(null),
    )


# ---------------------------------------------------------------------------
# R7, invariance
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Invariance:
    """R7: which parent sets look the same in every environment, and their intersection."""

    accepted_sets: Tuple[Tuple[int, ...], ...]
    intersection: Tuple[int, ...]
    model_rejected: bool

    def __str__(self) -> str:
        if self.model_rejected:
            return (
                "R7 invariance: no parent set is invariant; the deliverable is the null"
            )
        return (
            f"R7 invariance: {len(self.accepted_sets)} invariant sets, "
            f"intersection {list(self.intersection)}"
        )


def _invariant(
    response: np.ndarray,
    design: np.ndarray,
    environments: np.ndarray,
    alpha: float,
) -> bool:
    """Whether one parent set leaves residuals with the same mean and spread everywhere."""
    beta, _, _ = _least_squares(design, response)
    residual = response - design @ beta
    labels = np.unique(environments)
    tests = max(len(labels), 1)
    for env in labels:
        inside = residual[environments == env]
        outside = residual[environments != env]
        if inside.size < 2 or outside.size < 2:
            continue
        mean_p = float(stats.ttest_ind(inside, outside, equal_var=False).pvalue)
        var_p = float(stats.levene(inside, outside).pvalue)
        if min(mean_p, var_p) * 2 * tests <= alpha:
            return False
    return True


def invariance(
    response: np.ndarray,
    design: np.ndarray,
    environments: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
    max_size: int = 3,
) -> Invariance:
    """
    Which subsets of the candidate parents are invariant across the environments.

    The intersection of the accepted sets is what invariant causal prediction
    reports.  A participant is not an environment here, since the participant
    effect is a direct edge into every target; wave and day-index blocks are.
    """
    design = np.asarray(design, dtype=float)
    columns = list(range(design.shape[1]))
    accepted: List[Tuple[int, ...]] = []
    for size in range(0, min(max_size, len(columns)) + 1):
        for subset in itertools.combinations(columns, size):
            block = np.column_stack(
                [np.ones(len(response))] + [design[:, j] for j in subset]
            )
            if _invariant(response, block, environments, alpha):
                accepted.append(tuple(subset))
    if not accepted:
        return Invariance(accepted_sets=(), intersection=(), model_rejected=True)
    common = set(accepted[0]).intersection(*(set(s) for s in accepted))
    return Invariance(
        accepted_sets=tuple(accepted),
        intersection=tuple(sorted(common)),
        model_rejected=False,
    )


# ---------------------------------------------------------------------------
# R9, stability selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Stability:
    """R9: how often each candidate is selected over complementary subsamples."""

    frequency: Tuple[float, ...]
    threshold: float
    selected: Tuple[int, ...]
    expected_false: float
    n_pairs: int

    def __str__(self) -> str:
        return (
            f"R9 stability: {len(self.selected)} candidates above "
            f"{self.threshold:.2f} over {self.n_pairs} complementary pairs, "
            f"expected low-probability selections at most {self.expected_false:.2f}"
        )


def expected_false_selections(
    n_selected: float, n_candidates: int, threshold: float
) -> float:
    """
    The complementary-pairs bound on how many low-probability candidates are selected.

    Raises:
        ValueError: when the threshold is outside the half-open range the bound
            is stated on.
    """
    if not 0.5 < threshold <= 1.0:
        raise ValueError(
            f"the bound holds for a threshold in (0.5, 1]; got {threshold}"
        )
    if n_candidates <= 0:
        return 0.0
    return float(n_selected) ** 2 / ((2.0 * threshold - 1.0) * n_candidates)


def stability_selection(
    select_fn: Callable[[np.ndarray], Sequence[int]],
    units: Sequence[int],
    n_candidates: int,
    n_pairs: int = 25,
    threshold: float = 0.75,
    seed: int = 0,
) -> Stability:
    """
    Run the selector over complementary halves of the participants and count each hit.

    The sampling unit is the participant, so a split never puts one participant's
    epochs on both sides.
    """
    units = np.asarray(list(units))
    rng = np.random.default_rng(seed)
    counts = np.zeros(n_candidates, dtype=float)
    sizes = []
    for _ in range(n_pairs):
        order = rng.permutation(units)
        half = len(order) // 2
        for side in (order[:half], order[half : 2 * half]):
            chosen = list(select_fn(side))
            sizes.append(len(chosen))
            for index in chosen:
                counts[index] += 1.0
    draws = max(2 * n_pairs, 1)
    frequency = counts / draws
    selected = tuple(int(i) for i in np.nonzero(frequency >= threshold)[0])
    return Stability(
        frequency=tuple(float(f) for f in frequency),
        threshold=threshold,
        selected=selected,
        expected_false=expected_false_selections(
            float(np.mean(sizes)) if sizes else 0.0, n_candidates, threshold
        ),
        n_pairs=n_pairs,
    )


# ---------------------------------------------------------------------------
# R10, the E-value
# ---------------------------------------------------------------------------


def risk_ratio(estimate: float, spread: float) -> float:
    """An approximate risk ratio for a standardized effect."""
    if spread <= 0.0:
        raise ValueError(f"a standardized effect needs a positive spread; got {spread}")
    return float(math.exp(0.91 * estimate / spread))


def e_value(ratio: float) -> float:
    """
    How strong an unmeasured confounder must be to explain one association away.

    A ratio below one is reflected, so the answer is on the same scale either
    way, and a ratio of one returns one.

    Raises:
        ValueError: when the ratio is not positive.
    """
    if ratio <= 0.0:
        raise ValueError(f"a risk ratio is positive; got {ratio}")
    value = ratio if ratio >= 1.0 else 1.0 / ratio
    return float(value + math.sqrt(value * (value - 1.0)))


# ---------------------------------------------------------------------------
# What one reported edge carries
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EdgeRecord:
    """The three numbers 47.3 requires beside every reported edge."""

    key: AnchorKey
    inclusion: float
    e_value: float
    future_lag_p: float

    def as_row(self) -> Dict[str, object]:
        return {
            "scale": self.key.scale,
            "lag": self.key.lag,
            "source": self.key.source,
            "target": self.key.target,
            "inclusion": self.inclusion,
            "e_value": self.e_value,
            "future_lag_p": self.future_lag_p,
        }


def edge_records(
    keys: Sequence[AnchorKey],
    stability: Stability,
    ratios: Sequence[float],
    controls: Sequence[FutureLag],
) -> List[EdgeRecord]:
    """
    Assemble the inclusion frequency, E-value and control p-value of each edge.

    Raises:
        ValueError: when the four sequences do not describe the same edges.
    """
    if not len(keys) == len(ratios) == len(controls):
        raise ValueError(
            f"{len(keys)} edges against {len(ratios)} ratios and "
            f"{len(controls)} controls"
        )
    return [
        EdgeRecord(
            key=key,
            inclusion=stability.frequency[i] if i < len(stability.frequency) else 0.0,
            e_value=e_value(ratio),
            future_lag_p=control.p_value,
        )
        for i, (key, ratio, control) in enumerate(zip(keys, ratios, controls))
    ]


# ---------------------------------------------------------------------------
# What the last three rungs refuse to read
# ---------------------------------------------------------------------------


def _finite(values: Sequence[float], what: str) -> np.ndarray:
    """
    One array of floats, refused when an entry is not a finite number.

    A NaN compares false against every threshold it meets, so a rung that reads
    one reports a residual no reader can act on and a p-value taken against
    nothing.  The entry is named here instead.

    Raises:
        ValueError: naming the first entry that is not finite.
    """
    array = np.asarray(values, dtype=float)
    flat = array.ravel()
    bad = np.flatnonzero(~np.isfinite(flat))
    if bad.size:
        raise ValueError(f"{what} slot {int(bad[0])} holds {float(flat[bad[0]])}")
    return array


# ---------------------------------------------------------------------------
# R6, cross-resolution consistency
# ---------------------------------------------------------------------------

#: How many machine epsilons per added term a fold is allowed to land off by.
#: Summing `m` terms sequentially carries a rounding error bounded by `(m - 1)`
#: epsilons of the sum of their magnitudes, and the coarse entry beside it was
#: summed in some other order.  Integer counts fold exactly and report zero, so
#: this allowance is what a scale carrying a non-integer aggregate needs.
FOLD_ROUNDING = 4.0


@dataclass(frozen=True)
class FoldingResidual:
    """The worst gap between a folded fine array and the coarse array beside it."""

    residual: float
    epoch: int
    n_epochs: int
    tolerance: float = 0.0

    @property
    def folds(self) -> bool:
        """Whether the gap is inside the rounding error of the sum that made it."""
        return self.residual <= self.tolerance

    def __post_init__(self) -> None:
        """
        Raises:
            ValueError: when the gap is not a number at or above zero, or the
                epoch it names is not one the coarse scale holds.  An empty
                coarse scale is the one report with no epoch to name.
        """
        if not math.isfinite(self.residual) or self.residual < 0.0:
            raise ValueError(
                f"a folding residual is a gap at or above zero; got {self.residual}"
            )
        if not math.isfinite(self.tolerance) or self.tolerance < 0.0:
            raise ValueError(
                f"a rounding allowance is at or above zero; got {self.tolerance}"
            )
        empty = (self.epoch, self.n_epochs) == (-1, 0)
        if not (empty or 0 <= self.epoch < self.n_epochs):
            raise ValueError(
                f"epoch {self.epoch} is outside the {self.n_epochs} the coarse "
                "scale holds"
            )


def _parent_epochs(parents: Sequence[int], n_fine: int, n_epochs: int) -> np.ndarray:
    """
    A validated fine-to-coarse map, as one epoch index per fine slot.

    Raises:
        ValueError: when the map does not cover the fine array, runs backwards,
            or reaches an epoch the coarse array does not hold.
    """
    parents = np.asarray(parents, dtype=np.int64)
    if parents.size != n_fine:
        raise ValueError(
            f"the parent map covers {parents.size} fine slots against {n_fine}"
        )
    if parents.size == 0:
        return parents
    backwards = np.flatnonzero(np.diff(parents) < 0)
    if backwards.size:
        slot = int(backwards[0]) + 1
        raise ValueError(
            f"a parent map is non-decreasing; slot {slot} falls in epoch "
            f"{int(parents[slot])} under epoch {int(parents[slot - 1])}"
        )
    if parents[0] < 0 or parents[-1] >= n_epochs:
        raise ValueError(
            f"the parent map spans epochs {int(parents[0])} to "
            f"{int(parents[-1])}, outside the {n_epochs} the coarse array holds"
        )
    return parents


def folding_residual(
    fine: np.ndarray, coarse: np.ndarray, parents: Sequence[int]
) -> FoldingResidual:
    """
    The worst gap between a fine array folded into its parents and the coarse array.

    A reconciled log stores a coarse epoch's total as the sum of its children, so
    the residual is zero on counts and the epoch it names localizes any
    departure.  An empty coarse scale has no epoch to name and reports -1.

    A scale carrying a non-integer aggregate folds in whatever order each side
    was summed in, so the report carries the rounding error that allows: the
    worst epoch's summed magnitudes, priced at `FOLD_ROUNDING` epsilons per added
    term, which `folds` reads the residual against.

    Raises:
        ValueError: when either array holds an entry that is not finite, or the
            parent map does not carry one epoch index per fine slot, inside the
            coarse array's own range.
    """
    fine = _finite(fine, "the fine array")
    coarse = _finite(coarse, "the coarse array")
    parents = _parent_epochs(parents, fine.size, coarse.size)
    if coarse.size == 0:
        return FoldingResidual(residual=0.0, epoch=-1, n_epochs=0)
    gap = np.abs(segment_sum(fine, parents, coarse.size) - coarse)
    worst = int(np.argmax(gap))
    magnitude = segment_sum(np.abs(fine), parents, coarse.size)
    children = np.bincount(parents, minlength=coarse.size) if fine.size else magnitude
    return FoldingResidual(
        residual=float(gap[worst]),
        epoch=worst,
        n_epochs=int(coarse.size),
        tolerance=FOLD_ROUNDING
        * float(np.finfo(float).eps)
        * float(children[worst])
        * float(magnitude[worst]),
    )


def consistency_ladder(
    arrays: Sequence[np.ndarray], parent_maps: Sequence[Sequence[int]]
) -> List[FoldingResidual]:
    """
    Every adjacent scale pair's folding residual, finest first.

    Raises:
        ValueError: when the ladder holds no array, or the parent maps do not
            count the adjacent pairs.
    """
    arrays = list(arrays)
    parent_maps = list(parent_maps)
    if not arrays:
        raise ValueError("a scale ladder holds at least one array; got 0")
    if len(parent_maps) != len(arrays) - 1:
        raise ValueError(
            f"{len(arrays)} scales make {len(arrays) - 1} adjacent pairs; got "
            f"{len(parent_maps)} parent maps"
        )
    return [
        folding_residual(arrays[k], arrays[k + 1], parent_maps[k])
        for k in range(len(parent_maps))
    ]


@dataclass(frozen=True)
class CrossResolution:
    """R6: how far each scale pair of one log is from folding onto the next."""

    residuals: Tuple[FoldingResidual, ...]

    @property
    def passed(self) -> bool:
        """Whether every scale pair folds onto the next inside its rounding error."""
        return all(r.folds for r in self.residuals)


def profile_residual(
    fine: np.ndarray,
    coarse: np.ndarray,
    parents: Sequence[int],
    statistic: Callable[[np.ndarray], float],
) -> float:
    """
    The folding check with a statistic in place of the epoch total.

    A mean reproduces an entry the coarse scale records as its children's
    average.  A variance carries the within-epoch spread the coarse scale never
    recorded, so this number is reported and no rung asserts it is zero.

    Args:
        statistic: what to read off each epoch's fine children, compared
            against the coarse array's own entry for that epoch.

    Raises:
        ValueError: when either array holds an entry that is not finite, an
            epoch holds no fine slot, or the statistic returns something that is
            not finite.
    """
    fine = _finite(fine, "the fine array")
    coarse = _finite(coarse, "the coarse array")
    parents = _parent_epochs(parents, fine.size, coarse.size)
    folded = np.empty(coarse.size)
    for epoch in range(coarse.size):
        children = fine[parents == epoch]
        if children.size == 0:
            raise ValueError(f"epoch {epoch} holds no fine slot to read a profile on")
        value = float(statistic(children))
        if not math.isfinite(value):
            raise ValueError(f"the statistic returned {value} on epoch {epoch}")
        folded[epoch] = value
    return float(np.abs(folded - coarse).max(initial=0.0))


# ---------------------------------------------------------------------------
# R8, the permutation baseline
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PermutationBaseline:
    """R8: one observed score against what data carrying no relation reaches."""

    observed: float
    p_value: float
    null_median: float
    null_quantile: float
    n_at_or_above: int
    n_null: int
    alpha: float

    def __post_init__(self) -> None:
        """
        Raises:
            ValueError: when the two counts cannot have been taken together,
                when the p-value is not the one those counts give, when the
                level lies outside the open unit interval, or when a score is
                not finite.
        """
        if self.n_null < 1 or not 0 <= self.n_at_or_above <= self.n_null:
            raise ValueError(
                f"{self.n_at_or_above} of {self.n_null} null draws cannot stand "
                "at or above the observed score"
            )
        counted = (1 + self.n_at_or_above) / (1 + self.n_null)
        if not math.isclose(self.p_value, counted, abs_tol=1e-12):
            raise ValueError(
                f"a p-value of {self.p_value} does not count "
                f"{self.n_at_or_above} of {self.n_null} null draws"
            )
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"a level lies inside (0, 1); got {self.alpha}")
        scores = (self.observed, self.null_median, self.null_quantile)
        if not all(math.isfinite(value) for value in scores):
            raise ValueError(f"a baseline reads finite scores; got {scores}")

    @property
    def passed(self) -> bool:
        """Whether the exact permutation p-value reaches the declared level."""
        return self.p_value <= self.alpha

    def __str__(self) -> str:
        return (
            f"R8 permutation baseline: {self.observed:.3f} against a null median "
            f"of {self.null_median:.3f}, p = {1 + self.n_at_or_above}/"
            f"{1 + self.n_null} = {self.p_value:.3f} "
            f"{'passes' if self.passed else 'FAILS'}"
        )


def permuted_score(
    observed: float, scores: Sequence[float], alpha: float = DEFAULT_ALPHA
) -> PermutationBaseline:
    """
    Read one observed score against the scores the search reaches on permuted data.

    The permutation keeps every marginal and destroys the relations, so the null
    is the score of no structure at all and a reported score arrives with that
    floor printed beside it.

    The verdict is the p-value against the level.  The null's upper quantile is
    reported beside it and is never the verdict, since interpolating between two
    draws crosses the level at a score the exact p-value has not reached.

    Raises:
        ValueError: when no null score is supplied, when a score is not finite,
            when the level lies outside the open unit interval, or when the
            draws are too few for the smallest attainable p-value to reach that
            level.
    """
    scores = _finite(scores, "a null score").ravel()
    if scores.size == 0:
        raise ValueError("a permutation baseline needs at least one null score; got 0")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"a level lies inside (0, 1); got {alpha}")
    smallest = 1.0 / (scores.size + 1)
    if smallest > alpha:
        raise ValueError(
            f"{scores.size} null scores put the smallest attainable p-value at "
            f"{smallest:.4f}, which never reaches the level {alpha}"
        )
    observed = float(observed)
    if not math.isfinite(observed):
        raise ValueError(f"an observed score is a finite number; got {observed}")
    at_or_above = int(np.count_nonzero(scores >= observed))
    return PermutationBaseline(
        observed=observed,
        p_value=(1 + at_or_above) / (1 + scores.size),
        null_median=float(np.median(scores)),
        null_quantile=float(np.quantile(scores, 1.0 - alpha)),
        n_at_or_above=at_or_above,
        n_null=int(scores.size),
        alpha=float(alpha),
    )


def block_permutation(
    values: np.ndarray, who: np.ndarray, block: int, seed: int = 0
) -> np.ndarray:
    """
    Reorder whole contiguous blocks of epochs inside each participant.

    The relation between two series is carried across the blocks and is destroyed
    by the reordering.  The persistence inside a block survives it, so an
    autocorrelated series meets a null that carries its own persistence
    everywhere except the joins between blocks.  A plain reshuffle is the block
    of one epoch, and it strips that persistence, leaving a null so narrow that
    two unrelated persistent series clear it far more often than the level
    allows.

    Args:
        block: how many consecutive epochs move together.  A participant shorter
            than one block holds a single block and keeps its own order.

    Raises:
        ValueError: when the roster does not label every value, or the block is
            shorter than one epoch.
    """
    values = np.asarray(values, dtype=float)
    who = np.asarray(who, dtype=np.int64)
    if values.size != who.size:
        raise ValueError(f"{values.size} values against {who.size} participant labels")
    if block < 1:
        raise ValueError(f"a block is at least one epoch long; got {block}")
    rng = np.random.default_rng(seed)
    out = values.copy()
    for unit in np.unique(who):
        rows = np.nonzero(who == unit)[0]
        pieces = [rows[i : i + block] for i in range(0, rows.size, block)]
        order = rng.permutation(len(pieces))
        out[rows] = values[np.concatenate([pieces[j] for j in order])]
    return out


# ---------------------------------------------------------------------------
# R11, the held-out likelihood
# ---------------------------------------------------------------------------


def held_out_gain(full: float, base: float, n_slots: int) -> float:
    """
    The per-slot held-out log-likelihood gain, positive when the full model wins.

    Both log likelihoods are taken on slots the fit never read, so their
    difference is in nats, and the slot count puts two folds of different size
    on one scale.

    Raises:
        ValueError: when the slot count is not positive, since a per-slot gain
            has no denominator then, or a likelihood is not finite, since a
            model that gave a held-out slot no probability at all would
            otherwise buy the full model an infinite win.
    """
    if n_slots <= 0:
        raise ValueError(f"a per-slot gain needs at least one slot; got {n_slots}")
    if not (math.isfinite(full) and math.isfinite(base)):
        raise ValueError(
            f"a held-out log likelihood is a finite number; got {full} and {base}"
        )
    return (float(full) - float(base)) / float(n_slots)


@dataclass(frozen=True)
class HeldOut:
    """R11: what the accepted structure buys on slots the fit never read."""

    full: float
    base: float
    n_slots: int
    gain: float

    def __post_init__(self) -> None:
        """
        Raises:
            ValueError: when the gain is not the one the three numbers beside it
                give, so no report passes on a gain its own likelihoods deny.
        """
        earned = held_out_gain(self.full, self.base, self.n_slots)
        if not math.isclose(self.gain, earned, abs_tol=1e-12):
            raise ValueError(
                f"a gain of {self.gain} is not the {earned} that {self.full} "
                f"against {self.base} over {self.n_slots} slots leaves"
            )

    @property
    def passed(self) -> bool:
        """Whether the accepted structure gains on the slots the fit never read."""
        return self.gain > 0.0


def paired_gain(
    full: np.ndarray,
    base: np.ndarray,
    who: np.ndarray,
    n_flips: int = 999,
    seed: int = 0,
) -> Tuple[np.ndarray, float]:
    """
    Each participant's per-slot gain and a sign-flip p-value over participants.

    The participant is the sampling unit, so the slots inside one participant are
    averaged first and the test is taken over the averages.  Under the null the
    sign of a participant's gain is symmetric, and flipping signs at random gives
    the reference distribution of their mean.

    That null is granular at one part in two to the participant count, so the
    smallest p-value five participants can reach is 0.031 and four can reach
    0.062.  A level of 0.05 asks for five participants.

    Args:
        full: the full model's per-slot held-out log likelihood.
        base: the base model's, on the same slots.

    Returns:
        The gain of each participant, ordered by the sorted labels, and the
        one-sided p-value that their mean stands above zero.

    Raises:
        ValueError: when the three arrays disagree in length, a likelihood is
            not finite, no participant is present, or the test is asked for no
            flip at all.
    """
    full = _finite(full, "the full model's held-out likelihood")
    base = _finite(base, "the base model's held-out likelihood")
    who = np.asarray(who, dtype=np.int64)
    if not full.size == base.size == who.size:
        raise ValueError(
            f"{full.size} full and {base.size} base slots against {who.size} "
            "participant labels"
        )
    if n_flips < 1:
        raise ValueError(f"a sign-flip test needs at least one flip; got {n_flips}")
    units = np.unique(who)
    if units.size == 0:
        raise ValueError("a paired test needs at least one participant; got 0")
    per_slot = full - base
    gains = np.array([per_slot[who == unit].mean() for unit in units])
    observed = float(gains.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(int(n_flips), units.size))
    null = (signs * gains).mean(axis=1)
    beaten = int(np.count_nonzero(null >= observed))
    return gains, (1 + beaten) / (1 + int(n_flips))
