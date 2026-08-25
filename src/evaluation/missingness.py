"""
evaluation/missingness.py
=========================
The mechanism a record's gaps are declared to have, and the two places gaps bite.

One density, at the finest scale
--------------------------------
A slot contributes one Bernoulli start factor while its label is at risk, and a
slot that starts an episode contributes one negative binomial mark factor.
Given the predictor and the risk set those factors are conditionally
independent, and every coarse count and coarse duration is a deterministic sum
of the finest arrays.

An unobserved slot is an absent factor.  Summing the joint over that slot's own
support returns the product of the remaining factors, so the observed-data
likelihood is the same product with the term left out, and marginalizing an
unobserved child is deletion of a term.  The fit path already reads this way.
The start sum runs over a boolean mask, the upward message multiplies by the
mask before its segment sum, and the parameter step selects its rows with it.
Intersecting the observation mask into the risk set is the whole of the
response-side work.

The aggregation property of the Dirichlet-Multinomial collapses siblings that
share a drawn block total.  The finest-scale density draws no block total.  A
block sum is arithmetic over the slots inside it, the siblings carry no
constraint between them, and the property has nothing here to collapse.

An exposure offset moves the mean of a density over a coarse count.  Every
density here sits on a start indicator or on the length of one episode, and a
coarse count reaches the likelihood as a lagged regressor of a fine target.
Partial coverage of a coarse block is therefore measurement error in a
covariate.  The observed block sum is low by the share of the block that was
never recorded, and every fine slot whose parent reads that block reads the
same wrong number.  Two rules answer it, and both are arithmetic on the source
arrays before any column is built.  `coverage_scaled` raises the observed sum
to the whole block at the block's own coverage, which is the rate reading an
offset carried, applied to a regressor.  It is a plug-in, since the log of a
scaled count approximates the conditional mean of the log of the true count.
`readable_slots` refuses a block whose coverage sits at or below a declared
floor and drops from the design every finest slot whose parent reads it.

Imputation on the response side has nothing to write.  A value placed in an
unobserved slot enters the Bernoulli sum as a recorded start or a recorded
non-start, which is a statement the record does not make, and deletion of the
term is already exact.  Imputation on the feature side is unavoidable and is
already in the code.  A lagged feature reads log1p of a block sum, a zero fill
reads as a block that held no events, and the record's own left edge takes that
fill.  The choice there is between an unstated imputation and a stated one.
The coverage rules are the stated one, and `feature_exposure` counts the design
rows that stand on them, which is what a sensitivity report on the feature side
has to count.

The risk set is the one coupling the finest factors keep.  A start that
happened inside a gap is absent from the arrays, so the slots its mark covered
appear at risk and enter the Bernoulli sum as observed non-starts.  A start
recorded before a gap carries its mark at the start slot, so the running mask
bridges that gap on its own.  `dilate_after_gaps` clears the mask forward from
the end of each gap, per label, until the first observed start of that label,
which proves the label was at risk there, or until a declared cap of slots,
which is what keeps a rare label from losing the rest of its series.

The declaration
---------------
Whether any of this is ignorable is a claim about the world, so a dataset
states the claim and this module reports what the record can refuse.  A
declaration is one of `mcar`, `ignorable_given_calendar` or `nonignorable`.
Two questions decide whether it stands.  Do the gaps clump in time beyond what
a reshuffle of the same gap rate produces, and do the calendar and context
columns predict the gap indicator out of sample.  A gap opened by the clock, by
a maintenance window or by a device that sleeps overnight answers yes to the
second, and the conditioning set the fit already carries is the set that makes
it ignorable.  Gaps that clump and stay unpredictable are driven by something
outside that set, and the ignorable path is refused for them.  A record whose
gaps do neither is consistent with a completely random gap process.

The criterion and its threshold
-------------------------------
The criterion is the out-of-sample R^2 of a ridge linear probability fit of the
gap indicator on the calendar and context columns, pooled over contiguous
held-out folds.  The threshold on that number comes from the indicator itself.
A circular block permutation, blocked at the longest observed gap, keeps the
observed run lengths and destroys the alignment with the design, and the
level-alpha upper quantile of the permuted statistic is what the observed
statistic must clear.  A fixed R^2 cutoff would mean different things at
different horizons and different gap rates.  The serial dependence of a gap
indicator also breaks the chi-square null that a likelihood-ratio test reads
its threshold from, and the block permutation keeps that dependence inside the
null.  The level is declared by the caller and defaults to 0.05, and the number
of permutations has to make that level attainable.  A test that fails to reject
records what this many slots and this many permutations could refuse.

The mask arrives as an argument, so this module holds no opinion about where a
dataset stores it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from ..inference.bound import segment_sum

# A dataset declares one of these.  `mcar` says the gaps carry no information at
# all, `ignorable_given_calendar` says the calendar and context columns carry
# all of it, and `nonignorable` says the gap process belongs to the model.
MECHANISMS = ("mcar", "ignorable_given_calendar", "nonignorable")

DEFAULT_ALPHA = 0.05
DEFAULT_PERMUTATIONS = 999

# Contiguous folds, so a held-out block sits in time next to the blocks that
# predict it and a gap that spans several slots cannot be half in each.
N_FOLDS = 5

# A ridge of one on standardized columns conditions the solve and shrinks a
# coefficient by about one part in the training length.
RIDGE = 1.0

# A block permutation with fewer blocks than this has almost no support, so the
# block length is capped even when one gap runs longer.
MIN_BLOCKS = 4


# ---------------------------------------------------------------------------
# Shared checks
# ---------------------------------------------------------------------------


def _as_mask(observed: np.ndarray, what: str) -> np.ndarray:
    """The mask as a boolean array, with any other encoding refused."""
    mask = np.asarray(observed)
    if mask.dtype != np.bool_:
        raise ValueError(
            f"{what} is a boolean mask; this one has dtype {mask.dtype}, which "
            "the fit reads as row indices wherever it selects with a mask and "
            "as a multiplier wherever it reduces with one"
        )
    return mask


def _slot_mask(observed: np.ndarray) -> np.ndarray:
    """The mask of one label over the finest slots, as a boolean vector."""
    mask = _as_mask(observed, "an observation mask")
    if mask.ndim != 1:
        raise ValueError(
            f"a gap indicator carries one entry per finest slot; this one is "
            f"{mask.shape}. Pass one label's column, or call `audit`."
        )
    return mask


def _gap_indicator(mask: np.ndarray) -> np.ndarray:
    """The gap indicator of a slot mask, with a degenerate record refused."""
    gap = ~mask
    n_missing = int(gap.sum())
    if n_missing == 0:
        raise ValueError("this record has no gap, so no gap process is exercised by it")
    if n_missing == mask.size:
        raise ValueError(
            "no slot of this record is observed, so no mechanism can be "
            "checked against it"
        )
    return gap


def _check_level(alpha: float, n_permutations: int) -> None:
    """Refuse a level outside (0, 1) and a draw count that cannot reach it."""
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError(f"the level of a permutation test lies in (0, 1); got {alpha}")
    smallest = 1.0 / (int(n_permutations) + 1)
    if smallest > float(alpha):
        raise ValueError(
            f"{n_permutations} permutations put the smallest attainable "
            f"p-value at {smallest:.4f}, which never reaches the level {alpha}"
        )


def _share(count: int, total: int) -> float:
    """A count over a total, with an empty total reported as zero."""
    return float(count) / float(total) if total else 0.0


# ---------------------------------------------------------------------------
# The design the gaps are tested against
# ---------------------------------------------------------------------------


def slot_design(
    blocks: Sequence[np.ndarray], ancestors: Sequence[np.ndarray]
) -> np.ndarray:
    """
    Per-scale context blocks broadcast to finest slots and laid side by side.

    Args:
        blocks: one matrix per scale, finest first, each (T_k, d_k).
        ancestors: for each scale, the scale-k epoch of every finest slot.

    Returns:
        The (T_1, sum d_k) design a gap indicator is tested against.
    """
    if not blocks:
        raise ValueError("a slot design needs at least one scale's block")
    if len(blocks) != len(ancestors):
        raise ValueError(
            f"{len(blocks)} context blocks against {len(ancestors)} ancestor maps"
        )
    columns = [
        np.asarray(b, dtype=float)[np.asarray(a)] for b, a in zip(blocks, ancestors)
    ]
    return np.concatenate(columns, axis=1)


# ---------------------------------------------------------------------------
# Do the gaps clump
# ---------------------------------------------------------------------------


def gap_runs(gap: np.ndarray) -> np.ndarray:
    """Lengths of the maximal runs of True in a 1-D indicator."""
    flags = np.asarray(gap, dtype=bool).astype(np.int8)
    edges = np.diff(np.concatenate(([0], flags, [0])))
    return np.nonzero(edges == -1)[0] - np.nonzero(edges == 1)[0]


@dataclass(frozen=True)
class GapStructure:
    """How a gap indicator sits in time, against a reshuffle of the same rate."""

    n_slots: int
    n_missing: int
    missing_fraction: float
    longest_gap: int
    mean_gap: float
    mean_gap_null: float
    p_value: float
    clustered: bool


def gap_structure(
    observed: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = 0,
) -> GapStructure:
    """
    Whether the gaps clump beyond a reshuffle of the same number of gap slots.

    The statistic is the mean length of a gap run, which rises as the same gap
    slots gather into fewer spells.  The null holds the gap rate and permutes
    the positions, so it is the exchangeable-in-time gap process.

    Raises:
        ValueError: when the record is fully observed or fully unobserved, so
            neither the statistic nor its null is defined.
    """
    mask = _slot_mask(observed)
    _check_level(alpha, n_permutations)
    gap = _gap_indicator(mask)
    runs = gap_runs(gap)
    stat = float(runs.mean())
    rng = np.random.default_rng(seed)
    null = np.array(
        [float(gap_runs(rng.permutation(gap)).mean()) for _ in range(n_permutations)]
    )
    p_value = float((1 + np.sum(null >= stat)) / (1 + n_permutations))
    return GapStructure(
        n_slots=int(mask.size),
        n_missing=int(gap.sum()),
        missing_fraction=float(gap.mean()),
        longest_gap=int(runs.max()),
        mean_gap=stat,
        mean_gap_null=float(np.quantile(null, 1.0 - alpha)),
        p_value=p_value,
        clustered=p_value <= float(alpha),
    )


# ---------------------------------------------------------------------------
# Do the calendar and context columns predict the gaps
# ---------------------------------------------------------------------------


def _fold_operators(design: np.ndarray, n_folds: int) -> List[tuple]:
    """Per fold, the standardized held-out rows and the map response to weights."""
    n, d = design.shape
    edges = np.linspace(0, n, n_folds + 1).astype(int)
    folds: List[tuple] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        held = np.zeros(n, dtype=bool)
        held[lo:hi] = True
        train = ~held
        center = design[train].mean(axis=0)
        spread = design[train].std(axis=0)
        # A column that never moves inside the training fold carries no
        # contrast there, and dividing by one leaves it at zero after centering.
        spread = np.where(spread > 0.0, spread, 1.0)
        z_train = (design[train] - center) / spread
        z_held = (design[held] - center) / spread
        gram = z_train.T @ z_train + RIDGE * np.eye(d)
        folds.append((train, held, z_held, np.linalg.solve(gram, z_train.T)))
    return folds


def _oos_r2(y: np.ndarray, folds: Sequence[tuple]) -> float:
    """Out-of-sample R^2 of the pooled held-out predictions against the rate."""
    pred = np.empty(y.size)
    for train, held, z_held, weights in folds:
        y_train = y[train]
        level = float(y_train.mean())
        pred[held] = level + z_held @ (weights @ (y_train - level))
    pred = np.clip(pred, 0.0, 1.0)
    resid = float(np.sum((y - pred) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - resid / total


def _block_permutation(
    gap: np.ndarray, block: int, rng: np.random.Generator
) -> np.ndarray:
    """The indicator rotated and its blocks reordered, which keeps the runs."""
    n = gap.size
    rolled = np.roll(gap, int(rng.integers(n)))
    n_blocks = int(np.ceil(n / block))
    padded = np.concatenate([rolled, rolled[: n_blocks * block - n]])
    tiles = padded.reshape(n_blocks, block)
    return tiles[rng.permutation(n_blocks)].reshape(-1)[:n]


@dataclass(frozen=True)
class Predictability:
    """How much of a gap indicator the calendar and context columns carry."""

    n_slots: int
    n_columns: int
    block: int
    r2: float
    r2_null: float
    p_value: float
    predictable: bool


def predictability(
    observed: np.ndarray,
    design: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    n_folds: int = N_FOLDS,
    seed: int = 0,
) -> Predictability:
    """
    Whether the design predicts the gap indicator out of sample.

    Args:
        observed: one label's observation mask over the finest slots.
        design: the calendar and context columns at the finest slots.

    Returns:
        The held-out R^2, the level-alpha quantile the block permutation null
        reaches, the permutation p-value and the verdict at that level.

    Raises:
        ValueError: when the design does not carry one row per slot, when it
            has no columns, or when the series is too short for the folds.
    """
    mask = _slot_mask(observed)
    _check_level(alpha, n_permutations)
    matrix = np.asarray(design, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != mask.size:
        raise ValueError(
            f"the design carries one row per finest slot, so {mask.size} rows; "
            f"this one is {matrix.shape}"
        )
    if matrix.shape[1] == 0:
        raise ValueError(
            "the design has no columns, so it states nothing the gaps could follow"
        )
    if n_folds < 2 or mask.size < 2 * n_folds:
        raise ValueError(
            f"{mask.size} slots over {n_folds} folds leave a fold with nothing "
            "to hold out; a held-out fold needs at least one slot and at least "
            "two folds have to exist"
        )
    gap = _gap_indicator(mask).astype(float)
    runs = gap_runs(gap > 0.0)
    block = max(1, min(int(runs.max()), mask.size // MIN_BLOCKS))
    folds = _fold_operators(matrix, n_folds)
    stat = _oos_r2(gap, folds)
    rng = np.random.default_rng(seed)
    null = np.array(
        [
            _oos_r2(_block_permutation(gap, block, rng), folds)
            for _ in range(n_permutations)
        ]
    )
    p_value = float((1 + np.sum(null >= stat)) / (1 + n_permutations))
    return Predictability(
        n_slots=int(mask.size),
        n_columns=int(matrix.shape[1]),
        block=block,
        r2=stat,
        r2_null=float(np.quantile(null, 1.0 - alpha)),
        p_value=p_value,
        predictable=p_value <= float(alpha),
    )


# ---------------------------------------------------------------------------
# The declaration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MissingnessVerdict:
    """Whether one label's gaps support the mechanism the dataset declares."""

    label: str
    declared: str
    accepted: bool
    reason: str
    structure: Optional[GapStructure] = None
    predictability: Optional[Predictability] = None

    def __str__(self) -> str:
        state = "stands" if self.accepted else "REFUSED"
        return f"{self.label}: {self.declared} {state} ({self.reason})"


def _clumping_phrase(structure: GapStructure) -> str:
    """The clumping finding as one clause of a reason."""
    return (
        f"the gaps clump, at a mean run of {structure.mean_gap:.2f} slots "
        f"against {structure.mean_gap_null:.2f} for a reshuffle of the same "
        f"rate (p={structure.p_value:.3f})"
    )


def _prediction_phrase(pred: Predictability) -> str:
    """The predictability finding as one clause of a reason."""
    return (
        f"the calendar and context columns predict the gaps, at a held-out "
        f"R^2 of {pred.r2:.3f} against {pred.r2_null:.3f} for a block "
        f"permutation (p={pred.p_value:.3f})"
    )


def check(
    declared: str,
    observed: np.ndarray,
    design: np.ndarray,
    label: str = "gaps",
    alpha: float = DEFAULT_ALPHA,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    n_folds: int = N_FOLDS,
    seed: int = 0,
) -> MissingnessVerdict:
    """
    Judge one label's gaps against the mechanism the dataset declares.

    Raises:
        ValueError: when the declared mechanism is not one of `MECHANISMS`.
    """
    if declared not in MECHANISMS:
        raise ValueError(
            f"declared mechanism {declared!r}; expected one of {list(MECHANISMS)}"
        )
    mask = _slot_mask(observed)
    if mask.all():
        return MissingnessVerdict(
            label=label,
            declared=declared,
            accepted=True,
            reason="the record has no gap, so the mechanism is never exercised",
        )
    structure = gap_structure(mask, alpha, n_permutations, seed)
    pred = predictability(mask, design, alpha, n_permutations, n_folds, seed)

    if declared == "nonignorable":
        return MissingnessVerdict(
            label=label,
            declared=declared,
            accepted=True,
            reason=(
                "the gap process is declared informative, so the ignorable "
                "path stays closed and a fit carries the gaps in what it reports"
            ),
            structure=structure,
            predictability=pred,
        )

    if declared == "mcar":
        found = []
        if structure.clustered:
            found.append(_clumping_phrase(structure))
        if pred.predictable:
            found.append(_prediction_phrase(pred))
        if found:
            return MissingnessVerdict(
                label=label,
                declared=declared,
                accepted=False,
                reason=(
                    "a completely random gap process shows neither, and here "
                    + " and ".join(found)
                ),
                structure=structure,
                predictability=pred,
            )
        return MissingnessVerdict(
            label=label,
            declared=declared,
            accepted=True,
            reason=(
                f"at level {alpha} the gaps stay exchangeable in time and the "
                "calendar and context columns carry nothing about them"
            ),
            structure=structure,
            predictability=pred,
        )

    if structure.clustered and not pred.predictable:
        return MissingnessVerdict(
            label=label,
            declared=declared,
            accepted=False,
            reason=(
                _clumping_phrase(structure)
                + " and "
                + f"the design leaves them unpredicted (held-out R^2 "
                f"{pred.r2:.3f}, p={pred.p_value:.3f}), so the driver sits "
                "outside the conditioning set the fit carries"
            ),
            structure=structure,
            predictability=pred,
        )
    if pred.predictable:
        reason = (
            _prediction_phrase(pred)
            + ", so the conditioning set the fit carries covers the gap process"
        )
    else:
        reason = (
            f"at level {alpha} the gaps stay exchangeable in time, which a gap "
            "process no coarser than the calendar produces"
        )
    return MissingnessVerdict(
        label=label,
        declared=declared,
        accepted=True,
        reason=reason,
        structure=structure,
        predictability=pred,
    )


@dataclass(frozen=True)
class MissingnessAudit:
    """One verdict per label, and whether the dataset's declaration stands."""

    declared: str
    verdicts: List[MissingnessVerdict]

    @property
    def accepted(self) -> bool:
        return all(v.accepted for v in self.verdicts)

    def report(self) -> str:
        """The audit as text, for a run to print before it fits."""
        lines = [f"missingness declared {self.declared!r}"]
        lines.extend(f"  {v}" for v in self.verdicts)
        refused = [v.label for v in self.verdicts if not v.accepted]
        if refused:
            lines.append(
                f"  the declaration does not stand for {', '.join(refused)};"
                " a fit on this record either carries the gap process or"
                " reports it as a limitation"
            )
        return "\n".join(lines)


def audit(
    declared: str,
    observed: np.ndarray,
    design: np.ndarray,
    labels: Optional[Sequence[str]] = None,
    alpha: float = DEFAULT_ALPHA,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    n_folds: int = N_FOLDS,
    seed: int = 0,
) -> MissingnessAudit:
    """
    Judge every label's gaps against the mechanism the dataset declares.

    Args:
        observed: the observation mask, (T_1,) for one label or (T_1, L).

    Raises:
        ValueError: when the mask is neither one nor two dimensional, or when
            the label names do not match its columns.
    """
    mask = _as_mask(observed, "an observation mask")
    if mask.ndim == 1:
        mask = mask[:, None]
    if mask.ndim != 2:
        raise ValueError(
            f"an observation mask is indexed by slot and label; this one is "
            f"{mask.shape}"
        )
    names = (
        list(labels) if labels is not None else [f"L{j}" for j in range(mask.shape[1])]
    )
    if len(names) != mask.shape[1]:
        raise ValueError(
            f"{len(names)} label names against {mask.shape[1]} mask columns"
        )
    return MissingnessAudit(
        declared=declared,
        verdicts=[
            check(
                declared,
                mask[:, j],
                design,
                names[j],
                alpha,
                n_permutations,
                n_folds,
                seed,
            )
            for j in range(mask.shape[1])
        ],
    )


# ---------------------------------------------------------------------------
# The risk set after a gap
# ---------------------------------------------------------------------------


def duration_cap(starts: np.ndarray, marks: np.ndarray) -> np.ndarray:
    """
    Slots an unrecorded episode can still cover after a gap ends, per label.

    An episode of length y starting at the last slot of a gap covers at most
    y - 1 slots beyond it, so the longest episode the label was seen to have
    bounds the window that a start hidden in the gap can reach into.  A label
    with no recorded start returns zero, since no episode length was observed
    for it.
    """
    flags = _as_mask(starts, "a start indicator")
    lengths = np.asarray(marks)
    if lengths.shape != flags.shape:
        raise ValueError(
            f"the marks are {lengths.shape} and the starts {flags.shape}; both "
            "are indexed by slot and label"
        )
    longest = np.array(
        [
            int(np.max(lengths[flags[:, ell], ell], initial=1))
            for ell in range(flags.shape[1])
        ]
    )
    return np.maximum(longest - 1, 0)


def dilate_after_gaps(
    observed: np.ndarray, starts: np.ndarray, cap: np.ndarray
) -> np.ndarray:
    """
    Clear the mask forward from the end of each gap, per label, up to the cap.

    Args:
        observed: the observation mask, one flag per slot or one per slot and
            label.  A per-slot mask covers every label at that slot.
        starts: whether each slot records a start of that label.
        cap: slots of dilation, one number or one per label.

    Returns:
        The mask with the ambiguous slots after each gap cleared.  A slot that
        records a start ends the dilation and stays observed.
    """
    mask = _as_mask(observed, "an observation mask")
    flags = _as_mask(starts, "a start indicator")
    if flags.ndim != 2:
        raise ValueError(
            f"a start indicator is indexed by slot and label; this one is "
            f"{flags.shape}"
        )
    if mask.ndim == 1 and mask.size == flags.shape[0]:
        mask = np.repeat(mask[:, None], flags.shape[1], axis=1)
    if mask.shape != flags.shape:
        raise ValueError(
            f"the mask is {mask.shape} and the starts {flags.shape}; both are "
            "indexed by slot and label"
        )
    n_slots, n_labels = mask.shape
    caps = np.asarray(cap, dtype=np.int64)
    if caps.ndim == 0:
        caps = np.full(n_labels, int(caps), dtype=np.int64)
    if caps.shape != (n_labels,):
        raise ValueError(
            f"a cap is one number, or one per label, so {n_labels} of them; "
            f"this one is {caps.shape}"
        )
    if np.any(caps < 0):
        raise ValueError(
            f"a cap counts slots forward from a gap, so it starts at zero; the "
            f"smallest here is {int(caps.min())}"
        )
    out = mask.copy()
    for ell in range(n_labels):
        reach = int(caps[ell])
        for t in range(1, n_slots):
            if mask[t, ell] and not mask[t - 1, ell]:
                for u in range(t, min(t + reach, n_slots)):
                    if flags[u, ell]:
                        break
                    out[u, ell] = False
    return out


# ---------------------------------------------------------------------------
# Coverage of a coarse block, which a lagged feature reads
# ---------------------------------------------------------------------------


def block_coverage(
    observed: np.ndarray, ancestors: np.ndarray, n_epochs: int
) -> np.ndarray:
    """
    The share of each scale-k epoch that was recorded.

    An epoch holding no slots of its own reports zero coverage, which the floor
    in `readable_slots` then refuses.
    """
    mask = _as_mask(observed, "an observation mask")
    who = np.asarray(ancestors)
    if who.shape != (mask.shape[0],):
        raise ValueError(
            f"the ancestor map carries one entry per finest slot, so "
            f"{mask.shape[0]} of them; this one is {who.shape}"
        )
    seen = segment_sum(mask.astype(float), who, n_epochs)
    size = segment_sum(np.ones(mask.shape[0]), who, n_epochs)
    if seen.ndim == 2:
        size = size[:, None]
    return seen / np.maximum(size, 1.0)


def coverage_scaled(block_sums: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    """
    The observed block sum raised to the whole block, at the block's coverage.

    A block with no coverage returns zero, which carries no rate of its own and
    is left for the floor in `readable_slots` to remove.
    """
    values = np.asarray(block_sums, dtype=float)
    share = np.asarray(coverage, dtype=float)
    if share.shape != values.shape:
        raise ValueError(
            f"the coverage is {share.shape} and the block sums {values.shape}; "
            "both are indexed by epoch and label"
        )
    return np.where(share > 0.0, values / np.where(share > 0.0, share, 1.0), 0.0)


def readable_slots(
    coverage: np.ndarray, ancestors: np.ndarray, lag: int, floor: float = 0.0
) -> np.ndarray:
    """
    Finest slots whose lagged source epoch was recorded well enough to read.

    Args:
        coverage: the source label's coverage column at the source scale.
        ancestors: the source-scale epoch of every finest slot.
        lag: epochs the feature steps back before reading.
        floor: coverage at or below which a block is refused.  At the default
            only an empty block is refused, and a higher floor is a declaration
            about how much of a block may be reconstructed.

    Returns:
        A boolean mask over finest slots.  A slot whose source epoch predates
        the record is refused with the under-covered ones, since the zero fill
        that stands there reads as a block that held no events.
    """
    share = np.asarray(coverage, dtype=float)
    who = np.asarray(ancestors)
    if int(lag) < 0:
        raise ValueError(f"a lag steps back, so it starts at zero; got {lag}")
    if not 0.0 <= float(floor) < 1.0:
        raise ValueError(
            f"a coverage floor lies in [0, 1); got {floor}. A floor of one "
            "refuses every block, including the fully recorded ones."
        )
    if int(np.max(who, initial=-1)) >= share.size:
        raise ValueError(
            f"the ancestor map reaches epoch {int(np.max(who, initial=-1))} and "
            f"the coverage column holds {share.size}"
        )
    index = who - int(lag)
    inside = index >= 0
    good = np.zeros(who.size, dtype=bool)
    good[inside] = share[index[inside]] > float(floor)
    return good


@dataclass(frozen=True)
class FeatureExposure:
    """How much of one edge's column stands on a reconstructed block."""

    n_slots: int
    n_dropped: int
    n_scaled: int

    @property
    def dropped_fraction(self) -> float:
        return _share(self.n_dropped, self.n_slots)

    @property
    def scaled_fraction(self) -> float:
        return _share(self.n_scaled, self.n_slots)

    def __str__(self) -> str:
        return (
            f"{self.n_dropped} of {self.n_slots} rows dropped "
            f"({self.dropped_fraction:.3f}), {self.n_scaled} kept on a scaled "
            f"block ({self.scaled_fraction:.3f})"
        )


def feature_exposure(
    coverage: np.ndarray, ancestors: np.ndarray, lag: int, floor: float = 0.0
) -> FeatureExposure:
    """How many of one edge's design rows depend on the coverage rules."""
    good = readable_slots(coverage, ancestors, lag, floor)
    share = np.asarray(coverage, dtype=float)
    index = np.asarray(ancestors) - int(lag)
    inside = index >= 0
    partial = np.zeros(good.size, dtype=bool)
    partial[inside] = share[index[inside]] < 1.0
    return FeatureExposure(
        n_slots=int(good.size),
        n_dropped=int((~good).sum()),
        n_scaled=int((partial & good).sum()),
    )


def structure_dependence(baseline: Iterable, alternative: Iterable) -> Dict[str, float]:
    """
    How much of a learned edge set survives a second reading of the same gaps.

    Two fits under two coverage rules answer the sensitivity question that the
    feature side leaves open, and this is the arithmetic over their edge sets.
    """
    first = set(baseline)
    second = set(alternative)
    shared = len(first & second)
    return {
        "baseline": float(len(first)),
        "alternative": float(len(second)),
        "shared": float(shared),
        "kept": _share(shared, len(first)),
        "changed": float(len(first ^ second)),
    }
