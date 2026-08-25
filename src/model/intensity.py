"""
model/intensity.py
==================
One likelihood, at the finest scale, over event starts and their duration marks.

Everything the model says about an epoch of any scale enters here as a term in a
single linear predictor over finest slots.  A coarse count, a coarse duration and
a calendar node reach a fine slot through the epoch that contains it, so a
coarse-scale edge is a coarse-resolution parent of a fine target.  No separate
coarse likelihood carries one.  Coarse arrays are then deterministic sums
of the fine ones, which is why reconciliation is an identity here and not an
invariant to preserve.

The risk set is the correctness that makes this a process and not a regression.
A label that is already running cannot start, so the start density is evaluated
only where a start is possible.  Without it a log with no same-label overlap
leaves the cell (start, already running) empty by construction, an occupancy
coefficient has no finite maximizer, and a fitted model generates traces that are
not valid event logs.

Two predictors, one shape.  `eta` carries the start log-odds and `log nu` the
duration mark's log-mean, each with its own parents and coefficients, which is
what a target quantity is.

A participant axis enters as a level and adds no term.  `FineParams.offset`
holds one row per subject per quantity, mirroring `base` with a subject axis in
front, and that row reaches `linear_predictor` through the `offsets` argument a
regime chain already uses, where a (L,) row broadcasts against the (T_1, L) tile
of `base`.  `sample` indexes `offsets[q][t, ell]` and therefore needs the row
broadcast to (T, L) first, which `np.broadcast_to` gives as a view and copies
nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.typed_graph import (
    CALENDAR,
    COUNT,
    DURATION,
    QUANTITIES,
    TypedEdge,
    TypedGraph,
    read_basis,
)

_MAX_ETA = 30.0

#: Columns of the `clips` counter `sample` fills, in order.  A cap at the low
#: end says the label never fires in that slot and a cap at the high end says it
#: fires in every free one, so the two are different defects and a summed count
#: cannot tell them apart.
CLIP_START_LOW = 0
CLIP_START_HIGH = 1
CLIP_MARK_LOW = 2
CLIP_MARK_HIGH = 3
CLIP_COLUMNS = ("start_low", "start_high", "mark_low", "mark_high")


@dataclass
class FineSources:
    """
    Everything an edge can be read from, and the map from a slot to its epoch.

    Attributes
    ----------
    counts     : X per scale, finest first, each (T_k, L)
    durations  : Y per scale, finest first, each (T_k, L), initiation reading
    calendar   : calendar node columns per scale, finest first, each (T_k, d_k)
    ancestors  : for each scale k, the scale-k epoch of every finest slot, (T_1,)
    """

    counts: List[np.ndarray]
    durations: List[np.ndarray]
    calendar: List[np.ndarray]
    ancestors: List[np.ndarray]

    @property
    def n_slots(self) -> int:
        return int(self.counts[0].shape[0])

    @property
    def n_labels(self) -> int:
        return int(self.counts[0].shape[1])

    @property
    def K(self) -> int:
        return len(self.counts)

    def quantity(self, scale: int, quantity: str) -> np.ndarray:
        return (
            self.counts[scale - 1] if quantity == COUNT else self.durations[scale - 1]
        )


def ancestor_maps(parent_maps: Sequence[np.ndarray], n_slots: int) -> List[np.ndarray]:
    """The scale-k epoch of every finest slot, for k = 1 upward."""
    out = [np.arange(n_slots, dtype=np.int64)]
    for pm in parent_maps:
        out.append(np.asarray(pm, dtype=np.int64)[out[-1]])
    return out


def edge_feature(edge: TypedEdge, sources: FineSources) -> np.ndarray:
    """
    The edge's column over finest slots, read at its own resolution.

    A slot reads its ancestor epoch at scale `src_scale`, stepped back by the
    lag.  An epoch before the record opens contributes nothing, which is the same
    convention the lagged feature has always used.  The edge's basis term decides
    what is read off the epoch, so a band edge carries its indicator here and
    `fine_design` builds the same column for the solve.
    """
    who = sources.ancestors[edge.src_scale - 1]
    index = who - edge.lag
    inside = index >= 0
    if edge.src_kind == CALENDAR:
        column = sources.calendar[edge.src_scale - 1][:, edge.src_label]
        raw = np.zeros(sources.n_slots, dtype=float)
        raw[inside] = column[index[inside]]
        return raw
    array = sources.quantity(edge.src_scale, edge.src_quantity)[:, edge.src_label]
    raw = np.zeros(sources.n_slots, dtype=float)
    raw[inside] = read_basis(array[index[inside]], edge.basis)
    return raw


def assert_predictable(edge: TypedEdge, sources: FineSources) -> None:
    """
    Raise when the ladder would let a lagged feature read the future.

    A coarse epoch stepped back by one lag ends before the first slot of the
    epoch a target sits in, and that holds only because the parent maps are
    non-decreasing.  It is checked here, since nothing else
    in the project enforces monotone parent maps.  The other half of
    predictability belongs elsewhere: `TypedEdge` refuses a same-epoch read above
    the finest scale, and `TypedGraph.label_order` puts every same-epoch source
    before the label that reads it.
    """
    if edge.src_kind == CALENDAR:
        return
    who = sources.ancestors[edge.src_scale - 1]
    if np.any(np.diff(who) < 0):
        raise ValueError(
            f"the ancestor map of scale {edge.src_scale} is not non-decreasing, so "
            "a lagged feature at that scale can read the future"
        )


@dataclass
class FineParams:
    """Coefficients of the two finest-scale predictors."""

    n_labels: int
    n_subjects: int = 0
    base: Dict[str, np.ndarray] = field(default_factory=dict)
    offset: Dict[str, np.ndarray] = field(default_factory=dict)
    coef: Dict[Tuple[int, str], Dict[TypedEdge, float]] = field(default_factory=dict)
    mark_dispersion: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        for quantity in QUANTITIES:
            self.base.setdefault(quantity, np.zeros(self.n_labels))
            self.offset.setdefault(quantity, np.zeros((self.n_subjects, self.n_labels)))
        for ell in range(self.n_labels):
            for quantity in QUANTITIES:
                self.coef.setdefault((ell, quantity), {})
        if self.mark_dispersion is None:
            self.mark_dispersion = np.ones(self.n_labels)

    def level(self, quantity: str, subject: Optional[int] = None) -> np.ndarray:
        """
        The (L,) intercept vector in force at one subject's slots.

        A roster of no subjects gives a (0, L) offset and this returns `base`, so
        every fit with no participant axis reads the numbers it always read.
        """
        if self.n_subjects == 0 or subject is None:
            return self.base[quantity]
        return self.base[quantity] + self.offset[quantity][subject]


def linear_predictor(
    params: FineParams,
    graph: TypedGraph,
    sources: FineSources,
    quantity: str,
    offsets: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    The (T_1, L) predictor of one target quantity.

    `offsets` carries everything a latent contributes once its path is known, as
    one array of the same shape, so this function stays a statement about the
    observed structure and the latent parameterization lives with the inference.
    """
    if quantity not in QUANTITIES:
        raise ValueError(f"target quantity {quantity!r}; expected one of {QUANTITIES}")
    out = np.tile(params.base[quantity].astype(float), (sources.n_slots, 1))
    if offsets is not None:
        out = out + np.asarray(offsets, dtype=float)
    for ell in range(sources.n_labels):
        for edge in sorted(graph.get_parents(ell, quantity)):
            assert_predictable(edge, sources)
            weight = params.coef[(ell, quantity)].get(edge, 0.0)
            if weight:
                out[:, ell] += weight * edge_feature(edge, sources)
    return out


def running_mask(X1: np.ndarray, Y1: np.ndarray) -> np.ndarray:
    """
    Whether each label is already inside an episode at each finest slot.

    A slot that starts an episode is not counted as already running, since the
    start happened there; the slots it covers afterward are.
    """
    T, L = X1.shape
    out = np.zeros((T, L), dtype=bool)
    for t, ell in np.argwhere(X1 > 0):
        length = int(Y1[t, ell])
        if length > 1:
            out[t + 1 : min(t + length, T), ell] = True
    return out


def risk_set(X1: np.ndarray, Y1: np.ndarray) -> np.ndarray:
    """Slots at which a start is possible, which is where the density applies."""
    return ~running_mask(X1, Y1)


def usable_slots(
    X1: np.ndarray, Y1: np.ndarray, observed: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    The slots a start term runs over, which is the risk set inside the record.

    This is the one expression where the risk set and the observation mask meet.
    A slot the risk set drops carries a start probability of zero and the model
    explains its emptiness; a slot the mask drops carries no observation to
    explain.  Every consumer downstream receives their conjunction through the
    `at_risk` argument it already has, so the two are combined once here and
    stay two statements everywhere above.

    Args:
        observed: (T_1,) bool, the observation mask where nothing is lagged and
            its forward dilation, `mask.scorable_rows`, where a lagged design is
            being scored.

    Raises:
        ValueError: when the mask is not boolean, or does not carry one flag per
            finest slot.
    """
    at_risk = risk_set(X1, Y1)
    if observed is None:
        return at_risk
    flags = np.asarray(observed)
    if flags.dtype != np.bool_:
        raise ValueError(
            f"an observation mask is a boolean array; this one has dtype "
            f"{flags.dtype}, which reads as a flag wherever it is non-zero"
        )
    if flags.shape != (at_risk.shape[0],):
        raise ValueError(
            f"an observation mask carries one flag per finest slot, so "
            f"{at_risk.shape[0]} of them; this one is {flags.shape}"
        )
    return at_risk & flags[:, None]


def start_loglik(
    X1: np.ndarray, eta: np.ndarray, at_risk: Optional[np.ndarray] = None
) -> float:
    """Bernoulli log-likelihood of the starts, over the risk set only."""
    eta = np.clip(np.asarray(eta, dtype=float), -_MAX_ETA, _MAX_ETA)
    mask = np.ones_like(X1, dtype=bool) if at_risk is None else at_risk
    x = (np.asarray(X1) > 0).astype(float)
    terms = x * eta - np.logaddexp(0.0, eta)
    return float(terms[mask].sum())


def mark_loglik(
    X1: np.ndarray,
    Y1: np.ndarray,
    log_nu: np.ndarray,
    dispersion: np.ndarray,
    observed: Optional[np.ndarray] = None,
) -> float:
    """
    Negative binomial log-likelihood of the durations, shifted to at least one.

    Only a slot that starts an episode carries a mark, so the sum runs over the
    starts and a slot with no start contributes nothing.  The risk set has no
    place here, since a start is at risk by definition; the observation mask
    does, since a mark inside a gap was never recorded.
    """
    from scipy.special import gammaln

    starts = np.asarray(X1) > 0
    if observed is not None:
        starts = starts & np.asarray(observed, dtype=bool)[:, None]
    if not starts.any():
        return 0.0
    t_idx, ell_idx = np.nonzero(starts)
    mu = np.exp(np.clip(log_nu[t_idx, ell_idx], -_MAX_ETA, _MAX_ETA))
    r = np.asarray(dispersion, dtype=float)[ell_idx]
    y = np.asarray(Y1)[t_idx, ell_idx].astype(float) - 1.0
    y = np.maximum(y, 0.0)
    total = (
        gammaln(y + r)
        - gammaln(r)
        - gammaln(y + 1.0)
        + r * np.log(r / (r + mu))
        + y * np.log(mu / (r + mu))
    )
    return float(total.sum())


def total_loglik(
    X1: np.ndarray,
    Y1: np.ndarray,
    params: FineParams,
    graph: TypedGraph,
    sources: FineSources,
    offsets: Optional[Dict[str, np.ndarray]] = None,
    observed: Optional[np.ndarray] = None,
) -> float:
    """
    The whole model's log-likelihood, which lives entirely at the finest scale.

    No coarse array carries a density.  Every coarse quantity is a deterministic
    sum of these two, so adding a term for one would count the same data twice.
    """
    offsets = offsets or {}
    eta = linear_predictor(params, graph, sources, COUNT, offsets.get(COUNT))
    log_nu = linear_predictor(params, graph, sources, DURATION, offsets.get(DURATION))
    at_risk = usable_slots(X1, Y1, observed)
    return start_loglik(X1, eta, at_risk) + mark_loglik(
        X1, Y1, log_nu, params.mark_dispersion, observed
    )


def empty_sources(
    ancestors: Sequence[np.ndarray],
    calendar: Sequence[np.ndarray],
    n_labels: int,
) -> FineSources:
    """Sources with zeroed counts and durations, sized from the ancestor maps."""
    counts, durations = [], []
    for who in ancestors:
        n_epochs = int(who.max()) + 1 if who.size else 0
        counts.append(np.zeros((n_epochs, n_labels), dtype=np.int64))
        durations.append(np.zeros((n_epochs, n_labels), dtype=np.int64))
    return FineSources(
        counts=counts,
        durations=durations,
        calendar=[np.asarray(c, dtype=float) for c in calendar],
        ancestors=[np.asarray(a, dtype=np.int64) for a in ancestors],
    )


def sample(
    params: FineParams,
    graph: TypedGraph,
    sources: FineSources,
    rng: np.random.Generator,
    offsets: Optional[Dict[str, np.ndarray]] = None,
    clips: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Draw starts and marks slot by slot, filling `sources` as it goes.

    The draw is sequential because a lag-1 finest-scale parent reads the slot
    just written, and a coarse parent reads an epoch that is complete only once
    every slot inside it has been drawn.  Inside one slot the labels are drawn in
    `TypedGraph.label_order`, so a same-epoch parent is already written when the
    label reading it is reached.  The same risk set the likelihood uses decides
    where a start is possible, so a drawn trace holds no same-label overlap and
    is a valid event log by construction.

    `clips`, when given, is an `(L, 4)` integer array counting how often the
    predictor left the representable range and was capped, per label, with one
    column per entry of `CLIP_COLUMNS`.  The cap is what turns a coefficient no
    data supports into a log that still looks like a log -- a start term of
    -2.5e+11 reaches the draw as a flat "never" and nothing downstream can tell
    it apart from a label that is genuinely rare.  A caller that does not count
    these cannot see that its draw was decided by the cap rather than the fit.
    The end is counted as well as the side, since a cap at the low end
    suppresses a label and a cap at the high end fires it in every free slot,
    and a total over the two says which of those happened only by elimination.
    """
    offsets = offsets or {}
    T, L = sources.n_slots, sources.n_labels
    X1 = np.zeros((T, L), dtype=np.int64)
    Y1 = np.zeros((T, L), dtype=np.int64)
    busy_until = np.zeros(L, dtype=np.int64)

    parents = {
        (ell, quantity): sorted(graph.get_parents(ell, quantity))
        for ell in range(L)
        for quantity in QUANTITIES
    }
    for edges in parents.values():
        for edge in edges:
            assert_predictable(edge, sources)

    order = graph.label_order()
    for t in range(T):
        for ell in order:
            if busy_until[ell] > t:
                continue
            eta = float(params.base[COUNT][ell])
            if COUNT in offsets:
                eta += float(offsets[COUNT][t, ell])
            for edge in parents[(ell, COUNT)]:
                weight = params.coef[(ell, COUNT)].get(edge, 0.0)
                if weight:
                    eta += weight * _feature_at(edge, sources, t)
            capped = float(np.clip(eta, -_MAX_ETA, _MAX_ETA))
            if clips is not None and capped != eta:
                clips[ell, CLIP_START_HIGH if eta > 0 else CLIP_START_LOW] += 1
            eta = capped
            if rng.random() >= 1.0 / (1.0 + np.exp(-eta)):
                continue

            log_nu = float(params.base[DURATION][ell])
            if DURATION in offsets:
                log_nu += float(offsets[DURATION][t, ell])
            for edge in parents[(ell, DURATION)]:
                weight = params.coef[(ell, DURATION)].get(edge, 0.0)
                if weight:
                    log_nu += weight * _feature_at(edge, sources, t)
            capped_nu = float(np.clip(log_nu, -_MAX_ETA, _MAX_ETA))
            if clips is not None and capped_nu != log_nu:
                clips[ell, CLIP_MARK_HIGH if log_nu > 0 else CLIP_MARK_LOW] += 1
            mu = float(np.exp(capped_nu))
            r = float(params.mark_dispersion[ell])
            length = 1 + int(rng.negative_binomial(r, r / (r + mu)))

            X1[t, ell] = 1
            Y1[t, ell] = length
            busy_until[ell] = t + length
            for k in range(1, sources.K + 1):
                epoch = int(sources.ancestors[k - 1][t])
                sources.counts[k - 1][epoch, ell] += 1
                sources.durations[k - 1][epoch, ell] += length
    return X1, Y1


def _feature_at(edge: TypedEdge, sources: FineSources, t: int) -> float:
    """One edge's feature at one slot, read from the partially filled sources."""
    index = int(sources.ancestors[edge.src_scale - 1][t]) - edge.lag
    if index < 0:
        return 0.0
    if edge.src_kind == CALENDAR:
        return float(sources.calendar[edge.src_scale - 1][index, edge.src_label])
    array = sources.quantity(edge.src_scale, edge.src_quantity)
    return float(read_basis(array[index, edge.src_label], edge.basis))
