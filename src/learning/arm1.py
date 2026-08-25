"""
learning/arm1.py
================
The observational arm's fit: what it searches, what it pays, and what it accepts.

The pieces this composes all exist. `pooled.PooledFit` streams the cohort and
holds one graph, `relation_price` prices a relation once and gives the two
strata their own in-degree bounds, `calendar_arms` says which clock nodes an arm
offers, and `anchors` says what an accepted edge is scored against. What is here
is the candidate space, the price, and the translation from an accepted index to
the key the anchors are keyed on.

What the arm searches over
--------------------------
The search runs at every scale of the ladder, and the finest one is a source
like the rest. It was out while one refusal ended a target's round. The
minute-scale self lag of a label scores four orders of magnitude above every
cross relation, its move overshoots, and two targets stopped there having bought
nothing. A refused candidate is now blocked and the round carries on, with
`MAX_REFUSED` bounding the cohort passes a round spends on moves it does not
keep.

A same-epoch read of another label is a candidate at the minute scale. A
relation shorter than one epoch reaches a coarser scale as co-movement inside
one epoch, so a model carrying lagged reads alone holds no term for two labels
that move together inside the slot the likelihood lives on. The draw order is
what makes it predictable. `TypedGraph.label_order` puts every same-epoch source
before the label that reads it, `closes_cycle` keeps out the candidate that would
leave no such order, and a label reading itself in the epoch being drawn is never
offered.

A regime chain runs at the hour scale with two states. The panel's failing rows
are the within-day profile shape and the within-day dispersion, both read at the
hour scale, and an intercept with pooled coefficients carries nothing that
switches inside a day. Two states are the smallest term that does, and the chain
is fitted by the same pooled sweep the coefficients are, so its posterior, its
transitions and its effect all come off one bound.

Two settings carry a measurement each.

A proposed move takes three coefficient rounds before the guard reads it. One
Newton step at the current weights overshoots on a column that nearly separates
the response: on this log the strongest relation there is drops the objective by
678 after one step and raises it by 650 after four, so a guard reading one step
refuses the one relation the study guarantees.

The opening pass takes twenty coefficient rounds. Every intercept starts at
zero, which is one start per two slots, and a rare label's own rate is eleven
log-odds below that: after six rounds `mindfulness` carried 44 expected starts
against none observed and `nap` 46 against four, and the search then priced
every move against a baseline that fitted neither. Twenty rounds reach the score
equation exactly, at which the expected starts of every label equal the observed
ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..core.cohort import Cohort
from ..core.typed_graph import (
    CALENDAR,
    COUNT,
    DURATION,
    FINEST_SCALE,
    LEVEL,
    OBSERVED,
    TypedEdge,
    TypedGraph,
)
from ..data.anchors import AnchorKey, key_of
from ..data.label_axis import SCHEDULED_LABEL, LabelAxis
from ..data.lifesnaps_ladder import LADDER_UNITS
from ..model.intensity import FineParams
from .calendar_arms import HOME, calendar_candidates
from .fine_design import CalendarIndex
from .fine_sem import RegimeChain
from .pooled import PooledFit
from .relation_price import GroupReport, RelationPrice, candidate_relations

#: The scales the search runs at, the finest one included.
SEARCH_SCALES: Tuple[int, ...] = (1, 2, 3, 4)

#: The largest lag proposed at each scale, which is what the anchors declare.
MAX_LAG = 3

#: Coefficient rounds a proposed move takes before the guard reads its objective.
MOVE_ROUNDS = 3

#: Coefficient rounds the opening pass takes before the first move is priced.
OPENING_ROUNDS = 20

#: How many in-degree slots each stratum carries.
MAX_EDGES = 4
MAX_SELF = 2

#: Refusals a target's round takes before it stops.
MAX_REFUSED = 4

#: The scale the regime chain runs at, and how many states it carries.  One
#: state is no chain at all, which is the no-regime control.
REGIME_SCALE = 2
REGIME_STATES = 2

#: The quantile of the measured null a price is read off.
PRICE_QUANTILE = 0.95


def candidate_pool(
    axis: LabelAxis,
    arm: str = HOME,
    scales: Sequence[int] = SEARCH_SCALES,
    max_lag: int = MAX_LAG,
) -> List[TypedEdge]:
    """
    Every edge the search may propose into one target.

    Four families: a lagged read of an observed label at either quantity, a
    same-epoch read of an observed label at the finest scale, the clock nodes one
    ablation arm offers at lag zero, and the scheduled label, which is exogenous
    and may therefore be read at lag zero as well.
    """
    pool = [
        TypedEdge(OBSERVED, int(k), src, quantity, lag)
        for k in scales
        for src in range(axis.L)
        for quantity in (COUNT, DURATION)
        for lag in range(1, int(max_lag) + 1)
    ]
    if FINEST_SCALE in [int(k) for k in scales]:
        pool += [
            TypedEdge(OBSERVED, FINEST_SCALE, src, quantity, 0)
            for src in range(axis.L)
            for quantity in (COUNT, DURATION)
        ]
    pool += calendar_candidates(axis, arm)
    if SCHEDULED_LABEL in axis.exogenous:
        column = axis.exogenous_index(SCHEDULED_LABEL)
        pool += [
            TypedEdge(CALENDAR, int(k), column, LEVEL, lag)
            for k in scales
            for lag in range(0, int(max_lag) + 1)
        ]
    return pool


def target_pool(pool: Sequence[TypedEdge], target: int) -> List[TypedEdge]:
    """One target's candidates, without the same-epoch read of its own draw."""
    return [
        edge for edge in pool if not (edge.is_same_epoch and edge.src_label == target)
    ]


#: The share of a target's own information a self candidate has to carry before
#: the search may propose it.  Measured on the full roster at `max_lag = 3`, the
#: minute-scale lag-1 self column is identically zero on every scorable row for
#: `mindfulness`, `nap`, `sleep` and `walk`, and nonzero on 10 of 3,492,956 rows
#: for `gym`, because the risk set excludes exactly the slots where it would be
#: one; only `survey_response` carries a real one at 988 of 80,640.  Those are
#: 0, 2.9e-6 and 1.2e-2 of the target's information, so this floor sits two
#: orders clear of both sides.
MIN_SELF_SUPPORT = 1e-4


def drop_unsupported_self(
    pools: Mapping[int, Sequence[TypedEdge]],
    stats,
    min_support: float = MIN_SELF_SUPPORT,
) -> Tuple[Dict[int, List[TypedEdge]], Dict[int, List[TypedEdge]]]:
    """
    Refuse a self candidate whose column has no support on the scorable rows.

    A label reading its own recent history at the finest scale is asking about
    slots the risk set has already removed: if the label started a minute ago it
    is still busy now, so `now` is not scorable and the column is zero wherever
    it could be scored.  A column like that cannot estimate a coefficient, but it
    can still be bought, because the gain it is ranked on divides a finite
    residual by a near-zero Fisher information.  The 11248 nats the 2026-08-21
    exclusion was reversed over is that quotient, not a strong relation.

    Only self candidates are tested.  The finest scale stays open as a
    cross-label source, which is what the same measurement says is real:
    `notification_received -> survey_response` returns at +1.974, +1.419, +1.058.

    Returns:
        The pools to search, and the candidates refused, per target.
    """
    kept: Dict[int, List[TypedEdge]] = {}
    dropped: Dict[int, List[TypedEdge]] = {}
    for target, edges in pools.items():
        seen = stats.candidates.get(target)
        design = stats.designs.get(target)
        if seen is None or design is None:
            kept[target] = list(edges)
            continue
        floor = min_support * max(float(np.max(np.diag(design.A))), 1.0)
        keeping: List[TypedEdge] = []
        refusing: List[TypedEdge] = []
        for j, edge in enumerate(edges):
            own = (not edge.is_exogenous) and edge.src_label == target
            if own and float(seen.diagonal[j]) <= floor:
                refusing.append(edge)
            else:
                keeping.append(edge)
        kept[target] = keeping
        if refusing:
            dropped[target] = refusing
    return kept, dropped


def regime_chains(
    cohort: Cohort,
    axis: LabelAxis,
    scale: int = REGIME_SCALE,
    n_states: int = REGIME_STATES,
) -> List[RegimeChain]:
    """The latent chain the arm fits, which is empty where the arm declares none."""
    if n_states < 2:
        return []
    return [
        RegimeChain.uniform(
            scale=int(scale),
            n_states=int(n_states),
            n_labels=axis.L,
            n_epochs=cohort.hierarchy.n_epochs(int(scale)),
        )
    ]


def exogenous_index(axis: LabelAxis, n_scales: int) -> CalendarIndex:
    """Where each exogenous column sits, so a coefficient names the node it read."""
    return CalendarIndex(
        placed={
            (k, j): (name, name)
            for k in range(1, n_scales + 1)
            for j, name in enumerate(axis.exogenous)
        },
        widths={k: len(axis.exogenous) for k in range(1, n_scales + 1)},
    )


@dataclass
class Arm1Fit:
    """One fitted graph over the panel, and what the search paid for it."""

    graph: TypedGraph
    params: FineParams
    pool: List[TypedEdge]
    reports: Dict[int, GroupReport]
    price: float
    null_median: float
    null_max: float
    arm: str
    accepted: List[Tuple[TypedEdge, int]] = field(default_factory=list)
    chains: List[RegimeChain] = field(default_factory=list)
    # Columns the final solve held at zero, and targets it declined to write at
    # all, per target.  Empty on a healthy fit.  A run that reports coefficients
    # without reading this cannot tell a fitted zero from a refused one.
    refused: Dict[int, List[str]] = field(default_factory=dict)
    # Self candidates the pool refused before the search, for carrying no
    # support on the scorable rows.
    unsupported: Dict[int, List[str]] = field(default_factory=dict)

    @property
    def objective(self) -> float:
        return sum(report.trace[-1] for report in self.reports.values())

    def by_stratum(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for report in self.reports.values():
            for stratum, count in report.by_stratum().items():
                out[stratum] = out.get(stratum, 0) + count
        return out

    def keys(
        self, axis: LabelAxis, units: Sequence[str] = LADDER_UNITS
    ) -> List[AnchorKey]:
        """Every accepted edge as the key the anchors are keyed on."""
        return [key_of(edge, target, axis, units) for edge, target in self.accepted]

    def rejected(self) -> Dict[int, int]:
        """How many moves the guard turned away, per target."""
        return {t: len(r.rejected) for t, r in self.reports.items() if r.rejected}


def fit_arm1(
    cohort: Cohort,
    axis: LabelAxis,
    arm: str = HOME,
    scales: Sequence[int] = SEARCH_SCALES,
    max_lag: int = MAX_LAG,
    coefficient_rounds: int = OPENING_ROUNDS,
    move_rounds: int = MOVE_ROUNDS,
    max_edges: int = MAX_EDGES,
    max_self: int = MAX_SELF,
    quantile: float = PRICE_QUANTILE,
    price: Optional[float] = None,
    max_refused: int = MAX_REFUSED,
    regime_scale: int = REGIME_SCALE,
    regime_states: int = REGIME_STATES,
    seed: int = 0,
) -> Arm1Fit:
    """
    Fit one graph and one regime chain over the panel, at this cohort's own price.

    Args:
        price: a declared price; None reads one off the null measured on this
            roster, at `quantile`.
        regime_states: states the chain carries; one leaves the arm with no
            chain, which is the no-regime control.

    Returns:
        The fit, its accepted edges, its chain and the price it paid.
    """
    pool = candidate_pool(axis, arm, scales, max_lag)
    pools = {target: target_pool(pool, target) for target in range(axis.L)}
    graph = TypedGraph(
        label_names=list(axis.targets), calendar_names=list(axis.exogenous)
    )
    chains = regime_chains(cohort, axis, regime_scale, regime_states)
    fit = PooledFit(
        graph,
        FineParams(n_labels=axis.L),
        chains,
        {k: int(max_lag) for k in range(1, cohort.K + 1)},
        quantity=COUNT,
        index=exogenous_index(axis, cohort.K),
        max_sweeps=1,
        move_rounds=move_rounds,
        max_refused=max_refused,
    )
    # One round visits the chain, re-reads its transitions and takes a Newton
    # step on the coefficients, so the chain and the coefficients arrive at the
    # price on the same bound.  With no chain the first two steps do nothing.
    for _ in range(max(int(coefficient_rounds), 1)):
        stats = fit.accumulate(cohort, step=True)
        fit.set_transitions(stats)
        fit.solve(stats)

    fit.candidates = {target: list(edges) for target, edges in pools.items()}
    # One unrolled pass to read what each candidate column actually carries, so
    # a self column with no support on the scorable rows never reaches the
    # search.  The price is then measured on the pool that is searched rather
    # than on one holding candidates no target could estimate.
    measured = fit.accumulate(cohort, step=False)
    pools, unsupported = drop_unsupported_self(pools, measured)
    fit.candidates = {target: list(edges) for target, edges in pools.items()}
    nulls = fit.null_gains(cohort, seed=seed)
    bar = float(np.quantile(nulls, quantile)) if price is None else float(price)
    relation_price = RelationPrice(
        relation=bar,
        representation=bar / 10.0,
        n_relations=len(set(candidate_relations([(edge, 0) for edge in pool]))),
    )
    reports = fit.search_grouped(cohort, relation_price, max_edges, max_self)
    stats = fit.accumulate(cohort, step=True)
    fit.set_transitions(stats)
    fit.solve(stats)

    accepted = [
        (pools[target][i], target)
        for target, report in sorted(reports.items())
        for i in report.accepted
    ]
    return Arm1Fit(
        graph=graph,
        params=fit.params,
        pool=pool,
        reports=reports,
        price=bar,
        null_median=float(np.median(nulls)) if nulls.size else 0.0,
        null_max=float(nulls.max()) if nulls.size else 0.0,
        arm=arm,
        accepted=accepted,
        chains=list(fit.chains),
        refused={t: list(names) for t, names in fit.refused.items()},
        unsupported={t: [str(e) for e in edges] for t, edges in unsupported.items()},
    )


# ---------------------------------------------------------------------------
# The columns the falsification rungs read
# ---------------------------------------------------------------------------


def epoch_series(
    cohort: Cohort, scale: int, column: int, exogenous: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    One label's epochs at one scale, laid end to end over the roster.

    Returns:
        The values and, per epoch, which subject they belong to, so a lagged read
        and a permutation both stay inside one subject.
    """
    values: List[np.ndarray] = []
    who: List[np.ndarray] = []
    for index, name in enumerate(cohort.subjects):
        data = cohort.open(name)
        block = cohort.sources(data).calendar if exogenous else data.X
        own = np.asarray(block[scale - 1][:, column], dtype=float)
        values.append(own)
        who.append(np.full(own.size, index, dtype=np.int64))
    return np.concatenate(values), np.concatenate(who)


def edge_series(
    cohort: Cohort, edge: TypedEdge, target: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The response, the parent at its own lag and the participant index of an edge.

    The parent is stepped back inside its own subject, so no epoch of one
    participant is ever read as the past of another.
    """
    response, who = epoch_series(cohort, edge.src_scale, target)
    raw, _ = epoch_series(
        cohort, edge.src_scale, edge.src_label, exogenous=edge.is_exogenous
    )
    parent = np.zeros_like(raw)
    if edge.lag:
        for unit in np.unique(who):
            rows = np.nonzero(who == unit)[0]
            if rows.size > edge.lag:
                parent[rows[edge.lag :]] = raw[rows[: -edge.lag]]
    else:
        parent = raw
    return response, parent, who


def future_of(values: np.ndarray, who: np.ndarray, lead: int = 1) -> np.ndarray:
    """One epoch ahead inside each subject, which is what R4 adds to the fit."""
    out = np.zeros_like(values)
    for unit in np.unique(who):
        rows = np.nonzero(who == unit)[0]
        if rows.size > lead:
            out[rows[:-lead]] = values[rows[lead:]]
    return out


def environments(cohort: Cohort, waves: Mapping[str, str], scale: int) -> np.ndarray:
    """
    Which environment each epoch belongs to, for the invariance rung.

    A participant is not an environment: the participant effect is a direct edge
    into every target. The recruitment wave is, and so is the half of the window
    an epoch falls in.
    """
    labels: List[np.ndarray] = []
    for name in cohort.subjects:
        n = cohort.hierarchy.n_epochs(scale)
        half = (np.arange(n) >= n // 2).astype(np.int64)
        wave = 0 if waves.get(name, "spring") == "spring" else 1
        labels.append(2 * wave + half)
    return np.concatenate(labels)


# ---------------------------------------------------------------------------
# R2, the sham label
# ---------------------------------------------------------------------------

#: The label connected to nothing, whose every accepted edge is a false positive.
SHAM_LABEL = "sham"


def sham_cohort(
    cohort: Cohort, axis: LabelAxis, reference: str, seed: int = 0
) -> Tuple[Cohort, LabelAxis]:
    """
    The panel with one more label, matched to a reference and caused by nothing.

    The sham carries each subject's own values for the reference label, put in a
    new order drawn at the reference's own persistence, so the marginal and the
    participant heterogeneity are reproduced exactly and the ordering carries no
    relation to any other label. Every edge touching it is a false positive by
    construction, which is what makes it the one rung that measures that rate
    inside the behavioural block.
    """
    from dataclasses import replace as _replace

    from ..evaluation.falsification import sham_series

    column = axis.target_index(reference)
    names = list(axis.targets) + [SHAM_LABEL]
    wider = LabelAxis(
        targets=tuple(names),
        exogenous=axis.exogenous,
        grouped={**dict(axis.grouped), SHAM_LABEL: SHAM_LABEL},
    )
    order = {name: i for i, name in enumerate(cohort.subjects)}
    maps = cohort.hierarchy.parent_maps

    def reader(unit: str):
        data = cohort.reader(unit)
        rng = np.random.default_rng(seed + order[unit])
        own = np.asarray(data.X[0][:, column], dtype=float)
        drawn = sham_series(own, np.zeros(own.size, dtype=np.int64), rng)
        finest_x = np.concatenate(
            [data.X[0], drawn[:, None].astype(data.X[0].dtype)], axis=1
        )
        finest_y = np.concatenate(
            [data.Y[0], drawn[:, None].astype(data.Y[0].dtype)], axis=1
        )
        X = [finest_x]
        Y = [finest_y]
        for parent in maps[:-1]:
            size = int(np.max(parent)) + 1
            X.append(_fold(X[-1], parent, size))
            Y.append(_fold(Y[-1], parent, size))
        return _replace(data, label_names=names, X=X, Y=Y)

    return _replace(cohort, label_names=names, reader=reader), wider


def _fold(array: np.ndarray, parent: np.ndarray, size: int) -> np.ndarray:
    """One scale's arrays summed into the scale above."""
    out = np.zeros((size, array.shape[1]), dtype=array.dtype)
    np.add.at(out, np.asarray(parent, dtype=np.int64), array)
    return out


def touches_sham(fit: Arm1Fit, axis: LabelAxis) -> Tuple[int, int]:
    """
    How many accepted edges touch the sham label, and how many could have.

    Every one of them is a false positive by construction, so the pair is the
    numerator and the denominator R2 reports.
    """
    sham = axis.target_index(SHAM_LABEL)
    hits = sum(
        1
        for edge, target in fit.accepted
        if target == sham or (not edge.is_exogenous and edge.src_label == sham)
    )
    offered = sum(
        1 for edge in fit.pool if not edge.is_exogenous and edge.src_label == sham
    ) * (axis.L - 1) + len(fit.pool)
    return hits, offered
