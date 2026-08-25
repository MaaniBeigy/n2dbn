"""
experiments/exp11_pooling.py
============================
Experiment 11: what pooling buys, measured against a pre-registered prediction.

Every earlier experiment fits one record.  This one fits the same graph over a
roster and asks what the roster bought, on two arms over one cross.

  * The subject arm holds the horizon at 12 weeks and sweeps the roster size.
    The S=1 rung is fitted once per persona, so the claim that one person's
    history cannot carry a cohort rule does not rest on one draw.
  * The horizon arm holds the roster at its largest rung and sweeps the number
    of weeks, sharing the 12-week cell with the subject arm.

Each cell fits one pooled model, scores its typed graph against the declared
manifest and records four things: the recall of the placed declared rules per
scale, the edge count into declared-stationary labels in its own block with its
own denominator, the ledger's predicted recall beside the measured one, and a
claim string saying whether the cell answers a question about a population or a
question about one person.

What an edge costs
------------------
A price answers how large a gain a candidate reaches when there is no relation
there at all, so each cell measures that distribution before it searches:
`PooledFit.null_gains` rolls every candidate's own column, which keeps its
marginal distribution and its serial dependence and destroys only what it lines
up with, and scores the result against the same base set on the same roster.
The price is a stated quantile of what comes back, and the quantile is read off
the candidate space the search sweeps: at `C` candidates per target,
`1 - 1 / C` is the bar about one candidate per target reaches carrying nothing,
so the edges into declared-stationary labels are what that costs and the
stationary block is where the cost is counted.

The price is measured on the cell's own roster and is therefore not invariant to
it.  A gain is a sum over the records the cohort holds, so S copies of one
record legitimately carry S times the evidence to any likelihood criterion, and
a price that refused to move would suppress the very information distinct
participants bring, which is the effect this arm exists to measure.  What the
null fixes is the rate: no more than `1 - q` of the candidates that carry
nothing may clear the bar.

What a candidate reads
----------------------
A source offers the search the terms of its own basis.  The log term reads
log1p of the source and carries a relation that acts monotonically.  A band
reads the indicator that the source's value falls between two quantiles of the
cohort, cut on the rows the cell fits by `quantile_edges`, so a relation that
acts as a step at some threshold is expressible as the step between two adjacent
bands.  The cut points come from the cohort and from no declaration, so no
threshold is anticipated, every source of every scale and quantity is cut the
same way, and a cut set that straddles a threshold expresses it poorly.

The terms multiply the candidate count and the price moves with them: the 480
typed relations per target of the shipped benchmark become 1248 candidates at
the 30-subject roster, and the bar is the `1 - 1 / C` quantile of the null at
that larger `C`.  A rule counts as recovered when the search holds any term of
its relation, since the rule is the relation and the term is how a fit expressed
it.

The pre-registered prediction
-----------------------------
`src/evaluation/sufficiency.py` was fitted on the frozen generator and predicts,
at 60 labels, 91 day-scale epochs per subject and a limit of two lags: 0.000 at
1 subject, 0.139 at 5, 0.265 at 10, 0.418 at 20, 0.604 at 30, recall 0.5 at
24.40 subjects, recall 0.8 out of reach.  This run measures against those rows
and may refute them.  A departure is a finding about the ledger, since the
ledger's curve was swept on generated data and this arm is fitted on a log.

What a cell compares, and what it holds apart
---------------------------------------------
A declared rule names a lag in whole days.  That is a lag in epochs at a scale
whose epoch divides a day and whose declared limit admits the quotient, so a
rule is placed at those scales alone and those scales alone are scored.  On a
minute, hour, day, week ladder with a limit of two epochs, a one-day rule is one
epoch at the day scale and a seven-day rule is one epoch at the week scale.

A rule also names the quantity it drives.  The pooled fit scores one target
quantity, the count, so a rule driving a duration is held apart in its own list
and is outside the recall denominator.  No search over count targets could have
placed it, and counting it as missed would price the fit for a target it never
formed a design for.

A rule may be scoped to a persona.  A pooled fit sees every persona at once, so
every rule is read as applying, and the share of subjects a scoped rule has
evidence from travels with the placed rule as `universal`.

Run
---
    python -m experiments.exp11_pooling --dry-run

    python -m experiments.exp11_pooling
        --csv data/dynamic_patterns/event_log.tsv
        --manifest data/dynamic_patterns/manifest_declared.json
        --subject-arm 1,6,12,24,30 --horizon-arm 4,8,12

A small cell, which is what fits on one machine in minutes:

    python -m experiments.exp11_pooling
        --subject-arm 1,3 --horizon-arm "" --weeks 2 --max-labels 8
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.core.calendar import SlotCalendar
from src.core.cohort import Cohort
from src.core.scales import ScaleHierarchy
from src.core.typed_graph import COUNT, DURATION, OBSERVED, TypedEdge, TypedGraph
from src.data.dynamic_patterns import (
    BENCHMARK_ORIGIN,
    IDENTIFIABLE,
    CarryOverRule,
    DeclaredTruth,
    build_subject_data,
    read_declared_manifest,
    read_subject_event_log,
    relation_terms,
    subject_persona,
)
from src.data.event_log import ObservedEvent
from src.evaluation.sufficiency import (
    expected_recall,
    n_candidate_edges,
    scale_verdict,
    subjects_for_recall,
)
from src.learning.fine_sstep import basis_terms, quantile_edges
from src.learning.pooled import PooledFit
from src.model.intensity import FineParams
from src.utils.provenance import write_provenance

MINUTES_PER_DAY = 1440
DAYS_PER_WEEK = 7

# The ledger's reading of this benchmark, which the pre-registration was written
# from: 60 labels, 91 day-scale epochs per subject and a limit of two lags, so
# 120 candidate edges per target at the day scale.
LEDGER_LABELS = 60
LEDGER_EPOCHS_PER_SUBJECT = 91
LEDGER_MAX_LAG = 2
LEDGER_ROWS = (1, 5, 10, 20, 30)
LEDGER_TARGETS = (0.5, 0.8)

# One outer iteration over one subject at the full 12-week horizon costs about
# 12 seconds on 8 cores with a day-and-week candidate space, and the work is
# linear in subject-weeks, so one subject-week of one iteration is a twelfth of
# that.  The dry run prices a sweep off this one number.
SECONDS_PER_SUBJECT_WEEK = 1.0

DEFAULT_SUBJECT_ARM = "1,3,6,12,18,24,30"
DEFAULT_HORIZON_ARM = "2,4,6,8,12"
DEFAULT_WEEKS = 12

# The acceptance rule, as source, target and lag in days.  Sleep longer than 390
# minutes on day d-1 drives the walking count on day d, and it fires 745 times
# over the cohort.
KEY_RULE = "sleep|walking|1"

SUBJECT_ARM = "subject"
HORIZON_ARM = "horizon"

# Bins per source, so each relation offers a log term and up to three bands.
DEFAULT_BASIS_BINS = 4

# What a source with no measured bins offers, which is the log term alone.
_NO_BINS = np.array([-np.inf, np.inf])

# When a coefficient round has stopped buying anything.  A Fisher-scoring step
# doubles its correct digits per round near the maximizer, so the round after
# the one that reaches this is already at machine precision.
_COEF_TOL = 1e-6


# ---------------------------------------------------------------------------
# Where a declared rule lands on a ladder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlacedRule:
    """One declared carry-over rule placed at one scale as a typed edge."""

    rule: CarryOverRule
    edge: TypedEdge
    dst_label: int
    dst_quantity: str
    scale: int

    @property
    def name(self) -> str:
        """The name exp6 reports the same rule under."""
        return f"{self.rule.source}|{self.rule.target}|{self.rule.lag_days}d"

    @property
    def key(self) -> str:
        """Source, target and lag in days, which is how a run names one rule."""
        return f"{self.rule.source}|{self.rule.target}|{self.rule.lag_days}"

    @property
    def universal(self) -> bool:
        """Whether the rule names no persona, so every subject carries it."""
        return not self.rule.personas


def epoch_lag(lag_days: int, calendar: SlotCalendar, k: int) -> Optional[int]:
    """Epochs of scale k a lag of whole days spans, when it spans whole ones."""
    span = calendar.epoch_minutes(k)
    total = MINUTES_PER_DAY * int(lag_days)
    return total // span if total % span == 0 else None


def placed_typed_rules(
    truth: DeclaredTruth,
    labels: Sequence[str],
    calendar: SlotCalendar,
    max_lag: Mapping[int, int],
) -> Tuple[Dict[int, List[PlacedRule]], List[CarryOverRule], List[CarryOverRule]]:
    """
    Every declared rule at each scale whose epoch divides its lag, as a typed edge.

    `DeclaredTruth.typed_edges` writes the lag in days into the edge, which is
    the epoch lag at the day scale alone, so the source scale and the lag are
    both replaced here with what the scale being placed at reads.

    Args:
        truth: the declared manifest.
        labels: the realized label axis.
        calendar: the clock the scales are read off.
        max_lag: the declared epoch limit at each scale, keyed by scale.

    Returns:
        The placed rules per scale, the rules naming a label the log never
        realized, and the rules whose lag no scale of this ladder reaches.
    """
    mapped, off_axis = truth.typed_edges(labels)
    placed: Dict[int, List[PlacedRule]] = {}
    off_grid: List[CarryOverRule] = []
    for declared in mapped:
        carried = False
        for k in sorted(max_lag):
            lag = epoch_lag(declared.rule.lag_days, calendar, k)
            if lag is None or lag > int(max_lag[k]):
                continue
            placed.setdefault(k, []).append(
                PlacedRule(
                    rule=declared.rule,
                    edge=replace(declared.edge, src_scale=k, lag=lag),
                    dst_label=declared.dst_label,
                    dst_quantity=declared.dst_quantity,
                    scale=k,
                )
            )
            carried = True
        if not carried:
            off_grid.append(declared.rule)
    return placed, off_axis, off_grid


# ---------------------------------------------------------------------------
# What one learned graph came back with
# ---------------------------------------------------------------------------


def scale_recall(
    graph: TypedGraph, placed: Sequence[PlacedRule], quantity: str
) -> Dict[str, object]:
    """
    The placed rules one scale of a learned graph holds, against the ones it could.

    A rule driving a quantity the fit never formed a design for sits in
    `off_quantity` and outside the denominator, since no search over the fitted
    quantity could have placed it.  A rule is held when the graph carries any
    basis term of its relation, which `relation_terms` decides, and the terms
    that carried it travel beside the name.
    """
    scorable = [p for p in placed if p.dst_quantity == quantity]
    terms = {
        p.name: relation_terms(graph.get_parents(p.dst_label, p.dst_quantity), p.edge)
        for p in scorable
    }
    found = [p.name for p in scorable if terms[p.name]]
    return {
        "placed": [p.name for p in placed],
        "universal": [p.name for p in placed if p.universal],
        "off_quantity": [p.name for p in placed if p.dst_quantity != quantity],
        "scorable": [p.name for p in scorable],
        "found": found,
        "terms": {name: [e.basis_name for e in terms[name]] for name in found},
        "missed": [p.name for p in scorable if p.name not in found],
        "recall": (len(found) / len(scorable)) if scorable else None,
    }


def stationary_block(
    graph: TypedGraph,
    truth: DeclaredTruth,
    labels: Sequence[str],
    quantity: str,
    scale: int,
) -> Dict[str, object]:
    """
    Edges one scale put into declared-stationary labels, with its own denominator.

    A declared-stationary label is the target of no rule, so every edge into one
    is a false positive.  The denominator is the stationary labels the log
    realized, which is a different denominator from the recall block's.
    """
    index = {name: i for i, name in enumerate(labels)}
    ells = [index[name] for name in truth.stationary_labels if name in index]
    parents = {
        ell: [e for e in graph.get_parents(ell, quantity) if e.src_scale == scale]
        for ell in ells
    }
    return {
        "n_stationary": len(ells),
        "edges": sum(len(p) for p in parents.values()),
        "labels_with_parents": [labels[ell] for ell in ells if parents[ell]],
    }


# ---------------------------------------------------------------------------
# What the ledger predicted
# ---------------------------------------------------------------------------


def ledger_recall(n_epochs: int, n_labels: int, max_lag: int) -> float:
    """The recall the ledger's swept curve attaches to this many epochs."""
    n_candidates = n_candidate_edges(n_labels, max_lag)
    if n_candidates <= 0:
        return 0.0
    return float(expected_recall(float(n_epochs) / n_candidates))


def preregistered_rows() -> List[Dict[str, float]]:
    """The pre-registered prediction, recomputed from the ledger it came from."""
    return [
        {
            "subjects": float(s),
            "day_epochs": float(s * LEDGER_EPOCHS_PER_SUBJECT),
            "expected_recall": round(
                ledger_recall(
                    s * LEDGER_EPOCHS_PER_SUBJECT, LEDGER_LABELS, LEDGER_MAX_LAG
                ),
                3,
            ),
        }
        for s in LEDGER_ROWS
    ]


def preregistered_crossings() -> Dict[str, float]:
    """Subjects the ledger says each target recall needs, at its own epoch count."""
    return {
        f"recall_{target}": subjects_for_recall(
            target, LEDGER_LABELS, LEDGER_MAX_LAG, LEDGER_EPOCHS_PER_SUBJECT
        )
        for target in LEDGER_TARGETS
    }


# ---------------------------------------------------------------------------
# The roster, the window and the cohort
# ---------------------------------------------------------------------------


def persona_ladder(subjects: Sequence[str]) -> List[str]:
    """
    The order a nested subject ladder adds people in, balanced over the personas.

    Round robin over the personas, so every prefix holds as equal a share of
    each persona as its length allows and a shorter rung is a subset of every
    longer one.  A ladder that added one persona first would confound the roster
    size with the persona mix, and three of the six declared rules are scoped to
    one persona.
    """
    by_persona: Dict[str, List[str]] = {}
    for name in sorted(subjects):
        by_persona.setdefault(subject_persona(name), []).append(name)
    order: List[str] = []
    for i in range(max(len(v) for v in by_persona.values())):
        for persona in sorted(by_persona):
            if i < len(by_persona[persona]):
                order.append(by_persona[persona][i])
    return order


def week_slots(weeks: int, slot_minutes: int) -> int:
    """Finest slots in a whole number of weeks."""
    total = int(weeks) * DAYS_PER_WEEK * MINUTES_PER_DAY
    if total % int(slot_minutes):
        raise ValueError(
            f"a slot of {slot_minutes} minutes does not divide a week, so "
            f"{weeks} weeks is not a whole number of slots"
        )
    return total // int(slot_minutes)


def window_events(
    events: Sequence[ObservedEvent],
    cutoff: int,
    slot_minutes: int,
    keep: Sequence[str],
    origin: datetime,
) -> List[ObservedEvent]:
    """
    The episodes of one subject that fall whole inside the first `cutoff` slots.

    Clipping on the end keeps every kept episode whole, so a shorter horizon is
    a shorter log of the same kind and no duration is truncated into the arrays.
    """
    seconds = float(slot_minutes) * 60.0
    wanted = set(keep)
    out: List[ObservedEvent] = []
    for event in events:
        if event.label not in wanted:
            continue
        start = int((event.start - origin).total_seconds() // seconds)
        occupied = max(
            1, math.ceil((event.end - event.start).total_seconds() / seconds)
        )
        if start >= 0 and start + occupied <= cutoff:
            out.append(event)
    return out


def build_cohort(
    by_subject: Mapping[str, Sequence[ObservedEvent]],
    labels: Sequence[str],
    roster: Sequence[str],
    weeks: int,
    slot_minutes: int,
    branching: Sequence[int],
    calendar: SlotCalendar,
    origin: datetime = BENCHMARK_ORIGIN,
) -> Cohort:
    """
    One cell's cohort, watched over the first `weeks` weeks of the shared clock.

    The label axis is the whole log's, so a short cell carries a column of zeros
    where a label never fired and every cell of the sweep is scored against one
    denominator.  Every subject arrives through the reader and none of them is
    held, which is what keeps a 30-subject cell at one subject's finest arrays.
    """
    cutoff = week_slots(weeks, slot_minutes)
    span = 1
    for b in branching:
        span *= int(b)
    if cutoff % span:
        raise ValueError(
            f"{weeks} weeks is {cutoff} slots, which is not a whole number of "
            f"coarsest epochs of {span} slots, so the window would half-cover one"
        )

    def reader(person: str):
        return build_subject_data(
            window_events(by_subject[person], cutoff, slot_minutes, labels, origin),
            labels,
            IDENTIFIABLE,
            slot_minutes=slot_minutes,
            branching=branching,
            origin=origin,
            horizon_slots=cutoff,
            study_slots=cutoff,
        )[0]

    return Cohort(
        subjects=tuple(sorted(roster)),
        hierarchy=ScaleHierarchy.from_uniform_branching(cutoff, list(branching)),
        label_names=list(labels),
        reader=reader,
        masks=np.ones((len(roster), cutoff), dtype=bool),
        calendar=calendar,
    )


# ---------------------------------------------------------------------------
# The fit, taken as a parameter
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellRequest:
    """
    Everything one cell's fit is handed, so a stub can stand in for it.

    `price` is a declared price, and None asks the fit to read one off the null
    it measures on this cell's own roster.  `measured` is what the fit writes
    back about the price it used, so the record can carry a number nobody
    reading the results has to reconstruct.
    """

    cohort: Cohort
    candidates: Mapping[int, Sequence[TypedEdge]]
    lag_budget: Mapping[int, int]
    quantity: str
    price: Optional[float]
    iters: int
    max_edges: int
    price_quantile: float = 0.0
    measured: Dict[str, object] = field(default_factory=dict)


CellFit = Callable[[CellRequest], TypedGraph]


def sweep_quantile(n_candidates: int) -> float:
    """
    The quantile of the null a price is read off, from the candidate space itself.

    The search takes the largest gain over the candidates of one target, so a bar
    at the `1 - 1 / C` quantile is the gain about one of those C reaches carrying
    nothing about the target.  Read that way the price is a statement about the
    space being swept and never a number chosen to make a result come out.

    Raises:
        ValueError: when the candidate space is empty, since a bar over nothing
            names no rate.
    """
    if n_candidates < 1:
        raise ValueError(
            "a price is a quantile of what one sweep of the candidate space "
            "reaches, and this space holds no candidates"
        )
    return 1.0 - 1.0 / float(n_candidates)


def source_bins(
    cohort: Cohort, scales: Sequence[int], n_bins: int
) -> Dict[Tuple[int, int, str], np.ndarray]:
    """
    Bin edges of every source column, at the quantiles of the rows this cell fits.

    The cut points come from the cohort the search is taken over and from no
    declaration, so no particular threshold is anticipated: a source is cut at
    its own quantiles and a truth that steps anywhere inside its range lands
    between two bands.  A column whose quantiles sit on top of each other
    collapses to fewer bands, which `quantile_edges` does by dropping a cut that
    leaves an empty bin.

    Args:
        cohort: the roster the bins are measured over, streamed one subject at
            a time.
        scales: the source scales the candidate space reads.
        n_bins: bins per source, so one log term and up to n_bins - 1 bands.

    Returns:
        The bin edges of each (scale, source label, source quantity).
    """
    stacked: Dict[Tuple[int, str], List[np.ndarray]] = {}
    for name in cohort.subjects:
        data = cohort.open(name)
        for k in scales:
            stacked.setdefault((k, COUNT), []).append(
                np.asarray(data.get_X(k), dtype=float)
            )
            stacked.setdefault((k, DURATION), []).append(
                np.asarray(data.get_Y(k), dtype=float)
            )
    out: Dict[Tuple[int, int, str], np.ndarray] = {}
    for (k, quantity), parts in stacked.items():
        values = np.concatenate(parts, axis=0)
        for src in range(values.shape[1]):
            out[(k, src, quantity)] = quantile_edges(values[:, src], n_bins)
    return out


def candidate_space(
    n_labels: int,
    scales: Sequence[int],
    max_lag: int,
    bins: Optional[Mapping[Tuple[int, int, str], np.ndarray]] = None,
) -> Dict[int, List[TypedEdge]]:
    """
    The typed edges the search enumerates for every target, one per basis term.

    Both source quantities are admitted at every candidate scale, since a
    declared rule reads a count in three cases and a duration in three, and the
    search has no way to know which before it scores them.  Each of those
    relations then offers the terms of its source's basis: the log term, which
    carries a source acting monotonically, and one band per bin above the
    reference, where a source acting as a step at some threshold shows up as the
    step between two adjacent bands.  A relation whose source has no bins offers
    the log term alone, which is the plain typed edge.

    Args:
        bins: the bin edges of each (scale, source label, source quantity), from
            `source_bins`; None enumerates the log term of every relation.
    """
    table = dict(bins or {})
    pool = [
        TypedEdge(OBSERVED, int(k), src, quantity, lag, basis)
        for k in scales
        for src in range(n_labels)
        for quantity in (COUNT, DURATION)
        for lag in range(1, int(max_lag) + 1)
        for basis in basis_terms(table.get((int(k), src, quantity), _NO_BINS))
    ]
    return {target: list(pool) for target in range(n_labels)}


def pooled_graph(request: CellRequest) -> TypedGraph:
    """
    One cell's graph, from the pooled fit over the whole cohort.

    The coefficient rounds run first, so the search scores its candidates at
    weights the cohort agrees on, and one more round writes the coefficients of
    whatever the search accepted.  They run until the coefficients stop moving,
    since a gain is a statistic about a fitted base and a base short of its
    maximizer prices every candidate against the wrong curvature: `iters` is the
    budget and `_COEF_TOL` is when a round has bought nothing.  The price is read
    off the null measured at those same weights whenever the caller declared
    none.  No latent chain is carried; what pooling buys a regime is a separate
    measurement, and a chain here would move the recall this arm reports for a
    reason the arm does not sweep.
    """
    graph = TypedGraph(label_names=list(request.cohort.label_names))
    fit = PooledFit(
        graph,
        FineParams(n_labels=request.cohort.L),
        [],
        request.lag_budget,
        quantity=request.quantity,
        max_sweeps=1,
    )
    held: Dict[int, np.ndarray] = {}
    for _ in range(max(int(request.iters), 1)):
        beta = fit.solve(fit.accumulate(request.cohort, step=False))
        moved = max(
            (float(np.max(np.abs(beta[t] - held[t]))) for t in held), default=np.inf
        )
        held = beta
        if moved < _COEF_TOL:
            break
    # The coefficient rounds read no candidate, so the pool is handed over after
    # them.  Formed inside those rounds it would cost the whole candidate space
    # once per round and be thrown away each time.
    fit.candidates = {int(t): list(e) for t, e in dict(request.candidates).items()}
    price = _cell_price(fit, request)
    fit.search(request.cohort, price, max_edges=request.max_edges)
    fit.solve(fit.accumulate(request.cohort, step=False))
    return graph


def _cell_price(fit: PooledFit, request: CellRequest) -> float:
    """
    The declared price, or the one this cell's own null puts at the quantile.

    Raises:
        ValueError: when neither a price nor a quantile to read one at was given,
            since the lowest gain the null reaches is a bar every candidate
            clears.
    """
    if request.price is not None:
        request.measured.update(price=float(request.price), quantile=None)
        return float(request.price)
    quantile = float(request.price_quantile)
    if not 0.0 < quantile < 1.0:
        raise ValueError(
            f"a cell with no declared price reads one off a quantile of its own "
            f"null, and {quantile} is not one"
        )
    nulls = fit.null_gains(request.cohort)
    price = float(np.quantile(nulls, quantile))
    request.measured.update(
        price=price,
        quantile=quantile,
        null_draws=int(nulls.size),
        null_median=float(np.median(nulls)),
        null_max=float(nulls.max()),
        cleared=float((nulls > price).mean()),
    )
    return price


# ---------------------------------------------------------------------------
# The cells
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellPlan:
    """One cell of the sweep, before anything is fitted."""

    arm: str
    subjects: Tuple[str, ...]
    weeks: int

    @property
    def n_subjects(self) -> int:
        return len(self.subjects)

    @property
    def personas(self) -> Tuple[str, ...]:
        return tuple(sorted({subject_persona(name) for name in self.subjects}))

    @property
    def subject_weeks(self) -> int:
        return self.n_subjects * int(self.weeks)

    @property
    def tag(self) -> str:
        """The name the cell is printed and keyed under."""
        if self.n_subjects == 1:
            return f"S=1 W={self.weeks} {self.subjects[0]}"
        return f"S={self.n_subjects} W={self.weeks}"


def cell_claim(plan: CellPlan) -> str:
    """What one cell's numbers are a statement about, in words."""
    if plan.n_subjects == 1:
        return (
            f"n-of-1: {plan.subjects[0]} alone over {plan.weeks} weeks, so every "
            "number here is one person's own history and none of it is a "
            "statement about a population"
        )
    return (
        f"population: {plan.n_subjects} subjects over {len(plan.personas)} "
        f"personas and {plan.weeks} weeks, so the graph is a statement about the "
        "cohort and about no single person's history"
    )


def plan_cells(
    ladder: Sequence[str],
    subject_arm: Sequence[int],
    horizon_arm: Sequence[int],
    weeks: int,
    n_personas: int,
) -> List[CellPlan]:
    """
    The two arms as cells, one cross and never a grid.

    The subject arm holds the horizon and sweeps the roster; its S=1 rung is one
    cell per persona.  The horizon arm holds the roster at its largest rung and
    sweeps the weeks, and the cell the two arms share is planned once.

    Raises:
        ValueError: when a rung asks for more subjects than the roster holds.
    """
    plans: List[CellPlan] = []
    for s in subject_arm:
        if s > len(ladder):
            raise ValueError(
                f"the subject arm asks for {s} subjects and the roster holds "
                f"{len(ladder)}"
            )
        if s == 1:
            plans.extend(
                CellPlan(SUBJECT_ARM, (name,), weeks) for name in ladder[:n_personas]
            )
        else:
            plans.append(CellPlan(SUBJECT_ARM, tuple(sorted(ladder[:s])), weeks))
    if not horizon_arm:
        return plans
    # Every rung of the subject arm has already been checked against the roster,
    # so the largest of them is a roster the ladder holds.
    largest = max(subject_arm) if subject_arm else len(ladder)
    roster = tuple(sorted(ladder[:largest]))
    for w in horizon_arm:
        if w == weeks and any(p.subjects == roster and p.weeks == w for p in plans):
            continue
        plans.append(CellPlan(HORIZON_ARM, roster, w))
    return plans


def estimate_cost(
    plans: Sequence[CellPlan], iters: int, workers: int, n_targets: int = 0
) -> Dict[str, float]:
    """
    What the planned sweep costs, in cells, in subject-weeks and in wall clock.

    A cell cannot be split across workers, so the parallel estimate is floored
    by the largest cell however many workers are free.

    `cohort_passes_per_cell` is the second half of the price and the reason the
    wall clock moves with the label axis as well as with the roster.  A cell
    streams the cohort once to cut the bins its basis is built on, once per
    coefficient round, once to measure the null its price is read off, once to
    open the search, once per move a target actually makes, and once more to
    write the coefficients the search accepted.  Two moves per target is the
    planning figure: a round that buys one edge and has one more turned away.
    The coefficient rounds stop once they stop moving, so `iters` bounds them
    and the figure here is the bound.
    """
    per = [p.subject_weeks * int(iters) * SECONDS_PER_SUBJECT_WEEK for p in plans]
    total = float(sum(per))
    unit = max((p.subject_weeks for p in plans), default=0)
    return {
        "n_cells": len(plans),
        "subject_weeks": sum(p.subject_weeks for p in plans),
        "largest_cell_subject_weeks": unit,
        "full_cohort_equivalents": (
            round(sum(p.subject_weeks for p in plans) / unit, 2) if unit else 0.0
        ),
        "serial_hours": round(total / 3600.0, 2),
        "parallel_hours": round(
            max(max(per, default=0.0), total / workers) / 3600.0, 2
        ),
        "workers": int(workers),
        "cohort_passes_per_cell": int(iters) + 4 + 2 * int(n_targets),
    }


# ---------------------------------------------------------------------------
# The record one cell leaves
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellRecord:
    """
    One fitted cell, which cannot be recorded without saying what it is about.

    Raises:
        ValueError: when the claim is empty, since a recall number with no claim
            beside it reads as a population result whatever it was fitted on.
    """

    plan: CellPlan
    claim: str
    day_epochs: int
    measured: Dict[int, Dict[str, object]]
    stationary: Dict[int, Dict[str, object]]
    predicted: Dict[int, float]
    verdict: Dict[str, object]
    holds_key: Optional[bool]
    price: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.claim).strip():
            raise ValueError(
                "a cell record carries the claim its numbers support; without one "
                "a recall taken on a single subject reads as a population result"
            )

    @property
    def day_recall(self) -> Optional[float]:
        """The measured recall at the day scale, which is the arm's headline."""
        for k in sorted(self.measured):
            if self.measured[k]["is_day_scale"]:
                return self.measured[k]["recall"]
        return None

    def to_dict(self) -> Dict[str, object]:
        return {
            "arm": self.plan.arm,
            "tag": self.plan.tag,
            "subjects": list(self.plan.subjects),
            "personas": list(self.plan.personas),
            "n_subjects": self.plan.n_subjects,
            "weeks": self.plan.weeks,
            "subject_weeks": self.plan.subject_weeks,
            "claim": self.claim,
            "day_epochs": self.day_epochs,
            "measured_recall": self.day_recall,
            "predicted_recall": self.predicted,
            "ledger_verdict": self.verdict,
            "recall": {str(k): v for k, v in self.measured.items()},
            "stationary": {str(k): v for k, v in self.stationary.items()},
            "holds_key_rule": self.holds_key,
            "price": dict(self.price),
        }


def measure_cell(
    plan: CellPlan,
    graph: TypedGraph,
    placed: Mapping[int, Sequence[PlacedRule]],
    truth: DeclaredTruth,
    labels: Sequence[str],
    quantity: str,
    max_lag: int,
    day_scale: int,
    epochs_per_week: Mapping[int, int],
    key_rule: str,
    price: Optional[Mapping[str, object]] = None,
) -> CellRecord:
    """Score one fitted graph and record it beside what the ledger predicted."""
    measured: Dict[int, Dict[str, object]] = {}
    stationary: Dict[int, Dict[str, object]] = {}
    predicted: Dict[int, float] = {}
    for k in sorted(placed):
        block = scale_recall(graph, placed[k], quantity)
        block["is_day_scale"] = k == day_scale
        block["epochs"] = plan.n_subjects * epochs_per_week[k] * plan.weeks
        measured[k] = block
        stationary[k] = stationary_block(graph, truth, labels, quantity, k)
        predicted[k] = round(
            ledger_recall(int(block["epochs"]), len(labels), max_lag), 4
        )

    day_epochs = plan.n_subjects * epochs_per_week.get(day_scale, 0) * plan.weeks
    gate = scale_verdict(day_scale, day_epochs, len(labels), max_lag)
    key = next((p for k in placed for p in placed[k] if p.key == key_rule), None)
    holds_key = (
        None
        if key is None
        else bool(
            relation_terms(graph.get_parents(key.dst_label, key.dst_quantity), key.edge)
        )
    )
    return CellRecord(
        plan=plan,
        claim=cell_claim(plan),
        day_epochs=day_epochs,
        measured=measured,
        stationary=stationary,
        predicted=predicted,
        verdict={
            "epochs_per_candidate": round(gate.epochs_per_candidate, 4),
            "n_candidates": gate.n_candidates,
            "supported": gate.supported,
            "reason": gate.reason,
        },
        holds_key=holds_key,
        price=dict(price or {}),
    )


# ---------------------------------------------------------------------------
# Where the acceptance rule crosses
# ---------------------------------------------------------------------------


def crossing(records: Sequence[CellRecord], weeks: int) -> Dict[str, object]:
    """
    Where the acceptance rule enters the learned graph as the roster grows.

    Two numbers, because they can differ.  The first rung that holds the rule
    says when it became findable at all.  The first rung from which every larger
    one holds it says when it became reliable, and a gap between the two is a
    rule that flickered.
    """
    rungs: Dict[int, List[Optional[bool]]] = {}
    for record in records:
        if record.plan.arm != SUBJECT_ARM or record.plan.weeks != weeks:
            continue
        rungs.setdefault(record.plan.n_subjects, []).append(record.holds_key)
    sizes = sorted(rungs)
    held = {s: [bool(v) for v in rungs[s] if v is not None] for s in sizes}
    any_hold = [s for s in sizes if any(held[s])]
    all_hold = [s for s in sizes if held[s] and all(held[s])]
    stable = next(
        (s for s in sizes if all(t in all_hold for t in sizes if t >= s)), None
    )
    first = any_hold[0] if any_hold else None
    at_one = held.get(1, [])

    half = None
    for s in sizes:
        value = _rung_recall(records, s, weeks)
        if value is not None and value >= 0.5:
            half = s
            break

    return {
        "rungs": {
            str(s): {"cells": len(rungs[s]), "holding": sum(held[s])} for s in sizes
        },
        "first_hold": first,
        "stable_hold": stable,
        "first_and_stable_differ": first != stable,
        "n_of_1_cells": len(at_one),
        "n_of_1_holding": sum(at_one),
        "recall_half": half,
    }


def _rung_recall(
    records: Sequence[CellRecord], n_subjects: int, weeks: int
) -> Optional[float]:
    """The mean day-scale recall of the cells at one rung of the subject arm."""
    values = [
        r.day_recall
        for r in records
        if r.plan.arm == SUBJECT_ARM
        and r.plan.weeks == weeks
        and r.plan.n_subjects == n_subjects
        and r.day_recall is not None
    ]
    return sum(values) / len(values) if values else None


def population_statement(cross: Mapping[str, object], key_rule: str) -> str:
    """One sentence on whether the acceptance rule is a population result."""
    if cross["n_of_1_cells"] == 0:
        return (
            f"{key_rule} was never fitted at one subject, so this sweep says "
            "nothing about whether one person's history carries it"
        )
    if cross["n_of_1_holding"]:
        return (
            f"{key_rule} was recovered by {cross['n_of_1_holding']} of "
            f"{cross['n_of_1_cells']} one-subject fits, so it is recoverable "
            "within one person and pooling is not what places it"
        )
    if cross["first_hold"] is None:
        return (
            f"{key_rule} was recovered at no rung of this arm, so the arm "
            "separates nothing and the roster sizes it swept were all too small"
        )
    return (
        f"{key_rule} was recovered by none of the {cross['n_of_1_cells']} "
        f"one-subject fits and first appears at {cross['first_hold']} subjects, "
        "so it is a population claim and no single person's history carries it"
    )


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def _print_cell(record: CellRecord) -> None:
    """Print one cell, one block per claim, in the shape exp6 prints."""
    print(
        f"  [{record.plan.arm}] {record.plan.tag}, {record.plan.subject_weeks} subject-weeks"
    )
    print(f"    claim: {record.claim}")
    if record.price:
        print("    price: " + ", ".join(f"{k} {v}" for k, v in record.price.items()))
    for k in sorted(record.measured):
        block = record.measured[k]
        measured = block["recall"]
        shown = "none placed" if measured is None else f"{measured:.3f}"
        print(
            f"    k={k}: recall {len(block['found'])}/{len(block['scorable'])} "
            f"measured {shown}  ledger {record.predicted[k]:.3f}  "
            f"{block['epochs']} epochs"
        )
        if block["found"]:
            # The term the search expressed a rule in says whether the source
            # acted as a step or as a smooth one, so it is printed beside it.
            print(
                "      found: "
                + ", ".join(
                    f"{name} as {' '.join(block['terms'][name])}"
                    for name in block["found"]
                )
            )
        if block["missed"]:
            print("      missed: " + ", ".join(block["missed"]))
        if block["off_quantity"]:
            print("      not scored here: " + ", ".join(block["off_quantity"]))
        control = record.stationary[k]
        print(
            f"      {control['edges']} edges into "
            f"{len(control['labels_with_parents'])} of {control['n_stationary']} "
            "declared-stationary labels"
        )
    print(f"    ledger gate: {record.verdict['reason']}")


def _print_plan(plans: Sequence[CellPlan], cost: Mapping[str, object]) -> None:
    """Print the cell list and what it would cost, which is what a dry run is."""
    print("  cells")
    for plan in plans:
        print(
            f"    [{plan.arm}] {plan.tag}: {plan.n_subjects} subjects, "
            f"{plan.weeks} weeks, {plan.subject_weeks} subject-weeks"
        )
    print(
        f"  {cost['n_cells']} fits, {cost['subject_weeks']} subject-weeks, "
        f"{cost['full_cohort_equivalents']} times the largest cell"
    )
    print(
        f"  about {cost['serial_hours']} h serial and {cost['parallel_hours']} h "
        f"on {cost['workers']} workers, floored by the largest cell"
    )
    print(f"  {cost['cohort_passes_per_cell']} cohort passes per cell")


def main(
    csv: str = "data/dynamic_patterns/event_log.tsv",
    manifest: str = "data/dynamic_patterns/manifest_declared.json",
    slot_minutes: int = 1,
    branching: str = "60,24,7",
    weeks: int = DEFAULT_WEEKS,
    subject_arm: str = DEFAULT_SUBJECT_ARM,
    horizon_arm: str = DEFAULT_HORIZON_ARM,
    max_lag: int = 2,
    price: Optional[float] = None,
    iters: int = 20,
    max_edges: int = 8,
    max_labels: int = 0,
    basis_bins: int = DEFAULT_BASIS_BINS,
    workers: int = 4,
    key_rule: str = KEY_RULE,
    dry_run: bool = False,
    output_dir: str = "results/exp11",
    fit: Optional[CellFit] = None,
) -> Dict[str, object]:
    branch = [int(b) for b in branching.split(",") if b.strip()]
    arm_s = [int(s) for s in subject_arm.split(",") if s.strip()]
    arm_w = [int(w) for w in horizon_arm.split(",") if w.strip()]
    calendar = SlotCalendar.from_config(
        branch, slot_minutes=slot_minutes, origin=BENCHMARK_ORIGIN
    )
    K = len(branch) + 1

    by_subject = _read_log(csv)
    truth = read_declared_manifest(manifest)
    labels = _label_axis(by_subject, truth, max_labels)
    ladder = persona_ladder(by_subject)
    personas = sorted({subject_persona(name) for name in ladder})

    # The candidate space is the day scale and every scale above it.  A minute
    # or hour candidate multiplies the search term by about three and costs
    # 232 MB of resident columns, and no declared rule reads a lag under a day.
    scales = [
        k for k in range(1, K + 1) if calendar.epoch_minutes(k) >= MINUTES_PER_DAY
    ]
    day_scale = _day_scale(calendar, K)
    epochs_per_week = {
        k: DAYS_PER_WEEK * MINUTES_PER_DAY // calendar.epoch_minutes(k) for k in scales
    }
    limits = {k: int(max_lag) for k in scales}
    placed, off_axis, off_grid = placed_typed_rules(truth, labels, calendar, limits)

    plans = plan_cells(ladder, arm_s, arm_w, weeks, len(personas))
    cost = estimate_cost(plans, iters, workers, len(labels))

    print(f"[Exp11] pooling curve on {csv}")
    print(
        f"  {len(by_subject)} subjects over {len(personas)} personas, "
        f"{len(labels)} labels, K={K}, day scale k={day_scale}"
    )
    print(
        f"  {sum(len(v) for v in placed.values())} placed rules "
        f"at scales {sorted(placed)}, {len(off_axis)} naming a label the log "
        f"never realized, {len(off_grid)} whose lag no scale reaches"
    )
    relations = candidate_space(len(labels), scales, max_lag)
    print(
        f"  candidate space {len(relations[0])} typed relations per target over "
        f"scales {scales}; the ledger's denominator is "
        f"{n_candidate_edges(len(labels), max_lag)}"
    )
    print(
        f"  each relation offers the terms of a {basis_bins}-bin basis cut on the "
        "cell's own rows, so every cell counts its own candidates"
    )
    if price is None:
        print(
            "  price per cell at the 1 - 1/C quantile of its own null, which is "
            "the gain one of its C candidates reaches carrying nothing"
        )
    else:
        print(f"  price {price} declared, so no null is measured")
    print("  pre-registered prediction at 91 day epochs per subject")
    for row in preregistered_rows():
        print(
            f"    {int(row['subjects']):>2} subjects  "
            f"{int(row['day_epochs']):>5} epochs  recall {row['expected_recall']:.3f}"
        )
    crossings = preregistered_crossings()
    print("    " + "; ".join(_crossing_line(t, crossings) for t in LEDGER_TARGETS))
    _print_plan(plans, cost)

    if dry_run:
        return {
            "csv": csv,
            "dry_run": True,
            "labels": labels,
            "cells": [
                {
                    "arm": p.arm,
                    "tag": p.tag,
                    "n_subjects": p.n_subjects,
                    "weeks": p.weeks,
                    "subject_weeks": p.subject_weeks,
                    "claim": cell_claim(p),
                }
                for p in plans
            ],
            "cost": cost,
        }

    os.makedirs(output_dir, exist_ok=True)
    write_provenance(output_dir)
    run = pooled_graph if fit is None else fit
    budget = dict(limits)
    records: List[CellRecord] = []
    for plan in plans:
        cohort = build_cohort(
            by_subject,
            labels,
            plan.subjects,
            plan.weeks,
            slot_minutes,
            branch,
            calendar,
        )
        # The bins are cut on the rows this cell fits, so a band the search may
        # buy is a band of the record it is scored on.
        pool = candidate_space(
            len(labels), scales, max_lag, source_bins(cohort, scales, basis_bins)
        )
        request = CellRequest(
            cohort=cohort,
            candidates=pool,
            lag_budget=budget,
            quantity=COUNT,
            price=price,
            iters=iters,
            max_edges=max_edges,
            price_quantile=sweep_quantile(len(pool[0])),
            measured={"n_candidates": len(pool[0])},
        )
        graph = run(request)
        record = measure_cell(
            plan,
            graph,
            placed,
            truth,
            labels,
            COUNT,
            max_lag,
            day_scale,
            epochs_per_week,
            key_rule,
            request.measured,
        )
        records.append(record)
        _print_cell(record)

    cross = crossing(records, weeks)
    statement = population_statement(cross, key_rule)
    print(f"  acceptance rule {key_rule}")
    print(
        f"    first held at {cross['first_hold']} subjects, held from "
        f"{cross['stable_hold']} up, the two differ: "
        f"{cross['first_and_stable_differ']}"
    )
    print(f"    measured recall reaches 0.5 at {cross['recall_half']} subjects")
    print(f"    {statement}")

    out = {
        "csv": csv,
        "manifest": manifest,
        "slot_minutes": slot_minutes,
        "branching": branch,
        "weeks": weeks,
        "subject_arm": arm_s,
        "horizon_arm": arm_w,
        "max_lag": max_lag,
        "price": price,
        "iters": iters,
        "labels": labels,
        "personas": personas,
        "day_scale": day_scale,
        "candidate_scales": scales,
        "basis_bins": basis_bins,
        "n_relations_per_target": len(relations[0]),
        "n_candidates_per_target": {
            r.plan.tag: r.price.get("n_candidates") for r in records
        },
        "n_candidates_ledger": n_candidate_edges(len(labels), max_lag),
        "placed": {str(k): [p.name for p in placed[k]] for k in sorted(placed)},
        "off_axis": [f"{r.source}|{r.target}|{r.lag_days}d" for r in off_axis],
        "off_grid": [f"{r.source}|{r.target}|{r.lag_days}d" for r in off_grid],
        "preregistered": preregistered_rows(),
        "preregistered_crossings": {
            k: (None if math.isinf(v) else v) for k, v in crossings.items()
        },
        "cells": [r.to_dict() for r in records],
        "crossing": cross,
        "population_vs_n_of_1": statement,
        "cost": cost,
    }
    path = os.path.join(output_dir, "exp11_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {output_dir}/")
    return out


def _label_axis(
    by_subject: Mapping[str, Sequence[ObservedEvent]],
    truth: DeclaredTruth,
    max_labels: int,
) -> List[str]:
    """
    The label axis every cell shares, optionally cut down to a smaller run.

    A cut keeps every label a declared rule names, since dropping one would
    remove a rule from the comparison without saying so, and fills the rest in
    sorted order.
    """
    realized = sorted({e.label for events in by_subject.values() for e in events})
    if max_labels <= 0 or max_labels >= len(realized):
        return realized
    named = {r.source for r in truth.rules} | {r.target for r in truth.rules}
    keep = [name for name in realized if name in named]
    if len(keep) > max_labels:
        raise ValueError(
            f"a label axis of {max_labels} cannot hold the {len(keep)} labels the "
            "declared rules name, so the cut would drop a rule in silence"
        )
    rest = [name for name in realized if name not in named]
    return sorted(keep + rest[: max_labels - len(keep)])


def _crossing_line(target: float, crossings: Mapping[str, float]) -> str:
    """One pre-registered crossing as text, naming a target the sweep never reached."""
    subjects = crossings[f"recall_{target}"]
    if math.isinf(subjects):
        return f"recall {target} is out of reach of the swept curve"
    return f"recall {target} at {subjects:.2f} subjects"


def _read_log(path: str) -> Dict[str, List[ObservedEvent]]:
    """Every subject's episodes, from a log whose header states its separator."""
    with open(path, encoding="utf-8") as f:
        header = f.readline()
    return read_subject_event_log(path, delimiter="\t" if "\t" in header else ",")


def _day_scale(calendar: SlotCalendar, K: int) -> int:
    """
    The scale whose epoch is one day, which every declared lag is stated in.

    Raises:
        ValueError: when no scale of the ladder spans a day.
    """
    for k in range(1, K + 1):
        if calendar.epoch_minutes(k) == MINUTES_PER_DAY:
            return k
    raise ValueError(
        f"no scale of a {K}-scale ladder on {calendar.slot_minutes}-minute slots "
        "spans a day, and every declared lag is stated in whole days"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/dynamic_patterns/event_log.tsv")
    parser.add_argument(
        "--manifest", default="data/dynamic_patterns/manifest_declared.json"
    )
    parser.add_argument("--slot-minutes", type=int, default=1)
    parser.add_argument("--branching", default="60,24,7")
    parser.add_argument("--weeks", type=int, default=DEFAULT_WEEKS)
    parser.add_argument("--subject-arm", default=DEFAULT_SUBJECT_ARM)
    parser.add_argument("--horizon-arm", default=DEFAULT_HORIZON_ARM)
    parser.add_argument("--max-lag", type=int, default=2)
    parser.add_argument(
        "--price",
        type=float,
        default=None,
        help="a declared price; left off, each cell reads one off its own null",
    )
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--max-edges", type=int, default=8)
    parser.add_argument(
        "--max-labels",
        type=int,
        default=0,
        help="cut the shared label axis to this many labels for a smaller run",
    )
    parser.add_argument(
        "--basis-bins",
        type=int,
        default=DEFAULT_BASIS_BINS,
        help="bins per source, so one log term and up to that many bands less one",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--key-rule", default=KEY_RULE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default="results/exp11")
    args = parser.parse_args()
    main(
        csv=args.csv,
        manifest=args.manifest,
        slot_minutes=args.slot_minutes,
        branching=args.branching,
        weeks=args.weeks,
        subject_arm=args.subject_arm,
        horizon_arm=args.horizon_arm,
        max_lag=args.max_lag,
        price=args.price,
        iters=args.iters,
        max_edges=args.max_edges,
        max_labels=args.max_labels,
        basis_bins=args.basis_bins,
        workers=args.workers,
        key_rule=args.key_rule,
        dry_run=args.dry_run,
        output_dir=args.output_dir,
    )
