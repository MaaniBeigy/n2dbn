"""
experiments/exp12_fidelity.py
=============================
Experiment 12: does a log drawn from the fitted model hold the distribution of
the log it was fitted to.

Every other experiment on this benchmark asks what the fit recovered, an edge, a
coefficient, a regime.  This one asks the question the model exists to answer.
A simulated log is worth having when its distribution matches the record's at
every scale and in every calendar cell.  The timestamps themselves are free.

What is fitted
--------------
One model per subject, over the twelve-week window every subject shares and the
whole sixty-label axis, with four calendar nodes as the only parents: the hour
of the day, the day of the week, the weekend and the week index.  Both
predictors carry all four, so a count effect and a duration effect are each
fitted where the benchmark declares one.

Both halves are fitted at the maximum of the likelihood `intensity` states.  The
start coefficients maximize `start_loglik` over the risk set by Newton steps.
Every calendar column is constant inside an hour, so that term is a binomial
term over the hours of the record and the whole solve runs on two thousand rows.
The duration marks maximize the negative binomial `mark_loglik` states, over the
starts of each label.

The quadratic bound `fine_mstep` maximizes is a lower bound on the same start
term and aims at the same coefficients.  Its curvature at a log-odds of -5.5,
which is what a label of a few hundred episodes over 120,960 slots carries, is
twenty times the curvature of the term itself, so each re-anchoring moves a
twentieth of the way and hundreds of sweeps stand between a cold start and the
maximum.  A fit short of the maximum draws at the wrong rate, and the rate is
the whole question here, so the exact maximizer is what the replicates read.

The weekend indicator is the sum of the Saturday and Sunday indicators of the
day-of-week node, so the four nodes place one redundant direction.  A column
that is already a combination of the columns before it carries no direction of
its own, so it is left out and its coefficient stays zero; the weekend column is
the one that goes and the two day-of-week columns carry the weekend effect.

No structure is searched.  A carry-over edge is what the pooling experiment
measures.  Here the graph is declared and the question is entirely
distributional.

What is checked
---------------
`evaluation.fidelity` draws replicates on the fit's own grid and places the
observation in each cell's replicate spread.  Five things come back: the
calibration over every label-by-cell combination, the verdict over the families
those cells fall into, the benchmark's own four questions as cohort tables, the
cells whose observation sits beyond every replicate, and the reconciliation
identity, which is asserted on the record and on every replicate.

The calibration answers a question about the average cell, so a break confined
to one split of one label moves it by a thousandth.  The verdict sums each
family's standardized cells into one discrepancy and reports the family standing
furthest out, against the same figure taken on every replicate, so a break in
one family is read against its own siblings.

Run
---
    python -m experiments.exp12_fidelity --dry-run

    python -m experiments.exp12_fidelity
        --csv data/dynamic_patterns/event_log.tsv
        --subjects 8 --replicates 80 --workers 8
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import get_context
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import digamma, gammaln

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.core.calendar import SlotCalendar
from src.core.features import (
    CalendarBlocks,
    CalendarNode,
    calendar_blocks,
    day_of_week,
    horizon_trend,
    hour_of_day,
    weekend,
)
from src.core.typed_graph import CALENDAR, COUNT, DURATION, LEVEL, TypedEdge, TypedGraph
from src.data.dynamic_patterns import (
    BENCHMARK_BRANCHING,
    BENCHMARK_ORIGIN,
    BENCHMARK_SLOT_MINUTES,
    IDENTIFIABLE,
    build_subject_data,
    read_subject_event_log,
    subject_persona,
)
from src.data.event_log import ObservedEvent
from src.evaluation.fidelity import (
    CellCheck,
    CellGrid,
    CellKey,
    FamilyCheck,
    FidelityReport,
    Verdict,
    posterior_predictive,
)
from src.learning.fine_design import (
    CalendarIndex,
    DesignColumn,
    ScaleLadder,
    edge_columns,
)
from src.learning.scale_split import parent_map
from src.model.intensity import (
    FineParams,
    FineSources,
    ancestor_maps,
    linear_predictor,
    mark_loglik,
    start_loglik,
    usable_slots,
)
from src.utils.budget import (
    BudgetLedger,
    RunBudget,
    add_budget_arguments,
    budget_from_args,
)
from src.utils.provenance import write_provenance

MINUTES_PER_DAY = 1440
DAYS_PER_WEEK = 7

# The three time-of-day windows the benchmark declares walking is concentrated
# in, written as bins of `features.hour_of_day`, whose edges hold all three.
WALKING_WINDOWS = ("[7,10)", "[12,14)", "[17,21)")

# What this run of the benchmark realized over all thirty subjects, measured
# when the log was generated.  The cohort fitted here is a subset of those
# thirty, so its own observation is what the replicates are placed against and
# these numbers say what the whole log holds.
REALIZED = {
    "walking_week_first": 2.54,
    "walking_week_last": 3.74,
    "walking_weekday": 2.99,
    "walking_weekend": 4.43,
    "gym_minutes_first": 45.5,
    "gym_minutes_last": 65.0,
}

# The dispersion is searched in logs, floored and capped so a label whose marks
# carry no spread at all leaves the search at a finite value.
LOG_R_MIN = -6.0
LOG_R_MAX = 12.0

# The clip the start density and the sampler both apply.  A fit running past it
# would fit a predictor no draw can reach.
MAX_ETA = 30.0

# A ridge that keeps the Newton step solvable where a rare label's design is
# nearly separated, small against a weighted Gram over thousands of epochs.
RIDGE = 1e-8

# One replicate of one twelve-week subject over sixty labels costs about this
# many seconds, measured with fourteen calendar parents on both predictors.
SECONDS_PER_REPLICATE = 52.0

# Seconds one subject's fit costs, which every worker pays once.
SECONDS_PER_FIT = 90.0


# ---------------------------------------------------------------------------
# The roster and the window
# ---------------------------------------------------------------------------


def subject_ladder(subjects: Sequence[str]) -> List[str]:
    """
    The order subjects are added in, balanced over the personas.

    Round robin over the personas, so every prefix holds as equal a share of
    each persona as its length allows.  A roster that took one persona first
    would confound the cohort with the persona mix, and the benchmark scopes
    some of its declarations to a persona.
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


def clip_to_window(
    events: Sequence[ObservedEvent], cutoff: int, slot_minutes: int, origin: datetime
) -> Tuple[List[ObservedEvent], int]:
    """
    The episodes falling whole inside the first `cutoff` slots, and how many left.

    Clipping on the end keeps every kept episode whole, so the window is a
    shorter log of the same kind and no duration is truncated into the arrays.
    The record is then covered end to end, which is what lets a centered column
    carry one mean over the whole horizon.
    """
    seconds = float(slot_minutes) * 60.0
    kept: List[ObservedEvent] = []
    dropped = 0
    for event in events:
        start = int((event.start - origin).total_seconds() // seconds)
        occupied = max(
            1, math.ceil((event.end - event.start).total_seconds() / seconds)
        )
        if start >= 0 and start + occupied <= cutoff:
            kept.append(event)
        else:
            dropped += 1
    return kept, dropped


def calendar_nodes(origin: datetime) -> List[CalendarNode]:
    """The four splits the benchmark states its declarations in."""
    return [hour_of_day(), day_of_week(), weekend(), horizon_trend(origin, "week")]


# ---------------------------------------------------------------------------
# The declared design
# ---------------------------------------------------------------------------


def build_sources(
    data, calendar: SlotCalendar, nodes: Sequence[CalendarNode]
) -> Tuple[FineSources, CalendarBlocks]:
    """One subject's arrays as fine sources, with each node placed at its home scale."""
    K = data.hierarchy.K
    n_epochs = [int(data.get_X(k).shape[0]) for k in range(1, K + 1)]
    blocks = calendar_blocks(calendar, nodes, n_epochs)
    ancestors = ancestor_maps(
        [np.asarray(pm) for pm in data.hierarchy.parent_maps[:-1]], n_epochs[0]
    )
    sources = FineSources(
        counts=list(data.X),
        durations=list(data.Y),
        calendar=list(blocks.blocks),
        ancestors=ancestors,
    )
    return sources, blocks


def calendar_edges(blocks: CalendarBlocks) -> List[TypedEdge]:
    """One lag-zero edge per placed calendar column, finest scale first."""
    return [
        TypedEdge(CALENDAR, k, offset, LEVEL, 0)
        for k in range(1, len(blocks.blocks) + 1)
        for offset in range(blocks.widths[k - 1])
    ]


def independent_columns(matrix: np.ndarray, tol: float = 1e-8) -> List[int]:
    """
    The columns carrying a direction the columns before them do not.

    A column that is already a combination of its predecessors leaves the solve
    singular and its own coefficient undetermined, so it is left out and the
    predictor keeps every direction the design can hold.
    """
    columns = np.asarray(matrix, dtype=float)
    keep: List[int] = []
    basis = np.zeros((columns.shape[0], 0))
    for j in range(columns.shape[1]):
        vector = columns[:, j].copy()
        scale = float(np.linalg.norm(vector))
        if basis.shape[1]:
            vector = vector - basis @ (basis.T @ vector)
        residual = float(np.linalg.norm(vector))
        if residual > tol * max(scale, 1.0):
            keep.append(j)
            basis = np.column_stack([basis, vector / residual])
    return keep


def constant_removal(columns: Sequence[DesignColumn]) -> np.ndarray:
    """
    What centering took off each column, once per column.

    Centering removes a mean over the parent epoch, and on a record covered end
    to end over whole parents that mean is one number for the whole horizon, so
    it folds into the intercept and the fitted parameters carry no per-slot
    offset.

    Raises:
        ValueError: when a column's removed mean varies over its own grid, which
            leaves the predictor with a term the intercept cannot hold.
    """
    out = np.zeros(len(columns))
    for j, column in enumerate(columns):
        removed = np.asarray(column.removed, dtype=float)
        spread = float(removed.max() - removed.min()) if removed.size else 0.0
        if spread > 1e-9:
            raise ValueError(
                f"centering took a mean of between {removed.min():.6f} and "
                f"{removed.max():.6f} off {column.name!r}, so it is not one "
                "number the intercept can absorb"
            )
        out[j] = float(removed[0]) if removed.size else 0.0
    return out


# ---------------------------------------------------------------------------
# The start coefficients
# ---------------------------------------------------------------------------


def epoch_ancestors(ladder: ScaleLadder, scale: int) -> List[np.ndarray]:
    """For each epoch of `scale`, the epoch holding it at that scale and every coarser one."""
    out = [np.arange(ladder.n_epochs[scale - 1], dtype=np.int64)]
    for k in range(scale, ladder.K):
        out.append(parent_map(ladder, k)[out[-1]])
    return out


def design_matrix(
    columns: Sequence[DesignColumn],
    keep: Sequence[int],
    ladder: ScaleLadder,
    scale: int,
) -> np.ndarray:
    """
    The intercept and the kept columns on the grid the design varies at.

    Every calendar column is constant inside an epoch of the finest scale any of
    them varies at, so the start term over the finest slots is the same number as
    a binomial term over those epochs, and the whole solve runs on thousands of
    rows where the finest grid holds millions.
    """
    up = epoch_ancestors(ladder, scale)
    out = [np.ones(ladder.n_epochs[scale - 1])]
    for j in keep:
        column = columns[j]
        out.append(np.asarray(column.values, dtype=float)[up[column.scale - scale]])
    return np.stack(out, axis=1)


def start_counts(
    at_risk: np.ndarray, sources: FineSources, ladder: ScaleLadder, scale: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Starts and slots at risk in every epoch of the design's grid, per label.

    Returns:
        (H, L) starts, which is the count array the ladder already carries, and
        (H, L) slots the start density runs over.
    """
    n_epochs = ladder.n_epochs[scale - 1]
    who = np.asarray(ladder.ancestors[scale - 1])
    trials = np.stack(
        [
            np.bincount(who, weights=at_risk[:, ell].astype(float), minlength=n_epochs)
            for ell in range(at_risk.shape[1])
        ],
        axis=1,
    )
    return np.asarray(sources.counts[scale - 1], dtype=float), trials


def binomial_value(
    design: np.ndarray, successes: np.ndarray, trials: np.ndarray, beta: np.ndarray
) -> float:
    """The exact start term of one label, read on the design's own grid."""
    eta = np.clip(design @ beta, -MAX_ETA, MAX_ETA)
    return float(np.sum(successes * eta - trials * np.logaddexp(0.0, eta)))


def binomial_fit(
    design: np.ndarray,
    successes: np.ndarray,
    trials: np.ndarray,
    iters: int = 60,
    tol: float = 1e-10,
) -> Tuple[np.ndarray, int, float]:
    """
    One label's start coefficients, by Newton steps on the exact start term.

    The step is halved until the term itself rises, so the sequence is monotone
    and a nearly separated label leaves the search at the clip the sampler reads.

    Returns:
        The coefficients, the steps taken, and the start term at them.
    """
    p = design.shape[1]
    beta = np.zeros(p)
    total_k, total_n = float(successes.sum()), float(trials.sum())
    if total_k <= 0.0:
        beta[0] = -MAX_ETA
        return beta, 0, binomial_value(design, successes, trials, beta)
    beta[0] = math.log(total_k / max(total_n - total_k, 1e-12))
    value = binomial_value(design, successes, trials, beta)
    taken = 0
    for _ in range(int(iters)):
        eta = np.clip(design @ beta, -MAX_ETA, MAX_ETA)
        prob = 1.0 / (1.0 + np.exp(-eta))
        weights = trials * prob * (1.0 - prob)
        gradient = design.T @ (successes - trials * prob)
        hessian = design.T @ (design * weights[:, None]) + RIDGE * np.eye(p)
        step = np.linalg.solve(hessian, gradient)
        moved = False
        for _ in range(40):
            candidate = beta + step
            reached = binomial_value(design, successes, trials, candidate)
            if reached >= value:
                moved = True
                break
            step = step / 2.0
        if not moved:
            break
        taken += 1
        gain = reached - value
        beta, value = candidate, reached
        if gain < tol * (1.0 + abs(value)):
            break
    return beta, taken, value


def fit_counts(
    params: FineParams,
    graph: TypedGraph,
    edges: Sequence[TypedEdge],
    columns: Sequence[DesignColumn],
    keep: Sequence[int],
    removed: np.ndarray,
    sources: FineSources,
    ladder: ScaleLadder,
    at_risk: np.ndarray,
    scale: int,
) -> Dict[str, object]:
    """
    Every label's start coefficients, at the maximum of the exact start term.

    Returns:
        The steps the slowest label took, the start term summed over the labels,
        and how far the score equation stands from zero, which is what says the
        solve reached the maximum it was aimed at.
    """
    design = design_matrix(columns, keep, ladder, scale)
    starts, trials = start_counts(at_risk, sources, ladder, scale)
    steps: List[int] = []
    total = 0.0
    residual = 0.0
    for ell in range(sources.n_labels):
        beta, taken, value = binomial_fit(design, starts[:, ell], trials[:, ell])
        steps.append(taken)
        total += value
        eta = np.clip(design @ beta, -MAX_ETA, MAX_ETA)
        expected = float((trials[:, ell] / (1.0 + np.exp(-eta))).sum())
        observed = float(starts[:, ell].sum())
        residual = max(residual, abs(expected - observed) / max(observed, 1.0))
        level = float(beta[0])
        for slot, column in enumerate(keep, start=1):
            edge = edges[column]
            params.coef[(ell, COUNT)][edge] = float(beta[slot])
            level -= float(beta[slot]) * float(removed[column])
        params.base[COUNT][ell] = level
    return {
        "newton_steps_max": max(steps),
        "start_loglik_on_design_grid": round(total, 3),
        "score_residual_max": residual,
    }


# ---------------------------------------------------------------------------
# The duration marks
# ---------------------------------------------------------------------------


def nb_objective(
    theta: np.ndarray, design: np.ndarray, y: np.ndarray
) -> Tuple[float, np.ndarray]:
    """The negative log-likelihood of the shifted negative binomial, and its gradient."""
    p = design.shape[1]
    beta, log_r = theta[:p], float(theta[p])
    r = math.exp(log_r)
    mu = np.exp(np.clip(design @ beta, -30.0, 30.0))
    total = r + mu
    value = (
        gammaln(y + r)
        - gammaln(r)
        - gammaln(y + 1.0)
        + r * (math.log(r) - np.log(total))
        + y * (np.log(mu) - np.log(total))
    ).sum()
    d_eta = y - mu * (y + r) / total
    d_r = (
        digamma(y + r)
        - digamma(r)
        + math.log(r)
        - np.log(total)
        + 1.0
        - (y + r) / total
    )
    grad = np.concatenate([design.T @ d_eta, [r * float(d_r.sum())]])
    return -float(value), -grad


def nb_fit(design: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float, bool]:
    """
    Coefficients and dispersion of one label's marks, by maximum likelihood.

    Returns:
        The coefficient vector, the dispersion, and whether the search converged.
    """
    p = design.shape[1]
    start = np.zeros(p + 1)
    start[0] = math.log(max(float(np.mean(y)), 1e-3))
    bounds = [(-30.0, 30.0)] * p + [(LOG_R_MIN, LOG_R_MAX)]
    out = minimize(
        nb_objective,
        start,
        args=(design, y),
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500},
    )
    return out.x[:p], float(math.exp(out.x[p])), bool(out.success)


def feature_rows(
    columns: Sequence[DesignColumn], ladder: ScaleLadder, rows: np.ndarray
) -> np.ndarray:
    """The intercept and every candidate column, read at the given finest slots."""
    out = [np.ones(rows.size)]
    for column in columns:
        who = np.asarray(ladder.ancestors[column.scale - 1])
        out.append(np.asarray(column.values, dtype=float)[who[rows]])
    return np.stack(out, axis=1)


def fit_durations(
    X1: np.ndarray,
    Y1: np.ndarray,
    params: FineParams,
    graph: TypedGraph,
    edges: Sequence[TypedEdge],
    columns: Sequence[DesignColumn],
    removed: np.ndarray,
    ladder: ScaleLadder,
    labels: Sequence[str],
) -> Dict[str, object]:
    """
    Every label's mark model, fitted over its own starts, and what it left out.

    A column that never varies among one label's starts carries no contrast for
    that label, so it is left out there and its coefficient stays zero.
    """
    dropped: Dict[str, int] = {}
    failed: List[str] = []
    empty: List[str] = []
    capped: List[str] = []
    for ell, name in enumerate(labels):
        rows = np.flatnonzero(X1[:, ell] > 0)
        if rows.size == 0:
            empty.append(name)
            continue
        y = np.asarray(Y1[rows, ell], dtype=float) - 1.0
        design = feature_rows(columns, ladder, rows)
        keep = independent_columns(design)
        beta, dispersion, ok = nb_fit(design[:, keep], y)
        if not ok:
            failed.append(name)
        if dispersion >= math.exp(LOG_R_MAX) - 1.0:
            capped.append(name)
        dropped[name] = len(edges) + 1 - len(keep)
        level = float(beta[0])
        for slot, column in enumerate(keep[1:], start=1):
            edge = edges[column - 1]
            graph.add_edge(edge, ell, DURATION)
            params.coef[(ell, DURATION)][edge] = float(beta[slot])
            level -= float(beta[slot]) * float(removed[column - 1])
        params.base[DURATION][ell] = level
        params.mark_dispersion[ell] = dispersion
    return {
        "labels_without_episodes": empty,
        "labels_whose_search_stopped_early": failed,
        "labels_at_the_dispersion_cap": capped,
        "columns_dropped_max": max(dropped.values()) if dropped else 0,
        "columns_dropped_mean": (
            round(float(np.mean(list(dropped.values()))), 3) if dropped else 0.0
        ),
    }


# ---------------------------------------------------------------------------
# One subject, fitted and checked
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubjectTask:
    """Everything one worker needs to fit one subject and check its fit."""

    person: str
    events: Tuple[ObservedEvent, ...]
    labels: Tuple[str, ...]
    cutoff: int
    replicates: int
    interval: float
    min_episodes: float
    seed: int


def verdict_record(call: Verdict) -> Dict[str, object]:
    """One verdict as plain values, with the family it names."""
    worst = call.worst
    return {
        "n_families": call.n_families,
        "n_cells": call.n_cells,
        "p_value": round(call.p_value, 6),
        "nominal": call.nominal,
        "broken": call.broken,
        "worst_family": None if worst is None else worst.family,
        "worst_label": None if worst is None else worst.label,
        "worst_n_cells": None if worst is None else worst.n_cells,
        "worst_discrepancy": None if worst is None else round(worst.discrepancy, 3),
        "worst_replicate_mean": (
            None if worst is None else round(worst.replicate_mean, 3)
        ),
        "worst_low": None if worst is None else round(worst.low, 3),
        "worst_high": None if worst is None else round(worst.high, 3),
        "worst_standardized": None if worst is None else round(worst.standardized, 3),
    }


def run_subject(task: SubjectTask) -> Dict[str, object]:
    """Fit one subject's model, draw its replicates and place its observation."""
    opened = time.time()
    calendar = SlotCalendar.from_config(
        list(BENCHMARK_BRANCHING),
        slot_minutes=BENCHMARK_SLOT_MINUTES,
        origin=BENCHMARK_ORIGIN,
    )
    kept, clipped = clip_to_window(
        task.events, task.cutoff, BENCHMARK_SLOT_MINUTES, BENCHMARK_ORIGIN
    )
    data, _, counts = build_subject_data(
        kept,
        list(task.labels),
        IDENTIFIABLE,
        slot_minutes=BENCHMARK_SLOT_MINUTES,
        branching=list(BENCHMARK_BRANCHING),
        origin=BENCHMARK_ORIGIN,
        horizon_slots=task.cutoff,
        study_slots=task.cutoff,
    )
    X1, Y1 = data.get_X(1), data.get_Y(1)
    nodes = calendar_nodes(BENCHMARK_ORIGIN)
    sources, blocks = build_sources(data, calendar, nodes)
    ladder = ScaleLadder.from_sources(sources)
    index = CalendarIndex.place(nodes, calendar, sources, ladder)

    edges = calendar_edges(blocks)
    names = [index.column_name(e.src_scale, e.src_label) for e in edges]
    graph = TypedGraph(label_names=list(task.labels), calendar_names=list(names))
    columns = edge_columns(edges, graph, sources, ladder, index)
    removed = constant_removal(columns)
    scale = min(column.scale for column in columns)
    if scale < 2:
        raise ValueError(
            f"a calendar column varies at scale {scale}, the finest one, so the "
            "start term cannot be read on a grid coarser than the slots"
        )

    everywhere = design_matrix(columns, range(len(columns)), ladder, scale)
    keep = [j - 1 for j in independent_columns(everywhere) if j > 0]
    params = FineParams(n_labels=len(task.labels))
    for column in keep:
        for ell in range(len(task.labels)):
            graph.add_edge(edges[column], ell, COUNT)

    at_risk = usable_slots(X1, Y1)
    starts = fit_counts(
        params, graph, edges, columns, keep, removed, sources, ladder, at_risk, scale
    )
    marks = fit_durations(
        X1, Y1, params, graph, edges, columns, removed, ladder, list(task.labels)
    )
    eta = linear_predictor(params, graph, sources, COUNT)
    starts["start_loglik_on_slots"] = round(start_loglik(X1, eta, at_risk), 3)
    del eta, at_risk
    log_nu = linear_predictor(params, graph, sources, DURATION)
    mark_term = round(mark_loglik(X1, Y1, log_nu, params.mark_dispersion), 3)
    del log_nu
    fitted = time.time()

    grid = CellGrid.build(calendar, sources, list(task.labels))
    report = posterior_predictive(
        X1,
        Y1,
        params,
        graph,
        sources,
        grid,
        np.random.default_rng(task.seed),
        n_replicates=task.replicates,
        interval=task.interval,
    )
    summary = report.calibration(task.min_episodes)
    call = report.verdict(task.min_episodes)
    return {
        "person": task.person,
        "persona": subject_persona(task.person),
        "episodes_read": counts.read,
        "episodes_kept": counts.kept,
        "episodes_folded": counts.read - counts.kept,
        "episodes_clipped": clipped,
        "episodes_on_arrays": int(X1.sum()),
        "slots_booked": int(Y1.sum()),
        "count_columns_dropped": [names[j] for j in range(len(edges)) if j not in keep],
        "count_columns_kept": [names[j] for j in keep],
        "starts": starts,
        "mark_loglik": mark_term,
        "marks": marks,
        "dispersion_min": float(np.min(params.mark_dispersion)),
        "dispersion_max": float(np.max(params.mark_dispersion)),
        "calibration": {
            "n_cells": summary.n_cells,
            "n_sparse": summary.n_sparse,
            "nominal": summary.nominal,
            "outside_share": summary.outside_share,
            "n_misfits": summary.n_misfits,
        },
        "verdict": verdict_record(call),
        "cells": list(report.cells),
        "keys": list(grid.keys),
        "members": report.members,
        "n_reconciled": int(report.members.shape[0]),
        "seconds_fit": round(fitted - opened, 1),
        "seconds_check": round(time.time() - fitted, 1),
    }


# ---------------------------------------------------------------------------
# The four questions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Answer:
    """One statistic of the cohort: what the record holds against what it draws."""

    row: str
    observed: float
    replicate_mean: float
    low: float
    high: float
    p_value: float
    p_mid: float
    realized: Optional[float] = None

    @property
    def inside(self) -> bool:
        return self.low <= self.observed <= self.high

    def to_dict(self) -> Dict[str, object]:
        return {
            "row": self.row,
            "observed": round(self.observed, 4),
            "replicate_mean": round(self.replicate_mean, 4),
            "low": round(self.low, 4),
            "high": round(self.high, 4),
            "p_value": round(self.p_value, 4),
            "p_mid": round(self.p_mid, 4),
            "inside": self.inside,
            "realized_over_30_subjects": self.realized,
        }

    def __str__(self) -> str:
        mark = " " if self.inside else "*"
        stated = "" if self.realized is None else f"  declared run {self.realized:.2f}"
        return (
            f"{mark} {self.row:<22} observed {self.observed:8.3f}   replicates "
            f"{self.replicate_mean:8.3f} [{self.low:7.3f}, {self.high:7.3f}]   "
            f"p {self.p_value:.3f}{stated}"
        )


def place(
    row: str,
    observed: float,
    draws: np.ndarray,
    interval: float,
    realized: Optional[float] = None,
) -> Answer:
    """Where one statistic's observation falls in the spread of the replicates."""
    finite = np.asarray(draws, dtype=float)
    finite = finite[np.isfinite(finite)]
    tail = (1.0 - interval) / 2.0
    above = float((finite > observed).mean())
    equal = float((finite == observed).mean())
    return Answer(
        row=row,
        observed=float(observed),
        replicate_mean=float(finite.mean()),
        low=float(np.quantile(finite, tail)),
        high=float(np.quantile(finite, 1.0 - tail)),
        p_value=above + equal,
        p_mid=above + 0.5 * equal,
        realized=realized,
    )


class Cohort:
    """The cohort's cell values, the observation first and one row per replicate."""

    def __init__(self, keys: Sequence[CellKey], values: np.ndarray) -> None:
        self.index = {key: j for j, key in enumerate(keys)}
        self.values = np.asarray(values, dtype=float)

    def series(self, key: CellKey) -> np.ndarray:
        """One cell over the observation and every replicate."""
        return self.values[:, self.index[key]]


def walking_by_week(
    cohort: Cohort, weeks: int, n_subjects: int, interval: float
) -> List[Answer]:
    """Walking episodes a day, week by week, and the rise across the horizon."""
    per_day = float(DAYS_PER_WEEK * n_subjects)
    rows: List[Answer] = []
    for w in range(weeks):
        series = (
            cohort.series(CellKey("week_index", str(w), "walking", COUNT)) / per_day
        )
        realized = REALIZED["walking_week_first"] if w == 0 else None
        realized = REALIZED["walking_week_last"] if w == weeks - 1 else realized
        rows.append(place(f"week {w}", series[0], series[1:], interval, realized))
    first = cohort.series(CellKey("week_index", "0", "walking", COUNT)) / per_day
    last = cohort.series(CellKey("week_index", str(weeks - 1), "walking", COUNT))
    rise = last / per_day - first
    rows.append(
        place(
            "rise, first to last",
            rise[0],
            rise[1:],
            interval,
            REALIZED["walking_week_last"] - REALIZED["walking_week_first"],
        )
    )
    return rows


def walking_by_weekend(
    cohort: Cohort, weeks: int, n_subjects: int, interval: float
) -> List[Answer]:
    """Walking episodes a day on a weekday and on a weekend day, and their ratio."""
    weekdays = float(5 * weeks * n_subjects)
    weekend_days = float(2 * weeks * n_subjects)
    off = cohort.series(CellKey("weekend", "0", "walking", COUNT)) / weekdays
    on = cohort.series(CellKey("weekend", "1", "walking", COUNT)) / weekend_days
    ratio = on / np.maximum(off, 1e-12)
    return [
        place("weekday", off[0], off[1:], interval, REALIZED["walking_weekday"]),
        place("weekend", on[0], on[1:], interval, REALIZED["walking_weekend"]),
        place(
            "weekend / weekday",
            ratio[0],
            ratio[1:],
            interval,
            REALIZED["walking_weekend"] / REALIZED["walking_weekday"],
        ),
    ]


def walking_by_hour(cohort: Cohort, interval: float) -> List[Answer]:
    """The share of walking episodes each hour-of-day bin holds."""
    bins = [key.cell for key in cohort.index if key.family == "hour_of_day"]
    cells = sorted(set(bins), key=lambda name: int(name.split(",")[0][1:]))
    per_bin = {
        cell: cohort.series(CellKey("hour_of_day", cell, "walking", COUNT))
        for cell in cells
    }
    total = sum(per_bin.values())
    rows = [place(cell, *_share(per_bin[cell], total), interval) for cell in cells]
    declared = sum(per_bin[cell] for cell in WALKING_WINDOWS)
    rows.append(place("three declared windows", *_share(declared, total), interval))
    return rows


def _share(part: np.ndarray, total: np.ndarray) -> Tuple[float, np.ndarray]:
    """One cell's share of the total, as the observation and the replicates."""
    series = part / np.maximum(total, 1e-12)
    return float(series[0]), series[1:]


def gym_by_week(cohort: Cohort, weeks: int, interval: float) -> List[Answer]:
    """Minutes a gym episode books, week by week, and the rise across the horizon."""
    rows: List[Answer] = []
    length = []
    for w in range(weeks):
        minutes = cohort.series(CellKey("week_index", str(w), "gym", DURATION))
        episodes = cohort.series(CellKey("week_index", str(w), "gym", COUNT))
        series = np.where(episodes > 0, minutes / np.maximum(episodes, 1.0), np.nan)
        length.append(series)
        realized = REALIZED["gym_minutes_first"] if w == 0 else None
        realized = REALIZED["gym_minutes_last"] if w == weeks - 1 else realized
        rows.append(place(f"week {w}", series[0], series[1:], interval, realized))
    rise = length[-1] - length[0]
    rows.append(
        place(
            "rise, first to last",
            rise[0],
            rise[1:],
            interval,
            REALIZED["gym_minutes_last"] - REALIZED["gym_minutes_first"],
        )
    )
    return rows


# ---------------------------------------------------------------------------
# What the misfits say together
# ---------------------------------------------------------------------------


def misfit_summary(
    results: Sequence[Mapping[str, object]], min_episodes: float
) -> Dict[str, object]:
    """Which cells and which labels the subjects disagree with the model on."""
    by_cell: Dict[Tuple[str, str, str, str], int] = {}
    by_label: Dict[str, int] = {}
    by_family: Dict[str, int] = {}
    named: List[CellCheck] = []
    for result in results:
        report = FidelityReport(
            cells=tuple(result["cells"]),
            members=result["members"],
            interval=float(result["interval"]),
        )
        for cell in report.misfits(min_episodes):
            key = (cell.key.family, cell.key.cell, cell.key.label, cell.key.statistic)
            by_cell[key] = by_cell.get(key, 0) + 1
            by_label[cell.key.label] = by_label.get(cell.key.label, 0) + 1
            by_family[cell.key.family] = by_family.get(cell.key.family, 0) + 1
            named.append(cell)
    named.sort(key=lambda c: (-c.support, str(c.key)))
    return {
        "n_misfit_cells": len(named),
        "by_family": dict(sorted(by_family.items(), key=lambda kv: -kv[1])),
        "by_label": dict(sorted(by_label.items(), key=lambda kv: -kv[1])),
        "repeated": [
            {
                "family": k[0],
                "cell": k[1],
                "label": k[2],
                "statistic": k[3],
                "subjects": v,
            }
            for k, v in sorted(by_cell.items(), key=lambda kv: (-kv[1], kv[0]))
            if v > 1
        ],
        "busiest": [
            {
                "key": str(cell.key),
                "observed": cell.observed,
                "replicate_mean": round(cell.replicate_mean, 3),
                "low": cell.low,
                "high": cell.high,
                "p_value": round(cell.p_value, 4),
                "support": cell.support,
            }
            for cell in named[:40]
        ],
    }


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def estimate_cost(
    n_subjects: int, replicates: int, workers: int, weeks: int
) -> Dict[str, float]:
    """What the planned run costs, in draws and in wall clock."""
    share = float(weeks) / 12.0
    per_subject = SECONDS_PER_FIT * share + replicates * SECONDS_PER_REPLICATE * share
    rounds = math.ceil(n_subjects / max(workers, 1))
    return {
        "draws": n_subjects * replicates,
        "seconds_per_replicate": round(SECONDS_PER_REPLICATE * share, 1),
        "serial_hours": round(n_subjects * per_subject / 3600.0, 2),
        "parallel_hours": round(rounds * per_subject / 3600.0, 2),
        "worker_rounds": rounds,
        "workers": int(workers),
    }


def main(
    csv: str = "data/dynamic_patterns/event_log.tsv",
    subjects: int = 8,
    replicates: int = 80,
    weeks: int = 12,
    workers: int = 8,
    interval: float = 0.90,
    min_episodes: float = 5.0,
    seed: int = 0,
    dry_run: bool = False,
    output_dir: str = "results/exp12",
    budget: Optional[RunBudget] = None,
) -> Dict[str, object]:
    # The grid is one fit per subject and the cost is dominated by the
    # replicates inside each, so the budget records what the run spent and the
    # cell decomposition arrives with the rewrite of this arm.
    ledger = BudgetLedger(budget or RunBudget.full())
    cutoff = weeks * DAYS_PER_WEEK * MINUTES_PER_DAY // BENCHMARK_SLOT_MINUTES
    by_subject = _read_log(csv)
    labels = sorted({e.label for events in by_subject.values() for e in events})
    ladder = subject_ladder(by_subject)
    roster = ladder[: int(subjects)]
    personas = sorted({subject_persona(name) for name in roster})
    cost = estimate_cost(len(roster), replicates, workers, weeks)

    print(f"[Exp12] posterior predictive fidelity on {csv}")
    print(
        f"  {len(by_subject)} subjects in the log, {len(roster)} fitted over "
        f"{len(personas)} personas, {len(labels)} labels, {weeks} weeks"
    )
    print(f"  roster: {', '.join(roster)}")
    print(
        f"  {replicates} replicates per subject, {cost['draws']} draws, "
        f"{cost['seconds_per_replicate']} s a draw"
    )
    print(
        f"  about {cost['serial_hours']} h serial and {cost['parallel_hours']} h "
        f"on {workers} workers over {cost['worker_rounds']} rounds"
    )
    print(
        f"  central interval {interval:.2f}, so a nominal {1.0 - interval:.2f} of "
        f"the p-values fall outside it; cells whose busier side holds under "
        f"{min_episodes:.0f} episodes are dropped from the calibration"
    )
    if dry_run:
        return {"csv": csv, "dry_run": True, "roster": roster, "cost": cost}

    os.makedirs(output_dir, exist_ok=True)
    write_provenance(output_dir)

    tasks = [
        SubjectTask(
            person=name,
            events=tuple(by_subject[name]),
            labels=tuple(labels),
            cutoff=cutoff,
            replicates=replicates,
            interval=interval,
            min_episodes=min_episodes,
            seed=seed + i,
        )
        for i, name in enumerate(roster)
    ]
    started = time.time()
    results: List[Dict[str, object]] = []
    with get_context("fork").Pool(processes=int(workers)) as pool:
        for result in pool.imap_unordered(run_subject, tasks):
            result["interval"] = interval
            results.append(result)
            print(
                f"  {result['person']}: {result['episodes_on_arrays']} episodes, "
                f"calibration {result['calibration']['outside_share']:.3f} over "
                f"{result['calibration']['n_cells']} cells, "
                f"{result['calibration']['n_misfits']} misfits, "
                f"{result['seconds_fit']} s fit, {result['seconds_check']} s check",
                flush=True,
            )
    results.sort(key=lambda r: r["person"])
    elapsed = time.time() - started

    keys = list(results[0]["keys"])
    for result in results[1:]:
        if list(result["keys"]) != keys:
            raise ValueError(
                f"{result['person']} was checked on a different set of cells, so "
                "no cohort statistic adds the same numbers"
            )
    cohort = Cohort(keys, sum(result["members"] for result in results))

    pooled = FidelityReport(
        cells=tuple(cell for result in results for cell in result["cells"]),
        members=np.hstack([result["members"] for result in results]),
        interval=interval,
    )
    summary = pooled.calibration(min_episodes)
    call = pooled.verdict(min_episodes)
    reconciled = sum(int(result["n_reconciled"]) for result in results)

    answers = {
        "walking_by_week": walking_by_week(cohort, weeks, len(roster), interval),
        "walking_by_weekend": walking_by_weekend(cohort, weeks, len(roster), interval),
        "walking_by_hour": walking_by_hour(cohort, interval),
        "gym_by_week": gym_by_week(cohort, weeks, interval),
    }
    misfits = misfit_summary(results, min_episodes)

    _print_run(
        results,
        summary,
        call,
        pooled.families(min_episodes),
        answers,
        misfits,
        reconciled,
        replicates,
        elapsed,
        interval,
    )

    out = {
        "csv": csv,
        "weeks": weeks,
        "subjects": roster,
        "personas": personas,
        "labels": labels,
        "replicates": replicates,
        "interval": interval,
        "min_episodes": min_episodes,
        "seed": seed,
        "cost": cost,
        "seconds": round(elapsed, 1),
        "calendar_nodes": [node.name for node in calendar_nodes(BENCHMARK_ORIGIN)],
        "count_columns_kept": results[0]["count_columns_kept"],
        "count_columns_dropped": results[0]["count_columns_dropped"],
        "calibration": {
            "n_cells": summary.n_cells,
            "n_sparse": summary.n_sparse,
            "nominal": summary.nominal,
            "outside_share": summary.outside_share,
            "n_misfits": summary.n_misfits,
        },
        "verdict": verdict_record(call),
        "worst_families": [
            {
                "family": family.family,
                "label": family.label,
                "n_cells": family.n_cells,
                "discrepancy": round(family.discrepancy, 3),
                "replicate_mean": round(family.replicate_mean, 3),
                "low": round(family.low, 3),
                "high": round(family.high, 3),
                "standardized": round(family.standardized, 3),
                "p_value": round(family.p_value, 6),
            }
            for family in pooled.families(min_episodes)[:20]
        ],
        "reconciliation_checks": reconciled,
        "questions": {
            name: [answer.to_dict() for answer in rows]
            for name, rows in answers.items()
        },
        "misfits": misfits,
        "per_subject": [
            {
                key: value
                for key, value in result.items()
                if key not in ("cells", "keys", "members")
            }
            for result in results
        ],
    }
    path = os.path.join(output_dir, "exp12_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    ledger.finished = len(roster)
    with open(os.path.join(output_dir, "exp12_budget.json"), "w") as f:
        json.dump(ledger.record([]), f, indent=2)
    print(f"\nResults saved to {output_dir}/")
    return out


def _print_run(
    results: Sequence[Mapping[str, object]],
    summary,
    call: Verdict,
    families: Sequence[FamilyCheck],
    answers: Mapping[str, Sequence[Answer]],
    misfits: Mapping[str, object],
    reconciled: int,
    replicates: int,
    elapsed: float,
    interval: float,
) -> None:
    """Print the five things the check answers, one block each."""
    print(f"\n  fitted {len(results)} subjects in {elapsed / 60.0:.1f} minutes")
    print("  calibration over every label-by-cell combination")
    print(f"    {summary}")
    for result in results:
        block = result["calibration"]
        print(
            f"    {result['person']:<34} {block['outside_share']:.3f} over "
            f"{block['n_cells']:>5} cells, {block['n_misfits']:>4} misfits, "
            f"{result['episodes_on_arrays']} episodes"
        )

    print("\n  the verdict, over the families of the whole cohort")
    print(f"    {call}")
    for family in families[:10]:
        print(f"    {family}")
    for result in results:
        block = result["verdict"]
        print(
            f"    {result['person']:<34} p {block['p_value']:.4f}, worst "
            f"{block['worst_label']} {block['worst_family']} at z "
            f"{block['worst_standardized']:.2f}"
        )

    titles = {
        "walking_by_week": "does walking increase across the twelve weeks",
        "walking_by_weekend": "is walking higher at weekends",
        "walking_by_hour": "is walking concentrated in the declared windows",
        "gym_by_week": "does gym duration increase across the weeks",
    }
    for name, rows in answers.items():
        print(f"\n  {titles[name]}")
        for answer in rows:
            print(f"  {answer}")

    print(f"\n  misfits, over {misfits['n_misfit_cells']} cells")
    print(
        "    by family: "
        + ", ".join(f"{k} {v}" for k, v in misfits["by_family"].items())
    )
    print(
        "    by label: "
        + ", ".join(f"{k} {v}" for k, v in list(misfits["by_label"].items())[:12])
    )
    for row in misfits["busiest"][:15]:
        print(
            f"    {row['key']}: observed {row['observed']:.0f}, replicates "
            f"{row['replicate_mean']:.1f} [{row['low']:.0f}, {row['high']:.0f}], "
            f"p {row['p_value']:.3f}"
        )

    print(
        f"\n  reconciliation exact on {reconciled} logs, which is every replicate "
        f"and every record, at {interval:.2f} interval over {replicates} draws"
    )


def _read_log(path: str) -> Dict[str, List[ObservedEvent]]:
    """Every subject's episodes, from a log whose header states its separator."""
    with open(path, encoding="utf-8") as f:
        header = f.readline()
    return read_subject_event_log(path, delimiter="\t" if "\t" in header else ",")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/dynamic_patterns/event_log.tsv")
    parser.add_argument("--subjects", type=int, default=8)
    parser.add_argument("--replicates", type=int, default=80)
    parser.add_argument("--weeks", type=int, default=12)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--interval", type=float, default=0.90)
    parser.add_argument("--min-episodes", type=float, default=5.0)
    add_budget_arguments(parser)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default="results/exp12")
    args = parser.parse_args()
    main(
        csv=args.csv,
        subjects=2 if args.quick else args.subjects,
        replicates=5 if args.quick else args.replicates,
        weeks=2 if args.quick else args.weeks,
        workers=args.workers,
        interval=args.interval,
        min_episodes=args.min_episodes,
        seed=args.seed,
        dry_run=args.dry_run,
        output_dir=args.output_dir,
        budget=budget_from_args(args),
    )
