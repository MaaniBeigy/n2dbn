"""
experiments/exp16_round_trip.py
===============================
Experiment 16: the fit-then-simulate round trip, and the controls that grade it.

The observational arm fits a graph over the balanced panel. This draws logs back
out of that fit, writes them in the raw schema and reads them through the same
panel reader, so what the nonstationarity grid then compares is two logs and
never two readers.

Four legs are drawn. The home arm carries the searched graph with the clock
nodes at their home scale and the hour-scale regime chain the arm fits. The
no-regime arm is the same search with the chain given one state, which is no
chain at all, and it is the $M_k = 1$ leg of the no-regime control. The absent
arm is the same search with the clock nodes withheld, which is the
calendar-nodes-removed control. The empty graph carries an intercept and a
participant level and nothing else, and it is the control that says the panel
can fail at all.

Two shuffles of the home leg complete the four declared controls: a
within-participant time shuffle, which has to break every Layer A row, and a
label-shuffled cross-section, which has to break the joint rows and leave the
marginal ones alone.

The three changepoint rows report the six figures a changepoint form is stated
in: annotation error, F1 inside a margin, covering, Hausdorff distance and the
two Kolmogorov-Smirnov statistics between jump magnitudes and segment lengths.

Run
---
    python -m experiments.exp16_round_trip --quick

    python -m experiments.exp16_round_trip --output-dir results/exp16
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.core.typed_graph import COUNT, TypedGraph
from src.data.lifesnaps_panel import balanced_panel, panel_provenance
from src.evaluation.arm1_report import edge_name
from src.evaluation.heterogeneity import (
    compare_heterogeneity,
    separation,
    sparse_gap,
    weekly_blocks,
)
from src.evaluation.nonstationarity import CONTROLS, control_verdict, panel_verdict
from src.evaluation.panel_compare import (
    COMPARED_SCALES,
    LABEL_SHUFFLE,
    SCALE_NAMES,
    TIME_SHUFFLE,
    WEEK,
    WEEK_FORMS,
    WEEK_SCALE,
    by_key,
    compare,
    declared_rows,
    failed_rows,
    form_ids,
    panel_series,
    rows_tested,
    shuffled,
    summary,
)
from src.evaluation.round_trip import (
    calibrate_marks,
    center_chains,
    check_volume,
    missing_days,
    panel_counts,
    participant_levels,
    population_levels,
    reconciliation_error,
    replicates,
    start_log_odds,
)
from src.evaluation.segmentation import POISSON, SLOPE, VARIANCE, match_changepoints
from src.learning.arm1 import (
    MAX_LAG,
    REGIME_STATES,
    SEARCH_SCALES,
    exogenous_index,
    fit_arm1,
)
from src.learning.calendar_arms import ABSENT, HOME
from src.model.intensity import CLIP_COLUMNS, FineParams
from src.utils.budget import (
    EXHAUSTED,
    BudgetLedger,
    RunBudget,
    add_budget_arguments,
    budget_from_args,
)
from src.utils.provenance import write_provenance

#: The four legs, and which declared control each one stands for.  The home leg
#: is the whole fit; the other three each take one thing out of it.
EMPTY = "empty"
NOREGIME = "noregime"
LEGS: Tuple[str, ...] = (HOME, NOREGIME, ABSENT, EMPTY)
LEG_CONTROL: Dict[str, str] = {
    NOREGIME: "no regime",
    ABSENT: "calendar nodes removed",
}

#: Which calendar arm each leg fits under, and how many regime states it carries.
#: One state is no chain at all.
LEG_ARM: Dict[str, str] = {HOME: HOME, NOREGIME: HOME, ABSENT: ABSENT}
LEG_STATES: Dict[str, int] = {HOME: REGIME_STATES, NOREGIME: 1, ABSENT: REGIME_STATES}

#: Sweeps the participant level takes over the searched graph.  The level is a
#: coordinate ascent and its gain per sweep decays slowly: on ten subjects forty
#: sweeps gain 260 nats and the last four of them gain 14.  A sweep costs 3.3
#: seconds on the full roster, so the run takes sixty and reports where it
#: stopped.
LEVEL_SWEEPS = 60

#: The cost each changepoint row segments under.
CHANGEPOINT_COSTS: Dict[str, str] = {"A08": POISSON, "A09": VARIANCE, "A10": SLOPE}


def leg_graph(cohort, axis, arm: str, scales, max_lag: int):
    """The graph and the parameters one leg draws from, before the level is fitted."""
    if arm == EMPTY:
        graph = TypedGraph(
            label_names=list(axis.targets), calendar_names=list(axis.exogenous)
        )
        params = FineParams(n_labels=axis.L)
        params.base[COUNT][:] = start_log_odds(
            cohort, {k: max_lag for k in range(1, cohort.K + 1)}
        )
        return graph, params, None
    fit = fit_arm1(
        cohort,
        axis,
        LEG_ARM[arm],
        scales=scales,
        max_lag=max_lag,
        regime_states=LEG_STATES[arm],
    )
    return fit.graph, fit.params, fit


def draw_leg(
    cohort,
    axis,
    report,
    arm: str,
    scales,
    max_lag: int,
    n_replicates: int,
    output_dir: str,
    seed: int,
) -> Tuple[List, Dict[str, object]]:
    """Fit one leg, draw its replicates and report what the writer and reader agreed."""
    index = exogenous_index(axis, cohort.K)
    budget = {k: int(max_lag) for k in range(1, cohort.K + 1)}
    graph, params, fit = leg_graph(cohort, axis, arm, scales, max_lag)
    chains = [] if fit is None else list(fit.chains)
    params = calibrate_marks(params, cohort)
    # The level refit below carries no chain, so the chain's average is folded
    # into the intercept first and taken back out of the chain.  The predictor
    # is unchanged slot by slot; what changes is that the level fit is now
    # centred where the draw will be.
    params, chains = center_chains(params, chains)
    params, history, unidentified = participant_levels(
        cohort, graph, params, budget, index=index, max_sweeps=LEVEL_SWEEPS
    )
    # The contrast is between two groups of participants, so each replicate
    # draws its own population of levels off the fitted roster and no drawn
    # participant is the twin of a real one.
    trips = []
    for r in range(n_replicates):
        drawn_params = population_levels(cohort, params, unidentified, seed=seed + r)
        trips.extend(
            replicates(
                cohort,
                axis,
                report.schedule,
                graph,
                drawn_params,
                1,
                os.path.join(output_dir, "logs"),
                seed=seed + r,
                tag=f"{arm}_{r:02d}",
                index=index,
                lag_budget=budget,
                chains=chains,
            )
        )
    # The tripwire, before any row is computed over these logs.  A label the
    # real panel carries and the draw does not makes every row over it a
    # statement about a log the model did not make, so it is refused here rather
    # than reported four tables later.
    real_counts = panel_counts(cohort, axis)
    drawn_counts: Dict[str, int] = {label: 0 for label in axis.targets}
    clip_counts: Dict[str, int] = {label: 0 for label in axis.targets}
    clip_ends: Dict[str, Dict[str, int]] = {
        label: {end: 0 for end in CLIP_COLUMNS} for label in axis.targets
    }
    for trip in trips:
        for label, n in trip.per_label.items():
            drawn_counts[label] = drawn_counts.get(label, 0) + int(n)
        for label, ends in trip.clips.items():
            for end, n in ends.items():
                clip_ends[label][end] += int(n)
                clip_counts[label] = clip_counts.get(label, 0) + int(n)
    per_replicate = {label: n / max(len(trips), 1) for label, n in drawn_counts.items()}
    departures = check_volume(per_replicate, real_counts, where=f"leg {arm!r}")

    record = {
        "arm": arm,
        "real_per_label": real_counts,
        "drawn_per_label": {k: int(v) for k, v in drawn_counts.items()},
        "drawn_per_label_per_replicate": {
            k: float(v) for k, v in per_replicate.items()
        },
        "volume_departures": departures,
        "eta_clips_per_label": clip_counts,
        "eta_clips_by_end": clip_ends,
        "edges": (
            []
            if fit is None
            else [
                {
                    "target": axis.targets[t],
                    "name": edge_name(e, t, axis),
                    "edge": str(e),
                    "same_epoch": e.is_same_epoch,
                }
                for e, t in fit.accepted
            ]
        ),
        "refused_columns": (
            {}
            if fit is None
            else {axis.targets[t]: names for t, names in fit.refused.items()}
        ),
        "unsupported_self_candidates": (
            {}
            if fit is None
            else {axis.targets[t]: names for t, names in fit.unsupported.items()}
        ),
        "n_edges": 0 if fit is None else len(fit.accepted),
        "same_epoch_edges": (
            0 if fit is None else sum(1 for e, _ in fit.accepted if e.is_same_epoch)
        ),
        "finest_scale_edges": (
            0
            if fit is None
            else sum(
                1 for e, _ in fit.accepted if not e.is_exogenous and e.src_scale == 1
            )
        ),
        "regime_states": 0 if not chains else int(chains[0].n_states),
        "level_sweeps": len(history),
        "unidentified_levels": list(unidentified),
        "level_loglik": float(history[-1]),
        "level_gain": float(history[-1] - history[0]),
        "level_last_gain": (
            float(history[-1] - history[-2]) if len(history) > 1 else 0.0
        ),
        "n_replicates": len(trips),
        "drawn": [t.drawn for t in trips],
        "copied": [t.copied for t in trips],
        "reconciliation_error": max(reconciliation_error(t) for t in trips),
        "observed_days_left_empty": int(
            sum(sum(missing_days(t, report.schedule).values()) for t in trips)
        ),
    }
    return trips, record


def changepoint_table(
    real, drawn, labels: Sequence[str], scales: Sequence[int]
) -> List[Dict[str, object]]:
    """The six figures of every changepoint row, per label and scale."""
    real_by = by_key(real)
    drawn_by = by_key(drawn)
    rows: List[Dict[str, object]] = []
    for form, cost in CHANGEPOINT_COSTS.items():
        for scale in scales:
            if SCALE_NAMES[scale] == WEEK and form not in WEEK_FORMS:
                continue
            for label in labels:
                key = (label, SCALE_NAMES[scale])
                if key not in real_by or key not in drawn_by:
                    continue
                matches = [
                    match_changepoints(a, b, cost)
                    for a, b in zip(real_by[key].blocks, drawn_by[key].blocks)
                ]
                averaged = {
                    name: float(np.mean([m.as_row()[name] for m in matches]))
                    for name in matches[0].as_row()
                }
                rows.append(
                    {
                        "form": form,
                        "cost": cost,
                        "series": label,
                        "scale": SCALE_NAMES[scale],
                        **averaged,
                    }
                )
    return rows


def heterogeneity_table(
    cohort, drawn_cohorts: Sequence, labels: Sequence[str], scale: int
) -> List[Dict[str, object]]:
    """Each label's real and drawn CQV over weekly totals, and whether they meet."""
    rows: List[Dict[str, object]] = []
    for j, label in enumerate(labels):
        real_blocks = weekly_blocks(cohort, scale, j)
        drawn_blocks = [
            block for other in drawn_cohorts for block in weekly_blocks(other, scale, j)
        ]
        try:
            left, right, agree = compare_heterogeneity(
                real_blocks, drawn_blocks, label, seed=0
            )
        except ValueError as refused:
            rows.append({"series": label, "refused": str(refused)})
            continue
        rows.append(
            {
                "series": label,
                "real_cqv": left.cqv,
                "real_lower": left.lower,
                "real_upper": left.upper,
                "real_undefined_sparse": left.undefined_sparse,
                "real_pinned_share": left.pinned_share,
                "drawn_cqv": right.cqv,
                "drawn_lower": right.lower,
                "drawn_upper": right.upper,
                "drawn_undefined_sparse": right.undefined_sparse,
                "drawn_pinned_share": right.pinned_share,
                "comparable": agree,
                "separation": separation(left, right),
                "sparse_gap": sparse_gap(left, right),
                "replicates": left.replicates,
                "seed": left.seed,
            }
        )
    return rows


def main(
    participants: Optional[int] = None,
    scales: Sequence[int] = SEARCH_SCALES,
    max_lag: int = MAX_LAG,
    n_replicates: int = 5,
    compared_scales: Sequence[int] = COMPARED_SCALES,
    legs: Sequence[str] = LEGS,
    output_dir: str = "results/exp16",
    budget: Optional[RunBudget] = None,
) -> Dict[str, object]:
    os.makedirs(output_dir, exist_ok=True)
    ledger = BudgetLedger(budget or RunBudget.full())

    cohort, axis, report, cohort_report, _ = balanced_panel(participants=participants)
    print(report.summary())
    real = panel_series(cohort, axis, compared_scales)
    print(
        f"\n{cohort.S} subjects, {axis.L} targets, {len(real)} series over scales "
        f"{[SCALE_NAMES[k] for k in compared_scales]}"
    )

    drawn: Dict[str, List] = {}
    legs_record: Dict[str, object] = {}
    panels: Dict[str, object] = {}
    home_trips: List = []
    # A leg is this run's cell.  The 2026-08-23 run built a ledger, never asked
    # it anything, and reported `"complete"` after 15,952 s against a declared
    # 7,200 s; a budget nothing consults is a comment.
    not_reached: List[str] = []
    for index, arm in enumerate(legs):
        limit = ledger.exceeded()
        if limit is not None:
            ledger.reason = EXHAUSTED
            ledger.stopped_by = limit
            not_reached = list(legs[index:])
            print(f"\nbudget exhausted on {limit}; legs not reached: {not_reached}")
            break
        trips, record = draw_leg(
            cohort,
            axis,
            report,
            arm,
            scales,
            max_lag,
            n_replicates,
            output_dir,
            seed=100 * index,
        )
        if arm == HOME:
            home_trips = list(trips)
        drawn[arm] = [panel_series(t.cohort, t.axis, compared_scales) for t in trips]
        cells = compare(real, drawn[arm])
        panels[arm] = summary(cells)
        legs_record[arm] = record
        verdict, failing = panel_verdict(cells)
        print(
            f"\nleg {arm:<7} {record['n_edges']} edges, "
            f"{record['n_replicates']} replicates, reconciliation "
            f"{record['reconciliation_error']:.0f}, "
            f"{len(failing)} of {panels[arm]['n_tested']} tested cells failing, "
            f"panel {'passes' if verdict else 'fails'}"
        )
        for row in failing[:8]:
            print(
                f"    {row.form} {row.series} at the {row.scale} scale, "
                f"p {row.p_value:.3f}, magnitude {row.magnitude:.3g}"
            )
        ledger.finished += 1

    # The positive control of 39.7: one drawn log against the others drawn from
    # the same fit.  Nothing separates them, so a row failing here is reporting
    # its own sampling noise and its verdict on the real log carries nothing.
    positive: Dict[str, object] = {"note": "one replicate needs no control"}
    noisy: List[str] = []
    if len(drawn[HOME]) > 1:
        control_cells = compare(drawn[HOME][0], drawn[HOME][1:])
        noisy = list(failed_rows(control_cells))
        positive = summary(control_cells)
    print(
        f"\npositive control, one home replicate against the other "
        f"{max(len(drawn[HOME]) - 1, 0)}: rows failing their own noise {noisy}"
    )

    home_cells_failed = panels[HOME]["failed"]
    tested = panels[HOME]["tested_rows"]
    controls: Dict[str, object] = {}
    for control in CONTROLS:
        # Every control now takes something out of the home leg or perturbs its
        # draw, so what the home leg already fails is never the control's doing.
        baseline: Sequence[str] = home_cells_failed
        if control.name in LEG_CONTROL.values():
            arm = next(a for a, name in LEG_CONTROL.items() if name == control.name)
            broke = panels[arm]["failed"]
            put = panels[arm]["tested_rows"]
        else:
            kind = (
                TIME_SHUFFLE
                if control.name == "within-participant time shuffle"
                else LABEL_SHUFFLE
            )
            moved = [
                shuffled(panel, kind, seed=1 + i) for i, panel in enumerate(drawn[HOME])
            ]
            cells = compare(real, moved)
            broke = list(failed_rows(cells))
            put = list(rows_tested(cells))
        passed, detail = control_verdict(control, broke, put, baseline)
        controls[control.name] = {"passed": passed, "broke": list(broke), **detail}
        print(
            f"control {control.name:<32} "
            f"{'did its job' if passed else 'did not'}: {detail}"
        )

    empty_fails = not panels[EMPTY]["passed"]
    separates = sorted(set(panels[EMPTY]["failed"]) - set(panels[HOME]["failed"]))
    print(
        f"rows the empty graph fails that the fitted one passes: "
        f"{separates if separates else 'none'}"
    )
    print(
        f"\nempty-graph control: the panel "
        f"{'fails it, as it must' if empty_fails else 'passes it, which it must not'}"
    )

    print("\nper form, over the home leg:")
    for form, row in sorted(panels[HOME]["by_form"].items()):
        print(
            f"  {form:<4} {row['tested']:>3} tested of {row['cells']:>3} cells, "
            f"{row['failed']:>3} failing; descriptive {row['clauses']}"
        )

    # The heterogeneity comparison of 39.4.5, on weekly totals, with the
    # percentile interval taken over participants.
    hetero = heterogeneity_table(
        cohort, [t.cohort for t in home_trips], list(axis.targets), WEEK_SCALE
    )
    print(f"\nheterogeneity, CQV over weekly totals ({len(hetero)} labels):")
    for row in hetero:
        if "refused" in row:
            print(f"  {row['series']:<16} undefined_sparse on the whole roster")
            continue
        print(
            f"  {row['series']:<16} real {row['real_cqv']:.3f} "
            f"[{row['real_lower']:.3f}, {row['real_upper']:.3f}]  "
            f"drawn {row['drawn_cqv']:.3f} "
            f"[{row['drawn_lower']:.3f}, {row['drawn_upper']:.3f}]  "
            f"{'comparable' if row['comparable'] else 'apart by %.3f' % row['separation']}"
        )

    table = changepoint_table(real, drawn[HOME][0], list(axis.targets), compared_scales)
    print(f"\nchangepoint rows on both logs ({len(table)} cells):")
    for row in table[:10]:
        print(
            f"  {row['form']} {row['series']:<16} {row['scale']:<5} "
            f"AE {row['annotation_error']:.2f}  F1 {row['f1_margin']:.2f}  "
            f"cover {row['covering']:.2f}  Hd {row['hausdorff']:.1f}  "
            f"jumpKS {row['jump_ks']:.2f}  segKS {row['segment_ks']:.2f}"
        )

    out = {
        "provenance": panel_provenance(cohort, axis, report),
        "participants_kept": cohort_report.participants_kept,
        "n_replicates": n_replicates,
        "compared_scales": [SCALE_NAMES[k] for k in compared_scales],
        "forms_run": list(form_ids()),
        "forms_declared": list(declared_rows()),
        "legs": legs_record,
        "panels": panels,
        "controls": controls,
        "empty_graph_fails_the_panel": empty_fails,
        "positive_control": positive,
        "rows_failing_their_own_noise": noisy,
        "home_failed_net_of_noise": sorted(set(home_cells_failed) - set(noisy)),
        "changepoints": table,
        "heterogeneity": hetero,
        "rows_separating_the_fit_from_the_empty_graph": separates,
        "home_failed_rows": list(home_cells_failed),
        "home_tested_rows": list(tested),
    }
    with open(os.path.join(output_dir, "exp16_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(output_dir, "exp16_budget.json"), "w") as f:
        json.dump(ledger.record(not_reached), f, indent=2)
    write_provenance(output_dir)
    print(f"\nResults saved to {output_dir}/")
    return out


def _parse(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--participants", type=int, default=None)
    parser.add_argument("--max-lag", type=int, default=MAX_LAG)
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--output-dir", default="results/exp16")
    add_budget_arguments(parser)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse()
    main(
        participants=8 if args.quick else args.participants,
        max_lag=1 if args.quick else args.max_lag,
        n_replicates=2 if args.quick else args.replicates,
        compared_scales=(3,) if args.quick else COMPARED_SCALES,
        output_dir=args.output_dir,
        budget=budget_from_args(args),
    )
