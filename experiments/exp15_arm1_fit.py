"""
experiments/exp15_arm1_fit.py
=============================
Experiment 15: the observational arm's fit, and the battery it has to survive.

The graph is searched once over the balanced panel, at the price the panel's own
null puts up, with a relation priced once and the self stratum carrying its own
in-degree bound. What comes out is a set of lagged associations, and every rung
below is an attempt to kill one of them.

R1 counts the accepted edges that state a relation the study protocol forbids.
R2 injects a label matched on marginal, autocorrelation and participant
heterogeneity and connected to nothing, and counts what attaches to it. R3
counts the resolution-stratified positives the frozen anchors declare. R4 adds a
future value of each accepted parent to its own fit. R9 runs the search over
complementary halves of the roster and reports each edge's inclusion frequency
with the bound on low-probability selections. R10 reports how strong an
unmeasured confounder would have to be to explain each edge away.

The three-way calendar ablation is scored here too: absent, at the home scale,
and at every scale, each refitted from nothing and compared on the objective it
reaches.

No precision, no F1, no SHD. The observed log has no graph truth, so what is
reported is a falsification record and the rate at which the battery admits
relations the protocol forbids.

Run
---
    python -m experiments.exp15_arm1_fit --quick

    python -m experiments.exp15_arm1_fit --output-dir results/exp15
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.data.anchors import NEGATIVE, POSITIVE, anchor_set
from src.data.lifesnaps_panel import balanced_panel, panel_provenance, wave_of
from src.evaluation.arm1_report import edge_name
from src.evaluation.falsification import (
    anchor_recall,
    e_value,
    expected_false_selections,
    future_lag_control,
    invariance,
    protocol_negatives,
    risk_ratio,
    sham_rung,
)
from src.learning.arm1 import (
    MAX_LAG,
    SEARCH_SCALES,
    SHAM_LABEL,
    edge_series,
    environments,
    fit_arm1,
    future_of,
    sham_cohort,
    touches_sham,
)
from src.learning.calendar_arms import ABSENT, ARMS, EVERY, HOME
from src.utils.budget import (
    BudgetLedger,
    RunBudget,
    add_budget_arguments,
    budget_from_args,
)
from src.utils.provenance import write_provenance

#: The scale the invariance rung reads, where a subject has enough epochs to
#: split into environments and few enough to fit every subset.
INVARIANCE_SCALE = 3

STABILITY_THRESHOLD = 0.75


def evidence_for(cohort, fit, axis) -> List[Dict[str, object]]:
    """R4 and R10 for every accepted edge, on the columns the edge itself reads."""
    rows: List[Dict[str, object]] = []
    for edge, target in fit.accepted:
        response, parent, who = edge_series(cohort, edge, target)
        control = future_lag_control(response, parent, future_of(parent, who))
        spread = float(np.std(parent)) or 1.0
        coefficient = float(fit.params.coef[(target, "count")].get(edge, 0.0))
        rows.append(
            {
                "edge": edge_name(edge, target, axis),
                "scale": edge.src_scale,
                "lag": edge.lag,
                "coefficient": coefficient,
                "standardized": coefficient * spread,
                "future_lag_p": control.p_value,
                "future_lag_coefficient": control.coefficient,
                "e_value": e_value(risk_ratio(coefficient, 1.0 / spread)),
            }
        )
    return rows


def stability(
    cohort,
    axis,
    arm: str,
    n_pairs: int,
    scales,
    max_lag: int,
    price: float,
    n_candidates: int,
    seed: int,
) -> Dict[str, object]:
    """R9: how often each edge survives a refit on half the roster."""
    from dataclasses import replace

    rng = np.random.default_rng(seed)
    roster = np.array(cohort.subjects)
    counts: Dict[str, int] = {}
    sizes: List[int] = []
    draws = 0
    for _ in range(n_pairs):
        order = rng.permutation(len(roster))
        half = len(order) // 2
        for side in (order[:half], order[half : 2 * half]):
            keep = tuple(sorted(roster[np.sort(side)]))
            rows = np.array([cohort.subjects.index(name) for name in keep])
            half_cohort = replace(cohort, subjects=keep, masks=cohort.masks[rows])
            fitted = fit_arm1(
                half_cohort,
                axis,
                arm,
                scales=scales,
                max_lag=max_lag,
                price=price,
                seed=seed,
            )
            sizes.append(len(fitted.accepted))
            draws += 1
            for edge, target in fitted.accepted:
                name = edge_name(edge, target, axis)
                counts[name] = counts.get(name, 0) + 1
    frequency = {name: hits / draws for name, hits in counts.items()}
    return {
        "n_pairs": n_pairs,
        "threshold": STABILITY_THRESHOLD,
        "frequency": frequency,
        "selected": sorted(
            name for name, share in frequency.items() if share >= STABILITY_THRESHOLD
        ),
        "expected_false": expected_false_selections(
            float(np.mean(sizes)) if sizes else 0.0,
            n_candidates,
            STABILITY_THRESHOLD,
        ),
    }


def invariance_for(cohort, fit, axis, waves) -> Dict[str, object]:
    """R7: which parent sets look the same in the wave and half-window blocks."""
    if not fit.accepted:
        return {"model_rejected": True, "intersection": [], "note": "no accepted edge"}
    target, at_scale = fit.accepted[0][1], fit.accepted[0][0].src_scale
    columns = []
    names = []
    # One target and one scale, so every column has one row per epoch of that
    # scale and the residual they leave is one residual.
    for edge, dst in fit.accepted:
        if dst != target or edge.src_scale != at_scale or len(columns) >= 3:
            continue
        _, parent, _ = edge_series(cohort, edge, dst)
        columns.append(parent)
        names.append(edge_name(edge, dst, axis))
    if not columns:
        return {"model_rejected": True, "intersection": [], "note": "no shared target"}
    response, _, _ = edge_series(cohort, fit.accepted[0][0], target)
    report = invariance(
        response,
        np.column_stack(columns),
        environments(cohort, waves, at_scale),
        max_size=min(2, len(columns)),
    )
    return {
        "target": axis.targets[target],
        "columns": names,
        "model_rejected": report.model_rejected,
        "n_invariant_sets": len(report.accepted_sets),
        "intersection": [names[j] for j in report.intersection],
    }


def main(
    participants: Optional[int] = None,
    scales: Sequence[int] = SEARCH_SCALES,
    max_lag: int = MAX_LAG,
    n_pairs: int = 8,
    arms: Sequence[str] = ARMS,
    output_dir: str = "results/exp15",
    budget: Optional[RunBudget] = None,
) -> Dict[str, object]:
    os.makedirs(output_dir, exist_ok=True)
    ledger = BudgetLedger(budget or RunBudget.full())

    cohort, axis, panel, cohort_report, collapse = balanced_panel(
        participants=participants
    )
    print(panel.summary())
    print(f"\n{cohort.S} subjects, {axis.L} targets, {len(axis.exogenous)} exogenous")

    fit = fit_arm1(cohort, axis, HOME, scales=scales, max_lag=max_lag)
    print(
        f"\nprice {fit.price:.2f} (null median {fit.null_median:.2f}, "
        f"max {fit.null_max:.1f}); accepted {len(fit.accepted)} edges "
        f"{fit.by_stratum()}"
    )
    for edge, target in fit.accepted:
        print(f"  {edge_name(edge, target, axis)}")
    if fit.rejected():
        print(f"  the guard turned away {fit.rejected()}")

    anchors = anchor_set(axis)
    keys = fit.keys(axis)
    r1 = protocol_negatives(keys, anchors)
    r3 = anchor_recall(keys, anchors)
    reachable = [
        a.key
        for a in anchors.of_sign(POSITIVE, True)
        if a.key.scale in {"hour", "day", "week"}
    ]
    print(f"\n{r1}\n{r3}")
    print(
        f"  positives reachable at the scales this run searched: "
        f"{sum(1 for k in reachable if k in set(keys))}/{len(reachable)}"
    )

    sham_people, sham_axis = sham_cohort(cohort, axis, reference="walk")
    sham_fit = fit_arm1(
        sham_people, sham_axis, HOME, scales=scales, max_lag=max_lag, price=fit.price
    )
    hits, offered = touches_sham(sham_fit, sham_axis)
    r2 = sham_rung(hits, offered)
    print(f"\n{r2}")

    evidence = evidence_for(cohort, fit, axis)
    print(f"\nper-edge evidence ({len(evidence)} edges):")
    for row in evidence:
        print(
            f"  {row['edge']:<62} beta {row['coefficient']:+.3f}  "
            f"std {row['standardized']:+.2f}  E {row['e_value']:.3g}  "
            f"R4 p {row['future_lag_p']:.3f}"
        )

    r9 = stability(
        cohort,
        axis,
        HOME,
        n_pairs,
        scales,
        max_lag,
        fit.price,
        n_candidates=len(fit.pool) * axis.L,
        seed=0,
    )
    print(
        f"\nR9 stability over {n_pairs} complementary pairs: "
        f"{len(r9['selected'])} edges at or above {STABILITY_THRESHOLD}, "
        f"expected low-probability selections at most {r9['expected_false']:.2f}"
    )
    r7 = invariance_for(cohort, fit, axis, wave_of(panel))
    print(f"R7 invariance: {r7}")

    ablation = {}
    for arm in arms:
        arm_fit = (
            fit
            if arm == HOME
            else fit_arm1(
                cohort, axis, arm, scales=scales, max_lag=max_lag, price=fit.price
            )
        )
        ablation[arm] = {
            "n_candidates": len(arm_fit.pool),
            "n_accepted": len(arm_fit.accepted),
            "by_stratum": arm_fit.by_stratum(),
            "objective": arm_fit.objective,
        }
        print(
            f"calendar arm {arm:<7} objective {arm_fit.objective:12.2f}  "
            f"{len(arm_fit.accepted)} edges of {len(arm_fit.pool)} candidates"
        )

    out = {
        "provenance": panel_provenance(cohort, axis, panel),
        "collapse_agreement": collapse.agreement.share_in_agreement,
        "participants_kept": cohort_report.participants_kept,
        "search": {
            "scales": list(scales),
            "max_lag": max_lag,
            "price": fit.price,
            "null_median": fit.null_median,
            "null_max": fit.null_max,
            "n_candidates": len(fit.pool),
            "by_stratum": fit.by_stratum(),
            "accepted": [edge_name(e, t, axis) for e, t in fit.accepted],
            "guard_rejected": fit.rejected(),
        },
        "anchors": {"digest": anchors.digest(), "n_anchors": len(anchors)},
        "rungs": {
            "R1": {"hits": r1.hits, "denominator": r1.denominator, "passed": r1.passed},
            "R2": {"hits": r2.hits, "denominator": r2.denominator, "passed": r2.passed},
            "R3": {
                "hits": r3.hits,
                "denominator": r3.denominator,
                "reachable": len(reachable),
                "reachable_hits": sum(1 for k in reachable if k in set(keys)),
            },
            "R7": r7,
            "R9": r9,
        },
        "edges": evidence,
        "calendar_ablation": ablation,
        "negative_anchors": len(anchors.of_sign(NEGATIVE, True)),
    }
    with open(os.path.join(output_dir, "exp15_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(output_dir, "exp15_budget.json"), "w") as f:
        json.dump(ledger.record([]), f, indent=2)
    write_provenance(output_dir)
    print(f"\nResults saved to {output_dir}/")
    return out


def _parse(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--participants", type=int, default=None)
    parser.add_argument("--max-lag", type=int, default=MAX_LAG)
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--output-dir", default="results/exp15")
    add_budget_arguments(parser)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse()
    main(
        participants=10 if args.quick else args.participants,
        max_lag=1 if args.quick else args.max_lag,
        n_pairs=2 if args.quick else args.pairs,
        output_dir=args.output_dir,
        budget=budget_from_args(args),
    )
