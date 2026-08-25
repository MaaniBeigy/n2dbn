"""
experiments/exp14_arm1_lifesnaps.py
===================================
Experiment 14: the observational arm's data path, and what it settles before a
single edge is scored.

Six things are produced here and none of them needs a fit.

The three columns that record one act are collapsed, and what the collapse rests
on is measured on the log being read.  The epoch ladder is built on participant
time, so a staggered cohort does not read as three-quarters absent.  The absent
days are audited against the mechanism the ladder declares.  The anchor set is
written as data and hashed, so a later scoring run names the set it scored
against.  The coarse association tensor is computed and its agreement with the
fine lagged association is reported, together with the depth curve that says how
much of the space a given operating point reaches.  And the shift placebo is run
on the anchor relation itself, under the unrepaired null and under the repaired
one, which is what makes the repair a measurement.

The three-way calendar-node ablation is declared here and its placements are
recorded.  Scoring the arms needs the search, which is where they run.

Run
---
    python -m experiments.exp14_arm1_lifesnaps --quick

    python -m experiments.exp14_arm1_lifesnaps --output-dir results/exp14
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

from src.data.anchors import ANCHOR_READ, ANCHOR_SOURCE, NEGATIVE, POSITIVE, anchor_set
from src.data.label_axis import SCHEDULED_LABEL
from src.data.lifesnaps_ladder import (
    LADDER_UNITS,
    build_ladder,
    missingness_audit,
)
from src.evaluation.falsification import shift_null
from src.learning.calendar_arms import ARMS, calendar_candidates, placement
from src.learning.coarse_order import (
    association,
    association_tensor,
    drop_self,
    lagged,
    recall_against_depth,
    visit_order,
)
from src.utils.budget import (
    BudgetLedger,
    RunBudget,
    add_budget_arguments,
    budget_from_args,
)
from src.utils.provenance import write_provenance

#: The five calendar columns of the exogenous block.  The ordering partials them
#: out of every source and every target.
N_CALENDAR = 5

#: The scale the fine search runs at, and the coarse scales that order it.
FINE_UNIT = "hour"
COARSE_UNITS: Tuple[str, ...] = ("day", "week")

#: Where the depth curve is read, as a share of the candidate space.
DEPTH_SHARES: Tuple[float, ...] = (0.15, 0.3, 0.5, 0.75, 1.0)


def _blocks(record, scale: int) -> List[Tuple[int, int]]:
    """The half-open epoch range of each participant at one scale."""
    return [
        (b.offsets[scale - 1], b.offsets[scale - 1] + b.n_epochs(scale))
        for b in record.blocks
    ]


def _pairs(priority: np.ndarray) -> List[Tuple[int, int]]:
    """Every ordered source-target pair, in the order the flattened matrix holds."""
    n_sources, n_targets = priority.shape
    return [(i, j) for i in range(n_sources) for j in range(n_targets)]


def coarse_ordering(record, merge: str = "max") -> Dict[str, object]:
    """
    The coarse tensor, its agreement with the fine lagged association, and the curve.

    The coarse statistic is symmetric, so both directions of a pair share one
    priority; the fine gain is what separates them.  Self pairs are dropped,
    since a label correlates with itself by construction and carries its own
    stratum.
    """
    axis = record.axis
    source_names = list(axis.targets) + [SCHEDULED_LABEL]
    participants = [record.participant_index(k) for k in range(1, record.K + 1)]
    observed = [record.observed_at(k) for k in range(1, record.K + 1)]
    fine = record.scale_of(FINE_UNIT)

    tensor = association_tensor(
        counts=record.sources.counts,
        exogenous=record.sources.calendar,
        participants=participants,
        observed=observed,
        source_names=source_names,
        target_names=list(axis.targets),
        n_context=N_CALENDAR,
        scales=[record.scale_of(u) for u in COARSE_UNITS],
    )
    priority = drop_self(tensor.merged(fine, merge), source_names, axis.targets)

    block = record.exogenous_at(fine)
    sources = np.concatenate(
        [record.sources.counts[fine - 1], block[:, N_CALENDAR:]], axis=1
    )
    fine_matrix = drop_self(
        association(
            sources=lagged(sources, participants[fine - 1], lag=1),
            targets=record.sources.counts[fine - 1],
            context=block[:, :N_CALENDAR],
            who=participants[fine - 1],
            weights=observed[fine - 1].astype(float),
        ),
        source_names,
        axis.targets,
    )

    pairs = _pairs(priority)
    coarse_flat = np.array([priority[i, j] for i, j in pairs])
    fine_flat = np.array([fine_matrix[i, j] for i, j in pairs])
    keep = coarse_flat > 0.0
    agreement = float(stats.spearmanr(coarse_flat[keep], fine_flat[keep]).statistic)

    n = int(keep.sum())
    order = visit_order(np.where(keep, coarse_flat, -np.inf))[:n]
    strongest = [i for i in np.argsort(-fine_flat) if keep[i]]
    depths = sorted({max(1, int(round(share * n))) for share in DEPTH_SHARES})
    curve = []
    for depth in depths:
        reached = set(order[:depth])
        curve.append(
            {
                "depth": depth,
                "share_of_space": depth / n,
                "top_5_found": len(reached & set(strongest[:5])),
                "top_10_found": len(reached & set(strongest[:10])),
            }
        )

    per_scale = {
        unit: float(
            stats.spearmanr(
                *(
                    np.array(
                        [
                            [
                                drop_self(
                                    tensor.per_scale[record.scale_of(unit)],
                                    source_names,
                                    axis.targets,
                                )[i, j],
                                fine_matrix[i, j],
                            ]
                            for i, j in pairs
                            if priority[i, j] > 0.0
                        ]
                    ).T
                )
            ).statistic
        )
        for unit in COARSE_UNITS
    }

    return {
        "merge": merge,
        "fine_unit": FINE_UNIT,
        "coarse_units": list(COARSE_UNITS),
        "n_ordered_pairs": n,
        "spearman_merged": agreement,
        "spearman_per_scale": per_scale,
        "depth_curve": curve,
        "recall_against_depth": [
            list(row)
            for row in recall_against_depth(coarse_flat, strongest[:10], depths)
        ],
    }


def anchor_shift_placebo(record, n_shifts: int = 99) -> Dict[str, object]:
    """
    The shift placebo on the anchor relation, unrepaired and repaired.

    The unrepaired null rolls by an arbitrary offset, which breaks the diurnal
    alignment of any pair the clock drives.  The repaired one rolls by whole days
    and inside each participant, so the rhythm survives and only the alignment
    moves.
    """
    axis = record.axis
    fine = record.scale_of(FINE_UNIT)
    blocks = _blocks(record, fine)
    period = 24
    target = record.sources.counts[fine - 1][:, axis.target_index(ANCHOR_READ)].astype(
        float
    )
    parent = lagged(
        record.exogenous_at(fine)[:, axis.exogenous_index(ANCHOR_SOURCE)][:, None],
        record.participant_index(fine),
        lag=1,
    )[:, 0]

    out = {}
    for name, roll in (("unrepaired", 1), ("repaired", period)):
        report = shift_null(target, parent, blocks, roll, n_shifts=n_shifts)
        out[name] = {
            "statistic": report.statistic,
            "p_value": report.p_value,
            "n_shifts": report.n_shifts,
            "beats_null": report.beats_null(),
        }
    out["relation"] = f"{ANCHOR_SOURCE} -> {axis.axis_name(ANCHOR_READ)}"
    out["scale"] = FINE_UNIT
    return out


def main(
    participants: Optional[int] = None,
    merge: str = "max",
    n_permutations: int = 199,
    n_shifts: int = 99,
    output_dir: str = "results/exp14",
    budget: Optional[RunBudget] = None,
) -> Dict[str, object]:
    os.makedirs(output_dir, exist_ok=True)
    ledger = BudgetLedger(budget or RunBudget.full())

    record, cohort, collapse = build_ladder(participants=participants)
    axis = record.axis
    print(cohort.summary())
    print()
    print(collapse.summary())

    audit = missingness_audit(record, n_permutations=n_permutations)
    print()
    print(audit.report())

    anchors = anchor_set(axis)
    frozen = anchors.freeze(os.path.join(output_dir, "anchors.json"))
    print(
        f"\nanchors frozen: {frozen['n_anchors']} declared, "
        f"{frozen['n_assertable']} assertable, digest {frozen['digest'][:16]}"
    )

    ordering = coarse_ordering(record, merge)
    print(
        f"\ncoarse ordering over {ordering['n_ordered_pairs']} ordered pairs at the "
        f"{FINE_UNIT} scale, Spearman {ordering['spearman_merged']:.3f}"
    )
    for row in ordering["depth_curve"]:
        print(
            f"  depth {row['depth']:>4}  {row['share_of_space']:.0%} of the space  "
            f"top 5 {row['top_5_found']}/5  top 10 {row['top_10_found']}/10"
        )

    placebo = anchor_shift_placebo(record, n_shifts=n_shifts)
    print(f"\nshift placebo on {placebo['relation']} at the {placebo['scale']} scale:")
    for name in ("unrepaired", "repaired"):
        row = placebo[name]
        print(
            f"  {name:<11} p={row['p_value']:.4f} over {row['n_shifts']} shifts, "
            f"beats the null: {row['beats_null']}"
        )

    arms = {
        arm: {
            "placement": {
                node: list(scales) for node, scales in placement(arm).items()
            },
            "n_candidates": len(calendar_candidates(axis, arm)),
        }
        for arm in ARMS
    }

    out = {
        "provenance": record.provenance,
        "units": list(LADDER_UNITS),
        "n_epochs": [record.n_epochs(k) for k in range(1, record.K + 1)],
        "collapse": {
            "into": collapse.into,
            "group": list(collapse.group),
            "episodes": collapse.episodes,
            "active_days": collapse.agreement.active_days,
            "days_in_agreement": collapse.agreement.days_in_agreement,
            "share_in_agreement": collapse.agreement.share_in_agreement,
            "max_residual": collapse.agreement.max_residual,
            "day_correlation": collapse.agreement.day_correlation,
            "coincident": [list(row) for row in collapse.agreement.coincident],
        },
        "missingness": {
            "declared": audit.declared,
            "accepted": audit.accepted,
            "verdicts": [str(v) for v in audit.verdicts],
        },
        "anchors": {
            "digest": frozen["digest"],
            "n_anchors": frozen["n_anchors"],
            "n_assertable": frozen["n_assertable"],
            "n_positive": len(anchors.of_sign(POSITIVE, True)),
            "n_negative": len(anchors.of_sign(NEGATIVE, True)),
        },
        "coarse_ordering": ordering,
        "shift_placebo": placebo,
        "calendar_arms": arms,
    }
    with open(os.path.join(output_dir, "exp14_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(output_dir, "exp14_budget.json"), "w") as f:
        json.dump(ledger.record([]), f, indent=2)
    write_provenance(output_dir)
    print(f"\nResults saved to {output_dir}/")
    return out


def _parse(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--participants",
        type=int,
        default=None,
        help="how many participants of the kept cohort to read; all of them by default",
    )
    parser.add_argument("--merge", default="max", choices=("max", "mean"))
    parser.add_argument("--permutations", type=int, default=199)
    parser.add_argument("--shifts", type=int, default=99)
    parser.add_argument("--output-dir", default="results/exp14")
    add_budget_arguments(parser)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse()
    main(
        participants=8 if args.quick else args.participants,
        merge=args.merge,
        n_permutations=49 if args.quick else args.permutations,
        n_shifts=19 if args.quick else args.shifts,
        output_dir=args.output_dir,
        budget=budget_from_args(args),
    )
