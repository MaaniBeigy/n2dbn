"""Emit the observational-arm numbers the manuscript prints, from the stored runs."""

import json
import sys

sys.path.insert(0, "/workspace")

from src.evaluation.arm1_report import (
    ablation_rows,
    changepoint_rows,
    clip_rows,
    control_rows,
    edges_by_scale,
    evidence_bearing,
    heterogeneity_rows,
    leg_rows,
    receipts,
    rung_summary,
    separation_summary,
    volume_rows,
)

ROOT = "/workspace/results"

with open(f"{ROOT}/exp15/exp15_results.json") as f:
    fit = json.load(f)
with open(f"{ROOT}/exp16/exp16_results.json") as f:
    trip = json.load(f)

search = fit["search"]
print(
    f"\n% ---- the graph ({search['n_candidates']} candidates, price "
    f"{search['price']:.2f}, null median {search['null_median']:.2f}, "
    f"null max {search['null_max']:.1f}) ----"
)
print(r"% Scale & Exogenous & Self & Cross-label & Total \\")
for row in edges_by_scale(fit):
    print(
        f"% $k={row['scale']}$, {row['name']:6s} & {row['exogenous']} & "
        f"{row['self']} & {row['cross_label']} & {row['total']} " + r"\\"
    )

print("\n% ---- the falsification ladder ----")
for row in rung_summary(fit):
    print(f"% {row['rung']:4s} {row['hits']}/{row['denominator']:<5} {row['reads']}")

print("\n% ---- the calendar ablation ----")
print(r"% Arm & Candidates & Edges & Objective & vs home \\")
for row in ablation_rows(fit):
    print(
        f"% {row['arm']:7s} & {row['candidates']} & {row['accepted']} & "
        f"{row['objective']:.2f} & {row['against_home']:+.2f} " + r"\\"
    )

print("\n% ---- the receipts, by reselection rate ----")
for row in receipts(fit):
    print(
        f"% {row['edge']:62s} {row['kind']:11s} beta {row['coefficient']:+10.3f} "
        f"std {row['standardized']:+8.3f} E {row['e_value']:9.3g} "
        f"reselected {row['reselected']:.4f}"
    )

print("\n% ---- per-label volume, fitted leg ----")
print(r"% Label & Real & Drawn & Ratio \\")
for row in volume_rows(trip):
    print(
        f"% {row['label']:16s} & {row['real']:.0f} & {row['drawn']:.1f} & "
        f"{row['ratio']:.2f} " + r"\\"
    )

print("\n% ---- the grid, by leg ----")
print(r"% Leg & Edges & Tested & Failing cells & Failing rows \\")
for row in leg_rows(trip):
    print(
        f"% {row['leg']:9s} & {row['edges']} & {row['tested']} & "
        f"{row['failed_cells']} & {len(row['failed_rows'])} " + r"\\"
    )
read = evidence_bearing(trip)
print(f"% rows failing their own replicate noise: {read['noise_rows']}")
print(f"% rows carrying evidence: {read['evidence_rows']}")
print(f"% of those, passed: {read['passed_rows']}; failed: {read['failed_rows']}")
print(f"% separation: {separation_summary(trip)}")

print("\n% ---- changepoint agreement, Poisson cost ----")
scales = ("week", "day", "hour")
by_scale = {s: {r["series"]: r for r in changepoint_rows(trip, s)} for s in scales}
print(r"% Label & F1 wk & AE wk & F1 day & AE day & F1 hr & AE hr \\")
for label in sorted(by_scale["week"]):
    cells = []
    for s in scales:
        row = by_scale[s][label]
        cells += [f"{row['f1']:.3f}", f"{row['annotation_error']:.2f}"]
    print(f"% {label:16s} & " + " & ".join(cells) + r" \\")

print("\n% ---- between-participant dispersion ----")
for row in heterogeneity_rows(trip):
    if "refused" in row:
        print(f"% {row['series']:16s} refused: {row['refused']}")
        continue
    print(
        f"% {row['series']:16s} real {row['real']:.4f} "
        f"[{row['real_lower']:.4f}, {row['real_upper']:.4f}]  "
        f"drawn {row['drawn']:.4f} [{row['drawn_lower']:.4f}, "
        f"{row['drawn_upper']:.4f}]  comparable={row['comparable']} "
        f"separation {row['separation']:.4f} sparse gap {row['sparse_gap']:.3f}"
    )

print("\n% ---- the declared controls ----")
for row in control_rows(trip):
    print(
        f"% {row['control']:32s} passed={row['passed']} "
        f"missed={row['missed']} spurious={row['spurious']}"
    )


print("\n% ---- where the predictor left the representable range ----")
for row in clip_rows(trip):
    print(f"% {row['label']:16s} total {row['total']:8d}  {row['by_end']}")
