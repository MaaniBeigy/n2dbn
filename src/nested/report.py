"""Conservative claim checks from revised artifacts, including missing runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from . import FORMULATION
from .evaluate import summarize_forecasts, write_json


def result_table(root, lines, name, title, columns, rows):
    """Readable tables and unrounded CSVs from exactly the same populations."""
    if not rows:
        return
    folder = root / "tables"
    folder.mkdir(exist_ok=True)
    with (folder / (name + ".csv")).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)

    def display(value):
        if value is None:
            return "—"
        if isinstance(value, (float, np.floating)):
            return f"{value:.6g}"
        return str(value)

    lines.extend(
        [
            "",
            "## " + title,
            "",
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
        ]
    )
    lines.extend("| " + " | ".join(display(v) for v in row) + " |" for row in rows)
    lines.extend(["", f"[Unrounded CSV](tables/{name}.csv)"])


def paired_contrast(rows, left, right, metric="shd", bootstrap=2000, seed=0):
    """Left minus right, preserving fixed generating scenarios in resampling."""
    lookup = {
        (r.get("scenario"), r.get("replicate"), r.get("method")): r
        for r in rows
        if r.get("status") == "complete" and "metrics" in r
    }
    groups = {}
    for scenario, replicate, method in lookup:
        if method != left or (scenario, replicate, right) not in lookup:
            continue
        a = lookup[(scenario, replicate, left)]["metrics"][metric]
        b = lookup[(scenario, replicate, right)]["metrics"][metric]
        groups.setdefault(scenario, []).append(a - b)
    if not groups:
        return {"pairs": 0, "mean": None, "ci95": None}
    mean = float(np.mean([np.mean(values) for values in groups.values()]))
    if any(len(values) < 2 for values in groups.values()):
        return {"pairs": sum(map(len, groups.values())), "mean": mean, "ci95": None}
    rng = np.random.default_rng(seed)
    draws = np.mean(
        [
            rng.choice(values, (bootstrap, len(values))).mean(axis=1)
            for values in groups.values()
        ],
        axis=0,
    )
    return {
        "pairs": sum(map(len, groups.values())),
        "scenarios": len(groups),
        "mean": mean,
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "interval": "paired bootstrap within each fixed scenario; equal scenario weights",
    }


def report(root):
    root = Path(root)
    sections, claims = {}, []
    for name in (
        "synthetic",
        "benchmark",
        "lifesnaps",
        "forecast",
        "forecast-extended",
        "scalability",
    ):
        path = root / name
        ledger, config = path / "runs.json", path / "config.json"
        rows = json.loads(ledger.read_text()) if ledger.exists() else []
        settings = json.loads(config.read_text()) if config.exists() else {}
        valid = settings.get("model_formulation") == FORMULATION
        complete = (
            valid and bool(rows) and all(r.get("status") == "complete" for r in rows)
        )
        sections[name] = {
            "planned": len(rows),
            "completed": sum(r.get("status") == "complete" for r in rows),
            "complete": complete,
            "model_formulation": settings.get("model_formulation"),
            "source_digest": settings.get("source_digest"),
            "packages": settings.get("packages", {}),
            "failures": [
                {"key": r["key"], "status": r.get("status"), "reason": r.get("reason")}
                for r in rows
                if r.get("status") != "complete"
            ],
        }
        if name == "synthetic":
            for label, left, right in (
                (
                    "Nesting improves SHD with latent orders matched",
                    "independent+latent",
                    "nested+latent",
                ),
                (
                    "Nested latent model beats the empty graph in SHD",
                    "empty",
                    "nested+latent",
                ),
                (
                    "Latent conditioning improves SHD within nested models",
                    "nested-no-latent",
                    "nested+latent",
                ),
            ):
                contrast = paired_contrast(rows, left, right)
                enough = (
                    complete
                    and contrast.get("scenarios") == 4
                    and contrast["pairs"] >= 40
                )
                interval = contrast.get("ci95")
                status = "inconclusive"
                if enough and interval is not None:
                    status = "supported" if interval[0] > 0 else "not_supported"
                claims.append(
                    {
                        "claim": label,
                        "status": status,
                        "evidence": contrast,
                        "criterion": "all planned runs complete; four scenarios, >=40 pairs; paired SHD difference CI strictly above zero",
                    }
                )
            methods = sorted({r.get("method") for r in rows if r.get("method")})
            sections[name]["methods"] = {}
            for method in methods:
                own = [
                    r
                    for r in rows
                    if r.get("method") == method and r.get("status") == "complete"
                ]
                sections[name]["methods"][method] = {
                    "completed": len(own),
                    **{
                        metric: (
                            float(np.mean([r["metrics"][metric] for r in own]))
                            if own
                            else None
                        )
                        for metric in (
                            "f1",
                            "shd",
                            "empty_shd",
                            "self_recall",
                            "cross_recall",
                        )
                    },
                }
        elif name == "benchmark":
            sections[name]["methods"] = {
                row.get("method", row["key"]): {
                    key: row.get(key)
                    for key in (
                        "status",
                        "reachable_label_relations",
                        "reachable_label_matches",
                        "stationary_target_edges",
                        "manifest",
                    )
                }
                for row in rows
            }
            claims.append(
                {
                    "claim": "CalendarBench typed-rule recovery supports the manuscript claim",
                    "status": "inconclusive",
                    "criterion": "inspect manifest reachability and channel coefficients; label-only correspondence cannot establish cross-quantity recovery",
                    "evidence": sections[name]["methods"],
                }
            )
        elif name == "lifesnaps":
            full = next((r for r in rows if r["key"] == "full"), {})
            fidelity = full.get("fidelity", {})
            volumes = fidelity.get("volumes", {})
            sections[name]["volumes"] = volumes
            cells = fidelity.get("nonstationarity", {}).get("cells", [])
            sections[name]["form_statuses"] = {
                status: sum(c["status"] == status for c in cells)
                for status in ("equivalent", "not_equivalent", "descriptive")
            }
            sections[name]["changepoints"] = fidelity.get("nonstationarity", {}).get(
                "changepoints", []
            )
            stability = path / "stability.json"
            sections[name]["stability"] = (
                json.loads(stability.read_text()) if stability.exists() else None
            )
            enough = (
                complete
                and settings.get("participants", 0) >= 51
                and settings.get("weeks", 0) >= 8
                and settings.get("replicates", 0) >= 10
                and settings.get("pairs", 0) >= 25
            )
            observed_audit = path / "audit.json"
            if observed_audit.exists():
                sections[name]["audit"] = json.loads(observed_audit.read_text())
                enough &= (
                    sections[name]["audit"].get("valid", False)
                    and sections[name]["audit"].get("participants", 0) >= 51
                )
            else:
                enough = False
            baseline = next((r for r in rows if r["key"] == "intercept"), {})
            baseline_volumes = baseline.get("fidelity", {}).get("volumes", {})
            improvement = {}
            for channel in ("counts", "booked_duration"):
                improvement[channel] = {}
                for population in ("all_label", "four_label"):
                    metric = population + "_mean_absolute_log_ratio"
                    full_error = volumes.get(channel, {}).get(metric)
                    base_error = baseline_volumes.get(channel, {}).get(metric)
                    improvement[channel][population] = {
                        "full_error": full_error,
                        "intercept_error": base_error,
                        "factor": (
                            base_error / full_error
                            if full_error is not None
                            and full_error > 0
                            and base_error is not None
                            else None
                        ),
                    }
            sections[name]["improvement_over_intercept"] = improvement
            factors = [
                improvement[channel]["all_label"]["factor"] for channel in improvement
            ]
            claims.append(
                {
                    "claim": "LifeSnaps count and duration volume errors are at least four times smaller than the intercept baseline",
                    "status": (
                        (
                            "supported"
                            if all(x is not None and x >= 4 for x in factors)
                            else "not_supported"
                        )
                        if enough
                        else "inconclusive"
                    ),
                    "criterion": "intercept/full mean absolute log-volume error >=4 for both channels, using all labels; four-label subset also reported separately",
                    "evidence": improvement,
                }
            )
            sections[name]["anchors"] = full.get("anchors")
            sections[name]["accepted_edges"] = len(full.get("fit", {}).get("graph", []))
            sections[name]["graph"] = full.get("fit", {}).get("graph", [])
            stable = sections[name]["stability"]
            sections[name]["stable_edges"] = (
                sum(e["stable"] for e in stable.get("edges", [])) if stable else None
            )
            # A transparent operational threshold, not a p-value or a theorem.
            by_label = volumes.get("counts", {}).get("by_label", {})
            good = bool(by_label) and all(
                v.get("ratio") is not None and 0.8 <= v["ratio"] <= 1.25
                for v in by_label.values()
            )
            claims.append(
                {
                    "claim": "LifeSnaps reproduces whole-label count volumes",
                    "status": (
                        ("supported" if good else "not_supported")
                        if enough
                        else "inconclusive"
                    ),
                    "criterion": "every count-volume ratio in [0.8, 1.25]; all six panel arms and 50 half fits completed",
                    "evidence": by_label,
                }
            )
            claims.append(
                {
                    "claim": "LifeSnaps preserves every tested nonstationarity form",
                    "status": (
                        (
                            "supported"
                            if cells and all(c["status"] == "equivalent" for c in cells)
                            else "not_supported"
                        )
                        if enough
                        else "inconclusive"
                    ),
                    "criterion": "every applicable form passes its paired equivalence interval; descriptive cells do not count as passes",
                    "evidence": sections[name]["form_statuses"],
                }
            )
        elif name.startswith("forecast"):
            summaries = summarize_forecasts(rows)
            sections[name]["methods"] = summaries
            if name == "forecast-extended":
                continue
            full = summaries.get("nested", {})
            enough = (
                complete
                and full.get("runs", 0) >= 40
                and settings.get("series", 0) >= 40
                and settings.get("panels") == "min"
                and settings.get("split") == "cutoff"
                and settings.get("horizon_blocks") == 2
            )
            coherent = (
                full.get("failed_runs") == 0
                and full.get("coherence", {}).get("max", np.inf) <= 1e-10
                if full.get("coherence")
                else False
            )
            claims.append(
                {
                    "claim": "Nested forecasts are coherent",
                    "status": (
                        ("supported" if coherent else "not_supported")
                        if enough
                        else "inconclusive"
                    ),
                    "criterion": "all planned predictions finite; maximum relative discrepancy <=1e-10",
                    "evidence": full,
                }
            )
            names = [
                "seasonal-naive",
                "DLinear-bottom-up",
                "DLinear-ols",
                "DLinear-nnls",
                "PatchTST-bottom-up",
                "PatchTST-ols",
                "PatchTST-nnls",
            ]
            valid = full.get("failed_runs") == 0 and full.get("mae") is not None
            wins = valid and all(
                summaries.get(n, {}).get("failed_runs") == 0
                and summaries[n].get("mae") is not None
                and full["mae"]["mean"] <= summaries[n]["mae"]["mean"]
                and full["mae"]["median"] <= summaries[n]["mae"]["median"]
                for n in names
            )
            claims.append(
                {
                    "claim": "Nested forecasts match the accuracy of the strongest coherent baseline",
                    "status": (
                        ("supported" if wins else "not_supported")
                        if enough
                        else "inconclusive"
                    ),
                    "criterion": "mean and median standardized hourly MAE no worse than every listed coherent baseline, with no failures; descriptive comparison, not significance",
                    "evidence": {n: summaries.get(n) for n in ["nested"] + names},
                }
            )
        else:
            claims.append(
                {
                    "claim": "Scalability sweep is fully measured",
                    "status": "supported" if complete else "inconclusive",
                    "criterion": "all four one-factor sweeps completed; timed-out cells remain explicit",
                    "evidence": sections[name],
                }
            )
    digests = {
        (section["source_digest"], json.dumps(section["packages"], sort_keys=True))
        for section in sections.values()
        if section["source_digest"]
    }
    if len(digests) > 1:
        for claim in claims:
            claim["status"] = "inconclusive"
        provenance_warning = "Experiment source digests or package versions differ; rerun under one image before combining evidence."
    else:
        provenance_warning = None
    output = {
        "model_formulation": FORMULATION,
        "provenance_warning": provenance_warning,
        "interpretation": "Operational checks for revising claims, not automatic proof of manuscript conclusions.",
        "experiments": sections,
        "claims": claims,
    }
    write_json(root / "claims.json", output)
    lines = [
        "# Revised experiment report",
        "",
        output["interpretation"],
        "",
        "Old exp* results and smoke checks are excluded.",
        "",
        "| Claim | Assessment |",
        "|---|---|",
    ]
    lines += [f"| {claim['claim']} | {claim['status']} |" for claim in claims]
    lines += [
        "",
        "## Run completeness",
        "",
        "| Experiment | Complete / planned |",
        "|---|---|",
    ]
    lines += [
        f"| {name} | {section['completed']} / {section['planned']} |"
        for name, section in sections.items()
    ]
    if provenance_warning:
        lines += ["", provenance_warning]
    recovery = sections["synthetic"].get("methods", {})
    result_table(
        root,
        lines,
        "recovery",
        "Synthetic recovery (completed runs)",
        [
            "Method",
            "Completed",
            "F1",
            "SHD",
            "Empty SHD",
            "Self recall",
            "Cross recall",
        ],
        [
            [method]
            + [
                values.get(key)
                for key in (
                    "completed",
                    "f1",
                    "shd",
                    "empty_shd",
                    "self_recall",
                    "cross_recall",
                )
            ]
            for method, values in recovery.items()
        ],
    )
    for name in ("forecast", "forecast-extended"):
        methods = sections[name].get("methods", {})
        result_table(
            root,
            lines,
            name,
            name + " (errors summarize finite completed runs)",
            [
                "Method",
                "Completed",
                "Planned",
                "Failed/missing",
                "Extreme",
                "MAE mean",
                "MAE median",
                "MAE p90",
                "MAE max",
                "MSE mean",
                "MSE median",
                "Coherence max",
            ],
            [
                [
                    method,
                    v["finite_completed_runs"],
                    v["runs"],
                    v["failed_runs"],
                    v["extreme_runs"],
                ]
                + [
                    (v.get("mae") or {}).get(s)
                    for s in ("mean", "median", "p90", "max")
                ]
                + [(v.get("mse") or {}).get(s) for s in ("mean", "median")]
                + [(v.get("coherence") or {}).get("max")]
                for method, v in methods.items()
            ],
        )
    life = sections["lifesnaps"]
    for channel, values in life.get("volumes", {}).items():
        result_table(
            root,
            lines,
            "lifesnaps-" + channel,
            "LifeSnaps " + channel,
            ["Label", "Observed", "Simulated mean", "Ratio", "Absolute log ratio"],
            [
                [label]
                + [
                    v.get(key)
                    for key in (
                        "observed",
                        "simulated_mean",
                        "ratio",
                        "absolute_log_ratio",
                    )
                ]
                for label, v in values.get("by_label", {}).items()
            ],
        )
    result_table(
        root,
        lines,
        "lifesnaps-error",
        "LifeSnaps error against the intercept baseline",
        ["Channel", "Labels", "Full error", "Intercept error", "Improvement factor"],
        [
            [channel, population, v["full_error"], v["intercept_error"], v["factor"]]
            for channel, comparisons in life.get(
                "improvement_over_intercept", {}
            ).items()
            for population, v in comparisons.items()
        ],
    )
    result_table(
        root,
        lines,
        "lifesnaps-graph",
        "LifeSnaps accepted graph",
        [
            "Scale",
            "Target",
            "Source kind",
            "Source",
            "Lag",
            "Count coefficient",
            "Duration coefficient",
        ],
        [
            [
                edge[key]
                for key in (
                    "scale",
                    "target",
                    "kind",
                    "source",
                    "lag",
                    "count_coefficient",
                    "duration_coefficient",
                )
            ]
            for edge in life.get("graph", [])
        ],
    )
    lines += ["", "## Criteria and evidence", ""]
    for claim in claims:
        lines += [f"- **{claim['claim']}**: {claim['criterion']}"]
    lines += [
        "",
        "Full measurements, failure reasons, label ratios, stability frequencies, paired intervals and forecast tails are in [claims.json](claims.json).",
        "",
        "No manuscript numbers are updated automatically. Copy the new measurements only after reviewing the diagnostics and limitations.",
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(root / "REPORT.md")
    return output
