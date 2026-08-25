"""
experiments/exp17_coherence_gap.py
==================================
Experiment 17: what fitting one granularity at a time costs in coherence.

Why this run exists
-------------------
The manuscript states that fitting one granularity at a time gives answers that
disagree, because a total at one timescale differs from the sum of its parts at
another.  Every stored run reports `RecErr_X = RecErr_Y = 0` for every method
including the empty graph, which reads as evidence for that sentence and is not:
`MultiScaleData.check_reconciliation` reads the observed arrays, and those are
built by aggregation, so the zero is arithmetic and holds whatever was fitted.

`evaluation.coherence` supplies the quantity the sentence needs.  Each scale
predicts its own mean without consulting the scale above it, the children of a
block are summed, and the gap against the parent's own prediction is reported.
For the nested model the sum returns the parent mean by the tower property, so
the gap is zero up to floating point.  For two models fitted at two
granularities nothing constrains it.

Two arms
--------
`nested` carries the model at the parameters the generator wrote, so the gap it
reports is a property of the construction and not of an estimate.  `per_scale`
gives every scale its own zero-inflated emission through
`initialize_from_data(every_scale_coarse=True)` and fits it with the downward
message switched off, which is one model per granularity and nothing tying them.

Run
---
    python -m experiments.exp17_coherence_gap --quick

    python -m experiments.exp17_coherence_gap --output-dir results/exp17
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np

from src.core.graph import MultiScaleDBNGraph
from src.data.initialize import initialize_from_data
from src.evaluation.coherence import coherence_gap, predicted_means
from src.learning.evidence import evidence_chain
from src.synthetic.families import make_generator
from src.synthetic.scenarios import ScenarioGrid
from src.utils.provenance import write_provenance

NESTED = "nested"
PER_SCALE = "per_scale"


def one_replicate(cfg, seed: int) -> List[Dict[str, object]]:
    """Both arms on one generated replicate, one row each."""
    gen = make_generator(cfg, seed=seed)
    ds = gen.generate(seed=seed)
    data = ds.data
    graph = MultiScaleDBNGraph(hierarchy=ds.hierarchy, label_names=gen.label_names)

    rows: List[Dict[str, object]] = []

    summaries = evidence_chain(data, ds.params, graph).summaries
    gaps = coherence_gap(data, predicted_means(data, ds.params, summaries))
    rows.append({"scenario": cfg.short_id(), "seed": seed, "arm": NESTED, **gaps})

    per_scale = initialize_from_data(
        data,
        M_k=[p.regime.M for p in ds.params],
        r_k=[p.factor.r for p in ds.params],
        max_lag=[1] * data.K,
        d_ctx=[data.get_C(k).shape[1] for k in range(1, data.K + 1)],
        seed=seed,
        every_scale_coarse=True,
    )
    summaries = evidence_chain(data, per_scale, graph, couple_scales=False).summaries
    gaps = coherence_gap(data, predicted_means(data, per_scale, summaries))
    rows.append({"scenario": cfg.short_id(), "seed": seed, "arm": PER_SCALE, **gaps})
    return rows


def summarize(rows: List[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    """Each arm's gap over the grid, which is what the manuscript prints."""
    out: Dict[str, Dict[str, float]] = {}
    for arm in (NESTED, PER_SCALE):
        gaps = np.array([float(r["gap"]) for r in rows if r["arm"] == arm])
        if not gaps.size:
            continue
        out[arm] = {
            "n": int(gaps.size),
            "mean": float(gaps.mean()),
            "median": float(np.median(gaps)),
            "max": float(gaps.max()),
        }
    return out


def main(
    quick: bool = False, replicates: int = 3, output_dir: str = "results/exp17"
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    write_provenance(output_dir)
    grid = ScenarioGrid.sanity_check() if quick else ScenarioGrid.full_benchmark()
    scenarios = list(grid)
    print(f"[Exp17] {len(scenarios)} scenarios x {replicates} replicates", flush=True)

    rows: List[Dict[str, object]] = []
    for i, cfg in enumerate(scenarios):
        for rep in range(replicates):
            rows.extend(one_replicate(cfg, seed=cfg.seed * 1000 + rep))
        print(f"  {i + 1}/{len(scenarios)} {cfg.short_id()}", flush=True)

    summary = summarize(rows)
    for arm, s in summary.items():
        print(
            f"{arm:10s} n={s['n']:3d} mean gap {s['mean']:.6g} "
            f"median {s['median']:.6g} max {s['max']:.6g}"
        )

    with open(os.path.join(output_dir, "exp17_results.json"), "w") as handle:
        json.dump({"rows": rows, "summary": summary}, handle, indent=2)
    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--output-dir", default="results/exp17")
    args = ap.parse_args()
    main(quick=args.quick, replicates=args.replicates, output_dir=args.output_dir)
