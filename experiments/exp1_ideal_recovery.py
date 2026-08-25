"""
experiments/exp1_ideal_recovery.py
===================================
Experiment 1: Ideal structural and latent recovery on Families I-III.

Tests whether the N²DBN correctly recovers:
  * Ground-truth graph structure (SHD to 0, F1 to 1)
  * Regime sequences (Acc_S to 1)
  * Latent factor trajectories (factor correlation to 1)

Run
---
    python -m experiments.exp1_ideal_recovery [--quick] [--output-dir results/exp1]
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.evaluation.baselines import EmptyGraph, OracleGraph, OracleParams
from src.evaluation.protocol import EvaluationProtocol
from src.learning.structural_em import TopDownSEM
from src.synthetic.scenarios import ScenarioGrid
from src.utils.provenance import write_provenance


def main(
    quick: bool = False,
    jobs: int = 1,
    max_iter: int = 0,
    plots: bool = True,
    output_dir: str = "results/exp1",
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    write_provenance(output_dir)

    grid = ScenarioGrid.sanity_check() if quick else ScenarioGrid.full_benchmark()
    # Filter to families I, II, III only
    grid = ScenarioGrid(scenarios=[s for s in grid.scenarios if s.family in (1, 2, 3)])

    print(f"[Exp1] {len(grid)} scenarios, quick={quick}")

    # The edge penalty is the precision-recall knob of structure learning, and
    # the identifiability probe reports the level a non-edge reaches on any new
    # scenario.  On top of the penalty, every accepted edge must beat 32 shifted
    # copies of its own feature, circular at the coarsest scale and rolled
    # inside each parent block below it, which is what keeps latent-state
    # autocorrelation from being booked as a lagged self edge at the scales that
    # hold no true edge.
    sem_kw = dict(
        max_outer_iter=max_iter or (5 if quick else 10),
        # One coefficient cannot price an edge at both truth-bearing scales.
        # log T separates them by 1.79 while the prices they ask for differ by
        # four, so the coefficient is set per scale and held-out evidence picks
        # it: the coarsest scale reaches its lowest validation NLL at 0.5, above
        # 0.25 and above 2.0 on either side of it.
        lam_edge={1: 2.0, 2: 2.0, 3: 0.5},
        # Enumeration inside the in-degree bound is 211 parent sets per label at
        # the middle scale and 11 at the coarsest, which is cheaper than the
        # greedy passes it replaces and returns the certified per-target
        # maximizer, so the set does not depend on where the search started.
        sstep_kwargs=dict(confirm_edges=True, n_perm=32, search="exact"),
        verbose=0,
    )
    methods = {
        "n2dbn": TopDownSEM(**sem_kw),
        "empty_graph": EmptyGraph(**sem_kw),
        # Ground-truth parameters, no fitting: the ceiling the data allows.
        # Read AccS/FactCorr for n2dbn *relative to this row* - on scenarios
        # where the oracle itself scores 0.6, a fitted 0.55 is near-optimal,
        # not a failure.
        "oracle_params": OracleParams(),
    }

    protocol = EvaluationProtocol(
        methods=methods,
        n_replicates=2 if quick else 3,
        verbose=1,
        n_jobs=jobs,
    )

    csv_path = os.path.join(output_dir, "exp1_results.csv")
    json_path = os.path.join(output_dir, "exp1_results.json")
    results = protocol.run(grid, partial_csv=csv_path)

    protocol.save_csv(results, csv_path)
    protocol.save_json(results, json_path)

    print(
        "\n"
        + protocol.summary_table(
            results,
            metric_keys=["F1_mean", "SHD_total", "AccS_k1", "FactCorr_k1"],
        )
    )

    if plots:
        # One representative replicate per experiment: a timeline per scale
        # with true and learned edges, a snapshot sequence and a slice plot.
        try:
            from src.utils.temporal_viz import experiment_plots

            written = experiment_plots(
                json_path, os.path.join(output_dir, "plots"), grid=grid
            )
            print(f"[Exp1] {len(written)} figures in {output_dir}/plots/")
        except Exception as exc:
            print(f"[Exp1] plots skipped: {exc}")

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick", action="store_true", help="Use sanity-check grid (fast)"
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="replicates to fit in parallel (-1 = all cores)",
    )
    parser.add_argument(
        "--iters", type=int, default=0, help="override max SEM outer iterations"
    )
    parser.add_argument("--output-dir", default="results/exp1")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    main(
        quick=args.quick,
        jobs=args.jobs,
        max_iter=args.iters,
        output_dir=args.output_dir,
        plots=not args.no_plots,
    )
