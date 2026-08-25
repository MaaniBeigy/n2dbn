"""
experiments/exp5_algorithm_comparison.py
========================================
Experiment 5: the S-step of this model against published structure learners.

Every method fits the same latent-state model with the same parameter blocks on
the same data.  Only the graph differs: `n2dbn` searches parent sets against the
expected complete-data likelihood, while each classical method searches against
a criterion computed from the observed counts alone.  Differences in structure
recovery and in held-out likelihood are therefore attributable to the structure
search.

Run
---
    python -m experiments.exp5_algorithm_comparison [--quick] [--jobs N]
        [--algorithms hc-bic,pc,dynotears] [--output-dir results/exp5]
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.evaluation.baselines import (
    SORTNREGRESS,
    ClassicalStructure,
    EmptyGraph,
    SortnRegressStructure,
)
from src.evaluation.protocol import EvaluationProtocol
from src.learning.classical import ALGORITHMS, available_algorithms
from src.learning.structural_em import TopDownSEM
from src.synthetic.scenarios import ScenarioGrid
from src.utils.provenance import write_provenance

# `mmhc` runs two orders of magnitude slower than the rest at benchmark sizes,
# so it enters only when the algorithm list names it or asks for all of them.
DEFAULT_ALGORITHMS = (
    "hc-bic",
    "hc-bdeu",
    "exhaustive",
    "pc",
    "ges",
    "tree",
    "dynotears",
    "granger",
)


def resolve_algorithms(spec: str) -> list:
    """Algorithm names from a comma-separated list, `default` or `all`.

    A sorting baseline is accepted wherever a published learner is, since it is
    scored by the same code on the same graphs, and it needs no optional
    dependency so it is never skipped.  `all` keeps its meaning of every
    published learner, because widening it would change what a stored `all` run
    contains without the caller asking for it; a sorting row is requested by
    name.
    """
    usable = set(available_algorithms()) | set(SORTNREGRESS)
    known = list(ALGORITHMS) + list(SORTNREGRESS)
    if spec == "all":
        wanted = [a for a in ALGORITHMS if a != "empty"]
    elif spec == "default":
        wanted = list(DEFAULT_ALGORITHMS)
    else:
        wanted = [a.strip() for a in spec.split(",") if a.strip()]
    unknown = [a for a in wanted if a not in known]
    if unknown:
        raise SystemExit(f"unknown algorithms {unknown}, expected {known}")
    missing = [a for a in wanted if a not in usable]
    if missing:
        print(f"[Exp5] skipping {missing}: pgmpy is not installed")
    return [a for a in wanted if a in usable]


def main(
    quick: bool = False,
    jobs: int = 1,
    max_iter: int = 0,
    algorithms: str = "default",
    device: str = "cpu",
    plots: bool = True,
    output_dir: str = "results/exp5",
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    write_provenance(output_dir)

    grid = ScenarioGrid.sanity_check() if quick else ScenarioGrid.full_benchmark()
    grid = ScenarioGrid(scenarios=[s for s in grid.scenarios if s.family in (2, 3)])
    names = resolve_algorithms(algorithms)
    print(f"[Exp5] {len(grid)} scenarios, {len(names)} algorithms, quick={quick}")

    # Every edge the local score accepts must also beat 32 shifted copies of its
    # own feature, circular at the coarsest scale and rolled inside each parent
    # block below it.  The classical entries run with structure search off, so
    # the setting reaches only the n2dbn row.
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
    methods = {"n2dbn": TopDownSEM(**sem_kw), "empty_graph": EmptyGraph(**sem_kw)}
    for name in names:
        if name in SORTNREGRESS:
            methods[name] = SortnRegressStructure(algorithm=name, **sem_kw)
        else:
            methods[name] = ClassicalStructure(algorithm=name, device=device, **sem_kw)

    protocol = EvaluationProtocol(
        methods=methods,
        n_replicates=2 if quick else 3,
        verbose=1,
        n_jobs=jobs,
    )

    csv_path = os.path.join(output_dir, "exp5_results.csv")
    json_path = os.path.join(output_dir, "exp5_results.json")
    results = protocol.run(grid, partial_csv=csv_path)

    protocol.save_csv(results, csv_path)
    protocol.save_json(results, json_path)

    print(
        "\n"
        + protocol.summary_table(
            results,
            metric_keys=["Prec_k2", "Rec_k2", "F1_k2", "SHD_total", "val_loss"],
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
            print(f"[Exp5] {len(written)} figures in {output_dir}/plots/")
        except Exception as exc:
            print(f"[Exp5] plots skipped: {exc}")

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick",
        action="store_true",
        help="use the sanity grid instead of the full one",
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
    parser.add_argument(
        "--algorithms",
        default="default",
        help="comma-separated names, or 'default' or 'all'",
    )
    parser.add_argument(
        "--device", default="cpu", help="torch device for the DYNOTEARS solver"
    )
    parser.add_argument("--output-dir", default="results/exp5")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    main(
        quick=args.quick,
        jobs=args.jobs,
        max_iter=args.iters,
        algorithms=args.algorithms,
        device=args.device,
        output_dir=args.output_dir,
        plots=not args.no_plots,
    )
