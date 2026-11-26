"""Compare nested and independent latent DBNs on paired structure-recovery records."""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean

from src.core.graph import MultiScaleDBNGraph
from src.core.scales import ScaleHierarchy
from src.core.variables import MultiScaleData
from src.data.initialize import initialize_from_data
from src.evaluation.metrics import StructureMetrics
from src.learning.structural_em import TopDownSEM
from src.synthetic.families import make_generator
from src.synthetic.scenarios import ScenarioGrid
from src.utils.provenance import write_provenance


def fit(data, params, scales, cfg, iterations):
    """Fit the local scales with the corresponding original search budgets."""
    graph = MultiScaleDBNGraph(data.hierarchy, data.label_names)
    search = TopDownSEM(
        max_outer_iter=iterations,
        lam_edge={i: 0.5 if k == cfg.K else 2.0 for i, k in enumerate(scales, 1)},
        sstep_kwargs=dict(
            max_lag={i: cfg.max_lag[k - 1] for i, k in enumerate(scales, 1)},
            max_indegree={
                i: max(1, cfg.max_indegree[k - 1]) for i, k in enumerate(scales, 1)
            },
            confirm_edges=False,
            search="exact",
        ),
        verbose=0,
    )
    return search.fit(data, params, graph)[1]


def paired_record(task):
    """Return both graph scores and their paired differences on one record."""
    cfg, replicate, iterations = task
    seed = cfg.seed * 1000 + replicate
    record = make_generator(cfg, seed=seed).generate(seed=seed)
    data, _, _ = record.data.split()
    scales = range(1, data.K + 1)
    initial = dict(
        M_k=[p.M for p in record.params],
        r_k=[p.r for p in record.params],
        max_lag=cfg.max_lag,
        d_ctx=[c.shape[1] for c in data.C],
        seed=seed,
    )
    nested = fit(data, initialize_from_data(data, **initial), scales, cfg, iterations)
    per_scale = initialize_from_data(data, every_scale_coarse=True, **initial)
    truth = MultiScaleDBNGraph(data.hierarchy, data.label_names)
    independent = MultiScaleDBNGraph(data.hierarchy, data.label_names)
    for k in scales:
        i = k - 1
        local = MultiScaleData(
            ScaleHierarchy(1, [len(data.X[i])], [None]),
            data.label_names,
            [data.X[i]],
            [data.Y[i]],
            [data.C[i]],
            observed=data.get_observed(k),
        )
        per_scale[i].k = 1
        per_scale[i].regime.cross = None
        fitted = fit(local, [per_scale[i]], [k], cfg, iterations)
        for label in range(data.L):
            independent.set_parents(k, label, fitted.get_parents(1, label))
            truth.set_parents(k, label, record.params[i].parent_sets.get(label, set()))

    row = dict(scenario=cfg.name, replicate=replicate, seed=seed)
    for name, graph in (("nested", nested), ("independent", independent)):
        metrics = StructureMetrics.compute(graph, truth).to_dict()
        for key in ("F1_k2", "F1_k3", "SHD_total"):
            row[f"{key}_{name}"] = metrics[key]
            if name == "independent":
                row[f"{key}_difference"] = row[f"{key}_nested"] - metrics[key]
    return row


def main():
    """Write one paired row per scenario and replicate."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/latent-matched-recovery")
    )
    args = parser.parse_args()
    grid = [cfg for cfg in ScenarioGrid.full_benchmark() if cfg.family in (2, 3)]
    if args.quick:
        grid = grid[:1]
    tasks = [
        (cfg, rep, 1 if args.quick else 10)
        for cfg in grid
        for rep in range(1 if args.quick else 3)
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_provenance(str(args.output_dir))
    rows = []
    with (
        ProcessPoolExecutor(max_workers=args.jobs) as pool,
        (args.output_dir / "paired.csv").open("w", newline="") as output,
    ):
        for row in pool.map(paired_record, tasks):
            if not rows:
                writer = csv.DictWriter(output, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            output.flush()
            rows.append(row)
    for key in ("F1_k2", "F1_k3", "SHD_total"):
        print(
            key,
            "mean paired difference",
            mean(row[f"{key}_difference"] for row in rows),
        )


if __name__ == "__main__":
    main()
