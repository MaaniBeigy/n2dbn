"""
experiments/exp8_bnlearn_tiling.py
==================================
Experiment 8: structure and conditional-probability recovery against an
external ground truth.

Every other experiment in this repository scores a learner against data this
repository generated, which is the weakest possible evidence base for a claim
about a learner.  This one scores against the bnlearn repository: released
networks with an exact directed graph and exact conditional probability
tables, tiled into a two-slice temporal network by turning each static edge
into a lag-1 edge.  pgmpy vendors the networks inside its wheel, so the whole
experiment runs with no network access.

Two questions, and every method answers both on its own learned graph:
  * structure, scored by SHD, precision, recall, F1 and AUPRC against the
    released graph;
  * parameters, scored by the mean total variation distance per conditioning
    row between the conditional a method's own parent set implies and the
    released tables, which is the external counterpart of experiment 7.  A
    learned parent set of a different shape than the true one cannot be
    compared entry by entry, so what is compared is the distribution of a node
    given the whole previous epoch, which both models define.  The table
    estimated on the true graph is also reported, as the floor the sample size
    allows rather than as a method score.

The free quantities of this model, the regime count, the factor dimension and
the edge penalty, are selected on held-out log evidence rather than declared.
A classical learner chooses its penalty through its own score, so declaring
these would make the comparison a comparison of declarations.  The block length
of the imposed hierarchy stays a declaration, because it states which
timescales are being asked about and no held-out score supplies a question.

Run
---
    python -m experiments.exp8_bnlearn_tiling [--quick] [--networks asia,child]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.benchmarks.bnlearn_tiling import (
    cpt_error,
    predictive_cpt_error,
    sample_tiled,
    score_structure,
    tile_to_2tbn,
    to_counts,
)
from src.core.graph import MultiScaleDBNGraph
from src.evaluation.metrics import edge_auprc
from src.learning.classical import available_algorithms, learn_structure
from src.utils.provenance import write_provenance

# `mmhc` runs two orders of magnitude slower than the rest at these sizes, and
# `pc` and `ges` grow with both node count and sample size.
# `n2dbn` is this model's own local score; the rest are the published
# structure learners it is being compared against.
DEFAULT_ALGORITHMS = (
    "n2dbn",
    "hc-bic",
    "hc-bdeu",
    "exhaustive",
    "tree",
    "dynotears",
    "granger",
    "empty",
)
QUICK_NETWORKS = ("asia", "cancer", "earthquake")
FULL_NETWORKS = ("asia", "cancer", "earthquake", "sachs", "child", "insurance")


def _resolve(spec: str, pool) -> List[str]:
    """Names from a comma-separated list, `default` or `all`."""
    if spec == "all":
        return list(pool)
    if spec == "default":
        return list(pool)
    wanted = [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [w for w in wanted if w not in pool]
    if unknown:
        raise SystemExit(f"unknown names {unknown}, expected some of {list(pool)}")
    return wanted


def _edge_scores(X: np.ndarray, max_lag: int) -> Dict:
    """A score per candidate lagged edge, from the pairwise BIC gain."""
    T, L = X.shape
    scores = {}
    x = np.log1p(X.astype(float))
    for dst in range(L):
        y = x[:, dst]
        for src in range(L):
            for lag in range(1, max_lag + 1):
                a, b = x[:-lag, src], y[lag:]
                if a.std() < 1e-12 or b.std() < 1e-12:
                    scores[(src, lag, dst)] = 0.0
                    continue
                rho = float(np.corrcoef(a, b)[0, 1])
                # Half the deviance of the correlation, which is the BIC gain
                # of a single coefficient up to the shared penalty.
                scores[(src, lag, dst)] = (
                    -0.5 * len(b) * np.log(max(1.0 - rho * rho, 1e-12))
                )
    return scores


def _hierarchy(X: np.ndarray, labels: List[str], block: int):
    """Impose a two-scale hierarchy on a flat tiled series."""
    from src.core.scales import ScaleHierarchy
    from src.core.variables import MultiScaleData

    T = (X.shape[0] // block) * block
    Xf = X[:T].astype(np.int64)
    coarse = Xf.reshape(T // block, block, Xf.shape[1]).sum(axis=1)
    # A tiled state carries no duration, so an event occupies its own slot.
    return MultiScaleData(
        hierarchy=ScaleHierarchy.from_uniform_branching(T, [block]),
        label_names=list(labels),
        X=[Xf, coarse],
        Y=[Xf.copy(), coarse.copy()],
        C=[np.zeros((T, 0)), np.zeros((T // block, 0))],
    )


def block_length(n_epochs: int) -> int:
    """
    Block length of the imposed hierarchy, as the square root of the series.

    A tiled network has one timescale, so the second scale is the analyst's
    choice and cannot be read off the data.  The fine-scale score is a
    within-block allocation, so its evidence per block grows with the block
    length while the number of blocks shrinks; the square root balances the two
    and is fixed before any graph is scored.
    """
    return max(2, int(round(np.sqrt(max(n_epochs, 4)))))


def _n2dbn_parents(
    X: np.ndarray,
    labels: List[str],
    max_lag: int,
    max_indegree: int,
    iters: int,
    lam_grid: Sequence[float],
    seed: int = 0,
) -> Dict[int, set]:
    """Lagged parents the latent-conditional score selects on the tiled counts.

    The regime count, the factor dimension and the edge penalty are all free
    quantities of this model, so all three are selected here on held-out log
    evidence rather than declared.  Declaring them would make the comparison
    against the classical learners, which choose their own penalty through
    their own score, a comparison of declarations.
    """
    from src.learning.model_selection import select_model
    from src.learning.structural_em import TopDownSEM

    block = block_length(X.shape[0])
    split = (X.shape[0] * 3) // 4
    train = _hierarchy(X[:split], labels, block)
    valid = _hierarchy(X[split:], labels, block)

    def sem(lam: float):
        return TopDownSEM(
            max_outer_iter=iters,
            lam_edge=lam,
            sstep_kwargs=dict(
                confirm_edges=True,
                n_perm=32,
                max_lag={1: max_lag, 2: 1},
                max_indegree={1: max_indegree, 2: max_indegree},
            ),
            verbose=0,
        )

    sel = select_model(
        train,
        valid,
        sem,
        M_grid=(1, 2),
        r_grid=(0,),
        lam_grid=lam_grid,
        max_lag=[max_lag, 1],
        seed=seed,
    )

    from src.data.initialize import initialize_from_data

    full = _hierarchy(X, labels, block)
    params = initialize_from_data(
        full, M_k=sel.M_k, r_k=sel.r_k, max_lag=[max_lag, 1], d_ctx=0, seed=seed
    )
    graph = MultiScaleDBNGraph(hierarchy=full.hierarchy, label_names=list(labels))
    _, learned, _ = sem(sel.lam).fit(full, params, graph)
    return {dst: set(learned.get_parents(1, dst)) for dst in range(len(labels))}


def _fit_cpts(states: np.ndarray, tiled) -> Dict[str, float]:
    """Oracle table error: estimate each node's table given the true graph.

    This is the floor the sample size allows, not a method score.  Every method
    is scored by `predictive_cpt_error` against the same released tables.
    """
    names = tiled.node_names
    index = {n: i for i, n in enumerate(names)}
    errors = {}
    for node, cpt in tiled.cpds.items():
        parents = [v for v in cpt.variables[1:]]
        n_states = tiled.cardinalities[node]
        parent_cards = [tiled.cardinalities[p] for p in parents]
        n_rows = int(np.prod(parent_cards)) if parent_cards else 1
        counts = np.zeros((n_rows, n_states))
        # A tiled edge points from the previous epoch, so the parent
        # configuration of epoch t is read at t-1.
        cur = states[1:, index[node]]
        if parents:
            prev = np.stack([states[:-1, index[p]] for p in parents], axis=1)
            rows = np.ravel_multi_index(prev.T, parent_cards)
        else:
            rows = np.zeros(cur.shape, dtype=int)
        np.add.at(counts, (rows, cur), 1.0)
        counts += 1.0  # Laplace, so an unvisited configuration stays a distribution
        table = counts / counts.sum(axis=1, keepdims=True)
        errors[node] = cpt_error(table, tiled, node)
    return errors


def main(
    quick: bool = False,
    networks: str = "default",
    algorithms: str = "default",
    n_epochs: int = 0,
    replicates: int = 0,
    max_lag: int = 1,
    max_indegree: int = 3,
    iters: int = 6,
    lam_grid: Sequence[float] = (2.0, 1.0, 0.5, 0.25, 0.1),
    output_dir: str = "results/exp8",
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    write_provenance(output_dir)
    pool = QUICK_NETWORKS if quick else FULL_NETWORKS
    names = _resolve(networks, pool)
    usable = set(available_algorithms()) | {"n2dbn"}
    algos = [a for a in _resolve(algorithms, DEFAULT_ALGORITHMS) if a in usable]
    T = n_epochs or (600 if quick else 4000)
    n_rep = replicates or (1 if quick else 3)
    print(
        f"[Exp8] {len(names)} networks, {len(algos)} algorithms, "
        f"T={T}, {n_rep} replicates"
    )

    struct_rows: List[dict] = []
    cpt_rows: List[dict] = []

    for name in names:
        tiled = tile_to_2tbn(name)
        L = len(tiled.node_names)
        print(
            f"  {name}: {L} nodes, {len(tiled.true_edges)} true lagged edges",
            flush=True,
        )
        for rep in range(n_rep):
            states = sample_tiled(tiled, T, seed=rep)
            X = to_counts(states, tiled.cardinalities)

            for node, err in _fit_cpts(states, tiled).items():
                cpt_rows.append(
                    dict(
                        network=name,
                        replicate=rep,
                        method="oracle-graph",
                        node=node,
                        cpt_tv=err,
                    )
                )

            scores = _edge_scores(X, max_lag)
            auprc = edge_auprc(scores, tiled.true_edges)

            for algo in algos:
                t0 = time.time()
                try:
                    if algo == "n2dbn":
                        parents = _n2dbn_parents(
                            X,
                            tiled.node_names,
                            max_lag,
                            max_indegree,
                            iters,
                            lam_grid,
                            seed=rep,
                        )
                    else:
                        parents = learn_structure(
                            algo, X, max_lag=max_lag, max_indegree=max_indegree
                        )
                    row = score_structure(parents, tiled)
                    row["error"] = 0.0
                    # Each learner's own parent set gives its own tables, so
                    # structure and parameters are scored on the same graph.
                    tv = predictive_cpt_error(states, tiled, parents)
                    row["cpt_tv_mean"] = float(np.mean(list(tv.values())))
                    for node, err in tv.items():
                        cpt_rows.append(
                            dict(
                                network=name,
                                replicate=rep,
                                method=algo,
                                node=node,
                                cpt_tv=err,
                            )
                        )
                except Exception as exc:
                    print(f"    [{algo}] ERROR: {exc}", flush=True)
                    row = {"error": 1.0}
                struct_rows.append(
                    dict(
                        network=name,
                        n_nodes=L,
                        n_true_edges=len(tiled.true_edges),
                        replicate=rep,
                        method=algo,
                        wall_time_s=round(time.time() - t0, 3),
                        auprc_pairwise=auprc,
                        **row,
                    )
                )
        _save(struct_rows, cpt_rows, output_dir)

    _save(struct_rows, cpt_rows, output_dir)
    _summary(struct_rows, cpt_rows)
    print(f"\nResults saved to {output_dir}/")


def _save(struct_rows, cpt_rows, output_dir: str) -> None:
    for rows, stem in ((struct_rows, "exp8_structure"), (cpt_rows, "exp8_cpt")):
        if not rows:
            continue
        keys = sorted({k for r in rows for k in r})
        with open(os.path.join(output_dir, f"{stem}.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    with open(os.path.join(output_dir, "exp8_results.json"), "w") as f:
        json.dump({"structure": struct_rows, "cpt": cpt_rows}, f, indent=2)


def _summary(struct_rows, cpt_rows) -> None:
    import collections

    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in struct_rows:
        if r.get("error", 0.0):
            continue
        for key in ("SHD", "Precision", "Recall", "F1", "cpt_tv_mean"):
            if key in r:
                agg[r["method"]][key].append(r[key])
    print(
        f"\n{'method':<14}{'SHD':>9}{'Prec':>9}{'Rec':>9}{'F1':>9}{'CPT TV':>9}{'n':>5}"
    )
    print("-" * 64)
    for method, d in sorted(agg.items(), key=lambda kv: np.mean(kv[1]["SHD"] or [0])):
        tv = np.mean(d["cpt_tv_mean"]) if d["cpt_tv_mean"] else float("nan")
        print(
            f"{method:<14}{np.mean(d['SHD']):>9.2f}{np.mean(d['Precision']):>9.3f}"
            f"{np.mean(d['Recall']):>9.3f}{np.mean(d['F1']):>9.3f}{tv:>9.4f}"
            f"{len(d['SHD']):>5}"
        )
    if cpt_rows:
        by_net = collections.defaultdict(list)
        for r in cpt_rows:
            if r["method"] == "oracle-graph":
                by_net[r["network"]].append(r["cpt_tv"])
        print(f"\n{'network':<14}{'oracle CPT TV':>15}{'max':>9}{'nodes':>7}")
        print("-" * 45)
        for net, vals in sorted(by_net.items()):
            print(f"{net:<14}{np.mean(vals):>15.4f}{np.max(vals):>9.4f}{len(vals):>7}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--networks", default="default")
    parser.add_argument("--algorithms", default="default")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--replicates", type=int, default=0)
    parser.add_argument("--max-lag", type=int, default=1)
    parser.add_argument("--max-indegree", type=int, default=3)
    parser.add_argument("--iters", type=int, default=6)
    parser.add_argument(
        "--lam-grid",
        default="2.0,1.0,0.5,0.25,0.1",
        help="edge penalties the n2dbn row selects among on held-out evidence",
    )
    parser.add_argument("--output-dir", default="results/exp8")
    args = parser.parse_args()
    main(
        quick=args.quick,
        networks=args.networks,
        algorithms=args.algorithms,
        n_epochs=args.epochs,
        replicates=args.replicates,
        max_lag=args.max_lag,
        max_indegree=args.max_indegree,
        iters=args.iters,
        lam_grid=tuple(float(v) for v in args.lam_grid.split(",")),
        output_dir=args.output_dir,
    )
