"""
experiments/exp9_selection_and_search.py
========================================
Experiment 9: what actually limits structure recovery.

Experiment 8 left the local score behind hill climbing on external ground
truth, and three explanations were on the table.  This experiment tests all
three against both benchmarks, so that the answer is a measurement rather than
a preference.

Three questions, one per output table.  A fourth is closed and is recorded
here so it is not re-opened: greedy forward-backward, exhaustive per-target
enumeration and a genetic search returned the same graph to every recorded digit
on both benchmarks and at every scale (`results/exp9/exp9_measurements.csv`,
run of 2026-08-14), so the two costly arms bought nothing and the search table is
gone.  Its three search methods against seven penalties, six block lengths and
five subsample refits on two benchmarks are what made a full run unbounded.

  * block length.  A tiled network is flat, so the second scale is imposed.
    The finest-scale term of the score is a within-block allocation, so its
    evidence per block grows with the block length.
  * penalty.  The edge penalty was inherited from the synthetic grid.  This
    sweeps it on both benchmarks and reports held-out log evidence beside F1,
    which says whether the penalty can be selected without the true graph.
  * stability.  Selection over contiguous subsamples, as an error-rate
    controlling alternative to a single fit.

Run
---
    python -m experiments.exp9_selection_and_search [--quick]
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys
from copy import deepcopy
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.benchmarks.bnlearn_tiling import (
    sample_tiled,
    score_structure,
    tile_to_2tbn,
    to_counts,
)
from src.core.graph import MultiScaleDBNGraph
from src.core.scales import ScaleHierarchy
from src.core.variables import MultiScaleData
from src.data.initialize import initialize_from_data
from src.evaluation.baselines import _clear_parent_sets, _empty_graph
from src.evaluation.metrics import StructureMetrics, r2_sortability, varsortability
from src.learning.evidence import scoring_settings
from src.learning.model_selection import held_out_nll_per_scale
from src.learning.structural_em import TopDownSEM
from src.synthetic.families import make_generator
from src.synthetic.scenarios import ScenarioGrid
from src.utils.budget import (
    COMPLETE,
    Checkpoint,
    RunBudget,
    add_budget_arguments,
    budget_from_args,
    run_cells,
)
from src.utils.provenance import write_provenance

NETWORKS = ("asia", "cancer", "earthquake", "sachs")
# Sortability needs a sample and no fit, so it is cheap enough to report on
# every released network rather than only the ones that are fitted here.
SORTABILITY_NETWORKS = (
    "asia",
    "cancer",
    "earthquake",
    "sachs",
    "child",
    "insurance",
)
# The quick grid runs one scenario.  The penalty table is an average the
# manuscript quotes, so a full run takes every scenario.
SCENARIOS = ("F2_L5_M2_zil",)
# The frontier used to stop at 2.0, where the middle scale's F1 peaked, so that
# peak sat on the boundary of the sweep and was not shown to be one.  Two
# stricter settings extend it past the boundary.
LAM_GRID = (8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.1)
BLOCK_GRID = (2, 5, 10, 20, 40, 80)
THRESHOLDS = (0.6, 0.8, 1.0)
N_SUBSAMPLES, SUBSAMPLE_FRAC = 5, 0.6


def _sem(
    lam: float,
    max_lag,
    max_indegree,
    iters: int,
    penalty: str = "bic",
):
    return TopDownSEM(
        max_outer_iter=iters,
        lam_edge=lam,
        sstep_kwargs=dict(
            search="greedy",
            confirm_edges=False,
            max_lag=max_lag,
            max_indegree=max_indegree,
            penalty=penalty,
        ),
        verbose=0,
    )


def _hierarchy(X: np.ndarray, labels: Sequence[str], block: int) -> MultiScaleData:
    """Two-scale view of a flat series, blocks of `block` consecutive epochs."""
    n = (X.shape[0] // block) * block
    Xf = X[:n].astype(np.int64)
    coarse = Xf.reshape(n // block, block, Xf.shape[1]).sum(axis=1)
    return MultiScaleData(
        hierarchy=ScaleHierarchy.from_uniform_branching(n, [block]),
        label_names=list(labels),
        X=[Xf, coarse],
        Y=[Xf.copy(), coarse.copy()],
        C=[np.zeros((n, 0)), np.zeros((n // block, 0))],
    )


def _window(data: MultiScaleData, index: int, n_windows: int) -> MultiScaleData:
    """Contiguous window `index` of `n_windows`, so a subsample stays a series."""
    lo = index * (1.0 - SUBSAMPLE_FRAC) / max(n_windows - 1, 1)
    rest = 1.0 - SUBSAMPLE_FRAC - lo
    if lo <= 1e-9:
        sub, _, _ = data.split(fracs=(SUBSAMPLE_FRAC, rest / 2 + 1e-9, rest / 2 - 1e-9))
        return sub
    _, mid, _ = data.split(fracs=(lo, SUBSAMPLE_FRAC, max(rest, 1e-9)))
    return mid


def _stability_graph(
    data, params, graph, lam, max_lag, max_indegree, K, L, iters, threshold
) -> MultiScaleDBNGraph:
    """Edges reselected on at least `threshold` of the contiguous subsamples."""
    counts, fitted = collections.Counter(), 0
    for index in range(N_SUBSAMPLES):
        try:
            _, learned, _ = _sem(lam, max_lag, max_indegree, iters).fit(
                _window(data, index, N_SUBSAMPLES),
                [deepcopy(p) for p in params],
                deepcopy(graph),
            )
        except Exception:
            continue
        fitted += 1
        for k in range(1, K + 1):
            for dst in range(L):
                for src, lag in learned.get_parents(k, dst):
                    counts[(k, src, lag, dst)] += 1
    out = MultiScaleDBNGraph(
        hierarchy=data.hierarchy, label_names=list(graph.label_names)
    )
    for (k, src, lag, dst), seen in counts.items():
        if fitted and seen / fitted >= threshold:
            out.add_edge(k, src, lag, dst)
    return out


# ---------------------------------------------------------------------------
# External half
# ---------------------------------------------------------------------------


def _external_case(name: str, n_epochs: int, block: int):
    """Train and validation views of one tiled network, plus its truth."""
    tiled = tile_to_2tbn(name)
    states = sample_tiled(tiled, (n_epochs * 3) // 2, seed=0)
    X = to_counts(states, tiled.cardinalities)
    train = _hierarchy(X[:n_epochs], tiled.node_names, block)
    valid = _hierarchy(X[n_epochs:], tiled.node_names, block)
    graph = MultiScaleDBNGraph(
        hierarchy=train.hierarchy, label_names=list(tiled.node_names)
    )
    params = initialize_from_data(
        train, M_k=[1, 1], r_k=[0, 0], max_lag=[1, 1], d_ctx=0, seed=0
    )
    return tiled, train, valid, params, graph


def external_row(
    lam: float,
    block: int,
    n_epochs: int,
    iters: int,
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    """Mean structural score and held-out evidence over the tiled networks."""
    max_lag, max_indegree = {1: 1, 2: 1}, {1: 3, 2: 3}
    scores, nlls, per_scale = [], [], []
    for name in NETWORKS:
        tiled, train, valid, params, graph = _external_case(name, n_epochs, block)
        L = len(tiled.node_names)
        if threshold is None:
            sem = _sem(lam, max_lag, max_indegree, iters)
            fitted, learned, _ = sem.fit(train, params, graph)
        else:
            learned = _stability_graph(
                train,
                params,
                graph,
                lam,
                max_lag,
                max_indegree,
                2,
                L,
                max(iters // 2, 1),
                threshold,
            )
            sem = _sem(lam, max_lag, max_indegree, 2)
            fitted, _, _ = sem.fit(train, params, deepcopy(learned))
        scores.append(
            score_structure(
                {dst: set(learned.get_parents(1, dst)) for dst in range(L)}, tiled
            )
        )
        # A held-out score belongs to the model that was fitted, so the E-step
        # and the coupling come from the estimator that produced this row.
        estep, couple_scales = scoring_settings(sem)
        per_scale.append(
            held_out_nll_per_scale(
                valid, fitted, learned, estep=estep, couple_scales=couple_scales
            )
        )
        nlls.append(float(sum(per_scale[-1])))
    row = {k: float(np.mean([s[k] for s in scores])) for k in scores[0]}
    row["val_nll"] = float(np.mean(nlls))
    # The total hides which scale wants which penalty, and the per-scale terms
    # are the only truth-free quantity that could set one price per scale.
    for k in range(1, len(per_scale[0]) + 1):
        row[f"val_nll_k{k}"] = float(np.mean([p[k - 1] for p in per_scale]))
    return row


# ---------------------------------------------------------------------------
# Synthetic half
# ---------------------------------------------------------------------------


def _synthetic_case(name: str, grid):
    """Train and validation splits of one scenario, plus its true graph."""
    cfg = grid[name]
    seed = cfg.seed * 1000
    ds = make_generator(cfg, seed=seed).generate(seed=seed)
    train, valid, _ = ds.data.split(fracs=(0.7, 0.15, 0.15))
    truth = MultiScaleDBNGraph(
        hierarchy=ds.hierarchy, label_names=list(ds.data.label_names)
    )
    for ki, p in enumerate(ds.params):
        for dst, parents in p.parent_sets.items():
            for src, lag in parents:
                truth.add_edge(ki + 1, src, lag, dst)
    return cfg, ds, train, valid, truth


def synthetic_row(
    lam: float,
    iters: int,
    names: Sequence[str],
    threshold: Optional[float] = None,
    penalty: str = "bic",
) -> Dict[str, float]:
    """Mean structural score and held-out evidence over the named scenarios."""
    grid = {c.name: c for c in ScenarioGrid.full_benchmark().scenarios}
    scores, nlls, per_scale = [], [], []
    for name in names:
        cfg, ds, train, valid, truth = _synthetic_case(name, grid)
        max_lag = {k: cfg.max_lag[k - 1] for k in range(1, cfg.K + 1)}
        max_indegree = {k: max(cfg.max_indegree[k - 1], 1) for k in range(1, cfg.K + 1)}
        # The frontier prices a graph the search has to find, so the fit starts
        # from cleared parent sets and no edges; `truth` is read only by the
        # structural score afterwards.
        params = _clear_parent_sets(ds.params)
        start = _empty_graph(truth)
        if threshold is None:
            sem = _sem(lam, max_lag, max_indegree, iters, penalty)
            fitted, learned, _ = sem.fit(train, params, deepcopy(start))
        else:
            learned = _stability_graph(
                train,
                params,
                start,
                lam,
                max_lag,
                max_indegree,
                cfg.K,
                cfg.L,
                max(iters // 2, 1),
                threshold,
            )
            sem = _sem(lam, max_lag, max_indegree, 2)
            fitted, _, _ = sem.fit(
                train, _clear_parent_sets(ds.params), deepcopy(learned)
            )
        scores.append(StructureMetrics.compute(learned, truth).to_dict())
        # A held-out score belongs to the model that was fitted, so the E-step
        # and the coupling come from the estimator that produced this row.
        estep, couple_scales = scoring_settings(sem)
        per_scale.append(
            held_out_nll_per_scale(
                valid, fitted, learned, estep=estep, couple_scales=couple_scales
            )
        )
        nlls.append(float(sum(per_scale[-1])))
    # The grid used to carry true edges at the middle scale alone, so scoring
    # the frontier on F1_k2 read the whole truth.  It now carries them at the
    # coarsest scale too, so the coarsest column and the mean are recorded as
    # well and a frontier read on one scale can be told from one read on both.
    # The per-scale distances come with them, since the question the frontier
    # has to answer is which scale an edge appears at as the price falls, and a
    # scale with no true edge answers it in false positives alone.
    row = {
        k: float(np.mean([s[k] for s in scores]))
        for k in (
            "Prec_k2",
            "Rec_k2",
            "F1_k2",
            "Prec_k3",
            "Rec_k3",
            "F1_k3",
            "F1_mean",
            "SHD_k1",
            "SHD_k2",
            "SHD_k3",
            "SHD_total",
        )
        if all(k in s for s in scores)
    }
    row["val_nll"] = float(np.mean(nlls))
    # The total hides which scale wants which penalty, and the per-scale terms
    # are the only truth-free quantity that could set one price per scale.
    for k in range(1, len(per_scale[0]) + 1):
        row[f"val_nll_k{k}"] = float(np.mean([p[k - 1] for p in per_scale]))
    return row


def sortability_rows(n_epochs: int) -> List[dict]:
    """Varsortability and its scale-free counterpart, on both benchmarks."""
    rows = []
    for name in SORTABILITY_NETWORKS:
        tiled = tile_to_2tbn(name)
        X = to_counts(
            sample_tiled(tiled, n_epochs, seed=0), tiled.cardinalities
        ).astype(float)
        rows.append(
            dict(
                benchmark="external",
                case=name,
                varsortability=varsortability(X, tiled.true_edges),
                r2_sortability=r2_sortability(X, tiled.true_edges),
            )
        )
    for cfg in ScenarioGrid.full_benchmark().scenarios:
        seed = cfg.seed * 1000
        ds = make_generator(cfg, seed=seed).generate(seed=seed)
        for ki, p in enumerate(ds.params):
            edges = {
                (src, lag, dst)
                for dst, parents in p.parent_sets.items()
                for src, lag in parents
            }
            if not edges:
                continue
            X = np.asarray(ds.data.X[ki], dtype=float)
            rows.append(
                dict(
                    benchmark="synthetic",
                    case=f"{cfg.name}_k{ki + 1}",
                    varsortability=varsortability(X, edges),
                    r2_sortability=r2_sortability(X, edges),
                )
            )
    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _cells(
    T: int, block: int, blocks, lams, thresholds, penalty_scenarios, iters: int
) -> List[Tuple[str, dict]]:
    """
    The three tables as one list of cells, each keyed by what it measures.

    A key names the table and the setting alone, so a grid that grows keeps the
    cells a finished run already holds and adds the new ones beside them.
    """
    out: List[Tuple[str, dict]] = []
    for b in blocks:
        out.append((f"block:{b}", dict(table="block", block=b, lam=0.5)))
    for lam in lams:
        out.append((f"penalty:external:{lam}", dict(table="penalty", lam=lam)))
    for lam in lams:
        out.append(
            (f"penalty:synthetic:{lam}", dict(table="penalty", lam=lam, synthetic=True))
        )
    for threshold in thresholds:
        out.append(
            (
                f"stability:{threshold}",
                dict(table="stability", lam=0.25, threshold=threshold),
            )
        )
    return out


def _run_cell(cell: dict, T: int, block: int, iters: int, penalty_scenarios) -> dict:
    """One cell of one table, as the row it contributes."""
    if cell.get("synthetic"):
        row = synthetic_row(cell["lam"], iters, penalty_scenarios)
        return dict(table=cell["table"], benchmark="synthetic", lam=cell["lam"], **row)
    row = external_row(
        cell["lam"],
        cell.get("block", block),
        T,
        iters,
        threshold=cell.get("threshold"),
    )
    keys = {k: v for k, v in cell.items() if k not in ("lam", "table")}
    return dict(
        table=cell["table"], benchmark="external", lam=cell["lam"], **keys, **row
    )


def main(
    quick: bool = False,
    n_epochs: int = 0,
    iters: int = 6,
    output_dir: str = "results/exp9",
    budget: Optional[RunBudget] = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    write_provenance(output_dir)
    T = n_epochs or (600 if quick else 4000)
    block = max(2, int(round(np.sqrt(T))))
    lams = (2.0, 0.5) if quick else LAM_GRID
    blocks = (5, 20) if quick else BLOCK_GRID
    thresholds = THRESHOLDS[:1] if quick else THRESHOLDS
    penalty_scenarios = (
        SCENARIOS
        if quick
        else tuple(c.name for c in ScenarioGrid.full_benchmark().scenarios)
    )
    cells = _cells(T, block, blocks, lams, thresholds, penalty_scenarios, iters)
    print(
        f"[Exp9] T={T}, block={block}, {len(cells)} cells over "
        f"{len(penalty_scenarios)} scenarios",
        flush=True,
    )

    def work(cell: dict) -> dict:
        row = _run_cell(cell, T, block, iters, penalty_scenarios)
        print(
            f"    {cell} F1 {row.get('F1', row.get('F1_k2', float('nan'))):.3f}",
            flush=True,
        )
        return row

    done, record = run_cells(
        cells,
        work,
        budget or RunBudget.full(),
        Checkpoint(os.path.join(output_dir, "exp9_cells.jsonl")),
    )
    rows = [done[key] for key, _ in cells if key in done]
    if record["status"] != COMPLETE:
        print(
            f"\n{record['status']}: stopped at {record['limit']} with "
            f"{len(record['cells_not_reached'])} cells not reached",
            flush=True,
        )

    sortability = sortability_rows(T)
    _save(rows, sortability, output_dir)
    with open(os.path.join(output_dir, "exp9_budget.json"), "w") as handle:
        json.dump(record, handle, indent=2)
    _summary(rows, sortability)
    print(f"\nResults saved to {output_dir}/")


def _save(rows: List[dict], sortability: List[dict], output_dir: str) -> None:
    for data, stem in ((rows, "exp9_measurements"), (sortability, "exp9_sortability")):
        if not data:
            continue
        keys = sorted({k for r in data for k in r})
        with open(os.path.join(output_dir, f"{stem}.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(data)
    with open(os.path.join(output_dir, "exp9_results.json"), "w") as f:
        json.dump({"measurements": rows, "sortability": sortability}, f, indent=2)


def _summary(rows: List[dict], sortability: List[dict]) -> None:
    ext = [r for r in rows if r["table"] == "penalty" and r["benchmark"] == "external"]
    syn = [r for r in rows if r["table"] == "penalty" and r["benchmark"] == "synthetic"]
    # The synthetic grid carries true edges at two scales, so whether held-out
    # evidence agrees with F1 is asked of each scale and of the mean.  One
    # scale answering differently from the mean is a measurement artifact
    # rather than a property of the criterion.
    for tag, sel, keys in (
        ("external", ext, ("F1",)),
        ("synthetic", syn, ("F1_k2", "F1_k3", "F1_mean")),
    ):
        if not sel:
            continue
        by_nll = min(sel, key=lambda r: r["val_nll"])["lam"]
        for key in keys:
            if not all(key in r for r in sel):
                continue
            by_f1 = max(sel, key=lambda r: r[key])["lam"]
            print(
                f"{tag:<10} held-out evidence picks lam={by_nll}, "
                f"{key} peaks at lam={by_f1}, agree: {by_nll == by_f1}"
            )

    for tag in ("external", "synthetic"):
        sel = [r for r in sortability if r["benchmark"] == tag]
        if sel:
            v = np.mean([r["varsortability"] for r in sel])
            r2 = np.mean([r["r2_sortability"] for r in sel])
            print(f"{tag:<10} varsortability {v:.3f}, R-squared sortability {r2:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_budget_arguments(parser)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--iters", type=int, default=6)
    parser.add_argument("--output-dir", default="results/exp9")
    args = parser.parse_args()
    main(
        quick=args.quick,
        n_epochs=args.epochs,
        iters=args.iters,
        output_dir=args.output_dir,
        budget=budget_from_args(args),
    )
