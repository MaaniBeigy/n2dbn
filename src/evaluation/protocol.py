"""
evaluation/protocol.py
======================
Replicated train/validation/test evaluation runner.

Usage
-----
    from src.evaluation.protocol import EvaluationProtocol

    protocol = EvaluationProtocol(methods={"n2dbn": sem_instance}, n_replicates=5)
    results = protocol.run(scenario_grid)
    protocol.save_csv(results, "results.csv")
    print(protocol.summary_table(results))

Design
------
For each (scenario, replicate):
  1. Generate synthetic dataset from the scenario config + replicate seed.
  2. Initialise every method at the same start: the generating parameters with
     their parent sets and lag coefficients cleared, and a graph with no edges.
     A method that declares `wants_oracle_start` is a stated ceiling and gets
     the generating parent sets and graph.
  3. For each method, call method.fit(train_data, ...) to (params, graph, history).
  4. Evaluate on held-out test data: compute all metrics.
  5. Collect results into a flat dict per (scenario, method, replicate).

Thread safety: each (scenario, replicate, method) triple is independent,
so the outer loop can be parallelised trivially.
"""

from __future__ import annotations

import csv
import json
import os
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..core.graph import MultiScaleDBNGraph
from ..core.variables import MultiScaleData
from ..learning.estep import EStep
from ..learning.evidence import evidence_chain, scoring_settings
from ..model.params import ScaleParams
from ..synthetic.families import make_generator
from ..synthetic.scenarios import ScenarioConfig, ScenarioGrid
from .baselines import _clear_parent_sets, _empty_graph
from .metrics import (
    LatentMetrics,
    PredictiveMetrics,
    ReconciliationMetrics,
    StructureMetrics,
    aggregate_metrics,
)

# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """One (scenario, method, replicate) result record."""

    scenario_name: str
    method_name: str
    replicate: int
    metrics: Dict[str, float]
    wall_time_s: float
    elbo_history: List[float] = field(default_factory=list)
    n_outer_iters: int = 0
    # Learned graph as {scale: [[src, lag, dst], ...]}, so a stored run can be
    # plotted or re-scored without refitting.
    learned_edges: Dict[str, List[List[int]]] = field(default_factory=dict)
    # Sweeps the outer guard rejected during the fit.
    n_rejected_sweeps: int = 0


# ---------------------------------------------------------------------------
# Search-space configuration
# ---------------------------------------------------------------------------


def apply_search_space(method: Any, cfg: ScenarioConfig) -> None:
    """
    Point a method's structure search at the scenario's candidate space.

    The S-step defaults (max_lag=2, max_indegree=3) are independent of the
    scenario, so a scenario generated with lag-3 edges had its true edges
    outside the search space entirely - recall was capped below 1 before a
    single score was computed.  Three shapes of method are handled:
    `TopDownSEM` (which owns an `sstep`), the baseline wrappers (which
    forward `**sem_kwargs`), and any method declaring `set_search_space`,
    which is how a classical structure learner receives the same budget.
    """
    K = cfg.K
    max_lag = {k: cfg.max_lag[k - 1] for k in range(1, K + 1)}
    max_indegree = {k: max(cfg.max_indegree[k - 1], 1) for k in range(1, K + 1)}

    setter = getattr(method, "set_search_space", None)
    if callable(setter):
        setter(max_lag, max_indegree)
        return

    sstep = getattr(method, "sstep", None)
    if sstep is not None:
        sstep.max_lag = max_lag
        sstep.max_indegree = max_indegree
        return
    kwargs = getattr(method, "_sem_kwargs", None)
    if isinstance(kwargs, dict):
        sk = dict(kwargs.get("sstep_kwargs") or {})
        sk.setdefault("max_lag", max_lag)
        sk.setdefault("max_indegree", max_indegree)
        kwargs["sstep_kwargs"] = sk


def _replicate_worker(protocol, cfg, rep):
    """Process-pool entry point: fit one replicate of one scenario."""
    protocol.verbose = min(protocol.verbose, 1)
    return protocol._run_replicate(cfg, rep)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class EvaluationProtocol:
    """
    Replicated evaluation runner for N²DBN and baselines.

    Parameters
    ----------
    methods      : dict name to fitted-model-object with .fit() and optionally .predict()
    n_replicates : number of independent data replicates per scenario
    train_frac   : fraction of finest-scale epochs used for training
    valid_frac   : fraction used for validation (remainder is test)
    verbose      : 0 = silent, 1 = scenario-level, 2 = replicate-level
    configure_search_space : if True (default) each method's structure search is
                   pointed at the scenario's own (max_lag, max_indegree) space
                   before fitting - see :func:`apply_search_space`
    n_jobs       : replicates fitted in parallel (processes).  1 = serial,
                   -1 = one worker per core minus one.  Replicates are fully
                   independent, so this scales close to linearly.
    """

    def __init__(
        self,
        methods: Dict[str, Any],
        n_replicates: int = 5,
        train_frac: float = 0.7,
        valid_frac: float = 0.15,
        verbose: int = 1,
        configure_search_space: bool = True,
        n_jobs: int = 1,
    ) -> None:
        self.methods = methods
        self.n_replicates = n_replicates
        self.train_frac = train_frac
        self.valid_frac = valid_frac
        self.verbose = verbose
        self.configure_search_space = configure_search_space
        self.n_jobs = n_jobs

    # ------------------------------------------------------------------
    def run(
        self,
        grid: ScenarioGrid,
        init_params_fn: Optional[Callable] = None,
        partial_csv: Optional[str] = None,
    ) -> List[RunResult]:
        """
        Run the full evaluation protocol.

        Parameters
        ----------
        grid           : ScenarioGrid to evaluate over
        init_params_fn : callable(cfg, data) -> (params_list, graph)
                         If None, uses ground-truth family params as init.
        partial_csv    : if set, save results to this CSV after each scenario
                         completes (incremental saving - survives interrupts).

        Returns
        -------
        List of RunResult objects (one per scenario x method x replicate).
        """
        tasks = [(cfg, rep) for cfg in grid for rep in range(self.n_replicates)]
        n_jobs = self._resolve_jobs()

        if n_jobs > 1 and len(tasks) > 1 and init_params_fn is None:
            return self._run_parallel(tasks, n_jobs, partial_csv)

        results: List[RunResult] = []
        n_scenarios = len(grid)
        for i, cfg in enumerate(grid, 1):
            scenario_t0 = time.time()
            if self.verbose >= 1:
                print(
                    f"\n[Protocol] [{i}/{n_scenarios}] {cfg.name}  "
                    f"({self.n_replicates} reps x {len(self.methods)} methods)",
                    flush=True,
                )
            for rep in range(self.n_replicates):
                results.extend(self._run_replicate(cfg, rep, init_params_fn))

            if self.verbose >= 1:
                print(f"  done ({time.time() - scenario_t0:.0f}s)", flush=True)
            if partial_csv is not None:
                self.save_csv(results, partial_csv)

        return results

    # ------------------------------------------------------------------
    def _resolve_jobs(self) -> int:
        """Number of worker processes implied by `n_jobs`."""
        if self.n_jobs is None or self.n_jobs == 1:
            return 1
        n_cpu = os.cpu_count() or 1
        if self.n_jobs < 0:
            return max(1, n_cpu - 1)
        return max(1, min(self.n_jobs, n_cpu))

    def _run_parallel(
        self,
        tasks: List[Tuple[ScenarioConfig, int]],
        n_jobs: int,
        partial_csv: Optional[str] = None,
    ) -> List[RunResult]:
        """
        Fit every (scenario, replicate) pair over a process pool.

        The fan-out covers the whole grid rather than one scenario at a time,
        so a grid of 22 scenarios x 3 replicates keeps 20 cores busy instead of
        3.  Results are returned in grid order regardless of completion order.
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed

        n_workers = min(n_jobs, len(tasks))
        if self.verbose >= 1:
            print(
                f"\n[Protocol] {len(tasks)} (scenario, replicate) fits "
                f"x {len(self.methods)} methods on {n_workers} workers",
                flush=True,
            )

        t0 = time.time()
        done: dict = {}
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(_replicate_worker, self, cfg, rep): idx
                for idx, (cfg, rep) in enumerate(tasks)
            }
            for n_done, fut in enumerate(as_completed(futures), 1):
                idx = futures[fut]
                done[idx] = fut.result()
                cfg, rep = tasks[idx]
                if self.verbose >= 1:
                    print(
                        f"  [{n_done}/{len(tasks)}] {cfg.name} rep {rep} "
                        f"({time.time() - t0:.0f}s elapsed)",
                        flush=True,
                    )
                if partial_csv is not None:
                    self.save_csv(
                        [r for i in sorted(done) for r in done[i]], partial_csv
                    )

        return [r for idx in range(len(tasks)) for r in done[idx]]

    # ------------------------------------------------------------------
    def _run_replicate(
        self,
        cfg: ScenarioConfig,
        rep: int,
        init_params_fn: Optional[Callable] = None,
    ) -> List[RunResult]:
        """Generate one replicate and fit every method to it."""
        results: List[RunResult] = []
        seed = cfg.seed * 1000 + rep
        if self.verbose >= 2:
            print(f"  Replicate {rep+1}/{self.n_replicates} (seed={seed})", flush=True)

        # 1. Generate data
        gen = make_generator(cfg, seed=seed)
        ds = gen.generate(seed=seed)
        test_frac = 1.0 - self.train_frac - self.valid_frac
        train_data, valid_data, test_data = ds.data.split(
            fracs=(self.train_frac, self.valid_frac, test_frac),
        )

        # Build true graph from params' parent_sets
        true_graph = MultiScaleDBNGraph(
            hierarchy=ds.hierarchy,
            label_names=gen.label_names,
        )
        for ki, p in enumerate(ds.params):
            k_idx = ki + 1
            for dst_ell, parents in p.parent_sets.items():
                for src_ell, lag in parents:
                    true_graph.add_edge(k_idx, src_ell, lag, dst_ell)

        # 2. Initialise parameters
        #
        # Every method starts from the same place: the generating parameters
        # with their parent sets and lag coefficients cleared, and a graph with
        # no edges.  The structural part of the truth reaches only the rows that
        # declare `wants_oracle_start`, which are the stated ceilings, so the
        # rows being compared differ in the criterion that chose the graph and
        # in nothing else.
        if init_params_fn is not None:
            params_list, graph = init_params_fn(cfg, train_data)
            oracle_params, oracle_graph = params_list, graph
        else:
            oracle_params = [deepcopy(ds.params[k]) for k in range(cfg.K)]
            oracle_graph = deepcopy(true_graph)
            params_list = _clear_parent_sets(oracle_params)
            graph = _empty_graph(oracle_graph)

        # 3. Run each method
        for method_name, method in self.methods.items():
            t0 = time.time()
            try:
                if self.configure_search_space:
                    apply_search_space(method, cfg)
                start_params, start_graph = (
                    (oracle_params, oracle_graph)
                    if getattr(method, "wants_oracle_start", False)
                    else (params_list, graph)
                )
                learned_params, learned_graph, history = method.fit(
                    train_data,
                    deepcopy(start_params),
                    deepcopy(start_graph),
                )
                elapsed = time.time() - t0

                # 4. Evaluate on test data.  A method fitted with the
                # cross-scale coupling switched off, or with an E-step of its
                # own, is scored the same way it was fitted, so the reported
                # number belongs to the model that was actually trained.
                estep, couple_scales = scoring_settings(method)
                metrics = self._evaluate(
                    test_data=test_data,
                    params_list=learned_params,
                    graph=learned_graph,
                    true_graph=true_graph,
                    true_latents=ds.latents,
                    estep=estep,
                    couple_scales=couple_scales,
                )
                metrics["val_loss"] = self._validation_nll(
                    valid_data,
                    learned_params,
                    learned_graph,
                    estep=estep,
                    couple_scales=couple_scales,
                )

                learned_edges = {
                    str(k): sorted(
                        [int(src), int(d), int(dst)]
                        for dst in range(cfg.L)
                        for (src, d) in learned_graph.get_parents(k, dst)
                    )
                    for k in range(1, cfg.K + 1)
                }
                rr = RunResult(
                    scenario_name=cfg.name,
                    method_name=method_name,
                    replicate=rep,
                    metrics=metrics,
                    wall_time_s=elapsed,
                    elbo_history=list(history.elbo),
                    n_outer_iters=len(history.elbo),
                    learned_edges=learned_edges,
                    n_rejected_sweeps=len(getattr(history, "rejected", [])),
                )
            except Exception as exc:
                elapsed = time.time() - t0
                if self.verbose >= 1:
                    print(f"    [{method_name}] ERROR: {exc}", flush=True)
                rr = RunResult(
                    scenario_name=cfg.name,
                    method_name=method_name,
                    replicate=rep,
                    metrics={"error": 1.0},
                    wall_time_s=elapsed,
                )

            results.append(rr)
            if self.verbose >= 2:
                print(
                    f"    [{method_name}] "
                    + "  ".join(
                        f"{k}={v:.3g}" for k, v in rr.metrics.items() if not np.isnan(v)
                    )
                    + f"  ({elapsed:.1f}s)",
                    flush=True,
                )
        return results

    # ------------------------------------------------------------------
    def _evaluate(
        self,
        test_data: MultiScaleData,
        params_list: List[ScaleParams],
        graph: MultiScaleDBNGraph,
        true_graph: MultiScaleDBNGraph,
        true_latents,
        *,
        estep: Optional[EStep] = None,
        couple_scales: bool = True,
    ) -> Dict[str, float]:
        """Run all metrics on test data."""
        K = test_data.K
        summaries = evidence_chain(
            test_data,
            params_list,
            graph,
            estep=estep,
            couple_scales=couple_scales,
        ).summaries

        # Structure metrics
        struct = StructureMetrics.compute(graph, true_graph)

        # Latent metrics - slice true latents to match test data lengths
        gamma_list = [s.gamma for s in summaries]
        m_H_list = [s.m_H for s in summaries]
        true_S = []
        true_H = []
        for k in range(1, K + 1):
            T_test_k = test_data.get_X(k).shape[0]
            true_S.append(true_latents.get_S(k)[-T_test_k:])
            true_H.append(true_latents.get_H(k)[-T_test_k:])
        latent = LatentMetrics.compute(gamma_list, true_S, m_H_list, true_H)

        # Reconciliation on test data
        recon = ReconciliationMetrics.compute(test_data)

        return aggregate_metrics(
            structure=struct,
            latent=latent,
            reconciliation=recon,
        )

    # ------------------------------------------------------------------
    def _validation_nll(
        self,
        valid_data: MultiScaleData,
        params_list: List[ScaleParams],
        graph: MultiScaleDBNGraph,
        *,
        estep: Optional[EStep] = None,
        couple_scales: bool = True,
    ) -> float:
        """Approximate validation NLL = -Σ_k log_Z_k on validation data."""
        from ..learning.model_selection import held_out_nll

        return held_out_nll(
            valid_data,
            params_list,
            graph,
            estep=estep,
            couple_scales=couple_scales,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def save_csv(results: List[RunResult], path: str) -> None:
        """Save results to a CSV file."""
        if not results:
            return
        all_metric_keys = sorted({k for r in results for k in r.metrics.keys()})
        fieldnames = [
            "scenario",
            "method",
            "replicate",
            "wall_time_s",
            "n_outer_iters",
            "n_rejected_sweeps",
        ] + all_metric_keys
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                row = {
                    "scenario": r.scenario_name,
                    "method": r.method_name,
                    "replicate": r.replicate,
                    "wall_time_s": f"{r.wall_time_s:.2f}",
                    "n_outer_iters": r.n_outer_iters,
                    "n_rejected_sweeps": r.n_rejected_sweeps,
                }
                row.update({k: f"{v:.6g}" for k, v in r.metrics.items()})
                writer.writerow(row)

    # ------------------------------------------------------------------
    @staticmethod
    def save_json(results: List[RunResult], path: str) -> None:
        """Save results to a JSON file."""
        records = [
            {
                "scenario": r.scenario_name,
                "method": r.method_name,
                "replicate": r.replicate,
                "wall_time_s": r.wall_time_s,
                "n_outer_iters": r.n_outer_iters,
                "n_rejected_sweeps": r.n_rejected_sweeps,
                "metrics": r.metrics,
                "elbo_history": r.elbo_history,
                "learned_edges": r.learned_edges,
            }
            for r in results
        ]
        with open(path, "w") as f:
            json.dump(records, f, indent=2)

    # ------------------------------------------------------------------
    @staticmethod
    def summary_table(
        results: List[RunResult],
        metric_keys: Optional[List[str]] = None,
    ) -> str:
        """
        Return a formatted summary table (mean ± std across replicates).

        Parameters
        ----------
        results     : list of RunResult objects
        metric_keys : subset of metric keys to show (None = show all)
        """
        # Aggregate by (scenario, method)
        from collections import defaultdict

        agg: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for r in results:
            key = (r.scenario_name, r.method_name)
            for mk, mv in r.metrics.items():
                if not np.isnan(mv):
                    agg[key][mk].append(mv)
            agg[key]["wall_time_s"].append(r.wall_time_s)

        if metric_keys is None:
            metric_keys = sorted({k for d in agg.values() for k in d.keys()})

        # Build header.  Cells are mean±std, so the column has to be wide
        # enough for two full numbers plus a separating space, otherwise
        # adjacent columns run together.
        col_w = 16
        header = f"{'Scenario':<24} {'Method':<14}" + "".join(
            f"{mk:>{col_w}}" for mk in metric_keys
        )
        sep = "-" * len(header)
        rows = [header, sep]

        for (scen, meth), metrics in sorted(agg.items()):
            row = f"{scen:<24} {meth:<14}"
            for mk in metric_keys:
                vals = metrics.get(mk, [])
                if vals:
                    cell = f"{np.mean(vals):.3f}±{np.std(vals):.3f}"
                else:
                    cell = "N/A"
                row += f"{cell:>{col_w}}"
            rows.append(row)

        return "\n".join(rows)
