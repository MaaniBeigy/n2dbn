"""
experiments/exp13_classical_on_benchmark.py
===========================================
Experiment 13: published structure learners on the day scale of the
dynamic_patterns benchmark, scored against its declared manifest.

Why this run exists
-------------------
The fine-scale search scores the declared sleep-to-walking rule at 0.0677 nats
of rank-one gain over the pooled cohort, against a permutation null whose 95th
percentile is 0.0367.  That rule fires 745 times over 2520 observed
subject-days, so a gain that small is either a badly computed gain or a badly
chosen search.  Structure learning over 120 variables at lag 1 and 2 on 2460
rows is routine, so running the published learners on the same relation on the
same data says which.

Two figures come back and they answer different halves of the question.  The
rule ledger scores each declared edge on its own, as the BIC gain of adding
exactly that parent to exactly that target, and ranks it among every candidate
into the same target; that is what the relation is worth to a generic criterion.
Each learner then searches the whole space and its graph is scored against the
declared truth; that is what a search does with a space in which the relation
sits.

The variables
-------------
A declared rule reads a count or a duration and drives a count or a duration,
so the design carries both.  Label `l` of 60 supplies column `l` for its
episode count that day and column `l + 60` for the total minutes of the episodes
that began that day, which is what `DeclaredTruth` reads as COUNT and DURATION.
A node index of the scored graph is that column index, so an edge of the learned
graph and an edge of the declared truth are the same object.

Every participant supplies its own rows and its own lag history, and the rows
are stacked, so one graph is fitted to 30 records and no row reads across the
boundary between two people.

The counts enter a discrete learner through the quantile encoding, which cuts
each column at its own empirical quantiles.  The benchmark placed its declared
thresholds near each source's own median for exactly that reason, and it is the
only encoding here that keeps a duration column: presence marks sleep as
occurring on every day of every record, and a clip at four states reads 500
minutes and 40 minutes as the same state.

What is scorable
----------------
Five of the six declared rules name a lag of one day, which is one epoch at the
day scale.  The gym self edge names seven days, so a search bounded at lag 2
cannot express it and its recall ceiling is five of six; `--max-lag 7` opens it.

A declared-stationary label is the target of no rule, so every edge into one is
a false positive, and it is counted in its own block against its own
denominator: the candidate edges into the count and the duration of every
stationary label the log realized.

Run
---
    python -m experiments.exp13_classical_on_benchmark
        --csv data/dynamic_patterns/event_log.tsv
        --manifest data/dynamic_patterns/manifest_declared.json

    python -m experiments.exp13_classical_on_benchmark --algorithms hc-bic,granger
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.core.graph import MultiScaleDBNGraph
from src.core.scales import ScaleHierarchy
from src.core.typed_graph import COUNT, DURATION
from src.data.dynamic_patterns import (
    DAY_SCALE,
    VERBATIM,
    CarryOverRule,
    DeclaredTruth,
    build_population,
    read_declared_manifest,
    read_subject_event_log,
)
from src.evaluation.metrics import StructureMetrics
from src.learning.classical import (
    available_algorithms,
    build_pooled_frame,
    learn_pooled_structure,
)
from src.utils.budget import (
    COMPLETE,
    Checkpoint,
    RunBudget,
    add_budget_arguments,
    budget_from_args,
    run_cells,
)
from src.utils.provenance import write_provenance

# Hill climbing leads because it is the learner this project already records a
# deficit against.  `exhaustive` sits last of the score-based entries because
# its subset count is quadratic in the source count and it costs the most here.
# `pc`, `ges` and `mmhc` search the full node set with no whitelist, and at 345
# columns and 2460 rows `pc` alone runs past twenty minutes without returning,
# so they are reachable through `--algorithms all` and are left out of a run
# whose point is a table every learner appears in.
DEFAULT_ALGORITHMS = (
    "hc-bic",
    "hc-bdeu",
    "hc-k2",
    "hc-aic",
    "tree",
    "dynotears",
    "granger",
    "exhaustive",
    "empty",
)

# The two quantities one label supplies, in the order the design lays them out.
QUANTITIES = (COUNT, DURATION)


def _resolve(spec: str, default: Sequence[str], pool: Sequence[str]) -> List[str]:
    """Names from a comma-separated list, the default set, or every usable name."""
    if spec == "default":
        return list(default)
    if spec == "all":
        return list(pool)
    wanted = [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [w for w in wanted if w not in pool]
    if unknown:
        raise SystemExit(f"unknown names {unknown}, expected some of {list(pool)}")
    return wanted


# ---------------------------------------------------------------------------
# The day-scale tables
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DayTables:
    """One block of day-scale rows per participant, on a shared label axis."""

    labels: List[str]
    people: List[str]
    blocks: List[np.ndarray]

    @property
    def n_labels(self) -> int:
        return len(self.labels)

    @property
    def n_nodes(self) -> int:
        """Design columns, which is one count and one duration per label."""
        return len(self.labels) * len(QUANTITIES)

    @property
    def n_rows(self) -> int:
        return int(sum(b.shape[0] for b in self.blocks))


def observed_prefix(flags: np.ndarray) -> int:
    """
    Leading epochs the record covers, which are the rows a lag may run over.

    Args:
        flags: one flag per epoch of the scale.

    Raises:
        ValueError: if an observed epoch sits after an unobserved one, since
            the rows would then be spliced and a lag would read across the gap.
    """
    flags = np.asarray(flags, dtype=bool)
    gaps = np.flatnonzero(~flags)
    if gaps.size == 0:
        return int(flags.size)
    first = int(gaps[0])
    if flags[first:].any():
        raise ValueError(
            f"the record is unobserved at epoch {first} and observed again "
            "later, so its rows are not consecutive and a lagged design over "
            "them would read across the gap"
        )
    return first


def day_tables(csv_path: str, manifest_path: str) -> Tuple[DayTables, DeclaredTruth]:
    """
    Every participant's observed days as one array of counts beside durations.

    Column `l` holds the episodes of label `l` that began that day and column
    `l + L` the total minutes those episodes ran, which is what the resolved
    manifest records and what `compare_day_scale` checks the arrays against.
    """
    with open(csv_path, encoding="utf-8") as f:
        header = f.readline()
    by_subject = read_subject_event_log(
        csv_path, delimiter="\t" if "\t" in header else ","
    )
    labels = sorted({e.label for events in by_subject.values() for e in events})
    truth = read_declared_manifest(manifest_path)

    people: List[str] = []
    blocks: List[np.ndarray] = []
    for person, data, _ in build_population(by_subject, labels, policy=VERBATIM):
        rows = observed_prefix(data.get_observed(DAY_SCALE))
        X = np.asarray(data.get_X(DAY_SCALE))[:rows]
        Y = np.asarray(data.get_Y(DAY_SCALE))[:rows]
        people.append(person)
        blocks.append(np.hstack([X, Y]).astype(float))
    return DayTables(labels=labels, people=people, blocks=blocks), truth


def node_names(labels: Sequence[str]) -> List[str]:
    """One name per design column, the counts first and the durations after."""
    return [f"{name}:{q}" for q in QUANTITIES for name in labels]


# ---------------------------------------------------------------------------
# The declared truth on the node axis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeclaredArrow:
    """One declared rule as an edge between two columns of the design."""

    rule: CarryOverRule
    src_node: int
    dst_node: int
    lag: int

    @property
    def name(self) -> str:
        """The rule as the manifest names it."""
        r = self.rule
        return (
            f"{r.source}.{r.source_quantity}->{r.target}.{r.target_quantity}"
            f"@lag{r.lag_days}"
        )

    @property
    def key(self) -> Tuple[int, int]:
        """The parent key a learner returns for this arrow."""
        return (self.src_node, self.lag)


def declared_arrows(truth: DeclaredTruth, labels: Sequence[str]) -> List[DeclaredArrow]:
    """
    Every declared rule placed on the node axis, in manifest order.

    A rule is read at the day scale, where its lag in days is its lag in epochs,
    and every rule is read as applying, since a pooled design holds every
    persona at once.
    """
    L = len(labels)
    mapped, off_axis = truth.typed_edges(labels, DAY_SCALE)
    if off_axis:
        raise ValueError(
            f"{len(off_axis)} declared rule(s) name a label the log never "
            f"realized: {[r.source + '->' + r.target for r in off_axis]}"
        )
    return [
        DeclaredArrow(
            rule=d.rule,
            src_node=d.edge.src_label + (L if d.edge.src_quantity == DURATION else 0),
            dst_node=d.dst_label + (L if d.dst_quantity == DURATION else 0),
            lag=d.edge.lag,
        )
        for d in mapped
    ]


def as_graph(
    parents: Dict[int, Set[Tuple[int, int]]],
    labels: Sequence[str],
    n_rows: int,
) -> MultiScaleDBNGraph:
    """A parent-set dictionary as a one-scale graph over the design columns."""
    graph = MultiScaleDBNGraph(
        hierarchy=ScaleHierarchy.from_uniform_branching(n_rows, []),
        label_names=node_names(labels),
    )
    for dst, keys in parents.items():
        for src, lag in keys:
            graph.add_edge(1, src, lag, dst)
    return graph


def truth_parents(arrows: Sequence[DeclaredArrow]) -> Dict[int, Set[Tuple[int, int]]]:
    """The declared arrows as the parent-set dictionary a learner would return."""
    out: Dict[int, Set[Tuple[int, int]]] = {}
    for arrow in arrows:
        out.setdefault(arrow.dst_node, set()).add(arrow.key)
    return out


# ---------------------------------------------------------------------------
# Scoring one learner
# ---------------------------------------------------------------------------


def stationary_block(
    parents: Dict[int, Set[Tuple[int, int]]],
    truth: DeclaredTruth,
    labels: Sequence[str],
    max_lag: int,
) -> Dict[str, object]:
    """
    Edges a learner put into declared-stationary labels, with its own denominator.

    A declared-stationary label is the target of no rule, so every edge into its
    count or its duration is a false positive.  The denominator is the candidate
    edges into those targets, which is every source column at every lag.
    """
    L = len(labels)
    index = {name: i for i, name in enumerate(labels)}
    nodes = [
        index[name] + q * L
        for name in truth.stationary_labels
        if name in index
        for q in range(len(QUANTITIES))
    ]
    edges = {node: len(parents.get(node, set())) for node in nodes}
    hit = sorted({labels[node % L] for node, n in edges.items() if n})
    return {
        "n_stationary_labels": len(nodes) // len(QUANTITIES),
        "n_stationary_targets": len(nodes),
        "stationary_candidates": len(nodes) * L * len(QUANTITIES) * max_lag,
        "stationary_edges": int(sum(edges.values())),
        "stationary_labels_with_parents": hit,
    }


def score_learner(
    parents: Dict[int, Set[Tuple[int, int]]],
    arrows: Sequence[DeclaredArrow],
    truth: DeclaredTruth,
    tables: DayTables,
    max_lag: int,
) -> Dict[str, object]:
    """Structure metrics, the recovered rules and the stationary block of one graph."""
    learned = as_graph(parents, tables.labels, tables.n_rows)
    declared = as_graph(truth_parents(arrows), tables.labels, tables.n_rows)
    metrics = StructureMetrics.compute(learned, declared).to_dict()

    found = [a.name for a in arrows if a.key in parents.get(a.dst_node, set())]
    reachable = [a for a in arrows if a.lag <= max_lag]
    hit = [a.name for a in reachable if a.name in found]
    row: Dict[str, object] = {
        "n_edges": int(sum(len(v) for v in parents.values())),
        "precision": metrics["Prec_k1"],
        "recall": metrics["Rec_k1"],
        "f1": metrics["F1_k1"],
        "shd": int(metrics["SHD_k1"]),
        "n_declared": len(arrows),
        "n_reachable": len(reachable),
        "found": found,
        "missed": [a.name for a in arrows if a.name not in found],
        "recall_reachable": len(hit) / len(reachable) if reachable else float("nan"),
    }
    row.update(stationary_block(parents, truth, tables.labels, max_lag))
    return row


# ---------------------------------------------------------------------------
# What each declared edge is worth on its own
# ---------------------------------------------------------------------------


def rule_ledger(
    tables: DayTables,
    arrows: Sequence[DeclaredArrow],
    max_lag: int,
    encoding: str,
    n_bins: int,
) -> List[Dict[str, object]]:
    """
    The BIC gain of each declared edge alone, and its rank among the candidates.

    The gain answers whether the relation is in the data at all: it is the
    improvement in the BIC of one target when exactly the declared parent is
    added and nothing else.  The rank answers whether a search would reach it:
    it is the place that gain takes among every candidate parent of the same
    target, and the count of candidates carrying a positive gain says how much
    company it has.
    """
    from pgmpy.estimators import BIC

    lf = build_pooled_frame(tables.blocks, max_lag, encoding=encoding, n_bins=n_bins)
    # A column that never changes is dropped before the learners see it, so the
    # rank a declared edge takes here is a rank among the same candidates.
    varies = {c for c in lf.frame.columns if lf.frame[c].nunique() > 1}
    lagged = {key: name for key, name in lf.lagged.items() if name in varies}
    score = BIC(lf.frame)
    rows: List[Dict[str, object]] = []
    for arrow in arrows:
        target = lf.current[arrow.dst_node]
        row: Dict[str, object] = {
            "rule": arrow.name,
            "lag": arrow.lag,
            "in_lag_budget": arrow.lag <= max_lag,
            "target_varies": target in varies,
            "source_varies": arrow.key in lagged,
        }
        if not (row["in_lag_budget"] and row["target_varies"] and row["source_varies"]):
            rows.append({**row, "gain": None, "rank": None, "n_positive": None})
            continue
        base = score.local_score(target, [])
        gains = {
            key: score.local_score(target, [name]) - base
            for key, name in lagged.items()
        }
        order = sorted(gains, key=lambda k: -gains[k])
        rows.append(
            {
                **row,
                "gain": float(gains[arrow.key]),
                "rank": order.index(arrow.key) + 1,
                "n_candidates": len(gains),
                "n_positive": int(sum(1 for g in gains.values() if g > 0)),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def main(
    csv_path: str = "data/dynamic_patterns/event_log.tsv",
    manifest_path: str = "data/dynamic_patterns/manifest_declared.json",
    algorithms: str = "default",
    max_lag: int = 2,
    max_indegree: int = 2,
    encoding: str = "quantile",
    n_bins: int = 4,
    ledger: bool = True,
    output_dir: str = "results/exp13",
    budget: Optional[RunBudget] = None,
) -> Dict[str, object]:
    os.makedirs(output_dir, exist_ok=True)
    write_provenance(output_dir)

    usable = available_algorithms()
    algos = [
        a for a in _resolve(algorithms, DEFAULT_ALGORITHMS, usable) if a in set(usable)
    ]
    tables, truth = day_tables(csv_path, manifest_path)
    arrows = declared_arrows(truth, tables.labels)
    print(
        f"[Exp13] {len(tables.people)} participants, {tables.n_labels} labels, "
        f"{tables.n_nodes} design columns, {tables.n_rows} subject-days, "
        f"lag 1 to {max_lag}, {len(arrows)} declared rules"
    )

    ledger_rows: List[Dict[str, object]] = []
    if ledger:
        t0 = time.time()
        ledger_rows = rule_ledger(tables, arrows, max_lag, encoding, n_bins)
        print(f"  rule ledger in {time.time() - t0:.1f}s")
        for row in ledger_rows:
            print(f"    {_ledger_line(row)}", flush=True)

    def fit_one(algo: str) -> Dict[str, object]:
        t0 = time.time()
        try:
            parents = learn_pooled_structure(
                algo,
                tables.blocks,
                max_lag=max_lag,
                max_indegree=max_indegree,
                encoding=encoding,
                n_bins=n_bins,
            )
            row = score_learner(parents, arrows, truth, tables, max_lag)
            row["error"] = ""
        except Exception as exc:  # one learner failing leaves the rest scorable
            row = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"  [{algo}] ERROR: {exc}", flush=True)
        row = dict(method=algo, wall_time_s=round(time.time() - t0, 2), **row)
        print(f"  {_learner_line(row)}", flush=True)
        return row

    done, record = run_cells(
        [(algo, algo) for algo in algos],
        fit_one,
        budget or RunBudget.full(),
        Checkpoint(os.path.join(output_dir, "exp13_cells.jsonl")),
    )
    rows = [done[algo] for algo in algos if algo in done]
    if record["status"] != COMPLETE:
        print(
            f"\n{record['status']}: stopped at {record['limit']} with "
            f"{len(record['cells_not_reached'])} learners not reached",
            flush=True,
        )

    _save(rows, ledger_rows, output_dir)
    with open(os.path.join(output_dir, "exp13_budget.json"), "w") as handle:
        json.dump(record, handle, indent=2)
    _summary(rows, arrows)
    print(f"\nResults saved to {output_dir}/")
    return {"structure": rows, "ledger": ledger_rows, "budget": record}


def _ledger_line(row: Dict[str, object]) -> str:
    """One declared rule as the ledger scored it."""
    if row["gain"] is None:
        why = "lag past the budget" if not row["in_lag_budget"] else "column constant"
        return f"{row['rule']:<58} unscorable, {why}"
    return (
        f"{row['rule']:<58} gain {float(row['gain']):9.2f} nats  "
        f"rank {row['rank']:>4} of {row['n_candidates']}  "
        f"{row['n_positive']} candidates positive"
    )


def _learner_line(row: Dict[str, object]) -> str:
    """One learner's result as a single line."""
    if row.get("error"):
        return f"{row['method']:<12} failed in {row['wall_time_s']:.1f}s"
    return (
        f"{row['method']:<12} {row['wall_time_s']:>8.1f}s  "
        f"edges {row['n_edges']:>5}  rec {row['recall']:.3f}  "
        f"prec {row['precision']:.4f}  SHD {row['shd']:>5}  "
        f"stationary FP {row['stationary_edges']:>5}"
        f"/{row['stationary_candidates']}"
    )


def _flat(row: Dict[str, object]) -> Dict[str, object]:
    """One row with its lists joined, which is what a CSV cell can hold."""
    return {k: "; ".join(v) if isinstance(v, list) else v for k, v in row.items()}


def _save(rows, ledger_rows, output_dir: str) -> None:
    for records, stem in ((rows, "exp13_structure"), (ledger_rows, "exp13_ledger")):
        if not records:
            continue
        keys = sorted({k for r in records for k in r})
        with open(os.path.join(output_dir, f"{stem}.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(_flat(r) for r in records)
    with open(os.path.join(output_dir, "exp13_results.json"), "w") as f:
        json.dump({"structure": rows, "ledger": ledger_rows}, f, indent=2)


def _summary(
    rows: Sequence[Dict[str, object]], arrows: Sequence[DeclaredArrow]
) -> None:
    """The table this experiment exists to print."""
    sleep_rules = [a.name for a in arrows if a.rule.source == "sleep"]
    print(
        f"\n{'method':<12}{'time s':>9}{'edges':>8}{'recall':>8}{'prec':>8}"
        f"{'SHD':>7}{'statFP':>8}{'sleep->walking':>16}"
    )
    print("-" * 76)
    for row in rows:
        if row.get("error"):
            print(f"{row['method']:<12}{row['wall_time_s']:>9.1f}   {row['error']}")
            continue
        got = any(name in row["found"] for name in sleep_rules)
        print(
            f"{row['method']:<12}{row['wall_time_s']:>9.1f}{row['n_edges']:>8}"
            f"{row['recall']:>8.3f}{row['precision']:>8.4f}{row['shd']:>7}"
            f"{row['stationary_edges']:>8}{('yes' if got else 'no'):>16}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/dynamic_patterns/event_log.tsv")
    parser.add_argument(
        "--manifest", default="data/dynamic_patterns/manifest_declared.json"
    )
    parser.add_argument("--algorithms", default="default")
    parser.add_argument("--max-lag", type=int, default=2)
    parser.add_argument("--max-indegree", type=int, default=2)
    parser.add_argument("--encoding", default="quantile")
    parser.add_argument("--n-bins", type=int, default=4)
    parser.add_argument("--no-ledger", action="store_true")
    parser.add_argument("--output-dir", default="results/exp13")
    add_budget_arguments(parser)
    args = parser.parse_args()
    main(
        csv_path=args.csv,
        manifest_path=args.manifest,
        algorithms=args.algorithms,
        max_lag=args.max_lag,
        max_indegree=args.max_indegree,
        encoding=args.encoding,
        n_bins=args.n_bins,
        ledger=not args.no_ledger,
        output_dir=args.output_dir,
        budget=budget_from_args(args),
    )
