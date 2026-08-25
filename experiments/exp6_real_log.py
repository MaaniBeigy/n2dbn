"""
experiments/exp6_real_log.py
============================
Experiment 6: fit the model to an observed event log.

Every other experiment starts from a generated scenario whose true graph, true
regimes and true parameters are known.  A log states none of those in its rows,
so most of what this script reports is what the rows support on their own:

  * held-out log evidence, which is the one number that always means something;
  * exact reconciliation, which the reader guarantees by construction and which
    is therefore an assertion;
  * stability selection, refitting over a rolling origin of windows across the
    training split and reporting how often each edge is selected, since an edge
    that survives every window is evidence and one that flickers is not.

Structure against a declared manifest
-------------------------------------
A log may ship with a manifest declaring how it was built.  `--manifest` reads
one and scores the fitted graph against the two claims such a manifest makes.

A declared carry-over rule names a source label, a lag in whole days and a
target label.  A lag in days is a lag in epochs only at a scale whose epoch
divides a day and whose search limit admits the quotient, so a rule is placed
at those scales alone and those scales alone are scored.  On a minute, hour,
day, week ladder with a search limit of two epochs, a one-day rule is one epoch
at the day scale and a seven-day rule is one epoch at the week scale.  What the
placed rules support is recall: how many of them the fit came back with, per
scale, with the missing ones named.

A declared-stationary label is the target of no rule at all, so every edge the
fit puts into one is a false positive.  That count is kept in its own block,
because its denominator is the stationary labels on the axis while the recall
denominator is the rules a scale can carry.

Two things a rule states are left out of the comparison.  It names the quantity
it reads and the quantity it drives, and a lagged edge here is (source label,
lag) with the duration parent implied, so the learned side carries no quantity
to compare.  It may also be scoped to a persona, and a fit pooled over a whole
log sees every persona at once, so every rule is read as applying.

Run
---
    python -m experiments.exp6_real_log --csv data/events.csv --slot-minutes 1

    python -m experiments.exp6_real_log
        --csv data/dynamic_patterns/event_log.tsv
        --manifest data/dynamic_patterns/manifest_declared.json
        --slot-minutes 1 --branching 60,24,7
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.core.calendar import SlotCalendar
from src.core.graph import MultiScaleDBNGraph
from src.core.variables import MultiScaleData
from src.data.dynamic_patterns import (
    CarryOverRule,
    DeclaredTruth,
    read_declared_manifest,
    read_subject_event_log,
)
from src.data.event_log import ObservedEvent, build_multiscale_data, read_event_log
from src.data.initialize import initialize_from_data
from src.evaluation.metrics import StructureMetrics
from src.learning.evidence import evidence_chain, scoring_settings
from src.learning.structural_em import TopDownSEM
from src.utils.provenance import write_provenance

MINUTES_PER_DAY = 1440

# The column a multi-subject log names its subject in.  A log carrying it holds
# several people on one label axis, and this script reads them pooled.
SUBJECT_COL = "person_id"

# What a truth graph holding the placed carry-over rules alone supports at one
# scale.  Every entry is a share of the placed rules that the fit came back
# with, so an edge the manifest says nothing about leaves all three untouched.
STRUCTURE_STATS = ("Rec", "SelfRec", "CrossRec")

# Share of the training split one stability window covers.
SUBSAMPLE_FRAC = 0.6


def _read_log(path: str) -> Tuple[List[ObservedEvent], int]:
    """
    Every episode of a log, pooled over the subjects it names.

    Returns:
        The episodes and how many subjects they came from.
    """
    with open(path, encoding="utf-8") as f:
        header = f.readline()
    delimiter = "\t" if "\t" in header else ","
    if SUBJECT_COL not in header.strip().split(delimiter):
        return read_event_log(path), 1
    by_subject = read_subject_event_log(path, delimiter=delimiter)
    return [e for events in by_subject.values() for e in events], len(by_subject)


def _fit(data, labels, M_k, r_k, max_lag, max_indegree, iters, seed):
    """
    One fit from a data-driven start.

    Returns:
        The fitted parameters, the learned graph, the trace, and the estimator
        that produced them, which carries the settings its score has to use.
    """
    params = initialize_from_data(
        data,
        M_k=M_k,
        r_k=r_k,
        max_lag=max_lag,
        d_ctx=int(data.get_C(1).shape[1]),
        seed=seed,
    )
    graph = MultiScaleDBNGraph(hierarchy=data.hierarchy, label_names=labels)
    sem = TopDownSEM(
        max_outer_iter=iters,
        lam_edge=2.0,
        sstep_kwargs=dict(
            confirm_edges=True,
            n_perm=32,
            max_lag={k: max_lag[k - 1] for k in range(1, data.K + 1)},
            max_indegree={k: max_indegree[k - 1] for k in range(1, data.K + 1)},
        ),
        verbose=0,
    )
    fitted, learned, history = sem.fit(data, params, graph)
    return fitted, learned, history, sem


# ---------------------------------------------------------------------------
# The declared manifest
# ---------------------------------------------------------------------------


def _rule_name(rule: CarryOverRule) -> str:
    """The name a declared rule is reported under."""
    return f"{rule.source}|{rule.target}|{rule.lag_days}d"


def _epoch_lag(lag_days: int, calendar: SlotCalendar, k: int) -> Optional[int]:
    """Epochs of scale k that a lag of whole days spans, when it spans whole ones."""
    span = calendar.epoch_minutes(k)
    total = MINUTES_PER_DAY * int(lag_days)
    return total // span if total % span == 0 else None


def _place_rules(
    truth: DeclaredTruth,
    labels: Sequence[str],
    calendar: SlotCalendar,
    max_lag: Sequence[int],
) -> Tuple[
    Dict[int, List[Tuple[CarryOverRule, int, int, int]]],
    List[CarryOverRule],
    List[CarryOverRule],
]:
    """
    Put every declared rule at each scale that can carry its lag.

    Returns:
        The placed rules per scale as (rule, source, lag, target), the rules
        naming a label the log never realized, and the rules whose lag no scale
        of this ladder reaches.
    """
    mapped, off_axis = truth.typed_edges(labels)
    placed: Dict[int, List[Tuple[CarryOverRule, int, int, int]]] = {}
    off_grid: List[CarryOverRule] = []
    for declared in mapped:
        carried = False
        for k, limit in enumerate(max_lag, start=1):
            lag = _epoch_lag(declared.rule.lag_days, calendar, k)
            if lag is None or lag > limit:
                continue
            placed.setdefault(k, []).append(
                (declared.rule, declared.edge.src_label, lag, declared.dst_label)
            )
            carried = True
        if not carried:
            off_grid.append(declared.rule)
    return placed, off_axis, off_grid


def _finite(value: float) -> Optional[float]:
    """A metric a scale has no true edge to define comes back as null."""
    return None if math.isnan(value) else float(value)


def _structure_report(
    graph: MultiScaleDBNGraph,
    truth: DeclaredTruth,
    labels: Sequence[str],
    calendar: SlotCalendar,
    max_lag: Sequence[int],
) -> Dict[str, object]:
    """Score a fitted graph against the two claims a declared manifest makes."""
    placed, off_axis, off_grid = _place_rules(truth, labels, calendar, max_lag)
    truth_graph = MultiScaleDBNGraph(
        hierarchy=graph.hierarchy, label_names=list(labels)
    )
    for k, items in placed.items():
        for _rule, src, lag, dst in items:
            truth_graph.add_edge(k, src, lag, dst)
    # This truth graph holds the placed rules and nothing else, and a manifest
    # declares a label non-stationary without naming what drives it, so an edge
    # into such a label is an edge the manifest makes no claim about.  Precision
    # and the structural Hamming distance would count every one of them as a
    # false positive, so only the recall side of the comparison is read.
    metrics = StructureMetrics.compute(graph, truth_graph).to_dict()

    index = {name: i for i, name in enumerate(labels)}
    stationary = [index[name] for name in truth.stationary_labels if name in index]
    scored = sorted(placed)
    recall: Dict[str, Dict[str, List[str]]] = {}
    control: Dict[str, Dict[str, object]] = {}
    for k in scored:
        found = [
            _rule_name(rule)
            for rule, src, lag, dst in placed[k]
            if (src, lag) in graph.get_parents(k, dst)
        ]
        names = [_rule_name(rule) for rule, _s, _l, _d in placed[k]]
        recall[str(k)] = {
            "placed": names,
            "found": found,
            "missed": [n for n in names if n not in found],
        }
        control[str(k)] = {
            "edges": sum(len(graph.get_parents(k, i)) for i in stationary),
            "labels_with_parents": [
                labels[i] for i in stationary if graph.get_parents(k, i)
            ],
        }
    return {
        "scored_scales": scored,
        "n_declared": len(truth.rules),
        "off_axis": [_rule_name(r) for r in off_axis],
        "off_grid": [_rule_name(r) for r in off_grid],
        "recall": recall,
        "n_stationary": len(stationary),
        "stationary": control,
        "metrics": {
            f"{stat}_k{k}": _finite(metrics[f"{stat}_k{k}"])
            for k in scored
            for stat in STRUCTURE_STATS
        },
    }


def _print_structure(report: Dict[str, object], manifest: str) -> None:
    """Print the structural section, one block per claim."""
    print(f"  structure against {manifest}")
    print(
        f"    {report['n_declared']} declared rules, "
        f"{len(report['off_axis'])} naming a label the log never realized, "
        f"{len(report['off_grid'])} whose lag no scale reaches"
    )
    if not report["scored_scales"]:
        print("    no scale of this ladder carries a declared lag")
        return
    metrics = report["metrics"]
    for k in report["scored_scales"]:
        block = report["recall"][str(k)]
        line = f"    k={k}: recall {len(block['found'])}/{len(block['placed'])}"
        for stat in ("SelfRec", "CrossRec"):
            value = metrics[f"{stat}_k{k}"]
            if value is not None:
                line += f"  {stat} {value:.3f}"
        print(line)
        if block["found"]:
            print("      found: " + ", ".join(block["found"]))
        if block["missed"]:
            print("      missed: " + ", ".join(block["missed"]))
        control = report["stationary"][str(k)]
        print(
            f"      {control['edges']} edges into "
            f"{len(control['labels_with_parents'])} of {report['n_stationary']} "
            "declared-stationary labels"
        )


# ---------------------------------------------------------------------------
# Stability selection
# ---------------------------------------------------------------------------


def _subsample(train: MultiScaleData, index: int, n_windows: int) -> MultiScaleData:
    """
    Window `index` of a rolling origin over the training split.

    Every window starts later than the one before it, so the windows carry
    different epochs and each one is still a contiguous series.  A single
    window is the opening one.  The closing window is the one coarse epoch
    shortest, since the three-way cut it comes from keeps an epoch back.
    """
    start = index * (1.0 - SUBSAMPLE_FRAC) / max(n_windows - 1, 1)
    rest = 1.0 - SUBSAMPLE_FRAC - start
    if start <= 1e-9:
        window, _, _ = train.split(
            fracs=(SUBSAMPLE_FRAC, rest / 2 + 1e-9, rest / 2 - 1e-9)
        )
        return window
    _, window, _ = train.split(fracs=(start, SUBSAMPLE_FRAC, max(rest, 1e-9)))
    return window


def _fingerprint(data: MultiScaleData) -> str:
    """A digest of the finest-scale counts, so two windows can be told apart."""
    return hashlib.sha1(data.get_X(1).tobytes()).hexdigest()


def _stability(
    counts: Dict[str, "collections.Counter[str]"], n_ok: int
) -> Optional[Dict[str, Dict[str, float]]]:
    """
    How often each edge was selected, over the windows that were fitted.

    A run whose windows all failed measured nothing, and an empty frequency map
    reads as a run that selected no edge, so that run comes back as None and
    the recorded failures carry the reason.
    """
    if n_ok == 0:
        return None
    return {k: {e: c / n_ok for e, c in sorted(v.items())} for k, v in counts.items()}


def main(
    csv: str = "data/events.csv",
    manifest: str = "",
    slot_minutes: int = 1,
    branching: str = "10,6",
    regimes: int = 2,
    factors: int = 0,
    max_lag: int = 2,
    max_indegree: int = 2,
    iters: int = 8,
    subsamples: int = 4,
    output_dir: str = "results/exp6",
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    write_provenance(output_dir)
    branch = [int(b) for b in branching.split(",") if b.strip()]
    K = len(branch) + 1

    events, n_subjects = _read_log(csv)
    # An observed log records as many starts of a label in a slot as happened,
    # and a pooled slot carries one per person, so the finest count array holds
    # them all and the reader's at-most-one-start check stays off.
    data, calendar, labels = build_multiscale_data(
        events, slot_minutes=slot_minutes, branching=branch, finest_binary=False
    )
    print(
        f"[Exp6] {len(events)} episodes, {n_subjects} subject(s), "
        f"{len(labels)} labels, K={K}, "
        f"epochs {[data.get_X(k).shape[0] for k in range(1, K + 1)]}"
    )

    recon = data.check_reconciliation()
    assert recon["RecErr_X"] == 0.0 and recon["RecErr_Y"] == 0.0, recon
    print(f"  reconciliation {recon}")

    M_k = [regimes] * K
    r_k = [factors] * K
    lag = [max_lag] * K
    indeg = [max_indegree] * K

    train, valid, _ = data.split(fracs=(0.7, 0.15, 0.15))
    params, graph, history, sem = _fit(train, labels, M_k, r_k, lag, indeg, iters, 0)

    edges = {
        k: sorted(
            (int(s), int(d), int(t))
            for t in range(len(labels))
            for (s, d) in graph.get_parents(k, t)
        )
        for k in range(1, K + 1)
    }
    print(f"  learned edges per scale: { {k: len(v) for k, v in edges.items()} }")

    structure = None
    if manifest:
        structure = _structure_report(
            graph, read_declared_manifest(manifest), labels, calendar, lag
        )
        _print_structure(structure, manifest)

    # A score describes the model it was taken under, so the E-step and the
    # cross-scale coupling come from the estimator that produced the fit.
    estep, couple_scales = scoring_settings(sem)
    val = evidence_chain(
        valid, params, graph, estep=estep, couple_scales=couple_scales
    ).total
    print(f"  held-out log evidence {val:.1f}")

    # Stability selection over a rolling origin of windows across the training
    # split, so each refit sees different epochs and every window keeps its
    # order intact.
    counts: Dict[str, collections.Counter] = {
        str(k): collections.Counter() for k in range(1, K + 1)
    }
    failures: List[Dict[str, object]] = []
    digests: List[str] = []
    covered: List[float] = []
    n_ok = 0
    for i in range(subsamples):
        try:
            sub = _subsample(train, i, subsamples)
        except Exception as exc:
            failures.append({"window": i, "reason": f"{type(exc).__name__}: {exc}"})
            print(f"  window {i} failed: {exc}", flush=True)
            continue
        digest = _fingerprint(sub)
        if digest in digests:
            # A short coarse ladder rounds two origins onto the same epoch
            # boundary, and refitting one window twice would let it agree
            # with itself.
            failures.append({"window": i, "reason": "repeats an earlier window"})
            print(f"  window {i} repeats an earlier window", flush=True)
            continue
        try:
            _, sub_graph, _, _ = _fit(
                sub, labels, M_k, r_k, lag, indeg, max(iters // 2, 2), i
            )
        except Exception as exc:
            failures.append({"window": i, "reason": f"{type(exc).__name__}: {exc}"})
            print(f"  window {i} failed: {exc}", flush=True)
            continue
        n_ok += 1
        digests.append(digest)
        covered.append(sub.get_X(1).shape[0] / train.get_X(1).shape[0])
        for k in range(1, K + 1):
            for t in range(len(labels)):
                for s, d in sub_graph.get_parents(k, t):
                    counts[str(k)][f"{s},{d},{t}"] += 1

    n_distinct = len(set(digests))
    window_frac = round(sum(covered) / len(covered), 4) if covered else None
    stability = _stability(counts, n_ok)
    if stability is None:
        print(
            f"  none of the {subsamples} windows was fitted, so no edge "
            "carries a frequency"
        )
    else:
        print(
            f"  selection frequency over {n_ok} rolling-origin windows of the "
            f"training split, {n_distinct} of them distinct"
        )
        for k in sorted(stability):
            top = sorted(stability[k].items(), key=lambda kv: -kv[1])[:5]
            if top:
                print(f"    k={k}: " + "  ".join(f"{e}={f:.2f}" for e, f in top))

    out = dict(
        csv=csv,
        manifest=manifest or None,
        slot_minutes=slot_minutes,
        branching=branch,
        labels=labels,
        n_events=len(events),
        n_subjects=n_subjects,
        epochs=[int(data.get_X(k).shape[0]) for k in range(1, K + 1)],
        reconciliation=recon,
        learned_edges={str(k): v for k, v in edges.items()},
        structure=structure,
        val_log_evidence=float(val),
        elbo_history=[float(x) for x in history.elbo],
        n_rejected_sweeps=len(getattr(history, "rejected", [])),
        stability=stability,
        stability_scheme="rolling origin over the training split",
        stability_window_frac=window_frac,
        n_windows_fitted=n_ok,
        n_distinct_windows=n_distinct,
        stability_failures=failures,
    )
    with open(os.path.join(output_dir, "exp6_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/events.csv")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--slot-minutes", type=int, default=1)
    parser.add_argument("--branching", default="10,6")
    parser.add_argument("--regimes", type=int, default=2)
    parser.add_argument("--factors", type=int, default=0)
    parser.add_argument("--max-lag", type=int, default=2)
    parser.add_argument("--max-indegree", type=int, default=2)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument(
        "--subsamples",
        type=int,
        default=4,
        help="rolling-origin windows of the training split the stability loop uses",
    )
    parser.add_argument("--output-dir", default="results/exp6")
    args = parser.parse_args()
    main(
        csv=args.csv,
        manifest=args.manifest,
        slot_minutes=args.slot_minutes,
        branching=args.branching,
        regimes=args.regimes,
        factors=args.factors,
        max_lag=args.max_lag,
        max_indegree=args.max_indegree,
        iters=args.iters,
        subsamples=args.subsamples,
        output_dir=args.output_dir,
    )
