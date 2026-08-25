"""
experiments/exp7_parameter_recovery.py
======================================
Experiment 7: do the learned conditional distributions match the ones that
generated the data.

Structure recovery says which edges came back.  It says nothing about whether
the conditional distributions attached to them are right, and a model can
recover the true graph with wrong coefficients or the right coefficients on a
wrong graph.  This script scores the parameters directly, against a generator
that knows every one of them exactly.

Three methods separate the estimator from its starting point:
  * `oracle_params` fits nothing and is the floor, zero error by construction;
  * `n2dbn` starts from the ground-truth parameters, which measures the
    estimator alone;
  * `n2dbn_data_init` starts from moments of the data, which measures the
    estimator and the initializer together and is the only setting that
    transfers to a log with no ground truth.

A sample-size ladder is the headline: parameter recovery is a consistency
claim, so the error has to fall as the number of epochs grows.  Recovery that
does not improve with sample size is a bias rather than an estimate.

Run
---
    python -m experiments.exp7_parameter_recovery [--quick]
"""

from __future__ import annotations

import argparse
import os
import sys
from copy import deepcopy
from dataclasses import replace
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data.initialize import initialize_from_data
from src.evaluation.baselines import OracleParams
from src.evaluation.metrics import ParameterMetrics
from src.learning.structural_em import TopDownSEM
from src.synthetic.families import make_generator
from src.synthetic.scenarios import ScenarioConfig, ScenarioGrid
from src.utils.provenance import write_provenance

# Epochs at the finest scale, holding the hierarchy shape fixed so the ladder
# varies sample size and nothing else.  Chosen from the coarsest scale rather
# than the finest: these give it 168, 336 and 672 epochs, where the previous
# 1440, 2880 and 5760 gave it 24, 48 and 96 and left every conclusion drawn on
# them describing one to four days of a single subject.  The middle rung is the
# benchmark's own horizon.
LADDER = (10080, 20160, 40320)
QUICK_LADDER = (720, 1440)


class DataInitSEM:
    """Top-down SEM that builds its starting parameters from the data alone."""

    def __init__(
        self, cfg: ScenarioConfig, lag_free_level: bool = True, **sem_kwargs
    ) -> None:
        self.cfg = cfg
        self.lag_free_level = lag_free_level
        self._sem_kwargs = sem_kwargs

    def fit(self, data, params_list, graph):
        # The regime chain keys its context term off the observed width, so the
        # initializer has to read it from the data rather than assume none.
        init = initialize_from_data(
            data,
            M_k=list(self.cfg.M_k),
            r_k=list(self.cfg.r_k),
            max_lag=list(self.cfg.max_lag),
            d_ctx=int(data.get_C(1).shape[1]),
            seed=self.cfg.seed,
            lag_free_level=self.lag_free_level,
        )
        sem = TopDownSEM(**self._sem_kwargs)
        return sem.fit(data, init, graph)


def _scenarios(quick: bool) -> List[ScenarioConfig]:
    """One scenario per family and sample size on the ladder."""
    grid = ScenarioGrid.sanity_check() if quick else ScenarioGrid.full_benchmark()
    seeds = {}
    for cfg in grid.scenarios:
        if cfg.family in (1, 2, 3) and cfg.family not in seeds and cfg.L == 5:
            seeds[cfg.family] = cfg
    if not seeds:  # the sanity grid uses L=3
        for cfg in grid.scenarios:
            seeds.setdefault(cfg.family, cfg)

    ladder = QUICK_LADDER if quick else LADDER
    out: List[ScenarioConfig] = []
    for family, cfg in sorted(seeds.items()):
        for n in ladder:
            out.append(replace(cfg, name=f"{cfg.name}_T{n}", H_finest=n))
    return out


def _lag_keys(params, k: int) -> int:
    """How many lagged coefficient keys a fitted scale ended up carrying."""
    p = params[k - 1]
    blocks = (
        (p.coarse_count.betaLag, p.coarse_dur.alphaLag)
        if p.is_coarsest
        else (p.fine_count.betaLag, p.fine_dur.alphaLag)
    )
    return sum(len(b) for b in blocks)


def _recover(cfg: ScenarioConfig, method, rep: int) -> Dict[str, float]:
    """Fit one replicate and score the learned parameters against the true ones."""
    seed = cfg.seed * 1000 + rep
    gen = make_generator(cfg, seed=seed)
    ds = gen.generate(seed=seed)
    train, _, _ = ds.data.split(fracs=(0.7, 0.15, 0.15))
    truth = [deepcopy(p) for p in ds.params]

    from src.core.graph import MultiScaleDBNGraph

    graph = MultiScaleDBNGraph(hierarchy=ds.hierarchy, label_names=gen.label_names)
    learned, _, history = method.fit(train, [deepcopy(p) for p in ds.params], graph)
    out = ParameterMetrics.compute(truth, learned).to_dict()
    # The objective separates a start that found a worse optimum from one that
    # found an equally good optimum at different parameters.  Without it a
    # parameter error alone cannot say which of the two happened.
    # `oracle_params` installs the generating parameters and runs one E-step, so
    # its objective is the reference level and needs no separate probe.
    elbo = getattr(history, "elbo", None) or []
    out["Objective"] = float(elbo[-1]) if elbo else float("nan")
    for k in range(1, len(learned) + 1):
        out[f"LagKeys_k{k}"] = float(_lag_keys(learned, k))
        out[f"TrueLagKeys_k{k}"] = float(_lag_keys(truth, k))
    return out


def report_only(output_dir: str) -> None:
    """Re-read a stored run and print the summary, with no refitting."""
    import json

    path = os.path.join(output_dir, "exp7_results.json")
    with open(path) as f:
        rows = json.load(f)
    print(f"[Exp7] {len(rows)} stored rows from {path}")
    _summary(rows)


def main(
    quick: bool = False,
    max_iter: int = 0,
    replicates: int = 0,
    output_dir: str = "results/exp7",
    baseline: bool = False,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    write_provenance(output_dir)
    scenarios = _scenarios(quick)
    n_rep = replicates or (1 if quick else 3)
    print(f"[Exp7] {len(scenarios)} scenarios, {n_rep} replicates, quick={quick}")

    sem_kw = dict(
        max_outer_iter=max_iter or (5 if quick else 10),
        lam_edge=2.0,
        # The parameter step would otherwise fit the level against the empty
        # graph before the search runs, leaving a true lagged edge nothing to
        # add at the coarsest scale.
        structure_first=not baseline,
        # Family 3 reaches a strictly worse optimum from a data start in 9 of 9
        # fits, which is a basin rather than an estimation failure, so more than
        # one start is tried and the best objective kept.
        n_restarts=3,
        sstep_kwargs=dict(confirm_edges=True, n_perm=32),
        verbose=0,
    )

    rows = []
    for cfg in scenarios:
        methods = {
            "oracle_params": OracleParams(),
            "n2dbn": TopDownSEM(**sem_kw),
            "n2dbn_data_init": DataInitSEM(cfg, lag_free_level=not baseline, **sem_kw),
        }
        for name, method in methods.items():
            for rep in range(n_rep):
                try:
                    metrics = _recover(cfg, method, rep)
                except Exception as exc:  # a failed fit is a result, not a crash
                    print(f"  [{cfg.name}/{name}/{rep}] ERROR: {exc}", flush=True)
                    metrics = {"error": 1.0}
                rows.append(
                    dict(
                        scenario=cfg.name,
                        family=cfg.family,
                        T_finest=cfg.H_finest,
                        method=name,
                        replicate=rep,
                        **metrics,
                    )
                )
        print(f"  {cfg.name} done ({len(rows)} rows)", flush=True)
        _save(rows, output_dir)

    _save(rows, output_dir)
    _summary(rows)
    print(f"\nResults saved to {output_dir}/")


def _save(rows: List[dict], output_dir: str) -> None:
    import csv
    import json

    if not rows:
        return
    keys = sorted({k for r in rows for k in r})
    head = [
        k
        for k in ("scenario", "family", "T_finest", "method", "replicate")
        if k in keys
    ]
    rest = [k for k in keys if k not in head]
    with open(os.path.join(output_dir, "exp7_results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=head + rest)
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(output_dir, "exp7_results.json"), "w") as f:
        json.dump(rows, f, indent=2)


# Blocks that enter ParamErr_mean, then the ones scored but held out of it.
_IN_MEAN = ("TransTV", "EmisRel", "FactSub", "AllocShare")
_OUT_OF_MEAN = ("LagErr", "LagFalse", "EmisErr")
_FITTED = ("n2dbn", "n2dbn_data_init")


def _clean(values: List[float]) -> List[float]:
    """Drop the NaN a block scores when its quantity is not identified."""
    return [
        v
        for v in values
        if v is not None and not (isinstance(v, float) and np.isnan(v))
    ]


def _trend(values: List[float]) -> str:
    """Whether a ladder falls at every rung, which is the consistency bar."""
    kept = _clean(values)
    if len(kept) < 2:
        return "n/a"
    if all(b == a for a, b in zip(kept, kept[1:])):
        return "flat"
    if all(b < a for a, b in zip(kept, kept[1:])):
        return "falls"
    return "rises" if all(b > a for a, b in zip(kept, kept[1:])) else "NOT monotone"


def _cells(values: List[float]) -> str:
    """Format one ladder row, leaving an unidentified rung blank."""
    return "".join("     nan" if np.isnan(v) else f"{v:8.4f}" for v in values)


def _collect(rows: List[dict], key: str):
    """Group one column by (method, sample size), and by (method, family, size)."""
    import collections

    by_size = collections.defaultdict(list)
    by_family = collections.defaultdict(list)
    for r in rows:
        v = r.get(key)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        by_size[(r["method"], r["T_finest"])].append(v)
        by_family[(r["method"], r["family"], r["T_finest"])].append(v)
    return by_size, by_family


def _stat(agg, key, stat) -> float:
    vals = _clean(agg.get(key, []))
    return float(stat(vals)) if vals else float("nan")


def _ladder_by_block(rows: List[dict], ladder: List[int]) -> None:
    """Parameter error per scale and per block, which locates a broken ladder."""
    print("\nPer-scale, per-block ladder (mean over families and replicates)")
    print("  n is fits per cell; a block only one family scores has n = 3, not 9")
    for method in _FITTED:
        print(f"\n  {method}")
        head = f"  {'block':<12}{'scale':>6}" + "".join(f"{T:>8}" for T in ladder)
        print(head + f"{'n':>4}   trend")
        print("  " + "-" * (len(head) + 16))
        for block in _IN_MEAN + _OUT_OF_MEAN:
            if block == _OUT_OF_MEAN[0]:
                print(f"  {'held out of ParamErr_mean below':<40}")
            for k in (1, 2, 3):
                agg, _ = _collect(rows, f"{block}_k{k}")
                vals = [_stat(agg, (method, T), np.mean) for T in ladder]
                if all(np.isnan(v) for v in vals):
                    continue
                n = min(len(_clean(agg.get((method, T), []))) for T in ladder)
                print(f"  {block:<12}{k:>6}{_cells(vals)}{n:>4}   {_trend(vals)}")


def _ladder_spread(rows: List[dict], ladder: List[int]) -> None:
    """Mean beside median, which says whether one bad fit carries the ladder."""
    agg, by_family = _collect(rows, "ParamErr_mean")
    print("\nParamErr_mean, mean beside median")
    print(f"  {'method':<18}{'T':>8}{'mean':>9}{'median':>9}{'max':>9}{'n':>4}")
    print("  " + "-" * 57)
    for method in _FITTED:
        for T in ladder:
            vals = _clean(agg.get((method, T), []))
            if not vals:
                continue
            print(
                f"  {method:<18}{T:>8}{np.mean(vals):>9.4f}"
                f"{np.median(vals):>9.4f}{max(vals):>9.4f}{len(vals):>4}"
            )
    for method in _FITTED:
        means = [_stat(agg, (method, T), np.mean) for T in ladder]
        medians = [_stat(agg, (method, T), np.median) for T in ladder]
        print(f"  {method}: mean {_trend(means)}, median {_trend(medians)}")

    _per_family(rows, ladder, by_family)


def _row_blocks(row: dict) -> int:
    """Count the scale-free errors one fit contributed to its own mean."""
    stored = row.get("ParamErr_n")
    if stored is not None and not (isinstance(stored, float) and np.isnan(stored)):
        return int(float(stored))
    # A run stored before the count was recorded still carries the columns it
    # was taken over, so recount rather than refusing to report.
    return sum(
        len(_clean([row.get(f"{block}_k{k}")])) for block in _IN_MEAN for k in (1, 2, 3)
    )


def _blocks_averaged(rows: List[dict], family: str) -> str:
    """How many scale-free errors this family's ParamErr_mean averages."""
    counts = {_row_blocks(r) for r in rows if r["family"] == family}
    if not counts:
        return "?"
    return str(sorted(counts)[0]) if len(counts) == 1 else "varies"


def _per_family(rows: List[dict], ladder: List[int], by_family) -> None:
    """The per-family ladder, which is the only one whose terms are comparable."""
    families = sorted({r["family"] for r in rows})
    print("\nParamErr_mean per family, the only ladder whose terms are comparable")
    print("  blocks is how many errors the mean averages; it differs by family,")
    print("  so a level may be compared down a row and never across rows")
    head = f"  {'method':<18}{'family':>7}{'blocks':>7}"
    head += "".join(f"{T:>9}" for T in ladder)
    print(head + "   trend")
    print("  " + "-" * (len(head) + 12))
    for method in _FITTED:
        for fam in families:
            vals = [_stat(by_family, (method, fam, T), np.mean) for T in ladder]
            if all(np.isnan(v) for v in vals):
                continue
            cells = "".join(f"{v:9.4f}" for v in vals)
            blocks = _blocks_averaged(rows, fam)
            print(f"  {method:<18}{fam:>7}{blocks:>7}{cells}   {_trend(vals)}")


def _basin_or_identifiability(rows: List[dict]) -> None:
    """Whether the data start reached a worse optimum or an equally good one."""
    paired = {}
    for r in rows:
        if r["method"] not in _FITTED:
            continue
        key = (r["scenario"], r["replicate"])
        paired.setdefault(key, {})[r["method"]] = r
    usable = [
        (k, v)
        for k, v in sorted(paired.items())
        if len(v) == 2
        and all(not np.isnan(_as_float(v[m].get("Objective"))) for m in _FITTED)
    ]
    if not usable:
        print("\nNo objective was recorded, so the basin question cannot be read")
        return

    print("\nBasin or identifiability, per fit")
    print("  worse objective means a local optimum; an equal or better one at a")
    print("  worse parameter error means the likelihood does not pin it down")
    head = f"  {'scenario':<22}{'rep':>4}{'obj truth':>12}{'obj data':>12}"
    head += f"{'err truth':>11}{'err data':>10}"
    print(head + "   verdict")
    print("  " + "-" * (len(head) + 12))
    tally: Dict[str, int] = {}
    for (scenario, rep), pair in usable:
        o_t = _as_float(pair["n2dbn"].get("Objective"))
        o_d = _as_float(pair["n2dbn_data_init"].get("Objective"))
        e_t = _as_float(pair["n2dbn"].get("ParamErr_mean"))
        e_d = _as_float(pair["n2dbn_data_init"].get("ParamErr_mean"))
        verdict = _verdict_for(o_t, o_d, e_t, e_d)
        tally[verdict] = tally.get(verdict, 0) + 1
        print(
            f"  {scenario:<22}{rep:>4}{o_t:>12.2f}{o_d:>12.2f}"
            f"{e_t:>11.4f}{e_d:>10.4f}   {verdict}"
        )
    print("\n  " + ", ".join(f"{v} {k}" for k, v in sorted(tally.items())))


def _as_float(value) -> float:
    """Read a stored field that may be absent, blank or already numeric."""
    if value is None or value == "":
        return float("nan")
    return float(value)


def _verdict_for(o_t: float, o_d: float, e_t: float, e_d: float) -> str:
    """Name which of the two problems one paired fit shows."""
    if np.isnan(e_t) or np.isnan(e_d):
        return "no error"
    # A relative tolerance, since the objective runs to hundreds and an exact
    # tie is not what "reached the same optimum" means.
    tol = 1e-3 * (1.0 + abs(o_t))
    if o_d < o_t - tol:
        return "local optimum" if e_d > e_t else "worse optimum, better params"
    if e_d > e_t:
        return "not identified"
    return "data start is at least as good"


def _summary(rows: List[dict]) -> None:
    """The ladder, then where in the model a broken ladder lives."""
    agg, _ = _collect(rows, "ParamErr_mean")
    ladder = sorted({r["T_finest"] for r in rows})
    print(f"\n{'method':<18}{'T_finest':>10}{'ParamErr_mean':>16}{'n':>5}")
    print("-" * 49)
    for (method, T), vals in sorted(agg.items()):
        print(f"{method:<18}{T:>10}{np.mean(vals):>16.4f}{len(vals):>5}")
    _ladder_by_block(rows, ladder)
    _ladder_spread(rows, ladder)
    _lag_recovery(rows, ladder)
    _basin_or_identifiability(rows)


def _lag_recovery(rows: List[dict], ladder: List[int]) -> None:
    """How many lagged keys each start ends up carrying, against the truth."""
    agg, _ = _collect(rows, "LagKeys_k3")
    truth, _ = _collect(rows, "TrueLagKeys_k3")
    if not agg:
        return
    print("\nCoarsest-scale lagged keys learned, against the truth")
    head = f"  {'method':<18}" + "".join(f"{T:>9}" for T in ladder)
    print(head + f"{'true':>9}")
    print("  " + "-" * (len(head) + 9))
    for method in _FITTED:
        cells = "".join(f"{_stat(agg, (method, T), np.mean):9.2f}" for T in ladder)
        ref = _stat(truth, (method, ladder[0]), np.mean)
        print(f"  {method:<18}{cells}{ref:>9.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--iters", type=int, default=0)
    parser.add_argument("--replicates", type=int, default=0)
    parser.add_argument("--output-dir", default="results/exp7")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    if args.report_only:
        report_only(args.output_dir)
    else:
        main(
            quick=args.quick,
            max_iter=args.iters,
            replicates=args.replicates,
            output_dir=args.output_dir,
            baseline=args.baseline,
        )
