"""Emit the synthetic-grid numbers the manuscript prints, from the stored runs."""

import collections
import csv
import json
import sys

import numpy as np

ROOT = "/workspace/results"


def _rows(exp):
    with open(f"{ROOT}/{exp}/{exp}_results.json") as f:
        blob = json.load(f)
    rows = blob if isinstance(blob, list) else list(blob.values())[0]
    return [r for r in rows if not r.get("metrics", {}).get("error")]


def _mean(rows, method, key):
    vals = [r["metrics"][key] for r in rows if r["method"] == method and key in r["metrics"]]
    return float(np.nanmean(vals)) if vals else float("nan")


def _table(exp, keys, title):
    rows = _rows(exp)
    methods = sorted({r["method"] for r in rows})
    print(f"\n% ---- {title} ({exp}, {len(rows)} fits) ----")
    print("% method  " + "  ".join(keys))
    for m in methods:
        cells = "  ".join(f"{_mean(rows, m, k):.3f}" for k in keys)
        n = sum(1 for r in rows if r["method"] == m)
        print(f"%   {m:16s} {cells}   n={n}")


STRUCT = ["Prec_k2", "Rec_k2", "F1_k2", "SHD_k1", "SHD_total", "F1_mean"]
_table("exp1", STRUCT, "ideal recovery")
_table("exp2", STRUCT + ["val_loss"], "cross-scale coupling")
_table("exp3", STRUCT + ["AccS_k1", "FactCorr_k1"], "component ablation")
_table("exp4", ["F1_mean", "RecErr_X", "RecErr_Y", "val_loss"], "misspecification")

# Experiment 5 is the comparison table the manuscript prints in full.
rows = _rows("exp5")
methods = sorted({r["method"] for r in rows})
empty = _mean(rows, "empty_graph", "SHD_total")
print(f"\n% ---- comparison (exp5, {len(rows)} fits, empty SHD_total {empty:.3f}) ----")
print(r"% Method & Prec_k2 & Rec_k2 & F1_k2 & SHD_total & x empty & val_loss \\")
for m in sorted(methods, key=lambda m: -_mean(rows, m, "F1_k2")):
    shd = _mean(rows, m, "SHD_total")
    print(
        f"% {m:16s} & {_mean(rows,m,'Prec_k2'):.3f} & {_mean(rows,m,'Rec_k2'):.3f} & "
        f"{_mean(rows,m,'F1_k2'):.3f} & {shd:.3f} & {shd/empty:.2f} & "
        f"{_mean(rows,m,'val_loss'):.2f} " + r"\\"
    )

# Invariants and the guard, which the results section reports as counts.
allr = []
for exp in ("exp1", "exp2", "exp3", "exp4", "exp5"):
    try:
        allr += _rows(exp)
    except FileNotFoundError:
        pass
rec = [r["metrics"].get("RecErr_X", 0.0) for r in allr] + [
    r["metrics"].get("RecErr_Y", 0.0) for r in allr
]
traces = [r.get("elbo_history") or [] for r in allr]
multi = [h for h in traces if len(h) >= 2]
dips = sum(1 for h in multi for a, b in zip(h, h[1:]) if b < a - 1e-9)
rej = [r.get("n_rejected_sweeps", 0) for r in allr]
print(f"\n% ---- invariants over {len(allr)} fits ----")
print(f"% max reconciliation error: {max(rec) if rec else 0.0}")
print(f"% fits with >=2 sweeps: {len(multi)}; total steps: {sum(len(h)-1 for h in multi)}")
print(f"% decreasing steps: {dips}")
print(f"% fits with at least one rejected sweep: {sum(1 for v in rej if v):d} of {len(rej)}")
print(f"% mean sweeps: {np.mean([len(h) for h in traces if h]):.2f}")

# The parameter ladder.
lad = collections.defaultdict(list)
for r in json.load(open(f"{ROOT}/exp7/exp7_results.json")):
    lad[(r["method"], r["T_finest"])].append((r["ParamErr_mean"], r.get("TransTV_k2")))
print("\n% ---- parameter ladder (exp7) ----")
for m in sorted({m for m, _ in lad}):
    Ts = sorted({t for mm, t in lad if mm == m})
    pe = [float(np.nanmean([a for a, _ in lad[(m, t)]])) for t in Ts]
    tv = [float(np.nanmean([b for _, b in lad[(m, t)] if b is not None])) for t in Ts]
    mono = lambda v: all(b <= a + 1e-9 for a, b in zip(v, v[1:]))
    print(f"% {m:16s} ParamErr " + " ".join(f"{v:.4f}" for v in pe) + f"  monotone={mono(pe)}")
    print(f"% {'':16s} TransTV2 " + " ".join(f"{v:.4f}" for v in tv) + f"  monotone={mono(tv)}")
