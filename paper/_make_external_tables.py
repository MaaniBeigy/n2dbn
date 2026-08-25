"""Emit the external-benchmark LaTeX tables from the stored exp8 run."""

import collections
import csv
import sys

import numpy as np

STRUCT = "/workspace/results/exp8/exp8_structure.csv"
CPT = "/workspace/results/exp8/exp8_cpt.csv"

PRETTY = {
    "exhaustive": "Exhaustive, BIC",
    "hc-bic": "Hill climbing, BIC",
    "hc-bdeu": "Hill climbing, BDeu",
    "granger": "Granger, BIC",
    "dynotears": "DYNOTEARS",
    "tree": "Chow-Liu tree",
    "n2dbn": "n$^2$DBN, selected",
    "empty": "Empty graph",
}

rows = [r for r in csv.DictReader(open(STRUCT)) if not float(r.get("error") or 0)]
nets = sorted({r["network"] for r in rows})
agg = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    for key in ("SHD", "Precision", "Recall", "F1", "cpt_tv_mean"):
        if r.get(key):
            agg[r["method"]][key].append(float(r[key]))

print(f"% networks: {', '.join(nets)}; {len(rows)} scored fits")
print(r"\begin{tabular}{lrrrrr}")
print(r"\toprule")
print(r"Method & SHD & Precision & Recall & $F_1$ & CPT TV \\")
print(r"\midrule")
order = sorted(agg.items(), key=lambda kv: -float(np.mean(kv[1]["F1"])))
best = {
    "SHD": min(float(np.mean(d["SHD"])) for _, d in order),
    "Precision": max(float(np.mean(d["Precision"])) for _, d in order),
    "Recall": max(float(np.mean(d["Recall"])) for _, d in order),
    "F1": max(float(np.mean(d["F1"])) for _, d in order),
}


def cell(value, key, fmt):
    text = format(value, fmt)
    return rf"\textbf{{{text}}}" if abs(value - best[key]) < 5e-4 else text


for method, d in order:
    tv = float(np.mean(d["cpt_tv_mean"])) if d["cpt_tv_mean"] else float("nan")
    print(
        f"{PRETTY.get(method, method):<22} & "
        f"{cell(float(np.mean(d['SHD'])), 'SHD', '.2f')} & "
        f"{cell(float(np.mean(d['Precision'])), 'Precision', '.3f')} & "
        f"{cell(float(np.mean(d['Recall'])), 'Recall', '.3f')} & "
        f"{cell(float(np.mean(d['F1'])), 'F1', '.3f')} & "
        f"{tv:.4f} " + r"\\"
    )
print(r"\bottomrule")
print(r"\end{tabular}")

print()
print("% per-network n2dbn and hc-bic")
for net in nets:
    for m in ("n2dbn", "hc-bic"):
        sel = [r for r in rows if r["network"] == net and r["method"] == m]
        if not sel:
            continue
        g = lambda k: float(np.mean([float(r[k]) for r in sel]))
        print(
            f"%   {net:11s} {m:8s} P {g('Precision'):.3f} R {g('Recall'):.3f} "
            f"F1 {g('F1'):.3f} SHD {g('SHD'):.2f}"
        )

cpt = [r for r in csv.DictReader(open(CPT))]
by_net = collections.defaultdict(list)
for r in cpt:
    if r["method"] == "oracle-graph":
        by_net[r["network"]].append(float(r["cpt_tv"]))
print()
print("% oracle-graph CPT table")
for net in sorted(by_net):
    v = np.array(by_net[net])
    print(
        f"{net:<11}& {v.size} & {v.mean():.4f} & {np.median(v):.4f} & {v.max():.4f} "
        + r"\\"
    )
allv = np.concatenate([np.array(v) for v in by_net.values()])
print(
    f"All        & {allv.size} & {allv.mean():.4f} & {np.median(allv):.4f} & "
    f"{allv.max():.4f} " + r"\\"
)
print(f"% below 0.05: {100 * np.mean(allv < 0.05):.1f} pct, "
      f"below 0.10: {100 * np.mean(allv < 0.10):.1f} pct", file=sys.stderr)
