"""
experiments/exp19_quito_forecast.py
===================================
Experiment 19: forecasting QuitoBench application traffic on a nested ladder.

What is measured
----------------
QuitoBench publishes production traffic at ten minutes and at one hour, so every
scale of a ten-minute, hour, day ladder is populated by observations the file
holds.  That is the condition this arm exists to be run under, and it is what
separates this benchmark from an aggregate one.

The model is fitted per series on history before the forecast origin.  Its
parameters and its graph never read an epoch after it.  The published deep
forecasters are fitted one resolution at a time, which is the only way an
architecture reading a context window at one sampling rate can answer at three,
and the gap between their own three answers is what the coherence column reports.

Two things separate the arms.  Mean absolute error on the standardized scale is
the benchmark's own primary score.  The coherence column is what no arm of that
benchmark carries: it asks whether a method's own finest forecast sums to its own
coarsest one.

The two splits, and which one is comparable
-------------------------------------------
`cutoff` is the benchmark's declared boundary of 2023-07-28.  On the ten-minute
panel that leaves eighteen days of history, which is what the published protocol
gives a fitted model and what a comparison against published numbers has to use.
`extended` trains on more history and holds out the last blocks, which reads
epochs the published protocol assigns to its test block.  Every arm reads
whichever split is chosen, so the comparison between arms is matched either way;
only `cutoff` is comparable to numbers published elsewhere.

Scope
-----
Four of the five variates are integer at every observed row and are counts, so
their block sums are the counts of the coarser epoch.  `ind_1` is continuous and
is a rate, so summing it over an hour has no meaning and it is absent from every
arm.  The finer scales forecast without their own lagged block, which is
measured rather than assumed: that term enters as the log of a count of order
ten to the fifth inside a softmax over the children of a block, and substituting
a forecast for the realized count it conditions on costs more than it buys.

Run
---
    python -m experiments.exp19_quito_forecast --quick

    python -m experiments.exp19_quito_forecast --split cutoff --limit 40
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.core.graph import MultiScaleDBNGraph
from src.data.initialize import initialize_from_data
from src.data.quito import (
    COUNT_CHANNELS,
    CUTOFF,
    PANELS,
    Panel,
    collapse,
    four_channel_ids,
    read_panel,
    split_at_cutoff,
    whole_blocks_before,
    window_to_multiscale,
)
from src.evaluation.forecast_metrics import (
    coherence_error,
    mae,
    mse,
    relative_coherence,
    standardize,
    train_scaler,
)
from src.learning.evidence import evidence_chain
from src.learning.structural_em import TopDownSEM
from src.model.forecast import forecast_ladder
from src.utils.provenance import write_provenance

#: Regimes each scale carries.
REGIMES = 2

#: Sweeps the fit takes, and the relative tolerance that stops it.  The library
#: default stops while the objective is still moving on records this long.
OUTER_ITERS = 12
OUTER_TOL = 1e-8

#: Largest lag the structure search may propose at each scale.  `SStep` carries
#: its own budget and does not read the one the parameters were allocated with,
#: so a search confined to two lags is what a caller gets by saying nothing.
SEARCH_LAGS = {1: 24, 2: 24, 3: 7}

#: Coarsest epochs of history the fit reads where the split supplies them.
#: The declared cutoff leaves fewer than this on the ten-minute panel, so the
#: window is cut to what the split holds and the row reports which it used.
TRAIN_BLOCKS = 39

#: Coarsest epochs held out under the extended split.
TEST_BLOCKS = 2

#: The model's row name, and the anchors beside it.
MODEL = "n2dbn"
BASELINES = ("naive", "snaive-day")

#: Deep rows, added when a device is supplied.
DEEP = ("dlinear", "patchtst")

#: Context each resolution's deep model reads, finest first.
DEEP_CONTEXT = (864, 144, 14)

#: Passes the deep models take over their windows.
DEEP_EPOCHS = 25


def fit_once(counts: np.ndarray, panel: Panel, offset: int, total: int, seed: int):
    """
    Parameters and graph fitted on one window, with the search budget declared.

    Args:
        counts: `(T, L)` the training window at the panel's finest scale.
        panel: the ladder to fit.
        offset: finest epoch index the window starts at.
        total: finest epochs the whole record holds.
        seed: fixes the initialization.

    Returns:
        The fitted parameters, graph and history.
    """
    train = window_to_multiscale(counts, panel, offset=offset, total_finest=total)
    start = initialize_from_data(
        train,
        M_k=[REGIMES] * train.K,
        r_k=[0] * train.K,
        max_lag=[max(SEARCH_LAGS.values())] * train.K,
        d_ctx=[train.get_C(k).shape[1] for k in range(1, train.K + 1)],
        seed=seed,
    )
    graph = MultiScaleDBNGraph(
        hierarchy=train.hierarchy, label_names=list(COUNT_CHANNELS)
    )
    sem = TopDownSEM(
        max_outer_iter=OUTER_ITERS,
        outer_tol=OUTER_TOL,
        do_structure=True,
        verbose=0,
        sstep_kwargs={"max_lag": dict(SEARCH_LAGS)},
    )
    return sem.fit(train, start, graph)


def model_forecast(counts, panel, params, graph, origin, horizon, total, train_blocks):
    """
    The ladder's forecast of the epochs after `origin`, per scale, finest first.

    Args:
        counts: `(T, L)` the whole record at the panel's finest scale.
        panel: the ladder.
        params: parameters fitted on history before the origin.
        graph: the graph they carry.
        origin: finest epoch index the forecast starts at.
        horizon: finest epochs to forecast, a whole number of coarsest blocks.
        total: finest epochs the whole record holds.
        train_blocks: coarsest epochs of history the E-step reads.

    Returns:
        One `(H_k, L)` forecast per scale, finest first, and the window it sits
        on.
    """
    start = origin - train_blocks * panel.block
    history = window_to_multiscale(
        counts[start:origin], panel, offset=start, total_finest=total
    )
    summaries = evidence_chain(history, params, graph).summaries
    # The forecast window supplies the calendar columns and the ladder shape
    # only, so it is built at full length even where the record stops short.
    window = window_to_multiscale(
        np.zeros((horizon, counts.shape[1]), dtype=np.int64),
        panel,
        offset=origin,
        total_finest=total,
    )
    forecasts = forecast_ladder(
        params_list=params,
        origins_gamma=[np.asarray(s.gamma)[-1] for s in summaries],
        origins_factor=[
            (
                np.asarray(s.m_H)[-1]
                if s.m_H is not None and np.size(s.m_H)
                else np.zeros(0)
            )
            for s in summaries
        ],
        ctx_future=[window.get_C(k) for k in range(1, window.K + 1)],
        hierarchy=window.hierarchy,
        history_coarsest=history.get_X(history.K),
    )
    return forecasts, window


def anchor_forecasts(
    counts: np.ndarray, panel: Panel, origin: int, horizon: int
) -> Dict[str, np.ndarray]:
    """
    Each anchor's forecast at the panel's finest scale.

    Args:
        counts: `(T, L)` the whole record.
        panel: the ladder, which fixes what one day holds.
        origin: finest epoch index the forecast starts at.
        horizon: finest epochs to forecast.

    Returns:
        One `(horizon, L)` array per name in `BASELINES`.
    """
    last = counts[origin - 1][None, :].astype(float)
    day = panel.block // 7 if panel.name == "hour" else panel.block
    tail = counts[origin - day : origin].astype(float)
    return {
        "naive": np.repeat(last, horizon, axis=0),
        "snaive-day": np.array([tail[j % day] for j in range(horizon)]),
    }


def levels_of(finest: np.ndarray, window) -> List[np.ndarray]:
    """
    A finest-scale forecast and its block sums, finest first.

    Args:
        finest: `(H_1, L)` values.
        window: the multiscale window whose hierarchy fixes the blocks.

    Returns:
        One `(H_k, L)` array per scale.
    """
    out = [np.asarray(finest, dtype=float)]
    for ki in range(window.K - 1):
        up = np.zeros((window.hierarchy.H[ki + 1], out[-1].shape[1]))
        np.add.at(up, np.asarray(window.hierarchy.parent_maps[ki]), out[-1])
        out.append(up)
    return out


def deep_forecasts(counts, panel, start, origin, horizon, device, seed):
    """
    Each deep model's forecast, fitted one resolution at a time.

    Args:
        counts: `(T, L)` the whole record.
        panel: the ladder, whose scales fix the resolutions fitted.
        start: finest epoch index the training window starts at.
        origin: finest epoch index the forecast starts at.
        horizon: finest epochs to forecast.
        device: where to run the models.
        seed: fixes the initialization.

    Returns:
        One list of per-scale forecasts per name in `DEEP`, finest first.
    """
    from src.evaluation.deep_baselines import (
        DLinear,
        PatchTST,
        predict,
        train_model,
        windows,
    )

    units = [1] + [int(np.prod(panel.branching[:i])) for i in range(1, panel.K)]
    out: Dict[str, List[np.ndarray]] = {}
    for name, Model in zip(DEEP, (DLinear, PatchTST)):
        levels: List[np.ndarray] = []
        for k, unit in enumerate(units):
            history = collapse(counts[start:origin], unit).astype(np.float32)
            mean = history.mean(axis=0)
            sd = np.maximum(history.std(axis=0), 1e-8)
            scaled = (history - mean) / sd
            steps = horizon // unit
            context = min(DEEP_CONTEXT[k], max(scaled.shape[0] - steps - 1, 1))
            ctx, tgt = windows(scaled, context, steps, stride=max(1, steps // 6))
            model = train_model(
                Model(context, steps),
                ctx,
                tgt,
                epochs=DEEP_EPOCHS,
                device=device,
                seed=seed,
            )
            drawn = predict(model, scaled[-context:][None, ...], device=device)[0]
            levels.append(drawn * sd + mean)
        out[name] = levels
    return out


def one_series(
    counts,
    panel,
    item,
    origins,
    horizon,
    score_units,
    total,
    device,
    seed,
    train_blocks,
):
    """
    One row per method and origin for one series.

    Args:
        counts: `(T, L)` the whole record at the panel's finest scale.
        panel: the ladder.
        item: the series id, carried into every row.
        origins: finest epoch indices the forecasts start at.
        horizon: finest epochs forecast, a whole number of coarsest blocks.
        score_units: finest epochs per scored epoch, so every arm is scored at
            one common resolution.
        total: finest epochs the whole record holds.
        device: where the deep rows run, or `None` to leave them out.
        seed: fixes every fit.
        train_blocks: coarsest epochs of history every arm reads.

    Returns:
        One row per method and origin.
    """
    rows: List[Dict[str, object]] = []
    scored = horizon // score_units
    for origin in origins:
        start = origin - train_blocks * panel.block
        params, graph, _ = fit_once(counts[start:origin], panel, start, total, seed)
        train_scored = collapse(counts[start:origin], score_units)
        mean, sd = train_scaler(train_scored)
        actual = collapse(counts[origin : origin + horizon], score_units)
        truth = standardize(actual, mean, sd)

        predictions: Dict[str, List[np.ndarray]] = {}
        forecasts, window = model_forecast(
            counts, panel, params, graph, origin, horizon, total, train_blocks
        )
        predictions[MODEL] = forecasts
        for name, flat in anchor_forecasts(counts, panel, origin, horizon).items():
            predictions[name] = levels_of(flat, window)
        if device is not None:
            predictions.update(
                deep_forecasts(counts, panel, start, origin, horizon, device, seed)
            )

        for name, levels in predictions.items():
            at_scale = collapse(np.asarray(levels[0], dtype=float), score_units)
            at_scale = at_scale[:scored]
            rows.append(
                {
                    "item_id": item,
                    "origin": int(origin),
                    "method": name,
                    "mae": mae(truth, standardize(at_scale, mean, sd)),
                    "mse": mse(truth, standardize(at_scale, mean, sd)),
                    **coherence_error(levels, window.hierarchy),
                    "coh_rel": relative_coherence(levels, window.hierarchy),
                }
            )
    return rows


def summarize(rows: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    """
    Each method's median and mean score, and how many windows carried it.

    The median leads because one unstable series moves a mean over these counts
    by orders of magnitude, and both are reported so that is visible.

    Args:
        rows: every row produced by `one_series`.

    Returns:
        One entry per method.
    """
    if not rows:
        return {}
    metrics = [k for k in rows[0] if k not in ("item_id", "origin", "method")]
    out: Dict[str, Dict[str, float]] = {}
    for method in sorted({str(r["method"]) for r in rows}):
        picked = [r for r in rows if r["method"] == method]
        entry: Dict[str, float] = {"n_windows": float(len(picked))}
        for metric in metrics:
            values = np.array([float(r[metric]) for r in picked])
            finite = values[np.isfinite(values)]
            entry[metric] = float(np.median(finite)) if finite.size else float("nan")
            entry[f"{metric}_mean"] = (
                float(finite.mean()) if finite.size else float("nan")
            )
        out[method] = entry
    return out


def plan_origins(panel: Panel, n_epochs: int, split: str, grid: np.ndarray):
    """
    The forecast origins one split supplies.

    Args:
        panel: the ladder.
        n_epochs: finest epochs the record holds.
        split: `cutoff` for the declared boundary, `extended` for a longer
            training window that reads into the declared test block.
        grid: the shared timestamp array.

    Returns:
        The finest epoch indices the forecasts start at, and the coarsest epochs
        of history the window reads.

    Raises:
        ValueError: the split leaves too little history for any fit.
    """
    if split == "cutoff":
        boundary = split_at_cutoff(grid, CUTOFF)
    else:
        boundary = (n_epochs // panel.block - TEST_BLOCKS) * panel.block
    available = boundary // panel.block
    if available < 2:
        raise ValueError(
            f"the {split} split leaves {available} whole blocks of history, "
            "which supports no fit"
        )
    blocks = min(TRAIN_BLOCKS, available)
    whole_blocks_before(boundary, panel.block, blocks)
    return [boundary], blocks


def main(
    panel_name: str = "min",
    split: str = "extended",
    limit: int = 0,
    quick: bool = False,
    horizon_blocks: int = 2,
    score_units: int = 6,
    seed: int = 0,
    device: Optional[str] = "cuda",
    output_dir: str = "results/exp19",
    data_root: str = "data/quito",
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    write_provenance(output_dir)
    panel = PANELS[panel_name]
    grid, series = read_panel(panel, root=data_root)
    chosen = four_channel_ids(series)
    if quick:
        limit = limit or 4
    if limit:
        chosen = chosen[:limit]
    horizon = horizon_blocks * panel.block
    origins, train_blocks = plan_origins(panel, len(grid), split, grid)
    print(
        f"[Exp19] panel {panel.name} | {len(chosen)} four-channel series "
        f"| split {split} | origin(s) {origins} | history {train_blocks} blocks "
        f"| horizon {horizon} epochs | scored every {score_units}",
        flush=True,
    )

    rows: List[Dict[str, object]] = []
    t0 = time.time()
    for i, item in enumerate(chosen):
        rows.extend(
            one_series(
                series[item],
                panel,
                item,
                origins,
                horizon,
                score_units,
                len(grid),
                device,
                seed + i,
                train_blocks,
            )
        )
        if (i + 1) % 10 == 0:
            done = time.time() - t0
            print(
                f"  {i + 1}/{len(chosen)} | {done:.0f}s "
                f"| {done / (i + 1):.1f}s per series",
                flush=True,
            )

    summary = summarize(rows)
    order = list(BASELINES) + (list(DEEP) if device is not None else []) + [MODEL]
    for method in order:
        if method not in summary:
            continue
        s = summary[method]
        print(
            f"{method:11s} MAE {s['mae']:.4f} (median) {s['mae_mean']:.4f} (mean)"
            f"  coherence {s['coh']:.4g}  relative {s['coh_rel']:.3g}"
        )

    path = os.path.join(output_dir, f"exp19_{panel.name}_{split}.json")
    with open(path, "w") as handle:
        json.dump(
            {
                "panel": panel.name,
                "split": split,
                "train_blocks": train_blocks,
                "origins": [int(o) for o in origins],
                "horizon": horizon,
                "score_units": score_units,
                "n_series": len(chosen),
                "channels": list(COUNT_CHANNELS),
                "search_lags": SEARCH_LAGS,
                "rows": rows,
                "summary": summary,
            },
            handle,
            indent=2,
        )
    print(f"\nResults saved to {path}")


def _parse(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="min", choices=sorted(PANELS))
    ap.add_argument("--split", default="extended", choices=("cutoff", "extended"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--horizon-blocks", type=int, default=2)
    ap.add_argument("--score-units", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-deep", action="store_true")
    ap.add_argument("--output-dir", default="results/exp19")
    ap.add_argument("--data-root", default="data/quito")
    return ap.parse_args(argv)


if __name__ == "__main__":
    args = _parse()
    main(
        panel_name=args.panel,
        split=args.split,
        limit=args.limit,
        quick=args.quick,
        horizon_blocks=args.horizon_blocks,
        score_units=args.score_units,
        seed=args.seed,
        device=None if args.no_deep else args.device,
        output_dir=args.output_dir,
        data_root=args.data_root,
    )
