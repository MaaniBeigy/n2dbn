"""Forecast BPIC 2017 workflow starts and active time at hourly resolution."""

import argparse
import gzip
import hashlib
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from src.core.graph import MultiScaleDBNGraph
from src.core.scales import ScaleHierarchy
from src.core.variables import MultiScaleData
from src.data.initialize import initialize_from_data
from src.learning.estep import coarse_linear_predictors
from src.learning.evidence import evidence_chain
from src.learning.structural_em import TopDownSEM
from src.model.allocation import (
    allocate_counts_block,
    allocate_counts_block_sequential,
    allocate_durations_block,
)
from src.model.emissions import _sigmoid, aggregate_lagged, nb_sample, zinb_sample
from src.model.latent_regime import RegimeChain
from src.nested.evaluate import write_json
from src.utils.provenance import write_provenance

ORIGINS = pd.date_range("2016-11-07", periods=4, freq="7D", tz="UTC")
TRAIN_DAYS = 180
HOURS = 168
DRAWS = 64
SEED = 2026


def read_segments(path):
    """Pair adjacent transitions within each case and workflow activity."""
    rows = []
    opened = cases = events = 0
    with gzip.open(path) as stream:
        for _, trace in ElementTree.iterparse(stream, events=("end",)):
            if trace.tag.rsplit("}", 1)[-1] != "trace":
                continue
            cases += 1
            previous = {}
            for event in trace.findall("{*}event"):
                events += 1
                fields = {item.get("key"): item.get("value") for item in event}
                label = fields["concept:name"].strip()
                if not label.startswith("W_"):
                    continue
                state = fields["lifecycle:transition"].lower()
                time = fields["time:timestamp"]
                worker = fields.get("org:resource")
                last = previous.get(label)
                if state in ("start", "resume"):
                    opened += 1
                if (
                    last
                    and last[0] in ("start", "resume")
                    and state in ("suspend", "complete")
                ):
                    if worker and worker == last[2]:
                        rows.append((label, last[1], time))
                previous[label] = (state, time, worker)
            trace.clear()
    df = pd.DataFrame(rows, columns=["activity", "start", "end"])
    for col in ("start", "end"):
        df[col] = pd.to_datetime(df[col], utc=True, format="ISO8601")
    df = df[df.end > df.start].copy()
    if df.empty:
        raise ValueError(
            "The input has no matched positive-duration workflow intervals."
        )
    df["seconds"] = (df.end - df.start).dt.total_seconds()
    df["hour"] = df.start.dt.floor("h")
    audit = dict(
        cases=cases,
        transitions=events,
        starts_and_resumes=opened,
        retained=len(df),
        excluded_openings=opened - len(df),
        first_start=df.start.min().isoformat(),
        last_end=df.end.max().isoformat(),
        per_activity=dict(sorted(Counter(df.activity).items())),
    )
    return df, audit


def hourly_panel(segments, index, labels):
    groups = segments.groupby(["hour", "activity"]).seconds

    def grid(values):
        return (
            values.unstack().reindex(index=index, columns=labels).fillna(0).to_numpy()
        )

    # Book the full segment to its start hour. Sum the measured times before rounding.
    return grid(groups.size()).astype("int64"), np.rint(grid(groups.sum())).astype(
        "int64"
    )


def multiscale(index, labels, counts, seconds):
    hierarchy = ScaleHierarchy.from_uniform_branching(len(index), [24])

    def levels(values):
        return [values, values.reshape(-1, 24, len(labels)).sum(axis=1)]

    def calendar(index, hourly):
        phase = 2 * np.pi * index.dayofweek.to_numpy() / 7
        cols = [np.sin(phase), np.cos(phase)]
        if hourly:
            hour = 2 * np.pi * index.hour.to_numpy() / 24
            cols.extend([np.sin(hour), np.cos(hour)])
        return np.column_stack(cols)

    return MultiScaleData(
        hierarchy,
        labels,
        levels(counts),
        levels(seconds),
        [calendar(index, True), calendar(index[::24], False)],
    )


def draw_future_days(window, params, states, factors, first_day, rng):
    """Draw daily starts and active seconds conditional on the observed prefix."""
    x, y = window.X[1], window.Y[1]
    count, duration = params.coarse_count, params.coarse_dur
    for t in range(first_day, len(x)):
        lag_x = aggregate_lagged(t, x, params.parent_sets, count.betaLag, window.L)
        lag_y = aggregate_lagged(t, y, params.parent_sets, duration.alphaLag, window.L)
        # The current count is still zero. Add its duration term after drawing it.
        log_mu, logit, log_nu = coarse_linear_predictors(
            x[t : t + 1],
            factors[t : t + 1],
            window.C[1][t : t + 1],
            params,
            lag_x[None],
            lag_y[None],
        )
        state = states[t]
        x[t] = zinb_sample(
            _sigmoid(logit[0, state]), count.r_disp, np.exp(log_mu[0, state]), rng
        )
        log_mean = log_nu[0, state] + duration.alphaX * np.log1p(x[t])
        for label in np.flatnonzero(x[t]):
            mean = max(1.0, np.exp(log_mean[label]))
            y[t, label] = nb_sample(
                duration.s_disp[label : label + 1], np.array([mean]), rng
            )[0]


def draw_future_hours(window, params, states, factors, first_day, rng):
    """Allocate future daily totals by the existing count and duration laws."""
    x, y = window.X[0], window.Y[0]
    for day in range(first_day, len(window.X[1])):
        hours = window.hierarchy.children(1, day)
        for label in range(window.L):
            parents = params.parent_sets.get(label, set())
            common = dict(
                children=hours,
                ell=label,
                Y_parent=int(window.Y[1][day, label]),
                s_seq=states,
                H_traj=factors,
                ctx=window.C[0],
                X_fine=x,
                Y_fine=y,
                parent_sets_ell=parents,
                rng=rng,
            )
            count_args = dict(
                X_parent=int(window.X[1][day, label]), p=params.fine_count
            )
            if any(lag < len(hours) for _, lag in parents):
                counts = allocate_counts_block_sequential(
                    **common, **count_args, write=x[:, label].__setitem__
                )
            else:
                counts = allocate_counts_block(c=day, **common, **count_args)
            x[hours, label] = counts
            y[hours, label] = allocate_durations_block(
                c=day, **common, X_fine_children=counts, p=params.fine_dur
            )


def nested_forecast(train, past_index, future, draws=DRAWS, sweeps=6):
    params = initialize_from_data(
        train, M_k=[2, 2], r_k=[0, 0], max_lag=[2, 7], d_ctx=[4, 2], seed=SEED
    )
    learner = TopDownSEM(
        max_outer_iter=sweeps,
        verbose=0,
        sstep_kwargs=dict(max_lag={1: 2, 2: 7}, max_indegree=1, search="exact"),
    )
    graph = MultiScaleDBNGraph(train.hierarchy, train.label_names)
    params, graph, _ = learner.fit(train, params, graph)
    posterior = evidence_chain(train, params, graph).summaries
    # The preceding week supplies all fitted hourly and daily lags.
    prefix = 7 * 24
    index = past_index[-prefix:].append(future)
    zeros = np.zeros((len(future), train.L), dtype="int64")
    window = multiscale(
        index,
        train.label_names,
        np.vstack((train.X[0][-prefix:], zeros)),
        np.vstack((train.Y[0][-prefix:], zeros)),
    )
    rng = np.random.default_rng(SEED)
    sums = np.zeros((len(future), 2 * train.L))
    for _ in range(draws):
        parent = None
        for k, start in ((2, prefix // 24), (1, prefix)):
            i = k - 1
            window.X[i][start:] = 0
            window.Y[i][start:] = 0
            gamma = posterior[i].gamma[-1]
            last = rng.choice(len(gamma), p=gamma / gamma.sum())
            ctx = window.C[i]
            chain = RegimeChain(params[i].regime)
            path = chain.sample_sequence(
                len(ctx) - start, ctx[start:], parent, rng, prev_S=[last]
            )
            states = np.pad(path, (start, 0))
            factors = np.empty((len(ctx), 0))
            if k == 2:
                draw_future_days(window, params[i], states, factors, start, rng)
                parent = np.repeat(path, 24)
            else:
                draw_future_hours(window, params[i], states, factors, prefix // 24, rng)
        sums += np.column_stack((window.X[0][prefix:], window.Y[0][prefix:]))
    return sums / draws


def direct_forecast(history, model_type, horizon, epochs=25):
    import torch

    from src.evaluation.deep_baselines import predict, train_model, windows

    mean = history.mean(axis=0)
    sd = np.maximum(history.std(axis=0), 1e-8)
    scaled = ((history - mean) / sd).astype("float32")
    context = 168
    x, y = windows(scaled, context, horizon, stride=24)
    torch.manual_seed(SEED)
    model = train_model(model_type(context, horizon), x, y, epochs=epochs, seed=SEED)
    return predict(model, scaled[-context:][None])[0] * sd + mean


def run(path, out, audit_only=False):
    out.mkdir(parents=True, exist_ok=True)
    write_provenance(str(out))
    segments, audit = read_segments(path)
    audit["input_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(out / "interval-audit.json", audit)
    if audit_only:
        return
    from src.evaluation.deep_baselines import DLinear, PatchTST

    result = dict(
        input_sha256=audit["input_sha256"],
        training_days=TRAIN_DAYS,
        horizon_hours=HOURS,
        simulations=DRAWS,
        seed=SEED,
        origins=[],
    )
    for origin in ORIGINS:
        past = pd.date_range(end=origin, periods=24 * TRAIN_DAYS + 1, freq="h")[:-1]
        future = pd.date_range(start=origin, periods=HOURS, freq="h")
        if (
            past[0] < segments.start.min()
            or future[-1] + pd.Timedelta(hours=1) > segments.end.max()
        ):
            raise ValueError(
                "The requested forecast window lies outside the event log."
            )
        known = segments[
            (segments.start >= past[0])
            & (segments.end <= origin)
            & (segments.start < origin)
        ]
        labels = sorted(known.activity.unique())
        training = hourly_panel(known, past, labels)
        actual = segments[
            (segments.start >= origin)
            & (segments.start < origin + pd.Timedelta(hours=HOURS))
        ]
        held = np.column_stack(hourly_panel(actual, future, labels))
        history = np.column_stack(training)
        train = multiscale(past, labels, *training)
        record = dict(
            origin=origin.isoformat(),
            labels=labels,
            unresolved_training_segments=int(
                (
                    (segments.start >= past[0])
                    & (segments.start < origin)
                    & (segments.end > origin)
                ).sum()
            ),
            unseen_test_starts=int((~actual.activity.isin(labels)).sum()),
            methods={},
        )
        result["origins"].append(record)
        methods = {
            "weekly-persistence": lambda: history[-HOURS:],
            "dlinear": lambda: direct_forecast(history, DLinear, HOURS),
            "patchtst": lambda: direct_forecast(history, PatchTST, HOURS),
            "n2dbn": lambda: nested_forecast(train, past, future),
        }
        arrays = {"observed": held, "activities": np.asarray(labels, dtype=str)}
        for name, forecast in methods.items():
            try:
                predictions = forecast()
                if (
                    predictions.shape != held.shape
                    or not np.isfinite(predictions).all()
                ):
                    raise ValueError("Invalid forecast shape or nonfinite forecast")
                arrays[name] = predictions
                error = np.abs(predictions - held).mean(axis=0)
                starts, seconds = np.split(error, 2)
                minutes = seconds / 60
                summary = dict(
                    start_mae=float(starts.mean()),
                    active_minutes_mae=float(minutes.mean()),
                )
                record["methods"][name] = dict(
                    **summary,
                    start_mae_by_activity=starts.tolist(),
                    active_minutes_mae_by_activity=minutes.tolist(),
                )
                print(origin.date(), name, summary, flush=True)
            except (
                ValueError,
                FloatingPointError,
                OverflowError,
                RuntimeError,
                np.linalg.LinAlgError,
            ) as exc:
                record["methods"][name] = {"error": str(exc)}
                print(origin.date(), name, str(exc), flush=True)
            write_json(out / "hourly-forecasts.json", result)
        np.savez_compressed(out / f"hourly-{origin.date()}.npz", **arrays)
    if any(
        "error" in scores
        for row in result["origins"]
        for scores in row["methods"].values()
    ):
        raise RuntimeError("Some forecasts failed. See hourly-forecasts.json.")
    result["mean_across_origins"] = {
        name: {
            metric: float(
                np.mean([row["methods"][name][metric] for row in result["origins"]])
            )
            for metric in ("start_mae", "active_minutes_mae")
        }
        for name in result["origins"][0]["methods"]
    }
    write_json(out / "hourly-forecasts.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("data/bpic2017/workflow.xes.gz")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/paper/bpic2017-workflow-forecast"),
    )
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    run(args.input, args.output_dir, args.audit_only)
