"""Revised-paper experiments. Legacy exp* results are never read or overwritten.

python -m experiments.revision --help
Full runs are deliberately opt-in through the named Docker Compose services.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import pickle
import platform
import signal
import time
import traceback
from pathlib import Path

import numpy as np

from src.nested import FORMULATION
from src.nested.calendarbench import CALENDARBENCH_OBSERVATION, calendarbench
from src.nested.data import (
    LIFESNAPS_OBSERVATION,
    audit,
    check_round_trip,
    evaluation_view,
    from_finest,
    lifesnaps,
    read_log,
    with_sham,
    write_log,
)
from src.nested.evaluate import (
    aggregate_forecast,
    fit_summary,
    forecast_metrics,
    graph_rows,
    nonstationarity,
    reconcile,
    structural_metrics,
    summarize_forecasts,
    volume_totals,
    volumes,
    write_arrays,
    write_json,
)
from src.nested.fit import FitConfig, Fitter
from src.nested.forecast import predict
from src.nested.inference import infer
from src.nested.law import sample, target_score
from src.nested.model import Parent, Record, initialize
from src.utils.budget import peak_rss_gb
from src.utils.provenance import source_digest, write_provenance

CLASSICAL = (
    "hc-bic",
    "hc-bdeu",
    "exhaustive",
    "pc",
    "ges",
    "tree",
    "dynotears",
    "granger",
)
SORTING = ("var-sortnregress", "r2-sortnregress", "random-sortnregress")
FACTORIAL = (
    "nested+latent",
    "nested-no-latent",
    "independent+latent",
    "independent-no-latent",
)


@contextlib.contextmanager
def deadline(seconds):
    """Linux wall-time guard, including initialization and external learners."""
    if not seconds or not hasattr(signal, "setitimer"):
        yield
        return

    def expired(signum, frame):
        raise TimeoutError("declared per-cell wall-time budget exhausted")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class Run:
    """Atomic checkpoints with an immutable configuration and source digest."""

    def __init__(self, args, inputs=()):
        self.path = Path(args.output)
        self.path.mkdir(parents=True, exist_ok=True)
        config = dict(vars(args))
        config["source_digest"] = source_digest()
        config["model_formulation"] = FORMULATION
        if args.experiment == "lifesnaps":
            config["observation_definition"] = LIFESNAPS_OBSERVATION
        elif args.experiment == "benchmark":
            config["observation_definition"] = CALENDARBENCH_OBSERVATION
        config["packages"] = {}
        for package in (
            "numpy",
            "scipy",
            "pandas",
            "torch",
            "scikit-learn",
            "pgmpy",
            "pyarrow",
        ):
            try:
                config["packages"][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                config["packages"][package] = "not-installed"
        config["threads"] = {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        }
        config["inputs"] = {}
        for name in inputs:
            path = Path(name)
            digest = hashlib.sha256()
            if path.is_file():
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                config["inputs"][str(path)] = digest.hexdigest()
            else:
                config["inputs"][str(path)] = "missing"
        stored = self.path / "config.json"
        if stored.exists() and json.loads(stored.read_text()) != config:
            raise ValueError(
                "Output directory belongs to different code, options, or input data. Choose a new --output directory."
            )
        write_json(stored, config)
        write_provenance(str(self.path))
        self.rows = {}
        ledger = self.path / "runs.json"
        if ledger.exists():
            self.rows = {row["key"]: row for row in json.loads(ledger.read_text())}
        self.seconds = args.seconds

    def plan(self, keys):
        for key in keys:
            self.rows.setdefault(key, {"key": key, "status": "pending"})
        self.flush()

    def flush(self):
        write_json(self.path / "runs.json", list(self.rows.values()))

    def cell(self, key, function, **identity):
        existing = self.rows.get(key, {})
        if existing.get("status") == "complete":
            print("resume:", key, flush=True)
            return existing
        self.rows[key] = dict(key=key, status="running", **identity)
        self.flush()
        print("start:", key, flush=True)
        start = time.perf_counter()
        try:
            with deadline(self.seconds):
                result = function()
            row = dict(key=key, status="complete", **identity)
            row.update(result)
        except Exception as exc:
            row = dict(
                key=key,
                status="timeout" if isinstance(exc, TimeoutError) else "failed",
                reason=f"{type(exc).__name__}: {exc}",
                **identity,
            )
            (self.path / "errors").mkdir(exist_ok=True)
            (self.path / "errors" / (key.replace("/", "_") + ".txt")).write_text(
                traceback.format_exc()
            )
        row["wall_seconds"] = time.perf_counter() - start
        row["process_peak_rss_gb"] = peak_rss_gb()
        self.rows[key] = row
        self.flush()
        print(row["status"] + ":", key, flush=True)
        return row


def fit_config(args, **changes):
    values = dict(
        outer=args.outer,
        inner=args.inner,
        profile=args.profile,
        indegree=args.indegree,
        shifts=args.shifts,
        search=args.search,
        seed=args.seed,
        context_mode="home",
        jobs=args.jobs,
        memory_gb=args.memory_gb,
        verbose=True,
    )
    values.update(changes)
    return FitConfig(**values)


def orders(records, latent=True):
    K = records[0].data.K
    # Keep fine minute factors absent: their implemented posterior is prior-only.
    return (
        ([1] * (K - 2) + [2, 2], [0] * (K - 1) + [1])
        if latent and K >= 2
        else ([1] * K, [0] * K)
    )


def fitting(
    records,
    args,
    *,
    nested=True,
    latent=True,
    count_only=False,
    config=None,
    lag=None,
    penalty=None,
    fixed=None,
):
    K = records[0].data.K
    M, r = orders(records, latent)
    lag = lag or [args.lag] * (K - 1) + [min(args.lag, 1)]
    model = initialize(
        records,
        M,
        r,
        lag,
        count_only=count_only,
        nested=nested,
        penalty=args.penalty if penalty is None else penalty,
        seed=args.seed,
    )
    if fixed is not None:
        for k, targets in fixed.items():
            for ell, parents in targets.items():
                model.levels[k].targets[ell].parents = {
                    Parent("behavior", a, d): np.zeros(2) for a, d in parents
                }
    return Fitter(config or fit_config(args)).fit(records, model)


def save_fit(path, fit, records, context_names=None):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    temporary = path / "model.pkl.tmp"
    with temporary.open("wb") as handle:
        pickle.dump(fit.model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path / "model.pkl")
    summary = fit_summary(fit)
    summary["graph"] = graph_rows(fit.model, records[0].data.label_names, context_names)
    summary["lag_l1_count"] = [
        [
            sum(abs(v[0]) for p, v in t.parents.items() if p.kind == "behavior")
            for t in level.targets
        ]
        for level in fit.model.levels
    ]
    summary["factor_spectral_radius"] = [
        [float(max(abs(np.linalg.eigvals(f)), default=0)) for f in level.factor.F]
        for level in fit.model.levels
    ]
    write_json(path / "fit.json", summary)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(
        [r["iteration"] for r in fit.trace],
        [r["retained"] for r in fit.trace],
        label="retained",
    )
    ax.scatter(
        [r["iteration"] for r in fit.trace],
        [r["proposed"] for r in fit.trace],
        c=["tab:green" if r["accepted"] else "tab:red" for r in fit.trace],
        marker="x",
        label="proposed",
    )
    ax.set(xlabel="Outer iteration", ylabel="Auxiliary surrogate (not ELBO)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path / "objective.png", dpi=160)
    plt.close(fig)
    return summary


def selected_fit(
    records, args, *, nested=True, latent=True, count_only=False, config=None
):
    """Same prospective holdout and penalty grid for all four factorial arms.

    Validation selects within an arm only. Scores from different observation
    families are never compared as joint evidence. Validation uses smoothing and
    is a retrospective likelihood diagnostic, not a forecast accuracy estimate.
    """
    if not args.tune:
        return fitting(
            records,
            args,
            nested=nested,
            latent=latent,
            count_only=count_only,
            config=config,
        ), {"selection": "prespecified", "penalty": args.penalty}
    training, validation = [], []
    for rec in records:
        n = rec.data.hierarchy.H[-1]
        boundary = max(2, int(0.7 * n))
        if n - boundary < 2:
            raise ValueError(
                "penalty validation needs at least two complete coarsest validation epochs"
            )
        training.append(Record(rec.name, rec.data._slice_coarse(0, boundary)))
        validation.append(Record(rec.name, rec.data._slice_coarse(boundary, n)))
    for k in range(records[0].data.K):
        lag = args.lag if k < records[0].data.K - 1 else min(args.lag, 1)
        if not any(rec.rows(k, lag).any() for rec in validation):
            raise ValueError(f"no eligible validation observations at scale {k + 1}")
    choices = []
    for price in (0.5, 1.0, 2.0):
        fit = fitting(
            training,
            args,
            nested=nested,
            latent=latent,
            count_only=count_only,
            config=config,
            penalty=price,
        )
        value = 0.0
        for rec in validation:
            q = infer(rec, fit.model, args.inner)
            value += sum(
                target_score(rec, fit.model, k, ell, q)
                for k in range(fit.model.K)
                for ell in range(fit.model.L)
            )
        choices.append(
            {"penalty": price, "validation_score": value, "status": fit.status}
        )
    finite = [
        c
        for c in choices
        if c["status"] == "complete" and np.isfinite(c["validation_score"])
    ]
    if not finite:
        raise ValueError("no completed finite validation candidate")
    best = max(finite, key=lambda c: c["validation_score"])["penalty"]
    fit = fitting(
        records,
        args,
        nested=nested,
        latent=latent,
        count_only=count_only,
        config=config,
        penalty=best,
    )
    return fit, {
        "selection": "last 30% complete coarsest epochs; common three-price grid",
        "penalty": best,
        "candidates": choices,
    }


def generated(seed=0, n=1152, labels=4, participants=1, scenario=0):
    """Four declared scenarios; no legacy generator or fitted truth initialization."""
    blanks = [
        from_finest(
            f"p{i}",
            np.zeros((n, labels), dtype=int),
            np.zeros((n, labels), dtype=int),
            [6, 4],
        )
        for i in range(participants)
    ]
    model = initialize(blanks, [1, 2, 2], [0, 0, 1], [1, 1, 1], seed=seed)
    strength = (0.25, 0.55)[scenario % 2]
    latent_strength = (0.3, 1.0)[scenario // 2 % 2]
    for k, level in enumerate(model.levels):
        for ell, target in enumerate(level.targets):
            if k == 2:
                target.intercept[:] = np.log([2 + ell / labels, 12 + ell])
                target.zero_logit = -3
                target.factor[:] = 0.25
            if k:
                target.regime[:] = np.array([-latent_strength, latent_strength])
                target.parents[Parent("behavior", ell, 1)] = np.array(
                    [strength, strength / 2]
                )
                target.parents[Parent("behavior", (ell + 1) % labels, 1)] = np.array(
                    [-strength, strength / 2]
                )
    truth = [(k, ell, p.source, p.lag) for k, ell, p in model.edges()]
    return [
        sample(model, rec, seed + i * 1009)[0] for i, rec in enumerate(blanks)
    ], truth


def classical(records, method, lag, degree, seed=0):
    from src.learning.classical import learn_pooled_structure

    graphs = {}
    for k in range(records[0].data.K):
        own_lag = min(lag, 1) if k == records[0].data.K - 1 else lag
        # Split at every missing interval; never concatenate before lagging.
        blocks = []
        for rec in records:
            idx = np.flatnonzero(rec.data.get_observed(k + 1))
            for group in np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1):
                if len(group) > own_lag:
                    blocks.append(rec.data.X[k][group])
        if method in SORTING:
            if len(blocks) != 1:
                raise ValueError(
                    "sorting controls require one complete synthetic trajectory"
                )
            from src.evaluation.sortnregress import (
                r2_sort_regress,
                random_sort_regress,
                var_sort_regress,
            )

            function = {
                "var-sortnregress": var_sort_regress,
                "r2-sortnregress": r2_sort_regress,
                "random-sortnregress": random_sort_regress,
            }[method]
            found = function(
                blocks[0],
                max_lag=own_lag,
                **(
                    {"rng": np.random.default_rng(seed)}
                    if method == "random-sortnregress"
                    else {}
                ),
            )
            # Sorting references use their own sparsity rule; do not silently trim.
            graphs[k] = {
                ell: {(a, d) for a, d, b in found if b == ell}
                for ell in range(records[0].data.L)
            }
        else:
            graphs[k] = learn_pooled_structure(
                method, blocks, own_lag, degree, encoding="quantile"
            )
    return graphs


def synthetic(args):
    run = Run(args)
    methods = list(FACTORIAL) + ["empty"] + list(CLASSICAL) + list(SORTING)
    run.plan(
        [
            f"s{s}-r{r}/{m}"
            for s in range(4)
            for r in range(args.replicates)
            for m in methods
        ]
    )
    for scenario in range(4):
        for rep in range(args.replicates):
            seed = args.seed + 10000 * scenario + rep
            records, truth = generated(
                seed, args.length, args.labels, scenario=scenario
            )
            for method in methods:
                key = f"s{scenario}-r{rep}/{method}"

                def execute():
                    if method in FACTORIAL:
                        fit, validation = selected_fit(
                            records,
                            args,
                            nested=method.startswith("nested"),
                            latent=method.endswith("+latent"),
                        )
                    else:
                        graph = classical(
                            records, method, args.lag, args.indegree, seed
                        )
                        fit = fitting(
                            records,
                            args,
                            config=fit_config(args, structure=False, shifts=0),
                            fixed=graph,
                        )
                        validation = {
                            "selection": "prespecified external search; common emission-refit budget"
                        }
                    summary = save_fit(run.path / key, fit, records)
                    return dict(
                        status=fit.status,
                        metrics=structural_metrics(fit.model, truth),
                        fit=summary,
                        validation=validation,
                    )

                run.cell(key, execute, method=method, scenario=scenario, replicate=rep)


def simulations(
    path,
    records,
    model,
    replicates,
    seed,
    schedule=None,
    forms=True,
    observation_definition=None,
):
    simulations = []
    observed_totals, simulated_totals = volume_totals(records), []
    for rep in range(replicates):
        generated_records = [
            sample(model, rec, seed + 100003 * rep + i)[0]
            for i, rec in enumerate(records)
        ]
        filename = Path(path) / f"replicate-{rep:03}.csv"
        write_log(filename, generated_records, schedule)
        returned = read_log(filename, records, schedule)
        check_round_trip(generated_records, returned)
        simulated_totals.append(volume_totals(returned))
        if forms:
            simulations.append(evaluation_view(returned))
        del generated_records, returned
    result = {
        "round_trip_exact": True,
        "replicates": replicates,
        "volumes": volumes(
            records, observed_totals=observed_totals, simulated_totals=simulated_totals
        ),
    }
    if observation_definition is not None:
        result["observation_definition"] = observation_definition
    if forms:
        result["nonstationarity"] = nonstationarity(
            evaluation_view(records), simulations
        )
    write_json(Path(path) / "fidelity.json", result)
    return result


def stability_summary(rows, full_graph, pairs):
    fits = [row for row in rows if row.get("kind") == "half"]
    completed = [row for row in fits if row.get("status") == "complete"]

    def key(edge):
        return tuple(
            edge[field] for field in ("scale", "target", "kind", "source", "lag")
        )

    accepted = {key(edge) for edge in full_graph}
    universe = accepted | {key(e) for row in completed for e in row["fit"]["graph"]}
    sets = [{key(e) for e in row["fit"]["graph"]} for row in completed]
    valid = len(completed) == 2 * pairs and pairs > 0
    return {
        "planned_half_fits": 2 * pairs,
        "completed_half_fits": len(completed),
        "status": "complete" if valid else "incomplete",
        "threshold": 0.8,
        "interpretation": "empirical selection frequency; no error-bound theorem asserted",
        "edges": [
            dict(
                zip(("scale", "target", "kind", "source", "lag"), edge),
                accepted_full=edge in accepted,
                selection_frequency=(
                    sum(edge in s for s in sets) / len(sets) if sets else None
                ),
                stable=valid and sum(edge in s for s in sets) / len(sets) >= 0.8,
            )
            for edge in sorted(universe)
        ],
    }


def lifesnaps_run(args):
    run = Run(args, [args.csv])
    records, preparation, schedule = lifesnaps(
        args.csv,
        args.participants,
        args.weeks,
        annotation_path=run.path / "annotations.csv",
    )
    audit_path = run.path / "audit.json"
    write_json(audit_path, preparation)
    if not preparation["valid"]:
        collisions = {
            label: cells
            for label, cells in preparation["collision_cells"].items()
            if cells
        }
        errors = preparation["support_errors"]
        details = [
            "LifeSnaps audit failed; fitting has not started.",
            f"Full report: {audit_path}",
            "The count-duration model requires at most one start per label per "
            "minute, Y >= X, and Y = 0 when X = 0.",
            f"Observed minute cells with multiple starts, by label: {collisions}",
            f"First {min(5, len(errors))} of {len(errors)} validation errors:",
            *errors[:5],
        ]
        examples = preparation.get("collision_examples", [])[:3]
        if examples:
            details.extend(
                [
                    "First collision examples (CSV line numbers refer to the input):",
                    json.dumps(examples, indent=2),
                ]
            )
        details.append(
            "Grouping has not resolved this support error. Inspect the report "
            "and annotations.csv before fitting."
        )
        raise ValueError("\n".join(details))
    print(
        f"LifeSnaps audit passed: {preparation['participants']} participants. "
        f"Report: {audit_path}",
        flush=True,
    )
    print(
        "Observation unit: one occupied participant-label-minute; duration is "
        "the sum of its annotation marks. "
        f"Original annotations: {run.path / 'annotations.csv'}",
        flush=True,
    )
    for label, values in preparation["grouping"]["by_label"].items():
        print(
            f"  {label}: {values['annotations_before']} annotations -> "
            f"{values['groups_after']} groups; "
            f"{values['coincident_groups']} coincident groups; "
            f"duration {values['duration_after']} (unchanged)",
            flush=True,
        )
    if args.audit_only:
        return
    from src.data.anchors import anchor_set

    anchor_set(schedule.axis).freeze(str(run.path / "anchors.json"))
    arms = ["full", "calendar-all", "calendar-none", "intercept", "no-regimes", "sham"]
    half_keys = [
        f"half-{pair:03}-{side}" for pair in range(args.pairs) for side in range(2)
    ]
    run.plan(arms + half_keys)
    for arm in arms:
        own = with_sham(records, args.seed + 871) if arm == "sham" else records
        if arm in ("intercept", "calendar-none"):
            own = [
                Record(
                    r.name,
                    dataclasses.replace(r.data, C=[np.zeros_like(c) for c in r.data.C]),
                )
                for r in records
            ]

        def execute():
            context = {
                "calendar-all": "all",
                "calendar-none": "none",
                "intercept": "none",
            }.get(arm, "home")
            config = fit_config(
                args,
                context_mode=context,
                structure=arm != "intercept",
                depletion=arm != "intercept",
            )
            fit, validation = selected_fit(
                own, args, latent=arm not in ("intercept", "no-regimes"), config=config
            )
            summary = save_fit(run.path / arm, fit, own, preparation["context_names"])
            result = dict(
                status=fit.status,
                fit=summary,
                validation=validation,
                observation_definition=LIFESNAPS_OBSERVATION,
            )
            from src.nested.diagnostics import anchors, edge_checks, sham_denominator

            if arm != "sham":
                result["fidelity"] = simulations(
                    run.path / arm / "simulations",
                    own,
                    fit.model,
                    args.replicates,
                    args.seed + 9000,
                    schedule,
                    observation_definition=LIFESNAPS_OBSERVATION,
                )
                result["anchors"] = anchors(
                    own, fit.model, config, schedule, preparation["context_names"]
                )
                result["edge_diagnostics"] = edge_checks(own, fit.model)
            else:
                result["sham_incident_edges"] = [
                    e
                    for e in summary["graph"]
                    if e["source"] == "sham" or e["target"] == "sham"
                ]
                result["sham_candidate_denominator"] = sham_denominator(
                    own, fit.model, config, preparation["context_names"]
                )
            return result

        run.cell(arm, execute, kind="panel", method=arm)
    rng = np.random.default_rng(args.seed + 1234)
    for pair in range(args.pairs):
        permutation = rng.permutation(len(records))
        if len(records) < 2:
            raise ValueError(
                "complementary half-panel fits require at least two participants"
            )
        for side, indices in enumerate(np.array_split(permutation, 2)):
            key = f"half-{pair:03}-{side}"
            own = [records[i] for i in indices]

            def execute_half():
                fit, validation = selected_fit(own, args)
                return dict(
                    status=fit.status,
                    subjects=[r.name for r in own],
                    validation=validation,
                    observation_definition=LIFESNAPS_OBSERVATION,
                    fit=save_fit(
                        run.path / key, fit, own, preparation["context_names"]
                    ),
                )

            run.cell(key, execute_half, kind="half", pair=pair, side=side)
    full = run.rows["full"].get("fit", {}).get("graph", [])
    write_json(
        run.path / "stability.json",
        stability_summary(list(run.rows.values()), full, args.pairs),
    )


def deep_predictions(train, actual, method, epochs, seed):
    from src.evaluation import deep_baselines as deep

    if not deep.torch_available():
        raise ImportError("deep baselines require torch; use the CPU Docker image")
    import torch

    out = []
    for k, (history, future) in enumerate(zip(train, actual)):
        horizon = len(future)
        context = min(max(horizon, 4), len(history) - horizon)
        if context < 2:
            raise ValueError(
                "not enough training epochs for deep baseline at every scale"
            )
        mean, sd = history.mean(axis=0), history.std(axis=0)
        sd = np.where(sd > 1e-8, sd, 1)
        normalized = (history - mean) / sd
        X, Y = deep.windows(normalized, context, horizon)
        # Seed before parameter initialization, as well as minibatch training.
        torch.manual_seed(seed + k)
        model = getattr(deep, method)(context=context, horizon=horizon)
        deep.train_model(model, X, Y, epochs=epochs, seed=seed + k)
        forecast = deep.predict(model, normalized[-context:][None])[0]
        out.append(forecast * sd + mean)
    return out


def forecasting(args):
    from src.data.quito import (
        PANELS,
        four_channel_ids,
        read_panel,
        split_at_cutoff,
        window_to_multiscale,
    )

    inputs = [
        str(Path(args.data) / PANELS[name].filename) for name in args.panels.split(",")
    ]
    run = Run(args, inputs)
    methods = ["nested", "nested-constrained", "naive", "seasonal-naive"] + [
        m + "-" + r
        for m in ("DLinear", "PatchTST")
        for r in ("independent", "bottom-up", "ols", "nnls")
    ]
    windows_to_run = []
    for name in args.panels.split(","):
        panel = PANELS[name]
        grid, data = read_panel(panel, args.data)
        cut = (
            split_at_cutoff(grid)
            if args.split == "cutoff"
            else (len(grid) // panel.block - args.horizon_blocks - args.windows + 1)
            * panel.block
        )
        ids = four_channel_ids({item: values[:cut] for item, values in data.items()})
        if len(ids) < args.series:
            raise ValueError(
                f"{name} has {len(ids)} eligible series; requested {args.series}"
            )
        for item in ids[: args.series]:
            for window in range(args.windows):
                end = cut + window * panel.block
                horizon = args.horizon_blocks * panel.block
                if end + horizon > len(grid):
                    raise ValueError(
                        "requested evaluation windows exceed the published panel"
                    )
                start = end - min(args.train_blocks, end // panel.block) * panel.block
                history = Record(
                    str(item),
                    window_to_multiscale(
                        data[item][start:end],
                        panel,
                        offset=start,
                        total_finest=len(grid),
                    ),
                )
                future = Record(
                    str(item),
                    window_to_multiscale(
                        data[item][end : end + horizon],
                        panel,
                        offset=end,
                        total_finest=len(grid),
                    ),
                )
                windows_to_run.append(
                    (name, item, window, history, future, panel, str(grid[end]))
                )
    run.plan(
        [
            f"{name}-{item}-{window}/{method}"
            for name, item, window, *_ in windows_to_run
            for method in methods
        ]
    )
    for name, item, window, history, future, panel, origin in windows_to_run:
        prefix = f"{name}-{item}-{window}"
        identity = dict(
            panel=name,
            series=item,
            window=window,
            origin=origin,
            split=args.split,
            horizon_finest=len(future.data.X[0]),
            history_finest=len(history.data.X[0]),
        )

        def score(predicted):
            return forecast_metrics(
                future.data.X,
                predicted,
                history.data.X,
                future.data.hierarchy,
                evaluation_scale=1 if name == "min" else 0,
            )

        for method in ("nested", "nested-constrained"):

            def execute_nested():
                config = fit_config(
                    args, lag_radius=0.8 if method.endswith("constrained") else None
                )
                fit, validation = selected_fit(
                    [history], args, count_only=True, config=config
                )
                summary = save_fit(run.path / prefix / method, fit, [history])
                predicted, diagnostics = predict(
                    fit.model,
                    history,
                    future,
                    fit.posteriors[history.name],
                    return_diagnostics=True,
                )
                write_arrays(
                    run.path / prefix / method / "forecast.npz",
                    {f"scale{k + 1}": v for k, v in enumerate(predicted)},
                )
                return dict(
                    status=fit.status,
                    **score(predicted),
                    fit=summary,
                    validation=validation,
                    diagnostics=diagnostics,
                    predictor="coherent plug-in recursion, not an exact predictive expectation",
                    constraint=(
                        "prespecified sensitivity arm rho=0.8"
                        if method.endswith("constrained")
                        else "none"
                    ),
                )

            run.cell(prefix + "/" + method, execute_nested, method=method, **identity)
        for method in ("naive", "seasonal-naive"):

            def execute_naive():
                period = 1 if method == "naive" else (144 if name == "min" else 24)
                base = history.data.X[0][-period:]
                bottom = np.tile(
                    base, (int(np.ceil(len(future.data.X[0]) / len(base))), 1)
                )[: len(future.data.X[0])]
                return score(aggregate_forecast(bottom, future.data.hierarchy))

            run.cell(prefix + "/" + method, execute_naive, method=method, **identity)
        for method in ("DLinear", "PatchTST"):
            cache = run.path / prefix / (method + "-base.npz")

            def base_forecast():
                if cache.exists():
                    with np.load(cache) as stored:
                        return [stored[f"scale{k}"] for k in range(future.data.K)]
                values = deep_predictions(
                    history.data.X,
                    future.data.X,
                    method,
                    args.epochs,
                    args.seed + item + window,
                )
                cache.parent.mkdir(parents=True, exist_ok=True)
                write_arrays(cache, {f"scale{k}": v for k, v in enumerate(values)})
                return values

            for variant in ("independent", "bottom-up", "ols", "nnls"):

                def execute_deep():
                    base = base_forecast()
                    if variant == "bottom-up":
                        values = aggregate_forecast(base[0], future.data.hierarchy)
                    elif variant in ("ols", "nnls"):
                        values = reconcile(
                            base, future.data.hierarchy, nonnegative=variant == "nnls"
                        )
                    else:
                        values = base
                    return score(values)

                run.cell(
                    prefix + "/" + method + "-" + variant,
                    execute_deep,
                    method=method + "-" + variant,
                    **identity,
                )
        write_json(
            run.path / "summary.json", summarize_forecasts(list(run.rows.values()))
        )


def scalability(args):
    run = Run(args)
    settings = (
        [("labels", n) for n in (2, 4, 8, 16)]
        + [("length", n) for n in (576, 1152, 2304, 4608)]
        + [("lag", n) for n in (1, 2, 4, 8)]
        + [("indegree", n) for n in (1, 2, 3, 4)]
    )
    run.plan(
        [
            f"{axis}-{value}-r{rep}"
            for axis, value in settings
            for rep in range(args.replicates)
        ]
    )
    write_json(
        run.path / "hardware.json",
        {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "requested_jobs": args.jobs,
            "memory_budget_gb": args.memory_gb,
            "threads": {
                k: os.environ.get(k)
                for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
            },
            "memory_measure": "parent process lifetime resident high-water mark; excludes private worker allocations, not per-cell allocation",
        },
    )
    for axis, value in settings:
        for rep in range(args.replicates):
            key = f"{axis}-{value}-r{rep}"

            def execute():
                records, truth = generated(
                    args.seed + rep,
                    n=value if axis == "length" else 1152,
                    labels=value if axis == "labels" else 4,
                )
                config = fit_config(args, indegree=value if axis == "indegree" else 2)
                lag = [value] * 3 if axis == "lag" else [2, 2, 1]
                fit = fitting(records, args, config=config, lag=lag)
                return dict(
                    status=fit.status,
                    fit=save_fit(run.path / key, fit, records),
                    metrics=structural_metrics(fit.model, truth),
                )

            run.cell(key, execute, axis=axis, value=value, replicate=rep)


def benchmark(args):
    """CalendarBench manifest correspondence under the corrected model.

    The manifest contains cross-quantity rules outside the paper's two same-
    quantity predictors. Keep those in the reachability ledger; do not count a
    label-only match as recovery of an unsupported typed rule.
    """
    from src.data.dynamic_patterns import read_declared_manifest
    from src.data.lifesnaps_panel import PANEL_NODES

    run = Run(args, [args.csv, args.manifest])
    records, preparation = calendarbench(
        args.csv, args.participants, annotation_path=run.path / "annotations.csv"
    )
    labels = records[0].data.label_names
    write_json(run.path / "audit.json", preparation)
    if not preparation["valid"]:
        raise ValueError(
            f"CalendarBench grouped input violates support; inspect {run.path / 'audit.json'}:\n"
            + "\n".join(preparation["support_errors"][:5])
        )
    grouped_totals = preparation["grouping"]["by_label"]
    print(
        f"CalendarBench audit passed: {len(records)} participants, {len(labels)} labels. "
        f"Report: {run.path / 'audit.json'}",
        flush=True,
    )
    print(
        f"{sum(v['annotations_before'] for v in grouped_totals.values())} annotations -> "
        f"{sum(v['groups_after'] for v in grouped_totals.values())} groups; "
        "all duration marks preserved. "
        f"Original annotations: {run.path / 'annotations.csv'}",
        flush=True,
    )
    declared = read_declared_manifest(args.manifest)
    ledger, reachable = [], set()
    for rule in declared.rules:
        source_quantity = "count" if rule.source_quantity == "count" else "duration"
        target_quantity = "count" if rule.target_quantity == "episodes" else "duration"
        on_axis = rule.source in labels and rule.target in labels
        changed_counts = any(
            quantity == "count"
            and grouped_totals.get(label, {}).get("count_reduction", 0) > 0
            for label, quantity in (
                (rule.source, source_quantity),
                (rule.target, target_quantity),
            )
        )
        expressible = (
            on_axis
            and source_quantity == target_quantity
            and rule.lag_days <= args.lag
            and rule.source_quantity != "max_duration"
            and not changed_counts
        )
        ledger.append(
            dict(
                source=rule.source,
                target=rule.target,
                lag=rule.lag_days,
                source_quantity=rule.source_quantity,
                target_quantity=rule.target_quantity,
                observation_definition=CALENDARBENCH_OBSERVATION,
                count_quantity_changed_by_grouping=changed_counts,
                expressible=expressible,
                reason=(
                    "same-quantity log predictor in lag bound"
                    if expressible
                    else "grouped count changed, cross-quantity, maximum-duration functional, missing label, or lag outside search"
                ),
            )
        )
        if expressible:
            reachable.add(
                (2, labels.index(rule.target), labels.index(rule.source), rule.lag_days)
            )
    write_json(run.path / "manifest-reachability.json", ledger)
    if args.audit_only:
        return
    methods = ["nested+latent", "intercept"] + list(CLASSICAL)
    run.plan(methods)
    for method in methods:

        def execute():
            if method == "nested+latent":
                fit, validation = selected_fit(records, args)
            elif method == "intercept":
                own = [
                    Record(
                        r.name,
                        dataclasses.replace(
                            r.data, C=[np.zeros_like(c) for c in r.data.C]
                        ),
                    )
                    for r in records
                ]
                fit = fitting(
                    own,
                    args,
                    latent=False,
                    config=fit_config(
                        args,
                        structure=False,
                        context_mode="none",
                        shifts=0,
                        depletion=False,
                    ),
                )
                validation = {"selection": "prespecified intercept baseline"}
            else:
                fixed = classical(records, method, args.lag, args.indegree, args.seed)
                fit = fitting(
                    records,
                    args,
                    fixed=fixed,
                    config=fit_config(args, structure=False, shifts=0),
                )
                validation = {
                    "selection": "external count-graph search; duration-specific typed rules not evaluated"
                }
            summary = save_fit(run.path / method, fit, records, PANEL_NODES)
            found = {
                (k, ell, p.source, p.lag)
                for k, ell, p in fit.model.edges()
                if p.kind == "behavior" and k == 2
            }
            hits = []
            for rule in ledger:
                key = (
                    (
                        2,
                        labels.index(rule["target"]),
                        labels.index(rule["source"]),
                        rule["lag"],
                    )
                    if rule["target"] in labels and rule["source"] in labels
                    else None
                )
                hits.append(
                    dict(
                        rule,
                        selected_label_relation=key in found,
                        interpretation="label-level correspondence; inspect fitted channel coefficient before any typed recovery claim",
                    )
                )
            stationary = set(declared.stationary_labels)
            result = dict(
                status=fit.status,
                fit=summary,
                validation=validation,
                manifest=hits,
                observation_definition=CALENDARBENCH_OBSERVATION,
                stationary_target_edges=[
                    e
                    for e in summary["graph"]
                    if e["kind"] == "behavior" and e["target"] in stationary
                ],
                reachable_label_relations=len(reachable),
                reachable_label_matches=len(found & reachable),
            )
            if method in ("nested+latent", "intercept"):
                result["fidelity"] = simulations(
                    run.path / method / "simulations",
                    records,
                    fit.model,
                    args.replicates,
                    args.seed + 5000,
                    observation_definition=CALENDARBENCH_OBSERVATION,
                )
            return result

        run.cell(method, execute, method=method)


def smoke(args):
    """Small integration check; never evidence for a paper claim."""
    run = Run(args)
    run.plan(["cohort"])

    def execute():
        records, _ = generated(args.seed, n=48, labels=2, participants=2)
        fit = fitting(
            records,
            args,
            latent=False,
            config=fit_config(args, structure=False, shifts=0),
        )
        summary = save_fit(run.path / "cohort", fit, records)
        fidelity = simulations(
            run.path / "cohort" / "simulations",
            records,
            fit.model,
            2,
            args.seed,
            forms=False,
        )
        return dict(
            fit=summary, fidelity=fidelity, audit=audit(records), paper_evidence=False
        )

    row = run.cell("cohort", execute)
    if row["status"] != "complete":
        raise RuntimeError(row.get("reason", "smoke failed"))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "experiment",
        choices=[
            "smoke",
            "synthetic",
            "lifesnaps",
            "forecast",
            "scalability",
            "benchmark",
            "report",
        ],
    )
    p.add_argument("--output", default="results/revision")
    p.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="nested fit worker processes; 0 chooses up to 6 automatically",
    )
    p.add_argument(
        "--memory-gb",
        type=float,
        default=0,
        help="total estimated fitting RAM budget in GiB; 0 uses available RAM",
    )
    p.add_argument("--outer", type=int, default=5)
    p.add_argument("--inner", type=int, default=3)
    p.add_argument("--profile", type=int, default=25)
    p.add_argument("--indegree", type=int, default=2)
    p.add_argument("--lag", type=int, default=2)
    p.add_argument("--penalty", type=float, default=1)
    p.add_argument("--search", choices=["greedy", "exhaustive"], default="greedy")
    p.add_argument("--shifts", type=int, default=19)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument(
        "--seconds",
        type=float,
        default=0,
        help="per cell; zero means no wall-time limit",
    )
    p.add_argument(
        "--tune",
        action="store_true",
        help="select penalty on last 30%% of coarsest epochs before refitting all training data",
    )
    p.add_argument("--replicates", type=int, default=10)
    p.add_argument("--length", type=int, default=1152)
    p.add_argument("--labels", type=int, default=4)
    p.add_argument("--csv", default="data/observed_lifesnaps_event_log.csv")
    p.add_argument("--manifest", default="data/dynamic_patterns/manifest_declared.json")
    p.add_argument("--participants", type=int, default=51)
    p.add_argument("--weeks", type=int, default=8)
    p.add_argument("--pairs", type=int, default=25)
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--data", default="data/quito")
    p.add_argument("--panels", default="min")
    p.add_argument(
        "--series",
        type=int,
        default=40,
        help="training-eligible series per panel, sorted by identifier",
    )
    p.add_argument("--windows", type=int, default=1)
    p.add_argument("--split", choices=["cutoff", "extended"], default="cutoff")
    p.add_argument("--horizon-blocks", type=int, default=2)
    p.add_argument("--train-blocks", type=int, default=39)
    p.add_argument("--epochs", type=int, default=30)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.experiment == "report":
        from src.nested.report import report

        report(args.output)
        return
    if (
        min(
            args.replicates,
            args.outer,
            args.inner,
            args.profile,
            args.lag,
            args.series,
            args.windows,
        )
        < 1
        or args.pairs < 0
        or args.jobs < 0
        or not np.isfinite(args.memory_gb)
        or args.memory_gb < 0
    ):
        raise ValueError(
            "iteration, replicate, lag and series counts must be positive; pairs nonnegative"
        )
    functions = dict(
        smoke=smoke,
        synthetic=synthetic,
        lifesnaps=lifesnaps_run,
        forecast=forecasting,
        scalability=scalability,
        benchmark=benchmark,
    )
    functions[args.experiment](args)


if __name__ == "__main__":
    main()
