"""Claim-facing metrics: all labels, all runs, explicit failures and controls."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.optimize import nnls

from . import FORMULATION


def jsonable(value):
    if dataclasses.is_dataclass(value):
        return jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(jsonable(content), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_arrays(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def graph_rows(model, labels, context_names=None):
    return [
        {
            "scale": k + 1,
            "target": labels[ell],
            "kind": p.kind,
            "source": (
                labels[p.source]
                if p.kind == "behavior"
                else (
                    context_names[p.source] if context_names else f"context{p.source}"
                )
            ),
            "lag": p.lag,
            "count_coefficient": model.levels[k].targets[ell].parents[p][0],
            "duration_coefficient": model.levels[k].targets[ell].parents[p][1],
        }
        for k, ell, p in model.edges()
    ]


def fit_summary(fit):
    return {
        "model_formulation": FORMULATION,
        "count_only": fit.model.count_only,
        "nested": fit.model.nested,
        "status": fit.status,
        "elapsed_seconds": fit.elapsed_seconds,
        "score_evaluations": fit.evaluations,
        "objective_kind": "auxiliary_chain_surrogate_not_elbo",
        "trace": fit.trace,
        "shift_checks": fit.shift_checks,
        "regimes": [level.regime.M for level in fit.model.levels],
        "factors": [level.factor.r for level in fit.model.levels],
        "lag_bounds": fit.model.lag_bound,
        "penalties": fit.model.penalties,
        "shared_graph_penalty": fit.model.penalty(),
        "inference_seconds": fit.inference_seconds,
        "search_seconds": fit.search_seconds,
        "candidate_refits": fit.candidate_refits,
        "depletion_coefficients": [
            [target.depletion for target in level.targets] for level in fit.model.levels
        ],
    }


def structural_metrics(model, truth):
    selected = {
        (k, ell, p.source, p.lag) for k, ell, p in model.edges() if p.kind == "behavior"
    }
    truth = set(truth)
    tp = len(selected & truth)
    result = {
        "precision": tp / len(selected) if selected else 0.0,
        "recall": tp / len(truth) if truth else 0.0,
        "f1": 2 * tp / (len(selected) + len(truth)) if selected or truth else 1.0,
        "shd": len(selected ^ truth),
        "empty_shd": len(truth),
        "selected": len(selected),
        "true_edges": len(truth),
        "by_scale": [],
    }
    for k in range(model.K):
        a, b = {e for e in selected if e[0] == k}, {e for e in truth if e[0] == k}
        result["by_scale"].append(
            {
                "scale": k + 1,
                "f1": 2 * len(a & b) / (len(a) + len(b)) if a or b else 1.0,
                "false_positives": len(a - b),
                "shd": len(a ^ b),
            }
        )
    for name, self_edge in (("self_recall", True), ("cross_recall", False)):
        subset = {e for e in truth if (e[1] == e[2]) == self_edge}
        result[name] = len(selected & subset) / len(subset) if subset else None
    return result


def volume_totals(records):
    return {
        channel: np.sum(
            [
                getattr(r.data, channel)[0][r.data.get_observed(1)].sum(axis=0)
                for r in records
            ],
            axis=0,
        )
        for channel in ("X", "Y")
    }


def volumes(real, simulations=(), *, observed_totals=None, simulated_totals=None):
    labels = real[0].data.label_names
    observed_totals = (
        volume_totals(real) if observed_totals is None else observed_totals
    )
    simulated_totals = (
        [volume_totals(draw) for draw in simulations]
        if simulated_totals is None
        else simulated_totals
    )

    result = {}
    for name, channel in (("counts", "X"), ("booked_duration", "Y")):
        observed = observed_totals[channel]
        replicates = np.array([totals[channel] for totals in simulated_totals])
        drawn = replicates.mean(axis=0)
        ratios = np.divide(
            drawn, observed, out=np.full(len(labels), np.nan), where=observed > 0
        )
        log_errors = np.abs(
            np.log(ratios, where=ratios > 0, out=np.full(len(labels), -np.inf))
        )
        result[name] = {
            "by_label": {
                label: {
                    "observed": observed[i],
                    "simulated_mean": drawn[i],
                    "replicate_values": replicates[:, i],
                    "ratio": ratios[i],
                    "absolute_log_ratio": log_errors[i],
                }
                for i, label in enumerate(labels)
            },
            "all_label_mean_absolute_log_ratio": float(np.mean(log_errors)),
            "zero_observed_labels": [labels[i] for i in np.flatnonzero(observed == 0)],
            "zero_simulated_labels": [labels[i] for i in np.flatnonzero(drawn == 0)],
            "total_ratio": drawn.sum() / observed.sum() if observed.sum() else None,
        }
        subset = [
            labels.index(label)
            for label in ("gym", "nap", "sleep", "walk")
            if label in labels
        ]
        result[name]["four_label_mean_absolute_log_ratio"] = (
            float(np.mean(log_errors[subset])) if len(subset) == 4 else None
        )
    return result


def longest_run(mask):
    indices = np.flatnonzero(mask)
    if not len(indices):
        return np.array([], dtype=int)
    runs = np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1)
    return max(runs, key=len)


def nonstationarity(real, simulations):
    """All declared forms, preserving contiguous time and participant pairing."""
    from ..evaluation.nonstationarity import inclusion
    from ..evaluation.panel_compare import (
        FORMS,
        JOINT_LABEL,
        POWER_BOUND,
        Series,
        admits,
        denominator,
        form_values,
        minimum_detectable_cell,
        standard_error,
        tolerance_of,
    )
    from ..evaluation.segmentation import match_changepoints

    labels = real[0].data.label_names
    scale_names = {
        real[0].data.K - 3: "hour",
        real[0].data.K - 2: "day",
        real[0].data.K - 1: "week",
    }
    cells, changepoints = [], []
    for k, scale in scale_names.items():
        if k < 0:
            continue
        runs = [longest_run(r.data.get_observed(k + 1)) for r in real]
        for ell, label in list(enumerate(labels)) + [(None, JOINT_LABEL)]:

            def series(records):
                return Series(
                    label,
                    scale,
                    ell is None,
                    tuple(
                        (
                            r.data.X[k][idx].astype(float)
                            if ell is None
                            else r.data.X[k][idx, ell].astype(float)
                        )
                        for r, idx in zip(records, runs)
                        if len(idx) >= 4
                    ),
                )

            observed = series(real)
            draws = [series(s) for s in simulations]
            if not observed.blocks:
                cells.append(
                    {
                        "label": label,
                        "scale": scale,
                        "status": "descriptive",
                        "reason": "no four-epoch contiguous observed interval",
                    }
                )
                continue
            for spec in FORMS:
                if not admits(spec, observed):
                    continue
                try:
                    a = form_values(spec, observed)
                    b = np.array([form_values(spec, draw) for draw in draws])
                    tolerance = float(tolerance_of(spec, observed, 0, a))
                    detectable = minimum_detectable_cell(
                        spec, observed, tolerance=tolerance, base=a
                    )
                    counts = denominator(observed, observed.period)
                    eligibility = inclusion(
                        counts,
                        detectable,
                        POWER_BOUND,
                        needs_period=spec.id in ("A05", "A06"),
                        tolerance=tolerance,
                        standard_error=standard_error(a),
                    )
                    difference = a - b.mean(axis=0)
                    valid = np.isfinite(difference)
                    difference = difference[valid]
                    row = {
                        "form": spec.id,
                        "label": label,
                        "scale": scale,
                        "participants": len(difference),
                        "tolerance": tolerance,
                        "status": "descriptive",
                        "denominator": dataclasses.asdict(counts),
                        "minimum_detectable_effect": detectable,
                        "eligibility": dataclasses.asdict(eligibility),
                        "nonfinite_participants": int((~valid).sum()),
                    }
                    if len(difference) > 1 and np.isfinite(tolerance) and tolerance > 0:
                        mean = float(difference.mean())
                        half = float(
                            stats.t.ppf(0.95, len(difference) - 1)
                            * stats.sem(difference)
                        )
                        row.update(
                            {
                                "mean_difference": mean,
                                "ci90": [mean - half, mean + half],
                            }
                        )
                        if eligibility.tested and valid.all():
                            row["status"] = (
                                "equivalent"
                                if abs(mean) + half < tolerance
                                else "not_equivalent"
                            )
                        if len(b) > 1:
                            control = b[0] - b[1:].mean(axis=0)
                            control = control[np.isfinite(control)]
                            if len(control) > 1:
                                width = stats.t.ppf(0.95, len(control) - 1) * stats.sem(
                                    control
                                )
                                row["replicate_control_equivalent"] = bool(
                                    abs(control.mean()) + width < tolerance
                                )
                    cells.append(row)
                except (
                    ValueError,
                    np.linalg.LinAlgError,
                    FloatingPointError,
                    ZeroDivisionError,
                ) as exc:
                    cells.append(
                        {
                            "form": spec.id,
                            "label": label,
                            "scale": scale,
                            "status": "descriptive",
                            "reason": str(exc),
                        }
                    )
            if ell is not None:
                matches = []
                for draw in draws:
                    for a, b in zip(observed.blocks, draw.blocks):
                        matches.append(
                            match_changepoints(
                                a, b, penalty=3 * np.log(max(len(a), 2)), margin=1
                            ).as_row()
                        )
                changepoints.append(
                    {
                        "label": label,
                        "scale": scale,
                        "mean_f1": float(np.mean([m["f1_margin"] for m in matches])),
                        "comparisons": len(matches),
                        "rows": matches,
                    }
                )
    return {
        "cells": cells,
        "changepoints": changepoints,
        "missingness_policy": "longest contiguous observed interval per participant, shared with every replicate",
        "equivalence_policy": "paired participant differences; 90% t interval within observed-data tolerance; controls never remove rows",
    }


def aggregate_forecast(finest, hierarchy):
    arrays = [np.asarray(finest, dtype=float)]
    for k in range(hierarchy.K - 1):
        next_array = np.zeros((hierarchy.H[k + 1], finest.shape[1]))
        np.add.at(next_array, hierarchy.parent_maps[k], arrays[-1])
        arrays.append(next_array)
    return arrays


def reconcile(forecasts, hierarchy, nonnegative=False):
    """OLS reconciliation on original count units, never standardized units."""
    identity = np.eye(hierarchy.H[0])
    A = np.vstack(aggregate_forecast(identity, hierarchy))
    base = np.vstack(forecasts)
    if nonnegative:
        bottom = np.column_stack([nnls(A, base[:, j])[0] for j in range(base.shape[1])])
    else:
        bottom = np.linalg.lstsq(A, base, rcond=None)[0]
    return aggregate_forecast(bottom, hierarchy)


def forecast_metrics(actual, predicted, train, hierarchy, evaluation_scale=1):
    from ..evaluation.forecast_metrics import mae, mse, relative_coherence, train_scaler

    finite = all(np.isfinite(x).all() for x in predicted)
    if not finite:
        return {
            "finite": False,
            "mae": None,
            "mse": None,
            "coherence": None,
            "amplitude": None,
        }
    scale = min(evaluation_scale, len(actual) - 1)
    mean, sd = train_scaler(train[scale])
    amplitude = np.max(predicted[0] / (1 + np.max(train[0], axis=0)))
    result = {
        "finite": True,
        "mae": mae((actual[scale] - mean) / sd, (predicted[scale] - mean) / sd),
        "mse": mse((actual[scale] - mean) / sd, (predicted[scale] - mean) / sd),
        "coherence": relative_coherence(predicted, hierarchy),
        "amplitude": float(amplitude),
        "extreme": bool(amplitude > 100),
    }
    if not all(
        np.isfinite(result[name]) for name in ("mae", "mse", "coherence", "amplitude")
    ):
        result["finite"] = False
        result["reason"] = (
            "finite forecast entries but non-finite error or coherence calculation"
        )
    return result


def summarize_forecasts(rows):
    result = {}

    def name(row):
        return row.get("method", row.get("key", "unknown").split("/")[-1])

    for method in sorted({name(r) for r in rows}):
        own = [r for r in rows if name(r) == method]
        finite = [
            r
            for r in own
            if r.get("finite") and r.get("status", "complete") == "complete"
        ]
        entry = {
            "runs": len(own),
            "finite_completed_runs": len(finite),
            "failed_runs": len(own) - len(finite),
            "nonfinite_runs": sum(r.get("finite") is False for r in own),
            "execution_failures": sum(
                r.get("status") in ("failed", "timeout", "time_budget") for r in own
            ),
            "pending_or_running": sum(
                r.get("status") in ("pending", "running") for r in own
            ),
            "extreme_runs": sum(r.get("extreme", False) for r in finite),
            "summary_population": "finite completed runs only; failures listed separately",
        }
        for metric in ("mae", "mse", "coherence", "amplitude"):
            values = np.array([r[metric] for r in finite], dtype=float)
            entry[metric] = (
                None
                if not len(values)
                else {
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "p90": float(np.quantile(values, 0.9)),
                    "max": float(np.max(values)),
                }
            )
        result[method] = entry
    return result
