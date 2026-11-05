"""LifeSnaps checks on the actual revised candidate space and retained graph."""

from __future__ import annotations

import dataclasses

import numpy as np

from .fit import Fitter
from .law import feature


def candidate_keys(records, model, config, context_names):
    fitter = Fitter(config)
    labels = records[0].data.label_names
    keys = set()
    for k in range(model.K):
        for p in fitter.candidates(records, model, k):
            source = (
                labels[p.source] if p.kind == "behavior" else context_names[p.source]
            )
            for target in labels:
                keys.add((k + 1, p.lag, source, target))
    return keys


def anchors(records, model, config, schedule, context_names):
    from ..data.anchors import anchor_set

    # Preserve the original three-lag declaration. Inaccessible anchors are
    # reported explicitly, instead of changing the denominator after fitting.
    declared = anchor_set(schedule.axis)
    names = ("minute", "hour", "day", "week")
    offered = {
        (names[k - 1], lag, source, target)
        for k, lag, source, target in candidate_keys(
            records, model, config, context_names
        )
    }
    selected = set()
    labels = records[0].data.label_names
    for k, ell, p in model.edges():
        source = labels[p.source] if p.kind == "behavior" else context_names[p.source]
        selected.add((names[k], p.lag, source, labels[ell]))
    rows = []
    for anchor in declared.anchors:
        key = anchor.key.as_row()
        rows.append(
            dict(
                scale=key[0],
                lag=key[1],
                source=key[2],
                target=key[3],
                sign=anchor.sign,
                standing=anchor.standing,
                candidate=key in offered,
                selected=key in selected,
            )
        )
    result = {
        "declaration_digest": declared.digest(),
        "rows": rows,
        "interpretation": "protocol diagnostics, not proof of causal identification",
    }
    for sign in ("positive", "negative"):
        subset = [row for row in rows if row["sign"] == sign]
        reachable = [row for row in subset if row["candidate"]]
        result[sign] = {
            "declared": len(subset),
            "actually_offered": len(reachable),
            "selected": sum(row["selected"] for row in reachable),
            "not_offered": len(subset) - len(reachable),
        }
    return result


def edge_checks(records, model):
    """OLS future-column and half-window diagnostics, not causal tests.

    Rows are constructed inside each participant's observed intervals. These
    descriptive regressions do not replace the model's normalized score refits.
    """
    from ..evaluation.falsification import future_lag_control

    checks = []
    for k, ell, parent in model.edges():
        responses, pasts, futures, halves = [], [], [], []
        for rec in records:
            seen = rec.rows(k, max(model.lag_bound[k], parent.lag))
            times = np.arange(len(seen))
            shift = max(parent.lag, 1)
            future_idx = times + shift
            valid = seen & (future_idx < len(seen))
            for d in range(1, shift + 1):
                valid[: len(seen) - d] &= rec.data.get_observed(k + 1)[d:]
            idx = times[valid]
            source = (
                rec.data.C[k][:, parent.source]
                if parent.kind == "context"
                else np.log1p(rec.data.X[k][:, parent.source])
            )
            responses.extend(np.log1p(rec.data.X[k][idx, ell]))
            pasts.extend(feature(rec, k, parent, 0, indices=idx))
            futures.extend(source[future_idx[idx]])
            halves.extend(idx >= len(seen) // 2)
        response, past, future, half = map(
            np.asarray, (responses, pasts, futures, halves)
        )
        row = dict(
            scale=k + 1,
            target=ell,
            parent=vars(parent),
            observations=len(response),
            status="descriptive",
        )
        if len(response) > 4 and np.std(past) > 0:
            row["future_column"] = dataclasses.asdict(
                future_lag_control(response, past, future)
            )
            coefficients = []
            for side in (False, True):
                mask = half == side
                if mask.sum() > 2 and np.std(past[mask]) > 0:
                    coefficients.append(
                        float(
                            np.linalg.lstsq(
                                np.column_stack([np.ones(mask.sum()), past[mask]]),
                                response[mask],
                                rcond=None,
                            )[0][1]
                        )
                    )
                else:
                    coefficients.append(None)
            row["half_window_ols_coefficients"] = coefficients
        checks.append(row)
    return checks


def sham_denominator(records, model, config, context_names):
    candidates = candidate_keys(records, model, config, context_names)
    return sum(
        source == "sham" or target == "sham" for _, _, source, target in candidates
    )
