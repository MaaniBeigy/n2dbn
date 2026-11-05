"""Coherent count-only plug-in forecasts conditioned on the training window."""

from __future__ import annotations

import numpy as np
from scipy.special import expit, softmax

from ..core.variables import MultiScaleData
from .inference import infer
from .law import exp_mean, scores
from .model import Record


def predict(model, history, window, posterior=None, return_diagnostics=False):
    """Plug-in recursion, not an exact expectation of the stochastic sampler."""
    if not model.count_only:
        raise ValueError("this forecast path is the explicitly count-only variant")
    posterior = infer(history, model) if posterior is None else posterior
    diagnostics = {
        "log_link_evaluations": 0,
        "log_link_clipped": 0,
        "max_absolute_unclipped_log_mean": 0.0,
    }
    data = window.data
    X = [np.zeros_like(x, dtype=float) for x in data.X]
    Y = [np.zeros_like(y, dtype=float) for y in data.Y]
    future = Record(
        history.name, MultiScaleData(data.hierarchy, data.label_names, X, Y, data.C)
    )
    marginals, factors = [None] * model.K, [None] * model.K
    for k in range(model.K - 1, -1, -1):
        level = model.levels[k]
        gamma = np.zeros((len(X[k]), level.regime.M))
        h = np.zeros((len(X[k]), level.factor.r))
        previous_g = posterior[k].gamma[-1]
        previous_h = posterior[k].m_H[-1]
        for t in range(len(X[k])):
            rp = level.regime
            logits = rp.bias[:, None] + rp.trans + (rp.ctx @ data.C[k][t])[:, None]
            if rp.cross is not None:
                parent = marginals[k + 1][data.hierarchy.parent_maps[k][t]]
                logits += (rp.cross @ parent)[:, None]
            gamma[t] = softmax(logits, axis=0) @ previous_g
            if level.factor.r:
                h[t] = sum(
                    gamma[t, m]
                    * (
                        level.factor.F[m] @ previous_h
                        + level.factor.G[m] @ data.C[k][t]
                    )
                    for m in range(level.regime.M)
                )
            previous_g, previous_h = gamma[t], h[t]
        marginals[k], factors[k] = gamma, h

        def lag_value(parent, times, current):
            index = times - parent.lag
            values = np.zeros(len(times))
            if parent.kind == "context":
                inside = index >= 0
                values[inside] = data.C[k][index[inside], parent.source]
                past = history.data.C[k]
                valid = (index < 0) & (len(past) + index >= 0)
                values[valid] = past[index[valid], parent.source]
            else:
                inside = (index >= 0) & (index < current)
                values[inside] = np.log1p(X[k][index[inside], parent.source])
                past = history.data.X[k]
                valid = (index < 0) & (len(past) + index >= 0)
                values[valid] = np.log1p(past[index[valid], parent.source])
            return values

        if model.emission(k):
            for t in range(len(X[k])):
                for ell, target in enumerate(level.targets):
                    value = 0.0
                    lag = sum(
                        coef[0] * lag_value(parent, np.array([t]), t)[0]
                        for parent, coef in target.parents.items()
                    )
                    offset = (
                        model.offsets.get(history.name, np.zeros((2, model.L)))[0, ell]
                        if k == model.K - 1
                        else 0.0
                    )
                    for state in range(level.regime.M):
                        eta = (
                            target.intercept[0]
                            + offset
                            + target.regime[0, state]
                            + lag
                            + target.factor[0] @ h[t]
                        )
                        diagnostics["log_link_evaluations"] += 1
                        diagnostics["max_absolute_unclipped_log_mean"] = max(
                            diagnostics["max_absolute_unclipped_log_mean"],
                            float(abs(eta)),
                        )
                        zero = (
                            target.zero_logit
                            + target.zero_regime[state]
                            + target.zero_factor @ h[t]
                            + target.zero_context @ data.C[k][t]
                        )
                        value += gamma[t, state] * (1 - expit(zero)) * exp_mean(eta)
                    X[k][t, ell] = value
        else:
            bases = [
                scores(future, model, k, ell, gamma, h)[:, 0] for ell in range(model.L)
            ]
            for c in range(data.hierarchy.H[k + 1]):
                idx = np.asarray(data.hierarchy.children(k + 1, c))
                remaining = X[k + 1][c].copy()
                for i, t in enumerate(idx):
                    for ell, target in enumerate(level.targets):
                        eta = bases[ell][idx].copy()
                        for parent, coef in target.parents.items():
                            eta += coef[0] * lag_value(parent, idx, t)
                        prefix = np.cumsum(X[k][idx, ell]) - X[k][idx, ell]
                        eta += target.depletion[0] * np.log1p(
                            np.minimum(prefix, prefix[i])
                        )
                        weights = softmax(eta[i:])
                        value = (
                            remaining[ell]
                            if i == len(idx) - 1
                            else remaining[ell] * weights[0]
                        )
                        X[k][t, ell] = value
                        remaining[ell] -= value
    return (X, diagnostics) if return_diagnostics else X
