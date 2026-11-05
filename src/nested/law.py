"""A single normalized law for likelihood evaluation and chronological draws.

Allocation uses the full-block softmax at each chronological stage. Future
endogenous lag reads have the zero reference stated in REVISE.md. Counts are
truncated to the interval that leaves a feasible remainder; durations allocate
Y-X over active children. There is no post-sampling repair or occupancy mask.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import betaln, expit, gammaln, logsumexp, xlogy
from scipy.stats import nbinom

from ..core.variables import MultiScaleData
from ..model.latent_factor import LatentFactor
from ..model.latent_regime import RegimeChain
from .model import Model, Parent, Record


def exp_mean(value):
    """The paper's log link, without clipping an unstable forecast to a cap."""
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        return np.exp(value)


def nb_logpmf(x, shape, mean):
    x, shape, mean = np.broadcast_arrays(x, shape, mean)
    out = (
        gammaln(x + shape)
        - gammaln(shape)
        - gammaln(x + 1)
        + shape * (np.log(shape) - np.log(shape + mean))
        + xlogy(x, mean)
        - x * np.log(shape + mean)
    )
    return np.where((x >= 0) & (x == np.floor(x)), out, -np.inf)


def coarse_logpmf(
    x, y, eta, zeta, zero_logit, dispersion, capacity=None, count_only=False
):
    """Censor the ZINB tail at capacity, and score the NB duration surplus."""
    mu = exp_mean(eta)
    log_pi = -np.logaddexp(0, -zero_logit)
    log_nonzero = -np.logaddexp(0, zero_logit)
    log_count = log_nonzero + nb_logpmf(x, dispersion[0], mu)
    log_count = np.where(x == 0, np.logaddexp(log_pi, log_count), log_count)
    if capacity is not None:
        # Survival probability is P(uncapped X >= capacity).
        tail = nbinom.logsf(
            np.asarray(capacity) - 1,
            dispersion[0],
            dispersion[0] / (dispersion[0] + mu),
        )
        log_count = np.where(x == capacity, log_nonzero + tail, log_count)
        log_count = np.where(x > capacity, -np.inf, log_count)
    if count_only:
        return log_count
    nu = exp_mean(zeta)
    duration = np.where(x > 0, nb_logpmf(y - x, dispersion[1], nu), 0.0)
    valid = (y >= x) & ((x > 0) | (y == 0))
    return np.where(valid, log_count + duration, -np.inf)


def bb_logpmf(x, n, a, b):
    if np.any(np.asarray(a) <= 0) or np.any(np.asarray(b) <= 0):
        raise FloatingPointError(
            "beta shapes must be positive; check softmax underflow"
        )
    return (
        gammaln(n + 1)
        - gammaln(x + 1)
        - gammaln(n - x + 1)
        + betaln(x + a, n - x + b)
        - betaln(a, b)
    )


def split_logpmf(x, n, a, b, lower, upper):
    """Normalized feasible beta-binomial mass, including deterministic cases."""
    x, n, a, b, lower, upper = np.broadcast_arrays(x, n, a, b, lower, upper)
    out = np.full(x.shape, -np.inf, dtype=float)
    feasible = (lower <= x) & (x <= upper) & (x == np.floor(x)) & (lower <= upper)
    forced = (lower == upper) | (b == 0) | (n == 0)
    value = np.where(b == 0, n, lower)
    out[feasible & forced & (x == value)] = 0.0
    idx = feasible & ~forced
    if not idx.any():
        return out
    xx, nn, aa, bb = x[idx], n[idx], a[idx], b[idx]
    lo, hi = lower[idx], upper[idx]
    ll = bb_logpmf(xx, nn, aa, bb)
    truncated = (lo > 0) | (hi < nn)
    if truncated.any():
        width = int(np.max(hi[truncated] - lo[truncated])) + 1
        grid = lo[truncated, None] + np.arange(width)
        mass = bb_logpmf(
            grid, nn[truncated, None], aa[truncated, None], bb[truncated, None]
        )
        mass = np.where(grid <= hi[truncated, None], mass, -np.inf)
        ll[truncated] -= logsumexp(mass, axis=1)
    out[idx] = ll
    return out


def split_sample(n, a, b, lower, upper, rng):
    if lower > upper:
        raise ValueError("empty feasible allocation interval")
    if b == 0:
        if not lower <= n <= upper:
            raise ValueError("infeasible final child")
        return int(n)
    if lower == upper:
        return int(lower)
    if lower == 0 and upper == n:
        if a <= 0 or b <= 0:
            raise FloatingPointError(
                "beta shapes must be positive; check softmax underflow"
            )
        return int(rng.binomial(n, rng.beta(a, b)))
    support = np.arange(lower, upper + 1)
    logp = split_logpmf(support, n, a, b, lower, upper)
    return int(rng.choice(support, p=np.exp(logp - logsumexp(logp))))


def feature(record, k, parent: Parent, channel, override=None, indices=None):
    """Lagged column before prefix censoring; never a same-epoch behavior."""
    if override is not None and parent in override:
        value = override[parent][channel]
        return value if indices is None else value[indices]
    d = record.data
    if indices is not None:
        lagged = np.asarray(indices) - parent.lag
        valid = lagged >= 0
        out = np.zeros(lagged.shape)
        source = (
            d.C[k] if parent.kind == "context" else (d.X if channel == 0 else d.Y)[k]
        )
        values = source[lagged[valid], parent.source]
        out[valid] = values if parent.kind == "context" else np.log1p(values)
        return out
    if parent.kind == "context":
        values = d.C[k][:, parent.source]
    else:
        values = np.log1p((d.X if channel == 0 else d.Y)[k][:, parent.source])
    out = np.zeros(len(values))
    if parent.lag == 0:
        out[:] = values
    elif parent.lag < len(values):
        out[parent.lag :] = values[: -parent.lag]
    return out


def scores(record, model, k, ell, gamma, h):
    """History-free scores; endogenous lag terms are added by the law."""
    target = model.levels[k].targets[ell]
    base = np.tile(target.intercept, (len(gamma), 1))
    base += gamma @ target.regime.T
    if h.shape[1]:
        base += h @ target.factor.T
    if k == model.K - 1:
        base += model.offsets.get(record.name, np.zeros((2, model.L)))[:, ell]
    return base


def emission_terms(record, model, k, ell, gamma, h, override=None):
    target = model.levels[k].targets[ell]
    base = scores(record, model, k, ell, gamma, h)
    for parent, coef in target.parents.items():
        for channel in range(1 if model.count_only else 2):
            base[:, channel] += coef[channel] * feature(
                record, k, parent, channel, override
            )
    zero = target.zero_logit + gamma @ target.zero_regime
    if h.shape[1]:
        zero += h @ target.zero_factor
    zero += record.data.C[k] @ target.zero_context
    cap = None if model.count_only else record.capacities(k)
    return coarse_logpmf(
        record.data.X[k][:, ell],
        record.data.Y[k][:, ell],
        base[:, 0],
        base[:, 1],
        zero,
        target.dispersion,
        cap,
        model.count_only,
    )


@dataclass
class Blocks:
    indices: np.ndarray
    parents: np.ndarray
    x: np.ndarray
    y: np.ndarray
    capacity: np.ndarray


def blocks(record, model, k, ell):
    """Only complete, commonly eligible blocks with a nonzero parent count."""
    d = record.data
    groups = [d.hierarchy.children(k + 1, c) for c in range(d.hierarchy.H[k + 1])]
    sizes = {len(g) for g in groups}
    if len(sizes) != 1 or not groups or not groups[0]:
        raise ValueError(
            "revision experiments require regular nonempty calendar blocks"
        )
    indices = np.asarray(groups, dtype=int)
    good = record.rows(k, model.lag_bound[k])[indices].all(axis=1)
    good &= d.get_observed(k + 2)
    good &= d.X[k + 1][:, ell] > 0
    parents = np.flatnonzero(good)
    idx = indices[good]
    return Blocks(
        idx, parents, d.X[k][idx, ell], d.Y[k][idx, ell], record.capacities(k)[idx]
    )


def stage_scores(record, model, k, ell, batch, base, stage, channel, override=None):
    """Use only endogenous values strictly preceding the current epoch."""
    target = model.levels[k].targets[ell]
    idx = batch.indices
    out = base[:, :, channel].copy()
    for parent, coef in target.parents.items():
        values = feature(record, k, parent, channel, override, idx)
        if parent.kind == "behavior":
            available = (idx - parent.lag >= 0) & (
                idx - parent.lag < idx[:, stage, None]
            )
            values = np.where(available, values, 0.0)
        out += coef[channel] * values
    prefix = np.cumsum(batch.x, axis=1) - batch.x
    if channel == 0:
        prefix = np.minimum(prefix, prefix[:, stage, None])
    out += target.depletion[channel] * np.log1p(prefix)
    if channel == 1:
        out = np.where(batch.x > 0, out, -np.inf)
    return out


def allocation_logpmf(
    record, model, k, ell, gamma, h, *, base=None, batch=None, override=None
):
    """Full normalized block masses, replaying observed prefixes."""
    batch = blocks(record, model, k, ell) if batch is None else batch
    B, m = batch.indices.shape
    if not B:
        return np.zeros(0)
    if base is None:
        base = scores(record, model, k, ell, gamma, h)[batch.indices]
    totals = record.data.X[k + 1][batch.parents, ell]
    durations = record.data.Y[k + 1][batch.parents, ell]
    result = np.zeros(B)
    for channel in range(1 if model.count_only else 2):
        values = batch.x if channel == 0 else batch.y - batch.x
        remaining = (totals if channel == 0 else durations - totals).copy()
        concentration = model.levels[k].concentration[channel]
        for i in range(m):
            valid = np.ones(B, dtype=bool) if channel == 0 else batch.x[:, i] > 0
            if not valid.any():
                continue
            logits = stage_scores(
                record, model, k, ell, batch, base, i, channel, override
            )
            weights = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
            a = concentration * weights[:, i]
            b = concentration * weights[:, i + 1 :].sum(axis=1)
            if channel == 0 and not model.count_only:
                lower = np.maximum(
                    0, remaining - batch.capacity[:, i + 1 :].sum(axis=1)
                )
                upper = np.minimum(batch.capacity[:, i], remaining)
            else:
                lower, upper = np.zeros(B, dtype=int), remaining
            if np.any(a[valid] <= 0):
                raise FloatingPointError("active allocation softmax weight underflowed")
            ll = split_logpmf(
                values[valid, i],
                remaining[valid],
                a[valid],
                b[valid],
                lower[valid],
                upper[valid],
            )
            result[valid] += ll
            remaining -= np.where(valid, values[:, i], 0)
        result = np.where(remaining == 0, result, -np.inf)
    return result


def target_score(record, model, k, ell, posterior, override=None):
    q = posterior[k]
    if model.emission(k):
        ll = np.zeros(len(q.gamma))
        for m in range(q.gamma.shape[1]):
            one = np.zeros_like(q.gamma)
            one[:, m] = 1
            term = emission_terms(record, model, k, ell, one, q.m_H, override)
            ll += q.gamma[:, m] * term
        return float(ll[record.rows(k, model.lag_bound[k])].sum())
    return float(
        allocation_logpmf(
            record, model, k, ell, q.gamma, q.m_H, override=override
        ).sum()
    )


def sample(model, template: Record, seed=0, history=None):
    """Sample a coherent record from the latent prior.

    Optional history supplies lagged observations only, not posterior latent
    initial conditions. Forecast experiments use forecast.predict instead.
    """
    rng = np.random.default_rng(seed)
    d = template.data
    X = [np.zeros_like(a, dtype=np.int64) for a in d.X]
    Y = [np.zeros_like(a, dtype=np.int64) for a in d.Y]
    out = Record(
        template.name,
        MultiScaleData(
            d.hierarchy,
            d.label_names,
            X,
            Y,
            d.C,
            calendar=d.calendar,
            observed=d.observed,
        ),
    )
    states, factors = [None] * model.K, [None] * model.K
    for k in range(model.K - 1, -1, -1):
        level = model.levels[k]
        parent = None
        if k < model.K - 1 and model.nested:
            parent = states[k + 1][d.hierarchy.parent_maps[k]]
        states[k] = RegimeChain(level.regime).sample_sequence(
            len(X[k]), d.C[k], parent, rng
        )
        factors[k] = LatentFactor(level.factor).sample_trajectory(
            len(X[k]), states[k], d.C[k], rng
        )
        gamma = np.eye(level.regime.M)[states[k]]
        if model.emission(k):
            for t in range(len(X[k])):
                for ell, target in enumerate(level.targets):
                    base = scores(
                        out, model, k, ell, gamma[t : t + 1], factors[k][t : t + 1]
                    )[0]
                    # scores() only uses the supplied latent rows; context/lag
                    # contributions use the original epoch below.
                    for p, coef in target.parents.items():
                        for channel in range(1 if model.count_only else 2):
                            lag = t - p.lag
                            if p.kind == "context":
                                value = d.C[k][lag, p.source] if lag >= 0 else 0.0
                            elif lag >= 0:
                                value = np.log1p(
                                    (X if channel == 0 else Y)[k][lag, p.source]
                                )
                            elif (
                                history is not None
                                and len(history.data.X[k]) + lag >= 0
                            ):
                                value = np.log1p(
                                    (
                                        history.data.X
                                        if channel == 0
                                        else history.data.Y
                                    )[k][lag, p.source]
                                )
                            else:
                                value = 0.0
                            base[channel] += coef[channel] * value
                    zero = (
                        target.zero_logit
                        + target.zero_regime[states[k][t]]
                        + d.C[k][t] @ target.zero_context
                    )
                    if level.factor.r:
                        zero += factors[k][t] @ target.zero_factor
                    mu = exp_mean(base[0])
                    x = (
                        0
                        if rng.random() < expit(zero)
                        else rng.negative_binomial(
                            target.dispersion[0],
                            target.dispersion[0] / (target.dispersion[0] + mu),
                        )
                    )
                    if not model.count_only:
                        x = min(x, out.capacities(k)[t])
                    X[k][t, ell] = x
                    if x and not model.count_only:
                        nu = exp_mean(base[1])
                        Y[k][t, ell] = x + rng.negative_binomial(
                            target.dispersion[1],
                            target.dispersion[1] / (target.dispersion[1] + nu),
                        )
        else:
            base_all = [
                scores(out, model, k, ell, gamma, factors[k]) for ell in range(model.L)
            ]
            caps = out.capacities(k)
            for c in range(d.hierarchy.H[k + 1]):
                idx = np.asarray(d.hierarchy.children(k + 1, c))
                # Complete all labels at an epoch before advancing time.
                for channel in range(1 if model.count_only else 2):
                    remaining = (
                        X[k + 1][c].copy()
                        if channel == 0
                        else Y[k + 1][c] - X[k + 1][c]
                    )
                    for i, t in enumerate(idx):
                        for ell in range(model.L):
                            if X[k + 1][c, ell] == 0 or (
                                channel == 1 and X[k][t, ell] == 0
                            ):
                                continue
                            batch = Blocks(
                                idx[None],
                                np.array([c]),
                                X[k][idx, ell][None],
                                Y[k][idx, ell][None],
                                caps[idx][None],
                            )
                            logits = stage_scores(
                                out,
                                model,
                                k,
                                ell,
                                batch,
                                base_all[ell][idx][None],
                                i,
                                channel,
                            )[0]
                            if history is not None:
                                # Only lag reads before this future window come
                                # from the supplied past, never held-out data.
                                for p, coef in (
                                    model.levels[k].targets[ell].parents.items()
                                ):
                                    if p.kind == "behavior":
                                        past = (
                                            history.data.X[k]
                                            if channel == 0
                                            else history.data.Y[k]
                                        )
                                        lagidx = idx - p.lag
                                        ok = (lagidx < 0) & (len(past) + lagidx >= 0)
                                        logits[ok] += coef[channel] * np.log1p(
                                            past[lagidx[ok], p.source]
                                        )
                            weights = np.exp(logits - logsumexp(logits))
                            conc = level.concentration[channel]
                            a, b = conc * weights[i], conc * weights[i + 1 :].sum()
                            n = int(remaining[ell])
                            lo = (
                                max(0, n - int(caps[idx[i + 1 :]].sum()))
                                if channel == 0 and not model.count_only
                                else 0
                            )
                            hi = (
                                min(n, int(caps[t]))
                                if channel == 0 and not model.count_only
                                else n
                            )
                            value = split_sample(n, a, b, lo, hi, rng)
                            if channel == 0:
                                X[k][t, ell] = value
                            else:
                                Y[k][t, ell] = X[k][t, ell] + value
                            remaining[ell] -= value
                    if np.any(remaining):
                        raise RuntimeError("allocation left an unassigned remainder")
    if model.nested:
        out.validate(model.count_only)
    return out, states, factors
