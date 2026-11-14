"""A single normalized Dirichlet-multinomial law for likelihood and draws.

Per parent block, the allocation weights are the softmax of the child scores;
psi ~ Dir(kappa * omega) and the block total ~ Mult(psi). Endogenous lag reads
inside the current block use the zero reference in both the likelihood and the
sampler, so the two evaluate the same distribution. Durations allocate the
surplus Y - X over the active children with concentration tau. Draws above a
child capacity are repaired by redistribution, exactly as the accepted sampler.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit, gammaln, logsumexp, xlogy
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


@dataclass
class Prepared:
    """Cached block partition for one (record, scale, target); data-only."""

    record: object
    k: int
    ell: int
    lag_bound: int
    count_only: bool
    batch: Blocks

    def __getstate__(self):
        # Never serialize the record across a worker pipe; the parent rebuilds.
        state = dict(self.__dict__)
        state["record"] = None
        return state


def prepare(record, model, k, ell, batch=None):
    """Return the cached block partition, rebuilding only when inputs change."""
    key = (k, ell)
    cached = record._prepared.get(key) if record._cache_prepared else None
    if (
        cached is not None
        and cached.lag_bound == model.lag_bound[k]
        and cached.count_only == model.count_only
        and (batch is None or cached.batch is batch)
    ):
        return cached
    if batch is None:
        batch = blocks(record, model, k, ell)
    prepared = Prepared(record, k, ell, model.lag_bound[k], model.count_only, batch)
    if record._cache_prepared and (batch is None or cached is None or cached.batch is not batch):
        record._prepared[key] = prepared
    return prepared


def _channel_omega(
    prepared, model, k, ell, channel, base, override=None, active=None, history=None
):
    """Softmax allocation weights over the block children, shape (B, m).

    A behavior lag read is available only from a completed earlier block
    (idx - lag < block start); reads inside the current block use the zero
    reference in both the sampler and the likelihood, so the two agree.
    `active` restricts the duration channel to children with positive counts.
    """
    target = model.levels[k].targets[ell]
    idx = prepared.batch.indices
    eta = base[:, :, channel].copy()
    block_start = idx[:, :1]
    for p in target.parents:
        values = feature(prepared.record, k, p, channel, override, idx)
        if p.kind == "behavior":
            lagged = idx - p.lag
            avail = (lagged >= 0) & (lagged < block_start)
            values = np.where(avail, values, 0.0)
            if history is not None:
                past = (history.data.X if channel == 0 else history.data.Y)[k]
                back = (lagged < 0) & (len(past) + lagged >= 0)
                values = np.where(
                    back, np.log1p(past[len(past) + lagged[back], p.source]), values
                )
        eta += target.parents[p][channel] * values
    if active is not None:
        eta = np.where(active, eta, -np.inf)
    return np.exp(eta - logsumexp(eta, axis=1, keepdims=True))


def _dm_channel_logpmf(x, omega, conc, active=None):
    """Dirichlet-multinomial log-pmf of one channel, one value per block.

    Includes the multinomial coefficient and the block constant
    log Gamma(conc) - log Gamma(N + conc), so the masses over all allocations
    of a fixed block total sum to one (the sampler-likelihood identity).
    """
    a = np.maximum(conc * omega, 1e-10)
    xs = np.where(active, x, 0) if active is not None else x
    if active is not None:
        term = np.where(active, gammaln(xs + a) - gammaln(a), 0.0)
    else:
        term = gammaln(xs + a) - gammaln(a)
    n = xs.sum(axis=1)
    return (
        term.sum(axis=1)
        + gammaln(conc)
        - gammaln(n + conc)
        + gammaln(n + 1)
        - gammaln(xs + 1).sum(axis=1)
    )


def _allocation_from_omega(prepared, omega, concentration):
    """Per-block DM log-pmf from precomputed channel weights."""
    batch = prepared.batch
    result = _dm_channel_logpmf(batch.x, omega[0], concentration[0])
    if not prepared.count_only:
        active = batch.x > 0
        result = result + _dm_channel_logpmf(
            batch.y - batch.x, omega[1], concentration[1], active=active
        )
    return result


def allocation_logpmf(
    record, model, k, ell, gamma, h, *, base=None, batch=None, override=None
):
    """Per-block Dirichlet-multinomial log-pmf of the observed allocation.

    Per parent block, psi ~ Dir(kappa * omega) and the children draw
    Mult(X_c, psi), with omega the softmax of the child scores under the
    zero-reference convention for endogenous lag reads inside the current
    block; the sampler draws exactly this law. Durations allocate the
    surplus Y - X the same way over the active children with concentration
    tau.
    """
    prepared = prepare(record, model, k, ell, batch=batch)
    batch = prepared.batch
    B, m = batch.indices.shape
    if not B:
        return np.zeros(0)
    idx = batch.indices
    if base is None:
        base = scores(record, model, k, ell, gamma, h)[idx]
    level = model.levels[k]
    omega = [_channel_omega(prepared, model, k, ell, 0, base, override)]
    if not model.count_only:
        omega.append(
            _channel_omega(
                prepared, model, k, ell, 1, base, override, active=batch.x > 0
            )
        )
    return _allocation_from_omega(prepared, omega, level.concentration)


@dataclass
class Frozen:
    """Softmax allocation weights frozen across a concentration-only refit."""

    prepared: Prepared
    omega: list

    def light(self):
        """The picklable payload: weight arrays only, no record."""
        return self.omega

    @classmethod
    def assemble(cls, prepared, light):
        return cls(prepared, light)

    def logpmf(self, concentration):
        if not self.omega:
            return np.zeros(len(self.prepared.batch.parents))
        return _allocation_from_omega(self.prepared, self.omega, concentration)


def prepare_frozen(record, model, k, ell, gamma, h, override=None):
    prepared = prepare(record, model, k, ell)
    batch = prepared.batch
    if batch.indices.shape[0] == 0:
        return Frozen(prepared, [])
    base = scores(record, model, k, ell, gamma, h)[batch.indices]
    omega = [_channel_omega(prepared, model, k, ell, 0, base, override)]
    if not model.count_only:
        omega.append(
            _channel_omega(prepared, model, k, ell, 1, base, override, active=batch.x > 0)
        )
    return Frozen(prepared, omega)


def stack_prepared(parts):
    """Stack per-record block partitions for one (k, ell) into one batch.

    Every DM quantity is row-wise independent, so stacked rows behave exactly
    like the separate per-record evaluations.
    """
    first = parts[0]
    batch = Blocks(
        np.vstack([p.batch.indices for p in parts]),
        np.concatenate([p.batch.parents for p in parts]),
        np.vstack([p.batch.x for p in parts]),
        np.vstack([p.batch.y for p in parts]),
        np.vstack([p.batch.capacity for p in parts]),
    )
    return Prepared(None, first.k, first.ell, first.lag_bound, first.count_only, batch)


def cohort_prepare(records, model, k, ell):
    parts = [prepare(rec, model, k, ell) for rec in records]
    return stack_prepared(parts), parts


def cohort_target_score(stacked, parts, records, model, k, ell, posteriors, override=None):
    """One DM replay over the stacked cohort blocks.

    Per-record block masses are summed slice by slice in record order, so the
    value is identical to looping target_score over the records.
    """
    idx = stacked.batch.indices
    if idx.shape[0] == 0:
        return 0.0
    base = np.vstack(
        [
            scores(
                rec, model, k, ell,
                posteriors[rec.name][k].gamma, posteriors[rec.name][k].m_H,
            )[p.batch.indices]
            for rec, p in zip(records, parts)
        ]
    )
    target = model.levels[k].targets[ell]
    block_start = idx[:, :1]
    omega = []
    for channel in range(1 if model.count_only else 2):
        eta = base[:, :, channel].copy()
        for p in target.parents:
            values = np.vstack(
                [
                    feature(
                        rec, k, p, channel,
                        None if override is None else override.get(rec.name),
                        pp.batch.indices,
                    )
                    for rec, pp in zip(records, parts)
                ]
            )
            if p.kind == "behavior":
                avail = (idx - p.lag >= 0) & (idx - p.lag < block_start)
                values = np.where(avail, values, 0.0)
            eta += target.parents[p][channel] * values
        if channel == 1 and not model.count_only:
            eta = np.where(stacked.batch.x > 0, eta, -np.inf)
        omega.append(np.exp(eta - logsumexp(eta, axis=1, keepdims=True)))
    level = model.levels[k]
    masses = _allocation_from_omega(stacked, omega, level.concentration)
    total = 0.0
    at = 0
    for p in parts:
        b = p.batch.indices.shape[0]
        total += float(masses[at : at + b].sum())
        at += b
    return total


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
        allocation_logpmf(record, model, k, ell, q.gamma, q.m_H, override=override).sum()
    )


def _redistribute_excess(x, caps, weights, rng):
    """Move counts above a per-child capacity onto children with room.

    Mirrors the accepted sampler's repair: the total is preserved and each
    excess unit goes to a child with free capacity drawn with probability
    proportional to the allocation weights.
    """
    x = np.asarray(x, dtype=np.int64).copy()
    caps = np.asarray(caps, dtype=np.int64)
    if x.sum() > caps.sum():
        raise ValueError("block total exceeds the capacity of the block")
    excess = int(np.maximum(x - caps, 0).sum())
    if excess == 0:
        return x
    x = np.minimum(x, caps)
    w = np.asarray(weights, dtype=np.float64).copy()
    for _ in range(excess):
        free = caps - x > 0
        p = np.where(free, w, 0.0)
        total = p.sum()
        if total <= 0:
            p = free.astype(np.float64)
            total = p.sum()
        x[rng.choice(len(x), p=p / total)] += 1
    return x


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
                for ell in range(model.L):
                    n = int(X[k + 1][c, ell])
                    if n == 0:
                        continue
                    prepared = Prepared(
                        out,
                        k,
                        ell,
                        model.lag_bound[k],
                        model.count_only,
                        Blocks(
                            idx[None],
                            np.array([c]),
                            X[k][idx, ell][None],
                            Y[k][idx, ell][None],
                            caps[idx][None],
                        ),
                    )
                    base = base_all[ell][idx][None]
                    # The joint DM draw: psi ~ Dir(kappa * omega), then
                    # Mult(n, psi). Within-block lag reads are still unwritten,
                    # so the sampler sees the same zero reference as the likelihood.
                    omega = _channel_omega(
                        prepared, model, k, ell, 0, base, history=history
                    )[0]
                    psi = rng.dirichlet(np.maximum(level.concentration[0] * omega, 1e-10))
                    alloc = rng.multinomial(n, psi).astype(np.int64)
                    alloc = _redistribute_excess(alloc, caps[idx], omega, rng)
                    X[k][idx, ell] = alloc
                    if model.count_only:
                        continue
                    # Durations: the surplus Y - X goes to the active children,
                    # each keeping its own count as the floor.
                    y_parent = int(Y[k + 1][c, ell])
                    active = alloc > 0
                    free = y_parent - int(alloc.sum())
                    if free < 0:
                        raise RuntimeError(
                            "parent duration is below its own count"
                        )
                    # Every active child books its own count as the floor.
                    Y[k][idx, ell] = alloc
                    if free == 0 or not active.any():
                        continue
                    omega_d = _channel_omega(
                        prepared, model, k, ell, 1, base, active=active, history=history
                    )[0]
                    act = np.flatnonzero(active)
                    psi_d = rng.dirichlet(
                        np.maximum(level.concentration[1] * omega_d[act], 1e-10)
                    )
                    surplus = np.zeros(len(idx), dtype=np.int64)
                    surplus[act] = rng.multinomial(free, psi_d)
                    Y[k][idx, ell] = alloc + surplus

    if model.nested:
        out.validate(model.count_only)
    return out, states, factors
