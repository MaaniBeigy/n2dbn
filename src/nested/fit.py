"""Pooled approximate structural learning on the normalized nested model."""

from __future__ import annotations

import copy
import itertools
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .inference import infer, monitor, transition_inputs
from .law import feature, target_score
from .model import Parent


@dataclass
class FitConfig:
    outer: int = 5
    inner: int = 3
    profile: int = 25
    indegree: int = 2
    search: str = "greedy"
    shifts: int = 19
    shift_quantile: float = 0.95
    seed: int = 0
    tolerance: float = 1e-5
    max_seconds: float = 0
    lag_radius: float | None = None
    context_mode: str = "home"
    structure: bool = True
    subject_offsets: bool = True
    depletion: bool = True

    def __post_init__(self):
        if (
            min(self.outer, self.inner, self.profile) < 1
            or self.indegree < 0
            or self.shifts < 0
        ):
            raise ValueError("invalid iteration, degree, or shift budget")
        if self.search not in ("greedy", "exhaustive"):
            raise ValueError("search must be greedy or exhaustive")
        if self.context_mode not in ("home", "all", "none"):
            raise ValueError("context mode must be home, all, or none")
        if not 0 < self.shift_quantile <= 1:
            raise ValueError("invalid shift quantile")
        if self.lag_radius is not None and not 0 < self.lag_radius < 1:
            raise ValueError("lag radius must lie strictly between zero and one")


@dataclass
class FitResult:
    model: object
    posteriors: dict
    trace: list[dict] = field(default_factory=list)
    evaluations: int = 0
    elapsed_seconds: float = 0
    status: str = "complete"
    shift_checks: list[dict] = field(default_factory=list)
    inference_seconds: float = 0.0
    search_seconds: float = 0.0
    candidate_refits: int = 0


class Fitter:
    def __init__(self, config=None):
        self.config = config or FitConfig()
        self.evaluations = 0
        self.started = 0.0
        self.shift_checks = []
        self.candidate_refits = 0
        self._candidate_cache = {}

    def budget(self):
        if (
            self.config.max_seconds
            and time.perf_counter() - self.started > self.config.max_seconds
        ):
            raise TimeoutError("declared fitting time budget exhausted")

    def value(self, records, model, k, ell, posterior, overrides=None):
        self.budget()
        self.evaluations += 1
        return sum(
            target_score(
                rec,
                model,
                k,
                ell,
                posterior[rec.name],
                None if overrides is None else overrides.get(rec.name),
            )
            for rec in records
        )

    def profile_target(
        self, records, model, k, ell, posterior, parents, *, base=False, overrides=None
    ):
        original = copy.deepcopy(model.levels[k].targets[ell])
        target = copy.deepcopy(original)
        target.parents = {
            p: original.parents.get(p, np.zeros(2)).copy() for p in sorted(parents)
        }
        model.levels[k].targets[ell] = target
        entries, bounds = [], []

        def add(attr, bound, transform=False):
            value = np.asarray(getattr(target, attr))
            entries.append((attr, value.shape, value.size, transform))
            bounds.extend([bound] * value.size)
            return (np.log(value) if transform else value).ravel()

        pieces = []
        if base:
            if model.emission(k):
                pieces += [
                    add("intercept", (-20, 25)),
                    add("dispersion", (-4, 9), True),
                ]
                if model.count_only:
                    # Duration entries are kept fixed by unpack() below.
                    pass
                pieces += [
                    add("zero_logit", (-12, 8)),
                    add("zero_regime", (-5, 5)),
                    add("zero_factor", (-5, 5)),
                    add("zero_context", (-5, 5)),
                ]
            pieces += [add("regime", (-5, 5)), add("factor", (-5, 5))]
            if not model.emission(k) and self.config.depletion:
                pieces += [add("depletion", (-3, 3))]
        parent_order = sorted(target.parents)
        channels = 1 if model.count_only else 2
        pieces += [target.parents[p][:channels].copy() for p in parent_order]
        bounds += [(-model.coefficient_bound, model.coefficient_bound)] * (
            len(parent_order) * channels
        )
        if not pieces or not sum(a.size for a in pieces):
            value = self.value(records, model, k, ell, posterior, overrides)
            self.candidate_refits += 1
            return copy.deepcopy(target), value
        x0 = np.concatenate(pieces)

        def unpack(vector):
            at = 0
            for name, shape, size, transform in entries:
                values = vector[at : at + size].reshape(shape)
                setattr(target, name, np.exp(values) if transform else values.copy())
                at += size
            for p in parent_order:
                target.parents[p] = np.zeros(2)
                target.parents[p][:channels] = vector[at : at + channels]
                at += channels
            if self.config.lag_radius is not None:
                selected = [p for p in parent_order if p.kind == "behavior"]
                norm = sum(abs(target.parents[p][0]) for p in selected)
                if norm > self.config.lag_radius:
                    for p in selected:
                        target.parents[p][0] *= self.config.lag_radius / norm
            # A count-only fit never scores or changes duration parameters.
            if model.count_only:
                for name in (
                    "intercept",
                    "regime",
                    "factor",
                    "dispersion",
                    "depletion",
                ):
                    a = getattr(target, name)
                    a[1] = getattr(original, name)[1]

        best_x, best_value = x0.copy(), -np.inf

        def objective(vector):
            nonlocal best_x, best_value
            unpack(vector)
            value = self.value(records, model, k, ell, posterior, overrides)
            if np.isfinite(value) and value > best_value:
                best_value, best_x = value, vector.copy()
            return -value if np.isfinite(value) else 1e100

        objective(x0)
        minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.config.profile, "ftol": 1e-9},
        )
        unpack(best_x)
        if not np.isfinite(best_value):
            raise FloatingPointError("no finite coefficient profile was attained")
        self.candidate_refits += 1
        return copy.deepcopy(target), float(best_value)

    def candidates(self, records, model, k):
        if k in self._candidate_cache:
            return self._candidate_cache[k]
        candidates = [
            Parent("behavior", a, lag)
            for a in range(model.L)
            for lag in range(1, model.lag_bound[k] + 1)
        ]
        if self.config.context_mode == "none":
            return candidates
        width = records[0].data.C[k].shape[1]
        matching_columns = len({c.shape[1] for c in records[0].data.C}) == 1
        for column in range(width):
            for lag in range(model.lag_bound[k] + 1):
                parent = Parent("context", column, lag)
                identified = []
                for j in range(model.K):
                    if (not matching_columns and j != k) or lag > model.lag_bound[j]:
                        continue
                    varies = False
                    for rec in records:
                        c = feature(rec, j, parent, 0)
                        valid = rec.rows(j, model.lag_bound[j])
                        if model.emission(j):
                            seen = c[valid]
                            varies |= len(seen) > 0 and np.ptp(seen) > 1e-10
                        else:
                            groups = np.asarray(
                                [
                                    rec.data.hierarchy.children(j + 1, i)
                                    for i in range(rec.data.hierarchy.H[j + 1])
                                ]
                            )
                            usable = valid[groups].all(axis=1) & rec.data.get_observed(
                                j + 2
                            )
                            varies |= np.any(np.ptp(c[groups[usable]], axis=1) > 1e-10)
                    if varies:
                        identified.append(j)
                if k in identified and (
                    self.config.context_mode == "all" or k == min(identified)
                ):
                    candidates.append(parent)
        self._candidate_cache[k] = candidates
        return candidates

    def search_target(self, records, model, k, ell, posterior):
        start = copy.deepcopy(model.levels[k].targets[ell])
        pool = sorted(set(self.candidates(records, model, k)) | set(start.parents))
        cache = {}

        def evaluate(parents):
            key = tuple(sorted(parents))
            if key not in cache:
                model.levels[k].targets[ell] = copy.deepcopy(start)
                target, value = self.profile_target(
                    records, model, k, ell, posterior, key
                )
                cache[key] = target, value - model.penalties[k] * len(key)
            return cache[key]

        retained = tuple(sorted(start.parents))
        evaluate(retained)
        if self.config.search == "exhaustive":
            choices = itertools.chain.from_iterable(
                itertools.combinations(pool, n)
                for n in range(min(self.config.indegree, len(pool)) + 1)
            )
            for choice in choices:
                evaluate(choice)
        else:
            for _ in range(2 * max(self.config.indegree, 1) + 1):
                neighbors = {tuple(p for p in retained if p != old) for old in retained}
                if len(retained) < self.config.indegree:
                    neighbors |= {
                        tuple(sorted((*retained, p))) for p in pool if p not in retained
                    }
                for old in retained:
                    neighbors |= {
                        tuple(sorted((set(retained) - {old}) | {p}))
                        for p in pool
                        if p not in retained
                    }
                for choice in sorted(neighbors):
                    evaluate(choice)
                best = max(cache, key=lambda key: cache[key][1])
                if cache[best][1] <= cache[retained][1] + self.config.tolerance:
                    break
                retained = best
        best = max(cache, key=lambda key: cache[key][1])
        target = copy.deepcopy(cache[best][0])
        model.levels[k].targets[ell] = target
        if self.config.shifts and target.parents:
            target = self.confirm(records, model, k, ell, posterior)
        model.levels[k].targets[ell] = target

    def confirm(self, records, model, k, ell, posterior):
        rng = np.random.default_rng(self.config.seed + 101 * k + ell)
        while model.levels[k].targets[ell].parents:
            full = copy.deepcopy(model.levels[k].targets[ell])
            full_value = self.value(records, model, k, ell, posterior)
            failures = []
            for parent in sorted(full.parents):
                model.levels[k].targets[ell] = copy.deepcopy(full)
                reduced, reduced_value = self.profile_target(
                    records, model, k, ell, posterior, set(full.parents) - {parent}
                )
                gains = []
                for _ in range(self.config.shifts):
                    overrides = {}
                    for rec in records:
                        cols = [
                            feature(rec, k, parent, channel) for channel in range(2)
                        ]
                        if model.emission(k):
                            indices = np.flatnonzero(rec.rows(k, model.lag_bound[k]))
                            for idx in np.split(
                                indices, np.flatnonzero(np.diff(indices) != 1) + 1
                            ):
                                if len(idx) > 1:
                                    shift = int(rng.integers(1, len(idx)))
                                    for v in cols:
                                        v[idx] = np.roll(v[idx], shift)
                        else:
                            for c in range(rec.data.hierarchy.H[k + 1]):
                                idx = rec.data.hierarchy.children(k + 1, c)
                                if len(idx) > 1:
                                    shift = int(rng.integers(1, len(idx)))
                                    for v in cols:
                                        v[idx] = np.roll(v[idx], shift)
                        overrides[rec.name] = {parent: cols}
                    model.levels[k].targets[ell] = copy.deepcopy(reduced)
                    _, value = self.profile_target(
                        records,
                        model,
                        k,
                        ell,
                        posterior,
                        full.parents,
                        overrides=overrides,
                    )
                    gains.append(value - reduced_value)
                threshold = float(np.quantile(gains, self.config.shift_quantile))
                gain = full_value - reduced_value
                self.shift_checks.append(
                    {
                        "scale": k + 1,
                        "target": ell,
                        "parent": vars(parent),
                        "gain": gain,
                        "null_threshold": threshold,
                        "null_gains": gains,
                        "passed": bool(gain > threshold),
                    }
                )
                if gain <= threshold:
                    failures.append((gain - threshold, reduced))
            model.levels[k].targets[ell] = full
            if not failures:
                break
            model.levels[k].targets[ell] = min(failures, key=lambda entry: entry[0])[1]
        return copy.deepcopy(model.levels[k].targets[ell])

    def update_latents(self, records, model, k, posterior):
        rp = model.levels[k].regime
        attrs = ["init", "trans", "ctx"] + (["cross"] if rp.cross is not None else [])
        shapes = [getattr(rp, name).shape for name in attrs]
        x0 = np.concatenate([getattr(rp, name).ravel() for name in attrs])

        def unpack(x):
            pos = 0
            for name, shape in zip(attrs, shapes):
                size = int(np.prod(shape))
                setattr(rp, name, x[pos : pos + size].reshape(shape).copy())
                pos += size

        best, best_score = x0.copy(), np.inf

        def objective(x):
            nonlocal best, best_score
            self.budget()
            unpack(x)
            total = 0.0
            for rec in records:
                initial, transition = transition_inputs(
                    rec, model, k, posterior[rec.name]
                )
                q = posterior[rec.name][k]
                total += q.gamma[0] @ initial
                if len(q.gamma) > 1:
                    total += np.sum(q.xi.transpose(0, 2, 1) * transition)
            value = -float(total)
            if value < best_score:
                best, best_score = x.copy(), value
            return value

        objective(x0)
        if rp.M > 1:
            minimize(
                objective,
                x0,
                method="L-BFGS-B",
                bounds=[(-8, 8)] * len(x0),
                options={"maxiter": self.config.profile},
            )
        unpack(best)

    def update_offsets(self, records, model, posterior):
        if not self.config.subject_offsets or len(records) < 2:
            return
        k = model.K - 1
        for rec in records:
            for ell in range(model.L):
                original = model.offsets[rec.name][:, ell].copy()
                channels = 1 if model.count_only else 2
                best, best_value = original[:channels].copy(), -np.inf

                def objective(x):
                    nonlocal best, best_value
                    model.offsets[rec.name][:channels, ell] = x
                    value = self.value([rec], model, k, ell, posterior)
                    if value > best_value:
                        best, best_value = x.copy(), value
                    return -value

                objective(original[:channels])
                minimize(
                    objective,
                    original[:channels],
                    method="L-BFGS-B",
                    bounds=[(-10, 10)] * channels,
                    options={"maxiter": self.config.profile},
                )
                model.offsets[rec.name][:channels, ell] = best
        mean = np.mean(list(model.offsets.values()), axis=0)
        for offset in model.offsets.values():
            offset -= mean
        for ell, target in enumerate(model.levels[k].targets):
            target.intercept += mean[:, ell]

    def update_factors(self, records, model, k, posterior):
        fp = model.levels[k].factor
        r, dc = fp.r, fp.d_ctx
        if not r:
            return
        for state in range(fp.M):
            zz, hz, hh, weight = (
                np.zeros((r + dc, r + dc)),
                np.zeros((r, r + dc)),
                np.zeros((r, r)),
                0.0,
            )
            for rec in records:
                q = posterior[rec.name][k]
                for t in range(1, len(q.gamma)):
                    self.budget()
                    w = q.gamma[t, state]
                    z = np.r_[q.m_H[t - 1], rec.data.C[k][t]]
                    ezz = np.outer(z, z)
                    ezz[:r, :r] += q.Sigma_H[t - 1]
                    ehz = np.outer(q.m_H[t], z)
                    ehz[:, :r] += q.lag_H[t - 1].T
                    zz += w * ezz
                    hz += w * ehz
                    hh += w * (q.Sigma_H[t] + np.outer(q.m_H[t], q.m_H[t]))
                    weight += w
            if weight <= 1e-8:
                continue
            coefficient = np.linalg.solve(zz + np.eye(r + dc) * 1e-8, hz.T).T
            covariance = (
                hh
                - coefficient @ hz.T
                - hz @ coefficient.T
                + coefficient @ zz @ coefficient.T
            ) / weight
            eigen, vectors = np.linalg.eigh((covariance + covariance.T) / 2)
            fp.F[state], fp.G[state] = coefficient[:, :r], coefficient[:, r:]
            fp.Sigma[state] = (vectors * np.maximum(eigen, 1e-5)) @ vectors.T
        first = [posterior[rec.name][k] for rec in records]
        shifts = [
            np.array([fp.G[m] @ rec.data.C[k][0] for m in range(fp.M)])
            for rec in records
        ]
        fp.mu0 = np.mean(
            [q.m_H[0] - q.gamma[0] @ shift for q, shift in zip(first, shifts)], axis=0
        )
        fp.Sigma0 = np.mean(
            [
                q.Sigma_H[0]
                + sum(
                    q.gamma[0, m]
                    * np.outer(
                        q.m_H[0] - fp.mu0 - shift[m], q.m_H[0] - fp.mu0 - shift[m]
                    )
                    for m in range(fp.M)
                )
                for q, shift in zip(first, shifts)
            ],
            axis=0,
        )

    def fit(self, records, model):
        self.started = time.perf_counter()
        self._candidate_cache = {}
        self.evaluations = 0
        self.shift_checks = []
        self.candidate_refits = 0
        inference_start = time.perf_counter()
        model = model.clone()
        if not self.config.depletion:
            for level in model.levels:
                for target in level.targets:
                    target.depletion[:] = 0
        posterior = {rec.name: infer(rec, model, self.config.inner) for rec in records}
        retained = monitor(records, model, posterior)
        result = FitResult(
            model,
            posterior,
            [
                {
                    "iteration": 0,
                    "proposed": retained,
                    "retained": retained,
                    "accepted": True,
                }
            ],
        )
        result.inference_seconds = time.perf_counter() - inference_start
        for iteration in range(1, self.config.outer + 1):
            proposal = model.clone()
            try:
                for k in range(model.K - 1, -1, -1):
                    self.update_latents(records, proposal, k, posterior)
                    self.update_factors(records, proposal, k, posterior)
                    for ell in range(model.L):
                        self.profile_target(
                            records,
                            proposal,
                            k,
                            ell,
                            posterior,
                            proposal.levels[k].targets[ell].parents,
                            base=True,
                        )
                        if self.config.structure:
                            search_start = time.perf_counter()
                            self.search_target(records, proposal, k, ell, posterior)
                            result.search_seconds += time.perf_counter() - search_start
                    if not proposal.emission(k):
                        # Concentrations are shared across targets and refitted
                        # with all normalization terms retained.
                        level = proposal.levels[k]
                        old = np.log(level.concentration)

                        def objective(v):
                            level.concentration = np.exp(v)
                            return -sum(
                                self.value(records, proposal, k, ell, posterior)
                                for ell in range(model.L)
                            )

                        before = objective(old)
                        opt = minimize(
                            objective,
                            old,
                            bounds=[(-3, 10)] * 2,
                            method="L-BFGS-B",
                            options={"maxiter": self.config.profile},
                        )
                        objective(
                            opt.x if np.isfinite(opt.fun) and opt.fun <= before else old
                        )
                self.update_offsets(records, proposal, posterior)
                # No cached pre-M/S-step evidence is used here.
                inference_start = time.perf_counter()
                proposed_q = {
                    rec.name: infer(rec, proposal, self.config.inner) for rec in records
                }
                value = monitor(records, proposal, proposed_q)
                result.inference_seconds += time.perf_counter() - inference_start
                accepted = bool(np.isfinite(value) and value >= retained)
                improvement = value - retained
                if accepted:
                    model, posterior, retained = proposal, proposed_q, value
                result.trace.append(
                    {
                        "iteration": iteration,
                        "proposed": value,
                        "retained": retained,
                        "accepted": accepted,
                    }
                )
                if abs(improvement) <= self.config.tolerance * (1 + abs(retained)):
                    break
            except TimeoutError:
                result.status = "time_budget"
                break
        result.model, result.posteriors = model, posterior
        result.evaluations, result.elapsed_seconds = (
            self.evaluations,
            time.perf_counter() - self.started,
        )
        result.shift_checks = self.shift_checks
        result.candidate_refits = self.candidate_refits
        return result
