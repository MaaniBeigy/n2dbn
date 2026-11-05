"""Parameters, participant boundaries, and support for the revised model."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..core.variables import MultiScaleData
from ..model.params import FactorParams, RegimeParams


@dataclass(frozen=True, order=True)
class Parent:
    kind: str
    source: int
    lag: int

    def __post_init__(self):
        if self.kind not in ("behavior", "context"):
            raise ValueError("parent kind must be behavior or context")
        if self.source < 0 or self.lag < (1 if self.kind == "behavior" else 0):
            raise ValueError("behavioral parents require a positive lag")


@dataclass
class Target:
    # Two rows: count and duration surplus. No occupancy/risk-set term.
    intercept: np.ndarray
    regime: np.ndarray
    factor: np.ndarray
    parents: dict[Parent, np.ndarray] = field(default_factory=dict)
    dispersion: np.ndarray = field(default_factory=lambda: np.full(2, 10.0))
    depletion: np.ndarray = field(default_factory=lambda: np.zeros(2))
    zero_logit: float = -2.0
    zero_regime: np.ndarray = field(default_factory=lambda: np.zeros(1))
    zero_factor: np.ndarray = field(default_factory=lambda: np.zeros(0))
    zero_context: np.ndarray = field(default_factory=lambda: np.zeros(0))


@dataclass
class Level:
    targets: list[Target]
    regime: RegimeParams
    factor: FactorParams
    concentration: np.ndarray = field(default_factory=lambda: np.full(2, 30.0))


@dataclass
class Model:
    levels: list[Level]
    count_only: bool = False
    nested: bool = True
    offsets: dict[str, np.ndarray] = field(default_factory=dict)
    lag_bound: tuple[int, ...] = ()
    penalties: tuple[float, ...] = ()
    coefficient_bound: float = 3.0

    def clone(self):
        return copy.deepcopy(self)

    @property
    def K(self):
        return len(self.levels)

    @property
    def L(self):
        return len(self.levels[0].targets)

    def edges(self):
        return [
            (k, ell, parent)
            for k, level in enumerate(self.levels)
            for ell, target in enumerate(level.targets)
            for parent in sorted(target.parents)
        ]

    def penalty(self):
        return sum(self.penalties[k] for k, _, _ in self.edges())

    def emission(self, k):
        return k == self.K - 1 or not self.nested


@dataclass
class Record:
    name: str
    data: MultiScaleData
    _capacity: dict = field(default_factory=dict, repr=False)

    def rows(self, k, lag):
        """One fixed mask for all candidate sets, never bridge a missing lag."""
        observed = self.data.get_observed(k + 1)
        out = observed.copy()
        for d in range(1, lag + 1):
            out[:d] = False
            out[d:] &= observed[:-d]
        return out

    def capacities(self, k):
        if k in self._capacity:
            return self._capacity[k]
        counts = np.ones(self.data.hierarchy.H[0], dtype=np.int64)
        for j in range(k):
            out = np.zeros(self.data.hierarchy.H[j + 1], dtype=np.int64)
            np.add.at(out, self.data.hierarchy.parent_maps[j], counts)
            counts = out
        self._capacity[k] = counts
        return counts

    def validate(self, count_only=False):
        d = self.data
        if not d.L or not d.K:
            raise ValueError("empty label axis or hierarchy")
        for k in range(d.K):
            x, y, c = d.X[k], d.Y[k], d.C[k]
            expected = (d.hierarchy.H[k], d.L)
            if x.shape != expected or y.shape != expected or c.shape[0] != expected[0]:
                raise ValueError(f"{self.name}: inconsistent shapes at scale {k + 1}")
            if not np.isfinite(c).all():
                raise ValueError(f"{self.name}: non-finite context at scale {k + 1}")
            for name, a in (("count", x), ("duration", y)):
                if (
                    not np.isfinite(a).all()
                    or np.any(a < 0)
                    or np.any(a != np.floor(a))
                ):
                    raise ValueError(
                        f"{self.name}: {name} must be nonnegative integers"
                    )
            seen = d.get_observed(k + 1)
            if not count_only:
                bad = seen[:, None] & (
                    (x > self.capacities(k)[:, None]) | (y < x) | ((x == 0) & (y != 0))
                )
                if bad.any():
                    t, ell = np.argwhere(bad)[0]
                    raise ValueError(
                        f"{self.name}: paper support violation at scale {k + 1}, "
                        f"epoch {t}, label {d.label_names[ell]}: "
                        f"X={x[t, ell]}, Y={y[t, ell]}. "
                        "Audit simultaneous starts and duration recoding; "
                        "do not clip, jitter, or silently relax binary support."
                    )
        errors = d.check_reconciliation()
        if any(value != 0 for value in errors.values()):
            raise ValueError(f"{self.name}: inconsistent aggregates: {errors}")


def initialize(
    records: Sequence[Record],
    regimes: Sequence[int],
    factors: Sequence[int],
    lag_bound: Sequence[int],
    *,
    count_only=False,
    nested=True,
    penalty=1.0,
    seed=0,
):
    """Initialize from observed moments only, never generating parameters."""
    if not records or len({r.name for r in records}) != len(records):
        raise ValueError("a cohort requires distinct participant identifiers")
    first = records[0].data
    K, L = first.K, first.L
    if any(len(v) != K for v in (regimes, factors, lag_bound)):
        raise ValueError("one latent order and lag bound is required per scale")
    if any(m < 1 for m in regimes) or any(r < 0 for r in factors):
        raise ValueError("invalid latent order")
    rng = np.random.default_rng(seed)
    for record in records:
        record.validate(count_only)
        if record.data.label_names != first.label_names or record.data.K != K:
            raise ValueError("cohort label axes and numbers of scales must agree")
        if any(a.shape[1] != b.shape[1] for a, b in zip(record.data.C, first.C)):
            raise ValueError("cohort context axes must agree at every scale")
    levels, penalties = [], []
    for k, (M, r) in enumerate(zip(regimes, factors)):
        dc = first.C[k].shape[1]
        parent_M = regimes[k + 1] if k < K - 1 and nested else 0
        transition = np.full((M, M), 0.1 / M)
        transition += np.eye(M) * 0.9
        rp = RegimeParams(
            M,
            parent_M,
            dc,
            np.zeros(M),
            np.zeros(M),
            np.log(transition),
            np.eye(M, parent_M) * 0.2 if parent_M else None,
            np.zeros((M, dc)),
        )
        fp = FactorParams(
            M,
            r,
            dc,
            np.tile(np.eye(r) * 0.7, (M, 1, 1)),
            np.zeros((M, r, dc)),
            np.tile(np.eye(r) * 0.1, (M, 1, 1)),
            np.zeros(r),
            np.eye(r),
        )
        valid = [rec.rows(k, lag_bound[k]) for rec in records]
        if not any(row.any() for row in valid):
            raise ValueError(f"no eligible observations at scale {k + 1}")
        x = np.concatenate([rec.data.X[k][row] for rec, row in zip(records, valid)])
        y = np.concatenate([rec.data.Y[k][row] for rec, row in zip(records, valid)])
        targets = []
        for ell in range(L):
            active = x[:, ell] > 0
            surplus = y[active, ell] - x[active, ell]
            mean = [
                max(x[:, ell].mean(), 0.01),
                max(surplus.mean(), 0.01) if active.any() and not count_only else 0.01,
            ]
            intercept = np.log(mean) if k == K - 1 or not nested else np.zeros(2)
            effects = rng.normal(0, 0.1, (2, M))
            effects -= effects.mean(axis=1, keepdims=True)
            target = Target(intercept, effects, rng.normal(0, 0.05, (2, r)))
            target.zero_regime = np.zeros(M)
            target.zero_factor = np.zeros(r)
            target.zero_context = np.zeros(dc)
            targets.append(target)
        levels.append(Level(targets, rp, fp))
        penalties.append(float(penalty) * np.log(max(len(x), 2)))
    model = Model(
        levels,
        count_only,
        nested,
        {rec.name: np.zeros((2, L)) for rec in records},
        tuple(lag_bound),
        tuple(penalties),
    )
    return model
