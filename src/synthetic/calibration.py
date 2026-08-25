"""
synthetic/calibration.py
========================
Choosing a generating hyperparameter from the screens rather than by hand.

Frozen.  This module belongs to the allocation formulation, in which a coarse
count is drawn and then split downward among its children.  It keeps working
and keeps reproducing the stored results, and nothing new is built on it.

The generator's job is to build a nested model whose structure and latent states
are strong enough that a correct learner recovers them, at every scale a study
claims to work at.  Where a scale does not deliver that, the fault is a
generating value, and the two screens already say which scale is failing: the
edge screen scores a true lagged edge against the floor a non-edge reaches, and
the latent screen scores each latent path against the floor a method with no
information reaches.

What was missing is the sweep around them.  Both screens were used to decide
which scales the grid should stop placing structure at, which answers a
different question from the one that matters, namely which generating value
would let a scale carry structure at all.  This runs one lever over a range and
reports both screens per scale for every value, so a setting is read off a
measurement.

One trap is built into the arithmetic and is reported rather than avoided.
Widening a block divides the epochs of every scale above it, so a sweep of
`branching` moves the identifiability of a coarse scale through two channels at
once, more children per block and fewer epochs to estimate from.  `hold_coarsest`
lengthens the series to keep the coarsest epoch count fixed, which isolates the
first channel; without it the epoch counts travel in the report.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from .scenarios import ScenarioConfig

# In-degree used while screening edges.  A scale carrying no true edge reports
# no true gain, so every scale is given one for the measurement even where the
# grid places none, which is how the edge screen has always been run.
SCREEN_INDEGREE = 1

# The same rule for the two latents, and it is the one the earlier screens
# missed.  A scale the grid has already reduced to one regime reports "not
# applicable" whatever the lever is set to, so sweeping a lever against it
# measures nothing.  The screen places the quantity it is asking about, and the
# family projection then removes it again wherever that family forbids it, so a
# stationary family still reports no regime and a regime family still reports no
# factor.
SCREEN_REGIMES = 2
SCREEN_FACTORS = 1


@dataclass(frozen=True)
class ScaleReading:
    """What both screens say about one scale at one setting of one lever."""

    k: int
    n_epochs: int
    density: float
    edge_margin: float  # median true gain less the floor it must clear
    regime_dim: int
    regime_margin: float  # oracle accuracy less the modal share
    regime_verdict: str
    factor_dim: int
    factor_margin: float  # oracle correlation less the shifted null
    factor_verdict: str

    @property
    def edge_clears(self) -> bool:
        """True when a true edge at this scale beats what a non-edge reaches."""
        return bool(np.isfinite(self.edge_margin) and self.edge_margin > 0.0)

    def __str__(self) -> str:
        return (
            f"scale {self.k}: {self.n_epochs} epochs, density {self.density:.3f}, "
            f"edge {self.edge_margin:+.2f} nats, "
            f"regime (M={self.regime_dim}) {self.regime_margin:+.3f} "
            f"{self.regime_verdict}, "
            f"factor (r={self.factor_dim}) {self.factor_margin:+.3f} "
            f"{self.factor_verdict}"
        )


def read_scales(
    cfg: ScenarioConfig,
    seed: int = 0,
    screen_indegree: int = SCREEN_INDEGREE,
    screen_regimes: int = SCREEN_REGIMES,
    screen_factors: int = SCREEN_FACTORS,
) -> List[ScaleReading]:
    """
    Both screens on one scenario, one reading per scale, finest first.

    Every scale is given an edge, a regime and a factor before it is screened,
    since a scale the grid has already emptied cannot answer whether a
    generating value would let it carry anything.  The family projection then
    removes whichever quantity that family forbids.
    """
    from .latent_screen import screen_scenario
    from .scenarios import ScenarioGrid

    placed = replace(
        cfg,
        max_indegree=[max(d, screen_indegree) for d in cfg.max_indegree],
        M_k=[max(m, screen_regimes) for m in cfg.M_k],
        r_k=[max(r, screen_factors) for r in cfg.r_k],
    )
    edges = ScenarioGrid.identifiability_probe(placed, seed=seed)
    latents = screen_scenario(placed, seed=seed)
    by_scale = {(v.k, v.quantity): v for v in latents}

    out: List[ScaleReading] = []
    for k in range(1, cfg.K + 1):
        row = edges[k]
        gain = row["true_gain"]
        floor = max(row["null_gain95"], row["lam_bic"])
        regime = by_scale[(k, "regime")]
        factor = by_scale[(k, "factor")]
        out.append(
            ScaleReading(
                k=k,
                n_epochs=int(row["n_epochs"]),
                density=float(row["density"]),
                edge_margin=float("nan") if gain is None else float(gain) - floor,
                regime_dim=regime.dim,
                regime_margin=float(regime.margin),
                regime_verdict=regime.verdict,
                factor_dim=factor.dim,
                factor_margin=float(factor.margin),
                factor_verdict=factor.verdict,
            )
        )
    return out


def with_lever(cfg: ScenarioConfig, lever: str, value: Any) -> ScenarioConfig:
    """The scenario with one generating value replaced."""
    if not hasattr(cfg, lever):
        raise ValueError(f"no such generating value: {lever}")
    return replace(cfg, **{lever: value})


def hold_coarsest(cfg: ScenarioConfig, branching: Sequence[int]) -> ScenarioConfig:
    """
    The scenario at a new branching, lengthened to keep the coarsest epoch count.

    Widening a block divides the epochs of every scale above it, so a branching
    sweep otherwise moves two things at once.  This fixes the coarsest count at
    what the scenario already had and lets the finest series grow to pay for it.
    """
    target = cfg.H_k[cfg.K - 1]
    span = 1
    for b in branching:
        span *= int(b)
    return replace(cfg, branching=list(branching), H_finest=int(target) * span)


def sweep(
    cfg: ScenarioConfig,
    lever: str,
    values: Sequence[Any],
    seed: int = 0,
    keep_coarsest: bool = False,
) -> List[Tuple[Any, List[ScaleReading]]]:
    """
    One reading per scale for each value of one generating lever.

    `keep_coarsest` applies only to `branching`, where it lengthens the series so
    that the coarsest scale keeps the epoch count it had.
    """
    out = []
    for value in values:
        if lever == "branching" and keep_coarsest:
            trial = hold_coarsest(cfg, value)
        else:
            trial = with_lever(cfg, lever, value)
        out.append((value, read_scales(trial, seed=seed)))
    return out


@dataclass(frozen=True)
class ClearingRate:
    """How often a quantity clears at one scale, over scenarios and seeds."""

    k: int
    quantity: str
    cleared: int
    total: int
    median_margin: float

    @property
    def rate(self) -> float:
        """Share of the screened pairs where this quantity cleared."""
        return self.cleared / self.total if self.total else 0.0

    def __str__(self) -> str:
        return (
            f"scale {self.k} {self.quantity}: {self.cleared}/{self.total} clear, "
            f"median margin {self.median_margin:+.3f}"
        )


def sweep_grid(
    cfgs: Sequence[ScenarioConfig],
    lever: str,
    values: Sequence[Any],
    seeds: Sequence[int] = (0,),
    keep_coarsest: bool = False,
) -> List[Tuple[Any, List[ClearingRate]]]:
    """
    A sweep read over a set of scenarios and seeds rather than over one draw.

    One scenario at one seed cannot settle a generating value.  A margin close
    to its threshold is a draw, and adopting it is how a benchmark ends up tuned
    to a single replicate.  This reports the share of pairs that clear.
    """
    out = []
    for value in values:
        per: dict = {}
        for cfg in cfgs:
            trial = (
                hold_coarsest(cfg, value)
                if lever == "branching" and keep_coarsest
                else with_lever(cfg, lever, value)
            )
            for seed in seeds:
                for r in read_scales(trial, seed=seed):
                    for quantity, margin, clears in (
                        ("edge", r.edge_margin, r.edge_clears),
                        ("regime", r.regime_margin, r.regime_verdict == "clears"),
                        ("factor", r.factor_margin, r.factor_verdict == "clears"),
                    ):
                        entry = per.setdefault((r.k, quantity), [0, 0, []])
                        entry[1] += 1
                        entry[0] += int(clears)
                        if np.isfinite(margin):
                            entry[2].append(margin)
        rates = [
            ClearingRate(
                k=k,
                quantity=q,
                cleared=c,
                total=n,
                median_margin=float(np.median(m)) if m else float("nan"),
            )
            for (k, q), (c, n, m) in sorted(per.items())
        ]
        out.append((value, rates))
    return out


def report_grid(rows: Sequence[Tuple[Any, List[ClearingRate]]], lever: str) -> str:
    """The aggregated sweep as text, one line per scale and quantity."""
    lines = [f"{lever:<18}{'k':>3}{'quantity':>10}{'clears':>10}{'median':>10}"]
    for value, rates in rows:
        for r in rates:
            lines.append(
                f"{str(value):<18}{r.k:>3}{r.quantity:>10}"
                f"{f'{r.cleared}/{r.total}':>10}{r.median_margin:>+10.3f}"
            )
    return "\n".join(lines)


def report(rows: Sequence[Tuple[Any, List[ScaleReading]]], lever: str) -> str:
    """The sweep as text, so a setting can be argued from it."""
    lines = [
        f"{lever:<16}{'k':>3}{'T_k':>8}{'density':>9}{'edge':>9}"
        f"{'regime':>9}{'factor':>9}  verdicts"
    ]
    for value, readings in rows:
        for r in readings:
            lines.append(
                f"{str(value):<16}{r.k:>3}{r.n_epochs:>8}{r.density:>9.3f}"
                f"{r.edge_margin:>+9.2f}{r.regime_margin:>+9.3f}"
                f"{r.factor_margin:>+9.3f}  "
                f"edge {'yes' if r.edge_clears else 'no':<4}"
                f"regime {r.regime_verdict:<15}factor {r.factor_verdict}"
            )
    return "\n".join(lines)


def best_value(
    rows: Sequence[Tuple[Any, List[ScaleReading]]],
    k: int,
    quantity: str,
) -> Optional[Any]:
    """
    The lever value whose margin at scale `k` is largest for one quantity.

    Returns None when nothing at that scale clears, which is the answer a sweep
    is allowed to give and the only one that licenses leaving a scale empty.
    """
    best, best_margin = None, 0.0
    for value, readings in rows:
        r = next((x for x in readings if x.k == k), None)
        if r is None:
            continue
        margin = {
            "edge": r.edge_margin,
            "regime": r.regime_margin,
            "factor": r.factor_margin,
        }[quantity]
        if np.isfinite(margin) and margin > best_margin:
            best, best_margin = value, margin
    return best


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    from .scenarios import ScenarioGrid

    parser = argparse.ArgumentParser(description="sweep one generating value")
    parser.add_argument("--scenario", default="F2_L5_M2_zil")
    parser.add_argument("--lever", default="branching")
    parser.add_argument(
        "--values",
        default="[10,6];[10,12];[10,24]",
        help="semicolon separated, each a python literal",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep-coarsest", action="store_true")
    args = parser.parse_args(argv)

    from ast import literal_eval

    grid = {c.name: c for c in ScenarioGrid.full_benchmark()}
    cfg = grid[args.scenario]
    values = [literal_eval(v) for v in args.values.split(";")]
    rows = sweep(
        cfg, args.lever, values, seed=args.seed, keep_coarsest=args.keep_coarsest
    )
    print(report(rows, args.lever))
    for k in range(1, cfg.K + 1):
        for quantity in ("edge", "regime", "factor"):
            pick = best_value(rows, k, quantity)
            print(
                f"  scale {k} {quantity}: best {pick}"
                if pick is not None
                else f"  scale {k} {quantity}: nothing clears"
            )
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(main())
