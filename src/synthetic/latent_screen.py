"""
synthetic/latent_screen.py
==========================
Whether a scenario's latent states are recoverable from its own parameters.

Frozen.  This module belongs to the allocation formulation, in which a coarse
count is drawn and then split downward among its children.  It keeps working
and keeps reproducing the stored results, and nothing new is built on it.

A benchmark that places a latent state its own generating parameters cannot
recover tests nothing.  Every metric block driven by that state reports noise,
and a fit scored badly on it is not being told anything about the estimator.
The edge screen already asks this question of lagged edges, which is how the
in-degree per scale was decided; this asks it of the regime chain and of the
latent factor.

Each quantity is scored against the floor a method with no information already
reaches.  For a regime that floor is the constant predictor: a chain spending
most of its epochs in one state is recovered no better than always naming that
state, so what decides it is the oracle accuracy less the share of the modal
state rather than the accuracy alone.  A factor has no constant predictor, so
its floor is the correlation a circularly shifted copy of the same trajectory
reaches, which is the null the edge confirmation already uses.

The score is taken with the generating parameters and the generating graph
installed, so it is a ceiling.  A quantity whose ceiling does not clear its
floor is unrecoverable at that scale by any estimator, and the scale should
carry no such latent rather than carry one nothing can find.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .scenarios import ScenarioConfig

# Accuracy the oracle regime path must clear the constant predictor by.  A
# margin rather than a sign, since a path that beats naming the modal state by a
# thousandth carries no more information than one that ties it.  The value is
# about two standard errors of an accuracy estimate at the scales where the
# decision is close: a persistent two-state chain over 1411 epochs carries an
# effective sample size near 353, so one standard error is near 0.027 and a
# margin under 0.05 is not distinguishable from zero.  The decision is sensitive
# to this between 0.02 and 0.05 and to nothing above it.
MIN_ACC_MARGIN = 0.05

# Correlation the oracle factor must clear its shifted null by, on the same
# reading.
MIN_CORR_MARGIN = 0.05

# Circular shifts of the true trajectory that set the factor null.  The shift
# keeps the marginal distribution and the autocorrelation of the trajectory and
# destroys only its alignment with the inferred one.
N_SHIFTS = 16

CLEARS = "clears"
FAILS = "fails"
NOT_APPLICABLE = "not applicable"


@dataclass(frozen=True)
class LatentVerdict:
    """Whether one latent quantity at one scale is recoverable from the truth."""

    k: int
    quantity: str  # "regime" or "factor"
    dim: int  # M for a regime, r for a factor
    score: float  # oracle accuracy, or oracle correlation
    floor: float  # modal share, or the shifted-trajectory null
    margin: float
    verdict: str
    occupancy: Tuple[float, ...] = field(default=())

    @property
    def clears(self) -> bool:
        """True only when the quantity is present and beats its floor."""
        return self.verdict == CLEARS

    def __str__(self) -> str:
        head = (
            f"scale {self.k} {self.quantity} (dim {self.dim}): {self.verdict}, "
            f"oracle {self.score:.3f} against floor {self.floor:.3f}, "
            f"margin {self.margin:+.3f}"
        )
        if self.occupancy:
            shares = ", ".join(f"{s:.3f}" for s in self.occupancy)
            head += f", occupancy [{shares}]"
        return head


def occupancy(S: Sequence[int], M: int) -> np.ndarray:
    """Share of epochs the path spends in each state."""
    counts = np.bincount(np.asarray(S, dtype=int), minlength=M)[:M].astype(float)
    total = counts.sum()
    return counts / total if total > 0 else counts


def shifted_factor_null(
    H_hat: np.ndarray,
    H_true: np.ndarray,
    n_shifts: int = N_SHIFTS,
) -> float:
    """Highest correlation a circularly shifted copy of the trajectory reaches."""
    from ..evaluation.metrics import best_factor_corr

    T = H_true.shape[0]
    if T < 2 or H_hat.shape[1] == 0 or H_true.shape[1] == 0:
        return float("nan")
    step = max(T // (n_shifts + 1), 1)
    nulls = [
        best_factor_corr(H_hat, np.roll(H_true, step * i, axis=0))
        for i in range(1, n_shifts + 1)
    ]
    return float(np.nanmax(nulls))


@dataclass
class OraclePosterior:
    """The posterior a scenario's own generating parameters produce."""

    dataset: object
    train: object
    summaries: list
    graph: object


def oracle_posterior(cfg: ScenarioConfig, seed: int = 0) -> OraclePosterior:
    """Generate one replicate and run the E-step with the generating truth."""
    from ..core.graph import MultiScaleDBNGraph
    from ..learning.estep import EStep
    from .families import make_generator

    gen = make_generator(cfg, seed=seed)
    ds = gen.generate(seed=seed)
    graph = MultiScaleDBNGraph(hierarchy=ds.hierarchy, label_names=gen.label_names)
    for ki, p in enumerate(ds.params):
        for dst, parents in p.parent_sets.items():
            for src, lag in parents:
                graph.add_edge(ki + 1, src, lag, dst)

    train, _, _ = ds.data.split(fracs=(0.7, 0.15, 0.15))
    estep = EStep()
    summaries: List[Optional[object]] = [None] * cfg.K
    for k in range(cfg.K, 0, -1):
        summaries[k - 1] = estep.run(
            k,
            train,
            ds.params[k - 1],
            graph,
            summaries[k] if k < cfg.K else None,
        )
    return OraclePosterior(dataset=ds, train=train, summaries=summaries, graph=graph)


def screen_scenario(
    cfg: ScenarioConfig,
    seed: int = 0,
    n_shifts: int = N_SHIFTS,
    min_acc_margin: float = MIN_ACC_MARGIN,
    min_corr_margin: float = MIN_CORR_MARGIN,
) -> List[LatentVerdict]:
    """One verdict per latent quantity per scale, finest first."""
    from ..evaluation.metrics import LatentMetrics, best_factor_corr

    oracle = oracle_posterior(cfg, seed=seed)
    ds, train, summaries = oracle.dataset, oracle.train, oracle.summaries
    lengths = [train.get_X(k).shape[0] for k in range(1, cfg.K + 1)]
    S_true = [ds.latents.get_S(k)[: lengths[k - 1]] for k in range(1, cfg.K + 1)]
    H_true = [ds.latents.get_H(k)[: lengths[k - 1]] for k in range(1, cfg.K + 1)]
    latent = LatentMetrics.compute(
        [s.gamma for s in summaries],
        S_true,
        [s.m_H for s in summaries],
        H_true,
    )

    out: List[LatentVerdict] = []
    for k in range(1, cfg.K + 1):
        ki = k - 1
        M = summaries[ki].gamma.shape[1]
        shares = occupancy(S_true[ki], M)
        modal = float(shares.max())
        acc = float(latent.acc_S_per_scale[ki])
        out.append(
            LatentVerdict(
                k=k,
                quantity="regime",
                dim=M,
                score=acc,
                floor=modal,
                margin=acc - modal,
                verdict=(
                    NOT_APPLICABLE
                    if M <= 1
                    else CLEARS if acc - modal >= min_acc_margin else FAILS
                ),
                occupancy=tuple(round(float(s), 4) for s in shares),
            )
        )

        H_hat = summaries[ki].m_H
        r = H_hat.shape[1]
        corr = best_factor_corr(H_hat, H_true[ki])
        null = shifted_factor_null(H_hat, H_true[ki], n_shifts=n_shifts)
        margin = float(corr - null) if r > 0 else float("nan")
        out.append(
            LatentVerdict(
                k=k,
                quantity="factor",
                dim=r,
                score=float(corr),
                floor=float(null),
                margin=margin,
                verdict=(
                    NOT_APPLICABLE
                    if r == 0
                    else CLEARS if margin >= min_corr_margin else FAILS
                ),
            )
        )
    return out


def supported_regime_counts(
    verdicts: Sequence[LatentVerdict],
    M_k: Sequence[int],
) -> List[int]:
    """The regime count per scale, with every scale the screen refuses set to one."""
    refused = {v.k for v in verdicts if v.quantity == "regime" and v.verdict == FAILS}
    return [1 if k in refused else int(M) for k, M in enumerate(M_k, start=1)]


def report(verdicts: Sequence[LatentVerdict]) -> str:
    """The screen as text, for a grid change to be argued from."""
    lines = ["latent support, scored with the generating parameters installed"]
    lines.extend(f"  {v}" for v in verdicts)
    failed = [(v.k, v.quantity) for v in verdicts if v.verdict == FAILS]
    if failed:
        names = ", ".join(f"{q} at scale {k}" for k, q in failed)
        lines.append(
            f"  unrecoverable from the generating truth: {names};"
            " a scenario placing one there scores every block it drives on noise"
        )
    return "\n".join(lines)


def screen_grid(grid, seed: int = 0, n_shifts: int = N_SHIFTS) -> str:
    """Screen every scenario of a grid and print one row per latent quantity."""
    header = f"{'scenario':<20}{'k':>3}{'kind':>8}{'dim':>5}"
    header += f"{'oracle':>9}{'floor':>8}{'margin':>9}  verdict"
    lines = [header]
    for cfg in grid:
        for v in screen_scenario(cfg, seed=seed, n_shifts=n_shifts):
            lines.append(
                f"{cfg.name:<20}{v.k:>3}{v.quantity:>8}{v.dim:>5}"
                f"{v.score:>9.3f}{v.floor:>8.3f}{v.margin:>+9.3f}  {v.verdict}"
            )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    from .scenarios import ScenarioGrid

    parser = argparse.ArgumentParser(description="screen the grid's latent states")
    parser.add_argument("--grid", default="full", choices=("full", "sanity"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shifts", type=int, default=N_SHIFTS)
    parser.add_argument("--family", type=int, default=0, help="0 screens every family")
    parser.add_argument(
        "--regime-strength",
        type=float,
        default=0.0,
        help="override the emission contrast, to find where a regime becomes visible",
    )
    args = parser.parse_args(argv)

    grid = (
        ScenarioGrid.full_benchmark()
        if args.grid == "full"
        else ScenarioGrid.sanity_check()
    )
    if args.family:
        grid = grid.filter(family=args.family)
    if args.regime_strength:
        for cfg in grid:
            cfg.regime_strength = args.regime_strength
    print(screen_grid(grid, seed=args.seed, n_shifts=args.shifts))
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(main())
