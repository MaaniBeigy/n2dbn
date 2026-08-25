"""
synthetic/families.py
=====================
Ground-truth parameter factories for the four synthetic benchmark families.

Frozen.  This module belongs to the allocation formulation, in which a coarse
count is drawn and then split downward among its children.  It keeps working
and keeps reproducing the stored results, and nothing new is built on it.

Family I  - Stationary sparse DBN  (M=1, r=0)
Family II - Regime-switching        (M>1, r=0)
Family III- Same-slice co-movement  (M=1, r>0)
Family IV - Misspecified / stress   (M≥1, r≥0, with deliberate violations)

Usage
-----
    from n2dbn.synthetic.families import make_family_I
    gen = make_family_I(scenario, seed=42)
    dataset = gen.generate(seed=42)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..core.scales import ScaleHierarchy
from ..model.params import (
    CoarseCountParams,
    CoarseDurParams,
    FactorParams,
    FineCountParams,
    FineDurParams,
    RegimeParams,
    ScaleParams,
)
from .generator import N2DBNGenerator
from .scenarios import ScenarioConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _zeros(shape) -> np.ndarray:
    return np.zeros(shape, dtype=np.float64)


def _sample_sparse_graph(
    L: int,
    max_lag: int,
    max_indegree: int,
    rng: np.random.Generator,
) -> Dict[int, Set[Tuple[int, int]]]:
    """
    Sample a sparse random DAG as lagged parent sets.

    Returns
    -------
    parent_sets : {dst_ell: {(src_ell, d), ...}}
    """
    parent_sets: Dict[int, Set[Tuple[int, int]]] = {ell: set() for ell in range(L)}
    if max_indegree == 0 or max_lag == 0:
        return parent_sets

    candidates = [(src, d) for src in range(L) for d in range(1, max_lag + 1)]
    for dst in range(L):
        n_parents = rng.integers(0, max_indegree + 1)
        if n_parents > 0 and candidates:
            chosen = rng.choice(
                len(candidates), size=min(n_parents, len(candidates)), replace=False
            )
            for idx in chosen:
                parent_sets[dst].add(candidates[idx])
    return parent_sets


# At the coarsest scale a label's own lagged value feeds its log-mean, so the
# parent coefficients form an autoregression whose fixed point is
# intercept / (1 - total coefficient).  Bounding the total keeps the generated
# counts and durations inside the range the slot normalisation allows.  Finer
# scales need no such bound, since their scores pass through a softmax over the
# children of a block and cannot run away.
#
# The bound is a range rather than a ceiling, and the total is drawn inside it.
# Clipping to a ceiling made every coarse coefficient of the grid exactly 0.3
# over 94 edges, unchanged at strengths 10, 30 and 100, so a coarse edge had one
# strength by construction and no sweep could make one easier or harder.  A
# multiplier is the wrong knob at this scale for the same reason the ceiling
# was: the stability bound decides the answer whatever the multiplier says.  So
# the total itself is the knob here, and `edge_strength` governs the finer
# scales, where the softmax bounds the scores and no cap is needed.
#
# The upper end has to stay below one.  The fixed point of the autoregression is
# `intercept / (1 - total)`, so 0.3 multiplies the level by 1.43 and 0.9 by ten.
# Measured 2026-08-14 and re-measured after the persistence fix, over four
# scenarios by two seeds.  The coarsest-scale edge and every latent below it pull
# against each other, because strong coarse counts raise the level a non-edge
# reaches at the scale below:
#
#   range        coarsest edge      scale below it, regime
#   (0.1, 0.3)   5/8 at +2.75       7/8 at +0.089
#   (0.3, 0.5)   5/8 at +29.98      6/8 at +0.090
#   (0.5, 0.7)   6/8 at +105.60     4/8 at +0.062
#
# The middle row is taken.  It keeps the coarsest edge an order of magnitude
# above the weakest setting and gives up one scenario of the latent below it,
# where the strongest setting would give up three.  Beyond 0.7 the count
# autoregression runs away and the coarsest margin collapses to -13.84.
COARSE_LAG_TOTAL = (0.30, 0.50)

# Kept as the upper end of that range under its old name, since it is quoted in
# the record and read by tests.
MAX_TOTAL_LAG_COARSE = COARSE_LAG_TOTAL[1]


def _make_lag_coef(
    parent_sets: Dict[int, Set[Tuple[int, int]]],
    L: int,
    strength: float,
    rng: np.random.Generator,
    total_range: Optional[Tuple[float, float]] = None,
) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Build a coefficient dict  {(src_ell, lag): (L,) array}.

    Non-zero only where the (src, lag) is an actual parent of some label.  With
    `total_range` the coefficients of one label are rescaled to a total drawn
    inside that range, so labels differ in how strongly their parents act while
    every one of them stays inside the stability bound.  Without it the drawn
    coefficients stand, which is what the finer scales use.
    """
    all_edges: Set[Tuple[int, int]] = set()
    for pset in parent_sets.values():
        all_edges |= pset
    coef: Dict[Tuple[int, int], np.ndarray] = {}
    for src, d in all_edges:
        v = np.zeros(L)
        for dst in range(L):
            if (src, d) in parent_sets[dst]:
                v[dst] = rng.uniform(0.1 * strength, strength)
        coef[(src, d)] = v

    if total_range is not None:
        low, high = float(total_range[0]), float(total_range[1])
        for dst in range(L):
            total = sum(abs(float(v[dst])) for v in coef.values())
            if total <= 0.0:
                continue
            target = rng.uniform(low, high)
            factor = target / total
            for v in coef.values():
                v[dst] *= factor
    return coef


# ---------------------------------------------------------------------------
# Per-scale parameter builders
# ---------------------------------------------------------------------------


def _regime_effect(
    M: int,
    L: int,
    scale: float,
    strength: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Regime-specific offsets that contrast the regimes without moving the level.

    The rows are centred across regimes, so `strength` scales how far the
    regimes sit apart and leaves the average emission level to the intercept.
    An uncentred draw would move the level of every label by a random amount as
    soon as the strength is raised, which changes the data far more than it
    changes the regime signal.
    """
    eff = rng.standard_normal((M, L)) * scale * strength
    return eff - eff.mean(axis=0, keepdims=True)


# Self-transition probability of the chain at the **coarsest** scale, where a
# regime acts on the emission and persistence is the thing being modeled.  It
# sets how long a regime lasts and leaves the occupancy alone, which the free
# logits it replaced did not.
REGIME_PERSISTENCE = (0.70, 0.90)

# Runs a regime should complete inside one parent block, at every scale below
# the coarsest.  There a regime enters only the allocation, which is a softmax
# over the children of a block, and a softmax is invariant to anything added
# equally to every child.  A path holding one value across a block therefore
# contributes exactly nothing to it.  Drawing the same persistence at every
# scale made a fine regime last about as long as its own parent block, so it was
# constant within 48 percent of blocks at the finest scale and 63 percent at the
# scale below the coarsest.  Persistence is scale relative instead, which is what
# nesting means: a regime at scale k persists for a few epochs of scale k.
# Measured 2026-08-14 by sweeping it and re-screening the nine regime scenarios.
# The share whose scale-2 regime clears runs 2, 5, 5 and 6 of 9 at 3, 6, 12 and
# 30 runs per block, and the median margin runs +0.018, +0.062, +0.062, +0.062,
# so it rises steeply to 6 and saturates there.  Six is taken rather than thirty
# because the extra runs buy no margin and cost the regime what persistence it
# has left.
REGIME_RUNS_PER_BLOCK = 6.0

# Bounds on the drawn persistence, so a block narrower than the target runs
# still yields a usable chain rather than an impossible one.
REGIME_PERSISTENCE_BOUNDS = (0.05, 0.95)


def _persistence_for(k: int, K: int, branching: Sequence[int]) -> Tuple[float, float]:
    """Persistence range for scale k, relative to the block it sits inside."""
    if k >= K:
        return REGIME_PERSISTENCE
    width = float(branching[k - 1])
    target = 1.0 - REGIME_RUNS_PER_BLOCK / max(width, 1.0)
    lo, hi = REGIME_PERSISTENCE_BOUNDS
    return (
        float(np.clip(target - 0.10, lo, hi)),
        float(np.clip(target + 0.10, lo, hi)),
    )


# Half-width of the uniform draw around an equal share, as a fraction of it.  A
# state occupied far below the others is recovered no better than naming the
# modal state whatever the emission contrast, so the spread is bounded rather
# than left to whatever free logits imply.
REGIME_OCCUPANCY_SPREAD = 0.10


def _stationary_chain(
    M: int,
    rng: np.random.Generator,
    persistence: Tuple[float, float] = REGIME_PERSISTENCE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transition logits whose stationary distribution is a stated near-equal share.

    Each row of the transition matrix is `q e_m + (1 - q) pi`, a self transition
    mixed with a draw from `pi`, which leaves `pi` stationary exactly for any
    persistence `q`.  Free logits do not: their stationary distribution follows
    from the whole matrix, and the uniform bias they were drawn with put the
    finest-scale chain about 88 percent in one state, where no estimator
    recovers the path better than naming that state.

    Returns the (M, M) logits in the layout `logit_transition` reads, whose
    column is the previous state, and the stationary shares themselves.
    """
    share = np.ones(M) + rng.uniform(
        -REGIME_OCCUPANCY_SPREAD, REGIME_OCCUPANCY_SPREAD, M
    )
    share /= share.sum()
    q = rng.uniform(*persistence)
    P = q * np.eye(M) + (1.0 - q) * share[np.newaxis, :]
    return np.log(P).T, share


def _make_regime_params(
    M: int,
    parent_M: int,
    d_ctx: int,
    rng: np.random.Generator,
    persistence: Tuple[float, float] = REGIME_PERSISTENCE,
) -> RegimeParams:
    """Build realistic but random regime transition parameters."""
    trans, share = _stationary_chain(M, rng, persistence)

    # The transition already carries the whole stationary distribution, so a
    # further per-state offset would only move the occupancy away from it.
    bias = _zeros(M)

    # Cross-scale coupling (only for k < K), and the context weights, are
    # centred across the states they choose between, so each contrasts the
    # states without shifting which one the chain prefers on average.  The
    # coupling is centred across the parent states as well, since a tilt that
    # is the same sign under every parent state is a preference the parent
    # cannot average away, and that is what left the finer scales skewed after
    # the coarsest one was fixed.
    cross = None
    if parent_M > 0:
        cross = rng.standard_normal((M, parent_M)) * 0.5
        cross -= cross.mean(axis=0, keepdims=True)
        cross -= cross.mean(axis=1, keepdims=True)

    if d_ctx > 0:
        ctx = rng.standard_normal((M, d_ctx)) * 0.2
        ctx -= ctx.mean(axis=0, keepdims=True)
    else:
        ctx = _zeros((M, 0))

    return RegimeParams(
        M=M,
        parent_M=parent_M,
        d_ctx=d_ctx,
        init=np.log(share),
        bias=bias,
        trans=trans,
        cross=cross,
        ctx=ctx,
    )


def _make_factor_params(
    M: int,
    r: int,
    d_ctx: int,
    rng: np.random.Generator,
) -> FactorParams:
    """Build latent factor LGSSM parameters.  When r=0 returns trivial params."""
    if r == 0:
        return FactorParams(
            M=M,
            r=0,
            d_ctx=d_ctx,
            F=_zeros((M, 0, 0)),
            G=_zeros((M, 0, d_ctx)),
            Sigma=_zeros((M, 0, 0)),
            mu0=_zeros(0),
            Sigma0=_zeros((0, 0)),
        )
    # Stable AR(1): eigenvalues inside unit circle
    F_mats = np.stack(
        [
            np.eye(r) * rng.uniform(0.3, 0.8) + rng.standard_normal((r, r)) * 0.05
            for _ in range(M)
        ]
    )
    G_mats = (
        rng.standard_normal((M, r, d_ctx)) * 0.3 if d_ctx > 0 else _zeros((M, r, 0))
    )
    Sigmas = np.stack([np.eye(r) * rng.uniform(0.1, 0.5) for _ in range(M)])
    mu0 = _zeros(r)
    Sigma0 = np.eye(r) * 0.5
    return FactorParams(
        M=M, r=r, d_ctx=d_ctx, F=F_mats, G=G_mats, Sigma=Sigmas, mu0=mu0, Sigma0=Sigma0
    )


def _make_coarse_count_params(
    L: int,
    M: int,
    r: int,
    d_ctx: int,
    parent_sets: Dict[int, Set[Tuple[int, int]]],
    zi_prob: float,
    rng: np.random.Generator,
    edge_strength: float = 1.0,
    log_mean_base: Optional[float] = None,
    regime_strength: float = 1.0,
    coarse_lag_total: Tuple[float, float] = COARSE_LAG_TOTAL,
) -> CoarseCountParams:
    """ZINB count emission parameters for the coarsest scale."""
    betaLag = _make_lag_coef(
        parent_sets,
        L,
        strength=0.4 * edge_strength,
        rng=rng,
        total_range=coarse_lag_total,
    )
    if log_mean_base is None:
        beta0 = rng.uniform(0.5, 2.0, L)  # log mean ≈ 1..7 events
    else:
        beta0 = log_mean_base + rng.uniform(-0.4, 0.4, L)

    return CoarseCountParams(
        L=L,
        beta0=beta0,
        betaS=_regime_effect(M, L, 0.6, regime_strength, rng),
        betaH=rng.standard_normal((r, L)) * 0.5 if r > 0 else _zeros((0, L)),
        betaCtx=rng.standard_normal((d_ctx, L)) * 0.2 if d_ctx > 0 else _zeros((0, L)),
        betaLag=betaLag,
        r_disp=rng.uniform(1.0, 5.0, L),  # moderate dispersion
        gamma0=np.full(L, np.log(zi_prob / (1 - zi_prob + 1e-8))),  # logit(zi_prob)
        gammaS=_regime_effect(M, L, 0.3, regime_strength, rng),
        gammaH=rng.standard_normal((r, L)) * 0.2 if r > 0 else _zeros((0, L)),
        gammaCtx=rng.standard_normal((d_ctx, L)) * 0.1 if d_ctx > 0 else _zeros((0, L)),
    )


def _make_coarse_dur_params(
    L: int,
    M: int,
    r: int,
    d_ctx: int,
    parent_sets: Dict[int, Set[Tuple[int, int]]],
    rng: np.random.Generator,
    edge_strength: float = 1.0,
    regime_strength: float = 1.0,
    coarse_lag_total: Tuple[float, float] = COARSE_LAG_TOTAL,
) -> CoarseDurParams:
    """NB duration parameters for the coarsest scale."""
    alphaLag = _make_lag_coef(
        parent_sets,
        L,
        strength=0.3 * edge_strength,
        rng=rng,
        total_range=coarse_lag_total,
    )

    return CoarseDurParams(
        L=L,
        alpha0=rng.uniform(0.0, 1.0, L),  # mean duration 1 to 3 slots
        alphaS=_regime_effect(M, L, 0.4, regime_strength, rng),
        alphaH=rng.standard_normal((r, L)) * 0.3 if r > 0 else _zeros((0, L)),
        alphaCtx=rng.standard_normal((d_ctx, L)) * 0.1 if d_ctx > 0 else _zeros((0, L)),
        alphaX=rng.uniform(0.1, 0.3, L),  # positive count-dur coupling
        alphaLag=alphaLag,
        s_disp=rng.uniform(1.0, 4.0, L),
    )


def _make_fine_count_params(
    L: int,
    M: int,
    r: int,
    d_ctx: int,
    parent_sets: Dict[int, Set[Tuple[int, int]]],
    rng: np.random.Generator,
    kappa: float = 8.0,
    edge_strength: float = 1.0,
    regime_strength: float = 1.0,
) -> FineCountParams:
    """Dir-Mult count allocation score parameters for a finer scale."""
    betaLag = _make_lag_coef(parent_sets, L, strength=0.3 * edge_strength, rng=rng)

    return FineCountParams(
        L=L,
        beta0=rng.standard_normal(L) * 0.3,
        betaS=_regime_effect(M, L, 0.5, regime_strength, rng),
        betaH=rng.standard_normal((r, L)) * 0.4 if r > 0 else _zeros((0, L)),
        betaCtx=rng.standard_normal((d_ctx, L)) * 0.2 if d_ctx > 0 else _zeros((0, L)),
        betaP=rng.uniform(0.1, 0.5, L),  # parent count to allocation
        betaD=rng.uniform(0.0, 0.3, L),
        betaLag=betaLag,
        kappa=kappa,
    )


def _make_fine_dur_params(
    L: int,
    M: int,
    r: int,
    d_ctx: int,
    parent_sets: Dict[int, Set[Tuple[int, int]]],
    rng: np.random.Generator,
    tau: float = 8.0,
    edge_strength: float = 1.0,
    regime_strength: float = 1.0,
) -> FineDurParams:
    """Dir-Mult duration allocation score parameters for a finer scale."""
    alphaLag = _make_lag_coef(parent_sets, L, strength=0.25 * edge_strength, rng=rng)

    return FineDurParams(
        L=L,
        alpha0=rng.standard_normal(L) * 0.3,
        alphaS=_regime_effect(M, L, 0.4, regime_strength, rng),
        alphaH=rng.standard_normal((r, L)) * 0.3 if r > 0 else _zeros((0, L)),
        alphaCtx=rng.standard_normal((d_ctx, L)) * 0.1 if d_ctx > 0 else _zeros((0, L)),
        alphaX=rng.uniform(0.1, 0.4, L),
        alphaP=rng.uniform(0.05, 0.3, L),
        alphaLag=alphaLag,
        tau=tau,
    )


# ---------------------------------------------------------------------------
# Scale params builder (shared across families)
# ---------------------------------------------------------------------------


def _build_scale_params(
    cfg: ScenarioConfig,
    k: int,  # 1-indexed
    parent_M: int,
    rng: np.random.Generator,
) -> ScaleParams:
    """Assemble a ScaleParams for scale k given a ScenarioConfig."""
    ki = k - 1
    M = cfg.M_k[ki]
    r = cfg.r_k[ki]
    d_ctx = cfg.context_dim[ki]
    L = cfg.L
    zi_prob = cfg.zi_prob
    max_lag = cfg.max_lag[ki]
    max_deg = cfg.max_indegree[ki]

    regime = _make_regime_params(
        M, parent_M, d_ctx, rng, _persistence_for(k, cfg.K, cfg.branching)
    )
    factor = _make_factor_params(M, r, d_ctx, rng)

    # Sample sparse ground-truth graph
    if cfg.sparse_graph:
        parent_sets = _sample_sparse_graph(L, max_lag, max_deg, rng)
    else:
        parent_sets = {ell: set() for ell in range(L)}

    if k == cfg.K:  # coarsest scale
        cc = _make_coarse_count_params(
            L,
            M,
            r,
            d_ctx,
            parent_sets,
            zi_prob,
            rng,
            edge_strength=cfg.edge_strength,
            log_mean_base=cfg.coarse_log_mean,
            regime_strength=cfg.regime_strength,
            coarse_lag_total=cfg.coarse_lag_total or COARSE_LAG_TOTAL,
        )
        cd = _make_coarse_dur_params(
            L,
            M,
            r,
            d_ctx,
            parent_sets,
            rng,
            edge_strength=cfg.edge_strength,
            regime_strength=cfg.regime_strength,
            coarse_lag_total=cfg.coarse_lag_total or COARSE_LAG_TOTAL,
        )
        return ScaleParams(
            k=k,
            regime=regime,
            factor=factor,
            parent_sets=parent_sets,
            coarse_count=cc,
            coarse_dur=cd,
        )
    else:  # finer scale
        conc = cfg.dirichlet_conc
        fc = _make_fine_count_params(
            L,
            M,
            r,
            d_ctx,
            parent_sets,
            rng,
            edge_strength=cfg.edge_strength,
            regime_strength=cfg.regime_strength,
            **({"kappa": conc} if conc is not None else {}),
        )
        fd = _make_fine_dur_params(
            L,
            M,
            r,
            d_ctx,
            parent_sets,
            rng,
            edge_strength=cfg.edge_strength,
            regime_strength=cfg.regime_strength,
            **({"tau": conc} if conc is not None else {}),
        )
        return ScaleParams(
            k=k,
            regime=regime,
            factor=factor,
            parent_sets=parent_sets,
            fine_count=fc,
            fine_dur=fd,
        )


# ---------------------------------------------------------------------------
# Family factories
# ---------------------------------------------------------------------------


def make_family_I(cfg: ScenarioConfig, seed: Optional[int] = None) -> N2DBNGenerator:
    """
    Family I - Stationary sparse DBN (M=1 everywhere, r=0 everywhere).

    Forces M_k = [1,...,1], r_k = [0,...,0] regardless of config values.
    """
    rng = _rng(seed if seed is not None else cfg.seed)
    hier = ScaleHierarchy.from_uniform_branching(cfg.H_finest, cfg.branching)

    # Override to stationary
    cfg_I = ScenarioConfig(
        name=cfg.name + "_FI",
        K=cfg.K,
        H_finest=cfg.H_finest,
        branching=cfg.branching,
        L=cfg.L,
        M_k=[1] * cfg.K,
        r_k=[0] * cfg.K,
        max_lag=cfg.max_lag,
        max_indegree=cfg.max_indegree,
        zi_level=cfg.zi_level,
        sparse_graph=cfg.sparse_graph,
        family=1,
        seed=cfg.seed,
        n_replicates=cfg.n_replicates,
        context_dim=cfg.context_dim,
        edge_strength=cfg.edge_strength,
        regime_strength=cfg.regime_strength,
        dirichlet_conc=cfg.dirichlet_conc,
        finest_rate=cfg.finest_rate,
        slot_minutes=cfg.slot_minutes,
        origin=cfg.origin,
        finest_binary=cfg.finest_binary,
        finest_cap=cfg.finest_cap,
        coarse_lag_total=cfg.coarse_lag_total,
    )

    params_list = []
    parent_M = 0
    for k in range(cfg_I.K, 0, -1):  # coarse to fine
        sp = _build_scale_params(cfg_I, k, parent_M, rng)
        params_list.insert(0, sp)
        parent_M = sp.M

    return N2DBNGenerator(
        params_list,
        hier,
        [f"L{i}" for i in range(cfg.L)],
        calendar=cfg.calendar,
        finest_binary=cfg.finest_binary,
        finest_cap=cfg.finest_cap,
    )


def make_family_II(cfg: ScenarioConfig, seed: Optional[int] = None) -> N2DBNGenerator:
    """
    Family II - Regime-switching multiscale dynamics (M>1, r=0).

    A scale the config leaves at one regime carries none, since the latent
    screen puts a regime below the constant predictor at every scale it refuses.
    The coarsest scale is raised to two whatever the config says, since a regime
    family with a regime nowhere is family I.  Forces r_k = [0,...,0].
    """
    rng = _rng(seed if seed is not None else cfg.seed)
    hier = ScaleHierarchy.from_uniform_branching(cfg.H_finest, cfg.branching)

    M_II = [max(2, m) if m > 1 else 1 for m in cfg.M_k]
    M_II[cfg.K - 1] = max(2, M_II[cfg.K - 1])

    cfg_II = ScenarioConfig(
        name=cfg.name + "_FII",
        K=cfg.K,
        H_finest=cfg.H_finest,
        branching=cfg.branching,
        L=cfg.L,
        M_k=M_II,
        r_k=[0] * cfg.K,
        max_lag=cfg.max_lag,
        max_indegree=cfg.max_indegree,
        zi_level=cfg.zi_level,
        sparse_graph=cfg.sparse_graph,
        family=2,
        seed=cfg.seed,
        n_replicates=cfg.n_replicates,
        context_dim=cfg.context_dim,
        edge_strength=cfg.edge_strength,
        regime_strength=cfg.regime_strength,
        dirichlet_conc=cfg.dirichlet_conc,
        finest_rate=cfg.finest_rate,
        slot_minutes=cfg.slot_minutes,
        origin=cfg.origin,
        finest_binary=cfg.finest_binary,
        finest_cap=cfg.finest_cap,
        coarse_lag_total=cfg.coarse_lag_total,
    )

    params_list = []
    parent_M = 0
    for k in range(cfg_II.K, 0, -1):
        sp = _build_scale_params(cfg_II, k, parent_M, rng)
        params_list.insert(0, sp)
        parent_M = sp.M

    return N2DBNGenerator(
        params_list,
        hier,
        [f"L{i}" for i in range(cfg.L)],
        calendar=cfg.calendar,
        finest_binary=cfg.finest_binary,
        finest_cap=cfg.finest_cap,
    )


def make_family_III(cfg: ScenarioConfig, seed: Optional[int] = None) -> N2DBNGenerator:
    """
    Family III - Shared latent factors driving same-slice co-movement (M=1, r>0).

    Forces M_k = [1,...,1] and r_k = [max(1,r),...].
    """
    rng = _rng(seed if seed is not None else cfg.seed)
    hier = ScaleHierarchy.from_uniform_branching(cfg.H_finest, cfg.branching)

    cfg_III = ScenarioConfig(
        name=cfg.name + "_FIII",
        K=cfg.K,
        H_finest=cfg.H_finest,
        branching=cfg.branching,
        L=cfg.L,
        M_k=[1] * cfg.K,
        r_k=[max(1, r) for r in cfg.r_k],
        max_lag=cfg.max_lag,
        max_indegree=cfg.max_indegree,
        zi_level=cfg.zi_level,
        sparse_graph=cfg.sparse_graph,
        family=3,
        seed=cfg.seed,
        n_replicates=cfg.n_replicates,
        context_dim=cfg.context_dim,
        edge_strength=cfg.edge_strength,
        regime_strength=cfg.regime_strength,
        dirichlet_conc=cfg.dirichlet_conc,
        finest_rate=cfg.finest_rate,
        slot_minutes=cfg.slot_minutes,
        origin=cfg.origin,
        finest_binary=cfg.finest_binary,
        finest_cap=cfg.finest_cap,
        coarse_lag_total=cfg.coarse_lag_total,
    )

    params_list = []
    parent_M = 0
    for k in range(cfg_III.K, 0, -1):
        sp = _build_scale_params(cfg_III, k, parent_M, rng)
        params_list.insert(0, sp)
        parent_M = sp.M

    return N2DBNGenerator(
        params_list,
        hier,
        [f"L{i}" for i in range(cfg.L)],
        calendar=cfg.calendar,
        finest_binary=cfg.finest_binary,
        finest_cap=cfg.finest_cap,
    )


def make_family_IV(
    cfg: ScenarioConfig,
    misspec_type: str = "heavier_tail",
    seed: Optional[int] = None,
) -> N2DBNGenerator:
    """
    Family IV - Misspecified / stress-test data.

    Supported misspecification types
    ---------------------------------
    'heavier_tail'   : increase NB dispersion to near-Poisson limit (r to large)
                       but generate with a power-law mixture (approximated by
                       inflated dispersion).
    'omit_context'   : set context weights to zero in emission (context exists in
                       data but is ignored by the model).
    'allocation_mismatch': use non-log-linear allocation features (simulated by
                           adding a nonlinear bump to the scores).

    In all cases the data are generated with a valid N²DBN but with parameter
    values that deliberately stress-test the learning procedure.
    """
    rng = _rng(seed if seed is not None else cfg.seed)
    hier = ScaleHierarchy.from_uniform_branching(cfg.H_finest, cfg.branching)

    cfg_IV = ScenarioConfig(
        name=cfg.name + f"_FIV_{misspec_type}",
        K=cfg.K,
        H_finest=cfg.H_finest,
        branching=cfg.branching,
        L=cfg.L,
        M_k=cfg.M_k,
        r_k=cfg.r_k,
        max_lag=cfg.max_lag,
        max_indegree=cfg.max_indegree,
        zi_level=cfg.zi_level,
        sparse_graph=cfg.sparse_graph,
        family=4,
        misspec_type=misspec_type,
        seed=cfg.seed,
        n_replicates=cfg.n_replicates,
        context_dim=cfg.context_dim,
        edge_strength=cfg.edge_strength,
        regime_strength=cfg.regime_strength,
        dirichlet_conc=cfg.dirichlet_conc,
        finest_rate=cfg.finest_rate,
        slot_minutes=cfg.slot_minutes,
        origin=cfg.origin,
        finest_binary=cfg.finest_binary,
        finest_cap=cfg.finest_cap,
        coarse_lag_total=cfg.coarse_lag_total,
    )

    params_list = []
    parent_M = 0
    for k in range(cfg_IV.K, 0, -1):
        sp = _build_scale_params(cfg_IV, k, parent_M, rng)
        # Apply misspecification
        if misspec_type == "heavier_tail" and sp.is_coarsest:
            sp.coarse_count.r_disp[:] = 50.0  # near-Poisson (will be estimated wrong)
        elif misspec_type == "omit_context" and sp.is_coarsest:
            # Generate with context effect but set betaCtx large so it matters
            sp.coarse_count.betaCtx[:] = (
                rng.standard_normal(sp.coarse_count.betaCtx.shape) * 1.5
            )
        params_list.insert(0, sp)
        parent_M = sp.M

    return N2DBNGenerator(
        params_list,
        hier,
        [f"L{i}" for i in range(cfg.L)],
        calendar=cfg.calendar,
        finest_binary=cfg.finest_binary,
        finest_cap=cfg.finest_cap,
    )


# ---------------------------------------------------------------------------
# Dispatch by family index
# ---------------------------------------------------------------------------


def make_generator(cfg: ScenarioConfig, seed: Optional[int] = None) -> N2DBNGenerator:
    """Dispatch to the appropriate family factory based on cfg.family."""
    dispatch = {
        1: make_family_I,
        2: make_family_II,
        3: make_family_III,
        4: make_family_IV,
    }
    if cfg.family == 4:
        # Which misspecification a scenario carries lives on the config, so a
        # grid that varies misspec_type has to reach the factory through here.
        # "none" matches no branch of make_family_IV and gives the reference
        # scenario: the same structure with nothing misspecified.
        return make_family_IV(cfg, misspec_type=cfg.misspec_type or "none", seed=seed)
    return dispatch[cfg.family](cfg, seed=seed)
