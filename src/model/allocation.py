"""
model/allocation.py
===================
Dirichlet-Multinomial count allocation and duration allocation for finer scales.

Frozen.  This module belongs to the allocation formulation, in which a coarse
count is drawn and then split downward among its children.  It keeps working
and keeps reproducing the stored results, and nothing new is built on it.

Count allocation  (eq. fine_count_dir + fine_count_mult)
---------------------------------------------------------
  ψ_c ~ Dir( κ · ω )                where ω = softmax(η)
  (X_t)_{t∈children(c)} | X_c, ψ_c ~ Mult( X_c, ψ_c )
  ⟹  Σ X_t = X_c  a.s.

Duration allocation  (eq. fine_duration_dir + fine_duration_mult)
------------------------------------------------------------------
  Only over *active* children  𝒜_c = {t : X_t > 0}.
  ϱ_c ~ Dir( τ · ρ )                where ρ = softmax(ξ) over 𝒜_c
  (Y_t)_{t∈𝒜_c} | Y_c, ϱ_c ~ Mult( Y_c, ϱ_c )
  Y_t = 0  for  t ∉ 𝒜_c
  ⟹  Σ Y_t = Y_c  a.s.

Score computation
-----------------
η_t  and  ξ_t  are log-linear predictors that depend on latent state,
parent-scale totals, context, and lagged within-scale counts / durations.
The functions below compute these scores and drive the multinomial allocation.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from .params import FineCountParams, FineDurParams

LagKey = Tuple[int, int]


# ---------------------------------------------------------------------------
# Score helper
# ---------------------------------------------------------------------------


def _compute_score_vector(
    t: int,
    s_idx: int,
    h_t: np.ndarray,  # (r,)
    ctx_t: np.ndarray,  # (d_ctx,)
    X_parent_ell: int,  # X_c^{(k+1, ell)}
    Y_parent_ell: int,  # Y_c^{(k+1, ell)}
    X_fine: np.ndarray,  # (H_k, L)  already-filled part
    Y_fine: np.ndarray,  # (H_k, L)  already-filled part
    parent_sets_ell: Set[LagKey],
    beta0: np.ndarray,  # (1,)  or scalar
    betaS: np.ndarray,  # (M, 1)
    betaH: np.ndarray,  # (r, 1)
    betaCtx: np.ndarray,  # (d_ctx, 1)
    betaP: float,  # scalar
    betaD: float,  # scalar
    betaLag: Dict[LagKey, float],  # {(src, d): scalar coeff} for one dst label
    mode: str = "count",  # "count" | "duration"
    X_fine_ell: Optional[np.ndarray] = None,  # (H_k,) for duration α_X term
    alphaX: float = 0.0,
) -> float:
    """
    Compute the score η_t  (mode="count") or ξ_t  (mode="duration")
    for a single (epoch t, label ell) pair.

    All label-specific tensors are sliced before calling this function.

    Returns a scalar score.
    """
    score = float(beta0) + float(betaS[s_idx])

    if betaH.shape[0] > 0:  # r > 0
        score += float(betaH @ h_t)
    if betaCtx.shape[0] > 0:
        score += float(betaCtx @ ctx_t)

    score += betaP * np.log1p(X_parent_ell)
    score += betaD * np.log1p(Y_parent_ell)

    for (src_ell, d), coef in betaLag.items():
        t_lag = t - d
        if t_lag >= 0:
            if mode == "count":
                score += coef * np.log1p(float(X_fine[t_lag, src_ell]))
            else:
                score += coef * np.log1p(float(Y_fine[t_lag, src_ell]))

    if mode == "duration" and X_fine_ell is not None:
        score += alphaX * np.log1p(float(X_fine_ell[t]))

    return score


# ---------------------------------------------------------------------------
# Vectorised score computation for all children of one block
# ---------------------------------------------------------------------------


def compute_count_scores(
    children: List[int],
    s_seq: np.ndarray,  # (H_k,) regime sequence
    H_traj: np.ndarray,  # (H_k, r)
    ctx: np.ndarray,  # (H_k, d_ctx)
    X_parent: int,  # X_c^{(k+1, ell)}
    Y_parent: int,  # Y_c^{(k+1, ell)}
    X_fine: np.ndarray,  # (H_k, L)
    Y_fine: np.ndarray,  # (H_k, L)
    ell: int,  # destination label
    p: FineCountParams,
    parent_sets_ell: Set[LagKey],
) -> np.ndarray:
    """
    Compute η_t for all t in `children` for a fixed (parent block, label).

    Returns
    -------
    scores : (n_children,) float array.
    """
    n = len(children)
    scores = np.empty(n, dtype=np.float64)

    # Build label-specific lag dict
    betaLag_ell: Dict[LagKey, float] = {
        k: float(v[ell]) for k, v in p.betaLag.items() if k in parent_sets_ell
    }

    for i, t in enumerate(children):
        scores[i] = _compute_score_vector(
            t=t,
            s_idx=int(s_seq[t]),
            h_t=H_traj[t],
            ctx_t=ctx[t],
            X_parent_ell=X_parent,
            Y_parent_ell=Y_parent,
            X_fine=X_fine,
            Y_fine=Y_fine,
            parent_sets_ell=parent_sets_ell,
            beta0=p.beta0[ell],
            betaS=p.betaS[:, ell],
            betaH=p.betaH[:, ell],
            betaCtx=p.betaCtx[:, ell],
            betaP=float(p.betaP[ell]),
            betaD=float(p.betaD[ell]),
            betaLag=betaLag_ell,
            mode="count",
        )
    return scores


def compute_duration_scores(
    active_children: List[int],
    s_seq: np.ndarray,
    H_traj: np.ndarray,
    ctx: np.ndarray,
    Y_parent: int,
    X_fine: np.ndarray,
    Y_fine: np.ndarray,
    ell: int,
    p: FineDurParams,
    parent_sets_ell: Set[LagKey],
) -> np.ndarray:
    """
    Compute ξ_t for all t in `active_children` for a fixed (parent block, label).
    """
    n = len(active_children)
    scores = np.empty(n, dtype=np.float64)

    alphaLag_ell: Dict[LagKey, float] = {
        k: float(v[ell]) for k, v in p.alphaLag.items() if k in parent_sets_ell
    }

    for i, t in enumerate(active_children):
        scores[i] = _compute_score_vector(
            t=t,
            s_idx=int(s_seq[t]),
            h_t=H_traj[t],
            ctx_t=ctx[t],
            X_parent_ell=0,  # not used in duration score
            Y_parent_ell=Y_parent,
            X_fine=X_fine,
            Y_fine=Y_fine,
            parent_sets_ell=parent_sets_ell,
            beta0=p.alpha0[ell],
            betaS=p.alphaS[:, ell],
            betaH=p.alphaH[:, ell],
            betaCtx=p.alphaCtx[:, ell],
            betaP=0.0,
            betaD=float(p.alphaP[ell]),
            betaLag=alphaLag_ell,
            mode="duration",
            X_fine_ell=X_fine[:, ell],
            alphaX=float(p.alphaX[ell]),
        )
    return scores


# ---------------------------------------------------------------------------
# Dir-Mult sampling
# ---------------------------------------------------------------------------


def dirichlet_multinomial_sample(
    n: int,
    scores: np.ndarray,  # (n_categories,) unnormalised logit scores
    concentration: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample from a Dirichlet-Multinomial.

    1. ω = softmax(scores)
    2. ψ ~ Dir(κ · ω)
    3. x ~ Mult(n, ψ)

    Returns
    -------
    x : (n_categories,) int array summing to n.
    """
    if n == 0:
        return np.zeros(len(scores), dtype=np.int64)

    # Softmax scores → ω
    z = scores - scores.max()
    omega = np.exp(z)
    omega /= omega.sum()

    # Dirichlet draw
    alpha = concentration * omega + 1e-10  # prevent degenerate Dir
    psi = rng.dirichlet(alpha)

    # Multinomial draw
    return rng.multinomial(n, psi).astype(np.int64)


def redistribute_excess(
    x: np.ndarray,
    caps: np.ndarray,
    weights: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Move counts above a per-child capacity onto children with room left.

    The total is preserved exactly, so reconciliation still holds, and every
    entry ends at or below its capacity.  At the finest scale the capacity is
    one, which is the event-start reading of a slot: a slot either starts an
    episode of a label or it does not.

    Receivers are drawn with probability proportional to `weights` among the
    children that still have room, so the repair follows the same allocation
    scores as the draw it repairs.
    """
    x = np.asarray(x, dtype=np.int64).copy()
    caps = np.asarray(caps, dtype=np.int64)
    if x.sum() > caps.sum():
        raise ValueError(
            f"total {int(x.sum())} exceeds the capacity {int(caps.sum())} of the block"
        )
    excess = int(np.maximum(x - caps, 0).sum())
    if excess == 0:
        return x
    x = np.minimum(x, caps)
    w = np.asarray(weights, dtype=np.float64).copy()
    for _ in range(excess):
        room = caps - x
        free = room > 0
        p = np.where(free, w, 0.0)
        total = p.sum()
        if total <= 0:  # every score vanished, fall back to room
            p = free.astype(np.float64)
            total = p.sum()
        x[rng.choice(len(x), p=p / total)] += 1
    return x


# ---------------------------------------------------------------------------
# High-level block allocators
# ---------------------------------------------------------------------------


def sequential_split(
    n_remaining: int,
    omega_now: float,
    omega_rest: float,
    concentration: float,
    rng: np.random.Generator,
) -> int:
    """One child's share of a block total, from the beta-binomial factorization."""
    if n_remaining <= 0:
        return 0
    if omega_rest <= 0.0:
        # The last child of a block takes whatever the block has left.
        return int(n_remaining)
    a = max(concentration * omega_now, 1e-6)
    b = max(concentration * omega_rest, 1e-6)
    return int(rng.binomial(n_remaining, rng.beta(a, b)))


def allocate_counts_block_sequential(
    children: List[int],
    ell: int,
    X_parent: int,
    Y_parent: int,
    s_seq: np.ndarray,
    H_traj: np.ndarray,
    ctx: np.ndarray,
    X_fine: np.ndarray,
    Y_fine: np.ndarray,
    p: FineCountParams,
    parent_sets_ell: Set[LagKey],
    rng: np.random.Generator,
    write: Callable[[int, int], None],
) -> np.ndarray:
    """
    Allocate a block total over its children one child at a time, in time order.

    The joint form scores every child of a block before any of them is drawn, so
    a lagged score that reads a sibling inside the same block reads a value that
    does not exist yet.  Drawing the children in order and writing each one
    before the next is scored removes that, at the cost of one score evaluation
    per child.  `write` installs a drawn count so the next child's score can see
    it.

    The Dirichlet-Multinomial over a block factorizes exactly as a chain of
    beta-binomial draws, so with scores held fixed this returns the same
    distribution as the joint form; the scores are what is allowed to move.
    """
    n = len(children)
    alloc = np.zeros(n, dtype=np.int64)
    if X_parent == 0:
        return alloc

    remaining = int(X_parent)
    for i in range(n):
        if remaining <= 0:
            break
        rest = children[i:]
        scores = compute_count_scores(
            rest,
            s_seq,
            H_traj,
            ctx,
            X_parent,
            Y_parent,
            X_fine,
            Y_fine,
            ell,
            p,
            parent_sets_ell,
        )
        z = scores - scores.max()
        omega = np.exp(z)
        omega /= omega.sum()
        drawn = sequential_split(
            remaining, float(omega[0]), float(omega[1:].sum()), p.kappa, rng
        )
        alloc[i] = drawn
        remaining -= drawn
        write(children[i], drawn)
    return alloc


def allocate_counts_block(
    c: int,
    children: List[int],
    ell: int,
    X_parent: int,
    Y_parent: int,
    s_seq: np.ndarray,
    H_traj: np.ndarray,
    ctx: np.ndarray,
    X_fine: np.ndarray,
    Y_fine: np.ndarray,
    p: FineCountParams,
    parent_sets_ell: Set[LagKey],
    rng: np.random.Generator,
    caps: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Allocate X_parent events across children of block c for label ell.

    Writes result directly into X_fine[children, ell] and returns it.
    """
    if X_parent == 0:
        return np.zeros(len(children), dtype=np.int64)

    scores = compute_count_scores(
        children,
        s_seq,
        H_traj,
        ctx,
        X_parent,
        Y_parent,
        X_fine,
        Y_fine,
        ell,
        p,
        parent_sets_ell,
    )
    alloc = dirichlet_multinomial_sample(X_parent, scores, p.kappa, rng)
    if caps is not None:
        z = scores - scores.max()
        omega = np.exp(z)
        alloc = redistribute_excess(alloc, caps, omega / omega.sum(), rng)
    return alloc


def allocate_durations_block(
    c: int,
    children: List[int],
    ell: int,
    Y_parent: int,
    X_fine_children: np.ndarray,  # (n_children,) already allocated
    s_seq: np.ndarray,
    H_traj: np.ndarray,
    ctx: np.ndarray,
    X_fine: np.ndarray,
    Y_fine: np.ndarray,
    p: FineDurParams,
    parent_sets_ell: Set[LagKey],
    rng: np.random.Generator,
    floor_per_active: bool = False,
) -> np.ndarray:
    """
    Allocate Y_parent slot-duration units across active children of block c.

    Args:
        floor_per_active: give every active child at least as many duration
            units as it has events, since one event occupies at least the slot
            it starts in.  The units above that floor are what the allocation
            model distributes.  The floor keeps Y >= X at every scale, which is
            what lets the next finer scale apply the same floor.

    Returns
    -------
    Y_children : (n_children,) int array summing to Y_parent.
    """
    n_ch = len(children)
    Y_children = np.zeros(n_ch, dtype=np.int64)

    if Y_parent == 0:
        return Y_children

    active_mask = X_fine_children > 0
    active_idx = [i for i, m in enumerate(active_mask) if m]
    active_t = [children[i] for i in active_idx]

    if not active_t:
        # No active children: duration cannot be allocated - set to zero
        return Y_children

    scores = compute_duration_scores(
        active_t,
        s_seq,
        H_traj,
        ctx,
        Y_parent,
        X_fine,
        Y_fine,
        ell,
        p,
        parent_sets_ell,
    )
    if floor_per_active:
        floors = np.array([int(X_fine_children[i]) for i in active_idx], dtype=np.int64)
    else:
        floors = np.zeros(len(active_idx), dtype=np.int64)

    free = int(Y_parent) - int(floors.sum())
    if free < 0:
        # The parent holds fewer duration units than its own count, which the
        # coarsest-scale draw rules out; fall back to a proportional split.
        share = np.floor(floors * (int(Y_parent) / max(floors.sum(), 1))).astype(
            np.int64
        )
        for extra in range(int(Y_parent) - int(share.sum())):
            share[extra % len(share)] += 1
        for j, i in enumerate(active_idx):
            Y_children[i] = share[j]
        return Y_children

    alloc = dirichlet_multinomial_sample(free, scores, p.tau, rng)
    for j, i in enumerate(active_idx):
        Y_children[i] = alloc[j] + floors[j]

    return Y_children
