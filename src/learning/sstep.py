"""
learning/sstep.py
=================
S-step: update the graph G_k via sparse local structure search.

Algorithm
---------
For each label ℓ ∈ L at scale k, find the parent set P_{k,ℓ} that maximises:
    Q_{k,ℓ}(P; q, Θ) − λ_k |P|
where Q_{k,ℓ} is the expected local log-likelihood (Eq. 727-738).

Search strategies
-----------------
* LocalScore      - evaluates Q_{k,ℓ}(P) for a given candidate set P.
* GreedyForwardBackward - greedy add / remove / swap with max-indegree constraint.
* BeamSearch      - beam-search over parent sets (alternative).
* ExactSearch     - enumerates every parent set inside the in-degree bound.
* GeneticSearch   - evolutionary search, for spaces too large to enumerate.
* SStep.run       - dispatches per-label, updates MultiScaleDBNGraph.

Greedy accepts one edge at a time and so cannot reach a pair of edges that only
pays jointly, which is why the exact and evolutionary searches exist.  On both
benchmarks in this repository all three return the same parent sets, so on
those the greedy answer is already the certified per-target optimum and the
search is not what limits recovery.

Mathematical references
-----------------------
* Local score:  Eq. 727-750 of the manuscript.
* Penalty:      Eq. 511  Ω_G(G) = Σ_{k,ℓ} |P_{k,ℓ}|

Two properties make this score usable for search (both were missing before)
--------------------------------------------------------------------------
1. **Profiled coefficients.**  Q_{k,ℓ}(P) is the local expected log-likelihood
   *maximised over the lag coefficients of P*.  Scoring a candidate edge with
   whatever coefficient happens to sit in Θ (zero, for an edge not yet in the
   graph) makes every addition score exactly −λ, so no edge can ever enter the
   graph - the search could only delete.  The fitted coefficients are written
   back into the parameters, so a newly added edge starts life useful.
2. **Normalised finer-scale likelihood.**  At k < K the score is the
   multinomial-limit Dir-Mult log-likelihood *including the block
   log-normaliser* (see `estep.build_log_like_fine`).  The unnormalised
   Σ_t X_t η_t used before is linear in the parameters and therefore unbounded:
   its arg-max is meaningless, which is why structure recovery collapsed.
"""

from __future__ import annotations

from copy import deepcopy
from itertools import combinations, product
from math import comb
from typing import Callable, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.special import gammaln

from ..core.blocks import BlockIndex
from ..core.graph import MultiScaleDBNGraph
from ..core.mask import declared_rows, parent_set_lag
from ..core.scales import ScaleHierarchy
from ..core.variables import MultiScaleData
from ..inference.posterior import PosteriorSummaries
from ..model.emissions import _sigmoid, nb_log_pmf, zinb_log_pmf
from ..model.params import ScaleParams
from .estep import (
    _LOG_CLIP,
    _agg_lag_X,
    _agg_lag_Y,
    block_softmax,
    coarse_linear_predictors,
    dirmult_grad,
    dirmult_ll,
    fine_scores,
    lag_contrib,
    recorded_and_scored,
    rows_of,
)

LagKey = Tuple[int, int]


# ---------------------------------------------------------------------------
# Low-dimensional profile maximisation
# ---------------------------------------------------------------------------


def _maximise(
    fn: Callable[[np.ndarray], float],
    grad_fn: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    n_iter: int = 20,
    step0: float = 0.5,
    tol: float = 1e-7,
) -> Tuple[np.ndarray, float]:
    """
    Maximise a smooth low-dimensional objective by normalised-gradient ascent
    with an adaptive step size.

    Used to profile out the (≤ max_indegree) lag coefficients of a candidate
    parent set.  Only accepts steps that improve the objective, so the returned
    value is never worse than `fn(x0)` - which keeps the greedy search
    monotone.
    """
    x = np.asarray(x0, dtype=float).copy()
    f = float(fn(x))
    if x.size == 0:
        return x, f
    step = step0
    for _ in range(n_iter):
        g = np.asarray(grad_fn(x), dtype=float)
        gnorm = float(np.linalg.norm(g))
        if not np.isfinite(gnorm) or gnorm < 1e-10:
            break
        direction = g / gnorm
        improved = False
        for _ls in range(8):
            x_new = x + step * direction
            f_new = float(fn(x_new))
            if np.isfinite(f_new) and f_new > f + tol:
                x, f = x_new, f_new
                step *= 1.6
                improved = True
                break
            step *= 0.4
        if not improved:
            break
    return x, f


# ---------------------------------------------------------------------------
# Per-label scorers (vectorised, with profiled lag coefficients)
# ---------------------------------------------------------------------------


class _CoarseLabelScorer:
    """
    Penalised local score of one label at the coarsest scale.

    Everything that does not depend on the candidate parent set (regime/factor/
    context contributions, the ZINB and NB constants) is computed once; scoring
    a parent set then costs a handful of (T, M) array operations.

    Every term the score sums is weighted by γ, so an epoch outside `scored`
    is dropped by carrying a zero weight into all of them.
    """

    def __init__(
        self,
        ell: int,
        X_K: np.ndarray,
        Y_K: np.ndarray,
        q: PosteriorSummaries,
        ctx: np.ndarray,
        params: ScaleParams,
        lam: float,
        candidates: Sequence[LagKey],
        profile_steps: int = 20,
        free_intercept: bool = False,
        scored: Optional[np.ndarray] = None,
    ) -> None:
        self.ell = ell
        self.lam = lam
        self.profile_steps = profile_steps
        self.free_intercept = free_intercept
        self.gamma = np.where(
            rows_of(scored, X_K.shape[0])[:, None], q.gamma, 0.0
        )  # (T, M)
        cp, dp = params.coarse_count, params.coarse_dur
        L = params.L
        T = X_K.shape[0]

        # Base predictors for this label, excluding its own lag terms.
        zero_lag = np.zeros((T, L))
        log_mu, logit_pi, log_nu = coarse_linear_predictors(
            X_K, q.m_H, ctx, params, zero_lag, zero_lag
        )
        self.base_mu = log_mu[:, :, ell]  # (T, M)
        self.base_nu = log_nu[:, :, ell]  # (T, M)
        self.pi = _sigmoid(logit_pi[:, :, ell])  # (T, M)
        self.r_disp = float(cp.r_disp[ell])
        self.s_disp = float(max(dp.s_disp[ell], 0.05))
        self.x = X_K[:, ell].astype(float)[:, None]  # (T, 1)
        self.y = Y_K[:, ell].astype(float)[:, None]  # (T, 1)
        self.active = (X_K[:, ell] > 0)[:, None]  # (T, 1)

        # Design columns, built on demand (the current parent set may hold
        # edges outside the candidate list, e.g. a longer true max_lag).
        self.feat_x = _LagColumns(X_K)
        self.feat_y = _LagColumns(Y_K)
        self.b_init = {
            key: float(cp.betaLag.get(key, np.zeros(L))[ell]) for key in candidates
        }
        self.a_init = {
            key: float(dp.alphaLag.get(key, np.zeros(L))[ell]) for key in candidates
        }
        # Fitted coefficients of the last scored parent set.
        self.fitted_b: Dict[LagKey, float] = {}
        self.fitted_a: Dict[LagKey, float] = {}

    # -- count part ----------------------------------------------------
    def _count_ll(self, offset: np.ndarray) -> float:
        mu = np.exp(np.clip(self.base_mu + offset, -_LOG_CLIP, _LOG_CLIP))
        ll = zinb_log_pmf(self.x, self.pi, self.r_disp, mu)
        return float(np.sum(self.gamma * ll))

    def _count_grad_logmu(self, offset: np.ndarray) -> np.ndarray:
        mu = np.exp(np.clip(self.base_mu + offset, -_LOG_CLIP, _LOG_CLIP))
        rv = max(self.r_disp, 1e-4)
        nb_p = rv / (rv + mu)
        dnb = self.x - (rv + self.x) * (mu / (rv + mu))
        nb0 = rv * np.log(nb_p + 1e-15)
        log_p0 = np.logaddexp(np.log(self.pi + 1e-15), np.log1p(-self.pi + 1e-15) + nb0)
        zero_share = np.exp(np.log1p(-self.pi + 1e-15) + nb0 - log_p0)
        dnb0 = -rv * mu / (rv + mu)
        return np.where(self.x == 0, zero_share * dnb0, dnb)

    # -- duration part -------------------------------------------------
    def _dur_ll(self, offset: np.ndarray) -> float:
        nu = np.maximum(
            np.exp(np.clip(self.base_nu + offset, -_LOG_CLIP, _LOG_CLIP)), 1.0
        )
        ll = np.where(self.active, nb_log_pmf(self.y, self.s_disp, nu), 0.0)
        return float(np.sum(self.gamma * ll))

    def _dur_grad_lognu(self, offset: np.ndarray) -> np.ndarray:
        nu = np.maximum(
            np.exp(np.clip(self.base_nu + offset, -_LOG_CLIP, _LOG_CLIP)), 1.0
        )
        s = self.s_disp
        return np.where(self.active, self.y - (s + self.y) * nu / (s + nu), 0.0)

    # ------------------------------------------------------------------
    def score(self, parent_set: Set[LagKey]) -> float:
        keys = sorted(parent_set)
        ll_c, b = _profile_block(
            keys,
            [self.feat_x[k] for k in keys],
            self.b_init,
            self.gamma,
            self._count_ll,
            self._count_grad_logmu,
            self.profile_steps,
            self.free_intercept,
        )
        ll_d, a = _profile_block(
            keys,
            [self.feat_y[k] for k in keys],
            self.a_init,
            self.gamma,
            self._dur_ll,
            self._dur_grad_lognu,
            self.profile_steps,
            self.free_intercept,
        )
        self.fitted_b = dict(zip(keys, b))
        self.fitted_a = dict(zip(keys, a))
        return ll_c + ll_d - self.lam * len(parent_set)


class _FineLabelScorer:
    """
    Penalised local score of one label at a finer scale k < K.

    An epoch outside `recorded` leaves the block competition and carries a zero
    count, so the score is the allocation of what the record holds.  An epoch
    that is recorded and outside `scored` keeps its place in the competition
    and carries a zero count, which drops its own term and leaves the term's
    gradient at zero, so every parent set is compared on one row set.
    """

    def __init__(
        self,
        ell: int,
        X_k: np.ndarray,
        Y_k: np.ndarray,
        X_par: np.ndarray,
        Y_par: np.ndarray,
        q: PosteriorSummaries,
        ctx: np.ndarray,
        params: ScaleParams,
        hier: ScaleHierarchy,
        k: int,
        lam: float,
        candidates: Sequence[LagKey],
        profile_steps: int = 20,
        recorded: Optional[np.ndarray] = None,
        scored: Optional[np.ndarray] = None,
    ) -> None:
        self.ell = ell
        self.lam = lam
        self.profile_steps = profile_steps
        self.gamma = q.gamma  # (T, M)
        self.block = BlockIndex.from_hierarchy(hier, k)
        L = params.L
        T = X_k.shape[0]

        zero_lag = np.zeros((T, L))
        eta, xi = fine_scores(
            X_k, X_par, Y_par, q.m_H, ctx, params, self.block, zero_lag, zero_lag
        )
        self.base_eta = eta[:, :, ell]  # (T, M)
        self.base_xi = xi[:, :, ell]  # (T, M)
        # Regime-averaged base scores: fixed while profiling a parent set.
        self.base_eta_bar = np.sum(q.gamma * self.base_eta, axis=1)[:, None]
        self.base_xi_bar = np.sum(q.gamma * self.base_xi, axis=1)[:, None]
        kept, rows = recorded_and_scored(recorded, scored, T)
        summed = rows[:, None]
        self.seen = kept[:, None]
        self.x = np.where(summed, X_k[:, ell].astype(float)[:, None], 0.0)
        self.y = Y_k[:, ell].astype(float)[:, None]
        self.active = (X_k[:, ell] > 0)[:, None] & self.seen
        self.y_act = np.where(self.active & summed, self.y, 0.0)
        self.has_durations = bool(np.any(self.y_act > 0))
        self.kappa = float(params.fine_count.kappa)
        self.tau = float(params.fine_dur.tau)

        self.feat_x = _LagColumns(X_k)
        self.feat_y = _LagColumns(Y_k)
        fp, fdp = params.fine_count, params.fine_dur
        self.b_init = {
            key: float(fp.betaLag.get(key, np.zeros(L))[ell]) for key in candidates
        }
        self.a_init = {
            key: float(fdp.alphaLag.get(key, np.zeros(L))[ell]) for key in candidates
        }
        self.fitted_b: Dict[LagKey, float] = {}
        self.fitted_a: Dict[LagKey, float] = {}

    # ------------------------------------------------------------------
    def _omega(self, base_bar, offset, mask):
        """Block allocation probabilities at the regime-averaged score: (T, 1)."""
        return block_softmax(base_bar + offset, self.block, mask)

    def _count_ll(self, offset: np.ndarray) -> float:
        omega = self._omega(self.base_eta_bar, offset, self.seen)
        return float(np.sum(dirmult_ll(self.x, omega, self.kappa)))

    def _count_grad(self, offset: np.ndarray) -> np.ndarray:
        # d(ll)/d(η̄_t); _profile_block contracts it with γ (which sums to 1).
        omega = self._omega(self.base_eta_bar, offset, self.seen)
        g = dirmult_grad(self.x, omega, self.kappa, self.block)
        return np.broadcast_to(g, self.base_eta.shape)

    def _dur_ll(self, offset: np.ndarray) -> float:
        omega = self._omega(self.base_xi_bar, offset, self.active)
        return float(np.sum(dirmult_ll(self.y_act, omega, self.tau)))

    def _dur_grad(self, offset: np.ndarray) -> np.ndarray:
        omega = self._omega(self.base_xi_bar, offset, self.active)
        g = dirmult_grad(self.y_act, omega, self.tau, self.block)
        return np.broadcast_to(g, self.base_xi.shape)

    def score(self, parent_set: Set[LagKey]) -> float:
        keys = sorted(parent_set)
        ll_c, b = _profile_block(
            keys,
            [self.feat_x[k] for k in keys],
            self.b_init,
            self.gamma,
            self._count_ll,
            self._count_grad,
            self.profile_steps,
        )
        if self.has_durations:
            ll_d, a = _profile_block(
                keys,
                [self.feat_y[k] for k in keys],
                self.a_init,
                self.gamma,
                self._dur_ll,
                self._dur_grad,
                self.profile_steps,
            )
        else:
            ll_d, a = 0.0, np.zeros(len(keys))
        self.fitted_b = dict(zip(keys, b))
        self.fitted_a = dict(zip(keys, a))
        return ll_c + ll_d - self.lam * len(parent_set)


# ---------------------------------------------------------------------------
# Shared profiling helper
# ---------------------------------------------------------------------------


def _lag_column(arr: np.ndarray, key: LagKey) -> np.ndarray:
    """log(1 + arr[t-d, src]) as a (T, 1) column (0 for t < d)."""
    src, d = key
    T = arr.shape[0]
    col = np.zeros((T, 1))
    if 0 < d < T:
        col[d:, 0] = np.log1p(arr[:-d, src].astype(float))
    return col


def _null_threshold(
    scorer,
    keys: Sequence[LagKey],
    rng: np.random.Generator,
    n_perm: int,
    quantile: float,
) -> float:
    """
    Gain a candidate has to beat to look like more than noise.

    A lagged feature is scored again after a circular shift of its rows.  The
    shift keeps the marginal distribution and the autocorrelation of the
    feature, and destroys only its alignment with the target, so the gains it
    produces are what a feature of that shape is worth when it carries no
    information about this label.  Shuffling the rows instead would also destroy
    the autocorrelation, and would understate the floor: a lagged feature is
    attractive to the score precisely because it is autocorrelated, which is
    also what the latent regime explains.

    The gains are taken with the penalty switched off, because the caller
    compares this floor against the penalty itself.  `score` subtracts
    `lam` per parent, so measuring through it would return the null gain less
    one `lam`, which is a different quantity and can be negative.
    """
    if not keys:
        return 0.0
    n_rows = scorer.feat_x[keys[0]].shape[0]
    if n_rows < 8:
        return 0.0
    lam_held = scorer.lam
    scorer.lam = 0.0
    try:
        base = scorer.score(set())
        gains = []
        for i in range(n_perm):
            donor = keys[i % len(keys)]
            fake: LagKey = (-1, i)
            shift = int(rng.integers(n_rows // 8, n_rows - n_rows // 8))
            scorer.feat_x.set(fake, np.roll(scorer.feat_x[donor], shift, axis=0))
            scorer.feat_y.set(fake, np.roll(scorer.feat_y[donor], shift, axis=0))
            scorer.b_init[fake] = 0.0
            scorer.a_init[fake] = 0.0
            gains.append(scorer.score({fake}) - base)
    finally:
        scorer.lam = lam_held
    return float(np.quantile(gains, quantile))


def _block_roll(
    col: np.ndarray,
    block_ids: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Roll a feature column independently inside each parent block.

    The roll keeps which blocks carry the feature's mass, so whatever a
    candidate earns from within-block co-movement survives into the null, and
    destroys the epoch-level alignment, which is the only thing a real lagged
    edge adds beyond that co-movement.
    """
    out = col.copy()
    starts = np.flatnonzero(np.r_[True, block_ids[1:] != block_ids[:-1]])
    ends = np.r_[starts[1:], block_ids.size]
    for s, e in zip(starts, ends):
        n = e - s
        if n > 1:
            out[s:e] = np.roll(col[s:e], int(rng.integers(1, n)), axis=0)
    return out


def _shift_gains(
    scorer,
    edge: LagKey,
    base: Set[LagKey],
    rng: np.random.Generator,
    n_perm: int,
    base_score: Optional[float] = None,
    block_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Gains of shifted copies of one candidate feature, over one base set.

    Each draw replaces the candidate's feature with a copy of itself whose
    alignment with the target has been destroyed, and scores it against the
    same base set the real candidate is scored against.  Without `block_ids`
    the copy is a whole-series circular shift, which keeps the feature's
    marginal distribution and autocorrelation.  With `block_ids` the copy is
    rolled inside each parent block, which additionally keeps the feature's
    block membership, so gains that come from the shared block total, from a
    persistent latent state, or from the realized allocation split of a block
    appear in the null too, and only epoch-level alignment separates a real
    edge from it.
    """
    if base_score is None:
        base_score = scorer.score(set(base))
    n_rows = scorer.feat_x[edge].shape[0]
    gains = []
    for i in range(n_perm):
        fake: LagKey = (-1, i)
        if block_ids is not None:
            fx = _block_roll(scorer.feat_x[edge], block_ids, rng)
            fy = _block_roll(scorer.feat_y[edge], block_ids, rng)
        else:
            shift = int(rng.integers(n_rows // 8, n_rows - n_rows // 8))
            fx = np.roll(scorer.feat_x[edge], shift, axis=0)
            fy = np.roll(scorer.feat_y[edge], shift, axis=0)
        scorer.feat_x.set(fake, fx)
        scorer.feat_y.set(fake, fy)
        scorer.b_init[fake] = 0.0
        scorer.a_init[fake] = 0.0
        gains.append(scorer.score(set(base) | {fake}) - base_score)
    return np.asarray(gains)


def _confirm_edges(
    score_fn,
    scorer,
    parent_set: Set[LagKey],
    rng: np.random.Generator,
    n_perm: int,
    quantile: float,
    block_ids: Optional[np.ndarray] = None,
) -> Set[LagKey]:
    """
    Drop each edge whose gain a shifted copy of its own feature matches.

    The search accepts an edge when its gain clears the penalty, and at a
    scale where a latent state or the shared block total autocorrelates every
    label, a feature with no information about the target still clears it.
    This test asks the sharper question: does the edge add more than a
    shifted copy of the same feature adds to the same base set?  At a finer
    scale the copy is rolled inside each parent block, so the null carries the
    whole block-level signal and only epoch-level alignment can save an edge.
    It runs per accepted edge rather than per candidate, so it costs
    `n_perm` extra scores per edge the search proposes, and removing one
    edge re-checks the survivors, whose gains change with the base set.
    """
    kept = set(parent_set)
    if not kept:
        return kept
    n_rows = scorer.feat_x[next(iter(kept))].shape[0]
    if n_rows < 8:
        return kept
    changed = True
    while changed and kept:
        changed = False
        for edge in sorted(kept):
            base = kept - {edge}
            base_score = score_fn(base)
            gain = score_fn(kept) - base_score
            null = _shift_gains(scorer, edge, base, rng, n_perm, base_score, block_ids)
            if gain <= float(np.quantile(null, quantile)):
                kept = base
                changed = True
                break
    return kept


def candidate_floors(
    scorer,
    edge: LagKey,
    base: Set[LagKey],
    rng: np.random.Generator,
    n_perm: int = 32,
    quantile: float = 0.95,
    block_ids: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Both bars one candidate edge has to clear, by the paths the search uses.

    A diagnostic that recomputes a floor of its own can disagree with the fit
    and send a change the wrong way, which is what happened once already.  This
    calls the same scorer and the same shifted null the search calls, so the two
    cannot drift apart.

    The returned gain is penalised, as the search sees it, so the penalty bar is
    zero rather than a number to reconstruct.  `binding` names the higher bar,
    or `"none"` when the edge clears both.
    """
    base_set = set(base)
    base_score = scorer.score(base_set)
    gain = scorer.score(base_set | {edge}) - base_score
    null = _shift_gains(scorer, edge, base_set, rng, n_perm, base_score, block_ids)
    shift_floor = float(np.quantile(null, quantile))
    if gain <= 0.0:
        binding = "penalty"
    elif gain <= shift_floor:
        binding = "confirmation"
    else:
        binding = "none"
    return {
        "gain": float(gain),
        "penalty_floor": 0.0,
        "shift_floor": shift_floor,
        "binding": binding,
    }


class _LagColumns:
    """Memoised `log(1 + arr[t-d, src])` design columns, keyed by lagged edge."""

    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr
        self._cache: Dict[LagKey, np.ndarray] = {}

    def __getitem__(self, key: LagKey) -> np.ndarray:
        col = self._cache.get(key)
        if col is None:
            col = _lag_column(self._arr, key)
            self._cache[key] = col
        return col

    def set(self, key: LagKey, column: np.ndarray) -> None:
        """Install a column directly, used for permutation calibration."""
        self._cache[key] = column


def _profile_block(
    keys: Sequence[LagKey],
    columns: Sequence[np.ndarray],  # each (T, 1)
    init: Dict[LagKey, float],
    gamma: np.ndarray,  # (T, M)
    ll_fn,  # offset (T, M) to float
    grad_fn,  # offset (T, M) to (T, M) d ll / d predictor
    n_steps: int,
    free_intercept: bool = False,
) -> Tuple[float, np.ndarray]:
    """
    Maximise the local log-likelihood over the coefficients of `keys`.

    Returns `(max_ll, coefficients)`.  With no parents this is just the
    log-likelihood of the current predictor.

    With `free_intercept` a constant column is fitted beside them and dropped
    from the result, so every parent set is scored at its own best level.  The
    level is free under both hypotheses of the comparison, and holding it at
    the value fitted without a parent charges the parent for a shift the level
    would otherwise absorb.  It is off by default: the argument for it is sound
    and the measurement is against it, because giving the no-parent hypothesis
    a free level lets it absorb what a true edge explains when the level was
    already right.  It is meaningless below the coarsest scale in any case,
    where the allocation softmax runs over the children of one block and
    cancels every term that is constant inside it.
    """
    if not keys and not free_intercept:
        return ll_fn(0.0), np.zeros(0)

    cols = list(columns)
    start = [init.get(key, 0.0) for key in keys]
    if free_intercept:
        cols = [np.ones((gamma.shape[0], 1))] + cols
        start = [0.0] + start
    F = np.concatenate(cols, axis=1)  # (T, p)
    b0 = np.array(start, dtype=float)

    def _obj(b: np.ndarray) -> float:
        return ll_fn((F @ b)[:, None])

    def _grad(b: np.ndarray) -> np.ndarray:
        g = grad_fn((F @ b)[:, None])  # (T, M)
        return F.T @ np.sum(gamma * g, axis=1)  # (p,)

    b_hat, f_hat = _maximise(_obj, _grad, b0, n_iter=n_steps)
    return f_hat, (b_hat[1:] if free_intercept else b_hat)


# ---------------------------------------------------------------------------
# Local score (public API - thin wrappers around the scorers above)
# ---------------------------------------------------------------------------


class LocalScore:
    """
    Evaluate the penalised local expected log-likelihood for one (k, ℓ) pair.

    For the coarsest scale:
        Q_{K,ℓ}(P) = max_β Σ_t Σ_m γ_t(m) [log ZINB(X_{t,ℓ}|P) + log NB(Y_{t,ℓ}|P)]

    For finer scales:
        Q_{k,ℓ}(P) = max_β Σ_t Σ_m γ_t(m) [ X_{t,ℓ}(η_{t,ℓ,m} − logZ_c)
                                          + Y_{t,ℓ}(ξ_{t,ℓ,m} − logZ^Y_c) ]

    Penalty:  −λ |P|.
    """

    @staticmethod
    def compute_coarsest(
        parent_set: Set[LagKey],
        ell: int,
        X_K: np.ndarray,  # (T, L)
        Y_K: np.ndarray,  # (T, L)
        q: PosteriorSummaries,
        ctx: np.ndarray,  # (T, d_ctx)
        params: ScaleParams,
        lam: float = 1.0,
        free_intercept: bool = False,
        scored: Optional[np.ndarray] = None,
    ) -> float:
        """Penalised local score for label ℓ at coarsest scale K."""
        scorer = _CoarseLabelScorer(
            ell,
            X_K,
            Y_K,
            q,
            ctx,
            params,
            lam,
            sorted(parent_set),
            free_intercept=free_intercept,
            scored=scored,
        )
        return scorer.score(set(parent_set))

    @staticmethod
    def compute_fine(
        parent_set: Set[LagKey],
        ell: int,
        X_k: np.ndarray,  # (H_k, L)
        Y_k: np.ndarray,  # (H_k, L)
        X_par: np.ndarray,  # (H_{k+1}, L)
        Y_par: np.ndarray,  # (H_{k+1}, L)
        q: PosteriorSummaries,
        ctx: np.ndarray,  # (H_k, d_ctx)
        params: ScaleParams,
        hier: ScaleHierarchy,
        k: int,
        lam: float = 1.0,
        recorded: Optional[np.ndarray] = None,
        scored: Optional[np.ndarray] = None,
    ) -> float:
        """Penalised local score for label ℓ at finer scale k."""
        scorer = _FineLabelScorer(
            ell,
            X_k,
            Y_k,
            X_par,
            Y_par,
            q,
            ctx,
            params,
            hier,
            k,
            lam,
            sorted(parent_set),
            recorded=recorded,
            scored=scored,
        )
        return scorer.score(set(parent_set))


# ---------------------------------------------------------------------------
# Greedy forward-backward search
# ---------------------------------------------------------------------------


class GreedyForwardBackward:
    """
    Greedy forward-backward (hill-climbing) search over parent sets.

    At each step, considers all single-edge additions and removals
    (and optionally swaps), accepting whichever gives the largest
    score improvement.  Stops when no improvement is found.
    """

    def __init__(
        self,
        max_indegree: int = 3,
        max_iter: int = 50,
    ) -> None:
        self.max_indegree = max_indegree
        self.max_iter = max_iter

    def search(
        self,
        current_set: Set[LagKey],
        score_fn,  # callable(parent_set) -> float
        candidates: List[LagKey],  # all (src_ell, d) pairs to consider
    ) -> Set[LagKey]:
        """
        Run greedy forward-backward from `current_set`.

        Parameters
        ----------
        current_set : initial parent set
        score_fn    : score function mapping a parent set to a scalar
        candidates  : list of all candidate (src_ell, d) pairs

        Returns
        -------
        best_set : locally optimal parent set
        """
        best_set = set(current_set)
        best_score = score_fn(best_set)

        for _ in range(self.max_iter):
            improved = False
            # --- Try adding each candidate not yet in the set ---
            for edge in candidates:
                if edge in best_set:
                    continue
                if len(best_set) >= self.max_indegree:
                    continue
                candidate = best_set | {edge}
                s = score_fn(candidate)
                if s > best_score + 1e-8:
                    best_score = s
                    best_set = candidate
                    improved = True

            # --- Try removing each edge in the set ---
            for edge in list(best_set):
                candidate = best_set - {edge}
                s = score_fn(candidate)
                if s > best_score + 1e-8:
                    best_score = s
                    best_set = candidate
                    improved = True

            if not improved:
                break

        return best_set


# ---------------------------------------------------------------------------
# Exact and evolutionary search
# ---------------------------------------------------------------------------


def n_subsets(n_candidates: int, max_indegree: int) -> int:
    """Number of parent sets an exact search would enumerate."""
    total = 0
    for size in range(min(max_indegree, n_candidates) + 1):
        total += comb(n_candidates, size)
    return int(total)


class ExactSearch:
    """
    Enumerate every parent set inside the in-degree bound and keep the best.

    The penalized objective decomposes over targets and every candidate is
    lagged, so maximizing each target separately maximizes the whole and the
    bound keeps the subset count finite.  Greedy accepts one edge at a time and
    therefore cannot reach a pair that only pays jointly; enumeration has no
    such blind spot and returns the certified per-target optimum.
    """

    def __init__(self, max_indegree: int = 3) -> None:
        self.max_indegree = max_indegree

    def search(self, score_fn, candidates: List[LagKey]) -> Set[LagKey]:
        best_set: Set[LagKey] = set()
        best_score = score_fn(best_set)
        bound = min(self.max_indegree, len(candidates))
        for size in range(1, bound + 1):
            for combo in combinations(candidates, size):
                s = score_fn(set(combo))
                if s > best_score + 1e-9:
                    best_score = s
                    best_set = set(combo)
        return best_set


class GeneticSearch:
    """
    Evolutionary search over parent sets, for spaces too large to enumerate.

    An individual is a parent set inside the in-degree bound and its fitness is
    the penalized local score.  Crossover mixes two parent sets and mutation
    adds, drops or swaps one edge, so a pair of edges that only pays jointly
    can arrive in one step, which is what greedy cannot do.  The search is
    deterministic given `seed`.
    """

    def __init__(
        self,
        max_indegree: int = 3,
        population: int = 24,
        generations: int = 12,
        mutation_rate: float = 0.3,
        elite: int = 2,
        seed: int = 0,
    ) -> None:
        self.max_indegree = max_indegree
        self.population = population
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite = elite
        self.seed = seed

    def _random_set(self, rng, candidates) -> Set[LagKey]:
        bound = min(self.max_indegree, len(candidates))
        size = int(rng.integers(0, bound + 1))
        if size == 0:
            return set()
        idx = rng.choice(len(candidates), size=size, replace=False)
        return {candidates[i] for i in np.atleast_1d(idx)}

    def _mutate(self, rng, parent_set, candidates) -> Set[LagKey]:
        out = set(parent_set)
        inside = sorted(out)
        outside = [c for c in candidates if c not in out]
        room = len(out) < min(self.max_indegree, len(candidates))
        move = rng.random()
        drop = inside and (move < 0.34 or not outside)
        add = outside and room and not drop and (move < 0.67 or not inside)
        if drop:
            out.discard(inside[int(rng.integers(len(inside)))])
        elif add:
            out.add(outside[int(rng.integers(len(outside)))])
        elif inside and outside:
            out.discard(inside[int(rng.integers(len(inside)))])
            out.add(outside[int(rng.integers(len(outside)))])
        return out

    def _crossover(self, rng, a, b, candidates) -> Set[LagKey]:
        union = sorted(set(a) | set(b))
        if not union:
            return set()
        keep = [e for e in union if rng.random() < 0.5]
        bound = min(self.max_indegree, len(candidates))
        if len(keep) > bound:
            sel = rng.choice(len(keep), size=bound, replace=False)
            keep = [keep[i] for i in np.atleast_1d(sel)]
        return set(keep)

    def search(self, current_set, score_fn, candidates: List[LagKey]) -> Set[LagKey]:
        if not candidates:
            return set()
        rng = np.random.default_rng(self.seed)
        pop = [set(current_set), set()]
        while len(pop) < self.population:
            pop.append(self._random_set(rng, candidates))
        scored = [(score_fn(p), p) for p in pop]

        for _ in range(self.generations):
            scored.sort(key=lambda kv: -kv[0])
            survivors = [p for _, p in scored[: max(self.elite, 2)]]
            children: List[Set[LagKey]] = []
            seen = {frozenset(p) for p in survivors}
            while len(children) + len(survivors) < self.population:
                # Tournament of two keeps selection pressure without sorting
                # the whole population for every draw.
                i, j = rng.integers(len(scored)), rng.integers(len(scored))
                a = scored[i][1] if scored[i][0] >= scored[j][0] else scored[j][1]
                i, j = rng.integers(len(scored)), rng.integers(len(scored))
                b = scored[i][1] if scored[i][0] >= scored[j][0] else scored[j][1]
                child = self._crossover(rng, a, b, candidates)
                if rng.random() < self.mutation_rate:
                    child = self._mutate(rng, child, candidates)
                # A duplicate carries no information the population lacks, and
                # admitting one lets a single individual take over and stall the
                # search on a plateau.  Draw fresh instead.
                if frozenset(child) in seen:
                    child = self._random_set(rng, candidates)
                seen.add(frozenset(child))
                children.append(child)
            pop = survivors + children
            scored = [(score_fn(p), p) for p in pop]

        return max(scored, key=lambda kv: kv[0])[1]


# ---------------------------------------------------------------------------
# Beam search (alternative to greedy)
# ---------------------------------------------------------------------------


class BeamSearch:
    """
    Beam search over parent sets.

    Maintains a beam of `beam_width` candidate parent sets and
    expands each by one edge per round.
    """

    def __init__(
        self,
        beam_width: int = 5,
        max_indegree: int = 3,
        max_depth: int = 10,
    ) -> None:
        self.beam_width = beam_width
        self.max_indegree = max_indegree
        self.max_depth = max_depth

    def search(
        self,
        score_fn,
        candidates: List[LagKey],
    ) -> Set[LagKey]:
        """Search from the empty set."""
        # Beam: list of (score, parent_set)
        beam: List[Tuple[float, Set[LagKey]]] = [(score_fn(set()), set())]
        best: Tuple[float, Set[LagKey]] = (beam[0][0], set())

        for depth in range(self.max_depth):
            next_beam: List[Tuple[float, Set[LagKey]]] = []

            for _, parent_set in beam:
                if len(parent_set) >= self.max_indegree:
                    continue
                for edge in candidates:
                    if edge in parent_set:
                        continue
                    candidate = parent_set | {edge}
                    s = score_fn(candidate)
                    next_beam.append((s, candidate))

            if not next_beam:
                break

            # Keep top-beam_width unique sets by score
            next_beam.sort(key=lambda x: -x[0])
            if next_beam[0][0] > best[0]:
                best = (next_beam[0][0], set(next_beam[0][1]))
            seen: List[FrozenSet] = []
            filtered = []
            for s, ps in next_beam:
                fs = frozenset(ps)
                if fs not in seen:
                    seen.append(fs)
                    filtered.append((s, set(ps)))
                if len(filtered) >= self.beam_width:
                    break

            beam = filtered
            if all(len(ps) >= self.max_indegree for _, ps in beam):
                break

        return best[1]


# ---------------------------------------------------------------------------
# SStep: per-scale structure search
# ---------------------------------------------------------------------------


class SStep:
    """
    S-step: update the structural graph G_k at scale k.

    For each label ℓ ∈ L, searches for the parent set P_{k,ℓ} that
    maximises the penalised local score Q_{k,ℓ}(P) − λ_k |P|.

    Parameters
    ----------
    lam           : edge penalty coefficient
    penalty       : how the edge penalty is set.
                    `"bic"` (default) scales λ by log(T_k), the price of the
                    two coefficients an edge adds, so one `lam` behaves
                    sensibly at every scale and sample size.
                    `"calibrated"` raises that to the measured null
                    distribution of the score when the latter is higher: each
                    label scores `n_perm` permuted features, and the given
                    quantile of those gains becomes the floor.  It is the
                    setting to use when latent dynamics leave autocorrelation
                    that a lagged edge can absorb, which a fixed penalty cannot
                    anticipate.
                    `"fixed"` uses λ = lam as-is.
    confirm_edges : if True, every edge the search accepts must also beat the
                    given quantile of `n_perm` circularly shifted copies of
                    its own feature, scored against the same base set.  This
                    is the per-candidate form of the calibrated floor, applied
                    only to the edges the search proposes, so it costs
                    `n_perm` extra scores per accepted edge instead of per
                    label.  It is the guard against latent-state
                    autocorrelation and within-block co-movement being booked
                    as lagged edges at a scale that holds none.
    max_indegree  : maximum |P_{k,ℓ}|; int, or {scale: int} for per-scale budgets
    max_lag       : largest lag in the candidate set; int, or {scale: int}.
                    A search space narrower than the data-generating one caps
                    recall no matter how good the score is, so keep these in
                    step with the scenario (`EvaluationProtocol` does this
                    automatically).
    search        : `"greedy"` (default), `"beam"`, `"exact"`, `"genetic"`, or
                    `"auto"`, which enumerates when the subset count is within
                    `exact_budget` and runs the evolutionary search otherwise.
                    Greedy accepts one edge at a time and so cannot reach a
                    pair of edges that only pays jointly; the other two can.
    exact_budget  : largest number of parent sets `"auto"` will enumerate
    population    : individuals per generation in the evolutionary search
    generations   : generations of the evolutionary search
    profile_steps : gradient steps used to profile the lag coefficients of a
                    candidate parent set (0 disables profiling)
    """

    def __init__(
        self,
        lam: float = 1.0,
        max_indegree: int = 3,
        max_lag: int = 2,
        search: str = "greedy",  # "greedy" | "beam" | "exact" | "genetic" | "auto"
        exact_budget: int = 20000,
        population: int = 24,
        generations: int = 12,
        beam_width: int = 5,
        max_iter: int = 50,
        scoring_backend: str = "custom",  # "custom" | "pyagrum_bdeu" | "pyagrum_bic"
        n_bins: int = 5,
        penalty: str = "bic",  # "bic" | "calibrated" | "fixed"
        confirm_edges: bool = False,
        profile_steps: int = 20,
        n_perm: int = 16,
        perm_quantile: float = 0.95,
        seed: int = 0,
    ) -> None:
        self.lam = lam
        self.max_indegree = max_indegree
        self.max_lag = max_lag
        self.search = search
        self.exact_budget = exact_budget
        self.population = population
        self.generations = generations
        self.beam_width = beam_width
        self.max_iter = max_iter
        self.scoring_backend = scoring_backend
        self.n_bins = n_bins
        self.penalty = penalty
        self.confirm_edges = confirm_edges
        self.profile_steps = profile_steps
        self.n_perm = n_perm
        self.perm_quantile = perm_quantile
        self.seed = seed

    # ------------------------------------------------------------------
    @staticmethod
    def _per_scale(value, k: int) -> int:
        """Resolve an int-or-per-scale setting for scale k."""
        if isinstance(value, dict):
            return int(value[k])
        if isinstance(value, (list, tuple)):
            return int(value[k - 1])
        return int(value)

    # ------------------------------------------------------------------
    def _lam_effective(self, T: int, k: Optional[int] = None) -> float:
        """
        Edge penalty actually used at a scale with T epochs.

        `lam` may be a scalar or one value per scale.  A scalar cannot be right
        at every scale of this benchmark: the two scales that carry true edges
        reach their best structure four times apart, and no multiplier of
        `log T` spans that, since `log T` differs between them by 1.33 where the
        prices they ask for differ by 5.31.
        """
        lam = self._per_scale_float(self.lam, k)
        if self.penalty == "fixed":
            return lam
        # Each edge adds one count and one duration coefficient to BIC pays
        # 2 × ½ log T = log T per edge.
        return lam * float(np.log(max(T, 2)))

    @staticmethod
    def _per_scale_float(value, k: Optional[int]) -> float:
        """Resolve a float-or-per-scale penalty for scale k."""
        if isinstance(value, dict):
            return float(value[k])
        if isinstance(value, (list, tuple)):
            return float(value[k - 1])
        return float(value)

    # ------------------------------------------------------------------
    def run(
        self,
        k: int,
        q: PosteriorSummaries,
        params: ScaleParams,
        graph: MultiScaleDBNGraph,
        data: MultiScaleData,
        hier: Optional[ScaleHierarchy] = None,
    ) -> Tuple[MultiScaleDBNGraph, ScaleParams]:
        """
        Update G_k and the parent_sets in ScaleParams.

        Parameters
        ----------
        k      : 1-indexed scale
        q      : PosteriorSummaries from the E-step
        params : current ScaleParams for scale k
        graph  : current MultiScaleDBNGraph
        data   : full multiscale observed data

        Returns
        -------
        (new_graph, new_params) with updated parent sets
        """
        if hier is None:
            hier = data.hierarchy

        K = data.K
        L = params.L
        ctx_k = data.get_C(k)
        X_k = data.get_X(k)
        Y_k = data.get_Y(k)
        T = X_k.shape[0]
        new_graph = deepcopy(graph)
        new_params = deepcopy(params)
        lam_eff = self._lam_effective(T, k)
        max_lag = self._per_scale(self.max_lag, k)
        max_indegree = self._per_scale(self.max_indegree, k)

        # Candidate edges: (src_ell, lag) for src_ell in L, lag in 1..max_lag
        candidates: List[LagKey] = [
            (src, d) for src in range(L) for d in range(1, max_lag + 1)
        ]

        # Every parent set here is scored on one row set, eroded by the budget
        # the search declared.  Two sets scored on different rows are two
        # likelihoods of different data and their difference is not a gain, so
        # the erosion cannot read the model it is about to score.
        held = parent_set_lag(params.parent_sets)
        if held > max_lag:
            raise ValueError(
                f"scale {k} came in holding a lag of {held} while the search "
                f"budget is {max_lag}, so the rows the two would be scored on "
                "differ"
            )
        # The uneroded mask travels with the row set, since a finer-scale block
        # sum is over what the record holds.
        recorded = data.get_observed(k)
        rows = declared_rows(recorded, max_lag, "the S-step")

        # Optional pyAgrum scoring backend
        pyagrum_scorer = None
        if self.scoring_backend != "custom":
            from .pyagrum_scoring import PyAgrumScore

            score_map = {
                "pyagrum_bdeu": "BDeu",
                "pyagrum_bic": "BIC",
                "pyagrum_aic": "AIC",
                "pyagrum_k2": "K2",
            }
            score_type = score_map.get(self.scoring_backend, "BDeu")
            pyagrum_scorer = PyAgrumScore(
                X_k,
                graph.label_names,
                max_lag=max_lag,
                n_bins=self.n_bins,
                score_type=score_type,
            )

        if not params.is_coarsest:
            X_par = data.get_X(k + 1)
            Y_par = data.get_Y(k + 1)

        for ell in range(L):
            current_set = set(params.parent_sets.get(ell, set()))
            scorer = None

            if pyagrum_scorer is not None:

                def raw_score(ps, ell=ell):
                    return pyagrum_scorer.score_parent_set(ell, ps)

            elif params.is_coarsest:
                scorer = _CoarseLabelScorer(
                    ell,
                    X_k,
                    Y_k,
                    q,
                    ctx_k,
                    params,
                    lam_eff,
                    candidates,
                    profile_steps=self.profile_steps,
                    scored=rows,
                )
                raw_score = scorer.score
            else:
                scorer = _FineLabelScorer(
                    ell,
                    X_k,
                    Y_k,
                    X_par,
                    Y_par,
                    q,
                    ctx_k,
                    params,
                    hier,
                    k,
                    lam_eff,
                    candidates,
                    profile_steps=self.profile_steps,
                    recorded=recorded,
                    scored=rows,
                )
                raw_score = scorer.score

            if scorer is not None and self.penalty == "calibrated":
                rng = np.random.default_rng(self.seed + 1000 * k + ell)
                floor = _null_threshold(
                    scorer, candidates, rng, self.n_perm, self.perm_quantile
                )
                scorer.lam = max(lam_eff, floor)

            # Greedy/beam re-visit the same parent sets many times.
            cache: Dict[FrozenSet, float] = {}
            best_coeffs: Dict[FrozenSet, Tuple[dict, dict]] = {}

            def score_fn(
                ps, _raw=raw_score, _cache=cache, _sc=scorer, _coef=best_coeffs
            ):
                key = frozenset(ps)
                if key not in _cache:
                    _cache[key] = _raw(ps)
                    if _sc is not None:
                        _coef[key] = (dict(_sc.fitted_b), dict(_sc.fitted_a))
                return _cache[key]

            # Run search
            search = self.search
            if search == "auto":
                # Enumeration is exact and its cost is known in advance, so it
                # is preferred wherever the subset count is affordable.
                search = (
                    "exact"
                    if n_subsets(len(candidates), max_indegree) <= self.exact_budget
                    else "genetic"
                )
            if search == "greedy":
                best_set = GreedyForwardBackward(
                    max_indegree=max_indegree,
                    max_iter=self.max_iter,
                ).search(current_set, score_fn, candidates)
            elif search == "exact":
                best_set = ExactSearch(max_indegree=max_indegree).search(
                    score_fn, candidates
                )
            elif search == "genetic":
                best_set = GeneticSearch(
                    max_indegree=max_indegree,
                    population=self.population,
                    generations=self.generations,
                    seed=self.seed + 31 * k + ell,
                ).search(current_set, score_fn, candidates)
            else:
                best_set = BeamSearch(
                    beam_width=self.beam_width,
                    max_indegree=max_indegree,
                    max_depth=max_indegree,
                ).search(score_fn, candidates)

            # Confirm every accepted edge against its own shifted null.  At a
            # finer scale the null rolls the feature inside each parent block,
            # so within-block co-movement cannot save an edge.
            if scorer is not None and self.confirm_edges and best_set:
                block_ids = (
                    None if params.is_coarsest else np.asarray(hier.parent_maps[k - 1])
                )
                rng = np.random.default_rng(self.seed + 7919 * k + ell)
                best_set = _confirm_edges(
                    score_fn,
                    scorer,
                    best_set,
                    rng,
                    self.n_perm,
                    self.perm_quantile,
                    block_ids,
                )

            # Update parent sets
            new_params.parent_sets[ell] = best_set

            # Write the profiled coefficients of the winning set back into Θ so
            # a newly added edge starts from a fitted value, not from zero.
            fitted_b, fitted_a = best_coeffs.get(frozenset(best_set), ({}, {}))
            if new_params.is_coarsest:
                count_lag = new_params.coarse_count.betaLag
                dur_lag = new_params.coarse_dur.alphaLag
            else:
                count_lag = new_params.fine_count.betaLag
                dur_lag = new_params.fine_dur.alphaLag
            for key in best_set:
                if key not in count_lag:
                    count_lag[key] = np.zeros(L)
                if key not in dur_lag:
                    dur_lag[key] = np.zeros(L)
                if key in fitted_b:
                    count_lag[key][ell] = fitted_b[key]
                if key in fitted_a:
                    dur_lag[key][ell] = fitted_a[key]
            # Drop this label's coefficients for edges that were removed.
            for key, coef in count_lag.items():
                if key not in best_set:
                    coef[ell] = 0.0
            for key, coef in dur_lag.items():
                if key not in best_set:
                    coef[ell] = 0.0

            # Sync MultiScaleDBNGraph  (API: add_edge / remove_edge take lag before dst)
            # Remove old edges for this label
            for src_ell, d in list(graph.get_parents(k, ell)):
                new_graph.remove_edge(k, src_ell, d, ell)
            # Add new edges
            for src_ell, d in best_set:
                new_graph.add_edge(k, src_ell, d, ell)

        return new_graph, new_params
