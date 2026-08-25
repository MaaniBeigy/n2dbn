"""
model/params.py
===============
Parameter dataclasses for each component of the N²DBN.

Design notes
------------
* One `ScaleParams` object per scale holds everything the generator and
  learning procedures need for that scale.
* Coarsest scale (k=K) uses ZINB count + NB duration emissions.
* Finer scales (k<K) use Dir-Mult allocation scores for counts and durations.
* When r=0, all factor-related arrays are zero-dimensional and factor
  contributions vanish automatically via matrix multiplication.
* When M=1, regime switching is absent; the single-regime case is the
  stationary baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

LagKey = Tuple[int, int]  # (src_label_idx, lag)


# ---------------------------------------------------------------------------
# Sub-parameter blocks
# ---------------------------------------------------------------------------


@dataclass
class RegimeParams:
    """
    Context-dependent Markov chain for latent regime S_t^(k).

    Transition logit:
        a_m + A[m, m'] + C[m, n]  (n = parent regime, only for k < K)
             + b_m^T  ctx_t

    Attributes
    ----------
    M        : number of regimes
    parent_M : number of regimes at parent scale (0 if k=K)
    d_ctx    : context dimension
    init     : (M,) unnormalised log-probabilities for t=1 regime
    bias     : (M,) a_m  per-regime bias
    trans    : (M, M) A[m, m']  prev-regime logit  (rows = current, cols = prev)
    cross    : (M, parent_M) C[m, n]  cross-scale logit  (None if k=K)
    ctx      : (M, d_ctx) b_m  context weights
    """

    M: int
    parent_M: int
    d_ctx: int
    init: np.ndarray  # (M,)
    bias: np.ndarray  # (M,)
    trans: np.ndarray  # (M, M)
    cross: Optional[np.ndarray]  # (M, parent_M) or None
    ctx: np.ndarray  # (M, d_ctx)

    def logit_transition(
        self,
        prev_m: int,
        parent_n: Optional[int],
        ctx_t: np.ndarray,
    ) -> np.ndarray:
        """
        Compute unnormalised logits for S_t = m.

        Returns
        -------
        logits : (M,) array - apply softmax to get transition probabilities.
        """
        logits = self.bias + self.trans[:, prev_m]
        if parent_n is not None and self.cross is not None:
            logits = logits + self.cross[:, parent_n]
        if ctx_t.shape[0] > 0:
            logits = logits + self.ctx @ ctx_t
        return logits

    def logit_init(self, ctx_t: np.ndarray) -> np.ndarray:
        """Unnormalised logits for the initial regime at t=0."""
        logits = self.init.copy()
        if ctx_t.shape[0] > 0:
            logits = logits + self.ctx @ ctx_t
        return logits


# ---------------------------------------------------------------------------
@dataclass
class FactorParams:
    """
    Regime-dependent linear Gaussian state-space model for H_t^(k).

    H_t = F[s] H_{t-1}  +  G[s] ctx_t  +  ε_t,   ε_t ~ N(0, Σ[s])
    H_0 ~ N(mu0, Sigma0)

    When r=0, all arrays are zero-dimensional; factor contributions are
    zero automatically.

    Attributes
    ----------
    M      : number of regimes
    r      : factor dimension (0 = no factor)
    d_ctx  : context dimension
    F      : (M, r, r)  transition matrices per regime
    G      : (M, r, d_ctx) input matrices per regime
    Sigma  : (M, r, r) noise covariances per regime
    mu0    : (r,)  initial mean
    Sigma0 : (r, r) initial covariance
    """

    M: int
    r: int
    d_ctx: int
    F: np.ndarray  # (M, r, r)
    G: np.ndarray  # (M, r, d_ctx)
    Sigma: np.ndarray  # (M, r, r)
    mu0: np.ndarray  # (r,)
    Sigma0: np.ndarray  # (r, r)


# ---------------------------------------------------------------------------
@dataclass
class CoarseCountParams:
    """
    ZINB count emission at the coarsest scale.

    log μ_t = β_0  +  β_S · onehot(S_t)  +  β_H · H_t  +  β_ctx · ctx_t
              + Σ β_lag[(ℓ',d)] · log(1 + X_{t-d}^{ℓ'})
    logit π_t = γ_0 + γ_S · onehot(S_t) + γ_H · H_t + γ_ctx · ctx_t

    Attributes
    ----------
    L      : number of labels
    beta0  : (L,)
    betaS  : (M, L)
    betaH  : (r, L)  - (0,L) if r=0
    betaCtx: (d_ctx, L)
    betaLag: dict { (ℓ', d): (L,) }
    r_disp : (L,)  NB dispersion (> 0)
    gamma0 : (L,)
    gammaS : (M, L)
    gammaH : (r, L)
    gammaCtx: (d_ctx, L)
    """

    L: int
    beta0: np.ndarray  # (L,)
    betaS: np.ndarray  # (M, L)
    betaH: np.ndarray  # (r, L)
    betaCtx: np.ndarray  # (d_ctx, L)
    betaLag: Dict[LagKey, np.ndarray]  # {(ℓ',d): (L,)}
    r_disp: np.ndarray  # (L,)
    gamma0: np.ndarray  # (L,)
    gammaS: np.ndarray  # (M, L)
    gammaH: np.ndarray  # (r, L)
    gammaCtx: np.ndarray  # (d_ctx, L)


# ---------------------------------------------------------------------------
@dataclass
class CoarseDurParams:
    """
    NB duration emission at the coarsest scale  (conditional on X_t > 0).

    log ν_t = α_0 + α_S · onehot(S_t) + α_H · H_t + α_ctx · ctx_t
              + α_X · log(1 + X_t)
              + Σ α_lag[(ℓ',d)] · log(1 + Y_{t-d}^{ℓ'})

    Attributes
    ----------
    alphaX  : (L,)  count-coupling coefficient
    alphaLag: dict { (ℓ', d): (L,) }
    s_disp  : (L,)  NB dispersion
    """

    L: int
    alpha0: np.ndarray
    alphaS: np.ndarray
    alphaH: np.ndarray
    alphaCtx: np.ndarray
    alphaX: np.ndarray
    alphaLag: Dict[LagKey, np.ndarray]
    s_disp: np.ndarray


# ---------------------------------------------------------------------------
@dataclass
class FineCountParams:
    """
    Dir-Mult count allocation scores at a finer scale (k < K).

    η_t = β_0 + β_S · onehot(S_t) + β_H · H_t + β_ctx · ctx_t
          + β_P · log(1 + X_parent)  +  β_D · log(1 + Y_parent)
          + β_dep · log(1 + earlier siblings of the block)
          + Σ β_lag[(ℓ',d)] · log(1 + X_{t-d}^{ℓ'})

    Attributes
    ----------
    betaP   : (L,)  parent count effect
    betaD   : (L,)  parent duration effect
    betaDep : (L,)  depletion effect, the counts earlier children of the same
              block already took.  The children of a block compete for a fixed
              parent total, so a sibling that has taken counts leaves fewer for
              this epoch.  Without this term the only instrument the structure
              search has for that competition is a lagged self edge, which is
              how within-block competition was being reported as structure.
    kappa   : float  Dirichlet concentration  (κ > 0)
    """

    L: int
    beta0: np.ndarray
    betaS: np.ndarray
    betaH: np.ndarray
    betaCtx: np.ndarray
    betaP: np.ndarray
    betaD: np.ndarray
    betaLag: Dict[LagKey, np.ndarray]
    kappa: float = 10.0
    betaDep: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.betaDep is None:
            self.betaDep = np.zeros(self.L)


# ---------------------------------------------------------------------------
@dataclass
class FineDurParams:
    """
    Dir-Mult duration allocation scores at a finer scale (k < K).

    ξ_t = α_0 + α_S · onehot(S_t) + α_H · H_t + α_ctx · ctx_t
          + α_X · log(1 + X_t)  +  α_P · log(1 + Y_parent)
          + α_dep · log(1 + earlier siblings of the block)
          + Σ α_lag[(ℓ',d)] · log(1 + Y_{t-d}^{ℓ'})

    Attributes
    ----------
    alphaX   : (L,)  within-epoch count coupling
    alphaP   : (L,)  parent duration effect
    alphaDep : (L,)  depletion effect, the durations earlier children of the
               same block already took; see `FineCountParams.betaDep`
    tau      : float  Dirichlet concentration (τ > 0)
    """

    L: int
    alpha0: np.ndarray
    alphaS: np.ndarray
    alphaH: np.ndarray
    alphaCtx: np.ndarray
    alphaX: np.ndarray
    alphaP: np.ndarray
    alphaLag: Dict[LagKey, np.ndarray]
    tau: float = 10.0
    alphaDep: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.alphaDep is None:
            self.alphaDep = np.zeros(self.L)


# ---------------------------------------------------------------------------
# Top-level per-scale container
# ---------------------------------------------------------------------------


@dataclass
class ScaleParams:
    """
    All generative parameters for one scale of the N²DBN.

    For the coarsest scale (k=K): set `coarse_count` and `coarse_dur`.
    For finer scales (k < K):   set `fine_count` and `fine_dur`.
    """

    k: int  # 1-indexed scale
    regime: RegimeParams
    factor: FactorParams
    # Lagged parent sets: {dst_ell: {(src_ell, d), ...}}
    parent_sets: Dict[int, Set[LagKey]] = field(default_factory=dict)

    # Exactly one of these pairs is non-None:
    coarse_count: Optional[CoarseCountParams] = None
    coarse_dur: Optional[CoarseDurParams] = None
    fine_count: Optional[FineCountParams] = None
    fine_dur: Optional[FineDurParams] = None

    # ------------------------------------------------------------------
    @property
    def is_coarsest(self) -> bool:
        return self.coarse_count is not None

    @property
    def L(self) -> int:
        if self.coarse_count is not None:
            return self.coarse_count.L
        return self.fine_count.L  # type: ignore[union-attr]

    @property
    def M(self) -> int:
        return self.regime.M

    @property
    def r(self) -> int:
        return self.factor.r

    def __post_init__(self) -> None:
        if self.is_coarsest:
            assert (
                self.fine_count is None and self.fine_dur is None
            ), "Coarsest scale must not have fine_count/fine_dur"
            assert self.coarse_dur is not None, "Coarsest scale must have coarse_dur"
        else:
            assert (
                self.coarse_count is None and self.coarse_dur is None
            ), "Finer scale must not have coarse_count/coarse_dur"
            assert (
                self.fine_count is not None and self.fine_dur is not None
            ), "Finer scale must have both fine_count and fine_dur"
        # Initialise missing parent sets
        for ell in range(self.L):
            self.parent_sets.setdefault(ell, set())
