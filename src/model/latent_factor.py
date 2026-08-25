"""
model/latent_factor.py
======================
Regime-dependent linear Gaussian latent factor H_t^(k).

Generative model
----------------
H_0 ~ N(μ_0, Σ_0)
H_t = F[s_t] H_{t-1}  +  G[s_t] ctx_t  +  ε_t,   ε_t ~ N(0, Σ[s_t])

When r=0, all methods return zero-dimensional arrays and are no-ops.

Sampling and inference
----------------------
`sample_trajectory()` draws from this model.  The smoother and the
variational approximation used for inference live in `inference/posterior.py`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .params import FactorParams

# ---------------------------------------------------------------------------


class LatentFactor:
    """
    Generative wrapper for the regime-switching LGSSM latent factor.
    """

    def __init__(self, params: FactorParams) -> None:
        self.p = params

    # ------------------------------------------------------------------
    def sample_trajectory(
        self,
        T: int,
        regimes: np.ndarray,  # (T,) int  - already sampled
        ctx: np.ndarray,  # (T, d_ctx)
        rng: np.random.Generator,
        h_prev: Optional[np.ndarray] = None,  # (r,) warm start
    ) -> np.ndarray:
        """
        Forward-sample H_{0:T-1} given the regime sequence.

        Parameters
        ----------
        T        : number of epochs.
        regimes  : (T,) regime indices  (0-based).
        ctx      : (T, d_ctx) context features.
        rng      : numpy random generator.
        h_prev   : if given, used as H_{-1} for t=0 transition.

        Returns
        -------
        H : (T, r) float array.
            Returns (T, 0) if r=0 (no-op).
        """
        p = self.p
        r = p.r

        H = np.empty((T, r), dtype=np.float64)

        if r == 0:
            return H  # (T, 0) - factor contributions vanish everywhere

        for t in range(T):
            s = int(regimes[t])
            Fs = p.F[s]  # (r, r)
            Gs = p.G[s]  # (r, d_ctx)
            Sig = p.Sigma[s]  # (r, r)
            ctx_t = ctx[t]  # (d_ctx,)

            if t == 0:
                # Sample from initial distribution
                mean_t = p.mu0 + (Gs @ ctx_t if ctx_t.shape[0] > 0 else 0.0)
                H[t] = rng.multivariate_normal(mean_t, p.Sigma0)
            else:
                h_tm1 = H[t - 1] if h_prev is None else (H[t - 1] if t > 1 else h_prev)
                if t == 1 and h_prev is not None:
                    h_tm1 = h_prev
                else:
                    h_tm1 = H[t - 1]
                mean_t = Fs @ h_tm1 + (Gs @ ctx_t if ctx_t.shape[0] > 0 else 0.0)
                H[t] = rng.multivariate_normal(mean_t, Sig)

        return H

    # ------------------------------------------------------------------
    def log_prob(
        self,
        H: np.ndarray,  # (T, r)
        regimes: np.ndarray,  # (T,)
        ctx: np.ndarray,  # (T, d_ctx)
    ) -> float:
        """
        Log-probability of trajectory H given regimes and context.
        Returns 0.0 when r=0.
        """
        from scipy.stats import multivariate_normal as mvn

        p = self.p
        if p.r == 0:
            return 0.0

        T = len(H)
        log_p = 0.0

        for t in range(T):
            s = int(regimes[t])
            Fs = p.F[s]
            Gs = p.G[s]
            Sig = p.Sigma[s]
            ctx_t = ctx[t]

            if t == 0:
                mean_t = p.mu0 + (Gs @ ctx_t if ctx_t.shape[0] > 0 else 0.0)
                cov_t = p.Sigma0
            else:
                mean_t = Fs @ H[t - 1] + (Gs @ ctx_t if ctx_t.shape[0] > 0 else 0.0)
                cov_t = Sig

            log_p += mvn.logpdf(H[t], mean=mean_t, cov=cov_t)

        return float(log_p)
