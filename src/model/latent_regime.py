"""
model/latent_regime.py
======================
Sampling and log-probability computation for the latent regime chain S_t^(k).

Transition model
----------------
Coarsest scale (k=K):
    p(S_t = m | S_{t-1} = m', ctx_t) = softmax_m( a_m + A[m,m'] + b_m^T ctx_t )

Finer scales (k < K):
    p(S_t = m | S_{t-1} = m', S_{parent} = n, ctx_t)
        = softmax_m( a_m + A[m,m'] + C[m,n] + b_m^T ctx_t )
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .params import RegimeParams

# ---------------------------------------------------------------------------


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax.  Input: (M,) or (N, M)."""
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------


class RegimeChain:
    """
    Wrapper around RegimeParams that exposes `sample_sequence()` and
    `log_transition_prob()`.
    """

    def __init__(self, params: RegimeParams) -> None:
        self.p = params

    # ------------------------------------------------------------------
    def sample_sequence(
        self,
        T: int,
        ctx: np.ndarray,  # (T, d_ctx)
        parent_regimes: Optional[np.ndarray],  # (T,) int or None
        rng: np.random.Generator,
        prev_S: Optional[np.ndarray] = None,  # (T_prev,) for warm-start
    ) -> np.ndarray:
        """
        Sample a regime sequence S[0..T-1].

        Parameters
        ----------
        T              : number of epochs to sample.
        ctx            : (T, d_ctx) context features.
        parent_regimes : (T,) parent-scale regime for each epoch
                         (None if k=K, i.e., coarsest scale).
        rng            : numpy Generator.
        prev_S         : if given, S[-1] is used as the state at t=-1.

        Returns
        -------
        S : (T,) int array  (0-based regime indices).
        """
        p = self.p
        M = p.M
        S = np.empty(T, dtype=np.int32)

        for t in range(T):
            ctx_t = ctx[t] if ctx.shape[1] > 0 else ctx[t]
            parent_n = int(parent_regimes[t]) if parent_regimes is not None else None

            if t == 0 and prev_S is None:
                # Initial distribution
                logits = p.logit_init(ctx_t)
            else:
                prev_m = int(S[t - 1]) if t > 0 else int(prev_S[-1])
                logits = p.logit_transition(prev_m, parent_n, ctx_t)

            probs = _softmax(logits)
            S[t] = rng.choice(M, p=probs)

        return S

    # ------------------------------------------------------------------
    def log_transition_prob(
        self,
        S: np.ndarray,  # (T,)
        ctx: np.ndarray,  # (T, d_ctx)
        parent_regimes: Optional[np.ndarray],  # (T,) or None
    ) -> float:
        """
        Compute log p(S | ctx, parent_regimes).

        Returns scalar log-probability of the full sequence.
        """
        p = self.p
        T = len(S)
        log_prob = 0.0

        for t in range(T):
            ctx_t = ctx[t]
            parent_n = int(parent_regimes[t]) if parent_regimes is not None else None

            if t == 0:
                logits = p.logit_init(ctx_t)
            else:
                logits = p.logit_transition(int(S[t - 1]), parent_n, ctx_t)

            log_probs = logits - np.logaddexp.reduce(logits)  # log softmax
            log_prob += log_probs[S[t]]

        return float(log_prob)

    # ------------------------------------------------------------------
    def transition_matrix(
        self,
        ctx_t: np.ndarray,
        parent_n: Optional[int] = None,
    ) -> np.ndarray:
        """
        Full (M, M) transition matrix for epoch t given context / parent.

        Returns
        -------
        T_mat : (M, M)  T_mat[m, m'] = p(S_t=m | S_{t-1}=m', parent_n, ctx_t)
        """
        p = self.p
        M = p.M
        T_mat = np.empty((M, M), dtype=np.float64)
        for prev_m in range(M):
            logits = p.logit_transition(prev_m, parent_n, ctx_t)
            T_mat[:, prev_m] = _softmax(logits)
        return T_mat
