"""
model/emissions.py
==================
ZINB count emissions and NB duration emissions for the coarsest scale.

ZINB parameterisation
---------------------
  X ~ ZINB(π, r, μ)
    = π · δ_0  +  (1-π) · NB(r, μ)

  where NB(r, μ): P(X=x) ∝ Γ(x+r)/x! · (μ/(r+μ))^x · (r/(r+μ))^r
  and   E[X | NB] = μ,  Var[X|NB] = μ + μ²/r.

NB duration parameterisation
-----------------------------
  Y | X>0  ~  NB(s, ν)    where ν = E[Y | X>0],  s = dispersion.

Both classes are stateless (all parameters are passed as arguments or come
from ScaleParams), enabling straightforward M-step updates.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.special import gammaln

# ---------------------------------------------------------------------------
# Utility: numerically stable softmax (not the neural-net kind, used for
# link functions below)
# ---------------------------------------------------------------------------


def _softplus(x: np.ndarray) -> np.ndarray:
    """log(1 + exp(x)) - numerically stable."""
    return np.where(x > 30, x, np.log1p(np.exp(np.clip(x, -500, 30))))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    # np.where evaluates both branches, so the exponent is clipped first: a very
    # negative x would otherwise overflow exp(-x) before the branch is chosen.
    z = np.clip(x, -500.0, 500.0)
    pos = 1.0 / (1.0 + np.exp(-np.abs(z)))
    return np.where(z >= 0, pos, 1.0 - pos)


# ---------------------------------------------------------------------------
# NB log-PMF
# ---------------------------------------------------------------------------


def nb_log_pmf(x: np.ndarray, r: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """
    Log-PMF of Negative Binomial NB(r, μ).

    Parameters
    ----------
    x   : non-negative integer counts  (...,)
    r   : dispersion  > 0             (...,)
    mu  : mean        > 0             (...,)

    Returns
    -------
    log_p : same shape as x.
    """
    eps = 1e-8
    r = np.maximum(r, eps)
    mu = np.maximum(mu, eps)
    p = r / (r + mu)  # success probability
    log_p = (
        gammaln(x + r) - gammaln(r) - gammaln(x + 1) + r * np.log(p) + x * np.log1p(-p)
    )
    return log_p


def nb_sample(
    r: np.ndarray,
    mu: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample from NB(r, μ).  Returns integer array of same shape as r/mu.
    """
    eps = 1e-8
    r = np.maximum(r, eps)
    mu = np.maximum(mu, eps)
    p = r / (r + mu)
    return rng.negative_binomial(r, p).astype(np.int64)


# ---------------------------------------------------------------------------
# ZINB  (zero-inflated negative binomial)
# ---------------------------------------------------------------------------


def zinb_log_pmf(
    x: np.ndarray,
    pi: np.ndarray,
    r: np.ndarray,
    mu: np.ndarray,
) -> np.ndarray:
    """
    Log-PMF of ZINB(π, r, μ).

    P(X=0) = π + (1-π)·NB(0; r, μ)
    P(X=x) = (1-π)·NB(x; r, μ)    for x > 0.
    """
    eps = 1e-8
    pi = np.clip(pi, eps, 1 - eps)
    nb_0 = nb_log_pmf(np.zeros_like(x), r, mu)  # log NB(0; r, μ)

    log_p_zero = np.logaddexp(
        np.log(pi),
        np.log1p(-pi) + nb_0,
    )
    log_p_pos = np.log1p(-pi) + nb_log_pmf(x, r, mu)

    return np.where(x == 0, log_p_zero, log_p_pos)


def zinb_sample(
    pi: np.ndarray,
    r: np.ndarray,
    mu: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample from ZINB(π, r, μ).

    Algorithm:
      1. Draw structural-zero mask  z ~ Bernoulli(π).
      2. Draw NB count n ~ NB(r, μ).
      3. Return 0 where z=1, n otherwise.
    """
    eps = 1e-8
    pi = np.clip(pi, eps, 1 - eps)
    z = rng.random(size=pi.shape) < pi  # structural zeros
    n = nb_sample(r, mu, rng)
    return np.where(z, np.int64(0), n)


# ---------------------------------------------------------------------------
# Linear predictors  (shared by coarsest-scale emission & finer-scale scores)
# ---------------------------------------------------------------------------


def compute_log_mu(
    s_idx: int,
    h_t: np.ndarray,
    ctx_t: np.ndarray,
    beta0: np.ndarray,  # (L,)
    betaS: np.ndarray,  # (M, L)
    betaH: np.ndarray,  # (r, L)
    betaCtx: np.ndarray,  # (d, L)
    lagged_X: np.ndarray,  # (L,)  - already aggregated lag terms
) -> np.ndarray:
    """
    Compute log(μ_t) for all labels at once.  Returns (L,) array.

    Parameters
    ----------
    s_idx     : current regime index (0-based)
    h_t       : (r,) latent factor at current epoch  (empty if r=0)
    ctx_t     : (d,) context features
    lagged_X  : (L,) pre-computed Σ_{(ℓ',d)∈P} β_{ℓ'd} log(1+X_{t-d}^{ℓ'})
    """
    log_mu = beta0 + betaS[s_idx]
    if betaH.shape[0] > 0:  # r > 0
        log_mu = log_mu + betaH.T @ h_t
    if betaCtx.shape[0] > 0:  # d_ctx > 0
        log_mu = log_mu + betaCtx.T @ ctx_t
    log_mu = log_mu + lagged_X
    return log_mu  # (L,)


def compute_logit_pi(
    s_idx: int,
    h_t: np.ndarray,
    ctx_t: np.ndarray,
    gamma0: np.ndarray,
    gammaS: np.ndarray,
    gammaH: np.ndarray,
    gammaCtx: np.ndarray,
) -> np.ndarray:
    """Compute logit(π_t) for all labels.  Returns (L,) array."""
    logit = gamma0 + gammaS[s_idx]
    if gammaH.shape[0] > 0:
        logit = logit + gammaH.T @ h_t
    if gammaCtx.shape[0] > 0:
        logit = logit + gammaCtx.T @ ctx_t
    return logit


def aggregate_lagged(
    t: int,
    X_arr: np.ndarray,  # (T, L)  - full count array (0 for unset epochs)
    parent_sets: dict,  # {dst_ell: {(src_ell, d), ...}}
    coeff: dict,  # {(src_ell, d): (L,)}
    L: int,
) -> np.ndarray:
    """
    Aggregate lagged count contributions for all labels at epoch t.

    Returns
    -------
    (L,) array: Σ_{(ℓ',d)∈P_ℓ} coeff[(ℓ',d)][ℓ] · log(1 + X_{t-d}^{ℓ'})
                = 0 where t-d < 0.
    """
    result = np.zeros(L, dtype=np.float64)
    for dst_ell in range(L):
        for src_ell, d in parent_sets.get(dst_ell, set()):
            t_lag = t - d
            if t_lag >= 0:
                x_lag = float(X_arr[t_lag, src_ell])
                coef = coeff.get((src_ell, d), np.zeros(L))
                result[dst_ell] += coef[dst_ell] * np.log1p(x_lag)
    return result
