"""
inference/posterior.py
======================
Posterior inference for latent variables S^(k) and H^(k).

Implements:
    RegimePosterior    - exact HMM forward-backward in log-space for q(S^(k))
    FactorPosterior    - regime-averaged Kalman filter-smoother for q(H^(k))
    PosteriorSummaries - collects γ_t, ξ_t, h̄_t, Σ_t for M/S-step use

Mathematical references (N²DBN manuscript equations)
-----------------------------------------------------
* E-step:  Eq. 680-685
* Regime posterior (forward-backward):  standard HMM adapted to time-varying
  transitions with parent-scale coupling (Eq. 187-198).
* Factor posterior (Gaussian smoother):  regime-averaged assumed-density
  Kalman filter + RTS smoother (Eq. 3.2 of implementation plan).
* Posterior summaries for top-down transfer:  Eq. 756-777.

Design notes
------------
* RegimePosterior.forward_backward() requires pre-computed per-epoch
  log-likelihoods log p(x_t | S_t=m).  These are built in estep.py so that
  H's current mean estimate h̄ can be baked in.
* FactorPosterior.smoother() accepts optional Laplace pseudo-observations
  (z_t, R_t) derived from the gradient / Hessian of the expected emission
  log-likelihood w.r.t. H_t.  Without them it degrades to a prior-only
  Kalman smoother (useful for the first iteration).
* FactorPosterior.compute_pseudo_obs_coarse() computes the Laplace correction
  for the coarsest-scale emission (ZINB + NB) via numerical finite differences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..model.emissions import nb_log_pmf, zinb_log_pmf
from ..model.params import FactorParams, RegimeParams, ScaleParams

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def logsumexp(a: np.ndarray, axis=None, keepdims: bool = False) -> np.ndarray:
    """
    Stable log-sum-exp.

    Local re-implementation of `scipy.special.logsumexp`: the forward-backward
    recursion calls it O(T) times per E-step and SciPy's wrapper overhead
    (array-API promotion, dtype checks) dominated the profile.
    """
    a = np.asarray(a, dtype=float)
    amax = np.max(a, axis=axis, keepdims=True)
    amax = np.where(np.isfinite(amax), amax, 0.0)
    with np.errstate(divide="ignore"):
        out = np.log(np.sum(np.exp(a - amax), axis=axis, keepdims=True)) + amax
    if keepdims:
        return out
    if axis is None:
        return float(out.reshape(-1)[0])
    return np.squeeze(out, axis=axis)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    z = np.clip(x, -500.0, 500.0)
    pos = 1.0 / (1.0 + np.exp(-np.abs(z)))
    return np.where(z >= 0, pos, 1.0 - pos)


def _make_sym(A: np.ndarray) -> np.ndarray:
    """Force symmetry to counter floating-point drift: (A + A^T) / 2."""
    return 0.5 * (A + A.T)


def _safe_solve(A: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Solve A x = b with a ridge for stability."""
    r = A.shape[0]
    return np.linalg.solve(A + eps * np.eye(r), b)


def _safe_inv(A: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Invert A with a ridge for numerical stability."""
    r = A.shape[0]
    return np.linalg.inv(A + eps * np.eye(r))


def regime_offsets(
    params: RegimeParams,
    ctx: np.ndarray,  # (T, d_ctx)
    parent_gamma: Optional[np.ndarray] = None,  # (T, M_parent)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-epoch additive terms of the transition logits.

    Returns
    -------
    ctx_term : (T, M)  context contribution  b_m^T ctx_t
    offset   : (T, M)  ctx_term + cross-scale contribution Σ_n q_par(t,n) C[m,n]

    Shared by `RegimePosterior.build_log_trans` (E-step) and
    `mstep.update_regime_transitions` so both use identical logits.
    """
    T = ctx.shape[0]
    M = params.M
    if ctx.shape[1] > 0:
        ctx_term = ctx @ params.ctx.T  # (T, M)
    else:
        ctx_term = np.zeros((T, M))
    offset = ctx_term
    if parent_gamma is not None and params.cross is not None:
        if parent_gamma.shape[1] != params.cross.shape[1]:
            raise ValueError(
                f"parent posterior has {parent_gamma.shape[1]} regimes but the "
                f"cross-scale block expects {params.cross.shape[1]}"
            )
        offset = offset + parent_gamma @ params.cross.T  # (T, M)
    return ctx_term, offset


def _kalman_update(
    m_pred: np.ndarray,  # (r,)
    P_pred: np.ndarray,  # (r, r)
    z: np.ndarray,  # (r,)  pseudo-observation
    R: np.ndarray,  # (r, r) pseudo-noise covariance
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Kalman measurement update for the identity observation model  y = H + v.

    Parameters
    ----------
    m_pred : predicted mean
    P_pred : predicted covariance
    z      : pseudo-observation  z = H̄ - Hess^{-1} grad  (Newton correction)
    R      : pseudo-noise covariance  R = -Hess^{-1}

    Returns
    -------
    m_upd : (r,)  updated mean
    P_upd : (r, r) updated covariance
    """
    r = m_pred.shape[0]
    I_r = np.eye(r)
    S = P_pred + R  # innovation covariance
    K = P_pred @ _safe_inv(S)  # Kalman gain   (r, r)
    m_upd = m_pred + K @ (z - m_pred)
    P_upd = _make_sym((I_r - K) @ P_pred)
    return m_upd, P_upd


# ---------------------------------------------------------------------------
# RegimePosterior - exact HMM forward-backward
# ---------------------------------------------------------------------------


class RegimePosterior:
    """
    Exact log-space forward-backward algorithm for the latent regime chain
    S_0^(k), ..., S_{T-1}^(k).

    The transition probabilities may be time-varying (different context /
    parent-regime at each epoch).
    """

    @staticmethod
    def forward_backward(
        log_like: np.ndarray,  # (T, M)   log p(x_t | S_t=m)
        log_init: np.ndarray,  # (M,)     log p(S_0=m)  (unnormalised OK)
        log_trans: np.ndarray,  # (T-1, M, M) log_trans[t, m_next, m_prev]
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Run the forward-backward algorithm.

        Conventions
        -----------
        log_trans[t, m_next, m_prev] = log p(S_{t+1}=m_next | S_t=m_prev)
        xi[t, m, m']               = q(S_t=m, S_{t+1}=m')

        Parameters
        ----------
        log_like  : (T, M)  - log p(obs_t | S_t=m)
        log_init  : (M,)    - unnormalised log p(S_0=m)
        log_trans : (T-1, M, M) - log transition probabilities

        Returns
        -------
        gamma : (T, M)      - q(S_t=m)
        xi    : (T-1, M, M) - q(S_t=m, S_{t+1}=m')
        log_Z : float       - log p(observations)
        """
        T, M = log_like.shape
        if T > 1:
            assert log_trans.shape == (
                T - 1,
                M,
                M,
            ), f"log_trans shape {log_trans.shape} != ({T-1}, {M}, {M})"

        # ---- Degenerate single-regime case -------------------------------
        # With M=1 the chain carries no information: γ ≡ 1, ξ ≡ 1 and the
        # log-evidence is just the summed emission term.  Short-circuiting
        # here keeps the M=1 ablation (TopDownNoRegime) cheap.
        if M == 1:
            return (
                np.ones((T, 1)),
                np.ones((max(T - 1, 1), 1, 1)),
                float(log_like.sum()),
            )

        # ---- Forward pass ------------------------------------------------
        # log_alpha[t, m] = log p(S_t=m, x_{0:t})
        log_alpha = np.empty((T, M))
        log_alpha[0] = log_init - logsumexp(log_init) + log_like[0]

        for t in range(1, T):
            # log_alpha[t, m] = log_like[t, m]
            #   + logsumexp_{m'} ( log_trans[t-1, m, m'] + log_alpha[t-1, m'] )
            inner = (
                log_trans[t - 1]  # (M_next, M_prev)
                + log_alpha[t - 1][np.newaxis, :]  # broadcast over M_next
            )  # to (M, M)
            log_alpha[t] = log_like[t] + logsumexp(inner, axis=1)

        log_Z = float(logsumexp(log_alpha[T - 1]))

        # ---- Backward pass -----------------------------------------------
        # log_beta[t, m] = log p(x_{t+1:T} | S_t=m)
        log_beta = np.zeros((T, M))  # log_beta[T-1] = 0  (empty product)
        for t in range(T - 2, -1, -1):
            # log_beta[t, m] = logsumexp_{m'} (
            #     log_trans[t, m', m]
            #   + log_like[t+1, m']
            #   + log_beta[t+1, m'] )
            inner = (
                log_trans[t]  # (M_next, M_curr)
                + (log_like[t + 1] + log_beta[t + 1])[:, np.newaxis]
            )  # (M_next, 1)
            log_beta[t] = logsumexp(inner, axis=0)  # sum over M_next

        # ---- Marginals (gamma) -------------------------------------------
        log_gamma = log_alpha + log_beta  # (T, M)
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(np.clip(log_gamma, -700.0, 0.0))

        # ---- Joint marginals (xi) - fully vectorised over t ---------------
        # xi[t, m_curr, m_next] = q(S_t=m_curr, S_{t+1}=m_next)
        # = alpha_t(m_curr) A(m_next|m_curr) like_{t+1}(m_next) beta_{t+1}(m_next) / Z
        # log_trans[t] has shape (M_next, M_curr); transpose for (M_curr, M_next).
        if T > 1:
            log_xi = (
                log_alpha[:-1][:, :, np.newaxis]  # (T-1, M_curr, 1)
                + np.transpose(log_trans, (0, 2, 1))  # (T-1, M_curr, M_next)
                + (log_like[1:] + log_beta[1:])[:, np.newaxis, :]  # (T-1, 1, M_next)
            )
            log_xi -= logsumexp(log_xi.reshape(T - 1, -1), axis=1)[
                :, np.newaxis, np.newaxis
            ]
            xi = np.exp(np.clip(log_xi, -700.0, 0.0))
        else:
            xi = np.ones((1, M, M)) / (M * M)

        return gamma, xi, log_Z

    @staticmethod
    def build_log_trans(
        params: RegimeParams,
        ctx: np.ndarray,  # (T, d_ctx)
        parent_gamma: Optional[np.ndarray] = None,  # (T, M_parent) or None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construct log_init and log_trans from RegimeParams.

        For k < K the cross-scale term C[m, n] is averaged over the parent
        posterior, `Σ_n q_parent(t, n) C[m, n]`, i.e. the coupling enters
        through the *expected logit*.  This is the same first-order form the
        M-step differentiates (see `mstep.update_regime_transitions`), so
        E- and M-steps optimise one consistent objective; it is exact when the
        parent posterior is degenerate.

        Parameters
        ----------
        params       : RegimeParams for this scale
        ctx          : (T, d_ctx) context features
        parent_gamma : (T, M_parent) optional soft parent-regime assignments

        Returns
        -------
        log_init  : (M,)
        log_trans : (T-1, M, M)  log_trans[t, m_next, m_prev]
        """
        T, _ = ctx.shape
        M = params.M

        # Per-epoch additive terms that do not depend on the previous regime.
        ctx_term, offset = regime_offsets(params, ctx, parent_gamma)  # (T, M)

        # Initial log-distribution
        log_init_raw = params.init + ctx_term[0]
        log_init = log_init_raw - logsumexp(log_init_raw)

        if T == 1:
            return log_init, np.empty((0, M, M))

        # logits[t-1, m_next, m_prev] = bias[m] + trans[m, m'] + offset[t, m]
        logits = (params.bias[:, np.newaxis] + params.trans)[np.newaxis, :, :] + offset[
            1:
        ][
            :, :, np.newaxis
        ]  # (T-1, M, M)
        log_trans = logits - logsumexp(logits, axis=1, keepdims=True)

        return log_init, log_trans


# ---------------------------------------------------------------------------
# FactorPosterior - regime-averaged Kalman smoother
# ---------------------------------------------------------------------------


class FactorPosterior:
    """
    Gaussian smoother for q(H_t^(k)) conditioned on soft regime assignments.

    Algorithm (assumed-density / regime-averaged Kalman smoother)
    -------------------------------------------------------------
    1. Compute regime-averaged LGSSM parameters at each t:
           F̄_t = Σ_m γ_t(m) F_m
           ḡ_t = Σ_m γ_t(m) G_m ctx_t
           Q̄_t = Σ_m γ_t(m) Σ_m  +  Jensen-gap correction
    2. Optional emission update via Laplace pseudo-observations:
           Virtual observation  y_t = H_t + v_t,  v_t ~ N(0, R_t)
           where z_t and R_t come from compute_pseudo_obs_coarse().
    3. Kalman filter forward: m_f, P_f per epoch.
    4. RTS smoother backward: m_s, P_s per epoch.
    """

    @staticmethod
    def smoother(
        gamma: np.ndarray,  # (T, M)
        params: FactorParams,
        ctx: np.ndarray,  # (T, d_ctx)
        pseudo_obs: Optional[np.ndarray] = None,  # (T, r)
        pseudo_R: Optional[np.ndarray] = None,  # (T, r, r)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Regime-averaged Kalman filter + RTS smoother.

        Parameters
        ----------
        gamma      : (T, M) soft regime weights from forward-backward
        params     : FactorParams (F, G, Sigma per regime; mu0, Sigma0)
        ctx        : (T, d_ctx) context features
        pseudo_obs : (T, r) optional Laplace pseudo-observations z_t
        pseudo_R   : (T, r, r) optional pseudo-noise covariances R_t

        Returns
        -------
        m_H     : (T, r)    posterior mean  E_q[H_t]
        Sigma_H : (T, r, r) posterior covariance  Cov_q[H_t]
        """
        T, M = gamma.shape
        r = params.r

        if r == 0:
            return np.empty((T, 0)), np.empty((T, 0, 0))

        I_r = np.eye(r)
        eps = 1e-8

        # ---- Storage for filter pass ------------------------------------
        m_f = np.zeros((T, r))  # filtered mean
        P_f = np.zeros((T, r, r))  # filtered covariance
        m_pred = np.zeros((T, r))  # predictive mean
        P_pred = np.zeros((T, r, r))  # predictive covariance

        # t=0: initialise from prior
        m_pred[0] = params.mu0.copy()
        P_pred[0] = _make_sym(params.Sigma0) + eps * I_r

        if pseudo_obs is not None:
            m_f[0], P_f[0] = _kalman_update(
                m_pred[0], P_pred[0], pseudo_obs[0], pseudo_R[0]
            )
        else:
            m_f[0] = m_pred[0].copy()
            P_f[0] = P_pred[0].copy()

        for t in range(1, T):
            gam_t = gamma[t]  # (M,)

            # Regime-averaged transition matrix
            F_bar = np.einsum("m,mij->ij", gam_t, params.F)  # (r, r)

            # Regime-averaged context drift
            if params.d_ctx > 0:
                g_bar = np.einsum("m,mij,j->i", gam_t, params.G, ctx[t])
            else:
                g_bar = np.zeros(r)

            # Regime-averaged noise covariance + Jensen-gap
            Q_bar = np.einsum("m,mij->ij", gam_t, params.Sigma)  # (r, r)
            for m_idx in range(M):
                dF = params.F[m_idx] - F_bar  # (r, r)
                Q_bar = Q_bar + gam_t[m_idx] * (dF @ P_f[t - 1] @ dF.T)
            Q_bar = _make_sym(Q_bar) + eps * I_r

            # Prediction step
            m_pred[t] = F_bar @ m_f[t - 1] + g_bar
            P_pred[t] = _make_sym(F_bar @ P_f[t - 1] @ F_bar.T + Q_bar)

            # Update step (with or without emission correction)
            if pseudo_obs is not None:
                m_f[t], P_f[t] = _kalman_update(
                    m_pred[t], P_pred[t], pseudo_obs[t], pseudo_R[t]
                )
            else:
                m_f[t] = m_pred[t].copy()
                P_f[t] = _make_sym(P_pred[t])

        # ---- RTS smoother -----------------------------------------------
        m_s = m_f.copy()
        P_s = P_f.copy()

        for t in range(T - 2, -1, -1):
            # Smoother gain: G_t = P_f[t] F̄_{t+1}^T P_pred[t+1]^{-1}
            gam_t1 = gamma[t + 1]
            F_bar_t1 = np.einsum("m,mij->ij", gam_t1, params.F)
            G_gain = P_f[t] @ F_bar_t1.T @ _safe_inv(P_pred[t + 1])

            m_s[t] = m_f[t] + G_gain @ (m_s[t + 1] - m_pred[t + 1])
            P_s[t] = _make_sym(
                P_f[t] + G_gain @ (P_s[t + 1] - P_pred[t + 1]) @ G_gain.T
            )
            # Ensure PSD
            P_s[t] = P_s[t] + eps * I_r

        return m_s, P_s

    @staticmethod
    def compute_pseudo_obs_coarse(
        X_K: np.ndarray,  # (T, L) observed counts at coarsest scale
        Y_K: np.ndarray,  # (T, L) observed durations
        h_bar: np.ndarray,  # (T, r) current H posterior mean
        gamma: np.ndarray,  # (T, M) regime posteriors
        scale_params: ScaleParams,  # must be the coarsest scale (is_coarsest=True)
        ctx: np.ndarray,  # (T, d_ctx)
        lag_X: np.ndarray,  # (T, L) pre-aggregated lagged count features
        lag_Y: np.ndarray,  # (T, L) pre-aggregated lagged dur features
        delta: float = 1e-4,  # finite-difference step size
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Laplace pseudo-observations for the coarsest-scale H smoother.

        At each time t, evaluates
            ℓ_t(h) = Σ_m γ_t(m) Σ_ℓ [log p(X_{t,ℓ}|h, m) + log p(Y_{t,ℓ}|h, m, X_{t,ℓ}>0)]
        numerically and returns the Newton pseudo-observations / pseudo-noises
        that represent the Laplace approximation around the current mean h̄_t:
            z_t = h̄_t + R_t grad_t       (Newton-corrected centre)
            R_t = (−Hess_t)^{-1}         (curvature-based precision)

        Parameters
        ----------
        X_K, Y_K       : (T, L) observed arrays
        h_bar          : (T, r) current posterior mean (e.g. zeros on first pass)
        gamma          : (T, M) from forward-backward
        scale_params   : coarsest-scale ScaleParams (must have coarse_count, coarse_dur)
        ctx            : (T, d_ctx)
        lag_X, lag_Y   : (T, L) pre-computed lag contributions
        delta          : step size for finite-difference gradient/Hessian

        Returns
        -------
        z : (T, r)    Laplace pseudo-observations
        R : (T, r, r) Laplace pseudo-noise covariances
        """
        assert (
            scale_params.is_coarsest
        ), "compute_pseudo_obs_coarse: scale_params must be the coarsest scale"
        T, r = h_bar.shape
        if r == 0:
            return np.empty((T, 0)), np.empty((T, 0, 0))

        from ..learning.estep import expected_emission_ll_coarse

        def _ll(h: np.ndarray) -> np.ndarray:
            """Expected emission log-likelihood at every epoch: (T,)."""
            return expected_emission_ll_coarse(
                X_K, Y_K, h, gamma, scale_params, ctx, lag_X, lag_Y
            )

        # Central differences evaluated for *all* epochs at once: 2r + 2r^2
        # vectorised passes replace T x (2r + 2r^2) per-epoch evaluations.
        eye = np.eye(r) * delta
        grad = np.empty((T, r))
        for i in range(r):
            grad[:, i] = (_ll(h_bar + eye[i]) - _ll(h_bar - eye[i])) / (2.0 * delta)

        hess = np.empty((T, r, r))
        for i in range(r):
            for j in range(i, r):
                hij = (
                    _ll(h_bar + eye[i] + eye[j])
                    - _ll(h_bar + eye[i] - eye[j])
                    - _ll(h_bar - eye[i] + eye[j])
                    + _ll(h_bar - eye[i] - eye[j])
                ) / (4.0 * delta**2)
                hess[:, i, j] = hij
                hess[:, j, i] = hij

        # The Hessian of a log-likelihood is negative semi-definite.
        # Clip eigenvalues to (-inf, -eps] to ensure neg-definiteness.
        eigvals, eigvecs = np.linalg.eigh(hess)  # (T, r), (T, r, r)
        eigvals = np.minimum(eigvals, -1e-6)
        hess_nsd = np.einsum("tij,tj,tkj->tik", eigvecs, eigvals, eigvecs)

        R = np.linalg.inv(-hess_nsd + 1e-8 * np.eye(r))  # (T, r, r) PSD
        z = h_bar + np.einsum("tij,tj->ti", R, grad)  # Newton step

        return z, R


# ---------------------------------------------------------------------------
# PosteriorSummaries - collected posterior moments
# ---------------------------------------------------------------------------


@dataclass
class PosteriorSummaries:
    """
    All posterior moments at one scale, used by the M-step and S-step.

    Attributes
    ----------
    gamma   : (T, M)      q(S_t=m)           - soft regime assignments
    xi      : (T-1, M, M) q(S_t=m, S_{t+1}=m') - joint regime marginals
    m_H     : (T, r)      E_q[H_t]
    Sigma_H : (T, r, r)   Cov_q[H_t]
    log_Z   : float       log p(observations | Θ, G, parent summaries)
    """

    gamma: np.ndarray  # (T, M)
    xi: np.ndarray  # (T-1, M, M)
    m_H: np.ndarray  # (T, r)
    Sigma_H: np.ndarray  # (T, r, r)
    log_Z: float

    # ------------------------------------------------------------------
    @property
    def s_bar(self) -> np.ndarray:
        """(T, M) soft regime assignments  s̄_t = E_q[onehot(S_t)]."""
        return self.gamma

    @property
    def h_bar(self) -> np.ndarray:
        """(T, r) posterior mean  h̄_t = E_q[H_t]."""
        return self.m_H

    # ------------------------------------------------------------------
    def expected_hh(self) -> np.ndarray:
        """
        E_q[H_t H_t^T]  for all t.

        Returns
        -------
        (T, r, r)  =  Sigma_H[t] + m_H[t] ⊗ m_H[t]
        """
        T, r = self.m_H.shape
        if r == 0:
            return np.empty((T, 0, 0))
        return self.Sigma_H + np.einsum("ti,tj->tij", self.m_H, self.m_H)

    def expected_h_prev_h(
        self,
        P_cross: np.ndarray,  # (T-1, r, r) from smoother cross-covariances
    ) -> np.ndarray:
        """
        E_q[H_{t+1} H_t^T]  for t = 0, ..., T-2.

        Parameters
        ----------
        P_cross : (T-1, r, r)  smoother cross-covariances Cov(H_{t+1}, H_t)

        Returns
        -------
        (T-1, r, r)  =  P_cross[t] + m_H[t+1] ⊗ m_H[t]
        """
        if self.m_H.shape[1] == 0:
            T = self.m_H.shape[0]
            return np.empty((T - 1, 0, 0))
        return P_cross + np.einsum("ti,tj->tij", self.m_H[1:], self.m_H[:-1])

    def elbo_entropy(self) -> float:
        """
        Approximate entropy of the variational posterior.

            H[q(S)] = -Σ_t Σ_m γ_t(m) log γ_t(m)
            H[q(H)] ≈ Σ_t (r/2)(1 + log 2π) + (1/2) log det Σ_H[t]

        Returns scalar float.
        """
        eps = 1e-12
        H_S = float(-np.sum(self.gamma * np.log(np.maximum(self.gamma, eps))))

        H_H = 0.0
        r = self.m_H.shape[1]
        if r > 0:
            for t in range(self.m_H.shape[0]):
                sign, logdet = np.linalg.slogdet(self.Sigma_H[t])
                if sign > 0:
                    H_H += 0.5 * (r * (1.0 + np.log(2.0 * np.pi)) + logdet)

        return H_S + H_H

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        T = self.gamma.shape[0]
        M = self.gamma.shape[1]
        r = self.m_H.shape[1]
        return f"PosteriorSummaries(T={T}, M={M}, r={r}, log_Z={self.log_Z:.3f})"
