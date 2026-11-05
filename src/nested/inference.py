"""Explicit approximate inference and a freshly evaluated surrogate monitor."""

from __future__ import annotations

import numpy as np

from ..inference.posterior import PosteriorSummaries, RegimePosterior
from .law import allocation_logpmf, blocks, emission_terms, scores


def empty_posterior(record, model):
    result = []
    for k, level in enumerate(model.levels):
        T, M, r = record.data.hierarchy.H[k], level.regime.M, level.factor.r
        result.append(
            PosteriorSummaries(
                np.full((T, M), 1 / M),
                np.full((max(T - 1, 0), M, M), 1 / M**2),
                np.zeros((T, r)),
                np.tile(np.eye(r), (T, 1, 1)),
                0.0,
            )
        )
    return result


def transition_inputs(record, model, k, posterior):
    parent = None
    if model.nested and k < model.K - 1:
        parent = posterior[k + 1].gamma[record.data.hierarchy.parent_maps[k]]
    return RegimePosterior.build_log_trans(
        model.levels[k].regime, record.data.C[k], parent
    )


def potentials(record, model, k, posterior):
    """Full affected block factors, not just one child's DM contribution.

    For fine blocks we subtract the reference block mass and add its m-th
    fraction to each child. This fixes the otherwise arbitrary state-independent
    constants in the auxiliary-chain monitor across parameter/graph proposals.
    """
    q = posterior[k]
    T, M = q.gamma.shape
    out = np.zeros((T, M))
    for ell, target in enumerate(model.levels[k].targets):
        if model.emission(k):
            for state in range(M):
                gamma = np.zeros_like(q.gamma)
                gamma[:, state] = 1
                out[:, state] += emission_terms(record, model, k, ell, gamma, q.m_H)
        else:
            batch = blocks(record, model, k, ell)
            if not len(batch.parents):
                continue
            idx = batch.indices
            base = scores(record, model, k, ell, q.gamma, q.m_H)[idx]
            reference = allocation_logpmf(
                record, model, k, ell, q.gamma, q.m_H, batch=batch, base=base
            )
            width = idx.shape[1]
            if M == 1:
                out[idx, 0] += reference[:, None] / width
                continue
            averaged = (q.gamma @ target.regime.T)[idx]
            for j in range(width):
                for state in range(M):
                    changed = base.copy()
                    changed[:, j, :] += target.regime[:, state] - averaged[:, j, :]
                    ll = allocation_logpmf(
                        record, model, k, ell, q.gamma, q.m_H, batch=batch, base=changed
                    )
                    out[idx[:, j], state] += ll - reference + reference / width
    if model.emission(k):
        out[~record.rows(k, model.lag_bound[k])] = 0
    if not np.isfinite(out).all():
        raise FloatingPointError(f"non-finite likelihood potentials at scale {k + 1}")
    return out


def pseudo_observations(record, model, k, posterior, delta=1e-3):
    """Laplace/assumed-density update using the corrected emission law."""
    q = posterior[k]
    T, r = q.m_H.shape
    mask = record.rows(k, model.lag_bound[k])

    def value(h):
        total = np.zeros(T)
        for ell in range(model.L):
            for m in range(q.gamma.shape[1]):
                gamma = np.zeros_like(q.gamma)
                gamma[:, m] = 1
                total += q.gamma[:, m] * emission_terms(record, model, k, ell, gamma, h)
        return np.where(mask, total, 0.0)

    grad, hess = np.zeros((T, r)), np.zeros((T, r, r))
    eye = np.eye(r) * delta
    for i in range(r):
        grad[:, i] = (value(q.m_H + eye[i]) - value(q.m_H - eye[i])) / (2 * delta)
        for j in range(i + 1):
            entry = (
                value(q.m_H + eye[i] + eye[j])
                - value(q.m_H + eye[i] - eye[j])
                - value(q.m_H - eye[i] + eye[j])
                + value(q.m_H - eye[i] - eye[j])
            ) / (4 * delta**2)
            hess[:, i, j] = hess[:, j, i] = entry
    eigen, vectors = np.linalg.eigh(-hess)
    precision = np.einsum("tij,tj,tkj->tik", vectors, np.maximum(eigen, 1e-6), vectors)
    covariance = np.linalg.inv(precision)
    # Infinite-noise limit for missing observations, not an observed zero.
    covariance[~mask] = np.eye(r) * 1e12
    return q.m_H + np.einsum("tij,tj->ti", covariance, grad), covariance


def gaussian_smoother(gamma, parameters, context, observations=None, noise=None):
    """Moment-matched switching dynamics and RTS lag covariances.

    This is an assumed-density Gaussian approximation, not exact switching
    inference. Lag covariances are retained for the expected Gaussian M-step.
    """
    T, M = gamma.shape
    r = parameters.r
    means, covs = np.zeros((T, r)), np.zeros((T, r, r))
    predicted, predicted_cov = np.zeros_like(means), np.zeros_like(covs)
    matrices = np.einsum("tm,mij->tij", gamma, parameters.F)
    for t in range(T):
        if t == 0:
            state_means = np.array(
                [parameters.mu0 + parameters.G[m] @ context[t] for m in range(M)]
            )
            predicted[t] = gamma[t] @ state_means
            predicted_cov[t] = parameters.Sigma0.copy()
            for m in range(M):
                delta = state_means[m] - predicted[t]
                predicted_cov[t] += gamma[t, m] * np.outer(delta, delta)
        else:
            state_means = np.array(
                [
                    parameters.F[m] @ means[t - 1] + parameters.G[m] @ context[t]
                    for m in range(M)
                ]
            )
            predicted[t] = gamma[t] @ state_means
            for m in range(M):
                delta = state_means[m] - predicted[t]
                predicted_cov[t] += gamma[t, m] * (
                    parameters.F[m] @ covs[t - 1] @ parameters.F[m].T
                    + parameters.Sigma[m]
                    + np.outer(delta, delta)
                )
        means[t], covs[t] = predicted[t], predicted_cov[t]
        if observations is not None:
            gain = np.linalg.solve(
                (predicted_cov[t] + noise[t]).T, predicted_cov[t].T
            ).T
            means[t] += gain @ (observations[t] - predicted[t])
            covs[t] -= gain @ predicted_cov[t]
        covs[t] = (covs[t] + covs[t].T) / 2 + np.eye(r) * 1e-9
    smoothed, smoothed_cov = means.copy(), covs.copy()
    cross = np.zeros((max(T - 1, 0), r, r))
    for t in range(T - 2, -1, -1):
        gain = np.linalg.solve(
            predicted_cov[t + 1] + np.eye(r) * 1e-9, matrices[t + 1] @ covs[t]
        ).T
        smoothed[t] += gain @ (smoothed[t + 1] - predicted[t + 1])
        smoothed_cov[t] += gain @ (smoothed_cov[t + 1] - predicted_cov[t + 1]) @ gain.T
        smoothed_cov[t] = (smoothed_cov[t] + smoothed_cov[t].T) / 2
        cross[t] = gain @ smoothed_cov[t + 1]
    return smoothed, smoothed_cov, cross


def infer(record, model, inner=3, tolerance=1e-5):
    """Deterministic initialization makes repeated monitor comparisons reproducible."""
    posterior = empty_posterior(record, model)
    for k in range(model.K - 1, -1, -1):
        level = model.levels[k]
        initial, transition = transition_inputs(record, model, k, posterior)
        for _ in range(inner):
            old = posterior[k].gamma.copy()
            ll = potentials(record, model, k, posterior)
            gamma, xi, log_z = RegimePosterior.forward_backward(ll, initial, transition)
            if len(gamma) == 1:
                xi = np.empty((0, level.regime.M, level.regime.M))
            posterior[k].gamma, posterior[k].xi = gamma, xi
            posterior[k].log_Z = float(log_z)
            if level.factor.r:
                if model.emission(k):
                    z, noise = pseudo_observations(record, model, k, posterior)
                    h, covariance, cross = gaussian_smoother(
                        gamma, level.factor, record.data.C[k], z, noise
                    )
                else:
                    h, covariance, cross = gaussian_smoother(
                        gamma, level.factor, record.data.C[k]
                    )
                delta_h = np.max(np.abs(h - posterior[k].m_H))
                posterior[k].m_H, posterior[k].Sigma_H = h, covariance
                posterior[k].lag_H = cross
            else:
                delta_h = 0.0
            if np.max(np.abs(old - gamma)) < tolerance and delta_h < tolerance:
                break
    return posterior


def monitor_record(record, model, posterior):
    """Rebuild every potential from the supplied final state. Not an ELBO."""
    total = 0.0
    for k in range(model.K - 1, -1, -1):
        initial, transition = transition_inputs(record, model, k, posterior)
        _, _, log_z = RegimePosterior.forward_backward(
            potentials(record, model, k, posterior),
            initial,
            transition,
        )
        total += float(log_z)
    return total


def monitor(records, model, posteriors):
    return (
        sum(monitor_record(rec, model, posteriors[rec.name]) for rec in records)
        - model.penalty()
    )
