"""
data/initialize.py
==================
Build one `ScaleParams` per scale from observed data alone.

Why this exists
---------------
The evaluation protocol initializes every fit from the generating family, and a
real event log has no generating family to read.  This module produces the same
object from moments of the observed counts and durations, so a fit on real data
starts from the data instead of from the answer.

What it does not do
-------------------
No fitting loop runs here.  Every block is a moment estimate, a k-means summary
or a zero, and the graph starts empty: the structure step adds the first edges
and the cross-scale warm start of the outer loop fills the coupling block on the
first sweep.  The M-step owns every refinement, including the Dirichlet
concentrations, which is why they start at the midpoint of the range its
golden-section search brackets rather than at a fitted value.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Set, Tuple, Union

import numpy as np

from ..core.blocks import BlockIndex
from ..core.variables import MultiScaleData
from ..inference.posterior import PosteriorSummaries
from ..learning.param_estimation import (
    estimate_transition_table,
    regime_params_from_table,
)
from ..model.params import (
    CoarseCountParams,
    CoarseDurParams,
    FactorParams,
    FineCountParams,
    FineDurParams,
    LagKey,
    RegimeParams,
    ScaleParams,
)

_EPS = 1e-8
# A label that never fires would otherwise send log of its mean to -inf.
_MEAN_FLOOR = 0.1
_DISP_LO, _DISP_HI = 0.1, 100.0
# A zero-inflation probability of exactly 0 or 1 has no finite logit.
_ZI_LO, _ZI_HI = 0.01, 0.95
_PHI_LIMIT = 0.99
_INNOV_FLOOR = 1e-3
_LOADING_LIMIT = 2.0
# Geometric midpoint of the [0.5, 2000] bracket `estep.fit_concentration` opens,
# so the first golden-section step starts unbiased in either direction.
_DIFFUSE_CONC = 30.0


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------


def _kmeans(
    feat: np.ndarray,
    n_clusters: int,
    rng: np.random.Generator,
    n_iter: int = 100,
) -> np.ndarray:
    """
    Hard k-means assignment of the rows of `feat`, as an `(T,)` integer array.

    Args:
        feat: `(T, d)` feature matrix.
        n_clusters: number of clusters; 1 or fewer puts every row in cluster 0.
        rng: source of the initial centers.
        n_iter: iteration ceiling; Lloyd's algorithm reaches a fixed point well
            inside it and the loop then breaks.
    """
    T = feat.shape[0]
    if n_clusters <= 1:
        return np.zeros(T, dtype=np.int64)
    picks = rng.choice(T, size=n_clusters, replace=T < n_clusters)
    centers = np.array(feat[picks], dtype=float)
    labels = np.full(T, -1, dtype=np.int64)
    for _ in range(n_iter):
        dist = ((feat[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        assigned = dist.argmin(axis=1).astype(np.int64)
        if np.array_equal(assigned, labels):
            break
        labels = assigned
        counts = np.bincount(labels, minlength=n_clusters)[:, None]
        sums = np.zeros_like(centers)
        np.add.at(sums, labels, feat)
        # An emptied cluster keeps its previous center rather than collapsing
        # onto the origin, which would pull every later assignment toward it.
        centers = np.where(counts > 0, sums / np.maximum(counts, 1), centers)
    return labels


def _group_means(
    values: np.ndarray,
    labels: np.ndarray,
    n_groups: int,
) -> np.ndarray:
    """Column means per group, an unused group taking the overall mean."""
    counts = np.bincount(labels, minlength=n_groups)[:, None]
    sums = np.zeros((n_groups, values.shape[1]))
    np.add.at(sums, labels, values)
    return np.where(counts > 0, sums / np.maximum(counts, 1), values.mean(axis=0))


def _centered_group_means(
    values: np.ndarray,
    labels: np.ndarray,
    n_groups: int,
) -> np.ndarray:
    """
    Per-group column means with the mean over groups removed.

    The generator centers its regime effects across regimes, so the intercept
    carries the level and the regime block carries only the contrast.  An
    uncentered start puts the level in the wrong block and the M-step has to
    walk it back out.
    """
    means = _group_means(values, labels, n_groups)
    return means - means.mean(axis=0, keepdims=True)


def _nb_moments(
    values: np.ndarray,
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-column mean and negative binomial dispersion of `values` over `mask`.

    The dispersion inverts `var = mu + mu^2 / r`.  Underdispersed columns, for
    which that has no positive solution, land on the upper clip.

    Args:
        values: `(T, L)` observations.
        mask: `(T, L)` boolean, the entries each column's moments run over.
    """
    n = np.maximum(mask.sum(axis=0), 1)
    kept = np.where(mask, values, 0.0)
    mean = kept.sum(axis=0) / n
    var = np.where(mask, (values - mean) ** 2, 0.0).sum(axis=0) / n
    mean = np.maximum(mean, _MEAN_FLOOR)
    disp = np.clip(mean**2 / np.maximum(var - mean, _EPS), _DISP_LO, _DISP_HI)
    return mean, disp


def _mean_log_share(values: np.ndarray, block: BlockIndex) -> np.ndarray:
    """
    Per-label mean over epochs of the log share an epoch takes of its parent block.

    The half count added to every entry keeps an epoch that took nothing inside
    the log; the share is then bounded away from 0 and from 1.
    """
    smoothed = values.astype(float) + 0.5
    total = block.broadcast(block.sum(smoothed))
    return np.log(smoothed / total).mean(axis=0)


def _pca(resid: np.ndarray, r: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Top-`r` loadings `(r, L)` and unit-variance scores `(T, r)` of `resid`.

    The loadings absorb the score scale, so `scores @ loadings` reproduces the
    rank-`r` part of `resid` while the scores stay on the unit scale the factor
    prior assumes.  A short or narrow `resid` carries fewer than `r` singular
    directions; the rows past `min(T, L)` stay zero.
    """
    T, L = resid.shape
    left, sv, right = np.linalg.svd(resid, full_matrices=False)
    n = min(r, right.shape[0])
    comps = np.zeros((r, L))
    comps[:n] = right[:n]
    scores = np.zeros((T, r))
    scores[:, :n] = left[:, :n] * sv[:n]
    sd = np.maximum(scores.std(axis=0), _EPS)
    return comps * sd[:, None], scores / sd


def _ar1(scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-component AR(1) coefficient and innovation variance of `scores`."""
    T = scores.shape[0]
    lead, lag = scores[1:], scores[:-1]
    phi = np.clip(
        (lead * lag).sum(axis=0) / np.maximum((lag * lag).sum(axis=0), _EPS),
        -_PHI_LIMIT,
        _PHI_LIMIT,
    )
    innov = ((lead - phi * lag) ** 2).sum(axis=0) / max(T - 1, 1)
    return phi, np.maximum(innov, _INNOV_FLOOR)


def _hard_summaries(labels: np.ndarray, n_regimes: int) -> PosteriorSummaries:
    """Posterior summaries of a hard regime assignment, as the estimators expect."""
    T = labels.shape[0]
    gamma = np.zeros((T, n_regimes))
    gamma[np.arange(T), labels] = 1.0
    xi = np.zeros((T - 1, n_regimes, n_regimes))
    xi[np.arange(T - 1), labels[:-1], labels[1:]] = 1.0
    return PosteriorSummaries(
        gamma=gamma,
        xi=xi,
        m_H=np.zeros((T, 0)),
        Sigma_H=np.zeros((T, 0, 0)),
        log_Z=0.0,
    )


# ---------------------------------------------------------------------------
# Per-block builders
# ---------------------------------------------------------------------------


def _build_regime(
    labels: np.ndarray,
    M: int,
    parent_M: int,
    d_ctx: int,
) -> RegimeParams:
    """Regime chain whose table is the transition count of the k-means path."""
    base = RegimeParams(
        M=M,
        parent_M=parent_M,
        d_ctx=d_ctx,
        init=np.zeros(M),
        bias=np.zeros(M),
        trans=np.zeros((M, M)),
        cross=np.zeros((M, parent_M)) if parent_M > 0 else None,
        ctx=np.zeros((M, d_ctx)),
    )
    init, trans = estimate_transition_table(_hard_summaries(labels, M))
    return regime_params_from_table(base, init, trans)


def _build_factor(
    resid: np.ndarray,
    M: int,
    r: int,
    d_ctx: int,
) -> Tuple[FactorParams, np.ndarray]:
    """
    Factor state-space block and the `(r, L)` emission loadings that go with it.

    Args:
        resid: `(T, L)` standardized log1p counts with the per-cluster means
            removed, so what is left is the co-movement the factor explains.
    """
    L = resid.shape[1]
    if r == 0:
        factor = FactorParams(
            M=M,
            r=0,
            d_ctx=d_ctx,
            F=np.zeros((M, 0, 0)),
            G=np.zeros((M, 0, d_ctx)),
            Sigma=np.zeros((M, 0, 0)),
            mu0=np.zeros(0),
            Sigma0=np.zeros((0, 0)),
        )
        return factor, np.zeros((0, L))

    loadings, scores = _pca(resid, r)
    phi, innov = _ar1(scores)
    factor = FactorParams(
        M=M,
        r=r,
        d_ctx=d_ctx,
        F=np.tile(np.diag(phi), (M, 1, 1)),
        G=np.zeros((M, r, d_ctx)),
        Sigma=np.tile(np.diag(innov), (M, 1, 1)),
        mu0=np.zeros(r),
        Sigma0=np.eye(r),
    )
    return factor, loadings


_LEVEL_RIDGE = 1.0


def _lag_share(values: np.ndarray) -> np.ndarray:
    """Per-label part of a series' log level that its own lag-1 values explain."""
    T, L = values.shape
    if T < 3:
        return np.zeros(L)
    target = np.log1p(values[1:])
    lagged = np.log1p(values[:-1])
    design = np.concatenate([np.ones((T - 1, 1)), lagged], axis=1)
    penalty = _LEVEL_RIDGE * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    try:
        coef = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    except np.linalg.LinAlgError:
        return np.zeros(L)
    return np.maximum(lagged.mean(axis=0) @ coef[1:], 0.0)


def _build_coarse(
    X: np.ndarray,
    Y: np.ndarray,
    labels: np.ndarray,
    loadings: np.ndarray,
    M: int,
    r: int,
    d_ctx: int,
    lag_free_level: bool = True,
) -> Tuple[CoarseCountParams, CoarseDurParams]:
    """Zero-inflated negative binomial counts and negative binomial durations."""
    L = X.shape[1]
    counts = X.astype(float)
    everything = np.ones_like(counts, dtype=bool)
    _, r_disp = _nb_moments(counts, everything)

    fires = counts > 0
    n_fires = np.maximum(fires.sum(axis=0), 1)
    mu = counts.sum(axis=0) / n_fires
    share = _lag_share(counts) if lag_free_level else np.zeros(L)
    mu = np.maximum(mu * np.exp(-share), _MEAN_FLOOR)

    # Excess zeros: what the fitted negative binomial cannot account for.
    zero_frac = 1.0 - fires.sum(axis=0) / counts.shape[0]
    nb_zero = np.exp(r_disp * np.log(r_disp / (r_disp + mu)))
    pi = np.clip(
        (zero_frac - nb_zero) / np.maximum(1.0 - nb_zero, _EPS), _ZI_LO, _ZI_HI
    )

    dur_mean, s_disp = _nb_moments(Y.astype(float), fires)
    dur_mean = np.maximum(dur_mean * np.exp(-_lag_share(Y.astype(float))), _MEAN_FLOOR)

    count = CoarseCountParams(
        L=L,
        beta0=np.log(mu),
        betaS=_centered_group_means(np.log1p(counts), labels, M),
        betaH=loadings,
        betaCtx=np.zeros((d_ctx, L)),
        betaLag={},
        r_disp=r_disp,
        gamma0=np.log(pi / (1.0 - pi)),
        gammaS=np.zeros((M, L)),
        gammaH=np.zeros((r, L)),
        gammaCtx=np.zeros((d_ctx, L)),
    )
    dur = CoarseDurParams(
        L=L,
        alpha0=np.log(dur_mean),
        alphaS=_centered_group_means(np.log1p(Y.astype(float)), labels, M),
        alphaH=np.zeros((r, L)),
        alphaCtx=np.zeros((d_ctx, L)),
        alphaX=np.zeros(L),
        alphaLag={},
        s_disp=s_disp,
    )
    return count, dur


def _build_fine(
    X: np.ndarray,
    Y: np.ndarray,
    labels: np.ndarray,
    loadings: np.ndarray,
    block: BlockIndex,
    M: int,
    r: int,
    d_ctx: int,
) -> Tuple[FineCountParams, FineDurParams]:
    """Dirichlet-multinomial allocation scores for a scale below the coarsest."""
    L = X.shape[1]
    count = FineCountParams(
        L=L,
        beta0=_mean_log_share(X, block),
        betaS=_centered_group_means(np.log1p(X.astype(float)), labels, M),
        betaH=loadings,
        betaCtx=np.zeros((d_ctx, L)),
        betaP=np.zeros(L),
        betaD=np.zeros(L),
        betaLag={},
        kappa=_DIFFUSE_CONC,
        betaDep=np.zeros(L),
    )
    dur = FineDurParams(
        L=L,
        alpha0=_mean_log_share(Y, block),
        alphaS=_centered_group_means(np.log1p(Y.astype(float)), labels, M),
        alphaH=np.zeros((r, L)),
        alphaCtx=np.zeros((d_ctx, L)),
        alphaX=np.zeros(L),
        alphaP=np.zeros(L),
        alphaLag={},
        tau=_DIFFUSE_CONC,
        alphaDep=np.zeros(L),
    )
    return count, dur


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def initialize_from_data(
    data: MultiScaleData,
    M_k: List[int],
    r_k: List[int],
    max_lag: List[int],
    d_ctx: Union[int, Sequence[int]] = 0,
    seed: int = 0,
    lag_free_level: bool = True,
    every_scale_coarse: bool = False,
) -> List[ScaleParams]:
    """
    One `ScaleParams` per scale, estimated from `data` with an empty graph.

    Args:
        data: observed counts, durations and context at every scale.
        M_k: regimes per scale, index 0 = scale 1 = finest.
        r_k: factor dimension per scale, same order.
        max_lag: lag budget per scale, same order.  Every lag coefficient
            dictionary starts empty, so this fixes only the length contract the
            caller will hand to the structure step.
        d_ctx: context width the parameters carry, one value for every scale or
            one per scale.  Whichever form it takes, it has to equal the width
            observed at that scale, 0 included, because the regime chain
            multiplies the observed matrix by the context block whenever the
            observed matrix is non-empty, so a width that disagrees with it
            fails inside the E-step rather than here.  A per-scale sequence is
            what lets a calendar feature live at one scale and not another.
        seed: fixes the k-means initialization, and with it the whole result.
        every_scale_coarse: give every scale its own emission instead of an
            allocation, so no scale is tied to the total of the one above it.
            This is what a model fitted at one granularity at a time looks like,
            and it is the arm the coherence gap of `evaluation.coherence` is
            measured against.

    Returns:
        Parameters for scale 1 first, scale K last.

    Raises:
        ValueError: `M_k`, `r_k` or `max_lag` is not K long, or `d_ctx` does not
            match the observed context width.
    """
    K = data.K
    if not (len(M_k) == len(r_k) == len(max_lag) == K):
        raise ValueError(
            f"M_k, r_k and max_lag must each hold {K} entries, one per scale, "
            f"got {len(M_k)}, {len(r_k)} and {len(max_lag)}"
        )
    declared = [int(d_ctx)] * K if isinstance(d_ctx, int) else [int(w) for w in d_ctx]
    if len(declared) != K:
        raise ValueError(
            f"d_ctx must be one width or {K} of them, one per scale, "
            f"got {len(declared)}"
        )
    observed = [int(data.get_C(k).shape[1]) for k in range(1, K + 1)]
    if declared != observed:
        raise ValueError(
            f"d_ctx={d_ctx} does not match the observed context width(s) "
            f"{observed}; the regime chain keys its context term off the "
            f"observed width, so a narrower context block fails in the E-step"
        )

    rng = np.random.default_rng(seed)
    out: List[ScaleParams] = []

    for k in range(1, K + 1):
        X = data.get_X(k)
        Y = data.get_Y(k)
        M = M_k[k - 1]
        r = r_k[k - 1]
        L = X.shape[1]
        d_ctx_k = declared[k - 1]

        log_counts = np.log1p(X.astype(float))
        labels = _kmeans(log_counts, M, rng)

        # What the regimes already explain is not the factor's to explain.
        resid = log_counts - _group_means(log_counts, labels, M)[labels]
        col_sd = np.maximum(resid.std(axis=0), _EPS)
        factor, loadings = _build_factor(resid / col_sd, M, r, d_ctx_k)
        loadings = np.clip(loadings * col_sd[None, :], -_LOADING_LIMIT, _LOADING_LIMIT)

        regime = _build_regime(labels, M, M_k[k] if k < K else 0, d_ctx_k)
        parent_sets: Dict[int, Set[LagKey]] = {ell: set() for ell in range(L)}

        if k == K or every_scale_coarse:
            count, dur = _build_coarse(
                X, Y, labels, loadings, M, r, d_ctx_k, lag_free_level
            )
            out.append(
                ScaleParams(
                    k=k,
                    regime=regime,
                    factor=factor,
                    parent_sets=parent_sets,
                    coarse_count=count,
                    coarse_dur=dur,
                )
            )
        else:
            block = BlockIndex.from_hierarchy(data.hierarchy, k)
            count, dur = _build_fine(X, Y, labels, loadings, block, M, r, d_ctx_k)
            out.append(
                ScaleParams(
                    k=k,
                    regime=regime,
                    factor=factor,
                    parent_sets=parent_sets,
                    fine_count=count,
                    fine_dur=dur,
                )
            )

    return out
