"""
evaluation/metrics.py
=====================
All evaluation metrics from Section 8 of the paper.

Metric groups
-------------
PredictiveMetrics  : NLL, scale-NLL, MAE_X, MAE_Y
ReconciliationMetrics : RecErr_X, RecErr_Y
StructureMetrics   : Prec, Rec, F1, SHD per scale
LatentMetrics      : Acc_S, ARI, NMI, factor correlation
ReconstructionMetrics : start-F1, IoU, episode match error
CalibrationMetrics : coverage at 80/90/95% prediction intervals
ParameterMetrics   : transition-table TV, lag/emission/factor/allocation error

Usage
-----
    from src.evaluation.metrics import (
        PredictiveMetrics, StructureMetrics, LatentMetrics,
        ReconstructionMetrics, CalibrationMetrics, ReconciliationMetrics,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.special import xlogy

from ..core.graph import MultiScaleDBNGraph
from ..core.variables import MultiScaleData
from ..inference.reconstruction import EpisodeReconstructor
from ..model.params import RegimeParams, ScaleParams

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    return num / den if den > 0 else default


# ---------------------------------------------------------------------------
# Predictive metrics
# ---------------------------------------------------------------------------


@dataclass
class PredictiveMetrics:
    """
    Predictive performance metrics evaluated on held-out data.

    Attributes
    ----------
    nll          : total NLL (sum over all scales and epochs)
    nll_per_scale : NLL per scale k (list, index 0 = scale 1)
    mae_X        : mean absolute error for counts, per scale
    mae_Y        : mean absolute error for durations, per scale
    """

    nll: float = 0.0
    nll_per_scale: List[float] = field(default_factory=list)
    mae_X: List[float] = field(default_factory=list)
    mae_Y: List[float] = field(default_factory=list)

    @classmethod
    def compute(
        cls,
        data: MultiScaleData,
        log_likes_per_scale: List[np.ndarray],  # [(T_k, M), ...]; index 0 = scale 1
        gamma_per_scale: List[np.ndarray],  # [(T_k, M), ...]
        pred_X_per_scale: List[np.ndarray],  # [(T_k, L), ...]  - predicted E[X]
        pred_Y_per_scale: List[np.ndarray],  # [(T_k, L), ...]
    ) -> "PredictiveMetrics":
        """
        Compute predictive metrics from per-scale posterior distributions.

        Parameters
        ----------
        data                : observed multiscale data
        log_likes_per_scale : list of (T_k, M) log-likelihood arrays (one per scale)
        gamma_per_scale     : list of (T_k, M) regime posteriors
        pred_X_per_scale    : list of (T_k, L) predicted count means
        pred_Y_per_scale    : list of (T_k, L) predicted duration means
        """
        K = data.K
        nll_per_scale = []
        mae_X = []
        mae_Y = []

        for ki in range(K):
            k = ki + 1
            log_like = log_likes_per_scale[ki]  # (T_k, M)
            gamma = gamma_per_scale[ki]  # (T_k, M)

            # Marginalised log-likelihood: Σ_t log(Σ_m γ_t(m) exp(ll_t,m))
            # = Σ_t logsumexp_m(ll_t,m + log γ_t(m))
            log_joint = log_like + np.log(np.clip(gamma, 1e-300, None))
            nll_k = -float(np.sum(np.logaddexp.reduce(log_joint, axis=1)))
            nll_per_scale.append(nll_k)

            X_k = data.get_X(k).astype(float)
            Y_k = data.get_Y(k).astype(float)
            pred_X = pred_X_per_scale[ki]
            pred_Y = pred_Y_per_scale[ki]

            mae_X.append(float(np.mean(np.abs(X_k - pred_X))))
            mae_Y.append(float(np.mean(np.abs(Y_k - pred_Y))))

        return cls(
            nll=sum(nll_per_scale),
            nll_per_scale=nll_per_scale,
            mae_X=mae_X,
            mae_Y=mae_Y,
        )

    def to_dict(self) -> Dict[str, float]:
        d: Dict[str, float] = {"NLL": self.nll}
        for k, v in enumerate(self.nll_per_scale, start=1):
            d[f"NLL_k{k}"] = v
        for k, v in enumerate(self.mae_X, start=1):
            d[f"MAE_X_k{k}"] = v
        for k, v in enumerate(self.mae_Y, start=1):
            d[f"MAE_Y_k{k}"] = v
        return d


# ---------------------------------------------------------------------------
# Reconciliation metrics
# ---------------------------------------------------------------------------


@dataclass
class ReconciliationMetrics:
    """
    Reconciliation error: checks whether multi-scale predictions are consistent.

    RecErr_X = Σ_{k,c,ℓ} | X_c^(k+1) − Σ_{t∈ch(c)} X_t^(k) | / total_X
    RecErr_Y = analogous for Y
    """

    rec_err_X: float = 0.0
    rec_err_Y: float = 0.0

    @classmethod
    def compute(cls, data: MultiScaleData) -> "ReconciliationMetrics":
        errs = data.check_reconciliation()
        return cls(rec_err_X=errs["RecErr_X"], rec_err_Y=errs["RecErr_Y"])

    def to_dict(self) -> Dict[str, float]:
        return {"RecErr_X": self.rec_err_X, "RecErr_Y": self.rec_err_Y}


# ---------------------------------------------------------------------------
# Structure metrics
# ---------------------------------------------------------------------------


@dataclass
class StructureMetrics:
    """
    Per-scale structure recovery metrics comparing estimated to true graph.

    Attributes
    ----------
    precision_per_scale : true positives / (TP + FP)
    recall_per_scale    : true positives / (TP + FN)
    f1_per_scale        : harmonic mean of P and R
    shd_per_scale       : structural Hamming distance (insertions + deletions)
    self_recall_per_scale  : recall over true edges whose source is its target
    cross_recall_per_scale : recall over true edges between different labels

    The last two exist because a model can be blind to one kind and not the
    other, which a pooled recall hides.  Both are NaN at a scale that carries no
    true edge of that kind, so an average over scales has to skip them.
    """

    precision_per_scale: List[float] = field(default_factory=list)
    recall_per_scale: List[float] = field(default_factory=list)
    f1_per_scale: List[float] = field(default_factory=list)
    shd_per_scale: List[int] = field(default_factory=list)
    self_recall_per_scale: List[float] = field(default_factory=list)
    cross_recall_per_scale: List[float] = field(default_factory=list)

    @classmethod
    def compute(
        cls,
        estimated: MultiScaleDBNGraph,
        true_graph: MultiScaleDBNGraph,
    ) -> "StructureMetrics":
        """
        Compare estimated graph against ground-truth.

        Parameters
        ----------
        estimated  : learned MultiScaleDBNGraph
        true_graph : ground-truth MultiScaleDBNGraph
        """
        K = estimated.K
        precision_per_scale = []
        recall_per_scale = []
        f1_per_scale = []
        shd_per_scale = []
        self_recall_per_scale = []
        cross_recall_per_scale = []

        for k in range(1, K + 1):
            L = estimated.L
            est_edges: set = set()
            true_edges: set = set()

            for ell in range(L):
                for src_ell, lag in estimated.get_parents(k, ell):
                    est_edges.add((k, src_ell, lag, ell))
                for src_ell, lag in true_graph.get_parents(k, ell):
                    true_edges.add((k, src_ell, lag, ell))

            tp = len(est_edges & true_edges)
            fp = len(est_edges - true_edges)
            fn = len(true_edges - est_edges)

            if not est_edges and not true_edges:
                # A scale with no true edges and none predicted is perfectly
                # recovered; scoring it 0 (0/0) dragged F1_mean down by a whole
                # scale in every scenario whose coarsest scale has no edges.
                prec = rec = f1 = 1.0
            else:
                prec = _safe_div(tp, tp + fp)
                rec = _safe_div(tp, tp + fn)
                f1 = _safe_div(2 * prec * rec, prec + rec)
            shd = fp + fn

            precision_per_scale.append(prec)
            recall_per_scale.append(rec)
            f1_per_scale.append(f1)
            shd_per_scale.append(shd)

            for kind, keep in (
                (self_recall_per_scale, lambda e: e[1] == e[3]),
                (cross_recall_per_scale, lambda e: e[1] != e[3]),
            ):
                subset = {e for e in true_edges if keep(e)}
                kind.append(
                    float("nan")
                    if not subset
                    else _safe_div(len(subset & est_edges), len(subset))
                )

        return cls(
            precision_per_scale=precision_per_scale,
            recall_per_scale=recall_per_scale,
            f1_per_scale=f1_per_scale,
            shd_per_scale=shd_per_scale,
            self_recall_per_scale=self_recall_per_scale,
            cross_recall_per_scale=cross_recall_per_scale,
        )

    def to_dict(self) -> Dict[str, float]:
        d: Dict[str, float] = {}
        for k, (p, r, f, s) in enumerate(
            zip(
                self.precision_per_scale,
                self.recall_per_scale,
                self.f1_per_scale,
                self.shd_per_scale,
            ),
            start=1,
        ):
            d[f"Prec_k{k}"] = p
            d[f"Rec_k{k}"] = r
            d[f"F1_k{k}"] = f
            d[f"SHD_k{k}"] = float(s)
        for k, (sr, cr) in enumerate(
            zip(self.self_recall_per_scale, self.cross_recall_per_scale), start=1
        ):
            d[f"SelfRec_k{k}"] = sr
            d[f"CrossRec_k{k}"] = cr
        # Macro averages
        if self.f1_per_scale:
            d["F1_mean"] = float(np.mean(self.f1_per_scale))
            d["SHD_total"] = float(sum(self.shd_per_scale))
        return d


# ---------------------------------------------------------------------------
# Latent metrics
# ---------------------------------------------------------------------------


@dataclass
class LatentMetrics:
    """
    Latent variable recovery quality.

    Attributes
    ----------
    acc_S_per_scale : best-permutation accuracy for regime sequence S^(k)
    ari_per_scale   : adjusted Rand index
    nmi_per_scale   : normalised mutual information
    factor_corr     : mean absolute correlation between inferred and true H
                      (best linear match, per scale)

    All three regime figures are NaN at a scale whose generating path never
    leaves one state, since a constant labelling is recovered exactly by any
    method and a perfect score there would read as recovery where there is
    nothing to recover.  The factor already reported NaN when a scale carries
    none, and this is the same rule for the other latent.  A scale where the
    truth switches and the fit does not is scored rather than skipped, since
    what that returns is the modal share and is the number a constant predictor
    earns.
    """

    acc_S_per_scale: List[float] = field(default_factory=list)
    ari_per_scale: List[float] = field(default_factory=list)
    nmi_per_scale: List[float] = field(default_factory=list)
    factor_corr: List[float] = field(default_factory=list)

    @classmethod
    def compute(
        cls,
        gamma_per_scale: List[np.ndarray],  # [(T_k, M)]
        true_S_per_scale: List[np.ndarray],  # [(T_k,)] integer labels
        m_H_per_scale: List[np.ndarray],  # [(T_k, r)]
        true_H_per_scale: List[np.ndarray],  # [(T_k, r_true)]
    ) -> "LatentMetrics":
        K = len(gamma_per_scale)
        acc_S = []
        ari = []
        nmi = []
        factor_corr = []

        for ki in range(K):
            gamma = gamma_per_scale[ki]
            S_pred = np.argmax(gamma, axis=1)
            S_true = true_S_per_scale[ki]

            # Best-permutation accuracy, and nothing to score on a constant path
            if np.unique(S_true).size <= 1:
                acc_S.append(float("nan"))
                ari.append(float("nan"))
                nmi.append(float("nan"))
            else:
                acc_S.append(_best_perm_accuracy(S_pred, S_true, gamma.shape[1]))
                ari.append(_adjusted_rand_index(S_true, S_pred))
                nmi.append(_normalised_mutual_info(S_true, S_pred))

            # Factor correlation (if r > 0)
            factor_corr.append(
                best_factor_corr(m_H_per_scale[ki], true_H_per_scale[ki])
            )

        return cls(
            acc_S_per_scale=acc_S,
            ari_per_scale=ari,
            nmi_per_scale=nmi,
            factor_corr=factor_corr,
        )

    def to_dict(self) -> Dict[str, float]:
        d: Dict[str, float] = {}
        for k, (a, r, n, fc) in enumerate(
            zip(
                self.acc_S_per_scale,
                self.ari_per_scale,
                self.nmi_per_scale,
                self.factor_corr,
            ),
            start=1,
        ):
            d[f"AccS_k{k}"] = a
            d[f"ARI_k{k}"] = r
            d[f"NMI_k{k}"] = n
            d[f"FactCorr_k{k}"] = fc
        return d


# ---------------------------------------------------------------------------
# Reconstruction metrics
# ---------------------------------------------------------------------------


@dataclass
class ReconstructionMetrics:
    """
    Episode reconstruction quality at the finest scale.

    Attributes
    ----------
    start_f1 : F1 for start-point detection (per label, averaged)
    iou      : mean IoU between matched true and predicted episodes
    """

    start_f1: float = 0.0
    iou: float = 0.0

    @classmethod
    def compute(
        cls,
        true_X1: np.ndarray,  # (T, L) finest-scale counts (ground truth)
        true_Y1: np.ndarray,  # (T, L) finest-scale durations
        pred_X1: np.ndarray,  # (T, L) predicted
        pred_Y1: np.ndarray,  # (T, L) predicted
        tol: int = 1,
    ) -> "ReconstructionMetrics":
        """
        Compare predicted vs. true finest-scale episode traces.

        Parameters
        ----------
        true_X1, true_Y1 : ground-truth finest-scale arrays
        pred_X1, pred_Y1 : predicted finest-scale arrays (binarised at 0.5)
        tol              : start-detection tolerance in slots
        """
        pred_X1_bin = (pred_X1 >= 0.5).astype(int)
        pred_Y1_int = np.round(pred_Y1).astype(int)

        L = true_X1.shape[1]
        T = true_X1.shape[0]
        label_names = [f"l{i}" for i in range(L)]
        rec = EpisodeReconstructor(label_names)

        traces_true = rec.reconstruct(true_X1.astype(int), true_Y1.astype(int))
        traces_pred = rec.reconstruct(pred_X1_bin, pred_Y1_int)

        ious = rec.iou(traces_pred, traces_true, T)
        f1s = rec.start_detection_f1(traces_pred, traces_true)

        f1_per_label = []
        iou_per_label = []
        for ell in range(L):
            f1_per_label.append(f1s[ell]["f1"])
            iou_per_label.append(ious[ell])

        return cls(
            start_f1=float(np.mean(f1_per_label)),
            iou=float(np.mean(iou_per_label)),
        )

    def to_dict(self) -> Dict[str, float]:
        return {"StartF1": self.start_f1, "IoU": self.iou}


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------


@dataclass
class CalibrationMetrics:
    """
    Interval coverage: fraction of true values falling within the predicted
    80%, 90%, and 95% prediction intervals.
    """

    coverage_80: float = 0.0
    coverage_90: float = 0.0
    coverage_95: float = 0.0

    @classmethod
    def compute(
        cls,
        true_values: np.ndarray,  # (N,) or (N, D) flattened true obs
        pred_mean: np.ndarray,  # same shape as true_values
        pred_std: np.ndarray,  # predictive std (Gaussian approximation)
    ) -> "CalibrationMetrics":
        """
        Gaussian predictive interval coverage.

        Parameters
        ----------
        true_values : ground-truth values (any shape, flattened)
        pred_mean   : predictive mean
        pred_std    : predictive standard deviation
        """
        true_flat = true_values.ravel().astype(float)
        mu = pred_mean.ravel().astype(float)
        sig = np.maximum(pred_std.ravel().astype(float), 1e-8)

        z = np.abs(true_flat - mu) / sig

        return cls(
            coverage_80=float(np.mean(z <= 1.2816)),  # 80% two-sided
            coverage_90=float(np.mean(z <= 1.6449)),  # 90%
            coverage_95=float(np.mean(z <= 1.9600)),  # 95%
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "Coverage80": self.coverage_80,
            "Coverage90": self.coverage_90,
            "Coverage95": self.coverage_95,
        }


# ---------------------------------------------------------------------------
# Latent metric helpers (ARI / NMI without sklearn dependency)
# ---------------------------------------------------------------------------


def _contingency(labels_true: np.ndarray, labels_pred: np.ndarray) -> np.ndarray:
    """Build contingency matrix C[i,j] = |{t: true=i, pred=j}|."""
    classes_true = np.unique(labels_true)
    classes_pred = np.unique(labels_pred)
    C = np.zeros((len(classes_true), len(classes_pred)), dtype=np.int64)
    for i, ci in enumerate(classes_true):
        mask = labels_true == ci
        for j, cj in enumerate(classes_pred):
            C[i, j] = int(np.sum(mask & (labels_pred == cj)))
    return C


def _adjusted_rand_index(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Adjusted Rand Index (Hubert & Arabie, 1985)."""
    C = _contingency(labels_true, labels_pred)
    n = C.sum()
    sum_comb_C = float(np.sum(C * (C - 1)) / 2)
    a = C.sum(axis=1)
    b = C.sum(axis=0)
    sum_comb_a = float(np.sum(a * (a - 1)) / 2)
    sum_comb_b = float(np.sum(b * (b - 1)) / 2)
    comb_n = float(n * (n - 1) / 2)
    expected = sum_comb_a * sum_comb_b / comb_n if comb_n > 0 else 0.0
    max_val = (sum_comb_a + sum_comb_b) / 2
    denom = max_val - expected
    if denom == 0:
        return 1.0 if sum_comb_C == expected else 0.0
    return (sum_comb_C - expected) / denom


def _normalised_mutual_info(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Normalised Mutual Information (geometric mean normalisation)."""
    C = _contingency(labels_true, labels_pred)
    n = float(C.sum())
    if n == 0:
        return 0.0
    P = C / n
    p_true = P.sum(axis=1, keepdims=True)
    p_pred = P.sum(axis=0, keepdims=True)
    outer = p_true * p_pred
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(outer > 0, P / outer, 1.0)
        mi = float(np.sum(xlogy(P, ratio)))
    H_true = float(-np.sum(xlogy(p_true.ravel(), p_true.ravel())))
    H_pred = float(-np.sum(xlogy(p_pred.ravel(), p_pred.ravel())))
    denom = np.sqrt(H_true * H_pred)
    return mi / denom if denom > 0 else 0.0


def best_factor_corr(H_hat: np.ndarray, H_true: np.ndarray) -> float:
    """
    Mean absolute correlation of each inferred factor with its best true match.

    NaN when either side carries no factor, since a scale with r = 0 has no
    trajectory to match rather than a badly matched one.
    """
    if H_hat.shape[1] == 0 or H_true.shape[1] == 0:
        return float("nan")
    r_hat = H_hat.shape[1]
    corr = np.abs(np.corrcoef(H_hat.T, H_true.T)[:r_hat, r_hat:])
    return float(np.mean(np.max(corr, axis=1)))


def _best_perm_accuracy(S_pred: np.ndarray, S_true: np.ndarray, M: int) -> float:
    """
    Best-permutation accuracy: find the label mapping that maximises accuracy.
    Uses a simple greedy matching on the confusion matrix.
    """
    from itertools import permutations

    if M <= 8:
        # Exact enumeration for small M
        best = 0.0
        for perm in permutations(range(M)):
            mapped = np.array([perm[s] for s in S_pred])
            acc = float(np.mean(mapped == S_true))
            if acc > best:
                best = acc
        return best
    else:
        # Greedy matching via confusion matrix (Hungarian approximation)
        C = _contingency(S_true, S_pred)
        mapping = {}
        used_pred = set()
        for _ in range(min(C.shape)):
            best_i, best_j = np.unravel_index(
                np.argmax(C * (~np.isin(np.arange(C.shape[1]), list(used_pred)))),
                C.shape,
            )
            mapping[best_j] = best_i
            C[best_i, :] = 0
            used_pred.add(best_j)
        mapped = np.array([mapping.get(s, -1) for s in S_pred])
        return float(np.mean(mapped == S_true))


# ---------------------------------------------------------------------------
# Parameter metrics
# ---------------------------------------------------------------------------

# Floor for the relative errors and the cosine similarities, so an exactly
# constant true block or a vanishing loading vector stays finite. It doubles as
# the threshold below which a loading vector counts as carrying no direction.
_PARAM_EPS = 1e-6

EdgeKey = Tuple[int, int, int]  # (src_label, lag, dst_label)


def _regime_level(p: ScaleParams) -> np.ndarray:
    """
    Return the (M, L) identified regime-conditional emission level of a scale.

    Only this combination is pinned down by the data. At the coarsest scale the
    count mean is `exp(beta0 + betaS[s] + ...)`, so shifting `beta0` up by a
    constant and `betaS` down by the same constant leaves every distribution
    untouched; the sum is what the data see. At a finer scale the allocation
    softmax runs over the children of a single block, which cancels every term
    that is constant inside that block (`beta0`, `betaP`, `betaD`, and any
    per-label offset of `betaS`), so only the contrast of `betaS` across regimes
    survives.
    """
    if p.is_coarsest:
        c = p.coarse_count
        return c.beta0[None, :] + c.betaS
    bs = p.fine_count.betaS
    return bs - np.mean(bs, axis=0, keepdims=True)


def _emission_blocks(p: ScaleParams, perm: np.ndarray) -> Dict[str, np.ndarray]:
    """Return the identified emission blocks of a scale under a regime relabeling."""
    blocks = {"level": _regime_level(p)[perm]}
    if p.is_coarsest:
        blocks["disp"] = p.coarse_count.r_disp
    return blocks


def _factor_loadings(p: ScaleParams) -> np.ndarray:
    """Return the (r, L) factor loading block of a scale."""
    if p.is_coarsest:
        return p.coarse_count.betaH
    return p.fine_count.betaH


def _lag_blocks(p: ScaleParams) -> List[Dict[Tuple[int, int], np.ndarray]]:
    """Return the count and duration lag coefficient dicts of a scale."""
    if p.is_coarsest:
        return [p.coarse_count.betaLag, p.coarse_dur.alphaLag]
    return [p.fine_count.betaLag, p.fine_dur.alphaLag]


def _true_support(p: ScaleParams) -> Set[EdgeKey]:
    """Return the true lagged edges of a scale as (src, lag, dst) triples."""
    return {
        (src, lag, dst)
        for dst, parents in p.parent_sets.items()
        for src, lag in parents
    }


def _regime_share(level: np.ndarray) -> np.ndarray:
    """Softmax of an (M, L) emission level down the regime axis."""
    e = np.exp(level - np.max(level, axis=0, keepdims=True))
    return e / np.sum(e, axis=0, keepdims=True)


def _best_param_permutation(
    true_p: ScaleParams,
    learned_p: ScaleParams,
) -> np.ndarray:
    """
    Regime relabeling that best aligns a learned scale with the true one.

    The regime label is identifiable only up to permutation, so every
    regime-indexed block has to be compared under one consistent relabeling.
    The match is read off the identified emission levels, so that a learned
    model differing only in a non-identified offset still matches exactly.

    Returns
    -------
    perm : (M,) integer array; `perm[m]` is the learned regime matching true
           regime `m`, so an aligned learned block is `block[perm]`.
    """
    from itertools import permutations

    M = true_p.M
    true_bs = _regime_level(true_p)
    learned_bs = _regime_level(learned_p)
    cost = np.sum((true_bs[:, None, :] - learned_bs[None, :, :]) ** 2, axis=2)

    if M <= 8:
        rows = np.arange(M)
        best_perm = rows
        best_cost = np.inf
        for perm in permutations(range(M)):
            c = float(np.sum(cost[rows, list(perm)]))
            if c < best_cost:
                best_cost = c
                best_perm = np.array(perm, dtype=int)
        return best_perm

    # Greedy assignment above the enumeration limit, mirroring the greedy
    # branch of `_best_perm_accuracy`.
    perm = np.zeros(M, dtype=int)
    remaining = cost.astype(float)
    for _ in range(M):
        i, j = np.unravel_index(np.argmin(remaining), remaining.shape)
        perm[i] = j
        remaining[i, :] = np.inf
        remaining[:, j] = np.inf
    return perm


def transition_cpt(
    regime: RegimeParams,
    parent_share: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Row-stochastic (M_prev, M_cur) transition table implied by the regime logits.

    Parameters
    ----------
    regime       : regime block supplying `bias` and `trans`
    parent_share : optional distribution over parent-scale regimes; the cross
                   term is averaged over it instead of being dropped
    """
    # Context and cross terms enter the logits additively, so evaluating them at
    # zero leaves the identified object: the transition table itself.
    logits = regime.bias[None, :] + regime.trans.T
    if parent_share is not None and regime.cross is not None:
        avg_cross = regime.cross @ np.asarray(parent_share, dtype=float)
        logits = logits + avg_cross[None, :]
    logits = logits - np.max(logits, axis=1, keepdims=True)
    P = np.exp(logits)
    return P / np.sum(P, axis=1, keepdims=True)


@dataclass
class ParameterMetrics:
    """
    Recovery of learned parameters against the true ones, per scale.

    Every block is compared in its identified form only, so two parameter sets
    that induce the same distribution score 0 (see `_regime_level`).

    Attributes
    ----------
    trans_tv       : mean total variation distance between transition tables,
                     0 perfect and 1 worst; NaN when M = 1
    lag_err        : mean absolute lag coefficient error on the true support
    lag_false      : absolute learned lag coefficient mass off the true support
    emis_err       : mean absolute emission block error
    emis_rel       : emission error divided by the spread of the true block
    fact_sub       : 1 - mean best absolute cosine between factor loadings;
                     NaN when r = 0 or every true loading vanishes
    alloc_share    : mean absolute difference of the child shares a regime
                     implies for the allocation; NaN at the coarsest scale
    param_err_mean : mean of the available scale-free errors over all scales
    param_err_n    : how many errors that mean was taken over

    `param_err_mean` averages whichever scale-free errors are identified, and
    which ones those are depends on the model: `trans_tv` is unavailable at
    M = 1 and `fact_sub` at r = 0.  Two fits whose `param_err_n` differ are
    therefore averaging different quantities and their means do not compare.
    `param_err_n` is reported beside the mean so that a reader can tell.
    """

    trans_tv: List[float] = field(default_factory=list)
    lag_err: List[float] = field(default_factory=list)
    lag_false: List[float] = field(default_factory=list)
    emis_err: List[float] = field(default_factory=list)
    emis_rel: List[float] = field(default_factory=list)
    fact_sub: List[float] = field(default_factory=list)
    alloc_share: List[float] = field(default_factory=list)
    param_err_mean: float = float("nan")
    param_err_n: int = 0

    @classmethod
    def compute(
        cls,
        true_params: List[ScaleParams],
        learned_params: List[ScaleParams],
    ) -> "ParameterMetrics":
        """
        Score learned parameters against the true ones, scale by scale.

        Parameters
        ----------
        true_params    : ground-truth ScaleParams, index k-1 = scale k
        learned_params : estimated ScaleParams in the same order

        Raises
        ------
        ValueError : if the two lists cover a different number of scales, which
                     would otherwise be scored on the common prefix in silence
        """
        if len(true_params) != len(learned_params):
            raise ValueError(
                f"scale count mismatch: {len(true_params)} true, "
                f"{len(learned_params)} learned"
            )

        trans_tv: List[float] = []
        lag_err: List[float] = []
        lag_false: List[float] = []
        emis_err: List[float] = []
        emis_rel: List[float] = []
        fact_sub: List[float] = []
        alloc_share: List[float] = []

        for tp, lp in zip(true_params, learned_params):
            perm = _best_param_permutation(tp, lp)

            if tp.M == 1:
                trans_tv.append(float("nan"))
            else:
                P_true = transition_cpt(tp.regime)
                P_learn = transition_cpt(lp.regime)[np.ix_(perm, perm)]
                tv = 0.5 * np.sum(np.abs(P_true - P_learn), axis=1)
                trans_tv.append(float(np.mean(tv)))

            support = _true_support(tp)
            errs: List[float] = []
            false_mass = 0.0
            for t_dict, l_dict in zip(_lag_blocks(tp), _lag_blocks(lp)):
                for src, lag, dst in support:
                    a = t_dict[(src, lag)][dst] if (src, lag) in t_dict else 0.0
                    b = l_dict[(src, lag)][dst] if (src, lag) in l_dict else 0.0
                    errs.append(abs(float(a) - float(b)))
                for (src, lag), vec in l_dict.items():
                    for dst in range(tp.L):
                        if (src, lag, dst) not in support:
                            false_mass += abs(float(vec[dst]))
            lag_err.append(float(np.mean(errs)) if errs else 0.0)
            lag_false.append(false_mass)

            t_blocks = _emission_blocks(tp, np.arange(tp.M))
            l_blocks = _emission_blocks(lp, perm)
            maes: List[float] = []
            rels: List[float] = []
            for name, t_arr in t_blocks.items():
                mae = float(np.mean(np.abs(t_arr - l_blocks[name])))
                maes.append(mae)
                rels.append(mae / (float(np.std(t_arr)) + _PARAM_EPS))
            emis_err.append(float(np.mean(maes)))
            emis_rel.append(float(np.mean(rels)))

            H_true = _factor_loadings(tp)
            H_learn = _factor_loadings(lp)
            # A loading vector of length zero points nowhere, so it carries no
            # row space to recover and is dropped rather than scored a miss.
            n_true = np.linalg.norm(H_true, axis=1)
            keep = n_true > _PARAM_EPS
            if H_learn.shape[0] == 0 or not np.any(keep):
                fact_sub.append(float("nan"))
            else:
                # Only the row space of the loadings is identified, since
                # rotating the factors rotates the loadings back.
                A = H_true[keep] / n_true[keep][:, None]
                n_learn = np.linalg.norm(H_learn, axis=1, keepdims=True)
                B = H_learn / np.maximum(n_learn, _PARAM_EPS)
                cos = np.clip(np.abs(A @ B.T), 0.0, 1.0)
                fact_sub.append(float(1.0 - np.mean(np.max(cos, axis=1))))

            if tp.is_coarsest:
                alloc_share.append(float("nan"))
            else:
                # The allocation softmax runs over the children of one block, so
                # the share a regime claims is the softmax of the emission level
                # down each label column.
                share_t = _regime_share(t_blocks["level"])
                share_l = _regime_share(l_blocks["level"])
                alloc_share.append(float(np.mean(np.abs(share_t - share_l))))

        scale_free = [
            v for v in trans_tv + emis_rel + fact_sub + alloc_share if not np.isnan(v)
        ]
        param_err_mean = float(np.mean(scale_free)) if scale_free else float("nan")

        return cls(
            trans_tv=trans_tv,
            lag_err=lag_err,
            lag_false=lag_false,
            emis_err=emis_err,
            emis_rel=emis_rel,
            fact_sub=fact_sub,
            alloc_share=alloc_share,
            param_err_mean=param_err_mean,
            param_err_n=len(scale_free),
        )

    def to_dict(self) -> Dict[str, float]:
        d: Dict[str, float] = {}
        for k, (tv, le, lf, ee, er, fs, al) in enumerate(
            zip(
                self.trans_tv,
                self.lag_err,
                self.lag_false,
                self.emis_err,
                self.emis_rel,
                self.fact_sub,
                self.alloc_share,
            ),
            start=1,
        ):
            d[f"TransTV_k{k}"] = float(tv)
            d[f"LagErr_k{k}"] = float(le)
            d[f"LagFalse_k{k}"] = float(lf)
            d[f"EmisErr_k{k}"] = float(ee)
            d[f"EmisRel_k{k}"] = float(er)
            d[f"FactSub_k{k}"] = float(fs)
            d[f"AllocShare_k{k}"] = float(al)
        d["ParamErr_mean"] = float(self.param_err_mean)
        d["ParamErr_n"] = float(self.param_err_n)
        return d


def edge_auprc(scores: Dict[EdgeKey, float], true_edges: Set[EdgeKey]) -> float:
    """
    Average precision of a scored candidate edge set.

    Edge sets are sparse, so the area under the precision-recall curve separates
    rankings that a single thresholded F1 cannot. Candidates are sorted by score
    descending and the precision at every recall gain is summed; true edges that
    were never scored contribute nothing and still count in the denominator.
    Equally scored candidates form one indistinguishable block and are all
    credited the precision at the end of that block, which keeps the result
    independent of the order `scores` happens to enumerate.

    Parameters
    ----------
    scores     : {(src, lag, dst): score}, higher meaning more edge-like
    true_edges : the true (src, lag, dst) triples

    Returns
    -------
    AUPRC in [0, 1], or NaN when `true_edges` is empty.
    """
    if not true_edges:
        return float("nan")

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    hits = 0
    rank = 0
    total_precision = 0.0
    start = 0
    while start < len(ranked):
        stop = start
        block_hits = 0
        while stop < len(ranked) and ranked[stop][1] == ranked[start][1]:
            rank += 1
            if ranked[stop][0] in true_edges:
                hits += 1
                block_hits += 1
            stop += 1
        total_precision += block_hits * hits / rank
        start = stop
    return total_precision / len(true_edges)


# ---------------------------------------------------------------------------
# Aggregate score
# ---------------------------------------------------------------------------


def aggregate_metrics(
    predictive: Optional[PredictiveMetrics] = None,
    structure: Optional[StructureMetrics] = None,
    latent: Optional[LatentMetrics] = None,
    reconstruction: Optional[ReconstructionMetrics] = None,
    calibration: Optional[CalibrationMetrics] = None,
    reconciliation: Optional[ReconciliationMetrics] = None,
    *,
    parameter: Optional[ParameterMetrics] = None,
) -> Dict[str, float]:
    """Merge all metric dicts into a single flat dictionary."""
    result: Dict[str, float] = {}
    for m in (
        predictive,
        structure,
        latent,
        reconstruction,
        calibration,
        reconciliation,
        parameter,
    ):
        if m is not None:
            result.update(m.to_dict())
    return result


def _label_adjacency(edges: Set[EdgeKey], L: int) -> np.ndarray:
    """Label-level adjacency of a lagged edge set, lags collapsed."""
    A = np.zeros((L, L), dtype=float)
    for src, _lag, dst in edges:
        if 0 <= src < L and 0 <= dst < L:
            A[src, dst] = 1.0
    return A


def _sortability(scores: np.ndarray, A: np.ndarray, tol: float = 1e-9) -> float:
    """Fraction of directed paths along which `scores` increases."""
    L = A.shape[0]
    inc = total = 0.0
    reach = A.copy()
    for _ in range(max(L - 1, 1)):
        src, dst = np.nonzero(reach)
        for i, j in zip(src, dst):
            total += 1.0
            gap = scores[j] - scores[i]
            inc += 1.0 if gap > tol else (0.5 if abs(gap) <= tol else 0.0)
        reach = reach @ A
        if not reach.any():
            break
    return _safe_div(inc, total, default=0.5)


def varsortability(X: np.ndarray, edges: Set[EdgeKey]) -> float:
    """
    Fraction of directed paths along which the marginal variance increases.

    A generator that raises the marginal variance along its own edges lets a
    learner order the variables by variance alone and score well without
    recovering anything, which is the failure Reisach et al. (2021) name.  A
    value near 0.5 says the variance carries no ordering information; a value
    near 1 says a variance sort would reproduce the graph.

    Args:
        X: (T, L) observed series, one column per label.
        edges: true edges as (src_label, lag, dst_label) triples.

    Returns:
        Sortability in [0, 1], or 0.5 when the graph has no edge.
    """
    X = np.asarray(X, dtype=float)
    return _sortability(np.var(X, axis=0), _label_adjacency(edges, X.shape[1]))


def r2_sortability(X: np.ndarray, edges: Set[EdgeKey]) -> float:
    """
    Fraction of directed paths along which the explained variance increases.

    The variance-based check is defeated by standardizing, which is why
    Reisach et al. (2023) replace the marginal variance with the coefficient of
    determination of each label regressed on all the others.  That quantity is
    scale free, so standardizing does not move it.

    Args:
        X: (T, L) observed series, one column per label.
        edges: true edges as (src_label, lag, dst_label) triples.

    Returns:
        Sortability in [0, 1], or 0.5 when the graph has no edge.
    """
    X = np.asarray(X, dtype=float)
    L = X.shape[1]
    Xc = X - X.mean(axis=0, keepdims=True)
    r2 = np.zeros(L)
    for j in range(L):
        others = [i for i in range(L) if i != j]
        if not others:
            continue
        target, design = Xc[:, j], Xc[:, others]
        coef, *_ = np.linalg.lstsq(design, target, rcond=None)
        resid = target - design @ coef
        r2[j] = 1.0 - _safe_div(
            float(resid @ resid), float(target @ target), default=0.0
        )
    return _sortability(r2, _label_adjacency(edges, L))
