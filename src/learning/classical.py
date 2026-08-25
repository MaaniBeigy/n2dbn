"""
learning/classical.py
=====================
Published structure-learning algorithms, applied to the lagged graph of one
scale, so that the S-step of this model can be scored against them.

Why a separate module
---------------------
The S-step searches parent sets against the expected complete-data likelihood
of the N2DBN, which knows about zero inflation, overdispersion, the latent
regime, the latent factor and the block normalizer of the allocation.  Every
algorithm below instead searches against a generic criterion computed from the
observed counts alone.  Running both on the same data and scoring the resulting
graphs with the same metrics separates what the model contributes from what any
structure learner would find.

The search space
----------------
All observed-to-observed edges of the N2DBN are lagged and within-scale, so the
graph at scale `k` is a bipartite map from the lagged copies of the counts to
the current counts,

    X_{t-d}^{(k, l')}  gives a parent of  X_t^{(k, l)},   d >= 1.

`build_lagged_frame` expands the count array into exactly that: one column per
current label and one per (label, lag) pair.  Every lagged-only graph is
acyclic, and the score of each target decomposes from the scores of the others,
which is what makes an exhaustive per-target search feasible and makes the
acyclicity machinery of a general DAG learner unnecessary here.

Algorithms
----------
Score-based, from pgmpy, searching inside the bipartite space directly:

    `hc-bic`, `hc-bdeu`, `hc-k2`, `hc-aic`
        Hill climbing over the whitelisted edges with the named score.
    `exhaustive`
        The maximizer of the BIC local score over every parent set up to the
        in-degree bound, per target.  Exact, and feasible because the score
        decomposes over targets.

Constraint-based and hybrid, from pgmpy, searching the full node set and then
projected onto the lagged edges:

    `pc`        Peter-Clark with a chi-square conditional independence test.
    `ges`       Greedy equivalence search over the equivalence class.
    `mmhc`      Max-min hill climbing, constraint-based skeleton then a
                score-based orientation.
    `tree`      Chow-Liu tree.

Continuous-optimization and time-series methods, implemented here:

    `dynotears` The autoregressive block of DYNOTEARS (Pamfil et al., 2020).
                With same-slice edges excluded by the model, its acyclicity
                penalty is vacuous and the problem is the l1-penalized vector
                autoregression that remains, solved by FISTA.
    `granger`   Vector autoregression with parent sets selected by BIC, which
                is Granger causality read off a fitted lag polynomial.
    `empty`     No edges.  The control that says what the metrics read when a
                method reports nothing.

`learn_structure` dispatches on the name and returns parent sets in the form
`ScaleParams.parent_sets` uses, so the result drops straight into a graph.
`learn_pooled_structure` runs the same search over a cohort: each record is
lagged inside itself and the rows are stacked, so one graph is fitted to many
records and no row reads across a record boundary.

Reproducibility
---------------
The four hill-climbing entries and `pc` receive the bipartite space as a
whitelist, so their search never leaves it.  `ges`, `mmhc` and `tree` accept no
whitelist, so they search the full node set and their output is projected onto
the lagged edges afterwards.  For those three the surviving edges depend on how
an undirected skeleton is oriented, and pgmpy settles that by iterating over
sets of variable names.  Python randomizes string hashing per process, so the
same call on the same data can return different edges in two runs.  Fix
`PYTHONHASHSEED` before starting an experiment that includes them; the container
images set it to 0.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

LagKey = Tuple[int, int]  # (src_label_idx, lag)

# Names accepted by `learn_structure`, grouped by what they need.
PGMPY_SCORE_ALGORITHMS = ("hc-bic", "hc-bdeu", "hc-k2", "hc-aic")
PGMPY_GLOBAL_ALGORITHMS = ("pc", "ges", "mmhc", "tree")
PGMPY_ALGORITHMS = PGMPY_SCORE_ALGORITHMS + PGMPY_GLOBAL_ALGORITHMS + ("exhaustive",)
NATIVE_ALGORITHMS = ("dynotears", "granger", "empty")
ALGORITHMS = PGMPY_ALGORITHMS + NATIVE_ALGORITHMS

_PGMPY_SCORE_KEY = {
    "hc-bic": "bic-d",
    "hc-bdeu": "bdeu",
    "hc-k2": "k2",
    "hc-aic": "aic-d",
}


def available_algorithms() -> List[str]:
    """Algorithm names usable in this environment, in registry order."""
    if _pgmpy_available():
        return list(ALGORITHMS)
    return list(NATIVE_ALGORITHMS)


def _pgmpy_available() -> bool:
    """True when pgmpy imports."""
    try:
        import pgmpy  # noqa: F401
    except ImportError:
        return False
    return True


@contextmanager
def _quiet_pgmpy():
    """Silence the progress bars of algorithms that take no `show_progress`."""
    import pgmpy

    previous = pgmpy.config.get_show_progress()
    pgmpy.config.set_show_progress(False)
    try:
        yield
    finally:
        pgmpy.config.set_show_progress(previous)


# ---------------------------------------------------------------------------
# The lagged design
# ---------------------------------------------------------------------------


def encode_counts(
    X: np.ndarray,
    encoding: str = "binary",
    n_bins: int = 4,
) -> np.ndarray:
    """
    Map counts to the small integer alphabet a discrete learner needs.

    Args:
        X: `(T, L)` count array.
        encoding: `binary` marks presence, `clip` keeps counts up to
            `n_bins - 1` and collapses the tail, `quantile` cuts each label at
            its own empirical quantiles.
        n_bins: alphabet size for `clip` and `quantile`.

    Returns:
        `(T, L)` array of integer codes.
    """
    if encoding == "binary":
        return (np.asarray(X) > 0).astype(np.int64)
    if encoding == "clip":
        return np.clip(np.asarray(X), 0, max(1, n_bins - 1)).astype(np.int64)
    if encoding != "quantile":
        raise ValueError(
            f"unknown encoding {encoding!r}, expected binary, clip or quantile"
        )

    X = np.asarray(X)
    out = np.zeros_like(X, dtype=np.int64)
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    for ell in range(X.shape[1]):
        col = X[:, ell]
        edges = np.unique(np.quantile(col, qs))
        # A count column ties heavily and every cut point is itself one of the
        # tied values, so the mass sitting exactly on a cut belongs below it.  A
        # label absent on three days in four has all its cuts at zero, and it
        # then carries code 0 on its absent days and code 1 on the rest.
        out[:, ell] = np.searchsorted(edges, col, side="left")
    return out


@dataclass
class LaggedFrame:
    """
    Counts of one scale expanded into current and lagged columns.

    Both name maps are keyed by label index rather than positional, because
    `_drop_constant` can remove a column from the middle and a positional list
    would then report the parents of one label under the index of another.

    Attributes:
        frame: the design, one row per epoch that has a full lag history.
        current: `current[l]` is the column name of label `l` at time `t`.
        lagged: `lagged[(l, d)]` is the column name of label `l` at `t - d`.
    """

    frame: "object"  # pandas.DataFrame; typed loosely to keep the import lazy
    current: Dict[int, str]
    lagged: Dict[LagKey, str]

    @property
    def allowed_edges(self) -> List[Tuple[str, str]]:
        """Every lagged-to-current pair, the whole search space."""
        return [
            (src, dst) for src in self.lagged.values() for dst in self.current.values()
        ]

    def parent_sets(
        self, edges: Sequence[Tuple[str, str]], n_labels: int
    ) -> Dict[int, Set[LagKey]]:
        """Parent sets keyed by target label, ignoring edges outside the space."""
        by_name = {name: key for key, name in self.lagged.items()}
        target = {name: ell for ell, name in self.current.items()}
        out: Dict[int, Set[LagKey]] = {ell: set() for ell in range(n_labels)}
        for src, dst in edges:
            if src in by_name and dst in target:
                out[target[dst]].add(by_name[src])
        return out


def build_lagged_frame(
    X: np.ndarray,
    max_lag: int,
    encoding: str = "binary",
    n_bins: int = 4,
) -> LaggedFrame:
    """
    Expand a count array into the bipartite design a lagged learner searches.

    The first `max_lag` epochs carry an incomplete history and are dropped, so
    the design holds `T - max_lag` rows.  Columns are categorical, which is what
    the discrete scores and conditional independence tests of pgmpy require.
    """
    return build_pooled_frame([X], max_lag, encoding=encoding, n_bins=n_bins)


def build_pooled_frame(
    blocks: Sequence[np.ndarray],
    max_lag: int,
    encoding: str = "binary",
    n_bins: int = 4,
) -> LaggedFrame:
    """
    Expand several records into one design, each carrying its own lag history.

    A lag reads inside one record, so the records are expanded apart and their
    rows stacked, and no row of the design reads a feature from the record
    before it.  The encoding is fixed once on the stacked counts, so a code
    means the same thing in every record and a threshold read off one record's
    quantiles does not travel to the next.

    Raises:
        ValueError: if any record is shorter than the lag it has to supply.
    """
    import pandas as pd

    lengths = [int(np.asarray(b).shape[0]) for b in blocks]
    L = int(np.asarray(blocks[0]).shape[1])
    for T in lengths:
        if T <= max_lag:
            raise ValueError(
                f"{T} epochs cannot support a lag of {max_lag}: the design "
                "would be empty"
            )
    codes = encode_counts(
        np.vstack([np.asarray(b) for b in blocks]), encoding=encoding, n_bins=n_bins
    )

    current: Dict[int, str] = {ell: f"x{ell}_t" for ell in range(L)}
    lagged: Dict[LagKey, str] = {
        (ell, d): f"x{ell}_lag{d}" for ell in range(L) for d in range(1, max_lag + 1)
    }
    parts: Dict[str, List[np.ndarray]] = {
        name: [] for name in list(current.values()) + list(lagged.values())
    }
    start = 0
    for T in lengths:
        block = codes[start : start + T]
        start += T
        for ell in range(L):
            parts[current[ell]].append(block[max_lag:, ell])
            for d in range(1, max_lag + 1):
                parts[lagged[(ell, d)]].append(block[max_lag - d : T - d, ell])

    frame = pd.DataFrame(
        {name: np.concatenate(cols) for name, cols in parts.items()}
    ).astype("category")
    return LaggedFrame(frame=frame, current=current, lagged=lagged)


def _drop_constant(lf: LaggedFrame) -> LaggedFrame:
    """
    Remove columns that never change.

    A constant column carries no information, and a conditional independence
    test conditioned on one raises instead of returning independence.
    """
    keep = [c for c in lf.frame.columns if lf.frame[c].nunique() > 1]
    if len(keep) == len(lf.frame.columns):
        return lf
    kept = set(keep)
    return LaggedFrame(
        frame=lf.frame[keep],
        current={ell: c for ell, c in lf.current.items() if c in kept},
        lagged={k: v for k, v in lf.lagged.items() if v in kept},
    )


# ---------------------------------------------------------------------------
# Trimming to the in-degree budget
# ---------------------------------------------------------------------------


def _greedy_by_score(
    score_fn,
    target: str,
    candidates: Sequence[str],
    max_indegree: int,
) -> Set[str]:
    """Parents added one at a time while the local score improves."""
    chosen: Set[str] = set()
    best = score_fn(target, [])
    remaining = list(candidates)
    while remaining and len(chosen) < max_indegree:
        gains = [(score_fn(target, sorted(chosen | {c})), c) for c in remaining]
        gains.sort(key=lambda pair: -pair[0])
        top_score, top_edge = gains[0]
        if top_score <= best:
            break
        best = top_score
        chosen.add(top_edge)
        remaining.remove(top_edge)
    return chosen


def _trim_to_indegree(
    lf: LaggedFrame,
    parents: Dict[int, Set[LagKey]],
    max_indegree: int,
) -> Dict[int, Set[LagKey]]:
    """
    Cut over-full parent sets down to the budget by BIC local score.

    Algorithms that search the whole node set have no per-target in-degree
    control, so their output is trimmed here rather than left to compete against
    a bounded search on unequal terms.
    """
    if all(len(ps) <= max_indegree for ps in parents.values()):
        return parents
    from pgmpy.estimators import BIC

    score = BIC(lf.frame)
    out: Dict[int, Set[LagKey]] = {}
    for ell, keys in parents.items():
        if len(keys) <= max_indegree:
            out[ell] = set(keys)
            continue
        names = [lf.lagged[k] for k in keys if k in lf.lagged]
        chosen = _greedy_by_score(
            score.local_score, lf.current[ell], names, max_indegree
        )
        by_name = {lf.lagged[k]: k for k in keys if k in lf.lagged}
        out[ell] = {by_name[n] for n in chosen}
    return out


# ---------------------------------------------------------------------------
# pgmpy-backed learners
# ---------------------------------------------------------------------------


def _learn_pgmpy(
    lf: LaggedFrame,
    algorithm: str,
    n_labels: int,
    max_indegree: int,
    significance_level: float,
) -> Dict[int, Set[LagKey]]:
    """Run one pgmpy algorithm and read its lagged edges back."""
    from pgmpy.estimators import (
        GES,
        PC,
        ExpertKnowledge,
        HillClimbSearch,
        MmhcEstimator,
        TreeSearch,
    )

    frame = lf.frame
    space = lf.allowed_edges

    if algorithm in _PGMPY_SCORE_KEY:
        dag = HillClimbSearch(frame).estimate(
            scoring_method=_PGMPY_SCORE_KEY[algorithm],
            expert_knowledge=ExpertKnowledge(search_space=space),
            max_indegree=max_indegree,
            show_progress=False,
        )
        edges = list(dag.edges())
    elif algorithm == "pc":
        # `enforce_expert_knowledge` drops the whole skeleton when the search
        # space is bipartite, so the space guides the search and the projection
        # below does the restricting.
        dag = PC(frame).estimate(
            variant="stable",
            ci_test="chi_square",
            return_type="dag",
            expert_knowledge=ExpertKnowledge(search_space=space),
            enforce_expert_knowledge=False,
            significance_level=significance_level,
            show_progress=False,
            n_jobs=1,
        )
        edges = list(dag.edges())
    elif algorithm == "ges":
        edges = list(GES(frame).estimate(scoring_method="bic-d").edges())
    elif algorithm == "mmhc":
        edges = list(
            MmhcEstimator(frame).estimate(significance_level=significance_level).edges()
        )
    elif algorithm == "tree":
        edges = list(
            TreeSearch(frame, n_jobs=1)
            .estimate(estimator_type="chow-liu", show_progress=False)
            .edges()
        )
    else:  # pragma: no cover - guarded by learn_structure
        raise ValueError(f"unknown pgmpy algorithm {algorithm!r}")

    parents = lf.parent_sets(edges, n_labels)
    return _trim_to_indegree(lf, parents, max_indegree)


def _learn_exhaustive(
    lf: LaggedFrame,
    n_labels: int,
    max_indegree: int,
) -> Dict[int, Set[LagKey]]:
    """
    The BIC-optimal parent set of every target, searched over every subset.

    The score of a lagged-only graph is the sum of per-target local scores, so
    maximizing each target separately maximizes the whole.  The subset count is
    `sum_{j<=d} C(L*max_lag, j)`, which is why the in-degree bound is what keeps
    this tractable rather than an approximation.
    """
    from itertools import combinations

    from pgmpy.estimators import BIC

    score = BIC(lf.frame)
    names = {name: key for key, name in lf.lagged.items()}
    out: Dict[int, Set[LagKey]] = {ell: set() for ell in range(n_labels)}
    for ell, target in lf.current.items():
        pool = [n for n in names if n != target]
        best_set: Tuple[str, ...] = ()
        best = score.local_score(target, [])
        for size in range(1, max_indegree + 1):
            for subset in combinations(pool, size):
                value = score.local_score(target, list(subset))
                if value > best:
                    best, best_set = value, subset
        out[ell] = {names[n] for n in best_set}
    return out


# ---------------------------------------------------------------------------
# Continuous-optimization and time-series learners
# ---------------------------------------------------------------------------


def _design(X: np.ndarray, max_lag: int) -> Tuple[np.ndarray, np.ndarray, List[LagKey]]:
    """
    Standardized lag design, response and the key of each design column.

    Counts enter as `log(1 + x)`, matching the lag features the model itself
    uses, and every column is centered and scaled so that one coefficient
    threshold means the same thing for a rare label and a common one.
    """
    return _pooled_design([X], max_lag)


def _pooled_design(
    blocks: Sequence[np.ndarray], max_lag: int
) -> Tuple[np.ndarray, np.ndarray, List[LagKey]]:
    """
    `_design` over several records, lagged apart and stacked into one fit.

    The centering and the scaling are read off the stack, so one coefficient
    threshold means the same thing in every record.
    """
    Zs = [np.log1p(np.asarray(X, dtype=float)) for X in blocks]
    L = Zs[0].shape[1]
    keys = [(ell, d) for ell in range(L) for d in range(1, max_lag + 1)]
    W = np.vstack(
        [
            np.column_stack([Z[max_lag - d : Z.shape[0] - d, ell] for ell, d in keys])
            for Z in Zs
        ]
    )
    Y = np.vstack([Z[max_lag:, :] for Z in Zs])

    def _standardize(A):
        mean = A.mean(axis=0, keepdims=True)
        std = A.std(axis=0, keepdims=True)
        std = np.where(std < 1e-12, 1.0, std)
        return (A - mean) / std

    return _standardize(W), _standardize(Y), keys


def _fista(
    W: np.ndarray,
    Y: np.ndarray,
    lam: float,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    """Minimize `0.5/T ||Y - W A||^2 + lam ||A||_1` by proximal gradient."""
    T, p = W.shape
    gram = W.T @ W / T
    corr = W.T @ Y / T
    step = 1.0 / max(float(np.linalg.eigvalsh(gram).max()), 1e-12)

    A = np.zeros((p, Y.shape[1]))
    Z = A.copy()
    t_k = 1.0
    for _ in range(max_iter):
        grad = gram @ Z - corr
        cand = Z - step * grad
        A_new = np.sign(cand) * np.maximum(np.abs(cand) - step * lam, 0.0)
        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k))
        Z = A_new + ((t_k - 1.0) / t_next) * (A_new - A)
        delta = float(np.abs(A_new - A).max())
        A, t_k = A_new, t_next
        if delta < tol:
            break
    return A


def _fista_torch(
    W: np.ndarray,
    Y: np.ndarray,
    lam: float,
    max_iter: int,
    tol: float,
    device: str,
) -> np.ndarray:
    """`_fista` on a torch device, for designs large enough to pay the transfer."""
    import torch

    dev = torch.device(device)
    Wt = torch.as_tensor(W, dtype=torch.float64, device=dev)
    Yt = torch.as_tensor(Y, dtype=torch.float64, device=dev)
    T = Wt.shape[0]
    gram = Wt.T @ Wt / T
    corr = Wt.T @ Yt / T
    step = 1.0 / float(torch.linalg.eigvalsh(gram).max().clamp_min(1e-12))

    A = torch.zeros(Wt.shape[1], Yt.shape[1], dtype=torch.float64, device=dev)
    Z = A.clone()
    t_k = 1.0
    for _ in range(max_iter):
        cand = Z - step * (gram @ Z - corr)
        A_new = torch.sign(cand) * torch.clamp(cand.abs() - step * lam, min=0.0)
        t_next = 0.5 * (1.0 + float(np.sqrt(1.0 + 4.0 * t_k * t_k)))
        Z = A_new + ((t_k - 1.0) / t_next) * (A_new - A)
        delta = float((A_new - A).abs().max())
        A, t_k = A_new, t_next
        if delta < tol:
            break
    return A.cpu().numpy()


def _learn_dynotears(
    blocks: Sequence[np.ndarray],
    max_lag: int,
    max_indegree: int,
    lam: float = 0.05,
    w_threshold: float = 0.05,
    max_iter: int = 500,
    tol: float = 1e-8,
    device: str = "cpu",
) -> Dict[int, Set[LagKey]]:
    """
    Lagged coefficients from an l1-penalized vector autoregression.

    This is the `A` block of DYNOTEARS.  The `W` block, which holds the
    same-slice edges, is fixed at zero because the model routes same-slice
    co-movement through the latent variables, and with `W = 0` the acyclicity
    constraint of DYNOTEARS holds for every `A`.  What is left is a convex lasso,
    so the solution is the global optimum of that objective rather than a
    stationary point.
    """
    W, Y, keys = _pooled_design(blocks, max_lag)
    if _use_torch(device):
        A = _fista_torch(W, Y, lam, max_iter, tol, device)
    else:
        A = _fista(W, Y, lam, max_iter, tol)

    out: Dict[int, Set[LagKey]] = {}
    for ell in range(Y.shape[1]):
        weights = np.abs(A[:, ell])
        order = np.argsort(-weights)[:max_indegree]
        out[ell] = {keys[j] for j in order if weights[j] > w_threshold}
    return out


def _use_torch(device: str) -> bool:
    """True when the solver runs on torch rather than on numpy."""
    return device != "cpu" and _torch_device_available(device)


def _torch_device_available(device: str) -> bool:
    """True when torch is installed and the named device is usable."""
    try:
        import torch
    except ImportError:
        return False
    if device.startswith("cuda"):
        return bool(torch.cuda.is_available())
    return True


def _learn_granger(
    blocks: Sequence[np.ndarray],
    max_lag: int,
    max_indegree: int,
) -> Dict[int, Set[LagKey]]:
    """
    Parent sets of a vector autoregression, selected by BIC.

    A lagged label enters the model of a target when adding it lowers
    `T log(RSS/T) + p log T`, which is the Granger test read as model selection
    rather than as a sequence of F tests, so it needs no multiplicity correction.
    """
    W, Y, keys = _pooled_design(blocks, max_lag)
    T, p = W.shape
    out: Dict[int, Set[LagKey]] = {}

    for ell in range(Y.shape[1]):
        y = Y[:, ell]
        chosen: List[int] = []
        best = T * np.log(max(float(y @ y) / T, 1e-300))
        remaining = list(range(p))
        while remaining and len(chosen) < max_indegree:
            scores = []
            for j in remaining:
                cols = chosen + [j]
                resid = _residual_sum_of_squares(W[:, cols], y)
                bic = T * np.log(max(resid / T, 1e-300)) + len(cols) * np.log(T)
                scores.append((bic, j))
            scores.sort()
            bic, j = scores[0]
            if bic >= best:
                break
            best = bic
            chosen.append(j)
            remaining.remove(j)
        out[ell] = {keys[j] for j in chosen}
    return out


def _residual_sum_of_squares(A: np.ndarray, y: np.ndarray) -> float:
    """Least-squares residual sum of squares, 0 when the fit is exact."""
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    return float(resid @ resid)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def learn_structure(
    algorithm: str,
    X: np.ndarray,
    max_lag: int,
    max_indegree: int,
    encoding: str = "binary",
    n_bins: int = 4,
    significance_level: float = 0.01,
    device: str = "cpu",
    dynotears_lam: float = 0.05,
    dynotears_threshold: float = 0.05,
) -> Dict[int, Set[LagKey]]:
    """
    Lagged parent sets of one scale, learned by the named algorithm.

    Args:
        algorithm: one of `ALGORITHMS`.
        X: `(T, L)` counts of the scale.
        max_lag: longest lag the search considers.
        max_indegree: largest parent set the search may return.
        encoding: how counts become the alphabet of a discrete learner.
        n_bins: alphabet size for the `clip` and `quantile` encodings.
        significance_level: level of the conditional independence tests.
        device: `cpu`, or a torch device name for the DYNOTEARS solver.
        dynotears_lam: l1 weight of the DYNOTEARS objective.
        dynotears_threshold: coefficient magnitude an edge has to reach.

    Returns:
        `{target_label: {(source_label, lag), ...}}`, one entry per label.

    Raises:
        ValueError: the name is unknown, or names a pgmpy algorithm while pgmpy
            is not installed.
    """
    return learn_pooled_structure(
        algorithm,
        [X],
        max_lag=max_lag,
        max_indegree=max_indegree,
        encoding=encoding,
        n_bins=n_bins,
        significance_level=significance_level,
        device=device,
        dynotears_lam=dynotears_lam,
        dynotears_threshold=dynotears_threshold,
    )


def learn_pooled_structure(
    algorithm: str,
    blocks: Sequence[np.ndarray],
    max_lag: int,
    max_indegree: int,
    encoding: str = "binary",
    n_bins: int = 4,
    significance_level: float = 0.01,
    device: str = "cpu",
    dynotears_lam: float = 0.05,
    dynotears_threshold: float = 0.05,
) -> Dict[int, Set[LagKey]]:
    """
    One lagged graph learned from several records pooled as independent rows.

    A cohort holds one graph and many records of it, so every record supplies
    rows to one design and every record supplies its own lag history.  Stacking
    the records first and lagging afterwards would let the last epochs of one
    record predict the first epochs of the next, which is a relation between two
    people and not the relation being searched for.

    Args:
        algorithm: one of `ALGORITHMS`.
        blocks: one `(T_i, L)` count array per record, sharing a label axis.
        max_lag: longest lag the search considers.
        max_indegree: largest parent set the search may return.
        encoding: how counts become the alphabet of a discrete learner.
        n_bins: alphabet size for the `clip` and `quantile` encodings.
        significance_level: level of the conditional independence tests.
        device: `cpu`, or a torch device name for the DYNOTEARS solver.
        dynotears_lam: l1 weight of the DYNOTEARS objective.
        dynotears_threshold: coefficient magnitude an edge has to reach.

    Returns:
        `{target_label: {(source_label, lag), ...}}`, one entry per label.

    Raises:
        ValueError: the name is unknown, no record was handed over, or the name
            is a pgmpy algorithm while pgmpy is not installed.
    """
    if algorithm not in ALGORITHMS:
        raise ValueError(
            f"unknown algorithm {algorithm!r}, expected one of {list(ALGORITHMS)}"
        )
    if not blocks:
        raise ValueError("a pooled search needs at least one record to search over")
    L = int(np.asarray(blocks[0]).shape[1])
    max_lag = max(1, int(max_lag))
    max_indegree = max(0, int(max_indegree))

    if algorithm == "empty" or max_indegree == 0:
        return {ell: set() for ell in range(L)}
    if algorithm == "dynotears":
        return _learn_dynotears(
            blocks,
            max_lag,
            max_indegree,
            lam=dynotears_lam,
            w_threshold=dynotears_threshold,
            device=device,
        )
    if algorithm == "granger":
        return _learn_granger(blocks, max_lag, max_indegree)

    if not _pgmpy_available():
        raise ValueError(f"algorithm {algorithm!r} needs pgmpy, which is not installed")

    lf = _drop_constant(
        build_pooled_frame(blocks, max_lag, encoding=encoding, n_bins=n_bins)
    )
    if not lf.current or not lf.lagged:
        return {ell: set() for ell in range(L)}

    with warnings.catch_warnings(), _quiet_pgmpy():
        warnings.simplefilter("ignore")
        if algorithm == "exhaustive":
            found = _learn_exhaustive(lf, L, max_indegree)
        else:
            found = _learn_pgmpy(lf, algorithm, L, max_indegree, significance_level)

    # A label whose column was constant keeps an empty parent set.
    return {ell: found.get(ell, set()) for ell in range(L)}


def learn_multiscale_structure(
    algorithm: str,
    data,
    max_lag: Dict[int, int],
    max_indegree: Dict[int, int],
    **options,
) -> Dict[int, Dict[int, Set[LagKey]]]:
    """
    Run `learn_structure` at every scale of a `MultiScaleData`.

    Returns:
        `{scale: {target_label: {(source_label, lag), ...}}}`, scales 1-indexed.
    """
    out: Dict[int, Dict[int, Set[LagKey]]] = {}
    for k in range(1, data.K + 1):
        out[k] = learn_structure(
            algorithm,
            data.get_X(k),
            max_lag=int(max_lag.get(k, 1)),
            max_indegree=int(max_indegree.get(k, 0)),
            **options,
        )
    return out
