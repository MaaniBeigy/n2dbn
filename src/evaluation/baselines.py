"""
evaluation/baselines.py
=======================
Baseline models for the N²DBN evaluation protocol.

Baselines
---------
SingleScaleDBN    : fit a DBN at each scale independently (no hierarchy).
TopDownNoFactor   : top-down SEM with r=0 (no latent factor).
TopDownNoRegime   : top-down SEM with M=1 (no regime switching).
EmptyGraph        : fixed empty graph (no lagged edges anywhere).
OracleGraph       : fixed ground-truth graph (upper bound for structure).
OracleParams      : no fitting at all, the identifiability ceiling.
ClassicalStructure: graph from a published structure-learning algorithm, then
                    the parameters of this model fitted on that graph.

Each baseline exposes a `fit` / `predict` interface matching TopDownSEM
so they can be dropped into the evaluation protocol.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.graph import MultiScaleDBNGraph
from ..core.scales import ScaleHierarchy
from ..core.variables import MultiScaleData
from ..inference.posterior import PosteriorSummaries
from ..learning.estep import EStep
from ..learning.evidence import evidence_chain
from ..learning.structural_em import TopDownSEM, TrainingHistory
from ..model.params import ScaleParams

# ---------------------------------------------------------------------------
# Helper: build a zero-edge graph copy
# ---------------------------------------------------------------------------


def _empty_graph(graph: MultiScaleDBNGraph) -> MultiScaleDBNGraph:
    """Return a copy of `graph` with all edges removed."""
    g = deepcopy(graph)
    for k in range(1, g.K + 1):
        for ell in range(g.L):
            for src_ell, lag in list(g.get_parents(k, ell)):
                g.remove_edge(k, src_ell, lag, ell)
    return g


def _clear_parent_sets(params_list: List[ScaleParams]) -> List[ScaleParams]:
    """Return a copy of `params_list` with all parent_sets and lag coefficients cleared.

    This ensures baselines that should have no lagged structure do not
    inadvertently use ground-truth lag coefficients from the initialisation.
    """
    out = []
    for p in params_list:
        p = deepcopy(p)
        L = p.L
        p.parent_sets = {ell: set() for ell in range(L)}
        if p.is_coarsest:
            p.coarse_count.betaLag = {}
            p.coarse_dur.alphaLag = {}
        else:
            p.fine_count.betaLag = {}
            p.fine_dur.alphaLag = {}
        out.append(p)
    return out


def _seed_lag_coefficients(params: ScaleParams) -> None:
    """
    Give every parent named in `parent_sets` a zero coefficient to start from.

    The parameter step updates the coefficient of a lagged parent only when the
    key already exists in `betaLag` / `alphaLag`.  A graph installed from
    outside the S-step therefore has no effect on any likelihood until its keys
    are present, which would score a learned graph as if it were empty.
    """
    keys = set()
    for parents in params.parent_sets.values():
        keys.update(parents)
    if params.is_coarsest:
        blocks = (params.coarse_count.betaLag, params.coarse_dur.alphaLag)
    else:
        blocks = (params.fine_count.betaLag, params.fine_dur.alphaLag)
    for block in blocks:
        for key in keys:
            block.setdefault(key, np.zeros(params.L))


def _scoring_settings(sem_kwargs: dict) -> Tuple[EStep, bool]:
    """The E-step and the coupling a `TopDownSEM` built from `sem_kwargs` fits with."""
    # Every wrapper stores the pair under the two attribute names TopDownSEM
    # already carries, so whatever scores a fitted model reads both settings
    # the same way whichever wrapper produced it.
    return (
        EStep(**(sem_kwargs.get("estep_kwargs") or {})),
        bool(sem_kwargs.get("couple_scales", True)),
    )


# ---------------------------------------------------------------------------
# SingleScaleDBN: independent per-scale SEM, no cross-scale coupling
# ---------------------------------------------------------------------------


class SingleScaleDBN:
    """
    Fit an independent DBN at each scale, ignoring the cross-scale coupling.

    No parent-scale summaries reach the E-step or the M-step, so the regime
    chain of a scale is driven only by that scale's own data and context.  The
    allocation model still ties a finer scale to its parent totals, since that
    is what makes the counts reconcile; this baseline isolates the value of
    passing the parent *posterior* downward, not of the hierarchy itself.
    """

    def __init__(self, **sem_kwargs) -> None:
        kw = dict(sem_kwargs)
        kw["couple_scales"] = False
        self._sem_kwargs = kw
        self.estep, self.couple_scales = _scoring_settings(kw)

    def fit(
        self,
        data: MultiScaleData,
        params_list: List[ScaleParams],
        graph: MultiScaleDBNGraph,
    ) -> Tuple[List[ScaleParams], MultiScaleDBNGraph, TrainingHistory]:
        sem = TopDownSEM(**self._sem_kwargs)
        return sem.fit(data, params_list, graph)

    def predict(
        self,
        data: MultiScaleData,
        params_list: List[ScaleParams],
        graph: MultiScaleDBNGraph,
    ) -> List[PosteriorSummaries]:
        return evidence_chain(
            data,
            params_list,
            graph,
            estep=self.estep,
            couple_scales=self.couple_scales,
        ).summaries


# ---------------------------------------------------------------------------
# PerScaleDBN: one model per granularity, with nothing tying them together
# ---------------------------------------------------------------------------


class PerScaleDBN:
    """
    Fit every scale as its own model, with no allocation to its parent total.

    `SingleScaleDBN` withholds the parent posterior and keeps the allocation, so
    its scales still agree by construction and it measures what the downward
    message buys.  This one drops the allocation as well, which is what fitting
    one granularity at a time actually means: each scale carries its own
    emission, its own intercept and its own lag block, and nothing constrains
    the sum of a block of fine predictions to the coarse prediction over it.

    It is the arm the coherence gap of `evaluation.coherence` exists to measure,
    and the only arm on which that gap is free to be positive.
    """

    def __init__(self, **sem_kwargs) -> None:
        kw = dict(sem_kwargs)
        kw["couple_scales"] = False
        self._sem_kwargs = kw
        self.estep, self.couple_scales = _scoring_settings(kw)

    @staticmethod
    def initial_params(params_list: List[ScaleParams]) -> List[ScaleParams]:
        """The scales this baseline needs, which every one of them owning its emission.

        Raises:
            ValueError: a scale carries neither emission, so there is nothing to
                convert and the caller built the list some other way.
        """
        for params in params_list:
            if params.coarse_count is None and params.fine_count is None:
                raise ValueError(
                    f"scale {params.k} carries no emission to fit; build the list "
                    "with initialize_from_data(every_scale_coarse=True)"
                )
        return params_list

    def fit(
        self,
        data: MultiScaleData,
        params_list: List[ScaleParams],
        graph: MultiScaleDBNGraph,
    ) -> Tuple[List[ScaleParams], MultiScaleDBNGraph, TrainingHistory]:
        sem = TopDownSEM(**self._sem_kwargs)
        return sem.fit(data, self.initial_params(params_list), graph)

    def predict(
        self,
        data: MultiScaleData,
        params_list: List[ScaleParams],
        graph: MultiScaleDBNGraph,
    ) -> List[PosteriorSummaries]:
        return evidence_chain(
            data,
            params_list,
            graph,
            estep=self.estep,
            couple_scales=self.couple_scales,
        ).summaries


# ---------------------------------------------------------------------------
# TopDownNoFactor: top-down SEM with r=0 forced
# ---------------------------------------------------------------------------


class TopDownNoFactor:
    """
    Top-down SEM without latent factor (forces r=0 at every scale).

    Parameters of the factor model (F, G, Q, etc.) are ignored.
    """

    def __init__(self, **sem_kwargs) -> None:
        self._sem_kwargs = sem_kwargs
        self.estep, self.couple_scales = _scoring_settings(sem_kwargs)

    def fit(
        self,
        data: MultiScaleData,
        params_list: List[ScaleParams],
        graph: MultiScaleDBNGraph,
    ) -> Tuple[List[ScaleParams], MultiScaleDBNGraph, TrainingHistory]:
        # Zero out r in all params
        params_no_r = [_zero_r(p) for p in params_list]
        sem = TopDownSEM(**self._sem_kwargs)
        return sem.fit(data, params_no_r, graph)

    def predict(
        self,
        data: MultiScaleData,
        params_list: List[ScaleParams],
        graph: MultiScaleDBNGraph,
    ) -> List[PosteriorSummaries]:
        params_no_r = [_zero_r(p) for p in params_list]
        return evidence_chain(
            data,
            params_no_r,
            graph,
            estep=self.estep,
            couple_scales=self.couple_scales,
        ).summaries


# ---------------------------------------------------------------------------
# TopDownNoRegime: top-down SEM with M=1 forced
# ---------------------------------------------------------------------------


class TopDownNoRegime:
    """
    Top-down SEM with only one regime (M=1).  Regime transitions degenerate
    to a fixed prior.

    The initial params_list is reduced to M=1 by marginalising regime params.
    """

    def __init__(self, **sem_kwargs) -> None:
        self._sem_kwargs = sem_kwargs
        self.estep, self.couple_scales = _scoring_settings(sem_kwargs)

    def fit(
        self,
        data: MultiScaleData,
        params_list: List[ScaleParams],
        graph: MultiScaleDBNGraph,
    ) -> Tuple[List[ScaleParams], MultiScaleDBNGraph, TrainingHistory]:
        params_m1 = [_collapse_regimes(p) for p in params_list]
        sem = TopDownSEM(**self._sem_kwargs)
        return sem.fit(data, params_m1, graph)

    def predict(
        self,
        data: MultiScaleData,
        params_list: List[ScaleParams],
        graph: MultiScaleDBNGraph,
    ) -> List[PosteriorSummaries]:
        params_m1 = [_collapse_regimes(p) for p in params_list]
        return evidence_chain(
            data,
            params_m1,
            graph,
            estep=self.estep,
            couple_scales=self.couple_scales,
        ).summaries


# ---------------------------------------------------------------------------
# EmptyGraph: no lagged edges, but full latent model
# ---------------------------------------------------------------------------


class EmptyGraph:
    """
    Top-down SEM with a fixed empty graph (no lagged edges).

    Structure search (S-step) is disabled.  This isolates the contribution
    of graph structure from the latent model.
    """

    def __init__(self, **sem_kwargs) -> None:
        kw = dict(sem_kwargs)
        kw["do_structure"] = False
        self._sem_kwargs = kw
        self.estep, self.couple_scales = _scoring_settings(kw)

    def fit(
        self,
        data: MultiScaleData,
        params_list: List[ScaleParams],
        graph: MultiScaleDBNGraph,
    ) -> Tuple[List[ScaleParams], MultiScaleDBNGraph, TrainingHistory]:
        empty_g = _empty_graph(graph)
        clean_params = _clear_parent_sets(params_list)
        sem = TopDownSEM(**self._sem_kwargs)
        return sem.fit(data, clean_params, empty_g)

    def predict(
        self,
        data: MultiScaleData,
        params_list: List[ScaleParams],
        graph: MultiScaleDBNGraph,
    ) -> List[PosteriorSummaries]:
        empty_g = _empty_graph(graph)
        clean_params = _clear_parent_sets(params_list)
        return evidence_chain(
            data,
            clean_params,
            empty_g,
            estep=self.estep,
            couple_scales=self.couple_scales,
        ).summaries


# ---------------------------------------------------------------------------
# OracleGraph: fixed ground-truth graph (upper bound)
# ---------------------------------------------------------------------------


class OracleGraph:
    """
    Top-down SEM with the ground-truth graph fixed (no S-step).

    This provides an upper bound: best achievable by any structure-learning
    method that perfectly recovers the graph.
    """

    #: Reads the generating graph, so the protocol hands it the oracle start.
    wants_oracle_start = True

    def __init__(self, true_graph: MultiScaleDBNGraph, **sem_kwargs) -> None:
        kw = dict(sem_kwargs)
        kw["do_structure"] = False
        self._sem_kwargs = kw
        self.estep, self.couple_scales = _scoring_settings(kw)
        self._true_graph = true_graph

    def fit(
        self,
        data: MultiScaleData,
        params_list: List[ScaleParams],
        graph: MultiScaleDBNGraph,
    ) -> Tuple[List[ScaleParams], MultiScaleDBNGraph, TrainingHistory]:
        sem = TopDownSEM(**self._sem_kwargs)
        return sem.fit(data, params_list, self._true_graph)

    def predict(
        self,
        data: MultiScaleData,
        params_list: List[ScaleParams],
        graph: MultiScaleDBNGraph,
    ) -> List[PosteriorSummaries]:
        return evidence_chain(
            data,
            params_list,
            self._true_graph,
            estep=self.estep,
            couple_scales=self.couple_scales,
        ).summaries


# ---------------------------------------------------------------------------
# OracleParams: no fitting at all - the identifiability ceiling
# ---------------------------------------------------------------------------


class OracleParams:
    """
    Return the initial parameters and graph unchanged (no E/M/S steps).

    The protocol hands this row the generating parameters and the generating
    graph, so it reports what the *data* supports: metrics no learner can beat.
    Comparing a fitted model against it separates "the learner failed" from "the
    scenario does not identify this quantity" - on the current grids the
    finest-scale regime accuracy of a perfect model is far below 1, so a low
    Acc_S is not by itself evidence of a learning bug.
    """

    #: Reads the generating parameters and graph, so it is a declared ceiling
    #: and the protocol hands it the oracle start.
    wants_oracle_start = True

    def __init__(self, **sem_kwargs) -> None:
        self._sem_kwargs = sem_kwargs
        self.estep, self.couple_scales = _scoring_settings(sem_kwargs)

    def fit(
        self,
        data: MultiScaleData,
        params_list: List[ScaleParams],
        graph: MultiScaleDBNGraph,
    ) -> Tuple[List[ScaleParams], MultiScaleDBNGraph, TrainingHistory]:
        history = TrainingHistory()
        chain = evidence_chain(
            data,
            params_list,
            graph,
            estep=self.estep,
            couple_scales=self.couple_scales,
        )
        history.record(chain.summaries, graph)
        return params_list, graph, history

    def predict(
        self,
        data: MultiScaleData,
        params_list: List[ScaleParams],
        graph: MultiScaleDBNGraph,
    ) -> List[PosteriorSummaries]:
        return evidence_chain(
            data,
            params_list,
            graph,
            estep=self.estep,
            couple_scales=self.couple_scales,
        ).summaries


# ---------------------------------------------------------------------------
# ClassicalStructure: a published structure learner supplies the graph
# ---------------------------------------------------------------------------


class ClassicalStructure:
    """
    Learn the lagged graph with a published algorithm, then fit this model on it.

    The comparison this supports is a controlled one.  Everything downstream of
    the graph is identical to `n2dbn`: the same latent-state inference, the
    same parameter blocks, the same held-out objective.  Only the graph comes
    from somewhere else, so a difference in any metric is attributable to the
    structure search and to nothing else.

    Parameters
    ----------
    algorithm : name accepted by :func:`~learning.classical.learn_structure`
    encoding  : how counts become the alphabet a discrete learner needs
    n_bins    : alphabet size for the `clip` and `quantile` encodings
    device    : `cpu`, or a torch device name for the DYNOTEARS solver
    options   : forwarded to :func:`~learning.classical.learn_structure`
    sem_kwargs: forwarded to `TopDownSEM`; structure search is always off
    """

    def __init__(
        self,
        algorithm: str = "hc-bic",
        encoding: str = "binary",
        n_bins: int = 4,
        device: str = "cpu",
        options: Optional[dict] = None,
        **sem_kwargs,
    ) -> None:
        kw = dict(sem_kwargs)
        kw["do_structure"] = False
        self._sem_kwargs = kw
        self.estep, self.couple_scales = _scoring_settings(kw)
        self.algorithm = algorithm
        self.options = dict(options or {})
        self.options.update(encoding=encoding, n_bins=n_bins, device=device)
        self.max_lag: Dict[int, int] = {}
        self.max_indegree: Dict[int, int] = {}

    def set_search_space(
        self, max_lag: Dict[int, int], max_indegree: Dict[int, int]
    ) -> None:
        """Point the search at the candidate space of the current scenario."""
        self.max_lag = dict(max_lag)
        self.max_indegree = dict(max_indegree)

    # ------------------------------------------------------------------
    def learn_graph(
        self,
        data: MultiScaleData,
        graph: MultiScaleDBNGraph,
    ) -> MultiScaleDBNGraph:
        """Graph carrying only the edges the classical algorithm returned."""
        from ..learning.classical import learn_multiscale_structure

        found = learn_multiscale_structure(
            self.algorithm,
            data,
            max_lag={k: self.max_lag.get(k, 1) for k in range(1, data.K + 1)},
            max_indegree={k: self.max_indegree.get(k, 0) for k in range(1, data.K + 1)},
            **self.options,
        )
        out = _empty_graph(graph)
        for k, per_label in found.items():
            for dst_ell, parents in per_label.items():
                for src_ell, lag in parents:
                    out.add_edge(k, src_ell, lag, dst_ell)
        return out

    def fit(
        self,
        data: MultiScaleData,
        params_list: List[ScaleParams],
        graph: MultiScaleDBNGraph,
    ) -> Tuple[List[ScaleParams], MultiScaleDBNGraph, TrainingHistory]:
        learned = self.learn_graph(data, graph)
        params = _clear_parent_sets(params_list)
        for ki, p in enumerate(params):
            for ell in range(p.L):
                p.parent_sets[ell] = learned.get_parents(ki + 1, ell)
            _seed_lag_coefficients(p)
        sem = TopDownSEM(**self._sem_kwargs)
        return sem.fit(data, params, learned)

    def predict(
        self,
        data: MultiScaleData,
        params_list: List[ScaleParams],
        graph: MultiScaleDBNGraph,
    ) -> List[PosteriorSummaries]:
        return evidence_chain(
            data,
            params_list,
            self.learn_graph(data, graph),
            estep=self.estep,
            couple_scales=self.couple_scales,
        ).summaries


# ---------------------------------------------------------------------------
# Param reduction helpers
# ---------------------------------------------------------------------------


#: The three orderings the sorting baseline can be run under, each naming the
#: score its causal order is read off.
SORTNREGRESS = ("var-sortnregress", "r2-sortnregress", "random-sortnregress")


class SortnRegressStructure(ClassicalStructure):
    """
    Supply the graph by sorting the labels and regressing, then fit this model on it.

    A benchmark whose variables happen to be ordered by marginal variance along
    the causal order can be recovered by a baseline that uses only that order,
    so a method scored on such a benchmark is partly scored on a regularity of
    the simulation.  `evaluation.metrics.varsortability` says how strong the
    regularity is; this row says how much of a result it buys, which is the
    question a sortability value on its own cannot answer.

    Everything downstream of the graph is what every other row of the comparison
    runs, so the distance between this row and `n2dbn` is attributable to the
    structure search and to nothing else.  `random-sortnregress` is the floor the
    other two are read against: it runs the identical regression under an
    ordering that carries no information.

    Args:
        algorithm: one of `SORTNREGRESS`.
        seed: fixes the random ordering, and with it that row.
        sem_kwargs: forwarded to `TopDownSEM`; structure search is always off.

    Raises:
        ValueError: the name is not one of the three.
    """

    def __init__(
        self, algorithm: str = "var-sortnregress", seed: int = 0, **sem_kwargs
    ) -> None:
        if algorithm not in SORTNREGRESS:
            raise ValueError(
                f"{algorithm!r} is not a sorting baseline; expected one of "
                f"{list(SORTNREGRESS)}"
            )
        super().__init__(algorithm=algorithm, **sem_kwargs)
        self.seed = int(seed)

    def learn_graph(
        self,
        data: MultiScaleData,
        graph: MultiScaleDBNGraph,
    ) -> MultiScaleDBNGraph:
        """Graph carrying the edges the ordering and the regression returned."""
        from .sortnregress import r2_sort_regress, random_sort_regress, var_sort_regress

        out = _empty_graph(graph)
        for k in range(1, data.K + 1):
            X = data.get_X(k).astype(float)
            max_lag = int(self.max_lag.get(k, 1))
            if self.algorithm == "var-sortnregress":
                edges = var_sort_regress(X, max_lag)
            elif self.algorithm == "r2-sortnregress":
                edges = r2_sort_regress(X, max_lag)
            else:
                rng = np.random.default_rng(self.seed + k)
                edges = random_sort_regress(X, max_lag, rng=rng)
            budget = int(self.max_indegree.get(k, 0))
            for dst_ell in {dst for _, _, dst in edges}:
                chosen = [e for e in sorted(edges) if e[2] == dst_ell]
                for src_ell, lag, _ in chosen[:budget] if budget else chosen:
                    out.add_edge(k, src_ell, lag, dst_ell)
        return out


def _zero_r(params: ScaleParams) -> ScaleParams:
    """Return a copy of `params` with r set to 0 (no factor)."""
    import numpy as np

    from ..model.params import FactorParams

    p = deepcopy(params)
    # Replace factor params with empty r=0 version
    r0_factor = FactorParams(
        M=p.M,
        r=0,
        d_ctx=p.factor.d_ctx,
        F=np.empty((p.M, 0, 0)),
        G=np.empty((p.M, 0, p.factor.d_ctx)),
        Sigma=np.empty((p.M, 0, 0)),
        mu0=np.empty(0),
        Sigma0=np.empty((0, 0)),
    )
    p.__dict__["factor"] = r0_factor
    return p


def _collapse_regimes(params: ScaleParams) -> ScaleParams:
    """
    Return a copy of `params` with M reduced to 1 (no regime switching).

    The single regime is formed by averaging all regime-specific parameters.
    A scale already carrying one regime is rebuilt rather than returned as it
    stands, because its cross-scale block is still shaped for the parent it had
    before the parent was collapsed.  Averaging over a length-one regime axis is
    an identity, so the only thing the rebuild changes is that block.
    """
    p = deepcopy(params)

    import numpy as np

    from ..model.params import RegimeParams

    d_ctx = p.regime.d_ctx
    # Every scale is collapsed, so the parent scale also has a single regime:
    # the cross-scale block must be 1×1, not 1×M_parent, or it no longer
    # conforms to the parent posterior it multiplies.
    parent_M = 1 if p.regime.cross is not None else 0

    rp = RegimeParams(
        M=1,
        parent_M=parent_M,
        d_ctx=d_ctx,
        init=np.array([0.0]),
        bias=np.zeros(1),
        trans=np.zeros((1, 1)),
        cross=np.zeros((1, parent_M)) if p.regime.cross is not None else None,
        ctx=np.zeros((1, d_ctx)),
    )
    p.__dict__["regime"] = rp

    # Collapse factor params to single regime (average across regimes)
    fp = p.factor
    fp.__dict__["F"] = fp.F.mean(axis=0, keepdims=True)
    fp.__dict__["G"] = fp.G.mean(axis=0, keepdims=True)
    fp.__dict__["Sigma"] = fp.Sigma.mean(axis=0, keepdims=True)
    fp.__dict__["M"] = 1

    # Coarse count: average betaS across regimes to single regime
    if p.is_coarsest and p.coarse_count is not None:
        cc = p.coarse_count
        cc.__dict__["betaS"] = cc.betaS.mean(axis=0, keepdims=True)
        cc.__dict__["gammaS"] = cc.gammaS.mean(axis=0, keepdims=True)

    if p.is_coarsest and p.coarse_dur is not None:
        cd = p.coarse_dur
        cd.__dict__["alphaS"] = cd.alphaS.mean(axis=0, keepdims=True)

    if not p.is_coarsest:
        if p.fine_count is not None:
            fc = p.fine_count
            fc.__dict__["betaS"] = fc.betaS.mean(axis=0, keepdims=True)
        if p.fine_dur is not None:
            fd = p.fine_dur
            fd.__dict__["alphaS"] = fd.alphaS.mean(axis=0, keepdims=True)

    return p
