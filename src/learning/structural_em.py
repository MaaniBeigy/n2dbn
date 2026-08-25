"""
learning/structural_em.py
=========================
Top-down Variational Structural EM outer loop.

Algorithm (Eq. 1562-1582)
--------------------------
Initialise Θ, G (random or from families).

Repeat until outer convergence:
    For k = K, K-1, ..., 1  (coarse to fine):
        E-step:  q_k^new = argmax_q  𝒥(q, Θ_k, G_k)          [estep.py]
        M-step:  Θ_k^new = argmax_Θ 𝒥(q_k^new, Θ, G_k^old)    [mstep.py]
        S-step:  G_k^new = argmax_G  𝒥(q_k^new, Θ_k^new, G)    [sstep.py]
    Optionally: joint warm-start refinement stage.

Monitor ELBO = Σ_k log_Z_k (approximate; each scale's log-evidence).

Features
--------
* Checkpoint / resume via numpy .npz + pickle.
* Per-scale convergence tracking.
* Optional joint warm-start pass after top-down sweep.
"""

from __future__ import annotations

import copy
import os
import pickle
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from ..core.graph import MultiScaleDBNGraph
from ..core.scales import ScaleHierarchy
from ..core.variables import MultiScaleData
from ..evaluation.sufficiency import (
    ScaleVerdict,
    data_ledger,
    refused_scales,
    report,
)
from ..inference.posterior import PosteriorSummaries
from ..model.params import ScaleParams
from .estep import EStep
from .mstep import MStep
from .sstep import SStep

# ---------------------------------------------------------------------------
# Training history dataclass
# ---------------------------------------------------------------------------


@dataclass
class TrainingHistory:
    """
    Stores the monitored objective and per-scale diagnostics across sweeps.

    The objective is the one the three steps maximise between them,

        J = Σ_k log Z_k − Σ_k λ_k |E_k|,

    the summed per-scale log evidence less the edge penalty.  Tracking the log
    evidence alone leaves out the term the structure step trades against it, so
    a sweep that correctly buys an edge can look like a step backwards.
    """

    elbo: List[float] = field(default_factory=list)
    log_Z_per_scale: List[List[float]] = field(default_factory=list)
    n_edges_per_scale: List[List[int]] = field(default_factory=list)
    penalty: List[float] = field(default_factory=list)
    wall_time: List[float] = field(default_factory=list)
    # Objectives of sweeps the outer guard rejected.  These never enter `elbo`,
    # so the recorded trace is monotone while the rejections stay measurable.
    rejected: List[float] = field(default_factory=list)
    # The pre-fit sufficiency ledger, one verdict per scale, filled on every fit
    # whether or not the gate acts on it.  A reader who sees no edge at a scale
    # otherwise cannot tell an absence of structure from an absence of epochs.
    support: List[ScaleVerdict] = field(default_factory=list)

    def record(
        self,
        summaries: List[Optional[PosteriorSummaries]],
        graph: MultiScaleDBNGraph,
        penalty: float = 0.0,
    ) -> None:
        K = len(summaries)
        total_log_Z = sum(s.log_Z for s in summaries if s is not None)
        self.penalty.append(float(penalty))
        self.elbo.append(total_log_Z - float(penalty))
        self.log_Z_per_scale.append(
            [s.log_Z if s is not None else 0.0 for s in summaries]
        )
        self.n_edges_per_scale.append([graph.n_edges(k) for k in range(1, K + 1)])
        self.wall_time.append(time.time())

    def converged(self, tol: float = 1e-4, n_check: int = 2) -> bool:
        """
        True if the last `n_check` ELBO changes are all below `tol`
        *relative to the ELBO magnitude*.

        The check used to be absolute, and with |ELBO| in the hundreds or
        thousands a threshold of 1e-3 could never fire: every fit ran the full
        `max_outer_iter` sweeps no matter how long it had been converged.
        """
        if len(self.elbo) < n_check + 1:
            return False
        deltas = [abs(self.elbo[-i] - self.elbo[-i - 1]) for i in range(1, n_check + 1)]
        scale = 1.0 + abs(self.elbo[-1])
        return all(d < tol * scale for d in deltas)


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------


def _save_checkpoint(
    path: str,
    params_list: List[ScaleParams],
    graph: MultiScaleDBNGraph,
    history: TrainingHistory,
    outer_iter: int,
) -> None:
    """Save training state to disk."""
    state = {
        "params_list": params_list,
        "graph": graph,
        "history": history,
        "outer_iter": outer_iter,
    }
    with open(path, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_checkpoint(path: str):
    """Load training state from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# TopDownSEM: the main learning object
# ---------------------------------------------------------------------------


def _jitter_start(params_list, seed: int, scale: float = 0.25):
    """
    A copy of the starting parameters with their emission levels perturbed.

    Only the levels.  Widening this to the factor loadings was measured and is
    worse: it turned family 3's mixed verdicts into three basins and raised its
    parameter error as the restart count rose, which is what selecting harder on
    an objective looks like when the objective and the parameter error disagree.
    """
    rng = np.random.default_rng(seed)
    out = [copy.deepcopy(p) for p in params_list]
    for p in out:
        count = p.coarse_count if p.is_coarsest else p.fine_count
        dur = p.coarse_dur if p.is_coarsest else p.fine_dur
        count.beta0 = count.beta0 + rng.normal(0.0, scale, size=count.beta0.shape)
        dur.alpha0 = dur.alpha0 + rng.normal(0.0, scale, size=dur.alpha0.shape)
    return out


class TopDownSEM:
    """
    Top-down Variational Structural EM for the N²DBN.

    Parameters
    ----------
    max_outer_iter : maximum number of full K-scale sweeps
    outer_tol      : *relative* ELBO convergence threshold (stops early once the
                     last two sweeps each moved the ELBO by < outer_tol·|ELBO|)
    do_structure   : if False, skip S-step (fixed graph)
    require_support : if True, a scale the pre-fit sufficiency ledger refuses is
                     emptied in both the graph and `params.parent_sets` and its
                     S-step is skipped for the whole fit, while its E-step and
                     M-step run as usual so the parameters stay estimated and
                     the scales below keep their cross-scale coupling.  The
                     ledger lands on `history.support` whenever the structure
                     step runs, since a fit that leaves this off still has to
                     say what its epochs support.  Default off: the recall curve
                     was swept at five labels and one coarse lag, and the ratio
                     it thresholds is linear in the label count, so the screen
                     transfers as an order of magnitude until it is re-derived
                     at the label counts and lag budgets in use.
    couple_scales  : if False, no parent-scale summaries are passed downward, in
                     either the E-step or the M-step; every scale is then fitted
                     as an independent DBN over its own epochs
    do_warmstart   : if True, run a final joint refinement pass (all scales)
    checkpoint_dir : if given, save checkpoints after each outer iteration
    lam_edge       : edge sparsity penalty (λ_k) - uniform across scales
    monotone_outer : if True (default), a sweep that lowers the monitored
                     objective below the best sweep so far is rejected: the
                     state reverts to the best sweep and the sweep is retried
                     with a halved M-step learning rate, up to
                     `outer_backtracks` consecutive times, after which the
                     fit stops at its best state.  The allocation normalizer of
                     a finer scale is evaluated at the regime-averaged score,
                     so a parameter block accepted on its own objective can
                     still lower the evidence the outer loop monitors; the
                     guard turns that residual into damped retries and an
                     early stop instead of a decreasing trace, and the fit
                     returns its best sweep rather than its last one.
                     Rejected objectives are kept in `history.rejected`.
    outer_backtracks : consecutive rejected sweeps allowed before stopping
    estep_kwargs   : passed to EStep(...)
    mstep_kwargs   : passed to MStep(...)
    sstep_kwargs   : passed to SStep(...)
    verbose        : 0 = silent, 1 = outer iterations, 2 = per-scale
    """

    def __init__(
        self,
        max_outer_iter: int = 20,
        outer_tol: float = 1e-4,
        do_structure: bool = True,
        require_support: bool = False,
        couple_scales: bool = True,
        do_warmstart: bool = False,
        checkpoint_dir: Optional[str] = None,
        lam_edge: float = 1.0,
        monotone_outer: bool = True,
        outer_backtracks: int = 2,
        structure_first: bool = False,
        n_restarts: int = 1,
        restart_seed: int = 0,
        estep_kwargs: Optional[dict] = None,
        mstep_kwargs: Optional[dict] = None,
        sstep_kwargs: Optional[dict] = None,
        verbose: int = 1,
    ) -> None:
        self.max_outer_iter = max_outer_iter
        self.outer_tol = outer_tol
        self.do_structure = do_structure
        self.require_support = require_support
        self.couple_scales = couple_scales
        self.do_warmstart = do_warmstart
        self.checkpoint_dir = checkpoint_dir
        self.lam_edge = lam_edge
        self.monotone_outer = monotone_outer
        self.outer_backtracks = outer_backtracks
        self.verbose = verbose

        self.structure_first = structure_first
        self.n_restarts = int(n_restarts)
        self.restart_seed = int(restart_seed)
        self.estep = EStep(**(estep_kwargs or {}))
        self.mstep = MStep(**(mstep_kwargs or {}))

        sstep_kw = dict(lam=lam_edge)
        sstep_kw.update(sstep_kwargs or {})
        self.sstep = SStep(**sstep_kw)

    # ------------------------------------------------------------------
    def fit(
        self,
        data: MultiScaleData,
        params_list: List[ScaleParams],
        graph: MultiScaleDBNGraph,
        resume_from: Optional[str] = None,
    ) -> Tuple[List[ScaleParams], MultiScaleDBNGraph, TrainingHistory]:
        """
        Fit the model, keeping the best of `n_restarts` starting points.

        A start that reaches a worse optimum is a basin problem rather than an
        estimation one, and the standard answer is to try more than one start
        and keep whichever ends highest on the objective the loop maximises.
        With one restart this is the fit itself and costs nothing.
        """
        if self.n_restarts <= 1 or resume_from:
            return self._fit_once(data, params_list, graph, resume_from)

        best: Optional[Tuple] = None
        best_obj = -np.inf
        for attempt in range(self.n_restarts):
            start = (
                params_list
                if attempt == 0
                else _jitter_start(params_list, self.restart_seed + attempt)
            )
            out = self._fit_once(
                data, [copy.deepcopy(p) for p in start], copy.deepcopy(graph)
            )
            obj = out[2].elbo[-1] if out[2].elbo else -np.inf
            if obj > best_obj:
                best, best_obj = out, obj
        return best

    def _fit_once(
        self,
        data: MultiScaleData,
        params_list: List[ScaleParams],  # initial params, index 0 = scale 1
        graph: MultiScaleDBNGraph,  # initial graph
        resume_from: Optional[str] = None,
    ) -> Tuple[List[ScaleParams], MultiScaleDBNGraph, TrainingHistory]:
        """
        Fit the N²DBN to data.

        Parameters
        ----------
        data         : observed multiscale data
        params_list  : initial parameters (params_list[k-1] = scale k)
        graph        : initial structural graph
        resume_from  : path to a checkpoint file to resume from

        Returns
        -------
        (params_list, graph, history)  - learned parameters, graph, training history
        """
        K = data.K
        hier = data.hierarchy
        history = TrainingHistory()
        start_iter = 0

        # ---- Resume from checkpoint if given ----------------------------
        if resume_from and os.path.exists(resume_from):
            state = _load_checkpoint(resume_from)
            params_list = state["params_list"]
            graph = state["graph"]
            history = state["history"]
            start_iter = state["outer_iter"] + 1
            if self.verbose >= 1:
                print(
                    f"[SEM] Resumed from iteration {start_iter} (checkpoint: {resume_from})"
                )

        params_list = [deepcopy(p) for p in params_list]
        graph = deepcopy(graph)

        # ---- Pre-fit sufficiency ledger ---------------------------------
        # Judged on the epochs handed to this call, which is the training split
        # when the caller split its series, and against the candidate set the
        # S-step will actually enumerate at each scale.  A refused scale has
        # fewer epochs per candidate than the count at which the sweep behind
        # the curve recovered anything, and it returned edges there anyway, so
        # the gate empties that scale once and skips its search for the whole
        # fit.  The E-step and the M-step still run there, which keeps the
        # parameters estimated and the scales below coupled to it.
        #
        # A fit with the structure step off proposes no graph of its own, so it
        # has nothing for the ledger to qualify and the ledger stays empty, the
        # same view `_edge_penalty` takes of the edge price.  The verdicts are
        # recorded and printed whatever `require_support` says, since a run that
        # leaves the gate off still owes its reader the arithmetic.
        refused: Set[int] = set()
        # One budget for the whole fit, so every step and every model compared
        # inside it is scored over the same rows.
        lags = [self.sstep._per_scale(self.sstep.max_lag, k) for k in range(1, K + 1)]
        self.estep.lag_budget = lags
        self.mstep.lag_budget = lags
        if self.do_structure:
            history.support = data_ledger(data, lags)
            unsupported = refused_scales(history.support)
            if self.require_support:
                refused = unsupported
                for k in sorted(refused):
                    for ell in range(data.L):
                        graph.set_parents(k, ell, set())
                        params_list[k - 1].parent_sets[ell] = set()
            if unsupported and self.verbose >= 1:
                print(report(history.support))

        # Posterior summaries per scale (indexed 0 = scale 1 = finest)
        summaries: List[Optional[PosteriorSummaries]] = [None] * K

        # Best sweep so far: (objective, params_list, graph, summaries).  The
        # outer guard reverts to it and the fit returns it, so a fit ends at
        # its best sweep rather than at whatever the last sweep produced.
        best_state: Optional[tuple] = None
        n_backtracks = 0
        # A rejected sweep halves the M-step learning rate for the rest of the
        # fit; reset it on entry so one fit's damping never leaks into the
        # next when the same instance fits several replicates in sequence.
        if not hasattr(self, "_lr0"):
            self._lr0 = self.mstep.lr
        self.mstep.lr = self._lr0

        # ---- Outer loop -------------------------------------------------
        for outer_iter in range(start_iter, self.max_outer_iter):
            t0 = time.time()
            if self.verbose >= 1:
                print(f"\n[SEM] Outer iteration {outer_iter + 1}/{self.max_outer_iter}")

            # Top-down sweep: coarsest to finest
            for k in range(K, 0, -1):  # K, K-1, ..., 1
                ki = k - 1  # 0-indexed

                # Parent summaries for cross-scale coupling
                parent_summaries = (
                    summaries[ki + 1] if (k < K and self.couple_scales) else None
                )

                if self.verbose >= 2:
                    print(f"  [SEM] scale k={k}  E-step ...", end=" ", flush=True)

                # ---- E-step ---------------------------------------------
                q_k = self.estep.run(
                    k=k,
                    data=data,
                    params=params_list[ki],
                    graph=graph,
                    parent_summaries=parent_summaries,
                )
                summaries[ki] = q_k

                if outer_iter == start_iter and parent_summaries is not None:
                    self._init_cross_scale(
                        params_list[ki], q_k, parent_summaries, hier, k
                    )

                search_first = self.structure_first and outer_iter == start_iter
                if search_first and self.do_structure and k not in refused:
                    graph, params_list[ki] = self.sstep.run(
                        k=k,
                        q=q_k,
                        params=params_list[ki],
                        graph=graph,
                        data=data,
                        hier=hier,
                    )

                if self.verbose >= 2:
                    print(f"log_Z={q_k.log_Z:.3f}  M-step ...", end=" ", flush=True)

                # ---- M-step ---------------------------------------------
                params_list[ki] = self.mstep.run(
                    k=k,
                    q=q_k,
                    params=params_list[ki],
                    data=data,
                    hier=hier,
                    parent_summaries=parent_summaries,
                )

                if self.verbose >= 2:
                    print(f"S-step ...", end=" ", flush=True)

                # ---- S-step (optional) ----------------------------------
                if self.do_structure and not search_first and k not in refused:
                    graph, params_list[ki] = self.sstep.run(
                        k=k,
                        q=q_k,
                        params=params_list[ki],
                        graph=graph,
                        data=data,
                        hier=hier,
                    )
                    if self.verbose >= 2:
                        print(f"|E_k|={graph.n_edges(k)}", flush=True)
                else:
                    if self.verbose >= 2:
                        print("(skipped)", flush=True)

            # ---- Outer guard and history --------------------------------
            penalty = self._edge_penalty(graph, data)
            obj = sum(s.log_Z for s in summaries if s is not None) - penalty
            elapsed = time.time() - t0
            eps = 1e-9 * (1.0 + abs(obj))

            if (
                self.monotone_outer
                and best_state is not None
                and obj < best_state[0] - eps
            ):
                # The sweep lowered the objective, which the allocation
                # normalizer's mean-field evaluation allows even though every
                # parameter block passed its own check.  Revert to the best
                # sweep instead of recording the decrease.
                drop = best_state[0] - obj
                history.rejected.append(float(obj))
                params_list, graph, summaries = self._copy_state(*best_state[1:])
                if drop < self.outer_tol * (1.0 + abs(best_state[0])):
                    # A sub-tolerance wiggle is convergence, not a failure.
                    if self.verbose >= 1:
                        print(
                            f"[SEM] Converged at the best sweep "
                            f"(objective {best_state[0]:.2f})."
                        )
                    break
                n_backtracks += 1
                if n_backtracks > self.outer_backtracks:
                    if self.verbose >= 1:
                        print(
                            f"[SEM] Stopped at the best sweep after "
                            f"{n_backtracks} rejected sweeps "
                            f"(objective {best_state[0]:.2f})."
                        )
                    break
                self.mstep.lr *= 0.5
                if self.verbose >= 1:
                    print(
                        f"  sweep rejected ({obj:.2f} < {best_state[0]:.2f}), "
                        f"retrying at lr={self.mstep.lr:.4g}  ({elapsed:.1f}s)"
                    )
                continue

            n_backtracks = 0
            history.record(summaries, graph, penalty)
            if best_state is None or obj > best_state[0]:
                best_state = (obj, *self._copy_state(params_list, graph, summaries))

            if self.verbose >= 1:
                print(
                    f"  ELBO ≈ {history.elbo[-1]:.2f}  "
                    f"|E|={graph.n_edges()}  "
                    f"({elapsed:.1f}s)"
                )

            # ---- Checkpoint ---------------------------------------------
            if self.checkpoint_dir:
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                cp_path = os.path.join(
                    self.checkpoint_dir, f"checkpoint_{outer_iter:04d}.pkl"
                )
                _save_checkpoint(cp_path, params_list, graph, history, outer_iter)

            # ---- Convergence check ---------------------------------------
            if history.converged(tol=self.outer_tol):
                if self.verbose >= 1:
                    print(f"[SEM] Converged at outer iteration {outer_iter + 1}.")
                break

        # ---- Optional joint warm-start pass -----------------------------
        if self.do_warmstart:
            if self.verbose >= 1:
                print("[SEM] Running warm-start joint refinement ...")
            params_list, graph, summaries = self._warmstart_pass(
                data, params_list, graph, summaries, hier
            )
            penalty = self._edge_penalty(graph, data)
            obj = sum(s.log_Z for s in summaries if s is not None) - penalty
            if (
                self.monotone_outer
                and best_state is not None
                and obj < best_state[0] - 1e-9 * (1.0 + abs(obj))
            ):
                # The refinement pass is subject to the same guard as a sweep.
                history.rejected.append(float(obj))
                params_list, graph, summaries = self._copy_state(*best_state[1:])
            else:
                history.record(summaries, graph, penalty)

        return params_list, graph, history

    # ------------------------------------------------------------------
    @staticmethod
    def _copy_state(
        params_list: List[ScaleParams],
        graph: MultiScaleDBNGraph,
        summaries: List[Optional[PosteriorSummaries]],
    ) -> tuple:
        """Deep copies of the mutable fit state, in fit's own layout."""
        return (
            [deepcopy(p) for p in params_list],
            deepcopy(graph),
            [deepcopy(s) if s is not None else None for s in summaries],
        )

    # ------------------------------------------------------------------
    def _edge_penalty(self, graph: MultiScaleDBNGraph, data: MultiScaleData) -> float:
        """Σ_k λ_k |E_k| under the structure step's own penalty schedule."""
        if not self.do_structure:
            return 0.0
        total = 0.0
        for k in range(1, data.K + 1):
            lam = self.sstep._lam_effective(data.get_X(k).shape[0], k)
            total += lam * graph.n_edges(k)
        return float(total)

    # ------------------------------------------------------------------
    def _init_cross_scale(
        self,
        params: ScaleParams,
        q: PosteriorSummaries,
        parent_summaries: PosteriorSummaries,
        hier: ScaleHierarchy,
        k: int,
    ) -> None:
        """
        Warm-start the cross-scale block from the parent-child regime table.

        C[m, n] is set to the log ratio between the posterior probability of
        regime m among epochs whose parent sits in regime n and its overall
        posterior probability, which is the log-odds a single sweep of gradient
        ascent would have to travel to from zero.
        """
        regime = params.regime
        if regime.cross is None:
            return
        parent_gamma = parent_summaries.gamma[hier.parent_maps[k - 1]]  # (T, M_par)
        joint = q.gamma.T @ parent_gamma  # (M, M_par)
        eps = 1e-6
        cond = joint / np.maximum(joint.sum(axis=0, keepdims=True), eps)
        marg = q.gamma.mean(axis=0)[:, None]
        cross = np.log(np.maximum(cond, eps)) - np.log(np.maximum(marg, eps))
        regime.cross = np.clip(cross, -4.0, 4.0)

    # ------------------------------------------------------------------
    def _warmstart_pass(
        self,
        data: MultiScaleData,
        params_list: List[ScaleParams],
        graph: MultiScaleDBNGraph,
        summaries: List[Optional[PosteriorSummaries]],
        hier: ScaleHierarchy,
    ) -> Tuple[
        List[ScaleParams], MultiScaleDBNGraph, List[Optional[PosteriorSummaries]]
    ]:
        """
        One additional top-down sweep with structure search disabled,
        using the current posterior summaries as warm start.
        """
        K = data.K
        for k in range(K, 0, -1):
            ki = k - 1
            parent_summaries = (
                summaries[ki + 1] if (k < K and self.couple_scales) else None
            )
            q_k = self.estep.run(k, data, params_list[ki], graph, parent_summaries)
            summaries[ki] = q_k
            params_list[ki] = self.mstep.run(
                k,
                q_k,
                params_list[ki],
                data,
                hier,
                parent_summaries=parent_summaries,
            )
        return params_list, graph, summaries

    # ------------------------------------------------------------------
    def compute_elbo(
        self,
        summaries: List[Optional[PosteriorSummaries]],
    ) -> float:
        """
        Approximate ELBO = Σ_k log_Z_k.

        A full ELBO would include entropy terms and parameter priors,
        but the sum of scale-level log marginal likelihoods is the
        dominant quantity and sufficient for convergence monitoring.
        """
        return sum(s.log_Z for s in summaries if s is not None)
