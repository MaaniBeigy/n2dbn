"""
synthetic/generator.py
======================
Top-down N²DBN data generator.

Frozen.  This module belongs to the allocation formulation, in which a coarse
count is drawn and then split downward among its children.  It keeps working
and keeps reproducing the stored results, and nothing new is built on it.

Generation algorithm
--------------------
1.  For k = K down to 1:
    a.  Sample regime sequence  S^(k)[0..H_k-1]  (sequential Markov chain).
    b.  Sample factor trajectory H^(k)[0..H_k-1]  (LGSSM given regimes).

2.  At coarsest scale k=K:
    For each t and label ℓ:
      * Sample X_t^{(K,ℓ)} ~ ZINB(π_t, r, μ_t)
      * If X_t > 0: Sample Y_t^{(K,ℓ)} ~ NB(s, ν_t)
      * Else: Y_t = 0

3.  For k = K-1 down to 1 (finer scales):
    For each parent block c ∈ T^{(k+1)} and label ℓ:
      * Compute allocation scores η_t for children.
      * Sample (X_t)_{t∈children} ~ Dir-Mult(X_c, η).
      * Active set  𝒜_c = {t : X_t > 0}.
      * Compute allocation scores ξ_t for active children.
      * Sample (Y_t)_{t∈𝒜_c} ~ Dir-Mult(Y_c, ξ).
      * Set Y_t = 0 for inactive children.

4.  At finest scale k=1:
    Reconstruct episode traces from (X^(1), Y^(1)).

5.  Aggregate upward  to verify RecErr = 0.

Key reconciliation guarantee
-----------------------------
By construction of Dir-Mult:
    Σ_{t∈children(c)} X_t = X_c   and   Σ_{t∈children(c)} Y_t = Y_c
hold almost surely for every block c, scale k, and label ℓ.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.calendar import SlotCalendar, episodes_to_event_log
from ..core.scales import ScaleHierarchy
from ..core.variables import LatentTrajectories, MultiScaleData
from ..inference.reconstruction import EpisodeReconstructor
from ..model.allocation import (
    allocate_counts_block,
    allocate_counts_block_sequential,
    allocate_durations_block,
    compute_count_scores,
    redistribute_excess,
)
from ..model.emissions import (
    _sigmoid,
    compute_log_mu,
    compute_logit_pi,
    nb_sample,
    zinb_sample,
)
from ..model.latent_factor import LatentFactor
from ..model.latent_regime import RegimeChain
from ..model.params import ScaleParams

# ---------------------------------------------------------------------------


@dataclass
class GeneratedDataset:
    """
    Complete output of the generator.

    Attributes
    ----------
    data     : observed multiscale arrays (X, Y, C).
    latents  : ground-truth latent trajectories (S, H).
    traces   : reconstructed finest-scale episode traces.
    params   : per-scale parameter objects used for generation.
    hierarchy: temporal hierarchy.
    """

    data: MultiScaleData
    latents: LatentTrajectories
    traces: Dict[int, list]  # {label_idx: [Episode, ...]}
    params: List[ScaleParams]  # params[k-1] = ScaleParams for scale k
    hierarchy: ScaleHierarchy
    calendar: Optional[SlotCalendar] = None
    label_names: Optional[List[str]] = None

    def event_log(self):
        """
        Reconstructed episodes as wall-clock records.

        Each record carries the label, the start and end as `dd hh:mm`, the ISO
        timestamps and the duration in minutes.  Requires a calendar, which the
        generator attaches when the scenario declares `slot_minutes`.
        """
        if self.calendar is None:
            raise ValueError("no calendar attached, so epochs carry no wall-clock time")
        names = self.label_names or [f"L{i}" for i in range(self.data.L)]
        return episodes_to_event_log(self.traces, names, self.calendar)


# ---------------------------------------------------------------------------


def _needs_sequential(parent_sets_ell, block_size: int) -> bool:
    """Whether any lagged parent of this label can read inside its own block."""
    return any(lag < block_size for _src, lag in parent_sets_ell)


class N2DBNGenerator:
    """
    Top-down generative model for the N²DBN.

    Parameters
    ----------
    params_list  : list[ScaleParams]  one per scale, ordered finest (index 0)
                   to coarsest (index K-1).
    hierarchy    : ScaleHierarchy describing the nested temporal structure.
    label_names  : list of event label strings.
    calendar     : optional wall-clock semantics for the epoch indices.
    finest_cap: episode starts of one label a finest-scale epoch may hold, when
                `finest_binary` is set.  One is the event-start reading, at most
                one episode of a label begins per epoch.  A larger cap lets an
                epoch begin several, which is admissible because an episode is
                dated at the resolution of the finest scale and the log does not
                order events inside an epoch.
    finest_binary: if True, every epoch holds at most as many events of a label
                   as it has finest-scale slots, which forces X^(1) into {0,1}.
    """

    def __init__(
        self,
        params_list: List[ScaleParams],
        hierarchy: ScaleHierarchy,
        label_names: List[str],
        exclusivity_set: Optional[set] = None,
        calendar: Optional[SlotCalendar] = None,
        finest_binary: bool = False,
        finest_cap: int = 1,
    ) -> None:
        self.params_list = params_list  # params_list[k-1] = scale k
        self.hierarchy = hierarchy
        self.label_names = label_names
        self.K = hierarchy.K
        self.L = len(label_names)
        self.calendar = calendar
        self.finest_binary = finest_binary
        self.finest_cap = max(int(finest_cap), 1)
        self.reconstructor = EpisodeReconstructor(label_names, exclusivity_set)

        # Validate scales
        for ki, p in enumerate(params_list):
            assert p.k == ki + 1, f"params_list[{ki}].k should be {ki+1}"

    # ------------------------------------------------------------------
    def generate(
        self,
        seed: int = 0,
        context_fn=None,
    ) -> GeneratedDataset:
        """
        Generate one complete multiscale dataset.

        Parameters
        ----------
        seed        : random seed.
        context_fn  : optional callable (k, T, rng) to (T, d_ctx) context array.
                      If None, context is sampled from N(0,1).

        Returns
        -------
        GeneratedDataset with exact RecErr_X = RecErr_Y = 0.
        """
        rng = np.random.default_rng(seed)
        K = self.K
        hier = self.hierarchy

        # ----------------------------------------------------------------
        # Step 1 - generate context features at all scales
        # ----------------------------------------------------------------
        ctx_arrays: List[np.ndarray] = []
        for k in range(1, K + 1):
            p = self.params_list[k - 1]
            T = hier.n_epochs(k)
            d = p.factor.d_ctx
            if context_fn is not None:
                C_k = context_fn(k, T, rng)
            else:
                C_k = rng.standard_normal((T, d)) if d > 0 else np.empty((T, 0))
            ctx_arrays.append(C_k)

        # ----------------------------------------------------------------
        # Step 2 - sample latent trajectories at all scales (coarse to fine)
        # ----------------------------------------------------------------
        S_seqs: List[np.ndarray] = [None] * K  # S_seqs[k-1] = (H_k,) int
        H_trajs: List[np.ndarray] = [None] * K  # H_trajs[k-1] = (H_k, r_k)

        for k in range(K, 0, -1):  # K, K-1, ..., 1
            ki = k - 1
            p = self.params_list[ki]
            T = hier.n_epochs(k)
            C_k = ctx_arrays[ki]

            parent_S = S_seqs[ki + 1] if k < K else None
            if parent_S is not None:
                # Map coarser-scale S to each fine epoch via parent map
                pm = hier.parent_maps[ki]  # (H_k,) parent indices
                parent_S_expanded = parent_S[pm]  # (H_k,)
            else:
                parent_S_expanded = None

            S_seqs[ki] = RegimeChain(p.regime).sample_sequence(
                T, C_k, parent_S_expanded, rng
            )
            H_trajs[ki] = LatentFactor(p.factor).sample_trajectory(
                T, S_seqs[ki], C_k, rng
            )

        # ----------------------------------------------------------------
        # Step 3 - allocate observed data (coarse emissions + finer alloc)
        # ----------------------------------------------------------------
        # Pre-allocate arrays
        data = MultiScaleData.allocate(
            hierarchy=hier,
            label_names=self.label_names,
            context_dims=[p.factor.d_ctx for p in self.params_list],
            calendar=self.calendar,
        )
        # Fill context
        for k in range(1, K + 1):
            data.C[k - 1][:] = ctx_arrays[k - 1]

        # Coarsest scale
        self._generate_coarsest(
            data, S_seqs[K - 1], H_trajs[K - 1], ctx_arrays[K - 1], rng
        )

        # Finer scales (K-1 down to 1)
        for k in range(K - 1, 0, -1):
            self._allocate_finer_scale(
                k, data, S_seqs[k - 1], H_trajs[k - 1], ctx_arrays[k - 1], rng
            )

        # ----------------------------------------------------------------
        # Step 4 - reconstruct episodes at finest scale
        # ----------------------------------------------------------------
        traces = self.reconstructor.reconstruct(data.get_X(1), data.get_Y(1))

        # ----------------------------------------------------------------
        # Step 5 - pack and return
        # ----------------------------------------------------------------
        latents = LatentTrajectories(
            hierarchy=hier,
            S=[S_seqs[k - 1] for k in range(1, K + 1)],
            H_traj=[H_trajs[k - 1] for k in range(1, K + 1)],
        )

        return GeneratedDataset(
            data=data,
            latents=latents,
            traces=traces,
            params=self.params_list,
            hierarchy=hier,
            calendar=self.calendar,
            label_names=list(self.label_names),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _slots_per_epoch(self, k: int) -> int:
        """Finest-scale slots contained in one epoch at scale k."""
        span = 1
        for j in range(1, k):
            span *= self.hierarchy.n_epochs(j) // self.hierarchy.n_epochs(j + 1)
        return max(span, 1)

    def _generate_coarsest(
        self,
        data: MultiScaleData,
        S_K: np.ndarray,  # (H_K,) regime sequence
        H_K: np.ndarray,  # (H_K, r)
        ctx: np.ndarray,  # (H_K, d_ctx)
        rng: np.random.Generator,
    ) -> None:
        """
        Sample X^{(K)} ~ ZINB and Y^{(K)} ~ NB at the coarsest scale.
        Writes directly into data.X[K-1] and data.Y[K-1].
        """
        K = self.K
        p = self.params_list[K - 1]
        cp = p.coarse_count  # CoarseCountParams
        dp = p.coarse_dur  # CoarseDurParams
        T = self.hierarchy.n_epochs(K)
        L = self.L

        X_K = data.X[K - 1]  # (T, L)
        Y_K = data.Y[K - 1]

        for t in range(T):
            s = int(S_K[t])
            h_t = H_K[t]
            c_t = ctx[t]

            # Aggregate lagged count terms for each label
            lag_X = np.zeros(L)
            for dst_ell in range(L):
                for (src_ell, d), coef in cp.betaLag.items():
                    if (src_ell, d) in p.parent_sets.get(dst_ell, set()):
                        t_lag = t - d
                        if t_lag >= 0:
                            lag_X[dst_ell] += float(coef[dst_ell]) * np.log1p(
                                float(X_K[t_lag, src_ell])
                            )

            # ZINB log-mean and ZI logit for all labels
            log_mu = compute_log_mu(
                s, h_t, c_t, cp.beta0, cp.betaS, cp.betaH, cp.betaCtx, lag_X
            )
            logit_pi = compute_logit_pi(
                s, h_t, c_t, cp.gamma0, cp.gammaS, cp.gammaH, cp.gammaCtx
            )

            mu = np.exp(log_mu)  # (L,)
            pi = _sigmoid(logit_pi)  # (L,)

            X_t = zinb_sample(pi, cp.r_disp, mu, rng)  # (L,)
            if self.finest_binary:
                # An epoch cannot hold more event starts than it has slots.
                X_t = np.minimum(X_t, self.finest_cap * self._slots_per_epoch(K))
            X_K[t] = X_t

            # Duration: NB conditional on X_t > 0
            for ell in range(L):
                if X_t[ell] == 0:
                    Y_K[t, ell] = 0
                    continue

                # Aggregate lagged duration terms
                lag_Y_ell = 0.0
                for (src_ell, d), coef in dp.alphaLag.items():
                    if (src_ell, d) in p.parent_sets.get(ell, set()):
                        t_lag = t - d
                        if t_lag >= 0:
                            lag_Y_ell += float(coef[ell]) * np.log1p(
                                float(Y_K[t_lag, src_ell])
                            )

                log_nu = (
                    float(dp.alpha0[ell])
                    + float(dp.alphaS[s, ell])
                    + (
                        float(dp.alphaH[:, ell] @ h_t)
                        if dp.alphaH.shape[0] > 0
                        else 0.0
                    )
                    + (
                        float(dp.alphaCtx[:, ell] @ c_t)
                        if dp.alphaCtx.shape[0] > 0
                        else 0.0
                    )
                    + float(dp.alphaX[ell]) * np.log1p(float(X_t[ell]))
                    + lag_Y_ell
                )
                nu = max(np.exp(log_nu), 1.0)  # enforce mean >= 1 slot
                y = int(
                    nb_sample(
                        np.array([dp.s_disp[ell]]),
                        np.array([nu]),
                        rng,
                    )[0]
                )
                if self.finest_binary:
                    # Y is normalised by the slots of the epoch, so it cannot
                    # exceed them.  With the generating parameters chosen for
                    # that ceiling this binds only in the far tail.
                    y = min(y, self.finest_cap * self._slots_per_epoch(K))
                Y_K[t, ell] = y

    # ------------------------------------------------------------------
    def _allocate_finer_scale(
        self,
        k: int,  # finer scale to generate (1 <= k < K)
        data: MultiScaleData,
        S_k: np.ndarray,  # (H_k,)
        H_k: np.ndarray,  # (H_k, r)
        ctx: np.ndarray,  # (H_k, d_ctx)
        rng: np.random.Generator,
    ) -> None:
        """
        Allocate X^{(k)} and Y^{(k)} from parent-scale totals via Dir-Mult.

        Guarantees (by construction):
            Σ_{t∈children(c)} X_t^{(k,ℓ)} = X_c^{(k+1,ℓ)}  for all c, ℓ
            Σ_{t∈children(c)} Y_t^{(k,ℓ)} = Y_c^{(k+1,ℓ)}  for all c, ℓ
        """
        hier = self.hierarchy
        p = self.params_list[k - 1]
        fp = p.fine_count  # FineCountParams
        fdp = p.fine_dur  # FineDurParams

        X_fine = data.X[k - 1]  # (H_k, L) - being filled
        Y_fine = data.Y[k - 1]
        X_coarse = data.X[k]  # (H_{k+1}, L) - already filled
        Y_coarse = data.Y[k]

        H_coarse = hier.n_epochs(k + 1)

        for c in range(H_coarse):
            children = hier.children(k, c)
            if not children:
                continue

            for ell in range(self.L):
                X_parent = int(X_coarse[c, ell])
                Y_parent = int(Y_coarse[c, ell])

                # ---- Count allocation --------------------------------
                parents_ell = p.parent_sets.get(ell, set())
                caps = (
                    np.full(len(children), self.finest_cap * self._slots_per_epoch(k))
                    if self.finest_binary
                    else None
                )
                if _needs_sequential(parents_ell, len(children)):
                    # A lagged score that reads a sibling inside this block has
                    # to read a value that is already written, which the joint
                    # draw cannot provide.
                    x_alloc = allocate_counts_block_sequential(
                        children=children,
                        ell=ell,
                        X_parent=X_parent,
                        Y_parent=Y_parent,
                        s_seq=S_k,
                        H_traj=H_k,
                        ctx=ctx,
                        X_fine=X_fine,
                        Y_fine=Y_fine,
                        p=fp,
                        parent_sets_ell=parents_ell,
                        rng=rng,
                        write=lambda t, v: X_fine.__setitem__((t, ell), v),
                    )
                    if caps is not None:
                        scores = compute_count_scores(
                            children,
                            S_k,
                            H_k,
                            ctx,
                            X_parent,
                            Y_parent,
                            X_fine,
                            Y_fine,
                            ell,
                            fp,
                            parents_ell,
                        )
                        z = scores - scores.max()
                        omega = np.exp(z)
                        x_alloc = redistribute_excess(
                            x_alloc, caps, omega / omega.sum(), rng
                        )
                else:
                    x_alloc = allocate_counts_block(
                        c=c,
                        children=children,
                        ell=ell,
                        X_parent=X_parent,
                        Y_parent=Y_parent,
                        s_seq=S_k,
                        H_traj=H_k,
                        ctx=ctx,
                        X_fine=X_fine,
                        Y_fine=Y_fine,
                        p=fp,
                        parent_sets_ell=parents_ell,
                        rng=rng,
                        caps=caps,
                    )
                for i, t in enumerate(children):
                    X_fine[t, ell] = x_alloc[i]

                # ---- Duration allocation -----------------------------
                y_alloc = allocate_durations_block(
                    c=c,
                    children=children,
                    ell=ell,
                    Y_parent=Y_parent,
                    X_fine_children=x_alloc,
                    s_seq=S_k,
                    H_traj=H_k,
                    ctx=ctx,
                    X_fine=X_fine,
                    Y_fine=Y_fine,
                    p=fdp,
                    parent_sets_ell=parents_ell,
                    rng=rng,
                    floor_per_active=self.finest_binary,
                )
                for i, t in enumerate(children):
                    Y_fine[t, ell] = y_alloc[i]
