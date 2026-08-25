"""
synthetic/scenarios.py
======================
Scenario configuration dataclasses and a grid factory.

Frozen.  This module belongs to the allocation formulation, in which a coarse
count is drawn and then split downward among its children.  It keeps working
and keeps reproducing the stored results, and nothing new is built on it.

Each ScenarioConfig fully specifies a synthetic dataset:
  * temporal hierarchy (K scales, branching factors, H_finest)
  * label set (L labels, sparsity level)
  * per-scale model hyperparameters (M_k, r_k, max_lag, zero-inflation)
  * misspecification flags (Family IV)

The ScenarioGrid sweeps the parameter grid from Section 8 of the paper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple

# ---------------------------------------------------------------------------


@dataclass
class ScenarioConfig:
    """
    Full specification for one synthetic benchmark scenario.

    Attributes
    ----------
    name          : human-readable identifier.
    K             : number of scales.
    H_finest      : epochs at the finest scale.
    branching     : list[int] of length K-1; branching factor per level.
    L             : number of event labels.
    M_k           : list[int] of length K; regime count per scale.
    r_k           : list[int] of length K; latent-factor dim per scale.
    max_lag       : list[int] of length K; maximum lag searched per scale.
    max_indegree  : list[int] of length K; true max in-degree per scale.
    zi_level      : zero-inflation level ('low','medium','high').
    sparse_graph  : if True, sample sparse ground-truth graph; else empty.
    family        : data family (1=stationary, 2=regime, 3=factor, 4=misspec).
    misspec_type  : misspecification type for Family IV (None otherwise).
    seed          : random seed for reproducibility.
    n_replicates  : number of independent replicates.
    context_dim   : list[int] of length K; context feature dimension per scale.
    slot_minutes  : wall-clock length of one finest-scale epoch, in minutes.
                    With branching [10, 6] and slot_minutes = 1 the three scales
                    are one minute, ten minutes and one hour.
    origin        : ISO timestamp of the first finest-scale slot, default
                    2026-01-01 00:00.  Fixes the `dd hh:mm` stamps of the
                    generated event log.
    finest_binary : if True the finest-scale allocation is capacity-constrained
                    so that X^(1) in {0,1} holds exactly, which is the event-start
                    reading of the finest scale.  The parent count of a block is
                    then bounded by the number of child slots.
    regime_strength : multiplier on the regime effect in the emission and
                    allocation scores.  The base value leaves regimes too weak to
                    identify, so a grid that studies regime recovery raises it.
    edge_strength : multiplier on the ground-truth lagged-edge coefficients.
                    Controls how much evidence an edge leaves in the data, and
                    therefore whether the graph is *identifiable at all*.  With
                    the base value (1.0) a true finest-scale edge is worth well
                    under 1 nat of log-likelihood over 1000 epochs - less than
                    the best spurious edge - so no consistent structure learner
                    can recover it.  Use `ScenarioGrid.identifiability_probe`
                    to check the regime of a scenario before running a study.
                    **It governs the finer scales only.**  At the coarsest scale
                    the total is drawn from `coarse_lag_total` instead, because
                    the stability bound there decides the answer whatever a
                    multiplier says.
    coarse_lag_total : range the coarsest-scale total lag coefficient is drawn
                    from, per label, so coarse edges differ in strength.  The
                    upper end must stay below one, since the fixed point of the
                    count autoregression is the intercept over one minus the
                    total.
    """

    name: str
    K: int
    H_finest: int
    branching: List[int]  # length K-1
    L: int
    M_k: List[int]  # length K
    r_k: List[int]  # length K
    max_lag: List[int]  # length K
    max_indegree: List[int]  # length K  - in the *true* graph
    zi_level: str = "medium"  # 'low' | 'medium' | 'high'
    sparse_graph: bool = True
    family: int = 1  # 1 | 2 | 3 | 4
    misspec_type: Optional[str] = None
    seed: int = 0
    n_replicates: int = 5
    context_dim: Optional[List[int]] = None  # default: 1 per scale
    edge_strength: float = 1.0
    regime_strength: float = 1.0  # multiplier on the regime effect
    dirichlet_conc: Optional[float] = None  # κ/τ of the finer-scale allocations
    finest_rate: Optional[float] = None  # target mean count per finest epoch
    slot_minutes: int = 1  # wall-clock duration of one finest slot
    origin: Optional[str] = None  # ISO start time of the first slot
    finest_binary: bool = False  # cap the finest scale at `finest_cap` starts
    # Episode starts of one label a finest-scale epoch may hold.  One is the
    # event-start reading.  A larger cap is still an event log, since an episode
    # is dated at the resolution of the finest scale and the log does not claim
    # to order events inside an epoch.
    finest_cap: int = 1
    # Range the coarsest-scale total lag coefficient is drawn from, per label.
    # None takes the default in `families`.  The upper end must stay below one,
    # since the fixed point of the count autoregression is 1 / (1 - total).
    coarse_lag_total: Optional[Tuple[float, float]] = None

    def __post_init__(self) -> None:
        assert len(self.branching) == self.K - 1
        assert len(self.M_k) == self.K
        assert len(self.r_k) == self.K
        assert len(self.max_lag) == self.K
        assert len(self.max_indegree) == self.K
        assert self.zi_level in ("low", "medium", "high")
        assert self.family in (1, 2, 3, 4)
        if self.context_dim is None:
            self.context_dim = [2] * self.K

    @property
    def H_k(self) -> List[int]:
        """Derive epoch counts at each scale from H_finest and branching factors."""
        H = [self.H_finest]
        for bf in self.branching:
            H.append(H[-1] // bf)
        return H

    @property
    def zi_prob(self) -> float:
        """Structural zero probability implied by zi_level."""
        return {"low": 0.1, "medium": 0.35, "high": 0.65}[self.zi_level]

    @property
    def coarse_log_mean(self) -> Optional[float]:
        """
        Coarsest-scale log-mean count implied by `finest_rate` (None if unset).

        Counts are allocated top-down, so the finest scale sees the coarsest
        count spread over H_1 / H_K slots.  Leaving the coarsest log-mean at its
        default of one to seven events therefore empties the finest scale, since
        with 60 finest slots per coarsest epoch only about 3% of them carry any
        event, and neither lagged edges nor regimes are identifiable there.

        `finest_rate` fixes the mean count per finest epoch instead, counted
        over epochs where the label is not structurally absent.  Conditioning on
        presence keeps the coarsest mean below the capacity of the block when
        `finest_binary` is set, which a division by 1 - zi_prob would not.
        """
        if self.finest_rate is None:
            return None
        H = self.H_k
        slots_per_coarse = H[0] / H[self.K - 1]
        import math

        return math.log(self.finest_rate * slots_per_coarse)

    @property
    def calendar(self):
        """SlotCalendar implied by `slot_minutes`, `branching` and `origin`."""
        from datetime import datetime

        from ..core.calendar import DEFAULT_ORIGIN, SlotCalendar

        origin = datetime.fromisoformat(self.origin) if self.origin else DEFAULT_ORIGIN
        return SlotCalendar.from_config(
            branching=self.branching,
            slot_minutes=self.slot_minutes,
            origin=origin,
        )

    @property
    def scale_spans(self) -> List[str]:
        """Wall-clock length of one epoch at each scale, coarsest last."""
        cal = self.calendar
        return [cal.describe()[k] for k in range(1, self.K + 1)]

    def short_id(self) -> str:
        """Short string key for logging."""
        return (
            f"F{self.family}_L{self.L}_K{self.K}"
            f"_M{''.join(str(m) for m in self.M_k)}"
            f"_r{''.join(str(r) for r in self.r_k)}"
            f"_zi{self.zi_level[0]}"
            f"_s{self.seed}"
        )


# ---------------------------------------------------------------------------


@dataclass
class ScenarioGrid:
    """
    Iterable container of ScenarioConfig objects spanning the evaluation grid.

    The grid sweeps:
        L ∈ {5, 10, 20}
        M_k ∈ {1, 2, 4}  (same for all scales)
        r_k ∈ {0, 1, 2}  (same for all scales)
        d*_k ∈ {1, 2}    (max in-degree)
        H_finest ∈ {1440, 14400}  (1 day / 10 days at 1-min finest)
        zi_level ∈ {'low','medium','high'}
        family ∈ {1, 2, 3, 4}
    """

    scenarios: List[ScenarioConfig] = field(default_factory=list)

    def __iter__(self) -> Iterator[ScenarioConfig]:
        return iter(self.scenarios)

    def __len__(self) -> int:
        return len(self.scenarios)

    def filter(self, **kwargs) -> "ScenarioGrid":
        """Return a sub-grid matching keyword conditions."""
        filtered = [
            s
            for s in self.scenarios
            if all(getattr(s, k) == v for k, v in kwargs.items())
        ]
        return ScenarioGrid(scenarios=filtered)

    # ------------------------------------------------------------------
    @staticmethod
    def identifiability_probe(cfg: "ScenarioConfig", seed: int = 0) -> dict:
        """
        Ask of one scenario: *is there anything to learn?*

        Generates one replicate, runs the E-step with the ground-truth
        parameters and graph, and reports per scale

          `n_epochs`    epochs available for estimation (train split)
          `density`     fraction of epochs carrying at least one event
          `true_gain`   median log-likelihood gain of a **true** lagged edge
          `null_gain95` 95th percentile gain of a **non-edge**
          `lam_bic`     the BIC penalty an edge must beat (log T)
          `oracle_accS` best-permutation regime accuracy achievable with the
                          true parameters - an upper bound for any learner

        A scenario is only worth running a structure study on when
        `true_gain` comfortably exceeds both `null_gain95` and `lam_bic`;
        latent metrics are only meaningful relative to `oracle_accS`.
        """
        import numpy as np

        from ..evaluation.metrics import LatentMetrics
        from ..learning.sstep import _CoarseLabelScorer, _FineLabelScorer
        from .latent_screen import oracle_posterior

        oracle = oracle_posterior(cfg, seed=seed)
        ds, train, summaries = oracle.dataset, oracle.train, oracle.summaries
        K = cfg.K

        acc = LatentMetrics.compute(
            [s.gamma for s in summaries],
            [ds.latents.get_S(k)[: train.get_X(k).shape[0]] for k in range(1, K + 1)],
            [s.m_H for s in summaries],
            [ds.latents.get_H(k)[: train.get_X(k).shape[0]] for k in range(1, K + 1)],
        ).acc_S_per_scale

        out = {}
        for k in range(K, 0, -1):
            p = ds.params[k - 1]
            q = summaries[k - 1]
            X, Y, C = train.get_X(k), train.get_Y(k), train.get_C(k)
            T = X.shape[0]
            cands = [
                (src, d)
                for src in range(cfg.L)
                for d in range(1, cfg.max_lag[k - 1] + 1)
            ]
            true_gains, null_gains = [], []
            for ell in range(cfg.L):
                true_set = set(p.parent_sets.get(ell, set()))
                keys = sorted(set(cands) | true_set)
                if k == K:
                    sc = _CoarseLabelScorer(ell, X, Y, q, C, p, 0.0, keys)
                else:
                    sc = _FineLabelScorer(
                        ell,
                        X,
                        Y,
                        train.get_X(k + 1),
                        train.get_Y(k + 1),
                        q,
                        C,
                        p,
                        train.hierarchy,
                        k,
                        0.0,
                        keys,
                    )
                base = sc.score(set())
                for e in keys:
                    gain = sc.score({e}) - base
                    (true_gains if e in true_set else null_gains).append(gain)
            out[k] = {
                "n_epochs": T,
                "density": round(float(np.mean(X > 0)), 3),
                "n_true_edges": len(true_gains),
                "true_gain": (
                    round(float(np.median(true_gains)), 2) if true_gains else None
                ),
                "null_gain95": round(float(np.percentile(null_gains, 95)), 2),
                "lam_bic": round(float(np.log(max(T, 2))), 2),
                "oracle_accS": round(float(acc[k - 1]), 3),
            }
        return out

    # ------------------------------------------------------------------
    @classmethod
    def sanity_check(cls) -> "ScenarioGrid":
        """
        Small 3-scale sanity-check grid, fast enough for the tests.

        One minute, ten minutes and one hour over a day: 1440 / 144 / 24 epochs.
        """
        scenarios = []
        seed = 0
        for L in (3, 6):
            for M in (1, 2):
                for r in (0, 1):
                    for family in (1, 2, 3):
                        if family == 2 and M == 1:
                            continue  # regime family needs M > 1
                        if family == 3 and r == 0:
                            continue  # factor family needs r > 0
                        cfg = ScenarioConfig(
                            name=f"sanity_L{L}_M{M}_r{r}_F{family}",
                            K=3,
                            H_finest=1440,
                            branching=[10, 6],
                            L=L,
                            M_k=[M, M, M],
                            r_k=[r, r, r],
                            max_lag=[2, 1, 1],
                            max_indegree=[0, 1, 0],
                            zi_level="medium",
                            sparse_graph=True,
                            family=family,
                            seed=seed,
                            n_replicates=2,
                            edge_strength=10.0,
                            regime_strength=3.0,
                            finest_rate=0.15,
                            slot_minutes=1,
                            finest_binary=True,
                        )
                        scenarios.append(cfg)
                        seed += 1
        return cls(scenarios=scenarios)

    @classmethod
    def full_benchmark(cls) -> "ScenarioGrid":
        """
        Evaluation grid of 22 scenarios on two clock-aligned hierarchies.

        Minute grid, 20 scenarios: one minute, ten minutes, one hour, over two
        weeks, so 20160 / 2016 / 336 epochs.  Event starts are resolved to the
        minute, which is the resolution of the reconstructed event log.

        Day grid, 2 scenarios: five minutes, one hour, one day, over 30 days,
        so 8640 / 720 / 30 epochs.  It repeats the study when the coarsest scale
        is a calendar day.

        Where the true edges sit
        ------------------------
        The ten-minute and hourly scales carry lagged edges; the minute scale
        does not.  That split is measured rather than assumed.  Screening all
        22 scenarios at a nonzero in-degree on every scale, the median gain of a
        true edge less the floor it must clear, `max(null_gain95, lam_bic)`, is
        -3.40, -9.46 and -6.27 nats at the minute scale for families I, II and
        III, and only 3 of 21 scenarios clear at all.  At the hourly scale the
        same margins are +7.90, +9.31 and +9.20 with 16 of 22 clearing, which is
        at least as good as the ten-minute scale that already carried the edges,
        +3.46, +3.22 and +8.75 with 17 of 22.

        The minute scale fails because it is a binary event-start process: a
        lagged edge there can only shift which minute of a block begins an
        episode, and that is worth less than the price of the edge.  This is a
        property of the generator and of the binary constraint rather than an
        oversight, so the minute scale stays empty and remains a precision test.
        A method that adds edges there is reporting structure the data does not
        contain, and no measurement on this grid can distinguish that from
        correctly finding a minute-scale edge, because none can be placed.

        Where the latent states sit
        ---------------------------
        The same question was asked of the latents and answered separately for
        each of the two hierarchies, by installing the generating parameters,
        recovering each path from the posterior alone and scoring it against the
        floor a method with no information reaches.  A regime is scored against
        the constant predictor and a factor against a shifted copy of its own
        trajectory.

        The regime clears at the coarsest scale in all 9 scenarios, by +0.468 to
        +0.600 in accuracy, and fails at the minute scale in all 9.  At the
        middle scale the two hierarchies differ and the grid follows them.  On
        the minute hierarchy 1 of 8 clears at the grid's own contrast and it does
        not survive raising that contrast, so the ten-minute scale carries no
        regime.  On the daily hierarchy the hour scale clears at +0.157, and at
        +0.119 and +0.276 when the contrast is raised to 8 and to 20, so it does.
        Its blocks hold 24 children against the minute hierarchy's 6, which is
        the information a regime has to act through once it no longer reaches the
        count emission.  Raising the contrast is not what decides any of this:
        over 3, 8 and 20 the two finer scales move from 0 and 2 of 9 clearing to
        1 and 2 of 9, while the coarsest clears 9 of 9 throughout.

        The factor clears at **every** scale of every scenario, by +0.34 to
        +0.78 in correlation, including the binary minute scale.  So the two
        latents are not interchangeable on this benchmark: same-slice
        co-movement survives the event-start reading and persistent drift does
        not, which is why `r_k` is uniform across scales and `M_k` is not.
        """
        scenarios = []
        seed = 0

        minute = dict(
            K=3,
            H_finest=20160,
            branching=[10, 6],
            slot_minutes=1,
            max_lag=[3, 2, 1],
            max_indegree=[0, 2, 1],
            sparse_graph=True,
            n_replicates=3,
            edge_strength=10.0,
            regime_strength=3.0,
            finest_rate=0.15,
            finest_binary=True,
        )

        # --- Family I: stationary, varies labels and zero inflation ---
        for L in (5, 10):
            for zi in ("low", "high"):
                scenarios.append(
                    ScenarioConfig(
                        name=f"F1_L{L}_zi{zi[0]}",
                        L=L,
                        M_k=[1, 1, 1],
                        r_k=[0, 0, 0],
                        zi_level=zi,
                        family=1,
                        seed=seed,
                        **minute,
                    )
                )
                seed += 1

        # --- Family II: regime-switching, varies regime count ---
        # The scale below the coarsest carries a regime only where its own screen
        # clears it, which is six of these nine.  A rule for the whole grid would
        # either place a regime the oracle cannot recover in three of them or
        # refuse one it can recover in six, and the screen is a per-scenario
        # instrument.  Measured at the coarse total and the scale-relative
        # persistence this grid runs, the margins are +0.203, +0.081, +0.051,
        # +0.086, +0.172 and +0.111 for the six that clear, against +0.045 and
        # +0.048 for two that fall just short and -0.022 for the day hierarchy.
        SUBCOARSE_REGIME = {
            "F2_L5_M2_zil",
            "F2_L5_M2_zih",
            "F2_L5_M4_zil",
            "F2_L5_M4_zih",
            "F2_L10_M4_zil",
            "F2_L10_M4_zih",
        }
        for L in (5, 10):
            for M in (2, 4):
                for zi in ("low", "high"):
                    name = f"F2_L{L}_M{M}_zi{zi[0]}"
                    scenarios.append(
                        ScenarioConfig(
                            name=name,
                            L=L,
                            M_k=[1, M if name in SUBCOARSE_REGIME else 1, M],
                            r_k=[0, 0, 0],
                            zi_level=zi,
                            family=2,
                            seed=seed,
                            **minute,
                        )
                    )
                    seed += 1

        # --- Family III: factor-driven, varies factor dimension ---
        for L in (5, 10):
            for r in (1, 2):
                for zi in ("low", "high"):
                    scenarios.append(
                        ScenarioConfig(
                            name=f"F3_L{L}_r{r}_zi{zi[0]}",
                            L=L,
                            M_k=[1, 1, 1],
                            r_k=[r, r, r],
                            zi_level=zi,
                            family=3,
                            seed=seed,
                            **minute,
                        )
                    )
                    seed += 1

        # --- Day grid: five minutes, one hour, one day, over 30 days ---
        day = dict(
            K=3,
            H_finest=8640,
            branching=[12, 24],
            slot_minutes=5,
            max_lag=[3, 2, 1],
            max_indegree=[0, 2, 1],
            sparse_graph=True,
            n_replicates=3,
            edge_strength=10.0,
            regime_strength=3.0,
            finest_rate=0.15,
            finest_binary=True,
        )
        scenarios.append(
            ScenarioConfig(
                name="F2_daily_L5_M2",
                L=5,
                # The hour scale of this hierarchy cleared under the pinned
                # coarse coefficient and does not under a drawn one, at either
                # total swept, so it loses the regime it used to carry.
                M_k=[1, 1, 2],
                r_k=[0, 0, 0],
                zi_level="medium",
                family=2,
                seed=seed,
                **day,
            )
        )
        seed += 1
        scenarios.append(
            ScenarioConfig(
                name="F3_daily_L5_r1",
                L=5,
                M_k=[1, 1, 1],
                r_k=[1, 1, 1],
                zi_level="medium",
                family=3,
                seed=seed,
                **day,
            )
        )

        return cls(scenarios=scenarios)
