"""
learning/fine_sem.py
====================
One fit call for the fine-likelihood formulation.

The pieces built before this one each answer a question and none of them fits a
model.  `intensity.py` states the likelihood, `bound.py` supplies the objective
and the two adjoint operators, `fine_mstep.py` maximizes over the coefficients
and `fine_sstep.py` searches the graph.  This module is the loop that alternates
them and the latent-state step that was missing between them.

The latent step is where the upward message earns its name.  A scale-k regime
contributes an additive term to the one fine predictor, so the sufficient
statistic for that regime is the segment sum of the per-slot contributions, and
`forward_backward` runs on it unchanged.  Nothing here pools a posterior by a
rule invented for the purpose.

Order is a parameter.  Each latent step maximizes the bound over one chain with
the others held, and each coefficient step maximizes it over one block, so the
sweep may run coarse to fine, fine to coarse or interleaved and the bound cannot
fall.  The structure step is the exception and carries its own guard, because a
graph move changes the predictor and therefore the anchor and the posterior the
gain was scored under.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..core.mask import scorable_rows
from ..core.typed_graph import COUNT, DURATION
from ..inference.bound import (
    chain_entropy,
    evidence_bound,
    exact_start_term,
    pg_mean,
    pg_update,
    segment_sum,
    upward_message,
)
from ..inference.posterior import RegimePosterior
from ..model.intensity import FineParams, FineSources, linear_predictor, usable_slots

_FLOOR = 1e-12


def fit_rows(
    observed: Optional[np.ndarray],
    budget: Optional[Mapping[int, int]],
    sources: FineSources,
) -> Optional[np.ndarray]:
    """
    The finest slots a fit may run on, which is the mask dilated by the budget.

    Every edge the search may propose reads its source scale stepped back by
    its lag, so the mask is eroded forward at each scale by the largest lag the
    budget allows there.  A slot that survives reads no feature from inside a
    gap.

    The budget belongs to the candidate space and never to the graph in hand.
    Eroding by the graph's own edges would shrink the row set as the search
    added reach, so an edge carrying no coefficient at all would raise the
    objective by the rows it removed and the search would buy it for that.

    Raises:
        ValueError: if the data carries a gap and no budget was declared.
    """
    if observed is None:
        return None
    if budget is None:
        raise ValueError(
            "a masked fine fit needs the search's declared lag budget, since "
            "eroding by the graph in hand would price an edge by the rows it "
            "removes"
        )
    return scorable_rows(observed, sources.ancestors, dict(budget))


def start_bound(
    x: np.ndarray,
    eta: np.ndarray,
    variance: np.ndarray,
    at_risk: Optional[np.ndarray] = None,
) -> float:
    """
    One label's quadratic start term, at the anchor where the quadratic is tightest.

    Written as one (T_1, L) expression the term holds eight arrays of that shape
    at once, which is 505 MB at a one-minute slot over 60 labels.  One label at a
    time and over the risk set alone, the same arithmetic runs in arrays a
    fraction of that size, and the sum over labels is the same number.

    Args:
        x: (T_1,) the label's counts, read as a start indicator.
        eta: (T_1,) the label's predictor.
        variance: (T_1,) the posterior spread of that predictor, or the scalar
            zero when the model carries no latent part.
        at_risk: (T_1,) bool, the rows the term sums over.
    """
    keep = slice(None) if at_risk is None else np.asarray(at_risk)
    e = np.asarray(eta, dtype=float)[keep]
    v = np.asarray(variance, dtype=float)
    v = v if v.ndim == 0 else v[keep]
    second = e * e + v
    anchor = np.sqrt(np.maximum(second, 0.0))
    weights = pg_mean(anchor)
    starts = (np.asarray(x)[keep] > 0).astype(float)
    const = -np.logaddexp(0.0, anchor) + anchor / 2.0 + weights * anchor * anchor / 2.0
    return float(np.sum((starts - 0.5) * e - 0.5 * weights * second + const))


def _prior_pairs(n_epochs: int, n_states: int, log_trans: np.ndarray) -> np.ndarray:
    """Pair marginals of a chain that is uniform at every epoch."""
    if n_epochs < 2:
        return np.zeros((0, n_states, n_states))
    pair = np.exp(log_trans).T / n_states
    return np.broadcast_to(pair, (n_epochs - 1, n_states, n_states)).copy()


@dataclass
class RegimeChain:
    """One scale's regime chain: its grid, its effects and its transitions."""

    scale: int
    n_states: int
    effect: np.ndarray  # (M, L) contribution to the start predictor
    log_init: np.ndarray  # (M,)
    log_trans: np.ndarray  # (M, M), log p(next | previous), tiled over epochs
    gamma: Optional[np.ndarray] = None  # (T_k, M)
    xi: Optional[np.ndarray] = None  # (T_k - 1, M, M)

    @classmethod
    def uniform(
        cls,
        scale: int,
        n_states: int,
        n_labels: int,
        n_epochs: int,
        persistence: float = 0.9,
    ):
        """
        A chain that starts undecided and persistent, which is a neutral start.

        The posterior starts uniform and never None.  A chain with no posterior
        would contribute nothing to the predictor, which anchors the quadratic
        as though the chain's effect were absent and biases the first latent step
        toward whichever state has the smallest effect.
        """
        stay = np.log(persistence)
        leave = np.log((1.0 - persistence) / max(n_states - 1, 1))
        trans = np.full((n_states, n_states), leave)
        np.fill_diagonal(trans, stay)
        return cls(
            scale=scale,
            n_states=n_states,
            effect=np.zeros((n_states, n_labels)),
            log_init=np.full(n_states, -np.log(n_states)),
            log_trans=trans,
            gamma=np.full((int(n_epochs), n_states), 1.0 / n_states),
            # The pair marginal of the prior chain itself, so the entropy at step
            # zero is the same quantity every later step reports.  Leaving it
            # unset would start the fit on the marginal entropy, which is larger
            # than the chain entropy whenever the chain is persistent, and the
            # first step would look like a fall that never happened.
            xi=_prior_pairs(int(n_epochs), n_states, trans),
        )

    def draw(self, n_epochs: int, rng: np.random.Generator) -> np.ndarray:
        """
        One state per epoch, drawn from this chain's own initial and transition laws.

        A fitted chain is part of the model, so a log drawn from that model
        carries a trajectory of its own and never the posterior of the record it
        was fitted on.
        """
        states = np.zeros(int(n_epochs), dtype=np.int64)
        if states.size == 0:
            return states
        init = np.exp(self.log_init - self.log_init.max())
        step = np.exp(self.log_trans - self.log_trans.max(axis=0, keepdims=True))
        # The chain is sequential, so the loop stays; the uniforms are drawn once
        # and each state is a search of its own column's cumulative law, which
        # is the same draw at a fraction of the per-epoch cost.
        cdf = np.cumsum(step / step.sum(axis=0, keepdims=True), axis=0)
        uniform = rng.random(states.size)
        first = np.cumsum(init / init.sum())
        states[0] = min(int(np.searchsorted(first, uniform[0])), self.n_states - 1)
        for t in range(1, states.size):
            states[t] = min(
                int(np.searchsorted(cdf[:, states[t - 1]], uniform[t])),
                self.n_states - 1,
            )
        return states

    def path_contribution(
        self, states: np.ndarray, ancestors: Sequence[np.ndarray]
    ) -> np.ndarray:
        """What one drawn state path adds to the start predictor at every slot."""
        index = np.asarray(ancestors[self.scale - 1])
        return self.effect[np.asarray(states, dtype=np.int64)][index]

    def moments(self, ancestors: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mean and variance of the chain's contribution at every finest slot.

        The variance is what makes the anchor honest: a slot whose state the
        posterior is unsure about carries a wider predictor, and the quadratic
        has to be anchored for that spread.
        """
        gamma = self.gamma
        mean = gamma @ self.effect
        second = gamma @ (self.effect**2)
        index = np.asarray(ancestors[self.scale - 1])
        return mean[index], np.maximum(second - mean**2, 0.0)[index]

    def contribution(self, ancestors: Sequence[np.ndarray]) -> np.ndarray:
        """What the chain adds to the predictor at every finest slot, (T_1, L)."""
        return self.moments(ancestors)[0]


def latent_step(
    chain: RegimeChain,
    x: np.ndarray,
    eta_rest: np.ndarray,
    weights: np.ndarray,
    ancestors: Sequence[np.ndarray],
    n_epochs: int,
    at_risk: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    One chain's posterior, from the segment sum of the per-slot contributions.

    The pseudo-emission is `upward_message`, which is the adjoint of the
    broadcast that carried the chain down into the predictor, so the upward and
    downward directions use one operator read two ways.
    """
    log_like = upward_message(
        x,
        eta_rest,
        chain.effect,
        weights,
        ancestors[chain.scale - 1],
        n_epochs,
        at_risk,
    )
    T = log_like.shape[0]
    log_trans = np.broadcast_to(
        chain.log_trans, (max(T - 1, 0),) + chain.log_trans.shape
    )
    return RegimePosterior.forward_backward(
        log_like, chain.log_init, np.array(log_trans)
    )


def update_transitions(chain: RegimeChain, damping: float = 0.0) -> RegimeChain:
    """
    Transition and initial distributions from the posterior's own counts.

    A homogeneous chain's maximizer is the normalized pair marginal summed over
    epochs, which is closed form and cannot lower the bound.
    """
    if chain.xi is None or chain.xi.size == 0:
        return chain
    counts = chain.xi.sum(axis=0) + damping
    rows = counts.sum(axis=1, keepdims=True)
    chain.log_trans = np.log(np.maximum(counts / np.maximum(rows, _FLOOR), _FLOOR)).T
    chain.log_init = np.log(np.maximum(chain.gamma[0], _FLOOR))
    return chain


@dataclass
class FitHistory:
    """The bound after every step, and what each step was."""

    values: List[float] = field(default_factory=list)
    steps: List[str] = field(default_factory=list)
    rejected: List[str] = field(default_factory=list)

    def record(self, name: str, value: float) -> None:
        self.steps.append(name)
        self.values.append(value)

    @property
    def monotone(self) -> bool:
        return all(b >= a - 1e-6 for a, b in zip(self.values, self.values[1:]))


class FineSEM:
    """
    The latent-state fit: every chain's posterior and transitions, on one bound.

    The coefficient step and the structure step compose against this same
    objective through `fine_mstep` and `fine_sstep`, and they stay outside the
    loop until a design builder ties a graph to a set of columns.  What this
    class owns is the schedule and the objective.

    Attributes
    ----------
    max_sweeps : how many times to visit every chain and block
    tol        : relative change below which the fit stops
    schedule   : "coarse_to_fine", "fine_to_coarse" or a list of scale indices,
                 which changes nothing about the guarantee and is exposed
                 because the order is free
    """

    def __init__(
        self,
        max_sweeps: int = 12,
        tol: float = 1e-5,
        schedule: str = "coarse_to_fine",
        lag_budget: Optional[Mapping[int, int]] = None,
    ) -> None:
        self.max_sweeps = int(max_sweeps)
        self.tol = float(tol)
        self.schedule = schedule
        # The largest lag the search may propose at each source scale, which
        # fixes the row set for every model this fit compares.
        self.lag_budget = None if lag_budget is None else dict(lag_budget)

    # ------------------------------------------------------------------

    def _order(self, chains: Sequence[RegimeChain]) -> List[int]:
        if self.schedule == "coarse_to_fine":
            return sorted(range(len(chains)), key=lambda i: -chains[i].scale)
        if self.schedule == "fine_to_coarse":
            return sorted(range(len(chains)), key=lambda i: chains[i].scale)
        return list(self.schedule)

    def objective(
        self,
        X1: np.ndarray,
        Y1: np.ndarray,
        params: FineParams,
        graph,
        sources: FineSources,
        chains: Sequence[RegimeChain],
        penalty: float = 0.0,
        observed: Optional[np.ndarray] = None,
        eta: Optional[np.ndarray] = None,
    ) -> float:
        """
        The one bound, with every part named.

        Args:
            eta: the count predictor, when the caller already holds it.  A
                pooled pass does, and its array carries the mean the design's
                centering removed, which a predictor formed here would not.
        """
        from ..model.intensity import mark_loglik

        if eta is None:
            eta = linear_predictor(
                params, graph, sources, COUNT, self._offset(chains, sources)
            )
        variance = self._spread(chains, sources)
        rows = fit_rows(observed, self.lag_budget, sources)
        at_risk = usable_slots(X1, Y1, rows)

        # The anchor sits at the predictor's own second moment, where the
        # quadratic is tightest, so the value below is the bound at its best
        # anchor and is comparable across steps.
        flat = variance.ndim == 0
        start = sum(
            start_bound(
                X1[:, ell],
                eta[:, ell],
                variance if flat else variance[:, ell],
                at_risk[:, ell],
            )
            for ell in range(eta.shape[1])
        )
        del variance

        log_nu = linear_predictor(params, graph, sources, DURATION)
        entropy = sum(chain_entropy(c.gamma, c.xi) for c in chains)
        return evidence_bound(
            start,
            mark_loglik(X1, Y1, log_nu, params.mark_dispersion, rows),
            entropy,
            penalty,
        )

    @staticmethod
    def _offset(
        chains: Sequence[RegimeChain], sources: FineSources
    ) -> Optional[np.ndarray]:
        """
        What every chain adds to the predictor, or None when no chain runs.

        `linear_predictor` reads None as no offset at all, so a model with no
        latent part costs no (T_1, L) array of zeros, which is 58 MB at a
        one-minute slot over 60 labels.
        """
        if not chains:
            return None
        total = np.zeros((sources.n_slots, sources.n_labels))
        for chain in chains:
            total = total + chain.contribution(sources.ancestors)
        return total

    @staticmethod
    def _spread(chains: Sequence[RegimeChain], sources: FineSources) -> np.ndarray:
        """
        The posterior spread of the predictor, or the scalar zero where there is none.

        A model carrying no chain has no spread, and the scalar says so in one
        number where an array of it costs what the offset's would.
        """
        if not chains:
            return np.zeros(())
        total = np.zeros((sources.n_slots, sources.n_labels))
        for chain in chains:
            total = total + chain.moments(sources.ancestors)[1]
        return total

    # ------------------------------------------------------------------

    def fit(
        self,
        X1: np.ndarray,
        Y1: np.ndarray,
        params: FineParams,
        graph,
        sources: FineSources,
        chains: Sequence[RegimeChain],
        n_epochs: Sequence[int],
        penalty: float = 0.0,
        observed: Optional[np.ndarray] = None,
    ) -> Tuple[FineParams, List[RegimeChain], FitHistory]:
        """
        Alternate the latent step and the transition step until the bound settles.

        The coefficient step is left to the caller through `fine_mstep`, so this
        loop stays the one place the schedule lives and the objective is formed.
        """
        chains = list(chains)
        at_risk = usable_slots(X1, Y1, fit_rows(observed, self.lag_budget, sources))
        history = FitHistory()
        history.record(
            "start",
            self.objective(X1, Y1, params, graph, sources, chains, penalty, observed),
        )

        for _ in range(self.max_sweeps):
            before = history.values[-1]
            for index in self._order(chains):
                chain = chains[index]
                others = [c for j, c in enumerate(chains) if j != index]
                eta_rest = linear_predictor(
                    params, graph, sources, COUNT, self._offset(others, sources)
                )
                # The anchor carries this chain's own mean and the spread of
                # every chain, so the quadratic is centered where the predictor
                # actually sits.  Anchoring at `eta_rest` alone drops the chain's
                # effect and ranks the states by their quadratic penalty.
                mean, _ = chain.moments(sources.ancestors)
                weights = pg_update(eta_rest + mean, self._spread(chains, sources))

                gamma, xi, _ = latent_step(
                    chain,
                    X1,
                    eta_rest,
                    weights,
                    sources.ancestors,
                    n_epochs[chain.scale - 1],
                    at_risk,
                )
                chain.gamma, chain.xi = gamma, xi
                chains[index] = update_transitions(chain)
                history.record(
                    f"latent k={chain.scale}",
                    self.objective(
                        X1, Y1, params, graph, sources, chains, penalty, observed
                    ),
                )

            after = history.values[-1]
            if abs(after - before) < self.tol * (1.0 + abs(before)):
                break
        return params, chains, history
