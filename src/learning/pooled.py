"""
learning/pooled.py
==================
One graph over one label axis, fitted on the sum of the cohort's records.

The objective sums per-subject log-likelihood terms, because the log of a
product of independent records is a sum.  That makes the fit a stream: per
subject the latent sweep runs at the transitions the cohort shares, that
subject's design is formed, and only quantities that are sums over rows are
added into shared accumulators.  One subject is opened inside `_one_subject`
and everything derived from it dies with that frame, so the reader never builds
one subject's finest arrays while the previous subject's are still reachable.

What adds is named here and nowhere else: the `A` and `b` of the normal
equations, which `FineDesign.search_gram` and `search_target` form, the cross,
diagonal and response arrays `candidate_gains` reads, the pair counts
`chain.xi.sum(axis=0)`, the first-epoch mass `chain.gamma[0]`, each target's own
start term and the bound each subject's record contributes.

What cannot add is named just as precisely.  `BaseFit.inverse` and `BaseFit.value`
are nonlinear in the matrix, so they are formed once from the summed one.  The
ridge is charged once, since it regularizes one graph and a per-subject ridge
would grow with the roster and shrink a coefficient the cohort has more evidence
for.  The edge price is charged once for the same reason: the cohort holds one
graph.  `update_block`, `solve_coefficients`, `pg_mean`, `pg_update` and
`anchor_from` are all nonlinear in the sum and run at the pooled level alone.

Which curvature each step reads
-------------------------------
The latent visit reads the Polya-Gamma bound, which is what makes a scale's
pseudo-emission a segment sum, and the objective every guard weighs is that
bound.  The coefficient step and the candidate score read the exact curvature of
the start term, `bound.start_curvature`, against the Fisher-scoring response
`bound.fisher_response`.  The two curvatures agree at a rate of one half and
part company as the rate leaves it: at walking's 1.45e-3 the bound carries 52.5
times the term's own curvature, which is one factor of 52.5 out of every gain
and a contraction of 1 - 1/52.5 per coefficient round, so the same fit takes
1248 bound rounds and ten Fisher rounds to reach the same intercept.  The solve
is the same solve either way; the weights and the right-hand side handed to it
decide which step it takes.

What the price is
-----------------
A price answers how large a gain a candidate reaches when there is no relation
there at all, so `null_gains` measures that distribution and a caller reads its
price off a stated quantile.  Each candidate's own column is rolled, which keeps
its marginal distribution and its serial dependence and destroys only what it
lines up with, and the rolled column is summed over the same roster the real one
is.  A price read that way moves with the roster exactly as a gain does, which
is the behavior the arm exists to measure: S records of one person legitimately
carry S times the evidence, and a price that refused to move with them would
suppress the information distinct participants bring.

What the predictor holds
------------------------
Every edge column is centered over the record, so a coefficient fitted by the
solve stands against a centered column while `linear_predictor` reads the
uncentered feature.  `_predictor` adds the difference back, which is the mean
`FineDesign.offset` names, so the weights the next round forms and the bound it
records are both read at the predictor the solve fitted.

What is resident
----------------
Per subject: that subject's finest counts and durations, 116 MB at a one-minute
slot over 60 labels; the predictor and the quadratic's weights, 58 MB each; the
risk set, 7 MB; and the accumulators, which are O(P^2) per target and never
O(T).  The centering correction is held on the coarse grid it varies at, a few
thousand numbers, and the start term is summed one label at a time, since the
whole (T_1, L) expression holds eight arrays of that shape at once.

The declared lag budget is one per-source-scale dict, fixed before the first
subject is opened and handed unchanged to every `fit_rows` call.  One number
covers the cohort because the budget describes the candidate space, and one
graph over one shared label axis is one candidate space.  A per-subject budget
would let two models compared on the pooled sum run over different row totals,
which is the row-set rule failing at cohort scale.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..core.cohort import Cohort
from ..core.typed_graph import COUNT, TypedEdge, TypedGraph
from ..inference.bound import fisher_response, pg_update, start_curvature
from ..model.intensity import FineParams, linear_predictor, usable_slots
from .fine_design import (
    CalendarIndex,
    DesignColumn,
    FineDesign,
    ScaleLadder,
    build_design,
    center_at_scale,
    edge_columns,
    risk_multiplier,
    scale_response,
    scale_weights,
)
from .fine_mstep import _RIDGE, regime_block, solve_informed, stack_design
from .fine_sem import (
    FineSEM,
    FitHistory,
    RegimeChain,
    fit_rows,
    latent_step,
    start_bound,
)
from .fine_sstep import (
    BaseFit,
    SearchReport,
    candidate_gains,
    cross_and_diagonal,
    greedy_search,
    shifted_null,
)
from .relation_price import GroupReport, RelationPrice, grouped_search

_FLOOR = 1e-12


# ---------------------------------------------------------------------------
# The accumulators, which hold exactly the quantities that are sums over rows
# ---------------------------------------------------------------------------


@dataclass
class PooledDesign:
    """
    One target's summed solve matrices and summed search matrices.

    `A` carries no ridge.  The ridge belongs to the pooled solve and is added
    there once, so the penalty describes the graph and not the roster size.
    """

    edges: Tuple[TypedEdge, ...]
    A: np.ndarray
    b: np.ndarray

    def add(self, other: "PooledDesign") -> None:
        """
        Add another subject's contribution.

        Raises:
            ValueError: when the two were formed under different parent sets,
                since the summed matrix would then hold one subject's column in
                the position of another's.
        """
        if other.edges != self.edges:
            raise ValueError(
                f"a design over {len(other.edges)} edges cannot be added to one "
                f"over {len(self.edges)}; the cohort holds one graph and its "
                "columns sit in one order"
            )
        self.A += other.A
        self.b += other.b


@dataclass
class PooledCandidates:
    """The three arrays `candidate_gains` reads, summed over the cohort."""

    cross: np.ndarray
    diagonal: np.ndarray
    target: np.ndarray

    def add(self, other: "PooledCandidates") -> None:
        self.cross += other.cross
        self.diagonal += other.diagonal
        self.target += other.target


@dataclass
class PooledChain:
    """
    One chain's summed pair counts and summed first-epoch mass.

    These are the two statistics the homogeneous chain's maximizer is a function
    of, so summing them over subjects and normalizing once gives the transition
    parameter the whole cohort shares.
    """

    pairs: np.ndarray
    first: np.ndarray

    def add(self, other: "PooledChain") -> None:
        self.pairs += other.pairs
        self.first += other.first


@dataclass(frozen=True)
class TargetParams:
    """
    One target's coefficients, held apart from the ones the fit is carrying.

    A refused move is put back from one of these and a guard weighs the move
    against one of these, so the same three numbers describe both: the
    intercept, the coefficient of every edge and one effect column per chain.
    """

    base: float
    coef: Dict[TypedEdge, float]
    effect: List[np.ndarray]


@dataclass
class PooledSweep:
    """
    What one pass over the cohort accumulated.

    `starts` is each target's own start term.  A structure move changes the
    predictor of the target it lands on and of no other, so the number a guard
    compares before and after is that target's term and never the whole bound,
    which also carries every other target's.  `against` is the same term read at
    a second coefficient set in the same pass, which is what lets the guard
    weigh a move against the fit without it and not against the fit before it.
    """

    designs: Dict[int, PooledDesign]
    candidates: Dict[int, PooledCandidates]
    chains: List[PooledChain]
    starts: Dict[int, float]
    against: Dict[int, float]
    bound: float
    subjects: int
    rows: int

    def add(self, other: "PooledSweep") -> None:
        """Add one more subject's pass, statistic by statistic."""
        for target, design in other.designs.items():
            if target in self.designs:
                self.designs[target].add(design)
            else:
                self.designs[target] = design
        for target, seen in other.candidates.items():
            if target in self.candidates:
                self.candidates[target].add(seen)
            else:
                self.candidates[target] = seen
        for target, value in other.starts.items():
            self.starts[target] = self.starts.get(target, 0.0) + value
        for target, value in other.against.items():
            self.against[target] = self.against.get(target, 0.0) + value
        if not self.chains:
            self.chains = other.chains
        else:
            for total, one in zip(self.chains, other.chains):
                total.add(one)
        self.bound += other.bound
        self.subjects += other.subjects
        self.rows += other.rows


def _candidate_moments(
    design: FineDesign,
    columns: Sequence[DesignColumn],
    response: np.ndarray,
    weights: np.ndarray,
    at_risk: np.ndarray,
) -> PooledCandidates:
    """
    Each candidate against the base columns, against itself and against the response.

    Formed the way `FineDesign.search_gram` forms its own entries, so a gain
    scored off the sum of these is a gain the pooled solve realizes.  The base
    order is the intercept, then the edge columns, then the latent blocks, which
    is the order the summed Gram already holds.

    The full Gram of base and candidates together would carry a
    candidate-by-candidate block of C (C - 1) / 2 entries, which is 778128 at
    the benchmark's 1248 candidates per target and which the rank-one gain never
    opens.  Only the cross block and the diagonal are formed.
    """
    ladder = design.ladder
    base = design.scaled()
    cross, diagonal = cross_and_diagonal(
        [c.as_scaled() for c in columns],
        base,
        ladder.ancestors,
        scale_weights(weights, ladder, at_risk),
    )
    summed = scale_response(response, ladder, at_risk)
    target = np.array([summed[c.scale - 1] @ c.values for c in columns])
    if design.latent:
        w = np.asarray(weights, dtype=float) * risk_multiplier(at_risk, ladder.n_slots)
        means, _ = stack_design(design.latent)
        fine = np.stack([c.at_finest(ladder) for c in columns], axis=1)
        cross = np.concatenate([cross, fine.T @ (means * w[:, None])], axis=1)
    return PooledCandidates(cross=cross, diagonal=diagonal, target=target)


def _shifted(
    column: DesignColumn,
    ladder: ScaleLadder,
    budget: Mapping[int, int],
    seed: int,
) -> DesignColumn:
    """
    One candidate's own column with its alignment to the response destroyed.

    `fine_sstep.shifted_null` supplies the roll, which keeps the column's
    marginal distribution and its serial dependence.  The roll is taken on the
    uncentered values and the parent mean is removed again afterwards, so a null
    column carries the one centering convention every real candidate carries and
    the two are comparable.  The seed is the candidate's own position in the
    pool, so the same roll reaches every subject and the summed moments belong to
    one null candidate.  A seed drawn per subject would sum the moments of a
    different column each time.
    """
    raw = column.values + column.removed
    rolled = shifted_null(raw, np.random.default_rng(seed), 1)[0]
    scale = column.scale
    values = center_at_scale(rolled, ladder, scale, budget.get(scale, column.edge.lag))
    return replace(column, values=values, removed=rolled - values)


def _alternative(design: FineDesign, held: TargetParams) -> np.ndarray:
    """
    One target's predictor at a coefficient set the fit is not currently at.

    Read off the design in hand, so an edge the design carries and the held set
    says nothing about enters at zero, which is how a move is weighed against
    the same solve with the proposed column's coefficient dropped.
    """
    beta = [held.base] + [held.coef.get(c.edge, 0.0) for c in design.columns]
    for column in held.effect:
        beta.extend(np.asarray(column, dtype=float)[1:])
    return design.predictor(np.asarray(beta, dtype=float))


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------


class PooledFit:
    """
    The stream-and-accumulate fit of one graph over a cohort.

    The graph, the coefficients, the chain effects and the chain transitions are
    the cohort's and are held here.  A latent trajectory is one subject's own, so
    a posterior is held per subject, at the scale its chain runs on; a day-scale
    chain over 91 epochs costs the roster almost nothing while the finest counts
    and durations stream.

    Attributes
    ----------
    graph      : the one typed graph every subject is scored under
    params     : the coefficients the pooled solve writes into
    chains     : the shared chain parameters, one per scale carrying a regime
    lag_budget : the largest lag the search may propose at each source scale,
                 declared before the first subject is opened
    targets    : the target labels a design is formed for
    candidates : the edges the structure round may propose, per target
    """

    def __init__(
        self,
        graph: TypedGraph,
        params: FineParams,
        chains: Sequence[RegimeChain],
        lag_budget: Mapping[int, int],
        targets: Optional[Sequence[int]] = None,
        quantity: str = COUNT,
        candidates: Optional[Mapping[int, Sequence[TypedEdge]]] = None,
        index: Optional[CalendarIndex] = None,
        max_sweeps: int = 12,
        tol: float = 1e-5,
        schedule: str = "coarse_to_fine",
        damping: float = 0.0,
        move_rounds: int = 1,
        max_refused: int = 1,
    ) -> None:
        self.graph = graph
        self.params = params
        self.chains = list(chains)
        # A cohort always carries a mask, so it always carries a declared budget.
        # Fixed here, before the first subject is opened, and handed unchanged to
        # every row set the fit and the search are taken over.
        self.lag_budget = dict(lag_budget)
        self.targets = list(range(graph.L) if targets is None else targets)
        self.quantity = quantity
        self.candidates = {
            int(t): list(edges) for t, edges in dict(candidates or {}).items()
        }
        self.index = index if index is not None else CalendarIndex(placed={}, widths={})
        self.damping = float(damping)
        # How many coefficient rounds a proposed move takes before the guard
        # reads its objective.  One Newton step overshoots on a column that
        # nearly separates the response, and the guard then refuses a move that
        # is worth taking two steps later.
        self.move_rounds = max(int(move_rounds), 1)
        # Refusals a grouped round takes before it stops.  A refused move costs
        # the passes it took and buys nothing, and the candidate ranked behind
        # the refused one is untouched by the refusal.
        self.max_refused = max(int(max_refused), 1)
        # The schedule, the tolerance and the one objective all stay where they
        # already live, so a pooled sweep and a single-subject sweep are the same
        # sweep read over a longer sum.
        self.sem = FineSEM(max_sweeps, tol, schedule, self.lag_budget)
        self._posteriors: Dict[str, List[RegimeChain]] = {}
        # Columns the last solve held at zero because they carried no curvature,
        # per target.  A run that reports a fit over a refused column is
        # reporting a coefficient no data in the cohort speaks to.
        self.refused: Dict[int, List[str]] = {}

    # ------------------------------------------------------------------
    def _chains_for(self, subject: str) -> List[RegimeChain]:
        """
        One subject's chains, carrying its own posterior and the cohort's parameters.

        The trajectory is the subject's and survives between sweeps.  The effect,
        the initial distribution and the transitions are the cohort's and are
        refreshed here, so a subject's E-step never runs at a parameter the
        pooled maximizer has already replaced.
        """
        held = self._posteriors.get(subject)
        if held is None:
            held = copy.deepcopy(self.chains)
            self._posteriors[subject] = held
        for shared, own in zip(self.chains, held):
            own.effect = shared.effect
            own.log_init = shared.log_init
            own.log_trans = shared.log_trans
        return held

    def _latent(
        self,
        X1: np.ndarray,
        sources,
        chains: List[RegimeChain],
        at_risk: np.ndarray,
        n_epochs: Sequence[int],
    ) -> None:
        """
        One visit to every chain of one subject, at the transitions the cohort shares.

        A subject that re-fitted its own transitions would hand back statistics
        taken under a parameter no other subject used, and the pooled maximizer
        would then be a sum of counts from different models.
        """
        for which in self.sem._order(chains):
            chain = chains[which]
            others = [c for j, c in enumerate(chains) if j != which]
            eta_rest = linear_predictor(
                self.params,
                self.graph,
                sources,
                COUNT,
                FineSEM._offset(others, sources),
            )
            mean, _ = chain.moments(sources.ancestors)
            weights = pg_update(eta_rest + mean, FineSEM._spread(chains, sources))
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

    # ------------------------------------------------------------------
    def _centering(self, sources, ladder: ScaleLadder) -> Dict[int, np.ndarray]:
        """
        Per source scale, the mean centering took out of each target's own columns.

        Held on the coarse grid it varies at, so the whole correction over 60
        labels is a few thousand numbers where the same statement written at the
        finest grid is another 58 MB.
        """
        out: Dict[int, np.ndarray] = {}
        for ell in range(self.graph.L):
            parents = sorted(self.graph.get_parents(ell, self.quantity))
            if not parents:
                continue
            coef = self.params.coef[(ell, self.quantity)]
            columns = edge_columns(
                parents, self.graph, sources, ladder, self.index, self.lag_budget
            )
            for column in columns:
                weight = float(coef.get(column.edge, 0.0))
                if not weight:
                    continue
                if column.scale not in out:
                    out[column.scale] = np.zeros(
                        (ladder.n_epochs[column.scale - 1], self.graph.L)
                    )
                out[column.scale][:, ell] -= weight * column.removed
        return out

    def _predictor(
        self, sources, ladder: ScaleLadder, chains: Sequence[RegimeChain]
    ) -> np.ndarray:
        """
        The count predictor the design fits, at every finest slot.

        `linear_predictor` reads the uncentered feature of every edge while the
        solve fits a column the parent mean was taken out of, so the two differ
        by that mean.  Adding it back here is what puts the weights of the next
        round and the bound of this one at the predictor the solve maximized
        over.
        """
        eta = linear_predictor(
            self.params,
            self.graph,
            sources,
            COUNT,
            FineSEM._offset(chains, sources),
        )
        for scale, block in self._centering(sources, ladder).items():
            who = ladder.ancestors[scale - 1]
            # One label at a time, so broadcasting the coarse correction costs a
            # column and never a second (T_1, L) array.
            for ell in range(block.shape[1]):
                eta[:, ell] += block[who, ell]
        return eta

    # ------------------------------------------------------------------
    def accumulate(
        self,
        cohort: Cohort,
        step: bool = True,
        targets: Optional[Iterable[int]] = None,
        null_seed: Optional[int] = None,
        against: Optional[Tuple[int, TargetParams]] = None,
    ) -> PooledSweep:
        """
        One stream over the cohort, adding every statistic that is a sum over rows.

        Args:
            cohort: the roster and the reader the subjects arrive through.
            step: run the latent visit before reading each subject's statistics.
                False reads the cohort at the posteriors it already holds, which
                is what scoring a structure move needs.
            targets: the target labels to form a design for, the whole set by
                default.
            null_seed: roll every candidate column before its moments are taken,
                which is the null a price is calibrated against.
            against: one target and a coefficient set to read its start term at
                as well, so a guard can weigh a move without a second pass.

        Returns:
            The summed matrices, the summed chain counts, each target's summed
            start term, the summed bound and the rows it was summed over.
        """
        wanted = list(self.targets if targets is None else targets)
        n_epochs = list(cohort.hierarchy.H)
        total = PooledSweep(
            designs={},
            candidates={},
            chains=[],
            starts={},
            against={},
            bound=0.0,
            subjects=0,
            rows=0,
        )
        for name in cohort.subjects:
            total.add(
                self._one_subject(
                    cohort, name, wanted, step, n_epochs, null_seed, against
                )
            )
        return total

    def _one_subject(
        self,
        cohort: Cohort,
        name: str,
        wanted: Sequence[int],
        step: bool,
        n_epochs: Sequence[int],
        null_seed: Optional[int],
        against: Optional[Tuple[int, TargetParams]] = None,
    ) -> PooledSweep:
        """
        One subject's contribution, opened and dropped inside this call.

        Every array derived from the subject is a local of this frame, so the
        moment it returns nothing of that subject is reachable and the reader
        opens the next one against the accumulators alone.  A caller that held
        the subject in a loop variable would keep the previous record alive while
        the next was built, which doubles the peak for a benefit of nothing.
        """
        data = cohort.open(name)
        X1, Y1 = data.get_X(1), data.get_Y(1)
        sources = cohort.sources(data)
        chains = self._chains_for(name)
        at_risk = usable_slots(
            X1, Y1, fit_rows(data.observed, self.lag_budget, sources)
        )
        if step:
            self._latent(X1, sources, chains, at_risk, n_epochs)

        ladder = ScaleLadder.from_sources(sources, data.observed)
        eta = self._predictor(sources, ladder, chains)
        spread = FineSEM._spread(chains, sources)
        flat = spread.ndim == 0

        # The price is charged once against the cohort, so every subject
        # contributes its bound at a price of zero.
        bound = self.sem.objective(
            X1, Y1, self.params, self.graph, sources, chains, 0.0, data.observed, eta
        )
        latent = [
            regime_block(c.gamma, sources.ancestors[c.scale - 1], f"regime k{c.scale}")
            for c in chains
        ]
        designs: Dict[int, PooledDesign] = {}
        candidates: Dict[int, PooledCandidates] = {}
        starts: Dict[int, float] = {}
        other: Dict[int, float] = {}
        for ell in wanted:
            design = build_design(
                self.graph,
                ell,
                self.quantity,
                sources,
                ladder,
                self.index,
                latent,
                self.lag_budget,
            )
            # The step's weights are formed for the one label being read, since
            # every target reads its own column once and the (T_1, L) array of
            # them is 58 MB that nothing else looks at.
            variance = spread if flat else spread[:, ell]
            weights = start_curvature(eta[:, ell])
            response = fisher_response(X1[:, ell], eta[:, ell])
            designs[ell] = self._target_stats(
                design, response, weights, at_risk[:, ell]
            )
            starts[ell] = start_bound(
                X1[:, ell], eta[:, ell], variance, at_risk[:, ell]
            )
            if against is not None and against[0] == ell:
                other[ell] = start_bound(
                    X1[:, ell],
                    _alternative(design, against[1]),
                    variance,
                    at_risk[:, ell],
                )
            pool = self.candidates.get(ell)
            if not pool:
                continue
            columns = edge_columns(
                pool, self.graph, sources, ladder, self.index, self.lag_budget
            )
            if null_seed is not None:
                columns = [
                    _shifted(c, ladder, self.lag_budget, null_seed + j)
                    for j, c in enumerate(columns)
                ]
            candidates[ell] = _candidate_moments(
                design, columns, response, weights, at_risk[:, ell]
            )
        return PooledSweep(
            designs=designs,
            candidates=candidates,
            chains=self._chain_counts(chains),
            starts=starts,
            against=other,
            bound=bound,
            subjects=1,
            rows=int(at_risk.sum()),
        )

    @staticmethod
    def _chain_counts(chains: Sequence[RegimeChain]) -> List[PooledChain]:
        """One subject's pair counts and first-epoch mass, one entry per chain."""
        return [
            PooledChain(pairs=c.xi.sum(axis=0), first=c.gamma[0].copy()) for c in chains
        ]

    def _target_stats(
        self,
        design: FineDesign,
        response: np.ndarray,
        weights: np.ndarray,
        at_risk: np.ndarray,
    ) -> PooledDesign:
        """
        One subject's contribution to one target, as the two summable matrices.

        `search_gram` and `search_target` are the same `A` and `b`
        `fine_mstep.normal_equations` forms, reduced at each entry's own
        resolution, so the solve and the search read one pair of matrices and a
        base the search scores against is the base the solve inverted.
        """
        return PooledDesign(
            edges=tuple(c.edge for c in design.columns),
            A=design.search_gram(weights, at_risk),
            b=design.search_target(response, at_risk),
        )

    # ------------------------------------------------------------------
    def set_transitions(self, stats: PooledSweep) -> None:
        """
        The shared transition and initial distributions, from the summed counts.

        A homogeneous chain's maximizer is the normalized pair marginal summed
        over epochs, and independent records add their epochs, so the pooled
        maximizer is the same expression over the summed counts.  The initial
        distribution is a mean over subjects, since every subject contributes one
        first epoch.
        """
        for shared, counted in zip(self.chains, stats.chains):
            # A chain with no transition to count has no maximizer, which is the
            # case `update_transitions` leaves alone.
            if counted.pairs.sum() <= 0.0:
                continue
            counts = counted.pairs + self.damping
            totals = counts.sum(axis=1, keepdims=True)
            shared.log_trans = np.log(
                np.maximum(counts / np.maximum(totals, _FLOOR), _FLOOR)
            ).T
            shared.log_init = np.log(np.maximum(counted.first / stats.subjects, _FLOOR))

    def _write(self, target: int, edges: Sequence[TypedEdge], beta: np.ndarray) -> None:
        """
        One target's solved vector, into the intercept, the edges and the chains.

        The stacked design is the intercept, then the edge columns, then one
        block per chain, and a chain's block is its posterior with the first
        state left out as the reference.  Storing the first entries and dropping
        the rest would leave `RegimeChain.effect` at the zeros it was built with
        while the class holds it out as the fitted chain effect.
        """
        beta = np.asarray(beta, dtype=float)
        self.params.base[self.quantity][target] = float(beta[0])
        for j, edge in enumerate(edges):
            self.params.coef[(target, self.quantity)][edge] = float(beta[j + 1])
        start = 1 + len(edges)
        for chain in self.chains:
            width = chain.n_states - 1
            chain.effect[0, target] = 0.0
            chain.effect[1:, target] = beta[start : start + width]
            start += width

    def solve(self, stats: PooledSweep) -> Dict[int, np.ndarray]:
        """
        One coefficient vector per target, from the summed normal equations.

        The weights are the exact curvature of the start term and the response
        is the Fisher-scoring one, so the solve is a Newton step on the
        likelihood and the fixed point it walks to is the maximizer.  Ten of
        these reach the intercept 1248 rounds of the quadratic bound reach.

        The ridge is added here and only here.  Charged per subject it would grow
        with the roster and shrink the very coefficients a larger cohort carries
        more evidence for.

        A column with no curvature behind it is held at zero rather than solved.
        `A` is weighted and `b` is not, so once the curvature of a target
        underflows the diagonal reaches exactly zero while `b` does not, and the
        ridge alone would return the coefficient as `b * 1e8`.  A target whose
        remaining block is still ill-conditioned is not written at all and keeps
        the coefficients it came in with.  Which columns were refused is kept in
        `refused`, because a silent zero and a fitted zero are not the same
        claim.
        """
        out: Dict[int, np.ndarray] = {}
        self.refused = {}
        for target in sorted(stats.designs):
            design = stats.designs[target]
            beta, informed, conditioned = solve_informed(design.A, design.b, _RIDGE)
            names = self._column_names(design.edges)
            held = [names[i] for i in np.flatnonzero(~informed)]
            if not conditioned:
                self.refused[target] = held + ["<solve refused: ill-conditioned>"]
                continue
            if held:
                self.refused[target] = held
            self._write(target, design.edges, beta)
            out[target] = beta
        return out

    def _column_names(self, edges: Sequence[TypedEdge]) -> List[str]:
        """The stacked design's columns, named in the order `_write` reads them."""
        names = ["intercept"]
        names.extend(str(edge) for edge in edges)
        for j, chain in enumerate(self.chains):
            names.extend(f"chain{j}:state{s}" for s in range(1, chain.n_states))
        return names

    def fit(
        self, cohort: Cohort, penalty: float = 0.0
    ) -> Tuple[PooledSweep, FitHistory]:
        """
        Alternate the pooled latent sweep and the shared transition update.

        One sweep is one visit to every chain of every subject, so the schedule
        and the stopping rule are the single-subject ones read over a longer sum.
        The price enters once, because the cohort holds one graph.
        """
        history = FitHistory()
        stats = self.accumulate(cohort, step=False)
        history.record("start", stats.bound - penalty)
        for _ in range(self.sem.max_sweeps):
            before = history.values[-1]
            stats = self.accumulate(cohort, step=True)
            self.set_transitions(stats)
            history.record("latent", stats.bound - penalty)
            if abs(history.values[-1] - before) < self.sem.tol * (1.0 + abs(before)):
                break
        return stats, history

    # ------------------------------------------------------------------
    def null_gains(self, cohort: Cohort, seed: int = 0) -> np.ndarray:
        """
        The gain every candidate reaches once its alignment with its target is gone.

        One pass over the cohort with each candidate's own column rolled, scored
        against the same base set and summed over the same roster the real gains
        are.  What comes back is what a candidate of this shape is worth against
        a target it has no relation to, pooled over the targets a price has to
        serve, since the price is one number for the whole search.

        Args:
            seed: the first roll's seed; candidate j is rolled at `seed + j`, so
                one draw of the null is one integer and is reproducible.
        """
        stats = self.accumulate(cohort, step=False, null_seed=int(seed))
        if not stats.candidates:
            return np.zeros(0)
        return np.concatenate(
            [
                candidate_gains(
                    BaseFit(stats.designs[t].A, stats.designs[t].b),
                    stats.candidates[t].cross,
                    stats.candidates[t].diagonal,
                    stats.candidates[t].target,
                )
                for t in sorted(stats.candidates)
            ]
        )

    # ------------------------------------------------------------------
    def _hold(self, target: int) -> TargetParams:
        """One target's parameters, kept so a refused move can be put back."""
        return TargetParams(
            base=float(self.params.base[self.quantity][target]),
            coef=dict(self.params.coef[(target, self.quantity)]),
            effect=[c.effect[:, target].copy() for c in self.chains],
        )

    def _restore(self, target: int, held: TargetParams) -> None:
        """Put back what `_hold` kept."""
        self.params.base[self.quantity][target] = held.base
        self.params.coef[(target, self.quantity)] = dict(held.coef)
        for chain, column in zip(self.chains, held.effect):
            chain.effect[:, target] = column

    @staticmethod
    def _as_params(
        edges: Sequence[TypedEdge], chains: Sequence[RegimeChain], beta: np.ndarray
    ) -> TargetParams:
        """A solved vector as the coefficient set `_alternative` reads."""
        beta = np.asarray(beta, dtype=float)
        coef = {edge: float(beta[j + 1]) for j, edge in enumerate(edges)}
        effect: List[np.ndarray] = []
        start = 1 + len(edges)
        for chain in chains:
            width = chain.n_states - 1
            effect.append(np.concatenate([[0.0], beta[start : start + width]]))
            start += width
        return TargetParams(base=float(beta[0]), coef=coef, effect=effect)

    @staticmethod
    def _widened(
        stats: PooledSweep, target: int, candidate: int
    ) -> Optional[np.ndarray]:
        """
        One target's normal equations widened by one candidate, solved.

        The candidate's own moments are exactly the block the widened matrix
        needs: its product with every base column, with itself and with the
        response.  So the coefficient an accepted edge carries is already in hand
        and costs no second pass over the cohort.  What comes back is ordered
        intercept, base edges, the new edge, then the chain blocks, which is the
        order `_write` stores.

        `None` when the widened system carries no information to solve at.  This
        is the solve that writes a bought edge's coefficient, so it is the one
        that put a 1e11 intercept into the 2026-08-23 `exp16` legs, and a caller
        that gets `None` must put the move back rather than write what the ridge
        alone returns.
        """
        design, seen = stats.designs[target], stats.candidates[target]
        p = design.A.shape[0]
        A = np.zeros((p + 1, p + 1))
        A[:p, :p] = design.A
        A[:p, p] = A[p, :p] = seen.cross[candidate]
        A[p, p] = seen.diagonal[candidate]
        b = np.concatenate([design.b, [seen.target[candidate]]])
        beta, _, conditioned = solve_informed(A, b, _RIDGE)
        if not conditioned:
            return None
        cut = 1 + len(design.edges)
        return np.concatenate([beta[:cut], beta[-1:], beta[cut:-1]])

    def _move(
        self,
        cohort: Cohort,
        target: int,
        pool: Sequence[TypedEdge],
        picked: List[int],
        state: Dict[str, object],
    ) -> float:
        """
        The move actually made, put back when the edge's own contribution falls.

        The widened solve writes the coefficient the new edge carries, so the
        cohort is read again at a predictor the edge has moved.  With the edge
        carrying no coefficient, which is what the old apply returned, the
        predictor, the weights and the risk set are all unchanged and adding a
        column to a fixed quadratic raises its maximum by the rank-one gain and
        by exactly the rank-one gain, which is an identity and a test of nothing.

        The same pass reads the target's start term at the solve of the base
        alone, so the coefficient step both sides take cancels and what is left
        is the edge's.  Both terms are read at their own anchor, where the
        quadratic is tightest, and the two anchors are not the same one; that is
        the gap a rank-one gain cannot see and the reason this guard can refuse.

        `move_rounds` is how many coefficient rounds the move takes before the
        guard reads it.  A single Newton step at the current weights overshoots
        on a column that nearly separates the response, and the objective it
        lands at is below the one the move reaches two steps later, so a guard
        reading one step refuses a move worth making.
        """
        stats = state["stats"]
        candidate = picked[-1]
        edge = pool[candidate]
        design = stats.designs[target]
        held = self._hold(target)
        # A move whose solve carries no information is not a move.  Refusing it
        # here rather than writing what the ridge returns is what keeps a column
        # with no support out of the predictor; `grouped_search` reads the
        # non-finite value back as a refusal and moves to the next candidate.
        base_beta, _, conditioned = solve_informed(design.A, design.b, _RIDGE)
        if not conditioned:
            return float("-inf")
        alone = self._as_params(design.edges, self.chains, base_beta)
        widened_beta = self._widened(stats, target, candidate)
        if widened_beta is None:
            return float("-inf")
        self.graph.add_edge(edge, target, self.quantity)
        self._write(target, tuple(design.edges) + (edge,), widened_beta)
        trial = self.accumulate(
            cohort, step=False, targets=[target], against=(target, alone)
        )
        for _ in range(self.move_rounds - 1):
            widened = trial.designs[target]
            beta, _, conditioned = solve_informed(widened.A, widened.b, _RIDGE)
            if not conditioned:
                self.graph.remove_edge(edge, target, self.quantity)
                self._restore(target, held)
                return float("-inf")
            self._write(target, widened.edges, beta)
            trial = self.accumulate(
                cohort, step=False, targets=[target], against=(target, alone)
            )
        gained = trial.starts[target] - trial.against[target]
        if gained < 0.0:
            self.graph.remove_edge(edge, target, self.quantity)
            self._restore(target, held)
            return state["start"] + gained
        state["stats"], state["in"] = trial, picked
        state["start"] = state["start"] + gained
        return state["start"]

    def _moves(
        self, cohort: Cohort, target: int, stats: PooledSweep
    ) -> Tuple[List[TypedEdge], Callable, Callable]:
        """
        One target's candidate pool and the two functions a search reads it through.

        The gain reads the moments the pass already accumulated, so scoring the
        whole pool costs nothing.  The move is the pass, and the number it comes
        back with is the opening start term plus what every accepted edge has
        actually bought since.
        """
        pool = list(self.candidates[target])
        state: Dict[str, object] = {
            "stats": stats,
            "start": stats.starts[target],
            "in": [],
        }

        def gain_fn(_selected):
            # What is already in reaches this through the state, whose moments
            # were re-accumulated at the parameters the last accepted move wrote.
            held = state["stats"]
            design, seen = held.designs[target], held.candidates[target]
            return candidate_gains(
                BaseFit(design.A, design.b), seen.cross, seen.diagonal, seen.target
            )

        def apply_fn(selected):
            picked = list(selected)
            if picked == state["in"]:
                return state["start"]
            return self._move(cohort, target, pool, picked, state)

        return pool, gain_fn, apply_fn

    def _greedy(
        self,
        cohort: Cohort,
        target: int,
        stats: PooledSweep,
        price: float,
        max_edges: int,
    ) -> SearchReport:
        """One target's greedy round, at one cohort pass per move it actually makes."""
        pool, gain_fn, apply_fn = self._moves(cohort, target, stats)
        return greedy_search(gain_fn, apply_fn, len(pool), price, max_edges)

    def search(
        self, cohort: Cohort, price: float, max_edges: int = 8
    ) -> Dict[int, SearchReport]:
        """
        One greedy round per target, scored on the pooled sum and guarded per target.

        The gain of a candidate is the rank-one update of the summed matrices, so
        adding a subject raises every gain while the price stays where it was;
        that is how a larger cohort buys an edge a smaller one could not afford,
        and it is why the price is calibrated against a null measured on the same
        roster.  The guard stays per target, because a structure move changes the
        predictor of the target it lands on and of no other, which is also why
        one opening pass serves every target.

        The accepted edges are written into the graph, together with the
        coefficients the widened solve gave them.  A target the round refused
        leaves the parameters exactly where it found them.
        """
        stats = self.accumulate(cohort, step=False)
        return {
            target: self._greedy(cohort, target, stats, price, max_edges)
            for target in sorted(self.candidates)
        }

    def search_grouped(
        self,
        cohort: Cohort,
        price: RelationPrice,
        max_edges: int = 8,
        max_self: int = 2,
    ) -> Dict[int, GroupReport]:
        """
        One grouped round per target, with the two strata carrying their own bounds.

        A label's own history is declared structure, and on this cohort it
        outscores every cross-label relation by two orders of magnitude, so a
        flat in-degree bound spends the whole budget on self lags before a cross
        relation is reached.  The self stratum carries its own bound here and the
        cross budget is untouched by it.
        """
        stats = self.accumulate(cohort, step=False)
        return {
            target: self._grouped(cohort, target, stats, price, max_edges, max_self)
            for target in sorted(self.candidates)
        }

    def _grouped(
        self,
        cohort: Cohort,
        target: int,
        stats: PooledSweep,
        price: RelationPrice,
        max_edges: int,
        max_self: int,
    ) -> GroupReport:
        """One target's grouped round, over the same moves the flat round reads."""
        pool, gain_fn, apply_fn = self._moves(cohort, target, stats)
        # A same-epoch edge into this target closes a cycle only through a path
        # that leaves it, and every edge on such a path lands on another target,
        # so what is admissible here holds for the whole of this target's round.
        admits = [not self.graph.closes_cycle(edge, target) for edge in pool]
        return grouped_search(
            gain_fn,
            apply_fn,
            [(edge, target) for edge in pool],
            price,
            max_edges=max_edges,
            max_self=max_self,
            admits=admits,
            max_refused=self.max_refused,
        )
