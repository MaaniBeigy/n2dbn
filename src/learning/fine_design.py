"""
learning/fine_design.py
=======================
One target of a typed graph as the columns both halves of the fit read.

The parameter step consumes a list of `DesignBlock` and the structure step a
list of `ScaledColumn`.  Built in two places the two drift, and a gain scored on
one column is then realized on another.  Here both come from one centered
coarse-grid vector per edge, so the column the search scores is the column the
solve fits.

Three statements make a column.

An edge is read on its own source scale.  `intensity.edge_feature` returns that
value at every finest slot, and the same information takes `n_epochs[k]` numbers
at scale k, so the coarse vector is what is held and the broadcast is what the
solve sees.  The edge's basis term says which column of the source is read, the
log term or the indicator of one band, and a band is lagged and centered by the
same three lines the log term is.

A column is centered over the record, which removes the one direction the design
already carries at every scale, the intercept.  The mean is an average over what
the record holds, so an unobserved epoch and an epoch whose lagged read falls
inside a gap carry no weight in it, and a coefficient fitted at observed slots
stays free of what a gap hides.  A record with no gap weights every epoch by its
slot count.  The result is still constant inside a scale-k epoch, so a centered
column is still a scale-k column and the search can hold it as one.

A calendar node reaches the design as a lag-zero calendar edge at the scale
`features.home_scale` places it at.  Every coefficient then has a `TypedEdge`
key in `FineParams.coef`, and a calendar effect is stored the way a lagged count
is.

Blocks are the intercept, then one per scale from coarsest to finest, then
whatever latent blocks the caller brings.  Centering separates every column from
the intercept and leaves the columns correlated with each other, so the pairs
the constraint cannot separate share a solve.  A latent block varies at every slot
and carries a posterior covariance, so it has no coarse-grid form and enters the
search matrix at the finest grid, which is where the solve reads it too.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..core.calendar import SlotCalendar
from ..core.features import CalendarNode, home_scale, node_matrix
from ..core.mask import dilate_forward, reduce_by
from ..core.typed_graph import CALENDAR, TypedEdge, TypedGraph, read_basis
from ..inference.bound import broadcast, segment_sum
from ..model.intensity import FineParams, FineSources, assert_predictable
from .fine_mstep import DesignBlock, stack_design
from .fine_sstep import ScaledColumn, resolution_aware_gram, weights_by_scale

# ---------------------------------------------------------------------------
# The ladder both the centering and the reduction read
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScaleLadder:
    """
    The nesting of the scales, derived once and passed to everything that needs it.

    Attributes
    ----------
    ancestors : the scale-k epoch of every finest slot, one array per scale
    n_epochs  : epochs at each scale, finest first
    sizes     : finest slots inside each epoch, which is the weight centering
                removes a mean with wherever the record covers them
    observed  : the observation mask AND-reduced into each scale, so an epoch
                counts as observed when every finest slot under it is, and all
                true when the record carries no gap
    """

    ancestors: List[np.ndarray]
    n_epochs: List[int]
    sizes: List[np.ndarray]
    observed: List[np.ndarray]

    @property
    def K(self) -> int:
        return len(self.n_epochs)

    @property
    def n_slots(self) -> int:
        return int(self.n_epochs[0])

    @classmethod
    def from_sources(
        cls, sources: FineSources, observed: Optional[np.ndarray] = None
    ) -> "ScaleLadder":
        """
        The ladder of one source set, with the nesting verified.

        Args:
            observed: (T_1,) bool, one flag per finest slot; None is a record
                observed end to end.

        The nesting is verified and nothing of it is stored: a scale-k epoch
        holding slots from two scale-(k+1) parents would make a lagged coarse
        read reach across a boundary, and every consumer here reads the ancestor
        maps directly.

        Raises:
            ValueError: when a scale's epoch count disagrees with how far its
                ancestor map reaches, when a scale-k epoch holds slots from
                more than one scale-(k+1) parent, or when the mask carries some
                other number of flags than one per finest slot.
        """
        ancestors = [np.asarray(a, dtype=np.int64) for a in sources.ancestors]
        n_epochs = [int(c.shape[0]) for c in sources.counts]
        sizes: List[np.ndarray] = []
        for k, count in enumerate(n_epochs):
            reach = int(np.max(ancestors[k], initial=-1)) + 1
            if reach != count:
                raise ValueError(
                    f"scale {k + 1} holds {count} epochs and its ancestor map "
                    f"reaches {reach}, so an epoch has no slots of its own or "
                    "the map runs off the end of the arrays"
                )
            sizes.append(segment_sum(np.ones(ancestors[k].size), ancestors[k], count))

        if observed is None:
            flags = np.ones(n_epochs[0], dtype=bool)
        else:
            flags = np.asarray(observed)
            if flags.dtype != np.bool_:
                raise ValueError(
                    "an observation mask is a boolean array; this one has "
                    f"dtype {flags.dtype}"
                )
        if flags.shape != (n_epochs[0],):
            raise ValueError(
                f"an observation mask carries one flag per finest slot, so "
                f"{n_epochs[0]} of them; this one is {flags.shape}"
            )
        # A coarse epoch counts as observed only when the whole of it is, since
        # its stored count is a sum over the children the record covers and
        # nothing above can tell such a sum from a whole one.
        seen = [reduce_by(flags, ancestors[k], n) for k, n in enumerate(n_epochs)]

        for k in range(len(n_epochs) - 1):
            # The parent index is the mean of the finer map inside the epoch,
            # exactly when the ladder nests, which the next line checks.
            mean = segment_sum(
                ancestors[k + 1].astype(float), ancestors[k], n_epochs[k]
            ) / np.maximum(sizes[k], 1.0)
            pm = np.rint(mean).astype(np.int64)
            if not np.array_equal(broadcast(pm, ancestors[k]), ancestors[k + 1]):
                raise ValueError(
                    f"scale {k + 1} does not nest inside scale {k + 2}: one of "
                    "its epochs holds slots from more than one parent"
                )
        return cls(
            ancestors=ancestors,
            n_epochs=n_epochs,
            sizes=sizes,
            observed=seen,
        )


def center_at_scale(
    values: np.ndarray, ladder: ScaleLadder, scale: int, budget: int
) -> np.ndarray:
    """
    One scale's column or columns with the record's own weighted mean removed.

    The mean removed is the record's, which is the one direction every design
    here already carries: the intercept is constant over the record and a
    coefficient fitted against a centered column leaves it alone.

    A mean taken inside each scale-(k+1) epoch removes one number per parent
    epoch instead, and the design holds no coefficient for those numbers, so the
    between-parent part of a relation leaves the model with them.  A day-scale
    column centered inside its week keeps its within-week contrast alone, and at
    the benchmark's lagged sleep band that is 48 percent of the column's
    variance and 25 percent of its covariance with the target, so an exact
    likelihood ratio of 220 nats arrives at the search as 29.  Two columns of
    two scales are left correlated here, which the joint solve separates and
    which the Schur complement of `fine_sstep.candidate_gains` takes out of a
    candidate against whatever the base already holds.

    A block mean is an average over what was recorded, so an epoch is weighted
    by the finest slots it holds where the record covers it and by zero where it
    does not.  An epoch reading back over a gap holds a value nobody recorded
    and carries the same weight of zero, so what comes off a column at the rows
    a masked fit scores is a mean of recorded numbers alone.

    The erosion uses the budget the search declared and never the lag of the
    column in hand.  Two candidates the search compares are summed over one row
    set, so they have to be centered under one convention; keyed to their own
    lags, a lag-1 candidate keeps epochs a lag-3 candidate drops and the winner
    can change on the convention alone.

    A record carrying no weight anywhere has no mean and its entries keep the
    values they came in with.  That is safe exactly when the budget covers every
    lag the design reads, which `edge_columns` checks, since a shorter budget
    would score rows against uncentered values.

    Args:
        values: (H_k,) or (H_k, P), the column or columns on the scale's grid.
        budget: the largest lag the search may read at this scale, which is how
            far the mask erodes forward before it becomes the weight.

    Raises:
        ValueError: when the scale is off the ladder, or the column set is not
            one or two dimensional, or it holds some other number of entries
            than the scale has epochs.
    """
    values = np.asarray(values, dtype=float)
    if not 1 <= scale <= ladder.K:
        raise ValueError(f"scale {scale} is not one of the {ladder.K} scales")
    if values.ndim not in (1, 2):
        raise ValueError(
            f"a column set is {values.shape}, which is neither 1-D nor 2-D"
        )
    if values.shape[0] != ladder.n_epochs[scale - 1]:
        raise ValueError(
            f"a scale {scale} column has {ladder.n_epochs[scale - 1]} entries; "
            f"got {values.shape[0]}"
        )
    flat = values[:, None] if values.ndim == 1 else values
    recorded = dilate_forward(ladder.observed[scale - 1], budget)
    weight = (ladder.sizes[scale - 1] * recorded)[:, None]
    out = flat - (flat * weight).sum(axis=0) / max(float(weight.sum()), 1.0)
    return out[:, 0] if values.ndim == 1 else out


# ---------------------------------------------------------------------------
# Where a calendar node's columns sit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalendarIndex:
    """
    Where each calendar node's encoded columns sit inside `FineSources.calendar`.

    A `TypedEdge` from a calendar source carries a scale and an integer offset,
    so this is what names a fitted coefficient back to the node it came from.
    """

    placed: Dict[Tuple[int, int], Tuple[str, str]]
    widths: Dict[int, int]

    def column_name(self, scale: int, offset: int) -> str:
        """The name of the encoded column at one scale and offset."""
        if (scale, offset) not in self.placed:
            raise ValueError(
                f"no calendar column sits at scale {scale} offset {offset}; that "
                f"scale holds {self.widths.get(scale, 0)} of them"
            )
        return self.placed[(scale, offset)][1]

    @classmethod
    def place(
        cls,
        nodes: Sequence[CalendarNode],
        cal: SlotCalendar,
        sources: FineSources,
        ladder: ScaleLadder,
    ) -> "CalendarIndex":
        """
        Place every node at its home scale, in the order the blocks concatenate.

        Raises:
            ValueError: when the placed columns do not fill `sources.calendar`
                exactly, or when the columns a node claims hold something other
                than what that node encodes, since a coefficient could then be
                named after a column it was not fitted on.
        """
        if len(sources.calendar) != ladder.K:
            raise ValueError(
                f"the sources carry {len(sources.calendar)} calendar blocks for a "
                f"ladder of {ladder.K} scales"
            )
        placed: Dict[Tuple[int, int], Tuple[str, str]] = {}
        widths: Dict[int, int] = {k: 0 for k in range(1, ladder.K + 1)}
        claims: List[Tuple[int, int, CalendarNode]] = []
        for node in nodes:
            k = home_scale(cal, node, ladder.n_epochs)
            claims.append((k, widths[k], node))
            for name in node.columns:
                placed[(k, widths[k])] = (node.name, name)
                widths[k] += 1
        for k in range(1, ladder.K + 1):
            expected = (ladder.n_epochs[k - 1], widths[k])
            given = tuple(np.shape(sources.calendar[k - 1]))
            if given != expected:
                raise ValueError(
                    f"the scale {k} calendar block is {given} where the given "
                    f"nodes place {expected}, so a fitted coefficient could not "
                    "be named after the column it was fitted on"
                )
        # A column count survives any permutation of the nodes sharing a scale,
        # so each claim is read against the values the node itself encodes.
        for k, start, node in claims:
            block = np.asarray(sources.calendar[k - 1], dtype=float)
            own = node_matrix(cal, node, k, ladder.n_epochs[k - 1])
            if not np.allclose(block[:, start : start + node.width], own):
                raise ValueError(
                    f"the scale {k} calendar block does not hold {node.name!r} at "
                    f"offset {start}, so a fitted coefficient could not be named "
                    "after the column it was fitted on"
                )
        return cls(placed=placed, widths=widths)


# ---------------------------------------------------------------------------
# An edge as a column
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DesignColumn:
    """
    One edge as the coarse-grid column both steps read, with what centering took.

    `values` is centered inside the scale-(k+1) parent and `removed` is the mean
    that came out, so `values + removed` broadcast is what `edge_feature`
    returns and `removed` is what the predictor carries as an offset.
    """

    name: str
    scale: int
    values: np.ndarray
    removed: np.ndarray
    edge: TypedEdge

    def as_scaled(self) -> ScaledColumn:
        return ScaledColumn(self.name, self.scale, self.values)

    def at_finest(self, ladder: ScaleLadder) -> np.ndarray:
        return broadcast(self.values, ladder.ancestors[self.scale - 1])

    def removed_at_finest(self, ladder: ScaleLadder) -> np.ndarray:
        return broadcast(self.removed, ladder.ancestors[self.scale - 1])


def _source_vector(
    edge: TypedEdge, graph: TypedGraph, sources: FineSources, index: CalendarIndex
) -> Tuple[np.ndarray, str]:
    """
    The edge's source on its own scale, unlagged, and the name of that source.

    An observed source is read through the edge's basis term, so the log term
    and each band of one source are separate columns and the name says which is
    in hand.  `intensity.edge_feature` reads the same term off the same array,
    which is what keeps the column the solve fits and the column the predictor
    carries one column.
    """
    k = edge.src_scale
    if edge.src_kind == CALENDAR:
        # Named before it is read, so an offset outside the block comes back
        # with its reason attached.
        name = index.column_name(k, edge.src_label)
        return np.asarray(sources.calendar[k - 1], dtype=float)[:, edge.src_label], name
    array = sources.quantity(k, edge.src_quantity)[:, edge.src_label]
    label = graph.label_names[edge.src_label]
    name = f"{label}.{edge.src_quantity}:{edge.basis_name}"
    return read_basis(array, edge.basis), name


def edge_columns(
    edges: Sequence[TypedEdge],
    graph: TypedGraph,
    sources: FineSources,
    ladder: ScaleLadder,
    index: CalendarIndex,
    budget: Optional[Mapping[int, int]] = None,
) -> List[DesignColumn]:
    """
    Every edge as a lagged, centered column on its own source scale.

    `budget` is the largest lag the search may read at each scale, and every
    column is centered under it, so two candidates the search compares carry one
    convention.  An edge reading further back than its scale allows is refused,
    since the centering would then leave rows scored against uncentered values
    and a gap would reach the fit through them.

    Raises:
        ValueError: when an edge reads a scale the ladder does not have, a
            calendar offset nothing was placed at, or a lag past the declared
            budget of its scale.
    """
    allowed = dict(budget or {})
    out: List[DesignColumn] = []
    for edge in edges:
        if not 1 <= edge.src_scale <= ladder.K:
            raise ValueError(
                f"{edge} reads scale {edge.src_scale}, which is not one of the "
                f"{ladder.K} scales of this ladder"
            )
        assert_predictable(edge, sources)
        source, name = _source_vector(edge, graph, sources, index)
        raw = np.zeros(ladder.n_epochs[edge.src_scale - 1])
        reach = np.arange(raw.size) - edge.lag
        inside = reach >= 0
        raw[inside] = source[reach[inside]]
        reach_budget = allowed.get(edge.src_scale, edge.lag)
        if edge.lag > reach_budget:
            raise ValueError(
                f"an edge reads scale {edge.src_scale} at lag {edge.lag} while "
                f"the declared budget there is {reach_budget}, so the rows it "
                "would be scored on and the mean removed from it disagree"
            )
        centered = center_at_scale(raw, ladder, edge.src_scale, reach_budget)
        out.append(
            DesignColumn(
                name=f"{name} k{edge.src_scale} lag{edge.lag}",
                scale=edge.src_scale,
                values=centered,
                removed=raw - centered,
                edge=edge,
            )
        )
    return out


# ---------------------------------------------------------------------------
# The design
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FineDesign:
    """
    One target's columns, as blocks for the solve and as columns for the search.

    Attributes
    ----------
    blocks  : the intercept, then one block per scale coarsest to finest, then
              the caller's latent blocks
    columns : the edge columns in the order they sit in the stacked design,
              which is the order after the intercept
    latent  : the caller's blocks, which carry a posterior covariance and have
              no coarse-grid form
    """

    target: int
    quantity: str
    blocks: List[DesignBlock]
    columns: List[DesignColumn]
    latent: List[DesignBlock]
    ladder: ScaleLadder

    @property
    def width(self) -> int:
        return sum(b.width for b in self.blocks)

    @property
    def names(self) -> List[str]:
        """One name per coefficient, in the order the stacked design holds them."""
        out = ["intercept"] + [c.name for c in self.columns]
        for block in self.latent:
            out.extend(f"{block.name}[{j}]" for j in range(block.width))
        return out

    def scaled(self) -> List[ScaledColumn]:
        """
        The intercept and every edge column, each at the resolution it varies at.

        The intercept is a column of ones at the coarsest scale, which is where
        it is constant.  A latent block is left out, since a posterior column
        has no coarse-grid form and carries a covariance the reduction cannot
        represent.  `search_gram` picks it up at the finest grid.
        """
        top = ScaledColumn(
            "intercept", self.ladder.K, np.ones(self.ladder.n_epochs[-1])
        )
        return [top] + [c.as_scaled() for c in self.columns]

    def search_gram(
        self, weights: np.ndarray, at_risk: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        The weighted Gram of every coefficient, each entry at its own resolution.

        The intercept and the edge columns reduce at their own resolution, which
        is what keeps a sweep of the whole candidate set affordable.  A latent
        block varies at every slot and carries a posterior covariance, so its own
        entries and its products with the edge columns are the finest-grid ones
        `fine_mstep.normal_equations` forms, and the matrix the search scores
        stays the matrix the solve inverts.
        """
        coarse = resolution_aware_gram(
            self.scaled(),
            self.ladder.ancestors,
            scale_weights(weights, self.ladder, at_risk),
        )
        if not self.latent:
            return coarse
        w = np.asarray(weights, dtype=float) * risk_multiplier(
            at_risk, self.ladder.n_slots
        )
        means, cov = stack_design(self.latent)
        weighted = means * w[:, None]
        fine = np.stack(
            [np.ones(self.ladder.n_slots)]
            + [c.at_finest(self.ladder) for c in self.columns],
            axis=1,
        )
        cross = fine.T @ weighted
        p = coarse.shape[0]
        out = np.zeros((p + means.shape[1], p + means.shape[1]))
        out[:p, :p] = coarse
        out[:p, p:] = cross
        out[p:, :p] = cross.T
        out[p:, p:] = means.T @ weighted + np.einsum("t,tpq->pq", w, cov)
        return out

    def search_target(
        self, response: np.ndarray, at_risk: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        The per-slot response summed against every coefficient's column.

        This is the right-hand side of the same solve `search_gram` is the matrix
        of, so a gain read off the two is the gain the parameter step realizes.
        `inference.bound.fisher_response` is the response the fit puts here.
        """
        summed = scale_response(response, self.ladder, at_risk)
        out = [float(summed[-1].sum())] + [
            float(summed[c.scale - 1] @ c.values) for c in self.columns
        ]
        if not self.latent:
            return np.asarray(out)
        r = np.asarray(response, dtype=float) * risk_multiplier(
            at_risk, self.ladder.n_slots
        )
        means, _ = stack_design(self.latent)
        return np.concatenate([np.asarray(out), means.T @ r])

    def intercept(self, beta: np.ndarray) -> float:
        return float(np.asarray(beta, dtype=float)[0])

    def coefficients(self, beta: np.ndarray) -> Dict[TypedEdge, float]:
        """The fitted coefficient of every edge, keyed by the edge it belongs to."""
        beta = np.asarray(beta, dtype=float)
        return {c.edge: float(beta[j + 1]) for j, c in enumerate(self.columns)}

    def predictor(self, beta: np.ndarray) -> np.ndarray:
        """The design's own linear predictor at every finest slot."""
        means, _ = stack_design(self.blocks)
        return means @ np.asarray(beta, dtype=float)

    def offset(self, beta: np.ndarray) -> np.ndarray:
        """
        What the predictor holds beyond the intercept and the edge features.

        Centering took a parent mean out of every column, and a latent block
        has no edge key, so both reach `linear_predictor` through its offset
        argument and the two predictors agree slot by slot.
        """
        beta = np.asarray(beta, dtype=float)
        out = np.zeros(self.ladder.n_slots)
        for j, column in enumerate(self.columns):
            out = out - beta[j + 1] * column.removed_at_finest(self.ladder)
        start = 1 + len(self.columns)
        for block in self.latent:
            out = out + block.columns @ beta[start : start + block.width]
            start += block.width
        return out

    def write_into(self, params: FineParams, beta: np.ndarray) -> FineParams:
        """Store the intercept and the coefficients where the likelihood reads them."""
        beta = np.asarray(beta, dtype=float)
        params.base[self.quantity][self.target] = self.intercept(beta)
        for edge, value in self.coefficients(beta).items():
            params.coef[(self.target, self.quantity)][edge] = value
        return params


def build_design(
    graph: TypedGraph,
    target: int,
    quantity: str,
    sources: FineSources,
    ladder: ScaleLadder,
    index: CalendarIndex,
    latent: Sequence[DesignBlock] = (),
    budget: Optional[Mapping[int, int]] = None,
) -> FineDesign:
    """One target's parent set as the blocks of a solve and the columns of a search."""
    columns = edge_columns(
        sorted(graph.get_parents(target, quantity)),
        graph,
        sources,
        ladder,
        index,
        budget,
    )
    ordered: List[DesignColumn] = []
    blocks = [DesignBlock("intercept", np.ones((ladder.n_slots, 1)))]
    for k in range(ladder.K, 0, -1):
        here = [c for c in columns if c.scale == k]
        if not here:
            continue
        ordered.extend(here)
        blocks.append(
            DesignBlock(
                f"scale {k}", np.stack([c.at_finest(ladder) for c in here], axis=1)
            )
        )
    latent = list(latent)
    return FineDesign(
        target=int(target),
        quantity=quantity,
        blocks=blocks + latent,
        columns=ordered,
        latent=latent,
        ladder=ladder,
    )


# ---------------------------------------------------------------------------
# What the search reduces against
# ---------------------------------------------------------------------------


def risk_multiplier(at_risk: Optional[np.ndarray], n_slots: int) -> np.ndarray:
    """
    The risk set as the per-slot multiplier, with any other encoding refused.

    The parameter step selects its rows with `means[at_risk]`, so a boolean mask
    is the one encoding that means the same thing there and here.  A 0/1 integer
    array is elementwise identical to it and indexes rows 0 and 1 instead, which
    puts the search and the solve on two different problems and raises nothing.

    Raises:
        ValueError: when the mask is not boolean, or does not carry one entry
            per finest slot.
    """
    if at_risk is None:
        return np.ones(n_slots)
    mask = np.asarray(at_risk)
    if mask.dtype != np.bool_:
        raise ValueError(
            f"a risk set is a boolean mask; this one has dtype {mask.dtype}, "
            "which the parameter step reads as row indices while the reduction "
            "reads it as a multiplier"
        )
    if mask.shape != (n_slots,):
        raise ValueError(
            f"a risk set carries one entry per finest slot, so {n_slots} of "
            f"them; this one is {mask.shape}"
        )
    return mask.astype(float)


def scale_weights(
    weights: np.ndarray, ladder: ScaleLadder, at_risk: Optional[np.ndarray] = None
) -> List[np.ndarray]:
    """
    The quadratic's per-slot weights summed into every scale, off-risk slots zeroed.

    The parameter step forms its matrix over the risk set alone, so a weight
    left standing outside it puts the reduction and the solve on two different
    problems.

    Raises:
        ValueError: when the risk set is not a boolean mask of the finest slots.
    """
    w = np.asarray(weights, dtype=float) * risk_multiplier(at_risk, ladder.n_slots)
    return weights_by_scale(w, ladder.ancestors, ladder.n_epochs)


def scale_response(
    response: np.ndarray, ladder: ScaleLadder, at_risk: Optional[np.ndarray] = None
) -> List[np.ndarray]:
    """
    A per-slot response summed into every scale, off-risk slots zeroed.

    Raises:
        ValueError: when the risk set is not a boolean mask of the finest slots.
    """
    r = np.asarray(response, dtype=float) * risk_multiplier(at_risk, ladder.n_slots)
    return [
        segment_sum(r, ladder.ancestors[k], ladder.n_epochs[k]) for k in range(ladder.K)
    ]
