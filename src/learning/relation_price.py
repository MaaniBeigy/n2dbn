"""
learning/relation_price.py
==========================
One price per scientific relation, and a cheap refinement inside an accepted one.

An ordered pair of labels is a relation.  The source scale, the lag, the pairing
of source and target quantity and the basis term are representations of it, and
one relation offers hundreds of them.  Charging a full edge price for each makes
a simple relation compete against its own alternatives, and the loser of that
competition is the relation itself.  Here the price is nested: the first
representation of a relation pays for the relation, and every further one pays a
small refinement price.

A relation is priced once out of the whole candidate field, so the price carries
the search's own multiplicity.  Under a marginal-uniform prior on the active set
the j-th relation bought out of C costs `log((C - j + 1) / j)` above the base
price, which is what makes edge counts at two resolutions with different
candidate fields comparable.

A self relation is a declared part of the construction.  A label's own recent
history is in the model because the process has memory, and pricing it as a
discovery lets a strong autocorrelation take a slot the in-degree bound was
holding for a cross-label relation.  It is exempt from the relation price and
from the multiplicity term, it pays the refinement price alone, and it is
counted in its own stratum so no reported edge count mixes the two.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..core.typed_graph import OBSERVED, TypedEdge

#: The two strata an accepted edge is reported in.
CROSS = "cross"
SELF = "self"


@dataclass(frozen=True, order=True)
class RelationKey:
    """One ordered pair of a source and a target, which is what a price buys."""

    src_kind: str
    src_label: int
    dst_label: int

    @property
    def is_self(self) -> bool:
        """Whether the source is the target's own history."""
        return self.src_kind == OBSERVED and self.src_label == self.dst_label

    @property
    def stratum(self) -> str:
        return SELF if self.is_self else CROSS


def relation_of(edge: TypedEdge, dst_label: int) -> RelationKey:
    """The relation an edge is one representation of."""
    return RelationKey(edge.src_kind, edge.src_label, dst_label)


def candidate_relations(
    candidates: Sequence[Tuple[TypedEdge, int]],
) -> List[RelationKey]:
    """The distinct relations a candidate list represents, in sorted order."""
    return sorted({relation_of(edge, dst) for edge, dst in candidates})


@dataclass
class RelationPrice:
    """
    The nested price, held with the size of the field it charges against.

    Attributes
    ----------
    relation      : what the first representation of a cross-label relation costs
    representation: what every further representation of an accepted one costs
    n_relations   : the cross-label relations the search may choose among, which
                    is the multiplicity the price corrects for
    """

    relation: float
    representation: float
    n_relations: int

    def __post_init__(self) -> None:
        if self.relation < self.representation:
            raise ValueError(
                f"a relation costs at least as much as a refinement inside it; "
                f"got {self.relation} against {self.representation}"
            )
        if self.n_relations < 1:
            raise ValueError(
                f"a candidate field holds at least one relation; got {self.n_relations}"
            )

    def multiplicity(self, rank: int) -> float:
        """
        What the rank-th purchase costs above the base price, one-indexed.

        A rank past the size of the field has no purchase left to price and
        comes back infinite, which stops a search that declared a field smaller
        than the one it is searching.
        """
        if rank < 1:
            raise ValueError(f"a purchase is ranked from one; got {rank}")
        if rank > self.n_relations:
            return float("inf")
        return float(np.log((self.n_relations - rank + 1) / rank))

    def of(self, key: RelationKey, active: Mapping[RelationKey, int]) -> float:
        """
        What one more representation of a relation costs, given what is bought.

        Args:
            active: how many representations of each relation are already in.
        """
        if active.get(key, 0) > 0:
            return self.representation
        if key.is_self:
            return self.representation
        rank = 1 + sum(1 for other in active if not other.is_self)
        return self.relation + self.multiplicity(rank)


@dataclass
class GroupReport:
    """What a grouped search accepted, and what each stratum of it holds."""

    accepted: List[int] = field(default_factory=list)
    gains: List[float] = field(default_factory=list)
    prices: List[float] = field(default_factory=list)
    rejected: List[int] = field(default_factory=list)
    trace: List[float] = field(default_factory=list)
    relations: Dict[RelationKey, int] = field(default_factory=dict)

    def by_stratum(self) -> Dict[str, int]:
        """Accepted representations, counted in the stratum each belongs to."""
        out = Counter({CROSS: 0, SELF: 0})
        for key, count in self.relations.items():
            out[key.stratum] += count
        return dict(out)

    def relation_count(self, stratum: str = CROSS) -> int:
        """Distinct relations bought in one stratum."""
        return sum(1 for key in self.relations if key.stratum == stratum)


def grouped_search(
    gain_fn: Callable[[Sequence[int]], np.ndarray],
    apply_fn: Callable[[Sequence[int]], float],
    candidates: Sequence[Tuple[TypedEdge, int]],
    price: RelationPrice,
    max_edges: int = 8,
    max_self: int = 2,
    admits: Optional[Sequence[bool]] = None,
    max_refused: int = 1,
) -> GroupReport:
    """
    Add the best candidate that clears its own price, guarding each move.

    A candidate's price depends on the relations already bought, so the ordering
    of the candidates by gain and the ordering by gain less price differ, and it
    is the second one that decides.  A representation of an accepted relation is
    cheap, which is what lets the search read the lag, the quantity pairing and
    the basis out inside a group it has already paid for.

    The two strata carry their own in-degree bounds.  A label's own history is
    declared structure and a strong autocorrelation would otherwise take every
    slot `max_edges` holds, leaving a cross-label relation nothing to be bought
    with.

    `gain_fn` and `apply_fn` carry the same contract they do in `fine_sstep`: a
    gain is a rank-one score at the fitted base and the realized objective is
    what the move actually reached, and the two differ because a structure move
    moves the anchor and the posterior.

    A refused move blocks its own candidate and the round carries on to the next
    one.  The guard reads what a move actually reached, and a column that nearly
    separates the response can land below the base after the rounds it is given,
    which says nothing about the candidate ranked behind it.  `max_refused`
    bounds how many cohort passes a round spends on moves it does not keep, and
    one refusal is the bound that ends the round at the first one.

    Args:
        admits: one flag per candidate, false where the caller cannot hold it;
            a same-epoch edge that would leave the labels with no draw order is
            what this keeps out.
        max_refused: refusals the round takes before it stops.
    """
    report = GroupReport()
    keys = [relation_of(edge, dst) for edge, dst in candidates]
    allowed = (
        np.ones(len(keys), dtype=bool)
        if admits is None
        else np.asarray(admits, dtype=bool)
    )
    selected: List[int] = []
    refused: List[int] = []
    current = apply_fn(selected)
    report.trace.append(current)

    while True:
        taken = report.by_stratum()
        if taken[CROSS] >= max_edges and taken[SELF] >= max_self:
            break
        gains = np.array(gain_fn(selected), dtype=float, copy=True)
        gains[selected] = -np.inf
        prices = np.array([price.of(key, report.relations) for key in keys])
        net = gains - prices
        net[~allowed] = -np.inf
        net[refused] = -np.inf
        for i, key in enumerate(keys):
            if taken[key.stratum] >= (max_self if key.is_self else max_edges):
                net[i] = -np.inf
        best = int(np.argmax(net))
        if not np.isfinite(net[best]) or net[best] <= 0.0:
            break
        proposed = selected + [best]
        realized = apply_fn(proposed)
        # `realized < current` alone accepts a NaN, because every comparison
        # against NaN is False.  A move whose objective is not a number is a
        # move whose solve did not resolve, and it is refused like any other.
        if not np.isfinite(realized) or realized < current:
            report.rejected.append(best)
            refused.append(best)
            if len(refused) >= max_refused:
                break
            continue
        selected = proposed
        current = realized
        report.accepted.append(best)
        report.gains.append(float(gains[best]))
        report.prices.append(float(prices[best]))
        report.relations[keys[best]] = report.relations.get(keys[best], 0) + 1
        report.trace.append(current)
    return report
