"""
core/typed_graph.py
===================
Edges typed by source scale, source quantity and target quantity.

The allocation formulation keys a parent set by (scale, target label) and stores
an edge as `(source label, lag)`, which fixes the source scale to the target's
and the quantity to the count.  A duration source feeding a count target is not
expressible there, and neither is a coarse source feeding a fine target.

Here an edge is

    (src_kind, src_scale, src_label, src_quantity, lag, basis)
        -> (dst_label, dst_quantity)

with both quantities ranging over the count and the duration, so all four
pairings are candidates.  There is one target scale, the finest, because the
likelihood lives there, which is why the parent set is keyed by the target label
and its quantity.

This is a candidate space and not a list.  Nothing here privileges a pairing, a
lag, a resolution or a shape; the search proposes and the price decides.

Same-epoch reads
----------------
A relation shorter than one epoch survives aggregation as co-movement inside one
coarse epoch, so a model carrying lagged reads alone has no term for two labels
that move together inside the slot the likelihood lives on.  Such a read is a
candidate at the finest scale, where one epoch is one slot.  The order the labels
are drawn in is what keeps it predictable.  `label_order` sorts them so every
same-epoch source is drawn before the label that reads it, and `closes_cycle`
says which further edge would leave no such order.  A label
reading itself in the epoch being drawn is refused, since that is the value
being predicted.

The basis term
--------------
One source offers several columns and `basis` names the one this edge reads.
The empty term is the plain reading, which is log1p of an observed source and
the level of a calendar node.  A pair (lo, hi) is the indicator that the
source's value falls in that band.  A truth that steps at some threshold is
expressible as the step between two adjacent bands, and how well it is expressed
depends on where the cuts land.  A cut set that straddles the threshold puts
firing and non-firing rows in one band, and the band then scores below the log
term of the same source.  Where the truth is monotone the log term carries it
and the bands do not survive the price.

The term travels on the edge because a parent set, a candidate list and
`FineParams.coef` are all keyed by one.  Two terms of one source are two columns
carrying two coefficients, so they have to be two keys, and a structure keyed
beside the edge would have to reach the graph, the parameters, the design and
the null.  The field sits last and defaults to the log term, so an edge written
before a basis travelled with one hashes, sorts and keys exactly as it did.

Why this sits beside the older type
-----------------------------------
The older type cannot be widened without changing generated data.  Ground-truth
coefficients are drawn by iterating a `set` of edge keys and calling the random
generator inside that loop, so the keys' hash order fixes the draw sequence, and
any change of element type reorders it and moves every stored number.  Several
readers also rebuild the key by hand and fall back to a zero coefficient when the
lookup misses, so a widened key would be silently ignored and never rejected.
The two types therefore coexist until the learning stack reads this one.

Determinism
-----------
`iter_edges` returns edges in sorted order.  A set of records containing strings
iterates in an order that depends on string hashing, which is randomized per
process, and the older type only escapes that by holding integers.  Sorting at
the boundary is what keeps a result from depending on the interpreter's seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np

# What an edge reads from its source and writes to its target.  A calendar node
# carries neither a count nor a duration, only its own value.
COUNT = "count"
DURATION = "duration"
LEVEL = "level"

QUANTITIES = (COUNT, DURATION)

# Where a source lives.  An observed label is a column of the event log; a
# calendar node is a known function of the clock.
OBSERVED = "observed"
CALENDAR = "calendar"
SOURCE_KINDS = (OBSERVED, CALENDAR)

# The scale the likelihood lives on, where one epoch is one slot.  A same-epoch
# read of an observed source is admitted there and nowhere above it.
FINEST_SCALE = 1

# Which column of its source an edge reads.  The empty term is the plain
# reading and a pair is one band of the source's own range.
BasisTerm = Tuple[float, ...]
LOG_TERM: BasisTerm = ()


def read_basis(values: np.ndarray, basis: BasisTerm) -> np.ndarray:
    """One source's values through a basis term, which is the column an edge reads."""
    values = np.asarray(values, dtype=float)
    if not basis:
        return np.log1p(np.maximum(values, 0.0))
    return ((values >= basis[0]) & (values < basis[1])).astype(float)


def basis_label(basis: BasisTerm) -> str:
    """The name of a basis term, which every column built from it is named after."""
    if not basis:
        return "log"
    return f"[{basis[0]:.3g},{basis[1]:.3g})"


@dataclass(frozen=True, order=True)
class TypedEdge:
    """One lagged parent, typed by where it is read from and what it reads."""

    src_kind: str
    src_scale: int
    src_label: int
    src_quantity: str
    lag: int
    basis: BasisTerm = LOG_TERM

    def __post_init__(self) -> None:
        # Held as a tuple of plain floats, since the field is part of the hash
        # and of the order, and a band read off a quantile arrives as numpy.
        basis = tuple(float(v) for v in self.basis)
        if len(basis) not in (0, 2):
            raise ValueError(
                f"a basis term is the log term or one band of two edges; got {basis}"
            )
        if basis and not basis[0] < basis[1]:
            raise ValueError(f"a band runs from its lower edge upward; got {basis}")
        object.__setattr__(self, "basis", basis)
        if self.src_kind not in SOURCE_KINDS:
            raise ValueError(
                f"source kind {self.src_kind!r}; expected one of {list(SOURCE_KINDS)}"
            )
        if self.src_scale < 1:
            raise ValueError(f"source scale {self.src_scale} is not a scale index")
        if self.src_label < 0:
            raise ValueError(f"source label {self.src_label} is negative")
        if self.src_kind == CALENDAR:
            if self.src_quantity != LEVEL:
                raise ValueError(
                    f"a calendar source carries {LEVEL!r}, not {self.src_quantity!r}"
                )
            if basis:
                raise ValueError(
                    f"a calendar source carries its level whole, and {basis} bands it"
                )
        elif self.src_quantity not in QUANTITIES:
            raise ValueError(
                f"source quantity {self.src_quantity!r}; expected one of "
                f"{list(QUANTITIES)}"
            )
        if self.lag < 0:
            raise ValueError(f"lag {self.lag} is negative")
        # Lag zero reads the epoch being predicted.  A calendar source is fixed
        # by the clock, so reading it at t reveals nothing t does not already
        # give.  An observed source is read at the finest scale, where the epoch
        # is the slot itself and the order the labels are drawn in says what is
        # already known; a coarse epoch read at lag zero also holds the slots
        # that come after the one being predicted.
        if (
            self.lag == 0
            and self.src_kind == OBSERVED
            and self.src_scale != FINEST_SCALE
        ):
            raise ValueError(
                f"a same-epoch read of an observed source is admitted at scale "
                f"{FINEST_SCALE}, where one epoch is one slot; got scale "
                f"{self.src_scale}"
            )

    @property
    def is_exogenous(self) -> bool:
        return self.src_kind == CALENDAR

    @property
    def is_same_epoch(self) -> bool:
        """Whether this edge reads an observed source in the epoch being drawn."""
        return self.lag == 0 and self.src_kind == OBSERVED

    @property
    def relation(self) -> Tuple[str, int, int, str, int]:
        """The edge without its basis term, which is the relation it states."""
        return (
            self.src_kind,
            self.src_scale,
            self.src_label,
            self.src_quantity,
            self.lag,
        )

    @property
    def basis_name(self) -> str:
        """The name of this edge's basis term."""
        return basis_label(self.basis)


@dataclass
class TypedGraph:
    """
    Parent sets over typed edges, keyed by target label and target quantity.

    Attributes
    ----------
    label_names    : observed labels, which are the only targets
    calendar_names : exogenous calendar nodes, which are only ever sources
    """

    label_names: List[str]
    calendar_names: List[str] = field(default_factory=list)
    _parents: Dict[Tuple[int, str], Set[TypedEdge]] = field(
        default_factory=dict, repr=False
    )

    def __post_init__(self) -> None:
        for ell in range(self.L):
            for quantity in QUANTITIES:
                self._parents.setdefault((ell, quantity), set())

    @property
    def L(self) -> int:
        return len(self.label_names)

    @property
    def n_calendar(self) -> int:
        return len(self.calendar_names)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_edge(self, edge: TypedEdge, dst_label: int, dst_quantity: str) -> None:
        """Add one typed edge into the given target label and quantity."""
        self._check_target(dst_label, dst_quantity)
        self._check_source(edge, dst_label)
        self._parents[(dst_label, dst_quantity)].add(edge)

    def remove_edge(self, edge: TypedEdge, dst_label: int, dst_quantity: str) -> None:
        """Remove one typed edge, with no effect when it is absent."""
        self._check_target(dst_label, dst_quantity)
        self._parents[(dst_label, dst_quantity)].discard(edge)

    def set_parents(
        self, dst_label: int, dst_quantity: str, parents: Sequence[TypedEdge]
    ) -> None:
        """Replace the whole parent set of one target."""
        self._check_target(dst_label, dst_quantity)
        for edge in parents:
            self._check_source(edge, dst_label)
        self._parents[(dst_label, dst_quantity)] = set(parents)

    def _check_target(self, dst_label: int, dst_quantity: str) -> None:
        if not 0 <= dst_label < self.L:
            raise ValueError(f"target label {dst_label} is not one of {self.L}")
        if dst_quantity not in QUANTITIES:
            raise ValueError(
                f"target quantity {dst_quantity!r}; expected one of {list(QUANTITIES)}"
            )

    def _check_source(self, edge: TypedEdge, dst_label: int) -> None:
        limit = self.n_calendar if edge.is_exogenous else self.L
        if not 0 <= edge.src_label < limit:
            raise ValueError(
                f"source label {edge.src_label} is not one of the {limit} "
                f"{edge.src_kind} sources"
            )
        if edge.is_same_epoch and edge.src_label == dst_label:
            raise ValueError(
                f"a same-epoch edge from label {edge.src_label} into itself reads "
                "the value being predicted"
            )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_parents(self, dst_label: int, dst_quantity: str) -> Set[TypedEdge]:
        """A copy of the parent set of one target."""
        self._check_target(dst_label, dst_quantity)
        return set(self._parents[(dst_label, dst_quantity)])

    def n_edges(
        self, dst_label: Optional[int] = None, dst_quantity: Optional[str] = None
    ) -> int:
        """Edges in the graph, optionally restricted to one target."""
        return sum(
            len(edges)
            for (ell, quantity), edges in self._parents.items()
            if (dst_label is None or ell == dst_label)
            and (dst_quantity is None or quantity == dst_quantity)
        )

    def iter_edges(self) -> Iterator[Tuple[TypedEdge, int, str]]:
        """Every edge as (edge, target label, target quantity), in sorted order."""
        for ell, quantity in sorted(self._parents):
            for edge in sorted(self._parents[(ell, quantity)]):
                yield (edge, ell, quantity)

    def edge_set(self) -> frozenset:
        """The graph as a set of (edge, target label, target quantity) triples."""
        return frozenset(self.iter_edges())

    def structural_hamming_distance(self, other: "TypedGraph") -> int:
        """Edges to add plus edges to drop to turn this graph into the other."""
        return len(self.edge_set().symmetric_difference(other.edge_set()))

    # ------------------------------------------------------------------
    # Acyclicity
    # ------------------------------------------------------------------

    def same_epoch_children(self) -> Dict[int, Set[int]]:
        """Which labels read each label in the epoch they are all drawn at."""
        out: Dict[int, Set[int]] = {ell: set() for ell in range(self.L)}
        for edge, ell, _ in self.iter_edges():
            if edge.is_same_epoch:
                out[edge.src_label].add(ell)
        return out

    def label_order(self) -> List[int]:
        """
        The labels in an order that draws every same-epoch source before its reader.

        Ties go to the lower label index, so a graph carrying no same-epoch edge
        comes back in the plain order the labels were declared in.

        Raises:
            ValueError: when the same-epoch edges close a cycle over the labels,
                which leaves no label that can be drawn first.
        """
        children = self.same_epoch_children()
        waiting = {ell: 0 for ell in range(self.L)}
        for kids in children.values():
            for kid in kids:
                waiting[kid] += 1
        order: List[int] = []
        while waiting:
            ready = sorted(ell for ell, n in waiting.items() if n == 0)
            if not ready:
                raise ValueError(
                    "the same-epoch edges close a cycle over labels "
                    f"{sorted(waiting)}, so no label can be drawn first"
                )
            for ell in ready:
                del waiting[ell]
                for kid in children[ell]:
                    waiting[kid] -= 1
            order.extend(ready)
        return order

    def closes_cycle(self, edge: TypedEdge, dst_label: int) -> bool:
        """Whether adding this edge would leave the labels with no draw order."""
        if not edge.is_same_epoch:
            return False
        if edge.src_label == dst_label:
            return True
        children = self.same_epoch_children()
        seen = {dst_label}
        stack = [dst_label]
        while stack:
            for kid in children[stack.pop()]:
                if kid == edge.src_label:
                    return True
                if kid not in seen:
                    seen.add(kid)
                    stack.append(kid)
        return False

    def is_acyclic(self) -> bool:
        """
        Whether no directed cycle exists, which two properties together give.

        Every edge from an observed source at a lag of one or more points
        strictly backward in time and cannot close a cycle.  An edge at lag zero
        comes from a calendar node, which is never a target and therefore has no
        path back to it, or from another label at the finest scale, and those
        close a cycle exactly when no order draws every source before its reader.
        """
        try:
            self.label_order()
        except ValueError:
            return False
        return True

    def assert_acyclic(self) -> None:
        """Raise when the same-epoch edges leave no order to draw the labels in."""
        self.label_order()
