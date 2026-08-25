"""
learning/candidate_space.py
===========================
What the search may propose, built so the column it scores is the column it fits.

A source offers several readings of itself.  The log term is the plain one, a
band of the source's own range is a second, and the two summaries of the lag
axis in `model.recency` are a third and a fourth.  Every reading is a column,
and a column the search scores has to be the column the parameter step later
fits, or a gain measured on one is realized on another.

Two of the readings travel on a `TypedEdge`, since a band is what `basis` names
and `read_basis` builds it in both places.  `basis_expansion` cuts a source at
its own quantiles and returns the edges of that cut, so an enumeration and a
design agree by construction.

The two summaries of the lag axis are columns of the search and of the design
and carry no edge yet.  Reaching a generated log through `intensity.sample`
needs a source kind that reads the strict past and is updated as the sampler
fills it, which is a change to the generative path and not to this one.

The carrier a regime effect needs is decided here as well.  Below the coarsest
scale an additive regime level is constant on every block its path does not
switch inside and contributes nothing, so a regime enters through a feature that
varies inside a block.  `assert_carries_detail` refuses a carrier that does not.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from ..core.projection import BlockProjector
from ..core.typed_graph import OBSERVED, TypedEdge
from ..model.recency import geometric_lag, recency_column
from .fine_sstep import ScaledColumn, basis_columns, basis_terms, quantile_edges

#: The decay of the geometric-lag column, one epoch of memory beyond the last.
DEFAULT_DECAY = 0.5


def basis_expansion(
    values: np.ndarray,
    src_scale: int,
    src_label: int,
    src_quantity: str,
    lag: int,
    n_bins: int = 4,
    src_kind: str = OBSERVED,
) -> List[TypedEdge]:
    """
    One source as the edges of its own basis, the log term and each band above it.

    The cuts are the source's own quantiles, so no relation is anticipated, and a
    cut leaving its lower bin empty is dropped before the edges are formed.

    Args:
        values: the source on its own scale, unlagged.
        n_bins: bins the source is cut into, of which the first is the reference.
    """
    edges = quantile_edges(np.asarray(values, dtype=float), n_bins)
    return [
        TypedEdge(src_kind, src_scale, src_label, src_quantity, lag, term)
        for term in basis_terms(edges)
    ]


def basis_candidates(
    values: np.ndarray,
    src_scale: int,
    src_label: int,
    src_quantity: str,
    lag: int,
    name: str,
    n_bins: int = 4,
) -> Tuple[List[TypedEdge], List[ScaledColumn]]:
    """
    The basis edges of one source and the columns each of them reads.

    The two lists are in one order and one term decides both entries, so the
    column at position i is what the edge at position i carries.
    """
    values = np.asarray(values, dtype=float)
    edges = quantile_edges(values, n_bins)
    return (
        basis_expansion(values, src_scale, src_label, src_quantity, lag, n_bins),
        basis_columns(values, edges, name, src_scale),
    )


def derived_columns(
    values: np.ndarray, scale: int, name: str, decay: float = DEFAULT_DECAY
) -> List[ScaledColumn]:
    """
    The two summaries of the lag axis, as the columns a search scores them at.

    Both read the strict past of the series, so both may be read at the epoch
    they predict, and both keep a within-block contrast at every interior scale.
    """
    values = np.asarray(values, dtype=float)
    return [
        ScaledColumn(f"{name}:recency", scale, recency_column(values)),
        ScaledColumn(f"{name}:geometric{decay:g}", scale, geometric_lag(values, decay)),
    ]


def assert_carries_detail(
    column: ScaledColumn, projector: BlockProjector, tol: float = 1e-12
) -> None:
    """
    Refuse a column that a regime effect would ride on and that a block averages away.

    Raises:
        ValueError: when the column is constant across the children of every
            parent block, in which case the scale it sits at cannot identify a
            coefficient on it.
    """
    if projector.is_block_constant(column.values, tol):
        raise ValueError(
            f"{column.name!r} is constant across the children of every parent "
            "block, so a coefficient on it contributes nothing at this scale"
        )


def carrier_terms(
    columns: Sequence[ScaledColumn], projector: BlockProjector, tol: float = 1e-12
) -> List[ScaledColumn]:
    """The columns of a set that a regime effect can ride on at this scale."""
    return [c for c in columns if not projector.is_block_constant(c.values, tol)]
