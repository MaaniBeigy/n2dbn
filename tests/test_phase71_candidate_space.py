"""
Tests for the candidate space, where an enumeration and a design have to agree.

The property that matters is that the edge at position i and the column at
position i are the same reading of the same source, since a gain scored on one
and realized on the other is a gain that does not transfer.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.projection import BlockProjector
from src.core.typed_graph import COUNT, LOG_TERM, OBSERVED, read_basis
from src.learning.candidate_space import (
    DEFAULT_DECAY,
    assert_carries_detail,
    basis_candidates,
    basis_expansion,
    carrier_terms,
    derived_columns,
)
from src.learning.fine_sstep import ScaledColumn, quantile_edges


@pytest.fixture
def source():
    rng = np.random.default_rng(0)
    return rng.poisson(3.0, size=48).astype(float)


def test_the_expansion_offers_the_log_term_and_one_band_per_bin(source):
    edges = basis_expansion(source, 2, 1, COUNT, lag=3)
    assert edges[0].basis == LOG_TERM
    assert len(edges) == len(quantile_edges(source, 4)) - 1
    assert {e.src_scale for e in edges} == {2}
    assert {e.src_label for e in edges} == {1}
    assert {e.lag for e in edges} == {3}
    assert {e.src_kind for e in edges} == {OBSERVED}
    # Every band is a half-open interval and the bands do not overlap.
    bands = [e.basis for e in edges[1:]]
    assert all(lo < hi for lo, hi in bands)
    assert [lo for lo, _ in bands] == sorted(lo for lo, _ in bands)


def test_the_edge_and_the_column_at_one_position_are_one_reading(source):
    edges, columns = basis_candidates(source, 2, 1, COUNT, 3, "walking.count")
    assert len(edges) == len(columns)
    for edge, column in zip(edges, columns):
        assert column.scale == edge.src_scale
        assert np.allclose(column.values, read_basis(source, edge.basis))


def test_a_source_with_no_spread_carries_the_log_term_alone():
    flat = np.zeros(20)
    edges, columns = basis_candidates(flat, 1, 0, COUNT, 1, "flat")
    assert [e.basis for e in edges] == [LOG_TERM]
    assert len(columns) == 1


def test_the_two_derived_columns_are_named_after_what_they_read():
    events = np.array([0.0, 1.0, 0.0, 0.0, 2.0, 0.0])
    columns = derived_columns(events, 2, "sleep.count")
    assert [c.name for c in columns] == [
        "sleep.count:recency",
        f"sleep.count:geometric{DEFAULT_DECAY:g}",
    ]
    assert {c.scale for c in columns} == {2}
    assert all(c.values.size == events.size for c in columns)


def test_a_carrier_that_a_block_averages_away_is_refused():
    projector = BlockProjector.from_parent_map(np.arange(12) // 3, 4)
    flat = ScaledColumn("regime.level", 1, np.repeat([1.0, 0.0, 1.0, 0.0], 3))
    with pytest.raises(ValueError, match="constant across the children"):
        assert_carries_detail(flat, projector)


def test_a_carrier_with_a_within_block_contrast_is_admitted():
    projector = BlockProjector.from_parent_map(np.arange(12) // 3, 4)
    varying = ScaledColumn("recency", 1, np.arange(12, dtype=float))
    assert assert_carries_detail(varying, projector) is None


def test_the_carriers_of_a_set_are_the_columns_a_block_does_not_average_away():
    projector = BlockProjector.from_parent_map(np.arange(12) // 3, 4)
    columns = [
        ScaledColumn("flat", 1, np.repeat([1.0, 0.0, 1.0, 0.0], 3)),
        ScaledColumn("rising", 1, np.arange(12, dtype=float)),
    ]
    kept = carrier_terms(columns, projector)
    assert [c.name for c in kept] == ["rising"]
