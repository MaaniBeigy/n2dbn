"""
Tests for routing one column to the band each resolution identifies.

A column is written as a sum of bands plus a level, and the two properties that
make the decomposition worth having are that it adds back exactly and that no
band is visible to a coarser scale than its own.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.learning.fine_design import ScaleLadder
from src.learning.fine_sstep import ScaledColumn
from src.learning.scale_split import (
    band_decomposition,
    parent_map,
    projector_at,
    reconstruct,
    route,
    split_column,
)
from src.model.intensity import FineSources, ancestor_maps

T = 24
BRANCHING = (4, 3)  # 24 slots, 6 epochs, 2 epochs


@pytest.fixture
def ladder() -> ScaleLadder:
    maps = [np.arange(T) // 4, np.arange(6) // 3]
    ancestors = ancestor_maps(maps, T)
    counts = [np.zeros((int(a.max()) + 1, 2)) for a in ancestors]
    sources = FineSources(
        counts=counts,
        durations=[c.copy() for c in counts],
        calendar=[np.zeros((c.shape[0], 0)) for c in counts],
        ancestors=ancestors,
    )
    return ScaleLadder.from_sources(sources)


def test_the_ladder_nests_as_declared(ladder):
    assert ladder.K == 3
    assert ladder.n_epochs == [24, 6, 2]
    assert np.array_equal(parent_map(ladder, 1), np.arange(24) // 4)
    assert np.array_equal(parent_map(ladder, 2), np.arange(6) // 3)


def test_the_coarsest_scale_has_no_parent_on_the_ladder(ladder):
    with pytest.raises(ValueError, match="no parent"):
        parent_map(ladder, 3)
    with pytest.raises(ValueError, match="no parent"):
        parent_map(ladder, 0)


def test_the_coarsest_projector_is_the_whole_record(ladder):
    top = projector_at(ladder, ladder.K)
    assert top.n_blocks == 1
    assert top.size == 2
    values = np.array([3.0, 7.0])
    assert np.allclose(top.mean(values), [5.0])


def test_a_split_column_reconstructs_and_the_shadow_sits_one_rung_up(ladder):
    rng = np.random.default_rng(0)
    column = ScaledColumn("walking.count", 1, rng.normal(size=24))
    piece = split_column(column, ladder)
    assert piece.scale == 1
    assert piece.shadow.scale == 2
    assert piece.shadow.values.size == 6
    projector = projector_at(ladder, 1)
    assert np.allclose(
        piece.detail.values + projector.expand(piece.shadow.values), column.values
    )


def test_the_coarsest_shadow_is_one_number_naming_the_horizon(ladder):
    column = ScaledColumn("week", 3, np.array([2.0, 6.0]))
    piece = split_column(column, ladder)
    assert piece.shadow.scale == ladder.K + 1
    assert piece.shadow.values.tolist() == [4.0]
    assert piece.detail.values.tolist() == [-2.0, 2.0]


@pytest.mark.parametrize("scale", [1, 2, 3])
def test_the_bands_and_the_level_add_back_to_the_column(ladder, scale):
    rng = np.random.default_rng(scale)
    values = rng.normal(size=ladder.n_epochs[scale - 1])
    column = ScaledColumn("source", scale, values)
    bands, level = band_decomposition(column, ladder)

    assert sorted(bands) == list(range(scale, ladder.K + 1))
    assert np.allclose(
        reconstruct(bands, level, ladder), column.at_finest(ladder.ancestors)
    )
    assert level == pytest.approx(float(np.mean(values)))


def test_a_band_averages_to_zero_inside_every_parent_block(ladder):
    rng = np.random.default_rng(4)
    column = ScaledColumn("source", 1, rng.normal(size=24))
    bands, _ = band_decomposition(column, ladder)
    for scale, band in bands.items():
        assert np.allclose(projector_at(ladder, scale).mean(band.values), 0.0)


def test_the_bands_are_orthogonal_at_the_finest_grid(ladder):
    rng = np.random.default_rng(5)
    column = ScaledColumn("source", 1, rng.normal(size=24))
    bands, _ = band_decomposition(column, ladder)
    fine = {k: b.at_finest(ladder.ancestors) for k, b in bands.items()}
    for j in fine:
        for k in fine:
            if j < k:
                assert abs(float(fine[j] @ fine[k])) < 1e-12


def test_routing_gathers_every_band_under_the_scale_that_owns_it(ladder):
    rng = np.random.default_rng(6)
    columns = [
        ScaledColumn("hour", 1, rng.normal(size=24)),
        ScaledColumn("day", 2, rng.normal(size=6)),
    ]
    routed = route(columns, ladder)
    assert sorted(routed) == [1, 2, 3]
    # The hour column contributes to all three scales; the day column to two,
    # since a column carries no band below the scale it sits at.
    assert [len(routed[k]) for k in (1, 2, 3)] == [1, 2, 2]
