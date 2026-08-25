"""
Tests for the block-constant projector and the two components it separates.

The load-bearing one is `test_the_child_kernel_ignores_a_block_constant_shift`.
The whole account of what each resolution identifies rests on the child kernel
being blind to its own block-constant predictor, and that test measures it on
the kernel the project actually scores.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.blocks import BlockIndex
from src.core.projection import BlockProjector, ScaleBroadcast, invariance_residual
from src.learning.estep import block_softmax, dirmult_block_ll


def parents(sizes):
    """A parent map from the number of children each block holds."""
    return np.repeat(np.arange(len(sizes)), sizes)


@pytest.fixture
def projector():
    return BlockProjector.from_parent_map(parents([3, 2, 4]), 3)


def test_the_split_reconstructs_and_the_two_parts_are_orthogonal(projector):
    rng = np.random.default_rng(0)
    values = rng.normal(size=projector.size)
    shadow, detail = projector.split(values)
    assert np.allclose(shadow + detail, values)
    assert abs(float(shadow @ detail)) < 1e-12


def test_the_shadow_is_idempotent_and_the_detail_averages_to_zero(projector):
    rng = np.random.default_rng(1)
    values = rng.normal(size=projector.size)
    shadow = projector.shadow(values)
    assert np.allclose(projector.shadow(shadow), shadow)
    assert np.allclose(projector.mean(projector.detail(values)), 0.0)


def test_a_block_constant_vector_is_all_shadow(projector):
    per_block = np.array([1.5, -2.0, 0.25])
    values = projector.expand(per_block)
    assert projector.is_block_constant(values)
    assert np.allclose(projector.detail(values), 0.0)
    assert np.allclose(projector.mean(values), per_block)


def test_the_split_runs_over_the_leading_axis(projector):
    rng = np.random.default_rng(2)
    values = rng.normal(size=(projector.size, 4))
    shadow, detail = projector.split(values)
    assert shadow.shape == values.shape
    assert np.allclose(shadow + detail, values)
    assert np.allclose(projector.mean(detail), 0.0)


def test_an_empty_block_carries_a_mean_of_zero():
    # Block 1 has no child, so nothing averages into it and nothing reads it.
    projector = BlockProjector.from_parent_map(np.array([0, 0, 2, 2]), 3)
    values = np.array([1.0, 3.0, 5.0, 7.0])
    assert np.allclose(projector.mean(values), [2.0, 0.0, 6.0])


def test_a_wrong_length_array_is_refused(projector):
    with pytest.raises(ValueError, match="one row per epoch"):
        projector.mean(np.ones(projector.size + 1))
    with pytest.raises(ValueError, match="one row per epoch"):
        projector.mean(np.float64(1.0))
    with pytest.raises(ValueError, match="one entry per block"):
        projector.expand(np.ones(projector.n_blocks + 1))


# ---------------------------------------------------------------------------
# The falsification test of the whole account
# ---------------------------------------------------------------------------


def _kernel(counts, block, concentration):
    """The child kernel's log density as a function of its own predictor."""

    def log_density(eta):
        return dirmult_block_ll(counts, block_softmax(eta, block), concentration, block)

    return log_density


@pytest.mark.parametrize("sizes", [[4, 4, 4], [2, 5, 3, 6], [7, 1, 2]])
def test_the_child_kernel_ignores_a_block_constant_shift(sizes):
    rng = np.random.default_rng(7)
    block = BlockIndex.from_parent_map(parents(sizes), len(sizes))
    projector = BlockProjector(block)
    eta = rng.normal(size=(block.parent.size, 3))
    counts = rng.poisson(2.0, size=eta.shape).astype(float)
    offsets = rng.normal(scale=5.0, size=(len(sizes), 3))

    residual = invariance_residual(
        _kernel(counts, block, 12.0), eta, projector, offsets
    )
    assert residual < 1e-9


def test_a_within_block_shift_does_move_the_child_kernel():
    rng = np.random.default_rng(8)
    sizes = [4, 4]
    block = BlockIndex.from_parent_map(parents(sizes), len(sizes))
    projector = BlockProjector(block)
    eta = rng.normal(size=(block.parent.size, 2))
    counts = rng.poisson(3.0, size=eta.shape).astype(float)
    log_density = _kernel(counts, block, 12.0)

    detail = projector.detail(rng.normal(size=eta.shape))
    assert abs(log_density(eta + detail) - log_density(eta)) > 1e-3


# ---------------------------------------------------------------------------
# The coarse-to-fine operator
# ---------------------------------------------------------------------------


def test_a_parent_effect_broadcast_and_reaggregated_comes_back(projector):
    rng = np.random.default_rng(3)
    parent = rng.normal(size=projector.n_blocks)
    for kwargs in ({}, dict(sd_child=2.0, sd_parent=0.5, lag_child=2, lag_parent=7)):
        operator = ScaleBroadcast(projector, **kwargs)
        assert np.allclose(operator.to_parent(operator.to_child(parent)), parent)


def test_the_broadcast_lands_in_the_block_constant_subspace(projector):
    operator = ScaleBroadcast(projector, sd_child=3.0, sd_parent=1.5)
    child = operator.to_child(np.array([1.0, -1.0, 4.0]))
    assert projector.is_block_constant(child)
    assert np.allclose(projector.shadow(child), child)
    assert operator.factor == pytest.approx(2.0)


def test_a_broadcast_refuses_units_it_cannot_convert(projector):
    with pytest.raises(ValueError, match="standard deviation is positive"):
        ScaleBroadcast(projector, sd_child=0.0)
    with pytest.raises(ValueError, match="at least one epoch"):
        ScaleBroadcast(projector, lag_parent=0)
