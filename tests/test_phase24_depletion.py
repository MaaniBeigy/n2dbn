"""
tests/test_phase24_depletion.py
===============================
The depletion term of the allocation score: what earlier children of a block
have already taken of their parent's total.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from src.core.blocks import BlockIndex
from src.core.graph import MultiScaleDBNGraph
from src.learning.estep import EStep, fine_scores
from src.learning.mstep import MStep
from src.synthetic.families import make_generator
from src.synthetic.scenarios import ScenarioConfig


def _cfg(**over) -> ScenarioConfig:
    base = dict(
        name="dep",
        K=2,
        H_finest=120,
        branching=[10],
        L=3,
        M_k=[2, 2],
        r_k=[0, 0],
        max_lag=[2, 1],
        max_indegree=[0, 1],
        zi_level="medium",
        family=2,
        seed=4,
        n_replicates=1,
        edge_strength=10.0,
        regime_strength=3.0,
        finest_rate=0.6,
        finest_binary=False,
        slot_minutes=1,
    )
    base.update(over)
    return ScenarioConfig(**base)


@pytest.fixture(scope="module")
def dataset():
    cfg = _cfg()
    gen = make_generator(cfg, seed=4)
    ds = gen.generate(seed=4)
    graph = MultiScaleDBNGraph(hierarchy=ds.hierarchy, label_names=gen.label_names)
    for ki, p in enumerate(ds.params):
        for dst, parents in p.parent_sets.items():
            for src, lag in parents:
                graph.add_edge(ki + 1, src, lag, dst)
    return cfg, ds, graph


# ---------------------------------------------------------------------------
# The segment reduction
# ---------------------------------------------------------------------------


def test_exclusive_cumsum_sums_only_the_earlier_epochs_of_a_block():
    block = BlockIndex.from_parent_map(np.array([0, 0, 0, 1, 1, 2]), 3)
    arr = np.arange(6.0).reshape(6, 1)
    assert block.exclusive_cumsum(arr).ravel().tolist() == [0, 0, 1, 0, 3, 0]


def test_exclusive_cumsum_follows_the_block_order_not_the_row_order():
    """A parent map that is not sorted still accumulates within each block."""
    block = BlockIndex.from_parent_map(np.array([1, 0, 1, 0, 2]), 3)
    arr = np.array([[10.0], [1.0], [20.0], [2.0], [5.0]])
    assert block.exclusive_cumsum(arr).ravel().tolist() == [0, 0, 10, 1, 0]


def test_exclusive_cumsum_steps_over_an_empty_block():
    block = BlockIndex.from_parent_map(np.array([0, 0, 2, 2]), 3)
    arr = np.arange(4.0).reshape(4, 1)
    assert block.exclusive_cumsum(arr).ravel().tolist() == [0, 0, 0, 2]


def test_exclusive_cumsum_is_the_running_total_minus_the_row():
    """Within one block the result is the plain exclusive running total."""
    block = BlockIndex.from_parent_map(np.zeros(7, dtype=int), 1)
    arr = np.arange(7.0).reshape(7, 1)
    expected = np.cumsum(arr, axis=0) - arr
    assert np.array_equal(block.exclusive_cumsum(arr), expected)


# ---------------------------------------------------------------------------
# The parameters
# ---------------------------------------------------------------------------


def test_generated_parameters_start_with_no_depletion_effect(dataset):
    """The generator is unchanged, so a generated dataset is bit-identical."""
    _, ds, _ = dataset
    fine = ds.params[0]
    assert np.array_equal(fine.fine_count.betaDep, np.zeros(fine.L))
    assert np.array_equal(fine.fine_dur.alphaDep, np.zeros(fine.L))


def test_a_depletion_coefficient_moves_the_allocation_score(dataset):
    cfg, ds, _ = dataset
    params = deepcopy(ds.params[0])
    block = BlockIndex.from_hierarchy(ds.hierarchy, 1)
    X_k, Y_k = ds.data.get_X(1), ds.data.get_Y(1)
    X_par, Y_par = ds.data.get_X(2), ds.data.get_Y(2)
    ctx = ds.data.get_C(1)
    zero = np.zeros((X_k.shape[0], cfg.L))
    h = np.zeros((X_k.shape[0], max(params.r, 0)))

    args = (X_k, X_par, Y_par, h, ctx)
    eta0, xi0 = fine_scores(*args, params, block, zero, zero)
    params.fine_count.betaDep = np.full(cfg.L, -0.5)
    params.fine_dur.alphaDep = np.full(cfg.L, 0.25)
    eta1, xi1 = fine_scores(*args, params, block, zero, zero)

    dep = np.log1p(block.exclusive_cumsum(X_k.astype(float)))
    assert np.any(dep > 0), "the fixture must have blocks where an earlier child fired"
    assert np.allclose(eta1 - eta0, (-0.5 * dep)[:, None, :])
    assert np.allclose(xi1 - xi0, (0.25 * dep)[:, None, :])


def test_the_parameter_step_learns_a_depletion_coefficient(dataset):
    """The block competes for a fixed total, so the fitted effect is not zero."""
    cfg, ds, graph = dataset
    q = EStep().run(1, ds.data, ds.params[0], graph, None)
    new = MStep(n_steps_emission=40).run(
        k=1, q=q, params=deepcopy(ds.params[0]), data=ds.data, hier=ds.hierarchy
    )
    assert np.any(np.abs(new.fine_count.betaDep) > 1e-6)


def test_the_depletion_term_raises_the_allocation_likelihood(dataset):
    """Competition is real structure, so representing it cannot cost fit."""
    from src.learning.mstep import allocation_ll

    cfg, ds, graph = dataset
    q = EStep().run(1, ds.data, ds.params[0], graph, None)
    block = BlockIndex.from_hierarchy(ds.hierarchy, 1)
    args = (
        q,
        ds.data.get_X(1),
        ds.data.get_Y(1),
        ds.data.get_X(2),
        ds.data.get_Y(2),
        ds.data.get_C(1),
    )
    base = allocation_ll(*args, ds.params[0], block)
    fitted = MStep(n_steps_emission=40).run(
        k=1, q=q, params=deepcopy(ds.params[0]), data=ds.data, hier=ds.hierarchy
    )
    assert allocation_ll(*args, fitted, block) >= base
