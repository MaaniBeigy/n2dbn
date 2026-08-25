"""
Sequential in-block allocation.

The joint form scores every child of a block before any is drawn, so a lagged
score reading a sibling inside the same block reads a value that has not been
written. Self edges are the worst case and were roughly absent from the data
generated for them. These tests pin the sequential draw that fixes it.

1.  `sequential_split` gives the whole remainder to the last child
2.  `sequential_split` returns nothing when the block has nothing left
3.  `sequential_split` never exceeds the remaining budget
4.  `sequential_split` follows the allocation probability on average
5.  `allocate_counts_block_sequential` preserves the block total
6.  `allocate_counts_block_sequential` writes each child before the next
7.  `allocate_counts_block_sequential` returns zeros for an empty block
8.  `_needs_sequential` fires only for a lag that lands inside the block
9.  a true self edge leaves a lagged signal in the generated data
10. reconciliation and the binary cap survive the sequential path
"""

import numpy as np
import pytest

from src.evaluation.metrics import ReconciliationMetrics
from src.model.allocation import (
    allocate_counts_block_sequential,
    sequential_split,
)
from src.synthetic.families import make_generator
from src.synthetic.generator import _needs_sequential
from src.synthetic.scenarios import ScenarioGrid

# ---------------------------------------------------------------------------
# 1-4.  sequential_split
# ---------------------------------------------------------------------------


def test_sequential_split_gives_the_last_child_the_remainder():
    """With no allocation probability left, the block has one place to put it."""
    rng = np.random.default_rng(0)
    assert sequential_split(7, 1.0, 0.0, 50.0, rng) == 7


def test_sequential_split_of_an_exhausted_block_is_zero():
    """A block with nothing left allocates nothing, whatever the weights."""
    rng = np.random.default_rng(0)
    assert sequential_split(0, 0.5, 0.5, 50.0, rng) == 0
    assert sequential_split(-3, 0.5, 0.5, 50.0, rng) == 0


def test_sequential_split_never_exceeds_the_budget():
    """A child cannot take more than the block still holds."""
    rng = np.random.default_rng(1)
    for _ in range(200):
        assert 0 <= sequential_split(12, 0.9, 0.1, 5.0, rng) <= 12


def test_sequential_split_follows_its_allocation_probability():
    """Averaged over draws, a child takes its share of what is left."""
    rng = np.random.default_rng(2)
    draws = [sequential_split(100, 0.25, 0.75, 500.0, rng) for _ in range(400)]
    assert 20.0 < float(np.mean(draws)) < 30.0


# ---------------------------------------------------------------------------
# 5-7.  allocate_counts_block_sequential
# ---------------------------------------------------------------------------


class _Params:
    """Minimal count parameters: one lagged self edge, everything else flat."""

    def __init__(self, L=2, coef=1.5):
        self.L = L
        self.kappa = 50.0
        self.beta0 = np.zeros(L)
        self.betaP = np.zeros(L)
        self.betaD = np.zeros(L)
        self.betaS = np.zeros((1, L))
        self.betaH = np.zeros((0, L))
        self.betaCtx = np.zeros((0, L))
        self.betaLag = {(0, 1): np.full(L, coef)}
        self.betaDep = np.zeros(L)


def _fixture(n_children=5, L=2):
    children = list(range(n_children))
    return dict(
        children=children,
        ell=0,
        Y_parent=0,
        s_seq=np.zeros(n_children, dtype=int),
        H_traj=np.zeros((n_children, 0)),
        ctx=np.zeros((n_children, 0)),
        X_fine=np.zeros((n_children, L), dtype=np.int64),
        Y_fine=np.zeros((n_children, L), dtype=np.int64),
        p=_Params(L=L),
        parent_sets_ell={(0, 1)},
    )


def test_sequential_allocation_preserves_the_block_total():
    """Every event of the block lands on exactly one child."""
    kw = _fixture()
    alloc = allocate_counts_block_sequential(
        X_parent=20, rng=np.random.default_rng(0), write=lambda t, v: None, **kw
    )
    assert alloc.sum() == 20


def test_sequential_allocation_writes_each_child_before_the_next():
    """The write callback fires per child, in time order, before the next draw."""
    kw = _fixture()
    seen = []
    allocate_counts_block_sequential(
        X_parent=15,
        rng=np.random.default_rng(0),
        write=lambda t, v: seen.append(t),
        **kw,
    )
    assert seen == sorted(seen)
    assert seen[0] == kw["children"][0]


def test_sequential_allocation_of_an_empty_block():
    """A block holding no events allocates none."""
    kw = _fixture()
    alloc = allocate_counts_block_sequential(
        X_parent=0, rng=np.random.default_rng(0), write=lambda t, v: None, **kw
    )
    assert alloc.sum() == 0 and alloc.shape == (5,)


# ---------------------------------------------------------------------------
# 8.  the gate
# ---------------------------------------------------------------------------


def test_needs_sequential_fires_only_for_a_lag_inside_the_block():
    """A lag longer than the block can only read a sibling of an earlier one."""
    assert _needs_sequential({(0, 1)}, 10) is True
    assert _needs_sequential({(0, 9)}, 10) is True
    assert _needs_sequential({(0, 10)}, 10) is False
    assert _needs_sequential(set(), 10) is False


# ---------------------------------------------------------------------------
# 9-10.  end to end on the generator
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def generated():
    """A scenario whose ground truth contains a lagged self edge."""
    grid = {c.name: c for c in ScenarioGrid.full_benchmark().scenarios}
    cfg = grid["F1_L5_zil"]
    seed = cfg.seed * 1000
    return make_generator(cfg, seed=seed).generate(seed=seed)


def test_a_true_self_edge_leaves_a_lagged_signal(generated):
    """An edge that is in the ground truth has to be in the data it generated."""
    params = generated.params[1]
    X = np.asarray(generated.data.X[1], dtype=float)
    self_edges = [
        (dst, src, lag)
        for dst, ps in params.parent_sets.items()
        for src, lag in ps
        if src == dst
    ]
    if not self_edges:
        pytest.skip("this scenario drew no self edge")

    for dst, src, lag in self_edges:
        a, b = X[lag:, dst], X[: -lag or None, src]
        if a.std() < 1e-9 or b.std() < 1e-9:
            continue
        # The joint draw left this at 0.021, which is indistinguishable from
        # nothing; the sequential draw has to put the edge into the data.
        assert abs(float(np.corrcoef(a, b)[0, 1])) > 0.05


def test_sequential_path_keeps_reconciliation_and_the_binary_cap(generated):
    """The fix must not disturb the two invariants every fit is checked on."""
    got = ReconciliationMetrics.compute(generated.data).to_dict()
    assert got["RecErr_X"] == 0.0 and got["RecErr_Y"] == 0.0
    assert int(np.asarray(generated.data.X[0]).max()) == 1
