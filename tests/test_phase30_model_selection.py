"""
Model selection on held-out log evidence.

The regime count, the factor dimension and the edge penalty are free
quantities of this model.  Declaring them makes any comparison against a
learner that chooses its own penalty a comparison of declarations, so all
three are chosen from data by one held-out rule.

1.  `held_out_nll` is finite and matches a direct sum of scale log evidence
2.  `held_out_nll` grows when the parameters are moved away from the fit
3.  `select_model` returns the order with the lowest held-out NLL
4.  `select_model` records every candidate in its table
5.  `select_model` selects one regime on regime-free data
6.  `select_model` rejects an empty grid
7.  `select_model` survives a candidate that fails to fit
8.  `select_model` raises when no candidate can be fitted
9.  `ModelSelection.to_dict` is a flat record
"""

import numpy as np
import pytest

from src.core.graph import MultiScaleDBNGraph
from src.core.scales import ScaleHierarchy
from src.core.variables import MultiScaleData
from src.data.initialize import initialize_from_data
from src.learning.estep import EStep
from src.learning.model_selection import (
    ModelSelection,
    held_out_nll,
    select_model,
)
from src.learning.structural_em import TopDownSEM


def _flat_data(T=240, L=3, b=8, seed=0):
    """Counts with no persistent regime, so one regime is the true order."""
    rng = np.random.default_rng(seed)
    n = (T // b) * b
    Xf = rng.poisson(2.0, size=(n, L)).astype(np.int64)
    coarse = Xf.reshape(n // b, b, L).sum(axis=1)
    return MultiScaleData(
        hierarchy=ScaleHierarchy.from_uniform_branching(n, [b]),
        label_names=[f"v{i}" for i in range(L)],
        X=[Xf, coarse],
        Y=[Xf.copy(), coarse.copy()],
        C=[np.zeros((n, 0)), np.zeros((n // b, 0))],
    )


def _sem(lam=2.0):
    return TopDownSEM(
        max_outer_iter=1,
        lam_edge=lam,
        sstep_kwargs=dict(max_lag={1: 1, 2: 1}, max_indegree={1: 1, 2: 1}),
        verbose=0,
    )


@pytest.fixture(scope="module")
def flat():
    return _flat_data()


# ---------------------------------------------------------------------------
# 1-2.  held_out_nll
# ---------------------------------------------------------------------------


def test_held_out_nll_matches_summed_log_evidence(flat):
    """The score is minus the sum of per-scale log evidence, top down."""
    params = initialize_from_data(
        flat, M_k=[1, 1], r_k=[0, 0], max_lag=[1, 1], d_ctx=0, seed=0
    )
    graph = MultiScaleDBNGraph(hierarchy=flat.hierarchy, label_names=flat.label_names)

    estep = EStep()
    summaries, direct = [None, None], 0.0
    for k in (2, 1):
        s = estep.run(k, flat, params[k - 1], graph, summaries[k] if k < 2 else None)
        summaries[k - 1] = s
        direct += s.log_Z

    got = held_out_nll(flat, params, graph)
    assert np.isfinite(got)
    assert got == pytest.approx(-direct, rel=1e-9)


def test_held_out_nll_penalises_worse_parameters(flat):
    """Moving the emission level away from the data raises the score."""
    params = initialize_from_data(
        flat, M_k=[1, 1], r_k=[0, 0], max_lag=[1, 1], d_ctx=0, seed=0
    )
    graph = MultiScaleDBNGraph(hierarchy=flat.hierarchy, label_names=flat.label_names)
    good = held_out_nll(flat, params, graph)

    from copy import deepcopy

    spoiled = deepcopy(params)
    spoiled[1].coarse_count.beta0 -= 4.0
    assert held_out_nll(flat, spoiled, graph) > good


# ---------------------------------------------------------------------------
# 3-8.  select_model
# ---------------------------------------------------------------------------


def test_selection_returns_the_lowest_scoring_order(flat):
    """The reported order is the argmin of the reported table."""
    sel = select_model(
        flat, flat, _sem, M_grid=(1, 2), r_grid=(0,), max_lag=[1, 1], seed=0
    )
    best = min(sel.table, key=lambda kv: kv[1])
    assert list(best[0][0]) == sel.M_k
    assert best[0][2] == sel.lam
    assert sel.val_nll == pytest.approx(best[1])


def test_selection_records_every_candidate(flat):
    """One row per grid point, in grid order."""
    sel = select_model(
        flat,
        flat,
        _sem,
        M_grid=(1, 2),
        r_grid=(0, 1),
        lam_grid=(2.0, 1.0),
        max_lag=[1, 1],
        seed=0,
    )
    assert len(sel.table) == 8
    assert [row[0][0][0] for row in sel.table] == [1, 1, 1, 1, 2, 2, 2, 2]


def test_selection_picks_the_lowest_scoring_penalty(flat):
    """The penalty is chosen by the same held-out rule as the latent orders."""
    sel = select_model(
        flat,
        flat,
        _sem,
        M_grid=(1,),
        r_grid=(0,),
        lam_grid=(4.0, 1.0),
        max_lag=[1, 1],
        seed=0,
    )
    assert sel.lam == min(sel.table, key=lambda kv: kv[1])[0][2]


def test_selection_picks_one_regime_on_regime_free_data(flat):
    """Data generated without a regime must not buy a second one."""
    sel = select_model(
        flat, flat, _sem, M_grid=(1, 2), r_grid=(0,), max_lag=[1, 1], seed=0
    )
    assert sel.M_k == [1, 1]


def test_selection_rejects_empty_grid(flat):
    """An empty grid names no candidate, which is a caller error."""
    with pytest.raises(ValueError, match="must each hold a value"):
        select_model(flat, flat, _sem, M_grid=(), r_grid=(0,))


def test_selection_survives_a_failing_candidate(flat):
    """One unfittable order is recorded as infinite and does not stop the sweep."""

    class _FailsOnTwo:
        def __init__(self, M):
            self.M = M

        def fit(self, train, params, graph):
            if params[0].M == 2:
                raise RuntimeError("candidate blew up")
            return _sem().fit(train, params, graph)

    orders = iter([1, 2])
    sel = select_model(
        flat,
        flat,
        lambda lam: _FailsOnTwo(next(orders)),
        M_grid=(1, 2),
        r_grid=(0,),
        max_lag=[1, 1],
        seed=0,
    )
    assert sel.M_k == [1, 1]
    assert sel.table[1][1] == float("inf")


def test_selection_raises_when_nothing_fits(flat):
    """No finite candidate means no selection to report."""

    class _AlwaysFails:
        def fit(self, *_):
            raise RuntimeError("no")

    with pytest.raises(ValueError, match="no configuration"):
        select_model(
            flat,
            flat,
            lambda lam: _AlwaysFails(),
            M_grid=(1,),
            r_grid=(0,),
            max_lag=[1, 1],
        )


# ---------------------------------------------------------------------------
# 9.  ModelSelection
# ---------------------------------------------------------------------------


def test_order_selection_to_dict():
    """The record carries the orders, the score and the candidate count."""
    sel = ModelSelection(
        M_k=[1, 1],
        r_k=[0, 0],
        lam=2.0,
        val_nll=12.5,
        table=[(((1,), (0,), 2.0), 12.5)],
    )
    assert sel.to_dict() == {
        "M_k": [1, 1],
        "r_k": [0, 0],
        "lam": 2.0,
        "val_nll": 12.5,
        "n_candidates": 1,
    }
