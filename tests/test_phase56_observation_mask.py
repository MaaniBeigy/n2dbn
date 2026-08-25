"""
tests/test_phase56_observation_mask.py
======================================
The observation mask, and the forward dilation that keeps a gap out of a score.

Five claims carry this file.

A mask of all ones changes nothing.  The default is no mask at all, and the two
have to agree number for number at every step of both formulations, or every
stored result moves the day the field arrives.

A masked epoch contributes no emission term.  The coarsest-scale emission row
comes out at zero, which is the emission-equals-one rule `forward_backward`
already implements; the finer-scale epoch leaves the block competition and
carries a zero count, which zeroes its own row and its gradient.  Both are
checked against the objective summed by hand over the observed epochs.

Provenance and scorability stay two statements.  A block sum is over what the
record holds, and a likelihood term is summed over the rows the declared budget
leaves scorable.  The slot right after a gap belongs to the first and not to the
second, so it keeps its count in its block and contributes no term, and the
sum is additive over any split of the scorable rows.

A lagged feature that reads into a gap is a feature nobody recorded, so the row
that reads it leaves the score.  The case here is an edge whose whole apparent
gain comes from the epoch after a gap: scored on the raw mask it clears the
price and enters the graph, scored on the dilated mask it earns nothing.

The mask and the risk set stay two statements.  A running label is a slot the
model explains with a start probability of zero; a masked slot is one there is
nothing to explain.  They are conjoined at `intensity.usable_slots` and the two
exclusions remain separately recoverable from what it returns.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from src.core.graph import MultiScaleDBNGraph
from src.core.mask import (
    check_ladder,
    declared_rows,
    dilate_forward,
    lag_budget,
    observed_ladder,
    parent_set_lag,
    reduce_by,
    scorable_rows,
)
from src.core.scales import ScaleHierarchy
from src.core.typed_graph import COUNT, OBSERVED, TypedEdge, TypedGraph
from src.core.variables import MultiScaleData
from src.inference.bound import pg_mean
from src.inference.posterior import PosteriorSummaries
from src.learning.estep import (
    EStep,
    build_log_like_coarsest,
    build_log_like_fine,
    rows_of,
)
from src.learning.fine_design import CalendarIndex, ScaleLadder, build_design
from src.learning.fine_sem import FineSEM, RegimeChain, fit_rows
from src.learning.fine_sstep import BaseFit, greedy_search
from src.learning.mstep import (
    MStep,
    allocation_ll,
    coarse_emission_ll,
    update_coarse_count_params,
)
from src.learning.sstep import LocalScore, SStep
from src.model.intensity import (
    FineParams,
    ancestor_maps,
    empty_sources,
    mark_loglik,
    risk_set,
    total_loglik,
    usable_slots,
)
from src.model.params import (
    CoarseCountParams,
    CoarseDurParams,
    FactorParams,
    FineCountParams,
    FineDurParams,
    RegimeParams,
    ScaleParams,
)

# A twelve-slot record laid on four blocks of three, small enough that every
# number below can be summed by hand.
H1 = 12
BRANCHING = 3
H2 = H1 // BRANCHING
L = 2
M = 2


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _hierarchy() -> ScaleHierarchy:
    return ScaleHierarchy.from_uniform_branching(H1, [BRANCHING])


def _data(observed=None) -> MultiScaleData:
    """A fixed multiscale record, reconciled by construction."""
    hier = _hierarchy()
    rng = np.random.default_rng(11)
    X1 = rng.integers(0, 3, size=(H1, L)).astype(np.int64)
    Y1 = (X1 * rng.integers(1, 4, size=(H1, L))).astype(np.int64)
    pm = hier.parent_maps[0]
    X2 = np.zeros((H2, L), dtype=np.int64)
    Y2 = np.zeros((H2, L), dtype=np.int64)
    np.add.at(X2, pm, X1)
    np.add.at(Y2, pm, Y1)
    return MultiScaleData(
        hierarchy=hier,
        label_names=["a", "b"],
        X=[X1, X2],
        Y=[Y1, Y2],
        C=[np.zeros((H1, 0)), np.zeros((H2, 0))],
        observed=observed,
    )


def _regime() -> RegimeParams:
    return RegimeParams(
        M=M,
        parent_M=0,
        d_ctx=0,
        init=np.zeros(M),
        bias=np.zeros(M),
        trans=np.array([[0.4, -0.2], [-0.2, 0.4]]),
        cross=None,
        ctx=np.zeros((M, 0)),
    )


def _factor() -> FactorParams:
    return FactorParams(
        M=M,
        r=0,
        d_ctx=0,
        F=np.zeros((M, 0, 0)),
        G=np.zeros((M, 0, 0)),
        Sigma=np.zeros((M, 0, 0)),
        mu0=np.zeros(0),
        Sigma0=np.zeros((0, 0)),
    )


def _coarse_params() -> ScaleParams:
    count = CoarseCountParams(
        L=L,
        beta0=np.array([0.3, -0.1]),
        betaS=np.array([[0.2, -0.2], [-0.3, 0.4]]),
        betaH=np.zeros((0, L)),
        betaCtx=np.zeros((0, L)),
        betaLag={(0, 1): np.array([0.15, -0.05])},
        r_disp=np.array([2.0, 3.0]),
        gamma0=np.array([-1.0, -1.5]),
        gammaS=np.zeros((M, L)),
        gammaH=np.zeros((0, L)),
        gammaCtx=np.zeros((0, L)),
    )
    dur = CoarseDurParams(
        L=L,
        alpha0=np.array([0.8, 0.6]),
        alphaS=np.zeros((M, L)),
        alphaH=np.zeros((0, L)),
        alphaCtx=np.zeros((0, L)),
        alphaX=np.array([0.2, 0.1]),
        alphaLag={(0, 1): np.array([0.1, 0.0])},
        s_disp=np.array([2.0, 2.0]),
    )
    return ScaleParams(
        k=2,
        regime=_regime(),
        factor=_factor(),
        parent_sets={0: {(0, 1)}, 1: set()},
        coarse_count=count,
        coarse_dur=dur,
    )


def _fine_params() -> ScaleParams:
    count = FineCountParams(
        L=L,
        beta0=np.array([0.1, -0.2]),
        betaS=np.array([[0.2, 0.0], [-0.1, 0.3]]),
        betaH=np.zeros((0, L)),
        betaCtx=np.zeros((0, L)),
        betaP=np.array([0.3, 0.2]),
        betaD=np.array([0.05, 0.1]),
        betaLag={(1, 1): np.array([0.2, -0.1])},
        kappa=8.0,
    )
    dur = FineDurParams(
        L=L,
        alpha0=np.array([0.2, 0.1]),
        alphaS=np.zeros((M, L)),
        alphaH=np.zeros((0, L)),
        alphaCtx=np.zeros((0, L)),
        alphaX=np.array([0.1, 0.2]),
        alphaP=np.array([0.2, 0.1]),
        alphaLag={(1, 1): np.array([0.1, 0.0])},
        tau=8.0,
    )
    return ScaleParams(
        k=1,
        regime=_regime(),
        factor=_factor(),
        parent_sets={0: {(1, 1)}, 1: set()},
        fine_count=count,
        fine_dur=dur,
    )


def _posterior(n_epochs: int, seed: int = 3) -> PosteriorSummaries:
    rng = np.random.default_rng(seed)
    gamma = rng.random((n_epochs, M))
    gamma = gamma / gamma.sum(axis=1, keepdims=True)
    xi = np.full((n_epochs - 1, M, M), 1.0 / (M * M))
    return PosteriorSummaries(
        gamma=gamma,
        xi=xi,
        m_H=np.zeros((n_epochs, 0)),
        Sigma_H=np.zeros((n_epochs, 0, 0)),
        log_Z=0.0,
    )


# ---------------------------------------------------------------------------
# The three arrays, and what the container does with them
# ---------------------------------------------------------------------------


def test_no_mask_reads_as_fully_observed_at_every_scale() -> None:
    data = _data()
    assert data.observed is None
    assert data.get_observed(1).tolist() == [True] * H1
    assert data.get_observed(2).tolist() == [True] * H2


def test_a_coarse_epoch_is_observed_when_every_slot_under_it_is() -> None:
    flags = np.ones(H1, dtype=bool)
    flags[3:6] = False  # the whole of block 1
    flags[11] = False  # one slot of block 3
    data = _data(flags)
    assert data.get_observed(1).tolist() == flags.tolist()
    assert data.get_observed(2).tolist() == [True, False, True, False]
    # The ladder is derived once and handed back on every later call.
    assert data.get_observed(2) is data.get_observed(2)


def test_a_mask_that_is_not_boolean_is_refused() -> None:
    with pytest.raises(ValueError, match="boolean array"):
        _data(np.ones(H1, dtype=np.int64))


def test_a_mask_of_the_wrong_length_is_refused() -> None:
    with pytest.raises(ValueError, match="one flag per finest slot"):
        _data(np.ones(H1 - 1, dtype=bool))


def test_the_container_reports_how_much_of_itself_is_real() -> None:
    flags = np.ones(H1, dtype=bool)
    flags[3:6] = False
    text = repr(_data(flags))
    assert "observed=9" in text
    assert "observed=3" in text


def test_reconciliation_refuses_a_partly_observed_block() -> None:
    flags = np.ones(H1, dtype=bool)
    flags[4] = False  # one slot of block 1, leaving the other two recorded
    with pytest.raises(ValueError, match="both observed and unobserved"):
        _data(flags).check_reconciliation()


def test_reconciliation_still_holds_when_whole_blocks_are_masked() -> None:
    flags = np.ones(H1, dtype=bool)
    flags[3:6] = False
    assert _data(flags).check_reconciliation() == {"RecErr_X": 0.0, "RecErr_Y": 0.0}
    assert _data().check_reconciliation() == {"RecErr_X": 0.0, "RecErr_Y": 0.0}


def test_a_split_carries_the_mask_and_never_cuts_a_block() -> None:
    flags = np.ones(H1, dtype=bool)
    flags[0:3] = False
    train, valid, test = _data(flags).split((0.5, 0.25, 0.25))
    assert train.observed.tolist() == [False, False, False, True, True, True]
    assert valid.observed.tolist() == [True] * 3
    assert test.observed.tolist() == [True] * 3
    # No part gained a partly observed block, so each survives the same check.
    for part in (train, valid, test):
        assert part.check_reconciliation()["RecErr_X"] == 0.0


def test_a_split_part_with_nothing_observed_is_refused() -> None:
    flags = np.ones(H1, dtype=bool)
    flags[6:9] = False  # the whole of the part the middle fraction takes
    with pytest.raises(ValueError, match="valid part holds no observed"):
        _data(flags).split((0.5, 0.25, 0.25))


# ---------------------------------------------------------------------------
# The primitives
# ---------------------------------------------------------------------------


def test_the_reduction_is_an_and_over_the_children() -> None:
    flags = np.array([True, True, False, True])
    who = np.array([0, 0, 1, 1])
    assert reduce_by(flags, who, 2).tolist() == [True, False]


def test_the_ladder_stops_at_the_coarsest_scale() -> None:
    hier = _hierarchy()
    ladder = observed_ladder(np.ones(H1, dtype=bool), hier.parent_maps)
    assert [len(f) for f in ladder] == [H1, H2]


def test_a_full_mask_erodes_to_itself_whatever_the_lag() -> None:
    flags = np.ones(6, dtype=bool)
    for lag in (0, 1, 3, 9):
        assert dilate_forward(flags, lag).tolist() == [True] * 6


def test_erosion_drops_the_epochs_that_read_the_gap() -> None:
    flags = np.array([True, True, False, True, True, True])
    assert dilate_forward(flags, 0).tolist() == flags.tolist()
    assert dilate_forward(flags, 1).tolist() == [1, 1, 0, 0, 1, 1]
    assert dilate_forward(flags, 2).tolist() == [1, 1, 0, 0, 0, 1]


def test_the_lag_budget_is_the_largest_lag_read_at_each_scale() -> None:
    edges = [
        TypedEdge(OBSERVED, 1, 0, COUNT, 1),
        TypedEdge(OBSERVED, 2, 0, COUNT, 3),
        TypedEdge(OBSERVED, 2, 1, COUNT, 2),
    ]
    assert lag_budget(edges) == {1: 1, 2: 3}
    assert lag_budget([]) == {}
    assert parent_set_lag({0: {(0, 1), (1, 4)}, 1: set()}) == 4
    assert parent_set_lag({0: set()}) == 0


def test_scorable_rows_erode_at_the_scale_the_edge_reads() -> None:
    ancestors = ancestor_maps([np.repeat(np.arange(H2), BRANCHING)], H1)
    flags = np.ones(H1, dtype=bool)
    flags[3:6] = False  # block 1
    # With nothing lagged the mask stands as it is.
    assert scorable_rows(flags, ancestors, {}).tolist() == flags.tolist()
    # A coarse lag-1 edge makes block 2 unscorable as well.
    expected = flags.copy()
    expected[6:9] = False
    assert scorable_rows(flags, ancestors, {2: 1}).tolist() == expected.tolist()


def test_check_ladder_accepts_a_whole_block_and_a_full_mask() -> None:
    hier = _hierarchy()
    flags = np.ones(H1, dtype=bool)
    flags[6:9] = False
    assert check_ladder(flags, hier.parent_maps) is None
    assert check_ladder(np.ones(H1, dtype=bool), hier.parent_maps) is None


def test_rows_of_fills_in_when_no_mask_is_given() -> None:
    assert rows_of(None, 4).tolist() == [True] * 4
    given = np.array([True, False])
    assert rows_of(given, 2) is given


# ---------------------------------------------------------------------------
# A masked epoch contributes no emission term
# ---------------------------------------------------------------------------


def test_a_masked_coarse_epoch_has_a_zero_emission_row() -> None:
    data = _data()
    params = _coarse_params()
    X, Y, ctx = data.get_X(2), data.get_Y(2), data.get_C(2)
    h_bar = np.zeros((H2, 0))
    flags = np.array([True, False, True, True])

    full = build_log_like_coarsest(X, Y, h_bar, ctx, params)
    masked = build_log_like_coarsest(X, Y, h_bar, ctx, params, scored=flags)

    assert np.all(masked[~flags] == 0.0)
    np.testing.assert_array_equal(masked[flags], full[flags])


def test_the_masked_coarse_objective_is_the_hand_sum_over_observed_epochs() -> None:
    data = _data()
    params = _coarse_params()
    X, Y, ctx = data.get_X(2), data.get_Y(2), data.get_C(2)
    q = _posterior(H2)
    flags = np.array([True, False, True, True])

    full = build_log_like_coarsest(X, Y, np.zeros((H2, 0)), ctx, params)
    hand = sum(float(q.gamma[t] @ full[t]) for t in range(H2) if flags[t])
    assert coarse_emission_ll(q, X, Y, ctx, params, scored=flags) == pytest.approx(hand)
    assert coarse_emission_ll(q, X, Y, ctx, params) == pytest.approx(
        sum(float(q.gamma[t] @ full[t]) for t in range(H2))
    )


def test_a_masked_fine_epoch_leaves_its_block_and_its_row_at_zero() -> None:
    data = _data()
    params = _fine_params()
    hier = data.hierarchy
    flags = np.ones(H1, dtype=bool)
    flags[3:6] = False  # the whole of block 1

    args = (
        data.get_X(1),
        data.get_Y(1),
        data.get_X(2),
        data.get_Y(2),
        np.zeros((H1, 0)),
        data.get_C(1),
        params,
        hier,
        1,
    )
    full = build_log_like_fine(*args)
    masked = build_log_like_fine(*args, recorded=flags)

    assert np.all(masked[~flags] == 0.0)
    # A block is masked whole, so every surviving block allocates as before.
    np.testing.assert_array_equal(masked[flags], full[flags])


def test_the_allocation_objective_splits_along_the_mask() -> None:
    """A masked block's term leaves the sum and takes nothing else with it."""
    from src.core.blocks import BlockIndex

    data = _data()
    params = _fine_params()
    q = _posterior(H1, seed=5)
    block = BlockIndex.from_hierarchy(data.hierarchy, 1)
    one_block = np.zeros(H1, dtype=bool)
    one_block[3:6] = True

    args = (
        q,
        data.get_X(1),
        data.get_Y(1),
        data.get_X(2),
        data.get_Y(2),
        data.get_C(1),
        params,
        block,
    )
    whole = allocation_ll(*args)
    without = allocation_ll(*args, recorded=~one_block)
    alone = allocation_ll(*args, recorded=one_block)

    assert without != whole
    assert whole == pytest.approx(without + alone)
    assert allocation_ll(*args, recorded=np.zeros(H1, dtype=bool)) == 0.0


# ---------------------------------------------------------------------------
# Provenance and scorability are two statements
# ---------------------------------------------------------------------------


def _dilated_world():
    """Block 1 unrecorded, and the row set the lag-one budget leaves behind."""
    flags = np.ones(H1, dtype=bool)
    flags[3:6] = False
    data = _data(flags)
    params = _fine_params()
    budget = parent_set_lag(params.parent_sets)
    rows = declared_rows(data.get_observed(1), budget, "the case")
    return data, params, flags, rows


def _fine_args(data, params):
    return (
        data.get_X(1),
        data.get_Y(1),
        data.get_X(2),
        data.get_Y(2),
        np.zeros((H1, 0)),
        data.get_C(1),
        params,
        data.hierarchy,
        1,
    )


def test_the_dilation_costs_the_slot_after_the_gap_and_no_other() -> None:
    _, _, flags, rows = _dilated_world()
    assert flags.astype(int).tolist() == [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    assert rows.astype(int).tolist() == [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    # Slot 6 was recorded and cannot be scored, which is the whole case: the
    # two row sets differ, and block 2 holds one epoch from each of them.
    assert flags[6] and not rows[6]


def test_a_recorded_slot_keeps_its_count_in_its_block_sum() -> None:
    """The block sum is over the record, so it agrees with the parent total."""
    from src.core.blocks import BlockIndex

    data, _, flags, rows = _dilated_world()
    X1, X2 = data.get_X(1), data.get_X(2)
    block = BlockIndex.from_hierarchy(data.hierarchy, 1)

    assert X1[6].tolist() == [1, 0]
    assert X2[2].tolist() == [4, 2]
    np.testing.assert_array_equal(block.sum(np.where(flags[:, None], X1, 0))[2], X2[2])
    # Taking the block sum over the scorable rows instead would lose that start
    # while the parent total the allocation is measured against still holds it.
    assert block.sum(np.where(rows[:, None], X1, 0))[2].tolist() == [3, 2]


def test_the_unscorable_slot_leaves_the_sum_and_moves_no_sibling() -> None:
    data, params, flags, rows = _dilated_world()
    args = _fine_args(data, params)

    kept = build_log_like_fine(*args, recorded=flags)
    scored = build_log_like_fine(*args, recorded=flags, scored=rows)
    crossed = build_log_like_fine(*args, recorded=rows)

    # Slot 6 is recorded, so it has a term of its own until the budget erodes
    # it away, and then its row is zero and the chain bridges it.
    assert np.all(kept[6] != 0.0)
    assert np.all(scored[6] == 0.0)
    # Slots 7 and 8 are recorded and scorable, and block 2 is the block slot 6
    # sits in, so they are exactly where the whole record left them.
    np.testing.assert_array_equal(scored[7], kept[7])
    np.testing.assert_array_equal(scored[8], kept[8])
    # Reading the eroded row set as provenance is what used to move them.
    assert not np.allclose(crossed[7], kept[7])
    assert not np.allclose(crossed[8], kept[8])


def test_the_allocation_objective_is_additive_over_the_scored_rows() -> None:
    from src.core.blocks import BlockIndex

    data, params, flags, rows = _dilated_world()
    q = _posterior(H1, seed=5)
    block = BlockIndex.from_hierarchy(data.hierarchy, 1)
    args = (
        q,
        data.get_X(1),
        data.get_Y(1),
        data.get_X(2),
        data.get_Y(2),
        data.get_C(1),
        params,
        block,
    )

    whole = allocation_ll(*args, recorded=flags)
    scored = allocation_ll(*args, recorded=flags, scored=rows)
    dropped = allocation_ll(*args, recorded=flags, scored=flags & ~rows)
    assert whole == pytest.approx(scored + dropped)

    # A cut through the middle of block 2, which is where the two row sets
    # disagree, so the split has to survive it as well.
    head, tail = rows.copy(), rows.copy()
    head[8:] = False
    tail[:8] = False
    a = allocation_ll(*args, recorded=flags, scored=head)
    b = allocation_ll(*args, recorded=flags, scored=tail)
    assert scored == pytest.approx(a + b)

    # Reading the eroded row set as provenance loses that, since each part is
    # then a different block competition.
    crossed = allocation_ll(*args, recorded=rows)
    assert allocation_ll(*args, recorded=head) + allocation_ll(
        *args, recorded=tail
    ) != pytest.approx(crossed)


def test_a_scored_row_the_record_does_not_hold_is_refused() -> None:
    data, params, flags, rows = _dilated_world()
    with pytest.raises(ValueError, match="does not hold them"):
        build_log_like_fine(*_fine_args(data, params), recorded=rows, scored=flags)


def test_a_fully_masked_scale_moves_no_emission_parameter() -> None:
    data = _data()
    params = _coarse_params()
    q = _posterior(H2)
    nothing = np.zeros(H2, dtype=bool)
    updated = update_coarse_count_params(
        q, data.get_X(2), data.get_C(2), params, n_steps=5, scored=nothing
    )
    np.testing.assert_array_equal(updated.beta0, params.coarse_count.beta0)
    np.testing.assert_array_equal(updated.betaS, params.coarse_count.betaS)
    np.testing.assert_array_equal(updated.r_disp, params.coarse_count.r_disp)


def test_a_fully_masked_scale_lets_no_edge_into_the_graph() -> None:
    data = _data()
    params = _coarse_params()
    q = _posterior(H2)
    nothing = np.zeros(H2, dtype=bool)
    empty = LocalScore.compute_coarsest(
        set(),
        0,
        data.get_X(2),
        data.get_Y(2),
        q,
        data.get_C(2),
        params,
        lam=1.0,
        scored=nothing,
    )
    one_edge = LocalScore.compute_coarsest(
        {(1, 1)},
        0,
        data.get_X(2),
        data.get_Y(2),
        q,
        data.get_C(2),
        params,
        lam=1.0,
        scored=nothing,
    )
    assert empty == 0.0
    assert one_edge == -1.0


# ---------------------------------------------------------------------------
# A mask of all ones reproduces the unmasked fit, number for number
# ---------------------------------------------------------------------------


def _one_sweep(data: MultiScaleData, k: int, params: ScaleParams):
    """One E-step, one M-step and one S-step at scale k, as the fit runs them."""
    graph = MultiScaleDBNGraph(
        hierarchy=data.hierarchy, label_names=list(data.label_names)
    )
    # The declared budget of the S-step below, handed to every step the way
    # a fit hands one number to all of them.
    estep = EStep(max_inner_iter=3)
    mstep = MStep(n_steps_emission=4, n_steps_regime=4)
    estep.lag_budget = mstep.lag_budget = [1] * data.K
    q = estep.run(k, data, params, graph)
    updated = mstep.run(k, q, params, data)
    new_graph, new_params = SStep(max_lag=1, max_indegree=1, profile_steps=4).run(
        k, q, params, graph, data
    )
    return q, updated, new_params


@pytest.mark.parametrize("k", [1, 2])
def test_a_mask_of_all_ones_reproduces_the_unmasked_sweep(k) -> None:
    params = _fine_params() if k == 1 else _coarse_params()
    q0, m0, s0 = _one_sweep(_data(), k, deepcopy(params))
    q1, m1, s1 = _one_sweep(_data(np.ones(H1, dtype=bool)), k, deepcopy(params))

    np.testing.assert_array_equal(q0.gamma, q1.gamma)
    np.testing.assert_array_equal(q0.xi, q1.xi)
    assert q0.log_Z == q1.log_Z

    emission = "coarse_count" if k == 2 else "fine_count"
    a, b = getattr(m0, emission), getattr(m1, emission)
    np.testing.assert_array_equal(a.beta0, b.beta0)
    np.testing.assert_array_equal(a.betaS, b.betaS)
    np.testing.assert_array_equal(m0.regime.trans, m1.regime.trans)
    assert s0.parent_sets == s1.parent_sets


def test_the_search_refuses_a_graph_whose_lag_exceeds_its_budget() -> None:
    """The row set is eroded by the budget, so a longer lag would read a gap."""
    data = _data(np.ones(H1, dtype=bool))
    params = _fine_params()
    params.parent_sets = {0: {(1, 3)}}
    graph = MultiScaleDBNGraph(
        hierarchy=data.hierarchy, label_names=list(data.label_names)
    )
    with pytest.raises(ValueError, match="holding a lag of 3"):
        SStep(max_lag=1, max_indegree=1, profile_steps=2).run(
            1, EStep(max_inner_iter=2).run(1, data, params, graph), params, graph, data
        )


@pytest.mark.parametrize("k", [1, 2])
def test_a_real_gap_moves_the_sweep(k) -> None:
    """The equality above would hold for free if the mask were being ignored."""
    params = _fine_params() if k == 1 else _coarse_params()
    flags = np.ones(H1, dtype=bool)
    flags[3:6] = False
    q0, m0, _ = _one_sweep(_data(), k, deepcopy(params))
    q1, m1, _ = _one_sweep(_data(flags), k, deepcopy(params))

    assert not np.array_equal(q0.gamma, q1.gamma)
    emission = "coarse_count" if k == 2 else "fine_count"
    assert not np.array_equal(getattr(m0, emission).beta0, getattr(m1, emission).beta0)


# ---------------------------------------------------------------------------
# The fine formulation: the mask, the risk set, and the dilation
# ---------------------------------------------------------------------------


def test_a_masked_slot_and_a_running_label_stay_two_statements() -> None:
    X1 = np.zeros((6, 1), dtype=np.int64)
    Y1 = np.zeros((6, 1), dtype=np.int64)
    X1[1, 0], Y1[1, 0] = 1, 3  # an episode covering slots 1, 2 and 3
    X1[5, 0], Y1[5, 0] = 1, 1
    observed = np.ones(6, dtype=bool)
    observed[5] = False

    at_risk = risk_set(X1, Y1)
    usable = usable_slots(X1, Y1, observed)

    assert at_risk[:, 0].tolist() == [1, 1, 0, 0, 1, 1]
    assert usable[:, 0].tolist() == [1, 1, 0, 0, 1, 0]
    # Each exclusion is still readable on its own, and the two are disjoint.
    assert (~at_risk)[:, 0].tolist() == [0, 0, 1, 1, 0, 0]
    assert (at_risk & ~usable)[:, 0].tolist() == [0, 0, 0, 0, 0, 1]
    # Masking a slot says nothing about whether a start was possible there.
    np.testing.assert_array_equal(risk_set(X1, Y1), at_risk)
    np.testing.assert_array_equal(usable_slots(X1, Y1), at_risk)


def _one_mark(X1, Y1, log_nu, disp, slot: int) -> float:
    """The mark term of a single start, summed by hand from its own row."""
    only = np.zeros_like(X1), np.zeros_like(Y1)
    only[0][slot, 0], only[1][slot, 0] = X1[slot, 0], Y1[slot, 0]
    return mark_loglik(*only, log_nu, disp)


def test_the_mark_term_reads_the_mask_and_never_the_risk_set() -> None:
    X1 = np.zeros((6, 1), dtype=np.int64)
    Y1 = np.zeros((6, 1), dtype=np.int64)
    X1[1, 0], Y1[1, 0] = 1, 3
    X1[5, 0], Y1[5, 0] = 1, 1
    log_nu = np.full((6, 1), np.log(2.0))
    disp = np.array([4.0])
    observed = np.ones(6, dtype=bool)
    observed[5] = False

    both = mark_loglik(X1, Y1, log_nu, disp)
    kept = mark_loglik(X1, Y1, log_nu, disp, observed)

    only_first = np.zeros_like(X1), np.zeros_like(Y1)
    only_first[0][1, 0], only_first[1][1, 0] = 1, 3
    assert kept == pytest.approx(mark_loglik(*only_first, log_nu, disp))
    # The masked start's own term is what left the sum, and nothing else did.
    assert kept - both == pytest.approx(-_one_mark(X1, Y1, log_nu, disp, 5))
    # Slots 2 and 3 are outside the risk set and carry no mark either way, so
    # the mark term never had a use for the risk set.
    assert mark_loglik(X1, Y1, log_nu, disp, np.zeros(6, dtype=bool)) == 0.0


def test_usable_slots_refuses_a_mask_it_cannot_read() -> None:
    X1 = np.zeros((4, 1), dtype=np.int64)
    Y1 = np.zeros((4, 1), dtype=np.int64)
    with pytest.raises(ValueError, match="boolean array"):
        usable_slots(X1, Y1, np.ones(4, dtype=np.int64))
    with pytest.raises(ValueError, match="one flag per finest slot"):
        usable_slots(X1, Y1, np.ones(3, dtype=bool))


# ---- the case the dilation is for ------------------------------------------

SLOTS = 24
BLOCKS = 10
GAP = 4  # the coarse epoch nobody watched
BURST = 5  # the coarse epoch right after it


def _gap_world():
    """
    A record with one unwatched day, and a burst on the day after it.

    Nothing was recorded during the gap, so the source label's day total reads
    zero there and the lag-one day feature is distinctive at exactly one epoch,
    the one holding the burst.  The apparent edge is the gap.
    """
    n_slots = SLOTS * BLOCKS
    parent = np.repeat(np.arange(BLOCKS), SLOTS)
    ancestors = ancestor_maps([parent], n_slots)
    X1 = np.zeros((n_slots, 2), dtype=np.int64)
    Y1 = np.zeros((n_slots, 2), dtype=np.int64)

    observed = np.ones(n_slots, dtype=bool)
    observed[parent == GAP] = False

    # The source: a start every other slot on every recorded day.
    X1[np.arange(0, n_slots, 2), 1] = 1
    # The target: one start in eight, and every slot of the burst day.
    X1[np.arange(0, n_slots, 8), 0] = 1
    X1[parent == BURST, 0] = 1
    # Nothing is recorded inside the gap.
    X1[parent == GAP, :] = 0
    Y1[X1 > 0] = 1

    sources = empty_sources(
        ancestors, [np.zeros((n_slots, 0)), np.zeros((BLOCKS, 0))], 2
    )
    sources.counts[0][:] = X1
    sources.durations[0][:] = Y1
    np.add.at(sources.counts[1], parent, X1)
    np.add.at(sources.durations[1], parent, Y1)
    return X1, Y1, observed, sources, ancestors


# The day-scale lag the gap world's candidate space allows, which fixes the row
# set for every design compared in it.
GAP_BUDGET = {2: 1}


def _designs(sources):
    """The target's design with no parent, and with the day-lagged source."""
    from datetime import datetime, timedelta

    from src.core.calendar import SlotCalendar

    cal = SlotCalendar(
        origin=datetime(2026, 3, 2), slot=timedelta(hours=1), branching=(SLOTS,)
    )
    graph = TypedGraph(label_names=["target", "source"])
    ladder = ScaleLadder.from_sources(sources)
    index = CalendarIndex.place([], cal, sources, ladder)
    without = build_design(graph, 0, COUNT, sources, ladder, index)
    graph.add_edge(TypedEdge(OBSERVED, 2, 1, COUNT, 1), 0, COUNT)
    with_edge = build_design(graph, 0, COUNT, sources, ladder, index)
    return graph, without, with_edge


def _gain(without, with_edge, x, weights, at_risk) -> float:
    """The exact improvement the edge buys, which is what the search scores."""
    base = BaseFit(
        without.search_gram(weights, at_risk), without.search_target(x, at_risk)
    )
    full = BaseFit(
        with_edge.search_gram(weights, at_risk), with_edge.search_target(x, at_risk)
    )
    return full.value - base.value


def test_the_dilation_refuses_an_edge_that_only_reads_the_gap() -> None:
    X1, Y1, observed, sources, ancestors = _gap_world()
    graph, without, with_edge = _designs(sources)
    weights = pg_mean(np.zeros(X1.shape[0]))
    x = X1[:, 0]

    raw = usable_slots(X1, Y1, observed)[:, 0]
    dilated = usable_slots(X1, Y1, fit_rows(observed, GAP_BUDGET, sources))[:, 0]
    # The dilation costs exactly the day that reads the gap.
    assert int(raw.sum() - dilated.sum()) == SLOTS

    loose = _gain(without, with_edge, x, weights, raw)
    tight = _gain(without, with_edge, x, weights, dilated)
    assert loose > 1.0
    assert tight == pytest.approx(0.0, abs=1e-6)


def test_the_search_itself_accepts_the_edge_only_without_the_dilation() -> None:
    X1, Y1, observed, sources, ancestors = _gap_world()
    graph, without, with_edge = _designs(sources)
    weights = pg_mean(np.zeros(X1.shape[0]))
    x = X1[:, 0]

    def report(at_risk):
        def gain_fn(selected):
            return [_gain(without, with_edge, x, weights, at_risk)]

        def apply_fn(selected):
            design = with_edge if selected else without
            return BaseFit(
                design.search_gram(weights, at_risk),
                design.search_target(x, at_risk),
            ).value

        return greedy_search(gain_fn, apply_fn, 1, price=1.0, max_edges=1)

    raw = usable_slots(X1, Y1, observed)[:, 0]
    dilated = usable_slots(X1, Y1, fit_rows(observed, GAP_BUDGET, sources))[:, 0]
    assert report(raw).accepted == [0]
    assert report(dilated).accepted == []


def test_the_fine_bound_reads_the_mask_and_is_unchanged_without_one() -> None:
    X1, Y1, observed, sources, _ = _gap_world()
    graph = TypedGraph(label_names=["target", "source"])
    params = FineParams(n_labels=2)
    params.base[COUNT][:] = -2.0
    chains = [RegimeChain.uniform(scale=2, n_states=2, n_labels=2, n_epochs=BLOCKS)]
    sem = FineSEM(max_sweeps=2, lag_budget=GAP_BUDGET)

    plain = sem.objective(X1, Y1, params, graph, sources, chains)
    all_seen = sem.objective(
        X1, Y1, params, graph, sources, chains, 0.0, np.ones(X1.shape[0], dtype=bool)
    )
    with_gap = sem.objective(X1, Y1, params, graph, sources, chains, 0.0, observed)

    assert plain == all_seen
    assert with_gap > plain  # fewer negative start terms are summed
    assert fit_rows(None, GAP_BUDGET, sources) is None

    _, fitted, history = sem.fit(
        X1,
        Y1,
        params,
        graph,
        sources,
        chains,
        [X1.shape[0], BLOCKS],
        observed=observed,
    )
    assert history.monotone
    assert fitted[0].gamma.shape == (BLOCKS, 2)


def test_the_whole_fine_likelihood_reads_the_mask() -> None:
    X1, Y1, observed, sources, _ = _gap_world()
    graph = TypedGraph(label_names=["target", "source"])
    params = FineParams(n_labels=2)
    params.base[COUNT][:] = -1.5
    params.base["duration"][:] = np.log(2.0)

    whole = total_loglik(X1, Y1, params, graph, sources)
    part = total_loglik(X1, Y1, params, graph, sources, observed=observed)
    assert part > whole
    assert total_loglik(
        X1, Y1, params, graph, sources, observed=np.ones(X1.shape[0], dtype=bool)
    ) == pytest.approx(whole)
