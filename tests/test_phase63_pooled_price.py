"""
tests/test_phase63_pooled_price.py
==================================
What an edge costs the cohort, and what the pooled fit is actually reading.

Five properties carry this phase.

The price.  A price answers how large a gain a candidate reaches when there is
no relation there at all, so it is a quantile of a measured null and never a
number somebody liked.  The null rolls each candidate's own column, which keeps
its marginal distribution and its serial dependence and destroys only what it
lines up with, and what it must guarantee is a rate: no more of the candidates
that carry nothing may clear the bar than the quantile allows.  It moves with
the roster, because a gain is a sum over records and S records legitimately
carry S times the evidence.

The predictor.  Every edge column is centered over the record, so a
coefficient the solve fits stands against a centered column while the raw
feature is what a predictor reads.  The pass adds the difference back, so the
weights of the next round and the start term of this one are both taken at the
predictor the solve maximized over.  The gap between the two then stays at
rounding.  Left there it grows round by round, since every round re-weights
against what the round before it left.

The chain effect.  A latent block's coefficients are the chain's effect, so the
solve writes them where `RegimeChain.moments` reads them.  The first state is the
block's reference and stays at zero.

The guard.  With the proposed edge carrying no coefficient the predictor, the
weights and the risk set are all unchanged, and adding a column to a fixed
quadratic raises its maximum by the rank-one gain and by exactly the rank-one
gain; a guard reading that number is reading an identity.  What the move reaches
is the target's start term at the coefficients the widened solve writes, weighed
against the same term at the solve of the base alone, and a move that lowers it
is put back.

Residency.  The start term is summed one label at a time, which is the same
number the whole (T_1, L) expression gives and eight arrays of that shape
cheaper.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.exp11_pooling import CellRequest, _cell_price, sweep_quantile
from src.core.cohort import Cohort
from src.core.scales import ScaleHierarchy
from src.core.typed_graph import COUNT, DURATION, OBSERVED, TypedEdge, TypedGraph
from src.core.variables import MultiScaleData
from src.inference.bound import pg_mean, segment_sum
from src.learning.fine_design import ScaleLadder, build_design, edge_columns
from src.learning.fine_mstep import _RIDGE, regime_block
from src.learning.fine_sem import RegimeChain, fit_rows, start_bound
from src.learning.fine_sstep import BaseFit, candidate_gains
from src.learning.pooled import PooledFit, _shifted
from src.model.intensity import FineParams, usable_slots

# An hourly slot over four weeks, so the ladder is hour, day and week.  Eight
# labels over two candidate scales, two source quantities and two lags is 64
# candidates per target and 512 null draws over the whole axis, which is enough
# to read a rate off.
HOURS, DAYS, WEEKS = 24, 28, 4
T = HOURS * DAYS
LABELS = ["walking", "sleep", "snacking", "gym", "yoga", "reading", "calls", "chores"]
L = len(LABELS)
DAY_OF = np.repeat(np.arange(DAYS), HOURS).astype(np.int32)
WEEK_OF = np.repeat(np.arange(WEEKS), 7).astype(np.int32)

BUDGET = {2: 2, 3: 2}
# Sleep on day d-1 drives walking on day d, which is the one relation this world
# holds and the one the search has to find.
TRUE_EDGE = TypedEdge(OBSERVED, 2, 1, COUNT, 1)


def _hierarchy() -> ScaleHierarchy:
    return ScaleHierarchy(K=3, H=[T, DAYS, WEEKS], parent_maps=[DAY_OF, WEEK_OF, None])


def _arrays(seed: int):
    """One record: seven independent labels and a target the second one drives."""
    rng = np.random.default_rng(seed)
    X1 = np.zeros((T, L), dtype=np.int64)
    for ell in range(1, L):
        X1[:, ell] = rng.random(T) < (0.05 + 0.03 * ell)
    total = segment_sum(X1[:, 1].astype(float), DAY_OF, DAYS)
    previous = np.concatenate([[0.0], total[:-1]])
    odds = -3.5 + 0.55 * previous[DAY_OF]
    X1[:, 0] = rng.random(T) < 1.0 / (1.0 + np.exp(-odds))
    return X1, (X1 > 0).astype(np.int64)


def _subject(seed: int) -> MultiScaleData:
    X1, Y1 = _arrays(seed)
    day_x = segment_sum(X1.astype(float), DAY_OF, DAYS).astype(np.int64)
    day_y = segment_sum(Y1.astype(float), DAY_OF, DAYS).astype(np.int64)
    week_x = segment_sum(day_x.astype(float), WEEK_OF, WEEKS).astype(np.int64)
    week_y = segment_sum(day_y.astype(float), WEEK_OF, WEEKS).astype(np.int64)
    return MultiScaleData(
        hierarchy=_hierarchy(),
        label_names=list(LABELS),
        X=[X1, day_x, week_x],
        Y=[Y1, day_y, week_y],
        C=[np.zeros((T, 1)), np.zeros((DAYS, 1)), np.zeros((WEEKS, 1))],
    )


def _cohort(n: int, seeds=None) -> Cohort:
    names = tuple(f"p{i:02d}" for i in range(n))
    table = dict(zip(names, list(seeds) if seeds is not None else [0] * n))
    return Cohort(
        subjects=names,
        hierarchy=_hierarchy(),
        label_names=list(LABELS),
        reader=lambda name: _subject(table[name]),
        masks=np.ones((n, T), dtype=bool),
    )


def _pool():
    """Every typed edge the search may propose, at the day scale and the week."""
    return [
        TypedEdge(OBSERVED, k, src, quantity, lag)
        for k in (2, 3)
        for src in range(L)
        for quantity in (COUNT, DURATION)
        for lag in (1, 2)
    ]


def _model(n_chains: int = 0):
    """An empty graph, a flat intercept and as many day-scale chains as asked."""
    graph = TypedGraph(label_names=list(LABELS))
    params = FineParams(n_labels=L)
    params.base[COUNT][:] = -2.0
    chains = [
        RegimeChain.uniform(scale=2, n_states=2, n_labels=L, n_epochs=DAYS)
        for _ in range(n_chains)
    ]
    for chain in chains:
        chain.effect = np.tile(np.array([[0.0], [0.5]]), (1, L))
    return graph, params, chains


def _fitted(n: int, seeds=None, targets=None, chains: int = 0, rounds: int = 1):
    """A cohort and a fit that has taken as many coefficient rounds as asked."""
    cohort = _cohort(n, seeds)
    graph, params, chain_list = _model(chains)
    pool = _pool()
    fit = PooledFit(
        graph,
        params,
        chain_list,
        BUDGET,
        targets=targets,
        candidates={t: list(pool) for t in (targets or range(L))},
    )
    for _ in range(rounds):
        fit.solve(fit.accumulate(cohort, step=False))
    return cohort, fit, pool


def _gains(fit: PooledFit, stats, target: int) -> np.ndarray:
    design, seen = stats.designs[target], stats.candidates[target]
    return candidate_gains(
        BaseFit(design.A, design.b), seen.cross, seen.diagonal, seen.target
    )


# ---------------------------------------------------------------------------
# The price, and the null it is read off
# ---------------------------------------------------------------------------


def test_a_rolled_candidate_is_the_same_column_out_of_alignment() -> None:
    """
    The null column is the candidate's own, rotated and centered the same way.

    A roll keeps the marginal distribution and the serial dependence of the
    column and destroys only what it lines up with, so what the gain of the
    result measures is the alignment and nothing else.
    """
    cohort = _cohort(1, seeds=[3])
    data = cohort.open("p00")
    sources = cohort.sources(data)
    ladder = ScaleLadder.from_sources(sources, data.observed)
    graph, _params, _chains = _model(0)
    column = edge_columns(
        [TRUE_EDGE],
        graph,
        sources,
        ladder,
        PooledFit(graph, FineParams(n_labels=L), [], BUDGET).index,
        BUDGET,
    )[0]

    rolled = _shifted(column, ladder, BUDGET, seed=5)

    assert rolled.scale == column.scale
    assert rolled.edge == column.edge
    # The uncentered values are a rotation of the uncentered values.
    before = np.sort(column.values + column.removed)
    after = np.sort(rolled.values + rolled.removed)
    np.testing.assert_allclose(after, before)
    # And it carries the centering convention every real candidate carries: the
    # weighted mean over the record is gone.
    weight = ladder.sizes[1] * ladder.observed[1]
    assert float(rolled.values @ weight) == pytest.approx(0.0)
    assert not np.allclose(rolled.values, column.values)


def test_the_null_is_cleared_at_the_rate_the_quantile_names() -> None:
    """
    The one guarantee a price carries: candidates with nothing in them clear it rarely.

    In the draw the price was read off the rate is the quantile by construction.
    On a fresh draw it is that rate plus the sampling error of 512 draws, and
    what must not happen is a bar a null column walks over.
    """
    cohort, fit, _pool = _fitted(3, seeds=[1, 2, 3])

    measured = fit.null_gains(cohort, seed=11)
    fresh = fit.null_gains(cohort, seed=500_000)

    assert measured.size == 512
    for quantile in (0.9, 0.95):
        price = float(np.quantile(measured, quantile))
        assert float((measured > price).mean()) == pytest.approx(
            1.0 - quantile, abs=2.0 / measured.size
        )
        assert float((fresh > price).mean()) <= 1.5 * (1.0 - quantile)


def test_a_fit_with_no_candidate_measures_an_empty_null() -> None:
    """A price over a candidate space of nothing is a bar over nothing."""
    cohort = _cohort(1, seeds=[1])
    graph, params, chains = _model(0)
    fit = PooledFit(graph, params, chains, BUDGET, targets=[0])

    assert fit.null_gains(cohort).shape == (0,)


def test_the_price_and_the_gain_move_with_the_roster_together() -> None:
    """
    S copies of one record carry S times the evidence, and S times the price.

    A price the roster could not move would refuse the very information distinct
    participants bring, which is what the pooling arm exists to measure.  What
    the null fixes is the rate, and the rate is what stays put.
    """
    one_cohort, one_fit, pool = _fitted(1)
    four_cohort, four_fit, _ = _fitted(4)
    where = pool.index(TRUE_EDGE)

    one_null = one_fit.null_gains(one_cohort, seed=11)
    four_null = four_fit.null_gains(four_cohort, seed=11)
    one_gain = _gains(one_fit, one_fit.accumulate(one_cohort, step=False), 0)[where]
    four_gain = _gains(four_fit, four_fit.accumulate(four_cohort, step=False), 0)[where]

    np.testing.assert_allclose(four_null, 4.0 * one_null, rtol=1e-9)
    assert four_gain == pytest.approx(4.0 * one_gain, rel=1e-9)
    # So the edge that clears its own cohort's price clears it at either size.
    for null, gain in ((one_null, one_gain), (four_null, four_gain)):
        assert gain > float(np.quantile(null, sweep_quantile(len(pool))))


def test_a_declared_price_answers_the_roster_and_not_the_evidence() -> None:
    """
    A fixed bar, against the price this cohort's own null puts up.

    At one subject the true edge is worth 22.7 nats and a declared 40 refuses
    it; at three it is worth 60.9 and the same 40 takes it.  The bar moved
    nothing and the roster did all the deciding.  The measured price is under
    both.
    """
    bought = {}
    for n in (1, 3):
        cohort, fit, pool = _fitted(n, seeds=list(range(1, n + 1)), targets=[0])
        gain = float(_gains(fit, fit.accumulate(cohort, step=False), 0).max())
        price = float(np.quantile(fit.null_gains(cohort, seed=11), 0.95))
        bought[n] = (gain, price)

    assert bought[1][0] < 40.0 < bought[3][0]
    assert bought[1][1] < bought[1][0]
    assert bought[3][1] < bought[3][0]


def test_the_search_buys_the_true_edge_at_the_price_its_null_puts_up() -> None:
    """The declared relation is what the measured price leaves standing."""
    cohort, fit, pool = _fitted(3, seeds=[1, 2, 3], targets=[0])
    price = float(
        np.quantile(fit.null_gains(cohort, seed=11), sweep_quantile(len(pool)))
    )

    report = fit.search(cohort, price, max_edges=2)[0]

    assert pool[report.accepted[0]] == TRUE_EDGE
    assert TRUE_EDGE in fit.graph.get_parents(0, COUNT)
    assert fit.params.coef[(0, COUNT)][TRUE_EDGE] > 0.0


def test_the_sweep_quantile_is_read_off_the_candidate_space() -> None:
    """One candidate in the sweep is the rate, so the quantile is 1 - 1 / C."""
    assert sweep_quantile(480) == pytest.approx(1.0 - 1.0 / 480.0)
    # A sweep of one candidate names the lowest gain the null reaches, which
    # `_cell_price` refuses as a bar every candidate clears.
    assert sweep_quantile(1) == 0.0
    with pytest.raises(ValueError, match="holds no candidates"):
        sweep_quantile(0)


def test_a_cell_with_neither_a_price_nor_a_quantile_is_refused() -> None:
    """The smallest gain the null reaches is a bar every candidate clears."""
    cohort, fit, pool = _fitted(1, targets=[0])
    request = CellRequest(
        cohort=cohort,
        candidates={0: pool},
        lag_budget=BUDGET,
        quantity=COUNT,
        price=None,
        iters=1,
        max_edges=1,
    )

    with pytest.raises(ValueError, match="is not one"):
        _cell_price(fit, request)


def test_a_declared_price_is_recorded_as_declared() -> None:
    """A cell that was handed a price measures no null and says so."""
    cohort, fit, pool = _fitted(1, targets=[0])
    request = CellRequest(
        cohort=cohort,
        candidates={0: pool},
        lag_budget=BUDGET,
        quantity=COUNT,
        price=7.5,
        iters=1,
        max_edges=1,
    )

    assert _cell_price(fit, request) == 7.5
    assert request.measured == {"price": 7.5, "quantile": None}


def test_a_measured_price_carries_the_null_it_came_from() -> None:
    """The record cannot report a price without the draw that produced it."""
    cohort, fit, pool = _fitted(1, targets=[0])
    request = CellRequest(
        cohort=cohort,
        candidates={0: pool},
        lag_budget=BUDGET,
        quantity=COUNT,
        price=None,
        iters=1,
        max_edges=1,
        price_quantile=0.95,
    )

    price = _cell_price(fit, request)

    assert request.measured["price"] == price
    assert request.measured["quantile"] == 0.95
    assert request.measured["null_draws"] == len(pool)
    # The rate the quantile names, up to the one draw a discrete quantile of 64
    # of them can straddle.
    assert request.measured["cleared"] <= 0.05 + 1.0 / len(pool)
    assert request.measured["null_median"] < price <= request.measured["null_max"]


# ---------------------------------------------------------------------------
# The predictor the solve fits
# ---------------------------------------------------------------------------


def _design_gap(cohort: Cohort, fit: PooledFit, beta, target: int) -> float:
    """The largest distance between the design's predictor and the pass's own."""
    worst = 0.0
    for name in cohort.subjects:
        data = cohort.open(name)
        sources = cohort.sources(data)
        ladder = ScaleLadder.from_sources(sources, data.observed)
        chains = fit._chains_for(name)
        latent = [
            regime_block(c.gamma, sources.ancestors[c.scale - 1], f"regime k{c.scale}")
            for c in chains
        ]
        design = build_design(
            fit.graph, target, COUNT, sources, ladder, fit.index, latent, BUDGET
        )
        used = fit._predictor(sources, ladder, chains)[:, target]
        worst = max(worst, float(np.max(np.abs(design.predictor(beta) - used))))
    return worst


def test_the_predictor_the_pass_uses_is_the_one_the_design_fits() -> None:
    """
    Centering took a parent mean out of every column, and the pass puts it back.

    Without it the solve fits one predictor and the next round re-weights
    against another, and the two disagree by the mean that was removed.
    """
    cohort = _cohort(3, seeds=[1, 2, 3])
    graph, params, chains = _model(0)
    graph.add_edge(TRUE_EDGE, 0, COUNT)
    fit = PooledFit(graph, params, chains, BUDGET, targets=[0])
    beta = fit.solve(fit.accumulate(cohort, step=False))[0]

    assert _design_gap(cohort, fit, beta, 0) < 1e-12
    # The mean the centering removed is what the two would differ by, and it is
    # nothing like rounding.
    fit._centering = lambda sources, ladder: {}
    assert _design_gap(cohort, fit, beta, 0) > 1e-4


def test_the_gap_the_centering_leaves_does_not_grow_over_rounds() -> None:
    """
    Every round re-weights, so a gap the round leaves is a gap the next one carries.

    Twelve edges over four rounds is where it showed: the two predictors drifted
    apart round by round while the fit reported the wrong one.
    """
    cohort = _cohort(3, seeds=[1, 2, 3])
    graph, params, chains = _model(0)
    for src in range(1, L):
        for lag in (1, 2):
            graph.add_edge(TypedEdge(OBSERVED, 2, src, COUNT, lag), 0, COUNT)
    fit = PooledFit(graph, params, chains, BUDGET, targets=[0])

    gaps = []
    for _ in range(4):
        beta = fit.solve(fit.accumulate(cohort, step=False))[0]
        gaps.append(_design_gap(cohort, fit, beta, 0))

    assert len(fit.graph.get_parents(0, COUNT)) == 14
    assert max(gaps) < 1e-12


def test_the_start_term_the_pass_records_is_the_design_s_own() -> None:
    """
    The number the guard and the bound read is the design's, summed over subjects.

    A pass that re-weighted against the raw feature would record a start term of
    a model nobody fitted.
    """
    cohort = _cohort(2, seeds=[1, 2])
    graph, params, chains = _model(0)
    graph.add_edge(TRUE_EDGE, 0, COUNT)
    fit = PooledFit(graph, params, chains, BUDGET, targets=[0])
    stats = fit.accumulate(cohort, step=False)
    beta = fit.solve(stats)[0]
    after = fit.accumulate(cohort, step=False)

    expected = 0.0
    for name in cohort.subjects:
        data = cohort.open(name)
        sources = cohort.sources(data)
        ladder = ScaleLadder.from_sources(sources, data.observed)
        X1, Y1 = data.get_X(1), data.get_Y(1)
        at_risk = usable_slots(X1, Y1, fit_rows(data.observed, BUDGET, sources))
        design = build_design(
            fit.graph, 0, COUNT, sources, ladder, fit.index, [], BUDGET
        )
        expected += start_bound(
            X1[:, 0], design.predictor(beta), np.zeros(()), at_risk[:, 0]
        )

    assert after.starts[0] == pytest.approx(expected)
    # And the solve raised it, which is what a coefficient step is for.
    assert after.starts[0] > stats.starts[0]


def test_one_label_at_a_time_is_the_whole_expression() -> None:
    """
    The start term summed per label is the term the (T_1, L) expression gives.

    Eight arrays of that shape is 505 MB at a one-minute slot over 60 labels, and
    the sum over labels does not care in what order it was taken.
    """
    rng = np.random.default_rng(4)
    eta = rng.normal(size=(200, 3))
    variance = rng.random((200, 3))
    x = (rng.random((200, 3)) < 0.2).astype(np.int64)
    at_risk = rng.random((200, 3)) < 0.8

    anchor = np.sqrt(eta**2 + variance)
    weights = pg_mean(anchor)
    const = -np.logaddexp(0.0, anchor) + anchor / 2.0 + weights * anchor**2 / 2.0
    whole = (
        ((x > 0).astype(float) - 0.5) * eta
        - 0.5 * weights * (eta**2 + variance)
        + const
    )

    per_label = sum(
        start_bound(x[:, ell], eta[:, ell], variance[:, ell], at_risk[:, ell])
        for ell in range(3)
    )

    assert per_label == pytest.approx(float(whole[at_risk].sum()))
    # A model with no latent part carries the scalar zero and no array of it.
    assert start_bound(x[:, 0], eta[:, 0], np.zeros(()), None) == pytest.approx(
        start_bound(x[:, 0], eta[:, 0], np.zeros(200), None)
    )


# ---------------------------------------------------------------------------
# The chain effect the solve writes
# ---------------------------------------------------------------------------


def test_the_pooled_solve_writes_the_chain_effect_it_fitted() -> None:
    """
    A latent block's coefficients are the chain's effect, so they are stored there.

    Dropped, the chain keeps the value it started at while the class holds the
    effect out as what the fit put there, and the next latent step runs at a
    parameter no solve ever wrote.
    """
    cohort = _cohort(2, seeds=[1, 2])
    graph, params, chains = _model(1)
    started = chains[0].effect.copy()
    fit = PooledFit(graph, params, chains, BUDGET, targets=[0, 1])
    beta = fit.solve(fit.accumulate(cohort, step=True))

    for target in (0, 1):
        # The stacked design is the intercept, then the edges, then the chain,
        # whose first state is the block's reference and carries no column.
        assert fit.chains[0].effect[1, target] == pytest.approx(beta[target][-1])
        assert fit.chains[0].effect[0, target] == 0.0
        assert abs(fit.chains[0].effect[1, target] - started[1, target]) > 1e-3
    # A target the pass never formed a design for keeps what it came in with.
    np.testing.assert_allclose(fit.chains[0].effect[:, 2:], started[:, 2:])
    # And the subject's own copy reads the effect the cohort now holds.
    assert fit._posteriors["p00"][0].effect is fit.chains[0].effect


def test_the_written_effect_is_the_predictor_the_design_holds() -> None:
    """The chain reaches the predictor through the coefficients the solve wrote."""
    cohort = _cohort(2, seeds=[1, 2])
    graph, params, chains = _model(1)
    fit = PooledFit(graph, params, chains, BUDGET, targets=[0])
    beta = fit.solve(fit.accumulate(cohort, step=True))[0]

    assert abs(beta[-1]) > 1e-3
    assert _design_gap(cohort, fit, beta, 0) < 1e-12


# ---------------------------------------------------------------------------
# What the guard compares
# ---------------------------------------------------------------------------


def test_the_quadratic_maximum_rises_by_exactly_the_rank_one_gain() -> None:
    """
    The identity the old apply returned, which is why the guard could not refuse.

    With the proposed edge carrying no coefficient the predictor, the weights and
    the risk set are all unchanged, so the value is the maximum of one fixed
    quadratic over one more column, and the Schur complement says what that is
    to the last digit.
    """
    cohort, fit, pool = _fitted(2, seeds=[1, 2], targets=[0])
    stats = fit.accumulate(cohort, step=False)
    design, seen = stats.designs[0], stats.candidates[0]
    gains = _gains(fit, stats, 0)
    best = int(np.argmax(gains))

    p = design.A.shape[0]
    wide = np.zeros((p + 1, p + 1))
    wide[:p, :p] = design.A
    wide[:p, p] = wide[p, :p] = seen.cross[best]
    wide[p, p] = seen.diagonal[best]
    rhs = np.concatenate([design.b, [seen.target[best]]])

    increment = BaseFit(wide, rhs).value - BaseFit(design.A, design.b).value
    assert increment == pytest.approx(float(gains[best]), rel=1e-7)


def test_what_a_move_reaches_is_not_the_rank_one_gain() -> None:
    """
    The move writes the coefficient, so what comes back is the bound and not the gain.

    Both terms are read at their own anchor, and the two anchors are not the
    same one, which is the gap a rank-one gain taken at the old anchor cannot
    see.
    """
    cohort, fit, pool = _fitted(3, seeds=[1, 2, 3], targets=[0])

    report = fit.search(cohort, 0.0, max_edges=3)[0]

    assert len(report.accepted) == 3
    reached = np.diff(np.array(report.trace))
    gains = np.array(report.gains)
    assert np.all(reached > 0.0)
    # Every step differs from the gain that proposed it, and by far more than
    # the rounding the identity above holds to.
    assert np.all(np.abs(reached - gains) > 1e-3 * gains)


def test_a_move_the_data_does_not_support_is_put_back() -> None:
    """
    The guard's own case: an edge whose fitted coefficient lowers the start term.

    The candidate's moments are replaced by ones asking for a coefficient this
    record does not carry, which is the shape of the failure the guard exists
    for.  What must survive it is the graph and every parameter of the target.
    """
    cohort, fit, pool = _fitted(2, seeds=[1, 2], targets=[0])
    stats = fit.accumulate(cohort, step=False)
    held_coef = dict(fit.params.coef[(0, COUNT)])
    held_base = float(fit.params.base[COUNT][0])
    stats.candidates[0].target[3] *= 400.0
    state = {"stats": stats, "start": stats.starts[0], "in": []}

    reached = fit._move(cohort, 0, pool, [3], state)

    assert reached < state["start"]
    assert fit.graph.n_edges() == 0
    assert dict(fit.params.coef[(0, COUNT)]) == held_coef
    assert float(fit.params.base[COUNT][0]) == held_base
    assert state["in"] == []


def test_a_refused_move_puts_the_chain_effect_back_too() -> None:
    """A rejected widened solve wrote the chain's column, so it is restored."""
    cohort = _cohort(2, seeds=[1, 2])
    graph, params, chains = _model(1)
    fit = PooledFit(graph, params, chains, BUDGET, targets=[0], candidates={0: _pool()})
    fit.solve(fit.accumulate(cohort, step=True))
    stats = fit.accumulate(cohort, step=False)
    held = fit.chains[0].effect[:, 0].copy()
    stats.candidates[0].target[3] *= 400.0
    state = {"stats": stats, "start": stats.starts[0], "in": []}

    fit._move(cohort, 0, _pool(), [3], state)

    np.testing.assert_allclose(fit.chains[0].effect[:, 0], held)


def test_a_round_costs_one_cohort_pass_per_move_it_makes() -> None:
    """
    The pass that opens the search serves every target, and a move is a pass.

    The gain reads moments the opening pass already accumulated, so a round that
    buys nothing reads the cohort once for the whole label axis.
    """
    cohort, fit, pool = _fitted(2, seeds=[1, 2], targets=[0, 1])
    fit.candidates = {0: list(pool), 1: list(pool)}
    price = float(np.quantile(fit.null_gains(cohort, seed=11), 0.95))
    opened = []
    reader = cohort.reader
    cohort.reader = lambda name: (opened.append(name), reader(name))[1]

    reports = fit.search(cohort, price=1e9, max_edges=4)
    assert len(opened) == cohort.S
    assert all(not r.accepted for r in reports.values())

    opened.clear()
    reports = fit.search(cohort, price, max_edges=2)
    moves = sum(len(r.accepted) + len(r.rejected) for r in reports.values())
    assert moves > 0
    assert len(opened) == cohort.S * (1 + moves)
