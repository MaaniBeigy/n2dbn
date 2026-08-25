"""
tests/test_phase27_initialize.py
================================
Parameters built from observed data alone.

Two things are checked here.  Every moment estimate is re-derived from the
observed arrays by a second, independent expression, so an estimator that
inverted a formula the wrong way round fails rather than merely returning a
number.  And no array of any returned block holds a NaN or an infinity: every
downstream step multiplies these numbers together, and one non-finite entry
silently poisons a whole fit instead of failing.
"""

from __future__ import annotations

from copy import deepcopy
from typing import List

import numpy as np
import pytest

from src.core.graph import MultiScaleDBNGraph
from src.core.scales import ScaleHierarchy
from src.core.variables import MultiScaleData
from src.data.initialize import _ar1, _kmeans, _pca, initialize_from_data
from src.learning.estep import EStep
from src.learning.mstep import MStep
from src.learning.structural_em import TopDownSEM
from src.model.params import ScaleParams
from src.synthetic.families import make_generator
from src.synthetic.scenarios import ScenarioConfig

_BLOCKS = ("regime", "factor", "coarse_count", "coarse_dur", "fine_count", "fine_dur")


def _cfg(**over) -> ScenarioConfig:
    base = dict(
        name="init",
        K=2,
        H_finest=240,
        branching=[10],
        L=3,
        M_k=[2, 2],
        r_k=[1, 1],
        max_lag=[2, 1],
        max_indegree=[0, 1],
        # Family 4 with no misspecification is the only factory that honors
        # both M_k and r_k; families 1 to 3 force one of them to a fixed value.
        family=4,
        seed=3,
        n_replicates=1,
        edge_strength=10.0,
        regime_strength=3.0,
        finest_rate=0.2,
        finest_binary=True,
        slot_minutes=1,
    )
    base.update(over)
    return ScenarioConfig(**base)


@pytest.fixture(scope="module")
def dataset():
    """M=2, r=1, two context columns at both scales."""
    cfg = _cfg()
    gen = make_generator(cfg, seed=3)
    return cfg, gen, gen.generate(seed=3)


@pytest.fixture(scope="module")
def flat_dataset():
    """M=1, r=0: the single-regime, no-factor corner."""
    cfg = _cfg(name="init_flat", M_k=[1, 1], r_k=[0, 0], family=1, seed=5)
    gen = make_generator(cfg, seed=5)
    return cfg, gen, gen.generate(seed=5)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hand_made(
    X: List[np.ndarray],
    Y: List[np.ndarray],
    branching: List[int],
    d_ctx: int = 0,
) -> MultiScaleData:
    """Multiscale data from explicit arrays, finest first, with no context."""
    hier = ScaleHierarchy.from_uniform_branching(X[0].shape[0], branching)
    return MultiScaleData(
        hierarchy=hier,
        label_names=[f"l{i}" for i in range(X[0].shape[1])],
        X=X,
        Y=Y,
        C=[np.zeros((x.shape[0], d_ctx)) for x in X],
    )


def _coarse_only(X: np.ndarray, Y: np.ndarray) -> MultiScaleData:
    """One-scale data whose single scale is therefore the coarsest."""
    return _hand_made([X], [Y], [])


def _arrays(params: ScaleParams):
    """Every numeric leaf of one ScaleParams, as (name, array) pairs."""
    out = []
    for block_name in _BLOCKS:
        block = getattr(params, block_name)
        if block is None:
            continue
        for name, value in vars(block).items():
            tag = f"k{params.k}.{block_name}.{name}"
            if isinstance(value, np.ndarray):
                out.append((tag, value))
            elif isinstance(value, dict):
                out.extend((f"{tag}[{key}]", v) for key, v in value.items())
            elif isinstance(value, float):
                out.append((tag, np.asarray(value)))
    return out


def _assert_finite(params_list):
    for params in params_list:
        for tag, arr in _arrays(params):
            assert np.all(np.isfinite(arr)), f"{tag} holds a NaN or an infinity"


def _shapes(params: ScaleParams):
    """
    Shape of every fixed array-valued field, keyed by block and field name.

    The lag coefficient dictionaries are left out: they hold one entry per edge
    and the initializer deliberately starts with none, so their contents are
    compared by `test_graph_starts_empty` instead.
    """
    return {
        tag: arr.shape
        for tag, arr in _arrays(params)
        if arr.ndim > 0 and "[" not in tag
    }


# ---------------------------------------------------------------------------
# Shapes and dtypes
# ---------------------------------------------------------------------------


def test_shapes_match_the_generator(dataset):
    cfg, _, ds = dataset
    params = initialize_from_data(
        ds.data, cfg.M_k, cfg.r_k, cfg.max_lag, d_ctx=2, seed=0
    )
    assert len(params) == cfg.K
    for mine, truth in zip(params, ds.params):
        assert mine.k == truth.k
        assert mine.is_coarsest == truth.is_coarsest
        assert (mine.L, mine.M, mine.r) == (truth.L, truth.M, truth.r)
        assert _shapes(mine) == _shapes(truth)
    _assert_finite(params)


def test_every_array_is_float64(dataset):
    cfg, _, ds = dataset
    params = initialize_from_data(
        ds.data, cfg.M_k, cfg.r_k, cfg.max_lag, d_ctx=2, seed=0
    )
    for scale in params:
        for tag, arr in _arrays(scale):
            assert arr.dtype == np.float64, f"{tag} is {arr.dtype}"


def test_flat_scenario_matches_the_generator(flat_dataset):
    """M=1 and r=0 give the zero-dimensional factor the generator builds."""
    cfg, _, ds = flat_dataset
    params = initialize_from_data(
        ds.data, cfg.M_k, cfg.r_k, cfg.max_lag, d_ctx=2, seed=0
    )
    for mine, truth in zip(params, ds.params):
        assert _shapes(mine) == _shapes(truth)
        assert mine.factor.r == 0
        assert mine.factor.F.shape == (1, 0, 0)
        assert mine.factor.Sigma0.shape == (0, 0)
        assert mine.factor.mu0.shape == (0,)
        # A one-regime chain has a single transition, so its log table is 0.
        assert np.allclose(mine.regime.trans, 0.0)
        assert np.allclose(mine.regime.init, 0.0)
    _assert_finite(params)


# ---------------------------------------------------------------------------
# Contract of the returned object
# ---------------------------------------------------------------------------


def test_graph_starts_empty(dataset):
    cfg, _, ds = dataset
    params = initialize_from_data(
        ds.data, cfg.M_k, cfg.r_k, cfg.max_lag, d_ctx=2, seed=0
    )
    for scale in params:
        assert set(scale.parent_sets) == set(range(scale.L))
        assert all(not pset for pset in scale.parent_sets.values())
        if scale.is_coarsest:
            assert scale.coarse_count.betaLag == {}
            assert scale.coarse_dur.alphaLag == {}
        else:
            assert scale.fine_count.betaLag == {}
            assert scale.fine_dur.alphaLag == {}


def test_regime_effects_are_centered(dataset):
    cfg, _, ds = dataset
    params = initialize_from_data(
        ds.data, cfg.M_k, cfg.r_k, cfg.max_lag, d_ctx=2, seed=0
    )
    for scale in params:
        if scale.is_coarsest:
            effects = (scale.coarse_count.betaS, scale.coarse_dur.alphaS)
        else:
            effects = (scale.fine_count.betaS, scale.fine_dur.alphaS)
        for eff in effects:
            assert np.allclose(eff.mean(axis=0), 0.0, atol=1e-10)


def test_regime_effects_point_at_the_regime_that_owns_the_label():
    """
    Two halves with opposite label rates give opposite contrasts.

    The first half fires label 0 twenty times an epoch and label 1 once; the
    second half is the mirror image.  Whichever cluster the first half lands in,
    its contrast has to be positive on label 0 and negative on label 1, and the
    other cluster the reverse.
    """
    hi, lo = 20, 1
    first = np.tile([[hi, lo]], (30, 1))
    second = np.tile([[lo, hi]], (30, 1))
    X = np.vstack([first, second]).astype(np.int64)
    # Flat durations, so a contrast that had read the wrong array is visibly 0.
    Y = np.ones_like(X)
    params = initialize_from_data(_coarse_only(X, Y), [2], [0], [1], seed=0)

    betaS = params[0].coarse_count.betaS
    owner = int(np.argmax(betaS[:, 0]))
    gap = np.log1p(hi) - np.log1p(lo)
    assert np.allclose(betaS[owner], [gap / 2, -gap / 2])
    assert np.allclose(betaS[1 - owner], [-gap / 2, gap / 2])
    assert np.allclose(params[0].coarse_dur.alphaS, 0.0)


def test_transition_table_keeps_rows_current_and_columns_previous():
    """
    An asymmetric path pins down which index of `trans` is the previous regime.

    The pattern is one loud epoch followed by four quiet ones, so the loud
    regime always hands over to the quiet one while the quiet one hands back
    only one time in four.  A table stored the other way round would read those
    two conditionals swapped.
    """
    X = np.tile(np.array([[20], [1], [1], [1], [1]], dtype=np.int64), (10, 1))
    params = initialize_from_data(_coarse_only(X, X.copy()), [2], [0], [1], seed=0)

    trans = np.exp(params[0].regime.trans)
    assert np.allclose(trans.sum(axis=0), 1.0)
    loud = int(np.argmax(params[0].coarse_count.betaS[:, 0]))
    quiet = 1 - loud
    # 10 of 10 handovers out of loud, 9 of 39 out of quiet, both under a BDeu
    # prior of 1.25 per cell.
    assert np.isclose(trans[quiet, loud], 11.25 / 12.5)
    assert np.isclose(trans[loud, quiet], 10.25 / 41.5)

    # The first epoch is loud, so the initial distribution has to lean that way
    # rather than sitting on the prior's midpoint.
    init = np.exp(params[0].regime.init)
    assert np.isclose(init.sum(), 1.0)
    assert np.isclose(init[loud], 3.5 / 6.0)


def test_cross_scale_block_is_sized_by_the_parent_scale():
    """A finer scale couples to the regimes above it, not to its own."""
    rng = np.random.default_rng(10)
    X_fine = rng.poisson(0.7, size=(40, 2)).astype(np.int64)
    X_coarse = X_fine.reshape(10, 4, 2).sum(axis=1)
    Y_fine = (rng.poisson(3.0, size=(40, 2)) * (X_fine > 0)).astype(np.int64)
    Y_coarse = Y_fine.reshape(10, 4, 2).sum(axis=1)
    data = _hand_made([X_fine, X_coarse], [Y_fine, Y_coarse], [4])

    params = initialize_from_data(data, [2, 3], [1, 1], [1, 1], seed=0)
    assert params[0].regime.parent_M == 3
    assert params[0].regime.cross.shape == (2, 3)
    assert params[1].regime.parent_M == 0
    assert params[1].regime.cross is None


def test_factor_takes_only_what_the_regimes_leave():
    """Two regimes that explain every epoch exactly leave nothing to load on."""
    X = np.vstack([np.tile([[20, 1]], (30, 1)), np.tile([[1, 20]], (30, 1))]).astype(
        np.int64
    )
    params = initialize_from_data(_coarse_only(X, X.copy()), [2], [1], [1], seed=0)
    assert np.allclose(params[0].coarse_count.betaH, 0.0, atol=1e-8)


def test_loadings_come_back_on_the_raw_column_scale():
    """
    The factor is fitted on standardized residuals and mapped back afterwards.

    Two perfectly correlated columns whose residual spreads differ by a factor
    of six have to end up with loadings in that same ratio; loadings left on the
    standardized scale would be equal instead.
    """
    rng = np.random.default_rng(11)
    drive = rng.normal(size=200)
    log_scale = np.stack([1.2 + 0.9 * drive, 1.2 + 0.09 * drive], axis=1)
    X = np.maximum(np.rint(np.expm1(log_scale)), 0).astype(np.int64)
    params = initialize_from_data(_coarse_only(X, X.copy()), [1], [1], [1], seed=0)

    resid = np.log1p(X.astype(float))
    resid = resid - resid.mean(axis=0)
    col_sd = resid.std(axis=0)
    loadings = params[0].coarse_count.betaH[0]
    assert np.isclose(
        abs(loadings[0]) / abs(loadings[1]), col_sd[0] / col_sd[1], rtol=1e-8
    )


def test_dispersions_and_concentrations_stay_inside_their_brackets(dataset):
    cfg, _, ds = dataset
    params = initialize_from_data(
        ds.data, cfg.M_k, cfg.r_k, cfg.max_lag, d_ctx=2, seed=0
    )
    coarse = params[-1]
    for disp in (coarse.coarse_count.r_disp, coarse.coarse_dur.s_disp):
        assert np.all(disp >= 0.1) and np.all(disp <= 100.0)
    fine = params[0]
    # The M-step's golden-section search brackets the concentration in
    # [0.5, 2000]; a start outside that is unreachable from the first sweep.
    for conc in (fine.fine_count.kappa, fine.fine_dur.tau):
        assert 0.5 < conc < 2000.0


# ---------------------------------------------------------------------------
# The moment estimates, re-derived
# ---------------------------------------------------------------------------


def test_coarse_intercept_is_the_firing_rate_less_what_its_own_past_explains(
    dataset,
):
    """The rate over firing epochs, less the part lag-1 values account for."""
    from src.data.initialize import _lag_share

    cfg, _, ds = dataset
    params = initialize_from_data(
        ds.data, cfg.M_k, cfg.r_k, cfg.max_lag, d_ctx=2, seed=0
    )
    X = ds.data.get_X(cfg.K).astype(float)
    raw = np.array(
        [
            np.log(max(col[col > 0].mean() if (col > 0).any() else 0.0, 0.1))
            for col in X.T
        ]
    )
    want = np.maximum(raw - _lag_share(X), np.log(0.1))
    assert np.allclose(params[-1].coarse_count.beta0, want)
    assert np.all(params[-1].coarse_count.beta0 <= raw + 1e-12)

    Y = ds.data.get_Y(cfg.K).astype(float)
    raw_dur = np.array(
        [
            np.log(max(y[x > 0].mean() if (x > 0).any() else 0.0, 0.1))
            for x, y in zip(X.T, Y.T)
        ]
    )
    want_dur = np.maximum(raw_dur - _lag_share(Y), np.log(0.1))
    assert np.allclose(params[-1].coarse_dur.alpha0, want_dur)


def test_fine_intercept_is_the_mean_log_block_share(dataset):
    """Re-derived by looping over the children of each parent epoch."""
    cfg, _, ds = dataset
    params = initialize_from_data(
        ds.data, cfg.M_k, cfg.r_k, cfg.max_lag, d_ctx=2, seed=0
    )
    for block_name, obs in (
        ("fine_count", ds.data.get_X(1)),
        ("fine_dur", ds.data.get_Y(1)),
    ):
        smoothed = obs.astype(float) + 0.5
        share = np.zeros_like(smoothed)
        for c in range(ds.hierarchy.n_epochs(2)):
            children = ds.hierarchy.children(1, c)
            block = smoothed[children]
            share[children] = np.log(block / block.sum(axis=0))
        field = "beta0" if block_name == "fine_count" else "alpha0"
        got = getattr(getattr(params[0], block_name), field)
        assert np.allclose(got, share.mean(axis=0))


def test_dispersion_recovers_a_known_negative_binomial():
    """`var = mu + mu^2 / r` inverted the other way round would fail this."""
    rng = np.random.default_rng(0)
    r_true, mu_true = 2.0, 5.0
    draws = rng.negative_binomial(r_true, r_true / (r_true + mu_true), size=(8000, 1))
    X = draws.astype(np.int64)
    params = initialize_from_data(_coarse_only(X, X.copy()), [1], [0], [1], seed=0)
    r_hat = params[0].coarse_count.r_disp[0]
    assert 1.6 < r_hat < 2.5, r_hat
    assert np.isclose(np.exp(params[0].coarse_count.beta0[0]), mu_true, rtol=0.15)


def test_dispersion_separates_poisson_from_overdispersed():
    rng = np.random.default_rng(1)
    poisson = rng.poisson(5.0, size=(4000, 1))
    heavy = rng.negative_binomial(1.0, 1.0 / 6.0, size=(4000, 1))
    X = np.hstack([poisson, heavy]).astype(np.int64)
    params = initialize_from_data(_coarse_only(X, X.copy()), [1], [0], [1], seed=0)
    r_disp = params[0].coarse_count.r_disp
    assert r_disp[0] > 10.0 * r_disp[1]


def test_zero_inflation_logit_rises_with_the_injected_zeros():
    rng = np.random.default_rng(2)
    base = rng.negative_binomial(2.0, 2.0 / 7.0, size=(6000, 1))
    logits = []
    for zero_frac in (0.0, 0.3, 0.6, 0.9):
        col = base.copy()
        col[rng.random(size=col.shape) < zero_frac] = 0
        X = col.astype(np.int64)
        params = initialize_from_data(_coarse_only(X, X.copy()), [1], [0], [1], seed=0)
        logits.append(float(params[0].coarse_count.gamma0[0]))
    assert logits == sorted(logits)
    assert logits[0] < 0.0 < logits[-1]


def test_pca_reproduces_a_rank_two_matrix():
    rng = np.random.default_rng(3)
    truth = rng.normal(size=(50, 2)) @ rng.normal(size=(2, 4))
    truth -= truth.mean(axis=0)
    loadings, scores = _pca(truth, 2)
    assert loadings.shape == (2, 4) and scores.shape == (50, 2)
    assert np.allclose(scores @ loadings, truth)
    assert np.allclose(scores.std(axis=0), 1.0)


def test_pca_leaves_unavailable_components_at_zero():
    """A single column carries one direction, so a second one cannot exist."""
    rng = np.random.default_rng(4)
    loadings, scores = _pca(rng.normal(size=(20, 1)), 3)
    assert loadings.shape == (3, 1) and scores.shape == (20, 3)
    assert np.allclose(loadings[1:], 0.0)
    assert np.allclose(scores[:, 1:], 0.0)


def test_ar1_recovers_a_known_coefficient():
    rng = np.random.default_rng(5)
    noise = rng.normal(size=20000)
    series = np.zeros(20000)
    for t in range(1, series.size):
        series[t] = 0.7 * series[t - 1] + noise[t]
    phi, innov = _ar1(series[:, None])
    assert np.isclose(phi[0], 0.7, atol=0.02)
    assert np.isclose(innov[0], 1.0, rtol=0.05)


def test_ar1_regresses_the_lead_on_the_lag():
    """
    A geometric decay fixes the coefficient exactly at its ratio.

    Every step of `z_t = 0.5^t` is half the one before, so the regression of the
    lead on the lag is 0.5 with no residual left over; regressing the other way
    round would give 2 instead.
    """
    phi, innov = _ar1((0.5 ** np.arange(30.0))[:, None])
    assert np.isclose(phi[0], 0.5)
    # Nothing is unexplained, so the innovation lands on its floor.
    assert np.isclose(innov[0], 1e-3)


def test_ar1_clips_a_random_walk_below_one():
    rng = np.random.default_rng(6)
    walk = np.cumsum(rng.normal(size=(4000, 1)), axis=0)
    phi, innov = _ar1(walk)
    assert 0.9 < phi[0] <= 0.99
    assert innov[0] > 0.0


# ---------------------------------------------------------------------------
# The pipeline accepts the result unchanged
# ---------------------------------------------------------------------------


def test_estep_and_mstep_run(dataset):
    cfg, gen, ds = dataset
    params = initialize_from_data(
        ds.data, cfg.M_k, cfg.r_k, cfg.max_lag, d_ctx=2, seed=0
    )
    graph = MultiScaleDBNGraph.empty(ds.hierarchy, gen.label_names)

    q_coarse = EStep().run(cfg.K, ds.data, params[-1], graph, None)
    assert np.isfinite(q_coarse.log_Z)
    assert np.allclose(q_coarse.gamma.sum(axis=1), 1.0)
    updated = MStep().run(cfg.K, q_coarse, params[-1], ds.data, ds.hierarchy)
    _assert_finite([updated])

    q_fine = EStep().run(1, ds.data, params[0], graph, q_coarse)
    assert np.isfinite(q_fine.log_Z)
    assert np.allclose(q_fine.gamma.sum(axis=1), 1.0)
    updated_fine = MStep().run(
        1, q_fine, params[0], ds.data, ds.hierarchy, parent_summaries=q_coarse
    )
    _assert_finite([updated_fine])
    # The M-step has to be able to move off the start; a frozen block would mean
    # the initializer handed it a point its own gradient cannot leave.
    assert not np.array_equal(updated.coarse_count.beta0, params[-1].coarse_count.beta0)


def test_sem_fits_two_sweeps(dataset):
    cfg, gen, ds = dataset
    params = initialize_from_data(
        ds.data, cfg.M_k, cfg.r_k, cfg.max_lag, d_ctx=2, seed=0
    )
    graph = MultiScaleDBNGraph.empty(ds.hierarchy, gen.label_names)
    sem = TopDownSEM(max_outer_iter=2, do_structure=True, verbose=0)
    learned, learned_graph, history = sem.fit(ds.data, params, graph)

    # A rejected sweep never reaches history.elbo, so both lists count.
    assert len(history.elbo) + len(history.rejected) >= 2
    assert history.elbo and all(np.isfinite(v) for v in history.elbo)
    # Whatever the structure step found has to be mirrored in the parent sets
    # the emission actually reads its lag coefficients from.
    for k, src, lag, dst in learned_graph.iter_edges():
        assert (src, lag) in learned[k - 1].parent_sets[dst]
    _assert_finite(learned)


def test_cross_scale_block_is_filled_by_the_first_sweep(dataset):
    """The finer scale starts with a zero coupling and the warm start moves it."""
    cfg, gen, ds = dataset
    params = initialize_from_data(
        ds.data, cfg.M_k, cfg.r_k, cfg.max_lag, d_ctx=2, seed=0
    )
    assert np.allclose(params[0].regime.cross, 0.0)
    assert params[-1].regime.cross is None

    graph = MultiScaleDBNGraph.empty(ds.hierarchy, gen.label_names)
    learned, _, _ = TopDownSEM(max_outer_iter=1, do_structure=False, verbose=0).fit(
        ds.data, params, graph
    )
    assert np.all(np.isfinite(learned[0].regime.cross))
    assert np.abs(learned[0].regime.cross).max() > 0.0


# ---------------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------------


def test_label_that_never_fires_stays_finite(dataset):
    cfg, _, ds = dataset
    data = deepcopy(ds.data)
    for k in range(1, cfg.K + 1):
        data.get_X(k)[:, 0] = 0
        data.get_Y(k)[:, 0] = 0
    assert data.check_reconciliation() == {"RecErr_X": 0.0, "RecErr_Y": 0.0}

    params = initialize_from_data(data, cfg.M_k, cfg.r_k, cfg.max_lag, d_ctx=2, seed=0)
    _assert_finite(params)
    coarse = params[-1]
    assert coarse.coarse_count.r_disp[0] > 0
    assert coarse.coarse_dur.s_disp[0] > 0
    # The floor on the mean is what keeps the empty column out of log(0).
    assert np.isclose(coarse.coarse_count.beta0[0], np.log(0.1))
    assert np.isclose(coarse.coarse_dur.alpha0[0], np.log(0.1))


def test_all_zero_data_stays_finite(dataset):
    """Nothing ever happens: every moment fit falls back on its floor."""
    cfg, _, ds = dataset
    data = deepcopy(ds.data)
    for k in range(1, cfg.K + 1):
        data.get_X(k)[:] = 0
        data.get_Y(k)[:] = 0
    params = initialize_from_data(data, cfg.M_k, cfg.r_k, cfg.max_lag, d_ctx=2, seed=0)
    _assert_finite(params)
    coarse = params[-1]
    assert np.allclose(coarse.coarse_count.beta0, np.log(0.1))
    # No epoch separates from any other, so no regime contrast can exist.
    assert np.allclose(coarse.coarse_count.betaS, 0.0)
    # A block whose children all took nothing splits its parent evenly.
    fine = params[0]
    assert np.allclose(fine.fine_count.beta0, np.log(1.0 / 10))


def test_counts_without_durations_stay_finite():
    rng = np.random.default_rng(7)
    X = rng.poisson(3.0, size=(40, 2)).astype(np.int64)
    Y = np.zeros_like(X)
    params = initialize_from_data(_coarse_only(X, Y), [2], [1], [1], seed=0)
    _assert_finite(params)
    assert np.allclose(params[0].coarse_dur.alpha0, np.log(0.1))


def test_single_scale_hierarchy():
    """K=1 makes scale 1 the coarsest, so no fine block is built at all."""
    rng = np.random.default_rng(8)
    X = rng.poisson(3.0, size=(40, 2)).astype(np.int64)
    Y = (rng.poisson(4.0, size=(40, 2)) * (X > 0)).astype(np.int64)
    params = initialize_from_data(_coarse_only(X, Y), [2], [1], [1], seed=0)
    assert len(params) == 1
    assert params[0].is_coarsest
    assert params[0].fine_count is None and params[0].fine_dur is None
    assert params[0].regime.cross is None
    _assert_finite(params)


def test_single_label_and_single_epoch():
    """One label, one epoch: no transition and no lagged score to fit."""
    X = np.array([[4]], dtype=np.int64)
    params = initialize_from_data(_coarse_only(X, X.copy()), [2], [1], [1], seed=0)
    _assert_finite(params)
    # With no observed transition the BDeu prior alone decides, so the table is
    # uniform over the two regimes and the AR(1) has nothing to lean on.
    assert np.allclose(params[0].regime.trans, np.log(0.5))
    assert np.allclose(params[0].factor.F, 0.0)
    assert np.all(params[0].factor.Sigma > 0.0)


def test_more_regimes_than_epochs():
    """
    Three regimes over two epochs leaves one cluster no one can occupy.

    An unoccupied cluster takes the overall column mean, which after centering
    is exactly a zero contrast; anything else would invent a regime effect for a
    regime no epoch ever entered.
    """
    X = np.array([[1, 9], [9, 1]], dtype=np.int64)
    params = initialize_from_data(_coarse_only(X, X.copy()), [3], [2], [1], seed=0)
    _assert_finite(params)

    betaS = params[0].coarse_count.betaS
    assert betaS.shape == (3, 2)
    assert np.allclose(betaS.mean(axis=0), 0.0, atol=1e-10)
    gap = (np.log1p(9) - np.log1p(1)) / 2
    assert np.allclose(np.sort(np.abs(betaS).max(axis=1)), [0.0, gap, gap])
    # An unvisited regime still gets a proper transition column.
    assert np.allclose(np.exp(params[0].regime.trans).sum(axis=0), 1.0)


def test_kmeans_single_cluster_ignores_the_features():
    labels = _kmeans(np.arange(12.0).reshape(6, 2), 1, np.random.default_rng(0))
    assert labels.shape == (6,)
    assert np.all(labels == 0)


def test_kmeans_separates_two_groups():
    feat = np.vstack([np.zeros((8, 2)), np.full((8, 2), 10.0)])
    labels = _kmeans(feat, 2, np.random.default_rng(1))
    assert len(np.unique(labels)) == 2
    assert len(np.unique(labels[:8])) == 1
    assert labels[0] != labels[-1]


def test_kmeans_handles_more_clusters_than_rows():
    labels = _kmeans(np.array([[0.0], [5.0]]), 4, np.random.default_rng(2))
    assert labels.shape == (2,)
    assert labels.min() >= 0 and labels.max() < 4
    # Two rows five apart cannot share a center once the loop settles.
    assert labels[0] != labels[1]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_same_seed_gives_identical_params(dataset):
    cfg, _, ds = dataset
    first = initialize_from_data(
        ds.data, cfg.M_k, cfg.r_k, cfg.max_lag, d_ctx=2, seed=7
    )
    second = initialize_from_data(
        ds.data, cfg.M_k, cfg.r_k, cfg.max_lag, d_ctx=2, seed=7
    )
    for a, b in zip(first, second):
        left, right = dict(_arrays(a)), dict(_arrays(b))
        assert left.keys() == right.keys()
        for tag in left:
            assert np.array_equal(left[tag], right[tag]), tag


def test_seed_moves_only_the_clustered_blocks(dataset):
    """
    The intercepts and dispersions read the columns, never the clustering.

    A different seed may land on a different k-means path, so the regime
    contrasts and the factor may move; an intercept that moved with it would
    mean a moment estimate had picked up the labels by accident.
    """
    cfg, _, ds = dataset
    other = initialize_from_data(
        ds.data, cfg.M_k, cfg.r_k, cfg.max_lag, d_ctx=2, seed=99
    )
    reference = initialize_from_data(
        ds.data, cfg.M_k, cfg.r_k, cfg.max_lag, d_ctx=2, seed=7
    )
    _assert_finite(other)
    for a, b in zip(other, reference):
        assert _shapes(a) == _shapes(b)
    for a, b in ((other[-1], reference[-1]),):
        assert np.array_equal(a.coarse_count.beta0, b.coarse_count.beta0)
        assert np.array_equal(a.coarse_count.r_disp, b.coarse_count.r_disp)
        assert np.array_equal(a.coarse_count.gamma0, b.coarse_count.gamma0)
        assert np.array_equal(a.coarse_dur.alpha0, b.coarse_dur.alpha0)
        assert np.array_equal(a.coarse_dur.s_disp, b.coarse_dur.s_disp)
    assert np.array_equal(other[0].fine_count.beta0, reference[0].fine_count.beta0)
    assert np.array_equal(other[0].fine_dur.alpha0, reference[0].fine_dur.alpha0)


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def test_wrong_length_lists_raise(dataset):
    cfg, _, ds = dataset
    with pytest.raises(ValueError, match="one per scale"):
        initialize_from_data(ds.data, [2], cfg.r_k, cfg.max_lag)


def test_context_width_mismatch_raises(dataset):
    cfg, _, ds = dataset
    with pytest.raises(ValueError, match="does not match the observed context"):
        initialize_from_data(ds.data, cfg.M_k, cfg.r_k, cfg.max_lag, d_ctx=5)


def test_dropping_an_observed_context_raises(dataset):
    """
    d_ctx=0 against a context-carrying dataset is rejected here, not later.

    `posterior.regime_offsets` decides whether to apply the context block from
    the width of the observed matrix, not from d_ctx, so a zero-width block
    against a two-column context is a matmul error inside the E-step.
    """
    cfg, _, ds = dataset
    with pytest.raises(ValueError, match="does not match the observed context"):
        initialize_from_data(ds.data, cfg.M_k, cfg.r_k, cfg.max_lag, d_ctx=0)


def _two_scale_data(widths: List[int]) -> MultiScaleData:
    """Two scales whose context blocks carry the given widths."""
    rng = np.random.default_rng(4)
    X_fine = rng.poisson(0.6, size=(40, 2)).astype(np.int64)
    X_coarse = X_fine.reshape(10, 4, 2).sum(axis=1)
    Y_fine = (rng.poisson(3.0, size=(40, 2)) * (X_fine > 0)).astype(np.int64)
    Y_coarse = Y_fine.reshape(10, 4, 2).sum(axis=1)
    hier = ScaleHierarchy.from_uniform_branching(40, [4])
    return MultiScaleData(
        hierarchy=hier,
        label_names=["l0", "l1"],
        X=[X_fine, X_coarse],
        Y=[Y_fine, Y_coarse],
        C=[
            rng.standard_normal((40, widths[0])),
            rng.standard_normal((10, widths[1])),
        ],
    )


def test_a_context_width_may_differ_between_scales():
    """A calendar feature can live at one scale and not another."""
    data = _two_scale_data([0, 3])
    params = initialize_from_data(data, [2, 2], [1, 1], [1, 1], d_ctx=[0, 3], seed=0)
    assert params[0].regime.ctx.shape == (2, 0)
    assert params[1].regime.ctx.shape == (2, 3)
    assert params[0].fine_count.betaCtx.shape == (0, 2)
    assert params[1].coarse_count.betaCtx.shape == (3, 2)


def test_a_per_scale_width_that_disagrees_with_the_data_raises():
    data = _two_scale_data([0, 3])
    with pytest.raises(ValueError, match="does not match the observed context"):
        initialize_from_data(data, [2, 2], [1, 1], [1, 1], d_ctx=[3, 0], seed=0)


def test_a_scalar_width_against_differing_scales_raises():
    """The uniform reading is kept, and it is what a differing dataset fails."""
    data = _two_scale_data([0, 3])
    with pytest.raises(ValueError, match="does not match the observed context"):
        initialize_from_data(data, [2, 2], [1, 1], [1, 1], d_ctx=3, seed=0)


def test_a_per_scale_width_of_the_wrong_length_raises():
    data = _two_scale_data([0, 3])
    with pytest.raises(ValueError, match="one per scale"):
        initialize_from_data(data, [2, 2], [1, 1], [1, 1], d_ctx=[0, 3, 0], seed=0)


def test_zero_context_runs_end_to_end():
    """Data that carries no context at all keeps the empty context blocks."""
    rng = np.random.default_rng(9)
    X_fine = rng.poisson(0.6, size=(40, 2)).astype(np.int64)
    X_coarse = X_fine.reshape(10, 4, 2).sum(axis=1)
    Y_fine = (rng.poisson(3.0, size=(40, 2)) * (X_fine > 0)).astype(np.int64)
    Y_coarse = Y_fine.reshape(10, 4, 2).sum(axis=1)
    data = _hand_made([X_fine, X_coarse], [Y_fine, Y_coarse], [4], d_ctx=0)
    assert data.check_reconciliation() == {"RecErr_X": 0.0, "RecErr_Y": 0.0}

    params = initialize_from_data(data, [2, 2], [1, 1], [1, 1], d_ctx=0, seed=0)
    for scale in params:
        assert scale.regime.ctx.shape == (scale.M, 0)
        assert scale.factor.G.shape == (scale.M, scale.r, 0)
    _assert_finite(params)

    graph = MultiScaleDBNGraph.empty(data.hierarchy, data.label_names)
    q_coarse = EStep().run(2, data, params[-1], graph, None)
    q_fine = EStep().run(1, data, params[0], graph, q_coarse)
    assert np.isfinite(q_coarse.log_Z) and np.isfinite(q_fine.log_Z)


# ---------------------------------------------------------------------------
# A level that does not absorb the edges
# ---------------------------------------------------------------------------


def test_a_lagless_series_gives_up_none_of_its_level():
    """With no lagged structure the adjustment is exactly zero, so nothing moves."""
    from src.data.initialize import _lag_share

    rng = np.random.default_rng(0)
    flat = rng.negative_binomial(2.0, 2.0 / 7.0, size=(4000, 2)).astype(float)
    assert np.allclose(_lag_share(flat), 0.0)


def test_a_series_driven_by_its_own_past_gives_up_part_of_its_level():
    from src.data.initialize import _lag_share

    rng = np.random.default_rng(1)
    x = np.zeros((500, 1))
    for t in range(1, 500):
        x[t, 0] = rng.poisson(min(2.0 + 0.6 * x[t - 1, 0], 500.0))
    assert _lag_share(x)[0] > 0.1


def test_the_adjustment_only_ever_lowers_a_level():
    # A net inhibitory lag would otherwise raise the level, which recreates the
    # absorption in the other direction.
    from src.data.initialize import _lag_share

    rng = np.random.default_rng(2)
    x = np.zeros((400, 1))
    for t in range(1, 400):
        x[t, 0] = rng.poisson(max(8.0 - 1.2 * x[t - 1, 0], 0.05))
    assert _lag_share(x)[0] >= 0.0


def test_a_series_too_short_to_regress_keeps_its_level():
    from src.data.initialize import _lag_share

    assert np.allclose(_lag_share(np.array([[3.0, 1.0], [4.0, 2.0]])), 0.0)


def test_searching_before_the_parameter_step_is_off_by_default(dataset):
    cfg, _, ds = dataset
    assert TopDownSEM().structure_first is False


def test_searching_before_the_parameter_step_still_fits(dataset):
    """The level a search meets is the one it was handed, not one refitted first."""
    cfg, _, ds = dataset
    params = initialize_from_data(
        ds.data, cfg.M_k, cfg.r_k, cfg.max_lag, d_ctx=2, seed=0
    )
    graph = MultiScaleDBNGraph(
        hierarchy=ds.data.hierarchy, label_names=list(ds.data.label_names)
    )
    fitted, learned, history = TopDownSEM(
        max_outer_iter=2, structure_first=True, verbose=0
    ).fit(ds.data, params, graph)
    assert len(fitted) == cfg.K
    assert history.elbo
    assert learned.n_edges(cfg.K) >= 0


def test_a_singular_design_leaves_the_level_alone(monkeypatch):
    """A level that cannot be adjusted is kept, not dropped to zero."""
    import src.data.initialize as init_mod

    def _raise(*_args, **_kwargs):
        raise np.linalg.LinAlgError("singular")

    monkeypatch.setattr(init_mod.np.linalg, "solve", _raise)
    values = np.arange(30, dtype=float).reshape(10, 3)
    assert np.allclose(init_mod._lag_share(values), 0.0)


def test_restarts_are_off_by_default_so_a_fit_costs_what_it_did():
    assert TopDownSEM().n_restarts == 1


def test_jittering_moves_the_levels_and_nothing_else(dataset):
    from src.learning.structural_em import _jitter_start

    cfg, _, ds = dataset
    params = initialize_from_data(
        ds.data, cfg.M_k, cfg.r_k, cfg.max_lag, d_ctx=2, seed=0
    )
    moved = _jitter_start(params, seed=1)
    coarse, coarse_moved = params[-1].coarse_count, moved[-1].coarse_count
    # Widening this to the loadings was measured and is worse, so the narrow
    # perturbation is the shipped one and the test pins it.
    assert not np.allclose(coarse_moved.beta0, coarse.beta0)
    if coarse.betaH.size:
        assert np.allclose(coarse_moved.betaH, coarse.betaH)
    assert np.allclose(coarse_moved.betaS, coarse.betaS)
    assert np.allclose(coarse_moved.r_disp, coarse.r_disp)


def test_a_restarting_fit_keeps_the_best_objective(dataset):
    """More starts may not help, and must never return a worse one."""
    cfg, _, ds = dataset
    graph = MultiScaleDBNGraph(
        hierarchy=ds.data.hierarchy, label_names=list(ds.data.label_names)
    )
    kw = dict(max_outer_iter=2, do_structure=False, verbose=0)
    params = initialize_from_data(
        ds.data, cfg.M_k, cfg.r_k, cfg.max_lag, d_ctx=2, seed=0
    )
    _, _, one = TopDownSEM(**kw).fit(ds.data, params, graph)
    params = initialize_from_data(
        ds.data, cfg.M_k, cfg.r_k, cfg.max_lag, d_ctx=2, seed=0
    )
    _, _, many = TopDownSEM(n_restarts=3, **kw).fit(ds.data, params, graph)
    assert many.elbo[-1] >= one.elbo[-1] - 1e-6
