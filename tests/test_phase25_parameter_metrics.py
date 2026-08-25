"""
tests/test_phase25_parameter_metrics.py
=======================================
Parameter recovery scoring: how far learned parameters sit from the true ones.

The regime label carries no meaning of its own, so the whole comparison is only
defined once a single relabeling has been fixed for every regime-indexed block.
The alignment test below is what keeps the rest of the numbers honest: a copy of
the true parameters whose regimes have been swapped has to score as perfect.

The same argument runs one level deeper. Several parameter blocks cancel out of
every distribution the model can produce, so two parameter sets that differ only
along those directions generate byte-identical data and have to score identical
as well; otherwise the metric punishes a learner for something it cannot see.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from src.evaluation.metrics import (
    ParameterMetrics,
    ReconciliationMetrics,
    _best_param_permutation,
    aggregate_metrics,
    edge_auprc,
    r2_sortability,
    transition_cpt,
    varsortability,
)
from src.synthetic.families import make_generator
from src.synthetic.scenarios import ScenarioConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _cfg(**over) -> ScenarioConfig:
    base = dict(
        name="param",
        K=2,
        H_finest=120,
        branching=[10],
        L=3,
        M_k=[2, 2],
        r_k=[0, 0],
        max_lag=[2, 1],
        max_indegree=[1, 1],
        zi_level="medium",
        family=2,
        seed=4,
        n_replicates=1,
        edge_strength=10.0,
        regime_strength=3.0,
        finest_rate=0.6,
        slot_minutes=1,
    )
    base.update(over)
    return ScenarioConfig(**base)


def _dataset(mutate=None, **over):
    """Generate a small scenario, optionally mutating the parameters first."""
    cfg = _cfg(**over)
    gen = make_generator(cfg, seed=cfg.seed)
    if mutate is not None:
        mutate(gen.params_list)
    return gen.generate(seed=cfg.seed)


def _params(**over):
    """Ground-truth ScaleParams of a small generated scenario, finest first."""
    return _dataset(**over).params


@pytest.fixture(scope="module")
def true_params():
    return _params()


@pytest.fixture(scope="module")
def factor_params():
    return _params(r_k=[1, 1], family=3)


@pytest.fixture(scope="module")
def zero_row_params():
    """Two-factor parameters whose first finest-scale loading vector is zero."""
    params = _params(r_k=[2, 2], family=3)
    params[0].fine_count.betaH[0] = 0.0
    return params


def _swap_regimes(params):
    """Relabel every regime-indexed block of a copy, regime 0 for regime 1."""
    swapped = deepcopy(params)
    order = np.array([1, 0])
    for p in swapped:
        reg = p.regime
        reg.init = reg.init[order]
        reg.bias = reg.bias[order]
        reg.trans = reg.trans[np.ix_(order, order)]
        reg.ctx = reg.ctx[order]
        if reg.cross is not None:
            reg.cross = reg.cross[order]
        fac = p.factor
        fac.F = fac.F[order]
        fac.G = fac.G[order]
        fac.Sigma = fac.Sigma[order]
        if p.is_coarsest:
            p.coarse_count.betaS = p.coarse_count.betaS[order]
            p.coarse_count.gammaS = p.coarse_count.gammaS[order]
            p.coarse_dur.alphaS = p.coarse_dur.alphaS[order]
        else:
            p.fine_count.betaS = p.fine_count.betaS[order]
            p.fine_dur.alphaS = p.fine_dur.alphaS[order]
    return swapped


def _perturb(params, size, seed=0):
    """Add Gaussian noise of a given size to every scored parameter block."""
    rng = np.random.default_rng(seed)
    out = deepcopy(params)

    def jitter(arr):
        return arr + rng.standard_normal(arr.shape) * size

    for p in out:
        p.regime.bias = jitter(p.regime.bias)
        p.regime.trans = jitter(p.regime.trans)
        if p.is_coarsest:
            c, d = p.coarse_count, p.coarse_dur
            c.beta0 = jitter(c.beta0)
            c.betaS = jitter(c.betaS)
            c.r_disp = jitter(c.r_disp)
            for key in c.betaLag:
                c.betaLag[key] = jitter(c.betaLag[key])
            for key in d.alphaLag:
                d.alphaLag[key] = jitter(d.alphaLag[key])
            if c.betaH.shape[0] > 0:
                c.betaH = jitter(c.betaH)
        else:
            f, g = p.fine_count, p.fine_dur
            f.beta0 = jitter(f.beta0)
            f.betaS = jitter(f.betaS)
            f.betaP = jitter(f.betaP)
            f.betaD = jitter(f.betaD)
            for key in f.betaLag:
                f.betaLag[key] = jitter(f.betaLag[key])
            for key in g.alphaLag:
                g.alphaLag[key] = jitter(g.alphaLag[key])
            if f.betaH.shape[0] > 0:
                f.betaH = jitter(f.betaH)
    return out


# ---------------------------------------------------------------------------
# transition_cpt
# ---------------------------------------------------------------------------


def test_transition_cpt_is_row_stochastic(true_params):
    P = transition_cpt(true_params[0].regime)
    M = true_params[0].M
    assert P.shape == (M, M)
    assert np.allclose(P.sum(axis=1), 1.0)
    assert np.all(P > 0.0)


def test_transition_cpt_matches_the_regime_logits(true_params):
    reg = true_params[0].regime
    P = transition_cpt(reg)
    zero_ctx = np.zeros(reg.d_ctx)
    for prev in range(reg.M):
        logits = reg.logit_transition(prev, None, zero_ctx)
        expected = np.exp(logits - logits.max())
        expected /= expected.sum()
        assert np.allclose(P[prev], expected)


def test_transition_cpt_averages_the_cross_term(true_params):
    """The finest scale carries a cross block, so a parent share moves the table."""
    reg = true_params[0].regime
    assert reg.cross is not None
    share = np.array([1.0] + [0.0] * (reg.parent_M - 1))
    P = transition_cpt(reg, parent_share=share)
    assert np.allclose(P.sum(axis=1), 1.0)
    assert not np.allclose(P, transition_cpt(reg))

    logits = reg.bias + reg.trans[:, 0] + reg.cross[:, 0]
    expected = np.exp(logits - logits.max())
    expected /= expected.sum()
    assert np.allclose(P[0], expected)


def test_transition_cpt_ignores_parent_share_without_a_cross_block(true_params):
    """The coarsest scale has no parent, so a share cannot change its table."""
    reg = true_params[-1].regime
    assert reg.cross is None
    share = np.ones(2) / 2
    assert np.allclose(transition_cpt(reg, parent_share=share), transition_cpt(reg))


# ---------------------------------------------------------------------------
# Permutation alignment
# ---------------------------------------------------------------------------


def test_identity_permutation_for_identical_params(true_params):
    for p in true_params:
        assert list(_best_param_permutation(p, deepcopy(p))) == list(range(p.M))


def test_permutation_recovers_a_swap(true_params):
    swapped = _swap_regimes(true_params)
    for tp, sp in zip(true_params, swapped):
        assert list(_best_param_permutation(tp, sp)) == [1, 0]


def test_greedy_permutation_above_the_enumeration_limit():
    """M above 8 switches to greedy matching, which still recovers a swap."""
    cfg = _cfg(M_k=[9, 9], H_finest=20, branching=[10])
    gen = make_generator(cfg, seed=cfg.seed)
    params = gen.params_list
    assert params[0].M == 9

    order = np.roll(np.arange(9), 1)
    shifted = deepcopy(params)
    shifted[0].fine_count.betaS = shifted[0].fine_count.betaS[order]

    perm = _best_param_permutation(params[0], shifted[0])
    assert sorted(perm) == list(range(9))
    # `order` moves true regime m to row m of the shifted block, so the inverse
    # of `order` is the alignment the metric has to find.
    assert list(perm) == list(np.argsort(order))


# ---------------------------------------------------------------------------
# ParameterMetrics
# ---------------------------------------------------------------------------


def test_identical_params_score_zero(true_params):
    m = ParameterMetrics.compute(true_params, deepcopy(true_params))
    d = m.to_dict()
    K = len(true_params)
    for k in range(1, K + 1):
        assert d[f"TransTV_k{k}"] == pytest.approx(0.0, abs=1e-12)
        assert d[f"LagErr_k{k}"] == 0.0
        assert d[f"EmisErr_k{k}"] == 0.0
        assert d[f"EmisRel_k{k}"] == 0.0
    assert d["AllocShare_k1"] == pytest.approx(0.0, abs=1e-12)
    assert np.isnan(d[f"AllocShare_k{K}"])
    assert d["ParamErr_mean"] == pytest.approx(0.0, abs=1e-12)


def test_the_mean_reports_how_many_errors_it_was_taken_over(true_params):
    m = ParameterMetrics.compute(true_params, deepcopy(true_params))
    scored = m.trans_tv + m.emis_rel + m.fact_sub + m.alloc_share
    assert m.param_err_n == len([v for v in scored if not np.isnan(v)])
    assert m.to_dict()["ParamErr_n"] == float(m.param_err_n)


def test_a_model_without_a_regime_averages_fewer_errors_than_one_with():
    """The count is what makes two families' means incomparable, so it is scored."""
    with_regime = _params(M_k=[2, 2], r_k=[0, 0], family=2)
    without = _params(M_k=[1, 1], r_k=[0, 0], family=1)
    n_with = ParameterMetrics.compute(with_regime, deepcopy(with_regime)).param_err_n
    n_without = ParameterMetrics.compute(without, deepcopy(without)).param_err_n
    assert n_without < n_with


def test_identical_params_leave_no_false_lag_mass(true_params):
    m = ParameterMetrics.compute(true_params, deepcopy(true_params))
    for k, value in enumerate(m.lag_false, start=1):
        assert value == pytest.approx(0.0), f"scale {k} claims mass off the support"


def test_permuted_regimes_still_score_zero(true_params):
    """Relabeling the regimes changes no distribution, so nothing may move."""
    swapped = _swap_regimes(true_params)
    aligned = ParameterMetrics.compute(true_params, swapped).to_dict()
    identical = ParameterMetrics.compute(true_params, deepcopy(true_params)).to_dict()
    for key, value in identical.items():
        if np.isnan(value):
            assert np.isnan(aligned[key])
        else:
            assert aligned[key] == pytest.approx(value, abs=1e-10)


def test_unaligned_comparison_would_score_worse(true_params):
    """Without the alignment the swapped copy looks wrong, which is the point."""
    swapped = _swap_regimes(true_params)
    aligned = ParameterMetrics.compute(true_params, swapped)

    naive = 0.0
    for tp, sp in zip(true_params, swapped):
        t_bs = tp.fine_count.betaS if not tp.is_coarsest else tp.coarse_count.betaS
        s_bs = sp.fine_count.betaS if not sp.is_coarsest else sp.coarse_count.betaS
        naive += float(np.mean(np.abs(t_bs - s_bs)))
    assert naive > 0.1
    assert max(aligned.emis_err) < 1e-10


def test_error_grows_with_the_perturbation(true_params):
    sizes = [0.0, 0.05, 0.2, 0.5]
    means = [
        ParameterMetrics.compute(
            true_params, _perturb(true_params, s, seed=1)
        ).param_err_mean
        for s in sizes
    ]
    assert means[0] == pytest.approx(0.0, abs=1e-12)
    assert all(a < b for a, b in zip(means, means[1:]))


def test_transition_error_grows_with_the_perturbation(true_params):
    small = ParameterMetrics.compute(true_params, _perturb(true_params, 0.05, seed=2))
    large = ParameterMetrics.compute(true_params, _perturb(true_params, 0.6, seed=2))
    assert small.trans_tv[0] < large.trans_tv[0]
    assert 0.0 <= large.trans_tv[0] <= 1.0


def test_single_regime_gives_nan_transition_error():
    params = _params(M_k=[1, 1], family=1)
    m = ParameterMetrics.compute(params, deepcopy(params))
    assert all(np.isnan(v) for v in m.trans_tv)
    assert not np.isnan(m.param_err_mean)


def test_no_factor_gives_nan_subspace_distance(true_params):
    m = ParameterMetrics.compute(true_params, deepcopy(true_params))
    assert all(np.isnan(v) for v in m.fact_sub)


def test_factor_subspace_distance_is_zero_for_a_copy(factor_params):
    m = ParameterMetrics.compute(factor_params, deepcopy(factor_params))
    assert all(v == pytest.approx(0.0, abs=1e-9) for v in m.fact_sub)


def test_factor_subspace_distance_ignores_a_sign_flip(factor_params):
    """A loading vector and its negation span the same row space."""
    flipped = deepcopy(factor_params)
    flipped[0].fine_count.betaH = -flipped[0].fine_count.betaH
    m = ParameterMetrics.compute(factor_params, flipped)
    assert m.fact_sub[0] == pytest.approx(0.0, abs=1e-9)


def test_factor_subspace_distance_penalizes_a_rotated_loading(factor_params):
    rotated = deepcopy(factor_params)
    rotated[0].fine_count.betaH = np.roll(factor_params[0].fine_count.betaH, 1, axis=1)
    m = ParameterMetrics.compute(factor_params, rotated)

    a = factor_params[0].fine_count.betaH[0]
    b = np.roll(a, 1)
    expected = 1.0 - abs(float(a @ b)) / float(np.linalg.norm(a) * np.linalg.norm(b))
    assert expected > 1e-3
    assert m.fact_sub[0] == pytest.approx(expected)


def test_factor_subspace_distance_skips_a_vanishing_true_loading(zero_row_params):
    """A zero loading vector points nowhere, so it cannot be scored a miss."""
    m = ParameterMetrics.compute(zero_row_params, deepcopy(zero_row_params))
    assert zero_row_params[0].r == 2
    assert m.fact_sub[0] == pytest.approx(0.0)


def test_factor_subspace_distance_is_nan_when_every_loading_vanishes(zero_row_params):
    blank = deepcopy(zero_row_params)
    blank[0].fine_count.betaH = np.zeros_like(blank[0].fine_count.betaH)
    m = ParameterMetrics.compute(blank, deepcopy(blank))
    assert np.isnan(m.fact_sub[0])
    assert not np.isnan(m.fact_sub[1])


def test_coarsest_scale_gives_nan_allocation_share(true_params):
    m = ParameterMetrics.compute(true_params, deepcopy(true_params))
    assert np.isnan(m.alloc_share[-1])
    assert m.alloc_share[0] == pytest.approx(0.0)


def _shift_fine_beta0(p):
    p[0].fine_count.beta0 = p[0].fine_count.beta0 + 2.5


def _shift_fine_betaP(p):
    p[0].fine_count.betaP = p[0].fine_count.betaP + 5.0


def _shift_fine_betaD(p):
    p[0].fine_count.betaD = p[0].fine_count.betaD - 3.0


def _offset_fine_betaS_columns(p):
    p[0].fine_count.betaS = p[0].fine_count.betaS + np.array([1.0, -2.0, 4.0])


def _trade_coarse_intercept(p):
    c = p[-1].coarse_count
    c.beta0 = c.beta0 + 1.3
    c.betaS = c.betaS - 1.3


@pytest.mark.parametrize(
    "mutate",
    [
        # The fine allocation softmax runs over the children of one block, so
        # every term that is constant inside a block cancels exactly.
        _shift_fine_beta0,
        _shift_fine_betaP,
        _shift_fine_betaD,
        _offset_fine_betaS_columns,
        # At the coarsest scale the count mean sees only beta0 + betaS[s].
        _trade_coarse_intercept,
    ],
)
def test_unidentified_directions_score_zero(true_params, mutate):
    """
    A move along a direction the data cannot see must not register as an error.

    Each mutation leaves the generated dataset byte-identical, so a
    learner that lands on it has recovered the model exactly and the score has
    to say so. `test_unidentified_direction_leaves_the_data_unchanged` pins
    that premise down against the generator itself.
    """
    learned = deepcopy(true_params)
    mutate(learned)
    m = ParameterMetrics.compute(true_params, learned)
    assert m.emis_err == pytest.approx([0.0] * len(true_params), abs=1e-12)
    assert m.emis_rel == pytest.approx([0.0] * len(true_params), abs=1e-12)
    assert m.alloc_share[0] == pytest.approx(0.0, abs=1e-12)
    assert m.param_err_mean == pytest.approx(0.0, abs=1e-12)


def test_unidentified_direction_leaves_the_data_unchanged():
    """The premise of the test above, checked against the generator itself."""
    base = _dataset()
    moved = _dataset(mutate=_shift_fine_beta0)
    for k in (1, 2):
        assert np.array_equal(base.data.get_X(k), moved.data.get_X(k))
        assert np.array_equal(base.data.get_Y(k), moved.data.get_Y(k))
    assert not np.array_equal(
        base.params[0].fine_count.beta0, moved.params[0].fine_count.beta0
    )
    m = ParameterMetrics.compute(base.params, moved.params)
    assert m.param_err_mean == pytest.approx(0.0, abs=1e-12)


def test_emission_error_sees_a_real_regime_contrast(true_params):
    """The regime contrast of betaS is the part the allocation does depend on."""
    shift = 2.0
    learned = deepcopy(true_params)
    learned[0].fine_count.betaS[0, 0] += shift
    m = ParameterMetrics.compute(true_params, learned)
    M, L = true_params[0].M, true_params[0].L
    # Centering splits the shift evenly between the two regimes of that column,
    # so the two moved entries contribute `shift` in total.
    assert m.emis_err[0] == pytest.approx(shift / (M * L))
    # The allocation share sees the contrast, and sees more of a larger one.  A
    # fixed threshold here would be a threshold on one draw of the generator
    # rather than on the metric.
    assert m.alloc_share[0] > 0.0
    wider = deepcopy(true_params)
    wider[0].fine_count.betaS[0, 0] += 2 * shift
    assert ParameterMetrics.compute(true_params, wider).alloc_share[0] > (
        m.alloc_share[0]
    )


def test_relative_emission_error_penalizes_a_flat_true_block(true_params):
    """A flat true block gives an error nowhere to hide."""
    flat = deepcopy(true_params)
    flat[0].fine_count.betaS = np.zeros_like(flat[0].fine_count.betaS)
    learned = deepcopy(flat)
    learned[0].fine_count.betaS = np.array([[0.005] * 3, [-0.005] * 3])
    m = ParameterMetrics.compute(flat, learned)
    assert m.emis_err[0] == pytest.approx(0.005)
    assert m.emis_rel[0] == pytest.approx(0.005 / 1e-6)


def test_lag_error_reads_only_the_true_support(true_params):
    learned = deepcopy(true_params)
    src, lag, dst = sorted(
        (s, d, t) for t, ps in true_params[0].parent_sets.items() for s, d in ps
    )[0]
    learned[0].fine_count.betaLag[(src, lag)][dst] += 0.4
    m = ParameterMetrics.compute(true_params, learned)
    n_true = sum(len(ps) for ps in true_params[0].parent_sets.values())
    # Two blocks are scanned (counts and durations) over the same support.
    assert m.lag_err[0] == pytest.approx(0.4 / (2 * n_true))
    assert m.lag_false[0] == pytest.approx(0.0)


def test_lag_false_counts_mass_off_the_true_support(true_params):
    learned = deepcopy(true_params)
    L = true_params[0].L
    support = {(s, d, t) for t, ps in true_params[0].parent_sets.items() for s, d in ps}
    spurious = next(
        (s, d, t)
        for s in range(L)
        for d in (1, 2)
        for t in range(L)
        if (s, d, t) not in support
    )
    src, lag, dst = spurious
    vec = learned[0].fine_count.betaLag.get((src, lag), np.zeros(L)).copy()
    vec[dst] = -0.7
    learned[0].fine_count.betaLag[(src, lag)] = vec

    base = ParameterMetrics.compute(true_params, deepcopy(true_params))
    m = ParameterMetrics.compute(true_params, learned)
    assert m.lag_false[0] == pytest.approx(base.lag_false[0] + 0.7)
    assert m.lag_err[0] == pytest.approx(base.lag_err[0])


def test_missing_lag_key_counts_as_a_zero_coefficient(true_params):
    """A learned model that dropped an edge is scored by the coefficient it lost."""
    learned = deepcopy(true_params)
    learned[0].fine_count.betaLag = {}
    learned[0].fine_dur.alphaLag = {}
    m = ParameterMetrics.compute(true_params, learned)

    tp = true_params[0]
    support = [(s, d, t) for t, ps in tp.parent_sets.items() for s, d in ps]
    expected = float(
        np.mean(
            [
                abs(float(block.get((src, lag), np.zeros(tp.L))[dst]))
                for block in (tp.fine_count.betaLag, tp.fine_dur.alphaLag)
                for src, lag, dst in support
            ]
        )
    )
    assert expected > 0.0
    assert m.lag_err[0] == pytest.approx(expected)
    assert m.lag_false[0] == pytest.approx(0.0)


def test_empty_true_support_gives_zero_lag_error():
    params = _params(max_indegree=[0, 0])
    assert all(not ps for ps in params[0].parent_sets.values())
    m = ParameterMetrics.compute(params, deepcopy(params))
    assert m.lag_err[0] == 0.0


def test_to_dict_returns_only_floats(true_params):
    d = ParameterMetrics.compute(true_params, _perturb(true_params, 0.1)).to_dict()
    assert all(isinstance(v, float) for v in d.values())
    assert d.keys() >= {"TransTV_k1", "LagFalse_k2", "AllocShare_k1", "ParamErr_mean"}


def test_empty_parameter_lists_give_nan_mean():
    m = ParameterMetrics.compute([], [])
    assert np.isnan(m.param_err_mean)
    assert m.to_dict() == {
        "ParamErr_mean": pytest.approx(float("nan"), nan_ok=True),
        "ParamErr_n": 0.0,
    }


def test_default_construction_is_empty():
    m = ParameterMetrics()
    assert m.trans_tv == []
    assert np.isnan(m.to_dict()["ParamErr_mean"])


def test_mismatched_scale_counts_raise(true_params):
    """Scoring a learned model against a prefix of the truth hides the mistake."""
    with pytest.raises(ValueError, match="scale count mismatch"):
        ParameterMetrics.compute(true_params, deepcopy(true_params[:1]))


# ---------------------------------------------------------------------------
# edge_auprc
# ---------------------------------------------------------------------------


def test_edge_auprc_perfect_ranking():
    scores = {(0, 1, 0): 0.9, (1, 1, 0): 0.8, (0, 2, 1): 0.1}
    assert edge_auprc(scores, {(0, 1, 0), (1, 1, 0)}) == pytest.approx(1.0)


def test_edge_auprc_hand_computed_value():
    scores = {(0, 1, 0): 0.9, (1, 1, 1): 0.8, (0, 2, 2): 0.7, (2, 1, 0): 0.6}
    true_edges = {(0, 1, 0), (0, 2, 2)}
    # Hits at ranks 1 and 3: (1/1 + 2/3) / 2 = 5/6.
    assert edge_auprc(scores, true_edges) == pytest.approx(5.0 / 6.0)


def test_edge_auprc_ties_score_the_base_rate():
    """A scorer that ranks nothing must not profit from the enumeration order."""
    true_edges = {(0, 1, 0), (0, 2, 2)}
    lucky = {(0, 1, 0): 0.5, (0, 2, 2): 0.5, (1, 1, 1): 0.5, (2, 1, 0): 0.5}
    unlucky = {(1, 1, 1): 0.5, (2, 1, 0): 0.5, (0, 1, 0): 0.5, (0, 2, 2): 0.5}
    assert edge_auprc(lucky, true_edges) == pytest.approx(0.5)
    assert edge_auprc(unlucky, true_edges) == pytest.approx(0.5)


def test_edge_auprc_ranks_a_tied_block_by_its_end():
    """A true edge tied with a false one is credited the precision after both."""
    scores = {(0, 1, 0): 0.9, (1, 1, 1): 0.5, (0, 2, 2): 0.5}
    # Hits at ranks 1 and inside the tied block that ends at rank 3: 1 + 2/3.
    assert edge_auprc(scores, {(0, 1, 0), (0, 2, 2)}) == pytest.approx(5.0 / 6.0)


def test_edge_auprc_worst_ranking():
    scores = {(0, 1, 0): 0.9, (1, 1, 1): 0.8, (0, 2, 2): 0.1}
    assert edge_auprc(scores, {(0, 2, 2)}) == pytest.approx(1.0 / 3.0)


def test_edge_auprc_without_true_edges_is_nan():
    assert np.isnan(edge_auprc({(0, 1, 0): 0.5}, set()))


def test_edge_auprc_charges_for_an_unscored_true_edge():
    """A true edge that was never a candidate still counts against the score."""
    scores = {(0, 1, 0): 0.9}
    assert edge_auprc(scores, {(0, 1, 0), (1, 1, 1)}) == pytest.approx(0.5)


def test_edge_auprc_with_no_scores_is_zero():
    assert edge_auprc({}, {(0, 1, 0)}) == 0.0


# ---------------------------------------------------------------------------
# aggregate_metrics
# ---------------------------------------------------------------------------


def test_aggregate_metrics_merges_parameter_metrics(true_params):
    param = ParameterMetrics.compute(true_params, deepcopy(true_params))
    rec = ReconciliationMetrics(rec_err_X=0.0, rec_err_Y=0.0)
    merged = aggregate_metrics(reconciliation=rec, parameter=param)
    assert "RecErr_X" in merged
    assert merged["TransTV_k1"] == pytest.approx(0.0, abs=1e-12)
    assert merged["ParamErr_mean"] == pytest.approx(0.0, abs=1e-12)


def test_aggregate_metrics_without_parameter_metrics_is_unchanged():
    rec = ReconciliationMetrics(rec_err_X=0.1, rec_err_Y=0.2)
    assert aggregate_metrics(reconciliation=rec) == {"RecErr_X": 0.1, "RecErr_Y": 0.2}


# ===========================================================================
# Sortability: whether a generator lets a variance sort stand in for recovery
# ===========================================================================


def _chain(T=4000, seed=0):
    """Three labels in a chain whose marginal variance grows along it."""
    rng = np.random.default_rng(seed)
    x0 = rng.normal(0.0, 1.0, T)
    x1 = 2.0 * x0 + rng.normal(0.0, 1.0, T)
    x2 = 2.0 * x1 + rng.normal(0.0, 1.0, T)
    return np.stack([x0, x1, x2], axis=1), {(0, 1, 1), (1, 1, 2)}


def test_varsortability_is_one_when_variance_grows_along_every_edge():
    """A variance sort would reproduce this graph exactly."""
    X, edges = _chain()
    assert varsortability(X, edges) == pytest.approx(1.0)


def test_varsortability_is_defeated_by_standardising():
    """Equal variances make every comparison a tie, which scores one half."""
    X, edges = _chain()
    Xs = (X - X.mean(axis=0)) / X.std(axis=0)
    assert varsortability(Xs, edges) == pytest.approx(0.5)


def test_r2_sortability_survives_standardising():
    """Explained variance is scale free, so rescaling cannot move it."""
    X, edges = _chain()
    Xs = (X - X.mean(axis=0)) / X.std(axis=0)
    assert r2_sortability(X, edges) == pytest.approx(r2_sortability(Xs, edges))


def test_sortability_of_an_edgeless_graph_is_one_half():
    """With no path to walk there is nothing to order, so neither answer wins."""
    X, _ = _chain()
    assert varsortability(X, set()) == 0.5
    assert r2_sortability(X, set()) == 0.5


def test_sortability_ignores_labels_outside_the_series():
    """An edge naming a column the series does not carry contributes nothing."""
    X, edges = _chain()
    assert varsortability(X, edges | {(9, 1, 0)}) == pytest.approx(
        varsortability(X, edges)
    )


def test_sortability_of_a_single_column_series():
    """One label has no other label to regress on, so its R-squared stays zero."""
    X = np.random.default_rng(0).normal(size=(200, 1))
    assert r2_sortability(X, {(0, 1, 0)}) == 0.5
