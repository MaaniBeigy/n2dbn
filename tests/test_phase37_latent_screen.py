"""
tests/test_phase37_latent_screen.py
===================================
Whether a scenario's latent states can be recovered from its own parameters.

The edge screen already refuses to place a lagged edge at a scale where a true
one scores below the floor a non-edge reaches. A latent state needs the same
guard and did not have one: a regime spending most of its epochs in one state
is recovered no better than always naming that state, so every parameter block
it drives is scored on noise. These tests pin the two floors, the constant
predictor for a regime and the shifted trajectory for a factor, and pin the
generated chain to an occupancy that leaves its states separable at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import best_factor_corr
from src.synthetic.families import (
    REGIME_OCCUPANCY_SPREAD,
    REGIME_PERSISTENCE,
    _make_regime_params,
    _stationary_chain,
)
from src.synthetic.latent_screen import (
    CLEARS,
    FAILS,
    NOT_APPLICABLE,
    LatentVerdict,
    main,
    occupancy,
    oracle_posterior,
    report,
    screen_grid,
    screen_scenario,
    shifted_factor_null,
    supported_regime_counts,
)
from src.synthetic.scenarios import ScenarioConfig, ScenarioGrid

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _cfg(family: int, M: int = 2, r: int = 0) -> ScenarioConfig:
    """A two-scale scenario small enough to screen inside a test."""
    return ScenarioConfig(
        name=f"tiny_F{family}",
        K=2,
        H_finest=360,
        branching=[10],
        L=3,
        M_k=[M, M],
        r_k=[r, r],
        max_lag=[1, 1],
        max_indegree=[0, 1],
        zi_level="medium",
        family=family,
        seed=0,
        n_replicates=1,
        edge_strength=10.0,
        regime_strength=3.0,
        finest_rate=0.15,
        slot_minutes=1,
        finest_binary=True,
    )


# ---------------------------------------------------------------------------
# Occupancy, which is what the regime floor is built from
# ---------------------------------------------------------------------------


def test_occupancy_is_the_share_of_epochs_spent_in_each_state():
    got = occupancy([0, 0, 0, 1], M=2)
    assert got == pytest.approx([0.75, 0.25])


def test_an_empty_path_occupies_nothing_rather_than_dividing_by_zero():
    assert occupancy([], M=3) == pytest.approx([0.0, 0.0, 0.0])


def test_a_state_the_path_never_visits_still_gets_a_share():
    got = occupancy([0, 0], M=3)
    assert len(got) == 3 and got[2] == 0.0


# ---------------------------------------------------------------------------
# The factor floor
# ---------------------------------------------------------------------------


def test_a_trajectory_correlates_with_itself_and_not_with_its_shifts():
    rng = np.random.default_rng(0)
    H = np.cumsum(rng.standard_normal((400, 1)), axis=0)
    assert best_factor_corr(H, H) == pytest.approx(1.0)
    assert shifted_factor_null(H, H, n_shifts=8) < 1.0


def test_a_constant_true_path_has_no_regime_to_recover():
    """A perfect score on a one-state chain would read as recovery, not silence."""
    from src.evaluation.metrics import LatentMetrics

    constant = np.zeros(20, dtype=int)
    switching = np.arange(20) % 2
    got = LatentMetrics.compute(
        [np.ones((20, 1)), np.eye(2)[switching]],
        [constant, switching],
        [np.zeros((20, 0))] * 2,
        [np.zeros((20, 0))] * 2,
    )
    assert np.isnan(got.acc_S_per_scale[0])
    assert np.isnan(got.ari_per_scale[0]) and np.isnan(got.nmi_per_scale[0])
    assert got.acc_S_per_scale[1] == pytest.approx(1.0)


def test_a_fit_that_predicts_one_state_against_a_switching_truth_is_still_scored():
    # What that returns is the modal share, which is the number a constant
    # predictor earns and is worth reporting rather than skipping.
    from src.evaluation.metrics import LatentMetrics

    S_true = np.array([0] * 15 + [1] * 5)
    got = LatentMetrics.compute(
        [np.ones((20, 1))], [S_true], [np.zeros((20, 0))], [np.zeros((20, 0))]
    )
    assert got.acc_S_per_scale[0] == pytest.approx(0.75)


def test_two_constant_labellings_agree_completely_or_not_at_all():
    # LatentMetrics no longer reaches this pair, since it refuses to score a
    # constant truth, but the index itself still has to answer for it.
    from src.evaluation.metrics import _adjusted_rand_index

    const = np.zeros(8, dtype=int)
    assert _adjusted_rand_index(const, const) == pytest.approx(1.0)
    assert _adjusted_rand_index(const, np.ones(8, dtype=int)) == pytest.approx(1.0)


def test_a_scale_with_no_factor_has_no_correlation_to_score():
    empty = np.zeros((10, 0))
    assert np.isnan(best_factor_corr(empty, empty))
    assert np.isnan(shifted_factor_null(empty, empty))


def test_a_single_epoch_carries_no_shift_to_compare_against():
    assert np.isnan(shifted_factor_null(np.zeros((1, 1)), np.zeros((1, 1))))


# ---------------------------------------------------------------------------
# How a verdict reads
# ---------------------------------------------------------------------------


def test_a_regime_verdict_names_its_occupancy_and_a_factor_verdict_does_not():
    regime = LatentVerdict(1, "regime", 2, 0.9, 0.5, 0.4, CLEARS, (0.5, 0.5))
    factor = LatentVerdict(1, "factor", 1, 0.9, 0.5, 0.4, CLEARS)
    assert "occupancy" in str(regime) and "scale 1 regime" in str(regime)
    assert "occupancy" not in str(factor)


def test_only_a_present_quantity_that_beats_its_floor_clears():
    assert LatentVerdict(1, "regime", 2, 0.9, 0.5, 0.4, CLEARS).clears
    assert not LatentVerdict(1, "regime", 2, 0.5, 0.5, 0.0, FAILS).clears
    assert not LatentVerdict(1, "regime", 1, 1.0, 1.0, 0.0, NOT_APPLICABLE).clears


# ---------------------------------------------------------------------------
# The screen itself
# ---------------------------------------------------------------------------


def test_the_screen_judges_both_latents_at_every_scale():
    got = screen_scenario(_cfg(family=2), n_shifts=4)
    assert [(v.k, v.quantity) for v in got] == [
        (1, "regime"),
        (1, "factor"),
        (2, "regime"),
        (2, "factor"),
    ]
    assert all(sum(v.occupancy) == pytest.approx(1.0) for v in got if v.occupancy)


def test_a_one_state_chain_has_nothing_to_recover_rather_than_failing():
    # A stationary family carries M = 1 and r = 0, so its accuracy and its floor
    # are both exactly one and a strict comparison would print a failure.
    got = screen_scenario(_cfg(family=1), n_shifts=4)
    assert all(v.verdict == NOT_APPLICABLE for v in got)


def test_a_factor_family_is_scored_on_its_factor_and_not_on_a_regime():
    got = screen_scenario(_cfg(family=3, r=1), n_shifts=4)
    regimes = [v for v in got if v.quantity == "regime"]
    factors = [v for v in got if v.quantity == "factor"]
    assert all(v.verdict == NOT_APPLICABLE for v in regimes)
    assert all(v.dim == 1 for v in factors)


def test_the_margin_is_what_decides_a_verdict_rather_than_the_score():
    strict = screen_scenario(_cfg(family=2), n_shifts=4, min_acc_margin=1.0)
    loose = screen_scenario(_cfg(family=2), n_shifts=4, min_acc_margin=-1.0)
    assert all(v.verdict == FAILS for v in strict if v.quantity == "regime")
    assert all(v.verdict == CLEARS for v in loose if v.quantity == "regime")


def test_the_oracle_installs_the_generating_graph_and_nothing_else():
    oracle = oracle_posterior(_cfg(family=2))
    placed = sum(
        len(parents)
        for p in oracle.dataset.params
        for parents in p.parent_sets.values()
    )
    assert len(oracle.summaries) == 2
    assert oracle.graph.n_edges() == placed


# ---------------------------------------------------------------------------
# What the screen decides
# ---------------------------------------------------------------------------


def test_a_refused_scale_drops_to_one_state_and_the_others_are_kept():
    verdicts = [
        LatentVerdict(1, "regime", 2, 0.5, 0.5, 0.0, FAILS),
        LatentVerdict(2, "regime", 2, 0.9, 0.5, 0.4, CLEARS),
    ]
    assert supported_regime_counts(verdicts, [2, 2]) == [1, 2]


def test_a_quantity_that_is_not_applicable_does_not_refuse_its_scale():
    verdicts = [LatentVerdict(1, "regime", 1, 1.0, 1.0, 0.0, NOT_APPLICABLE)]
    assert supported_regime_counts(verdicts, [1]) == [1]


def test_the_report_names_what_cannot_be_recovered():
    text = report([LatentVerdict(1, "regime", 2, 0.5, 0.6, -0.1, FAILS, (0.6, 0.4))])
    assert "regime at scale 1" in text
    assert "scores every block it drives on noise" in text


def test_a_report_with_nothing_to_refuse_makes_no_such_claim():
    text = report([LatentVerdict(1, "regime", 2, 0.9, 0.5, 0.4, CLEARS)])
    assert "unrecoverable" not in text


def test_a_grid_screen_writes_one_row_per_quantity_per_scale():
    grid = ScenarioGrid(scenarios=[_cfg(family=2)])
    lines = screen_grid(grid, n_shifts=4).splitlines()
    assert lines[0].startswith("scenario")
    assert len(lines) == 1 + 4


def test_the_command_screens_the_grid_it_is_asked_for(monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(
        "src.synthetic.latent_screen.screen_grid",
        lambda grid, seed, n_shifts: seen.append(list(grid)) or "table",
    )
    assert main(["--grid", "sanity"]) == 0
    assert main(["--grid", "full"]) == 0
    assert [len(g) for g in seen] == [
        len(ScenarioGrid.sanity_check()),
        len(ScenarioGrid.full_benchmark()),
    ]
    assert capsys.readouterr().out.count("table") == 2


def test_the_command_can_narrow_the_grid_and_raise_the_contrast(monkeypatch):
    seen = []
    monkeypatch.setattr(
        "src.synthetic.latent_screen.screen_grid",
        lambda grid, seed, n_shifts: seen.append(list(grid)) or "table",
    )
    main(["--grid", "sanity", "--family", "2", "--regime-strength", "8"])
    assert seen[0] and all(cfg.family == 2 for cfg in seen[0])
    assert all(cfg.regime_strength == 8.0 for cfg in seen[0])


# ---------------------------------------------------------------------------
# The generated chain, which is what the screen sent back for repair
# ---------------------------------------------------------------------------


def _stationary(P: np.ndarray) -> np.ndarray:
    """The left eigenvector of a transition matrix, normalised to a distribution."""
    vals, vecs = np.linalg.eig(P.T)
    v = np.real(vecs[:, int(np.argmax(np.real(vals)))])
    return v / v.sum()


def test_the_chain_stands_on_the_shares_it_was_built_from():
    for M in (2, 3, 4):
        logits, share = _stationary_chain(M, np.random.default_rng(M))
        P = np.exp(logits.T)  # rows index the previous state
        assert P.sum(axis=1) == pytest.approx(np.ones(M))
        assert _stationary(P) == pytest.approx(share, abs=1e-9)


def test_no_state_is_left_far_below_an_equal_share():
    for seed in range(20):
        _, share = _stationary_chain(3, np.random.default_rng(seed))
        assert share.min() >= (1.0 - REGIME_OCCUPANCY_SPREAD) / 3.0 * 0.9


def test_the_chain_is_persistent_enough_to_have_regimes_at_all():
    logits, _ = _stationary_chain(2, np.random.default_rng(0))
    P = np.exp(logits.T)
    assert np.diag(P).min() >= REGIME_PERSISTENCE[0]


def test_the_transition_carries_the_whole_preference_between_states():
    p = _make_regime_params(3, parent_M=2, d_ctx=2, rng=np.random.default_rng(0))
    assert p.bias == pytest.approx(np.zeros(3))
    assert p.cross.mean(axis=0) == pytest.approx(np.zeros(2))
    assert p.ctx.mean(axis=0) == pytest.approx(np.zeros(2))
    assert np.exp(p.init).sum() == pytest.approx(1.0)


def test_a_coarsest_scale_has_no_parent_to_couple_to():
    p = _make_regime_params(2, parent_M=0, d_ctx=0, rng=np.random.default_rng(0))
    assert p.cross is None
    assert p.ctx.shape == (2, 0)


def test_a_generated_path_occupies_its_states_closely_enough_to_tell_apart():
    # The failure this replaces put the finest-scale path about 88 percent in one
    # state, where naming that state beats any estimator.
    oracle = oracle_posterior(_cfg(family=2))
    for k in (1, 2):
        S = oracle.dataset.latents.get_S(k)
        assert occupancy(S, 2).min() > 0.3


# ---------------------------------------------------------------------------
# The ablation, which the per-scale regime placement broke
# ---------------------------------------------------------------------------


def test_collapsing_narrows_the_cross_block_of_a_scale_that_already_had_one_regime():
    """A finer scale keeps a block shaped for the parent it had before collapse."""
    from copy import deepcopy

    from src.evaluation.baselines import _collapse_regimes
    from src.synthetic.families import make_generator

    cfg = _cfg(family=2)
    cfg.M_k = [1, 2]
    ds = make_generator(cfg, seed=0).generate(seed=0)
    fine = deepcopy(ds.params[0])
    assert fine.M == 1 and fine.regime.cross.shape == (1, 2)

    collapsed = _collapse_regimes(fine)
    assert collapsed.M == 1
    assert collapsed.regime.parent_M == 1
    assert collapsed.regime.cross.shape == (1, 1)


def test_the_no_regime_ablation_runs_where_only_the_coarsest_scale_switches():
    from copy import deepcopy

    from src.core.graph import MultiScaleDBNGraph
    from src.evaluation.baselines import TopDownNoRegime
    from src.synthetic.families import make_generator

    cfg = _cfg(family=2)
    cfg.M_k = [1, 2]
    gen = make_generator(cfg, seed=0)
    ds = gen.generate(seed=0)
    graph = MultiScaleDBNGraph(hierarchy=ds.hierarchy, label_names=gen.label_names)
    params, _, hist = TopDownNoRegime(max_outer_iter=1, verbose=0).fit(
        ds.data, [deepcopy(p) for p in ds.params], graph
    )
    assert all(p.M == 1 for p in params)
    assert np.isfinite(hist.elbo[-1])


# ---------------------------------------------------------------------------
# The calibrated floor, which the caller compares against the penalty itself
# ---------------------------------------------------------------------------


class _Scorer:
    """A scorer whose likelihood is fixed, so only the penalty can move a gain."""

    def __init__(self, lam: float, ll: float) -> None:
        self.lam = lam
        self.ll = ll
        self.feat_x = _Store()
        self.feat_y = _Store()
        self.b_init: dict = {}
        self.a_init: dict = {}

    def score(self, parent_set) -> float:
        return self.ll * len(parent_set) - self.lam * len(parent_set)


class _Store(dict):
    def set(self, key, value):
        self[key] = value


def test_the_null_floor_is_a_raw_gain_and_not_a_penalized_one():
    """The caller compares the floor against lam, so lam must not be inside it."""
    from src.learning.sstep import _null_threshold

    keys = [(0, 1)]
    scorer = _Scorer(lam=5.0, ll=3.0)
    scorer.feat_x.set((0, 1), np.zeros((64, 1)))
    scorer.feat_y.set((0, 1), np.zeros((64, 1)))
    got = _null_threshold(scorer, keys, np.random.default_rng(0), 4, 0.95)
    assert got == pytest.approx(3.0)
    assert scorer.lam == 5.0


def test_a_scale_with_too_few_rows_has_no_floor_to_measure():
    from src.learning.sstep import _null_threshold

    scorer = _Scorer(lam=1.0, ll=3.0)
    scorer.feat_x.set((0, 1), np.zeros((4, 1)))
    scorer.feat_y.set((0, 1), np.zeros((4, 1)))
    assert _null_threshold(scorer, [(0, 1)], np.random.default_rng(0), 4, 0.95) == 0.0
    assert _null_threshold(scorer, [], np.random.default_rng(0), 4, 0.95) == 0.0


def test_the_penalty_can_differ_per_scale_because_the_scales_disagree():
    """No scalar serves two scales whose best prices are four times apart."""
    import numpy as np

    from src.learning.sstep import SStep

    per_scale = SStep(lam={1: 8.0, 2: 2.0, 3: 0.5})
    assert per_scale._lam_effective(14100, 1) == pytest.approx(8.0 * np.log(14100))
    assert per_scale._lam_effective(1410, 2) == pytest.approx(2.0 * np.log(1410))
    assert per_scale._lam_effective(235, 3) == pytest.approx(0.5 * np.log(235))

    as_list = SStep(lam=[8.0, 2.0, 0.5])
    assert as_list._lam_effective(1410, 2) == pytest.approx(2.0 * np.log(1410))

    scalar = SStep(lam=2.0)
    assert scalar._lam_effective(1410, 2) == scalar._lam_effective(1410)


def test_a_per_scale_penalty_survives_the_fixed_mode_too():
    from src.learning.sstep import SStep

    assert SStep(lam=[8.0, 2.0, 0.5], penalty="fixed")._lam_effective(99, 3) == 0.5


# ---------------------------------------------------------------------------
# The coarsest-scale edge strength, which is one value whatever is asked for
# ---------------------------------------------------------------------------


def test_the_coarsest_edge_strength_spans_a_range_and_stays_stable():
    """Clipping to the bound gave every coarse edge one strength; a draw does not."""
    from src.synthetic.families import COARSE_LAG_TOTAL, make_generator

    seen = set()
    for strength in (10.0, 30.0, 100.0):
        cfg = _cfg(family=1)
        cfg.edge_strength = strength
        cfg.max_indegree = [0, 1]
        p = make_generator(cfg, seed=0).generate(seed=0).params[cfg.K - 1]
        for dst, parents in p.parent_sets.items():
            for key in parents:
                seen.add(round(float(p.coarse_count.betaLag[key][dst]), 9))
                seen.add(round(float(p.coarse_dur.alphaLag[key][dst]), 9))
    assert len(seen) > 1
    assert min(seen) >= COARSE_LAG_TOTAL[0] - 1e-9
    assert max(seen) <= COARSE_LAG_TOTAL[1] + 1e-9


def test_the_coarsest_total_is_drawn_inside_the_range_it_is_given():
    """A label with one parent carries the whole total, so the draw is readable."""
    from src.synthetic.families import make_generator

    cfg = _cfg(family=1)
    cfg.max_indegree = [0, 1]
    cfg.coarse_lag_total = (0.5, 0.6)
    p = make_generator(cfg, seed=0).generate(seed=0).params[cfg.K - 1]
    got = [
        float(p.coarse_count.betaLag[key][dst])
        for dst, parents in p.parent_sets.items()
        for key in parents
    ]
    assert got and all(0.5 - 1e-9 <= x <= 0.6 + 1e-9 for x in got)


def test_a_finer_scale_keeps_the_strength_it_was_asked_for():
    """The bound is coarsest-scale only, so the knob still works below it."""
    from src.synthetic.families import MAX_TOTAL_LAG_COARSE, make_generator

    cfg = _cfg(family=1)
    cfg.edge_strength = 10.0
    cfg.max_indegree = [1, 1]
    p = make_generator(cfg, seed=0).generate(seed=0).params[0]
    fine = [
        float(p.fine_count.betaLag[key][dst])
        for dst, parents in p.parent_sets.items()
        for key in parents
    ]
    assert fine and max(fine) > MAX_TOTAL_LAG_COARSE
