"""
tests/test_phase38_calibration.py
=================================
Choosing a generating value from the screens rather than by hand.

Both screens were used to decide which scales the grid should stop placing
structure at, which answers a different question from the one that matters:
which generating value would let a scale carry structure at all. These tests pin
the sweep around them, and pin the one piece of arithmetic that would otherwise
confound it, since widening a block also divides the epochs of every scale above
it.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.synthetic.calibration import (
    ScaleReading,
    best_value,
    hold_coarsest,
    main,
    read_scales,
    report,
    sweep,
    with_lever,
)
from src.synthetic.scenarios import ScenarioConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _cfg(**over) -> ScenarioConfig:
    """A two-scale scenario small enough to screen inside a test."""
    base = dict(
        name="tiny",
        K=2,
        H_finest=360,
        branching=[10],
        L=3,
        M_k=[1, 2],
        r_k=[0, 0],
        max_lag=[1, 1],
        max_indegree=[0, 1],
        zi_level="medium",
        family=2,
        seed=0,
        n_replicates=1,
        edge_strength=10.0,
        regime_strength=3.0,
        finest_rate=0.15,
        slot_minutes=1,
        finest_binary=True,
    )
    base.update(over)
    return ScenarioConfig(**base)


def _reading(k=1, edge=1.0, regime=0.2, factor=float("nan")) -> ScaleReading:
    return ScaleReading(
        k=k,
        n_epochs=100,
        density=0.3,
        edge_margin=edge,
        regime_dim=2,
        regime_margin=regime,
        regime_verdict="clears",
        factor_dim=0,
        factor_margin=factor,
        factor_verdict="not applicable",
    )


# ---------------------------------------------------------------------------
# The lever
# ---------------------------------------------------------------------------


def test_a_lever_replaces_one_generating_value_and_leaves_the_rest():
    got = with_lever(_cfg(), "regime_strength", 8.0)
    assert got.regime_strength == 8.0
    assert got.finest_rate == 0.15 and got.L == 3


def test_a_value_the_generator_does_not_have_is_an_error_rather_than_a_guess():
    with pytest.raises(ValueError, match="no such generating value"):
        with_lever(_cfg(), "not_a_field", 1.0)


# ---------------------------------------------------------------------------
# The confound the sweep has to avoid
# ---------------------------------------------------------------------------


def test_widening_a_block_costs_coarse_epochs_unless_the_series_grows():
    """Otherwise a branching sweep moves two things at once."""
    cfg = _cfg(H_finest=360, branching=[10])
    assert cfg.H_k == [360, 36]

    naive = with_lever(cfg, "branching", [30])
    assert naive.H_k[-1] == 12

    held = hold_coarsest(cfg, [30])
    assert held.H_k[-1] == 36
    assert held.H_finest == 36 * 30


def test_holding_the_coarsest_count_works_through_several_levels():
    cfg = _cfg(
        K=3,
        branching=[10, 6],
        H_finest=1200,
        M_k=[1, 1, 2],
        r_k=[0, 0, 0],
        max_lag=[1, 1, 1],
        max_indegree=[0, 1, 1],
    )
    assert cfg.H_k[-1] == 20
    held = hold_coarsest(cfg, [10, 12])
    assert held.H_k[-1] == 20 and held.H_finest == 20 * 120


# ---------------------------------------------------------------------------
# What a reading says
# ---------------------------------------------------------------------------


def test_both_screens_are_read_at_every_scale():
    got = read_scales(_cfg(), seed=0)
    assert [r.k for r in got] == [1, 2]
    assert all(r.n_epochs > 0 for r in got)
    assert all(0.0 <= r.density <= 1.0 for r in got)


def test_a_scale_is_given_an_edge_to_score_even_where_the_grid_places_none(
    monkeypatch,
):
    """The grid leaves scale 1 empty, and an empty scale reports no true gain."""
    seen = {}
    from src.synthetic.scenarios import ScenarioGrid

    real = ScenarioGrid.identifiability_probe

    def spy(cfg, seed=0):
        seen["indegree"] = list(cfg.max_indegree)
        seen["M_k"] = list(cfg.M_k)
        seen["r_k"] = list(cfg.r_k)
        return real(cfg, seed=seed)

    monkeypatch.setattr(ScenarioGrid, "identifiability_probe", staticmethod(spy))
    read_scales(_cfg(max_indegree=[0, 1], M_k=[1, 2], r_k=[0, 0]), seed=0)
    assert seen["indegree"] == [1, 1]
    assert seen["M_k"] == [2, 2]
    assert seen["r_k"] == [1, 1]


def test_an_edge_clears_only_on_a_positive_margin():
    assert _reading(edge=0.5).edge_clears
    assert not _reading(edge=-0.5).edge_clears
    assert not _reading(edge=float("nan")).edge_clears


def test_a_reading_names_its_scale_and_both_verdicts():
    text = str(_reading(k=2))
    assert "scale 2" in text and "regime" in text and "factor" in text


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------


def test_a_sweep_returns_one_set_of_readings_per_value():
    rows = sweep(_cfg(), "regime_strength", [3.0, 8.0], seed=0)
    assert [v for v, _ in rows] == [3.0, 8.0]
    assert all(len(r) == 2 for _, r in rows)


def test_a_branching_sweep_can_hold_the_coarsest_count_fixed():
    rows = sweep(_cfg(), "branching", [[10], [12]], seed=0, keep_coarsest=True)
    coarse = [r[-1].n_epochs for _, r in rows]
    assert coarse[0] == coarse[1]


def test_a_branching_sweep_without_the_correction_loses_coarse_epochs():
    rows = sweep(_cfg(), "branching", [[10], [30]], seed=0, keep_coarsest=False)
    coarse = [r[-1].n_epochs for _, r in rows]
    assert coarse[1] < coarse[0]


# ---------------------------------------------------------------------------
# Reading a setting off the sweep
# ---------------------------------------------------------------------------


def test_the_best_value_is_the_one_with_the_widest_margin():
    rows = [("a", [_reading(k=1, edge=1.0)]), ("b", [_reading(k=1, edge=4.0)])]
    assert best_value(rows, 1, "edge") == "b"


def test_nothing_clearing_is_an_answer_rather_than_a_pick():
    """The only thing that licenses leaving a scale empty is a sweep that failed."""
    rows = [("a", [_reading(k=1, edge=-1.0)]), ("b", [_reading(k=1, edge=-4.0)])]
    assert best_value(rows, 1, "edge") is None


def test_a_scale_the_sweep_never_reached_is_skipped():
    rows = [("a", [_reading(k=1)])]
    assert best_value(rows, 9, "edge") is None


def test_each_quantity_is_picked_on_its_own_margin():
    rows = [
        ("a", [_reading(k=1, edge=5.0, regime=0.01)]),
        ("b", [_reading(k=1, edge=0.1, regime=0.40)]),
    ]
    assert best_value(rows, 1, "edge") == "a"
    assert best_value(rows, 1, "regime") == "b"
    assert best_value(rows, 1, "factor") is None


def test_the_report_carries_the_epoch_count_beside_every_margin():
    text = report([("[10]", [_reading(k=1)])], "branching")
    assert "branching" in text and "T_k" in text
    assert "edge yes" in text and "regime clears" in text


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------


def test_the_command_sweeps_the_values_it_is_given(monkeypatch, capsys):
    seen = {}

    def fake(cfg, lever, values, seed, keep_coarsest):
        seen.update(lever=lever, values=values, keep=keep_coarsest)
        return [(values[0], [_reading(k=1), _reading(k=2, regime=-0.1)])]

    monkeypatch.setattr("src.synthetic.calibration.sweep", fake)
    assert main(["--lever", "finest_rate", "--values", "0.15;0.5"]) == 0
    assert seen["lever"] == "finest_rate" and seen["values"] == [0.15, 0.5]
    assert seen["keep"] is False
    out = capsys.readouterr().out
    assert "best" in out and "nothing clears" in out


def test_the_command_can_hold_the_coarsest_count(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "src.synthetic.calibration.sweep",
        lambda cfg, lever, values, seed, keep_coarsest: seen.update(keep=keep_coarsest)
        or [(values[0], [_reading()])],
    )
    main(["--values", "[10,6]", "--keep-coarsest"])
    assert seen["keep"] is True


def test_the_screen_places_the_quantity_it_is_asking_about():
    """A scale the grid already emptied cannot answer whether a lever would help."""
    from src.synthetic.calibration import read_scales

    emptied = _cfg(M_k=[1, 2])
    got = read_scales(emptied, seed=0)
    assert [r.regime_dim for r in got] == [2, 2]
    assert all(r.regime_verdict != "not applicable" for r in got)


def test_a_family_that_forbids_a_latent_still_reports_none():
    """The family projection removes what the screen placed, which is correct."""
    from src.synthetic.calibration import read_scales

    stationary = read_scales(_cfg(family=1), seed=0)
    assert all(r.regime_dim == 1 for r in stationary)
    assert all(r.factor_dim == 0 for r in stationary)

    regime_family = read_scales(_cfg(family=2), seed=0)
    assert all(r.factor_dim == 0 for r in regime_family)


def test_every_family_forwards_the_coarse_total_it_was_given():
    """A family rebuilds the config field by field, so a new field can be lost."""
    from src.synthetic.families import make_generator

    for family in (1, 2, 3, 4):
        cfg = _cfg(family=family, K=2, M_k=[2, 2], r_k=[1, 1], max_indegree=[0, 1])
        cfg.coarse_lag_total = (0.5, 0.6)
        p = make_generator(cfg, seed=0).generate(seed=0).params[cfg.K - 1]
        got = [
            float(p.coarse_count.betaLag[key][dst])
            for dst, parents in p.parent_sets.items()
            for key in parents
        ]
        assert got, f"family {family} placed no coarse edge to check"
        assert all(0.5 - 1e-9 <= x <= 0.6 + 1e-9 for x in got), f"family {family}"


# ---------------------------------------------------------------------------
# One draw cannot settle a generating value
# ---------------------------------------------------------------------------


def test_a_value_is_judged_on_how_often_it_clears_not_on_one_draw():
    from src.synthetic.calibration import sweep_grid

    cfgs = [_cfg(seed=0), _cfg(seed=1)]
    rows = sweep_grid(cfgs, "regime_strength", [3.0], seeds=(0, 1))
    assert len(rows) == 1
    rates = {(r.k, r.quantity): r for _, rates in rows for r in rates}
    assert rates[(1, "edge")].total == 4
    assert 0.0 <= rates[(1, "edge")].rate <= 1.0


def test_the_aggregated_report_names_the_share_that_cleared():
    from src.synthetic.calibration import ClearingRate, report_grid

    rows = [("a", [ClearingRate(2, "regime", 4, 10, 0.037)])]
    text = report_grid(rows, "branching")
    assert "4/10" in text and "+0.037" in text


def test_a_clearing_rate_reads_as_a_share_and_prints_its_margin():
    from src.synthetic.calibration import ClearingRate

    got = ClearingRate(2, "regime", 4, 10, 0.037)
    assert got.rate == pytest.approx(0.4)
    assert "4/10" in str(got)
    assert ClearingRate(2, "regime", 0, 0, float("nan")).rate == 0.0


# ---------------------------------------------------------------------------
# Persistence is relative to the block a scale sits inside
# ---------------------------------------------------------------------------


def test_a_regime_below_the_coarsest_completes_several_runs_inside_its_block():
    """A path constant across a block cancels in the allocation softmax."""
    from src.synthetic.families import REGIME_RUNS_PER_BLOCK, _persistence_for

    for width in (6, 10, 24):
        lo, hi = _persistence_for(1, 3, [width, 6])
        mean_run = 1.0 / (1.0 - (lo + hi) / 2.0)
        assert mean_run < width, f"width {width} run {mean_run}"
        assert width / mean_run == pytest.approx(REGIME_RUNS_PER_BLOCK, rel=0.2)


def test_the_coarsest_scale_keeps_the_persistence_that_models_drift():
    from src.synthetic.families import REGIME_PERSISTENCE, _persistence_for

    assert _persistence_for(3, 3, [10, 6]) == REGIME_PERSISTENCE
    assert _persistence_for(1, 3, [10, 6]) != REGIME_PERSISTENCE


def test_a_block_too_narrow_for_the_target_still_yields_a_usable_chain():
    from src.synthetic.families import REGIME_PERSISTENCE_BOUNDS, _persistence_for

    lo, hi = _persistence_for(1, 2, [2])
    assert REGIME_PERSISTENCE_BOUNDS[0] <= lo <= hi <= REGIME_PERSISTENCE_BOUNDS[1]


def test_a_wider_block_affords_a_longer_run():
    from src.synthetic.families import _persistence_for

    narrow = _persistence_for(1, 3, [6, 6])
    wide = _persistence_for(1, 3, [24, 6])
    assert wide[0] > narrow[0] and wide[1] > narrow[1]
