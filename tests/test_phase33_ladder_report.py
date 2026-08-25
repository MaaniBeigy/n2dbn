"""
tests/test_phase33_ladder_report.py
===================================
The per-scale, per-block reading of the parameter-recovery ladder.

A single averaged error says a ladder is broken without saying where, and the
two answers it hides are different problems. If one scale carries the growth,
the sample size at that scale is the suspect; if it is spread, the estimator is.
The report also separates the blocks that enter the averaged error from the ones
scored beside it, and the test below pins that separation to what the metric
actually averages rather than to a comment.
"""

from __future__ import annotations

import json
from copy import deepcopy

import numpy as np
import pytest

from experiments.exp7_parameter_recovery import (
    _IN_MEAN,
    _OUT_OF_MEAN,
    _as_float,
    _basin_or_identifiability,
    _blocks_averaged,
    _cells,
    _clean,
    _collect,
    _lag_keys,
    _row_blocks,
    _stat,
    _summary,
    _trend,
    _verdict_for,
    report_only,
)
from src.evaluation.metrics import ParameterMetrics
from src.synthetic.families import make_generator
from src.synthetic.scenarios import ScenarioConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _row(method: str, family: int, T: int, **over) -> dict:
    """One stored result row, with every scored column present."""
    row = dict(
        scenario=f"s{family}", family=family, T_finest=T, method=method, replicate=0
    )
    for block in _IN_MEAN + _OUT_OF_MEAN:
        for k in (1, 2, 3):
            row[f"{block}_k{k}"] = 0.5
    row["ParamErr_mean"] = 0.5
    row.update(over)
    return row


@pytest.fixture
def rows():
    """A two-method, two-family ladder whose mean falls for one method only."""
    out = []
    for T, a, b in ((100, 0.9, 0.3), (200, 0.6, 0.4), (400, 0.3, 0.5)):
        out.append(_row("n2dbn", 1, T, ParamErr_mean=a))
        out.append(_row("n2dbn", 2, T, ParamErr_mean=a))
        out.append(_row("n2dbn_data_init", 1, T, ParamErr_mean=b))
        out.append(_row("n2dbn_data_init", 2, T, ParamErr_mean=b))
    return out


# ---------------------------------------------------------------------------
# The trend verdict
# ---------------------------------------------------------------------------


def test_a_falling_ladder_is_the_only_one_that_passes():
    assert _trend([0.9, 0.6, 0.3]) == "falls"


def test_a_rising_ladder_is_named_rather_than_lumped_with_noise():
    assert _trend([0.3, 0.6, 0.9]) == "rises"


def test_a_flat_ladder_is_not_reported_as_non_monotone():
    # A block pinned to one value at every rung is a finding of its own, and
    # calling it non-monotone would bury it among the noisy rows.
    assert _trend([0.3, 0.3, 0.3]) == "flat"


def test_a_ladder_that_turns_is_non_monotone():
    assert _trend([0.9, 1.2, 0.3]) == "NOT monotone"


def test_a_ladder_with_one_readable_rung_has_no_verdict():
    assert _trend([float("nan"), 0.3, float("nan")]) == "n/a"


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def test_unidentified_rungs_are_dropped_rather_than_scored_zero():
    assert _clean([0.2, float("nan"), None, 0.4]) == [0.2, 0.4]


def test_an_empty_group_reads_as_unidentified_rather_than_zero():
    assert np.isnan(_stat({}, ("n2dbn", 100), np.mean))


def test_a_cell_with_no_value_is_printed_rather_than_skipped():
    assert "nan" in _cells([float("nan")])
    assert _cells([0.25]).strip() == "0.2500"


def test_grouping_splits_by_method_and_by_family(rows):
    by_size, by_family = _collect(rows, "ParamErr_mean")
    assert by_size[("n2dbn", 100)] == [0.9, 0.9]
    assert by_family[("n2dbn_data_init", 2, 400)] == [0.5]


def test_grouping_skips_a_rung_the_metric_could_not_score(rows):
    rows[0]["ParamErr_mean"] = float("nan")
    by_size, _ = _collect(rows, "ParamErr_mean")
    assert by_size[("n2dbn", 100)] == [0.9]


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def test_the_report_names_the_method_whose_ladder_turns(rows, capsys):
    _summary(rows)
    out = capsys.readouterr().out
    assert "n2dbn: mean falls" in out
    assert "n2dbn_data_init: mean rises" in out


def test_the_report_gives_every_scale_and_block_its_own_row(rows, capsys):
    _summary(rows)
    out = capsys.readouterr().out
    for block in _IN_MEAN + _OUT_OF_MEAN:
        assert block in out
    assert "held out of ParamErr_mean" in out


def test_the_report_shows_the_median_beside_the_mean(rows, capsys):
    # A mean and a median that disagree mean the ladder is being carried by a
    # tail rather than by the typical fit, which changes the diagnosis.
    _summary(rows)
    out = capsys.readouterr().out
    assert "mean" in out and "median" in out


def test_a_block_that_never_scores_is_left_out_of_the_table(rows, capsys):
    for row in rows:
        row["AllocShare_k3"] = float("nan")
    _summary(rows)
    assert "AllocShare       3" not in capsys.readouterr().out


def _block_n(out: str, block: str, scale: int, n_rungs: int) -> int:
    """Read the fit count the block table prints for one row."""
    for line in out.splitlines():
        parts = line.split()
        if len(parts) > n_rungs + 2 and parts[0] == block and parts[1] == str(scale):
            return int(parts[2 + n_rungs])
    raise AssertionError(f"no {block} row at scale {scale}")


def test_the_block_table_states_how_many_fits_each_cell_averages(rows, capsys):
    # A block only one family scores averages fewer fits than one every family
    # scores, and a level read without that count reads as the same study.
    for row in rows:
        if row["family"] != 1:
            row["TransTV_k1"] = float("nan")
    _summary(rows)
    out = capsys.readouterr().out
    assert _block_n(out, "TransTV", 1, 3) < _block_n(out, "EmisRel", 1, 3)


def test_the_family_table_states_how_many_errors_the_mean_averages(rows, capsys):
    _summary(rows)
    out = capsys.readouterr().out
    assert "blocks" in out
    assert "never across rows" in out


def test_the_block_count_is_read_from_the_row_when_it_was_stored():
    assert _row_blocks({"ParamErr_n": 7.0}) == 7


def test_the_block_count_is_recounted_for_a_run_stored_before_it_existed():
    row = {f"{b}_k{k}": 0.5 for b in _IN_MEAN for k in (1, 2, 3)}
    row["AllocShare_k3"] = float("nan")
    assert _row_blocks(row) == 11


def test_a_family_whose_fits_disagree_on_the_count_is_flagged_rather_than_averaged():
    rows = [
        dict(family=1, ParamErr_n=5.0),
        dict(family=1, ParamErr_n=8.0),
        dict(family=2, ParamErr_n=8.0),
    ]
    assert _blocks_averaged(rows, 1) == "varies"
    assert _blocks_averaged(rows, 2) == "8"
    assert _blocks_averaged(rows, 3) == "?"


def test_a_stored_run_is_re_read_without_refitting(rows, tmp_path, capsys):
    (tmp_path / "exp7_results.json").write_text(json.dumps(rows))
    report_only(str(tmp_path))
    assert f"{len(rows)} stored rows" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Basin against identifiability
# ---------------------------------------------------------------------------


def test_a_worse_objective_at_a_worse_error_is_a_local_optimum():
    assert _verdict_for(100.0, 90.0, 0.2, 0.5) == "local optimum"


def test_an_equal_objective_at_a_worse_error_is_an_unidentified_parameter():
    # The two starts agree on the likelihood and disagree on the parameters, so
    # no better start can fix it and only a constraint or a prior can.
    assert _verdict_for(100.0, 100.0, 0.2, 0.5) == "not identified"


def test_an_objective_within_tolerance_counts_as_the_same_optimum():
    assert _verdict_for(1000.0, 999.9, 0.2, 0.5) == "not identified"


def test_a_data_start_that_matches_on_both_is_not_reported_as_a_problem():
    assert _verdict_for(100.0, 100.0, 0.5, 0.2) == "data start is at least as good"


def test_a_worse_objective_at_a_better_error_is_named_rather_than_lumped_in():
    assert _verdict_for(100.0, 90.0, 0.5, 0.2) == "worse optimum, better params"


def test_a_pair_with_no_parameter_error_is_not_given_a_verdict():
    assert _verdict_for(100.0, 90.0, float("nan"), 0.5) == "no error"


def test_a_missing_objective_is_read_as_absent_rather_than_zero():
    assert np.isnan(_as_float(None))
    assert np.isnan(_as_float(""))
    assert _as_float("2.5") == 2.5


def test_a_run_without_objectives_says_so_instead_of_reporting_nothing(rows, capsys):
    _basin_or_identifiability(rows)
    assert "cannot be read" in capsys.readouterr().out


def test_the_basin_table_pairs_the_two_starts_on_the_same_fit(capsys):
    paired = [
        _row("n2dbn", 1, 100, replicate=0, Objective=500.0, ParamErr_mean=0.2),
        _row(
            "n2dbn_data_init", 1, 100, replicate=0, Objective=400.0, ParamErr_mean=0.5
        ),
    ]
    _basin_or_identifiability(paired)
    out = capsys.readouterr().out
    assert "local optimum" in out
    assert "1 local optimum" in out


def test_a_start_with_no_partner_is_left_out_of_the_pairing(capsys):
    lone = [_row("n2dbn", 1, 100, replicate=0, Objective=500.0, ParamErr_mean=0.2)]
    _basin_or_identifiability(lone)
    assert "cannot be read" in capsys.readouterr().out


def test_the_learned_lagged_keys_are_counted_at_the_scale_they_sit_on():
    params = _small_params()
    counts = [_lag_keys(params, k) for k in (1, 2)]
    assert all(isinstance(c, int) and c >= 0 for c in counts)
    assert sum(counts) > 0, "this scenario carries lagged edges, so some key exists"


# ---------------------------------------------------------------------------
# The separation the report claims
# ---------------------------------------------------------------------------


def _small_params():
    cfg = ScenarioConfig(
        name="ladder",
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
    return make_generator(cfg, seed=cfg.seed).generate(seed=cfg.seed).params


def test_a_lag_coefficient_moves_lagerr_and_leaves_the_averaged_error_alone():
    # The report prints the lag blocks under a "held out" heading. That claim is
    # only true while param_err_mean averages the scale-free blocks alone, so it
    # is asserted against the metric rather than trusted.
    truth = _small_params()
    learned = deepcopy(truth)
    before = ParameterMetrics.compute(truth, learned).to_dict()
    for block in (learned[0].fine_count.betaLag, learned[0].fine_dur.alphaLag):
        for key in block:
            block[key] = block[key] + 5.0
    after = ParameterMetrics.compute(truth, learned).to_dict()

    assert after["LagErr_k1"] > before["LagErr_k1"] or after["LagFalse_k1"] > 0.0
    assert after["ParamErr_mean"] == pytest.approx(before["ParamErr_mean"])


def test_the_report_splits_the_blocks_the_metric_averages_from_the_rest():
    truth = _small_params()
    scored = set(ParameterMetrics.compute(truth, deepcopy(truth)).to_dict())
    for block in _IN_MEAN + _OUT_OF_MEAN:
        assert any(name.startswith(f"{block}_k") for name in scored)
    assert not set(_IN_MEAN) & set(_OUT_OF_MEAN)
