"""
Tests for the declared ceiling and the partial result a run returns at it.

The two properties a killed run needed and did not have are here.  A run that
reaches a limit stops and says which cells it never reached, and a run that is
interrupted resumes at the first cell its ledger does not hold.
"""

from __future__ import annotations

import argparse
import json

import pytest

from src.utils.budget import (
    COMPLETE,
    EXHAUSTED,
    BudgetLedger,
    Checkpoint,
    RunBudget,
    add_budget_arguments,
    budget_from_args,
    peak_rss_gb,
    run_cells,
)

CELLS = [(f"cell{i}", i) for i in range(6)]


def double(value):
    return value * 2


def test_the_high_water_mark_is_a_positive_number_of_gibibytes():
    assert peak_rss_gb() > 0.0


def test_a_missing_kernel_mark_falls_back_to_the_resource_module(tmp_path):
    assert peak_rss_gb(str(tmp_path / "nothing")) > 0.0


def test_a_status_file_without_the_mark_falls_back_too(tmp_path):
    path = tmp_path / "status"
    path.write_text("Name:\tpython\nVmRSS:\t1024 kB\n")
    assert peak_rss_gb(str(path)) > 0.0


def test_a_budget_declares_three_positive_ceilings():
    for bad in ({"seconds": 0.0}, {"peak_rss_gb": -1.0}, {"cells": 0}):
        kwargs = {"seconds": 1.0, "peak_rss_gb": 1.0, "cells": 1}
        kwargs.update(bad)
        with pytest.raises(ValueError, match="is positive"):
            RunBudget(**kwargs)


def test_the_quick_grid_is_smaller_than_the_full_one_on_every_axis():
    quick, full = RunBudget.quick(), RunBudget.full()
    assert quick.seconds < full.seconds
    assert quick.peak_rss_gb < full.peak_rss_gb
    assert quick.cells < full.cells


def test_a_run_inside_its_budget_finishes_every_cell():
    done, record = run_cells(CELLS, double, RunBudget(60.0, 64.0, 100))
    assert done == {f"cell{i}": 2 * i for i in range(6)}
    assert record["status"] == COMPLETE
    assert record["limit"] is None
    assert record["cells_finished"] == 6
    assert record["cells_not_reached"] == []
    assert record["budget"]["cells"] == 100


def test_a_run_stopped_by_its_cell_count_names_what_it_did_not_reach():
    done, record = run_cells(CELLS, double, RunBudget(60.0, 64.0, 2))
    assert sorted(done) == ["cell0", "cell1"]
    assert record["status"] == EXHAUSTED
    assert record["limit"] == "cells"
    assert record["cells_not_reached"] == ["cell2", "cell3", "cell4", "cell5"]


def test_a_run_stopped_by_its_clock_says_so():
    budget = RunBudget(1e-9, 64.0, 100)
    _, record = run_cells(CELLS, double, budget)
    assert record["limit"] == "seconds"
    assert record["cells_finished"] == 0


def test_a_run_stopped_by_its_memory_cap_says_so():
    budget = RunBudget(60.0, 1e-9, 100)
    _, record = run_cells(CELLS, double, budget)
    assert record["limit"] == "peak_rss_gb"


def test_an_interrupted_run_resumes_at_the_first_cell_its_ledger_lacks(tmp_path):
    path = str(tmp_path / "nested" / "ledger.jsonl")
    checkpoint = Checkpoint(path)
    first, _ = run_cells(CELLS, double, RunBudget(60.0, 64.0, 3), checkpoint)
    assert sorted(first) == ["cell0", "cell1", "cell2"]

    seen = []

    def watched(value):
        seen.append(value)
        return value * 2

    second, record = run_cells(CELLS, watched, RunBudget(60.0, 64.0, 100), checkpoint)
    assert seen == [3, 4, 5]
    assert second == {f"cell{i}": 2 * i for i in range(6)}
    assert record["status"] == COMPLETE
    assert record["cells_finished"] == 6


def test_a_ledger_reads_back_what_it_wrote_and_skips_blank_lines(tmp_path):
    path = tmp_path / "ledger.jsonl"
    checkpoint = Checkpoint(str(path))
    assert checkpoint.done() == {}
    checkpoint.append("cell0", {"score": 1.5})
    with open(path, "a") as handle:
        handle.write("\n")
    checkpoint.append("cell1", 7)
    assert checkpoint.done() == {"cell0": {"score": 1.5}, "cell1": 7}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert [row["cell"] for row in rows] == ["cell0", "cell1"]


def test_a_resumed_run_already_at_its_cell_count_reaches_nothing_new(tmp_path):
    checkpoint = Checkpoint(str(tmp_path / "ledger.jsonl"))
    run_cells(CELLS, double, RunBudget(60.0, 64.0, 3), checkpoint)
    _, record = run_cells(CELLS, double, RunBudget(60.0, 64.0, 3), checkpoint)
    assert record["status"] == EXHAUSTED
    assert record["cells_not_reached"] == ["cell3", "cell4", "cell5"]


def test_a_fresh_ledger_reports_no_limit_and_a_running_clock():
    ledger = BudgetLedger(RunBudget(60.0, 64.0, 10))
    assert ledger.exceeded() is None
    assert ledger.elapsed >= 0.0


def test_a_command_line_declares_the_ceilings_and_the_quick_grid():
    parser = argparse.ArgumentParser()
    add_budget_arguments(parser)

    full = budget_from_args(parser.parse_args([]))
    assert full == RunBudget.full()
    assert budget_from_args(parser.parse_args(["--quick"])) == RunBudget.quick()

    stated = budget_from_args(
        parser.parse_args(
            ["--budget-seconds", "30", "--budget-rss-gb", "2", "--budget-cells", "5"]
        )
    )
    assert stated == RunBudget(30.0, 2.0, 5)


def test_a_stated_ceiling_wins_over_the_grid_it_was_stated_beside():
    parser = argparse.ArgumentParser()
    add_budget_arguments(parser)
    mixed = budget_from_args(parser.parse_args(["--quick", "--budget-cells", "3"]))
    assert mixed.cells == 3
    assert mixed.seconds == RunBudget.quick().seconds
    assert mixed.peak_rss_gb == RunBudget.quick().peak_rss_gb


def test_a_namespace_carrying_no_budget_at_all_reads_as_the_full_grid():
    assert budget_from_args(argparse.Namespace()) == RunBudget.full()
