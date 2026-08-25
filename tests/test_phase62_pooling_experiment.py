"""
tests/test_phase62_pooling_experiment.py
========================================
The pooling sweep: where a declared rule lands, what a cell claims, what it costs.

Four properties carry this phase.

Placement.  A declared lag is stated in whole days and reaches a scale whose
epoch divides it, so the shipped manifest lands five rules at the day scale and
one at the week scale.  The placement is pinned against exp6's own arithmetic,
since two experiments reporting different recalls for one manifest would be
reporting the placement and not the fit.

The prediction.  The pre-registered rows are recomputed from the ledger they
came from, so a change to the swept curve moves this test before it moves a
result.

The claim.  A cell record cannot be built without saying whether its numbers are
about a population or about one person, and the acceptance crossing separates
the first rung that holds the rule from the first rung every larger one holds it
from.

The run.  Everything above is exercised on a synthetic log of six subjects over
three personas on an hourly grid, and the fit is a parameter, so the sweep is
tested end to end without a fit and once against the real pooled one.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import List

import numpy as np
import pytest

from experiments.exp6_real_log import _place_rules, _rule_name
from experiments.exp11_pooling import (
    KEY_RULE,
    SUBJECT_ARM,
    CellPlan,
    CellRecord,
    CellRequest,
    _read_log,
    build_cohort,
    candidate_space,
    cell_claim,
    crossing,
    epoch_lag,
    estimate_cost,
    ledger_recall,
    main,
    persona_ladder,
    placed_typed_rules,
    plan_cells,
    pooled_graph,
    population_statement,
    preregistered_crossings,
    preregistered_rows,
    scale_recall,
    source_bins,
    stationary_block,
    week_slots,
)
from src.core.calendar import SlotCalendar
from src.core.typed_graph import (
    COUNT,
    DURATION,
    LOG_TERM,
    OBSERVED,
    TypedEdge,
    TypedGraph,
)
from src.data.dynamic_patterns import (
    BENCHMARK_BRANCHING,
    BENCHMARK_ORIGIN,
    read_declared_manifest,
)
from src.learning.fine_sstep import basis_terms, quantile_edges

BENCHMARK = Path("data/dynamic_patterns")

# The synthetic world: three personas of two subjects each on an hourly grid, so
# the ladder is hour, day, week and a two-week cell is 336 slots.
PERSONAS = ("alpha", "beta", "gamma")
SLOT_MINUTES = 60
BRANCHING = "24,7"
DAYS = 21
WEEKS = 2

# Its label axis, and the rules the synthetic manifest declares over it.  The
# unrealized label, the lag no scale reaches and the rule driving a duration are
# all here on purpose, since each one is a branch of the placement.
LABELS = ("gym", "sleep", "walking", "yoga")
UNREALIZED = "ghost"


def _calendar(branching=(24, 7), slot_minutes: int = SLOT_MINUTES) -> SlotCalendar:
    return SlotCalendar.from_config(
        list(branching), slot_minutes=slot_minutes, origin=BENCHMARK_ORIGIN
    )


def _episode(person: str, label: str, day: int, hour: int, minutes: int) -> str:
    start = BENCHMARK_ORIGIN + timedelta(days=day, hours=hour)
    end = start + timedelta(minutes=minutes)
    return f"{person}\t{label}\t{start.isoformat()}\t{end.isoformat()}"


def _write_log(path: Path) -> Path:
    """
    A log of six subjects, one episode per label per day at a fixed hour.

    One episode per label per day is what an hourly slot can hold under the
    identifiable policy, and the sleep episode of the last day runs past the
    two-week cutoff, so the window has something to drop.
    """
    rows = ["person_id\tlabel\tstart_iso\tend_iso"]
    for persona in PERSONAS:
        for i in range(2):
            person = f"{persona}_{i:04d}"
            for day in range(DAYS):
                rows.append(_episode(person, "sleep", day, 23, 480))
                rows.append(_episode(person, "walking", day, 9, 30))
                if day % 2 == 0:
                    rows.append(_episode(person, "yoga", day, 18, 60))
                if persona == "alpha" and day % 7 == 0:
                    rows.append(_episode(person, "gym", day, 17, 90))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _rule(source, target, source_quantity, target_quantity, lag_days, personas=()):
    return {
        "source": source,
        "target": target,
        "source_quantity": source_quantity,
        "target_quantity": target_quantity,
        "lag_days": lag_days,
        "amount": 1.0,
        "direction": "increasing",
        "personas": list(personas),
    }


def _write_manifest(path: Path) -> Path:
    """
    A manifest declaring one rule of each placement kind.

    `sleep|walking|1` is the key and lands at the day scale, `gym|gym|7` lands at
    the week scale, `gym|walking|1` drives a duration the count fit never scores,
    `ghost|gym|1` names a label the log never realized and `walking|gym|3` reads
    a lag no scale of this ladder carries under a limit of two epochs.
    """
    key = _rule("sleep", "walking", "max_duration", "episodes", 1)
    weekly = _rule("gym", "gym", "count", "episodes", 7, ["alpha"])
    duration = _rule("gym", "walking", "count", "duration", 1)
    off_axis = _rule(UNREALIZED, "gym", "count", "episodes", 1)
    off_grid = _rule("walking", "gym", "count", "episodes", 3)
    entries = {
        "sleep": {"stationary": True, "edges_out": [key]},
        "walking": {
            "stationary": False,
            "edges_in": [key, duration],
            "edges_out": [off_grid],
        },
        "gym": {
            "stationary": False,
            "edges_out": [weekly, duration],
            "edges_in": [weekly, off_axis, off_grid],
        },
        "yoga": {"stationary": True},
        UNREALIZED: {"stationary": False, "edges_out": [off_axis]},
    }
    raw = {
        name: {"kind": "routine", "category": "activity", **entry}
        for name, entry in entries.items()
    }
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


@pytest.fixture()
def world(tmp_path: Path):
    """The synthetic log, its manifest and the calendar they are read on."""
    return {
        "csv": str(_write_log(tmp_path / "log.tsv")),
        "manifest": str(_write_manifest(tmp_path / "manifest.json")),
        "out": str(tmp_path / "out"),
    }


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not BENCHMARK.is_dir(), reason="the dynamic_patterns benchmark is not in the tree"
)
def test_the_shipped_manifest_lands_five_day_rules_and_one_week_rule() -> None:
    truth = read_declared_manifest(str(BENCHMARK / "manifest_declared.json"))
    # Every declared label except the one the log never realized.
    labels = sorted(name for name in truth.labels if name != "short_movement_break")
    calendar = _calendar(BENCHMARK_BRANCHING, slot_minutes=1)

    placed, off_axis, off_grid = placed_typed_rules(
        truth, labels, calendar, {3: 2, 4: 2}
    )

    assert sorted(placed) == [3, 4]
    assert [p.name for p in placed[3]] == [
        "meeting|journaling|1d",
        "office_work|sleep|1d",
        "running|yoga|1d",
        "sleep|walking|1d",
        "walking|walking|1d",
    ]
    assert [p.name for p in placed[4]] == ["gym|gym|7d"]
    assert (off_axis, off_grid) == ([], [])
    # The lag is the epoch lag of the scale it was placed at, and the source
    # scale is that scale, which is what `typed_edge` alone does not write.
    assert {(p.edge.src_scale, p.edge.lag) for p in placed[3]} == {(3, 1)}
    assert [(p.edge.src_scale, p.edge.lag) for p in placed[4]] == [(4, 1)]
    assert sum(1 for p in placed[3] if p.universal) == 3
    assert len([name for name in truth.stationary_labels if name in labels]) == 42


@pytest.mark.skipif(
    not BENCHMARK.is_dir(), reason="the dynamic_patterns benchmark is not in the tree"
)
def test_the_placement_agrees_with_exp6_rule_for_rule() -> None:
    truth = read_declared_manifest(str(BENCHMARK / "manifest_declared.json"))
    labels = sorted(name for name in truth.labels if name != "short_movement_break")
    calendar = _calendar(BENCHMARK_BRANCHING, slot_minutes=1)

    mine, _, _ = placed_typed_rules(truth, labels, calendar, {1: 2, 2: 2, 3: 2, 4: 2})
    theirs, _, _ = _place_rules(truth, labels, calendar, [2, 2, 2, 2])

    assert sorted(mine) == sorted(theirs)
    for k in theirs:
        assert [p.name for p in mine[k]] == [
            _rule_name(r) for r, _s, _l, _d in theirs[k]
        ]
        assert [(p.edge.src_label, p.edge.lag, p.dst_label) for p in mine[k]] == [
            (s, lag, d) for _r, s, lag, d in theirs[k]
        ]


def test_a_lag_no_epoch_divides_reaches_no_scale(world) -> None:
    truth = read_declared_manifest(world["manifest"])
    placed, off_axis, off_grid = placed_typed_rules(
        truth, list(LABELS), _calendar(), {2: 2, 3: 2}
    )

    assert [p.name for p in placed[2]] == ["gym|walking|1d", "sleep|walking|1d"]
    assert [p.name for p in placed[3]] == ["gym|gym|7d"]
    assert [r.source for r in off_axis] == [UNREALIZED]
    assert [f"{r.source}|{r.target}|{r.lag_days}d" for r in off_grid] == [
        "walking|gym|3d"
    ]
    assert [p.universal for p in placed[3]] == [False]


def test_an_epoch_that_does_not_divide_the_lag_places_nothing() -> None:
    calendar = _calendar()
    assert epoch_lag(1, calendar, 2) == 1
    assert epoch_lag(1, calendar, 3) is None
    assert epoch_lag(7, calendar, 3) == 1


# ---------------------------------------------------------------------------
# What a graph came back with
# ---------------------------------------------------------------------------


def _placed_here(world):
    truth = read_declared_manifest(world["manifest"])
    placed, _, _ = placed_typed_rules(truth, list(LABELS), _calendar(), {2: 2, 3: 2})
    return truth, placed


def test_a_rule_driving_a_quantity_the_fit_never_scored_is_held_apart(world) -> None:
    _truth, placed = _placed_here(world)
    graph = TypedGraph(label_names=list(LABELS))
    key = next(p for p in placed[2] if p.name == "sleep|walking|1d")
    graph.add_edge(key.edge, key.dst_label, key.dst_quantity)

    block = scale_recall(graph, placed[2], COUNT)

    assert block["off_quantity"] == ["gym|walking|1d"]
    assert block["scorable"] == ["sleep|walking|1d"]
    assert block["found"] == ["sleep|walking|1d"]
    assert block["missed"] == []
    assert block["recall"] == 1.0


def test_a_band_of_the_declared_source_is_the_rule_found(world) -> None:
    """The rule is the relation, so the term the search expressed it in travels along."""
    _truth, placed = _placed_here(world)
    graph = TypedGraph(label_names=list(LABELS))
    key = next(p for p in placed[2] if p.name == "sleep|walking|1d")
    graph.add_edge(replace(key.edge, basis=(390.0, np.inf)), key.dst_label, COUNT)

    block = scale_recall(graph, placed[2], COUNT)

    assert block["found"] == ["sleep|walking|1d"]
    assert block["terms"] == {"sleep|walking|1d": ["[390,inf)"]}
    assert block["recall"] == 1.0


def test_a_scale_with_no_scorable_rule_carries_no_recall(world) -> None:
    _truth, placed = _placed_here(world)
    graph = TypedGraph(label_names=list(LABELS))

    block = scale_recall(
        graph, [p for p in placed[2] if p.dst_quantity == DURATION], COUNT
    )

    assert block["scorable"] == []
    assert block["recall"] is None


def test_the_stationary_block_counts_the_edges_of_its_own_scale(world) -> None:
    truth, _placed = _placed_here(world)
    graph = TypedGraph(label_names=list(LABELS))
    sleep = LABELS.index("sleep")
    graph.add_edge(TypedEdge(OBSERVED, 2, 0, COUNT, 1), sleep, COUNT)
    graph.add_edge(TypedEdge(OBSERVED, 3, 0, COUNT, 1), sleep, COUNT)

    day = stationary_block(graph, truth, list(LABELS), COUNT, 2)
    week = stationary_block(graph, truth, list(LABELS), COUNT, 3)

    # sleep and yoga are declared stationary and both are on the axis.
    assert day["n_stationary"] == week["n_stationary"] == 2
    assert day["edges"] == week["edges"] == 1
    assert day["labels_with_parents"] == ["sleep"]


# ---------------------------------------------------------------------------
# The pre-registered prediction
# ---------------------------------------------------------------------------


def test_the_predicted_curve_reproduces_the_pre_registration() -> None:
    rows = {
        int(row["subjects"]): row["expected_recall"] for row in preregistered_rows()
    }

    assert rows == {1: 0.0, 5: 0.139, 10: 0.265, 20: 0.418, 30: 0.604}


def test_the_pre_registered_crossings_are_the_recorded_ones() -> None:
    crossings = preregistered_crossings()

    assert round(crossings["recall_0.5"], 2) == 24.40
    assert crossings["recall_0.8"] == float("inf")


def test_a_scale_with_no_candidate_predicts_nothing() -> None:
    assert ledger_recall(100, 0, 2) == 0.0


# ---------------------------------------------------------------------------
# The roster, the window and the claim
# ---------------------------------------------------------------------------


def test_the_persona_ladder_is_balanced_and_nested() -> None:
    names = [f"{p}_{i:04d}" for p in PERSONAS for i in range(4)]

    ladder = persona_ladder(names)

    assert ladder[:3] == ["alpha_0000", "beta_0000", "gamma_0000"]
    # Every prefix holds each persona as evenly as its length allows, and a
    # shorter rung is a subset of a longer one.
    assert set(ladder[:3]) < set(ladder[:6]) < set(ladder[:9])
    assert sorted(ladder) == sorted(names)
    for size in (3, 6, 9, 12):
        counts = {p: sum(1 for n in ladder[:size] if n.startswith(p)) for p in PERSONAS}
        assert set(counts.values()) == {size // 3}


def test_a_rung_larger_than_the_roster_is_refused() -> None:
    with pytest.raises(ValueError, match="asks for 4 subjects"):
        plan_cells(["a", "b", "c"], [4], (), 2, 3)


def _full_sweep() -> List[CellPlan]:
    """The two arms of the benchmark, planned off a roster of its shape."""
    ladder = persona_ladder(f"{p}_{i:04d}" for i in range(10) for p in PERSONAS)
    return plan_cells(ladder, [1, 3, 6, 12, 18, 24, 30], [2, 4, 6, 8, 12], 12, 3)


def test_the_two_arms_share_the_cell_they_both_name() -> None:
    plans = _full_sweep()

    assert len(plans) == 13
    assert sum(p.subject_weeks for p in plans) == 1752
    assert sum(1 for p in plans if p.n_subjects == 1) == 3
    assert sum(1 for p in plans if p.arm == "horizon") == 4
    assert sum(1 for p in plans if p.weeks == 12 and p.n_subjects == 30) == 1


def test_the_cost_of_the_full_sweep_is_the_pre_registered_one() -> None:
    cost = estimate_cost(_full_sweep(), iters=8, workers=4, n_targets=60)

    assert cost["n_cells"] == 13
    assert cost["subject_weeks"] == 1752
    assert cost["full_cohort_equivalents"] == 4.87
    assert cost["serial_hours"] == 3.89
    # Eight coefficient rounds, the pass that reads the bin edges off the
    # training split, the null pass the price is read off, the pass that opens
    # the search, two moves per target and the pass that writes what the search
    # accepted.
    assert cost["cohort_passes_per_cell"] == 132


def test_an_empty_sweep_costs_nothing() -> None:
    cost = estimate_cost([], iters=4, workers=4)

    assert cost["n_cells"] == 0
    assert cost["full_cohort_equivalents"] == 0.0
    assert cost["parallel_hours"] == 0.0


def test_a_week_is_a_whole_number_of_slots_or_it_is_refused() -> None:
    assert week_slots(2, SLOT_MINUTES) == 336
    with pytest.raises(ValueError, match="not a whole number of slots"):
        week_slots(1, 11)


def test_a_window_half_covering_a_coarsest_epoch_is_refused(world) -> None:
    by_subject = _read_log(world["csv"])
    with pytest.raises(ValueError, match="not a whole number of coarsest epochs"):
        build_cohort(
            by_subject,
            list(LABELS),
            sorted(by_subject)[:2],
            1,
            SLOT_MINUTES,
            [24, 5],
            _calendar((24, 5)),
        )


def test_a_two_week_cell_holds_two_weeks_and_drops_what_overruns(world) -> None:
    by_subject = _read_log(world["csv"])
    cohort = build_cohort(
        by_subject,
        list(LABELS),
        sorted(by_subject)[:2],
        WEEKS,
        SLOT_MINUTES,
        [24, 7],
        _calendar(),
    )

    assert cohort.S == 2
    assert cohort.n_slots == 336
    data = cohort.open(cohort.subjects[0])
    assert data.observed.all()
    # 14 walking starts, and 13 sleep starts, since the sleep of the last day
    # runs past the cutoff and a clipped episode is dropped whole.
    day = data.get_X(2)
    assert int(day[:, LABELS.index("walking")].sum()) == 14
    assert int(day[:, LABELS.index("sleep")].sum()) == 13


def test_the_claim_says_which_question_a_cell_answers() -> None:
    alone = CellPlan(SUBJECT_ARM, ("alpha_0000",), 12)
    many = CellPlan(SUBJECT_ARM, ("alpha_0000", "beta_0000"), 12)

    assert cell_claim(alone).startswith("n-of-1")
    assert "one person's own history" in cell_claim(alone)
    assert cell_claim(many).startswith("population")
    assert "2 subjects over 2 personas" in cell_claim(many)


def _record(
    n_subjects: int, holds, recall=None, arm: str = SUBJECT_ARM, weeks: int = 12
):
    plan = CellPlan(arm, tuple(f"p{i:02d}" for i in range(n_subjects)), weeks)
    return CellRecord(
        plan=plan,
        claim=cell_claim(plan),
        day_epochs=n_subjects * 7 * weeks,
        measured={3: {"is_day_scale": True, "recall": recall}},
        stationary={},
        predicted={3: 0.0},
        verdict={},
        holds_key=holds,
    )


def test_a_cell_record_cannot_be_built_without_its_claim() -> None:
    plan = CellPlan(SUBJECT_ARM, ("alpha_0000",), 12)
    with pytest.raises(ValueError, match="carries the claim"):
        CellRecord(
            plan=plan,
            claim="  ",
            day_epochs=84,
            measured={},
            stationary={},
            predicted={},
            verdict={},
            holds_key=None,
        )


def test_a_cell_with_no_day_scale_block_reports_no_day_recall() -> None:
    record = _record(3, True)
    record.measured[3]["is_day_scale"] = False

    assert record.day_recall is None
    assert record.to_dict()["measured_recall"] is None


# ---------------------------------------------------------------------------
# The crossing
# ---------------------------------------------------------------------------


def test_the_crossing_separates_the_first_rung_from_the_stable_one() -> None:
    records = [
        _record(1, False),
        _record(1, False),
        _record(1, False),
        _record(6, True, recall=0.2),
        _record(12, False, recall=0.4),
        _record(24, True, recall=0.6),
        _record(30, True, recall=0.8),
        _record(30, True, recall=0.9, arm="horizon", weeks=4),
    ]

    cross = crossing(records, 12)

    assert cross["first_hold"] == 6
    assert cross["stable_hold"] == 24
    assert cross["first_and_stable_differ"] is True
    assert cross["n_of_1_cells"] == 3
    assert cross["n_of_1_holding"] == 0
    assert cross["recall_half"] == 24
    assert cross["rungs"]["1"] == {"cells": 3, "holding": 0}


def test_a_rule_no_rung_holds_crosses_nowhere() -> None:
    cross = crossing([_record(1, False), _record(6, False)], 12)

    assert cross["first_hold"] is None
    assert cross["stable_hold"] is None
    assert cross["first_and_stable_differ"] is False
    assert cross["recall_half"] is None


def test_a_key_the_manifest_never_names_leaves_the_holding_unknown() -> None:
    cross = crossing([_record(1, None), _record(6, None)], 12)

    assert cross["rungs"]["6"] == {"cells": 1, "holding": 0}
    assert cross["first_hold"] is None


def test_the_population_statement_covers_every_reading() -> None:
    nothing = crossing([_record(6, False)], 12)
    assert "says nothing about whether one person" in population_statement(
        nothing, KEY_RULE
    )

    personal = crossing([_record(1, True), _record(6, True)], 12)
    assert "recoverable within one person" in population_statement(personal, KEY_RULE)

    empty = crossing([_record(1, False), _record(6, False)], 12)
    assert "recovered at no rung" in population_statement(empty, KEY_RULE)

    pooled = crossing([_record(1, False), _record(6, True)], 12)
    assert "no single person's history carries it" in population_statement(
        pooled, KEY_RULE
    )


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


class _StubFit:
    """A fit that places the key edge once the roster is large enough."""

    def __init__(self, holds_from: int) -> None:
        self.holds_from = holds_from
        self.seen: List[int] = []

    def __call__(self, request: CellRequest) -> TypedGraph:
        labels = list(request.cohort.label_names)
        graph = TypedGraph(label_names=labels)
        self.seen.append(request.cohort.S)
        if request.cohort.S >= self.holds_from:
            graph.add_edge(
                TypedEdge(OBSERVED, 2, labels.index("sleep"), DURATION, 1),
                labels.index("walking"),
                COUNT,
            )
        return graph


def test_the_sweep_runs_end_to_end_against_a_stub_fit(world) -> None:
    fit = _StubFit(holds_from=2)

    out = main(
        csv=world["csv"],
        manifest=world["manifest"],
        slot_minutes=SLOT_MINUTES,
        branching=BRANCHING,
        weeks=WEEKS,
        subject_arm="1,2",
        horizon_arm="1",
        iters=1,
        output_dir=world["out"],
        fit=fit,
    )

    assert fit.seen == [1, 1, 1, 2, 2]
    assert [c["n_subjects"] for c in out["cells"]] == [1, 1, 1, 2, 2]
    assert out["day_scale"] == 2
    # The bin edges come from each cell's own training split, so a cell holding
    # more subjects reaches more distinct quantiles and offers more terms.
    per_cell = out["n_candidates_per_target"]
    assert set(per_cell) == {c["tag"] for c in out["cells"]}
    assert per_cell["S=1 W=2 alpha_0000"] == 48
    assert per_cell["S=2 W=2"] == 52
    # 32 typed relations, each offering the log term and the bands its own rows
    # cut, so no cell offers fewer candidates than there are relations.
    assert all(n >= out["n_relations_per_target"] for n in per_cell.values())
    assert out["n_candidates_ledger"] == 8

    # The measured recall sits beside the ledger's prediction in every cell, and
    # the claim travels with both.
    for cell in out["cells"]:
        assert cell["claim"]
        assert cell["measured_recall"] is not None
        assert out["day_scale"] in cell["predicted_recall"]
    alone = out["cells"][0]
    pooled = out["cells"][3]
    assert alone["measured_recall"] == 0.0
    assert pooled["measured_recall"] == 1.0
    assert alone["holds_key_rule"] is False
    assert pooled["holds_key_rule"] is True

    assert out["crossing"]["first_hold"] == 2
    assert out["crossing"]["n_of_1_holding"] == 0
    assert "no single person's history carries it" in out["population_vs_n_of_1"]

    stored = json.loads(Path(world["out"], "exp11_results.json").read_text())
    assert stored["population_vs_n_of_1"] == out["population_vs_n_of_1"]
    assert Path(world["out"], "provenance.json").is_file()


def test_a_key_no_declared_rule_names_reports_no_crossing(world) -> None:
    out = main(
        csv=world["csv"],
        manifest=world["manifest"],
        slot_minutes=SLOT_MINUTES,
        branching=BRANCHING,
        weeks=WEEKS,
        subject_arm="2",
        horizon_arm="",
        iters=1,
        key_rule="nobody|nothing|1",
        output_dir=world["out"],
        fit=_StubFit(holds_from=1),
    )

    assert out["cells"][0]["holds_key_rule"] is None
    assert out["crossing"]["first_hold"] is None
    assert "never fitted at one subject" in out["population_vs_n_of_1"]


def test_a_dry_run_plans_the_sweep_and_fits_nothing(world) -> None:
    def refuse(request):  # pragma: no cover - the dry run must never call it
        raise AssertionError("a dry run fitted a cell")

    out = main(
        csv=world["csv"],
        manifest=world["manifest"],
        slot_minutes=SLOT_MINUTES,
        branching=BRANCHING,
        weeks=WEEKS,
        subject_arm="1,6",
        horizon_arm="1",
        dry_run=True,
        output_dir=world["out"],
        fit=refuse,
    )

    assert out["dry_run"] is True
    assert [c["n_subjects"] for c in out["cells"]] == [1, 1, 1, 6, 6]
    assert out["cost"]["subject_weeks"] == 3 * 2 + 12 + 6
    assert all(c["claim"] for c in out["cells"])
    assert not Path(world["out"], "exp11_results.json").exists()


def test_a_ladder_with_no_day_scale_is_refused(world) -> None:
    with pytest.raises(ValueError, match="spans a day"):
        main(
            csv=world["csv"],
            manifest=world["manifest"],
            slot_minutes=SLOT_MINUTES,
            branching="30,24",
            weeks=WEEKS,
            subject_arm="1",
            horizon_arm="",
            dry_run=True,
            output_dir=world["out"],
        )


def test_a_label_axis_too_small_for_the_declared_rules_is_refused(world) -> None:
    with pytest.raises(ValueError, match="declared rules name"):
        main(
            csv=world["csv"],
            manifest=world["manifest"],
            slot_minutes=SLOT_MINUTES,
            branching=BRANCHING,
            weeks=WEEKS,
            subject_arm="1",
            horizon_arm="",
            max_labels=2,
            dry_run=True,
            output_dir=world["out"],
        )


def test_the_label_axis_cut_keeps_every_label_a_rule_names(world) -> None:
    out = main(
        csv=world["csv"],
        manifest=world["manifest"],
        slot_minutes=SLOT_MINUTES,
        branching=BRANCHING,
        weeks=WEEKS,
        subject_arm="1",
        horizon_arm="",
        max_labels=3,
        dry_run=True,
        output_dir=world["out"],
    )

    # yoga is the one label no rule names, so the cut drops it and keeps the
    # three the declared rules read or write.
    assert out["labels"] == ["gym", "sleep", "walking"]


def test_the_candidate_space_offers_each_relation_the_terms_of_its_own_basis(
    world,
) -> None:
    """The bins are cut on the cohort's rows, so no threshold is anticipated."""
    by_subject = _read_log(world["csv"])
    cohort = build_cohort(
        by_subject,
        list(LABELS),
        sorted(by_subject)[:2],
        1,
        SLOT_MINUTES,
        [24, 7],
        _calendar(),
    )
    bins = source_bins(cohort, [2], 4)
    pool = candidate_space(len(LABELS), [2], 1, bins)

    plain = candidate_space(len(LABELS), [2], 1)
    assert [e for e in pool[0] if e.basis == LOG_TERM] == plain[0]
    assert len(pool[0]) > len(plain[0]), "some source of this world has a band"
    for edge in pool[0]:
        source = (edge.src_scale, edge.src_label, edge.src_quantity)
        assert edge.basis in basis_terms(bins[source])
    # Every cut point is a quantile of the column it cuts, and the cohort is the
    # only place those numbers come from.
    day = np.concatenate([cohort.open(s).get_X(2) for s in cohort.subjects], axis=0)
    for src in range(len(LABELS)):
        assert np.array_equal(bins[(2, src, COUNT)], quantile_edges(day[:, src], 4))


def test_the_pooled_fit_returns_a_graph_on_a_tiny_cohort(world) -> None:
    """The default fit is the real pooled one, run at the smallest size it takes."""
    by_subject = _read_log(world["csv"])
    cohort = build_cohort(
        by_subject,
        list(LABELS),
        sorted(by_subject)[:2],
        1,
        SLOT_MINUTES,
        [24, 7],
        _calendar(),
    )
    pool = candidate_space(len(LABELS), [2], 1)

    graph = pooled_graph(
        CellRequest(
            cohort=cohort,
            candidates=pool,
            lag_budget={2: 1},
            quantity=COUNT,
            price=1e9,
            iters=1,
            max_edges=1,
        )
    )

    # The price is far above any gain this world can pay, so the search buys
    # nothing and what comes back is the empty graph over the shared axis.
    assert graph.label_names == list(LABELS)
    assert graph.n_edges() == 0
