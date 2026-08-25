"""
tests/test_phase54_exp6_structural.py
=====================================
The structural section of experiment 6, and the bugs beside it.

A declared manifest states a carry-over rule as a lag in whole days, and the
model holds a lag in epochs of one scale.  The arithmetic between the two is
what decides which scales are scored, so it is pinned first: a scale carries a
declared lag when its epoch divides that many days and the search limit admits
the quotient.  Everything else in the section follows from the placement, so
the report is checked against a graph whose edges are known.

The truth graph built from the placed rules holds those rules alone, and a
manifest declares a label non-stationary without naming what drives it, so an
edge into such a label is one the manifest makes no claim about.  What the
report may carry is therefore pinned as well: the recall of the placed rules,
and the count of edges into declared-stationary labels under its own
denominator, with precision, F1 and the structural Hamming distance absent.

The bugs are pinned at the end.  The held-out evidence used to be scored under
literal settings, so a fit under any other pair reported a number belonging to
a model nobody fitted.  The stability loop used to cut the same block of the
training split on every pass, so the frequencies counted restarts.  A run whose
subsamples all failed used to write an empty frequency map, which reads as a
fit that selected no edge.  The dead `--jobs` argument was passed by the
compose service and used nowhere.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.exp6_real_log import (
    _epoch_lag,
    _fingerprint,
    _finite,
    _place_rules,
    _read_log,
    _rule_name,
    _stability,
    _structure_report,
    _subsample,
    main,
)
from src.core.calendar import SlotCalendar
from src.core.graph import MultiScaleDBNGraph
from src.core.scales import ScaleHierarchy
from src.data.dynamic_patterns import CarryOverRule, DeclaredLabel, DeclaredTruth
from src.data.event_log import build_multiscale_data, read_event_log

ROOT = Path(__file__).resolve().parents[1]

# Minute, hour, day, week.
LADDER = (60, 24, 7)


def _calendar(branching=LADDER, slot_minutes=1):
    return SlotCalendar.from_config(branching, slot_minutes)


def _rule(source, target, lag_days=1, personas=()):
    return CarryOverRule(
        source=source,
        target=target,
        source_quantity="count",
        target_quantity="episodes",
        lag_days=lag_days,
        amount=1.0,
        direction="increasing",
        personas=tuple(personas),
    )


def _truth(rules, stationary=(), extra_labels=()):
    """A declared truth over the labels the rules name plus the ones given."""
    names = set(stationary) | set(extra_labels)
    for rule in rules:
        names |= {rule.source, rule.target}
    labels = {
        name: DeclaredLabel(
            name=name,
            kind="event",
            category="test",
            stationary=name in stationary,
        )
        for name in sorted(names)
    }
    return DeclaredTruth(labels=labels, rules=tuple(rules))


def _graph(labels, branching=LADDER, n_finest=10080):
    hierarchy = ScaleHierarchy.from_uniform_branching(n_finest, list(branching))
    return MultiScaleDBNGraph(hierarchy=hierarchy, label_names=list(labels))


# ---------------------------------------------------------------------------
# A lag in days as a lag in epochs
# ---------------------------------------------------------------------------


def test_a_day_is_as_many_epochs_as_the_scale_fits_into_it() -> None:
    cal = _calendar()
    assert [_epoch_lag(1, cal, k) for k in (1, 2, 3)] == [1440, 24, 1]


def test_a_week_epoch_cannot_denote_a_lag_of_one_day() -> None:
    assert _epoch_lag(1, _calendar(), 4) is None


def test_a_lag_of_seven_days_is_one_week_epoch_and_seven_day_epochs() -> None:
    cal = _calendar()
    assert _epoch_lag(7, cal, 3) == 7
    assert _epoch_lag(7, cal, 4) == 1


def test_an_epoch_that_does_not_divide_a_day_denotes_no_day_lag() -> None:
    # Seven-minute epochs: 1440 minutes is not a whole number of them.
    assert _epoch_lag(1, _calendar(branching=(7,)), 2) is None


# ---------------------------------------------------------------------------
# Placing the declared rules
# ---------------------------------------------------------------------------


def test_a_rule_lands_only_where_the_search_limit_admits_its_lag() -> None:
    truth = _truth([_rule("a", "b")])
    placed, off_axis, off_grid = _place_rules(
        truth, ["a", "b"], _calendar(), [2, 2, 2, 2]
    )
    assert sorted(placed) == [3]
    assert placed[3] == [(truth.rules[0], 0, 1, 1)]
    assert off_axis == [] and off_grid == []


def test_a_wider_search_limit_places_one_rule_at_two_scales() -> None:
    truth = _truth([_rule("a", "b")])
    placed, _, _ = _place_rules(truth, ["a", "b"], _calendar(), [2, 24, 2, 2])
    assert sorted(placed) == [2, 3]
    assert placed[2][0][2] == 24
    assert placed[3][0][2] == 1


def test_a_weekly_rule_lands_on_the_week_scale() -> None:
    truth = _truth([_rule("gym", "gym", lag_days=7)])
    placed, _, off_grid = _place_rules(truth, ["gym"], _calendar(), [1, 1, 1, 1])
    assert sorted(placed) == [4]
    assert placed[4] == [(truth.rules[0], 0, 1, 0)]
    assert off_grid == []


def test_a_rule_naming_a_label_the_log_never_realized_is_off_the_axis() -> None:
    truth = _truth([_rule("running", "yoga")])
    placed, off_axis, off_grid = _place_rules(
        truth, ["running"], _calendar(), [2, 2, 2, 2]
    )
    assert placed == {}
    assert [_rule_name(r) for r in off_axis] == ["running|yoga|1d"]
    assert off_grid == []


def test_a_rule_no_scale_of_the_ladder_reaches_is_off_the_grid() -> None:
    truth = _truth([_rule("a", "b")])
    placed, off_axis, off_grid = _place_rules(
        truth, ["a", "b"], _calendar(branching=(7,)), [2, 2]
    )
    assert placed == {} and off_axis == []
    assert [_rule_name(r) for r in off_grid] == ["a|b|1d"]


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def test_the_report_names_the_rule_the_fit_found_and_the_one_it_missed() -> None:
    labels = ["a", "b", "c"]
    truth = _truth([_rule("a", "b"), _rule("b", "c")])
    graph = _graph(labels)
    graph.add_edge(3, 0, 1, 1)
    report = _structure_report(graph, truth, labels, _calendar(), [2, 2, 2, 2])

    assert report["scored_scales"] == [3]
    assert report["n_declared"] == 2
    assert report["recall"]["3"] == {
        "placed": ["a|b|1d", "b|c|1d"],
        "found": ["a|b|1d"],
        "missed": ["b|c|1d"],
    }
    assert report["metrics"]["Rec_k3"] == 0.5


def test_an_edge_the_manifest_makes_no_claim_about_is_left_unscored() -> None:
    labels = ["a", "b", "c"]
    truth = _truth([_rule("a", "b")])
    graph = _graph(labels)
    graph.add_edge(3, 0, 1, 1)
    graph.add_edge(3, 2, 2, 1)
    report = _structure_report(graph, truth, labels, _calendar(), [2, 2, 2, 2])

    # `c` is declared neither stationary nor the source of a rule, so the
    # lag-two edge it carries is outside both denominators.
    assert report["metrics"]["Rec_k3"] == 1.0
    assert report["stationary"]["3"] == {"edges": 0, "labels_with_parents": []}


def test_the_report_carries_no_metric_the_partial_truth_cannot_support() -> None:
    labels = ["a", "b", "c"]
    truth = _truth([_rule("a", "b")])
    graph = _graph(labels)
    graph.add_edge(3, 2, 2, 1)
    report = _structure_report(graph, truth, labels, _calendar(), [2, 2, 2, 2])

    assert set(report["metrics"]) == {"Rec_k3", "SelfRec_k3", "CrossRec_k3"}


def test_every_edge_into_a_declared_stationary_label_is_counted() -> None:
    labels = ["a", "b", "sleep_in", "reading"]
    truth = _truth([_rule("a", "b")], stationary=("sleep_in", "reading"))
    graph = _graph(labels)
    graph.add_edge(3, 0, 1, 1)
    graph.add_edge(3, 0, 1, 2)
    graph.add_edge(3, 1, 2, 2)
    report = _structure_report(graph, truth, labels, _calendar(), [2, 2, 2, 2])

    assert report["n_stationary"] == 2
    assert report["stationary"]["3"] == {
        "edges": 2,
        "labels_with_parents": ["sleep_in"],
    }


def test_a_stationary_label_the_log_never_realized_is_not_a_denominator() -> None:
    labels = ["a", "b"]
    truth = _truth([_rule("a", "b")], stationary=("naps",))
    report = _structure_report(_graph(labels), truth, labels, _calendar(), [2] * 4)

    assert report["n_stationary"] == 0
    assert report["stationary"]["3"] == {"edges": 0, "labels_with_parents": []}


def test_a_scale_with_no_cross_label_rule_reports_no_cross_recall() -> None:
    labels = ["gym"]
    truth = _truth([_rule("gym", "gym", lag_days=7)])
    graph = _graph(labels)
    graph.add_edge(4, 0, 1, 0)
    report = _structure_report(graph, truth, labels, _calendar(), [1, 1, 1, 1])

    assert report["metrics"]["SelfRec_k4"] == 1.0
    assert report["metrics"]["CrossRec_k4"] is None
    assert json.dumps(report["metrics"])


def test_a_manifest_no_scale_can_carry_scores_no_scale() -> None:
    labels = ["a", "b"]
    truth = _truth([_rule("a", "b")], stationary=("a",))
    report = _structure_report(
        _graph(labels, branching=(7,), n_finest=70),
        truth,
        labels,
        _calendar(branching=(7,)),
        [2, 2],
    )

    assert report["scored_scales"] == []
    assert report["recall"] == {} and report["stationary"] == {}
    assert report["metrics"] == {}
    assert report["off_grid"] == ["a|b|1d"]


def test_a_metric_a_scale_cannot_define_is_written_as_null() -> None:
    assert _finite(float("nan")) is None
    assert _finite(2.0) == 2.0


# ---------------------------------------------------------------------------
# Reading a log that names its subjects
# ---------------------------------------------------------------------------


def test_a_log_naming_subjects_is_read_pooled_over_them(tmp_path) -> None:
    path = tmp_path / "log.tsv"
    path.write_text(
        "person_id\tlabel\tstart_iso\tend_iso\n"
        "p_0\ta\t2026-02-16T08:00:00\t2026-02-16T08:30:00\n"
        "p_1\ta\t2026-02-16T08:00:00\t2026-02-16T08:30:00\n"
        "p_1\tb\t2026-02-17T09:00:00\t2026-02-17T09:15:00\n",
        encoding="utf-8",
    )
    events, n_subjects = _read_log(str(path))
    assert n_subjects == 2
    assert [e.label for e in events] == ["a", "a", "b"]


def test_a_log_without_a_subject_column_holds_one_subject(tmp_path) -> None:
    path = tmp_path / "log.csv"
    path.write_text(
        "label,start_iso,end_iso\n" "a,2026-02-16T08:00:00,2026-02-16T08:30:00\n",
        encoding="utf-8",
    )
    events, n_subjects = _read_log(str(path))
    assert n_subjects == 1
    assert [e.label for e in events] == ["a"]


# ---------------------------------------------------------------------------
# The bugs
# ---------------------------------------------------------------------------


def test_a_run_with_no_fitted_subsample_carries_no_frequency_map() -> None:
    assert _stability({"1": {}}, 0) is None
    assert json.dumps({"stability": _stability({"1": {}}, 0)}) == '{"stability": null}'


def test_a_frequency_is_the_share_of_the_subsamples_that_selected_the_edge() -> None:
    counts = {
        "1": collections.Counter({"0,1,2": 2, "1,1,0": 1}),
        "2": collections.Counter(),
    }
    assert _stability(counts, 2) == {"1": {"0,1,2": 1.0, "1,1,0": 0.5}, "2": {}}


def test_the_dead_jobs_argument_is_gone_from_the_script_and_the_service() -> None:
    script = (ROOT / "experiments" / "exp6_real_log.py").read_text(encoding="utf-8")
    assert "jobs" not in script
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    exp6 = compose.split("  exp6:", 1)[1].split("\n  exp7:", 1)[0]
    assert "exp6_real_log" in exp6
    assert "--jobs" not in exp6


# ---------------------------------------------------------------------------
# End to end on a log small enough to fit twice
# ---------------------------------------------------------------------------

DAYS = 12
LABELS = ["a", "b", "c"]


def _small_log(path: Path) -> None:
    """A twelve-day hourly log where `a` on one day is followed by `b` on the next."""
    rows = ["label,start_iso,end_iso"]
    for day in range(DAYS):
        for hour, label in ((7, "a"), (12, "c")):
            if label == "a" and day % 2:
                continue
            rows.append(
                f"{label},2026-03-{day + 1:02d}T{hour:02d}:00:00,"
                f"2026-03-{day + 1:02d}T{hour:02d}:30:00"
            )
        if day % 2:
            rows.append(
                f"b,2026-03-{day + 1:02d}T18:00:00,2026-03-{day + 1:02d}T18:45:00"
            )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _small_manifest(path: Path) -> None:
    """Two rules, one of them naming a label the log never realized."""
    rule_b = {
        "source": "a",
        "target": "b",
        "source_quantity": "count",
        "target_quantity": "episodes",
        "lag_days": 1,
        "amount": 1.0,
        "direction": "increasing",
        "source_threshold": None,
        "personas": None,
        "unit": "count",
    }
    rule_e = dict(rule_b, target="e")
    entry = dict(kind="event", category="test", patterns=[], personas=None)
    path.write_text(
        json.dumps(
            {
                "a": dict(entry, stationary=False, edges_out=[rule_b, rule_e]),
                "b": dict(entry, stationary=False, edges_in=[rule_b]),
                "c": dict(entry, stationary=True),
                "d": dict(entry, stationary=True),
                "e": dict(entry, stationary=False, edges_in=[rule_e]),
            }
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# The windows the stability loop refits on
# ---------------------------------------------------------------------------


def _varied_log(path: Path) -> None:
    """Twelve days carrying one episode each, at a different hour on every day."""
    rows = ["label,start_iso,end_iso"]
    for day in range(DAYS):
        rows.append(
            f"a,2026-03-{day + 1:02d}T{day + 4:02d}:00:00,"
            f"2026-03-{day + 1:02d}T{day + 4:02d}:30:00"
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _training_split(path: Path):
    """The training split of a log, laid on an hour, day ladder."""
    data, _cal, _labels = build_multiscale_data(
        read_event_log(str(path)),
        slot_minutes=60,
        branching=[24],
        finest_binary=False,
    )
    train, _valid, _test = data.split(fracs=(0.7, 0.15, 0.15))
    return train


def _starts_at(train, window, start: int) -> bool:
    """Whether a window is the block of the training split beginning at `start`."""
    part = train.get_X(1)[start : start + window.get_X(1).shape[0]]
    return np.array_equal(part, window.get_X(1))


def test_every_window_of_the_rolling_origin_carries_different_epochs(tmp_path) -> None:
    _varied_log(tmp_path / "log.csv")
    train = _training_split(tmp_path / "log.csv")
    windows = [_subsample(train, i, 4) for i in range(4)]

    assert len({_fingerprint(w) for w in windows}) == 4


def test_each_window_is_a_contiguous_block_starting_later_than_the_last(
    tmp_path,
) -> None:
    _varied_log(tmp_path / "log.csv")
    train = _training_split(tmp_path / "log.csv")
    windows = [_subsample(train, i, 4) for i in range(4)]

    # A day is twenty four finest epochs, and the origin rolls one day per pass
    # over a training split of eight days, five of which a window covers.  The
    # closing window loses the day the three-way cut keeps back.
    assert all(_starts_at(train, w, 24 * i) for i, w in enumerate(windows))
    assert [w.get_X(1).shape[0] for w in windows] == [120, 120, 120, 96]


def test_a_single_window_is_the_opening_block_of_the_training_split(tmp_path) -> None:
    _varied_log(tmp_path / "log.csv")
    train = _training_split(tmp_path / "log.csv")
    window = _subsample(train, 0, 1)

    assert _starts_at(train, window, 0)
    assert _fingerprint(window) == _fingerprint(_subsample(train, 0, 4))


@pytest.fixture(scope="module")
def scored_run(tmp_path_factory):
    """One fit of the small log against the small manifest."""
    work = tmp_path_factory.mktemp("exp6")
    _small_log(work / "log.csv")
    _small_manifest(work / "manifest.json")
    main(
        csv=str(work / "log.csv"),
        manifest=str(work / "manifest.json"),
        slot_minutes=60,
        branching="24",
        regimes=2,
        max_lag=1,
        max_indegree=1,
        iters=1,
        subsamples=1,
        output_dir=str(work / "out"),
    )
    return json.loads((work / "out" / "exp6_results.json").read_text())


def test_the_run_scores_the_day_scale_alone(scored_run) -> None:
    structure = scored_run["structure"]
    assert scored_run["labels"] == LABELS
    assert structure["scored_scales"] == [2]
    assert structure["n_declared"] == 2
    assert structure["off_axis"] == ["a|e|1d"]
    assert structure["off_grid"] == []
    assert structure["recall"]["2"]["placed"] == ["a|b|1d"]
    assert structure["n_stationary"] == 1


def test_the_report_agrees_with_the_edges_the_run_recorded(scored_run) -> None:
    learned = [tuple(e) for e in scored_run["learned_edges"]["2"]]
    structure = scored_run["structure"]
    found = (0, 1, 1) in learned

    assert structure["recall"]["2"]["found"] == (["a|b|1d"] if found else [])
    assert structure["recall"]["2"]["missed"] == ([] if found else ["a|b|1d"])
    assert structure["stationary"]["2"] == {
        "edges": sum(1 for e in learned if e[2] == 2),
        "labels_with_parents": ["c"] if any(e[2] == 2 for e in learned) else [],
    }
    tp = 1 if found else 0
    assert structure["metrics"]["Rec_k2"] == float(tp)
    assert structure["metrics"]["CrossRec_k2"] == float(tp)
    assert "Prec_k2" not in structure["metrics"]
    assert "SHD_k2" not in structure["metrics"]


def test_the_run_reports_a_stability_frequency_for_its_one_window(
    scored_run,
) -> None:
    assert scored_run["n_windows_fitted"] == 1
    assert scored_run["n_distinct_windows"] == 1
    assert scored_run["stability_scheme"] == "rolling origin over the training split"
    assert scored_run["stability_failures"] == []
    assert set(scored_run["stability"]) == {"1", "2"}
    assert all(f == 1.0 for f in scored_run["stability"]["2"].values())


def test_a_run_whose_subsamples_all_fail_writes_null_and_the_reasons(
    tmp_path, monkeypatch
) -> None:
    import experiments.exp6_real_log as exp6

    _small_log(tmp_path / "log.csv")
    fits = {"n": 0}
    fit = exp6._fit

    def failing(*args, **kwargs):
        fits["n"] += 1
        if fits["n"] == 1:
            return fit(*args, **kwargs)
        raise RuntimeError("the subsample carries no episode of any label")

    monkeypatch.setattr(exp6, "_fit", failing)
    exp6.main(
        csv=str(tmp_path / "log.csv"),
        slot_minutes=60,
        branching="24",
        regimes=2,
        max_lag=1,
        max_indegree=1,
        iters=1,
        subsamples=2,
        output_dir=str(tmp_path / "out"),
    )
    out = json.loads((tmp_path / "out" / "exp6_results.json").read_text())

    assert out["structure"] is None
    assert out["manifest"] is None
    assert out["stability"] is None
    assert out["n_windows_fitted"] == 0
    assert out["n_distinct_windows"] == 0
    assert [f["window"] for f in out["stability_failures"]] == [0, 1]
    assert all(
        f["reason"].startswith("RuntimeError: the subsample carries no episode")
        for f in out["stability_failures"]
    )


def _spy_on_evidence(exp6, monkeypatch) -> dict:
    """Record the settings the reported held-out evidence is taken under."""
    seen: dict = {}
    chain = exp6.evidence_chain

    def spy(data, params, graph, **kwargs):
        seen.update(kwargs)
        return chain(data, params, graph, **kwargs)

    monkeypatch.setattr(exp6, "evidence_chain", spy)
    return seen


def _run_small(exp6, tmp_path) -> dict:
    """One end-to-end run of the small log, returning what it wrote."""
    _small_log(tmp_path / "log.csv")
    exp6.main(
        csv=str(tmp_path / "log.csv"),
        slot_minutes=60,
        branching="24",
        regimes=2,
        max_lag=1,
        max_indegree=1,
        iters=1,
        subsamples=1,
        output_dir=str(tmp_path / "out"),
    )
    return json.loads((tmp_path / "out" / "exp6_results.json").read_text())


def test_the_reported_evidence_carries_the_estep_the_fit_ran(
    tmp_path, monkeypatch
) -> None:
    import experiments.exp6_real_log as exp6

    seen = _spy_on_evidence(exp6, monkeypatch)
    estimators = []
    fit = exp6._fit

    def record(*args, **kwargs):
        out = fit(*args, **kwargs)
        estimators.append(out[3])
        return out

    monkeypatch.setattr(exp6, "_fit", record)
    _run_small(exp6, tmp_path)

    assert seen["estep"] is estimators[0].estep
    assert seen["couple_scales"] == estimators[0].couple_scales


def test_a_fit_with_the_coupling_off_is_scored_with_the_coupling_off(
    tmp_path, monkeypatch
) -> None:
    import experiments.exp6_real_log as exp6
    from src.learning.structural_em import TopDownSEM

    seen = _spy_on_evidence(exp6, monkeypatch)
    monkeypatch.setattr(
        exp6,
        "TopDownSEM",
        lambda **kwargs: TopDownSEM(
            couple_scales=False, estep_kwargs=dict(max_inner_iter=3), **kwargs
        ),
    )
    out = _run_small(exp6, tmp_path)

    assert seen["couple_scales"] is False
    assert seen["estep"].max_inner_iter == 3
    assert np.isfinite(out["val_log_evidence"])
