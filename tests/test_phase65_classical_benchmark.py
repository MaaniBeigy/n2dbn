"""
tests/test_phase65_classical_benchmark.py
=========================================
Published structure learners on a pooled day-scale table, and the run that
scores them against a declared manifest.

Four properties carry the pooled design.  A lag reads inside one record, so a
design over two records holds no row pairing the tail of the first with the head
of the second.  The encoding is fixed once on the stacked rows, so a code means
the same thing in every record.  A label absent on three days in four keeps two
codes under the quantile encoding, which is what keeps a sparse column in the
search at all.  Pooling records adds evidence, so a relation too thin to clear
the BIC penalty in one record clears it over several.

Four properties carry the experiment.  The day-scale table lays a label's
episode count and its total minutes side by side, so a declared duration source
has a column.  A declared rule lands on the column pair its quantities name.  A
graph holding exactly the declared arrows scores perfect recall and no
structural distance.  Every edge into a declared-stationary label counts as a
false positive against the candidates into those labels alone.
"""

from __future__ import annotations

import json
from datetime import timedelta

import numpy as np
import pytest

from experiments import exp13_classical_on_benchmark as E
from src.core.typed_graph import COUNT, DURATION
from src.data.dynamic_patterns import BENCHMARK_ORIGIN, read_declared_manifest
from src.learning import classical as C

DAYS = 28
LABELS = ("quiet", "reading", "sleep", "walking")
LONG_NIGHT = 480
SHORT_NIGHT = 300
HEADER = ("person_id", "label", "start_iso", "end_iso")


# ---------------------------------------------------------------------------
# A benchmark small enough to check by hand
# ---------------------------------------------------------------------------


def _row(person: str, label: str, day: int, minute: int, length: int):
    start = BENCHMARK_ORIGIN + timedelta(days=day, minutes=minute)
    return (
        person,
        label,
        start.isoformat(timespec="seconds"),
        (start + timedelta(minutes=length)).isoformat(timespec="seconds"),
    )


def _episodes(person: str, offset: int):
    """
    One record where a long night is followed by extra walking the next day.

    Sleep runs long on the days whose index has bit 1 set, which alternates in
    pairs and so is uncorrelated with the day of the week.  Walking then holds
    one episode after a short night and three after a long one, which is the
    shape the benchmark's own sleep rule declares.
    """
    rows = []
    for day in range(DAYS):
        long_night = bool((day + offset) & 2)
        rows.append(
            _row(person, "sleep", day, 0, LONG_NIGHT if long_night else SHORT_NIGHT)
        )
        rows.append(_row(person, "reading", day, 700, 30 + 10 * (day % 3)))
        rows.append(_row(person, "quiet", day, 800, 20))
        yesterday_long = bool((day - 1 + offset) & 2) and day > 0
        for k in range(3 if yesterday_long else 1):
            rows.append(_row(person, "walking", day, 900 + 30 * k, 15))
    return rows


def _manifest():
    """A manifest declaring one carry-over rule and two stationary controls."""
    rule = {
        "source": "sleep",
        "target": "walking",
        "source_quantity": "max_duration",
        "target_quantity": "episodes",
        "lag_days": 1,
        "amount": 2.0,
        "direction": "increasing",
        "source_threshold": 390,
    }
    far = dict(rule, source="reading", source_quantity="count", lag_days=7)

    def entry(stationary, **extra):
        return {
            "kind": "routine",
            "category": "daily",
            "stationary": stationary,
            "personas": [],
            "patterns": [],
            **extra,
        }

    return {
        "quiet": entry(True),
        "reading": entry(False, edges_out=[far]),
        "sleep": entry(False, edges_out=[rule]),
        "walking": entry(False, edges_in=[rule, far]),
    }


@pytest.fixture(scope="module")
def benchmark(tmp_path_factory):
    """A three-record log and its declared manifest, written to disk."""
    root = tmp_path_factory.mktemp("dynamic")
    rows = []
    for offset, person in enumerate(("p_1", "p_2", "p_3")):
        rows.extend(_episodes(person, offset))
    log = root / "event_log.tsv"
    log.write_text(
        "\n".join(["\t".join(HEADER)] + ["\t".join(r) for r in rows]) + "\n",
        encoding="utf-8",
    )
    manifest = root / "manifest_declared.json"
    manifest.write_text(json.dumps(_manifest()), encoding="utf-8")
    return str(log), str(manifest)


@pytest.fixture(scope="module")
def tables(benchmark):
    return E.day_tables(*benchmark)


# ---------------------------------------------------------------------------
# The pooled design
# ---------------------------------------------------------------------------


def test_a_lag_reads_inside_one_record():
    first = np.arange(1, 5).reshape(4, 1)
    second = np.arange(11, 15).reshape(4, 1)
    lf = C.build_pooled_frame([first, second], max_lag=1, encoding="clip", n_bins=20)
    now = np.asarray(lf.frame["x0_t"]).astype(int).tolist()
    before = np.asarray(lf.frame["x0_lag1"]).astype(int).tolist()
    assert now == [2, 3, 4, 12, 13, 14]
    assert before == [1, 2, 3, 11, 12, 13]


def test_the_pooled_design_holds_every_record_minus_its_history():
    blocks = [np.arange(6).reshape(6, 1), np.arange(4).reshape(4, 1)]
    lf = C.build_pooled_frame(blocks, max_lag=2, encoding="clip", n_bins=8)
    assert lf.frame.shape == (4 + 2, 1 + 2)


def test_a_record_shorter_than_the_lag_is_refused():
    with pytest.raises(ValueError, match="cannot support a lag"):
        C.build_pooled_frame(
            [np.ones((5, 1)), np.ones((2, 1))], max_lag=3, encoding="binary"
        )


def test_the_encoding_is_read_off_the_stacked_records():
    """One record sits wholly below the pooled median and carries one code."""
    low = np.zeros((10, 1))
    high = np.full((10, 1), 5.0)
    lf = C.build_pooled_frame([low, high], max_lag=1, encoding="quantile", n_bins=2)
    codes = np.asarray(lf.frame["x0_t"]).astype(int)
    assert codes[:9].tolist() == [0] * 9
    assert codes[9:].tolist() == [1] * 9


def test_a_sparse_label_keeps_both_codes_under_the_quantile_cut():
    """A cut point is itself a tied value, and the mass on it stays below."""
    col = np.zeros((100, 1))
    col[:10] = 4.0
    codes = C.encode_counts(col, "quantile", n_bins=4)
    assert set(np.unique(codes)) == {0, 1}
    assert int((codes == 1).sum()) == 10


def test_the_pooled_design_matrix_stacks_and_scales_once():
    blocks = [np.arange(8).reshape(8, 1).astype(float) for _ in range(3)]
    W, Y, keys = C._pooled_design(blocks, max_lag=1)
    assert W.shape == (21, 1) and Y.shape == (21, 1) and keys == [(0, 1)]
    assert W.mean() == pytest.approx(0.0, abs=1e-12)
    assert W.std() == pytest.approx(1.0, abs=1e-12)


def test_one_record_pooled_is_the_single_record_search():
    rng = np.random.default_rng(0)
    X = rng.poisson(1.5, size=(120, 3))
    assert C.learn_pooled_structure(
        "hc-bic", [X], max_lag=1, max_indegree=1
    ) == C.learn_structure("hc-bic", X, max_lag=1, max_indegree=1)


def test_a_pooled_search_needs_a_record():
    with pytest.raises(ValueError, match="at least one record"):
        C.learn_pooled_structure("hc-bic", [], max_lag=1, max_indegree=1)


def test_pooling_records_buys_the_edge_one_record_cannot_carry():
    """The BIC penalty is paid once and the evidence for it is paid per record."""
    rng = np.random.default_rng(1)
    blocks = []
    for _ in range(12):
        driver = rng.integers(0, 2, size=12)
        follower = np.zeros(12, dtype=int)
        keep = rng.random(11) < 0.68
        follower[1:] = np.where(keep, driver[:-1], 1 - driver[:-1])
        blocks.append(np.column_stack([driver, follower]))
    alone = C.learn_pooled_structure(
        "hc-bic", blocks[:1], max_lag=1, max_indegree=1, encoding="binary"
    )
    pooled = C.learn_pooled_structure(
        "hc-bic", blocks, max_lag=1, max_indegree=1, encoding="binary"
    )
    assert alone[1] == set()
    assert pooled[1] == {(0, 1)}


def test_the_native_learners_read_every_pooled_record():
    """A relation planted in every record survives the stack the solvers fit."""
    rng = np.random.default_rng(5)
    blocks = []
    for _ in range(8):
        driver = rng.poisson(3.0, size=40)
        follower = np.zeros(40, dtype=int)
        follower[1:] = driver[:-1] * 3 + rng.poisson(0.2, size=39)
        blocks.append(np.column_stack([driver, follower]))
    for algorithm in ("granger", "dynotears"):
        got = C.learn_pooled_structure(algorithm, blocks, max_lag=1, max_indegree=1)
        assert (0, 1) in got[1], algorithm


# ---------------------------------------------------------------------------
# The day-scale table the run is built on
# ---------------------------------------------------------------------------


def test_the_observed_prefix_is_the_leading_run():
    assert E.observed_prefix(np.ones(5, dtype=bool)) == 5
    assert E.observed_prefix(np.array([True, True, False, False])) == 2


def test_a_record_observed_again_after_a_gap_is_refused():
    with pytest.raises(ValueError, match="read across the gap"):
        E.observed_prefix(np.array([True, False, True]))


def test_the_table_lays_counts_beside_durations(tables):
    day, truth = tables
    assert day.labels == list(LABELS)
    assert day.people == ["p_1", "p_2", "p_3"]
    assert day.n_labels == 4 and day.n_nodes == 8
    assert day.n_rows == 3 * DAYS
    assert all(b.shape == (DAYS, 8) for b in day.blocks)
    assert sorted(truth.stationary_labels) == ["quiet"]


def test_the_duration_column_carries_the_minutes_the_log_holds(tables):
    day, _ = tables
    sleep = day.labels.index("sleep")
    block = day.blocks[0]
    assert set(np.unique(block[:, sleep])) == {1.0}
    assert set(np.unique(block[:, sleep + day.n_labels])) == {
        float(SHORT_NIGHT),
        float(LONG_NIGHT),
    }


def test_a_node_name_carries_its_quantity(tables):
    day, _ = tables
    names = E.node_names(day.labels)
    assert names[0] == f"quiet:{COUNT}"
    assert names[day.n_labels] == f"quiet:{DURATION}"
    assert len(names) == day.n_nodes


# ---------------------------------------------------------------------------
# The declared truth on the node axis
# ---------------------------------------------------------------------------


def test_a_declared_rule_lands_on_the_columns_its_quantities_name(tables):
    day, truth = tables
    arrows = {a.name: a for a in E.declared_arrows(truth, day.labels)}
    sleep = arrows["sleep.max_duration->walking.episodes@lag1"]
    assert sleep.src_node == day.labels.index("sleep") + day.n_labels
    assert sleep.dst_node == day.labels.index("walking")
    assert sleep.key == (sleep.src_node, 1)


def test_a_rule_naming_an_unrealized_label_is_refused(tables):
    day, truth = tables
    with pytest.raises(ValueError, match="never realized"):
        E.declared_arrows(truth, [name for name in day.labels if name != "sleep"])


def test_the_declared_graph_scores_itself_perfectly(tables):
    day, truth = tables
    arrows = E.declared_arrows(truth, day.labels)
    row = E.score_learner(E.truth_parents(arrows), arrows, truth, day, max_lag=7)
    assert row["recall"] == 1.0 and row["precision"] == 1.0
    assert row["shd"] == 0 and row["n_declared"] == 2
    assert row["stationary_edges"] == 0


def test_the_lag_budget_caps_what_a_search_can_reach(tables):
    day, truth = tables
    arrows = E.declared_arrows(truth, day.labels)
    row = E.score_learner(E.truth_parents(arrows), arrows, truth, day, max_lag=2)
    assert row["n_reachable"] == 1
    assert row["recall_reachable"] == 1.0


# ---------------------------------------------------------------------------
# The stationary block
# ---------------------------------------------------------------------------


def test_every_edge_into_a_stationary_label_is_a_false_positive(tables):
    day, truth = tables
    quiet = day.labels.index("quiet")
    block = E.stationary_block({quiet: {(0, 1), (3, 2)}}, truth, day.labels, max_lag=2)
    assert block["n_stationary_labels"] == 1
    assert block["n_stationary_targets"] == 2
    assert block["stationary_candidates"] == 2 * 8 * 2
    assert block["stationary_edges"] == 2
    assert block["stationary_labels_with_parents"] == ["quiet"]


# ---------------------------------------------------------------------------
# What one declared edge is worth on its own
# ---------------------------------------------------------------------------


def test_the_ledger_ranks_a_planted_relation_first(tables):
    day, truth = tables
    arrows = E.declared_arrows(truth, day.labels)
    rows = {r["rule"]: r for r in E.rule_ledger(day, arrows, 2, "quantile", 2)}
    sleep = rows["sleep.max_duration->walking.episodes@lag1"]
    assert sleep["gain"] > 0.0
    assert sleep["rank"] == 1
    assert 1 <= sleep["n_positive"] <= sleep["n_candidates"]
    assert "rank    1" in E._ledger_line(sleep)


def test_the_ledger_reports_a_rule_the_lag_budget_cannot_reach(tables):
    day, truth = tables
    arrows = E.declared_arrows(truth, day.labels)
    rows = {r["rule"]: r for r in E.rule_ledger(day, arrows, 2, "quantile", 2)}
    far = rows["reading.count->walking.episodes@lag7"]
    assert far["in_lag_budget"] is False
    assert far["gain"] is None
    assert "lag past the budget" in E._ledger_line(far)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def test_the_run_scores_every_learner_and_writes_its_rows(benchmark, tmp_path):
    log, manifest = benchmark
    out = tmp_path / "exp13"
    result = E.main(
        csv_path=log,
        manifest_path=manifest,
        algorithms="hc-bic,empty",
        max_lag=2,
        max_indegree=2,
        n_bins=2,
        output_dir=str(out),
    )
    methods = [r["method"] for r in result["structure"]]
    assert methods == ["hc-bic", "empty"]
    hc, blank = result["structure"]
    assert "sleep.max_duration->walking.episodes@lag1" in hc["found"]
    assert hc["recall_reachable"] == 1.0
    assert blank["n_edges"] == 0 and blank["recall"] == 0.0
    assert len(result["ledger"]) == 2
    for stem in ("exp13_structure.csv", "exp13_ledger.csv", "exp13_results.json"):
        assert (out / stem).read_text().strip()


def test_a_failing_learner_leaves_the_others_scorable(benchmark, tmp_path, monkeypatch):
    log, manifest = benchmark
    monkeypatch.setattr(E, "learn_pooled_structure", lambda *a, **k: 1 / 0)
    result = E.main(
        csv_path=log,
        manifest_path=manifest,
        algorithms="empty",
        ledger=False,
        output_dir=str(tmp_path / "broken"),
    )
    assert result["structure"][0]["error"].startswith("ZeroDivisionError")


def test_an_unknown_learner_name_stops_the_run():
    with pytest.raises(SystemExit, match="unknown names"):
        E._resolve("hc-bic,notears-plus", E.DEFAULT_ALGORITHMS, ["hc-bic"])


def test_the_named_sets_resolve_to_the_lists_they_stand_for():
    assert E._resolve("default", ("a", "b"), ("a", "b", "c")) == ["a", "b"]
    assert E._resolve("all", ("a", "b"), ("a", "b", "c")) == ["a", "b", "c"]


def test_the_declared_manifest_of_the_benchmark_carries_six_rules():
    truth = read_declared_manifest("data/dynamic_patterns/manifest_declared.json")
    assert len(truth.rules) == 6
    assert len(truth.stationary_labels) == 43
