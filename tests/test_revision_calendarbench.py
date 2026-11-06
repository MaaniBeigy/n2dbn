"""Small fixtures for CalendarBench grouping; no experiment grids are run."""

import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import revision
from src.data.dynamic_patterns import (
    BENCHMARK_ORIGIN,
    VERBATIM,
    build_population,
    read_subject_event_log,
)
from src.nested.calendarbench import CALENDARBENCH_OBSERVATION, calendarbench
from src.nested.data import check_round_trip, read_log, write_log
from src.nested.report import report


@pytest.fixture
def calendarbench_tsv(tmp_path):
    path = tmp_path / "events.tsv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["person_id", "label", "start_iso", "end_iso"])
        writer.writerows(
            [
                ("a", "hungry", "2026-02-16T00:00:00", "2026-02-16T00:10:00"),
                ("a", "hungry", "2026-02-16T00:00:20", "2026-02-16T00:01:50"),
                ("a", "hungry", "2026-02-16T00:01:00", "2026-02-16T00:04:00"),
                ("a", "sleep", "2026-02-16T03:00:00", "2026-02-24T03:00:00"),
                ("a", "sleep", "2026-02-16T03:00:00", "2026-02-24T04:00:00"),
                ("a", "walking", "2026-02-17T00:00:00", "2026-02-17T01:00:00"),
                ("b", "hungry", "2026-02-16T00:00:00", "2026-02-16T00:00:00"),
                ("b", "walking", "2026-02-17T01:00:00", "2026-02-17T02:00:00"),
                ("c", "walking", "2026-02-16T00:00:00", "2026-02-16T01:00:00"),
            ]
        )
    return path


def test_calendarbench_grouping_preserves_marks_schedule_and_original_rows(
    tmp_path, calendarbench_tsv
):
    original = calendarbench_tsv.read_bytes()
    destination = tmp_path / "annotations.csv"
    records, audit = calendarbench(calendarbench_tsv, 2, annotation_path=destination)
    assert audit["valid"]
    assert not audit["before_grouping"]["valid"]
    assert audit["observation_definition"] == CALENDARBENCH_OBSERVATION
    assert sum(audit["before_grouping"]["collision_cells"].values()) == 2
    assert (
        sum(v["annotations_before"] for v in audit["grouping"]["by_label"].values())
        == 8
    )
    assert sum(v["groups_after"] for v in audit["grouping"]["by_label"].values()) == 6
    data = records[0].data
    hungry, sleep = data.label_names.index("hungry"), data.label_names.index("sleep")
    assert data.X[0][0, hungry] == data.X[0][1, hungry] == 1
    assert data.Y[0][0, hungry] == 12  # ceil each mark individually, then sum
    assert data.Y[0][1, hungry] == 3  # overlapping, different-minute start survives
    assert data.Y[0][180, sleep] == 23100
    assert records[1].data.Y[0][0, hungry] == 1
    assert data.calendar.origin == BENCHMARK_ORIGIN
    assert (
        data.hierarchy.H[0] == 2 * 7 * 1440
    )  # original horizon, not summed-mark reach
    assert data.get_observed(1).sum() == 7 * 1440

    subjects = read_subject_event_log(str(calendarbench_tsv))
    subjects.pop("c")
    for rec, (name, raw, _) in zip(
        records, build_population(subjects, data.label_names, policy=VERBATIM)
    ):
        assert rec.name == name
        np.testing.assert_array_equal(rec.data.get_observed(1), raw.get_observed(1))
        for k in range(data.K):
            np.testing.assert_array_equal(rec.data.Y[k], raw.Y[k])
        rec.validate()
    with calendarbench_tsv.open() as handle:
        raw_rows = list(csv.DictReader(handle, delimiter="\t"))
    with destination.open() as handle:
        saved_rows = list(csv.DictReader(handle))
    assert len(saved_rows) == len(raw_rows) == audit["annotation_audit"]["rows"] == 9
    for line, (raw, saved) in enumerate(zip(raw_rows, saved_rows), start=2):
        assert saved["tsv_line"] == str(line)
        assert all(saved["raw_" + key] == value for key, value in raw.items())
    assert saved_rows[-1]["status"] == "excluded_participant"
    assert saved_rows[0]["group_id"] == saved_rows[1]["group_id"]
    assert saved_rows[2]["group_id"] != saved_rows[0]["group_id"]
    assert calendarbench_tsv.read_bytes() == original
    exported = tmp_path / "grouped.csv"
    write_log(exported, records)
    assert check_round_trip(records, read_log(exported, records))


def test_calendarbench_audit_stops_before_fitting_and_flags_changed_manifest_counts(
    tmp_path, calendarbench_tsv, monkeypatch, capsys
):
    from src.data import dynamic_patterns

    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    rules = [
        SimpleNamespace(
            source=source,
            target="walking",
            source_quantity=quantity,
            target_quantity=target,
            lag_days=1,
        )
        for source, quantity, target in (
            ("hungry", "count", "episodes"),
            ("walking", "count", "episodes"),
            ("sleep", "total_duration", "duration"),
            ("sleep", "max_duration", "episodes"),
        )
    ]
    monkeypatch.setattr(
        dynamic_patterns,
        "read_declared_manifest",
        lambda path: SimpleNamespace(rules=rules),
    )
    monkeypatch.setattr(
        revision, "selected_fit", lambda *a, **kw: pytest.fail("audit must not fit")
    )
    monkeypatch.setattr(
        revision, "fitting", lambda *a, **kw: pytest.fail("audit must not fit")
    )
    folder = tmp_path / "benchmark"
    args = revision.parser().parse_args(
        [
            "benchmark",
            "--audit-only",
            "--participants",
            "2",
            "--csv",
            str(calendarbench_tsv),
            "--manifest",
            str(manifest),
            "--output",
            str(folder),
        ]
    )
    revision.benchmark(args)
    assert "8 annotations -> 6 groups" in capsys.readouterr().out
    assert not (folder / "runs.json").exists()
    assert (folder / "annotations.csv").is_file()
    assert json.loads((folder / "audit.json").read_text())["valid"]
    ledger = json.loads((folder / "manifest-reachability.json").read_text())
    assert ledger[0]["count_quantity_changed_by_grouping"]
    assert not ledger[0]["expressible"]
    assert ledger[1]["expressible"]
    assert ledger[2]["expressible"]  # summed duration was preserved
    assert not ledger[3]["expressible"]  # maximum is still not a summed duration
    result = report(tmp_path)
    assert (
        result["experiments"]["benchmark"]["observation_definition"]
        == CALENDARBENCH_OBSERVATION
    )
    assert (tmp_path / "tables" / "benchmark-grouping.csv").is_file()
    markdown = (tmp_path / "REPORT.md").read_text()
    assert "raw manifest episode counts can differ" in markdown
    assert all(c["status"] == "inconclusive" for c in result["claims"])
