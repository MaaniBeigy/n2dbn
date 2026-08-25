"""
tests/test_phase35_provenance.py
================================
Recording which code produced a stored result.

The experiment services run the source baked into the image and the development
services mount the working tree over it, so a results file on its own cannot say
which program wrote it. The digest below is taken at build time and again at run
time; equal means the image's own code ran, different means something was
mounted over it. These tests pin that the comparison is made from the source
that decides a result and from nothing else.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.utils.provenance import (
    BUILD_INFO_PATH,
    SOURCE_ROOTS,
    check,
    main,
    provenance,
    read_build_info,
    source_digest,
    write_build_info,
    write_provenance,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tree(tmp_path):
    """A miniature source tree with one file under each recorded root."""
    for name in SOURCE_ROOTS:
        (tmp_path / name).mkdir()
        (tmp_path / name / "mod.py").write_text(f"VALUE = '{name}'\n")
    return tmp_path


# ---------------------------------------------------------------------------
# The digest
# ---------------------------------------------------------------------------


def test_the_same_source_digests_the_same_way(tree):
    assert source_digest(str(tree)) == source_digest(str(tree))


def test_changing_a_source_file_changes_the_digest(tree):
    before = source_digest(str(tree))
    (tree / SOURCE_ROOTS[0] / "mod.py").write_text("VALUE = 'moved'\n")
    assert source_digest(str(tree)) != before


def test_renaming_a_file_changes_the_digest(tree):
    # The path is hashed beside the content, so moving code between files is a
    # change even when every byte of every file is preserved.
    before = source_digest(str(tree))
    (tree / SOURCE_ROOTS[0] / "mod.py").rename(tree / SOURCE_ROOTS[0] / "other.py")
    assert source_digest(str(tree)) != before


def test_a_change_outside_the_recorded_roots_is_not_a_change(tree):
    # Tests and the manuscript do not decide what a run computes, so editing
    # them leaves a stored result valid and must not read as a stale image.
    before = source_digest(str(tree))
    (tree / "tests").mkdir()
    (tree / "tests" / "test_x.py").write_text("assert True\n")
    assert source_digest(str(tree)) == before


def test_bytecode_caches_are_not_part_of_the_source(tree):
    before = source_digest(str(tree))
    cache = tree / SOURCE_ROOTS[0] / "__pycache__"
    cache.mkdir()
    (cache / "mod.py").write_text("garbage\n")
    assert source_digest(str(tree)) == before


def test_a_missing_root_is_skipped_rather_than_raising(tmp_path):
    assert isinstance(source_digest(str(tmp_path)), str)


# ---------------------------------------------------------------------------
# The build stamp
# ---------------------------------------------------------------------------


def test_the_stamp_records_the_digest_and_the_revision(tree, tmp_path):
    path = tmp_path / "BUILD_INFO.json"
    info = write_build_info(str(path), str(tree), git_rev="abc1234")
    assert info["source_digest"] == source_digest(str(tree))
    assert info["git_rev"] == "abc1234"
    assert json.loads(path.read_text()) == info


def test_an_unstamped_image_reads_as_empty_rather_than_raising(tmp_path):
    assert read_build_info(str(tmp_path / "absent.json")) == {}


def test_an_unreadable_stamp_reads_as_empty_rather_than_raising(tmp_path):
    path = tmp_path / "BUILD_INFO.json"
    path.write_text("not json at all")
    assert read_build_info(str(path)) == {}


def test_the_default_stamp_path_is_the_image_workspace():
    assert BUILD_INFO_PATH.endswith("BUILD_INFO.json")


# ---------------------------------------------------------------------------
# The provenance record
# ---------------------------------------------------------------------------


def test_a_run_on_the_source_it_was_built_from_says_so(tree, tmp_path):
    path = tmp_path / "BUILD_INFO.json"
    write_build_info(str(path), str(tree), git_rev="deadbee")
    record = provenance(str(tree), str(path))
    assert record["runs_image_source"] is True
    assert record["git_rev"] == "deadbee"
    assert record["image_source_digest"] == record["live_source_digest"]


def test_a_run_on_source_mounted_over_the_image_says_so(tree, tmp_path):
    path = tmp_path / "BUILD_INFO.json"
    write_build_info(str(path), str(tree))
    (tree / SOURCE_ROOTS[0] / "mod.py").write_text("VALUE = 'mounted'\n")
    record = provenance(str(tree), str(path))
    assert record["runs_image_source"] is False
    assert record["image_source_digest"] != record["live_source_digest"]


def test_an_unstamped_image_is_not_reported_as_a_match(tree, tmp_path):
    record = provenance(str(tree), str(tmp_path / "absent.json"))
    assert record["image_source_digest"] == "unknown"
    assert record["runs_image_source"] is False


def test_the_record_carries_the_versions_that_can_move_a_number(tree, tmp_path):
    record = provenance(str(tree), str(tmp_path / "absent.json"))
    assert record["packages"]["numpy"] != "absent"
    assert "python" in record and "run_at" in record


def test_a_package_that_is_not_installed_is_named_rather_than_omitted(
    tree, tmp_path, monkeypatch
):
    import src.utils.provenance as prov

    monkeypatch.setattr(prov, "TRACKED_PACKAGES", ("no-such-package-anywhere",))
    record = provenance(str(tree), str(tmp_path / "absent.json"))
    assert record["packages"]["no-such-package-anywhere"] == "absent"


def test_the_record_is_written_beside_the_results(tree, tmp_path):
    out = tmp_path / "results" / "expX"
    record = write_provenance(str(out), str(tree), str(tmp_path / "absent.json"))
    assert json.loads((out / "provenance.json").read_text()) == record


def test_writing_the_record_creates_a_results_directory_that_does_not_exist(
    tree, tmp_path
):
    out = tmp_path / "deep" / "nested" / "expY"
    write_provenance(str(out), str(tree), str(tmp_path / "absent.json"))
    assert (out / "provenance.json").is_file()


# ---------------------------------------------------------------------------
# The staleness check
# ---------------------------------------------------------------------------


def test_a_current_image_checks_clean(tree, tmp_path, capsys):
    path = tmp_path / "BUILD_INFO.json"
    write_build_info(str(path), str(tree))
    assert check(str(tree), str(path)) == 0
    assert "current" in capsys.readouterr().out


def test_a_stale_image_fails_the_check_and_says_how_to_fix_it(tree, tmp_path, capsys):
    path = tmp_path / "BUILD_INFO.json"
    write_build_info(str(path), str(tree))
    (tree / SOURCE_ROOTS[0] / "mod.py").write_text("VALUE = 'newer'\n")
    assert check(str(tree), str(path)) == 1
    out = capsys.readouterr().out
    assert "STALE" in out
    assert "docker compose build" in out


def test_an_image_with_no_stamp_is_distinguished_from_a_stale_one(
    tree, tmp_path, capsys
):
    # An image built before provenance existed cannot be compared, which is a
    # different situation from one that is provably behind.
    assert check(str(tree), str(tmp_path / "absent.json")) == 2
    assert "no build stamp" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def test_the_write_subcommand_stamps_and_reports(tree, tmp_path, capsys):
    path = tmp_path / "BUILD_INFO.json"
    code = main(["--write", "--root", str(tree), "--build-info", str(path)])
    assert code == 0
    assert "stamped" in capsys.readouterr().out
    assert Path(path).is_file()


def test_the_check_subcommand_returns_the_check_status(tree, tmp_path):
    path = tmp_path / "BUILD_INFO.json"
    main(["--write", "--root", str(tree), "--build-info", str(path)])
    assert main(["--check", "--root", str(tree), "--build-info", str(path)]) == 0


def test_with_no_subcommand_the_record_is_printed(tree, tmp_path, capsys):
    code = main(["--root", str(tree), "--build-info", str(tmp_path / "absent.json")])
    assert code == 0
    assert "live_source_digest" in json.loads(capsys.readouterr().out)
