"""
Machine resources and the budgets derived from them.

Nothing here changes a fitted number.  What it guards is that a study asks for
a worker count the machine can actually hold, and that every fallback path of
the detection code returns something usable instead of raising.
"""

from __future__ import annotations

import subprocess

import pytest

from src.utils import resources as R

# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


def test_read_meminfo_parses_kilobytes(tmp_path):
    path = tmp_path / "meminfo"
    path.write_text(
        "MemTotal:       16384000 kB\n"
        "MemFree:         1024000 kB\n"
        "MemAvailable:    8192000 kB\n"
        "HugePages_Total:       0\n"
    )
    info = R._read_meminfo(str(path))
    assert info["MemTotal"] == pytest.approx(16384000 * 1024 / 1024**3)
    assert "HugePages_Total" in info


def test_read_meminfo_returns_empty_when_absent(tmp_path):
    assert R._read_meminfo(str(tmp_path / "nope")) == {}


def test_memory_gb_uses_meminfo(monkeypatch):
    monkeypatch.setattr(
        R, "_read_meminfo", lambda *a: {"MemTotal": 8.0, "MemAvailable": 3.0}
    )
    assert R._memory_gb() == (8.0, 3.0)


def test_memory_gb_falls_back_to_memfree(monkeypatch):
    monkeypatch.setattr(
        R, "_read_meminfo", lambda *a: {"MemTotal": 8.0, "MemFree": 2.0}
    )
    assert R._memory_gb() == (8.0, 2.0)


def test_memory_gb_falls_back_to_sysconf(monkeypatch):
    monkeypatch.setattr(R, "_read_meminfo", lambda *a: {})
    monkeypatch.setattr(
        R.os, "sysconf", lambda name: 1024 if name == "SC_PHYS_PAGES" else 1024**2
    )
    total, avail = R._memory_gb()
    assert total == avail == pytest.approx(1.0)


def test_memory_gb_returns_zero_when_nothing_answers(monkeypatch):
    monkeypatch.setattr(R, "_read_meminfo", lambda *a: {})

    def boom(_name):
        raise ValueError("no such configuration value")

    monkeypatch.setattr(R.os, "sysconf", boom)
    assert R._memory_gb() == (0.0, 0.0)


# ---------------------------------------------------------------------------
# GPUs
# ---------------------------------------------------------------------------


class _Completed:
    def __init__(self, stdout, returncode=0):
        self.stdout = stdout
        self.returncode = returncode


def test_gpu_devices_empty_without_nvidia_smi(monkeypatch):
    monkeypatch.setattr(R.shutil, "which", lambda _name: None)
    assert R.gpu_devices() == []


def test_gpu_devices_parses_two_cards(monkeypatch):
    monkeypatch.setattr(R.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        R.subprocess,
        "run",
        lambda *a, **k: _Completed("0, H200, 7168, 7000\n1, H200, 7168, 1024\n"),
    )
    gpus = R.gpu_devices()
    assert [g.index for g in gpus] == [0, 1]
    assert gpus[0].name == "H200"
    assert gpus[0].total_memory_gb == pytest.approx(7.0)
    assert gpus[1].free_memory_gb == pytest.approx(1.0)


def test_gpu_devices_skips_unparseable_rows(monkeypatch):
    monkeypatch.setattr(R.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        R.subprocess,
        "run",
        lambda *a, **k: _Completed(
            "not, enough\n0, H200, seven, 7000\n1, H200, 7168, 7000\n"
        ),
    )
    gpus = R.gpu_devices()
    assert [g.index for g in gpus] == [1]


def test_gpu_devices_empty_on_nonzero_exit(monkeypatch):
    monkeypatch.setattr(R.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(R.subprocess, "run", lambda *a, **k: _Completed("", 9))
    assert R.gpu_devices() == []


def test_gpu_devices_empty_when_the_call_fails(monkeypatch):
    monkeypatch.setattr(R.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")

    def boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=1)

    monkeypatch.setattr(R.subprocess, "run", boom)
    assert R.gpu_devices() == []


def test_detect_reports_the_machine(monkeypatch):
    monkeypatch.setattr(R, "_memory_gb", lambda: (16.0, 8.0))
    monkeypatch.setattr(R, "gpu_devices", lambda: [])
    res = R.detect()
    assert res.n_cpu >= 1
    assert res.total_memory_gb == 16.0
    assert not res.has_gpu


# ---------------------------------------------------------------------------
# Budgets
# ---------------------------------------------------------------------------


def _machine(n_cpu=8, avail=16.0, gpus=()):
    return R.MachineResources(
        n_cpu=n_cpu, total_memory_gb=avail, available_memory_gb=avail, gpus=list(gpus)
    )


@pytest.mark.parametrize("requested", [1, None])
def test_resolve_jobs_serial(requested):
    assert R.resolve_jobs(requested, resources=_machine()) == 1


def test_resolve_jobs_all_cores_leaves_one():
    assert R.resolve_jobs(-1, resources=_machine(n_cpu=8)) == 7


def test_resolve_jobs_caps_at_the_core_count():
    assert R.resolve_jobs(64, resources=_machine(n_cpu=8)) == 8


def test_resolve_jobs_caps_at_memory():
    # 16 GB available less 1 GB reserved, at 3 GB a worker, is five workers.
    got = R.resolve_jobs(8, memory_per_job_gb=3.0, resources=_machine(avail=16.0))
    assert got == 5


def test_resolve_jobs_never_returns_zero():
    got = R.resolve_jobs(8, memory_per_job_gb=100.0, resources=_machine(avail=2.0))
    assert got == 1


def test_resolve_jobs_ignores_memory_when_it_is_unknown():
    machine = R.MachineResources(n_cpu=4, total_memory_gb=0.0, available_memory_gb=0.0)
    assert R.resolve_jobs(-1, memory_per_job_gb=3.0, resources=machine) == 3


def test_resolve_jobs_detects_when_no_machine_is_given():
    assert R.resolve_jobs(2) >= 1


# ---------------------------------------------------------------------------
# Threads and description
# ---------------------------------------------------------------------------


def test_configure_threads_sets_every_variable(monkeypatch):
    for var in R._THREAD_VARS:
        monkeypatch.delenv(var, raising=False)
    applied = R.configure_threads(2)
    assert set(applied) == set(R._THREAD_VARS)
    assert all(R.os.environ[v] == "2" for v in R._THREAD_VARS)


def test_configure_threads_leaves_an_explicit_setting_alone(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "6")
    for var in R._THREAD_VARS[1:]:
        monkeypatch.delenv(var, raising=False)
    applied = R.configure_threads(1)
    assert "OMP_NUM_THREADS" not in applied
    assert R.os.environ["OMP_NUM_THREADS"] == "6"


def test_describe_without_a_gpu():
    text = R.describe(_machine())
    assert "gpu     : none visible" in text
    assert "cpu     : 8 logical processors" in text
    assert "jobs    :" in text


def test_describe_lists_every_gpu():
    gpu = R.GpuInfo(index=0, name="H200", total_memory_gb=7.0, free_memory_gb=7.0)
    text = R.describe(_machine(gpus=[gpu]))
    assert "gpu 0" in text and "H200" in text


def test_describe_detects_when_no_machine_is_given():
    assert "cpu     :" in R.describe()
