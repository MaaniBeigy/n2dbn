"""Small equivalence/resource checks, never the full experiment grids."""

import dataclasses
import multiprocessing as mp
import os
import time

import numpy as np
import pytest

from src.nested import law, parallel
from src.nested.data import from_finest
from src.nested.fit import FitConfig, Fitter
from src.nested.inference import empty_posterior, monitor
from src.nested.model import Parent, initialize

fork_only = pytest.mark.skipif(
    "fork" not in mp.get_all_start_methods(), reason="Linux fork backend"
)


def small_cohort(count_only=False, labels=2):
    rng = np.random.default_rng(44)
    records = []
    for i in range(2):
        x = rng.binomial(1, 0.3, (24, labels))
        if count_only:
            x *= 2
        y = x * rng.integers(1, 6, x.shape)
        records.append(from_finest(str(i), x, y, [3, 2]))
    model = initialize(
        records,
        [1, 1, 2],
        [0, 0, 1],
        [1, 1, 1],
        penalty=0,
        count_only=count_only,
        seed=9,
    )
    return records, model


def assert_same(a, b):
    if dataclasses.is_dataclass(a):
        for field in dataclasses.fields(a):
            assert_same(getattr(a, field.name), getattr(b, field.name))
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_same(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_same(x, y)
    else:
        assert a == b


@fork_only
@pytest.mark.parametrize("count_only", [False, True])
def test_parallel_fit_matches_serial_parameters_search_and_monitor(
    monkeypatch, count_only
):
    # Force two workers on small CI containers; the fixture is only 48 slots.
    monkeypatch.setattr(
        parallel, "resources", lambda: (2, 10 * parallel.GIB, parallel.GIB)
    )
    records, model = small_cohort(count_only, labels=3)
    options = dict(outer=1, inner=1, profile=1, shifts=2, indegree=1, seed=7)
    serial = Fitter(FitConfig(**options, jobs=1)).fit(records, model)
    forked = Fitter(FitConfig(**options, jobs=2)).fit(records, model)
    assert any(p["workers"] == 2 for p in forked.parallel_phases)
    assert serial.candidate_refits == forked.candidate_refits
    assert serial.evaluations == forked.evaluations
    assert serial.shift_checks  # Exercise seeded null refits, not only empty graphs.
    assert_same(serial.shift_checks, forked.shift_checks)
    assert_same(serial.trace, forked.trace)
    assert_same(serial.model, forked.model)
    assert_same(serial.posteriors, forked.posteriors)
    assert forked.trace[-1]["retained"] == monitor(
        records, forked.model, forked.posteriors
    )


@pytest.mark.parametrize("count_only", [False, True])
def test_cached_likelihood_matches_recomputed_stages(monkeypatch, count_only):
    records, model = small_cohort(count_only)
    rec = records[0]
    q = empty_posterior(rec, model)
    target = model.levels[0].targets[0]
    parent = Parent("behavior", 1, 1)
    target.parents[parent] = np.array([0.3, -0.2])
    target.depletion[:] = [0.4, 0.7]
    override = {parent: [np.arange(24) / 24, np.arange(24)[::-1] / 24]}
    # A stage-by-stage reference deliberately bypasses the optimization.
    old_stage = law.stage_scores
    calls = []
    old_feature = law.feature

    def counted(*args, **kwargs):
        calls.append(1)
        return old_feature(*args, **kwargs)

    monkeypatch.setattr(law, "feature", counted)
    fast = law.allocation_logpmf(
        rec, model, 0, 0, q[0].gamma, q[0].m_H, override=override
    )
    assert len(calls) == (1 if count_only else 2)

    def recompute(*args, **kwargs):
        kwargs.pop("prefix", None)
        kwargs.pop("features", None)
        return old_stage(*args, **kwargs)

    monkeypatch.setattr(law, "stage_scores", recompute)
    slow = law.allocation_logpmf(
        rec, model, 0, 0, q[0].gamma, q[0].m_H, override=override
    )
    np.testing.assert_array_equal(fast, slow)
    assert len(calls) > 2


@fork_only
def test_worker_plan_caps_memory_cpu_and_task_count(monkeypatch):
    monkeypatch.setattr(
        parallel, "resources", lambda: (8, 15 * parallel.GIB, 6 * parallel.GIB)
    )
    assert parallel.worker_plan(8, 12, 60)["workers"] == 6
    assert parallel.worker_plan(8, 12, 2)["workers"] == 2
    assert parallel.worker_plan(1, 12, 60)["workers"] == 1
    monkeypatch.setattr(
        parallel, "resources", lambda: (8, parallel.GIB // 2, 6 * parallel.GIB)
    )
    assert parallel.worker_plan(8, 12, 60)["workers"] == 1


def test_resources_honor_docker_cgroup_limits(monkeypatch):
    paths = {
        "/proc/meminfo": "MemAvailable: 20000000 kB\n",
        "/proc/self/statm": "10000 1000\n",
        "/sys/fs/cgroup/cpu.max": "200000 100000",
        "/sys/fs/cgroup/memory.max": str(8 * parallel.GIB),
        "/sys/fs/cgroup/memory.current": str(6 * parallel.GIB),
    }

    def read(path):
        if str(path) not in paths:
            raise FileNotFoundError(str(path))
        return paths[str(path)]

    monkeypatch.setattr(parallel.Path, "read_text", read)
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    monkeypatch.setattr(
        os, "sched_getaffinity", lambda pid: set(range(8)), raising=False
    )
    cpus, available, _ = parallel.resources()
    assert cpus == 2
    assert available == 2 * parallel.GIB


@fork_only
@pytest.mark.parametrize("failure", ["exception", "timeout", "killed"])
def test_workers_cleanup_after_failure(failure):
    before = {p.pid for p in mp.active_children()}

    def work(i):
        if i == 0:
            if failure == "killed":
                os._exit(7)
            raise (TimeoutError if failure == "timeout" else ValueError)(
                "fixture failure"
            )
        time.sleep(0.3)
        return i

    error = {"exception": ValueError, "timeout": TimeoutError, "killed": RuntimeError}[
        failure
    ]
    with pytest.raises(error):
        parallel.ordered_map(work, [0, 1], {"workers": 2})
    assert {p.pid for p in mp.active_children()} == before
    assert parallel._FUNCTION is None
    # Completion order and a fresh snapshot cannot change task/result pairing.
    multiplier = 3
    assert parallel.ordered_map(
        lambda i: i * multiplier, [2, 0, 1], {"workers": 2}
    ) == [6, 0, 3]


def test_runner_passes_resource_options_to_fit():
    from experiments.revision import fit_config, parser

    args = parser().parse_args(["benchmark", "--jobs", "6", "--memory-gb", "12"])
    config = fit_config(args)
    assert config.jobs == 6
    assert config.memory_gb == 12


@fork_only
def test_parent_deadline_terminates_workers():
    from experiments.revision import deadline

    before = {p.pid for p in mp.active_children()}
    with pytest.raises(TimeoutError):
        with deadline(0.3):
            parallel.ordered_map(lambda i: time.sleep(2), [0, 1], {"workers": 2})
    assert {p.pid for p in mp.active_children()} == before
    assert parallel._FUNCTION is None
