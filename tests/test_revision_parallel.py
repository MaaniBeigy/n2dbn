"""Small equivalence/resource checks, never the full experiment grids."""

import dataclasses
import multiprocessing as mp
import os
import time

import numpy as np
import pytest
from scipy.special import gammaln, logsumexp

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
def test_cached_likelihood_matches_recomputed_dm(monkeypatch, count_only):
    records, model = small_cohort(count_only)
    rec = records[0]
    q = empty_posterior(rec, model)
    target = model.levels[0].targets[0]
    parent = Parent("behavior", 1, 1)
    target.parents[parent] = np.array([0.3, -0.2])
    override = {parent: [np.arange(24) / 24, np.arange(24)[::-1] / 24]}
    # The optimized path reads each lagged feature once per channel.
    old_feature = law.feature
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return old_feature(*args, **kwargs)

    monkeypatch.setattr(law, "feature", counted)
    fast = law.allocation_logpmf(
        rec, model, 0, 0, q[0].gamma, q[0].m_H, override=override
    )
    assert len(calls) == (1 if count_only else 2)

    # A from-scratch reference recomputes every DM block mass by hand,
    # bypassing the cached replay structures.
    batch = law.blocks(rec, model, 0, 0)
    idx = batch.indices
    base = law.scores(rec, model, 0, 0, q[0].gamma, q[0].m_H)[idx]
    conc = model.levels[0].concentration
    block_start = idx[:, :1]
    slow = np.zeros(len(batch.parents))
    for channel in range(1 if count_only else 2):
        eta = base[:, :, channel].copy()
        for p in target.parents:
            values = law.feature(rec, 0, p, channel, override, idx)
            if p.kind == "behavior":
                lagged = idx - p.lag
                avail = (lagged >= 0) & (lagged < block_start)
                values = np.where(avail, values, 0.0)
            eta += target.parents[p][channel] * values
        if channel == 1 and not count_only:
            eta = np.where(batch.x > 0, eta, -np.inf)
        omega = np.exp(eta - logsumexp(eta, axis=1, keepdims=True))
        x = batch.x if channel == 0 else batch.y - batch.x
        a = np.maximum(conc[channel] * omega, 1e-10)
        if channel == 1 and not count_only:
            xs = np.where(batch.x > 0, x, 0)
            term = np.where(batch.x > 0, gammaln(xs + a) - gammaln(a), 0.0).sum(axis=1)
        else:
            xs = x
            term = (gammaln(xs + a) - gammaln(a)).sum(axis=1)
        n = xs.sum(axis=1)
        slow += (
            term
            + gammaln(conc[channel])
            - gammaln(n + conc[channel])
            + gammaln(n + 1)
            - gammaln(xs + 1).sum(axis=1)
        )
    np.testing.assert_allclose(fast, slow, rtol=1e-12, atol=1e-12)
    assert len(calls) > 2 or count_only


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


@pytest.mark.parametrize("count_only", [False, True])
def test_frozen_concentration_path_matches_direct_replay(count_only):
    """The concentration refit freezes softmax weights; its objective must equal
    a from-scratch replay at every proposed concentration."""
    records, model = small_cohort(count_only)
    rec = records[0]
    rec._cache_prepared = True
    q = empty_posterior(rec, model)
    target = model.levels[0].targets[0]
    parent = Parent("behavior", 1, 1)
    target.parents[parent] = np.array([0.3, -0.2])
    target.depletion[:] = [0.4, 0.7]
    frozen = law.prepare_frozen(rec, model, 0, 0, q[0].gamma, q[0].m_H)
    for conc in (np.array([0.5, 2.0]), np.array([30.0, 30.0]), np.array([200.0, 0.1])):
        model.levels[0].concentration = conc
        direct = law.allocation_logpmf(rec, model, 0, 0, q[0].gamma, q[0].m_H)
        fast = frozen.logpmf(conc)
        np.testing.assert_allclose(fast, direct, rtol=1e-12, atol=1e-12)


def test_prepared_cache_rebuilds_when_lag_bound_changes():
    records, model = small_cohort()
    rec = records[0]
    rec._cache_prepared = True
    q = empty_posterior(rec, model)
    first = law.prepare(rec, model, 0, 0)
    again = law.prepare(rec, model, 0, 0)
    assert first is again
    wider = dataclasses.replace(model)
    wider.lag_bound = (2, 1, 1)
    rebuilt = law.prepare(rec, wider, 0, 0)
    assert rebuilt is not first and rebuilt.lag_bound == 2


def test_frozen_worker_payload_stays_small_across_a_pipe():
    """Frozen weights returned by fork workers must not carry the record."""
    import pickle

    records, model = small_cohort()
    rec = records[0]
    rec._cache_prepared = True
    q = empty_posterior(rec, model)
    model.levels[0].targets[0].parents[Parent("behavior", 1, 1)] = np.array([0.3, -0.2])
    frozen = law.prepare_frozen(rec, model, 0, 0, q[0].gamma, q[0].m_H)
    blob = pickle.dumps(frozen.light())
    assert len(blob) < 1_000_000
    model.levels[0].concentration = np.array([7.0, 11.0])
    direct = law.allocation_logpmf(rec, model, 0, 0, q[0].gamma, q[0].m_H)
    restored = law.Frozen.assemble(law.prepare(rec, model, 0, 0), pickle.loads(blob))
    np.testing.assert_allclose(
        restored.logpmf(model.levels[0].concentration), direct, rtol=1e-12, atol=1e-12
    )
