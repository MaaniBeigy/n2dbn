"""Bounded Linux fork workers sharing the read-only cohort by copy-on-write.

Only small task identifiers and results cross the process queues. Whole fits
are never dispatched to separate workers: their minute panels are too large.
The memory calculation is a conservative admission estimate, not a hard limit.
"""

from __future__ import annotations

import gc
import multiprocessing as mp
import os
import signal
import threading
from pathlib import Path

from threadpoolctl import threadpool_limits

GIB = 1024**3
_FUNCTION = None


def _number(path):
    try:
        return int(Path(path).read_text().strip())
    except (OSError, ValueError):
        return None


def resources():
    """Host availability constrained by affinity and common Docker cgroups."""
    cpus = os.cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        cpus = min(cpus, len(os.sched_getaffinity(0)))
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            cpus = min(cpus, max(1, int(quota) // int(period)))
    except (OSError, ValueError):
        quota = _number("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        period = _number("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        if quota and quota > 0 and period:
            cpus = min(cpus, max(1, quota // period))
    available = None
    rss = 0
    try:
        fields = dict(
            line.split(":", 1)
            for line in Path("/proc/meminfo").read_text().splitlines()
        )
        available = int(fields["MemAvailable"].split()[0]) * 1024
        rss = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf(
            "SC_PAGE_SIZE"
        )
    except (OSError, ValueError, KeyError, IndexError):
        pass
    for limit_path, used_path in (
        ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
        (
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
            "/sys/fs/cgroup/memory/memory.usage_in_bytes",
        ),
    ):
        limit, used = _number(limit_path), _number(used_path)
        if limit is not None and used is not None:
            remaining = max(0, limit - used)
            available = remaining if available is None else min(available, remaining)
    return cpus, available, rss


def worker_plan(jobs, memory_gb, tasks, private_bytes=GIB):
    """Reserve 1 GiB outside workers; cap total estimated RSS if requested."""
    if jobs < 0 or memory_gb < 0 or tasks < 1:
        raise ValueError("jobs/memory must be nonnegative and tasks positive")
    cpus, available, rss = resources()
    requested = jobs or min(6, max(1, cpus - 1))
    workers = min(requested, cpus, tasks)
    headroom = None if available is None else max(0, available - GIB)
    if memory_gb:
        budget = max(0, int(memory_gb * GIB) - rss)
        headroom = budget if headroom is None else min(headroom, budget)
    if headroom is None:
        workers = 1  # Unknown memory: do not multiply large working sets.
    else:
        workers = min(workers, max(1, int(headroom // private_bytes)))
    if "fork" not in mp.get_all_start_methods() or mp.current_process().daemon:
        workers = 1
    return dict(
        workers=workers,
        requested_jobs=jobs,
        cpu_limit=cpus,
        memory_budget_gb=memory_gb,
        available_gb=None if available is None else available / GIB,
        parent_rss_gb=rss / GIB,
        estimated_private_gb_per_worker=private_bytes / GIB,
        backend="fork" if workers > 1 else "serial",
    )


def _initialize_worker():
    # The parent handles Ctrl-C and terminates every worker in finally.
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _invoke(item):
    return _FUNCTION(item)


def ordered_map(function, items, plan):
    """Deterministic result order; terminate workers on errors and timeouts.

    A fresh pool inherits the current model/posterior snapshot. Reusing a pool
    across M-steps would score stale parameters. BLAS remains single-threaded
    inside each worker. This helper deliberately disallows nested/threaded use.
    """
    global _FUNCTION
    items = list(items)
    if plan["workers"] == 1 or len(items) < 2:
        return [function(item) for item in items]
    if (
        threading.current_thread() is not threading.main_thread()
        or _FUNCTION is not None
    ):
        raise RuntimeError("fork work must run in the main thread without nested pools")
    _FUNCTION = function
    enabled = gc.isenabled()
    pool = None
    try:
        # Avoid collection dirtying inherited object pages unnecessarily.
        gc.disable()
        with threadpool_limits(limits=1):
            pool = mp.get_context("fork").Pool(
                plan["workers"], initializer=_initialize_worker
            )
            original_workers = list(pool._pool)
            pending = pool.imap(_invoke, items, chunksize=1)
            result = []
            while len(result) < len(items):
                try:
                    result.append(pending.next(timeout=0.2))
                except mp.TimeoutError:
                    # Pool otherwise replaces a killed worker and waits forever
                    # for its lost result (e.g. after the Linux OOM killer).
                    if any(p.exitcode is not None for p in original_workers):
                        raise RuntimeError(
                            "A fitting worker exited unexpectedly; reduce --jobs and inspect container memory"
                        )
            return result
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
        _FUNCTION = None
        if enabled:
            gc.enable()
