"""
utils/resources.py
==================
What the machine has, and how much of it one run is allowed to use.

Three questions are answered here.

* How many worker processes may a replicated study start?  Cores alone is the
  wrong answer: every worker holds its own copy of the dataset, and a study that
  asks for one worker per core on a machine with little memory per core dies
  inside a worker with an import error that reads like a code fault.
  `resolve_jobs` bounds the count by memory as well as by cores.
* How many threads may each worker's linear algebra use?  A pool of `n` workers
  each starting `n_cpu` BLAS threads oversubscribes the machine by a factor of
  `n`, which is slower than one thread per worker.  `configure_threads` pins it.
* Is there a usable GPU, and which array library can reach it?  `gpu_devices`
  reports what is visible without importing any GPU library.

Nothing here changes a result.  Worker count and thread count move wall-clock
time only, which the protocol tests assert directly.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Environment variables every common BLAS reads for its thread count.
_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

_BYTES_PER_GB = 1024.0**3


# ---------------------------------------------------------------------------
# Description of the machine
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GpuInfo:
    """One visible GPU, as reported by `nvidia-smi`."""

    index: int
    name: str
    total_memory_gb: float
    free_memory_gb: float


@dataclass(frozen=True)
class MachineResources:
    """Cores, memory and GPUs visible to this process."""

    n_cpu: int
    total_memory_gb: float
    available_memory_gb: float
    gpus: List[GpuInfo] = field(default_factory=list)

    @property
    def has_gpu(self) -> bool:
        return len(self.gpus) > 0


def _read_meminfo(path: str = "/proc/meminfo") -> Dict[str, float]:
    """Memory totals in GB keyed by the `/proc/meminfo` field name."""
    out: Dict[str, float] = {}
    try:
        with open(path) as fh:
            for line in fh:
                name, _, rest = line.partition(":")
                parts = rest.split()
                if parts:
                    out[name] = float(parts[0]) * 1024.0 / _BYTES_PER_GB
    except OSError:
        return {}
    return out


def _memory_gb() -> tuple:
    """Total and available system memory in GB, both 0.0 when unknown."""
    info = _read_meminfo()
    if info:
        total = info.get("MemTotal", 0.0)
        avail = info.get("MemAvailable", info.get("MemFree", 0.0))
        return total, avail
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        return 0.0, 0.0
    total = pages * page_size / _BYTES_PER_GB
    return total, total


def gpu_devices(timeout_s: float = 5.0) -> List[GpuInfo]:
    """Visible GPUs, empty when `nvidia-smi` is absent or reports nothing."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return []
    query = "index,name,memory.total,memory.free"
    try:
        raw = subprocess.run(
            [exe, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if raw.returncode != 0:
        return []

    devices: List[GpuInfo] = []
    for line in raw.stdout.splitlines():
        fields = [f.strip() for f in line.split(",")]
        if len(fields) != 4:
            continue
        try:
            devices.append(
                GpuInfo(
                    index=int(fields[0]),
                    name=fields[1],
                    total_memory_gb=float(fields[2]) / 1024.0,
                    free_memory_gb=float(fields[3]) / 1024.0,
                )
            )
        except ValueError:
            continue
    return devices


def detect() -> MachineResources:
    """Cores, memory and GPUs of the machine this process runs on."""
    total, avail = _memory_gb()
    return MachineResources(
        n_cpu=os.cpu_count() or 1,
        total_memory_gb=round(total, 2),
        available_memory_gb=round(avail, 2),
        gpus=gpu_devices(),
    )


# ---------------------------------------------------------------------------
# Budgets
# ---------------------------------------------------------------------------


def resolve_jobs(
    n_jobs: int,
    memory_per_job_gb: float = 0.0,
    resources: Optional[MachineResources] = None,
    reserve_gb: float = 1.0,
) -> int:
    """
    Number of worker processes to start.

    Args:
        n_jobs: requested count.  1 runs serially, a negative value asks for one
            worker per core less one, and a positive value is a ceiling.
        memory_per_job_gb: resident memory one worker needs.  0 disables the
            memory bound.
        resources: machine description, detected when omitted.
        reserve_gb: memory left for the parent process and the operating system.

    Returns:
        At least 1, at most the number of cores, and at most what the available
        memory holds once `reserve_gb` is set aside.
    """
    res = resources or detect()
    if n_jobs is None or n_jobs == 1:
        return 1

    want = max(1, res.n_cpu - 1) if n_jobs < 0 else min(int(n_jobs), res.n_cpu)

    if memory_per_job_gb > 0 and res.available_memory_gb > 0:
        budget = max(0.0, res.available_memory_gb - reserve_gb)
        by_memory = int(budget // memory_per_job_gb)
        want = min(want, max(1, by_memory))
    return max(1, want)


def configure_threads(n_threads: int = 1) -> Dict[str, str]:
    """
    Pin every BLAS thread-count variable and return what was set.

    A variable already present in the environment is left alone, so an explicit
    setting by the caller still wins.  Numpy reads these at import time, so this
    has an effect only when called before numpy is imported, which is what the
    worker start-up path does.
    """
    value = str(max(1, int(n_threads)))
    applied: Dict[str, str] = {}
    for var in _THREAD_VARS:
        if var not in os.environ:
            os.environ[var] = value
            applied[var] = value
    return applied


def describe(resources: Optional[MachineResources] = None) -> str:
    """One block of text naming cores, memory and every visible GPU."""
    res = resources or detect()
    lines = [
        f"cpu     : {res.n_cpu} logical processors",
        f"memory  : {res.total_memory_gb:.1f} GB total, "
        f"{res.available_memory_gb:.1f} GB available",
    ]
    if not res.gpus:
        lines.append("gpu     : none visible")
    for g in res.gpus:
        lines.append(
            f"gpu {g.index}   : {g.name}, {g.total_memory_gb:.1f} GB total, "
            f"{g.free_memory_gb:.1f} GB free"
        )
    lines.append(
        f"jobs    : {resolve_jobs(-1, resources=res)} workers at one dataset per "
        f"worker, {resolve_jobs(-1, memory_per_job_gb=1.0, resources=res)} at 1 GB each"
    )
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - a command, not an import
    print(describe())
