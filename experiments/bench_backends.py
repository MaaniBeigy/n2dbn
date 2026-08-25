"""
experiments/bench_backends.py
=============================
Where the GPU starts to pay for the DYNOTEARS solver, measured rather than
assumed.

The solver is the one dense-linear-algebra kernel in the repository, so it is
the only place a GPU could help.  Everything else is a long sequence of small
array operations over `(epoch, regime, label)` interleaved with Python control
flow, where the launch overhead of a device call exceeds the work it carries.

The design of the solver at scale `k` has `T_k - max_lag` rows and
`L * max_lag` columns, which for the benchmark grid is about 14000 by 30.  The
table this script prints says what that size costs on each backend, and how much
larger a study would have to be before the GPU changes an answer.

Run
---
    python -m experiments.bench_backends [--reps 3]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np

from src.learning.classical import (
    _design,
    _fista,
    _fista_torch,
    _torch_device_available,
)
from src.utils.resources import describe

# (finest epochs, labels, max lag).  The first row is the benchmark grid.
SIZES = (
    (14100, 10, 3),
    (20000, 20, 4),
    (100000, 40, 5),
    (200000, 60, 5),
    (400000, 80, 5),
)


def _time(fn, reps: int, sync=None) -> float:
    """Best of `reps` wall-clock times, in seconds."""
    best = float("inf")
    for _ in range(reps):
        if sync is not None:
            sync()
        t0 = time.time()
        fn()
        if sync is not None:
            sync()
        best = min(best, time.time() - t0)
    return best


def main(reps: int = 3, lam: float = 0.05, max_iter: int = 500) -> None:
    print(describe())
    has_torch = _torch_device_available("cpu")
    has_cuda = _torch_device_available("cuda")
    sync = None
    if has_cuda:
        import torch

        sync = torch.cuda.synchronize

    rng = np.random.default_rng(0)

    # One small solve on each backend first, so the table times the arithmetic
    # rather than the one-off cost of loading a library or a device context.
    warm = _design(rng.poisson(0.3, size=(500, 4)), 2)
    _fista(warm[0], warm[1], lam, 10, 1e-8)
    if has_torch:
        _fista_torch(warm[0], warm[1], lam, 10, 1e-8, "cpu")
    if has_cuda:
        _fista_torch(warm[0], warm[1], lam, 10, 1e-8, "cuda")

    header = f"{'design':>18} {'numpy':>10} {'torch cpu':>10} {'torch cuda':>11} {'ratio':>8}"
    print("\n" + header)
    print("-" * len(header))
    for T, L, max_lag in SIZES:
        X = rng.poisson(0.3, size=(T, L))
        W, Y, _ = _design(X, max_lag)
        t_np = _time(lambda: _fista(W, Y, lam, max_iter, 1e-8), reps)
        t_cpu = (
            _time(lambda: _fista_torch(W, Y, lam, max_iter, 1e-8, "cpu"), reps)
            if has_torch
            else float("nan")
        )
        t_gpu = (
            _time(lambda: _fista_torch(W, Y, lam, max_iter, 1e-8, "cuda"), reps, sync)
            if has_cuda
            else float("nan")
        )
        ratio = t_np / t_gpu if has_cuda and t_gpu > 0 else float("nan")
        print(
            f"{str(W.shape):>18} {t_np:10.3f} {t_cpu:10.3f} {t_gpu:11.3f} {ratio:7.1f}x"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=3)
    args = parser.parse_args()
    main(reps=args.reps)
