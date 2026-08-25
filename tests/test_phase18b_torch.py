"""
The torch solver of the DYNOTEARS learner.

The whole point of a second backend is that it computes the same thing, so the
assertions are equality against the numpy solver rather than a fresh check of
the mathematics.  The CUDA test runs only when a card is visible; the routing
test does not need one, because it forces the torch path onto the cpu device.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.learning import classical as C

torch = pytest.importorskip("torch")

_HAS_CUDA = torch.cuda.is_available()

TRUE_EDGES = {0: {(1, 1)}, 1: set(), 2: {(0, 2)}, 3: set()}


@pytest.fixture(scope="module")
def series():
    rng = np.random.default_rng(0)
    T, L = 1500, 4
    X = np.zeros((T, L), dtype=np.int64)
    for t in range(2, T):
        X[t, 0] = rng.random() < 0.10 + 0.60 * (X[t - 1, 1] > 0)
        X[t, 1] = rng.random() < 0.25
        X[t, 2] = rng.random() < 0.10 + 0.60 * (X[t - 2, 0] > 0)
        X[t, 3] = rng.random() < 0.25
    return X


def test_a_cpu_device_is_available_whenever_torch_is():
    assert C._torch_device_available("cpu") is True


def test_a_cuda_device_is_available_only_when_a_card_is():
    assert C._torch_device_available("cuda") is _HAS_CUDA


def test_the_torch_solver_matches_the_numpy_solver():
    rng = np.random.default_rng(4)
    W = rng.normal(size=(500, 6))
    Y = W @ rng.normal(size=(6, 3)) + 0.1 * rng.normal(size=(500, 3))
    on_numpy = C._fista(W, Y, lam=0.05, max_iter=400, tol=1e-12)
    on_torch = C._fista_torch(W, Y, lam=0.05, max_iter=400, tol=1e-12, device="cpu")
    assert np.allclose(on_numpy, on_torch, atol=1e-10)


def test_the_torch_solver_stops_on_the_tolerance():
    """A tolerance above the first step ends the loop after one iteration."""
    rng = np.random.default_rng(5)
    W = rng.normal(size=(120, 3))
    Y = rng.normal(size=(120, 2))
    got = C._fista_torch(W, Y, lam=0.0, max_iter=100, tol=1e9, device="cpu")
    assert got.shape == (3, 2)


def test_dynotears_routes_to_torch_when_the_device_is_usable(series, monkeypatch):
    """Forced onto the torch path, on the cpu device, so no card is needed."""
    monkeypatch.setattr(C, "_use_torch", lambda _d: True)
    assert C._learn_dynotears([series], 2, 2, device="cpu") == TRUE_EDGES


def test_a_named_device_routes_to_torch_when_torch_can_serve_it():
    assert C._use_torch("cuda") is _HAS_CUDA
    assert C._use_torch("meta") is True


@pytest.mark.skipif(not _HAS_CUDA, reason="no CUDA device visible")
def test_dynotears_on_the_gpu_finds_the_same_graph(series):
    assert C._learn_dynotears([series], 2, 2, device="cuda") == TRUE_EDGES


@pytest.mark.skipif(not _HAS_CUDA, reason="no CUDA device visible")
def test_the_gpu_solver_agrees_with_the_numpy_solver():
    rng = np.random.default_rng(6)
    W = rng.normal(size=(2000, 12))
    Y = W @ rng.normal(size=(12, 4)) + 0.1 * rng.normal(size=(2000, 4))
    on_numpy = C._fista(W, Y, lam=0.05, max_iter=300, tol=1e-12)
    on_gpu = C._fista_torch(W, Y, lam=0.05, max_iter=300, tol=1e-12, device="cuda")
    assert np.allclose(on_numpy, on_gpu, atol=1e-9)
