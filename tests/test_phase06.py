"""
tests/test_phase06.py
=====================
The TopDownSEM outer loop, integrating the three learning steps.

Tests
-----
1.  Smoke         - TopDownSEM.fit completes without error
2.  Return types  - (params_list, graph, history) have correct types
3.  ELBO          - history.elbo is a non-empty list of finite floats
4.  Params count  - len(params_list) == K
5.  Graph type    - graph is MultiScaleDBNGraph
6.  Convergence   - history.converged() returns a bool
7.  Checkpoint    - _save/_load_checkpoint roundtrip preserves ELBO
8.  ELBO monotone - ELBO does not catastrophically decrease between iterations
9.  Regime-switching - Family II (M=2) fits without error
10. Factor model  - Family III (r>0) fits without error

The tests are intentionally kept small (H_finest=12, 2 outer iterations,
no structure learning) to ensure they run quickly.

Run with:
    pytest tests/test_phase06.py -v
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from src.core.graph import MultiScaleDBNGraph
from src.learning.structural_em import (
    TopDownSEM,
    TrainingHistory,
    _load_checkpoint,
    _save_checkpoint,
)
from src.synthetic.families import make_generator
from src.synthetic.scenarios import ScenarioConfig

# ---------------------------------------------------------------------------
# Shared scenario configs and helpers
# ---------------------------------------------------------------------------

_SEM_KWARGS = dict(
    max_outer_iter=2,
    do_structure=False,  # skip S-step for speed
    do_warmstart=False,
    verbose=0,
    estep_kwargs={"max_inner_iter": 2, "use_laplace": False},
)


def _run_sem(cfg: ScenarioConfig) -> tuple:
    """Generate data from cfg, run SEM, return (ds, params_list, graph, history)."""
    gen = make_generator(cfg, seed=cfg.seed)
    ds = gen.generate(seed=cfg.seed)
    graph = MultiScaleDBNGraph(hierarchy=ds.hierarchy, label_names=gen.label_names)
    sem = TopDownSEM(**_SEM_KWARGS)
    params_list, graph_out, history = sem.fit(ds.data, ds.params, graph)
    return ds, params_list, graph_out, history


# Minimal Family-I scenario: stationary, K=2, r=0, M=1
_CFG_FAMILY_I = ScenarioConfig(
    name="sem_mini_fam1",
    K=2,
    H_finest=12,
    branching=[3],
    L=2,
    M_k=[1, 1],
    r_k=[0, 0],
    max_lag=[1, 1],
    max_indegree=[0, 0],
    zi_level="low",
    sparse_graph=False,
    family=1,
    seed=10,
)

# Family-II: regime-switching, M=2
_CFG_FAMILY_II = ScenarioConfig(
    name="sem_mini_fam2",
    K=2,
    H_finest=12,
    branching=[3],
    L=2,
    M_k=[2, 2],
    r_k=[0, 0],
    max_lag=[1, 1],
    max_indegree=[0, 0],
    zi_level="low",
    sparse_graph=False,
    family=2,
    seed=20,
)

# Family-III: latent factor, r=1
_CFG_FAMILY_III = ScenarioConfig(
    name="sem_mini_fam3",
    K=2,
    H_finest=12,
    branching=[3],
    L=2,
    M_k=[1, 1],
    r_k=[1, 1],
    max_lag=[1, 1],
    max_indegree=[0, 0],
    zi_level="low",
    sparse_graph=False,
    family=3,
    seed=30,
)


# ---------------------------------------------------------------------------
# Module-level fixture: run SEM once for Family I
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fam1_result():
    return _run_sem(_CFG_FAMILY_I)


# ---------------------------------------------------------------------------
# 1.  Smoke
# ---------------------------------------------------------------------------


def test_sem_smoke(fam1_result) -> None:
    """TopDownSEM.fit must complete without raising."""
    ds, params_list, graph_out, history = fam1_result
    assert params_list is not None
    assert graph_out is not None
    assert history is not None


# ---------------------------------------------------------------------------
# 2.  Return types
# ---------------------------------------------------------------------------


def test_sem_return_types(fam1_result) -> None:
    _, params_list, graph_out, history = fam1_result
    assert isinstance(params_list, list)
    assert isinstance(graph_out, MultiScaleDBNGraph)
    assert isinstance(history, TrainingHistory)


# ---------------------------------------------------------------------------
# 3.  ELBO list is non-empty and contains finite values
# ---------------------------------------------------------------------------


def test_sem_elbo_finite(fam1_result) -> None:
    _, _, _, history = fam1_result
    assert len(history.elbo) > 0, "history.elbo must be non-empty"
    for i, elbo in enumerate(history.elbo):
        assert np.isfinite(elbo), f"ELBO[{i}] = {elbo} is not finite"


# ---------------------------------------------------------------------------
# 4.  Params count
# ---------------------------------------------------------------------------


def test_sem_params_count(fam1_result) -> None:
    ds, params_list, _, _ = fam1_result
    assert (
        len(params_list) == ds.data.K
    ), f"Expected {ds.data.K} param blocks, got {len(params_list)}"


# ---------------------------------------------------------------------------
# 5.  Graph is MultiScaleDBNGraph
# ---------------------------------------------------------------------------


def test_sem_graph_is_correct_type(fam1_result) -> None:
    _, _, graph_out, _ = fam1_result
    assert isinstance(graph_out, MultiScaleDBNGraph)


# ---------------------------------------------------------------------------
# 6.  Convergence flag returns bool
# ---------------------------------------------------------------------------


def test_sem_converged_returns_bool(fam1_result) -> None:
    _, _, _, history = fam1_result
    result = history.converged()
    assert isinstance(result, bool)


def test_sem_converged_strict_single_iter() -> None:
    """converged() must return False when there is only 1 ELBO entry."""
    h = TrainingHistory()
    h.elbo.append(-100.0)
    assert h.converged() is False


def test_sem_converged_flat_history() -> None:
    """converged() must return True once last n_check deltas are all < tol."""
    h = TrainingHistory()
    for v in [-100.0, -100.0, -100.0, -100.0]:
        h.elbo.append(v)
    assert h.converged(tol=1e-3, n_check=3) is True


# ---------------------------------------------------------------------------
# 7.  Checkpoint save / load roundtrip
# ---------------------------------------------------------------------------


def test_checkpoint_roundtrip(fam1_result) -> None:
    """Saved checkpoint must reload with identical ELBO history."""
    _, params_list, graph_out, history = fam1_result
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "cp.pkl")
        _save_checkpoint(path, params_list, graph_out, history, outer_iter=1)
        state = _load_checkpoint(path)

    assert state["outer_iter"] == 1
    assert state["history"].elbo == history.elbo
    assert len(state["params_list"]) == len(params_list)


# ---------------------------------------------------------------------------
# 8.  ELBO does not catastrophically decrease
# ---------------------------------------------------------------------------


def test_sem_elbo_no_catastrophic_drop(fam1_result) -> None:
    """ELBO should not drop by more than 50× the initial magnitude between steps."""
    _, _, _, history = fam1_result
    if len(history.elbo) < 2:
        pytest.skip("Need at least 2 ELBO values")
    for i in range(1, len(history.elbo)):
        prev, curr = history.elbo[i - 1], history.elbo[i]
        drop = prev - curr  # positive = got worse
        # Allow up to 1000 absolute units of degradation on a small problem
        assert (
            drop < 1000.0
        ), f"ELBO dropped by {drop:.1f} from iter {i-1} ({prev:.1f}) to iter {i} ({curr:.1f})"


# ---------------------------------------------------------------------------
# 9.  Regime-switching model (Family II, M=2)
# ---------------------------------------------------------------------------


def test_sem_family_ii_smoke() -> None:
    """SEM must run on a regime-switching (M=2) scenario without error."""
    ds, params_list, graph_out, history = _run_sem(_CFG_FAMILY_II)
    assert len(params_list) == ds.data.K
    assert all(np.isfinite(e) for e in history.elbo)


# ---------------------------------------------------------------------------
# 10.  Latent factor model (Family III, r>0)
# ---------------------------------------------------------------------------


def test_sem_family_iii_smoke() -> None:
    """SEM must run on a latent-factor (r=1) scenario without error."""
    ds, params_list, graph_out, history = _run_sem(_CFG_FAMILY_III)
    assert len(params_list) == ds.data.K
    assert all(np.isfinite(e) for e in history.elbo)
    # Factor dimension must be preserved
    for k_i, p in enumerate(params_list):
        expected_r = _CFG_FAMILY_III.r_k[k_i]
        assert p.r == expected_r, f"scale {k_i+1}: expected r={expected_r}, got r={p.r}"


# ---------------------------------------------------------------------------
# TODO-4 gap-filling: Structure learning and checkpoint resume
# ---------------------------------------------------------------------------


def test_sem_with_structure_learning() -> None:
    """TopDownSEM with do_structure=True must complete without error."""
    cfg = ScenarioConfig(
        name="sem_struct",
        K=2,
        H_finest=12,
        branching=[3],
        L=2,
        M_k=[1, 1],
        r_k=[0, 0],
        max_lag=[1, 1],
        max_indegree=[1, 1],
        zi_level="low",
        sparse_graph=False,
        family=1,
        seed=40,
    )
    gen = make_generator(cfg, seed=40)
    ds = gen.generate(seed=40)
    graph = MultiScaleDBNGraph(hierarchy=ds.hierarchy, label_names=gen.label_names)

    sem = TopDownSEM(
        max_outer_iter=1,
        do_structure=True,
        do_warmstart=False,
        verbose=0,
        estep_kwargs={"max_inner_iter": 2, "use_laplace": False},
    )
    params_list, graph_out, history = sem.fit(ds.data, ds.params, graph)

    assert isinstance(params_list, list)
    assert len(params_list) == ds.data.K
    assert isinstance(graph_out, MultiScaleDBNGraph)
    assert len(history.elbo) > 0
    assert all(np.isfinite(e) for e in history.elbo)


def test_sem_resume_from_checkpoint() -> None:
    """TopDownSEM.fit with resume_from= must resume from a saved checkpoint."""
    # Run 1 outer iteration and save a checkpoint manually
    ds, params1, graph1, hist1 = _run_sem(_CFG_FAMILY_I)

    with tempfile.TemporaryDirectory() as tmpdir:
        cp_path = os.path.join(tmpdir, "cp.pkl")
        _save_checkpoint(cp_path, params1, graph1, hist1, outer_iter=0)

        # Resume: max_outer_iter=3 with start_iter=1 to runs 2 more iterations
        gen = make_generator(_CFG_FAMILY_I, seed=_CFG_FAMILY_I.seed)
        ds2 = gen.generate(seed=_CFG_FAMILY_I.seed)
        graph2 = MultiScaleDBNGraph(
            hierarchy=ds2.hierarchy, label_names=gen.label_names
        )

        sem2 = TopDownSEM(
            max_outer_iter=3,
            do_structure=False,
            do_warmstart=False,
            verbose=0,
            estep_kwargs={"max_inner_iter": 2, "use_laplace": False},
        )
        params2, graph_out2, hist2 = sem2.fit(
            ds2.data, ds2.params, graph2, resume_from=cp_path
        )

    assert isinstance(params2, list)
    assert len(params2) == ds2.data.K
    # Resumed run should have at least as many ELBO entries as the checkpoint's history
    assert len(hist2.elbo) >= len(hist1.elbo)
    assert all(np.isfinite(e) for e in hist2.elbo)
