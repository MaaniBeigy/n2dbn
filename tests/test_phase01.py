"""
tests/test_phase01.py
=====================
Core data structures and the synthetic generator.

Run with:
    pytest tests/test_phase01.py -v

Tests
-----
1.  Smoke     : generate() runs without error for every sanity-check scenario.
2.  RecErr    : RecErr_X == RecErr_Y == 0 (exact reconciliation invariant).
3.  Shapes    : every array has the shape implied by ScenarioConfig.
4.  Regression: same seed to byte-identical output (reproducibility).
5.  Binary    : finest-scale counts X^(1) ∈ {0, 1}.
6.  YX-compat : Y_{t,ℓ} == 0 wherever X_{t,ℓ} == 0 at the coarsest scale.
7.  Integrality: all X, Y arrays are non-negative integers.
8.  Family IV : misspecification variants run without error and satisfy RecErr=0.
9.  Split     : chronological split preserves reconciliation in each partition.
10. All families: Families I-III each generate valid data across a mini-grid.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.scales import ScaleHierarchy
from src.synthetic.families import make_generator
from src.synthetic.scenarios import ScenarioConfig, ScenarioGrid

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SANITY_GRID = ScenarioGrid.sanity_check()


def _generate(cfg: ScenarioConfig, seed: int = 42):
    """Generate one dataset from config; raises on any error."""
    gen = make_generator(cfg, seed=seed)
    return gen.generate(seed=seed)


# ---------------------------------------------------------------------------
# 1.  Smoke - every sanity-check scenario must run without error
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cfg", SANITY_GRID.scenarios, ids=lambda c: c.name)
def test_smoke_generate(cfg: ScenarioConfig):
    """N2DBNGenerator.generate() must complete without raising."""
    ds = _generate(cfg)
    assert ds is not None
    assert ds.data is not None
    assert ds.latents is not None
    assert ds.hierarchy is not None


# ---------------------------------------------------------------------------
# 2.  Reconciliation invariant - RecErr_X == RecErr_Y == 0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cfg", SANITY_GRID.scenarios, ids=lambda c: c.name)
def test_reconciliation_exact(cfg: ScenarioConfig):
    """
    Σ_{t∈ch_k(c)} X_t^(k,ℓ) == X_c^(k+1,ℓ)  and  same for Y,  for all k, c, ℓ.
    This is the central correctness invariant of the generator.
    """
    ds = _generate(cfg)
    errs = ds.data.check_reconciliation()
    assert (
        errs["RecErr_X"] == 0.0
    ), f"[{cfg.name}] RecErr_X = {errs['RecErr_X']:.6g}  (must be 0)"
    assert (
        errs["RecErr_Y"] == 0.0
    ), f"[{cfg.name}] RecErr_Y = {errs['RecErr_Y']:.6g}  (must be 0)"


# ---------------------------------------------------------------------------
# 3.  Shape invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cfg", SANITY_GRID.scenarios, ids=lambda c: c.name)
def test_array_shapes(cfg: ScenarioConfig):
    """All observed and latent arrays must have the shapes dictated by the config."""
    ds = _generate(cfg)
    hier = ds.hierarchy
    L = cfg.L

    for k in range(1, cfg.K + 1):
        H_k = hier.n_epochs(k)
        # Use the actual r from generated params: family factories may override r_k
        actual_r = ds.params[k - 1].r

        # Observed
        assert ds.data.get_X(k).shape == (
            H_k,
            L,
        ), f"[{cfg.name}] k={k}: X shape {ds.data.get_X(k).shape} != ({H_k},{L})"
        assert ds.data.get_Y(k).shape == (
            H_k,
            L,
        ), f"[{cfg.name}] k={k}: Y shape {ds.data.get_Y(k).shape} != ({H_k},{L})"
        # Latent
        assert ds.latents.get_S(k).shape == (
            H_k,
        ), f"[{cfg.name}] k={k}: S shape {ds.latents.get_S(k).shape} != ({H_k},)"
        assert ds.latents.get_H(k).shape == (
            H_k,
            actual_r,
        ), f"[{cfg.name}] k={k}: H shape {ds.latents.get_H(k).shape} != ({H_k},{actual_r})"


# ---------------------------------------------------------------------------
# 4.  Regression / reproducibility - same seed yields identical arrays
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cfg", SANITY_GRID.scenarios[:4], ids=lambda c: c.name)
def test_regression_reproducibility(cfg: ScenarioConfig):
    """Two calls with the same seed must return byte-identical X, Y, S, H arrays."""
    ds1 = _generate(cfg, seed=17)
    ds2 = _generate(cfg, seed=17)

    for k in range(1, cfg.K + 1):
        np.testing.assert_array_equal(
            ds1.data.get_X(k),
            ds2.data.get_X(k),
            err_msg=f"[{cfg.name}] k={k}: X not identical across runs",
        )
        np.testing.assert_array_equal(
            ds1.data.get_Y(k),
            ds2.data.get_Y(k),
            err_msg=f"[{cfg.name}] k={k}: Y not identical across runs",
        )
        np.testing.assert_array_equal(
            ds1.latents.get_S(k),
            ds2.latents.get_S(k),
            err_msg=f"[{cfg.name}] k={k}: S not identical across runs",
        )


# ---------------------------------------------------------------------------
# 5.  Finest-scale binary constraint: X^(1)_{t,ℓ} ∈ {0, 1}
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cfg", SANITY_GRID.scenarios, ids=lambda c: c.name)
def test_finest_scale_binary(cfg: ScenarioConfig):
    """
    A finest-scale slot either starts one episode of a label or none.

    The scenario sets `finest_binary`, so the allocator caps each epoch at the
    number of finest slots it contains and redistributes anything above it.
    """
    ds = _generate(cfg)
    X1 = ds.data.get_X(1)
    bad = X1[~np.isin(X1, [0, 1])]
    assert (
        bad.size == 0
    ), f"[{cfg.name}] finest X contains values outside {{0,1}}: {np.unique(bad)}"


@pytest.mark.parametrize("cfg", SANITY_GRID.scenarios[:4], ids=lambda c: c.name)
def test_started_episode_lasts_at_least_one_slot(cfg: ScenarioConfig):
    """Every reconstructed episode occupies at least the slot it starts in."""
    ds = _generate(cfg)
    lengths = [ep.duration for eps in ds.traces.values() for ep in eps]
    assert lengths, f"[{cfg.name}] no episode was reconstructed"
    assert min(lengths) >= 1, f"[{cfg.name}] an episode occupies no slot at all"


# ---------------------------------------------------------------------------
# 6.  Count-duration consistency at the coarsest scale
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cfg", SANITY_GRID.scenarios, ids=lambda c: c.name)
def test_coarsest_y_zero_when_x_zero(cfg: ScenarioConfig):
    """
    Y_t^(K,ℓ) must be 0 whenever X_t^(K,ℓ) == 0  (no episode → no duration).
    """
    ds = _generate(cfg)
    K = cfg.K
    X_K = ds.data.get_X(K)
    Y_K = ds.data.get_Y(K)

    violating = np.sum((X_K == 0) & (Y_K > 0))
    assert violating == 0, f"[{cfg.name}] k={K}: {violating} cells have Y>0 where X==0"


# ---------------------------------------------------------------------------
# 7.  Non-negativity and integrality of all observed arrays
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cfg", SANITY_GRID.scenarios, ids=lambda c: c.name)
def test_nonneg_integer_arrays(cfg: ScenarioConfig):
    """All X and Y arrays must be non-negative and have integer dtype."""
    ds = _generate(cfg)
    int_dtypes = {np.int32, np.int64, np.dtype("int32"), np.dtype("int64")}

    for k in range(1, cfg.K + 1):
        X = ds.data.get_X(k)
        Y = ds.data.get_Y(k)
        assert X.dtype in int_dtypes or np.issubdtype(
            X.dtype, np.integer
        ), f"[{cfg.name}] k={k}: X dtype = {X.dtype}"
        assert Y.dtype in int_dtypes or np.issubdtype(
            Y.dtype, np.integer
        ), f"[{cfg.name}] k={k}: Y dtype = {Y.dtype}"
        assert np.all(X >= 0), f"[{cfg.name}] k={k}: X has negatives"
        assert np.all(Y >= 0), f"[{cfg.name}] k={k}: Y has negatives"


# ---------------------------------------------------------------------------
# 8.  Regime values in range
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cfg", SANITY_GRID.scenarios, ids=lambda c: c.name)
def test_regime_range(cfg: ScenarioConfig):
    """S_t^(k) must be in {0, ..., M_k-1} at every scale."""
    ds = _generate(cfg)
    for k in range(1, cfg.K + 1):
        M_k = cfg.M_k[k - 1]
        S_k = ds.latents.get_S(k)
        assert np.all(S_k >= 0), f"[{cfg.name}] k={k}: S contains negatives"
        assert np.all(S_k < M_k), f"[{cfg.name}] k={k}: S contains values >= M_k={M_k}"


# ---------------------------------------------------------------------------
# 9.  Family IV - misspecification variants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("misspec_type", [None, "heavier_tail", "omit_context"])
def test_family_iv_smoke_and_recon(misspec_type):
    """Family IV generators must produce valid data with RecErr == 0."""
    cfg = ScenarioConfig(
        name=f"fiv_{misspec_type}",
        K=3,
        H_finest=168,
        branching=[24, 7],
        L=3,
        M_k=[2, 2, 2],
        r_k=[0, 0, 0],
        max_lag=[1, 1, 0],
        max_indegree=[1, 1, 0],
        zi_level="medium",
        sparse_graph=True,
        family=4,
        misspec_type=misspec_type,
        seed=0,
        n_replicates=1,
    )
    gen = make_generator(cfg, seed=0)
    ds = gen.generate(seed=0)
    errs = ds.data.check_reconciliation()
    assert errs["RecErr_X"] == 0.0, f"Family IV ({misspec_type}): RecErr_X != 0"
    assert errs["RecErr_Y"] == 0.0, f"Family IV ({misspec_type}): RecErr_Y != 0"


# ---------------------------------------------------------------------------
# 10.  Chronological split preserves reconciliation
# ---------------------------------------------------------------------------


def test_split_preserves_reconciliation():
    """Each of train/valid/test partitions must independently satisfy RecErr==0.

    The hierarchy must leave the coarsest scale with enough epochs to split:
    with the old 168/[24,7] setting there was exactly one coarsest epoch, so
    two of the three partitions came back empty and the assertion was vacuous.
    """
    cfg = ScenarioConfig(
        name="split_test",
        K=3,
        H_finest=720,
        branching=[6, 6],
        L=3,
        M_k=[1, 1, 1],
        r_k=[0, 0, 0],
        max_lag=[1, 1, 0],
        max_indegree=[1, 1, 0],
        zi_level="low",
        sparse_graph=True,
        family=1,
        seed=99,
        n_replicates=1,
    )
    gen = make_generator(cfg, seed=99)
    ds = gen.generate(seed=99)

    train, valid, test = ds.data.split()
    for name, split in [("train", train), ("valid", valid), ("test", test)]:
        errs = split.check_reconciliation()
        assert errs["RecErr_X"] == 0.0, f"{name}: RecErr_X = {errs['RecErr_X']}"
        assert errs["RecErr_Y"] == 0.0, f"{name}: RecErr_Y = {errs['RecErr_Y']}"


# ---------------------------------------------------------------------------
# 11.  All three non-misspec families smoke + recon
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "family,M,r",
    [
        (1, 1, 0),  # stationary
        (2, 2, 0),  # regime-switching
        (3, 1, 1),  # latent factor
    ],
)
def test_all_families_recon(family, M, r):
    """Families I, II, III must each satisfy the reconciliation invariant."""
    cfg = ScenarioConfig(
        name=f"family{family}_M{M}_r{r}",
        K=3,
        H_finest=168,
        branching=[24, 7],
        L=4,
        M_k=[M, M, M],
        r_k=[r, r, r],
        max_lag=[2, 1, 0],
        max_indegree=[1, 1, 0],
        zi_level="medium",
        sparse_graph=True,
        family=family,
        seed=42,
        n_replicates=1,
    )
    gen = make_generator(cfg, seed=42)
    ds = gen.generate(seed=42)
    errs = ds.data.check_reconciliation()
    assert errs["RecErr_X"] == 0.0, f"Family {family}: RecErr_X = {errs['RecErr_X']}"
    assert errs["RecErr_Y"] == 0.0, f"Family {family}: RecErr_Y = {errs['RecErr_Y']}"


# ---------------------------------------------------------------------------
# TODO-4 gap-filling: ScaleHierarchy.validate and from_explicit_maps
# ---------------------------------------------------------------------------


def test_scales_validate_passes():
    """validate() must pass without error on a correctly built uniform branching hierarchy."""
    hier = ScaleHierarchy.from_uniform_branching(12, branching_factors=[3])
    # Should not raise
    hier.validate()
    assert hier.K == 2
    assert hier.n_epochs(1) == 12
    assert hier.n_epochs(2) == 4


def test_scales_from_explicit_maps():
    """from_explicit_maps roundtrips correctly and satisfies validate()."""
    H = [12, 4, 1]
    pm0 = np.arange(12) // 3  # scale-1 (12 epochs) to scale-2 (4 epochs)
    pm1 = np.zeros(4, dtype=np.int32)  # scale-2 (4 epochs) to scale-3 (1 epoch)
    parent_maps = [pm0, pm1, None]

    hier = ScaleHierarchy.from_explicit_maps(H=H, parent_maps=parent_maps)
    assert hier.K == 3
    assert hier.n_epochs(1) == 12
    assert hier.n_epochs(2) == 4
    assert hier.n_epochs(3) == 1
    # parent(k=1, t=9) should map to epoch 9//3 = 3
    assert hier.parent(1, 9) == 3
    hier.validate()


# ---------------------------------------------------------------------------
# TODO-4 gap-filling: ScenarioConfig.H_k property and ScenarioGrid.filter
# ---------------------------------------------------------------------------


def test_scenario_config_H_k_property():
    """H_k property must derive epoch counts matching H_finest and branching_factors."""
    cfg = ScenarioConfig(
        name="hk_test",
        K=3,
        H_finest=12,
        branching=[3, 2],
        L=2,
        M_k=[1, 1, 1],
        r_k=[0, 0, 0],
        max_lag=[1, 1, 1],
        max_indegree=[0, 0, 0],
        zi_level="low",
        sparse_graph=False,
        family=1,
        seed=0,
    )
    H_k = cfg.H_k
    assert len(H_k) == 3
    assert H_k[0] == 12  # finest  (k=1)
    assert H_k[1] == 4  # 12 // 3 (k=2)
    assert H_k[2] == 2  # 4  // 2 (k=3, coarsest)


def test_scenario_grid_filter():
    """ScenarioGrid.filter must return the correct sub-grid matching keyword conditions."""
    grid = ScenarioGrid.sanity_check()
    # Filter to scenarios with L=6 only
    sub = grid.filter(L=6)
    assert len(sub) > 0, "sanity_check grid must contain L=6 scenarios"
    for cfg in sub:
        assert cfg.L == 6, f"Expected L=6, got L={cfg.L} in {cfg.name}"

    # Filter to non-existent combination - must return empty grid
    empty = grid.filter(L=999)
    assert len(empty) == 0


def test_full_benchmark_branching_valid():
    """All full_benchmark scenarios must have valid branching and unique names.

    Regression tests:
    - branching=[24, 7] was incompatible with H_finest=1440/14400.
    - duplicate names made summary_table aggregate incompatible results.
    """
    grid = ScenarioGrid.full_benchmark()
    assert 15 <= len(grid) <= 50, f"Expected 15-50 targeted scenarios, got {len(grid)}"

    names = []
    for cfg in grid:
        # Branching divisibility
        H = cfg.H_finest
        for bf in cfg.branching:
            assert (
                H % bf == 0
            ), f"Scenario {cfg.name}: H={H} not divisible by branching factor {bf}"
            H = H // bf
        assert H >= 1, f"Scenario {cfg.name}: coarsest H={H} < 1"
        names.append(cfg.name)

    # All names must be unique
    assert len(names) == len(
        set(names)
    ), f"Duplicate scenario names: {[n for n in names if names.count(n) > 1]}"
