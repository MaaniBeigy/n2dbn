"""
The paths that only run when something is missing or degenerate.

Eight statements of the library are `except ImportError` guards for matplotlib
and pyAgrum, and two more are reachable only from a starting point the library
itself never produces.  Each is exercised here by removing the package for the
length of one reload, or by handing the code the degenerate input directly, so
that no statement of `src` is left unmeasured.
"""

from __future__ import annotations

import builtins
import importlib
import sys
from contextlib import contextmanager

import numpy as np
import pytest


@contextmanager
def _without(*packages: str):
    """
    Reload the caller's modules with `packages` unimportable, then restore.

    Blocking an import is not enough on its own, because a package already in
    `sys.modules` never reaches the import hook, so each name is removed for the
    length of the block and put back afterwards.
    """
    real_import = builtins.__import__
    saved = {
        name: sys.modules.pop(name)
        for name in list(sys.modules)
        if any(name == p or name.startswith(p + ".") for p in packages)
    }

    def blocked(name, *args, **kwargs):
        if any(name == p or name.startswith(p + ".") for p in packages):
            raise ImportError(f"{name} is unavailable")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = blocked
    try:
        yield
    finally:
        builtins.__import__ = real_import
        sys.modules.update(saved)


def _reload(name: str):
    return importlib.reload(importlib.import_module(name))


# ---------------------------------------------------------------------------
# pyAgrum
# ---------------------------------------------------------------------------


def test_the_graph_module_reports_pyagrum_as_absent():
    module = "src.core.graph"
    try:
        with _without("pyagrum"):
            reloaded = _reload(module)
            assert reloaded._PYAGRUM_AVAILABLE is False
            with pytest.raises(ImportError, match="requires pyAgrum"):
                reloaded._require_pyagrum("to_pyagrum_bn")
    finally:
        assert _reload(module)._PYAGRUM_AVAILABLE is True


def test_the_scoring_backend_reports_pyagrum_as_absent():
    module = "src.learning.pyagrum_scoring"
    try:
        with _without("pyagrum"):
            reloaded = _reload(module)
            assert reloaded._PYAGRUM_AVAILABLE is False
            with pytest.raises(ImportError, match="requires pyAgrum"):
                reloaded._require_pyagrum("PyAgrumScore")
    finally:
        assert _reload(module)._PYAGRUM_AVAILABLE is True


# ---------------------------------------------------------------------------
# matplotlib and pyAgrum together
# ---------------------------------------------------------------------------


def test_the_plot_module_reports_both_optional_packages_as_absent():
    module = "src.utils.visualization"
    try:
        with _without("matplotlib", "pyagrum"):
            reloaded = _reload(module)
            assert reloaded._MPL_AVAILABLE is False
            assert reloaded._PYAGRUM_AVAILABLE is False
            with pytest.raises(ImportError, match="matplotlib"):
                reloaded._require_mpl("plot_elbo_history")
    finally:
        restored = _reload(module)
        assert restored._MPL_AVAILABLE is True
        assert restored._PYAGRUM_AVAILABLE is True


# ---------------------------------------------------------------------------
# Beam search at the in-degree bound
# ---------------------------------------------------------------------------


def test_the_beam_skips_a_parent_set_that_is_already_full():
    """
    With no room for a first parent, the beam has nothing to extend.

    `SStep` never sets `max_depth` below `max_indegree`, so this branch is
    unreachable through it; a caller using `BeamSearch` directly can reach it.
    """
    from src.learning.sstep import BeamSearch

    seen = []

    def score_fn(parent_set):
        seen.append(frozenset(parent_set))
        return float(len(parent_set))

    best = BeamSearch(beam_width=2, max_indegree=0, max_depth=2).search(
        score_fn, [(0, 1), (1, 1)]
    )
    assert best == set()
    assert seen == [frozenset()]


# ---------------------------------------------------------------------------
# A parent epoch with no children
# ---------------------------------------------------------------------------


def test_a_parent_block_with_no_children_is_skipped():
    """
    Uniform branching never leaves a coarse epoch childless, an irregular
    nesting can, and the allocation pass has to step over it rather than divide
    a total among nobody.
    """
    from src.core.scales import ScaleHierarchy
    from src.synthetic.families import make_family_I
    from src.synthetic.generator import N2DBNGenerator
    from src.synthetic.scenarios import ScenarioConfig

    cfg = ScenarioConfig(
        name="childless",
        K=2,
        H_finest=4,
        branching=[2],
        L=2,
        M_k=[1, 1],
        r_k=[0, 0],
        max_lag=[1, 1],
        max_indegree=[0, 0],
    )
    reference = make_family_I(cfg, seed=0)

    # Parent 1 has no children; parents 0 and 2 have two each.
    hier = ScaleHierarchy(
        K=2,
        H=[4, 3],
        parent_maps=[np.array([0, 0, 2, 2], dtype=np.int32), None],
    )
    gen = N2DBNGenerator(
        params_list=reference.params_list,
        hierarchy=hier,
        label_names=reference.label_names,
    )
    ds = gen.generate(seed=0)

    assert hier.children(1, 1) == []
    assert ds.data.get_X(2)[1].sum() >= 0
    assert ds.data.check_reconciliation() == {"RecErr_X": 0.0, "RecErr_Y": 0.0}
