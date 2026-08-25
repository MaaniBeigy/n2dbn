"""
True edges at every scale the data can carry, and the trace a fit maps back to.

The grid used to place every true edge at the middle scale, so the other two
scored a method for staying silent rather than for recovering anything. The
coarsest scale is now measured to carry an edge at least as well as the middle
one, so it holds edges; the finest is measured not to and stays empty. Recall is
reported separately for edges into a label from itself and from another label,
because a model can be blind to one kind and a pooled recall hides it.

1.  every scenario declares a nonzero in-degree at the coarsest scale
2.  the finest scale stays empty, which is a measured choice
3.  the generated grid places true edges at the coarsest scale
4.  the coarsest scale carries both self and cross edges
5.  reconciliation and the binary finest scale survive the change
6.  `SelfRec` and `CrossRec` separate the two kinds of true edge
7.  a scale with no true edge of one kind reports NaN for it
8.  the split appears in `to_dict` per scale
9.  a fitted model's sample maps back to an event trace exactly
10. the round trip counts the episodes it passed through
"""

import numpy as np
import pytest

from src.core.graph import MultiScaleDBNGraph
from src.core.scales import ScaleHierarchy
from src.data.event_log import fitted_trace_round_trip
from src.evaluation.metrics import ReconciliationMetrics, StructureMetrics
from src.synthetic.families import make_generator
from src.synthetic.scenarios import ScenarioGrid

# ---------------------------------------------------------------------------
# 1-5.  the grid
# ---------------------------------------------------------------------------


def test_every_scenario_carries_edges_at_the_coarsest_scale():
    """The screen puts the coarsest scale above its floor in every family."""
    for cfg in ScenarioGrid.full_benchmark().scenarios:
        assert cfg.max_indegree[-1] >= 1, cfg.name


def test_the_finest_scale_stays_empty():
    """A minute-scale edge is worth less than its price, so none is placed."""
    for cfg in ScenarioGrid.full_benchmark().scenarios:
        assert cfg.max_indegree[0] == 0, cfg.name


@pytest.fixture(scope="module")
def generated():
    """One scenario of the grid, generated once."""
    cfg = ScenarioGrid.full_benchmark().scenarios[0]
    seed = cfg.seed * 1000
    return cfg, make_generator(cfg, seed=seed).generate(seed=seed)


def _edges(params, ki):
    return {
        (s, lag, dst) for dst, ps in params[ki].parent_sets.items() for s, lag in ps
    }


def test_the_generator_places_coarsest_scale_edges(generated):
    """A declared in-degree at the coarsest scale has to reach the ground truth."""
    _cfg, ds = generated
    assert _edges(ds.params, len(ds.params) - 1)


def test_the_coarsest_scale_carries_both_kinds_of_edge():
    """Self and cross edges both have to appear, or one kind is untested."""
    kinds = set()
    for cfg in ScenarioGrid.full_benchmark().scenarios[:6]:
        seed = cfg.seed * 1000
        ds = make_generator(cfg, seed=seed).generate(seed=seed)
        for src, _lag, dst in _edges(ds.params, len(ds.params) - 1):
            kinds.add("self" if src == dst else "cross")
    assert kinds == {"self", "cross"}


def test_the_change_preserves_reconciliation_and_the_binary_finest_scale(generated):
    """The two invariants every fit is checked on must not move."""
    _cfg, ds = generated
    got = ReconciliationMetrics.compute(ds.data).to_dict()
    assert got["RecErr_X"] == 0.0 and got["RecErr_Y"] == 0.0
    assert int(np.asarray(ds.data.X[0]).max()) == 1


# ---------------------------------------------------------------------------
# 6-8.  recall by edge kind
# ---------------------------------------------------------------------------


def _graph(edges, hierarchy, names):
    g = MultiScaleDBNGraph(hierarchy=hierarchy, label_names=names)
    for k, src, lag, dst in edges:
        g.add_edge(k, src, lag, dst)
    return g


@pytest.fixture(scope="module")
def split_case():
    """A truth with both kinds at one scale, and an estimate that finds one."""
    hierarchy = ScaleHierarchy.from_uniform_branching(20, [5])
    names = ["a", "b", "c"]
    truth = _graph([(1, 0, 1, 0), (1, 1, 1, 2), (2, 2, 1, 2)], hierarchy, names)
    est = _graph([(1, 0, 1, 0), (2, 2, 1, 2)], hierarchy, names)
    return StructureMetrics.compute(est, truth).to_dict()


def test_self_and_cross_recall_separate_the_two_kinds(split_case):
    """Finding the self edge and missing the cross one has to show as such."""
    assert split_case["SelfRec_k1"] == 1.0
    assert split_case["CrossRec_k1"] == 0.0


def test_a_kind_with_no_true_edge_reports_nan(split_case):
    """Scale 2 carries a self edge and no cross edge, so cross recall is undefined."""
    assert split_case["SelfRec_k2"] == 1.0
    assert np.isnan(split_case["CrossRec_k2"])


def test_the_pooled_recall_still_averages_the_two(split_case):
    """One of two true edges found at scale 1, whichever kind they are."""
    assert split_case["Rec_k1"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 9-10.  the trace a fit maps back to
# ---------------------------------------------------------------------------


def test_a_fitted_model_maps_back_to_an_event_trace():
    """Counts and durations survive being written as dated episodes and read back."""
    from copy import deepcopy

    from src.learning.structural_em import TopDownSEM

    cfg = ScenarioGrid.sanity_check().scenarios[0]
    seed = cfg.seed * 1000
    gen = make_generator(cfg, seed=seed)
    ds = gen.generate(seed=seed)
    truth = MultiScaleDBNGraph(hierarchy=ds.hierarchy, label_names=gen.label_names)
    for ki, p in enumerate(ds.params):
        for dst, ps in p.parent_sets.items():
            for src, lag in ps:
                truth.add_edge(ki + 1, src, lag, dst)

    fitted, _, _ = TopDownSEM(max_outer_iter=1, lam_edge=2.0, verbose=0).fit(
        ds.data, [deepcopy(p) for p in ds.params], deepcopy(truth)
    )
    got = fitted_trace_round_trip(
        fitted,
        ds.hierarchy,
        gen.label_names,
        ds.calendar,
        slot_minutes=cfg.slot_minutes,
        branching=cfg.branching,
        seed=1,
    )
    assert got["count_error"] == 0.0
    assert got["duration_error"] == 0.0
    assert got["n_episodes"] > 0
