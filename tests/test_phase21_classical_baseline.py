"""
The baseline that lets a published structure learner supply the graph.

The comparison in experiment 5 is only meaningful when everything except the
graph is held fixed, so the assertions here are about the plumbing that keeps it
that way.  The one that matters most is that a graph installed from outside the
structure step reaches the likelihood at all: a lagged parent with no
coefficient in the parameter block is invisible to every objective, and a
baseline in that state would be scored as if it had reported no edges.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.graph import MultiScaleDBNGraph
from src.evaluation.baselines import (
    ClassicalStructure,
    EmptyGraph,
    _seed_lag_coefficients,
)
from src.evaluation.protocol import EvaluationProtocol, apply_search_space
from src.learning.structural_em import TopDownSEM
from src.synthetic.families import make_generator
from src.synthetic.scenarios import ScenarioGrid

pytest.importorskip("pgmpy")

SEM_KW = dict(max_outer_iter=1, lam_edge=2.0, verbose=0)


@pytest.fixture(scope="module")
def scenario():
    return next(
        s for s in ScenarioGrid.sanity_check() if s.name == "sanity_L3_M2_r0_F2"
    )


@pytest.fixture(scope="module")
def dataset(scenario):
    return make_generator(scenario, seed=0).generate(seed=0)


def _graph(dataset):
    return MultiScaleDBNGraph(hierarchy=dataset.hierarchy, label_names=["a", "b", "c"])


def _configured(algorithm, scenario, **kwargs):
    method = ClassicalStructure(algorithm=algorithm, **SEM_KW, **kwargs)
    apply_search_space(method, scenario)
    return method


# ---------------------------------------------------------------------------
# The search space reaches the learner
# ---------------------------------------------------------------------------


def test_the_protocol_hands_the_scenario_budget_to_the_learner(scenario):
    method = _configured("granger", scenario)
    assert method.max_lag == {k: scenario.max_lag[k - 1] for k in (1, 2, 3)}
    assert method.max_indegree == {
        k: max(scenario.max_indegree[k - 1], 1) for k in (1, 2, 3)
    }


def test_an_unconfigured_learner_defaults_to_no_edges(dataset):
    method = ClassicalStructure(algorithm="granger", **SEM_KW)
    assert method.learn_graph(dataset.data, _graph(dataset)).n_edges() == 0


def test_the_learned_graph_stays_inside_the_budget(scenario, dataset):
    method = _configured("granger", scenario)
    graph = method.learn_graph(dataset.data, _graph(dataset))
    for k in (1, 2, 3):
        for ell in range(3):
            parents = graph.get_parents(k, ell)
            assert len(parents) <= max(scenario.max_indegree[k - 1], 1)
            assert all(1 <= lag <= scenario.max_lag[k - 1] for _, lag in parents)


def test_the_learned_graph_replaces_whatever_it_was_given(scenario, dataset):
    method = _configured("empty", scenario)
    seeded = _graph(dataset)
    seeded.add_edge(2, 0, 1, 1)
    assert method.learn_graph(dataset.data, seeded).n_edges() == 0


# ---------------------------------------------------------------------------
# Lag coefficients
# ---------------------------------------------------------------------------


def test_seeding_creates_one_coefficient_per_named_parent(dataset):
    params = dataset.params[1]
    params.parent_sets[0] = {(2, 1)}
    params.fine_count.betaLag = {}
    params.fine_dur.alphaLag = {}
    _seed_lag_coefficients(params)
    assert (2, 1) in params.fine_count.betaLag
    assert (2, 1) in params.fine_dur.alphaLag
    assert params.fine_count.betaLag[(2, 1)].shape == (params.L,)


def test_seeding_covers_the_coarsest_scale(dataset):
    params = dataset.params[-1]
    params.parent_sets[0] = {(1, 1)}
    params.coarse_count.betaLag = {}
    params.coarse_dur.alphaLag = {}
    _seed_lag_coefficients(params)
    assert (1, 1) in params.coarse_count.betaLag
    assert (1, 1) in params.coarse_dur.alphaLag


def test_seeding_never_overwrites_a_fitted_coefficient(dataset):
    params = dataset.params[1]
    params.parent_sets[0] = {(2, 1)}
    params.fine_count.betaLag = {(2, 1): np.full(params.L, 0.7)}
    _seed_lag_coefficients(params)
    assert np.allclose(params.fine_count.betaLag[(2, 1)], 0.7)


def test_a_learned_edge_changes_the_held_out_objective(scenario, dataset):
    """A graph with no coefficients would score exactly as the empty graph."""
    train, valid, _ = dataset.data.split(fracs=(0.7, 0.15, 0.15))
    graph = _graph(dataset)

    method = _configured("granger", scenario)
    learned_params, learned_graph, _ = method.fit(train, dataset.params, graph)
    assert learned_graph.n_edges() > 0

    empty = EmptyGraph(**SEM_KW)
    empty_params, empty_graph, _ = empty.fit(train, dataset.params, graph)

    protocol = EvaluationProtocol(methods={}, n_replicates=1, verbose=0)
    with_edges = protocol._validation_nll(valid, learned_params, learned_graph)
    without = protocol._validation_nll(valid, empty_params, empty_graph)
    assert with_edges != pytest.approx(without)


# ---------------------------------------------------------------------------
# The full interface
# ---------------------------------------------------------------------------


def test_fit_returns_the_same_shapes_as_the_model_itself(scenario, dataset):
    train, _, _ = dataset.data.split(fracs=(0.7, 0.15, 0.15))
    graph = _graph(dataset)
    method = _configured("dynotears", scenario)
    params, learned, history = method.fit(train, dataset.params, graph)
    assert len(params) == 3
    assert learned.K == 3
    assert np.isfinite(history.elbo[-1])


def test_predict_returns_one_summary_per_scale(scenario, dataset):
    method = _configured("granger", scenario)
    summaries = method.predict(dataset.data, dataset.params, _graph(dataset))
    assert len(summaries) == 3
    assert all(s is not None for s in summaries)


def test_the_structure_step_stays_off(scenario, dataset):
    method = _configured("granger", scenario)
    assert method._sem_kwargs["do_structure"] is False


def test_options_reach_the_learner(scenario, dataset):
    method = _configured(
        "hc-bic", scenario, encoding="quantile", n_bins=3, options={"n_bins": 9}
    )
    assert method.options["encoding"] == "quantile"
    assert method.options["n_bins"] == 3
    assert method.learn_graph(dataset.data, _graph(dataset)) is not None


# ---------------------------------------------------------------------------
# Inside the protocol
# ---------------------------------------------------------------------------


def test_the_protocol_scores_a_classical_learner_beside_the_model(scenario):
    grid = ScenarioGrid(scenarios=[scenario])
    protocol = EvaluationProtocol(
        methods={
            "n2dbn": TopDownSEM(**SEM_KW),
            "granger": ClassicalStructure(algorithm="granger", **SEM_KW),
        },
        n_replicates=1,
        verbose=0,
    )
    results = protocol.run(grid)
    assert {r.method_name for r in results} == {"n2dbn", "granger"}
    assert all("error" not in r.metrics for r in results)
    assert all(np.isfinite(r.metrics["SHD_total"]) for r in results)


# ---------------------------------------------------------------------------
# The experiment script
# ---------------------------------------------------------------------------


def test_the_algorithm_list_resolves_its_three_forms():
    from experiments.exp5_algorithm_comparison import (
        DEFAULT_ALGORITHMS,
        resolve_algorithms,
    )
    from src.learning.classical import ALGORITHMS

    assert resolve_algorithms("default") == list(DEFAULT_ALGORITHMS)
    assert resolve_algorithms("all") == [a for a in ALGORITHMS if a != "empty"]
    assert resolve_algorithms("granger, dynotears") == ["granger", "dynotears"]


def test_an_unknown_algorithm_name_stops_the_script():
    from experiments.exp5_algorithm_comparison import resolve_algorithms

    with pytest.raises(SystemExit, match="unknown algorithms"):
        resolve_algorithms("notears-plus")


def test_a_pgmpy_algorithm_is_dropped_when_pgmpy_is_missing(monkeypatch, capsys):
    import experiments.exp5_algorithm_comparison as exp5

    monkeypatch.setattr(exp5, "available_algorithms", lambda: ["dynotears", "granger"])
    assert exp5.resolve_algorithms("hc-bic,granger") == ["granger"]
    assert "skipping ['hc-bic']" in capsys.readouterr().out
