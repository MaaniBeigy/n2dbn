"""
The tiled bnlearn benchmark.

bnlearn is the only external source of exact discrete conditional probability
tables, so these tests check both halves of what that buys: the tiling of a
released structure into lag-1 edges, and the scoring of a table estimate
against the published one.
"""

from __future__ import annotations

import builtins
import importlib
import importlib.util
import sys
from contextlib import contextmanager

import numpy as np
import pytest

from src.benchmarks.bnlearn_tiling import (
    AVAILABLE_NETWORKS,
    CPT,
    TiledNetwork,
    cpt_error,
    load_network,
    predictive_cpt_error,
    sample_tiled,
    score_structure,
    tile_to_2tbn,
    to_counts,
)

_HAS_PGMPY = importlib.util.find_spec("pgmpy") is not None
requires_pgmpy = pytest.mark.skipif(not _HAS_PGMPY, reason="pgmpy is not installed")

pytestmark = requires_pgmpy

# Node and edge counts of the released networks, pinned so a silently swapped
# or truncated download fails the load test instead of passing it.
NETWORK_SIZES = {
    "asia": (8, 8),
    "cancer": (5, 4),
    "earthquake": (5, 4),
    "sachs": (11, 17),
    "child": (20, 25),
    "insurance": (27, 52),
    "alarm": (37, 46),
}


@pytest.fixture(scope="module")
def asia() -> TiledNetwork:
    return tile_to_2tbn("asia")


def _copy_chain() -> TiledNetwork:
    """Two nodes where b deterministically copies whatever a it reads."""
    return TiledNetwork(
        name="copy",
        n_slices=2,
        node_names=["a", "b"],
        true_edges={(0, 1, 1)},
        cpds={
            "a": CPT(variables=("a",), table=np.array([[0.5, 0.5]])),
            "b": CPT(variables=("b", "a"), table=np.eye(2)),
        },
        cardinalities={"a": 2, "b": 2},
    )


def _ragged_parents() -> TiledNetwork:
    """Node c deterministically encodes its (a, b) parent pair as a*3 + b."""
    return TiledNetwork(
        name="ragged",
        n_slices=2,
        node_names=["a", "b", "c"],
        true_edges={(0, 1, 2), (1, 1, 2)},
        cpds={
            "a": CPT(variables=("a",), table=np.full((1, 2), 0.5)),
            "b": CPT(variables=("b",), table=np.full((1, 3), 1.0 / 3.0)),
            "c": CPT(variables=("c", "a", "b"), table=np.eye(6)),
        },
        cardinalities={"a": 2, "b": 3, "c": 6},
    )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", AVAILABLE_NETWORKS)
def test_every_listed_network_loads_at_its_published_size(name: str):
    model = load_network(name)
    n_nodes, n_edges = NETWORK_SIZES[name]

    assert len(model.nodes()) == n_nodes
    assert len(model.edges()) == n_edges
    # One released table per node, and every table variable is a node of the
    # graph; a partially parsed BIF fails both.
    cpds = model.get_cpds()
    assert len(cpds) == n_nodes
    assert {str(c.variable) for c in cpds} == {str(v) for v in model.nodes()}
    for cpd in cpds:
        # Released BIF tables round to three decimals, so the column sums are
        # near one rather than exactly one.
        assert np.abs(cpd.get_values().sum(axis=0) - 1.0).max() < 1e-2


def test_load_network_returns_the_same_object_for_both_spellings():
    short = load_network("asia")
    qualified = load_network("bnlearn/asia")
    # The prefix is stripped before the cache lookup, so the two names collapse
    # onto one parse rather than onto two equal copies.
    assert short is qualified


def test_load_network_names_the_available_set_for_an_unknown_network():
    with pytest.raises(ValueError, match="nope") as excinfo:
        load_network("nope")
    message = str(excinfo.value)
    assert all(name in message for name in AVAILABLE_NETWORKS)


# ---------------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------------


def test_tiling_asia_maps_each_static_edge_to_the_same_direction_at_lag_one(asia):
    static = load_network("asia")
    index = {name: i for i, name in enumerate(asia.node_names)}
    expected = {(index[str(u)], 1, index[str(v)]) for u, v in static.edges()}

    assert asia.true_edges == expected
    # asia has no reciprocal pair, so a source/destination swap is detectable.
    reversed_edges = {(index[str(v)], 1, index[str(u)]) for u, v in static.edges()}
    assert asia.true_edges != reversed_edges
    assert ("smoke", "lung") in set(static.edges())
    assert (index["smoke"], 1, index["lung"]) in asia.true_edges
    assert (index["lung"], 1, index["smoke"]) not in asia.true_edges

    assert asia.cardinalities == {name: 2 for name in asia.node_names}
    assert asia.name == "asia"
    assert asia.n_slices == 2


def test_node_names_are_a_topological_order_of_the_static_graph(asia):
    static = load_network("asia")
    position = {name: i for i, name in enumerate(asia.node_names)}
    assert sorted(position) == sorted(str(v) for v in static.nodes())
    for u, v in static.edges():
        assert position[str(u)] < position[str(v)]


def test_each_released_table_carries_its_variable_order_and_sums_to_one(asia):
    for node, cpt in asia.cpds.items():
        assert cpt.variables[0] == node
        assert cpt.table.shape[1] == asia.cardinalities[node]
        n_rows = int(np.prod([asia.cardinalities[p] for p in cpt.variables[1:]] or [1]))
        assert cpt.table.shape[0] == n_rows
        assert np.allclose(cpt.table.sum(axis=1), 1.0)


def test_the_stored_row_index_matches_the_pgmpy_state_lookup():
    # sachs carries three states per node, so a C/Fortran mix-up in the row
    # index or a missing transpose gives a different number here.
    model = load_network("sachs")
    tiled = tile_to_2tbn("sachs")
    state_names = {v: model.states[v] for v in model.nodes()}

    checked = 0
    for node, cpt in tiled.cpds.items():
        parents = cpt.variables[1:]
        dims = tuple(tiled.cardinalities[p] for p in parents)
        cpd = model.get_cpds(node)
        for config in np.ndindex(*dims) if dims else [()]:
            row = int(np.ravel_multi_index(config, dims)) if dims else 0
            for state in range(tiled.cardinalities[node]):
                lookup = {node: state_names[node][state]}
                lookup.update(
                    {p: state_names[p][c] for p, c in zip(parents, config)},
                )
                assert cpt.table[row, state] == pytest.approx(cpd.get_value(**lookup))
                checked += 1
    assert checked > 0
    assert any(len(c.variables) > 2 for c in tiled.cpds.values())


def test_tiling_rejects_a_template_of_one_slice():
    with pytest.raises(ValueError, match="at least 2"):
        tile_to_2tbn("asia", n_slices=1)


def test_a_qualified_name_and_a_wider_template_leave_the_edges_unchanged(asia):
    wide = tile_to_2tbn("bnlearn/asia", n_slices=5)
    assert wide.name == "asia"
    assert wide.n_slices == 5
    # The tiled process is stationary, so widening the template adds slices but
    # no new lags; the recorded edge set stays the lag-1 one.
    assert wide.true_edges == asia.true_edges
    assert {lag for _, lag, _ in wide.true_edges} == {1}


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def test_sampling_is_deterministic_in_the_seed_and_stays_in_range(asia):
    states = sample_tiled(asia, n_epochs=40, seed=0)
    same = sample_tiled(asia, n_epochs=40, seed=0)
    other = sample_tiled(asia, n_epochs=40, seed=1)

    assert states.shape == (40, len(asia.node_names))
    assert states.dtype == np.int64
    assert np.array_equal(states, same)
    assert not np.array_equal(states, other)

    for j, node in enumerate(asia.node_names):
        column = states[:, j]
        assert column.min() >= 0
        assert column.max() < asia.cardinalities[node]


def test_a_child_reads_its_parent_from_the_previous_epoch():
    states = sample_tiled(_copy_chain(), n_epochs=40, seed=7)
    a, b = states[:, 0], states[:, 1]

    # The parent has to move, otherwise "previous epoch" and "this epoch" agree
    # and the assertion below holds for the wrong reason.
    assert np.any(a[1:] != a[:-1])
    assert np.array_equal(b[1:], a[:-1])
    assert not np.array_equal(b[1:], a[1:])


def test_epoch_zero_is_drawn_from_the_static_joint():
    # A seed whose first parent draw is state 1, so copying the all-zero
    # initialization instead of the freshly drawn parent is distinguishable.
    seeds = [s for s in range(20) if sample_tiled(_copy_chain(), 1, seed=s)[0, 0] == 1]
    assert seeds

    states = sample_tiled(_copy_chain(), n_epochs=1, seed=seeds[0])
    assert states.shape == (1, 2)
    assert states[0, 0] == 1
    assert states[0, 1] == 1


def test_a_ragged_parent_pair_is_read_in_row_major_order():
    states = sample_tiled(_ragged_parents(), n_epochs=60, seed=11)
    a, b, c = states[:, 0], states[:, 1], states[:, 2]

    assert np.array_equal(c[1:], 3 * a[:-1] + b[:-1])
    # Both parents vary, so a Fortran-order read (a + 2*b) would disagree.
    assert len(np.unique(a)) == 2
    assert len(np.unique(b)) == 3
    assert not np.array_equal(c[1:], a[:-1] + 2 * b[:-1])


def test_sampling_zero_epochs_gives_an_empty_matrix():
    states = sample_tiled(_copy_chain(), n_epochs=0, seed=0)
    assert states.shape == (0, 2)
    assert states.dtype == np.int64


def test_a_degenerate_table_always_draws_its_only_supported_state():
    pinned = TiledNetwork(
        name="pinned",
        n_slices=2,
        node_names=["a"],
        true_edges=set(),
        cpds={"a": CPT(variables=("a",), table=np.array([[0.0, 1.0]]))},
        cardinalities={"a": 2},
    )
    assert np.array_equal(sample_tiled(pinned, n_epochs=12, seed=4)[:, 0], np.ones(12))


def test_a_table_that_sums_below_one_still_draws_a_valid_state():
    # Released tables round to three decimals, so the inverse CDF can be handed
    # a uniform above the last cumulative sum; the draw is clamped, not out of
    # bounds.
    short = TiledNetwork(
        name="short",
        n_slices=2,
        node_names=["a"],
        true_edges=set(),
        cpds={"a": CPT(variables=("a",), table=np.array([[0.4, 0.4]]))},
        cardinalities={"a": 2},
    )
    states = sample_tiled(short, n_epochs=400, seed=5)
    assert states.min() == 0
    assert states.max() == 1
    # The 0.2 of missing mass lands on the last state, so state 1 takes 0.6.
    assert 0.5 < states.mean() < 0.7


# ---------------------------------------------------------------------------
# Counts
# ---------------------------------------------------------------------------


def test_to_counts_marks_every_state_above_the_first():
    states = np.array([[0, 0, 0], [1, 2, 5], [0, 1, 3], [1, 0, 0]], dtype=np.int64)
    counts = to_counts(states, {"a": 2, "b": 3, "c": 6})

    assert counts.dtype == np.int64
    assert np.array_equal(
        counts,
        np.array([[0, 0, 0], [1, 1, 1], [0, 1, 1], [1, 0, 0]], dtype=np.int64),
    )


def test_to_counts_on_a_single_epoch_and_on_no_epochs():
    single = to_counts(np.array([[0, 2]], dtype=np.int64), {"a": 2, "b": 3})
    assert np.array_equal(single, np.array([[0, 1]], dtype=np.int64))

    empty = to_counts(np.zeros((0, 2), dtype=np.int64), {"a": 2, "b": 3})
    assert empty.shape == (0, 2)
    assert empty.dtype == np.int64


def test_to_counts_rejects_states_outside_their_variable(asia):
    states = sample_tiled(asia, n_epochs=5, seed=2)
    bad = states.copy()
    bad[0, 0] = 99
    with pytest.raises(ValueError, match="valid state index"):
        to_counts(bad, asia.cardinalities)

    negative = states.copy()
    negative[1, 1] = -1
    with pytest.raises(ValueError, match="valid state index"):
        to_counts(negative, asia.cardinalities)

    with pytest.raises(ValueError, match="valid state index"):
        to_counts(states[:, :2], asia.cardinalities)


def test_to_counts_rejects_a_one_dimensional_array():
    with pytest.raises(ValueError, match="valid state index"):
        to_counts(np.array([0, 1], dtype=np.int64), {"a": 2, "b": 2})


def test_a_state_index_at_the_top_of_its_range_is_accepted():
    # The bound is exclusive on the cardinality, so the last state is legal.
    counts = to_counts(np.array([[1, 2]], dtype=np.int64), {"a": 2, "b": 3})
    assert np.array_equal(counts, np.array([[1, 1]], dtype=np.int64))


# ---------------------------------------------------------------------------
# Structure scoring
# ---------------------------------------------------------------------------


def _as_parent_sets(edges):
    learned: dict = {}
    for src, lag, dst in edges:
        learned.setdefault(dst, set()).add((src, lag))
    return learned


def test_scoring_the_true_edge_set_is_perfect(asia):
    scores = score_structure(_as_parent_sets(asia.true_edges), asia)
    assert scores == {"SHD": 0.0, "Precision": 1.0, "Recall": 1.0, "F1": 1.0}


def test_scoring_a_reversed_edge_set_is_not_perfect(asia):
    flipped = {(dst, lag, src) for src, lag, dst in asia.true_edges}
    scores = score_structure(_as_parent_sets(flipped), asia)
    assert scores["Precision"] == 0.0
    assert scores["Recall"] == 0.0
    assert scores["SHD"] == float(2 * len(asia.true_edges))


def test_scoring_counts_one_miss_and_one_false_edge_exactly(asia):
    edges = sorted(asia.true_edges)
    n_true = len(edges)
    # Drop one true edge and add one at a lag the tiling never produces.
    wrong = set(edges[1:]) | {(0, 2, 0)}
    scores = score_structure(_as_parent_sets(wrong), asia)

    tp = n_true - 1
    assert scores["SHD"] == 2.0
    assert scores["Precision"] == pytest.approx(tp / n_true)
    assert scores["Recall"] == pytest.approx(tp / n_true)
    assert scores["F1"] == pytest.approx(tp / n_true)


def test_the_empty_graph_has_shd_equal_to_the_number_of_true_edges(asia):
    scores = score_structure({}, asia)
    assert scores["SHD"] == float(len(asia.true_edges))
    assert scores["Precision"] == 0.0
    assert scores["Recall"] == 0.0
    assert scores["F1"] == 0.0


def test_predicting_edges_against_an_empty_truth_scores_zero():
    empty = _empty_network()
    scores = score_structure({0: {(0, 1)}, 1: {(0, 3)}}, empty)
    assert scores == {"SHD": 2.0, "Precision": 0.0, "Recall": 0.0, "F1": 0.0}


def test_an_empty_truth_predicted_empty_scores_perfect():
    scores = score_structure({}, _empty_network())
    assert scores == {"SHD": 0.0, "Precision": 1.0, "Recall": 1.0, "F1": 1.0}


def test_a_destination_listed_with_no_learned_parents_is_ignored(asia):
    truth = _as_parent_sets(asia.true_edges)
    roots = [d for d in range(len(asia.node_names)) if d not in truth]
    assert roots

    padded = dict(truth)
    for dst in roots:
        padded[dst] = set()
    assert score_structure(padded, asia) == score_structure(truth, asia)


def _empty_network() -> TiledNetwork:
    return TiledNetwork(
        name="empty",
        n_slices=2,
        node_names=["a"],
        true_edges=set(),
        cpds={"a": CPT(variables=("a",), table=np.array([[0.5, 0.5]]))},
        cardinalities={"a": 2},
    )


# ---------------------------------------------------------------------------
# Table scoring
# ---------------------------------------------------------------------------


def test_a_released_table_scores_zero_against_itself(asia):
    for node, cpt in asia.cpds.items():
        assert cpt_error(cpt.table, asia, node) == 0.0


def test_a_perturbed_table_scores_a_positive_distance(asia):
    node = "dysp"
    released = asia.cpds[node].table
    perturbed = np.roll(released, shift=1, axis=1)
    assert cpt_error(perturbed, asia, node) > 0.0

    # Swapping a two-state row end to end costs exactly |p0 - p1| in TV.
    expected = float(np.mean(np.abs(released[:, 0] - released[:, 1])))
    assert cpt_error(perturbed, asia, node) == pytest.approx(expected)


def test_the_error_grows_with_the_size_of_the_perturbation(asia):
    node = "dysp"
    released = asia.cpds[node].table
    uniform = np.full_like(released, 1.0 / released.shape[1])

    errors = [
        cpt_error((1 - w) * released + w * uniform, asia, node)
        for w in (0.0, 0.1, 0.4, 1.0)
    ]
    assert errors[0] == 0.0
    assert errors == sorted(errors)
    assert errors[1] < errors[2] < errors[3]


def test_the_error_reaches_one_when_two_deterministic_tables_disagree():
    net = _copy_chain()
    assert cpt_error(np.eye(2), net, "b") == 0.0
    assert cpt_error(np.eye(2)[::-1], net, "b") == pytest.approx(1.0)


def test_a_table_of_the_wrong_shape_is_rejected(asia):
    with pytest.raises(ValueError, match="expected"):
        cpt_error(np.zeros((3, 3)), asia, "dysp")
    with pytest.raises(ValueError, match="expected"):
        cpt_error(np.zeros((4,)), asia, "dysp")


# ---------------------------------------------------------------------------
# Missing pgmpy
# ---------------------------------------------------------------------------


@contextmanager
def _without(*packages: str):
    """Reload the caller's modules with `packages` unimportable, then restore."""
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


def test_the_tiling_module_reports_pgmpy_as_absent():
    module = "src.benchmarks.bnlearn_tiling"
    try:
        with _without("pgmpy"):
            reloaded = importlib.reload(importlib.import_module(module))
            assert reloaded._PGMPY_AVAILABLE is False
            with pytest.raises(ImportError, match="load_network.. requires pgmpy"):
                reloaded.load_network("asia")
            with pytest.raises(ImportError, match="tile_to_2tbn.. requires pgmpy"):
                reloaded.tile_to_2tbn("asia")
    finally:
        assert importlib.reload(importlib.import_module(module))._PGMPY_AVAILABLE


# ===========================================================================
# predictive_cpt_error: comparing tables across learners that disagree
# ===========================================================================


@pytest.fixture(scope="module")
def asia_states(asia):
    """One sample long enough that the estimated tables are close to exact."""
    return sample_tiled(asia, 4000, seed=0)


def _true_parents(tiled):
    out = {}
    for src, lag, dst in tiled.true_edges:
        out.setdefault(dst, set()).add((src, lag))
    return out


def _mean(errors):
    return float(np.mean(list(errors.values())))


def test_predictive_cpt_error_true_parents_beat_none(asia, asia_states):
    """Estimating on the true parents is closer to the release than the marginal."""
    on_truth = _mean(predictive_cpt_error(asia_states, asia, _true_parents(asia)))
    on_empty = _mean(predictive_cpt_error(asia_states, asia, {}))
    assert 0.0 <= on_truth < on_empty


def test_predictive_cpt_error_scores_every_node(asia, asia_states):
    """One score per released table, all finite and in the unit interval."""
    errors = predictive_cpt_error(asia_states, asia, _true_parents(asia))
    assert set(errors) == set(asia.cpds)
    assert all(0.0 <= v <= 1.0 for v in errors.values())


def test_predictive_cpt_error_charges_spurious_parents(asia, asia_states):
    """A parent set holding every node pays for the configurations it splits."""
    L = len(asia.node_names)
    everything = {dst: {(src, 1) for src in range(L)} for dst in range(L)}
    on_truth = _mean(predictive_cpt_error(asia_states, asia, _true_parents(asia)))
    assert _mean(predictive_cpt_error(asia_states, asia, everything)) > on_truth


def test_predictive_cpt_error_missing_target_is_the_marginal(asia, asia_states):
    """A target absent from the mapping is scored as having no parents."""
    absent = predictive_cpt_error(asia_states, asia, {})
    empty = predictive_cpt_error(
        asia_states, asia, {d: set() for d in range(len(asia.node_names))}
    )
    assert absent == pytest.approx(empty)


def test_predictive_cpt_error_rejects_wrong_width(asia, asia_states):
    """A state matrix that is not one column per node cannot be scored."""
    with pytest.raises(ValueError, match="columns"):
        predictive_cpt_error(asia_states[:, :3], asia, {})
