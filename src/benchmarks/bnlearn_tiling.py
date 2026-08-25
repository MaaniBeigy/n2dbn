"""
benchmarks/bnlearn_tiling.py
============================
Tiled bnlearn networks, the external ground truth that ships exact CPTs.

The bnlearn repository is the only external source of released discrete
conditional probability tables, which makes it the external counterpart to
parameter recovery: a learned table is scored against a published one instead
of against this project's own generator.

Tiling
------
Each intra-slice edge u to v of the static network becomes a lag-1 edge from u
at t-1 to v at t.  This is the standard tiling and matches the published
precedents.  Because no edge stays inside a slice, the tiled network is acyclic
by construction and its score decomposes per target.

Usage
-----
    from src.benchmarks.bnlearn_tiling import (
        tile_to_2tbn, sample_tiled, to_counts, score_structure, cpt_error,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Set, Tuple

import numpy as np

from ..evaluation.metrics import _safe_div

try:
    import networkx as nx
    from pgmpy import example_models

    _PGMPY_AVAILABLE = True
except ImportError:
    _PGMPY_AVAILABLE = False


# Small-to-medium discrete networks; every name here was checked to load and to
# carry both a structure and its released CPDs.
AVAILABLE_NETWORKS: Tuple[str, ...] = (
    "asia",
    "cancer",
    "earthquake",
    "sachs",
    "child",
    "insurance",
    "alarm",
)

_PREFIX = "bnlearn/"


def _require_pgmpy(fn: str) -> None:
    """Raise a named ImportError when pgmpy is unavailable."""
    if not _PGMPY_AVAILABLE:
        raise ImportError(f"{fn}() requires pgmpy. Install: pip install pgmpy")


def _short_name(name: str) -> str:
    """Strip the `bnlearn/` prefix so both spellings of a name agree."""
    return name[len(_PREFIX) :] if name.startswith(_PREFIX) else name


@lru_cache(maxsize=None)
def _load_cached(short: str) -> object:
    """Parse one released network, keeping the result for later calls."""
    # Each load is an HTTP round trip to the model repository, so the parsed
    # model is kept and every caller treats it as read-only.
    return example_models.load_model(_PREFIX + short)


@dataclass(frozen=True)
class CPT:
    """
    One released conditional probability table in conditioning-row layout.

    Attributes
    ----------
    variables : target first, then its parents in the order the rows index them
    table     : (n_parent_configs, n_states) array whose rows each sum to 1
    """

    variables: Tuple[str, ...]
    table: np.ndarray


@dataclass(frozen=True)
class TiledNetwork:
    """
    A static bnlearn network tiled into a temporal one, with its exact CPTs.

    Attributes
    ----------
    name          : short network name, without the `bnlearn/` prefix
    n_slices      : number of slices the tiled template spans
    node_names    : variables in a fixed topological order; the column order
    true_edges    : (src_index, lag, dst_index) triples, lag >= 1
    cpds          : node name to its released CPT
    cardinalities : node name to its number of states
    """

    name: str
    n_slices: int
    node_names: List[str]
    true_edges: Set[Tuple[int, int, int]]
    cpds: Dict[str, CPT] = field(default_factory=dict)
    cardinalities: Dict[str, int] = field(default_factory=dict)


def load_network(name: str) -> object:
    """
    Load a released bnlearn discrete network by short or qualified name.

    Args:
        name: `asia` or `bnlearn/asia`.

    Returns:
        The pgmpy DiscreteBayesianNetwork carrying its released CPDs.

    Raises:
        ValueError: the name is outside AVAILABLE_NETWORKS.
    """
    _require_pgmpy("load_network")
    short = _short_name(name)
    if short not in AVAILABLE_NETWORKS:
        raise ValueError(
            f"unknown bnlearn network {name!r}; "
            f"available: {', '.join(AVAILABLE_NETWORKS)}"
        )
    return _load_cached(short)


def tile_to_2tbn(name: str, n_slices: int = 2) -> TiledNetwork:
    """
    Tile a static bnlearn network into a temporal one with lag-1 edges.

    Args:
        name: short or qualified network name.
        n_slices: number of slices the template spans; 2 gives the 2-TBN.

    Returns:
        A TiledNetwork holding the lagged edges and the released tables.

    Raises:
        ValueError: n_slices is below 2.
    """
    _require_pgmpy("tile_to_2tbn")
    if n_slices < 2:
        raise ValueError(f"n_slices must be at least 2, got {n_slices}")

    model = load_network(name)
    # Lexicographic order makes the column layout depend on the node names
    # alone, not on the order the parser happened to insert them.
    node_names = [str(v) for v in nx.lexicographical_topological_sort(model)]
    index = {v: i for i, v in enumerate(node_names)}

    # The tiled structure is stationary, so the same lag-1 edge set holds
    # between every adjacent pair of the n_slices slices.
    true_edges = {(index[str(u)], 1, index[str(v)]) for u, v in model.edges()}

    cardinalities = {v: int(model.get_cardinality(v)) for v in node_names}
    cpds: Dict[str, CPT] = {}
    for cpd in model.get_cpds():
        # pgmpy lays the parent configurations along the columns; transposing
        # puts one conditional distribution per row.
        table = np.asarray(cpd.get_values(), dtype=float).T
        cpds[str(cpd.variable)] = CPT(
            variables=tuple(str(v) for v in cpd.variables),
            table=table,
        )

    return TiledNetwork(
        name=_short_name(name),
        n_slices=n_slices,
        node_names=node_names,
        true_edges=true_edges,
        cpds=cpds,
        cardinalities=cardinalities,
    )


def sample_tiled(tiled: TiledNetwork, n_epochs: int, seed: int = 0) -> np.ndarray:
    """
    Draw an ancestral sample of `n_epochs` epochs from a tiled network.

    Args:
        tiled: the tiled network to sample.
        n_epochs: number of epochs to draw.
        seed: seed of the generator; the draw is a function of it alone.

    Returns:
        (n_epochs, n_nodes) int64 state indices, columns in node_names order.
    """
    rng = np.random.default_rng(seed)
    index = {v: i for i, v in enumerate(tiled.node_names)}
    states = np.zeros((n_epochs, len(tiled.node_names)), dtype=np.int64)

    for t in range(n_epochs):
        # Epoch 0 comes from the static joint, so its parents are read from the
        # row being filled; topological order guarantees they are already set.
        # Every later epoch reads its parents from the previous epoch.
        source = states[t] if t == 0 else states[t - 1]
        for name in tiled.node_names:
            cpt = tiled.cpds[name]
            parents = cpt.variables[1:]
            if parents:
                config = tuple(int(source[index[p]]) for p in parents)
                dims = tuple(tiled.cardinalities[p] for p in parents)
                row = int(np.ravel_multi_index(config, dims))
            else:
                row = 0
            probs = cpt.table[row]
            # Inverse-CDF rather than rng.choice: released tables sum to 1 only
            # up to rounding, which rng.choice rejects.
            draw = int(np.searchsorted(np.cumsum(probs), rng.random()))
            states[t, index[name]] = min(draw, probs.size - 1)

    return states


def to_counts(states: np.ndarray, cardinalities: Dict[str, int]) -> np.ndarray:
    """
    Turn categorical states into the binary count matrix this model consumes.

    A variable counts 1 on every state except its first one, which is the
    baseline.

    Args:
        states: (n_epochs, n_nodes) state indices, columns in node_names order.
        cardinalities: node name to number of states, in that same order.

    Returns:
        (n_epochs, n_nodes) int64 matrix of event starts.

    Raises:
        ValueError: states is not 2-D, or a column count or a state index is
            outside its variable.
    """
    cards = np.asarray(list(cardinalities.values()), dtype=np.int64)
    if (
        states.ndim != 2
        or states.shape[1] != cards.size
        or np.any(states < 0)
        or np.any(states >= cards)
    ):
        raise ValueError("states must hold one valid state index per variable column")
    # The finest scale of this model is an event-start process, so a categorical
    # variable enters as present or absent rather than as a magnitude. State 0
    # is the baseline, which is not the same as the negative state: the released
    # networks list a boolean variable as ("yes", "no"), so a boolean counts 1
    # on its "no" state and multi-state variables such as insurance ThisCarDam
    # count 1 on everything above "None".
    return (states != 0).astype(np.int64)


def score_structure(
    learned: Dict[int, Set[Tuple[int, int]]],
    tiled: TiledNetwork,
) -> Dict[str, float]:
    """
    Score learned lagged parent sets against the tiled ground truth.

    Args:
        learned: destination node index to its set of (source index, lag) pairs.
        tiled: the tiled network holding the true lagged edges.

    Returns:
        SHD, Precision, Recall and F1 over the lagged edge set.
    """
    # StructureMetrics.compute takes two MultiScaleDBNGraph objects, which an
    # external network has no counterpart for, so its definitions are applied
    # here to the lagged triples instead: SHD = false positives + false
    # negatives, and an empty truth predicted empty scores a perfect 1.
    estimated = {
        (src, lag, dst) for dst, parents in learned.items() for src, lag in parents
    }
    truth = set(tiled.true_edges)

    tp = len(estimated & truth)
    fp = len(estimated - truth)
    fn = len(truth - estimated)

    if not estimated and not truth:
        prec = rec = f1 = 1.0
    else:
        prec = _safe_div(tp, tp + fp)
        rec = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * prec * rec, prec + rec)

    return {"SHD": float(fp + fn), "Precision": prec, "Recall": rec, "F1": f1}


def cpt_error(learned_table: np.ndarray, tiled: TiledNetwork, node: str) -> float:
    """
    Mean total variation distance per conditioning row against a released CPT.

    Args:
        learned_table: (n_parent_configs, n_states) estimate, rows summing to 1.
        tiled: the tiled network holding the released tables.
        node: name of the node whose table is scored.

    Returns:
        Mean over conditioning rows of 0.5 * sum |learned - released|.

    Raises:
        ValueError: the estimate does not match the released table shape.
    """
    released = tiled.cpds[node].table
    learned = np.asarray(learned_table, dtype=float)
    if learned.shape != released.shape:
        raise ValueError(
            f"table for {node!r} has shape {learned.shape}, "
            f"expected {released.shape}"
        )
    return float(np.mean(0.5 * np.abs(learned - released).sum(axis=1)))


def predictive_cpt_error(
    states: np.ndarray,
    tiled: TiledNetwork,
    parents: Dict[int, set],
    alpha: float = 1.0,
) -> Dict[str, float]:
    """
    Score a learned parent set by the conditional it implies, node by node.

    A learned parent set that differs from the true one gives a table of a
    different shape, so tables cannot be compared entry by entry across
    methods.  Both the learned model and the released network do answer the
    same question, namely the distribution of a node at epoch t given the whole
    previous epoch, so that conditional is what is scored: the table is
    estimated from `states` on the learned parents alone, the released tables
    are read at the true parent configuration, and the two are compared by
    total variation, averaged over the observed epochs.  An empty learned
    parent set is a valid answer under this score and is charged the distance
    from the marginal to the truth.

    Args:
        states: (n_epochs, n_nodes) integer states, columns in `node_names` order.
        tiled: the tiled network holding the released tables.
        parents: learned lagged parents per target column index, as returned by
            a structure learner; the lag is ignored because the tiling is 2-slice.
        alpha: Laplace count added to every cell, so an unvisited configuration
            still names a distribution.

    Returns:
        Mean total variation distance per node, keyed by node name.

    Raises:
        ValueError: `states` does not have one column per node.
    """
    names = list(tiled.node_names)
    if states.shape[1] != len(names):
        raise ValueError(f"states has {states.shape[1]} columns, expected {len(names)}")
    index = {n: i for i, n in enumerate(names)}
    prev, cur = states[:-1], states[1:]

    errors: Dict[str, float] = {}
    for node, cpt in tiled.cpds.items():
        j = index[node]
        n_states = tiled.cardinalities[node]

        # Released conditional, read at the true parent configuration of each
        # observed epoch.
        true_par = [index[v] for v in cpt.variables[1:]]
        if true_par:
            cards = [tiled.cardinalities[names[i]] for i in true_par]
            rows = np.ravel_multi_index(prev[:, true_par].T, cards)
        else:
            rows = np.zeros(len(prev), dtype=int)
        released = cpt.table[rows]

        # Learned conditional, estimated from the same epochs on the learned
        # parents alone.
        learn_par = sorted({int(s) for s, _ in parents.get(j, set())})
        if learn_par:
            cards = [tiled.cardinalities[names[i]] for i in learn_par]
            lrows = np.ravel_multi_index(prev[:, learn_par].T, cards)
            n_rows = int(np.prod(cards))
        else:
            lrows = np.zeros(len(prev), dtype=int)
            n_rows = 1
        counts = np.full((n_rows, n_states), float(alpha))
        np.add.at(counts, (lrows, cur[:, j]), 1.0)
        table = counts / counts.sum(axis=1, keepdims=True)

        errors[node] = float(np.mean(0.5 * np.abs(table[lrows] - released).sum(axis=1)))
    return errors
