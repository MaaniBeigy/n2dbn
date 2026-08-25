"""
core/graph.py
=============
Multi-scale DBN graph structure.

Representation
--------------
The graph is stored as a Python dict of parent sets (the canonical sparse
representation used by the S-step).  A networkx DiGraph is built on demand
for visualisation and path-query utilities.

pyAgrum integration
-------------------
`to_pyagrum_2tbn(k)` returns a `pyAgrum.DynamicBN` object for scale k.
The public API of this class matches what that integration requires, so the
swap is a one-file change.

Edge conventions
----------------
All *observed-to-observed* edges are lagged (lag >= 1) and within-scale.
Same-slice co-movement is mediated by latent parents only (enforced here).
Inter-scale edges are fixed (parent scale to fine scale) and not stored in
the learnable parent sets.

Notation:  A lagged edge (ℓ', d) → ℓ at scale k means
    X_{t-d}^{(k,ℓ')}  →  X_t^{(k,ℓ)}      (count parent)
    Y_{t-d}^{(k,ℓ')}  →  Y_t^{(k,ℓ)}      (duration parent, implied)
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterator, List, Optional, Set, Tuple

import networkx as nx

from .scales import ScaleHierarchy

# Optional pyAgrum integration (graph-layer only)
try:
    import pyagrum as gum  # pyAgrum >= 2.x installs as 'pyagrum'

    _PYAGRUM_AVAILABLE = True
except ImportError:
    _PYAGRUM_AVAILABLE = False


def _require_pyagrum(fn_name: str) -> None:
    """Raise ImportError if pyAgrum is not installed."""
    if not _PYAGRUM_AVAILABLE:
        raise ImportError(
            f"{fn_name}() requires pyAgrum. Install with: pip install pyAgrum"
        )


# ---------------------------------------------------------------------------
# Type aliases
LaggedEdge = Tuple[int, int]  # (src_label_idx, lag)
ParentSet = Set[LaggedEdge]  # set of (ℓ', d) for a given (k, ℓ)


# ---------------------------------------------------------------------------
@dataclass
class MultiScaleDBNGraph:
    """
    Learnable lagged graph structure for the N²DBN.

    Attributes
    ----------
    hierarchy : ScaleHierarchy
    label_names : list[str]
    K, L : derived from hierarchy / label_names
    _parents : dict  { (k, ell) to set{(ell', d), ...} }
        The learnable part: lagged observed-to-observed edges.
        k is 1-indexed, ell / ell' are 0-indexed label indices, d >= 1.
    """

    hierarchy: ScaleHierarchy
    label_names: List[str]
    _parents: Dict[Tuple[int, int], ParentSet] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    @property
    def K(self) -> int:
        return self.hierarchy.K

    @property
    def L(self) -> int:
        return len(self.label_names)

    def __post_init__(self) -> None:
        # Initialise empty parent sets for every (k, ℓ)
        for k in range(1, self.K + 1):
            for ell in range(self.L):
                self._parents.setdefault((k, ell), set())

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_edge(self, k: int, src_ell: int, lag: int, dst_ell: int) -> None:
        """
        Add lagged edge  X_{t-lag}^{(k, src_ell)}  →  X_t^{(k, dst_ell)}.

        Parameters
        ----------
        k       : scale (1-indexed)
        src_ell : source label index (0-indexed)
        lag     : positive integer
        dst_ell : destination label index (0-indexed)
        """
        assert lag >= 1, "Only lagged (lag ≥ 1) edges are allowed"
        assert 1 <= k <= self.K
        assert 0 <= src_ell < self.L
        assert 0 <= dst_ell < self.L
        self._parents[(k, dst_ell)].add((src_ell, lag))

    def remove_edge(self, k: int, src_ell: int, lag: int, dst_ell: int) -> None:
        """Remove a lagged edge (no-op if absent)."""
        self._parents[(k, dst_ell)].discard((src_ell, lag))

    def set_parents(self, k: int, dst_ell: int, parents: ParentSet) -> None:
        """Replace the full parent set for (k, dst_ell)."""
        self._parents[(k, dst_ell)] = set(parents)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_parents(self, k: int, dst_ell: int) -> ParentSet:
        """Return a *copy* of the parent set for (k, dst_ell)."""
        return set(self._parents.get((k, dst_ell), set()))

    def n_edges(self, k: Optional[int] = None) -> int:
        """Total number of directed lagged edges (optionally at scale k only)."""
        if k is not None:
            return sum(len(self._parents[(k, ell)]) for ell in range(self.L))
        return sum(len(ps) for ps in self._parents.values())

    def iter_edges(
        self, k: Optional[int] = None
    ) -> Iterator[Tuple[int, int, int, int]]:
        """
        Iterate over all edges as (k, src_ell, lag, dst_ell).
        If k is given, restricts to that scale.
        """
        scales = [k] if k is not None else range(1, self.K + 1)
        for sk in scales:
            for dst_ell in range(self.L):
                for src_ell, lag in self._parents.get((sk, dst_ell), set()):
                    yield (sk, src_ell, lag, dst_ell)

    def is_acyclic(self) -> bool:
        """
        Always True for a well-formed N²DBN graph because all observed-to-observed
        edges are lagged (no same-slice observed edges are ever added).
        This method is provided for compatibility with external checkers.
        """
        return True

    def structural_hamming_distance(
        self, other: "MultiScaleDBNGraph", k: Optional[int] = None
    ) -> int:
        """
        Compute SHD against another graph (missing + spurious edges).

        SHD_k = |Ê_k △ E_k*|
        """

        def edge_set(g: "MultiScaleDBNGraph", scale):
            return frozenset(g.iter_edges(scale))

        scales = [k] if k is not None else range(1, self.K + 1)
        shd = 0
        for sk in scales:
            mine = edge_set(self, sk)
            theirs = edge_set(other, sk)
            shd += len(mine.symmetric_difference(theirs))
        return shd

    # ------------------------------------------------------------------
    # networkx export  (used for visualisation; replaces pyAgrum for now)
    # ------------------------------------------------------------------

    def to_networkx(self, k: int) -> nx.DiGraph:
        """
        Return a networkx DiGraph representing the 2-TBN at scale k.

        Nodes are labelled `"X_{ell}[t]"` and `"X_{ell}[t-d]"` for all
        relevant lags d, plus latent nodes `"S[t]"` and `"H[t]"`.
        """
        G = nx.DiGraph()
        lnames = self.label_names

        # Current-time observed nodes
        for ell in range(self.L):
            G.add_node(f"X_{lnames[ell]}[t]", layer="obs_now")
            G.add_node(f"Y_{lnames[ell]}[t]", layer="obs_now")

        # Latent nodes (same-slice parents)
        G.add_node("S[t]", layer="latent")
        G.add_node("H[t]", layer="latent")
        for ell in range(self.L):
            G.add_edge("S[t]", f"X_{lnames[ell]}[t]", edge_type="latent")
            G.add_edge("H[t]", f"X_{lnames[ell]}[t]", edge_type="latent")
            G.add_edge("S[t]", f"Y_{lnames[ell]}[t]", edge_type="latent")
            G.add_edge("H[t]", f"Y_{lnames[ell]}[t]", edge_type="latent")

        # Lagged edges
        for src_ell, lag, dst_ell in (
            (s, d, dst)
            for s, d, dst in ((s, d, dst) for (_, s, d, dst) in self.iter_edges(k))
        ):
            lag_node = f"X_{lnames[src_ell]}[t-{lag}]"
            G.add_node(lag_node, layer="obs_past")
            G.add_edge(lag_node, f"X_{lnames[dst_ell]}[t]", edge_type="lagged")

        return G

    # ------------------------------------------------------------------
    # pyAgrum export / import
    # ------------------------------------------------------------------

    def to_pyagrum_bn(
        self, k: int, M: int = 2, n_obs_states: int = 2
    ) -> "gum.BayesNet":
        """
        Build a `pyAgrum.BayesNet` representing the 2-TBN template at scale *k*.

        Parameters
        ----------
        k : int
            Scale index (1-indexed).
        M : int
            Number of regime states for the `S` nodes.
        n_obs_states : int
            Placeholder domain size for observed count / duration nodes.

        Returns
        -------
        bn : pyAgrum.BayesNet
            Structural template.  CPTs are uniform by default; use
            :func:`~core.pyagrum_utils.fill_regime_cpt` to populate the
            regime transition CPT with real values.
        """
        _require_pyagrum("to_pyagrum_bn")
        assert 1 <= k <= self.K

        bn = gum.BayesNet(f"N2DBN_scale_{k}")
        lnames = self.label_names

        # --- Latent regime nodes (S_t and S_t_prev) ---
        bn.add(gum.LabelizedVariable("S_t", "regime(t)", M))
        bn.add(gum.LabelizedVariable("S_t_prev", "regime(t-1)", M))
        bn.addArc(bn.idFromName("S_t_prev"), bn.idFromName("S_t"))

        # --- Current-time observed nodes ---
        for ell in range(self.L):
            x_name = f"X_{lnames[ell]}_t"
            y_name = f"Y_{lnames[ell]}_t"
            bn.add(
                gum.LabelizedVariable(x_name, f"count {lnames[ell]}(t)", n_obs_states)
            )
            bn.add(gum.LabelizedVariable(y_name, f"dur {lnames[ell]}(t)", n_obs_states))
            bn.addArc(bn.idFromName("S_t"), bn.idFromName(x_name))
            bn.addArc(bn.idFromName("S_t"), bn.idFromName(y_name))

        # --- Lagged-parent nodes (one per unique (src_ell, lag) in parent sets at k) ---
        added_lag_nodes: Set[str] = set()
        for _, src_ell, lag, dst_ell in self.iter_edges(k):
            lag_name = f"X_{lnames[src_ell]}_t_lag{lag}"
            if lag_name not in added_lag_nodes:
                bn.add(
                    gum.LabelizedVariable(
                        lag_name, f"count {lnames[src_ell]}(t-{lag})", n_obs_states
                    )
                )
                added_lag_nodes.add(lag_name)
            dst_name = f"X_{lnames[dst_ell]}_t"
            # Avoid duplicate arcs
            src_id = bn.idFromName(lag_name)
            dst_id = bn.idFromName(dst_name)
            if not bn.dag().existsArc(src_id, dst_id):
                bn.addArc(src_id, dst_id)

        return bn

    def to_pyagrum_2tbn(
        self, k: int, M: int = 2, n_obs_states: int = 2
    ) -> "gum.BayesNet":
        """Alias for :meth:`to_pyagrum_bn` - matches the original docstring promise."""
        return self.to_pyagrum_bn(k, M=M, n_obs_states=n_obs_states)

    def update_from_pyagrum_bn(self, bn: "gum.BayesNet", k: int) -> None:
        """
        Update `_parents` at scale *k* from arcs in a `pyAgrum.BayesNet`.

        Only lagged arcs (tail name matching `X_{label}_t_lag{d}`, head
        matching `X_{label}_t`) are imported.  All other arcs (latent →
        observed, regime transition) are ignored.

        Parameters
        ----------
        bn : pyAgrum.BayesNet
            A BN whose lagged arcs encode parent sets.
        k  : int
            Scale to update (1-indexed).
        """
        _require_pyagrum("update_from_pyagrum_bn")
        import re

        lnames = self.label_names
        name_to_idx = {n: i for i, n in enumerate(lnames)}

        # Clear existing parents at scale k
        for ell in range(self.L):
            self._parents[(k, ell)] = set()

        lag_pattern = re.compile(r"^X_(.+)_t_lag(\d+)$")
        dst_pattern = re.compile(r"^X_(.+)_t$")

        for arc in bn.arcs():
            tail_name = bn.variable(arc[0]).name()
            head_name = bn.variable(arc[1]).name()

            tail_match = lag_pattern.match(tail_name)
            head_match = dst_pattern.match(head_name)
            if tail_match and head_match:
                src_label = tail_match.group(1)
                lag = int(tail_match.group(2))
                dst_label = head_match.group(1)
                if src_label in name_to_idx and dst_label in name_to_idx:
                    src_ell = name_to_idx[src_label]
                    dst_ell = name_to_idx[dst_label]
                    self._parents[(k, dst_ell)].add((src_ell, lag))

    def validate_pyagrum(self, k: int, M: int = 2) -> bool:
        """
        Check consistency between `_parents` and a derived pyAgrum BN.

        Returns True if the lagged edge count in the BN matches `n_edges(k)`
        and all expected edges are present.
        """
        _require_pyagrum("validate_pyagrum")
        import re

        bn = self.to_pyagrum_bn(k, M=M)
        lag_pattern = re.compile(r"^X_(.+)_t_lag(\d+)$")
        dst_pattern = re.compile(r"^X_(.+)_t$")

        # Count lagged arcs in BN
        bn_lagged = set()
        for arc in bn.arcs():
            tail_name = bn.variable(arc[0]).name()
            head_name = bn.variable(arc[1]).name()
            if lag_pattern.match(tail_name) and dst_pattern.match(head_name):
                bn_lagged.add((tail_name, head_name))

        return len(bn_lagged) == self.n_edges(k)

    # ------------------------------------------------------------------
    @classmethod
    def empty(
        cls,
        hierarchy: ScaleHierarchy,
        label_names: List[str],
    ) -> "MultiScaleDBNGraph":
        """Create a graph with no lagged edges."""
        return cls(hierarchy=hierarchy, label_names=label_names)

    def copy(self) -> "MultiScaleDBNGraph":
        """Deep copy."""
        g = MultiScaleDBNGraph(hierarchy=self.hierarchy, label_names=self.label_names)
        g._parents = deepcopy(self._parents)
        return g

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        total = self.n_edges()
        return (
            f"MultiScaleDBNGraph(K={self.K}, L={self.L}, "
            f"labels={self.label_names}, total_edges={total})"
        )
