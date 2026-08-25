"""
utils/temporal_viz.py
=====================
Timeline and snapshot figures of events and their lagged relations.

Functions
---------
edge_occurrences        : epochs at which a lagged edge links two events
plot_event_timeline     : one row per label, one column per epoch, an arc per
                          active lagged edge
to_temporal_graph       : networkx-temporal graph, one snapshot per epoch
plot_snapshot_sequence  : the active relations of consecutive epochs, one
                          panel each
plot_teneto_slices      : teneto slice plot of the co-activity contacts
experiment_plots        : the figures above for one stored experiment run

The timeline reads like a slice plot: a dot where an episode of a label
starts, sized by the count of the epoch, and an arc from the source event to
the target event for every epoch at which a lagged edge is active.
matplotlib is required for every figure, networkx-temporal for the snapshot
sequence, teneto for the slice plot; a missing package raises a named error
when the function is called, not at import.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

try:
    import networkx_temporal as _tx

    _TX_AVAILABLE = True
except ImportError:
    _TX_AVAILABLE = False

Edge = Tuple[int, int, int]  # (source label, lag, target label)


# ---------------------------------------------------------------------------
# Guard helpers
# ---------------------------------------------------------------------------


def _require_mpl(fn_name: str) -> None:
    if not _MPL_AVAILABLE:
        raise ImportError(
            f"{fn_name}: matplotlib is required for visualisation. "
            "Install it with: pip install matplotlib"
        )


def _require_tx(fn_name: str) -> None:
    if not _TX_AVAILABLE:
        raise ImportError(
            f"{fn_name}: networkx-temporal is required. "
            "Install it with: pip install networkx-temporal"
        )


def _import_teneto(fn_name: str):
    """teneto, imported on first use since its import cost is high."""
    try:
        from teneto import TemporalNetwork
    except ImportError as exc:
        raise ImportError(
            f"{fn_name}: teneto is required. "
            "Install it with: pip install teneto 'nilearn<0.13'"
        ) from exc
    return TemporalNetwork


# ---------------------------------------------------------------------------
# Edge instances
# ---------------------------------------------------------------------------


def edge_occurrences(
    starts: np.ndarray,  # (T, L)
    edges: Sequence[Edge],
    t0: int = 0,
    t1: Optional[int] = None,
) -> List[Tuple[int, int, int, int]]:
    """
    Active instances of each lagged edge in [t0, t1).

    An edge (src, lag, dst) is active at epoch t when the source label has an
    event at t - lag and the target label has one at t.  Returns tuples
    (src, t_src, dst, t_dst) for plotting.
    """
    T = starts.shape[0]
    t1 = T if t1 is None else min(t1, T)
    occ = []
    for src, lag, dst in edges:
        for t in range(max(t0, lag), t1):
            if starts[t - lag, src] > 0 and starts[t, dst] > 0:
                occ.append((int(src), int(t - lag), int(dst), int(t)))
    return occ


# ---------------------------------------------------------------------------
# Timeline figure
# ---------------------------------------------------------------------------


def plot_event_timeline(
    starts: np.ndarray,  # (T, L)
    label_names: Optional[List[str]] = None,
    edges: Optional[Sequence[Edge]] = None,
    edges_true: Optional[Sequence[Edge]] = None,
    t0: int = 0,
    t1: Optional[int] = None,
    stamps: Optional[List[str]] = None,
    title: str = "Temporal network",
    figsize: Tuple[float, float] = (12, 4),
) -> "plt.Figure":
    """
    Events as dots on a label-by-epoch grid, lagged relations as arcs.

    `edges` draws solid black arcs, `edges_true` dashed green ones, so a
    learned graph is read against the one that generated the data.  `stamps`
    labels the shown epochs on the x axis, sparsely.
    """
    _require_mpl("plot_event_timeline")
    T, L = starts.shape
    t1 = T if t1 is None else min(t1, T)
    label_names = label_names or [f"l{i}" for i in range(L)]
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.Set2(np.linspace(0, 1, max(L, 3)))

    for ell in range(L):
        ax.axhline(ell, color="0.85", lw=0.8, zorder=1)
    for t in range(t0, t1):
        ax.axvline(t, color="0.92", lw=0.5, zorder=0)

    for ell in range(L):
        counts = starts[t0:t1, ell]
        ts = np.flatnonzero(counts > 0) + t0
        sizes = 40 + 25 * np.minimum(counts[counts > 0], 4)
        ax.scatter(
            ts,
            np.full(ts.shape, ell, dtype=float),
            s=sizes,
            color=colors[ell % len(colors)],
            zorder=3,
        )

    def _arcs(edge_list, color, style):
        for src, t_src, dst, t_dst in edge_occurrences(starts, edge_list, t0, t1):
            ax.add_patch(
                mpatches.FancyArrowPatch(
                    (t_src, src),
                    (t_dst, dst),
                    connectionstyle="arc3,rad=-0.35",
                    arrowstyle="-|>",
                    mutation_scale=9,
                    lw=1.2,
                    color=color,
                    linestyle=style,
                    alpha=0.85,
                    zorder=2,
                )
            )

    if edges_true:
        _arcs(edges_true, "seagreen", "--")
    if edges:
        _arcs(edges, "black", "-")

    ax.set_yticks(range(L), label_names)
    if stamps is not None:
        step = max(1, (t1 - t0) // 8)
        ticks = list(range(t0, t1, step))
        ax.set_xticks(ticks, [stamps[t - t0] for t in ticks], rotation=45, ha="right")
    ax.set_xlim(t0 - 0.7, t1 - 0.3)
    ax.set_ylim(-0.7, L - 0.3)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    handles = []
    if edges:
        handles.append(Line2D([], [], color="black", label="learned edge"))
    if edges_true:
        handles.append(
            Line2D([], [], color="seagreen", linestyle="--", label="true edge")
        )
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# networkx-temporal snapshots
# ---------------------------------------------------------------------------


def to_temporal_graph(
    starts: np.ndarray,  # (T, L)
    edges: Sequence[Edge],
    t0: int = 0,
    t1: Optional[int] = None,
    label_names: Optional[List[str]] = None,
):
    """
    networkx-temporal directed graph with one snapshot per epoch in [t0, t1).

    A snapshot holds a node for every label with an event in its epoch and an
    arrow for every lagged edge active at that epoch, pointing from the source
    label to the target label.
    """
    _require_tx("to_temporal_graph")
    T, L = starts.shape
    t1 = T if t1 is None else min(t1, T)
    label_names = label_names or [f"l{i}" for i in range(L)]
    TG = _tx.temporal_graph(t=max(t1 - t0, 1), directed=True, multigraph=False)
    for t in range(t0, t1):
        panel = TG[t - t0]
        for ell in np.flatnonzero(starts[t] > 0):
            panel.add_node(label_names[int(ell)])
        for src, _, dst, _ in edge_occurrences(starts, edges, t, t + 1):
            panel.add_edge(label_names[src], label_names[dst])
    return TG


def plot_snapshot_sequence(
    starts: np.ndarray,  # (T, L)
    edges: Sequence[Edge],
    t0: int = 0,
    n_panels: int = 4,
    label_names: Optional[List[str]] = None,
    stamps: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> "plt.Figure":
    """
    One panel per epoch with the relations active at that epoch.

    Drawn through networkx-temporal, so a panel shows the labels that fired
    in its epoch and an arrow for every lagged edge whose source and target
    both fired.
    """
    _require_mpl("plot_snapshot_sequence")
    _require_tx("plot_snapshot_sequence")
    TG = to_temporal_graph(starts, edges, t0, t0 + n_panels, label_names)
    fig = _tx.draw(
        TG,
        layout="kamada_kawai",
        figsize=figsize or (2.6 * n_panels, 2.8),
        suptitle=False,
        with_labels=True,
        node_color="tab:red",
        font_color="white",
        font_size=7,
        node_size=420,
    )
    for i, ax in enumerate(fig.axes):
        label = stamps[i] if stamps is not None else str(t0 + i)
        ax.set_title(f"t = {label}", fontsize=10)
    return fig


# ---------------------------------------------------------------------------
# teneto slices
# ---------------------------------------------------------------------------


def plot_teneto_slices(
    starts: np.ndarray,  # (T, L)
    t0: int = 0,
    t1: Optional[int] = None,
    label_names: Optional[List[str]] = None,
    title: str = "Temporal network",
    figsize: Tuple[float, float] = (12, 4),
) -> "plt.Figure":
    """
    teneto slice plot of the co-activity contacts of a window.

    A contact joins two labels that fire in the same epoch, which is the
    same-slice co-movement the latent factor accounts for, drawn as one
    column of nodes per epoch with an arc per contact.
    """
    _require_mpl("plot_teneto_slices")
    TemporalNetwork = _import_teneto("plot_teneto_slices")
    T, L = starts.shape
    t1 = T if t1 is None else min(t1, T)
    label_names = label_names or [f"l{i}" for i in range(L)]
    W = max(t1 - t0, 1)

    arr = np.zeros((L, L, W))
    for t in range(t0, t1):
        active = np.flatnonzero(starts[t] > 0)
        for a in range(active.size):
            for b in range(a + 1, active.size):
                arr[active[a], active[b], t - t0] = 1.0
                arr[active[b], active[a], t - t0] = 1.0

    fig, ax = plt.subplots(figsize=figsize)
    if arr.sum() == 0.0:
        # teneto cannot draw a window with no contact; show the bare grid.
        for t in range(W):
            ax.scatter([t] * L, range(L), s=100, color="0.7")
        ax.set_yticks(range(L), label_names)
        ax.set_xticks(range(W), list(range(t0, t1)))
    else:
        tn = TemporalNetwork(from_array=arr, nettype="bu")
        tn.plot(
            "slice_plot",
            ax=ax,
            nodelabels=label_names,
            timelabels=list(range(t0, t1)),
            cmap="Set2",
        )
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Stored-run orchestrator
# ---------------------------------------------------------------------------


def experiment_plots(
    results_json: str,
    output_dir: str,
    grid=None,
    method: str = "n2dbn",
    scenario: Optional[str] = None,
    replicate: int = 0,
    k_focus: int = 2,
    window: int = 40,
    n_panels: int = 4,
) -> List[str]:
    """
    Write the timeline, snapshot and slice figures for one stored run.

    Reads the learned edges of one (scenario, method, replicate) record from
    an experiment's results file, regenerates that replicate's dataset, and
    writes one timeline per scale plus a snapshot sequence and a teneto slice
    plot at `k_focus` into `output_dir`.  Returns the written paths; the
    slice plot is skipped when teneto is not installed.
    """
    _require_mpl("experiment_plots")
    from ..synthetic.families import make_generator
    from ..synthetic.scenarios import ScenarioGrid

    with open(results_json) as f:
        records = json.load(f)
    rows = [
        r
        for r in records
        if r["method"] == method
        and r.get("learned_edges")
        and r["replicate"] == replicate
        and (scenario is None or r["scenario"] == scenario)
    ]
    if not rows:
        raise ValueError(
            f"experiment_plots: no record with learned edges for method "
            f"{method!r}, scenario {scenario!r}, replicate {replicate}"
        )
    rec = rows[0]

    cfgs = {}
    if grid is not None:
        cfgs.update({c.name: c for c in grid.scenarios})
    else:
        for g in (ScenarioGrid.full_benchmark(), ScenarioGrid.sanity_check()):
            cfgs.update({c.name: c for c in g.scenarios})
    if rec["scenario"] not in cfgs:
        raise ValueError(f"experiment_plots: unknown scenario {rec['scenario']!r}")
    cfg = cfgs[rec["scenario"]]

    seed = cfg.seed * 1000 + rec["replicate"]
    gen = make_generator(cfg, seed=seed)
    ds = gen.generate(seed=seed)
    label_names = gen.label_names

    true_edges = {
        k
        + 1: sorted(
            (int(src), int(lag), int(dst))
            for dst, parents in ds.params[k].parent_sets.items()
            for (src, lag) in parents
        )
        for k in range(cfg.K)
    }
    learned = {int(k): [tuple(e) for e in v] for k, v in rec["learned_edges"].items()}

    os.makedirs(output_dir, exist_ok=True)
    base = f"{rec['scenario']}_rep{rec['replicate']}"
    paths: List[str] = []

    for k in range(1, cfg.K + 1):
        starts = ds.data.get_X(k)
        w = min(window, starts.shape[0])
        stamps = [ds.calendar.stamp(k, t) for t in range(w)]
        fig = plot_event_timeline(
            starts,
            label_names,
            edges=learned.get(k, []),
            edges_true=true_edges.get(k, []),
            t0=0,
            t1=w,
            stamps=stamps,
            title=f"{rec['scenario']}  scale k={k}  ({method})",
        )
        path = os.path.join(output_dir, f"{base}_timeline_k{k}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

    k = min(k_focus, cfg.K)
    starts = ds.data.get_X(k)
    stamps = [ds.calendar.stamp(k, t) for t in range(n_panels)]
    fig = plot_snapshot_sequence(
        starts,
        learned.get(k, []),
        t0=0,
        n_panels=n_panels,
        label_names=label_names,
        stamps=stamps,
    )
    path = os.path.join(output_dir, f"{base}_snapshots_k{k}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    w = min(window, starts.shape[0])
    try:
        fig = plot_teneto_slices(
            starts,
            0,
            w,
            label_names,
            title=f"{rec['scenario']}  co-activity  k={k}",
        )
    except ImportError:
        return paths
    path = os.path.join(output_dir, f"{base}_teneto_k{k}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)
    return paths
