"""
utils/visualization.py
======================
Plotting utilities for the N²DBN model.

Functions
---------
plot_multiscale_counts      : stacked bar / line charts of X^(k) across scales
plot_regime_sequence        : colour-coded regime timeline at each scale
plot_factor_trajectory      : H^(k) components over time
plot_elbo_history           : ELBO convergence curve
plot_graph_structure        : per-scale edge diagram
plot_episode_comparison     : true vs. reconstructed episode traces

All functions return matplotlib Figure objects (or None if matplotlib is not
installed), so they work headlessly in CI and gracefully in notebooks.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

# Lazy import matplotlib so the module can be imported even without it
try:
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend by default
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

from ..core.variables import MultiScaleData
from ..inference.posterior import PosteriorSummaries

# ---------------------------------------------------------------------------
# Guard helper
# ---------------------------------------------------------------------------


def _require_mpl(fn_name: str):
    if not _MPL_AVAILABLE:
        raise ImportError(
            f"{fn_name}: matplotlib is required for visualisation. "
            "Install it with: pip install matplotlib"
        )


# ---------------------------------------------------------------------------
# Count / duration overview
# ---------------------------------------------------------------------------


def plot_multiscale_counts(
    data: MultiScaleData,
    label_names: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (14, 3),
    max_epochs: int = 200,
) -> "plt.Figure":
    """
    Plot X^(k) counts for each scale as a separate subplot row.

    Parameters
    ----------
    data        : MultiScaleData
    label_names : optional list of label strings (length L)
    max_epochs  : truncate display to this many epochs per scale
    """
    _require_mpl("plot_multiscale_counts")
    K = data.K
    L = data.get_X(1).shape[1]
    label_names = label_names or [f"ℓ{i}" for i in range(L)]

    fig, axes = plt.subplots(K, 1, figsize=(figsize[0], figsize[1] * K), squeeze=False)
    colors = plt.cm.tab10(np.linspace(0, 1, L))

    for ki, ax in enumerate(axes[:, 0]):
        k = K - ki  # coarsest at top
        X_k = data.get_X(k)[:max_epochs]
        T = X_k.shape[0]
        x = np.arange(T)
        bottom = np.zeros(T)
        for ell in range(L):
            ax.bar(
                x,
                X_k[:, ell],
                bottom=bottom,
                color=colors[ell],
                label=label_names[ell],
                width=1.0,
                align="edge",
            )
            bottom += X_k[:, ell]
        ax.set_ylabel(f"k={k}")
        ax.set_xlim(0, T)
        ax.tick_params(labelsize=7)
        if ki == 0:
            ax.legend(loc="upper right", fontsize=7, ncol=L)
    axes[-1, 0].set_xlabel("Epoch")
    fig.suptitle("Multi-scale counts X^(k)", fontsize=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Regime sequence
# ---------------------------------------------------------------------------


def plot_regime_sequence(
    summaries: List[PosteriorSummaries],
    true_S: Optional[List[np.ndarray]] = None,
    figsize: Tuple[float, float] = (14, 2),
    max_epochs: int = 200,
) -> "plt.Figure":
    """
    Colour-coded regime posterior γ_t(m) as an image, one row per scale.

    Parameters
    ----------
    summaries   : list of PosteriorSummaries (index 0 = scale 1)
    true_S      : optional list of true regime sequences for comparison
    max_epochs  : truncate display
    """
    _require_mpl("plot_regime_sequence")
    K = len(summaries)
    n_rows = K * (2 if true_S is not None else 1)
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(figsize[0], figsize[1] * n_rows), squeeze=False
    )
    row = 0
    for ki in range(K - 1, -1, -1):  # coarsest first
        k = ki + 1
        s = summaries[ki]
        gamma = s.gamma[:max_epochs]  # (T, M)
        ax = axes[row, 0]
        im = ax.imshow(
            gamma.T,
            aspect="auto",
            interpolation="nearest",
            vmin=0,
            vmax=1,
            cmap="Blues",
        )
        ax.set_ylabel(f"k={k} q(S)", fontsize=8)
        ax.set_yticks(range(gamma.shape[1]))
        ax.set_yticklabels([f"m={m}" for m in range(gamma.shape[1])], fontsize=6)
        row += 1
        if true_S is not None:
            S_k = true_S[ki][:max_epochs]
            ax2 = axes[row, 0]
            M = gamma.shape[1]
            S_img = np.zeros((1, len(S_k)))
            S_img[0] = S_k / max(M - 1, 1)
            ax2.imshow(
                S_img,
                aspect="auto",
                interpolation="nearest",
                vmin=0,
                vmax=1,
                cmap="tab10",
            )
            ax2.set_ylabel(f"k={k} true S", fontsize=8)
            ax2.set_yticks([])
            row += 1
    axes[-1, 0].set_xlabel("Epoch")
    fig.suptitle("Regime posterior γ_t(m)", fontsize=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Factor trajectory
# ---------------------------------------------------------------------------


def plot_factor_trajectory(
    summaries: List[PosteriorSummaries],
    true_H: Optional[List[np.ndarray]] = None,
    scale: int = 1,
    figsize: Tuple[float, float] = (14, 2),
    max_epochs: int = 200,
) -> "plt.Figure":
    """
    Plot the posterior mean H^(k) trajectory with ±2σ bands.

    Parameters
    ----------
    summaries : list of PosteriorSummaries
    true_H    : optional list of true H arrays for comparison
    scale     : which scale to plot (1-indexed)
    """
    _require_mpl("plot_factor_trajectory")
    ki = scale - 1
    s = summaries[ki]
    m_H = s.m_H[:max_epochs]  # (T, r)
    Sigma_H = s.Sigma_H[:max_epochs]  # (T, r, r)
    T, r = m_H.shape
    if r == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            f"r=0 at scale k={scale}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig

    fig, axes = plt.subplots(r, 1, figsize=(figsize[0], figsize[1] * r), squeeze=False)
    x = np.arange(T)
    for ri, ax in enumerate(axes[:, 0]):
        mu = m_H[:, ri]
        sig = np.sqrt(np.clip(Sigma_H[:, ri, ri], 0, None))
        ax.plot(x, mu, color="steelblue", label=f"q(H_{ri+1})")
        ax.fill_between(x, mu - 2 * sig, mu + 2 * sig, alpha=0.2, color="steelblue")
        if true_H is not None and true_H[ki].shape[1] > ri:
            ax.plot(
                x,
                true_H[ki][:T, ri],
                color="tomato",
                linestyle="--",
                label="true H",
                alpha=0.7,
            )
        ax.set_ylabel(f"H_{ri+1}", fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
    axes[-1, 0].set_xlabel("Epoch")
    fig.suptitle(f"Latent factor trajectory  k={scale}", fontsize=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# ELBO history
# ---------------------------------------------------------------------------


def plot_elbo_history(
    elbo: List[float],
    figsize: Tuple[float, float] = (7, 3),
) -> "plt.Figure":
    """Plot ELBO vs. outer EM iteration."""
    _require_mpl("plot_elbo_history")
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(range(1, len(elbo) + 1), elbo, marker="o", color="steelblue")
    ax.set_xlabel("Outer iteration")
    ax.set_ylabel("ELBO")
    ax.set_title("ELBO convergence")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Graph structure
# ---------------------------------------------------------------------------


def plot_graph_structure(
    graph,  # MultiScaleDBNGraph
    label_names: Optional[List[str]] = None,
    scale: int = 1,
    figsize: Tuple[float, float] = (6, 6),
) -> "plt.Figure":
    """
    Draw the learned DBN edges at a given scale as a directed graph.

    Requires networkx (already a project dependency).
    """
    _require_mpl("plot_graph_structure")
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("networkx is required for graph visualisation.")

    L = graph.L
    label_names = label_names or [f"ℓ{i}" for i in range(L)]

    G = nx.DiGraph()
    G.add_nodes_from(label_names)
    edge_labels = {}
    for ell in range(L):
        for src_ell, lag in graph.get_parents(scale, ell):
            src = label_names[src_ell]
            dst = label_names[ell]
            G.add_edge(src, dst)
            edge_labels[(src, dst)] = f"d={lag}"

    fig, ax = plt.subplots(figsize=figsize)
    pos = nx.circular_layout(G)
    nx.draw_networkx(
        G,
        pos=pos,
        ax=ax,
        with_labels=True,
        node_color="lightsteelblue",
        node_size=800,
        font_size=9,
        arrows=True,
        arrowsize=15,
        edge_color="steelblue",
    )
    nx.draw_networkx_edge_labels(
        G, pos=pos, edge_labels=edge_labels, ax=ax, font_size=7
    )
    ax.set_title(f"DBN structure at scale k={scale}", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Episode comparison
# ---------------------------------------------------------------------------


def plot_episode_comparison(
    true_X1: np.ndarray,
    true_Y1: np.ndarray,
    pred_X1: np.ndarray,
    pred_Y1: np.ndarray,
    label: int = 0,
    max_slots: int = 500,
    figsize: Tuple[float, float] = (14, 3),
) -> "plt.Figure":
    """
    Compare true vs. reconstructed episode trace for one label.

    Parameters
    ----------
    true_X1, true_Y1 : (T, L) ground-truth finest-scale arrays
    pred_X1, pred_Y1 : (T, L) predicted (will be binarised)
    label            : label index to visualise
    """
    _require_mpl("plot_episode_comparison")
    from ..inference.reconstruction import EpisodeReconstructor

    T = min(true_X1.shape[0], max_slots)
    L = true_X1.shape[1]
    pred_X1_bin = (pred_X1[:T] >= 0.5).astype(int)
    pred_Y1_int = np.round(pred_Y1[:T]).astype(int)

    label_names = [f"l{i}" for i in range(L)]
    rec = EpisodeReconstructor(label_names)

    traces_true = rec.reconstruct(true_X1[:T].astype(int), true_Y1[:T].astype(int))
    traces_pred = rec.reconstruct(pred_X1_bin, pred_Y1_int)

    true_eps = traces_true.get(label, [])
    pred_eps = traces_pred.get(label, [])

    fig, ax = plt.subplots(figsize=figsize)
    for ep in true_eps:
        ax.barh(
            0.6,
            ep.duration,
            left=ep.t_start,
            height=0.3,
            color="steelblue",
            alpha=0.7,
            label="true",
        )
    for ep in pred_eps:
        ax.barh(
            0.2,
            ep.duration,
            left=ep.t_start,
            height=0.3,
            color="tomato",
            alpha=0.7,
            label="predicted",
        )

    handles = [
        mpatches.Patch(color="steelblue", label="True episodes"),
        mpatches.Patch(color="tomato", label="Predicted episodes"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    ax.set_yticks([0.2, 0.6])
    ax.set_yticklabels(["Predicted", "True"], fontsize=8)
    ax.set_xlabel("Finest-scale slot", fontsize=9)
    ax.set_title(f"Episode trace comparison  label ℓ={label}", fontsize=10)
    ax.set_xlim(0, T)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# pyAgrum-based graph rendering
# ---------------------------------------------------------------------------

# Lazy import for pyAgrum
try:
    import pyagrum as _gum

    _PYAGRUM_AVAILABLE = True
except ImportError:
    _PYAGRUM_AVAILABLE = False


def plot_graph_pyagrum(
    graph,  # MultiScaleDBNGraph
    scale: int = 1,
    M: int = 2,
    size: str = "6",
) -> str:
    """
    Render the 2-TBN at scale *k* using pyAgrum's graphviz backend.

    Parameters
    ----------
    graph : MultiScaleDBNGraph
    scale : scale index (1-indexed).
    M     : number of regime states.
    size  : graphviz size attribute.

    Returns
    -------
    dot_source : str - DOT source that can be rendered with graphviz.
        In a Jupyter notebook, use `IPython.display.SVG(...)` or
        `pyagrum.lib.notebook.showBN(bn)` for interactive display.
    """
    if not _PYAGRUM_AVAILABLE:
        raise ImportError(
            "plot_graph_pyagrum() requires pyAgrum. Install: pip install pyAgrum"
        )

    bn = graph.to_pyagrum_bn(scale, M=M)
    dot = bn.toDot()
    return dot


def show_regime_cpt(
    graph,  # MultiScaleDBNGraph
    regime_params,  # RegimeParams
    scale: int = 1,
    ctx_t: Optional[np.ndarray] = None,
) -> str:
    """
    Return a human-readable string of the regime transition CPT.

    Parameters
    ----------
    graph : MultiScaleDBNGraph
    regime_params : RegimeParams
    scale : scale index.
    ctx_t : optional context vector.

    Returns
    -------
    table_str : formatted CPT table.
    """
    if not _PYAGRUM_AVAILABLE:
        raise ImportError(
            "show_regime_cpt() requires pyAgrum. Install: pip install pyAgrum"
        )
    from ..core.pyagrum_utils import fill_regime_cpt

    M = regime_params.M
    bn = graph.to_pyagrum_bn(scale, M=M)
    fill_regime_cpt(bn, regime_params, ctx_t=ctx_t)
    return str(bn.cpt("S_t"))
