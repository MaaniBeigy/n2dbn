"""
core/mask.py
============
The observation mask, its reduction up the ladder and its forward dilation.

Three arrays live here and none of them folds into another.  `observed` is one
flag per finest slot and says whether the record would have shown a start had
one occurred, which is provenance and no function of the data.  `observed_k` is
its AND-reduction into a scale-k epoch, so an epoch counts as observed when
every slot under it is.  `rows_k` is `observed_k` eroded forward by the largest
lag read at that scale, which is the row set a likelihood term may run over,
since an epoch whose lagged feature reaches into a gap carries a feature nobody
recorded.

The risk set of `model/intensity.py` is a fourth array with its own subject.  It
says a start was impossible because the label was already running, an absence
the model explains with a start probability of zero.  A masked slot holds no
observation to explain.  The two meet at `intensity.usable_slots` and nowhere
else, and the dilation below acts on the mask alone.

Only the finest mask is ever stored.  Every coarser one is derived here, which
leaves `check_ladder` one thing to refuse: a coarse epoch holding both observed
and unobserved slots.  Its stored count is then a sum over the observed children
alone while nothing in the container says so, and every term that reads a parent
total would take that partial sum for the whole.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .typed_graph import TypedEdge


def reduce_by(flags: np.ndarray, who: np.ndarray, n_epochs: int) -> np.ndarray:
    """
    The flags AND-reduced into the epochs `who` names, which is the definition.

    Args:
        flags: (T,) bool, one flag per finer epoch.
        who: (T,) int, the coarser epoch each finer epoch belongs to.
        n_epochs: how many coarser epochs there are.

    Returns:
        (n_epochs,) bool, true where every finer epoch under it is true.
    """
    out = np.ones(int(n_epochs), dtype=bool)
    np.logical_and.at(
        out, np.asarray(who, dtype=np.int64), np.asarray(flags, dtype=bool)
    )
    return out


def observed_ladder(
    observed: np.ndarray, parent_maps: Sequence[Optional[np.ndarray]]
) -> List[np.ndarray]:
    """
    `observed_k` for every scale, finest first, each derived from the one below.

    Args:
        observed: (H_1,) bool, one flag per finest slot.
        parent_maps: the scale-(k+1) epoch of every scale-k epoch, `None` at the
            coarsest scale.
    """
    out = [np.asarray(observed, dtype=bool)]
    for pm in parent_maps:
        if pm is None:
            break
        pm = np.asarray(pm, dtype=np.int64)
        out.append(reduce_by(out[-1], pm, int(np.max(pm, initial=-1)) + 1))
    return out


def dilate_forward(flags: np.ndarray, lag: int) -> np.ndarray:
    """
    One scale's flags eroded forward by `lag` epochs.

    An epoch whose window of `lag` predecessors holds a gap reads a feature
    nobody recorded, so it drops.  An index before the first epoch is a
    different case: `intensity.edge_feature` gives it a zero feature and always
    has, so the leading rows survive and a fully observed mask erodes to itself.
    """
    flags = np.asarray(flags, dtype=bool)
    out = flags.copy()
    for step in range(1, int(lag) + 1):
        shifted = np.ones_like(flags)
        shifted[step:] = flags[: max(flags.size - step, 0)]
        out &= shifted
    return out


def declared_rows(
    observed: np.ndarray, lag_budget: Optional[int], where: str
) -> np.ndarray:
    """
    One scale's scorable rows, eroded by the budget the search declared.

    The budget belongs to the candidate space and never to the model being
    scored.  Eroding by a fitted model's own longest lag would sum a model with
    long lags over fewer rows than a model with short ones, so the model with
    the wider reach would score higher on the row count alone, and every
    comparison between them, the structure guard and the outer monotonicity
    guard would read that difference as fit.

    Args:
        observed: (H_k,) bool, one flag per epoch of the scale.
        lag_budget: the largest lag the search may propose at this scale.
        where: the caller, named in the error a missing budget raises.

    Raises:
        ValueError: if the data carries a gap and no budget was declared.
    """
    flags = np.asarray(observed, dtype=bool)
    if flags.all():
        return flags
    if lag_budget is None:
        raise ValueError(
            f"{where} was handed a dataset with {int((~flags).sum())} unobserved "
            "epochs and no declared lag budget, so the rows it would score "
            "depend on the model it is scoring"
        )
    return dilate_forward(flags, int(lag_budget))


def scorable_rows(
    observed: np.ndarray,
    ancestors: Sequence[np.ndarray],
    budget: Mapping[int, int],
) -> np.ndarray:
    """
    The finest slots a lagged design may be scored on, which is the dilated mask.

    An edge reads its source scale stepped back by its lag, so a slot whose
    source epoch falls in a gap carries a feature nobody recorded.  The mask is
    eroded forward at every scale by the largest lag read there, and the
    conjunction of those row sets at the finest grid is what a term may sum over.

    Args:
        observed: (T_1,) bool, one flag per finest slot.
        ancestors: the scale-k epoch of every finest slot, finest first.
        budget: the largest lag read at each source scale, keyed by scale.
    """
    flags = np.asarray(observed, dtype=bool)
    out = flags.copy()
    for scale, lag in sorted(budget.items()):
        who = np.asarray(ancestors[scale - 1], dtype=np.int64)
        rows = dilate_forward(
            reduce_by(flags, who, int(np.max(who, initial=-1)) + 1), lag
        )
        out &= rows[who]
    return out


def lag_budget(edges: Iterable[TypedEdge]) -> Dict[int, int]:
    """The largest lag these edges read at each source scale, keyed by scale."""
    out: Dict[int, int] = {}
    for edge in edges:
        out[edge.src_scale] = max(out.get(edge.src_scale, 0), edge.lag)
    return out


def parent_set_lag(parent_sets: Mapping[int, Iterable[Tuple[int, int]]]) -> int:
    """The largest lag one scale's parent sets read, which is its lag budget."""
    return max((lag for keys in parent_sets.values() for _, lag in keys), default=0)


def check_ladder(
    observed: np.ndarray, parent_maps: Sequence[Optional[np.ndarray]]
) -> None:
    """
    Refuse a coarse epoch that holds both observed and unobserved finest slots.

    The exposure such a block needs is a piece of modeling of its own, and this
    refusal is what keeps it from being assumed.

    Raises:
        ValueError: when any epoch above the finest scale is partly observed.
    """
    flags = np.asarray(observed, dtype=bool)
    for k, pm in enumerate(parent_maps, start=1):
        if pm is None:
            break
        pm = np.asarray(pm, dtype=np.int64)
        size = int(np.max(pm, initial=-1)) + 1
        every = reduce_by(flags, pm, size)
        some = np.zeros(size, dtype=bool)
        np.logical_or.at(some, pm, flags)
        partial = np.flatnonzero(some & ~every)
        if partial.size:
            raise ValueError(
                f"{partial.size} epoch(s) of scale {k + 1}, the first of them "
                f"{int(partial[0])}, hold both observed and unobserved finest "
                "slots; their counts and durations are sums over the observed "
                "children alone and no reader of a parent total can tell such a "
                "sum from a whole one"
            )
        flags = every
