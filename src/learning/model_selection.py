"""
learning/model_selection.py
===========================
Choose the latent orders and the edge penalty on held-out log evidence.

Why this exists
---------------
The number of regimes and the factor dimension are model orders, not
quantities the fit can read off the data, and a wrong choice is not neutral.
Fitting two regimes to a series that has one lets the regime chain absorb
variation that belongs to the emission, which shows up as edges the structure
step then has to pay for; fitting one regime to a series that has two leaves
persistent drift in the residual, where it looks like a lagged self edge.  A
comparison against a learner with no latent state is only about the score if
the orders were chosen by the same rule that chose everything else, so the
choice is made here, from data, rather than declared.

What is selected and what is not
--------------------------------
`M_k`, `r_k` and the edge penalty `lam` are selected.  The graph budget
(`max_lag`, `max_indegree`) and the block lengths of the hierarchy are not:
both state which lags and which timescales the analyst is asking about, and no
held-out score supplies a question.  Every candidate sees the same data and the
same fitting procedure, so what separates them is the model, not the protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from ..core.graph import MultiScaleDBNGraph
from ..core.variables import MultiScaleData
from ..data.initialize import initialize_from_data
from ..model.params import ScaleParams
from .estep import EStep
from .evidence import evidence_chain, scoring_settings

Candidate = Tuple[Tuple[int, ...], Tuple[int, ...], float]


def held_out_nll_per_scale(
    data: MultiScaleData,
    params_list: List[ScaleParams],
    graph: MultiScaleDBNGraph,
    *,
    estep: Optional[EStep] = None,
    couple_scales: bool = True,
) -> List[float]:
    """
    Negative variational log evidence of `data`, one entry per scale.

    The total this sums to is what a selector reads today, and summing hides
    the only thing that could set a penalty per scale without a true graph:
    each scale contributes its own term, so each can be scored on its own.
    Index 0 is scale 1.

    Args:
        estep: the E-step configuration the model was fitted with.
        couple_scales: whether a parent posterior reaches a finer scale, which
            has to match the setting the model was fitted under.
    """
    chain = evidence_chain(
        data, params_list, graph, estep=estep, couple_scales=couple_scales
    )
    return [-z for z in chain.log_Z]


def held_out_nll(
    data: MultiScaleData,
    params_list: List[ScaleParams],
    graph: MultiScaleDBNGraph,
    *,
    estep: Optional[EStep] = None,
    couple_scales: bool = True,
) -> float:
    """Negative variational log evidence of `data`, summed over scales."""
    return float(
        sum(
            held_out_nll_per_scale(
                data, params_list, graph, estep=estep, couple_scales=couple_scales
            )
        )
    )


@dataclass
class ModelSelection:
    """The selected configuration and the held-out score of every candidate."""

    M_k: List[int]
    r_k: List[int]
    lam: float
    val_nll: float
    table: List[Tuple[Candidate, float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Flat record for a results table."""
        return {
            "M_k": list(self.M_k),
            "r_k": list(self.r_k),
            "lam": float(self.lam),
            "val_nll": float(self.val_nll),
            "n_candidates": len(self.table),
        }


def select_model(
    train: MultiScaleData,
    valid: MultiScaleData,
    sem_factory: Callable[[float], object],
    M_grid: Sequence[int] = (1, 2, 3),
    r_grid: Sequence[int] = (0,),
    lam_grid: Sequence[float] = (2.0,),
    max_lag: Optional[Sequence[int]] = None,
    d_ctx: int = 0,
    seed: int = 0,
) -> ModelSelection:
    """
    Fit one model per configuration and keep the one with the lowest held-out NLL.

    Args:
        train: data the candidates are fitted on.
        valid: held-out data the candidates are scored on.
        sem_factory: given an edge penalty, returns a fresh estimator, so no
            state, damping or learning rate carries between candidates.  Its
            E-step and its cross-scale coupling are what the candidate is then
            scored under.
        M_grid: regime counts to try; the same count is used at every scale.
        r_grid: factor dimensions to try; the same dimension at every scale.
        lam_grid: edge penalties to try.
        max_lag: lag budget per scale, index 0 = scale 1.  Defaults to 1.
        d_ctx: context width the parameters carry.
        seed: fixes the initializer, and with it the whole comparison.

    Returns:
        The selected configuration, its held-out NLL, and the candidate table.

    Raises:
        ValueError: a grid is empty, or no candidate could be fitted.
    """
    if not M_grid or not r_grid or not lam_grid:
        raise ValueError("M_grid, r_grid and lam_grid must each hold a value")

    K = train.K
    lags = list(max_lag) if max_lag is not None else [1] * K
    table: List[Tuple[Candidate, float]] = []
    best: Optional[Tuple[float, List[int], List[int], float]] = None

    for M in M_grid:
        for r in r_grid:
            for lam in lam_grid:
                M_k, r_k = [int(M)] * K, [int(r)] * K
                try:
                    params = initialize_from_data(
                        train, M_k=M_k, r_k=r_k, max_lag=lags, d_ctx=d_ctx, seed=seed
                    )
                    graph = MultiScaleDBNGraph(
                        hierarchy=train.hierarchy,
                        label_names=list(train.label_names),
                    )
                    sem = sem_factory(float(lam))
                    fitted, learned, _ = sem.fit(train, params, graph)
                    sem_estep, sem_couple = scoring_settings(sem)
                    nll = held_out_nll(
                        valid,
                        fitted,
                        learned,
                        estep=sem_estep,
                        couple_scales=sem_couple,
                    )
                except Exception:
                    # A configuration that cannot be fitted is not evidence
                    # against the others, so it is recorded and skipped.
                    nll = float("inf")
                table.append(((tuple(M_k), tuple(r_k), float(lam)), float(nll)))
                if nll < float("inf") and (best is None or nll < best[0]):
                    best = (float(nll), M_k, r_k, float(lam))

    if best is None:
        raise ValueError("no configuration could be fitted on the given data")
    return ModelSelection(
        M_k=best[1], r_k=best[2], lam=best[3], val_nll=best[0], table=table
    )
