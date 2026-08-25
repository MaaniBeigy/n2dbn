"""
learning/evidence.py
====================
One coarse-to-fine posterior sweep, and the log evidence it produces.

Why this exists
---------------
Scoring a fitted model means running the E-step once at every scale, coarsest
first, handing each scale's posterior down to the next as its parent.  That
sweep was written out separately at every place that scores something: the
held-out objective, the evaluation protocol, each baseline wrapper and the
observed-log experiment.  The copies agreed on the loop.  They disagreed on the
two settings that change the answer, and the disagreement was invisible because
each copy stated its choice as a literal.

The two settings
----------------
`couple_scales` says whether a parent posterior reaches a finer scale, and
`estep` says which E-step configuration runs.  Both belong to the fit.  A model
trained with the coupling switched off is a different model from one trained
with it on, and a score taken under the other setting describes a model nobody
fitted.  Both carry the names and the defaults `TopDownSEM` uses, so an
estimator can answer for itself and the score follows the fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from ..core.graph import MultiScaleDBNGraph
from ..core.variables import MultiScaleData
from ..inference.posterior import PosteriorSummaries
from ..model.params import ScaleParams
from .estep import EStep


@dataclass(frozen=True)
class ChainEvidence:
    """Posterior summaries of one coarse-to-fine sweep and their log evidence."""

    summaries: List[PosteriorSummaries]

    @property
    def log_Z(self) -> List[float]:
        """Log evidence contributed by each scale, index 0 is scale 1."""
        return [float(q.log_Z) for q in self.summaries]

    @property
    def total(self) -> float:
        """Log evidence of the whole model, summed over scales."""
        return float(sum(self.log_Z))


def evidence_chain(
    data: MultiScaleData,
    params_list: List[ScaleParams],
    graph: MultiScaleDBNGraph,
    *,
    estep: Optional[EStep] = None,
    couple_scales: bool = True,
    lag_budget: Optional[Sequence[int]] = None,
) -> ChainEvidence:
    """
    Run the E-step at every scale, coarsest first, and keep what it produced.

    Args:
        data: the observed data every scale is scored on.
        params_list: parameters per scale, index 0 is scale 1.
        graph: the structure the E-step runs under.
        estep: the configured E-step to run.  `EStep()` is what a `TopDownSEM`
            built with no `estep_kwargs` fits with, so that is the default.
        couple_scales: if False, no parent posterior reaches a finer scale,
            which is how a model fitted with the coupling off has to be scored.
        lag_budget: the largest lag the search may propose at each scale, index
            0 is scale 1.  A masked dataset needs it, because eroding the mask
            by the scored model's own lags would let a model with a longer
            reach sum over fewer rows and score higher for it.

    Returns:
        The per-scale summaries and their log evidence, index 0 is scale 1.
    """
    run = estep if estep is not None else EStep()
    if lag_budget is not None:
        run.lag_budget = list(lag_budget)
    summaries: List[PosteriorSummaries] = []
    # The parent travels in a local, which is what lets the coarsest scale run
    # through the same statement as every other one.  It is None on the first
    # pass, and a single-scale model never reaches a second one.
    parent: Optional[PosteriorSummaries] = None
    for k in range(data.K, 0, -1):
        q = run.run(k, data, params_list[k - 1], graph, parent)
        summaries.append(q)
        parent = q if couple_scales else None
    summaries.reverse()
    return ChainEvidence(summaries=summaries)


def scoring_settings(method: object) -> Tuple[Optional[EStep], bool]:
    """The E-step and the cross-scale coupling an estimator was fitted with."""
    # TopDownSEM carries both as attributes and each baseline wrapper derives
    # them from the keyword arguments it forwards, so one reader answers for
    # every estimator that gets scored.  An object carrying neither falls back
    # to the defaults TopDownSEM itself uses.
    return getattr(method, "estep", None), bool(getattr(method, "couple_scales", True))
