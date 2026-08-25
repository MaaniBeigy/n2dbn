"""
evaluation/sufficiency.py
=========================
Whether a scale carries enough data to support a structural claim.

A coarse scale of a nested hierarchy holds a small fraction of the epochs the
finest one does, and a run that searches there anyway returns an empty graph,
which reads as "no structure here" when it means "not enough data to tell".
This reports the difference before any fitting happens.

What it reports is measured rather than assumed, and the measurement changed
the design.  Coarse-edge recovery was swept against the coarsest epoch count on
a fixed generator and rises steadily rather than switching on at a threshold, so
a scale carries an **expected recall** rather than a verdict of yes or no.  Only
the bottom of that curve is a genuine refusal: below it the sweep recovered
nothing and still returned edges, which is the one regime where a graph is worse
than no graph.  A scale is judged on the epochs a fit will actually see, which is
the training split rather than the whole series.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Set

from ..core.variables import MultiScaleData

# Coarse-edge recall against epochs per candidate edge, swept on a fixed
# generator over three families.  Recovery rises steadily rather than switching
# on at a threshold, so a scale is described by the recall its epoch count
# implies rather than by a yes or a no.  Points are (epochs per candidate,
# recall) and come from the sweep recorded in the project plan.
RECALL_CURVE = (
    (1.6, 0.000),
    (3.4, 0.125),
    (6.8, 0.250),
    (13.4, 0.375),
    (23.6, 0.625),
    (47.0, 0.750),
)

# Below this, the sweep recovered nothing and still returned edges, which is the
# one regime where a graph is worse than no graph.
MIN_EPOCHS_PER_EDGE = 3.0

# A floor below which no candidate count makes a scale usable.  Three epochs is
# already the least a split will produce, so anything at or under it carries no
# structural information at all.
ABSOLUTE_MIN_EPOCHS = 12


def expected_recall(epochs_per_candidate: float) -> float:
    """Coarse-edge recall this many epochs per candidate has bought, when swept."""
    xs = [x for x, _ in RECALL_CURVE]
    ys = [y for _, y in RECALL_CURVE]
    if epochs_per_candidate <= xs[0]:
        return ys[0]
    if epochs_per_candidate >= xs[-1]:
        return ys[-1]
    for (x0, y0), (x1, y1) in zip(RECALL_CURVE, RECALL_CURVE[1:]):
        if x0 <= epochs_per_candidate <= x1:
            span = x1 - x0
            w = (epochs_per_candidate - x0) / span if span else 0.0
            return y0 + w * (y1 - y0)
    return ys[-1]  # pragma: no cover - the loop covers the interior


def epochs_for_recall(
    target_recall: float,
    n_labels: int,
    max_lag: int,
) -> float:
    """
    Epochs a scale needs before the sweep saw the given recall there.

    The inverse of the measured curve, which is the form a study design reads:
    how long must the series be at this scale to expect to find this much.
    Returns infinity above the recall the sweep reached, since claiming more
    than was measured would be inventing it.
    """
    n_cand = n_candidate_edges(n_labels, max_lag)
    if n_cand <= 0:
        return 0.0
    if target_recall > RECALL_CURVE[-1][1]:
        return float("inf")
    for (x0, y0), (x1, y1) in zip(RECALL_CURVE, RECALL_CURVE[1:]):
        if y0 <= target_recall <= y1:
            span = y1 - y0
            w = (target_recall - y0) / span if span else 0.0
            return (x0 + w * (x1 - x0)) * n_cand
    return RECALL_CURVE[0][0] * n_cand


def horizon_for_recall(
    target_recall: float,
    n_labels: int,
    max_lag: int,
    epochs_below: int,
) -> float:
    """
    Finest-scale epochs a scale needs, given how many of them one of its own holds.

    `epochs_below` is the product of the branching factors beneath the scale, so
    a coarsest scale of a `[10, 6]` hierarchy passes 60.  A run collects at the
    finest resolution, so this is the number a study is actually committing to.
    """
    return epochs_for_recall(target_recall, n_labels, max_lag) * int(epochs_below)


# A lagged edge at a scale is only visible inside one subject's own series: a
# subject contributes no transition at all unless it spans more than the lag.
# Pooling adds subjects, and it cannot buy a subject the epochs it does not have.
MIN_EPOCHS_PER_SUBJECT_OVER_LAG = 2


def min_epochs_per_subject(max_lag: int) -> int:
    """Fewest epochs one subject needs before it shows a lagged transition at all."""
    return int(max_lag) + MIN_EPOCHS_PER_SUBJECT_OVER_LAG


def subjects_for_recall(
    target_recall: float,
    n_labels: int,
    max_lag: int,
    epochs_per_subject: int,
) -> float:
    """
    Subjects a pooled fit needs, given how many epochs each one contributes.

    A pooled fit shares one graph and one parameter set, so subjects and epochs
    enter the same total and a coarse scale can be reached by many short series
    where one long series is impossible to collect.  Returns infinity when a
    subject is too short to show a lagged transition, since no number of such
    subjects carries the dependency, and when the target is above the recall the
    sweep reached.
    """
    if epochs_per_subject < min_epochs_per_subject(max_lag):
        return float("inf")
    needed = epochs_for_recall(target_recall, n_labels, max_lag)
    if needed == float("inf"):
        return float("inf")
    if needed <= 0.0:
        return 0.0
    return needed / float(epochs_per_subject)


def pooled_design(
    target_recall: float,
    n_labels: int,
    max_lag: int,
    epochs_per_subject: Sequence[int],
) -> List[Dict[str, float]]:
    """
    The subjects needed at each per-subject series length, as a design table.

    One row per candidate study shape.  `subjects` is rounded up, since half a
    subject collects nothing, and `total_epochs` is what the pooled fit actually
    sees.
    """
    import math

    out: List[Dict[str, float]] = []
    for per in epochs_per_subject:
        s = subjects_for_recall(target_recall, n_labels, max_lag, per)
        rounded = float("inf") if s == float("inf") else float(math.ceil(s))
        total = float("inf") if rounded == float("inf") else rounded * per
        out.append(
            {
                "epochs_per_subject": float(per),
                "subjects": rounded,
                "total_epochs": total,
            }
        )
    return out


@dataclass(frozen=True)
class ScaleVerdict:
    """Whether one scale can support a structural claim, and why."""

    k: int
    n_epochs: int
    n_candidates: int
    epochs_per_candidate: float
    expected_recall: float
    supported: bool
    reason: str

    def __str__(self) -> str:
        state = "supported" if self.supported else "NOT supported"
        return (
            f"scale {self.k}: {state}, {self.n_epochs} epochs, "
            f"{self.n_candidates} candidates, {self.epochs_per_candidate:.2f} each,"
            f" expect recall about {self.expected_recall:.2f} ({self.reason})"
        )


def n_candidate_edges(n_labels: int, max_lag: int) -> int:
    """How many lagged parents one label may choose between at a scale."""
    return int(n_labels) * int(max_lag)


def scale_verdict(
    k: int,
    n_epochs: int,
    n_labels: int,
    max_lag: int,
    min_epochs_per_edge: float = MIN_EPOCHS_PER_EDGE,
) -> ScaleVerdict:
    """Judge one scale on the epochs it has against the choices it must make."""
    n_cand = n_candidate_edges(n_labels, max_lag)
    per = float(n_epochs) / n_cand if n_cand else float("inf")
    recall = expected_recall(per)
    if n_epochs < ABSOLUTE_MIN_EPOCHS:
        supported, reason = False, f"fewer than {ABSOLUTE_MIN_EPOCHS} epochs"
    elif per < min_epochs_per_edge:
        supported, reason = (
            False,
            "under the count at which the sweep recovered anything, and it "
            "returned edges there anyway",
        )
    else:
        supported, reason = True, "above the count at which recovery begins"
    return ScaleVerdict(k, int(n_epochs), n_cand, per, recall, supported, reason)


def ledger(
    epochs_per_scale: Sequence[int],
    n_labels: int,
    max_lag: Sequence[int],
    min_epochs_per_edge: float = MIN_EPOCHS_PER_EDGE,
) -> List[ScaleVerdict]:
    """One verdict per scale, finest first, before anything is fitted."""
    if len(epochs_per_scale) != len(max_lag):
        raise ValueError(
            f"scale count mismatch: {len(epochs_per_scale)} epoch counts, "
            f"{len(max_lag)} lag bounds"
        )
    return [
        scale_verdict(k, n, n_labels, lag, min_epochs_per_edge)
        for k, (n, lag) in enumerate(zip(epochs_per_scale, max_lag), start=1)
    ]


def data_ledger(
    data: MultiScaleData,
    max_lag: Sequence[int],
    min_epochs_per_edge: float = MIN_EPOCHS_PER_EDGE,
) -> List[ScaleVerdict]:
    """
    One verdict per scale of a dataset, on the epochs that dataset holds.

    The epochs a fit sees are the ones in the object handed to it, so a caller
    that has already split its series passes the training part and gets the
    ledger of what will actually be searched.
    """
    epochs = [data.get_X(k).shape[0] for k in range(1, data.K + 1)]
    return ledger(epochs, data.L, max_lag, min_epochs_per_edge)


def refused_scales(verdicts: Sequence[ScaleVerdict]) -> Set[int]:
    """The scales whose epoch count carries no structural claim."""
    return {v.k for v in verdicts if not v.supported}


def supported_scales(verdicts: Sequence[ScaleVerdict]) -> Dict[int, bool]:
    """The verdict per scale index, for a caller that wants only the answer."""
    return {v.k: v.supported for v in verdicts}


def report(verdicts: Sequence[ScaleVerdict]) -> str:
    """The ledger as text, for a run to print before it searches."""
    lines = ["scale support, judged before fitting"]
    lines.extend(f"  {v}" for v in verdicts)
    unsupported = [v.k for v in verdicts if not v.supported]
    if unsupported:
        names = ", ".join(str(k) for k in unsupported)
        lines.append(
            f"  no structural claim is supported at scale {names};"
            " a graph there would be reported without evidence"
        )
    return "\n".join(lines)
