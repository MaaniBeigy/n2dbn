"""
src/evaluation/arm1_report.py
=============================
The numbers the observational arm reports, read off its two stored runs.

Two runs make the arm.  The fit run searches one graph over the balanced panel
and grades it on the falsification ladder; the round-trip run draws logs back
out of that fit and compares them with the panel on the nonstationarity grid.
Everything here reads those two stored records and returns rows.  No measurement
happens here, so a printed table and the run behind it cannot disagree.

Three readings are stated here, because a table that omits them says more than
the runs do.

*A row of the grid carries evidence only where it passes its own noise.*  The
positive control grades one replicate of the fitted leg against the other four.
A row it fails has a tolerance narrower than the replicate spread, so the fitted
leg failing that same row says nothing about the fit.  `evidence_bearing`
removes those rows before the leg is graded, and reports both counts.

*The per-label volume is the marginal rate the grid never reads.*  Every
marginal form but A12 divides by the series mean and A12 reads the share of
empty epochs, so a leg drawing a label at half its real rate can still pass
every cell it is tested at.  The volume rows are that missing comparison, and
they come from the tripwire the draw already runs.

*A stratum is named by the edge, not by the scale.*  A relation between two
distinct labels and a label's relation to its own past pay different prices and
answer different questions, so the per-scale counts carry both.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..core.typed_graph import TypedEdge
from ..data.label_axis import LabelAxis

#: The scales the fit searches, coarsest last, as the ladder names them.
SCALE_NAMES: Dict[int, str] = {1: "minute", 2: "hour", 3: "day", 4: "week"}

#: The leg whose graph is the one the arm claims, and the control it is read
#: against.  The empty leg carries an intercept and a participant level.
HOME = "home"
EMPTY = "empty"

#: The three kinds of source an accepted edge can read, which the two price
#: strata do not separate.
SELF = "self"
EXOGENOUS = "exogenous"
CROSS_LABEL = "cross_label"
SOURCE_KINDS = (EXOGENOUS, SELF, CROSS_LABEL)


def edge_name(edge: TypedEdge, target: int, axis: LabelAxis) -> str:
    """One accepted edge, written the way both arm 1 runs record it."""
    source = axis.exogenous if edge.is_exogenous else axis.targets
    return (
        f"{source[edge.src_label]} -> {axis.targets[target]} "
        f"(k={edge.src_scale}, {edge.src_quantity}, lag={edge.lag})"
    )


def endpoints(edge: str) -> Tuple[str, str]:
    """The source and target labels of a stored edge string.

    Args:
        edge: a stored edge, laid out as `source -> target (k=2, level, lag=0)`.

    Returns:
        The source and the target.

    Raises:
        ValueError: when the string carries no arrow or no parenthesis.
    """
    if " -> " not in edge or " (" not in edge:
        raise ValueError(f"not a stored edge string: {edge!r}")
    source, rest = edge.split(" -> ", 1)
    return source.strip(), rest.split(" (", 1)[0].strip()


def is_self(edge: str) -> bool:
    """Whether a stored edge reads the target's own past."""
    source, target = endpoints(edge)
    return source == target


def source_kind(edge: str, exogenous: Sequence[str]) -> str:
    """Which of the three kinds of source a stored edge reads.

    The price has two strata, a label's own past against everything else, so a
    run reports one `cross` count over two kinds of source that answer different
    questions.  A clock column is a deterministic function of the epoch; an edge
    between two behavior labels is the claim the model exists to be able to
    make.
    """
    source, target = endpoints(edge)
    if source == target:
        return SELF
    return EXOGENOUS if source in set(exogenous) else CROSS_LABEL


def edges_by_scale(fit: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """One row per searched scale, counting the edges bought at it by source kind."""
    exogenous = fit["provenance"]["exogenous"]
    counts: Dict[int, Dict[str, int]] = {
        int(k): {kind: 0 for kind in SOURCE_KINDS} for k in fit["search"]["scales"]
    }
    for record in fit["edges"]:
        scale = int(record["scale"])
        row = counts.setdefault(scale, {kind: 0 for kind in SOURCE_KINDS})
        row[source_kind(record["edge"], exogenous)] += 1
    rows = []
    for scale in sorted(counts):
        row = counts[scale]
        rows.append(
            {
                "scale": scale,
                "name": SCALE_NAMES.get(scale, str(scale)),
                **row,
                "cross": row[EXOGENOUS] + row[CROSS_LABEL],
                "total": sum(row.values()),
            }
        )
    return rows


def receipts(fit: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Every accepted edge with its coefficient, its E-value and its reselection rate."""
    frequency = fit["rungs"]["R9"]["frequency"]
    exogenous = fit["provenance"]["exogenous"]
    rows = []
    for record in fit["edges"]:
        source, target = endpoints(record["edge"])
        rows.append(
            {
                "edge": record["edge"],
                "source": source,
                "target": target,
                "kind": source_kind(record["edge"], exogenous),
                "scale": int(record["scale"]),
                "lag": int(record["lag"]),
                "self": is_self(record["edge"]),
                "coefficient": float(record["coefficient"]),
                "standardized": float(record["standardized"]),
                "e_value": float(record["e_value"]),
                "future_lag_p": float(record["future_lag_p"]),
                "reselected": float(frequency.get(record["edge"], 0.0)),
            }
        )
    rows.sort(key=lambda r: -r["reselected"])
    return rows


def rung_summary(fit: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The falsification rungs the fit run grades, each with its own denominator."""
    rungs = fit["rungs"]
    r1, r2, r3, r9 = rungs["R1"], rungs["R2"], rungs["R3"], rungs["R9"]
    return [
        {
            "rung": "R1",
            "reads": "declared protocol negatives selected",
            "hits": int(r1["hits"]),
            "denominator": int(r1["denominator"]),
            "passed": bool(r1["passed"]),
        },
        {
            "rung": "R2",
            "reads": "sham-label candidates selected",
            "hits": int(r2["hits"]),
            "denominator": int(r2["denominator"]),
            "passed": bool(r2["passed"]),
        },
        {
            "rung": "R3",
            "reads": "declared positives recovered",
            "hits": int(r3["hits"]),
            "denominator": int(r3["denominator"]),
            "passed": None,
        },
        {
            "rung": "R3r",
            "reads": "positives reachable at the searched scales",
            "hits": int(r3["reachable_hits"]),
            "denominator": int(r3["reachable"]),
            "passed": None,
        },
        {
            "rung": "R9",
            "reads": f"edges reselected at or above {r9['threshold']}",
            "hits": len(r9["selected"]),
            "denominator": len(fit["edges"]),
            "passed": None,
        },
    ]


def ablation_rows(fit: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The calendar-node ablation, each arm read against the home arm's objective."""
    ablation = fit["calendar_ablation"]
    home = float(ablation[HOME]["objective"])
    rows = []
    for arm in ("absent", HOME, "every"):
        record = ablation[arm]
        rows.append(
            {
                "arm": arm,
                "candidates": int(record["n_candidates"]),
                "accepted": int(record["n_accepted"]),
                "objective": float(record["objective"]),
                "against_home": float(record["objective"]) - home,
            }
        )
    return rows


def volume_rows(trip: Mapping[str, Any], leg: str = HOME) -> List[Dict[str, Any]]:
    """Per label, the real panel's episode count against one replicate of the draw."""
    record = trip["legs"][leg]
    rows = []
    for label in sorted(record["real_per_label"]):
        real = float(record["real_per_label"][label])
        drawn = float(record["drawn_per_label_per_replicate"][label])
        rows.append(
            {
                "label": label,
                "real": real,
                "drawn": drawn,
                "ratio": drawn / real if real else float("nan"),
                "clips": int(record["eta_clips_per_label"].get(label, 0)),
            }
        )
    return rows


def clip_rows(trip: Mapping[str, Any], leg: str = HOME) -> List[Dict[str, Any]]:
    """Per label, which end of the representable range the predictor was capped at.

    A cap at the low end says the label never fires in that slot; a cap at the
    high end says it fires in every free one.  A leg recorded before the counter
    was split by end carries a total alone and reports it with no breakdown.
    """
    record = trip["legs"][leg]
    ends = record.get("eta_clips_by_end", {})
    rows = []
    for label in sorted(record["eta_clips_per_label"]):
        row = {
            "label": label,
            "total": int(record["eta_clips_per_label"][label]),
            "by_end": {k: int(v) for k, v in ends.get(label, {}).items()},
        }
        rows.append(row)
    return rows


def volume_agreement(
    trip: Mapping[str, Any],
    leg: str = HOME,
    labels: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """How far one leg's drawn volumes sit from the panel's, in log ratio.

    The measure is the mean absolute log of the drawn-to-real ratio, which is
    symmetric in the direction of the miss and so does not reward a leg for
    drawing too few of one label and too many of another.

    Args:
        trip: the stored round-trip record.
        leg: which leg to read.
        labels: the labels to average over; all of them when not given.

    Raises:
        ValueError: when a named label has no episodes on the real panel, since
            its ratio has no value.
    """
    rows = {row["label"]: row for row in volume_rows(trip, leg)}
    chosen = list(rows) if labels is None else list(labels)
    empty = [label for label in chosen if not rows[label]["real"]]
    if empty:
        raise ValueError(
            f"the real panel carries no {empty[0]!r}, so its volume ratio has no value"
        )
    gaps = [abs(math.log(rows[label]["ratio"])) for label in chosen]
    return {
        "leg": leg,
        "labels": chosen,
        "mean_abs_log_ratio": sum(gaps) / len(gaps),
        "worst": max(zip(gaps, chosen))[1],
    }


def leg_rows(trip: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """One row per leg: what it bought, what it drew and how the panel graded it."""
    rows = []
    for leg, record in trip["legs"].items():
        panel = trip["panels"][leg]
        rows.append(
            {
                "leg": leg,
                "edges": int(record["n_edges"]),
                "regime_states": int(record["regime_states"]),
                "reconciliation_error": float(record["reconciliation_error"]),
                "tested": int(panel["n_tested"]),
                "failed_cells": int(panel["n_tested"]) - _passed_cells(panel),
                "failed_rows": list(panel["failed"]),
                "passed": bool(panel["passed"]),
            }
        )
    return rows


def _passed_cells(panel: Mapping[str, Any]) -> int:
    """How many tested cells of one panel their own tolerance accepted."""
    return sum(
        int(form["tested"]) - int(form["failed"]) for form in panel["by_form"].values()
    )


def evidence_bearing(trip: Mapping[str, Any], leg: str = HOME) -> Dict[str, Any]:
    """The rows the panel tests, less the rows that fail their own replicate noise."""
    panel = trip["panels"][leg]
    noise = set(trip["rows_failing_their_own_noise"])
    tested = [row for row in panel["tested_rows"] if row not in noise]
    failed = [row for row in panel["failed"] if row not in noise]
    return {
        "leg": leg,
        "tested_rows": list(panel["tested_rows"]),
        "noise_rows": sorted(noise),
        "evidence_rows": tested,
        "failed_rows": failed,
        "passed_rows": [row for row in tested if row not in failed],
    }


def changepoint_rows(
    trip: Mapping[str, Any], scale: str = "week", form: str = "A08"
) -> List[Dict[str, Any]]:
    """The changepoint agreement rows for one form at one scale, one per label.

    Args:
        trip: the stored round-trip record.
        scale: the calendar scale the segmentation was read at.
        form: the grid row, which is also the cost the segmentation minimized.
    """
    rows = [
        row
        for row in trip["changepoints"]
        if row["scale"] == scale and row["form"] == form
    ]
    rows.sort(key=lambda row: row["series"])
    return [
        {
            "series": row["series"],
            "form": row["form"],
            "cost": row["cost"],
            "annotation_error": float(row["annotation_error"]),
            "f1": float(row["f1_margin"]),
            "covering": float(row["covering"]),
            "hausdorff": float(row["hausdorff"]),
            "jump_ks": float(row["jump_ks"]),
            "segment_ks": float(row["segment_ks"]),
        }
        for row in rows
    ]


def heterogeneity_rows(trip: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The between-participant dispersion rows, with the refused labels named."""
    rows = []
    for record in trip["heterogeneity"]:
        if "refused" in record:
            rows.append({"series": record["series"], "refused": record["refused"]})
            continue
        rows.append(
            {
                "series": record["series"],
                "real": float(record["real_cqv"]),
                "real_lower": float(record["real_lower"]),
                "real_upper": float(record["real_upper"]),
                "drawn": float(record["drawn_cqv"]),
                "drawn_lower": float(record["drawn_lower"]),
                "drawn_upper": float(record["drawn_upper"]),
                "comparable": bool(record["comparable"]),
                "separation": float(record["separation"]),
                "sparse_gap": float(record["sparse_gap"]),
            }
        )
    return rows


def control_rows(trip: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Each declared control with the rows it was meant to break and did not."""
    return [
        {
            "control": name,
            "passed": bool(record["passed"]),
            "broke": list(record["broke"]),
            "missed": list(record["missed"]),
            "spurious": list(record["spurious"]),
            "untested": list(record["untested"]),
        }
        for name, record in trip["controls"].items()
    ]


def separation_summary(trip: Mapping[str, Any]) -> Dict[str, Any]:
    """What the fitted graph buys over the empty one, on the grid's own terms."""
    home, empty = trip["panels"][HOME], trip["panels"][EMPTY]
    return {
        "home_failed_cells": int(home["n_tested"]) - _passed_cells(home),
        "home_tested_cells": int(home["n_tested"]),
        "empty_failed_cells": int(empty["n_tested"]) - _passed_cells(empty),
        "empty_tested_cells": int(empty["n_tested"]),
        "empty_fails_the_panel": bool(trip["empty_graph_fails_the_panel"]),
        "separating_rows": list(trip["rows_separating_the_fit_from_the_empty_graph"]),
    }


def scales_reached(fit: Mapping[str, Any]) -> Sequence[int]:
    """The scales the search bought at least one edge at, coarsest last."""
    return sorted({int(record["scale"]) for record in fit["edges"]})


def edge_agreement(
    fit: Mapping[str, Any], trip: Mapping[str, Any], leg: str = HOME
) -> Dict[str, Any]:
    """Whether the fit run and one leg of the round trip found the same graph.

    The two runs search the same panel with the same declared settings, so a
    disagreement is a disagreement about the code that produced them and not
    about the data.  A leg whose record predates the readable edge name carries
    none, and the comparison then reports what it could read.
    """
    searched = {record["edge"] for record in fit["edges"]}
    drawn = {
        record["name"] for record in trip["legs"][leg]["edges"] if "name" in record
    }
    return {
        "leg": leg,
        "searched": len(searched),
        "drawn": len(drawn),
        "agreeing": len(searched & drawn),
        "only_searched": sorted(searched - drawn),
        "only_drawn": sorted(drawn - searched),
    }
