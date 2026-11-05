"""Observation audits and raw-log round trips with a fixed cohort schedule."""

from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from ..core.scales import ScaleHierarchy
from ..core.variables import MultiScaleData
from .model import Record


def from_finest(
    name, counts, durations, branching, context=None, observed=None, labels=None
):
    hierarchy = ScaleHierarchy.from_uniform_branching(len(counts), list(branching))
    X, Y = [np.asarray(counts, dtype=np.int64)], [np.asarray(durations, dtype=np.int64)]
    for k in range(hierarchy.K - 1):
        for arrays in (X, Y):
            up = np.zeros((hierarchy.H[k + 1], counts.shape[1]), dtype=np.int64)
            np.add.at(up, hierarchy.parent_maps[k], arrays[-1])
            arrays.append(up)
    if context is None:
        context = [np.empty((n, 0)) for n in hierarchy.H]
    return Record(
        str(name),
        MultiScaleData(
            hierarchy,
            labels or [f"label{i}" for i in range(counts.shape[1])],
            X,
            Y,
            context,
            observed=observed,
        ),
    )


def audit(records, count_only=False):
    labels = records[0].data.label_names
    collisions = np.zeros(len(labels), dtype=int)
    maxima = np.zeros(len(labels), dtype=int)
    observed_slots, errors = 0, []
    for rec in records:
        x = rec.data.X[0][rec.data.get_observed(1)]
        observed_slots += len(x)
        collisions += (x > 1).sum(axis=0)
        if len(x):
            maxima = np.maximum(maxima, x.max(axis=0))
        try:
            rec.validate(count_only)
        except ValueError as exc:
            errors.append(str(exc))
    return {
        "participants": len(records),
        "observed_participant_slots": observed_slots,
        "window_slots_per_participant": [r.data.hierarchy.H[0] for r in records],
        "collision_cells": dict(zip(labels, collisions.tolist())),
        "max_starts_per_slot": dict(zip(labels, maxima.tolist())),
        "valid": not errors,
        "support_errors": errors,
        "count_only": count_only,
        "duration_convention": "round seconds/60 to nearest even; minimum one minute",
        "missingness": "event-presence day coverage; exclude incomplete weeks; fixed schedule in simulation",
    }


def lifesnaps(path, participants=51, weeks=8):
    from ..data.lifesnaps_panel import PANEL_EXOGENOUS, balanced_panel

    if not Path(path).is_file():
        raise FileNotFoundError(
            f"LifeSnaps CSV missing: {path}. Run lifesnaps-restore and lifesnaps-build first."
        )
    cohort, axis, report, cohort_report, collapse = balanced_panel(
        participants=participants,
        path=str(path),
        weeks=weeks,
    )
    records = [Record(name, cohort.open(name)) for name in cohort.subjects]
    preparation = audit(records)
    preparation.update(
        {
            "source": str(path),
            "coverage_before_week_mask": [s.coverage for s in report.subjects],
            "slots_removed_by_week_mask": list(report.rounded_slots),
            "raw_participants": cohort_report.participants_raw,
            "collapsed_labels": list(collapse.group),
            "context_names": list(PANEL_EXOGENOUS),
            "raw_preparation": {
                "rows_raw": cohort_report.n_rows_raw,
                "rows_kept": cohort_report.n_rows_kept,
                "instant_durations_recoded": cohort_report.injected_instants,
                "noninstant_nonpositive_durations_recoded": cohort_report.repaired_zeros,
                "exercise_labels_merged": cohort_report.merged_into_gym,
            },
        }
    )
    # Show which original annotations collided after the declared label map.
    # This makes a support failure reviewable without guessing that coincident
    # annotations are duplicate episodes or deleting distinct events.
    from ..data.lifesnaps import GYM_LABEL, GYM_MODALITIES

    examples = {}
    for rec in records:
        for t, ell in np.argwhere(
            (rec.data.X[0] > 1) & rec.data.get_observed(1)[:, None]
        ):
            if len(examples) >= 20:
                break
            day, minute = divmod(int(t), 1440)
            key = (
                rec.name,
                report.schedule.offsets[rec.name] + day,
                minute,
                rec.data.label_names[ell],
            )
            examples[key] = []
    if examples:
        with open(path, newline="", encoding="utf-8") as handle:
            for line, row in enumerate(csv.DictReader(handle), start=2):
                label = GYM_LABEL if row["event"] in GYM_MODALITIES else row["event"]
                label = axis.grouped.get(label, label)
                key = (
                    row["person_id"],
                    int(row["day_index"]),
                    int(row["start_min"]),
                    label,
                )
                if key in examples:
                    examples[key].append(
                        {
                            "csv_line": line,
                            "raw_label": row["event"],
                            "duration_min": row["duration_min"],
                        }
                    )
    preparation["collision_examples"] = [
        dict(
            participant=key[0],
            raw_day_index=key[1],
            start_min=key[2],
            modeled_label=key[3],
            original_annotations=value,
        )
        for key, value in examples.items()
    ]
    return records, preparation, report.schedule


def with_sham(records, seed=0):
    """Independent binary label, with its own positive duration marks."""
    rng = np.random.default_rng(seed)
    out = []
    for rec in records:
        d = rec.data
        x = np.column_stack([d.X[0], rng.binomial(1, 0.001, len(d.X[0]))])
        y = np.column_stack([d.Y[0], x[:, -1]])
        branching = [d.hierarchy.H[k] // d.hierarchy.H[k + 1] for k in range(d.K - 1)]
        out.append(
            from_finest(
                rec.name, x, y, branching, d.C, d.observed, d.label_names + ["sham"]
            )
        )
    return out


def write_log(path, records, schedule=None):
    """Export starts with full duration marks, including end times past a block."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "person_id",
                "event",
                "day_index",
                "start_min",
                "duration_min",
                "start_datetime",
            ],
        )
        writer.writeheader()
        for rec in records:
            offset = 0 if schedule is None else schedule.offsets[rec.name]
            day_zero = (
                (
                    rec.data.calendar.origin
                    if rec.data.calendar
                    else datetime(2020, 1, 6)
                )
                if schedule is None
                else datetime.combine(schedule.day_zero[rec.name], datetime.min.time())
            )
            for t, ell in np.argwhere(
                (rec.data.X[0] > 0) & rec.data.get_observed(1)[:, None]
            ):
                if rec.data.X[0][t, ell] != 1:
                    raise ValueError("raw episode export requires binary finest starts")
                day, minute = divmod(int(t), 1440)
                when = day_zero + timedelta(days=offset + day, minutes=minute)
                writer.writerow(
                    {
                        "person_id": rec.name,
                        "event": rec.data.label_names[ell],
                        "day_index": offset + day,
                        "start_min": minute,
                        "duration_min": int(rec.data.Y[0][t, ell]),
                        "start_datetime": when.strftime("%m/%d/%Y %H:%M"),
                    }
                )


def read_log(path, templates, schedule=None):
    """Read the exported normalized labels using the declared roster and mask.

    Participants with no generated events remain in the roster. Scheduled context
    is retained from the design; its observation schedule is never inferred from
    whether a simulated day happened to contain an event.
    """
    by_name = {r.name: r for r in templates}
    arrays = {
        name: (np.zeros_like(r.data.X[0]), np.zeros_like(r.data.Y[0]))
        for name, r in by_name.items()
    }
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = row["person_id"]
            if name not in by_name:
                raise ValueError(f"unknown participant {name}")
            rec = by_name[name]
            ell = rec.data.label_names.index(row["event"])
            offset = 0 if schedule is None else schedule.offsets[name]
            t = (int(row["day_index"]) - offset) * 1440 + int(row["start_min"])
            if not 0 <= t < len(arrays[name][0]) or not rec.data.get_observed(1)[t]:
                raise ValueError(
                    "exported start falls outside the declared observation schedule"
                )
            duration = int(row["duration_min"])
            if duration < 1:
                raise ValueError("generated duration marks must be positive")
            arrays[name][0][t, ell] += 1
            arrays[name][1][t, ell] += duration
    out = []
    for rec in templates:
        d = rec.data
        branching = [d.hierarchy.H[k] // d.hierarchy.H[k + 1] for k in range(d.K - 1)]
        result = from_finest(
            rec.name, *arrays[rec.name], branching, d.C, d.observed, d.label_names
        )
        result.data.calendar = d.calendar
        result.validate()
        out.append(result)
    return out


def check_round_trip(original, returned):
    if [r.name for r in original] != [r.name for r in returned]:
        raise AssertionError("round trip changed the roster")
    for left, right in zip(original, returned):
        for k in range(left.data.K):
            mask = left.data.get_observed(k + 1)
            np.testing.assert_array_equal(mask, right.data.get_observed(k + 1))
            for a, b in ((left.data.X, right.data.X), (left.data.Y, right.data.Y)):
                np.testing.assert_array_equal(a[k][mask], b[k][mask])
    return True


def evaluation_view(records):
    """Retain at most three coarsest levels for form evaluation, never fitting.

    Count/duration volumes must be accumulated from the original observed minute
    rows first. In particular, a partly observed hour cannot replace that volume.
    """
    result = []
    for record in records:
        d = record.data
        first = max(0, d.K - 3)
        if first == 0:
            result.append(record)
            continue
        hierarchy = ScaleHierarchy(
            d.K - first, d.hierarchy.H[first:], d.hierarchy.parent_maps[first:]
        )
        result.append(
            Record(
                record.name,
                MultiScaleData(
                    hierarchy,
                    d.label_names,
                    d.X[first:],
                    d.Y[first:],
                    d.C[first:],
                    observed=d.get_observed(first + 1).copy(),
                ),
            )
        )
    return result
