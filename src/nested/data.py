"""Observation audits and raw-log round trips with a fixed cohort schedule."""

from __future__ import annotations

import csv
import json
from collections import Counter
from contextlib import ExitStack
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from ..core.scales import ScaleHierarchy
from ..core.variables import MultiScaleData
from .model import Record

LIFESNAPS_OBSERVATION = "coincident-annotation-groups-v1"


def group_coincident_annotations(records):
    """One start per participant/label/minute, with all duration marks summed.

    This defines grouped observations; it does not identify duplicate episodes.
    Duration arrays, contexts, calendars and masks retain their original values.
    All coarser counts are rebuilt from the binary finest-scale group indicators.
    """
    out = []
    for rec in records:
        rec.validate(count_only=True)
        d = rec.data
        x, y = d.X[0], d.Y[0]
        if np.any(y < x) or np.any((x == 0) & (y != 0)):
            raise ValueError(
                f"{rec.name}: grouping cannot repair invalid annotation durations"
            )
        counts = [(x > 0).astype(np.int64)]
        for k, parent in enumerate(d.hierarchy.parent_maps[:-1]):
            up = np.zeros_like(d.X[k + 1])
            np.add.at(up, parent, counts[-1])
            counts.append(up)
        grouped = Record(rec.name, replace(d, X=counts))
        grouped.validate()
        out.append(grouped)
    return out


def _annotation_audit(path, records, axis, schedule, examples, destination):
    """Preserve every source row and verify its membership against raw arrays."""
    from ..data.lifesnaps import GYM_LABEL, GYM_MODALITIES

    by_name = {rec.name: rec for rec in records}
    columns = {label: i for i, label in enumerate(records[0].data.label_names)}
    status_counts = Counter()
    membership = {}
    destination = Path(destination) if destination is not None else None
    if destination is not None:
        if destination.resolve() == Path(path).resolve():
            raise ValueError("annotation audit must not overwrite the source CSV")
        destination.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        source = csv.DictReader(
            stack.enter_context(open(path, newline="", encoding="utf-8"))
        )
        writer = None
        if destination is not None:
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            output = stack.enter_context(
                temporary.open("w", newline="", encoding="utf-8")
            )
            writer = csv.DictWriter(
                output,
                fieldnames=[
                    "csv_line",
                    "status",
                    "modeled_label",
                    "panel_epoch",
                    "group_id",
                    "group_annotation_count",
                    "group_count",
                    "duration_mark",
                    "group_duration",
                ]
                + ["raw_" + name for name in source.fieldnames],
            )
            writer.writeheader()
        for line, row in enumerate(source, start=2):
            label = GYM_LABEL if row["event"] in GYM_MODALITIES else row["event"]
            label = axis.grouped.get(label, label)
            key = (
                row["person_id"],
                int(row["day_index"]),
                int(row["start_min"]),
                label,
            )
            # The input CSV contains whole-minute durations. This is exactly
            # the legacy seconds-to-minutes conversion, including its floor.
            mark = max(1, int(row["duration_min"]))
            entry = dict(csv_line=line, modeled_label=label)
            rec = by_name.get(key[0])
            if rec is None:
                status = "excluded_participant"
            else:
                t = (key[1] - schedule.offsets[rec.name]) * 1440 + key[2]
                entry["panel_epoch"] = t
                if not 0 <= t < len(rec.data.X[0]):
                    status = "outside_panel_window"
                elif label not in columns:
                    status = "context"
                else:
                    ell = columns[label]
                    count = int(rec.data.X[0][t, ell])
                    duration = int(rec.data.Y[0][t, ell])
                    status = (
                        "modeled" if rec.data.get_observed(1)[t] else "unobserved_week"
                    )
                    entry.update(
                        group_id=json.dumps(key, separators=(",", ":")),
                        group_annotation_count=count,
                        group_count=1,
                        duration_mark=mark,
                        group_duration=duration,
                    )
                    own = membership.setdefault((rec.name, t, ell), [0, 0])
                    own[0] += 1
                    own[1] += mark
            entry["status"] = status
            status_counts[status] += 1
            if key in examples:
                examples[key].append(
                    dict(
                        csv_line=line,
                        raw_label=row["event"],
                        duration_min=row["duration_min"],
                        duration_mark=mark,
                        start_datetime=row.get("start_datetime"),
                        end_datetime=row.get("end_datetime"),
                    )
                )
            if writer is not None:
                writer.writerow(entry | {"raw_" + k: v for k, v in row.items()})
        expected = sum(np.count_nonzero(rec.data.X[0]) for rec in records)
        if len(membership) != expected:
            raise ValueError("annotation audit does not cover every panel group")
        for (name, t, ell), (count, duration) in membership.items():
            d = by_name[name].data
            if count != d.X[0][t, ell] or duration != d.Y[0][t, ell]:
                raise ValueError(
                    f"annotation audit disagrees with panel at {name}, {t}, {ell}"
                )
    if destination is not None:
        temporary.replace(destination)
    return {
        "file": destination.name if destination is not None else None,
        "rows": sum(status_counts.values()),
        "status_counts": dict(status_counts),
        "group_membership_verified": True,
        "original_columns": "all input columns preserved with raw_ prefix",
    }


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


def lifesnaps(path, participants=51, weeks=8, *, annotation_path=None):
    """Read LifeSnaps as binary groups of coincident mapped annotations.

    The experiment supplies annotation_path to retain every source CSV row in
    a separate audit trail. Neither the source CSV nor legacy readers is edited.
    """
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
    # Retain evidence about multiplicities before defining grouped starts.
    limit = 20
    by_label = {label: [] for label in records[0].data.label_names}
    values = {}
    for rec in records:
        for ell, label in enumerate(rec.data.label_names):
            remaining = limit - len(by_label[label])
            if remaining <= 0:
                continue
            times = np.flatnonzero(
                (rec.data.X[0][:, ell] > 1) & rec.data.get_observed(1)
            )[:remaining]
            for t in times:
                day, minute = divmod(int(t), 1440)
                key = (
                    rec.name,
                    report.schedule.offsets[rec.name] + day,
                    minute,
                    label,
                )
                by_label[label].append(key)
                values[key] = {
                    "count": int(rec.data.X[0][t, ell]),
                    "duration": int(rec.data.Y[0][t, ell]),
                }
    # Interleave labels so both the terminal preview and the saved report show
    # rare collisions instead of letting survey annotations occupy every slot.
    examples = {
        keys[i]: [] for i in range(limit) for keys in by_label.values() if i < len(keys)
    }
    preparation["annotation_audit"] = _annotation_audit(
        path, records, axis, report.schedule, examples, annotation_path
    )
    preparation["collision_examples"] = [
        dict(
            participant=key[0],
            raw_day_index=key[1],
            start_min=key[2],
            modeled_label=key[3],
            observed_count=values[key]["count"],
            observed_duration=values[key]["duration"],
            original_annotations=value,
        )
        for key, value in examples.items()
    ]
    preparation["collision_examples_limit_per_label"] = limit
    preparation["collision_examples_complete_by_label"] = {
        label: len(keys) == preparation["collision_cells"][label]
        for label, keys in by_label.items()
    }
    preparation["collision_examples_basis"] = "before grouping"
    preparation["before_grouping"] = audit(records)
    grouped = group_coincident_annotations(records)
    by_label = {}
    for ell, label in enumerate(records[0].data.label_names):
        before = sum(
            int(r.data.X[0][r.data.get_observed(1), ell].sum()) for r in records
        )
        after = sum(
            int(r.data.X[0][r.data.get_observed(1), ell].sum()) for r in grouped
        )
        duration = sum(
            int(r.data.Y[0][r.data.get_observed(1), ell].sum()) for r in records
        )
        by_label[label] = {
            "annotations_before": before,
            "groups_after": after,
            "count_reduction": before - after,
            "coincident_groups": preparation["collision_cells"][label],
            "duration_before": duration,
            "duration_after": duration,
        }
    preparation["observation_definition"] = LIFESNAPS_OBSERVATION
    preparation["grouping"] = {
        "policy": LIFESNAPS_OBSERVATION,
        "unit": "one occupied participant-label-minute",
        "duration": "sum of individual positive integer annotation marks",
        "basis": "observed slots of the selected panel",
        "duplicates_inferred": False,
        "by_label": by_label,
    }
    preparation.update(audit(grouped))
    preparation["duration_convention"] = (
        "sum of individually recoded annotation marks; each mark rounds "
        "seconds/60 to nearest even with minimum one minute"
    )
    return grouped, preparation, report.schedule


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
    """Export modeled starts and full marks, including marks past a block.

    LifeSnaps rows describe annotation groups; they do not reconstruct the
    original group's individual annotations or their multiplicities.
    """
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
                    raise ValueError(
                        "modeled-start export requires binary finest starts"
                    )
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
