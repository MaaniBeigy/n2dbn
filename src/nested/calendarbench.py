"""CalendarBench observation groups, retaining the original episode ledger."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from contextlib import ExitStack
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np

from ..data.dynamic_patterns import (
    BENCHMARK_ORIGIN,
    VERBATIM,
    build_population,
    read_subject_event_log,
)
from ..data.lifesnaps_panel import shared_calendar
from .data import audit, group_coincident_annotations
from .model import Record

CALENDARBENCH_OBSERVATION = "coincident-annotation-groups-v1"
DURATION_CONVENTION = "ceil duration seconds / 60; minimum one minute per annotation"
MISSINGNESS = "declared common study window, with padding excluded"


def _placement(start, end):
    return (
        int((start - BENCHMARK_ORIGIN).total_seconds() // 60),
        max(1, math.ceil((end - start).total_seconds() / 60)),
    )


def _annotations(path, destination, records, groups, examples):
    """Write every source row, including excluded subjects and repeated rows."""
    by_name = {r.name: r for r in records}
    seen, marks, statuses = Counter(), Counter(), Counter()
    destination = Path(destination) if destination is not None else None
    if destination is not None:
        if destination.resolve() == Path(path).resolve():
            raise ValueError("annotation audit must not overwrite the source TSV")
        destination.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        reader = csv.DictReader(
            stack.enter_context(open(path, newline="", encoding="utf-8")),
            delimiter="\t",
        )
        writer = None
        if destination is not None:
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            handle = stack.enter_context(
                temporary.open("w", newline="", encoding="utf-8")
            )
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "tsv_line",
                    "status",
                    "modeled_label",
                    "panel_epoch",
                    "group_id",
                    "group_annotation_count",
                    "group_count",
                    "duration_mark",
                    "group_duration",
                ]
                + ["raw_" + name for name in reader.fieldnames],
            )
            writer.writeheader()
        for line, row in enumerate(reader, start=2):
            name, label = row["person_id"].strip(), row["label"]
            start, end = (
                datetime.fromisoformat(row[k]) for k in ("start_iso", "end_iso")
            )
            t, mark = _placement(start, end)
            key = (name, t, label)
            entry = dict(tsv_line=line, modeled_label=label, panel_epoch=t)
            if name not in by_name:
                status = "excluded_participant"
            else:
                count, duration = groups[key]
                status = (
                    "modeled"
                    if by_name[name].data.get_observed(1)[t]
                    else "unobserved_padding"
                )
                entry.update(
                    group_id=json.dumps(key, separators=(",", ":")),
                    group_annotation_count=count,
                    group_count=1,
                    duration_mark=mark,
                    group_duration=duration,
                )
                seen[key] += 1
                marks[key] += mark
            entry["status"] = status
            statuses[status] += 1
            if key in examples:
                examples[key].append(dict(tsv_line=line, duration_mark=mark, **row))
            if writer is not None:
                writer.writerow(entry | {"raw_" + k: v for k, v in row.items()})
        if set(seen) != set(groups) or any(
            (seen[key], marks[key]) != tuple(value) for key, value in groups.items()
        ):
            raise ValueError(
                "CalendarBench annotation ledger does not match the input groups"
            )
    if destination is not None:
        temporary.replace(destination)
    return dict(
        file=destination.name if destination is not None else None,
        rows=sum(statuses.values()),
        status_counts=dict(statuses),
        group_membership_verified=True,
        original_columns="all input columns preserved with raw_ prefix",
    )


def calendarbench(path, participants=30, *, annotation_path=None):
    """Keep binary starts and summed marks on the original shared calendar.

    Build and group one participant at a time: no second full raw cohort is
    retained. Starts at different minutes remain distinct even when durations
    overlap. The legacy IDENTIFIABLE overlap-merging policy is not used.
    """
    subjects = read_subject_event_log(str(path))
    subjects = {key: subjects[key] for key in sorted(subjects)[:participants]}
    if not subjects:
        raise ValueError("CalendarBench requires at least one participant")
    labels = sorted({e.label for events in subjects.values() for e in events})
    groups = {}
    for name, events in subjects.items():
        for event in events:
            t, mark = _placement(event.start, event.end)
            own = groups.setdefault((name, t, event.label), [0, 0])
            own[0] += 1
            own[1] += mark

    records, raw_audits = [], []
    totals = {
        label: dict(
            annotations_before=0,
            groups_after=0,
            count_reduction=0,
            coincident_groups=0,
            duration_before=0,
            duration_after=0,
        )
        for label in labels
    }
    for name, data, _ in build_population(subjects, labels, policy=VERBATIM):
        raw = Record(name, replace(data, C=shared_calendar(data.hierarchy.H[-1])))
        raw_audits.append(audit([raw]))
        grouped = group_coincident_annotations([raw])[0]
        mask = data.get_observed(1)
        for ell, label in enumerate(labels):
            before, after = data.X[0][mask, ell], grouped.data.X[0][mask, ell]
            values = totals[label]
            values["annotations_before"] += int(before.sum())
            values["groups_after"] += int(after.sum())
            values["count_reduction"] += int(before.sum() - after.sum())
            values["coincident_groups"] += int((before > 1).sum())
            values["duration_before"] += int(data.Y[0][mask, ell].sum())
            values["duration_after"] += int(grouped.data.Y[0][mask, ell].sum())
        records.append(grouped)

    by_name = {rec.name: rec for rec in records}
    columns = {label: ell for ell, label in enumerate(labels)}
    if sum(np.count_nonzero(r.data.X[0]) for r in records) != len(groups):
        raise ValueError("CalendarBench grouped arrays omit source starts")
    for (name, t, label), (_, duration) in groups.items():
        d, ell = by_name[name].data, columns[label]
        if d.X[0][t, ell] != 1 or d.Y[0][t, ell] != duration:
            raise ValueError("CalendarBench grouped arrays disagree with source marks")

    before = dict(raw_audits[0])
    before.update(
        participants=len(records),
        observed_participant_slots=sum(
            a["observed_participant_slots"] for a in raw_audits
        ),
        window_slots_per_participant=[
            n for a in raw_audits for n in a["window_slots_per_participant"]
        ],
        collision_cells={label: totals[label]["coincident_groups"] for label in labels},
        max_starts_per_slot={
            label: max(a["max_starts_per_slot"][label] for a in raw_audits)
            for label in labels
        },
        valid=all(a["valid"] for a in raw_audits),
        support_errors=[e for a in raw_audits for e in a["support_errors"]],
        duration_convention=DURATION_CONVENTION,
        missingness=MISSINGNESS,
    )
    limit, keys_by_label = 20, {label: [] for label in labels}
    for key, (count, _) in groups.items():
        name, t, label = key
        if (
            count > 1
            and by_name[name].data.get_observed(1)[t]
            and len(keys_by_label[label]) < limit
        ):
            keys_by_label[label].append(key)
    examples = {
        keys[i]: []
        for i in range(limit)
        for keys in keys_by_label.values()
        if i < len(keys)
    }
    ledger = _annotations(path, annotation_path, records, groups, examples)
    preparation = audit(records)
    preparation.update(
        source=str(path),
        observation_definition=CALENDARBENCH_OBSERVATION,
        duration_convention="sum of annotation marks: " + DURATION_CONVENTION,
        missingness=MISSINGNESS,
        before_grouping=before,
        annotation_audit=ledger,
        grouping=dict(
            policy=CALENDARBENCH_OBSERVATION,
            unit="one occupied participant-label-minute",
            duration="sum of individual positive integer annotation marks",
            basis="observed slots of the selected panel",
            duplicates_inferred=False,
            by_label=totals,
        ),
        collision_examples_basis="before grouping",
        collision_examples_limit_per_label=limit,
        collision_examples_complete_by_label={
            label: len(keys) == totals[label]["coincident_groups"]
            for label, keys in keys_by_label.items()
        },
        collision_examples=[
            dict(
                participant=key[0],
                panel_epoch=key[1],
                modeled_label=key[2],
                observed_count=groups[key][0],
                observed_duration=groups[key][1],
                original_annotations=value,
            )
            for key, value in examples.items()
        ],
    )
    return records, preparation
