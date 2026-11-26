"""Count annotations, occupied minutes and duration recodings on the fitted panel."""

import argparse
import json
from pathlib import Path

import numpy as np

from src.data.label_axis import collapse_labels
from src.data.lifesnaps import RAW_CSV, read_lifesnaps
from src.data.lifesnaps_panel import balanced_panel, panel_provenance


def audit(path):
    """Measure multiplicity and recoding on the same observation schedule as the fit."""
    cohort, axis, report, _, _ = balanced_panel(path=path)
    result = panel_provenance(cohort, axis, report, path)
    rows = {
        label: dict(
            annotations=0,
            occupied_minutes=0,
            multiple_start_minutes=0,
            maximum_starts=0,
            recoded_nonpositive_durations=0,
        )
        for label in axis.targets
    }
    for _, data in cohort.stream():
        counts = data.X[0][data.get_observed(1)]
        for index, row in enumerate(rows.values()):
            values = counts[:, index]
            row["annotations"] += int(values.sum())
            row["occupied_minutes"] += int(np.count_nonzero(values))
            row["multiple_start_minutes"] += int(np.count_nonzero(values > 1))
            row["maximum_starts"] = max(
                row["maximum_starts"], int(values.max(initial=0))
            )
    events, _ = read_lifesnaps(path)
    collapsed, _ = collapse_labels(events)
    for event in collapsed:
        if (
            event.injected
            and event.label in rows
            and event.day_index in report.schedule.days.get(event.unit, ())
        ):
            rows[event.label]["recoded_nonpositive_durations"] += 1
    result["labels"] = rows
    result["retained_participant_days"] = sum(map(len, report.schedule.days.values()))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=RAW_CSV)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/lifesnaps-observation-audit")
    )
    args = parser.parse_args()
    result = audit(args.csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf8"
    )
