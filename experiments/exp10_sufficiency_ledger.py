"""
experiments/exp10_sufficiency_ledger.py
=======================================
Experiment 10: what a log's epochs support, judged before anything is fitted.

Every other experiment reports what a fit found.  This one reports what the
epochs could have supported, which is the quantity a reader needs before an
empty scale can be read at all.  A coarse scale of a nested hierarchy holds a
small fraction of the epochs the finest one does, and the sweep behind
`src/evaluation/sufficiency.py` found one regime where a search recovers nothing
and returns edges anyway.  A scale in that regime is refused, and the refusal
names a property of the epochs the scale holds.

Two datasets are read, and they are chosen so the pair carries the finding.  The
first is the observed log itself, on the ladder the fit uses.  The second is the
same log clipped to a short horizon, which starves the coarse scales while
leaving the label axis, the slot length and the branching untouched, so the only
thing that moved between the two ledgers is the number of epochs.  The run
fails if the starved copy is not refused, since a screen that passes a dataset
built to fail it is measuring nothing.

The ledger judges the training split as well as the whole series, because the
split is what a fit sees.  Both are reported: the whole series states what the
collection bought, and the split states what the search will work with.

Run
---
    python -m experiments.exp10_sufficiency_ledger

    python -m experiments.exp10_sufficiency_ledger
        --csv data/dynamic_patterns/event_log.tsv
        --branching 60,24,7 --starved-minutes 60480
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import timedelta
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data.dynamic_patterns import read_subject_event_log
from src.data.event_log import ObservedEvent, build_multiscale_data, read_event_log
from src.evaluation.sufficiency import ScaleVerdict, data_ledger, refused_scales, report
from src.utils.budget import (
    BudgetLedger,
    RunBudget,
    add_budget_arguments,
    budget_from_args,
)

# The column a multi-subject log names its subject in, read the same pooled way
# the fitting experiment reads it.
SUBJECT_COL = "person_id"


def _read_log(path: str) -> List[ObservedEvent]:
    """Every episode of a log, pooled over the subjects it names."""
    with open(path, encoding="utf-8") as f:
        header = f.readline()
    delimiter = "\t" if "\t" in header else ","
    if SUBJECT_COL not in header.strip().split(delimiter):
        return read_event_log(path)
    by_subject = read_subject_event_log(path, delimiter=delimiter)
    return [e for events in by_subject.values() for e in events]


def _clip(events: Sequence[ObservedEvent], minutes: int) -> List[ObservedEvent]:
    """
    The episodes of a log that finish within the first `minutes` of it.

    Clipping on the end keeps every kept episode whole, so the starved copy is
    a shorter log of the same kind and no duration is truncated into it.
    """
    limit = min(e.start for e in events) + timedelta(minutes=int(minutes))
    return [e for e in events if e.end <= limit]


def _rows(verdicts: Sequence[ScaleVerdict]) -> List[Dict[str, object]]:
    """The ledger as records, for the results file."""
    return [
        {
            "k": v.k,
            "n_epochs": v.n_epochs,
            "n_candidates": v.n_candidates,
            "epochs_per_candidate": round(v.epochs_per_candidate, 4),
            "expected_recall": round(v.expected_recall, 4),
            "supported": v.supported,
            "reason": v.reason,
        }
        for v in verdicts
    ]


def _ledger_case(
    name: str,
    events: Sequence[ObservedEvent],
    slot_minutes: int,
    branching: Sequence[int],
    max_lag: int,
    train_frac: float,
) -> Dict[str, object]:
    """Lay one log on the ladder and judge both the whole series and the split."""
    data, _calendar, labels = build_multiscale_data(
        events,
        slot_minutes=slot_minutes,
        branching=list(branching),
        finest_binary=False,
    )
    lags = [max_lag] * data.K
    rest = (1.0 - train_frac) / 2.0
    train, _valid, _test = data.split(fracs=(train_frac, rest, rest))
    whole = data_ledger(data, lags)
    split = data_ledger(train, lags)

    print(f"\n[{name}] {len(events)} episodes, {len(labels)} labels, K={data.K}")
    print(f"  whole series, epochs {[v.n_epochs for v in whole]}")
    print(report(whole))
    print(f"  training split, epochs {[v.n_epochs for v in split]}")
    print(report(split))
    return {
        "name": name,
        "n_events": len(events),
        "labels": labels,
        "K": data.K,
        "max_lag": lags,
        "whole_series": _rows(whole),
        "training_split": _rows(split),
        "refused_whole_series": sorted(refused_scales(whole)),
        "refused_training_split": sorted(refused_scales(split)),
    }


def main(
    csv: str = "data/events.csv",
    slot_minutes: int = 1,
    branching: str = "10,6",
    max_lag: int = 2,
    starved_minutes: int = 360,
    train_frac: float = 0.7,
    output_dir: str = "results/exp10",
    budget: Optional[RunBudget] = None,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    os.makedirs(output_dir, exist_ok=True)
    # The grid is the observed log and its starved copy, and the screen means
    # nothing without both, so the budget here records what the run spent and
    # stops nothing.
    ledger = BudgetLedger(budget or RunBudget.full())
    branch = [int(b) for b in branching.split(",") if b.strip()]
    events = _read_log(csv)

    observed = _ledger_case(csv, events, slot_minutes, branch, max_lag, train_frac)
    starved = _ledger_case(
        f"{csv} clipped to {starved_minutes} minutes",
        _clip(events, starved_minutes),
        slot_minutes,
        branch,
        max_lag,
        train_frac,
    )

    # A screen that passes a dataset built to fail it has measured nothing, so
    # the starved verdict is asserted here and the run stops without it.
    assert starved["refused_whole_series"], (
        "the starved copy was supported at every scale, so the ledger is reading "
        "something other than the epoch count it claims to read"
    )
    # Clipping the horizon lowers the epoch count at every scale and moves
    # nothing else, so a scale the full log was refused at stays refused.
    assert set(starved["refused_whole_series"]) >= set(
        observed["refused_whole_series"]
    ), "the starved copy gained support at a scale the full log was refused at"

    out = {
        "csv": csv,
        "slot_minutes": slot_minutes,
        "branching": branch,
        "starved_minutes": starved_minutes,
        "train_frac": train_frac,
        "observed": observed,
        "starved": starved,
    }
    path = os.path.join(output_dir, "exp10_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    ledger.finished = 2
    with open(os.path.join(output_dir, "exp10_budget.json"), "w") as f:
        json.dump(ledger.record([]), f, indent=2)
    print(f"\nResults saved to {output_dir}/")
    return observed, starved


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/events.csv")
    parser.add_argument("--slot-minutes", type=int, default=1)
    parser.add_argument("--branching", default="10,6")
    parser.add_argument("--max-lag", type=int, default=2)
    parser.add_argument(
        "--starved-minutes",
        type=int,
        default=360,
        help=(
            "horizon the starved copy of the same log is clipped to; it has to "
            "leave six coarsest epochs, which is the fewest a three-way split "
            "fills"
        ),
    )
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--output-dir", default="results/exp10")
    add_budget_arguments(parser)
    args = parser.parse_args()
    main(
        csv=args.csv,
        slot_minutes=args.slot_minutes,
        branching=args.branching,
        max_lag=args.max_lag,
        starved_minutes=args.starved_minutes,
        train_frac=args.train_frac,
        output_dir=args.output_dir,
        budget=budget_from_args(args),
    )
