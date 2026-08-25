"""
utils/budget.py
===============
A declared ceiling on every run, and the partial result a run returns at it.

An experiment that states no ceiling runs until something kills it, and what a
kill leaves behind is a directory of half-written files and no record of how far
the grid got.  Three days of wall clock were spent that way.  Every run here
declares a wall-clock budget, a peak-resident-memory cap and a cell count before
it starts, and on reaching any of them it stops, keeps what it finished and
writes `budget_exhausted` with the cells it never reached.  A partial result
that names its own gap is usable.

Cells are checkpointed one at a time.  A finished cell is appended to a ledger
as soon as it lands, so an interrupted run resumes at the first cell the ledger
does not hold and repeats nothing.  The key of a cell is the caller's, which
lets a grid change without invalidating the cells it kept.

Peak memory is read from the kernel's own high-water mark, so a spike between
two checks is caught the same as a steady climb.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

#: What a run reports when it stopped at one of its own declared limits.
EXHAUSTED = "budget_exhausted"
COMPLETE = "complete"

_GIB = 1024.0**2  # VmHWM is in kibibytes, so this converts to gibibytes


def peak_rss_gb(status_path: str = "/proc/self/status") -> float:
    """
    The high-water mark of resident memory, in gibibytes.

    Falls back to the resource module where the kernel does not publish the mark,
    which is every platform that is not Linux.
    """
    try:
        with open(status_path) as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return float(line.split()[1]) / _GIB
    except OSError:
        pass
    import resource

    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / _GIB


@dataclass(frozen=True)
class RunBudget:
    """
    What one run may spend before it stops and reports what it has.

    Attributes
    ----------
    seconds     : wall clock the whole run may take
    peak_rss_gb : resident memory the process may reach
    cells       : cells the run may finish
    """

    seconds: float
    peak_rss_gb: float
    cells: int

    def __post_init__(self) -> None:
        for name, value in (
            ("seconds", self.seconds),
            ("peak_rss_gb", self.peak_rss_gb),
            ("cells", self.cells),
        ):
            if value <= 0:
                raise ValueError(f"a budget's {name} is positive; got {value}")

    @classmethod
    def quick(cls) -> "RunBudget":
        """The grid continuous integration runs, small enough to finish every time."""
        return cls(seconds=120.0, peak_rss_gb=4.0, cells=8)

    @classmethod
    def full(cls) -> "RunBudget":
        """The default ceiling of a published run, two hours and thirty-two gibibytes."""
        return cls(seconds=7200.0, peak_rss_gb=32.0, cells=100000)


@dataclass
class BudgetLedger:
    """What a run has spent, and which limit stopped it."""

    budget: RunBudget
    started: float = field(default_factory=time.monotonic)
    finished: int = 0
    reason: str = COMPLETE
    stopped_by: Optional[str] = None

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def exceeded(self) -> Optional[str]:
        """Which declared limit the run has reached, or nothing when it has not."""
        if self.finished >= self.budget.cells:
            return "cells"
        if self.elapsed >= self.budget.seconds:
            return "seconds"
        if peak_rss_gb() >= self.budget.peak_rss_gb:
            return "peak_rss_gb"
        return None

    def record(self, not_reached: Sequence[str]) -> Dict[str, object]:
        """The run's own account of what it spent and what it did not reach."""
        return {
            "status": self.reason,
            "limit": None if self.reason == COMPLETE else self.stopped_by,
            "elapsed_s": round(self.elapsed, 3),
            "peak_rss_gb": round(peak_rss_gb(), 3),
            "cells_finished": self.finished,
            "cells_not_reached": list(not_reached),
            "budget": {
                "seconds": self.budget.seconds,
                "peak_rss_gb": self.budget.peak_rss_gb,
                "cells": self.budget.cells,
            },
        }


@dataclass
class Checkpoint:
    """A ledger of finished cells, appended to as each one lands."""

    path: str

    def done(self) -> Dict[str, object]:
        """Every cell the ledger already holds, keyed by the caller's own key."""
        out: Dict[str, object] = {}
        if not os.path.exists(self.path):
            return out
        with open(self.path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                out[row["cell"]] = row["value"]
        return out

    def append(self, cell: str, value: object) -> None:
        """Write one finished cell, flushed so a kill after it keeps it."""
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.path, "a") as handle:
            handle.write(json.dumps({"cell": cell, "value": value}) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def run_cells(
    cells: Sequence[Tuple[str, object]],
    work: Callable[[object], object],
    budget: RunBudget,
    checkpoint: Optional[Checkpoint] = None,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    """
    Run a grid one cell at a time, under a budget, resuming from what is done.

    Args:
        cells: (key, argument) pairs, where the key names the cell in the ledger.
        work: what one cell's argument is turned into.
        checkpoint: where finished cells are kept, or nothing to keep none.

    Returns:
        The value of every finished cell keyed by its own key, and the run's
        record, whose status is `budget_exhausted` when a limit stopped it.
    """
    done: Dict[str, object] = checkpoint.done() if checkpoint else {}
    ledger = BudgetLedger(budget)
    ledger.finished = len(done)
    pending: List[str] = []

    for index, (key, argument) in enumerate(cells):
        if key in done:
            continue
        limit = ledger.exceeded()
        if limit is not None:
            ledger.reason = EXHAUSTED
            ledger.stopped_by = limit
            pending = [name for name, _ in cells[index:] if name not in done]
            break
        done[key] = work(argument)
        ledger.finished += 1
        if checkpoint:
            checkpoint.append(key, done[key])
    return done, ledger.record(pending)


def add_budget_arguments(parser) -> None:
    """
    Give a command the quick grid and the three ceilings.

    Each ceiling defaults to nothing, so a value the caller states wins over the
    grid it was stated beside and a grid alone supplies all three.
    """
    parser.add_argument("--quick", action="store_true", help="run the small grid")
    parser.add_argument("--budget-seconds", type=float, default=None)
    parser.add_argument("--budget-rss-gb", type=float, default=None)
    parser.add_argument("--budget-cells", type=int, default=None)


def budget_from_args(args) -> RunBudget:
    """The budget a parsed command line declares, stated values over the grid's."""
    grid = RunBudget.quick() if getattr(args, "quick", False) else RunBudget.full()
    stated = {
        "seconds": getattr(args, "budget_seconds", None),
        "peak_rss_gb": getattr(args, "budget_rss_gb", None),
        "cells": getattr(args, "budget_cells", None),
    }
    return RunBudget(
        **{
            name: getattr(grid, name) if value is None else value
            for name, value in stated.items()
        }
    )
