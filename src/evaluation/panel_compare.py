"""
evaluation/panel_compare.py
===========================
The nonstationarity grid run on a real panel and the synthetic panels drawn from it.

`nonstationarity` holds the forms, the inclusion rule, the equivalence test and
the negative-control declarations.  `round_trip` produces the synthetic side.
What is here is the loop between them: one series per label per scale, one
scalar per participant per form, and the paired difference the equivalence test
reads.

The participant is the sampling unit and the contrast is between two groups of
them.  A form gives one number per participant on the real log, and one per
participant per replicate on the drawn one, and every replicate's participants
join one drawn group.  A drawn participant carries a level taken from the fitted
roster and is never the twin of a real participant, so the two groups are
independent and the test is a two-group one.  A drawn log handing every
participant the cohort-mean profile therefore fails on the spread as well as on
the mean.

Which scales a form is read at is declared and not chosen.  The minute scale
carries no test at all, and the week scale carries only the three forms that are
defined on eight epochs.  Every tolerance and the power bound are declared here,
before a synthetic log exists, which is what keeps the inclusion rule a
partition of the grid on the real log alone.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..core.cohort import Cohort
from ..data.label_axis import LabelAxis
from .count_hmm import regime_coupling, regime_entropy
from .factor_model import factor_share
from .nonstationarity import (
    GRID,
    JOINT,
    MARGINAL,
    Cell,
    Denominator,
    dependence_distance,
    group_permutation_p,
    group_tost,
    haar_detail_energy,
    harmonic,
    hedges_g,
    inclusion,
    inject,
    lagged_correlation,
    periodogram,
    shuffle_labels,
    shuffle_within_participant,
    theil_sen,
    windowed_correlation,
)
from .omnibus import cost_drop_fraction, window_mmd
from .persistence import persistence_drift
from .placement import placement_concentration
from .seasonality_drift import amplitude_drift
from .segmentation import COST_PARAMETERS, POISSON, SLOPE, VARIANCE, cut_rate
from .trend_shape import adf_test, spline_curvature

#: Which scale of the panel ladder each named calendar scale is.
SCALE_NAMES: Dict[int, str] = {2: "hour", 3: "day", 4: "week"}

#: The calendar period each scale carries, in epochs of that scale.
SCALE_PERIOD: Dict[str, float] = {"hour": 24.0, "day": 7.0, "week": 4.0}

#: The scales the comparison runs at.  The minute scale carries no test.
COMPARED_SCALES: Tuple[int, ...] = (2, 3, 4)

#: The coarsest scale, which carries eight epochs of the panel window.
WEEK = "week"

#: Which scale number the week is.  The heterogeneity protocol is stated on
#: weekly totals whatever scales the grid is run over, so it reads this and never
#: the coarsest scale a given run happened to compare.
WEEK_SCALE: int = next(k for k, name in SCALE_NAMES.items() if name == WEEK)

#: The finest scale the comparison runs at, at 1344 epochs of a panel window.
HOUR = "hour"

#: The rows a fitted regime path is read on.  A regime is a level held across
#: calendar cycles, and the cycle of the hour scale is the day, so a regime
#: there is the daily rhythm A06 reads as a profile and A15 as a placement.
#: Both rows are read from the day scale up.
REGIME_FORMS: Tuple[str, ...] = ("A14", "B5")

#: The forms the week scale admits.  Eight epochs define a trend, a two-level
#: detail profile and a level segmentation, and define nothing else.
WEEK_FORMS: Tuple[str, ...] = ("A01", "A04", "A08")

#: The smallest injection a cell has to catch to be tested, as a share of the
#: series level.
POWER_BOUND = 0.4

#: The injection sizes the calibration sweeps.
INJECTION_SIZES: Tuple[float, ...] = (0.05, 0.1, 0.2, 0.4, 0.8)

#: Replicates are averaged inside a participant, so a difference is paired.
MIN_WINDOW = 4


def _mean(x: np.ndarray) -> float:
    value = float(np.mean(x))
    return value if value else 1.0


def _trend(x: np.ndarray, period: float) -> float:
    """A01: the median pairwise slope over the window, relative to the level."""
    return float(theil_sen(x) * x.size / _mean(x))


def _drift(x: np.ndarray, period: float) -> float:
    """A04: the share of the Haar detail energy the coarsest level holds."""
    levels = max(1, int(math.log2(max(x.size, 2))))
    energy = haar_detail_energy(x, levels)
    total = float(energy.sum())
    return float(energy[-1] / total) if total > 0.0 else 0.0


def _period_power(x: np.ndarray, period: float) -> float:
    """A05: normalized power at the scale's own calendar period."""
    return float(periodogram(x, [period])[0])


def _profile(x: np.ndarray, period: float) -> float:
    """A06: the harmonic amplitude at that period, relative to the level."""
    return float(harmonic(x, period)[0] / _mean(x))


def _zero_share(x: np.ndarray, period: float) -> float:
    """A12: the share of epochs holding nothing."""
    return float(np.mean(x == 0.0))


def _dispersion(x: np.ndarray, period: float) -> float:
    """A13: the variance-to-mean ratio of the whole series."""
    mean = float(np.mean(x))
    return float(np.var(x) / mean) if mean > 0.0 else 0.0


def changepoint_penalty(n: int, cost: str) -> float:
    """The price of one cut, which is what one more segment costs a Bayesian criterion."""
    return COST_PARAMETERS[cost] * math.log(max(n, 2))


def _cuts(cost: str) -> Callable[[np.ndarray, float], float]:
    def scalar(x: np.ndarray, period: float) -> float:
        return cut_rate(x, cost, changepoint_penalty(x.size, cost))

    return scalar


def _unit_root(x: np.ndarray, period: float) -> float:
    """A03: the augmented Dickey-Fuller t-ratio, which is more negative the more the level returns."""
    try:
        return float(adf_test(x).statistic)
    except ValueError:
        # A series the regression refuses carries no evidence about its level,
        # and the cell reports the value a series with no level structure has.
        return 0.0


def _common_cut(x: np.ndarray, period: float) -> float:
    """B1: the share of the achievable cost drop a cut common to every label buys."""
    return float(cost_drop_fraction(x, penalty=changepoint_penalty(x.shape[0], SLOPE)))


def _dependence(x: np.ndarray, period: float) -> float:
    """B3: how far the same-epoch correlation moves between the two end windows."""
    window = max(MIN_WINDOW, x.shape[0] // 4)
    # Two windows are what a drift is read between, and a block holding fewer
    # epochs than that carries no trajectory to compare.
    if x.shape[0] < 2 * window:
        return 0.0
    trajectory = windowed_correlation(x, window)
    return float(dependence_distance(trajectory[:1], trajectory[-1:]))


def _lagged_dependence(x: np.ndarray, period: float) -> float:
    """B4: how far the lag-one cross-correlation moves between the two end windows."""
    window = max(MIN_WINDOW, x.shape[0] // 4)
    if x.shape[0] < 2 * window:
        return 0.0
    trajectory = lagged_correlation(x, window)
    return float(dependence_distance(trajectory[:1], trajectory[-1:]))


@dataclass(frozen=True)
class FormSpec:
    """One row of the grid, the scalar it reads, and the effect it exists to catch."""

    id: str
    layer: str
    joint: bool
    effect: str
    size: float
    scalar: Callable[[np.ndarray, float], float]


#: What each row exists to catch, declared as a perturbation and its size.  This
#: is the claim about what matters, and it is made with no synthetic log in hand.
#:
#: A tolerance picked by eye and a power bound picked by eye are two independent
#: numbers, and nothing makes the second imply the first: a row forgiving more
#: than its own bound perturbs is underpowered by construction and the power
#: clause drops it.  Six of these ten rows were in that state.  The tolerance is
#: derived from the perturbation now, so a row catches its own declared effect by
#: construction and the power clause checks the arithmetic.
#:
#: The perturbation has to be one the form can see.  A phase shift moves neither
#: the power at a period nor the amplitude of a harmonic, so A05 and A06 are
#: declared against the perturbation that washes the rhythm out.
FORMS: Tuple[FormSpec, ...] = (
    FormSpec("A01", MARGINAL, False, "slope", 0.20, _trend),
    FormSpec("A02", MARGINAL, False, "level_step", 0.20, spline_curvature),
    FormSpec("A03", MARGINAL, False, "level_step", 0.20, _unit_root),
    FormSpec("A04", MARGINAL, False, "level_step", 0.20, _drift),
    FormSpec("A05", MARGINAL, False, "correlation", 0.20, _period_power),
    FormSpec("A06", MARGINAL, False, "correlation", 0.20, _profile),
    FormSpec("A07", MARGINAL, False, "correlation", 0.20, amplitude_drift),
    FormSpec("A08", MARGINAL, False, "level_step", 0.20, _cuts(POISSON)),
    FormSpec("A09", MARGINAL, False, "dispersion", 0.20, _cuts(VARIANCE)),
    FormSpec("A10", MARGINAL, False, "slope_break", 0.20, _cuts(SLOPE)),
    FormSpec("A11", MARGINAL, False, "correlation", 0.20, persistence_drift),
    FormSpec("A12", MARGINAL, False, "zero_rate", 0.20, _zero_share),
    FormSpec("A13", MARGINAL, False, "dispersion", 0.20, _dispersion),
    FormSpec("A14", MARGINAL, False, "level_step", 0.20, regime_entropy),
    FormSpec("A15", MARGINAL, False, "phase", 0.20, placement_concentration),
    FormSpec("A16", MARGINAL, False, "correlation", 0.20, window_mmd),
    FormSpec("B1", JOINT, True, "level_step", 0.20, _common_cut),
    FormSpec("B2", JOINT, False, "correlation", 0.20, factor_share),
    FormSpec("B3", JOINT, True, "correlation", 0.20, _dependence),
    FormSpec("B4", JOINT, True, "correlation", 0.20, _lagged_dependence),
    FormSpec("B5", JOINT, False, "level_step", 0.20, regime_coupling),
)

#: The share of its own declared effect a cell forgives.  A half puts the
#: declared effect at twice the tolerance, so the effect clears the equivalence
#: bound with the same margin at every cell.
TOLERANCE_SHARE = 0.5

#: A tolerance below this is no tolerance: the row cannot see its own declared
#: effect on that series, and the cell is reported descriptive for want of power.
TOLERANCE_FLOOR = 1e-12


def perturbed(spec: FormSpec, series: Series, size: float, seed: int = 0) -> np.ndarray:
    """The form's scalar per participant after one perturbation of a stated size."""
    return np.array(
        [
            spec.scalar(
                inject_block(block, spec.effect, size, series.period, seed + i),
                series.period,
            )
            for i, block in enumerate(series.blocks)
        ]
    )


def declared_effect(
    spec: FormSpec,
    series: Series,
    seed: int = 0,
    base: Optional[np.ndarray] = None,
) -> float:
    """
    How far the row's own declared perturbation moves the form, on this series.

    The measurement runs on the real panel alone, before any synthetic log
    exists, so the tolerance it sets cannot be moved by a result.

    Args:
        base: the unperturbed scalars, when a caller already holds them.  A
            regime fit costs a tenth of a second per participant and the
            comparison reads the same series four times, so what it holds is
            passed down.
    """
    if base is None:
        base = form_values(spec, series)
    return abs(float(np.mean(base - perturbed(spec, series, spec.size, seed))))


def tolerance_of(
    spec: FormSpec,
    series: Series,
    seed: int = 0,
    base: Optional[np.ndarray] = None,
) -> float:
    """
    The tolerance of one cell, derived from the effect the row exists to catch.

    Returns:
        Half the measured effect, or zero when the perturbation moves the form by
        nothing at all, which says the row cannot see its own declared effect.
    """
    effect = declared_effect(spec, series, seed, base)
    return TOLERANCE_SHARE * effect if effect > TOLERANCE_FLOOR else 0.0


def tolerance_table(
    real: Sequence[Series], forms: Sequence[FormSpec] = FORMS, seed: int = 0
) -> Dict[str, float]:
    """Every cell's derived tolerance, keyed by form, series and scale."""
    return {
        f"{spec.id}|{series.label}|{series.scale}": tolerance_of(spec, series, seed)
        for series in real
        for spec in forms
        if admits(spec, series)
    }


def table_digest(table: Mapping[str, float]) -> str:
    """A stable digest of a tolerance table, so a run names the table it read."""
    payload = "|".join(f"{key}={table[key]:.12g}" for key in sorted(table))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def form_ids() -> Tuple[str, ...]:
    """The rows this comparison runs, which the grid marks implemented."""
    return tuple(spec.id for spec in FORMS)


# ---------------------------------------------------------------------------
# What a panel offers the grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Series:
    """One label at one calendar scale, or the whole label block, per participant."""

    label: str
    scale: str
    joint: bool
    blocks: Tuple[np.ndarray, ...]

    @property
    def period(self) -> float:
        return SCALE_PERIOD[self.scale]

    @property
    def key(self) -> Tuple[str, str]:
        return self.label, self.scale

    @property
    def n_epochs(self) -> int:
        return int(self.blocks[0].shape[0]) if self.blocks else 0


JOINT_LABEL = "all labels"


def panel_series(
    cohort: Cohort,
    axis: LabelAxis,
    scales: Sequence[int] = COMPARED_SCALES,
    names: Mapping[int, str] = SCALE_NAMES,
) -> List[Series]:
    """
    Every label's counts at every compared scale, and the joint block beside them.

    Args:
        scales: which scales of the cohort's own ladder to read.
        names: the calendar scale each of them is, which is what a cell reports
            and what decides the period a periodic form is read at.
    """
    blocks: Dict[Tuple[str, str], List[np.ndarray]] = {}
    for name in cohort.subjects:
        data = cohort.open(name)
        for scale in scales:
            counts = np.asarray(data.X[scale - 1], dtype=float)
            for j, label in enumerate(axis.targets):
                blocks.setdefault((label, names[scale]), []).append(counts[:, j])
            blocks.setdefault((JOINT_LABEL, names[scale]), []).append(counts)
    return [
        Series(
            label=label,
            scale=scale,
            joint=label == JOINT_LABEL,
            blocks=tuple(rows),
        )
        for (label, scale), rows in blocks.items()
    ]


def by_key(series: Sequence[Series]) -> Dict[Tuple[str, str], Series]:
    """The series of a panel, keyed by the label and the scale it belongs to."""
    return {item.key: item for item in series}


def admits(spec: FormSpec, series: Series) -> bool:
    """Whether one form is read on one series at all."""
    if spec.joint != series.joint:
        return False
    if series.scale == WEEK and spec.id not in WEEK_FORMS:
        return False
    if series.scale == HOUR and spec.id in REGIME_FORMS:
        return False
    return True


def form_values(spec: FormSpec, series: Series) -> np.ndarray:
    """One scalar per participant."""
    return np.array(
        [spec.scalar(block, series.period) for block in series.blocks], dtype=float
    )


# ---------------------------------------------------------------------------
# The injection calibration, taken per participant
# ---------------------------------------------------------------------------


def inject_block(
    block: np.ndarray, kind: str, size: float, period: float, seed: int
) -> np.ndarray:
    """Put a perturbation into one participant's block, column by column if joint."""
    values = np.asarray(block, dtype=float)
    if values.ndim == 1:
        return inject(values, kind, size, period=period, seed=seed)
    return np.column_stack(
        [
            inject(values[:, j], kind, size, period=period, seed=seed + j)
            for j in range(values.shape[1])
        ]
    )


def minimum_detectable_cell(
    spec: FormSpec,
    series: Series,
    sizes: Sequence[float] = INJECTION_SIZES,
    seed: int = 0,
    tolerance: Optional[float] = None,
    base: Optional[np.ndarray] = None,
) -> float:
    """
    The smallest injected size whose effect on the form clears the cell's tolerance.

    An equivalence claim is only as strong as what its tolerance can see, so the
    calibration is stated against that tolerance and not against a significance
    level.  A perturbation moving the form by less than the tolerance would pass
    the cell whatever the sample size, which is what the power clause of the
    inclusion rule drops.

    The tolerance is half the effect of the row's own declared perturbation, so
    that perturbation clears it and the answer is at most the declared size.  A
    row whose perturbation moves the form by nothing has no tolerance and reports
    an infinite effect, which is what the power clause exists to catch.

    Returns:
        The first size in the grid that clears the tolerance, or infinity when no
        size in the grid does.
    """
    if tolerance is None:
        tolerance = tolerance_of(spec, series, seed, base)
    if tolerance <= 0.0:
        return math.inf
    if base is None:
        base = form_values(spec, series)
    for size in sorted(float(s) for s in sizes):
        moved = perturbed(spec, series, size, seed)
        if abs(float(np.mean(base - moved))) > tolerance:
            return float(size)
    return math.inf


# ---------------------------------------------------------------------------
# The cells
# ---------------------------------------------------------------------------


def standard_error(values: np.ndarray) -> float:
    """The spread of one form's per-participant values, over the root of their count."""
    x = np.asarray(values, dtype=float)
    if x.size < 2:
        return 0.0
    return float(np.std(x, ddof=1) / math.sqrt(x.size))


def denominator(series: Series, period: float) -> Denominator:
    """What the real log gives one cell to work with."""
    totals = np.array([float(np.sum(block)) for block in series.blocks])
    return Denominator(
        episodes=int(totals.sum()),
        carriers=int(np.sum(totals > 0.0)),
        epochs_per_participant=series.n_epochs,
        cycles=series.n_epochs / period if period > 0.0 else 0.0,
    )


def compare(
    real: Sequence[Series],
    replicates: Sequence[Sequence[Series]],
    forms: Sequence[FormSpec] = FORMS,
    sizes: Sequence[float] = INJECTION_SIZES,
    power_bound: float = POWER_BOUND,
    seed: int = 0,
) -> List[Cell]:
    """
    One cell per form, series and scale, with the paired difference each reports.

    Raises:
        ValueError: when no replicate is offered, or when a replicate carries a
            different series set than the real panel.
    """
    if not replicates:
        raise ValueError(
            "the comparison needs at least one synthetic panel; none was offered"
        )
    drawn = [by_key(panel) for panel in replicates]
    cells: List[Cell] = []
    for series in real:
        for spec in forms:
            if not admits(spec, series):
                continue
            for panel in drawn:
                if series.key not in panel:
                    raise ValueError(
                        f"a synthetic panel carries no {series.label!r} at the "
                        f"{series.scale} scale, which the real panel does"
                    )
            observed = form_values(spec, series)
            # A drawn participant is a draw of the population and not the twin
            # of a real one, so every replicate's participants join one group
            # and the contrast is between the two groups.
            per_panel = [form_values(spec, panel[series.key]) for panel in drawn]
            synthetic = np.concatenate(per_panel)
            # The replicates redraw the same windows and masks, so the panel and
            # not the draw is the independent unit; charging Welch the pooled
            # length would understate the standard error R-fold.
            units = int(min(part.size for part in per_panel)) if per_panel else 0
            counts = denominator(series, series.period)
            tolerance = tolerance_of(spec, series, seed, observed)
            cells.append(
                Cell(
                    form=spec.id,
                    series=series.label,
                    scale=series.scale,
                    magnitude=float(np.mean(observed)),
                    spread=float(np.std(observed)),
                    effect=hedges_g(observed, synthetic),
                    p_value=(
                        group_tost(
                            observed, synthetic, tolerance, synthetic_units=units
                        ).p_value
                        if tolerance > 0.0
                        else 1.0
                    ),
                    # The equivalence test answers "close enough"; this answers
                    # "distinguishable at all", and a cell where both are small
                    # is one the tolerance is too loose for.
                    permutation_p=(
                        group_permutation_p(observed, synthetic, seed=seed)
                        if observed.size and synthetic.size
                        else 1.0
                    ),
                    inclusion=inclusion(
                        counts,
                        minimum_detectable_cell(
                            spec,
                            series,
                            sizes,
                            seed=seed,
                            tolerance=tolerance,
                            base=observed,
                        ),
                        power_bound,
                        needs_period=spec.id in ("A05", "A06"),
                        tolerance=tolerance,
                        standard_error=standard_error(observed),
                    ),
                    denominator=counts,
                )
            )
    return cells


def failed_rows(cells: Sequence[Cell], alpha: float = 0.05) -> Tuple[str, ...]:
    """The forms holding at least one tested cell that did not pass."""
    return tuple(
        sorted({c.form for c in cells if c.inclusion.tested and c.p_value > alpha})
    )


def rows_tested(cells: Sequence[Cell]) -> Tuple[str, ...]:
    """The forms the run put to the test at all."""
    return tuple(sorted({c.form for c in cells if c.inclusion.tested}))


def declared_rows() -> Tuple[str, ...]:
    """The grid rows still waiting on a dependency."""
    return tuple(row.id for row in GRID if row.id not in form_ids())


# ---------------------------------------------------------------------------
# The two shuffles, applied to a drawn panel
# ---------------------------------------------------------------------------

TIME_SHUFFLE = "within-participant time shuffle"
LABEL_SHUFFLE = "label-shuffled cross-section"


def shuffled(panel: Sequence[Series], kind: str, seed: int = 0) -> List[Series]:
    """
    A drawn panel with one structure removed, as a negative control.

    Raises:
        ValueError: when the shuffle is neither of the two declared ones.
    """
    if kind not in (TIME_SHUFFLE, LABEL_SHUFFLE):
        raise ValueError(
            f"shuffle {kind!r}; expected {TIME_SHUFFLE!r} or {LABEL_SHUFFLE!r}"
        )
    out: List[Series] = []
    for series in panel:
        blocks = []
        for i, block in enumerate(series.blocks):
            values = np.asarray(block, dtype=float)
            if kind == TIME_SHUFFLE:
                if values.ndim == 1:
                    moved = shuffle_within_participant(
                        values, np.zeros(values.shape[0], dtype=np.int64), seed + i
                    )
                else:
                    rows = np.random.default_rng(seed + i).permutation(values.shape[0])
                    moved = values[rows]
            else:
                moved = (
                    values
                    if values.ndim == 1
                    else shuffle_labels(values, seed=seed + i)
                )
            blocks.append(moved)
        out.append(
            Series(
                label=series.label,
                scale=series.scale,
                joint=series.joint,
                blocks=tuple(blocks),
            )
        )
    return out


def by_form(cells: Sequence[Cell], alpha: float = 0.05) -> Dict[str, Dict[str, object]]:
    """
    Per form: how many cells it holds, how many were tested, and why the rest were not.

    A row reported only as passed or failed hides how much of the grid it stood
    on, and a row that is descriptive everywhere reads the same as a row that
    held.
    """
    out: Dict[str, Dict[str, object]] = {}
    for cell in cells:
        row = out.setdefault(
            cell.form, {"cells": 0, "tested": 0, "failed": 0, "clauses": {}}
        )
        row["cells"] += 1
        if cell.inclusion.tested:
            row["tested"] += 1
            row["failed"] += int(cell.p_value > alpha)
        else:
            clauses = row["clauses"]
            clauses[cell.inclusion.clause] = clauses.get(cell.inclusion.clause, 0) + 1
    return out


def summary(cells: Sequence[Cell], alpha: float = 0.05) -> Dict[str, object]:
    """What a run reports of one comparison, holding the failing cells and no others."""
    tested = [c for c in cells if c.inclusion.tested]
    clauses: Dict[str, int] = {}
    for cell in cells:
        if not cell.inclusion.tested:
            clauses[cell.inclusion.clause] = clauses.get(cell.inclusion.clause, 0) + 1
    return {
        "n_cells": len(cells),
        "n_tested": len(tested),
        "descriptive_by_clause": clauses,
        "failed": list(failed_rows(cells, alpha)),
        # A form is named as failing by one cell of it, so the row list alone
        # does not say which label or which scale the departure sits at, and the
        # per-cell effect size is the only place the size of a departure lives.
        "tested_cells": [c.as_row() for c in tested],
        "tested_rows": list(rows_tested(cells)),
        "by_form": by_form(cells, alpha),
        "passed": not failed_rows(cells, alpha),
    }
