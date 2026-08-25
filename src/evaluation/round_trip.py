"""
evaluation/round_trip.py
========================
A log drawn from a fit, written as a CSV, and read back through the panel reader.

The observational arm fits a graph over the balanced panel and stops there.  A
fidelity claim needs a second log to compare the first one against, drawn from
that fit and arriving through the same door: the writer emits the raw schema
`data.lifesnaps` reads, and `lifesnaps_panel.balanced_panel` reads it back.  Any
difference the comparison then reports is a difference between two logs, since
one reader produced both panels.

Three things are exogenous and are carried across unchanged.  The observation
schedule says which days the study watched, and `PanelSchedule` hands it to the
reader so the drawn panel is laid on the same window and the same mask.  The
scheduled label is delivered by the study, so its slots are copied from the
record and never drawn.  And the clock nodes are a function of the window, so
the reader rebuilds them identically.

Two blocks of the fit are set here, and the search touches neither.  The search
prices count relations, so the mark block carries no relation and its intercept
and dispersion are moment-matched to the panel's own episode lengths, which is
what puts a drawn episode on a plausible span.  The participant effect is fitted:
`subject_offset.SubjectLevelFit` holds the searched graph fixed and gives every
subject its own level.  A drawn participant then takes a level resampled off that
fitted roster, so a drawn log carries a population and never a copy of the one it
was fitted on.

A fitted regime chain crosses over as a trajectory and never as a posterior.  The
coefficients were maximized with the chain's contribution in the predictor, so a
draw that dropped it would sit at a rate the fit never held; a draw that carried
the posterior of a real participant would hand a synthetic one that participant's
own history.  `RegimeChain.draw` walks the fitted initial and transition laws, and
what the path adds enters the count predictor through the same offset argument
the participant level does.
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..core.cohort import Cohort
from ..core.typed_graph import COUNT, DURATION, TypedGraph
from ..data.label_axis import SCHEDULED_LABEL, LabelAxis
from ..data.lifesnaps import _RAW_FORMAT
from ..data.lifesnaps_panel import (
    MINUTES_PER_DAY,
    PanelReport,
    PanelSchedule,
    balanced_panel,
)
from ..learning.fine_design import CalendarIndex, ScaleLadder, build_design
from ..learning.fine_sem import RegimeChain, fit_rows
from ..learning.subject_offset import SubjectLevelFit, level_rows
from ..model.intensity import (
    CLIP_COLUMNS,
    FineParams,
    empty_sources,
    sample,
    usable_slots,
)
from .fidelity import check_reconciles

#: The columns `data.lifesnaps.read_lifesnaps` reads.
RAW_FIELDS: Tuple[str, ...] = (
    "person_id",
    "event",
    "day_index",
    "start_min",
    "duration_min",
    "start_datetime",
)

#: The smallest mean an episode length may be matched to, in slots.
MARK_FLOOR = 1e-3

#: The negative binomial size is clipped here, so a label whose lengths are
#: under-dispersed gets the Poisson end of the family and never a negative size.
MARK_SIZE_RANGE: Tuple[float, float] = (1e-2, 1e6)


@dataclass(frozen=True)
class Episode:
    """One drawn or copied episode, in the raw log's own coordinates."""

    unit: str
    label: str
    day_index: int
    start_min: int
    duration_min: int


@dataclass(frozen=True)
class RoundTrip:
    """One synthetic panel and what the writer put in the file it came from."""

    path: str
    cohort: Cohort
    axis: LabelAxis
    report: PanelReport
    drawn: int
    copied: int
    # In-mask drawn episodes per label, and how often the predictor was capped
    # at `_MAX_ETA` per label.  Both are empty on a trip built before these were
    # recorded; neither is derivable from `drawn`, which is a single total and
    # is why a leg missing two of its six labels read as healthy.
    per_label: Dict[str, int] = field(default_factory=dict)
    clips: Dict[str, Dict[str, int]] = field(default_factory=dict)

    @property
    def episodes(self) -> int:
        return self.drawn + self.copied


# ---------------------------------------------------------------------------
# The two blocks the search does not price
# ---------------------------------------------------------------------------


def mark_moments(cohort: Cohort) -> Tuple[np.ndarray, np.ndarray]:
    """
    Each label's mean and variance of episode length, in finest slots.

    A slot that starts one episode stores that episode's whole span, so the
    lengths are read off those slots alone.  A slot where two episodes start at
    once stores their sum, which is no episode's length, and it is left out.
    """
    total = np.zeros(cohort.L)
    squares = np.zeros(cohort.L)
    n = np.zeros(cohort.L)
    for _, data in cohort.stream():
        X1, Y1 = np.asarray(data.get_X(1)), np.asarray(data.get_Y(1))
        for ell in range(cohort.L):
            lengths = Y1[X1[:, ell] == 1, ell].astype(float)
            n[ell] += lengths.size
            total[ell] += lengths.sum()
            squares[ell] += float(lengths @ lengths)
    mean = np.divide(total, n, out=np.ones(cohort.L), where=n > 0)
    variance = np.divide(squares, n, out=np.zeros(cohort.L), where=n > 0) - mean**2
    return mean, np.maximum(variance, 0.0)


def calibrate_marks(params: FineParams, cohort: Cohort) -> FineParams:
    """Set the mark intercept and dispersion to the panel's own length moments."""
    mean, variance = mark_moments(cohort)
    # `sample` draws one plus a negative binomial, so the matched mean is the
    # mean length less the slot the episode starts in.
    mu = np.maximum(mean - 1.0, MARK_FLOOR)
    excess = np.maximum(variance - mu, MARK_FLOOR)
    size = np.clip(mu**2 / excess, *MARK_SIZE_RANGE)
    params.base[DURATION][:] = np.log(mu)
    params.mark_dispersion = size
    return params


def start_log_odds(cohort: Cohort, lag_budget: Mapping[int, int]) -> np.ndarray:
    """
    Each label's start log-odds over the rows a fit scores, as an opening intercept.

    An intercept starting at zero is one start per two slots, and a label whose
    own rate is eleven log-odds below that takes many Newton steps to reach it.
    Opening at the record's own rate costs one pass and starts the solve where a
    model with no parent already sits.
    """
    starts = np.zeros(cohort.L)
    rows = np.zeros(cohort.L)
    for _, data in cohort.stream():
        X1, Y1 = data.get_X(1), data.get_Y(1)
        at_risk = usable_slots(
            X1, Y1, fit_rows(data.observed, lag_budget, cohort.sources(data))
        )
        starts += (np.asarray(X1) > 0).sum(axis=0)
        rows += at_risk.sum(axis=0)
    # A label with no start at all has no finite log-odds, so the count and the
    # denominator each carry the half a start a continuity correction adds.
    rate = (starts + 0.5) / (rows + 1.0)
    return np.log(rate / (1.0 - rate))


def identified_levels(cohort: Cohort, lag_budget: Mapping[int, int]) -> np.ndarray:
    """
    Which subjects bring a scorable row to every target under the declared budget.

    A lag budget of three at the week scale needs three whole weeks of observed
    history before a row is scorable, and a subject whose record covers no more
    than that brings none. Its level is identified by nothing, so it is left at
    the population intercept and the coding never carries a column for it.
    """
    rows = level_rows(cohort, dict(lag_budget))
    return np.flatnonzero(rows.min(axis=1) > 0)


#: Power iterations allowed when solving a chain for its stationary law.  A
#: two-state chain converges in a few; the cap is a guard, not a schedule.
_STATIONARY_ITERS = 1000


def stationary_law(chain: RegimeChain) -> np.ndarray:
    """
    The chain's own long-run state distribution.

    `log_trans` is stored as `log p(next | previous)` transposed, so
    `exp(log_trans)` is column-stochastic and the stationary law is its right
    eigenvector at eigenvalue one.  Power iteration from the fitted initial
    distribution reaches it for any chain that mixes, and a chain that does not
    keeps the initial law, which is the right answer for an absorbing one.
    """
    matrix = np.exp(np.asarray(chain.log_trans, dtype=float))
    law = np.exp(np.asarray(chain.log_init, dtype=float))
    total = law.sum()
    law = law / total if total > 0 else np.full(chain.n_states, 1.0 / chain.n_states)
    for _ in range(_STATIONARY_ITERS):
        stepped = matrix @ law
        total = stepped.sum()
        if total <= 0:
            return law
        stepped = stepped / total
        if np.max(np.abs(stepped - law)) < 1e-12:
            return stepped
        law = stepped
    return law


def center_chains(
    params: FineParams, chains: Sequence[RegimeChain]
) -> Tuple[FineParams, List[RegimeChain]]:
    """
    Move each chain's average contribution into the intercept it belongs to.

    `participant_levels` refits every subject's level through `SubjectLevelFit`,
    which is documented "one graph over a cohort, with each subject's level free
    **and no chain**", on coefficients that were maximized with the chain in the
    predictor.  The level fit therefore has to absorb the chain's average
    contribution into the levels, and the stored run shows the cost: a
    `level_gain` of 3.00e8 against 1391 for the one-state leg, ending 3.3 times
    worse with more parameters and still climbing after sixty sweeps.

    Writing `base + E[effect]` for the intercept and `effect - E[effect]` for the
    chain leaves the predictor identical at every slot and every state path,
    since the two shifts cancel; what changes is that the chain now averages to
    zero, so the chain-free level fit is centred where the draw will be.
    """
    out = FineParams(n_labels=params.n_labels, n_subjects=params.n_subjects)
    for quantity in (COUNT, DURATION):
        out.base[quantity][:] = params.base[quantity]
        out.offset[quantity][:] = params.offset[quantity]
    for key, coefficients in params.coef.items():
        out.coef[key] = dict(coefficients)
    out.mark_dispersion = np.asarray(params.mark_dispersion, dtype=float).copy()

    centered: List[RegimeChain] = []
    for chain in chains:
        law = stationary_law(chain)
        average = law @ np.asarray(chain.effect, dtype=float)
        out.base[COUNT][:] = out.base[COUNT] + average
        centered.append(replace(chain, effect=np.asarray(chain.effect) - average))
    return out, centered


def participant_levels(
    cohort: Cohort,
    graph: TypedGraph,
    params: FineParams,
    lag_budget: Mapping[int, int],
    index: Optional[CalendarIndex] = None,
    max_sweeps: int = 8,
) -> Tuple[FineParams, List[float], Tuple[str, ...]]:
    """
    Refit the searched graph with every identified subject's own level free.

    Returns:
        Parameters carrying one offset row per subject of the whole roster, the
        start log-likelihood before each solve, and the subjects whose level no
        row identifies, which stay at the population intercept.
    """
    wider = FineParams(n_labels=cohort.L, n_subjects=cohort.S)
    for quantity in (COUNT, DURATION):
        wider.base[quantity][:] = params.base[quantity]
    for key, coefficients in params.coef.items():
        wider.coef[key] = dict(coefficients)
    wider.mark_dispersion = np.asarray(params.mark_dispersion, dtype=float).copy()

    keep = identified_levels(cohort, lag_budget)
    dropped = tuple(
        name for j, name in enumerate(cohort.subjects) if j not in set(keep.tolist())
    )
    inner = FineParams(n_labels=cohort.L, n_subjects=int(keep.size))
    for quantity in (COUNT, DURATION):
        inner.base[quantity][:] = wider.base[quantity]
    for key, coefficients in wider.coef.items():
        inner.coef[key] = dict(coefficients)
    inner.mark_dispersion = np.asarray(wider.mark_dispersion, dtype=float).copy()
    seen = replace(
        cohort,
        subjects=tuple(cohort.subjects[j] for j in keep),
        masks=cohort.masks[keep],
    )
    fit = SubjectLevelFit(graph, inner, lag_budget, index=index, max_sweeps=max_sweeps)
    _, history = fit.fit(seen)

    for quantity in (COUNT, DURATION):
        wider.base[quantity][:] = inner.base[quantity]
        wider.offset[quantity][keep] = inner.offset[quantity]
    for key, coefficients in inner.coef.items():
        wider.coef[key] = dict(coefficients)
    return wider, history, dropped


def population_levels(
    cohort: Cohort,
    params: FineParams,
    unidentified: Sequence[str] = (),
    seed: int = 0,
) -> FineParams:
    """
    Give every drawn participant a level vector taken from the fitted roster.

    A drawn participant is a draw of the population and never the twin of a real
    one, so the level is resampled with replacement from the levels the fit
    identified.  A whole row moves at once and a participant's count row and
    duration row move together, so someone who sleeps much and walks little
    arrives that way and the co-variation between labels crosses into the drawn
    population, which is what the joint layer is stated on.

    The drawn population's distribution of levels is the fitted one by
    construction.  A form that reads the participant level alone therefore
    agrees by construction and its cell carries nothing about the fit.

    Args:
        unidentified: the subjects whose level no row identifies.  They sit at
            the population intercept and are not resampled from.

    Raises:
        ValueError: when no subject of the roster carries an identified level,
            since there is then no population to draw one from.
    """
    absent = set(unidentified)
    keep = [j for j, name in enumerate(cohort.subjects) if name not in absent]
    if not keep:
        raise ValueError(
            f"a drawn population needs one identified level; all {cohort.S} "
            "subjects sit at the population intercept"
        )
    out = FineParams(n_labels=cohort.L, n_subjects=cohort.S)
    for quantity in (COUNT, DURATION):
        out.base[quantity][:] = params.base[quantity]
    for key, coefficients in params.coef.items():
        out.coef[key] = dict(coefficients)
    out.mark_dispersion = np.asarray(params.mark_dispersion, dtype=float).copy()
    picks = np.random.default_rng(seed).choice(
        np.asarray(keep, dtype=np.int64), size=cohort.S, replace=True
    )
    for quantity in (COUNT, DURATION):
        out.offset[quantity][:] = params.offset[quantity][picks]
    return out


# ---------------------------------------------------------------------------
# The draw
# ---------------------------------------------------------------------------


def centering_offset(
    graph: TypedGraph,
    params: FineParams,
    sources,
    observed,
    quantity: str,
    index: Optional[CalendarIndex],
    lag_budget: Mapping[int, int],
) -> np.ndarray:
    """
    What the fit holds beyond the intercept and the raw edge features.

    The design centers every edge column on the record's own mean, so the stored
    intercept belongs to the centered parametrization and `linear_predictor`
    reading raw features is short by the mean that came out.  The likelihood
    puts that term back through its offset argument, and a draw has to read the
    same predictor the likelihood was maximized on.
    """
    ladder = ScaleLadder.from_sources(sources, observed)
    keyed = index if index is not None else CalendarIndex(placed={}, widths={})
    out = np.zeros((sources.n_slots, sources.n_labels))
    for ell in range(sources.n_labels):
        design = build_design(
            graph, ell, quantity, sources, ladder, keyed, budget=lag_budget
        )
        held = np.array(
            [params.coef[(ell, quantity)].get(c.edge, 0.0) for c in design.columns]
        )
        out[:, ell] = design.offset(np.concatenate([[0.0], held]))
    return out


def regime_offset(
    chains: Sequence[RegimeChain], sources, rng: np.random.Generator
) -> np.ndarray:
    """Each chain's own drawn state path, as what it adds to the start predictor."""
    total = np.zeros((sources.n_slots, sources.n_labels))
    for chain in chains:
        epochs = int(np.shape(sources.counts[chain.scale - 1])[0])
        total = total + chain.path_contribution(
            chain.draw(epochs, rng), sources.ancestors
        )
    return total


def draw_offsets(
    graph: TypedGraph,
    params: FineParams,
    sources,
    observed,
    subject: int,
    index: Optional[CalendarIndex],
    lag_budget: Mapping[int, int],
    chains: Sequence[RegimeChain] = (),
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, np.ndarray]:
    """
    One subject's level, the design's centering term and one drawn regime path.

    The chains contribute to the start predictor alone, which is where the fit
    put them.

    Raises:
        ValueError: when a chain is given with no generator to draw its path.
    """
    out: Dict[str, np.ndarray] = {}
    for quantity in (COUNT, DURATION):
        block = centering_offset(
            graph, params, sources, observed, quantity, index, lag_budget
        )
        if params.n_subjects:
            block = block + params.offset[quantity][subject]
        out[quantity] = block
    if chains:
        if rng is None:
            raise ValueError(
                f"{len(chains)} fitted chain(s) arrived with no generator to draw "
                "a state path from, and a drawn log carries a trajectory of its own"
            )
        out[COUNT] = out[COUNT] + regime_offset(chains, sources, rng)
    return out


def draw_subject(
    cohort: Cohort,
    axis: LabelAxis,
    name: str,
    subject: int,
    graph: TypedGraph,
    params: FineParams,
    rng: np.random.Generator,
    offset_days: int,
    index: Optional[CalendarIndex] = None,
    lag_budget: Optional[Mapping[int, int]] = None,
    chains: Sequence[RegimeChain] = (),
    clips: Optional[np.ndarray] = None,
) -> Tuple[List[Episode], List[Episode]]:
    """
    Draw one subject's behaviour and copy the scheduled label it was delivered.

    The process runs whether or not the study is watching, so the draw covers
    the whole window and the episodes outside the subject's mask are dropped
    afterward.

    `clips`, when given, is the `(L, 4)` counter `intensity.sample` fills, shared
    across the subjects of one draw.

    Returns:
        The drawn episodes and the copied scheduled ones.
    """
    data = cohort.open(name)
    sources = cohort.sources(data)
    budget = dict(lag_budget or {k: 1 for k in range(1, cohort.K + 1)})
    fresh = empty_sources(sources.ancestors, sources.calendar, axis.L)
    X1, Y1 = sample(
        params,
        graph,
        fresh,
        rng,
        draw_offsets(
            graph,
            params,
            sources,
            data.observed,
            subject,
            index,
            budget,
            chains,
            rng,
        ),
        clips,
    )
    check_reconciles(X1, Y1, fresh.counts, fresh.durations, fresh.ancestors)

    observed = np.asarray(data.observed, dtype=bool)
    drawn: List[Episode] = []
    for slot, column in np.argwhere(X1 > 0):
        if not observed[slot]:
            continue
        label = axis.targets[column]
        drawn.append(
            Episode(
                unit=name,
                label=label,
                day_index=offset_days + int(slot) // MINUTES_PER_DAY,
                start_min=int(slot) % MINUTES_PER_DAY,
                # An instant is written as a zero the reader's own rule fills in,
                # so the two logs carry the same injected length on those labels.
                duration_min=0 if axis.is_instant(label) else int(Y1[slot, column]),
            )
        )

    column = axis.exogenous_index(SCHEDULED_LABEL)
    delivered = np.asarray(sources.calendar[0][:, column])
    copied = [
        Episode(
            unit=name,
            label=SCHEDULED_LABEL,
            day_index=offset_days + int(slot) // MINUTES_PER_DAY,
            start_min=int(slot) % MINUTES_PER_DAY,
            duration_min=0,
        )
        for slot in np.flatnonzero(delivered > 0)
        for _ in range(int(delivered[slot]))
    ]
    return drawn, copied


def write_raw_log(
    path: str, episodes: Sequence[Episode], day_zero: Mapping[str, date]
) -> None:
    """Write episodes in the raw schema, stamped in each subject's own calendar."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RAW_FIELDS))
        writer.writeheader()
        for episode in episodes:
            when = datetime.combine(
                day_zero[episode.unit], datetime.min.time()
            ) + timedelta(days=episode.day_index, minutes=episode.start_min)
            writer.writerow(
                {
                    "person_id": episode.unit,
                    "event": episode.label,
                    "day_index": episode.day_index,
                    "start_min": episode.start_min,
                    "duration_min": episode.duration_min,
                    "start_datetime": when.strftime(_RAW_FORMAT),
                }
            )


def round_trip(
    cohort: Cohort,
    axis: LabelAxis,
    schedule: PanelSchedule,
    graph: TypedGraph,
    params: FineParams,
    rng: np.random.Generator,
    path: str,
    index: Optional[CalendarIndex] = None,
    lag_budget: Optional[Mapping[int, int]] = None,
    chains: Sequence[RegimeChain] = (),
) -> RoundTrip:
    """
    Draw one log over the panel, write it, and read it back through the panel.

    Raises:
        ValueError: when the schedule does not name every subject of the cohort,
            when the drawn log induces a different label axis, or when the panel
            it reads back onto is not the one it was drawn over.
    """
    absent = [name for name in cohort.subjects if name not in schedule.offsets]
    if absent:
        raise ValueError(
            f"the schedule names no window for {len(absent)} subject(s) of the "
            f"cohort, first {absent[0]!r}, so the draw covers a different window"
        )
    drawn: List[Episode] = []
    copied: List[Episode] = []
    clips = np.zeros((axis.L, len(CLIP_COLUMNS)), dtype=np.int64)
    for subject, name in enumerate(cohort.subjects):
        own, given = draw_subject(
            cohort,
            axis,
            name,
            subject,
            graph,
            params,
            rng,
            schedule.offsets[name],
            index,
            lag_budget,
            chains,
            clips,
        )
        drawn.extend(own)
        copied.extend(given)
    write_raw_log(path, drawn + copied, schedule.day_zero)

    read, read_axis, report, _, _ = balanced_panel(
        weeks=schedule.weeks, path=path, schedule=schedule
    )
    if read_axis.digest() != axis.digest():
        raise ValueError(
            "the drawn log induces a different label axis than the record it was "
            f"drawn over: {list(read_axis.targets)} against {list(axis.targets)}"
        )
    if read.subjects != cohort.subjects or not np.array_equal(read.masks, cohort.masks):
        raise ValueError(
            "the drawn log reads back onto a different window than it was drawn "
            f"over: {read.S} subjects and {int(read.masks.sum())} observed slots "
            f"against {cohort.S} and {int(cohort.masks.sum())}"
        )
    counted: Dict[str, int] = {label: 0 for label in axis.targets}
    for episode in drawn:
        counted[episode.label] = counted.get(episode.label, 0) + 1
    return RoundTrip(
        path=path,
        cohort=read,
        axis=read_axis,
        report=report,
        drawn=len(drawn),
        copied=len(copied),
        per_label=counted,
        clips={
            label: dict(zip(CLIP_COLUMNS, (int(v) for v in clips[j])))
            for j, label in enumerate(axis.targets)
        },
    )


def replicates(
    cohort: Cohort,
    axis: LabelAxis,
    schedule: PanelSchedule,
    graph: TypedGraph,
    params: FineParams,
    n_replicates: int,
    directory: str,
    seed: int = 0,
    tag: str = "leg",
    index: Optional[CalendarIndex] = None,
    lag_budget: Optional[Mapping[int, int]] = None,
    chains: Sequence[RegimeChain] = (),
) -> List[RoundTrip]:
    """
    Draw R logs over the panel, each written to its own file.

    Raises:
        ValueError: when fewer than one replicate is asked for, since the
            comparison would then have no drawn group to read.
    """
    if n_replicates < 1:
        raise ValueError(
            f"{n_replicates} replicate(s) leave the drawn side of the "
            "comparison empty; the contrast needs at least one"
        )
    return [
        round_trip(
            cohort,
            axis,
            schedule,
            graph,
            params,
            np.random.default_rng(seed + r),
            os.path.join(directory, f"{tag}_{r:02d}.csv"),
            index,
            lag_budget,
            chains,
        )
        for r in range(int(n_replicates))
    ]


#: How far a drawn label's volume may sit from the real panel's before the trip
#: is reported as a departure.  Three is loose on purpose: the draw resamples
#: participant levels, so a factor of two is ordinary and only a gross departure
#: says the fit is wrong rather than the draw being noisy.
VOLUME_FACTOR = 3.0

#: How many episodes the real panel needs before drawing none of a label is
#: refused rather than merely reported.  Drawing zero of a label that occurs R
#: times has probability near exp(-R) for a process of the same rate, so at 20
#: an absence is a one-in-a-billion event and at 1 it is ordinary.  The 2026-08-23
#: defect was `survey_response` at 10,162 and `mindfulness` at 68, both far
#: above this; a six-subject smoke where `mindfulness` occurs once is not.
ABSENCE_FLOOR = 20


def panel_counts(cohort: Cohort, axis: LabelAxis) -> Dict[str, int]:
    """Each label's in-mask episode count over a whole panel, real or drawn."""
    totals = np.zeros(cohort.L, dtype=np.int64)
    for _, data in cohort.stream():
        X1 = np.asarray(data.get_X(1))
        observed = np.asarray(data.observed, dtype=bool)
        totals += X1[observed].sum(axis=0).astype(np.int64)
    return {label: int(totals[j]) for j, label in enumerate(axis.targets)}


def volume_departures(
    drawn: Mapping[str, int],
    real: Mapping[str, int],
    factor: float = VOLUME_FACTOR,
) -> List[Dict[str, object]]:
    """
    Labels whose drawn volume sits further than `factor` from the real panel's.

    This is the check nothing in the pipeline was making.  `check_reconciles`
    compares a draw against itself, the mask check compares windows, and the
    axis-digest check cannot fire because `PanelSchedule` declares the axis, so
    an empty label survives as a column.  A label the panel has and the draw
    does not is reported with `absent` set, which is the 2026-08-23 defect:
    `survey_response` and `mindfulness` drawn zero times against 10,162 and 68.
    """
    out: List[Dict[str, object]] = []
    for label in sorted(real):
        expected = int(real[label])
        got = int(drawn.get(label, 0))
        if expected == 0:
            continue
        ratio = got / expected
        if got == 0 or ratio > factor or ratio < 1.0 / factor:
            out.append(
                {
                    "label": label,
                    "real": expected,
                    "drawn": got,
                    "ratio": float(ratio),
                    "absent": got == 0,
                }
            )
    return out


def check_volume(
    drawn: Mapping[str, int],
    real: Mapping[str, int],
    factor: float = VOLUME_FACTOR,
    where: str = "the draw",
) -> List[Dict[str, object]]:
    """
    Report every volume departure and refuse a label that is missing outright.

    A ratio departure is returned rather than raised, because a draw is
    stochastic and a leg is worth finishing so its rows can be read.  An absent
    label is raised, because every row computed over it afterward describes a
    log that is not the model's: a series of zeros divides by `_mean(x)`, which
    returns 1.0, so it reports a finite number and never an error.

    Raises:
        ValueError: when a label the real panel carries is drawn zero times.
    """
    departures = volume_departures(drawn, real, factor)
    # An absence is only evidence when the label was common enough that drawing
    # none of it cannot be sampling.  Below the floor it stays in the record as
    # a departure and decides nothing.
    gone = [d for d in departures if d["absent"] and int(d["real"]) >= ABSENCE_FLOOR]
    if gone:
        named = ", ".join(f"{d['label']} (real {d['real']})" for d in gone)
        raise ValueError(
            f"{where} holds no episode of {len(gone)} label(s) the real panel "
            f"carries: {named}. Every row computed over this log would describe "
            "a draw the model did not make."
        )
    return departures


def reconciliation_error(trip: RoundTrip) -> float:
    """The largest disagreement between a subject's coarse totals and its own fine ones."""
    worst = 0.0
    for errors in trip.cohort.reconcile().values():
        worst = max(worst, max(abs(float(v)) for v in errors.values()))
    return worst


def missing_days(trip: RoundTrip, schedule: PanelSchedule) -> Dict[str, int]:
    """
    Observed days the drawn log left with no episode of its own, per subject.

    The schedule keeps such a day inside the mask, so the panel is unaffected;
    the count says how far the draw is from filling the window it was given.
    """
    out: Dict[str, int] = {}
    for row, name in enumerate(trip.cohort.subjects):
        data = trip.cohort.open(name)
        per_day = np.asarray(data.X[2]).sum(axis=1)
        watched = np.asarray(trip.cohort.masks[row]).reshape(-1, MINUTES_PER_DAY)
        out[name] = int(np.sum(watched.any(axis=1) & (per_day == 0)))
    return out


def episode_lengths(cohort: Cohort) -> Dict[str, Tuple[float, float]]:
    """Each label's mean and standard deviation of episode length, in slots."""
    mean, variance = mark_moments(cohort)
    return {
        name: (float(mean[j]), float(math.sqrt(variance[j])))
        for j, name in enumerate(cohort.label_names)
    }
