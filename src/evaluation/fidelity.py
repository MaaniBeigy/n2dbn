"""
evaluation/fidelity.py
======================
Whether a log drawn from the fitted model holds the distribution of the log it
was fitted to.

The claim the model makes is distributional.  A replicate holds about the
episodes the record holds, in about the same calendar cells, with about the same
lengths, and the timestamps themselves are free.  This module draws replicates
and compares every summary the study names, cell by cell.

A cell is a set of epochs and one label, and it carries two numbers: the
episodes that began inside it and the slots those episodes booked, which is the
initiation reading of `core.durations`.  Every length here is in finest slots,
which are minutes on the benchmark's ladder.  The cells are the whole record at
every scale, the weekend against the weekday, the hour-of-day bins of
`features.hour_of_day`, and the week index, which are the splits the study's own
questions name.  Two families describe the shape of the log itself: how many
episodes fall in each length bin, and how many days hold each number of
episodes.

The verdict per cell is a posterior-predictive p-value, the fraction of
replicates at or above the observation.  Ties are common on a count, so the
mid-p, which splits the replicates equal to the observation, is reported beside
it and is what the calibration reads.  Under a correct model those p-values are
uniform, so the calibration reports the share falling outside a central interval
against the nominal rate.  A cell whose observation sits outside every replicate
is a misfit and is named.

Three things the check depends on.

A p-value near 0 or 1 on a cell holding two episodes says almost nothing, so
every cell carries the episode count behind it and the calibration drops the
cells below a declared floor.  The count is read on both sides, as the larger of
what the record holds and what the replicates draw, since a cell the record
leaves empty and the model fills heavily carries no observed episode at all and
a floor read off the record alone would drop the one reading that names
over-generation.  This filters; it never weights.

A replicate is drawn over the ancestor maps and the calendar columns the fit
read, and `intensity.sample` applies the same risk set the likelihood is written
over, so a replicate is a valid event log on the fit's own grid.  A check run on
any other grid measures the drawing code.

A replicate's coarse arrays are the sums of its own finest ones.  That is exact
by construction, so it is asserted on every replicate, and a failure says the
sampler and the ladder disagree.

The total family reports one comparison at every scale.  A coarse epoch counts
as observed when every finest slot under it is, so the observed epochs agree up
the ladder and the K rows of a label carry the same number.

One share over every cell answers a question about the average cell.  A model
whose morning window sits in the evening breaks a dozen cells of one label among
thousands, which moves that share by a thousandth and leaves the model passing.
So the cells are read again in families, one family per split and label.  Inside
a family every cell is centered and scaled over the members, which are the
observation and the replicates together, and the squares are summed into a
single discrepancy, so a family holds one number per member.  Each family's
discrepancy is then centered and scaled over the members in turn, which puts
families of different widths on one axis, and each member keeps the largest
figure any of its families reached.  The verdict is where the observation's
largest figure falls among the members', reported as the share of members at or
beyond it.

Every step of that reads the members through one symmetric function of all of
them, so under a correct model the members are exchangeable and the observation's
rank is uniform.  The share therefore lands on a multiple of 1/(1+R) with equal
probability whatever the number of families, and the multiplicity over families
is paid for exactly, with no correction to declare.

The floor the verdict reads is the one symmetric part of this that is easy to
lose.  The busier side of a cell is what says whether the record and the model
disagree there, and it is what the misfit list filters on, and it reads the
observation and the replicates differently.  A column set chosen that way holds
the observation's own high cells and none of the replicates', which is a bias
towards calling the model broken, so the verdict filters on the count averaged
over the members, which every member enters the same way.  A cell one
member alone is busy in is then left to the misfit list, which is where a label
the model draws nothing of is named.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.calendar import SlotCalendar
from ..core.features import (
    CalendarNode,
    home_scale,
    horizon_trend,
    hour_of_day,
    weekend,
)
from ..core.mask import reduce_by
from ..core.typed_graph import COUNT, DURATION, TypedGraph
from ..inference.bound import segment_sum
from ..model.intensity import FineParams, FineSources, empty_sources, sample

# What a histogram family counts, beside the count and the duration an epoch
# family carries.
EPISODES = "episodes"
DAYS = "days"

# Episode lengths in finest slots, the last bin open above.  A ladder of powers
# is what a length distribution is read on, since an episode of five slots and
# one of six are the same claim and one of three hundred is a different claim.
LENGTH_EDGES: Tuple[int, ...] = (1, 5, 15, 30, 60, 120, 240)

# Episodes in one day, the last bin open above.  Zero, one and two are separate
# cells because a stationary label spends most days in them.
PER_DAY_EDGES: Tuple[int, ...] = (0, 1, 2, 3, 5)

# Episodes a cell needs on its busier side before its p-value enters the
# calibration.
MIN_EPISODES = 5.0

# The spread a cell needs before its deviation from the members' center is
# scaled by that spread.  Every cell value is a whole count of episodes or of
# booked slots, so a spread under half a unit is a cell the members agree on and
# a one-unit difference there is a rounding, not an eight-sigma event.
SPREAD_FLOOR = 0.5

DEFAULT_INTERVAL = 0.95


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class CellKey:
    """One comparable number: a statistic of one label inside one cell."""

    family: str
    cell: str
    label: str
    statistic: str

    def __str__(self) -> str:
        return f"{self.label} {self.family} {self.cell} {self.statistic}"


@dataclass(frozen=True)
class EpochFamily:
    """
    One partition of a scale's epochs into cells.

    Attributes
    ----------
    name   : the split the cells come from
    scale  : the scale whose epochs are partitioned
    groups : (H_k,) the cell each epoch falls in, and -1 for an epoch the record
             does not cover, which then reaches no cell
    cells  : the cell names, indexed by `groups`
    """

    name: str
    scale: int
    groups: np.ndarray
    cells: Tuple[str, ...]


def _bin_index(values: np.ndarray, edges: Sequence[int]) -> np.ndarray:
    """
    The bin each value falls in, over edges whose last bin is open above.

    Raises:
        ValueError: when a value falls below the first edge, where no bin is.
    """
    idx = np.searchsorted(np.asarray(edges), np.asarray(values), side="right") - 1
    if idx.size and int(idx.min()) < 0:
        raise ValueError(
            f"a value of {int(np.min(values))} falls below the first bin edge "
            f"{edges[0]}, so it belongs to no bin"
        )
    return idx


def _bin_names(edges: Sequence[int]) -> Tuple[str, ...]:
    """One name per bin, the last one open above."""
    pairs = list(zip(edges, edges[1:]))
    return tuple([f"[{lo},{hi})" for lo, hi in pairs] + [f"[{edges[-1]},inf)"])


def _node_cells(
    node: CalendarNode, raw: np.ndarray
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """
    One calendar node's epoch values as cell indices, with the cells they index.

    A one-hot node keeps every declared level as a cell and a binned node every
    declared bin, so the reference level the design drops is still a cell here,
    where the question is what the record holds and no contrast is being fitted.

    Raises:
        ValueError: when an epoch's value sits at or above the node's top bin
            edge, or is not one of the levels the node declares.
    """
    values = np.asarray(raw, dtype=np.int64)
    if node.encoding == "bins":
        groups = _bin_index(values, node.edges)
        if groups.size and int(groups.max()) >= len(node.edges) - 1:
            raise ValueError(
                f"{node.name!r} reads {int(values.max())} at some epoch, at or "
                f"above its top bin edge {node.edges[-1]}, so that epoch falls "
                "in no declared bin"
            )
        pairs = list(zip(node.edges, node.edges[1:]))
        return groups, tuple(f"[{lo},{hi})" for lo, hi in pairs)
    if node.encoding == "onehot":
        index = {int(v): j for j, v in enumerate(node.levels)}
        missing = sorted(set(values.tolist()) - set(index))
        if missing:
            raise ValueError(
                f"{node.name!r} reads {missing} at some epoch, which its declared "
                f"levels {list(node.levels)} do not hold"
            )
        groups = np.asarray([index[v] for v in values.tolist()], dtype=np.int64)
        return groups, tuple(str(v) for v in node.levels)
    present = np.unique(values)
    return np.searchsorted(present, values), tuple(str(int(v)) for v in present)


def _calendar_nodes(calendar: SlotCalendar) -> Tuple[CalendarNode, ...]:
    """The splits the study's questions name, each placed at its home scale."""
    return (hour_of_day(), weekend(), horizon_trend(calendar.origin, "week"))


def _day_scale(calendar: SlotCalendar, K: int) -> int:
    """
    The scale whose epochs are one day, which the per-day family is read at.

    Raises:
        ValueError: when no scale of the ladder spans a day.
    """
    for k in range(1, K + 1):
        if calendar.unit_of(k) is None and calendar.epoch_span(k) == timedelta(days=1):
            return k
    raise ValueError(
        f"no scale of this ladder of {K} spans one day, so the episodes a day "
        "holds is a quantity the ladder cannot express"
    )


def _group_sums(array: np.ndarray, groups: np.ndarray, n_cells: int) -> np.ndarray:
    """(n_cells, L) sums of the rows each cell holds, rows keyed -1 dropped."""
    values = np.asarray(array, dtype=float)
    # The indicator times the array is the same sum `np.add.at` gives and runs
    # in BLAS, which the finest scale of a one-minute ladder needs.
    indicator = np.zeros((values.shape[0], n_cells))
    rows = np.flatnonzero(groups >= 0)
    indicator[rows, groups[rows]] = 1.0
    return indicator.T @ values


def _histogram(bins: np.ndarray, labels: np.ndarray, n_bins: int, L: int) -> np.ndarray:
    """(n_bins, L) counts of the items each bin and label holds."""
    flat = np.bincount(bins * L + labels, minlength=n_bins * L)
    return flat.astype(float).reshape(n_bins, L)


# ---------------------------------------------------------------------------
# The grid of cells
# ---------------------------------------------------------------------------


@dataclass
class CellGrid:
    """
    Every cell the check compares, fixed once from the clock the fit ran on.

    Attributes
    ----------
    labels       : the label axis, in the column order of the arrays
    ancestors    : the scale-k epoch of every finest slot, finest first
    n_epochs     : epochs at each scale, finest first
    observed     : (T_1,) bool, the slots the record covers
    observed_at  : the mask AND-reduced into each scale
    families     : the epoch partitions, the totals first
    length_edges : bins of the episode-length family, in finest slots
    per_day_edges: bins of the episodes-per-day family
    day_scale    : the scale the per-day family is read at
    keys         : one key per comparable number, in the order `summarize` holds
    support      : for each key, the key whose observed value is the episode
                   count behind it, which is what the calibration filters on
    """

    labels: Tuple[str, ...]
    ancestors: List[np.ndarray]
    n_epochs: List[int]
    observed: np.ndarray
    observed_at: List[np.ndarray]
    families: Tuple[EpochFamily, ...]
    length_edges: Tuple[int, ...]
    per_day_edges: Tuple[int, ...]
    day_scale: int
    keys: Tuple[CellKey, ...] = field(init=False)
    support: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        keys: List[CellKey] = []
        support: List[int] = []
        for family in self.families:
            for cell in family.cells:
                for label in self.labels:
                    here = len(keys)
                    keys.append(CellKey(family.name, cell, label, COUNT))
                    keys.append(CellKey(family.name, cell, label, DURATION))
                    support.extend([here, here])
        for name, statistic, edges in (
            ("episode_length", EPISODES, self.length_edges),
            ("episodes_per_day", DAYS, self.per_day_edges),
        ):
            for cell in _bin_names(edges):
                for label in self.labels:
                    support.append(len(keys))
                    keys.append(CellKey(name, cell, label, statistic))
        self.keys = tuple(keys)
        self.support = np.asarray(support, dtype=np.int64)

    # ------------------------------------------------------------------
    @property
    def L(self) -> int:
        return len(self.labels)

    @property
    def n_slots(self) -> int:
        return int(self.n_epochs[0])

    @classmethod
    def build(
        cls,
        calendar: SlotCalendar,
        sources: FineSources,
        labels: Sequence[str],
        observed: Optional[np.ndarray] = None,
        length_edges: Sequence[int] = LENGTH_EDGES,
        per_day_edges: Sequence[int] = PER_DAY_EDGES,
    ) -> "CellGrid":
        """
        The cells of one fit, read off the ladder and the clock the fit used.

        Args:
            calendar: the clock behind the finest slots.
            sources: the fit's own sources, which carry the ancestor maps and the
                epoch counts every cell is defined over.
            labels: the label axis in column order.
            observed: (T_1,) bool, the slots the record covers, or None for a
                record covered end to end.

        Raises:
            ValueError: when the label axis disagrees with the arrays, when the
                mask is not one boolean flag per finest slot, or when no scale
                of the ladder spans a day.
        """
        ancestors = [np.asarray(a, dtype=np.int64) for a in sources.ancestors]
        n_epochs = [int(c.shape[0]) for c in sources.counts]
        if len(labels) != sources.n_labels:
            raise ValueError(
                f"the label axis carries {len(labels)} names for arrays of "
                f"{sources.n_labels} columns, so a cell would be named after "
                "another label's episodes"
            )
        if observed is None:
            flags = np.ones(n_epochs[0], dtype=bool)
        else:
            flags = np.asarray(observed)
            if flags.dtype != np.bool_:
                raise ValueError(
                    f"an observation mask is a boolean array; this one has dtype "
                    f"{flags.dtype}, which reads as a flag wherever it is non-zero"
                )
            if flags.shape != (n_epochs[0],):
                raise ValueError(
                    f"an observation mask carries one flag per finest slot, so "
                    f"{n_epochs[0]} of them; this one is {flags.shape}"
                )
        seen = [reduce_by(flags, ancestors[k], n) for k, n in enumerate(n_epochs)]

        families = [
            EpochFamily("total", k + 1, np.where(seen[k], 0, -1), (f"k{k + 1}",))
            for k in range(len(n_epochs))
        ]
        for node in _calendar_nodes(calendar):
            k = home_scale(calendar, node, n_epochs)
            raw = np.asarray(
                [node.value(calendar.start_of(k, t)) for t in range(n_epochs[k - 1])]
            )
            groups, cells = _node_cells(node, raw)
            families.append(
                EpochFamily(node.name, k, np.where(seen[k - 1], groups, -1), cells)
            )

        return cls(
            labels=tuple(str(name) for name in labels),
            ancestors=ancestors,
            n_epochs=n_epochs,
            observed=flags,
            observed_at=seen,
            families=tuple(families),
            length_edges=tuple(int(e) for e in length_edges),
            per_day_edges=tuple(int(e) for e in per_day_edges),
            day_scale=_day_scale(calendar, len(n_epochs)),
        )

    # ------------------------------------------------------------------
    def summarize(
        self,
        X1: np.ndarray,
        Y1: np.ndarray,
        counts: Sequence[np.ndarray],
        durations: Sequence[np.ndarray],
    ) -> np.ndarray:
        """
        Every cell's value for one log, in the order `keys` names them.

        The epoch families read the ladder the log already carries, so an hourly
        split costs a sum over the hours the record holds.

        Args:
            X1: (T_1, L) the log's finest counts.
            Y1: (T_1, L) the log's finest durations, initiation reading.
            counts: the log's count arrays, finest first.
            durations: the log's duration arrays, finest first.
        """
        blocks: List[np.ndarray] = []
        for family in self.families:
            n = len(family.cells)
            episodes = _group_sums(counts[family.scale - 1], family.groups, n)
            minutes = _group_sums(durations[family.scale - 1], family.groups, n)
            blocks.append(np.stack([episodes, minutes], axis=2).ravel())
        blocks.append(self._episode_length(X1, Y1).ravel())
        blocks.append(self._episodes_per_day(counts).ravel())
        return np.concatenate(blocks)

    def _episode_length(self, X1: np.ndarray, Y1: np.ndarray) -> np.ndarray:
        """
        Starts of each label whose recorded length falls in each bin.

        A slot holding two starts of one label carries their summed length as a
        single mark, so it enters one bin here while the total family counts both
        episodes.  That gap is the collision the finest scale cannot hold apart.
        """
        starts = (np.asarray(X1) > 0) & self.observed[:, None]
        slot, label = np.nonzero(starts)
        lengths = np.asarray(Y1)[slot, label]
        return _histogram(
            _bin_index(lengths, self.length_edges),
            label,
            len(self.length_edges),
            self.L,
        )

    def _episodes_per_day(self, counts: Sequence[np.ndarray]) -> np.ndarray:
        """Days of the record holding each number of episodes of each label."""
        per_day = np.asarray(counts[self.day_scale - 1])[
            self.observed_at[self.day_scale - 1]
        ]
        label = np.tile(np.arange(self.L), per_day.shape[0])
        return _histogram(
            _bin_index(per_day.ravel(), self.per_day_edges),
            label,
            len(self.per_day_edges),
            self.L,
        )


# ---------------------------------------------------------------------------
# The identity every replicate satisfies
# ---------------------------------------------------------------------------


def check_reconciles(
    X1: np.ndarray,
    Y1: np.ndarray,
    counts: Sequence[np.ndarray],
    durations: Sequence[np.ndarray],
    ancestors: Sequence[np.ndarray],
) -> None:
    """
    Refuse a log whose coarse arrays disagree with the sums of its own finest ones.

    Summation up the ladder is the whole cross-scale mechanism, so this holds by
    construction on anything the sampler wrote, and a failure says the sampler
    and the ladder disagree.

    Raises:
        ValueError: at the first scale and quantity where the sum disagrees.
    """
    for k, who in enumerate(ancestors):
        for name, fine, coarse in (
            (COUNT, X1, counts[k]),
            (DURATION, Y1, durations[k]),
        ):
            fine = np.asarray(fine, dtype=float)
            n = int(np.shape(coarse)[0])
            summed = fine if k == 0 else segment_sum(fine, who, n)
            stored = np.asarray(coarse, dtype=float)
            if not np.array_equal(summed, stored):
                gaps = np.flatnonzero(np.abs(summed - stored).sum(axis=1))
                epoch = int(gaps[0])
                raise ValueError(
                    f"the scale {k + 1} {name} array disagrees with the sum of the "
                    f"finest one, first at epoch {epoch}, where it holds "
                    f"{stored[epoch].sum():.0f} against a sum of "
                    f"{summed[epoch].sum():.0f}"
                )


def replicate(
    params: FineParams,
    graph: TypedGraph,
    sources: FineSources,
    rng: np.random.Generator,
    offsets: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, FineSources]:
    """
    One log drawn over the fit's own ancestor maps and calendar columns.

    The draw fills a fresh source set, so the fit's own arrays stay as they were
    read and the replicate carries a ladder of its own.

    Returns:
        The replicate's finest counts and durations, and its filled sources.
    """
    fresh = empty_sources(sources.ancestors, sources.calendar, sources.n_labels)
    X1, Y1 = sample(params, graph, fresh, rng, offsets)
    check_reconciles(X1, Y1, fresh.counts, fresh.durations, fresh.ancestors)
    return X1, Y1, fresh


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellCheck:
    """
    Where one cell's observation falls in the distribution of the replicates.

    Attributes
    ----------
    key            : what number this is
    observed       : the value the record holds
    replicate_mean : the mean over replicates
    low, high      : the central interval of the replicates
    p_value        : the fraction of replicates at or above the observation,
                     which is the upper tail the check is stated in
    p_mid          : the same with the replicates equal to it split in half,
                     which is the figure the calibration reads
    support        : episodes behind the cell, the larger of what the record
                     holds and what the replicates draw, which is what the
                     calibration and the misfit list filter on
    shared_support : the same count averaged over the members, which is what
                     the verdict filters on, since a column set the observation
                     helped choose would leave its own rank short of uniform
    outside        : whether the observation sits beyond every replicate
    """

    key: CellKey
    observed: float
    replicate_mean: float
    low: float
    high: float
    p_value: float
    p_mid: float
    support: float
    shared_support: float
    outside: bool

    def __str__(self) -> str:
        return (
            f"{self.key}: observed {self.observed:.0f}, replicates "
            f"{self.replicate_mean:.1f} [{self.low:.0f}, {self.high:.0f}], "
            f"p {self.p_value:.3f}"
        )


@dataclass(frozen=True)
class Calibration:
    """
    How uniform the p-values are over the cells that carry enough episodes.

    Attributes
    ----------
    n_cells       : cells the share was taken over
    n_sparse      : cells dropped for holding too few episodes on either side
    min_episodes  : the floor they were dropped by
    nominal       : the share expected outside the interval under a correct model
    outside_share : the share actually outside it, NaN when no cell survived
    n_misfits     : cells whose observation sits beyond every replicate
    """

    n_cells: int
    n_sparse: int
    min_episodes: float
    nominal: float
    outside_share: float
    n_misfits: int

    def __str__(self) -> str:
        return (
            f"{self.n_cells} cells over {self.min_episodes:.0f} episodes, "
            f"{self.n_sparse} dropped under it; p-values outside the central "
            f"interval {self.outside_share:.3f} against a nominal "
            f"{self.nominal:.3f}; {self.n_misfits} misfits"
        )


@dataclass(frozen=True)
class FamilyCheck:
    """
    How far one split of one label stands from the replicates, over all its cells.

    Attributes
    ----------
    family, label  : the split and the label whose cells were summed
    n_cells        : cells over the floor that entered the sum
    discrepancy    : the observation's summed squared deviation from the members
    replicate_mean : the mean of the same sum over the replicates
    low, high      : the central interval of the replicates' sums
    standardized   : the observation's sum, centered and scaled over the members,
                     which is the axis families of different widths share
    p_value        : the share of members whose sum is at or above the
                     observation's, which never falls below 1/(1+R)
    """

    family: str
    label: str
    n_cells: int
    discrepancy: float
    replicate_mean: float
    low: float
    high: float
    standardized: float
    p_value: float

    def __str__(self) -> str:
        return (
            f"{self.label} {self.family} over {self.n_cells} cells: discrepancy "
            f"{self.discrepancy:.1f}, replicates {self.replicate_mean:.1f} "
            f"[{self.low:.1f}, {self.high:.1f}], z {self.standardized:.2f}, "
            f"p {self.p_value:.4f}"
        )


@dataclass(frozen=True)
class Verdict:
    """
    Whether any one family stands further out than a correct model would put it.

    Attributes
    ----------
    n_families : families the maximum was taken over
    n_cells    : cells over the floor those families hold
    worst      : the observation's furthest family, None when no family holds
                 cells the members read differently
    p_value    : the share of members whose own furthest family stands at or
                 beyond the observation's, which is the family-wise figure
    nominal    : the level the p-value is declared against
    """

    n_families: int
    n_cells: int
    worst: Optional[FamilyCheck]
    p_value: float
    nominal: float

    @property
    def broken(self) -> bool:
        return self.p_value <= self.nominal

    def __str__(self) -> str:
        if self.worst is None:
            return "no family holds a comparison, so the verdict is empty"
        return (
            f"worst of {self.n_families} families over {self.n_cells} cells, "
            f"{self.worst}; family-wise p {self.p_value:.4f} against a nominal "
            f"{self.nominal:.3f}"
        )


def _family_columns(
    cells: Sequence[CellCheck], min_episodes: float
) -> Dict[Tuple[str, str], List[int]]:
    """The cells over the floor each split-and-label pair holds, in key order."""
    columns: Dict[Tuple[str, str], List[int]] = {}
    for j, cell in enumerate(cells):
        if cell.shared_support >= min_episodes:
            columns.setdefault((cell.key.family, cell.key.label), []).append(j)
    return columns


@dataclass(frozen=True)
class FidelityReport:
    """Every cell's check, and what the set of them says about the model."""

    cells: Tuple[CellCheck, ...]
    members: np.ndarray
    interval: float

    @property
    def n_replicates(self) -> int:
        return int(self.members.shape[0]) - 1

    def misfits(self, min_episodes: float = MIN_EPISODES) -> List[CellCheck]:
        """Cells whose observation sits beyond every replicate, busiest first."""
        named = [c for c in self.cells if c.outside and c.support >= min_episodes]
        return sorted(named, key=lambda c: (-c.support, c.key))

    def calibration(self, min_episodes: float = MIN_EPISODES) -> Calibration:
        """The share of p-values outside the central interval, against nominal."""
        nominal = 1.0 - self.interval
        kept = [c for c in self.cells if c.support >= min_episodes]
        flags = [c.p_mid < nominal / 2.0 or c.p_mid > 1.0 - nominal / 2.0 for c in kept]
        return Calibration(
            n_cells=len(kept),
            n_sparse=len(self.cells) - len(kept),
            min_episodes=float(min_episodes),
            nominal=nominal,
            outside_share=float(np.mean(flags)) if kept else float("nan"),
            n_misfits=len(self.misfits(min_episodes)),
        )

    def _discrepancy(
        self, min_episodes: float
    ) -> Tuple[Tuple[Tuple[str, str], ...], np.ndarray, np.ndarray]:
        """
        Each family's summed squared deviation, member by member, raw and scaled.

        Returns:
            The split-and-label pairs in a fixed order, the (1 + R, G) sums, and
            the same sums centered and scaled over the members.
        """
        columns = _family_columns(self.cells, min_episodes)
        ordered = sorted(columns)
        if not ordered:
            empty = np.zeros((self.members.shape[0], 0))
            return (), empty, empty
        center = self.members.mean(axis=0)
        spread = np.maximum(self.members.std(axis=0), SPREAD_FLOOR)
        squared = ((self.members - center) / spread) ** 2
        stat = np.stack([squared[:, columns[n]].sum(axis=1) for n in ordered], axis=1)
        sd = stat.std(axis=0)
        # A family every member scores the same on would put one figure on the
        # axis for all of them, which ties the members that reach no further
        # anywhere else, so it leaves the maximum.
        alive = sd > 0.0
        names = tuple(n for n, keep in zip(ordered, alive) if keep)
        stat = stat[:, alive]
        return names, stat, (stat - stat.mean(axis=0)) / sd[alive]

    def families(self, min_episodes: float = MIN_EPISODES) -> List[FamilyCheck]:
        """Every family's discrepancy against the replicates', furthest first."""
        columns = _family_columns(self.cells, min_episodes)
        names, stat, scaled = self._discrepancy(min_episodes)
        tail = (1.0 - self.interval) / 2.0
        out = [
            FamilyCheck(
                family=family,
                label=label,
                n_cells=len(columns[(family, label)]),
                discrepancy=float(stat[0, g]),
                replicate_mean=float(stat[1:, g].mean()),
                low=float(np.quantile(stat[1:, g], tail)),
                high=float(np.quantile(stat[1:, g], 1.0 - tail)),
                standardized=float(scaled[0, g]),
                p_value=float((stat[:, g] >= stat[0, g]).mean()),
            )
            for g, (family, label) in enumerate(names)
        ]
        return sorted(out, key=lambda f: (-f.standardized, f.label, f.family))

    def verdict(self, min_episodes: float = MIN_EPISODES) -> Verdict:
        """Where the observation's furthest family falls among the members'."""
        names, _, scaled = self._discrepancy(min_episodes)
        nominal = 1.0 - self.interval
        if not names:
            return Verdict(
                n_families=0,
                n_cells=0,
                worst=None,
                p_value=float("nan"),
                nominal=nominal,
            )
        furthest = scaled.max(axis=1)
        ranked = self.families(min_episodes)
        return Verdict(
            n_families=len(names),
            n_cells=sum(f.n_cells for f in ranked),
            worst=ranked[0],
            p_value=float((furthest >= furthest[0]).mean()),
            nominal=nominal,
        )

    def report(self, min_episodes: float = MIN_EPISODES, named: int = 10) -> str:
        """The check as text, with the verdict and the worst misfits named."""
        summary = self.calibration(min_episodes)
        lines = [
            f"posterior predictive check over {self.n_replicates} replicates, "
            f"{self.interval:.2f} interval",
            f"  {summary}",
            f"  {self.verdict(min_episodes)}",
        ]
        worst = self.misfits(min_episodes)
        lines.extend(f"  {cell}" for cell in worst[:named])
        if len(worst) > named:
            lines.append(f"  and {len(worst) - named} further misfits")
        return "\n".join(lines)


def posterior_predictive(
    X1: np.ndarray,
    Y1: np.ndarray,
    params: FineParams,
    graph: TypedGraph,
    sources: FineSources,
    grid: CellGrid,
    rng: np.random.Generator,
    n_replicates: int = 200,
    offsets: Optional[Dict[str, np.ndarray]] = None,
    interval: float = DEFAULT_INTERVAL,
) -> FidelityReport:
    """
    Draw replicate logs and place the observation in each cell's replicate spread.

    Args:
        X1, Y1: the observed finest arrays, whose ladder `sources` carries.
        params, graph, sources: the fit, read exactly as the likelihood reads it.
        grid: the cells, built over the same ladder and clock.
        rng: the generator every replicate is drawn from.
        n_replicates: how many logs to draw.
        offsets: what a subject level or a latent part adds to each predictor,
            the same argument the likelihood was evaluated with.
        interval: the central mass the reported interval covers.

    Raises:
        ValueError: when fewer than two replicates are asked for, when the
            observed arrays are not one row per finest slot and one column per
            label, or when either log fails to reconcile.
    """
    if n_replicates < 2:
        raise ValueError(
            f"{n_replicates} replicate(s) give no spread to place the observation "
            "in; a posterior-predictive interval needs at least two"
        )
    shape = (grid.n_slots, grid.L)
    if np.shape(X1) != shape or np.shape(Y1) != shape:
        raise ValueError(
            f"the observed arrays are {np.shape(X1)} and {np.shape(Y1)} where the "
            f"grid holds {shape[0]} slots and {shape[1]} labels"
        )

    check_reconciles(X1, Y1, sources.counts, sources.durations, grid.ancestors)
    members = np.empty((int(n_replicates) + 1, len(grid.keys)))
    members[0] = grid.summarize(X1, Y1, sources.counts, sources.durations)
    for r in range(int(n_replicates)):
        rep_X1, rep_Y1, rep = replicate(params, graph, sources, rng, offsets)
        members[r + 1] = grid.summarize(rep_X1, rep_Y1, rep.counts, rep.durations)

    observed, draws = members[0], members[1:]
    tail = (1.0 - interval) / 2.0
    low = np.quantile(draws, tail, axis=0)
    high = np.quantile(draws, 1.0 - tail, axis=0)
    above = (draws > observed).mean(axis=0)
    equal = (draws == observed).mean(axis=0)
    beyond = above + equal
    outside = (observed < draws.min(axis=0)) | (observed > draws.max(axis=0))
    backing = members[:, grid.support]
    support = np.maximum(backing[0], backing[1:].mean(axis=0))
    shared = backing.mean(axis=0)

    cells = tuple(
        CellCheck(
            key=key,
            observed=float(observed[j]),
            replicate_mean=float(draws[:, j].mean()),
            low=float(low[j]),
            high=float(high[j]),
            p_value=float(beyond[j]),
            p_mid=float(above[j] + 0.5 * equal[j]),
            support=float(support[j]),
            shared_support=float(shared[j]),
            outside=bool(outside[j]),
        )
        for j, key in enumerate(grid.keys)
    )
    return FidelityReport(cells=cells, members=members, interval=float(interval))


# ---------------------------------------------------------------------------
# The split check, which the plug-in check is biased against
# ---------------------------------------------------------------------------


def chronological_split(
    ancestors: Sequence[np.ndarray],
    scale: int,
    blocks: Sequence[Tuple[int, int]],
    fraction: float = 0.25,
) -> np.ndarray:
    """
    The last share of each block's epochs, as a mask over finest slots.

    Args:
        ancestors: the scale-k epoch of every finest slot, finest first.
        scale: the scale the split is taken at, so a held-out span is a whole
            number of epochs there.
        blocks: the half-open epoch range of each participant at that scale.
        fraction: the share of each block held out.

    Raises:
        ValueError: when the fraction leaves either side of the split empty.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(
            f"a split fraction sits strictly inside (0, 1); got {fraction}"
        )
    who = np.asarray(ancestors[scale - 1], dtype=np.int64)
    held_epochs = np.zeros(int(who.max(initial=-1)) + 1, dtype=bool)
    for start, stop in blocks:
        span = stop - start
        cut = stop - int(round(span * fraction))
        if cut <= start or cut >= stop:
            raise ValueError(
                f"a {fraction} split of the block [{start}, {stop}) leaves one "
                "side empty"
            )
        held_epochs[cut:stop] = True
    return held_epochs[who]


def split_predictive(
    X1: np.ndarray,
    Y1: np.ndarray,
    refit,
    calendar: SlotCalendar,
    labels: Sequence[str],
    sources: FineSources,
    observed: np.ndarray,
    held: np.ndarray,
    rng: np.random.Generator,
    n_replicates: int = 200,
    interval: float = DEFAULT_INTERVAL,
) -> FidelityReport:
    """
    Fit on the training slots and place the held-out observation in the spread.

    `posterior_predictive` compares a record to replicates drawn from a fit that
    read that same record, so the replicates are pulled toward the observation
    and the check is biased toward passing.  Here the fit never sees the held-out
    slots and the cells are read only there, so a p-value carries what it claims
    to carry.

    Args:
        refit: called with the training mask and returning the parameters, the
            graph and the offsets fitted without the held-out slots.

    Raises:
        ValueError: when the split leaves either side of the record empty.
    """
    observed = np.asarray(observed, dtype=bool)
    held = np.asarray(held, dtype=bool) & observed
    train = observed & ~held
    if not train.any():
        raise ValueError("the split leaves no training slot for the fit to read")
    if not held.any():
        raise ValueError("the split holds out no observed slot to score")

    params, graph, offsets = refit(train)
    grid = CellGrid.build(calendar, sources, labels, observed=held)
    return posterior_predictive(
        X1,
        Y1,
        params,
        graph,
        sources,
        grid,
        rng,
        n_replicates=n_replicates,
        offsets=offsets,
        interval=interval,
    )
