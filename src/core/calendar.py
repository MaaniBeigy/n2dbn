"""
core/calendar.py
================
Wall-clock semantics for the epoch indices of a ScaleHierarchy.

An epoch index carries no time unit by itself.  A SlotCalendar fixes the origin
and the duration of one finest-scale slot, from which the duration of an epoch
at every coarser scale follows through the branching factors.  With a one-minute
slot and branching [10, 6], scale 1 is a minute, scale 2 a ten-minute block and
scale 3 an hour.

The slot is a `timedelta`, so a scale may be as fine as a second or finer, and
`slot_minutes` remains available as a derived whole number of minutes that
raises rather than rounds when the slot is not one.

The clock is uniform.  Every epoch of a scale has the same duration and the
origin is a naive datetime, so a caller holding local times across a
daylight-saving boundary converts to a uniform clock before ingestion rather
than relying on this module to do it.  A timezone-aware origin is refused for
that reason.

The calendar is what turns reconstructed episodes into an event log with
`dd hh:mm` start and end times.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple, Union

DEFAULT_ORIGIN = datetime(2026, 1, 1, 0, 0)
DEFAULT_SLOT = timedelta(minutes=1)

_MINUTE = timedelta(minutes=1)
_SECOND = timedelta(seconds=1)
_DAY = timedelta(days=1)

# Scales whose epochs are not a constant number of minutes, coarsest last.  A
# month runs 28 to 31 days and a quarter 90 to 92, so these are named by a
# calendar rule instead of by a branching factor.  Each nests in the next: three
# months make a quarter and four quarters make a year, exactly.
CALENDAR_UNITS = ("month", "quarter", "year")


def _month_ordinal(when: datetime) -> int:
    """Months elapsed at `when`, counted from an arbitrary fixed epoch."""
    return when.year * 12 + when.month - 1


def _from_month_ordinal(n: int) -> datetime:
    return datetime(n // 12, n % 12 + 1, 1)


def _unit_ordinal(unit: str, when: datetime) -> int:
    """Index of the calendar unit containing `when`, on that unit's own count."""
    if unit == "month":
        return _month_ordinal(when)
    if unit == "quarter":
        return _month_ordinal(when) // 3
    return when.year


def _unit_start(unit: str, ordinal: int) -> datetime:
    """Wall-clock start of the calendar unit with the given index."""
    if unit == "month":
        return _from_month_ordinal(ordinal)
    if unit == "quarter":
        return _from_month_ordinal(ordinal * 3)
    return datetime(ordinal, 1, 1)


def require_naive(when: datetime, what: str) -> datetime:
    """Return `when` when it carries no timezone, and raise when it does."""
    if when.tzinfo is not None and when.utcoffset() is not None:
        raise ValueError(
            f"{what} is timezone-aware ({when!r}). The model is defined on a "
            "uniform clock, so convert to a single offset and drop the "
            "timezone before ingestion."
        )
    return when


@dataclass(frozen=True)
class SlotCalendar:
    """
    Maps (scale, epoch) pairs to wall-clock intervals.

    Attributes
    ----------
    origin     : datetime of the first finest-scale slot, naive
    slot       : duration of one finest-scale slot
    branching  : branching factors, so an epoch at scale k spans
                 slot x prod(branching[:k-1])
    """

    origin: datetime = DEFAULT_ORIGIN
    slot: timedelta = field(default=DEFAULT_SLOT)
    branching: tuple = ()
    units: tuple = ()

    def __post_init__(self) -> None:
        require_naive(self.origin, "calendar origin")
        if self.slot <= timedelta(0):
            raise ValueError(f"slot must be positive; got {self.slot!r}")
        if not self.units:
            return
        unknown = [u for u in self.units if u not in CALENDAR_UNITS]
        if unknown:
            raise ValueError(
                f"unknown calendar unit(s) {unknown}; expected one of "
                f"{list(CALENDAR_UNITS)}"
            )
        order = [CALENDAR_UNITS.index(u) for u in self.units]
        if order != sorted(set(order)) or len(order) != len(set(order)):
            raise ValueError(
                f"calendar units must go from finer to coarser without repeating; "
                f"got {list(self.units)}"
            )
        # A month starts at midnight on the first, so the epochs below it have to
        # land on day boundaries or the nesting is not a partition.
        finest_calendar_child = self.epoch_span(self.n_fixed_scales)
        if _DAY % finest_calendar_child:
            raise ValueError(
                f"scale {self.n_fixed_scales} spans {finest_calendar_child!r}, "
                "which does not divide one day, so it cannot nest inside "
                f"{self.units[0]!r}"
            )
        if self.origin != datetime(
            self.origin.year, self.origin.month, self.origin.day
        ):
            raise ValueError(
                f"origin {self.origin.isoformat()} is not at midnight, so day "
                f"boundaries do not fall on epoch boundaries and a "
                f"{self.units[0]!r} scale cannot nest"
            )

    # ------------------------------------------------------------------
    @classmethod
    def from_config(
        cls,
        branching: Sequence[int],
        slot_minutes: Optional[float] = None,
        origin: Optional[datetime] = None,
        slot: Optional[timedelta] = None,
        units: Sequence[str] = (),
    ) -> "SlotCalendar":
        """Build a calendar from a whole number of minutes or from a timedelta."""
        if slot is not None and slot_minutes is not None:
            raise ValueError("give either slot or slot_minutes, not both")
        if slot is None:
            minutes = 1.0 if slot_minutes is None else float(slot_minutes)
            if minutes != int(minutes):
                raise ValueError(
                    f"slot_minutes must be a whole number of minutes; got "
                    f"{slot_minutes!r}. Pass slot=timedelta(...) for a finer slot."
                )
            slot = timedelta(minutes=int(minutes))
        return cls(
            origin=origin or DEFAULT_ORIGIN,
            slot=slot,
            branching=tuple(int(b) for b in branching),
            units=tuple(str(u) for u in units),
        )

    # ------------------------------------------------------------------
    @property
    def slot_minutes(self) -> int:
        """Duration of one finest-scale slot as a whole number of minutes."""
        return _whole_minutes(self.slot, "slot")

    @property
    def n_fixed_scales(self) -> int:
        """Scales whose epochs are a constant multiple of the slot."""
        return len(self.branching) + 1

    @property
    def K(self) -> int:
        """Scales in the ladder, fixed ones and calendar ones together."""
        return self.n_fixed_scales + len(self.units)

    def unit_of(self, k: int) -> Optional[str]:
        """The calendar unit scale k names, or None when it is a fixed width."""
        if k <= self.n_fixed_scales:
            return None
        return self.units[k - self.n_fixed_scales - 1]

    def epoch_span(self, k: int) -> timedelta:
        """Duration of one epoch at scale k, which a calendar scale does not have."""
        unit = self.unit_of(k)
        if unit is not None:
            raise ValueError(
                f"scale {k} is a {unit!r}, whose epochs differ in length. "
                "Read span_of(k, t) for one epoch."
            )
        span = self.slot
        for b in self.branching[: k - 1]:
            span = span * int(b)
        return span

    def span_of(self, k: int, t: int) -> timedelta:
        """Duration of epoch t at scale k, whether or not the scale is regular."""
        return self.start_of(k, t + 1) - self.start_of(k, t)

    def epoch_minutes(self, k: int) -> int:
        """Duration of one epoch at scale k, as a whole number of minutes."""
        return _whole_minutes(self.epoch_span(k), f"epoch at scale {k}")

    def start_of(self, k: int, t: int) -> datetime:
        """Wall-clock start of epoch t at scale k."""
        unit = self.unit_of(k)
        if unit is None:
            return self.origin + self.epoch_span(k) * int(t)
        return _unit_start(unit, _unit_ordinal(unit, self.origin) + int(t))

    def end_of(self, k: int, t: int) -> datetime:
        """Wall-clock end of epoch t at scale k, exclusive."""
        return self.start_of(k, t + 1)

    def epoch_index_of(self, k: int, when: datetime) -> int:
        """Index of the epoch at scale k that contains `when`."""
        unit = self.unit_of(k)
        if unit is None:
            return int((when - self.origin) // self.epoch_span(k))
        return _unit_ordinal(unit, when) - _unit_ordinal(unit, self.origin)

    def parent_map(self, k: int, n_epochs: int) -> List[int]:
        """Parent epoch at scale k+1 for each of the first `n_epochs` at scale k."""
        if not 1 <= k < self.K:
            raise ValueError(f"scale {k} has no parent in a ladder of {self.K}")
        # A calendar epoch can open before the origin, as the season holding
        # 1 January opened the previous December, so the epoch is placed by its
        # first observed instant rather than by a start nobody recorded.
        return [
            self.epoch_index_of(k + 1, max(self.start_of(k, t), self.origin))
            for t in range(n_epochs)
        ]

    def hierarchy_maps(self, n_finest: int) -> Tuple[List[int], List[List[int]]]:
        """
        Epoch counts and parent maps for the whole ladder, finest first.

        A calendar scale holds as many epochs as the finest-scale horizon
        reaches into, so the counts are read off the clock rather than divided
        out of a branching factor.
        """
        counts = [int(n_finest)]
        maps: List[List[int]] = []
        for k in range(1, self.K):
            pm = self.parent_map(k, counts[-1])
            maps.append(pm)
            counts.append(pm[-1] + 1 if pm else 0)
        return counts, maps

    def stamp(self, k: int, t: int) -> str:
        """`dd hh:mm` label of the start of epoch t at scale k."""
        return format_stamp(self.start_of(k, t))

    def timespec(self) -> str:
        """The `isoformat` precision the slot width implies."""
        if self.slot % _SECOND:
            return "microseconds"
        if self.slot % _MINUTE:
            return "seconds"
        return "minutes"

    def shifted(self, n_slots: int) -> "SlotCalendar":
        """The same calendar with the origin advanced by n finest-scale slots."""
        return SlotCalendar(
            origin=self.origin + self.slot * int(n_slots),
            slot=self.slot,
            branching=self.branching,
            units=self.units,
        )

    def describe(self) -> Dict[int, str]:
        """Human-readable duration of one epoch at each scale."""
        out: Dict[int, str] = {}
        for k in range(1, self.K + 1):
            unit = self.unit_of(k)
            out[k] = f"1 {unit}" if unit else format_duration(self.epoch_span(k))
        return out


def _whole_minutes(span: timedelta, what: str) -> int:
    """The span as a whole number of minutes, raising when it is not one."""
    if span % _MINUTE:
        raise ValueError(
            f"{what} is {span!r}, which is not a whole number of minutes. "
            "Read the span as a timedelta instead."
        )
    return int(span // _MINUTE)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def format_stamp(when: datetime, origin: Optional[datetime] = None) -> str:
    """
    Format a datetime as `dd hh:mm`, with `dd` the day index from the origin.

    Day 1 is the day containing the origin, so an event log reads
    `01 09:15` for a quarter past nine on the first day.
    """
    ref = origin or datetime(when.year, when.month, when.day)
    day = (when.date() - ref.date()).days + 1
    return f"{day:02d} {when.hour:02d}:{when.minute:02d}"


def format_duration(span: Union[timedelta, int, float]) -> str:
    """Format a duration as the plainest correct unit, from seconds to weeks."""
    if not isinstance(span, timedelta):
        span = timedelta(minutes=float(span))
    units = (
        ("week", timedelta(weeks=1)),
        ("day", timedelta(days=1)),
        ("hour", timedelta(hours=1)),
        ("minute", _MINUTE),
        ("second", _SECOND),
    )
    for name, unit in units:
        if span >= unit and not span % unit:
            n = int(span // unit)
            return f"1 {name}" if n == 1 else f"{n} {name}s"
    total = span.total_seconds()
    return f"{total:g} seconds"


# ---------------------------------------------------------------------------
# Event log
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EventRecord:
    """One reconstructed episode with wall-clock start and end."""

    label: str
    start: datetime
    end: datetime
    duration: timedelta
    slot_start: int
    slot_end: int

    @property
    def duration_minutes(self) -> float:
        """Length of the episode in minutes, whole where the slot allows it."""
        minutes = self.duration / _MINUTE
        return int(minutes) if minutes == int(minutes) else minutes

    def as_row(self, origin: datetime, timespec: str = "minutes") -> Dict[str, object]:
        return {
            "label": self.label,
            "start": format_stamp(self.start, origin),
            "end": format_stamp(self.end, origin),
            "start_iso": self.start.isoformat(timespec=timespec),
            "end_iso": self.end.isoformat(timespec=timespec),
            "duration_minutes": self.duration_minutes,
            "slot_start": self.slot_start,
            "slot_end": self.slot_end,
        }


def episodes_to_event_log(
    traces: Dict[int, list],
    label_names: Sequence[str],
    calendar: SlotCalendar,
) -> List[EventRecord]:
    """
    Convert reconstructed finest-scale episodes into wall-clock event records.

    An episode occupying slots [t_start, t_end] inclusive starts at the wall
    clock time of slot t_start and ends at the end of slot t_end, so its
    duration is (t_end - t_start + 1) finest-scale slots.
    """
    records: List[EventRecord] = []
    for ell, episodes in sorted(traces.items()):
        name = label_names[ell] if ell < len(label_names) else f"L{ell}"
        for ep in episodes:
            records.append(
                EventRecord(
                    label=name,
                    start=calendar.start_of(1, ep.t_start),
                    end=calendar.end_of(1, ep.t_end),
                    duration=calendar.slot * (ep.t_end - ep.t_start + 1),
                    slot_start=int(ep.t_start),
                    slot_end=int(ep.t_end),
                )
            )
    records.sort(key=lambda r: (r.start, r.label))
    return records


def write_event_log(
    path: str,
    records: Sequence[EventRecord],
    calendar: SlotCalendar,
) -> None:
    """Write an event log as CSV with `dd hh:mm` and ISO timestamps."""
    fields = [
        "label",
        "start",
        "end",
        "start_iso",
        "end_iso",
        "duration_minutes",
        "slot_start",
        "slot_end",
    ]
    timespec = calendar.timespec()
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow(r.as_row(calendar.origin, timespec))
