"""
data/anchors.py
===============
The anchor set of the observed log, frozen as data and hashed.

An anchor is a statement about one relation at one resolution, keyed by
(scale, lag, source, target) and carrying the reason it is asserted.  A
positive is a relation the study protocol guarantees; a negative is one the
protocol forbids.  The set is built from declared rules and digested, so a run
that reports a hit rate names the set it scored against and a change to the set
changes the digest.

The interesting property is that the same relation is a positive at one
resolution and a negative at another.  `notification_received -> read` is a
mechanism: the median receipt-to-read lag is 22 minutes and three reads in four
land more than a minute after the nearest preceding receipt, so at the minute
and the hour it is a lagged relation.  At the day and the week essentially every
pair is same-epoch, `read <= received` holds on 99.7 percent of participant-days
and the reversed day-lag-1 correlation (0.37) exceeds the forward one (0.28),
so a lagged edge there is wrong.  The day-scale co-movement is real and belongs
to the latent factor, which is where a relation shorter than the epoch sits.

Applying the collapse of `label_axis` before the anchors are scored moves most
of one negative family out of the ladder and into the type system: nothing can
point into the scheduled label, so those negatives are structural and are never
a test.  Each anchor carries which of the two it is, and the ladder counts them
apart.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from ..core.typed_graph import TypedEdge
from .label_axis import SCHEDULED_LABEL, LabelAxis
from .lifesnaps_ladder import LADDER_UNITS

POSITIVE = "positive"
NEGATIVE = "negative"

STRUCTURAL = "structural"
ASSERTABLE = "assertable"

#: The relation the protocol guarantees, and the raw column it lands in.  The
#: anchor is keyed on the vertex that column becomes, which is what an accepted
#: edge can name.
ANCHOR_SOURCE = SCHEDULED_LABEL
ANCHOR_READ = "notification_read"

#: A protocol statement is about the relation, so it is asserted at every lag a
#: search may propose, and the search proposes lags 1 to `MAX_LAG` at each scale.
MAX_LAG = 3

#: Where the receipt-to-read mechanism is a lagged relation, and where it is not.
POSITIVE_SCALES: Tuple[str, ...] = ("minute", "hour")
NEGATIVE_SCALES: Tuple[str, ...] = ("day", "week")

_LAG_REASON = (
    "the median receipt-to-read lag is 22 minutes and three reads in four land "
    "more than a minute after the nearest preceding receipt"
)
_EPOCH_REASON = (
    "essentially every receipt-read pair is same-epoch here, read <= received "
    "holds on 99.7 percent of participant-days, and the reversed lag-1 "
    "correlation exceeds the forward one, so the co-movement belongs to the "
    "latent factor and a lagged edge has mislabelled the scale"
)
_SCHEDULE_REASON = (
    "the study delivers the notification on a schedule the participant does not "
    "set, so no behaviour precedes it"
)


@dataclass(frozen=True, order=True)
class AnchorKey:
    """One relation at one resolution."""

    scale: str
    lag: int
    source: str
    target: str

    def as_row(self) -> Tuple[str, int, str, str]:
        return (self.scale, self.lag, self.source, self.target)


@dataclass(frozen=True, order=True)
class Anchor:
    """One anchor, with its sign, its reason and whether the axis can express it."""

    key: AnchorKey
    sign: str
    standing: str
    justification: str

    @property
    def assertable(self) -> bool:
        return self.standing == ASSERTABLE


@dataclass(frozen=True)
class AnchorSet:
    """The frozen anchors, with the digest a run reports beside its hit rate."""

    anchors: Tuple[Anchor, ...]
    axis_digest: str

    def __len__(self) -> int:
        return len(self.anchors)

    def of_sign(self, sign: str, assertable_only: bool = False) -> List[Anchor]:
        """Every anchor of one sign, optionally only the ones the axis can express."""
        return [
            a
            for a in self.anchors
            if a.sign == sign and (a.assertable or not assertable_only)
        ]

    def digest(self) -> str:
        """A stable digest of the whole set, including every reason it carries."""
        payload = "\n".join(
            "|".join(
                [
                    a.key.scale,
                    str(a.key.lag),
                    a.key.source,
                    a.key.target,
                    a.sign,
                    a.standing,
                    a.justification,
                ]
            )
            for a in sorted(self.anchors)
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def as_rows(self) -> List[Dict[str, object]]:
        return [
            {
                "scale": a.key.scale,
                "lag": a.key.lag,
                "source": a.key.source,
                "target": a.key.target,
                "sign": a.sign,
                "standing": a.standing,
                "justification": a.justification,
            }
            for a in sorted(self.anchors)
        ]

    def freeze(self, path: str) -> Dict[str, object]:
        """Write the set and its digest, which is what a scoring run reads."""
        record = {
            "digest": self.digest(),
            "axis_digest": self.axis_digest,
            "n_anchors": len(self.anchors),
            "n_assertable": sum(1 for a in self.anchors if a.assertable),
            "anchors": self.as_rows(),
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        return record


def anchor_set(axis: LabelAxis, max_lag: int = MAX_LAG) -> AnchorSet:
    """Build the declared anchors for one axis."""
    lags = range(1, max_lag + 1)
    read = axis.axis_name(ANCHOR_READ)
    anchors: List[Anchor] = []

    for scale in POSITIVE_SCALES:
        for lag in lags:
            anchors.append(
                Anchor(
                    key=AnchorKey(scale, lag, ANCHOR_SOURCE, read),
                    sign=POSITIVE,
                    standing=_standing(axis, ANCHOR_SOURCE, read),
                    justification=_LAG_REASON,
                )
            )
    for scale in NEGATIVE_SCALES:
        for lag in lags:
            anchors.append(
                Anchor(
                    key=AnchorKey(scale, lag, ANCHOR_SOURCE, read),
                    sign=NEGATIVE,
                    standing=_standing(axis, ANCHOR_SOURCE, read),
                    justification=_EPOCH_REASON,
                )
            )
    for source in axis.targets:
        for scale in LADDER_UNITS:
            for lag in lags:
                anchors.append(
                    Anchor(
                        key=AnchorKey(scale, lag, source, SCHEDULED_LABEL),
                        sign=NEGATIVE,
                        standing=_standing(axis, source, SCHEDULED_LABEL),
                        justification=_SCHEDULE_REASON,
                    )
                )
    return AnchorSet(anchors=tuple(sorted(anchors)), axis_digest=axis.digest())


def _standing(axis: LabelAxis, source: str, target: str) -> str:
    """Whether the axis lets a search state this relation at all."""
    return ASSERTABLE if axis.expressible(source, target) else STRUCTURAL


def key_of(
    edge: TypedEdge,
    dst_label: int,
    axis: LabelAxis,
    units: Sequence[str] = LADDER_UNITS,
) -> AnchorKey:
    """The anchor key one accepted edge states."""
    names = axis.exogenous if edge.is_exogenous else axis.targets
    return AnchorKey(
        scale=units[edge.src_scale - 1],
        lag=edge.lag,
        source=names[edge.src_label],
        target=axis.targets[dst_label],
    )


@dataclass(frozen=True)
class AnchorScore:
    """One sign's tally, with the denominator it was taken against."""

    sign: str
    denominator: int
    hits: int
    structural: int
    missed: Tuple[AnchorKey, ...]

    @property
    def rate(self) -> float:
        return self.hits / self.denominator if self.denominator else 0.0

    def __str__(self) -> str:
        return (
            f"{self.sign}: {self.hits}/{self.denominator} ({self.rate:.3f}), "
            f"{self.structural} of the declared set structural"
        )


def score(accepted: Iterable[AnchorKey], anchors: AnchorSet) -> Dict[str, AnchorScore]:
    """
    Score an accepted set against the anchors, one tally per sign.

    A structural anchor is one the axis cannot express, so it is reported and
    left out of the denominator; the rate a run publishes is over the anchors a
    search could actually have stated.
    """
    seen = set(accepted)
    out: Dict[str, AnchorScore] = {}
    for sign in (POSITIVE, NEGATIVE):
        declared = anchors.of_sign(sign)
        assertable = [a for a in declared if a.assertable]
        # A negative is hit when the accepted set states it, a positive when it
        # does; the two tallies read opposite ways and the caller knows which.
        hits = [a for a in assertable if a.key in seen]
        out[sign] = AnchorScore(
            sign=sign,
            denominator=len(assertable),
            hits=len(hits),
            structural=len(declared) - len(assertable),
            missed=tuple(sorted(a.key for a in assertable if a.key not in seen)),
        )
    return out
