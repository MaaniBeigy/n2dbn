"""Build the observed LifeSnaps event log from the raw MongoDB collections.

The published dataset is a MongoDB dump of three collections.  `fitbit` holds
one document per recorded item, keyed by `type`; `sema` holds one document per
scheduled survey.  Six of those feed the episode log this project fits, and
each contributes a start, a duration and a label.

  fitbit, type `exercise`               one episode per logged workout
  fitbit, type `sleep`                  `mainSleep` splits `sleep` from `nap`
  fitbit, type `mindfulness_sessions`   one episode per session
  sema, `SCHEDULED_TS`                  `notification_received`, instantaneous
  sema, `STARTED_TS`                    `notification_read`, instantaneous
  sema, `STARTED_TS` to `COMPLETED_TS`  `mood_survey`, `step_goal_survey`

Three conventions are measured against the released log.  A start is
truncated to its minute.  A duration is the millisecond field
rounded to the nearest minute, so a session of 90 seconds is two minutes and
one of 89 seconds is one.  An end is the start plus that whole-minute duration,
which keeps `end - start` equal to `duration_min` on every row.

Two anchors differ from the field a reader would guess, and both follow the
`event_timestamp` the matching fetch module passes.  A mindfulness episode is
anchored at `end_date_time`.  An exercise episode is identified by
`lastModified`, and two sessions sharing one `lastModified` collapse to the
first, which is where the released log loses 678 of its 3414 exercise rows.
`COLLAPSE_EXERCISE_ON_LAST_MODIFIED` carries that rule so a run can drop it.
"""

from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Reproducing the released log keeps the collapse.  Setting this False keeps
# every logged session and returns 678 more exercise episodes.
COLLAPSE_EXERCISE_ON_LAST_MODIFIED = True

#: The two SEMA surveys and the episode label each completed one carries.
SURVEY_LABELS = {
    "Context and Mood Survey": "mood_survey",
    "Step Goal Survey": "step_goal_survey",
}

#: Columns of the written file, in order.
COLUMNS = (
    "person_id",
    "user_id",
    "event",
    "day_index",
    "start_min",
    "duration_min",
    "start_datetime",
    "end_datetime",
)

Episode = Tuple[str, str, datetime, int]


def label_of(activity_name: str) -> str:
    """Episode label for a Fitbit `activityName`."""
    return activity_name.lower().replace("/", "_").replace(" ", "_")


def _stamp(value: str) -> datetime:
    """Parse an ISO timestamp truncated to its minute."""
    return datetime.fromisoformat(str(value)[:19]).replace(second=0, microsecond=0)


def _minutes(milliseconds: float) -> int:
    """Whole minutes of a millisecond duration, rounded half away from zero."""
    return int(milliseconds / 60000.0 + 0.5)


def fetch_fitbit_episodes(collection) -> List[Episode]:
    """Exercise, sleep and mindfulness episodes from the `fitbit` collection."""
    # `sleep` and `exercise` carry the same session under more than one
    # document, so a session is taken once by the id it logs itself under.
    once: Dict[Tuple[str, str, str], Episode] = {}
    episodes: List[Episode] = []
    query = {"type": {"$in": ["exercise", "sleep", "mindfulness_sessions"]}}
    for item in collection.find(query):
        kind = item["type"]
        data = item["data"]
        user = str(item["id"])
        if kind == "exercise":
            episode = (
                user,
                label_of(data["activityName"]),
                _stamp(data["startTime"]),
                _minutes(data["duration"]),
            )
            key = (
                str(data["lastModified"])
                if COLLAPSE_EXERCISE_ON_LAST_MODIFIED
                else str(data["logId"])
            )
            once.setdefault((user, kind, key), episode)
        elif kind == "sleep":
            once.setdefault(
                (user, kind, str(data["logId"])),
                (
                    user,
                    "sleep" if data.get("mainSleep") else "nap",
                    _stamp(data["startTime"]),
                    _minutes(data["duration"]),
                ),
            )
        else:
            # The fetch module reads this type at `end_date_time`, and the
            # released log anchors the episode there.
            once.setdefault(
                (user, kind, str(data["session_id"])),
                (
                    user,
                    "mindfulness",
                    _stamp(data["end_date_time"]),
                    _minutes(data["duration"]),
                ),
            )
    return episodes + list(once.values())


def fetch_sema_episodes(collection) -> List[Episode]:
    """Notification and completed-survey episodes from the `sema` collection."""
    episodes: List[Episode] = []
    for item in collection.find({}):
        data = item.get("data") or {}
        user = str(item["user_id"])
        if data.get("SCHEDULED_TS"):
            episodes.append(
                (user, "notification_received", _stamp(data["SCHEDULED_TS"]), 0)
            )
        if data.get("STARTED_TS"):
            episodes.append((user, "notification_read", _stamp(data["STARTED_TS"]), 0))
        label = SURVEY_LABELS.get(data.get("SURVEY_NAME"))
        if label and data.get("COMPLETED_TS") and data.get("UPLOADED_TS"):
            # The completed survey is anchored at the upload, which trails the
            # start by a minute on one row in eight.
            start = _stamp(data["UPLOADED_TS"])
            end = _stamp(data["COMPLETED_TS"])
            episodes.append(
                (user, label, start, int((end - start).total_seconds() // 60))
            )
    return episodes


def assemble(
    episodes: Sequence[Episode], participants: Optional[Sequence[str]]
) -> List[dict]:
    """Episodes as written rows, ordered by participant and start."""
    if participants is not None:
        allowed = set(participants)
        episodes = [e for e in episodes if e[0] in allowed]

    # `person_id` numbers the participants in the order the roster gives them,
    # so a stored roster reproduces the identifiers of an earlier run.
    order = (
        list(participants)
        if participants is not None
        else sorted({e[0] for e in episodes})
    )
    person_of = {
        user: i for i, user in enumerate(order) if user in {e[0] for e in episodes}
    }

    first_day: Dict[str, datetime] = {}
    for user, _, start, _ in episodes:
        day = start.replace(hour=0, minute=0)
        if user not in first_day or day < first_day[user]:
            first_day[user] = day

    rows = []
    for user, label, start, duration in episodes:
        end = start + timedelta(minutes=duration)
        rows.append(
            {
                "person_id": person_of[user],
                "user_id": user,
                "event": label,
                "day_index": (start.replace(hour=0, minute=0) - first_day[user]).days,
                "start_min": start.hour * 60 + start.minute,
                "duration_min": duration,
                "start_datetime": start,
                "end_datetime": end,
            }
        )
    rows.sort(key=lambda r: (r["person_id"], r["start_datetime"]))
    return rows


def _format(value: datetime) -> str:
    """Timestamp in the m/d/Y H:M form the released file carries."""
    return f"{value.month}/{value.day}/{value.year} {value.hour}:{value.minute:02d}"


def write_event_log(rows: Iterable[dict], path: str) -> int:
    """Write rows to `path` and return how many were written."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    written = 0
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for row in rows:
            writer.writerow(
                [
                    row["person_id"],
                    row["user_id"],
                    row["event"],
                    row["day_index"],
                    row["start_min"],
                    row["duration_min"],
                    _format(row["start_datetime"]),
                    _format(row["end_datetime"]),
                ]
            )
            written += 1
    return written


def build(mongo_uri: str, database: str, roster: Optional[Sequence[str]]) -> List[dict]:
    """Every episode of the log, read from MongoDB and assembled."""
    from pymongo import MongoClient

    client = MongoClient(mongo_uri)
    db = client[database]
    episodes = fetch_fitbit_episodes(db["fitbit"]) + fetch_sema_episodes(db["sema"])
    client.close()
    return assemble(episodes, roster)


def read_roster(path: Optional[str]) -> Optional[List[str]]:
    """Participant identifiers, one per line, in the order they are numbered."""
    if not path:
        return None
    with open(path) as handle:
        lines = (line.strip() for line in handle)
        return [line for line in lines if line and not line.startswith("#")]


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Write the observed LifeSnaps event log."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--mongo-uri", default=os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
    )
    parser.add_argument("--database", default="rais_anonymized")
    parser.add_argument("--roster", default="src/lifesnaps/roster.txt")
    parser.add_argument("--output", default="data/observed_lifesnaps_event_log.csv")
    parser.add_argument(
        "--keep-collapsed-exercise",
        action="store_true",
        help="keep every exercise session, dropping the lastModified collapse",
    )
    args = parser.parse_args(argv)

    global COLLAPSE_EXERCISE_ON_LAST_MODIFIED
    if args.keep_collapsed_exercise:
        COLLAPSE_EXERCISE_ON_LAST_MODIFIED = False

    roster = read_roster(args.roster if os.path.exists(args.roster) else None)
    rows = build(args.mongo_uri, args.database, roster)
    written = write_event_log(rows, args.output)
    participants = len({r["user_id"] for r in rows})
    labels = len({r["event"] for r in rows})
    print(
        f"{written} episodes, {participants} participants, {labels} labels -> {args.output}"
    )


if __name__ == "__main__":
    main()
