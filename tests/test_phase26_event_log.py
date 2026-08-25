"""
Reading an observed event log back into multiscale arrays.

The acceptance property of the reader is the round trip: the event log of a
generated dataset, fed back through `build_multiscale_data` with the same slot
length and branching, must rebuild the arrays the generator produced.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from src.core.calendar import SlotCalendar, write_event_log
from src.data.event_log import ObservedEvent, build_multiscale_data, read_event_log
from src.synthetic.families import make_generator
from src.synthetic.scenarios import ScenarioConfig

ORIGIN = datetime(2026, 1, 1, 0, 0)


def _config(**kw) -> ScenarioConfig:
    base = dict(
        name="event-log-round-trip",
        K=2,
        H_finest=120,
        branching=[10],
        L=3,
        slot_minutes=1,
        finest_binary=True,
        zi_level="medium",
        finest_rate=0.02,
        seed=17,
    )
    base.update(kw)
    K = base["K"]
    base.update(M_k=[2] * K, r_k=[1] * K, max_lag=[1] * K, max_indegree=[1] * K)
    return ScenarioConfig(**base)


def _generate(cfg: ScenarioConfig):
    return make_generator(cfg, seed=cfg.seed).generate(seed=cfg.seed)


def _rebuild(ds, cfg):
    """Round trip the dataset through its own event log."""
    return build_multiscale_data(
        ds.event_log(),
        slot_minutes=cfg.slot_minutes,
        branching=cfg.branching,
        labels=ds.data.label_names,
        origin=ds.calendar.origin,
    )


def _events(spans, label="A", origin=ORIGIN):
    """Build events from (start minute, duration minute) pairs."""
    return [
        ObservedEvent(
            label=label,
            start=origin + timedelta(minutes=a),
            end=origin + timedelta(minutes=a + d),
        )
        for a, d in spans
    ]


# ---------------------------------------------------------------------------
# The acceptance check
# ---------------------------------------------------------------------------


def test_round_trip_rebuilds_the_generated_arrays_exactly():
    cfg = _config()
    ds = _generate(cfg)
    X1, Y1 = ds.data.get_X(1), ds.data.get_Y(1)

    # The reconstructor gives every start at least the slot it starts in, so a
    # zero-duration start is unrepresentable in a log; this scenario has none.
    assert int(Y1[X1 > 0].min()) >= 1
    assert int(X1.sum()) > 0

    data, calendar, labels = _rebuild(ds, cfg)

    assert labels == ds.data.label_names
    assert calendar.origin == ds.calendar.origin
    for k in range(1, ds.data.K + 1):
        assert np.array_equal(data.get_X(k), ds.data.get_X(k))
        assert np.array_equal(data.get_Y(k), ds.data.get_Y(k))


def test_round_trip_holds_across_three_scales():
    cfg = _config(K=3, H_finest=1440, branching=[10, 6], finest_rate=None, seed=7)
    ds = _generate(cfg)
    X1, Y1 = ds.data.get_X(1), ds.data.get_Y(1)

    # The scenario must exercise the lossy case, otherwise the maximum() below
    # would be doing nothing.
    assert int((Y1[X1 > 0] == 0).sum()) > 0
    assert int(X1.sum()) > 100

    data, _, _ = _rebuild(ds, cfg)

    # Two documented losses: a log holds no trailing empty slots, so the horizon
    # stops at the last coarsest epoch that carries an event, and a start booked
    # zero duration units comes back as the one slot it occupies.
    T = data.hierarchy.n_epochs(1)
    expected_X = X1[:T]
    expected_Y = np.maximum(Y1, X1)[:T]
    assert X1[T:].sum() == 0 and Y1[T:].sum() == 0

    # X loses nothing at any scale, so it is compared against the generator's
    # own coarse arrays rather than against a re-aggregation of the finest.
    assert T == X1.shape[0]
    for k in range(1, data.K + 1):
        assert np.array_equal(data.get_X(k), ds.data.get_X(k))
        assert np.array_equal(data.get_Y(k), _aggregate(expected_Y, data, k))

    # Y differs from the generated array precisely on the promoted starts.
    assert not np.array_equal(data.get_Y(1), Y1)
    assert np.array_equal(data.get_Y(1) - Y1, ((X1 > 0) & (Y1 == 0)).astype(Y1.dtype))


def _aggregate(fine: np.ndarray, data, k: int) -> np.ndarray:
    """Sum a finest-scale array up to scale k over the rebuilt hierarchy."""
    out = fine
    for j in range(1, k):
        shape = (data.hierarchy.n_epochs(j + 1), fine.shape[1])
        parent = np.zeros(shape, dtype=out.dtype)
        np.add.at(parent, data.hierarchy.parent_maps[j - 1], out)
        out = parent
    return out


def test_rebuilt_data_reconciles_exactly():
    cfg = _config()
    ds = _generate(cfg)
    data, _, _ = _rebuild(ds, cfg)
    # All-zero arrays would reconcile too, so pin the content first.
    assert data.get_X(1).sum() > 0 and data.get_Y(1).sum() > 0
    assert data.check_reconciliation() == {"RecErr_X": 0.0, "RecErr_Y": 0.0}


# ---------------------------------------------------------------------------
# CSV round trip
# ---------------------------------------------------------------------------


def test_csv_round_trip_preserves_the_log(tmp_path):
    cfg = _config()
    ds = _generate(cfg)
    records = ds.event_log()
    path = str(tmp_path / "log.csv")
    write_event_log(path, records, ds.calendar)

    events = read_event_log(path)

    assert [e.label for e in events] == [r.label for r in records]
    assert [e.start for e in events] == [r.start for r in records]
    assert [e.end for e in events] == [r.end for r in records]

    data, _, _ = build_multiscale_data(
        events,
        slot_minutes=cfg.slot_minutes,
        branching=cfg.branching,
        labels=ds.data.label_names,
        origin=ds.calendar.origin,
    )
    assert np.array_equal(data.get_X(1), ds.data.get_X(1))


def test_read_event_log_accepts_custom_column_names(tmp_path):
    path = tmp_path / "custom.csv"
    path.write_text(
        "who,from,to\nA,2026-01-01T00:00,2026-01-01T00:30\n", encoding="utf-8"
    )
    events = read_event_log(str(path), label_col="who", start_col="from", end_col="to")
    assert events == [
        ObservedEvent("A", datetime(2026, 1, 1, 0, 0), datetime(2026, 1, 1, 0, 30))
    ]


# ---------------------------------------------------------------------------
# Grid, padding and defaults
# ---------------------------------------------------------------------------


def test_horizon_pads_to_whole_coarsest_epochs():
    data, calendar, labels = build_multiscale_data(
        _events([(0, 5)]), slot_minutes=1, branching=[10, 6]
    )
    assert labels == ["A"]
    assert calendar.branching == (10, 6)
    assert [data.hierarchy.n_epochs(k) for k in (1, 2, 3)] == [60, 6, 1]
    assert data.get_X(1)[0, 0] == 1 and data.get_Y(1)[0, 0] == 5
    assert data.get_X(3)[0, 0] == 1 and data.get_Y(3)[0, 0] == 5
    assert data.get_C(1).shape == (60, 0)
    assert data.check_reconciliation() == {"RecErr_X": 0.0, "RecErr_Y": 0.0}


def test_default_labels_and_origin_come_from_the_events():
    start = datetime(2026, 3, 4, 9, 7)
    events = [
        ObservedEvent("nap", start + timedelta(minutes=30), start + timedelta(hours=1)),
        ObservedEvent("meal", start, start + timedelta(minutes=8)),
    ]
    data, calendar, labels = build_multiscale_data(
        events, slot_minutes=15, branching=[4]
    )
    assert labels == ["meal", "nap"]
    # 09:07 floors to the 15-minute grid of its own day.
    assert calendar.origin == datetime(2026, 3, 4, 9, 0)
    assert data.get_X(1)[0, 0] == 1  # the meal starts in the first slot
    assert data.get_Y(1)[0, 0] == 1  # 8 minutes occupy one 15-minute slot
    assert data.get_X(1)[2, 1] == 1  # the nap starts 30 minutes later
    assert data.get_Y(1)[2, 1] == 2  # and covers two slots


def test_zero_length_event_occupies_the_slot_it_starts_in():
    when = ORIGIN + timedelta(minutes=7)
    data, _, _ = build_multiscale_data(
        [ObservedEvent("A", when, when)], slot_minutes=1, branching=[4], origin=ORIGIN
    )
    assert data.hierarchy.n_epochs(1) == 8
    assert data.get_Y(1)[7, 0] == 1


def test_supplied_context_is_kept():
    # The horizon pads to 4 finest slots, so scale 2 holds 2 epochs.
    C = [np.arange(8, dtype=np.int64).reshape(4, 2), np.full((2, 2), 7)]
    data, _, _ = build_multiscale_data(
        _events([(0, 3)]), slot_minutes=1, branching=[2], origin=ORIGIN, context=C
    )
    assert np.array_equal(data.get_C(1), np.arange(8).reshape(4, 2))
    assert np.array_equal(data.get_C(2), np.full((2, 2), 7.0))
    assert data.get_C(1).dtype == np.float64
    assert data.get_C(2).dtype == np.float64


def test_context_of_the_wrong_height_is_rejected():
    # The caller cannot see the padded horizon, so a silent mismatch would put
    # the wrong context row against every epoch.
    with pytest.raises(ValueError, match=r"scale 1 has shape \(3, 1\).*\(4, d_1\)"):
        build_multiscale_data(
            _events([(0, 3)]),
            slot_minutes=1,
            branching=[2],
            origin=ORIGIN,
            context=[np.zeros((3, 1)), np.zeros((2, 1))],
        )
    with pytest.raises(ValueError, match=r"scale 2 has shape \(2,\)"):
        build_multiscale_data(
            _events([(0, 3)]),
            slot_minutes=1,
            branching=[2],
            origin=ORIGIN,
            context=[np.zeros((4, 1)), np.zeros(2)],
        )


def test_context_with_too_few_scales_is_rejected():
    with pytest.raises(ValueError, match="1 arrays but the hierarchy has 2 scales"):
        build_multiscale_data(
            _events([(0, 3)]),
            slot_minutes=1,
            branching=[2],
            origin=ORIGIN,
            context=[np.zeros((4, 1))],
        )


def test_duration_is_rounded_up_to_whole_slots_not_slots_overlapped():
    # 00:10 to 00:23 straddles the 15-minute boundary at 00:15, touching two
    # grid slots, but its duration is 13 minutes, so Y is one slot.  The start
    # sits two thirds into slot 0, which also pins the index to a floor.
    data, _, _ = build_multiscale_data(
        [
            ObservedEvent(
                "A", ORIGIN + timedelta(minutes=10), ORIGIN + timedelta(minutes=23)
            )
        ],
        slot_minutes=15,
        branching=[4],
        origin=ORIGIN,
    )
    assert list(data.get_Y(1)[:, 0]) == [1, 0, 0, 0]
    assert list(data.get_X(1)[:, 0]) == [1, 0, 0, 0]


def test_two_labels_may_share_one_slot_under_finest_binary():
    # finest_binary constrains each label separately; a per-slot total would
    # wrongly reject a slot holding one start of two different labels.
    data, _, labels = build_multiscale_data(
        _events([(0, 2)], label="A") + _events([(0, 2)], label="B"),
        slot_minutes=1,
        branching=[2],
        origin=ORIGIN,
    )
    assert labels == ["A", "B"]
    assert list(data.get_X(1)[0]) == [1, 1]
    assert list(data.get_Y(1)[0]) == [2, 2]


def test_supplied_labels_fix_the_column_order_and_keep_unobserved_columns():
    data, _, labels = build_multiscale_data(
        _events([(0, 1)], label="nap") + _events([(2, 1)], label="meal"),
        slot_minutes=1,
        branching=[2],
        origin=ORIGIN,
        labels=["nap", "meal", "walk"],
    )
    # The supplied order wins over the sorted order that the default would give.
    assert labels == ["nap", "meal", "walk"]
    assert list(data.get_X(1).sum(axis=0)) == [1, 1, 0]
    assert data.get_X(1)[0, 0] == 1 and data.get_X(1)[2, 1] == 1
    assert data.get_X(2).shape == (2, 3)


def test_repeated_starts_are_summed_when_binary_is_off():
    data, _, _ = build_multiscale_data(
        _events([(0, 2), (0, 3)]),
        slot_minutes=5,
        branching=[2],
        origin=ORIGIN,
        finest_binary=False,
    )
    assert data.get_X(1)[0, 0] == 2
    assert data.get_Y(1)[0, 0] == 2  # each event covers one 5-minute slot


# ---------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------


def test_missing_column_names_the_column(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("label,start_iso\nA,2026-01-01T00:00\n", encoding="utf-8")
    with pytest.raises(ValueError, match="end_iso"):
        read_event_log(str(path))


def test_end_before_start_names_the_row(tmp_path):
    path = tmp_path / "reversed.csv"
    path.write_text(
        "label,start_iso,end_iso\n"
        "A,2026-01-01T00:00,2026-01-01T00:30\n"
        "B,2026-01-01T02:00,2026-01-01T01:00\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="row 3"):
        read_event_log(str(path))


def test_a_file_with_no_header_names_the_missing_column(tmp_path):
    path = tmp_path / "blank.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match=r"column 'label' is missing.*has \[\]"):
        read_event_log(str(path))


def test_a_header_only_file_reads_as_no_events(tmp_path):
    path = tmp_path / "header.csv"
    path.write_text("label,start_iso,end_iso\n", encoding="utf-8")
    assert read_event_log(str(path)) == []


def test_an_unreadable_timestamp_names_the_row(tmp_path):
    path = tmp_path / "junk.csv"
    path.write_text(
        "label,start_iso,end_iso\n"
        "A,2026-01-01T00:00,2026-01-01T00:30\n"
        "B,yesterday,2026-01-01T01:00\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="row 3 .* unreadable timestamp"):
        read_event_log(str(path))


def test_a_missing_timestamp_cell_names_the_row(tmp_path):
    path = tmp_path / "short.csv"
    path.write_text("label,start_iso,end_iso\nA,2026-01-01T00:00\n", encoding="utf-8")
    with pytest.raises(ValueError, match="row 2 .* unreadable timestamp"):
        read_event_log(str(path))


def test_unknown_label_is_rejected():
    with pytest.raises(ValueError, match="'B'"):
        build_multiscale_data(
            _events([(0, 1)], label="B"), slot_minutes=1, branching=[2], labels=["A"]
        )


def test_two_starts_of_one_label_in_one_slot_are_rejected():
    clash = _events([(0, 2), (3, 2)])
    with pytest.raises(ValueError, match="2 starts in finest slot 0") as caught:
        build_multiscale_data(clash, slot_minutes=10, branching=[2], origin=ORIGIN)

    # The remedy has to be a shorter slot; a longer one keeps the two starts
    # together, so advising a coarser grid would send the caller backwards.
    assert "shorter slot_minutes" in str(caught.value)
    with pytest.raises(ValueError):
        build_multiscale_data(clash, slot_minutes=20, branching=[2], origin=ORIGIN)
    data, _, _ = build_multiscale_data(
        clash, slot_minutes=1, branching=[2], origin=ORIGIN
    )
    assert list(data.get_X(1)[:, 0]) == [1, 0, 0, 1, 0, 0]


def test_reversed_event_is_rejected_before_it_becomes_a_one_slot_episode():
    when = ORIGIN + timedelta(minutes=10)
    with pytest.raises(ValueError, match="ends at .* before its start"):
        build_multiscale_data(
            [ObservedEvent("A", when, when - timedelta(minutes=5))],
            slot_minutes=1,
            branching=[2],
            origin=ORIGIN,
        )


def test_empty_log_is_rejected():
    with pytest.raises(ValueError, match="empty event log"):
        build_multiscale_data([], slot_minutes=1, branching=[2])


def test_event_before_the_origin_is_rejected():
    with pytest.raises(ValueError, match="before the origin"):
        build_multiscale_data(
            _events([(0, 1)]),
            slot_minutes=1,
            branching=[2],
            origin=ORIGIN + timedelta(minutes=5),
        )


def test_calendar_matches_the_slot_grid():
    _, calendar, _ = build_multiscale_data(
        _events([(0, 1)]), slot_minutes=15, branching=[4], origin=ORIGIN
    )
    assert calendar == SlotCalendar.from_config(
        branching=[4], slot_minutes=15, origin=ORIGIN
    )
    assert calendar.slot == timedelta(minutes=15)
