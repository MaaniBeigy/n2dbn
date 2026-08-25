"""
Tests for the reader that lays the two stored arm 1 runs out as table rows.

The module makes no measurement, so what is pinned here is that it reads each
field from the record the run actually writes, that a stratum is decided by the
edge's own endpoints, and that the two readings the manuscript depends on are
the ones it computes: a row carries evidence only where it passes its own
replicate noise, and the per-label volume is compared per replicate.
"""

from __future__ import annotations

import math

import pytest

from src.core.typed_graph import CALENDAR, OBSERVED, TypedEdge
from src.data.label_axis import LabelAxis
from src.evaluation.arm1_report import (
    CROSS_LABEL,
    EMPTY,
    EXOGENOUS,
    HOME,
    SCALE_NAMES,
    SELF,
    ablation_rows,
    changepoint_rows,
    clip_rows,
    control_rows,
    edge_agreement,
    edge_name,
    edges_by_scale,
    endpoints,
    evidence_bearing,
    heterogeneity_rows,
    is_self,
    leg_rows,
    receipts,
    rung_summary,
    scales_reached,
    separation_summary,
    source_kind,
    volume_agreement,
    volume_rows,
)


def _fit():
    """A fit record shaped like the one `exp15` stores, small enough to read."""
    return {
        "provenance": {
            "targets": [
                "gym",
                "mindfulness",
                "nap",
                "sleep",
                "survey_response",
                "walk",
            ],
            "exogenous": ["hour_sin", "hour_index", "notification_received"],
        },
        "search": {
            "scales": [1, 2, 3, 4],
            "price": 21.3,
            "null_median": 0.51,
            "n_candidates": 180,
            "by_stratum": {"cross": 2, "self": 1},
        },
        "edges": [
            {
                "edge": "hour_sin -> gym (k=2, level, lag=0)",
                "scale": 2,
                "lag": 0,
                "coefficient": -0.98,
                "standardized": -0.70,
                "future_lag_p": 1e-9,
                "e_value": 3.17,
            },
            {
                "edge": "notification_received -> survey_response (k=2, level, lag=1)",
                "scale": 2,
                "lag": 1,
                "coefficient": 1.42,
                "standardized": 0.52,
                "future_lag_p": 0.0,
                "e_value": 2.58,
            },
            {
                "edge": "walk -> walk (k=4, duration, lag=3)",
                "scale": 4,
                "lag": 3,
                "coefficient": -0.08,
                "standardized": -0.28,
                "future_lag_p": 0.0,
                "e_value": 1.92,
            },
        ],
        "rungs": {
            "R1": {"hits": 0, "denominator": 6, "passed": True},
            "R2": {"hits": 1, "denominator": 362, "passed": False},
            "R3": {"hits": 2, "denominator": 6, "reachable": 3, "reachable_hits": 2},
            "R9": {
                "threshold": 0.75,
                "selected": ["walk -> walk (k=4, duration, lag=3)"],
                "frequency": {
                    "hour_sin -> gym (k=2, level, lag=0)": 0.5,
                    "walk -> walk (k=4, duration, lag=3)": 1.0,
                },
            },
        },
        "calendar_ablation": {
            "absent": {"n_candidates": 172, "n_accepted": 21, "objective": -62217.5},
            "home": {"n_candidates": 180, "n_accepted": 23, "objective": -59921.8},
            "every": {"n_candidates": 204, "n_accepted": 25, "objective": -59955.7},
        },
    }


def _panel(tested, failed, by_form):
    return {
        "n_cells": 228,
        "n_tested": tested,
        "descriptive_by_clause": {"precision": 127},
        "failed": failed,
        "tested_rows": ["A01", "A06", "A11", "B3"],
        "by_form": by_form,
        "passed": not failed,
    }


def _leg(edges, states):
    return {
        "arm": "x",
        "real_per_label": {"gym": 500, "nap": 200},
        "drawn_per_label": {"gym": 2500, "nap": 500},
        "drawn_per_label_per_replicate": {"gym": 500.0, "nap": 100.0},
        "eta_clips_per_label": {"gym": 0, "nap": 38233},
        "eta_clips_by_end": {
            "gym": {"start_low": 0, "start_high": 0, "mark_low": 0, "mark_high": 0},
            "nap": {
                "start_low": 38233,
                "start_high": 0,
                "mark_low": 0,
                "mark_high": 0,
            },
        },
        "edges": [],
        "n_edges": edges,
        "regime_states": states,
        "reconciliation_error": 0.0,
    }


def _trip():
    """A round-trip record shaped like the one `exp16` stores."""
    forms_home = {
        "A01": {"cells": 18, "tested": 9, "failed": 0, "clauses": {}},
        "A06": {"cells": 12, "tested": 3, "failed": 2, "clauses": {}},
        "A11": {"cells": 12, "tested": 7, "failed": 2, "clauses": {}},
        "B3": {"cells": 2, "tested": 1, "failed": 0, "clauses": {}},
    }
    forms_empty = dict(forms_home)
    forms_empty["B3"] = {"cells": 2, "tested": 1, "failed": 1, "clauses": {}}
    return {
        "n_replicates": 5,
        "legs": {HOME: _leg(23, 2), EMPTY: _leg(0, 0)},
        "panels": {
            HOME: _panel(20, ["A06", "A11"], forms_home),
            EMPTY: _panel(20, ["A06", "A11", "B3"], forms_empty),
        },
        "controls": {
            "no regime": {
                "passed": True,
                "broke": ["A06"],
                "missed": [],
                "spurious": [],
                "untested": ["A08"],
            }
        },
        "rows_failing_their_own_noise": ["A11", "B3"],
        "rows_separating_the_fit_from_the_empty_graph": ["B3"],
        "empty_graph_fails_the_panel": True,
        "changepoints": [
            {
                "form": "A08",
                "cost": "poisson",
                "series": "nap",
                "scale": "hour",
                "annotation_error": 0.12,
                "f1_margin": 0.90,
                "covering": 0.97,
                "hausdorff": 108.5,
                "jump_ks": 0.10,
                "segment_ks": 0.09,
            },
            {
                "form": "A08",
                "cost": "poisson",
                "series": "gym",
                "scale": "day",
                "annotation_error": 0.29,
                "f1_margin": 0.78,
                "covering": 0.92,
                "hausdorff": 12.1,
                "jump_ks": 0.22,
                "segment_ks": 0.22,
            },
        ],
        "heterogeneity": [
            {
                "series": "nap",
                "real_cqv": 0.3333,
                "real_lower": 0.1351,
                "real_upper": 0.3846,
                "drawn_cqv": 0.3333,
                "drawn_lower": 0.2549,
                "drawn_upper": 0.3846,
                "comparable": True,
                "separation": 0.0,
                "sparse_gap": 0.106,
            },
            {"series": "mindfulness", "refused": "every participant is pinned at 1.0"},
        ],
    }


AXIS = LabelAxis(
    targets=("gym", "survey_response"),
    exogenous=("hour_sin", "notification_received"),
    grouped={"gym": "gym", "mood_survey": "survey_response"},
)


# ------------------------------------------------------------------ the name


def test_an_exogenous_source_is_named_off_the_calendar_block():
    edge = TypedEdge(CALENDAR, 2, 0, "level", 0)
    assert edge_name(edge, 0, AXIS) == "hour_sin -> gym (k=2, level, lag=0)"


def test_an_observed_source_is_named_off_the_target_block():
    edge = TypedEdge(OBSERVED, 4, 0, "count", 3)
    assert edge_name(edge, 0, AXIS) == "gym -> gym (k=4, count, lag=3)"


def test_the_name_round_trips_through_the_reader_that_splits_it():
    edge = TypedEdge(CALENDAR, 2, 1, "level", 1)
    name = edge_name(edge, 1, AXIS)
    assert endpoints(name) == ("notification_received", "survey_response")
    assert not is_self(name)


# ------------------------------------------------------------------ endpoints


def test_the_endpoints_come_off_the_stored_string():
    assert endpoints("hour_sin -> gym (k=2, level, lag=0)") == ("hour_sin", "gym")


def test_a_self_edge_is_one_whose_endpoints_agree():
    assert is_self("walk -> walk (k=4, duration, lag=3)")
    assert not is_self("hour_sin -> gym (k=2, level, lag=0)")


@pytest.mark.parametrize("bad", ["gym (k=2)", "hour_sin -> gym", "", "a -> b"])
def test_a_string_that_is_not_an_edge_is_refused(bad):
    with pytest.raises(ValueError, match="not a stored edge"):
        endpoints(bad)


# ----------------------------------------------------------- edges and scales


def test_every_searched_scale_gets_a_row_even_where_it_bought_nothing():
    rows = edges_by_scale(_fit())
    assert [row["scale"] for row in rows] == [1, 2, 3, 4]
    assert rows[0]["total"] == 0
    assert rows[0]["name"] == SCALE_NAMES[1]


def test_the_strata_split_on_the_endpoints_and_sum_to_the_edge_count():
    rows = edges_by_scale(_fit())
    hour = next(row for row in rows if row["scale"] == 2)
    week = next(row for row in rows if row["scale"] == 4)
    assert (hour["cross"], hour["self"]) == (2, 0)
    assert (week["cross"], week["self"]) == (0, 1)
    assert sum(row["total"] for row in rows) == len(_fit()["edges"])


def test_the_price_stratum_hides_a_split_the_source_kind_makes():
    hour = next(row for row in edges_by_scale(_fit()) if row["scale"] == 2)
    # Both of the hour scale's cross edges read an exogenous column, so the
    # stratum reads 2 where the label-to-label count is 0.
    assert hour["cross"] == 2
    assert (hour[EXOGENOUS], hour[CROSS_LABEL], hour[SELF]) == (2, 0, 0)


def test_a_label_to_label_edge_is_counted_apart_from_an_exogenous_one():
    fit = _fit()
    fit["edges"].append(
        {
            "edge": "sleep -> walk (k=3, count, lag=1)",
            "scale": 3,
            "lag": 1,
            "coefficient": 0.2,
            "standardized": 0.1,
            "future_lag_p": 0.0,
            "e_value": 1.1,
        }
    )
    day = next(row for row in edges_by_scale(fit) if row["scale"] == 3)
    assert (day[CROSS_LABEL], day[EXOGENOUS], day["cross"]) == (1, 0, 1)


@pytest.mark.parametrize(
    "edge,kind",
    [
        ("walk -> walk (k=4, duration, lag=3)", SELF),
        ("hour_sin -> gym (k=2, level, lag=0)", EXOGENOUS),
        ("notification_received -> survey_response (k=2, level, lag=1)", EXOGENOUS),
        ("sleep -> walk (k=3, count, lag=1)", CROSS_LABEL),
    ],
)
def test_each_source_kind_is_read_off_the_edge_and_the_exogenous_block(edge, kind):
    assert source_kind(edge, _fit()["provenance"]["exogenous"]) == kind


def test_a_scale_outside_the_declared_ladder_still_gets_a_row():
    fit = _fit()
    fit["search"]["scales"] = [2]
    rows = edges_by_scale(fit)
    assert {row["scale"] for row in rows} == {2, 4}
    assert next(row for row in rows if row["scale"] == 4)["name"] == "week"


def test_a_scale_with_no_name_falls_back_to_its_number():
    fit = _fit()
    fit["search"]["scales"] = [9]
    assert next(row for row in edges_by_scale(fit) if row["scale"] == 9)["name"] == "9"


def test_only_the_scales_that_bought_an_edge_are_reached():
    assert list(scales_reached(_fit())) == [2, 4]


# -------------------------------------------------------------------- receipts


def test_a_receipt_carries_its_reselection_rate_and_sorts_on_it():
    rows = receipts(_fit())
    assert [row["reselected"] for row in rows] == [1.0, 0.5, 0.0]
    assert rows[0]["target"] == "walk" and rows[0]["self"]
    assert rows[0]["kind"] == SELF and rows[1]["kind"] == EXOGENOUS


def test_an_edge_absent_from_the_stability_record_reads_as_never_reselected():
    rows = receipts(_fit())
    survey = next(row for row in rows if row["target"] == "survey_response")
    assert survey["reselected"] == 0.0
    assert survey["source"] == "notification_received"
    assert survey["scale"] == 2 and survey["lag"] == 1


# ----------------------------------------------------------------- the rungs


def test_each_rung_carries_its_own_denominator():
    rows = {row["rung"]: row for row in rung_summary(_fit())}
    assert (rows["R1"]["hits"], rows["R1"]["denominator"]) == (0, 6)
    assert (rows["R2"]["hits"], rows["R2"]["denominator"]) == (1, 362)
    assert (rows["R3"]["hits"], rows["R3"]["denominator"]) == (2, 6)
    assert (rows["R3r"]["hits"], rows["R3r"]["denominator"]) == (2, 3)


def test_the_stability_rung_reads_against_the_edges_that_were_accepted():
    rows = {row["rung"]: row for row in rung_summary(_fit())}
    assert (rows["R9"]["hits"], rows["R9"]["denominator"]) == (1, 3)
    assert "0.75" in rows["R9"]["reads"]


def test_a_rung_with_no_stated_verdict_reports_none():
    rows = {row["rung"]: row for row in rung_summary(_fit())}
    assert rows["R1"]["passed"] is True
    assert rows["R2"]["passed"] is False
    assert rows["R3"]["passed"] is None


# -------------------------------------------------------------- the ablation


def test_the_ablation_reads_every_arm_against_the_home_arm():
    rows = {row["arm"]: row for row in ablation_rows(_fit())}
    assert rows["home"]["against_home"] == 0.0
    assert rows["absent"]["against_home"] == pytest.approx(-2295.7, abs=0.1)
    assert rows["every"]["against_home"] == pytest.approx(-33.9, abs=0.1)


def test_the_ablation_keeps_the_arms_in_the_order_the_question_is_asked_in():
    assert [row["arm"] for row in ablation_rows(_fit())] == ["absent", "home", "every"]


# ---------------------------------------------------------------- the volumes


def test_the_volume_row_reads_the_per_replicate_draw_and_not_the_pooled_one():
    rows = {row["label"]: row for row in volume_rows(_trip())}
    assert rows["gym"]["drawn"] == 500.0 and rows["gym"]["ratio"] == 1.0
    assert rows["nap"]["ratio"] == pytest.approx(0.5)
    assert rows["nap"]["clips"] == 38233


def test_a_label_the_panel_never_carries_has_no_ratio():
    trip = _trip()
    trip["legs"][HOME]["real_per_label"]["yoga"] = 0
    trip["legs"][HOME]["drawn_per_label_per_replicate"]["yoga"] = 3.0
    row = next(r for r in volume_rows(trip) if r["label"] == "yoga")
    assert row["ratio"] != row["ratio"]


def test_a_leg_other_than_home_can_be_read():
    assert volume_rows(_trip(), leg=EMPTY)[0]["label"] == "gym"


# ------------------------------------------------------------------ the legs


def test_a_leg_row_counts_the_failing_cells_off_the_per_form_record():
    rows = {row["leg"]: row for row in leg_rows(_trip())}
    assert rows[HOME]["failed_cells"] == 4
    assert rows[HOME]["tested"] == 20
    assert rows[HOME]["edges"] == 23 and rows[HOME]["regime_states"] == 2
    assert rows[HOME]["reconciliation_error"] == 0.0
    assert rows[HOME]["passed"] is False


# ----------------------------------------------------- the evidence-bearing set


def test_a_row_failing_its_own_noise_is_removed_before_the_leg_is_graded():
    read = evidence_bearing(_trip())
    assert read["noise_rows"] == ["A11", "B3"]
    assert read["evidence_rows"] == ["A01", "A06"]
    assert read["failed_rows"] == ["A06"]
    assert read["passed_rows"] == ["A01"]


def test_the_tested_set_is_reported_beside_the_evidence_bearing_one():
    read = evidence_bearing(_trip())
    assert read["tested_rows"] == ["A01", "A06", "A11", "B3"]
    assert read["leg"] == HOME


def test_with_no_noisy_row_the_two_sets_agree():
    trip = _trip()
    trip["rows_failing_their_own_noise"] = []
    read = evidence_bearing(trip)
    assert read["evidence_rows"] == read["tested_rows"]


# -------------------------------------------------------------- the separation


def test_the_separation_reads_the_fitted_leg_against_the_empty_one():
    read = separation_summary(_trip())
    assert read["home_failed_cells"] == 4 and read["empty_failed_cells"] == 5
    assert read["home_tested_cells"] == read["empty_tested_cells"] == 20
    assert read["empty_fails_the_panel"] is True
    assert read["separating_rows"] == ["B3"]


# --------------------------------------------------------------- the controls


def test_a_control_reports_what_it_was_meant_to_break_and_did_not():
    rows = control_rows(_trip())
    assert rows[0]["control"] == "no regime" and rows[0]["passed"] is True
    assert rows[0]["broke"] == ["A06"] and rows[0]["untested"] == ["A08"]


# ------------------------------------------------ changepoints and dispersion


def test_the_changepoint_rows_are_read_at_one_scale_and_one_form_at_a_time():
    hour = changepoint_rows(_trip(), "hour")
    assert [row["series"] for row in hour] == ["nap"]
    assert hour[0]["f1"] == 0.90 and hour[0]["covering"] == 0.97
    assert hour[0]["form"] == "A08" and hour[0]["cost"] == "poisson"
    assert [row["series"] for row in changepoint_rows(_trip(), "day")] == ["gym"]


def test_a_second_form_at_the_same_scale_is_a_different_row():
    trip = _trip()
    trip["changepoints"].append(
        {
            "form": "A09",
            "cost": "variance",
            "series": "nap",
            "scale": "hour",
            "annotation_error": 4.16,
            "f1_margin": 0.12,
            "covering": 0.54,
            "hausdorff": 640.0,
            "jump_ks": 0.22,
            "segment_ks": 0.44,
        }
    )
    assert changepoint_rows(trip, "hour", "A08")[0]["f1"] == 0.90
    assert changepoint_rows(trip, "hour", "A09")[0]["f1"] == 0.12


def test_a_refused_dispersion_label_carries_its_reason_and_no_value():
    rows = {row["series"]: row for row in heterogeneity_rows(_trip())}
    assert "pinned" in rows["mindfulness"]["refused"]
    assert "real" not in rows["mindfulness"]
    assert rows["nap"]["comparable"] is True and rows["nap"]["separation"] == 0.0


# ------------------------------------------------------- the two runs agreeing


def _named_trip(names):
    trip = _trip()
    trip["legs"][HOME]["edges"] = [{"target": "gym", "name": name} for name in names]
    return trip


def test_the_two_runs_are_reported_as_agreeing_edge_for_edge():
    names = [record["edge"] for record in _fit()["edges"]]
    read = edge_agreement(_fit(), _named_trip(names))
    assert read["searched"] == read["drawn"] == read["agreeing"] == 3
    assert read["only_searched"] == [] and read["only_drawn"] == []


def test_an_edge_only_one_run_found_is_named_on_its_own_side():
    names = [record["edge"] for record in _fit()["edges"][:2]]
    read = edge_agreement(
        _fit(), _named_trip(names + ["nap -> nap (k=2, count, lag=2)"])
    )
    assert read["agreeing"] == 2
    assert read["only_searched"] == ["walk -> walk (k=4, duration, lag=3)"]
    assert read["only_drawn"] == ["nap -> nap (k=2, count, lag=2)"]


def test_a_leg_recorded_before_the_readable_name_reports_what_it_could_read():
    read = edge_agreement(_fit(), _trip())
    assert read["drawn"] == 0 and read["agreeing"] == 0
    assert len(read["only_searched"]) == 3


# ------------------------------------------------------------------ the clips


def test_a_clip_row_names_the_end_the_predictor_left_the_range_at():
    rows = {row["label"]: row for row in clip_rows(_trip())}
    assert rows["nap"]["total"] == 38233
    assert rows["nap"]["by_end"]["start_low"] == 38233
    assert rows["nap"]["by_end"]["start_high"] == 0
    assert rows["gym"]["total"] == 0


def test_a_leg_recorded_before_the_split_reports_its_total_and_no_breakdown():
    trip = _trip()
    del trip["legs"][HOME]["eta_clips_by_end"]
    rows = {row["label"]: row for row in clip_rows(trip)}
    assert rows["nap"]["total"] == 38233 and rows["nap"]["by_end"] == {}


# --------------------------------------------------------- the volume measure


def test_the_volume_measure_is_symmetric_in_the_direction_of_the_miss():
    trip = _trip()
    # gym is drawn at exactly its rate; nap at half of it.
    read = volume_agreement(trip)
    assert read["mean_abs_log_ratio"] == pytest.approx(abs(math.log(0.5)) / 2)
    assert read["worst"] == "nap"
    assert read["labels"] == ["gym", "nap"]


def test_a_leg_drawing_twice_and_a_leg_drawing_half_are_equally_far():
    trip = _trip()
    trip["legs"][EMPTY]["drawn_per_label_per_replicate"]["nap"] = 400.0
    half = volume_agreement(trip, HOME, ["nap"])["mean_abs_log_ratio"]
    twice = volume_agreement(trip, EMPTY, ["nap"])["mean_abs_log_ratio"]
    assert half == pytest.approx(twice)


def test_the_measure_can_be_read_over_a_named_subset():
    read = volume_agreement(_trip(), labels=["gym"])
    assert read["mean_abs_log_ratio"] == 0.0 and read["labels"] == ["gym"]


def test_a_label_the_panel_never_carries_is_refused():
    trip = _trip()
    trip["legs"][HOME]["real_per_label"]["yoga"] = 0
    trip["legs"][HOME]["drawn_per_label_per_replicate"]["yoga"] = 3.0
    with pytest.raises(ValueError, match="carries no 'yoga'"):
        volume_agreement(trip)
