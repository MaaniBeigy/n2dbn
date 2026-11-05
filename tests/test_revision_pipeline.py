"""Focused integration checks; these do not run the full experiment grids."""

import copy
import json

import numpy as np
import pytest

from experiments.revision import (
    Run,
    fitting,
    generated,
    parser,
    simulations,
    stability_summary,
)
from src.nested.data import (
    LIFESNAPS_OBSERVATION,
    audit,
    check_round_trip,
    from_finest,
    group_coincident_annotations,
    read_log,
    write_log,
)
from src.nested.evaluate import (
    aggregate_forecast,
    forecast_metrics,
    nonstationarity,
    reconcile,
    summarize_forecasts,
)
from src.nested.fit import FitConfig, Fitter
from src.nested.forecast import predict
from src.nested.law import feature
from src.nested.model import Parent, Record, initialize
from src.nested.report import paired_contrast, report


def args(tmp_path):
    return parser().parse_args(
        [
            "smoke",
            "--output",
            str(tmp_path),
            "--outer",
            "1",
            "--inner",
            "1",
            "--profile",
            "1",
            "--shifts",
            "0",
        ]
    )


def test_round_trip_keeps_empty_subject_and_unclipped_marks(tmp_path):
    x, y = np.zeros((24, 2), dtype=int), np.zeros((24, 2), dtype=int)
    x[23, 0], y[23, 0] = 1, 3000
    a = from_finest("a", x, y, [3, 2])
    b = from_finest("b", x * 0, y * 0, [3, 2])
    filename = tmp_path / "events.csv"
    write_log(filename, [a, b])
    returned = read_log(filename, [a, b])
    assert check_round_trip([a, b], returned)
    assert returned[0].data.Y[0][23, 0] == 3000
    assert [r.name for r in returned] == ["a", "b"]


def test_audit_exposes_simultaneous_starts():
    x = np.zeros((24, 1), dtype=int)
    x[2] = 2
    rec = from_finest("a", x, x, [3, 2])
    result = audit([rec])
    assert not result["valid"]
    assert result["collision_cells"] == {"label0": 1}
    assert audit([rec], count_only=True)["valid"]


@pytest.mark.parametrize("audit_only", [True, False])
def test_lifesnaps_failure_shows_evidence_and_stops_before_fitting(
    tmp_path, monkeypatch, audit_only
):
    from experiments import revision

    x = np.zeros((24, 1), dtype=int)
    x[2] = 2
    rec = from_finest("a", x, x, [3, 2], labels=["survey_response"])
    preparation = audit([rec])
    preparation["collision_examples"] = [
        dict(
            participant="a",
            raw_day_index=0,
            start_min=2,
            modeled_label="survey_response",
            original_annotations=[
                dict(csv_line=2, raw_label="notification_read", duration_min="0"),
                dict(csv_line=3, raw_label="mood_survey", duration_min="0"),
            ],
        )
    ]
    monkeypatch.setattr(
        revision, "lifesnaps", lambda *args, **kwargs: ([rec], preparation, None)
    )

    def unexpected_fit(*args, **kwargs):
        pytest.fail("an invalid panel must stop before fitting")

    monkeypatch.setattr(revision, "selected_fit", unexpected_fit)
    options = args(tmp_path)
    options.experiment = "lifesnaps"
    options.audit_only = audit_only
    with pytest.raises(ValueError) as failure:
        revision.lifesnaps_run(options)
    message = str(failure.value)
    assert str(tmp_path / "audit.json") in message
    assert "'survey_response': 1" in message
    assert "X=2, Y=2" in message
    assert "notification_read" in message
    assert "mood_survey" in message
    assert json.loads((tmp_path / "audit.json").read_text()) == preparation
    assert not (tmp_path / "runs.json").exists()
    assert rec.data.X[0][2, 0] == 2


def test_indexed_history_equals_full_column():
    rec = generated(n=48, labels=2)[0][0]
    parent = Parent("behavior", 0, 3)
    indices = np.array([[0, 2, 4], [10, 11, 12]])
    for channel in (0, 1):
        np.testing.assert_array_equal(
            feature(rec, 0, parent, channel, indices=indices),
            feature(rec, 0, parent, channel)[indices],
        )


def test_count_forecast_is_coherent_and_ignores_test_values():
    x = np.tile(np.arange(1, 4), (48, 1))
    rec = from_finest("a", x, x * 0, [3, 2])
    model = initialize([rec], [1, 1, 1], [0, 0, 0], [1, 1, 1], count_only=True)
    model.levels[-1].targets[0].parents[Parent("behavior", 0, 1)] = np.array([0.3, 0])
    future = from_finest("a", x[:6], x[:6] * 0, [3, 2])
    first, diagnostic = predict(model, rec, future, return_diagnostics=True)
    changed = from_finest("a", x[:6] * 10000, x[:6] * 0, [3, 2])
    second = predict(model, rec, changed)
    for a, b, c in zip(
        first, second, aggregate_forecast(first[0], future.data.hierarchy)
    ):
        np.testing.assert_allclose(a, b)
        np.testing.assert_allclose(a, c)
    assert diagnostic["log_link_evaluations"] > 0


@pytest.mark.parametrize("nonnegative", [False, True])
def test_reconciliation_in_original_units(nonnegative):
    rec = from_finest(
        "a", np.zeros((6, 2), dtype=int), np.zeros((6, 2), dtype=int), [3, 2]
    )
    raw = [np.full((6, 2), -2.0), np.full((2, 2), 15.0), np.full((1, 2), 60.0)]
    fitted = reconcile(raw, rec.data.hierarchy, nonnegative)
    for a, b in zip(fitted, aggregate_forecast(fitted[0], rec.data.hierarchy)):
        np.testing.assert_allclose(a, b)
    if nonnegative:
        assert all((a >= 0).all() for a in fitted)
    assert not np.allclose(fitted[0], raw[0])


def test_forecast_summary_retains_failures_and_pending_rows():
    good = dict(
        method="a",
        status="complete",
        finite=True,
        mae=1.0,
        mse=2.0,
        coherence=0.0,
        amplitude=1.0,
    )
    result = summarize_forecasts(
        [
            good,
            dict(method="a", status="failed"),
            dict(key="second/a", status="pending"),
        ]
    )
    assert result["a"]["runs"] == 3
    assert result["a"]["failed_runs"] == 2
    assert result["a"]["execution_failures"] == 1
    assert result["a"]["pending_or_running"] == 1
    assert result["a"]["mae"]["mean"] == 1


def test_hourly_scoring_does_not_accidentally_score_days():
    x = np.arange(1, 25).reshape(-1, 1)
    rec = from_finest("a", x, x * 0, [3, 2])
    actual = rec.data.X
    predicted = [a.astype(float) for a in actual]
    predicted[0] = predicted[0] + 4
    hour = forecast_metrics(
        actual, predicted, actual, rec.data.hierarchy, evaluation_scale=0
    )
    day = forecast_metrics(
        actual, predicted, actual, rec.data.hierarchy, evaluation_scale=1
    )
    assert hour["mae"] > 0
    assert day["mae"] == 0


def test_run_manifest_resumes_and_rejects_changed_config(tmp_path):
    options = args(tmp_path)
    run = Run(options)
    run.plan(["first", "unstarted"])
    row = run.cell("first", lambda: dict(value=3))
    assert row["status"] == "complete"
    resumed = Run(options)
    assert resumed.cell("first", lambda: pytest.fail("must not rerun"))["value"] == 3
    options.outer = 2
    with pytest.raises(ValueError, match="different code"):
        Run(options)
    assert json.loads((tmp_path / "runs.json").read_text())[1]["status"] == "pending"


def test_report_does_not_turn_missing_data_into_support(tmp_path):
    result = report(tmp_path)
    assert all(c["status"] == "inconclusive" for c in result["claims"])
    assert (tmp_path / "REPORT.md").exists()


def test_pairing_preserves_scenarios_and_replications():
    rows = [
        dict(
            status="complete",
            scenario=s,
            replicate=r,
            method=method,
            metrics=dict(shd=value),
        )
        for s in range(4)
        for r in range(3)
        for method, value in [("a", 5 + r), ("b", 3 + r)]
    ]
    result = paired_contrast(rows, "a", "b", bootstrap=50)
    assert result["pairs"] == 12
    assert result["ci95"] == [2, 2]


def test_stability_requires_every_half_fit():
    edge = dict(scale=2, target="a", kind="behavior", source="b", lag=1)
    rows = [dict(kind="half", status="complete", fit=dict(graph=[edge]))]
    summary = stability_summary(rows, [edge], 1)
    assert summary["edges"][0]["selection_frequency"] == 1
    assert not summary["edges"][0]["stable"]
    rows.append(copy.deepcopy(rows[0]))
    assert stability_summary(rows, [edge], 1)["edges"][0]["stable"]


def test_small_cohort_fit_and_simulation_interface(tmp_path):
    options = args(tmp_path)
    records, _ = generated(n=48, labels=2, participants=2)
    fit = fitting(
        records,
        options,
        latent=False,
        config=FitConfig(outer=1, inner=1, profile=1, shifts=0, structure=False),
    )
    result = simulations(
        tmp_path,
        records,
        fit.model,
        2,
        51,
        forms=False,
        observation_definition=LIFESNAPS_OBSERVATION,
    )
    assert result["round_trip_exact"]
    saved = json.loads((tmp_path / "fidelity.json").read_text())
    assert saved["observation_definition"] == LIFESNAPS_OBSERVATION
    assert set(fit.model.offsets) == {"p0", "p1"}
    assert np.allclose(sum(fit.model.offsets.values()), 0)


def test_nonstationarity_keeps_descriptive_and_control_rows():
    records, _ = generated(n=96, labels=2, participants=3)
    result = nonstationarity(records, [records, copy.deepcopy(records)])
    assert result["cells"]
    assert all(
        c["status"] in ("equivalent", "not_equivalent", "descriptive")
        for c in result["cells"]
    )
    assert any(c.get("replicate_control_equivalent") is True for c in result["cells"])


def test_outer_rollback_keeps_parameters_and_graph(monkeypatch):
    from src.nested import fit as module

    records, _ = generated(n=48, labels=2)
    model = initialize(records, [1, 1, 1], [0, 0, 0], [1, 1, 1])
    values = iter([10.0, 9.0])
    monkeypatch.setattr(module, "monitor", lambda *a: next(values))
    result = Fitter(
        FitConfig(outer=1, inner=1, profile=1, shifts=0, structure=False)
    ).fit(records, model)
    assert result.trace[-1]["accepted"] is False
    assert result.model.edges() == model.edges()
    for old, new in zip(model.levels[-1].targets, result.model.levels[-1].targets):
        np.testing.assert_array_equal(old.intercept, new.intercept)


@pytest.mark.parametrize("method", ["DLinear", "PatchTST"])
def test_tiny_deep_baseline_smoke(method):
    torch = pytest.importorskip("torch")
    from experiments.revision import deep_predictions

    torch.set_num_threads(1)
    train = [np.arange(32, dtype=float).reshape(-1, 1)]
    actual = [np.zeros((2, 1))]
    a = deep_predictions(train, actual, method, epochs=1, seed=8)
    b = deep_predictions(train, actual, method, epochs=1, seed=8)
    assert a[0].shape == (2, 1)
    assert np.isfinite(a[0]).all()
    np.testing.assert_array_equal(a[0], b[0])


def test_synthetic_search_and_sorting_controls(tmp_path):
    from experiments.revision import classical

    pytest.importorskip("sklearn")
    records, _ = generated(n=192, labels=2)
    for method in (
        "empty",
        "granger",
        "var-sortnregress",
        "r2-sortnregress",
        "random-sortnregress",
    ):
        graph = classical(records, method, lag=1, degree=1, seed=12)
        assert all(
            d == 1
            for targets in graph.values()
            for parents in targets.values()
            for _, d in parents
        )
    model = initialize(records, [1, 1, 1], [0, 0, 0], [1, 1, 1])
    fit = Fitter(FitConfig(outer=1, inner=1, profile=1, shifts=2, indegree=1)).fit(
        records, model
    )
    assert fit.candidate_refits > 6
    assert all(
        len(target.parents) <= 1
        for level in fit.model.levels
        for target in level.targets
    )


def test_contextual_parents_allow_positive_lags():
    records, _ = generated(n=48, labels=2)
    for rec in records:
        rec.data.C = [np.arange(len(x))[:, None] for x in rec.data.X]
    model = initialize(records, [1, 1, 1], [0, 0, 0], [2, 2, 1])
    candidates = Fitter(FitConfig(context_mode="all")).candidates(records, model, 0)
    assert {p.lag for p in candidates if p.kind == "context"} == {0, 1, 2}


def test_initial_factor_context_matches_generative_dynamics():
    from src.nested.inference import gaussian_smoother

    records, _ = generated(n=48, labels=2)
    for rec in records:
        rec.data.C = [np.ones((len(x), 1)) for x in rec.data.X]
    model = initialize(records, [1, 1, 1], [0, 0, 1], [1, 1, 1])
    parameters = model.levels[-1].factor
    parameters.mu0[:] = 2
    parameters.G[:] = 3
    mean, _, _ = gaussian_smoother(np.ones((1, 1)), parameters, np.ones((1, 1)))
    assert mean[0, 0] == pytest.approx(5)


def test_log_link_is_not_silently_clipped():
    from src.nested.law import exp_mean

    assert exp_mean(40.0) == pytest.approx(np.exp(40.0))
    with pytest.raises(FloatingPointError):
        exp_mean(1000.0)


@pytest.fixture
def lifesnaps_csv(tmp_path):
    import csv
    from datetime import datetime, timedelta

    path = tmp_path / "input.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "person_id",
                "event",
                "day_index",
                "start_min",
                "duration_min",
                "start_datetime",
            ],
        )
        writer.writeheader()
        for day in range(56):
            when = datetime(2026, 2, 16) + timedelta(days=day, minutes=60)
            for event in ("notification_read", "ema_completed", "walk"):
                writer.writerow(
                    dict(
                        person_id="a",
                        event=event,
                        day_index=day,
                        start_min=60,
                        duration_min=1,
                        start_datetime=when.strftime("%m/%d/%Y %H:%M"),
                    )
                )
        # Rare collisions occur after the first twenty survey collisions.
        # The diagnostic must retain evidence for every affected label.
        for minute, event, duration in (
            (61, "workout", 32),
            (61, "weights", 32),
            (62, "mindfulness", 2),
            (62, "mindfulness", 2),
            (63, "mindfulness", 2),
            (63, "mindfulness", 2),
            (64, "notification_received", 0),
            (64, "notification_received", 0),
            (65, "notification_read", 0),
            (65, "mood_survey", 0),
            (65, "notification_read", 0),
            (65, "step_goal_survey", 0),
            (65, "walk", 3000),
            (66, "mood_survey", 0),
        ):
            when = datetime(2026, 2, 16) + timedelta(days=55, minutes=minute)
            writer.writerow(
                dict(
                    person_id="a",
                    event=event,
                    day_index=55,
                    start_min=minute,
                    duration_min=duration,
                    start_datetime=when.strftime("%m/%d/%Y %H:%M"),
                )
            )
        # A retained participant's annotation outside the eight-week window.
        when = datetime(2026, 2, 16) + timedelta(days=56, minutes=60)
        writer.writerow(
            dict(
                person_id="a",
                event="walk",
                day_index=56,
                start_min=60,
                duration_min=1,
                start_datetime=when.strftime("%m/%d/%Y %H:%M"),
            )
        )
        # A short record excluded by the cohort filter.
        writer.writerow(
            dict(
                person_id="b",
                event="walk",
                day_index=0,
                start_min=60,
                duration_min=1,
                start_datetime="02/16/2026 01:00",
            )
        )
        # A retained participant whose last panel week is incomplete.
        for day in range(63):
            if day == 55:
                continue
            when = datetime(2026, 2, 16) + timedelta(days=day, minutes=60)
            writer.writerow(
                dict(
                    person_id="c",
                    event="walk",
                    day_index=day,
                    start_min=60,
                    duration_min=1,
                    start_datetime=when.strftime("%m/%d/%Y %H:%M"),
                )
            )
    # Use an actual member of the declared collapse, not an invented label.
    from src.data.label_axis import COLLAPSE_GROUP

    text = path.read_text().replace("ema_completed", COLLAPSE_GROUP[1])
    path.write_text(text)
    return path


def test_lifesnaps_groups_preserve_annotations_marks_and_round_trip(
    tmp_path, lifesnaps_csv
):
    import csv

    from src.data.label_axis import COLLAPSED_LABEL
    from src.nested.data import lifesnaps

    source_bytes = lifesnaps_csv.read_bytes()
    ledger_path = tmp_path / "annotations.csv"
    records, preparation, schedule = lifesnaps(
        lifesnaps_csv, participants=2, weeks=8, annotation_path=ledger_path
    )
    assert schedule.weeks == 8
    assert preparation["valid"]
    assert not preparation["before_grouping"]["valid"]
    before = preparation["before_grouping"]
    assert before["collision_cells"][COLLAPSED_LABEL] == 57
    assert before["max_starts_per_slot"][COLLAPSED_LABEL] == 4
    assert all(n == 0 for n in preparation["collision_cells"].values())
    examples = preparation["collision_examples"]
    assert len(examples) == 23
    assert {row["modeled_label"] for row in examples[:3]} == {
        "gym",
        "mindfulness",
        COLLAPSED_LABEL,
    }
    assert all(len(row["original_annotations"]) == 2 for row in examples)
    assert before["collision_cells"]["gym"] == 1
    assert before["collision_cells"]["mindfulness"] == 2
    assert preparation["collision_examples_complete_by_label"] == {
        "gym": True,
        "mindfulness": True,
        COLLAPSED_LABEL: False,
        "walk": True,
    }
    assert examples[0]["observed_count"] == 2
    assert examples[0]["observed_duration"] == 64
    assert examples[0]["original_annotations"][0]["raw_label"] == "workout"
    assert [r.name for r in records] == ["a", "c"]
    assert preparation["observation_definition"] == LIFESNAPS_OBSERVATION
    volumes = preparation["grouping"]["by_label"][COLLAPSED_LABEL]
    assert volumes == dict(
        annotations_before=117,
        groups_after=58,
        count_reduction=59,
        coincident_groups=57,
        duration_before=117,
        duration_after=117,
    )
    own = records[0].data
    t = 55 * 1440
    survey, walk = own.label_names.index(COLLAPSED_LABEL), own.label_names.index("walk")
    assert (own.X[0][t + 65, survey], own.Y[0][t + 65, survey]) == (1, 4)
    assert (own.X[0][t + 66, survey], own.Y[0][t + 66, survey]) == (1, 1)
    assert (own.X[0][t + 65, walk], own.Y[0][t + 65, walk]) == (1, 3000)
    assert own.C[0][t + 64, -1] == 2  # scheduled context remains a count
    assert not records[1].data.get_observed(1)[49 * 1440 :].any()
    with lifesnaps_csv.open() as handle:
        source = list(csv.DictReader(handle))
    with ledger_path.open() as handle:
        ledger = list(csv.DictReader(handle))
    assert len(ledger) == len(source) == preparation["annotation_audit"]["rows"]
    assert {r["status"] for r in ledger} == {
        "modeled",
        "excluded_participant",
        "outside_panel_window",
        "context",
        "unobserved_week",
    }
    for line, (raw, saved) in enumerate(zip(source, ledger), start=2):
        assert saved["csv_line"] == str(line)
        assert all(saved["raw_" + key] == value for key, value in raw.items())
    four = [r for r in ledger if r.get("group_annotation_count") == "4"]
    assert len(four) == 4
    assert len({r["group_id"] for r in four}) == 1
    assert {r["group_duration"] for r in four} == {"4"}
    assert lifesnaps_csv.read_bytes() == source_bytes
    exported = tmp_path / "grouped.csv"
    write_log(exported, records, schedule)
    assert check_round_trip(records, read_log(exported, records, schedule))


def test_grouping_reaggregates_counts_and_does_not_hide_invalid_durations():
    x = np.zeros((24, 2), dtype=int)
    x[2] = [2, 1]
    x[3] = [4, 0]
    y = 3 * x
    mask = np.ones(24, dtype=bool)
    mask[-6:] = False
    original = from_finest("a", x, y, [3, 2], observed=mask)
    other = from_finest("b", x, y, [3, 2], observed=mask)
    grouped = group_coincident_annotations([original, other])
    assert [rec.name for rec in grouped] == ["a", "b"]
    assert original.data.X[0][2, 0] == 2
    assert grouped[0].data.X[0][:, 0].sum() == 2
    assert grouped[0].data.X[-1][0, 0] == 2
    for rec in grouped:
        rec.validate()
        np.testing.assert_array_equal(rec.data.observed, mask)
        for k in range(rec.data.K):
            np.testing.assert_array_equal(rec.data.Y[k], original.data.Y[k])
            np.testing.assert_array_equal(rec.data.C[k], original.data.C[k])
    invalid = from_finest("bad", x, (x > 0).astype(int), [3, 2])
    with pytest.raises(ValueError, match="cannot repair invalid annotation durations"):
        group_coincident_annotations([invalid])


def test_lifesnaps_audit_cli_records_grouping_without_fitting(
    tmp_path, lifesnaps_csv, monkeypatch, capsys
):
    from experiments import revision

    options = parser().parse_args(
        [
            "lifesnaps",
            "--audit-only",
            "--participants",
            "2",
            "--csv",
            str(lifesnaps_csv),
            "--output",
            str(tmp_path / "audit"),
        ]
    )
    monkeypatch.setattr(
        revision, "selected_fit", lambda *a, **kw: pytest.fail("audit must not fit")
    )
    revision.lifesnaps_run(options)
    folder = tmp_path / "audit"
    result = json.loads((folder / "audit.json").read_text())
    assert result["valid"]
    assert (folder / "annotations.csv").is_file()
    assert not (folder / "runs.json").exists()
    assert (
        json.loads((folder / "config.json").read_text())["observation_definition"]
        == LIFESNAPS_OBSERVATION
    )
    assert "117 annotations -> 58 groups" in capsys.readouterr().out


def test_lagged_context_can_vary_inside_a_raw_constant_block():
    records, _ = generated(n=192, labels=2)
    rec = records[0]
    rec.data.C = [
        np.repeat(np.arange(32), 6)[:, None],
        np.repeat(np.arange(8), 4)[:, None],
        np.arange(8)[:, None],
    ]
    model = initialize(records, [1, 1, 1], [0, 0, 0], [2, 2, 1])
    offered = Fitter(FitConfig(context_mode="all")).candidates(records, model, 0)
    assert Parent("context", 0, 0) not in offered
    assert Parent("context", 0, 1) in offered


def test_report_tables_keep_completed_and_failed_denominators(tmp_path):
    from src.nested import FORMULATION
    from src.nested.evaluate import write_json

    folder = tmp_path / "forecast"
    write_json(
        folder / "config.json",
        dict(model_formulation=FORMULATION, source_digest="test-fixture"),
    )
    write_json(
        folder / "runs.json",
        [
            dict(
                key="1/nested",
                method="nested",
                status="complete",
                finite=True,
                mae=1.0,
                mse=2.0,
                coherence=0.0,
                amplitude=1.0,
            ),
            dict(
                key="2/nested", method="nested", status="failed", reason="test failure"
            ),
        ],
    )
    result = report(tmp_path)
    assert result["experiments"]["forecast"]["methods"]["nested"]["failed_runs"] == 1
    assert (tmp_path / "tables" / "forecast.csv").exists()
    assert "Failed/missing" in (tmp_path / "REPORT.md").read_text()


def test_lifesnaps_report_labels_groups_and_rejects_unknown_observation_units(tmp_path):
    from src.nested import FORMULATION
    from src.nested.evaluate import write_json

    folder = tmp_path / "lifesnaps"
    settings = dict(
        model_formulation=FORMULATION,
        source_digest="fixture",
        observation_definition=LIFESNAPS_OBSERVATION,
        participants=51,
        weeks=8,
        replicates=10,
        pairs=25,
    )
    write_json(folder / "config.json", settings)
    grouped = dict(
        annotations_before=117,
        groups_after=58,
        count_reduction=59,
        coincident_groups=57,
        duration_before=117,
        duration_after=117,
    )
    write_json(
        folder / "audit.json",
        dict(
            valid=True,
            participants=51,
            observation_definition=LIFESNAPS_OBSERVATION,
            grouping=dict(by_label={"survey_response": grouped}),
        ),
    )
    rows = [
        dict(key=key, status="complete", observation_definition=LIFESNAPS_OBSERVATION)
        for key in (
            ["full", "intercept", "calendar-all", "calendar-none", "no-regimes", "sham"]
            + [f"half-{pair}-{side}" for pair in range(25) for side in (0, 1)]
        )
    ]
    for row in rows[:2]:
        row["fidelity"] = dict(
            observation_definition=LIFESNAPS_OBSERVATION,
            volumes={
                channel: dict(
                    all_label_mean_absolute_log_ratio=(
                        0.1 if row["key"] == "full" else 0.8
                    ),
                    by_label={
                        "survey_response": dict(
                            observed=58,
                            simulated_mean=58,
                            ratio=1,
                            absolute_log_ratio=0,
                        )
                    },
                )
                for channel in ("counts", "booked_duration")
            },
        )
    write_json(folder / "runs.json", rows)
    result = report(tmp_path)
    count_claim = next(
        c for c in result["claims"] if "occupied-start-minute" in c["claim"]
    )
    assert count_claim["status"] == "supported"
    assert (tmp_path / "tables" / "lifesnaps-grouping.csv").exists()
    markdown = (tmp_path / "REPORT.md").read_text()
    assert "raw episode multiplicities" in markdown
    assert "annotations.csv" in markdown
    assert "| survey_response | 117 | 58 | 59 | 117 |" in markdown

    settings["observation_definition"] = "unknown-legacy-observations"
    write_json(folder / "config.json", settings)
    changed = report(tmp_path)
    assert all(
        c["status"] == "inconclusive"
        for c in changed["claims"]
        if c["claim"].startswith("LifeSnaps")
    )


def test_intercept_baseline_does_not_fit_depletion_predictors():
    records, _ = generated(n=48, labels=2)
    model = initialize(records, [1, 1, 1], [0, 0, 0], [1, 1, 1])
    model.levels[0].targets[0].depletion[:] = 1
    fit = Fitter(
        FitConfig(
            outer=1,
            inner=1,
            profile=1,
            structure=False,
            context_mode="none",
            depletion=False,
        )
    ).fit(records, model)
    assert all(
        np.array_equal(target.depletion, [0, 0])
        for level in fit.model.levels
        for target in level.targets
    )


def test_evaluation_compaction_preserves_coarse_forms_and_exact_volume():
    from src.nested.data import evaluation_view
    from src.nested.evaluate import volume_totals, volumes

    x = (np.arange(48) % 3 == 0).astype(int)[:, None]
    seen = np.ones(48, dtype=bool)
    seen[2] = False
    record = from_finest("a", x, x * 5, [2, 3, 2], observed=seen)
    compact = evaluation_view([record])
    assert compact[0].data.K == 3
    for k in range(3):
        np.testing.assert_array_equal(compact[0].data.X[k], record.data.X[k + 1])
        np.testing.assert_array_equal(
            compact[0].data.get_observed(k + 1), record.data.get_observed(k + 2)
        )
    exact = volume_totals([record])
    full = volumes([record], [[record]])
    reduced = volumes([record], observed_totals=exact, simulated_totals=[exact])
    assert full["counts"]["total_ratio"] == reduced["counts"]["total_ratio"] == 1
    assert (
        full["booked_duration"]["total_ratio"]
        == reduced["booked_duration"]["total_ratio"]
        == 1
    )
