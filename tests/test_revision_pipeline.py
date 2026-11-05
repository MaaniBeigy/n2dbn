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
from src.nested.data import audit, check_round_trip, from_finest, read_log, write_log
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
    result = simulations(tmp_path, records, fit.model, 2, 51, forms=False)
    assert result["round_trip_exact"]
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


def test_lifesnaps_reader_audits_collapsed_annotation_collisions(tmp_path):
    import csv
    from datetime import datetime, timedelta

    from src.data.label_axis import COLLAPSED_LABEL
    from src.nested.data import lifesnaps

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
    # Use an actual member of the declared collapse, not an invented label.
    from src.data.label_axis import COLLAPSE_GROUP

    text = path.read_text().replace("ema_completed", COLLAPSE_GROUP[1])
    path.write_text(text)
    records, preparation, schedule = lifesnaps(path, participants=1, weeks=8)
    assert schedule.weeks == 8
    assert not preparation["valid"]
    assert preparation["collision_cells"][COLLAPSED_LABEL] == 56
    assert len(preparation["collision_examples"][0]["original_annotations"]) == 2
    assert len(records) == 1


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
