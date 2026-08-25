"""
Tests for the grid run on a real panel and the synthetic panels drawn from it.

The comparison has to be able to fail, so what is pinned here is that each form
reads the structure it names, that the participant is the unit a difference is
paired on, that the inclusion rule partitions the grid on the real panel alone,
and that a shuffle removing one structure breaks the rows stated against it.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.cohort import Cohort
from src.core.scales import ScaleHierarchy
from src.core.variables import MultiScaleData
from src.data.label_axis import LabelAxis
from src.evaluation.nonstationarity import (
    DESCRIPTIVE,
    GRID,
    MARGINAL,
    TESTED,
    implemented_forms,
)
from src.evaluation.panel_compare import (
    COMPARED_SCALES,
    FORMS,
    INJECTION_SIZES,
    JOINT_LABEL,
    LABEL_SHUFFLE,
    POWER_BOUND,
    REGIME_FORMS,
    SCALE_NAMES,
    TIME_SHUFFLE,
    TOLERANCE_SHARE,
    WEEK,
    WEEK_FORMS,
    FormSpec,
    Series,
    admits,
    by_form,
    by_key,
    changepoint_penalty,
    compare,
    declared_effect,
    declared_rows,
    denominator,
    failed_rows,
    form_ids,
    form_values,
    inject_block,
    minimum_detectable_cell,
    panel_series,
    perturbed,
    rows_tested,
    shuffled,
    standard_error,
    summary,
    table_digest,
    tolerance_of,
    tolerance_table,
)
from src.evaluation.segmentation import POISSON, VARIANCE
from src.inference.bound import segment_sum

HOURS, DAYS, WEEKS = 24, 35, 5
T = DAYS * HOURS
LABELS = ["walk", "sleep", "gym"]
DAY_OF = np.repeat(np.arange(DAYS), HOURS).astype(np.int32)
WEEK_OF = np.repeat(np.arange(WEEKS), 7).astype(np.int32)
SUBJECTS = 22
SPEC = {spec.id: spec for spec in FORMS}

#: The test ladder is hour, day and week, which is the panel ladder without its
#: minute scale.
NAMES = {1: "hour", 2: "day", 3: "week"}

#: A shorter sweep than the declared one, so a test measures power in seconds.
SIZES = (0.1, 0.4, 1.6)


def hierarchy() -> ScaleHierarchy:
    return ScaleHierarchy(K=3, H=[T, DAYS, WEEKS], parent_maps=[DAY_OF, WEEK_OF, None])


def axis() -> LabelAxis:
    return LabelAxis(
        targets=tuple(LABELS),
        exogenous=("hour_index",),
        grouped={name: name for name in LABELS},
    )


def subject_counts(seed: int, drift: float = 0.0) -> np.ndarray:
    """One subject's hourly counts, with a daily rhythm and an optional drift."""
    rng = np.random.default_rng(seed)
    hour = np.arange(T) % HOURS
    shape = 1.0 + np.cos(2.0 * np.pi * hour / HOURS)
    ramp = 1.0 + drift * np.arange(T) / T
    out = np.zeros((T, len(LABELS)), dtype=np.int64)
    for j in range(len(LABELS)):
        out[:, j] = rng.poisson(0.3 * (j + 1) * shape * ramp)
    return out


def cohort_of(drift: float = 0.0, offset: int = 0, n: int = SUBJECTS) -> Cohort:
    names = tuple(f"p{i:02d}" for i in range(n))

    def reader(name: str) -> MultiScaleData:
        hourly = subject_counts(offset + names.index(name), drift)
        day = segment_sum(hourly.astype(float), DAY_OF, DAYS).astype(np.int64)
        week = segment_sum(day.astype(float), WEEK_OF, WEEKS).astype(np.int64)
        return MultiScaleData(
            hierarchy=hierarchy(),
            label_names=list(LABELS),
            X=[hourly, day, week],
            Y=[hourly, day, week],
            C=[np.zeros((T, 1)), np.zeros((DAYS, 1)), np.zeros((WEEKS, 1))],
        )

    return Cohort(
        subjects=names,
        hierarchy=hierarchy(),
        label_names=list(LABELS),
        reader=reader,
        masks=np.ones((n, T), dtype=bool),
    )


@pytest.fixture(scope="module")
def real():
    return panel_series(cohort_of(), axis(), (1, 2, 3), NAMES)


@pytest.fixture(scope="module")
def drawn():
    return [panel_series(cohort_of(offset=100), axis(), (1, 2, 3), NAMES)]


@pytest.fixture(scope="module")
def cells(real, drawn):
    return compare(real, drawn, sizes=SIZES)


# ---------------------------------------------------------------------------
# What a panel offers
# ---------------------------------------------------------------------------


def test_a_panel_offers_one_series_per_label_and_one_joint_block_per_scale(real):
    keys = by_key(real)
    assert len(real) == (len(LABELS) + 1) * 3
    for scale in NAMES.values():
        for label in LABELS + [JOINT_LABEL]:
            assert (label, scale) in keys
    assert keys[("walk", "day")].blocks[0].shape == (DAYS,)
    assert keys[(JOINT_LABEL, "day")].blocks[0].shape == (DAYS, len(LABELS))


def test_a_series_names_its_scale_and_the_period_that_scale_carries(real):
    hourly = by_key(real)[("walk", "hour")]
    assert hourly.period == 24.0
    assert hourly.n_epochs == T
    assert hourly.key == ("walk", "hour")
    assert by_key(real)[("walk", "day")].period == 7.0


def test_an_empty_series_reports_no_epochs():
    assert Series("walk", "day", False, ()).n_epochs == 0


def test_the_panel_ladder_names_its_three_compared_scales():
    assert COMPARED_SCALES == (2, 3, 4)
    assert [SCALE_NAMES[k] for k in COMPARED_SCALES] == ["hour", "day", "week"]
    assert 1 not in SCALE_NAMES


# ---------------------------------------------------------------------------
# Which form is read where
# ---------------------------------------------------------------------------


def test_a_marginal_form_never_reads_the_joint_block_and_the_reverse(real):
    keys = by_key(real)
    assert admits(SPEC["A01"], keys[("walk", "day")])
    assert not admits(SPEC["A01"], keys[(JOINT_LABEL, "day")])
    assert admits(SPEC["B3"], keys[(JOINT_LABEL, "day")])
    assert not admits(SPEC["B3"], keys[("walk", "day")])


def test_the_week_scale_admits_only_the_three_forms_eight_epochs_define(real):
    week = by_key(real)[("walk", WEEK)]
    assert WEEK_FORMS == ("A01", "A04", "A08")
    for spec in FORMS:
        if spec.joint:
            continue
        assert admits(spec, week) == (spec.id in WEEK_FORMS)


def test_the_regime_rows_are_read_from_the_day_scale_up():
    hour = Series("walk", "hour", False, (np.ones(1344),))
    day = Series("walk", "day", False, (np.ones(56),))
    for spec in FORMS:
        if spec.joint:
            continue
        assert admits(spec, hour) == (spec.id not in REGIME_FORMS)
        assert admits(spec, day)
    assert REGIME_FORMS == ("A14", "B5")


def test_the_rows_this_comparison_runs_are_the_ones_the_grid_marks_implemented():
    assert form_ids() == tuple(implemented_forms())
    assert len(form_ids()) == 21
    assert declared_rows() == ()
    assert not set(form_ids()) & set(declared_rows())
    # Every row's layer is the one the grid states it in, and the joint flag is
    # separate from it: B2 and B5 are Layer B questions read on one series.
    layers = {row.id: row.layer for row in GRID}
    assert all(spec.layer == layers[spec.id] for spec in FORMS)
    assert {spec.id for spec in FORMS if spec.joint} == {"B1", "B3", "B4"}


# ---------------------------------------------------------------------------
# What each form reads
# ---------------------------------------------------------------------------


def test_the_trend_form_separates_a_drifting_panel_from_a_flat_one():
    flat = by_key(panel_series(cohort_of(), axis(), (2,), NAMES))[("walk", "day")]
    rising = by_key(panel_series(cohort_of(drift=3.0), axis(), (2,), NAMES))[
        ("walk", "day")
    ]
    assert (
        form_values(SPEC["A01"], rising).mean()
        > form_values(SPEC["A01"], flat).mean() + 0.5
    )


def test_the_period_form_finds_the_rhythm_the_world_was_built_with(real):
    hourly = by_key(real)[("walk", "hour")]
    at_day = form_values(SPEC["A05"], hourly)
    off_period = np.array([SPEC["A05"].scalar(block, 17.0) for block in hourly.blocks])
    assert at_day.mean() > 3.0 * off_period.mean()
    assert form_values(SPEC["A06"], hourly).mean() > 0.1


def test_the_zero_and_dispersion_forms_read_the_shape_of_the_counts(real):
    zero = form_values(SPEC["A12"], by_key(real)[("walk", "hour")])
    assert np.all((zero > 0.2) & (zero < 1.0))
    dispersion = form_values(SPEC["A13"], by_key(real)[("walk", "day")])
    assert np.all(dispersion > 0.0)


def test_the_dependence_form_needs_two_windows_before_it_reports_anything(real):
    short = Series(
        JOINT_LABEL, "day", True, tuple(np.ones((5, 3)) for _ in range(SUBJECTS))
    )
    assert form_values(SPEC["B3"], short).tolist() == [0.0] * SUBJECTS
    assert np.any(form_values(SPEC["B3"], by_key(real)[(JOINT_LABEL, "day")]) > 0.0)


def test_a_block_whose_labels_never_move_reports_no_dependence_drift():
    flat = Series(JOINT_LABEL, "day", True, tuple(np.ones((8, 3)) for _ in range(2)))
    assert form_values(SPEC["B3"], flat).tolist() == [0.0, 0.0]


def test_the_lagged_row_reads_a_drift_the_same_epoch_row_passes_by():
    rng = np.random.default_rng(21)
    driver = rng.normal(size=120) + 5.0
    held = np.column_stack([driver, np.roll(driver, 1), rng.normal(size=120)])
    broken = held.copy()
    # The lagged relation between the first two columns stops half way and the
    # same-epoch cross-section is where it was, so B4 moves and B3 stays.
    broken[60:, 1] = rng.normal(size=60) + 5.0
    kept = Series(JOINT_LABEL, "day", True, (held, held))
    lost = Series(JOINT_LABEL, "day", True, (broken, broken))
    b3 = abs(form_values(SPEC["B3"], kept)[0] - form_values(SPEC["B3"], lost)[0])
    b4 = abs(form_values(SPEC["B4"], kept)[0] - form_values(SPEC["B4"], lost)[0])
    assert b4 > b3


def test_a_block_too_short_for_two_windows_reports_no_lagged_drift():
    short = Series(
        JOINT_LABEL, "day", True, tuple(np.ones((5, 3)) for _ in range(SUBJECTS))
    )
    assert form_values(SPEC["B4"], short).tolist() == [0.0] * SUBJECTS


def test_the_cross_scale_row_reads_one_series_against_its_parent(real):
    day = by_key(real)[("walk", "day")]
    values = form_values(SPEC["B5"], day)
    assert values.size == len(day.blocks)
    assert np.all(np.isfinite(values))
    assert np.all((values >= -1.0) & (values <= 1.0))


def test_a_series_of_no_contrast_gives_every_form_a_finite_number():
    blocks = tuple(np.zeros(DAYS) for _ in range(SUBJECTS))
    series = Series("walk", "day", False, blocks)
    for spec in FORMS:
        if spec.joint:
            continue
        assert np.all(np.isfinite(form_values(spec, series)))


def test_the_cut_price_is_what_one_more_segment_costs_a_bayesian_criterion():
    assert changepoint_penalty(100, POISSON) == pytest.approx(np.log(100))
    assert changepoint_penalty(100, VARIANCE) == pytest.approx(2 * np.log(100))
    assert changepoint_penalty(1, POISSON) == pytest.approx(np.log(2))


# ---------------------------------------------------------------------------
# The injection calibration
# ---------------------------------------------------------------------------


def test_an_injection_reaches_every_column_of_a_joint_block(real):
    block = by_key(real)[(JOINT_LABEL, "day")].blocks[0]
    moved = inject_block(block, "level_step", 1.0, 7.0, seed=0)
    assert moved.shape == block.shape
    assert np.all(moved[DAYS // 2 :].sum(axis=0) > block[DAYS // 2 :].sum(axis=0))


def test_an_injection_into_a_marginal_block_moves_that_series_alone(real):
    block = by_key(real)[("walk", "day")].blocks[0]
    moved = inject_block(block, "level_step", 1.0, 7.0, seed=0)
    assert moved.ndim == 1
    assert np.array_equal(moved[: DAYS // 2], block[: DAYS // 2])


def test_a_form_catches_the_injection_it_is_calibrated_against(real):
    trend = minimum_detectable_cell(
        SPEC["A01"], by_key(real)[("walk", "day")], sizes=SIZES
    )
    assert trend <= POWER_BOUND


def test_a_size_grid_holding_nothing_the_form_catches_reports_no_effect(real):
    tiny = minimum_detectable_cell(
        SPEC["A01"], by_key(real)[("walk", "day")], sizes=(1e-9,)
    )
    assert tiny == np.inf


# ---------------------------------------------------------------------------
# The cells
# ---------------------------------------------------------------------------


def test_the_denominator_counts_what_the_real_panel_holds(real):
    counts = denominator(by_key(real)[("walk", "day")], 7.0)
    assert counts.carriers == SUBJECTS
    assert counts.episodes > 0
    assert counts.epochs_per_participant == DAYS
    assert counts.cycles == pytest.approx(DAYS / 7.0)


def test_a_panel_drawn_from_the_same_world_fails_less_than_a_broken_one(real, drawn):
    """The panel has to separate a draw of the world from one with a structure gone."""
    same = set(failed_rows(compare(real, drawn, sizes=SIZES)))
    broken = set(
        failed_rows(
            compare(real, [shuffled(drawn[0], TIME_SHUFFLE, seed=1)], sizes=SIZES)
        )
    )
    assert broken - same
    # A within-participant time shuffle is not a monotone degradation.  It
    # destroys the autocorrelation the unit-root row reads, and this draw's own
    # noise on `gym` at the day scale ran away from the real panel, so shuffling
    # it carries that one cell back into agreement: 0.393 unshuffled against
    # 0.004 shuffled.  Every other row the shuffle only breaks.
    assert same - broken == {"A03"}


def test_a_cell_whose_tolerance_is_under_its_own_spread_is_descriptive(real):
    """An equivalence claim narrower than the noise fails on two draws of one world."""
    cells = compare(real, [real[:]], sizes=SIZES)
    thin = [c for c in cells if c.inclusion.clause == "precision"]
    assert thin
    for cell in thin:
        assert not cell.inclusion.tested


def test_a_draw_against_another_draw_is_the_positive_control(real):
    """One draw against another separates nothing, so what fails there is noise."""
    third = panel_series(cohort_of(offset=200), axis(), (1, 2, 3), NAMES)
    fourth = panel_series(cohort_of(offset=300), axis(), (1, 2, 3), NAMES)
    control = compare(third, [fourth], sizes=SIZES)
    assert rows_tested(control)
    # Nothing separates the two, so a row failing here reports its own spread.
    assert set(failed_rows(control)) < set(rows_tested(control))


def test_the_changepoint_count_is_too_noisy_per_participant_to_claim_equivalence(real):
    """Two draws of one world differ by more cuts than the tolerance allows."""
    other = by_key(panel_series(cohort_of(offset=100), axis(), (1,), {1: "hour"}))
    series = by_key(real)[("gym", "hour")]
    a = form_values(SPEC["A08"], series) * series.n_epochs
    b = form_values(SPEC["A08"], other[("gym", "hour")]) * series.n_epochs
    assert abs(float(np.mean(a - b))) < 4.0
    # The mean is small and the spread across participants is not, which is
    # what the positive control is there to say.
    assert float(np.std(a - b, ddof=1)) > 4.0


def test_every_replicates_participants_join_one_drawn_group(real):
    # The contrast is between two groups of participants, so a second replicate
    # brings a second set of drawn participants and never a second reading of
    # the first set.  The drawn group doubles and the contrast tightens.
    once = compare(real, [real[:]], sizes=SIZES)
    twice = compare(real, [real[:], real[:]], sizes=SIZES)
    assert [c.p_value for c in twice] != [c.p_value for c in once]
    pairs = [
        (a.p_value, b.p_value)
        for a, b in zip(once, twice)
        if a.inclusion.tested and 0.0 < a.p_value < 1.0
    ]
    assert pairs
    assert all(wide <= tight for tight, wide in pairs)


def test_a_drawn_group_of_one_world_carries_no_standardized_difference(real):
    for cell in compare(real, [real[:], real[:]], sizes=SIZES):
        assert abs(cell.effect) < 1e-9


def test_the_standardized_difference_grows_with_a_moved_drawn_group(real):
    moved = [
        Series(s.label, s.scale, s.joint, tuple(b + 3.0 for b in s.blocks))
        for s in real
    ]
    cells = {
        c.form + c.series + c.scale: c.effect
        for c in compare(real, [moved], sizes=SIZES)
    }
    assert any(abs(v) > 0.5 for v in cells.values())


def test_a_cell_reports_the_clause_that_made_it_descriptive():
    thin = panel_series(cohort_of(n=4), axis(), (2,), NAMES)
    other = panel_series(cohort_of(offset=100, n=4), axis(), (2,), NAMES)
    thin_cells = compare(thin, [other], sizes=SIZES)
    assert all(c.inclusion.status == DESCRIPTIVE for c in thin_cells)
    assert {c.inclusion.clause for c in thin_cells} == {"denominator"}


def test_the_week_scale_is_descriptive_for_want_of_epochs(cells):
    week = [c for c in cells if c.scale == WEEK]
    assert week
    assert {c.inclusion.clause for c in week} == {"epochs"}


def test_a_tested_cell_names_no_clause(cells):
    tested = [c for c in cells if c.inclusion.tested]
    assert tested
    assert all(
        c.inclusion.clause == "" and c.inclusion.status == TESTED for c in tested
    )


def test_a_comparison_with_no_synthetic_panel_is_refused(real):
    with pytest.raises(ValueError, match="at least one"):
        compare(real, [])


def test_a_synthetic_panel_missing_a_series_is_refused(real, drawn):
    short = [item for item in drawn[0] if item.key != ("walk", "day")]
    with pytest.raises(ValueError, match="carries no"):
        compare(real, [short], sizes=SIZES)


def test_the_summary_reports_the_cells_without_holding_them(cells):
    report = summary(cells)
    assert report["n_cells"] > report["n_tested"] >= 1
    assert report["passed"] == (report["failed"] == [])
    assert set(report["failed"]) <= set(report["tested_rows"])
    assert set(report["tested_rows"]) <= set(form_ids())
    assert "epochs" in report["descriptive_by_clause"]


def test_the_summary_names_the_label_scale_and_effect_of_every_tested_cell(cells):
    report = summary(cells)
    rows = report["tested_cells"]
    assert len(rows) == report["n_tested"]
    assert all(row["status"] == TESTED for row in rows)
    assert all({"series", "scale", "magnitude", "effect"} <= set(row) for row in rows)
    failing = [row for row in rows if row["p_value"] > 0.05]
    assert {row["form"] for row in failing} == set(report["failed"])


# ---------------------------------------------------------------------------
# The two shuffles
# ---------------------------------------------------------------------------


def test_a_time_shuffle_breaks_the_trend_and_keeps_the_histogram():
    drifting = panel_series(cohort_of(drift=3.0), axis(), (2,), NAMES)
    moved = shuffled(drifting, TIME_SHUFFLE, seed=1)
    before = by_key(drifting)[("walk", "day")]
    after = by_key(moved)[("walk", "day")]
    assert np.array_equal(np.sort(before.blocks[0]), np.sort(after.blocks[0]))
    kept = float(np.abs(form_values(SPEC["A01"], before)).mean())
    gone = float(np.abs(form_values(SPEC["A01"], after)).mean())
    assert kept > 1.0
    assert gone < 0.3 * kept


def test_a_label_shuffle_breaks_the_cross_section_and_leaves_the_marginals(real, drawn):
    moved = shuffled(drawn[0], LABEL_SHUFFLE, seed=1)
    keys = by_key(moved)
    assert np.array_equal(
        keys[("walk", "day")].blocks[0], by_key(drawn[0])[("walk", "day")].blocks[0]
    )
    joint = keys[(JOINT_LABEL, "day")].blocks[0]
    for j in range(len(LABELS)):
        assert np.array_equal(
            np.sort(joint[:, j]),
            np.sort(by_key(drawn[0])[(JOINT_LABEL, "day")].blocks[0][:, j]),
        )


def test_an_undeclared_shuffle_is_refused(drawn):
    with pytest.raises(ValueError, match="expected"):
        shuffled(drawn[0], "bootstrap")


# ---------------------------------------------------------------------------
# What a cell is priced at
# ---------------------------------------------------------------------------


def test_a_tolerance_is_half_the_effect_the_row_exists_to_catch(real):
    series = by_key(real)[("walk", "hour")]
    effect = declared_effect(SPEC["A01"], series)
    assert effect > 0.0
    assert tolerance_of(SPEC["A01"], series) == pytest.approx(TOLERANCE_SHARE * effect)


def test_a_tolerance_is_derived_per_series_and_not_shared_across_scales(real):
    keys = by_key(real)
    hourly = tolerance_of(SPEC["A08"], keys[("walk", "hour")])
    daily = tolerance_of(SPEC["A08"], keys[("walk", "day")])
    assert hourly > 0.0 and daily > 0.0
    # A cut rate on 840 epochs and one on 35 are different quantities, so the
    # effect measured on each sets its own tolerance.
    assert hourly != daily


def test_a_form_that_sees_its_own_perturbation_catches_it_at_the_declared_size(real):
    """The tolerance is half the declared effect, so the declared size clears it."""
    keys = by_key(real)
    seen = 0
    for spec in FORMS:
        for series in real:
            if not admits(spec, series):
                continue
            if declared_effect(spec, series) > 0.0:
                assert minimum_detectable_cell(spec, series) <= spec.size, spec.id
                seen += 1
            else:
                assert minimum_detectable_cell(spec, series) == np.inf, spec.id
    assert seen > 0
    assert keys
    assert all(spec.size in INJECTION_SIZES for spec in FORMS)
    assert all(spec.size <= POWER_BOUND for spec in FORMS)


def test_a_slope_break_is_what_the_slope_changepoint_row_can_see(real):
    """A linear cost absorbs a tilt of the whole series, so only a break buys a cut."""
    tilted = FormSpec("A10", MARGINAL, False, "slope", 0.8, SPEC["A10"].scalar)
    broken = FormSpec("A10", MARGINAL, False, "slope_break", 0.8, SPEC["A10"].scalar)
    moved = 0
    for series in real:
        if series.joint:
            continue
        assert declared_effect(tilted, series) == 0.0, series.key
        moved += declared_effect(broken, series) > 0.0
    assert moved > 0


def test_the_standard_error_of_one_participant_is_zero(real):
    assert standard_error(np.array([2.0])) == 0.0
    assert standard_error(np.array([])) == 0.0
    assert standard_error(form_values(SPEC["A01"], by_key(real)[("walk", "day")])) > 0.0


def test_a_row_that_cannot_see_its_own_effect_has_no_tolerance_and_no_power(real):
    flat = Series(
        "walk", "day", False, tuple(np.full(DAYS, 10.0) for _ in range(SUBJECTS))
    )
    # A dispersion perturbation scales each epoch's deviation from the level, and
    # a series of no deviation has none to scale, so this row is left with
    # nothing to forgive and nothing to detect.
    assert declared_effect(SPEC["A13"], flat) == 0.0
    assert tolerance_of(SPEC["A13"], flat) == 0.0
    assert minimum_detectable_cell(SPEC["A13"], flat) == np.inf
    cells = compare([flat], [[flat]], forms=(SPEC["A13"],), sizes=SIZES)
    assert cells[0].inclusion.clause == "power"
    assert cells[0].p_value == 1.0


def test_the_tolerance_table_names_every_cell_and_digests_stably(real):
    table = tolerance_table(real)
    assert table
    assert all(key.count("|") == 2 for key in table)
    assert all(value >= 0.0 for value in table.values())
    assert table_digest(table) == table_digest(tolerance_table(real))
    moved = dict(table)
    moved[sorted(moved)[0]] += 1.0
    assert table_digest(moved) != table_digest(table)


def test_a_perturbation_of_no_size_moves_the_form_by_nothing(real):
    series = by_key(real)[("walk", "hour")]
    assert np.allclose(
        perturbed(SPEC["A01"], series, 0.0), form_values(SPEC["A01"], series)
    )


def test_the_per_form_breakdown_says_how_much_of_the_grid_each_row_stood_on(cells):
    rows = by_form(cells)
    assert set(rows) <= set(form_ids())
    for form, row in rows.items():
        assert row["cells"] == row["tested"] + sum(row["clauses"].values())
        assert row["failed"] <= row["tested"]
    assert sum(row["cells"] for row in rows.values()) == len(cells)
