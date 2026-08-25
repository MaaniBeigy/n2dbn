"""
Tests for the enumerated non-stationarity grid.

Three properties carry the panel.  Every implemented form has a finite minimum
detectable effect, so a silent check is separated from an underpowered one.  Each
negative control breaks the rows it names and leaves the rows it does not.  And
the inclusion rule partitions a cell on the real log alone, naming the clause
that decided it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.evaluation.nonstationarity import (
    CONTROLS,
    DECLARED,
    DESCRIPTIVE,
    GRID,
    IMPLEMENTED,
    INJECTIONS,
    JOINT,
    MARGINAL,
    TESTED,
    Cell,
    Denominator,
    Inclusion,
    NegativeControl,
    benjamini_hochberg,
    circular_distance,
    control_verdict,
    dependence_distance,
    detail_profile_distance,
    group_permutation_p,
    group_tost,
    haar_detail_energy,
    harmonic,
    hedges_g,
    implemented_forms,
    inclusion,
    inject,
    jaccard,
    lagged_correlation,
    mann_kendall,
    minimum_detectable,
    panel_verdict,
    period_significance,
    periodogram,
    run_lengths,
    shuffle_labels,
    shuffle_within_participant,
    sign_flip_p,
    significant_periods,
    theil_sen,
    tost,
    variance_mean_ratio,
    windowed_correlation,
    zero_rate,
)

PERIOD = 24.0
PERIODS = (6.0, 8.0, 12.0, PERIOD, 48.0)
N = 480


def rhythm(seed: int = 0, amplitude: float = 1.0, noise: float = 0.2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(N, dtype=float)
    return (
        5.0 + amplitude * np.sin(2.0 * np.pi * t / PERIOD) + noise * rng.normal(size=N)
    )


N_UNITS = 8
UNIT_LENGTH = N // N_UNITS
UNITS = [(i * UNIT_LENGTH, (i + 1) * UNIT_LENGTH) for i in range(N_UNITS)]


def paired_detector(statistic):
    """Compare two series by the sign-flip null on their per-participant difference."""

    def detect(a, b):
        return sign_flip_p(
            [
                float(statistic(a[start:stop]) - statistic(b[start:stop]))
                for start, stop in UNITS
            ],
            n_flips=199,
        )

    return detect


# ---------------------------------------------------------------------------
# The catalogue
# ---------------------------------------------------------------------------


def test_the_catalogue_holds_every_declared_row_and_names_what_each_waits_on():
    ids = [row.id for row in GRID]
    assert len([i for i in ids if i.startswith("A")]) == 16
    assert len([i for i in ids if i.startswith("B")]) == 5
    assert len(set(ids)) == len(ids)
    for row in GRID:
        assert row.layer in (MARGINAL, JOINT)
        assert row.status in (IMPLEMENTED, DECLARED)
        if row.status == DECLARED:
            assert row.needs, f"{row.id} omits its reason"
        else:
            assert not row.needs


def test_the_implemented_rows_are_the_ones_this_module_computes():
    assert implemented_forms() == (
        "A01",
        "A02",
        "A03",
        "A04",
        "A05",
        "A06",
        "A07",
        "A08",
        "A09",
        "A10",
        "A11",
        "A12",
        "A13",
        "A14",
        "A15",
        "A16",
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
    )
    # B4 reads the lagged cross-section the same way B3 reads the same-epoch
    # one, and B5 folds a series into its parent blocks and decodes both, so
    # each one fits the one-series-at-one-scale cell after all.
    assert [row.id for row in GRID if row.status == DECLARED] == []
    assert len(implemented_forms()) == len(GRID) == 21


# ---------------------------------------------------------------------------
# A01, monotone trend
# ---------------------------------------------------------------------------


def test_the_trend_statistic_reads_a_rise_and_a_fall_the_slope_cannot():
    t = np.arange(200, dtype=float)
    rise_then_fall = np.concatenate([t[:100], t[:100][::-1]])
    assert abs(np.polyfit(np.arange(200.0), rise_then_fall, 1)[0]) < 1e-9
    assert mann_kendall(rise_then_fall[100:]).p_value < 1e-6
    rising = mann_kendall(t)
    assert rising.s > 0 and rising.p_value < 1e-6
    assert theil_sen(t) == pytest.approx(1.0)


def test_a_flat_or_a_short_series_carries_no_trend():
    assert mann_kendall(np.ones(50)).p_value == 1.0
    assert mann_kendall([1.0, 2.0]).s == 0.0
    assert mann_kendall([1.0]).p_value == 1.0
    assert theil_sen([3.0]) == 0.0
    assert mann_kendall(np.ones(3)).z == 0.0


# ---------------------------------------------------------------------------
# A04, scale-localized drift
# ---------------------------------------------------------------------------


def test_a_level_shift_puts_its_energy_in_the_coarse_details():
    flat = np.zeros(64)
    stepped = flat.copy()
    stepped[32:] = 4.0
    quiet = haar_detail_energy(flat, 6)
    loud = haar_detail_energy(stepped, 6)
    assert np.all(quiet == 0.0)
    assert loud[0] == 0.0
    assert loud[-1] > 0.0
    assert detail_profile_distance(loud, quiet) > 0.0
    assert detail_profile_distance(loud, loud) == 0.0


def test_a_series_too_short_to_fold_stops_where_it_runs_out():
    assert np.array_equal(haar_detail_energy([1.0], 3), np.zeros(3))
    assert haar_detail_energy([1.0, 3.0], 2)[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# A05 and A06, period discovery and profile
# ---------------------------------------------------------------------------


def test_the_period_of_a_diurnal_series_is_discovered_and_nothing_else_is():
    assert significant_periods(rhythm(), PERIODS) == (PERIOD,)
    noise = np.random.default_rng(1).normal(size=N)
    assert significant_periods(noise, PERIODS) == ()


def test_a_flat_or_a_one_point_series_has_no_periodogram():
    assert np.all(periodogram(np.ones(50), PERIODS) == 0.0)
    assert np.all(periodogram([1.0], PERIODS) == 0.0)
    assert period_significance(np.ones(50), PERIODS) == (0.0, 1.0)
    assert period_significance([1.0], ()) == (0.0, 1.0)
    assert significant_periods(np.ones(50), PERIODS) == ()
    assert significant_periods(rhythm(), ()) == ()


def test_the_harmonic_reads_the_amplitude_and_the_phase_it_was_given():
    t = np.arange(N, dtype=float)
    amplitude, phase = harmonic(2.5 * np.cos(2.0 * np.pi * t / PERIOD - 0.7), PERIOD)
    assert amplitude == pytest.approx(2.5, abs=1e-6)
    assert circular_distance(phase, 0.7) < 1e-6


def test_two_phases_are_compared_the_short_way_round():
    assert circular_distance(0.1, 2.0 * math.pi - 0.1) == pytest.approx(0.2, abs=1e-9)
    assert circular_distance(1.0, 1.0) == 0.0


def test_two_discovered_sets_are_compared_by_overlap():
    assert jaccard((), ()) == 1.0
    assert jaccard((24.0,), (24.0,)) == 1.0
    assert jaccard((24.0,), (12.0,)) == 0.0
    assert jaccard((24.0, 12.0), (24.0,)) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# A12 and A13
# ---------------------------------------------------------------------------


def test_the_zero_rate_and_the_run_lengths_read_the_intermittency():
    values = np.array([0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 3.0])
    assert np.array_equal(zero_rate(values, 4), np.array([0.5, 0.75]))
    assert np.array_equal(run_lengths(values, zero=True), np.array([2, 3]))
    assert np.array_equal(run_lengths(values, zero=False), np.array([2, 1]))


def test_the_dispersion_ratio_is_zero_where_the_window_is_empty():
    values = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 5.0, 1.0, 5.0])
    ratios = variance_mean_ratio(values, 4)
    assert ratios[0] == 0.0
    assert ratios[1] > 0.0


def test_a_window_holds_at_least_two_epochs():
    with pytest.raises(ValueError, match="at least two epochs"):
        zero_rate(np.zeros(10), 1)


# ---------------------------------------------------------------------------
# B3
# ---------------------------------------------------------------------------


def test_a_level_change_does_not_arrive_as_a_dependence_change():
    rng = np.random.default_rng(2)
    driver = rng.normal(size=200)
    matrix = np.column_stack([driver, driver + 0.1 * rng.normal(size=200)])
    shifted = matrix.copy()
    shifted[100:] += 20.0
    plain = windowed_correlation(matrix, 50)
    stepped = windowed_correlation(shifted, 50)
    assert plain.shape == (4, 2, 2)
    assert dependence_distance(plain, stepped) < 1e-9


def test_a_dependence_break_does_arrive_as_one():
    rng = np.random.default_rng(3)
    driver = rng.normal(size=200)
    matrix = np.column_stack([driver, driver + 0.1 * rng.normal(size=200)])
    broken = matrix.copy()
    broken[100:, 1] = rng.normal(size=100)
    assert (
        dependence_distance(
            windowed_correlation(matrix, 50), windowed_correlation(broken, 50)
        )
        > 0.3
    )
    assert dependence_distance(np.zeros((0, 2, 2)), np.zeros((0, 2, 2))) == 0.0


def test_a_relation_that_lives_at_a_lag_is_invisible_at_the_same_epoch():
    rng = np.random.default_rng(11)
    driver = rng.normal(size=200)
    follower = np.roll(driver, 1)
    matrix = np.column_stack([driver, follower])
    same = windowed_correlation(matrix, 50)
    lagged = lagged_correlation(matrix, 50)
    assert same.shape == lagged.shape == (4, 2, 2)
    # Column 1 repeats column 0 one epoch later, so the pair meets at lag one
    # and passes the same-epoch cross-section by.
    assert abs(same[0, 0, 1]) < 0.2
    assert lagged[0, 0, 1] > 0.9
    # The matrix is asymmetric: the relation runs one way in time.
    assert abs(lagged[0, 1, 0]) < 0.2


def test_a_lagged_dependence_break_moves_the_lagged_matrix_alone():
    rng = np.random.default_rng(12)
    driver = rng.normal(size=200)
    matrix = np.column_stack([driver, np.roll(driver, 1)])
    broken = matrix.copy()
    broken[100:, 1] = rng.normal(size=100)
    assert (
        dependence_distance(
            lagged_correlation(matrix, 50), lagged_correlation(broken, 50)
        )
        > 0.3
    )


def test_a_lagged_correlation_needs_a_lag_and_a_window_to_hold_it():
    matrix = np.column_stack([np.arange(20.0), np.arange(20.0) ** 2])
    with pytest.raises(ValueError, match="one epoch or more"):
        lagged_correlation(matrix, 10, lag=0)
    with pytest.raises(ValueError, match="holds no pair at lag"):
        lagged_correlation(matrix, 3, lag=3)


def test_a_column_that_never_moves_carries_no_lagged_relation():
    matrix = np.column_stack([np.arange(20.0), np.ones(20)])
    windows = lagged_correlation(matrix, 10)
    assert np.all(windows[:, 0, 1] == 0.0)
    assert np.all(windows[:, 1, 0] == 0.0)


def test_a_column_that_never_moves_correlates_with_nothing():
    matrix = np.column_stack([np.arange(20.0), np.ones(20)])
    windows = windowed_correlation(matrix, 10)
    assert np.all(windows[:, 0, 1] == 0.0)


# ---------------------------------------------------------------------------
# How a cell is compared
# ---------------------------------------------------------------------------


def test_the_sign_flip_null_reads_the_participant_as_the_unit():
    assert sign_flip_p([]) == 1.0
    assert sign_flip_p([5.0] * 12, n_flips=999) < 0.01
    assert sign_flip_p([0.1, -0.1, 0.05, -0.05], n_flips=999) > 0.2


def test_a_difference_inside_its_tolerance_is_equivalent_and_outside_it_is_not():
    rng = np.random.default_rng(4)
    tight = tost(0.01 + 0.02 * rng.normal(size=40), tolerance=0.2)
    assert tight.equivalent()
    assert tight.p_value < 0.05
    wide = tost(0.5 + 0.02 * rng.normal(size=40), tolerance=0.2)
    assert not wide.equivalent()
    assert wide.difference == pytest.approx(0.5, abs=0.02)


def test_an_equivalence_needs_a_positive_tolerance_and_more_than_one_unit():
    with pytest.raises(ValueError, match="is positive"):
        tost([0.1, 0.2], 0.0)
    assert tost([0.1], 0.5).p_value == 1.0
    assert tost([], 0.5).difference == 0.0
    assert tost([0.1, 0.1], 0.5).equivalent()
    assert not tost([9.0, 9.0], 0.5).equivalent()


def test_the_difference_direction_is_controlled_inside_one_form_family():
    rejected = benjamini_hochberg([0.001, 0.2, 0.6, 0.9], q=0.05)
    assert list(rejected) == [True, False, False, False]
    assert not benjamini_hochberg([0.4, 0.5]).any()
    assert benjamini_hochberg([]).size == 0


# ---------------------------------------------------------------------------
# The inclusion rule
# ---------------------------------------------------------------------------


def test_a_cell_clearing_every_clause_is_tested():
    plenty = Denominator(
        episodes=1000, carriers=40, epochs_per_participant=60, cycles=8
    )
    verdict = inclusion(plenty, minimum_detectable_effect=0.1, bound=0.5)
    assert verdict.tested
    assert verdict.status == TESTED
    assert verdict.clause == ""


@pytest.mark.parametrize(
    "denominator, mde, needs_period, clause",
    [
        (Denominator(100, 40, 60, 8), 0.1, False, "denominator"),
        (Denominator(1000, 5, 60, 8), 0.1, False, "denominator"),
        (Denominator(1000, 40, 10, 8), 0.1, False, "epochs"),
        (Denominator(1000, 40, 60, 1.5), 0.1, True, "definedness"),
        (Denominator(1000, 40, 60, 8), 0.9, False, "power"),
        (Denominator(1000, 40, 60, 8), math.inf, False, "power"),
    ],
)
def test_a_cell_that_fails_a_clause_is_descriptive_and_names_it(
    denominator, mde, needs_period, clause
):
    verdict = inclusion(denominator, mde, bound=0.5, needs_period=needs_period)
    assert verdict.status == DESCRIPTIVE
    assert verdict.clause == clause
    assert not verdict.tested


def test_a_bound_that_is_not_positive_is_refused():
    with pytest.raises(ValueError, match="is positive"):
        inclusion(Denominator(1000, 40, 60, 8), 0.1, bound=0.0)


# ---------------------------------------------------------------------------
# The injection calibration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", INJECTIONS)
def test_every_injection_moves_the_series_it_is_put_into(kind):
    base = rhythm()
    moved = inject(base, kind, 0.5, period=PERIOD)
    assert moved.shape == base.shape
    assert not np.allclose(moved, base)
    assert np.allclose(inject(base, kind, 0.0, period=PERIOD), base, atol=1e-9)


def test_an_injection_nobody_declared_is_refused():
    with pytest.raises(ValueError, match="injection"):
        inject(rhythm(), "teleport", 0.5)


def test_every_implemented_form_has_a_finite_minimum_detectable_effect():
    base = rhythm()
    sizes = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
    calibration = {
        "A01": ("slope", paired_detector(lambda x: mann_kendall(x).s)),
        "A04": ("level_step", paired_detector(lambda x: haar_detail_energy(x, 3)[-1])),
        "A06": ("phase", paired_detector(lambda x: harmonic(x, PERIOD)[0])),
        "A12": ("zero_rate", paired_detector(lambda x: float((x == 0.0).mean()))),
        "A13": ("dispersion", paired_detector(lambda x: float(np.var(x)))),
    }
    for form, (kind, detector) in calibration.items():
        mde = minimum_detectable(
            base, kind, detector, sizes, period=PERIOD, blocks=UNITS
        )
        assert math.isfinite(mde), f"{form} is never active"
        assert mde <= 1.0


def test_a_check_that_never_is_active_reports_an_infinite_effect():
    assert (
        minimum_detectable(rhythm(), "level_step", lambda a, b: 1.0, [0.1, 1.0])
        == math.inf
    )


def test_the_zero_rate_injection_only_ever_removes_mass():
    base = rhythm()
    dropped = inject(base, "zero_rate", 0.8)
    assert (dropped == 0.0).sum() > 0
    assert np.all((dropped == 0.0) | (dropped == base))


# ---------------------------------------------------------------------------
# The negative controls
# ---------------------------------------------------------------------------


def test_the_four_controls_each_name_the_rows_they_must_break():
    assert len(CONTROLS) == 4
    names = {control.name for control in CONTROLS}
    assert "no regime" in names
    assert "within-participant time shuffle" in names
    for control in CONTROLS:
        assert control.must_fail
        assert not set(control.must_fail) & set(control.must_pass)


def test_a_within_participant_shuffle_breaks_the_marginal_rows_and_keeps_the_histogram():
    who = np.repeat(np.arange(2), N // 2)
    t = np.arange(N, dtype=float)
    trending = rhythm() + 0.05 * t
    shuffled = shuffle_within_participant(trending, who, seed=5)
    assert np.array_equal(np.sort(shuffled), np.sort(trending))
    for unit in (0, 1):
        rows = who == unit
        assert np.array_equal(np.sort(shuffled[rows]), np.sort(trending[rows]))
    # A01 and A05 both read the ordering inside a participant, and the shuffle
    # is what removes it while leaving the between-participant contrast alone.
    for unit in (0, 1):
        rows = who == unit
        assert mann_kendall(trending[rows]).p_value < 1e-6
        assert mann_kendall(shuffled[rows]).p_value > 0.05
    assert significant_periods(trending, PERIODS) == (PERIOD,)
    assert PERIOD not in significant_periods(shuffled, PERIODS)


def test_a_label_shuffle_breaks_the_cross_section_and_leaves_each_column_alone():
    rng = np.random.default_rng(6)
    driver = rng.normal(size=200)
    matrix = np.column_stack([driver, driver + 0.1 * rng.normal(size=200)])
    shuffled = shuffle_labels(matrix, seed=7)
    for j in range(matrix.shape[1]):
        assert np.array_equal(np.sort(shuffled[:, j]), np.sort(matrix[:, j]))
    assert (
        dependence_distance(
            windowed_correlation(matrix, 50), windowed_correlation(shuffled, 50)
        )
        > 0.5
    )


def test_a_control_that_breaks_what_it_named_did_its_job():
    control = NegativeControl("test", ("A01", "A05"), ("B3",))
    passed, detail = control_verdict(control, ["A01", "A05"])
    assert passed
    assert detail == {
        "missed": [],
        "spurious": [],
        "untested": [],
        "already_failing": [],
    }


def test_a_control_that_misses_a_row_or_breaks_a_spare_one_did_not():
    control = NegativeControl("test", ("A01", "A05"), ("B3",))
    passed, detail = control_verdict(control, ["A01", "B3"])
    assert not passed
    assert detail == {
        "missed": ["A05"],
        "spurious": ["B3"],
        "untested": [],
        "already_failing": [],
    }


def test_a_row_the_run_never_tested_is_reported_apart_from_one_that_held():
    control = NegativeControl("test", ("A01", "A05"), ())
    passed, detail = control_verdict(control, ["A01"], computed=["A01"])
    assert passed
    assert detail["untested"] == ["A05"]
    assert detail["missed"] == []


def test_a_row_already_failing_is_not_credited_to_the_control():
    control = NegativeControl("test", ("A01", "A05"), ("B3",))
    passed, detail = control_verdict(
        control, ["A01", "A05"], computed=["A01", "A05"], baseline=["A01"]
    )
    assert passed
    assert detail["already_failing"] == ["A01"]
    assert detail["missed"] == []


def test_a_row_already_failing_is_not_charged_to_the_control_either():
    control = NegativeControl("test", ("A05",), ("A01",))
    passed, detail = control_verdict(
        control, ["A01", "A05"], computed=["A01", "A05"], baseline=["A01"]
    )
    assert passed
    assert detail["spurious"] == []


# ---------------------------------------------------------------------------
# What a cell reports
# ---------------------------------------------------------------------------


def cell(form: str, p_value: float, status: str) -> Cell:
    return Cell(
        form=form,
        series="walk",
        scale="day",
        magnitude=0.1,
        spread=0.02,
        p_value=p_value,
        inclusion=Inclusion(status, "" if status == TESTED else "power"),
        denominator=Denominator(1000, 40, 60, 8),
    )


def test_one_failing_tested_cell_decides_the_panel():
    passing = [cell("A01", 0.01, TESTED), cell("A04", 0.02, TESTED)]
    verdict, failing = panel_verdict(passing)
    assert verdict and failing == []
    verdict, failing = panel_verdict(passing + [cell("A05", 0.4, TESTED)])
    assert not verdict
    assert [c.form for c in failing] == ["A05"]


def test_a_descriptive_cell_is_reported_and_never_decides_the_panel():
    cells = [cell("A01", 0.01, TESTED), cell("A12", 0.9, DESCRIPTIVE)]
    verdict, failing = panel_verdict(cells)
    assert verdict and failing == []
    row = cells[1].as_row()
    assert row["status"] == DESCRIPTIVE
    assert row["clause"] == "power"
    assert row["episodes"] == 1000
    assert row["form"] == "A12"


# ---------------------------------------------------------------------------
# The two-group contrast, which the population design of 39.2 reads
# ---------------------------------------------------------------------------


def test_two_groups_of_one_world_are_equivalent_inside_a_wide_tolerance():
    rng = np.random.default_rng(0)
    a = rng.normal(10.0, 1.0, 60)
    b = rng.normal(10.0, 1.0, 60)
    assert group_tost(a, b, 1.0).equivalent()
    assert not group_tost(a, b + 3.0, 1.0).equivalent()


def test_the_two_group_test_asks_neither_group_for_the_others_spread():
    rng = np.random.default_rng(1)
    tight = rng.normal(5.0, 0.1, 40)
    wide = rng.normal(5.0, 4.0, 40)
    # Welch keeps the wide group's own spread, so the pair is not called
    # equivalent on the tight group's precision.
    assert not group_tost(tight, wide, 0.2).equivalent()
    assert group_tost(tight, rng.normal(5.0, 0.1, 40), 0.2).equivalent()


def test_a_group_of_one_carries_no_spread_and_claims_no_equivalence():
    assert group_tost([1.0], [1.0, 1.0, 1.0], 1.0).p_value == 1.0
    assert group_tost([1.0, 1.0], [1.0, 1.0], 1.0).p_value == 0.0
    assert group_tost([1.0, 1.0], [9.0, 9.0], 1.0).p_value == 1.0
    with pytest.raises(ValueError, match="tolerance is positive"):
        group_tost([1.0, 2.0], [1.0, 2.0], 0.0)


def test_the_effect_size_is_the_two_group_g_and_never_the_paired_one():
    rng = np.random.default_rng(2)
    a = rng.normal(1.0, 1.0, 5000)
    b = rng.normal(0.0, 1.0, 5000)
    # One pooled standard deviation apart, less the small-sample factor.
    assert hedges_g(a, b) == pytest.approx(1.0, abs=0.05)
    assert hedges_g(b, a) == pytest.approx(-hedges_g(a, b))
    assert hedges_g(a, a) == pytest.approx(0.0, abs=1e-12)


def test_a_standardized_difference_with_no_spread_or_no_group_reports_zero():
    assert hedges_g([1.0, 1.0], [2.0, 2.0]) == 0.0
    assert hedges_g([1.0], [2.0, 3.0]) == 0.0
    assert hedges_g([], []) == 0.0


def test_the_small_sample_factor_pulls_the_standardized_difference_toward_zero():
    rng = np.random.default_rng(3)
    a = rng.normal(1.0, 1.0, 6)
    b = rng.normal(0.0, 1.0, 6)
    dof = 10
    pooled = np.sqrt((5 * a.var(ddof=1) + 5 * b.var(ddof=1)) / dof)
    raw = (a.mean() - b.mean()) / pooled
    assert abs(hedges_g(a, b)) < abs(raw)
    assert hedges_g(a, b) == pytest.approx(raw * (1 - 3 / (4 * dof - 1)))


def test_the_null_moves_whole_participants_between_the_groups():
    rng = np.random.default_rng(4)
    same = rng.normal(0.0, 1.0, 40)
    other = rng.normal(0.0, 1.0, 40)
    assert group_permutation_p(same, other, n_draws=199, seed=0) > 0.1
    assert group_permutation_p(same, other + 5.0, n_draws=199, seed=0) < 0.01


def test_a_permutation_null_needs_both_groups_and_a_draw():
    with pytest.raises(ValueError, match="needs both groups"):
        group_permutation_p([], [1.0], n_draws=9)
    with pytest.raises(ValueError, match="needs both groups"):
        group_permutation_p([1.0], [], n_draws=9)
    with pytest.raises(ValueError, match="at least one draw"):
        group_permutation_p([1.0], [2.0], n_draws=0)


def test_the_permutation_p_value_is_bounded_by_its_own_draw_count():
    p = group_permutation_p([0.0, 0.0, 0.0], [9.0, 9.0, 9.0], n_draws=19, seed=1)
    assert p == pytest.approx(1 / 20)
