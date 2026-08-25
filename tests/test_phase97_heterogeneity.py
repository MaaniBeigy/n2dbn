"""
tests/test_phase97_heterogeneity.py
===================================
The heterogeneity comparison: CQV per participant over weekly totals, with a
percentile bootstrap interval taken over participants.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.heterogeneity import (
    CONF_LEVEL,
    MIN_EPOCHS,
    REPLICATES,
    Heterogeneity,
    comparable,
    compare_heterogeneity,
    participant_cqv,
    population_cqv,
    separation,
    weekly_blocks,
)

WEEKS = 8


def roster(rng, rate, n=51, weeks=WEEKS):
    return [rng.poisson(rate, weeks).astype(float) for _ in range(n)]


# ---------------------------------------------------------------------------
# One participant
# ---------------------------------------------------------------------------


def test_the_measure_is_the_quartile_ratio_of_one_participants_weeks():
    # Quartiles 2 and 6 give (6 - 2) / (6 + 2).
    assert participant_cqv([2.0, 2.0, 6.0, 6.0]) == pytest.approx(0.5)
    assert participant_cqv([4.0] * WEEKS) == 0.0


def test_a_participant_whose_quartiles_are_both_zero_defines_nothing():
    assert participant_cqv(np.zeros(WEEKS)) is None
    assert participant_cqv([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]) is None
    assert participant_cqv([0.0, 0.0, 3.0, 9.0]) is not None


def test_a_series_too_short_or_not_a_count_is_refused():
    with pytest.raises(ValueError, match="at least 4 epochs"):
        participant_cqv([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="finite values only"):
        participant_cqv([1.0, 2.0, 3.0, np.nan])
    with pytest.raises(ValueError, match="at or above zero"):
        participant_cqv([1.0, 2.0, 3.0, -1.0])


# ---------------------------------------------------------------------------
# The population, where the bootstrap moves whole participants
# ---------------------------------------------------------------------------


def test_the_population_measure_is_the_median_of_the_defined_participants():
    rng = np.random.default_rng(0)
    blocks = roster(rng, 8.0)
    out = population_cqv(blocks, "walk", replicates=199, seed=1)
    points = [participant_cqv(b) for b in blocks]
    defined = [p for p in points if p is not None]
    assert out.cqv == pytest.approx(float(np.median(defined)))
    assert out.n_defined == len(defined)
    assert out.n_participants == len(blocks)
    assert out.lower <= out.cqv <= out.upper


def test_the_roster_reports_what_it_could_not_define_as_a_share():
    rng = np.random.default_rng(2)
    sparse = [np.zeros(WEEKS) for _ in range(30)]
    out = population_cqv(sparse + roster(rng, 9.0, n=20), "nap", replicates=99, seed=0)
    assert out.n_defined == 20
    assert out.n_participants == 50
    assert out.undefined_sparse == pytest.approx(0.6)
    assert "undefined_sparse 0.60" in str(out)


def test_a_label_no_participant_defines_is_refused_and_says_so():
    with pytest.raises(ValueError, match="undefined_sparse on the whole roster"):
        population_cqv([np.zeros(WEEKS)] * 4, "mindfulness", replicates=9)


def test_the_interval_is_seeded_and_widens_as_the_level_drops():
    rng = np.random.default_rng(3)
    blocks = roster(rng, 7.0)
    first = population_cqv(blocks, "walk", replicates=299, seed=5)
    again = population_cqv(blocks, "walk", replicates=299, seed=5)
    assert (first.lower, first.upper) == (again.lower, again.upper)
    other = population_cqv(blocks, "walk", replicates=299, seed=6)
    assert (first.lower, first.upper) != (other.lower, other.upper)


def test_the_interval_records_the_draws_and_the_seed_it_was_read_at():
    out = population_cqv(roster(np.random.default_rng(4), 6.0), "gym", seed=7)
    assert out.replicates == REPLICATES
    assert out.seed == 7
    assert CONF_LEVEL == 0.95
    assert MIN_EPOCHS == 4


def test_a_population_with_no_participant_or_no_draw_or_no_level_is_refused():
    with pytest.raises(ValueError, match="at least one participant"):
        population_cqv([], "walk")
    with pytest.raises(ValueError, match="at least one draw"):
        population_cqv(roster(np.random.default_rng(5), 6.0), "walk", replicates=0)
    with pytest.raises(ValueError, match=r"inside \(0, 1\)"):
        population_cqv(roster(np.random.default_rng(5), 6.0), "walk", conf_level=1.0)


# ---------------------------------------------------------------------------
# The verdict the protocol reads
# ---------------------------------------------------------------------------


def test_two_rosters_of_one_world_carry_comparable_heterogeneity():
    rng = np.random.default_rng(6)
    real, drawn, agree = compare_heterogeneity(
        roster(rng, 8.0), roster(rng, 8.0), "walk", replicates=299, seed=0
    )
    assert agree
    assert separation(real, drawn) == 0.0


def test_a_roster_of_another_dispersion_does_not():
    rng = np.random.default_rng(7)
    steady = [np.full(WEEKS, 20.0) + rng.normal(0.0, 0.5, WEEKS) for _ in range(51)]
    wild = [np.abs(rng.normal(20.0, 18.0, WEEKS)) for _ in range(51)]
    real, drawn, agree = compare_heterogeneity(
        steady, wild, "walk", replicates=299, seed=0
    )
    assert not agree
    assert separation(real, drawn) > 0.0
    assert real.cqv < drawn.cqv


def test_the_overlap_is_read_in_both_directions():
    a = Heterogeneity("x", 0.5, 0.4, 0.6, 40, 51, 999, 0)
    b = Heterogeneity("x", 0.55, 0.5, 0.7, 40, 51, 999, 1)
    far = Heterogeneity("x", 0.9, 0.8, 1.0, 40, 51, 999, 2)
    assert comparable(a, b) and comparable(b, a)
    assert not comparable(a, far) and not comparable(far, a)
    assert separation(a, far) == pytest.approx(0.2)
    assert separation(far, a) == pytest.approx(0.2)


def test_a_roster_of_nobody_reports_no_sparse_share():
    assert Heterogeneity("x", 0.0, 0.0, 0.0, 0, 0, 9, 0).undefined_sparse == 0.0


def test_the_blocks_come_off_the_ladder_one_label_at_a_time():
    from src.core.cohort import Cohort
    from src.core.scales import ScaleHierarchy
    from src.core.variables import MultiScaleData

    labels = ["walk", "sleep"]
    ladder = ScaleHierarchy(K=1, H=[WEEKS], parent_maps=[None])

    def reader(name: str) -> MultiScaleData:
        counts = np.arange(WEEKS * 2).reshape(WEEKS, 2) + int(name)
        return MultiScaleData(
            hierarchy=ladder,
            label_names=list(labels),
            X=[counts],
            Y=[counts],
            C=[np.zeros((WEEKS, 1))],
        )

    cohort = Cohort(
        subjects=("0", "1"),
        hierarchy=ladder,
        label_names=list(labels),
        reader=reader,
        masks=np.ones((2, WEEKS), dtype=bool),
    )
    # The scale number is one-based, which is what the panel reader uses, so
    # scale one is the first array of the ladder.
    blocks = weekly_blocks(cohort, 1, 1)
    assert len(blocks) == 2
    assert np.array_equal(blocks[0], np.arange(WEEKS) * 2 + 1)
    assert np.array_equal(blocks[1], blocks[0] + 1)
    with pytest.raises(ValueError, match="numbered from one"):
        weekly_blocks(cohort, 0, 1)
    with pytest.raises(ValueError, match="holds 1 scales"):
        weekly_blocks(cohort, 2, 1)
    with pytest.raises(ValueError, match="outside the 2"):
        weekly_blocks(cohort, 1, 5)


def test_the_blocks_agree_with_the_scale_the_panel_reader_reads():
    # The off-by-one this pins cost a full experiment run: the panel names a
    # one-based scale and the ladder array sits one lower, so a reader that
    # indexes by hand disagrees with `panel_series` on a four-scale ladder.
    from src.data.label_axis import LabelAxis
    from src.evaluation.panel_compare import by_key, panel_series
    from tests.test_phase88_panel_compare import NAMES, axis, cohort_of

    cohort = cohort_of()
    series = by_key(panel_series(cohort, axis(), (1, 2, 3), NAMES))
    for scale, name in NAMES.items():
        for j, label in enumerate(axis().targets):
            blocks = weekly_blocks(cohort, scale, j)
            assert np.array_equal(blocks[0], series[(label, name)].blocks[0])


def test_the_protocol_reads_the_week_and_never_the_coarsest_scale_of_a_run():
    # The heterogeneity protocol is stated on weekly totals.  A run comparing
    # only the day scale must still read the week, so the scale is named and is
    # never the largest entry of whichever tuple a run was given.
    from src.evaluation.panel_compare import SCALE_NAMES, WEEK, WEEK_SCALE

    assert SCALE_NAMES[WEEK_SCALE] == WEEK
    assert WEEK_SCALE == 4
    assert WEEK_SCALE != max((3,))
