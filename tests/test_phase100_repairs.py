"""
Tests for 39.4.10a items 2, 3, 4 and 6.

Each of these closes one way the 2026-08-23 `exp16` run could be wrong without
saying so: a level refit blind to the chain it was fitted beside, a draw missing
two of its six labels that no check looked at, a self column bought where the
risk set guarantees it is empty, and four smaller items where a number was
reported that nothing stood behind.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.core.typed_graph import COUNT, DURATION, OBSERVED, TypedEdge, TypedGraph
from src.evaluation.heterogeneity import (
    Heterogeneity,
    comparable,
    is_pinned,
    participant_cqv,
    population_cqv,
    separation,
    sparse_gap,
)
from src.evaluation.nonstationarity import (
    TESTED,
    Cell,
    Denominator,
    Inclusion,
    group_tost,
)
from src.evaluation.round_trip import (
    ABSENCE_FLOOR,
    center_chains,
    check_volume,
    stationary_law,
    volume_departures,
)
from src.learning.arm1 import MIN_SELF_SUPPORT, drop_unsupported_self
from src.learning.fine_sem import RegimeChain
from src.model.intensity import CLIP_COLUMNS, FineParams, empty_sources, sample

# ---------------------------------------------------------------------------
# Item 2: the level refit sees the chain


def two_state_chain(effect: np.ndarray, stay: float = 0.8) -> RegimeChain:
    """A homogeneous two-state chain with a declared persistence."""
    leave = 1.0 - stay
    # Stored as log p(next | previous) transposed, so column i is "from i".
    trans = np.log(np.array([[stay, leave], [leave, stay]])).T
    return RegimeChain(
        scale=2,
        n_states=2,
        effect=np.asarray(effect, dtype=float),
        log_init=np.log(np.array([0.5, 0.5])),
        log_trans=trans,
    )


def test_a_symmetric_chain_is_stationary_at_one_half():
    law = stationary_law(two_state_chain(np.zeros((2, 3))))
    np.testing.assert_allclose(law, [0.5, 0.5], atol=1e-9)


def test_an_asymmetric_chain_reaches_its_own_stationary_law():
    # p(stay in 0) = 0.9, p(stay in 1) = 0.6, so the law is (0.8, 0.2) whatever
    # the chain starts at.
    trans = np.log(np.array([[0.9, 0.1], [0.4, 0.6]])).T
    chain = RegimeChain(
        scale=2,
        n_states=2,
        effect=np.zeros((2, 2)),
        log_init=np.log(np.array([0.5, 0.5])),
        log_trans=trans,
    )
    np.testing.assert_allclose(stationary_law(chain), [0.8, 0.2], atol=1e-9)


def test_centering_leaves_the_predictor_identical_at_every_state():
    params = FineParams(n_labels=3)
    params.base[COUNT][:] = [-7.0, -9.0, -11.0]
    effect = np.array([[0.5, -1.0, 2.0], [-0.5, 1.0, -2.0]])
    chain = two_state_chain(effect)

    moved, centered = center_chains(params, [chain])

    for state in range(2):
        np.testing.assert_allclose(
            moved.base[COUNT] + centered[0].effect[state],
            params.base[COUNT] + chain.effect[state],
            atol=1e-12,
        )


def test_a_centered_chain_averages_to_nothing_under_its_own_law():
    params = FineParams(n_labels=2)
    chain = two_state_chain(np.array([[3.0, -4.0], [-1.0, 6.0]]), stay=0.7)
    _, centered = center_chains(params, [chain])
    law = stationary_law(centered[0])
    np.testing.assert_allclose(law @ centered[0].effect, np.zeros(2), atol=1e-10)


def test_centering_moves_the_average_into_the_intercept():
    params = FineParams(n_labels=1)
    params.base[COUNT][:] = [-5.0]
    # A symmetric chain averages its two effects, so the intercept takes 1.0.
    chain = two_state_chain(np.array([[3.0], [-1.0]]))
    moved, _ = center_chains(params, [chain])
    assert moved.base[COUNT][0] == pytest.approx(-4.0)


def test_centering_leaves_the_caller_s_parameters_alone():
    params = FineParams(n_labels=1)
    params.base[COUNT][:] = [-5.0]
    chain = two_state_chain(np.array([[3.0], [-1.0]]))
    center_chains(params, [chain])
    assert params.base[COUNT][0] == pytest.approx(-5.0)
    assert chain.effect[0, 0] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Item 3: the volume tripwire and the clip counter


REAL = {"gym": 506, "mindfulness": 68, "sleep": 1841, "survey_response": 10162}


def test_a_label_drawn_none_is_reported_absent():
    drawn = dict(REAL, survey_response=0)
    found = volume_departures(drawn, REAL)
    assert [d["label"] for d in found] == ["survey_response"]
    assert found[0]["absent"] is True
    assert found[0]["real"] == 10162


def test_a_draw_that_matches_the_panel_reports_nothing():
    assert volume_departures(dict(REAL), REAL) == []


def test_a_volume_inside_the_declared_factor_is_not_a_departure():
    assert volume_departures({**REAL, "gym": 506 * 2}, REAL) == []


def test_a_volume_beyond_the_declared_factor_is_reported_but_not_absent():
    found = volume_departures({**REAL, "gym": 506 * 12}, REAL)
    assert [d["label"] for d in found] == ["gym"]
    assert found[0]["absent"] is False
    assert found[0]["ratio"] == pytest.approx(12.0)


def test_a_label_the_panel_does_not_carry_is_not_a_departure():
    assert volume_departures({"gym": 0}, {"gym": 0}) == []


def test_check_volume_refuses_a_missing_label_and_names_it():
    with pytest.raises(ValueError, match="survey_response"):
        check_volume(dict(REAL, survey_response=0), REAL)


def test_check_volume_returns_a_ratio_departure_rather_than_raising():
    found = check_volume({**REAL, "gym": 506 * 12}, REAL)
    assert [d["label"] for d in found] == ["gym"]


def test_a_rare_label_drawn_none_is_reported_but_not_refused():
    # A six-subject smoke carries one `mindfulness` episode, and drawing none of
    # one is ordinary sampling.  It is still recorded as a departure.
    real = {"mindfulness": 1, "sleep": 200}
    found = check_volume({"mindfulness": 0, "sleep": 200}, real)
    assert [d["label"] for d in found] == ["mindfulness"]
    assert found[0]["absent"] is True


def test_an_absence_at_the_floor_is_refused():
    real = {"mindfulness": ABSENCE_FLOOR}
    with pytest.raises(ValueError, match="mindfulness"):
        check_volume({"mindfulness": 0}, real)


def _clip_counts(base_count, base_duration=None, slots=20):
    """The clip counter after drawing `slots` slots for one label at that level."""
    graph = TypedGraph(label_names=["a"], calendar_names=[])
    params = FineParams(n_labels=1)
    params.base[COUNT][:] = [base_count]
    if base_duration is not None:
        params.base[DURATION][:] = [base_duration]
    ancestors = [np.arange(slots, dtype=np.int64)]
    sources = empty_sources(ancestors, [np.zeros((slots, 0))], 1)
    clips = np.zeros((1, len(CLIP_COLUMNS)), dtype=np.int64)
    sample(params, graph, sources, np.random.default_rng(0), None, clips)
    return dict(zip(CLIP_COLUMNS, (int(v) for v in clips[0])))


def test_an_extreme_predictor_is_counted_as_a_clip():
    # Every scorable slot was decided by the cap, not by the fit.
    assert _clip_counts(-2.5e11)["start_low"] == 20


def test_an_ordinary_predictor_is_never_clipped():
    assert _clip_counts(-3.0) == {end: 0 for end in CLIP_COLUMNS}


def test_the_two_ends_of_the_start_term_are_counted_apart():
    # A label at +30 fires in every free slot and its episode occupies the ones
    # after it, so the count is the number of slots the draw reached.
    high = _clip_counts(2.5e11, base_duration=0.0)
    assert high["start_high"] > 0 and high["start_low"] == 0
    assert _clip_counts(-2.5e11)["start_high"] == 0


def test_the_mark_is_counted_apart_from_the_start():
    counts = _clip_counts(2.5e11, base_duration=-2.5e11)
    assert counts["mark_low"] > 0
    assert counts["start_high"] == counts["mark_low"]
    assert counts["mark_high"] == 0


# ---------------------------------------------------------------------------
# Item 4: the unsupported self candidates


def edge(src_label: int, lag: int = 1, scale: int = 1) -> TypedEdge:
    return TypedEdge(OBSERVED, scale, src_label, COUNT, lag)


def stats_with(diagonals, edges, intercept=80640.0):
    """A stand-in carrying only what `drop_unsupported_self` reads."""
    return SimpleNamespace(
        candidates={0: SimpleNamespace(diagonal=np.asarray(diagonals, dtype=float))},
        designs={0: SimpleNamespace(A=np.diag([intercept]))},
    )


def test_a_self_candidate_with_no_support_is_refused():
    edges = [edge(0)]
    kept, dropped = drop_unsupported_self({0: edges}, stats_with([0.0], edges))
    assert kept[0] == []
    assert dropped[0] == edges


def test_a_self_candidate_with_almost_no_support_is_refused():
    # The gym case: 10 rows of 3,492,956, which is 2.9e-6 of the information.
    edges = [edge(0)]
    support = 2.9e-6 * 80640.0
    kept, _ = drop_unsupported_self({0: edges}, stats_with([support], edges))
    assert kept[0] == []


def test_a_self_candidate_with_real_support_is_kept():
    # The survey_response case: 988 rows of 80,640, which is 1.2e-2.
    edges = [edge(0)]
    support = 1.2e-2 * 80640.0
    kept, dropped = drop_unsupported_self({0: edges}, stats_with([support], edges))
    assert kept[0] == edges
    assert dropped == {}


def test_the_finest_scale_stays_open_as_a_cross_label_source():
    # An empty column that is not the target's own is left alone; only a self
    # candidate is tested, which is what keeps the anchor relation reachable.
    edges = [edge(0), edge(3)]
    kept, dropped = drop_unsupported_self({0: edges}, stats_with([0.0, 0.0], edges))
    assert kept[0] == [edge(3)]
    assert dropped[0] == [edge(0)]


def test_a_target_with_no_measured_candidates_keeps_its_pool():
    edges = [edge(0)]
    empty = SimpleNamespace(candidates={}, designs={})
    kept, dropped = drop_unsupported_self({0: edges}, empty)
    assert kept[0] == edges
    assert dropped == {}


def test_the_floor_sits_between_the_two_measured_cases():
    assert 2.9e-6 < MIN_SELF_SUPPORT < 1.2e-2


# ---------------------------------------------------------------------------
# Item 6: the TOST denominator, the permutation null, the heterogeneity protocol


def test_pooling_replicates_no_longer_shrinks_the_standard_error():
    rng = np.random.default_rng(0)
    real = rng.normal(0.0, 1.0, 51)
    # Five draws over the same 51 units, pooled into one group of 255.
    synthetic = np.concatenate([rng.normal(0.4, 1.0, 51) for _ in range(5)])

    pooled = group_tost(real, synthetic, tolerance=0.5)
    clustered = group_tost(real, synthetic, tolerance=0.5, synthetic_units=51)
    # Counting the panel rather than the draw widens the interval, so the cell
    # is harder to pass, which is the direction the error ran.
    assert clustered.p_value > pooled.p_value


def test_claiming_more_units_than_values_is_refused():
    with pytest.raises(ValueError, match="independent units"):
        group_tost(np.zeros(10), np.zeros(10), tolerance=1.0, synthetic_units=11)


def test_a_cell_carries_its_permutation_p_into_its_row():
    cell = Cell(
        form="A12",
        series="survey_response",
        scale="day",
        magnitude=0.41,
        spread=0.1,
        p_value=0.02,
        inclusion=Inclusion(status=TESTED, clause="power"),
        denominator=Denominator(
            episodes=10162, carriers=51, epochs_per_participant=56, cycles=8.0
        ),
        effect=-2.3,
        permutation_p=0.004,
    )
    assert cell.as_row()["permutation_p"] == 0.004


def test_a_participant_whose_lower_quartile_is_zero_is_pinned():
    assert is_pinned([0.0, 0.0, 3.0, 4.0, 5.0])
    assert not is_pinned([1.0, 2.0, 3.0, 4.0])
    # Both quartiles zero is undefined, not pinned.
    assert not is_pinned([0.0, 0.0, 0.0, 0.0])


def test_an_all_zero_participant_has_no_cqv():
    assert participant_cqv([0.0, 0.0, 0.0, 0.0]) is None


def test_a_pinned_participant_is_counted_and_left_out_of_the_median():
    interior = [[1.0, 2.0, 3.0, 4.0]] * 4
    pinned = [[0.0, 0.0, 3.0, 4.0]] * 6
    out = population_cqv(interior + pinned, "nap", replicates=50, seed=0)
    assert out.n_pinned == 6
    assert out.n_defined == 10
    assert out.pinned_share == pytest.approx(0.6)
    # The median is the interior one, not the 1.0 the pinned majority carries.
    assert out.cqv < 1.0


def test_a_label_pinned_everywhere_is_refused_rather_than_reported_as_one():
    with pytest.raises(ValueError, match="pinned"):
        population_cqv([[0.0, 0.0, 3.0, 4.0]] * 5, "gym", replicates=10)


def hetero(sparse: float, low: float = 0.3, high: float = 0.7) -> Heterogeneity:
    return Heterogeneity(
        label="walk",
        cqv=0.5,
        lower=low,
        upper=high,
        n_defined=int(round((1.0 - sparse) * 51)),
        n_participants=51,
        replicates=999,
        seed=0,
    )


def test_two_panels_defining_the_measure_for_the_same_share_are_comparable():
    assert comparable(hetero(0.08), hetero(0.10))


def test_a_panel_defining_the_measure_for_far_fewer_is_not_comparable():
    # 0.078 real against 0.788 drawn, on intervals that overlap perfectly.
    real, drawn = hetero(0.078), hetero(0.788)
    assert separation(real, drawn) == 0.0
    assert sparse_gap(real, drawn) > 0.15
    assert not comparable(real, drawn)


def test_separation_is_never_negative_when_the_intervals_overlap():
    assert separation(hetero(0.1, 0.3, 0.7), hetero(0.9, 0.2, 0.8)) == 0.0
