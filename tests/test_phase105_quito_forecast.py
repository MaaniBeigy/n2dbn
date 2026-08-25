"""
tests/test_phase105_quito_forecast.py
=====================================
The QuitoBench forecasting arm: its loader, its scores, its forecast, its deep rows.

The load-bearing check is `test_the_forecast_sums_across_every_scale_pair`.
Everything else fixes an interface; that one is the property the arm exists to
demonstrate, and it reads a forecast walked down a fitted ladder.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.graph import MultiScaleDBNGraph
from src.core.scales import ScaleHierarchy
from src.data.initialize import initialize_from_data
from src.data.quito import (
    COUNT_CHANNELS,
    HOUR_PANEL,
    MIN_PANEL,
    PANELS,
    RATE_CHANNEL,
    ZERO_LEVEL,
    Panel,
    aggregate_up,
    collapse,
    context_ladder,
    four_channel_ids,
    read_panel,
    seasonal_context,
    split_at_cutoff,
    trend_context,
    whole_blocks_before,
    window_to_multiscale,
)
from src.evaluation.coherence import _MAX_ETA
from src.evaluation.deep_baselines import torch_available, windows
from src.evaluation.forecast_metrics import (
    coherence_error,
    mae,
    mse,
    relative_coherence,
    standardize,
    train_scaler,
)
from src.learning.evidence import evidence_chain
from src.model.forecast import (
    allocation_weights,
    coarse_mixture_mean,
    forecast_coarse,
    forecast_ladder,
    lag_block,
    lag_offset,
    roll_factor,
    roll_regime,
    transition_matrix,
)

#: A ladder small enough to fit in a test and deep enough to have an interior.
TOY = Panel(
    name="toy",
    filename="toy.parquet",
    length=6 * 24 * 5,
    branching=(6, 24),
    season=(6, 24, 1),
    harmonics=(1, 2, 0),
    trend=(False, False, True),
    block=144,
)


def toy_counts(days: int = 5, seed: int = 0) -> np.ndarray:
    """Counts on the toy ladder, with a daily shape and four channels."""
    rng = np.random.default_rng(seed)
    t = np.arange(days * TOY.block)
    rate = 20.0 + 12.0 * np.sin(2.0 * np.pi * t / TOY.block)
    return rng.poisson(rate[:, None] * np.array([1.0, 0.6, 0.3, 0.1])[None, :])


def fitted(days: int = 5, seed: int = 0):
    """A window, an initialized model and its posterior on the toy ladder."""
    counts = toy_counts(days, seed)
    data = window_to_multiscale(counts, TOY, offset=0, total_finest=counts.shape[0])
    params = initialize_from_data(
        data,
        M_k=[2] * data.K,
        r_k=[1, 0, 0],
        max_lag=[2] * data.K,
        d_ctx=[data.get_C(k).shape[1] for k in range(1, data.K + 1)],
        seed=seed,
    )
    graph = MultiScaleDBNGraph(
        hierarchy=data.hierarchy, label_names=list(COUNT_CHANNELS)
    )
    return data, params, graph, evidence_chain(data, params, graph).summaries


# ---------------------------------------------------------------------------
# The loader
# ---------------------------------------------------------------------------


def test_the_two_published_panels_are_named_and_carry_nesting_chains():
    assert set(PANELS) == {"min", "hour"}
    assert PANELS["min"] is MIN_PANEL and PANELS["hour"] is HOUR_PANEL
    assert MIN_PANEL.K == 3 and HOUR_PANEL.K == 3
    # One coarsest epoch holds exactly the product of the branching factors.
    for panel in PANELS.values():
        assert panel.block == int(np.prod(panel.branching))
    assert RATE_CHANNEL not in COUNT_CHANNELS


def test_a_missing_cached_split_is_named_rather_than_guessed(tmp_path):
    with pytest.raises(FileNotFoundError, match="absent"):
        read_panel(MIN_PANEL, root=str(tmp_path))


def test_a_seasonal_column_repeats_on_its_period_and_zero_harmonics_is_empty():
    columns = seasonal_context(48, 24, 1)
    assert columns.shape == (48, 2)
    assert np.allclose(columns[:24], columns[24:])
    assert seasonal_context(5, 24, 0).shape == (5, 0)


def test_a_trend_column_spans_minus_one_to_one_and_continues_past_the_window():
    column = trend_context(4, 0, 4)
    assert column[0, 0] == pytest.approx(-1.0)
    assert column[-1, 0] == pytest.approx(1.0)
    assert trend_context(1, 0, 1)[0, 0] == pytest.approx(-1.0)
    assert trend_context(2, 4, 4)[0, 0] > 1.0


def test_the_context_ladder_spends_harmonics_and_the_trend_where_it_is_told():
    ladder = context_ladder(
        n_epochs=[144, 24, 1],
        offsets=[0, 0, 0],
        totals=[144, 24, 1],
        season=TOY.season,
        harmonics=TOY.harmonics,
        trend=TOY.trend,
    )
    # One harmonic pair, two pairs, and the trend column alone.
    assert [c.shape[1] for c in ladder] == [2, 4, 1]


def test_aggregating_up_and_collapsing_agree_on_the_block_sums():
    hierarchy = ScaleHierarchy.from_uniform_branching(144, list(TOY.branching))
    finest = np.arange(144 * 4, dtype=np.int64).reshape(144, 4)
    levels = aggregate_up(finest, hierarchy)
    assert [a.shape for a in levels] == [(144, 4), (24, 4), (1, 4)]
    assert np.array_equal(levels[1], collapse(finest, 6))
    assert np.array_equal(levels[2], collapse(finest, 144))


def test_a_window_becomes_multiscale_data_whose_scales_reconcile():
    data = window_to_multiscale(toy_counts(3), TOY, total_finest=3 * TOY.block)
    assert data.K == 3
    assert data.check_reconciliation() == {"RecErr_X": 0.0, "RecErr_Y": 0.0}
    assert not data.get_Y(1).any()
    assert data.label_names == list(COUNT_CHANNELS)


def test_whole_blocks_takes_the_tail_and_refuses_what_it_cannot_supply():
    assert whole_blocks_before(1000, 144) == (1000 - 6 * 144, 1000)
    assert whole_blocks_before(1000, 144, 2) == (1000 - 2 * 144, 1000)
    with pytest.raises(ValueError, match="whole blocks"):
        whole_blocks_before(1000, 144, 99)
    with pytest.raises(ValueError, match="whole blocks"):
        whole_blocks_before(10, 144)


def test_the_cutoff_is_read_off_the_shared_timestamp_grid():
    import pandas as pd

    grid = pd.date_range("2023-07-26", periods=96, freq="h").to_numpy()
    assert split_at_cutoff(grid, "2023-07-28 00:00:00") == 48


def test_the_panel_keeps_only_series_whose_four_channels_all_fire():
    series = {
        1: np.array([[1, 1, 1, 1], [0, 2, 3, 4]], dtype=np.int64),
        2: np.array([[1, 0, 1, 1], [2, 0, 3, 4]], dtype=np.int64),
    }
    assert four_channel_ids(series) == [1]


def test_the_zero_level_sits_above_the_published_sentinel():
    assert 1e-5 < ZERO_LEVEL < 1.0


# ---------------------------------------------------------------------------
# The scores
# ---------------------------------------------------------------------------


def test_the_scaler_comes_from_training_alone_and_floors_a_flat_channel():
    train = np.array([[1.0, 5.0], [3.0, 5.0]])
    mean, sd = train_scaler(train)
    assert mean.tolist() == [2.0, 5.0]
    assert sd[0] == pytest.approx(1.0)
    assert sd[1] > 0.0
    scaled = standardize(np.array([[2.0, 5.0]]), mean, sd)
    assert scaled[0, 0] == pytest.approx(0.0)


def test_the_two_errors_are_the_mean_absolute_and_mean_squared_gap():
    actual = np.array([[2.0, 4.0]])
    predicted = np.array([[1.0, 6.0]])
    assert mae(actual, predicted) == pytest.approx(1.5)
    assert mse(actual, predicted) == pytest.approx(2.5)


def test_the_coherence_column_is_zero_on_sums_and_positive_on_anything_else():
    hierarchy = ScaleHierarchy.from_uniform_branching(144, list(TOY.branching))
    finest = np.arange(144 * 4, dtype=float).reshape(144, 4)
    levels = [a.astype(float) for a in aggregate_up(finest.astype(np.int64), hierarchy)]
    assert coherence_error(levels, hierarchy)["coh"] == pytest.approx(0.0)
    assert relative_coherence(levels, hierarchy) == pytest.approx(0.0)

    broken = [levels[0], levels[1] + 1.0, levels[2]]
    assert coherence_error(broken, hierarchy)["coh"] > 0.0
    assert relative_coherence(broken, hierarchy) > 0.0

    flat = ScaleHierarchy(K=1, H=[4], parent_maps=[None])
    assert coherence_error([np.zeros((4, 1))], flat) == {"coh": 0.0}
    assert relative_coherence([np.zeros((4, 1))], flat) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# The mean forecast
# ---------------------------------------------------------------------------


def test_a_transition_matrix_is_a_distribution_over_the_state_it_enters():
    _, params, _, _ = fitted()
    plain = transition_matrix(params[2].regime, np.zeros(params[2].regime.d_ctx))
    assert np.allclose(plain.sum(axis=0), 1.0)

    fine = params[0].regime
    coupled = transition_matrix(
        fine, np.zeros(fine.d_ctx), np.ones(fine.parent_M) / fine.parent_M
    )
    assert np.allclose(coupled.sum(axis=0), 1.0)


def test_the_rolled_regime_stays_a_distribution_with_and_without_a_parent():
    _, params, _, summaries = fitted()
    ctx = np.zeros((4, params[2].regime.d_ctx))
    rolled = roll_regime(params[2].regime, np.asarray(summaries[2].gamma)[-1], ctx)
    assert rolled.shape == (4, params[2].regime.M)
    assert np.allclose(rolled.sum(axis=1), 1.0)

    fine_ctx = np.zeros((4, params[0].regime.d_ctx))
    parent = np.tile(np.ones(params[1].regime.M) / params[1].regime.M, (4, 1))
    coupled = roll_regime(
        params[0].regime,
        np.asarray(summaries[0].gamma)[-1],
        fine_ctx,
        parent_gamma=parent,
    )
    assert np.allclose(coupled.sum(axis=1), 1.0)


def test_a_scale_with_no_factor_rolls_an_empty_one_and_one_with_it_stays_finite():
    _, params, _, summaries = fitted()
    empty = roll_factor(
        params[2].factor,
        np.zeros(0),
        np.ones((3, params[2].regime.M)) / params[2].regime.M,
        np.zeros((3, params[2].regime.d_ctx)),
    )
    assert empty.shape == (3, 0)

    carried = roll_factor(
        params[0].factor,
        np.asarray(summaries[0].m_H)[-1],
        np.tile(np.ones(params[0].regime.M) / params[0].regime.M, (3, 1)),
        np.zeros((3, params[0].regime.d_ctx)),
    )
    assert carried.shape == (3, params[0].factor.r)
    assert np.all(np.isfinite(carried))


def test_the_regime_mixture_averages_the_per_state_means_and_refuses_an_allocation():
    _, params, _, _ = fitted()
    coarse = params[2]
    gamma = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    ctx = np.zeros((3, coarse.regime.d_ctx))
    out = coarse_mixture_mean(coarse, gamma, np.zeros((3, 0)), ctx)
    assert out.shape == (3, coarse.L)
    # The mixed row is the average of the two one-hot rows, by construction.
    assert np.allclose(out[2], 0.5 * (out[0] + out[1]))

    with pytest.raises(ValueError, match="allocation"):
        coarse_mixture_mean(params[0], gamma, np.zeros((3, 0)), ctx)


def test_the_lagged_block_reads_the_record_then_its_own_earlier_forecasts():
    _, params, _, _ = fitted()
    coarse = params[2]
    coarse.coarse_count.betaLag = {(0, 1): np.full(coarse.L, 0.5)}
    history = np.array([[3, 3, 3, 3], [7, 7, 7, 7]], dtype=np.int64)

    from_record = lag_offset(coarse, history, 0, np.zeros((0, coarse.L)))
    assert from_record[0] == pytest.approx(0.5 * np.log1p(7.0))

    pending = np.full((1, coarse.L), 100.0)
    from_forecast = lag_offset(coarse, history, 1, pending)
    assert from_forecast[0] == pytest.approx(0.5 * np.log1p(100.0))

    # A lag reaching past the start of the record contributes nothing.
    coarse.coarse_count.betaLag = {(0, 9): np.full(coarse.L, 0.5)}
    assert lag_offset(coarse, history, 0, np.zeros((0, coarse.L)))[0] == 0.0


def test_the_fine_lagged_block_reads_history_and_pending_the_same_way():
    history = np.array([[2, 2, 2, 2], [4, 4, 4, 4]], dtype=np.int64)
    beta = {(0, 1): np.full(4, 0.25)}
    without = lag_block(beta, history, None, (2, 4))
    assert without[0, 0] == pytest.approx(0.25 * np.log1p(4.0))
    # With nothing pending, an epoch whose lag lands inside the window reads zero.
    assert without[1, 0] == pytest.approx(0.0)

    pending = np.full((2, 4), 8.0)
    with_pending = lag_block(beta, history, pending, (2, 4))
    assert with_pending[1, 0] == pytest.approx(0.25 * np.log1p(8.0))


def test_the_coarsest_forecast_is_finite_and_non_negative():
    data, params, _, summaries = fitted()
    ctx = data.get_C(3)[:1]
    gamma = roll_regime(params[2].regime, np.asarray(summaries[2].gamma)[-1], ctx)
    factor = roll_factor(params[2].factor, np.zeros(0), gamma, ctx)
    mean = forecast_coarse(params[2], gamma, factor, ctx, data.get_X(3))
    assert mean.shape == (1, params[2].L)
    assert np.all(np.isfinite(mean)) and np.all(mean >= 0.0)


def test_allocation_weights_of_one_block_sum_to_one_with_and_without_lags():
    data, params, _, _ = fitted()
    blocks = np.asarray(data.hierarchy.parent_maps[0])
    gamma = np.tile(np.ones(params[0].regime.M) / params[0].regime.M, (blocks.size, 1))
    factor = np.zeros((blocks.size, 0))
    ctx = data.get_C(1)

    plain = allocation_weights(params[0], gamma, factor, ctx, blocks)
    params[0].fine_count.betaLag = {(0, 1): np.full(params[0].L, 0.3)}
    lagged = allocation_weights(
        params[0],
        gamma,
        factor,
        ctx,
        blocks,
        data.get_X(1),
        data.get_X(1)[: blocks.size],
    )
    for weights in (plain, lagged):
        for parent in np.unique(blocks):
            rows = np.flatnonzero(blocks == parent)
            assert np.allclose(weights[rows].sum(axis=0), 1.0)


def test_a_predictor_driven_past_the_clip_still_returns_weights_that_sum():
    data, params, _, _ = fitted()
    params[0].fine_count.beta0 = np.full(params[0].L, 10.0 * _MAX_ETA)
    blocks = np.asarray(data.hierarchy.parent_maps[0])
    weights = allocation_weights(
        params[0],
        np.tile(np.ones(params[0].regime.M) / params[0].regime.M, (blocks.size, 1)),
        np.zeros((blocks.size, 0)),
        data.get_C(1),
        blocks,
    )
    assert np.all(np.isfinite(weights))
    for parent in np.unique(blocks):
        rows = np.flatnonzero(blocks == parent)
        assert weights[rows][:, 0].sum() == pytest.approx(1.0)


def test_the_forecast_sums_across_every_scale_pair():
    data, params, _, summaries = fitted()
    window = window_to_multiscale(
        np.zeros((TOY.block, 4), dtype=np.int64), TOY, offset=data.hierarchy.H[0]
    )
    forecasts = forecast_ladder(
        params_list=params,
        origins_gamma=[np.asarray(s.gamma)[-1] for s in summaries],
        origins_factor=[
            (
                np.asarray(s.m_H)[-1]
                if s.m_H is not None and np.size(s.m_H)
                else np.zeros(0)
            )
            for s in summaries
        ],
        ctx_future=[window.get_C(k) for k in range(1, window.K + 1)],
        hierarchy=window.hierarchy,
        history_coarsest=data.get_X(3),
    )
    assert [f.shape for f in forecasts] == [(144, 4), (24, 4), (1, 4)]
    assert coherence_error(forecasts, window.hierarchy)["coh"] == pytest.approx(
        0.0, abs=1e-6
    )
    assert forecasts[0].sum() == pytest.approx(forecasts[2].sum())


def test_the_ladder_reads_a_fine_history_when_one_is_supplied():
    data, params, _, summaries = fitted()
    params[0].fine_count.betaLag = {(0, 1): np.full(params[0].L, 0.2)}
    window = window_to_multiscale(
        np.zeros((TOY.block, 4), dtype=np.int64), TOY, offset=data.hierarchy.H[0]
    )
    forecasts = forecast_ladder(
        params_list=params,
        origins_gamma=[np.asarray(s.gamma)[-1] for s in summaries],
        origins_factor=[
            (
                np.asarray(s.m_H)[-1]
                if s.m_H is not None and np.size(s.m_H)
                else np.zeros(0)
            )
            for s in summaries
        ],
        ctx_future=[window.get_C(k) for k in range(1, window.K + 1)],
        hierarchy=window.hierarchy,
        history_coarsest=data.get_X(3),
        history_fine=[data.get_X(k) for k in (1, 2, 3)],
    )
    assert coherence_error(forecasts, window.hierarchy)["coh"] == pytest.approx(
        0.0, abs=1e-6
    )


# ---------------------------------------------------------------------------
# The deep rows
# ---------------------------------------------------------------------------


def test_windows_cut_a_series_into_context_and_horizon_pairs():
    series = np.arange(40, dtype=float).reshape(20, 2)
    ctx, tgt = windows(series, context=5, horizon=3, stride=2)
    assert ctx.shape[1:] == (5, 2) and tgt.shape[1:] == (3, 2)
    assert ctx.shape[0] == tgt.shape[0]
    # The target of a window starts where its context ends.
    assert np.allclose(tgt[0, 0], series[5])


@pytest.mark.skipif(not torch_available(), reason="torch is absent")
def test_both_deep_models_map_a_context_to_the_horizon_asked_for():
    from src.evaluation.deep_baselines import DLinear, PatchTST, predict, train_model

    rng = np.random.default_rng(0)
    series = rng.normal(size=(120, 2)).astype(np.float32)
    ctx, tgt = windows(series, context=32, horizon=8, stride=4)
    for Model in (DLinear, PatchTST):
        model = train_model(Model(32, 8), ctx, tgt, epochs=1, batch=16)
        out = predict(model, ctx[:2])
        assert out.shape == (2, 8, 2)
        assert np.all(np.isfinite(out))


@pytest.mark.skipif(not torch_available(), reason="torch is absent")
def test_a_context_shorter_than_a_patch_is_still_cut_into_one():
    from src.evaluation.deep_baselines import PatchTST, predict

    model = PatchTST(context=4, horizon=2, patch=16, stride=8).eval()
    out = predict(model, np.zeros((1, 4, 3), dtype=np.float32))
    assert out.shape == (1, 2, 3)


def test_a_written_panel_reads_back_and_a_ragged_one_is_refused(tmp_path):
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    grid = pd.date_range("2023-07-27", periods=TOY.block, freq="10min")
    frames = []
    for item in (1, 2):
        frame = pd.DataFrame({"item_id": item, "date_time": grid})
        for j, channel in enumerate(COUNT_CHANNELS):
            frame[channel] = np.where(
                np.arange(TOY.block) % (j + 2) == 0, 1e-5, float(item + j)
            )
        frames.append(frame)
    path = tmp_path / TOY.filename
    pq.write_table(pa.Table.from_pandas(pd.concat(frames)), path)

    read = Panel(
        name=TOY.name,
        filename=TOY.filename,
        length=TOY.block,
        branching=TOY.branching,
        season=TOY.season,
        harmonics=TOY.harmonics,
        trend=TOY.trend,
        block=TOY.block,
    )
    stamps, series = read_panel(read, root=str(tmp_path))
    assert stamps.size == TOY.block
    assert sorted(series) == [1, 2]
    assert series[1].shape == (TOY.block, 4)
    # The published sentinel is read as a zero count.
    assert (series[1] == 0).any()
    assert four_channel_ids(series) == [1, 2]

    ragged = pd.concat([frames[0], frames[1].iloc[: TOY.block // 2]])
    short = tmp_path / "short.parquet"
    pq.write_table(pa.Table.from_pandas(ragged), short)
    read.filename = "short.parquet"
    with pytest.raises(ValueError, match="one grid shared"):
        read_panel(read, root=str(tmp_path))


def test_a_coarsest_scale_carrying_a_factor_reads_it_in_the_mixture():
    counts = toy_counts(5)
    data = window_to_multiscale(counts, TOY, offset=0, total_finest=counts.shape[0])
    params = initialize_from_data(
        data,
        M_k=[2] * data.K,
        r_k=[0, 0, 1],
        max_lag=[2] * data.K,
        d_ctx=[data.get_C(k).shape[1] for k in range(1, data.K + 1)],
        seed=0,
    )
    coarse = params[2]
    gamma = np.array([[1.0, 0.0]])
    ctx = data.get_C(3)[:1]
    without = coarse_mixture_mean(coarse, gamma, np.zeros((1, 0)), ctx)
    with_factor = coarse_mixture_mean(coarse, gamma, np.ones((1, coarse.factor.r)), ctx)
    assert with_factor.shape == (1, coarse.L)
    assert np.all(np.isfinite(with_factor))
    # A loaded factor moves the mean, so the block above it is live.
    assert not np.allclose(without, with_factor)
