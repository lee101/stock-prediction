from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pufferlib_market.allocation_refiner as module
import pytest
import torch
from pufferlib_market.allocation_refiner import (
    FrozenTrace,
    RefinerConfig,
    compute_refiner_objective,
    fit_refiner,
    predict_allocations,
    replay_hourly_actions_with_allocations,
    simulate_daily_actions_with_allocations,
)
from pufferlib_market.hourly_replay import FEATURES_PER_SYM, HourlyMarket, MktdData
from pufferlib_market.realism import PRODUCTION_SHORT_BORROW_APR


def _dummy_trace(*, signed_returns: list[float], target_ids: list[int]) -> FrozenTrace:
    steps = len(signed_returns)
    zeros = torch.zeros((steps, 2), dtype=torch.float32)
    return FrozenTrace(
        observations=torch.zeros((steps, FEATURES_PER_SYM + 6), dtype=torch.float32),
        latents=zeros.clone(),
        selected_features=torch.zeros((steps, FEATURES_PER_SYM), dtype=torch.float32),
        portfolio_state=torch.zeros((steps, 5), dtype=torch.float32),
        confidence=torch.ones((steps,), dtype=torch.float32) * 0.5,
        logit_gap=torch.ones((steps,), dtype=torch.float32),
        selected_position=torch.zeros((steps,), dtype=torch.float32),
        directions=torch.tensor([1.0 if tid >= 0 else 0.0 for tid in target_ids], dtype=torch.float32),
        action_active=torch.tensor([1.0 if tid >= 0 else 0.0 for tid in target_ids], dtype=torch.float32),
        signed_returns=torch.tensor(signed_returns, dtype=torch.float32),
        short_mask=torch.zeros((steps,), dtype=torch.float32),
        target_ids=torch.tensor(target_ids, dtype=torch.long),
        actions=np.asarray([1 if tid >= 0 else 0 for tid in target_ids], dtype=np.int32),
        max_steps=steps,
        num_symbols=1,
    )


def test_compute_refiner_objective_penalizes_drawdown_excess() -> None:
    trace = _dummy_trace(signed_returns=[0.03, -0.12, 0.02], target_ids=[0, 0, 0])
    config = RefinerConfig(
        max_leverage=5.0,
        drawdown_limit=0.25,
        drawdown_penalty=2.0,
        drawdown_excess_penalty=12.0,
        sortino_weight=0.0,
        smoothness_penalty=0.0,
        leverage_penalty=0.0,
        turnover_penalty=0.0,
    )

    aggressive = torch.ones((3,), dtype=torch.float32)
    conservative = torch.full((3,), 0.2, dtype=torch.float32)

    _, aggressive_summary = compute_refiner_objective(aggressive, trace, config)
    _, conservative_summary = compute_refiner_objective(conservative, trace, config)

    assert aggressive_summary.max_drawdown > conservative_summary.max_drawdown
    assert aggressive_summary.drawdown_excess > 0.0
    assert aggressive_summary.objective < conservative_summary.objective


def test_refiner_config_defaults_to_production_borrow_apr() -> None:
    assert RefinerConfig().short_borrow_apr == pytest.approx(PRODUCTION_SHORT_BORROW_APR)


def _single_symbol_data(prices: list[float]) -> MktdData:
    timesteps = len(prices)
    features = np.zeros((timesteps, 1, FEATURES_PER_SYM), dtype=np.float32)
    ohlcv = np.zeros((timesteps, 1, 5), dtype=np.float32)
    for i, price in enumerate(prices):
        ohlcv[i, 0, 0] = price
        ohlcv[i, 0, 1] = price
        ohlcv[i, 0, 2] = price
        ohlcv[i, 0, 3] = price
        ohlcv[i, 0, 4] = 1.0
    tradable = np.ones((timesteps, 1), dtype=np.uint8)
    return MktdData(
        version=2,
        symbols=["BTCUSD"],
        features=features,
        prices=ohlcv,
        tradable=tradable,
    )


def test_simulate_daily_actions_with_allocations_scales_leverage() -> None:
    data = _single_symbol_data([100.0, 110.0])

    half = simulate_daily_actions_with_allocations(
        data=data,
        actions=np.asarray([1], dtype=np.int32),
        allocation_pcts=np.asarray([0.2], dtype=np.float64),
        max_steps=1,
        fee_rate=0.0,
        max_leverage=5.0,
        periods_per_year=365.0,
        fill_buffer_bps=0.0,
    )
    full = simulate_daily_actions_with_allocations(
        data=data,
        actions=np.asarray([1], dtype=np.int32),
        allocation_pcts=np.asarray([1.0], dtype=np.float64),
        max_steps=1,
        fee_rate=0.0,
        max_leverage=5.0,
        periods_per_year=365.0,
        fill_buffer_bps=0.0,
    )

    assert half.total_return == pytest.approx(0.10, rel=0, abs=1e-6)
    assert full.total_return == pytest.approx(0.50, rel=0, abs=1e-6)
    assert full.total_return > half.total_return


def test_replay_hourly_actions_with_allocations_matches_daily_scale() -> None:
    data = _single_symbol_data([100.0, 110.0])
    index = pd.date_range("2025-01-01 00:00", "2025-01-02 23:00", freq="h", tz="UTC")
    close = np.concatenate([np.full((24,), 100.0), np.full((24,), 110.0)])
    market = HourlyMarket(
        index=index,
        close={"BTCUSD": close.astype(np.float64)},
        tradable={"BTCUSD": np.ones_like(close, dtype=bool)},
    )

    result = replay_hourly_actions_with_allocations(
        data=data,
        actions=np.asarray([1], dtype=np.int32),
        allocation_pcts=np.asarray([0.2], dtype=np.float64),
        market=market,
        start_date="2025-01-01",
        end_date="2025-01-02",
        max_steps=1,
        fee_rate=0.0,
        max_leverage=5.0,
        periods_per_year=8760.0,
    )

    assert result.total_return == pytest.approx(0.10, rel=0, abs=1e-6)
    assert result.num_orders >= 2


def test_fit_refiner_improves_positive_trace_and_respects_inactive_actions() -> None:
    trace = _dummy_trace(signed_returns=[0.05, 0.04, 0.0], target_ids=[0, 0, -1])
    config = RefinerConfig(
        max_leverage=2.0,
        epochs=80,
        lr=1e-2,
        hidden_size=32,
        sortino_weight=0.0,
        drawdown_penalty=0.0,
        drawdown_excess_penalty=0.0,
        smoothness_penalty=0.0,
        leverage_penalty=0.0,
        turnover_penalty=0.0,
        init_allocation_pct=0.2,
        seed=7,
    )

    baseline_alloc = torch.full((trace.max_steps,), config.init_allocation_pct, dtype=torch.float32)
    _, baseline_summary = compute_refiner_objective(baseline_alloc, trace, config)

    model, metrics = fit_refiner(
        train_trace=trace,
        val_trace=trace,
        config=config,
        device=torch.device("cpu"),
    )
    predicted = predict_allocations(model, trace, device=torch.device("cpu"))
    _, fitted_summary = compute_refiner_objective(torch.tensor(predicted, dtype=torch.float32), trace, config)

    assert metrics["epoch"] >= 1
    assert fitted_summary.total_return > baseline_summary.total_return
    assert predicted[0] > config.init_allocation_pct
    assert predicted[1] > config.init_allocation_pct
    assert predicted[2] == pytest.approx(0.0)


def test_main_writes_model_and_reports_atomically(monkeypatch, tmp_path: Path, capsys) -> None:
    trace = _dummy_trace(signed_returns=[0.02, -0.01], target_ids=[0, 0])
    model = module.ResidualExposureHead(
        input_dim=trace.observations.shape[1],
        hidden_size=8,
        init_allocation_pct=0.2,
    )
    torch_writes: list[tuple[object, Path]] = []
    json_writes: list[tuple[Path, dict[str, object], dict[str, object]]] = []

    monkeypatch.setattr(module, "collect_frozen_trace", lambda **kwargs: trace)
    monkeypatch.setattr(module, "fit_refiner", lambda **kwargs: (model, {"epoch": 3, "best_val_objective": 1.25}))
    monkeypatch.setattr(
        module,
        "_evaluate_refined_trace",
        lambda **kwargs: {
            "daily": {"total_return": 0.05},
            "hourly_replay": {"sortino": 1.5},
            "allocation_summary": {"mean_alloc_pct": 0.4},
        },
    )
    monkeypatch.setattr(module, "save_torch_atomic", lambda payload, path: torch_writes.append((payload, Path(path))))
    monkeypatch.setattr(
        module,
        "write_json_atomic",
        lambda path, payload, **kwargs: json_writes.append((Path(path), payload, kwargs)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "allocation_refiner.py",
            "--checkpoint",
            str(tmp_path / "policy.pt"),
            "--train-data",
            str(tmp_path / "train.bin"),
            "--val-data",
            str(tmp_path / "val.bin"),
            "--val-start-date",
            "2026-01-01",
            "--val-end-date",
            "2026-01-10",
            "--output-dir",
            str(tmp_path / "out"),
            "--cpu",
        ],
    )

    module.main()

    captured = capsys.readouterr()
    assert '"val_report"' in captured.out
    assert [path.name for _, path in torch_writes] == ["refiner.pt"]
    assert torch_writes[0][0]["checkpoint"] == str(tmp_path / "policy.pt")
    assert torch_writes[0][0]["config"]["short_borrow_apr"] == pytest.approx(PRODUCTION_SHORT_BORROW_APR)
    assert [path.name for path, _, _ in json_writes] == ["train_metrics.json", "val_report.json"]
    assert all(kwargs == {"sort_keys": True} for _, _, kwargs in json_writes)
