from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

try:
    from TimingBot.config import DatasetConfig, FeatureConfig, ModelConfig, SimulationConfig, TrainingConfig
    from TimingBot.data import build_dataset_bundle
    from TimingBot.meta import MetaSelectorConfig, StrategyTrace, run_meta_selector
    from TimingBot.model import TimingHeadOutput, decode_target_fraction
    from TimingBot.production import plan_rebalance
    from TimingBot.simulator import compute_metrics, simulate_fractional_positions_numpy
    from TimingBot.trainer import TimingBotTrainer
except (ImportError, ModuleNotFoundError):
    pytest.skip("Required module TimingBot not available", allow_module_level=True)


def _write_symbol_csv(root: Path, folder: str, symbol: str, closes: list[float]) -> None:
    target = root / folder
    target.mkdir(parents=True, exist_ok=True)
    timestamps = pd.date_range("2026-01-01T00:00:00Z", periods=len(closes), freq="1h")
    rows = []
    for idx, (ts, close) in enumerate(zip(timestamps, closes)):
        rows.append(
            {
                "timestamp": ts,
                "open": close * (0.998 if idx % 2 == 0 else 1.002),
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 1000.0 + idx,
                "trade_count": 10 + idx,
                "vwap": close * 1.0005,
                "symbol": symbol,
            }
        )
    pd.DataFrame(rows).to_csv(target / f"{symbol}.csv", index=False)


def test_dataset_bundle_loads_hourly_symbols(tmp_path: Path) -> None:
    _write_symbol_csv(tmp_path, "stocks", "NVDA", [100.0 + i * 0.5 for i in range(72)])
    _write_symbol_csv(tmp_path, "crypto", "BTCUSD", [200.0 + i * 1.0 for i in range(72)])

    bundle = build_dataset_bundle(
        DatasetConfig(
            data_root=tmp_path,
            symbols=("NVDA", "BTCUSD"),
            sequence_length=8,
            min_history_hours=32,
            validation_days=1,
            allow_short=True,
            feature_config=FeatureConfig(return_windows=(1, 2, 4), trend_window=4, volume_window=4, volatility_window=4),
        )
    )

    assert {item.symbol for item in bundle.train_sets} == {"NVDA", "BTCUSD"}
    assert bundle.train_sets[0].features.ndim == 3
    btc = next(item for item in bundle.train_sets if item.symbol == "BTCUSD")
    assert btc.can_long is True
    assert btc.can_short is False


def test_decode_target_fraction_masks_short_for_long_only_assets() -> None:
    output = TimingHeadOutput(
        direction_logits=torch.tensor([[8.0, 0.0, -8.0]], dtype=torch.float32),
        size_logits=torch.tensor([[0.0, 0.0, 10.0]], dtype=torch.float32),
    )
    decoded = decode_target_fraction(
        output,
        size_buckets=(0.0, 0.5, 1.0),
        can_long=True,
        can_short=False,
        discrete=True,
    )
    assert decoded["target_fraction"].item() == 0.0


def test_numpy_simulator_matches_ramped_bar_close_logic() -> None:
    timestamps = pd.date_range("2026-01-01T00:00:00Z", periods=3, freq="1h")
    result = simulate_fractional_positions_numpy(
        timestamps=timestamps,
        target_fraction=np.array([1.0, 1.0, 0.0], dtype=np.float64),
        execution_price=np.array([100.0, 110.0, 110.0], dtype=np.float64),
        next_execution_price=np.array([110.0, 110.0, 100.0], dtype=np.float64),
        config=SimulationConfig(spread_bps=0.0, slippage_bps=0.0, max_trade_fraction=1.0),
    )
    metrics = compute_metrics(result.step_returns, result.turnover, SimulationConfig(spread_bps=0.0, slippage_bps=0.0, max_trade_fraction=1.0))
    assert result.final_equity == 1.1
    assert result.position_fraction[-1] == 0.0
    assert metrics.max_drawdown == 0.0


def test_plan_rebalance_clips_to_max_trade_fraction() -> None:
    plan = plan_rebalance(
        current_fraction=0.0,
        desired_fraction=1.0,
        portfolio_value=10_000.0,
        price=100.0,
        max_trade_fraction=0.25,
    )
    assert plan.side == "buy"
    assert plan.executed_fraction == 0.25
    assert plan.signed_quantity == 25.0


def test_meta_selector_switches_when_frontier_improves() -> None:
    timestamps = tuple(pd.date_range("2026-01-01T00:00:00Z", periods=4, freq="1h"))
    result = run_meta_selector(
        {
            "flat": StrategyTrace(timestamps=timestamps, step_returns=np.array([0.0, 0.0, 0.0, 0.0]), turnover=np.zeros(4)),
            "trend": StrategyTrace(timestamps=timestamps, step_returns=np.array([0.0, 0.02, 0.02, 0.02]), turnover=np.zeros(4)),
        },
        MetaSelectorConfig(lookback=2, switch_penalty=0.0001, turnover_penalty=0.0, drawdown_penalty=0.0),
    )
    assert result.selected_strategy[-1] == "trend"
    assert result.meta_equity_curve[-1] > 1.03


def test_timingbot_trainer_smoke(tmp_path: Path) -> None:
    _write_symbol_csv(tmp_path, "stocks", "DBX", [50.0 + i * 0.2 for i in range(64)])
    _write_symbol_csv(tmp_path, "crypto", "ETHUSD", [120.0 + (i % 6) * 0.4 + i * 0.1 for i in range(64)])

    bundle = build_dataset_bundle(
        DatasetConfig(
            data_root=tmp_path,
            symbols=("DBX", "ETHUSD"),
            sequence_length=6,
            min_history_hours=24,
            validation_days=0,
            val_fraction=0.2,
            allow_short=True,
            feature_config=FeatureConfig(return_windows=(1, 2, 4), trend_window=4, volume_window=4, volatility_window=4),
        )
    )
    trainer = TimingBotTrainer(
        bundle,
        model_config=ModelConfig(hidden_dim=32, num_layers=2, num_heads=4, dropout=0.0, max_len=32),
        simulation_config=SimulationConfig(spread_bps=0.0, slippage_bps=0.0, max_trade_fraction=1.0),
        training_config=TrainingConfig(
            epochs=1,
            symbol_batch_size=1,
            window_batch_size=32,
            learning_rate=1e-3,
            weight_decay=0.0,
            optimizer_name="adamw",
            use_compile=False,
            use_bf16=False,
            checkpoint_root=tmp_path / "checkpoints",
            run_name="smoke",
        ),
    )
    artifacts = trainer.train()

    assert artifacts.best_checkpoint.exists()
    assert len(artifacts.history) == 2
    assert np.isfinite(artifacts.best_val_score)
