import math

import numpy as np
import pandas as pd
from neuraldailymarketsimulator.simulator import NeuralDailyMarketSimulator
from neuraldailytraining import (
    DailyDataModule,
    DailyDatasetConfig,
    DailyTradingRuntime,
    DailyTrainingConfig,
    NeuralDailyTrainer,
)
from neuraldailytraining.checkpoints import save_checkpoint


def _write_symbol_data(root, forecast_root, symbol: str, days: int = 360) -> None:
    root.mkdir(parents=True, exist_ok=True)
    forecast_root.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("2023-01-01", periods=days, freq="D", tz="UTC")
    base = 100 + np.cumsum(np.random.randn(days)).clip(-5, 5).cumsum()
    open_prices = base + np.random.randn(days) * 0.5
    close_prices = base + np.random.randn(days) * 0.5
    high = np.maximum(open_prices, close_prices) + np.abs(np.random.randn(days))
    low = np.minimum(open_prices, close_prices) - np.abs(np.random.randn(days))
    volume = np.abs(np.random.randn(days) * 1_000) + 1_000
    frame = pd.DataFrame(
        {
            "timestamp": dates,
            "open": open_prices,
            "high": high,
            "low": low,
            "close": close_prices,
            "volume": volume,
            "trade_count": 0,
            "vwap": close_prices,
            "symbol": symbol,
        }
    )
    frame.to_csv(root / f"{symbol}.csv", index=False)
    spread = np.abs(np.random.randn(days)) * 0.5
    forecast = pd.DataFrame(
        {
            "symbol": symbol,
            "timestamp": dates,
            "predicted_close": close_prices + np.random.randn(days) * 0.3,
            "predicted_high": high + np.random.rand(days),
            "predicted_low": low - np.random.rand(days),
            "predicted_close_p10": close_prices - spread,
            "predicted_close_p90": close_prices + spread,
            "forecast_move_pct": np.random.randn(days) * 0.01,
            "forecast_volatility_pct": np.random.rand(days) * 0.02,
        }
    )
    forecast.to_parquet(forecast_root / f"{symbol}.parquet", index=False)


def test_daily_data_module_builds_sequences(tmp_path):
    data_root = tmp_path / "data"
    forecast_root = tmp_path / "forecast"
    _write_symbol_data(data_root, forecast_root, "TEST", days=380)
    cfg = DailyDatasetConfig(
        symbols=("TEST",),
        data_root=data_root,
        forecast_cache_dir=forecast_root,
        sequence_length=32,
        val_fraction=0.2,
        validation_days=40,
    )
    module = DailyDataModule(cfg)
    train_loader = module.train_dataloader(batch_size=4)
    batch = next(iter(train_loader))
    assert batch["features"].shape[-1] == len(module.feature_columns)
    assert batch["close"].shape[-1] == cfg.sequence_length


def test_neural_daily_trainer_overfits_tiny_dataset(tmp_path):
    data_root = tmp_path / "data"
    forecast_root = tmp_path / "forecast"
    for symbol in ("AAA", "BBB"):
        _write_symbol_data(data_root, forecast_root, symbol, days=320)
    dataset_cfg = DailyDatasetConfig(
        symbols=("AAA", "BBB"),
        data_root=data_root,
        forecast_cache_dir=forecast_root,
        sequence_length=32,
        val_fraction=0.25,
        validation_days=32,
        min_history_days=200,
    )
    train_cfg = DailyTrainingConfig(
        epochs=2,
        batch_size=4,
        sequence_length=32,
        learning_rate=1e-3,
        checkpoint_root=tmp_path / "ckpts",
        device="cpu",
        use_compile=False,
        use_amp=False,
        dataset=dataset_cfg,
        risk_threshold=0.05,
        exposure_penalty=10.0,
    )
    train_cfg.log_dir = tmp_path / "logs"
    train_cfg.wandb_project = None
    module = DailyDataModule(dataset_cfg)
    trainer = NeuralDailyTrainer(train_cfg, module)
    artifacts = trainer.train()
    assert artifacts.state_dict
    assert artifacts.history
    checkpoint_path = tmp_path / "runtime_ckpt.pt"
    save_checkpoint(
        checkpoint_path,
        state_dict=artifacts.state_dict,
        normalizer=module.normalizer,
        feature_columns=list(module.feature_columns),
        metrics={},
        config=train_cfg,
    )
    runtime = DailyTradingRuntime(
        checkpoint_path,
        dataset_config=dataset_cfg,
        device="cpu",
    )
    frame = runtime._builder.build("AAA")  # type: ignore[attr-defined]
    for column in module.feature_columns:
        assert column in frame.columns
    plan = runtime.plan_for_symbol("AAA")
    assert plan is not None
    assert plan.trade_amount <= 0.05 + 1e-6
    simulator = NeuralDailyMarketSimulator(runtime, ("AAA", "BBB"), maker_fee=0.0008, initial_cash=1.0)
    sim_results, summary = simulator.run(days=3)
    assert len(sim_results) == 3
    assert not math.isnan(summary["sortino"])
