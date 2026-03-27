import numpy as np
import pandas as pd

from neuraldailytraining import DailyDataModule, DailyDatasetConfig, DailyTrainingConfig, NeuralDailyTrainer



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



def test_max_train_batches_can_span_multiple_epochs(tmp_path):
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
        epochs=5,
        batch_size=4,
        sequence_length=32,
        learning_rate=1e-3,
        checkpoint_root=tmp_path / "ckpts",
        device="cpu",
        use_compile=False,
        use_amp=False,
        wandb_project=None,
        dataset=dataset_cfg,
        max_train_batches=1,
        max_val_batches=1,
        dry_train_steps=2,
    )
    train_cfg.log_dir = tmp_path / "logs"

    module = DailyDataModule(dataset_cfg)
    trainer = NeuralDailyTrainer(train_cfg, module)
    artifacts = trainer.train()

    assert len(artifacts.history) == 2
    assert artifacts.summary is not None
    assert artifacts.summary.best_epoch == 1 or artifacts.summary.best_epoch == 2



def test_trainer_disables_wandb_when_project_missing(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    forecast_root = tmp_path / "forecast"
    _write_symbol_data(data_root, forecast_root, "AAA", days=320)

    class DummyTracker:
        seen_kwargs = None

        def __init__(self, **kwargs):
            type(self).seen_kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def watch(self, *_args, **_kwargs):
            return None

        def log(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr("neuraldailytraining.trainer.WandBoardLogger", DummyTracker)

    dataset_cfg = DailyDatasetConfig(
        symbols=("AAA",),
        data_root=data_root,
        forecast_cache_dir=forecast_root,
        sequence_length=32,
        validation_days=32,
        min_history_days=200,
    )
    train_cfg = DailyTrainingConfig(
        epochs=1,
        batch_size=4,
        sequence_length=32,
        learning_rate=1e-3,
        checkpoint_root=tmp_path / "ckpts",
        device="cpu",
        use_compile=False,
        use_amp=False,
        wandb_project=None,
        dataset=dataset_cfg,
        max_train_batches=1,
        max_val_batches=1,
        dry_train_steps=1,
    )
    train_cfg.log_dir = tmp_path / "logs"

    module = DailyDataModule(dataset_cfg)
    trainer = NeuralDailyTrainer(train_cfg, module)
    trainer.train()

    assert DummyTracker.seen_kwargs is not None
    assert DummyTracker.seen_kwargs["enable_wandb"] is False
