import pandas as pd
import numpy as np

from hourlycryptotraining import DatasetConfig, HourlyCryptoDataModule


def _build_price_frame(hours: int = 96) -> pd.DataFrame:
    base_ts = pd.Timestamp("2024-01-01", tz="UTC")
    timestamps = [base_ts + pd.Timedelta(hours=i) for i in range(hours)]
    prices = np.linspace(10.0, 12.0, hours)
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": prices,
            "high": prices + 0.1,
            "low": prices - 0.1,
            "close": prices,
            "volume": np.linspace(1_000, 5_000, hours),
            "symbol": "LINKUSD",
        }
    )
    return frame


def _build_forecast_frame(price_frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": price_frame["timestamp"],
            "symbol": "LINKUSD",
            "predicted_close_p50": price_frame["close"] + 0.05,
            "predicted_high_p50": price_frame["high"] + 0.05,
            "predicted_low_p50": price_frame["low"] - 0.05,
        }
    )


def test_data_module_produces_sequences(tmp_path):
    price = _build_price_frame(120)
    forecast = _build_forecast_frame(price)
    data_root = tmp_path
    cache_dir = tmp_path / "forecast_cache"
    data_root.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    price[["timestamp", "open", "high", "low", "close", "volume", "symbol"]].to_csv(
        data_root / "LINKUSD.csv",
        index=False,
    )
    forecast.to_parquet(cache_dir / "LINKUSD.parquet", index=False)
    config = DatasetConfig(
        symbol="LINKUSD",
        data_root=data_root,
        forecast_cache_dir=cache_dir,
        sequence_length=12,
        val_fraction=0.2,
        min_history_hours=60,
        feature_columns=["return_1h", "chronos_close_delta", "chronos_high_delta", "chronos_low_delta"],
    )
    module = HourlyCryptoDataModule(config)
    sample = module.train_dataset[0]
    assert sample["features"].shape == (12, len(module.feature_columns))
    assert sample["high"].shape == (12,)
    assert module.train_dataset.__len__() > 0
    assert module.val_dataset.__len__() > 0
