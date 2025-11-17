import numpy as np
import pandas as pd

from hourlycryptotraining import DatasetConfig, HourlyCryptoDataModule
from hourlycryptotraining.data import MultiSymbolDataModule


def _build_price_frame(symbol: str = "LINKUSD", hours: int = 96) -> pd.DataFrame:
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
            "symbol": symbol.upper(),
        }
    )
    return frame


def _build_forecast_frame(price_frame: pd.DataFrame, symbol: str = "LINKUSD") -> pd.DataFrame:
    symbol = symbol.upper()
    return pd.DataFrame(
        {
            "timestamp": price_frame["timestamp"],
            "symbol": symbol,
            "predicted_close_p50": price_frame["close"] + 0.05,
            "predicted_high_p50": price_frame["high"] + 0.05,
            "predicted_low_p50": price_frame["low"] - 0.05,
        }
    )


def _write_symbol_payload(data_root, cache_dir, symbol: str, hours: int = 120) -> None:
    price = _build_price_frame(symbol, hours)
    forecast = _build_forecast_frame(price, symbol)
    data_root.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    price[["timestamp", "open", "high", "low", "close", "volume", "symbol"]].to_csv(
        data_root / f"{symbol.upper()}.csv",
        index=False,
    )
    forecast.to_parquet(cache_dir / f"{symbol.upper()}.parquet", index=False)


def test_data_module_produces_sequences(tmp_path):
    price = _build_price_frame("LINKUSD", 120)
    forecast = _build_forecast_frame(price, "LINKUSD")
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
        refresh_hours=0,
        validation_days=5,
    )
    module = HourlyCryptoDataModule(config)
    sample = module.train_dataset[0]
    assert sample["features"].shape == (12, len(module.feature_columns))
    assert sample["high"].shape == (12,)
    assert module.train_dataset.__len__() > 0
    assert module.val_dataset.__len__() > 0


def test_multisymbol_module_exposes_hourly_interface(tmp_path):
    data_root = tmp_path / "data"
    cache_dir = tmp_path / "forecast"
    for symbol in ("LINKUSD", "BTCUSD", "ETHUSD"):
        _write_symbol_payload(data_root, cache_dir, symbol)
    config = DatasetConfig(
        symbol="LINKUSD",
        data_root=data_root,
        forecast_cache_dir=cache_dir,
        sequence_length=16,
        val_fraction=0.2,
        min_history_hours=60,
        feature_columns=["return_1h", "chronos_close_delta", "chronos_high_delta", "chronos_low_delta"],
        refresh_hours=0,
        validation_days=5,
    )
    module = MultiSymbolDataModule(["BTCUSD", "ETHUSD"], config)
    assert module.target_symbol == "LINKUSD"
    assert module.symbols[0] == "LINKUSD"
    assert set(module.modules) == {"LINKUSD", "BTCUSD", "ETHUSD"}
    assert module.frame.equals(module.modules["LINKUSD"].frame)
    assert module.normalizer is module.modules["LINKUSD"].normalizer
    assert module.feature_columns == module.modules["LINKUSD"].feature_columns
    loader = module.train_dataloader(batch_size=4)
    batch = next(iter(loader))
    assert batch["features"].shape[1] == config.sequence_length
    assert batch["features"].shape[2] == len(module.feature_columns)
    expected_len = sum(len(mod.train_dataset) for mod in module.modules.values())
    assert len(module.train_dataset) == expected_len


def test_multisymbol_module_accepts_target_in_list(tmp_path):
    data_root = tmp_path / "data"
    cache_dir = tmp_path / "forecast"
    for symbol in ("LINKUSD", "BTCUSD"):
        _write_symbol_payload(data_root, cache_dir, symbol)
    config = DatasetConfig(
        symbol="LINKUSD",
        data_root=data_root,
        forecast_cache_dir=cache_dir,
        sequence_length=16,
        val_fraction=0.2,
        min_history_hours=60,
        feature_columns=["return_1h", "chronos_close_delta", "chronos_high_delta", "chronos_low_delta"],
        refresh_hours=0,
        validation_days=5,
    )
    module = MultiSymbolDataModule(["BTCUSD", "LINKUSD"], config)
    assert module.symbols[0] == "LINKUSD"
    assert module.frame.iloc[0]["symbol"] == "LINKUSD"
    assert module.val_dataset is module.modules["LINKUSD"].val_dataset
