from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def _write_hourly_history_csv(path: Path, symbol: str, timestamps: pd.DatetimeIndex) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1_000.0,
            "symbol": symbol,
        }
    )
    frame.to_csv(path, index=False)


def _write_forecast_parquet(path: Path, symbol: str, timestamps: pd.DatetimeIndex) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": symbol,
            "issued_at": timestamps,
            "predicted_close_p50": 100.0,
            "predicted_close_p10": 99.0,
            "predicted_close_p90": 101.0,
            "predicted_high_p50": 101.0,
            "predicted_low_p50": 99.0,
        }
    )
    frame.to_parquet(path, index=False)


def test_min_history_hours_converted_for_stocks(tmp_path: Path, monkeypatch) -> None:
    import newnanoalpacahourlyexp.data as data_mod
    from newnanoalpacahourlyexp.config import DatasetConfig

    def _identity_build_feature_frame(frame: pd.DataFrame, *, config: DatasetConfig) -> pd.DataFrame:
        work = frame.copy()
        work["reference_close"] = work["close"]
        return work

    monkeypatch.setattr(data_mod, "build_feature_frame", _identity_build_feature_frame)

    symbol = "TEST"  # treated as a stock-like symbol by is_crypto_symbol()
    timestamps = pd.date_range("2026-01-01", periods=250, freq="h", tz="UTC")
    _write_hourly_history_csv(tmp_path / "stocks" / f"{symbol}.csv", symbol, timestamps)
    _write_forecast_parquet(tmp_path / "cache" / "h1" / f"{symbol}.parquet", symbol, timestamps)

    cfg = DatasetConfig(
        symbol=symbol,
        data_root=tmp_path / "stocks",
        forecast_cache_root=tmp_path / "cache",
        forecast_horizons=(1,),
        cache_only=True,
        sequence_length=32,
        # 30d * 24 = 720 bars would reject stock histories; we convert to ~210 bars.
        min_history_hours=24 * 30,
        validation_days=0,
        max_feature_lookback_hours=0,
        feature_columns=(
            "close",
            "predicted_close_p50_h1",
            "predicted_high_p50_h1",
            "predicted_low_p50_h1",
        ),
    )

    module = data_mod.AlpacaHourlyDataModule(cfg)
    assert len(module.frame) == 250


def test_min_history_hours_not_converted_for_crypto(tmp_path: Path, monkeypatch) -> None:
    import newnanoalpacahourlyexp.data as data_mod
    from newnanoalpacahourlyexp.config import DatasetConfig

    def _identity_build_feature_frame(frame: pd.DataFrame, *, config: DatasetConfig) -> pd.DataFrame:
        work = frame.copy()
        work["reference_close"] = work["close"]
        return work

    monkeypatch.setattr(data_mod, "build_feature_frame", _identity_build_feature_frame)

    symbol = "BTCUSD"  # treated as crypto-like by is_crypto_symbol()
    timestamps = pd.date_range("2026-01-01", periods=250, freq="h", tz="UTC")
    _write_hourly_history_csv(tmp_path / "crypto" / f"{symbol}.csv", symbol, timestamps)
    _write_forecast_parquet(tmp_path / "cache" / "h1" / f"{symbol}.parquet", symbol, timestamps)

    cfg = DatasetConfig(
        symbol=symbol,
        data_root=tmp_path / "crypto",
        forecast_cache_root=tmp_path / "cache",
        forecast_horizons=(1,),
        cache_only=True,
        sequence_length=32,
        min_history_hours=24 * 30,
        validation_days=0,
        max_feature_lookback_hours=0,
        feature_columns=(
            "close",
            "predicted_close_p50_h1",
            "predicted_high_p50_h1",
            "predicted_low_p50_h1",
        ),
    )

    with pytest.raises(ValueError, match="Insufficient hourly history"):
        data_mod.AlpacaHourlyDataModule(cfg)


def test_hourly_data_module_normalizes_file_symbol_to_requested_symbol(tmp_path: Path, monkeypatch) -> None:
    import newnanoalpacahourlyexp.data as data_mod
    from newnanoalpacahourlyexp.config import DatasetConfig

    def _identity_build_feature_frame(frame: pd.DataFrame, *, config: DatasetConfig) -> pd.DataFrame:
        work = frame.copy()
        work["reference_close"] = work["close"]
        return work

    monkeypatch.setattr(data_mod, "build_feature_frame", _identity_build_feature_frame)

    requested_symbol = "BTCUSD"
    timestamps = pd.date_range("2026-01-01", periods=120, freq="h", tz="UTC")
    _write_hourly_history_csv(tmp_path / "crypto" / f"{requested_symbol}.csv", "BTCUSDT", timestamps)
    _write_forecast_parquet(tmp_path / "cache" / "h1" / f"{requested_symbol}.parquet", requested_symbol, timestamps)

    cfg = DatasetConfig(
        symbol=requested_symbol,
        data_root=tmp_path / "crypto",
        forecast_cache_root=tmp_path / "cache",
        forecast_horizons=(1,),
        cache_only=True,
        sequence_length=32,
        min_history_hours=48,
        validation_days=0,
        max_feature_lookback_hours=0,
        feature_columns=(
            "close",
            "predicted_close_p50_h1",
            "predicted_high_p50_h1",
            "predicted_low_p50_h1",
        ),
    )

    module = data_mod.AlpacaHourlyDataModule(cfg)
    assert set(module.frame["symbol"].unique()) == {requested_symbol}
