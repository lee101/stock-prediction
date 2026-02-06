from __future__ import annotations

from pathlib import Path

import pandas as pd


def _write_hourly_history_csv(path: Path, symbol: str, timestamps: pd.DatetimeIndex) -> None:
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


def test_build_forecast_bundle_respects_start_end(tmp_path: Path) -> None:
    from binanceneural.forecasts import build_forecast_bundle

    symbol = "TEST"
    timestamps = pd.date_range("2026-01-01", periods=10, freq="h", tz="UTC")

    data_root = tmp_path / "data"
    cache_root = tmp_path / "cache"
    data_root.mkdir(parents=True, exist_ok=True)
    _write_hourly_history_csv(data_root / f"{symbol}.csv", symbol, timestamps)

    # Provide cached forecasts so cache_only avoids generating anything.
    _write_forecast_parquet(cache_root / "h1" / f"{symbol}.parquet", symbol, timestamps)
    _write_forecast_parquet(cache_root / "h24" / f"{symbol}.parquet", symbol, timestamps)

    start = timestamps[2]
    end = timestamps[7]

    bundle = build_forecast_bundle(
        symbol=symbol,
        data_root=data_root,
        cache_root=cache_root,
        horizons=(1, 24),
        context_hours=1,
        quantile_levels=(0.1, 0.5, 0.9),
        batch_size=128,
        cache_only=True,
        start=start,
        end=end,
    )

    # build_forecast_bundle uses (start, end] semantics.
    assert pd.to_datetime(bundle["timestamp"], utc=True).min() == timestamps[3]
    assert pd.to_datetime(bundle["timestamp"], utc=True).max() == timestamps[7]


def test_alpaca_hourly_data_module_windows_forecasts_to_lookback(tmp_path: Path, monkeypatch) -> None:
    import newnanoalpacahourlyexp.data as data_mod
    from newnanoalpacahourlyexp.config import DatasetConfig

    # Avoid expensive feature engineering in this unit test; we're validating windowing.
    def _identity_build_feature_frame(frame: pd.DataFrame, *, config: DatasetConfig) -> pd.DataFrame:
        work = frame.copy()
        # BinanceExp1Dataset expects this column for scaling trade distances.
        work["reference_close"] = work["close"]
        return work

    monkeypatch.setattr(data_mod, "build_feature_frame", _identity_build_feature_frame)

    symbol = "TEST"
    timestamps = pd.date_range("2026-01-01", periods=50, freq="h", tz="UTC")
    _write_hourly_history_csv(tmp_path / f"{symbol}.csv", symbol, timestamps)
    _write_forecast_parquet(tmp_path / "forecast_cache" / "h1" / f"{symbol}.parquet", symbol, timestamps)

    cfg = DatasetConfig(
        symbol=symbol,
        data_root=tmp_path,
        forecast_cache_root=tmp_path / "forecast_cache",
        forecast_horizons=(1,),
        cache_only=True,
        sequence_length=4,
        min_history_hours=5,
        max_feature_lookback_hours=10,
        validation_days=0,
        feature_columns=(
            "close",
            "predicted_close_p50_h1",
            "predicted_high_p50_h1",
            "predicted_low_p50_h1",
        ),
    )

    module = data_mod.AlpacaHourlyDataModule(cfg)
    assert len(module.frame) == 10
    assert pd.to_datetime(module.frame["timestamp"], utc=True).min() == timestamps[-10]
