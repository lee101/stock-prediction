import pandas as pd
from pathlib import Path

from neuraldailytraining.config import DailyDatasetConfig
from neuraldailytraining.data import SymbolFrameBuilder


def _price_csv(path: Path, symbol: str):
    dates = pd.date_range("2025-01-01", periods=3, freq="D", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": dates,
            "open": [10, 10.5, 10.2],
            "high": [10.8, 10.7, 10.4],
            "low": [9.8, 10.0, 10.0],
            "close": [10.5, 10.2, 10.3],
            "volume": [100, 120, 130],
        }
    )
    frame.to_csv(path / f"{symbol}.csv", index=False)


def _forecast_parquet(path: Path, symbol: str):
    # Missing the last day
    dates = pd.date_range("2025-01-01", periods=2, freq="D", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": dates,
            "predicted_close": [10.4, 10.1],
            "predicted_high": [10.6, 10.3],
            "predicted_low": [10.0, 9.9],
        }
    )
    frame.to_parquet(path / f"{symbol}.parquet", index=False)


def test_forecast_missing_rows_filled_and_cached(tmp_path: Path):
    data_root = tmp_path / "train"
    fc_root = tmp_path / "fc"
    data_root.mkdir()
    fc_root.mkdir()
    symbol = "TEST"
    _price_csv(data_root, symbol)
    _forecast_parquet(fc_root, symbol)

    cfg = DailyDatasetConfig(
        data_root=data_root,
        forecast_cache_dir=fc_root,
        require_forecasts=True,
        forecast_fill_strategy="persistence",
        forecast_cache_writeback=True,
    )
    builder = SymbolFrameBuilder(cfg, feature_columns=("close", "predicted_close", "predicted_high", "predicted_low"))
    frame = builder.build(symbol)

    # Last day should be filled with persistence from close
    assert frame["predicted_close"].iloc[-1] == frame["close"].iloc[-1]

    # Writeback should exist and include all three dates
    cache_path = fc_root / f"{symbol}.parquet"
    cached = pd.read_parquet(cache_path)
    assert len(cached) == 3


def test_forecast_missing_rows_fail_when_required(tmp_path: Path):
    data_root = tmp_path / "train"
    fc_root = tmp_path / "fc"
    data_root.mkdir()
    fc_root.mkdir()
    symbol = "FAIL"
    _price_csv(data_root, symbol)
    _forecast_parquet(fc_root, symbol)

    cfg = DailyDatasetConfig(
        data_root=data_root,
        forecast_cache_dir=fc_root,
        require_forecasts=True,
        forecast_fill_strategy="fail",
        forecast_cache_writeback=False,
    )
    builder = SymbolFrameBuilder(cfg, feature_columns=("close", "predicted_close"))
    try:
        builder.build(symbol)
    except ValueError as exc:
        assert "Missing forecasts" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing forecasts")
