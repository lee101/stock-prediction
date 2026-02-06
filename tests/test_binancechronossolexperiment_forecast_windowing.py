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
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def test_chronos_sol_data_module_windows_forecasts_when_max_history_days_set(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import binancechronossolexperiment.data as data_mod
    from binancechronossolexperiment.data import ChronosSolDataModule, SplitConfig

    symbol = "TEST"
    timestamps = pd.date_range("2026-01-01", periods=100, freq="h", tz="UTC")
    _write_hourly_history_csv(tmp_path / f"{symbol}.csv", symbol, timestamps)

    calls: dict[str, pd.Timestamp | None] = {"start": None, "end": None}

    def _fake_build_feature_frame(frame: pd.DataFrame, *, horizons, max_lookback: int) -> pd.DataFrame:
        work = frame.copy()
        work["reference_close"] = work["close"]
        return work

    def _fake_build_forecast_bundle(
        *,
        symbol: str,
        data_root: Path,
        cache_root: Path,
        horizons,
        context_hours: int,
        quantile_levels,
        batch_size: int,
        model_id: str,
        device_map: str = "cuda",
        preaugmentation_dirs=None,
        cache_only: bool = False,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
        wrapper_factory=None,
    ) -> pd.DataFrame:
        calls["start"] = start
        calls["end"] = end
        window = timestamps
        if start is not None:
            window = window[window > start]
        if end is not None:
            window = window[window <= end]
        return pd.DataFrame(
            {
                "timestamp": window,
                "symbol": symbol,
                "predicted_close_p50_h1": 100.0,
                "predicted_high_p50_h1": 101.0,
                "predicted_low_p50_h1": 99.0,
            }
        )

    monkeypatch.setattr(data_mod, "build_feature_frame", _fake_build_feature_frame)
    monkeypatch.setattr(data_mod, "build_forecast_bundle", _fake_build_forecast_bundle)

    split_config = SplitConfig(val_days=1, test_days=1)
    module = ChronosSolDataModule(
        symbol=symbol,
        data_root=tmp_path,
        forecast_cache_root=tmp_path / "cache",
        forecast_horizons=(1,),
        context_hours=16,
        quantile_levels=(0.1, 0.5, 0.9),
        batch_size=8,
        model_id="dummy",
        sequence_length=4,
        split_config=split_config,
        max_feature_lookback_hours=10,
        min_history_hours=5,
        max_history_days=3,
        feature_columns=(
            "close",
            "predicted_close_p50_h1",
            "predicted_high_p50_h1",
            "predicted_low_p50_h1",
        ),
        cache_only=True,
    )

    expected_end = timestamps.max()
    expected_start = expected_end - pd.Timedelta(hours=float(3 * 24 + 10))

    assert calls["end"] == expected_end
    assert calls["start"] == expected_start
    assert pd.to_datetime(module.full_frame["timestamp"], utc=True).min() > expected_start
