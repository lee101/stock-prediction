from __future__ import annotations

from pathlib import Path

import pandas as pd

from binanceexp1 import joint_chronos_forecast_cache as cache_mod


class _DummyBatch:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.quantile_frames = {0.1: frame, 0.5: frame, 0.9: frame}


class _RecordingWrapper:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def predict_ohlc_batch(
        self,
        contexts: list[pd.DataFrame],
        *,
        symbols: list[str] | None = None,
        prediction_length: int,
        context_length: int | None = None,
        known_future_covariates: list[str] | None = None,
        future_covariates: list[pd.DataFrame | None] | None = None,
        batch_size: int | None = None,
        predict_kwargs: dict | None = None,
    ) -> list[_DummyBatch]:
        self.calls.append(
            {
                "symbols": symbols,
                "prediction_length": prediction_length,
                "context_length": context_length,
                "known_future_covariates": known_future_covariates,
                "future_covariates": future_covariates,
                "batch_size": batch_size,
                "predict_kwargs": predict_kwargs,
            }
        )
        batches: list[_DummyBatch] = []
        for context in contexts:
            last_ts = pd.to_datetime(context["timestamp"].iloc[-1], utc=True)
            future = [last_ts + pd.Timedelta(hours=i) for i in range(1, int(prediction_length) + 1)]
            frame = pd.DataFrame(
                {
                    "close": [101.0] * int(prediction_length),
                    "high": [102.0] * int(prediction_length),
                    "low": [100.0] * int(prediction_length),
                },
                index=pd.DatetimeIndex(future, name="timestamp"),
            )
            batches.append(_DummyBatch(frame))
        return batches


def _write_history(path: Path, symbol: str, rows: int = 36) -> None:
    timestamps = pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 10.0,
            "symbol": symbol,
        }
    )
    frame.to_csv(path, index=False)


def test_build_joint_forecast_cache_writes_symbol_parquets(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    cache_root = tmp_path / "cache"
    data_root.mkdir()
    _write_history(data_root / "BTCUSD.csv", "BTCUSD")
    _write_history(data_root / "ETHUSD.csv", "ETHUSD")

    wrapper = _RecordingWrapper()

    class _WrapperFactory:
        @staticmethod
        def from_pretrained(*args, **kwargs):  # type: ignore[no-untyped-def]
            del args, kwargs
            return wrapper

    monkeypatch.setattr(
        cache_mod,
        "resolve_chronos2_params",
        lambda symbol, frequency="hourly": {  # type: ignore[no-untyped-def]
            "model_id": "amazon/chronos-2",
            "device_map": "cpu",
            "quantile_levels": (0.1, 0.5, 0.9),
        },
    )
    monkeypatch.setattr(cache_mod, "Chronos2OHLCWrapper", _WrapperFactory)

    summary = cache_mod.build_joint_forecast_cache(
        symbols=["BTCUSD", "ETHUSD"],
        data_root=data_root,
        cache_root=cache_root,
        horizons=(1,),
        context_hours=16,
        batch_size=8,
        use_cross_learning=True,
        use_time_covariates=True,
        force_rebuild=True,
    )

    assert summary["h1"]["BTCUSD"] > 0
    assert summary["h1"]["ETHUSD"] > 0

    btc_cache = pd.read_parquet(cache_root / "h1" / "BTCUSD.parquet")
    eth_cache = pd.read_parquet(cache_root / "h1" / "ETHUSD.parquet")
    assert not btc_cache.empty
    assert not eth_cache.empty
    assert "predicted_close_p50" in btc_cache.columns
    assert wrapper.calls
    first_call = wrapper.calls[0]
    assert first_call["predict_kwargs"] == {"predict_batches_jointly": True}
    assert first_call["known_future_covariates"] == list(cache_mod._TIME_COVARIATE_COLUMNS)


def test_build_joint_forecast_cache_limits_history_window(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    cache_root = tmp_path / "cache"
    data_root.mkdir()
    _write_history(data_root / "BTCUSD.csv", "BTCUSD", rows=120)

    wrapper = _RecordingWrapper()

    class _WrapperFactory:
        @staticmethod
        def from_pretrained(*args, **kwargs):  # type: ignore[no-untyped-def]
            del args, kwargs
            return wrapper

    monkeypatch.setattr(
        cache_mod,
        "resolve_chronos2_params",
        lambda symbol, frequency="hourly": {  # type: ignore[no-untyped-def]
            "model_id": "amazon/chronos-2",
            "device_map": "cpu",
            "quantile_levels": (0.1, 0.5, 0.9),
        },
    )
    monkeypatch.setattr(cache_mod, "Chronos2OHLCWrapper", _WrapperFactory)

    summary = cache_mod.build_joint_forecast_cache(
        symbols=["BTCUSD"],
        data_root=data_root,
        cache_root=cache_root,
        horizons=(1,),
        context_hours=16,
        batch_size=8,
        max_history_hours=48,
        use_cross_learning=False,
        use_time_covariates=False,
        force_rebuild=True,
    )

    timestamps = pd.date_range("2026-01-01", periods=120, freq="h", tz="UTC")
    btc_cache = pd.read_parquet(cache_root / "h1" / "BTCUSD.parquet")

    assert summary["h1"]["BTCUSD"] == 32
    assert pd.to_datetime(btc_cache["timestamp"], utc=True).min() == timestamps[-32]


def test_build_joint_forecast_cache_can_apply_narrative_overlay(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    cache_root = tmp_path / "cache"
    summary_root = tmp_path / "summary_cache"
    data_root.mkdir()
    _write_history(data_root / "BTCUSD.csv", "BTCUSD", rows=72)

    wrapper = _RecordingWrapper()

    class _WrapperFactory:
        @staticmethod
        def from_pretrained(*args, **kwargs):  # type: ignore[no-untyped-def]
            del args, kwargs
            return wrapper

    monkeypatch.setattr(
        cache_mod,
        "resolve_chronos2_params",
        lambda symbol, frequency="hourly": {  # type: ignore[no-untyped-def]
            "model_id": "amazon/chronos-2",
            "device_map": "cpu",
            "quantile_levels": (0.1, 0.5, 0.9),
        },
    )
    monkeypatch.setattr(cache_mod, "Chronos2OHLCWrapper", _WrapperFactory)

    summary = cache_mod.build_joint_forecast_cache(
        symbols=["BTCUSD"],
        data_root=data_root,
        cache_root=cache_root,
        horizons=(1,),
        context_hours=24,
        batch_size=8,
        use_cross_learning=False,
        use_time_covariates=False,
        force_rebuild=True,
        narrative_backend="heuristic",
        narrative_summary_cache_root=summary_root,
        narrative_context_hours=24 * 3,
    )

    out = pd.read_parquet(cache_root / "h1" / "BTCUSD.parquet")
    assert summary["h1"]["BTCUSD"] > 0
    assert "narrative_summary" in out.columns
    assert "base_predicted_close_p50" in out.columns
    assert (summary_root / "h1" / "BTCUSD.parquet").exists()
