from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from binanceneural.config import ForecastConfig
from binanceneural.forecasts import ChronosForecastManager


@dataclass
class _DummyBatch:
    quantile_frames: dict[float, pd.DataFrame]


def _make_history(symbol: str, rows: int = 80) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")
    close = pd.Series(range(rows), dtype="float32") + 100.0
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1.0,
            "symbol": symbol,
        }
    )


def _batch_from_context(context: pd.DataFrame, prediction_length: int, base: float) -> _DummyBatch:
    last_ts = pd.to_datetime(context["timestamp"].iloc[-1], utc=True)
    future = [last_ts + pd.Timedelta(hours=i) for i in range(1, int(prediction_length) + 1)]
    idx = pd.DatetimeIndex(future, name="timestamp")
    frame = pd.DataFrame(
        {
            "close": [float(base)] * int(prediction_length),
            "high": [float(base) + 0.5] * int(prediction_length),
            "low": [float(base) - 0.5] * int(prediction_length),
        },
        index=idx,
    )
    return _DummyBatch(quantile_frames={0.1: frame, 0.5: frame, 0.9: frame})


class _ModeWrapper:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def predict_ohlc_batch(
        self,
        contexts: list[pd.DataFrame],
        *,
        symbols: list[str] | None = None,
        prediction_length: int,
        context_length: int | None = None,
        batch_size: int | None = None,
        predict_kwargs: dict | None = None,
        **_: object,
    ) -> list[_DummyBatch]:
        del symbols, context_length, batch_size, predict_kwargs
        self.calls.append("batch")
        return [_batch_from_context(ctx, prediction_length, base=101.0) for ctx in contexts]

    def predict_ohlc_joint(
        self,
        contexts: list[pd.DataFrame],
        *,
        symbols: list[str],
        prediction_length: int,
        context_length: int | None = None,
        quantile_levels: list[float] | None = None,
        predict_batches_jointly: bool = False,
        batch_size: int | None = None,
        **_: object,
    ) -> list[_DummyBatch]:
        del symbols, context_length, quantile_levels, predict_batches_jointly, batch_size
        self.calls.append("joint")
        return [_batch_from_context(ctx, prediction_length, base=202.0) for ctx in contexts]

    def predict_ohlc_multivariate(
        self,
        context_df: pd.DataFrame,
        *,
        symbol: str | None = None,
        prediction_length: int = 1,
        context_length: int | None = None,
        quantile_levels: list[float] | None = None,
        batch_size: int | None = None,
    ) -> _DummyBatch:
        del symbol, context_length, quantile_levels, batch_size
        self.calls.append("multivariate")
        return _batch_from_context(context_df, prediction_length, base=303.0)


class _BatchOnlyWrapper:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def predict_ohlc_batch(
        self,
        contexts: list[pd.DataFrame],
        *,
        symbols: list[str] | None = None,
        prediction_length: int,
        context_length: int | None = None,
        batch_size: int | None = None,
        predict_kwargs: dict | None = None,
        **_: object,
    ) -> list[_DummyBatch]:
        del symbols, context_length, batch_size, predict_kwargs
        self.calls.append("batch")
        return [_batch_from_context(ctx, prediction_length, base=111.0) for ctx in contexts]

    def predict_ohlc_joint(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        self.calls.append("joint")
        raise RuntimeError("joint unavailable")


def _build_cfg(tmp_path: Path, symbol: str = "TEST") -> ForecastConfig:
    return ForecastConfig(
        symbol=symbol,
        data_root=tmp_path,
        context_hours=24,
        prediction_horizon_hours=1,
        quantile_levels=(0.1, 0.5, 0.9),
        batch_size=8,
        cache_dir=tmp_path / "cache",
    )


def test_manager_uses_joint_multivariate_when_enabled(tmp_path: Path) -> None:
    history = _make_history("TEST", rows=80)
    wrapper = _ModeWrapper()
    manager = ChronosForecastManager(_build_cfg(tmp_path), wrapper_factory=lambda: wrapper)
    manager._use_multivariate = True
    manager._use_cross_learning = True

    out = manager._generate_forecast_chunk(history, [40, 41])

    assert out is not None
    assert not out.empty
    assert set(out["predicted_close_p50"].tolist()) == {202.0}
    assert wrapper.calls.count("joint") == 1
    assert "batch" not in wrapper.calls


def test_manager_uses_per_symbol_multivariate_without_cross_learning(tmp_path: Path) -> None:
    history = _make_history("TEST", rows=80)
    wrapper = _ModeWrapper()
    manager = ChronosForecastManager(_build_cfg(tmp_path), wrapper_factory=lambda: wrapper)
    manager._use_multivariate = True
    manager._use_cross_learning = False

    out = manager._generate_forecast_chunk(history, [40, 41])

    assert out is not None
    assert not out.empty
    assert set(out["predicted_close_p50"].tolist()) == {303.0}
    assert wrapper.calls.count("multivariate") == 2
    assert "batch" not in wrapper.calls


def test_manager_falls_back_to_batch_when_joint_and_multivariate_unavailable(tmp_path: Path) -> None:
    history = _make_history("TEST", rows=80)
    wrapper = _BatchOnlyWrapper()
    manager = ChronosForecastManager(_build_cfg(tmp_path), wrapper_factory=lambda: wrapper)
    manager._use_multivariate = True
    manager._use_cross_learning = True

    out = manager._generate_forecast_chunk(history, [40, 41])

    assert out is not None
    assert not out.empty
    assert set(out["predicted_close_p50"].tolist()) == {111.0}
    assert wrapper.calls.count("joint") == 1
    assert wrapper.calls.count("batch") == 1


def test_manager_loads_inference_policy_from_chronos_params(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    def _fake_resolve(symbol: str, *, frequency: str | None = None, default_prediction_length: int = 7):  # type: ignore[no-untyped-def]
        del symbol, frequency, default_prediction_length
        return {
            "predict_kwargs": {"predict_batches_jointly": True},
            "use_multivariate": True,
            "use_cross_learning": True,
        }

    monkeypatch.setattr("binanceneural.forecasts.resolve_chronos2_params", _fake_resolve)

    manager = ChronosForecastManager(_build_cfg(tmp_path), wrapper_factory=lambda: object())
    assert manager._use_multivariate is True
    assert manager._use_cross_learning is True
    assert manager._predict_kwargs.get("predict_batches_jointly") is True
