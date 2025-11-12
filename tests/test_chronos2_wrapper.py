"""Unit tests for the Chronos2 OHLC helper."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.chronos2_wrapper import (
    Chronos2OHLCWrapper,
    Chronos2PreparedPanel,
    DEFAULT_QUANTILE_LEVELS,
)


def _make_dataframe(rows: int = 64) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="D", tz="UTC")
    data = {
        "timestamp": index,
        "open": np.linspace(10.0, 10.0 + rows - 1, rows, dtype=np.float32),
        "high": np.linspace(12.0, 12.0 + rows - 1, rows, dtype=np.float32),
        "low": np.linspace(8.0, 8.0 + rows - 1, rows, dtype=np.float32),
        "close": np.linspace(11.0, 11.0 + rows - 1, rows, dtype=np.float32),
        "symbol": ["BTCUSD"] * rows,
        "volume": np.linspace(1000.0, 2000.0, rows, dtype=np.float32),
    }
    return pd.DataFrame(data)


def test_build_panel_uses_requested_window() -> None:
    df = _make_dataframe(80)
    panel = Chronos2OHLCWrapper.build_panel(
        context_df=df.iloc[:-10],
        holdout_df=df.iloc[-10:],
        future_covariates=None,
        symbol="BTCUSD",
        id_column="symbol",
        timestamp_column="timestamp",
        target_columns=("open", "close"),
        prediction_length=10,
        context_length=32,
    )

    assert isinstance(panel, Chronos2PreparedPanel)
    assert panel.context_length == 32
    assert len(panel.context_df) == 32
    assert panel.future_df is None
    assert len(panel.actual_df) == 10
    assert list(panel.context_df.columns) == ["symbol", "timestamp", "open", "close"]
    assert panel.actual_df.index.is_monotonic_increasing


def test_build_panel_truncates_when_history_is_short() -> None:
    df = _make_dataframe(20)
    panel = Chronos2OHLCWrapper.build_panel(
        context_df=df.iloc[:-5],
        holdout_df=df.iloc[-5:],
        future_covariates=None,
        symbol="BTCUSD",
        id_column="symbol",
        timestamp_column="timestamp",
        target_columns=("open", "close"),
        prediction_length=5,
        context_length=32,
    )

    assert panel.context_length == 15  # 20 total rows - 5 holdout rows
    assert len(panel.context_df) == 15
    assert panel.future_df is None
    assert len(panel.actual_df) == 5


class _DummyPipeline:
    def __init__(self) -> None:
        self.recorded_kwargs = []

    def predict_df(self, context_df, future_df=None, **kwargs):
        self.recorded_kwargs.append(kwargs)
        rows = []
        quantiles = kwargs.get("quantile_levels", DEFAULT_QUANTILE_LEVELS)
        if "prediction_length" in kwargs and kwargs["prediction_length"] is not None:
            prediction_length = int(kwargs["prediction_length"])
        elif future_df is not None:
            prediction_length = len(future_df)
        else:
            prediction_length = len(quantiles)

        if future_df is not None:
            timestamps = pd.to_datetime(future_df["timestamp"])
        else:
            ts_series = pd.to_datetime(context_df["timestamp"])
            freq = pd.infer_freq(ts_series)
            if freq is None:
                diffs = ts_series.diff().dropna()
                offset = pd.to_timedelta(diffs.median())
                freq = offset
            if isinstance(freq, str):
                date_index = pd.date_range(ts_series.iloc[-1], periods=prediction_length + 1, freq=freq, tz="UTC")[1:]
            else:
                last = ts_series.iloc[-1]
                date_index = [last + freq * (i + 1) for i in range(prediction_length)]
            timestamps = pd.to_datetime(date_index)

        for ts in timestamps:
            for target in kwargs["target"]:
                payload = {
                    "symbol": context_df["symbol"].iat[0],
                    "timestamp": ts,
                    "target_name": target,
                    "predictions": 0.0,
                }
                for level in quantiles:
                    payload[_quantile_column(level)] = float(level) * 100 + ts.day
                rows.append(payload)
        return pd.DataFrame(rows)


def _quantile_column(level: float) -> str:
    return format(level, "g")


def test_predict_ohlc_pivots_quantiles() -> None:
    df = _make_dataframe(40)
    pipeline = _DummyPipeline()
    wrapper = Chronos2OHLCWrapper(
        pipeline=pipeline,
        id_column="symbol",
        timestamp_column="timestamp",
        target_columns=("open", "close"),
        default_context_length=16,
        quantile_levels=(0.1, 0.5, 0.9),
    )

    context = df.iloc[:-4]
    holdout = df.iloc[-4:]
    batch = wrapper.predict_ohlc(
        context,
        symbol="BTCUSD",
        prediction_length=4,
        context_length=12,
        evaluation_df=holdout,
    )

    assert isinstance(batch.panel, Chronos2PreparedPanel)
    assert set(batch.quantile_frames.keys()) == {0.1, 0.5, 0.9}

    median = batch.median
    assert median.shape == (4, 2)
    assert all(col in median.columns for col in ("open", "close"))

    timestamps = batch.panel.actual_df.index
    assert np.allclose(
        median.loc[timestamps[0], "open"],
        0.5 * 100 + timestamps[0].day,
    )

    with pytest.raises(KeyError):
        batch.quantile(0.95)


def test_unload_blocks_future_predictions() -> None:
    df = _make_dataframe(24)
    wrapper = Chronos2OHLCWrapper(
        pipeline=_DummyPipeline(),
        default_context_length=8,
    )
    wrapper.unload()

    with pytest.raises(RuntimeError):
        wrapper.predict_ohlc(df, symbol="BTCUSD", prediction_length=4)
