from __future__ import annotations

from types import SimpleNamespace
from typing import List

import numpy as np
import pandas as pd
import pytest
import torch

from src.models import kronos_wrapper as kw
from src.models import toto_wrapper as tw


class DummyMaskedTimeseries:
    def __init__(self, *, series, padding_mask, id_mask, timestamp_seconds, time_interval_seconds):
        self.series = series
        self.padding_mask = padding_mask
        self.id_mask = id_mask
        self.timestamp_seconds = timestamp_seconds
        self.time_interval_seconds = time_interval_seconds


class DummyForecast:
    def __init__(self, samples: torch.Tensor):
        self.samples = samples


def _build_toto_pipeline() -> tw.TotoPipeline:
    pipeline = object.__new__(tw.TotoPipeline)
    pipeline.device = "cpu"
    pipeline.model_dtype = torch.float32
    pipeline._autocast_dtype = None
    pipeline.forecaster = SimpleNamespace()
    pipeline.max_oom_retries = 0
    pipeline.min_samples_per_batch = 1
    pipeline.min_num_samples = 1
    pipeline._torch_compile_enabled = False
    pipeline._torch_compile_success = False
    pipeline._compile_mode = None
    pipeline._compile_backend = None
    pipeline._compiled = False
    return pipeline  # type: ignore[return-value]


def _fake_forecast_with_retries(
    _forecaster,
    *,
    inputs,
    prediction_length: int,
    num_samples: int,
    samples_per_batch: int,
    **_: object,
):
    series = inputs.series.detach().cpu().numpy()
    batch_size = series.shape[0]
    step_template = np.arange(1, prediction_length + 1, dtype=np.float32)
    forecasts: List[np.ndarray] = []
    for idx in range(batch_size):
        last_value = float(series[idx, -1])
        sample_paths = []
        for sample_idx in range(num_samples):
            sample_paths.append(last_value + step_template + float(sample_idx))
        sample_array = np.stack(sample_paths, axis=-1)  # (prediction_length, num_samples)
        forecasts.append(sample_array)
    stacked = np.stack(forecasts, axis=0)  # (batch, prediction_length, num_samples)
    stacked = np.expand_dims(stacked, axis=1)  # (batch, 1, prediction_length, num_samples)
    samples_tensor = torch.tensor(stacked, dtype=torch.float32)
    return DummyForecast(samples_tensor), num_samples, samples_per_batch


def test_toto_batch_forecast_matches_single(monkeypatch):
    pipeline = _build_toto_pipeline()
    monkeypatch.setattr(tw, "MaskedTimeseries", DummyMaskedTimeseries)
    monkeypatch.setattr(tw, "_forecast_with_retries", _fake_forecast_with_retries)

    batch_context = torch.tensor([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]], dtype=torch.float32)
    num_samples = 4
    prediction_length = 3

    batched = pipeline.predict(
        context=batch_context,
        prediction_length=prediction_length,
        num_samples=num_samples,
        samples_per_batch=2,
    )

    assert len(batched) == batch_context.shape[0]
    assert pipeline._last_run_metadata["batch_size"] == batch_context.shape[0]

    singles = []
    for idx in range(batch_context.shape[0]):
        singles.append(
            pipeline.predict(
                context=batch_context[idx],
                prediction_length=prediction_length,
                num_samples=num_samples,
                samples_per_batch=2,
            )[0]
        )

    for batch_forecast, single_forecast in zip(batched, singles):
        batch_matrix = batch_forecast.numpy()
        single_matrix = single_forecast.numpy()
        mae = np.mean(np.abs(batch_matrix - single_matrix))
        assert mae == pytest.approx(0.0)


class DummyKronosPredictor:
    def __init__(self):
        self.predict_batch_called = False

    def predict(self, feature_frame, *, x_timestamp, y_timestamp, pred_len, **_: object):
        return self._build_forecast(feature_frame, y_timestamp, pred_len)

    def predict_batch(self, df_list, x_timestamp_list, y_timestamp_list, pred_len, **_: object):
        self.predict_batch_called = True
        forecasts = []
        for frame, y_ts in zip(df_list, y_timestamp_list):
            forecasts.append(self._build_forecast(frame, y_ts, pred_len))
        return forecasts

    @staticmethod
    def _build_forecast(frame: pd.DataFrame, y_timestamp, pred_len: int) -> pd.DataFrame:
        last_close = float(frame["close"].iloc[-1])
        steps = np.arange(1, pred_len + 1, dtype=np.float64)
        close = last_close + steps
        index = pd.DatetimeIndex(y_timestamp)
        data = {
            "open": close + 0.1,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": np.full_like(close, 123.0),
            "amount": np.full_like(close, 456.0),
        }
        return pd.DataFrame(data, index=index)


def _build_kronos_wrapper(dummy_predictor: DummyKronosPredictor) -> kw.KronosForecastingWrapper:
    wrapper = object.__new__(kw.KronosForecastingWrapper)
    wrapper.temperature = 0.75
    wrapper.top_p = 0.9
    wrapper.top_k = 0
    wrapper.sample_count = 8
    wrapper.verbose = False
    wrapper.requested_device = "cpu"
    wrapper._device = "cpu"
    wrapper._preferred_dtype = None
    wrapper._predictor = dummy_predictor
    wrapper._ensure_predictor = lambda: dummy_predictor  # type: ignore[assignment]
    return wrapper  # type: ignore[return-value]


def test_kronos_batch_forecast_matches_single():
    predictor = DummyKronosPredictor()
    wrapper = _build_kronos_wrapper(predictor)

    base_dates = pd.date_range("2025-01-01", periods=16, freq="H")
    data_frames = []
    for offset in (0.0, 10.0, 20.0):
        frame = pd.DataFrame(
            {
                "timestamp": base_dates,
                "close": np.linspace(100.0 + offset, 102.0 + offset, len(base_dates)),
                "volume": np.linspace(1.0, 2.0, len(base_dates)),
            }
        )
        data_frames.append(frame)

    columns = ["close"]
    pred_len = 5

    batched = wrapper.predict_series_batch(
        data_frames=data_frames,
        timestamp_col="timestamp",
        columns=columns,
        pred_len=pred_len,
    )

    assert predictor.predict_batch_called is True
    assert len(batched) == len(data_frames)

    singles = [
        wrapper.predict_series(
            data=frame,
            timestamp_col="timestamp",
            columns=columns,
            pred_len=pred_len,
        )
        for frame in data_frames
    ]

    for batch_result, single_result in zip(batched, singles):
        batch_close = batch_result["close"]
        single_close = single_result["close"]
        assert batch_close.timestamps.equals(single_close.timestamps)
        mae_abs = np.mean(np.abs(batch_close.absolute - single_close.absolute))
        mae_pct = np.mean(np.abs(batch_close.percent - single_close.percent))
        assert mae_abs == pytest.approx(0.0)
        assert mae_pct == pytest.approx(0.0)

