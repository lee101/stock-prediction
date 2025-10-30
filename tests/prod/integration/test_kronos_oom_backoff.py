from __future__ import annotations

import types
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest
import torch

from src.models.kronos_wrapper import KronosForecastingWrapper, setup_kronos_wrapper_imports


class DummyPredictor:
    def __init__(self) -> None:
        self.calls = 0
        self.sample_counts: List[int] = []
        self.model = types.SimpleNamespace(to=lambda *_, **__: None)
        self.tokenizer = types.SimpleNamespace(to=lambda *_, **__: None)

    def predict(
        self,
        *_,
        pred_len: int,
        sample_count: int,
        **__,
    ) -> pd.DataFrame:
        self.calls += 1
        self.sample_counts.append(int(sample_count))
        if sample_count > 16:
            raise RuntimeError("CUDA out of memory")
        values = np.linspace(100.0, 100.0 + pred_len, pred_len)
        return pd.DataFrame({"close": values})

    def predict_batch(self, *args, **kwargs):
        raise AssertionError("Batch path should not be exercised in this test.")


@pytest.fixture(autouse=True)
def _patch_cuda(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True, raising=False)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None, raising=False)
    return


def _build_input_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=64, freq="D")
    base_values = np.linspace(95.0, 105.0, len(timestamps))
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": base_values + 0.5,
            "high": base_values + 1.0,
            "low": base_values - 1.0,
            "close": base_values,
            "volume": np.full(len(timestamps), 1_000.0),
        }
    )


def test_kronos_predict_series_adapts_sample_count(monkeypatch: pytest.MonkeyPatch) -> None:
    setup_kronos_wrapper_imports(torch_module=torch, numpy_module=np, pandas_module=pd)

    predictor = DummyPredictor()

    def fake_ensure_predictor(self: KronosForecastingWrapper, *, device_override=None):
        self._predictor = predictor
        return predictor

    monkeypatch.setattr(
        KronosForecastingWrapper,
        "_ensure_predictor",
        fake_ensure_predictor,
        raising=False,
    )

    wrapper = KronosForecastingWrapper(
        model_name="NeoQuasar/Kronos-base",
        tokenizer_name="NeoQuasar/Kronos-Tokenizer-base",
        device="cuda:0",
        sample_count=64,
    )

    result: Dict[str, object] = wrapper.predict_series(
        data=_build_input_frame(),
        timestamp_col="timestamp",
        columns=["Close"],
        pred_len=7,
        lookback=32,
    )

    assert "Close" in result, "Expected Kronos wrapper to return Close predictions."
    assert predictor.sample_counts[:3] == [64, 32, 16], "Sample count backoff sequence unexpected."
    assert wrapper._adaptive_sample_count == 16, "Wrapper did not persist adaptive limit after OOM recovery."

    result_second = wrapper.predict_series(
        data=_build_input_frame(),
        timestamp_col="timestamp",
        columns=["Close"],
        pred_len=7,
        lookback=32,
    )

    assert "Close" in result_second
    assert predictor.sample_counts[3] == 16, "Adaptive limit should cap subsequent invocations."
    assert predictor.calls == 4, "Predictor call count mismatch after adaptive recovery."
