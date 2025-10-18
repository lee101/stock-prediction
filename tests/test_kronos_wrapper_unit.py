from __future__ import annotations

import sys
import types
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

import torch

from src.models import kronos_wrapper as kw


class _StubTokenizer:
    def __init__(self) -> None:
        self.to_calls: list[str] = []

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        return cls()

    def to(self, device: str):
        self.to_calls.append(device)
        return self


class _StubModel:
    def __init__(self) -> None:
        self.to_calls: list[str] = []

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        return cls()

    def to(self, device: str):
        self.to_calls.append(device)
        return self

    def eval(self):
        return self


class _StubPredictor:
    def __init__(self, *, model, tokenizer, device, max_context, clip) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_context = max_context
        self.clip = clip
        self.calls: list[dict[str, object]] = []

    def predict(
        self,
        frame: pd.DataFrame,
        *,
        x_timestamp,
        y_timestamp,
        pred_len,
        T,
        top_k,
        top_p,
        sample_count,
        verbose,
    ):
        self.calls.append(
            {
                "frame": frame.copy(),
                "x_timestamp": pd.DatetimeIndex(x_timestamp),
                "y_timestamp": pd.DatetimeIndex(y_timestamp),
                "pred_len": pred_len,
                "T": T,
                "top_k": top_k,
                "top_p": top_p,
                "sample_count": sample_count,
                "verbose": verbose,
            }
        )
        base = np.linspace(100.0, 100.0 + (pred_len - 1), pred_len, dtype=np.float64)
        data = {
            "open": base + 1.0,
            "high": base + 2.0,
            "low": base - 1.0,
            "close": base,
            "volume": np.full(pred_len, 10.0),
            "amount": np.full(pred_len, 1000.0),
        }
        return pd.DataFrame(data, index=pd.DatetimeIndex(y_timestamp))


def _install_stub_module(monkeypatch: pytest.MonkeyPatch, predictor_cls=_StubPredictor) -> _StubPredictor:
    module = types.ModuleType("external.kronos.model")
    module.KronosTokenizer = _StubTokenizer
    module.Kronos = _StubModel
    module.KronosPredictor = predictor_cls
    monkeypatch.setitem(sys.modules, "external.kronos.model", module)
    return predictor_cls  # type: ignore[return-value]


@pytest.fixture(autouse=True)
def _ensure_stubbed_kronos(monkeypatch: pytest.MonkeyPatch):
    _install_stub_module(monkeypatch)
    yield
    sys.modules.pop("external.kronos.model", None)


def _make_sample_frame() -> pd.DataFrame:
    base = datetime(2024, 1, 1)
    rows = []
    for idx in range(12):
        timestamp = base + timedelta(days=idx)
        rows.append(
            {
                "timestamp": timestamp,
                "close": 100.0 + idx,
                "open": 99.5 + idx,
                "high": 100.5 + idx,
                "low": 98.5 + idx,
                "volume": 1000 + idx,
                "amount": (1000 + idx) * (100.0 + idx),
            }
        )
    return pd.DataFrame(rows)


def test_predict_series_returns_expected_structure(monkeypatch: pytest.MonkeyPatch):
    wrapper = kw.KronosForecastingWrapper(
        model_name="stub/model",
        tokenizer_name="stub/tokenizer",
        device="cpu",
        max_context=16,
        sample_count=3,
    )

    frame = _make_sample_frame()
    results = wrapper.predict_series(
        data=frame,
        timestamp_col="timestamp",
        columns=["close", "high"],
        pred_len=4,
        lookback=10,
    )

    assert set(results.keys()) == {"close", "high"}
    close_result = results["close"]
    assert isinstance(close_result, kw.KronosForecastResult)
    assert close_result.absolute.shape == (4,)
    assert close_result.percent.shape == (4,)
    assert np.isclose(close_result.percent[0], (close_result.absolute[0] - frame["close"].iloc[-1]) / frame["close"].iloc[-1])
    assert len(wrapper._predictor.calls) == 1
    call = wrapper._predictor.calls[0]
    assert call["pred_len"] == 4
    assert call["T"] == wrapper.temperature
    assert call["sample_count"] == 3


def test_predict_series_missing_column_raises(monkeypatch: pytest.MonkeyPatch):
    class _MissingColumnPredictor(_StubPredictor):
        def predict(self, *args, **kwargs):
            df = super().predict(*args, **kwargs)
            return df.drop(columns=["high"])

    _install_stub_module(monkeypatch, predictor_cls=_MissingColumnPredictor)
    wrapper = kw.KronosForecastingWrapper(
        model_name="stub/model",
        tokenizer_name="stub/tokenizer",
        device="cpu",
    )
    frame = _make_sample_frame()
    frame = frame.drop(columns=["high"])
    with pytest.raises(KeyError):
        wrapper.predict_series(
            data=frame,
            timestamp_col="timestamp",
            columns=["close", "high"],
            pred_len=2,
        )


def test_predict_series_cpu_fallback(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    wrapper = kw.KronosForecastingWrapper(
        model_name="stub/model",
        tokenizer_name="stub/tokenizer",
        device="cuda:0",
    )
    frame = _make_sample_frame()
    wrapper.predict_series(
        data=frame,
        timestamp_col="timestamp",
        columns=["close"],
        pred_len=1,
    )
    assert wrapper._device == "cpu"
