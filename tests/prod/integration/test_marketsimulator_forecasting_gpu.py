from __future__ import annotations

import json
import shutil
import time

import pytest
import torch

from marketsimulator.environment import activate_simulation
from marketsimulator.state import get_state

from src.models.model_cache import ModelCacheManager, dtype_to_token


KronosModelId = "NeoQuasar/Kronos-base"


def _skip_if_no_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required for marketsimulator forecasting cache test.")


@pytest.mark.cuda_required
@pytest.mark.integration
def test_marketsimulator_kronos_cache_fp32(monkeypatch):
    _skip_if_no_cuda()

    import predict_stock_forecasting as real_forecasting

    monkeypatch.setattr(real_forecasting, "KRONOS_SAMPLE_COUNT", 4, raising=False)
    monkeypatch.setattr(real_forecasting, "forecasting_wrapper", None, raising=False)

    manager = ModelCacheManager("kronos")
    dtype_token = dtype_to_token(torch.float32)
    metadata_path = manager.metadata_path(KronosModelId, dtype_token)
    cache_dir = metadata_path.parent
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

    weights_dir = manager.weights_dir(KronosModelId, dtype_token)
    weights_path = weights_dir / "model_state.pt"

    with activate_simulation(symbols=["AAPL"], use_mock_analytics=False, force_kronos=True):
        state = get_state()
        price_frame = state.prices["AAPL"].frame.copy()
        window = price_frame.tail(256)

        real_forecasting.load_pipeline()
        wrapper = real_forecasting.forecasting_wrapper
        assert wrapper is not None

        payload = window[["timestamp", "Open", "High", "Low", "Close", "Volume"]]

        torch.cuda.synchronize()
        start = time.perf_counter()
        first_result = wrapper.predict_series(
            data=payload,
            timestamp_col="timestamp",
            columns=["Close", "High", "Low"],
            pred_len=4,
        )
        torch.cuda.synchronize()
        first_duration = time.perf_counter() - start

        assert metadata_path.exists(), "Kronos metadata not persisted after first inference."
        assert weights_dir.exists(), "Kronos weights directory missing after first inference."
        if not weights_path.exists():
            weights_path = weights_dir / "model.safetensors"
        assert weights_path.exists(), "Kronos weights file not persisted after first inference."

        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        assert metadata.get("device", "").startswith("cuda")
        assert metadata.get("dtype") == "fp32"
        assert metadata.get("prefer_fp32") is True

        tokenizer_dir = weights_dir / "tokenizer"
        assert tokenizer_dir.exists(), "Kronos tokenizer cache directory missing."

        meta_mtime = metadata_path.stat().st_mtime
        weights_mtime = weights_path.stat().st_mtime

        torch.cuda.synchronize()
        start = time.perf_counter()
        second_result = wrapper.predict_series(
            data=payload,
            timestamp_col="timestamp",
            columns=["Close", "High", "Low"],
            pred_len=4,
        )
        torch.cuda.synchronize()
        second_duration = time.perf_counter() - start

        assert metadata_path.stat().st_mtime == pytest.approx(meta_mtime, rel=0, abs=1e-3)
        assert weights_path.stat().st_mtime == pytest.approx(weights_mtime, rel=0, abs=1e-3)

        if first_duration > 0.5:
            assert second_duration <= first_duration

    assert set(first_result.keys()) == {"Close", "High", "Low"}
    assert set(second_result.keys()) == {"Close", "High", "Low"}
    assert wrapper._device.startswith("cuda")
    assert wrapper._preferred_dtype is None


@pytest.mark.cuda_required
@pytest.mark.integration
def test_marketsimulator_kronos_cache_multi_symbol(monkeypatch):
    _skip_if_no_cuda()

    import predict_stock_forecasting as real_forecasting

    monkeypatch.setattr(real_forecasting, "KRONOS_SAMPLE_COUNT", 4, raising=False)
    monkeypatch.setattr(real_forecasting, "forecasting_wrapper", None, raising=False)

    manager = ModelCacheManager("kronos")
    dtype_token = dtype_to_token(torch.float32)
    metadata_path = manager.metadata_path(KronosModelId, dtype_token)
    assert metadata_path.exists(), "Expected Kronos metadata from prior cache warm-up."

    symbols = ["AAPL", "MSFT"]
    with activate_simulation(symbols=symbols, use_mock_analytics=False, force_kronos=True):
        state = get_state()

        real_forecasting.load_pipeline()
        wrapper = real_forecasting.forecasting_wrapper
        assert wrapper is not None

        def _payload(symbol: str):
            frame = state.prices[symbol].frame.copy()
            return frame[["timestamp", "Open", "High", "Low", "Close", "Volume"]].tail(256)

        first_durations = []
        first_outputs = {}
        for symbol in symbols:
            payload = _payload(symbol)
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = wrapper.predict_series(
                data=payload,
                timestamp_col="timestamp",
                columns=["Close"],
                pred_len=4,
            )
            torch.cuda.synchronize()
            first_durations.append(time.perf_counter() - start)
            first_outputs[symbol] = result

        second_durations = []
        second_outputs = {}
        for symbol in symbols:
            payload = _payload(symbol)
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = wrapper.predict_series(
                data=payload,
                timestamp_col="timestamp",
                columns=["Close"],
                pred_len=4,
            )
            torch.cuda.synchronize()
            second_durations.append(time.perf_counter() - start)
            second_outputs[symbol] = result

        for symbol in symbols:
            assert "Close" in first_outputs[symbol]
            assert "Close" in second_outputs[symbol]

        longest_first = max(first_durations)
        longest_second = max(second_durations)
        if longest_first > 0.5:
            assert longest_second <= longest_first

        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        assert metadata.get("device", "").startswith("cuda")
        assert metadata.get("dtype") == "fp32"
        assert metadata.get("prefer_fp32") is True
