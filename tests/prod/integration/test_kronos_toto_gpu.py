from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Dict, Iterator, Optional

import numpy as np
import pandas as pd
import pytest
import torch

from src.models.kronos_wrapper import KronosForecastResult, KronosForecastingWrapper, setup_kronos_wrapper_imports
from src.models.model_cache import ModelCacheManager, dtype_to_token
from src.models.toto_wrapper import TotoPipeline, setup_toto_wrapper_imports


_DATA_PATH = Path("trainingdata/BTCUSD.csv")


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required for Kronos/Toto integration tests.")


@pytest.fixture(scope="module")
def btc_series() -> pd.DataFrame:
    if not _DATA_PATH.exists():
        pytest.skip(f"Required dataset {_DATA_PATH} is missing.")
    frame = pd.read_csv(_DATA_PATH)
    required = {"timestamp", "open", "high", "low", "close"}
    missing = required.difference(frame.columns)
    if missing:
        pytest.skip(f"Dataset {_DATA_PATH} missing columns: {sorted(missing)}")
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    return frame


@pytest.fixture(scope="module")
def kronos_wrapper() -> Iterator[KronosForecastingWrapper]:
    _require_cuda()
    setup_kronos_wrapper_imports(torch_module=torch, numpy_module=np, pandas_module=pd)
    wrapper = KronosForecastingWrapper(
        model_name="NeoQuasar/Kronos-small",
        tokenizer_name="NeoQuasar/Kronos-Tokenizer-base",
        device="cuda:0",
        max_context=256,
        sample_count=4,
        clip=2.0,
        temperature=0.9,
        top_p=0.9,
        prefer_fp32=True,
    )
    try:
        yield wrapper
    finally:
        with contextlib.suppress(Exception):
            wrapper.unload()


@pytest.fixture(scope="module")
def toto_pipeline() -> Iterator[TotoPipeline]:
    _require_cuda()
    setup_toto_wrapper_imports(torch_module=torch, numpy_module=np)
    manager = ModelCacheManager("toto")
    dtype_token = dtype_to_token(torch.float32)
    metadata = manager.load_metadata("Datadog/Toto-Open-Base-1.0", dtype_token)
    refresh_needed = True
    if metadata is not None:
        refresh_needed = any(
            (
                metadata.get("device") != "cuda",
                metadata.get("dtype") != "fp32",
                metadata.get("amp_dtype") != "fp32",
                metadata.get("amp_autocast") is not False,
            )
        )
    preferred_dtype = torch.float32
    pipeline = TotoPipeline.from_pretrained(
        model_id="Datadog/Toto-Open-Base-1.0",
        device_map="cuda",
        torch_dtype=preferred_dtype,
        amp_dtype=None,
        amp_autocast=False,
        compile_model=False,
        torch_compile=False,
        warmup_sequence=0,
        cache_policy="prefer",
        force_refresh=refresh_needed,
        min_num_samples=64,
        min_samples_per_batch=16,
        max_oom_retries=0,
    )
    try:
        yield pipeline
    finally:
        with contextlib.suppress(Exception):
            pipeline.unload()


@pytest.mark.cuda_required
@pytest.mark.integration
def test_kronos_gpu_forecast(kronos_wrapper: KronosForecastingWrapper, btc_series: pd.DataFrame) -> None:
    window = btc_series[["timestamp", "open", "high", "low", "close", "volume"]].tail(320).copy()
    results = kronos_wrapper.predict_series(
        data=window,
        timestamp_col="timestamp",
        columns=["close"],
        pred_len=4,
    )

    assert "close" in results, "Kronos forecast missing 'close' column."
    forecast: KronosForecastResult = results["close"]
    assert forecast.absolute.shape == (4,), "Unexpected Kronos forecast horizon."
    assert np.isfinite(forecast.absolute).all(), "Kronos produced non-finite price levels."
    assert np.isfinite(forecast.percent).all(), "Kronos produced non-finite returns."

    assert kronos_wrapper._device.startswith("cuda"), "Kronos wrapper did not select GPU device."
    assert getattr(kronos_wrapper, "_preferred_dtype", None) is None, "Kronos wrapper selected reduced precision despite prefer_fp32."
    predictor = getattr(kronos_wrapper, "_predictor", None)
    assert predictor is not None, "Kronos predictor not initialised."
    device_attr = getattr(predictor, "device", "")
    assert isinstance(device_attr, str) and device_attr.startswith("cuda"), "Kronos predictor not using CUDA."


@pytest.mark.cuda_required
@pytest.mark.integration
def test_toto_gpu_forecast(toto_pipeline: TotoPipeline, btc_series: pd.DataFrame) -> None:
    context = torch.tensor(
        btc_series["close"].tail(256).to_numpy(),
        dtype=torch.float32,
        device="cuda",
    )

    forecasts = toto_pipeline.predict(
        context=context,
        prediction_length=4,
        num_samples=64,
        samples_per_batch=16,
        max_oom_retries=0,
    )

    assert len(forecasts) == 1, "Toto pipeline should return a single forecast batch."
    forecast = forecasts[0]
    numpy_forecast = forecast.numpy()
    assert numpy_forecast.shape == (64, 4), "Toto numpy() output shape mismatch."
    assert np.isfinite(numpy_forecast).all(), "Toto produced non-finite samples."

    assert toto_pipeline.device == "cuda", f"Expected Toto pipeline to run on CUDA, got {toto_pipeline.device!r}."
    assert toto_pipeline._autocast_dtype is None, "Toto pipeline unexpectedly enabled autocast in FP32 mode."
    param = next(toto_pipeline.model.parameters())
    assert param.dtype == torch.float32, f"Toto model parameter dtype {param.dtype} is not FP32."
    assert param.device.type == "cuda", "Toto model parameters are not resident on CUDA device."

    metadata: Optional[Dict[str, object]] = toto_pipeline.last_run_metadata
    assert metadata is not None, "Toto pipeline did not record run metadata."
    assert metadata.get("num_samples_used") == 64, "Toto adjusted num_samples away from the request."
    assert metadata.get("samples_per_batch_used") == 16, "Unexpected samples_per_batch adjustment."
    assert metadata.get("torch_dtype") == str(torch.float32), "Toto metadata recorded incorrect dtype."

    manager = ModelCacheManager("toto")
    cache_metadata = manager.load_metadata("Datadog/Toto-Open-Base-1.0", dtype_to_token(torch.float32))
    assert cache_metadata is not None, "Compiled Toto FP32 cache metadata missing."
    assert cache_metadata.get("device") == "cuda", "Cached Toto model not marked for CUDA device."
    assert cache_metadata.get("dtype") == "fp32", "Cached Toto model dtype mismatch."
    assert cache_metadata.get("amp_dtype") == "fp32", "Cached Toto model amp dtype mismatch."
    assert cache_metadata.get("amp_autocast") is False, "Compiled cache indicates autocast enabled when disabled."
