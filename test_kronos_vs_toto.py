#!/usr/bin/env python3
"""
Hyperparameter sweep for Kronos vs Toto forecasting on BTCUSD closing prices.

Each run forecasts the final ``FORECAST_HORIZON`` steps of the dataset using:
    * NeoQuasar Kronos (via ``KronosForecastingWrapper``)
    * Datadog Toto (via ``TotoPipeline``)

For both models we evaluate several sampling configurations (temperature, top-p,
sample counts, aggregation strategy, etc.) and report:
    * Mean absolute error on closing prices
    * Mean absolute error on step-wise returns
    * Total inference latency
"""

from __future__ import annotations

import time
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple, TypeVar

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error

from src.models.kronos_wrapper import KronosForecastingWrapper
from src.models.toto_wrapper import TotoPipeline


_ENV_FORECAST_HORIZON = os.environ.get("FORECAST_HORIZON")
if _ENV_FORECAST_HORIZON:
    try:
        FORECAST_HORIZON = max(1, int(_ENV_FORECAST_HORIZON))
    except ValueError as exc:  # pragma: no cover - defensive guardrail
        raise ValueError("FORECAST_HORIZON must be an integer") from exc
else:
    FORECAST_HORIZON = 1


@dataclass(frozen=True)
class KronosRunConfig:
    name: str
    temperature: float
    top_p: float
    top_k: int
    sample_count: int
    max_context: int = 512
    clip: float = 5.0


@dataclass(frozen=True)
class TotoRunConfig:
    name: str
    num_samples: int
    aggregate: str = "mean"
    samples_per_batch: int = 256


@dataclass
class ForecastResult:
    prices: np.ndarray
    returns: np.ndarray
    latency_s: float
    metadata: Optional[dict] = None


@dataclass
class ModelEvaluation:
    name: str
    price_mae: float
    return_mae: float
    latency_s: float
    predicted_prices: np.ndarray
    predicted_returns: np.ndarray
    config: dict
    metadata: Optional[dict] = None


_ConfigT = TypeVar("_ConfigT")


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: Optional[int] = None) -> Optional[int]:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError as exc:  # pragma: no cover - defensive guardrail
        raise ValueError(f"Environment variable {name} must be an integer, got '{value}'.") from exc


def _parse_torch_dtype_from_env() -> Optional[torch.dtype]:
    value = os.environ.get("TOTO_TORCH_DTYPE")
    if value is None or value.strip() == "":
        return None
    normalized = value.strip().lower()
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized in {"auto", "default"}:
        return None
    dtype = mapping.get(normalized)
    if dtype is None:
        raise ValueError(
            f"Unsupported TOTO_TORCH_DTYPE '{value}'. "
            "Supported values: float32, float16, bfloat16."
        )
    return dtype


def _should_use_torch_compile() -> Tuple[bool, Optional[str], Optional[str]]:
    if not _env_flag("TOTO_TORCH_COMPILE"):
        return False, None, None
    mode = os.environ.get("TOTO_COMPILE_MODE")
    backend = os.environ.get("TOTO_COMPILE_BACKEND")
    return True, mode, backend


def _limit_configs(configs: Tuple[_ConfigT, ...], limit: Optional[int]) -> Tuple[_ConfigT, ...]:
    if limit is None or limit <= 0 or limit >= len(configs):
        return configs
    return configs[:limit]


DEFAULT_KRONOS_CONFIG = KronosRunConfig(
    name="kronos_default",
    temperature=0.60,
    top_p=0.85,
    top_k=0,
    sample_count=32,
)

KRONOS_SWEEP: Tuple[KronosRunConfig, ...] = (
    DEFAULT_KRONOS_CONFIG,
    KronosRunConfig(
        name="kronos_temp0.40_p0.90_s96_clip4_ctx384",
        temperature=0.40,
        top_p=0.90,
        top_k=0,
        sample_count=96,
        max_context=384,
        clip=4.0,
    ),
    KronosRunConfig(
        name="kronos_temp0.30_p0.88_s128_clip4_ctx384",
        temperature=0.30,
        top_p=0.88,
        top_k=0,
        sample_count=128,
        max_context=384,
        clip=4.0,
    ),
    KronosRunConfig(
        name="kronos_temp0.24_p0.87_s128_clip3.5_ctx448",
        temperature=0.24,
        top_p=0.87,
        top_k=0,
        sample_count=128,
        max_context=448,
        clip=3.5,
    ),
    KronosRunConfig(
        name="kronos_temp0.22_p0.88_s192_clip5_ctx512",
        temperature=0.22,
        top_p=0.88,
        top_k=0,
        sample_count=192,
        max_context=512,
        clip=5.0,
    ),
    KronosRunConfig(
        name="kronos_temp0.20_p0.90_s256_k32_clip5_ctx512",
        temperature=0.20,
        top_p=0.90,
        top_k=32,
        sample_count=256,
        max_context=512,
        clip=5.0,
    ),
    KronosRunConfig(
        name="kronos_temp0.18_p0.85_s192_clip3_ctx384",
        temperature=0.18,
        top_p=0.85,
        top_k=0,
        sample_count=192,
        max_context=384,
        clip=3.0,
    ),
    KronosRunConfig(
        name="kronos_temp0.18_p0.82_s160_clip3_ctx256",
        temperature=0.18,
        top_p=0.82,
        top_k=0,
        sample_count=160,
        max_context=256,
        clip=3.0,
    ),
    KronosRunConfig(
        name="kronos_temp0.16_p0.80_s192_k16_clip2_ctx256",
        temperature=0.16,
        top_p=0.80,
        top_k=16,
        sample_count=192,
        max_context=256,
        clip=2.0,
    ),
    KronosRunConfig(
        name="kronos_temp0.28_p0.90_s160_clip4_ctx512",
        temperature=0.28,
        top_p=0.90,
        top_k=0,
        sample_count=160,
        max_context=512,
        clip=4.0,
    ),
    KronosRunConfig(
        name="kronos_temp0.26_p0.86_s144_clip3_ctx320",
        temperature=0.26,
        top_p=0.86,
        top_k=0,
        sample_count=144,
        max_context=320,
        clip=3.0,
    ),
    KronosRunConfig(
        name="kronos_temp0.15_p0.82_s208_k16_clip1.8_ctx224",
        temperature=0.15,
        top_p=0.82,
        top_k=16,
        sample_count=208,
        max_context=224,
        clip=1.8,
    ),
    KronosRunConfig(
        name="kronos_temp0.145_p0.82_s208_k16_clip1.75_ctx224",
        temperature=0.145,
        top_p=0.82,
        top_k=16,
        sample_count=208,
        max_context=224,
        clip=1.75,
    ),
    KronosRunConfig(
        name="kronos_temp0.148_p0.81_s240_k18_clip1.7_ctx224",
        temperature=0.148,
        top_p=0.81,
        top_k=18,
        sample_count=240,
        max_context=224,
        clip=1.7,
    ),
    KronosRunConfig(
        name="kronos_temp0.152_p0.83_s192_k20_clip1.85_ctx232",
        temperature=0.152,
        top_p=0.83,
        top_k=20,
        sample_count=192,
        max_context=232,
        clip=1.85,
    ),
    KronosRunConfig(
        name="kronos_temp0.155_p0.82_s224_k18_clip1.9_ctx240",
        temperature=0.155,
        top_p=0.82,
        top_k=18,
        sample_count=224,
        max_context=240,
        clip=1.9,
    ),
    KronosRunConfig(
        name="kronos_temp0.14_p0.80_s200_k24_clip1.6_ctx224",
        temperature=0.14,
        top_p=0.80,
        top_k=24,
        sample_count=200,
        max_context=224,
        clip=1.6,
    ),
    KronosRunConfig(
        name="kronos_temp0.12_p0.78_s224_k24_clip1.5_ctx224",
        temperature=0.12,
        top_p=0.78,
        top_k=24,
        sample_count=224,
        max_context=224,
        clip=1.5,
    ),
    KronosRunConfig(
        name="kronos_temp0.18_p0.84_s224_k8_clip2.5_ctx288",
        temperature=0.18,
        top_p=0.84,
        top_k=8,
        sample_count=224,
        max_context=288,
        clip=2.5,
    ),
    KronosRunConfig(
        name="kronos_temp0.20_p0.82_s224_k12_clip2_ctx288",
        temperature=0.20,
        top_p=0.82,
        top_k=12,
        sample_count=224,
        max_context=288,
        clip=2.0,
    ),
    KronosRunConfig(
        name="kronos_temp0.22_p0.83_s192_clip2.5_ctx320",
        temperature=0.22,
        top_p=0.83,
        top_k=0,
        sample_count=192,
        max_context=320,
        clip=2.5,
    ),
    KronosRunConfig(
        name="kronos_temp0.24_p0.80_s224_clip2_ctx320",
        temperature=0.24,
        top_p=0.80,
        top_k=0,
        sample_count=224,
        max_context=320,
        clip=2.0,
    ),
    KronosRunConfig(
        name="kronos_temp0.14_p0.82_s240_k20_clip1.6_ctx208",
        temperature=0.14,
        top_p=0.82,
        top_k=20,
        sample_count=240,
        max_context=208,
        clip=1.6,
    ),
    KronosRunConfig(
        name="kronos_temp0.13_p0.79_s256_k24_clip1.5_ctx208",
        temperature=0.13,
        top_p=0.79,
        top_k=24,
        sample_count=256,
        max_context=208,
        clip=1.5,
    ),
    KronosRunConfig(
        name="kronos_temp0.12_p0.76_s256_k28_clip1.4_ctx192",
        temperature=0.12,
        top_p=0.76,
        top_k=28,
        sample_count=256,
        max_context=192,
        clip=1.4,
    ),
    KronosRunConfig(
        name="kronos_temp0.11_p0.75_s240_k28_clip1.3_ctx192",
        temperature=0.11,
        top_p=0.75,
        top_k=28,
        sample_count=240,
        max_context=192,
        clip=1.3,
    ),
    KronosRunConfig(
        name="kronos_temp0.10_p0.74_s288_k32_clip1.2_ctx192",
        temperature=0.10,
        top_p=0.74,
        top_k=32,
        sample_count=288,
        max_context=192,
        clip=1.2,
    ),
    KronosRunConfig(
        name="kronos_temp0.16_p0.78_s208_k18_clip1.9_ctx240",
        temperature=0.16,
        top_p=0.78,
        top_k=18,
        sample_count=208,
        max_context=240,
        clip=1.9,
    ),
    KronosRunConfig(
        name="kronos_temp0.18_p0.80_s208_k16_clip2.1_ctx256",
        temperature=0.18,
        top_p=0.80,
        top_k=16,
        sample_count=208,
        max_context=256,
        clip=2.1,
    ),
    KronosRunConfig(
        name="kronos_temp0.17_p0.79_s224_k12_clip1.8_ctx240",
        temperature=0.17,
        top_p=0.79,
        top_k=12,
        sample_count=224,
        max_context=240,
        clip=1.8,
    ),
    KronosRunConfig(
        name="kronos_temp0.118_p0.755_s288_k26_clip1.35_ctx192",
        temperature=0.118,
        top_p=0.755,
        top_k=26,
        sample_count=288,
        max_context=192,
        clip=1.35,
    ),
    KronosRunConfig(
        name="kronos_temp0.122_p0.765_s320_k28_clip1.4_ctx192",
        temperature=0.122,
        top_p=0.765,
        top_k=28,
        sample_count=320,
        max_context=192,
        clip=1.4,
    ),
    KronosRunConfig(
        name="kronos_temp0.115_p0.75_s256_k30_clip1.3_ctx176",
        temperature=0.115,
        top_p=0.75,
        top_k=30,
        sample_count=256,
        max_context=176,
        clip=1.3,
    ),
    KronosRunConfig(
        name="kronos_temp0.125_p0.77_s256_k24_clip1.45_ctx192",
        temperature=0.125,
        top_p=0.77,
        top_k=24,
        sample_count=256,
        max_context=192,
        clip=1.45,
    ),
)

TOTO_SWEEP: Tuple[TotoRunConfig, ...] = (
    TotoRunConfig(
        name="toto_mean_2048",
        num_samples=2048,
        aggregate="mean",
        samples_per_batch=256,
    ),
    TotoRunConfig(
        name="toto_median_2048",
        num_samples=2048,
        aggregate="median",
        samples_per_batch=256,
    ),
    TotoRunConfig(
        name="toto_quantile35_2048",
        num_samples=2048,
        aggregate="quantile_0.35",
        samples_per_batch=256,
    ),
    TotoRunConfig(
        name="toto_quantile25_2048",
        num_samples=2048,
        aggregate="quantile_0.25",
        samples_per_batch=256,
    ),
    TotoRunConfig(
        name="toto_lowertrim20_2048",
        num_samples=2048,
        aggregate="lower_trimmed_mean_20",
        samples_per_batch=256,
    ),
    TotoRunConfig(
        name="toto_trimmed10_3072",
        num_samples=3072,
        aggregate="trimmed_mean_10",
        samples_per_batch=384,
    ),
    TotoRunConfig(
        name="toto_mean_minus_std05_3072",
        num_samples=3072,
        aggregate="mean_minus_std_0.5",
        samples_per_batch=384,
    ),
    TotoRunConfig(
        name="toto_quantile18_4096",
        num_samples=4096,
        aggregate="quantile_0.18",
        samples_per_batch=512,
    ),
    TotoRunConfig(
        name="toto_quantile20_4096",
        num_samples=4096,
        aggregate="quantile_0.20",
        samples_per_batch=512,
    ),
    TotoRunConfig(
        name="toto_quantile22_4096",
        num_samples=4096,
        aggregate="quantile_0.22",
        samples_per_batch=512,
    ),
    TotoRunConfig(
        name="toto_mean_minus_std09_4096",
        num_samples=4096,
        aggregate="mean_minus_std_0.9",
        samples_per_batch=512,
    ),
    TotoRunConfig(
        name="toto_mean_minus_std10_4096",
        num_samples=4096,
        aggregate="mean_minus_std_1.0",
        samples_per_batch=512,
    ),
    TotoRunConfig(
        name="toto_lowertrim30_4096",
        num_samples=4096,
        aggregate="lower_trimmed_mean_30",
        samples_per_batch=512,
    ),
    TotoRunConfig(
        name="toto_quantile15_4096",
        num_samples=4096,
        aggregate="quantile_0.15",
        samples_per_batch=512,
    ),
    TotoRunConfig(
        name="toto_quantile12_4096",
        num_samples=4096,
        aggregate="quantile_0.12",
        samples_per_batch=512,
    ),
    TotoRunConfig(
        name="toto_quantile25_3072",
        num_samples=3072,
        aggregate="quantile_0.25",
        samples_per_batch=384,
    ),
    TotoRunConfig(
        name="toto_mean_minus_std08_3072",
        num_samples=3072,
        aggregate="mean_minus_std_0.8",
        samples_per_batch=384,
    ),
    TotoRunConfig(
        name="toto_quantile16_4096",
        num_samples=4096,
        aggregate="quantile_0.16",
        samples_per_batch=512,
    ),
    TotoRunConfig(
        name="toto_quantile17_4096",
        num_samples=4096,
        aggregate="quantile_0.17",
        samples_per_batch=512,
    ),
    TotoRunConfig(
        name="toto_quantile19_4096",
        num_samples=4096,
        aggregate="quantile_0.19",
        samples_per_batch=512,
    ),
    TotoRunConfig(
        name="toto_quantile21_4096",
        num_samples=4096,
        aggregate="quantile_0.21",
        samples_per_batch=512,
    ),
    TotoRunConfig(
        name="toto_quantile23_4096",
        num_samples=4096,
        aggregate="quantile_0.23",
        samples_per_batch=512,
    ),
    TotoRunConfig(
        name="toto_mean_quantile_mix_0.18_0.6",
        num_samples=4096,
        aggregate="mean_quantile_mix_0.18_0.6",
        samples_per_batch=512,
    ),
    TotoRunConfig(
        name="toto_mean_quantile_mix_0.17_0.5",
        num_samples=4096,
        aggregate="mean_quantile_mix_0.17_0.5",
        samples_per_batch=512,
    ),
    TotoRunConfig(
        name="toto_mean_quantile_mix_0.16_0.4",
        num_samples=4096,
        aggregate="mean_quantile_mix_0.16_0.4",
        samples_per_batch=512,
    ),
    TotoRunConfig(
        name="toto_mean_quantile_mix_0.15_0.3",
        num_samples=4096,
        aggregate="mean_quantile_mix_0.15_0.3",
        samples_per_batch=512,
    ),
    TotoRunConfig(
        name="toto_mean_quantile_mix_0.18_0.4",
        num_samples=3072,
        aggregate="mean_quantile_mix_0.18_0.4",
        samples_per_batch=384,
    ),
    TotoRunConfig(
        name="toto_quantile14_4096",
        num_samples=4096,
        aggregate="quantile_0.14",
        samples_per_batch=512,
    ),
    TotoRunConfig(
        name="toto_quantile145_4096",
        num_samples=4096,
        aggregate="quantile_0.145",
        samples_per_batch=512,
    ),
    TotoRunConfig(
        name="toto_quantile155_4096",
        num_samples=4096,
        aggregate="quantile_0.155",
        samples_per_batch=512,
    ),
    TotoRunConfig(
        name="toto_quantile165_4096",
        num_samples=4096,
        aggregate="quantile_0.165",
        samples_per_batch=512,
    ),
    TotoRunConfig(
        name="toto_mean_quantile_mix_0.15_0.5",
        num_samples=4096,
        aggregate="mean_quantile_mix_0.15_0.5",
        samples_per_batch=512,
    ),
    TotoRunConfig(
        name="toto_mean_quantile_mix_0.145_0.35",
        num_samples=4096,
        aggregate="mean_quantile_mix_0.145_0.35",
        samples_per_batch=512,
    ),
    TotoRunConfig(
        name="toto_mean_quantile_mix_0.145_0.40",
        num_samples=4096,
        aggregate="mean_quantile_mix_0.145_0.4",
        samples_per_batch=512,
    ),
    TotoRunConfig(
        name="toto_quantile_plus_std_0165_012",
        num_samples=4096,
        aggregate="quantile_plus_std_0.165_0.12",
        samples_per_batch=512,
    ),
    TotoRunConfig(
        name="toto_quantile_plus_std_0165_018",
        num_samples=4096,
        aggregate="quantile_plus_std_0.165_0.18",
        samples_per_batch=512,
    ),
    TotoRunConfig(
        name="toto_quantile_plus_std_015_015",
        num_samples=4096,
        aggregate="quantile_plus_std_0.15_0.15",
        samples_per_batch=512,
    ),
    TotoRunConfig(
        name="toto_quantile_plus_std_015_012",
        num_samples=4096,
        aggregate="quantile_plus_std_0.15_0.12",
        samples_per_batch=512,
    ),
    TotoRunConfig(
        name="toto_quantile_plus_std_0145_018",
        num_samples=4096,
        aggregate="quantile_plus_std_0.145_0.18",
        samples_per_batch=512,
    ),
    TotoRunConfig(
        name="toto_quantile_plus_std_016_020",
        num_samples=4096,
        aggregate="quantile_plus_std_0.16_0.20",
        samples_per_batch=512,
    ),
    TotoRunConfig(
        name="toto_quantile30_4096",
        num_samples=4096,
        aggregate="quantile_0.30",
        samples_per_batch=512,
    ),
    TotoRunConfig(
        name="toto_mean_minus_std075_4096",
        num_samples=4096,
        aggregate="mean_minus_std_0.75",
        samples_per_batch=512,
    ),
    TotoRunConfig(
        name="toto_quantile40_1024",
        num_samples=1024,
        aggregate="quantile_0.40",
        samples_per_batch=256,
    ),
    TotoRunConfig(
        name="toto_quantile15_3072",
        num_samples=3072,
        aggregate="quantile_0.15",
        samples_per_batch=384,
    ),
)

_kronos_wrapper: KronosForecastingWrapper | None = None
_toto_pipeline: TotoPipeline | None = None


def _load_kronos_wrapper() -> KronosForecastingWrapper:
    global _kronos_wrapper
    if _kronos_wrapper is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        cfg = DEFAULT_KRONOS_CONFIG
        _kronos_wrapper = KronosForecastingWrapper(
            model_name="NeoQuasar/Kronos-base",
            tokenizer_name="NeoQuasar/Kronos-Tokenizer-base",
            device=device,
            max_context=cfg.max_context,
            clip=cfg.clip,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            sample_count=cfg.sample_count,
        )
    return _kronos_wrapper


def _load_toto_pipeline() -> TotoPipeline:
    global _toto_pipeline
    if _toto_pipeline is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = _parse_torch_dtype_from_env()
        pipeline_kwargs = {}
        max_retries = _env_int("TOTO_MAX_OOM_RETRIES")
        if max_retries is not None:
            pipeline_kwargs["max_oom_retries"] = max_retries
        min_spb = _env_int("TOTO_MIN_SAMPLES_PER_BATCH")
        if min_spb is not None:
            pipeline_kwargs["min_samples_per_batch"] = min_spb
        min_samples = _env_int("TOTO_MIN_NUM_SAMPLES")
        if min_samples is not None:
            pipeline_kwargs["min_num_samples"] = min_samples
        torch_compile, compile_mode, compile_backend = _should_use_torch_compile()
        if torch_compile:
            pipeline_kwargs.update(
                {
                    "torch_compile": True,
                    "compile_mode": compile_mode,
                    "compile_backend": compile_backend,
                }
            )

        _toto_pipeline = TotoPipeline.from_pretrained(
            model_id="Datadog/Toto-Open-Base-1.0",
            device_map=device,
            torch_dtype=torch_dtype,
            **pipeline_kwargs,
        )
    return _toto_pipeline


def _config_to_dict(config) -> dict:
    data = asdict(config)
    data.pop("name", None)
    return data


def _compute_actuals(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    if len(df) <= FORECAST_HORIZON:
        raise ValueError("Dataset must contain more rows than the forecast horizon.")

    closing_prices = df["close"].to_numpy(dtype=np.float64)
    context_prices = closing_prices[:-FORECAST_HORIZON]
    target_prices = closing_prices[-FORECAST_HORIZON:]

    returns = []
    prev_price = context_prices[-1]
    for price in target_prices:
        if prev_price == 0:
            returns.append(0.0)
        else:
            returns.append((price - prev_price) / prev_price)
        prev_price = price

    return target_prices, np.asarray(returns, dtype=np.float64)


def _ensure_sample_matrix(samples: np.ndarray) -> np.ndarray:
    arr = np.asarray(samples)
    arr = np.squeeze(arr)

    if arr.ndim == 1:
        return arr.reshape(-1, 1).astype(np.float64)

    if arr.ndim == 2:
        if arr.shape[1] == FORECAST_HORIZON:
            return arr.astype(np.float64, copy=False)
        if arr.shape[0] == FORECAST_HORIZON:
            return arr.T.astype(np.float64, copy=False)

    if arr.ndim == 3 and 1 in arr.shape:
        arr = np.squeeze(arr, axis=tuple(idx for idx, size in enumerate(arr.shape) if size == 1))
        return _ensure_sample_matrix(arr)

    raise ValueError(f"Unrecognised sample tensor shape: {arr.shape}")


def _trimmed_mean(matrix: np.ndarray, fraction: float) -> np.ndarray:
    if not 0.0 <= fraction < 0.5:
        raise ValueError("Trimmed mean fraction must be in [0, 0.5).")

    sorted_matrix = np.sort(matrix, axis=0)
    total = sorted_matrix.shape[0]
    trim = int(total * fraction)

    if trim == 0 or trim * 2 >= total:
        return sorted_matrix.mean(axis=0, dtype=np.float64)

    return sorted_matrix[trim : total - trim].mean(axis=0, dtype=np.float64)


def _parse_percentage_token(token: str) -> float:
    value = float(token)
    if value > 1.0:
        value /= 100.0
    return value


def _aggregate_samples(samples: np.ndarray, method: str) -> np.ndarray:
    matrix = _ensure_sample_matrix(samples)

    if method == "mean":
        return matrix.mean(axis=0, dtype=np.float64)
    if method == "median":
        return np.median(matrix, axis=0)
    if method == "p10":
        return np.quantile(matrix, 0.10, axis=0)
    if method == "p90":
        return np.quantile(matrix, 0.90, axis=0)
    if method.startswith("trimmed_mean_"):
        try:
            fraction = _parse_percentage_token(method.split("_")[-1])
        except ValueError as exc:
            raise ValueError(f"Invalid trimmed mean specifier: '{method}'") from exc
        return _trimmed_mean(matrix, fraction)
    if method.startswith("lower_trimmed_mean_"):
        try:
            fraction = _parse_percentage_token(method.split("_")[-1])
        except ValueError as exc:
            raise ValueError(f"Invalid lower trimmed mean specifier: '{method}'") from exc
        sorted_matrix = np.sort(matrix, axis=0)
        total = sorted_matrix.shape[0]
        cutoff = max(1, int(total * (1.0 - fraction)))
        return sorted_matrix[:cutoff].mean(axis=0, dtype=np.float64)
    if method.startswith("upper_trimmed_mean_"):
        try:
            fraction = _parse_percentage_token(method.split("_")[-1])
        except ValueError as exc:
            raise ValueError(f"Invalid upper trimmed mean specifier: '{method}'") from exc
        sorted_matrix = np.sort(matrix, axis=0)
        total = sorted_matrix.shape[0]
        start = min(total - 1, int(total * fraction))
        return sorted_matrix[start:].mean(axis=0, dtype=np.float64)
    if method.startswith("quantile_"):
        try:
            quantile = _parse_percentage_token(method.split("_")[-1])
        except ValueError as exc:
            raise ValueError(f"Invalid quantile specifier: '{method}'") from exc
        return np.quantile(matrix, quantile, axis=0)
    if method.startswith("mean_minus_std_"):
        try:
            factor = float(method.split("_")[-1])
        except ValueError as exc:
            raise ValueError(f"Invalid mean_minus_std specifier: '{method}'") from exc
        mean = matrix.mean(axis=0, dtype=np.float64)
        std = matrix.std(axis=0, dtype=np.float64)
        return mean - factor * std
    if method.startswith("mean_plus_std_"):
        try:
            factor = float(method.split("_")[-1])
        except ValueError as exc:
            raise ValueError(f"Invalid mean_plus_std specifier: '{method}'") from exc
        mean = matrix.mean(axis=0, dtype=np.float64)
        std = matrix.std(axis=0, dtype=np.float64)
        return mean + factor * std
    if method.startswith("mean_quantile_mix_"):
        parts = method.split("_")
        if len(parts) < 5:
            raise ValueError(f"Invalid mean_quantile_mix specifier: '{method}'")
        try:
            quantile = _parse_percentage_token(parts[-2])
            mean_weight = float(parts[-1])
        except ValueError as exc:
            raise ValueError(f"Invalid mean_quantile_mix parameters in '{method}'") from exc
        mean_weight = np.clip(mean_weight, 0.0, 1.0)
        mean_val = matrix.mean(axis=0, dtype=np.float64)
        quant_val = np.quantile(matrix, quantile, axis=0)
        return mean_weight * mean_val + (1.0 - mean_weight) * quant_val
    if method.startswith("quantile_plus_std_"):
        parts = method.split("_")
        if len(parts) < 5:
            raise ValueError(f"Invalid quantile_plus_std specifier: '{method}'")
        try:
            quantile = _parse_percentage_token(parts[-2])
            factor = float(parts[-1])
        except ValueError as exc:
            raise ValueError(f"Invalid quantile_plus_std parameters in '{method}'") from exc
        quant_val = np.quantile(matrix, quantile, axis=0)
        std = matrix.std(axis=0, dtype=np.float64)
        return quant_val + factor * std

    raise ValueError(f"Unknown aggregation method '{method}'")


def _forecast_with_kronos(df: pd.DataFrame, config: KronosRunConfig) -> ForecastResult:
    wrapper = _load_kronos_wrapper()
    if hasattr(wrapper, "_predictor"):
        if wrapper.clip != config.clip or wrapper.max_context != config.max_context:
            wrapper.clip = config.clip
            wrapper.max_context = config.max_context
            wrapper._predictor = None
    start_time = time.perf_counter()
    results = wrapper.predict_series(
        data=df,
        timestamp_col="timestamp",
        columns=["close"],
        pred_len=FORECAST_HORIZON,
        lookback=config.max_context,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        sample_count=config.sample_count,
    )
    latency = time.perf_counter() - start_time

    kronos_result = results.get("close")
    if kronos_result is None:
        raise RuntimeError("Kronos did not return forecasts for the 'close' column.")

    prices = kronos_result.absolute.astype(np.float64)
    returns = kronos_result.percent.astype(np.float64)
    metadata = {
        "sample_count_used": getattr(wrapper, "_last_sample_count", None),
        "requested_sample_count": config.sample_count,
    }
    return ForecastResult(prices=prices, returns=returns, latency_s=latency, metadata=metadata)


def _forecast_with_toto(
    context: np.ndarray,
    last_price: float,
    config: TotoRunConfig,
) -> ForecastResult:
    pipeline = _load_toto_pipeline()

    context_tensor = np.asarray(context, dtype=np.float32)

    start_time = time.perf_counter()
    forecasts = pipeline.predict(
        context=context_tensor,
        prediction_length=FORECAST_HORIZON,
        num_samples=config.num_samples,
        samples_per_batch=config.samples_per_batch,
    )
    latency = time.perf_counter() - start_time

    run_metadata = dict(getattr(pipeline, "_last_run_metadata", {}) or {})
    if not run_metadata:
        run_metadata = {
            "num_samples_requested": config.num_samples,
            "samples_per_batch_requested": config.samples_per_batch,
        }
    run_metadata.setdefault("config_num_samples", config.num_samples)
    run_metadata.setdefault("config_samples_per_batch", config.samples_per_batch)
    run_metadata["torch_dtype"] = str(getattr(pipeline, "model_dtype", "unknown"))

    if not forecasts:
        raise RuntimeError("Toto did not return any forecasts.")

    step_values = _aggregate_samples(forecasts[0].samples, config.aggregate)
    step_values = np.asarray(step_values, dtype=np.float64)
    if step_values.size != FORECAST_HORIZON:
        raise ValueError(
            f"Aggregated Toto step values shape {step_values.shape} does not match horizon {FORECAST_HORIZON}"
        )

    prices = []
    returns = []
    prev_price = float(last_price)
    for price in step_values:
        price_float = float(price)
        prices.append(price_float)
        if prev_price == 0.0:
            returns.append(0.0)
        else:
            returns.append((price_float - prev_price) / prev_price)
        prev_price = price_float

    return ForecastResult(
        prices=np.asarray(prices, dtype=np.float64),
        returns=np.asarray(returns, dtype=np.float64),
        latency_s=latency,
        metadata=run_metadata,
    )


def _evaluate_kronos(
    df: pd.DataFrame,
    actual_prices: np.ndarray,
    actual_returns: np.ndarray,
    config: KronosRunConfig,
) -> ModelEvaluation:
    forecast = _forecast_with_kronos(df.copy(), config)
    return ModelEvaluation(
        name=f"Kronos/{config.name}",
        price_mae=mean_absolute_error(actual_prices, forecast.prices),
        return_mae=mean_absolute_error(actual_returns, forecast.returns),
        latency_s=forecast.latency_s,
        predicted_prices=forecast.prices,
        predicted_returns=forecast.returns,
        config=_config_to_dict(config),
        metadata=forecast.metadata,
    )


def _evaluate_toto(
    context: np.ndarray,
   last_price: float,
    actual_prices: np.ndarray,
    actual_returns: np.ndarray,
    config: TotoRunConfig,
) -> ModelEvaluation:
    forecast = _forecast_with_toto(context, last_price, config)
    config_dict = _config_to_dict(config)
    metadata = forecast.metadata or {}
    dtype_value = metadata.get("torch_dtype")
    if dtype_value is not None:
        config_dict = {**config_dict, "torch_dtype": dtype_value}
    return ModelEvaluation(
        name=f"Toto/{config.name}",
        price_mae=mean_absolute_error(actual_prices, forecast.prices),
        return_mae=mean_absolute_error(actual_returns, forecast.returns),
        latency_s=forecast.latency_s,
        predicted_prices=forecast.prices,
        predicted_returns=forecast.returns,
        config=config_dict,
        metadata=metadata,
    )


def _format_seconds(seconds: float) -> str:
    return f"{seconds:.3f}s"


def _print_ranked_results(title: str, evaluations: Tuple[ModelEvaluation, ...]) -> None:
    print(title)
    ordered = sorted(evaluations, key=lambda item: item.price_mae)
    for entry in ordered:
        cfg = ", ".join(f"{k}={v}" for k, v in entry.config.items())
        meta = ""
        if entry.metadata:
            meta_values = ", ".join(f"{k}={v}" for k, v in entry.metadata.items())
            meta = f" | meta: {meta_values}"
        print(
            f"  {entry.name:<32} "
            f"price_mae={entry.price_mae:.6f} "
            f"return_mae={entry.return_mae:.6f} "
            f"latency={_format_seconds(entry.latency_s)} "
            f"[{cfg}]{meta}"
        )
    print()


def main() -> None:
    data_path = Path(__file__).parent / "trainingdata" / "BTCUSD.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Expected dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    if "timestamp" not in df.columns:
        raise KeyError("Dataset must include a 'timestamp' column.")

    df = df.sort_values("timestamp").reset_index(drop=True)

    actual_prices, actual_returns = _compute_actuals(df)

    skip_kronos = _env_flag("SKIP_KRONOS")
    skip_toto = _env_flag("SKIP_TOTO")
    kronos_limit = _env_int("KRONOS_SWEEP_LIMIT", default=0)
    toto_limit = _env_int("TOTO_SWEEP_LIMIT", default=0)

    kronos_configs = _limit_configs(KRONOS_SWEEP, kronos_limit)
    toto_configs = _limit_configs(TOTO_SWEEP, toto_limit)

    kronos_evals: Tuple[ModelEvaluation, ...] = tuple()
    if skip_kronos:
        print("Skipping Kronos sweep (SKIP_KRONOS=1).")
    elif kronos_configs:
        kronos_evals = tuple(
            _evaluate_kronos(df, actual_prices, actual_returns, cfg) for cfg in kronos_configs
        )
    else:
        print("No Kronos configurations selected (KRONOS_SWEEP_LIMIT=0).")

    context_series = df["close"].to_numpy(dtype=np.float64)
    context_slice = context_series[:-FORECAST_HORIZON]
    last_price = float(context_slice[-1])

    toto_evals: Tuple[ModelEvaluation, ...] = tuple()
    if skip_toto:
        print("Skipping Toto sweep (SKIP_TOTO=1).")
    elif toto_configs:
        try:
            pipeline = _load_toto_pipeline()
        except Exception as exc:
            print(f"Failed to load Toto pipeline: {exc}")
            toto_evals = tuple()
        else:
            print(
                "Loaded Toto pipeline on device '%s' with dtype %s (torch.compile=%s)"
                % (
                    pipeline.device,
                    getattr(pipeline, "model_dtype", "unknown"),
                    getattr(pipeline, "_torch_compile_success", False),
                )
            )
            toto_evals = tuple(
                _evaluate_toto(context_slice, last_price, actual_prices, actual_returns, cfg)
                for cfg in toto_configs
            )
    else:
        print("No Toto configurations selected (TOTO_SWEEP_LIMIT=0).")

    if not kronos_evals and not toto_evals:
        print("Nothing to evaluate. Adjust environment flags or config limits.")
        return

    print("==== Kronos vs Toto Forecast Benchmark ====")
    print(f"Forecast horizon: {FORECAST_HORIZON} steps")
    print(f"Context length: {len(df) - FORECAST_HORIZON}")
    print()

    if kronos_evals:
        _print_ranked_results("Kronos hyperparameter sweep", kronos_evals)
        best_kronos = min(kronos_evals, key=lambda item: item.price_mae)
        print("Best Kronos configuration (price MAE)")
        print(
            f"  {best_kronos.name}: price_mae={best_kronos.price_mae:.6f}, "
            f"return_mae={best_kronos.return_mae:.6f}, "
            f"latency={_format_seconds(best_kronos.latency_s)}"
        )
        print(f"  Predicted prices:  {np.round(best_kronos.predicted_prices, 4)}")
        print(f"  Predicted returns: {np.round(best_kronos.predicted_returns, 6)}")
        print()

    if toto_evals:
        _print_ranked_results("Toto hyperparameter sweep", toto_evals)
        best_toto = min(toto_evals, key=lambda item: item.price_mae)
        print("Best Toto configuration (price MAE)")
        print(
            f"  {best_toto.name}: price_mae={best_toto.price_mae:.6f}, "
            f"return_mae={best_toto.return_mae:.6f}, "
            f"latency={_format_seconds(best_toto.latency_s)}"
        )
        print(f"  Predicted prices:  {np.round(best_toto.predicted_prices, 4)}")
        print(f"  Predicted returns: {np.round(best_toto.predicted_returns, 6)}")
        print()

    print("Actual future prices")
    print(f"  Prices:  {np.round(actual_prices, 4)}")
    print(f"  Returns: {np.round(actual_returns, 6)}")


if __name__ == "__main__":
    main()
