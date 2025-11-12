#!/usr/bin/env python3
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import pytest

from src.models.kronos_wrapper import KronosForecastingWrapper
from src.models.toto_wrapper import TotoPipeline
from src.models.toto_aggregation import aggregate_with_spec

try:  # pragma: no cover - torch is required for real inference but may be missing in stubbed envs
    import torch  # type: ignore
except Exception:  # pragma: no cover - handled via module-level skip below
    torch = None  # type: ignore

if torch is None:  # pragma: no cover - ensures clear skip when torch unavailable
    pytest.skip("PyTorch is required for OHLC batching tests", allow_module_level=True)

COLUMNS: Tuple[str, ...] = ("open", "high", "low", "close")
DATA_ROOT = Path("trainingdata")
TOTO_CACHE_ROOT = Path("compiled_models/toto/Datadog-Toto-Open-Base-1.0/fp32/weights")
KRONOS_CACHE_ROOT = Path("compiled_models/kronos/NeoQuasar-Kronos-base/fp32/weights")
DEFAULT_SYMBOL = "AAPL"
CONTEXT_LENGTH = 96
PREDICTION_LENGTH = 1
WINDOW_COUNT = 3
TOTO_NUM_SAMPLES = 12
TOTO_SAMPLES_PER_BATCH = 6
TOTO_SEEDS: Tuple[int, ...] = (3, 7, 11)
TOTO_MAE_REPEATS = 3
TOTO_BATCH_TOLERANCE = 2.5e-1
TOTO_COMPILE_TOLERANCE = 5e-1
KRONOS_SAMPLE_COUNT = 16
KRONOS_BATCH_TOLERANCE = 5e-1
KRONOS_COMPILE_TOLERANCE = 6e-1

_TOTO_PIPELINE_CACHE: Dict[bool, TotoPipeline] = {}
_KRONOS_WRAPPER_CACHE: Dict[bool, KronosForecastingWrapper] = {}


@pytest.fixture(scope="session", autouse=True)
def _cleanup_models(request) -> None:
    def _cleanup() -> None:
        for pipeline in _TOTO_PIPELINE_CACHE.values():
            pipeline.unload()
        for wrapper in _KRONOS_WRAPPER_CACHE.values():
            wrapper.unload()

    request.addfinalizer(_cleanup)


@pytest.fixture(scope="session")
def ohlc_windows() -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    return _make_windows(DEFAULT_SYMBOL, WINDOW_COUNT)


@pytest.mark.parametrize("compiled", [False, True], ids=["toto_eager", "toto_compiled"])
def test_toto_batching_preserves_mae(compiled: bool, ohlc_windows: List[Tuple[pd.DataFrame, pd.DataFrame]]) -> None:
    pipeline = _get_toto_pipeline(compiled)
    sequential = _evaluate_toto_mae(
        pipeline,
        ohlc_windows,
        batch=False,
    )
    batched = _evaluate_toto_mae(
        pipeline,
        ohlc_windows,
        batch=True,
    )
    assert abs(sequential - batched) <= TOTO_BATCH_TOLERANCE


def test_toto_compilation_mae_stability(ohlc_windows: List[Tuple[pd.DataFrame, pd.DataFrame]]) -> None:
    eager = _evaluate_toto_mae(_get_toto_pipeline(False), ohlc_windows, batch=True)
    compiled = _evaluate_toto_mae(_get_toto_pipeline(True), ohlc_windows, batch=True)
    assert abs(eager - compiled) <= TOTO_COMPILE_TOLERANCE


@pytest.mark.parametrize("compiled", [False, True], ids=["kronos_eager", "kronos_compiled"])
def test_kronos_batching_preserves_mae(compiled: bool, ohlc_windows: List[Tuple[pd.DataFrame, pd.DataFrame]]) -> None:
    wrapper = _get_kronos_wrapper(compiled)
    sequential = _evaluate_kronos_mae(wrapper, ohlc_windows, batch=False)
    batched = _evaluate_kronos_mae(wrapper, ohlc_windows, batch=True)
    assert abs(sequential - batched) <= KRONOS_BATCH_TOLERANCE


def test_kronos_compilation_mae_stability(ohlc_windows: List[Tuple[pd.DataFrame, pd.DataFrame]]) -> None:
    eager = _evaluate_kronos_mae(_get_kronos_wrapper(False), ohlc_windows, batch=True)
    compiled = _evaluate_kronos_mae(_get_kronos_wrapper(True), ohlc_windows, batch=True)
    assert abs(eager - compiled) <= KRONOS_COMPILE_TOLERANCE


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_windows(symbol: str, count: int) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    path = DATA_ROOT / f"{symbol}.csv"
    if not path.exists():
        pytest.skip(f"Training data missing for symbol {symbol}: {path}")
    df = pd.read_csv(path)
    required = set(COLUMNS) | {"timestamp"}
    missing = required - set(df.columns)
    if missing:
        pytest.skip(f"Dataset {path} missing required columns: {sorted(missing)}")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)

    windows: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
    end = len(df) - PREDICTION_LENGTH
    while end >= CONTEXT_LENGTH and len(windows) < count:
        start = end - CONTEXT_LENGTH
        history = df.iloc[start:end].copy().reset_index(drop=True)
        target = df.iloc[end : end + PREDICTION_LENGTH].copy().reset_index(drop=True)
        if history.empty or target.empty:
            break
        windows.append((history, target))
        end -= max(PREDICTION_LENGTH, 4)

    if len(windows) < count:
        pytest.skip(
            f"Unable to build {count} OHLC windows for {symbol}; only {len(windows)} available"
        )
    return windows


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - GPU rarely present in CI
        torch.cuda.manual_seed_all(seed)


def _get_toto_pipeline(compiled: bool) -> TotoPipeline:
    cached = _TOTO_PIPELINE_CACHE.get(compiled)
    if cached is not None:
        return cached
    if not TOTO_CACHE_ROOT.exists():
        pytest.skip(f"Precompiled Toto weights missing at {TOTO_CACHE_ROOT}")
    if compiled and not hasattr(torch, "compile"):
        pytest.skip("torch.compile is unavailable; cannot test compiled Toto pipeline")

    pipeline = TotoPipeline.from_pretrained(
        model_id="Datadog/Toto-Open-Base-1.0",
        device_map="cpu",
        torch_dtype=torch.float32,
        amp_dtype=torch.float16,
        amp_autocast=True,
        compile_model=False,
        torch_compile=compiled,
        compile_mode="reduce-overhead" if compiled else None,
        compile_backend="inductor" if compiled else None,
        cache_policy="prefer",
        warmup_sequence=0,
    )
    _TOTO_PIPELINE_CACHE[compiled] = pipeline
    if compiled and not pipeline.compiled:
        pytest.skip("torch.compile fallback prevented Toto compilation in this environment")
    return pipeline


def _get_kronos_wrapper(compiled: bool) -> KronosForecastingWrapper:
    cached = _KRONOS_WRAPPER_CACHE.get(compiled)
    if cached is not None:
        return cached
    if not KRONOS_CACHE_ROOT.exists():
        pytest.skip(f"Precompiled Kronos weights missing at {KRONOS_CACHE_ROOT}")
    if compiled and not hasattr(torch, "compile"):
        pytest.skip("torch.compile is unavailable; cannot test compiled Kronos wrapper")

    wrapper = KronosForecastingWrapper(
        model_name=str(KRONOS_CACHE_ROOT),
        tokenizer_name=str(KRONOS_CACHE_ROOT / "tokenizer"),
        device="cpu",
        max_context=CONTEXT_LENGTH,
        clip=5.0,
        temperature=0.6,
        top_p=0.85,
        top_k=0,
        sample_count=KRONOS_SAMPLE_COUNT,
        cache_dir=str(KRONOS_CACHE_ROOT),
        verbose=False,
        prefer_fp32=True,
        compile=compiled,
        compile_mode="reduce-overhead",
        compile_backend="inductor",
    )
    _KRONOS_WRAPPER_CACHE[compiled] = wrapper
    if compiled and not getattr(wrapper, 'compile', False):
        pytest.skip("Kronos torch.compile fallback prevented compiled wrapper setup")
    return wrapper


def _evaluate_toto_mae(
    pipeline: TotoPipeline,
    windows: Sequence[Tuple[pd.DataFrame, pd.DataFrame]],
    *,
    batch: bool,
) -> float:
    run_averages: List[float] = []
    for repeat in range(TOTO_MAE_REPEATS):
        errors: List[float] = []
        seeds = [seed + repeat * 9973 for seed in TOTO_SEEDS]
        for idx, (history, target) in enumerate(windows):
            seed = seeds[idx % len(seeds)]
            _set_seed(seed)
            contexts = [history[column].to_numpy(dtype=np.float32, copy=False) for column in COLUMNS]
            predictions: List[float] = []
            if batch:
                batched_context = np.stack(contexts, axis=0)
                outputs = pipeline.predict(
                    context=batched_context,
                    prediction_length=PREDICTION_LENGTH,
                    num_samples=TOTO_NUM_SAMPLES,
                    samples_per_batch=TOTO_SAMPLES_PER_BATCH,
                )
                if len(outputs) != len(COLUMNS):
                    raise RuntimeError(
                        f"Expected {len(COLUMNS)} Toto forecasts, received {len(outputs)}"
                    )
                for output in outputs:
                    aggregated = aggregate_with_spec(output.samples, "mean")
                    predictions.append(float(np.asarray(aggregated, dtype=np.float64).ravel()[0]))
            else:
                for column_idx, context in enumerate(contexts):
                    _set_seed(seed + column_idx)
                    outputs = pipeline.predict(
                        context=context,
                        prediction_length=PREDICTION_LENGTH,
                        num_samples=TOTO_NUM_SAMPLES,
                        samples_per_batch=min(TOTO_SAMPLES_PER_BATCH, TOTO_NUM_SAMPLES),
                    )
                    if not outputs:
                        raise RuntimeError("Toto pipeline returned no samples for sequential run")
                    aggregated = aggregate_with_spec(outputs[0].samples, "mean")
                    predictions.append(float(np.asarray(aggregated, dtype=np.float64).ravel()[0]))

            actuals = [float(target[column].iloc[0]) for column in COLUMNS]
            errors.append(float(np.mean([abs(pred - act) for pred, act in zip(predictions, actuals)])))
        run_averages.append(float(np.mean(errors)))
    return float(np.mean(run_averages))


def _evaluate_kronos_mae(
    wrapper: KronosForecastingWrapper,
    windows: Sequence[Tuple[pd.DataFrame, pd.DataFrame]],
    *,
    batch: bool,
) -> float:
    frames: List[pd.DataFrame] = []
    targets: List[List[float]] = []
    for history, target in windows:
        frame = pd.concat([history, target], ignore_index=True)
        frames.append(frame)
        targets.append([float(target[column].iloc[0]) for column in COLUMNS])

    lookback = min(max(1, CONTEXT_LENGTH), wrapper.max_context)

    if batch:
        results = wrapper.predict_series_batch(
            data_frames=frames,
            timestamp_col="timestamp",
            columns=COLUMNS,
            pred_len=PREDICTION_LENGTH,
            lookback=lookback,
            sample_count=KRONOS_SAMPLE_COUNT,
        )
    else:
        results = [
            wrapper.predict_series(
                data=frame,
                timestamp_col="timestamp",
                columns=COLUMNS,
                pred_len=PREDICTION_LENGTH,
                lookback=lookback,
                sample_count=KRONOS_SAMPLE_COUNT,
            )
            for frame in frames
        ]

    errors: List[float] = []
    for payload, actuals in zip(results, targets):
        per_column = []
        for column, actual in zip(COLUMNS, actuals):
            kronos_result = payload[column]
            if kronos_result.absolute.size < PREDICTION_LENGTH:
                raise RuntimeError(
                    f"Kronos forecast for {column} missing horizon entries"
                )
            per_column.append(abs(float(kronos_result.absolute[0]) - actual))
        errors.append(float(np.mean(per_column)))
    return float(np.mean(errors))
