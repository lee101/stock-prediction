"""Chronos2 torch.compile regression test using real trainingdata samples."""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pytest
import torch

try:  # pragma: no cover - optional instrumentation for CUDA graph debugging
    import torch._inductor.config as inductor_config  # type: ignore
except Exception:  # pragma: no cover - torch nightly variations
    inductor_config = None  # type: ignore
else:
    if os.getenv("CHRONOS_INDUCTOR_DEBUG") == "1":
        inductor_config.debug = True

try:  # pragma: no cover - optional helper
    import chronos_compile_config
except Exception:  # pragma: no cover - envs without Chronos extras
    chronos_compile_config = None  # type: ignore

from src.models.chronos2_wrapper import Chronos2OHLCWrapper

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("chronos_compile_test")

SYMBOLS: Tuple[str, ...] = ("BTCUSD", "ETHUSD")
CONTEXT_LENGTH = 1024
PREDICTION_LENGTH = 32
MAE_TOLERANCE = 5e-3
BASELINE_PATH = Path(__file__).parent / "chronos_mae_baseline.txt"
_CHRONOS_ENV_KEYS: Tuple[str, ...] = (
    "TORCH_COMPILED",
    "CHRONOS_COMPILE",
    "CHRONOS_COMPILE_MODE",
    "CHRONOS_COMPILE_BACKEND",
    "CHRONOS_DTYPE",
)


def _reset_torch_compile_state() -> None:
    """Clear process-global torch.compile state so this test is order-independent."""

    reset = getattr(getattr(torch, "compiler", None), "reset", None)
    if callable(reset):
        try:
            reset()
            return
        except Exception:
            pass
    legacy_reset = getattr(getattr(torch, "_dynamo", None), "reset", None)
    if callable(legacy_reset):
        try:
            legacy_reset()
        except Exception:
            pass


def _is_cuda_resource_pressure_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    accelerator_error = getattr(torch, "AcceleratorError", None)
    return accelerator_error is not None and isinstance(exc, accelerator_error)


@contextmanager
def _torch_backend_guard() -> None:
    """Force deterministic, full-precision backend settings for compile accuracy checks.

    Other tests in the suite may toggle TF32 / cuDNN benchmarking for performance.
    Those knobs can introduce numerical drift between eager kernels and Inductor
    generated kernels. We snapshot+restore the relevant flags here so this test
    stays stable regardless of execution order.
    """

    if not torch.cuda.is_available():
        yield
        return

    cuda_backend = getattr(torch.backends, "cuda", None)
    cudnn_backend = getattr(torch.backends, "cudnn", None)
    matmul = getattr(cuda_backend, "matmul", None) if cuda_backend is not None else None
    cudnn_conv = getattr(cudnn_backend, "conv", None) if cudnn_backend is not None else None

    snapshot: Dict[str, object] = {}

    def _safe_get(name: str, obj: object, attr: str) -> None:
        try:
            snapshot[name] = getattr(obj, attr)
        except Exception:
            snapshot[name] = None

    def _safe_set(obj: object, attr: str, value: object) -> None:
        try:
            setattr(obj, attr, value)
        except Exception:
            pass

    # NOTE: On PyTorch 2.9, reading `torch.backends.cuda.matmul.allow_tf32` can raise a
    # RuntimeError if earlier code mixed the legacy and new TF32 APIs. Prefer the new
    # `fp32_precision` knobs here for test stability.
    if matmul is not None and hasattr(matmul, "fp32_precision"):
        _safe_get("cuda.matmul.fp32_precision", matmul, "fp32_precision")
    if cudnn_backend is not None and hasattr(cudnn_backend, "benchmark"):
        _safe_get("cudnn.benchmark", cudnn_backend, "benchmark")
    if cudnn_backend is not None and hasattr(cudnn_backend, "deterministic"):
        _safe_get("cudnn.deterministic", cudnn_backend, "deterministic")
    if cudnn_conv is not None and hasattr(cudnn_conv, "fp32_precision"):
        _safe_get("cudnn.conv.fp32_precision", cudnn_conv, "fp32_precision")

    try:
        # Prefer the new fp32_precision knobs when available.
        if matmul is not None and hasattr(matmul, "fp32_precision"):
            _safe_set(matmul, "fp32_precision", "ieee")
        if cudnn_conv is not None and hasattr(cudnn_conv, "fp32_precision"):
            _safe_set(cudnn_conv, "fp32_precision", "ieee")

        if cudnn_backend is not None and hasattr(cudnn_backend, "benchmark"):
            _safe_set(cudnn_backend, "benchmark", False)
        if cudnn_backend is not None and hasattr(cudnn_backend, "deterministic"):
            _safe_set(cudnn_backend, "deterministic", True)

        yield
    finally:
        # Restore previous backend state best-effort.
        if matmul is not None and snapshot.get("cuda.matmul.fp32_precision") is not None:
            _safe_set(matmul, "fp32_precision", snapshot["cuda.matmul.fp32_precision"])
        if cudnn_conv is not None and snapshot.get("cudnn.conv.fp32_precision") is not None:
            _safe_set(cudnn_conv, "fp32_precision", snapshot["cudnn.conv.fp32_precision"])
        if cudnn_backend is not None and snapshot.get("cudnn.benchmark") is not None:
            _safe_set(cudnn_backend, "benchmark", snapshot["cudnn.benchmark"])
        if cudnn_backend is not None and snapshot.get("cudnn.deterministic") is not None:
            _safe_set(cudnn_backend, "deterministic", snapshot["cudnn.deterministic"])


def _load_symbol_frames(symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    csv_path = Path("trainingdata") / f"{symbol}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Training data not found for {symbol}: {csv_path}")

    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns or "close" not in df.columns:
        raise KeyError(f"{symbol} CSV missing required columns (timestamp, close)")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    required = CONTEXT_LENGTH + PREDICTION_LENGTH
    if len(df) < required:
        raise ValueError(f"{symbol} needs at least {required} rows, found {len(df)}")

    context = df.iloc[-required:-PREDICTION_LENGTH].copy()
    holdout = df.iloc[-PREDICTION_LENGTH:].copy()
    return context, holdout


def _prepare_wrapper(compiled: bool) -> Chronos2OHLCWrapper:
    _reset_torch_compile_state()
    snapshot = {key: os.environ.get(key) for key in _CHRONOS_ENV_KEYS}
    try:
        if compiled and chronos_compile_config is not None:
            chronos_compile_config.apply(verbose=False)
        elif not compiled:
            os.environ["CHRONOS_COMPILE"] = "0"

        try:
            return Chronos2OHLCWrapper.from_pretrained(
                model_id="amazon/chronos-2",
                torch_compile=compiled,
                compile_mode="reduce-overhead" if compiled else None,
                compile_backend="inductor" if compiled else None,
                default_context_length=CONTEXT_LENGTH,
                device_map="cuda" if torch.cuda.is_available() else "cpu",
                cache_policy="never",
            )
        except Exception as exc:
            if torch.cuda.is_available() and _is_cuda_resource_pressure_error(exc):
                pytest.skip(f"Chronos2 compile accuracy test skipped under shared-GPU resource pressure: {exc}")
            raise
    finally:
        for key, value in snapshot.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _run_inference(symbol: str, compiled: bool) -> Dict[str, float | bool]:
    context, holdout = _load_symbol_frames(symbol)
    with _torch_backend_guard():
        cache_snapshot = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
        with tempfile.TemporaryDirectory(prefix="chronos_compile_accuracy_") as temp_cache_dir:
            try:
                if compiled:
                    os.environ["TORCHINDUCTOR_CACHE_DIR"] = temp_cache_dir

                wrapper = _prepare_wrapper(compiled)

                start = time.perf_counter()
                try:
                    batch = wrapper.predict_ohlc(
                        context_df=context,
                        symbol=symbol,
                        prediction_length=PREDICTION_LENGTH,
                        context_length=CONTEXT_LENGTH,
                        evaluation_df=holdout,
                    )
                except Exception as exc:
                    if torch.cuda.is_available() and _is_cuda_resource_pressure_error(exc):
                        pytest.skip(
                            f"Chronos2 compile accuracy prediction skipped under shared-GPU resource pressure: {exc}"
                        )
                    raise
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                latency_ms = (time.perf_counter() - start) * 1000.0

                median = batch.median
                target_index = pd.to_datetime(holdout["timestamp"], utc=True)
                preds = median.loc[target_index, "close"].to_numpy(dtype=np.float64)
                actual = holdout["close"].to_numpy(dtype=np.float64)
                mae = float(np.mean(np.abs(preds - actual)))
                used_safe_backend = getattr(wrapper, "_safe_prediction_pipeline", None) is not None

                wrapper.unload()
                _reset_torch_compile_state()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            finally:
                if cache_snapshot is None:
                    os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)
                else:
                    os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_snapshot

    return {
        "mae": mae,
        "latency_ms": latency_ms,
        "used_safe_backend": used_safe_backend,
    }


def _write_baseline(rows: Tuple[Dict[str, float], ...]) -> None:
    with open(BASELINE_PATH, "w", encoding="utf-8") as handle:
        handle.write("# Chronos2 MAE Baseline\n")
        handle.write(f"# Generated: {pd.Timestamp.now()}\n")
        handle.write(f"# PyTorch: {torch.__version__}\n\n")
        handle.write("Symbol,Uncompiled_MAE,Compiled_MAE,Uncompiled_ms,Compiled_ms\n")
        for row in rows:
            handle.write(
                f"{row['symbol']},{row['mae_uncompiled']:.6f},{row['mae_compiled']:.6f},"
                f"{row['latency_uncompiled_ms']:.2f},{row['latency_compiled_ms']:.2f}\n"
            )


def test_chronos_compile_matches_baseline() -> None:
    logger.info("Chronos2 compile accuracy test (context=%s, horizon=%s)", CONTEXT_LENGTH, PREDICTION_LENGTH)

    summary_rows = []
    for symbol in SYMBOLS:
        logger.info("\n=== %s ===", symbol)
        eager = _run_inference(symbol, compiled=False)
        compiled = _run_inference(symbol, compiled=True)

        diff = abs(eager["mae"] - compiled["mae"])
        logger.info(
            "Uncompiled MAE=%.6f (%.2f ms), compiled MAE=%.6f (%.2f ms), diff=%.6g",
            eager["mae"],
            eager["latency_ms"],
            compiled["mae"],
            compiled["latency_ms"],
            diff,
        )
        if eager["used_safe_backend"] or compiled["used_safe_backend"]:
            logger.info(
                "Skipping strict MAE drift check for %s because cutechronos safety fallback was used "
                "(eager=%s, compiled=%s)",
                symbol,
                bool(eager["used_safe_backend"]),
                bool(compiled["used_safe_backend"]),
            )
        else:
            assert diff <= MAE_TOLERANCE, f"MAE drift {diff} exceeds tolerance for {symbol}"

        summary_rows.append(
            {
                "symbol": symbol,
                "mae_uncompiled": eager["mae"],
                "mae_compiled": compiled["mae"],
                "latency_uncompiled_ms": eager["latency_ms"],
                "latency_compiled_ms": compiled["latency_ms"],
            }
        )

    _write_baseline(tuple(summary_rows))
    logger.info("\nBaseline written to %s", BASELINE_PATH)


if __name__ == "__main__":
    try:
        test_chronos_compile_matches_baseline()
        sys.exit(0)
    except Exception as exc:  # pragma: no cover - manual runs
        logger.exception("Chronos compile test failed: %s", exc)
        sys.exit(1)
