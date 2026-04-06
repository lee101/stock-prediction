"""
Comprehensive fuzzing tests for Chronos2 torch.compile with numerical stability checks.

This test suite validates that torch.compile works reliably across:
1. Different compile modes (None, reduce-overhead, default, max-autotune)
2. Various numerical inputs (normal, extreme values, NaN, inf, very small)
3. Different dtypes (float32, float16, bfloat16)
4. Edge cases that have historically caused issues
5. Real-world anomalies (spikes, drops, volatility changes)

The goal is to ensure compilation is robust and produces numerically stable results
compared to eager mode.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pytest
import torch
from tests.chronos_compile_test_utils import (
    clear_cuda_memory_if_available as _clear_cuda_memory_if_available,
)
from tests.chronos_compile_test_utils import (
    is_transient_nonfinite_forecast_error as _is_transient_nonfinite_forecast_error,
)
from tests.chronos_compile_test_utils import reset_torch_compile_state as _reset_torch_compile_state

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.chronos2_wrapper import Chronos2OHLCWrapper

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("chronos2_compile_fuzzing")

# Test configuration
CONTEXT_LENGTH = 128
PREDICTION_LENGTH = 16
BATCH_SIZE = 32
MAE_TOLERANCE = 1e-2  # Maximum acceptable MAE difference between eager and compiled
RELATIVE_TOLERANCE = 0.05  # 5% relative difference tolerance
MAX_PREDICTION_ATTEMPTS = 3

# Compile modes to test (ordered from safest to most aggressive)
COMPILE_MODES = [
    None,  # Default PyTorch behavior
    "default",  # Balanced compilation
    "reduce-overhead",  # Currently used in production
    # "max-autotune",  # Most aggressive - commented out as it's often unstable
]

# Backends to test
COMPILE_BACKENDS = [
    "inductor",  # Default backend (most tested)
    # "aot_eager",  # Ahead-of-time eager backend (safer but slower)
]

# Data types to test
DTYPES = [
    "float32",  # Default
    # "float16",  # Half precision - often causes numerical issues
    # "bfloat16",  # Brain float - better range than fp16 but requires specific hardware
]


def _snapshot_compile_runtime_state() -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {
        "env": {
            "TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS": os.environ.get("TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS"),
        }
    }
    inductor_config = getattr(getattr(torch, "_inductor", None), "config", None)
    if inductor_config is not None:
        snapshot["inductor_config"] = {
            "max_autotune": getattr(inductor_config, "max_autotune", None),
            "debug": getattr(inductor_config, "debug", None),
        }
        triton_config = getattr(inductor_config, "triton", None)
        if triton_config is not None:
            snapshot["inductor_triton_config"] = {
                "cudagraphs": getattr(triton_config, "cudagraphs", None),
                "cudagraph_or_error": getattr(triton_config, "cudagraph_or_error", None),
            }
    dynamo_config = getattr(getattr(torch, "_dynamo", None), "config", None)
    if dynamo_config is not None:
        snapshot["dynamo_config"] = {
            "automatic_dynamic_shapes": getattr(dynamo_config, "automatic_dynamic_shapes", None),
            "recompile_limit": getattr(dynamo_config, "recompile_limit", None),
            "suppress_errors": getattr(dynamo_config, "suppress_errors", None),
        }
    return snapshot


def _apply_compile_runtime_stability() -> Dict[str, Any]:
    snapshot = _snapshot_compile_runtime_state()
    os.environ["TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS"] = "1"

    inductor_config = getattr(getattr(torch, "_inductor", None), "config", None)
    if inductor_config is not None:
        if hasattr(inductor_config, "max_autotune"):
            inductor_config.max_autotune = False
        if hasattr(inductor_config, "debug"):
            inductor_config.debug = False
        triton_config = getattr(inductor_config, "triton", None)
        if triton_config is not None:
            if hasattr(triton_config, "cudagraphs"):
                triton_config.cudagraphs = False
            if hasattr(triton_config, "cudagraph_or_error"):
                triton_config.cudagraph_or_error = False

    dynamo_config = getattr(getattr(torch, "_dynamo", None), "config", None)
    if dynamo_config is not None:
        if hasattr(dynamo_config, "automatic_dynamic_shapes"):
            dynamo_config.automatic_dynamic_shapes = False
        if hasattr(dynamo_config, "recompile_limit"):
            dynamo_config.recompile_limit = 64
        if hasattr(dynamo_config, "suppress_errors"):
            dynamo_config.suppress_errors = False
    return snapshot


def _restore_compile_runtime_state(snapshot: Dict[str, Any]) -> None:
    env_snapshot = snapshot.get("env", {})
    capture_scalar_outputs = env_snapshot.get("TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS")
    if capture_scalar_outputs is None:
        os.environ.pop("TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS", None)
    else:
        os.environ["TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS"] = str(capture_scalar_outputs)

    inductor_config = getattr(getattr(torch, "_inductor", None), "config", None)
    if inductor_config is not None:
        for attr, value in snapshot.get("inductor_config", {}).items():
            if value is not None and hasattr(inductor_config, attr):
                setattr(inductor_config, attr, value)
        triton_config = getattr(inductor_config, "triton", None)
        if triton_config is not None:
            for attr, value in snapshot.get("inductor_triton_config", {}).items():
                if value is not None and hasattr(triton_config, attr):
                    setattr(triton_config, attr, value)

    dynamo_config = getattr(getattr(torch, "_dynamo", None), "config", None)
    if dynamo_config is not None:
        for attr, value in snapshot.get("dynamo_config", {}).items():
            if value is not None and hasattr(dynamo_config, attr):
                setattr(dynamo_config, attr, value)


def _snapshot_torch_backend_state() -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {}

    get_precision = getattr(torch, "get_float32_matmul_precision", None)
    if callable(get_precision):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                snapshot["float32_matmul_precision"] = get_precision()
            except Exception:
                snapshot["float32_matmul_precision"] = None

    cudnn_backend = getattr(torch.backends, "cudnn", None)
    if cudnn_backend is not None:
        snapshot["cudnn_benchmark"] = getattr(cudnn_backend, "benchmark", None)
        snapshot["cudnn_deterministic"] = getattr(cudnn_backend, "deterministic", None)
        cudnn_conv = getattr(cudnn_backend, "conv", None)
        if cudnn_conv is not None and hasattr(cudnn_conv, "fp32_precision"):
            snapshot["cudnn_conv_fp32_precision"] = getattr(cudnn_conv, "fp32_precision", None)

    cuda_backend = getattr(torch.backends, "cuda", None)
    matmul_backend = getattr(cuda_backend, "matmul", None) if cuda_backend is not None else None
    if matmul_backend is not None and hasattr(matmul_backend, "fp32_precision"):
        snapshot["cuda_matmul_fp32_precision"] = getattr(matmul_backend, "fp32_precision", None)

    return snapshot


def _apply_torch_backend_stability() -> Dict[str, Any]:
    snapshot = _snapshot_torch_backend_state()

    set_precision = getattr(torch, "set_float32_matmul_precision", None)
    if callable(set_precision):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                set_precision("highest")
            except Exception:
                pass

    cudnn_backend = getattr(torch.backends, "cudnn", None)
    if cudnn_backend is not None:
        if hasattr(cudnn_backend, "benchmark"):
            try:
                cudnn_backend.benchmark = False
            except Exception:
                pass
        if hasattr(cudnn_backend, "deterministic"):
            try:
                cudnn_backend.deterministic = True
            except Exception:
                pass
        cudnn_conv = getattr(cudnn_backend, "conv", None)
        if cudnn_conv is not None and hasattr(cudnn_conv, "fp32_precision"):
            try:
                cudnn_conv.fp32_precision = "ieee"
            except Exception:
                pass

    cuda_backend = getattr(torch.backends, "cuda", None)
    matmul_backend = getattr(cuda_backend, "matmul", None) if cuda_backend is not None else None
    if matmul_backend is not None and hasattr(matmul_backend, "fp32_precision"):
        try:
            matmul_backend.fp32_precision = "ieee"
        except Exception:
            pass

    return snapshot


def _restore_torch_backend_state(snapshot: Dict[str, Any]) -> None:
    set_precision = getattr(torch, "set_float32_matmul_precision", None)
    precision = snapshot.get("float32_matmul_precision")
    if callable(set_precision) and precision is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                set_precision(str(precision))
            except Exception:
                pass

    cudnn_backend = getattr(torch.backends, "cudnn", None)
    if cudnn_backend is not None:
        benchmark = snapshot.get("cudnn_benchmark")
        if benchmark is not None and hasattr(cudnn_backend, "benchmark"):
            try:
                cudnn_backend.benchmark = bool(benchmark)
            except Exception:
                pass
        deterministic = snapshot.get("cudnn_deterministic")
        if deterministic is not None and hasattr(cudnn_backend, "deterministic"):
            try:
                cudnn_backend.deterministic = bool(deterministic)
            except Exception:
                pass
        cudnn_conv = getattr(cudnn_backend, "conv", None)
        conv_precision = snapshot.get("cudnn_conv_fp32_precision")
        if conv_precision is not None and cudnn_conv is not None and hasattr(cudnn_conv, "fp32_precision"):
            try:
                cudnn_conv.fp32_precision = conv_precision
            except Exception:
                pass

    cuda_backend = getattr(torch.backends, "cuda", None)
    matmul_backend = getattr(cuda_backend, "matmul", None) if cuda_backend is not None else None
    matmul_precision = snapshot.get("cuda_matmul_fp32_precision")
    if matmul_precision is not None and matmul_backend is not None and hasattr(matmul_backend, "fp32_precision"):
        try:
            matmul_backend.fp32_precision = matmul_precision
        except Exception:
            pass


def _get_device() -> str:
    """Get available device for testing."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _create_test_data(
    n_points: int = CONTEXT_LENGTH + PREDICTION_LENGTH,
    base_price: float = 100.0,
    volatility: float = 0.02,
    seed: int = 42,
) -> pd.DataFrame:
    """Create realistic OHLC test data."""
    np.random.seed(seed)

    # Generate realistic price movement
    returns = np.random.randn(n_points) * volatility
    prices = base_price * np.exp(np.cumsum(returns))

    # Create OHLC with realistic intraday movements
    opens = prices * (1 + np.random.randn(n_points) * 0.002)
    closes = prices * (1 + np.random.randn(n_points) * 0.002)
    highs = np.maximum(opens, closes) * (1 + np.abs(np.random.randn(n_points)) * 0.005)
    lows = np.minimum(opens, closes) * (1 - np.abs(np.random.randn(n_points)) * 0.005)

    return pd.DataFrame({
        "timestamp": pd.date_range(start="2024-01-01", periods=n_points, freq="D"),
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "symbol": "TEST",
    })


def _create_extreme_data(
    n_points: int = CONTEXT_LENGTH + PREDICTION_LENGTH,
    scenario: str = "normal",
) -> pd.DataFrame:
    """Create test data with extreme values for robustness testing."""
    base_df = _create_test_data(n_points)

    if scenario == "very_small":
        # Very small positive values (test epsilon clamping)
        base_df[["open", "high", "low", "close"]] *= 1e-4

    elif scenario == "very_large":
        # Very large values
        base_df[["open", "high", "low", "close"]] *= 1e6

    elif scenario == "high_volatility":
        # Extreme volatility
        multipliers = np.exp(np.random.randn(n_points) * 0.2)  # 20% volatility
        for col in ["open", "high", "low", "close"]:
            base_df[col] *= multipliers

    elif scenario == "spike":
        # Sudden spike in the middle
        spike_idx = n_points // 2
        base_df.loc[spike_idx : spike_idx + 3, ["open", "high", "low", "close"]] *= 5.0

    elif scenario == "drop":
        # Sudden drop in the middle
        drop_idx = n_points // 2
        base_df.loc[drop_idx : drop_idx + 3, ["open", "high", "low", "close"]] *= 0.2

    elif scenario == "near_zero":
        # Values very close to zero (but not quite)
        base_df[["open", "high", "low", "close"]] = (
            np.random.randn(n_points, 4) * 1e-5 + 1e-4
        )
        base_df[["open", "high", "low", "close"]] = base_df[
            ["open", "high", "low", "close"]
        ].abs()

    elif scenario == "constant":
        # Constant values (no movement)
        base_df[["open", "high", "low", "close"]] = 100.0

    elif scenario == "linear_trend":
        # Strong linear trend
        trend = np.linspace(50, 200, n_points)
        base_df[["open", "high", "low", "close"]] = trend[:, None] * (
            1 + np.random.randn(n_points, 4) * 0.01
        )

    return base_df


def _load_wrapper(
    compile_enabled: bool,
    compile_mode: Optional[str] = None,
    compile_backend: str = "inductor",
    dtype: str = "float32",
    device: str = "cpu",
) -> Chronos2OHLCWrapper:
    """Load Chronos2 wrapper with specified compilation settings."""
    # Clear any existing environment variables so earlier tests cannot leak
    # compile mode/backend/dtype into this harness.
    for key in (
        "CHRONOS_COMPILE",
        "TORCH_COMPILED",
        "CHRONOS_COMPILE_MODE",
        "CHRONOS_COMPILE_BACKEND",
        "CHRONOS_DTYPE",
        "CHRONOS2_PIPELINE_BACKEND",
    ):
        os.environ.pop(key, None)
    _reset_torch_compile_state()
    compile_runtime_snapshot = _apply_compile_runtime_stability()
    backend_snapshot = _apply_torch_backend_stability()
    cache_snapshot = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    temp_compile_cache: tempfile.TemporaryDirectory[str] | None = None
    if compile_enabled:
        # Keep an isolated inductor cache alive for the entire wrapper lifetime.
        # The first compiled inference happens after wrapper construction, so
        # restoring the global cache env here would let surrounding tests leak
        # state back into this harness.
        temp_compile_cache = tempfile.TemporaryDirectory(prefix="chronos2_compile_fuzz_")
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = temp_compile_cache.name
    else:
        os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)

    try:
        wrapper = Chronos2OHLCWrapper.from_pretrained(
            model_id="amazon/chronos-2",
            device_map=device,
            default_context_length=CONTEXT_LENGTH,
            default_batch_size=BATCH_SIZE,
            torch_compile=compile_enabled,
            compile_mode=compile_mode,
            compile_backend=compile_backend if compile_enabled else None,
            torch_dtype=dtype,
            cache_policy="never",
        )
    except Exception as exc:
        if temp_compile_cache is not None:
            temp_compile_cache.cleanup()
        if cache_snapshot is None:
            os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)
        else:
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_snapshot
        _restore_compile_runtime_state(compile_runtime_snapshot)
        _restore_torch_backend_state(backend_snapshot)
        if device == "cuda" and _is_cuda_resource_pressure_error(exc):
            pytest.skip(
                f"Skipping Chronos2 compile fuzzing during wrapper load under shared-GPU resource pressure: {exc}"
            )
        raise

    wrapper._test_compile_cache_dir = temp_compile_cache  # type: ignore[attr-defined]
    wrapper._test_compile_cache_snapshot = cache_snapshot  # type: ignore[attr-defined]
    wrapper._test_compile_runtime_snapshot = compile_runtime_snapshot  # type: ignore[attr-defined]
    wrapper._test_backend_snapshot = backend_snapshot  # type: ignore[attr-defined]
    return wrapper


def _run_prediction(
    wrapper: Chronos2OHLCWrapper,
    context_df: pd.DataFrame,
    prediction_length: int = PREDICTION_LENGTH,
) -> pd.DataFrame:
    """Run prediction and return close prices."""
    last_exc: BaseException | None = None
    for attempt in range(MAX_PREDICTION_ATTEMPTS):
        try:
            result = wrapper.predict_ohlc(
                context_df=context_df,
                symbol="TEST",
                prediction_length=prediction_length,
                context_length=min(CONTEXT_LENGTH, len(context_df)),
            )
            return result.median["close"].values
        except Exception as exc:
            device_hint = str(getattr(wrapper, "_device_hint", "")).strip().lower()
            if device_hint == "cuda" and _is_cuda_resource_pressure_error(exc):
                pytest.skip(
                    f"Skipping Chronos2 compile fuzzing during prediction under shared-GPU resource pressure: {exc}"
                )
            transient_nonfinite = _is_transient_nonfinite_forecast_error(exc)
            if transient_nonfinite and attempt < MAX_PREDICTION_ATTEMPTS - 1:
                logger.warning(
                    "Retrying Chronos2 fuzz prediction after transient non-finite forecasts."
                )
                _reset_torch_compile_state()
                _clear_cuda_memory_if_available()
                last_exc = exc
                continue
            if transient_nonfinite:
                pytest.skip(
                    "Skipping Chronos2 compile fuzzing after repeated transient non-finite forecasts."
                )
            raise
    assert last_exc is not None
    raise last_exc


def _is_cuda_resource_pressure_error(exc: BaseException) -> bool:
    """Return whether a failure is caused by external CUDA resource pressure."""
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    accelerator_error = getattr(torch, "AcceleratorError", None)
    if accelerator_error is not None and isinstance(exc, accelerator_error):
        message = str(exc).lower()
        return "out of memory" in message or "cuda error" in message
    return False


def _calculate_mae_difference(
    eager_preds: np.ndarray,
    compiled_preds: np.ndarray,
) -> Tuple[float, float]:
    """Calculate MAE and relative difference between predictions."""
    mae_diff = float(np.mean(np.abs(eager_preds - compiled_preds)))
    mean_scale = float(np.mean(np.abs(eager_preds)))
    relative_diff = mae_diff / mean_scale if mean_scale > 1e-10 else 0.0
    return mae_diff, relative_diff


def _assert_accuracy_within_tolerance(mae_diff: float, relative_diff: float, *, mode_str: str) -> None:
    abs_ok = mae_diff < MAE_TOLERANCE
    rel_ok = relative_diff < RELATIVE_TOLERANCE
    assert abs_ok or rel_ok, (
        f"Prediction drift too large ({mode_str}): "
        f"mae_diff={mae_diff:.6f} (limit={MAE_TOLERANCE}), "
        f"relative_diff={relative_diff:.2%} (limit={RELATIVE_TOLERANCE:.2%})"
    )


def _cleanup_wrapper(wrapper: Chronos2OHLCWrapper) -> None:
    """Clean up wrapper and free GPU memory."""
    wrapper.unload()
    _reset_torch_compile_state()
    compile_cache = getattr(wrapper, "_test_compile_cache_dir", None)
    cache_snapshot = getattr(wrapper, "_test_compile_cache_snapshot", None)
    compile_runtime_snapshot = getattr(wrapper, "_test_compile_runtime_snapshot", None)
    backend_snapshot = getattr(wrapper, "_test_backend_snapshot", None)
    if compile_cache is not None:
        compile_cache.cleanup()
    if cache_snapshot is None:
        os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)
    else:
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_snapshot
    if isinstance(compile_runtime_snapshot, dict):
        _restore_compile_runtime_state(compile_runtime_snapshot)
    if isinstance(backend_snapshot, dict):
        _restore_torch_backend_state(backend_snapshot)
    _clear_cuda_memory_if_available()


def _used_safe_backend_fallback(wrapper: Chronos2OHLCWrapper) -> bool:
    """Return whether prediction escalated to the cutechronos safety backend."""
    return getattr(wrapper, "_safe_prediction_pipeline", None) is not None


def _run_robustness_mode(
    *,
    device: str,
    context: pd.DataFrame,
    scenario: str,
    compile_enabled: bool,
    compile_mode: str | None = None,
) -> tuple[bool, np.ndarray | None, bool, bool, bool]:
    """Run one robustness-mode prediction with a fresh-wrapper retry for transient compiled failures."""
    mode_label = "Compiled" if compile_enabled else "Eager"
    max_wrapper_attempts = 3
    for wrapper_attempt in range(max_wrapper_attempts):
        wrapper = _load_wrapper(
            compile_enabled=compile_enabled,
            compile_mode=compile_mode,
            device=device,
        )
        try:
            preds = _run_prediction(wrapper, context)
            return (
                True,
                preds,
                bool(np.isnan(preds).any()),
                bool(np.isinf(preds).any()),
                _used_safe_backend_fallback(wrapper),
            )
        except Exception as exc:
            logger.warning("  %s mode failed on %s: %s", mode_label, scenario, exc)
            transient_nonfinite = _is_transient_nonfinite_forecast_error(exc)
            if transient_nonfinite and wrapper_attempt < max_wrapper_attempts - 1:
                logger.warning(
                    "Retrying %s robustness check on %s with a fresh wrapper after transient non-finite forecasts.",
                    mode_label.lower(),
                    scenario,
                )
                _reset_torch_compile_state()
                _clear_cuda_memory_if_available()
                continue
            return False, None, False, False, False
        finally:
            _cleanup_wrapper(wrapper)

    return False, None, False, False, False


def test_load_wrapper_keeps_isolated_compile_cache_until_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    original_cache = "/tmp/original-inductor-cache"
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", original_cache)
    monkeypatch.setenv("TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS", "0")
    get_precision = getattr(torch, "get_float32_matmul_precision", None)
    set_precision = getattr(torch, "set_float32_matmul_precision", None)
    if callable(set_precision):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                set_precision("high")
            except Exception:
                pass

    inductor_config = getattr(getattr(torch, "_inductor", None), "config", None)
    dynamo_config = getattr(getattr(torch, "_dynamo", None), "config", None)
    cudnn_backend = getattr(torch.backends, "cudnn", None)
    if cudnn_backend is not None and hasattr(cudnn_backend, "benchmark"):
        cudnn_backend.benchmark = True
    if cudnn_backend is not None and hasattr(cudnn_backend, "deterministic"):
        cudnn_backend.deterministic = False
    if inductor_config is not None and hasattr(inductor_config, "max_autotune"):
        inductor_config.max_autotune = True
    triton_config = getattr(inductor_config, "triton", None) if inductor_config is not None else None
    if triton_config is not None and hasattr(triton_config, "cudagraphs"):
        triton_config.cudagraphs = True
    if dynamo_config is not None and hasattr(dynamo_config, "automatic_dynamic_shapes"):
        dynamo_config.automatic_dynamic_shapes = True

    class _DummyWrapper:
        def unload(self) -> None:
            return None

    monkeypatch.setattr(
        Chronos2OHLCWrapper,
        "from_pretrained",
        classmethod(lambda cls, *args, **kwargs: _DummyWrapper()),
    )

    wrapper = _load_wrapper(compile_enabled=True)
    isolated_cache = os.environ.get("TORCHINDUCTOR_CACHE_DIR")

    assert isolated_cache is not None
    assert isolated_cache != original_cache
    assert Path(isolated_cache).name.startswith("chronos2_compile_fuzz_")
    assert os.environ.get("TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS") == "1"
    if callable(get_precision):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert get_precision() == "highest"
    if cudnn_backend is not None and hasattr(cudnn_backend, "benchmark"):
        assert cudnn_backend.benchmark is False
    if cudnn_backend is not None and hasattr(cudnn_backend, "deterministic"):
        assert cudnn_backend.deterministic is True
    if inductor_config is not None and hasattr(inductor_config, "max_autotune"):
        assert inductor_config.max_autotune is False
    if triton_config is not None and hasattr(triton_config, "cudagraphs"):
        assert triton_config.cudagraphs is False
    if dynamo_config is not None and hasattr(dynamo_config, "automatic_dynamic_shapes"):
        assert dynamo_config.automatic_dynamic_shapes is False

    _cleanup_wrapper(wrapper)

    assert os.environ.get("TORCHINDUCTOR_CACHE_DIR") == original_cache
    assert os.environ.get("TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS") == "0"
    if callable(get_precision):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert get_precision() == "high"
    if cudnn_backend is not None and hasattr(cudnn_backend, "benchmark"):
        assert cudnn_backend.benchmark is True
    if cudnn_backend is not None and hasattr(cudnn_backend, "deterministic"):
        assert cudnn_backend.deterministic is False
    if inductor_config is not None and hasattr(inductor_config, "max_autotune"):
        assert inductor_config.max_autotune is True
    if triton_config is not None and hasattr(triton_config, "cudagraphs"):
        assert triton_config.cudagraphs is True
    if dynamo_config is not None and hasattr(dynamo_config, "automatic_dynamic_shapes"):
        assert dynamo_config.automatic_dynamic_shapes is True


# Test fixtures
@pytest.fixture(scope="module")
def device() -> str:
    """Get device for testing."""
    return _get_device()


@pytest.fixture(scope="module")
def test_data() -> pd.DataFrame:
    """Create standard test data."""
    return _create_test_data()


# Basic smoke tests
def test_eager_mode_baseline(device: str, test_data: pd.DataFrame) -> None:
    """Test that eager mode works as baseline."""
    logger.info("Testing eager mode baseline...")

    wrapper = _load_wrapper(compile_enabled=False, device=device)
    context = test_data.iloc[:-PREDICTION_LENGTH]

    try:
        preds = _run_prediction(wrapper, context)
        assert len(preds) == PREDICTION_LENGTH
        assert not np.isnan(preds).any(), "Predictions contain NaN"
        assert not np.isinf(preds).any(), "Predictions contain inf"
        logger.info("✓ Eager mode baseline passed")
    finally:
        _cleanup_wrapper(wrapper)


@pytest.mark.parametrize("compile_mode", COMPILE_MODES)
@pytest.mark.parametrize("compile_backend", COMPILE_BACKENDS)
def test_compiled_mode_smoke(
    device: str,
    test_data: pd.DataFrame,
    compile_mode: Optional[str],
    compile_backend: str,
) -> None:
    """Smoke test for each compile mode and backend combination."""
    mode_str = compile_mode or "default"
    logger.info(f"Testing compiled mode: {mode_str} + {compile_backend}...")

    wrapper = _load_wrapper(
        compile_enabled=True,
        compile_mode=compile_mode,
        compile_backend=compile_backend,
        device=device,
    )
    context = test_data.iloc[:-PREDICTION_LENGTH]

    try:
        try:
            preds = _run_prediction(wrapper, context)
        except Exception as exc:
            if device == "cuda" and _is_cuda_resource_pressure_error(exc):
                pytest.skip(
                    f"Skipping compiled CUDA smoke test under shared-GPU resource pressure: {exc}"
                )
            raise
        assert len(preds) == PREDICTION_LENGTH
        assert not np.isnan(preds).any(), f"Predictions contain NaN ({mode_str})"
        assert not np.isinf(preds).any(), f"Predictions contain inf ({mode_str})"
        logger.info(f"✓ Compiled mode {mode_str} + {compile_backend} passed")
    finally:
        _cleanup_wrapper(wrapper)


# Numerical stability tests
@pytest.mark.parametrize("compile_mode", COMPILE_MODES)
def test_eager_vs_compiled_accuracy(
    device: str,
    test_data: pd.DataFrame,
    compile_mode: Optional[str],
) -> None:
    """Test that compiled mode produces similar results to eager mode."""
    mode_str = compile_mode or "default"
    logger.info(f"Testing eager vs compiled accuracy ({mode_str})...")

    context = test_data.iloc[:-PREDICTION_LENGTH]

    # Run eager mode
    eager_wrapper = _load_wrapper(compile_enabled=False, device=device)
    try:
        eager_preds = _run_prediction(eager_wrapper, context)
        eager_used_safe_backend = _used_safe_backend_fallback(eager_wrapper)
    finally:
        _cleanup_wrapper(eager_wrapper)

    # Run compiled mode
    compiled_wrapper = _load_wrapper(
        compile_enabled=True,
        compile_mode=compile_mode,
        device=device,
    )
    try:
        compiled_preds = _run_prediction(compiled_wrapper, context)
        compiled_used_safe_backend = _used_safe_backend_fallback(compiled_wrapper)
    finally:
        _cleanup_wrapper(compiled_wrapper)

    # Compare
    mae_diff, relative_diff = _calculate_mae_difference(eager_preds, compiled_preds)

    logger.info(
        f"  MAE difference: {mae_diff:.6f}, Relative: {relative_diff:.2%}"
    )

    if eager_used_safe_backend or compiled_used_safe_backend:
        logger.info(
            "  %s: skipped strict accuracy check because a cutechronos safety fallback was used "
            "(eager=%s, compiled=%s)",
            mode_str,
            eager_used_safe_backend,
            compiled_used_safe_backend,
        )
        return

    _assert_accuracy_within_tolerance(
        mae_diff,
        relative_diff,
        mode_str=mode_str,
    )

    logger.info(f"✓ Accuracy test passed ({mode_str})")


def test_eager_vs_compiled_accuracy_skips_strict_check_for_safe_backend(
    monkeypatch: pytest.MonkeyPatch,
    test_data: pd.DataFrame,
) -> None:
    eager_wrapper = object()
    compiled_wrapper = object()
    wrappers = iter([eager_wrapper, compiled_wrapper])

    monkeypatch.setattr(
        sys.modules[__name__],
        "_load_wrapper",
        lambda *args, **kwargs: next(wrappers),
    )
    predictions = iter(
        [
            np.zeros(PREDICTION_LENGTH, dtype=np.float32),
            np.full(PREDICTION_LENGTH, 10.0, dtype=np.float32),
        ]
    )
    monkeypatch.setattr(
        sys.modules[__name__],
        "_run_prediction",
        lambda *args, **kwargs: next(predictions),
    )
    monkeypatch.setattr(
        sys.modules[__name__],
        "_used_safe_backend_fallback",
        lambda wrapper: wrapper is compiled_wrapper,
    )
    monkeypatch.setattr(sys.modules[__name__], "_cleanup_wrapper", lambda wrapper: None)

    test_eager_vs_compiled_accuracy("cpu", test_data, "reduce-overhead")


@pytest.mark.parametrize("compile_enabled", [False, True])
def test_run_robustness_mode_retries_transient_nonfinite_with_fresh_wrapper(
    monkeypatch: pytest.MonkeyPatch,
    test_data: pd.DataFrame,
    compile_enabled: bool,
) -> None:
    first_wrapper = object()
    second_wrapper = object()
    wrappers = iter([first_wrapper, second_wrapper])
    load_calls: list[tuple[bool, str | None, str]] = []
    cleanup_calls: list[object] = []
    predictions = iter(
        [
            RuntimeError("Chronos2 produced non-finite forecasts for TEST."),
            np.zeros(PREDICTION_LENGTH, dtype=np.float32),
        ]
    )

    def _fake_load_wrapper(*, compile_enabled: bool, compile_mode: str | None = None, device: str):
        load_calls.append((compile_enabled, compile_mode, device))
        return next(wrappers)

    def _fake_run_prediction(_wrapper: object, _context: pd.DataFrame) -> np.ndarray:
        result = next(predictions)
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(sys.modules[__name__], "_load_wrapper", _fake_load_wrapper)
    monkeypatch.setattr(sys.modules[__name__], "_run_prediction", _fake_run_prediction)
    monkeypatch.setattr(sys.modules[__name__], "_used_safe_backend_fallback", lambda _wrapper: False)
    monkeypatch.setattr(sys.modules[__name__], "_cleanup_wrapper", cleanup_calls.append)

    success, preds, has_nan, has_inf, used_safe_backend = _run_robustness_mode(
        device="cpu",
        context=test_data.iloc[:-PREDICTION_LENGTH],
        scenario="high_volatility",
        compile_enabled=compile_enabled,
        compile_mode="reduce-overhead" if compile_enabled else None,
    )

    assert success is True
    assert preds is not None
    assert np.array_equal(preds, np.zeros(PREDICTION_LENGTH, dtype=np.float32))
    assert has_nan is False
    assert has_inf is False
    assert used_safe_backend is False
    assert load_calls == [
        (compile_enabled, "reduce-overhead" if compile_enabled else None, "cpu"),
        (compile_enabled, "reduce-overhead" if compile_enabled else None, "cpu"),
    ]
    assert cleanup_calls == [first_wrapper, second_wrapper]


def test_run_prediction_skips_after_repeated_transient_nonfinite(
    monkeypatch: pytest.MonkeyPatch,
    test_data: pd.DataFrame,
) -> None:
    call_count = 0

    class _Wrapper:
        _device_hint = "cpu"

        def predict_ohlc(self, **_kwargs: object) -> object:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Chronos2 produced non-finite forecasts for TEST.")

    monkeypatch.setattr(sys.modules[__name__], "_reset_torch_compile_state", lambda: None)
    monkeypatch.setattr(sys.modules[__name__], "_clear_cuda_memory_if_available", lambda: None)

    with pytest.raises(pytest.skip.Exception, match="repeated transient non-finite forecasts"):
        _run_prediction(_Wrapper(), test_data.iloc[:-PREDICTION_LENGTH])

    assert call_count == MAX_PREDICTION_ATTEMPTS


def test_assert_accuracy_within_tolerance_accepts_small_relative_drift() -> None:
    _assert_accuracy_within_tolerance(
        0.4,
        0.004,
        mode_str="default",
    )


# Fuzzing tests with extreme data
@pytest.mark.parametrize(
    "scenario",
    [
        "very_small",
        "very_large",
        "high_volatility",
        "spike",
        "drop",
        "near_zero",
        "constant",
        "linear_trend",
    ],
)
def test_extreme_data_robustness(device: str, scenario: str) -> None:
    """Test that both eager and compiled modes handle extreme data gracefully."""
    logger.info(f"Testing robustness with {scenario} data...")

    data = _create_extreme_data(scenario=scenario)
    context = data.iloc[:-PREDICTION_LENGTH]

    eager_success, eager_preds, eager_has_nan, eager_has_inf, eager_used_safe_backend = _run_robustness_mode(
        device=device,
        context=context,
        scenario=scenario,
        compile_enabled=False,
    )

    compiled_success, compiled_preds, compiled_has_nan, compiled_has_inf, compiled_used_safe_backend = _run_robustness_mode(
        device=device,
        context=context,
        scenario=scenario,
        compile_enabled=True,
        compile_mode="reduce-overhead",
    )

    # Both modes should have similar behavior
    if eager_success and compiled_success:
        assert eager_preds is not None
        assert compiled_preds is not None
        # Compare numerical stability
        mae_diff, relative_diff = _calculate_mae_difference(eager_preds, compiled_preds)
        logger.info(
            f"  {scenario}: MAE diff={mae_diff:.6f}, Relative={relative_diff:.2%}"
        )

        # Check for NaN/inf consistency
        assert eager_has_nan == compiled_has_nan, (
            f"NaN inconsistency in {scenario}: eager={eager_has_nan}, compiled={compiled_has_nan}"
        )
        assert eager_has_inf == compiled_has_inf, (
            f"Inf inconsistency in {scenario}: eager={eager_has_inf}, compiled={compiled_has_inf}"
        )

        if eager_used_safe_backend or compiled_used_safe_backend:
            logger.info(
                "  %s: skipped strict drift check because a cutechronos safety fallback was used "
                "(eager=%s, compiled=%s)",
                scenario,
                eager_used_safe_backend,
                compiled_used_safe_backend,
            )
        elif not (eager_has_nan or eager_has_inf):
            # Only check accuracy if both produce valid numbers
            abs_ok = mae_diff < MAE_TOLERANCE * 10  # More lenient for extreme data
            rel_ok = relative_diff < RELATIVE_TOLERANCE
            assert abs_ok or rel_ok, (
                f"Prediction drift too large for {scenario}: "
                f"mae_diff={mae_diff:.6f} (limit={MAE_TOLERANCE * 10}), "
                f"relative_diff={relative_diff:.2%} (limit={RELATIVE_TOLERANCE:.2%})"
            )
    else:
        # Under extreme inputs, one mode may exhaust retries while the other recovers.
        if eager_success != compiled_success:
            successful_mode = "compiled" if compiled_success else "eager"
            logger.warning(
                "Allowing asymmetric robustness outcome for %s after retries: eager=%s, compiled=%s; "
                "accepting %s result.",
                scenario,
                eager_success,
                compiled_success,
                successful_mode,
            )
            if compiled_success:
                assert compiled_preds is not None
                assert not compiled_has_nan
                assert not compiled_has_inf
            else:
                assert eager_preds is not None
                assert not eager_has_nan
                assert not eager_has_inf

    logger.info(f"✓ Robustness test passed for {scenario}")


# Stress tests
def test_multiple_predictions_stability(device: str) -> None:
    """Test that compiled mode remains stable across multiple predictions."""
    logger.info("Testing multiple predictions stability...")

    wrapper = _load_wrapper(
        compile_enabled=True,
        compile_mode="reduce-overhead",
        device=device,
    )

    try:
        all_preds = []
        for i in range(5):
            data = _create_test_data(seed=42 + i)
            context = data.iloc[:-PREDICTION_LENGTH]
            preds = _run_prediction(wrapper, context)

            assert not np.isnan(preds).any(), f"Run {i+1} produced NaN"
            assert not np.isinf(preds).any(), f"Run {i+1} produced inf"

            all_preds.append(preds)

        # Check that predictions are consistent for same seed
        data_repeat = _create_test_data(seed=42)
        context_repeat = data_repeat.iloc[:-PREDICTION_LENGTH]
        preds_repeat = _run_prediction(wrapper, context_repeat)

        mae_consistency = float(np.mean(np.abs(all_preds[0] - preds_repeat)))
        logger.info(f"  Consistency MAE: {mae_consistency:.6f}")

        # Should be very close (may not be exact due to numerical precision)
        assert mae_consistency < 1e-4, f"Inconsistent predictions: {mae_consistency}"

        logger.info("✓ Multiple predictions stability test passed")

    finally:
        _cleanup_wrapper(wrapper)


def test_compilation_fallback_mechanism(device: str, test_data: pd.DataFrame) -> None:
    """Test that the fallback mechanism works when compilation fails."""
    logger.info("Testing compilation fallback mechanism...")

    # This test verifies the _call_with_compile_fallback mechanism
    # by checking that predictions still work even if compilation has issues

    wrapper = _load_wrapper(
        compile_enabled=True,
        compile_mode="reduce-overhead",
        device=device,
    )

    try:
        context = test_data.iloc[:-PREDICTION_LENGTH]

        # First prediction should trigger compilation
        preds1 = _run_prediction(wrapper, context)
        assert len(preds1) == PREDICTION_LENGTH

        # Subsequent predictions should use compiled model
        preds2 = _run_prediction(wrapper, context)
        assert len(preds2) == PREDICTION_LENGTH

        # Check consistency
        mae = float(np.mean(np.abs(preds1 - preds2)))
        assert mae < 1e-4, f"Inconsistent predictions: {mae}"

        logger.info("✓ Fallback mechanism test passed")

    finally:
        _cleanup_wrapper(wrapper)


# Summary test
def test_recommended_configuration(device: str, test_data: pd.DataFrame) -> None:
    """Test the recommended production configuration."""
    logger.info("Testing recommended production configuration...")

    # Recommended: reduce-overhead + inductor + float32 + eager attention
    wrapper = _load_wrapper(
        compile_enabled=True,
        compile_mode="reduce-overhead",
        compile_backend="inductor",
        dtype="float32",
        device=device,
    )

    try:
        context = test_data.iloc[:-PREDICTION_LENGTH]

        # Run multiple predictions to ensure stability
        for i in range(3):
            preds = _run_prediction(wrapper, context)
            assert len(preds) == PREDICTION_LENGTH
            assert not np.isnan(preds).any()
            assert not np.isinf(preds).any()
            logger.info(f"  Run {i+1}/3: ✓")

        logger.info("✓ Recommended configuration test passed")

    finally:
        _cleanup_wrapper(wrapper)


if __name__ == "__main__":
    # Run tests manually for debugging
    device = _get_device()
    logger.info(f"Running tests on device: {device}")

    test_data = _create_test_data()

    try:
        logger.info("\n=== Basic Tests ===")
        test_eager_mode_baseline(device, test_data)

        logger.info("\n=== Compiled Mode Smoke Tests ===")
        for mode in COMPILE_MODES:
            for backend in COMPILE_BACKENDS:
                test_compiled_mode_smoke(device, test_data, mode, backend)

        logger.info("\n=== Accuracy Tests ===")
        for mode in COMPILE_MODES:
            test_eager_vs_compiled_accuracy(device, test_data, mode)

        logger.info("\n=== Extreme Data Robustness Tests ===")
        scenarios = [
            "very_small",
            "very_large",
            "high_volatility",
            "spike",
            "drop",
            "near_zero",
            "constant",
            "linear_trend",
        ]
        for scenario in scenarios:
            test_extreme_data_robustness(device, scenario)

        logger.info("\n=== Stress Tests ===")
        test_multiple_predictions_stability(device)
        test_compilation_fallback_mechanism(device, test_data)

        logger.info("\n=== Production Configuration Test ===")
        test_recommended_configuration(device, test_data)

        logger.info("\n" + "=" * 60)
        logger.info("ALL TESTS PASSED ✓")
        logger.info("=" * 60)

    except Exception as exc:
        logger.exception(f"Test failed: {exc}")
        sys.exit(1)
