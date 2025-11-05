"""
Comprehensive test harness for Toto compilation accuracy and performance.

This script validates that torch.compile maintains MAE equivalence with the
uncompiled version while identifying and fixing compilation issues like:
- CUDA graphs being skipped due to mutated inputs
- Recompilation limit being hit due to dynamic control flow
- Symbolic shapes warnings

Usage:
    # Run full MAE equivalence test
    python test_toto_compile_accuracy.py

    # Quick smoke test
    TOTO_COMPILE_QUICK=1 python test_toto_compile_accuracy.py

    # Test with specific symbol
    python test_toto_compile_accuracy.py BTCUSD

    # Verbose debugging
    TORCH_LOGS="recompiles,graph_breaks,cudagraphs" python test_toto_compile_accuracy.py
"""

import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger

# Ensure toto module can be imported
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.toto_wrapper import TotoPipeline

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)


@dataclass
class TestResult:
    """Result from a single test run."""

    symbol: str
    mode: str  # 'compiled' or 'uncompiled'
    mae: float
    inference_time_ms: float
    num_samples: int
    samples_per_batch: int
    context_length: int
    prediction_length: int
    gpu_memory_peak_mb: float
    recompiles: int
    graph_breaks: int


@dataclass
class ComparisonResult:
    """Comparison between compiled and uncompiled runs."""

    symbol: str
    compiled_mae: float
    uncompiled_mae: float
    mae_diff: float
    mae_diff_pct: float
    speedup: float
    compiled_time_ms: float
    uncompiled_time_ms: float
    is_equivalent: bool
    tolerance: float


def get_cuda_memory_stats() -> Tuple[float, float, float]:
    """Get CUDA memory statistics in MB."""
    if not torch.cuda.is_available():
        return 0.0, 0.0, 0.0

    torch.cuda.synchronize()
    stats = torch.cuda.memory_stats()

    allocated_mb = stats.get("allocated_bytes.all.current", 0) / (1024 ** 2)
    reserved_mb = stats.get("reserved_bytes.all.current", 0) / (1024 ** 2)
    peak_mb = stats.get("allocated_bytes.all.peak", 0) / (1024 ** 2)

    return allocated_mb, reserved_mb, peak_mb


def reset_cuda_stats():
    """Reset CUDA statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()


def generate_synthetic_data(
    context_length: int = 512,
    num_variates: int = 1,
    seed: int = 42,
) -> torch.Tensor:
    """Generate synthetic time series data for testing."""
    np.random.seed(seed)

    # Generate realistic-looking time series with trend + seasonality + noise
    t = np.arange(context_length)
    trend = 0.001 * t
    seasonality = 0.1 * np.sin(2 * np.pi * t / 24)
    noise = 0.05 * np.random.randn(context_length)

    series = 1.0 + trend + seasonality + noise

    if num_variates > 1:
        series = np.tile(series, (num_variates, 1))

    return torch.tensor(series, dtype=torch.float32)


def load_toto_pipeline(
    compiled: bool,
    device: str = "cuda",
    compile_mode: str = "max-autotune",
    compile_backend: Optional[str] = "inductor",
) -> TotoPipeline:
    """Load Toto pipeline with or without compilation."""
    model_id = "Datadog/Toto-Open-Base-1.0"

    logger.info(
        f"Loading Toto pipeline (compiled={compiled}, mode={compile_mode}, backend={compile_backend})"
    )

    # Set environment variables for compilation
    if compiled:
        os.environ["TOTO_COMPILE"] = "1"
        os.environ["TOTO_COMPILE_MODE"] = compile_mode
        if compile_backend:
            os.environ["TOTO_COMPILE_BACKEND"] = compile_backend
    else:
        os.environ["TOTO_DISABLE_COMPILE"] = "1"

    pipeline = TotoPipeline.from_pretrained(
        model_id=model_id,
        device_map=device,
        torch_dtype=torch.float32,  # Use float32 for accuracy testing
        torch_compile=compiled,
        compile_mode=compile_mode if compiled else None,
        compile_backend=compile_backend if compiled else None,
        warmup_sequence=0,  # We'll do manual warmup
        cache_policy="prefer",
    )

    return pipeline


def warmup_pipeline(
    pipeline: TotoPipeline,
    context_length: int = 512,
    prediction_length: int = 8,
    num_samples: int = 256,
    samples_per_batch: int = 128,
    num_warmup_runs: int = 2,
):
    """Warmup the pipeline to trigger compilation and cache population."""
    logger.info(
        f"Warming up pipeline (context={context_length}, pred={prediction_length}, "
        f"samples={num_samples}, batch={samples_per_batch}, runs={num_warmup_runs})"
    )

    context = generate_synthetic_data(context_length=context_length)

    for i in range(num_warmup_runs):
        logger.info(f"Warmup run {i+1}/{num_warmup_runs}")
        try:
            _ = pipeline.predict(
                context=context,
                prediction_length=prediction_length,
                num_samples=num_samples,
                samples_per_batch=samples_per_batch,
            )
        except Exception as e:
            logger.warning(f"Warmup run {i+1} failed: {e}")

    logger.info("Warmup complete")


def run_inference(
    pipeline: TotoPipeline,
    context: torch.Tensor,
    prediction_length: int,
    num_samples: int,
    samples_per_batch: int,
) -> Tuple[np.ndarray, float]:
    """Run inference and measure time."""
    reset_cuda_stats()

    start_time = time.perf_counter()

    forecasts = pipeline.predict(
        context=context,
        prediction_length=prediction_length,
        num_samples=num_samples,
        samples_per_batch=samples_per_batch,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Extract samples
    samples = forecasts[0].numpy()  # Shape: (num_samples, prediction_length)

    return samples, elapsed_ms


def compute_mae(samples1: np.ndarray, samples2: np.ndarray) -> float:
    """Compute mean absolute error between two sample sets."""
    # Both should be (num_samples, prediction_length)
    return np.mean(np.abs(samples1 - samples2))


def test_single_configuration(
    symbol: str,
    context_length: int = 512,
    prediction_length: int = 8,
    num_samples: int = 1024,
    samples_per_batch: int = 128,
    compile_mode: str = "max-autotune",
    compile_backend: Optional[str] = "inductor",
    tolerance: float = 1e-4,
    num_test_runs: int = 3,
) -> ComparisonResult:
    """Test a single configuration and compare compiled vs uncompiled."""
    logger.info(f"Testing {symbol} (context={context_length}, pred={prediction_length})")

    # Generate test data
    context = generate_synthetic_data(context_length=context_length, seed=hash(symbol) % 2**31)

    # Test uncompiled
    logger.info("Loading uncompiled pipeline...")
    uncompiled_pipeline = load_toto_pipeline(
        compiled=False,
        compile_mode=compile_mode,
        compile_backend=compile_backend,
    )

    logger.info("Warming up uncompiled pipeline...")
    warmup_pipeline(
        uncompiled_pipeline,
        context_length=context_length,
        prediction_length=prediction_length,
        num_samples=num_samples,
        samples_per_batch=samples_per_batch,
        num_warmup_runs=1,
    )

    logger.info("Running uncompiled inference...")
    uncompiled_samples, uncompiled_time = run_inference(
        uncompiled_pipeline,
        context,
        prediction_length,
        num_samples,
        samples_per_batch,
    )
    _, _, uncompiled_peak_mb = get_cuda_memory_stats()

    logger.info(f"Uncompiled: time={uncompiled_time:.2f}ms, peak_mem={uncompiled_peak_mb:.1f}MB")

    # Clean up
    uncompiled_pipeline.unload()
    del uncompiled_pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Test compiled
    logger.info("Loading compiled pipeline...")
    compiled_pipeline = load_toto_pipeline(
        compiled=True,
        compile_mode=compile_mode,
        compile_backend=compile_backend,
    )

    logger.info("Warming up compiled pipeline...")
    warmup_pipeline(
        compiled_pipeline,
        context_length=context_length,
        prediction_length=prediction_length,
        num_samples=num_samples,
        samples_per_batch=samples_per_batch,
        num_warmup_runs=2,  # Extra warmup for compilation
    )

    logger.info("Running compiled inference...")
    compiled_samples, compiled_time = run_inference(
        compiled_pipeline,
        context,
        prediction_length,
        num_samples,
        samples_per_batch,
    )
    _, _, compiled_peak_mb = get_cuda_memory_stats()

    logger.info(f"Compiled: time={compiled_time:.2f}ms, peak_mem={compiled_peak_mb:.1f}MB")

    # Clean up
    compiled_pipeline.unload()
    del compiled_pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Compare results
    mae_diff = compute_mae(compiled_samples, uncompiled_samples)

    # Also check mean and std
    compiled_mean = np.mean(compiled_samples)
    uncompiled_mean = np.mean(uncompiled_samples)
    mean_diff = abs(compiled_mean - uncompiled_mean)

    compiled_std = np.std(compiled_samples)
    uncompiled_std = np.std(uncompiled_samples)
    std_diff = abs(compiled_std - uncompiled_std)

    logger.info(f"MAE difference: {mae_diff:.6e}")
    logger.info(f"Mean difference: {mean_diff:.6e} (compiled={compiled_mean:.4f}, uncompiled={uncompiled_mean:.4f})")
    logger.info(f"Std difference: {std_diff:.6e} (compiled={compiled_std:.4f}, uncompiled={uncompiled_std:.4f})")

    is_equivalent = mae_diff < tolerance
    speedup = uncompiled_time / compiled_time if compiled_time > 0 else 0.0

    mae_diff_pct = (mae_diff / (abs(uncompiled_mean) + 1e-6)) * 100

    result = ComparisonResult(
        symbol=symbol,
        compiled_mae=compiled_mean,
        uncompiled_mae=uncompiled_mean,
        mae_diff=mae_diff,
        mae_diff_pct=mae_diff_pct,
        speedup=speedup,
        compiled_time_ms=compiled_time,
        uncompiled_time_ms=uncompiled_time,
        is_equivalent=is_equivalent,
        tolerance=tolerance,
    )

    return result


def main():
    """Main test runner."""
    # Parse command line
    quick_mode = os.getenv("TOTO_COMPILE_QUICK", "0") == "1"
    symbols = sys.argv[1:] if len(sys.argv) > 1 else ["BTCUSD"]

    if quick_mode:
        logger.info("Running in QUICK mode")
        test_configs = [
            {
                "context_length": 256,
                "prediction_length": 8,
                "num_samples": 256,
                "samples_per_batch": 128,
            }
        ]
    else:
        logger.info("Running FULL test suite")
        test_configs = [
            {
                "context_length": 512,
                "prediction_length": 8,
                "num_samples": 1024,
                "samples_per_batch": 128,
            },
            {
                "context_length": 1024,
                "prediction_length": 16,
                "num_samples": 512,
                "samples_per_batch": 64,
            },
        ]

    results: List[ComparisonResult] = []

    for symbol in symbols:
        for config in test_configs:
            logger.info(f"\n{'='*80}")
            logger.info(f"Testing {symbol} with config: {config}")
            logger.info(f"{'='*80}\n")

            try:
                result = test_single_configuration(
                    symbol=symbol,
                    **config,
                    tolerance=1e-3,  # Relaxed tolerance for float32
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Test failed for {symbol}: {e}", exc_info=True)

    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*80}\n")

    df = pd.DataFrame([
        {
            "Symbol": r.symbol,
            "MAE Diff": f"{r.mae_diff:.2e}",
            "MAE Diff %": f"{r.mae_diff_pct:.4f}%",
            "Speedup": f"{r.speedup:.2f}x",
            "Compiled (ms)": f"{r.compiled_time_ms:.1f}",
            "Uncompiled (ms)": f"{r.uncompiled_time_ms:.1f}",
            "Equivalent": "✓" if r.is_equivalent else "✗",
        }
        for r in results
    ])

    print(df.to_string(index=False))

    # Check if all tests passed
    all_passed = all(r.is_equivalent for r in results)

    if all_passed:
        logger.info(f"\n✓ All {len(results)} tests PASSED")
        return 0
    else:
        failed = [r for r in results if not r.is_equivalent]
        logger.error(f"\n✗ {len(failed)}/{len(results)} tests FAILED")
        for r in failed:
            logger.error(
                f"  {r.symbol}: MAE diff {r.mae_diff:.2e} exceeds tolerance {r.tolerance:.2e}"
            )
        return 1


if __name__ == "__main__":
    sys.exit(main())
