"""
Toto warmup helper utilities.

Provides convenient functions to warm up Toto pipelines in production.
"""

import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def warmup_toto_pipeline(
    pipeline,
    num_warmup_runs: int = 2,
    context_length: int = 512,
    prediction_length: int = 8,
    num_samples: int = 256,
    samples_per_batch: int = 128,
    verbose: bool = True,
) -> float:
    """
    Warm up a Toto pipeline with dummy predictions.

    This triggers torch.compile graph compilation and ensures stable performance
    for subsequent predictions.

    Args:
        pipeline: TotoPipeline instance
        num_warmup_runs: Number of warmup predictions (default: 2)
        context_length: Length of dummy context (default: 512)
        prediction_length: Prediction horizon (default: 8)
        num_samples: Number of samples per prediction (default: 256)
        samples_per_batch: Batch size for sampling (default: 128)
        verbose: Print warmup progress (default: True)

    Returns:
        Total warmup time in seconds

    Example:
        >>> pipeline = TotoPipeline.from_pretrained(...)
        >>> warmup_time = warmup_toto_pipeline(pipeline)
        >>> print(f"Warmup completed in {warmup_time:.1f}s")
    """
    import time

    if verbose:
        logger.info(f"Warming up Toto pipeline ({num_warmup_runs} runs)...")

    # Generate dummy context
    dummy_context = torch.randn(context_length, dtype=torch.float32)

    start_time = time.time()

    for i in range(num_warmup_runs):
        if verbose:
            logger.info(f"  Warmup run {i+1}/{num_warmup_runs}...")

        try:
            _ = pipeline.predict(
                context=dummy_context,
                prediction_length=prediction_length,
                num_samples=num_samples,
                samples_per_batch=samples_per_batch,
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

        except Exception as e:
            logger.warning(f"Warmup run {i+1} failed: {e}")
            continue

    elapsed = time.time() - start_time

    if verbose:
        logger.info(f"✓ Warmup complete ({elapsed:.2f}s)")

    return elapsed


def verify_warmup_effectiveness(
    pipeline,
    real_context: torch.Tensor,
    prediction_length: int = 8,
    num_samples: int = 256,
    num_test_runs: int = 3,
) -> dict:
    """
    Verify that warmup has taken effect by checking prediction variance.

    Runs multiple predictions on the same input and measures variance.
    Low variance indicates warmup was effective.

    Args:
        pipeline: Warmed-up TotoPipeline instance
        real_context: Real input data to test with
        prediction_length: Prediction horizon
        num_samples: Number of samples per prediction
        num_test_runs: Number of test runs to measure variance

    Returns:
        Dictionary with variance statistics

    Example:
        >>> warmup_toto_pipeline(pipeline)
        >>> stats = verify_warmup_effectiveness(pipeline, my_context)
        >>> print(f"MAE variance: {stats['mae_std']:.2e}")
    """
    import time

    maes = []
    times = []

    logger.info(f"Verifying warmup effectiveness ({num_test_runs} test runs)...")

    for i in range(num_test_runs):
        start = time.time()

        forecast = pipeline.predict(
            context=real_context,
            prediction_length=prediction_length,
            num_samples=num_samples,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed_ms = (time.time() - start) * 1000

        samples = forecast[0].numpy()
        mae = np.mean(np.abs(samples))

        maes.append(mae)
        times.append(elapsed_ms)

        logger.info(f"  Run {i+1}: MAE={mae:.4f}, Time={elapsed_ms:.1f}ms")

    mae_array = np.array(maes)
    time_array = np.array(times)

    stats = {
        "mae_mean": np.mean(mae_array),
        "mae_std": np.std(mae_array),
        "mae_cv": np.std(mae_array) / np.mean(mae_array),  # Coefficient of variation
        "time_mean_ms": np.mean(time_array),
        "time_std_ms": np.std(time_array),
        "time_cv": np.std(time_array) / np.mean(time_array),
        "num_runs": num_test_runs,
    }

    logger.info(f"MAE variance: {stats['mae_std']:.2e} (CV: {stats['mae_cv']:.4f})")
    logger.info(f"Time variance: {stats['time_std_ms']:.2f}ms (CV: {stats['time_cv']:.4f})")

    # Check if warmup was effective
    time_cv_threshold = 0.5  # 50% time variance is acceptable
    if stats['time_cv'] > time_cv_threshold:
        logger.warning(
            f"High time variance detected (CV={stats['time_cv']:.2f}). "
            "May need more warmup runs or model is still recompiling."
        )
    else:
        logger.info("✓ Warmup appears effective (low time variance)")

    return stats


# Convenient presets
def quick_warmup(pipeline) -> float:
    """Quick warmup (1 run, minimal time)."""
    return warmup_toto_pipeline(pipeline, num_warmup_runs=1, verbose=False)


def standard_warmup(pipeline) -> float:
    """Standard warmup (2 runs, recommended for production)."""
    return warmup_toto_pipeline(pipeline, num_warmup_runs=2, verbose=True)


def thorough_warmup(pipeline) -> float:
    """Thorough warmup (3 runs, for maximum stability)."""
    return warmup_toto_pipeline(pipeline, num_warmup_runs=3, verbose=True)


if __name__ == "__main__":
    # Example usage
    print("Toto Warmup Helper")
    print("=" * 60)
    print()
    print("Usage:")
    print()
    print("from toto_warmup_helper import standard_warmup")
    print("import toto_compile_config")
    print()
    print("# Apply optimizations")
    print("toto_compile_config.apply()")
    print()
    print("# Load pipeline")
    print("pipeline = TotoPipeline.from_pretrained(...)")
    print()
    print("# Warmup (recommended)")
    print("warmup_time = standard_warmup(pipeline)")
    print()
    print("# Make predictions")
    print("forecast = pipeline.predict(...)")
