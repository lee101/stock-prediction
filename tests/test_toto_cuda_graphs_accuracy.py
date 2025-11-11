"""
Test that Toto CUDA graph optimizations maintain prediction accuracy.

This test ensures that the fix for CUDA graphs (replacing .item() with int())
produces identical predictions to the original implementation.
"""

import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

# Ensure TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS is set before any torch imports
os.environ.setdefault("TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS", "1")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(Path(__file__).parent.parent / "compiled_models" / "torch_inductor"))

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from toto.toto.pipelines.time_series_forecasting import TotoPipeline

# Test configuration
TOTO_MODEL_ID = "Datadog/Toto-Open-Base-1.0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PREDICTION_LENGTH = 30
NUM_TRIALS = 3  # Run multiple trials to ensure consistency


def capture_cuda_graph_warnings():
    """Context manager to capture stderr and check for CUDA graph warnings."""
    import io
    import contextlib

    stderr_capture = io.StringIO()

    @contextlib.contextmanager
    def capturing():
        old_stderr = sys.stderr
        try:
            sys.stderr = stderr_capture
            yield stderr_capture
        finally:
            sys.stderr = old_stderr

    return capturing()


def check_cuda_graphs_enabled(compile_logs: str) -> dict:
    """
    Parse compilation logs to determine if CUDA graphs are enabled.

    Returns dict with:
        - cudagraphs_skipped: bool
        - skip_reasons: list of reasons
        - mutated_inputs_count: int
    """
    results = {
        "cudagraphs_skipped": False,
        "skip_reasons": [],
        "mutated_inputs_count": 0,
    }

    if "skipping cudagraphs" in compile_logs:
        results["cudagraphs_skipped"] = True

        # Count different skip reasons
        if "disabling cudagraphs due to incompatible op aten._local_scalar_dense" in compile_logs:
            results["skip_reasons"].append("incompatible_op_local_scalar_dense")

        if "mutated inputs" in compile_logs:
            results["skip_reasons"].append("mutated_inputs")
            # Count occurrences
            results["mutated_inputs_count"] = compile_logs.count("skipping cudagraphs due to mutated inputs")

    return results


def generate_sample_data(num_variates: int = 10, context_length: int = 512) -> dict:
    """Generate sample time series data for testing."""
    np.random.seed(42)
    torch.manual_seed(42)

    # Generate synthetic time series with trend and seasonality
    t = np.arange(context_length)
    data = []
    for i in range(num_variates):
        trend = 0.1 * t
        seasonal = 10 * np.sin(2 * np.pi * t / 50)
        noise = np.random.randn(context_length) * 2
        series = 100 + trend + seasonal + noise
        data.append(series)

    return {
        "target": np.array(data),
        "freq": "1min",
    }


def run_prediction_test(
    use_compile: bool = True,
    verbose: bool = False
) -> tuple[np.ndarray, float, dict]:
    """
    Run a single prediction test.

    Returns:
        - predictions: numpy array of predictions
        - inference_time: time taken for inference
        - cuda_graph_info: dict with CUDA graph diagnostics
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Running test: use_compile={use_compile}")
        print(f"{'='*60}")

    # Load pipeline
    print(f"Loading Toto pipeline on {DEVICE}...")
    start_load = time.time()

    pipeline = TotoPipeline.from_pretrained(
        model_id=TOTO_MODEL_ID,
        device=DEVICE,
        torch_dtype=torch.float32,  # Use float32 for accuracy testing
    )

    load_time = time.time() - start_load
    if verbose:
        print(f"Pipeline loaded in {load_time:.2f}s")

    # Apply torch.compile if requested
    if use_compile and DEVICE == "cuda":
        print("Applying torch.compile with mode='reduce-overhead'...")
        compile_start = time.time()

        # Capture compilation warnings
        import io
        import contextlib

        stderr_capture = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = stderr_capture

        try:
            pipeline.model = torch.compile(
                pipeline.model,
                mode="reduce-overhead",
                backend="inductor",
            )
            compile_time = time.time() - compile_start
            if verbose:
                print(f"Compilation setup done in {compile_time:.2f}s")
        finally:
            sys.stderr = old_stderr
            compile_logs = stderr_capture.getvalue()
    else:
        compile_logs = ""

    # Generate test data
    print("Generating test data...")
    data = generate_sample_data(num_variates=10, context_length=512)

    # Run prediction with timing
    print("Running prediction...")
    start_pred = time.time()

    # Capture stderr during prediction to catch CUDA graph warnings
    stderr_capture = io.StringIO()
    old_stderr = sys.stderr
    sys.stderr = stderr_capture

    try:
        predictions = pipeline(
            target=data["target"],
            freq=data["freq"],
            prediction_length=PREDICTION_LENGTH,
        )
    finally:
        sys.stderr = old_stderr
        pred_logs = stderr_capture.getvalue()

    inference_time = time.time() - start_pred

    # Combine all logs
    all_logs = compile_logs + pred_logs

    # Check CUDA graph status
    cuda_graph_info = check_cuda_graphs_enabled(all_logs)

    if verbose:
        print(f"Prediction completed in {inference_time:.2f}s")
        print(f"Predictions shape: {predictions.shape}")
        print(f"CUDA graphs skipped: {cuda_graph_info['cudagraphs_skipped']}")
        if cuda_graph_info['skip_reasons']:
            print(f"Skip reasons: {cuda_graph_info['skip_reasons']}")
            print(f"Mutated inputs count: {cuda_graph_info['mutated_inputs_count']}")

    # Clean up
    del pipeline
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return predictions, inference_time, cuda_graph_info


def test_accuracy_maintained():
    """
    Test that predictions are identical with and without torch.compile.
    This is the critical accuracy test.
    """
    print("\n" + "="*80)
    print("ACCURACY TEST: Comparing predictions with torch.compile optimization")
    print("="*80)

    if not torch.cuda.is_available():
        print("⚠️  CUDA not available - skipping test")
        return True

    # Run without compile (baseline)
    print("\n1. Running WITHOUT torch.compile (baseline)...")
    pred_no_compile, time_no_compile, _ = run_prediction_test(use_compile=False, verbose=True)

    # Run with compile (optimized)
    print("\n2. Running WITH torch.compile (optimized)...")
    pred_with_compile, time_with_compile, cuda_info = run_prediction_test(use_compile=True, verbose=True)

    # Compare predictions
    print("\n" + "="*80)
    print("RESULTS:")
    print("="*80)

    # Check if predictions are identical
    max_diff = np.max(np.abs(pred_no_compile - pred_with_compile))
    mean_diff = np.mean(np.abs(pred_no_compile - pred_with_compile))
    rel_diff = max_diff / (np.abs(pred_no_compile).mean() + 1e-8)

    print(f"\nAccuracy comparison:")
    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  Mean absolute difference: {mean_diff:.2e}")
    print(f"  Relative difference: {rel_diff:.2e}")

    # Check CUDA graph status
    print(f"\nCUDA graph status:")
    print(f"  CUDA graphs skipped: {cuda_info['cudagraphs_skipped']}")
    if cuda_info['skip_reasons']:
        print(f"  Skip reasons: {', '.join(cuda_info['skip_reasons'])}")
        print(f"  Mutated inputs warnings: {cuda_info['mutated_inputs_count']}")

    # Performance comparison
    speedup = time_no_compile / time_with_compile
    print(f"\nPerformance comparison:")
    print(f"  Time without compile: {time_no_compile:.2f}s")
    print(f"  Time with compile: {time_with_compile:.2f}s")
    print(f"  Speedup: {speedup:.2f}x")

    # Determine success
    accuracy_ok = max_diff < 1e-5  # Very tight tolerance for identical predictions

    if accuracy_ok:
        print("\n✅ ACCURACY TEST PASSED: Predictions are identical")
    else:
        print(f"\n❌ ACCURACY TEST FAILED: Predictions differ by {max_diff:.2e}")

    # Check if CUDA graphs are enabled (should NOT be skipped after fix)
    if cuda_info['cudagraphs_skipped']:
        print("\n⚠️  WARNING: CUDA graphs are still being skipped!")
        print("   This means the optimization is not fully effective.")
        if "incompatible_op_local_scalar_dense" in cuda_info['skip_reasons']:
            print("   ❌ CRITICAL: Still seeing .item() incompatible ops!")
            return False
    else:
        print("\n✅ CUDA graphs are ENABLED - optimization is working!")

    return accuracy_ok


def test_consistency():
    """
    Test that predictions are consistent across multiple runs.
    """
    print("\n" + "="*80)
    print("CONSISTENCY TEST: Running multiple trials")
    print("="*80)

    if not torch.cuda.is_available():
        print("⚠️  CUDA not available - skipping test")
        return True

    predictions_list = []
    times_list = []

    for trial in range(NUM_TRIALS):
        print(f"\nTrial {trial + 1}/{NUM_TRIALS}...")
        pred, inf_time, _ = run_prediction_test(use_compile=True, verbose=False)
        predictions_list.append(pred)
        times_list.append(inf_time)

    # Check consistency
    print("\n" + "="*80)
    print("CONSISTENCY RESULTS:")
    print("="*80)

    # Compare all predictions
    max_diff_across_trials = 0
    for i in range(1, NUM_TRIALS):
        diff = np.max(np.abs(predictions_list[0] - predictions_list[i]))
        max_diff_across_trials = max(max_diff_across_trials, diff)

    print(f"\nMax difference across {NUM_TRIALS} trials: {max_diff_across_trials:.2e}")

    # Timing stats
    mean_time = np.mean(times_list)
    std_time = np.std(times_list)
    print(f"\nInference time statistics:")
    print(f"  Mean: {mean_time:.3f}s")
    print(f"  Std: {std_time:.3f}s")
    print(f"  Min: {min(times_list):.3f}s")
    print(f"  Max: {max(times_list):.3f}s")

    consistent = max_diff_across_trials < 1e-5

    if consistent:
        print("\n✅ CONSISTENCY TEST PASSED: Predictions are consistent")
    else:
        print(f"\n❌ CONSISTENCY TEST FAILED: Predictions vary by {max_diff_across_trials:.2e}")

    return consistent


if __name__ == "__main__":
    print("="*80)
    print("TOTO CUDA GRAPH OPTIMIZATION - ACCURACY & PERFORMANCE TEST")
    print("="*80)
    print(f"\nEnvironment:")
    print(f"  Device: {DEVICE}")
    print(f"  TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS: {os.environ.get('TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS')}")
    print(f"  PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Run tests
    try:
        accuracy_passed = test_accuracy_maintained()
        consistency_passed = test_consistency()

        print("\n" + "="*80)
        print("FINAL RESULTS:")
        print("="*80)
        print(f"  Accuracy test: {'✅ PASSED' if accuracy_passed else '❌ FAILED'}")
        print(f"  Consistency test: {'✅ PASSED' if consistency_passed else '❌ FAILED'}")

        if accuracy_passed and consistency_passed:
            print("\n🎉 ALL TESTS PASSED! CUDA graph optimization is working correctly.")
            sys.exit(0)
        else:
            print("\n❌ SOME TESTS FAILED! Review the output above.")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ TEST CRASHED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
