"""
Quick test to verify CUDA graphs are no longer being skipped.

This is a faster smoke test that just checks for the absence of
CUDA graph warning messages during compilation and inference.
"""

import os
import sys
import io
from pathlib import Path

import torch

# Ensure TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS is set before any torch imports
os.environ.setdefault("TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS", "1")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(Path(__file__).parent.parent / "compiled_models" / "torch_inductor"))

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_cuda_graph_warnings():
    """
    Quick test that checks for CUDA graph skip warnings.
    """
    print("="*80)
    print("QUICK CUDA GRAPH CHECK")
    print("="*80)

    if not torch.cuda.is_available():
        print("⚠️  CUDA not available - test not applicable")
        return True

    print(f"\nDevice: {torch.cuda.get_device_name(0)}")
    print(f"TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS: {os.environ.get('TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS')}")

    # Import after env vars are set
    from toto.toto.pipelines.time_series_forecasting import TotoPipeline
    import numpy as np

    # Load pipeline
    print("\nLoading Toto pipeline...")
    pipeline = TotoPipeline.from_pretrained(
        model_id="Datadog/Toto-Open-Base-1.0",
        device="cuda",
        torch_dtype=torch.float32,
    )

    # Capture stderr during compilation
    print("Applying torch.compile...")
    stderr_capture = io.StringIO()
    old_stderr = sys.stderr
    sys.stderr = stderr_capture

    try:
        pipeline.model = torch.compile(
            pipeline.model,
            mode="reduce-overhead",
            backend="inductor",
        )

        # Generate minimal test data
        np.random.seed(42)
        data = {
            "target": np.random.randn(5, 64),  # Small data for quick test
            "freq": "1min",
        }

        # Run one prediction to trigger compilation
        print("Running test prediction to trigger compilation...")
        _ = pipeline(
            target=data["target"],
            freq=data["freq"],
            prediction_length=10,
        )

    finally:
        sys.stderr = old_stderr
        compile_logs = stderr_capture.getvalue()

    # Analyze logs
    print("\n" + "="*80)
    print("RESULTS:")
    print("="*80)

    issues = []

    # Check for .item() incompatibility
    if "aten._local_scalar_dense.default" in compile_logs:
        issues.append("❌ CRITICAL: Found incompatible .item() operation (aten._local_scalar_dense.default)")
        # Show the line
        for line in compile_logs.split('\n'):
            if "aten._local_scalar_dense" in line or "util_compile_friendly.py" in line:
                print(f"  {line}")

    # Check for general CUDA graph skipping
    if "skipping cudagraphs" in compile_logs:
        skip_count = compile_logs.count("skipping cudagraphs")
        issues.append(f"⚠️  CUDA graphs skipped {skip_count} times")

        # Count specific reasons
        if "mutated inputs" in compile_logs:
            mutated_count = compile_logs.count("mutated inputs")
            issues.append(f"  - Mutated inputs: {mutated_count} instances")

        if "non gpu ops" in compile_logs:
            non_gpu_count = compile_logs.count("non gpu ops")
            issues.append(f"  - Non-GPU ops: {non_gpu_count} instances")

    # Report
    if not issues:
        print("✅ SUCCESS: No CUDA graph issues detected!")
        print("\nCUDA graphs should be fully enabled for Toto inference.")
        return True
    else:
        print("Issues detected:\n")
        for issue in issues:
            print(issue)

        print("\n" + "="*80)
        print("FULL COMPILATION LOG:")
        print("="*80)
        print(compile_logs)

        return False


if __name__ == "__main__":
    try:
        success = test_cuda_graph_warnings()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ TEST CRASHED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
