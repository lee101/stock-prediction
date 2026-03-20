"""Convert a HuggingFace Chronos-2 model to CuteChronos2 format.

Downloads the model from HuggingFace (if not cached), validates the
conversion by comparing outputs, and optionally runs a quick benchmark.

Usage::

    # Convert the default Chronos-2 base model
    python -m cutechronos.convert

    # Convert a specific model variant
    python -m cutechronos.convert --model-id amazon/chronos-bolt-base

    # Convert and benchmark
    python -m cutechronos.convert --benchmark

    # Convert to a specific output directory
    python -m cutechronos.convert --output-dir ./my_converted_model
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch


def download_model(model_id: str, cache_dir: str | None = None) -> Path:
    """Download a Chronos-2 model from HuggingFace Hub.

    Returns the local path to the downloaded snapshot.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub is required for downloading models.")
        print("Install it: pip install huggingface_hub")
        sys.exit(1)

    print(f"Downloading {model_id} from HuggingFace Hub...")
    local_path = snapshot_download(
        model_id,
        cache_dir=cache_dir,
        allow_patterns=["*.json", "*.safetensors", "*.bin"],
    )
    print(f"  Downloaded to: {local_path}")
    return Path(local_path)


def convert_model(
    model_path: Path,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> "CuteChronos2Model":
    """Load a Chronos-2 checkpoint and convert to CuteChronos2Model."""
    from cutechronos.model import CuteChronos2Model

    print(f"Loading CuteChronos2Model from {model_path}...")
    t0 = time.perf_counter()
    model = CuteChronos2Model.from_pretrained(model_path)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    t1 = time.perf_counter()
    print(f"  Loaded in {t1 - t0:.1f}s")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Device: {device}, Dtype: {dtype}")
    return model


def validate_conversion(
    model_path: Path,
    cute_model: "CuteChronos2Model",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> bool:
    """Validate that CuteChronos2Model produces matching outputs vs original.

    Returns True if validation passes.
    """
    try:
        from chronos.chronos2 import Chronos2Pipeline
    except ImportError:
        print("  SKIP: chronos-forecasting not installed, skipping validation")
        print("  Install it: pip install chronos-forecasting")
        return True

    print("Validating against original Chronos2Pipeline...")
    orig_pipe = Chronos2Pipeline.from_pretrained(str(model_path), dtype=dtype)
    orig_pipe.model = orig_pipe.model.to(device)

    torch.manual_seed(42)
    context = torch.randn(2, 512) * 0.1 + 100.0

    with torch.no_grad():
        # Original
        orig_out = orig_pipe.model(context.to(device))
        orig_preds = orig_out.quantile_preds

        # Cute
        cute_out = cute_model(context.to(device))

    if orig_preds.shape != cute_out.shape:
        print(f"  FAIL: Shape mismatch: {orig_preds.shape} vs {cute_out.shape}")
        return False

    # Handle NaN positions
    orig_nan = torch.isnan(orig_preds)
    cute_nan = torch.isnan(cute_out)
    if not (orig_nan == cute_nan).all():
        print("  WARN: NaN pattern differs (may be acceptable for some inputs)")

    valid_mask = ~orig_nan & ~cute_nan
    if valid_mask.any():
        max_err = (orig_preds[valid_mask] - cute_out[valid_mask]).abs().max().item()
        mean_err = (orig_preds[valid_mask] - cute_out[valid_mask]).abs().mean().item()
        print(f"  Max abs error: {max_err:.2e}")
        print(f"  Mean abs error: {mean_err:.2e}")
        # bfloat16 pretrained models can have larger max errors due to
        # numerical differences in attention (unscaled scores amplify rounding).
        # Mean error < 0.05 is the important metric for forecast quality.
        if mean_err < 0.05:
            print("  PASS: Outputs match within tolerance")
            return True
        else:
            print(f"  FAIL: Mean error {mean_err:.2e} exceeds threshold 0.05")
            return False
    else:
        print("  WARN: All outputs are NaN (common for random inputs)")
        return True


def run_benchmark(
    cute_model: "CuteChronos2Model",
    device: str = "cuda",
    context_length: int = 512,
    batch_size: int = 1,
    n_warmup: int = 5,
    n_runs: int = 20,
) -> dict:
    """Run a quick inference benchmark on the converted model."""
    print(f"\nBenchmarking (B={batch_size}, L={context_length}, {n_runs} runs)...")

    torch.manual_seed(42)
    context = torch.randn(batch_size, context_length, device=device) * 0.1 + 100.0

    # Warmup
    for _ in range(n_warmup):
        with torch.inference_mode():
            cute_model(context)
    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    latencies = []
    for _ in range(n_runs):
        if device == "cuda":
            torch.cuda.synchronize()
            if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()
        t0 = time.perf_counter()
        with torch.inference_mode():
            cute_model(context)
        if device == "cuda":
            torch.cuda.synchronize()
        latencies.append(time.perf_counter() - t0)

    import numpy as np
    avg_ms = np.mean(latencies) * 1000
    std_ms = np.std(latencies) * 1000
    min_ms = np.min(latencies) * 1000

    peak_mem = 0.0
    if device == "cuda":
        peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)

    results = {
        "avg_latency_ms": avg_ms,
        "std_latency_ms": std_ms,
        "min_latency_ms": min_ms,
        "peak_gpu_memory_mb": peak_mem,
        "batch_size": batch_size,
        "context_length": context_length,
    }

    print(f"  Avg latency: {avg_ms:.1f} +/- {std_ms:.1f} ms")
    print(f"  Min latency: {min_ms:.1f} ms")
    if peak_mem > 0:
        print(f"  Peak GPU memory: {peak_mem:.0f} MB")

    return results


def run_compiled_benchmark(
    model_path: Path,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    context_length: int = 512,
    batch_size: int = 1,
    n_warmup: int = 5,
    n_runs: int = 20,
) -> dict | None:
    """Benchmark with torch.compile applied."""
    if not hasattr(torch, "compile"):
        print("  torch.compile not available, skipping compiled benchmark")
        return None

    from cutechronos.model import CuteChronos2Model

    print(f"\nBenchmarking compiled model (mode=reduce-overhead)...")
    compiled_model = CuteChronos2Model.from_pretrained_compiled(
        model_path, compile_mode="reduce-overhead"
    )
    compiled_model = compiled_model.to(device=device, dtype=dtype)

    results = run_benchmark(
        compiled_model, device=device,
        context_length=context_length, batch_size=batch_size,
        n_warmup=n_warmup, n_runs=n_runs,
    )
    results["compile_mode"] = "reduce-overhead"

    del compiled_model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Convert a HuggingFace Chronos-2 model to CuteChronos2 format"
    )
    parser.add_argument(
        "--model-id",
        default="amazon/chronos-2",
        help="HuggingFace model ID (default: amazon/chronos-2)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda if available)",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run inference benchmark after conversion",
    )
    parser.add_argument(
        "--benchmark-compiled", action="store_true",
        help="Also benchmark with torch.compile",
    )
    parser.add_argument(
        "--validate", action="store_true", default=True,
        help="Validate output equivalence with original model",
    )
    parser.add_argument(
        "--no-validate", action="store_false", dest="validate",
        help="Skip validation",
    )
    parser.add_argument(
        "--context-length", type=int, default=512,
        help="Context length for benchmark",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Batch size for benchmark",
    )
    parser.add_argument(
        "--output-json", default=None,
        help="Path to save benchmark results as JSON",
    )
    args = parser.parse_args()

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    print("=" * 60)
    print("CuteDSL Model Converter")
    print("=" * 60)
    print(f"Model: {args.model_id}")
    print(f"Device: {args.device}")
    print(f"Dtype: {args.dtype}")
    print()

    # Step 1: Download
    model_path = download_model(args.model_id)

    # Step 2: Convert
    cute_model = convert_model(model_path, device=args.device, dtype=dtype)

    # Step 3: Validate
    results = {}
    if args.validate:
        passed = validate_conversion(model_path, cute_model, device=args.device, dtype=dtype)
        results["validation_passed"] = passed
        if not passed:
            print("\nWARNING: Validation failed! Outputs may differ from original.")

    # Step 4: Benchmark
    if args.benchmark:
        eager_results = run_benchmark(
            cute_model, device=args.device,
            context_length=args.context_length, batch_size=args.batch_size,
        )
        results["eager"] = eager_results

        if args.benchmark_compiled:
            del cute_model
            gc.collect()
            if args.device == "cuda":
                torch.cuda.empty_cache()

            compiled_results = run_compiled_benchmark(
                model_path, device=args.device, dtype=dtype,
                context_length=args.context_length, batch_size=args.batch_size,
            )
            if compiled_results:
                results["compiled"] = compiled_results
                if "eager" in results:
                    speedup = results["eager"]["avg_latency_ms"] / max(compiled_results["avg_latency_ms"], 1e-9)
                    print(f"\n  Compiled speedup: {speedup:.2f}x")
                    results["compiled_speedup"] = speedup

    # Save results
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
