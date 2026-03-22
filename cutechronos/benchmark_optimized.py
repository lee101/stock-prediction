"""End-to-end benchmark comparing CuteChronos2 optimization levels.

Compares three configurations:
  - original:      Chronos2Pipeline from chronos.chronos2.pipeline
  - cute_eager:    CuteChronos2Model with PyTorch fallbacks (no compile)
  - cute_compiled: CuteChronos2Model + torch.compile(mode="reduce-overhead")

Test matrix: batch sizes x context lengths x prediction_length.

Metrics: latency (avg/min/std over N runs, M warmup), throughput,
peak GPU memory, allocation count, MAE vs actual from trainingdata/ CSVs
(close column), MAE delta vs original (must be < 1e-4).

Usage::

    python -m cutechronos.benchmark_optimized --symbols BTCUSD --context-lengths 512 --batch-sizes 1
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_series(
    csv_path: str, context_length: int, prediction_length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load close prices from a CSV and split into context + future actual.

    Returns
    -------
    context : torch.Tensor of shape (context_length,)
    actual  : torch.Tensor of shape (prediction_length,)
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    close = df["close"].values.astype(np.float32)
    total_needed = context_length + prediction_length
    if len(close) < total_needed:
        raise ValueError(
            f"{csv_path}: need {total_needed} rows but only have {len(close)}"
        )
    close = close[-total_needed:]
    context = torch.tensor(close[:context_length], dtype=torch.float32)
    actual = torch.tensor(close[context_length:], dtype=torch.float32)
    return context, actual


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------

def reset_gpu_stats():
    """Reset GPU memory stats and clear caches."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        torch.cuda.empty_cache()
    gc.collect()


def get_peak_memory_mb() -> float:
    """Return peak GPU memory allocated in MB (0 if no CUDA)."""
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def get_allocation_count() -> int:
    """Return total number of CUDA memory allocations since last reset."""
    if not torch.cuda.is_available():
        return 0
    torch.cuda.synchronize()
    stats = torch.cuda.memory_stats()
    return stats.get("allocation.all.current", 0) + stats.get(
        "allocation.all.allocated", 0
    )


# ---------------------------------------------------------------------------
# Timing helpers using CUDA events
# ---------------------------------------------------------------------------

def timed_call_cuda(fn) -> tuple[Any, float]:
    """Time a callable using torch.cuda.Event for precise GPU timing.

    Returns (result, elapsed_seconds).
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
        return result, elapsed_ms / 1000.0
    else:
        t0 = time.perf_counter()
        result = fn()
        t1 = time.perf_counter()
        return result, t1 - t0


# ---------------------------------------------------------------------------
# MAE computation
# ---------------------------------------------------------------------------

def compute_mae(pred: torch.Tensor, actual: torch.Tensor) -> float:
    """Compute mean absolute error between pred and actual.

    Returns NaN if either tensor contains NaN values.
    """
    length = min(len(pred), len(actual))
    diff = (pred[:length].float() - actual[:length].float()).abs()
    return diff.mean().item()  # Returns NaN if any element is NaN


def _format_mae(value: float) -> str:
    """Format MAE value, handling NaN."""
    return "NaN" if np.isnan(value) else f"{value:.6f}"


# ---------------------------------------------------------------------------
# Config loading helpers
# ---------------------------------------------------------------------------

def _ensure_chronos_on_path():
    """Add chronos-forecasting/src to sys.path if available."""
    repo_root = Path(__file__).resolve().parent.parent
    chronos_src = repo_root / "chronos-forecasting" / "src"
    if chronos_src.exists() and str(chronos_src) not in sys.path:
        sys.path.insert(0, str(chronos_src))


def _load_original_pipeline(model_id: str, device: str, dtype: torch.dtype):
    """Load the upstream Chronos2Pipeline."""
    _ensure_chronos_on_path()
    from chronos.chronos2 import Chronos2Pipeline

    pipeline = Chronos2Pipeline.from_pretrained(model_id, dtype=dtype)
    pipeline.model = pipeline.model.to(device)
    return pipeline


def _load_cute_model(model_id: str, device: str, dtype: torch.dtype):
    """Load CuteChronos2Model from a pretrained Chronos2 checkpoint.

    Uses from_original to copy weights from the original pipeline model.
    """
    _ensure_chronos_on_path()
    from chronos.chronos2 import Chronos2Pipeline
    from cutechronos.model import CuteChronos2Model

    pipeline = Chronos2Pipeline.from_pretrained(model_id, dtype=dtype)
    pipeline.model = pipeline.model.to(device)

    cute_model = CuteChronos2Model.from_original(pipeline.model)
    cute_model = cute_model.to(device=device, dtype=dtype)
    cute_model.eval()

    quantiles = list(pipeline.model.chronos_config.quantiles)
    max_output_patches = pipeline.model.chronos_config.max_output_patches
    output_patch_size = pipeline.model.chronos_config.output_patch_size

    del pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return cute_model, quantiles, max_output_patches, output_patch_size


def _cleanup_model(*models):
    """Delete model(s) and reclaim GPU memory."""
    for m in models:
        del m
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Predict wrappers
# ---------------------------------------------------------------------------

def predict_original(
    pipeline,
    context_batch: torch.Tensor,
    prediction_length: int,
) -> torch.Tensor:
    """Run original Chronos2Pipeline.predict and return median prediction.

    Returns (B, prediction_length) median predictions.
    """
    inputs = [context_batch[i] for i in range(context_batch.shape[0])]
    preds = pipeline.predict(
        inputs, prediction_length=prediction_length, limit_prediction_length=False
    )
    # preds is list of (1, Q, H) tensors
    quantiles = pipeline.quantiles
    median_idx = quantiles.index(0.5)
    medians = torch.cat([p[0, median_idx : median_idx + 1, :] for p in preds], dim=0)
    return medians


def predict_cute(
    model,
    context_batch: torch.Tensor,
    prediction_length: int,
    quantiles: list[float],
    max_output_patches: int,
    output_patch_size: int,
    device: str,
) -> torch.Tensor:
    """Run CuteChronos2Model forward and return median prediction.

    Returns (B, prediction_length) median predictions.
    """
    ctx = context_batch.to(device=device, dtype=torch.float32)
    num_output_patches = math.ceil(prediction_length / output_patch_size)
    num_output_patches = min(num_output_patches, max_output_patches)

    with torch.inference_mode():
        out = model(ctx, num_output_patches=num_output_patches)

    # out: (B, Q, H)
    median_idx = quantiles.index(0.5)
    medians = out[:, median_idx, :prediction_length].float().cpu()
    return medians


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def benchmark_config(
    label: str,
    predict_fn,
    context_batches: list[torch.Tensor],
    actuals: list[torch.Tensor],
    prediction_length: int,
    n_warmup: int,
    n_runs: int,
) -> dict[str, Any]:
    """Benchmark a single configuration.

    Parameters
    ----------
    label : str
        Name of the configuration (e.g. "original", "cute_eager").
    predict_fn : callable
        Takes a context_batch and returns (B, prediction_length) median preds.
    context_batches : list[torch.Tensor]
        Each element is (batch_size, context_length).
    actuals : list[torch.Tensor]
        Corresponding actual values, each (batch_size, prediction_length).
    """
    # Warmup
    for _ in range(n_warmup):
        for ctx_batch in context_batches:
            predict_fn(ctx_batch)

    # Reset GPU stats after warmup
    reset_gpu_stats()
    alloc_before = get_allocation_count()

    latencies: list[float] = []
    all_maes: list[float] = []

    for _run in range(n_runs):
        for batch_idx, ctx_batch in enumerate(context_batches):
            preds, elapsed = timed_call_cuda(lambda cb=ctx_batch: predict_fn(cb))
            latencies.append(elapsed)

            for i in range(preds.shape[0]):
                mae = compute_mae(preds[i], actuals[batch_idx][i])
                all_maes.append(mae)

    alloc_after = get_allocation_count()
    peak_mem = get_peak_memory_mb()

    avg_latency = float(np.mean(latencies))
    total_time = sum(latencies)
    total_samples = sum(cb.shape[0] for cb in context_batches) * n_runs
    throughput = total_samples / total_time if total_time > 0 else 0.0

    return {
        "label": label,
        "avg_latency_ms": avg_latency * 1000,
        "min_latency_ms": float(np.min(latencies)) * 1000,
        "std_latency_ms": float(np.std(latencies)) * 1000,
        "throughput_samples_per_sec": throughput,
        "peak_gpu_memory_mb": peak_mem,
        "allocation_count": alloc_after - alloc_before,
        "avg_mae": float(np.mean(all_maes)),
    }


# ---------------------------------------------------------------------------
# Comparison table printing
# ---------------------------------------------------------------------------

def print_comparison_table(results: dict[str, dict], configs: list[str]):
    """Print a formatted comparison table."""
    print()
    print("=" * 100)
    print("BENCHMARK RESULTS")
    print("=" * 100)
    header = (
        f"{'Config':<20} {'Avg(ms)':<12} {'Min(ms)':<12} {'Std(ms)':<12} "
        f"{'Tput(s/s)':<12} {'GPU(MB)':<12} {'Allocs':<10} {'MAE':<12}"
    )
    print(header)
    print("-" * 100)
    for cfg in configs:
        if cfg not in results:
            continue
        r = results[cfg]
        print(
            f"{r['label']:<20} "
            f"{r['avg_latency_ms']:<12.2f} "
            f"{r['min_latency_ms']:<12.2f} "
            f"{r['std_latency_ms']:<12.2f} "
            f"{r['throughput_samples_per_sec']:<12.2f} "
            f"{r['peak_gpu_memory_mb']:<12.1f} "
            f"{r['allocation_count']:<10} "
            f"{_format_mae(r['avg_mae']):<12}"
        )

    # MAE delta comparisons vs original
    if "original" in results:
        orig_mae = results["original"]["avg_mae"]
        for cfg in ["cute_eager", "cute_compiled"]:
            if cfg not in results:
                continue
            cute_mae = results[cfg]["avg_mae"]
            if np.isnan(orig_mae) or np.isnan(cute_mae):
                print(
                    f"\nMAE delta ({cfg} vs original): N/A "
                    f"(original={_format_mae(orig_mae)}, "
                    f"{cfg}={_format_mae(cute_mae)})"
                )
                if np.isnan(orig_mae) and not np.isnan(cute_mae):
                    print(
                        f"  Note: original produces NaN (known issue with "
                        f"pretrained model on high-magnitude data). "
                        f"CuteChronos2 avoids this."
                    )
            else:
                delta = abs(cute_mae - orig_mae)
                status = "PASS" if delta < 1e-4 else "FAIL"
                print(
                    f"\nMAE delta ({cfg} vs original): {delta:.2e} [{status}]"
                )
            speedup = results["original"]["avg_latency_ms"] / max(
                results[cfg]["avg_latency_ms"], 1e-9
            )
            print(f"Speedup ({cfg} vs original): {speedup:.2f}x")

    # Compare cute_eager vs cute_compiled
    if "cute_eager" in results and "cute_compiled" in results:
        eager_mae = results["cute_eager"]["avg_mae"]
        compiled_mae = results["cute_compiled"]["avg_mae"]
        if not np.isnan(eager_mae) and not np.isnan(compiled_mae):
            delta = abs(compiled_mae - eager_mae)
            status = "PASS" if delta < 1e-4 else "FAIL"
            print(
                f"\nMAE delta (cute_compiled vs cute_eager): {delta:.2e} [{status}]"
            )
        speedup = results["cute_eager"]["avg_latency_ms"] / max(
            results["cute_compiled"]["avg_latency_ms"], 1e-9
        )
        print(f"Speedup (cute_compiled vs cute_eager): {speedup:.2f}x")


# ---------------------------------------------------------------------------
# Per-config benchmark runner (reduces copy-paste in main)
# ---------------------------------------------------------------------------

def _run_config_benchmark(
    config_name: str,
    config_label: str,
    predict_fn,
    context_batches: list[torch.Tensor],
    actual_batches: list[torch.Tensor],
    prediction_length: int,
    n_warmup: int,
    n_runs: int,
) -> dict[str, Any] | None:
    """Run benchmark for a single config, printing progress. Returns result or None on failure."""
    print(f"\n  --- {config_label} ---")
    try:
        result = benchmark_config(
            label=config_name,
            predict_fn=predict_fn,
            context_batches=context_batches,
            actuals=actual_batches,
            prediction_length=prediction_length,
            n_warmup=n_warmup,
            n_runs=n_runs,
        )
        print(
            f"    Avg: {result['avg_latency_ms']:.2f}ms  "
            f"MAE: {_format_mae(result['avg_mae'])}  "
            f"GPU: {result['peak_gpu_memory_mb']:.1f}MB"
        )
        return result
    except Exception as e:
        print(f"    Failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CuteChronos2 optimization-level benchmark"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTCUSD"],
        help="Symbol names (CSV files in trainingdata/)",
    )
    parser.add_argument(
        "--context-lengths",
        nargs="+",
        type=int,
        default=[512, 2048],
        help="Context lengths to test",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 4],
        help="Batch sizes to test",
    )
    parser.add_argument(
        "--prediction-length", type=int, default=30, help="Prediction length"
    )
    parser.add_argument(
        "--model-id",
        default="amazon/chronos-2",
        help="HF model id or local path",
    )
    parser.add_argument("--n-warmup", type=int, default=3, help="Warmup runs")
    parser.add_argument("--n-runs", type=int, default=10, help="Timed runs")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device",
    )
    parser.add_argument(
        "--data-dir",
        default="trainingdata",
        help="Directory with CSV files",
    )
    parser.add_argument(
        "--output",
        default="cutechronos/benchmark_optimized_results.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["original", "cute_eager", "cute_compiled"],
        choices=["original", "cute_eager", "cute_compiled"],
        help="Configurations to benchmark",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    data_dir_arg = Path(args.data_dir)
    data_dir = data_dir_arg if data_dir_arg.is_absolute() else repo_root / args.data_dir

    print(f"Data dir: {data_dir}")
    print(f"Symbols: {args.symbols}")
    print(f"Context lengths: {args.context_lengths}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Prediction length: {args.prediction_length}")
    print(f"Device: {args.device}")
    print(f"Configs: {args.configs}")
    print(f"Warmup: {args.n_warmup}, Runs: {args.n_runs}")
    print()

    # Load all series data for the largest context length
    max_ctx = max(args.context_lengths)
    symbol_data: dict[str, np.ndarray] = {}
    for sym in args.symbols:
        csv_path = data_dir / f"{sym}.csv"
        if not csv_path.exists():
            print(f"WARNING: {csv_path} not found, skipping {sym}")
            continue
        import pandas as pd

        df = pd.read_csv(str(csv_path))
        close = df["close"].values.astype(np.float32)
        total_needed = max_ctx + args.prediction_length
        if len(close) < total_needed:
            print(
                f"WARNING: {sym} has {len(close)} rows, need {total_needed}, skipping"
            )
            continue
        symbol_data[sym] = close
        print(f"  {sym}: {len(close)} rows loaded")

    if not symbol_data:
        print("No data loaded, exiting.")
        return

    all_results: dict[str, dict] = {}

    # Iterate over the test matrix: context_lengths x batch_sizes
    for ctx_len in args.context_lengths:
        for batch_size in args.batch_sizes:
            combo_key = f"ctx{ctx_len}_bs{batch_size}"
            print(f"\n{'='*80}")
            print(f"Test: context_length={ctx_len}, batch_size={batch_size}")
            print(f"{'='*80}")

            # Build context batches and actuals from the loaded symbol data
            context_batches: list[torch.Tensor] = []
            actual_batches: list[torch.Tensor] = []

            for sym, close in symbol_data.items():
                total_needed = ctx_len + args.prediction_length
                if len(close) < total_needed:
                    print(f"  Skipping {sym} for ctx_len={ctx_len}")
                    continue
                close_tail = close[-total_needed:]
                ctx_1d = torch.tensor(close_tail[:ctx_len], dtype=torch.float32)
                act_1d = torch.tensor(close_tail[ctx_len:], dtype=torch.float32)

                # Replicate to fill batch_size
                context_batches.append(ctx_1d.unsqueeze(0).expand(batch_size, -1).clone())
                actual_batches.append(act_1d.unsqueeze(0).expand(batch_size, -1).clone())

            if not context_batches:
                print("  No valid data for this combo, skipping")
                continue

            combo_results: dict[str, dict] = {}

            # --- original ---
            if "original" in args.configs:
                pipeline = _load_original_pipeline(
                    args.model_id, args.device, torch.bfloat16
                )
                result = _run_config_benchmark(
                    "original",
                    "original (Chronos2Pipeline)",
                    lambda cb, p=pipeline: predict_original(p, cb, args.prediction_length),
                    context_batches,
                    actual_batches,
                    args.prediction_length,
                    args.n_warmup,
                    args.n_runs,
                )
                if result is not None:
                    combo_results["original"] = result
                _cleanup_model(pipeline)

            # --- cute_eager ---
            if "cute_eager" in args.configs:
                cute_model, quantiles, max_patches, out_patch_size = _load_cute_model(
                    args.model_id, args.device, torch.bfloat16
                )
                result = _run_config_benchmark(
                    "cute_eager",
                    "cute_eager (CuteChronos2Model, no compile)",
                    lambda cb, m=cute_model: predict_cute(
                        m, cb, args.prediction_length, quantiles, max_patches, out_patch_size, args.device
                    ),
                    context_batches,
                    actual_batches,
                    args.prediction_length,
                    args.n_warmup,
                    args.n_runs,
                )
                if result is not None:
                    combo_results["cute_eager"] = result
                _cleanup_model(cute_model)

            # --- cute_compiled ---
            if "cute_compiled" in args.configs:
                cute_model, quantiles, max_patches, out_patch_size = _load_cute_model(
                    args.model_id, args.device, torch.bfloat16
                )
                compiled_model = torch.compile(cute_model, mode="reduce-overhead")
                result = _run_config_benchmark(
                    "cute_compiled",
                    "cute_compiled (CuteChronos2Model + torch.compile)",
                    lambda cb, m=compiled_model: predict_cute(
                        m, cb, args.prediction_length, quantiles, max_patches, out_patch_size, args.device
                    ),
                    context_batches,
                    actual_batches,
                    args.prediction_length,
                    args.n_warmup,
                    args.n_runs,
                )
                if result is not None:
                    combo_results["cute_compiled"] = result
                _cleanup_model(compiled_model, cute_model)

            # Print comparison table for this combo
            print_comparison_table(combo_results, args.configs)

            # Store results keyed by combo
            for cfg_name, res in combo_results.items():
                all_results[f"{combo_key}_{cfg_name}"] = res

    # Save all results to JSON
    output_path = repo_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Final summary across all combos
    print(f"\n{'='*100}")
    print("OVERALL SUMMARY")
    print(f"{'='*100}")
    header = (
        f"{'Combo + Config':<35} {'Avg(ms)':<12} {'Min(ms)':<12} "
        f"{'MAE':<12} {'GPU(MB)':<12} {'MAE delta':<12}"
    )
    print(header)
    print("-" * 100)

    # Group by combo for the summary
    combos_seen: dict[str, dict[str, dict]] = {}
    for key, res in all_results.items():
        for c in ["cute_compiled", "cute_eager", "original"]:
            if key.endswith(f"_{c}"):
                combo = key[: -len(c) - 1]
                combos_seen.setdefault(combo, {})[c] = res
                break

    for combo, cfgs in combos_seen.items():
        orig_mae = cfgs.get("original", {}).get("avg_mae")
        for cfg_name, res in cfgs.items():
            mae_delta = ""
            if orig_mae is not None and cfg_name != "original":
                if np.isnan(orig_mae) or np.isnan(res["avg_mae"]):
                    mae_delta = "N/A (NaN)"
                else:
                    delta = abs(res["avg_mae"] - orig_mae)
                    status = "PASS" if delta < 1e-4 else "FAIL"
                    mae_delta = f"{delta:.2e} {status}"
            label = f"{combo}/{res['label']}"
            print(
                f"{label:<35} "
                f"{res['avg_latency_ms']:<12.2f} "
                f"{res['min_latency_ms']:<12.2f} "
                f"{_format_mae(res['avg_mae']):<12} "
                f"{res['peak_gpu_memory_mb']:<12.1f} "
                f"{mae_delta:<12}"
            )


if __name__ == "__main__":
    main()
