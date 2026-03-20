"""End-to-end benchmark comparing CuteChronos2Pipeline vs upstream Chronos2Pipeline.

Usage::

    python -m cutechronos.benchmark --symbols BTCUSD ETHUSD --context-length 512

Measures latency, MAE of median prediction vs actual, and peak GPU memory.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_series(csv_path: str, context_length: int, prediction_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Load close prices from a CSV and split into context + future actual.

    Returns
    -------
    context : torch.Tensor of shape (context_length,)
    actual  : torch.Tensor of shape (prediction_length,)
    """
    df = pd.read_csv(csv_path)
    close = df["close"].values.astype(np.float32)
    total_needed = context_length + prediction_length
    if len(close) < total_needed:
        raise ValueError(
            f"{csv_path}: need {total_needed} rows but only have {len(close)}"
        )
    # Take the last total_needed points so we test on the most recent data
    close = close[-total_needed:]
    context = torch.tensor(close[:context_length], dtype=torch.float32)
    actual = torch.tensor(close[context_length:], dtype=torch.float32)
    return context, actual


def measure_gpu_memory() -> float:
    """Return peak GPU memory allocated in MB (0 if no CUDA)."""
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def reset_gpu_memory_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()


def median_from_predictions(predictions: list[torch.Tensor], quantiles: list[float]) -> torch.Tensor:
    """Extract median (q=0.5) prediction from a list of (1, Q, H) tensors."""
    median_idx = quantiles.index(0.5)
    # predictions[0] shape: (1, Q, H)
    return predictions[0][0, median_idx, :]  # (H,)


def compute_mae(pred: torch.Tensor, actual: torch.Tensor) -> float:
    length = min(len(pred), len(actual))
    diff = (pred[:length] - actual[:length]).abs()
    valid = ~torch.isnan(diff)
    if not valid.any():
        return float("nan")
    return diff[valid].mean().item()


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

def benchmark_pipeline(
    pipeline,
    contexts: list[torch.Tensor],
    prediction_length: int,
    quantiles: list[float],
    actuals: list[torch.Tensor],
    n_warmup: int = 2,
    n_runs: int = 10,
    label: str = "pipeline",
    wrap_input_as_list: bool = False,
) -> dict:
    """Run benchmark for a single pipeline, return results dict.

    Parameters
    ----------
    wrap_input_as_list
        If True, each context tensor is wrapped in a list before calling
        predict().  The upstream Chronos2Pipeline requires list-of-tensors
        input, while CuteChronos2Pipeline accepts bare tensors.
    """

    def _call_predict(ctx):
        inp = [ctx] if wrap_input_as_list else ctx
        return pipeline.predict(inp, prediction_length=prediction_length, limit_prediction_length=False)

    # Warmup
    for _ in range(n_warmup):
        for ctx in contexts:
            _call_predict(ctx)

    # Timed runs
    reset_gpu_memory_stats()
    latencies = []
    all_maes: list[list[float]] = [[] for _ in contexts]

    for run_idx in range(n_runs):
        for sym_idx, ctx in enumerate(contexts):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            preds = _call_predict(ctx)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append(t1 - t0)

            median_pred = median_from_predictions(preds, quantiles)
            mae = compute_mae(median_pred, actuals[sym_idx])
            all_maes[sym_idx].append(mae)

    peak_mem = measure_gpu_memory()

    result = {
        "label": label,
        "avg_latency_ms": np.mean(latencies) * 1000,
        "std_latency_ms": np.std(latencies) * 1000,
        "min_latency_ms": np.min(latencies) * 1000,
        "peak_gpu_memory_mb": peak_mem,
        "per_symbol_mae": {f"sym_{i}": float(np.mean(m)) for i, m in enumerate(all_maes)},
        "avg_mae": float(np.mean([np.mean(m) for m in all_maes])),
    }
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CuteChronos2 benchmark")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTCUSD", "ETHUSD"],
        help="Symbol names (CSV files in trainingdata/)",
    )
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--prediction-length", type=int, default=30)
    parser.add_argument("--model-id", default="amazon/chronos-2", help="HF model id or local path")
    parser.add_argument("--n-warmup", type=int, default=2)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-dir", default=None, help="Directory with CSV files (close column required)")
    parser.add_argument("--output", default="benchmark_results.json")
    args = parser.parse_args()

    # Resolve data directory
    repo_root = Path(__file__).resolve().parent.parent
    if args.data_dir is None:
        # Generate synthetic data if no data directory specified
        data_dir = None
    else:
        data_dir = Path(args.data_dir)
        if not data_dir.is_absolute():
            data_dir = repo_root / data_dir

    print(f"Loading data from {data_dir or 'synthetic'}")
    print(f"Symbols: {args.symbols}")
    print(f"Context length: {args.context_length}, Prediction length: {args.prediction_length}")
    print(f"Device: {args.device}")
    print()

    # Load data
    contexts = []
    actuals = []
    if data_dir is not None:
        for sym in args.symbols:
            csv_path = data_dir / f"{sym}.csv"
            if not csv_path.exists():
                print(f"WARNING: {csv_path} not found, skipping {sym}")
                continue
            ctx, act = load_series(str(csv_path), args.context_length, args.prediction_length)
            contexts.append(ctx)
            actuals.append(act)
            print(f"  {sym}: context {ctx.shape}, actual {act.shape}")

    if not contexts:
        # Generate synthetic time series data for benchmarking
        print("  No CSV data found, generating synthetic series...")
        torch.manual_seed(42)
        for i, sym in enumerate(args.symbols):
            total = args.context_length + args.prediction_length
            # Random walk with drift (mimics price data)
            returns = torch.randn(total) * 0.02 + 0.0001
            prices = 100.0 * torch.exp(returns.cumsum(0))
            ctx = prices[:args.context_length]
            act = prices[args.context_length:]
            contexts.append(ctx)
            actuals.append(act)
            print(f"  {sym} (synthetic): context {ctx.shape}, actual {act.shape}")

    results = {}

    # 1. Original Chronos2Pipeline
    print("\n--- Loading original Chronos2Pipeline ---")
    try:
        from chronos.chronos2 import Chronos2Pipeline

        original_pipe = Chronos2Pipeline.from_pretrained(args.model_id, dtype=torch.bfloat16)
        original_pipe.model = original_pipe.model.to(args.device)
        original_quantiles = original_pipe.quantiles
        print(f"  Loaded. quantiles={len(original_quantiles)}, device={original_pipe.model.device}")

        result_orig = benchmark_pipeline(
            pipeline=original_pipe,
            contexts=contexts,
            prediction_length=args.prediction_length,
            quantiles=original_quantiles,
            actuals=actuals,
            n_warmup=args.n_warmup,
            n_runs=args.n_runs,
            label="original_chronos2",
            wrap_input_as_list=True,
        )
        results["original_chronos2"] = result_orig
        print(f"  Avg latency: {result_orig['avg_latency_ms']:.1f} ms")
        print(f"  Avg MAE: {result_orig['avg_mae']:.4f}")
        print(f"  Peak GPU mem: {result_orig['peak_gpu_memory_mb']:.1f} MB")

        # free original
        del original_pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"  Failed to load original pipeline: {e}")

    # 2. CuteChronos2Pipeline
    print("\n--- Loading CuteChronos2Pipeline ---")
    try:
        from cutechronos.pipeline import CuteChronos2Pipeline

        cute_pipe = CuteChronos2Pipeline.from_pretrained(args.model_id, device=args.device, dtype=torch.bfloat16)
        cute_quantiles = cute_pipe.quantiles
        print(f"  Loaded. quantiles={len(cute_quantiles)}, device={cute_pipe.device}")

        result_cute = benchmark_pipeline(
            pipeline=cute_pipe,
            contexts=contexts,
            prediction_length=args.prediction_length,
            quantiles=cute_quantiles,
            actuals=actuals,
            n_warmup=args.n_warmup,
            n_runs=args.n_runs,
            label="cute_chronos2",
        )
        results["cute_chronos2"] = result_cute
        print(f"  Avg latency: {result_cute['avg_latency_ms']:.1f} ms")
        print(f"  Avg MAE: {result_cute['avg_mae']:.4f}")
        print(f"  Peak GPU mem: {result_cute['peak_gpu_memory_mb']:.1f} MB")

        del cute_pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"  Failed to load cute pipeline: {e}")

    # 3. Comparison table
    print("\n" + "=" * 72)
    print("BENCHMARK RESULTS")
    print("=" * 72)
    header = f"{'Pipeline':<25} {'Latency(ms)':<15} {'MAE':<15} {'GPU(MB)':<15}"
    print(header)
    print("-" * 72)
    for key, res in results.items():
        line = f"{res['label']:<25} {res['avg_latency_ms']:<15.1f} {res['avg_mae']:<15.6f} {res['peak_gpu_memory_mb']:<15.1f}"
        print(line)

    if "original_chronos2" in results and "cute_chronos2" in results:
        orig = results["original_chronos2"]
        cute = results["cute_chronos2"]
        mae_delta = abs(cute["avg_mae"] - orig["avg_mae"])
        speedup = orig["avg_latency_ms"] / max(cute["avg_latency_ms"], 1e-9)
        print()
        print(f"MAE delta (|cute - orig|): {mae_delta:.6f}")
        print(f"Speedup: {speedup:.2f}x")
        results["comparison"] = {
            "mae_delta": mae_delta,
            "speedup": speedup,
        }

    # Save results
    output_path = repo_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
