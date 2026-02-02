#!/usr/bin/env python3
"""Memory leak check for NeuralTraderV3 inference loop.

Runs repeated inference cycles and reports RSS/GPU usage deltas in markdown.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import psutil
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from run_neural_trader_v3 import NeuralTraderV3, DEFAULT_CHECKPOINT
from bagsfm import BagsConfig


def _bytes_to_mb(value: int) -> float:
    return value / (1024 * 1024)


def _snapshot(tag: str) -> dict:
    proc = psutil.Process()
    rss = proc.memory_info().rss
    cuda_alloc = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    cuda_reserved = torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
    return {
        "tag": tag,
        "rss_mb": _bytes_to_mb(rss),
        "cuda_alloc_mb": _bytes_to_mb(cuda_alloc),
        "cuda_reserved_mb": _bytes_to_mb(cuda_reserved),
    }


def _print_markdown(rows: list[dict]) -> None:
    print("\n### NeuralTraderV3 Memory Check\n")
    print("| Step | RSS (MB) | CUDA Alloc (MB) | CUDA Reserved (MB) |")
    print("| --- | ---: | ---: | ---: |")
    for row in rows:
        print(
            f"| {row['tag']} | {row['rss_mb']:.2f} | "
            f"{row['cuda_alloc_mb']:.2f} | {row['cuda_reserved_mb']:.2f} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Check memory usage for NeuralTraderV3")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--reload-data", action="store_true", help="Reload OHLC data each iteration")
    parser.add_argument("--sleep-offload", action="store_true", help="Move model to CPU between cycles")
    parser.add_argument("--sleep-unload", action="store_true", help="Unload model between cycles")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument(
        "--ohlc",
        type=Path,
        default=Path("bagstraining/ohlc_data_recent_2m_clean_holdout.csv"),
        help="OHLC CSV to load (default: recent holdout slice)",
    )

    args = parser.parse_args()

    if args.sleep_offload and args.sleep_unload:
        raise SystemExit("Use only one of --sleep-offload or --sleep-unload")

    trader = NeuralTraderV3(
        checkpoint_path=args.checkpoint,
        bags_config=BagsConfig(),
        dry_run=True,
        device=args.device,
        sleep_offload=args.sleep_offload,
        sleep_unload=args.sleep_unload,
    )

    import run_neural_trader_v3 as v3_module
    v3_module.OHLC_DATA_PATH = args.ohlc

    df = trader.load_ohlc_data()
    if df is None or len(df) < trader.context_length:
        raise SystemExit("Not enough OHLC data to run memory check.")

    rows = [_snapshot("start")]

    for i in range(args.iterations):
        if args.reload_data:
            df = trader.load_ohlc_data()
            if df is None or len(df) < trader.context_length:
                raise SystemExit("Not enough OHLC data to run memory check.")

        trader.ensure_model_loaded()
        _ = trader.predict(df, len(df))

        trader.release_model_for_sleep()
        if args.sleep_seconds:
            time.sleep(args.sleep_seconds)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        rows.append(_snapshot(f"iter {i+1}"))

    _print_markdown(rows)


if __name__ == "__main__":
    main()
