#!/usr/bin/env python3
"""Test maximum batch size for Chronos2 on available VRAM."""

import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch

# Enable torch compile for realistic test
os.environ["TORCH_COMPILED"] = "1"
os.environ["CHRONOS_COMPILE"] = "1"

from src.models.chronos2_wrapper import Chronos2OHLCWrapper

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_DATA = REPO_ROOT / "trainingdata" / "BTCUSD.csv"
BATCH_SIZES = [128, 256, 512, 1024, 1536, 2048]
CONTEXT_LENGTH = 2048


def test_batch_size(wrapper: Chronos2OHLCWrapper, df: pd.DataFrame, batch_size: int) -> tuple[bool, float]:
    """Test if a batch size works without OOM."""
    try:
        torch.cuda.empty_cache()
        start = time.perf_counter()

        batch = wrapper.predict_ohlc(
            df.iloc[:-7],
            symbol="BTCUSD",
            prediction_length=7,
            context_length=CONTEXT_LENGTH,
            batch_size=batch_size,
        )

        latency = time.perf_counter() - start
        print(f"✓ Batch size {batch_size:4d}: {latency:.2f}s (VRAM: {torch.cuda.max_memory_allocated()/1e9:.1f}GB)")
        return True, latency

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"✗ Batch size {batch_size:4d}: OOM")
        return False, 0.0
    except Exception as exc:
        print(f"✗ Batch size {batch_size:4d}: {exc}")
        return False, 0.0


def main():
    if not TEST_DATA.exists():
        print(f"Error: Test data not found at {TEST_DATA}")
        sys.exit(1)

    print("=" * 60)
    print("Chronos2 Batch Size Test (RTX 5090 32GB)")
    print("=" * 60)
    print(f"Context length: {CONTEXT_LENGTH}")
    print(f"Torch compile: ENABLED")
    print(f"Testing batch sizes: {BATCH_SIZES}")
    print()

    df = pd.read_csv(TEST_DATA)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    print("Loading Chronos2 (compiled)...")
    wrapper = Chronos2OHLCWrapper.from_pretrained(
        "amazon/chronos-2",
        device_map="cuda",
        default_context_length=CONTEXT_LENGTH,
        default_batch_size=128,
        torch_compile=True,
    )
    print()

    max_working_batch = 128
    best_latency = float("inf")
    best_batch = 128

    for batch_size in BATCH_SIZES:
        success, latency = test_batch_size(wrapper, df, batch_size)
        if success:
            max_working_batch = batch_size
            if latency < best_latency:
                best_latency = latency
                best_batch = batch_size
        else:
            break  # Stop at first OOM

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Maximum working batch size: {max_working_batch}")
    print(f"Fastest batch size: {best_batch} ({best_latency:.2f}s)")
    print(f"Recommended for tuning: {max_working_batch}")
    print()
    print(f"VRAM usage at max batch: {torch.cuda.max_memory_allocated()/1e9:.1f}GB / 32GB")
    print()


if __name__ == "__main__":
    main()
