#!/usr/bin/env python3
"""
Quick test to verify attention.py fix eliminates recompilation warnings.
"""

import os
import sys
import logging
from pathlib import Path

import torch
import pandas as pd
import numpy as np

# Configure logging to see dynamo warnings
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Apply compile config
import toto_compile_config
toto_compile_config.apply(verbose=True)

print("=" * 80)
print("ATTENTION.PY FIX VERIFICATION")
print("=" * 80)
print()
print("Testing if attention.py graph break eliminates recompilation warnings...")
print()

# Load pipeline with compilation
from src.models.toto_wrapper import TotoPipeline

print("Loading Toto pipeline with torch_compile=True...")
pipeline = TotoPipeline.from_pretrained(
    "Datadog/Toto-Open-Base-1.0",
    device_map="cuda",
    torch_compile=True,
)

print(f"Pipeline loaded (compiled={pipeline.compiled})")
print()

# Load test data
symbol = "BTCUSD"
csv_path = Path("trainingdata") / f"{symbol}.csv"

if not csv_path.exists():
    print(f"ERROR: {csv_path} not found")
    sys.exit(1)

df = pd.read_csv(csv_path)
prices = df['close'].values[-512:].astype(np.float32)
context = torch.from_numpy(prices)

print(f"Testing {symbol} (last 512 prices)...")
print()

# Run a single inference to trigger compilation
print("Running inference (this will compile and may show initial warnings)...")
print()

forecast = pipeline.predict(
    context=context,
    prediction_length=8,
    num_samples=1024,
    samples_per_batch=128,
)

print()
print("=" * 80)
print("FIRST INFERENCE COMPLETE")
print("=" * 80)
print()
print("Check the output above:")
print("  ✓ GOOD: If you see minimal/no 'recompile_limit' warnings")
print("  ✗ BAD: If you still see 'torch._dynamo hit config.recompile_limit (8)'")
print("          with 'function: positional_embedding'")
print()

# Run second inference to check stability
print("Running second inference to verify no recompilations...")
print()

forecast2 = pipeline.predict(
    context=context,
    prediction_length=8,
    num_samples=1024,
    samples_per_batch=128,
)

print()
print("=" * 80)
print("SECOND INFERENCE COMPLETE")
print("=" * 80)
print()
print("If you see NO warnings above, the fix is working!")
print()
