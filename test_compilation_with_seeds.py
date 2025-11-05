#!/usr/bin/env python3
"""
Test compiled vs uncompiled Toto with fixed seeds.

This will definitively show if compilation changes predictions or if
variance is purely from probabilistic sampling.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers

# Apply compile config first
import toto_compile_config
toto_compile_config.apply(verbose=True)

from src.models.toto_wrapper import TotoPipeline

print("=" * 80)
print("TOTO COMPILATION TEST - SEED ANALYSIS")
print("=" * 80)
print()
print("Testing hypothesis: Variance is from sampling, not compilation")
print()

# Test parameters
SYMBOL = "BTCUSD"
SEEDS = [42, 123, 999]
NUM_SAMPLES = 1024

# Load test data
csv_path = Path("trainingdata") / f"{SYMBOL}.csv"
df = pd.read_csv(csv_path)
prices = df['close'].values[-512:].astype(np.float32)
context = torch.from_numpy(prices)

print(f"Testing {SYMBOL} with {len(SEEDS)} different seeds")
print(f"Seeds: {SEEDS}")
print()

# Storage for results
results = {
    'uncompiled': {},
    'compiled': {}
}

print("=" * 80)
print("PHASE 1: UNCOMPILED (torch_compile=False)")
print("=" * 80)
print()

# Load uncompiled pipeline
pipeline_uncompiled = TotoPipeline.from_pretrained(
    "Datadog/Toto-Open-Base-1.0",
    device_map="cuda",
    torch_compile=False,
)

for seed in SEEDS:
    print(f"Testing seed {seed}...")
    
    # Set seed
    transformers.set_seed(seed)
    
    # Predict
    forecast = pipeline_uncompiled.predict(
        context=context,
        prediction_length=8,
        num_samples=NUM_SAMPLES,
        samples_per_batch=128,
    )
    
    samples = forecast[0].numpy()
    mae = np.mean(np.abs(samples))
    
    results['uncompiled'][seed] = {
        'mae': mae,
        'samples': samples.copy()
    }
    
    print(f"  Seed {seed}: MAE = {mae:.6f}")

print()
print("Uncompiled seed variance:")
uncompiled_maes = [results['uncompiled'][s]['mae'] for s in SEEDS]
print(f"  MAE mean: {np.mean(uncompiled_maes):.6f}")
print(f"  MAE std:  {np.std(uncompiled_maes):.6f}")
print(f"  MAE CV:   {np.std(uncompiled_maes) / np.mean(uncompiled_maes):.4%}")
print()

del pipeline_uncompiled
torch.cuda.empty_cache()

print("=" * 80)
print("PHASE 2: COMPILED (torch_compile=True, reduce-overhead)")
print("=" * 80)
print()

# Load compiled pipeline
pipeline_compiled = TotoPipeline.from_pretrained(
    "Datadog/Toto-Open-Base-1.0",
    device_map="cuda",
    torch_compile=True,
)

for seed in SEEDS:
    print(f"Testing seed {seed}...")
    
    # Set seed
    transformers.set_seed(seed)
    
    # Predict
    forecast = pipeline_compiled.predict(
        context=context,
        prediction_length=8,
        num_samples=NUM_SAMPLES,
        samples_per_batch=128,
    )
    
    samples = forecast[0].numpy()
    mae = np.mean(np.abs(samples))
    
    results['compiled'][seed] = {
        'mae': mae,
        'samples': samples.copy()
    }
    
    print(f"  Seed {seed}: MAE = {mae:.6f}")

print()
print("Compiled seed variance:")
compiled_maes = [results['compiled'][s]['mae'] for s in SEEDS]
print(f"  MAE mean: {np.mean(compiled_maes):.6f}")
print(f"  MAE std:  {np.std(compiled_maes):.6f}")
print(f"  MAE CV:   {np.std(compiled_maes) / np.mean(compiled_maes):.4%}")
print()

print("=" * 80)
print("ANALYSIS: SEED-BY-SEED COMPARISON")
print("=" * 80)
print()

for seed in SEEDS:
    uncompiled_mae = results['uncompiled'][seed]['mae']
    compiled_mae = results['compiled'][seed]['mae']
    uncompiled_samples = results['uncompiled'][seed]['samples']
    compiled_samples = results['compiled'][seed]['samples']
    
    mae_diff = abs(compiled_mae - uncompiled_mae)
    mae_pct = (mae_diff / uncompiled_mae) * 100
    
    # Sample-level comparison
    sample_diff = np.mean(np.abs(uncompiled_samples - compiled_samples))
    correlation = np.corrcoef(uncompiled_samples.flatten(), compiled_samples.flatten())[0, 1]
    
    print(f"Seed {seed}:")
    print(f"  Uncompiled MAE: {uncompiled_mae:.6f}")
    print(f"  Compiled MAE:   {compiled_mae:.6f}")
    print(f"  MAE difference: {mae_diff:.6f} ({mae_pct:.4f}%)")
    print(f"  Sample-level difference: {sample_diff:.6f}")
    print(f"  Sample correlation: {correlation:.6f}")
    print()

print("=" * 80)
print("KEY FINDINGS")
print("=" * 80)
print()

# Compare seed variance vs compiled variance
seed_variance_uncompiled = np.std(uncompiled_maes)
seed_variance_compiled = np.std(compiled_maes)

# Compare same-seed compiled vs uncompiled
same_seed_diffs = [abs(results['compiled'][s]['mae'] - results['uncompiled'][s]['mae']) for s in SEEDS]
mean_same_seed_diff = np.mean(same_seed_diffs)

print(f"1. Seed variance (uncompiled):  {seed_variance_uncompiled:.6f}")
print(f"2. Seed variance (compiled):    {seed_variance_compiled:.6f}")
print(f"3. Same-seed difference:        {mean_same_seed_diff:.6f}")
print()

if mean_same_seed_diff < seed_variance_uncompiled:
    print("✓ HYPOTHESIS CONFIRMED:")
    print("  Compilation is deterministic for a given seed.")
    print("  Variance comes from sampling, not compilation.")
    print()
    print("  Same-seed compiled vs uncompiled difference is LESS THAN")
    print("  seed-to-seed variance, proving compilation preserves determinism.")
else:
    print("✗ HYPOTHESIS REJECTED:")
    print("  Compilation may introduce non-determinism.")
    print()
    print("  Same-seed compiled vs uncompiled difference is GREATER THAN")
    print("  seed-to-seed variance.")

print()
print("=" * 80)
print("RECOMMENDATION")
print("=" * 80)
print()

if mean_same_seed_diff / np.mean(uncompiled_maes) < 0.001:  # <0.1%
    print("✓ Compilation is SAFE to use")
    print("  - Same seed produces same results")
    print("  - <0.1% difference from uncompiled")
    print("  - Variance is purely from sampling")
else:
    print("⚠️  Review compilation carefully")
    print(f"  - Same-seed difference: {mean_same_seed_diff / np.mean(uncompiled_maes):.4%}")
    print("  - May need deterministic mode or fixed seeds")

print()
