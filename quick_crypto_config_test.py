#!/usr/bin/env python3
"""
Quick test of improved configs for ETHUSD.

ETHUSD currently has the worst performance at 3.75% pct_return_mae.
Let's test if simply increasing samples helps.
"""
import json

# Load current ETHUSD config
with open("hyperparams/best/ETHUSD.json") as f:
    current = json.load(f)

print("Current ETHUSD config:")
print(f"  Model: {current['model']}")
print(f"  Samples: {current['config']['num_samples']}")
print(f"  Aggregate: {current['config']['aggregate']}")
print(f"  Test pct MAE: {current['test']['pct_return_mae']:.4f} (3.75%)")
print(f"  Latency: {current['test']['latency_s']:.2f}s")

print("\nSuggested improvements to test:")
print("\n1. INCREASE SAMPLES (most impactful)")
print("   - Try 512 samples (4x current)")
print("   - Try 1024 samples (8x current, like BTCUSD)")
print("   - Expected: Better MAE, higher latency")

print("\n2. ADJUST AGGREGATION")
print("   - Current: trimmed_mean_20 (aggressive trimming)")
print("   - Try: trimmed_mean_10 (less aggressive)")
print("   - Try: trimmed_mean_5 (like BTCUSD)")
print("   - Expected: More stable predictions")

print("\n3. ENSEMBLE APPROACH")
print("   - Combine Kronos + Toto predictions")
print("   - Weight by validation performance")
print("   - Expected: 10-20% improvement")

print("\n4. CRYPTO-SPECIFIC FINE-TUNING")
print("   - Retrain Toto on crypto-only dataset")
print("   - Focus on high-volatility periods")
print("   - Expected: 20-30% improvement")

print("\nPriority order:")
print("  1. Quick win: Test 512/1024 samples")
print("  2. Medium: Try different aggregations")
print("  3. Advanced: Ensemble models")
print("  4. Long-term: Full retraining")
