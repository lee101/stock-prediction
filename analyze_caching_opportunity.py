#!/usr/bin/env python3
"""
Analyze caching opportunity in walk-forward backtest.
Shows how much computation is duplicated across simulations.
"""

import numpy as np

def analyze_overlap(total_days=200, num_sims=70):
    """
    Analyze data overlap in walk-forward validation.

    Sim 0: days [0, 1, 2, ..., 199] (all 200 days)
    Sim 1: days [0, 1, 2, ..., 198] (199 days)
    Sim 2: days [0, 1, 2, ..., 197] (198 days)
    ...
    Sim 69: days [0, 1, 2, ..., 130] (131 days)
    """

    print("="*80)
    print("WALK-FORWARD BACKTEST DATA OVERLAP ANALYSIS")
    print("="*80)
    print(f"Total days: {total_days}")
    print(f"Simulations: {num_sims}")
    print()

    # Calculate data usage
    sim_lengths = []
    for sim in range(num_sims):
        length = total_days - sim
        sim_lengths.append(length)

    print(f"Simulation data lengths:")
    print(f"  Sim 0: {sim_lengths[0]} days (newest)")
    print(f"  Sim {num_sims//2}: {sim_lengths[num_sims//2]} days")
    print(f"  Sim {num_sims-1}: {sim_lengths[-1]} days (oldest)")
    print()

    # Calculate how many times each day is processed
    day_usage = np.zeros(total_days, dtype=int)
    for sim in range(num_sims):
        length = total_days - sim
        day_usage[:length] += 1

    print("Days processed multiple times:")
    print(f"  Days 0-{sim_lengths[-1]-1}: Used in ALL {num_sims} simulations")
    print(f"  Days {sim_lengths[-1]}-{sim_lengths[-2]-1}: Used in {num_sims-1} simulations")
    print(f"  Last {num_sims} days: Each used only once")
    print()

    # Calculate redundant computation
    unique_days = total_days
    total_day_processings = day_usage.sum()
    redundant = total_day_processings - unique_days

    print("Computation analysis:")
    print(f"  Unique days: {unique_days}")
    print(f"  Total day×model calls: {total_day_processings}")
    print(f"  Redundant calls: {redundant} ({redundant/total_day_processings*100:.1f}%)")
    print()

    # With 4 keys (Close, High, Low, Open)
    keys = 4
    unique_predictions = unique_days * keys
    actual_predictions = total_day_processings * keys

    print(f"With {keys} keys per day:")
    print(f"  Unique predictions needed: {unique_predictions}")
    print(f"  Actual predictions made: {actual_predictions}")
    print(f"  Redundant predictions: {actual_predictions - unique_predictions}")
    print()

    # Potential speedup with caching
    print("="*80)
    print("CACHING POTENTIAL")
    print("="*80)

    # If we cache all predictions
    cache_all_speedup = actual_predictions / unique_predictions
    print(f"\nScenario 1: Cache all predictions")
    print(f"  Speedup: {cache_all_speedup:.2f}x on model inference")
    print(f"  Implementation: Cache by (day_index, key)")
    print()

    # Realistic: Some overhead, not all time is model inference
    model_time_pct = 0.70  # 70% of time is model inference
    realistic_speedup = 1 / (1 - model_time_pct + model_time_pct / cache_all_speedup)
    print(f"Scenario 2: Assuming {model_time_pct*100:.0f}% time is model inference")
    print(f"  Total speedup: {realistic_speedup:.2f}x")
    print()

    # Show day usage histogram
    print("="*80)
    print("DAY REUSE HISTOGRAM")
    print("="*80)

    print(f"\n{'Times used':<15} {'Days':<10} {'Cumulative':<15}")
    print("-"*40)

    unique, counts = np.unique(day_usage[::-1], return_counts=True)
    cumulative = 0
    for uses, count in zip(unique[::-1], counts[::-1]):
        cumulative += count
        pct = cumulative / total_days * 100
        print(f"{uses}x{'':<13} {count:<10} {cumulative:<10} ({pct:.1f}%)")

    # Implementation suggestion
    print("\n" + "="*80)
    print("IMPLEMENTATION STRATEGY")
    print("="*80)

    print("""
1. **Full Cache** (Maximum speedup: {:.1f}x)
   - Cache predictions by (data_hash, key, day_index)
   - Before each simulation, check cache for all days
   - Only predict missing days
   - Best for repeated runs on same data

2. **Incremental Cache** (Simpler)
   - Process simulations in reverse order (oldest first)
   - Cache predictions as you go
   - Each new simulation reuses previous predictions
   - Reduces redundant calls within single backtest run

3. **Hybrid** (Practical)
   - Cache only days used in ≥10 simulations (oldest ~60%)
   - Reduces cache size while capturing most benefit
   - ~{:.1f}x speedup with less memory

Current bottleneck: {} model calls per backtest
With caching: {} unique calls + cache lookups
Reduction: {} calls (-{:.0f}%)

Key insight: Predictions DON'T change as we walk forward!
    - Prediction for day 50 is the same in sim 0, 1, 2, ... 19
    - We're recomputing it {} times!
    """.format(
        cache_all_speedup,
        cache_all_speedup * 0.6,
        actual_predictions,
        unique_predictions,
        actual_predictions - unique_predictions,
        (actual_predictions - unique_predictions) / actual_predictions * 100,
        min(num_sims, total_days - sim_lengths[-1])
    ))


if __name__ == '__main__':
    import sys
    total_days = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    num_sims = int(sys.argv[2]) if len(sys.argv) > 2 else 70

    analyze_overlap(total_days, num_sims)
