#!/usr/bin/env python3
"""
Experiment 3: Narrow Cooldown Range
Goal: Optimize cooldown within winning range (120-200s)
"""

import sys

sys.path.insert(0, "..")

import pandas as pd
from scipy.optimize import differential_evolution
from simulator import SimConfig


def run_experiment():
    strategy_df = pd.read_parquet("full_strategy_dataset_20251101_211202_strategy_performance.parquet")

    def objective(params):
        config = SimConfig(
            crypto_ooh_force_count=int(params[0]),
            crypto_ooh_tolerance_pct=params[1],
            crypto_normal_tolerance_pct=params[2],
            entry_tolerance_pct=params[3],
            protection_pct=params[4],
            cooldown_seconds=int(params[5]),
            fight_threshold=int(params[6]),
            fight_cooldown_seconds=int(params[7]),
        )

        from simulator import run_simulation

        results = run_simulation(config, "../trainingdatahourly", strategy_df)

        score = (
            results["total_pnl"] * 0.5 + results["sharpe"] * 1000 + results["win_rate"] * 2000 - results["blocks"] * 10
        )

        print(f"\nConfig: {params}", flush=True)
        print(f"Results: {results}", flush=True)
        print(f"Score: {score}", flush=True)

        return -score

    # Focus on cooldown optimization
    bounds = [
        (2, 3),
        (0.03, 0.04),
        (0.009, 0.012),
        (0.007, 0.011),
        (0.0014, 0.0019),  # protection around 0.15%
        (120, 200),  # FOCUS: narrow cooldown range
        (7, 9),
        (600, 800),
    ]

    print("=" * 60)
    print("EXPERIMENT 3: NARROW COOLDOWN RANGE")
    print("=" * 60)
    print("Cooldown range: 120-200s (2-3.3 minutes)")
    print("Baseline to beat: $19.1M PnL, 12 steals, 0 blocks")
    print("=" * 60)

    result = differential_evolution(
        objective, bounds, maxiter=12, popsize=4, seed=45, workers=1, updating="deferred", polish=False
    )

    optimal = SimConfig(
        crypto_ooh_force_count=int(result.x[0]),
        crypto_ooh_tolerance_pct=result.x[1],
        crypto_normal_tolerance_pct=result.x[2],
        entry_tolerance_pct=result.x[3],
        protection_pct=result.x[4],
        cooldown_seconds=int(result.x[5]),
        fight_threshold=int(result.x[6]),
        fight_cooldown_seconds=int(result.x[7]),
    )

    print("\n" + "=" * 60)
    print("EXPERIMENT 3 - OPTIMAL CONFIGURATION:")
    print("=" * 60)
    for field, value in optimal.__dict__.items():
        print(f"{field}: {value}")

    from simulator import run_simulation

    final_results = run_simulation(optimal, "../trainingdatahourly", strategy_df)
    print("\nFINAL PERFORMANCE:")
    for metric, value in final_results.items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    run_experiment()
