#!/usr/bin/env python3
"""
Experiment 1: Narrow Protection Range
Goal: Find optimal protection within 0.12-0.22% range
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

    # NARROW RANGES based on winners
    bounds = [
        (2, 3),  # crypto_ooh_force_count - winners use 2-3
        (0.025, 0.045),  # crypto_ooh_tolerance_pct - narrow around 3-4%
        (0.009, 0.012),  # crypto_normal_tolerance_pct - 0.9-1.2%
        (0.006, 0.012),  # entry_tolerance_pct - 0.6-1.2%
        (0.0012, 0.0022),  # protection_pct - CRITICAL: 0.12-0.22%
        (140, 200),  # cooldown_seconds - winners use 140-200s
        (7, 10),  # fight_threshold - higher end
        (550, 900),  # fight_cooldown_seconds - narrow around winners
    ]

    print("=" * 60)
    print("EXPERIMENT 1: NARROW PROTECTION RANGE")
    print("=" * 60)
    print("Protection range: 0.12-0.22%")
    print("Baseline to beat: $19.1M PnL, 12 steals, 0 blocks")
    print("=" * 60)

    result = differential_evolution(
        objective, bounds, maxiter=15, popsize=5, seed=43, workers=1, updating="deferred", polish=False
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
    print("EXPERIMENT 1 - OPTIMAL CONFIGURATION:")
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
