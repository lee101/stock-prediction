#!/usr/bin/env python3
"""
Quick evaluation script to test current models on all stock pairs
and track MAE improvements
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def load_stock_data(csv_path: Path) -> pd.DataFrame:
    """Load and prepare stock data"""
    df = pd.read_csv(csv_path)

    # Ensure we have required columns
    if 'close' in df.columns:
        price_col = 'close'
    elif 'Close' in df.columns:
        price_col = 'Close'
    else:
        raise ValueError(f"No close price column found in {csv_path}")

    return df, price_col


def compute_naive_baseline(prices: np.ndarray, horizon: int = 64) -> float:
    """Compute naive baseline (persistence model) MAE"""
    if len(prices) <= horizon:
        return float('inf')

    # Last value persistence
    errors = []
    for i in range(len(prices) - horizon):
        pred = np.full(horizon, prices[i + horizon - 1])
        actual = prices[i + horizon : i + 2 * horizon] if i + 2 * horizon <= len(prices) else prices[i + horizon:]
        if len(actual) > 0:
            errors.append(np.abs(pred[:len(actual)] - actual).mean())

    return np.mean(errors) if errors else float('inf')


def evaluate_all_stocks(data_dir: Path = Path("trainingdata")) -> Dict:
    """Evaluate baseline performance on all stocks"""

    results = {}
    csv_files = sorted(data_dir.glob("*.csv"))

    # Filter out summary files
    csv_files = [f for f in csv_files if "summary" not in f.name.lower()]

    print(f"Evaluating {len(csv_files)} stock pairs...")
    print("="*80)

    for csv_file in csv_files:
        try:
            df, price_col = load_stock_data(csv_file)
            prices = df[price_col].values

            # Compute statistics
            stats = {
                "stock": csv_file.stem,
                "num_samples": len(prices),
                "price_mean": float(np.mean(prices)),
                "price_std": float(np.std(prices)),
                "price_min": float(np.min(prices)),
                "price_max": float(np.max(prices)),
            }

            # Compute naive baseline for different horizons
            for horizon in [16, 32, 64, 128]:
                naive_mae = compute_naive_baseline(prices, horizon)
                stats[f"naive_mae_h{horizon}"] = naive_mae

                # Compute as percentage
                if stats["price_mean"] > 0:
                    stats[f"naive_mae_pct_h{horizon}"] = (naive_mae / stats["price_mean"]) * 100

            results[csv_file.stem] = stats

            print(f"{csv_file.stem:15s} | Samples: {len(prices):6d} | "
                  f"Price: ${stats['price_mean']:8.2f} | "
                  f"Naive MAE (h64): ${stats['naive_mae_h64']:.3f} ({stats.get('naive_mae_pct_h64', 0):.2f}%)")

        except Exception as e:
            print(f"Error processing {csv_file}: {e}")

    print("="*80)
    return results


def save_baseline_results(results: Dict, output_file: Path = Path("tototraining/baseline_results.json")):
    """Save baseline results"""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Create summary DataFrame
    df = pd.DataFrame(results).T
    summary_csv = output_file.with_suffix('.csv')
    df.to_csv(summary_csv)

    print(f"\nResults saved to:")
    print(f"  JSON: {output_file}")
    print(f"  CSV:  {summary_csv}")

    return df


def analyze_results(results_df: pd.DataFrame):
    """Analyze and summarize results"""

    print("\n" + "="*80)
    print("BASELINE ANALYSIS")
    print("="*80)

    # Overall statistics
    print("\nOverall Statistics:")
    print(f"  Total stocks: {len(results_df)}")
    print(f"  Avg samples per stock: {results_df['num_samples'].mean():.0f}")
    print(f"  Avg price: ${results_df['price_mean'].mean():.2f}")

    # Naive MAE statistics for h64
    print("\nNaive Baseline (h=64):")
    print(f"  Mean MAE: ${results_df['naive_mae_h64'].mean():.3f}")
    print(f"  Median MAE: ${results_df['naive_mae_h64'].median():.3f}")
    print(f"  Mean MAE%: {results_df['naive_mae_pct_h64'].mean():.2f}%")
    print(f"  Median MAE%: {results_df['naive_mae_pct_h64'].median():.2f}%")

    # Best and worst stocks (by percentage MAE)
    print("\nEasiest to predict (lowest naive MAE%):")
    easiest = results_df.nsmallest(5, 'naive_mae_pct_h64')[['naive_mae_h64', 'naive_mae_pct_h64']]
    print(easiest.to_string())

    print("\nHardest to predict (highest naive MAE%):")
    hardest = results_df.nlargest(5, 'naive_mae_pct_h64')[['naive_mae_h64', 'naive_mae_pct_h64']]
    print(hardest.to_string())

    print("\n" + "="*80)


def create_improvement_tracker():
    """Create a tracker for monitoring improvements over time"""

    tracker_file = Path("tototraining/improvement_tracker.json")

    if tracker_file.exists():
        with open(tracker_file, 'r') as f:
            tracker = json.load(f)
    else:
        tracker = {
            "experiments": [],
            "best_overall": {
                "mae": float('inf'),
                "config": {},
                "timestamp": None
            }
        }

    return tracker


def update_improvement_tracker(experiment_name: str,
                               config: Dict,
                               mae: float,
                               metrics: Dict):
    """Update improvement tracker with new results"""

    tracker = create_improvement_tracker()

    entry = {
        "name": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "mae": mae,
        "metrics": metrics
    }

    tracker["experiments"].append(entry)

    # Update best if this is better
    if mae < tracker["best_overall"]["mae"]:
        tracker["best_overall"] = {
            "mae": mae,
            "config": config,
            "timestamp": entry["timestamp"],
            "experiment": experiment_name
        }

    # Save
    tracker_file = Path("tototraining/improvement_tracker.json")
    with open(tracker_file, 'w') as f:
        json.dump(tracker, f, indent=2)

    print(f"\n✓ Tracked improvement: {experiment_name}")
    print(f"  MAE: {mae:.4f}")
    if mae < tracker["best_overall"]["mae"]:
        print(f"  🎉 NEW BEST! Previous: {tracker['best_overall']['mae']:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick evaluation and baseline analysis")
    parser.add_argument("--mode", choices=["baseline", "analyze"],
                       default="baseline",
                       help="Mode: baseline evaluation or analyze results")

    args = parser.parse_args()

    if args.mode == "baseline":
        # Run baseline evaluation
        results = evaluate_all_stocks()
        df = save_baseline_results(results)
        analyze_results(df)

    elif args.mode == "analyze":
        # Load and analyze existing results
        baseline_file = Path("tototraining/baseline_results.csv")
        if baseline_file.exists():
            df = pd.read_csv(baseline_file, index_col=0)
            analyze_results(df)
        else:
            print("No baseline results found. Run with --mode baseline first.")
