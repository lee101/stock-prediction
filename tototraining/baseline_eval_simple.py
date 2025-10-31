#!/usr/bin/env python3
"""
Simple baseline evaluation using only standard library
"""

import csv
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List


def load_csv(csv_path: Path) -> List[float]:
    """Load close prices from CSV"""
    prices = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)

        # Try different column name variations
        for row in reader:
            price = None
            for col in ['close', 'Close', 'CLOSE', 'price', 'Price']:
                if col in row:
                    try:
                        price = float(row[col])
                        break
                    except (ValueError, TypeError):
                        pass

            if price is not None:
                prices.append(price)

    return prices


def compute_stats(prices: List[float]) -> Dict:
    """Compute basic statistics"""
    if not prices:
        return {}

    mean_price = sum(prices) / len(prices)
    variance = sum((p - mean_price) ** 2 for p in prices) / len(prices)
    std_price = variance ** 0.5

    return {
        "count": len(prices),
        "mean": mean_price,
        "std": std_price,
        "min": min(prices),
        "max": max(prices),
    }


def compute_naive_mae(prices: List[float], horizon: int = 64) -> float:
    """Compute naive baseline MAE (persistence model)"""
    if len(prices) <= horizon:
        return float('inf')

    errors = []

    # Use last value as prediction for next horizon steps
    for i in range(0, len(prices) - horizon, horizon):
        last_price = prices[i + horizon - 1] if i + horizon - 1 < len(prices) else prices[-1]

        # Predict constant value
        for j in range(horizon):
            if i + horizon + j < len(prices):
                actual = prices[i + horizon + j]
                error = abs(last_price - actual)
                errors.append(error)

    return sum(errors) / len(errors) if errors else float('inf')


def evaluate_all_stocks(data_dir: Path = Path("trainingdata")) -> Dict:
    """Evaluate baseline on all stocks"""

    results = {}
    csv_files = sorted(data_dir.glob("*.csv"))

    # Filter out summary files
    csv_files = [f for f in csv_files if "summary" not in f.name.lower()]

    print(f"Evaluating {len(csv_files)} stock pairs...")
    print("="*100)
    print(f"{'Stock':<15} | {'Samples':<8} | {'Price Mean':<12} | {'Price Std':<12} | {'Naive MAE (h64)':<20} | {'MAE %':<10}")
    print("="*100)

    for csv_file in csv_files:
        try:
            prices = load_csv(csv_file)

            if not prices:
                continue

            stats = compute_stats(prices)

            # Compute naive baseline for different horizons
            horizons = [16, 32, 64, 128]
            naive_maes = {}

            for h in horizons:
                mae = compute_naive_mae(prices, h)
                naive_maes[f"h{h}"] = mae
                naive_maes[f"h{h}_pct"] = (mae / stats["mean"] * 100) if stats["mean"] > 0 else 0

            result = {
                "stock": csv_file.stem,
                **stats,
                **naive_maes,
            }

            results[csv_file.stem] = result

            print(f"{csv_file.stem:<15} | {stats['count']:<8} | "
                  f"${stats['mean']:<11.2f} | ${stats['std']:<11.2f} | "
                  f"${naive_maes['h64']:<19.3f} | {naive_maes['h64_pct']:<9.2f}%")

        except Exception as e:
            print(f"Error processing {csv_file}: {e}")

    print("="*100)
    return results


def save_results(results: Dict, output_file: Path = Path("tototraining/baseline_results.json")):
    """Save results to JSON"""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    return output_file


def analyze_results(results: Dict):
    """Analyze and print summary statistics"""

    if not results:
        print("No results to analyze")
        return

    # Compute aggregate statistics
    all_maes_h64 = [r["h64"] for r in results.values() if "h64" in r]
    all_maes_h64_pct = [r["h64_pct"] for r in results.values() if "h64_pct" in r]
    all_counts = [r["count"] for r in results.values() if "count" in r]
    all_means = [r["mean"] for r in results.values() if "mean" in r]

    print("\n" + "="*100)
    print("BASELINE ANALYSIS SUMMARY")
    print("="*100)

    print(f"\nDataset Statistics:")
    print(f"  Total stocks: {len(results)}")
    print(f"  Total samples: {sum(all_counts)}")
    print(f"  Avg samples per stock: {sum(all_counts) / len(all_counts):.0f}")
    print(f"  Avg price: ${sum(all_means) / len(all_means):.2f}")

    print(f"\nNaive Baseline Performance (h=64):")
    print(f"  Mean MAE: ${sum(all_maes_h64) / len(all_maes_h64):.3f}")
    sorted_maes = sorted(all_maes_h64)
    median_idx = len(sorted_maes) // 2
    print(f"  Median MAE: ${sorted_maes[median_idx]:.3f}")
    print(f"  Min MAE: ${min(all_maes_h64):.3f}")
    print(f"  Max MAE: ${max(all_maes_h64):.3f}")

    print(f"\n  Mean MAE%: {sum(all_maes_h64_pct) / len(all_maes_h64_pct):.2f}%")
    sorted_pcts = sorted(all_maes_h64_pct)
    median_pct_idx = len(sorted_pcts) // 2
    print(f"  Median MAE%: {sorted_pcts[median_pct_idx]:.2f}%")

    # Top/bottom performers
    sorted_by_pct = sorted(results.items(), key=lambda x: x[1].get("h64_pct", float('inf')))

    print(f"\nTop 5 Easiest to Predict (lowest MAE%):")
    for stock, data in sorted_by_pct[:5]:
        print(f"  {stock:<15} - {data.get('h64_pct', 0):.2f}% (${data.get('h64', 0):.3f})")

    print(f"\nTop 5 Hardest to Predict (highest MAE%):")
    for stock, data in reversed(sorted_by_pct[-5:]):
        print(f"  {stock:<15} - {data.get('h64_pct', 0):.2f}% (${data.get('h64', 0):.3f})")

    print("\n" + "="*100)

    # Our target: beat the naive baseline!
    target_mae_pct = sorted_pcts[median_pct_idx]
    print(f"\n🎯 TARGET TO BEAT: {target_mae_pct:.2f}% MAE (median naive baseline)")
    print(f"   This represents: ${sorted_maes[median_idx]:.3f} absolute MAE\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("trainingdata"))

    args = parser.parse_args()

    # Run evaluation
    results = evaluate_all_stocks(args.data_dir)

    # Save results
    save_results(results)

    # Analyze
    analyze_results(results)
