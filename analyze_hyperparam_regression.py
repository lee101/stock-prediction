#!/usr/bin/env python3
"""
Analyze which models got worse after hyperparameter tuning.
Compares baseline vs optimized performance.
"""

import json
from pathlib import Path
from typing import Dict, List
import sys


def load_results() -> Dict:
    """Load all results files."""
    results = {}

    # Load baseline
    baseline_path = Path("tototraining/baseline_results.json")
    if baseline_path.exists():
        with open(baseline_path) as f:
            results["baseline"] = json.load(f)

    # Load optimization results
    opt_path = Path("full_optimization_results.json")
    if opt_path.exists():
        with open(opt_path) as f:
            results["optimization"] = json.load(f)

    # Load comparison results
    comp_path = Path("comparison_results.json")
    if comp_path.exists():
        with open(comp_path) as f:
            results["comparison"] = json.load(f)

    return results


def analyze_regressions(results: Dict) -> List[Dict]:
    """Find stocks where performance got worse."""
    regressions = []

    if "baseline" not in results or "optimization" not in results:
        print("Missing baseline or optimization results")
        return regressions

    baseline = results["baseline"]
    opt_results = results["optimization"].get("results", [])

    for opt_result in opt_results:
        symbol = opt_result.get("symbol")
        if not symbol or opt_result.get("status") != "success":
            continue

        # Get baseline MAE
        if symbol not in baseline:
            continue

        baseline_mae = baseline[symbol].get("h64_pct", 0)
        opt_mae = opt_result.get("best_mae", 0) * 100  # Convert to percentage

        # Check if it got worse (allowing 1% tolerance)
        if opt_mae > baseline_mae + 1.0:
            regression = {
                "symbol": symbol,
                "baseline_mae_pct": baseline_mae,
                "optimized_mae_pct": opt_mae,
                "regression_pct": opt_mae - baseline_mae,
                "best_model": opt_result.get("best_model", "unknown"),
            }
            regressions.append(regression)

    return regressions


def analyze_failures(results: Dict) -> List[Dict]:
    """Find stocks that failed during optimization."""
    failures = []

    if "optimization" not in results:
        return failures

    opt_results = results["optimization"].get("results", [])

    for opt_result in opt_results:
        if opt_result.get("status") == "failed":
            failures.append({
                "symbol": opt_result.get("symbol"),
                "error": opt_result.get("error", "Unknown error")[:200],
            })

    return failures


def main():
    print("=" * 60)
    print("Hyperparameter Regression Analysis")
    print("=" * 60)
    print()

    results = load_results()

    # Analyze regressions
    regressions = analyze_regressions(results)

    if regressions:
        print(f"Found {len(regressions)} stocks with worse performance:\n")
        regressions.sort(key=lambda x: x["regression_pct"], reverse=True)

        for reg in regressions:
            print(f"  {reg['symbol']:8s} - Baseline: {reg['baseline_mae_pct']:6.2f}% "
                  f"→ Optimized: {reg['optimized_mae_pct']:6.2f}% "
                  f"(+{reg['regression_pct']:5.2f}%) [{reg['best_model']}]")
    else:
        print("No regressions found! All models improved or stayed the same.")

    print()

    # Analyze failures
    failures = analyze_failures(results)

    if failures:
        print(f"Found {len(failures)} failed optimizations:\n")
        for fail in failures:
            print(f"  {fail['symbol']:8s} - {fail['error']}")
    else:
        print("No failures found!")

    print()
    print("=" * 60)
    print("Recommendations")
    print("=" * 60)
    print()

    if regressions:
        print("For stocks with regressions:")
        print("1. Retrain with longer epochs and adjusted hyperparameters")
        print("2. Use extended training with --extended-epochs flag")
        print("3. Validate on 7-day holdout to tune inference params")
        print()
        print("Example:")
        print(f"  uv run python retrain_all_stocks.py --stocks {' '.join([r['symbol'] for r in regressions[:3]])}")
        print()

    if failures:
        print("For failed stocks:")
        print("1. Check error messages above")
        print("2. Ensure model cache directories exist")
        print("3. Verify sufficient GPU memory")
        print()

    # Save regression report
    if regressions or failures:
        report = {
            "regressions": regressions,
            "failures": failures,
            "summary": {
                "total_regressions": len(regressions),
                "total_failures": len(failures),
                "avg_regression_pct": sum(r["regression_pct"] for r in regressions) / len(regressions) if regressions else 0,
            }
        }

        report_path = Path("regression_analysis.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"Detailed report saved to: {report_path}")
        print()


if __name__ == "__main__":
    main()
