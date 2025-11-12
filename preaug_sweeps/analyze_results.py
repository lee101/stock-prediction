#!/usr/bin/env python3
"""
Analyze and visualize pre-augmentation sweep results.

Provides detailed analysis of which strategies work best.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd


def _resolve_selection_value(config: Dict) -> Tuple[str, Optional[float]]:
    """Return the metric name/value pair used when selecting the best strategy."""

    metric = config.get("selection_metric")
    if not metric:
        metric = "mae_percent" if config.get("mae_percent") is not None else "mae"

    value = config.get("selection_value")
    if value is None:
        value = config.get(metric)
        if value is None and metric != "mae":
            value = config.get("mae")

    return metric, value


def load_best_configs(best_dir: Path = Path("preaugstrategies/best")) -> Dict:
    """Load all best configurations."""
    configs = {}
    for json_file in best_dir.glob("*.json"):
        symbol = json_file.stem
        with open(json_file) as f:
            configs[symbol] = json.load(f)
    return configs


def load_latest_sweep_results(reports_dir: Path = Path("preaug_sweeps/reports")) -> Dict:
    """Load the most recent sweep results."""
    json_files = sorted(reports_dir.glob("sweep_results_*.json"), reverse=True)
    if not json_files:
        return {}

    with open(json_files[0]) as f:
        return json.load(f)


def analyze_best_strategies(configs: Dict) -> None:
    """Analyze best strategies across symbols."""
    print("\n" + "=" * 80)
    print("BEST STRATEGIES PER SYMBOL")
    print("=" * 80)

    rows = []
    for symbol, config in sorted(configs.items()):
        strategy = config["best_strategy"]
        mae = config["mae"]
        mae_percent = config.get("mae_percent")
        rmse = config["rmse"]
        mape = config["mape"]
        selection_metric, selection_value = _resolve_selection_value(config)

        print(f"\n{symbol}:")
        print(f"  Strategy: {strategy}")
        print(f"  MAE:      {mae:.6f}")
        if mae_percent is not None:
            print(f"  MAE%:     {mae_percent:.4f}%")
        print(f"  RMSE:     {rmse:.6f}")
        print(f"  MAPE:     {mape:.4f}%")
        if selection_value is not None:
            print(f"  Selection ({selection_metric}): {selection_value:.6f}")

        # Show improvement over baseline
        improvement = None
        if "baseline" in config.get("comparison", {}):
            baseline = config["comparison"]["baseline"]
            baseline_value = baseline.get(selection_metric) or baseline.get("mae")
            if baseline_value and selection_value and baseline_value != 0:
                improvement = ((baseline_value - selection_value) / baseline_value) * 100
                print(f"  Improvement ({selection_metric}): {improvement:+.2f}% vs baseline")
            else:
                print("  Improvement: n/a (missing baseline metric)")
        else:
            print("  Improvement: n/a (baseline missing)")

        rows.append({
            "Symbol": symbol,
            "Strategy": strategy,
            "MAE": mae,
            "MAE_percent": mae_percent,
            "RMSE": rmse,
            "MAPE": mape,
            "SelectionMetric": selection_metric,
            "SelectionValue": selection_value,
            "Improvement_pct": improvement,
        })

    # Create summary table
    df = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(df.to_string(index=False))

    # Strategy frequency
    print("\n" + "=" * 80)
    print("STRATEGY POPULARITY")
    print("=" * 80)
    strategy_counts = df["Strategy"].value_counts()
    for strategy, count in strategy_counts.items():
        print(f"  {strategy:25s}: {count} symbols")


def analyze_strategy_performance(results: Dict) -> None:
    """Analyze each strategy's performance across symbols."""
    print("\n" + "=" * 80)
    print("STRATEGY PERFORMANCE ACROSS SYMBOLS")
    print("=" * 80)

    # Collect all strategy results
    strategy_results = {}
    for symbol, symbol_results in results.items():
        for strategy, result in symbol_results.items():
            if result.get("status") != "success":
                continue

            if strategy not in strategy_results:
                strategy_results[strategy] = []

            strategy_results[strategy].append({
                "symbol": symbol,
                "mae": result["mae"],
                "mae_percent": result.get("mae_percent"),
                "rmse": result["rmse"],
                "mape": result["mape"],
            })

    # Compute averages
    rows = []
    for strategy, results_list in sorted(strategy_results.items()):
        avg_mae = sum(r["mae"] for r in results_list) / len(results_list)
        avg_rmse = sum(r["rmse"] for r in results_list) / len(results_list)
        avg_mape = sum(r["mape"] for r in results_list) / len(results_list)
        mae_percent_values = [r["mae_percent"] for r in results_list if r.get("mae_percent") is not None]
        avg_mae_percent = (
            sum(mae_percent_values) / len(mae_percent_values)
            if mae_percent_values else None
        )

        rows.append({
            "Strategy": strategy,
            "Avg_MAE": avg_mae,
            "Avg_MAE_percent": avg_mae_percent,
            "Avg_RMSE": avg_rmse,
            "Avg_MAPE": avg_mape,
            "Symbols": len(results_list),
        })

        print(f"\n{strategy}:")
        print(f"  Avg MAE:  {avg_mae:.6f}")
        if avg_mae_percent is not None:
            print(f"  Avg MAE%: {avg_mae_percent:.4f}%")
        print(f"  Avg RMSE: {avg_rmse:.6f}")
        print(f"  Avg MAPE: {avg_mape:.4f}%")
        print(f"  Tested on {len(results_list)} symbols")

    # Rank by MAE
    if rows:
        df = pd.DataFrame(rows).sort_values("Avg_MAE")
        print("\n" + "=" * 80)
        print("STRATEGIES RANKED BY AVERAGE MAE")
        print("=" * 80)
        print(df.to_string(index=False))
    else:
        print("\nNo successful strategy results to summarize.")


def generate_comparison_matrix(results: Dict) -> None:
    """Generate a comparison matrix of strategies vs symbols."""
    print("\n" + "=" * 80)
    print("MAE COMPARISON MATRIX")
    print("=" * 80)

    # Build matrix
    data_mae = []
    data_mae_pct = []
    for symbol, symbol_results in sorted(results.items()):
        row_mae = {"Symbol": symbol}
        row_pct = {"Symbol": symbol}
        for strategy, result in symbol_results.items():
            if result.get("status") == "success":
                row_mae[strategy] = result["mae"]
                if result.get("mae_percent") is not None:
                    row_pct[strategy] = result["mae_percent"]
        data_mae.append(row_mae)
        data_mae_pct.append(row_pct)

    if not data_mae:
        print("No successful strategy data to compare.")
        return

    df_mae = pd.DataFrame(data_mae)
    if "Symbol" in df_mae.columns:
        df_mae = df_mae.set_index("Symbol")

    if df_mae.shape[1] == 0:
        print("No successful strategy data to compare.")
        return

    print(df_mae.to_string())

    df_pct = pd.DataFrame(data_mae_pct)
    if "Symbol" in df_pct.columns:
        df_pct = df_pct.set_index("Symbol")

    if not df_pct.empty and df_pct.select_dtypes(include=['number']).notna().any().any():
        print("\n" + "=" * 80)
        print("MAE% COMPARISON MATRIX")
        print("=" * 80)
        print(df_pct.to_string())

    metric_df = df_pct if not df_pct.empty and df_pct.notna().any().any() else df_mae
    metric_label = "MAE%" if metric_df is df_pct else "MAE"

    # Find best for each symbol
    print("\n" + "=" * 80)
    print(f"BEST STRATEGY PER SYMBOL ({metric_label})")
    print("=" * 80)
    for symbol in metric_df.index:
        row = metric_df.loc[symbol].dropna()
        if row.empty:
            print(f"  {symbol:15s}: n/a")
            continue
        best_strategy = row.astype(float).idxmin()
        best_value = row.min()
        print(f"  {symbol:15s}: {best_strategy:25s} ({metric_label}: {best_value:.6f})")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze pre-augmentation sweep results")
    parser.add_argument(
        "--best-dir",
        type=Path,
        default=Path("preaugstrategies/best"),
        help="Directory with best configs"
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("preaug_sweeps/reports"),
        help="Directory with sweep reports"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("PRE-AUGMENTATION SWEEP ANALYSIS")
    print("=" * 80)

    # Load data
    print("\nLoading results...")
    configs = load_best_configs(args.best_dir)
    results = load_latest_sweep_results(args.reports_dir)

    if not configs and not results:
        print("No results found. Run sweep first!")
        return 1

    # Analysis
    if configs:
        analyze_best_strategies(configs)

    if results:
        analyze_strategy_performance(results)
        generate_comparison_matrix(results)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
