#!/usr/bin/env python3
"""
Analyze and compare hyperparameter optimization results.

This script:
1. Loads results from hyperparams/ and hyperparams_extended/ directories
2. Compares Kronos vs Toto performance
3. Identifies best hyperparameters per symbol
4. Generates summary statistics and visualizations
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def load_results(results_dir: Path, model: str) -> Dict[str, dict]:
    """Load all results for a model from a results directory."""
    model_dir = results_dir / model
    if not model_dir.exists():
        return {}

    results = {}
    for json_file in model_dir.glob("*.json"):
        symbol = json_file.stem
        with json_file.open("r") as f:
            data = json.load(f)
            results[symbol] = data
    return results


def analyze_model_results(results: Dict[str, dict]) -> pd.DataFrame:
    """Analyze results for a single model."""
    rows = []
    for symbol, data in results.items():
        config = data.get("config", {})
        validation = data.get("validation", {})
        test = data.get("test", {})

        row = {
            "symbol": symbol,
            "val_price_mae": validation.get("price_mae"),
            "val_return_mae": validation.get("pct_return_mae"),
            "test_price_mae": test.get("price_mae"),
            "test_return_mae": test.get("pct_return_mae"),
            "val_latency": validation.get("latency_s"),
            "config_name": config.get("name"),
        }

        # Add model-specific config details
        if "temperature" in config:  # Kronos
            row.update({
                "temperature": config.get("temperature"),
                "top_p": config.get("top_p"),
                "top_k": config.get("top_k"),
                "sample_count": config.get("sample_count"),
                "max_context": config.get("max_context"),
                "clip": config.get("clip"),
            })
        elif "num_samples" in config:  # Toto
            row.update({
                "num_samples": config.get("num_samples"),
                "aggregate": config.get("aggregate"),
                "samples_per_batch": config.get("samples_per_batch"),
            })

        rows.append(row)

    return pd.DataFrame(rows)


def compare_models(
    kronos_df: pd.DataFrame,
    toto_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compare Kronos vs Toto results per symbol."""
    if kronos_df.empty or toto_df.empty:
        return pd.DataFrame()

    kronos_subset = kronos_df[["symbol", "val_price_mae", "test_price_mae"]].copy()
    kronos_subset.columns = ["symbol", "kronos_val_mae", "kronos_test_mae"]

    toto_subset = toto_df[["symbol", "val_price_mae", "test_price_mae"]].copy()
    toto_subset.columns = ["symbol", "toto_val_mae", "toto_test_mae"]

    comparison = pd.merge(kronos_subset, toto_subset, on="symbol", how="outer")
    comparison["best_model"] = comparison.apply(
        lambda row: (
            "Kronos" if row["kronos_test_mae"] < row["toto_test_mae"]
            else "Toto" if row["toto_test_mae"] < row["kronos_test_mae"]
            else "Tie"
        ),
        axis=1
    )
    comparison["mae_improvement"] = comparison.apply(
        lambda row: abs(row["kronos_test_mae"] - row["toto_test_mae"]),
        axis=1
    )

    return comparison


def print_summary(
    model: str,
    df: pd.DataFrame,
    results_dir: str,
) -> None:
    """Print summary statistics for a model."""
    if df.empty:
        print(f"\n{'='*60}")
        print(f"{model.upper()} - No results found")
        print(f"{'='*60}")
        return

    print(f"\n{'='*60}")
    print(f"{model.upper()} Results Summary ({results_dir})")
    print(f"{'='*60}")
    print(f"Total symbols tested: {len(df)}")
    print(f"\nValidation MAE Statistics:")
    print(f"  Mean:   {df['val_price_mae'].mean():.4f}")
    print(f"  Median: {df['val_price_mae'].median():.4f}")
    print(f"  Std:    {df['val_price_mae'].std():.4f}")
    print(f"  Min:    {df['val_price_mae'].min():.4f} ({df.loc[df['val_price_mae'].idxmin(), 'symbol']})")
    print(f"  Max:    {df['val_price_mae'].max():.4f} ({df.loc[df['val_price_mae'].idxmax(), 'symbol']})")

    print(f"\nTest MAE Statistics:")
    print(f"  Mean:   {df['test_price_mae'].mean():.4f}")
    print(f"  Median: {df['test_price_mae'].median():.4f}")
    print(f"  Std:    {df['test_price_mae'].std():.4f}")
    print(f"  Min:    {df['test_price_mae'].min():.4f} ({df.loc[df['test_price_mae'].idxmin(), 'symbol']})")
    print(f"  Max:    {df['test_price_mae'].max():.4f} ({df.loc[df['test_price_mae'].idxmax(), 'symbol']})")

    print(f"\nTop 5 Best Performing Symbols (by test MAE):")
    top5 = df.nsmallest(5, "test_price_mae")[["symbol", "test_price_mae", "config_name"]]
    for idx, row in top5.iterrows():
        print(f"  {row['symbol']:10s} MAE: {row['test_price_mae']:.4f}  ({row['config_name']})")

    print(f"\nTop 5 Worst Performing Symbols (by test MAE):")
    bottom5 = df.nlargest(5, "test_price_mae")[["symbol", "test_price_mae", "config_name"]]
    for idx, row in bottom5.iterrows():
        print(f"  {row['symbol']:10s} MAE: {row['test_price_mae']:.4f}  ({row['config_name']})")


def print_comparison_summary(comparison_df: pd.DataFrame) -> None:
    """Print model comparison summary."""
    if comparison_df.empty:
        print("\nNo comparison data available")
        return

    print(f"\n{'='*60}")
    print("Kronos vs Toto Comparison")
    print(f"{'='*60}")

    kronos_wins = (comparison_df["best_model"] == "Kronos").sum()
    toto_wins = (comparison_df["best_model"] == "Toto").sum()
    ties = (comparison_df["best_model"] == "Tie").sum()

    print(f"\nOverall Performance:")
    print(f"  Kronos wins: {kronos_wins}")
    print(f"  Toto wins:   {toto_wins}")
    print(f"  Ties:        {ties}")

    print(f"\nAverage MAE Improvement by Winner:")
    if kronos_wins > 0:
        kronos_improvements = comparison_df[comparison_df["best_model"] == "Kronos"]["mae_improvement"]
        print(f"  Kronos avg improvement: {kronos_improvements.mean():.4f}")
    if toto_wins > 0:
        toto_improvements = comparison_df[comparison_df["best_model"] == "Toto"]["mae_improvement"]
        print(f"  Toto avg improvement:   {toto_improvements.mean():.4f}")

    print(f"\nBiggest Kronos Wins:")
    kronos_best = comparison_df[comparison_df["best_model"] == "Kronos"].nlargest(3, "mae_improvement")
    for idx, row in kronos_best.iterrows():
        print(f"  {row['symbol']:10s} Kronos: {row['kronos_test_mae']:.4f} vs Toto: {row['toto_test_mae']:.4f} (Δ {row['mae_improvement']:.4f})")

    print(f"\nBiggest Toto Wins:")
    toto_best = comparison_df[comparison_df["best_model"] == "Toto"].nlargest(3, "mae_improvement")
    for idx, row in toto_best.iterrows():
        print(f"  {row['symbol']:10s} Toto: {row['toto_test_mae']:.4f} vs Kronos: {row['kronos_test_mae']:.4f} (Δ {row['mae_improvement']:.4f})")


def analyze_hyperparameter_trends(df: pd.DataFrame, model: str) -> None:
    """Analyze which hyperparameters correlate with better performance."""
    if df.empty:
        return

    print(f"\n{'='*60}")
    print(f"{model.upper()} - Hyperparameter Analysis")
    print(f"{'='*60}")

    if model.lower() == "kronos":
        if "temperature" in df.columns:
            print(f"\nTemperature vs MAE:")
            temp_groups = df.groupby(pd.cut(df["temperature"], bins=5))["test_price_mae"].agg(["mean", "count"])
            print(temp_groups)

        if "top_p" in df.columns:
            print(f"\nTop-P vs MAE:")
            top_p_groups = df.groupby(pd.cut(df["top_p"], bins=5))["test_price_mae"].agg(["mean", "count"])
            print(top_p_groups)

        if "sample_count" in df.columns:
            print(f"\nSample Count vs MAE:")
            sample_groups = df.groupby(pd.cut(df["sample_count"], bins=5))["test_price_mae"].agg(["mean", "count"])
            print(sample_groups)

    elif model.lower() == "toto":
        if "num_samples" in df.columns:
            print(f"\nNumber of Samples vs MAE:")
            sample_groups = df.groupby(pd.cut(df["num_samples"], bins=5))["test_price_mae"].agg(["mean", "count"])
            print(sample_groups)

        if "aggregate" in df.columns:
            print(f"\nAggregation Strategy vs MAE:")
            # Group similar aggregation strategies
            df["agg_type"] = df["aggregate"].apply(lambda x: x.split("_")[0] if isinstance(x, str) else "unknown")
            agg_groups = df.groupby("agg_type")["test_price_mae"].agg(["mean", "count"]).sort_values("mean")
            print(agg_groups)


def main():
    parser = argparse.ArgumentParser(description="Analyze hyperparameter optimization results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="hyperparams",
        help="Results directory to analyze (default: hyperparams)",
    )
    parser.add_argument(
        "--compare-dirs",
        nargs=2,
        metavar=("DIR1", "DIR2"),
        help="Compare results from two directories",
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        help="Export results to CSV file",
    )
    args = parser.parse_args()

    if args.compare_dirs:
        # Compare two result directories
        dir1, dir2 = [Path(d) for d in args.compare_dirs]

        print(f"\nComparing results from:")
        print(f"  Directory 1: {dir1}")
        print(f"  Directory 2: {dir2}")

        # Load and analyze both directories
        for directory in [dir1, dir2]:
            if not directory.exists():
                print(f"\nWarning: {directory} does not exist")
                continue

            kronos_results = load_results(directory, "kronos")
            toto_results = load_results(directory, "toto")

            kronos_df = analyze_model_results(kronos_results)
            toto_df = analyze_model_results(toto_results)

            print_summary("Kronos", kronos_df, str(directory))
            print_summary("Toto", toto_df, str(directory))

            comparison = compare_models(kronos_df, toto_df)
            print_comparison_summary(comparison)

    else:
        # Analyze single directory
        results_dir = Path(args.results_dir)
        if not results_dir.exists():
            print(f"Error: {results_dir} does not exist")
            return

        print(f"\nAnalyzing results from: {results_dir}")

        # Load results
        kronos_results = load_results(results_dir, "kronos")
        toto_results = load_results(results_dir, "toto")

        kronos_df = analyze_model_results(kronos_results)
        toto_df = analyze_model_results(toto_results)

        # Print summaries
        print_summary("Kronos", kronos_df, str(results_dir))
        print_summary("Toto", toto_df, str(results_dir))

        # Compare models
        comparison = compare_models(kronos_df, toto_df)
        print_comparison_summary(comparison)

        # Analyze hyperparameter trends
        analyze_hyperparameter_trends(kronos_df, "Kronos")
        analyze_hyperparameter_trends(toto_df, "Toto")

        # Export if requested
        if args.export_csv:
            combined_df = pd.concat([
                kronos_df.assign(model="Kronos"),
                toto_df.assign(model="Toto")
            ])
            combined_df.to_csv(args.export_csv, index=False)
            print(f"\nResults exported to: {args.export_csv}")


if __name__ == "__main__":
    main()
