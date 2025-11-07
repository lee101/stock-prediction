#!/usr/bin/env python3
"""
Analyze and Visualize Optimizer Benchmark Results

Provides comprehensive analysis of optimizer benchmark data including:
- Performance comparisons
- Statistical significance tests
- Convergence analysis
- Optimizer rankings

Usage:
    python analyze_optimizer_benchmarks.py benchmark_results/real_pnl_benchmark_*.parquet
    python analyze_optimizer_benchmarks.py --plot benchmark_results/*.parquet
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
import numpy as np


def load_benchmark_results(file_paths: List[str]) -> pd.DataFrame:
    """Load one or more benchmark result files"""
    dfs = []
    for path in file_paths:
        if Path(path).exists():
            df = pd.read_parquet(path)
            dfs.append(df)
            print(f"Loaded {len(df)} results from {path}")
        else:
            print(f"Warning: File not found: {path}")

    if not dfs:
        raise FileNotFoundError("No benchmark files found!")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal results: {len(combined)}")
    return combined


def compute_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute comprehensive summary statistics by optimizer"""
    summary = df.groupby('optimizer_name').agg({
        'best_value': ['count', 'mean', 'std', 'min', 'median', 'max'],
        'time_seconds': ['mean', 'std', 'min', 'max'],
        'num_evaluations': ['mean', 'std', 'min', 'max'],
        'converged': ['sum', 'mean'],
    }).round(6)

    summary.columns = ['_'.join(col) for col in summary.columns]

    # Add coefficient of variation for best_value
    summary['best_value_cv'] = (
        summary['best_value_std'] / summary['best_value_mean'].abs()
    ).round(4)

    # Sort by mean best value
    summary = summary.sort_values('best_value_mean')

    return summary


def compute_win_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Compute win rate for each optimizer (best P&L per problem)"""
    # Group by problem, find best optimizer
    best_by_problem = (
        df.groupby('strategy_name')
        .apply(lambda g: g.loc[g['best_value'].idxmin(), 'optimizer_name'])
        .reset_index()
    )
    best_by_problem.columns = ['strategy_name', 'winner']

    # Count wins
    win_counts = best_by_problem['winner'].value_counts()

    # Create win rate dataframe
    total_problems = len(best_by_problem)
    win_rate_df = pd.DataFrame({
        'optimizer': win_counts.index,
        'wins': win_counts.values,
        'win_rate': (win_counts.values / total_problems * 100).round(2),
    })

    return win_rate_df.sort_values('wins', ascending=False)


def compute_relative_performance(df: pd.DataFrame, baseline: str = 'DIRECT') -> pd.DataFrame:
    """Compute relative performance vs baseline optimizer"""
    if baseline not in df['optimizer_name'].unique():
        print(f"Warning: Baseline {baseline} not found, using first optimizer")
        baseline = df['optimizer_name'].iloc[0]

    # Get baseline results
    baseline_df = df[df['optimizer_name'] == baseline][['strategy_name', 'best_value']].rename(
        columns={'best_value': 'baseline_value'}
    )

    # Merge with all results
    merged = df.merge(baseline_df, on='strategy_name', how='left')

    # Compute relative improvement
    merged['relative_improvement'] = (
        (merged['baseline_value'] - merged['best_value']) / merged['baseline_value'].abs() * 100
    )

    # Summarize by optimizer
    relative_perf = merged.groupby('optimizer_name').agg({
        'relative_improvement': ['mean', 'median', 'std', 'min', 'max']
    }).round(2)

    relative_perf.columns = ['_'.join(col) for col in relative_perf.columns]
    relative_perf = relative_perf.sort_values('relative_improvement_mean', ascending=False)

    return relative_perf


def compute_efficiency_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute efficiency metrics (value per evaluation, value per second)"""
    efficiency = df.groupby('optimizer_name').apply(
        lambda g: pd.Series({
            'value_per_eval': (g['best_value'].mean() / g['num_evaluations'].mean()),
            'value_per_second': (g['best_value'].mean() / g['time_seconds'].mean()),
            'evals_per_second': (g['num_evaluations'].mean() / g['time_seconds'].mean()),
        })
    ).round(6)

    return efficiency.sort_values('value_per_second')


def analyze_convergence_speed(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze convergence speed (time to reach good solutions)"""
    # Group by optimizer and compute quartiles of time
    convergence = df.groupby('optimizer_name').agg({
        'time_seconds': ['min', lambda x: x.quantile(0.25), 'median', lambda x: x.quantile(0.75), 'max'],
        'num_evaluations': ['min', lambda x: x.quantile(0.25), 'median', lambda x: x.quantile(0.75), 'max'],
    }).round(3)

    convergence.columns = [
        f"{col[0]}_{col[1]}" if isinstance(col[1], str) else f"{col[0]}_q{int(col[1]*100)}"
        for col in convergence.columns
    ]

    return convergence


def compute_statistical_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise statistical significance tests"""
    try:
        from scipy.stats import mannwhitneyu
    except ImportError:
        print("scipy not available, skipping statistical tests")
        return pd.DataFrame()

    optimizers = df['optimizer_name'].unique()
    results = []

    for i, opt1 in enumerate(optimizers):
        for opt2 in optimizers[i+1:]:
            values1 = df[df['optimizer_name'] == opt1]['best_value'].values
            values2 = df[df['optimizer_name'] == opt2]['best_value'].values

            if len(values1) > 0 and len(values2) > 0:
                statistic, pvalue = mannwhitneyu(values1, values2, alternative='two-sided')

                results.append({
                    'optimizer_1': opt1,
                    'optimizer_2': opt2,
                    'mean_1': values1.mean(),
                    'mean_2': values2.mean(),
                    'difference': values1.mean() - values2.mean(),
                    'statistic': statistic,
                    'p_value': pvalue,
                    'significant': pvalue < 0.05,
                })

    if not results:
        return pd.DataFrame()

    sig_df = pd.DataFrame(results).sort_values('p_value')
    return sig_df


def create_optimizer_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Create overall optimizer ranking considering multiple metrics"""
    summary = compute_summary_statistics(df)
    win_rates = compute_win_rates(df)
    efficiency = compute_efficiency_metrics(df)

    # Merge all metrics
    ranking = summary.copy()
    ranking = ranking.merge(
        win_rates.set_index('optimizer'),
        left_index=True,
        right_index=True,
        how='left'
    )
    ranking = ranking.merge(
        efficiency,
        left_index=True,
        right_index=True,
        how='left'
    )

    # Fill NaN win rates with 0
    ranking['wins'] = ranking['wins'].fillna(0)
    ranking['win_rate'] = ranking['win_rate'].fillna(0)

    # Compute composite score (lower is better for P&L, higher is better for wins)
    # Normalize metrics and compute weighted average
    ranking['normalized_pnl'] = (
        (ranking['best_value_mean'] - ranking['best_value_mean'].min()) /
        (ranking['best_value_mean'].max() - ranking['best_value_mean'].min())
    )
    ranking['normalized_win_rate'] = ranking['win_rate'] / 100
    ranking['normalized_speed'] = (
        1 - (ranking['time_seconds_mean'] - ranking['time_seconds_mean'].min()) /
        (ranking['time_seconds_mean'].max() - ranking['time_seconds_mean'].min())
    )

    # Composite score: 50% P&L, 30% win rate, 20% speed
    ranking['composite_score'] = (
        -0.5 * ranking['normalized_pnl'] +  # Negative because lower P&L is better
        0.3 * ranking['normalized_win_rate'] +
        0.2 * ranking['normalized_speed']
    ).round(4)

    ranking = ranking.sort_values('composite_score', ascending=False)

    return ranking


def print_detailed_analysis(df: pd.DataFrame):
    """Print comprehensive analysis report"""
    print("\n" + "=" * 80)
    print("OPTIMIZER BENCHMARK ANALYSIS")
    print("=" * 80)

    # Basic info
    print(f"\nDataset Info:")
    print(f"  Total benchmarks: {len(df)}")
    print(f"  Optimizers tested: {df['optimizer_name'].nunique()}")
    print(f"  Unique problems: {df['strategy_name'].nunique()}")
    print(f"  Trials per problem: {len(df) // df['strategy_name'].nunique():.1f}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    summary = compute_summary_statistics(df)
    print(summary.to_string())

    # Win rates
    print("\n" + "=" * 80)
    print("WIN RATES (Best P&L per problem)")
    print("=" * 80)
    win_rates = compute_win_rates(df)
    print(win_rates.to_string(index=False))

    # Relative performance
    print("\n" + "=" * 80)
    print("RELATIVE PERFORMANCE (vs DIRECT baseline)")
    print("=" * 80)
    relative = compute_relative_performance(df, baseline='DIRECT')
    print(relative.to_string())

    # Efficiency metrics
    print("\n" + "=" * 80)
    print("EFFICIENCY METRICS")
    print("=" * 80)
    efficiency = compute_efficiency_metrics(df)
    print(efficiency.to_string())

    # Convergence speed
    print("\n" + "=" * 80)
    print("CONVERGENCE SPEED")
    print("=" * 80)
    convergence = analyze_convergence_speed(df)
    print(convergence.to_string())

    # Overall ranking
    print("\n" + "=" * 80)
    print("OVERALL RANKING")
    print("=" * 80)
    ranking = create_optimizer_ranking(df)
    print(ranking[['composite_score', 'wins', 'win_rate', 'best_value_mean', 'time_seconds_mean']].to_string())

    # Statistical significance
    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE (Mann-Whitney U test)")
    print("=" * 80)
    sig_tests = compute_statistical_tests(df)
    if not sig_tests.empty:
        print(sig_tests.head(10).to_string(index=False))
    else:
        print("Statistical tests not available (scipy not installed)")

    # Top recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    best_overall = ranking.index[0]
    print(f"🏆 Best Overall: {best_overall}")
    print(f"   Composite Score: {ranking.loc[best_overall, 'composite_score']:.4f}")
    print(f"   Win Rate: {ranking.loc[best_overall, 'win_rate']:.1f}%")
    print(f"   Mean P&L: {ranking.loc[best_overall, 'best_value_mean']:.6f}")
    print(f"   Mean Time: {ranking.loc[best_overall, 'time_seconds_mean']:.2f}s")

    fastest = summary['time_seconds_mean'].idxmin()
    print(f"\n⚡ Fastest: {fastest}")
    print(f"   Mean Time: {summary.loc[fastest, 'time_seconds_mean']:.2f}s")

    best_pnl = summary['best_value_mean'].idxmin()
    print(f"\n💰 Best P&L: {best_pnl}")
    print(f"   Mean P&L: {summary.loc[best_pnl, 'best_value_mean']:.6f}")

    most_wins = win_rates.iloc[0]['optimizer']
    print(f"\n🎯 Most Wins: {most_wins}")
    print(f"   Win Rate: {win_rates.iloc[0]['win_rate']:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Analyze optimizer benchmarks")
    parser.add_argument(
        'files',
        nargs='+',
        help='Benchmark result files (parquet format)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for analysis results'
    )

    args = parser.parse_args()

    # Load results
    df = load_benchmark_results(args.files)

    # Print analysis
    print_detailed_analysis(df)

    # Save analysis if output dir specified
    if args.output_dir:
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        summary = compute_summary_statistics(df)
        win_rates = compute_win_rates(df)
        ranking = create_optimizer_ranking(df)
        efficiency = compute_efficiency_metrics(df)

        summary.to_csv(output_path / "summary_statistics.csv")
        win_rates.to_csv(output_path / "win_rates.csv", index=False)
        ranking.to_csv(output_path / "overall_ranking.csv")
        efficiency.to_csv(output_path / "efficiency_metrics.csv")

        print(f"\n\nAnalysis saved to {output_path}/")


if __name__ == "__main__":
    main()
