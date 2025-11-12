"""
Generate strategy PnL reports (summary + daily) from a collected dataset.

Outputs two CSV files:
1. {output_prefix}.csv             -> Daily PnL per (symbol, strategy, date)
2. {output_prefix}_summary.csv     -> Aggregated stats per (symbol, strategy)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def build_summary(perf_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate strategy performance into per-symbol summary metrics."""
    summary = (
        perf_df.groupby(["symbol", "strategy"])
        .agg(
            is_crypto=("is_crypto", "max"),
            num_windows=("window_num", "count"),
            total_pnl=("total_pnl", "sum"),
            avg_return=("total_return", "mean"),
            median_return=("total_return", "median"),
            avg_sharpe=("sharpe_ratio", "mean"),
            win_rate=("win_rate", "mean"),
            positive_window_pct=("total_pnl", lambda x: np.mean(x > 0.0)),
            avg_trades_per_window=("num_trades", "mean"),
            max_drawdown=("max_drawdown", "mean"),
        )
        .reset_index()
        .sort_values(["symbol", "strategy"])
    )
    summary["avg_return_pct"] = summary["avg_return"] * 100.0
    summary["win_rate_pct"] = summary["win_rate"] * 100.0
    summary["positive_window_pct"] = summary["positive_window_pct"] * 100.0
    return summary[
        [
            "symbol",
            "strategy",
            "is_crypto",
            "num_windows",
            "total_pnl",
            "avg_return_pct",
            "median_return",
            "avg_sharpe",
            "win_rate_pct",
            "positive_window_pct",
            "avg_trades_per_window",
            "max_drawdown",
        ]
    ]


def build_daily(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trades into daily PnL series for every strategy/symbol."""
    trades = trades_df.copy()
    trades["exit_timestamp"] = pd.to_datetime(trades["exit_timestamp"], utc=True)
    trades["date"] = trades["exit_timestamp"].dt.date

    daily = (
        trades.groupby(["symbol", "strategy", "date"])
        .agg(
            day_pnl=("pnl", "sum"),
            trades_executed=("pnl", "count"),
            avg_trade_pnl=("pnl", "mean"),
            max_trade_pnl=("pnl", "max"),
            min_trade_pnl=("pnl", "min"),
        )
        .reset_index()
        .sort_values(["symbol", "strategy", "date"])
    )

    daily["cumulative_pnl"] = daily.groupby(["symbol", "strategy"])["day_pnl"].cumsum()
    daily["positive_day"] = daily["day_pnl"] > 0.0
    return daily


def main():
    parser = argparse.ArgumentParser(description="Generate strategy PnL reports.")
    parser.add_argument(
        "dataset_base",
        help="Base path to dataset (without suffix). Example: strategytraining/datasets/strategy_pnl_15d_20250101_120000",
    )
    parser.add_argument(
        "--output-prefix",
        default="strategytraining/reports/maxdiff",
        help="Output prefix for CSV files (without extension).",
    )

    args = parser.parse_args()

    base_path = Path(args.dataset_base)
    perf_path = Path(f"{base_path}_strategy_performance.parquet")
    trades_path = Path(f"{base_path}_trades.parquet")
    metadata_path = Path(f"{base_path}_metadata.json")

    if not perf_path.exists() or not trades_path.exists():
        raise FileNotFoundError(f"Dataset files not found for base path: {base_path}")

    perf_df = pd.read_parquet(perf_path)
    trades_df = pd.read_parquet(trades_path)

    summary_df = build_summary(perf_df)
    daily_df = build_daily(trades_df)

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    daily_path = output_prefix.with_suffix(".csv")
    summary_path = output_prefix.parent / f"{output_prefix.stem}_summary.csv"

    daily_df.to_csv(daily_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

    print(f"✓ Daily PnL report written to {daily_path}")
    print(f"✓ Summary report written to {summary_path}")
    if metadata:
        print(
            f"Dataset: {metadata.get('dataset_name')} | symbols={metadata.get('num_symbols')} | strategies={metadata.get('num_strategies')}"
        )


if __name__ == "__main__":
    main()
