#!/usr/bin/env python3
"""Compare meta-selector vs unified model vs individual stocks vs buy-and-hold.

Usage:
    python -u -m fastalgorithms.per_stock.compare_strategies --holdout-days 90
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from loguru import logger

from fastalgorithms.per_stock.meta_selector import (
    MetaSelectorConfig,
    compute_sortino,
    run_meta_simulation,
    run_meta_lookback_sweep,
)
from fastalgorithms.per_stock.eval_per_stock import (
    evaluate_checkpoint,
    select_best_per_stock,
)


def compute_buy_and_hold(
    symbol: str,
    holdout_days: int = 90,
    data_root: Path = Path("trainingdatahourly/stocks"),
    direction: str = "long",
) -> dict:
    """Compute buy-and-hold return for a single stock."""
    csv_path = data_root / f"{symbol}.csv"
    if not csv_path.exists():
        return {"symbol": symbol, "total_return": 0.0, "sortino": 0.0, "max_drawdown": 0.0}

    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    if holdout_days > 0:
        cutoff = df["timestamp"].max() - pd.Timedelta(days=holdout_days)
        df = df[df["timestamp"] >= cutoff].reset_index(drop=True)

    if len(df) < 2:
        return {"symbol": symbol, "total_return": 0.0, "sortino": 0.0, "max_drawdown": 0.0}

    close = df["close"].values
    if direction == "short":
        # Short-and-hold: profit from price decline
        equity = 10000.0 * (2.0 - close / close[0])  # short with initial capital
    else:
        equity = 10000.0 * (close / close[0])

    total_return = (equity[-1] - equity[0]) / equity[0]
    sortino_val = compute_sortino(equity)
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / (peak + 1e-10)
    max_dd = float(np.max(dd))

    return {
        "symbol": symbol,
        "total_return": total_return,
        "sortino": sortino_val,
        "max_drawdown": max_dd,
        "final_equity": equity[-1],
    }


def run_comparison(
    best_per_stock: Dict[str, dict],
    holdout_days: int = 90,
    lookback_hours_list: list[int] = None,
    data_root: Path = Path("trainingdatahourly/stocks"),
) -> pd.DataFrame:
    """Run full strategy comparison.

    Args:
        best_per_stock: Output from select_best_per_stock()
        holdout_days: OOS evaluation period
        lookback_hours_list: Lookback windows to sweep for meta-selector

    Returns:
        Comparison DataFrame
    """
    if lookback_hours_list is None:
        lookback_hours_list = [24, 48, 72, 168, 336, 720]

    rows = []

    # 1. Per-stock individual results
    per_stock_equity = {}
    for sym, info in best_per_stock.items():
        m = info["metrics"]
        rows.append({
            "strategy": f"individual_{sym}",
            "total_return_pct": m["total_return"] * 100,
            "sortino": m["sortino"],
            "max_drawdown_pct": m.get("max_drawdown", 0) * 100,
            "num_switches": 0,
            "bars_in_cash": 0,
            "final_equity": m.get("final_equity", 10000),
        })
        per_stock_equity[sym] = info["equity_df"]

    # 2. Buy-and-hold benchmarks
    from src.trade_directions import DEFAULT_SHORT_ONLY_STOCKS
    for sym in best_per_stock:
        direction = "short" if sym in DEFAULT_SHORT_ONLY_STOCKS else "long"
        bah = compute_buy_and_hold(sym, holdout_days=holdout_days,
                                    data_root=data_root, direction=direction)
        rows.append({
            "strategy": f"buyhold_{sym}",
            "total_return_pct": bah["total_return"] * 100,
            "sortino": bah["sortino"],
            "max_drawdown_pct": bah.get("max_drawdown", 0) * 100,
            "num_switches": 0,
            "bars_in_cash": 0,
            "final_equity": bah.get("final_equity", 10000),
        })

    # 3. Meta-selector with lookback sweep
    if len(per_stock_equity) >= 2:
        for lb in lookback_hours_list:
            config = MetaSelectorConfig(
                lookback_hours=lb,
                sit_out_if_all_negative=True,
            )
            result = run_meta_simulation(per_stock_equity, config)
            rows.append({
                "strategy": f"meta_sortino_lb{lb}h",
                "total_return_pct": result.total_return * 100,
                "sortino": result.sortino,
                "max_drawdown_pct": result.max_drawdown * 100,
                "num_switches": result.num_switches,
                "bars_in_cash": result.bars_in_cash,
                "final_equity": result.equity_curve[-1],
            })

        # 4. Meta-selector without sit-out
        for lb in [72, 168]:
            config = MetaSelectorConfig(
                lookback_hours=lb,
                sit_out_if_all_negative=False,
            )
            result = run_meta_simulation(per_stock_equity, config)
            rows.append({
                "strategy": f"meta_nosit_lb{lb}h",
                "total_return_pct": result.total_return * 100,
                "sortino": result.sortino,
                "max_drawdown_pct": result.max_drawdown * 100,
                "num_switches": result.num_switches,
                "bars_in_cash": 0,
                "final_equity": result.equity_curve[-1],
            })

    # 5. Equal-weight (average of all individual returns)
    if len(per_stock_equity) >= 2:
        # Compute equal-weight equity curve (average of individual curves)
        from fastalgorithms.per_stock.meta_selector import align_equity_curves
        try:
            common_ts, equities, _ = align_equity_curves(per_stock_equity)
            n = len(common_ts)
            n_stocks = len(equities)
            # Each stock gets 1/n of capital
            eq_weight_equity = np.zeros(n)
            for sym, eq in equities.items():
                # Normalize to start at initial_cash/n_stocks
                per_stock_alloc = 10000.0 / n_stocks
                normalized = per_stock_alloc * eq / eq[0]
                eq_weight_equity += normalized

            total_ret = (eq_weight_equity[-1] - eq_weight_equity[0]) / eq_weight_equity[0]
            sort_val = compute_sortino(eq_weight_equity)
            peak = np.maximum.accumulate(eq_weight_equity)
            dd = (peak - eq_weight_equity) / (peak + 1e-10)
            max_dd = float(np.max(dd))

            rows.append({
                "strategy": "equal_weight",
                "total_return_pct": total_ret * 100,
                "sortino": sort_val,
                "max_drawdown_pct": max_dd * 100,
                "num_switches": 0,
                "bars_in_cash": 0,
                "final_equity": eq_weight_equity[-1],
            })
        except Exception as e:
            logger.warning("Equal-weight failed: {}", e)

    df = pd.DataFrame(rows)
    return df.sort_values("sortino", ascending=False).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Compare per-stock strategies")
    parser.add_argument("--checkpoint-root", type=Path,
                        default=Path("fastalgorithms/per_stock/checkpoints"))
    parser.add_argument("--symbols", type=str, default="NVDA,PLTR,GOOG,DBX,TRIP,MTCH")
    parser.add_argument("--holdout-days", type=int, default=90)
    parser.add_argument("--min-edge", type=float, default=0.0)
    parser.add_argument("--max-hold-hours", type=int, default=6)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")]

    logger.info("=== Strategy Comparison ({} stocks, {}d holdout) ===", len(symbols), args.holdout_days)

    # Select best models per stock
    best = select_best_per_stock(
        checkpoint_root=args.checkpoint_root,
        symbols=symbols,
        holdout_days=args.holdout_days,
        min_edge=args.min_edge,
        max_hold_hours=args.max_hold_hours,
    )

    if not best:
        logger.error("No per-stock models found. Run training first.")
        return

    # Run comparison
    df = run_comparison(best, holdout_days=args.holdout_days)

    # Print results table
    logger.info("\n=== Strategy Comparison Results ===")
    logger.info("{:30s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s}",
                "Strategy", "Return%", "Sortino", "MaxDD%", "Switches", "Cash#")
    logger.info("-" * 80)
    for _, row in df.iterrows():
        logger.info("{:30s} {:+7.2f}% {:8.2f} {:7.1f}% {:8d} {:8d}",
                    row["strategy"],
                    row["total_return_pct"],
                    row["sortino"],
                    row["max_drawdown_pct"],
                    int(row["num_switches"]),
                    int(row["bars_in_cash"]))

    # Save results
    output_path = args.output or (args.checkpoint_root / f"comparison_{args.holdout_days}d.json")
    df.to_json(output_path, orient="records", indent=2)
    logger.info("\nResults saved to {}", output_path)


if __name__ == "__main__":
    main()
