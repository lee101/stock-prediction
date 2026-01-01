#!/usr/bin/env python3
"""Tune max position size for optimal risk-adjusted returns.

Tests different position sizes on the same model to find optimal allocation.

Usage:
    python tune_position_size.py --checkpoint neuraldailyv4/checkpoints/v10_extended/epoch_0020.pt
    python tune_position_size.py --checkpoint ... --days 60
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))


SYMBOLS = [
    "SPY", "QQQ", "IWM", "XLF", "XLK", "DIA",
    "AAPL", "GOOGL", "NVDA",
    "V", "MA",
    "AMD", "LRCX", "AMAT", "QCOM",
    "MCD", "TGT",
    "LLY", "PFE",
    "ADBE",
]


def calculate_sortino(returns: List[float], target: float = 0.0) -> float:
    """Calculate Sortino ratio."""
    if not returns or len(returns) < 2:
        return 0.0
    returns_arr = np.array(returns)
    excess = returns_arr - target
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float('inf') if np.mean(excess) > 0 else 0.0
    downside_std = np.std(downside)
    if downside_std < 1e-8:
        return float('inf') if np.mean(excess) > 0 else 0.0
    return float(np.mean(excess) / downside_std)


def run_backtest_with_position_size(
    checkpoint_path: Path,
    max_position: float,
    days: int = 90,
) -> Dict:
    """Run backtest with specific position size."""
    import pandas as pd
    from backtest_v4 import run_backtest
    from neuraldailyv4.runtime import DailyTradingRuntimeV4

    try:
        runtime = DailyTradingRuntimeV4(
            checkpoint_path,
            trade_crypto=False,
            max_exit_days=2,
            max_position_override=max_position,
        )

        # Calculate date range
        end_date = pd.Timestamp.now(tz="UTC")
        start_date = end_date - pd.Timedelta(days=days)

        trades = run_backtest(runtime, SYMBOLS, start_date, end_date)

        if not trades:
            return {"error": "No trades", "max_position": max_position}

        returns = [t["return_pct"] for t in trades]
        wins = sum(1 for r in returns if r > 0)
        tp_hits = sum(1 for t in trades if t.get("exit_type") == "take_profit")

        total_return = sum(returns)
        avg_return = np.mean(returns)
        std_return = np.std(returns) if len(returns) > 1 else 0.001

        # Calculate max drawdown (simplified)
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0

        return {
            "max_position": max_position,
            "total_return": total_return,
            "avg_return": avg_return,
            "std_return": std_return,
            "sharpe": avg_return / std_return if std_return > 0 else 0.0,
            "sortino": calculate_sortino(returns),
            "win_rate": wins / len(trades) * 100,
            "tp_rate": tp_hits / len(trades) * 100,
            "num_trades": len(trades),
            "max_drawdown": max_drawdown,
            "risk_adj_return": total_return / (max_drawdown + 1),  # Return per unit risk
        }

    except Exception as e:
        logger.error(f"Backtest failed for position {max_position}: {e}")
        return {"error": str(e), "max_position": max_position}


def main():
    parser = argparse.ArgumentParser(description="Tune Position Size")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--positions", type=str, default="0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50",
                       help="Comma-separated position sizes to test")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    position_sizes = [float(p) for p in args.positions.split(",")]

    logger.info("=" * 70)
    logger.info("POSITION SIZE TUNING")
    logger.info("=" * 70)
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Backtest days: {args.days}")
    logger.info(f"Testing positions: {[f'{p:.0%}' for p in position_sizes]}")
    logger.info("")

    results = []

    for pos_size in position_sizes:
        logger.info(f"Testing {pos_size:.0%} max position...")
        result = run_backtest_with_position_size(checkpoint_path, pos_size, args.days)
        results.append(result)

        if "error" not in result:
            logger.info(f"  Return: {result['total_return']:.2f}%, "
                       f"Sharpe: {result['sharpe']:.3f}, "
                       f"Sortino: {result['sortino']:.3f}, "
                       f"MaxDD: {result['max_drawdown']:.2f}%")

    # Find best by different metrics
    valid_results = [r for r in results if "error" not in r]

    if not valid_results:
        logger.error("No valid results!")
        return

    best_return = max(valid_results, key=lambda x: x["total_return"])
    best_sharpe = max(valid_results, key=lambda x: x["sharpe"])
    best_sortino = max(valid_results, key=lambda x: x["sortino"])
    best_risk_adj = max(valid_results, key=lambda x: x["risk_adj_return"])

    logger.info("")
    logger.info("=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info("")
    logger.info(f"{'Position':<10} {'Return':<10} {'Sharpe':<10} {'Sortino':<10} {'WinRate':<10} {'MaxDD':<10} {'Trades':<8}")
    logger.info("-" * 70)

    for r in valid_results:
        pos_str = f"{r['max_position']:.0%}"
        logger.info(f"{pos_str:<10} "
                   f"{r['total_return']:>8.2f}% "
                   f"{r['sharpe']:>9.3f} "
                   f"{r['sortino']:>9.3f} "
                   f"{r['win_rate']:>8.1f}% "
                   f"{r['max_drawdown']:>8.2f}% "
                   f"{r['num_trades']:>7}")

    logger.info("")
    logger.info("BEST BY METRIC:")
    logger.info(f"  Best Return:   {best_return['max_position']:.0%} ({best_return['total_return']:.2f}%)")
    logger.info(f"  Best Sharpe:   {best_sharpe['max_position']:.0%} ({best_sharpe['sharpe']:.3f})")
    logger.info(f"  Best Sortino:  {best_sortino['max_position']:.0%} ({best_sortino['sortino']:.3f})")
    logger.info(f"  Best Risk-Adj: {best_risk_adj['max_position']:.0%} (return/DD = {best_risk_adj['risk_adj_return']:.2f})")

    # Recommend position size
    # Weight: 40% Sortino, 30% Sharpe, 20% Return, 10% Risk-Adj
    scores = {}
    for r in valid_results:
        pos = r['max_position']
        # Normalize each metric relative to best
        sortino_score = r['sortino'] / best_sortino['sortino'] if best_sortino['sortino'] > 0 else 0
        sharpe_score = r['sharpe'] / best_sharpe['sharpe'] if best_sharpe['sharpe'] > 0 else 0
        return_score = r['total_return'] / best_return['total_return'] if best_return['total_return'] > 0 else 0
        risk_adj_score = r['risk_adj_return'] / best_risk_adj['risk_adj_return'] if best_risk_adj['risk_adj_return'] > 0 else 0

        scores[pos] = 0.4 * sortino_score + 0.3 * sharpe_score + 0.2 * return_score + 0.1 * risk_adj_score

    recommended = max(scores.items(), key=lambda x: x[1])

    logger.info("")
    logger.info(f"RECOMMENDED: {recommended[0]:.0%} max position (composite score: {recommended[1]:.3f})")

    # Save results
    output_dir = Path("experiments/position_tuning")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"position_tune_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump({
            "checkpoint": str(checkpoint_path),
            "days": args.days,
            "results": results,
            "recommended_position": recommended[0],
            "composite_score": recommended[1],
        }, f, indent=2)

    logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
