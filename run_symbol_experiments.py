#!/usr/bin/env python3
"""Systematic Symbol Pair Experiments for V4 Strategy.

Tests different symbol combinations and allocation sizes to find:
1. Most profitable symbol pairs for this strategy
2. Optimal allocation sizes for better Sortino
3. Best combination of symbols for portfolio

Usage:
    python run_symbol_experiments.py --checkpoint <path> --days 60
    python run_symbol_experiments.py --checkpoint <path> --mode sortino
    python run_symbol_experiments.py --checkpoint <path> --mode pairs
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

from backtest_v4 import run_backtest, simulate_trade, get_future_ohlc
from neuraldailyv4.runtime import DailyTradingRuntimeV4


# Symbol universe for experiments
ALL_SYMBOLS = {
    "mega_cap": ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
    "semiconductors": ["AMD", "AVGO", "QCOM", "MU", "MRVL", "AMAT", "LRCX", "INTC"],
    "software": ["CRM", "ORCL", "SNOW", "PLTR", "NOW", "DDOG", "ADBE"],
    "fintech": ["V", "MA", "PYPL", "SQ", "COIN"],
    "consumer": ["WMT", "HD", "NKE", "COST", "TGT", "MCD"],
    "healthcare": ["JNJ", "UNH", "LLY", "ABBV", "MRK", "PFE"],
    "crypto": ["BTCUSD", "ETHUSD", "SOLUSD"],
    "etfs": ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF"],
}

# Allocation sizes to test (as fraction of portfolio)
ALLOCATION_SIZES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

# Stop loss levels to test
STOP_LOSS_LEVELS = [None, 0.02, 0.03, 0.05, 0.07, 0.10]


def calculate_sortino(returns: List[float], target: float = 0.0) -> float:
    """Calculate Sortino ratio (penalizes downside volatility)."""
    if not returns:
        return 0.0
    returns_arr = np.array(returns)
    excess = returns_arr - target
    downside = excess[excess < 0]
    downside_std = np.std(downside) if len(downside) > 0 else 1e-8
    return np.mean(excess) / downside_std if downside_std > 0 else 0.0


def calculate_max_drawdown(returns: List[float]) -> float:
    """Calculate maximum drawdown from cumulative returns."""
    if not returns:
        return 0.0
    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative - running_max
    return abs(min(drawdowns)) if len(drawdowns) > 0 else 0.0


def run_single_symbol_backtest(
    runtime: DailyTradingRuntimeV4,
    symbol: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    stop_loss_pct: Optional[float] = None,
) -> Dict:
    """Run backtest for a single symbol."""
    trades = run_backtest(
        runtime,
        [symbol],
        start_date,
        end_date,
        verbose=False,
        stop_loss_pct=stop_loss_pct,
    )

    if not trades:
        return {
            "symbol": symbol,
            "num_trades": 0,
            "total_return": 0.0,
            "avg_return": 0.0,
            "win_rate": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "tp_rate": 0.0,
            "avg_hold_days": 0.0,
        }

    returns = [t["return_pct"] for t in trades]
    total_return = sum(returns)
    avg_return = np.mean(returns)
    std_return = np.std(returns) if len(returns) > 1 else 1e-8
    sharpe = avg_return / std_return if std_return > 0 else 0.0
    sortino = calculate_sortino(returns)
    win_rate = sum(1 for r in returns if r > 0) / len(returns)
    tp_rate = sum(1 for t in trades if t.get("tp_hit")) / len(trades)
    max_dd = calculate_max_drawdown(returns)
    avg_hold = np.mean([t["actual_hold_days"] for t in trades])

    return {
        "symbol": symbol,
        "num_trades": len(trades),
        "total_return": total_return,
        "avg_return": avg_return,
        "win_rate": win_rate,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "tp_rate": tp_rate,
        "avg_hold_days": avg_hold,
        "trades": trades,
    }


def run_symbol_pair_experiment(
    runtime: DailyTradingRuntimeV4,
    symbols: List[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    stop_loss_pct: Optional[float] = None,
) -> Dict:
    """Test a specific symbol combination."""
    all_trades = run_backtest(
        runtime,
        symbols,
        start_date,
        end_date,
        verbose=False,
        stop_loss_pct=stop_loss_pct,
    )

    if not all_trades:
        return {
            "symbols": symbols,
            "num_trades": 0,
            "total_return": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
        }

    returns = [t["return_pct"] for t in all_trades]
    total_return = sum(returns)
    avg_return = np.mean(returns)
    std_return = np.std(returns) if len(returns) > 1 else 1e-8
    sharpe = avg_return / std_return if std_return > 0 else 0.0
    sortino = calculate_sortino(returns)
    win_rate = sum(1 for r in returns if r > 0) / len(returns)
    max_dd = calculate_max_drawdown(returns)

    return {
        "symbols": symbols,
        "num_trades": len(all_trades),
        "total_return": total_return,
        "avg_return": avg_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "win_rate": win_rate,
        "max_drawdown": max_dd,
    }


def find_best_symbol_pairs(
    runtime: DailyTradingRuntimeV4,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    top_k: int = 20,
) -> List[Dict]:
    """Find best performing individual symbols."""
    logger.info("=" * 60)
    logger.info("INDIVIDUAL SYMBOL ANALYSIS")
    logger.info("=" * 60)

    all_symbols = []
    for group, syms in ALL_SYMBOLS.items():
        all_symbols.extend(syms)
    all_symbols = list(set(all_symbols))  # Remove duplicates

    results = []
    for symbol in all_symbols:
        logger.info(f"Testing {symbol}...")
        try:
            result = run_single_symbol_backtest(runtime, symbol, start_date, end_date)
            results.append(result)
        except Exception as e:
            logger.warning(f"Failed to test {symbol}: {e}")

    # Sort by Sortino ratio (risk-adjusted returns favoring upside)
    results.sort(key=lambda x: x["sortino"], reverse=True)

    logger.info("\n" + "=" * 60)
    logger.info("TOP PERFORMING SYMBOLS (by Sortino)")
    logger.info("=" * 60)
    for i, r in enumerate(results[:top_k]):
        logger.info(
            f"{i+1:2}. {r['symbol']:8} | Sortino: {r['sortino']:6.3f} | "
            f"Sharpe: {r['sharpe']:6.3f} | Return: {r['total_return']:7.2%} | "
            f"Win: {r['win_rate']:5.1%} | Trades: {r['num_trades']:3}"
        )

    # Also show worst performers
    logger.info("\n" + "=" * 60)
    logger.info("WORST PERFORMING SYMBOLS (avoid these)")
    logger.info("=" * 60)
    worst = sorted(results, key=lambda x: x["sortino"])[:10]
    for i, r in enumerate(worst):
        logger.info(
            f"{i+1:2}. {r['symbol']:8} | Sortino: {r['sortino']:6.3f} | "
            f"Sharpe: {r['sharpe']:6.3f} | Return: {r['total_return']:7.2%}"
        )

    return results


def test_allocation_sizes(
    runtime: DailyTradingRuntimeV4,
    symbols: List[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> List[Dict]:
    """Test different allocation sizes and their impact on Sortino."""
    logger.info("\n" + "=" * 60)
    logger.info("ALLOCATION SIZE EXPERIMENT")
    logger.info("=" * 60)
    logger.info(f"Testing symbols: {symbols}")

    results = []

    # Get base trades first
    base_trades = run_backtest(
        runtime, symbols, start_date, end_date, verbose=False
    )

    if not base_trades:
        logger.warning("No trades generated for allocation experiment")
        return results

    for alloc_size in ALLOCATION_SIZES:
        # Scale returns by allocation size
        scaled_returns = [t["return_pct"] * alloc_size for t in base_trades]

        total_return = sum(scaled_returns)
        avg_return = np.mean(scaled_returns)
        std_return = np.std(scaled_returns) if len(scaled_returns) > 1 else 1e-8
        sharpe = avg_return / std_return if std_return > 0 else 0.0
        sortino = calculate_sortino(scaled_returns)
        max_dd = calculate_max_drawdown(scaled_returns)

        result = {
            "allocation": alloc_size,
            "total_return": total_return,
            "avg_return": avg_return,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_dd,
            "num_trades": len(base_trades),
        }
        results.append(result)

        logger.info(
            f"Alloc {alloc_size:5.0%} | Return: {total_return:7.2%} | "
            f"Sortino: {sortino:6.3f} | Sharpe: {sharpe:6.3f} | MaxDD: {max_dd:6.2%}"
        )

    # Find optimal allocation for Sortino
    best_sortino = max(results, key=lambda x: x["sortino"])
    logger.info(f"\nBest Sortino allocation: {best_sortino['allocation']:.0%}")

    return results


def test_stop_loss_levels(
    runtime: DailyTradingRuntimeV4,
    symbols: List[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> List[Dict]:
    """Test different stop loss levels."""
    logger.info("\n" + "=" * 60)
    logger.info("STOP LOSS EXPERIMENT")
    logger.info("=" * 60)

    results = []

    for sl in STOP_LOSS_LEVELS:
        sl_str = f"{sl:.0%}" if sl else "None"
        trades = run_backtest(
            runtime, symbols, start_date, end_date, verbose=False, stop_loss_pct=sl
        )

        if not trades:
            continue

        returns = [t["return_pct"] for t in trades]
        total_return = sum(returns)
        avg_return = np.mean(returns)
        std_return = np.std(returns) if len(returns) > 1 else 1e-8
        sharpe = avg_return / std_return if std_return > 0 else 0.0
        sortino = calculate_sortino(returns)
        win_rate = sum(1 for r in returns if r > 0) / len(returns)
        max_dd = calculate_max_drawdown(returns)
        sl_hits = sum(1 for t in trades if t.get("sl_hit"))

        result = {
            "stop_loss": sl,
            "total_return": total_return,
            "sharpe": sharpe,
            "sortino": sortino,
            "win_rate": win_rate,
            "max_drawdown": max_dd,
            "sl_hits": sl_hits,
            "num_trades": len(trades),
        }
        results.append(result)

        logger.info(
            f"SL {sl_str:5} | Return: {total_return:7.2%} | "
            f"Sortino: {sortino:6.3f} | MaxDD: {max_dd:6.2%} | "
            f"SL Hits: {sl_hits:3}/{len(trades)}"
        )

    return results


def find_best_sector_combinations(
    runtime: DailyTradingRuntimeV4,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> List[Dict]:
    """Test different sector combinations."""
    logger.info("\n" + "=" * 60)
    logger.info("SECTOR COMBINATION EXPERIMENT")
    logger.info("=" * 60)

    results = []

    # Test each sector individually
    for sector, symbols in ALL_SYMBOLS.items():
        logger.info(f"Testing sector: {sector}")
        result = run_symbol_pair_experiment(runtime, symbols, start_date, end_date)
        result["sector"] = sector
        results.append(result)

        logger.info(
            f"  {sector:15} | Sortino: {result['sortino']:6.3f} | "
            f"Sharpe: {result['sharpe']:6.3f} | Return: {result['total_return']:7.2%}"
        )

    # Test sector combinations (2 sectors at a time)
    sector_pairs = list(itertools.combinations(ALL_SYMBOLS.keys(), 2))
    for s1, s2 in sector_pairs[:10]:  # Limit to top 10 pairs
        combined = ALL_SYMBOLS[s1] + ALL_SYMBOLS[s2]
        logger.info(f"Testing {s1} + {s2}...")
        result = run_symbol_pair_experiment(runtime, combined, start_date, end_date)
        result["sector"] = f"{s1}+{s2}"
        results.append(result)

    # Sort by Sortino
    results.sort(key=lambda x: x["sortino"], reverse=True)

    logger.info("\n" + "=" * 60)
    logger.info("BEST SECTOR COMBINATIONS (by Sortino)")
    logger.info("=" * 60)
    for i, r in enumerate(results[:10]):
        logger.info(
            f"{i+1:2}. {r['sector']:25} | Sortino: {r['sortino']:6.3f} | "
            f"Return: {r['total_return']:7.2%}"
        )

    return results


def run_comprehensive_experiment(
    checkpoint_path: str,
    days: int = 60,
    output_dir: str = "experiments/symbol_analysis",
) -> Dict:
    """Run all experiments and save results."""
    logger.info("=" * 70)
    logger.info("V4 COMPREHENSIVE SYMBOL EXPERIMENT")
    logger.info("=" * 70)

    # Load runtime
    runtime = DailyTradingRuntimeV4(Path(checkpoint_path))

    end_date = pd.Timestamp.now(tz="UTC")
    start_date = end_date - timedelta(days=days)

    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Period: {start_date.date()} to {end_date.date()} ({days} days)")

    results = {}

    # 1. Find best individual symbols
    symbol_results = find_best_symbol_pairs(runtime, start_date, end_date)
    results["individual_symbols"] = symbol_results

    # Get top 10 symbols for further experiments
    top_symbols = [r["symbol"] for r in symbol_results[:10] if r["num_trades"] > 0]
    logger.info(f"\nUsing top symbols for experiments: {top_symbols}")

    # 2. Test allocation sizes
    alloc_results = test_allocation_sizes(runtime, top_symbols, start_date, end_date)
    results["allocation_sizes"] = alloc_results

    # 3. Test stop loss levels
    sl_results = test_stop_loss_levels(runtime, top_symbols, start_date, end_date)
    results["stop_loss_levels"] = sl_results

    # 4. Test sector combinations
    sector_results = find_best_sector_combinations(runtime, start_date, end_date)
    results["sector_combinations"] = sector_results

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_path / f"experiment_results_{timestamp}.json"

    # Convert to JSON-serializable format
    json_results = {
        "checkpoint": checkpoint_path,
        "period_days": days,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "individual_symbols": [
            {k: v for k, v in r.items() if k != "trades"}
            for r in symbol_results
        ],
        "allocation_sizes": alloc_results,
        "stop_loss_levels": sl_results,
        "sector_combinations": sector_results,
        "recommendations": {
            "best_symbols": top_symbols[:5],
            "optimal_allocation": max(alloc_results, key=lambda x: x["sortino"])["allocation"]
            if alloc_results else 0.15,
            "optimal_stop_loss": max(sl_results, key=lambda x: x["sortino"])["stop_loss"]
            if sl_results else None,
        },
    }

    with open(results_file, "w") as f:
        json.dump(json_results, f, indent=2, default=str)

    logger.info(f"\nResults saved to: {results_file}")

    # Print final recommendations
    logger.info("\n" + "=" * 70)
    logger.info("FINAL RECOMMENDATIONS")
    logger.info("=" * 70)
    logger.info(f"Best symbols: {json_results['recommendations']['best_symbols']}")
    logger.info(f"Optimal allocation: {json_results['recommendations']['optimal_allocation']:.0%}")
    logger.info(f"Optimal stop loss: {json_results['recommendations']['optimal_stop_loss']}")

    return json_results


def main():
    parser = argparse.ArgumentParser(description="V4 Symbol Experiments")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--days", type=int, default=60, help="Backtest period in days")
    parser.add_argument("--output-dir", type=str, default="experiments/symbol_analysis",
                        help="Output directory for results")
    parser.add_argument("--mode", type=str, default="full",
                        choices=["full", "symbols", "allocation", "stoploss", "sectors"],
                        help="Experiment mode")
    args = parser.parse_args()

    if args.mode == "full":
        run_comprehensive_experiment(args.checkpoint, args.days, args.output_dir)
    else:
        runtime = DailyTradingRuntimeV4(Path(args.checkpoint))
        end_date = pd.Timestamp.now(tz="UTC")
        start_date = end_date - timedelta(days=args.days)

        if args.mode == "symbols":
            find_best_symbol_pairs(runtime, start_date, end_date)
        elif args.mode == "allocation":
            top_symbols = ["SPY", "QQQ", "AAPL", "NVDA", "AMD"]
            test_allocation_sizes(runtime, top_symbols, start_date, end_date)
        elif args.mode == "stoploss":
            top_symbols = ["SPY", "QQQ", "AAPL", "NVDA", "AMD"]
            test_stop_loss_levels(runtime, top_symbols, start_date, end_date)
        elif args.mode == "sectors":
            find_best_sector_combinations(runtime, start_date, end_date)


if __name__ == "__main__":
    main()
