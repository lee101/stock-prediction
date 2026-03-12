"""Benchmark unified vs individual trading systems.

Runs stock-only, crypto-only, and unified simulations on the same
time period to measure whether cross-asset coordination adds value.

Usage:
  python -m unified_orchestrator.benchmark --days 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

from newnanoalpacahourlyexp.marketsimulator.hourly_trader import (
    HourlyTraderMarketSimulator,
    HourlyTraderSimulationConfig,
)
from src.fees import get_fee_for_symbol
from src.symbol_utils import is_crypto_symbol


# Default symbol sets
STOCK_SYMBOLS = ["NVDA", "PLTR", "GOOG", "DBX", "TRIP", "MTCH"]
CRYPTO_SYMBOLS = ["BTCUSD", "ETHUSD", "SOLUSD"]


def load_bars_and_actions(
    symbols: list[str],
    days: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load hourly bars and latest actions for symbols."""
    all_bars = []
    all_actions = []

    for sym in symbols:
        is_crypto = is_crypto_symbol(sym)

        # Load bar data
        if is_crypto:
            # Try binance spot data
            csv_path = REPO / f"trainingdatahourly/crypto/{sym}.csv"
            if not csv_path.exists():
                csv_path = REPO / f"binance_spot_hourly/{sym.replace('USD', 'USDT')}.csv"
        else:
            csv_path = REPO / f"trainingdatahourly/stocks/{sym}_hist.pkl"

        try:
            if csv_path.suffix == ".pkl":
                raw = pd.read_pickle(csv_path)
                df = raw["data"] if isinstance(raw, dict) and "data" in raw else raw
            else:
                df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  Skip {sym}: {e}")
            continue

        if "timestamp" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)
        if "timestamp" not in df.columns and "date" in df.columns:
            df["timestamp"] = pd.to_datetime(df["date"], utc=True)
        elif "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        else:
            continue

        for col in ["open", "high", "low", "close"]:
            if col not in df.columns:
                continue

        df["symbol"] = sym
        df = df.sort_values("timestamp")

        # Take last N days
        if days > 0:
            cutoff = df["timestamp"].max() - pd.Timedelta(days=days)
            df = df[df["timestamp"] >= cutoff]

        all_bars.append(df[["timestamp", "symbol", "open", "high", "low", "close"]].copy())

        # Generate simple actions: use Chronos forecasts if available, else heuristic
        actions = df[["timestamp", "symbol"]].copy()
        # Simple momentum-based actions for benchmarking
        closes = df["close"].values
        sma_12 = pd.Series(closes).rolling(12, min_periods=1).mean().values
        sma_48 = pd.Series(closes).rolling(48, min_periods=1).mean().values

        # Buy when short SMA > long SMA, sell at +1% target
        buy_signal = sma_12 > sma_48
        actions["buy_price"] = np.where(buy_signal, closes * 0.998, 0.0)
        actions["sell_price"] = np.where(buy_signal, closes * 1.01, closes * 0.998)
        actions["buy_amount"] = np.where(buy_signal, 80.0, 0.0)
        actions["sell_amount"] = np.where(~buy_signal, 80.0, 0.0)

        all_actions.append(actions)

    if not all_bars:
        raise ValueError("No bar data loaded")

    bars = pd.concat(all_bars, ignore_index=True)
    actions = pd.concat(all_actions, ignore_index=True)
    return bars, actions


def run_benchmark(
    stock_symbols: list[str],
    crypto_symbols: list[str],
    days: int = 30,
    initial_cash: float = 10_000.0,
) -> dict:
    """Run all three benchmark modes."""
    results = {}

    # 1. Stock-only
    print(f"\n{'=' * 60}")
    print(f"BENCHMARK: Stock-only ({len(stock_symbols)} symbols, {days}d)")
    print(f"{'=' * 60}")
    try:
        bars, actions = load_bars_and_actions(stock_symbols, days)
        cfg = HourlyTraderSimulationConfig(
            initial_cash=initial_cash,
            allocation_pct=1.0 / len(stock_symbols),
            max_leverage=2.0,
            enforce_market_hours=True,
            allow_short=True,
            decision_lag_bars=1,
            fill_buffer_bps=5.0,
            partial_fill_on_touch=True,
        )
        sim = HourlyTraderMarketSimulator(cfg)
        result = sim.run(bars, actions)
        total_return = (result.equity_curve.iloc[-1] / initial_cash - 1) * 100
        results["stock_only"] = {
            "return_pct": total_return,
            "final_equity": result.equity_curve.iloc[-1],
            "fills": len(result.fills),
            "max_drawdown": result.metrics.get("max_drawdown_pct", 0),
            "sortino": result.metrics.get("sortino", 0),
        }
        print(f"  Return: {total_return:+.2f}%")
        print(f"  Final equity: ${result.equity_curve.iloc[-1]:,.2f}")
        print(f"  Fills: {len(result.fills)}")
    except Exception as e:
        print(f"  Error: {e}")
        results["stock_only"] = {"error": str(e)}

    # 2. Crypto-only
    print(f"\n{'=' * 60}")
    print(f"BENCHMARK: Crypto-only ({len(crypto_symbols)} symbols, {days}d)")
    print(f"{'=' * 60}")
    try:
        bars, actions = load_bars_and_actions(crypto_symbols, days)
        cfg = HourlyTraderSimulationConfig(
            initial_cash=initial_cash,
            allocation_pct=1.0 / len(crypto_symbols),
            max_leverage=1.0,
            enforce_market_hours=False,
            allow_short=False,
            decision_lag_bars=1,
            fill_buffer_bps=5.0,
            partial_fill_on_touch=True,
        )
        sim = HourlyTraderMarketSimulator(cfg)
        result = sim.run(bars, actions)
        total_return = (result.equity_curve.iloc[-1] / initial_cash - 1) * 100
        results["crypto_only"] = {
            "return_pct": total_return,
            "final_equity": result.equity_curve.iloc[-1],
            "fills": len(result.fills),
            "max_drawdown": result.metrics.get("max_drawdown_pct", 0),
            "sortino": result.metrics.get("sortino", 0),
        }
        print(f"  Return: {total_return:+.2f}%")
        print(f"  Final equity: ${result.equity_curve.iloc[-1]:,.2f}")
        print(f"  Fills: {len(result.fills)}")
    except Exception as e:
        print(f"  Error: {e}")
        results["crypto_only"] = {"error": str(e)}

    # 3. Unified (separate cash pools)
    print(f"\n{'=' * 60}")
    print(f"BENCHMARK: Unified ({len(stock_symbols)} stocks + {len(crypto_symbols)} crypto, {days}d)")
    print(f"{'=' * 60}")
    try:
        all_symbols = stock_symbols + crypto_symbols
        bars, actions = load_bars_and_actions(all_symbols, days)

        stock_cash = initial_cash * 0.6
        crypto_cash = initial_cash * 0.4

        cfg = HourlyTraderSimulationConfig(
            initial_cash=initial_cash,
            separate_cash_pools=True,
            initial_cash_stock=stock_cash,
            initial_cash_crypto=crypto_cash,
            allocation_pct=1.0 / len(all_symbols),
            max_leverage=2.0,  # Stocks get 2x, crypto capped at 1x in _available_cash
            enforce_market_hours=True,
            allow_short=True,
            decision_lag_bars=1,
            fill_buffer_bps=5.0,
            partial_fill_on_touch=True,
        )
        sim = HourlyTraderMarketSimulator(cfg)
        result = sim.run(bars, actions)
        total_return = (result.equity_curve.iloc[-1] / initial_cash - 1) * 100
        results["unified"] = {
            "return_pct": total_return,
            "final_equity": result.equity_curve.iloc[-1],
            "fills": len(result.fills),
            "max_drawdown": result.metrics.get("max_drawdown_pct", 0),
            "sortino": result.metrics.get("sortino", 0),
        }
        print(f"  Return: {total_return:+.2f}%")
        print(f"  Final equity: ${result.equity_curve.iloc[-1]:,.2f}")
        print(f"  Fills: {len(result.fills)}")
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        results["unified"] = {"error": str(e)}

    # Summary
    print(f"\n{'=' * 60}")
    print(f"BENCHMARK SUMMARY")
    print(f"{'=' * 60}")
    for mode, data in results.items():
        if "error" in data:
            print(f"  {mode}: ERROR - {data['error']}")
        else:
            print(f"  {mode}: {data['return_pct']:+.2f}% | "
                  f"${data['final_equity']:,.0f} | "
                  f"{data['fills']} fills")
    print(f"{'=' * 60}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark unified trading system")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--cash", type=float, default=10000.0)
    parser.add_argument("--stock-symbols", nargs="+", default=STOCK_SYMBOLS)
    parser.add_argument("--crypto-symbols", nargs="+", default=CRYPTO_SYMBOLS)
    args = parser.parse_args()

    run_benchmark(args.stock_symbols, args.crypto_symbols, args.days, args.cash)


if __name__ == "__main__":
    main()
