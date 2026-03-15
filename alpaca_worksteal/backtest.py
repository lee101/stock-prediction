#!/usr/bin/env python3
"""Backtest work-stealing on Alpaca stocks + crypto (hourly)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpaca_worksteal.strategy import (
    AlpacaWorkStealConfig, load_hourly_bars, run_alpaca_worksteal_backtest,
)

# Top liquid stocks + crypto for Alpaca
STOCK_UNIVERSE = [
    # Mega-cap tech
    "AAPL", "MSFT", "NVDA", "GOOG", "AMZN", "META", "TSLA",
    # Growth/momentum
    "PLTR", "NET", "SHOP", "NFLX", "AMD", "CRM", "ADBE",
    # Financial
    "JPM", "GS", "V", "MA",
    # Other high-volume
    "COST", "WMT", "BA", "LLY", "AVGO",
    # ETFs
    "SPY", "QQQ",
]

CRYPTO_UNIVERSE = [
    "BTCUSD", "ETHUSD", "SOLUSD", "LTCUSD", "AVAXUSD",
    "DOGEUSD", "LINKUSD", "AAVEUSD",
]

FULL_UNIVERSE = STOCK_UNIVERSE + CRYPTO_UNIVERSE


def print_results(equity_df, trades, metrics):
    print(f"\n{'='*60}")
    print(f"ALPACA WORK-STEALING BACKTEST")
    print(f"{'='*60}")
    print(f"Total Return: {metrics.get('total_return_pct', 0):.2f}%")
    print(f"Sortino:      {metrics.get('sortino', 0):.2f}")
    print(f"Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%")
    print(f"Win Rate:     {metrics.get('win_rate', 0):.1f}%")
    print(f"Final Equity: ${metrics.get('final_equity', 0):.2f}")
    print(f"Trades:       {metrics.get('n_trades', 0)}")
    print(f"Bars:         {metrics.get('n_bars', 0)}")

    if not trades:
        return

    buys = [t for t in trades if t.side == "buy"]
    exits = [t for t in trades if t.side in ("sell", "cover")]
    winners = [t for t in exits if t.pnl > 0]
    losers = [t for t in exits if t.pnl <= 0]

    print(f"\nEntries: {len(buys)}")
    print(f"Exits: {len(exits)} ({len(winners)} wins, {len(losers)} losses)")
    if winners:
        print(f"Avg Win:  ${sum(t.pnl for t in winners)/len(winners):.2f}")
    if losers:
        print(f"Avg Loss: ${sum(t.pnl for t in losers)/len(losers):.2f}")
    total_pnl = sum(t.pnl for t in exits)
    total_fees = sum(t.fee for t in trades)
    print(f"Total PnL: ${total_pnl:.2f}")
    print(f"Total Fees: ${total_fees:.2f}")

    # Per-symbol breakdown
    sym_pnl = {}
    sym_n = {}
    for t in exits:
        sym_pnl[t.symbol] = sym_pnl.get(t.symbol, 0) + t.pnl
        sym_n[t.symbol] = sym_n.get(t.symbol, 0) + 1
    if sym_pnl:
        print(f"\nPer-Symbol PnL (top/bottom 10):")
        sorted_syms = sorted(sym_pnl, key=lambda s: sym_pnl[s], reverse=True)
        for sym in sorted_syms[:5]:
            print(f"  {sym:8s} ${sym_pnl[sym]:>8.2f} ({sym_n[sym]} trades)")
        if len(sorted_syms) > 10:
            print(f"  ...")
        for sym in sorted_syms[-5:]:
            print(f"  {sym:8s} ${sym_pnl[sym]:>8.2f} ({sym_n[sym]} trades)")

    # Exit reasons
    reasons = {}
    for t in exits:
        reasons[t.reason] = reasons.get(t.reason, 0) + 1
    print(f"\nExit Reasons:")
    for r, c in sorted(reasons.items(), key=lambda x: x[1], reverse=True):
        print(f"  {r:20s} {c}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="trainingdatahourly")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--stocks-only", action="store_true")
    parser.add_argument("--crypto-only", action="store_true")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--days", type=int, default=60, help="Backtest last N days")
    parser.add_argument("--dip-pct", type=float, default=0.05)
    parser.add_argument("--profit-target", type=float, default=0.03)
    parser.add_argument("--stop-loss", type=float, default=0.05)
    parser.add_argument("--max-positions", type=int, default=8)
    parser.add_argument("--max-hold-hours", type=int, default=48)
    parser.add_argument("--trailing-stop", type=float, default=0.003)
    parser.add_argument("--cash", type=float, default=50000.0)
    parser.add_argument("--leverage", type=float, default=2.0)
    args = parser.parse_args()

    if args.symbols:
        symbols = args.symbols
    elif args.stocks_only:
        symbols = STOCK_UNIVERSE
    elif args.crypto_only:
        symbols = CRYPTO_UNIVERSE
    else:
        symbols = FULL_UNIVERSE

    print(f"Loading {len(symbols)} symbols from {args.data_dir}")
    all_bars = load_hourly_bars(args.data_dir, symbols)
    print(f"Loaded {len(all_bars)} symbols with data")

    if not all_bars:
        print("ERROR: No data loaded")
        return 1

    if not args.start_date:
        import pandas as pd
        latest = max(df["timestamp"].max() for df in all_bars.values())
        args.end_date = str(latest.date())
        args.start_date = str((latest - pd.Timedelta(days=args.days)).date())
    print(f"Date range: {args.start_date} to {args.end_date}")

    config = AlpacaWorkStealConfig(
        dip_pct=args.dip_pct,
        profit_target_pct=args.profit_target,
        stop_loss_pct=args.stop_loss,
        max_positions=args.max_positions,
        max_hold_hours=args.max_hold_hours,
        trailing_stop_pct=args.trailing_stop,
        initial_cash=args.cash,
        max_leverage=args.leverage,
    )

    print(f"\nConfig: dip={config.dip_pct:.0%} tp={config.profit_target_pct:.0%} "
          f"sl={config.stop_loss_pct:.0%} trail={config.trailing_stop_pct:.1%} "
          f"maxpos={config.max_positions} maxhold={config.max_hold_hours}h "
          f"lev={config.max_leverage:.0f}x cash=${config.initial_cash:,.0f}")

    equity_df, trades_list, metrics = run_alpaca_worksteal_backtest(
        all_bars, config,
        # start_date/end_date filtering would need to be added to the strategy
    )

    print_results(equity_df, trades_list, metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
