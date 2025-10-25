#!/usr/bin/env python3
"""Quick analytics for simulator trade logs.

Usage:
    python scripts/analyze_trades_csv.py marketsimulator/run_logs/trades.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

ISO_FORMATS = [
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S.%f%z",
    "%Y-%m-%d %H:%M:%S%z",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze simulator trade CSV exports.")
    parser.add_argument("trades_csv", type=Path, help="Path to the trades.csv file.")
    parser.add_argument("--per-cycle", action="store_true", help="List individual holding cycles.")
    return parser.parse_args()


def parse_timestamp(value: str) -> datetime:
    for fmt in ISO_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unrecognised timestamp format: {value}")


def load_trades(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def analyse(trades: List[Dict[str, str]], list_cycles: bool = False) -> None:
    if not trades:
        print("No trades found.")
        return

    per_symbol: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    cycles: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    positions: Dict[str, float] = defaultdict(float)
    entry_time: Dict[str, datetime] = {}
    cumulative_pnl = 0.0
    worst_pnl = 0.0

    for trade in trades:
        symbol = trade["symbol"]
        side = trade["side"].lower()
        qty = float(trade["qty"])
        notional = float(trade["notional"])
        fee = float(trade["fee"])
        cash_delta = float(trade["cash_delta"])
        slip_bps = float(trade.get("slip_bps", 0.0))
        timestamp = parse_timestamp(trade["timestamp"])

        stats = per_symbol[symbol]
        stats["trades"] += 1
        stats[f"{side}_trades"] += 1
        stats["gross_notional"] += abs(notional)
        stats["fees"] += fee
        stats["cash_delta"] += cash_delta
        stats["total_qty"] += qty
        stats["slip_bps_sum"] += abs(slip_bps)

        signed_qty = qty if side == "buy" else -qty
        if abs(positions[symbol]) < 1e-9:
            entry_time[symbol] = timestamp
        positions[symbol] += signed_qty
        if abs(positions[symbol]) < 1e-9:
            start = entry_time.pop(symbol, None)
            if start:
                duration = (timestamp - start).total_seconds()
                cycles[symbol].append(
                    {"start": start.timestamp(), "end": timestamp.timestamp(), "seconds": duration}
                )

        cumulative_pnl += cash_delta
        worst_pnl = min(worst_pnl, cumulative_pnl)

    print("=== Trade Summary ===")
    print(f"Total trades: {int(sum(s['trades'] for s in per_symbol.values()))}")
    print(f"Aggregate PnL: ${cumulative_pnl:,.2f}")
    print(f"Aggregate fees: ${sum(s['fees'] for s in per_symbol.values()):,.2f}")
    print(f"Worst cumulative cash delta: ${worst_pnl:,.2f}")

    header = (
        "Symbol",
        "Trades",
        "Buys",
        "Sells",
        "Net PnL ($)",
        "Fees ($)",
        "Avg Qty",
        "Avg Slip (bps)",
        "Avg Hold (h)",
    )
    print("\n" + " | ".join(f"{h:>12}" for h in header))
    print("-" * (len(header) * 15))
    for symbol, stats in sorted(per_symbol.items()):
        trades_count = int(stats["trades"])
        avg_qty = stats["total_qty"] / trades_count if trades_count else 0.0
        slip = stats["slip_bps_sum"] / trades_count if trades_count else 0.0
        durations = [cycle["seconds"] for cycle in cycles.get(symbol, [])]
        avg_hold_hours = (sum(durations) / len(durations) / 3600.0) if durations else 0.0
        print(
            f"{symbol:>12} | "
            f"{trades_count:12d} | "
            f"{int(stats['buy_trades']):12d} | "
            f"{int(stats['sell_trades']):12d} | "
            f"{stats['cash_delta']:12.2f} | "
            f"{stats['fees']:12.2f} | "
            f"{avg_qty:12.3f} | "
            f"{slip:12.3f} | "
            f"{avg_hold_hours:12.2f}"
        )

    if list_cycles:
        print("\n=== Holding Cycles (seconds) ===")
        for symbol, rows in cycles.items():
            if not rows:
                continue
            print(f"{symbol}: " + ", ".join(f"{cycle['seconds']:.0f}" for cycle in rows))


def main() -> None:
    args = parse_args()
    if not args.trades_csv.exists():
        print(f"Trades file not found: {args.trades_csv}")
        return
    trades = load_trades(args.trades_csv)
    analyse(trades, list_cycles=args.per_cycle)


if __name__ == "__main__":
    main()
