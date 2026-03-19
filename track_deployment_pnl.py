#!/usr/bin/env python
"""Post-deploy PnL tracker.

Reads fill events from ``strategy_state/fill_events.jsonl`` and the current
portfolio snapshot from ``strategy_state/unified_state.json`` to compute
production PnL metrics.

Metrics: daily PnL, cumulative PnL, annualized return, Sortino ratio,
max drawdown, win rate, trades per day.

Usage::

    python track_deployment_pnl.py
    python track_deployment_pnl.py --since 2026-03-01
    python track_deployment_pnl.py --checkpoint my_checkpoint --output pnl.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from src.trade_pnl_analyzer import _parse_timestamp


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

FILL_EVENTS_FILE = Path("strategy_state/fill_events.jsonl")
UNIFIED_STATE_FILE = Path("strategy_state/unified_state.json")

CSV_FIELDNAMES = ["date", "daily_pnl", "cumulative_pnl", "num_trades", "wins", "losses"]


def load_fill_events(path: Path) -> list[dict]:
    """Load JSONL fill events.  Returns [] on missing / empty file."""
    if not path.exists():
        return []
    events: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def load_unified_state(path: Path) -> dict:
    """Load the unified portfolio snapshot.  Returns {} on missing file."""
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}


# ---------------------------------------------------------------------------
# Per-trade PnL via FIFO lot matching
# ---------------------------------------------------------------------------

def compute_closed_trades(events: list[dict]) -> list[dict]:
    """Match buy/sell events per symbol using FIFO lots.

    Returns a list of closed-trade records, each with:
      symbol, close_date, pnl, entry_price, exit_price, qty, side
    """
    lots_by_symbol: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    # lots: (remaining_qty, price, action) — positive qty always
    trades: list[dict] = []

    for ev in events:
        symbol = ev.get("symbol", "")
        action = ev.get("action", "").lower()
        price = float(ev.get("fill_price", 0.0))
        qty = float(ev.get("fill_qty", 0.0))
        ts_raw = ev.get("timestamp", "")

        if not symbol or qty <= 0 or price <= 0:
            continue

        try:
            ts = _parse_timestamp(ts_raw)
        except (ValueError, TypeError):
            ts = datetime.now(timezone.utc)

        lots = lots_by_symbol[symbol]

        # Determine if this event closes existing lots
        # buy closes sell lots; sell closes buy lots
        opposite = "sell" if action == "buy" else "buy"
        remaining = qty

        while remaining > 1e-12 and lots:
            lot_qty, lot_price, lot_action = lots[0]
            if lot_action != opposite:
                break
            closable = min(remaining, lot_qty)
            if action == "sell":
                pnl = (price - lot_price) * closable
                entry_side = "long"
            else:
                pnl = (lot_price - price) * closable
                entry_side = "short"

            trades.append({
                "symbol": symbol,
                "close_date": ts.date().isoformat(),
                "close_ts": ts.isoformat(),
                "pnl": pnl,
                "entry_price": lot_price,
                "exit_price": price,
                "qty": closable,
                "side": entry_side,
            })

            lot_qty -= closable
            remaining -= closable
            if lot_qty < 1e-12:
                lots.pop(0)
            else:
                lots[0] = (lot_qty, lot_price, lot_action)

        # Remaining qty becomes a new lot
        if remaining > 1e-12:
            lots.append((remaining, price, action))

    return trades


# ---------------------------------------------------------------------------
# Daily aggregation
# ---------------------------------------------------------------------------

def aggregate_daily(trades: list[dict]) -> list[dict]:
    """Aggregate closed trades into daily PnL buckets.

    Returns sorted list of dicts with: date, daily_pnl, num_trades, wins, losses.
    """
    by_day: dict[str, dict] = {}
    for t in trades:
        d = t["close_date"]
        if d not in by_day:
            by_day[d] = {"date": d, "daily_pnl": 0.0, "num_trades": 0, "wins": 0, "losses": 0}
        by_day[d]["daily_pnl"] += t["pnl"]
        by_day[d]["num_trades"] += 1
        if t["pnl"] > 0:
            by_day[d]["wins"] += 1
        elif t["pnl"] < 0:
            by_day[d]["losses"] += 1

    return sorted(by_day.values(), key=lambda r: r["date"])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(daily_rows: list[dict],
                    starting_capital: float = 0.0) -> dict:
    """Compute summary metrics from daily PnL rows.

    Args:
        daily_rows: output of aggregate_daily()
        starting_capital: portfolio value at start of tracking period.
            When positive, percentage-based metrics (annualized return,
            Sortino, max drawdown) are computed relative to this capital.
            When zero the script infers equity from cumulative PnL only.

    Returns dict with: cumulative_pnl, annualized_return, sortino,
    max_drawdown, win_rate, trades_per_day, num_days, num_trades.
    """
    if not daily_rows:
        return {
            "cumulative_pnl": 0.0,
            "annualized_return": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "trades_per_day": 0.0,
            "num_days": 0,
            "num_trades": 0,
        }

    daily_pnls = np.array([r["daily_pnl"] for r in daily_rows], dtype=np.float64)
    cumulative_pnl = float(daily_pnls.sum())
    num_days = len(daily_rows)

    # Win rate across all individual trades
    total_trades = sum(r["num_trades"] for r in daily_rows)
    total_wins = sum(r["wins"] for r in daily_rows)
    win_rate = total_wins / total_trades if total_trades > 0 else 0.0

    trades_per_day = total_trades / num_days if num_days > 0 else 0.0

    # Build equity curve.  With starting_capital the curve represents
    # actual portfolio value; without it we use cumulative PnL from 0.
    base = max(starting_capital, 0.0)
    equity = np.concatenate([[base], base + np.cumsum(daily_pnls)])

    # Max drawdown (percentage of peak)
    running_peak = np.maximum.accumulate(equity)
    if running_peak.max() > 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            drawdowns = np.where(running_peak > 0, (equity - running_peak) / running_peak, 0.0)
        drawdowns = np.where(np.isfinite(drawdowns), drawdowns, 0.0)
        max_dd = float(drawdowns.min())
    else:
        max_dd = float(equity.min()) if equity.min() < 0 else 0.0

    # Percentage returns: daily_pnl / start-of-day equity
    ann_return = 0.0
    sortino = 0.0
    if num_days >= 2:
        pct_returns = []
        for i in range(len(daily_pnls)):
            day_base = equity[i]  # equity at start of day i
            if abs(day_base) > 1e-8:
                pct_returns.append(float(daily_pnls[i] / abs(day_base)))
            else:
                pct_returns.append(0.0)
        pct_arr = np.array(pct_returns, dtype=np.float64)

        if pct_arr.size > 0 and np.any(np.isfinite(pct_arr)):
            mean_ret = float(np.nanmean(pct_arr))
            ann_return = mean_ret * 252

            downside = pct_arr[pct_arr < 0]
            if downside.size >= 1:
                if downside.size == 1:
                    dd_std = float(np.abs(downside[0]))
                else:
                    dd_std = float(np.std(downside, ddof=1))
                if dd_std > 0:
                    sortino = float((mean_ret * 252) / (dd_std * np.sqrt(252)))
            elif downside.size == 0 and mean_ret > 0:
                # No losing days
                std_all = float(np.std(pct_arr, ddof=1)) if pct_arr.size > 1 else 0.0
                if std_all > 0:
                    sortino = float((mean_ret * 252) / (std_all * np.sqrt(252)))

    return {
        "cumulative_pnl": round(cumulative_pnl, 2),
        "annualized_return": round(ann_return, 4),
        "sortino": round(sortino, 4),
        "max_drawdown": round(max_dd, 4),
        "win_rate": round(win_rate, 4),
        "trades_per_day": round(trades_per_day, 2),
        "num_days": num_days,
        "num_trades": total_trades,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute production PnL metrics from fill events."
    )
    parser.add_argument(
        "--since",
        type=str,
        default=None,
        help="Only include trades on or after this date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint name (logged in output for tracking).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write daily PnL rows to this CSV path.",
    )
    parser.add_argument(
        "--fills-file",
        type=str,
        default=None,
        help="Override path to fill_events.jsonl.",
    )
    parser.add_argument(
        "--state-file",
        type=str,
        default=None,
        help="Override path to unified_state.json.",
    )
    parser.add_argument(
        "--starting-capital",
        type=float,
        default=None,
        help="Starting capital for percentage-based metrics.  Defaults to total_value from unified_state.json.",
    )
    return parser.parse_args(argv)


def _write_csv(output_path: Optional[str], rows: list[dict]) -> None:
    """Write daily rows to CSV if output_path is set."""
    if output_path is None:
        return
    with open(output_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in CSV_FIELDNAMES})
    label = f"{len(rows)} rows" if rows else "empty CSV"
    print(f"Wrote {label} to {output_path}")


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)

    fills_path = Path(args.fills_file) if args.fills_file else FILL_EVENTS_FILE
    state_path = Path(args.state_file) if args.state_file else UNIFIED_STATE_FILE

    since: Optional[date] = None
    if args.since:
        try:
            since = date.fromisoformat(args.since)
        except ValueError:
            print(f"ERROR: invalid --since date: {args.since!r}, expected YYYY-MM-DD", file=sys.stderr)
            sys.exit(1)

    # --- Load data ----------------------------------------------------------
    events = load_fill_events(fills_path)
    state = load_unified_state(state_path)

    if not events:
        print(f"No fill events found in {fills_path}")
        if state:
            ts = state.get("timestamp", "unknown")
            total = state.get("total_value", 0.0)
            print(f"Portfolio snapshot ({ts}): total_value=${total:,.2f}")
        _write_csv(args.output, [])
        return

    # --- Compute trades and daily PnL ---------------------------------------
    trades = compute_closed_trades(events)
    if since:
        since_str = since.isoformat()
        trades = [t for t in trades if t["close_date"] >= since_str]

    daily_rows = aggregate_daily(trades)

    # Determine starting capital for percentage metrics
    starting_capital = 0.0
    if args.starting_capital is not None:
        starting_capital = args.starting_capital
    elif state:
        starting_capital = float(state.get("total_value", 0.0))

    metrics = compute_metrics(daily_rows, starting_capital=starting_capital)

    # Add cumulative column to daily rows
    cum = 0.0
    for row in daily_rows:
        cum += row["daily_pnl"]
        row["cumulative_pnl"] = round(cum, 2)

    # --- Print summary ------------------------------------------------------
    header = "Deployment PnL Summary"
    if args.checkpoint:
        header += f"  (checkpoint: {args.checkpoint})"
    print(header)
    print("=" * len(header))

    if state:
        ts = state.get("timestamp", "unknown")
        total = state.get("total_value", 0.0)
        print(f"Portfolio snapshot: {ts}  total_value=${total:,.2f}")

    if not daily_rows:
        since_msg = f" since {since.isoformat()}" if since else ""
        print(f"No closed trades found{since_msg}.")
        _write_csv(args.output, [])
        return

    print(f"Period:            {daily_rows[0]['date']} to {daily_rows[-1]['date']}  ({metrics['num_days']} trading days)")
    print(f"Total trades:      {metrics['num_trades']}")
    print(f"Trades/day:        {metrics['trades_per_day']:.2f}")
    print(f"Win rate:          {metrics['win_rate']:.2%}")
    print(f"Cumulative PnL:    ${metrics['cumulative_pnl']:,.2f}")
    print(f"Annualized return: {metrics['annualized_return']:.2%}")
    print(f"Sortino ratio:     {metrics['sortino']:.4f}")
    print(f"Max drawdown:      {metrics['max_drawdown']:.2%}")

    print()
    print("Daily PnL:")
    for row in daily_rows:
        sign = "+" if row["daily_pnl"] >= 0 else ""
        print(f"  {row['date']}  {sign}${row['daily_pnl']:,.2f}  (cum: ${row['cumulative_pnl']:,.2f})  trades={row['num_trades']}")

    _write_csv(args.output, daily_rows)


if __name__ == "__main__":
    main()
