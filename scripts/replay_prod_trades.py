#!/usr/bin/env python3
"""Replay production trades through a simple portfolio simulator to compare actual vs simulated P&L."""

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta

LOG_FILE = "/var/log/supervisor/binance-hybrid-spot-error.log"
FEE_BPS = 10
MARGIN_ANNUAL_RATE = 0.0625
HOURS_PER_YEAR = 8760

RE_TIMESTAMP = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})")

RE_MARGIN_BUY = re.compile(
    r"MARGIN BUY (\w+USDT): qty=([\d.]+), price=([\d.]+), notional=([\d.]+)"
)
RE_MARGIN_SELL = re.compile(
    r"MARGIN SELL (\w+USDT): qty=([\d.]+), price=([\d.]+), notional=([\d.]+)"
)

# "Portfolio: $3058.69 | Cash: $2109.05 | borrowable_usdt=11474.92"
RE_PORTFOLIO_NEW = re.compile(
    r"Portfolio: \$([\d.]+) \| Cash: \$([\d.]+)"
)
# "Portfolio: FDUSD=0.00, USDT=3311.39, borrowable_usdt=13257.81, total=3315.85"
RE_PORTFOLIO_OLD = re.compile(
    r"Portfolio:.*total=([\d.]+)"
)

RE_FORCED_EXIT = re.compile(
    r"FORCED EXIT (\w+USD): held ([\d.]+)h"
)


@dataclass
class Trade:
    ts: datetime
    symbol: str
    side: str  # BUY or SELL
    qty: float
    price: float
    notional: float
    is_forced_exit: bool = False


@dataclass
class PortfolioSnapshot:
    ts: datetime
    total: float
    cash: float = 0.0


@dataclass
class Position:
    qty: float
    avg_price: float
    entry_time: datetime
    notional: float = 0.0


@dataclass
class SimState:
    cash: float = 0.0
    positions: dict = field(default_factory=dict)
    total_fees: float = 0.0
    total_margin_interest: float = 0.0
    trade_pnls: list = field(default_factory=list)
    per_symbol_pnl: dict = field(default_factory=lambda: defaultdict(float))


def parse_ts(line):
    m = RE_TIMESTAMP.match(line)
    if m:
        return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f")
    return None


def parse_log(log_file, cutoff_ts):
    trades = []
    snapshots = []
    pending_forced = {}

    with open(log_file, "r") as f:
        for line in f:
            ts = parse_ts(line)
            if ts is None or ts < cutoff_ts:
                continue

            m = RE_FORCED_EXIT.search(line)
            if m:
                sym_short = m.group(1)
                sym_usdt = sym_short.replace("USD", "USDT")
                pending_forced[sym_usdt] = True

            m = RE_MARGIN_BUY.search(line)
            if m:
                sym = m.group(1)
                trades.append(Trade(
                    ts=ts, symbol=sym, side="BUY",
                    qty=float(m.group(2)), price=float(m.group(3)),
                    notional=float(m.group(4)),
                ))
                continue

            m = RE_MARGIN_SELL.search(line)
            if m:
                sym = m.group(1)
                is_forced = sym in pending_forced
                if is_forced:
                    del pending_forced[sym]
                trades.append(Trade(
                    ts=ts, symbol=sym, side="SELL",
                    qty=float(m.group(2)), price=float(m.group(3)),
                    notional=float(m.group(4)),
                    is_forced_exit=is_forced,
                ))
                continue

            m = RE_PORTFOLIO_NEW.search(line)
            if m:
                snapshots.append(PortfolioSnapshot(
                    ts=ts, total=float(m.group(1)), cash=float(m.group(2))
                ))
                continue

            m = RE_PORTFOLIO_OLD.search(line)
            if m:
                snapshots.append(PortfolioSnapshot(
                    ts=ts, total=float(m.group(1))
                ))

    return trades, snapshots


def _apply_buy(positions, t):
    """Update positions dict for a BUY trade. Returns the new/updated Position."""
    pos = positions.get(t.symbol)
    if pos:
        total_qty = pos.qty + t.qty
        pos.avg_price = (pos.qty * pos.avg_price + t.qty * t.price) / total_qty
        pos.qty = total_qty
        pos.notional += t.notional
    else:
        pos = Position(qty=t.qty, avg_price=t.price, entry_time=t.ts, notional=t.notional)
        positions[t.symbol] = pos
    return pos


def _apply_sell(positions, t):
    """Close/reduce position for a SELL trade. Returns (sell_qty, pnl, hours_held, margin_cost, entry_price) or None."""
    pos = positions.get(t.symbol)
    if not pos:
        return None
    sell_qty = min(t.qty, pos.qty)
    entry_price = pos.avg_price
    pnl = sell_qty * (t.price - entry_price)
    hours_held = (t.ts - pos.entry_time).total_seconds() / 3600.0
    margin_cost = pos.notional * MARGIN_ANNUAL_RATE * hours_held / HOURS_PER_YEAR
    remaining = pos.qty - sell_qty
    if remaining < 1e-8:
        del positions[t.symbol]
    else:
        pos.qty = remaining
        pos.notional = remaining * entry_price
    return sell_qty, pnl, hours_held, margin_cost, entry_price


def simulate(trades, initial_equity):
    state = SimState(cash=initial_equity)

    for t in trades:
        fee = t.notional * FEE_BPS / 10000.0
        state.total_fees += fee

        if t.side == "BUY":
            _apply_buy(state.positions, t)
            state.cash -= (t.notional + fee)

        elif t.side == "SELL":
            result = _apply_sell(state.positions, t)
            if result:
                sell_qty, pnl, hours_held, margin_cost, entry_price = result
                state.total_margin_interest += margin_cost
                net_pnl = pnl - fee - margin_cost
                state.per_symbol_pnl[t.symbol] += net_pnl
                state.trade_pnls.append({
                    "ts": t.ts.isoformat(),
                    "symbol": t.symbol,
                    "side": "SELL",
                    "qty": sell_qty,
                    "entry_price": entry_price,
                    "exit_price": t.price,
                    "gross_pnl": round(pnl, 2),
                    "fee": round(fee, 2),
                    "margin_interest": round(margin_cost, 2),
                    "net_pnl": round(net_pnl, 2),
                    "hours_held": round(hours_held, 1),
                    "forced_exit": t.is_forced_exit,
                })
                state.cash += (t.notional - fee)
            else:
                state.cash += (t.notional - fee)
                state.per_symbol_pnl[t.symbol] -= fee

    return state


def build_comparison(snapshots, trades, initial_equity):
    """Track cumulative realized P&L against actual portfolio snapshots.

    Uses margin-aware accounting: sim_equity = initial_equity + realized_net_pnl - buy_fees
    """
    positions = {}
    realized_pnl = 0.0
    buy_fees = 0.0
    sim_curve = []

    events = []
    for s in snapshots:
        events.append(("snap", s.ts, s))
    for t in trades:
        events.append(("trade", t.ts, t))
    events.sort(key=lambda x: x[1])

    for etype, ets, obj in events:
        if etype == "trade":
            t = obj
            fee = t.notional * FEE_BPS / 10000.0
            if t.side == "BUY":
                _apply_buy(positions, t)
                buy_fees += fee
            elif t.side == "SELL":
                result = _apply_sell(positions, t)
                if result:
                    _, pnl, _, margin_cost, _ = result
                    realized_pnl += pnl - fee - margin_cost
                else:
                    realized_pnl -= fee

        elif etype == "snap":
            s = obj
            sim_total = initial_equity + realized_pnl - buy_fees
            sim_curve.append({
                "ts": s.ts.isoformat(),
                "actual": s.total,
                "simulated": round(sim_total, 2),
                "diff": round(s.total - sim_total, 2),
                "open_positions": {sym: round(p.qty * p.avg_price, 2) for sym, p in positions.items()},
            })

    return sim_curve


def print_report(trades, snapshots, sim_state, sim_curve, initial_equity):
    print("=" * 80)
    print("PROD TRADE REPLAY REPORT")
    print("=" * 80)

    if snapshots:
        print(f"\nPeriod: {snapshots[0].ts.strftime('%Y-%m-%d %H:%M')} -> {snapshots[-1].ts.strftime('%Y-%m-%d %H:%M')}")

    print(f"Total trades parsed: {len(trades)}")
    buys = [t for t in trades if t.side == "BUY"]
    sells = [t for t in trades if t.side == "SELL"]
    forced = [t for t in trades if t.is_forced_exit]
    print(f"  Buys: {len(buys)}, Sells: {len(sells)}, Forced exits: {len(forced)}")

    print(f"\nInitial equity (first snapshot): ${initial_equity:.2f}")
    if snapshots:
        print(f"Final actual equity: ${snapshots[-1].total:.2f}")
        actual_ret = (snapshots[-1].total - initial_equity) / initial_equity * 100
        print(f"Actual return: {actual_ret:+.2f}%")

    print(f"\n--- Simulated P&L ---")
    print(f"Total fees: ${sim_state.total_fees:.2f}")
    print(f"Total margin interest: ${sim_state.total_margin_interest:.2f}")

    total_net = sum(sim_state.per_symbol_pnl.values())
    sim_ret = total_net / initial_equity * 100
    print(f"Net P&L: ${total_net:.2f} ({sim_ret:+.2f}%)")

    print(f"\n--- Per-Symbol P&L ---")
    print(f"{'Symbol':<12} {'Net P&L':>10} {'Trades':>8}")
    print("-" * 32)
    sym_trades = defaultdict(int)
    for tp in sim_state.trade_pnls:
        sym_trades[tp["symbol"]] += 1
    for sym in sorted(sim_state.per_symbol_pnl.keys()):
        pnl = sim_state.per_symbol_pnl[sym]
        print(f"{sym:<12} ${pnl:>9.2f} {sym_trades.get(sym, 0):>8}")

    print(f"\n--- Trade Details ---")
    print(f"{'Timestamp':<20} {'Symbol':<12} {'Entry':>8} {'Exit':>8} {'Qty':>10} {'Gross':>8} {'Net':>8} {'Hrs':>5} {'Forced'}")
    print("-" * 100)
    for tp in sim_state.trade_pnls:
        forced_str = "YES" if tp["forced_exit"] else ""
        print(f"{tp['ts'][:19]:<20} {tp['symbol']:<12} {tp['entry_price']:>8.2f} {tp['exit_price']:>8.2f} {tp['qty']:>10.4f} {tp['gross_pnl']:>8.2f} {tp['net_pnl']:>8.2f} {tp['hours_held']:>5.1f} {forced_str}")

    if sim_curve:
        print(f"\n--- Equity Curve (sampled) ---")
        print(f"{'Timestamp':<20} {'Actual':>10} {'Simulated':>10} {'Diff':>10}")
        print("-" * 55)
        step = max(1, len(sim_curve) // 20)
        for i in range(0, len(sim_curve), step):
            c = sim_curve[i]
            print(f"{c['ts'][:19]:<20} ${c['actual']:>9.2f} ${c['simulated']:>9.2f} ${c['diff']:>9.2f}")
        if len(sim_curve) % step != 1:
            c = sim_curve[-1]
            print(f"{c['ts'][:19]:<20} ${c['actual']:>9.2f} ${c['simulated']:>9.2f} ${c['diff']:>9.2f}")


def main():
    parser = argparse.ArgumentParser(description="Replay prod trades through simulator")
    parser.add_argument("--log-file", default=LOG_FILE)
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--output", type=str, default=None, help="Write JSON report to file")
    args = parser.parse_args()

    cutoff = datetime.now() - timedelta(days=args.days)
    try:
        trades, snapshots = parse_log(args.log_file, cutoff)
    except FileNotFoundError:
        print(f"Log file not found: {args.log_file}")
        sys.exit(1)

    if not trades:
        print("No trades found in the specified time range.")
        sys.exit(0)

    initial_equity = snapshots[0].total if snapshots else 3000.0

    sim_state = simulate(trades, initial_equity)
    sim_curve = build_comparison(snapshots, trades, initial_equity)
    print_report(trades, snapshots, sim_state, sim_curve, initial_equity)

    if args.output:
        report = {
            "period_start": cutoff.isoformat(),
            "period_end": datetime.now().isoformat(),
            "initial_equity": initial_equity,
            "final_actual": snapshots[-1].total if snapshots else None,
            "total_trades": len(trades),
            "total_fees": round(sim_state.total_fees, 2),
            "total_margin_interest": round(sim_state.total_margin_interest, 2),
            "net_pnl": round(sum(sim_state.per_symbol_pnl.values()), 2),
            "per_symbol_pnl": {k: round(v, 2) for k, v in sim_state.per_symbol_pnl.items()},
            "trade_details": sim_state.trade_pnls,
            "equity_curve": sim_curve,
        }
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nJSON report written to {args.output}")


if __name__ == "__main__":
    main()
