#!/usr/bin/env python3
"""Offline DeepSeek agent benchmarks on cached OHLC data."""

from __future__ import annotations

import argparse
import json
import sys
import types
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Stub Alpaca dependencies so benchmarks do not touch live accounts.
if "alpaca_wrapper" not in sys.modules:
    alpaca_stub = types.ModuleType("alpaca_wrapper")

    @dataclass
    class _StubAccount:
        equity: float = 0.0
        cash: float = 0.0
        buying_power: float = 0.0

    alpaca_stub.get_account = lambda: _StubAccount()
    alpaca_stub.get_all_positions = lambda: []
    alpaca_stub.open_order_at_price_or_all = lambda *args, **kwargs: None
    sys.modules["alpaca_wrapper"] = alpaca_stub

from stockagent.agentsimulator.data_models import AccountPosition, AccountSnapshot, TradingPlanEnvelope
from stockagent.agentsimulator.market_data import MarketDataBundle
from stockagent.agentsimulator.risk_strategies import ProbeTradeStrategy, ProfitShutdownStrategy
from stockagent.agentsimulator.simulator import AgentSimulator
from stockagentcombined_entrytakeprofit.simulator import EntryTakeProfitSimulator
from stockagentdeepseek_maxdiff.simulator import MaxDiffSimulator


def _load_bundle(symbol: str, csv_path: Path, lookback: int) -> MarketDataBundle:
    frame = pd.read_csv(csv_path, parse_dates=["timestamp"]).set_index("timestamp").tz_localize("UTC")
    if lookback and len(frame) > lookback:
        frame = frame.tail(lookback)
    ohlc = frame[["open", "high", "low", "close"]].copy()
    return MarketDataBundle(
        bars={symbol.upper(): ohlc},
        lookback_days=len(ohlc),
        as_of=ohlc.index[-1].to_pydatetime(),
    )


def _baseline_snapshot() -> AccountSnapshot:
    return AccountSnapshot(
        equity=20_000.0,
        cash=20_000.0,
        buying_power=30_000.0,
        timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
        positions=[],
    )


def _daily_plan(
    *,
    symbol: str,
    timestamp: pd.Timestamp,
    quantity: float,
    entry_price: float,
    exit_price: float,
    note: str,
) -> Dict:
    iso_date = timestamp.date().isoformat()
    return {
        "target_date": iso_date,
        "instructions": [
            {
                "symbol": symbol.upper(),
                "action": "buy",
                "quantity": quantity,
                "execution_session": "market_open",
                "entry_price": entry_price,
                "exit_price": exit_price,
                "exit_reason": "planned exit",
                "notes": note,
            },
            {
                "symbol": symbol.upper(),
                "action": "sell",
                "quantity": quantity,
                "execution_session": "market_close",
                "entry_price": entry_price,
                "exit_price": exit_price,
                "exit_reason": "session close exit",
                "notes": "flatten holdings",
            },
        ],
        "metadata": {"capital_allocation_plan": f"{quantity} units on {iso_date}"},
    }


def _simulate_agent(plan_payload: Dict, *, bundle: MarketDataBundle, snapshot: AccountSnapshot):
    plan = TradingPlanEnvelope.from_json(json.dumps(plan_payload)).plan
    simulator = AgentSimulator(market_data=bundle, account_snapshot=snapshot, starting_cash=snapshot.cash)
    result = simulator.simulate([plan], strategies=[ProbeTradeStrategy(), ProfitShutdownStrategy()])
    return result


def _simulate_entry_takeprofit(plan_payload: Dict, *, bundle: MarketDataBundle):
    plan = TradingPlanEnvelope.from_json(json.dumps(plan_payload)).plan
    simulator = EntryTakeProfitSimulator(market_data=bundle)
    return simulator.run([plan])


def _simulate_maxdiff(plan_payload: Dict, *, bundle: MarketDataBundle):
    plan = TradingPlanEnvelope.from_json(json.dumps(plan_payload)).plan
    simulator = MaxDiffSimulator(market_data=bundle)
    return simulator.run([plan])


def _next_snapshot(sim_result, timestamp: pd.Timestamp) -> AccountSnapshot:
    positions: list[AccountPosition] = []
    for symbol, payload in sim_result.final_positions.items():
        qty = float(payload.get("quantity", 0.0) or 0.0)
        if qty == 0:
            continue
        avg_price = float(payload.get("avg_price", 0.0) or 0.0)
        side = "long" if qty >= 0 else "short"
        positions.append(
            AccountPosition(
                symbol=symbol.upper(),
                quantity=qty,
                side=side,
                market_value=qty * avg_price,
                avg_entry_price=avg_price,
                unrealized_pl=0.0,
                unrealized_plpc=0.0,
            )
        )
    return AccountSnapshot(
        equity=sim_result.ending_equity,
        cash=sim_result.ending_cash,
        buying_power=sim_result.ending_equity,
        timestamp=datetime.fromtimestamp(timestamp.timestamp(), tz=timezone.utc),
        positions=positions,
    )


def benchmark(symbol: str, csv_path: Path, lookback: int) -> Dict[str, Dict[str, float]]:
    bundle = _load_bundle(symbol, csv_path, lookback)
    history = bundle.get_symbol_bars(symbol)
    if len(history) < 2:
        raise ValueError("Need at least two trading days for benchmarking.")
    day1, day2 = list(history.index[-2:])
    snapshot = _baseline_snapshot()
    row1 = history.loc[day1]

    baseline_payload = _daily_plan(
        symbol=symbol,
        timestamp=day1,
        quantity=8,
        entry_price=float(row1["open"]),
        exit_price=float(row1["close"]),
        note="baseline close-out",
    )
    neural_payload = _daily_plan(
        symbol=symbol,
        timestamp=day1,
        quantity=5,
        entry_price=float(row1["open"]),
        exit_price=float(row1["close"]) * 1.01,
        note="neural bias (slightly extended exit)",
    )

    entry_payload = {
        "target_date": day1.date().isoformat(),
        "instructions": [
            {
                "symbol": symbol.upper(),
                "action": "buy",
                "quantity": 6,
                "execution_session": "market_open",
                "entry_price": float(row1["open"]),
                "exit_price": float(row1["high"]),
                "exit_reason": "take profit hit",
                "notes": "entry/take-profit benchmark",
            },
            {
                "symbol": symbol.upper(),
                "action": "exit",
                "quantity": 6,
                "execution_session": "market_close",
                "entry_price": None,
                "exit_price": float(row1["high"]),
                "exit_reason": "target reached intraday",
                "notes": "flatten to cash",
            },
        ],
        "metadata": {"capital_allocation_plan": "Take-profit toward intraday highs"},
    }

    maxdiff_entry_price = float(row1["low"] + 0.3 * (row1["high"] - row1["low"]))
    maxdiff_payload = {
        "target_date": day1.date().isoformat(),
        "instructions": [
            {
                "symbol": symbol.upper(),
                "action": "buy",
                "quantity": 5,
                "execution_session": "market_open",
                "entry_price": maxdiff_entry_price,
                "exit_price": float(row1["high"]),
                "exit_reason": "maxdiff profit target",
                "notes": "enter on intraday retrace",
            },
            {
                "symbol": symbol.upper(),
                "action": "exit",
                "quantity": 5,
                "execution_session": "market_close",
                "entry_price": None,
                "exit_price": float(row1["high"]),
                "exit_reason": "target reached",
                "notes": "close to realize swing",
            },
        ],
        "metadata": {"capital_allocation_plan": "MaxDiff intraday swing"},
    }

    baseline = _simulate_agent(baseline_payload, bundle=bundle, snapshot=snapshot)
    neural = _simulate_agent(neural_payload, bundle=bundle, snapshot=snapshot)
    entry = _simulate_entry_takeprofit(entry_payload, bundle=bundle)
    maxdiff = _simulate_maxdiff(maxdiff_payload, bundle=bundle)

    # Sequential replanning across the last two days.
    replan_snapshot = snapshot
    replan_equity = snapshot.cash
    for ts, qty in zip([day1, day2], [8, 6]):
        payload = _daily_plan(
            symbol=symbol,
            timestamp=ts,
            quantity=qty,
            entry_price=float(history.loc[ts]["open"]),
            exit_price=float(history.loc[ts]["close"]),
            note="replanning benchmark",
        )
        sim_result = _simulate_agent(payload, bundle=bundle, snapshot=replan_snapshot)
        replan_equity = sim_result.ending_equity
        replan_snapshot = _next_snapshot(sim_result, ts)
    replan_total = (replan_equity - snapshot.cash) / snapshot.cash
    replan_annualized = (replan_equity / snapshot.cash) ** (252 / 2) - 1

    return {
        "baseline": {
            "target_date": day1.date().isoformat(),
            "realized_pnl": baseline.realized_pnl,
            "fees": baseline.total_fees,
            "net_pnl": baseline.realized_pnl - baseline.total_fees,
        },
        "neural": {
            "target_date": day1.date().isoformat(),
            "realized_pnl": neural.realized_pnl,
            "fees": neural.total_fees,
            "net_pnl": neural.realized_pnl - neural.total_fees,
        },
        "entry_takeprofit": {
            "target_date": day1.date().isoformat(),
            "realized_pnl": entry.realized_pnl,
            "fees": entry.total_fees,
            "net_pnl": entry.net_pnl,
        },
        "maxdiff": {
            "target_date": day1.date().isoformat(),
            "realized_pnl": maxdiff.realized_pnl,
            "fees": maxdiff.total_fees,
            "net_pnl": maxdiff.net_pnl,
        },
        "replan": {
            "start_date": day1.date().isoformat(),
            "end_date": day2.date().isoformat(),
            "total_return_pct": replan_total,
            "annualized_return_pct": replan_annualized,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="AAPL", help="Ticker to benchmark (default: %(default)s)")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("trainingdata/AAPL.csv"),
        help="Path to cached OHLC CSV (default: %(default)s)",
    )
    parser.add_argument("--lookback", type=int, default=30, help="Days of history to include (default: %(default)s)")
    parser.add_argument("--format", choices={"json", "table"}, default="table", help="Output format")
    args = parser.parse_args(argv)

    metrics = benchmark(args.symbol, args.csv, args.lookback)

    if args.format == "json":
        print(json.dumps(metrics, indent=2))
        return 0

    def _fmt_money(value: float) -> str:
        return f"{value:>8.2f}"

    def _fmt_pct(value: float) -> str:
        return f"{value * 100:>7.2f}%"

    print(f"DeepSeek agent benchmark ({args.symbol.upper()} | source={args.csv})\n")
    header = "Scenario        Realized   Fees     Net PnL"
    divider = "--------------  --------  -------  ---------"
    print(header)
    print(divider)
    for key in ("baseline", "neural", "entry_takeprofit", "maxdiff"):
        payload = metrics[key]
        print(
            f"{key:14}  {_fmt_money(payload['realized_pnl'])}  "
            f"{_fmt_money(payload['fees'])}  {_fmt_money(payload['net_pnl'])}"
        )
    replan = metrics["replan"]
    print(divider)
    print(f"replan {replan['start_date']}→{replan['end_date']}")
    print(
        f"  total return {_fmt_pct(replan['total_return_pct'])}, "
        f"annualized {_fmt_pct(replan['annualized_return_pct'])}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
