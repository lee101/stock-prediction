#!/usr/bin/env python3
"""Run a live DeepSeek simulation and print the resulting PnL summary."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from typing import Sequence

from loguru import logger

from stockagent.agentsimulator.data_models import AccountPosition, AccountSnapshot
from stockagent.agentsimulator.market_data import MarketDataBundle, fetch_latest_ohlc
from stockagentdeepseek.agent import simulate_deepseek_plan
from stockagentdeepseek_entrytakeprofit.agent import simulate_deepseek_entry_takeprofit_plan
from stockagentdeepseek_maxdiff.agent import simulate_deepseek_maxdiff_plan
from stockagentdeepseek_combinedmaxdiff.agent import simulate_deepseek_combined_maxdiff_plan
from stockagentdeepseek_neural.agent import simulate_deepseek_neural_plan

STRATEGIES = ("baseline", "entry_takeprofit", "maxdiff", "neural", "combined_maxdiff")


def _default_account_snapshot(equity: float, symbols: Sequence[str]) -> AccountSnapshot:
    timestamp = datetime.now(timezone.utc)
    positions = [
        AccountPosition(
            symbol=symbol.upper(),
            quantity=0.0,
            side="flat",
            market_value=0.0,
            avg_entry_price=0.0,
            unrealized_pl=0.0,
            unrealized_plpc=0.0,
        )
        for symbol in symbols
    ]
    return AccountSnapshot(
        equity=equity,
        cash=equity,
        buying_power=equity,
        timestamp=timestamp,
        positions=positions,
    )


def _target_dates(bundle: MarketDataBundle, days: int) -> list[datetime]:
    trading_days = bundle.trading_days()
    if not trading_days:
        raise ValueError("No trading days available in market data bundle.")
    selected = trading_days[-days:]
    return [ts.to_pydatetime().astimezone(timezone.utc).date() for ts in selected]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=["AAPL", "NVDA", "MSFT"], help="Symbols to include.")
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=90,
        help="Historical lookback window when fetching OHLC data.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=2,
        help="Number of most recent sessions to simulate.",
    )
    parser.add_argument(
        "--equity",
        type=float,
        default=50_000.0,
        help="Starting equity for the simulated account.",
    )
    parser.add_argument(
        "--strategy",
        choices=STRATEGIES,
        default="neural",
        help="DeepSeek strategy variant to run.",
    )
    parser.add_argument(
        "--include-history",
        action="store_true",
        help="Include full market history in the prompt instead of symbol summaries only.",
    )
    args = parser.parse_args()

    logger.info("Fetching latest OHLC data for symbols: %s", ", ".join(args.symbols))
    bundle = fetch_latest_ohlc(symbols=args.symbols, lookback_days=args.lookback_days)
    dates = _target_dates(bundle, args.days)
    logger.info("Simulating DeepSeek strategy '%s' over dates: %s", args.strategy, ", ".join(map(str, dates)))

    snapshot = _default_account_snapshot(args.equity, args.symbols)

    for target_date in dates:
        logger.info("Running simulation for %s", target_date.isoformat())
        if args.strategy == "entry_takeprofit":
            result = simulate_deepseek_entry_takeprofit_plan(
                market_data=bundle,
                account_snapshot=snapshot,
                target_date=target_date,
                include_market_history=args.include_history,
            )
            summary = result.simulation.summary(starting_nav=snapshot.equity, periods=1)
            plan_dict = result.plan.to_dict()
        elif args.strategy == "maxdiff":
            result = simulate_deepseek_maxdiff_plan(
                market_data=bundle,
                account_snapshot=snapshot,
                target_date=target_date,
                include_market_history=args.include_history,
            )
            summary = result.simulation.summary(starting_nav=snapshot.equity, periods=1)
            plan_dict = result.plan.to_dict()
        elif args.strategy == "neural":
            result = simulate_deepseek_neural_plan(
                market_data=bundle,
                account_snapshot=snapshot,
                target_date=target_date,
                include_market_history=args.include_history,
            )
            summary = {
                "realized_pnl": result.simulation.realized_pnl,
                "total_fees": result.simulation.total_fees,
                "ending_cash": result.simulation.ending_cash,
                "ending_equity": result.simulation.ending_equity,
            }
            plan_dict = result.plan.to_dict()
        elif args.strategy == "combined_maxdiff":
            combined = simulate_deepseek_combined_maxdiff_plan(
                market_data=bundle,
                account_snapshot=snapshot,
                target_date=target_date,
                include_market_history=args.include_history,
            )
            summary = dict(combined.summary)
            summary.update({f"calibration_{k}": v for k, v in combined.calibration.items()})
            plan_dict = combined.plan.to_dict()
        else:
            result = simulate_deepseek_plan(
                market_data=bundle,
                account_snapshot=snapshot,
                target_date=target_date,
                include_market_history=args.include_history,
            )
            summary = {
                "realized_pnl": result.simulation.realized_pnl,
                "total_fees": result.simulation.total_fees,
                "ending_cash": result.simulation.ending_cash,
                "ending_equity": result.simulation.ending_equity,
            }
            plan_dict = result.plan.to_dict()

        print(json.dumps({"date": target_date.isoformat(), "plan": plan_dict, "summary": summary}, indent=2))

    logger.info("Simulation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
