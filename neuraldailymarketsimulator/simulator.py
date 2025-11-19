from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from neuraldailytraining import DailyTradingRuntime
from neuraldailytraining.config import DailyDatasetConfig
from neural_trade_stock_e2e import _build_dataset_config


@dataclass
class SimulationResult:
    date: pd.Timestamp
    equity: float
    cash: float
    daily_return: float
    leverage: float = 0.0
    leverage_cost: float = 0.0


class NeuralDailyMarketSimulator:
    def __init__(
        self,
        runtime: DailyTradingRuntime,
        symbols: Sequence[str],
        *,
        maker_fee: float = 0.0008,
        initial_cash: float = 1.0,
        leverage_fee_rate: float = 0.065,
        equity_max_leverage: float = 2.0,
        crypto_max_leverage: float = 1.0,
    ) -> None:
        self.runtime = runtime
        self.symbols = [symbol.upper() for symbol in symbols]
        self.maker_fee = maker_fee
        self.initial_cash = initial_cash
        self.leverage_fee_rate = leverage_fee_rate
        self.daily_leverage_rate = leverage_fee_rate / 365.0
        self.equity_max_leverage = equity_max_leverage
        self.crypto_max_leverage = crypto_max_leverage
        self.frames: Dict[str, pd.DataFrame] = {}
        self.crypto_symbols: set[str] = set()

        for symbol in self.symbols:
            frame = runtime._builder.build(symbol)  # type: ignore[attr-defined]
            if frame.empty:
                continue
            frame = frame.copy()
            frame["date"] = pd.to_datetime(frame["date"], utc=True)
            self.frames[symbol] = frame

            # Track which symbols are crypto (end with USD or -USD)
            if symbol.upper().endswith("USD") or symbol.upper().endswith("-USD"):
                self.crypto_symbols.add(symbol)

        if not self.frames:
            raise ValueError("No symbols have usable historical data for simulation.")

    def _available_dates(self) -> List[pd.Timestamp]:
        union: set[pd.Timestamp] = set()
        for frame in self.frames.values():
            union.update(frame["date"])
        ordered = sorted(union)
        return ordered

    def _select_dates(self, start_date: Optional[str], days: int) -> List[pd.Timestamp]:
        ordered = self._available_dates()
        if not ordered:
            raise ValueError("No historical dates available for simulation.")
        if start_date:
            cutoff = pd.to_datetime(start_date, utc=True)
            ordered = [date for date in ordered if date >= cutoff]
        if len(ordered) < days:
            raise ValueError(f"Requested {days} simulation days but only {len(ordered)} available.")
        return ordered[:days]

    def run(self, *, start_date: Optional[str] = None, days: int = 5) -> Tuple[List[SimulationResult], Dict[str, float]]:
        dates = self._select_dates(start_date, days)
        cash = self.initial_cash
        prev_value = cash
        inventory: Dict[str, float] = {symbol: 0.0 for symbol in self.symbols}
        last_close: Dict[str, float] = {symbol: float(frame["close"].iloc[-1]) for symbol, frame in self.frames.items()}
        results: List[SimulationResult] = []
        daily_returns: List[float] = []

        for date in dates:
            # Calculate leverage cost at start of day
            # Separate stock and crypto positions
            stock_value = sum(
                inventory.get(symbol, 0.0) * last_close.get(symbol, 0.0)
                for symbol in self.symbols if symbol not in self.crypto_symbols
            )
            crypto_value = sum(
                inventory.get(symbol, 0.0) * last_close.get(symbol, 0.0)
                for symbol in self.symbols if symbol in self.crypto_symbols
            )
            positions_value = stock_value + crypto_value
            current_equity = cash + positions_value

            # Calculate overall leverage for reporting
            leverage = positions_value / current_equity if current_equity > 0 else 0.0
            leverage_cost = 0.0

            # Only charge leverage fees on stocks (crypto can't leverage)
            # Stocks can leverage up to 2x, crypto max 1x
            if current_equity > 0 and stock_value > current_equity:
                # Stock positions are leveraged beyond 1x
                leveraged_amount = min(stock_value - current_equity, current_equity)  # Cap at 2x
                leverage_cost = leveraged_amount * self.daily_leverage_rate
                cash -= leverage_cost

            symbol_rows: Dict[str, pd.Series] = {}
            for symbol, frame in self.frames.items():
                rows = frame[frame["date"] == date]
                if not rows.empty:
                    symbol_rows[symbol] = rows.iloc[0]
            for symbol, row in symbol_rows.items():
                last_close[symbol] = float(row["close"])
                plan = self.runtime.plan_for_symbol(symbol, as_of=date)
                if plan is None:
                    continue
                buy_price = max(plan.buy_price, 1e-6)
                sell_price = max(plan.sell_price, 1e-6)
                target_amount = float(plan.trade_amount)
                high = float(row["high"])
                low = float(row["low"])
                # Buy leg
                max_affordable = cash / (buy_price * (1.0 + self.maker_fee))
                buy_qty = min(target_amount, max_affordable)
                if low <= buy_price and buy_qty > 0:
                    cash -= buy_qty * buy_price * (1.0 + self.maker_fee)
                    inventory[symbol] += buy_qty
                # Sell leg
                sellable = min(target_amount, inventory[symbol])
                if high >= sell_price and sellable > 0:
                    cash += sellable * sell_price * (1.0 - self.maker_fee)
                    inventory[symbol] -= sellable
            portfolio_value = cash
            for symbol, qty in inventory.items():
                close_price = last_close.get(symbol)
                if close_price is None:
                    continue
                portfolio_value += qty * close_price
            daily_return = 0.0
            if prev_value > 0:
                daily_return = (portfolio_value - prev_value) / prev_value
            results.append(
                SimulationResult(
                    date=date,
                    equity=float(portfolio_value),
                    cash=float(cash),
                    daily_return=float(daily_return),
                    leverage=float(leverage),
                    leverage_cost=float(leverage_cost),
                )
            )
            daily_returns.append(float(daily_return))
            prev_value = portfolio_value

        sortino = self._sortino_ratio(daily_returns)
        final_equity = results[-1].equity if results else self.initial_cash
        total_leverage_costs = sum(result.leverage_cost for result in results)
        max_leverage = max((result.leverage for result in results), default=0.0)
        summary = {
            "final_equity": final_equity,
            "pnl": final_equity - self.initial_cash,
            "sortino": sortino,
            "total_leverage_costs": total_leverage_costs,
            "max_leverage": max_leverage,
        }
        return results, summary

    @staticmethod
    def _sortino_ratio(returns: Sequence[float]) -> float:
        if not returns:
            return 0.0
        arr = np.asarray(returns, dtype=np.float64)
        downside = arr[arr < 0]
        downside_std = float(np.sqrt(np.mean(np.square(downside)))) if downside.size else 0.0
        mean_return = float(arr.mean())
        if downside_std == 0.0:
            return float("inf") if mean_return > 0 else 0.0
        return mean_return / downside_std


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate neural daily trading over recent days.")
    parser.add_argument("--checkpoint", required=True, help="Path to the neuraldaily checkpoint to evaluate.")
    parser.add_argument("--mode", choices=("plan", "simulate"), default="simulate")
    parser.add_argument("--symbols", nargs="*", help="Optional subset of symbols.")
    parser.add_argument("--data-root", default="trainingdata/train")
    parser.add_argument("--forecast-cache", default="strategytraining/forecast_cache")
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--validation-days", type=int, default=40)
    parser.add_argument("--device", default=None)
    parser.add_argument("--start-date", help="Optional ISO start date for the simulation window.")
    parser.add_argument("--days", type=int, default=5)
    parser.add_argument("--initial-cash", type=float, default=1.0)
    parser.add_argument("--maker-fee", type=float, default=0.0008)
    parser.add_argument("--risk-threshold", type=float, help="Optional override for the runtime risk threshold.")
    return parser.parse_args()


def run_cli_simulation() -> None:
    args = parse_args()
    dataset_cfg = _build_dataset_config(args)
    runtime = DailyTradingRuntime(
        args.checkpoint,
        dataset_config=dataset_cfg,
        device=args.device,
        risk_threshold=args.risk_threshold,
    )
    symbols = args.symbols or list(dataset_cfg.symbols)
    simulator = NeuralDailyMarketSimulator(
        runtime,
        symbols,
        maker_fee=args.maker_fee,
        initial_cash=args.initial_cash,
    )
    results, summary = simulator.run(start_date=args.start_date, days=args.days)
    print(f"{'Date':<15} {'Equity':>12} {'Cash':>12} {'Return':>10} {'Leverage':>10} {'LevCost':>10}")
    for entry in results:
        date_str = entry.date.strftime("%Y-%m-%d")
        print(f"{date_str:<15} {entry.equity:>12.4f} {entry.cash:>12.4f} {entry.daily_return:>10.4f} {entry.leverage:>10.2f}x {entry.leverage_cost:>10.6f}")
    print("\nSimulation Summary")
    print(f"Final Equity      : {summary['final_equity']:.4f}")
    print(f"Net PnL           : {summary['pnl']:.4f}")
    print(f"Sortino Ratio     : {summary['sortino']:.4f}")
    print(f"Max Leverage      : {summary['max_leverage']:.2f}x")
    print(f"Total Lev. Costs  : {summary['total_leverage_costs']:.6f}")


if __name__ == "__main__":
    run_cli_simulation()
