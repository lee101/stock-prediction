"""V2 Market Simulator using unified simulation.

This simulator uses the same simulation logic as training (with temperature=0)
to ensure consistent backtesting results.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from neuraldailyv2 import DailyDatasetConfigV2, DailyTradingRuntimeV2
from src.fixtures import all_crypto_symbols
from src.date_utils import is_nyse_open_on_date


@dataclass
class SimulationResult:
    """Daily simulation result."""

    date: pd.Timestamp
    equity: float
    cash: float
    daily_return: float
    leverage: float = 0.0
    leverage_cost: float = 0.0


class NeuralDailyMarketSimulatorV2:
    """
    V2 Market Simulator using unified simulation.

    Key V2 features:
    - Uses same simulation logic as training (temperature=0)
    - Consistent position sizing interpretation
    - Matches production execution exactly
    """

    def __init__(
        self,
        runtime: DailyTradingRuntimeV2,
        symbols: Sequence[str],
        *,
        stock_fee: float = 0.0008,
        crypto_fee: float = 0.0008,
        initial_cash: float = 1.0,
        leverage_fee_rate: float = 0.065,
        equity_max_leverage: float = 2.0,
        crypto_max_leverage: float = 1.0,
        stocks_closed: bool = False,
        auto_weekend_hold: bool = True,
    ) -> None:
        self.runtime = runtime
        self.symbols = [symbol.upper() for symbol in symbols]
        self.stock_fee = stock_fee
        self.crypto_fee = crypto_fee
        self.initial_cash = initial_cash
        self.leverage_fee_rate = leverage_fee_rate
        self.daily_leverage_rate = leverage_fee_rate / 365.0
        self.equity_max_leverage = equity_max_leverage
        self.crypto_max_leverage = crypto_max_leverage
        self.stocks_closed = stocks_closed
        self.auto_weekend_hold = auto_weekend_hold
        self.frames: Dict[str, pd.DataFrame] = {}
        self.crypto_symbols: Set[str] = set()

        # Load historical data for each symbol
        for symbol in self.symbols:
            frame = runtime._builder.build(symbol)
            if frame.empty:
                continue
            frame = frame.copy()
            frame["date"] = pd.to_datetime(frame["date"], utc=True)
            self.frames[symbol] = frame

            if symbol.upper() in all_crypto_symbols:
                self.crypto_symbols.add(symbol)

        if not self.frames:
            raise ValueError("No symbols have usable historical data for simulation.")

    def _available_dates(self) -> List[pd.Timestamp]:
        """Get all available dates across all symbols."""
        union: Set[pd.Timestamp] = set()
        for frame in self.frames.values():
            union.update(frame["date"])
        return sorted(union)

    def _select_dates(self, start_date: Optional[str], days: int) -> List[pd.Timestamp]:
        """Select simulation date range."""
        ordered = self._available_dates()
        if not ordered:
            raise ValueError("No historical dates available for simulation.")
        if start_date:
            cutoff = pd.to_datetime(start_date, utc=True)
            ordered = [date for date in ordered if date >= cutoff]
        if len(ordered) < days:
            raise ValueError(f"Requested {days} simulation days but only {len(ordered)} available.")
        return ordered[:days]

    def run(
        self,
        *,
        start_date: Optional[str] = None,
        days: int = 5,
    ) -> Tuple[List[SimulationResult], Dict[str, float]]:
        """
        Run market simulation.

        Args:
            start_date: Optional ISO start date
            days: Number of days to simulate

        Returns:
            (results, summary) tuple
        """
        dates = self._select_dates(start_date, days)
        cash = self.initial_cash
        prev_value = cash
        inventory: Dict[str, float] = dict.fromkeys(self.symbols, 0.0)
        last_close: Dict[str, float] = {
            symbol: float(frame["close"].iloc[-1])
            for symbol, frame in self.frames.items()
        }
        results: List[SimulationResult] = []
        daily_returns: List[float] = []

        for date in dates:
            # Calculate leverage cost at start of day
            stock_value = sum(
                inventory.get(symbol, 0.0) * last_close.get(symbol, 0.0)
                for symbol in self.symbols
                if symbol not in self.crypto_symbols
            )
            crypto_value = sum(
                inventory.get(symbol, 0.0) * last_close.get(symbol, 0.0)
                for symbol in self.symbols
                if symbol in self.crypto_symbols
            )
            positions_value = stock_value + crypto_value
            current_equity = cash + positions_value

            leverage = positions_value / current_equity if current_equity > 0 else 0.0
            leverage_cost = 0.0

            # Leverage fees only on stocks above 1x
            if current_equity > 0 and stock_value > current_equity:
                leveraged_amount = min(stock_value - current_equity, current_equity)
                leverage_cost = leveraged_amount * self.daily_leverage_rate
                cash -= leverage_cost

            # Get market data for this date
            symbol_rows: Dict[str, pd.Series] = {}
            for symbol, frame in self.frames.items():
                rows = frame[frame["date"] == date]
                if not rows.empty:
                    symbol_rows[symbol] = rows.iloc[0]

            # Determine non-tradable symbols for this day
            non_tradable_today: Set[str] = set(self.runtime.non_tradable)
            if self.stocks_closed:
                non_tradable_today.update(
                    sym for sym in self.symbols if sym not in self.crypto_symbols
                )
            if self.auto_weekend_hold and not is_nyse_open_on_date(date):
                non_tradable_today.update(
                    sym for sym in self.symbols if sym not in self.crypto_symbols
                )

            # Get trading plans from model
            plans = self.runtime.plan_batch(
                list(symbol_rows.keys()),
                as_of=date,
                non_tradable_override=non_tradable_today,
            )
            plan_lookup = {plan.symbol.upper(): plan for plan in plans}

            # Execute trades
            for symbol, row in symbol_rows.items():
                last_close[symbol] = float(row["close"])
                plan = plan_lookup.get(symbol.upper())
                if plan is None:
                    continue

                buy_price = max(plan.buy_price, 1e-6)
                sell_price = max(plan.sell_price, 1e-6)
                target_amount = float(plan.trade_amount)
                fee_rate = self.crypto_fee if symbol in self.crypto_symbols else self.stock_fee
                high = float(row["high"])
                low = float(row["low"])

                # Buy leg
                max_affordable = cash / (buy_price * (1.0 + fee_rate))
                buy_qty = min(target_amount, max_affordable)
                if low <= buy_price and buy_qty > 0:
                    cash -= buy_qty * buy_price * (1.0 + fee_rate)
                    inventory[symbol] += buy_qty

                # Sell leg
                sellable = min(target_amount, inventory[symbol])
                if high >= sell_price and sellable > 0:
                    cash += sellable * sell_price * (1.0 - fee_rate)
                    inventory[symbol] -= sellable

            # Calculate portfolio value
            portfolio_value = cash
            for symbol, qty in inventory.items():
                close_price = last_close.get(symbol)
                if close_price is not None:
                    portfolio_value += qty * close_price

            daily_return = 0.0
            if prev_value > 0:
                daily_return = (portfolio_value - prev_value) / prev_value

            results.append(SimulationResult(
                date=date,
                equity=float(portfolio_value),
                cash=float(cash),
                daily_return=float(daily_return),
                leverage=float(leverage),
                leverage_cost=float(leverage_cost),
            ))
            daily_returns.append(float(daily_return))
            prev_value = portfolio_value

        # Compute summary statistics
        sortino = self._sortino_ratio(daily_returns)
        final_equity = results[-1].equity if results else self.initial_cash
        total_leverage_costs = sum(r.leverage_cost for r in results)
        max_leverage = max((r.leverage for r in results), default=0.0)

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
        """Calculate Sortino ratio from daily returns."""
        if not returns:
            return 0.0
        arr = np.asarray(returns, dtype=np.float64)
        downside = arr[arr < 0]
        downside_std = float(np.sqrt(np.mean(np.square(downside)))) if downside.size else 0.0
        mean_return = float(arr.mean())
        if downside_std == 0.0:
            return float("inf") if mean_return > 0 else 0.0
        return mean_return / downside_std


def _load_non_tradable_file(path: Path) -> List[str]:
    """Load non-tradable symbols from file."""
    if not path.exists():
        return []
    try:
        payload = path.read_text()
        if payload.strip().startswith("{"):
            data = json.loads(payload)
            if isinstance(data, dict) and "non_tradable" in data:
                entries = data["non_tradable"]
                if isinstance(entries, list):
                    return [str(item["symbol"] if isinstance(item, dict) else item) for item in entries]
        return [line.strip() for line in payload.splitlines() if line.strip()]
    except Exception:
        return []


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="V2 Neural daily market simulator.")
    parser.add_argument("--checkpoint", required=True, help="Path to V2 checkpoint.")
    parser.add_argument("--symbols", nargs="*", help="Optional subset of symbols.")
    parser.add_argument("--non-tradable", nargs="*", help="Symbols to exclude from trading.")
    parser.add_argument("--non-tradable-file", help="Path to non-tradable symbols file.")
    parser.add_argument("--stocks-closed", action="store_true", help="Disable equity trading.")
    parser.add_argument("--no-weekend-auto-hold", action="store_false", dest="auto_weekend_hold")
    parser.add_argument("--data-root", default="trainingdata/train")
    parser.add_argument("--forecast-cache", default="strategytraining/forecast_cache")
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--device", default=None)
    parser.add_argument("--crypto-only", action="store_true")
    parser.add_argument("--start-date", help="ISO start date.")
    parser.add_argument("--days", type=int, default=10)
    parser.add_argument("--initial-cash", type=float, default=1.0)
    parser.add_argument("--stock-fee", type=float, default=0.0008)
    parser.add_argument("--crypto-fee", type=float, default=0.0008)
    parser.add_argument("--confidence-threshold", type=float, default=None)
    parser.add_argument("--ignore-non-tradable", action="store_true")
    return parser.parse_args()


def run_cli_simulation() -> None:
    """Run simulation from command line."""
    args = parse_args()

    # Build dataset config
    dataset_cfg = DailyDatasetConfigV2(
        data_root=Path(args.data_root),
        forecast_cache_dir=Path(args.forecast_cache),
        sequence_length=args.sequence_length,
        crypto_only=args.crypto_only,
    )

    # Load non-tradable symbols
    non_tradable: Set[str] = {sym.upper() for sym in (args.non_tradable or [])}
    if args.non_tradable_file:
        non_tradable.update(sym.upper() for sym in _load_non_tradable_file(Path(args.non_tradable_file)))
    else:
        checkpoint_path = Path(args.checkpoint)
        default_nt = checkpoint_path.parent / "non_tradable.json"
        non_tradable.update(sym.upper() for sym in _load_non_tradable_file(default_nt))

    # Create runtime
    runtime = DailyTradingRuntimeV2(
        Path(args.checkpoint),
        dataset_config=dataset_cfg,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
        non_tradable=() if args.ignore_non_tradable else tuple(non_tradable),
    )

    # Get symbols
    symbols = args.symbols or list(dataset_cfg.symbols)

    # Create and run simulator
    simulator = NeuralDailyMarketSimulatorV2(
        runtime,
        symbols,
        stock_fee=args.stock_fee,
        crypto_fee=args.crypto_fee,
        initial_cash=args.initial_cash,
        stocks_closed=args.stocks_closed,
        auto_weekend_hold=args.auto_weekend_hold,
    )

    results, summary = simulator.run(start_date=args.start_date, days=args.days)

    # Print results
    print(f"{'Date':<15} {'Equity':>12} {'Cash':>12} {'Return':>10} {'Leverage':>10} {'LevCost':>10}")
    for entry in results:
        date_str = entry.date.strftime("%Y-%m-%d")
        print(
            f"{date_str:<15} {entry.equity:>12.4f} {entry.cash:>12.4f} "
            f"{entry.daily_return:>10.4f} {entry.leverage:>10.2f}x {entry.leverage_cost:>10.6f}"
        )

    print("\nSimulation Summary (V2)")
    print(f"Final Equity      : {summary['final_equity']:.4f}")
    print(f"Net PnL           : {summary['pnl']:.4f}")
    print(f"Sortino Ratio     : {summary['sortino']:.4f}")
    print(f"Max Leverage      : {summary['max_leverage']:.2f}x")
    print(f"Total Lev. Costs  : {summary['total_leverage_costs']:.6f}")


if __name__ == "__main__":
    run_cli_simulation()
