#!/usr/bin/env python3
"""
Compare normal trading vs probe-mode trading side-by-side.

Simulates the same model with and without probe mode risk controls to
show the impact of conservative position sizing after losses.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from neuraldailytraining import DailyTradingRuntime
from neuraldailytraining.config import DailyDatasetConfig
from neural_trade_stock_e2e import _build_dataset_config


PROBE_NOTIONAL_LIMIT = 300.0  # Max position size in probe mode


@dataclass
class TradeOutcome:
    """Record of a completed trade."""
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    pnl_pct: float


@dataclass
class ComparisonResult:
    """Results for one simulation mode."""
    date: pd.Timestamp
    equity: float
    cash: float
    daily_return: float
    leverage: float
    leverage_cost: float
    probe_trades: int = 0  # Number of trades in probe mode (for probe sim)


class DualModeSimulator:
    """Simulate normal and probe modes simultaneously for comparison."""

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
        probe_notional_limit: float = PROBE_NOTIONAL_LIMIT,
    ) -> None:
        self.runtime = runtime
        self.symbols = [symbol.upper() for symbol in symbols]
        self.maker_fee = maker_fee
        self.initial_cash = initial_cash
        self.leverage_fee_rate = leverage_fee_rate
        self.daily_leverage_rate = leverage_fee_rate / 365.0
        self.equity_max_leverage = equity_max_leverage
        self.crypto_max_leverage = crypto_max_leverage
        self.probe_notional_limit = probe_notional_limit
        self.frames: Dict[str, pd.DataFrame] = {}
        self.crypto_symbols: set[str] = set()

        for symbol in self.symbols:
            frame = runtime._builder.build(symbol)  # type: ignore[attr-defined]
            if frame.empty:
                continue
            frame = frame.copy()
            frame["date"] = pd.to_datetime(frame["date"], utc=True)
            self.frames[symbol] = frame

            # Track which symbols are crypto
            if symbol.upper().endswith("USD") or symbol.upper().endswith("-USD"):
                self.crypto_symbols.add(symbol)

        if not self.frames:
            raise ValueError("No symbols have usable historical data for simulation.")

    def _available_dates(self) -> List[pd.Timestamp]:
        union: set[pd.Timestamp] = set()
        for frame in self.frames.values():
            union.update(frame["date"])
        return sorted(union)

    def _select_dates(self, start_date: Optional[str], days: int) -> List[pd.Timestamp]:
        ordered = self._available_dates()
        if not ordered:
            raise ValueError("No historical dates available.")
        if start_date:
            cutoff = pd.to_datetime(start_date, utc=True)
            ordered = [date for date in ordered if date >= cutoff]
        if len(ordered) < days:
            raise ValueError(f"Requested {days} days but only {len(ordered)} available.")
        return ordered[:days]

    def _should_probe(self, symbol: str, side: str, history: List[TradeOutcome]) -> Tuple[bool, Optional[str]]:
        """
        Determine if probe mode should be used based on trade history.

        Matches the logic from neural_daily_trade_with_probe.py.
        """
        # Get recent trades for this symbol/side
        recent_trades = [t for t in history if t.symbol == symbol and t.side == side]

        if len(recent_trades) >= 2:
            # Have 2+ trades: check sum of last 2
            last_two = recent_trades[-2:]
            pnl_sum = sum(t.pnl_pct for t in last_two)
            if pnl_sum <= 0:
                return True, f"recent_pnl_sum={pnl_sum:.4f}"
        elif len(recent_trades) == 1:
            # Have 1 trade: check if negative
            pnl = recent_trades[0].pnl_pct
            if pnl <= 0:
                return True, f"single_trade_negative={pnl:.4f}"

        return False, None

    def _calculate_leverage_cost(
        self,
        cash: float,
        inventory: Dict[str, float],
        last_close: Dict[str, float],
    ) -> Tuple[float, float]:
        """Calculate leverage and associated costs."""
        stock_value = sum(
            inventory.get(symbol, 0.0) * last_close.get(symbol, 0.0)
            for symbol in self.symbols if symbol not in self.crypto_symbols
        )
        crypto_value = sum(
            inventory.get(symbol, 0.0) * last_close.get(symbol, 0.0)
            for symbol in self.symbols if symbol in self.crypto_symbols
        )
        current_equity = cash + stock_value + crypto_value
        leverage = (stock_value + crypto_value) / current_equity if current_equity > 0 else 0.0
        leverage_cost = 0.0

        if current_equity > 0 and stock_value > current_equity:
            leveraged_amount = min(stock_value - current_equity, current_equity)
            leverage_cost = leveraged_amount * self.daily_leverage_rate

        return leverage, leverage_cost

    def run_comparison(
        self, *, start_date: Optional[str] = None, days: int = 5
    ) -> Tuple[List[ComparisonResult], List[ComparisonResult], Dict[str, float]]:
        """
        Run both normal and probe simulations side-by-side.

        Returns:
            (normal_results, probe_results, summary)
        """
        dates = self._select_dates(start_date, days)

        # Normal mode state
        normal_cash = self.initial_cash
        normal_inventory: Dict[str, float] = {symbol: 0.0 for symbol in self.symbols}
        normal_history: List[TradeOutcome] = []

        # Probe mode state
        probe_cash = self.initial_cash
        probe_inventory: Dict[str, float] = {symbol: 0.0 for symbol in self.symbols}
        probe_history: List[TradeOutcome] = []

        # Shared state
        last_close: Dict[str, float] = {
            symbol: float(frame["close"].iloc[-1]) for symbol, frame in self.frames.items()
        }

        normal_results: List[ComparisonResult] = []
        probe_results: List[ComparisonResult] = []
        normal_prev_value = normal_cash
        probe_prev_value = probe_cash

        for date in dates:
            # Calculate leverage costs for both modes
            normal_leverage, normal_lev_cost = self._calculate_leverage_cost(
                normal_cash, normal_inventory, last_close
            )
            probe_leverage, probe_lev_cost = self._calculate_leverage_cost(
                probe_cash, probe_inventory, last_close
            )

            normal_cash -= normal_lev_cost
            probe_cash -= probe_lev_cost

            # Get market data for this day
            symbol_rows: Dict[str, pd.Series] = {}
            for symbol, frame in self.frames.items():
                rows = frame[frame["date"] == date]
                if not rows.empty:
                    symbol_rows[symbol] = rows.iloc[0]

            probe_trades_today = 0

            # Process each symbol
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

                # === NORMAL MODE TRADING ===
                # Buy leg
                max_affordable = normal_cash / (buy_price * (1.0 + self.maker_fee))
                normal_buy_qty = min(target_amount, max_affordable)
                if low <= buy_price and normal_buy_qty > 0:
                    normal_cash -= normal_buy_qty * buy_price * (1.0 + self.maker_fee)
                    normal_inventory[symbol] += normal_buy_qty

                # Sell leg
                normal_sellable = min(target_amount, normal_inventory[symbol])
                if high >= sell_price and normal_sellable > 0:
                    entry_price = normal_inventory[symbol] * buy_price  # Approximate
                    exit_value = normal_sellable * sell_price * (1.0 - self.maker_fee)
                    normal_cash += exit_value
                    pnl = exit_value - (normal_sellable * buy_price * (1.0 + self.maker_fee))
                    pnl_pct = pnl / (normal_sellable * buy_price) if buy_price > 0 else 0.0

                    # Record outcome
                    normal_history.append(
                        TradeOutcome(
                            symbol=symbol,
                            side="buy",
                            entry_price=buy_price,
                            exit_price=sell_price,
                            qty=normal_sellable,
                            pnl=pnl,
                            pnl_pct=pnl_pct,
                        )
                    )
                    normal_inventory[symbol] -= normal_sellable

                # === PROBE MODE TRADING ===
                # Check if probe mode is active for this symbol
                should_probe, probe_reason = self._should_probe(symbol, "buy", probe_history)

                # Calculate position size (probe or normal)
                if should_probe:
                    # Probe mode: cap at probe limit
                    probe_target_notional = min(self.probe_notional_limit, probe_cash * 0.01)
                    probe_target_qty = probe_target_notional / buy_price
                    probe_trades_today += 1
                else:
                    # Normal mode: use model's suggestion
                    probe_target_qty = target_amount

                # Buy leg
                probe_max_affordable = probe_cash / (buy_price * (1.0 + self.maker_fee))
                probe_buy_qty = min(probe_target_qty, probe_max_affordable)
                if low <= buy_price and probe_buy_qty > 0:
                    probe_cash -= probe_buy_qty * buy_price * (1.0 + self.maker_fee)
                    probe_inventory[symbol] += probe_buy_qty

                # Sell leg (also capped if in probe mode)
                if should_probe:
                    probe_sell_target = min(probe_target_qty, probe_inventory[symbol])
                else:
                    probe_sell_target = min(target_amount, probe_inventory[symbol])

                if high >= sell_price and probe_sell_target > 0:
                    exit_value = probe_sell_target * sell_price * (1.0 - self.maker_fee)
                    probe_cash += exit_value
                    pnl = exit_value - (probe_sell_target * buy_price * (1.0 + self.maker_fee))
                    pnl_pct = pnl / (probe_sell_target * buy_price) if buy_price > 0 else 0.0

                    # Record outcome
                    probe_history.append(
                        TradeOutcome(
                            symbol=symbol,
                            side="buy",
                            entry_price=buy_price,
                            exit_price=sell_price,
                            qty=probe_sell_target,
                            pnl=pnl,
                            pnl_pct=pnl_pct,
                        )
                    )
                    probe_inventory[symbol] -= probe_sell_target

            # Calculate portfolio values
            normal_portfolio_value = normal_cash + sum(
                normal_inventory.get(symbol, 0.0) * last_close.get(symbol, 0.0)
                for symbol in self.symbols
            )
            probe_portfolio_value = probe_cash + sum(
                probe_inventory.get(symbol, 0.0) * last_close.get(symbol, 0.0)
                for symbol in self.symbols
            )

            # Calculate returns
            normal_return = (
                (normal_portfolio_value - normal_prev_value) / normal_prev_value
                if normal_prev_value > 0
                else 0.0
            )
            probe_return = (
                (probe_portfolio_value - probe_prev_value) / probe_prev_value
                if probe_prev_value > 0
                else 0.0
            )

            # Store results
            normal_results.append(
                ComparisonResult(
                    date=date,
                    equity=float(normal_portfolio_value),
                    cash=float(normal_cash),
                    daily_return=float(normal_return),
                    leverage=float(normal_leverage),
                    leverage_cost=float(normal_lev_cost),
                )
            )
            probe_results.append(
                ComparisonResult(
                    date=date,
                    equity=float(probe_portfolio_value),
                    cash=float(probe_cash),
                    daily_return=float(probe_return),
                    leverage=float(probe_leverage),
                    leverage_cost=float(probe_lev_cost),
                    probe_trades=probe_trades_today,
                )
            )

            normal_prev_value = normal_portfolio_value
            probe_prev_value = probe_portfolio_value

        # Calculate summary
        normal_final = normal_results[-1].equity if normal_results else self.initial_cash
        probe_final = probe_results[-1].equity if probe_results else self.initial_cash
        normal_returns = [r.daily_return for r in normal_results]
        probe_returns = [r.daily_return for r in probe_results]

        summary = {
            "normal_final_equity": normal_final,
            "normal_pnl": normal_final - self.initial_cash,
            "normal_pnl_pct": ((normal_final - self.initial_cash) / self.initial_cash) * 100,
            "probe_final_equity": probe_final,
            "probe_pnl": probe_final - self.initial_cash,
            "probe_pnl_pct": ((probe_final - self.initial_cash) / self.initial_cash) * 100,
            "difference": probe_final - normal_final,
            "difference_pct": ((probe_final - normal_final) / normal_final) * 100 if normal_final > 0 else 0,
            "normal_sortino": self._sortino_ratio(normal_returns),
            "probe_sortino": self._sortino_ratio(probe_returns),
            "total_probe_trades": sum(r.probe_trades for r in probe_results),
        }

        return normal_results, probe_results, summary

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
    parser = argparse.ArgumentParser(description="Compare normal vs probe mode trading")
    parser.add_argument("--checkpoint", required=True, help="Path to neuraldaily checkpoint")
    parser.add_argument("--symbols", nargs="*", help="Symbols to trade")
    parser.add_argument("--data-root", default="trainingdata/train")
    parser.add_argument("--forecast-cache", default="strategytraining/forecast_cache")
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--validation-days", type=int, default=40)
    parser.add_argument("--device", default=None)
    parser.add_argument("--start-date", help="ISO start date")
    parser.add_argument("--days", type=int, default=5)
    parser.add_argument("--initial-cash", type=float, default=1.0)
    parser.add_argument("--maker-fee", type=float, default=0.0008)
    parser.add_argument("--risk-threshold", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_cfg = _build_dataset_config(args)
    runtime = DailyTradingRuntime(
        args.checkpoint,
        dataset_config=dataset_cfg,
        device=args.device,
        risk_threshold=args.risk_threshold,
    )
    symbols = args.symbols or list(dataset_cfg.symbols)

    simulator = DualModeSimulator(
        runtime,
        symbols,
        maker_fee=args.maker_fee,
        initial_cash=args.initial_cash,
    )

    normal_results, probe_results, summary = simulator.run_comparison(
        start_date=args.start_date, days=args.days
    )

    # Print comparison table
    print("\n" + "=" * 120)
    print("NORMAL MODE vs PROBE MODE COMPARISON")
    print("=" * 120)
    print(
        f"{'Date':<12} | {'Normal Equity':>14} {'Return':>10} | "
        f"{'Probe Equity':>14} {'Return':>10} {'ProbeΔ':>8} | {'Probe Trades':>13}"
    )
    print("-" * 120)

    for normal, probe in zip(normal_results, probe_results):
        date_str = normal.date.strftime("%Y-%m-%d")
        probe_marker = " 🔬" if probe.probe_trades > 0 else ""
        print(
            f"{date_str:<12} | "
            f"{normal.equity:>14.4f} {normal.daily_return:>9.2%} | "
            f"{probe.equity:>14.4f} {probe.daily_return:>9.2%} "
            f"{probe.equity - normal.equity:>+8.4f} | "
            f"{probe.probe_trades:>13}{probe_marker}"
        )

    # Print summary
    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)
    print(f"\n{'Metric':<30} {'Normal Mode':>20} {'Probe Mode':>20} {'Difference':>20}")
    print("-" * 95)
    print(
        f"{'Final Equity':<30} {summary['normal_final_equity']:>20.4f} "
        f"{summary['probe_final_equity']:>20.4f} {summary['difference']:>+20.4f}"
    )
    print(
        f"{'Net PnL':<30} {summary['normal_pnl']:>20.4f} "
        f"{summary['probe_pnl']:>20.4f} {summary['difference']:>+20.4f}"
    )
    print(
        f"{'PnL %':<30} {summary['normal_pnl_pct']:>19.2f}% "
        f"{summary['probe_pnl_pct']:>19.2f}% {summary['difference_pct']:>+19.2f}%"
    )
    print(
        f"{'Sortino Ratio':<30} {summary['normal_sortino']:>20.4f} "
        f"{summary['probe_sortino']:>20.4f} "
        f"{summary['probe_sortino'] - summary['normal_sortino']:>+20.4f}"
    )
    print(f"\n{'Total Probe Trades':<30} {'-':>20} {summary['total_probe_trades']:>20}")

    # Interpretation
    print("\n" + "=" * 120)
    print("INTERPRETATION")
    print("=" * 120)

    if summary["probe_pnl"] > summary["normal_pnl"]:
        print("✅ Probe mode performed BETTER:")
        print(f"   - Protected capital by reducing exposure during losses")
        print(f"   - Outperformed by {summary['difference']:+.4f} ({summary['difference_pct']:+.2f}%)")
    elif summary["probe_pnl"] < summary["normal_pnl"]:
        print("⚠️  Probe mode performed WORSE:")
        print(f"   - Conservative sizing limited gains")
        print(f"   - Underperformed by {summary['difference']:+.4f} ({summary['difference_pct']:+.2f}%)")
    else:
        print("➖ Both modes performed the same")

    print(f"\n   Probe mode was active for {summary['total_probe_trades']} trades")
    print("   (Normal trades use full position sizing)")


if __name__ == "__main__":
    main()
