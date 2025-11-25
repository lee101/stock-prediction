from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from neural_trade_stock_e2e import _build_dataset_config
from neuraldailytraining import DailyTradingRuntime
from src.fixtures import all_crypto_symbols


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
        stock_fee: float = 0.0005,
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
        self.frames: dict[str, pd.DataFrame] = {}
        self.crypto_symbols: set[str] = set()

        for symbol in self.symbols:
            frame = runtime._builder.build(symbol)  # type: ignore[attr-defined]
            if frame.empty:
                continue
            frame = frame.copy()
            frame["date"] = pd.to_datetime(frame["date"], utc=True)
            self.frames[symbol] = frame

            # Track which symbols are crypto
            if symbol.upper() in all_crypto_symbols:
                self.crypto_symbols.add(symbol)

        if not self.frames:
            raise ValueError("No symbols have usable historical data for simulation.")

    def _available_dates(self) -> list[pd.Timestamp]:
        union: set[pd.Timestamp] = set()
        for frame in self.frames.values():
            union.update(frame["date"])
        ordered = sorted(union)
        return ordered

    def _select_dates(self, start_date: str | None, days: int) -> list[pd.Timestamp]:
        ordered = self._available_dates()
        if not ordered:
            raise ValueError("No historical dates available for simulation.")
        if start_date:
            cutoff = pd.to_datetime(start_date, utc=True)
            ordered = [date for date in ordered if date >= cutoff]
        if len(ordered) < days:
            raise ValueError(f"Requested {days} simulation days but only {len(ordered)} available.")
        return ordered[:days]

    def run(self, *, start_date: str | None = None, days: int = 5) -> tuple[list[SimulationResult], dict[str, float]]:
        dates = self._select_dates(start_date, days)
        cash = self.initial_cash
        prev_value = cash
        inventory: dict[str, float] = dict.fromkeys(self.symbols, 0.0)
        last_close: dict[str, float] = {symbol: float(frame["close"].iloc[-1]) for symbol, frame in self.frames.items()}
        results: list[SimulationResult] = []
        daily_returns: list[float] = []

        for date in dates:
            # Calculate leverage cost at start of day
            # Separate stock and crypto positions
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

            symbol_rows: dict[str, pd.Series] = {}
            for symbol, frame in self.frames.items():
                rows = frame[frame["date"] == date]
                if not rows.empty:
                    symbol_rows[symbol] = rows.iloc[0]
            # Determine which symbols are tradable today
            non_tradable_today = set(self.runtime.non_tradable)
            if self.stocks_closed:
                non_tradable_today.update(sym for sym in self.symbols if sym not in self.crypto_symbols)
            if self.auto_weekend_hold and date.weekday() >= 5:
                non_tradable_today.update(sym for sym in self.symbols if sym not in self.crypto_symbols)

            plans = self.runtime.plan_batch(
                list(symbol_rows.keys()), as_of=date, non_tradable_override=non_tradable_today
            )
            plan_lookup = {plan.symbol.upper(): plan for plan in plans}
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


def _load_non_tradable_file(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        payload = path.read_text()
        # Support json with {"non_tradable": [...]} or newline separated
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
    parser = argparse.ArgumentParser(description="Simulate neural daily trading over recent days.")
    parser.add_argument("--checkpoint", required=True, help="Path to the neuraldaily checkpoint to evaluate.")
    parser.add_argument("--mode", choices=("plan", "simulate"), default="simulate")
    parser.add_argument("--symbols", nargs="*", help="Optional subset of symbols.")
    parser.add_argument("--non-tradable", nargs="*", help="Symbols to feed as context but never trade.")
    parser.add_argument(
        "--non-tradable-file",
        help="Path to JSON/list of non-tradable symbols; defaults to checkpoint dir non_tradable.json if present.",
    )
    parser.add_argument(
        "--stocks-closed", action="store_true", help="Disable all equity trading for the run (crypto-only)."
    )
    parser.add_argument(
        "--no-weekend-auto-hold",
        action="store_false",
        dest="auto_weekend_hold",
        help="Disable automatic equity freeze on weekends.",
    )
    parser.add_argument("--data-root", default="trainingdata/train")
    parser.add_argument("--forecast-cache", default="strategytraining/forecast_cache")
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--validation-days", type=int, default=40)
    parser.add_argument("--device", default=None)
    parser.add_argument("--crypto-only", action="store_true", help="Restrict dataset/simulation to crypto symbols only.")
    parser.add_argument("--start-date", help="Optional ISO start date for the simulation window.")
    parser.add_argument("--days", type=int, default=5)
    parser.add_argument("--initial-cash", type=float, default=1.0)
    parser.add_argument("--stock-fee", type=float, default=0.0005, help="Per-leg fee rate for stocks (fractional).")
    parser.add_argument(
        "--crypto-fee", type=float, default=0.0008, help="Per-leg fee rate for crypto (fractional, 8 bps)."
    )
    parser.add_argument("--maker-fee", type=float, default=0.0008)
    parser.add_argument("--risk-threshold", type=float, help="Optional override for the runtime risk threshold.")
    parser.add_argument("--confidence-threshold", type=float, default=None, help="Minimum confidence required to trade.")
    parser.add_argument(
        "--ignore-non-tradable",
        action="store_true",
        help="Ignore checkpoint non_tradable file and allow all symbols.",
    )
    # Dataset / grouping flags (kept consistent with neural_trade_stock_e2e)
    parser.add_argument("--require-forecasts", action=argparse.BooleanOptionalAction, default=False, help="Fail if forecast cache missing rows.")
    parser.add_argument("--forecast-fill-strategy", choices=("persistence", "fail"), default="persistence")
    parser.add_argument("--forecast-cache-writeback", action=argparse.BooleanOptionalAction, default=True, help="Persist filled forecasts to cache.")
    parser.add_argument("--grouping-strategy", choices=("static", "correlation"), default="correlation")
    parser.add_argument("--symbol-dropout-rate", type=float, default=0.1)
    parser.add_argument("--exclude-symbols-file", help="Optional newline-delimited list of symbols to drop from training.")
    parser.add_argument("--exclude-symbol", nargs="*", help="Inline symbols to drop from training.")
    parser.add_argument("--corr-min", type=float, default=0.6, help="Correlation threshold for grouping.")
    parser.add_argument("--corr-max-group", type=int, default=12, help="Max symbols per correlation cluster.")
    parser.add_argument("--corr-window-days", type=int, default=252)
    parser.add_argument("--corr-min-overlap", type=int, default=60)
    return parser.parse_args()


def run_cli_simulation() -> None:
    args = parse_args()
    dataset_cfg = _build_dataset_config(args)
    checkpoint_path = Path(args.checkpoint)
    non_tradable: set[str] = {sym.upper() for sym in (args.non_tradable or [])}
    if args.non_tradable_file:
        non_tradable.update(sym.upper() for sym in _load_non_tradable_file(Path(args.non_tradable_file)))
    else:
        default_nt = checkpoint_path.parent / "non_tradable.json"
        non_tradable.update(sym.upper() for sym in _load_non_tradable_file(default_nt))

    runtime = DailyTradingRuntime(
        checkpoint_path,
        dataset_config=dataset_cfg,
        device=args.device,
        risk_threshold=args.risk_threshold,
        non_tradable=() if args.ignore_non_tradable else non_tradable,
        confidence_threshold=args.confidence_threshold,
    )
    symbols = args.symbols or list(dataset_cfg.symbols)
    simulator = NeuralDailyMarketSimulator(
        runtime,
        symbols,
        stock_fee=args.stock_fee,
        crypto_fee=args.crypto_fee,
        initial_cash=args.initial_cash,
    )
    results, summary = simulator.run(start_date=args.start_date, days=args.days)
    print(f"{'Date':<15} {'Equity':>12} {'Cash':>12} {'Return':>10} {'Leverage':>10} {'LevCost':>10}")
    for entry in results:
        date_str = entry.date.strftime("%Y-%m-%d")
        print(
            f"{date_str:<15} {entry.equity:>12.4f} {entry.cash:>12.4f} {entry.daily_return:>10.4f} {entry.leverage:>10.2f}x {entry.leverage_cost:>10.6f}"
        )
    print("\nSimulation Summary")
    print(f"Final Equity      : {summary['final_equity']:.4f}")
    print(f"Net PnL           : {summary['pnl']:.4f}")
    print(f"Sortino Ratio     : {summary['sortino']:.4f}")
    print(f"Max Leverage      : {summary['max_leverage']:.2f}x")
    print(f"Total Lev. Costs  : {summary['total_leverage_costs']:.6f}")


if __name__ == "__main__":
    run_cli_simulation()
