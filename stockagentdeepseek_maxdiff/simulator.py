"""Limit-entry/exit simulator for DeepSeek plans."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from stockagent.agentsimulator.data_models import (
    ExecutionSession,
    PlanActionType,
    TradingInstruction,
    TradingPlan,
)
from stockagent.agentsimulator.market_data import MarketDataBundle
from agentsimulatorshared.metrics import ReturnMetrics, compute_return_metrics
from src.fixtures import crypto_symbols


def _get_day_frame(symbol: str, session_date: date, bundle: MarketDataBundle) -> pd.Series | None:
    frame = bundle.get_symbol_bars(symbol)
    if frame.empty:
        return None
    try:
        row = frame.loc[frame.index.date == session_date].iloc[0]
    except IndexError:
        return None
    return row


def _resolve_entry_price(instruction: TradingInstruction, day_bar: pd.Series) -> float | None:
    entry = instruction.entry_price
    if entry is None:
        return None
    high = float(day_bar["high"])
    low = float(day_bar["low"])
    if instruction.action == PlanActionType.BUY and entry <= high and entry >= low:
        return float(entry)
    if instruction.action == PlanActionType.SELL and entry >= low and entry <= high:
        return float(entry)
    return None


def _session_price(day_bar: pd.Series, session: ExecutionSession) -> float:
    if session == ExecutionSession.MARKET_OPEN:
        return float(day_bar.get("open", day_bar.get("close")))
    return float(day_bar.get("close"))


@dataclass
class MaxDiffResult:
    realized_pnl: float
    total_fees: float
    ending_cash: float
    ending_equity: float

    @property
    def net_pnl(self) -> float:
        return self.realized_pnl - self.total_fees

    def return_metrics(
        self,
        *,
        starting_nav: float,
        periods: int,
        trading_days_per_year: int = 252,
    ) -> ReturnMetrics:
        return compute_return_metrics(
            net_pnl=self.net_pnl,
            starting_nav=starting_nav,
            periods=periods,
            trading_days_per_year=trading_days_per_year,
        )

    def summary(
        self,
        *,
        starting_nav: float,
        periods: int,
        trading_days_per_year: int = 252,
    ) -> Dict[str, float]:
        metrics = self.return_metrics(
            starting_nav=starting_nav,
            periods=periods,
            trading_days_per_year=trading_days_per_year,
        )
        return {
            "realized_pnl": self.realized_pnl,
            "fees": self.total_fees,
            "net_pnl": self.net_pnl,
            "ending_cash": self.ending_cash,
            "ending_equity": self.ending_equity,
            "daily_return_pct": metrics.daily_pct,
            "annual_return_pct": metrics.annual_pct,
        }


class MaxDiffSimulator:
    """Simulate a limit-entry/exit strategy that only trades when price triggers are touched."""

    def __init__(
        self,
        *,
        market_data: MarketDataBundle,
        trading_fee: float = 0.0005,
        crypto_fee: float = 0.0015,
    ) -> None:
        self.market_data = market_data
        self.trading_fee = trading_fee
        self.crypto_fee = crypto_fee

    def run(self, plans: Iterable[TradingPlan]) -> MaxDiffResult:
        cash = 0.0
        positions: Dict[str, Tuple[float, float]] = {}
        realized = 0.0
        fees = 0.0

        for plan in sorted(plans, key=lambda p: p.target_date):
            entries: List[TradingInstruction] = []
            exits: Dict[str, TradingInstruction] = {}
            for instruction in plan.instructions:
                if instruction.action in (PlanActionType.BUY, PlanActionType.SELL):
                    entries.append(instruction)
                elif instruction.action == PlanActionType.EXIT:
                    exits[instruction.symbol] = instruction

            for instruction in entries:
                day_bar = _get_day_frame(instruction.symbol, plan.target_date, self.market_data)
                if day_bar is None:
                    continue
                fill_price = _resolve_entry_price(instruction, day_bar)
                if fill_price is None:
                    continue
                qty = float(instruction.quantity or 0.0)
                if qty <= 0:
                    continue
                fee_rate = self._fee_rate(instruction.symbol)
                fee_paid = qty * fill_price * fee_rate
                fees += fee_paid

                if instruction.action == PlanActionType.BUY:
                    cash -= qty * fill_price + fee_paid
                    pos_qty, pos_avg = positions.get(instruction.symbol, (0.0, 0.0))
                    new_qty = pos_qty + qty
                    new_avg = (
                        (pos_qty * pos_avg + qty * fill_price) / new_qty if new_qty != 0 else 0.0
                    )
                    positions[instruction.symbol] = (new_qty, new_avg)
                else:
                    cash += qty * fill_price - fee_paid
                    pos_qty, pos_avg = positions.get(instruction.symbol, (0.0, 0.0))
                    new_qty = pos_qty - qty
                    new_avg = (
                        (pos_qty * pos_avg - qty * fill_price) / new_qty if new_qty != 0 else 0.0
                    )
                    positions[instruction.symbol] = (new_qty, new_avg)

            for symbol, exit_instruction in exits.items():
                day_bar = _get_day_frame(symbol, plan.target_date, self.market_data)
                if day_bar is None:
                    continue
                high = float(day_bar["high"])
                low = float(day_bar["low"])
                close_price = float(day_bar["close"])

                pos_qty, pos_avg = positions.get(symbol, (0.0, 0.0))
                if pos_qty == 0.0:
                    continue
                target = exit_instruction.exit_price
                fee_rate = self._fee_rate(symbol)
                exit_qty = abs(pos_qty) if exit_instruction.quantity <= 0 else min(abs(pos_qty), exit_instruction.quantity)
                if exit_qty <= 0:
                    continue

                if pos_qty > 0:
                    if target is not None and target <= high:
                        execution_price = target
                    else:
                        execution_price = close_price
                    pnl = (execution_price - pos_avg) * exit_qty
                    cash += exit_qty * execution_price
                else:
                    if target is not None and target >= low:
                        execution_price = target
                    else:
                        execution_price = close_price
                    pnl = (pos_avg - execution_price) * exit_qty
                    cash -= exit_qty * execution_price

                realized += pnl
                fees += exit_qty * execution_price * fee_rate
                remaining_qty = pos_qty - exit_qty if pos_qty > 0 else pos_qty + exit_qty
                if abs(remaining_qty) < 1e-9:
                    positions.pop(symbol, None)
                else:
                    positions[symbol] = (remaining_qty, pos_avg)

        ending_equity = cash
        for symbol, (qty, avg_price) in positions.items():
            day_bar = _get_day_frame(symbol, self.market_data.as_of.date(), self.market_data)
            if day_bar is None:
                continue
            ending_equity += qty * float(day_bar["close"])

        return MaxDiffResult(
            realized_pnl=realized,
            total_fees=fees,
            ending_cash=cash,
            ending_equity=ending_equity,
        )

    def _fee_rate(self, symbol: str) -> float:
        return self.crypto_fee if symbol.upper() in crypto_symbols else self.trading_fee
