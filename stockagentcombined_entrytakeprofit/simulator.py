from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, Iterable, List, Tuple

from stockagent.agentsimulator.data_models import (
    ExecutionSession,
    PlanActionType,
    TradingInstruction,
    TradingPlan,
)
from stockagent.agentsimulator.market_data import MarketDataBundle
from agentsimulatorshared.metrics import ReturnMetrics, compute_return_metrics


@dataclass
class EntryTakeProfitResult:
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
        trading_days_per_month: int = 21,
        trading_days_per_year: int = 252,
    ) -> ReturnMetrics:
        return compute_return_metrics(
            net_pnl=self.net_pnl,
            starting_nav=starting_nav,
            periods=periods,
            trading_days_per_month=trading_days_per_month,
            trading_days_per_year=trading_days_per_year,
        )

    def summary(
        self,
        *,
        starting_nav: float,
        periods: int,
        trading_days_per_month: int = 21,
        trading_days_per_year: int = 252,
    ) -> Dict[str, float]:
        metrics = self.return_metrics(
            starting_nav=starting_nav,
            periods=periods,
            trading_days_per_month=trading_days_per_month,
            trading_days_per_year=trading_days_per_year,
        )
        return {
            "realized_pnl": self.realized_pnl,
            "fees": self.total_fees,
            "net_pnl": self.net_pnl,
            "ending_cash": self.ending_cash,
            "ending_equity": self.ending_equity,
            "daily_return_pct": metrics.daily_pct,
            "monthly_return_pct": metrics.monthly_pct,
            "annual_return_pct": metrics.annual_pct,
        }


class EntryTakeProfitSimulator:
    """
    Simulates an entry + take-profit strategy where entries are filled at the specified
    session price (open/close) and exits are attempted intraday at their target prices.

    If the profit target is not reached during the session, the position is flattened at
    the session's close price.
    """

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

    def run(self, plans: Iterable[TradingPlan]) -> EntryTakeProfitResult:
        cash = 0.0
        positions: Dict[str, Tuple[float, float]] = {}  # symbol -> (quantity, avg_price)
        realized = 0.0
        fees = 0.0

        for plan in sorted(plans, key=lambda p: p.target_date):
            day_high: Dict[str, float] = {}
            day_low: Dict[str, float] = {}
            day_close: Dict[str, float] = {}

            exits: Dict[str, TradingInstruction] = {}
            entries: List[TradingInstruction] = []
            for instruction in plan.instructions:
                if instruction.action in (PlanActionType.BUY, PlanActionType.SELL):
                    entries.append(instruction)
                elif instruction.action == PlanActionType.EXIT:
                    exits[instruction.symbol] = instruction

            for instruction in entries:
                day_frame = self._get_day_frame_for_symbol(instruction.symbol, plan.target_date)
                if day_frame is None:
                    continue
                day_high[instruction.symbol] = float(day_frame["high"])
                day_low[instruction.symbol] = float(day_frame["low"])
                day_close[instruction.symbol] = float(day_frame["close"])

                price = self._resolve_price(day_frame, instruction.execution_session)
                qty = instruction.quantity
                if qty <= 0:
                    continue
                fee_rate = self._fee_rate(instruction.symbol)
                fee_paid = abs(qty) * price * fee_rate
                fees += fee_paid

                if instruction.action == PlanActionType.BUY:
                    cash -= qty * price + fee_paid
                    pos_qty, pos_avg = positions.get(instruction.symbol, (0.0, 0.0))
                    new_qty = pos_qty + qty
                    new_avg = (
                        (pos_qty * pos_avg + qty * price) / new_qty
                        if new_qty != 0
                        else 0.0
                    )
                    positions[instruction.symbol] = (new_qty, new_avg)
                else:
                    # SELL to open short
                    cash += qty * price - fee_paid
                    pos_qty, pos_avg = positions.get(instruction.symbol, (0.0, 0.0))
                    new_qty = pos_qty - qty
                    new_avg = (
                        (pos_qty * pos_avg - qty * price) / new_qty
                        if new_qty != 0
                        else 0.0
                    )
                    positions[instruction.symbol] = (new_qty, new_avg)

            for symbol, instruction in exits.items():
                day_frame = self._get_day_frame_for_symbol(symbol, plan.target_date)
                if day_frame is None:
                    continue
                high = day_high.get(symbol, float(day_frame["high"]))
                low = day_low.get(symbol, float(day_frame["low"]))
                close_price = day_close.get(symbol, float(day_frame["close"]))

                pos_qty, pos_avg = positions.get(symbol, (0.0, 0.0))
                if pos_qty == 0.0:
                    continue
                target = instruction.exit_price
                fee_rate = self._fee_rate(symbol)
                exit_qty = abs(pos_qty) if instruction.quantity <= 0 else min(abs(pos_qty), instruction.quantity)
                exit_qty = float(exit_qty)
                if exit_qty == 0.0:
                    continue

                if pos_qty > 0:  # long position
                    execution_price = self._pick_take_profit_price(
                        target_price=target,
                        hit_condition=lambda tgt: tgt is not None and tgt <= high,
                        default_price=close_price,
                    )
                    pnl = (execution_price - pos_avg) * exit_qty
                    cash += exit_qty * execution_price
                    realized += pnl
                    fees += exit_qty * execution_price * fee_rate
                    remaining_qty = pos_qty - exit_qty
                else:  # short position
                    execution_price = self._pick_take_profit_price(
                        target_price=target,
                        hit_condition=lambda tgt: tgt is not None and tgt >= low,
                        default_price=close_price,
                    )
                    pnl = (pos_avg - execution_price) * exit_qty
                    cash -= exit_qty * execution_price
                    realized += pnl
                    fees += exit_qty * execution_price * fee_rate
                    remaining_qty = pos_qty + exit_qty  # pos_qty is negative, so add qty

                if abs(remaining_qty) < 1e-9:
                    positions.pop(symbol, None)
                else:
                    positions[symbol] = (remaining_qty, pos_avg)

        ending_equity = cash
        for symbol, (qty, avg) in positions.items():
            day_frame = self._get_day_frame_for_symbol(symbol, self.market_data.as_of.date())
            if day_frame is None:
                continue
            market_price = float(day_frame["close"])
            ending_equity += qty * market_price

        return EntryTakeProfitResult(
            realized_pnl=realized,
            total_fees=fees,
            ending_cash=cash,
            ending_equity=ending_equity,
        )

    def _get_day_frame_for_symbol(self, symbol: str, target_date: date):
        frame = self.market_data.bars.get(symbol.upper())
        if frame is None:
            return None
        mask = frame.index.date == target_date
        if not mask.any():
            return None
        return frame.loc[mask].iloc[0]

    @staticmethod
    def _pick_take_profit_price(
        *,
        target_price: float | None,
        hit_condition,
        default_price: float,
    ) -> float:
        if target_price is not None and hit_condition(target_price):
            return float(target_price)
        return float(default_price)

    def _fee_rate(self, symbol: str) -> float:
        return self.crypto_fee if "USD" in symbol and len(symbol) > 4 else self.trading_fee

    @staticmethod
    def _resolve_price(day_frame, session: ExecutionSession) -> float:
        if session == ExecutionSession.MARKET_OPEN:
            return float(day_frame["open"])
        return float(day_frame["close"])
