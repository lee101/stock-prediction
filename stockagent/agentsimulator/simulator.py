"""Trading simulator for plan evaluation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, asdict
from datetime import date
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from loguru import logger

from .data_models import (
    AccountSnapshot,
    ExecutionSession,
    PlanActionType,
    TradingInstruction,
    TradingPlan,
)
from .interfaces import BaseRiskStrategy, DaySummary
from .market_data import MarketDataBundle
from ..constants import SIMULATION_DAYS, TRADING_FEE, CRYPTO_TRADING_FEE
from src.fixtures import crypto_symbols


@dataclass
class PositionState:
    quantity: float = 0.0
    avg_price: float = 0.0

    def market_value(self, price: float) -> float:
        return self.quantity * price

    def unrealized(self, price: float) -> float:
        if self.quantity > 0:
            return (price - self.avg_price) * self.quantity
        if self.quantity < 0:
            return (self.avg_price - price) * abs(self.quantity)
        return 0.0

    @property
    def side(self) -> str:
        if self.quantity > 0:
            return "long"
        if self.quantity < 0:
            return "short"
        return "flat"


@dataclass
class TradeExecution:
    trade_date: date
    symbol: str
    direction: str
    action: str
    quantity: float
    price: float
    execution_session: ExecutionSession
    requested_price: Optional[float]
    realized_pnl: float
    fee_paid: float

    def to_dict(self) -> Dict[str, float]:
        payload = asdict(self)
        payload["execution_session"] = self.execution_session.value
        return payload


@dataclass
class SimulationResult:
    starting_cash: float
    ending_cash: float
    ending_equity: float
    realized_pnl: float
    unrealized_pnl: float
    equity_curve: List[Dict[str, float]]
    trades: List[Dict[str, float]]
    final_positions: Dict[str, Dict[str, float]]
    total_fees: float

    def to_dict(self) -> Dict:
        return {
            "starting_cash": self.starting_cash,
            "ending_cash": self.ending_cash,
            "ending_equity": self.ending_equity,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "equity_curve": self.equity_curve,
            "trades": self.trades,
            "final_positions": self.final_positions,
            "total_fees": self.total_fees,
        }


class AgentSimulator:
    def __init__(
        self,
        market_data: MarketDataBundle,
        account_snapshot: Optional[AccountSnapshot] = None,
        starting_cash: Optional[float] = None,
    ):
        self.market_data = market_data
        self.trade_log: List[TradeExecution] = []
        self.equity_curve: List[Dict[str, float]] = []
        self.positions: Dict[str, PositionState] = {}
        self.realized_pnl: float = 0.0
        self.cash: float = starting_cash if starting_cash is not None else 0.0
        self._strategies: List[BaseRiskStrategy] = []
        self.total_fees: float = 0.0

        if account_snapshot is not None:
            self.cash = starting_cash if starting_cash is not None else account_snapshot.cash
            for position in account_snapshot.positions:
                self.positions[position.symbol] = PositionState(
                    quantity=position.quantity,
                    avg_price=position.avg_entry_price,
                )
        self.starting_cash = self.cash

    def _get_symbol_frame(self, symbol: str) -> pd.DataFrame:
        df = self.market_data.get_symbol_bars(symbol)
        if df.empty:
            raise KeyError(f"No OHLC data for symbol {symbol}")
        return df

    def _price_for(self, symbol: str, target_date: date, session: ExecutionSession) -> float:
        df = self._get_symbol_frame(symbol)
        try:
            row = df[df.index.date == target_date].iloc[0]
        except IndexError as exc:
            raise KeyError(f"No price data for {symbol} on {target_date}") from exc
        if session == ExecutionSession.MARKET_OPEN:
            return float(row.get("open", row.get("close")))
        return float(row.get("close"))

    def _apply_trade(self, trade_date: date, instruction: TradingInstruction, execution_price: float) -> None:
        symbol = instruction.symbol
        if instruction.action == PlanActionType.HOLD:
            return
        position = self.positions.setdefault(symbol, PositionState())
        signed_qty = instruction.quantity if instruction.action == PlanActionType.BUY else -instruction.quantity

        if instruction.action == PlanActionType.EXIT:
            if position.quantity == 0:
                logger.debug("EXIT ignored for %s (no position)", symbol)
                return
            trade_side = -1 if position.quantity > 0 else 1
            signed_qty = trade_side * abs(instruction.quantity or position.quantity)
            direction_label = "long" if position.quantity > 0 else "short"
        else:
            direction_label = "long" if instruction.action == PlanActionType.BUY else "short"

        if signed_qty == 0:
            logger.debug("Zero quantity instruction for %s", symbol)
            return

        abs_qty = abs(signed_qty)
        fee_rate = CRYPTO_TRADING_FEE if symbol in crypto_symbols else TRADING_FEE
        fee_paid = abs_qty * execution_price * fee_rate
        closing_qty = 0.0
        realized = 0.0

        self.cash -= signed_qty * execution_price
        self.cash -= fee_paid
        self.total_fees += fee_paid

        previous_qty = position.quantity
        same_direction = previous_qty == 0 or (previous_qty > 0 and signed_qty > 0) or (previous_qty < 0 and signed_qty < 0)

        if same_direction:
            new_qty = previous_qty + signed_qty
            if new_qty == 0:
                position.avg_price = 0.0
            else:
                total_cost = position.avg_price * previous_qty + execution_price * signed_qty
                position.avg_price = total_cost / new_qty
            position.quantity = new_qty
        else:
            closing_qty = min(abs(previous_qty), abs_qty)
            if closing_qty > 0:
                sign = 1 if previous_qty > 0 else -1
                realized = closing_qty * (execution_price - position.avg_price) * sign
                self.realized_pnl += realized
            new_qty = previous_qty + signed_qty
            if new_qty == 0:
                position.quantity = 0.0
                position.avg_price = 0.0
            elif (previous_qty > 0 and new_qty > 0) or (previous_qty < 0 and new_qty < 0):
                position.quantity = new_qty
            else:
                position.quantity = new_qty
                position.avg_price = execution_price

        closing_fee = fee_paid * (closing_qty / abs_qty) if abs_qty > 0 else 0.0
        if closing_fee:
            realized -= closing_fee
            self.realized_pnl -= closing_fee

        self.trade_log.append(
            TradeExecution(
                trade_date=trade_date,
                symbol=symbol,
                direction=direction_label,
                action=instruction.action.value,
                quantity=signed_qty,
                price=execution_price,
                execution_session=instruction.execution_session,
                requested_price=instruction.entry_price,
                realized_pnl=realized,
                fee_paid=fee_paid,
            )
        )

    def _mark_to_market(self, target_date: date) -> Dict[str, float]:
        equity = self.cash
        unrealized_total = 0.0
        for symbol, position in self.positions.items():
            if position.quantity == 0:
                continue
            try:
                price = self._price_for(symbol, target_date, ExecutionSession.MARKET_CLOSE)
            except KeyError:
                continue
            unrealized = position.unrealized(price)
            unrealized_total += unrealized
            equity += position.market_value(price)
        snapshot = {
            "date": target_date.isoformat(),
            "cash": self.cash,
            "equity": equity,
            "unrealized_pnl": unrealized_total,
            "realized_pnl": self.realized_pnl,
            "total_fees": self.total_fees,
        }
        self.equity_curve.append(snapshot)
        return snapshot

    def simulate(
        self,
        plans: Iterable[TradingPlan],
        strategies: Optional[Sequence[BaseRiskStrategy]] = None,
    ) -> SimulationResult:
        plans = sorted(plans, key=lambda plan: plan.target_date)
        if not plans:
            raise ValueError("No trading plans supplied to simulator")

        self._strategies = list(strategies or [])
        for strategy in self._strategies:
            strategy.on_simulation_start()

        previous_realized = self.realized_pnl

        for index, plan in enumerate(plans):
            if index >= SIMULATION_DAYS:
                logger.info("Simulation truncated at %d days", SIMULATION_DAYS)
                break

            instructions = [deepcopy(instruction) for instruction in plan.instructions]
            for strategy in self._strategies:
                instructions = strategy.before_day(
                    day_index=index,
                    date=plan.target_date,
                    instructions=[deepcopy(instruction) for instruction in instructions],
                    simulator=self,
                )

            trade_log_start = len(self.trade_log)
            for instruction in instructions:
                try:
                    execution_price = self._price_for(
                        instruction.symbol,
                        plan.target_date,
                        instruction.execution_session,
                    )
                except KeyError as exc:
                    logger.warning("Skipping %s: %s", instruction.symbol, exc)
                    continue
                self._apply_trade(plan.target_date, instruction, execution_price)
            self._mark_to_market(plan.target_date)

            day_trades = self.trade_log[trade_log_start:]
            daily_realized = self.realized_pnl - previous_realized
            previous_realized = self.realized_pnl

            per_symbol_direction: Dict[Tuple[str, str], float] = {}
            trades_payload: List[Dict[str, float]] = []
            for trade in day_trades:
                key = (trade.symbol, trade.direction)
                per_symbol_direction[key] = per_symbol_direction.get(key, 0.0) + trade.realized_pnl
                trades_payload.append(trade.to_dict())

            day_summary = DaySummary(
                date=plan.target_date,
                realized_pnl=daily_realized,
                total_equity=self.equity_curve[-1]["equity"],
                trades=trades_payload,
                per_symbol_direction=per_symbol_direction,
            )
            for strategy in self._strategies:
                strategy.after_day(day_summary)

        final_snapshot = self.equity_curve[-1] if self.equity_curve else {"equity": self.cash, "unrealized_pnl": 0.0}
        ending_equity = final_snapshot["equity"]
        ending_unrealized = final_snapshot["unrealized_pnl"]

        final_positions = {
            symbol: {"quantity": state.quantity, "avg_price": state.avg_price}
            for symbol, state in self.positions.items()
            if state.quantity != 0
        }

        for strategy in self._strategies:
            strategy.on_simulation_end()

        return SimulationResult(
            starting_cash=self.starting_cash,
            ending_cash=self.cash,
            ending_equity=ending_equity,
            realized_pnl=self.realized_pnl,
            unrealized_pnl=ending_unrealized,
            equity_curve=self.equity_curve,
            trades=[trade.to_dict() for trade in self.trade_log],
            final_positions=final_positions,
            total_fees=self.total_fees,
        )
