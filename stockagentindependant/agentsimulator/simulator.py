"""Minimal simulator for stateless agent backtests."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, asdict
from datetime import date
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from loguru import logger

from .data_models import ExecutionSession, PlanActionType, TradingInstruction, TradingPlan
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


@dataclass
class TradeExecution:
    trade_date: date
    symbol: str
    direction: str
    action: str
    quantity: float
    price: float
    execution_session: ExecutionSession
    realized_pnl: float
    fee_paid: float

    def to_dict(self) -> Dict[str, float]:
        payload = asdict(self)
        payload["execution_session"] = self.execution_session.value
        return payload


@dataclass
class SimulationResult:
    realized_pnl: float
    total_fees: float
    trades: List[Dict[str, float]]


class AgentSimulator:
    """Simple simulator that assumes starting from cash each day."""

    def __init__(self, market_data):
        self.market_data = market_data
        self.trade_log: List[TradeExecution] = []
        self.realized_pnl: float = 0.0
        self.total_fees: float = 0.0
        self.positions: Dict[str, PositionState] = {}

    def reset(self) -> None:
        self.trade_log.clear()
        self.realized_pnl = 0.0
        self.total_fees = 0.0
        self.positions.clear()

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
        return float(row.get("open" if session == ExecutionSession.MARKET_OPEN else "close"))

    def _apply_trade(self, trade_date: date, instruction: TradingInstruction, price: float) -> None:
        symbol = instruction.symbol
        if instruction.action == PlanActionType.HOLD:
            return

        position = self.positions.setdefault(symbol, PositionState())
        signed_qty = instruction.quantity if instruction.action == PlanActionType.BUY else -instruction.quantity
        fee_rate = CRYPTO_TRADING_FEE if symbol in crypto_symbols else TRADING_FEE
        fee_paid = abs(signed_qty) * price * fee_rate
        self.total_fees += fee_paid

        realized = 0.0
        if instruction.action == PlanActionType.EXIT:
            realized = (price - position.avg_price) * position.quantity
            position.quantity = 0.0
            position.avg_price = 0.0
            signed_qty = -position.quantity
        else:
            if instruction.action == PlanActionType.BUY:
                new_qty = position.quantity + signed_qty
                total_cost = position.avg_price * position.quantity + price * signed_qty
                position.quantity = new_qty
                position.avg_price = total_cost / new_qty if new_qty != 0 else 0.0
            else:  # SELL
                realized = (price - position.avg_price) * min(position.quantity, instruction.quantity)
                position.quantity -= instruction.quantity
                if position.quantity == 0:
                    position.avg_price = 0.0

        self.realized_pnl += realized - fee_paid
        direction = "long" if signed_qty > 0 else "short"
        self.trade_log.append(
            TradeExecution(
                trade_date=trade_date,
                symbol=symbol,
                direction=direction,
                action=instruction.action.value,
                quantity=signed_qty,
                price=price,
                execution_session=instruction.execution_session,
                realized_pnl=realized - fee_paid,
                fee_paid=fee_paid,
            )
        )

    def simulate(self, plans: Iterable[TradingPlan]) -> SimulationResult:
        self.reset()
        plans = sorted(plans, key=lambda plan: plan.target_date)
        for index, plan in enumerate(plans):
            if index >= SIMULATION_DAYS:
                break
            instructions = [deepcopy(instr) for instr in plan.instructions]
            for instruction in instructions:
                try:
                    price = self._price_for(instruction.symbol, plan.target_date, instruction.execution_session)
                except KeyError as exc:
                    logger.warning("Skipping %s: %s", instruction.symbol, exc)
                    continue
                self._apply_trade(plan.target_date, instruction, price)
        return SimulationResult(
            realized_pnl=self.realized_pnl,
            total_fees=self.total_fees,
            trades=[trade.to_dict() for trade in self.trade_log],
        )
