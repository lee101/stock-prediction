"""Optional risk overlays for the simulator."""

from __future__ import annotations

from copy import deepcopy
from datetime import date
from typing_extensions import override

from loguru import logger

from .data_models import PlanActionType, TradingInstruction
from .interfaces import BaseRiskStrategy, DaySummary


class ProbeTradeStrategy(BaseRiskStrategy):
    def __init__(self, probe_multiplier: float = 0.05, min_quantity: float = 0.01):
        self.probe_multiplier: float = probe_multiplier
        self.min_quantity: float = min_quantity
        self._status: dict[tuple[str, str], bool] = {}

    @override
    def on_simulation_start(self) -> None:
        self._status = {}

    @override
    def before_day(
        self,
        *,
        day_index: int,
        date: date,
        instructions: list[TradingInstruction],
        simulator: object,
    ) -> list[TradingInstruction]:
        adjusted: list[TradingInstruction] = []
        for instruction in instructions:
            item = deepcopy(instruction)
            if item.action in (PlanActionType.BUY, PlanActionType.SELL):
                direction = "long" if item.action == PlanActionType.BUY else "short"
                allowed = self._status.get((item.symbol, direction), True)
                if not allowed and item.quantity > 0:
                    base_qty = item.quantity
                    probe_qty = max(base_qty * self.probe_multiplier, self.min_quantity)
                    logger.debug(f"ProbeTrade: {item.symbol} {direction} {base_qty:.4f} -> {probe_qty:.4f}")
                    item.quantity = probe_qty
                    item.notes = (item.notes or "") + "|probe_trade"
            adjusted.append(item)
        return adjusted

    @override
    def after_day(self, summary: DaySummary) -> None:
        for (symbol, direction), pnl in summary.per_symbol_direction.items():
            if pnl > 0:
                self._status[(symbol, direction)] = True
            elif pnl < 0:
                self._status[(symbol, direction)] = False


class ProfitShutdownStrategy(BaseRiskStrategy):
    def __init__(self, probe_multiplier: float = 0.05, min_quantity: float = 0.01):
        self.probe_multiplier: float = probe_multiplier
        self.min_quantity: float = min_quantity
        self._probe_mode: bool = False

    @override
    def on_simulation_start(self) -> None:
        self._probe_mode = False

    @override
    def before_day(
        self,
        *,
        day_index: int,
        date: date,
        instructions: list[TradingInstruction],
        simulator: object,
    ) -> list[TradingInstruction]:
        if not self._probe_mode:
            return instructions

        adjusted: list[TradingInstruction] = []
        for instruction in instructions:
            item = deepcopy(instruction)
            if item.action in (PlanActionType.BUY, PlanActionType.SELL) and item.quantity > 0:
                base_qty = item.quantity
                item.quantity = max(base_qty * self.probe_multiplier, self.min_quantity)
                item.notes = (item.notes or "") + "|profit_shutdown_probe"
            adjusted.append(item)
        return adjusted

    @override
    def after_day(self, summary: DaySummary) -> None:
        self._probe_mode = summary.realized_pnl <= 0
