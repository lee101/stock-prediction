"""Optional risk overlays for the simulator."""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List

from loguru import logger

from .data_models import ExecutionSession, PlanActionType, TradingInstruction
from .interfaces import BaseRiskStrategy, DaySummary


class ProbeTradeStrategy(BaseRiskStrategy):
    """Uses small probe trades until a symbol-direction proves profitable."""

    def __init__(self, probe_multiplier: float = 0.05, min_quantity: float = 0.01):
        self.probe_multiplier = probe_multiplier
        self.min_quantity = min_quantity
        self._status: Dict[tuple[str, str], bool] = {}

    def on_simulation_start(self) -> None:
        self._status = {}

    def before_day(
        self,
        *,
        day_index: int,
        date,
        instructions: List[TradingInstruction],
        simulator,
    ) -> List[TradingInstruction]:
        adjusted: List[TradingInstruction] = []
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
            adjusted.append(item)
        return adjusted

    def after_day(self, summary: DaySummary) -> None:
        for (symbol, direction), pnl in summary.per_symbol_direction.items():
            if pnl > 0:
                self._status[(symbol, direction)] = True
            elif pnl < 0:
                self._status[(symbol, direction)] = False


class ProfitShutdownStrategy(BaseRiskStrategy):
    """After a losing day, turns new trades into small probe positions."""

    def __init__(self, probe_multiplier: float = 0.05, min_quantity: float = 0.01):
        self.probe_multiplier = probe_multiplier
        self.min_quantity = min_quantity
        self._probe_mode = False

    def on_simulation_start(self) -> None:
        self._probe_mode = False

    def before_day(
        self,
        *,
        day_index: int,
        date,
        instructions: List[TradingInstruction],
        simulator,
    ) -> List[TradingInstruction]:
        if not self._probe_mode:
            return instructions

        adjusted: List[TradingInstruction] = []
        for instruction in instructions:
            item = deepcopy(instruction)
            if item.action in (PlanActionType.BUY, PlanActionType.SELL) and item.quantity > 0:
                base_qty = item.quantity
                item.quantity = max(base_qty * self.probe_multiplier, self.min_quantity)
            adjusted.append(item)
        return adjusted

    def after_day(self, summary: DaySummary) -> None:
        self._probe_mode = summary.realized_pnl <= 0
