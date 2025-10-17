from __future__ import annotations

from copy import deepcopy
from datetime import date
from typing import Dict, Tuple

from loguru import logger
from typing_extensions import override

from stockagent.agentsimulator.data_models import PlanActionType, TradingInstruction
from stockagent.agentsimulator.interfaces import BaseRiskStrategy, DaySummary


class SymbolDirectionLossGuard(BaseRiskStrategy):
    """
    Skips future trades for any symbol/side pair whose most recent realized P&L was negative.

    The guard watches the per-symbol, per-direction realized P&L reported at the end of each
    simulated day. If the most recent value is negative, subsequent BUY (long) or SELL (short)
    instructions for that symbol are dropped entirely until the direction posts a profit again.
    """

    def __init__(self, ignore_on_zero: bool = True) -> None:
        self.ignore_on_zero = ignore_on_zero
        self._allow_map: Dict[Tuple[str, str], bool] = {}

    @override
    def on_simulation_start(self) -> None:
        self._allow_map = {}

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
                allowed = self._allow_map.get((item.symbol, direction), True)
                if not allowed:
                    logger.debug(
                        "LossGuard: skipping %s %s trade on %s due to last loss.",
                        item.symbol,
                        direction,
                        date,
                    )
                    continue  # drop the trade entirely
            adjusted.append(item)
        return adjusted

    @override
    def after_day(self, summary: DaySummary) -> None:
        for (symbol, direction), pnl in summary.per_symbol_direction.items():
            if pnl > 0:
                self._allow_map[(symbol, direction)] = True
            elif pnl < 0:
                self._allow_map[(symbol, direction)] = False
            elif not self.ignore_on_zero:
                # Neutral P&L counts as a loss if the guard is configured accordingly.
                self._allow_map[(symbol, direction)] = False
