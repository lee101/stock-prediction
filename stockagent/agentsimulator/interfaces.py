"""Interfaces shared by simulator extensions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Tuple

from .data_models import TradingInstruction


@dataclass
class DaySummary:
    date: date
    realized_pnl: float
    total_equity: float
    trades: List[Dict[str, float]]
    per_symbol_direction: Dict[Tuple[str, str], float]


class BaseRiskStrategy:
    def on_simulation_start(self) -> None:
        """Hook called at the beginning of simulation."""

    def on_simulation_end(self) -> None:
        """Hook called at the end of simulation."""

    def before_day(
        self,
        *,
        day_index: int,
        date: date,
        instructions: List[TradingInstruction],
        simulator: "AgentSimulator",
    ) -> List[TradingInstruction]:
        return instructions

    def after_day(self, summary: DaySummary) -> None:
        """Hook invoked after the day completes."""


from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .simulator import AgentSimulator
