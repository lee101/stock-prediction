"""Interfaces shared by simulator extensions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import List, Dict, Tuple

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
        """Hook called at the beginning of a simulation run."""

    def on_simulation_end(self) -> None:
        """Hook called at the end of a simulation run."""

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
        """Hook invoked after a day completes."""


from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .simulator import AgentSimulator
