"""Interfaces shared by simulator extensions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .data_models import TradingInstruction


@dataclass
class DaySummary:
    date: date
    realized_pnl: float
    total_equity: float
    trades: list[dict[str, float]]
    per_symbol_direction: dict[tuple[str, str], float]


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
        instructions: list[TradingInstruction],
        simulator: object,
    ) -> list[TradingInstruction]:
        return instructions

    def after_day(self, summary: DaySummary) -> None:
        """Hook invoked after the day completes."""
