"""Entry/take-profit evaluation pipeline for DeepSeek plans."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping, Sequence

from stockagent.agentsimulator.data_models import AccountSnapshot, TradingPlan
from stockagent.agentsimulator.market_data import MarketDataBundle
from stockagentcombined_entrytakeprofit.simulator import EntryTakeProfitResult, EntryTakeProfitSimulator

from stockagentdeepseek.agent import generate_deepseek_plan


@dataclass(slots=True)
class DeepSeekEntryTakeProfitResult:
    plan: TradingPlan
    raw_response: str
    simulation: EntryTakeProfitResult

    def summary(
        self,
        *,
        starting_nav: float,
        periods: int,
        trading_days_per_year: int = 252,
    ) -> dict[str, float]:
        return self.simulation.summary(
            starting_nav=starting_nav,
            periods=periods,
            trading_days_per_year=trading_days_per_year,
        )


def simulate_deepseek_entry_takeprofit_plan(
    *,
    market_data: MarketDataBundle,
    account_snapshot: AccountSnapshot,
    target_date: date,
    symbols: Sequence[str] | None = None,
    include_market_history: bool = True,
    deepseek_kwargs: Mapping[str, Any] | None = None,
    simulator: EntryTakeProfitSimulator | None = None,
) -> DeepSeekEntryTakeProfitResult:
    """Generate a DeepSeek plan and evaluate it with the entry/take-profit simulator."""
    plan, raw_response = generate_deepseek_plan(
        market_data=market_data,
        account_snapshot=account_snapshot,
        target_date=target_date,
        symbols=symbols,
        include_market_history=include_market_history,
        deepseek_kwargs=deepseek_kwargs,
    )
    simulator = simulator or EntryTakeProfitSimulator(market_data=market_data)
    simulation = simulator.run([plan])
    return DeepSeekEntryTakeProfitResult(plan=plan, raw_response=raw_response, simulation=simulation)


__all__ = ["DeepSeekEntryTakeProfitResult", "simulate_deepseek_entry_takeprofit_plan"]
