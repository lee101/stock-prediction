"""High-level utilities for generating and simulating DeepSeek trading plans."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping, MutableMapping, Sequence

from deepseek_wrapper import call_deepseek_chat
from stockagent.agentsimulator.data_models import (
    AccountSnapshot,
    TradingPlan,
    TradingPlanEnvelope,
)
from stockagent.agentsimulator.interfaces import BaseRiskStrategy
from stockagent.agentsimulator.market_data import MarketDataBundle
from stockagent.agentsimulator.risk_strategies import (
    ProfitShutdownStrategy,
    ProbeTradeStrategy,
)
from stockagent.agentsimulator.simulator import AgentSimulator, SimulationResult

from .prompt_builder import build_deepseek_messages


def _default_strategies() -> list[BaseRiskStrategy]:
    return [ProbeTradeStrategy(), ProfitShutdownStrategy()]


@dataclass(slots=True)
class DeepSeekPlanResult:
    plan: TradingPlan
    raw_response: str
    simulation: SimulationResult


def generate_deepseek_plan(
    *,
    market_data: MarketDataBundle,
    account_snapshot: AccountSnapshot,
    target_date: date,
    symbols: Sequence[str] | None = None,
    include_market_history: bool = True,
    deepseek_kwargs: Mapping[str, Any] | None = None,
) -> tuple[TradingPlan, str]:
    """Request a trading plan from DeepSeek and return the parsed plan with raw JSON."""
    messages = build_deepseek_messages(
        market_data=market_data,
        target_date=target_date,
        account_snapshot=account_snapshot,
        symbols=symbols,
        include_market_history=include_market_history,
    )
    kwargs: MutableMapping[str, Any] = dict(deepseek_kwargs or {})
    raw_text = call_deepseek_chat(messages, **kwargs)
    plan = TradingPlanEnvelope.from_json(raw_text).plan
    return plan, raw_text


def simulate_deepseek_plan(
    *,
    market_data: MarketDataBundle,
    account_snapshot: AccountSnapshot,
    target_date: date,
    symbols: Sequence[str] | None = None,
    include_market_history: bool = True,
    deepseek_kwargs: Mapping[str, Any] | None = None,
    strategies: Sequence[BaseRiskStrategy] | None = None,
    starting_cash: float | None = None,
) -> DeepSeekPlanResult:
    """Generate a DeepSeek plan and evaluate it with the stock agent simulator."""
    plan, raw_text = generate_deepseek_plan(
        market_data=market_data,
        account_snapshot=account_snapshot,
        target_date=target_date,
        symbols=symbols,
        include_market_history=include_market_history,
        deepseek_kwargs=deepseek_kwargs,
    )
    simulator = AgentSimulator(
        market_data=market_data,
        account_snapshot=account_snapshot,
        starting_cash=starting_cash if starting_cash is not None else account_snapshot.cash,
    )
    strategy_list = list(strategies) if strategies is not None else _default_strategies()
    simulation = simulator.simulate([plan], strategies=strategy_list)
    return DeepSeekPlanResult(plan=plan, raw_response=raw_text, simulation=simulation)


__all__ = ["DeepSeekPlanResult", "generate_deepseek_plan", "simulate_deepseek_plan"]
