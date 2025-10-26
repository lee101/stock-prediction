"""Neural forecast integration for DeepSeek planning."""

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
from stockagent.agentsimulator.risk_strategies import ProfitShutdownStrategy, ProbeTradeStrategy
from stockagent.agentsimulator.simulator import AgentSimulator, SimulationResult

from .forecaster import NeuralForecast, build_neural_forecasts
from .prompt_builder import build_neural_messages


def _default_strategies() -> list[BaseRiskStrategy]:
    return [ProbeTradeStrategy(), ProfitShutdownStrategy()]


@dataclass(slots=True)
class DeepSeekNeuralPlanResult:
    plan: TradingPlan
    raw_response: str
    forecasts: Mapping[str, NeuralForecast]
    simulation: SimulationResult


def generate_deepseek_neural_plan(
    *,
    market_data: MarketDataBundle,
    account_snapshot: AccountSnapshot,
    target_date: date,
    symbols: Sequence[str] | None = None,
    include_market_history: bool = True,
    deepseek_kwargs: Mapping[str, Any] | None = None,
    forecasts: Mapping[str, NeuralForecast] | None = None,
) -> tuple[TradingPlan, str, Mapping[str, NeuralForecast]]:
    """Request a DeepSeek plan with neural forecasts."""
    symbol_list = list(symbols or market_data.bars.keys())
    if forecasts is None:
        forecasts = build_neural_forecasts(
            symbols=symbol_list,
            market_data=market_data,
        )

    messages = build_neural_messages(
        forecasts=forecasts,
        market_data=market_data,
        target_date=target_date,
        account_snapshot=account_snapshot,
        symbols=symbol_list,
        include_market_history=include_market_history,
    )
    kwargs: MutableMapping[str, Any] = dict(deepseek_kwargs or {})
    raw_text = call_deepseek_chat(messages, **kwargs)
    plan = TradingPlanEnvelope.from_json(raw_text).plan
    return plan, raw_text, forecasts


def simulate_deepseek_neural_plan(
    *,
    market_data: MarketDataBundle,
    account_snapshot: AccountSnapshot,
    target_date: date,
    symbols: Sequence[str] | None = None,
    include_market_history: bool = True,
    deepseek_kwargs: Mapping[str, Any] | None = None,
    strategies: Sequence[BaseRiskStrategy] | None = None,
    starting_cash: float | None = None,
    forecasts: Mapping[str, NeuralForecast] | None = None,
) -> DeepSeekNeuralPlanResult:
    """Generate a DeepSeek plan with neural context and evaluate it."""
    plan, raw_text, resolved_forecasts = generate_deepseek_neural_plan(
        market_data=market_data,
        account_snapshot=account_snapshot,
        target_date=target_date,
        symbols=symbols,
        include_market_history=include_market_history,
        deepseek_kwargs=deepseek_kwargs,
        forecasts=forecasts,
    )

    simulator = AgentSimulator(
        market_data=market_data,
        account_snapshot=account_snapshot,
        starting_cash=starting_cash if starting_cash is not None else account_snapshot.cash,
    )
    strategy_list = list(strategies) if strategies is not None else _default_strategies()
    simulation = simulator.simulate([plan], strategies=strategy_list)
    return DeepSeekNeuralPlanResult(
        plan=plan,
        raw_response=raw_text,
        forecasts=resolved_forecasts,
        simulation=simulation,
    )


__all__ = [
    "DeepSeekNeuralPlanResult",
    "generate_deepseek_neural_plan",
    "simulate_deepseek_neural_plan",
]
