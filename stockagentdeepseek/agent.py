"""High-level utilities for generating and simulating DeepSeek trading plans."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from loguru import logger
from deepseek_wrapper import call_deepseek_chat
from stockagent.agentsimulator.data_models import (
    AccountPosition,
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


def _snapshot_equity(snapshot: AccountSnapshot) -> float:
    cash = float(snapshot.cash or 0.0)
    position_value = 0.0
    for position in getattr(snapshot, "positions", []):
        market_value = getattr(position, "market_value", None)
        if market_value is None:
            avg_price = float(getattr(position, "avg_entry_price", 0.0) or 0.0)
            quantity = float(getattr(position, "quantity", 0.0) or 0.0)
            market_value = avg_price * quantity
        position_value += float(market_value or 0.0)
    total = cash + position_value
    if total > 0:
        return total
    equity = getattr(snapshot, "equity", None)
    return float(equity) if equity is not None else total


@dataclass(slots=True)
class DeepSeekPlanResult:
    plan: TradingPlan
    raw_response: str
    simulation: SimulationResult


@dataclass(slots=True)
class DeepSeekPlanStep:
    date: date
    plan: TradingPlan
    raw_response: str
    simulation: SimulationResult
    starting_equity: float
    ending_equity: float
    daily_return_pct: float


@dataclass(slots=True)
class DeepSeekReplanResult:
    steps: list[DeepSeekPlanStep]
    starting_equity: float
    ending_equity: float
    total_return_pct: float
    annualized_return_pct: float

    def summary(self) -> str:
        lines = [
            "DeepSeek replanning results:",
            f"  Days simulated: {len(self.steps)}",
            f"  Total return: {self.total_return_pct:.2%}",
            f"  Annualized return: {self.annualized_return_pct:.2%}",
        ]
        for step in self.steps:
            lines.append(
                f"  {step.date.isoformat()}: daily return {step.daily_return_pct:.3%}, "
                f"realized PnL ${step.simulation.realized_pnl:,.2f}"
            )
        return "\n".join(lines)


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


def _snapshot_from_simulation(
    *,
    previous_snapshot: AccountSnapshot,
    simulation: SimulationResult,
    snapshot_date: date,
) -> AccountSnapshot:
    """Build a lightweight account snapshot for the next planning round."""
    positions: list[AccountPosition] = []
    for symbol, payload in simulation.final_positions.items():
        quantity = float(payload.get("quantity", 0.0) or 0.0)
        if quantity == 0:
            continue
        avg_price = float(payload.get("avg_price", 0.0) or 0.0)
        side = "long" if quantity >= 0 else "short"
        market_value = quantity * avg_price
        positions.append(
            AccountPosition(
                symbol=symbol.upper(),
                quantity=quantity,
                side=side,
                market_value=market_value,
                avg_entry_price=avg_price,
                unrealized_pl=0.0,
                unrealized_plpc=0.0,
            )
        )

    timestamp = datetime.combine(snapshot_date, datetime.min.time()).replace(tzinfo=timezone.utc)
    return AccountSnapshot(
        equity=simulation.ending_equity,
        cash=simulation.ending_cash,
        buying_power=simulation.ending_equity,
        timestamp=timestamp,
        positions=positions,
    )


def simulate_deepseek_replanning(
    *,
    market_data_by_date: Mapping[date, MarketDataBundle] | Iterable[tuple[date, MarketDataBundle]],
    account_snapshot: AccountSnapshot,
    target_dates: Sequence[date],
    symbols: Sequence[str] | None = None,
    include_market_history: bool = True,
    deepseek_kwargs: Mapping[str, Any] | None = None,
    strategies: Sequence[BaseRiskStrategy] | None = None,
) -> DeepSeekReplanResult:
    """Iteratively generate DeepSeek plans for each date, updating the portfolio state."""
    if not target_dates:
        raise ValueError("target_dates must not be empty.")

    if isinstance(market_data_by_date, Mapping):
        data_lookup: Mapping[date, MarketDataBundle] = market_data_by_date
    else:
        data_lookup = {key: value for key, value in market_data_by_date}

    current_snapshot = account_snapshot
    steps: list[DeepSeekPlanStep] = []
    initial_equity = _snapshot_equity(account_snapshot)

    for current_date in target_dates:
        bundle = data_lookup.get(current_date)
        if bundle is None:
            raise KeyError(f"No market data bundle provided for {current_date}.")

        starting_equity = _snapshot_equity(current_snapshot)

        plan_result = simulate_deepseek_plan(
            market_data=bundle,
            account_snapshot=current_snapshot,
            target_date=current_date,
            symbols=symbols,
            include_market_history=include_market_history,
            deepseek_kwargs=deepseek_kwargs,
            strategies=strategies,
            starting_cash=current_snapshot.cash,
        )
        ending_equity = plan_result.simulation.ending_equity
        if starting_equity and starting_equity > 0:
            daily_return_pct = (ending_equity - starting_equity) / starting_equity
        else:
            daily_return_pct = 0.0
        logger.info(
            "DeepSeek plan %s: realized PnL $%.2f (daily return %.3f%%)",
            current_date.isoformat(),
            plan_result.simulation.realized_pnl,
            daily_return_pct * 100,
        )

        steps.append(
            DeepSeekPlanStep(
                date=current_date,
                plan=plan_result.plan,
                raw_response=plan_result.raw_response,
                simulation=plan_result.simulation,
                starting_equity=starting_equity,
                ending_equity=ending_equity,
                daily_return_pct=daily_return_pct,
            )
        )
        current_snapshot = _snapshot_from_simulation(
            previous_snapshot=current_snapshot,
            simulation=plan_result.simulation,
            snapshot_date=current_date,
        )

    final_equity = steps[-1].ending_equity if steps else initial_equity
    if initial_equity and initial_equity > 0:
        total_return_pct = (final_equity - initial_equity) / initial_equity
    else:
        total_return_pct = 0.0
    day_count = len(steps)
    if day_count > 0 and initial_equity > 0 and final_equity > 0:
        growth = final_equity / initial_equity
        if growth > 0:
            annualized_return_pct = growth ** (365 / day_count) - 1
        else:
            annualized_return_pct = 0.0
    else:
        annualized_return_pct = 0.0
    logger.info(
        "DeepSeek replanning summary: total return %.3f%%, annualized %.3f%% over %d days",
        total_return_pct * 100,
        annualized_return_pct * 100,
        day_count,
    )
    return DeepSeekReplanResult(
        steps=steps,
        starting_equity=initial_equity,
        ending_equity=final_equity,
        total_return_pct=total_return_pct,
        annualized_return_pct=annualized_return_pct,
    )


__all__ = [
    "DeepSeekPlanResult",
    "DeepSeekPlanStep",
    "DeepSeekReplanResult",
    "generate_deepseek_plan",
    "simulate_deepseek_plan",
    "simulate_deepseek_replanning",
]
