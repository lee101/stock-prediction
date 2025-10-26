"""High-level utilities for generating and simulating GPT-5 trading plans."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from loguru import logger

from gpt5_queries import query_gpt5_structured
from stockagent.constants import DEFAULT_REASONING_EFFORT
from stockagent.agentsimulator.data_models import (
    AccountPosition,
    AccountSnapshot,
    ExecutionSession,
    TradingPlan,
    TradingPlanEnvelope,
)
from stockagent.agentsimulator.interfaces import BaseRiskStrategy
from stockagent.agentsimulator.market_data import MarketDataBundle
from stockagent.agentsimulator.prompt_builder import (
    SYSTEM_PROMPT,
    build_daily_plan_prompt,
    plan_response_schema,
)
from stockagent.agentsimulator.risk_strategies import (
    ProfitShutdownStrategy,
    ProbeTradeStrategy,
)
from stockagent.agentsimulator.simulator import AgentSimulator, SimulationResult


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


def _infer_trading_days_per_year(bundles: Sequence[MarketDataBundle]) -> int:
    for bundle in bundles:
        for trading_day in bundle.trading_days():
            try:
                weekday = trading_day.weekday()
            except AttributeError:
                continue
            if weekday >= 5:
                return 365
    return 252


def _parse_json_response(raw_json: str) -> Mapping[str, Any]:
    try:
        return json.loads(raw_json)
    except json.JSONDecodeError:
        first_brace = raw_json.find("{")
        last_brace = raw_json.rfind("}")
        while first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            candidate = raw_json[first_brace : last_brace + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                last_brace = raw_json.rfind("}", 0, last_brace)
        raise ValueError("GPT-5 response did not contain valid JSON.")


def _normalize_instruction(detail: Mapping[str, Any], symbol: str, action: str) -> dict[str, Any]:
    symbol_str = str(symbol or detail.get("symbol", "")).upper()
    action_str = action or str(detail.get("action", "hold"))
    quantity = float(detail.get("quantity", 0.0) or 0.0)
    execution_session = detail.get(
        "execution_session",
        detail.get("execution_window", ExecutionSession.MARKET_OPEN.value),
    )
    entry_price = detail.get("entry_price")
    exit_price = detail.get("exit_price")
    exit_reason = detail.get("exit_reason")
    notes = detail.get("risk_notes") or detail.get("notes")
    return {
        "symbol": symbol_str,
        "action": action_str,
        "quantity": quantity,
        "execution_session": execution_session,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "notes": notes,
    }


def _normalize_plan_payload(data: Mapping[str, Any], target_date: date) -> Mapping[str, Any]:
    plan_source: MutableMapping[str, Any] | None = None
    if isinstance(data, Mapping):
        candidate = data.get("plan")
        if isinstance(candidate, Mapping):
            plan_source = dict(candidate)
        else:
            plan_source = dict(data)
    if plan_source is None:
        plan_source = {}

    metadata_keys = {
        "target_date",
        "instructions",
        "risk_notes",
        "focus_symbols",
        "stop_trading_symbols",
        "metadata",
        "execution_window",
    }
    stop_trading_symbols: list[str] = []

    plan_block: MutableMapping[str, Any] | None = plan_source

    if isinstance(plan_block, dict) and "instructions" not in plan_block:
        instructions: list[dict[str, Any]] = []
        for symbol, detail in list(plan_block.items()):
            if symbol in metadata_keys or not isinstance(detail, Mapping):
                continue
            action = str(detail.get("action", "hold"))
            if action == "stop_trading":
                stop_trading_symbols.append(str(symbol).upper())
                action = "hold"
            instructions.append(_normalize_instruction(detail, str(symbol), action))
        plan_block = {
            "target_date": plan_block.get("target_date", target_date.isoformat()),
            "instructions": instructions,
            "risk_notes": plan_block.get("risk_notes") or data.get("risk_notes"),
            "focus_symbols": plan_block.get("focus_symbols", []),
            "stop_trading_symbols": plan_block.get("stop_trading_symbols", []) + stop_trading_symbols,
            "metadata": plan_block.get("metadata", {}),
            "execution_window": plan_block.get(
                "execution_window",
                data.get("execution_window", ExecutionSession.MARKET_OPEN.value),
            ),
        }
    elif isinstance(plan_block, dict):
        plan_block.setdefault("target_date", target_date.isoformat())
        plan_block.setdefault("instructions", [])
        plan_block.setdefault("risk_notes", data.get("risk_notes"))
        plan_block.setdefault("focus_symbols", [])
        plan_block.setdefault("stop_trading_symbols", [])
        plan_block.setdefault("metadata", {})
        plan_block.setdefault(
            "execution_window",
            data.get("execution_window", ExecutionSession.MARKET_OPEN.value),
        )
        plan_block["instructions"] = [
            _normalize_instruction(instr, str(instr.get("symbol")), str(instr.get("action")))
            if isinstance(instr, Mapping)
            else _normalize_instruction({}, str(instr), "hold")
            for instr in plan_block["instructions"]
        ]
    else:
        plan_block = {
            "target_date": target_date.isoformat(),
            "instructions": [],
            "risk_notes": data.get("risk_notes"),
            "focus_symbols": [],
            "stop_trading_symbols": [],
            "metadata": {},
            "execution_window": ExecutionSession.MARKET_OPEN.value,
        }

    plan_block["stop_trading_symbols"] = sorted(
        {str(sym).upper() for sym in plan_block.get("stop_trading_symbols", [])}
    )
    return plan_block


def _parse_envelope(raw_json: str, target_date: date) -> TradingPlanEnvelope:
    try:
        return TradingPlanEnvelope.from_json(raw_json)
    except ValueError:
        normalized = _normalize_plan_payload(_parse_json_response(raw_json), target_date)
        return TradingPlanEnvelope.from_json(json.dumps(normalized))


@dataclass(slots=True)
class StockAgentPlanResult:
    plan: TradingPlan
    raw_response: str
    simulation: SimulationResult


@dataclass(slots=True)
class StockAgentPlanStep:
    date: date
    plan: TradingPlan
    raw_response: str
    simulation: SimulationResult
    starting_equity: float
    ending_equity: float
    daily_return_pct: float


@dataclass(slots=True)
class StockAgentReplanResult:
    steps: list[StockAgentPlanStep]
    starting_equity: float
    ending_equity: float
    total_return_pct: float
    annualized_return_pct: float
    annualization_days: int

    def summary(self) -> str:
        lines = [
            "StockAgent replanning results:",
            f"  Days simulated: {len(self.steps)}",
            f"  Total return: {self.total_return_pct:.2%}",
            f"  Annualized return ({self.annualization_days}d/yr): {self.annualized_return_pct:.2%}",
        ]
        for idx, step in enumerate(self.steps, start=1):
            lines.append(
                f"  Step {idx}: daily return {step.daily_return_pct:.3%}, "
                f"realized PnL ${step.simulation.realized_pnl:,.2f}"
            )
        return "\n".join(lines)


def generate_stockagent_plan(
    *,
    market_data: MarketDataBundle,
    account_snapshot: AccountSnapshot,
    target_date: date,
    symbols: Sequence[str] | None = None,
    include_market_history: bool = True,
    reasoning_effort: str | None = None,
    gpt_kwargs: Mapping[str, Any] | None = None,
) -> tuple[TradingPlanEnvelope, str]:
    """Request a trading plan from GPT-5 and parse the structured response."""
    prompt_text, payload = build_daily_plan_prompt(
        market_data=market_data,
        account_payload=account_snapshot.to_payload(),
        target_date=target_date,
        symbols=symbols,
        include_market_history=include_market_history,
    )
    kwargs: MutableMapping[str, Any] = dict(gpt_kwargs or {})
    kwargs.setdefault("reasoning_effort", reasoning_effort or DEFAULT_REASONING_EFFORT)
    raw_text = query_gpt5_structured(
        system_message=SYSTEM_PROMPT,
        user_prompt=prompt_text,
        response_schema=plan_response_schema(),
        user_payload_json=json.dumps(payload, ensure_ascii=False),
        **kwargs,
    )
    envelope = _parse_envelope(raw_text, target_date)
    return envelope, raw_text


def simulate_stockagent_plan(
    *,
    market_data: MarketDataBundle,
    account_snapshot: AccountSnapshot,
    target_date: date,
    symbols: Sequence[str] | None = None,
    include_market_history: bool = True,
    reasoning_effort: str | None = None,
    gpt_kwargs: Mapping[str, Any] | None = None,
    strategies: Sequence[BaseRiskStrategy] | None = None,
    starting_cash: float | None = None,
) -> StockAgentPlanResult:
    """Generate a GPT-5 plan and evaluate it with the stock agent simulator."""
    envelope, raw_response = generate_stockagent_plan(
        market_data=market_data,
        account_snapshot=account_snapshot,
        target_date=target_date,
        symbols=symbols,
        include_market_history=include_market_history,
        reasoning_effort=reasoning_effort,
        gpt_kwargs=gpt_kwargs,
    )
    plan = envelope.plan
    simulator = AgentSimulator(
        market_data=market_data,
        account_snapshot=account_snapshot,
        starting_cash=starting_cash if starting_cash is not None else account_snapshot.cash,
    )
    strategy_list = list(strategies) if strategies is not None else _default_strategies()
    simulation = simulator.simulate([plan], strategies=strategy_list)
    return StockAgentPlanResult(plan=plan, raw_response=raw_response, simulation=simulation)


def _snapshot_from_simulation(
    *,
    previous_snapshot: AccountSnapshot,
    simulation: SimulationResult,
    snapshot_date: date,
) -> AccountSnapshot:
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


def simulate_stockagent_replanning(
    *,
    market_data_by_date: Mapping[date, MarketDataBundle] | Iterable[tuple[date, MarketDataBundle]],
    account_snapshot: AccountSnapshot,
    target_dates: Sequence[date],
    symbols: Sequence[str] | None = None,
    include_market_history: bool = True,
    reasoning_effort: str | None = None,
    gpt_kwargs: Mapping[str, Any] | None = None,
    strategies: Sequence[BaseRiskStrategy] | None = None,
    trading_days_per_year: int | None = None,
) -> StockAgentReplanResult:
    """Iteratively generate GPT-5 plans, updating the portfolio snapshot each session."""
    if not target_dates:
        raise ValueError("target_dates must not be empty.")

    if isinstance(market_data_by_date, Mapping):
        data_lookup: Mapping[date, MarketDataBundle] = market_data_by_date
    else:
        data_lookup = {key: value for key, value in market_data_by_date}

    ordered_bundles: list[MarketDataBundle] = [
        data_lookup[plan_date] for plan_date in target_dates if plan_date in data_lookup
    ]
    annualization_days = (
        trading_days_per_year if trading_days_per_year is not None else _infer_trading_days_per_year(ordered_bundles)
    )

    current_snapshot = account_snapshot
    steps: list[StockAgentPlanStep] = []
    initial_equity = _snapshot_equity(account_snapshot)

    for step_index, current_date in enumerate(target_dates, start=1):
        bundle = data_lookup.get(current_date)
        if bundle is None:
            raise KeyError(f"No market data bundle provided for {current_date}.")

        starting_equity = _snapshot_equity(current_snapshot)

        plan_result = simulate_stockagent_plan(
            market_data=bundle,
            account_snapshot=current_snapshot,
            target_date=current_date,
            symbols=symbols,
            include_market_history=include_market_history,
            reasoning_effort=reasoning_effort,
            gpt_kwargs=gpt_kwargs,
            strategies=strategies,
            starting_cash=current_snapshot.cash,
        )
        ending_equity = plan_result.simulation.ending_equity
        if starting_equity and starting_equity > 0:
            daily_return_pct = (ending_equity - starting_equity) / starting_equity
        else:
            daily_return_pct = 0.0
        logger.info(
            f"StockAgent plan step {step_index}: realized PnL ${plan_result.simulation.realized_pnl:,.2f} "
            f"(daily return {daily_return_pct * 100:.3f}%)"
        )

        steps.append(
            StockAgentPlanStep(
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
    annualized_return_pct = 0.0
    if day_count > 0 and initial_equity > 0 and final_equity > 0:
        growth = final_equity / initial_equity
        if growth > 0:
            annualized_return_pct = growth ** (annualization_days / day_count) - 1
    logger.info(
        f"StockAgent replanning summary: total return {total_return_pct * 100:.3f}%, "
        f"annualized {annualized_return_pct * 100:.3f}% over {day_count} sessions "
        f"(annualized with {annualization_days} days/year)"
    )
    return StockAgentReplanResult(
        steps=steps,
        starting_equity=initial_equity,
        ending_equity=final_equity,
        total_return_pct=total_return_pct,
        annualized_return_pct=annualized_return_pct,
        annualization_days=annualization_days,
    )


__all__ = [
    "StockAgentPlanResult",
    "StockAgentPlanStep",
    "StockAgentReplanResult",
    "generate_stockagent_plan",
    "simulate_stockagent_plan",
    "simulate_stockagent_replanning",
]
