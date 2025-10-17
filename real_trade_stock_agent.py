import argparse
import json
from copy import deepcopy
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pytz
from loguru import logger

import alpaca_wrapper
from gpt5_queries import query_gpt5_structured
from src.logging_utils import setup_logging
from src.process_utils import backout_near_market
from stockagent import DEFAULT_SYMBOLS, DEFAULT_REASONING_EFFORT, SIMULATION_DAYS
from stockagent.agentsimulator import (
    AgentSimulator,
    AccountSnapshot,
    ProbeTradeStrategy,
    ProfitShutdownStrategy,
    PlanActionType,
    ExecutionSession,
    TradingInstruction,
    TradingPlan,
    TradingPlanEnvelope,
    fetch_latest_ohlc,
    get_account_snapshot,
    plan_response_schema,
    build_daily_plan_prompt,
    SYSTEM_PROMPT,
)


logger = setup_logging("trade_stock_agent.log")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stateful GPT-5 trading planner.")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--lookback", type=int, default=30, help="Number of recent days used in prompts.")
    parser.add_argument("--live", action="store_true", help="Execute orders.")
    parser.add_argument("--skip-simulation", action="store_true")
    parser.add_argument("--print-json", action="store_true")
    parser.add_argument("--include-history", action="store_true")
    parser.add_argument("--local-data-dir", default="trainingdata", help="Directory containing cached OHLC data.")
    parser.add_argument("--allow-remote-data", action="store_true")
    return parser.parse_args()


def next_trading_day(after: date) -> date:
    d = after + timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def determine_target_date(as_of: datetime) -> date:
    eastern = pytz.timezone("US/Eastern")
    return next_trading_day(as_of.astimezone(eastern).date())


def request_plan(
    symbols: List[str],
    lookback: int,
    include_history: bool,
    local_data_dir: Optional[Path],
    allow_remote: bool,
) -> Tuple[TradingPlanEnvelope, AccountSnapshot]:
    market_data = fetch_latest_ohlc(
        symbols=symbols,
        lookback_days=lookback,
        local_data_dir=local_data_dir,
        allow_remote_download=allow_remote,
    )
    snapshot = get_account_snapshot()
    target_date = determine_target_date(market_data.as_of)

    prompt_text, payload = build_daily_plan_prompt(
        market_data=market_data,
        account_payload=snapshot.to_payload(),
        target_date=target_date,
        symbols=symbols,
        include_market_history=include_history,
    )
    raw_json = query_gpt5_structured(
        system_message=SYSTEM_PROMPT,
        user_prompt=prompt_text,
        response_schema=plan_response_schema(),
        user_payload_json=json.dumps(payload),
        reasoning_effort=DEFAULT_REASONING_EFFORT,
    )
    logger.info(f"GPT raw response: {raw_json}")
    try:
        envelope = TradingPlanEnvelope.from_json(raw_json)
    except ValueError as exc:
        logger.warning(f"Failed to parse GPT response ({exc}); normalizing payload")
        normalized = _normalize_plan_payload(_parse_json_response(raw_json), target_date)
        envelope = TradingPlanEnvelope.from_json(json.dumps(normalized))
    logger.info(f"Received GPT plan for {envelope.plan.target_date.isoformat()}")
    return envelope, snapshot


def _normalize_plan_payload(data: Dict[str, Any], target_date: date) -> Dict[str, Any]:
    plan_block = data.get("plan", {})
    metadata_keys = {"target_date", "instructions", "risk_notes", "focus_symbols", "stop_trading_symbols", "metadata", "execution_window"}

    stop_trading_symbols: List[str] = []

    if isinstance(plan_block, dict) and "instructions" not in plan_block:
        instructions = []
        for symbol, detail in plan_block.items():
            if symbol in metadata_keys or not isinstance(detail, dict):
                continue
            action = detail.get("action", "hold")
            if action == "stop_trading":
                stop_trading_symbols.append(symbol.upper())
                action = "hold"
            instructions.append(_normalize_instruction(detail, symbol, action))
        plan_block = {
            "target_date": plan_block.get("target_date", target_date.isoformat()),
            "instructions": instructions,
            "risk_notes": plan_block.get("risk_notes") or data.get("risk_notes"),
            "focus_symbols": plan_block.get("focus_symbols", []),
            "stop_trading_symbols": plan_block.get("stop_trading_symbols", []) + stop_trading_symbols,
            "metadata": plan_block.get("metadata", {}),
            "execution_window": plan_block.get("execution_window", data.get("execution_window", "market_open")),
        }
    elif isinstance(plan_block, dict):
        plan_block.setdefault("target_date", target_date.isoformat())
        plan_block.setdefault("instructions", [])
        plan_block.setdefault("risk_notes", data.get("risk_notes"))
        plan_block.setdefault("focus_symbols", [])
        plan_block.setdefault("stop_trading_symbols", [])
        plan_block.setdefault("metadata", {})
        plan_block.setdefault("execution_window", data.get("execution_window", "market_open"))
        plan_block["instructions"] = [
            _normalize_instruction(instr, instr.get("symbol"), instr.get("action"))
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
            "execution_window": "market_open",
        }

    plan_block["stop_trading_symbols"] = sorted(set(sym.upper() for sym in plan_block["stop_trading_symbols"]))

    return {
        "plan": plan_block,
        "commentary": data.get("commentary"),
    }


def _normalize_instruction(detail: Dict[str, Any], symbol: str, action: str) -> Dict[str, Any]:
    symbol = str(symbol or detail.get("symbol", "")).upper()
    action = action or detail.get("action", "hold")
    quantity = float(detail.get("quantity", 0) or 0)
    execution_session = detail.get("execution_session", detail.get("execution_window", ExecutionSession.MARKET_OPEN.value))
    entry_price = detail.get("entry_price")
    exit_price = detail.get("exit_price")
    exit_reason = detail.get("exit_reason")
    notes = detail.get("risk_notes") or detail.get("notes")
    return {
        "symbol": symbol,
        "action": action,
        "quantity": quantity,
        "execution_session": execution_session,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "notes": notes,
    }


def _parse_json_response(raw_json: str) -> Dict[str, Any]:
    try:
        return json.loads(raw_json)
    except json.JSONDecodeError:
        first = raw_json.find("{")
        last = raw_json.rfind("}")
        while first != -1 and last != -1 and last > first:
            candidate = raw_json[first : last + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                last = raw_json.rfind("}", 0, last)
        raise ValueError("GPT response was not valid JSON")


def log_plan(envelope: TradingPlanEnvelope) -> None:
    plan = envelope.plan
    logger.info("Plan commentary: %s", envelope.commentary or "No commentary provided")
    for instruction in plan.instructions:
        logger.info(
            "Instruction: %s %s qty=%.4f session=%s entry=%s exit=%s notes=%s",
            instruction.action.value,
            instruction.symbol,
            instruction.quantity,
            instruction.execution_session.value,
            instruction.entry_price,
            instruction.exit_price,
            instruction.notes,
        )
    if plan.stop_trading_symbols:
        logger.info("Stop trading symbols: %s", ", ".join(plan.stop_trading_symbols))
    if plan.focus_symbols:
        logger.info("Focus symbols: %s", ", ".join(plan.focus_symbols))
    if plan.risk_notes:
        logger.info("Risk notes: %s", plan.risk_notes)


def clone_plan_for_days(plan: TradingPlan, trading_days: Iterable[datetime]) -> List[TradingPlan]:
    cloned: List[TradingPlan] = []
    for ts in trading_days:
        target = ts.date() if isinstance(ts, datetime) else ts
        cloned.append(
            TradingPlan(
                target_date=target,
                instructions=[deepcopy(instr) for instr in plan.instructions],
                risk_notes=plan.risk_notes,
                focus_symbols=list(plan.focus_symbols),
                stop_trading_symbols=list(plan.stop_trading_symbols),
                metadata=dict(plan.metadata),
                execution_window=plan.execution_window,
            )
        )
    return cloned


def evaluate_plan(plan: TradingPlan, market_data, snapshot: AccountSnapshot) -> Dict[str, "SimulationResult"]:
    trading_days = market_data.trading_days()
    if not trading_days:
        logger.warning("No trading history available for simulation.")
        return {}
    evaluation_days = trading_days[-SIMULATION_DAYS:]
    scenarios = {
        "baseline": [],
        "probe_trade": [ProbeTradeStrategy()],
        "profit_shutdown": [ProfitShutdownStrategy()],
        "both": [ProbeTradeStrategy(), ProfitShutdownStrategy()],
    }
    results = {}
    for name, strategies in scenarios.items():
        simulator = AgentSimulator(market_data, snapshot)
        plans = clone_plan_for_days(plan, evaluation_days)
        result = simulator.simulate(plans, strategies=strategies)
        results[name] = result
        logger.info(
            "[Simulation:%s] ending_equity=%.2f realized=%.2f unrealized=%.2f fees=%.2f trades=%d",
            name,
            result.ending_equity,
            result.realized_pnl,
            result.unrealized_pnl,
            result.total_fees,
            len(result.trades),
        )
    return results


def execute_plan(plan: TradingPlan, live: bool) -> None:
    for symbol in plan.stop_trading_symbols:
        logger.info("Stopping trading for %s", symbol)
        if live:
            backout_near_market(symbol)

    for instruction in plan.instructions:
        if instruction.action == PlanActionType.HOLD:
            logger.info("Hold %s (qty=%.4f) - no action taken", instruction.symbol, instruction.quantity)
            continue
        if instruction.action == PlanActionType.EXIT:
            logger.info("Exit %s requested qty=%.4f session=%s", instruction.symbol, instruction.quantity, instruction.execution_session.value)
            if live:
                backout_near_market(instruction.symbol)
            continue

        side = "buy" if instruction.action == PlanActionType.BUY else "sell"
        price = _resolve_price(instruction)
        logger.info(
            "Plan execution: %s %s qty=%.4f @ %.2f (%s)",
            side,
            instruction.symbol,
            instruction.quantity,
            price,
            instruction.execution_session.value,
        )
        if live:
            try:
                alpaca_wrapper.open_order_at_price_or_all(
                    instruction.symbol,
                    qty=instruction.quantity,
                    side=side,
                    price=price,
                )
            except Exception as exc:
                logger.error("Failed to submit order for %s: %s", instruction.symbol, exc)


def _resolve_price(instruction: TradingInstruction) -> float:
    if instruction.entry_price is not None:
        return instruction.entry_price
    if instruction.exit_price is not None:
        return instruction.exit_price
    quote = alpaca_wrapper.latest_data(instruction.symbol)
    ask = float(getattr(quote, "ask_price", 0.0) or 0.0)
    bid = float(getattr(quote, "bid_price", 0.0) or 0.0)
    last = float(getattr(quote, "last", 0.0) or getattr(quote, "last_price", 0.0) or 0.0)
    midpoint = (ask + bid) / 2 if ask and bid else 0.0
    price = ask or midpoint or last if instruction.action == PlanActionType.BUY else bid or midpoint or last
    if price <= 0:
        raise ValueError(f"Unable to determine execution price for {instruction.symbol}")
    return price


def main() -> None:
    args = parse_args()
    local_dir = Path(args.local_data_dir) if args.local_data_dir else None
    try:
        envelope, snapshot = request_plan(
            symbols=args.symbols,
            lookback=args.lookback,
            include_history=args.include_history,
            local_data_dir=local_dir,
            allow_remote=args.allow_remote_data,
        )
    except Exception as exc:
        logger.error("Failed to build trading plan: %s", exc)
        raise

    if args.print_json:
        print(envelope.to_json())

    log_plan(envelope)

    if not args.skip_simulation:
        try:
            market_data = fetch_latest_ohlc(
                symbols=args.symbols,
                lookback_days=args.lookback,
                local_data_dir=local_dir,
                allow_remote_download=args.allow_remote_data,
            )
            evaluate_plan(envelope.plan, market_data, snapshot)
        except Exception as exc:
            logger.error("Simulation step failed: %s", exc)

    execute_plan(envelope.plan, live=args.live)


if __name__ == "__main__":
    main()
