import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import pytz
from loguru import logger

from gpt5_queries import query_gpt5_structured
from stockagentindependant import DEFAULT_SYMBOLS
from stockagentindependant.agentsimulator import (
    ExecutionSession,
    TradingPlan,
    TradingPlanEnvelope,
    build_daily_plan_prompt,
    fetch_latest_ohlc,
    plan_response_schema,
    SYSTEM_PROMPT,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stateless GPT-5 trading planner.")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--lookback", type=int, default=30)
    history = parser.add_mutually_exclusive_group()
    history.add_argument("--include-history", dest="include_history", action="store_true", help="Include percent-change history (default).")
    history.add_argument("--no-history", dest="include_history", action="store_false", help="Omit history payload.")
    parser.set_defaults(include_history=True)
    parser.add_argument("--local-data-dir", default="trainingdata", help="Directory containing cached OHLC data.")
    parser.add_argument("--allow-remote-data", action="store_true")
    parser.add_argument("--print-json", action="store_true")
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
) -> TradingPlanEnvelope:
    market_data = fetch_latest_ohlc(
        symbols=symbols,
        lookback_days=lookback,
        local_data_dir=local_data_dir,
        allow_remote_download=allow_remote,
    )
    target_date = determine_target_date(market_data.as_of)

    prompt_text, payload = build_daily_plan_prompt(
        market_data=market_data,
        target_date=target_date,
        symbols=symbols,
        include_market_history=include_history,
    )
    raw_json = query_gpt5_structured(
        system_message=SYSTEM_PROMPT,
        user_prompt=prompt_text,
        response_schema=plan_response_schema(),
        user_payload_json=json.dumps(payload),
    )
    logger.info(f"GPT raw response: {raw_json}")
    try:
        return TradingPlanEnvelope.from_json(raw_json)
    except ValueError as exc:
        logger.warning(f"Failed to parse GPT response ({exc}); normalizing payload")
        normalized = _normalize_plan_payload(_parse_json_response(raw_json), target_date)
        return TradingPlanEnvelope.from_json(json.dumps(normalized))


def _normalize_plan_payload(data: Dict[str, Any], target_date: date) -> Dict[str, Any]:
    plan_source: Dict[str, Any] | None = None
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
    stop_trading_symbols: List[str] = []

    plan_block: Dict[str, Any] | None = plan_source

    if isinstance(plan_block, dict) and "instructions" not in plan_block:
        instructions = []
        for symbol, detail in list(plan_block.items()):
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
            "execution_window": plan_block.get("execution_window", data.get("execution_window", ExecutionSession.MARKET_OPEN.value)),
        }
    elif isinstance(plan_block, dict):
        plan_block.setdefault("target_date", target_date.isoformat())
        plan_block.setdefault("instructions", [])
        plan_block.setdefault("risk_notes", data.get("risk_notes"))
        plan_block.setdefault("focus_symbols", [])
        plan_block.setdefault("stop_trading_symbols", [])
        plan_block.setdefault("metadata", {})
        plan_block.setdefault("execution_window", data.get("execution_window", ExecutionSession.MARKET_OPEN.value))
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
            "execution_window": ExecutionSession.MARKET_OPEN.value,
        }

    plan_block["stop_trading_symbols"] = sorted(set(sym.upper() for sym in plan_block["stop_trading_symbols"]))
    return plan_block


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


def log_plan(plan: TradingPlan) -> None:
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


def main() -> None:
    args = parse_args()
    local_dir = Path(args.local_data_dir) if args.local_data_dir else None
    try:
        envelope = request_plan(
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

    log_plan(envelope.plan)


if __name__ == "__main__":
    main()
