"""Prompt construction helpers for the stateful agent."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import date, datetime, timedelta, timezone
from typing import Any

from loguru import logger

from .account_state import get_account_snapshot
from .market_data import MarketDataBundle
from ..constants import DEFAULT_SYMBOLS, SIMULATION_DAYS, TRADING_FEE, CRYPTO_TRADING_FEE
from stock.state import resolve_state_suffix
from stock.state_utils import StateLoadError, load_all_state


SYSTEM_PROMPT = (
    "You are GPT-5, a cautious equities and crypto execution planner that always replies using the enforced JSON schema."
)


def plan_response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "plan": {
                "type": "object",
                "properties": {
                    "target_date": {"type": "string", "format": "date"},
                    "instructions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "symbol": {"type": "string"},
                                "action": {"type": "string", "enum": ["buy", "sell", "exit", "hold"]},
                                "quantity": {"type": "number", "minimum": 0},
                                "execution_session": {"type": "string", "enum": ["market_open", "market_close"]},
                                "entry_price": {"type": ["number", "null"]},
                                "exit_price": {"type": ["number", "null"]},
                                "exit_reason": {"type": ["string", "null"]},
                                "notes": {"type": ["string", "null"]},
                            },
                            "required": [
                                "symbol",
                                "action",
                                "quantity",
                                "execution_session",
                                "entry_price",
                                "exit_price",
                                "exit_reason",
                                "notes",
                            ],
                            "additionalProperties": False,
                        },
                    },
                    "risk_notes": {"type": ["string", "null"]},
                    "focus_symbols": {"type": "array", "items": {"type": "string"}},
                    "stop_trading_symbols": {"type": "array", "items": {"type": "string"}},
                    "execution_window": {"type": "string", "enum": ["market_open", "market_close"]},
                    "metadata": {"type": "object"},
                },
                "required": ["target_date", "instructions"],
                "additionalProperties": False,
            },
            "commentary": {"type": ["string", "null"]},
        },
        "required": ["plan"],
        "additionalProperties": False,
    }


def _parse_timestamp(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _symbol_close_price(symbol: str, market_data: MarketDataBundle) -> float | None:
    frame = market_data.get_symbol_bars(symbol)
    if frame.empty:
        return None
    try:
        return float(frame["close"].iloc[-1])
    except (KeyError, IndexError, ValueError, TypeError):
        pass
    # Fall back to the last available numeric column if `close` is missing.
    for column in ("adj_close", "Adj Close", "Close"):
        if column in frame.columns:
            try:
                return float(frame[column].iloc[-1])
            except (IndexError, ValueError, TypeError):
                continue
    return None


def _summarize_recent_losses(
    *,
    state_suffix: str,
    window: timedelta,
    limit: int = 4,
) -> list[str]:
    try:
        state = load_all_state(state_suffix)
    except StateLoadError as exc:
        logger.debug("Skipping loss summary; state load failed: %s", exc)
        return []

    history = state.get("trade_history", {})
    if not isinstance(history, dict) or not history:
        return []

    cutoff = datetime.now(timezone.utc) - window
    per_symbol: dict[str, dict[str, float]] = {}

    for key, entries in history.items():
        if not isinstance(entries, list):
            continue
        symbol = key.split("|", 1)[0].upper()
        bucket = per_symbol.setdefault(symbol, {"pnl": 0.0, "trades": 0.0})
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            closed_at = _parse_timestamp(entry.get("closed_at"))
            if closed_at is None or closed_at < cutoff:
                continue
            try:
                pnl = float(entry.get("pnl", 0.0) or 0.0)
            except (TypeError, ValueError):
                pnl = 0.0
            bucket["pnl"] += pnl
            bucket["trades"] += 1

    negatives = [
        (symbol, stats["pnl"], int(stats["trades"]))
        for symbol, stats in per_symbol.items()
        if stats["pnl"] < 0.0 and stats["trades"] > 0
    ]
    negatives.sort(key=lambda item: item[1])

    lines: list[str] = []
    for symbol, pnl, trades in negatives[:limit]:
        lines.append(f"{symbol}: ${pnl:,.0f} across {trades} trades (last {window.days}d)")
    return lines


def _summarize_active_exposure(
    *,
    state_suffix: str,
    market_data: MarketDataBundle,
    notional_cap: float,
    limit: int = 4,
) -> list[str]:
    try:
        state = load_all_state(state_suffix)
    except StateLoadError:
        return []

    active = state.get("active_trades", {})
    if not isinstance(active, dict) or not active:
        return []

    exposures: list[tuple[str, str, float, float | None]] = []
    for key, details in active.items():
        if not isinstance(details, dict):
            continue
        symbol = key.split("|", 1)[0].upper()
        mode = str(details.get("mode", "unknown"))
        try:
            qty = float(details.get("qty", 0.0) or 0.0)
        except (TypeError, ValueError):
            qty = 0.0
        price = _symbol_close_price(symbol, market_data)
        notional = abs(qty) * price if price is not None else None
        exposures.append((symbol, mode, qty, notional))

    exposures.sort(key=lambda item: item[3] or 0.0, reverse=True)

    lines: list[str] = []
    for symbol, mode, qty, notional in exposures[:limit]:
        scale = f"≈${notional:,.0f}" if notional is not None else "notional unknown"
        flag = " (above cap!)" if notional is not None and notional > notional_cap else ""
        lines.append(f"{symbol} {mode} qty={qty:.3f} {scale}{flag}")
    return lines


def build_daily_plan_prompt(
    market_data: MarketDataBundle,
    account_payload: dict[str, Any],
    target_date: date,
    symbols: Sequence[str] | None = None,
    include_market_history: bool = True,
) -> tuple[str, dict[str, Any]]:
    symbols = list(symbols) if symbols is not None else list(DEFAULT_SYMBOLS)
    market_payload = market_data.to_payload() if include_market_history else {"symbols": list(symbols)}

    equity = float(account_payload.get("equity") or 0.0)
    max_notional = max(25_000.0, equity * 0.05)
    state_suffix = resolve_state_suffix()
    loss_lines = _summarize_recent_losses(state_suffix=state_suffix, window=timedelta(days=2))
    exposure_lines = _summarize_active_exposure(
        state_suffix=state_suffix,
        market_data=market_data,
        notional_cap=max_notional,
    )

    risk_highlights = ""
    if loss_lines:
        loss_blob = "\n  * ".join(loss_lines)
        risk_highlights += (
            "\n- Recent realized losses demand caution; stay on HOLD or use <=5% probe sizing until the symbol turns profitable:"
            f"\n  * {loss_blob}"
        )
    if exposure_lines:
        exposure_blob = "\n  * ".join(exposure_lines)
        risk_highlights += (
            "\n- Active exposure snapshot (trim these before adding risk elsewhere):"
            f"\n  * {exposure_blob}"
        )

    prompt = f"""
You are a disciplined multi-asset execution planner. Build a one-day trading plan for {target_date.isoformat()}.

Context:
- You may trade the following symbols only: {', '.join(symbols)}.
- Account details include current positions and PnL metrics, but we're operating in an isolated backtest—do not rely on live brokerage data beyond what is provided.
- Historical context: the payload includes the last {market_data.lookback_days} trading days of OHLC percent changes per symbol sourced from trainingdata/.
- Your first task is capital allocation: decide how to distribute available cash across the allowed symbols before issuing trade instructions.
- Plans must respect position sizing, preserve capital and explicitly call out assets to stop trading.
- Valid execution windows are `market_open` (09:30 ET) and `market_close` (16:00 ET). Choose one per instruction.
- Simulation harness will run your plan across {SIMULATION_DAYS} days to evaluate performance.
- Assume round-trip trading fees of {TRADING_FEE:.4%} for equities and {CRYPTO_TRADING_FEE:.4%} for crypto; ensure the plan remains profitable after fees.
- Max notional per new instruction is ${max_notional:,.0f}; smaller is preferred unless conviction is exceptionally high.{risk_highlights}

Structured output requirements:
- Produce JSON matching the provided schema exactly.
- The top-level object must contain only the keys ``plan`` and ``commentary``.
- Use `exit` to close positions you no longer want, specifying the quantity to exit (0 = all) and an `exit_reason`.
- Provide realistic limit prices using `entry_price` / `exit_price` fields reflecting desired fills for the session.
- Include `risk_notes` summarizing risk considerations in under 3 sentences.
- Populate `metadata` with a `capital_allocation_plan` string that explains how cash is apportioned across symbols (list weights or dollar targets).
- Return ONLY the JSON object; do not include markdown or extra fields.
- Every instruction must include values for `entry_price`, `exit_price`, `exit_reason`, and `notes` (use `null` when not applicable).
- Populate `execution_window` to indicate whether trades are intended for market_open or market_close.
""".strip()

    user_payload: dict[str, Any] = {
        "account": account_payload,
        "market_data": market_payload,
        "target_date": target_date.isoformat(),
    }

    return prompt, user_payload


def dump_prompt_package(
    market_data: MarketDataBundle,
    target_date: date,
    include_market_history: bool = True,
) -> dict[str, str]:
    try:
        snapshot = get_account_snapshot()
        account_payload = snapshot.to_payload()
    except Exception as exc:  # pragma: no cover - network/API failure paths
        logger.warning("Falling back to synthetic account snapshot: %s", exc)
        now = datetime.now(timezone.utc)
        account_payload = {
            "equity": 1_000_000.0,
            "cash": 1_000_000.0,
            "buying_power": 1_000_000.0,
            "timestamp": now.isoformat(),
            "positions": [],
        }
    prompt, user_payload = build_daily_plan_prompt(
        market_data=market_data,
        account_payload=account_payload,
        target_date=target_date,
        include_market_history=include_market_history,
    )
    return {
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": prompt,
        "user_payload_json": json.dumps(user_payload, ensure_ascii=False, indent=2),
    }
