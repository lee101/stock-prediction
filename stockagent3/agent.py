"""Two-stage LLM portfolio allocator + rebalance planner for stockagent3."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, TYPE_CHECKING

import anthropic
from loguru import logger

from stockagent_pctline.data_formatter import PctLineData, format_multi_symbol_data
from src.symbol_utils import is_crypto_symbol

from .async_api import _get_api_key, async_api_call, async_api_call_with_thinking, parse_json_robust
if TYPE_CHECKING:  # pragma: no cover - typing only
    from .forecaster import Chronos2Forecast, Chronos2Forecaster


MODEL_SONNET = "claude-sonnet-4-20250514"
MODEL_OPUS = "claude-opus-4-5-20251101"

MODEL_ALLOC = MODEL_OPUS
MODEL_REFINE = MODEL_OPUS

THINKING_BUDGET = 60000


PORTFOLIO_SYSTEM_PROMPT = """You are an aggressive quantitative trader maximizing multi-day PnL.

INPUT:
- Historical price movements in BASIS POINTS (bps): close_bps, high_bps, low_bps per day
- Chronos2 forecasts (expected return and range)
- Current portfolio value
- Asset class indicator (crypto vs stock)
- Day of week and timestamp

GOAL: Allocate the entire portfolio to maximize expected return with controlled risk.

OUTPUT JSON FORMAT:
{
  "overall_confidence": 0.0-1.0,
  "reasoning": "brief explanation",
  "allocations": {
    "SYMBOL_X": {
      "alloc": 0.0-1.0,  // portfolio fraction (0 = skip)
      "direction": "long" or "short",
      "confidence": 0.0-1.0,
      "rationale": "1-line explanation",
      "leverage": 1.0-2.0  // stocks only (crypto must be 1.0)
    }
  }
}

TRADING RULES:
- Allocate across symbols but keep sum(alloc) <= 1.0
- Minimum 5% allocation to any symbol you trade
- Crypto: LONG ONLY (no shorting), leverage fixed at 1.0
- Stocks: can short and use leverage up to 2x
- If expected move < 30 bps, skip the symbol
- Shorting is higher risk; reserve for strong bearish setups

RESPOND WITH JSON ONLY."""


REFINE_SYSTEM_PROMPT = """You are a portfolio rebalancer optimizing risk/reward over 1-8 trading days.

INPUT:
- Initial allocation suggestions
- Chronos2 forecasts and recent price action in bps
- Current open positions (if any)

TASK:
Return a refined TARGET portfolio snapshot with per-symbol trade plans including entry, exit, and expiry.

OUTPUT JSON FORMAT:
{
  "overall_confidence": 0.0-1.0,
  "risk_notes": "brief notes",
  "positions": [
    {
      "symbol": "SYMBOL",
      "target_alloc": 0.0-1.0,
      "direction": "long" or "short",
      "leverage": 1.0-2.0,
      "entry_bps": int,        // bps from last close
      "exit_bps": int,         // bps from last close
      "stop_bps": int | null,  // optional stop (bps)
      "entry_mode": "watch" | "ramp" | "market",
      "entry_expiry_days": 1-8,
      "hold_expiry_days": 1-8,
      "confidence": 0.0-1.0,
      "rationale": "1-line explanation"
    }
  ]
}

RULES:
- target_alloc sum <= 1.0
- Crypto LONG ONLY; leverage=1.0
- Stocks can short; leverage up to 2x
- entry_expiry_days <= hold_expiry_days
- Use entry_mode "watch" for limit entries, "ramp" for immediate staged entry
- For crypto, ensure exit_bps >= +30 for longs (fees)
- For shorts, exit_bps must be negative (cover lower)

RESPOND WITH JSON ONLY."""


@dataclass
class SymbolAllocation:
    symbol: str
    alloc: float
    direction: str
    confidence: float
    rationale: str
    leverage: float = 1.0


@dataclass
class PortfolioAllocation:
    overall_confidence: float
    reasoning: str
    allocations: Dict[str, SymbolAllocation]

    def should_trade(self, min_confidence: float = 0.3) -> bool:
        return self.overall_confidence >= min_confidence and bool(self.allocations)


@dataclass
class TradePosition:
    symbol: str
    target_alloc: float
    direction: str
    leverage: float
    entry_bps: int
    exit_bps: int
    stop_bps: Optional[int]
    entry_mode: str
    entry_expiry_days: int
    hold_expiry_days: int
    confidence: float
    rationale: str
    last_close: float
    entry_price: float
    exit_price: float
    stop_price: Optional[float]


@dataclass
class TradePlan:
    overall_confidence: float
    risk_notes: str
    positions: Sequence[TradePosition]


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _normalize_direction(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw.startswith("s"):
        return "short"
    return "long"


def _normalize_entry_mode(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"ramp", "ramp_in", "ramp-into", "ramp_into"}:
        return "ramp"
    if raw in {"market", "mkt", "immediate"}:
        return "market"
    if raw in {"limit", "watch", "watcher"}:
        return "watch"
    return "watch"


def _bps_from_price(last_close: float, price: float) -> Optional[int]:
    if last_close <= 0 or price <= 0:
        return None
    return int(round((price / last_close - 1.0) * 10000))


def _price_from_bps(last_close: float, bps: int) -> float:
    return float(last_close) * (1.0 + bps / 10000.0)


def _scale_allocations(allocations: Dict[str, SymbolAllocation]) -> Dict[str, SymbolAllocation]:
    total = sum(max(0.0, alloc.alloc) for alloc in allocations.values())
    if total <= 1.0 or total <= 0:
        return allocations
    scale = 1.0 / total
    scaled: Dict[str, SymbolAllocation] = {}
    for symbol, alloc in allocations.items():
        scaled[symbol] = SymbolAllocation(
            symbol=symbol,
            alloc=alloc.alloc * scale,
            direction=alloc.direction,
            confidence=alloc.confidence,
            rationale=alloc.rationale,
            leverage=alloc.leverage,
        )
    return scaled


def _scale_positions(positions: Sequence[TradePosition]) -> Sequence[TradePosition]:
    total = sum(max(0.0, pos.target_alloc) for pos in positions)
    if total <= 1.0 or total <= 0:
        return positions
    scale = 1.0 / total
    scaled: list[TradePosition] = []
    for pos in positions:
        scaled.append(
            TradePosition(
                symbol=pos.symbol,
                target_alloc=pos.target_alloc * scale,
                direction=pos.direction,
                leverage=pos.leverage,
                entry_bps=pos.entry_bps,
                exit_bps=pos.exit_bps,
                stop_bps=pos.stop_bps,
                entry_mode=pos.entry_mode,
                entry_expiry_days=pos.entry_expiry_days,
                hold_expiry_days=pos.hold_expiry_days,
                confidence=pos.confidence,
                rationale=pos.rationale,
                last_close=pos.last_close,
                entry_price=pos.entry_price,
                exit_price=pos.exit_price,
                stop_price=pos.stop_price,
            )
        )
    return scaled


def _format_forecast_summary(forecast: "Chronos2Forecast") -> str:
    exp_bps = forecast.expected_return_pct * 10000
    vol_bps = forecast.volatility_range_pct * 10000
    return (
        f"expected_return={exp_bps:+.0f}bps, "
        f"range={vol_bps:.0f}bps, "
        f"pred_close={forecast.predicted_close:.4f}"
    )


def _pct_lines_to_bps(lines: str, max_lines: int) -> str:
    rows = []
    for line in lines.split("\n"):
        if not line.strip():
            continue
        parts = line.strip().split(",")
        if len(parts) < 3:
            continue
        try:
            close_bps = float(parts[0]) * 100
            high_bps = float(parts[1]) * 100
            low_bps = float(parts[2]) * 100
        except ValueError:
            continue
        rows.append(f"{close_bps:+.0f},{high_bps:+.0f},{low_bps:+.0f}")
    if max_lines > 0 and len(rows) > max_lines:
        rows = rows[-max_lines:]
    return "\n".join(rows)


def build_allocation_prompt(
    *,
    pct_data: Mapping[str, PctLineData],
    forecasts: Mapping[str, Chronos2Forecast],
    portfolio_value: float,
    max_lines: int = 200,
) -> str:
    parts = [
        f"Timestamp: {datetime.utcnow().isoformat()}Z",
        f"Portfolio value: ${portfolio_value:,.2f}",
        "",
    ]

    for symbol, pct in pct_data.items():
        asset_type = "CRYPTO" if is_crypto_symbol(symbol) else "STOCK"
        parts.append(f"## {symbol} ({asset_type})")
        if symbol in forecasts:
            parts.append(f"Forecast: {_format_forecast_summary(forecasts[symbol])}")
        parts.append(f"Last close: {pct.last_close:.4f}")
        if pct.lines:
            parts.append("History (close_bps,high_bps,low_bps):")
            parts.append(_pct_lines_to_bps(pct.lines, max_lines=max_lines))
        parts.append("")

    parts.append("Return JSON only.")
    return "\n".join(parts)


def _format_current_portfolio(positions: Sequence[Mapping[str, Any]]) -> str:
    if not positions:
        return "None"
    lines = []
    for pos in positions:
        symbol = str(pos.get("symbol") or "").upper()
        side = str(pos.get("side") or pos.get("direction") or "").lower()
        qty = _coerce_float(pos.get("qty") or pos.get("quantity"), default=0.0)
        avg = _coerce_float(pos.get("avg_price") or pos.get("avg_entry_price"), default=0.0)
        held = pos.get("days_held")
        expiry = pos.get("expiry_date")
        line = f"{symbol} {side} qty={qty:.4f} avg={avg:.4f}"
        if held is not None:
            line += f" days_held={held}"
        if expiry:
            line += f" expiry={expiry}"
        lines.append(line)
    return "\n".join(lines)


def build_refine_prompt(
    *,
    pct_data: Mapping[str, PctLineData],
    forecasts: Mapping[str, Chronos2Forecast],
    allocation: PortfolioAllocation,
    current_positions: Sequence[Mapping[str, Any]],
    portfolio_value: float,
    max_lines: int = 120,
) -> str:
    payload = {
        "overall_confidence": allocation.overall_confidence,
        "reasoning": allocation.reasoning,
        "allocations": {
            symbol: {
                "alloc": alloc.alloc,
                "direction": alloc.direction,
                "confidence": alloc.confidence,
                "leverage": alloc.leverage,
                "rationale": alloc.rationale,
            }
            for symbol, alloc in allocation.allocations.items()
        },
    }

    parts = [
        f"Timestamp: {datetime.utcnow().isoformat()}Z",
        f"Portfolio value: ${portfolio_value:,.2f}",
        "",
        "Current portfolio:",
        _format_current_portfolio(current_positions),
        "",
        "Initial allocations:",
        json.dumps(payload, indent=2, sort_keys=True),
        "",
    ]

    for symbol, pct in pct_data.items():
        asset_type = "CRYPTO" if is_crypto_symbol(symbol) else "STOCK"
        parts.append(f"## {symbol} ({asset_type})")
        if symbol in forecasts:
            parts.append(f"Forecast: {_format_forecast_summary(forecasts[symbol])}")
        parts.append(f"Last close: {pct.last_close:.4f}")
        if pct.lines:
            parts.append("History (close_bps,high_bps,low_bps):")
            parts.append(_pct_lines_to_bps(pct.lines, max_lines=max_lines))
        parts.append("")

    parts.append("Return JSON only.")
    return "\n".join(parts)


def _parse_allocations_response(
    raw: str,
    *,
    symbols: Iterable[str],
) -> PortfolioAllocation:
    payload = parse_json_robust(raw)
    overall_confidence = _clamp(_coerce_float(payload.get("overall_confidence"), 0.0), 0.0, 1.0)
    reasoning = str(payload.get("reasoning") or "").strip()

    allocs_raw = payload.get("allocations")
    allocations: Dict[str, SymbolAllocation] = {}
    symbol_set = {s.upper() for s in symbols}

    if isinstance(allocs_raw, dict):
        items = allocs_raw.items()
    elif isinstance(allocs_raw, list):
        items = ((item.get("symbol"), item) for item in allocs_raw if isinstance(item, dict))
    else:
        items = []

    for symbol, data in items:
        if not symbol:
            continue
        sym = str(symbol).upper()
        if sym not in symbol_set:
            continue
        if not isinstance(data, dict):
            continue
        alloc = _clamp(_coerce_float(data.get("alloc"), 0.0), 0.0, 1.0)
        if alloc < 0.05:
            continue
        direction = _normalize_direction(data.get("direction"))
        confidence = _clamp(_coerce_float(data.get("confidence"), 0.0), 0.0, 1.0)
        rationale = str(data.get("rationale") or "").strip()
        leverage = _clamp(_coerce_float(data.get("leverage"), 1.0), 1.0, 2.0)

        if is_crypto_symbol(sym):
            if direction == "short":
                continue
            leverage = 1.0

        allocations[sym] = SymbolAllocation(
            symbol=sym,
            alloc=alloc,
            direction=direction,
            confidence=confidence,
            rationale=rationale,
            leverage=leverage,
        )

    allocations = _scale_allocations(allocations)
    return PortfolioAllocation(
        overall_confidence=overall_confidence,
        reasoning=reasoning,
        allocations=allocations,
    )


def _parse_trade_plan_response(
    raw: str,
    *,
    symbols: Iterable[str],
    pct_data: Mapping[str, PctLineData],
) -> TradePlan:
    payload = parse_json_robust(raw)
    overall_confidence = _clamp(_coerce_float(payload.get("overall_confidence"), 0.0), 0.0, 1.0)
    risk_notes = str(payload.get("risk_notes") or "").strip()

    positions_raw = payload.get("positions")
    if not positions_raw and isinstance(payload.get("portfolio"), list):
        positions_raw = payload.get("portfolio")
    if not positions_raw and isinstance(payload.get("allocations"), list):
        positions_raw = payload.get("allocations")

    symbol_set = {s.upper() for s in symbols}
    positions: list[TradePosition] = []

    if isinstance(positions_raw, dict):
        items = [
            {"symbol": key, **val}
            for key, val in positions_raw.items()
            if isinstance(val, dict)
        ]
    elif isinstance(positions_raw, list):
        items = [item for item in positions_raw if isinstance(item, dict)]
    else:
        items = []

    for item in items:
        symbol = str(item.get("symbol") or "").upper()
        if not symbol or symbol not in symbol_set:
            continue
        last_close = pct_data.get(symbol).last_close if symbol in pct_data else 0.0
        if last_close <= 0:
            continue

        target_alloc = _clamp(_coerce_float(item.get("target_alloc") or item.get("alloc"), 0.0), 0.0, 1.0)
        if target_alloc < 0.05:
            continue

        direction = _normalize_direction(item.get("direction"))
        leverage = _clamp(_coerce_float(item.get("leverage"), 1.0), 1.0, 2.0)
        entry_mode = _normalize_entry_mode(item.get("entry_mode"))
        entry_expiry_days = _clamp(_coerce_int(item.get("entry_expiry_days"), 1), 1, 8)
        hold_expiry_days = _clamp(_coerce_int(item.get("hold_expiry_days"), 3), 1, 8)
        confidence = _clamp(_coerce_float(item.get("confidence"), 0.0), 0.0, 1.0)
        rationale = str(item.get("rationale") or "").strip()

        if entry_expiry_days > hold_expiry_days:
            entry_expiry_days = hold_expiry_days

        if is_crypto_symbol(symbol):
            if direction == "short":
                continue
            leverage = 1.0

        entry_bps = item.get("entry_bps")
        exit_bps = item.get("exit_bps")
        stop_bps = item.get("stop_bps")

        entry_price_raw = item.get("entry_price")
        exit_price_raw = item.get("exit_price")
        stop_price_raw = item.get("stop_price")

        if entry_bps is None and entry_price_raw is not None:
            entry_bps = _bps_from_price(last_close, _coerce_float(entry_price_raw, 0.0))
        if exit_bps is None and exit_price_raw is not None:
            exit_bps = _bps_from_price(last_close, _coerce_float(exit_price_raw, 0.0))
        if stop_bps is None and stop_price_raw is not None:
            stop_bps = _bps_from_price(last_close, _coerce_float(stop_price_raw, 0.0))

        entry_bps_val = _coerce_int(entry_bps, 0)
        exit_bps_val = _coerce_int(exit_bps, 0)
        stop_bps_val = _coerce_int(stop_bps, 0) if stop_bps is not None else None

        if direction == "long":
            if entry_bps_val > 0:
                entry_bps_val = 0
            if exit_bps_val <= 0:
                exit_bps_val = max(30, abs(exit_bps_val))
            if stop_bps_val is not None and stop_bps_val >= 0:
                stop_bps_val = -abs(stop_bps_val)
        else:
            if entry_bps_val < 0:
                entry_bps_val = 0
            if exit_bps_val >= 0:
                exit_bps_val = -max(30, abs(exit_bps_val))
            if stop_bps_val is not None and stop_bps_val <= 0:
                stop_bps_val = abs(stop_bps_val)

        if is_crypto_symbol(symbol) and direction == "long" and exit_bps_val < 30:
            exit_bps_val = 30

        entry_price = _price_from_bps(last_close, entry_bps_val)
        exit_price = _price_from_bps(last_close, exit_bps_val)
        stop_price = _price_from_bps(last_close, stop_bps_val) if stop_bps_val is not None else None

        positions.append(
            TradePosition(
                symbol=symbol,
                target_alloc=target_alloc,
                direction=direction,
                leverage=leverage,
                entry_bps=entry_bps_val,
                exit_bps=exit_bps_val,
                stop_bps=stop_bps_val,
                entry_mode=entry_mode,
                entry_expiry_days=int(entry_expiry_days),
                hold_expiry_days=int(hold_expiry_days),
                confidence=confidence,
                rationale=rationale,
                last_close=float(last_close),
                entry_price=entry_price,
                exit_price=exit_price,
                stop_price=stop_price,
            )
        )

    positions = list(_scale_positions(positions))
    return TradePlan(
        overall_confidence=overall_confidence,
        risk_notes=risk_notes,
        positions=positions,
    )


async def async_generate_trade_plan(
    *,
    symbols: Sequence[str],
    market_data: Mapping[str, Any],
    portfolio_value: float,
    current_positions: Sequence[Mapping[str, Any]],
    max_lines: int = 160,
    use_thinking: bool = False,
    chronos_frequency: Optional[str] = None,
    client: Optional[anthropic.AsyncAnthropic] = None,
    close_client: bool = True,
) -> TradePlan:
    symbols = [str(s).upper() for s in symbols]
    pct_data = format_multi_symbol_data(market_data, max_days=max_lines)
    from .forecaster import Chronos2Forecaster

    forecaster = Chronos2Forecaster()
    forecasts = forecaster.forecast_all(
        market_data=market_data,
        symbols=symbols,
        frequency=chronos_frequency,
    )

    alloc_prompt = build_allocation_prompt(
        pct_data=pct_data,
        forecasts=forecasts,
        portfolio_value=portfolio_value,
        max_lines=max_lines,
    )

    client_provided = client is not None
    if client is None:
        client = anthropic.AsyncAnthropic(api_key=_get_api_key())
    try:
        if use_thinking:
            alloc_raw = await async_api_call_with_thinking(
                client,
                model=MODEL_ALLOC,
                system=PORTFOLIO_SYSTEM_PROMPT,
                user_prompt=alloc_prompt,
                max_tokens=64000,
                thinking_budget=THINKING_BUDGET,
            )
        else:
            alloc_raw = await async_api_call(
                client,
                model=MODEL_ALLOC,
                system=PORTFOLIO_SYSTEM_PROMPT,
                user_prompt=alloc_prompt,
                max_tokens=4096,
            )

        allocation = _parse_allocations_response(alloc_raw, symbols=symbols)

        refine_prompt = build_refine_prompt(
            pct_data=pct_data,
            forecasts=forecasts,
            allocation=allocation,
            current_positions=current_positions,
            portfolio_value=portfolio_value,
            max_lines=max_lines,
        )

        if use_thinking:
            refine_raw = await async_api_call_with_thinking(
                client,
                model=MODEL_REFINE,
                system=REFINE_SYSTEM_PROMPT,
                user_prompt=refine_prompt,
                max_tokens=64000,
                thinking_budget=THINKING_BUDGET,
            )
        else:
            refine_raw = await async_api_call(
                client,
                model=MODEL_REFINE,
                system=REFINE_SYSTEM_PROMPT,
                user_prompt=refine_prompt,
                max_tokens=4096,
            )

        plan = _parse_trade_plan_response(refine_raw, symbols=symbols, pct_data=pct_data)
        if plan.positions:
            logger.info("Generated trade plan with {} positions", len(plan.positions))
        else:
            logger.warning("Generated trade plan with no positions")
        return plan
    finally:
        if close_client and not client_provided:
            try:
                await client.close()
            except Exception as exc:  # pragma: no cover - best effort cleanup
                logger.debug("Failed closing Anthropic client: {}", exc)


def generate_trade_plan(
    *,
    symbols: Sequence[str],
    market_data: Mapping[str, Any],
    portfolio_value: float,
    current_positions: Sequence[Mapping[str, Any]],
    max_lines: int = 160,
    use_thinking: bool = False,
    chronos_frequency: Optional[str] = None,
) -> TradePlan:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        raise RuntimeError("generate_trade_plan() cannot be called from a running event loop; use async_generate_trade_plan")
    return asyncio.run(
        async_generate_trade_plan(
            symbols=symbols,
            market_data=market_data,
            portfolio_value=portfolio_value,
            current_positions=current_positions,
            max_lines=max_lines,
            use_thinking=use_thinking,
            chronos_frequency=chronos_frequency,
        )
    )


__all__ = [
    "PortfolioAllocation",
    "SymbolAllocation",
    "TradePosition",
    "TradePlan",
    "build_allocation_prompt",
    "build_refine_prompt",
    "async_generate_trade_plan",
    "generate_trade_plan",
]
