"""High-level utilities for Claude Opus trading plans with Chronos2 forecasting."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from loguru import logger

from stockagent.agentsimulator.data_models import (
    AccountPosition,
    AccountSnapshot,
    TradingPlan,
    TradingPlanEnvelope,
)
from stockagent.agentsimulator.market_data import MarketDataBundle
from stockagentdeepseek_maxdiff.simulator import MaxDiffResult, MaxDiffSimulator

from .prompt_builder import SYSTEM_PROMPT, build_opus_messages
from .forecaster import Chronos2Forecast, generate_chronos2_forecasts
from .models import TradingPlanOutput

# Try to import anthropic for structured outputs
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None

# Cache directory for structured output responses
CACHE_DIR = Path(".opus_chronos2_cache")


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


def _get_cache_key(messages: list[dict], system_prompt: str) -> str:
    """Generate a cache key from messages and system prompt."""
    payload = json.dumps({"messages": messages, "system": system_prompt}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _load_from_cache(cache_key: str) -> dict | None:
    """Load a cached response if available."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        try:
            with open(cache_file) as f:
                data = json.load(f)
            logger.debug(f"Cache hit for key {cache_key[:16]}...")
            return data
        except Exception:
            pass
    return None


def _save_to_cache(cache_key: str, response: dict) -> None:
    """Save a response to cache."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{cache_key}.json"
    try:
        with open(cache_file, "w") as f:
            json.dump(response, f)
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")


def call_opus_structured(
    messages: list[dict],
    system_prompt: str = SYSTEM_PROMPT,
    model: str = "claude-sonnet-4-20250514",
) -> TradingPlanOutput:
    """Call Claude with structured outputs for guaranteed schema compliance."""
    if not ANTHROPIC_AVAILABLE:
        raise RuntimeError("anthropic package not available")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable not set")

    # Check cache first
    cache_key = _get_cache_key(messages, system_prompt)
    cached = _load_from_cache(cache_key)
    if cached and "response" in cached:
        logger.debug("Using cached structured response")
        return TradingPlanOutput.model_validate_json(cached["response"])

    client = anthropic.Anthropic(api_key=api_key)

    try:
        response = client.beta.messages.parse(
            model=model,
            betas=["structured-outputs-2025-11-13"],
            max_tokens=4096,
            system=system_prompt,
            messages=messages,
            output_format=TradingPlanOutput,
        )

        # Get the parsed output
        result = response.parsed

        # Cache the response
        _save_to_cache(cache_key, {
            "response": result.model_dump_json(),
            "model": model,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        })

        return result

    except Exception as e:
        logger.error(f"Structured output call failed: {e}")
        raise


def call_opus_fallback(
    messages: list[dict],
    system_prompt: str = SYSTEM_PROMPT,
    model: str = "claude-sonnet-4-20250514",
    max_retries: int = 8,
) -> dict:
    """Fallback to regular chat API if structured outputs fail."""
    if not ANTHROPIC_AVAILABLE:
        raise RuntimeError("anthropic package not available")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable not set")

    # Check cache first
    cache_key = _get_cache_key(messages, system_prompt)
    cached = _load_from_cache(cache_key)
    if cached and "raw_response" in cached:
        logger.debug("Using cached fallback response")
        return json.loads(cached["raw_response"])

    client = anthropic.Anthropic(api_key=api_key)

    # Retry with exponential backoff for overloaded errors
    last_exception = None
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                system=system_prompt,
                messages=messages,
            )
            break
        except anthropic._exceptions.OverloadedError as e:
            last_exception = e
            wait_time = 5 * (2 ** attempt)  # 5, 10, 20, 40, 80, 160... seconds
            logger.warning(f"API overloaded, attempt {attempt + 1}/{max_retries}, waiting {wait_time}s")
            time.sleep(wait_time)
        except anthropic._exceptions.RateLimitError as e:
            last_exception = e
            wait_time = 5 * (2 ** attempt)
            logger.warning(f"Rate limited, attempt {attempt + 1}/{max_retries}, waiting {wait_time}s")
            time.sleep(wait_time)
    else:
        raise last_exception

    # Extract text content
    raw_text = ""
    for block in response.content:
        if hasattr(block, "text"):
            raw_text += block.text

    # Parse JSON from response
    json_start = raw_text.find("{")
    json_end = raw_text.rfind("}") + 1
    if json_start >= 0 and json_end > json_start:
        json_text = raw_text[json_start:json_end]
    else:
        json_text = raw_text

    parsed = json.loads(json_text)

    # Cache the response
    _save_to_cache(cache_key, {
        "raw_response": json.dumps(parsed),
        "model": model,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    })

    return parsed


@dataclass(slots=True)
class OpusPlanResult:
    """Result from a single Opus plan generation and simulation."""
    plan: TradingPlan
    raw_response: str
    simulation: MaxDiffResult
    chronos2_forecasts: Mapping[str, Chronos2Forecast] | None = None


@dataclass(slots=True)
class OpusPlanStep:
    """A single day's planning step in a multi-day backtest."""
    date: date
    plan: TradingPlan
    raw_response: str
    simulation: MaxDiffResult
    starting_equity: float
    ending_equity: float
    daily_return_pct: float
    chronos2_forecasts: Mapping[str, Chronos2Forecast] | None = None


@dataclass(slots=True)
class OpusReplanResult:
    """Results from iterative multi-day Opus planning."""
    steps: list[OpusPlanStep]
    starting_equity: float
    ending_equity: float
    total_return_pct: float
    annualized_return_pct: float
    annualization_days: int

    def summary(self) -> str:
        lines = [
            "Claude Opus (Chronos2) replanning results:",
            f"  Days simulated: {len(self.steps)}",
            f"  Total return: {self.total_return_pct:.2%}",
            f"  Annualized return ({self.annualization_days}d/yr): {self.annualized_return_pct:.2%}",
        ]
        for idx, step in enumerate(self.steps, start=1):
            lines.append(
                f"  Step {idx}: daily return {step.daily_return_pct:.3%}, "
                f"net PnL ${step.simulation.net_pnl:,.2f}"
            )
        return "\n".join(lines)


def generate_opus_plan(
    *,
    market_data: MarketDataBundle,
    account_snapshot: AccountSnapshot,
    target_date: date,
    symbols: Sequence[str] | None = None,
    include_market_history: bool = True,
    opus_kwargs: Mapping[str, Any] | None = None,
    chronos2_forecasts: Mapping[str, Chronos2Forecast] | None = None,
    use_structured_outputs: bool = True,
) -> tuple[TradingPlan, str]:
    """Request a trading plan from Claude Opus with Chronos2 forecasts."""
    messages = build_opus_messages(
        market_data=market_data,
        target_date=target_date,
        account_snapshot=account_snapshot,
        symbols=symbols,
        include_market_history=include_market_history,
        chronos2_forecasts=chronos2_forecasts,
    )

    kwargs: MutableMapping[str, Any] = dict(opus_kwargs or {})
    model = kwargs.pop("model", "claude-sonnet-4-20250514")
    system_prompt = kwargs.pop("system_prompt", SYSTEM_PROMPT)

    parsed = None
    raw_text = ""

    if use_structured_outputs:
        try:
            result = call_opus_structured(messages, system_prompt=system_prompt, model=model)
            # Convert Pydantic model to dict
            parsed = result.model_dump()
            raw_text = result.model_dump_json()
            logger.info("Successfully used structured outputs")
        except Exception as e:
            logger.warning(f"Structured outputs failed: {e}, falling back to regular API")
            use_structured_outputs = False

    if not use_structured_outputs or parsed is None:
        try:
            parsed = call_opus_fallback(messages, system_prompt=system_prompt, model=model)
            raw_text = json.dumps(parsed)
        except Exception as e:
            logger.error(f"Fallback API also failed: {e}")
            raise

    # Ensure target_date is present
    if "target_date" not in parsed:
        parsed["target_date"] = target_date.isoformat()
        logger.warning(f"Response missing target_date, using provided date: {target_date}")

    if "instructions" not in parsed:
        parsed["instructions"] = []
        logger.warning("Response missing instructions, using empty list")

    # Normalize action types and field names
    for instr in parsed.get("instructions", []):
        action = instr.get("action", "").lower() if isinstance(instr.get("action"), str) else str(instr.get("action", ""))
        if action in ("enter", "long", "open"):
            instr["action"] = "buy"
        elif action in ("short",):
            instr["action"] = "sell"
        elif action in ("close", "flatten"):
            instr["action"] = "exit"

        # Normalize qty -> quantity
        if "qty" in instr and "quantity" not in instr:
            instr["quantity"] = instr.pop("qty")

        # Normalize execution_window -> execution_session
        if "execution_window" in instr and "execution_session" not in instr:
            instr["execution_session"] = instr.pop("execution_window")

        # Ensure quantity is int
        if "quantity" in instr:
            instr["quantity"] = int(instr["quantity"])

    json_text = json.dumps(parsed)
    plan = TradingPlanEnvelope.from_json(json_text).plan
    return plan, raw_text


def simulate_opus_plan(
    *,
    market_data: MarketDataBundle,
    account_snapshot: AccountSnapshot,
    target_date: date,
    symbols: Sequence[str] | None = None,
    include_market_history: bool = True,
    opus_kwargs: Mapping[str, Any] | None = None,
    simulator: MaxDiffSimulator | None = None,
    chronos2_device: str = "cuda",
    chronos2_context_length: int = 512,
    use_structured_outputs: bool = True,
) -> OpusPlanResult:
    """Generate an Opus plan with Chronos2 forecasts and evaluate with max-diff simulator."""
    symbol_list = list(symbols or market_data.bars.keys())

    # Generate Chronos2 forecasts
    chronos2_forecasts = None
    try:
        chronos2_forecasts = generate_chronos2_forecasts(
            market_data=market_data,
            symbols=symbol_list,
            prediction_length=1,
            context_length=chronos2_context_length,
            device_map=chronos2_device,
        )
        if chronos2_forecasts:
            logger.info(f"Generated Chronos2 forecasts for {len(chronos2_forecasts)} symbols")
    except Exception as e:
        logger.warning(f"Failed to generate Chronos2 forecasts: {e}")
        chronos2_forecasts = None

    plan, raw_text = generate_opus_plan(
        market_data=market_data,
        account_snapshot=account_snapshot,
        target_date=target_date,
        symbols=symbols,
        include_market_history=include_market_history,
        opus_kwargs=opus_kwargs,
        chronos2_forecasts=chronos2_forecasts,
        use_structured_outputs=use_structured_outputs,
    )

    sim = simulator or MaxDiffSimulator(market_data=market_data)
    result = sim.run([plan])

    return OpusPlanResult(
        plan=plan,
        raw_response=raw_text,
        simulation=result,
        chronos2_forecasts=chronos2_forecasts,
    )


def _snapshot_from_simulation(
    *,
    previous_snapshot: AccountSnapshot,
    simulation: MaxDiffResult,
    snapshot_date: date,
    final_positions: Mapping[str, Mapping[str, float]] | None = None,
) -> AccountSnapshot:
    """Build a lightweight account snapshot for the next planning round."""
    positions: list[AccountPosition] = []

    if final_positions:
        for symbol, payload in final_positions.items():
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

    # Preserve previous equity if simulation returned zero or negative (no trades executed)
    ending_equity = simulation.ending_equity
    ending_cash = simulation.ending_cash
    prev_equity = _snapshot_equity(previous_snapshot)

    # If simulation returns implausible values, use previous equity adjusted by PnL
    if ending_equity <= 0.0 or ending_cash <= 0.0:
        ending_equity = prev_equity + simulation.net_pnl
        ending_cash = float(previous_snapshot.cash or prev_equity) + simulation.net_pnl

    return AccountSnapshot(
        equity=ending_equity,
        cash=ending_cash,
        buying_power=ending_equity,
        timestamp=timestamp,
        positions=positions,
    )


def simulate_opus_replanning(
    *,
    market_data_by_date: Mapping[date, MarketDataBundle] | Iterable[tuple[date, MarketDataBundle]],
    account_snapshot: AccountSnapshot,
    target_dates: Sequence[date],
    symbols: Sequence[str] | None = None,
    include_market_history: bool = True,
    opus_kwargs: Mapping[str, Any] | None = None,
    trading_days_per_year: int | None = None,
    chronos2_device: str = "cuda",
    chronos2_context_length: int = 512,
    use_structured_outputs: bool = True,
) -> OpusReplanResult:
    """Iteratively generate Opus plans with Chronos2 for each date."""
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
    steps: list[OpusPlanStep] = []
    initial_equity = _snapshot_equity(account_snapshot)

    for step_index, current_date in enumerate(target_dates, start=1):
        bundle = data_lookup.get(current_date)
        if bundle is None:
            raise KeyError(f"No market data bundle provided for {current_date}.")

        starting_equity = _snapshot_equity(current_snapshot)

        plan_result = simulate_opus_plan(
            market_data=bundle,
            account_snapshot=current_snapshot,
            target_date=current_date,
            symbols=symbols,
            include_market_history=include_market_history,
            opus_kwargs=opus_kwargs,
            chronos2_device=chronos2_device,
            chronos2_context_length=chronos2_context_length,
            use_structured_outputs=use_structured_outputs,
        )

        ending_equity = starting_equity + plan_result.simulation.net_pnl
        if starting_equity and starting_equity > 0:
            daily_return_pct = plan_result.simulation.net_pnl / starting_equity
        else:
            daily_return_pct = 0.0

        logger.info(
            f"Opus+Chronos2 plan step {step_index}: net PnL ${plan_result.simulation.net_pnl:,.2f} "
            f"(daily return {daily_return_pct * 100:.3f}%)"
        )

        steps.append(
            OpusPlanStep(
                date=current_date,
                plan=plan_result.plan,
                raw_response=plan_result.raw_response,
                simulation=plan_result.simulation,
                starting_equity=starting_equity,
                ending_equity=ending_equity,
                daily_return_pct=daily_return_pct,
                chronos2_forecasts=plan_result.chronos2_forecasts,
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
        f"Opus+Chronos2 replanning summary: total return {total_return_pct * 100:.3f}%, "
        f"annualized {annualized_return_pct * 100:.3f}% over {day_count} sessions "
        f"(annualized with {annualization_days} days/year)"
    )

    return OpusReplanResult(
        steps=steps,
        starting_equity=initial_equity,
        ending_equity=final_equity,
        total_return_pct=total_return_pct,
        annualized_return_pct=annualized_return_pct,
        annualization_days=annualization_days,
    )


__all__ = [
    "OpusPlanResult",
    "OpusPlanStep",
    "OpusReplanResult",
    "generate_opus_plan",
    "simulate_opus_plan",
    "simulate_opus_replanning",
]
