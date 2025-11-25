"""High-level utilities for generating and simulating Claude Opus trading plans."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Iterable, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
from loguru import logger

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
from stockagentdeepseek_maxdiff.simulator import MaxDiffResult, MaxDiffSimulator

from .opus_wrapper import call_opus_chat, call_opus_structured, TradingPlanModel
from .prompt_builder import SYSTEM_PROMPT, build_opus_messages

# Feature flag for structured outputs
# Disabled: current anthropic SDK doesn't have beta.messages.parse
USE_STRUCTURED_OUTPUTS = False

try:
    from stockagentdeepseek_neural.forecaster import NeuralForecast, build_neural_forecasts
except ImportError:
    NeuralForecast = None  # type: ignore
    build_neural_forecasts = None  # type: ignore

try:
    from stockagentcombined.forecaster import CombinedForecastGenerator
except ImportError:
    CombinedForecastGenerator = None  # type: ignore

try:
    from backtest_test3_inline import calibrate_signal
except Exception:
    def calibrate_signal(predictions: np.ndarray, actual_returns: np.ndarray) -> Tuple[float, float]:
        matched = min(len(predictions), len(actual_returns))
        if matched > 1:
            slope, intercept = np.polyfit(predictions[:matched], actual_returns[:matched], 1)
            return float(slope), float(intercept)
        return 1.0, 0.0

from src.fixtures import crypto_symbols


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


def _has_crypto(plan: TradingPlan) -> bool:
    return any(instr.symbol in crypto_symbols for instr in plan.instructions)


def _has_equities(plan: TradingPlan) -> bool:
    return any(instr.symbol not in crypto_symbols for instr in plan.instructions)


@dataclass(slots=True)
class OpusPlanResult:
    """Result from a single Opus plan generation and simulation."""
    plan: TradingPlan
    raw_response: str
    simulation: MaxDiffResult
    forecasts: Mapping[str, "NeuralForecast"] | None = None
    calibration: Mapping[str, float] | None = None


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
    forecasts: Mapping[str, "NeuralForecast"] | None = None
    calibration: Mapping[str, float] | None = None


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
            "Claude Opus replanning results:",
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


def _calibrate_symbol(
    *,
    generator: "CombinedForecastGenerator",
    bundle: MarketDataBundle,
    symbol: str,
    target_date: date,
    window: int,
    forecast: "NeuralForecast",
) -> Tuple[float, float, float, float]:
    """Calibrate forecast signals against historical returns."""
    frame = bundle.get_symbol_bars(symbol)
    if frame.empty:
        return 1.0, 0.0, 0.0, 0.0
    frame = frame.sort_index()

    predictions: list[float] = []
    actuals: list[float] = []

    total_rows = len(frame)
    if window > 0:
        start_idx = max(1, total_rows - window - 1)
    else:
        start_idx = 1
    if start_idx >= total_rows:
        start_idx = max(1, total_rows - 1)

    for idx in range(start_idx, total_rows):
        hist = frame.iloc[:idx]
        if hist.empty:
            continue
        prev_close = float(hist.iloc[-1]["close"])
        try:
            combined = generator.generate_for_symbol(
                symbol,
                prediction_length=1,
                historical_frame=hist,
            )
        except Exception:
            continue
        predicted_close = float(combined.combined.get("close", prev_close))
        predictions.append((predicted_close - prev_close) / prev_close if prev_close else 0.0)

        current_close = float(frame.iloc[idx]["close"])
        actuals.append((current_close - prev_close) / prev_close if prev_close else 0.0)

    if len(predictions) > window:
        predictions = predictions[-window:]
        actuals = actuals[-window:]

    if len(predictions) < 2:
        slope, intercept = 1.0, 0.0
    else:
        slope, intercept = calibrate_signal(
            np.array(predictions, dtype=np.float64),
            np.array(actuals, dtype=np.float64),
        )

    if symbol in bundle.bars and not bundle.bars[symbol].empty:
        last_close = float(bundle.bars[symbol].iloc[-1]["close"])
    else:
        last_close = 0.0
    predicted_close = float(forecast.combined.get("close", last_close))
    raw_move = (predicted_close - last_close) / last_close if last_close else 0.0
    calibrated_move = float(slope * raw_move + intercept)

    return float(slope), float(intercept), raw_move, calibrated_move


def generate_opus_plan(
    *,
    market_data: MarketDataBundle,
    account_snapshot: AccountSnapshot,
    target_date: date,
    symbols: Sequence[str] | None = None,
    include_market_history: bool = True,
    opus_kwargs: Mapping[str, Any] | None = None,
    forecasts: Mapping[str, "NeuralForecast"] | None = None,
    calibration: Mapping[str, float] | None = None,
) -> tuple[TradingPlan, str]:
    """Request a trading plan from Claude Opus and return the parsed plan with raw JSON."""
    import json as json_module

    messages = build_opus_messages(
        market_data=market_data,
        target_date=target_date,
        account_snapshot=account_snapshot,
        symbols=symbols,
        include_market_history=include_market_history,
        forecasts=forecasts,
        calibration=calibration,
    )
    kwargs: MutableMapping[str, Any] = dict(opus_kwargs or {})
    if "system_prompt" not in kwargs:
        kwargs["system_prompt"] = SYSTEM_PROMPT

    if USE_STRUCTURED_OUTPUTS:
        # Use structured outputs for guaranteed schema compliance
        try:
            plan_model = call_opus_structured(messages, **kwargs)

            # Convert Pydantic model to dict
            plan_dict = plan_model.model_dump()

            # Fix target_date if needed
            if not plan_dict.get("target_date"):
                plan_dict["target_date"] = target_date.isoformat()

            # Convert instructions format for compatibility
            for instr in plan_dict.get("instructions", []):
                # Convert enum values to strings
                if hasattr(instr.get("action"), "value"):
                    instr["action"] = instr["action"].value
                if hasattr(instr.get("execution_session"), "value"):
                    instr["execution_session"] = instr["execution_session"].value

            json_text = json_module.dumps(plan_dict)
            plan = TradingPlanEnvelope.from_json(json_text).plan
            return plan, json_text

        except Exception as e:
            logger.warning(f"Structured outputs failed, falling back to unstructured: {e}")

    # Fallback: use unstructured chat
    raw_text = call_opus_chat(messages, **kwargs)

    # Extract JSON from response (may contain thinking/markdown)
    json_start = raw_text.find("{")
    json_end = raw_text.rfind("}") + 1
    if json_start >= 0 and json_end > json_start:
        json_text = raw_text[json_start:json_end]
    else:
        json_text = raw_text

    # Parse and validate/fix the JSON
    try:
        parsed = json_module.loads(json_text)
    except json_module.JSONDecodeError as e:
        logger.error(f"Failed to parse Opus response as JSON: {e}")
        logger.error(f"Raw response (first 500 chars): {raw_text[:500]}")
        raise ValueError(f"Opus returned invalid JSON: {e}") from e

    # Ensure target_date is present
    if "target_date" not in parsed:
        parsed["target_date"] = target_date.isoformat()
        logger.warning(f"Opus response missing target_date, using provided date: {target_date}")

    # Ensure instructions is present
    if "instructions" not in parsed:
        parsed["instructions"] = []
        logger.warning("Opus response missing instructions, using empty list")

    # Normalize action types and field names
    for instr in parsed.get("instructions", []):
        action = instr.get("action", "").lower()
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

    json_text = json_module.dumps(parsed)
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
    forecasts: Mapping[str, "NeuralForecast"] | None = None,
    generator: "CombinedForecastGenerator | None" = None,
    calibration_window: int = 14,
    simulator: MaxDiffSimulator | None = None,
) -> OpusPlanResult:
    """Generate an Opus plan with neural forecasts and evaluate with max-diff simulator."""
    symbol_list = list(symbols or market_data.bars.keys())

    working_generator = None
    if generator is not None:
        working_generator = generator
    elif CombinedForecastGenerator is not None:
        try:
            working_generator = CombinedForecastGenerator()
        except Exception:
            pass

    if forecasts is None and build_neural_forecasts is not None and working_generator is not None:
        try:
            forecasts = build_neural_forecasts(
                symbols=symbol_list,
                market_data=market_data,
                prediction_length=1,
                generator=working_generator,
            )
        except Exception as e:
            logger.warning("Failed to generate neural forecasts: %s", e)
            forecasts = None

    calibration: MutableMapping[str, float] = {}
    if calibration_window > 1 and forecasts and working_generator is not None:
        for symbol in symbol_list:
            if symbol not in forecasts:
                continue
            try:
                slope, intercept, raw_move, cal_move = _calibrate_symbol(
                    generator=working_generator,
                    bundle=market_data,
                    symbol=symbol,
                    target_date=target_date,
                    window=calibration_window,
                    forecast=forecasts[symbol],
                )
                calibration[f"{symbol}_calibration_slope"] = slope
                calibration[f"{symbol}_calibration_intercept"] = intercept
                calibration[f"{symbol}_raw_expected_move_pct"] = raw_move
                calibration[f"{symbol}_calibrated_expected_move_pct"] = cal_move
            except Exception as e:
                logger.warning("Calibration failed for %s: %s", symbol, e)

    plan, raw_text = generate_opus_plan(
        market_data=market_data,
        account_snapshot=account_snapshot,
        target_date=target_date,
        symbols=symbols,
        include_market_history=include_market_history,
        opus_kwargs=opus_kwargs,
        forecasts=forecasts,
        calibration=dict(calibration) if calibration else None,
    )

    sim = simulator or MaxDiffSimulator(market_data=market_data)
    result = sim.run([plan])

    return OpusPlanResult(
        plan=plan,
        raw_response=raw_text,
        simulation=result,
        forecasts=forecasts,
        calibration=dict(calibration) if calibration else None,
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

    # Preserve previous equity if simulation returned zero (no trades executed)
    ending_equity = simulation.ending_equity
    ending_cash = simulation.ending_cash
    if ending_equity == 0.0 and ending_cash == 0.0:
        ending_equity = _snapshot_equity(previous_snapshot)
        ending_cash = float(previous_snapshot.cash or ending_equity)

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
    generator: "CombinedForecastGenerator | None" = None,
    calibration_window: int = 14,
    trading_days_per_year: int | None = None,
) -> OpusReplanResult:
    """Iteratively generate Opus plans for each date, updating portfolio state."""
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
            generator=generator,
            calibration_window=calibration_window,
        )

        ending_equity = starting_equity + plan_result.simulation.net_pnl
        if starting_equity and starting_equity > 0:
            daily_return_pct = plan_result.simulation.net_pnl / starting_equity
        else:
            daily_return_pct = 0.0

        logger.info(
            f"Opus plan step {step_index}: net PnL ${plan_result.simulation.net_pnl:,.2f} "
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
                forecasts=plan_result.forecasts,
                calibration=plan_result.calibration,
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
        f"Opus replanning summary: total return {total_return_pct * 100:.3f}%, "
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
