from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from loguru import logger

from stockagent.agentsimulator import (
    ExecutionSession,
    PlanActionType,
    TradingInstruction,
    TradingPlan,
)

from ..forecaster import CombinedForecast, CombinedForecastGenerator


@dataclass
class SimulationConfig:
    symbols: Sequence[str] | None = None
    lookback_days: int = 120
    simulation_days: int = 5
    starting_cash: float = 1_000_000.0
    min_history: int = 64
    min_signal: float = 0.0025
    error_multiplier: float = 1.5
    base_quantity: float = 50.0
    max_quantity_multiplier: float = 4.0
    min_quantity: float = 5.0
    allow_short: bool = True


def _collect_histories(
    *,
    market_frames: Mapping[str, pd.DataFrame],
    target_timestamp: pd.Timestamp,
    min_history: int,
) -> dict[str, pd.DataFrame]:
    histories: dict[str, pd.DataFrame] = {}
    for symbol, frame in market_frames.items():
        history = frame[frame.index < target_timestamp]
        if len(history) < min_history:
            continue
        histories[symbol] = history.copy()
    return histories


def _prepare_history_payload(history: pd.DataFrame) -> pd.DataFrame:
    result = history.reset_index().rename(columns={"index": "timestamp"})
    if "timestamp" not in result.columns:
        raise ValueError("History frame missing timestamp column after reset_index.")
    return result


def _weighted_mae(forecast: CombinedForecast) -> float:
    weights = forecast.weights or {}
    total = 0.0
    used = 0.0
    for name, model_forecast in forecast.model_forecasts.items():
        weight = weights.get(name, 0.0)
        if weight <= 0.0:
            continue
        total += weight * model_forecast.average_price_mae
        used += weight
    if used <= 0.0 and forecast.model_forecasts:
        total = sum(model.average_price_mae for model in forecast.model_forecasts.values()) / len(
            forecast.model_forecasts
        )
    return total


def _build_instruction_payload(
    *,
    symbol: str,
    forecast: CombinedForecast,
    history: pd.DataFrame,
    config: SimulationConfig,
) -> tuple[TradingInstruction, float] | None:
    last_row = history.iloc[-1]
    last_close = float(last_row["close"])
    if not np.isfinite(last_close) or last_close <= 0.0:
        return None

    predicted_close = float(forecast.combined.get("close", last_close))
    if not np.isfinite(predicted_close):
        return None

    predicted_return = (predicted_close - last_close) / last_close

    mae_value = _weighted_mae(forecast)
    error_pct = mae_value / last_close if last_close else 0.0
    threshold = max(config.min_signal, error_pct * config.error_multiplier)

    if abs(predicted_return) <= threshold:
        return None

    direction = PlanActionType.BUY if predicted_return > 0 else PlanActionType.SELL
    if direction == PlanActionType.SELL and not config.allow_short:
        return None

    signal_strength = abs(predicted_return) - threshold
    multiplier = 1.0 + signal_strength / max(threshold, 1e-6)
    multiplier = min(multiplier, config.max_quantity_multiplier)
    quantity = max(config.min_quantity, round(config.base_quantity * multiplier))

    entry_price = float(forecast.combined.get("open", last_row.get("open", last_close)))
    if not np.isfinite(entry_price):
        entry_price = last_close

    notes = f"pred_return={predicted_return:.4f}; threshold={threshold:.4f}; mae={mae_value:.4f}"

    entry = TradingInstruction(
        symbol=symbol,
        action=direction,
        quantity=float(quantity),
        execution_session=ExecutionSession.MARKET_OPEN,
        entry_price=entry_price,
        exit_price=predicted_close,
        notes=notes,
    )
    return entry, predicted_close


class CombinedPlanBuilder:
    """
    Convert blended Toto/Kronos forecasts into executable trading plans that can be
    consumed by the shared :class:`stockagent.agentsimulator.AgentSimulator`.
    """

    def __init__(
        self,
        generator: CombinedForecastGenerator,
        config: SimulationConfig,
    ) -> None:
        self.generator = generator
        self.config = config

    def build_for_day(
        self,
        *,
        target_timestamp: pd.Timestamp,
        market_frames: Mapping[str, pd.DataFrame],
    ) -> TradingPlan | None:
        histories = _collect_histories(
            market_frames=market_frames,
            target_timestamp=target_timestamp,
            min_history=self.config.min_history,
        )
        if not histories:
            return None

        forecasts: dict[str, CombinedForecast] = {}
        for symbol, history in histories.items():
            try:
                payload = _prepare_history_payload(history)
                forecasts[symbol] = self.generator.generate_for_symbol(
                    symbol,
                    prediction_length=1,
                    historical_frame=payload,
                )
            except Exception as exc:
                logger.warning("Forecast failed for %s on %s: %s", symbol, target_timestamp.date(), exc)

        instructions: list[TradingInstruction] = []
        for symbol, forecast in forecasts.items():
            history = histories.get(symbol)
            if history is None:
                continue
            payload = _build_instruction_payload(
                symbol=symbol,
                forecast=forecast,
                history=history,
                config=self.config,
            )
            if payload is not None:
                entry_instruction, predicted_close = payload
                instructions.append(entry_instruction)
                exit_instruction = TradingInstruction(
                    symbol=symbol,
                    action=PlanActionType.EXIT,
                    quantity=0.0,
                    execution_session=ExecutionSession.MARKET_CLOSE,
                    exit_price=predicted_close,
                    notes="auto-exit at market close",
                )
                instructions.append(exit_instruction)

        if not instructions:
            return None

        metadata = {
            "generated_by": "stockagentcombined",
            "symbols_considered": list(histories.keys()),
            "symbols_traded": [instruction.symbol for instruction in instructions],
        }

        plan = TradingPlan(
            target_date=target_timestamp.date(),
            instructions=instructions,
            metadata=metadata,
        )
        return plan


def build_daily_plans(
    *,
    builder: CombinedPlanBuilder,
    market_frames: Mapping[str, pd.DataFrame],
    trading_days: Iterable[pd.Timestamp],
) -> list[TradingPlan]:
    plans: list[TradingPlan] = []
    for timestamp in trading_days:
        plan = builder.build_for_day(target_timestamp=timestamp, market_frames=market_frames)
        if plan is not None:
            plans.append(plan)
    return plans


def create_builder(
    *,
    generator: CombinedForecastGenerator,
    symbols: Sequence[str] | None,
    lookback_days: int,
) -> CombinedPlanBuilder:
    config = SimulationConfig(symbols=symbols, lookback_days=lookback_days)
    return CombinedPlanBuilder(generator=generator, config=config)
