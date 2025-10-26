from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from stockagent.agentsimulator import (
    AccountPosition,
    AccountSnapshot,
    ExecutionSession,
    PlanActionType,
    TradingInstruction,
    TradingPlan,
)

from ..config import PipelineConfig
from ..forecasting import ForecastReturnSet
from ..pipeline import AllocationPipeline, AllocationResult
from ..views_schema import LLMViews, TickerView
from .forecast_adapter import CombinedForecastAdapter, SymbolForecast


@dataclass
class PipelineSimulationConfig:
    symbols: Sequence[str] | None = None
    lookback_days: int = 120
    sample_count: int = 512
    min_trade_value: float = 250.0
    min_volatility: float = 0.002
    confidence_floor: float = 0.05
    confidence_ceiling: float = 0.9
    llm_horizon_days: int = 5


def _extract_history(
    *,
    market_frames: Mapping[str, pd.DataFrame],
    target_timestamp: pd.Timestamp,
    min_length: int,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, float]]:
    histories: Dict[str, pd.DataFrame] = {}
    latest_prices: Dict[str, float] = {}
    for symbol, frame in market_frames.items():
        history = frame[frame.index < target_timestamp]
        if len(history) < min_length:
            continue
        histories[symbol] = history.copy()
        last_row = history.iloc[-1]
        latest_prices[symbol] = float(last_row.get("close", np.nan))
    return histories, latest_prices


def _positions_to_signed_quantities(positions: Sequence[AccountPosition]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for position in positions:
        qty = float(position.quantity)
        if position.side.lower() == "short":
            qty = -abs(qty)
        result[position.symbol.upper()] = qty
    return result


def _build_llm_views(
    *,
    forecasts: Dict[str, SymbolForecast],
    horizon_days: int,
    config: PipelineSimulationConfig,
) -> LLMViews:
    views: list[TickerView] = []
    for stats in forecasts.values():
        mu = stats.predicted_return
        volatility = max(stats.error_pct, config.min_volatility)

        signal_strength = max(abs(mu) - volatility, 0.0)
        if volatility <= 0:
            raw_confidence = 0.5
        else:
            raw_confidence = signal_strength / (volatility + 1e-6)
        confidence = float(np.clip(raw_confidence, config.confidence_floor, config.confidence_ceiling))

        view = TickerView(
            ticker=stats.symbol,
            horizon_days=horizon_days,
            mu_bps=mu * 1e4 * horizon_days,
            stdev_bps=volatility * 1e4 * np.sqrt(horizon_days),
            confidence=confidence,
            half_life_days=max(3, min(30, int(2 * horizon_days))),
            rationale=f"Combined forecast projected return {mu:.4f}, volatility proxy {volatility:.4f}",
        )
        views.append(view)
    symbols = list(forecasts.keys())
    return LLMViews(asof=pd.Timestamp.utcnow().date().isoformat(), universe=symbols, views=views)


class PipelinePlanBuilder:
    """
    Build execution-ready trading plans by pairing probabilistic forecasts with
    the second-generation allocation pipeline.
    """

    def __init__(
        self,
        *,
        pipeline: AllocationPipeline,
        forecast_adapter: CombinedForecastAdapter,
        pipeline_config: PipelineSimulationConfig,
        pipeline_params: PipelineConfig,
    ) -> None:
        self.pipeline = pipeline
        self.forecast_adapter = forecast_adapter
        self.config = pipeline_config
        self.pipeline_params = pipeline_params
        self._previous_weights: Dict[str, float] = {}
        self._rng = np.random.default_rng(42)
        self.last_allocation: Optional[AllocationResult] = None

    def build_for_day(
        self,
        *,
        target_timestamp: pd.Timestamp,
        market_frames: Mapping[str, pd.DataFrame],
        account_snapshot: AccountSnapshot,
    ) -> Optional[TradingPlan]:
        histories, latest_prices = _extract_history(
            market_frames=market_frames,
            target_timestamp=target_timestamp,
            min_length=self.pipeline_params.annualisation_periods // 4,
        )
        if not histories:
            return None

        forecasts: Dict[str, SymbolForecast] = {}
        for symbol, history in histories.items():
            symbol_upper = symbol.upper()
            forecast = self.forecast_adapter.forecast(symbol_upper, history)
            if forecast is not None and np.isfinite(forecast.predicted_close):
                forecasts[symbol_upper] = forecast

        if not forecasts:
            logger.warning("No forecasts available for %s", target_timestamp.date())
            return None

        universe = tuple(sorted(forecasts.keys()))
        samples_primary = self._generate_return_samples(universe, forecasts, scale=1.0)
        samples_secondary = self._generate_return_samples(universe, forecasts, scale=1.35)

        chronos_set = ForecastReturnSet(universe=universe, samples=samples_primary)
        timesfm_set = ForecastReturnSet(universe=universe, samples=samples_secondary)

        previous = np.array([self._previous_weights.get(symbol, 0.0) for symbol in universe], dtype=float)
        llm_views = _build_llm_views(
            forecasts=forecasts,
            horizon_days=self.config.llm_horizon_days,
            config=self.config,
        )

        try:
            allocation = self.pipeline.run(
                chronos=chronos_set,
                timesfm=timesfm_set,
                llm_views=llm_views,
                previous_weights=previous,
            )
        except Exception as exc:
            logger.error("Pipeline allocation failed on %s: %s", target_timestamp.date(), exc)
            return None
        self._previous_weights = {
            symbol: weight for symbol, weight in zip(universe, allocation.weights)
        }
        self.last_allocation = allocation

        instructions = self._weights_to_instructions(
            universe=universe,
            weights=allocation.weights,
            forecasts=forecasts,
            latest_prices=latest_prices,
            account_snapshot=account_snapshot,
        )

        if not instructions:
            logger.info("No actionable instructions produced for %s", target_timestamp.date())
            return None

        metadata = {
            "generated_by": "stockagent2",
            "diagnostics": allocation.diagnostics,
            "universe": universe,
        }

        return TradingPlan(
            target_date=target_timestamp.date(),
            instructions=instructions,
            metadata=metadata,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _generate_return_samples(
        self,
        universe: Tuple[str, ...],
        forecasts: Dict[str, SymbolForecast],
        *,
        scale: float,
    ) -> np.ndarray:
        sample_count = self.config.sample_count
        matrix = np.zeros((sample_count, len(universe)), dtype=float)
        for idx, symbol in enumerate(universe):
            stats = forecasts[symbol]
            mu = stats.predicted_return
            sigma = max(stats.error_pct, self.config.min_volatility) * scale
            samples = self._rng.normal(loc=mu, scale=sigma, size=sample_count)
            matrix[:, idx] = np.clip(samples, -0.25, 0.25)
        return matrix

    def _weights_to_instructions(
        self,
        *,
        universe: Tuple[str, ...],
        weights: np.ndarray,
        forecasts: Dict[str, SymbolForecast],
        latest_prices: Mapping[str, float],
        account_snapshot: AccountSnapshot,
    ) -> list[TradingInstruction]:
        nav = account_snapshot.equity if account_snapshot.equity > 0 else account_snapshot.cash
        positions = _positions_to_signed_quantities(account_snapshot.positions)

        instructions: list[TradingInstruction] = []
        universe_set = set(universe)
        for symbol, weight in zip(universe, weights):
            price = latest_prices.get(symbol)
            if price is None or not np.isfinite(price) or price <= 0:
                continue
            target_qty = (weight * nav) / price
            current_qty = positions.get(symbol, 0.0)
            delta = target_qty - current_qty
            notional_change = abs(delta) * price
            if notional_change < self.config.min_trade_value:
                continue

            action = PlanActionType.BUY if delta > 0 else PlanActionType.SELL
            instruction = TradingInstruction(
                symbol=symbol,
                action=action,
                quantity=abs(float(delta)),
                execution_session=ExecutionSession.MARKET_OPEN,
                entry_price=forecasts[symbol].entry_price,
                notes=f"target_weight={weight:.4f}; predicted_return={forecasts[symbol].predicted_return:.4f}",
            )
            instructions.append(instruction)

        # Flatten any positions outside the optimisation universe
        for symbol, qty in positions.items():
            if symbol in universe_set:
                continue
            price = latest_prices.get(symbol)
            if price is None or not np.isfinite(price) or price <= 0:
                continue
            notional = abs(qty) * price
            if notional < self.config.min_trade_value:
                continue
            action = PlanActionType.SELL if qty > 0 else PlanActionType.BUY
            instructions.append(
                TradingInstruction(
                    symbol=symbol,
                    action=action,
                    quantity=abs(float(qty)),
                    execution_session=ExecutionSession.MARKET_OPEN,
                    entry_price=price,
                    notes="Outside-universe position rebalance",
                )
            )

        return instructions
