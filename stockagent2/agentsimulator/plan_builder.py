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

DEFAULT_HISTORY_MIN_PERIOD_DIVISOR = 4
DEFAULT_SECONDARY_SAMPLE_SCALE = 1.35
DEFAULT_SAMPLE_RETURN_CLIP = 0.25
DEFAULT_MIN_VIEW_HALF_LIFE_DAYS = 3
DEFAULT_MAX_VIEW_HALF_LIFE_DAYS = 30
DEFAULT_PIPELINE_RNG_SEED = 42


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
    history_min_period_divisor: int = DEFAULT_HISTORY_MIN_PERIOD_DIVISOR
    secondary_sample_scale: float = DEFAULT_SECONDARY_SAMPLE_SCALE
    sample_return_clip: float = DEFAULT_SAMPLE_RETURN_CLIP
    min_view_half_life_days: int = DEFAULT_MIN_VIEW_HALF_LIFE_DAYS
    max_view_half_life_days: int = DEFAULT_MAX_VIEW_HALF_LIFE_DAYS
    rng_seed: int = DEFAULT_PIPELINE_RNG_SEED

    def __post_init__(self) -> None:
        if self.history_min_period_divisor < 1:
            raise ValueError("history_min_period_divisor must be at least 1")
        if self.secondary_sample_scale <= 0:
            raise ValueError("secondary_sample_scale must be greater than 0")
        if self.sample_return_clip <= 0:
            raise ValueError("sample_return_clip must be greater than 0")
        if self.min_view_half_life_days < 1:
            raise ValueError("min_view_half_life_days must be at least 1")
        if self.max_view_half_life_days < self.min_view_half_life_days:
            raise ValueError("max_view_half_life_days must be greater than or equal to min_view_half_life_days")
        if self.confidence_floor < 0 or self.confidence_ceiling > 1 or self.confidence_floor > self.confidence_ceiling:
            raise ValueError("confidence_floor/confidence_ceiling must satisfy 0 <= floor <= ceiling <= 1")


@dataclass(frozen=True)
class PipelinePlanBuildDiagnostics:
    target_date: str
    status: str
    symbols_considered: Tuple[str, ...]
    symbols_with_history: Tuple[str, ...]
    insufficient_history_symbols: Tuple[str, ...]
    forecasted_symbols: Tuple[str, ...]
    forecast_failure_reasons: Dict[str, str]
    generated_instruction_count: int
    skipped_min_trade_symbols: Tuple[str, ...]
    missing_price_symbols: Tuple[str, ...]
    allocation_error: str | None = None


def _extract_history(
    *,
    market_frames: Mapping[str, pd.DataFrame],
    target_timestamp: pd.Timestamp,
    min_length: int,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, float]]:
    histories: Dict[str, pd.DataFrame] = {}
    latest_prices: Dict[str, float] = {}
    for symbol, frame in market_frames.items():
        end_idx = int(frame.index.searchsorted(target_timestamp, side="left"))
        if end_idx < min_length:
            continue
        history = frame.iloc[:end_idx]
        histories[symbol] = history
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
            half_life_days=max(
                config.min_view_half_life_days,
                min(config.max_view_half_life_days, int(2 * horizon_days)),
            ),
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
        self.last_allocation: Optional[AllocationResult] = None
        self.last_build_diagnostics: Optional[PipelinePlanBuildDiagnostics] = None

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
            min_length=max(1, self.pipeline_params.annualisation_periods // self.config.history_min_period_divisor),
        )
        symbols_considered = tuple(sorted(str(symbol).upper() for symbol in market_frames))
        symbols_with_history = tuple(sorted(histories))
        insufficient_history_symbols = tuple(
            symbol for symbol in symbols_considered if symbol not in histories
        )
        if not histories:
            self.last_build_diagnostics = PipelinePlanBuildDiagnostics(
                target_date=target_timestamp.date().isoformat(),
                status="no_history",
                symbols_considered=symbols_considered,
                symbols_with_history=(),
                insufficient_history_symbols=insufficient_history_symbols,
                forecasted_symbols=(),
                forecast_failure_reasons={},
                generated_instruction_count=0,
                skipped_min_trade_symbols=(),
                missing_price_symbols=(),
            )
            return None

        forecasts: Dict[str, SymbolForecast] = {}
        forecast_failure_reasons: Dict[str, str] = {}
        for symbol, history in histories.items():
            symbol_upper = symbol.upper()
            if hasattr(self.forecast_adapter, "forecast_with_reason"):
                forecast, reason = self.forecast_adapter.forecast_with_reason(symbol_upper, history)
            else:
                forecast = self.forecast_adapter.forecast(symbol_upper, history)
                reason = None if forecast is not None else "forecast_unavailable"
            if forecast is not None and np.isfinite(forecast.predicted_close):
                forecasts[symbol_upper] = forecast
            elif reason is not None:
                forecast_failure_reasons[symbol_upper] = reason
            else:
                forecast_failure_reasons[symbol_upper] = "invalid_predicted_close"

        if not forecasts:
            logger.warning("No forecasts available for %s", target_timestamp.date())
            self.last_build_diagnostics = PipelinePlanBuildDiagnostics(
                target_date=target_timestamp.date().isoformat(),
                status="no_forecasts",
                symbols_considered=symbols_considered,
                symbols_with_history=symbols_with_history,
                insufficient_history_symbols=insufficient_history_symbols,
                forecasted_symbols=(),
                forecast_failure_reasons=forecast_failure_reasons,
                generated_instruction_count=0,
                skipped_min_trade_symbols=(),
                missing_price_symbols=(),
            )
            return None

        universe = tuple(sorted(forecasts.keys()))
        rng = self._rng_for_timestamp(target_timestamp)
        samples_primary = self._generate_return_samples(universe, forecasts, scale=1.0, rng=rng)
        samples_secondary = self._generate_return_samples(
            universe,
            forecasts,
            scale=self.config.secondary_sample_scale,
            rng=rng,
        )

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
            self.last_build_diagnostics = PipelinePlanBuildDiagnostics(
                target_date=target_timestamp.date().isoformat(),
                status="allocation_failed",
                symbols_considered=symbols_considered,
                symbols_with_history=symbols_with_history,
                insufficient_history_symbols=insufficient_history_symbols,
                forecasted_symbols=tuple(sorted(forecasts)),
                forecast_failure_reasons=forecast_failure_reasons,
                generated_instruction_count=0,
                skipped_min_trade_symbols=(),
                missing_price_symbols=(),
                allocation_error=str(exc),
            )
            return None
        self._previous_weights = {
            symbol: weight for symbol, weight in zip(universe, allocation.weights)
        }
        self.last_allocation = allocation

        instructions, skipped_min_trade_symbols, missing_price_symbols = self._weights_to_instructions(
            universe=universe,
            weights=allocation.weights,
            forecasts=forecasts,
            latest_prices=latest_prices,
            account_snapshot=account_snapshot,
        )

        if not instructions:
            logger.info("No actionable instructions produced for %s", target_timestamp.date())
            self.last_build_diagnostics = PipelinePlanBuildDiagnostics(
                target_date=target_timestamp.date().isoformat(),
                status="no_instructions",
                symbols_considered=symbols_considered,
                symbols_with_history=symbols_with_history,
                insufficient_history_symbols=insufficient_history_symbols,
                forecasted_symbols=tuple(sorted(forecasts)),
                forecast_failure_reasons=forecast_failure_reasons,
                generated_instruction_count=0,
                skipped_min_trade_symbols=skipped_min_trade_symbols,
                missing_price_symbols=missing_price_symbols,
            )
            return None

        metadata = {
            "generated_by": "stockagent2",
            "diagnostics": allocation.diagnostics,
            "universe": universe,
        }
        self.last_build_diagnostics = PipelinePlanBuildDiagnostics(
            target_date=target_timestamp.date().isoformat(),
            status="ok",
            symbols_considered=symbols_considered,
            symbols_with_history=symbols_with_history,
            insufficient_history_symbols=insufficient_history_symbols,
            forecasted_symbols=tuple(sorted(forecasts)),
            forecast_failure_reasons=forecast_failure_reasons,
            generated_instruction_count=len(instructions),
            skipped_min_trade_symbols=skipped_min_trade_symbols,
            missing_price_symbols=missing_price_symbols,
        )

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
        rng: np.random.Generator,
    ) -> np.ndarray:
        sample_count = self.config.sample_count
        matrix = np.zeros((sample_count, len(universe)), dtype=float)
        for idx, symbol in enumerate(universe):
            stats = forecasts[symbol]
            mu = stats.predicted_return
            sigma = max(stats.error_pct, self.config.min_volatility) * scale
            samples = rng.normal(loc=mu, scale=sigma, size=sample_count)
            matrix[:, idx] = np.clip(samples, -self.config.sample_return_clip, self.config.sample_return_clip)
        return matrix

    def _rng_for_timestamp(self, target_timestamp: pd.Timestamp) -> np.random.Generator:
        timestamp = pd.Timestamp(target_timestamp)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        timestamp_ns = int(timestamp.value)
        seed = np.random.SeedSequence(
            [
                int(self.config.rng_seed) & 0xFFFFFFFF,
                timestamp_ns & 0xFFFFFFFF,
                (timestamp_ns >> 32) & 0xFFFFFFFF,
            ]
        )
        return np.random.default_rng(seed)

    def _weights_to_instructions(
        self,
        *,
        universe: Tuple[str, ...],
        weights: np.ndarray,
        forecasts: Dict[str, SymbolForecast],
        latest_prices: Mapping[str, float],
        account_snapshot: AccountSnapshot,
    ) -> tuple[list[TradingInstruction], tuple[str, ...], tuple[str, ...]]:
        nav = account_snapshot.equity if account_snapshot.equity > 0 else account_snapshot.cash
        positions = _positions_to_signed_quantities(account_snapshot.positions)

        instructions: list[TradingInstruction] = []
        skipped_min_trade_symbols: set[str] = set()
        missing_price_symbols: set[str] = set()
        universe_set = set(universe)
        for symbol, weight in zip(universe, weights):
            price = latest_prices.get(symbol)
            if price is None or not np.isfinite(price) or price <= 0:
                missing_price_symbols.add(symbol)
                continue
            target_qty = (weight * nav) / price
            current_qty = positions.get(symbol, 0.0)
            delta = target_qty - current_qty
            notional_change = abs(delta) * price
            if notional_change < self.config.min_trade_value:
                skipped_min_trade_symbols.add(symbol)
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
                missing_price_symbols.add(symbol)
                continue
            notional = abs(qty) * price
            if notional < self.config.min_trade_value:
                skipped_min_trade_symbols.add(symbol)
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

        return (
            instructions,
            tuple(sorted(skipped_min_trade_symbols)),
            tuple(sorted(missing_price_symbols)),
        )
