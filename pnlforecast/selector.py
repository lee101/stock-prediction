"""PnL forecasting and strategy selection using Chronos2."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import ForecastConfigPnL, StrategyConfigPnL, SimulationConfigPnL
from .strategy import StrategyThresholds
from .simulator import StrategyPnLResult, get_strategy_cumulative_history

logger = logging.getLogger(__name__)


@dataclass
class PnLForecast:
    """Forecast for a strategy's future PnL."""

    strategy_id: str
    forecast_date: date
    current_cumulative_pnl: float
    predicted_pnl_change: float  # Predicted change in cumulative PnL
    predicted_pnl_p10: float  # 10th percentile
    predicted_pnl_p50: float  # Median
    predicted_pnl_p90: float  # 90th percentile
    confidence: float = 0.0  # Confidence measure

    @property
    def predicted_cumulative_pnl(self) -> float:
        """Predicted cumulative PnL after next day."""
        return self.current_cumulative_pnl + self.predicted_pnl_change


class PnLForecaster:
    """Uses Chronos2 to forecast strategy PnL curves."""

    def __init__(
        self,
        forecast_config: ForecastConfigPnL,
    ) -> None:
        self.config = forecast_config
        self._pipeline = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization of Chronos2 pipeline."""
        if self._initialized:
            return

        try:
            # Try local Chronos2Pipeline first (has proper config)
            try:
                from chronos.chronos2.pipeline import Chronos2Pipeline
                import torch

                self._pipeline = Chronos2Pipeline.from_pretrained(
                    self.config.model_id,
                    device_map=self.config.device_map,
                    torch_dtype=torch.bfloat16,
                )
                self._initialized = True
                logger.info("Initialized Chronos2Pipeline for PnL forecasting")
                return
            except Exception as e1:
                logger.debug("Chronos2Pipeline failed: %s, trying fallback", e1)

            # Fallback to base ChronosPipeline
            from chronos import BaseChronosPipeline
            import torch

            self._pipeline = BaseChronosPipeline.from_pretrained(
                self.config.model_id,
                device_map=self.config.device_map,
                torch_dtype=torch.bfloat16,
            )
            self._initialized = True
            logger.info("Initialized BaseChronosPipeline for PnL forecasting")

        except Exception as e:
            logger.error("Failed to initialize Chronos2: %s", e)
            raise

    def forecast_pnl(
        self,
        strategy_result: StrategyPnLResult,
        target_date: date,
        context_days: Optional[int] = None,
    ) -> Optional[PnLForecast]:
        """Forecast next-day PnL for a strategy.

        Uses the cumulative PnL curve as input to Chronos2 to predict
        the next day's PnL movement.

        Args:
            strategy_result: Historical simulation result for the strategy
            target_date: Date to forecast
            context_days: Number of historical days to use as context

        Returns:
            PnLForecast or None if insufficient data
        """
        self._ensure_initialized()

        ctx_days = context_days or self.config.pnl_context_days

        # Get cumulative PnL history
        pnl_history = get_strategy_cumulative_history(strategy_result, n_days=ctx_days)

        if len(pnl_history) < 3:
            logger.debug(
                "Insufficient PnL history for %s (got %d days, need 3+)",
                strategy_result.strategy.strategy_id,
                len(pnl_history),
            )
            return None

        current_pnl = float(pnl_history[-1])

        try:
            import torch

            # Convert to tensor - scale for better numerical stability
            # PnL values are typically small (e.g., 0.001 = 0.1%)
            scale_factor = 100.0  # Scale up for forecasting
            scaled_history = pnl_history * scale_factor

            # Chronos2Pipeline expects 3D: (n_series, n_variates, history_length)
            # For univariate time series: (1, 1, history_length)
            context = torch.tensor(scaled_history, dtype=torch.float32)
            context = context.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, history_length)

            # Get forecasts - API may vary between pipeline versions
            try:
                # Chronos2Pipeline API
                forecasts = self._pipeline.predict(
                    context,
                    prediction_length=1,
                    num_samples=100,
                )
            except TypeError:
                # Fallback for different API
                forecasts = self._pipeline.predict(
                    context,
                    prediction_length=1,
                )

            # Extract quantiles - handle different output formats
            if hasattr(forecasts, 'numpy'):
                samples = forecasts.numpy().squeeze()
            elif isinstance(forecasts, np.ndarray):
                samples = forecasts.squeeze()
            else:
                samples = np.array(forecasts).squeeze()

            # Unscale predictions
            samples = samples / scale_factor

            # Handle single sample vs multiple samples
            if samples.ndim == 0:
                p10 = p50 = p90 = float(samples)
            else:
                p10 = float(np.percentile(samples, 10))
                p50 = float(np.percentile(samples, 50))
                p90 = float(np.percentile(samples, 90))

            # Predicted change = predicted_cumulative - current_cumulative
            predicted_change = p50 - current_pnl

            # Confidence based on prediction spread
            spread = p90 - p10
            confidence = 1.0 / (1.0 + abs(spread) * 100) if spread != 0 else 1.0

            return PnLForecast(
                strategy_id=strategy_result.strategy.strategy_id,
                forecast_date=target_date,
                current_cumulative_pnl=current_pnl,
                predicted_pnl_change=predicted_change,
                predicted_pnl_p10=p10 - current_pnl,
                predicted_pnl_p50=predicted_change,
                predicted_pnl_p90=p90 - current_pnl,
                confidence=confidence,
            )

        except Exception as e:
            logger.error(
                "PnL forecast failed for %s: %s",
                strategy_result.strategy.strategy_id,
                e,
            )
            return None

    def forecast_all_strategies(
        self,
        strategy_results: Dict[str, StrategyPnLResult],
        target_date: date,
    ) -> Dict[str, PnLForecast]:
        """Forecast PnL for all strategies.

        Args:
            strategy_results: Dict of strategy_id -> StrategyPnLResult
            target_date: Date to forecast

        Returns:
            Dict of strategy_id -> PnLForecast
        """
        forecasts = {}
        for strategy_id, result in strategy_results.items():
            forecast = self.forecast_pnl(result, target_date)
            if forecast is not None:
                forecasts[strategy_id] = forecast

        logger.debug(
            "Generated %d PnL forecasts for %s",
            len(forecasts),
            target_date,
        )
        return forecasts

    def unload(self) -> None:
        """Release GPU memory."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            self._initialized = False
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            logger.info("PnL forecaster unloaded")


class StrategySelector:
    """Selects the best strategy based on forecasted PnL."""

    def __init__(
        self,
        strategy_config: StrategyConfigPnL,
        sim_config: SimulationConfigPnL,
    ) -> None:
        self.strategy_config = strategy_config
        self.sim_config = sim_config

    def select_best_strategy(
        self,
        pnl_forecasts: Dict[str, PnLForecast],
        strategy_results: Dict[str, StrategyPnLResult],
        min_history_days: Optional[int] = None,
    ) -> Optional[Tuple[str, PnLForecast]]:
        """Select the best strategy based on forecasted PnL.

        Args:
            pnl_forecasts: Dict of strategy_id -> PnLForecast
            strategy_results: Dict of strategy_id -> StrategyPnLResult
            min_history_days: Minimum days of history required

        Returns:
            Tuple of (strategy_id, forecast) for best strategy, or None
        """
        min_days = min_history_days or self.sim_config.min_pnl_history_days

        candidates = []
        for strategy_id, forecast in pnl_forecasts.items():
            # Check history requirement
            result = strategy_results.get(strategy_id)
            if result is None or len(result.dates) < min_days:
                continue

            # Only consider strategies with positive predicted PnL
            if forecast.predicted_pnl_change > 0:
                candidates.append((strategy_id, forecast))

        if not candidates:
            logger.debug("No profitable strategies found")
            return None

        # Sort by predicted PnL change (descending)
        candidates.sort(key=lambda x: x[1].predicted_pnl_change, reverse=True)

        best_id, best_forecast = candidates[0]
        logger.debug(
            "Selected strategy %s with predicted PnL change: %.4f%%",
            best_id,
            best_forecast.predicted_pnl_change * 100,
        )

        return best_id, best_forecast

    def rank_strategies(
        self,
        pnl_forecasts: Dict[str, PnLForecast],
        top_n: int = 5,
    ) -> List[Tuple[str, PnLForecast]]:
        """Rank strategies by forecasted PnL.

        Args:
            pnl_forecasts: Dict of strategy_id -> PnLForecast
            top_n: Number of top strategies to return

        Returns:
            List of (strategy_id, forecast) tuples, sorted by predicted PnL
        """
        ranked = [
            (sid, f)
            for sid, f in pnl_forecasts.items()
        ]
        ranked.sort(key=lambda x: x[1].predicted_pnl_change, reverse=True)

        return ranked[:top_n]

    def get_strategy_score(
        self,
        forecast: PnLForecast,
        result: StrategyPnLResult,
    ) -> float:
        """Compute a composite score for a strategy.

        Combines predicted PnL, historical performance, and confidence.

        Args:
            forecast: PnL forecast for the strategy
            result: Historical simulation result

        Returns:
            Composite score (higher is better)
        """
        # Predicted PnL component (50%)
        pnl_score = forecast.predicted_pnl_change

        # Historical win rate component (25%)
        win_rate_score = result.win_rate * 0.01  # Scale to similar range

        # Confidence component (25%)
        confidence_score = forecast.confidence * 0.01

        # Combined score
        score = (
            0.5 * pnl_score
            + 0.25 * win_rate_score
            + 0.25 * confidence_score
        )

        return score


@dataclass
class DailySelection:
    """Record of strategy selection for a single day."""

    selection_date: date
    symbol: str
    selected_strategy_id: Optional[str]
    selected_forecast: Optional[PnLForecast]
    all_forecasts: Dict[str, PnLForecast]
    num_candidates: int
    selection_reason: str = ""


def create_selector(
    strategy_config: StrategyConfigPnL,
    sim_config: SimulationConfigPnL,
) -> StrategySelector:
    """Factory function to create a StrategySelector.

    Args:
        strategy_config: Strategy configuration
        sim_config: Simulation configuration

    Returns:
        Configured StrategySelector
    """
    return StrategySelector(strategy_config, sim_config)
