"""Chronos2 forecasting for both price and PnL prediction.

Key innovation: Use Chronos2 to forecast:
1. Next-period price (high/low/close) for trade entry/exit
2. Next-day PnL based on historical PnL series (as a "judge")

The PnL forecast serves as a training signal - optimize strategy
so Chronos2 predicts profitability tomorrow.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from chronospnltrader.config import ForecastConfig

logger = logging.getLogger(__name__)


@dataclass
class PriceForecast:
    """Price forecast for a single timestep."""

    timestamp: pd.Timestamp
    current_close: float
    predicted_close: float
    predicted_high: float
    predicted_low: float
    predicted_close_p10: float
    predicted_close_p90: float

    @property
    def predicted_return(self) -> float:
        """Predicted percentage return."""
        if self.current_close <= 0:
            return 0.0
        return (self.predicted_close - self.current_close) / self.current_close

    @property
    def predicted_range(self) -> float:
        """Predicted high-low range as percentage."""
        if self.current_close <= 0:
            return 0.0
        return (self.predicted_high - self.predicted_low) / self.current_close

    @property
    def confidence_spread(self) -> float:
        """Uncertainty measure (p90 - p10)."""
        if self.current_close <= 0:
            return float("inf")
        return (self.predicted_close_p90 - self.predicted_close_p10) / self.current_close


@dataclass
class PnLForecast:
    """PnL forecast for the next period.

    This is the key innovation: forecast future PnL to determine
    if current strategy will be profitable.
    """

    pnl_history: np.ndarray  # Historical PnL values
    predicted_pnl: float  # Predicted next-period PnL
    predicted_pnl_p10: float  # 10th percentile (downside)
    predicted_pnl_p90: float  # 90th percentile (upside)
    cumulative_pnl: float  # Total PnL so far
    predicted_cumulative: float  # Predicted cumulative after next period

    @property
    def is_profitable(self) -> bool:
        """Is next period predicted to be profitable?"""
        return self.predicted_pnl > 0

    @property
    def expected_improvement(self) -> float:
        """Expected improvement in cumulative PnL."""
        return self.predicted_cumulative - self.cumulative_pnl

    @property
    def downside_risk(self) -> float:
        """Downside risk (negative = potential loss)."""
        return self.predicted_pnl_p10

    @property
    def upside_potential(self) -> float:
        """Upside potential."""
        return self.predicted_pnl_p90


class Chronos2Forecaster:
    """Chronos2-based forecaster for both prices and PnL.

    Uses the amazon/chronos-2 model for time series forecasting.
    Lazy-loads the model to save memory when not in use.
    """

    def __init__(self, config: ForecastConfig) -> None:
        self.config = config
        self._wrapper = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization of Chronos2 wrapper."""
        if self._initialized:
            return

        try:
            from src.models.chronos2_wrapper import Chronos2OHLCWrapper

            self._wrapper = Chronos2OHLCWrapper.from_pretrained(
                model_id=self.config.model_id,
                device_map=self.config.device_map,
                default_context_length=self.config.context_length,
                quantile_levels=list(self.config.quantile_levels),
                default_batch_size=self.config.batch_size,
            )
            self._initialized = True
            logger.info("Initialized Chronos2 forecaster with model %s", self.config.model_id)
        except Exception as e:
            logger.error("Failed to initialize Chronos2: %s", e)
            raise

    def forecast_prices(
        self,
        context_df: pd.DataFrame,
        current_close: float,
    ) -> Optional[PriceForecast]:
        """Generate price forecast from OHLC context.

        Args:
            context_df: DataFrame with timestamp, open, high, low, close, volume
            current_close: Current close price

        Returns:
            PriceForecast or None if failed
        """
        self._ensure_initialized()

        if context_df.empty or len(context_df) < 10:
            logger.warning("Insufficient context for price forecast")
            return None

        try:
            batch = self._wrapper.predict_ohlc(
                context_df,
                symbol="STOCK",
                prediction_length=self.config.prediction_length,
                context_length=len(context_df),
                quantile_levels=list(self.config.quantile_levels),
            )

            q50 = batch.quantile_frames.get(0.5)
            q10 = batch.quantile_frames.get(0.1)
            q90 = batch.quantile_frames.get(0.9)

            if q50 is None or q50.empty:
                logger.warning("Empty price predictions")
                return None

            # Get first prediction (next hour)
            pred_close = float(q50.iloc[0].get("close", current_close))
            pred_high = float(q50.iloc[0].get("high", pred_close))
            pred_low = float(q50.iloc[0].get("low", pred_close))

            pred_close_p10 = float(q10.iloc[0].get("close", pred_close * 0.99)) if q10 is not None else pred_close * 0.99
            pred_close_p90 = float(q90.iloc[0].get("close", pred_close * 1.01)) if q90 is not None else pred_close * 1.01

            return PriceForecast(
                timestamp=pd.Timestamp.now(tz="UTC"),
                current_close=current_close,
                predicted_close=pred_close,
                predicted_high=pred_high,
                predicted_low=pred_low,
                predicted_close_p10=pred_close_p10,
                predicted_close_p90=pred_close_p90,
            )

        except Exception as e:
            logger.error("Price forecast failed: %s", e)
            return None

    def forecast_pnl(
        self,
        pnl_history: np.ndarray,
        prediction_length: int = 7,
    ) -> Optional[PnLForecast]:
        """Forecast future PnL based on historical PnL series.

        This is the key "judge" function - it predicts whether
        the current strategy will be profitable in the next period.

        Args:
            pnl_history: Array of historical PnL values (per-bar returns)
            prediction_length: How many periods ahead to forecast

        Returns:
            PnLForecast or None if failed
        """
        self._ensure_initialized()

        if len(pnl_history) < 10:
            logger.warning("Insufficient PnL history for forecast")
            return None

        # Cumulative PnL
        cumulative = float(np.sum(pnl_history))

        try:
            # Create a simple univariate time series for PnL
            # Chronos2 expects a DataFrame-like structure
            pnl_series = torch.tensor(pnl_history, dtype=torch.float32).unsqueeze(0)

            # Use the raw Chronos2 pipeline for univariate forecasting
            if hasattr(self._wrapper, "pipeline"):
                pipeline = self._wrapper.pipeline

                # Forecast
                forecast = pipeline.predict(
                    context=pnl_series,
                    prediction_length=prediction_length,
                    num_samples=100,
                )

                # Get quantiles
                samples = forecast.numpy()[0]  # (num_samples, prediction_length)

                # Average over prediction horizon
                avg_samples = samples.mean(axis=1)  # Average PnL per period

                pred_pnl = float(np.median(avg_samples))
                pred_pnl_p10 = float(np.percentile(avg_samples, 10))
                pred_pnl_p90 = float(np.percentile(avg_samples, 90))

            else:
                # Fallback: simple exponential weighted prediction
                weights = np.exp(np.linspace(-2, 0, len(pnl_history)))
                weights /= weights.sum()
                pred_pnl = float(np.average(pnl_history, weights=weights))
                pred_pnl_p10 = pred_pnl - float(np.std(pnl_history))
                pred_pnl_p90 = pred_pnl + float(np.std(pnl_history))

            predicted_cumulative = cumulative + pred_pnl * prediction_length

            return PnLForecast(
                pnl_history=pnl_history,
                predicted_pnl=pred_pnl,
                predicted_pnl_p10=pred_pnl_p10,
                predicted_pnl_p90=pred_pnl_p90,
                cumulative_pnl=cumulative,
                predicted_cumulative=predicted_cumulative,
            )

        except Exception as e:
            logger.error("PnL forecast failed: %s", e)
            return None

    def forecast_pnl_differentiable(
        self,
        pnl_history: torch.Tensor,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Differentiable PnL forecast for training.

        Instead of actually calling Chronos2 (which is not differentiable),
        we use a differentiable approximation that captures the key behavior:
        - Trend following: if recent PnL is positive, predict positive
        - Mean reversion: extreme PnL tends to revert
        - Volatility scaling: higher vol = less confident prediction

        The actual Chronos2 is used during evaluation for the "ground truth"
        prediction that we try to match.

        Args:
            pnl_history: (batch, history_length) tensor of PnL values
            device: Device to use

        Returns:
            Dict with predicted_pnl, confidence, trend_signal
        """
        # Exponential weights for recency
        history_len = pnl_history.size(1)
        weights = torch.exp(torch.linspace(-2, 0, history_len, device=device))
        weights = weights / weights.sum()

        # Weighted average (trend following)
        weighted_avg = (pnl_history * weights).sum(dim=1)

        # Recent momentum (last 7 bars)
        recent = pnl_history[:, -7:] if history_len >= 7 else pnl_history
        momentum = recent.mean(dim=1)

        # Volatility
        pnl_std = pnl_history.std(dim=1) + 1e-8

        # Mean reversion factor
        cumulative = pnl_history.sum(dim=1)
        mean_reversion = -0.1 * torch.tanh(cumulative / (pnl_std * 10))

        # Combined prediction
        predicted_pnl = 0.5 * weighted_avg + 0.3 * momentum + 0.2 * mean_reversion

        # Confidence based on consistency
        sign_consistency = (pnl_history.sign() == predicted_pnl.unsqueeze(1).sign()).float().mean(dim=1)
        confidence = sign_consistency * torch.sigmoid(-pnl_std * 10)

        return {
            "predicted_pnl": predicted_pnl,
            "confidence": confidence,
            "momentum": momentum,
            "volatility": pnl_std,
            "cumulative": cumulative,
        }

    def compute_pnl_forecast_loss(
        self,
        predicted_pnl: torch.Tensor,
        actual_next_pnl: torch.Tensor,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss between predicted and actual PnL.

        We want the strategy to be optimized so that:
        1. Chronos2 predicts positive PnL (is_profitable)
        2. The actual PnL matches/exceeds the prediction

        Args:
            predicted_pnl: Model's differentiable PnL prediction
            actual_next_pnl: Actual PnL from simulation
            confidence: Prediction confidence

        Returns:
            Loss tensor
        """
        # Core loss: maximize actual PnL weighted by prediction
        # If predicted positive, reward positive actual
        # If predicted negative, penalize (should have skipped trade)
        alignment_loss = -predicted_pnl * actual_next_pnl

        # Prediction accuracy loss
        prediction_error = (predicted_pnl - actual_next_pnl).abs()

        # Confidence calibration: high confidence should mean accurate prediction
        calibration_loss = confidence * prediction_error

        return alignment_loss.mean() + 0.1 * calibration_loss.mean()

    def unload(self) -> None:
        """Release GPU memory."""
        if self._wrapper is not None:
            self._wrapper.unload()
            self._wrapper = None
            self._initialized = False
            logger.info("Chronos2 forecaster unloaded")


def create_forecaster(config: ForecastConfig) -> Chronos2Forecaster:
    """Factory function to create a forecaster."""
    return Chronos2Forecaster(config)
