"""Chronos2 forecaster for direct price prediction."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    """Result of a Chronos2 forecast."""

    current_price: float
    predicted_prices: np.ndarray  # Shape: (prediction_length,)
    predicted_low: np.ndarray  # 10th percentile
    predicted_high: np.ndarray  # 90th percentile

    @property
    def predicted_return(self) -> float:
        """Predicted return from current to mean of forecast."""
        if self.current_price <= 0:
            return 0.0
        mean_future = np.mean(self.predicted_prices)
        return (mean_future - self.current_price) / self.current_price

    @property
    def predicted_max_return(self) -> float:
        """Max predicted return (to highest point)."""
        if self.current_price <= 0:
            return 0.0
        max_future = np.max(self.predicted_prices)
        return (max_future - self.current_price) / self.current_price

    @property
    def predicted_min_return(self) -> float:
        """Min predicted return (to lowest point)."""
        if self.current_price <= 0:
            return 0.0
        min_future = np.min(self.predicted_prices)
        return (min_future - self.current_price) / self.current_price

    @property
    def upside_ratio(self) -> float:
        """Ratio of upside to downside."""
        up = max(0, self.predicted_max_return)
        down = abs(min(0, self.predicted_min_return))
        if down < 1e-9:
            return 10.0 if up > 0 else 1.0
        return up / down


class DirectForecaster:
    """Direct Chronos2 forecaster for price prediction."""

    def __init__(
        self,
        model_id: str = "amazon/chronos-t5-small",
        prediction_length: int = 6,
        context_length: int = 64,
        device: str = "cuda",
    ):
        self.model_id = model_id
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.device = device

        self._pipeline = None

    def _ensure_loaded(self) -> None:
        """Lazy load the Chronos pipeline."""
        if self._pipeline is not None:
            return

        try:
            from chronos import ChronosPipeline

            logger.info(f"Loading Chronos model: {self.model_id}")
            self._pipeline = ChronosPipeline.from_pretrained(
                self.model_id,
                device_map=self.device if torch.cuda.is_available() else "cpu",
                torch_dtype=torch.float32,
            )
            logger.info("Chronos model loaded")

        except ImportError:
            raise ImportError(
                "chronos package required. Install with: pip install chronos-forecasting"
            )

    def forecast(
        self,
        prices: np.ndarray,
        num_samples: int = 20,
    ) -> Optional[ForecastResult]:
        """Generate forecast from price history.

        Args:
            prices: Historical prices, shape (context_length,)
            num_samples: Number of forecast samples for quantiles

        Returns:
            ForecastResult or None if insufficient data
        """
        if len(prices) < self.context_length:
            return None

        self._ensure_loaded()

        # Use last context_length prices
        context = prices[-self.context_length:]
        current_price = float(context[-1])

        if current_price <= 0:
            return None

        # Convert to tensor
        context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)

        # Generate forecasts
        with torch.no_grad():
            forecasts = self._pipeline.predict(
                context_tensor,
                prediction_length=self.prediction_length,
                num_samples=num_samples,
            )

        # forecasts shape: (1, num_samples, prediction_length)
        samples = forecasts[0].numpy()

        # Compute quantiles
        predicted_prices = np.median(samples, axis=0)
        predicted_low = np.percentile(samples, 10, axis=0)
        predicted_high = np.percentile(samples, 90, axis=0)

        return ForecastResult(
            current_price=current_price,
            predicted_prices=predicted_prices,
            predicted_low=predicted_low,
            predicted_high=predicted_high,
        )

    def forecast_batch(
        self,
        price_windows: np.ndarray,
        num_samples: int = 20,
    ) -> list[Optional[ForecastResult]]:
        """Batch forecast for multiple windows.

        Args:
            price_windows: Shape (batch, context_length)
            num_samples: Number of samples per forecast

        Returns:
            List of ForecastResults
        """
        self._ensure_loaded()

        results = []
        valid_indices = []
        valid_contexts = []

        for i, window in enumerate(price_windows):
            if len(window) >= self.context_length and window[-1] > 0:
                valid_indices.append(i)
                valid_contexts.append(window[-self.context_length:])

        if not valid_contexts:
            return [None] * len(price_windows)

        # Batch predict
        contexts_tensor = torch.tensor(
            np.array(valid_contexts), dtype=torch.float32
        )

        with torch.no_grad():
            forecasts = self._pipeline.predict(
                contexts_tensor,
                prediction_length=self.prediction_length,
                num_samples=num_samples,
            )

        # Build results
        result_map = {}
        for idx, (orig_idx, context) in enumerate(zip(valid_indices, valid_contexts)):
            samples = forecasts[idx].numpy()
            result_map[orig_idx] = ForecastResult(
                current_price=float(context[-1]),
                predicted_prices=np.median(samples, axis=0),
                predicted_low=np.percentile(samples, 10, axis=0),
                predicted_high=np.percentile(samples, 90, axis=0),
            )

        for i in range(len(price_windows)):
            results.append(result_map.get(i))

        return results
