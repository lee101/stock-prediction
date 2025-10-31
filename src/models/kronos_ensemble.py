"""
Kronos Ensemble Wrapper with Aggregation Strategies.

This module implements ensemble forecasting for Kronos by:
1. Generating multiple samples with different temperatures/configs
2. Applying Toto-style aggregation (trimmed_mean, etc.)
3. Providing more robust predictions similar to Toto's approach
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from src.models.kronos_wrapper import KronosForecastingWrapper
from src.models.toto_aggregation import aggregate_with_spec


class KronosEnsembleWrapper:
    """
    Ensemble wrapper for Kronos that generates multiple samples and aggregates.

    This combines Kronos's autoregressive forecasting with Toto's robust
    aggregation strategies.
    """

    def __init__(
        self,
        model_name: str = "NeoQuasar/Kronos-base",
        tokenizer_name: str = "NeoQuasar/Kronos-Tokenizer-base",
        device: Optional[str] = None,
        max_context: int = 192,
        clip: float = 1.8,
    ):
        """
        Initialize Kronos ensemble wrapper.

        Args:
            model_name: Kronos model identifier
            tokenizer_name: Kronos tokenizer identifier
            device: Device to run on (auto-detected if None)
            max_context: Maximum context length
            clip: Clipping value for predictions
        """
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.wrapper = KronosForecastingWrapper(
            model_name=model_name,
            tokenizer_name=tokenizer_name,
            device=device,
            max_context=max_context,
            clip=clip,
        )
        self.max_context = max_context
        self.clip = clip

    def predict_ensemble(
        self,
        data: pd.DataFrame,
        timestamp_col: str = "timestamp",
        columns: List[str] = None,
        pred_len: int = 1,
        lookback: Optional[int] = None,
        num_samples: int = 10,
        base_temperature: float = 0.15,
        temperature_range: Tuple[float, float] = (0.10, 0.25),
        top_p: float = 0.82,
        top_k: int = 0,
        aggregate: str = "trimmed_mean_10",
    ) -> Dict[str, np.ndarray]:
        """
        Generate ensemble predictions with aggregation.

        Args:
            data: Input time series data
            timestamp_col: Name of timestamp column
            columns: Columns to forecast (default: ["close"])
            pred_len: Forecast horizon
            lookback: Context length (default: max_context)
            num_samples: Number of ensemble samples
            base_temperature: Base temperature for sampling
            temperature_range: Range for temperature variation
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            aggregate: Aggregation method (trimmed_mean_X, median, etc.)

        Returns:
            Dictionary with aggregated predictions
        """
        if columns is None:
            columns = ["close"]
        if lookback is None:
            lookback = self.max_context

        # Generate multiple samples with temperature variation
        samples = []
        temperatures_used = []

        for i in range(num_samples):
            # Vary temperature across ensemble
            if num_samples > 1:
                # Linearly interpolate between temperature range
                t_min, t_max = temperature_range
                temperature = t_min + (t_max - t_min) * (i / (num_samples - 1))
            else:
                temperature = base_temperature

            temperatures_used.append(temperature)

            # Generate prediction
            result = self.wrapper.predict_series(
                data=data,
                timestamp_col=timestamp_col,
                columns=columns,
                pred_len=pred_len,
                lookback=lookback,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                sample_count=1,  # Single sample per temperature
            )

            # Extract predictions
            for col in columns:
                if col not in result:
                    continue

                forecast = result[col]
                samples.append({
                    "column": col,
                    "absolute": float(forecast.absolute[0]) if forecast.absolute.size > 0 else 0.0,
                    "percent": float(forecast.percent[0]) if forecast.percent.size > 0 else 0.0,
                })

        # Aggregate predictions by column
        aggregated_results = {}

        for col in columns:
            col_samples_abs = [s["absolute"] for s in samples if s["column"] == col]
            col_samples_pct = [s["percent"] for s in samples if s["column"] == col]

            if not col_samples_abs:
                continue

            # Apply aggregation
            agg_absolute = aggregate_with_spec(
                np.array(col_samples_abs).reshape(-1, 1), aggregate
            )[0]
            agg_percent = aggregate_with_spec(
                np.array(col_samples_pct).reshape(-1, 1), aggregate
            )[0]

            aggregated_results[col] = {
                "absolute": agg_absolute,
                "percent": agg_percent,
                "samples_absolute": col_samples_abs,
                "samples_percent": col_samples_pct,
                "temperatures": temperatures_used,
            }

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return aggregated_results

    def predict_with_multiple_configs(
        self,
        data: pd.DataFrame,
        timestamp_col: str = "timestamp",
        columns: List[str] = None,
        pred_len: int = 1,
        lookback: Optional[int] = None,
        configs: Optional[List[Dict]] = None,
        aggregate: str = "trimmed_mean_10",
    ) -> Dict[str, np.ndarray]:
        """
        Generate predictions using multiple hyperparameter configurations.

        This is more expensive but can produce even more robust predictions.

        Args:
            data: Input time series data
            timestamp_col: Name of timestamp column
            columns: Columns to forecast
            pred_len: Forecast horizon
            lookback: Context length
            configs: List of config dicts with {temperature, top_p, top_k}
            aggregate: Aggregation method

        Returns:
            Dictionary with aggregated predictions
        """
        if columns is None:
            columns = ["close"]
        if lookback is None:
            lookback = self.max_context

        if configs is None:
            # Default diverse configs
            configs = [
                {"temperature": 0.10, "top_p": 0.80, "top_k": 0},
                {"temperature": 0.15, "top_p": 0.82, "top_k": 0},
                {"temperature": 0.20, "top_p": 0.85, "top_k": 16},
                {"temperature": 0.25, "top_p": 0.85, "top_k": 24},
            ]

        samples = []

        for config in configs:
            result = self.wrapper.predict_series(
                data=data,
                timestamp_col=timestamp_col,
                columns=columns,
                pred_len=pred_len,
                lookback=lookback,
                temperature=config["temperature"],
                top_p=config["top_p"],
                top_k=config["top_k"],
                sample_count=1,
            )

            for col in columns:
                if col not in result:
                    continue

                forecast = result[col]
                samples.append({
                    "column": col,
                    "absolute": float(forecast.absolute[0]) if forecast.absolute.size > 0 else 0.0,
                    "percent": float(forecast.percent[0]) if forecast.percent.size > 0 else 0.0,
                    "config": config,
                })

        # Aggregate by column
        aggregated_results = {}

        for col in columns:
            col_samples_abs = [s["absolute"] for s in samples if s["column"] == col]
            col_samples_pct = [s["percent"] for s in samples if s["column"] == col]

            if not col_samples_abs:
                continue

            agg_absolute = aggregate_with_spec(
                np.array(col_samples_abs).reshape(-1, 1), aggregate
            )[0]
            agg_percent = aggregate_with_spec(
                np.array(col_samples_pct).reshape(-1, 1), aggregate
            )[0]

            aggregated_results[col] = {
                "absolute": agg_absolute,
                "percent": agg_percent,
                "samples_absolute": col_samples_abs,
                "samples_percent": col_samples_pct,
                "configs": configs,
            }

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return aggregated_results
