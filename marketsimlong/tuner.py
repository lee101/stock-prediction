"""Chronos2 hyperparameter tuning for long-term simulation."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import DataConfigLong, ForecastConfigLong, TuningConfigLong
from .data import DailyDataLoader, is_crypto_symbol
from .forecaster import Chronos2Forecaster, DailyForecasts

logger = logging.getLogger(__name__)


def compute_forecast_metrics(
    forecasts: List[DailyForecasts],
    data_loader: DailyDataLoader,
) -> Dict[str, float]:
    """Compute forecast accuracy metrics across multiple days.

    Args:
        forecasts: List of daily forecasts
        data_loader: Data loader to get actual prices

    Returns:
        Dict with mae, mape, rmse, directional_accuracy
    """
    errors = []
    pct_errors = []
    sq_errors = []
    direction_correct = []

    for daily in forecasts:
        for symbol, forecast in daily.forecasts.items():
            actual = data_loader.get_price_on_date(symbol, daily.forecast_date)
            if actual is None:
                continue

            actual_close = actual["close"]
            pred_close = forecast.predicted_close
            current_close = forecast.current_close

            # Absolute error
            error = abs(pred_close - actual_close)
            errors.append(error)

            # Percentage error
            if actual_close > 0:
                pct_errors.append(error / actual_close)

            # Squared error
            sq_errors.append(error ** 2)

            # Directional accuracy
            pred_direction = pred_close > current_close
            actual_direction = actual_close > current_close
            direction_correct.append(pred_direction == actual_direction)

    if not errors:
        return {
            "mae": float("nan"),
            "mape": float("nan"),
            "rmse": float("nan"),
            "directional_accuracy": float("nan"),
            "n_samples": 0,
        }

    return {
        "mae": float(np.mean(errors)),
        "mape": float(np.mean(pct_errors)) * 100,
        "rmse": float(np.sqrt(np.mean(sq_errors))),
        "directional_accuracy": float(np.mean(direction_correct)) * 100,
        "n_samples": len(errors),
    }


class Chronos2Tuner:
    """Tunes Chronos2 hyperparameters for lowest MAE."""

    def __init__(
        self,
        data_config: DataConfigLong,
        tuning_config: TuningConfigLong,
    ) -> None:
        self.data_config = data_config
        self.tuning_config = tuning_config
        self.data_loader = DailyDataLoader(data_config)
        self.best_params: Dict[str, Any] = {}
        self.best_metrics: Dict[str, float] = {}
        self.trial_results: List[Dict[str, Any]] = []

    def _get_validation_dates(self) -> List[date]:
        """Get dates for validation period."""
        end_date = self.data_config.end_date
        start_date = end_date - timedelta(days=self.tuning_config.val_days)

        # Filter to trading days
        val_dates = []
        current = start_date
        while current <= end_date:
            if current.weekday() < 5:  # Skip weekends for stocks
                val_dates.append(current)
            current += timedelta(days=1)

        return val_dates

    def evaluate_config(
        self,
        forecast_config: ForecastConfigLong,
        symbols: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Evaluate a single forecast configuration.

        Args:
            forecast_config: Configuration to evaluate
            symbols: Optional subset of symbols

        Returns:
            Dict with evaluation metrics
        """
        self.data_loader.load_all_symbols()

        forecaster = Chronos2Forecaster(self.data_loader, forecast_config)

        val_dates = self._get_validation_dates()
        if symbols is None:
            symbols = list(self.data_config.all_symbols)

        all_forecasts = []

        try:
            for val_date in val_dates:
                available = self.data_loader.get_tradable_symbols_on_date(val_date)
                eval_symbols = [s for s in symbols if s in available]

                if not eval_symbols:
                    continue

                forecasts = forecaster.forecast_all_symbols(val_date, eval_symbols)
                if forecasts.forecasts:
                    all_forecasts.append(forecasts)

            metrics = compute_forecast_metrics(all_forecasts, self.data_loader)
            return metrics

        finally:
            forecaster.unload()

    def run_grid_search(
        self,
        symbols: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run grid search over hyperparameter space.

        Args:
            symbols: Optional subset of symbols to tune on

        Returns:
            Best configuration found
        """
        logger.info("Starting grid search tuning...")

        best_score = float("inf")
        best_config = None
        best_metrics = None

        config_count = 0
        total_configs = (
            len(self.tuning_config.context_lengths)
            * len(self.tuning_config.prediction_lengths)
            * len(self.tuning_config.use_multivariate_options)
            * len(self.tuning_config.preaug_strategies)
        )

        for context_length in self.tuning_config.context_lengths:
            for prediction_length in self.tuning_config.prediction_lengths:
                for use_multivariate in self.tuning_config.use_multivariate_options:
                    for preaug_strategy in self.tuning_config.preaug_strategies:
                        config_count += 1
                        logger.info(
                            "Evaluating config %d/%d: ctx=%d, pred=%d, mv=%s, preaug=%s",
                            config_count,
                            total_configs,
                            context_length,
                            prediction_length,
                            use_multivariate,
                            preaug_strategy,
                        )

                        # Build forecast config
                        forecast_config = ForecastConfigLong(
                            context_length=context_length,
                            prediction_length=prediction_length,
                            use_multivariate=use_multivariate,
                            use_preaugmentation=preaug_strategy != "baseline",
                            preaugmentation_dirs=(
                                (f"preaugstrategies/{preaug_strategy}",)
                                if preaug_strategy != "baseline"
                                else ()
                            ),
                        )

                        try:
                            metrics = self.evaluate_config(forecast_config, symbols)

                            # Get target metric
                            score = metrics.get(self.tuning_config.metric, float("inf"))

                            self.trial_results.append({
                                "context_length": context_length,
                                "prediction_length": prediction_length,
                                "use_multivariate": use_multivariate,
                                "preaug_strategy": preaug_strategy,
                                **metrics,
                            })

                            logger.info("  -> %s: %.4f", self.tuning_config.metric, score)

                            if score < best_score:
                                best_score = score
                                best_config = {
                                    "context_length": context_length,
                                    "prediction_length": prediction_length,
                                    "use_multivariate": use_multivariate,
                                    "preaug_strategy": preaug_strategy,
                                }
                                best_metrics = metrics
                                logger.info("  -> New best! %s=%.4f", self.tuning_config.metric, score)

                        except Exception as e:
                            logger.error("Config evaluation failed: %s", e)
                            continue

        self.best_params = best_config or {}
        self.best_metrics = best_metrics or {}

        if self.tuning_config.save_best and best_config:
            self._save_best_config(best_config, best_metrics)

        return best_config or {}

    def run_optuna_search(
        self,
        symbols: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run Optuna-based hyperparameter search.

        Args:
            symbols: Optional subset of symbols

        Returns:
            Best configuration found
        """
        try:
            import optuna
        except ImportError:
            logger.warning("Optuna not installed, falling back to grid search")
            return self.run_grid_search(symbols)

        def objective(trial: optuna.Trial) -> float:
            context_length = trial.suggest_categorical(
                "context_length",
                list(self.tuning_config.context_lengths),
            )
            prediction_length = trial.suggest_categorical(
                "prediction_length",
                list(self.tuning_config.prediction_lengths),
            )
            use_multivariate = trial.suggest_categorical(
                "use_multivariate",
                list(self.tuning_config.use_multivariate_options),
            )
            preaug_strategy = trial.suggest_categorical(
                "preaug_strategy",
                list(self.tuning_config.preaug_strategies),
            )

            forecast_config = ForecastConfigLong(
                context_length=context_length,
                prediction_length=prediction_length,
                use_multivariate=use_multivariate,
                use_preaugmentation=preaug_strategy != "baseline",
                preaugmentation_dirs=(
                    (f"preaugstrategies/{preaug_strategy}",)
                    if preaug_strategy != "baseline"
                    else ()
                ),
            )

            metrics = self.evaluate_config(forecast_config, symbols)
            score = metrics.get(self.tuning_config.metric, float("inf"))

            self.trial_results.append({
                "trial_number": trial.number,
                "context_length": context_length,
                "prediction_length": prediction_length,
                "use_multivariate": use_multivariate,
                "preaug_strategy": preaug_strategy,
                **metrics,
            })

            return score

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.tuning_config.n_trials)

        best_trial = study.best_trial
        self.best_params = best_trial.params
        self.best_metrics = {
            self.tuning_config.metric: best_trial.value,
        }

        if self.tuning_config.save_best:
            self._save_best_config(self.best_params, self.best_metrics)

        return self.best_params

    def tune_per_symbol(
        self,
        symbols: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Tune hyperparameters for each symbol individually.

        Args:
            symbols: Symbols to tune

        Returns:
            Dict mapping symbol -> best config
        """
        if symbols is None:
            symbols = list(self.data_config.all_symbols)

        per_symbol_configs = {}

        for symbol in symbols:
            logger.info("Tuning for symbol: %s", symbol)

            try:
                best_config = self.run_grid_search(symbols=[symbol])
                per_symbol_configs[symbol] = {
                    "config": best_config,
                    "metrics": self.best_metrics.copy(),
                }
                logger.info(
                    "Best config for %s: %s (MAE=%.4f)",
                    symbol,
                    best_config,
                    self.best_metrics.get("mae", float("nan")),
                )
            except Exception as e:
                logger.error("Failed to tune %s: %s", symbol, e)
                continue

        # Save per-symbol configs
        if self.tuning_config.save_best:
            output_path = self.tuning_config.output_dir / "per_symbol_configs.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(per_symbol_configs, f, indent=2, default=str)

        return per_symbol_configs

    def _save_best_config(
        self,
        config: Dict[str, Any],
        metrics: Dict[str, float],
    ) -> None:
        """Save best configuration to disk."""
        output_path = self.tuning_config.output_dir / "best_config.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "config": config,
            "metrics": metrics,
            "tuning_config": {
                "metric": self.tuning_config.metric,
                "val_days": self.tuning_config.val_days,
            },
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info("Saved best config to %s", output_path)

    def get_trial_results_df(self) -> pd.DataFrame:
        """Get trial results as DataFrame."""
        return pd.DataFrame(self.trial_results)


def tune_chronos2(
    data_config: DataConfigLong,
    tuning_config: TuningConfigLong,
    symbols: Optional[List[str]] = None,
    method: str = "grid",
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """Run Chronos2 hyperparameter tuning.

    Args:
        data_config: Data configuration
        tuning_config: Tuning configuration
        symbols: Optional subset of symbols
        method: "grid" or "optuna"

    Returns:
        Tuple of (best_config, best_metrics)
    """
    tuner = Chronos2Tuner(data_config, tuning_config)

    if method == "optuna":
        best_config = tuner.run_optuna_search(symbols)
    else:
        best_config = tuner.run_grid_search(symbols)

    return best_config, tuner.best_metrics
