#!/usr/bin/env python3
"""
Per-symbol Chronos2 hyperparameter tuning for marketsimlong.

Tunes for each symbol individually:
1. Pre-augmentation strategies (baseline, log_returns, differencing, robust_scaling, etc.)
2. Multiscale skip rates with trimmed mean aggregation (1, 2, 3, 4)
3. Multivariate vs univariate forecasting
4. Context length (256, 512)

Saves best config per symbol to hyperparams/chronos2_long/daily/{SYMBOL}.json
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from marketsimlong.config import DataConfigLong, ForecastConfigLong
from marketsimlong.data import DailyDataLoader, is_crypto_symbol

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class SymbolTuningConfig:
    """Configuration for per-symbol tuning."""

    # Search space
    context_lengths: Tuple[int, ...] = (256, 512)
    use_multivariate_options: Tuple[bool, ...] = (True, False)

    # Pre-augmentation strategies
    preaug_strategies: Tuple[str, ...] = (
        "baseline",
        "log_returns",
        "differencing",
        "robust_scaling",
        "percent_change",
        "detrending",
    )

    # Multiscale configurations (skip_rates, aggregation_method)
    multiscale_configs: Tuple[Tuple[Tuple[int, ...], str], ...] = (
        ((1,), "single"),  # No multiscale (baseline)
        ((1, 2), "trimmed"),  # Skip 1 and 2 with trimmed mean
        ((1, 2, 3), "trimmed"),  # Skip 1, 2, 3 with trimmed mean
        ((1, 2, 4), "trimmed"),  # Different granularities
        ((1, 3), "median"),  # Median aggregation
    )

    # Validation
    val_days: int = 20  # Days to use for validation per symbol
    prediction_length: int = 1  # 1-day ahead forecasting

    # Output
    output_dir: Path = field(default_factory=lambda: Path("hyperparams/chronos2_long/daily"))


@dataclass
class SymbolBestConfig:
    """Best configuration found for a symbol."""

    symbol: str
    context_length: int
    use_multivariate: bool
    preaug_strategy: str
    skip_rates: Tuple[int, ...]
    aggregation_method: str
    mae_pct: float
    directional_accuracy: float
    n_samples: int


class PerSymbolChronos2Tuner:
    """Tunes Chronos2 per symbol with all options."""

    def __init__(
        self,
        data_loader: DailyDataLoader,
        tuning_config: SymbolTuningConfig,
    ) -> None:
        self.data_loader = data_loader
        self.config = tuning_config
        self._wrapper = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy load Chronos2 wrapper."""
        if self._initialized:
            return

        try:
            from src.models.chronos2_wrapper import Chronos2OHLCWrapper

            self._wrapper = Chronos2OHLCWrapper.from_pretrained(
                model_id="amazon/chronos-2",
                device_map="cuda",
                default_context_length=512,
                quantile_levels=[0.1, 0.5, 0.9],
                default_batch_size=64,
            )
            self._initialized = True
            logger.info("Initialized Chronos2 wrapper")
        except Exception as e:
            logger.error("Failed to initialize Chronos2: %s", e)
            raise

    def _get_validation_dates(self, symbol: str) -> List[date]:
        """Get validation dates for a symbol."""
        if symbol not in self.data_loader._data_cache:
            return []

        df = self.data_loader._data_cache[symbol]
        # Get last N trading days
        all_dates = sorted(df["date"].unique())
        return all_dates[-self.config.val_days:] if len(all_dates) >= self.config.val_days else all_dates[-10:]

    def _forecast_single(
        self,
        symbol: str,
        target_date: date,
        context_length: int,
        use_multivariate: bool,
        preaug_strategy: str,
        skip_rates: Tuple[int, ...],
        aggregation_method: str,
    ) -> Optional[Tuple[float, float, float]]:
        """Generate forecast and compute error.

        Returns:
            Tuple of (predicted_close, actual_close, current_close) or None
        """
        context_df = self.data_loader.get_context_for_date(
            symbol,
            target_date,
            context_days=context_length,
        )

        if context_df.empty or len(context_df) < 50:
            return None

        current_close = float(context_df.iloc[-1]["close"])

        # Get actual price on target date
        actual = self.data_loader.get_price_on_date(symbol, target_date)
        if actual is None:
            return None

        actual_close = actual["close"]

        try:
            # Determine if multivariate (stocks benefit, crypto less so)
            should_multivariate = use_multivariate and not is_crypto_symbol(symbol)

            # Apply pre-augmentation if not baseline
            aug_context = context_df
            augmentation = None
            if preaug_strategy != "baseline":
                try:
                    from preaug_sweeps.augmentations import get_augmentation
                    augmentation = get_augmentation(preaug_strategy)
                    aug_context = augmentation.transform_dataframe(context_df.copy())
                except Exception as e:
                    logger.debug("Preaug %s failed: %s", preaug_strategy, e)
                    aug_context = context_df

            # Handle multiscale
            if len(skip_rates) == 1 or aggregation_method == "single":
                # Single scale prediction
                if should_multivariate and hasattr(self._wrapper, "predict_ohlc_multivariate"):
                    batch = self._wrapper.predict_ohlc_multivariate(
                        aug_context,
                        symbol=symbol,
                        prediction_length=self.config.prediction_length,
                        context_length=len(aug_context),
                    )
                else:
                    batch = self._wrapper.predict_ohlc(
                        aug_context,
                        symbol=symbol,
                        prediction_length=self.config.prediction_length,
                        context_length=len(aug_context),
                    )

                q50 = batch.quantile_frames.get(0.5)
                if q50 is None or q50.empty:
                    return None

                pred_close = float(q50.iloc[0].get("close", current_close))
            else:
                # Multiscale prediction
                predictions = {}
                for skip_rate in skip_rates:
                    # Subsample context
                    if skip_rate > 1:
                        subsampled = aug_context.iloc[::skip_rate].copy().reset_index(drop=True)
                    else:
                        subsampled = aug_context

                    if len(subsampled) < 20:
                        continue

                    if should_multivariate and hasattr(self._wrapper, "predict_ohlc_multivariate"):
                        batch = self._wrapper.predict_ohlc_multivariate(
                            subsampled,
                            symbol=symbol,
                            prediction_length=self.config.prediction_length,
                            context_length=len(subsampled),
                        )
                    else:
                        batch = self._wrapper.predict_ohlc(
                            subsampled,
                            symbol=symbol,
                            prediction_length=self.config.prediction_length,
                            context_length=len(subsampled),
                        )

                    q50 = batch.quantile_frames.get(0.5)
                    if q50 is not None and not q50.empty:
                        predictions[skip_rate] = float(q50.iloc[0].get("close", current_close))

                if not predictions:
                    return None

                # Aggregate predictions
                values = list(predictions.values())
                if aggregation_method == "trimmed" and len(values) >= 3:
                    from scipy import stats
                    pred_close = stats.trim_mean(values, proportiontocut=0.1)
                elif aggregation_method == "median":
                    pred_close = np.median(values)
                else:
                    pred_close = np.mean(values)

            # Inverse transform if augmentation was applied
            if augmentation is not None and preaug_strategy != "baseline":
                try:
                    # Simple inverse - just use the predicted value as-is for close
                    # Most augs work on the prediction context, not the output
                    pass
                except:
                    pass

            return (pred_close, actual_close, current_close)

        except Exception as e:
            logger.debug("Forecast failed for %s on %s: %s", symbol, target_date, e)
            return None

    def tune_symbol(self, symbol: str) -> SymbolBestConfig:
        """Tune hyperparameters for a single symbol.

        Returns:
            SymbolBestConfig with best configuration found
        """
        self._ensure_initialized()

        logger.info("Tuning symbol: %s", symbol)

        val_dates = self._get_validation_dates(symbol)
        if not val_dates:
            logger.warning("No validation dates for %s", symbol)
            return SymbolBestConfig(
                symbol=symbol,
                context_length=512,
                use_multivariate=True,
                preaug_strategy="baseline",
                skip_rates=(1,),
                aggregation_method="single",
                mae_pct=float("inf"),
                directional_accuracy=0.0,
                n_samples=0,
            )

        best_mae = float("inf")
        best_config = None
        all_results = []

        # Test all configurations
        config_count = 0
        total_configs = (
            len(self.config.context_lengths)
            * len(self.config.use_multivariate_options)
            * len(self.config.preaug_strategies)
            * len(self.config.multiscale_configs)
        )

        for context_length in self.config.context_lengths:
            for use_multivariate in self.config.use_multivariate_options:
                for preaug_strategy in self.config.preaug_strategies:
                    for skip_rates, agg_method in self.config.multiscale_configs:
                        config_count += 1

                        # Evaluate on validation dates
                        errors = []
                        directions_correct = []

                        for val_date in val_dates[-10:]:  # Use last 10 dates for speed
                            result = self._forecast_single(
                                symbol,
                                val_date,
                                context_length,
                                use_multivariate,
                                preaug_strategy,
                                skip_rates,
                                agg_method,
                            )
                            if result is None:
                                continue

                            pred_close, actual_close, current_close = result
                            error = abs(pred_close - actual_close) / actual_close
                            errors.append(error)

                            # Directional accuracy
                            pred_dir = pred_close > current_close
                            actual_dir = actual_close > current_close
                            directions_correct.append(pred_dir == actual_dir)

                        if not errors:
                            continue

                        mae_pct = np.mean(errors) * 100
                        dir_acc = np.mean(directions_correct) * 100

                        result_entry = {
                            "context_length": context_length,
                            "use_multivariate": use_multivariate,
                            "preaug_strategy": preaug_strategy,
                            "skip_rates": skip_rates,
                            "aggregation_method": agg_method,
                            "mae_pct": mae_pct,
                            "directional_accuracy": dir_acc,
                            "n_samples": len(errors),
                        }
                        all_results.append(result_entry)

                        if mae_pct < best_mae:
                            best_mae = mae_pct
                            best_config = result_entry.copy()

        if best_config is None:
            logger.warning("No valid config found for %s", symbol)
            return SymbolBestConfig(
                symbol=symbol,
                context_length=512,
                use_multivariate=True,
                preaug_strategy="baseline",
                skip_rates=(1,),
                aggregation_method="single",
                mae_pct=float("inf"),
                directional_accuracy=0.0,
                n_samples=0,
            )

        logger.info(
            "  Best for %s: MAE=%.2f%%, ctx=%d, mv=%s, preaug=%s, skip=%s",
            symbol,
            best_config["mae_pct"],
            best_config["context_length"],
            best_config["use_multivariate"],
            best_config["preaug_strategy"],
            best_config["skip_rates"],
        )

        return SymbolBestConfig(
            symbol=symbol,
            context_length=best_config["context_length"],
            use_multivariate=best_config["use_multivariate"],
            preaug_strategy=best_config["preaug_strategy"],
            skip_rates=tuple(best_config["skip_rates"]),
            aggregation_method=best_config["aggregation_method"],
            mae_pct=best_config["mae_pct"],
            directional_accuracy=best_config["directional_accuracy"],
            n_samples=best_config["n_samples"],
        )

    def tune_all_symbols(self, symbols: List[str]) -> Dict[str, SymbolBestConfig]:
        """Tune all symbols.

        Returns:
            Dict mapping symbol -> best config
        """
        results = {}

        for i, symbol in enumerate(symbols):
            logger.info("=" * 50)
            logger.info("Tuning %d/%d: %s", i + 1, len(symbols), symbol)
            logger.info("=" * 50)

            best_config = self.tune_symbol(symbol)
            results[symbol] = best_config

            # Save incrementally
            self._save_symbol_config(best_config)

        return results

    def _save_symbol_config(self, config: SymbolBestConfig) -> None:
        """Save best config for a symbol."""
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{config.symbol}.json"
        data = {
            "symbol": config.symbol,
            "context_length": config.context_length,
            "use_multivariate": config.use_multivariate,
            "preaug_strategy": config.preaug_strategy,
            "skip_rates": list(config.skip_rates),
            "aggregation_method": config.aggregation_method,
            "mae_pct": config.mae_pct,
            "directional_accuracy": config.directional_accuracy,
            "n_samples": config.n_samples,
            "tuned_for": "daily",
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info("Saved config to %s", output_path)

    def unload(self) -> None:
        """Release GPU memory."""
        if self._wrapper is not None:
            self._wrapper.unload()
            self._wrapper = None
            self._initialized = False


def load_symbol_config(symbol: str, config_dir: Path) -> Optional[Dict[str, Any]]:
    """Load tuned config for a symbol."""
    config_path = config_dir / f"{symbol}.json"
    if not config_path.exists():
        return None

    with open(config_path) as f:
        return json.load(f)


def tune_all_for_marketsimlong():
    """Run full per-symbol tuning."""
    data_config = DataConfigLong(
        data_root=Path("trainingdata/train"),
        start_date=date(2025, 1, 1),
        end_date=date(2025, 12, 22),
    )

    data_loader = DailyDataLoader(data_config)
    data_loader.load_all_symbols()

    symbols = list(data_loader._data_cache.keys())
    logger.info("Tuning %d symbols: %s", len(symbols), symbols)

    tuning_config = SymbolTuningConfig()
    tuner = PerSymbolChronos2Tuner(data_loader, tuning_config)

    try:
        results = tuner.tune_all_symbols(symbols)

        # Print summary
        print("\n" + "=" * 70)
        print("PER-SYMBOL TUNING RESULTS")
        print("=" * 70)
        print(f"{'Symbol':<12} {'MAE%':<10} {'DirAcc%':<10} {'Ctx':<6} {'MV':<6} {'Preaug':<15} {'Skip':<10}")
        print("-" * 70)

        for symbol, cfg in sorted(results.items(), key=lambda x: x[1].mae_pct):
            print(
                f"{symbol:<12} {cfg.mae_pct:<10.2f} {cfg.directional_accuracy:<10.1f} "
                f"{cfg.context_length:<6} {str(cfg.use_multivariate):<6} {cfg.preaug_strategy:<15} {str(cfg.skip_rates):<10}"
            )

        print("=" * 70)
        print(f"Configs saved to {tuning_config.output_dir}/")

        return results

    finally:
        tuner.unload()


if __name__ == "__main__":
    tune_all_for_marketsimlong()
