#!/usr/bin/env python3
"""
Pre-Augmentation Sweep Runner

Tests multiple pre-augmentation strategies to find which improves MAE.
Runs training for each strategy and compares results.
"""

import argparse
import json
import logging
import math
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import sys

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from augmentations import get_augmentation, AUGMENTATION_REGISTRY
from augmented_dataset import AugmentedDatasetBuilder
from kronostraining.config import KronosTrainingConfig
from kronostraining.trainer import KronosTrainer

ALLOWED_SELECTION_METRICS = {"mae_percent", "mae", "rmse", "mape"}


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('preaug_sweeps/logs/sweep.log')
    ]
)
logger = logging.getLogger(__name__)


class PreAugmentationSweep:
    """Manages the pre-augmentation sweep experiment."""

    def __init__(
        self,
        data_dir: str = "trainingdata",
        symbols: Optional[List[str]] = None,
        strategies: Optional[List[str]] = None,
        epochs: int = 3,
        batch_size: int = 16,
        lookback: int = 64,
        horizon: int = 30,
        validation_days: int = 30,
        best_configs_dir: Union[str, Path] = "preaugstrategies/best",
        selection_metric: str = "mae_percent",
    ):
        """
        Args:
            data_dir: Path to original training data
            symbols: List of symbols to test (e.g., ["ETHUSD", "UNIUSD", "BTCUSD"])
            strategies: List of augmentation strategies to test (None = all)
            epochs: Training epochs for each run
            batch_size: Batch size
            lookback: Lookback window
            horizon: Forecast horizon
            validation_days: Validation period
        """
        self.data_dir = Path(data_dir)
        self.symbols = symbols or ["ETHUSD", "UNIUSD", "BTCUSD"]
        self.strategies = strategies or list(AUGMENTATION_REGISTRY.keys())
        self.epochs = epochs
        self.batch_size = batch_size
        self.lookback = lookback
        self.horizon = horizon
        self.validation_days = validation_days

        if selection_metric not in ALLOWED_SELECTION_METRICS:
            raise ValueError(
                f"selection_metric must be one of {sorted(ALLOWED_SELECTION_METRICS)}, got '{selection_metric}'"
            )
        self.selection_metric = selection_metric

        self.results_dir = Path("preaug_sweeps/results")
        self.temp_dir = Path("preaug_sweeps/temp")
        self.reports_dir = Path("preaug_sweeps/reports")
        self.best_configs_dir = Path(best_configs_dir)

        # Create directories
        for dir_path in [self.results_dir, self.temp_dir, self.reports_dir, self.best_configs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.results: Dict[str, Dict[str, Dict]] = {symbol: {} for symbol in self.symbols}

    def run_sweep(self) -> None:
        """Run the complete sweep experiment."""
        logger.info("=" * 80)
        logger.info("Starting Pre-Augmentation Sweep")
        logger.info("=" * 80)
        logger.info(f"Symbols: {self.symbols}")
        logger.info(f"Strategies: {self.strategies}")
        logger.info(f"Epochs: {self.epochs}")
        logger.info("=" * 80)

        start_time = time.time()

        # Test each strategy for each symbol
        for symbol in self.symbols:
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing symbol: {symbol}")
            logger.info(f"{'='*80}\n")

            for strategy_name in self.strategies:
                logger.info(f"\n{'-'*80}")
                logger.info(f"Testing strategy: {strategy_name} on {symbol}")
                logger.info(f"Selection metric: {self.selection_metric}")
                logger.info(f"{'-'*80}")

                try:
                    result = self._test_strategy(symbol, strategy_name)
                    self.results[symbol][strategy_name] = result
                    mae_percent = result.get("mae_percent")
                    percent_display = f", MAE% = {mae_percent:.4f}%" if mae_percent is not None else ""
                    logger.info(
                        f"✓ Completed {strategy_name} on {symbol}: MAE = {result['mae']:.6f}{percent_display}"
                    )
                except Exception as e:
                    logger.error(f"✗ Failed {strategy_name} on {symbol}: {e}", exc_info=True)
                    self.results[symbol][strategy_name] = {
                        "status": "failed",
                        "error": str(e)
                    }

        duration = time.time() - start_time
        logger.info(f"\n{'='*80}")
        logger.info(f"Sweep completed in {duration/60:.2f} minutes")
        logger.info(f"{'='*80}\n")

        # Generate reports
        self._save_results()
        self._generate_reports()
        self._save_best_configs()

        logger.info("\n" + "="*80)
        logger.info("SWEEP COMPLETE!")
        logger.info("="*80)

    def _test_strategy(self, symbol: str, strategy_name: str) -> Dict:
        """
        Test a single augmentation strategy on a symbol.

        Returns:
            Dictionary with results including MAE
        """
        # Create augmentation
        augmentation = get_augmentation(strategy_name)

        # Create augmented dataset
        aug_dir = self.temp_dir / f"{symbol}_{strategy_name}"
        builder = AugmentedDatasetBuilder(
            source_dir=self.data_dir,
            augmentation=augmentation,
            target_symbols=[symbol]
        )
        augmented_path = builder.create_augmented_dataset(str(aug_dir))

        logger.info(f"Created augmented dataset at {augmented_path}")

        try:
            # Train model on augmented data
            output_dir = self.results_dir / symbol / strategy_name
            output_dir.mkdir(parents=True, exist_ok=True)

            config = KronosTrainingConfig(
                data_dir=augmented_path,
                output_dir=output_dir,
                lookback_window=self.lookback,
                prediction_length=self.horizon,
                validation_days=self.validation_days,
                batch_size=self.batch_size,
                epochs=self.epochs,
                learning_rate=4e-5,
                weight_decay=0.01,
                seed=1337,
                torch_compile=False,  # Disabled to avoid Triton compilation issues
                use_fused_optimizer=True,
            )

            logger.info(f"Training with config: epochs={self.epochs}, batch_size={self.batch_size}")
            trainer = KronosTrainer(config)
            summary = trainer.train()

            logger.info("Evaluating on validation set...")
            metrics = trainer.evaluate_holdout()

            # Extract MAE
            mae = metrics["aggregate"]["mae"]
            rmse = metrics["aggregate"]["rmse"]
            mape = metrics["aggregate"]["mape"]
            mae_percent = metrics["aggregate"].get("mae_percent")

            result = {
                "status": "success",
                "strategy": strategy_name,
                "symbol": symbol,
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
                "mae_percent": mae_percent,
                "best_val_loss": summary["best_val_loss"],
                "epochs": self.epochs,
                "config": augmentation.get_config(),
                "timestamp": datetime.utcnow().isoformat(),
            }

            # Save individual result
            result_file = output_dir / "result.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)

            return result

        finally:
            # Cleanup temporary augmented dataset
            logger.info(f"Cleaning up {aug_dir}")
            builder.cleanup()

    def _save_results(self) -> None:
        """Save all results to JSON."""
        results_file = self.reports_dir / f"sweep_results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Saved results to {results_file}")

    def _generate_reports(self) -> None:
        """Generate summary reports."""
        logger.info("\n" + "="*80)
        logger.info("RESULTS SUMMARY")
        logger.info("="*80 + "\n")

        summary_rows = []

        for symbol in self.symbols:
            logger.info(f"\n{symbol}:")
            logger.info("-" * 60)

            symbol_results = []
            for strategy, result in self.results[symbol].items():
                if result.get("status") == "success":
                    mae = result["mae"]
                    rmse = result["rmse"]
                    mape = result["mape"]
                    mae_percent = result.get("mae_percent")
                    mae_percent_str = f"{mae_percent:8.4f}%" if mae_percent is not None else "   n/a  "

                    logger.info(
                        f"  {strategy:25s} | MAE: {mae:10.6f} | MAE%: {mae_percent_str} | RMSE: {rmse:10.6f} | MAPE: {mape:8.4f}%"
                    )

                    symbol_results.append({
                        "strategy": strategy,
                        "mae": mae,
                        "mae_percent": mae_percent,
                        "rmse": rmse,
                        "mape": mape,
                    })
                    summary_rows.append({
                        "symbol": symbol,
                        "strategy": strategy,
                        "mae": mae,
                        "mae_percent": mae_percent,
                        "rmse": rmse,
                        "mape": mape,
                    })

            # Show best for this symbol
            if symbol_results:
                best = min(
                    symbol_results,
                    key=lambda x: self._resolve_selection_value(x)[0]
                )
                logger.info(
                    f"\n  ★ BEST for {symbol}: {best['strategy']} ({self.selection_metric}: {self._resolve_selection_value(best)[0]:.6f})"
                )

        # Create summary CSV
        if summary_rows:
            df_summary = pd.DataFrame(summary_rows)
            summary_csv = self.reports_dir / f"summary_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
            df_summary.to_csv(summary_csv, index=False)
            logger.info(f"\nSummary CSV saved to {summary_csv}")

            # Pivot table for easy comparison
            pivot_mae = df_summary.pivot(index='strategy', columns='symbol', values='mae')
            logger.info("\n" + "="*80)
            logger.info("MAE COMPARISON TABLE")
            logger.info("="*80)
            logger.info("\n" + str(pivot_mae))

            if df_summary["mae_percent"].notna().any():
                pivot_mae_pct = df_summary.pivot(index='strategy', columns='symbol', values='mae_percent')
                logger.info("\n" + "="*80)
                logger.info("MAE% COMPARISON TABLE")
                logger.info("="*80)
                logger.info("\n" + str(pivot_mae_pct))

    def _save_best_configs(self) -> None:
        """Save best configuration for each symbol."""
        logger.info("\n" + "="*80)
        logger.info("SAVING BEST CONFIGURATIONS")
        logger.info("="*80 + "\n")

        for symbol in self.symbols:
            symbol_results = self.results[symbol]
            successful = {
                k: v for k, v in symbol_results.items()
                if v.get("status") == "success"
            }

            if not successful:
                logger.warning(f"No successful runs for {symbol}")
                continue

            # Find best by configured selection metric
            best_strategy = min(successful.items(), key=lambda x: self._resolve_selection_value(x[1])[0])
            strategy_name, result = best_strategy
            selection_value, using_primary = self._resolve_selection_value(result)

            # Save best config
            best_config = {
                "symbol": symbol,
                "best_strategy": strategy_name,
                "mae": result["mae"],
                "mae_percent": result.get("mae_percent"),
                "rmse": result["rmse"],
                "mape": result["mape"],
                "config": result["config"],
                "timestamp": result["timestamp"],
                "selection_metric": self.selection_metric,
                "selection_value": selection_value,
                "comparison": {
                    name: {
                        "mae": res["mae"],
                        "mae_percent": res.get("mae_percent"),
                        "rmse": res["rmse"],
                        "mape": res["mape"],
                    }
                    for name, res in successful.items()
                }
            }

            config_file = self.best_configs_dir / f"{symbol}.json"
            with open(config_file, 'w') as f:
                json.dump(best_config, f, indent=2)

            selection_label = f"{self.selection_metric}{'' if using_primary else ' (fallback to MAE)'}"
            logger.info(
                f"★ {symbol}: Best strategy = {strategy_name} ({selection_label}: {selection_value:.6f})"
            )
            logger.info(f"  Saved to {config_file}")

            # Calculate improvement over baseline
            if "baseline" in successful:
                baseline_value = self._resolve_selection_value(successful["baseline"])[0]
                if math.isfinite(baseline_value) and baseline_value != 0 and math.isfinite(selection_value):
                    improvement = ((baseline_value - selection_value) / baseline_value) * 100
                    logger.info(
                        f"  Improvement over baseline ({self.selection_metric}): {improvement:+.2f}%"
                    )
                else:
                    logger.info("  Improvement over baseline: n/a (missing metrics)")

    def _resolve_selection_value(self, result: Dict[str, float]) -> Tuple[float, bool]:
        """Return the metric value used for selection, falling back to MAE if missing."""

        value = result.get(self.selection_metric)
        if value is None:
            fallback = result.get("mae")
            if fallback is None:
                return float("inf"), False
            return fallback, False
        return value, True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run pre-augmentation sweep")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["ETHUSD", "UNIUSD", "BTCUSD"],
        help="Symbols to test"
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        help="Strategies to test (default: all)"
    )
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lookback", type=int, default=64, help="Lookback window")
    parser.add_argument("--horizon", type=int, default=30, help="Forecast horizon")
    parser.add_argument("--validation-days", type=int, default=30, help="Validation days")
    parser.add_argument("--data-dir", type=str, default="trainingdata", help="Training data directory")
    parser.add_argument("--best-dir", type=str, default="preaugstrategies/best", help="Directory to save best configs")
    parser.add_argument(
        "--selection-metric",
        type=str,
        default="mae_percent",
        choices=sorted(ALLOWED_SELECTION_METRICS),
        help="Metric used to select the best strategy",
    )

    args = parser.parse_args()

    # Create sweep runner
    sweep = PreAugmentationSweep(
        data_dir=args.data_dir,
        symbols=args.symbols,
        strategies=args.strategies,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lookback=args.lookback,
        horizon=args.horizon,
        validation_days=args.validation_days,
        best_configs_dir=args.best_dir,
        selection_metric=args.selection_metric,
    )

    # Run sweep
    sweep.run_sweep()


if __name__ == "__main__":
    main()
