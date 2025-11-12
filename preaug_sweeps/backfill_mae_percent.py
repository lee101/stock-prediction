#!/usr/bin/env python3
"""Backfill mae_percent for existing pre-augmentation sweep results."""

import argparse
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from augmentations import get_augmentation
from augmented_dataset import AugmentedDatasetBuilder
from kronostraining.metrics_utils import compute_mae_percent
from sweep_runner import PreAugmentationSweep

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _build_augmented_series(
    symbol: str,
    strategy: str,
    data_dir: Path,
    validation_days: int,
) -> np.ndarray:
    """Return the validation window of augmented close values."""

    augmentation = get_augmentation(strategy)
    builder = AugmentedDatasetBuilder(
        source_dir=str(data_dir),
        augmentation=augmentation,
        target_symbols=[symbol],
    )

    temp_dir = Path(tempfile.mkdtemp(prefix=f"preaug_backfill_{symbol}_{strategy}_"))
    augmented_dir = builder.create_augmented_dataset(str(temp_dir))

    try:
        csv_path = augmented_dir / f"{symbol}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Augmented CSV missing: {csv_path}")
        df = pd.read_csv(csv_path)
        if "close" not in df.columns:
            raise ValueError(f"close column missing in {csv_path}")
        if len(df) < validation_days:
            raise ValueError(
                f"Dataset for {symbol} shorter than validation window ({len(df)} < {validation_days})"
            )
        return df["close"].to_numpy(dtype=np.float64)[-validation_days:]
    finally:
        builder.cleanup()
        shutil.rmtree(temp_dir, ignore_errors=True)


def _update_result_file(
    symbol: str,
    strategy_dir: Path,
    data_dir: Path,
    validation_days: int,
) -> Optional[Dict[str, object]]:
    result_path = strategy_dir / "result.json"
    if not result_path.exists():
        return None

    with result_path.open() as fp:
        payload = json.load(fp)

    if payload.get("status") != "success":
        return payload

    actual_series = _build_augmented_series(symbol, payload["strategy"], data_dir, validation_days)
    mae_percent = compute_mae_percent(float(payload["mae"]), actual_series)
    payload["mae_percent"] = mae_percent

    with result_path.open("w") as fp:
        json.dump(payload, fp, indent=2)

    metrics_path = strategy_dir / "metrics" / "evaluation.json"
    if metrics_path.exists():
        with metrics_path.open() as fp:
            metrics_payload = json.load(fp)
        metrics_payload.setdefault("aggregate", {})["mae_percent"] = mae_percent
        for entry in metrics_payload.get("per_symbol", []):
            entry["mae_percent"] = mae_percent
        with metrics_path.open("w") as fp:
            json.dump(metrics_payload, fp, indent=2)

    logger.info(
        "Updated %s/%s: MAE=%.6f -> MAE%%=%.4f",
        symbol,
        payload["strategy"],
        payload["mae"],
        mae_percent,
    )
    return payload


def load_existing_results(results_root: Path, symbols: Optional[List[str]] = None) -> Dict[str, Dict[str, Dict[str, object]]]:
    payloads: Dict[str, Dict[str, Dict[str, object]]] = {}
    for symbol_dir in sorted(results_root.iterdir()):
        if not symbol_dir.is_dir():
            continue
        symbol = symbol_dir.name
        if symbols and symbol not in symbols:
            continue

        symbol_results: Dict[str, Dict[str, object]] = {}
        for strategy_dir in sorted(symbol_dir.iterdir()):
            if not strategy_dir.is_dir():
                continue
            strategy = strategy_dir.name
            result_path = strategy_dir / "result.json"
            if not result_path.exists():
                continue
            with result_path.open() as fp:
                payload = json.load(fp)
            symbol_results[strategy] = payload
        if symbol_results:
            payloads[symbol] = symbol_results
    return payloads


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill mae_percent for pre-augmentation sweeps")
    parser.add_argument("--results-dir", type=Path, default=Path("preaug_sweeps/results"))
    parser.add_argument("--data-dir", type=Path, default=Path("trainingdata"))
    parser.add_argument("--best-dir", type=Path, default=Path("preaugstrategies/best"))
    parser.add_argument("--validation-days", type=int, default=30)
    parser.add_argument("--symbols", nargs="*", default=None)
    args = parser.parse_args()

    if not args.results_dir.exists():
        raise SystemExit(f"Results directory not found: {args.results_dir}")

    updated_results: Dict[str, Dict[str, Dict[str, object]]] = {}

    for symbol_dir in sorted(args.results_dir.iterdir()):
        if not symbol_dir.is_dir():
            continue
        symbol = symbol_dir.name
        if args.symbols and symbol not in args.symbols:
            continue

        symbol_payloads: Dict[str, Dict[str, object]] = {}
        for strategy_dir in sorted(symbol_dir.iterdir()):
            if not strategy_dir.is_dir():
                continue
            result = _update_result_file(symbol, strategy_dir, args.data_dir, args.validation_days)
            if result is not None:
                symbol_payloads[result["strategy"]] = result
        if symbol_payloads:
            updated_results[symbol] = symbol_payloads

    if not updated_results:
        logger.warning("No results updated. Nothing to do.")
        return 0

    sweep = PreAugmentationSweep(
        data_dir=str(args.data_dir),
        symbols=list(updated_results.keys()),
        best_configs_dir=str(args.best_dir),
        selection_metric="mae_percent",
    )
    sweep.symbols = list(updated_results.keys())
    sweep.results = updated_results
    sweep._save_best_configs()

    logger.info("Completed mae_percent backfill for %d symbols", len(updated_results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
