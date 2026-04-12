#!/usr/bin/env python3
"""
Generate default Chronos2 hyperparam configs for all symbols in trainingdata/
that don't already have one.  Used to bootstrap preaug sweeps for new symbols.

The default is ctx=1024, scaler=none, mv=False (univariate), batch=128.
These are the most common settings in the existing 64 calibrated configs,
and perform well across a wide range of stocks.

After preaug sweeps are done, run benchmark_chronos2.py --search-method direct
--update-hyperparams to find the true best config per symbol.

Usage:
    python scripts/generate_default_hyperparams.py
    python scripts/generate_default_hyperparams.py --data-dir trainingdata --min-rows 200
    python scripts/generate_default_hyperparams.py --force   # overwrite existing
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Default hyperparam config — matches the most common existing configs
DEFAULT_CONTEXT_LENGTH = 1024
DEFAULT_BATCH_SIZE = 128
DEFAULT_SCALER = "none"
DEFAULT_MULTIVARIATE = False
DEFAULT_AGGREGATION = "median"
DEFAULT_SAMPLE_COUNT = 0
DEFAULT_QUANTILE_LEVELS = [0.1, 0.5, 0.9]


def make_config(symbol: str, context_length: int = DEFAULT_CONTEXT_LENGTH) -> dict:
    return {
        "symbol": symbol,
        "model": "chronos2",
        "config": {
            "name": f"default_ctx{context_length}_bs{DEFAULT_BATCH_SIZE}_{DEFAULT_AGGREGATION}_uni",
            "model_id": "amazon/chronos-2",
            "device_map": "cuda",
            "context_length": context_length,
            "batch_size": DEFAULT_BATCH_SIZE,
            "quantile_levels": DEFAULT_QUANTILE_LEVELS,
            "aggregation": DEFAULT_AGGREGATION,
            "sample_count": DEFAULT_SAMPLE_COUNT,
            "scaler": DEFAULT_SCALER,
            "use_multivariate": DEFAULT_MULTIVARIATE,
            "predict_kwargs": {},
        },
        "validation": {},
        "test": {},
        "windows": {
            "val_window": 20,
            "test_window": 20,
            "forecast_horizon": 1,
        },
        "metadata": {
            "source": "generate_default_hyperparams",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "note": "default config — run benchmark_chronos2.py --search-method direct to calibrate",
        },
    }


def count_rows(csv_path: Path) -> int:
    try:
        with csv_path.open("rb") as f:
            # Fast line count without pandas
            return sum(1 for _ in f) - 1  # subtract header
    except Exception:
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="trainingdata", help="Directory with {symbol}.csv files")
    parser.add_argument("--hyperparam-dir", default="hyperparams/chronos2", help="Output hyperparam directory")
    parser.add_argument("--min-rows", type=int, default=200, help="Minimum CSV rows to generate config")
    parser.add_argument("--force", action="store_true", help="Overwrite existing configs")
    parser.add_argument("--dry-run", action="store_true", help="Count without writing")
    parser.add_argument("--limit", type=int, default=0, help="Limit to first N symbols")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    hp_dir = Path(args.hyperparam_dir)
    hp_dir.mkdir(parents=True, exist_ok=True)

    csvs = sorted(data_dir.glob("*.csv"))
    if args.limit:
        csvs = csvs[:args.limit]

    created = skipped_existing = skipped_thin = 0
    print(f"Scanning {len(csvs)} CSVs in {data_dir}...")

    for csv_path in csvs:
        symbol = csv_path.stem.upper()
        hp_path = hp_dir / f"{symbol}.json"

        if hp_path.exists() and not args.force:
            skipped_existing += 1
            continue

        rows = count_rows(csv_path)
        if rows < args.min_rows:
            skipped_thin += 1
            continue

        # Use shorter context for short series
        ctx = DEFAULT_CONTEXT_LENGTH
        if rows < 500:
            ctx = 256
        elif rows < 800:
            ctx = 512

        if not args.dry_run:
            config = make_config(symbol, context_length=ctx)
            hp_path.write_text(json.dumps(config, indent=2))
        created += 1

    print(f"\nResults:")
    print(f"  Created: {created}")
    print(f"  Skipped (already exists): {skipped_existing}")
    print(f"  Skipped (thin data): {skipped_thin}")
    print(f"  Hyperparam configs in {hp_dir}: {sum(1 for _ in hp_dir.glob('*.json'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
