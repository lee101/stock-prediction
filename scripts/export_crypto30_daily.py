#!/usr/bin/env python3
"""Export crypto30 daily MKTD binaries for PPO training."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pufferlib_market.export_data_daily import export_binary

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT",
    "AAVEUSDT", "LTCUSDT", "XRPUSDT", "DOTUSDT", "UNIUSDT", "NEARUSDT",
    "APTUSDT", "ICPUSDT", "SHIBUSDT", "ADAUSDT", "FILUSDT", "ARBUSDT",
    "OPUSDT", "INJUSDT", "SUIUSDT", "TIAUSDT", "SEIUSDT", "ATOMUSDT",
    "ALGOUSDT", "BCHUSDT", "BNBUSDT", "TRXUSDT", "PEPEUSDT", "MATICUSDT",
]

DATA_ROOT = Path("trainingdata/train")
OUT_DIR = Path("pufferlib_market/data")

TRAIN_START = "2020-01-01"
TRAIN_END = "2025-09-30"
VAL_START = "2025-10-01"
VAL_END = "2026-04-10"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Filter to symbols that have daily CSVs
    available = [s for s in SYMBOLS if (DATA_ROOT / f"{s}.csv").exists()]
    missing = set(SYMBOLS) - set(available)
    if missing:
        print(f"Missing daily CSVs for: {sorted(missing)}")
    print(f"Exporting {len(available)} symbols")

    if len(available) < 5:
        print("ERROR: need at least 5 symbols with data")
        return 1

    # Train split -- union_dates=True so staggered crypto launches don't truncate
    # early BTC/ETH data. Newer coins get tradable=0 until their listing date.
    print(f"\n--- Train: {TRAIN_START} to {TRAIN_END} ---")
    export_binary(
        available,
        DATA_ROOT,
        OUT_DIR / "crypto30_daily_train.bin",
        start_date=TRAIN_START,
        end_date=TRAIN_END,
        min_days=200,
        cross_features=False,
        union_dates=True,
    )

    # Val split
    print(f"\n--- Val: {VAL_START} to {VAL_END} ---")
    export_binary(
        available,
        DATA_ROOT,
        OUT_DIR / "crypto30_daily_val.bin",
        start_date=VAL_START,
        end_date=VAL_END,
        min_days=30,
        cross_features=False,
        union_dates=True,
    )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
