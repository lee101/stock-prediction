#!/usr/bin/env python3
"""Export a wider Binance USD-proxy daily MKTD dataset for PPO training."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pufferlib_market.export_data_daily import export_binary


BINANCE33_SYMBOLS = [
    "AAVEUSD",
    "ADAUSD",
    "ALGOUSD",
    "APTUSD",
    "ARBUSD",
    "ATOMUSD",
    "AVAXUSD",
    "BCHUSD",
    "BNBUSD",
    "BTCUSD",
    "DOGEUSD",
    "DOTUSD",
    "ETHUSD",
    "FILUSD",
    "ICPUSD",
    "INJUSD",
    "LINKUSD",
    "LTCUSD",
    "NEARUSD",
    "OPUSD",
    "PEPEUSD",
    "POLUSD",
    "RNDRUSD",
    "SEIUSD",
    "SHIBUSD",
    "SOLUSD",
    "SUIUSD",
    "TAOUSD",
    "TIAUSD",
    "TRXUSD",
    "UNIUSD",
    "XLMUSD",
    "XRPUSD",
]

DEFAULT_DATA_ROOT = Path("trainingdatadailybinance")
DEFAULT_OUT_DIR = Path("pufferlib_market/data")
DEFAULT_TRAIN_START = "2021-01-01"
DEFAULT_TRAIN_END = "2025-09-30"
DEFAULT_VAL_START = "2025-10-01"
DEFAULT_VAL_END = "2026-03-14"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export Binance33 daily MKTD train/val files.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--train-start", default=DEFAULT_TRAIN_START)
    parser.add_argument("--train-end", default=DEFAULT_TRAIN_END)
    parser.add_argument("--val-start", default=DEFAULT_VAL_START)
    parser.add_argument("--val-end", default=DEFAULT_VAL_END)
    parser.add_argument("--symbols", default=",".join(BINANCE33_SYMBOLS))
    args = parser.parse_args(argv)

    symbols = [s.strip().upper() for s in str(args.symbols).split(",") if s.strip()]
    missing = [s for s in symbols if not (args.data_root / f"{s}.csv").exists()]
    if missing:
        print(f"ERROR: missing daily CSVs under {args.data_root}: {missing}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_out = args.out_dir / "binance33_daily_train.bin"
    val_out = args.out_dir / "binance33_daily_val.bin"

    print(f"Exporting {len(symbols)} Binance daily symbols from {args.data_root}")
    print(f"Train: {args.train_start} -> {args.train_end}")
    export_binary(
        symbols,
        args.data_root,
        train_out,
        start_date=args.train_start,
        end_date=args.train_end,
        min_days=200,
        union_dates=True,
    )

    print(f"\nVal: {args.val_start} -> {args.val_end}")
    export_binary(
        symbols,
        args.data_root,
        val_out,
        start_date=args.val_start,
        end_date=args.val_end,
        min_days=50,
        union_dates=True,
    )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
