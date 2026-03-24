#!/usr/bin/env python3
"""Export crypto40 and crypto70 daily + hourly datasets from downloaded Binance data.

Run after download_binance_data.py finishes.

Usage:
    source .venv313/bin/activate
    python export_crypto40_70.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent

# Symbol groups
ORIGINAL_30 = [
    "BTC", "ETH", "SOL", "DOGE", "AVAX", "LINK", "AAVE", "LTC", "XRP", "DOT",
    "UNI", "NEAR", "APT", "ICP", "SHIB", "ADA", "FIL", "ARB", "OP", "INJ",
    "SUI", "TIA", "SEI", "ATOM", "ALGO", "BCH", "BNB", "TRX", "PEPE", "MATIC",
]

EXPANDED_40 = [
    "HBAR", "VET", "RENDER", "FET", "GRT",
    "SAND", "MANA", "AXS", "CRV", "COMP",
    "MKR", "SNX", "ENJ", "1INCH", "SUSHI",
    "YFI", "BAT", "ZRX", "THETA", "FTM",
    "RUNE", "KAVA", "EGLD", "CHZ", "GALA",
    "APE", "LDO", "GMX", "PENDLE", "WLD",
    "JUP", "W", "ENA", "STX", "FLOKI",
    "TON", "KAS", "ONDO", "JASMY", "CFX",
]

DATA_DIR = REPO / "pufferlib_market" / "data"
DAILY_SRC = REPO / "trainingdata" / "train"
HOURLY_SRC = REPO / "trainingdatahourlybinance"

# ── Helpers ──────────────────────────────────────────────────────────────────

def _available_symbols(candidates: list[str], src_dir: Path, suffixes=("USDT", "FDUSD", "BUSD")) -> list[str]:
    """Return symbols from candidates that have at least one data file in src_dir."""
    present = []
    for s in candidates:
        for suf in suffixes:
            if (src_dir / f"{s}{suf}.csv").exists():
                present.append(s)
                break
    return present


def _run_export(label: str, symbols: list[str], out_train: Path, out_val: Path,
                daily: bool = True) -> bool:
    sym_arg = ",".join(symbols)
    train_end = "2025-08-31"
    val_start = "2025-09-01"

    script = "pufferlib_market.export_data_daily_v3" if daily else "pufferlib_market.export_data_hourly_priceonly"

    for out_path, date_range in [(out_train, ("2019-01-01", train_end)),
                                  (out_val,   (val_start,  "2026-03-20"))]:
        if out_path.exists():
            print(f"  SKIP {out_path.name} (already exists)")
            continue

        cmd = [
            sys.executable, "-m", script,
            "--symbols", sym_arg,
            "--output", str(out_path),
            "--start", date_range[0],
            "--end",   date_range[1],
        ]
        print(f"  Exporting {out_path.name} ({len(symbols)} syms)...")
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            print(f"  ERROR: {out_path.name} failed")
            return False
    return True


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    # crypto40 = EXPANDED_40 only (with data available)
    e40_daily  = _available_symbols(EXPANDED_40, DAILY_SRC)
    e40_hourly = _available_symbols(EXPANDED_40, HOURLY_SRC)

    # crypto70 = ORIGINAL_30 + available EXPANDED_40
    o30_daily  = _available_symbols(ORIGINAL_30, DAILY_SRC)
    o30_hourly = _available_symbols(ORIGINAL_30, HOURLY_SRC)

    c70_daily  = o30_daily  + [s for s in e40_daily  if s not in o30_daily]
    c70_hourly = o30_hourly + [s for s in e40_hourly if s not in o30_hourly]

    print(f"crypto40 daily   : {len(e40_daily)} symbols available")
    print(f"crypto40 hourly  : {len(e40_hourly)} symbols available")
    print(f"crypto70 daily   : {len(c70_daily)} symbols available")
    print(f"crypto70 hourly  : {len(c70_hourly)} symbols available")

    exports = []
    if len(e40_daily) >= 5:
        exports.append(("crypto40_daily", e40_daily, True))
    if len(e40_hourly) >= 5:
        exports.append(("crypto40_hourly", e40_hourly, False))
    if len(c70_daily) > 30:
        exports.append(("crypto70_daily", c70_daily, True))
    if len(c70_hourly) > 30:
        exports.append(("crypto70_hourly", c70_hourly, False))

    for label, syms, is_daily in exports:
        print(f"\n== {label} ({len(syms)} symbols) ==")
        out_train = DATA_DIR / f"{label}_train.bin"
        out_val   = DATA_DIR / f"{label}_val.bin"
        _run_export(label, syms, out_train, out_val, daily=is_daily)

    print("\nDone.")


if __name__ == "__main__":
    main()
