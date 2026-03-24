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

import pandas as pd

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

def _available_pairs(
    candidates: list[str],
    src_dir: Path,
    suffixes=("USDT", "FDUSD", "BUSD"),
    require_start_before: str | None = None,
    require_end_after: str | None = None,
    min_rows: int = 300,
) -> list[str]:
    """Return full pair names (e.g. BTCUSDT) for candidates with at least one data file.

    Optionally filter by:
      require_start_before: symbol must have data starting before this ISO date
      require_end_after:    symbol must have data ending after this ISO date
      min_rows:             minimum number of rows required
    """
    start_cutoff = pd.Timestamp(require_start_before, tz="UTC") if require_start_before else None
    end_cutoff   = pd.Timestamp(require_end_after,    tz="UTC") if require_end_after    else None

    present = []
    for s in candidates:
        for suf in suffixes:
            pair = f"{s}{suf}"
            path = src_dir / f"{pair}.csv"
            if not path.exists():
                continue
            if start_cutoff is not None or end_cutoff is not None or min_rows:
                try:
                    df = pd.read_csv(path, usecols=["timestamp"])
                    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
                    df = df.dropna(subset=["timestamp"])
                    if len(df) < min_rows:
                        break
                    if start_cutoff is not None and df["timestamp"].min() >= start_cutoff:
                        break  # starts too late
                    if end_cutoff is not None and df["timestamp"].max() <= end_cutoff:
                        break  # ends too early
                except Exception:
                    break
            present.append(pair)
            break
    return present


def _run_export_daily(label: str, pairs: list[str], out_train: Path, out_val: Path) -> bool:
    """Export daily binary using export_data_daily_v3 (--single-output per split)."""
    sym_arg = ",".join(pairs)
    splits = [
        (out_train, "2019-01-01", "2025-08-31"),
        (out_val,   "2025-09-01", "2026-03-31"),
    ]
    for out_path, start, end in splits:
        if out_path.exists():
            print(f"  SKIP {out_path.name} (already exists)")
            continue
        cmd = [
            sys.executable, "-m", "pufferlib_market.export_data_daily_v3",
            "--symbols", sym_arg,
            "--data-root", str(DAILY_SRC),
            "--hourly-root", str(HOURLY_SRC),
            "--single-output", str(out_path),
            "--start-date", start,
            "--end-date", end,
        ]
        print(f"  Exporting {out_path.name} ({len(pairs)} pairs)...")
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            print(f"  ERROR: {out_path.name} failed")
            return False
    return True


def _run_export_hourly(label: str, pairs: list[str], out_train: Path, out_val: Path) -> bool:
    """Export hourly binary using export_data_hourly_priceonly."""
    sym_arg = ",".join(pairs)
    splits = [
        (out_train, "2019-01-01", "2025-08-31"),
        (out_val,   "2025-09-01", "2026-03-31"),
    ]
    for out_path, start, end in splits:
        if out_path.exists():
            print(f"  SKIP {out_path.name} (already exists)")
            continue
        cmd = [
            sys.executable, "-m", "pufferlib_market.export_data_hourly_priceonly",
            "--symbols", sym_arg,
            "--data-root", str(HOURLY_SRC),
            "--output", str(out_path),
            "--start-date", start,
            "--end-date", end,
        ]
        print(f"  Exporting {out_path.name} ({len(pairs)} pairs)...")
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            print(f"  ERROR: {out_path.name} failed")
            return False
    return True


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Filter: symbol must have started before 2024-01-01 and ended after 2025-12-01
    # This ensures enough training history and at least 3 months of val coverage.
    # Excludes MKRUSDT (ended 2025-09-15), FTMUSDT (ended 2025-03-18), and very new listings.
    kw = dict(require_start_before="2024-01-01", require_end_after="2025-12-01", min_rows=300)

    # crypto40 = EXPANDED_40 only (with data available)
    e40_daily  = _available_pairs(EXPANDED_40, DAILY_SRC,  **kw)
    e40_hourly = _available_pairs(EXPANDED_40, HOURLY_SRC, **kw)

    # crypto70 = ORIGINAL_30 + available EXPANDED_40
    o30_daily  = _available_pairs(ORIGINAL_30, DAILY_SRC,  **kw)
    o30_hourly = _available_pairs(ORIGINAL_30, HOURLY_SRC, **kw)

    c70_daily  = o30_daily  + [s for s in e40_daily  if s not in o30_daily]
    c70_hourly = o30_hourly + [s for s in e40_hourly if s not in o30_hourly]

    print(f"crypto40 daily   : {len(e40_daily)} pairs  ({', '.join(e40_daily[:5])}...)")
    print(f"crypto40 hourly  : {len(e40_hourly)} pairs")
    print(f"crypto70 daily   : {len(c70_daily)} pairs")
    print(f"crypto70 hourly  : {len(c70_hourly)} pairs")

    if len(e40_daily) >= 5:
        print(f"\n== crypto40_daily ({len(e40_daily)} pairs) ==")
        _run_export_daily("crypto40_daily", e40_daily,
                          DATA_DIR / "crypto40_daily_train.bin",
                          DATA_DIR / "crypto40_daily_val.bin")

    if len(e40_hourly) >= 5:
        print(f"\n== crypto40_hourly ({len(e40_hourly)} pairs) ==")
        _run_export_hourly("crypto40_hourly", e40_hourly,
                           DATA_DIR / "crypto40_hourly_train.bin",
                           DATA_DIR / "crypto40_hourly_val.bin")

    if len(c70_daily) > 30:
        print(f"\n== crypto70_daily ({len(c70_daily)} pairs) ==")
        _run_export_daily("crypto70_daily", c70_daily,
                          DATA_DIR / "crypto70_daily_train.bin",
                          DATA_DIR / "crypto70_daily_val.bin")

    if len(c70_hourly) > 30:
        print(f"\n== crypto70_hourly ({len(c70_hourly)} pairs) ==")
        _run_export_hourly("crypto70_hourly", c70_hourly,
                           DATA_DIR / "crypto70_hourly_train.bin",
                           DATA_DIR / "crypto70_hourly_val.bin")

    print("\nDone.")


if __name__ == "__main__":
    main()
