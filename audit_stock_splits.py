#!/usr/bin/env python3
"""Comprehensive stock split audit and auto-fix tool.

Usage:
    python audit_stock_splits.py [--dry-run] [--verify] [--no-reexport]

    --dry-run     Show what would be fixed without modifying files
    --verify      Check current state only, print summary, no modifications
    --no-reexport Skip binary re-export after fixes
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PRICE_COLS = ["open", "high", "low", "close"]
VOL_COL = "volume"
DATE_COL = "timestamp"

# Single-day drop threshold that triggers a split check
DROP_THRESHOLD = 0.30

# Tolerance for matching the expected 1/split_factor ratio
SPLIT_RATIO_TOLERANCE = 0.15

# Minimum split factor to consider as a real split.
# yfinance sometimes returns fractional "splits" for spin-offs/adjustments (e.g. 1.128).
# Real forward stock splits are always integer factors >= 2.
MIN_SPLIT_FACTOR = 1.9

# Max parallel workers for yfinance requests
YFINANCE_WORKERS = 40

# Known real events that explain large drops (not splits)
# Format: (symbol, date_str, description)
KNOWN_REAL_EVENTS: list[tuple[str, str, str]] = [
    ("NFLX", "2022-04-20", "Q1 2022 earnings miss (-35%)"),
    ("META", "2022-02-03", "Q4 2021 earnings miss (-26%)"),
    ("INTC", "2024-08-01", "Q2 2024 earnings miss (-26%)"),
]

# Symbols whose names contain commas or look like crypto — skip yfinance for these
# (crypto pairs like BTCUSD, ETHUSD etc. will fail yfinance with 404)
_CRYPTO_SUFFIXES = ("USD", "USDT", "BTC", "ETH")

# Binaries to re-export after fixes
BINARY_EXPORTS = [
    {
        "name": "stocks12_daily_train",
        "symbols": "AAPL,MSFT,NVDA,GOOG,META,TSLA,SPY,QQQ,PLTR,JPM,V,AMZN",
        "start": "2022-02-07",
        "end": "2025-08-31",
        "output": "pufferlib_market/data/stocks12_daily_train.bin",
        "extra_args": [],
    },
    {
        "name": "stocks12_daily_val",
        "symbols": "AAPL,MSFT,NVDA,GOOG,META,TSLA,SPY,QQQ,PLTR,JPM,V,AMZN",
        "start": "2025-09-01",
        "end": "2026-02-28",
        "output": "pufferlib_market/data/stocks12_daily_val.bin",
        "extra_args": ["--min-days", "100"],
    },
    {
        "name": "stocks20_daily_train",
        "symbols": "AAPL,MSFT,NVDA,GOOG,META,TSLA,AMZN,AMD,JPM,SPY,QQQ,PLTR,NET,NFLX,ADBE,CRM,AVGO,V,COST,ADSK",
        "start": "2022-02-07",
        "end": "2025-08-31",
        "output": "pufferlib_market/data/stocks20_daily_train.bin",
        "extra_args": [],
    },
    {
        "name": "stocks20_daily_val",
        "symbols": "AAPL,MSFT,NVDA,GOOG,META,TSLA,AMZN,AMD,JPM,SPY,QQQ,PLTR,NET,NFLX,ADBE,CRM,AVGO,V,COST,ADSK",
        "start": "2025-09-01",
        "end": "2026-02-28",
        "output": "pufferlib_market/data/stocks20_daily_val.bin",
        "extra_args": ["--min-days", "100"],
    },
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class AuditRecord(NamedTuple):
    symbol: str
    split_date: str
    factor: float
    status: str  # OK / FIXED / FIXED_HOURLY / UNRECOGNIZED_DROP / REAL_EVENT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_valid_ticker(sym: str) -> bool:
    """Return True only for symbols that look like real stock tickers.

    Rejects files like 'download_summary', comma-separated multi-symbol files,
    and crypto pairs.  Valid tickers consist of uppercase letters, digits,
    dots, hyphens, and carets (BRK.B, BF.B, SPY^, etc.).
    """
    if not sym:
        return False
    if "," in sym:
        return False
    # Must start with uppercase letter and be composed of valid ticker chars
    if not re.match(r"^[A-Z][A-Z0-9.\-^]*$", sym):
        return False
    return True


def _is_crypto_symbol(sym: str) -> bool:
    """Return True for crypto-like symbols that yfinance won't have split data for."""
    if "," in sym:
        return True
    for suffix in _CRYPTO_SUFFIXES:
        if sym.endswith(suffix) and len(sym) > len(suffix):
            return True
    return False


def _load_csv_normalized(csv_path: Path) -> pd.DataFrame:
    """Load a CSV, parse timestamp to UTC, normalize to midnight (daily granularity).

    Returns an empty DataFrame if the file lacks a timestamp column.
    """
    df = pd.read_csv(csv_path)
    if DATE_COL not in df.columns:
        return pd.DataFrame(columns=[DATE_COL, "close"])
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], utc=True).dt.normalize()
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    return df


def _price_ratio_at_split(df: pd.DataFrame, split_ts: pd.Timestamp) -> float | None:
    """Return (close_after_split / close_before_split), or None if data is missing."""
    idx_after = df.index[df[DATE_COL] >= split_ts].tolist()
    if not idx_after or idx_after[0] == 0:
        return None
    first_after = idx_after[0]
    pre_close = float(df.loc[first_after - 1, "close"])
    post_close = float(df.loc[first_after, "close"])
    if pre_close <= 0:
        return None
    return post_close / pre_close


def _scan_big_drops(df: pd.DataFrame) -> list[tuple[str, float]]:
    """Return list of (date_str, pct_change) for single-day drops > DROP_THRESHOLD.

    Uses the last close price per unique date to avoid false positives from
    duplicate rows (e.g. SPAC re-listings that produce two rows per day).
    """
    # Keep last row per normalized date
    daily = df.groupby(DATE_COL, as_index=False).last().sort_values(DATE_COL).reset_index(drop=True)
    closes = daily["close"].values
    dates = daily[DATE_COL].values
    drops = []
    for i in range(1, len(closes)):
        prev = float(closes[i - 1])
        if prev > 0:
            pct = (float(closes[i]) - prev) / prev
            if pct < -DROP_THRESHOLD:
                date_str = pd.Timestamp(dates[i]).date().isoformat()
                drops.append((date_str, pct))
    return drops


def _get_yfinance_splits(sym: str) -> pd.Series:
    """Fetch full split history from yfinance; return empty Series on any error."""
    try:
        splits = yf.Ticker(sym).get_splits(period="max")
        return splits if splits is not None and len(splits) > 0 else pd.Series(dtype=float)
    except Exception:
        return pd.Series(dtype=float)


def _backup_csv(csv_path: Path) -> None:
    """Copy CSV to .pre_split_backup unless the backup already exists."""
    backup = csv_path.with_suffix(".csv.pre_split_backup")
    if not backup.exists():
        shutil.copy2(csv_path, backup)


def _apply_split_to_csv(csv_path: Path, split_date: str, factor: float) -> None:
    """Divide pre-split prices by factor; multiply pre-split volume by factor.

    Always backs up the file first (idempotent — won't overwrite existing backup).
    Preserves the original timestamp string format by re-reading raw CSV.
    """
    _backup_csv(csv_path)

    df = pd.read_csv(csv_path)
    ts_parsed = pd.to_datetime(df[DATE_COL], utc=True).dt.normalize()
    split_ts = pd.Timestamp(split_date, tz="UTC")
    mask = ts_parsed < split_ts

    for col in PRICE_COLS:
        if col in df.columns:
            df.loc[mask, col] = df.loc[mask, col].astype(float) / factor
    if VOL_COL in df.columns:
        df.loc[mask, VOL_COL] = df.loc[mask, VOL_COL].astype(float) * factor

    df.to_csv(csv_path, index=False)


def _build_real_event_lookup() -> dict[tuple[str, str], str]:
    return {(sym, d): desc for sym, d, desc in KNOWN_REAL_EVENTS}


REAL_EVENT_LOOKUP = _build_real_event_lookup()


# ---------------------------------------------------------------------------
# Per-symbol audit logic
# ---------------------------------------------------------------------------


def audit_symbol(
    sym: str,
    daily_path: Path | None,
    hourly_path: Path | None,
    yf_splits: pd.Series,
    dry_run: bool,
) -> list[AuditRecord]:
    """Audit one symbol across daily + hourly CSVs.

    Returns list of AuditRecord entries (one per split event or notable drop).
    """
    records: list[AuditRecord] = []

    # --- Check known yfinance splits against daily file ---
    if daily_path is not None and daily_path.exists():
        df = _load_csv_normalized(daily_path)
        if "close" not in df.columns or df.empty:
            return records
        csv_start = df[DATE_COL].min()
        # Build set of known split dates (for filtering out from "unrecognized" scan)
        known_split_dates: set[str] = set()

        for split_dt_tz, factor in yf_splits.items():
            if factor < MIN_SPLIT_FACTOR:
                continue  # spin-off adjustments, reverse splits, or no-ops
            split_ts = pd.Timestamp(split_dt_tz).tz_convert("UTC").normalize()
            split_date_str = split_ts.date().isoformat()

            # Only audit splits whose date falls within our CSV's date range
            if split_ts < csv_start:
                continue

            known_split_dates.add(split_date_str)
            ratio = _price_ratio_at_split(df, split_ts)
            if ratio is None:
                continue

            expected = 1.0 / factor
            if abs(ratio - expected) <= SPLIT_RATIO_TOLERANCE:
                # Price dropped by ~1/factor: unadjusted split detected
                if dry_run:
                    records.append(AuditRecord(sym, split_date_str, factor, "FIXED(dry)"))
                else:
                    _apply_split_to_csv(daily_path, split_date_str, factor)
                    df = _load_csv_normalized(daily_path)  # reload after fix
                    records.append(AuditRecord(sym, split_date_str, factor, "FIXED"))
            else:
                # Price didn't drop by expected amount — already adjusted
                records.append(AuditRecord(sym, split_date_str, factor, "OK"))

        # --- Scan for unrecognized big drops (not explained by any known split) ---
        for drop_date, _pct in _scan_big_drops(df):
            if drop_date in known_split_dates:
                continue
            real_key = (sym, drop_date)
            if real_key in REAL_EVENT_LOOKUP:
                records.append(AuditRecord(sym, drop_date, 0.0, "REAL_EVENT"))
            else:
                records.append(AuditRecord(sym, drop_date, 0.0, "UNRECOGNIZED_DROP"))

    # --- Check hourly file for same splits ---
    if hourly_path is not None and hourly_path.exists() and len(yf_splits) > 0:
        df_h = _load_csv_normalized(hourly_path)
        fixed_hourly_dates: set[str] = set()

        for split_dt_tz, factor in yf_splits.items():
            if factor < MIN_SPLIT_FACTOR:
                continue  # spin-off adjustments, reverse splits, or no-ops
            split_ts = pd.Timestamp(split_dt_tz).tz_convert("UTC").normalize()
            split_date_str = split_ts.date().isoformat()

            # Look at bars in a ±1-day window around the split date
            window = df_h[
                (df_h[DATE_COL] >= split_ts - pd.Timedelta(days=1))
                & (df_h[DATE_COL] <= split_ts + pd.Timedelta(days=1))
            ].reset_index(drop=True)

            if len(window) < 2:
                continue

            closes = window["close"].values
            drops_in_window = [
                (float(closes[i]) - float(closes[i - 1])) / float(closes[i - 1])
                for i in range(1, len(closes))
                if float(closes[i - 1]) > 0
            ]
            if not drops_in_window:
                continue

            max_drop = min(drops_in_window)
            expected_drop = -1.0 + (1.0 / factor)  # e.g. -0.9 for 10:1 split

            if abs(max_drop - expected_drop) < SPLIT_RATIO_TOLERANCE + 0.05:
                # Hourly file also appears unadjusted
                fixed_hourly_dates.add(split_date_str)
                if dry_run:
                    records.append(AuditRecord(sym, split_date_str, factor, "FIXED_HOURLY(dry)"))
                else:
                    _apply_split_to_csv(hourly_path, split_date_str, factor)
                    records.append(AuditRecord(sym, split_date_str, factor, "FIXED_HOURLY"))

    return records


# ---------------------------------------------------------------------------
# Parallel yfinance fetch
# ---------------------------------------------------------------------------


def fetch_all_splits(symbols: list[str]) -> dict[str, pd.Series]:
    """Fetch split history for all symbols in parallel, skipping crypto."""
    stock_syms = [s for s in symbols if not _is_crypto_symbol(s)]
    result: dict[str, pd.Series] = {}

    print(f"Fetching split history from yfinance for {len(stock_syms)} stock symbols "
          f"({YFINANCE_WORKERS} workers)...", flush=True)

    done = 0
    total = len(stock_syms)

    with ThreadPoolExecutor(max_workers=YFINANCE_WORKERS) as executor:
        future_to_sym = {executor.submit(_get_yfinance_splits, sym): sym for sym in stock_syms}
        for future in as_completed(future_to_sym):
            sym = future_to_sym[future]
            splits = future.result()
            result[sym] = splits
            done += 1
            if done % 50 == 0 or done == total:
                print(f"  {done}/{total} symbols fetched", flush=True)

    return result


# ---------------------------------------------------------------------------
# Binary re-export
# ---------------------------------------------------------------------------


def reexport_binaries(fixed_symbols: set[str], repo_root: Path) -> None:
    """Re-export MKTD binaries whose symbol lists overlap with fixed_symbols."""
    for spec in BINARY_EXPORTS:
        sym_set = set(spec["symbols"].split(","))
        if not (sym_set & fixed_symbols):
            continue

        output_path = repo_root / spec["output"]
        if not output_path.parent.exists():
            print(f"  SKIP {spec['name']}: output directory {output_path.parent} does not exist")
            continue

        cmd = [
            sys.executable, "-m", "pufferlib_market.export_data_daily",
            "--symbols", spec["symbols"],
            "--start-date", spec["start"],
            "--end-date", spec["end"],
            "--output", str(output_path),
        ] + spec["extra_args"]

        print(f"  Exporting {spec['name']} ...")
        result = subprocess.run(cmd, cwd=str(repo_root), capture_output=False)
        if result.returncode != 0:
            print(f"  ERROR: export failed for {spec['name']} (exit {result.returncode})")
        else:
            print(f"  Done: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Audit and auto-fix unadjusted stock splits in training CSVs."
    )
    ap.add_argument("--dry-run", action="store_true", help="Show fixes without modifying files")
    ap.add_argument(
        "--verify",
        action="store_true",
        help="Print summary only, no modifications (implies --dry-run)",
    )
    ap.add_argument(
        "--no-reexport", action="store_true", help="Skip binary re-export after fixes"
    )
    args = ap.parse_args()

    if args.verify:
        args.dry_run = True

    repo_root = Path(__file__).parent
    daily_dir = repo_root / "trainingdata" / "train"
    hourly_dir = repo_root / "trainingdatahourly" / "stocks"

    if not daily_dir.exists():
        print(f"ERROR: daily CSV directory not found: {daily_dir}", file=sys.stderr)
        print("Run this script from the main repo directory.", file=sys.stderr)
        return 1

    # Collect symbols from daily CSVs (valid ticker names only)
    daily_csvs: dict[str, Path] = {
        p.stem: p for p in sorted(daily_dir.glob("*.csv")) if _is_valid_ticker(p.stem)
    }
    hourly_csvs: dict[str, Path] = (
        {p.stem: p for p in sorted(hourly_dir.glob("*.csv")) if _is_valid_ticker(p.stem)}
        if hourly_dir.exists()
        else {}
    )

    all_symbols = sorted(set(daily_csvs) | set(hourly_csvs))

    print(f"=== Split Audit: {date.today().isoformat()} ===")
    print(f"Daily CSVs:  {len(daily_csvs)} symbols in {daily_dir}")
    print(f"Hourly CSVs: {len(hourly_csvs)} symbols in {hourly_dir}")
    print(f"Total unique symbols: {len(all_symbols)}")
    if args.dry_run:
        print("Mode: DRY-RUN (no files will be modified)")
    print()

    # Fetch split history for all stock symbols in parallel
    splits_by_sym = fetch_all_splits(all_symbols)
    print()

    # --- Audit each symbol ---
    all_records: list[AuditRecord] = []
    fixed_symbols: set[str] = set()

    ok_lines: list[str] = []
    fixed_lines: list[str] = []
    real_lines: list[str] = []
    unrecognized_lines: list[str] = []

    for sym in all_symbols:
        yf_splits = splits_by_sym.get(sym, pd.Series(dtype=float))
        daily_path = daily_csvs.get(sym)
        hourly_path = hourly_csvs.get(sym)

        records = audit_symbol(sym, daily_path, hourly_path, yf_splits, dry_run=args.dry_run)
        all_records.extend(records)

        for r in records:
            if "FIXED" in r.status:
                fixed_symbols.add(sym)
                label = "DRY" if args.dry_run else "FIXED"
                fixed_lines.append(
                    f"  [{label}]   {sym}: {r.factor:.0f}:1 ({r.split_date}) [{r.status}]"
                )
            elif r.status == "OK":
                ok_lines.append(f"  [OK]    {sym}: {r.factor:.0f}:1 ({r.split_date})")
            elif r.status == "REAL_EVENT":
                desc = REAL_EVENT_LOOKUP.get((sym, r.split_date), "known event")
                real_lines.append(f"  [REAL]  {sym}: {r.split_date} ({desc})")
            elif r.status == "UNRECOGNIZED_DROP":
                unrecognized_lines.append(
                    f"  [WARN]  {sym}: {r.split_date} UNRECOGNIZED drop >30%"
                )

    # --- Print summary ---
    print(f"Symbols checked: {len(all_symbols)}")
    print()

    if ok_lines:
        print("Already-adjusted splits:")
        for line in ok_lines:
            print(line)
        print()

    if fixed_lines:
        action = "Would fix" if args.dry_run else "Fixed"
        print(f"{action}:")
        for line in fixed_lines:
            print(line)
        print()

    if real_lines:
        print("Known real events (classified as REAL_EVENT):")
        for line in real_lines:
            print(line)
        print()

    if unrecognized_lines:
        print("Unrecognized large drops (>30%) — review manually:")
        for line in unrecognized_lines:
            print(line)
        print()

    if not fixed_lines and not unrecognized_lines:
        print("No unadjusted splits found. All clean.")

    # --- Write report CSV ---
    report_path = repo_root / "splits_audit_report.csv"
    with open(report_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["symbol", "split_date", "factor", "status"])
        for r in all_records:
            writer.writerow([r.symbol, r.split_date, r.factor, r.status])
    print(f"\nReport written to: {report_path}  ({len(all_records)} entries)")

    # --- Re-export binaries if any symbols were fixed ---
    if not args.dry_run and not args.no_reexport and fixed_symbols:
        print(f"\nFixed symbols: {sorted(fixed_symbols)}")
        print("Re-exporting MKTD binaries for affected datasets...")
        reexport_binaries(fixed_symbols, repo_root)

    return 0


if __name__ == "__main__":
    sys.exit(main())
