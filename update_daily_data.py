#!/usr/bin/env python3
"""
Refresh daily training data and keep the canonical `trainingdata/train` CSVs in sync.

The high-level workflow is:
1. List every symbol that already has a training CSV.
2. Run the Alpaca downloader (which writes timestamped snapshots under data/train).
3. For each symbol, locate the freshest snapshot, normalize the columns, and append any new
   rows into `trainingdata/train/<SYMBOL>.csv`, preserving historical data.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from loguru import logger

from data_curate_daily import download_daily_stock_data
from src.symbol_utils import is_crypto_symbol

TRAINING_DIR = Path("trainingdata/train")
SNAPSHOT_DIR = Path("data/train")
BASE_COLUMN_ORDER: Tuple[str, ...] = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trade_count",
    "vwap",
    "symbol",
)
ZERO_DEFAULT_COLUMNS = {"volume", "trade_count"}


def _storage_symbol(symbol: str) -> str:
    """Canonical symbol name we use for filenames and within CSVs."""
    return symbol.replace("/", "-").upper()


def _stem_to_symbol(stem: str) -> str:
    """Infer the logical symbol name from a training CSV filename stem.

    Keep symbols as-is without slash conversion since Alpaca stock API
    doesn't support slash format for crypto.
    """
    return stem.upper()

def _canonical_cli_symbol(symbol: str) -> str:
    """Normalize CLI symbols to the canonical storage form (BTC/USD -> BTCUSD)."""
    return symbol.replace("/", "").replace("-", "").upper()


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Lowercase + underscore normalize column names for internal processing."""
    renamed = {
        col: col.strip().lower().replace(" ", "_")
        for col in frame.columns
    }
    return frame.rename(columns=renamed)


def _prepare_training_frame(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Ensure snapshots follow the canonical schema expected by downstream loaders."""
    work = _normalize_columns(df.copy())
    if "timestamp" not in work.columns:
        raise ValueError("Snapshot is missing a timestamp column.")
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work = work.dropna(subset=["timestamp"])
    work.sort_values("timestamp", inplace=True)
    storage_symbol = _storage_symbol(symbol)
    work["symbol"] = storage_symbol

    for column in ("open", "high", "low", "close"):
        if column not in work.columns:
            raise ValueError(f"Snapshot for {symbol} lacks required column '{column}'")
        work[column] = pd.to_numeric(work[column], errors="coerce")

    # Ensure optional numeric columns exist so feature engineering stays stable.
    if "vwap" not in work.columns:
        work["vwap"] = work["close"]
    else:
        work["vwap"] = pd.to_numeric(work["vwap"], errors="coerce").fillna(work["close"])

    for column in ZERO_DEFAULT_COLUMNS:
        if column not in work.columns:
            work[column] = 0.0
        else:
            work[column] = pd.to_numeric(work[column], errors="coerce").fillna(0.0)

    # Keep any other columns (e.g., dividends) for future processing.
    ordered_columns = list(BASE_COLUMN_ORDER)
    for column in work.columns:
        if column not in ordered_columns:
            ordered_columns.append(column)

    work = work[ordered_columns]
    work.reset_index(drop=True, inplace=True)
    return work


def _merge_training_frames(existing: pd.DataFrame, updates: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Merge freshly downloaded rows into the historical frame."""
    if existing.empty:
        combined = updates.copy()
    else:
        combined = pd.concat([existing, updates], ignore_index=True, sort=False)
    combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
    combined.sort_values("timestamp", inplace=True)
    combined.reset_index(drop=True, inplace=True)

    before = existing["timestamp"].nunique() if not existing.empty else 0
    after = combined["timestamp"].nunique()
    new_rows = max(0, after - before)
    return combined, new_rows


def _find_latest_snapshot(symbol: str, snapshot_dir: Path) -> Optional[Path]:
    """Return the newest timestamped download for a symbol."""
    safe = _storage_symbol(symbol)
    pattern = f"{safe}-*.csv"
    matches = sorted(snapshot_dir.glob(pattern))
    if not matches:
        return None
    return matches[-1]


def _load_existing_training(symbol: str, training_dir: Path) -> pd.DataFrame:
    """Load and normalize the persisted training CSV, if it exists."""
    safe = _storage_symbol(symbol)
    path = training_dir / f"{safe}.csv"
    if not path.exists():
        return pd.DataFrame(columns=BASE_COLUMN_ORDER)
    frame = pd.read_csv(path)
    return _prepare_training_frame(frame, symbol)


def _sync_symbol(symbol: str, snapshot_dir: Path, training_dir: Path) -> int:
    """Append the newest snapshot rows to the canonical training CSV."""
    latest_snapshot = _find_latest_snapshot(symbol, snapshot_dir)
    if latest_snapshot is None:
        logger.warning("No downloaded snapshot found for {}", symbol)
        return 0

    snapshot_df = pd.read_csv(latest_snapshot)
    prepared_snapshot = _prepare_training_frame(snapshot_df, symbol)
    existing = _load_existing_training(symbol, training_dir)
    merged, new_rows = _merge_training_frames(existing, prepared_snapshot)
    if new_rows == 0:
        logger.debug("{}: no new rows to append", symbol)
        return 0

    target_path = training_dir / f"{_storage_symbol(symbol)}.csv"
    merged.to_csv(target_path, index=False)
    logger.info("{}: appended {} new rows (total={})", symbol, new_rows, merged.shape[0])
    return new_rows


_NON_SYMBOL_STEMS = frozenset({
    "correlation_matrix",
    "data_summary",
    "volatility_metrics",
    "download_metadata",
    "combined_training_data",
})


def _is_valid_symbol_stem(stem: str) -> bool:
    """Check if a stem looks like a valid stock/crypto symbol."""
    lower = stem.lower()
    if lower in _NON_SYMBOL_STEMS:
        return False
    # Symbols are uppercase letters possibly followed by USD for crypto
    upper = stem.upper()
    if upper.endswith("USD"):
        base = upper[:-3]
        return base.isalpha() and len(base) >= 1
    return upper.isalpha() and 1 <= len(upper) <= 5


def list_symbols(training_dir: Path = TRAINING_DIR) -> List[str]:
    """Enumerate symbols that already have training CSVs."""
    if not training_dir.exists():
        raise FileNotFoundError(f"Training directory {training_dir} is missing.")
    symbols = []
    for csv_file in training_dir.glob("*.csv"):
        stem = csv_file.stem
        if not _is_valid_symbol_stem(stem):
            logger.debug("Skipping non-symbol file: {}", csv_file.name)
            continue
        symbol = _stem_to_symbol(stem)
        symbols.append(symbol)
    unique = sorted(set(symbols))
    if not unique:
        raise RuntimeError(f"No symbols discovered under {training_dir}")
    return unique


def download_and_sync(symbols: Sequence[str], *, strict: bool = False) -> Dict[str, int]:
    """Download the latest bars and sync them into trainingdata/train."""
    logger.info("Updating {} symbols (strict={})", len(symbols), strict)
    download_daily_stock_data(path="train", all_data_force=True, symbols=list(symbols), strict=strict)
    snapshot_dir = SNAPSHOT_DIR
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    training_dir = TRAINING_DIR
    training_dir.mkdir(parents=True, exist_ok=True)

    appended: Dict[str, int] = {}
    for symbol in symbols:
        try:
            appended[symbol] = _sync_symbol(symbol, snapshot_dir, training_dir)
        except Exception as exc:
            logger.exception("Failed to sync {}: {}", symbol, exc)
            appended[symbol] = 0
    total_new = sum(appended.values())
    logger.info("Daily data sync complete ({} new rows).", total_new)
    return appended


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh daily training data and sync `trainingdata/train`.",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Symbols to refresh (default: discover from existing trainingdata/train/*.csv).",
    )
    parser.add_argument("--only-stocks", action="store_true", help="Only refresh stock symbols.")
    parser.add_argument("--only-crypto", action="store_true", help="Only refresh crypto symbols.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail the download step if a symbol cannot be downloaded and has no cached dataset.",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()

    if args.only_stocks and args.only_crypto:
        logger.error("Cannot specify both --only-stocks and --only-crypto.")
        return 2

    try:
        if args.symbols:
            symbols = [_canonical_cli_symbol(s) for s in args.symbols]
        else:
            symbols = list_symbols()
    except Exception as exc:
        logger.error("Unable to enumerate symbols: {}", exc)
        return 1

    if args.only_stocks:
        symbols = [s for s in symbols if not is_crypto_symbol(s)]
    elif args.only_crypto:
        symbols = [s for s in symbols if is_crypto_symbol(s)]

    download_and_sync(symbols, strict=bool(args.strict))
    return 0


if __name__ == "__main__":
    sys.exit(main())
