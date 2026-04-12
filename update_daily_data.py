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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from loguru import logger

from data_curate_daily import download_daily_stock_data
from src.alpaca_stock_expansion import default_stock_expansion_candidates, get_sp500_symbols
from src.symbol_file_utils import (
    SymbolFileOrdering,
    load_symbols_from_file as load_symbols_from_text_file,
)
from src.symbol_utils import is_crypto_symbol
from src.trade_directions import DEFAULT_ALPACA_LIVE8_STOCKS

TRAINING_DIR = Path("trainingdata/train")
SNAPSHOT_DIR = Path("data/train")
IPO_SYMBOLS_FILE = Path(__file__).resolve().parent / "symbol_lists" / "ipo_candidates_2025_2026.txt"
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
SYMBOL_SET_ALIASES = {
    "stock-expansion": lambda: [candidate.symbol for candidate in default_stock_expansion_candidates()],
    "alpaca-live8": lambda: list(DEFAULT_ALPACA_LIVE8_STOCKS),
    "sp500": lambda: list(get_sp500_symbols(use_cache=True)),
    "ipo-2025-2026": lambda: load_symbols_file(IPO_SYMBOLS_FILE),
}


def _storage_symbol(symbol: str) -> str:
    """Canonical symbol name we use for filenames and within CSVs."""
    return symbol.replace("/", "-").replace(".", "-").upper()


def _stem_to_symbol(stem: str) -> str:
    """Infer the logical symbol name from a training CSV filename stem.

    Keep symbols as-is without slash conversion since Alpaca stock API
    doesn't support slash format for crypto.
    """
    return stem.upper()

def _canonical_cli_symbol(symbol: str) -> str:
    """Normalize CLI symbols while preserving equity share-class markers."""
    normalized = str(symbol or "").strip().upper()
    if not normalized:
        return ""
    if normalized.endswith("-USD"):
        return normalized.replace("-", "")
    if "/" in normalized:
        return normalized.replace("/", "")
    return normalized.replace(".", "-")


def _market_data_symbol(symbol: str) -> str:
    """Translate canonical symbols to the format expected by Alpaca market data."""
    normalized = str(symbol or "").strip().upper()
    if not normalized:
        return ""
    if "/" in normalized or is_crypto_symbol(normalized):
        return normalized
    if "." in normalized:
        return normalized
    if "-" in normalized:
        base, suffix = normalized.rsplit("-", 1)
        if base and len(suffix) == 1 and suffix.isalnum():
            return f"{base}.{suffix}"
    return normalized


def _split_symbol_tokens(raw: str) -> List[str]:
    return [
        _canonical_cli_symbol(str(token).strip())
        for token in str(raw or "").replace("\n", ",").split(",")
        if str(token).strip()
    ]


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Lowercase + underscore normalize column names for internal processing."""
    normalized = frame.copy()
    normalized.columns = [
        str(col).strip().lower().replace(" ", "_")
        for col in normalized.columns
    ]
    if not normalized.columns.has_duplicates:
        return normalized

    deduped = pd.DataFrame(index=normalized.index)
    seen: set[str] = set()
    for column in normalized.columns:
        if column in seen:
            continue
        seen.add(column)
        subset = normalized.loc[:, normalized.columns == column]
        if isinstance(subset, pd.Series):
            deduped[column] = subset
            continue
        # Legacy training files can contain both Open/open style columns. Prefer the
        # first non-null value across duplicate normalized columns.
        collapsed = subset.iloc[:, 0]
        for idx in range(1, subset.shape[1]):
            next_values = subset.iloc[:, idx]
            if next_values.isna().all():
                continue
            if collapsed.isna().all():
                collapsed = next_values
                continue
            collapsed = collapsed.combine_first(next_values)
        deduped[column] = collapsed
    return deduped


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
    stems = {
        _storage_symbol(symbol),
        _market_data_symbol(symbol).replace("/", "-"),
    }
    matches = sorted(
        path
        for stem in stems
        for path in snapshot_dir.glob(f"{stem}-*.csv")
    )
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


def _flatten_download_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame.columns, pd.MultiIndex):
        return frame
    flattened: list[str] = []
    for column in frame.columns:
        if isinstance(column, tuple):
            flattened.append(str(column[0]).strip())
        else:
            flattened.append(str(column).strip())
    result = frame.copy()
    result.columns = flattened
    return result


def _download_stock_with_yfinance(symbol: str) -> pd.DataFrame:
    import yfinance as yf  # noqa: PLC0415

    end = datetime.now(timezone.utc).date() + timedelta(days=1)
    start = end - timedelta(days=365 * 5)
    yf_symbol = str(symbol).strip().upper().replace(".", "-")
    frame = yf.download(
        yf_symbol,
        start=start.isoformat(),
        end=end.isoformat(),
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
    )
    if frame is None or frame.empty:
        raise RuntimeError(f"No yfinance history returned for {symbol}")
    frame = _flatten_download_columns(frame)
    frame = frame.reset_index()
    if "Date" in frame.columns and "timestamp" not in frame.columns:
        frame = frame.rename(columns={"Date": "timestamp"})
    return _prepare_training_frame(frame, symbol)


def _merge_direct_training_frame(
    symbol: str,
    updates: pd.DataFrame,
    *,
    training_dir: Path,
) -> int:
    existing = _load_existing_training(symbol, training_dir)
    merged, new_rows = _merge_training_frames(existing, updates)
    target_path = Path(training_dir) / f"{_storage_symbol(symbol)}.csv"
    if new_rows > 0 or not target_path.exists():
        target_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(target_path, index=False)
    return new_rows


def _fallback_sync_with_yfinance(
    symbols: Sequence[str],
    *,
    training_dir: Path = TRAINING_DIR,
) -> Dict[str, int]:
    appended: Dict[str, int] = {}
    stock_symbols = [symbol for symbol in symbols if not is_crypto_symbol(symbol)]
    if not stock_symbols:
        return appended

    logger.warning(
        "Primary daily refresh failed; attempting yfinance fallback for %d stock symbols.",
        len(stock_symbols),
    )
    for symbol in stock_symbols:
        try:
            updates = _download_stock_with_yfinance(symbol)
            appended[symbol] = _merge_direct_training_frame(
                symbol,
                updates,
                training_dir=training_dir,
            )
            logger.info(
                "{}: yfinance fallback merged {} rows",
                symbol,
                appended[symbol],
            )
        except Exception as exc:
            logger.warning("yfinance fallback failed for {}: {}", symbol, exc)
            appended[symbol] = 0
    return appended


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


def resolve_symbol_set(name: str) -> List[str]:
    normalized = str(name or "").strip().lower()
    if not normalized:
        raise ValueError("Symbol set name is required.")
    loader = SYMBOL_SET_ALIASES.get(normalized)
    if loader is None:
        supported = ", ".join(sorted(SYMBOL_SET_ALIASES))
        raise ValueError(f"Unknown symbol set {name!r}. Supported values: {supported}")
    return sorted(set(loader()))


def load_symbols_file(path: Path) -> List[str]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Symbols file does not exist: {file_path}")
    unique = load_symbols_from_text_file(
        file_path,
        normalize_symbol=_canonical_cli_symbol,
        dedupe=True,
        ordering=SymbolFileOrdering.SORTED,
    )
    if not unique:
        raise RuntimeError(f"No symbols found in {file_path}")
    return unique


def resolve_requested_symbols(
    *,
    cli_symbols: Optional[Sequence[str]] = None,
    symbol_set: Optional[str] = None,
    symbols_file: Optional[Path] = None,
    training_dir: Path = TRAINING_DIR,
) -> List[str]:
    resolved: List[str] = []
    if cli_symbols:
        resolved.extend(_canonical_cli_symbol(symbol) for symbol in cli_symbols)
    if symbol_set:
        resolved.extend(resolve_symbol_set(symbol_set))
    if symbols_file is not None:
        resolved.extend(load_symbols_file(symbols_file))
    if not resolved:
        return list_symbols(training_dir)
    return sorted(set(resolved))


def build_sync_report(
    symbols: Sequence[str],
    appended: Dict[str, int],
    *,
    training_dir: Path = TRAINING_DIR,
    as_of: Optional[pd.Timestamp] = None,
) -> List[Dict[str, Any]]:
    report_time = as_of
    if report_time is None:
        report_time = pd.Timestamp(datetime.now(timezone.utc))

    rows: List[Dict[str, Any]] = []
    for symbol in sorted(set(str(symbol).strip().upper() for symbol in symbols if str(symbol).strip())):
        path = Path(training_dir) / f"{_storage_symbol(symbol)}.csv"
        if not path.exists():
            rows.append(
                {
                    "symbol": symbol,
                    "path": str(path),
                    "exists": False,
                    "appended_rows": int(appended.get(symbol, 0) or 0),
                    "row_count": 0,
                    "first_timestamp": "",
                    "last_timestamp": "",
                    "stale_days": None,
                }
            )
            continue

        frame = pd.read_csv(path, usecols=["timestamp"])
        timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dropna()
        if timestamps.empty:
            rows.append(
                {
                    "symbol": symbol,
                    "path": str(path),
                    "exists": True,
                    "appended_rows": int(appended.get(symbol, 0) or 0),
                    "row_count": 0,
                    "first_timestamp": "",
                    "last_timestamp": "",
                    "stale_days": None,
                }
            )
            continue

        first_timestamp = timestamps.min()
        last_timestamp = timestamps.max()
        stale_days = int((report_time.floor("D") - last_timestamp.floor("D")).days)
        rows.append(
            {
                "symbol": symbol,
                "path": str(path),
                "exists": True,
                "appended_rows": int(appended.get(symbol, 0) or 0),
                "row_count": int(timestamps.shape[0]),
                "first_timestamp": first_timestamp.isoformat(),
                "last_timestamp": last_timestamp.isoformat(),
                "stale_days": stale_days,
            }
        )
    return rows


def write_sync_report(path: Path, rows: Sequence[Dict[str, Any]]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(list(rows))
    if output_path.suffix.lower() == ".json":
        output_path.write_text(frame.to_json(orient="records", indent=2) + "\n")
    else:
        frame.to_csv(output_path, index=False)
    return output_path


def download_and_sync(symbols: Sequence[str], *, strict: bool = False) -> Dict[str, int]:
    """Download the latest bars and sync them into trainingdata/train."""
    logger.info("Updating {} symbols (strict={})", len(symbols), strict)
    download_symbols = [_market_data_symbol(symbol) for symbol in symbols]
    snapshot_dir = SNAPSHOT_DIR
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    training_dir = TRAINING_DIR
    training_dir.mkdir(parents=True, exist_ok=True)

    fallback_appended: Dict[str, int] = {}
    try:
        download_daily_stock_data(
            path="train",
            all_data_force=True,
            symbols=download_symbols,
            strict=strict,
        )
    except Exception as exc:
        logger.warning("Primary daily refresh failed: {}", exc)
        fallback_appended = _fallback_sync_with_yfinance(
            symbols,
            training_dir=training_dir,
        )

    appended: Dict[str, int] = {}
    for symbol in symbols:
        if symbol in fallback_appended:
            appended[symbol] = int(fallback_appended[symbol])
            continue
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
    parser.add_argument(
        "--symbol-set",
        choices=sorted(SYMBOL_SET_ALIASES),
        help="Named symbol universe to refresh in addition to any explicit --symbols.",
    )
    parser.add_argument(
        "--symbols-file",
        type=Path,
        help="Optional newline/comma separated symbol file to refresh.",
    )
    parser.add_argument("--only-stocks", action="store_true", help="Only refresh stock symbols.")
    parser.add_argument("--only-crypto", action="store_true", help="Only refresh crypto symbols.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail the download step if a symbol cannot be downloaded and has no cached dataset.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        help="Optional CSV/JSON file to write a post-sync freshness report.",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()

    if args.only_stocks and args.only_crypto:
        logger.error("Cannot specify both --only-stocks and --only-crypto.")
        return 2

    try:
        symbols = resolve_requested_symbols(
            cli_symbols=args.symbols,
            symbol_set=args.symbol_set,
            symbols_file=args.symbols_file,
        )
    except Exception as exc:
        logger.error("Unable to enumerate symbols: {}", exc)
        return 1

    if args.only_stocks:
        symbols = [s for s in symbols if not is_crypto_symbol(s)]
    elif args.only_crypto:
        symbols = [s for s in symbols if is_crypto_symbol(s)]

    appended = download_and_sync(symbols, strict=bool(args.strict))
    report_rows = build_sync_report(symbols, appended)
    if args.report_path is not None:
        report_path = write_sync_report(args.report_path, report_rows)
        logger.info("Wrote sync report to {}", report_path)

    stale_rows = [row for row in report_rows if row.get("exists") and isinstance(row.get("stale_days"), int)]
    if stale_rows:
        stalest = sorted(stale_rows, key=lambda row: int(row["stale_days"]), reverse=True)[:10]
        for row in stalest:
            logger.info(
                "Freshness {}: last={} stale_days={} appended_rows={}",
                row["symbol"],
                row["last_timestamp"],
                row["stale_days"],
                row["appended_rows"],
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
