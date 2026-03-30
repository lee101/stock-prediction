"""Shared local OHLC cache helpers for stockagent variants."""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import pandas as pd
from loguru import logger

from .market_data_bundle import MarketDataBundle

_MARKET_SYMBOL_RE = re.compile(r"^[A-Z0-9]+(?:[._-][A-Z0-9]+)*$")
_MAX_SYMBOL_LENGTH = 24
_SUPPORTED_LOCAL_DATA_SUFFIXES = frozenset({".parquet", ".pq", ".csv", ".json"})
USE_DEFAULT = object()
LOCAL_DATA_DIR_ENV = "STOCKAGENT_LOCAL_DATA_DIR"
USE_FALLBACK_DATA_DIRS_ENV = "STOCKAGENT_USE_FALLBACK_DATA_DIRS"
_DEFAULT_LOCAL_DATA_DIR = Path("trainingdata")
RemoteMarketDataLoader = Callable[[str], pd.DataFrame]


def default_local_data_dir() -> Path | None:
    raw = os.getenv(LOCAL_DATA_DIR_ENV)
    if raw is None:
        return _DEFAULT_LOCAL_DATA_DIR
    cleaned = raw.strip()
    if not cleaned:
        return None
    return Path(cleaned)


def default_use_fallback_data_dirs() -> bool:
    raw = os.getenv(USE_FALLBACK_DATA_DIRS_ENV)
    if raw is None or not raw.strip():
        return True
    normalised = raw.strip().lower()
    if normalised in {"1", "true", "yes", "on"}:
        return True
    if normalised in {"0", "false", "no", "off"}:
        return False
    logger.warning(
        "Invalid {}={!r}; falling back to enabled fallback data dirs.",
        USE_FALLBACK_DATA_DIRS_ENV,
        raw,
    )
    return True


def normalize_market_symbol(raw: str) -> str:
    symbol = str(raw or "").strip().upper().replace("/", "-")
    if not symbol:
        raise ValueError("At least one symbol is required.")
    if len(symbol) > _MAX_SYMBOL_LENGTH:
        raise ValueError(f"Unsupported symbol {raw!r}: symbol is longer than {_MAX_SYMBOL_LENGTH} characters.")
    if not _MARKET_SYMBOL_RE.fullmatch(symbol):
        raise ValueError(
            f"Unsupported symbol {raw!r}: only alphanumerics plus '.', '_' or '-' separators are allowed."
        )
    return symbol


def normalize_market_symbols(symbols: Iterable[str]) -> List[str]:
    return [normalize_market_symbol(str(symbol)) for symbol in symbols]


def resolve_local_data_dirs(
    *,
    local_data_dir: Optional[Path] | object = USE_DEFAULT,
    fallback_data_dirs: Sequence[Path],
    use_fallback_data_dirs: bool | object = USE_DEFAULT,
) -> List[Path]:
    if local_data_dir is USE_DEFAULT:
        local_data_dir = default_local_data_dir()
    if use_fallback_data_dirs is USE_DEFAULT:
        use_fallback_data_dirs = default_use_fallback_data_dirs()

    candidate_dirs: List[Path] = []
    if local_data_dir is not None:
        candidate_dirs.append(Path(local_data_dir))
    if use_fallback_data_dirs:
        candidate_dirs.extend(Path(path) for path in fallback_data_dirs)

    unique_dirs: List[Path] = []
    for path in candidate_dirs:
        if path not in unique_dirs:
            unique_dirs.append(path)
    return unique_dirs


def build_market_data_bundle(
    *,
    symbols: Optional[Iterable[str]],
    default_symbols: Sequence[str],
    lookback_days: int,
    as_of: datetime,
    local_data_dir: Optional[Path] | object = USE_DEFAULT,
    fallback_data_dirs: Sequence[Path],
    use_fallback_data_dirs: bool | object = USE_DEFAULT,
    remote_loader: RemoteMarketDataLoader | None = None,
) -> MarketDataBundle:
    requested_symbols = list(symbols) if symbols is not None else list(default_symbols)
    normalized_symbols = normalize_market_symbols(requested_symbols or default_symbols)

    candidate_dirs = resolve_local_data_dirs(
        local_data_dir=local_data_dir,
        fallback_data_dirs=fallback_data_dirs,
        use_fallback_data_dirs=use_fallback_data_dirs,
    )
    existing_dirs = [path for path in candidate_dirs if path.exists()]
    for missing in [path for path in candidate_dirs if not path.exists()]:
        logger.debug(f"Local market data dir {missing} not found.")
    if not existing_dirs:
        logger.warning("No local market data directories available; continuing without cached OHLC data.")

    local_files = find_latest_local_symbol_files(symbols=normalized_symbols, directories=existing_dirs)
    bars: Dict[str, pd.DataFrame] = {}
    for symbol in normalized_symbols:
        local_file = local_files.get(symbol)
        df = load_local_data_file(symbol=symbol, path=local_file) if local_file is not None else pd.DataFrame()
        if df.empty and remote_loader is not None:
            df = remote_loader(symbol)
        bars[symbol] = ensure_datetime_index(df).tail(lookback_days)

    return MarketDataBundle(bars=bars, lookback_days=lookback_days, as_of=as_of)


def find_latest_local_symbol_files(
    *,
    symbols: Sequence[str],
    directories: Sequence[Path],
) -> Dict[str, Path]:
    """Resolve the newest matching local data file per symbol.

    Directory precedence is preserved: once a symbol is found in an earlier
    directory, later directories are ignored for that symbol.
    """

    pending = list(dict.fromkeys(normalize_market_symbols(symbols)))
    unresolved = set(pending)
    latest_files: Dict[str, Path] = {}
    for directory in directories:
        if not unresolved:
            break
        indexed = _index_latest_local_symbol_files(
            directory=directory,
            requested_symbols=tuple(symbol for symbol in pending if symbol in unresolved),
        )
        for symbol in pending:
            if symbol in unresolved and symbol in indexed:
                latest_files[symbol] = indexed[symbol]
                unresolved.remove(symbol)
    return latest_files


def load_local_symbol_data(*, symbol: str, directory: Path) -> pd.DataFrame:
    normalized_symbol = normalize_market_symbol(symbol)
    indexed = _index_latest_local_symbol_files(
        directory=directory,
        requested_symbols=(normalized_symbol,),
    )
    latest = indexed.get(normalized_symbol)
    if latest is None:
        return pd.DataFrame()
    return load_local_data_file(symbol=symbol, path=latest)


def load_local_data_file(*, symbol: str, path: Path) -> pd.DataFrame:
    try:
        suffix = path.suffix.lower()
        if suffix in {".parquet", ".pq"}:
            df = pd.read_parquet(path)
        elif suffix == ".json":
            df = pd.read_json(path)
        else:
            df = pd.read_csv(path)
    except Exception as exc:
        logger.warning(f"Failed to load {symbol} data from {path}: {exc}")
        return pd.DataFrame()
    df.columns = [col.lower() for col in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.rename(columns={"time": "timestamp", "date": "timestamp", "datetime": "timestamp"})
    return df


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    elif isinstance(df.index, pd.DatetimeIndex):
        indexed = df.copy()
        indexed.index = pd.to_datetime(indexed.index, utc=True, errors="coerce")
        indexed = indexed.loc[~indexed.index.isna()].sort_index()
        return indexed
    if "timestamp" not in df.columns:
        logger.warning("Received OHLC frame without timestamp column; skipping dataset")
        return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    return df


def _index_latest_local_symbol_files(
    *,
    directory: Path,
    requested_symbols: Sequence[str],
) -> Dict[str, Path]:
    requested = tuple(dict.fromkeys(normalize_market_symbol(symbol) for symbol in requested_symbols))
    if not requested:
        return {}
    requested_set = set(requested)

    latest_by_symbol: Dict[str, tuple[int, int, Path]] = {}
    try:
        entries = list(Path(directory).iterdir())
    except OSError as exc:
        logger.warning(f"Failed to scan local market data dir {directory}: {exc}")
        return {}

    for path in entries:
        if not path.is_file() or path.suffix.lower() not in _SUPPORTED_LOCAL_DATA_SUFFIXES:
            continue
        matched_symbols = _matched_local_symbols_for_stem(path.stem.upper(), requested_set)
        if not matched_symbols:
            continue
        try:
            stat_result = path.stat()
        except OSError as exc:
            logger.debug(f"Failed to stat market data file {path}: {exc}")
            continue
        mtime_ns = getattr(stat_result, "st_mtime_ns", int(stat_result.st_mtime * 1_000_000_000))
        ctime_ns = getattr(stat_result, "st_ctime_ns", int(stat_result.st_ctime * 1_000_000_000))
        for symbol in matched_symbols:
            previous = latest_by_symbol.get(symbol)
            candidate = (mtime_ns, ctime_ns, path)
            if previous is None or candidate >= previous:
                latest_by_symbol[symbol] = candidate

    return {symbol: path for symbol, (_mtime_ns, _ctime_ns, path) in latest_by_symbol.items()}


def _matched_local_symbols_for_stem(stem_upper: str, requested_symbols: set[str]) -> tuple[str, ...]:
    matches: list[str] = []
    if stem_upper in requested_symbols:
        matches.append(stem_upper)
    for idx, char in enumerate(stem_upper):
        if char.isalnum():
            continue
        candidate = stem_upper[:idx]
        if candidate and candidate in requested_symbols:
            matches.append(candidate)
    return tuple(dict.fromkeys(matches))
