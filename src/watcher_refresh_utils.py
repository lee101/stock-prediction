"""Utilities for refreshing maxdiff watchers while maintaining plan consistency."""

from __future__ import annotations

import os
from collections import OrderedDict
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from threading import Event, Lock
from typing import Dict, Optional, Tuple

from src.logging_utils import setup_logging, get_log_filename

# Detect if we're in hourly mode based on TRADE_STATE_SUFFIX env var
_is_hourly = os.getenv("TRADE_STATE_SUFFIX", "") == "hourly"
logger = setup_logging(get_log_filename("watcher_refresh_utils.log", is_hourly=_is_hourly))
DEFAULT_CRYPTO_WATCHER_REUSE_MAX_AGE_HOURS = 24.0
_WATCHER_FILE_INDEX_CACHE_MAX_ENTRIES = 128
_WATCHER_METADATA_CACHE_MAX_ENTRIES = 512
_WatcherLookupKey = tuple[str, str, "WatcherMode"]
_WATCHER_FILE_INDEX_CACHE: OrderedDict[
    tuple[Path, int],
    dict[_WatcherLookupKey, tuple[Path, ...]],
] = OrderedDict()
_WATCHER_FILE_INDEX_CACHE_LOCK = Lock()
_WATCHER_FILE_INDEX_CACHE_INFLIGHT: dict[tuple[Path, int], Event] = {}
_WATCHER_METADATA_CACHE: OrderedDict[
    tuple[Path, int, int],
    Optional[Dict],
] = OrderedDict()
_WATCHER_METADATA_CACHE_LOCK = Lock()
_WATCHER_METADATA_CACHE_INFLIGHT: dict[tuple[Path, int, int], Event] = {}


class WatcherMode(StrEnum):
    ENTRY = "entry"
    EXIT = "exit"


WATCHER_MODE_PRICE_FIELDS: dict[WatcherMode, str] = {
    WatcherMode.ENTRY: "limit_price",
    WatcherMode.EXIT: "takeprofit_price",
}


def _coerce_watcher_mode(mode: str | WatcherMode) -> WatcherMode | None:
    if isinstance(mode, WatcherMode):
        return mode
    if isinstance(mode, str):
        normalized = mode.strip().lower()
        if not normalized:
            return None
        try:
            return WatcherMode(normalized)
        except ValueError:
            return None
    return None


def _watcher_price_field(mode: str | WatcherMode) -> str | None:
    resolved_mode = _coerce_watcher_mode(mode)
    if resolved_mode is None:
        return None
    return WATCHER_MODE_PRICE_FIELDS.get(resolved_mode)


def _watcher_lookup_key(
    symbol: str,
    side: str,
    mode: WatcherMode,
) -> _WatcherLookupKey:
    return (str(symbol).upper(), str(side).lower(), mode)


def _watcher_directory_listing_version(watcher_dir: Path) -> int | None:
    try:
        return watcher_dir.resolve(strict=False).stat().st_mtime_ns
    except OSError:
        return None


def _prune_watcher_file_index_cache() -> None:
    while len(_WATCHER_FILE_INDEX_CACHE) > _WATCHER_FILE_INDEX_CACHE_MAX_ENTRIES:
        _WATCHER_FILE_INDEX_CACHE.popitem(last=False)


def _build_watcher_file_index(watcher_dir: Path) -> dict[_WatcherLookupKey, tuple[Path, ...]]:
    index: dict[_WatcherLookupKey, list[Path]] = {}
    for watcher_file in sorted(watcher_dir.iterdir(), key=lambda path: path.name):
        if watcher_file.suffix != ".json":
            continue
        stem_parts = watcher_file.stem.split("_", 3)
        if len(stem_parts) < 4:
            continue
        resolved_mode = _coerce_watcher_mode(stem_parts[2])
        if resolved_mode is None:
            continue
        lookup_key = _watcher_lookup_key(stem_parts[0], stem_parts[1], resolved_mode)
        index.setdefault(lookup_key, []).append(watcher_file)
    return {key: tuple(paths) for key, paths in index.items()}


def _watcher_metadata_version(watcher_file: Path) -> tuple[Path, int, int] | None:
    try:
        resolved_file = watcher_file.resolve(strict=False)
        stat_result = resolved_file.stat()
    except OSError:
        return None
    return (resolved_file, stat_result.st_size, stat_result.st_mtime_ns)


def _prune_watcher_metadata_cache() -> None:
    while len(_WATCHER_METADATA_CACHE) > _WATCHER_METADATA_CACHE_MAX_ENTRIES:
        _WATCHER_METADATA_CACHE.popitem(last=False)


def _load_cached_watcher_metadata(watcher_file: Path) -> Optional[Dict]:
    version = _watcher_metadata_version(watcher_file)
    if version is None:
        return None

    should_load = False
    with _WATCHER_METADATA_CACHE_LOCK:
        cached = _WATCHER_METADATA_CACHE.get(version)
        if cached is not None or version in _WATCHER_METADATA_CACHE:
            _WATCHER_METADATA_CACHE.move_to_end(version)
            return None if cached is None else dict(cached)
        inflight = _WATCHER_METADATA_CACHE_INFLIGHT.get(version)
        if inflight is None:
            inflight = Event()
            _WATCHER_METADATA_CACHE_INFLIGHT[version] = inflight
            should_load = True

    if not should_load:
        inflight.wait()
        with _WATCHER_METADATA_CACHE_LOCK:
            cached = _WATCHER_METADATA_CACHE.get(version)
            if cached is not None or version in _WATCHER_METADATA_CACHE:
                _WATCHER_METADATA_CACHE.move_to_end(version)
                return None if cached is None else dict(cached)
        return None

    # Import here to avoid circular dependency
    from src.process_utils import load_watcher_metadata

    metadata: Optional[Dict] = None
    try:
        metadata = load_watcher_metadata(watcher_file)
        with _WATCHER_METADATA_CACHE_LOCK:
            for existing_key in list(_WATCHER_METADATA_CACHE):
                if existing_key[0] == version[0] and existing_key != version:
                    del _WATCHER_METADATA_CACHE[existing_key]
            _WATCHER_METADATA_CACHE[version] = None if metadata is None else dict(metadata)
            _WATCHER_METADATA_CACHE.move_to_end(version)
            _prune_watcher_metadata_cache()
    finally:
        with _WATCHER_METADATA_CACHE_LOCK:
            inflight = _WATCHER_METADATA_CACHE_INFLIGHT.pop(version, None)
            if inflight is not None:
                inflight.set()

    return None if metadata is None else dict(metadata)


def _watcher_file_candidates(
    watcher_dir: Path,
    symbol: str,
    side: str,
    mode: WatcherMode,
) -> tuple[Path, ...]:
    resolved_dir = watcher_dir.resolve(strict=False)
    listing_version = _watcher_directory_listing_version(resolved_dir)
    if listing_version is None:
        return ()
    cache_key = (resolved_dir, listing_version)
    should_build = False
    with _WATCHER_FILE_INDEX_CACHE_LOCK:
        cached = _WATCHER_FILE_INDEX_CACHE.get(cache_key)
        if cached is not None:
            _WATCHER_FILE_INDEX_CACHE.move_to_end(cache_key)
            return cached.get(_watcher_lookup_key(symbol, side, mode), ())
        inflight = _WATCHER_FILE_INDEX_CACHE_INFLIGHT.get(cache_key)
        if inflight is None:
            inflight = Event()
            _WATCHER_FILE_INDEX_CACHE_INFLIGHT[cache_key] = inflight
            should_build = True

    if not should_build:
        inflight.wait()
        with _WATCHER_FILE_INDEX_CACHE_LOCK:
            cached = _WATCHER_FILE_INDEX_CACHE.get(cache_key)
            if cached is not None:
                _WATCHER_FILE_INDEX_CACHE.move_to_end(cache_key)
                return cached.get(_watcher_lookup_key(symbol, side, mode), ())
        return ()

    try:
        cached = _build_watcher_file_index(resolved_dir)
        with _WATCHER_FILE_INDEX_CACHE_LOCK:
            for existing_key in list(_WATCHER_FILE_INDEX_CACHE):
                if existing_key[0] == resolved_dir and existing_key != cache_key:
                    del _WATCHER_FILE_INDEX_CACHE[existing_key]
            _WATCHER_FILE_INDEX_CACHE[cache_key] = cached
            _WATCHER_FILE_INDEX_CACHE.move_to_end(cache_key)
            _prune_watcher_file_index_cache()
    finally:
        with _WATCHER_FILE_INDEX_CACHE_LOCK:
            inflight = _WATCHER_FILE_INDEX_CACHE_INFLIGHT.pop(cache_key, None)
            if inflight is not None:
                inflight.set()

    return cached.get(_watcher_lookup_key(symbol, side, mode), ())


def should_use_existing_watcher_prices(
    watcher_metadata: Dict,
    is_crypto: bool,
    max_age_hours: float = DEFAULT_CRYPTO_WATCHER_REUSE_MAX_AGE_HOURS,
) -> Tuple[bool, Optional[str]]:
    """
    Determine if we should use existing watcher prices or refresh with new forecast.

    For crypto within 24hrs, we want to stick to the original plan to avoid overtrading.
    For stocks or expired watchers, use new forecast prices.

    Args:
        watcher_metadata: Dictionary from watcher config JSON
        is_crypto: Whether this is a crypto asset
        max_age_hours: Maximum age in hours to reuse existing prices (default 24hrs)

    Returns:
        Tuple of (should_use_existing: bool, reason: str)
        - should_use_existing: True if should keep existing prices
        - reason: Human-readable explanation of the decision
    """
    if not watcher_metadata:
        return False, "no_metadata"

    started_at_str = watcher_metadata.get("started_at")
    expiry_at_str = watcher_metadata.get("expiry_at")

    if not started_at_str or not expiry_at_str:
        return False, "missing_timestamps"

    try:
        started_at = datetime.fromisoformat(started_at_str.replace("Z", "+00:00"))
        expiry_at = datetime.fromisoformat(expiry_at_str.replace("Z", "+00:00"))
    except (ValueError, TypeError) as exc:
        logger.debug(f"Failed to parse watcher timestamps: {exc}")
        return False, "invalid_timestamps"

    now = datetime.now(timezone.utc)
    age_hours = (now - started_at).total_seconds() / 3600
    is_expired = now >= expiry_at

    if is_expired:
        return False, f"expired_{age_hours:.1f}hrs_old"

    if not is_crypto:
        # For stocks, always use new forecast (market conditions change)
        return False, f"stock_market_conditions_changed_{age_hours:.1f}hrs_old"

    if age_hours >= max_age_hours:
        # Even for crypto, refresh after max_age_hours
        return False, f"age_exceeded_{age_hours:.1f}hrs_old"

    # Crypto within max_age_hours and not expired - stick to the plan
    return True, f"within_{age_hours:.1f}hrs_keeping_original_plan"


def find_existing_watcher_price(
    watcher_dir: Path,
    symbol: str,
    side: str,
    mode: str | WatcherMode,
    is_crypto: bool,
    max_age_hours: float = DEFAULT_CRYPTO_WATCHER_REUSE_MAX_AGE_HOURS,
) -> Tuple[Optional[float], Optional[str]]:
    """
    Find existing watcher and determine if its price should be reused.

    Args:
        watcher_dir: Directory containing watcher config files
        symbol: Trading symbol
        side: Position side ("buy" or "sell")
        mode: Watcher mode ("entry" or "exit")
        is_crypto: Whether this is a crypto asset
        max_age_hours: Maximum age to reuse prices (default 24hrs)

    Returns:
        Tuple of (price: Optional[float], reason: Optional[str])
        - price: Existing price if should be reused, None otherwise
        - reason: Human-readable decision reason
    """
    if not watcher_dir.exists():
        return None, "watcher_dir_not_found"

    # Search for existing watchers matching symbol/side/mode
    resolved_mode = _coerce_watcher_mode(mode)
    if resolved_mode is None:
        logger.warning(f"Unknown watcher mode: {mode}")
        return None, "unknown_watcher_mode"
    price_field = WATCHER_MODE_PRICE_FIELDS[resolved_mode]
    for watcher_file in _watcher_file_candidates(watcher_dir, symbol, side, resolved_mode):
        metadata = _load_cached_watcher_metadata(watcher_file)
        if not metadata:
            continue

        should_use, reason = should_use_existing_watcher_prices(
            metadata,
            is_crypto,
            max_age_hours,
        )

        if should_use:
            price = metadata.get(price_field)
            if price is not None:
                logger.info(
                    f"{symbol} {side} {mode}: Using existing watcher price={price:.4f} ({reason})"
                )
                return price, reason
        else:
            logger.debug(
                f"{symbol} {side} {mode}: Not using existing watcher ({reason})"
            )

    return None, "no_suitable_watcher_found"


def should_spawn_watcher(
    existing_price: Optional[float],
    new_price: Optional[float],
    mode: str | WatcherMode,
) -> Tuple[bool, Optional[float], str]:
    """
    Decide whether to spawn a watcher and which price to use.

    Args:
        existing_price: Price from existing watcher (if reusable)
        new_price: Price from new forecast
        mode: Watcher mode for logging

    Returns:
        Tuple of (should_spawn: bool, price_to_use: Optional[float], reason: str)
    """
    if existing_price is not None:
        # Have valid existing watcher - don't spawn, use existing
        return False, existing_price, "existing_watcher_valid"

    if new_price is None or new_price <= 0:
        # No valid price to use
        return False, None, "invalid_new_price"

    # Need to spawn with new price
    return True, new_price, "spawning_with_new_forecast"
