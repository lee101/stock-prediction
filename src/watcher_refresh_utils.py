"""Utilities for refreshing maxdiff watchers while maintaining plan consistency."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

from src.logging_utils import setup_logging, get_log_filename

# Detect if we're in hourly mode based on TRADE_STATE_SUFFIX env var
_is_hourly = os.getenv("TRADE_STATE_SUFFIX", "") == "hourly"
logger = setup_logging(get_log_filename("watcher_refresh_utils.log", is_hourly=_is_hourly))


def should_use_existing_watcher_prices(
    watcher_metadata: Dict,
    is_crypto: bool,
    max_age_hours: float = 24.0,
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
    mode: str,
    is_crypto: bool,
    max_age_hours: float = 24.0,
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

    # Import here to avoid circular dependency
    from src.process_utils import _load_watcher_metadata

    # Search for existing watchers matching symbol/side/mode
    pattern = f"{symbol}_{side}_{mode}_*.json"

    for watcher_file in watcher_dir.glob(pattern):
        metadata = _load_watcher_metadata(watcher_file)
        if not metadata:
            continue

        should_use, reason = should_use_existing_watcher_prices(
            metadata,
            is_crypto,
            max_age_hours,
        )

        if should_use:
            # Extract the appropriate price field based on mode
            if mode == "entry":
                price = metadata.get("limit_price")
            elif mode == "exit":
                price = metadata.get("takeprofit_price")
            else:
                logger.warning(f"Unknown watcher mode: {mode}")
                price = None

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
    mode: str,
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
