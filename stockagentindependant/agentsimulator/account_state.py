"""Helpers for condensing live account data."""

from __future__ import annotations

from datetime import datetime, timezone

from loguru import logger

import alpaca_wrapper

from .data_models import AccountPosition, AccountSnapshot


def get_account_snapshot() -> AccountSnapshot:
    try:
        account = alpaca_wrapper.get_account()
    except Exception as exc:
        logger.error(f"Failed to fetch Alpaca account: {exc}")
        raise

    try:
        raw_positions = alpaca_wrapper.get_all_positions()
    except Exception as exc:
        logger.error(f"Failed to fetch positions: {exc}")
        raw_positions = []

    positions = []
    for position in raw_positions:
        try:
            positions.append(AccountPosition.from_alpaca(position))
        except Exception as exc:
            logger.warning(f"Skipping malformed position {position}: {exc}")

    snapshot = AccountSnapshot(
        equity=float(getattr(account, "equity", 0.0)),
        cash=float(getattr(account, "cash", 0.0)),
        buying_power=float(getattr(account, "buying_power", 0.0)) if getattr(account, "buying_power", None) is not None else None,
        timestamp=datetime.now(timezone.utc),
        positions=positions,
    )
    return snapshot
