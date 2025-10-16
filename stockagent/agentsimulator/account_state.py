"""Helpers to gather a condensed view of the live account."""

from __future__ import annotations

from datetime import datetime, timezone

from loguru import logger

import alpaca_wrapper

from .data_models import AccountPosition, AccountSnapshot


def _collect_positions() -> list[AccountPosition]:
    try:
        raw_positions = alpaca_wrapper.get_all_positions()
    except Exception as exc:
        logger.error(f"Failed to fetch positions: {exc}")
        return []

    positions: list[AccountPosition] = []
    for position in raw_positions:
        try:
            positions.append(AccountPosition.from_alpaca(position))
        except Exception as exc:
            logger.warning(f"Skipping malformed position {position}: {exc}")
    return positions


def get_account_snapshot() -> AccountSnapshot:
    try:
        account = alpaca_wrapper.get_account()
    except Exception as exc:
        logger.error(f"Failed to fetch Alpaca account: {exc}")
        raise

    snapshot = AccountSnapshot(
        equity=float(getattr(account, "equity", 0.0)),
        cash=float(getattr(account, "cash", 0.0)),
        buying_power=float(getattr(account, "buying_power", 0.0)) if getattr(account, "buying_power", None) is not None else None,
        timestamp=datetime.now(timezone.utc),
        positions=_collect_positions(),
    )
    return snapshot
