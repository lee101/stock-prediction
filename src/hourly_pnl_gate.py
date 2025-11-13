"""PnL-based trading gate for hourly trading bot.

This module provides functionality to block trading on symbol+side pairs
where recent trades have been unprofitable. This helps prevent repeatedly
trading losing strategies.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from jsonshelve import FlatShelf
from src.trade_stock_state_utils import normalize_side_for_key, state_key

StoreLoader = Callable[[], Optional[FlatShelf]]
LoggerLike = Optional[logging.Logger]


def get_recent_trade_pnl(
    store_loader: StoreLoader,
    symbol: str,
    side: str,
    *,
    strategy: Optional[str] = None,
    max_trades: int = 2,
    logger: LoggerLike = None,
) -> Tuple[List[Dict[str, Any]], float]:
    """Get recent trade history and sum of PnL for a symbol+side.

    Args:
        store_loader: Function that returns the trade history store
        symbol: Trading symbol (e.g., "BTCUSD", "AAPL")
        side: Trade side ("buy" or "sell")
        strategy: Optional strategy name for strategy-specific filtering
        max_trades: Maximum number of recent trades to consider (default 2)
        logger: Optional logger for warnings

    Returns:
        Tuple of (list of recent trade records, sum of PnL)
        If no trades found, returns ([], 0.0)
    """
    store = store_loader()
    if store is None:
        return [], 0.0

    # Only load if the store appears uninitialized
    # Check if store has a 'data' attribute and if it's None
    try:
        if not hasattr(store, 'data') or store.data is None:
            store.load()
    except Exception as exc:
        if logger is not None:
            logger.error(f"Failed loading trade history store for PnL check: {exc}")
        return [], 0.0

    normalized_side = normalize_side_for_key(side)
    key = state_key(symbol, normalized_side, strategy)
    history = store.get(key, [])

    if not history:
        return [], 0.0

    # Get the most recent trades
    recent_trades = history[-max_trades:]
    total_pnl = sum(float(trade.get("pnl", 0.0)) for trade in recent_trades)

    return recent_trades, total_pnl


def should_block_trade_by_pnl(
    store_loader: StoreLoader,
    symbol: str,
    side: str,
    *,
    strategy: Optional[str] = None,
    max_trades: int = 2,
    logger: LoggerLike = None,
) -> Tuple[bool, Optional[str]]:
    """Check if trading should be blocked based on recent negative PnL.

    Blocks trading if the sum of the last 1-2 trades (depending on availability)
    for the same symbol+side+strategy is negative.

    Args:
        store_loader: Function that returns the trade history store
        symbol: Trading symbol (e.g., "BTCUSD", "AAPL")
        side: Trade side ("buy" or "sell")
        strategy: Optional strategy name for strategy-specific gating
        max_trades: Maximum number of recent trades to consider (default 2)
        logger: Optional logger for warnings

    Returns:
        Tuple of (should_block: bool, reason: Optional[str])
        - If blocked: (True, reason_string)
        - If not blocked: (False, None)
    """
    recent_trades, total_pnl = get_recent_trade_pnl(
        store_loader,
        symbol,
        side,
        strategy=strategy,
        max_trades=max_trades,
        logger=logger,
    )

    if not recent_trades:
        # No trade history, allow trading (probe trade scenario)
        return False, None

    if total_pnl < 0:
        trade_count = len(recent_trades)
        strategy_suffix = f" ({strategy})" if strategy else ""
        reason = (
            f"Last {trade_count} trade{'s' if trade_count > 1 else ''} "
            f"for {symbol} {side}{strategy_suffix} had negative PnL: {total_pnl:.2f}"
        )
        return True, reason

    return False, None


def get_pnl_blocking_report(
    store_loader: StoreLoader,
    symbols: List[str],
    *,
    strategies: Optional[List[str]] = None,
    max_trades: int = 2,
    logger: LoggerLike = None,
) -> Dict[str, Any]:
    """Generate a report of which symbol+side pairs would be blocked by PnL gate.

    Args:
        store_loader: Function that returns the trade history store
        symbols: List of trading symbols to check
        strategies: Optional list of strategies to check (if None, checks without strategy)
        max_trades: Maximum number of recent trades to consider
        logger: Optional logger

    Returns:
        Dictionary with blocked and allowed symbols, including reasons
    """
    blocked = {}
    allowed = {}

    sides = ["buy", "sell"]
    strategies_to_check = strategies if strategies is not None else [None]

    for symbol in symbols:
        for side in sides:
            for strategy in strategies_to_check:
                key = f"{symbol}_{side}"
                if strategy:
                    key = f"{key}_{strategy}"

                should_block, reason = should_block_trade_by_pnl(
                    store_loader,
                    symbol,
                    side,
                    strategy=strategy,
                    max_trades=max_trades,
                    logger=logger,
                )

                if should_block:
                    blocked[key] = {
                        "symbol": symbol,
                        "side": side,
                        "strategy": strategy,
                        "reason": reason,
                    }
                else:
                    allowed[key] = {
                        "symbol": symbol,
                        "side": side,
                        "strategy": strategy,
                    }

    return {
        "blocked_count": len(blocked),
        "allowed_count": len(allowed),
        "blocked": blocked,
        "allowed": allowed,
    }


__all__ = [
    "get_recent_trade_pnl",
    "should_block_trade_by_pnl",
    "get_pnl_blocking_report",
]
