"""Live trading runner for stockagent3 strategy."""

from __future__ import annotations

import argparse
from datetime import datetime
from typing import List, Mapping

import alpaca_wrapper
from loguru import logger

from stockagent.agentsimulator.market_data import fetch_latest_ohlc
from src.trading_obj_utils import filter_to_realistic_positions
from src.process_utils import (
    ramp_into_position,
    spawn_close_position_at_maxdiff_takeprofit,
    spawn_open_position_at_maxdiff_takeprofit,
)

from .agent import generate_trade_plan
from .expiry_watchers import (
    ExpiryWatcherConfig,
    compute_expiry_at,
    list_expiry_watchers,
    spawn_expiry_watcher,
    stop_expiry_watchers,
)


STRATEGY_NAME = "stockagent3"
MIN_CONFIDENCE = 0.3


def _coerce_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _get_portfolio_value() -> float:
    try:
        status = alpaca_wrapper.get_account_status(force_refresh=True)
        equity = status.get("equity")
        if equity is not None:
            return float(equity)
    except Exception as exc:
        logger.warning("Failed to refresh account status: %s", exc)
    return float(getattr(alpaca_wrapper, "equity", 0.0) or 0.0)


def _current_positions_snapshot() -> List[Mapping[str, object]]:
    positions = filter_to_realistic_positions(alpaca_wrapper.get_all_positions())
    snapshot: List[Mapping[str, object]] = []
    for pos in positions:
        symbol = str(getattr(pos, "symbol", "")).upper()
        side = str(getattr(pos, "side", ""))
        qty = _coerce_float(getattr(pos, "qty", None) or getattr(pos, "quantity", None))
        avg_price = _coerce_float(getattr(pos, "avg_entry_price", None))
        snapshot.append(
            {
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "avg_price": avg_price,
            }
        )
    return snapshot


def _cleanup_stale_expiry_watchers(active_symbols: set[str]) -> None:
    for metadata in list_expiry_watchers(strategy=STRATEGY_NAME):
        symbol = str(metadata.get("symbol") or "").upper()
        side = str(metadata.get("side") or "").lower()
        if symbol and symbol not in active_symbols:
            stop_expiry_watchers(symbol=symbol, side=side, strategy=STRATEGY_NAME, reason="symbol_removed")


def _execute_plan(symbols: List[str], plan) -> None:
    if plan.overall_confidence < MIN_CONFIDENCE:
        logger.warning("Plan confidence %.2f below threshold %.2f; skipping", plan.overall_confidence, MIN_CONFIDENCE)
        return
    if not plan.positions:
        logger.warning("No positions to execute")
        return

    portfolio_value = _get_portfolio_value()
    active_symbols = {pos.symbol for pos in plan.positions}
    _cleanup_stale_expiry_watchers(active_symbols)

    for position in plan.positions:
        symbol = position.symbol
        side = "buy" if position.direction == "long" else "sell"
        entry_price = position.entry_price
        exit_price = position.exit_price
        if entry_price <= 0 or exit_price <= 0:
            logger.warning("Skipping %s due to invalid prices", symbol)
            continue

        stop_expiry_watchers(symbol=symbol, side=side, strategy=STRATEGY_NAME, reason="replaced")

        notional = portfolio_value * position.target_alloc * position.leverage
        target_qty = notional / entry_price if entry_price > 0 else 0.0
        if target_qty <= 0:
            logger.warning("Skipping %s due to zero target qty", symbol)
            continue

        if position.entry_mode in {"market", "ramp"}:
            ramp_into_position(symbol, side=side, target_qty=target_qty)
        else:
            spawn_open_position_at_maxdiff_takeprofit(
                symbol,
                side=side,
                limit_price=entry_price,
                target_qty=target_qty,
                entry_strategy=STRATEGY_NAME,
                expiry_minutes=int(position.entry_expiry_days * 1440),
            )

        spawn_close_position_at_maxdiff_takeprofit(
            symbol,
            side=side,
            takeprofit_price=exit_price,
            entry_strategy=STRATEGY_NAME,
            expiry_minutes=int(position.hold_expiry_days * 1440),
            target_qty=target_qty,
        )

        expiry_at = compute_expiry_at(symbol, position.hold_expiry_days)
        spawn_expiry_watcher(
            ExpiryWatcherConfig(
                symbol=symbol,
                side=side,
                expiry_at=expiry_at,
                strategy=STRATEGY_NAME,
                reason="plan_expiry",
            )
        )

        logger.info(
            "%s %s alloc=%.2f qty=%.4f entry=%.4f exit=%.4f expiry=%s",
            symbol,
            position.direction,
            position.target_alloc,
            target_qty,
            entry_price,
            exit_price,
            expiry_at.isoformat(),
        )


def run(symbols: List[str], lookback_days: int, max_lines: int, use_thinking: bool) -> None:
    portfolio_value = _get_portfolio_value()
    if portfolio_value <= 0:
        logger.error("Invalid portfolio value; aborting")
        return

    data_bundle = fetch_latest_ohlc(symbols=symbols, lookback_days=lookback_days)
    market_data = data_bundle.bars

    plan = generate_trade_plan(
        symbols=symbols,
        market_data=market_data,
        portfolio_value=portfolio_value,
        current_positions=_current_positions_snapshot(),
        max_lines=max_lines,
        use_thinking=use_thinking,
    )

    _execute_plan(symbols, plan)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stockagent3 live trading plan")
    parser.add_argument("--symbols", required=True, help="Comma-separated symbol list")
    parser.add_argument("--lookback-days", type=int, default=400)
    parser.add_argument("--max-lines", type=int, default=160)
    parser.add_argument("--use-thinking", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    symbols = [sym.strip().upper() for sym in args.symbols.split(",") if sym.strip()]
    if not symbols:
        logger.error("No symbols provided")
        return

    logger.info("Running stockagent3 trade plan (%s)", datetime.utcnow().isoformat())
    run(symbols, args.lookback_days, args.max_lines, args.use_thinking)


if __name__ == "__main__":
    main()
