#!/usr/bin/env python3
"""Production live trading script for Two-Stage Portfolio Optimization.

This script runs the backtested two-stage strategy in production:
1. Stage 1: Portfolio allocation across symbols (Claude)
2. Stage 2: Precise entry/exit/stop prices per symbol (Claude)

Then executes trades via maxdiff/ramp_into_position style order management.

Usage:
    # Paper trading (dry run)
    PAPER=1 python twostage_trade_live.py --symbols BTCUSD ETHUSD LINKUSD

    # Live trading
    python twostage_trade_live.py --symbols BTCUSD ETHUSD LINKUSD --interval-seconds 300
"""
from __future__ import annotations

import argparse
import asyncio
import signal
from datetime import datetime, timezone, time as dtime, timedelta
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence
from zoneinfo import ZoneInfo

import anthropic
import alpaca_wrapper
from loguru import logger

from stockagent.agentsimulator.market_data import fetch_latest_ohlc
from stockagent_pctline.data_formatter import format_pctline_data, PctLineData
from stockagent_twostage.portfolio_allocator import (
    async_allocate_portfolio,
    PortfolioAllocation,
    is_crypto,
)
from stockagent_twostage.price_predictor import async_predict_prices, PricePrediction
from src.logging_utils import setup_logging, get_log_filename
from src.process_utils import (
    spawn_open_position_at_maxdiff_takeprofit,
    spawn_close_position_at_maxdiff_takeprofit,
    stop_all_entry_watchers,
)
from src.fixtures import active_crypto_symbols
from src.symbol_utils import is_crypto_symbol


DEFAULT_INTERVAL_SECONDS = 300  # 5 minutes
MIN_STOCK_QTY = 1.0
MIN_CRYPTO_QTY = 0.001
MIN_ORDER_NOTIONAL = 1.0
ENTRY_STRATEGY = "twostage"

# Trading windows (Eastern time)
WINDOW_OPEN = (dtime(9, 30), dtime(10, 30))    # US equities open
WINDOW_CLOSE = (dtime(15, 30), dtime(16, 0))   # US equities close
WINDOW_CRYPTO = (dtime(0, 0), dtime(1, 0))     # Crypto daily bar (UTC)


def _load_account_equity() -> float:
    """Get account equity from Alpaca."""
    try:
        account = alpaca_wrapper.get_account()
    except Exception as exc:
        logger.error(f"Unable to load Alpaca account: {exc}")
        return 1.0
    try:
        equity = float(account.equity or account.cash or 0.0)
    except Exception:
        equity = 0.0
    if equity <= 0:
        logger.warning("Account equity unavailable; defaulting to 1.0 for sizing.")
        return 1.0
    return equity


def _resolve_symbols(args: argparse.Namespace) -> Sequence[str]:
    """Resolve symbols from args, filtering inactive crypto."""
    if args.symbols:
        raw = tuple(symbol.upper() for symbol in args.symbols)
    else:
        # Default symbols - mix of crypto and high-vol stocks
        raw = ("BTCUSD", "ETHUSD", "LINKUSD", "SPY", "QQQ")

    active_crypto = {sym.upper() for sym in active_crypto_symbols}
    filtered = []
    for sym in raw:
        up = sym.upper()
        if is_crypto_symbol(up) and up not in active_crypto:
            logger.warning(f"Skipping inactive crypto: {up}")
            continue
        filtered.append(up)
    return tuple(filtered)


@dataclass
class TwoStagePlan:
    """Trading plan from two-stage analysis."""
    symbol: str
    direction: str  # "long" or "short"
    allocation: float  # 0-1
    leverage: float  # 1.0-2.0
    entry_price: float
    exit_price: float
    stop_loss_price: float
    confidence: float
    rationale: str


class TwoStageTradingLoop:
    """Main trading loop for two-stage portfolio optimization."""

    def __init__(
        self,
        symbols: Sequence[str],
        *,
        interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
        min_confidence: float = 0.3,
        force_immediate: bool = False,
        skip_equity_weekends: bool = True,
        log_windows: bool = False,
        max_lines: int = 300,
        use_chronos: bool = True,
        use_thinking: bool = False,
    ) -> None:
        self.symbols = [symbol.upper() for symbol in symbols]
        self.interval_seconds = max(30, int(interval_seconds))
        self.min_confidence = min_confidence
        self.force_immediate = bool(force_immediate)
        self.skip_equity_weekends = bool(skip_equity_weekends)
        self.log_windows = bool(log_windows)
        self.max_lines = max_lines
        self.use_chronos = use_chronos
        self.use_thinking = use_thinking
        self._stop_requested = False
        self._bootstrapped = False
        self._last_run_date: Optional[datetime.date] = None

    def run(self, *, once: bool = False) -> None:
        """Main loop."""
        while not self._stop_requested:
            try:
                asyncio.run(self._tick())
            except Exception as exc:
                logger.exception(f"Two-stage trading tick failed: {exc}")
            if once:
                break
            time.sleep(self.interval_seconds)

    async def _tick(self) -> None:
        """Execute one trading cycle."""
        window = self._current_window()
        has_crypto = any(is_crypto_symbol(sym) for sym in self.symbols)

        if window is None:
            if has_crypto:
                window = "crypto_anytime"
            elif not self._bootstrapped:
                window = "bootstrap"
            else:
                (logger.info if self.log_windows else logger.debug)(
                    "Outside trading windows; skipping tick."
                )
                return

        logger.info(f"Two-stage tick starting (window={window})")

        # Get account equity
        account_equity = _load_account_equity()
        logger.info(f"Account equity: ${account_equity:,.2f}")

        # Generate trading plans
        plans = await self._generate_plans()
        if not plans:
            logger.info("No eligible two-stage plans for current tick.")
            return

        logger.info(f"Generated {len(plans)} trading plans")
        for plan in plans:
            self._dispatch_plan(plan, account_equity)

        self._bootstrapped = True
        self._close_orphan_positions()

    def _current_window(self) -> Optional[str]:
        """Check if we're in a trading window."""
        now_utc = datetime.now(timezone.utc)
        now_et = now_utc.astimezone(ZoneInfo("US/Eastern"))

        def _in_window(window: tuple[dtime, dtime], clock: dtime) -> bool:
            return window[0] <= clock < window[1]

        if _in_window(WINDOW_OPEN, now_et.time()):
            return "open"
        if _in_window(WINDOW_CLOSE, now_et.time()):
            return "close"
        if _in_window(WINDOW_CRYPTO, now_utc.time()):
            return "crypto"
        return None

    async def _generate_plans(self) -> list[TwoStagePlan]:
        """Run two-stage portfolio optimization."""
        today = datetime.now(timezone.utc).date()

        # Fetch market data
        as_of = datetime.now(timezone.utc)
        try:
            bundle = fetch_latest_ohlc(
                symbols=self.symbols,
                lookback_days=1000,
                as_of=as_of,
            )
            all_data = bundle.bars
        except Exception as e:
            logger.error(f"Failed to fetch market data: {e}")
            return []

        # Log loaded data
        for sym, df in all_data.items():
            logger.info(f"Loaded {len(df)} bars for {sym}")

        # Build pct-line data
        pct_data: dict[str, PctLineData] = {}
        for symbol, df in all_data.items():
            if not hasattr(df.index, 'date') or len(df) == 0:
                continue
            if len(df) > 10:
                pct_data[symbol] = format_pctline_data(df, symbol, self.max_lines)

        if not pct_data:
            logger.warning("No valid pct-line data generated")
            return []

        # Generate Chronos forecasts (optional)
        chronos_forecasts = None
        if self.use_chronos:
            chronos_forecasts = self._generate_chronos_forecasts(all_data, today)

        # Create Anthropic client for Claude API calls
        client = anthropic.AsyncAnthropic()

        # Stage 1: Portfolio allocation
        try:
            allocations = await async_allocate_portfolio(
                client,
                pct_data,
                chronos_forecasts=chronos_forecasts,
                use_thinking=self.use_thinking,
            )
        except Exception as e:
            logger.error(f"Stage 1 allocation failed: {e}")
            return []

        logger.info(
            f"Stage 1: overall confidence {allocations.overall_confidence:.2f}, "
            f"{len(allocations.allocations)} allocations"
        )

        if not allocations.should_trade(self.min_confidence):
            logger.info(
                f"Skipping: confidence {allocations.overall_confidence:.2f} < {self.min_confidence}"
            )
            return []

        # Stage 2: Price predictions
        try:
            predictions = await async_predict_prices(
                client,
                pct_data,
                allocations.allocations,
                chronos_forecasts=chronos_forecasts,
            )
        except Exception as e:
            logger.error(f"Stage 2 price prediction failed: {e}")
            return []

        logger.info(f"Stage 2: {len(predictions)} price predictions")

        # Build trading plans
        plans = []
        for symbol, alloc in allocations.allocations.items():
            if alloc.alloc <= 0.001:
                continue
            if symbol not in predictions:
                continue

            pred = predictions[symbol]
            plan = TwoStagePlan(
                symbol=symbol,
                direction=alloc.direction,
                allocation=alloc.alloc,
                leverage=alloc.leverage,
                entry_price=pred.entry_price,
                exit_price=pred.exit_price,
                stop_loss_price=pred.stop_loss_price,
                confidence=pred.confidence,
                rationale=pred.rationale,
            )
            plans.append(plan)
            logger.info(
                f"  {symbol} {alloc.direction}: entry ${pred.entry_price:.2f}, "
                f"exit ${pred.exit_price:.2f}, stop ${pred.stop_loss_price:.2f}, "
                f"conf {pred.confidence:.2f}"
            )

        # Sort by confidence * allocation
        plans.sort(key=lambda p: p.confidence * p.allocation, reverse=True)
        return plans

    def _generate_chronos_forecasts(
        self, all_data: dict, trade_date
    ) -> dict | None:
        """Generate Chronos2 forecasts if available."""
        try:
            from stockagentopus_chronos2.forecaster import generate_chronos2_forecasts
            from stockagent.agentsimulator.market_data import MarketDataBundle

            filtered_bars = {}
            for symbol, df in all_data.items():
                if not hasattr(df.index, 'date') or len(df) == 0:
                    continue
                if len(df) > 10:
                    filtered_bars[symbol] = df

            if not filtered_bars:
                return None

            bundle = MarketDataBundle(
                bars=filtered_bars,
                lookback_days=512,
                as_of=datetime.now(timezone.utc),
            )

            forecasts = generate_chronos2_forecasts(
                market_data=bundle,
                symbols=list(filtered_bars.keys()),
                prediction_length=1,
                context_length=512,
                device_map="cuda",
            )
            logger.info(f"Generated Chronos2 forecasts for {len(forecasts)} symbols")
            return forecasts
        except ImportError:
            logger.debug("Chronos2 not available")
            return None
        except Exception as e:
            logger.warning(f"Chronos2 forecast failed: {e}")
            return None

    def _dispatch_plan(self, plan: TwoStagePlan, account_equity: float) -> None:
        """Execute a trading plan via order management."""
        symbol = plan.symbol
        asset_is_crypto = is_crypto(symbol)

        # Skip opening equity positions on weekends
        if self.skip_equity_weekends and not asset_is_crypto:
            if datetime.now(timezone.utc).weekday() >= 5:
                logger.info(f"Weekend skip for {symbol} (equity)")
                return

        # Skip equities outside market hours
        if not asset_is_crypto:
            try:
                if not alpaca_wrapper.alpaca_api.get_clock().is_open:
                    logger.info(f"Market closed; skipping {symbol}")
                    return
            except Exception as exc:
                logger.warning(f"Clock lookup failed: {exc}")

        # Calculate position size
        leverage = plan.leverage if not asset_is_crypto else 1.0
        leverage = min(2.0, max(1.0, leverage))

        base_notional = account_equity * plan.allocation
        leveraged_notional = base_notional * leverage

        min_qty = MIN_CRYPTO_QTY if asset_is_crypto else MIN_STOCK_QTY
        qty = leveraged_notional / plan.entry_price if plan.entry_price > 0 else 0

        # Enforce minimum notional
        min_notional_qty = MIN_ORDER_NOTIONAL / max(plan.entry_price, 1e-9)
        qty = max(qty, min_qty, min_notional_qty)

        if qty <= 0:
            logger.debug(f"Skipping {symbol}: zero qty")
            return

        # Clear existing watchers
        stop_all_entry_watchers(symbol, reason="twostage_reset")

        logger.info(
            f"Dispatching {symbol} {plan.direction} | "
            f"entry ${plan.entry_price:.4f} exit ${plan.exit_price:.4f} "
            f"stop ${plan.stop_loss_price:.4f} qty={qty:.4f} "
            f"alloc={plan.allocation:.1%} leverage={leverage:.1f}x"
        )

        # Spawn entry watchers
        if plan.direction == "long":
            self._spawn_entry(symbol, "buy", plan.entry_price, qty, asset_is_crypto)
        else:
            self._spawn_entry(symbol, "sell", plan.entry_price, qty, asset_is_crypto)

    def _spawn_entry(
        self, symbol: str, side: str, limit_price: float, qty: float, is_crypto: bool
    ) -> None:
        """Spawn an entry watcher process."""
        tolerance_pct = 0.001 if is_crypto else None
        force_immediate = True if is_crypto else self.force_immediate

        spawn_open_position_at_maxdiff_takeprofit(
            symbol=symbol,
            side=side,
            limit_price=float(limit_price),
            target_qty=float(qty),
            tolerance_pct=tolerance_pct,
            entry_strategy=ENTRY_STRATEGY,
            force_immediate=force_immediate,
        )

    def _close_orphan_positions(self) -> None:
        """Close positions not in current symbol set."""
        try:
            positions = alpaca_wrapper.get_all_positions()
        except Exception as exc:
            logger.warning(f"Unable to load positions: {exc}")
            return

        tradable = set(self.symbols)
        for pos in positions:
            sym = getattr(pos, "symbol", "").upper()
            if sym and sym not in tradable:
                try:
                    qty = float(getattr(pos, "qty", 0.0))
                except Exception:
                    qty = 0.0
                if qty == 0.0:
                    continue

                side = "sell" if qty > 0 else "buy"
                try:
                    quote = alpaca_wrapper.get_latest_quote(sym)
                    tp_price = float(
                        quote.ask_price if side == "sell" else quote.bid_price
                    )
                except Exception:
                    continue

                spawn_close_position_at_maxdiff_takeprofit(
                    sym,
                    side,
                    tp_price,
                    expiry_minutes=60,
                    price_tolerance=0.0066,
                    target_qty=abs(qty),
                    entry_strategy=ENTRY_STRATEGY,
                )
                logger.info(f"Spawned orphan closer for {sym}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-Stage Portfolio Trading - Live"
    )
    parser.add_argument(
        "--symbols", nargs="*",
        help="Symbols to trade (default: BTCUSD ETHUSD LINKUSD SPY QQQ)"
    )
    parser.add_argument(
        "--interval-seconds", type=int, default=DEFAULT_INTERVAL_SECONDS,
        help="Seconds between trading cycles"
    )
    parser.add_argument(
        "--min-confidence", type=float, default=0.3,
        help="Minimum overall confidence to trade"
    )
    parser.add_argument(
        "--max-lines", type=int, default=300,
        help="Max history lines to send to Claude"
    )
    parser.add_argument(
        "--no-chronos", action="store_true",
        help="Disable Chronos2 forecasts"
    )
    parser.add_argument(
        "--thinking", action="store_true",
        help="Use extended thinking (Opus) for Stage 1"
    )
    parser.add_argument(
        "--force-immediate", action="store_true",
        help="Force immediate order posting"
    )
    parser.add_argument(
        "--no-weekend-skip", action="store_true",
        help="Allow equity entries on weekends"
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run a single tick instead of looping"
    )
    parser.add_argument(
        "--log-windows", action="store_true",
        help="Log window-skips at info level"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(get_log_filename("twostage_trade_live.log", is_hourly=False))
    symbols = _resolve_symbols(args)

    logger.info(f"Starting Two-Stage Trading Loop")
    logger.info(f"  Symbols: {symbols}")
    logger.info(f"  Interval: {args.interval_seconds}s")
    logger.info(f"  Min confidence: {args.min_confidence}")
    logger.info(f"  Chronos: {'enabled' if not args.no_chronos else 'disabled'}")
    logger.info(f"  Thinking: {'enabled' if args.thinking else 'disabled'}")

    loop = TwoStageTradingLoop(
        symbols,
        interval_seconds=args.interval_seconds,
        min_confidence=args.min_confidence,
        force_immediate=args.force_immediate,
        skip_equity_weekends=not args.no_weekend_skip,
        log_windows=args.log_windows,
        max_lines=args.max_lines,
        use_chronos=not args.no_chronos,
        use_thinking=args.thinking,
    )

    def _handle_signal(signum, _frame):
        logger.warning(f"Received signal {signum} - stopping")
        loop._stop_requested = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    loop.run(once=args.once)


if __name__ == "__main__":
    main()
