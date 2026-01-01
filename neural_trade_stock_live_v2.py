#!/usr/bin/env python3
"""V2 Neural Daily live trading script.

Uses V2 runtime with unified simulation (temperature=0).
This ensures inference exactly matches what the model was trained toward.
"""
from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import alpaca_wrapper
from loguru import logger

from neuraldailyv2 import DailyDatasetConfigV2, DailyTradingRuntimeV2, TradingPlan
from src.fixtures import active_crypto_symbols
from src.logging_utils import get_log_filename, setup_logging
from src.process_utils import (
    enforce_min_spread,
    spawn_close_position_at_maxdiff_takeprofit,
    spawn_open_position_at_maxdiff_takeprofit,
    stop_all_entry_watchers,
)
from src.symbol_utils import is_crypto_symbol
from src.date_utils import is_nyse_open_on_date


DEFAULT_INTERVAL_SECONDS = 300
ENTRY_STRATEGY = "neuraldailyv2"
MIN_STOCK_QTY = 1.0
MIN_CRYPTO_QTY = 0.001
MIN_ORDER_NOTIONAL = 1.0

# Trading windows (Eastern time)
WINDOW_OPEN = (dtime(9, 30), dtime(10, 0))
WINDOW_CLOSE = (dtime(15, 30), dtime(16, 0))
WINDOW_CRYPTO = (dtime(0, 0), dtime(0, 30))  # UTC


def _load_account_equity() -> float:
    """Load current account equity."""
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
        logger.warning("Account equity unavailable; defaulting to 1.0")
        return 1.0
    return equity


def _resolve_symbols(args: argparse.Namespace) -> Sequence[str]:
    """Resolve symbols to trade."""
    if args.symbols:
        raw = tuple(symbol.upper() for symbol in args.symbols)
    else:
        raw = DailyDatasetConfigV2().symbols
    active_crypto = {sym.upper() for sym in active_crypto_symbols}
    filtered = []
    for sym in raw:
        up = sym.upper()
        if is_crypto_symbol(up) and up not in active_crypto:
            continue
        filtered.append(up)
    return tuple(filtered)


@dataclass
class NeuralPlan:
    """Enriched trading plan with priority."""
    plan: TradingPlan
    symbol: str
    asset_flag: float
    priority: int


class NeuralTradingLoopV2:
    """V2 trading loop using unified simulation runtime."""

    def __init__(
        self,
        runtime: DailyTradingRuntimeV2,
        symbols: Sequence[str],
        *,
        non_tradable: Optional[Sequence[str]] = None,
        interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
        min_trade_amount: float = 0.05,
        max_plans: int = 0,
        force_immediate: bool = False,
        skip_equity_weekends: bool = True,
        log_windows: bool = False,
    ) -> None:
        self.runtime = runtime
        # All symbols for data/cross-attention
        self.symbols = [symbol.upper() for symbol in symbols]
        # Symbols excluded from trading but still used for cross-attention
        self.non_tradable = set(s.upper() for s in (non_tradable or []))
        # Symbols we actively trade
        self.tradable_symbols = [s for s in self.symbols if s not in self.non_tradable]
        self.interval_seconds = max(30, int(interval_seconds))
        self.min_trade_amount = max(0.0, float(min_trade_amount))
        self.max_plans = max(0, int(max_plans))
        self.force_immediate = bool(force_immediate)
        self.skip_equity_weekends = bool(skip_equity_weekends)
        self.log_windows = bool(log_windows)
        self._stop_requested = False
        self._bootstrapped = False
        self._last_forecast_refresh: Optional[datetime.date] = None

        if self.non_tradable:
            logger.info(f"Non-tradable symbols (data only): {sorted(self.non_tradable)}")
            logger.info(f"Actively trading: {self.tradable_symbols}")

    def run(self, *, once: bool = False) -> None:
        """Main trading loop."""
        while not self._stop_requested:
            try:
                self._tick()
            except Exception as exc:
                logger.exception(f"V2 trading tick failed: {exc}")
            if once:
                break
            time.sleep(self.interval_seconds)

    def _tick(self) -> None:
        """Execute one trading tick."""
        window = self._current_window()
        has_crypto = any(is_crypto_symbol(sym.upper()) for sym in self.symbols)

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

        # Refresh forecasts daily
        self._refresh_forecasts_if_needed()

        account_equity = _load_account_equity()
        plans = self._generate_plans()
        if not plans:
            logger.info("No eligible V2 plans for current tick.")
            return

        total_weight = sum(plan.plan.trade_amount for plan in plans)
        logger.info(f"Generated {len(plans)} V2 plans (aggregate weight {total_weight:.3f})")

        for neural_plan in plans:
            self._dispatch_plan(neural_plan, account_equity)

        self._bootstrapped = True
        self._close_orphan_positions()

    def _current_window(self) -> Optional[str]:
        """Determine current trading window."""
        now_utc = datetime.now(timezone.utc)
        now_et = now_utc.astimezone(ZoneInfo("US/Eastern"))

        def _in_window(window: Tuple[dtime, dtime], clock: dtime) -> bool:
            return window[0] <= clock < window[1]

        if _in_window(WINDOW_OPEN, now_et.time()):
            return "open"
        if _in_window(WINDOW_CLOSE, now_et.time()):
            return "close"
        if _in_window(WINDOW_CRYPTO, now_utc.time()):
            return "crypto"
        return None

    def _refresh_forecasts_if_needed(self) -> None:
        """Refresh Chronos forecasts daily."""
        today = datetime.now(timezone.utc).date()
        if self._last_forecast_refresh == today:
            return
        logger.info("Refreshing Chronos forecasts (V2).")
        try:
            subprocess.run(
                [sys.executable, "update_chronos_forecasts.py"],
                cwd=Path.cwd(),
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception as exc:
            logger.warning(f"Forecast refresh failed: {exc}")
        self._last_forecast_refresh = today

    def _generate_plans(self) -> List[NeuralPlan]:
        """Generate trading plans from V2 runtime."""
        # Pass ALL symbols for cross-attention, filter results for tradable only
        raw_plans = self.runtime.generate_plans(self.symbols)
        enriched: List[NeuralPlan] = []
        tradable_set = set(self.tradable_symbols)

        for plan in raw_plans:
            sym = plan.symbol.upper()
            # Skip non-tradable symbols
            if sym not in tradable_set:
                continue
            if plan.trade_amount <= self.min_trade_amount:
                continue
            enriched.append(NeuralPlan(
                plan=plan,
                symbol=sym,
                asset_flag=plan.asset_class,
                priority=0,
            ))

        # Sort by trade_amount (trust the model)
        enriched.sort(key=lambda item: item.plan.trade_amount, reverse=True)

        # Apply max_plans limit
        if self.max_plans > 0 and len(enriched) > self.max_plans:
            enriched = enriched[:self.max_plans]

        for idx, item in enumerate(enriched, start=1):
            item.priority = idx

        return enriched

    def _dispatch_plan(self, neural_plan: NeuralPlan, account_equity: float) -> None:
        """Dispatch a trading plan to the order system."""
        plan = neural_plan.plan
        symbol = neural_plan.symbol
        asset_is_crypto = (neural_plan.asset_flag > 0.5) or is_crypto_symbol(symbol)

        # Skip equity on weekends
        if self.skip_equity_weekends and not asset_is_crypto:
            if not is_nyse_open_on_date(datetime.now(timezone.utc)):
                logger.info(f"Market closed skip for {symbol} (equity).")
                return

        # Skip equities outside market hours
        if not asset_is_crypto:
            try:
                if not alpaca_wrapper.alpaca_api.get_clock().is_open:
                    logger.info(f"Market closed; skipping equity plan for {symbol}")
                    return
            except Exception as exc:
                logger.warning(f"Clock lookup failed ({exc}); proceeding for {symbol}")

        # Enforce minimum spread
        buy_price, sell_price = enforce_min_spread(
            plan.buy_price, plan.sell_price, min_spread_pct=0.0003
        )

        # Position sizing - trust the model's trade_amount
        # V2 interpretation: trade_amount is already scaled by leverage
        if asset_is_crypto:
            allocation = min(plan.trade_amount, 1.0)
        else:
            allocation = min(plan.trade_amount, 2.0)

        target_notional = max(account_equity * allocation, 0.0)
        limit_price = buy_price
        min_qty = MIN_CRYPTO_QTY if asset_is_crypto else MIN_STOCK_QTY
        qty = target_notional / limit_price

        # Enforce broker minimums
        min_notional_qty = MIN_ORDER_NOTIONAL / max(limit_price, 1e-9)
        qty = max(qty, min_qty, min_notional_qty)

        if qty <= 0:
            logger.debug(f"Skipping {symbol} due to zero qty.")
            return

        # Clear existing watchers
        stop_all_entry_watchers(symbol, reason="neuraldailyv2_reset")

        logger.info(
            f"Dispatching V2 {symbol} | buy @ {buy_price:.4f} sell @ {sell_price:.4f} "
            f"qty={qty:.4f} weight={plan.trade_amount:.3f} alloc={allocation:.3f} "
            f"notional=${target_notional:.0f} priority={neural_plan.priority}"
        )

        self._spawn_entry(symbol, "buy", buy_price, qty, neural_plan)
        self._spawn_entry(symbol, "sell", sell_price, qty, neural_plan)

    def _spawn_entry(
        self, symbol: str, side: str, limit_price: float, qty: float, plan: NeuralPlan
    ) -> None:
        """Spawn entry order watcher."""
        asset_is_crypto = (plan.asset_flag > 0.5) or is_crypto_symbol(symbol)
        force_immediate = True if asset_is_crypto else self.force_immediate
        tolerance_pct = 0.001 if asset_is_crypto else None
        priority_rank = plan.priority
        crypto_rank = priority_rank if asset_is_crypto else None

        spawn_open_position_at_maxdiff_takeprofit(
            symbol=symbol,
            side=side,
            limit_price=float(limit_price),
            target_qty=float(qty),
            tolerance_pct=tolerance_pct,
            entry_strategy=ENTRY_STRATEGY,
            force_immediate=force_immediate,
            priority_rank=priority_rank,
            crypto_rank=crypto_rank,
        )

    def _close_orphan_positions(self) -> None:
        """Close positions not in current tradable set (includes non-tradable symbols)."""
        try:
            positions = alpaca_wrapper.get_all_positions()
        except Exception as exc:
            logger.warning(f"Unable to load positions: {exc}")
            return

        # Only keep positions in actively tradable symbols
        tradable = set(self.tradable_symbols)
        orphans = []

        for pos in positions:
            sym = getattr(pos, "symbol", "").upper()
            # Close if not in tradable set OR if explicitly marked non-tradable
            if sym and (sym not in tradable or sym in self.non_tradable):
                try:
                    qty = float(getattr(pos, "qty", 0.0))
                except Exception:
                    qty = 0.0
                if qty == 0.0:
                    continue
                side = "sell" if qty > 0 else "buy"
                orphans.append((sym, side, abs(qty)))

        for sym, exit_side, qty in orphans:
            # Get exit price
            takeprofit_price = None
            try:
                plan = self.runtime.plan_for_symbol(sym)
                if plan:
                    takeprofit_price = plan.sell_price if exit_side == "sell" else plan.buy_price
            except Exception:
                pass

            if takeprofit_price is None:
                try:
                    quote = alpaca_wrapper.get_latest_quote(sym)
                    takeprofit_price = float(
                        quote.ask_price if exit_side == "sell" else quote.bid_price
                    )
                except Exception:
                    pass

            if takeprofit_price is None:
                logger.warning(f"Skipping orphan close for {sym}; no price.")
                continue

            spawn_close_position_at_maxdiff_takeprofit(
                sym,
                exit_side,
                takeprofit_price,
                expiry_minutes=60,
                price_tolerance=0.0066,
                target_qty=qty,
                entry_strategy=ENTRY_STRATEGY,
            )
            logger.info(f"Spawned V2 orphan closer for {sym}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="V2 Neural daily live trading.")
    parser.add_argument("--checkpoint", required=True, help="Path to V2 checkpoint.")
    parser.add_argument("--symbols", nargs="*", help="Symbols for data/cross-attention.")
    parser.add_argument("--non-tradable", nargs="*", default=[], help="Symbols to exclude from trading (still used for cross-attention).")
    parser.add_argument("--data-root", default="trainingdata/train")
    parser.add_argument("--forecast-cache", default="strategytraining/forecast_cache")
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--device", default=None)
    parser.add_argument("--interval-seconds", type=int, default=DEFAULT_INTERVAL_SECONDS)
    parser.add_argument("--min-trade-amount", type=float, default=0.05)
    parser.add_argument("--max-plans", type=int, default=0)
    parser.add_argument("--force-immediate", action="store_true")
    parser.add_argument("--no-weekend-skip", action="store_true")
    parser.add_argument("--confidence-threshold", type=float, default=None)
    parser.add_argument("--ignore-non-tradable", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--log-windows", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    setup_logging(get_log_filename("neural_trade_stock_live_v2.log", is_hourly=False))

    symbols = _resolve_symbols(args)
    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    dataset_cfg = DailyDatasetConfigV2(
        symbols=tuple(symbols),
        data_root=Path(args.data_root),
        forecast_cache_dir=Path(args.forecast_cache),
        sequence_length=args.sequence_length,
    )

    runtime = DailyTradingRuntimeV2(
        checkpoint_path,
        dataset_config=dataset_cfg,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
        non_tradable=() if args.ignore_non_tradable else None,
    )

    # Get non-tradable symbols (use underscore for Python attr access)
    non_tradable = getattr(args, 'non_tradable', []) or []

    loop = NeuralTradingLoopV2(
        runtime,
        symbols,
        non_tradable=non_tradable,
        interval_seconds=args.interval_seconds,
        min_trade_amount=args.min_trade_amount,
        max_plans=args.max_plans,
        force_immediate=args.force_immediate,
        skip_equity_weekends=not args.no_weekend_skip,
        log_windows=args.log_windows,
    )

    def _handle_signal(signum, _frame):
        logger.warning(f"Received signal {signum} - stopping V2 trading loop.")
        loop._stop_requested = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    logger.info("Starting NeuralDaily V2 live trading loop.")
    loop.run(once=args.once)


if __name__ == "__main__":
    main()
