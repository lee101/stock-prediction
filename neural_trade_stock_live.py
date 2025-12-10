#!/usr/bin/env python3
from __future__ import annotations

import argparse
import signal
from datetime import datetime, timezone, time as dtime, timedelta
import sys
import time
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo
import subprocess

import alpaca_wrapper
from loguru import logger

from neuraldailytraining import DailyTradingRuntime
from neuraldailytraining.config import DailyDatasetConfig
from neuraldailytraining.runtime import TradingPlan
from src.logging_utils import setup_logging, get_log_filename
from src.process_utils import (
    spawn_open_position_at_maxdiff_takeprofit,
    spawn_close_position_at_maxdiff_takeprofit,
    stop_all_entry_watchers,
    enforce_min_spread,
)
from src.fixtures import active_crypto_symbols
from src.symbol_utils import is_crypto_symbol


DEFAULT_INTERVAL_SECONDS = 300
ENTRY_STRATEGY = "neuraldaily"
MIN_STOCK_QTY = 1.0
MIN_CRYPTO_QTY = 0.001
MIN_ORDER_NOTIONAL = 1.0  # Alpaca enforces >= $1 cost basis per order
# Run windows (Eastern time) to avoid spamming daily logic.
WINDOW_OPEN = (dtime(9, 30), dtime(10, 0))       # US equities open
WINDOW_CLOSE = (dtime(15, 30), dtime(16, 0))     # US equities close
# Crypto daily bar alignment (UTC)
WINDOW_CRYPTO = (dtime(0, 0), dtime(0, 30))      # 00:00–00:30 UTC


def _load_account_equity() -> float:
    try:
        account = alpaca_wrapper.get_account()
    except Exception as exc:  # pragma: no cover - defensive
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
    if args.symbols:
        raw = tuple(symbol.upper() for symbol in args.symbols)
    else:
        raw = DailyDatasetConfig().symbols
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
    plan: TradingPlan
    symbol: str
    asset_flag: float
    priority: int


class NeuralTradingLoop:
    def __init__(
        self,
        runtime: DailyTradingRuntime,
        symbols: Sequence[str],
        *,
        interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
        min_trade_amount: float = 0.05,
        max_plans: int = 0,  # 0 = unlimited, trade all symbols the model recommends
        force_immediate: bool = False,
        skip_equity_weekends: bool = True,
        log_windows: bool = False,
    ) -> None:
        self.runtime = runtime
        self.symbols = [symbol.upper() for symbol in symbols]
        self.interval_seconds = max(30, int(interval_seconds))
        self.min_trade_amount = max(0.0, float(min_trade_amount))
        self.max_plans = max(0, int(max_plans))  # 0 = unlimited
        self.force_immediate = bool(force_immediate)
        self.skip_equity_weekends = bool(skip_equity_weekends)
        self.log_windows = bool(log_windows)
        self._stop_requested = False
        self._bootstrapped = False
        self._last_forecast_refresh: Optional[datetime.date] = None

    def run(self, *, once: bool = False) -> None:
        while not self._stop_requested:
            try:
                self._tick()
            except Exception as exc:  # pragma: no cover - log + continue
                logger.exception(f"Neural trading tick failed: {exc}")
            if once:
                break
            time.sleep(self.interval_seconds)

    def _tick(self) -> None:
        window = self._current_window()
        has_crypto = any(is_crypto_symbol(sym.upper()) for sym in self.symbols)
        if window is None:
            if has_crypto:
                window = "crypto_anytime"
            elif not self._bootstrapped:
                window = "bootstrap"
            else:
                (logger.info if self.log_windows else logger.debug)("Outside trading windows; skipping tick.")
                return

        # Refresh forecasts once per day before planning
        self._refresh_forecasts_if_needed()

        account_equity = _load_account_equity()
        plans = self._generate_plans()
        if not plans:
            logger.info("No eligible neural plans for current tick.")
            return
        total_weight = sum(plan.plan.trade_amount for plan in plans)
        logger.info(f"Generated {len(plans)} plans (aggregate weight {total_weight:.3f})")
        for neural_plan in plans:
            self._dispatch_plan(neural_plan, account_equity)
        self._bootstrapped = True
        self._close_orphan_positions()

    def _current_window(self) -> Optional[str]:
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
        today = datetime.now(timezone.utc).date()
        if self._last_forecast_refresh == today:
            return
        logger.info("Refreshing Chronos forecasts for all symbols (daily).")
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "update_chronos_forecasts.py",
                ],
                cwd=Path.cwd(),
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info("Forecast refresh completed: %s", completed.stdout.strip()[:200])
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("Forecast refresh failed: %s", exc)
        self._last_forecast_refresh = today

    def _cancel_equity_entries_if_closed(self) -> None:
        try:
            clock = alpaca_wrapper.alpaca_api.get_clock()
            is_open = getattr(clock, "is_open", False)
        except Exception:
            is_open = False
        if is_open:
            return
        for sym in self.symbols:
            if not is_crypto_symbol(sym):
                stop_all_entry_watchers(sym, reason="market_closed_equity")

    def _close_orphan_positions(self) -> None:
        """Close positions that are not in the current tradable symbol set."""
        try:
            positions = alpaca_wrapper.get_all_positions()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Unable to load positions for orphan close: {exc}")
            return

        tradable = set(self.symbols)
        orphans = []
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
                orphans.append((sym, side, abs(qty)))

        if not orphans:
            return

        for sym, exit_side, qty in orphans:
            # Try to get a model-informed price; fall back to current_quote if unavailable
            takeprofit_price = None
            try:
                plan = self.runtime.plan_for_symbol(sym)
                if plan:
                    takeprofit_price = plan.sell_price if exit_side == "sell" else plan.buy_price
            except Exception:
                plan = None
            if takeprofit_price is None:
                try:
                    quote = alpaca_wrapper.get_latest_quote(sym)
                    takeprofit_price = float(quote.ask_price if exit_side == "sell" else quote.bid_price)
                except Exception:
                    takeprofit_price = None
            if takeprofit_price is None:
                logger.warning("Skipping orphan close for %s; no price available.", sym)
                continue

            tolerance = 0.0066  # reuse entry tolerance
            expiry_min = 60  # 1 hour re-arm window
            spawn_close_position_at_maxdiff_takeprofit(
                sym,
                exit_side,
                takeprofit_price,
                expiry_minutes=expiry_min,
                price_tolerance=tolerance,
                target_qty=qty,
                entry_strategy=ENTRY_STRATEGY,
            )
            logger.info(
                "Spawned orphan closer for %s side=%s qty=%.4f tp=%.4f (strategy=%s)",
                sym,
                exit_side,
                qty,
                takeprofit_price,
                ENTRY_STRATEGY,
            )

    def _generate_plans(self) -> List[NeuralPlan]:
        raw_plans = self.runtime.generate_plans(self.symbols)
        enriched: List[NeuralPlan] = []
        for plan in raw_plans:
            if plan.trade_amount <= self.min_trade_amount:
                continue
            asset_flag = 1.0 if is_crypto_symbol(plan.symbol.upper()) else 0.0
            enriched.append(
                NeuralPlan(
                    plan=plan,
                    symbol=plan.symbol.upper(),
                    asset_flag=asset_flag,
                    priority=0,
                )
            )
        # Trust the neural network's trade_amount directly - it has all the data
        # (prices, Chronos forecasts, features) and was trained to output optimal allocation
        enriched.sort(key=lambda item: item.plan.trade_amount, reverse=True)

        # Apply max_plans limit if set, otherwise trade all eligible symbols
        if self.max_plans > 0 and len(enriched) > self.max_plans:
            enriched = enriched[:self.max_plans]

        for idx, item in enumerate(enriched, start=1):
            item.priority = idx
        return enriched

    def _dispatch_plan(self, neural_plan: NeuralPlan, account_equity: float) -> None:
        plan = neural_plan.plan
        symbol = neural_plan.symbol
        # Use model-provided asset flag first, fall back to symbol heuristic
        asset_is_crypto = (neural_plan.asset_flag > 0.5) or is_crypto_symbol(symbol)
        # Skip opening new equity positions on weekends to avoid consuming buying power with stale orders
        if self.skip_equity_weekends and not asset_is_crypto and datetime.now(timezone.utc).weekday() >= 5:
            logger.info(f"Weekend skip for {symbol} (equity); not dispatching new entries.")
            return
        # Skip equities outside market hours
        if not asset_is_crypto:
            try:
                if not alpaca_wrapper.alpaca_api.get_clock().is_open:
                    logger.info("Market closed; skipping equity plan for %s", symbol)
                    self._cancel_equity_entries_if_closed()
                    return
            except Exception as exc:
                logger.warning("Clock lookup failed (%s); proceeding cautiously for %s", exc, symbol)
        buy_price, sell_price = enforce_min_spread(plan.buy_price, plan.sell_price, min_spread_pct=0.0003)
        if sell_price > plan.sell_price + 1e-12:
            logger.warning(
                "Adjusted sell_price for %s to enforce min spread (%.4f -> %.4f)",
                symbol,
                plan.sell_price,
                sell_price,
            )
        limit_price = buy_price
        # Trust the neural network's trade_amount directly - it learned optimal allocation
        # Only cap crypto at 1x (no leverage available), stocks can use up to 2x as model learned
        if asset_is_crypto:
            allocation = min(plan.trade_amount, 1.0)  # Crypto: no leverage available
        else:
            allocation = min(plan.trade_amount, 2.0)  # Stocks: broker allows 2x margin
        target_notional = max(account_equity * allocation, 0.0)
        min_qty = MIN_CRYPTO_QTY if asset_is_crypto else MIN_STOCK_QTY
        qty = target_notional / limit_price
        # Enforce broker min notional to avoid 403 "cost basis must be >= 1"
        min_notional_qty = MIN_ORDER_NOTIONAL / max(limit_price, 1e-9)
        qty = max(qty, min_qty, min_notional_qty)
        if qty <= 0:
            logger.debug(f"Skipping {symbol} plan due to zero qty (notional {target_notional:.2f}).")
            return
        # Clear existing entry watchers to avoid overlapping prices from older strategies
        stop_all_entry_watchers(symbol, reason="neuraldaily_reset")
        logger.info(
            f"Dispatching {symbol} plan | buy @ {buy_price:.4f} sell @ {sell_price:.4f} qty={qty:.4f} "
            f"weight={plan.trade_amount:.3f} alloc={allocation:.3f} notional=${target_notional:.0f} priority={neural_plan.priority}"
        )
        self._spawn_entry(symbol, "buy", buy_price, qty, neural_plan)
        self._spawn_entry(symbol, "sell", sell_price, qty, neural_plan)

    def _spawn_entry(self, symbol: str, side: str, limit_price: float, qty: float, plan: NeuralPlan) -> None:
        tolerance_pct = None  # let the watcher decide
        asset_is_crypto = (plan.asset_flag > 0.5) or is_crypto_symbol(symbol)
        # Crypto: always force immediate to post orders even out of hours
        force_immediate = True if asset_is_crypto else self.force_immediate
        if asset_is_crypto:
            tolerance_pct = 0.001  # 10 bps for faster posting
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Neural daily production trading loop.")
    parser.add_argument("--checkpoint", help="Explicit path to a neuraldaily checkpoint.")
    parser.add_argument("--checkpoint-run-dir", help="Run directory containing manifest.json used to pick the best checkpoint automatically.")
    parser.add_argument("--symbols", nargs="*", help="Optional subset of symbols to trade.")
    parser.add_argument("--data-root", default="trainingdata/train")
    parser.add_argument("--forecast-cache", default="strategytraining/forecast_cache")
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--validation-days", type=int, default=40)
    parser.add_argument("--device", default=None)
    parser.add_argument("--interval-seconds", type=int, default=DEFAULT_INTERVAL_SECONDS)
    parser.add_argument("--min-trade-amount", type=float, default=0.05)
    parser.add_argument("--max-plans", type=int, default=0, help="Max plans per tick (0 = unlimited, trust the model).")
    parser.add_argument("--force-immediate", action="store_true", help="Force immediate execution on spawned entries.")
    parser.add_argument("--no-weekend-skip", action="store_true", help="Allow equity entries on weekends (default skips).")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="Minimum neural confidence required to dispatch a plan (0-1). Default: trust the model.",
    )
    parser.add_argument(
        "--ignore-non-tradable",
        action="store_true",
        help="Ignore checkpoint non_tradable file and allow all symbols.",
    )
    parser.add_argument("--once", action="store_true", help="Run a single tick instead of looping.")
    parser.add_argument("--log-windows", action="store_true", help="Log window-skips at info level (otherwise debug).")
    return parser.parse_args()


def _resolve_checkpoint_path(args: argparse.Namespace) -> Path:
    if args.checkpoint:
        path = Path(args.checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint {path} not found.")
        return path
    if args.checkpoint_run_dir:
        run_dir = Path(args.checkpoint_run_dir)
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found under {manifest_path}")
        data = json.loads(manifest_path.read_text())
        checkpoints = data.get("checkpoints") or []
        if not checkpoints:
            raise ValueError(f"No checkpoints listed in {manifest_path}")
        # min() correctly selects most negative val_loss (best) since loss = -score
        best = min(checkpoints, key=lambda item: item.get("val_loss", float("inf")))
        best_path = run_dir / best["path"]
        if not best_path.exists():
            raise FileNotFoundError(f"Best checkpoint {best_path} missing.")
        logger.info(
            f"Resolved best checkpoint {best_path} (val_loss {best.get('val_loss'):.6f} epoch {best.get('epoch')})"
        )
        return best_path
    raise ValueError("Must supply either --checkpoint or --checkpoint-run-dir.")


def build_dataset_config(args: argparse.Namespace, symbols: Sequence[str]) -> DailyDatasetConfig:
    return DailyDatasetConfig(
        symbols=tuple(symbols),
        data_root=Path(args.data_root),
        forecast_cache_dir=Path(args.forecast_cache),
        sequence_length=args.sequence_length,
        val_fraction=args.val_fraction,
        validation_days=args.validation_days,
    )


def main() -> None:
    args = parse_args()
    setup_logging(get_log_filename("neural_trade_stock_live.log", is_hourly=False))
    symbols = _resolve_symbols(args)
    checkpoint_path = _resolve_checkpoint_path(args)
    dataset_cfg = build_dataset_config(args, symbols)
    runtime = DailyTradingRuntime(
        checkpoint_path,
        dataset_config=dataset_cfg,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
        non_tradable=() if args.ignore_non_tradable else None,
    )
    loop = NeuralTradingLoop(
        runtime,
        symbols,
        interval_seconds=args.interval_seconds,
        min_trade_amount=args.min_trade_amount,
        max_plans=args.max_plans,
        force_immediate=args.force_immediate,
        skip_equity_weekends=not args.no_weekend_skip,
        log_windows=args.log_windows,
    )

    def _handle_signal(signum, _frame):  # pragma: no cover - signal handling
        logger.warning(f"Received signal {signum} - stopping neural trading loop.")
        loop._stop_requested = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    loop.run(once=args.once)


if __name__ == "__main__":
    main()
