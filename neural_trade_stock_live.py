#!/usr/bin/env python3
from __future__ import annotations

import argparse
import signal
from datetime import datetime, timezone, time as dtime
import sys
import time
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple
from zoneinfo import ZoneInfo

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
from src.symbol_utils import is_crypto_symbol


DEFAULT_INTERVAL_SECONDS = 300
DEFAULT_ACCOUNT_FRACTION = 0.15
ENTRY_STRATEGY = "neuraldaily"
MIN_STOCK_QTY = 1.0
MIN_CRYPTO_QTY = 0.001
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
        return tuple(symbol.upper() for symbol in args.symbols)
    dataset = DailyDatasetConfig()
    return dataset.symbols


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
        account_fraction: float = DEFAULT_ACCOUNT_FRACTION,
        min_trade_amount: float = 0.05,
        max_plans: int = 2,
        force_immediate: bool = False,
        skip_equity_weekends: bool = True,
        log_windows: bool = False,
    ) -> None:
        self.runtime = runtime
        self.symbols = [symbol.upper() for symbol in symbols]
        self.interval_seconds = max(30, int(interval_seconds))
        self.account_fraction = max(0.01, float(account_fraction))
        self.min_trade_amount = max(0.0, float(min_trade_amount))
        self.max_plans = max(1, int(max_plans))
        self.force_immediate = bool(force_immediate)
        self.skip_equity_weekends = bool(skip_equity_weekends)
        self.log_windows = bool(log_windows)
        self._stop_requested = False
        self._last_windows_run: Set[Tuple[datetime.date, str]] = set()
        self._last_crypto_anytime: Optional[datetime.date] = None
        self._bootstrapped = False

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
        if window is None:
            today = datetime.now(timezone.utc).date()
            has_crypto = any(sym.endswith("USD") for sym in self.symbols)
            if has_crypto and self._last_crypto_anytime != today:
                window = "crypto_anytime"
                self._last_crypto_anytime = today
            elif not self._bootstrapped:
                window = "bootstrap"
            else:
                (logger.info if self.log_windows else logger.debug)("Outside trading windows; skipping tick.")
                return
        today = datetime.now(timezone.utc).date()
        key = (today, window)
        if key in self._last_windows_run:
            (logger.info if self.log_windows else logger.debug)(
                "Window %s already processed today; skipping repeat tick.", window
            )
            return

        account_equity = _load_account_equity()
        plans = self._generate_plans()
        if not plans:
            logger.info("No eligible neural plans for current tick.")
            return
        total_weight = sum(plan.plan.trade_amount for plan in plans)
        logger.info(f"Generated {len(plans)} plans (aggregate weight {total_weight:.3f})")
        for neural_plan in plans:
            self._dispatch_plan(neural_plan, account_equity)
        self._last_windows_run.add(key)
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
            asset_flag = 1.0 if plan.symbol.upper().endswith("-USD") else 0.0
            edge = max(plan.sell_price - plan.buy_price, 0.0) / max(plan.buy_price, 1e-9)
            score = plan.trade_amount * edge
            enriched.append(
                NeuralPlan(
                    plan=plan,
                    symbol=plan.symbol.upper(),
                    asset_flag=asset_flag,
                    priority=0,
                    # Attach score to prioritize tighter/edge-rich opportunities
                    # (used only for sorting; not part of dataclass fields)
                )
            )
        # Sort by expected edge*size, then fall back to trade_amount
        enriched.sort(key=lambda item: (item.plan.trade_amount * max(item.plan.sell_price - item.plan.buy_price, 0.0) / max(item.plan.buy_price, 1e-9),
                                        item.plan.trade_amount),
                      reverse=True)
        enriched = enriched[: self.max_plans]
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
        allocation = min(plan.trade_amount, self.runtime.risk_threshold)
        notional_cap = account_equity * self.account_fraction
        target_notional = max(notional_cap * allocation, 0.0)
        min_qty = MIN_CRYPTO_QTY if asset_is_crypto else MIN_STOCK_QTY
        qty = target_notional / limit_price
        qty = max(qty, min_qty)
        if qty <= 0:
            logger.debug(f"Skipping {symbol} plan due to zero qty (notional {target_notional:.2f}).")
            return
        # Clear existing entry watchers to avoid overlapping prices from older strategies
        stop_all_entry_watchers(symbol, reason="neuraldaily_reset")
        logger.info(
            f"Dispatching {symbol} plan | buy @ {buy_price:.4f} sell @ {sell_price:.4f} qty={qty:.4f} "
            f"weight={plan.trade_amount:.3f} priority={neural_plan.priority}"
        )
        self._spawn_entry(symbol, "buy", buy_price, qty, neural_plan)
        self._spawn_entry(symbol, "sell", sell_price, qty, neural_plan)

    def _spawn_entry(self, symbol: str, side: str, limit_price: float, qty: float, plan: NeuralPlan) -> None:
        tolerance_pct = None  # let the watcher decide
        asset_is_crypto = (plan.asset_flag > 0.5) or is_crypto_symbol(symbol)
        force_immediate = True if asset_is_crypto else self.force_immediate
        priority_rank = plan.priority
        crypto_rank = 1 if asset_is_crypto else None
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
    parser.add_argument("--risk-threshold", type=float, help="Override runtime risk clamp.")
    parser.add_argument("--interval-seconds", type=int, default=DEFAULT_INTERVAL_SECONDS)
    parser.add_argument("--account-fraction", type=float, default=DEFAULT_ACCOUNT_FRACTION)
    parser.add_argument("--min-trade-amount", type=float, default=0.05)
    parser.add_argument("--max-plans", type=int, default=2, help="Max number of plans to dispatch per tick.")
    parser.add_argument("--force-immediate", action="store_true", help="Force immediate execution on spawned entries.")
    parser.add_argument("--no-weekend-skip", action="store_true", help="Allow equity entries on weekends (default skips).")
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
        risk_threshold=args.risk_threshold,
    )
    loop = NeuralTradingLoop(
        runtime,
        symbols,
        interval_seconds=args.interval_seconds,
        account_fraction=args.account_fraction,
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
