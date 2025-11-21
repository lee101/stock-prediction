#!/usr/bin/env python3
"""
Neural daily trading with probe mode support.
Integrates risk controls to reduce position sizing when experiencing losses.
"""
from __future__ import annotations

import argparse
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import alpaca_wrapper
from jsonshelve import FlatShelf
from loguru import logger

from neuraldailytraining import DailyTradingRuntime
from neuraldailytraining.config import DailyDatasetConfig
from neuraldailytraining.runtime import TradingPlan
from src.logging_utils import setup_logging
from src.process_utils import spawn_open_position_at_maxdiff_takeprofit
from src.risk_state import resolve_probe_state, ProbeState
from src.symbol_utils import is_crypto_symbol
from stock.state import ensure_state_dir, get_state_file, resolve_state_suffix


DEFAULT_INTERVAL_SECONDS = 300
DEFAULT_ACCOUNT_FRACTION = 0.15
ENTRY_STRATEGY = "neuraldaily"
MIN_STOCK_QTY = 1.0
MIN_CRYPTO_QTY = 0.001
PROBE_NOTIONAL_LIMIT = 300.0  # Max notional for probe trades


# Trade history tracking
STATE_SUFFIX = resolve_state_suffix()
TRADE_HISTORY_FILE = get_state_file("neural_trade_history", STATE_SUFFIX)
_trade_history_store: Optional[FlatShelf] = None


def _init_trade_history_store() -> Optional[FlatShelf]:
    """Initialize the trade history store."""
    global _trade_history_store
    if _trade_history_store is not None:
        return _trade_history_store
    try:
        ensure_state_dir()
        _trade_history_store = FlatShelf(str(TRADE_HISTORY_FILE))
        logger.debug(f"Initialized trade history store at {TRADE_HISTORY_FILE}")
        return _trade_history_store
    except Exception as exc:
        logger.error(f"Failed to initialize trade history store: {exc}")
        return None


def _get_trade_history_store() -> Optional[FlatShelf]:
    """Get the trade history store, initializing if needed."""
    if _trade_history_store is None:
        return _init_trade_history_store()
    return _trade_history_store


def _state_key(symbol: str, side: str) -> str:
    """Generate a state key for symbol and side."""
    return f"{symbol.upper()}_{side.lower()}"


def record_trade_outcome(symbol: str, side: str, pnl: float, pnl_pct: float) -> None:
    """Record the outcome of a completed trade."""
    store = _get_trade_history_store()
    if store is None:
        return
    try:
        store.load()
        key = _state_key(symbol, side)
        history = store.get(key, [])
        if not isinstance(history, list):
            history = []

        entry = {
            "symbol": symbol,
            "side": side,
            "pnl": float(pnl),
            "pnl_pct": float(pnl_pct),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        history.append(entry)

        # Keep only last 10 trades
        if len(history) > 10:
            history = history[-10:]

        store[key] = history
        store.save()
        logger.info(f"Recorded trade outcome for {symbol} {side}: PnL={pnl:.2f} ({pnl_pct:.2%})")
    except Exception as exc:
        logger.error(f"Failed to record trade outcome for {symbol} {side}: {exc}")


def _get_recent_trade_pnl_pcts(symbol: str, side: str, limit: int = 2) -> List[float]:
    """Get the most recent trade PnL percentages for a symbol and side."""
    store = _get_trade_history_store()
    if store is None or limit <= 0:
        return []
    try:
        store.load()
        key = _state_key(symbol, side)
        history = store.get(key, [])
        if not isinstance(history, list) or not history:
            return []

        recent: List[float] = []
        for entry in reversed(history):
            pnl_pct = entry.get("pnl_pct")
            if pnl_pct is not None:
                recent.append(float(pnl_pct))
                if len(recent) >= limit:
                    break
        return recent
    except Exception as exc:
        logger.error(f"Failed to get recent trade PnL for {symbol} {side}: {exc}")
        return []


def _should_use_probe_mode(symbol: str, side: str, probe_state: ProbeState) -> tuple[bool, Optional[str]]:
    """
    Determine if probe mode should be used for this symbol/side.

    Returns:
        (should_probe, reason)
    """
    # Check global probe state (negative account PnL)
    if probe_state.force_probe:
        reason = probe_state.reason or "previous_day_loss"
        return True, f"global:{reason}"

    # Check per-symbol recent trade performance
    recent_pnls = _get_recent_trade_pnl_pcts(symbol, side, limit=2)
    if len(recent_pnls) >= 2:
        # Have 2+ trades: sum them and check if non-positive
        pnl_sum = sum(recent_pnls)
        if pnl_sum <= 0:
            return True, f"recent_pnl_sum={pnl_sum:.4f} [{', '.join(f'{p:.4f}' for p in recent_pnls)}]"
    elif len(recent_pnls) == 1:
        # Have only 1 trade: check if it's negative
        pnl = recent_pnls[0]
        if pnl <= 0:
            return True, f"single_trade_negative={pnl:.4f}"

    return False, None


def _load_account_equity() -> float:
    """Load current account equity from Alpaca."""
    try:
        account = alpaca_wrapper.get_account()
    except Exception as exc:
        logger.error("Unable to load Alpaca account: %s", exc)
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
    """Resolve the symbols to trade."""
    if args.symbols:
        return tuple(symbol.upper() for symbol in args.symbols)
    dataset = DailyDatasetConfig()
    return dataset.symbols


@dataclass
class NeuralPlan:
    """Enhanced trading plan with probe mode support."""
    plan: TradingPlan
    symbol: str
    asset_flag: float
    priority: int
    is_probe: bool = False
    probe_reason: Optional[str] = None


class NeuralTradingLoop:
    """Neural trading loop with probe mode risk controls."""

    def __init__(
        self,
        runtime: DailyTradingRuntime,
        symbols: Sequence[str],
        *,
        interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
        account_fraction: float = DEFAULT_ACCOUNT_FRACTION,
        min_trade_amount: float = 0.05,
        enable_probe_mode: bool = True,
    ) -> None:
        self.runtime = runtime
        self.symbols = [symbol.upper() for symbol in symbols]
        self.interval_seconds = max(30, int(interval_seconds))
        self.account_fraction = max(0.01, float(account_fraction))
        self.min_trade_amount = max(0.0, float(min_trade_amount))
        self.enable_probe_mode = enable_probe_mode
        self._stop_requested = False

    def run(self, *, once: bool = False) -> None:
        """Run the trading loop."""
        while not self._stop_requested:
            try:
                self._tick()
            except Exception as exc:
                logger.exception("Neural trading tick failed: %s", exc)
            if once:
                break
            time.sleep(self.interval_seconds)

    def _tick(self) -> None:
        """Execute one trading tick."""
        account_equity = _load_account_equity()
        probe_state = resolve_probe_state() if self.enable_probe_mode else ProbeState(False, None, None, {})

        if probe_state.force_probe:
            logger.warning(f"🚨 GLOBAL PROBE MODE ACTIVE: {probe_state.reason}")

        plans = self._generate_plans(probe_state)
        if not plans:
            logger.info("No eligible neural plans for current tick.")
            return

        probe_count = sum(1 for p in plans if p.is_probe)
        total_weight = sum(plan.plan.trade_amount for plan in plans)
        logger.info(
            f"Generated {len(plans)} plans (probe: {probe_count}, normal: {len(plans) - probe_count}, "
            f"aggregate weight: {total_weight:.3f})"
        )

        for neural_plan in plans:
            self._dispatch_plan(neural_plan, account_equity)

    def _generate_plans(self, probe_state: ProbeState) -> List[NeuralPlan]:
        """Generate trading plans with probe mode annotations."""
        raw_plans = self.runtime.generate_plans(self.symbols)
        enriched: List[NeuralPlan] = []

        for plan in raw_plans:
            if plan.trade_amount <= self.min_trade_amount:
                continue

            asset_flag = 1.0 if plan.symbol.upper().endswith("-USD") else 0.0

            # Determine if probe mode should be used
            is_probe = False
            probe_reason = None
            if self.enable_probe_mode:
                is_probe, probe_reason = _should_use_probe_mode(plan.symbol, "buy", probe_state)

            enriched.append(
                NeuralPlan(
                    plan=plan,
                    symbol=plan.symbol.upper(),
                    asset_flag=asset_flag,
                    priority=0,
                    is_probe=is_probe,
                    probe_reason=probe_reason,
                )
            )

        # Sort by trade amount (descending) and assign priorities
        enriched.sort(key=lambda item: item.plan.trade_amount, reverse=True)
        for idx, item in enumerate(enriched, start=1):
            item.priority = idx

        return enriched

    def _dispatch_plan(self, neural_plan: NeuralPlan, account_equity: float) -> None:
        """Dispatch a trading plan, respecting probe mode limits."""
        plan = neural_plan.plan
        symbol = neural_plan.symbol
        asset_is_crypto = is_crypto_symbol(symbol)
        limit_price = max(plan.buy_price, 1e-6)

        # Calculate quantity based on probe mode
        if neural_plan.is_probe:
            # Probe mode: use minimal notional
            target_notional = min(PROBE_NOTIONAL_LIMIT, account_equity * 0.01)
            mode_label = "PROBE"
            logger.info(
                f"🔬 {symbol} entering probe mode (reason: {neural_plan.probe_reason})"
            )
        else:
            # Normal mode: use account fraction and allocation
            allocation = min(plan.trade_amount, self.runtime.risk_threshold)
            notional_cap = account_equity * self.account_fraction
            target_notional = max(notional_cap * allocation, 0.0)
            mode_label = "NORMAL"

        min_qty = MIN_CRYPTO_QTY if asset_is_crypto else MIN_STOCK_QTY
        qty = target_notional / limit_price
        qty = max(qty, min_qty)

        if qty <= 0:
            logger.debug(f"Skipping {symbol} plan due to zero qty (notional {target_notional:.2f}).")
            return

        logger.info(
            f"[{mode_label}] Dispatching {symbol} | buy @ {plan.buy_price:.4f} "
            f"qty={qty:.4f} weight={plan.trade_amount:.3f} priority={neural_plan.priority}"
        )

        self._spawn_entry(symbol, "buy", plan.buy_price, qty, neural_plan)
        self._spawn_entry(symbol, "sell", plan.sell_price, qty, neural_plan)

    def _spawn_entry(
        self,
        symbol: str,
        side: str,
        limit_price: float,
        qty: float,
        plan: NeuralPlan
    ) -> None:
        """Spawn a watcher process for the entry."""
        tolerance_pct = None
        force_immediate = False
        priority_rank = plan.priority
        crypto_rank = plan.priority if plan.asset_flag > 0.5 else None

        spawn_open_position_at_maxdiff_takeprofit(
            symbol=symbol,
            side=side,
            entry_limit=limit_price,
            qty=qty,
            maxdiff_pct=0.0,
            takeprofit_limit=None,
            mode="entry",
            strategy=ENTRY_STRATEGY,
            tolerance_pct=tolerance_pct,
            force_immediate=force_immediate,
            priority=priority_rank,
            crypto_priority_override=crypto_rank,
        )

    def request_stop(self) -> None:
        """Request the loop to stop."""
        self._stop_requested = True


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Neural daily trading with probe mode")
    parser.add_argument("--checkpoint", required=True, help="Path to the neuraldaily checkpoint")
    parser.add_argument("--symbols", nargs="*", help="Symbols to trade (default: all from config)")
    parser.add_argument("--data-root", default="trainingdata/train")
    parser.add_argument("--forecast-cache", default="strategytraining/forecast_cache")
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--validation-days", type=int, default=40)
    parser.add_argument("--device", default=None)
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL_SECONDS)
    parser.add_argument("--account-fraction", type=float, default=DEFAULT_ACCOUNT_FRACTION)
    parser.add_argument("--min-trade-amount", type=float, default=0.05)
    parser.add_argument("--risk-threshold", type=float, help="Override risk threshold")
    parser.add_argument("--disable-probe-mode", action="store_true", help="Disable probe mode")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Setup logging
    log_file = setup_logging("neural_daily_trade_with_probe")
    logger.info(f"Starting neural daily trading with probe mode")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Probe mode: {'DISABLED' if args.disable_probe_mode else 'ENABLED (default)'}")

    # Build dataset config
    from neural_trade_stock_e2e import _build_dataset_config
    dataset_cfg = _build_dataset_config(args)

    # Initialize runtime
    runtime = DailyTradingRuntime(
        args.checkpoint,
        dataset_config=dataset_cfg,
        device=args.device,
        risk_threshold=args.risk_threshold,
    )

    symbols = args.symbols or list(dataset_cfg.symbols)
    logger.info(f"Trading {len(symbols)} symbols: {', '.join(symbols)}")
    logger.info(f"Risk threshold: {runtime.risk_threshold}")

    # Initialize trading loop
    trading_loop = NeuralTradingLoop(
        runtime,
        symbols,
        interval_seconds=args.interval,
        account_fraction=args.account_fraction,
        min_trade_amount=args.min_trade_amount,
        enable_probe_mode=not args.disable_probe_mode,
    )

    # Setup signal handlers
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, requesting stop...")
        trading_loop.request_stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run the trading loop
    try:
        trading_loop.run(once=args.once)
        logger.info("Trading loop completed successfully")
        return 0
    except Exception as exc:
        logger.exception(f"Trading loop failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
