#!/usr/bin/env python3
"""SUI trading bot using btcmarketsbot-style trained model."""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import torch

from src.price_guard import enforce_gap
from src.process_utils import enforce_min_spread
from src.binan import binance_wrapper
from src.binance_hourly_csv_utils import append_hourly_binance_bars

from binanceneural.binance_watchers import WatcherPlan, spawn_watcher, cancel_entry_watchers
from binanceneural.execution import compute_order_quantities, get_free_balances, resolve_symbol_rules
from binanceneural.inference import generate_latest_action

from binancechronossolexperiment.inference import load_policy_checkpoint
from binancechronossolexperiment.data import ChronosSolDataModule, SplitConfig

TRADE_SYMBOL = "SUIUSDT"  # Trading pair on Binance
DATA_SYMBOL = "SUIUSDT"   # Data/model symbol
DATA_ROOT = Path("trainingdatahourlybinance")
FORECAST_CACHE = Path("binancechronossolexperiment/forecast_cache_sui_10bp")
DEFAULT_CHECKPOINT = Path("bitbankstylelongsuitrain/checkpoints/bitbank_simple/policy_checkpoint.pt")
STATE_DIR = Path("strategy_state")


@dataclass
class TradingPlan:
    symbol: str
    buy_price: float
    sell_price: float
    buy_amount: float
    sell_amount: float
    current_price: float
    timestamp: datetime


def _refresh_price_csv() -> None:
    """Append latest hourly candles from Binance."""
    try:
        from binance_data_wrapper import fetch_binance_hourly_bars
    except ImportError:
        return
    csv_path = DATA_ROOT / f"{DATA_SYMBOL}.csv"
    try:
        result = append_hourly_binance_bars(
            csv_path,
            fetch_symbol=DATA_SYMBOL,
            csv_symbol=DATA_SYMBOL,
            fetcher=fetch_binance_hourly_bars,
        )
    except Exception as e:
        print(f"[data-refresh] error: {e}")
        return
    if result.get("status") in {"updated", "synced"}:
        print(
            f"[data-refresh] {DATA_SYMBOL}: +{result.get('rows_added', 0)} rows, "
            f"last={result.get('end')}"
        )


def _refresh_forecast_cache() -> None:
    """Refresh Chronos forecasts for latest data (optional, uses cache if fails)."""
    try:
        from binancechronossolexperiment.forecasts import build_forecast_bundle
        build_forecast_bundle(
            symbol=DATA_SYMBOL,
            data_root=DATA_ROOT,
            cache_root=FORECAST_CACHE,
            horizons=(1, 4, 24),
            context_hours=512,
            quantile_levels=(0.1, 0.5, 0.9),
            batch_size=32,
            model_id="amazon/chronos-t5-small",
            cache_only=False,
        )
        print("[forecast-refresh] updated cache")
    except Exception:
        pass  # Use existing cache


def _load_data_module(sequence_length: int = 72) -> ChronosSolDataModule:
    """Load data module with forecast cache."""
    return ChronosSolDataModule(
        symbol=DATA_SYMBOL,
        data_root=DATA_ROOT,
        forecast_cache_root=FORECAST_CACHE,
        forecast_horizons=(1, 4, 24),
        context_hours=512,
        quantile_levels=(0.1, 0.5, 0.9),
        batch_size=32,
        model_id="amazon/chronos-t5-small",
        sequence_length=sequence_length,
        split_config=SplitConfig(val_days=7, test_days=7),
        max_history_days=365,
        cache_only=True,
    )


def _generate_action(
    checkpoint_path: Path,
    sequence_length: int = 72,
    horizon: int = 1,
) -> dict:
    """Generate trading action from model."""
    model, normalizer, feature_columns, cfg = load_policy_checkpoint(str(checkpoint_path))
    dm = _load_data_module(sequence_length)
    action = generate_latest_action(
        model=model,
        frame=dm.full_frame,
        feature_columns=feature_columns,
        normalizer=normalizer,
        sequence_length=sequence_length,
        horizon=horizon,
        require_gpu=True,
    )
    action["current_price"] = float(dm.full_frame["close"].iloc[-1])
    return action


def _build_plan(action: dict, intensity_scale: float = 1.0) -> TradingPlan:
    """Build trading plan from action."""
    buy_amount = max(0.0, min(100.0, action["buy_amount"] * intensity_scale))
    sell_amount = max(0.0, min(100.0, action["sell_amount"] * intensity_scale))
    return TradingPlan(
        symbol=TRADE_SYMBOL,
        buy_price=float(action["buy_price"]),
        sell_price=float(action["sell_price"]),
        buy_amount=buy_amount,
        sell_amount=sell_amount,
        current_price=float(action.get("current_price", action["buy_price"])),
        timestamp=action["timestamp"],
    )


def _place_orders(
    plan: TradingPlan,
    *,
    poll_seconds: int,
    expiry_minutes: int,
    price_tolerance: float,
    dry_run: bool,
) -> None:
    """Place buy and sell limit orders."""
    try:
        quote_free, base_free = get_free_balances(TRADE_SYMBOL)
    except Exception as exc:
        print(f"Failed to get balances: {exc}")
        return

    try:
        rules = resolve_symbol_rules(TRADE_SYMBOL)
    except Exception as exc:
        print(f"Failed to get symbol rules: {exc}")
        return

    sizing = compute_order_quantities(
        symbol=TRADE_SYMBOL,
        buy_amount=plan.buy_amount,
        sell_amount=plan.sell_amount,
        buy_price=plan.buy_price,
        sell_price=plan.sell_price,
        quote_free=quote_free,
        base_free=base_free,
        rules=rules,
    )

    print(f"[sizing] buy_qty={sizing.buy_qty:.6f} sell_qty={sizing.sell_qty:.6f} "
          f"buy_notional=${sizing.buy_notional:.2f} sell_notional=${sizing.sell_notional:.2f}")

    if dry_run:
        print(f"[dry-run] would place buy@{plan.buy_price:.4f} sell@{plan.sell_price:.4f}")
        return

    cancel_entry_watchers()

    if sizing.buy_qty > 0 and sizing.buy_notional >= rules.min_notional:
        buy_plan = WatcherPlan(
            symbol=TRADE_SYMBOL,
            side="buy",
            mode="entry",
            limit_price=plan.buy_price,
            target_qty=sizing.buy_qty,
            poll_seconds=poll_seconds,
            expiry_minutes=expiry_minutes,
            price_tolerance=price_tolerance,
        )
        spawn_watcher(buy_plan)
        print(f"[order] spawned buy watcher: {sizing.buy_qty:.6f} @ {plan.buy_price:.4f}")

    if sizing.sell_qty > 0 and sizing.sell_notional >= rules.min_notional:
        sell_plan = WatcherPlan(
            symbol=TRADE_SYMBOL,
            side="sell",
            mode="entry",
            limit_price=plan.sell_price,
            target_qty=sizing.sell_qty,
            poll_seconds=poll_seconds,
            expiry_minutes=expiry_minutes,
            price_tolerance=price_tolerance,
        )
        spawn_watcher(sell_plan)
        print(f"[order] spawned sell watcher: {sizing.sell_qty:.6f} @ {plan.sell_price:.4f}")


def _log_metrics(plan: TradingPlan) -> None:
    """Log trading metrics."""
    log_path = STATE_DIR / "sui_trading_metrics.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    spread_pct = (plan.sell_price - plan.buy_price) / plan.current_price * 100

    row = {
        "timestamp": now.isoformat(),
        "current_price": plan.current_price,
        "buy_price": plan.buy_price,
        "sell_price": plan.sell_price,
        "spread_pct": spread_pct,
        "buy_amount": plan.buy_amount,
        "sell_amount": plan.sell_amount,
    }

    df = pd.DataFrame([row])
    df.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)

    print(f"[metrics] price={plan.current_price:.4f} buy={plan.buy_price:.4f} "
          f"sell={plan.sell_price:.4f} spread={spread_pct:.2f}%")


def _run_cycle(
    checkpoint_path: Path,
    *,
    sequence_length: int,
    horizon: int,
    intensity_scale: float,
    price_offset_pct: float,
    min_gap_pct: float,
    poll_seconds: int,
    expiry_minutes: int,
    price_tolerance: float,
    refresh_data: bool,
    dry_run: bool,
) -> None:
    """Run one trading cycle."""
    print(f"\n{'='*60}")
    print(f"[cycle] {datetime.now(timezone.utc).isoformat()}")

    if refresh_data:
        _refresh_price_csv()
        _refresh_forecast_cache()

    try:
        action = _generate_action(checkpoint_path, sequence_length, horizon)
    except Exception as exc:
        print(f"[error] failed to generate action: {exc}")
        return

    plan = _build_plan(action, intensity_scale)

    buy_price = plan.buy_price * (1.0 - price_offset_pct)
    sell_price = plan.sell_price * (1.0 + price_offset_pct)

    buy_price, sell_price = enforce_min_spread(buy_price, sell_price, min_spread_pct=min_gap_pct)
    buy_price, sell_price = enforce_gap(TRADE_SYMBOL, buy_price, sell_price, min_gap_pct=min_gap_pct)

    plan = TradingPlan(
        symbol=plan.symbol,
        buy_price=buy_price,
        sell_price=sell_price,
        buy_amount=plan.buy_amount,
        sell_amount=plan.sell_amount,
        current_price=plan.current_price,
        timestamp=plan.timestamp,
    )

    if buy_price <= 0 or sell_price <= 0 or buy_price >= sell_price:
        print(f"[error] invalid levels: buy={buy_price:.4f} sell={sell_price:.4f}")
        return

    _log_metrics(plan)
    _place_orders(
        plan,
        poll_seconds=poll_seconds,
        expiry_minutes=expiry_minutes,
        price_tolerance=price_tolerance,
        dry_run=dry_run,
    )


def main():
    parser = argparse.ArgumentParser(description="SUI trading bot")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sequence-length", type=int, default=72)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--intensity-scale", type=float, default=6.0)
    parser.add_argument("--price-offset-pct", type=float, default=0.0005)
    parser.add_argument("--min-gap-pct", type=float, default=0.0003)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--expiry-minutes", type=int, default=90)
    parser.add_argument("--price-tolerance", type=float, default=0.0008)
    parser.add_argument("--cycle-minutes", type=int, default=5)
    parser.add_argument("--refresh-data", action="store_true", default=True)
    parser.add_argument("--no-refresh-data", dest="refresh_data", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--once", action="store_true", help="Run once then exit")
    args = parser.parse_args()

    print(f"SUI Trading Bot")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Cycle: {args.cycle_minutes}min, Poll: {args.poll_seconds}s, Expiry: {args.expiry_minutes}min")
    print(f"Intensity: {args.intensity_scale}x, Offset: {args.price_offset_pct*100:.2f}%")
    print(f"Dry run: {args.dry_run}")

    while True:
        try:
            _run_cycle(
                args.checkpoint,
                sequence_length=args.sequence_length,
                horizon=args.horizon,
                intensity_scale=args.intensity_scale,
                price_offset_pct=args.price_offset_pct,
                min_gap_pct=args.min_gap_pct,
                poll_seconds=args.poll_seconds,
                expiry_minutes=args.expiry_minutes,
                price_tolerance=args.price_tolerance,
                refresh_data=args.refresh_data,
                dry_run=args.dry_run,
            )
        except Exception as exc:
            print(f"[error] cycle failed: {exc}")
            import traceback
            traceback.print_exc()

        if args.once:
            break

        sleep_seconds = args.cycle_minutes * 60
        print(f"[sleep] {sleep_seconds}s until next cycle")
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()
