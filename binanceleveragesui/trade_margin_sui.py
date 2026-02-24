#!/usr/bin/env python3
"""Binance cross-margin trading bot (generic symbol).

Trades any symbol with configurable leverage. Auto-borrows USDT on entry
(MARGIN_BUY), auto-repays on exit (AUTO_REPAY). Repays outstanding
loans when flat.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.binan import binance_wrapper
from src.binan.binance_margin import (
    cancel_all_margin_orders,
    get_margin_asset_balance,
    get_margin_borrowed_balance,
    get_margin_free_balance,
    margin_repay_all,
)
from src.price_guard import enforce_gap
from src.process_utils import enforce_min_spread

from binanceneural.binance_watchers import WatcherPlan, spawn_watcher
from binanceneural.execution import resolve_symbol_rules, quantize_qty
from binanceneural.inference import generate_latest_action
from binanceneural.trade_binance_hourly import _ensure_valid_levels

from binancechronossolexperiment.data import ChronosSolDataModule, SplitConfig
from binancechronossolexperiment.inference import load_policy_checkpoint

SYMBOL = "SUIUSDT"  # overridden by --symbol
BASE_ASSET = "SUI"  # derived from SYMBOL
TAG = "margin"  # log prefix
STATE_FILE = Path("strategy_state/margin_sui_state.json")
MIN_POSITION_NOTIONAL = 5.0


@dataclass
class MarginState:
    in_position: bool = False
    open_ts: Optional[str] = None
    open_price: float = 0.0

    def hours_held(self) -> float:
        if not self.open_ts:
            return 0.0
        opened = datetime.fromisoformat(self.open_ts)
        if opened.tzinfo is None:
            opened = opened.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - opened).total_seconds() / 3600.0

    def save(self, path: Path = STATE_FILE):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "in_position": self.in_position,
            "open_ts": self.open_ts,
            "open_price": self.open_price,
        }))

    @classmethod
    def load(cls, path: Path = STATE_FILE) -> MarginState:
        if not path.exists():
            return cls()
        try:
            d = json.loads(path.read_text())
            return cls(
                in_position=d.get("in_position", False),
                open_ts=d.get("open_ts"),
                open_price=d.get("open_price", 0.0),
            )
        except Exception:
            return cls()


def _refresh_price_csv(data_symbol: str, data_root: Path):
    try:
        from binance_data_wrapper import fetch_binance_hourly_bars
    except ImportError:
        return
    csv_path = data_root / f"{data_symbol.upper()}.csv"
    if not csv_path.exists():
        return
    existing = pd.read_csv(csv_path)
    existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)
    last_ts = existing["timestamp"].max()
    # Use SYMBOL (Binance trading pair) for API, data_symbol for file
    try:
        new = fetch_binance_hourly_bars(SYMBOL, start=last_ts, end=datetime.now(timezone.utc))
    except Exception:
        return
    if new is None or len(new) == 0:
        return
    new = new.reset_index()
    new["symbol"] = data_symbol.upper()
    new["timestamp"] = pd.to_datetime(new["timestamp"], utc=True)
    new = new[new["timestamp"] > last_ts]
    if len(new) == 0:
        return
    combined = pd.concat([existing, new], ignore_index=True)
    combined = combined.drop_duplicates(subset="timestamp", keep="last")
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    combined.to_csv(csv_path, index=False)
    print(f"[{TAG}] data: +{len(new)} rows")


def _load_live_frame(
    symbol: str,
    data_root: Path,
    forecast_cache: Path,
    forecast_horizons: tuple,
    sequence_length: int,
) -> pd.DataFrame:
    dm = ChronosSolDataModule(
        symbol=symbol,
        data_root=data_root,
        forecast_cache_root=forecast_cache,
        forecast_horizons=forecast_horizons,
        context_hours=512,
        quantile_levels=(0.1, 0.5, 0.9),
        batch_size=32,
        model_id="amazon/chronos-t5-small",
        sequence_length=sequence_length,
        split_config=SplitConfig(val_days=1, test_days=1),
        cache_only=True,
    )
    return dm.full_frame


def _get_margin_equity() -> dict:
    """Returns margin account balances and total equity in USDT."""
    usdt_entry = get_margin_asset_balance("USDT")
    asset_entry = get_margin_asset_balance(BASE_ASSET)
    usdt_free = float(usdt_entry.get("free", 0)) if usdt_entry else 0.0
    usdt_locked = float(usdt_entry.get("locked", 0)) if usdt_entry else 0.0
    usdt_borrowed = float(usdt_entry.get("borrowed", 0)) if usdt_entry else 0.0
    usdt_net = float(usdt_entry.get("netAsset", 0)) if usdt_entry else 0.0
    asset_free = float(asset_entry.get("free", 0)) if asset_entry else 0.0
    asset_locked = float(asset_entry.get("locked", 0)) if asset_entry else 0.0
    asset_borrowed = float(asset_entry.get("borrowed", 0)) if asset_entry else 0.0
    asset_net = float(asset_entry.get("netAsset", 0)) if asset_entry else 0.0
    asset_total = asset_free + asset_locked
    try:
        market_price = float(binance_wrapper.get_symbol_price(SYMBOL))
    except Exception:
        market_price = 0.0
    asset_value = asset_total * market_price
    equity = usdt_net + asset_net * market_price
    return {
        "usdt_free": usdt_free, "usdt_locked": usdt_locked,
        "usdt_borrowed": usdt_borrowed, "usdt_net": usdt_net,
        "asset_free": asset_free, "asset_locked": asset_locked,
        "asset_borrowed": asset_borrowed, "asset_total": asset_total,
        "asset_value": asset_value, "market_price": market_price,
        "equity": equity,
    }


def _repay_outstanding(asset: str = "USDT"):
    borrowed = get_margin_borrowed_balance(asset)
    if borrowed <= 0.01:
        return
    cancel_all_margin_orders(SYMBOL)
    try:
        margin_repay_all(asset)
        print(f"[{TAG}] repaid {borrowed:.4f} {asset}")
    except Exception as exc:
        free = get_margin_free_balance(asset)
        if free > 0.01:
            from src.binan.binance_margin import margin_repay
            try:
                margin_repay(asset, free * 0.999)
                print(f"[{TAG}] partial repay {free:.4f} {asset}")
            except Exception as exc2:
                print(f"[{TAG}] repay failed: {exc2}")
        else:
            print(f"[{TAG}] repay failed: {exc}")


def _refresh_signal(
    model,
    normalizer,
    feature_columns,
    *,
    horizon: int,
    sequence_length: int,
    intensity_scale: float,
    data_root: Path,
    forecast_cache: Path,
    forecast_horizons: tuple,
    data_symbol: str,
) -> dict:
    _refresh_price_csv(data_symbol, data_root)
    frame = _load_live_frame(data_symbol, data_root, forecast_cache, forecast_horizons, sequence_length)
    action = generate_latest_action(
        model=model, frame=frame, feature_columns=feature_columns,
        normalizer=normalizer, sequence_length=sequence_length,
        horizon=horizon, require_gpu=True,
    )
    action["symbol"] = SYMBOL
    last_row = frame.iloc[-1]
    for col in [f"predicted_high_p50_h{horizon}", f"predicted_low_p50_h{horizon}", f"predicted_close_p50_h{horizon}"]:
        if col in frame.columns:
            action[col] = float(last_row[col])

    buy_price = float(action.get("buy_price", 0))
    sell_price = float(action.get("sell_price", 0))
    buy_amount = max(0.0, min(100.0, float(action.get("buy_amount", 0)) * intensity_scale))
    sell_amount = max(0.0, min(100.0, float(action.get("sell_amount", 0)) * intensity_scale))

    MIN_SPREAD_BP = 0.002
    if buy_price > 0 and sell_price > 0 and sell_price <= buy_price * (1 + MIN_SPREAD_BP):
        mid = (buy_price + sell_price) / 2
        buy_price = mid * (1 - MIN_SPREAD_BP / 2)
        sell_price = mid * (1 + MIN_SPREAD_BP / 2)
        print(f"[{TAG}] WARNING: sell<=buy, forced 20bp spread buy={buy_price:.4f} sell={sell_price:.4f}")

    print(f"[{TAG}] signal buy={buy_price:.4f}({buy_amount:.1f}%) sell={sell_price:.4f}({sell_amount:.1f}%)")
    return {"buy_price": buy_price, "sell_price": sell_price,
            "buy_amount": buy_amount, "sell_amount": sell_amount,
            "ts": time.monotonic()}


def _fast_check(
    signal: dict,
    rules,
    *,
    max_leverage: float,
    min_gap_pct: float,
    max_hold_hours: Optional[int],
    poll_seconds: int,
    expiry_minutes: int,
    price_tolerance: float,
    state_path: Path,
    dry_run: bool,
):
    buy_price = signal["buy_price"]
    sell_price = signal["sell_price"]
    buy_amount = signal["buy_amount"]
    sell_amount = signal["sell_amount"]

    bal = _get_margin_equity()
    equity = bal["equity"]
    market_price = bal["market_price"]
    asset_total = bal["asset_total"]
    asset_value = bal["asset_value"]

    state = MarginState.load(state_path)

    if asset_value >= MIN_POSITION_NOTIONAL:
        if not state.in_position:
            print(f"[{TAG}] detected position: {asset_total:.4f} {BASE_ASSET} (${asset_value:.2f})")
            state.in_position = True
            if not state.open_ts:
                state.open_ts = datetime.now(timezone.utc).isoformat()
            if state.open_price <= 0:
                state.open_price = market_price
            state.save(state_path)
    else:
        if state.in_position:
            print(f"[{TAG}] position closed")
            state = MarginState()
            state.save(state_path)
        _repay_outstanding("USDT")

    current_leverage = asset_value / equity if equity > 0 else 0.0
    print(
        f"[{TAG}] eq=${equity:.2f} usdt_free=${bal['usdt_free']:.2f} usdt_borrow=${bal['usdt_borrowed']:.2f} "
        f"{BASE_ASSET}={asset_total:.4f} pos=${asset_value:.2f} lev={current_leverage:.1f}x price={market_price:.4f}"
    )

    if state.in_position:
        _handle_exit(
            state, asset_total, market_price, rules,
            sell_price=sell_price, sell_amount=sell_amount,
            buy_price=buy_price,
            min_gap_pct=min_gap_pct, max_hold_hours=max_hold_hours,
            poll_seconds=poll_seconds, expiry_minutes=expiry_minutes,
            price_tolerance=price_tolerance, dry_run=dry_run,
            state_path=state_path,
        )
        if current_leverage < max_leverage and buy_amount > 0 and buy_price > 0:
            remaining_power = equity * max_leverage - asset_value
            if remaining_power > rules.min_notional:
                _handle_add(
                    remaining_power, rules,
                    buy_price=buy_price, sell_price=sell_price,
                    min_gap_pct=min_gap_pct,
                    poll_seconds=poll_seconds, expiry_minutes=expiry_minutes,
                    price_tolerance=price_tolerance, dry_run=dry_run,
                )
    else:
        _handle_entry(
            equity, rules,
            max_leverage=max_leverage,
            buy_price=buy_price, sell_price=sell_price, buy_amount=buy_amount,
            min_gap_pct=min_gap_pct,
            poll_seconds=poll_seconds, expiry_minutes=expiry_minutes,
            price_tolerance=price_tolerance, dry_run=dry_run,
        )


def _handle_exit(
    state: MarginState,
    asset_total: float,
    market_price: float,
    rules,
    *,
    sell_price: float,
    sell_amount: float,
    buy_price: float,
    min_gap_pct: float,
    max_hold_hours: Optional[int],
    poll_seconds: int,
    expiry_minutes: int,
    price_tolerance: float,
    dry_run: bool,
    state_path: Path,
):
    hours = state.hours_held()
    force_close = max_hold_hours is not None and hours >= max_hold_hours

    if force_close:
        print(f"[{TAG}] FORCE CLOSE after {hours:.1f}h (max={max_hold_hours}h)")
        sell_p = market_price * 0.999
        validated = _ensure_valid_levels(SYMBOL, sell_p * 0.99, sell_p, min_gap_pct=min_gap_pct, rules=rules)
        if validated is None:
            print(f"[{TAG}] invalid force-close price")
            return
        _, sell_p = validated
        sell_qty = quantize_qty(asset_total, step_size=rules.step_size)
        if sell_qty > 0:
            spawn_watcher(WatcherPlan(
                symbol=SYMBOL, side="sell", mode="exit",
                limit_price=sell_p, target_qty=sell_qty,
                expiry_minutes=expiry_minutes, poll_seconds=poll_seconds,
                price_tolerance=price_tolerance * 3, dry_run=dry_run,
                margin=True, side_effect_type="AUTO_REPAY",
            ))
            print(f"[{TAG}] force-close sell={sell_p:.4f} qty={sell_qty:.4f}")
        return

    if sell_amount <= 0 or sell_price <= 0:
        print(f"[{TAG}] holding ({hours:.1f}h), no sell signal")
        return

    bp = buy_price if buy_price > 0 else sell_price * 0.99
    bp, sp = enforce_min_spread(bp, sell_price, min_spread_pct=min_gap_pct)
    bp, sp = enforce_gap(SYMBOL, bp, sp, min_gap_pct=min_gap_pct)
    validated = _ensure_valid_levels(SYMBOL, bp, sp, min_gap_pct=min_gap_pct, rules=rules)
    if validated is None:
        print(f"[{TAG}] invalid exit prices")
        return
    _, sp = validated

    sell_intensity = max(0.0, min(1.0, sell_amount / 100.0))
    sell_qty = quantize_qty(asset_total * sell_intensity, step_size=rules.step_size)

    if sell_qty <= 0:
        print(f"[{TAG}] holding ({hours:.1f}h), sell qty too small")
        return
    if rules.min_notional and sell_qty * sp < rules.min_notional:
        print(f"[{TAG}] holding ({hours:.1f}h), below min notional")
        return

    spawn_watcher(WatcherPlan(
        symbol=SYMBOL, side="sell", mode="exit",
        limit_price=sp, target_qty=sell_qty,
        expiry_minutes=expiry_minutes, poll_seconds=poll_seconds,
        price_tolerance=price_tolerance, dry_run=dry_run,
        margin=True, side_effect_type="AUTO_REPAY",
    ))
    print(f"[{TAG}] EXIT sell={sp:.4f} qty={sell_qty:.4f} amt={sell_amount:.1f}%")


def _handle_add(
    remaining_power: float,
    rules,
    *,
    buy_price: float,
    sell_price: float,
    min_gap_pct: float,
    poll_seconds: int,
    expiry_minutes: int,
    price_tolerance: float,
    dry_run: bool,
):
    bp, sp = enforce_min_spread(buy_price, sell_price, min_spread_pct=min_gap_pct)
    bp, sp = enforce_gap(SYMBOL, bp, sp, min_gap_pct=min_gap_pct)
    if bp <= 0 or sp <= 0 or bp >= sp:
        return
    validated = _ensure_valid_levels(SYMBOL, bp, sp, min_gap_pct=min_gap_pct, rules=rules)
    if validated is None:
        return
    bp, _ = validated
    buy_qty = quantize_qty(remaining_power / bp, step_size=rules.step_size)
    if buy_qty <= 0 or (rules.min_notional and buy_qty * bp < rules.min_notional):
        return
    spawn_watcher(WatcherPlan(
        symbol=SYMBOL, side="buy", mode="entry",
        limit_price=bp, target_qty=buy_qty,
        expiry_minutes=expiry_minutes, poll_seconds=poll_seconds,
        price_tolerance=price_tolerance, dry_run=dry_run,
        margin=True, side_effect_type="MARGIN_BUY",
    ))
    notional = buy_qty * bp
    print(f"[{TAG}] ADD buy={bp:.4f} qty={buy_qty:.2f} notional=${notional:.2f} (remaining leverage)")


def _handle_entry(
    equity: float,
    rules,
    *,
    max_leverage: float,
    buy_price: float,
    sell_price: float,
    buy_amount: float,
    min_gap_pct: float,
    poll_seconds: int,
    expiry_minutes: int,
    price_tolerance: float,
    dry_run: bool,
):
    if buy_amount <= 0 or buy_price <= 0:
        print(f"[{TAG}] flat, no buy signal")
        return
    if equity <= 0:
        print(f"[{TAG}] no equity in margin account")
        return

    bp, sp = enforce_min_spread(buy_price, sell_price, min_spread_pct=min_gap_pct)
    bp, sp = enforce_gap(SYMBOL, bp, sp, min_gap_pct=min_gap_pct)
    if bp <= 0 or sp <= 0 or bp >= sp:
        print(f"[{TAG}] invalid entry prices")
        return

    validated = _ensure_valid_levels(SYMBOL, bp, sp, min_gap_pct=min_gap_pct, rules=rules)
    if validated is None:
        print(f"[{TAG}] invalid entry levels")
        return
    bp, _ = validated

    buying_power = equity * max_leverage
    buy_qty = quantize_qty(buying_power / bp, step_size=rules.step_size)

    if buy_qty <= 0:
        print(f"[{TAG}] buy qty too small")
        return
    if rules.min_notional and buy_qty * bp < rules.min_notional:
        print(f"[{TAG}] below min notional")
        return

    spawn_watcher(WatcherPlan(
        symbol=SYMBOL, side="buy", mode="entry",
        limit_price=bp, target_qty=buy_qty,
        expiry_minutes=expiry_minutes, poll_seconds=poll_seconds,
        price_tolerance=price_tolerance, dry_run=dry_run,
        margin=True, side_effect_type="MARGIN_BUY",
    ))
    notional = buy_qty * bp
    print(f"[{TAG}] ENTER buy={bp:.4f} qty={buy_qty:.2f} notional=${notional:.2f} lev={max_leverage}x")


def main():
    global SYMBOL, BASE_ASSET, TAG, STATE_FILE
    parser = argparse.ArgumentParser(description="Binance margin trading bot")
    parser.add_argument("--symbol", default="SUIUSDT", help="Trading pair e.g. DOGEUSDT")
    parser.add_argument("--data-symbol", default=None, help="Symbol for data/cache lookup (default: same as --symbol)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-leverage", type=float, default=4.0)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=72)
    parser.add_argument("--intensity-scale", type=float, default=5.0)
    parser.add_argument("--min-gap-pct", type=float, default=0.0003)
    parser.add_argument("--max-hold-hours", type=int, default=6)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--expiry-minutes", type=int, default=90)
    parser.add_argument("--price-tolerance", type=float, default=0.0008)
    parser.add_argument("--data-root", default="trainingdatahourlybinance")
    parser.add_argument("--forecast-cache", default="binancechronossolexperiment/forecast_cache_sui_10bp")
    parser.add_argument("--forecast-horizons", default="1,4,24")
    parser.add_argument("--state-path", default=str(STATE_FILE))
    parser.add_argument("--cycle-minutes", type=int, default=5)
    parser.add_argument("--fast-check-seconds", type=int, default=30)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    SYMBOL = args.symbol.upper()
    DATA_SYMBOL = (args.data_symbol or SYMBOL).upper()
    BASE_ASSET = SYMBOL.replace("USDT", "").replace("USD", "")
    TAG = f"margin-{BASE_ASSET.lower()}"
    if args.state_path == str(Path("strategy_state/margin_sui_state.json")):
        STATE_FILE = Path(f"strategy_state/margin_{BASE_ASSET.lower()}_state.json")
        state_path = STATE_FILE
    else:
        state_path = Path(args.state_path)

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    forecast_horizons = tuple(int(h) for h in args.forecast_horizons.split(","))
    data_root = Path(args.data_root)
    forecast_cache = Path(args.forecast_cache)
    signal_interval = args.cycle_minutes * 60
    fast_interval = args.fast_check_seconds

    print(f"[{TAG}] loading model from {checkpoint_path}")
    model, normalizer, feature_columns, cfg = load_policy_checkpoint(str(checkpoint_path))
    rules = resolve_symbol_rules(SYMBOL)
    print(f"[{TAG}] {SYMBOL} model loaded, signal every {signal_interval}s, fast check every {fast_interval}s")

    signal: Optional[dict] = None
    last_signal_time = 0.0

    while True:
        now = time.monotonic()

        if now - last_signal_time >= signal_interval or signal is None:
            try:
                signal = _refresh_signal(
                    model, normalizer, feature_columns,
                    horizon=args.horizon,
                    sequence_length=args.sequence_length,
                    intensity_scale=args.intensity_scale,
                    data_root=data_root,
                    forecast_cache=forecast_cache,
                    forecast_horizons=forecast_horizons,
                    data_symbol=DATA_SYMBOL,
                )
                last_signal_time = now
            except Exception as exc:
                print(f"[{TAG}] signal error: {exc}")
                import traceback; traceback.print_exc()
                if signal is None:
                    time.sleep(fast_interval)
                    continue

        signal_age = now - last_signal_time
        if signal_age > signal_interval * 2:
            print(f"[{TAG}] WARNING: signal is {signal_age:.0f}s old (stale)")

        try:
            _fast_check(
                signal, rules,
                max_leverage=args.max_leverage,
                min_gap_pct=args.min_gap_pct,
                max_hold_hours=args.max_hold_hours,
                poll_seconds=args.poll_seconds,
                expiry_minutes=args.expiry_minutes,
                price_tolerance=args.price_tolerance,
                state_path=state_path,
                dry_run=args.dry_run,
            )
        except Exception as exc:
            print(f"[{TAG}] fast_check error: {exc}")
            import traceback; traceback.print_exc()

        if args.once:
            break
        print(f"[{TAG}] sleeping {fast_interval}s...")
        time.sleep(fast_interval)


if __name__ == "__main__":
    main()
