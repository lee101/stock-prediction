from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, Sequence

import pandas as pd
from loguru import logger

from src.chronos2_params import resolve_chronos2_params
from src.models.chronos2_wrapper import Chronos2OHLCWrapper
from src.price_guard import enforce_gap
from src.process_utils import enforce_min_spread
from src.binan import binance_wrapper
from src.tradinglib.direction_filter import compute_predicted_close_return, should_trade_predicted_close_return

from .binance_watchers import WatcherPlan, spawn_watcher, stop_existing_watcher
from .execution import (
    get_free_balances,
    quantize_price,
    quantize_qty,
    resolve_binance_symbol,
    resolve_symbol_rules,
)


@dataclass(frozen=True)
class DailyLevels:
    day_start: pd.Timestamp
    issued_at: pd.Timestamp
    buy_price: float
    sell_price: float
    predicted_close_p50: Optional[float] = None
    prev_close: Optional[float] = None
    predicted_close_return: Optional[float] = None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _coerce_ts(value: object) -> pd.Timestamp:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid timestamp {value!r}")
    return pd.Timestamp(ts)


def _load_daily_history(path: Path, *, symbol: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    if "timestamp" not in df.columns:
        raise KeyError(f"{path} missing timestamp column")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise KeyError(f"{path} missing required column {col!r}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.upper()
        df = df[df["symbol"] == symbol.upper()].reset_index(drop=True)
    return df


def _forecast_today_levels(
    *,
    symbol: str,
    daily_root: Path,
    buy_quantile: float,
    sell_quantile: float,
    context_length: Optional[int],
    batch_size: Optional[int],
    model_id: Optional[str],
    device_map: Optional[str],
) -> DailyLevels:
    symbol = symbol.upper()
    path = Path(daily_root) / f"{symbol}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing daily dataset {path}")

    daily = _load_daily_history(path, symbol=symbol)
    if daily.empty:
        raise RuntimeError(f"Daily dataset {path} is empty after filtering.")

    now = _utc_now()
    day_start = pd.Timestamp(now).floor("D")
    # Ensure we only use fully-closed daily bars: drop today's bar if present.
    if pd.Timestamp(daily["timestamp"].iloc[-1]).floor("D") >= day_start:
        daily = daily.iloc[:-1].reset_index(drop=True)
    if len(daily) < 32:
        raise RuntimeError(f"Insufficient daily history for {symbol}: {len(daily)} rows.")

    params = resolve_chronos2_params(symbol, frequency="daily", default_prediction_length=1)
    resolved_model_id = str(model_id or params.get("model_id") or "amazon/chronos-2")
    resolved_device_map = device_map or params.get("device_map") or "cuda"
    resolved_ctx = int(context_length or params.get("context_length") or 512)
    resolved_bs = int(batch_size or params.get("batch_size") or 32)
    predict_kwargs = params.get("predict_kwargs")
    predict_kwargs = dict(predict_kwargs) if isinstance(predict_kwargs, dict) else None

    quantiles = sorted({float(buy_quantile), float(sell_quantile), 0.5})
    wrapper = Chronos2OHLCWrapper.from_pretrained(
        model_id=resolved_model_id,
        device_map=resolved_device_map,
        default_context_length=resolved_ctx,
        default_batch_size=resolved_bs,
        quantile_levels=tuple(quantiles),
        preaugmentation_dirs=[Path("preaugstrategies") / "chronos2"],
    )

    context = daily.iloc[-min(len(daily), resolved_ctx) :].copy()
    issued_at = _coerce_ts(context["timestamp"].max())

    batch = wrapper.predict_ohlc_batch(
        [context[["timestamp", "open", "high", "low", "close"]]],
        symbols=[symbol],
        prediction_length=1,
        context_length=resolved_ctx,
        quantile_levels=tuple(quantiles),
        batch_size=resolved_bs,
        predict_kwargs=predict_kwargs,
    )[0]

    quantile_frames = batch.quantile_frames
    if not quantile_frames:
        raise RuntimeError("Chronos2 returned no quantile frames.")

    # Future timestamp should correspond to today (UTC day start) when context ends at yesterday.
    buy_frame = quantile_frames.get(float(buy_quantile))
    sell_frame = quantile_frames.get(float(sell_quantile))
    p50_frame = quantile_frames.get(0.5)
    if buy_frame is None or sell_frame is None:
        raise RuntimeError("Requested quantiles missing from Chronos2 output.")
    # Defensive: pick the first row if alignment is off.
    buy_row = buy_frame.iloc[0]
    sell_row = sell_frame.iloc[0]
    buy_price = float(buy_row["low"])
    sell_price = float(sell_row["high"])

    predicted_close_p50: Optional[float] = None
    if p50_frame is not None and not p50_frame.empty and "close" in p50_frame.columns:
        try:
            predicted_close_p50 = float(p50_frame.iloc[0]["close"])
        except (TypeError, ValueError):
            predicted_close_p50 = None

    prev_close: Optional[float] = None
    try:
        prev_close = float(context["close"].iloc[-1])
    except Exception:
        prev_close = None
    predicted_close_return = compute_predicted_close_return(predicted_close_p50, prev_close)

    return DailyLevels(
        day_start=day_start,
        issued_at=issued_at,
        buy_price=buy_price,
        sell_price=sell_price,
        predicted_close_p50=predicted_close_p50,
        prev_close=prev_close,
        predicted_close_return=predicted_close_return,
    )


def _minutes_until_day_end(now: datetime) -> int:
    cur = now.astimezone(timezone.utc)
    day_start = cur.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = day_start + timedelta(days=1)
    remaining = (day_end - cur).total_seconds() / 60.0
    return max(1, int(remaining))


def _load_watcher_status(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _wait_for_watcher(path: Path, *, poll_seconds: int) -> dict:
    poll = max(1, int(poll_seconds))
    while True:
        status = _load_watcher_status(path)
        if not status.get("active"):
            return status
        time.sleep(poll)


def _extract_fill_qty(status: dict) -> float:
    try:
        qty = float(status.get("fill_qty", 0.0))
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, qty)


def _extract_fill_price(status: dict) -> Optional[float]:
    try:
        price = float(status.get("fill_price"))
    except (TypeError, ValueError):
        return None
    if price <= 0:
        return None
    return price


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Trade Binance spot using daily Chronos2 buy/sell levels (grid-like cycles).")
    parser.add_argument("--symbol", default="SOLUSD", help="Internal USD-quoted symbol (default: SOLUSD).")
    parser.add_argument("--daily-root", default="trainingdatadailybinance", help="Root with daily CSVs (default: trainingdatadailybinance).")
    parser.add_argument("--buy-quantile", type=float, default=0.35)
    parser.add_argument("--sell-quantile", type=float, default=0.65)
    parser.add_argument("--price-offset-pct", type=float, default=0.0, help="Extra offset applied to levels (buy down, sell up).")
    parser.add_argument("--min-spread-pct", type=float, default=0.0003)
    parser.add_argument("--allocation-usdt", type=float, default=None, help="Notional allocation per cycle in quote units (e.g., FDUSD/USDT).")
    parser.add_argument("--max-cycles", type=int, default=0, help="Max buy->sell cycles to run (0 = unlimited until day end).")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--price-tolerance", type=float, default=0.0008)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--maker-fee", type=float, default=0.0)
    parser.add_argument(
        "--min-predicted-close-return-pct",
        type=float,
        default=None,
        help="Optional direction filter: only open new entries if predicted_close_p50 >= prev_close*(1+threshold).",
    )
    parser.add_argument(
        "--stop-loss-pct",
        type=float,
        default=0.0,
        help="Optional stop-loss from entry fill price (fractional, e.g. 0.02 == 2%%). 0 disables.",
    )
    parser.add_argument(
        "--stop-loss-lockout-until-next-day",
        action="store_true",
        default=False,
        help="After a stop-loss exit, stop trading until the next UTC day (default: off).",
    )
    parser.add_argument(
        "--stop-loss-market-buffer-pct",
        type=float,
        default=0.02,
        help="When stop-loss triggers, place a marketable limit sell at min(stop_price,current_price)*(1-buffer).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    symbol = str(args.symbol).strip().upper()
    buy_q = float(args.buy_quantile)
    sell_q = float(args.sell_quantile)
    if not (0.0 < buy_q < 1.0) or not (0.0 < sell_q < 1.0):
        raise ValueError("buy-quantile and sell-quantile must be in (0, 1).")
    if sell_q <= buy_q:
        raise ValueError("sell-quantile must be greater than buy-quantile.")
    stop_loss_pct = float(args.stop_loss_pct)
    if stop_loss_pct < 0.0 or stop_loss_pct >= 1.0:
        raise ValueError("--stop-loss-pct must be in [0, 1).")
    stop_loss_buffer_pct = float(args.stop_loss_market_buffer_pct)
    if stop_loss_buffer_pct < 0.0 or stop_loss_buffer_pct >= 1.0:
        raise ValueError("--stop-loss-market-buffer-pct must be in [0, 1).")

    levels = _forecast_today_levels(
        symbol=symbol,
        daily_root=Path(args.daily_root),
        buy_quantile=buy_q,
        sell_quantile=sell_q,
        context_length=args.context_length,
        batch_size=args.batch_size,
        model_id=args.model_id,
        device_map=args.device_map,
    )

    buy_price = float(levels.buy_price) * (1.0 - float(args.price_offset_pct))
    sell_price = float(levels.sell_price) * (1.0 + float(args.price_offset_pct))
    buy_price, sell_price = enforce_min_spread(buy_price, sell_price, min_spread_pct=float(args.min_spread_pct))
    buy_price, sell_price = enforce_gap(symbol, buy_price, sell_price, min_gap_pct=float(args.min_spread_pct))

    rules = resolve_symbol_rules(symbol)
    buy_price = quantize_price(buy_price, tick_size=rules.tick_size, side="buy")
    sell_price = quantize_price(sell_price, tick_size=rules.tick_size, side="sell")
    if sell_price <= buy_price:
        raise RuntimeError(f"Invalid quantized spread: buy={buy_price:.8f} sell={sell_price:.8f}")

    trade_allowed = should_trade_predicted_close_return(
        levels.predicted_close_return,
        min_return_pct=args.min_predicted_close_return_pct,
    )
    if not trade_allowed and args.min_predicted_close_return_pct is not None:
        logger.info(
            "Direction filter: no new entries today for {} (predicted_close_return={} < threshold={} predicted_close_p50={} prev_close={})",
            symbol,
            levels.predicted_close_return,
            float(args.min_predicted_close_return_pct),
            levels.predicted_close_p50,
            levels.prev_close,
        )

    logger.info(
        "Daily levels for {} (issued_at={} day={}): buy={:.6f} sell={:.6f} (q_low={:.2f} q_high={:.2f})",
        symbol,
        levels.issued_at,
        levels.day_start,
        buy_price,
        sell_price,
        buy_q,
        sell_q,
    )

    cycle = 0
    while True:
        now = _utc_now()
        if pd.Timestamp(now).floor("D") != levels.day_start:
            logger.info("UTC day changed; exiting daily trader.")
            break
        if args.max_cycles and args.max_cycles > 0 and cycle >= int(args.max_cycles):
            logger.info("Reached max_cycles={}; exiting.", int(args.max_cycles))
            break

        remaining_minutes = _minutes_until_day_end(now)
        expiry_minutes = max(1, remaining_minutes)

        # Always reconcile balances per-loop: this script is intended to run indefinitely in prod.
        quote_free, base_free = get_free_balances(symbol)

        # If we have any meaningful base balance, prioritize selling it at today's sell level
        # before starting a new entry cycle. This avoids accidental position stacking when
        # a watcher partially fills and then expires/cancels.
        base_qty = quantize_qty(float(base_free), step_size=rules.step_size)
        if base_qty > 0 and (rules.min_qty is None or base_qty >= float(rules.min_qty)):
            if rules.min_notional is None or base_qty * sell_price >= float(rules.min_notional):
                logger.info(
                    "Existing base position detected for {}: qty={:.8f}; selling @ {:.6f} (expiry_minutes={})",
                    symbol,
                    base_qty,
                    sell_price,
                    expiry_minutes,
                )
                exit_path = spawn_watcher(
                    WatcherPlan(
                        symbol=symbol,
                        side="sell",
                        mode="daily_exit",
                        limit_price=sell_price,
                        target_qty=float(base_qty),
                        expiry_minutes=int(expiry_minutes),
                        poll_seconds=int(args.poll_seconds),
                        price_tolerance=float(args.price_tolerance),
                        dry_run=bool(args.dry_run),
                    )
                )
                if exit_path is None:
                    logger.warning("Failed to spawn exit watcher for existing position; sleeping.")
                    time.sleep(max(5, int(args.poll_seconds)))
                    continue
                exit_status = _wait_for_watcher(exit_path, poll_seconds=int(args.poll_seconds))
                logger.info("Exit watcher finished (state={}).", exit_status.get("state"))
                continue

        if not trade_allowed:
            logger.info("No existing base position and direction filter blocks new entries; exiting.")
            break

        # Compute buy sizing (per-cycle) based on allocation_usdt and current free balances.
        max_quote = float(args.allocation_usdt) if args.allocation_usdt is not None else float(quote_free)
        if max_quote <= 0:
            logger.warning("No free quote balance available for {}; sleeping.", symbol)
            time.sleep(max(5, int(args.poll_seconds)))
            continue

        cost_per_unit = buy_price * (1.0 + float(args.maker_fee))
        buy_qty = max_quote / cost_per_unit if cost_per_unit > 0 else 0.0
        buy_qty = quantize_qty(buy_qty, step_size=rules.step_size)
        if rules.min_qty and buy_qty < float(rules.min_qty):
            logger.warning(
                "Buy quantity below min_qty for {}: qty={:.8f} min_qty={}",
                symbol,
                buy_qty,
                rules.min_qty,
            )
            time.sleep(max(5, int(args.poll_seconds)))
            continue
        if rules.min_notional and buy_qty * buy_price < float(rules.min_notional):
            logger.warning(
                "Buy notional below min_notional for {}: qty={:.8f} price={:.6f} min_notional={}",
                symbol,
                buy_qty,
                buy_price,
                rules.min_notional,
            )
            time.sleep(max(5, int(args.poll_seconds)))
            continue

        cycle += 1
        logger.info(
            "Cycle {}: waiting to buy {} qty={:.8f} @ {:.6f} (expiry_minutes={})",
            cycle,
            symbol,
            buy_qty,
            buy_price,
            expiry_minutes,
        )

        entry_path = spawn_watcher(
            WatcherPlan(
                symbol=symbol,
                side="buy",
                mode="daily_entry",
                limit_price=buy_price,
                target_qty=float(buy_qty),
                expiry_minutes=int(expiry_minutes),
                poll_seconds=int(args.poll_seconds),
                price_tolerance=float(args.price_tolerance),
                dry_run=bool(args.dry_run),
            )
        )
        if entry_path is None:
            logger.warning("Failed to spawn entry watcher; sleeping.")
            time.sleep(max(5, int(args.poll_seconds)))
            continue
        entry_status = _wait_for_watcher(entry_path, poll_seconds=int(args.poll_seconds))
        filled_qty = _extract_fill_qty(entry_status)
        filled_price = _extract_fill_price(entry_status)
        if filled_qty <= 0:
            logger.info("Entry watcher finished without any fills (state={}).", entry_status.get("state"))
            continue

        # Exit on the same filled quantity.
        filled_qty = quantize_qty(filled_qty, step_size=rules.step_size)
        if rules.min_qty and filled_qty < float(rules.min_qty):
            logger.warning("Entry fill qty below min_qty; skipping exit. qty={:.8f} min_qty={}", filled_qty, rules.min_qty)
            continue
        if rules.min_notional and filled_qty * sell_price < float(rules.min_notional):
            logger.warning(
                "Exit notional below min_notional; skipping exit. qty={:.8f} price={:.6f} min_notional={}",
                filled_qty,
                sell_price,
                rules.min_notional,
            )
            continue
        logger.info("Cycle {}: filled buy qty={:.8f}; waiting to sell @ {:.6f}", cycle, filled_qty, sell_price)

        exit_path = spawn_watcher(
            WatcherPlan(
                symbol=symbol,
                side="sell",
                mode="daily_exit",
                limit_price=sell_price,
                target_qty=float(filled_qty),
                expiry_minutes=int(expiry_minutes),
                poll_seconds=int(args.poll_seconds),
                price_tolerance=float(args.price_tolerance),
                dry_run=bool(args.dry_run),
            )
        )
        if exit_path is None:
            logger.warning("Failed to spawn exit watcher; continuing.")
            continue

        stop_price: Optional[float] = None
        if stop_loss_pct > 0.0:
            entry_price_for_stop = float(filled_price) if filled_price is not None else float(buy_price)
            if entry_price_for_stop > 0:
                stop_price = entry_price_for_stop * (1.0 - stop_loss_pct)
                if stop_price <= 0:
                    stop_price = None

        if stop_price is None:
            exit_status = _wait_for_watcher(exit_path, poll_seconds=int(args.poll_seconds))
            logger.info("Exit watcher finished (state={}).", exit_status.get("state"))
            continue

        binance_symbol = resolve_binance_symbol(symbol)
        poll = max(1, int(args.poll_seconds))
        while True:
            status = _load_watcher_status(exit_path)
            if not status.get("active"):
                logger.info("Exit watcher finished (state={}).", status.get("state"))
                break

            current_price = binance_wrapper.get_symbol_price(binance_symbol)
            if current_price is not None and current_price > 0 and current_price <= float(stop_price):
                logger.warning(
                    "Stop-loss triggered for {}: current_price={:.6f} stop_price={:.6f} (stop_loss_pct={:.4f})",
                    symbol,
                    float(current_price),
                    float(stop_price),
                    float(stop_loss_pct),
                )
                stop_existing_watcher(exit_path, reason="stop_loss")

                _, base_free2 = get_free_balances(symbol)
                base_qty2 = quantize_qty(float(base_free2), step_size=rules.step_size)
                if base_qty2 > 0 and (rules.min_qty is None or base_qty2 >= float(rules.min_qty)):
                    if rules.min_notional is None or base_qty2 * float(current_price) >= float(rules.min_notional):
                        marketable_price = min(float(current_price), float(stop_price)) * (1.0 - stop_loss_buffer_pct)
                        marketable_price = max(0.0, float(marketable_price))
                        logger.warning(
                            "Placing stop-loss exit for {}: qty={:.8f} limit_price={:.6f} (buffer_pct={:.4f})",
                            symbol,
                            base_qty2,
                            marketable_price,
                            float(stop_loss_buffer_pct),
                        )
                        sl_path = spawn_watcher(
                            WatcherPlan(
                                symbol=symbol,
                                side="sell",
                                mode="stop_loss",
                                limit_price=float(marketable_price),
                                target_qty=float(base_qty2),
                                expiry_minutes=int(expiry_minutes),
                                poll_seconds=int(args.poll_seconds),
                                price_tolerance=float(args.price_tolerance),
                                dry_run=bool(args.dry_run),
                            )
                        )
                        if sl_path is None:
                            logger.warning("Failed to spawn stop-loss watcher.")
                        else:
                            sl_status = _wait_for_watcher(sl_path, poll_seconds=int(args.poll_seconds))
                            logger.warning("Stop-loss watcher finished (state={}).", sl_status.get("state"))

                if bool(args.stop_loss_lockout_until_next_day):
                    logger.warning("Stop-loss lockout enabled; exiting for the day.")
                    return
                break

            time.sleep(poll)


if __name__ == "__main__":
    main()
