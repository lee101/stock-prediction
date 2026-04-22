"""Crypto weekend-only live trader.

Runs as a long-lived daemon. Poll cadence default 5 min. Windows (UTC):

  - BUY  trigger: Saturday 00:30 – 23:00 UTC  (Friday daily bar has closed)
  - HOLD window : Sat + Sun + Mon 00:00 – 12:30 UTC
  - SELL trigger: Monday 00:10 – 12:30 UTC   (well before US stock open 13:30)
  - FLAT window : Tue 00:00 – Fri 23:59 UTC  (no positions)

Strategy (validated OOS — see crypto_weekend/backtest_tight.py):
  signal = fri_close > sma20 * 1.05   AND   vol_20d <= 0.03
  picks  = {BTC, ETH, SOL} that pass
  size   = equal-weight, total gross = MAX_GROSS of equity (default 0.5 = half)

Singleton: ALPACA_ACCOUNT_NAME=alpaca_crypto_writer (separate from
alpaca_live_writer stock bot). Coexists with xgb-daily-trader-live.

DOES NOT write to the stock bot's buy-memory (guard is account-scoped), so
overnight / weekend-hold crypto sells are not refused. Crypto symbols
use Alpaca slashed format "BTC/USD".
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
LOG_DIR = REPO / "analysis" / "crypto_weekend_live"

# Trio: (alpaca_order_symbol, compact_for_match, hist_symbol)
# alpaca_wrapper routes crypto via compact form "BTCUSD" — its internal
# `remap_symbols` converts to slashed "BTC/USD" for the Alpaca order API,
# and its `latest_data` / `download_symbol_history` crypto routing both key
# off `all_crypto_symbols` which contains compact form only.
SYMBOLS = [
    ("BTCUSD", "BTCUSD", "BTCUSD"),
    ("ETHUSD", "ETHUSD", "ETHUSD"),
    ("SOLUSD", "SOLUSD", "SOLUSD"),
]

SMA_MULT = 1.05
VOL_CAP = 0.03
FEE_BPS = 10.0  # informational only — Alpaca quotes its own fee
DEFAULT_MAX_GROSS = 0.5  # conservative vs backtest 1.0


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def in_hold_window(now: datetime) -> bool:
    """We may carry a crypto position at this time."""
    dow = now.weekday()  # Mon=0 .. Sun=6
    if dow in (5, 6):  # Sat, Sun
        return True
    if dow == 0:  # Mon
        return now.hour < 13  # up to 13:00 UTC (30min before US open 13:30)
    return False


def is_buy_trigger(now: datetime) -> bool:
    """Saturday 00:30 – 23:00 UTC — eligible to open new positions."""
    return now.weekday() == 5 and 0 <= now.hour < 23


def is_sell_trigger(now: datetime) -> bool:
    """Monday 00:10 – 12:30 UTC — eligible to close positions before US open."""
    if now.weekday() != 0:
        return False
    # Force flat if we're anywhere on Monday before 12:30 UTC.
    return now.hour < 12 or (now.hour == 12 and now.minute < 30)


def fetch_daily_closes(alpaca_wrapper_mod, symbol: str, n_days: int = 30) -> pd.DataFrame:
    """Return last `n_days` daily closes for a crypto symbol via Alpaca.

    Returned df has a timestamp index (UTC) and a `close` column with only
    FULLY-CLOSED bars (any bar whose UTC day is still today is dropped).
    """
    from datetime import timedelta
    end_dt = utc_now()
    start_dt = end_dt - timedelta(days=n_days + 5)  # pad for weekends/holidays
    df = alpaca_wrapper_mod.download_symbol_history(
        symbol, start=start_dt, end=end_dt, include_latest=False,
    )
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["close"])
    # Keep only fully-closed daily bars (strictly before today UTC).
    today_utc = utc_now().date()
    keep = df.index.to_series().apply(lambda t: t.date() < today_utc)
    closed_df = df.loc[keep.values, ["close"]].copy()
    return closed_df


def compute_signal_from_df(df: pd.DataFrame, symbol: str = "") -> dict:
    """Compute weekend signal from a dataframe with a `close` column.

    Splits the latest closed bar as the reference (~ Friday when polled
    on Saturday morning UTC) from the prior 20-day window used for SMA
    and vol features.
    """
    if df is None or len(df) < 21:
        return {"passes": False, "reason": "not_enough_history",
                "n_bars": int(0 if df is None else len(df)), "symbol": symbol}
    last = df.iloc[-1]
    fri_close = float(last["close"])
    window = df.iloc[-21:-1]
    if len(window) != 20:
        return {"passes": False, "reason": "bad_window", "symbol": symbol}
    sma_20 = float(window["close"].mean())
    rets = window["close"].pct_change().dropna()
    vol_20d = float(rets.std()) if len(rets) > 1 else float("nan")
    above_sma = fri_close > sma_20 * SMA_MULT
    vol_ok = (vol_20d == vol_20d) and vol_20d <= VOL_CAP  # NaN-safe
    passes = bool(above_sma and vol_ok)
    return {
        "passes": passes,
        "symbol": symbol,
        "fri_close": fri_close,
        "sma_20": sma_20,
        "vol_20d": vol_20d,
        "above_sma": bool(above_sma),
        "vol_ok": bool(vol_ok),
        "reference_bar_time": str(df.index[-1]),
    }


def compute_signal(alpaca_wrapper_mod, hist_symbol: str) -> dict:
    df = fetch_daily_closes(alpaca_wrapper_mod, hist_symbol, n_days=30)
    return compute_signal_from_df(df, symbol=hist_symbol)


def normalize_symbol_for_match(sym: str) -> str:
    return str(sym).upper().replace("/", "")


def open_log_file(day: str) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR / f"{day}.jsonl"


def log_event(event: str, **kwargs) -> None:
    rec = {"ts": utc_now().isoformat(), "event": event}
    rec.update(kwargs)
    day = utc_now().strftime("%Y-%m-%d")
    line = json.dumps(rec, default=str) + "\n"
    try:
        with open_log_file(day).open("a") as f:
            f.write(line)
    except Exception as exc:
        print(f"[log_event] write failed: {exc}", file=sys.stderr, flush=True)
    print(line.rstrip(), flush=True)


DUST_MARKET_VALUE_USD = 5.0  # anything below $5 is abandoned dust, not a position


def get_crypto_positions(alpaca_wrapper_mod, dust_threshold_usd: float = DUST_MARKET_VALUE_USD):
    """Return only crypto positions we care about, excluding dust.

    Alpaca accounts accumulate sub-cent dust from past trading (e.g. BTCUSD
    qty=1e-8, mv=$0.0008). These match our symbols but should not block new
    buys or trigger sells. Filter anything with |market_value| < threshold.
    """
    try:
        positions = alpaca_wrapper_mod.get_all_positions()
    except Exception as exc:
        log_event("positions_error", err=str(exc))
        return []
    our_syms = {normalize_symbol_for_match(s) for trio in SYMBOLS for s in trio[:2]}
    ours = []
    for pos in positions:
        sym_norm = normalize_symbol_for_match(getattr(pos, "symbol", ""))
        if sym_norm not in our_syms:
            continue
        try:
            mv = abs(float(getattr(pos, "market_value", 0.0) or 0.0))
        except (TypeError, ValueError):
            mv = 0.0
        if mv < dust_threshold_usd:
            continue
        ours.append(pos)
    return ours


def do_sell(alpaca_wrapper_mod, dry_run: bool) -> int:
    positions = get_crypto_positions(alpaca_wrapper_mod)
    if not positions:
        log_event("sell_skip_no_positions")
        return 0
    closed = 0
    for pos in positions:
        sym = getattr(pos, "symbol", "")
        qty = float(getattr(pos, "qty", 0.0) or 0.0)
        log_event("sell_submit", symbol=sym, qty=qty, dry_run=dry_run)
        if dry_run:
            closed += 1
            continue
        try:
            alpaca_wrapper_mod.close_position_violently(pos)
            log_event("sell_submitted", symbol=sym, qty=qty)
            closed += 1
        except Exception as exc:
            log_event("sell_error", symbol=sym, err=str(exc),
                      trace=traceback.format_exc(limit=3))
    return closed


def do_buy(alpaca_wrapper_mod, picks: list[dict], max_gross: float,
           dry_run: bool) -> int:
    if not picks:
        log_event("buy_skip_no_picks")
        return 0
    try:
        account = alpaca_wrapper_mod.alpaca_api.get_account()
        equity = float(account.equity)
        cash = float(getattr(account, "cash", 0.0) or 0.0)
        buying_power = float(getattr(account, "buying_power", 0.0) or 0.0)
    except Exception as exc:
        log_event("equity_error", err=str(exc))
        return 0
    # Never over-lever: if stock bot holds positions, cash < equity; size off the
    # smaller of (equity × max_gross) and (cash). A tiny $50 buffer keeps us from
    # consuming the full cash pool, in case Alpaca has any intraday settlement lag.
    target_gross = max_gross * equity
    available_cash = max(0.0, cash - 50.0)
    gross_cap = min(target_gross, available_cash)
    if gross_cap < 100.0:
        log_event("buy_skip_insufficient_cash", equity=equity, cash=cash,
                  buying_power=buying_power, target_gross=target_gross,
                  gross_cap=gross_cap)
        return 0
    per_pick_gross = gross_cap / len(picks)
    log_event("buy_plan", n_picks=len(picks), equity=equity, cash=cash,
              buying_power=buying_power, max_gross=max_gross,
              target_gross=target_gross, gross_cap=gross_cap,
              per_pick_gross=per_pick_gross)
    submitted = 0
    for p in picks:
        alpaca_sym = p["alpaca_symbol"]
        ref_price = p["fri_close"]
        # Try latest Alpaca quote for precise pricing.
        price = ref_price
        try:
            from alpaca_wrapper import latest_data  # type: ignore
            quote = latest_data(alpaca_sym)
            if quote is not None:
                ask = float(getattr(quote, "ask_price", 0.0) or 0.0)
                bid = float(getattr(quote, "bid_price", 0.0) or 0.0)
                if ask > 0 and bid > 0:
                    price = (ask + bid) / 2.0
        except Exception as exc:
            log_event("latest_data_error", symbol=alpaca_sym, err=str(exc))
        qty = per_pick_gross / max(price, 1e-9)
        qty = round(qty, 6)
        if qty <= 0:
            log_event("buy_skip_zero_qty", symbol=alpaca_sym, qty=qty, price=price)
            continue
        log_event("buy_submit", symbol=alpaca_sym, qty=qty, price=price,
                  notional=per_pick_gross, dry_run=dry_run)
        if dry_run:
            submitted += 1
            continue
        try:
            result = alpaca_wrapper_mod.open_order_at_price(alpaca_sym, qty, "buy", price)
            log_event("buy_submitted", symbol=alpaca_sym, qty=qty, price=price,
                      order_id=str(getattr(result, "id", "")) if result else "")
            submitted += 1
        except Exception as exc:
            log_event("buy_error", symbol=alpaca_sym, err=str(exc),
                      trace=traceback.format_exc(limit=3))
    return submitted


def evaluate_signals(alpaca_wrapper_mod) -> list[dict]:
    picks = []
    for alpaca_sym, compact_sym, hist_sym in SYMBOLS:
        try:
            sig = compute_signal(alpaca_wrapper_mod, hist_sym)
        except Exception as exc:
            log_event("signal_error", symbol=alpaca_sym, err=str(exc),
                      trace=traceback.format_exc(limit=3))
            continue
        sig["alpaca_symbol"] = alpaca_sym
        log_event("signal_check", **sig)
        if sig.get("passes"):
            picks.append(sig)
    return picks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poll-seconds", type=int, default=300)
    ap.add_argument("--max-gross", type=float, default=DEFAULT_MAX_GROSS,
                    help="Total crypto gross exposure, as a fraction of equity")
    ap.add_argument("--dry-run", action="store_true",
                    help="Log intent only, do not submit orders")
    ap.add_argument("--once", action="store_true",
                    help="Single poll iteration, then exit")
    ap.add_argument("--force-buy", action="store_true",
                    help="(Test) ignore day-of-week gate for buy trigger")
    ap.add_argument("--force-sell", action="store_true",
                    help="(Test) ignore day-of-week gate for sell trigger")
    args = ap.parse_args()

    os.environ.setdefault("ALPACA_ACCOUNT_NAME", "alpaca_crypto_writer")
    os.environ.setdefault("ALPACA_SERVICE_NAME", f"crypto_weekend_live_{os.getpid()}")

    # Import AFTER env vars are set so singleton uses our account.
    try:
        import alpaca_wrapper as aw
    except SystemExit as exc:
        print(f"[crypto_weekend_live] singleton refused startup: {exc}",
              file=sys.stderr, flush=True)
        return int(getattr(exc, "code", 42))

    log_event("start", pid=os.getpid(), dry_run=args.dry_run,
              account=os.environ["ALPACA_ACCOUNT_NAME"],
              max_gross=args.max_gross, poll_seconds=args.poll_seconds,
              paper=os.environ.get("ALP_PAPER", "0"))

    last_action_date = None  # date-string: "YYYY-MM-DD"

    while True:
        try:
            now = utc_now()
            positions = get_crypto_positions(aw)
            n_positions = len(positions)
            sell_win = is_sell_trigger(now) or args.force_sell
            buy_win = is_buy_trigger(now) or args.force_buy
            today = now.strftime("%Y-%m-%d")

            if sell_win and n_positions > 0:
                log_event("sell_trigger", n_positions=n_positions, now=str(now))
                do_sell(aw, args.dry_run)
                last_action_date = today
            elif buy_win and n_positions == 0 and last_action_date != today:
                picks = evaluate_signals(aw)
                log_event("buy_trigger", n_picks=len(picks), now=str(now))
                do_buy(aw, picks, args.max_gross, args.dry_run)
                last_action_date = today
            else:
                log_event("poll", n_positions=n_positions,
                          buy_win=buy_win, sell_win=sell_win,
                          hold_win=in_hold_window(now))
        except Exception as exc:
            log_event("loop_error", err=str(exc),
                      trace=traceback.format_exc(limit=5))

        if args.once:
            break
        time.sleep(args.poll_seconds)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
