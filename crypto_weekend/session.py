"""Crypto weekend session — callable from inside the xgb leader process.

Design: this module is imported by the xgb live trader (which already
holds the single `alpaca_live_writer` fcntl lock). All crypto actions
go through the xgb bot's already-constructed `TradingClient`, so there
is exactly ONE program talking to Alpaca.

It does NOT import alpaca_wrapper — doing so would trigger a second
`enforce_live_singleton` call under a different service_name and crash
the xgb bot at startup with "Alpaca account writer lock is already held
in-process" (the in-process registry refuses same-account + different-
service_name pairs).

Strategy is identical to the standalone `crypto_weekend/live_trader.py`:
  - Buy Sat 00:30-23:00 UTC on picks passing `fri_close > SMA20 × 1.05`
    AND `vol_20d ≤ 0.03`.
  - Sell Mon 00:10-12:30 UTC (well before US stock open 13:30).
  - Symbols BTC/ETH/SOL, equal-weight, capped at
    `min(cash - $50, equity × max_gross)`.
"""
from __future__ import annotations

import json
import os
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

# Module-level imports of alpaca-py so tests can patch them and inspect
# constructor call args (the alpaca types are stubbed in tests/conftest.py).
from alpaca.trading.enums import OrderSide, TimeInForce  # noqa: E402
from alpaca.trading.requests import MarketOrderRequest  # noqa: E402

# Re-export the pure signal logic from the standalone module so we don't
# duplicate window-gating / signal math (and keep the 23-test suite valid).
from crypto_weekend.live_trader import (  # noqa: E402
    DUST_MARKET_VALUE_USD,
    SMA_MULT,
    SYMBOLS,
    VOL_CAP,
    compute_signal_from_df,
    in_hold_window,
    is_buy_trigger,
    is_sell_trigger,
    normalize_symbol_for_match,
)

REPO = Path(__file__).resolve().parent.parent
LOG_DIR = REPO / "analysis" / "crypto_weekend_live"

DEFAULT_MAX_GROSS = 0.5


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _open_log_file(day: str) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR / f"{day}.jsonl"


def log_event(event: str, **kwargs: Any) -> None:
    rec = {"ts": utc_now().isoformat(), "event": event, "src": "session"}
    rec.update(kwargs)
    day = utc_now().strftime("%Y-%m-%d")
    line = json.dumps(rec, default=str) + "\n"
    try:
        with _open_log_file(day).open("a") as f:
            f.write(line)
    except Exception as exc:  # noqa: BLE001 — logging must never crash the loop
        print(f"[crypto_session] log write failed: {exc}", flush=True)
    print(line.rstrip(), flush=True)


# ---------- alpaca-py adapters ----------

def _crypto_data_client():
    """Create a CryptoHistoricalDataClient using the same keys as the
    xgb bot. Alpaca crypto data does not require keys but we pass them
    for consistency / higher rate limits.
    """
    from alpaca.data.historical import CryptoHistoricalDataClient
    key = os.environ.get("APCA_API_KEY_ID") or os.environ.get("ALP_KEY_ID")
    sec = os.environ.get("APCA_API_SECRET_KEY") or os.environ.get("ALP_SECRET")
    return CryptoHistoricalDataClient(api_key=key, secret_key=sec)


def fetch_daily_closes(symbol: str, n_days: int = 30) -> pd.DataFrame:
    """Return last `n_days` fully-closed daily bars (UTC) for a crypto
    symbol. Uses Alpaca's public `CryptoHistoricalDataClient` — the
    endpoint accepts compact ("BTCUSD") and slashed ("BTC/USD") forms;
    we request slashed because the alpaca-py client expects it.
    """
    from alpaca.data.requests import CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = _crypto_data_client()
    end_dt = utc_now()
    start_dt = end_dt - timedelta(days=n_days + 5)  # pad for weekends

    # Compact → slashed for the Alpaca API.
    if "/" not in symbol and symbol.upper().endswith("USD"):
        slashed = f"{symbol[:-3]}/{symbol[-3:]}"
    else:
        slashed = symbol

    req = CryptoBarsRequest(
        symbol_or_symbols=slashed,
        timeframe=TimeFrame.Day,
        start=start_dt,
        end=end_dt,
    )
    bars_df = client.get_crypto_bars(req).df
    if bars_df is None or len(bars_df) == 0:
        return pd.DataFrame(columns=["close"])

    # Multi-index: (symbol, timestamp). Drop the symbol level.
    if isinstance(bars_df.index, pd.MultiIndex):
        bars_df = bars_df.reset_index(level=0, drop=True)

    # Keep only fully-closed bars (strictly before today UTC).
    today_utc = utc_now().date()
    keep = bars_df.index.to_series().apply(lambda t: t.date() < today_utc)
    closed_df = bars_df.loc[keep.values, ["close"]].copy()
    return closed_df


def compute_signal(hist_symbol: str) -> dict:
    df = fetch_daily_closes(hist_symbol, n_days=30)
    return compute_signal_from_df(df, symbol=hist_symbol)


def evaluate_signals() -> list[dict]:
    picks = []
    for alpaca_sym, _compact, hist_sym in SYMBOLS:
        try:
            sig = compute_signal(hist_sym)
        except Exception as exc:  # noqa: BLE001
            log_event("signal_error", symbol=alpaca_sym, err=str(exc),
                      trace=traceback.format_exc(limit=3))
            continue
        sig["alpaca_symbol"] = alpaca_sym
        log_event("signal_check", **sig)
        if sig.get("passes"):
            picks.append(sig)
    return picks


# ---------- position + order helpers (TradingClient) ----------

def get_crypto_positions(trading_client, dust_threshold_usd: float = DUST_MARKET_VALUE_USD):
    """Return only crypto positions we care about, excluding dust (<$5)."""
    try:
        positions = trading_client.get_all_positions()
    except Exception as exc:  # noqa: BLE001
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


def _to_slashed(symbol: str) -> str:
    """Alpaca trading API wants slashed crypto symbols (BTC/USD)."""
    s = symbol.upper()
    if "/" in s:
        return s
    if s.endswith("USD"):
        return f"{s[:-3]}/{s[-3:]}"
    return s


def do_sell(trading_client, positions, dry_run: bool) -> int:
    """Close each position via MarketOrder GTC."""
    if not positions:
        log_event("sell_skip_no_positions")
        return 0
    closed = 0
    for pos in positions:
        sym = getattr(pos, "symbol", "")
        qty = abs(float(getattr(pos, "qty", 0.0) or 0.0))
        if qty <= 0:
            continue
        side_raw = str(getattr(pos, "side", "long")).lower()
        # Close long → sell; close short → buy (we never go short crypto,
        # but defend against it).
        side = OrderSide.SELL if "long" in side_raw else OrderSide.BUY
        log_event("sell_submit", symbol=sym, qty=qty, side=side.value,
                  dry_run=dry_run)
        if dry_run:
            closed += 1
            continue
        try:
            req = MarketOrderRequest(
                symbol=_to_slashed(sym),
                qty=qty,
                side=side,
                time_in_force=TimeInForce.GTC,
            )
            order = trading_client.submit_order(order_data=req)
            log_event("sell_submitted", symbol=sym, qty=qty,
                      order_id=str(getattr(order, "id", "")))
            closed += 1
        except Exception as exc:  # noqa: BLE001
            log_event("sell_error", symbol=sym, err=str(exc),
                      trace=traceback.format_exc(limit=3))
    return closed


def do_buy(trading_client, picks: list[dict], max_gross: float,
           dry_run: bool) -> int:
    """Equal-weight buy into `picks`, capped at min(cash - $50, equity × max_gross)."""
    if not picks:
        log_event("buy_skip_no_picks")
        return 0
    try:
        account = trading_client.get_account()
        equity = float(account.equity)
        cash = float(getattr(account, "cash", 0.0) or 0.0)
        buying_power = float(getattr(account, "buying_power", 0.0) or 0.0)
    except Exception as exc:  # noqa: BLE001
        log_event("equity_error", err=str(exc))
        return 0

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
        ref_price = float(p.get("fri_close") or 0)
        if ref_price <= 0:
            log_event("buy_skip_no_price", symbol=alpaca_sym)
            continue
        qty = per_pick_gross / ref_price
        qty = round(qty, 6)
        if qty <= 0:
            log_event("buy_skip_zero_qty", symbol=alpaca_sym, qty=qty)
            continue
        log_event("buy_submit", symbol=alpaca_sym, qty=qty, price=ref_price,
                  notional=per_pick_gross, dry_run=dry_run)
        if dry_run:
            submitted += 1
            continue
        try:
            req = MarketOrderRequest(
                symbol=_to_slashed(alpaca_sym),
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC,
            )
            order = trading_client.submit_order(order_data=req)
            log_event("buy_submitted", symbol=alpaca_sym, qty=qty,
                      price=ref_price,
                      order_id=str(getattr(order, "id", "")))
            submitted += 1
        except Exception as exc:  # noqa: BLE001
            log_event("buy_error", symbol=alpaca_sym, err=str(exc),
                      trace=traceback.format_exc(limit=3))
    return submitted


# ---------- main entry point for the xgb bot ----------

_LAST_ACTION_DATE: Optional[str] = None


def run_crypto_tick(trading_client, *, max_gross: float = DEFAULT_MAX_GROSS,
                    dry_run: bool = False, now: Optional[datetime] = None) -> dict:
    """One crypto poll. Called from the xgb bot's inter-session sleep.

    Returns a small status dict for logging/testing. Idempotent: if today
    already triggered an action, subsequent calls are no-ops until the
    UTC date rolls over.
    """
    global _LAST_ACTION_DATE
    now = now or utc_now()
    today = now.strftime("%Y-%m-%d")
    positions = get_crypto_positions(trading_client)
    n_positions = len(positions)
    sell_win = is_sell_trigger(now)
    buy_win = is_buy_trigger(now)

    status = {
        "ts": now.isoformat(), "n_positions": n_positions,
        "buy_win": buy_win, "sell_win": sell_win,
        "hold_win": in_hold_window(now),
        "last_action_date": _LAST_ACTION_DATE,
        "action": "none",
    }

    if sell_win and n_positions > 0:
        log_event("sell_trigger", n_positions=n_positions, now=str(now))
        do_sell(trading_client, positions, dry_run)
        _LAST_ACTION_DATE = today
        status["action"] = "sell"
    elif buy_win and n_positions == 0 and _LAST_ACTION_DATE != today:
        picks = evaluate_signals()
        log_event("buy_trigger", n_picks=len(picks), now=str(now))
        do_buy(trading_client, picks, max_gross, dry_run)
        _LAST_ACTION_DATE = today
        status["action"] = "buy"
    return status


def reset_state_for_testing() -> None:
    """Unit-test helper."""
    global _LAST_ACTION_DATE
    _LAST_ACTION_DATE = None
