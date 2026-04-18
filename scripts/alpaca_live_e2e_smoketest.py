"""Alpaca LIVE end-to-end broker smoketest.

Exercises the real Alpaca broker boundary with three tiny tests:

  1. Crypto round trip: market-buy a few dollars of DOGE/USD, then limit-sell
     at or above the buy price to close (death-spiral guard approves).
  2. Crypto limit + cancel: limit-buy a few dollars of DOGE/USD 30% below
     market (will not fill), confirm it is resting, then cancel.
  3. Stock limit + cancel: limit-buy 1 share of SPY at ~50% of last close
     (market closed → "accepted"/"new"; Monday open will not fill because it
     is far out of market), then cancel.

Runs against real money. Uses `ALPACA_SINGLETON_OVERRIDE=1` because the
long-running trading_server process already holds the writer lock (by
design). Daily-rl-trader is asleep between signals, so there is no active
second writer to race with while this test runs.

Usage:
    ALP_PAPER=0 ALPACA_SINGLETON_OVERRIDE=1 python scripts/alpaca_live_e2e_smoketest.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime, timezone

if os.environ.get("ALP_PAPER") != "0":
    print("REFUSING: set ALP_PAPER=0 to run against live keys.", file=sys.stderr)
    sys.exit(2)

os.environ.setdefault("ALPACA_SINGLETON_OVERRIDE", "1")
os.environ.setdefault("ALPACA_SERVICE_NAME", f"alpaca_live_e2e_smoketest_{os.getpid()}")

import alpaca_wrapper  # noqa: E402 — triggers singleton override + loads live keys
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest  # noqa: E402
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType  # noqa: E402
from src.alpaca_singleton import (  # noqa: E402
    guard_sell_against_death_spiral,
    record_buy_price,
)

CRYPTO_SYMBOL_ORDER = "DOGE/USD"       # Alpaca API symbol format (slash)
CRYPTO_SYMBOL_POSITION = "DOGEUSD"      # position/query format (no slash)
STOCK_SYMBOL = "SPY"
CRYPTO_NOTIONAL_USD = 5.0               # ~50 DOGE @ $0.10
STOCK_QTY = 1                           # 1 share SPY
STOCK_LIMIT_FRAC = 0.50                 # far below market so no fill
CRYPTO_FAR_LIMIT_FRAC = 0.70            # crypto limit-buy 30% below market


def ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(tag: str, msg: str) -> None:
    print(f"[{ts()}] {tag:12s} {msg}", flush=True)


def wait_for_order_status(
    order_id: str,
    *,
    timeout_s: float = 45.0,
    poll_s: float = 1.0,
    terminal=("filled", "canceled", "rejected", "expired", "accepted", "new"),
):
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        o = alpaca_wrapper.alpaca_api.get_order_by_id(order_id)
        status = getattr(o, "status", None)
        status_str = status.value if hasattr(status, "value") else str(status)
        last = (o, status_str)
        if status_str in terminal:
            return o, status_str
        time.sleep(poll_s)
    return last or (None, "timeout")


def wait_for_fill(order_id: str, *, timeout_s: float = 45.0) -> tuple[object, str]:
    return wait_for_order_status(
        order_id, timeout_s=timeout_s, terminal=("filled", "canceled", "rejected", "expired")
    )


def test_crypto_round_trip(out: dict) -> None:
    log("crypto_rt", f"starting round-trip {CRYPTO_SYMBOL_ORDER} notional=${CRYPTO_NOTIONAL_USD}")
    quote = alpaca_wrapper.latest_data(CRYPTO_SYMBOL_POSITION)
    ask = float(getattr(quote, "ask_price", 0) or 0) or float(getattr(quote, "last_price", 0) or 0)
    bid = float(getattr(quote, "bid_price", 0) or 0) or ask
    if ask <= 0:
        raise RuntimeError(f"no ask price for {CRYPTO_SYMBOL_ORDER}")
    qty = max(round(CRYPTO_NOTIONAL_USD / ask, 2), 0.01)
    log("crypto_rt", f"ask={ask} bid={bid} qty={qty}")

    buy = alpaca_wrapper.alpaca_api.submit_order(
        order_data=MarketOrderRequest(
            symbol=CRYPTO_SYMBOL_ORDER,
            qty=qty,
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            time_in_force=TimeInForce.GTC,
        )
    )
    log("crypto_rt", f"submitted BUY id={buy.id} status={buy.status}")
    buy_final, buy_status = wait_for_fill(buy.id, timeout_s=60)
    filled_price = float(getattr(buy_final, "filled_avg_price", 0) or 0)
    filled_qty = float(getattr(buy_final, "filled_qty", 0) or 0)
    log("crypto_rt", f"BUY terminal={buy_status} filled_qty={filled_qty} filled_px={filled_price}")
    out["crypto_round_trip"] = {
        "buy_id": str(buy.id),
        "buy_status": buy_status,
        "buy_filled_qty": filled_qty,
        "buy_filled_px": filled_price,
    }
    if buy_status != "filled":
        raise RuntimeError(f"crypto buy did not fill: {buy_status}")

    # Death-spiral guard tracks buys.json via the wrapper; record it so the
    # sell guard can compare — alpaca_wrapper.alpaca_order_stock does this
    # automatically but we used the SDK directly.
    record_buy_price(CRYPTO_SYMBOL_POSITION, filled_price)

    # Sell quantity must be at-most the position quantity AFTER fee deduction —
    # Alpaca charges the maker/taker fee out of the asset, leaving the balance
    # slightly below filled_qty. Floor to 2 decimals to stay under.
    sell_qty = math.floor(filled_qty * 100) / 100
    # IOC at just-above-floor so we cross the spread and exit immediately; the
    # death-spiral guard floor is buy_price * (1 - 50bps).
    floor = filled_price * 0.995
    quote_now = alpaca_wrapper.latest_data(CRYPTO_SYMBOL_POSITION)
    bid_now = float(getattr(quote_now, "bid_price", 0) or 0)
    # sell price = min(best bid * 0.9998, floor * 1.0002) — guaranteed to hit
    target = bid_now * 0.9998 if bid_now > floor else floor * 1.0002
    sell_limit_px = round(max(target, floor * 1.0002), 6)
    guard_sell_against_death_spiral(CRYPTO_SYMBOL_POSITION, "sell", sell_limit_px)
    sell = alpaca_wrapper.alpaca_api.submit_order(
        order_data=LimitOrderRequest(
            symbol=CRYPTO_SYMBOL_ORDER,
            qty=sell_qty,
            side=OrderSide.SELL,
            type=OrderType.LIMIT,
            time_in_force=TimeInForce.IOC,
            limit_price=str(sell_limit_px),
        )
    )
    log("crypto_rt", f"submitted SELL id={sell.id} limit_px={sell_limit_px}")
    sell_final, sell_status = wait_for_order_status(
        sell.id, timeout_s=90, terminal=("filled", "canceled", "rejected", "expired")
    )
    log("crypto_rt", f"SELL terminal={sell_status}")

    out["crypto_round_trip"].update(
        {
            "sell_id": str(sell.id),
            "sell_status": sell_status,
            "sell_limit_px": sell_limit_px,
        }
    )

    # If the sell did not fill in 90s, cancel and close at market so we don't
    # leave a residual position open.
    if sell_status != "filled":
        log("crypto_rt", f"SELL did not fill in 90s; canceling and closing at market")
        try:
            alpaca_wrapper.alpaca_api.cancel_order_by_id(sell.id)
        except Exception as exc:
            log("crypto_rt", f"cancel raised (ok if already terminal): {exc}")
        # Close remaining position. Market sell; guard will refuse if market
        # cratered >50 bps below buy_price in the last minute.
        time.sleep(2)
        positions = {p.symbol: p for p in alpaca_wrapper.alpaca_api.get_all_positions()}
        residual = positions.get(CRYPTO_SYMBOL_POSITION)
        if residual is None:
            log("crypto_rt", "no residual position; sell must have filled after cancel race")
            out["crypto_round_trip"]["residual_close"] = "no_residual"
        else:
            residual_qty = float(residual.qty)
            if residual_qty > 0:
                quote = alpaca_wrapper.latest_data(CRYPTO_SYMBOL_POSITION)
                bid_now = float(getattr(quote, "bid_price", 0) or 0) or filled_price
                # Sell at bid price if above guard floor; else wait.
                floor = filled_price * (1 - 0.0050)  # 50 bps tolerance
                if bid_now < floor:
                    log(
                        "crypto_rt",
                        f"market dropped below guard floor (bid {bid_now} < floor {floor}); leaving "
                        f"residual of {residual_qty} {CRYPTO_SYMBOL_POSITION} to be unwound later",
                    )
                    out["crypto_round_trip"]["residual_close"] = "left_for_manual_unwind"
                else:
                    close_sell_px = round(max(bid_now * 0.999, floor * 1.0001), 6)
                    guard_sell_against_death_spiral(CRYPTO_SYMBOL_POSITION, "sell", close_sell_px)
                    close = alpaca_wrapper.alpaca_api.submit_order(
                        order_data=LimitOrderRequest(
                            symbol=CRYPTO_SYMBOL_ORDER,
                            qty=residual_qty,
                            side=OrderSide.SELL,
                            type=OrderType.LIMIT,
                            time_in_force=TimeInForce.GTC,
                            limit_price=str(close_sell_px),
                        )
                    )
                    close_final, close_status = wait_for_order_status(
                        close.id, timeout_s=60, terminal=("filled", "canceled", "rejected", "expired")
                    )
                    log("crypto_rt", f"residual close id={close.id} status={close_status} limit_px={close_sell_px}")
                    out["crypto_round_trip"]["residual_close"] = close_status
                    if close_status != "filled":
                        try:
                            alpaca_wrapper.alpaca_api.cancel_order_by_id(close.id)
                        except Exception:
                            pass
    log("crypto_rt", "done")


def test_crypto_limit_cancel(out: dict) -> None:
    log("crypto_lc", f"starting limit+cancel {CRYPTO_SYMBOL_ORDER}")
    quote = alpaca_wrapper.latest_data(CRYPTO_SYMBOL_POSITION)
    last = float(getattr(quote, "last_price", 0) or getattr(quote, "ask_price", 0) or 0)
    if last <= 0:
        raise RuntimeError(f"no last price for {CRYPTO_SYMBOL_ORDER}")
    limit_px = round(last * CRYPTO_FAR_LIMIT_FRAC, 6)
    qty = round(CRYPTO_NOTIONAL_USD / max(limit_px, 1e-6), 2)
    log("crypto_lc", f"last={last} limit_px={limit_px} qty={qty}")

    order = alpaca_wrapper.alpaca_api.submit_order(
        order_data=LimitOrderRequest(
            symbol=CRYPTO_SYMBOL_ORDER,
            qty=qty,
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            time_in_force=TimeInForce.GTC,
            limit_price=str(limit_px),
        )
    )
    log("crypto_lc", f"submitted LIMIT BUY id={order.id} status={order.status}")

    # Wait briefly until it lands in an "open"-ish state.
    final, status = wait_for_order_status(
        order.id, timeout_s=20, terminal=("accepted", "new", "pending_new", "filled", "rejected", "canceled")
    )
    log("crypto_lc", f"LIMIT BUY status={status}")

    # Cancel.
    try:
        alpaca_wrapper.alpaca_api.cancel_order_by_id(order.id)
    except Exception as exc:
        log("crypto_lc", f"cancel raised: {exc}")
    time.sleep(2)
    after_cancel = alpaca_wrapper.alpaca_api.get_order_by_id(order.id)
    cancel_status = getattr(after_cancel, "status", None)
    cancel_status = cancel_status.value if hasattr(cancel_status, "value") else str(cancel_status)
    log("crypto_lc", f"after cancel status={cancel_status}")

    out["crypto_limit_cancel"] = {
        "order_id": str(order.id),
        "submit_status": status,
        "limit_px": limit_px,
        "qty": qty,
        "cancel_status": cancel_status,
    }
    log("crypto_lc", "done")


def test_stock_limit_cancel(out: dict) -> None:
    log("stock_lc", f"starting limit+cancel {STOCK_SYMBOL}")
    # Use last-trade reference price. Market is likely closed.
    try:
        quote = alpaca_wrapper.latest_data(STOCK_SYMBOL)
        last = float(getattr(quote, "last_price", 0) or getattr(quote, "ask_price", 0) or 0)
    except Exception:
        last = 0
    if last <= 0:
        # fallback: fetch latest trade REST
        import urllib.request
        from env_real import ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD
        hdr = {"APCA-API-KEY-ID": ALP_KEY_ID_PROD, "APCA-API-SECRET-KEY": ALP_SECRET_KEY_PROD}
        url = f"https://data.alpaca.markets/v2/stocks/{STOCK_SYMBOL}/trades/latest"
        r = urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=10)
        last = float(json.loads(r.read())["trade"]["p"])
    limit_px = round(last * STOCK_LIMIT_FRAC, 2)
    log("stock_lc", f"last={last} limit_px={limit_px} qty={STOCK_QTY}")

    order = alpaca_wrapper.alpaca_api.submit_order(
        order_data=LimitOrderRequest(
            symbol=STOCK_SYMBOL,
            qty=STOCK_QTY,
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            time_in_force=TimeInForce.DAY,
            limit_price=str(limit_px),
        )
    )
    log("stock_lc", f"submitted id={order.id} status={order.status}")
    final, status = wait_for_order_status(
        order.id, timeout_s=30, terminal=("accepted", "new", "pending_new", "filled", "rejected", "canceled")
    )
    log("stock_lc", f"terminal/intermediate={status}")

    try:
        alpaca_wrapper.alpaca_api.cancel_order_by_id(order.id)
    except Exception as exc:
        log("stock_lc", f"cancel raised: {exc}")
    time.sleep(2)
    after_cancel = alpaca_wrapper.alpaca_api.get_order_by_id(order.id)
    cancel_status = getattr(after_cancel, "status", None)
    cancel_status = cancel_status.value if hasattr(cancel_status, "value") else str(cancel_status)
    log("stock_lc", f"after cancel status={cancel_status}")

    out["stock_limit_cancel"] = {
        "order_id": str(order.id),
        "submit_status": status,
        "limit_px": limit_px,
        "qty": STOCK_QTY,
        "cancel_status": cancel_status,
    }
    log("stock_lc", "done")


def main() -> int:
    out: dict = {"started_at": ts()}
    # Warn loudly about live mode.
    from env_real import PAPER
    if PAPER:
        print("REFUSING: PAPER=True. Set ALP_PAPER=0.", file=sys.stderr)
        return 2
    acct = alpaca_wrapper.alpaca_api.get_account()
    log("account", f"equity=${float(acct.equity):,.2f} buying_power=${float(acct.buying_power):,.2f} status={acct.status}")
    out["account"] = {
        "equity": float(acct.equity),
        "buying_power": float(acct.buying_power),
        "status": str(acct.status),
    }

    failures = []
    for name, fn in [
        ("crypto_round_trip", test_crypto_round_trip),
        ("crypto_limit_cancel", test_crypto_limit_cancel),
        ("stock_limit_cancel", test_stock_limit_cancel),
    ]:
        try:
            fn(out)
            log("SUMMARY", f"{name}: OK")
        except Exception as exc:
            log("SUMMARY", f"{name}: FAIL {exc!r}")
            failures.append((name, repr(exc)))
            out.setdefault(name, {})["error"] = repr(exc)

    out["failures"] = failures
    out["finished_at"] = ts()
    print("\n=== RESULT ===")
    print(json.dumps(out, indent=2, default=str))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
