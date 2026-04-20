#!/usr/bin/env python3
"""XGBoost daily open-to-close live trader.

Strategy: each trading morning, score all stocks → buy top-N at open → sell at close.

Session flow
------------
1. Pre-open  (~9:25 ET): load yesterday's OHLCV, compute features, score 846 stocks
2. At open   (9:30 ET): place market BUY orders for top-N candidates
3. Near close (15:50 ET): place market SELL orders for all held positions
4. Post-close: log results

Safety
------
- Imports src.alpaca_singleton for live-writer singleton guard
- Only one live process allowed (fcntl lock on alpaca_live_writer.lock)
- ALP_PAPER=1 runs in paper mode (bypasses singleton)
- Reads API keys from env_real.py

Usage
-----
  # Paper mode (safe for testing):
  ALP_PAPER=1 python -m xgbnew.live_trader --top-n 2

  # Live mode:
  python -m xgbnew.live_trader --top-n 2 --live --allocation 0.25

  # Dry run (score only, no orders):
  python -m xgbnew.live_trader --top-n 2 --dry-run
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from xgbnew.dataset import build_daily_dataset, _load_symbol_csv
from xgbnew.features import DAILY_FEATURE_COLS, build_features_for_symbol
from xgbnew.model import XGBStockModel
from xgbnew.backtest import BacktestConfig

logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")

MARKET_OPEN  = (9, 30)   # HH, MM ET
MARKET_CLOSE = (15, 50)  # sell by 15:50 (10 min before 4pm close)
DEFAULT_MODEL_PATH = REPO / "analysis/xgbnew_daily/live_model.pkl"
DEFAULT_LIVE_BAR_BATCH_SIZE = 200
MIN_FRACTIONAL_QTY = 0.0001


# ── Alpaca client ─────────────────────────────────────────────────────────────

def _build_trading_client(paper: bool):
    """Build Alpaca TradingClient with correct keys."""
    try:
        import importlib
        env_real = importlib.import_module("env_real")
    except ImportError as exc:
        raise RuntimeError("env_real.py not found — cannot get Alpaca keys") from exc

    if paper:
        key_id = getattr(env_real, "ALP_KEY_ID_PAPER", getattr(env_real, "ALP_KEY_ID", ""))
        secret = getattr(env_real, "ALP_SECRET_KEY_PAPER", getattr(env_real, "ALP_SECRET_KEY", ""))
    else:
        key_id = getattr(env_real, "ALP_KEY_ID_PROD", "")
        secret = getattr(env_real, "ALP_SECRET_KEY_PROD", "")

    from alpaca.trading.client import TradingClient
    return TradingClient(key_id, secret, paper=paper)


def _submit_market_order(client, *, symbol: str, qty: float, side: str):
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    side_val = OrderSide.BUY if side == "buy" else OrderSide.SELL
    req = MarketOrderRequest(
        symbol=symbol,
        qty=round(float(qty), 4),
        side=side_val,
        time_in_force=TimeInForce.DAY,
    )
    return client.submit_order(req)


def _poll_filled_avg_price(client, order_id: str, *, timeout_s: float = 30.0) -> float | None:
    """Poll an order until it reports a filled_avg_price or timeout.

    Returns the float fill price (None on timeout/no fill). Used to feed the
    death-spiral guard's buy-price memory so future sells are vetted against
    the actual cost basis rather than a pre-fill estimate.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            o = client.get_order_by_id(order_id)
        except Exception:
            time.sleep(1.0)
            continue
        price = getattr(o, "filled_avg_price", None)
        if price is not None:
            try:
                p = float(price)
                if p > 0:
                    return p
            except (TypeError, ValueError):
                pass
        status = str(getattr(o, "status", "")).lower()
        if status in ("canceled", "rejected", "expired"):
            return None
        time.sleep(1.0)
    return None


def _get_account(client):
    return client.get_account()


def _is_today_trading_day(client, now: datetime | None = None) -> tuple[bool, str]:
    """Query Alpaca's market clock and return (is_trading_day, reason).

    Uses the broker's own calendar — covers weekends AND holidays without
    needing pandas_market_calendars. Reason string is for logging.

    The check is: if `is_open` is True, obviously a trading day. Otherwise,
    if `next_open.date()` equals today (ET), the market will open today →
    trading day. Anything else (weekend, holiday) is not a trading day.
    """
    now_et_date = (now or datetime.now(timezone.utc)).astimezone(ET).date()
    try:
        clock = client.get_clock()
    except Exception as exc:
        return True, f"clock_query_failed: {exc} (assuming trading day)"
    is_open = bool(getattr(clock, "is_open", False))
    next_open = getattr(clock, "next_open", None)
    next_open_date = None
    if next_open is not None:
        try:
            next_open_date = next_open.astimezone(ET).date()
        except Exception:
            try:
                next_open_date = next_open.date()
            except Exception:
                next_open_date = None
    if is_open:
        return True, f"market_open (is_open=true)"
    if next_open_date == now_et_date:
        return True, f"pre-open (next_open={next_open_date})"
    return False, f"closed (next_open={next_open_date}, today={now_et_date})"


def _get_positions(client) -> dict[str, float]:
    """Return {symbol: qty} for all open positions."""
    positions = client.get_all_positions()
    return {str(p.symbol): float(p.qty) for p in positions}


def _get_position_details(client) -> dict[str, dict]:
    """Return {symbol: {qty, current_price, avg_entry_price}} for open positions.

    current_price is the price the guard consults before submitting a sell.
    """
    positions = client.get_all_positions()
    out: dict[str, dict] = {}
    for p in positions:
        sym = str(p.symbol)
        try:
            cur = float(getattr(p, "current_price", 0.0) or 0.0)
        except (TypeError, ValueError):
            cur = 0.0
        try:
            entry = float(getattr(p, "avg_entry_price", 0.0) or 0.0)
        except (TypeError, ValueError):
            entry = 0.0
        out[sym] = {
            "qty": float(p.qty),
            "current_price": cur,
            "avg_entry_price": entry,
        }
    return out


def _previous_trading_day(day: date) -> date:
    return (pd.Timestamp(day) - BDay(1)).date()


def _expected_latest_daily_bar_date(now: datetime | None = None) -> date:
    now_et = (now or datetime.now(timezone.utc)).astimezone(ET)
    market_open = now_et.replace(hour=MARKET_OPEN[0], minute=MARKET_OPEN[1], second=0, microsecond=0)
    if now_et >= market_open:
        return now_et.date()
    return _previous_trading_day(now_et.date())


def _latest_daily_bar_is_fresh(df: pd.DataFrame, *, now: datetime | None = None) -> bool:
    if df is None or len(df) == 0 or "timestamp" not in df.columns:
        return False
    latest = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dropna()
    if latest.empty:
        return False
    latest_date = latest.max().tz_convert(ET).date()
    expected_latest = _expected_latest_daily_bar_date(now=now)
    return latest_date >= expected_latest


def _iter_symbol_batches(symbols: list[str], batch_size: int = DEFAULT_LIVE_BAR_BATCH_SIZE):
    size = max(int(batch_size), 1)
    for i in range(0, len(symbols), size):
        yield symbols[i : i + size]


def _get_latest_bars(
    symbols: list[str],
    n_days: int = 5,
    *,
    batch_size: int = DEFAULT_LIVE_BAR_BATCH_SIZE,
) -> dict[str, pd.DataFrame]:
    """Fetch last N calendar days of daily bars from Alpaca data API.

    Data API is read-only; falls through paper → prod keys so it still
    works when paper credentials are revoked (which blocks trading but
    shouldn't block scoring/dry-run).
    """
    try:
        import importlib
        env_real = importlib.import_module("env_real")
        from alpaca.data import StockHistoricalDataClient, StockBarsRequest, TimeFrame
        from alpaca.data.enums import DataFeed
        candidate_keys = [
            (getattr(env_real, "ALP_KEY_ID_PAPER", ""),
             getattr(env_real, "ALP_SECRET_KEY_PAPER", "")),
            (getattr(env_real, "ALP_KEY_ID_PROD", ""),
             getattr(env_real, "ALP_SECRET_KEY_PROD", "")),
            (getattr(env_real, "ALP_KEY_ID", ""),
             getattr(env_real, "ALP_SECRET_KEY", "")),
        ]
        data_client = None
        for key_id, secret in candidate_keys:
            if not key_id or not secret:
                continue
            try:
                test_client = StockHistoricalDataClient(key_id, secret)
                test_req = StockBarsRequest(
                    symbol_or_symbols="AAPL",
                    timeframe=TimeFrame.Day,
                    start=datetime.now(timezone.utc) - timedelta(days=7),
                    end=datetime.now(timezone.utc),
                    feed=DataFeed.IEX,
                )
                test_client.get_stock_bars(test_req)
                data_client = test_client
                break
            except Exception:
                continue
        if data_client is None:
            raise RuntimeError("no working Alpaca data credentials in env_real.py")
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=max(n_days * 2, 14))  # extra buffer for holidays
        # Alpaca uses BRK.B / BF.B; symbol list uses BRK-B / BF-B. Translate
        # on the wire then map back when stitching results.
        def _to_api(sym: str) -> str:
            return sym.replace("-", ".") if "-" in sym else sym

        api_to_local = {_to_api(s): s for s in symbols}

        result = {}
        for batch in _iter_symbol_batches(symbols, batch_size=batch_size):
            api_batch = [_to_api(s) for s in batch]
            req = StockBarsRequest(
                symbol_or_symbols=api_batch,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
                feed=DataFeed.IEX,
            )
            bars = data_client.get_stock_bars(req)
            for api_sym in api_batch:
                if api_sym in bars.data:
                    sym = api_to_local.get(api_sym, api_sym)
                    rows = [
                        {"timestamp": b.timestamp, "open": b.open, "high": b.high,
                         "low": b.low, "close": b.close, "volume": b.volume}
                        for b in bars.data[api_sym]
                    ]
                    if rows:
                        df = pd.DataFrame(rows)
                        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                        result[sym] = df
        return result
    except Exception as exc:
        logger.warning("Alpaca data API fetch failed: %s — will use local CSVs only", exc)
        return {}


# ── Feature computation for today ─────────────────────────────────────────────

def _extend_with_live_bars(
    local_df: pd.DataFrame,
    live_bars: dict[str, pd.DataFrame],
    symbol: str,
) -> pd.DataFrame:
    """Append any live bars newer than what's in local_df."""
    if symbol not in live_bars:
        return local_df
    new_df = live_bars[symbol]
    if local_df is None or len(local_df) == 0:
        return new_df
    latest_local = local_df["timestamp"].max()
    new_rows = new_df[new_df["timestamp"] > latest_local]
    if len(new_rows) == 0:
        return local_df
    combined = pd.concat([local_df, new_rows], ignore_index=True)
    return combined.sort_values("timestamp").reset_index(drop=True)


def score_all_symbols(
    symbols: list[str],
    data_root: Path,
    model: XGBStockModel | list[XGBStockModel],
    live_bars: dict[str, pd.DataFrame] | None = None,
    min_dollar_vol: float = 5e6,
    min_vol_20d: float = 0.0,
    now: datetime | None = None,
) -> pd.DataFrame:
    """Score all symbols for today's open-to-close trade.

    ``model`` may be a single ``XGBStockModel`` or a list of models — in the
    latter case per-model ``predict_scores`` are averaged (prob-mean blend,
    matches ``xgbnew/eval_pretrained.py`` default).

    Returns DataFrame with columns [symbol, score, spread_bps, last_close]
    sorted by score descending.
    """
    if live_bars is None:
        live_bars = {}

    models: list[XGBStockModel] = model if isinstance(model, list) else [model]
    if not models:
        raise ValueError("score_all_symbols: no models provided")

    rows = []

    for sym in symbols:
        local_df = _load_symbol_csv(sym, data_root)
        if local_df is None:
            continue
        df = _extend_with_live_bars(local_df, live_bars, sym)
        if len(df) < 60:
            continue
        if not _latest_daily_bar_is_fresh(df, now=now):
            continue

        feat = build_features_for_symbol(df, symbol=sym)
        # Get the LAST row — that's features for trading today
        last = feat.dropna(subset=DAILY_FEATURE_COLS[:5])
        if len(last) == 0:
            continue
        last_row = last.iloc[[-1]].copy()

        # Liquidity check
        dolvol = float(last_row["dolvol_20d_log"].iloc[0])
        if dolvol < np.log1p(min_dollar_vol):
            continue

        spread = float(last_row.get("spread_bps", pd.Series([25.0])).iloc[0])
        if spread > 30.0:  # skip illiquid
            continue

        # Realised-vol floor — matches BacktestConfig.min_vol_20d.
        # Drops dead-zone names LOBO flagged; strict-dominance at vol=0.10.
        if min_vol_20d > 0.0 and "vol_20d" in last_row.columns:
            v20 = float(last_row["vol_20d"].iloc[0])
            if not np.isfinite(v20) or v20 < min_vol_20d:
                continue

        # Add Chronos2 zeros
        for col in ["chronos_oc_return", "chronos_cc_return",
                    "chronos_pred_range", "chronos_available"]:
            if col not in last_row.columns:
                last_row[col] = 0.0

        if len(models) == 1:
            score = float(models[0].predict_scores(last_row).iloc[0])
        else:
            seed_scores = [float(m.predict_scores(last_row).iloc[0]) for m in models]
            score = float(np.mean(seed_scores))
        last_close = float(last_row["actual_close"].iloc[0])
        rows.append({
            "symbol": sym,
            "score": score,
            "spread_bps": spread,
            "last_close": last_close,
            "dolvol_20d_log": dolvol,
        })

    if not rows:
        return pd.DataFrame()

    df_scores = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return df_scores


# ── Main trading loop ─────────────────────────────────────────────────────────

def _wait_until(hour: int, minute: int, tz: ZoneInfo, poll_secs: float = 10.0) -> None:
    """Block until local clock reaches HH:MM in tz."""
    while True:
        now = datetime.now(tz)
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        secs = (target - now).total_seconds()
        if secs <= 0:
            return
        wait = min(secs, poll_secs)
        logger.debug("Waiting %.0fs until %02d:%02d %s", secs, hour, minute, tz)
        time.sleep(wait)


def _target_buy_qty(*, buy_notional: float, price: float) -> float:
    if not np.isfinite(buy_notional) or not np.isfinite(price) or buy_notional <= 0 or price <= 0:
        return 0.0
    qty = round(float(buy_notional) / float(price), 4)
    if qty <= 0:
        return 0.0
    return max(qty, MIN_FRACTIONAL_QTY)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols-file", type=Path,
                   default=REPO / "symbol_lists/stocks_wide_1000_v1.txt")
    p.add_argument("--data-root",    type=Path, default=REPO / "trainingdata")
    p.add_argument("--model-path",   type=Path, default=DEFAULT_MODEL_PATH,
                   help="Pre-trained XGBStockModel pickle (ignored when --model-paths is set)")
    p.add_argument("--model-paths",  type=str, default="",
                   help="Comma-separated list of pickles — ensemble mean-blend (predict_proba avg). "
                        "Takes precedence over --model-path when set.")
    p.add_argument("--top-n",        type=int,   default=2,
                   help="Number of stocks to buy per day")
    p.add_argument("--allocation",   type=float, default=0.25,
                   help="Fraction of portfolio to deploy (0.25 = 25%% per pick, shared)")
    p.add_argument("--min-score",    type=float, default=0.0,
                   help="Skip pick if blended predict_proba < min_score. "
                        "0.0 (default) = no filter. 0.55-0.70 gates on conviction. "
                        "If all top_n candidates fail, session holds cash.")
    p.add_argument("--commission-bps", type=float, default=10.0)
    p.add_argument("--min-dollar-vol", type=float, default=5e6)
    p.add_argument("--min-vol-20d", type=float, default=0.0,
                   help="Realised 20d annualised vol floor (e.g. 0.10). 0 "
                        "disables. Drops dead-zone symbols that LOBO flagged; "
                        "strict-dominance at 0.10 (deploy + stress36x).")
    p.add_argument("--hold-through", action="store_true",
                   help="If tomorrow's picks match today's held positions, skip the "
                        "sell-at-close + buy-at-open round-trip. Rotation now happens "
                        "at next-open (not close): sell only names that dropped out of "
                        "picks, buy only names that just entered. Saves 2×(fee+buffer) "
                        "per carried day and captures overnight drift. Backtest-validated "
                        "strict-dominance upgrade (docs/xgbnew_hold_through_20260419.md).")
    p.add_argument("--live",         action="store_true",
                   help="Use live Alpaca account (default: paper)")
    p.add_argument("--dry-run",      action="store_true",
                   help="Score only — do not place any orders")
    p.add_argument("--loop",         action="store_true",
                   help="Keep running (wait for next market open after each session)")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def _load_symbols(path: Path) -> list[str]:
    syms = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip().split("#", 1)[0].strip().upper()
        if s:
            syms.append(s)
    return syms


def _score_and_pick(symbols, data_root, model, args) -> pd.DataFrame:
    """Fetch live bars, score all symbols, apply min_score filter, return top-N picks."""
    print("[xgb-live] Fetching live bars from Alpaca...", flush=True)
    try:
        live_bars = _get_latest_bars(symbols, n_days=10)
    except Exception as exc:
        logger.warning("Live bar fetch failed: %s", exc)
        live_bars = {}

    print(f"[xgb-live] Scoring {len(symbols)} symbols...", flush=True)
    scores_df = score_all_symbols(
        symbols, data_root, model, live_bars,
        min_dollar_vol=args.min_dollar_vol,
        min_vol_20d=float(getattr(args, "min_vol_20d", 0.0) or 0.0),
    )

    if len(scores_df) == 0:
        return scores_df

    min_score = float(getattr(args, "min_score", 0.0) or 0.0)
    if min_score > 0.0:
        filtered = scores_df[scores_df["score"] >= min_score]
        print(f"[xgb-live] Conviction filter min_score={min_score:.2f}: "
              f"{len(filtered)}/{len(scores_df)} candidates pass "
              f"(top score={scores_df['score'].iloc[0]:.4f})", flush=True)
        if len(filtered) == 0:
            return filtered
        return filtered.head(args.top_n)
    return scores_df.head(args.top_n)


def run_session_hold_through(
    symbols: list[str],
    data_root: Path,
    model: XGBStockModel | list[XGBStockModel],
    client,
    args: argparse.Namespace,
) -> None:
    """Hold-through variant: rotate at NEXT-OPEN instead of selling at close.

    Flow per session (fired each trading morning):
      1. Gate on Alpaca calendar (/v2/clock).
      2. Score + pick top-N.
      3. Wait for open (9:30 ET).
      4. Query currently-held positions.
      5. If held_syms == pick_syms AND both non-empty → HOLD (no trades).
         This is the win case: skip 2×(fee+buffer) and earn overnight drift.
      6. Otherwise rotate:
           SELL (old ∖ new) — each goes through guard_sell_against_death_spiral.
           BUY (new ∖ old) — each records buy price via record_buy_price.
      7. Skip the 15:50 sell entirely. Positions carry overnight; next session
         decides whether to hold or rotate.

    No 15:50 sell means a DOWN day passes through without the "sell at close"
    mitigation — but the backtest validates this is net-positive because on
    top_n=1 the same pick repeating is common and saves round-trip cost.
    Regime gate / trading-calendar skip (weekend, holiday) does NOT flatten
    positions — but that's correct: we want to carry through the gap and
    continue holding if the signal agrees on the next trading day.
    """
    paper = not args.live
    today_str = date.today().isoformat()
    print(f"\n[xgb-live/hold-through] Session {today_str}  paper={paper}  "
          f"dry_run={args.dry_run}", flush=True)

    if client is not None and not args.dry_run:
        is_trading, reason = _is_today_trading_day(client)
        if not is_trading:
            print(f"[xgb-live] {today_str} is NOT a trading day ({reason}) — "
                  f"skipping session.", flush=True)
            return

    picks = _score_and_pick(symbols, data_root, model, args)
    if len(picks) == 0:
        print(f"[xgb-live] No picks today — holding current positions (if any).",
              flush=True)
        return

    pick_syms = set(picks["symbol"].astype(str))

    print(f"\n[xgb-live/hold-through] Top-{args.top_n} picks for {today_str}:")
    for _, row in picks.iterrows():
        print(f"  {row['symbol']:<8}  score={row['score']:.4f}  "
              f"last_close=${row['last_close']:.2f}  spread={row['spread_bps']:.1f}bps")

    if args.dry_run:
        print("[xgb-live/hold-through] DRY RUN — no orders placed.", flush=True)
        return

    now_et = datetime.now(ET)
    if now_et.hour < MARKET_OPEN[0] or (now_et.hour == MARKET_OPEN[0] and
                                         now_et.minute < MARKET_OPEN[1]):
        print(f"[xgb-live] Waiting for {MARKET_OPEN[0]:02d}:{MARKET_OPEN[1]:02d} ET...",
              flush=True)
        _wait_until(MARKET_OPEN[0], MARKET_OPEN[1], ET)

    position_details = _get_position_details(client)
    held_syms = {s for s, det in position_details.items() if det["qty"] > 0}

    if held_syms == pick_syms:
        print(f"[xgb-live/hold-through] HOLD — picks unchanged "
              f"({sorted(pick_syms)}). Skipping round-trip.", flush=True)
        return

    to_sell = held_syms - pick_syms
    to_buy = pick_syms - held_syms
    print(f"[xgb-live/hold-through] Rotation: sell={sorted(to_sell)}  "
          f"buy={sorted(to_buy)}  keep={sorted(held_syms & pick_syms)}", flush=True)

    # HARD RULE #3: every sell must pass through guard_sell_against_death_spiral.
    from src.alpaca_singleton import (
        guard_sell_against_death_spiral,
        record_buy_price,
    )

    # SELL dropped-out names FIRST (frees up buying power before buys).
    for sym in sorted(to_sell):
        det = position_details[sym]
        qty = det["qty"]
        if qty <= 0:
            continue
        current_price = det["current_price"] or det["avg_entry_price"]
        if current_price <= 0:
            logger.error("SELL skipped for %s — no usable price for death-spiral "
                         "guard", sym)
            continue
        guard_sell_against_death_spiral(sym, "sell", float(current_price))
        try:
            order = _submit_market_order(client, symbol=sym, qty=abs(qty), side="sell")
            print(f"  SELL {sym:<8}  qty={qty:.2f}  px={current_price:.2f}  "
                  f"order_id={getattr(order, 'id', '?')}", flush=True)
        except Exception as exc:
            logger.error("SELL failed for %s: %s", sym, exc)

    # BUY new picks.
    account = _get_account(client)
    portfolio_value = float(getattr(account, "portfolio_value", 0.0) or 0.0)
    buy_notional = portfolio_value * args.allocation / args.top_n
    print(f"[xgb-live/hold-through] BUY notional=${buy_notional:,.0f}/pick "
          f"(portfolio=${portfolio_value:,.0f})", flush=True)

    for _, row in picks.iterrows():
        sym = str(row["symbol"])
        if sym not in to_buy:
            continue
        price = float(row["last_close"])
        if price <= 0:
            logger.warning("Skipping BUY %s — invalid last_close %s", sym, price)
            continue
        qty = _target_buy_qty(buy_notional=buy_notional, price=price)
        if qty <= 0:
            logger.warning("Skipping BUY %s — invalid target qty", sym)
            continue
        try:
            order = _submit_market_order(client, symbol=sym, qty=qty, side="buy")
            order_id = str(getattr(order, "id", "") or "")
            print(f"  BUY  {sym:<8}  qty={qty:.2f}  ~${qty*price:,.0f}  "
                  f"order_id={order_id or '?'}", flush=True)
            fill_px = _poll_filled_avg_price(client, order_id) if order_id else None
            recorded = fill_px if (fill_px and fill_px > 0) else price
            try:
                record_buy_price(sym, float(recorded))
                src = "fill" if fill_px else "last_close"
                print(f"       recorded buy_price={recorded:.4f} ({src}) for guard",
                      flush=True)
            except Exception as rec_exc:
                logger.warning("record_buy_price failed for %s: %s", sym, rec_exc)
        except Exception as exc:
            logger.error("BUY failed for %s: %s", sym, exc)

    print(f"[xgb-live/hold-through] Rotation complete: "
          f"sold {len(to_sell)}, bought {len(to_buy)}, held across "
          f"{len(held_syms & pick_syms)}.", flush=True)


def run_session(
    symbols: list[str],
    data_root: Path,
    model: XGBStockModel | list[XGBStockModel],
    client,
    args: argparse.Namespace,
) -> None:
    """Execute one full trading session (score → buy → sell)."""
    if getattr(args, "hold_through", False):
        return run_session_hold_through(symbols, data_root, model, client, args)

    paper = not args.live
    today_str = date.today().isoformat()
    print(f"\n[xgb-live] Session {today_str}  paper={paper}  dry_run={args.dry_run}", flush=True)

    # ── Trading-day gate ──────────────────────────────────────────────────────
    # Alpaca accepts DAY orders submitted outside RTH and queues them for the
    # next open — which is a footgun: a Saturday BUY fills at Monday's open
    # BEFORE the daemon has re-scored with fresh data. Always gate the whole
    # session on Alpaca's own calendar.
    if client is not None and not args.dry_run:
        is_trading, reason = _is_today_trading_day(client)
        if not is_trading:
            print(f"[xgb-live] {today_str} is NOT a trading day ({reason}) — "
                  f"skipping session.", flush=True)
            return

    # ── Score ─────────────────────────────────────────────────────────────────
    print("[xgb-live] Fetching live bars from Alpaca...", flush=True)
    # Only need recent data — fetch last 10 bars to extend local CSVs
    try:
        live_bars = _get_latest_bars(symbols, n_days=10)
    except Exception as exc:
        logger.warning("Live bar fetch failed: %s", exc)
        live_bars = {}

    print(f"[xgb-live] Scoring {len(symbols)} symbols...", flush=True)
    scores_df = score_all_symbols(
        symbols, data_root, model, live_bars,
        min_dollar_vol=args.min_dollar_vol,
        min_vol_20d=float(getattr(args, "min_vol_20d", 0.0) or 0.0),
    )

    if len(scores_df) == 0:
        print("[xgb-live] ERROR: No scoreable symbols today.", file=sys.stderr)
        return

    min_score = float(getattr(args, "min_score", 0.0) or 0.0)
    if min_score > 0.0:
        filtered = scores_df[scores_df["score"] >= min_score]
        print(f"[xgb-live] Conviction filter min_score={min_score:.2f}: "
              f"{len(filtered)}/{len(scores_df)} candidates pass "
              f"(top score={scores_df['score'].iloc[0]:.4f})", flush=True)
        if len(filtered) == 0:
            print(f"[xgb-live] NO pick meets min_score={min_score:.2f} — "
                  f"holding cash for {today_str}.", flush=True)
            return
        picks = filtered.head(args.top_n)
    else:
        picks = scores_df.head(args.top_n)

    print(f"\n[xgb-live] Top-{args.top_n} picks for {today_str}:")
    for _, row in picks.iterrows():
        print(f"  {row['symbol']:<8}  score={row['score']:.4f}  "
              f"last_close=${row['last_close']:.2f}  spread={row['spread_bps']:.1f}bps")

    if args.dry_run:
        print("[xgb-live] DRY RUN — no orders placed.", flush=True)
        return

    # ── Wait for market open ──────────────────────────────────────────────────
    now_et = datetime.now(ET)
    if now_et.hour < MARKET_OPEN[0] or (now_et.hour == MARKET_OPEN[0] and
                                         now_et.minute < MARKET_OPEN[1]):
        print(f"[xgb-live] Waiting for {MARKET_OPEN[0]:02d}:{MARKET_OPEN[1]:02d} ET...", flush=True)
        _wait_until(MARKET_OPEN[0], MARKET_OPEN[1], ET)

    # ── Buy ───────────────────────────────────────────────────────────────────
    account = _get_account(client)
    portfolio_value = float(getattr(account, "portfolio_value", 0.0) or 0.0)
    buy_notional = portfolio_value * args.allocation / args.top_n

    print(f"\n[xgb-live] Placing BUY orders  portfolio=${portfolio_value:,.0f}  "
          f"notional/pick=${buy_notional:,.0f}", flush=True)

    # HARD RULE #3 (CLAUDE.md): record each buy's actual fill price so future
    # sells are vetted against the real cost basis by the death-spiral guard.
    from src.alpaca_singleton import record_buy_price

    for _, row in picks.iterrows():
        sym = str(row["symbol"])
        price = float(row["last_close"])
        if price <= 0:
            logger.warning("Skipping %s — invalid last_close %s", sym, price)
            continue
        qty = _target_buy_qty(buy_notional=buy_notional, price=price)
        if qty <= 0:
            logger.warning("Skipping %s — invalid target qty for price=%s notional=%s", sym, price, buy_notional)
            continue
        try:
            order = _submit_market_order(client, symbol=sym, qty=qty, side="buy")
            order_id = str(getattr(order, "id", "") or "")
            print(f"  BUY  {sym:<8}  qty={qty:.2f}  ~${qty*price:,.0f}  "
                  f"order_id={order_id or '?'}", flush=True)
            fill_px = _poll_filled_avg_price(client, order_id) if order_id else None
            recorded = fill_px if (fill_px and fill_px > 0) else price
            try:
                record_buy_price(sym, float(recorded))
                src = "fill" if fill_px else "last_close"
                print(f"       recorded buy_price={recorded:.4f} ({src}) for guard",
                      flush=True)
            except Exception as rec_exc:
                logger.warning("record_buy_price failed for %s: %s", sym, rec_exc)
        except Exception as exc:
            logger.error("BUY failed for %s: %s", sym, exc)

    # ── Wait for close ────────────────────────────────────────────────────────
    print(f"\n[xgb-live] Waiting for {MARKET_CLOSE[0]:02d}:{MARKET_CLOSE[1]:02d} ET to sell...",
          flush=True)
    _wait_until(MARKET_CLOSE[0], MARKET_CLOSE[1], ET)

    # ── Sell all positions ────────────────────────────────────────────────────
    # HARD RULE #3: every sell passes through guard_sell_against_death_spiral
    # with the current quote; if we're >50 bps below the recorded buy, the
    # guard raises RuntimeError and crashes the loop (supervisor autorestart).
    from src.alpaca_singleton import guard_sell_against_death_spiral

    position_details = _get_position_details(client)
    xgb_positions = {sym: det for sym, det in position_details.items()
                     if sym in picks["symbol"].values}

    print(f"\n[xgb-live] Placing SELL orders for {len(xgb_positions)} positions", flush=True)
    for sym, det in xgb_positions.items():
        qty = det["qty"]
        if qty <= 0:
            continue
        current_price = det["current_price"]
        if current_price <= 0:
            current_price = det["avg_entry_price"]
        if current_price <= 0:
            logger.error("SELL skipped for %s — no current_price or avg_entry_price "
                         "available; cannot invoke death-spiral guard safely", sym)
            continue
        # Guard raises RuntimeError on death-spiral sells → propagate (crash).
        guard_sell_against_death_spiral(sym, "sell", float(current_price))
        try:
            order = _submit_market_order(client, symbol=sym, qty=abs(qty), side="sell")
            print(f"  SELL {sym:<8}  qty={qty:.2f}  px={current_price:.2f}  "
                  f"order_id={getattr(order, 'id', '?')}", flush=True)
        except Exception as exc:
            logger.error("SELL failed for %s: %s", sym, exc)

    print(f"[xgb-live] Session {today_str} complete.", flush=True)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    paper = not args.live

    # ── Singleton guard ────────────────────────────────────────────────────────
    if not paper and not args.dry_run:
        from src.alpaca_singleton import enforce_live_singleton
        _lock = enforce_live_singleton(
            service_name="xgb_live_trader",
            account_name="alpaca_live_writer",
        )

    # ── Load model(s) ─────────────────────────────────────────────────────────
    if args.model_paths.strip():
        paths = [Path(p.strip()) for p in args.model_paths.split(",") if p.strip()]
        for mp in paths:
            if not mp.exists():
                print(f"ERROR: Ensemble model not found at {mp}", file=sys.stderr)
                return 1
        print(f"[xgb-live] Loading ensemble of {len(paths)} models", flush=True)
        for mp in paths:
            print(f"[xgb-live]   - {mp}", flush=True)
        model: XGBStockModel | list[XGBStockModel] = [XGBStockModel.load(mp) for mp in paths]
    else:
        if not args.model_path.exists():
            print(f"ERROR: Model not found at {args.model_path}", file=sys.stderr)
            print("Run train_alltrain.py to create it, or set --model-paths for an ensemble.",
                  file=sys.stderr)
            return 1
        print(f"[xgb-live] Loading model from {args.model_path}", flush=True)
        model = XGBStockModel.load(args.model_path)

    # ── Load symbols ───────────────────────────────────────────────────────────
    symbols = _load_symbols(args.symbols_file)
    print(f"[xgb-live] {len(symbols)} symbols  paper={paper}  top_n={args.top_n}  "
          f"allocation={args.allocation:.0%}", flush=True)

    # ── Build Alpaca client ────────────────────────────────────────────────────
    if not args.dry_run:
        client = _build_trading_client(paper=paper)
        acct = _get_account(client)
        print(f"[xgb-live] Account equity=${float(getattr(acct, 'equity', 0)):,.0f}  "
              f"buying_power=${float(getattr(acct, 'buying_power', 0)):,.0f}", flush=True)
    else:
        client = None

    # ── Run session(s) ─────────────────────────────────────────────────────────
    while True:
        run_session(symbols, args.data_root, model, client, args)

        if not args.loop:
            break

        # Wait until next market morning (~9:20 ET next business day)
        now_et = datetime.now(ET)
        next_day = now_et.date() + timedelta(days=1)
        # Skip weekends
        while next_day.weekday() >= 5:
            next_day += timedelta(days=1)
        next_open = datetime(next_day.year, next_day.month, next_day.day,
                             9, 20, 0, tzinfo=ET)
        wait_secs = (next_open - datetime.now(ET)).total_seconds()
        if wait_secs > 0:
            print(f"[xgb-live] Next session at {next_open.isoformat()}  "
                  f"(sleeping {wait_secs/3600:.1f}h)", flush=True)
            time.sleep(wait_secs)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
