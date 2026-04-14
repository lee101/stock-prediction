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


def _get_account(client):
    return client.get_account()


def _get_positions(client) -> dict[str, float]:
    """Return {symbol: qty} for all open positions."""
    positions = client.get_all_positions()
    return {str(p.symbol): float(p.qty) for p in positions}


def _get_latest_bars(symbols: list[str], n_days: int = 5) -> dict[str, pd.DataFrame]:
    """Fetch last N calendar days of daily bars from Alpaca data API."""
    try:
        import importlib
        env_real = importlib.import_module("env_real")
        key_id = getattr(env_real, "ALP_KEY_ID_PAPER", getattr(env_real, "ALP_KEY_ID", ""))
        secret = getattr(env_real, "ALP_SECRET_KEY_PAPER", getattr(env_real, "ALP_SECRET_KEY", ""))
        from alpaca.data import StockHistoricalDataClient, StockBarsRequest, TimeFrame
        from alpaca.data.enums import DataFeed
        data_client = StockHistoricalDataClient(key_id, secret)
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=max(n_days * 2, 14))  # extra buffer for holidays
        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            feed=DataFeed.IEX,
        )
        bars = data_client.get_stock_bars(req)
        result = {}
        for sym in symbols:
            if sym in bars.data:
                rows = [
                    {"timestamp": b.timestamp, "open": b.open, "high": b.high,
                     "low": b.low, "close": b.close, "volume": b.volume}
                    for b in bars.data[sym]
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
    model: XGBStockModel,
    live_bars: dict[str, pd.DataFrame] | None = None,
    min_dollar_vol: float = 5e6,
) -> pd.DataFrame:
    """Score all symbols for today's open-to-close trade.

    Returns DataFrame with columns [symbol, score, spread_bps, last_close]
    sorted by score descending.
    """
    if live_bars is None:
        live_bars = {}

    today = date.today()
    rows = []

    for sym in symbols:
        local_df = _load_symbol_csv(sym, data_root)
        if local_df is None:
            continue
        df = _extend_with_live_bars(local_df, live_bars, sym)
        if len(df) < 60:
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

        # Add Chronos2 zeros
        for col in ["chronos_oc_return", "chronos_cc_return",
                    "chronos_pred_range", "chronos_available"]:
            if col not in last_row.columns:
                last_row[col] = 0.0

        score = float(model.predict_scores(last_row).iloc[0])
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


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols-file", type=Path,
                   default=REPO / "symbol_lists/stocks_wide_1000_v1.txt")
    p.add_argument("--data-root",    type=Path, default=REPO / "trainingdata")
    p.add_argument("--model-path",   type=Path, default=DEFAULT_MODEL_PATH,
                   help="Pre-trained XGBStockModel pickle (from eval_multiwindow --model-save-path)")
    p.add_argument("--top-n",        type=int,   default=2,
                   help="Number of stocks to buy per day")
    p.add_argument("--allocation",   type=float, default=0.25,
                   help="Fraction of portfolio to deploy (0.25 = 25%% per pick, shared)")
    p.add_argument("--commission-bps", type=float, default=10.0)
    p.add_argument("--min-dollar-vol", type=float, default=5e6)
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


def run_session(
    symbols: list[str],
    data_root: Path,
    model: XGBStockModel,
    client,
    args: argparse.Namespace,
) -> None:
    """Execute one full trading session (score → buy → sell)."""
    paper = not args.live
    today_str = date.today().isoformat()
    print(f"\n[xgb-live] Session {today_str}  paper={paper}  dry_run={args.dry_run}", flush=True)

    # ── Score ─────────────────────────────────────────────────────────────────
    print("[xgb-live] Fetching live bars from Alpaca...", flush=True)
    # Only need recent data — fetch last 10 bars to extend local CSVs
    try:
        live_bars = _get_latest_bars(symbols[:200], n_days=10)  # sample to avoid rate limit
    except Exception as exc:
        logger.warning("Live bar fetch failed: %s", exc)
        live_bars = {}

    print(f"[xgb-live] Scoring {len(symbols)} symbols...", flush=True)
    scores_df = score_all_symbols(
        symbols, data_root, model, live_bars, min_dollar_vol=args.min_dollar_vol
    )

    if len(scores_df) == 0:
        print("[xgb-live] ERROR: No scoreable symbols today.", file=sys.stderr)
        return

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

    for _, row in picks.iterrows():
        sym = str(row["symbol"])
        price = float(row["last_close"])
        if price <= 0:
            logger.warning("Skipping %s — invalid last_close %s", sym, price)
            continue
        qty = max(1.0, round(buy_notional / price, 4))
        try:
            order = _submit_market_order(client, symbol=sym, qty=qty, side="buy")
            print(f"  BUY  {sym:<8}  qty={qty:.2f}  ~${qty*price:,.0f}  "
                  f"order_id={getattr(order, 'id', '?')}", flush=True)
        except Exception as exc:
            logger.error("BUY failed for %s: %s", sym, exc)

    # ── Wait for close ────────────────────────────────────────────────────────
    print(f"\n[xgb-live] Waiting for {MARKET_CLOSE[0]:02d}:{MARKET_CLOSE[1]:02d} ET to sell...",
          flush=True)
    _wait_until(MARKET_CLOSE[0], MARKET_CLOSE[1], ET)

    # ── Sell all positions ────────────────────────────────────────────────────
    positions = _get_positions(client)
    xgb_positions = {sym: qty for sym, qty in positions.items()
                     if sym in picks["symbol"].values}

    print(f"\n[xgb-live] Placing SELL orders for {len(xgb_positions)} positions", flush=True)
    for sym, qty in xgb_positions.items():
        if qty <= 0:
            continue
        try:
            order = _submit_market_order(client, symbol=sym, qty=abs(qty), side="sell")
            print(f"  SELL {sym:<8}  qty={qty:.2f}  "
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

    # ── Load model ─────────────────────────────────────────────────────────────
    if not args.model_path.exists():
        print(f"ERROR: Model not found at {args.model_path}", file=sys.stderr)
        print("Run eval_multiwindow.py first with --model-save-path to create it.", file=sys.stderr)
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
