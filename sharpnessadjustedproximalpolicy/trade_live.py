#!/usr/bin/env python3
"""SAPP Portfolio Live Trader.

Runs hourly per-symbol inference on Binance, generates portfolio signals,
executes limit orders with sqrt_sortino allocation weights.

Usage:
    # Paper mode (log signals, no orders)
    python -m sharpnessadjustedproximalpolicy.trade_live --paper

    # Live mode
    python -m sharpnessadjustedproximalpolicy.trade_live --live

    # Daemon mode (hourly loop)
    python -m sharpnessadjustedproximalpolicy.trade_live --live --daemon
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sharpnessadjustedproximalpolicy.live_signal import SAPPPortfolioSignal, SymbolSignal

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SYMBOL_BINANCE_MAP = {
    "BTCUSD": ("BTCFDUSD", "BTC", "FDUSD"),
    "ETHUSD": ("ETHFDUSD", "ETH", "FDUSD"),
    "SOLUSD": ("SOLUSDT", "SOL", "USDT"),
    "DOGEUSD": ("DOGEUSDT", "DOGE", "USDT"),
    "LINKUSD": ("LINKUSDT", "LINK", "USDT"),
    "ADAUSD": ("ADAUSDT", "ADA", "USDT"),
    "ALGOUSD": ("ALGOUSDT", "ALGO", "USDT"),
    "DOTUSD": ("DOTUSDT", "DOT", "USDT"),
    "ATOMUSD": ("ATOMUSDT", "ATOM", "USDT"),
    "FILUSD": ("FILUSDT", "FIL", "USDT"),
    "ICPUSD": ("ICPUSDT", "ICP", "USDT"),
    "BCHUSD": ("BCHUSDT", "BCH", "USDT"),
    "ARBUSD": ("ARBUSDT", "ARB", "USDT"),
    "APTUSD": ("APTUSDT", "APT", "USDT"),
    "INJUSD": ("INJUSDT", "INJ", "USDT"),
    "NEARUSD": ("NEARUSDT", "NEAR", "USDT"),
    "OPUSD": ("OPUSDT", "OP", "USDT"),
    "SEIUSD": ("SEIUSDT", "SEI", "USDT"),
    "PEPEUSD": ("PEPEUSDT", "PEPE", "USDT"),
    "POLUSD": ("POLUSDT", "POL", "USDT"),
    "XLMUSD": ("XLMUSDT", "XLM", "USDT"),
    "TIAUSD": ("TIAUSDT", "TIA", "USDT"),
    "TAOUSD": ("TAOUSDT", "TAO", "USDT"),
    "RENDERUSD": ("RENDERUSDT", "RENDER", "USDT"),
    "XRPUSD": ("XRPUSDT", "XRP", "USDT"),
    "BNBUSD": ("BNBUSDT", "BNB", "USDT"),
    "LTCUSD": ("LTCUSDT", "LTC", "USDT"),
}

STATE_FILE = Path("sharpnessadjustedproximalpolicy/sapp_live_state.json")
EVENTS_FILE = Path("sharpnessadjustedproximalpolicy/sapp_events.jsonl")


def _fetch_klines(client, pair: str, interval: str = "1h", limit: int = 250) -> pd.DataFrame:
    """Fetch hourly klines from Binance."""
    klines = client.get_klines(symbol=pair, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ])
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def build_features(klines_df: pd.DataFrame, chronos_cache: Optional[dict] = None,
                   horizons: tuple = (1, 24)) -> pd.DataFrame:
    """Build feature frame from klines + optional chronos cache."""
    df = klines_df.copy()

    # Base features
    df["return_1h"] = df["close"].pct_change(1)
    df["return_4h"] = df["close"].pct_change(4)
    df["return_24h"] = df["close"].pct_change(24)
    df["volatility_24h"] = df["return_1h"].rolling(24).std()
    df["range_pct"] = (df["high"] - df["low"]).abs() / df["close"].replace(0.0, np.nan)

    # Volume z-score
    window = min(168, len(df))
    vol_mean = df["volume"].rolling(window).mean()
    vol_std = df["volume"].rolling(window).std().replace(0.0, 1.0)
    df["volume_z"] = (df["volume"] - vol_mean) / vol_std

    # Cyclic time
    hours = df["timestamp"].dt.hour
    dow = df["timestamp"].dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    # Reference close
    df["reference_close"] = df["close"]

    # Chronos features
    for h in horizons:
        if chronos_cache and f"h{h}" in chronos_cache:
            cache_df = chronos_cache[f"h{h}"].copy()
            # Cache uses generic names; rename to horizon-specific
            rename_map = {}
            for col in cache_df.columns:
                if col.startswith("predicted_") and not col.endswith(f"_h{h}"):
                    rename_map[col] = f"{col}_h{h}"
            if rename_map:
                cache_df = cache_df.rename(columns=rename_map)
            merge_cols = ["timestamp"] + [c for c in cache_df.columns
                                          if c.startswith(f"predicted_") and c.endswith(f"_h{h}")]
            merge_cols = [c for c in merge_cols if c in cache_df.columns]
            df = df.merge(cache_df[merge_cols], on="timestamp", how="left")

        # Fill missing with neutral (close/high/low)
        for prefix, fallback in [("close_p50", "close"), ("high_p50", "high"), ("low_p50", "low")]:
            col = f"predicted_{prefix}_h{h}"
            if col not in df.columns:
                df[col] = df[fallback]
            else:
                df[col] = df[col].fillna(df[fallback])

        ref = df["close"].replace(0.0, np.nan)
        df[f"chronos_close_delta_h{h}"] = (df[f"predicted_close_p50_h{h}"] - ref) / ref
        df[f"chronos_high_delta_h{h}"] = (df[f"predicted_high_p50_h{h}"] - ref) / ref
        df[f"chronos_low_delta_h{h}"] = (df[f"predicted_low_p50_h{h}"] - ref) / ref

    # Chronos high/low for action decoding (max/min across horizons)
    ch_high_cols = [f"predicted_high_p50_h{h}" for h in horizons if f"predicted_high_p50_h{h}" in df.columns]
    ch_low_cols = [f"predicted_low_p50_h{h}" for h in horizons if f"predicted_low_p50_h{h}" in df.columns]
    df["chronos_high"] = df[ch_high_cols].max(axis=1) if ch_high_cols else df["high"]
    df["chronos_low"] = df[ch_low_cols].min(axis=1) if ch_low_cols else df["low"]

    return df


def load_chronos_cache(symbol: str, horizons: tuple = (1, 24)) -> dict:
    """Load chronos forecast cache for a symbol."""
    cache_root = Path("binanceneural/forecast_cache")
    result = {}
    for h in horizons:
        path = cache_root / f"h{h}" / f"{symbol}.parquet"
        if path.exists():
            try:
                df = pd.read_parquet(path)
                if "timestamp" not in df.columns and "issued_at" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["issued_at"], utc=True)
                result[f"h{h}"] = df
            except Exception as e:
                logger.warning(f"Failed to load cache {path}: {e}")
    return result


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"positions": {}, "equity": 0, "peak_equity": 0, "trades": []}


def save_state(state: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def log_event(event: dict):
    EVENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    event["ts"] = datetime.now(timezone.utc).isoformat()
    with open(EVENTS_FILE, "a") as f:
        f.write(json.dumps(event, default=str) + "\n")


def run_cycle(
    signal_gen: SAPPPortfolioSignal,
    client,
    dry_run: bool = True,
    max_position_pct: float = 0.10,
    min_intensity: float = 0.3,
):
    """Run one hourly inference + execution cycle."""
    now = datetime.now(timezone.utc)
    symbols = signal_gen.get_symbols()
    weights = signal_gen.get_weights()

    # Get account equity
    equity = 0.0
    if client and not dry_run:
        try:
            info = client.get_account()
            equity = sum(float(b["free"]) + float(b["locked"])
                        for b in info["balances"]
                        if b["asset"] in ("USDT", "FDUSD", "BUSD"))
            # Add position values
            for b in info["balances"]:
                amt = float(b["free"]) + float(b["locked"])
                if amt > 0 and b["asset"] not in ("USDT", "FDUSD", "BUSD"):
                    # Approximate with ticker
                    try:
                        ticker = client.get_symbol_ticker(symbol=b["asset"] + "USDT")
                        equity += amt * float(ticker["price"])
                    except Exception:
                        pass
        except Exception as e:
            logger.error(f"Failed to get account: {e}")
            equity = 3000.0  # fallback
    else:
        equity = 3000.0  # paper mode

    logger.info(f"Cycle start: {len(symbols)} symbols, equity=${equity:.0f}")

    signals = {}
    for sym in symbols:
        if sym not in SYMBOL_BINANCE_MAP:
            continue
        pair, base, quote = SYMBOL_BINANCE_MAP[sym]

        try:
            # Fetch klines
            if client:
                klines_df = _fetch_klines(client, pair, limit=250)
            else:
                # Paper mode: use cached data
                data_path = Path("trainingdatahourly/crypto") / f"{sym}.csv"
                if not data_path.exists():
                    continue
                klines_df = pd.read_csv(data_path, parse_dates=["timestamp"])
                klines_df = klines_df.tail(250)

            # Load chronos cache
            model_loader = signal_gen.models[sym]
            chronos = load_chronos_cache(sym, model_loader.horizons)

            # Build features
            feat_df = build_features(klines_df, chronos, model_loader.horizons)
            feat_cols = model_loader.feature_columns
            feat_df = feat_df.dropna(subset=[c for c in feat_cols if c in feat_df.columns])

            if len(feat_df) < model_loader.seq_len:
                logger.warning(f"{sym}: insufficient data ({len(feat_df)} < {model_loader.seq_len})")
                continue

            # Get last seq_len rows
            window = feat_df.tail(model_loader.seq_len)
            features_np = window[[c for c in feat_cols if c in window.columns]].to_numpy(dtype=np.float32)

            # Normalize
            if model_loader.normalizer:
                features_np = model_loader.normalizer.transform(features_np)

            features_t = torch.from_numpy(features_np)
            ref_close = float(window["reference_close"].iloc[-1])
            ch_high = float(window["chronos_high"].iloc[-1])
            ch_low = float(window["chronos_low"].iloc[-1])

            # Generate signal
            sig = signal_gen.generate_signal(sym, features_t, ref_close, ch_high, ch_low)
            if sig:
                signals[sym] = sig

        except Exception as e:
            logger.error(f"{sym}: {e}")
            continue

    # Log signals
    buy_signals = []
    sell_signals = []
    for sym, sig in signals.items():
        if sig.buy_intensity > min_intensity:
            alloc = equity * weights.get(sym, 0) * max_position_pct * sig.buy_intensity
            buy_signals.append({
                "symbol": sym,
                "price": round(sig.buy_price, 8),
                "intensity": round(sig.buy_intensity, 3),
                "alloc_usd": round(alloc, 2),
                "weight": round(sig.weight, 4),
            })
        if sig.sell_intensity > min_intensity:
            sell_signals.append({
                "symbol": sym,
                "price": round(sig.sell_price, 8),
                "intensity": round(sig.sell_intensity, 3),
                "weight": round(sig.weight, 4),
            })

    # Sort by intensity
    buy_signals.sort(key=lambda x: x["intensity"], reverse=True)
    sell_signals.sort(key=lambda x: x["intensity"], reverse=True)

    logger.info(f"Signals: {len(buy_signals)} buys, {len(sell_signals)} sells out of {len(signals)} symbols")

    # Log top signals
    for b in buy_signals[:5]:
        logger.info(f"  BUY  {b['symbol']:<12} price={b['price']:<12.6f} int={b['intensity']:.2f} alloc=${b['alloc_usd']:.0f}")
    for s in sell_signals[:5]:
        logger.info(f"  SELL {s['symbol']:<12} price={s['price']:<12.6f} int={s['intensity']:.2f}")

    log_event({
        "type": "cycle",
        "equity": equity,
        "n_symbols": len(signals),
        "n_buys": len(buy_signals),
        "n_sells": len(sell_signals),
        "buys": buy_signals[:10],
        "sells": sell_signals[:10],
        "dry_run": dry_run,
    })

    # Execute orders (only in live mode)
    if not dry_run and client:
        for b in buy_signals[:3]:  # Max 3 buys per cycle
            sym = b["symbol"]
            pair, base, quote = SYMBOL_BINANCE_MAP[sym]
            try:
                order = client.create_order(
                    symbol=pair,
                    side="BUY",
                    type="LIMIT",
                    timeInForce="GTC",
                    quantity=round(b["alloc_usd"] / b["price"], 4),
                    price=str(round(b["price"], 8)),
                )
                logger.info(f"ORDER {sym}: {order['orderId']} BUY {order['origQty']} @ {order['price']}")
                log_event({"type": "order", "symbol": sym, "side": "BUY", "order_id": order["orderId"],
                          "qty": order["origQty"], "price": order["price"]})
            except Exception as e:
                logger.error(f"Order failed {sym}: {e}")
                log_event({"type": "order_failed", "symbol": sym, "side": "BUY", "error": str(e)})

    return {"buys": buy_signals, "sells": sell_signals, "equity": equity}


def main():
    parser = argparse.ArgumentParser(description="SAPP Portfolio Live Trader")
    parser.add_argument("--leaderboard", default=None, help="Best leaderboard CSV")
    parser.add_argument("--min-sortino", type=float, default=83.0)
    parser.add_argument("--paper", action="store_true", help="Paper trading (no orders)")
    parser.add_argument("--live", action="store_true", help="Live trading")
    parser.add_argument("--daemon", action="store_true", help="Run hourly loop")
    parser.add_argument("--max-position-pct", type=float, default=0.10)
    parser.add_argument("--min-intensity", type=float, default=0.3)
    parser.add_argument("--interval-s", type=int, default=3600, help="Daemon interval seconds")
    args = parser.parse_args()

    # Find latest leaderboard
    sap_dir = Path("sharpnessadjustedproximalpolicy")
    if args.leaderboard:
        lb_path = Path(args.leaderboard)
    else:
        lbs = sorted(sap_dir.glob("best_leaderboard_*.csv"))
        lb_path = lbs[-1] if lbs else None
    if not lb_path or not lb_path.exists():
        logger.error("No leaderboard found")
        return

    logger.info(f"Loading models from {lb_path.name}")
    signal_gen = SAPPPortfolioSignal(
        leaderboard_path=lb_path,
        min_sortino=args.min_sortino,
        alloc_method="sqrt_sortino",
        device="cuda",
    )
    logger.info(f"Loaded {len(signal_gen.models)} symbols")

    # Binance client
    client = None
    dry_run = args.paper or not args.live
    if args.live:
        try:
            from binance.client import Client as BinanceClient
            import os
            client = BinanceClient(
                os.environ.get("BINANCE_API_KEY", ""),
                os.environ.get("BINANCE_API_SECRET", ""),
            )
            logger.info("Binance client connected")
        except Exception as e:
            logger.error(f"Binance client failed: {e}")
            dry_run = True

    if dry_run:
        logger.info("PAPER MODE - no orders will be placed")

    # Run
    if args.daemon:
        logger.info(f"Daemon mode: cycle every {args.interval_s}s")
        while True:
            try:
                result = run_cycle(signal_gen, client, dry_run=dry_run,
                                   max_position_pct=args.max_position_pct,
                                   min_intensity=args.min_intensity)
                logger.info(f"Cycle done: {result['n_buys'] if isinstance(result, dict) and 'n_buys' in result else len(result.get('buys',[]))} buys")
            except Exception as e:
                logger.error(f"Cycle error: {e}")
                import traceback; traceback.print_exc()
            time.sleep(args.interval_s)
    else:
        result = run_cycle(signal_gen, client, dry_run=dry_run,
                           max_position_pct=args.max_position_pct,
                           min_intensity=args.min_intensity)


if __name__ == "__main__":
    main()
