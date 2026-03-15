#!/usr/bin/env python3
"""
Production daily trading bot for mixed stock+crypto (23 symbols).

Uses the ent_anneal 23-sym daily RL checkpoint for signal generation.
The RL model picks ONE symbol per day to trade (long/short/flat).
Optionally refines entry/exit prices via Gemini LLM.

Architecture matches C env exactly:
- Agent holds SINGLE position at a time (or flat)
- Each day: observe → decide → execute
- Model trained on 23 symbols, picks best opportunity

Usage:
    # Generate today's signal
    python trade_mixed_daily.py --once

    # Backtest on validation period
    python trade_mixed_daily.py --backtest --start 2025-06-01 --end 2025-12-01

    # Run as daily daemon
    python trade_mixed_daily.py --daemon
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


# Default 23 symbols (must match training data order)
DEFAULT_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "GOOG", "AMZN", "META", "TSLA", "PLTR", "NET",
    "JPM", "V", "SPY", "QQQ", "NFLX", "AMD",
    "BTCUSD", "ETHUSD", "SOLUSD", "LTCUSD", "AVAXUSD", "DOGEUSD", "LINKUSD", "AAVEUSD",
]

CRYPTO_SYMBOLS = {"BTCUSD", "ETHUSD", "SOLUSD", "LTCUSD", "AVAXUSD", "DOGEUSD", "LINKUSD", "AAVEUSD"}
FDUSD_SYMBOLS = {"BTCUSD", "ETHUSD", "SOLUSD"}

DEFAULT_CHECKPOINT = "pufferlib_market/checkpoints/autoresearch_mixed23_daily/ent_anneal/best.pt"


@dataclass
class DailySignal:
    symbol: str
    direction: str          # "long", "short", "flat"
    confidence: float       # softmax probability
    action_idx: int         # raw action index
    value_estimate: float   # critic value
    all_probs: list         # full probability vector


def load_checkpoint(path: str, device: str = "cpu"):
    """Load trained checkpoint and build policy."""
    from pufferlib_market.train import TradingPolicy

    ckpt = torch.load(path, map_location=device, weights_only=False)
    state_dict = ckpt["model"]

    # Infer architecture from state dict
    if "encoder.0.weight" in state_dict:
        obs_size = state_dict["encoder.0.weight"].shape[1]
        hidden_size = state_dict["encoder.0.weight"].shape[0]
        num_actions = state_dict["actor.2.bias"].shape[0]
    else:
        raise ValueError("Cannot infer architecture from checkpoint")

    policy = TradingPolicy(obs_size, num_actions, hidden_size)
    policy.load_state_dict(state_dict)
    policy.eval()
    policy.to(device)

    # Extract metadata
    disable_shorts = ckpt.get("disable_shorts", False)
    alloc_bins = ckpt.get("action_allocation_bins", 1)
    level_bins = ckpt.get("action_level_bins", 1)
    max_offset_bps = ckpt.get("action_max_offset_bps", 0.0)

    print(f"Loaded: obs={obs_size}, actions={num_actions}, hidden={hidden_size}")
    print(f"  shorts={'disabled' if disable_shorts else 'enabled'}, "
          f"alloc_bins={alloc_bins}, level_bins={level_bins}")

    return policy, obs_size, num_actions, hidden_size, disable_shorts


def compute_daily_features(df: pd.DataFrame) -> np.ndarray:
    """Compute 16 daily features matching export_data_daily.py exactly."""
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    ret_1d = close.pct_change(1).fillna(0.0).clip(-0.5, 0.5)
    ret_5d = close.pct_change(5).fillna(0.0).clip(-1.0, 1.0)
    ret_20d = close.pct_change(20).fillna(0.0).clip(-2.0, 2.0)

    vol_5d = ret_1d.rolling(5, min_periods=1).std(ddof=0).fillna(0.01).clip(0.0, 1.0)
    vol_20d = ret_1d.rolling(20, min_periods=1).std(ddof=0).fillna(0.01).clip(0.0, 1.0)

    ma5 = close.rolling(5, min_periods=1).mean()
    ma20 = close.rolling(20, min_periods=1).mean()
    ma60 = close.rolling(60, min_periods=1).mean()
    ma_d5 = ((close - ma5) / ma5.clip(lower=1e-8)).fillna(0.0).clip(-0.5, 0.5)
    ma_d20 = ((close - ma20) / ma20.clip(lower=1e-8)).fillna(0.0).clip(-0.5, 0.5)
    ma_d60 = ((close - ma60) / ma60.clip(lower=1e-8)).fillna(0.0).clip(-0.5, 0.5)

    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14, min_periods=1).mean()
    atr_pct = (atr14 / close.clip(lower=1e-8)).fillna(0.0).clip(0.0, 0.5)
    range_pct = ((high - low) / close.clip(lower=1e-8)).fillna(0.0).clip(0.0, 0.5)

    trend_20d = close.pct_change(20).fillna(0.0).clip(-2.0, 2.0)
    trend_60d = close.pct_change(60).fillna(0.0).clip(-3.0, 3.0)

    rm20 = close.rolling(20, min_periods=1).max()
    rm60 = close.rolling(60, min_periods=1).max()
    dd_20d = ((close - rm20) / rm20.clip(lower=1e-8)).fillna(0.0).clip(-1.0, 0.0)
    dd_60d = ((close - rm60) / rm60.clip(lower=1e-8)).fillna(0.0).clip(-1.0, 0.0)

    log_vol = np.log1p(volume.clip(lower=0.0))
    lv_m20 = log_vol.rolling(20, min_periods=1).mean()
    lv_s20 = log_vol.rolling(20, min_periods=1).std(ddof=0).replace(0.0, 1.0)
    log_vol_z = ((log_vol - lv_m20) / lv_s20).fillna(0.0).clip(-5.0, 5.0)
    log_vol_d5 = (log_vol - log_vol.rolling(5, min_periods=1).mean()).fillna(0.0).clip(-10.0, 10.0)

    features = np.column_stack([
        ret_1d.values, ret_5d.values, ret_20d.values,
        vol_5d.values, vol_20d.values,
        ma_d5.values, ma_d20.values, ma_d60.values,
        atr_pct.values, range_pct.values,
        trend_20d.values, trend_60d.values,
        dd_20d.values, dd_60d.values,
        log_vol_z.values, log_vol_d5.values,
    ]).astype(np.float32)

    return features


def build_observation(
    all_features: Dict[str, np.ndarray],
    symbols: List[str],
    step: int,
    current_position: int,  # -1=flat, 0..S-1=long, S..2S-1=short
    cash: float,
    position_qty: float,
    entry_price: float,
    hold_days: int,
    max_steps: int,
) -> np.ndarray:
    """Build observation vector matching C env exactly."""
    S = len(symbols)
    obs = np.zeros(S * 16 + 5 + S, dtype=np.float32)

    # Per-symbol features (use step-1 for observation lag)
    for i, sym in enumerate(symbols):
        feats = all_features.get(sym)
        if feats is not None and step - 1 >= 0 and step - 1 < len(feats):
            obs[i * 16:(i + 1) * 16] = feats[step - 1]

    # Portfolio state
    base = S * 16
    obs[base + 0] = cash / 10000.0
    pos_val = 0.0
    unreal = 0.0
    if current_position >= 0 and current_position < S:
        # Long position
        sym = symbols[current_position]
        feats = all_features.get(sym)
        if feats is not None and step - 1 >= 0 and step - 1 < len(feats):
            # Use close price from features is not directly available,
            # but we track entry_price and qty
            pos_val = position_qty * entry_price  # approximate
    elif current_position >= S:
        pos_val = -(position_qty * entry_price)

    obs[base + 1] = pos_val / 10000.0
    obs[base + 2] = unreal / 10000.0
    obs[base + 3] = hold_days / max(max_steps, 1)
    obs[base + 4] = step / max(max_steps, 1)

    # One-hot position
    if current_position >= 0 and current_position < S:
        obs[base + 5 + current_position] = 1.0
    elif current_position >= S and current_position < 2 * S:
        obs[base + 5 + (current_position - S)] = -1.0

    return obs


def get_daily_signal(
    policy,
    obs: np.ndarray,
    symbols: List[str],
    device: str = "cpu",
    deterministic: bool = True,
) -> DailySignal:
    """Run policy forward pass and decode action."""
    obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = policy.actor(policy.encoder(obs_tensor))
        value = policy.critic(policy.encoder(obs_tensor))

    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    if deterministic:
        action = int(logits.argmax().item())
    else:
        action = int(torch.multinomial(torch.softmax(logits, dim=-1), 1).item())

    S = len(symbols)
    confidence = float(probs[action])
    value_est = float(value.item())

    if action == 0:
        return DailySignal("", "flat", confidence, action, value_est, probs.tolist())
    elif action <= S:
        sym_idx = action - 1
        return DailySignal(symbols[sym_idx], "long", confidence, action, value_est, probs.tolist())
    elif action <= 2 * S:
        sym_idx = action - S - 1
        return DailySignal(symbols[sym_idx], "short", confidence, action, value_est, probs.tolist())
    else:
        return DailySignal("", "flat", confidence, action, value_est, probs.tolist())


def load_daily_bars(symbol: str) -> Optional[pd.DataFrame]:
    """Load daily OHLCV from available sources."""
    for root in ["trainingdata/train"]:
        for subdir in ["", "crypto", "stocks"]:
            p = Path(root) / subdir / f"{symbol}.csv" if subdir else Path(root) / f"{symbol}.csv"
            if p.exists():
                df = pd.read_csv(p)
                df.columns = [c.lower() for c in df.columns]
                ts = "timestamp" if "timestamp" in df.columns else "date"
                df["timestamp"] = pd.to_datetime(df[ts], utc=True)
                df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
                return df[["timestamp", "open", "high", "low", "close", "volume"]].reset_index(drop=True)
    return None


def run_backtest(
    symbols: List[str],
    checkpoint: str,
    start_date: str,
    end_date: str,
    max_steps: int = 90,
    slippage_bps: float = 5.0,
    initial_cash: float = 10000.0,
) -> dict:
    """Backtest matching C env logic: single position, daily bars."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy, obs_size, num_actions, hidden_size, disable_shorts = load_checkpoint(checkpoint, device)

    print(f"\nLoading data for {len(symbols)} symbols...")
    all_bars = {}
    all_features = {}
    for sym in symbols:
        df = load_daily_bars(sym)
        if df is not None and len(df) > 60:
            all_bars[sym] = df
            all_features[sym] = compute_daily_features(df)
            print(f"  {sym}: {len(df)} bars")

    # Find common dates in validation period
    start = pd.Timestamp(start_date, tz="UTC")
    end = pd.Timestamp(end_date, tz="UTC")
    dates = sorted(set().union(*(
        set(df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]["timestamp"].values)
        for df in all_bars.values()
    )))

    print(f"\nBacktest: {len(dates)} days, {start.date()} to {end.date()}")

    # State (matches C env)
    cash = initial_cash
    position_sym = -1  # -1 = flat
    position_qty = 0.0
    entry_price = 0.0
    hold_days = 0
    num_trades = 0
    winning_trades = 0
    peak_equity = initial_cash

    equity_curve = []
    trades = []
    S = len(symbols)

    for step_idx, date in enumerate(dates):
        date = pd.Timestamp(date)
        if date.tz is None:
            date = date.tz_localize("UTC")

        # Map date to feature index for each symbol
        date_indices = {}
        prices = {}
        for sym in symbols:
            if sym in all_bars:
                df = all_bars[sym]
                match = df[df["timestamp"] == date]
                if not match.empty:
                    date_indices[sym] = match.index[0]
                    prices[sym] = float(match.iloc[0]["close"])

        if not prices:
            continue

        # Build observation with per-symbol feature lookup
        S = len(symbols)
        obs = np.zeros(S * 16 + 5 + S, dtype=np.float32)

        # Per-symbol features (lagged by 1 bar, matching C env)
        for i, sym in enumerate(symbols):
            if sym in date_indices and sym in all_features:
                feat_idx = date_indices[sym] - 1  # observation lag
                feats = all_features[sym]
                if 0 <= feat_idx < len(feats):
                    obs[i * 16:(i + 1) * 16] = feats[feat_idx]

        # Portfolio state
        base = S * 16
        obs[base + 0] = cash / 10000.0

        # Position value using LAGGED price (t-1)
        pos_val = 0.0
        unreal = 0.0
        if position_sym >= 0 and position_sym < S:
            sym = symbols[position_sym]
            if sym in date_indices and sym in all_features:
                feat_idx = date_indices[sym] - 1
                if feat_idx >= 0:
                    # Get close price from bars at t-1
                    prev_bar = all_bars[sym].iloc[feat_idx] if feat_idx < len(all_bars[sym]) else None
                    if prev_bar is not None:
                        prev_price = float(prev_bar["close"])
                        pos_val = position_qty * prev_price
                        unreal = position_qty * (prev_price - entry_price)
        elif position_sym >= S:
            sym = symbols[position_sym - S]
            if sym in date_indices and sym in all_features:
                feat_idx = date_indices[sym] - 1
                if feat_idx >= 0:
                    prev_bar = all_bars[sym].iloc[feat_idx] if feat_idx < len(all_bars[sym]) else None
                    if prev_bar is not None:
                        prev_price = float(prev_bar["close"])
                        pos_val = -(position_qty * prev_price)
                        unreal = position_qty * (entry_price - prev_price)

        obs[base + 1] = pos_val / 10000.0
        obs[base + 2] = unreal / 10000.0
        obs[base + 3] = hold_days / max(max_steps, 1)
        obs[base + 4] = step_idx / max(max_steps, 1)

        # One-hot position encoding
        if position_sym >= 0 and position_sym < S:
            obs[base + 5 + position_sym] = 1.0
        elif position_sym >= S and position_sym < 2 * S:
            obs[base + 5 + (position_sym - S)] = -1.0

        # Get signal
        signal = get_daily_signal(policy, obs, symbols, device=device)

        # Execute action (matching C env logic)
        # 1. Compute equity before
        eq_before = cash
        if position_sym >= 0 and position_sym < S:
            sym = symbols[position_sym]
            if sym in prices:
                eq_before += position_qty * prices[sym]
        elif position_sym >= S:
            sym = symbols[position_sym - S]
            if sym in prices:
                eq_before -= position_qty * prices[sym]

        # 2. Close existing position if action differs
        if signal.direction == "flat" and position_sym >= 0:
            sym = symbols[position_sym % S]
            if sym in prices:
                close_price = prices[sym]
                if position_sym < S:  # long
                    cash += position_qty * close_price
                    pnl = close_price - entry_price
                else:  # short
                    cash -= position_qty * close_price
                    pnl = entry_price - close_price
                if pnl > 0:
                    winning_trades += 1
                num_trades += 1
                trades.append({"date": str(date.date()), "symbol": sym,
                              "side": "sell" if position_sym < S else "cover",
                              "price": close_price, "pnl": pnl * position_qty})
                position_sym = -1
                position_qty = 0.0
                hold_days = 0

        elif signal.direction in ("long", "short") and signal.symbol:
            target_sym_idx = symbols.index(signal.symbol) if signal.symbol in symbols else -1
            if target_sym_idx < 0:
                pass
            else:
                target_pos = target_sym_idx if signal.direction == "long" else S + target_sym_idx
                if target_pos != position_sym:
                    # Close existing
                    if position_sym >= 0:
                        sym = symbols[position_sym % S]
                        if sym in prices:
                            close_price = prices[sym]
                            if position_sym < S:
                                cash += position_qty * close_price
                                pnl = close_price - entry_price
                            else:
                                cash -= position_qty * close_price
                                pnl = entry_price - close_price
                            if pnl > 0:
                                winning_trades += 1
                            num_trades += 1
                            trades.append({"date": str(date.date()), "symbol": sym,
                                          "side": "sell" if position_sym < S else "cover",
                                          "price": close_price, "pnl": pnl * position_qty})
                        position_sym = -1
                        position_qty = 0.0

                    # Open new position
                    if signal.symbol in prices and cash > 0:
                        open_price = prices[signal.symbol] * (1 + slippage_bps / 10000)
                        if signal.direction == "long":
                            qty = cash / open_price
                            cash -= qty * open_price
                            position_sym = target_sym_idx
                        else:
                            open_price = prices[signal.symbol] * (1 - slippage_bps / 10000)
                            qty = cash / open_price
                            cash += qty * open_price  # short proceeds
                            position_sym = S + target_sym_idx
                        position_qty = qty
                        entry_price = open_price
                        hold_days = 0
                        trades.append({"date": str(date.date()), "symbol": signal.symbol,
                                      "side": signal.direction, "price": open_price, "pnl": 0})
                else:
                    hold_days += 1

        # 3. Compute equity after (use last known price if no bar today)
        equity = cash
        if position_sym >= 0:
            sym = symbols[position_sym % S]
            # Get price: today's bar, or fallback to entry price
            cur_price = prices.get(sym, entry_price)
            if position_sym < S:  # long
                equity += position_qty * cur_price
            else:  # short: equity = cash - qty*price (cash includes short proceeds)
                equity -= position_qty * cur_price

        if equity > peak_equity:
            peak_equity = equity

        equity_curve.append({"date": date, "equity": equity, "signal": f"{signal.direction} {signal.symbol}",
                            "confidence": signal.confidence})

        if (step_idx + 1) % 30 == 0:
            ret = (equity - initial_cash) / initial_cash * 100
            print(f"  Day {step_idx+1}/{len(dates)}: equity=${equity:,.2f} ({ret:+.1f}%) "
                  f"pos={signal.direction} {signal.symbol} conf={signal.confidence:.2f}")

    # Final metrics
    equities = [e["equity"] for e in equity_curve]
    if len(equities) < 2:
        return {"total_return": 0}

    total_ret = (equities[-1] - initial_cash) / initial_cash
    returns = np.diff(equities) / np.clip(equities[:-1], 1e-8, None)
    neg = returns[returns < 0]
    ds_std = neg.std() if len(neg) > 0 else 1e-8
    sortino = returns.mean() / ds_std * np.sqrt(365) if ds_std > 0 else 0
    peak = np.maximum.accumulate(equities)
    max_dd = float(((np.array(equities) - peak) / peak).min())
    n_days = len(equities)
    annualized = (1 + total_ret) ** (365 / max(n_days, 1)) - 1 if total_ret > -1 else -1

    print(f"\n{'='*60}")
    print(f"BACKTEST RESULTS ({n_days} days)")
    print(f"{'='*60}")
    print(f"  Total return:   {total_ret*100:+.2f}%")
    print(f"  Annualized:     {annualized*100:+.1f}%")
    print(f"  Sortino:        {sortino:.2f}")
    print(f"  Max drawdown:   {max_dd*100:.2f}%")
    print(f"  Trades:         {num_trades}")
    print(f"  Win rate:       {winning_trades/max(num_trades,1)*100:.1f}%")
    print(f"  Final equity:   ${equities[-1]:,.2f}")

    # Show last 5 trades
    if trades:
        print(f"\nLast 5 trades:")
        for t in trades[-5:]:
            print(f"  {t['date']} {t['side']:6s} {t['symbol']:8s} @ ${t['price']:.2f} PnL=${t['pnl']:+.2f}")

    return {
        "total_return": total_ret,
        "annualized": annualized,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "trades": num_trades,
        "win_rate": winning_trades / max(num_trades, 1),
        "final_equity": equities[-1],
    }


def main():
    parser = argparse.ArgumentParser(description="Mixed daily trading bot")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--start", default="2025-06-01")
    parser.add_argument("--end", default="2025-12-01")
    parser.add_argument("--cash", type=float, default=10000.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--once", action="store_true", help="Generate one signal and exit")
    args = parser.parse_args()

    symbols = args.symbols or DEFAULT_SYMBOLS

    if args.backtest:
        run_backtest(symbols, args.checkpoint, args.start, args.end,
                    slippage_bps=args.slippage_bps, initial_cash=args.cash)
    elif args.once:
        print("TODO: Live signal generation")
    else:
        print("Use --backtest or --once")


if __name__ == "__main__":
    main()
