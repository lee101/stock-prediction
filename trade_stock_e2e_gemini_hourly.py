#!/usr/bin/env python3
"""End-to-end strategy-trained + Gemini meta-layer trading backtest.

Two approaches are compared on the same data windows:
  A) "Strategy+Gemini":  Train strategies (maxdiff/pctdiff/highlow via SciPy DIRECT
     optimizer + Chronos2), present ALL strategy results + price context to Gemini,
     let Gemini pick the best position.
  B) "RL+Gemini" (current prod):  PPO checkpoint signals refined by Gemini.

Modes:
  --mode hourly   : evaluate every bar  (default)
  --mode daily    : evaluate only at market-open+1h and market-close bars

Data sources:
  --data-dir trainingdatahourly  (hourly OHLCV parquets, default)
  --data-dir trainingdata        (daily CSVs — will be resampled)

Usage:
  python trade_stock_e2e_gemini_hourly.py --symbols NVDA PLTR META --once
  python trade_stock_e2e_gemini_hourly.py --symbols BTCUSD ETHUSD --mode daily --once
  python trade_stock_e2e_gemini_hourly.py --symbols NVDA --backtest-days 90
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from llm_hourly_trader.gemini_wrapper import (
    TradePlan,
    call_gemini_structured,
    _format_history_table,
    _format_forecasts,
)
from llm_hourly_trader.providers import call_llm
from src.symbol_utils import is_crypto_symbol

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FEE_BPS = 8.0  # 8bps per side (realistic for limit orders)
MAX_HOLD_BARS = 6  # max bars to hold before force-exit
INITIAL_CAPITAL = 10_000.0
MAX_POSITIONS = 3
TRAILING_STOP_PCT = 0.5  # 0.5% trailing stop

DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite-preview"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_hourly_data(symbol: str, data_dir: str = "trainingdatahourly") -> pd.DataFrame:
    """Load hourly OHLCV data for a symbol from parquet or CSV."""
    base = REPO / data_dir

    # Try stocks/ then crypto/ subdirectories, then root
    candidates = [
        base / "stocks" / f"{symbol}.parquet",
        base / "crypto" / f"{symbol}.parquet",
        base / f"{symbol}.parquet",
        base / "stocks" / f"{symbol}.csv",
        base / "crypto" / f"{symbol}.csv",
        base / f"{symbol}.csv",
    ]

    for path in candidates:
        if path.exists():
            if path.suffix == ".parquet":
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path)
            break
    else:
        raise FileNotFoundError(f"No data found for {symbol} in {data_dir}")

    # Normalize columns
    col_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ("timestamp", "date", "datetime", "time"):
            col_map[c] = "timestamp"
        elif cl == "open":
            col_map[c] = "open"
        elif cl == "high":
            col_map[c] = "high"
        elif cl == "low":
            col_map[c] = "low"
        elif cl == "close":
            col_map[c] = "close"
        elif cl in ("volume", "vol"):
            col_map[c] = "volume"
    df = df.rename(columns=col_map)

    if "timestamp" not in df.columns:
        if df.index.name and "time" in df.index.name.lower():
            df = df.reset_index()
            df = df.rename(columns={df.columns[0]: "timestamp"})
        else:
            # Try first column
            df = df.rename(columns={df.columns[0]: "timestamp"})

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            raise ValueError(f"Missing column {col} in {symbol} data")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "volume" not in df.columns:
        df["volume"] = 0.0
    else:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)

    df = df.dropna(subset=["open", "high", "low", "close"])
    return df


def load_daily_data(symbol: str, data_dir: str = "trainingdata") -> pd.DataFrame:
    """Load daily OHLCV and resample to approximate hourly (for daily-only backtests)."""
    base = REPO / data_dir
    candidates = [
        base / f"{symbol}.csv",
        base / "stocks" / f"{symbol}.csv",
        base / "crypto" / f"{symbol}.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            break
    else:
        raise FileNotFoundError(f"No daily data for {symbol} in {data_dir}")

    # Normalize columns same as hourly
    col_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ("timestamp", "date", "datetime", "time"):
            col_map[c] = "timestamp"
        elif cl == "open":
            col_map[c] = "open"
        elif cl == "high":
            col_map[c] = "high"
        elif cl == "low":
            col_map[c] = "low"
        elif cl == "close":
            col_map[c] = "close"
        elif cl in ("volume", "vol"):
            col_map[c] = "volume"
    df = df.rename(columns=col_map)

    if "timestamp" not in df.columns:
        df = df.rename(columns={df.columns[0]: "timestamp"})

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "volume" not in df.columns:
        df["volume"] = 0.0

    df = df.dropna(subset=["open", "high", "low", "close"])
    return df


# ---------------------------------------------------------------------------
# Chronos2 forecasting (lightweight wrapper for backtest)
# ---------------------------------------------------------------------------


def get_chronos2_forecasts(
    df: pd.DataFrame,
    bar_idx: int,
    lookback: int = 72,
) -> Tuple[Optional[dict], Optional[dict]]:
    """Generate Chronos2 forecasts at a given bar index.

    Returns (forecast_1h, forecast_24h) dicts or None if unavailable.
    Uses the lookback window ending at bar_idx (exclusive of bar_idx for causal).
    """
    try:
        from src.models.chronos2_wrapper import Chronos2Pipeline
    except ImportError:
        return None, None

    if bar_idx < lookback:
        return None, None

    window = df.iloc[bar_idx - lookback : bar_idx].copy()
    if len(window) < 24:
        return None, None

    last_close = float(window["close"].iloc[-1])

    try:
        pipe = _get_chronos2_pipeline()
        # 1-hour forecast
        preds_1h = pipe.predict_df(
            window[["open", "high", "low", "close"]],
            quantiles=(0.1, 0.5, 0.9),
            prediction_length=1,
        )
        fc_1h = _extract_forecast(preds_1h, last_close)

        # 24-hour forecast
        preds_24h = pipe.predict_df(
            window[["open", "high", "low", "close"]],
            quantiles=(0.1, 0.5, 0.9),
            prediction_length=24,
        )
        fc_24h = _extract_forecast(preds_24h, last_close, horizon_idx=-1)

        return fc_1h, fc_24h
    except Exception as e:
        logger.debug(f"Chronos2 forecast failed: {e}")
        return None, None


_chronos2_pipe = None


def _get_chronos2_pipeline():
    global _chronos2_pipe
    if _chronos2_pipe is None:
        from src.models.chronos2_wrapper import Chronos2Pipeline
        _chronos2_pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-t5-small")
    return _chronos2_pipe


def _extract_forecast(preds, last_close: float, horizon_idx: int = 0) -> Optional[dict]:
    """Extract forecast dict from Chronos2 predictions."""
    if preds is None:
        return None
    try:
        if isinstance(preds, dict):
            return {
                "predicted_close_p50": float(preds.get("close_p50", [last_close])[horizon_idx]),
                "predicted_close_p10": float(preds.get("close_p10", [last_close])[horizon_idx]),
                "predicted_close_p90": float(preds.get("close_p90", [last_close])[horizon_idx]),
                "predicted_high_p50": float(preds.get("high_p50", [last_close])[horizon_idx]),
                "predicted_low_p50": float(preds.get("low_p50", [last_close])[horizon_idx]),
            }
        elif isinstance(preds, pd.DataFrame):
            row = preds.iloc[horizon_idx] if len(preds) > abs(horizon_idx) else preds.iloc[-1]
            return {
                "predicted_close_p50": float(row.get("predicted_close_p50", last_close)),
                "predicted_close_p10": float(row.get("predicted_close_p10", last_close)),
                "predicted_close_p90": float(row.get("predicted_close_p90", last_close)),
                "predicted_high_p50": float(row.get("predicted_high_p50", last_close)),
                "predicted_low_p50": float(row.get("predicted_low_p50", last_close)),
            }
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Strategy evaluation (lightweight inline versions for backtest speed)
# ---------------------------------------------------------------------------


@dataclass
class StrategyResult:
    """Result from evaluating one strategy on a window."""
    name: str
    avg_daily_return: float  # percent
    sharpe: float
    win_rate: float
    num_trades: int
    entry_price: float = 0.0
    exit_price: float = 0.0
    direction: str = "long"


def evaluate_strategies(
    df: pd.DataFrame,
    bar_idx: int,
    lookback: int = 120,
    forecast_1h: Optional[dict] = None,
    forecast_24h: Optional[dict] = None,
) -> List[StrategyResult]:
    """Evaluate multiple strategies on recent history window.

    Returns a list of StrategyResult with metrics for each strategy.
    """
    if bar_idx < lookback:
        return []

    window = df.iloc[bar_idx - lookback : bar_idx].copy()
    closes = window["close"].values.astype(float)
    highs = window["high"].values.astype(float)
    lows = window["low"].values.astype(float)
    current = closes[-1]

    fee = FEE_BPS / 10000.0
    results = []

    # 1. Simple momentum strategy
    returns_1h = np.diff(closes) / closes[:-1]
    momentum = np.mean(returns_1h[-12:]) if len(returns_1h) >= 12 else 0
    mom_trades = _simulate_momentum(closes, highs, lows, fee, momentum_threshold=0.0001)
    results.append(StrategyResult(
        name="momentum",
        avg_daily_return=mom_trades["avg_return"] * 100,
        sharpe=mom_trades["sharpe"],
        win_rate=mom_trades["win_rate"],
        num_trades=mom_trades["num_trades"],
        entry_price=current * (1 - 0.001) if momentum > 0 else 0,
        exit_price=current * (1 + 0.003) if momentum > 0 else 0,
        direction="long" if momentum > 0 else "hold",
    ))

    # 2. Mean reversion strategy
    ma24 = np.mean(closes[-24:])
    deviation = (current - ma24) / ma24
    mr_trades = _simulate_mean_reversion(closes, highs, lows, fee)
    mr_direction = "long" if deviation < -0.002 else ("short" if deviation > 0.005 else "hold")
    results.append(StrategyResult(
        name="mean_reversion",
        avg_daily_return=mr_trades["avg_return"] * 100,
        sharpe=mr_trades["sharpe"],
        win_rate=mr_trades["win_rate"],
        num_trades=mr_trades["num_trades"],
        entry_price=current * 0.999 if mr_direction == "long" else 0,
        exit_price=ma24 if mr_direction == "long" else 0,
        direction=mr_direction,
    ))

    # 3. MaxDiff-style (optimized entry/exit from high-low spread)
    md_trades = _simulate_maxdiff(closes, highs, lows, fee)
    best_entry = current * (1 - md_trades["optimal_dip_pct"])
    best_exit = current * (1 + md_trades["optimal_take_pct"])
    results.append(StrategyResult(
        name="maxdiff",
        avg_daily_return=md_trades["avg_return"] * 100,
        sharpe=md_trades["sharpe"],
        win_rate=md_trades["win_rate"],
        num_trades=md_trades["num_trades"],
        entry_price=best_entry,
        exit_price=best_exit,
        direction="long" if md_trades["avg_return"] > fee else "hold",
    ))

    # 4. HighLow range strategy
    hl_trades = _simulate_highlow(closes, highs, lows, fee)
    results.append(StrategyResult(
        name="highlow",
        avg_daily_return=hl_trades["avg_return"] * 100,
        sharpe=hl_trades["sharpe"],
        win_rate=hl_trades["win_rate"],
        num_trades=hl_trades["num_trades"],
        entry_price=hl_trades["entry"],
        exit_price=hl_trades["exit"],
        direction="long" if hl_trades["avg_return"] > fee else "hold",
    ))

    # 5. Forecast-aligned strategy (if forecasts available)
    if forecast_1h and forecast_24h:
        fc_trades = _simulate_forecast_aligned(
            closes, highs, lows, fee, forecast_1h, forecast_24h, current,
        )
        results.append(StrategyResult(
            name="forecast_aligned",
            avg_daily_return=fc_trades["avg_return"] * 100,
            sharpe=fc_trades["sharpe"],
            win_rate=fc_trades["win_rate"],
            num_trades=fc_trades["num_trades"],
            entry_price=fc_trades["entry"],
            exit_price=fc_trades["exit"],
            direction=fc_trades["direction"],
        ))

    # 6. PctDiff (percentage band) strategy
    pct_trades = _simulate_pctdiff(closes, highs, lows, fee)
    results.append(StrategyResult(
        name="pctdiff",
        avg_daily_return=pct_trades["avg_return"] * 100,
        sharpe=pct_trades["sharpe"],
        win_rate=pct_trades["win_rate"],
        num_trades=pct_trades["num_trades"],
        entry_price=pct_trades["entry"],
        exit_price=pct_trades["exit"],
        direction="long" if pct_trades["avg_return"] > fee else "hold",
    ))

    return results


# --- Strategy simulators (fast numpy vectorized) ---


def _simulate_momentum(closes, highs, lows, fee, momentum_threshold=0.0001):
    returns = np.diff(closes) / closes[:-1]
    n = len(returns)
    if n < 24:
        return {"avg_return": 0, "sharpe": 0, "win_rate": 0, "num_trades": 0}

    pnls = []
    in_trade = False
    entry = 0
    for i in range(24, n):
        mom = np.mean(returns[i - 12 : i])
        if not in_trade and mom > momentum_threshold:
            entry = closes[i + 1] if i + 1 < len(closes) else closes[i]
            in_trade = True
        elif in_trade:
            held = i - (len(pnls) + 24) if pnls else 1
            curr = closes[i + 1] if i + 1 < len(closes) else closes[i]
            if mom < 0 or held >= MAX_HOLD_BARS:
                pnl = (curr - entry) / entry - 2 * fee
                pnls.append(pnl)
                in_trade = False

    if not pnls:
        return {"avg_return": 0, "sharpe": 0, "win_rate": 0, "num_trades": 0}

    pnls = np.array(pnls)
    return {
        "avg_return": float(np.mean(pnls)),
        "sharpe": float(np.mean(pnls) / (np.std(pnls) + 1e-8) * np.sqrt(252)),
        "win_rate": float(np.mean(pnls > 0)),
        "num_trades": len(pnls),
    }


def _simulate_mean_reversion(closes, highs, lows, fee):
    n = len(closes)
    if n < 48:
        return {"avg_return": 0, "sharpe": 0, "win_rate": 0, "num_trades": 0}

    pnls = []
    in_trade = False
    entry = 0
    for i in range(48, n - 1):
        ma = np.mean(closes[i - 24 : i])
        dev = (closes[i] - ma) / ma
        if not in_trade and dev < -0.002:
            entry = closes[i]
            in_trade = True
        elif in_trade:
            curr = closes[i]
            ma_now = np.mean(closes[i - 24 : i])
            if curr >= ma_now or (curr - entry) / entry > 0.005 or (i - len(pnls) - 48) >= MAX_HOLD_BARS:
                pnl = (curr - entry) / entry - 2 * fee
                pnls.append(pnl)
                in_trade = False

    if not pnls:
        return {"avg_return": 0, "sharpe": 0, "win_rate": 0, "num_trades": 0}

    pnls = np.array(pnls)
    return {
        "avg_return": float(np.mean(pnls)),
        "sharpe": float(np.mean(pnls) / (np.std(pnls) + 1e-8) * np.sqrt(252)),
        "win_rate": float(np.mean(pnls > 0)),
        "num_trades": len(pnls),
    }


def _simulate_maxdiff(closes, highs, lows, fee):
    """Optimize entry dip% and exit take% using scipy DIRECT on recent window."""
    n = len(closes)
    if n < 48:
        return {"avg_return": 0, "sharpe": 0, "win_rate": 0, "num_trades": 0,
                "optimal_dip_pct": 0.002, "optimal_take_pct": 0.005}

    try:
        from scipy.optimize import direct as scipy_direct
    except ImportError:
        # Fallback to fixed params
        return _simulate_maxdiff_fixed(closes, highs, lows, fee, 0.002, 0.005)

    def objective(params):
        dip_pct, take_pct = params
        return -_maxdiff_pnl(closes, highs, lows, fee, dip_pct, take_pct)

    try:
        result = scipy_direct(
            objective,
            bounds=[(0.0005, 0.015), (0.001, 0.025)],
            maxfun=200,
            maxiter=50,
        )
        best_dip, best_take = result.x
    except Exception:
        best_dip, best_take = 0.002, 0.005

    stats = _simulate_maxdiff_fixed(closes, highs, lows, fee, best_dip, best_take)
    stats["optimal_dip_pct"] = best_dip
    stats["optimal_take_pct"] = best_take
    return stats


def _maxdiff_pnl(closes, highs, lows, fee, dip_pct, take_pct):
    """Compute total PnL for maxdiff with given dip/take parameters."""
    pnls = []
    in_trade = False
    entry = 0
    bars_held = 0

    for i in range(1, len(closes)):
        if not in_trade:
            # Enter if price dips below close * (1 - dip_pct) during bar
            limit_entry = closes[i - 1] * (1 - dip_pct)
            if lows[i] <= limit_entry:
                entry = limit_entry
                in_trade = True
                bars_held = 0
        else:
            bars_held += 1
            # Exit if price reaches take profit or max hold
            limit_exit = entry * (1 + take_pct)
            if highs[i] >= limit_exit:
                pnl = take_pct - 2 * fee
                pnls.append(pnl)
                in_trade = False
            elif bars_held >= MAX_HOLD_BARS:
                pnl = (closes[i] - entry) / entry - 2 * fee
                pnls.append(pnl)
                in_trade = False

    return sum(pnls) if pnls else 0


def _simulate_maxdiff_fixed(closes, highs, lows, fee, dip_pct, take_pct):
    pnls = []
    in_trade = False
    entry = 0
    bars_held = 0

    for i in range(1, len(closes)):
        if not in_trade:
            limit_entry = closes[i - 1] * (1 - dip_pct)
            if lows[i] <= limit_entry:
                entry = limit_entry
                in_trade = True
                bars_held = 0
        else:
            bars_held += 1
            limit_exit = entry * (1 + take_pct)
            if highs[i] >= limit_exit:
                pnl = take_pct - 2 * fee
                pnls.append(pnl)
                in_trade = False
            elif bars_held >= MAX_HOLD_BARS:
                pnl = (closes[i] - entry) / entry - 2 * fee
                pnls.append(pnl)
                in_trade = False

    if not pnls:
        return {"avg_return": 0, "sharpe": 0, "win_rate": 0, "num_trades": 0}

    pnls = np.array(pnls)
    return {
        "avg_return": float(np.mean(pnls)),
        "sharpe": float(np.mean(pnls) / (np.std(pnls) + 1e-8) * np.sqrt(252)),
        "win_rate": float(np.mean(pnls > 0)),
        "num_trades": len(pnls),
    }


def _simulate_highlow(closes, highs, lows, fee):
    """Buy at predicted low, sell at predicted high using recent range."""
    n = len(closes)
    if n < 48:
        return {"avg_return": 0, "sharpe": 0, "win_rate": 0, "num_trades": 0,
                "entry": 0, "exit": 0}

    recent_range = highs[-24:] - lows[-24:]
    avg_range_pct = float(np.mean(recent_range / closes[-24:]))
    current = closes[-1]
    entry_price = current * (1 - avg_range_pct * 0.3)
    exit_price = current * (1 + avg_range_pct * 0.5)

    pnls = []
    in_trade = False
    entry = 0
    bars_held = 0

    for i in range(24, n):
        local_range = np.mean((highs[i - 12 : i] - lows[i - 12 : i]) / closes[i - 12 : i])
        limit_buy = closes[i] * (1 - local_range * 0.3)
        limit_sell = closes[i] * (1 + local_range * 0.5)

        if not in_trade:
            if lows[i] <= limit_buy:
                entry = limit_buy
                in_trade = True
                bars_held = 0
        else:
            bars_held += 1
            if highs[i] >= limit_sell or bars_held >= MAX_HOLD_BARS:
                exit_p = min(limit_sell, highs[i]) if highs[i] >= limit_sell else closes[i]
                pnl = (exit_p - entry) / entry - 2 * fee
                pnls.append(pnl)
                in_trade = False

    if not pnls:
        return {"avg_return": 0, "sharpe": 0, "win_rate": 0, "num_trades": 0,
                "entry": entry_price, "exit": exit_price}

    pnls = np.array(pnls)
    return {
        "avg_return": float(np.mean(pnls)),
        "sharpe": float(np.mean(pnls) / (np.std(pnls) + 1e-8) * np.sqrt(252)),
        "win_rate": float(np.mean(pnls > 0)),
        "num_trades": len(pnls),
        "entry": entry_price,
        "exit": exit_price,
    }


def _simulate_forecast_aligned(closes, highs, lows, fee, fc_1h, fc_24h, current):
    """Strategy that only trades when forecast and momentum align."""
    pred_close_1h = fc_1h.get("predicted_close_p50", current)
    pred_close_24h = fc_24h.get("predicted_close_p50", current)
    pred_high = fc_1h.get("predicted_high_p50", current)
    pred_low = fc_1h.get("predicted_low_p50", current)

    fc_return_1h = (pred_close_1h - current) / current
    fc_return_24h = (pred_close_24h - current) / current

    # Momentum check
    if len(closes) >= 12:
        mom = (closes[-1] - closes[-12]) / closes[-12]
    else:
        mom = 0

    direction = "hold"
    entry_price = 0.0
    exit_price = 0.0

    if fc_return_1h > 0.001 and fc_return_24h > 0 and mom > -0.002:
        direction = "long"
        entry_price = max(pred_low, current * 0.998)
        exit_price = min(pred_high, current * 1.01)
    elif fc_return_1h < -0.001 and fc_return_24h < 0 and mom < 0.002:
        direction = "short"
        entry_price = current * 1.002
        exit_price = current * 0.99

    # Backtest this on recent window
    n = len(closes)
    pnls = []
    in_trade = False
    entry = 0
    bars_held = 0

    for i in range(max(24, n - 48), n - 1):
        if not in_trade:
            ret_12 = (closes[i] - closes[max(0, i - 12)]) / closes[max(0, i - 12)]
            if ret_12 > 0.001:
                entry = closes[i]
                in_trade = True
                bars_held = 0
        else:
            bars_held += 1
            if (closes[i] - entry) / entry > 0.005 or bars_held >= MAX_HOLD_BARS:
                pnl = (closes[i] - entry) / entry - 2 * fee
                pnls.append(pnl)
                in_trade = False

    if not pnls:
        return {"avg_return": 0, "sharpe": 0, "win_rate": 0, "num_trades": 0,
                "entry": entry_price, "exit": exit_price, "direction": direction}

    pnls = np.array(pnls)
    return {
        "avg_return": float(np.mean(pnls)),
        "sharpe": float(np.mean(pnls) / (np.std(pnls) + 1e-8) * np.sqrt(252)),
        "win_rate": float(np.mean(pnls > 0)),
        "num_trades": len(pnls),
        "entry": entry_price,
        "exit": exit_price,
        "direction": direction,
    }


def _simulate_pctdiff(closes, highs, lows, fee):
    """Percentage-band entry/exit strategy."""
    n = len(closes)
    if n < 48:
        return {"avg_return": 0, "sharpe": 0, "win_rate": 0, "num_trades": 0,
                "entry": 0, "exit": 0}

    # Compute optimal band width using simple grid search
    best_pnl = -999
    best_band = 0.003

    for band in [0.001, 0.002, 0.003, 0.005, 0.008, 0.01]:
        total = 0
        in_trade = False
        entry = 0
        bars_held = 0
        for i in range(24, n):
            ma = np.mean(closes[max(0, i - 24) : i])
            if not in_trade:
                if closes[i] < ma * (1 - band):
                    entry = closes[i]
                    in_trade = True
                    bars_held = 0
            else:
                bars_held += 1
                if closes[i] > ma * (1 + band) or bars_held >= MAX_HOLD_BARS:
                    total += (closes[i] - entry) / entry - 2 * fee
                    in_trade = False
        if total > best_pnl:
            best_pnl = total
            best_band = band

    # Run with best band
    pnls = []
    in_trade = False
    entry = 0
    bars_held = 0
    current = closes[-1]
    ma_now = np.mean(closes[-24:])

    for i in range(24, n):
        ma = np.mean(closes[max(0, i - 24) : i])
        if not in_trade:
            if closes[i] < ma * (1 - best_band):
                entry = closes[i]
                in_trade = True
                bars_held = 0
        else:
            bars_held += 1
            if closes[i] > ma * (1 + best_band) or bars_held >= MAX_HOLD_BARS:
                pnl = (closes[i] - entry) / entry - 2 * fee
                pnls.append(pnl)
                in_trade = False

    entry_price = ma_now * (1 - best_band)
    exit_price = ma_now * (1 + best_band)

    if not pnls:
        return {"avg_return": 0, "sharpe": 0, "win_rate": 0, "num_trades": 0,
                "entry": entry_price, "exit": exit_price}

    pnls = np.array(pnls)
    return {
        "avg_return": float(np.mean(pnls)),
        "sharpe": float(np.mean(pnls) / (np.std(pnls) + 1e-8) * np.sqrt(252)),
        "win_rate": float(np.mean(pnls > 0)),
        "num_trades": len(pnls),
        "entry": entry_price,
        "exit": exit_price,
    }


# ---------------------------------------------------------------------------
# Gemini meta-layer: present all strategies + context, let LLM pick
# ---------------------------------------------------------------------------


def build_gemini_meta_prompt(
    symbol: str,
    current_price: float,
    history_rows: list[dict],
    strategies: List[StrategyResult],
    forecast_1h: Optional[dict],
    forecast_24h: Optional[dict],
    portfolio_state: dict,
) -> str:
    """Build a comprehensive prompt presenting all strategy results to Gemini."""

    # Format strategy results table
    strat_lines = [
        "| Strategy | Direction | Avg Return | Sharpe | Win Rate | Trades | Entry | Exit |",
        "|----------|-----------|------------|--------|----------|--------|-------|------|",
    ]
    for s in strategies:
        strat_lines.append(
            f"| {s.name} | {s.direction} | {s.avg_daily_return:+.4f}% | "
            f"{s.sharpe:.2f} | {s.win_rate:.1%} | {s.num_trades} | "
            f"${s.entry_price:.2f} | ${s.exit_price:.2f} |"
        )
    strat_table = "\n".join(strat_lines)

    # Format forecasts
    fc_text = _format_forecasts(forecast_1h, forecast_24h)

    # Format history
    hist_text = _format_history_table(history_rows, n=24)

    # Portfolio context
    cash = portfolio_state.get("cash", INITIAL_CAPITAL)
    equity = portfolio_state.get("equity", INITIAL_CAPITAL)
    positions = portfolio_state.get("positions", {})
    pos_str = "Flat (no positions)" if not positions else json.dumps(positions, indent=2)

    # Best strategy summary
    best = max(strategies, key=lambda s: s.sharpe) if strategies else None
    best_str = f"{best.name} (Sharpe={best.sharpe:.2f}, return={best.avg_daily_return:+.4f}%)" if best else "none"

    asset_label = "cryptocurrency" if is_crypto_symbol(symbol) else "stock"

    prompt = f"""You are an expert quantitative portfolio manager. You have access to multiple
trained trading strategies and ML forecasts. Your job is to synthesize all signals
and decide the optimal action for {symbol} ({asset_label}).

## Current State
Symbol: {symbol}
Current Price: ${current_price:.2f}
Cash: ${cash:,.2f} | Equity: ${equity:,.2f}
Positions: {pos_str}

## Recent Hourly OHLCV (last 24 bars):
{hist_text}

## ML Forecasts (Chronos2):
{fc_text}

## Trained Strategy Results (backtested on recent data):
{strat_table}

Best strategy by Sharpe: {best_str}

## Your Task
Analyze ALL the strategy signals above. Consider:
1. Which strategies agree on direction? (consensus = higher confidence)
2. Do the ML forecasts support the strategies?
3. What entry/exit prices optimize risk-adjusted returns given the strategy outputs?
4. Is there enough edge to overcome {FEE_BPS:.0f}bp fees per side?

You must decide:
- direction: "long", "short", or "hold"
- buy_price: limit entry price (0 if hold)
- sell_price: limit exit/take-profit price (0 if hold)
- confidence: 0.0 to 1.0

Rules:
- If strategies DISAGREE and no clear majority, prefer HOLD
- If Sharpe < 0.5 on all strategies, prefer HOLD
- If entering LONG: buy_price < current_price (dip buy), sell_price > buy_price
- Be precise with prices — use the strategy entry/exit levels as reference points
- Weight strategies by their Sharpe and win rate, not just returns

Respond with JSON: {{"direction": "...", "buy_price": "...", "sell_price": "...", "confidence": "..."}}"""

    return prompt


# ---------------------------------------------------------------------------
# RL+Gemini approach (for comparison)
# ---------------------------------------------------------------------------


def get_rl_gemini_signal(
    symbol: str,
    df: pd.DataFrame,
    bar_idx: int,
    forecast_1h: Optional[dict],
    forecast_24h: Optional[dict],
    model: str = DEFAULT_GEMINI_MODEL,
) -> Optional[TradePlan]:
    """Get RL+Gemini signal at a given bar (prod-equivalent approach)."""
    try:
        from pufferlib_market.inference import PPOTrader, compute_hourly_features
        from unified_orchestrator.rl_gemini_bridge import (
            RLGeminiBridge,
            build_hybrid_prompt,
            RLSignal,
        )
    except ImportError:
        return None

    if bar_idx < 72:
        return None

    window = df.iloc[bar_idx - 72 : bar_idx]
    current_price = float(df.iloc[bar_idx]["close"])

    history_rows = []
    for _, row in window.tail(24).iterrows():
        history_rows.append({
            "timestamp": str(row.get("timestamp", ""))[:16],
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row.get("volume", 0)),
        })

    # Try to get RL signal
    rl_signal = None
    for ckpt_candidates in [
        REPO / "pufferlib_market/checkpoints/autoresearch/longonly_forecast/best.pt",
        REPO / "pufferlib_market/checkpoints/autoresearch/slip_5bps/best.pt",
    ]:
        if ckpt_candidates.exists():
            try:
                bridge = RLGeminiBridge(checkpoint_path=str(ckpt_candidates))
                features = compute_hourly_features(window)
                spec = bridge.get_checkpoint_spec()
                num_sym = (spec.obs_size - 5) // 17
                # Build single-symbol observation
                feat_arr = np.zeros((num_sym, 16), dtype=np.float32)
                feat_arr[0] = features
                from unified_orchestrator.rl_gemini_bridge import build_portfolio_observation
                obs = build_portfolio_observation(feat_arr)
                signals = bridge.get_rl_signals(obs, num_sym, [symbol] * num_sym, top_k=1)
                if signals:
                    rl_signal = signals[0]
                break
            except Exception as e:
                logger.debug(f"RL signal failed: {e}")
                continue

    if rl_signal is None:
        # Fallback to pure Gemini
        rl_signal = RLSignal(
            symbol_idx=0,
            symbol_name=symbol,
            direction="long",
            confidence=0.5,
            logit_gap=0.0,
            allocation_pct=0.5,
        )

    prompt = build_hybrid_prompt(
        symbol=symbol,
        rl_signal=rl_signal,
        history_rows=history_rows,
        current_price=current_price,
        forecast_1h=forecast_1h,
        forecast_24h=forecast_24h,
    )

    try:
        plan = call_llm(prompt, model=model)
        return plan
    except Exception as e:
        logger.debug(f"RL+Gemini LLM call failed: {e}")
        # Fallback to RL-only
        if rl_signal.direction == "long":
            return TradePlan("long", current_price * 0.998, current_price * 1.01, rl_signal.confidence)
        return TradePlan("hold", 0, 0, 0)


# ---------------------------------------------------------------------------
# Market simulator (shared by both approaches)
# ---------------------------------------------------------------------------


@dataclass
class SimPosition:
    symbol: str
    direction: str  # "long" or "short"
    entry_price: float
    qty: float
    entry_bar: int
    peak_price: float = 0.0  # for trailing stop

    def __post_init__(self):
        self.peak_price = self.entry_price


@dataclass
class SimResult:
    """Result of a full backtest simulation."""
    approach: str
    symbol: str
    initial_capital: float
    final_equity: float
    total_return_pct: float
    num_trades: int
    win_rate: float
    sharpe: float
    sortino: float
    max_drawdown_pct: float
    avg_trade_pnl_pct: float
    trade_log: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)


def run_backtest(
    symbol: str,
    df: pd.DataFrame,
    approach: str,
    mode: str = "hourly",
    start_bar: int = 120,
    model: str = DEFAULT_GEMINI_MODEL,
    skip_llm: bool = False,
) -> SimResult:
    """Run a full market simulator backtest for one approach.

    Args:
        approach: "strategy_gemini" or "rl_gemini"
        mode: "hourly" (every bar) or "daily" (open+1h and close only)
        skip_llm: If True, use strategy signals directly without Gemini calls
    """
    cash = INITIAL_CAPITAL
    position: Optional[SimPosition] = None
    trade_log = []
    equity_curve = []
    fee_rate = FEE_BPS / 10000.0

    n = len(df)
    eval_bars = _get_eval_bars(df, mode, start_bar)

    logger.info(f"  [{approach}] Backtesting {symbol}: {len(eval_bars)} eval bars, "
                f"{n} total bars, mode={mode}")

    for bar_idx in eval_bars:
        if bar_idx >= n:
            break

        row = df.iloc[bar_idx]
        current_price = float(row["close"])
        bar_high = float(row["high"])
        bar_low = float(row["low"])
        timestamp = row.get("timestamp", bar_idx)

        # Track equity
        pos_value = 0.0
        if position is not None:
            if position.direction == "long":
                pos_value = position.qty * current_price
            else:
                pos_value = position.qty * (2 * position.entry_price - current_price)
        equity = cash + pos_value
        equity_curve.append({"bar": bar_idx, "equity": equity, "timestamp": str(timestamp)})

        # Check force-exit on max hold
        if position is not None:
            bars_held = bar_idx - position.entry_bar
            if bars_held >= MAX_HOLD_BARS:
                pnl = _close_position(position, current_price, fee_rate)
                cash += position.qty * current_price * (1 - fee_rate) if position.direction == "long" \
                    else position.qty * (2 * position.entry_price - current_price) * (1 - fee_rate)
                trade_log.append({
                    "bar": bar_idx, "action": "force_exit", "price": current_price,
                    "pnl_pct": pnl * 100, "reason": "max_hold",
                })
                position = None
                continue

            # Trailing stop check
            if position.direction == "long":
                position.peak_price = max(position.peak_price, bar_high)
                if current_price < position.peak_price * (1 - TRAILING_STOP_PCT / 100):
                    pnl = (current_price - position.entry_price) / position.entry_price - 2 * fee_rate
                    cash += position.qty * current_price * (1 - fee_rate)
                    trade_log.append({
                        "bar": bar_idx, "action": "trailing_stop", "price": current_price,
                        "pnl_pct": pnl * 100,
                    })
                    position = None
                    continue

        # Get signal based on approach
        plan = _get_signal(
            approach, symbol, df, bar_idx, current_price, cash, equity,
            position, model, skip_llm,
        )

        if plan is None:
            continue

        # Execute signal
        if plan.direction == "hold" or plan.confidence < 0.3:
            continue

        if position is None and plan.direction in ("long", "short"):
            # Check if limit entry would fill on this bar
            entry_price = plan.buy_price if plan.buy_price > 0 else current_price
            if plan.direction == "long":
                if entry_price >= bar_low:  # limit buy would fill
                    fill_price = min(entry_price, current_price)
                    qty = (cash * 0.95) / (fill_price * (1 + fee_rate))
                    cost = qty * fill_price * (1 + fee_rate)
                    cash -= cost
                    position = SimPosition(
                        symbol=symbol,
                        direction="long",
                        entry_price=fill_price,
                        qty=qty,
                        entry_bar=bar_idx,
                    )
                    trade_log.append({
                        "bar": bar_idx, "action": "buy", "price": fill_price,
                        "qty": qty, "target_exit": plan.sell_price,
                    })

        elif position is not None and plan.sell_price > 0:
            # Check if take-profit would fill
            if position.direction == "long" and bar_high >= plan.sell_price:
                fill_price = plan.sell_price
                pnl = (fill_price - position.entry_price) / position.entry_price - 2 * fee_rate
                cash += position.qty * fill_price * (1 - fee_rate)
                trade_log.append({
                    "bar": bar_idx, "action": "take_profit", "price": fill_price,
                    "pnl_pct": pnl * 100,
                })
                position = None

    # Force close any remaining position at last bar
    if position is not None:
        final_price = float(df.iloc[min(len(df) - 1, eval_bars[-1] if eval_bars else n - 1)]["close"])
        pnl = (final_price - position.entry_price) / position.entry_price - 2 * fee_rate
        cash += position.qty * final_price * (1 - fee_rate)
        trade_log.append({
            "bar": len(df) - 1, "action": "final_close", "price": final_price,
            "pnl_pct": pnl * 100,
        })
        position = None

    # Compute metrics
    final_equity = cash
    total_return = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    pnls = [t["pnl_pct"] for t in trade_log if "pnl_pct" in t]
    num_trades = len(pnls)
    win_rate = np.mean([p > 0 for p in pnls]) if pnls else 0
    avg_pnl = np.mean(pnls) if pnls else 0

    # Sharpe from equity curve
    equities = [e["equity"] for e in equity_curve]
    if len(equities) > 1:
        eq_returns = np.diff(equities) / np.array(equities[:-1])
        sharpe = float(np.mean(eq_returns) / (np.std(eq_returns) + 1e-8) * np.sqrt(252 * 24))
        neg_returns = eq_returns[eq_returns < 0]
        sortino = float(np.mean(eq_returns) / (np.std(neg_returns) + 1e-8) * np.sqrt(252 * 24)) if len(neg_returns) > 0 else sharpe
        # Max drawdown
        peak = np.maximum.accumulate(equities)
        drawdown = (peak - equities) / peak * 100
        max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0
    else:
        sharpe = sortino = max_dd = 0.0

    return SimResult(
        approach=approach,
        symbol=symbol,
        initial_capital=INITIAL_CAPITAL,
        final_equity=final_equity,
        total_return_pct=total_return,
        num_trades=num_trades,
        win_rate=win_rate,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown_pct=max_dd,
        avg_trade_pnl_pct=avg_pnl,
        trade_log=trade_log,
        equity_curve=equity_curve,
    )


def _get_eval_bars(df: pd.DataFrame, mode: str, start_bar: int) -> list[int]:
    """Determine which bars to evaluate based on mode."""
    n = len(df)

    if mode == "hourly":
        return list(range(start_bar, n))

    # Daily mode: only evaluate at ~10:30 (open+1h) and ~15:45 (near close)
    eval_bars = []
    if "timestamp" in df.columns:
        for i in range(start_bar, n):
            ts = df.iloc[i]["timestamp"]
            if pd.notna(ts):
                try:
                    hour = ts.hour if hasattr(ts, "hour") else pd.Timestamp(ts).hour
                    # Market open+1h (~10:30 ET = 15:30 UTC) or close (~15:45 ET = 20:45 UTC)
                    if hour in (14, 15, 20, 21):  # approximate UTC hours
                        eval_bars.append(i)
                except Exception:
                    pass

    # Fallback: if no timestamps match or crypto (24/7), sample every 7 bars (~daily)
    if not eval_bars:
        eval_bars = list(range(start_bar, n, 7))

    return eval_bars


def _get_signal(
    approach: str,
    symbol: str,
    df: pd.DataFrame,
    bar_idx: int,
    current_price: float,
    cash: float,
    equity: float,
    position: Optional[SimPosition],
    model: str,
    skip_llm: bool,
) -> Optional[TradePlan]:
    """Get a trading signal for the given approach."""

    if approach == "strategy_gemini":
        return _get_strategy_gemini_signal(
            symbol, df, bar_idx, current_price, cash, equity, position, model, skip_llm,
        )
    elif approach == "rl_gemini":
        if skip_llm:
            return _get_rl_only_signal(symbol, df, bar_idx, current_price)
        return get_rl_gemini_signal(
            symbol, df, bar_idx,
            *get_chronos2_forecasts(df, bar_idx),
            model=model,
        )
    return None


def _get_strategy_gemini_signal(
    symbol: str,
    df: pd.DataFrame,
    bar_idx: int,
    current_price: float,
    cash: float,
    equity: float,
    position: Optional[SimPosition],
    model: str,
    skip_llm: bool,
) -> Optional[TradePlan]:
    """Strategy+Gemini approach: train strategies then let Gemini pick."""

    # Get forecasts
    forecast_1h, forecast_24h = get_chronos2_forecasts(df, bar_idx)

    # Evaluate all strategies on recent window
    strategies = evaluate_strategies(df, bar_idx, forecast_1h=forecast_1h, forecast_24h=forecast_24h)
    if not strategies:
        return TradePlan("hold", 0, 0, 0)

    if skip_llm:
        # Use best strategy directly (no Gemini call)
        best = max(strategies, key=lambda s: s.sharpe)
        if best.sharpe < 0.5 or best.direction == "hold":
            return TradePlan("hold", 0, 0, 0)
        return TradePlan(
            direction=best.direction,
            buy_price=best.entry_price,
            sell_price=best.exit_price,
            confidence=min(1.0, best.sharpe / 3.0),
        )

    # Build history rows for Gemini
    window = df.iloc[max(0, bar_idx - 24) : bar_idx]
    history_rows = []
    for _, row in window.iterrows():
        history_rows.append({
            "timestamp": str(row.get("timestamp", ""))[:16],
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row.get("volume", 0)),
        })

    portfolio_state = {
        "cash": cash,
        "equity": equity,
        "positions": {position.symbol: {
            "direction": position.direction,
            "entry_price": position.entry_price,
            "qty": position.qty,
            "bars_held": bar_idx - position.entry_bar,
        }} if position else {},
    }

    prompt = build_gemini_meta_prompt(
        symbol=symbol,
        current_price=current_price,
        history_rows=history_rows,
        strategies=strategies,
        forecast_1h=forecast_1h,
        forecast_24h=forecast_24h,
        portfolio_state=portfolio_state,
    )

    try:
        plan = call_llm(prompt, model=model)
        return plan
    except Exception as e:
        logger.debug(f"Strategy+Gemini LLM call failed: {e}")
        # Fallback: use best strategy
        best = max(strategies, key=lambda s: s.sharpe)
        return TradePlan(
            direction=best.direction if best.sharpe > 0.5 else "hold",
            buy_price=best.entry_price,
            sell_price=best.exit_price,
            confidence=min(1.0, best.sharpe / 3.0),
        )


def _get_rl_only_signal(
    symbol: str,
    df: pd.DataFrame,
    bar_idx: int,
    current_price: float,
) -> Optional[TradePlan]:
    """RL-only signal (no Gemini) for skip_llm mode."""
    try:
        from pufferlib_market.inference import compute_hourly_features

        if bar_idx < 72:
            return TradePlan("hold", 0, 0, 0)

        window = df.iloc[bar_idx - 72 : bar_idx]
        features = compute_hourly_features(window)

        # Simple momentum from features
        ret_1h = features[0] if len(features) > 0 else 0
        ret_24h = features[2] if len(features) > 2 else 0

        if ret_1h > 0.001 and ret_24h > -0.005:
            return TradePlan(
                "long",
                current_price * 0.998,
                current_price * 1.01,
                min(1.0, abs(ret_1h) * 100),
            )
        return TradePlan("hold", 0, 0, 0)
    except ImportError:
        return TradePlan("hold", 0, 0, 0)


def _close_position(position: SimPosition, price: float, fee_rate: float) -> float:
    """Compute PnL% for closing a position."""
    if position.direction == "long":
        return (price - position.entry_price) / position.entry_price - 2 * fee_rate
    else:
        return (position.entry_price - price) / position.entry_price - 2 * fee_rate


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------


def print_comparison(results: List[SimResult]):
    """Print a comparison table of all backtest results."""
    print("\n" + "=" * 90)
    print("BACKTEST COMPARISON REPORT")
    print("=" * 90)

    # Group by symbol
    symbols = sorted(set(r.symbol for r in results))
    approaches = sorted(set(r.approach for r in results))

    header = f"{'Symbol':<10} {'Approach':<20} {'Return%':>10} {'Trades':>7} {'WinRate':>8} {'Sharpe':>8} {'Sortino':>8} {'MaxDD%':>8} {'AvgPnL%':>8}"
    print(header)
    print("-" * len(header))

    for sym in symbols:
        for approach in approaches:
            matching = [r for r in results if r.symbol == sym and r.approach == approach]
            for r in matching:
                print(
                    f"{r.symbol:<10} {r.approach:<20} {r.total_return_pct:>+10.2f} "
                    f"{r.num_trades:>7} {r.win_rate:>8.1%} {r.sharpe:>8.2f} "
                    f"{r.sortino:>8.2f} {r.max_drawdown_pct:>8.2f} {r.avg_trade_pnl_pct:>+8.3f}"
                )
        print()

    # Overall summary
    print("\n" + "-" * 60)
    print("AGGREGATE BY APPROACH:")
    for approach in approaches:
        group = [r for r in results if r.approach == approach]
        if not group:
            continue
        avg_ret = np.mean([r.total_return_pct for r in group])
        avg_sharpe = np.mean([r.sharpe for r in group])
        avg_sortino = np.mean([r.sortino for r in group])
        avg_wr = np.mean([r.win_rate for r in group])
        total_trades = sum(r.num_trades for r in group)
        avg_dd = np.mean([r.max_drawdown_pct for r in group])
        print(
            f"  {approach:<20}: return={avg_ret:+.2f}% sharpe={avg_sharpe:.2f} "
            f"sortino={avg_sortino:.2f} WR={avg_wr:.1%} trades={total_trades} maxDD={avg_dd:.2f}%"
        )

    print("=" * 90)


def save_results(results: List[SimResult], output_dir: str = "backtest_results"):
    """Save results to JSON and CSV."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Summary CSV
    rows = []
    for r in results:
        rows.append({
            "symbol": r.symbol,
            "approach": r.approach,
            "initial_capital": r.initial_capital,
            "final_equity": r.final_equity,
            "total_return_pct": r.total_return_pct,
            "num_trades": r.num_trades,
            "win_rate": r.win_rate,
            "sharpe": r.sharpe,
            "sortino": r.sortino,
            "max_drawdown_pct": r.max_drawdown_pct,
            "avg_trade_pnl_pct": r.avg_trade_pnl_pct,
        })
    pd.DataFrame(rows).to_csv(out / f"comparison_{ts}.csv", index=False)

    # Detailed JSON with trade logs
    detailed = []
    for r in results:
        detailed.append({
            "symbol": r.symbol,
            "approach": r.approach,
            "metrics": {
                "total_return_pct": r.total_return_pct,
                "num_trades": r.num_trades,
                "win_rate": r.win_rate,
                "sharpe": r.sharpe,
                "sortino": r.sortino,
                "max_drawdown_pct": r.max_drawdown_pct,
            },
            "trade_log": r.trade_log[:200],  # cap for file size
            "equity_curve_sample": r.equity_curve[::max(1, len(r.equity_curve) // 100)],
        })
    with open(out / f"detailed_{ts}.json", "w") as f:
        json.dump(detailed, f, indent=2, default=str)

    logger.info(f"Results saved to {out}/comparison_{ts}.csv")
    return str(out / f"comparison_{ts}.csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="E2E Strategy+Gemini vs RL+Gemini backtest comparison",
    )
    parser.add_argument(
        "--symbols", nargs="+",
        default=["NVDA", "PLTR", "BTCUSD", "ETHUSD"],
        help="Symbols to backtest",
    )
    parser.add_argument(
        "--mode", choices=["hourly", "daily"], default="hourly",
        help="Evaluation frequency: every bar or daily only",
    )
    parser.add_argument(
        "--data-dir", default="trainingdatahourly",
        help="Data directory (trainingdatahourly or trainingdata)",
    )
    parser.add_argument(
        "--backtest-days", type=int, default=60,
        help="Number of days of data to backtest over",
    )
    parser.add_argument(
        "--model", default=DEFAULT_GEMINI_MODEL,
        help="Gemini model for LLM calls",
    )
    parser.add_argument(
        "--skip-llm", action="store_true",
        help="Skip LLM calls (use strategy/RL signals directly, much faster)",
    )
    parser.add_argument(
        "--approaches", nargs="+",
        default=["strategy_gemini", "rl_gemini"],
        help="Which approaches to compare",
    )
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument(
        "--output-dir", default="backtest_results",
        help="Directory to save results",
    )

    args = parser.parse_args()

    logger.info(f"Trade Stock E2E Gemini Hourly")
    logger.info(f"  Symbols: {args.symbols}")
    logger.info(f"  Mode: {args.mode}")
    logger.info(f"  Data: {args.data_dir}")
    logger.info(f"  Backtest days: {args.backtest_days}")
    logger.info(f"  Approaches: {args.approaches}")
    logger.info(f"  Skip LLM: {args.skip_llm}")

    all_results = []

    for symbol in args.symbols:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing {symbol}")
        logger.info(f"{'=' * 60}")

        # Load data
        try:
            if args.data_dir == "trainingdata":
                df = load_daily_data(symbol, args.data_dir)
            else:
                df = load_hourly_data(symbol, args.data_dir)
        except FileNotFoundError as e:
            logger.warning(f"  Skipping {symbol}: {e}")
            continue

        logger.info(f"  Loaded {len(df)} bars from {args.data_dir}")

        # Trim to backtest window
        if args.backtest_days > 0 and args.data_dir == "trainingdatahourly":
            max_bars = args.backtest_days * 24
            if len(df) > max_bars + 120:
                df = df.iloc[-(max_bars + 120) :].reset_index(drop=True)
                logger.info(f"  Trimmed to {len(df)} bars ({args.backtest_days} days + 120 warmup)")
        elif args.backtest_days > 0 and args.data_dir == "trainingdata":
            max_bars = args.backtest_days
            if len(df) > max_bars + 120:
                df = df.iloc[-(max_bars + 120) :].reset_index(drop=True)

        # Run backtests for each approach
        for approach in args.approaches:
            try:
                t0 = time.time()
                result = run_backtest(
                    symbol=symbol,
                    df=df,
                    approach=approach,
                    mode=args.mode,
                    model=args.model,
                    skip_llm=args.skip_llm,
                )
                elapsed = time.time() - t0
                logger.info(
                    f"  [{approach}] {symbol}: return={result.total_return_pct:+.2f}% "
                    f"trades={result.num_trades} WR={result.win_rate:.1%} "
                    f"Sharpe={result.sharpe:.2f} ({elapsed:.1f}s)"
                )
                all_results.append(result)
            except Exception as e:
                logger.error(f"  [{approach}] {symbol} failed: {e}")
                traceback.print_exc()

    if all_results:
        print_comparison(all_results)
        save_results(all_results, args.output_dir)

    # If mode is daily, also try hourly for comparison
    if args.mode == "daily" and not args.once:
        logger.info("\n\nNow running hourly mode for comparison...")
        for symbol in args.symbols:
            try:
                if args.data_dir == "trainingdata":
                    df = load_daily_data(symbol, args.data_dir)
                else:
                    df = load_hourly_data(symbol, args.data_dir)
            except FileNotFoundError:
                continue

            if args.backtest_days > 0:
                max_bars = args.backtest_days * (24 if "hourly" in args.data_dir else 1)
                if len(df) > max_bars + 120:
                    df = df.iloc[-(max_bars + 120) :].reset_index(drop=True)

            for approach in args.approaches:
                try:
                    result = run_backtest(
                        symbol=symbol, df=df, approach=approach,
                        mode="hourly", model=args.model, skip_llm=args.skip_llm,
                    )
                    result.approach = f"{approach}_hourly"
                    all_results.append(result)
                except Exception:
                    pass

        if all_results:
            print_comparison(all_results)
            save_results(all_results, args.output_dir)


if __name__ == "__main__":
    main()
