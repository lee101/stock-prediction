"""Vectorized feature engineering for XGBoost stock model.

All features are computed with strict no-lookahead:
  features[row D] use only OHLCV data strictly before day D.

This means: for each row D in the DataFrame, the features reflect what
was knowable at end-of-day D-1 (close of prior trading session).

Chronos2 forecast features are stored separately and merged at dataset
build time (see dataset.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── RSI helper ────────────────────────────────────────────────────────────────

def _rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI as a series (vectorized, no loops).

    Uses Wilder's smoothing (equivalent to EWM with alpha=1/period).
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


# ── Corwin-Schultz spread (vectorized) ───────────────────────────────────────

def _cs_spread_series(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """Corwin-Schultz (2012) bid-ask spread estimate as a rolling series (bps).

    Negative estimates are clipped to 0.  Rolling mean over ``window`` pairs.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        log_hl = np.log(high / low).where(high > 0, 0.0).where(low > 0, 0.0)

    beta = log_hl ** 2 + log_hl.shift(1) ** 2
    h2 = pd.concat([high, high.shift(1)], axis=1).max(axis=1)
    l2 = pd.concat([low,  low.shift(1)],  axis=1).min(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        gamma = np.log(h2 / l2) ** 2

    k = 3.0 - 2.0 * np.sqrt(2.0)  # ≈ 0.1716
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    exp_a = np.exp(alpha)
    spread = (2.0 * (exp_a - 1.0) / (1.0 + exp_a)).clip(lower=0.0)
    return spread.rolling(window, min_periods=max(2, window // 2)).mean() * 10_000.0


# ── Main feature builder (daily) ──────────────────────────────────────────────

DAILY_FEATURE_COLS = [
    "ret_1d", "ret_2d", "ret_5d", "ret_10d", "ret_20d",
    "rsi_14",
    "vol_5d", "vol_20d",
    "atr_14",
    "spread_bps",
    "dolvol_20d_log",
    "price_vs_52w_high", "price_vs_52w_range",
    "day_of_week",
    "last_close_log",
]

CHRONOS_FEATURE_COLS = [
    "chronos_oc_return",
    "chronos_cc_return",
    "chronos_pred_range",
    "chronos_available",
]

ALL_FEATURE_COLS = DAILY_FEATURE_COLS + CHRONOS_FEATURE_COLS


def build_features_for_symbol(
    df: pd.DataFrame,
    symbol: str | None = None,
) -> pd.DataFrame:
    """Compute all daily features for one symbol (vectorized, no-lookahead).

    Args:
        df: DataFrame with columns [timestamp, open, high, low, close, volume].
            Must be sorted by timestamp ascending.
        symbol: Optional symbol name for the output column.

    Returns:
        DataFrame indexed the same as ``df`` with feature columns, timestamp,
        date, symbol, and target columns.  Rows with NaN features are kept
        (caller decides when to drop them).
    """
    df = df.sort_values("timestamp").reset_index(drop=True)

    close  = df["close"].astype(float)
    open_  = df["open"].astype(float)
    high   = df["high"].astype(float)
    low    = df["low"].astype(float)
    vol    = df["volume"].astype(float).fillna(0.0)

    feat = pd.DataFrame(index=df.index)
    feat["timestamp"] = df["timestamp"]
    feat["date"] = pd.to_datetime(df["timestamp"]).dt.date

    if symbol is not None:
        feat["symbol"] = symbol

    # ── no-lookahead returns: shift(1) so features[D] use close[D-1] ────────
    prev_close = close.shift(1)

    feat["ret_1d"]  = prev_close / close.shift(2) - 1.0
    feat["ret_2d"]  = prev_close / close.shift(3) - 1.0
    feat["ret_5d"]  = prev_close / close.shift(6) - 1.0
    feat["ret_10d"] = prev_close / close.shift(11) - 1.0
    feat["ret_20d"] = prev_close / close.shift(21) - 1.0

    # ── RSI(14) of prev close ────────────────────────────────────────────────
    feat["rsi_14"] = _rsi_series(prev_close, 14)

    # ── Realized volatility (annualised) ────────────────────────────────────
    log_ret_prev = np.log(prev_close / prev_close.shift(1))
    feat["vol_5d"]  = log_ret_prev.rolling(5,  min_periods=3).std() * np.sqrt(252)
    feat["vol_20d"] = log_ret_prev.rolling(20, min_periods=10).std() * np.sqrt(252)

    # ── ATR(14) as fraction of price ─────────────────────────────────────────
    prev_high  = high.shift(1)
    prev_low   = low.shift(1)
    prev_close2 = close.shift(2)
    true_range = pd.concat([
        prev_high - prev_low,
        (prev_high - prev_close2).abs(),
        (prev_low  - prev_close2).abs(),
    ], axis=1).max(axis=1)
    feat["atr_14"] = (true_range.rolling(14, min_periods=7).mean() / prev_close).clip(upper=1.0)

    # ── Corwin-Schultz spread (shifted so we don't see today's H/L) ─────────
    feat["spread_bps"] = _cs_spread_series(high.shift(1), low.shift(1), window=20)

    # ── Dollar volume (log) ──────────────────────────────────────────────────
    dolvol = (prev_close * vol.shift(1)).rolling(20, min_periods=5).mean()
    feat["dolvol_20d_log"] = np.log1p(dolvol.clip(lower=0.0))

    # ── Price vs 52-week range (use prev values) ─────────────────────────────
    high52 = prev_high.rolling(252, min_periods=63).max()
    low52  = prev_low.rolling(252,  min_periods=63).min()
    range52 = (high52 - low52).clip(lower=0.01)
    feat["price_vs_52w_high"]  = (prev_close / high52).clip(0.01, 2.0)
    feat["price_vs_52w_range"] = ((prev_close - low52) / range52).clip(0.0, 1.0)

    # ── Calendar features ─────────────────────────────────────────────────────
    feat["day_of_week"] = pd.to_datetime(df["timestamp"]).dt.dayofweek.astype(float)

    # ── Last close (log, for filtering) ──────────────────────────────────────
    feat["last_close_log"] = np.log1p(prev_close.clip(lower=0.01))

    # ── Target: same-day open-to-close return ────────────────────────────────
    feat["target_oc"] = ((close - open_) / open_.clip(lower=0.01)).clip(-0.5, 0.5)
    feat["target_oc_up"] = (feat["target_oc"] > 0.0).astype(np.int8)

    # ── Actual open/close for backtest use ───────────────────────────────────
    feat["actual_open"]  = open_
    feat["actual_close"] = close

    return feat


# ── Hourly feature builder ────────────────────────────────────────────────────

HOURLY_FEATURE_COLS = [
    "ret_1h", "ret_4h", "ret_8h",
    "rsi_14",
    "vol_4h", "vol_8h",
    "atr_4h",
    "spread_bps",
    "dolvol_4h_log",
    "hour_of_day",
    "day_of_week",
    "last_close_log",
]


def build_features_for_symbol_hourly(
    df: pd.DataFrame,
    symbol: str | None = None,
) -> pd.DataFrame:
    """Compute hourly features for one symbol (no-lookahead).

    Same structure as ``build_features_for_symbol`` but for hourly bars.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)

    close = df["close"].astype(float)
    open_ = df["open"].astype(float)
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    vol   = df["volume"].astype(float).fillna(0.0)

    feat = pd.DataFrame(index=df.index)
    feat["timestamp"] = df["timestamp"]
    if symbol is not None:
        feat["symbol"] = symbol

    prev_close = close.shift(1)

    feat["ret_1h"] = prev_close / close.shift(2) - 1.0
    feat["ret_4h"] = prev_close / close.shift(5) - 1.0
    feat["ret_8h"] = prev_close / close.shift(9) - 1.0

    feat["rsi_14"] = _rsi_series(prev_close, 14)

    log_ret_prev = np.log(prev_close / prev_close.shift(1))
    feat["vol_4h"] = log_ret_prev.rolling(4,  min_periods=2).std() * np.sqrt(252 * 6.5)
    feat["vol_8h"] = log_ret_prev.rolling(8,  min_periods=4).std() * np.sqrt(252 * 6.5)

    prev_high, prev_low = high.shift(1), low.shift(1)
    prev_close2 = close.shift(2)
    tr = pd.concat([
        prev_high - prev_low,
        (prev_high - prev_close2).abs(),
        (prev_low  - prev_close2).abs(),
    ], axis=1).max(axis=1)
    feat["atr_4h"] = (tr.rolling(4, min_periods=2).mean() / prev_close.clip(lower=0.01)).clip(upper=1.0)

    feat["spread_bps"] = _cs_spread_series(prev_high, prev_low, window=8)

    dolvol = (prev_close * vol.shift(1)).rolling(8, min_periods=2).mean()
    feat["dolvol_4h_log"] = np.log1p(dolvol.clip(lower=0.0))

    ts = pd.to_datetime(df["timestamp"])
    feat["hour_of_day"] = ts.dt.hour.astype(float)
    feat["day_of_week"] = ts.dt.dayofweek.astype(float)

    feat["last_close_log"] = np.log1p(prev_close.clip(lower=0.01))

    feat["target_oc"] = ((close - open_) / open_.clip(lower=0.01)).clip(-0.5, 0.5)
    feat["target_oc_up"] = (feat["target_oc"] > 0.0).astype(np.int8)

    feat["actual_open"]  = open_
    feat["actual_close"] = close

    return feat


__all__ = [
    "DAILY_FEATURE_COLS",
    "CHRONOS_FEATURE_COLS",
    "ALL_FEATURE_COLS",
    "HOURLY_FEATURE_COLS",
    "build_features_for_symbol",
    "build_features_for_symbol_hourly",
]
