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
    rsi = 100.0 - 100.0 / (1.0 + rs)
    # When avg_loss == 0 and avg_gain > 0, all moves are up → RSI = 100
    return rsi.where(avg_loss > 1e-10, 100.0)


# ── Corwin-Schultz spread (vectorized) ───────────────────────────────────────

def _cs_spread_series(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """Corwin-Schultz (2012) bid-ask spread estimate as a rolling series (bps).

    NOTE: For daily OHLCV on liquid stocks, C-S gives large values (100-300 bps)
    because it interprets intraday price volatility as bid-ask spread. Use only
    as a model feature (captures relative liquidity/volatility), not for cost modelling.
    Use volume-based spread tiers for actual trade cost estimates.

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


def _vol_spread_series(
    dolvol: pd.Series,
    tiers: tuple = (1e10, 5e8, 1e8, 5e7, 1e7),
    bps:   tuple = (2.0,  3.0,  7.0, 12.0, 25.0),
    default_bps: float = 50.0,
) -> pd.Series:
    """Volume-based bid-ask spread estimate in bps (realistic cost model).

    Uses dollar-volume tiers: larger stocks have tighter spreads.
    Uses np.select so first matching condition wins (no overwrites).
    """
    conditions = [dolvol >= t for t in tiers]
    spread_vals = np.select(conditions, list(bps), default=default_bps)
    return pd.Series(spread_vals, index=dolvol.index).where(dolvol.notna(), np.nan)


# ── Main feature builder (daily) ──────────────────────────────────────────────

DAILY_FEATURE_COLS = [
    "ret_1d", "ret_2d", "ret_5d", "ret_10d", "ret_20d",
    "rsi_14",
    "vol_5d", "vol_20d",
    "atr_14",
    "cs_spread_bps",   # C-S H/L spread (model feature: captures intraday liquidity)
    "dolvol_20d_log",
    "price_vs_52w_high", "price_vs_52w_range",
    "day_of_week",
    "last_close_log",
]

# Cross-sectional pct-rank features (per-day across all symbols).
# Opt-in: only present when the dataset builder is called with
# include_cross_sectional_ranks=True. Models trained without them keep
# working because XGBStockModel.feature_cols is saved per-pkl.
DAILY_RANK_FEATURE_COLS = [
    "rank_ret_1d",
    "rank_ret_5d",
    "rank_vol_20d",
    "rank_dolvol_20d_log",
    "rank_rsi_14",
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

    # ── Dollar volume (log) ──────────────────────────────────────────────────
    dolvol = (prev_close * vol.shift(1)).rolling(20, min_periods=5).mean()
    feat["dolvol_20d_log"] = np.log1p(dolvol.clip(lower=0.0))

    # ── Spread estimates ─────────────────────────────────────────────────────
    # cs_spread_bps: Corwin-Schultz H/L estimate (feature only — captures
    #   intraday volatility proxy; large for liquid stocks due to wide H/L)
    feat["cs_spread_bps"] = _cs_spread_series(high.shift(1), low.shift(1), window=20)
    # spread_bps: volume-based cost estimate (used in backtest cost calculation)
    feat["spread_bps"] = _vol_spread_series(dolvol)

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

    # Volume-based spread for hourly bars (scale daily $ vol tiers by 6.5 bars/day)
    dolvol_raw_h = (close.shift(1) * vol.shift(1))
    dolvol_8h = dolvol_raw_h.rolling(8, min_periods=2).mean()
    h_scale = 6.5
    feat["spread_bps"] = _vol_spread_series(
        dolvol_8h,
        tiers=(1e10 / h_scale, 5e8 / h_scale, 1e8 / h_scale, 5e7 / h_scale, 1e7 / h_scale),
    )

    feat["dolvol_4h_log"] = np.log1p(dolvol_8h.clip(lower=0.0))

    ts = pd.to_datetime(df["timestamp"])
    feat["hour_of_day"] = ts.dt.hour.astype(float)
    feat["day_of_week"] = ts.dt.dayofweek.astype(float)

    feat["last_close_log"] = np.log1p(prev_close.clip(lower=0.01))

    feat["target_oc"] = ((close - open_) / open_.clip(lower=0.01)).clip(-0.5, 0.5)
    feat["target_oc_up"] = (feat["target_oc"] > 0.0).astype(np.int8)

    feat["actual_open"]  = open_
    feat["actual_close"] = close

    # XGBoost requires NaN for missing; inf raises. Zero/negative prices in
    # a few hourly bars can produce log(0)=-inf via log-returns. Keep NaNs
    # (XGBoost treats as missing) but strip infinities.
    feat[HOURLY_FEATURE_COLS] = (
        feat[HOURLY_FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
    )

    return feat


def add_cross_sectional_ranks(
    df: pd.DataFrame,
    *,
    source_to_rank: dict[str, str] | None = None,
    fill_value: float = 0.5,
) -> pd.DataFrame:
    """Add per-day pct-rank columns across all symbols.

    For each (source_col → rank_col) pair, computes
    ``df.groupby("date")[source_col].rank(pct=True, method="average")`` — a
    float in (0, 1] per row. NaN source values stay NaN in the rank output;
    they are then filled with ``fill_value`` (default 0.5 = neutral) so
    XGBoost doesn't silently impute with column median (which would be
    meaningless for a rank).

    The base per-symbol features in ``source_col`` must already be computed
    (call this AFTER ``build_features_for_symbol`` per-symbol + concat).

    Single-symbol days (only one row for that date) get rank = 1.0; that's
    the pandas default and is OK — such days shouldn't exist in the trading
    universe we evaluate on, but guard anyway so the code doesn't crash on
    toy tests.

    Does not modify ``df`` in place.
    """
    if source_to_rank is None:
        source_to_rank = {
            "ret_1d":         "rank_ret_1d",
            "ret_5d":         "rank_ret_5d",
            "vol_20d":        "rank_vol_20d",
            "dolvol_20d_log": "rank_dolvol_20d_log",
            "rsi_14":         "rank_rsi_14",
        }
    if "date" not in df.columns:
        raise ValueError(
            "add_cross_sectional_ranks requires a 'date' column in df; "
            "call this after the per-symbol features are concatenated."
        )
    out = df.copy()
    grp = out.groupby("date", sort=False)
    for src, dst in source_to_rank.items():
        if src not in out.columns:
            raise ValueError(
                f"add_cross_sectional_ranks: source column {src!r} missing "
                f"from df (columns={list(out.columns)[:20]}...)"
            )
        ranks = grp[src].rank(pct=True, method="average")
        out[dst] = ranks.fillna(fill_value).astype(np.float32)
    return out


__all__ = [
    "DAILY_FEATURE_COLS",
    "DAILY_RANK_FEATURE_COLS",
    "CHRONOS_FEATURE_COLS",
    "ALL_FEATURE_COLS",
    "HOURLY_FEATURE_COLS",
    "build_features_for_symbol",
    "build_features_for_symbol_hourly",
    "add_cross_sectional_ranks",
    "_rsi_series",
    "_cs_spread_series",
]
