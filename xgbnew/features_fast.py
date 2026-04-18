"""Fast feature builder — polars-native rolling, per-symbol partitioned.

Drop-in for ``features.build_features_for_symbol`` when called over many
symbols. Keeps the same feature schema (DAILY_FEATURE_COLS) so downstream
model/backtest code does not change.

Design notes
------------
- Polars ``over("symbol")`` runs rolling ops per partition on CPU with
  Rust-level parallelism across cores. For ~1 M rows across ~1 k symbols
  this is typically 10–30× faster than pandas and easily matches naive
  GPU rolling (transfer + launch overhead dominates at this size).
- A GPU rewrite only pays off for *much* larger universes (10 k+ symbols
  or intraday 1-minute bars). For daily stocks on 846 symbols, the
  XGBoost training step now dominates, and with device='cuda' that's
  <5 s — so polars ETL is Pareto-optimal for now.
- Keep function-only API (no classes) — easy to unit-test and compose.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Reuse canonical feature list from the reference implementation so tests
# against the pandas builder compare like-for-like.
from .features import DAILY_FEATURE_COLS


_OHLCV_COLS = ("open", "high", "low", "close", "volume")


def _read_one_csv_polars(path: Path, symbol: str):
    """Read one OHLCV CSV, return a polars DataFrame with a uniform schema.

    Handles the 3 schema variants in trainingdata/train (9, 10, 15 cols) by
    reading all columns as utf8, normalising names, then casting.
    """
    import polars as pl

    # Read with infer_schema_length=0 to force utf8, then cast what we need.
    try:
        df = pl.read_csv(str(path), infer_schema_length=0, has_header=True)
    except Exception as exc:
        logger.warning("failed to read %s: %s", path, exc)
        return None
    # Some merged CSVs contain both "Dividends" and "dividends" (etc.) from
    # overlapping yfinance+alpaca ingests. Lowercasing duplicates, so keep the
    # first occurrence and drop the rest before renaming.
    seen: set[str] = set()
    keep_idx: list[int] = []
    for i, c in enumerate(df.columns):
        lc = c.strip().lower()
        if lc in seen:
            continue
        seen.add(lc)
        keep_idx.append(i)
    if len(keep_idx) != len(df.columns):
        df = df.select([df.columns[i] for i in keep_idx])
    df = df.rename({c: c.strip().lower() for c in df.columns})

    ts_col = "timestamp" if "timestamp" in df.columns else ("date" if "date" in df.columns else None)
    if ts_col is None:
        return None
    # Data uses ISO8601 with tz suffix (e.g. "2021-01-04 00:00:00+00:00").
    df = df.with_columns(
        pl.col(ts_col).str.to_datetime(time_zone="UTC", strict=False).alias("timestamp")
    )

    # Build a consistent set of OHLCV columns (fill missing with null-float).
    exprs = [pl.lit(symbol).alias("symbol"), pl.col("timestamp")]
    for c in _OHLCV_COLS:
        if c in df.columns:
            exprs.append(pl.col(c).cast(pl.Float64, strict=False).alias(c))
        else:
            exprs.append(pl.lit(None, dtype=pl.Float64).alias(c))
    out = df.select(exprs)
    out = out.filter(pl.col("close").is_not_null() & pl.col("timestamp").is_not_null())
    if out.height < 60:
        return None
    return out


def _read_ohlcv_polars(
    data_root: Path, symbols: Sequence[str], subdir: str = "train",
):
    """Read all symbols' CSVs with polars (per-file, schema-tolerant).

    Returns a single long DataFrame keyed by (symbol, timestamp, date) with
    OHLCV columns, sorted ascending.
    """
    import polars as pl

    parts = []
    for sym in symbols:
        p = data_root / subdir / f"{sym}.csv"
        if not p.exists():
            continue
        piece = _read_one_csv_polars(p, sym)
        if piece is not None:
            parts.append(piece)
    if not parts:
        raise FileNotFoundError(f"no usable CSVs under {data_root}/{subdir} for {len(symbols)} symbols")

    df = pl.concat(parts, how="vertical")
    df = df.with_columns(pl.col("timestamp").dt.date().alias("date"))
    df = df.unique(subset=["symbol", "timestamp"], keep="last", maintain_order=False)
    df = df.sort(["symbol", "timestamp"])
    return df


def _rsi_expr(prev_close, period: int = 14):
    """RSI(14) on prev_close. Polars expression."""
    import polars as pl
    delta = prev_close.diff().over("symbol")
    gain = pl.when(delta > 0).then(delta).otherwise(0.0)
    loss = pl.when(delta < 0).then(-delta).otherwise(0.0)
    avg_gain = gain.rolling_mean(window_size=period, min_samples=period).over("symbol")
    avg_loss = loss.rolling_mean(window_size=period, min_samples=period).over("symbol")
    rs = avg_gain / avg_loss.clip(lower_bound=1e-12)
    return 100.0 - 100.0 / (1.0 + rs)


def build_daily_features_fast(
    data_root: Path,
    symbols: Sequence[str],
    *,
    subdir: str = "train",
) -> pd.DataFrame:
    """Polars-native replacement for iterating ``build_features_for_symbol``.

    Returns a pandas DataFrame with the same schema (DAILY_FEATURE_COLS +
    target_oc / target_oc_up / actual_open / actual_close / date / symbol).
    The conversion to pandas happens at the end because downstream code
    (xgbnew.dataset, backtest, model) expects pandas.
    """
    import polars as pl

    df = _read_ohlcv_polars(data_root, symbols, subdir=subdir)

    # Compute previous-close / previous-high / previous-low (lag 1) per symbol.
    df = df.with_columns([
        pl.col("close").shift(1).over("symbol").alias("prev_close"),
        pl.col("high").shift(1).over("symbol").alias("prev_high"),
        pl.col("low").shift(1).over("symbol").alias("prev_low"),
        pl.col("volume").shift(1).over("symbol").alias("prev_vol"),
        pl.col("close").shift(2).over("symbol").alias("close_lag2"),
    ])

    # ── returns ──────────────────────────────────────────────────────────
    def ret(n):
        return (
            pl.col("prev_close") / pl.col("close").shift(1 + n).over("symbol") - 1.0
        ).alias(f"ret_{n}d")

    df = df.with_columns([ret(1), ret(2), ret(5), ret(10), ret(20)])

    # ── RSI(14) — Wilder's smoothing (EWM alpha=1/14, adjust=False) ──────
    # Mirrors xgbnew.features._rsi_series so the two paths agree.
    delta = pl.col("prev_close").diff().over("symbol")
    gain = pl.when(delta > 0).then(delta).otherwise(0.0)
    loss = pl.when(delta < 0).then(-delta).otherwise(0.0)
    df = df.with_columns([
        gain.ewm_mean(alpha=1.0 / 14.0, adjust=False, min_samples=14).over("symbol").alias("_avg_gain"),
        loss.ewm_mean(alpha=1.0 / 14.0, adjust=False, min_samples=14).over("symbol").alias("_avg_loss"),
    ])
    df = df.with_columns(
        pl.when(pl.col("_avg_loss") > 1e-10)
        .then(100.0 - 100.0 / (1.0 + pl.col("_avg_gain") / pl.col("_avg_loss")))
        .otherwise(100.0)
        .alias("rsi_14")
    )

    # ── Realized vol (log-return std, annualised). log_ret is on prev_close. ──
    log_ret = (pl.col("prev_close") / pl.col("prev_close").shift(1).over("symbol")).log()
    df = df.with_columns([
        (log_ret.rolling_std(window_size=5, min_samples=3).over("symbol") * np.sqrt(252)).alias("vol_5d"),
        (log_ret.rolling_std(window_size=20, min_samples=10).over("symbol") * np.sqrt(252)).alias("vol_20d"),
    ])

    # ── ATR(14) ──────────────────────────────────────────────────────────
    tr1 = pl.col("prev_high") - pl.col("prev_low")
    tr2 = (pl.col("prev_high") - pl.col("close_lag2")).abs()
    tr3 = (pl.col("prev_low") - pl.col("close_lag2")).abs()
    true_range = pl.max_horizontal([tr1, tr2, tr3])
    df = df.with_columns(
        (true_range.rolling_mean(window_size=14, min_samples=7).over("symbol")
         / pl.col("prev_close")).clip(upper_bound=1.0).alias("atr_14")
    )

    # ── dollar volume (log) ──────────────────────────────────────────────
    df = df.with_columns(
        (pl.col("prev_close") * pl.col("prev_vol"))
        .rolling_mean(window_size=20, min_samples=5).over("symbol")
        .clip(lower_bound=0.0).log1p().alias("dolvol_20d_log")
    )

    # ── cs_spread_bps (simplified: prev range / prev_close in bps) ───────
    # This is a volatility proxy, not the full Corwin-Schultz estimator.
    df = df.with_columns(
        (((pl.col("prev_high") - pl.col("prev_low")) / pl.col("prev_close")) * 10_000.0)
        .rolling_mean(window_size=20, min_samples=5).over("symbol")
        .clip(lower_bound=0.0, upper_bound=10_000.0).alias("cs_spread_bps")
    )

    # ── spread_bps — tiered volume-based cost, matches features._vol_spread_series.
    # Tiers are on linear dolvol (not log): larger $-vol → tighter spread.
    # Default 50 bps when dolvol is below the smallest tier.
    dolvol_lin = pl.col("dolvol_20d_log").exp() - 1.0
    df = df.with_columns(
        pl.when(dolvol_lin >= 1e10).then(2.0)
        .when(dolvol_lin >= 5e8).then(3.0)
        .when(dolvol_lin >= 1e8).then(7.0)
        .when(dolvol_lin >= 5e7).then(12.0)
        .when(dolvol_lin >= 1e7).then(25.0)
        .otherwise(50.0)
        .alias("spread_bps")
    )

    # ── 52-week highs / lows ─────────────────────────────────────────────
    df = df.with_columns([
        pl.col("prev_high").rolling_max(window_size=252, min_samples=63).over("symbol").alias("_h52"),
        pl.col("prev_low").rolling_min(window_size=252, min_samples=63).over("symbol").alias("_l52"),
    ])
    df = df.with_columns([
        (pl.col("prev_close") / pl.col("_h52")).clip(lower_bound=0.01, upper_bound=2.0).alias("price_vs_52w_high"),
        ((pl.col("prev_close") - pl.col("_l52"))
         / (pl.col("_h52") - pl.col("_l52")).clip(lower_bound=0.01)
         ).clip(lower_bound=0.0, upper_bound=1.0).alias("price_vs_52w_range"),
    ])

    # ── calendar + last-close features ──────────────────────────────────
    df = df.with_columns([
        pl.col("timestamp").dt.weekday().cast(pl.Float64).alias("day_of_week"),
        pl.col("prev_close").clip(lower_bound=0.01).log1p().alias("last_close_log"),
    ])

    # ── targets ─────────────────────────────────────────────────────────
    df = df.with_columns(
        ((pl.col("close") - pl.col("open")) / pl.col("open").clip(lower_bound=0.01))
        .clip(lower_bound=-0.5, upper_bound=0.5).alias("target_oc")
    )
    df = df.with_columns(
        (pl.col("target_oc") > 0.0).cast(pl.Int8).alias("target_oc_up")
    )

    # Keep actual_open / actual_close for backtest.
    df = df.rename({"open": "actual_open", "close": "actual_close"})

    drop_cols = [
        "prev_close", "prev_high", "prev_low", "prev_vol", "close_lag2",
        "_avg_gain", "_avg_loss", "_h52", "_l52",
    ]
    df = df.drop([c for c in drop_cols if c in df.columns])

    pdf = df.to_pandas()
    # Drop rows with all core features NaN (mirrors the pandas path's dropna).
    pdf = pdf.dropna(subset=DAILY_FEATURE_COLS[:5])
    pdf["date"] = pd.to_datetime(pdf["date"]).dt.date
    return pdf


__all__ = ["build_daily_features_fast"]
