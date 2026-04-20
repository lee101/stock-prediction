"""Dataset builder for XGBoost stock model.

Builds train / val / test DataFrames from:
  1. Daily OHLCV CSVs (wide universe, ~846 symbols)
  2. Optional Chronos2 forecast cache (JSON files per trading day)

No-lookahead guarantee: feature computation delegates to features.py which
uses only data available at end of day D-1 for each row D.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from .features import (
    CHRONOS_FEATURE_COLS,
    DAILY_FEATURE_COLS,
    HOURLY_FEATURE_COLS,
    add_cross_sectional_ranks,
    build_features_for_symbol,
    build_features_for_symbol_hourly,
)

logger = logging.getLogger(__name__)


# ── CSV loader (reuse screener approach) ─────────────────────────────────────

def _load_symbol_csv(symbol: str, data_root: Path) -> pd.DataFrame | None:
    for sub in ("train", "stocks", ""):
        path = (data_root / sub / f"{symbol}.csv") if sub else (data_root / f"{symbol}.csv")
        if path.exists():
            try:
                df = pd.read_csv(path)
            except Exception:
                return None
            df.columns = df.columns.str.strip().str.lower()
            for col in ("open", "high", "low", "close"):
                if col not in df.columns:
                    return None
            ts_col = next((c for c in ("timestamp", "date") if c in df.columns), df.columns[0])
            df["timestamp"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
            df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
            df = df.drop_duplicates(subset=["timestamp"], keep="last")
            for col in ("open", "high", "low", "close"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["close"])
            if "volume" not in df.columns:
                df["volume"] = 0.0
            else:
                df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
            if len(df) < 60:
                return None
            return df[["timestamp", "open", "high", "low", "close", "volume"]]
    return None


# ── Chronos2 cache loader ────────────────────────────────────────────────────

def load_chronos_cache(cache_dir: Path) -> dict[date, dict[str, dict]]:
    """Load all cached Chronos2 forecasts.

    Returns {trading_day: {symbol: {oc_return_pct, cc_return_pct, ...}}}
    Supports both old (float) and new (dict) cache formats.
    """
    out: dict[date, dict[str, dict]] = {}
    if not cache_dir.exists():
        return out
    for path in sorted(cache_dir.glob("*.json")):
        try:
            day = date.fromisoformat(path.stem)
            raw = json.loads(path.read_text(encoding="utf-8"))
            # Upgrade old flat-float format
            if raw and isinstance(next(iter(raw.values())), (int, float)):
                raw = {sym: {"cc_return_pct": float(v), "oc_return_pct": None,
                             "pred_open": None, "pred_close": None}
                       for sym, v in raw.items()}
            out[day] = raw  # type: ignore[assignment]
        except Exception:
            continue
    return out


# ── per-symbol feature iteration (memory-efficient) ─────────────────────────

def _symbol_features_iter(
    data_root: Path,
    symbols: list[str],
    start_date: date | None = None,
    end_date: date | None = None,
    min_rows: int = 100,
) -> Iterator[pd.DataFrame]:
    """Yield feature DataFrames per symbol, filtered to [start_date, end_date]."""
    for symbol in symbols:
        df = _load_symbol_csv(symbol, data_root)
        if df is None or len(df) < min_rows:
            continue
        feat = build_features_for_symbol(df, symbol=symbol)
        if start_date is not None:
            feat = feat[feat["date"] >= start_date]
        if end_date is not None:
            feat = feat[feat["date"] <= end_date]
        feat = feat.dropna(subset=DAILY_FEATURE_COLS[:5])  # require core features
        if len(feat) < 10:
            continue
        yield feat


def _attach_chronos_features(
    feat_df: pd.DataFrame,
    chronos_cache: dict[date, dict[str, dict]],
) -> pd.DataFrame:
    """Merge Chronos2 forecast features into feature DataFrame.

    Sets chronos_available=1 and fills oc/cc return + pred_range when available.
    Falls back to zeros when not available (model learned to ignore those).
    """
    feat_df = feat_df.copy()
    feat_df["chronos_oc_return"]  = 0.0
    feat_df["chronos_cc_return"]  = 0.0
    feat_df["chronos_pred_range"] = 0.0
    feat_df["chronos_available"]  = 0.0

    if not chronos_cache:
        return feat_df

    for i, row in feat_df.iterrows():
        day = row["date"]
        sym = row.get("symbol", "")
        if day not in chronos_cache:
            continue
        info = chronos_cache[day].get(sym)
        if info is None:
            continue

        feat_df.at[i, "chronos_available"] = 1.0

        oc = info.get("oc_return_pct")
        if oc is not None and np.isfinite(float(oc)):
            feat_df.at[i, "chronos_oc_return"] = float(oc)

        cc = info.get("cc_return_pct")
        if cc is not None and np.isfinite(float(cc)):
            feat_df.at[i, "chronos_cc_return"] = float(cc)

        po = info.get("pred_open")
        pc = info.get("pred_close")
        if (po is not None and pc is not None
                and np.isfinite(float(po)) and np.isfinite(float(pc)) and float(po) > 0):
            rng = (float(pc) - float(po)) / float(po)
            feat_df.at[i, "chronos_pred_range"] = float(rng)

    return feat_df


def _attach_chronos_features_fast(
    feat_df: pd.DataFrame,
    chronos_cache: dict[date, dict[str, dict]],
) -> pd.DataFrame:
    """Vectorized version of _attach_chronos_features (faster for large DataFrames)."""
    feat_df = feat_df.copy()
    feat_df["chronos_oc_return"]  = 0.0
    feat_df["chronos_cc_return"]  = 0.0
    feat_df["chronos_pred_range"] = 0.0
    feat_df["chronos_available"]  = 0.0

    if not chronos_cache or "symbol" not in feat_df.columns:
        return feat_df

    rows = []
    for day, sym_map in chronos_cache.items():
        for sym, info in sym_map.items():
            oc = info.get("oc_return_pct")
            cc = info.get("cc_return_pct")
            po = info.get("pred_open")
            pc = info.get("pred_close")
            rng = None
            if (po is not None and pc is not None
                    and np.isfinite(float(po) if po else float("nan"))
                    and float(po) > 0):
                rng = (float(pc) - float(po)) / float(po)
            rows.append({
                "date": day,
                "symbol": sym,
                "_oc": float(oc) if oc is not None and np.isfinite(float(oc)) else 0.0,
                "_cc": float(cc) if cc is not None and np.isfinite(float(cc)) else 0.0,
                "_rng": float(rng) if rng is not None and np.isfinite(rng) else 0.0,
                "_avail": 1.0,
            })

    if not rows:
        return feat_df

    chron_df = pd.DataFrame(rows)
    merged = feat_df.merge(chron_df, on=["date", "symbol"], how="left")
    for col, key in [("chronos_oc_return", "_oc"), ("chronos_cc_return", "_cc"),
                     ("chronos_pred_range", "_rng"), ("chronos_available", "_avail")]:
        feat_df[col] = merged[key].fillna(0.0).values

    return feat_df


# ── Main dataset builders ─────────────────────────────────────────────────────

def build_daily_dataset(
    data_root: Path,
    symbols: list[str],
    train_start: date,
    train_end: date,
    val_start: date,
    val_end: date,
    test_start: date,
    test_end: date,
    chronos_cache: dict[date, dict[str, dict]] | None = None,
    min_dollar_vol: float = 1e6,
    fast_features: bool = False,
    include_cross_sectional_ranks: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build train / val / test DataFrames for daily XGBoost model.

    Args:
        data_root: Root of OHLCV CSV data.
        symbols: List of symbols to include.
        train_start / train_end: Training period.
        val_start / val_end: Validation period.
        test_start / test_end: Test period (receives Chronos2 features if cache given).
        chronos_cache: Optional cached Chronos2 forecasts (from load_chronos_cache).
        min_dollar_vol: Minimum average daily dollar volume for a row to be included.
        fast_features: If True, use polars-native builder (~3× faster). RSI and
            rolling-std columns have small numerical divergence (corr > 0.98) from
            the pandas path; treat A/B numbers from the two paths as comparable
            but not bit-identical.

    Returns:
        (train_df, val_df, test_df) — each with DAILY_FEATURE_COLS + target columns.
    """
    if fast_features:
        return _build_daily_dataset_fast(
            data_root, symbols, train_start, train_end, val_start, val_end,
            test_start, test_end, chronos_cache=chronos_cache,
            min_dollar_vol=min_dollar_vol,
            include_cross_sectional_ranks=include_cross_sectional_ranks,
        )

    train_parts, val_parts, test_parts = [], [], []

    logger.info("Building dataset for %d symbols...", len(symbols))
    for i, sym in enumerate(symbols):
        df = _load_symbol_csv(sym, data_root)
        if df is None:
            continue
        feat = build_features_for_symbol(df, symbol=sym)
        feat = feat.dropna(subset=DAILY_FEATURE_COLS[:5])

        # Liquidity filter
        feat = feat[feat["dolvol_20d_log"] >= np.log1p(min_dollar_vol)]

        if len(feat) < 50:
            continue

        tr = feat[(feat["date"] >= train_start) & (feat["date"] <= train_end)].copy()
        va = feat[(feat["date"] >= val_start)   & (feat["date"] <= val_end)].copy()
        te = feat[(feat["date"] >= test_start)  & (feat["date"] <= test_end)].copy()

        if len(tr) > 0:
            train_parts.append(tr)
        if len(va) > 0:
            val_parts.append(va)
        if len(te) > 0:
            test_parts.append(te)

        if (i + 1) % 100 == 0:
            logger.info("  processed %d / %d symbols", i + 1, len(symbols))

    def _concat(parts: list[pd.DataFrame]) -> pd.DataFrame:
        if not parts:
            return pd.DataFrame()
        df = pd.concat(parts, ignore_index=True)
        # Chronos features default to zero
        for col in CHRONOS_FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0.0
        return df

    train_df = _concat(train_parts)
    val_df   = _concat(val_parts)
    test_df  = _concat(test_parts)

    # Cross-sectional per-day ranks must be computed on the merged multi-symbol
    # panel (not per-symbol). Compute within each split independently so a
    # row's rank is only over symbols present in the SAME split+day — avoids
    # train leaking into val/test.
    if include_cross_sectional_ranks:
        if len(train_df) > 0:
            train_df = add_cross_sectional_ranks(train_df)
        if len(val_df) > 0:
            val_df = add_cross_sectional_ranks(val_df)
        if len(test_df) > 0:
            test_df = add_cross_sectional_ranks(test_df)

    # Attach Chronos2 features to test set (and optionally val)
    if chronos_cache and len(test_df) > 0:
        test_df = _attach_chronos_features_fast(test_df, chronos_cache)

    logger.info(
        "Dataset sizes: train=%d val=%d test=%d rows",
        len(train_df), len(val_df), len(test_df),
    )
    return train_df, val_df, test_df


def _build_daily_dataset_fast(
    data_root: Path,
    symbols: list[str],
    train_start: date,
    train_end: date,
    val_start: date,
    val_end: date,
    test_start: date,
    test_end: date,
    chronos_cache: dict[date, dict[str, dict]] | None = None,
    min_dollar_vol: float = 1e6,
    include_cross_sectional_ranks: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Polars-native dataset builder. Same splits and columns as the pandas
    path, but features come from ``xgbnew.features_fast.build_daily_features_fast``
    — one ETL pass over all symbols instead of N per-symbol passes.
    """
    from .features_fast import build_daily_features_fast

    logger.info("Building dataset (fast/polars) for %d symbols...", len(symbols))
    feat = build_daily_features_fast(data_root, symbols)
    feat = feat.dropna(subset=DAILY_FEATURE_COLS[:5])
    feat = feat[feat["dolvol_20d_log"] >= np.log1p(min_dollar_vol)]
    # Drop raw OHLCV columns left over from the polars path so the returned
    # DataFrame has the same column set as the pandas builder.
    for _extra in ("high", "low", "volume"):
        if _extra in feat.columns:
            feat = feat.drop(columns=_extra)
    # Drop any symbol group that ended up with < 50 usable rows, to match
    # the behaviour of the pandas path.
    sym_counts = feat.groupby("symbol").size()
    keep = sym_counts[sym_counts >= 50].index
    feat = feat[feat["symbol"].isin(keep)]

    tr = feat[(feat["date"] >= train_start) & (feat["date"] <= train_end)].copy()
    va = feat[(feat["date"] >= val_start)   & (feat["date"] <= val_end)].copy()
    te = feat[(feat["date"] >= test_start)  & (feat["date"] <= test_end)].copy()

    for df in (tr, va, te):
        for col in CHRONOS_FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0.0

    if include_cross_sectional_ranks:
        if len(tr) > 0:
            tr = add_cross_sectional_ranks(tr)
        if len(va) > 0:
            va = add_cross_sectional_ranks(va)
        if len(te) > 0:
            te = add_cross_sectional_ranks(te)

    if chronos_cache and len(te) > 0:
        te = _attach_chronos_features_fast(te, chronos_cache)

    logger.info(
        "Dataset sizes (fast): train=%d val=%d test=%d rows",
        len(tr), len(va), len(te),
    )
    return tr, va, te


__all__ = [
    "load_chronos_cache",
    "build_daily_dataset",
    "build_features_for_symbol",
    "build_hourly_dataset",
    "load_hourly_symbol_csv",
    "list_hourly_symbols",
]


# ── Hourly CSV loader ────────────────────────────────────────────────────────

def _classify_symbol(symbol: str) -> str:
    """Return ``'crypto'`` for crypto pairs (USD/USDT/USDC/FDUSD/BUSD/TUSD/USDP
    quote suffix) else ``'stocks'``. Uses ``src.symbol_utils.is_crypto_symbol``
    when available so the rule stays in lock-step with the prod fee path."""
    sym = str(symbol).upper().strip()
    try:
        from src.symbol_utils import is_crypto_symbol  # type: ignore
        return "crypto" if bool(is_crypto_symbol(sym)) else "stocks"
    except Exception:
        for q in ("USDT", "USDC", "FDUSD", "BUSD", "TUSD", "USDP"):
            if sym.endswith(q) and len(sym) > len(q):
                return "crypto"
        if sym.endswith("USD") and len(sym) > 3:
            return "crypto"
        return "stocks"


def load_hourly_symbol_csv(symbol: str, data_root: Path) -> pd.DataFrame | None:
    """Load one symbol's hourly OHLCV CSV from ``<data_root>/<kind>/<symbol>.csv``.

    Tries ``stocks`` and ``crypto`` subdirs. Returns a cleaned DataFrame with
    ``timestamp, open, high, low, close, volume, symbol, _kind`` columns, or
    ``None`` if the file is missing / malformed / has <200 rows.

    ``_kind`` is derived from the *symbol* (USD-quote suffix → crypto), not
    the subdirectory — the legacy ``trainingdatahourly/crypto/`` folder
    contains a few stock CSVs (AAPL, AMZN, DBX) that should still be priced
    with stock fees.
    """
    root = Path(data_root)
    for subdir in ("stocks", "crypto"):
        path = root / subdir / f"{symbol}.csv"
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            return None
        df.columns = df.columns.str.strip().str.lower()
        if not {"timestamp", "open", "high", "low", "close"}.issubset(df.columns):
            return None
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        df = df.drop_duplicates(subset=["timestamp"], keep="last")
        for col in ("open", "high", "low", "close"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close"])
        # Drop bars with non-positive prices (log-returns would blow to -inf)
        df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)]
        if "volume" not in df.columns:
            df["volume"] = 0.0
        else:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
        if len(df) < 200:
            return None
        df["symbol"] = symbol.upper()
        df["_kind"] = _classify_symbol(symbol)
        return df[["timestamp", "open", "high", "low", "close", "volume", "symbol", "_kind"]]
    return None


def list_hourly_symbols(
    data_root: Path,
    *,
    universe: str = "stocks",
) -> list[str]:
    """Return sorted symbol list for the requested universe.

    ``universe`` in {"stocks", "crypto", "both"}.
    """
    root = Path(data_root)
    if universe not in ("stocks", "crypto", "both"):
        raise ValueError(f"universe must be stocks|crypto|both, got {universe!r}")
    syms: list[str] = []
    # Walk both subdirs; classification (stocks vs crypto) goes by symbol, not
    # the subdir, since the legacy ``crypto/`` dir holds a few stock CSVs.
    for subdir in ("stocks", "crypto"):
        d = root / subdir
        if not d.exists():
            continue
        for path in sorted(d.glob("*.csv")):
            syms.append(path.stem.upper())
    # De-duplicate while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for s in syms:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    if universe == "both":
        return deduped
    return [s for s in deduped if _classify_symbol(s) == universe]


def build_hourly_dataset(
    data_root: Path,
    symbols: list[str] | None,
    *,
    train_start: pd.Timestamp | None,
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
    test_end: pd.Timestamp,
    universe: str = "stocks",
    min_bars: int = 400,
    min_dollar_vol: float = 5e5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Build train / val / test DataFrames from hourly OHLCV CSVs.

    Splits by timestamp:
      train:  (..., train_end]
      val:    (train_end, val_end]
      test:   (val_end, test_end]

    Returns (train_df, val_df, test_df, kind_map) where ``kind_map`` maps
    symbol → 'stocks' | 'crypto' (useful for fee / 24-7 calendar branching).

    Features: HOURLY_FEATURE_COLS + target_oc / target_oc_up / actual_open /
    actual_close / timestamp / symbol.

    ``min_bars``: drop symbols that yield fewer usable feature rows than this.
    ``min_dollar_vol``: per-bar dollar volume floor (hourly bars, so lower than
    the daily 5e6).
    """
    root = Path(data_root)
    if symbols is None:
        symbols = list_hourly_symbols(root, universe=universe)

    train_end_ts = pd.Timestamp(train_end, tz="UTC") if pd.Timestamp(train_end).tzinfo is None else pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end, tz="UTC") if pd.Timestamp(val_end).tzinfo is None else pd.Timestamp(val_end)
    test_end_ts = pd.Timestamp(test_end, tz="UTC") if pd.Timestamp(test_end).tzinfo is None else pd.Timestamp(test_end)
    train_start_ts = None
    if train_start is not None:
        train_start_ts = pd.Timestamp(train_start, tz="UTC") if pd.Timestamp(train_start).tzinfo is None else pd.Timestamp(train_start)

    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    kind_map: dict[str, str] = {}

    logger.info("build_hourly_dataset: scanning %d symbols in %s", len(symbols), root)
    for i, sym in enumerate(symbols):
        raw = load_hourly_symbol_csv(sym, root)
        if raw is None:
            continue
        kind = str(raw["_kind"].iloc[0])
        feat = build_features_for_symbol_hourly(raw, symbol=sym)
        # Core feature subset must be non-NaN (ret_1h, ret_4h, rsi_14, vol_4h)
        feat = feat.dropna(subset=HOURLY_FEATURE_COLS[:4])
        # Liquidity filter on hourly dolvol
        if "dolvol_4h_log" in feat.columns:
            feat = feat[feat["dolvol_4h_log"] >= np.log1p(min_dollar_vol)]
        if len(feat) < min_bars:
            continue
        kind_map[sym.upper()] = kind

        tr_mask = feat["timestamp"] <= train_end_ts
        if train_start_ts is not None:
            tr_mask = tr_mask & (feat["timestamp"] >= train_start_ts)
        va_mask = (feat["timestamp"] > train_end_ts) & (feat["timestamp"] <= val_end_ts)
        te_mask = (feat["timestamp"] > val_end_ts) & (feat["timestamp"] <= test_end_ts)

        if tr_mask.any():
            train_parts.append(feat[tr_mask].copy())
        if va_mask.any():
            val_parts.append(feat[va_mask].copy())
        if te_mask.any():
            test_parts.append(feat[te_mask].copy())

        if (i + 1) % 100 == 0:
            logger.info("  processed %d / %d symbols", i + 1, len(symbols))

    def _concat(parts: list[pd.DataFrame]) -> pd.DataFrame:
        if not parts:
            return pd.DataFrame()
        df = pd.concat(parts, ignore_index=True)
        # Provide a 'date' column so regime-gate / multi-window logic can bucket.
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date
        return df

    return _concat(train_parts), _concat(val_parts), _concat(test_parts), kind_map
