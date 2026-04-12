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
    build_features_for_symbol,
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

    Returns:
        (train_df, val_df, test_df) — each with DAILY_FEATURE_COLS + target columns.
    """
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

    # Attach Chronos2 features to test set (and optionally val)
    if chronos_cache and len(test_df) > 0:
        test_df = _attach_chronos_features_fast(test_df, chronos_cache)

    logger.info(
        "Dataset sizes: train=%d val=%d test=%d rows",
        len(train_df), len(val_df), len(test_df),
    )
    return train_df, val_df, test_df


__all__ = [
    "load_chronos_cache",
    "build_daily_dataset",
    "build_features_for_symbol",
]
