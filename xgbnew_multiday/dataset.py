"""Dataset builder for multi-horizon XGBoost.

Reuses ``xgbnew.features_fast.build_daily_features_fast`` to get the same
no-lookahead feature set as the deployed single-day model, then adds extra
N-day forward-return columns for N in HORIZONS.

For horizon N we define:
    target_fwd_{N}d      = close_{t+N-1} / open_{t} - 1   (trade return if
                            we buy at the open of day t and sell at the close
                            of day t+N-1)
    target_fwd_{N}d_up   = (target_fwd_{N}d > 0).astype(int)
    abs_fwd_{N}d         = abs(target_fwd_{N}d)           (magnitude — used by
                            the meta-selector's expected-value estimate)

N=1 is identical to the xgbnew ``target_oc`` label by construction.

The ``valid_fwd_{N}d`` mask is True only when a close exists N-1 bars ahead
(i.e., the row is not in the trailing N-1 rows of a symbol's history).
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

HORIZONS: tuple[int, ...] = (1, 2, 3, 5, 10)


def _fwd_columns_for_symbol(
    df: pd.DataFrame, horizons: Sequence[int] = HORIZONS,
) -> pd.DataFrame:
    """Given a symbol's bar-sorted DataFrame with 'open' and 'close' columns,
    return a DataFrame of forward-N-day targets indexed to ``df.index``."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    open_t = df["open"].astype(float)
    close = df["close"].astype(float)

    out = pd.DataFrame(index=df.index)
    for n in horizons:
        fwd_close = close.shift(-(n - 1))
        ret = fwd_close / open_t.clip(lower=0.01) - 1.0
        ret = ret.clip(-0.7, 0.7)  # clamp outliers but leave room for wide moves
        valid = fwd_close.notna() & (open_t > 0) & (fwd_close > 0)
        out[f"target_fwd_{n}d"] = ret
        out[f"target_fwd_{n}d_up"] = (ret > 0.0).astype(np.int8)
        out[f"abs_fwd_{n}d"] = ret.abs()
        out[f"valid_fwd_{n}d"] = valid.astype(np.int8)
    return out


def _load_one_symbol(symbol: str, data_root: Path) -> pd.DataFrame | None:
    for sub in ("", "stocks", "train"):
        path = (data_root / sub / f"{symbol}.csv") if sub else (data_root / f"{symbol}.csv")
        if path.exists():
            try:
                df = pd.read_csv(path)
            except Exception:
                return None
            df.columns = df.columns.str.strip().str.lower()
            required = {"open", "high", "low", "close"}
            if not required.issubset(df.columns):
                return None
            ts_col = next((c for c in ("timestamp", "date") if c in df.columns), df.columns[0])
            df["timestamp"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
            df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
            df = df.drop_duplicates(subset=["timestamp"], keep="last")
            for col in ("open", "high", "low", "close"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["close", "open"])
            if "volume" not in df.columns:
                df["volume"] = 0.0
            else:
                df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
            if len(df) < 60:
                return None
            df["symbol"] = symbol
            df["date"] = df["timestamp"].dt.date
            return df[["timestamp", "date", "symbol", "open", "high", "low", "close", "volume"]]
    return None


def build_multi_horizon_dataset(
    data_root: Path,
    symbols: Iterable[str],
    *,
    train_end: date,
    test_start: date,
    test_end: date,
    horizons: Sequence[int] = HORIZONS,
    use_fast_features: bool = True,
    min_dollar_vol: float = 5e6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build train + test DataFrames with multi-horizon targets.

    Uses xgbnew.features_fast for the feature columns (no-lookahead) and then
    joins in forward-N-day target columns computed per-symbol from raw OHLC.

    Train period: everything up to (and including) ``train_end``.
    Test period: ``[test_start, test_end]`` inclusive.
    """
    symbols = list(symbols)
    if use_fast_features:
        from xgbnew.features_fast import build_daily_features_fast
        feat_df = build_daily_features_fast(data_root, symbols)
    else:
        # Slow fallback; build per-symbol and concatenate
        from xgbnew.features import build_features_for_symbol
        parts = []
        for sym in symbols:
            raw = _load_one_symbol(sym, data_root)
            if raw is None:
                continue
            f = build_features_for_symbol(
                raw.rename(columns={})[["timestamp", "open", "high", "low", "close", "volume"]],
                symbol=sym,
            )
            parts.append(f)
        feat_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    if len(feat_df) == 0:
        return feat_df, feat_df

    # Liquidity filter (same default as xgbnew)
    if "dolvol_20d_log" in feat_df.columns:
        feat_df = feat_df[feat_df["dolvol_20d_log"] >= np.log1p(min_dollar_vol)]

    # Build forward targets per symbol from raw OHLC — the feature path only
    # exposes same-day OC (`target_oc`). We need multi-day forward windows.
    fwd_parts = []
    for sym in symbols:
        raw = _load_one_symbol(sym, data_root)
        if raw is None:
            continue
        # Raw is sorted; compute forward targets
        fwd = _fwd_columns_for_symbol(raw, horizons)
        fwd["symbol"] = sym
        fwd["date"] = raw["date"].values
        fwd_parts.append(fwd)
    if not fwd_parts:
        return feat_df.iloc[:0], feat_df.iloc[:0]

    fwd_df = pd.concat(fwd_parts, ignore_index=True)

    # Merge — keep only rows present in both feature and fwd DataFrames
    merged = feat_df.merge(fwd_df, on=["symbol", "date"], how="inner")

    train_df = merged[merged["date"] <= train_end].copy()
    test_df = merged[(merged["date"] >= test_start) & (merged["date"] <= test_end)].copy()

    return train_df, test_df
