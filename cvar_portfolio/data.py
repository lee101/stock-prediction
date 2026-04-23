"""Load our trainingdata/ daily OHLCV CSVs into a (dates x tickers) price panel.

Mirrors the loader convention in `xgbnew/dataset.py::_load_symbol_csv` so we
stay on the same data the XGB daily-trader trains on (see
project_xgb_stale_training_csvs.md).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _load_close(symbol: str, data_root: Path) -> pd.Series | None:
    for sub in ("", "stocks", "train"):
        path = (data_root / sub / f"{symbol}.csv") if sub else (data_root / f"{symbol}.csv")
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            return None
        df.columns = df.columns.str.strip().str.lower()
        if "close" not in df.columns:
            return None
        ts_col = next((c for c in ("timestamp", "date") if c in df.columns), df.columns[0])
        df["timestamp"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        df = df.drop_duplicates(subset=["timestamp"], keep="last")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"])
        if len(df) < 60:
            return None
        s = pd.Series(df["close"].values, index=df["timestamp"].dt.tz_convert("UTC").dt.floor("D"), name=symbol)
        # collapse any same-day dups to last
        s = s.groupby(s.index).last()
        return s
    return None


def load_price_panel(
    symbols: list[str],
    data_root: Path,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    min_days: int = 252,
    min_avg_dollar_vol: float | None = None,
) -> pd.DataFrame:
    """Load close prices for the given tickers into a wide DataFrame.

    Symbols with insufficient history or non-overlapping dates are dropped.
    """
    frames: list[pd.Series] = []
    for sym in symbols:
        s = _load_close(sym, data_root)
        if s is None or len(s) < min_days:
            continue
        frames.append(s)
    if not frames:
        raise RuntimeError("No usable symbols loaded.")
    df = pd.concat(frames, axis=1).sort_index()
    if start is not None:
        df = df.loc[pd.Timestamp(start, tz="UTC"):]
    if end is not None:
        df = df.loc[:pd.Timestamp(end, tz="UTC")]
    df = df.ffill(limit=3).dropna(axis=1, thresh=max(min_days, int(0.9 * len(df))))
    df = df.dropna()
    if min_avg_dollar_vol is not None:
        keep = _filter_by_dollar_vol(df.columns.tolist(), data_root, df.index[-1], min_avg_dollar_vol)
        df = df[keep]
    return df


def _filter_by_dollar_vol(
    symbols: list[str], data_root: Path, asof: pd.Timestamp, min_dollar_vol: float, lookback: int = 20
) -> list[str]:
    keep: list[str] = []
    for sym in symbols:
        for sub in ("", "stocks", "train"):
            path = (data_root / sub / f"{sym}.csv") if sub else (data_root / f"{sym}.csv")
            if path.exists():
                try:
                    df = pd.read_csv(path)
                except Exception:
                    break
                df.columns = df.columns.str.strip().str.lower()
                if "close" not in df.columns or "volume" not in df.columns:
                    break
                ts_col = next((c for c in ("timestamp", "date") if c in df.columns), df.columns[0])
                df["timestamp"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
                df = df.dropna(subset=["timestamp"])
                df = df[df["timestamp"] <= asof].tail(lookback)
                if len(df) < lookback:
                    break
                dol = (pd.to_numeric(df["close"], errors="coerce") * pd.to_numeric(df["volume"], errors="coerce")).dropna()
                if dol.mean() >= min_dollar_vol:
                    keep.append(sym)
                break
    return keep


def read_symbol_list(path: Path) -> list[str]:
    out = []
    for line in Path(path).read_text().splitlines():
        s = line.strip().upper()
        if s and not s.startswith("#"):
            out.append(s)
    return out


def log_returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns with first NaN dropped, zero-fill intra-series NaNs."""
    r = np.log(prices) - np.log(prices.shift(1))
    r = r.dropna(how="all").fillna(0.0)
    return r
