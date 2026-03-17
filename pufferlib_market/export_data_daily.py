"""
Export daily OHLCV data to a flat binary file for the C trading env.

This is a daily (calendar-day) variant of `export_data.py` intended for running
the PufferLib C environment on `trainingdata/` style datasets.

Binary format (v2):
  Header (64 bytes):
    magic:            4 bytes  "MKTD"
    version:          uint32   (2)
    num_symbols:      uint32
    num_timesteps:    uint32   (calendar days)
    features_per_sym: uint32   (always 16)
    price_features:   uint32   (always 5: OHLCV)
    padding:          40 bytes

  Symbol table: num_symbols * 16 bytes (null-padded ASCII names)

  Feature data: float32[num_timesteps][num_symbols][features_per_sym]
    (Price-only features, no Chronos cache required.)

  Price data:   float32[num_timesteps][num_symbols][5]  (open, high, low, close, volume)

  Tradable mask (optional, v2+): uint8[num_timesteps][num_symbols]
    1 when the asset has an actual bar for that calendar day; 0 when the day is
    market-closed (e.g. weekends/holidays for stocks). The simulator will refuse
    to open/close positions in symbols with tradable=0 at that timestep.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES_PER_SYM = 16
PRICE_FEATURES = 5
MAGIC = b"MKTD"
VERSION = 2


def _normalise_daily_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "date" in df.columns:
        ts = pd.to_datetime(df["date"], utc=True, errors="coerce")
    elif "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    else:
        raise ValueError("CSV must contain 'timestamp' or 'date' column")
    df["date"] = ts.dt.floor("D")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date").drop_duplicates(subset="date", keep="last")
    return df.set_index("date")


def _coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not df.columns.duplicated().any():
        return df

    merged: dict[str, pd.Series] = {}
    for name in dict.fromkeys(str(col) for col in df.columns):
        cols = df.loc[:, df.columns == name]
        if isinstance(cols, pd.Series):
            merged[name] = cols
            continue
        if cols.shape[1] == 1:
            merged[name] = cols.iloc[:, 0]
            continue

        combined = cols.bfill(axis=1).iloc[:, 0]
        for col_idx in range(cols.shape[1]):
            series = cols.iloc[:, col_idx]
            overlap = combined.notna() & series.notna()
            if not bool(overlap.any()):
                continue
            left = pd.to_numeric(combined[overlap], errors="coerce")
            right = pd.to_numeric(series[overlap], errors="coerce")
            valid = left.notna() & right.notna()
            if bool(valid.any()) and not np.allclose(
                left[valid].to_numpy(dtype=float, copy=False),
                right[valid].to_numpy(dtype=float, copy=False),
                atol=1e-12,
                rtol=0.0,
                equal_nan=True,
            ):
                raise ValueError(f"Conflicting duplicate column values for {name!r}")
        merged[name] = combined

    return pd.DataFrame(merged, index=df.index)


def load_price_data(symbol: str, data_root: Path) -> pd.DataFrame:
    """Load daily OHLCV CSV for a symbol from either flat or {crypto,stocks}/ layout."""
    symbol = symbol.upper()
    candidates = [
        data_root / f"{symbol}.csv",
        data_root / "crypto" / f"{symbol}.csv",
        data_root / "stocks" / f"{symbol}.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(f"No price data for {symbol} under {data_root}")
    df = pd.read_csv(path)
    df.columns = [str(c).lower() for c in df.columns]
    df = _coalesce_duplicate_columns(df)
    df = _normalise_daily_index(df)
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return df[required].astype(float)


def compute_daily_features(price_df: pd.DataFrame) -> pd.DataFrame:
    """Compute 16 price-only daily features for the RL agent."""
    close = price_df["close"].astype(float)
    high = price_df["high"].astype(float)
    low = price_df["low"].astype(float)
    volume = price_df["volume"].astype(float)

    feat = pd.DataFrame(index=price_df.index)

    ret_1d = close.pct_change(1).fillna(0.0)
    feat["return_1d"] = ret_1d.clip(-0.5, 0.5)
    feat["return_5d"] = close.pct_change(5).fillna(0.0).clip(-1.0, 1.0)
    feat["return_20d"] = close.pct_change(20).fillna(0.0).clip(-2.0, 2.0)

    feat["volatility_5d"] = ret_1d.rolling(5, min_periods=1).std(ddof=0).fillna(0.01).clip(0.0, 1.0)
    feat["volatility_20d"] = ret_1d.rolling(20, min_periods=1).std(ddof=0).fillna(0.01).clip(0.0, 1.0)

    ma5 = close.rolling(5, min_periods=1).mean()
    ma20 = close.rolling(20, min_periods=1).mean()
    ma60 = close.rolling(60, min_periods=1).mean()
    feat["ma_delta_5d"] = ((close - ma5) / ma5.clip(lower=1e-8)).fillna(0.0).clip(-0.5, 0.5)
    feat["ma_delta_20d"] = ((close - ma20) / ma20.clip(lower=1e-8)).fillna(0.0).clip(-0.5, 0.5)
    feat["ma_delta_60d"] = ((close - ma60) / ma60.clip(lower=1e-8)).fillna(0.0).clip(-0.5, 0.5)

    # ATR % of close (14d)
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    atr14 = tr.rolling(14, min_periods=1).mean()
    feat["atr_pct_14d"] = (atr14 / close.clip(lower=1e-8)).fillna(0.0).clip(0.0, 0.5)
    feat["range_pct_1d"] = ((high - low) / close.clip(lower=1e-8)).fillna(0.0).clip(0.0, 0.5)

    feat["trend_20d"] = close.pct_change(20).fillna(0.0).clip(-2.0, 2.0)
    feat["trend_60d"] = close.pct_change(60).fillna(0.0).clip(-3.0, 3.0)

    roll_max_20 = close.rolling(20, min_periods=1).max()
    roll_max_60 = close.rolling(60, min_periods=1).max()
    feat["drawdown_20d"] = ((close - roll_max_20) / roll_max_20.clip(lower=1e-8)).fillna(0.0).clip(-1.0, 0.0)
    feat["drawdown_60d"] = ((close - roll_max_60) / roll_max_60.clip(lower=1e-8)).fillna(0.0).clip(-1.0, 0.0)

    log_vol = np.log1p(volume.clip(lower=0.0))
    log_vol_mean20 = log_vol.rolling(20, min_periods=1).mean()
    log_vol_std20 = log_vol.rolling(20, min_periods=1).std(ddof=0).replace(0.0, 1.0)
    feat["log_volume_z20d"] = ((log_vol - log_vol_mean20) / log_vol_std20).fillna(0.0).clip(-5.0, 5.0)
    feat["log_volume_delta_5d"] = (log_vol - log_vol.rolling(5, min_periods=1).mean()).fillna(0.0).clip(-10.0, 10.0)

    expected_cols = [
        "return_1d",
        "return_5d",
        "return_20d",
        "volatility_5d",
        "volatility_20d",
        "ma_delta_5d",
        "ma_delta_20d",
        "ma_delta_60d",
        "atr_pct_14d",
        "range_pct_1d",
        "trend_20d",
        "trend_60d",
        "drawdown_20d",
        "drawdown_60d",
        "log_volume_z20d",
        "log_volume_delta_5d",
    ]
    feat = feat[expected_cols].fillna(0.0).astype(np.float32)
    if feat.shape[1] != FEATURES_PER_SYM:
        raise AssertionError("feature column mismatch")
    return feat


def export_binary(
    symbols: list[str],
    data_root: Path,
    output_path: Path,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    min_days: int = 200,
) -> None:
    symbols = [s.strip().upper() for s in symbols if s.strip()]
    if not symbols:
        raise ValueError("No symbols provided")
    if len(symbols) > 32:
        raise ValueError("Too many symbols (max 32)")

    original_prices: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        original_prices[sym] = load_price_data(sym, data_root)

    # Limit common window to where all symbols have at least some data.
    starts = [df.index.min() for df in original_prices.values()]
    ends = [df.index.max() for df in original_prices.values()]
    start = max(starts)
    end = min(ends)
    if start_date is not None:
        start = max(start, pd.to_datetime(start_date, utc=True))
    if end_date is not None:
        end = min(end, pd.to_datetime(end_date, utc=True))
    if start >= end:
        raise ValueError(f"Invalid date window: start={start} end={end}")

    # Calendar-day timeline (keeps crypto weekends vs stock closures explicit).
    full_index = pd.date_range(start.floor("D"), end.floor("D"), freq="D", tz="UTC")
    if len(full_index) < min_days:
        raise ValueError(f"Not enough days after alignment: {len(full_index)} (need {min_days})")

    # Build per-symbol aligned frames + tradable masks.
    aligned_prices: dict[str, pd.DataFrame] = {}
    aligned_feats: dict[str, pd.DataFrame] = {}
    tradable: dict[str, np.ndarray] = {}

    for sym, df in original_prices.items():
        # 1 when real bar exists on that day, else 0 (weekends/holidays/etc).
        mask = full_index.isin(df.index).astype(np.uint8)
        # Forward-fill prices across closed days; set volume=0 where closed.
        re = df.reindex(full_index, method="ffill")
        re["volume"] = re["volume"].where(mask.astype(bool), 0.0)
        re = re.bfill().fillna(0.0)
        aligned_prices[sym] = re
        aligned_feats[sym] = compute_daily_features(re)
        tradable[sym] = mask

    num_timesteps = len(full_index)
    num_symbols = len(symbols)

    feature_arr = np.zeros((num_timesteps, num_symbols, FEATURES_PER_SYM), dtype=np.float32)
    price_arr = np.zeros((num_timesteps, num_symbols, PRICE_FEATURES), dtype=np.float32)
    mask_arr = np.zeros((num_timesteps, num_symbols), dtype=np.uint8)

    for si, sym in enumerate(symbols):
        f = aligned_feats[sym].reindex(full_index).fillna(0.0)
        p = aligned_prices[sym].reindex(full_index).ffill().bfill().fillna(0.0)
        feature_arr[:, si, :] = f.values.astype(np.float32, copy=False)
        price_arr[:, si, :] = p[["open", "high", "low", "close", "volume"]].values.astype(np.float32, copy=False)
        mask_arr[:, si] = tradable[sym]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        header = struct.pack(
            "<4sIIIII40s",
            MAGIC,
            VERSION,
            num_symbols,
            num_timesteps,
            FEATURES_PER_SYM,
            PRICE_FEATURES,
            b"\x00" * 40,
        )
        f.write(header)
        for sym in symbols:
            raw = sym.encode("ascii", errors="ignore")[:15]
            f.write(raw + b"\x00" * (16 - len(raw)))
        f.write(feature_arr.tobytes(order="C"))
        f.write(price_arr.tobytes(order="C"))
        f.write(mask_arr.tobytes(order="C"))

    print(f"Wrote {output_path} ({num_symbols} symbols, {num_timesteps} days)")
    print(f"Date range: {full_index[0]} to {full_index[-1]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export daily data to MKTD binary for pufferlib_market")
    parser.add_argument("--symbols", required=True, help="Comma-separated symbol list (e.g. BTCUSD,ETHUSD,AAPL)")
    parser.add_argument("--data-root", default="trainingdata/train", help="Directory containing daily CSVs")
    parser.add_argument("--output", default="pufferlib_market/data/daily_market_data.bin", help="Output .bin path")
    parser.add_argument("--start-date", default=None, help="Optional ISO start date")
    parser.add_argument("--end-date", default=None, help="Optional ISO end date")
    parser.add_argument("--min-days", type=int, default=200, help="Minimum aligned calendar days required")
    args = parser.parse_args()

    try:
        export_binary(
            symbols=[s for s in args.symbols.split(",")],
            data_root=Path(args.data_root),
            output_path=Path(args.output),
            start_date=args.start_date,
            end_date=args.end_date,
            min_days=args.min_days,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
