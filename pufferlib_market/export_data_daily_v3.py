"""
Export daily OHLCV + intraday features (MKTD v3) to a flat binary for the C trading env.

MKTD v3 extends v2 with 4 extra intraday features per symbol computed from hourly bars:
  Feature 17: intraday_vol   — std of hourly log-returns within the day
  Feature 18: morning_ret    — return from open to ~noon (first 4 trading hours)
  Feature 19: vwap_dev       — close / VWAP - 1 (deviation from VWAP, using hourly bars)
  Feature 20: gap_open       — today_open / yesterday_close - 1 (overnight gap)

All 4 intraday features are z-score normalised per-symbol with a rolling 60-day window.
If hourly data is unavailable for a symbol/date the intraday features default to 0.0.

Binary format (v3):
  Header (64 bytes):
    magic:            4 bytes  "MKTD"
    version:          uint32   (3)
    num_symbols:      uint32
    num_timesteps:    uint32   (calendar days)
    features_per_sym: uint32   (20)
    price_features:   uint32   (5)
    padding:          40 bytes

  Symbol table: num_symbols * 16 bytes (null-padded ASCII names)

  Feature data: float32[num_timesteps][num_symbols][20]
    Features 0-15: same as MKTD v2 (price-only daily features)
    Features 16-19: intraday_vol, morning_ret, vwap_dev, gap_open

  Price data:   float32[num_timesteps][num_symbols][5]  (open, high, low, close, volume)

  Tradable mask: uint8[num_timesteps][num_symbols]
    1 when the asset has an actual bar for that calendar day; 0 otherwise.

Backwards compatibility:
  v1/v2 readers will be unaffected (they read their own data files).
  The C env reads features_per_sym from the header at runtime and adapts.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse v2 daily feature computation and utilities
from pufferlib_market.export_data_daily import (
    FEATURES_PER_SYM as BASE_FEATURES,
    PRICE_FEATURES,
    MAGIC,
    load_price_data,
    compute_daily_features,
    _normalise_daily_index,
    _coalesce_duplicate_columns,
)

INTRADAY_FEATURES = 4
FEATURES_PER_SYM_V3 = BASE_FEATURES + INTRADAY_FEATURES  # 20
VERSION = 3


def load_hourly_data(symbol: str, hourly_root: Path) -> pd.DataFrame | None:
    """Load hourly OHLCV CSV for a symbol. Returns None if not found."""
    symbol = symbol.upper()
    candidates = [
        hourly_root / f"{symbol}.csv",
        hourly_root / "stocks" / f"{symbol}.csv",
        hourly_root / "crypto" / f"{symbol}.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return None

    df = pd.read_csv(path)
    df.columns = [str(c).lower() for c in df.columns]
    df = _coalesce_duplicate_columns(df)

    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    elif "date" in df.columns:
        ts = pd.to_datetime(df["date"], utc=True, errors="coerce")
    else:
        return None

    df.index = ts
    df = df.dropna(subset=["close"])
    df = df.sort_index()

    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return None

    return df[required].astype(float)


def compute_intraday_features(
    hourly_df: pd.DataFrame | None,
    daily_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each daily bar, compute 4 intraday features from hourly bars of that day.

    Returns DataFrame with same index as daily_df, columns:
      intraday_vol, morning_ret, vwap_dev, gap_open
    All values are raw (before z-score normalisation).
    Missing intraday data → 0.0.
    """
    n = len(daily_df)
    intraday_vol = np.zeros(n, dtype=np.float32)
    morning_ret = np.zeros(n, dtype=np.float32)
    vwap_dev = np.zeros(n, dtype=np.float32)

    # gap_open: vectorised from daily prices (no hourly data needed)
    close_shifted = daily_df["close"].shift(1)
    gap_open_series = (daily_df["open"] / close_shifted.clip(lower=1e-8) - 1.0).fillna(0.0)
    gap_open = gap_open_series.values.astype(np.float32)

    if hourly_df is None or hourly_df.empty:
        return pd.DataFrame(
            {"intraday_vol": intraday_vol, "morning_ret": morning_ret,
             "vwap_dev": vwap_dev, "gap_open": gap_open},
            index=daily_df.index,
        )

    # Group hourly bars by calendar date using pandas groupby (avoids iterrows)
    date_keys = hourly_df.index.date
    date_to_idx = {d: i for i, d in enumerate(daily_df.index.date)}

    for date_key, day_hourly in hourly_df.groupby(date_keys):
        i = date_to_idx.get(date_key)
        if i is None or len(day_hourly) < 2:
            continue

        closes = day_hourly["close"].values.astype(float)
        volumes = day_hourly["volume"].values.astype(float)
        opens = day_hourly["open"].values.astype(float)

        # intraday_vol: std of hourly log-returns
        log_rets = np.diff(np.log(np.maximum(closes, 1e-12)))
        intraday_vol[i] = float(np.std(log_rets))

        # morning_ret: first 4 hours return (open of first bar → close of 4th bar)
        n_morning = min(4, len(closes))
        if opens[0] > 1e-8:
            morning_ret[i] = float(closes[n_morning - 1] / opens[0] - 1.0)

        # vwap_dev: daily_close / VWAP - 1
        total_vol = float(volumes.sum())
        vwap = float((closes * volumes).sum() / total_vol) if total_vol > 1e-8 else float(closes.mean())
        daily_close = float(daily_df["close"].iloc[i])
        if vwap > 1e-8:
            vwap_dev[i] = float(daily_close / vwap - 1.0)

    return pd.DataFrame(
        {"intraday_vol": intraday_vol, "morning_ret": morning_ret,
         "vwap_dev": vwap_dev, "gap_open": gap_open},
        index=daily_df.index,
    )


def zscore_normalise(
    arr: np.ndarray,
    window: int = 60,
    clip: float = 5.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Rolling z-score normalisation along axis 0 (time).
    For each timestep t: z = (arr[t] - mean(arr[t-window:t])) / std(arr[t-window:t])
    Window is exclusive of the current bar (look-back only). Clipped to [-clip, clip].
    """
    s = pd.Series(arr.astype(np.float64))
    # shift(1) so current bar is excluded from its own normalisation window
    roll = s.shift(1).rolling(window=window, min_periods=2)
    mu = roll.mean()
    sigma = roll.std(ddof=0)
    z = (s - mu) / sigma.clip(lower=eps)
    return z.fillna(0.0).clip(-clip, clip).values.astype(np.float32)


def export_binary(
    symbols: list[str],
    data_root: Path,
    hourly_root: Path,
    output_path: Path,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    min_days: int = 200,
    zscore_window: int = 60,
) -> None:
    symbols = [s.strip().upper() for s in symbols if s.strip()]
    if not symbols:
        raise ValueError("No symbols provided")
    if len(symbols) > 64:
        raise ValueError("Too many symbols (max 64)")

    original_prices: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        original_prices[sym] = load_price_data(sym, data_root)

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

    full_index = pd.date_range(start.floor("D"), end.floor("D"), freq="D", tz="UTC")
    if len(full_index) < min_days:
        raise ValueError(f"Not enough days after alignment: {len(full_index)} (need {min_days})")

    # Load hourly data for all symbols (None if unavailable)
    hourly_data: dict[str, pd.DataFrame | None] = {}
    for sym in symbols:
        hdf = load_hourly_data(sym, hourly_root)
        hourly_data[sym] = hdf
        if hdf is None:
            print(f"  WARNING: no hourly data for {sym} — intraday features will be 0", file=sys.stderr)
        else:
            print(f"  Hourly data loaded for {sym}: {len(hdf)} bars")

    aligned_prices: dict[str, pd.DataFrame] = {}
    aligned_feats_base: dict[str, pd.DataFrame] = {}
    aligned_feats_intra: dict[str, np.ndarray] = {}  # [T, 4]
    tradable: dict[str, np.ndarray] = {}

    for sym, df in original_prices.items():
        mask = full_index.isin(df.index).astype(np.uint8)
        re = df.reindex(full_index, method="ffill")
        re["volume"] = re["volume"].where(mask.astype(bool), 0.0)
        re = re.bfill().fillna(0.0)
        aligned_prices[sym] = re
        aligned_feats_base[sym] = compute_daily_features(re)
        tradable[sym] = mask

        # Compute raw intraday features on the aligned (forward-filled) daily frame
        intra_raw = compute_intraday_features(hourly_data[sym], re)

        # Z-score normalise each intraday feature with rolling window
        intra_norm = np.zeros((len(full_index), INTRADAY_FEATURES), dtype=np.float32)
        for fi, col in enumerate(["intraday_vol", "morning_ret", "vwap_dev", "gap_open"]):
            raw_vals = intra_raw[col].values.astype(np.float32)
            intra_norm[:, fi] = zscore_normalise(raw_vals, window=zscore_window)

        aligned_feats_intra[sym] = intra_norm

    num_timesteps = len(full_index)
    num_symbols = len(symbols)

    feature_arr = np.zeros((num_timesteps, num_symbols, FEATURES_PER_SYM_V3), dtype=np.float32)
    price_arr = np.zeros((num_timesteps, num_symbols, PRICE_FEATURES), dtype=np.float32)
    mask_arr = np.zeros((num_timesteps, num_symbols), dtype=np.uint8)

    for si, sym in enumerate(symbols):
        feature_arr[:, si, :BASE_FEATURES] = aligned_feats_base[sym].values.astype(np.float32, copy=False)
        feature_arr[:, si, BASE_FEATURES:] = aligned_feats_intra[sym]
        price_arr[:, si, :] = aligned_prices[sym][["open", "high", "low", "close", "volume"]].values.astype(np.float32, copy=False)
        mask_arr[:, si] = tradable[sym]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        header = struct.pack(
            "<4sIIIII40s",
            MAGIC,
            VERSION,
            num_symbols,
            num_timesteps,
            FEATURES_PER_SYM_V3,
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

    print(
        f"Wrote {output_path} ({num_symbols} symbols, {num_timesteps} days, "
        f"features_per_sym={FEATURES_PER_SYM_V3})"
    )
    print(f"Date range: {full_index[0].date()} to {full_index[-1].date()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export daily MKTD v3 binary with 20 features (16 daily + 4 intraday)"
    )
    parser.add_argument(
        "--symbols",
        required=True,
        help="Comma-separated symbol list (e.g. AAPL,MSFT,NVDA)",
    )
    parser.add_argument(
        "--data-root",
        default="trainingdata",
        help="Directory containing daily CSVs (default: trainingdata)",
    )
    parser.add_argument(
        "--hourly-root",
        default="trainingdatahourly/stocks",
        help="Directory containing hourly CSVs (default: trainingdatahourly/stocks)",
    )
    parser.add_argument(
        "--output-train",
        default="pufferlib_market/data/stocks12_daily_v3_train.bin",
        help="Output path for train split",
    )
    parser.add_argument(
        "--output-val",
        default="pufferlib_market/data/stocks12_daily_v3_val.bin",
        help="Output path for val split",
    )
    parser.add_argument("--start-date", default=None, help="Optional ISO start date for train split")
    parser.add_argument("--end-date", default=None, help="Optional ISO end date for train split")
    parser.add_argument(
        "--val-days",
        type=int,
        default=None,
        help="Hold out last N calendar days as validation split (overrides --val-start-date)",
    )
    parser.add_argument("--val-start-date", default=None, help="ISO start date for val split")
    parser.add_argument("--min-days", type=int, default=200, help="Minimum days required (default: 200)")
    parser.add_argument(
        "--zscore-window",
        type=int,
        default=60,
        help="Rolling window (days) for z-score normalisation of intraday features (default: 60)",
    )
    parser.add_argument(
        "--single-output",
        default=None,
        help="If set, write one file for the full date range (ignores train/val split)",
    )
    args = parser.parse_args()

    syms = [s for s in args.symbols.split(",")]
    data_root = Path(args.data_root)
    hourly_root = Path(args.hourly_root)

    if args.single_output:
        export_binary(
            symbols=syms,
            data_root=data_root,
            hourly_root=hourly_root,
            output_path=Path(args.single_output),
            start_date=args.start_date,
            end_date=args.end_date,
            min_days=args.min_days,
            zscore_window=args.zscore_window,
        )
        return

    # Determine split: load all price data once to compute the common date range
    all_price_dfs = {s: load_price_data(s, data_root) for s in syms}
    overall_start = max(df.index.min() for df in all_price_dfs.values())
    overall_end = min(df.index.max() for df in all_price_dfs.values())

    if args.start_date:
        overall_start = max(overall_start, pd.to_datetime(args.start_date, utc=True))

    if args.val_days is not None:
        # Split by day count from the end
        full_idx = pd.date_range(overall_start.floor("D"), overall_end.floor("D"), freq="D", tz="UTC")
        if len(full_idx) < args.val_days + args.min_days:
            raise ValueError(
                f"Not enough days ({len(full_idx)}) for val_days={args.val_days} + min_days={args.min_days}"
            )
        val_start_idx = len(full_idx) - args.val_days
        val_start = full_idx[val_start_idx]
        train_end = full_idx[val_start_idx - 1]
    elif args.val_start_date:
        val_start = pd.to_datetime(args.val_start_date, utc=True)
        train_end = val_start - pd.Timedelta(days=1)
    else:
        # Default: 85/15 split by total days
        full_idx = pd.date_range(overall_start.floor("D"), overall_end.floor("D"), freq="D", tz="UTC")
        split_idx = int(len(full_idx) * 0.85)
        train_end = full_idx[split_idx - 1]
        val_start = full_idx[split_idx]

    print(f"Train: {overall_start.date()} → {train_end.date()}")
    print(f"Val:   {val_start.date()} → {overall_end.date()}")

    export_binary(
        symbols=syms,
        data_root=data_root,
        hourly_root=hourly_root,
        output_path=Path(args.output_train),
        start_date=str(overall_start.date()),
        end_date=str(train_end.date()),
        min_days=args.min_days,
        zscore_window=args.zscore_window,
    )
    export_binary(
        symbols=syms,
        data_root=data_root,
        hourly_root=hourly_root,
        output_path=Path(args.output_val),
        start_date=str(val_start.date()),
        end_date=str(overall_end.date()),
        min_days=50,  # val splits can be short
        zscore_window=args.zscore_window,
    )


if __name__ == "__main__":
    main()
