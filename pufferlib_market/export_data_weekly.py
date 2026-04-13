"""
Export weekly OHLCV data to MKTD binary for the C trading environment.

Weekly bars are aggregated from daily CSVs (same source as export_data_daily.py).
Each week uses the last available trading day's close as the week close.

Feature layout (16 base + optional 4 Chronos2 7d forecast = 16 or 20):

  BASE WEEKLY FEATURES (indices 0-15):
    0  return_1w          weekly return (clipped ±0.5)
    1  return_4w          4-week return (clipped ±1.0)
    2  return_13w         13-week return (clipped ±2.0)
    3  return_52w         52-week return (clipped ±3.0)
    4  volatility_4w      rolling 4w std of weekly returns
    5  volatility_13w     rolling 13w std of weekly returns
    6  ma_delta_4w        (close - 4w MA) / close
    7  ma_delta_13w       (close - 13w MA) / close
    8  ma_delta_52w       (close - 52w MA) / close
    9  atr_pct_13w        13w ATR as % of close
   10  range_pct_1w       weekly (high - low) / close
   11  rsi_14w            RSI(14) on weekly bars, normalized to [-1, 1]
   12  trend_52w          52-week price change
   13  drawdown_13w       (close - 13w rolling max) / 13w rolling max
   14  drawdown_52w       (close - 52w rolling max) / 52w rolling max
   15  log_volume_z13w    log-volume z-score over 13w rolling window

  CHRONOS2 7-DAY FORECAST FEATURES (indices 16-19, requires --chronos-cache):
   16  weekly_close_delta  (predicted_close_7d_p50 - close) / close
   17  weekly_upside       (predicted_close_7d_p90 - close) / close
   18  weekly_downside     (predicted_close_7d_p10 - close) / close
   19  weekly_confidence   1 / (1 + (p90 - p10) / close)

  Forecast derived by compounding next-5-trading-day daily Chronos2 predictions
  from `strategytraining/forecast_cache/{SYMBOL}.parquet`. Falls back to zeros
  if cache unavailable.

Usage:
    # Base weekly (16 features):
    python -m pufferlib_market.export_data_weekly \\
        --symbols AAPL,MSFT,NVDA \\
        --output pufferlib_market/data/weekly_test.bin

    # With Chronos2 7d forecast features (20 features):
    python -m pufferlib_market.export_data_weekly \\
        --symbols AAPL,MSFT,NVDA \\
        --chronos-cache strategytraining/forecast_cache \\
        --output pufferlib_market/data/weekly_chronos_test.bin
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pufferlib_market.export_data_daily import load_price_data, MAGIC, PRICE_FEATURES

FEATURES_PER_SYM_BASE = 16
FEATURES_PER_SYM_CHRONOS = 20  # +4 forecast
VERSION = 2

BASE_FEATURE_NAMES = [
    "return_1w", "return_4w", "return_13w", "return_52w",
    "volatility_4w", "volatility_13w",
    "ma_delta_4w", "ma_delta_13w", "ma_delta_52w",
    "atr_pct_13w", "range_pct_1w",
    "rsi_14w",
    "trend_52w",
    "drawdown_13w", "drawdown_52w",
    "log_volume_z13w",
]
assert len(BASE_FEATURE_NAMES) == FEATURES_PER_SYM_BASE

CHRONOS_FEATURE_NAMES = [
    "weekly_close_delta",
    "weekly_upside",
    "weekly_downside",
    "weekly_confidence",
]


def _resample_weekly(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily OHLCV to weekly bars (week ending Friday)."""
    df = daily_df.copy()
    weekly = df.resample("W-FRI").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })
    weekly = weekly.dropna(how="all").ffill().bfill()
    volume_count = df["volume"].clip(lower=0.0).resample("W-FRI").count()
    weekly["_tradable"] = (volume_count > 0).astype(np.uint8).reindex(weekly.index, fill_value=0)
    return weekly


def compute_weekly_features(price_df: pd.DataFrame) -> pd.DataFrame:
    """Compute 16 weekly RL features from a weekly OHLCV DataFrame."""
    close = price_df["close"].astype(float)
    high = price_df["high"].astype(float)
    low = price_df["low"].astype(float)
    volume = price_df["volume"].astype(float)

    feat = pd.DataFrame(index=price_df.index)

    feat["return_1w"] = close.pct_change(1).fillna(0.0).clip(-0.5, 0.5)
    feat["return_4w"] = close.pct_change(4).fillna(0.0).clip(-1.0, 1.0)
    feat["return_13w"] = close.pct_change(13).fillna(0.0).clip(-2.0, 2.0)
    feat["return_52w"] = close.pct_change(52).fillna(0.0).clip(-3.0, 3.0)

    ret_1w = close.pct_change(1).fillna(0.0)
    feat["volatility_4w"] = ret_1w.rolling(4, min_periods=1).std(ddof=0).fillna(0.01).clip(0.0, 1.0)
    feat["volatility_13w"] = ret_1w.rolling(13, min_periods=1).std(ddof=0).fillna(0.01).clip(0.0, 1.0)

    ma4 = close.rolling(4, min_periods=1).mean()
    ma13 = close.rolling(13, min_periods=1).mean()
    ma52 = close.rolling(52, min_periods=1).mean()
    feat["ma_delta_4w"] = ((close - ma4) / ma4.clip(lower=1e-8)).fillna(0.0).clip(-0.5, 0.5)
    feat["ma_delta_13w"] = ((close - ma13) / ma13.clip(lower=1e-8)).fillna(0.0).clip(-0.5, 0.5)
    feat["ma_delta_52w"] = ((close - ma52) / ma52.clip(lower=1e-8)).fillna(0.0).clip(-0.5, 0.5)

    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    atr13 = tr.rolling(13, min_periods=1).mean()
    feat["atr_pct_13w"] = (atr13 / close.clip(lower=1e-8)).fillna(0.0).clip(0.0, 0.5)
    feat["range_pct_1w"] = ((high - low) / close.clip(lower=1e-8)).fillna(0.0).clip(0.0, 0.5)

    delta = close.diff()
    gain = delta.clip(lower=0.0).rolling(14, min_periods=1).mean()
    loss = (-delta.clip(upper=0.0)).rolling(14, min_periods=1).mean()
    rs = gain / loss.clip(lower=1e-8)
    feat["rsi_14w"] = (2.0 * (100.0 - 100.0 / (1.0 + rs)) / 100.0 - 1.0).fillna(0.0).clip(-1.0, 1.0)

    feat["trend_52w"] = close.pct_change(52).fillna(0.0).clip(-3.0, 3.0)

    roll_max_13 = close.rolling(13, min_periods=1).max()
    roll_max_52 = close.rolling(52, min_periods=1).max()
    feat["drawdown_13w"] = ((close - roll_max_13) / roll_max_13.clip(lower=1e-8)).fillna(0.0).clip(-1.0, 0.0)
    feat["drawdown_52w"] = ((close - roll_max_52) / roll_max_52.clip(lower=1e-8)).fillna(0.0).clip(-1.0, 0.0)

    log_vol = np.log1p(volume.clip(lower=0.0))
    log_vol_mean13 = log_vol.rolling(13, min_periods=1).mean()
    log_vol_std13 = log_vol.rolling(13, min_periods=1).std(ddof=0).replace(0.0, 1.0).clip(lower=1e-8)
    feat["log_volume_z13w"] = ((log_vol - log_vol_mean13) / log_vol_std13).fillna(0.0).clip(-5.0, 5.0)

    feat = feat[BASE_FEATURE_NAMES].fillna(0.0).astype(np.float32)
    assert feat.shape[1] == FEATURES_PER_SYM_BASE
    return feat


def _load_daily_forecast_cache(symbol: str, cache_root: Path) -> pd.DataFrame | None:
    """Load daily Chronos forecast parquet (same format as export_data_daily_v4.py)."""
    path = cache_root / f"{symbol.upper()}.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        df.columns = [str(c).strip().lower() for c in df.columns]

        if "date" in df.columns:
            idx = pd.to_datetime(df["date"], utc=True, errors="coerce")
        elif "timestamp" in df.columns:
            idx = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        elif isinstance(df.index, pd.DatetimeIndex):
            idx = df.index
            idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
        else:
            return None

        idx_dt = pd.to_datetime(idx, utc=True, errors="coerce")
        if isinstance(idx_dt, pd.Series):
            idx_dt = pd.DatetimeIndex(idx_dt)
        df.index = idx_dt.floor("D")
        df = df[~df.index.isna()].sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return df
    except Exception as exc:
        print(f"  WARNING: could not load forecast cache for {symbol}: {exc}", file=sys.stderr)
        return None


def compute_weekly_chronos_features(
    daily_forecast: pd.DataFrame | None,
    weekly_index: pd.DatetimeIndex,
    daily_close: pd.Series,
) -> pd.DataFrame:
    """
    Derive 4 weekly Chronos features from daily forecast cache.

    For each weekly bar, compounds the next 5 business days of daily Chronos
    p50/p10/p90 predictions to estimate end-of-next-week return distribution.
    """
    result = pd.DataFrame(
        {"weekly_close_delta": 0.0, "weekly_upside": 0.0, "weekly_downside": 0.0, "weekly_confidence": 0.5},
        index=weekly_index,
        dtype=np.float32,
    )

    if daily_forecast is None or daily_forecast.empty:
        return result

    p50_col = next((c for c in ("predicted_close", "predicted_close_p50") if c in daily_forecast.columns), None)
    p90_col = "predicted_close_p90" if "predicted_close_p90" in daily_forecast.columns else None
    p10_col = "predicted_close_p10" if "predicted_close_p10" in daily_forecast.columns else None

    if p50_col is None:
        return result

    fc_p50 = pd.to_numeric(daily_forecast[p50_col], errors="coerce").dropna()
    fc_p90 = pd.to_numeric(daily_forecast[p90_col], errors="coerce").dropna() if p90_col else None
    fc_p10 = pd.to_numeric(daily_forecast[p10_col], errors="coerce").dropna() if p10_col else None

    for week_end in weekly_index:
        if week_end not in daily_close.index:
            continue
        base_close = float(daily_close.loc[week_end])
        if base_close <= 0.0:
            continue

        next_bdays = pd.bdate_range(week_end + pd.Timedelta(days=1), periods=5, freq="B")
        next_bdays_utc = next_bdays.tz_localize("UTC") if next_bdays.tz is None else next_bdays.tz_convert("UTC")
        next_bdays_utc = next_bdays_utc.floor("D")

        p50_vals = fc_p50.reindex(next_bdays_utc).dropna()
        if p50_vals.empty:
            continue

        compound_p50 = float(p50_vals.iloc[-1]) / base_close - 1.0
        compound_p90 = None
        compound_p10 = None
        if fc_p90 is not None:
            p90_vals = fc_p90.reindex(next_bdays_utc).dropna()
            if not p90_vals.empty:
                compound_p90 = float(p90_vals.iloc[-1]) / base_close - 1.0
        if fc_p10 is not None:
            p10_vals = fc_p10.reindex(next_bdays_utc).dropna()
            if not p10_vals.empty:
                compound_p10 = float(p10_vals.iloc[-1]) / base_close - 1.0

        result.at[week_end, "weekly_close_delta"] = float(np.clip(compound_p50, -1.0, 1.0))
        if compound_p90 is not None:
            result.at[week_end, "weekly_upside"] = float(np.clip(compound_p90, -1.0, 1.0))
        if compound_p10 is not None:
            result.at[week_end, "weekly_downside"] = float(np.clip(compound_p10, -1.0, 1.0))
        if compound_p90 is not None and compound_p10 is not None:
            spread = abs(compound_p90 - compound_p10) * base_close
            conf = 1.0 / (1.0 + spread / max(base_close, 1e-8))
            result.at[week_end, "weekly_confidence"] = float(np.clip(conf, 0.0, 1.0))

    return result.astype(np.float32)


def export_binary(
    symbols: list[str],
    data_root: Path,
    output_path: Path,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    min_weeks: int = 52,
    chronos_cache: Path | None = None,
) -> None:
    """Write a weekly MKTD v2 binary for *symbols* covering the requested date range."""
    symbols = [s.strip().upper() for s in symbols if s.strip()]
    if not symbols:
        raise ValueError("No symbols provided")
    if len(symbols) > 128:
        raise ValueError("Too many symbols (max 128)")

    features_per_sym = FEATURES_PER_SYM_CHRONOS if chronos_cache is not None else FEATURES_PER_SYM_BASE

    daily_prices: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        daily_prices[sym] = load_price_data(sym, data_root)

    common_start = max(df.index.min() for df in daily_prices.values())
    common_end = min(df.index.max() for df in daily_prices.values())
    if start_date:
        common_start = max(common_start, pd.to_datetime(start_date, utc=True))
    if end_date:
        common_end = min(common_end, pd.to_datetime(end_date, utc=True))
    if common_start >= common_end:
        raise ValueError(f"Invalid date window: {common_start} to {common_end}")

    weekly_per_sym: dict[str, pd.DataFrame] = {}
    for sym, df in daily_prices.items():
        trimmed = df[(df.index >= common_start) & (df.index <= common_end)]
        weekly_per_sym[sym] = _resample_weekly(trimmed)

    all_indices = [set(w.index) for w in weekly_per_sym.values()]
    common_weeks = sorted(all_indices[0].intersection(*all_indices[1:]) if len(all_indices) > 1 else all_indices[0])
    common_index = pd.DatetimeIndex(common_weeks, tz="UTC")

    if len(common_index) < min_weeks:
        raise ValueError(f"Only {len(common_index)} weekly bars after alignment (need {min_weeks})")

    num_timesteps = len(common_index)
    num_symbols = len(symbols)
    feature_arr = np.zeros((num_timesteps, num_symbols, features_per_sym), dtype=np.float32)
    price_arr = np.zeros((num_timesteps, num_symbols, PRICE_FEATURES), dtype=np.float32)
    mask_arr = np.zeros((num_timesteps, num_symbols), dtype=np.uint8)

    for sym_idx, sym in enumerate(symbols):
        weekly = weekly_per_sym[sym].reindex(common_index, method="ffill").fillna(0.0)

        base_feats = compute_weekly_features(weekly)
        feature_arr[:, sym_idx, :FEATURES_PER_SYM_BASE] = base_feats.values

        if chronos_cache is not None:
            daily_fc = _load_daily_forecast_cache(sym, chronos_cache)
            if daily_fc is None:
                print(f"  WARNING: no daily forecast cache for {sym} — weekly forecast features = 0", file=sys.stderr)
            trimmed_daily = daily_prices[sym]
            daily_close = trimmed_daily["close"].copy()
            daily_close.index = daily_close.index.floor("D")
            chronos_feats = compute_weekly_chronos_features(daily_fc, common_index, daily_close)
            feature_arr[:, sym_idx, FEATURES_PER_SYM_BASE:features_per_sym] = chronos_feats.values

        price_arr[:, sym_idx, :] = weekly[["open", "high", "low", "close", "volume"]].values.astype(np.float32)
        if "_tradable" in weekly.columns:
            mask_arr[:, sym_idx] = weekly["_tradable"].values.astype(np.uint8)
        else:
            mask_arr[:, sym_idx] = 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as fh:
        header = struct.pack(
            "<4sIIIII40s",
            MAGIC, VERSION, num_symbols, num_timesteps, features_per_sym, PRICE_FEATURES,
            b"\x00" * 40,
        )
        fh.write(header)
        for sym in symbols:
            raw = sym.encode("ascii", errors="ignore")[:15]
            fh.write(raw + b"\x00" * (16 - len(raw)))
        fh.write(feature_arr.tobytes(order="C"))
        fh.write(price_arr.tobytes(order="C"))
        fh.write(mask_arr.tobytes(order="C"))

    label = "CHRONOS" if chronos_cache else "base"
    print(f"Wrote {output_path} ({num_symbols} symbols, {num_timesteps} weeks, features_per_sym={features_per_sym}, {label})")
    print(f"Date range: {common_index[0].date()} to {common_index[-1].date()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export weekly MKTD binary (16 base + optional 4 Chronos2 7d forecast)")
    parser.add_argument("--symbols", required=True)
    parser.add_argument("--data-root", default="trainingdata")
    parser.add_argument("--chronos-cache", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--output-train", default="pufferlib_market/data/weekly_train.bin")
    parser.add_argument("--output-val", default="pufferlib_market/data/weekly_val.bin")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--val-start-date", default=None)
    parser.add_argument("--min-weeks", type=int, default=52)
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    data_root = Path(args.data_root)
    chronos_cache = Path(args.chronos_cache) if args.chronos_cache else None

    if args.output:
        export_binary(symbols=symbols, data_root=data_root, output_path=Path(args.output),
                      start_date=args.start_date, end_date=args.end_date,
                      min_weeks=args.min_weeks, chronos_cache=chronos_cache)
        return

    if args.val_start_date:
        val_start = pd.to_datetime(args.val_start_date, utc=True)
        train_end = str((val_start - pd.Timedelta(days=7)).date())
    else:
        train_end = args.end_date
        val_start = None

    export_binary(symbols=symbols, data_root=data_root, output_path=Path(args.output_train),
                  start_date=args.start_date, end_date=train_end,
                  min_weeks=args.min_weeks, chronos_cache=chronos_cache)

    if val_start is not None:
        export_binary(symbols=symbols, data_root=data_root, output_path=Path(args.output_val),
                      start_date=str(val_start.date()), end_date=args.end_date,
                      min_weeks=26, chronos_cache=chronos_cache)


if __name__ == "__main__":
    main()
