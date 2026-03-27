"""
Export daily OHLCV + hourly-context + Chronos forecast features (MKTD v4).

MKTD v4 extends the daily RL dataset so the policy still trades a daily plan,
but each day carries richer multi-frequency context:
  - 16 base daily technical features (same as MKTD v2)
  - 4 realised intraday features from hourly bars (same as MKTD v3)
  - 4 daily Chronos forecast features from a daily forecast cache
  - 4 aggregated hourly Chronos forecast-context features from hourly caches

The key idea is to summarise the *previous day's* intraday trajectory and hourly
forecast evolution into day-level features. The C env already lags observations by
one bar, so these features remain causal while letting the daily policy learn from
hourly context.

Feature layout (28 floats per symbol):
  0-15   base daily features
  16-19  intraday_vol, morning_ret, vwap_dev, gap_open
  20-23  daily_close_delta, daily_high_delta, daily_low_delta, daily_confidence
  24-27  hourly_close_delta_mean_h1, hourly_close_delta_slope_h1,
          hourly_close_delta_mean_h24, hourly_confidence_mean_h24
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pufferlib_market.export_data_daily import (
    FEATURES_PER_SYM as BASE_FEATURES,
    PRICE_FEATURES,
    MAGIC,
    load_price_data,
    compute_daily_features,
)
from pufferlib_market.export_data_daily_v3 import (
    INTRADAY_FEATURES,
    compute_intraday_features,
    load_hourly_data,
    zscore_normalise,
)
from pufferlib_market.export_data_hourly_forecast import _read_forecast as read_hourly_forecast

DAILY_FORECAST_FEATURES = 4
HOURLY_FORECAST_FEATURES = 4
FEATURES_PER_SYM_V4 = BASE_FEATURES + INTRADAY_FEATURES + DAILY_FORECAST_FEATURES + HOURLY_FORECAST_FEATURES
VERSION = 4

DAILY_FORECAST_COLUMNS = [
    "daily_close_delta",
    "daily_high_delta",
    "daily_low_delta",
    "daily_confidence",
]
HOURLY_CONTEXT_COLUMNS = [
    "hourly_close_delta_mean_h1",
    "hourly_close_delta_slope_h1",
    "hourly_close_delta_mean_h24",
    "hourly_confidence_mean_h24",
]
INTRADAY_COLUMNS = ["intraday_vol", "morning_ret", "vwap_dev", "gap_open"]


def load_daily_forecast(symbol: str, forecast_root: Path) -> pd.DataFrame | None:
    """Load a daily forecast parquet for *symbol* if available."""
    path = Path(forecast_root) / f"{symbol.upper()}.parquet"
    if not path.exists():
        return None

    frame = pd.read_parquet(path)
    frame.columns = [str(col).strip().lower() for col in frame.columns]

    if "date" in frame.columns:
        idx = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    elif "timestamp" in frame.columns:
        idx = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    elif isinstance(frame.index, pd.DatetimeIndex):
        idx = frame.index
        idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    else:
        return None

    out = frame.copy()
    idx = pd.to_datetime(idx, utc=True, errors="coerce")
    if isinstance(idx, pd.Series):
        idx = idx.dt.floor("D")
    else:
        idx = idx.floor("D")
    out.index = idx
    out = out[~out.index.isna()]
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def _relative_delta(series: pd.Series | None, base: pd.Series, *, clip: float = 1.0) -> pd.Series:
    if series is None:
        return pd.Series(0.0, index=base.index, dtype=np.float32)
    numeric = pd.to_numeric(series, errors="coerce")
    delta = (numeric - base) / base.clip(lower=1e-8)
    return delta.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-clip, clip).astype(np.float32)


def _confidence_from_spread(spread: pd.Series | None, base: pd.Series, *, default: float = 0.5) -> pd.Series:
    if spread is None:
        return pd.Series(float(default), index=base.index, dtype=np.float32)
    numeric = pd.to_numeric(spread, errors="coerce").abs()
    conf = 1.0 / (1.0 + numeric / base.clip(lower=1e-8))
    return conf.replace([np.inf, -np.inf], np.nan).fillna(float(default)).clip(0.0, 1.0).astype(np.float32)


def compute_daily_forecast_features(
    forecast_df: pd.DataFrame | None,
    daily_df: pd.DataFrame,
) -> pd.DataFrame:
    """Convert a daily forecast cache into 4 clipped day-level RL features."""
    index = daily_df.index
    close = daily_df["close"].astype(float)
    result = pd.DataFrame(0.0, index=index, columns=DAILY_FORECAST_COLUMNS, dtype=np.float32)
    if forecast_df is None or forecast_df.empty:
        return result

    aligned = forecast_df.reindex(index)
    result["daily_close_delta"] = _relative_delta(aligned.get("predicted_close"), close)
    result["daily_high_delta"] = _relative_delta(aligned.get("predicted_high"), close)
    result["daily_low_delta"] = _relative_delta(aligned.get("predicted_low"), close)

    spread: pd.Series | None = None
    p90 = aligned.get("predicted_close_p90")
    p10 = aligned.get("predicted_close_p10")
    if p90 is not None and p10 is not None:
        spread = pd.to_numeric(p90, errors="coerce") - pd.to_numeric(p10, errors="coerce")
    elif "forecast_volatility_pct" in aligned.columns:
        spread = pd.to_numeric(aligned["forecast_volatility_pct"], errors="coerce") * close
    result["daily_confidence"] = _confidence_from_spread(spread, close)
    return result.astype(np.float32)


def _safe_read_hourly_forecast(symbol: str, forecast_root: Path, horizon: int) -> pd.DataFrame | None:
    try:
        return read_hourly_forecast(symbol, Path(forecast_root), int(horizon))
    except (FileNotFoundError, ValueError):
        return None


def compute_hourly_forecast_context_features(
    forecast_h1: pd.DataFrame | None,
    forecast_h24: pd.DataFrame | None,
    daily_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate hourly forecast evolution into day-level context features.

    These summarize the *shape* of the forecast trajectory across the day rather
    than exposing raw hourly sequences to the daily RL policy.
    """
    result = pd.DataFrame(0.0, index=daily_df.index, columns=HOURLY_CONTEXT_COLUMNS, dtype=np.float32)
    if daily_df.empty:
        return result

    date_to_pos = {date_key: idx for idx, date_key in enumerate(daily_df.index.date)}
    close = daily_df["close"].astype(float).clip(lower=1e-8)

    if forecast_h1 is not None and not forecast_h1.empty and "predicted_close_p50" in forecast_h1.columns:
        for date_key, day_fc in forecast_h1.groupby(forecast_h1.index.date):
            pos = date_to_pos.get(date_key)
            if pos is None:
                continue
            base_close = float(close.iloc[pos])
            deltas = pd.to_numeric(day_fc["predicted_close_p50"], errors="coerce")
            deltas = ((deltas - base_close) / base_close).replace([np.inf, -np.inf], np.nan).dropna().clip(-1.0, 1.0)
            if deltas.empty:
                continue
            result.iat[pos, 0] = float(deltas.mean())
            result.iat[pos, 1] = float(deltas.iloc[-1] - deltas.iloc[0]) if len(deltas) >= 2 else 0.0

    if forecast_h24 is not None and not forecast_h24.empty and "predicted_close_p50" in forecast_h24.columns:
        for date_key, day_fc in forecast_h24.groupby(forecast_h24.index.date):
            pos = date_to_pos.get(date_key)
            if pos is None:
                continue
            base_close = float(close.iloc[pos])
            deltas = pd.to_numeric(day_fc["predicted_close_p50"], errors="coerce")
            deltas = ((deltas - base_close) / base_close).replace([np.inf, -np.inf], np.nan).dropna().clip(-1.0, 1.0)
            if not deltas.empty:
                result.iat[pos, 2] = float(deltas.mean())

            spread = None
            if "predicted_close_p90" in day_fc.columns and "predicted_close_p10" in day_fc.columns:
                spread = (
                    pd.to_numeric(day_fc["predicted_close_p90"], errors="coerce")
                    - pd.to_numeric(day_fc["predicted_close_p10"], errors="coerce")
                ).abs()
            if spread is not None:
                conf = (1.0 / (1.0 + spread / base_close)).replace([np.inf, -np.inf], np.nan).dropna().clip(0.0, 1.0)
                if not conf.empty:
                    result.iat[pos, 3] = float(conf.mean())

    return result.astype(np.float32)


def export_binary(
    symbols: list[str],
    data_root: Path,
    hourly_root: Path,
    daily_forecast_root: Path,
    hourly_forecast_root: Path,
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

    hourly_price_data: dict[str, pd.DataFrame | None] = {}
    daily_forecasts: dict[str, pd.DataFrame | None] = {}
    hourly_forecasts_h1: dict[str, pd.DataFrame | None] = {}
    hourly_forecasts_h24: dict[str, pd.DataFrame | None] = {}

    for sym in symbols:
        hourly_price_data[sym] = load_hourly_data(sym, hourly_root)
        if hourly_price_data[sym] is None:
            print(f"  WARNING: no hourly prices for {sym} — realised intraday features will be 0", file=sys.stderr)

        daily_forecasts[sym] = load_daily_forecast(sym, daily_forecast_root)
        if daily_forecasts[sym] is None:
            print(f"  WARNING: no daily forecast cache for {sym} — daily forecast features will be 0", file=sys.stderr)

        hourly_forecasts_h1[sym] = _safe_read_hourly_forecast(sym, hourly_forecast_root, 1)
        hourly_forecasts_h24[sym] = _safe_read_hourly_forecast(sym, hourly_forecast_root, 24)
        if hourly_forecasts_h1[sym] is None and hourly_forecasts_h24[sym] is None:
            print(f"  WARNING: no hourly forecast cache for {sym} — hourly context features will be 0", file=sys.stderr)

    aligned_prices: dict[str, pd.DataFrame] = {}
    aligned_feats_base: dict[str, pd.DataFrame] = {}
    aligned_feats_intraday: dict[str, np.ndarray] = {}
    aligned_feats_daily_fc: dict[str, pd.DataFrame] = {}
    aligned_feats_hourly_fc: dict[str, pd.DataFrame] = {}
    tradable: dict[str, np.ndarray] = {}

    for sym, df in original_prices.items():
        mask = full_index.isin(df.index).astype(np.uint8)
        aligned = df.reindex(full_index, method="ffill")
        aligned["volume"] = aligned["volume"].where(mask.astype(bool), 0.0)
        aligned = aligned.bfill().fillna(0.0)

        aligned_prices[sym] = aligned
        aligned_feats_base[sym] = compute_daily_features(aligned)
        tradable[sym] = mask

        intra_raw = compute_intraday_features(hourly_price_data[sym], aligned)
        intra_norm = np.zeros((len(full_index), INTRADAY_FEATURES), dtype=np.float32)
        for feature_idx, column in enumerate(INTRADAY_COLUMNS):
            intra_norm[:, feature_idx] = zscore_normalise(
                intra_raw[column].values.astype(np.float32),
                window=zscore_window,
            )
        aligned_feats_intraday[sym] = intra_norm
        aligned_feats_daily_fc[sym] = compute_daily_forecast_features(daily_forecasts[sym], aligned)
        aligned_feats_hourly_fc[sym] = compute_hourly_forecast_context_features(
            hourly_forecasts_h1[sym],
            hourly_forecasts_h24[sym],
            aligned,
        )

    num_timesteps = len(full_index)
    num_symbols = len(symbols)

    feature_arr = np.zeros((num_timesteps, num_symbols, FEATURES_PER_SYM_V4), dtype=np.float32)
    price_arr = np.zeros((num_timesteps, num_symbols, PRICE_FEATURES), dtype=np.float32)
    mask_arr = np.zeros((num_timesteps, num_symbols), dtype=np.uint8)

    for symbol_idx, sym in enumerate(symbols):
        offset = 0
        base = aligned_feats_base[sym].values.astype(np.float32, copy=False)
        feature_arr[:, symbol_idx, offset:offset + BASE_FEATURES] = base
        offset += BASE_FEATURES
        feature_arr[:, symbol_idx, offset:offset + INTRADAY_FEATURES] = aligned_feats_intraday[sym]
        offset += INTRADAY_FEATURES
        feature_arr[:, symbol_idx, offset:offset + DAILY_FORECAST_FEATURES] = aligned_feats_daily_fc[sym].values.astype(np.float32, copy=False)
        offset += DAILY_FORECAST_FEATURES
        feature_arr[:, symbol_idx, offset:offset + HOURLY_FORECAST_FEATURES] = aligned_feats_hourly_fc[sym].values.astype(np.float32, copy=False)

        price_arr[:, symbol_idx, :] = aligned_prices[sym][["open", "high", "low", "close", "volume"]].values.astype(np.float32, copy=False)
        mask_arr[:, symbol_idx] = tradable[sym]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as handle:
        header = struct.pack(
            "<4sIIIII40s",
            MAGIC,
            VERSION,
            num_symbols,
            num_timesteps,
            FEATURES_PER_SYM_V4,
            PRICE_FEATURES,
            b"\x00" * 40,
        )
        handle.write(header)
        for sym in symbols:
            raw = sym.encode("ascii", errors="ignore")[:15]
            handle.write(raw + b"\x00" * (16 - len(raw)))
        handle.write(feature_arr.tobytes(order="C"))
        handle.write(price_arr.tobytes(order="C"))
        handle.write(mask_arr.tobytes(order="C"))

    print(
        f"Wrote {output_path} ({num_symbols} symbols, {num_timesteps} days, "
        f"features_per_sym={FEATURES_PER_SYM_V4})"
    )
    print(f"Date range: {full_index[0].date()} to {full_index[-1].date()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export daily MKTD v4 binary with daily+hourly fused features"
    )
    parser.add_argument("--symbols", required=True, help="Comma-separated symbol list")
    parser.add_argument("--data-root", default="trainingdata", help="Directory containing daily CSVs")
    parser.add_argument(
        "--hourly-root",
        default="trainingdatahourly/stocks",
        help="Directory containing hourly CSVs",
    )
    parser.add_argument(
        "--daily-forecast-root",
        default="strategytraining/forecast_cache",
        help="Directory containing daily Chronos forecast parquets",
    )
    parser.add_argument(
        "--hourly-forecast-root",
        default="binanceneural/forecast_cache",
        help="Directory containing hourly Chronos forecast parquets",
    )
    parser.add_argument(
        "--output-train",
        default="pufferlib_market/data/stocks12_daily_v4_fused_train.bin",
        help="Output path for train split",
    )
    parser.add_argument(
        "--output-val",
        default="pufferlib_market/data/stocks12_daily_v4_fused_val.bin",
        help="Output path for val split",
    )
    parser.add_argument("--start-date", default=None, help="Optional ISO start date for train split")
    parser.add_argument("--end-date", default=None, help="Optional ISO end date for train split")
    parser.add_argument("--val-days", type=int, default=None, help="Hold out last N calendar days as validation split")
    parser.add_argument("--val-start-date", default=None, help="ISO start date for val split")
    parser.add_argument("--min-days", type=int, default=200, help="Minimum days required")
    parser.add_argument(
        "--zscore-window",
        type=int,
        default=60,
        help="Rolling window (days) for z-score normalisation of intraday features",
    )
    parser.add_argument(
        "--single-output",
        default=None,
        help="If set, write one file for the full date range (ignores train/val split)",
    )
    args = parser.parse_args()

    symbols = [part.strip().upper() for part in str(args.symbols).split(",") if part.strip()]
    data_root = Path(args.data_root)
    hourly_root = Path(args.hourly_root)
    daily_forecast_root = Path(args.daily_forecast_root)
    hourly_forecast_root = Path(args.hourly_forecast_root)

    if args.single_output:
        export_binary(
            symbols=symbols,
            data_root=data_root,
            hourly_root=hourly_root,
            daily_forecast_root=daily_forecast_root,
            hourly_forecast_root=hourly_forecast_root,
            output_path=Path(args.single_output),
            start_date=args.start_date,
            end_date=args.end_date,
            min_days=args.min_days,
            zscore_window=args.zscore_window,
        )
        return

    all_price_dfs = {symbol: load_price_data(symbol, data_root) for symbol in symbols}
    overall_start = max(df.index.min() for df in all_price_dfs.values())
    overall_end = min(df.index.max() for df in all_price_dfs.values())

    if args.start_date:
        overall_start = max(overall_start, pd.to_datetime(args.start_date, utc=True))

    if args.val_days is not None:
        full_index = pd.date_range(overall_start.floor("D"), overall_end.floor("D"), freq="D", tz="UTC")
        if len(full_index) < args.val_days + args.min_days:
            raise ValueError(
                f"Not enough days ({len(full_index)}) for val_days={args.val_days} + min_days={args.min_days}"
            )
        val_start_idx = len(full_index) - args.val_days
        val_start = full_index[val_start_idx]
        train_end = full_index[val_start_idx - 1]
    elif args.val_start_date:
        val_start = pd.to_datetime(args.val_start_date, utc=True)
        train_end = val_start - pd.Timedelta(days=1)
    else:
        full_index = pd.date_range(overall_start.floor("D"), overall_end.floor("D"), freq="D", tz="UTC")
        split_idx = int(len(full_index) * 0.85)
        train_end = full_index[split_idx - 1]
        val_start = full_index[split_idx]

    print(f"Train: {overall_start.date()} → {train_end.date()}")
    print(f"Val:   {val_start.date()} → {overall_end.date()}")

    export_binary(
        symbols=symbols,
        data_root=data_root,
        hourly_root=hourly_root,
        daily_forecast_root=daily_forecast_root,
        hourly_forecast_root=hourly_forecast_root,
        output_path=Path(args.output_train),
        start_date=str(overall_start.date()),
        end_date=str(train_end.date()),
        min_days=args.min_days,
        zscore_window=args.zscore_window,
    )
    export_binary(
        symbols=symbols,
        data_root=data_root,
        hourly_root=hourly_root,
        daily_forecast_root=daily_forecast_root,
        hourly_forecast_root=hourly_forecast_root,
        output_path=Path(args.output_val),
        start_date=str(val_start.date()),
        end_date=str(overall_end.date()),
        min_days=50,
        zscore_window=args.zscore_window,
    )


if __name__ == "__main__":
    main()
