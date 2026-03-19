"""
Export hourly OHLCV + forecast cache features to MKTD v2 binary for ``pufferlib_market``.

The output format matches the compiled C environment expectations:
  - Header (MKTD v2)
  - Symbol table (16-byte null-padded ASCII names)
  - Feature tensor: float32[T, S, 16]
  - Price tensor: float32[T, S, 5]   (open, high, low, close, volume)
  - Tradable mask: uint8[T, S]       (1=observed bar, 0=missing/non-tradable)

Feature layout (16 floats per symbol):
  0  chronos_close_delta_h1
  1  chronos_high_delta_h1
  2  chronos_low_delta_h1
  3  chronos_close_delta_h24
  4  chronos_high_delta_h24
  5  chronos_low_delta_h24
  6  forecast_confidence_h1
  7  forecast_confidence_h24
  8  return_1h
  9  return_24h
  10 volatility_24h
  11 ma_delta_24h
  12 ma_delta_72h
  13 atr_pct_24h
  14 trend_72h
  15 drawdown_72h
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

FEATURES_PER_SYM = 16
PRICE_FEATURES = 5
MAGIC = b"MKTD"
VERSION = 2


def _candidate_paths(symbol: str, data_root: Path) -> list[Path]:
    upper = symbol.upper()
    return [
        data_root / f"{upper}.csv",
        data_root / "crypto" / f"{upper}.csv",
        data_root / "stocks" / f"{upper}.csv",
    ]


def discover_symbols(data_root: Path) -> list[str]:
    root = Path(data_root)
    if not root.exists():
        return []
    found: set[str] = set()
    for folder in (root, root / "crypto", root / "stocks"):
        if not folder.exists():
            continue
        for path in folder.glob("*.csv"):
            found.add(path.stem.upper())
    return sorted(found)


def _read_hourly_prices(symbol: str, data_root: Path) -> pd.DataFrame:
    path = next((p for p in _candidate_paths(symbol, data_root) if p.exists()), None)
    if path is None:
        raise FileNotFoundError(f"No hourly CSV found for {symbol} under {data_root}")

    frame = pd.read_csv(path)
    frame.columns = [str(col).lower() for col in frame.columns]

    if "timestamp" in frame.columns:
        ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    elif "date" in frame.columns:
        ts = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    else:
        raise ValueError(f"{path} must contain 'timestamp' or 'date' column")

    frame["timestamp"] = ts.dt.floor("h")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    frame = frame.drop_duplicates(subset="timestamp", keep="last").set_index("timestamp")

    required = ("open", "high", "low", "close", "volume")
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    out = frame.loc[:, list(required)].astype(float)
    for col in ("open", "high", "low", "close"):
        out[col] = out[col].where(out[col] > 0.0, np.nan)
    out["volume"] = out["volume"].where(out["volume"] >= 0.0, 0.0)
    out = out.dropna(subset=["open", "high", "low", "close"])
    if out.empty:
        raise ValueError(f"{path} contains no valid OHLC rows after filtering")
    return out


def _read_forecast(symbol: str, cache_root: Path, horizon: int) -> pd.DataFrame:
    path = cache_root / f"h{int(horizon)}" / f"{symbol.upper()}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No forecast parquet for {symbol} horizon h{horizon} at {path}")

    frame = pd.read_parquet(path)
    frame.columns = [str(col).strip() for col in frame.columns]
    # Prefer `issued_at` for causal alignment: forecasts at `timestamp` are generated using
    # context ending at `issued_at` (typically `timestamp - 1h`). Using `issued_at` lets
    # us attach the prediction to the time when it becomes available.
    if "issued_at" in frame.columns:
        issued = pd.to_datetime(frame["issued_at"], utc=True, errors="coerce")
        frame = frame.assign(issued_at=issued).dropna(subset=["issued_at"])
        frame = frame.sort_values("issued_at").drop_duplicates(subset="issued_at", keep="last")
        frame = frame.set_index("issued_at")
    elif "timestamp" in frame.columns:
        ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame = frame.assign(timestamp=ts).dropna(subset=["timestamp"])
        frame = frame.sort_values("timestamp").drop_duplicates(subset="timestamp", keep="last")
        frame = frame.set_index("timestamp")
    elif isinstance(frame.index, pd.DatetimeIndex):
        idx = frame.index
        idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
        frame = frame.copy()
        frame.index = idx
        frame = frame.sort_index()
    else:
        raise ValueError(f"{path} must contain a 'timestamp' column or DatetimeIndex")

    frame.index = frame.index.floor("h")
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame


def _to_utc_timestamp(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _resolve_symbols(symbols: Iterable[str], data_root: Path) -> list[str]:
    explicit = [s.strip().upper() for s in symbols if s and s.strip()]
    if explicit:
        return explicit
    discovered = discover_symbols(data_root)
    if not discovered:
        raise ValueError(f"No symbols discovered under {data_root}")
    return discovered


def _forecast_delta(
    forecast: pd.DataFrame,
    index: pd.DatetimeIndex,
    col: str,
    close: pd.Series,
) -> pd.Series:
    if col not in forecast.columns:
        return pd.Series(0.0, index=index, dtype=np.float32)
    aligned = forecast[col].reindex(index)
    aligned = pd.to_numeric(aligned, errors="coerce")
    delta = (aligned - close) / close.clip(lower=1e-8)
    return delta.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)


def _forecast_confidence(
    forecast: pd.DataFrame,
    index: pd.DatetimeIndex,
    close: pd.Series,
) -> pd.Series:
    p90 = "predicted_close_p90"
    p10 = "predicted_close_p10"
    if p90 not in forecast.columns or p10 not in forecast.columns:
        return pd.Series(0.5, index=index, dtype=np.float32)
    spread = (forecast[p90] - forecast[p10]).reindex(index).abs()
    spread = pd.to_numeric(spread, errors="coerce")
    conf = 1.0 / (1.0 + spread / close.clip(lower=1e-8))
    return conf.replace([np.inf, -np.inf], np.nan).fillna(0.5).astype(np.float32)


def compute_features(price_df: pd.DataFrame, fc_h1: pd.DataFrame, fc_h24: pd.DataFrame) -> pd.DataFrame:
    close = price_df["close"].astype(float)
    high = price_df["high"].astype(float)
    low = price_df["low"].astype(float)

    feat = pd.DataFrame(index=price_df.index)

    # Forecast deltas (h1)
    feat["chronos_close_delta_h1"] = _forecast_delta(fc_h1, feat.index, "predicted_close_p50", close)
    feat["chronos_high_delta_h1"] = _forecast_delta(fc_h1, feat.index, "predicted_high_p50", close)
    feat["chronos_low_delta_h1"] = _forecast_delta(fc_h1, feat.index, "predicted_low_p50", close)

    # Forecast deltas (h24)
    feat["chronos_close_delta_h24"] = _forecast_delta(fc_h24, feat.index, "predicted_close_p50", close)
    feat["chronos_high_delta_h24"] = _forecast_delta(fc_h24, feat.index, "predicted_high_p50", close)
    feat["chronos_low_delta_h24"] = _forecast_delta(fc_h24, feat.index, "predicted_low_p50", close)

    # Forecast confidence (inverse spread)
    feat["forecast_confidence_h1"] = _forecast_confidence(fc_h1, feat.index, close)
    feat["forecast_confidence_h24"] = _forecast_confidence(fc_h24, feat.index, close)

    # Technical features
    ret_1h = close.pct_change(1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feat["return_1h"] = ret_1h.clip(-0.5, 0.5).astype(np.float32)
    feat["return_24h"] = (
        close.pct_change(24).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0).astype(np.float32)
    )
    feat["volatility_24h"] = (
        ret_1h.rolling(24, min_periods=1).std(ddof=0).fillna(0.0).clip(0.0, 1.0).astype(np.float32)
    )

    ma24 = close.rolling(24, min_periods=1).mean()
    ma72 = close.rolling(72, min_periods=1).mean()
    feat["ma_delta_24h"] = ((close - ma24) / ma24.clip(lower=1e-8)).fillna(0.0).clip(-0.5, 0.5).astype(np.float32)
    feat["ma_delta_72h"] = ((close - ma72) / ma72.clip(lower=1e-8)).fillna(0.0).clip(-0.5, 0.5).astype(np.float32)

    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    atr24 = tr.rolling(24, min_periods=1).mean()
    feat["atr_pct_24h"] = (atr24 / close.clip(lower=1e-8)).fillna(0.0).clip(0.0, 0.5).astype(np.float32)

    feat["trend_72h"] = close.pct_change(72).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0).astype(np.float32)

    roll_max = close.rolling(72, min_periods=1).max()
    feat["drawdown_72h"] = ((close - roll_max) / roll_max.clip(lower=1e-8)).fillna(0.0).clip(-1.0, 0.0).astype(np.float32)

    expected = [
        "chronos_close_delta_h1",
        "chronos_high_delta_h1",
        "chronos_low_delta_h1",
        "chronos_close_delta_h24",
        "chronos_high_delta_h24",
        "chronos_low_delta_h24",
        "forecast_confidence_h1",
        "forecast_confidence_h24",
        "return_1h",
        "return_24h",
        "volatility_24h",
        "ma_delta_24h",
        "ma_delta_72h",
        "atr_pct_24h",
        "trend_72h",
        "drawdown_72h",
    ]
    feat = feat[expected].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    if feat.shape[1] != FEATURES_PER_SYM:
        raise AssertionError("Feature-width mismatch for MKTD export")
    return feat


def export_binary(
    symbols: Iterable[str],
    data_root: Path,
    forecast_cache_root: Path,
    output_path: Path,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    feature_lag: int = 0,
    min_hours: int = 24 * 90,
    min_coverage: float = 0.95,
) -> dict[str, object]:
    root = Path(data_root)
    cache_root = Path(forecast_cache_root)
    symbol_list = _resolve_symbols(symbols, root)
    if len(symbol_list) > 32:
        raise ValueError(f"MKTD format supports max 32 symbols, got {len(symbol_list)}")

    raw_prices: dict[str, pd.DataFrame] = {}
    for sym in symbol_list:
        raw_prices[sym] = _read_hourly_prices(sym, root)

    overlap_start = max(df.index.min() for df in raw_prices.values())
    overlap_end = min(df.index.max() for df in raw_prices.values())

    req_start = _to_utc_timestamp(start_date)
    req_end = _to_utc_timestamp(end_date)
    if req_start is not None:
        overlap_start = max(overlap_start, req_start)
    if req_end is not None:
        overlap_end = min(overlap_end, req_end)

    overlap_start = overlap_start.floor("h")
    overlap_end = overlap_end.floor("h")
    if overlap_start >= overlap_end:
        raise ValueError(f"Invalid overlap window: start={overlap_start}, end={overlap_end}")

    index = pd.date_range(start=overlap_start, end=overlap_end, freq="h", tz="UTC")
    if len(index) < int(min_hours):
        raise ValueError(f"Only {len(index)} aligned hours; require at least {min_hours}")

    aligned_symbols: list[str] = []
    feature_list: list[np.ndarray] = []
    price_list: list[np.ndarray] = []
    mask_list: list[np.ndarray] = []

    for sym in symbol_list:
        frame = raw_prices[sym]
        observed_mask = index.isin(frame.index).astype(np.uint8, copy=False)
        coverage = float(observed_mask.mean()) if len(observed_mask) else 0.0
        if coverage < float(min_coverage):
            continue

        try:
            fc_h1 = _read_forecast(sym, cache_root, 1)
        except FileNotFoundError:
            fc_h1 = pd.DataFrame()
        try:
            fc_h24 = _read_forecast(sym, cache_root, 24)
        except FileNotFoundError:
            fc_h24 = pd.DataFrame()

        aligned = frame.reindex(index, method="ffill").bfill()
        aligned["volume"] = aligned["volume"].where(observed_mask.astype(bool), 0.0)
        aligned = aligned.fillna(0.0)

        feat = compute_features(aligned, fc_h1, fc_h24)

        aligned_symbols.append(sym)
        feature_list.append(feat.to_numpy(dtype=np.float32, copy=False))
        price_list.append(aligned[["open", "high", "low", "close", "volume"]].to_numpy(dtype=np.float32, copy=False))
        mask_list.append(observed_mask.astype(np.uint8, copy=False))

    if not aligned_symbols:
        raise ValueError(
            "No symbols met coverage threshold and forecast availability "
            f"min_coverage={min_coverage:.3f} over {len(index)} aligned hours"
        )

    num_symbols = len(aligned_symbols)
    num_timesteps = len(index)
    feature_arr = np.stack(feature_list, axis=1)  # [T, S, 16]
    price_arr = np.stack(price_list, axis=1)      # [T, S, 5]
    mask_arr = np.stack(mask_list, axis=1)        # [T, S]

    lag = int(feature_lag)
    if lag < 0:
        raise ValueError("feature_lag must be >= 0")
    if lag:
        if lag >= int(num_timesteps):
            raise ValueError(f"feature_lag={lag} must be < num_timesteps={num_timesteps}")
        # Lag features relative to prices/mask so actions cannot directly use the same-hour close-derived signals.
        shifted = np.zeros_like(feature_arr)
        shifted[lag:] = feature_arr[:-lag]
        feature_arr = shifted

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
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
        handle.write(header)
        for sym in aligned_symbols:
            raw = sym.encode("ascii", errors="ignore")[:15]
            handle.write(raw + b"\x00" * (16 - len(raw)))
        handle.write(feature_arr.tobytes(order="C"))
        handle.write(price_arr.tobytes(order="C"))
        handle.write(mask_arr.tobytes(order="C"))

    return {
        "path": str(output_path),
        "symbols": aligned_symbols,
        "num_symbols": num_symbols,
        "num_timesteps": num_timesteps,
        "feature_lag": int(lag),
        "start": str(index[0]),
        "end": str(index[-1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export hourly MKTD v2 binary with forecast-derived features")
    parser.add_argument(
        "--symbols",
        default="",
        help="Comma-separated symbol list. If empty, discover all CSV symbols under data-root.",
    )
    parser.add_argument("--data-root", default="trainingdatahourly", help="Root folder containing hourly CSV files")
    parser.add_argument("--forecast-cache-root", required=True, help="Forecast cache root containing h1/ and h24/ parquets")
    parser.add_argument("--output", default="pufferlib_market/data/hourly_forecast_mktd_v2.bin", help="Output .bin file")
    parser.add_argument("--start-date", default=None, help="Optional inclusive start timestamp/date (UTC)")
    parser.add_argument("--end-date", default=None, help="Optional inclusive end timestamp/date (UTC)")
    parser.add_argument(
        "--feature-lag",
        type=int,
        default=0,
        help="Shift feature tensor by N hours (features[t]=features[t-N], pad with zeros).",
    )
    parser.add_argument("--min-hours", type=int, default=24 * 90, help="Minimum aligned hours required")
    parser.add_argument("--min-coverage", type=float, default=0.95, help="Minimum per-symbol coverage ratio [0,1]")
    args = parser.parse_args()

    symbols = [s for s in args.symbols.split(",")] if args.symbols else []
    report = export_binary(
        symbols=symbols,
        data_root=Path(args.data_root),
        forecast_cache_root=Path(args.forecast_cache_root),
        output_path=Path(args.output),
        start_date=args.start_date,
        end_date=args.end_date,
        feature_lag=int(args.feature_lag),
        min_hours=max(2, int(args.min_hours)),
        min_coverage=min(max(float(args.min_coverage), 0.0), 1.0),
    )
    print(
        "Wrote {path} | symbols={num_symbols} | timesteps={num_timesteps} | "
        "start={start} | end={end}".format(**report)
    )
    print("Symbols:", ",".join(report["symbols"]))  # type: ignore[index]


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
