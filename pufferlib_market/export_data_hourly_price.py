"""
Export hourly OHLCV crypto data to MKTD v2 binary for ``pufferlib_market``.

The output format matches ``pufferlib_market`` C environment expectations:
  - Header (MKTD v2)
  - Symbol table
  - Feature tensor: float32[T, S, 16]
  - Price tensor: float32[T, S, 5]   (open, high, low, close, volume)
  - Tradable mask: uint8[T, S]
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
    # Guard rails against pathological rows.
    for col in ("open", "high", "low", "close"):
        out[col] = out[col].where(out[col] > 0.0, np.nan)
    out["volume"] = out["volume"].where(out["volume"] >= 0.0, 0.0)
    out = out.dropna(subset=["open", "high", "low", "close"])
    if out.empty:
        raise ValueError(f"{path} contains no valid OHLC rows after filtering")
    return out


def compute_hourly_features(price_df: pd.DataFrame) -> pd.DataFrame:
    close = price_df["close"].astype(float)
    high = price_df["high"].astype(float)
    low = price_df["low"].astype(float)
    volume = price_df["volume"].astype(float)

    feat = pd.DataFrame(index=price_df.index)

    ret_1h = close.pct_change(1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feat["return_1h"] = ret_1h.clip(-0.5, 0.5)
    feat["return_6h"] = close.pct_change(6).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0)
    feat["return_24h"] = close.pct_change(24).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-2.0, 2.0)
    feat["return_72h"] = close.pct_change(72).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-3.0, 3.0)

    feat["volatility_6h"] = ret_1h.rolling(6, min_periods=1).std(ddof=0).fillna(0.0).clip(0.0, 1.0)
    feat["volatility_24h"] = ret_1h.rolling(24, min_periods=1).std(ddof=0).fillna(0.0).clip(0.0, 1.0)
    feat["volatility_72h"] = ret_1h.rolling(72, min_periods=1).std(ddof=0).fillna(0.0).clip(0.0, 1.0)

    ma6 = close.rolling(6, min_periods=1).mean()
    ma24 = close.rolling(24, min_periods=1).mean()
    ma72 = close.rolling(72, min_periods=1).mean()
    feat["ma_delta_6h"] = ((close - ma6) / ma6.clip(lower=1e-8)).fillna(0.0).clip(-0.5, 0.5)
    feat["ma_delta_24h"] = ((close - ma24) / ma24.clip(lower=1e-8)).fillna(0.0).clip(-0.75, 0.75)
    feat["ma_delta_72h"] = ((close - ma72) / ma72.clip(lower=1e-8)).fillna(0.0).clip(-1.0, 1.0)

    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    atr24 = tr.rolling(24, min_periods=1).mean()
    feat["atr_pct_24h"] = (atr24 / close.clip(lower=1e-8)).fillna(0.0).clip(0.0, 0.75)
    feat["range_pct_1h"] = ((high - low) / close.clip(lower=1e-8)).fillna(0.0).clip(0.0, 0.75)

    feat["trend_24h"] = close.pct_change(24).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-2.0, 2.0)
    feat["trend_72h"] = close.pct_change(72).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-3.0, 3.0)

    roll_max = close.rolling(72, min_periods=1).max()
    feat["drawdown_72h"] = ((close - roll_max) / roll_max.clip(lower=1e-8)).fillna(0.0).clip(-1.0, 0.0)

    log_vol = np.log1p(volume.clip(lower=0.0))
    vol_mean = log_vol.rolling(24, min_periods=1).mean()
    vol_std = log_vol.rolling(24, min_periods=1).std(ddof=0).replace(0.0, 1.0)
    feat["log_volume_z24h"] = ((log_vol - vol_mean) / vol_std).fillna(0.0).clip(-8.0, 8.0)

    expected = [
        "return_1h",
        "return_6h",
        "return_24h",
        "return_72h",
        "volatility_6h",
        "volatility_24h",
        "volatility_72h",
        "ma_delta_6h",
        "ma_delta_24h",
        "ma_delta_72h",
        "atr_pct_24h",
        "range_pct_1h",
        "trend_24h",
        "trend_72h",
        "drawdown_72h",
        "log_volume_z24h",
    ]
    feat = feat[expected].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    if feat.shape[1] != FEATURES_PER_SYM:
        raise AssertionError("Feature-width mismatch for MKTD export")
    return feat


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


def export_binary(
    symbols: Iterable[str],
    data_root: Path,
    output_path: Path,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    min_hours: int = 24 * 90,
    min_coverage: float = 0.95,
) -> dict[str, object]:
    root = Path(data_root)
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

        aligned = frame.reindex(index, method="ffill").bfill()
        aligned["volume"] = aligned["volume"].where(observed_mask.astype(bool), 0.0)
        aligned = aligned.fillna(0.0)
        features = compute_hourly_features(aligned)

        aligned_symbols.append(sym)
        feature_list.append(features.to_numpy(dtype=np.float32, copy=False))
        price_list.append(aligned[["open", "high", "low", "close", "volume"]].to_numpy(dtype=np.float32, copy=False))
        mask_list.append(observed_mask.astype(np.uint8, copy=False))

    if not aligned_symbols:
        raise ValueError(
            "No symbols met coverage threshold "
            f"min_coverage={min_coverage:.3f} over {len(index)} aligned hours"
        )

    num_symbols = len(aligned_symbols)
    num_timesteps = len(index)
    feature_arr = np.stack(feature_list, axis=1)  # [T, S, 16]
    price_arr = np.stack(price_list, axis=1)      # [T, S, 5]
    mask_arr = np.stack(mask_list, axis=1)        # [T, S]

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
        "start": str(index[0]),
        "end": str(index[-1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export hourly price-only MKTD binary for pufferlib_market")
    parser.add_argument(
        "--symbols",
        default="",
        help="Comma-separated symbol list. If empty, discover all CSV symbols under data-root.",
    )
    parser.add_argument("--data-root", default="trainingdatahourly", help="Root folder containing hourly CSV files")
    parser.add_argument("--output", default="pufferlib_market/data/hourly_price_mktd_v2.bin", help="Output .bin file")
    parser.add_argument("--start-date", default=None, help="Optional inclusive start timestamp/date (UTC)")
    parser.add_argument("--end-date", default=None, help="Optional inclusive end timestamp/date (UTC)")
    parser.add_argument("--min-hours", type=int, default=24 * 90, help="Minimum aligned hours required")
    parser.add_argument("--min-coverage", type=float, default=0.95, help="Minimum per-symbol coverage ratio [0,1]")
    args = parser.parse_args()

    symbols = [s for s in args.symbols.split(",")] if args.symbols else []
    report = export_binary(
        symbols=symbols,
        data_root=Path(args.data_root),
        output_path=Path(args.output),
        start_date=args.start_date,
        end_date=args.end_date,
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
