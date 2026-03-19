"""
Export market-hours-only OHLCV data to MKTD v2 binary.
Only includes hours when all symbols actually traded (no forward-fill gaps).
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

FEATURES_PER_SYM = 16
PRICE_FEATURES = 5
MAGIC = b"MKTD"
VERSION = 2


def _read_hourly_prices(symbol: str, data_root: Path) -> pd.DataFrame:
    paths = [
        data_root / f"{symbol.upper()}.csv",
        data_root / "crypto" / f"{symbol.upper()}.csv",
        data_root / "stocks" / f"{symbol.upper()}.csv",
    ]
    path = next((p for p in paths if p.exists()), None)
    if path is None:
        raise FileNotFoundError(f"No CSV found for {symbol} under {data_root}")

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
        raise ValueError(f"{path} missing: {missing}")

    out = frame.loc[:, list(required)].astype(float)
    for col in ("open", "high", "low", "close"):
        out[col] = out[col].where(out[col] > 0.0, np.nan)
    out["volume"] = out["volume"].where(out["volume"] >= 0.0, 0.0)
    out = out.dropna(subset=["open", "high", "low", "close"])
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
        "return_1h", "return_6h", "return_24h", "return_72h",
        "volatility_6h", "volatility_24h", "volatility_72h",
        "ma_delta_6h", "ma_delta_24h", "ma_delta_72h",
        "atr_pct_24h", "range_pct_1h",
        "trend_24h", "trend_72h", "drawdown_72h", "log_volume_z24h",
    ]
    feat = feat[expected].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    return feat


def export_binary_market_hours(
    symbols: list[str],
    data_root: Path,
    output_path: Path,
    *,
    min_hours: int = 2000,
) -> dict:
    """Export market-hours-only data (no forward fill)."""
    root = Path(data_root)
    symbol_list = [s.strip().upper() for s in symbols]

    # Load all price data
    raw_prices: dict[str, pd.DataFrame] = {}
    for sym in symbol_list:
        raw_prices[sym] = _read_hourly_prices(sym, root)
        print(f"Loaded {sym}: {len(raw_prices[sym])} hours")

    # Find common hours where ALL symbols have data
    common_hours = None
    for sym, df in raw_prices.items():
        hours = set(df.index)
        if common_hours is None:
            common_hours = hours
        else:
            common_hours &= hours

    common_hours = sorted(common_hours)
    print(f"Common trading hours: {len(common_hours)}")

    if len(common_hours) < min_hours:
        raise ValueError(f"Only {len(common_hours)} common hours, need {min_hours}")

    # Align all data to common hours
    aligned: dict[str, pd.DataFrame] = {}
    for sym in symbol_list:
        aligned[sym] = raw_prices[sym].loc[common_hours]

    # Compute features
    feature_list = []
    price_list = []
    mask_list = []

    for sym in symbol_list:
        df = aligned[sym]
        features = compute_hourly_features(df)
        feature_list.append(features.to_numpy(dtype=np.float32))
        price_list.append(df[["open", "high", "low", "close", "volume"]].to_numpy(dtype=np.float32))
        mask_list.append(np.ones(len(df), dtype=np.uint8))

    num_symbols = len(symbol_list)
    num_timesteps = len(common_hours)
    feature_arr = np.stack(feature_list, axis=1)
    price_arr = np.stack(price_list, axis=1)
    mask_arr = np.stack(mask_list, axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        header = struct.pack(
            "<4sIIIII40s",
            MAGIC, VERSION, num_symbols, num_timesteps,
            FEATURES_PER_SYM, PRICE_FEATURES, b"\x00" * 40,
        )
        handle.write(header)
        for sym in symbol_list:
            raw = sym.encode("ascii", errors="ignore")[:15]
            handle.write(raw + b"\x00" * (16 - len(raw)))
        handle.write(feature_arr.tobytes(order="C"))
        handle.write(price_arr.tobytes(order="C"))
        handle.write(mask_arr.tobytes(order="C"))

    return {
        "path": str(output_path),
        "symbols": symbol_list,
        "num_symbols": num_symbols,
        "num_timesteps": num_timesteps,
        "start": str(common_hours[0]),
        "end": str(common_hours[-1]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", required=True, help="Comma-separated symbol list")
    parser.add_argument("--data-root", default="trainingdatahourly", help="Root folder")
    parser.add_argument("--output", required=True, help="Output .bin file")
    parser.add_argument("--min-hours", type=int, default=2000, help="Minimum required hours")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    report = export_binary_market_hours(
        symbols=symbols,
        data_root=Path(args.data_root),
        output_path=Path(args.output),
        min_hours=args.min_hours,
    )
    print(f"Wrote {report['path']}")
    print(f"Symbols: {','.join(report['symbols'])}")
    print(f"Timesteps: {report['num_timesteps']}")
    print(f"Range: {report['start']} to {report['end']}")


if __name__ == "__main__":
    main()
