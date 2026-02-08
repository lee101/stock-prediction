"""
Export Chronos2 forecasts + hourly price data to a flat binary file for the C trading env.

Binary format:
  Header (64 bytes):
    magic:          4 bytes  "MKTD"
    version:        uint32
    num_symbols:    uint32
    num_timesteps:  uint32
    features_per_sym: uint32
    price_features: uint32  (always 5: OHLCV)
    padding:        40 bytes

  Symbol table: num_symbols * 16 bytes (null-padded ASCII names)

  Feature data:  float32[num_timesteps][num_symbols][features_per_sym]
    Per-symbol features:
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

  Price data:    float32[num_timesteps][num_symbols][5]  (open, high, low, close, volume)
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES_PER_SYM = 16
PRICE_FEATURES = 5
MAGIC = b"MKTD"
VERSION = 1


def load_price_data(symbol: str, data_root: Path) -> pd.DataFrame:
    """Load hourly OHLCV CSV for a symbol."""
    crypto_path = data_root / "crypto" / f"{symbol}.csv"
    stock_path = data_root / "stocks" / f"{symbol}.csv"
    path = crypto_path if crypto_path.exists() else stock_path
    if not path.exists():
        raise FileNotFoundError(f"No price data for {symbol} at {path}")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp", keep="last")
    df = df.set_index("timestamp")
    return df


def load_forecast(symbol: str, cache_root: Path, horizon: int) -> pd.DataFrame:
    """Load forecast parquet for a symbol/horizon."""
    path = cache_root / f"h{horizon}" / f"{symbol}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No forecast at {path}")
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"No timestamp index in {path}")
    df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    return df.sort_index()


def compute_features(price_df: pd.DataFrame, fc_h1: pd.DataFrame, fc_h24: pd.DataFrame) -> pd.DataFrame:
    """Compute the 16 features per symbol from price + forecast data."""
    close = price_df["close"].astype(float)
    high = price_df["high"].astype(float)
    low = price_df["low"].astype(float)

    feat = pd.DataFrame(index=price_df.index)

    # Forecast deltas (h1)
    for col_prefix, feat_name in [
        ("predicted_close_p50", "chronos_close_delta"),
        ("predicted_high_p50", "chronos_high_delta"),
        ("predicted_low_p50", "chronos_low_delta"),
    ]:
        src_col = f"{col_prefix}_h1" if f"{col_prefix}_h1" in fc_h1.columns else col_prefix
        if src_col in fc_h1.columns:
            aligned = fc_h1[src_col].reindex(feat.index)
            feat[f"{feat_name}_h1"] = (aligned - close) / close.clip(lower=1e-8)
        else:
            feat[f"{feat_name}_h1"] = 0.0

    # Forecast deltas (h24)
    for col_prefix, feat_name in [
        ("predicted_close_p50", "chronos_close_delta"),
        ("predicted_high_p50", "chronos_high_delta"),
        ("predicted_low_p50", "chronos_low_delta"),
    ]:
        src_col = f"{col_prefix}_h24" if f"{col_prefix}_h24" in fc_h24.columns else col_prefix
        if src_col in fc_h24.columns:
            aligned = fc_h24[src_col].reindex(feat.index)
            feat[f"{feat_name}_h24"] = (aligned - close) / close.clip(lower=1e-8)
        else:
            feat[f"{feat_name}_h24"] = 0.0

    # Forecast confidence (inverse quantile spread)
    for h, fc in [("h1", fc_h1), ("h24", fc_h24)]:
        p90_col = f"predicted_close_p90_{h}" if f"predicted_close_p90_{h}" in fc.columns else "predicted_close_p90"
        p10_col = f"predicted_close_p10_{h}" if f"predicted_close_p10_{h}" in fc.columns else "predicted_close_p10"
        if p90_col in fc.columns and p10_col in fc.columns:
            spread = (fc[p90_col] - fc[p10_col]).reindex(feat.index).abs()
            feat[f"forecast_confidence_{h}"] = 1.0 / (1.0 + spread / close.clip(lower=1e-8))
        else:
            feat[f"forecast_confidence_{h}"] = 0.5

    # Technical features from price data
    feat["return_1h"] = close.pct_change(1).fillna(0).clip(-0.5, 0.5)
    feat["return_24h"] = close.pct_change(24).fillna(0).clip(-1.0, 1.0)
    feat["volatility_24h"] = close.pct_change(1).rolling(24, min_periods=1).std().fillna(0.01)

    ma24 = close.rolling(24, min_periods=1).mean()
    ma72 = close.rolling(72, min_periods=1).mean()
    feat["ma_delta_24h"] = ((close - ma24) / ma24.clip(lower=1e-8)).clip(-0.5, 0.5)
    feat["ma_delta_72h"] = ((close - ma72) / ma72.clip(lower=1e-8)).clip(-0.5, 0.5)

    # ATR % of close
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr24 = tr.rolling(24, min_periods=1).mean()
    feat["atr_pct_24h"] = (atr24 / close.clip(lower=1e-8)).clip(0, 0.5)

    # Trend (linear regression slope over 72h, normalized)
    feat["trend_72h"] = close.pct_change(72).fillna(0).clip(-1.0, 1.0)

    # Drawdown from 72h rolling max
    roll_max = close.rolling(72, min_periods=1).max()
    feat["drawdown_72h"] = ((close - roll_max) / roll_max.clip(lower=1e-8)).clip(-1.0, 0.0)

    # Fill NaN and ensure column order
    expected_cols = [
        "chronos_close_delta_h1", "chronos_high_delta_h1", "chronos_low_delta_h1",
        "chronos_close_delta_h24", "chronos_high_delta_h24", "chronos_low_delta_h24",
        "forecast_confidence_h1", "forecast_confidence_h24",
        "return_1h", "return_24h", "volatility_24h",
        "ma_delta_24h", "ma_delta_72h", "atr_pct_24h",
        "trend_72h", "drawdown_72h",
    ]
    for col in expected_cols:
        if col not in feat.columns:
            feat[col] = 0.0

    feat = feat[expected_cols].fillna(0.0)
    return feat


def export_binary(
    symbols: list[str],
    forecast_cache_root: Path,
    data_root: Path,
    output_path: Path,
    min_rows: int = 200,
):
    """Export all data to a single binary file."""
    print(f"Exporting data for {len(symbols)} symbols to {output_path}")

    # Load and align all data
    all_features = {}
    all_prices = {}
    common_index = None

    for sym in symbols:
        print(f"  Loading {sym}...")
        try:
            prices = load_price_data(sym, data_root)
            fc_h1 = load_forecast(sym, forecast_cache_root, 1)
            fc_h24 = load_forecast(sym, forecast_cache_root, 24)
        except FileNotFoundError as e:
            print(f"  SKIP {sym}: {e}")
            continue

        feats = compute_features(prices, fc_h1, fc_h24)

        # Use intersection of forecast and price timestamps
        valid_idx = feats.index.intersection(prices.index)
        valid_idx = valid_idx.intersection(fc_h1.index).intersection(fc_h24.index)
        if len(valid_idx) < min_rows:
            print(f"  SKIP {sym}: only {len(valid_idx)} valid rows (need {min_rows})")
            continue

        all_features[sym] = feats.loc[valid_idx]
        all_prices[sym] = prices.loc[valid_idx, ["open", "high", "low", "close", "volume"]]

        if common_index is None:
            common_index = valid_idx
        else:
            common_index = common_index.intersection(valid_idx)

    if common_index is None or len(common_index) < min_rows:
        print(f"ERROR: Not enough common timestamps ({len(common_index) if common_index is not None else 0})")
        sys.exit(1)

    common_index = common_index.sort_values()
    valid_symbols = [s for s in symbols if s in all_features]
    num_symbols = len(valid_symbols)
    num_timesteps = len(common_index)

    print(f"  Common timesteps: {num_timesteps}, symbols: {num_symbols}")
    print(f"  Time range: {common_index[0]} to {common_index[-1]}")

    # Build arrays
    feature_arr = np.zeros((num_timesteps, num_symbols, FEATURES_PER_SYM), dtype=np.float32)
    price_arr = np.zeros((num_timesteps, num_symbols, PRICE_FEATURES), dtype=np.float32)

    for si, sym in enumerate(valid_symbols):
        f = all_features[sym].reindex(common_index).fillna(0.0)
        p = all_prices[sym].reindex(common_index).fillna(method="ffill").fillna(method="bfill")
        feature_arr[:, si, :] = f.values.astype(np.float32)
        price_arr[:, si, :] = p.values.astype(np.float32)

    # Write binary file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        # Header (64 bytes)
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

        # Symbol table
        for sym in valid_symbols:
            name_bytes = sym.encode("ascii")[:15].ljust(16, b"\x00")
            f.write(name_bytes)

        # Feature data
        f.write(feature_arr.tobytes())

        # Price data
        f.write(price_arr.tobytes())

    total_size = output_path.stat().st_size
    print(f"  Written {total_size:,} bytes ({total_size / 1024:.1f} KB)")
    print(f"  Symbols: {valid_symbols}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export trading data to binary format for C env")
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols")
    parser.add_argument("--forecast-cache-root", required=True, help="Path to forecast cache root")
    parser.add_argument("--data-root", default="trainingdatahourly", help="Path to hourly price data root")
    parser.add_argument("--output", default="pufferlib_market/data/market_data.bin", help="Output binary file")
    parser.add_argument("--min-rows", type=int, default=200, help="Minimum valid rows per symbol")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]
    export_binary(
        symbols=symbols,
        forecast_cache_root=Path(args.forecast_cache_root),
        data_root=Path(args.data_root),
        output_path=Path(args.output),
        min_rows=args.min_rows,
    )


if __name__ == "__main__":
    main()
