"""
Export hourly OHLCV data to MKTD binary WITHOUT requiring Chronos2 forecasts.

Uses 16 technical features computed from hourly price data, analogous to
the daily features in export_data_daily.py but with hour-appropriate lookbacks.

Binary format matches trading_env.h exactly (MKTD v1).
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
VERSION = 1


def load_price_data(symbol: str, data_root: Path) -> pd.DataFrame:
    """Load hourly OHLCV CSV for a symbol."""
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
    if "timestamp" not in df.columns:
        raise ValueError(f"{path} missing timestamp column")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp", keep="last")
    df = df.set_index("timestamp")
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return df[required].astype(float)


def compute_hourly_features(price_df: pd.DataFrame) -> pd.DataFrame:
    """Compute 16 price-only hourly features for the RL agent.

    These mirror the daily features in export_data_daily.py but use
    hour-appropriate lookback windows:
      daily 1d/5d/20d/60d → hourly 1h/24h/120h(5d)/480h(20d)
    """
    close = price_df["close"].astype(float)
    high = price_df["high"].astype(float)
    low = price_df["low"].astype(float)
    volume = price_df["volume"].astype(float)

    feat = pd.DataFrame(index=price_df.index)

    ret_1h = close.pct_change(1).fillna(0.0)
    feat["return_1h"] = ret_1h.clip(-0.5, 0.5)
    feat["return_24h"] = close.pct_change(24).fillna(0.0).clip(-1.0, 1.0)
    feat["return_120h"] = close.pct_change(120).fillna(0.0).clip(-2.0, 2.0)

    feat["volatility_24h"] = ret_1h.rolling(24, min_periods=1).std(ddof=0).fillna(0.01).clip(0.0, 1.0)
    feat["volatility_120h"] = ret_1h.rolling(120, min_periods=1).std(ddof=0).fillna(0.01).clip(0.0, 1.0)

    ma24 = close.rolling(24, min_periods=1).mean()
    ma120 = close.rolling(120, min_periods=1).mean()
    ma480 = close.rolling(480, min_periods=1).mean()
    feat["ma_delta_24h"] = ((close - ma24) / ma24.clip(lower=1e-8)).fillna(0.0).clip(-0.5, 0.5)
    feat["ma_delta_120h"] = ((close - ma120) / ma120.clip(lower=1e-8)).fillna(0.0).clip(-0.5, 0.5)
    feat["ma_delta_480h"] = ((close - ma480) / ma480.clip(lower=1e-8)).fillna(0.0).clip(-0.5, 0.5)

    # ATR % of close (24h)
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    atr24 = tr.rolling(24, min_periods=1).mean()
    feat["atr_pct_24h"] = (atr24 / close.clip(lower=1e-8)).fillna(0.0).clip(0.0, 0.5)
    feat["range_pct_1h"] = ((high - low) / close.clip(lower=1e-8)).fillna(0.0).clip(0.0, 0.5)

    feat["trend_120h"] = close.pct_change(120).fillna(0.0).clip(-2.0, 2.0)
    feat["trend_480h"] = close.pct_change(480).fillna(0.0).clip(-3.0, 3.0)

    roll_max_120 = close.rolling(120, min_periods=1).max()
    roll_max_480 = close.rolling(480, min_periods=1).max()
    feat["drawdown_120h"] = ((close - roll_max_120) / roll_max_120.clip(lower=1e-8)).fillna(0.0).clip(-1.0, 0.0)
    feat["drawdown_480h"] = ((close - roll_max_480) / roll_max_480.clip(lower=1e-8)).fillna(0.0).clip(-1.0, 0.0)

    log_vol = np.log1p(volume.clip(lower=0.0))
    log_vol_mean120 = log_vol.rolling(120, min_periods=1).mean()
    log_vol_std120 = log_vol.rolling(120, min_periods=1).std(ddof=0).replace(0.0, 1.0)
    feat["log_volume_z120h"] = ((log_vol - log_vol_mean120) / log_vol_std120).fillna(0.0).clip(-5.0, 5.0)
    feat["log_volume_delta_24h"] = (log_vol - log_vol.rolling(24, min_periods=1).mean()).fillna(0.0).clip(-10.0, 10.0)

    feat = feat.fillna(0.0).astype(np.float32)
    if feat.shape[1] != FEATURES_PER_SYM:
        raise AssertionError(f"Expected {FEATURES_PER_SYM} features, got {feat.shape[1]}")
    return feat


def export_binary(
    symbols: list[str],
    data_root: Path,
    output_path: Path,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    min_rows: int = 200,
) -> None:
    symbols = [s.strip().upper() for s in symbols if s.strip()]
    if not symbols:
        raise ValueError("No symbols provided")

    all_prices: dict[str, pd.DataFrame] = {}
    all_feats: dict[str, pd.DataFrame] = {}

    for sym in symbols:
        df = load_price_data(sym, data_root)
        all_prices[sym] = df
        all_feats[sym] = compute_hourly_features(df)

    # Find common timestamps
    common_idx = all_feats[symbols[0]].index
    for sym in symbols[1:]:
        common_idx = common_idx.intersection(all_feats[sym].index)

    common_idx = common_idx.sort_values()

    if start_date:
        common_idx = common_idx[common_idx >= pd.Timestamp(start_date, tz="UTC")]
    if end_date:
        common_idx = common_idx[common_idx <= pd.Timestamp(end_date, tz="UTC")]

    if len(common_idx) < min_rows:
        raise ValueError(f"Not enough common timestamps: {len(common_idx)} (need {min_rows})")

    num_timesteps = len(common_idx)
    num_symbols = len(symbols)

    feature_arr = np.zeros((num_timesteps, num_symbols, FEATURES_PER_SYM), dtype=np.float32)
    price_arr = np.zeros((num_timesteps, num_symbols, PRICE_FEATURES), dtype=np.float32)

    for si, sym in enumerate(symbols):
        f = all_feats[sym].reindex(common_idx).fillna(0.0)
        p = all_prices[sym].reindex(common_idx).ffill().bfill().fillna(0.0)
        feature_arr[:, si, :] = f.values.astype(np.float32, copy=False)
        price_arr[:, si, :] = p[["open", "high", "low", "close", "volume"]].values.astype(np.float32, copy=False)

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

    print(f"Wrote {output_path} ({num_symbols} symbols, {num_timesteps} hours)")
    print(f"Date range: {common_idx[0]} to {common_idx[-1]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export hourly price-only data to MKTD binary")
    parser.add_argument("--symbols", required=True, help="Comma-separated symbol list")
    parser.add_argument("--data-root", default="trainingdatahourly", help="Directory containing hourly CSVs")
    parser.add_argument("--output", required=True, help="Output .bin path")
    parser.add_argument("--start-date", default=None, help="Optional ISO start date")
    parser.add_argument("--end-date", default=None, help="Optional ISO end date")
    parser.add_argument("--min-rows", type=int, default=200, help="Minimum valid rows")
    args = parser.parse_args()

    export_binary(
        symbols=[s for s in args.symbols.split(",")],
        data_root=Path(args.data_root),
        output_path=Path(args.output),
        start_date=args.start_date,
        end_date=args.end_date,
        min_rows=args.min_rows,
    )


if __name__ == "__main__":
    main()
