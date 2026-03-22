"""
Export stocks20 daily OHLCV data to MKTD v2 binary format for pufferlib_market.

20 diverse symbols chosen for broad market coverage, liquidity, and long history:
  Core tech: AAPL, MSFT, NVDA, GOOG, META, TSLA, AMZN, AMD
  Finance/ETFs: JPM, SPY, QQQ
  Growth tech: PLTR, NET, NFLX, ADBE, CRM
  Diversification: AVGO, V, COST, ADSK

Data is pulled from:
  1. trainingdata/train/{SYM}.csv  — daily bars (long history, ~2022+)
  2. trainingdatahourly/stocks/{SYM}.csv — hourly bars resampled to daily (recent extension)

Binary format (v2):
  Header (64 bytes): magic "MKTD" + version + num_symbols + num_timesteps + reserved
  Symbol table: num_symbols * 16 bytes
  Feature data: float32[T][S][16]
  Price data:   float32[T][S][5]
  Tradable mask: uint8[T][S]

Train/val split: val = 2025-09-01 onwards, train = everything before.

Outputs:
  pufferlib_market/data/stocks20_daily_train.bin
  pufferlib_market/data/stocks20_daily_val.bin
"""

from __future__ import annotations

import argparse
import struct
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Locate the repo root so the script works from any cwd including worktrees
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
# Walk up until we find trainingdata/train (repo root indicator)
_REPO_ROOT = _SCRIPT_DIR
for _ in range(5):
    if (_REPO_ROOT / "trainingdata" / "train").exists():
        break
    _REPO_ROOT = _REPO_ROOT.parent
else:
    _REPO_ROOT = _SCRIPT_DIR

sys.path.insert(0, str(_REPO_ROOT))

from export_data_daily import compute_daily_features  # noqa: E402

FEATURES_PER_SYM = 16
PRICE_FEATURES = 5
MAGIC = b"MKTD"
VERSION = 2

SYMBOLS_20 = [
    # Core tech (large cap, long history)
    "AAPL", "MSFT", "NVDA", "GOOG", "META", "TSLA", "AMZN", "AMD",
    # Finance & ETFs
    "JPM", "SPY", "QQQ",
    # Growth tech
    "PLTR", "NET", "NFLX", "ADBE", "CRM",
    # Additional diversification
    "AVGO", "V", "COST", "ADSK",
]

DAILY_DATA_ROOT = _REPO_ROOT / "trainingdata" / "train"
HOURLY_DATA_ROOT = _REPO_ROOT / "trainingdatahourly" / "stocks"
OUTPUT_DIR = _SCRIPT_DIR / "pufferlib_market" / "data"
VAL_SPLIT_DATE = "2025-09-01"
MIN_TRAIN_DAYS = 500
MIN_VAL_DAYS = 30


def _load_daily_csv(path: Path) -> pd.DataFrame:
    """Load a daily OHLCV CSV (trainingdata/train format)."""
    df = pd.read_csv(path)
    df.columns = [str(c).lower() for c in df.columns]
    ts_col = "date" if "date" in df.columns else "timestamp"
    df["_ts"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=["_ts"]).set_index("_ts").sort_index()
    df.index = df.index.floor("D")
    df = df[~df.index.duplicated(keep="first")]
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            raise ValueError(f"{path} missing column '{col}'")
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def _load_hourly_csv_as_daily(path: Path) -> pd.DataFrame:
    """Resample hourly OHLCV CSV to daily bars."""
    df = pd.read_csv(path)
    df.columns = [str(c).lower() for c in df.columns]
    ts_col = "timestamp" if "timestamp" in df.columns else "date"
    df["_ts"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=["_ts"]).set_index("_ts").sort_index()
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            raise ValueError(f"{path} missing column '{col}'")
    daily = df[["open", "high", "low", "close", "volume"]].resample("D").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    daily.index = daily.index.floor("D")
    daily = daily.dropna(subset=["open"])
    return daily.astype(float)


def load_symbol_daily(symbol: str) -> pd.DataFrame:
    """Merge daily CSV + hourly-resampled CSV; daily data takes priority."""
    symbol = symbol.upper()
    frames: list[pd.DataFrame] = []

    daily_path = DAILY_DATA_ROOT / f"{symbol}.csv"
    if daily_path.exists():
        try:
            frames.append(_load_daily_csv(daily_path))
        except Exception as e:
            warnings.warn(f"[{symbol}] daily CSV load failed: {e}")

    hourly_path = HOURLY_DATA_ROOT / f"{symbol}.csv"
    if hourly_path.exists():
        try:
            frames.append(_load_hourly_csv_as_daily(hourly_path))
        except Exception as e:
            warnings.warn(f"[{symbol}] hourly CSV load failed: {e}")

    if not frames:
        raise FileNotFoundError(
            f"No daily or hourly data found for {symbol} "
            f"(looked in {DAILY_DATA_ROOT} and {HOURLY_DATA_ROOT})"
        )

    if len(frames) == 1:
        return frames[0]

    base, extra = frames[0], frames[1]
    new_dates = extra.index.difference(base.index)
    merged = pd.concat([base, extra.loc[new_dates]]).sort_index()
    merged = merged[~merged.index.duplicated(keep="first")]
    return merged


def write_mktd_v2(
    output_path: Path,
    symbols: list[str],
    full_index: pd.DatetimeIndex,
    aligned_prices: dict[str, pd.DataFrame],
    aligned_feats: dict[str, pd.DataFrame],
    tradable: dict[str, np.ndarray],
) -> None:
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
    with open(output_path, "wb") as fh:
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
        fh.write(header)
        for sym in symbols:
            raw = sym.encode("ascii", errors="ignore")[:15]
            fh.write(raw + b"\x00" * (16 - len(raw)))
        fh.write(feature_arr.tobytes(order="C"))
        fh.write(price_arr.tobytes(order="C"))
        fh.write(mask_arr.tobytes(order="C"))

    print(f"Wrote {output_path} ({num_symbols} symbols, {num_timesteps} days)")
    print(f"  Date range : {full_index[0].date()} -> {full_index[-1].date()}")


def export_stocks20(
    symbols: list[str] | None = None,
    output_dir: Path = OUTPUT_DIR,
    val_split_date: str = VAL_SPLIT_DATE,
) -> tuple[Path, Path]:
    """Export stocks20 as MKTD v2 train + val binary files."""
    if symbols is None:
        symbols = list(SYMBOLS_20)

    symbols = [s.strip().upper() for s in symbols if s.strip()]
    print(f"Exporting {len(symbols)} symbols: {symbols}")
    print(f"  daily_root  : {DAILY_DATA_ROOT}")
    print(f"  hourly_root : {HOURLY_DATA_ROOT}")

    raw_prices: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            df = load_symbol_daily(sym)
            raw_prices[sym] = df
            print(f"  {sym}: {df.index.min().date()} - {df.index.max().date()}, {len(df)} daily bars")
        except FileNotFoundError as e:
            warnings.warn(f"Skipping {sym}: {e}")

    if not raw_prices:
        raise RuntimeError("No symbol data loaded — aborting")

    active_symbols = list(raw_prices.keys())
    print(f"\n{len(active_symbols)} symbols loaded successfully")

    # Common calendar window (intersection of all symbols)
    start = max(df.index.min() for df in raw_prices.values())
    end = min(df.index.max() for df in raw_prices.values())
    if start >= end:
        raise ValueError(f"No overlapping date window: start={start} end={end}")

    full_index = pd.date_range(start.floor("D"), end.floor("D"), freq="D", tz="UTC")
    print(f"\nCommon window: {full_index[0].date()} -> {full_index[-1].date()} ({len(full_index)} calendar days)")

    aligned_prices: dict[str, pd.DataFrame] = {}
    aligned_feats: dict[str, pd.DataFrame] = {}
    tradable: dict[str, np.ndarray] = {}

    for sym, df in raw_prices.items():
        mask = full_index.isin(df.index).astype(np.uint8)
        re = df.reindex(full_index, method="ffill")
        re["volume"] = re["volume"].where(mask.astype(bool), 0.0)
        re = re.bfill().fillna(0.0)
        aligned_prices[sym] = re
        aligned_feats[sym] = compute_daily_features(re)
        tradable[sym] = mask

    val_start = pd.Timestamp(val_split_date, tz="UTC")
    train_idx = full_index[full_index < val_start]
    val_idx = full_index[full_index >= val_start]

    if len(train_idx) < MIN_TRAIN_DAYS:
        raise ValueError(f"Only {len(train_idx)} train days (need {MIN_TRAIN_DAYS})")
    if len(val_idx) < MIN_VAL_DAYS:
        raise ValueError(f"Only {len(val_idx)} val days (need {MIN_VAL_DAYS})")

    print(f"\nTrain: {len(train_idx)} days  ({train_idx[0].date()} -> {train_idx[-1].date()})")
    print(f"Val  : {len(val_idx)} days  ({val_idx[0].date()} -> {val_idx[-1].date()})")

    train_tradable = {sym: tradable[sym][full_index < val_start] for sym in active_symbols}
    val_tradable = {sym: tradable[sym][full_index >= val_start] for sym in active_symbols}

    train_prices = {sym: aligned_prices[sym].reindex(train_idx) for sym in active_symbols}
    val_prices = {sym: aligned_prices[sym].reindex(val_idx) for sym in active_symbols}

    train_feats = {sym: aligned_feats[sym].reindex(train_idx) for sym in active_symbols}
    val_feats = {sym: aligned_feats[sym].reindex(val_idx) for sym in active_symbols}

    train_path = output_dir / "stocks20_daily_train.bin"
    val_path = output_dir / "stocks20_daily_val.bin"

    print()
    write_mktd_v2(train_path, active_symbols, train_idx, train_prices, train_feats, train_tradable)
    write_mktd_v2(val_path, active_symbols, val_idx, val_prices, val_feats, val_tradable)

    return train_path, val_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export stocks20 as MKTD v2 daily binary for pufferlib_market"
    )
    parser.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated symbol list; defaults to SYMBOLS_20",
    )
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--val-split-date", default=VAL_SPLIT_DATE)
    args = parser.parse_args()

    syms = [s.strip() for s in args.symbols.split(",")] if args.symbols else None
    try:
        export_stocks20(
            symbols=syms,
            output_dir=Path(args.output_dir),
            val_split_date=args.val_split_date,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
