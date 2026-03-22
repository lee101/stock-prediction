"""CLI batch forecaster using CuteChronos2Pipeline for worksteal universe.

Usage:
    python batch_forecast.py --symbols BTCUSDT ETHUSDT --data-dir trainingdata/train/
    python batch_forecast.py --all --data-dir trainingdata/train/ --use-cute
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from binance_worksteal.cute_forecast import forecast_batch, get_pipeline

logger = logging.getLogger(__name__)

ALL_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT",
    "AAVEUSDT", "LTCUSDT", "XRPUSDT", "DOTUSDT", "UNIUSDT", "NEARUSDT",
    "APTUSDT", "ICPUSDT", "SHIBUSDT", "ADAUSDT", "FILUSDT", "ARBUSDT",
    "OPUSDT", "INJUSDT", "SUIUSDT", "TIAUSDT", "SEIUSDT", "ATOMUSDT",
    "ALGOUSDT", "BCHUSDT", "BNBUSDT", "TRXUSDT", "PEPEUSDT", "MATICUSDT",
]

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / "binanceneural" / "forecast_cache" / "h24"


def load_bars(data_dir: Path, symbol: str) -> pd.DataFrame:
    csv_path = data_dir / f"{symbol}.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def forecast_to_cache_rows(
    symbol: str,
    bars_df: pd.DataFrame,
    forecast: dict,
    horizon: int = 24,
) -> pd.DataFrame:
    """Convert forecast dict to cache-compatible rows."""
    last_row = bars_df.iloc[-1]
    ts = last_row.get("timestamp", pd.Timestamp.utcnow())
    if not isinstance(ts, pd.Timestamp):
        ts = pd.to_datetime(ts, utc=True)

    rows = []
    for step in range(horizon):
        target_ts = ts + pd.Timedelta(hours=step + 1)
        row = {
            "timestamp": target_ts,
            "symbol": symbol,
            "issued_at": ts,
            "target_timestamp": target_ts,
            "horizon_hours": horizon,
            "predicted_close_p50": float(forecast["close"]["p50"][step]),
            "predicted_close_p10": float(forecast["close"]["p10"][step]),
            "predicted_close_p90": float(forecast["close"]["p90"][step]),
            "predicted_high_p50": float(forecast["high"]["p50"][step]),
            "predicted_low_p50": float(forecast["low"]["p50"][step]),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def run_batch(
    symbols: list[str],
    data_dir: Path,
    cache_dir: Path,
    horizon: int = 24,
    batch_size: int = 8,
    use_cute: bool = True,
    context_length: int = 512,
    device: str = "cuda",
):
    bars_dict: dict[str, pd.DataFrame] = {}
    skipped = []
    for sym in symbols:
        df = load_bars(data_dir, sym)
        if df.empty or len(df) < 32:
            skipped.append(sym)
            continue
        bars_dict[sym] = df

    if skipped:
        logger.info("Skipped %d symbols (no data): %s", len(skipped), skipped)

    valid_symbols = list(bars_dict.keys())
    if not valid_symbols:
        logger.error("No valid symbols to forecast")
        return

    logger.info("Forecasting %d symbols, horizon=%d, batch=%d, cute=%s",
                len(valid_symbols), horizon, batch_size, use_cute)

    t0 = time.time()

    # warm up pipeline
    _ = get_pipeline(device=device, use_cute=use_cute)

    results = forecast_batch(
        valid_symbols, bars_dict,
        horizon=horizon, device=device, use_cute=use_cute,
        batch_size=batch_size, context_length=context_length,
    )

    elapsed = time.time() - t0
    logger.info("Batch forecast done in %.1fs for %d symbols", elapsed, len(results))

    cache_dir.mkdir(parents=True, exist_ok=True)
    for sym, forecast in results.items():
        df = bars_dict[sym]
        cache_rows = forecast_to_cache_rows(sym, df, forecast, horizon)

        cache_path = cache_dir / f"{sym}.parquet"
        if cache_path.exists():
            existing = pd.read_parquet(cache_path)
            existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)
            combined = pd.concat([existing, cache_rows], ignore_index=True)
            combined = combined.drop_duplicates(subset=["timestamp", "symbol"], keep="last")
            combined = combined.sort_values("timestamp").reset_index(drop=True)
            combined.to_parquet(cache_path, index=False)
            logger.info("Updated %s: %d rows total", cache_path.name, len(combined))
        else:
            cache_rows.to_parquet(cache_path, index=False)
            logger.info("Created %s: %d rows", cache_path.name, len(cache_rows))

    logger.info("All done. %d symbols written to %s", len(results), cache_dir)


def main():
    parser = argparse.ArgumentParser(description="Batch Chronos2 forecasting for worksteal symbols")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--all", action="store_true", help="Use all 30 worksteal symbols")
    parser.add_argument("--data-dir", type=str, default="trainingdata/train/")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--use-cute", action="store_true", default=True)
    parser.add_argument("--no-cute", dest="use_cute", action="store_false")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    symbols = args.symbols or (ALL_SYMBOLS if args.all else ALL_SYMBOLS[:5])
    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else DEFAULT_CACHE_DIR

    run_batch(
        symbols=symbols,
        data_dir=data_dir,
        cache_dir=cache_dir,
        horizon=args.horizon,
        batch_size=args.batch_size,
        use_cute=args.use_cute,
        context_length=args.context_length,
        device=args.device,
    )


if __name__ == "__main__":
    main()
