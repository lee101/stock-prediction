from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
from alpaca.data import TimeFrame, TimeFrameUnit
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alpaca_wrapper import download_symbol_history


def refresh(symbol: str, data_path: Path, max_days: int = 4 * 365) -> Path:
    symbol = symbol.upper()
    data_path.parent.mkdir(parents=True, exist_ok=True)

    existing = pd.DataFrame()
    if data_path.exists():
        existing = pd.read_csv(data_path)
        existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=max_days)

    print(f"Downloading {symbol} hourly bars from {start} to {end} …")
    fresh = download_symbol_history(
        symbol=symbol,
        start=start,
        end=end,
        include_latest=True,
        timeframe=TimeFrame(1, TimeFrameUnit.Hour),
    ).reset_index()
    fresh["timestamp"] = pd.to_datetime(fresh["timestamp"], utc=True)
    fresh["symbol"] = symbol

    combined = (
        pd.concat([existing, fresh], ignore_index=True)
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    combined.to_csv(data_path, index=False)
    print(f"Saved merged {len(combined)} rows to {data_path}")

    # Report largest gap
    ts = combined["timestamp"]
    deltas = ts.diff().dt.total_seconds().div(3600)
    max_gap = deltas.max()
    if pd.notna(max_gap) and max_gap > 1.5:
        idx = deltas.idxmax()
        print(
            f"Largest gap: {max_gap:.2f}h between {ts.iloc[idx-1]} and {ts.iloc[idx]} "
            f"({len(combined)} rows total)",
        )
    else:
        print("No gaps > 1.5h detected.")

    return data_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Force-refresh full hourly crypto history for a symbol.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("trainingdatahourly/crypto"),
        help="Directory containing SYMBOL.csv",
    )
    parser.add_argument("--days", type=int, default=4 * 365, help="Lookback days to download (default 4y).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    path = args.data_root / f"{args.symbol.upper()}.csv"
    refresh(args.symbol, path, max_days=args.days)
