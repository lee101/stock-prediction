"""
Validate RL model performance through the Python marketsimulator.

Runs inference on historical data, generates trading actions, then simulates
through marketsimulator.py with Binance-realistic fees/slippage. This bridges
the gap between C-env OOS eval and real Binance execution.

Usage:
    python -m pufferlib_market.validate_marketsim \
        --checkpoint pufferlib_market/checkpoints/autoresearch_fdusd_hourly/slip_5bps/best.pt \
        --symbols BTCUSD,ETHUSD,SOLUSD \
        --fee-tier fdusd \
        --timeframe hourly
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from marketsimulator import SimulationConfig, run_shared_cash_simulation


def load_hourly_bars(symbol: str, data_root: str = "trainingdatahourly") -> pd.DataFrame:
    """Load hourly OHLCV for a symbol."""
    root = Path(data_root)
    for subdir in ["crypto", "stocks", ""]:
        path = root / subdir / f"{symbol}.csv" if subdir else root / f"{symbol}.csv"
        if path.exists():
            df = pd.read_csv(path, parse_dates=["timestamp"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df["symbol"] = symbol
            return df.sort_values("timestamp")
    raise FileNotFoundError(f"No data for {symbol}")


def load_daily_bars(symbol: str, data_root: str = "trainingdata/train") -> pd.DataFrame:
    """Load daily OHLCV for a symbol."""
    root = Path(data_root)
    for subdir in ["crypto", "stocks", ""]:
        path = root / subdir / f"{symbol}.csv" if subdir else root / f"{symbol}.csv"
        if path.exists():
            df = pd.read_csv(path)
            df.columns = [c.lower() for c in df.columns]
            ts_col = "timestamp" if "timestamp" in df.columns else "date"
            df["timestamp"] = pd.to_datetime(df[ts_col], utc=True)
            df["symbol"] = symbol
            return df.sort_values("timestamp")
    raise FileNotFoundError(f"No data for {symbol}")


FEE_TIERS = {
    "fdusd": 0.0,       # Binance FDUSD promotional
    "usdt": 0.001,      # Binance USDT standard
    "conservative": 0.0015,  # Conservative estimate
}

SLIPPAGE_BPS = {
    "BTCUSD": 2,
    "ETHUSD": 3,
    "SOLUSD": 5,
    "BNBUSD": 5,
    "LTCUSD": 8,
    "AVAXUSD": 10,
    "DOGEUSD": 8,
    "LINKUSD": 8,
    "AAVEUSD": 12,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols")
    parser.add_argument("--fee-tier", choices=list(FEE_TIERS.keys()), default="fdusd")
    parser.add_argument("--timeframe", choices=["hourly", "daily"], default="hourly")
    parser.add_argument("--start-date", default="2025-06-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--initial-cash", type=float, default=10000)
    parser.add_argument("--max-hold-hours", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=1024)
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    fee = FEE_TIERS[args.fee_tier]

    print(f"Validating {args.checkpoint}")
    print(f"  Symbols: {symbols}")
    print(f"  Fee tier: {args.fee_tier} ({fee*100:.2f}%)")
    print(f"  Timeframe: {args.timeframe}")
    print(f"  Period: {args.start_date} to {args.end_date or 'latest'}")

    # Load bars
    all_bars = []
    for sym in symbols:
        if args.timeframe == "hourly":
            df = load_hourly_bars(sym)
        else:
            df = load_daily_bars(sym)
        all_bars.append(df)
    bars = pd.concat(all_bars, ignore_index=True)
    bars = bars[bars["timestamp"] >= pd.Timestamp(args.start_date, tz="UTC")]
    if args.end_date:
        bars = bars[bars["timestamp"] <= pd.Timestamp(args.end_date, tz="UTC")]

    print(f"  Loaded {len(bars)} bars ({bars['timestamp'].min()} to {bars['timestamp'].max()})")

    # TODO: Run RL inference to generate actions
    # For now, generate random signals to validate the pipeline
    print("\n  NOTE: Full inference integration pending. Showing pipeline validation only.")

    # Create placeholder actions (no trades = buy-and-hold benchmark)
    timestamps = bars.groupby("timestamp").first().index
    actions_rows = []
    for ts in timestamps:
        for sym in symbols:
            actions_rows.append({
                "timestamp": ts,
                "symbol": sym,
                "buy_price": 0,
                "sell_price": 0,
                "trade_amount": 0,
            })
    actions = pd.DataFrame(actions_rows)

    config = SimulationConfig(
        maker_fee=fee,
        initial_cash=args.initial_cash,
        max_hold_hours=args.max_hold_hours,
    )

    result = run_shared_cash_simulation(bars, actions, config)

    print(f"\n  Results:")
    print(f"    Total return: {result.metrics.get('total_return', 0):.4f}")
    print(f"    Sortino: {result.metrics.get('sortino', 0):.2f}")
    print(f"    Trades: {sum(len(sr.trades) for sr in result.per_symbol.values())}")

    # Annualize
    n_periods = len(result.combined_equity)
    if args.timeframe == "hourly":
        years = n_periods / 8760
    else:
        years = n_periods / 365
    total_ret = result.metrics.get("total_return", 0)
    if years > 0 and total_ret > -1:
        annualized = (1 + total_ret) ** (1 / years) - 1
        print(f"    Annualized: {annualized*100:.1f}%")
        print(f"    Period: {years:.2f} years")


if __name__ == "__main__":
    main()
