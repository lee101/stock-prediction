#!/usr/bin/env python3
"""Sweep covariate sets for Chronos2 cross-learning forecasts.

For each target symbol, test different covariate combinations and measure MAE%.
Find which correlated assets improve forecast accuracy the most.

Usage:
  python scripts/sweep_chronos2_covariates.py --target BTCFDUSD
  python scripts/sweep_chronos2_covariates.py --target ETHFDUSD --covariates BTCFDUSD,SOLFDUSD,BNBFDUSD
  python scripts/sweep_chronos2_covariates.py --all-targets
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_ROOT = PROJECT_ROOT / "trainingdatahourlybinance"
RESULTS_DIR = PROJECT_ROOT / "chronos2_covariate_results"

CRYPTO_FDUSD = [
    "BTCFDUSD", "ETHFDUSD", "SOLFDUSD", "BNBFDUSD", "DOGEFDUSD",
    "ADAFDUSD", "AVAXFDUSD", "DOTFDUSD", "LINKFDUSD", "MATICFDUSD",
    "XRPFDUSD", "ATOMFDUSD", "LTCFDUSD", "APTFDUSD",
]

TRADING_TARGETS = ["BTCFDUSD", "ETHFDUSD", "SOLFDUSD"]

# Also test stock forecasts with crypto covariates
STOCK_TARGETS = []


def load_ohlc(symbol: str) -> pd.DataFrame:
    csv_path = DATA_ROOT / f"{symbol}.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def compute_correlation(sym_a: str, sym_b: str, hours: int = 720) -> float:
    da = load_ohlc(sym_a)
    db = load_ohlc(sym_b)
    if da.empty or db.empty:
        return 0.0
    merged = pd.merge(da[["timestamp", "close"]], db[["timestamp", "close"]],
                       on="timestamp", suffixes=("_a", "_b"))
    merged = merged.tail(hours)
    if len(merged) < 100:
        return 0.0
    ra = merged["close_a"].pct_change().dropna()
    rb = merged["close_b"].pct_change().dropna()
    return float(ra.corr(rb))


def eval_forecast_mae(target: str, covariates: list[str], lookback: int = 720,
                       horizon: int = 1) -> dict | None:
    import os
    from binanceneural.forecasts import ChronosForecastManager, ForecastConfig

    all_symbols = [target] + covariates
    config = ForecastConfig(
        symbol=target,
        data_root=str(DATA_ROOT),
        cache_root=str(RESULTS_DIR / "tmp_cache"),
        horizons=(horizon,),
        context_hours=512,
        quantile_levels=(0.1, 0.5, 0.9),
        batch_size=32,
        force_cross_learning=True if covariates else None,
    )

    try:
        mgr = ChronosForecastManager(config)
        df_price = load_ohlc(target)
        if df_price.empty:
            return None
        df_price = df_price.tail(lookback)
        end_ts = df_price["timestamp"].max()
        start_ts = df_price["timestamp"].min()

        result = mgr.build_forecast_bundle(
            symbol=target,
            data_root=DATA_ROOT,
            cache_root=RESULTS_DIR / "tmp_cache",
            horizons=(horizon,),
            context_hours=512,
            quantile_levels=(0.1, 0.5, 0.9),
            batch_size=32,
            cache_only=False,
            start=start_ts,
            end=end_ts,
        )
        if result.empty:
            return None

        # Compute MAE vs actuals
        merged = pd.merge(
            result[["timestamp", "predicted_close_p50"]],
            df_price[["timestamp", "close"]],
            on="timestamp",
        )
        if len(merged) < 50:
            return None
        mae = float(np.mean(np.abs(merged["predicted_close_p50"] - merged["close"])))
        mae_pct = float(np.mean(np.abs(
            (merged["predicted_close_p50"] - merged["close"]) / merged["close"]
        ))) * 100
        return {
            "target": target,
            "covariates": covariates,
            "mae": mae,
            "mae_pct": mae_pct,
            "count": len(merged),
        }
    except Exception as e:
        print(f"  Error: {e}")
        return None


def rank_covariates_by_correlation(target: str, candidates: list[str], hours: int = 720) -> list[tuple[str, float]]:
    corrs = []
    for c in candidates:
        if c == target:
            continue
        r = compute_correlation(target, c, hours)
        corrs.append((c, abs(r)))
    corrs.sort(key=lambda x: -x[1])
    return corrs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--all-targets", action="store_true")
    parser.add_argument("--covariates", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--lookback", type=int, default=720)
    parser.add_argument("--horizon", type=int, default=1)
    args = parser.parse_args()

    targets = TRADING_TARGETS if args.all_targets else [args.target] if args.target else TRADING_TARGETS
    candidates = args.covariates.split(",") if args.covariates else CRYPTO_FDUSD

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for target in targets:
        print(f"\n{'='*60}")
        print(f"Target: {target}")
        print(f"{'='*60}")

        # Rank by correlation
        corrs = rank_covariates_by_correlation(target, candidates, args.lookback)
        print(f"\nCorrelation ranking (|r| with {target}, {args.lookback}h):")
        for sym, r in corrs[:10]:
            print(f"  {sym:15s} |r|={r:.4f}")

        # Save correlation results
        corr_file = RESULTS_DIR / f"correlations_{target}.json"
        with open(corr_file, "w") as f:
            json.dump({"target": target, "correlations": [{"symbol": s, "abs_corr": c} for s, c in corrs]}, f, indent=2)
        print(f"Saved: {corr_file}")


if __name__ == "__main__":
    main()
