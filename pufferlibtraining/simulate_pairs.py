#!/usr/bin/env python3
"""
Run a lightweight simulation across all unordered pairs from a symbol list
using the PufferLib multi-asset environment with leverage/fee rules.

This is a mechanics/PNL sanity runner (not RL). It loads CSVs from
``--trainingdata-dir`` and evaluates an equal-weight baseline under the
requested trade timing (open or close), per-asset base fees, transaction costs,
and leverage constraints (4x intraday, 2x overnight by default).
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import numpy as np

from pufferlibtraining.envs.stock_env import StockTradingEnv
from stockagent.constants import DEFAULT_SYMBOLS


def _parse_symbols(arg: str | None) -> List[str]:
    if not arg:
        return list(DEFAULT_SYMBOLS)
    return [s.strip().upper() for s in arg.split(",") if s.strip()]


def _load_frame(data_root: Path, symbol: str) -> pd.DataFrame:
    # Prefer exact match; fall back to substring search
    candidates = list((data_root).glob(f"**/*{symbol}*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV found for {symbol} under {data_root}")
    path = sorted(candidates, key=lambda p: len(p.name))[0]
    return pd.read_csv(path)


def _align_frames(frames: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    # Minimal alignment: require 'date' or 'timestamp' and inner join on it.
    normalised: Dict[str, pd.DataFrame] = {}
    for sym, df in frames.items():
        d = df.copy()
        cols = {c.lower(): c for c in d.columns}
        if "date" in cols:
            d["date"] = pd.to_datetime(d[cols["date"]])
        elif "timestamp" in cols:
            d["date"] = pd.to_datetime(d[cols["timestamp"]])
        else:
            raise ValueError(f"{sym} CSV missing date/timestamp column")
        # Keep common OHLCV columns when present
        kept = [c for c in d.columns if c.lower() in {"date", "open", "high", "low", "close", "volume"}]
        if "open" not in {c.lower() for c in kept} or "close" not in {c.lower() for c in kept}:
            raise ValueError(f"{sym} CSV must contain at least open/close columns")
        normalised[sym] = d[kept].sort_values("date").reset_index(drop=True)
    # Intersect dates
    common = None
    for df in normalised.values():
        idx = pd.Index(df["date"])  # naive tz
        common = idx if common is None else common.intersection(idx)
    if common is None or len(common) < 3:
        raise RuntimeError("No overlapping dates across provided frames")
    common = pd.Index(sorted(common))
    aligned = {sym: df[df["date"].isin(common)].reset_index(drop=True) for sym, df in normalised.items()}
    return aligned


def simulate_pair(
    data_root: Path,
    pair: Tuple[str, str],
    trade_timing: str,
    risk_scale: float,
    transaction_cost_bps: float,
    spread_bps: float,
) -> Dict[str, float]:
    frames = {sym: _load_frame(data_root, sym) for sym in pair}
    frames = _align_frames(frames)
    env = StockTradingEnv(
        frames,
        window_size=30,
        initial_balance=100_000.0,
        transaction_cost_bps=transaction_cost_bps,
        spread_bps=spread_bps,
        max_intraday_leverage=4.0,
        max_overnight_leverage=2.0,
        trade_timing=trade_timing,
        risk_scale=risk_scale,
    )
    obs, _ = env.reset()
    done = False
    # Equal-weight baseline in raw action space (tanh squashes)
    raw_equal = np.array([0.5, 0.5], dtype=np.float32)
    while not done:
        obs, reward, term, trunc, _info = env.step(raw_equal)
        done = term or trunc
    metrics = env.get_metrics()
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Simulate all unordered pairs with PufferLib env mechanics.")
    ap.add_argument("--symbols", type=str, default=",")
    ap.add_argument("--trainingdata-dir", type=str, default="trainingdata/train")
    ap.add_argument("--trade-timing", choices=["open", "close"], default="open")
    ap.add_argument("--risk-scale", type=float, default=1.0)
    ap.add_argument("--transaction-cost-bps", type=float, default=10.0)
    ap.add_argument("--spread-bps", type=float, default=1.0)
    ap.add_argument("--output", type=str, default="pufferlibtraining/models/simulations/pairs_summary.csv")
    args = ap.parse_args()

    symbols = _parse_symbols(args.symbols)
    data_root = Path(args.trainingdata_dir).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, float]] = []
    for a, b in itertools.combinations(symbols, 2):
        try:
            metrics = simulate_pair(
                data_root,
                (a, b),
                trade_timing=args.trade_timing,
                risk_scale=float(args.risk_scale),
                transaction_cost_bps=float(args.transaction_cost_bps),
                spread_bps=float(args.spread_bps),
            )
            rows.append({"pair": f"{a}_{b}", **metrics})
        except Exception as exc:  # continue on missing data
            rows.append({"pair": f"{a}_{b}", "error": str(exc)})

    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Pair simulation summary written to {output_path}")


if __name__ == "__main__":
    main()
