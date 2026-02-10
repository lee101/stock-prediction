from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from newnanoalpacahourlyexp.marketsimulator.hourly_trader import (
    HourlyTraderMarketSimulator,
    HourlyTraderSimulationConfig,
)


ROOT = Path(__file__).resolve().parent
BARS_CSV = ROOT / "bars.csv"
ACTIONS_CSV = ROOT / "actions.csv"
OUT_CSV = ROOT / "tune_results.csv"


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    values = equity.to_numpy(dtype=np.float64)
    if values.size < 2:
        return 0.0
    running_max = np.maximum.accumulate(values)
    # Guard against divide-by-zero.
    running_max = np.where(running_max > 0, running_max, 1.0)
    dd = (values - running_max) / running_max
    return float(abs(np.min(dd)))


def main() -> None:
    if not BARS_CSV.exists() or not ACTIONS_CSV.exists():
        raise FileNotFoundError("Expected bars.csv and actions.csv in this directory.")

    bars = pd.read_csv(BARS_CSV)
    actions = pd.read_csv(ACTIONS_CSV)

    allocation_pcts = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    intensity_scales = [0.5, 1.0, 1.5, 2.0]
    min_gap_pcts = [0.001, 0.002, 0.003, 0.005]

    rows = []
    total = len(allocation_pcts) * len(intensity_scales) * len(min_gap_pcts)
    idx = 0

    for alloc in allocation_pcts:
        for intensity in intensity_scales:
            for gap in min_gap_pcts:
                idx += 1
                cfg = HourlyTraderSimulationConfig(
                    initial_cash=10_000.0,
                    allocation_usd=None,
                    allocation_pct=float(alloc),
                    allocation_mode="portfolio",
                    intensity_scale=float(intensity),
                    price_offset_pct=0.0,
                    min_gap_pct=float(gap),
                    decision_lag_bars=1,
                    enforce_market_hours=True,
                    keep_similar_orders=True,
                )
                sim = HourlyTraderMarketSimulator(cfg)
                result = sim.run(bars, actions)
                metrics = result.metrics
                equity = result.equity_curve

                rows.append(
                    {
                        "allocation_pct": float(alloc),
                        "intensity_scale": float(intensity),
                        "min_gap_pct": float(gap),
                        "total_return": float(metrics.get("total_return", 0.0)),
                        "sortino": float(metrics.get("sortino", 0.0)),
                        "mean_hourly_return": float(metrics.get("mean_hourly_return", 0.0)),
                        "max_drawdown": _max_drawdown(equity),
                        "fills": int(len(result.fills)),
                        "open_orders_end": int(len(result.open_orders)),
                        "positions_end": int(len(result.final_positions)),
                    }
                )

                if idx % 10 == 0 or idx == total:
                    print(f"[{idx}/{total}] alloc={alloc} intensity={intensity} gap={gap} ret={rows[-1]['total_return']:.4f}")

    rows.sort(key=lambda r: (r["total_return"], r["sortino"]), reverse=True)
    print("Top 10 configs by total_return:")
    for r in rows[:10]:
        print(r)

    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
