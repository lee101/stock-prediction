"""Very conservative conviction filter: only trade when multiple conditions
agree AND size small. Target the 'additive, low-DD' bar."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from backtest import build_weekend_panel, weekend_pnl_series, add_holdout_rows, summarize, REPO

OUT = REPO / "crypto_weekend" / "results_tight"
OUT.mkdir(parents=True, exist_ok=True)


def tight_filter(panel: pd.DataFrame, *,
                 sma_mult: float = 1.00,
                 mom7_min: float = 0.0,
                 mom30_min: float = 0.0,
                 vol_max: float = 0.04) -> pd.DataFrame:
    df = panel.copy()
    # Need mom_30d feature; derive via sma_20 proxy is not available — recompute:
    # Our panel has mom_7d, sma_20, vol_20d. Approx mom_30 via close/sma_20.
    df["mom_vs_sma"] = df["fri_close"] / df["sma_20"] - 1.0
    mask = (df["fri_close"] > df["sma_20"] * sma_mult) & (df["mom_7d"] > mom7_min) & \
           (df["mom_vs_sma"] > mom30_min) & (df["vol_20d"] <= vol_max) & \
           df["sma_20"].notna() & df["mom_7d"].notna() & df["vol_20d"].notna()
    return df[mask]


def main():
    panel = build_weekend_panel(["BTCUSDT", "ETHUSDT"])
    split = pd.Timestamp("2022-06-30", tz="UTC")
    all_fridays = sorted(panel["fri_date"].unique())

    print("Tight conviction grid — BTC+ETH")
    print("=" * 110)
    header = f"{'config':<60s} {'trade':>5s} {'med':>7s} {'p10':>7s} {'neg%':>5s} {'dd%':>7s} {'sortino':>7s} {'mo%':>7s}"
    print(header)
    for sma_m in [1.00, 1.02, 1.05]:
        for mom7 in [0.0, 0.005, 0.01, 0.02]:
            for mom_sma in [0.0, 0.03, 0.05]:
                for vol_max in [0.03, 0.04, 0.06]:
                    for gross in [1.0, 0.5, 0.25]:
                        p = tight_filter(panel, sma_mult=sma_m, mom7_min=mom7,
                                         mom30_min=mom_sma, vol_max=vol_max)
                        weekly_raw = weekend_pnl_series(p, fee_bps=10.0, max_gross=gross)
                        weekly = add_holdout_rows(weekly_raw, all_fridays)
                        oos = weekly[weekly["fri_date"] > split]
                        s = summarize(oos, "t")
                        if s["n_trade_weekends"] < 10:
                            continue
                        # filter for quality — pass if BOTH p10>-3 AND neg<20
                        if s["neg_weekend_rate_pct"] > 20.0:
                            continue
                        if s["p10_weekly_pnl_pct"] < -3.0:
                            continue
                        cfg = f"sm={sma_m} m7={mom7} msm={mom_sma} vm={vol_max} g={gross}"
                        print(f"{cfg:<60s} {s['n_trade_weekends']:>5d} "
                              f"{s['median_weekly_pnl_pct']:>+7.3f} "
                              f"{s['p10_weekly_pnl_pct']:>+7.3f} "
                              f"{s['neg_weekend_rate_pct']:>5.1f} "
                              f"{s['max_dd_pct']:>+7.2f} "
                              f"{s['sortino_weekly']:>+7.3f} "
                              f"{s['monthly_contribution_pct']:>+7.3f}")

if __name__ == "__main__":
    main()
