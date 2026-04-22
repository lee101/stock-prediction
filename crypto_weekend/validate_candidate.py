"""Validate the best tight-filter candidate.
   Config: sma>=SMA20*1.00, mom_vs_sma20>=5%, vol_20d<=0.03, gross=1.0.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from backtest import build_weekend_panel, weekend_pnl_series, add_holdout_rows, summarize
from backtest_tight import tight_filter

def main():
    for sym_set, label in [
        (["BTCUSDT"], "BTC"),
        (["BTCUSDT", "ETHUSDT"], "BTC+ETH"),
        (["BTCUSDT", "ETHUSDT", "SOLUSDT"], "BTC+ETH+SOL"),
    ]:
        panel = build_weekend_panel(sym_set)
        p = tight_filter(panel, sma_mult=1.0, mom7_min=0.0,
                         mom30_min=0.05, vol_max=0.03)
        weekly_raw = weekend_pnl_series(p, fee_bps=10.0, max_gross=1.0)
        all_fridays = sorted(panel["fri_date"].unique())
        weekly = add_holdout_rows(weekly_raw, all_fridays)

        split = pd.Timestamp("2022-06-30", tz="UTC")
        for slot, df in [("ALL", weekly),
                         ("IN-SAMPLE (..2022-06-30)", weekly[weekly["fri_date"] <= split]),
                         ("OOS (>2022-06-30)",        weekly[weekly["fri_date"] > split])]:
            s = summarize(df, label)
            print(f"\n[{label}] {slot}")
            print(f"  n_weeks={s['n_weekends']} trade_wks={s['n_trade_weekends']}")
            print(f"  median_weekly={s['median_weekly_pnl_pct']:+.3f}%  "
                  f"mean_weekly={s['mean_weekly_pnl_pct']:+.3f}%  "
                  f"p10={s['p10_weekly_pnl_pct']:+.3f}%  p90={s['p90_weekly_pnl_pct']:+.3f}%")
            print(f"  neg%={s['neg_weekend_rate_pct']:.1f}  worst={s['worst_weekly_pnl_pct']:+.3f}%  "
                  f"best={s['best_weekly_pnl_pct']:+.3f}%")
            print(f"  max_dd={s['max_dd_pct']:+.3f}%  sortino={s['sortino_weekly']:+.3f}  "
                  f"cum={s['cum_return_pct']:+.2f}%  mo={s['monthly_contribution_pct']:+.3f}%")
            if s["n_trade_weekends"] > 0:
                trade = df[df["n_picks"] > 0]["pnl_fraction"].values * 100
                print(f"  TRADE-ONLY: med={np.median(trade):+.3f}% mean={np.mean(trade):+.3f}% "
                      f"p10={np.percentile(trade, 10):+.3f}% neg%={ (trade<0).mean()*100:.1f}")

    # Show the last 30 OOS trade-weekends for BTC+ETH
    panel = build_weekend_panel(["BTCUSDT", "ETHUSDT"])
    p = tight_filter(panel, sma_mult=1.0, mom7_min=0.0, mom30_min=0.05, vol_max=0.03)
    weekly_raw = weekend_pnl_series(p, fee_bps=10.0, max_gross=1.0)
    weekly_raw = weekly_raw[weekly_raw["fri_date"] > pd.Timestamp("2022-06-30", tz="UTC")]
    weekly_raw = weekly_raw[weekly_raw["n_picks"] > 0].sort_values("fri_date").tail(30)
    print("\nLAST 30 OOS TRADE WEEKENDS (BTC+ETH):")
    for _, r in weekly_raw.iterrows():
        print(f"  {r['fri_date'].date()}  n={int(r['n_picks'])}  pnl={r['pnl_fraction']*100:+.3f}%")


if __name__ == "__main__":
    main()
