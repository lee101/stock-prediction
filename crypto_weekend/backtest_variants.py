"""Additional variants:
 - BTC-only, ETH-only baselines
 - Mean-reversion angle: buy when Fri close *down* week
 - Selective regime: up-trend AND low vol AND top1 conviction
 - Exit Saturday (skip Sun tail) vs Sunday close
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import (
    build_weekend_panel,
    apply_signal,
    weekend_pnl_series,
    add_holdout_rows,
    summarize,
    load_symbol,
    REPO,
)

OUT = REPO / "crypto_weekend" / "results_variants"
OUT.mkdir(parents=True, exist_ok=True)


def build_panel_custom_exit(symbols: list[str], exit_dow: int = 6) -> pd.DataFrame:
    """Like build_weekend_panel but exit bar is configurable.

    exit_dow=6 -> Sunday daily bar close (default; Sun 23:59 UTC = Mon 00:00).
    exit_dow=5 -> Saturday daily bar close (Sat 23:59 UTC = Sun 00:00). Shorter hold.
    """
    rows = []
    for sym in symbols:
        try:
            df = load_symbol(sym)
        except FileNotFoundError:
            continue
        df_by_date = df.set_index("date")
        fridays = df[df["dow"] == 4]
        days_to_exit = exit_dow - 4  # Fri=4, Sat=5 -> 1, Sun=6 -> 2
        for _, frow in fridays.iterrows():
            fri_date = frow["date"]
            exit_date = fri_date + pd.Timedelta(days=days_to_exit)
            if exit_date not in df_by_date.index:
                continue
            exit_row = df_by_date.loc[exit_date]
            fri_close = float(frow["close"])
            exit_close = float(exit_row["close"])
            rows.append({
                "symbol": sym,
                "fri_date": fri_date,
                "fri_close": fri_close,
                "sun_close": exit_close,
                "sma_20": frow["sma_20"],
                "mom_7d": frow["mom_7d"],
                "vol_20d": frow["vol_20d"],
                "weekend_ret": exit_close / fri_close - 1.0,
            })
    return pd.DataFrame(rows)


def run(name, panel, *, filt_kwargs, fee_bps=10.0, max_gross=1.0,
        is_end="2022-06-30"):
    picked = apply_signal(panel, **filt_kwargs)
    weekly_raw = weekend_pnl_series(picked, fee_bps=fee_bps, max_gross=max_gross)
    all_fridays = sorted(panel["fri_date"].unique())
    weekly = add_holdout_rows(weekly_raw, all_fridays)
    split = pd.Timestamp(is_end, tz="UTC")
    oos_df = weekly[weekly["fri_date"] > split]
    s = summarize(oos_df, f"{name}_oos")
    print(f"  {name:40s}: n={s['n_weekends']:3d} trade={s['n_trade_weekends']:3d} "
          f"med={s['median_weekly_pnl_pct']:+.3f} p10={s['p10_weekly_pnl_pct']:+.3f} "
          f"neg={s['neg_weekend_rate_pct']:4.1f}% dd={s['max_dd_pct']:+6.2f}% "
          f"sortino={s['sortino_weekly']:+.3f} mo={s['monthly_contribution_pct']:+.2f}%")
    return name, s


DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
                   "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT", "DOTUSDT"]


def main():
    print("=" * 90)
    print("VARIANT A: different symbol subsets, long_all")
    print("=" * 90)
    results = []
    for syms, tag in [
        (["BTCUSDT"], "BTC_only"),
        (["ETHUSDT"], "ETH_only"),
        (["BTCUSDT", "ETHUSDT"], "BTC+ETH"),
        (["BTCUSDT", "ETHUSDT", "SOLUSDT"], "BTC+ETH+SOL"),
        (DEFAULT_SYMBOLS, "all_10"),
    ]:
        panel = build_weekend_panel(syms)
        results.append(run(f"{tag}_longall",
                           panel,
                           filt_kwargs=dict(require_above_sma=False,
                                            require_mom7_pos=False,
                                            vol_cap=None)))

    print()
    print("=" * 90)
    print("VARIANT B: BTC+ETH with SMA/momentum/vol filters")
    print("=" * 90)
    panel2 = build_weekend_panel(["BTCUSDT", "ETHUSDT"])
    for cfg_name, cfg in [
        ("sma_only", dict(require_above_sma=True, require_mom7_pos=False, vol_cap=None)),
        ("sma+mom", dict(require_above_sma=True, require_mom7_pos=True, vol_cap=None)),
        ("sma+mom+volcap", dict(require_above_sma=True, require_mom7_pos=True, vol_cap=0.12)),
        ("sma+mom+volcap_low", dict(require_above_sma=True, require_mom7_pos=True, vol_cap=0.08)),
        ("sma_only_top1", dict(require_above_sma=True, require_mom7_pos=False, vol_cap=None, top_k=1)),
    ]:
        results.append(run(f"BTC+ETH_{cfg_name}", panel2, filt_kwargs=cfg))

    print()
    print("=" * 90)
    print("VARIANT C: MEAN-REVERSION (buy when Fri below 20d SMA)")
    print("=" * 90)
    # Synthesize via apply_signal contract: invert sma filter manually
    for syms, tag in [(["BTCUSDT"], "BTC"), (["BTCUSDT","ETHUSDT"], "BTC+ETH")]:
        panel = build_weekend_panel(syms)
        p = panel[(panel["fri_close"] < panel["sma_20"]) & panel["sma_20"].notna()]
        weekly_raw = weekend_pnl_series(p, fee_bps=10.0, max_gross=1.0)
        all_fridays = sorted(panel["fri_date"].unique())
        weekly = add_holdout_rows(weekly_raw, all_fridays)
        split = pd.Timestamp("2022-06-30", tz="UTC")
        oos = weekly[weekly["fri_date"] > split]
        s = summarize(oos, f"mean_rev_{tag}")
        print(f"  mean_rev_{tag:30s}: n={s['n_weekends']:3d} trade={s['n_trade_weekends']:3d} "
              f"med={s['median_weekly_pnl_pct']:+.3f} p10={s['p10_weekly_pnl_pct']:+.3f} "
              f"neg={s['neg_weekend_rate_pct']:4.1f}% dd={s['max_dd_pct']:+6.2f}% "
              f"sortino={s['sortino_weekly']:+.3f} mo={s['monthly_contribution_pct']:+.2f}%")
        results.append((f"mean_rev_{tag}", s))

    print()
    print("=" * 90)
    print("VARIANT D: exit Saturday close (shorter hold)")
    print("=" * 90)
    for syms, tag in [(["BTCUSDT"], "BTC"), (["BTCUSDT","ETHUSDT"], "BTC+ETH")]:
        panel = build_panel_custom_exit(syms, exit_dow=5)  # Sat close
        results.append(run(f"satonly_{tag}_longall",
                           panel,
                           filt_kwargs=dict(require_above_sma=False,
                                            require_mom7_pos=False,
                                            vol_cap=None)))
        results.append(run(f"satonly_{tag}_sma+mom",
                           panel,
                           filt_kwargs=dict(require_above_sma=True,
                                            require_mom7_pos=True,
                                            vol_cap=None)))

    print()
    print("=" * 90)
    print("VARIANT E: conservative half-sized long_all on BTC+ETH")
    print("=" * 90)
    panel = build_weekend_panel(["BTCUSDT", "ETHUSDT"])
    for max_g in [0.5, 0.33, 0.25]:
        # long_all with half gross
        results.append(run(f"BTC+ETH_longall_gross{max_g}",
                           panel,
                           filt_kwargs=dict(require_above_sma=False,
                                            require_mom7_pos=False,
                                            vol_cap=None),
                           max_gross=max_g))

    (OUT / "variants_summary.json").write_text(
        json.dumps([{"name": n, **s} for n, s in results], indent=2, default=str))


if __name__ == "__main__":
    main()
