"""Crypto weekend-only trader — OOS backtest prototype.

Strategy contract (MVP):
  - Enter long at Friday 00:00 UTC close (the daily bar dated Friday).
    This is the last bar before US stock market close Friday 20:00 UTC,
    but we use daily bar close = Friday 00:00 UTC (= Thursday end-of-day
    UTC) which the 24/7 crypto market provides. We actually want to
    BUY at the start of Saturday = Friday close-of-day bar. The binance
    daily bar with timestamp "2025-01-03 00:00:00+00:00" represents the
    Jan 3 UTC day; its close is Jan 3 23:59:59 UTC ≈ start of Jan 4
    (Saturday). That IS Saturday 00:00 UTC — correct entry point.
  - Exit at the close of the Sunday daily bar. That close ≈ Monday
    00:00 UTC, which is ~13.5h before the US stock market opens Monday
    13:30 UTC. Comfortable margin, no conflict with xgb-daily-trader-live.
  - Binary fills, fee_bps=10 per side (20 round-trip).
  - Position: equal-weight across selected symbols, bounded leverage.

Signal (keep SIMPLE, conservative):
  - Use a week-over-week momentum filter: buy only when
    close[Fri] > sma_20(close[Fri]) AND close[Fri] > close[Fri-7d].
    (Market-regime "up-trend" filter.)
  - Size = 1/N per selected symbol (N symbols passing filter), capped
    at max_gross = 1.0 (no leverage).
  - If nothing passes, hold cash.

We score on weekends since 2020-01-01 (in-sample through 2022-06-30) and
report OOS metrics on 2022-07-01 through the last available Sunday.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "binance_spot_1d"

DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT", "DOTUSDT",
]


def load_symbol(symbol: str) -> pd.DataFrame:
    path = DATA_DIR / f"{symbol}.csv"
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    df["date"] = df["timestamp"].dt.tz_convert("UTC").dt.normalize()
    df["dow"] = df["date"].dt.dayofweek  # Mon=0 ... Sat=5, Sun=6
    df = df.sort_values("date").reset_index(drop=True)
    df["close_prev"] = df["close"].shift(1)
    df["ret_1d"] = df["close"] / df["close_prev"] - 1.0
    df["sma_20"] = df["close"].rolling(20, min_periods=20).mean()
    df["close_7d_ago"] = df["close"].shift(7)
    df["mom_7d"] = df["close"] / df["close_7d_ago"] - 1.0
    df["vol_20d"] = df["ret_1d"].rolling(20, min_periods=20).std()
    return df


def build_weekend_panel(symbols: list[str]) -> pd.DataFrame:
    """Build one row per (friday-date, symbol) with features + weekend return.

    Friday daily bar close = Saturday 00:00 UTC = BUY.
    Sunday daily bar close = Monday 00:00 UTC = SELL.
    Weekend return = close[Sun] / close[Fri] - 1.
    """
    rows = []
    for sym in symbols:
        try:
            df = load_symbol(sym)
        except FileNotFoundError:
            continue
        df_by_date = df.set_index("date")
        fridays = df[df["dow"] == 4]  # Friday
        for _, frow in fridays.iterrows():
            fri_date = frow["date"]
            sun_date = fri_date + pd.Timedelta(days=2)
            if sun_date not in df_by_date.index:
                continue
            sun_row = df_by_date.loc[sun_date]
            fri_close = float(frow["close"])
            sun_close = float(sun_row["close"])
            sma_20 = float(frow["sma_20"]) if not pd.isna(frow["sma_20"]) else np.nan
            mom_7d = float(frow["mom_7d"]) if not pd.isna(frow["mom_7d"]) else np.nan
            vol_20d = float(frow["vol_20d"]) if not pd.isna(frow["vol_20d"]) else np.nan
            weekend_ret = sun_close / fri_close - 1.0
            rows.append({
                "symbol": sym,
                "fri_date": fri_date,
                "fri_close": fri_close,
                "sun_close": sun_close,
                "sma_20": sma_20,
                "mom_7d": mom_7d,
                "vol_20d": vol_20d,
                "weekend_ret": weekend_ret,
            })
    return pd.DataFrame(rows)


def apply_signal(
    panel: pd.DataFrame,
    *,
    require_above_sma: bool = True,
    require_mom7_pos: bool = True,
    vol_cap: float | None = 0.12,
    top_k: int | None = None,
) -> pd.DataFrame:
    """Return filtered panel with only the rows we BUY (one row per sym-weekend)."""
    df = panel.copy()
    mask = pd.Series(True, index=df.index)
    if require_above_sma:
        mask &= df["fri_close"] > df["sma_20"]
    if require_mom7_pos:
        mask &= df["mom_7d"] > 0.0
    if vol_cap is not None:
        mask &= df["vol_20d"] <= vol_cap
    # Drop rows missing features
    mask &= df["sma_20"].notna() & df["mom_7d"].notna() & df["vol_20d"].notna()
    df = df[mask]
    if top_k is not None:
        # Rank per friday by mom_7d, keep top_k
        df = df.sort_values(["fri_date", "mom_7d"], ascending=[True, False])
        df = df.groupby("fri_date").head(top_k)
    return df


def weekend_pnl_series(
    picked: pd.DataFrame,
    *,
    fee_bps: float = 10.0,
    max_gross: float = 1.0,
) -> pd.DataFrame:
    """Compute per-weekend PnL, equal-weighted across picks, with binary fills.

    Fee applied twice (entry + exit), on the notional of each leg.
    """
    out = []
    for fri_date, g in picked.groupby("fri_date"):
        n = len(g)
        if n == 0:
            out.append({"fri_date": fri_date, "n_picks": 0, "gross": 0.0,
                        "pnl_fraction": 0.0})
            continue
        # Equal-weight, bounded by max_gross
        w = min(max_gross / n, max_gross)
        gross = w * n  # total notional used
        # Per-pick: gross return * w, minus fee for entry + exit (2 * fee_bps * w)
        gross_ret = float((g["weekend_ret"] * w).sum())
        fee_cost = 2.0 * (fee_bps / 1e4) * gross  # entry + exit, on full gross
        pnl = gross_ret - fee_cost
        out.append({
            "fri_date": fri_date,
            "n_picks": n,
            "gross": gross,
            "pnl_fraction": pnl,
        })
    return pd.DataFrame(out).sort_values("fri_date").reset_index(drop=True)


def add_holdout_rows(weekly: pd.DataFrame, all_fridays: Iterable[pd.Timestamp]) -> pd.DataFrame:
    """Make sure every Friday appears (even cash weekends with pnl=0)."""
    all_df = pd.DataFrame({"fri_date": sorted(set(all_fridays))})
    out = all_df.merge(weekly, on="fri_date", how="left")
    out["n_picks"] = out["n_picks"].fillna(0).astype(int)
    out["gross"] = out["gross"].fillna(0.0)
    out["pnl_fraction"] = out["pnl_fraction"].fillna(0.0)
    return out


def summarize(weekly: pd.DataFrame, name: str) -> dict:
    pnl = weekly["pnl_fraction"].values
    # Weekly-compounded equity curve
    eq = np.cumprod(1.0 + pnl)
    if len(eq) == 0:
        return {"name": name, "n_weekends": 0}
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd = float(dd.min())
    neg_mask = pnl < 0
    # Sortino weekly (no annualization factor; PnL is already % per weekend)
    downside = pnl[pnl < 0]
    if len(downside) > 1:
        downside_dev = float(np.sqrt(np.mean(downside ** 2)))
    else:
        downside_dev = 0.0
    sortino = float(np.mean(pnl) / downside_dev) if downside_dev > 0 else float("inf")
    # Convert weekly -> monthly (≈4.333 weekends/month)
    monthly_pnl = float(np.mean(pnl) * 4.333)
    cum_ret = float(eq[-1] - 1.0)
    return {
        "name": name,
        "n_weekends": int(len(pnl)),
        "n_trade_weekends": int((weekly["n_picks"] > 0).sum()),
        "median_weekly_pnl_pct": float(np.median(pnl) * 100.0),
        "mean_weekly_pnl_pct": float(np.mean(pnl) * 100.0),
        "p10_weekly_pnl_pct": float(np.percentile(pnl, 10) * 100.0),
        "p90_weekly_pnl_pct": float(np.percentile(pnl, 90) * 100.0),
        "worst_weekly_pnl_pct": float(np.min(pnl) * 100.0),
        "best_weekly_pnl_pct": float(np.max(pnl) * 100.0),
        "neg_weekend_rate_pct": float(neg_mask.mean() * 100.0),
        "sortino_weekly": sortino,
        "max_dd_pct": float(max_dd * 100.0),
        "cum_return_pct": float(cum_ret * 100.0),
        "monthly_contribution_pct": float(monthly_pnl * 100.0),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    p.add_argument("--in-sample-end", default="2022-06-30")
    p.add_argument("--fee-bps", type=float, default=10.0)
    p.add_argument("--vol-cap", type=float, default=0.12)
    p.add_argument("--max-gross", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--no-sma-filter", action="store_true")
    p.add_argument("--no-mom-filter", action="store_true")
    p.add_argument("--output-dir", default=str(REPO / "crypto_weekend" / "results"))
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    panel = build_weekend_panel(args.symbols)
    print(f"Loaded {len(panel)} (friday, symbol) rows across {len(args.symbols)} symbols")
    print(f"Date range: {panel['fri_date'].min()} to {panel['fri_date'].max()}")

    all_fridays = sorted(panel["fri_date"].unique())

    for name, filt_kwargs in [
        ("baseline_long_all",
         dict(require_above_sma=False, require_mom7_pos=False, vol_cap=None)),
        ("sma_only",
         dict(require_above_sma=True, require_mom7_pos=False, vol_cap=None)),
        ("sma+mom",
         dict(require_above_sma=True, require_mom7_pos=True, vol_cap=None)),
        ("sma+mom+volcap",
         dict(require_above_sma=True, require_mom7_pos=True, vol_cap=args.vol_cap)),
        ("sma+mom+volcap+top3",
         dict(require_above_sma=True, require_mom7_pos=True, vol_cap=args.vol_cap, top_k=3)),
        ("sma+mom+volcap+top1",
         dict(require_above_sma=True, require_mom7_pos=True, vol_cap=args.vol_cap, top_k=1)),
    ]:
        picked = apply_signal(panel, **filt_kwargs)
        weekly_raw = weekend_pnl_series(picked, fee_bps=args.fee_bps,
                                        max_gross=args.max_gross)
        weekly = add_holdout_rows(weekly_raw, all_fridays)
        # In-sample / OOS split
        split = pd.Timestamp(args.in_sample_end, tz="UTC")
        is_df = weekly[weekly["fri_date"] <= split]
        oos_df = weekly[weekly["fri_date"] > split]
        print(f"\n=== {name} ===")
        summary = {
            "config": {
                "name": name,
                "fee_bps": args.fee_bps,
                "max_gross": args.max_gross,
                **filt_kwargs,
            },
            "all": summarize(weekly, f"{name}_all"),
            "in_sample": summarize(is_df, f"{name}_in_sample"),
            "oos": summarize(oos_df, f"{name}_oos"),
        }
        for slot in ("in_sample", "oos"):
            s = summary[slot]
            print(f"  {slot}: n={s['n_weekends']} trade_wks={s.get('n_trade_weekends', 0)} "
                  f"med={s['median_weekly_pnl_pct']:.3f}% p10={s['p10_weekly_pnl_pct']:.3f}% "
                  f"neg={s['neg_weekend_rate_pct']:.1f}% sortino={s['sortino_weekly']:.2f} "
                  f"maxdd={s['max_dd_pct']:.2f}% cum={s['cum_return_pct']:.2f}% "
                  f"mo={s['monthly_contribution_pct']:.2f}%")
        out_path = out_dir / f"summary_{name}.json"
        out_path.write_text(json.dumps(summary, indent=2, default=str))
        weekly.to_csv(out_dir / f"weekly_{name}.csv", index=False)


if __name__ == "__main__":
    main()
