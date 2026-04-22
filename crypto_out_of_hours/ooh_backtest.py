"""Out-of-hours crypto backtester.

Given binance hourly (or daily) bars and a list of out-of-hours sessions
from sessions.py, compute per-session PnL for a long-only strategy.

Signal (keep it SIMPLE, trend-follow + vol cap, mirroring the conservative
weekend MVP from `crypto_weekend/backtest_tight.py`):
  - At session start, for each symbol:
      * fri_close = last hourly close at or before session_start
      * sma_20_daily  = 20-day SMA of daily closes (computed on daily bars)
      * mom_7d  = close/close_7d_ago - 1 (daily)
      * vol_20d  = std of daily returns over 20d
  - Require fri_close > sma_20 * sma_mult AND mom_7d > mom7_min
    AND fri_close/sma_20 - 1 > mom30_min AND vol_20d <= vol_max.
  - Equal-weight across picks, bounded max_gross.
  - Entry fill at session_start hourly close. Exit fill at session_end hourly close.
  - Binary fills, fee=fee_bps per side.

For sessions where hourly data is not available (e.g. post 2026-03-24),
we fall back to daily close approximation: entry = close of day D (session
starts on D at 20:00/21:00 UTC; daily bar close is 00:00 UTC of D+1, which
is ~3h after session_start — small timing error; acceptable for coarse sim).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from sessions import Session, build_sessions

REPO = Path(__file__).resolve().parent.parent
HOURLY_DIR = REPO / "binance_spot_hourly"
DAILY_DIR = REPO / "binance_spot_1d"

DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT", "DOTUSDT",
]


def load_hourly(symbol: str) -> Optional[pd.DataFrame]:
    path = HOURLY_DIR / f"{symbol}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_daily(symbol: str) -> Optional[pd.DataFrame]:
    path = DAILY_DIR / f"{symbol}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["date"] = df["timestamp"].dt.normalize()
    df = df.sort_values("date").reset_index(drop=True)
    df["close_prev"] = df["close"].shift(1)
    df["ret_1d"] = df["close"] / df["close_prev"] - 1.0
    df["sma_20"] = df["close"].rolling(20, min_periods=20).mean()
    df["close_7d_ago"] = df["close"].shift(7)
    df["mom_7d"] = df["close"] / df["close_7d_ago"] - 1.0
    df["vol_20d"] = df["ret_1d"].rolling(20, min_periods=20).std()
    return df


def _price_at_or_before(hourly: pd.DataFrame, ts: pd.Timestamp) -> Optional[float]:
    """Return the close of the hourly bar with timestamp <= ts (latest)."""
    sub = hourly[hourly["timestamp"] <= ts]
    if sub.empty:
        return None
    return float(sub.iloc[-1]["close"])


def _daily_close_for(daily: pd.DataFrame, date: pd.Timestamp) -> Optional[float]:
    """Return the close of the daily bar on `date`."""
    rows = daily[daily["date"] == date.normalize()]
    if rows.empty:
        return None
    return float(rows.iloc[0]["close"])


def get_entry_exit_prices(
    hourly: Optional[pd.DataFrame],
    daily: pd.DataFrame,
    session: Session,
) -> tuple[Optional[float], Optional[float], str]:
    """Return (entry_price, exit_price, source). source='hourly' or 'daily'."""
    if hourly is not None and not hourly.empty:
        latest_hourly = hourly["timestamp"].max()
        # Only use hourly if BOTH entry and exit timestamps are covered
        if latest_hourly >= session.end:
            entry = _price_at_or_before(hourly, session.start)
            exit_ = _price_at_or_before(hourly, session.end)
            if entry is not None and exit_ is not None:
                return entry, exit_, "hourly"
    # Fallback: daily bars. Entry ≈ close of the day on which session STARTS.
    # session.start is 20:00 or 21:00 UTC on day D → the daily bar's close is at
    # 00:00 UTC of day D+1, which is 3-4h AFTER session.start. Imperfect but
    # acceptable for the post-hourly tail.
    # Exit ≈ close of day PRIOR to session.end (session.end is 13:30/14:30 UTC
    # on day D+1 → the daily bar closing at 00:00 UTC on D+1 is 13-14h BEFORE.)
    entry_date = session.start.normalize()
    # For exit: session ends 13:30 UTC on day D+k — daily bar 00:00 UTC of D+k
    # has closed ≈13h before, but more accurate: use the close of day D+k-1 is
    # the same 00:00 UTC timestamp (since bar at 00:00 D+k represents D+k, close
    # = 00:00 D+(k+1))... actually binance daily bar with timestamp 00:00 UTC
    # D+k is the bar FOR day D+k (closing at 00:00 D+(k+1)). So to approximate
    # exit at 13:30 D+k, use the bar for D+k's close — but that's the close of
    # the whole day, ~10h too late.
    # Best daily-fallback: entry = close of day D-1 (closed at start of D,
    # i.e. ~20h before session.start, too early), exit = close of day before
    # session end (closes 13h before session end).
    # This fallback is coarse; warn users. We align to "close of day BEFORE
    # session_start" for entry and "close of day session_end-1" for exit.
    # The most reasonable choice: use the bar whose END is CLOSEST to the
    # session boundary without looking ahead.
    # Entry: bar whose close (timestamp + 1d) <= session.start, take the latest.
    # Exit: bar whose close (timestamp + 1d) <= session.end, take the latest.
    # This is equivalent to: entry_bar.timestamp = session.start.normalize() - 1d
    # (its close = session.start.normalize() = ~20-21h before session.start → stale).
    # Alternative: entry_bar.timestamp = session.start.normalize() (its close is
    # the NEXT day 00:00 UTC, which is 3-4h AFTER session.start — lookahead!).
    # We must not peek. Use entry_bar.timestamp = session.start.normalize() - 1d.
    entry_daily_date = entry_date - pd.Timedelta(days=1)
    exit_daily_date = session.end.normalize() - pd.Timedelta(days=1)
    entry = _daily_close_for(daily, entry_daily_date)
    exit_ = _daily_close_for(daily, exit_daily_date)
    if entry is None or exit_ is None:
        return None, None, "missing"
    return entry, exit_, "daily"


def build_session_panel(
    symbols: list[str],
    sessions: list[Session],
) -> pd.DataFrame:
    """One row per (session_start, symbol) with entry/exit prices and features."""
    rows = []
    hourlies = {s: load_hourly(s) for s in symbols}
    dailies = {s: load_daily(s) for s in symbols}
    for session in sessions:
        for sym in symbols:
            daily = dailies.get(sym)
            if daily is None:
                continue
            # Feature snapshot as of session.start — use latest daily bar with
            # timestamp < session.start (no lookahead).
            feat_cutoff = session.start.normalize() - pd.Timedelta(days=1)
            feat_row = daily[daily["date"] == feat_cutoff]
            if feat_row.empty:
                continue
            feat_row = feat_row.iloc[0]
            sma_20 = feat_row["sma_20"]
            mom_7d = feat_row["mom_7d"]
            vol_20d = feat_row["vol_20d"]
            if pd.isna(sma_20) or pd.isna(mom_7d) or pd.isna(vol_20d):
                continue
            hourly = hourlies.get(sym)
            entry, exit_, source = get_entry_exit_prices(hourly, daily, session)
            if entry is None or exit_ is None:
                continue
            session_ret = exit_ / entry - 1.0
            rows.append({
                "session_start": session.start,
                "session_end": session.end,
                "kind": session.kind,
                "duration_hours": session.duration_hours,
                "symbol": sym,
                "entry_price": entry,
                "exit_price": exit_,
                "session_ret": session_ret,
                "sma_20": float(sma_20),
                "mom_7d": float(mom_7d),
                "vol_20d": float(vol_20d),
                "price_source": source,
            })
    return pd.DataFrame(rows)


def apply_signal(
    panel: pd.DataFrame,
    *,
    sma_mult: float = 1.0,
    mom7_min: float = 0.0,
    mom_vs_sma_min: float = 0.0,
    vol_max: float = 0.04,
    top_k: Optional[int] = None,
    kinds: Optional[set] = None,
) -> pd.DataFrame:
    df = panel.copy()
    if kinds is not None:
        df = df[df["kind"].isin(kinds)]
    df["mom_vs_sma"] = df["entry_price"] / df["sma_20"] - 1.0
    mask = (df["entry_price"] > df["sma_20"] * sma_mult) & \
           (df["mom_7d"] > mom7_min) & \
           (df["mom_vs_sma"] > mom_vs_sma_min) & \
           (df["vol_20d"] <= vol_max)
    df = df[mask]
    if top_k is not None:
        df = df.sort_values(["session_start", "mom_7d"], ascending=[True, False])
        df = df.groupby("session_start").head(top_k)
    return df


def session_pnl_series(
    picked: pd.DataFrame,
    all_sessions: list[Session],
    *,
    fee_bps: float = 10.0,
    max_gross: float = 1.0,
) -> pd.DataFrame:
    """One row per session with PnL fraction. Non-picked sessions = 0."""
    pick_map = {s.start: s for s in all_sessions}
    rows = []
    picked_by_session = picked.groupby("session_start")
    picked_starts = set(picked_by_session.groups.keys())
    for s in all_sessions:
        if s.start not in picked_starts:
            rows.append({
                "session_start": s.start,
                "session_end": s.end,
                "kind": s.kind,
                "n_picks": 0,
                "gross": 0.0,
                "pnl_fraction": 0.0,
            })
            continue
        g = picked_by_session.get_group(s.start)
        n = len(g)
        w = min(max_gross / n, max_gross)
        gross = w * n
        gross_ret = float((g["session_ret"] * w).sum())
        fee_cost = 2.0 * (fee_bps / 1e4) * gross
        pnl = gross_ret - fee_cost
        rows.append({
            "session_start": s.start,
            "session_end": s.end,
            "kind": s.kind,
            "n_picks": n,
            "gross": gross,
            "pnl_fraction": pnl,
        })
    return pd.DataFrame(rows).sort_values("session_start").reset_index(drop=True)


def summarize(series: pd.DataFrame, name: str, *, periods_per_month: float) -> dict:
    pnl = series["pnl_fraction"].values.astype(float)
    if len(pnl) == 0:
        return {"name": name, "n_sessions": 0}
    eq = np.cumprod(1.0 + pnl)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd = float(dd.min())
    neg_mask = pnl < 0
    downside = pnl[pnl < 0]
    if len(downside) > 1:
        downside_dev = float(np.sqrt(np.mean(downside ** 2)))
    else:
        downside_dev = 0.0
    sortino = float(np.mean(pnl) / downside_dev) if downside_dev > 0 else float("inf")
    monthly_pnl = float(np.mean(pnl) * periods_per_month)
    cum_ret = float(eq[-1] - 1.0)
    return {
        "name": name,
        "n_sessions": int(len(pnl)),
        "n_trade_sessions": int((series["n_picks"] > 0).sum()),
        "median_session_pnl_pct": float(np.median(pnl) * 100.0),
        "mean_session_pnl_pct": float(np.mean(pnl) * 100.0),
        "p10_session_pnl_pct": float(np.percentile(pnl, 10) * 100.0),
        "p90_session_pnl_pct": float(np.percentile(pnl, 90) * 100.0),
        "worst_session_pnl_pct": float(np.min(pnl) * 100.0),
        "best_session_pnl_pct": float(np.max(pnl) * 100.0),
        "neg_session_rate_pct": float(neg_mask.mean() * 100.0),
        "sortino": sortino,
        "max_dd_pct": float(max_dd * 100.0),
        "cum_return_pct": float(cum_ret * 100.0),
        "monthly_contribution_pct": float(monthly_pnl * 100.0),
    }


def run_config(
    panel: pd.DataFrame,
    sessions: list[Session],
    *,
    cfg: dict,
    fee_bps: float,
    max_gross: float,
    periods_per_month: float,
) -> dict:
    name = cfg.get("name", "cfg")
    cfg_filter = {k: v for k, v in cfg.items() if k != "name"}
    picked = apply_signal(panel, **cfg_filter)
    series = session_pnl_series(picked, sessions, fee_bps=fee_bps, max_gross=max_gross)
    s = summarize(series, name, periods_per_month=periods_per_month)
    s["config"] = {**cfg, "fee_bps": fee_bps, "max_gross": max_gross}
    s["series"] = series  # carry through for plotting / aggregating
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    p.add_argument("--start", default="2025-12-17")
    p.add_argument("--end", default="2026-04-15")
    p.add_argument("--fee-bps", type=float, default=10.0)
    p.add_argument("--max-gross", type=float, default=1.0)
    p.add_argument("--output-dir", default=str(REPO / "crypto_out_of_hours" / "results"))
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sessions = build_sessions(args.start, args.end)
    print(f"Sessions in window [{args.start}, {args.end}): {len(sessions)}")
    total_hours = sum(s.duration_hours for s in sessions)
    n_days = (pd.Timestamp(args.end, tz="UTC") - pd.Timestamp(args.start, tz="UTC")).days
    # periods_per_month: if N sessions span ~n_days days, sessions per month is
    # N * (30.0 / n_days).
    periods_per_month = len(sessions) * (30.0 / n_days)
    print(f"Total hours: {total_hours:.1f}h, window days: {n_days}, "
          f"sessions/month ≈ {periods_per_month:.2f}")

    panel = build_session_panel(args.symbols, sessions)
    print(f"Panel rows: {len(panel)}  "
          f"(per symbol: {dict(panel.groupby('symbol').size())})")
    print(f"Price source breakdown: {dict(panel.groupby('price_source').size())}")

    configs = [
        dict(name="baseline_sma_mom_vol",
             sma_mult=1.0, mom7_min=0.0, mom_vs_sma_min=0.0, vol_max=0.04),
        dict(name="tight_5pct_sma_vol3",
             sma_mult=1.0, mom7_min=0.0, mom_vs_sma_min=0.05, vol_max=0.03),
        dict(name="tight_5pct_sma_vol3_top3",
             sma_mult=1.0, mom7_min=0.0, mom_vs_sma_min=0.05, vol_max=0.03, top_k=3),
        dict(name="tight_5pct_sma_vol3_top1",
             sma_mult=1.0, mom7_min=0.0, mom_vs_sma_min=0.05, vol_max=0.03, top_k=1),
        dict(name="medium_2pct_sma_vol4",
             sma_mult=1.0, mom7_min=0.0, mom_vs_sma_min=0.02, vol_max=0.04),
        dict(name="medium_2pct_sma_vol4_top3",
             sma_mult=1.0, mom7_min=0.0, mom_vs_sma_min=0.02, vol_max=0.04, top_k=3),
        dict(name="weekend_only_replica",
             sma_mult=1.0, mom7_min=0.0, mom_vs_sma_min=0.05, vol_max=0.03,
             kinds={"weekend"}),
        dict(name="weekday_only_tight",
             sma_mult=1.0, mom7_min=0.0, mom_vs_sma_min=0.05, vol_max=0.03,
             kinds={"weekday"}),
        dict(name="weekday+holiday_tight",
             sma_mult=1.0, mom7_min=0.0, mom_vs_sma_min=0.05, vol_max=0.03,
             kinds={"weekday", "holiday"}),
        dict(name="wider_weekday_momonly",
             sma_mult=1.0, mom7_min=0.0, mom_vs_sma_min=0.0, vol_max=0.05,
             kinds={"weekday"}),
    ]

    results = {}
    header = (f"{'name':<35s} {'trade':>6s} {'med%':>7s} {'mean%':>7s} "
              f"{'p10%':>7s} {'neg%':>5s} {'dd%':>7s} {'sortino':>7s} "
              f"{'mo%':>7s} {'cum%':>7s}")
    print("\n" + header)
    print("=" * len(header))
    for cfg in configs:
        res = run_config(
            panel, sessions,
            cfg=cfg, fee_bps=args.fee_bps, max_gross=args.max_gross,
            periods_per_month=periods_per_month,
        )
        name = cfg["name"]
        results[name] = res
        print(f"{name:<35s} {res['n_trade_sessions']:>6d} "
              f"{res['median_session_pnl_pct']:>+7.3f} "
              f"{res['mean_session_pnl_pct']:>+7.3f} "
              f"{res['p10_session_pnl_pct']:>+7.3f} "
              f"{res['neg_session_rate_pct']:>5.1f} "
              f"{res['max_dd_pct']:>+7.2f} "
              f"{res['sortino']:>+7.3f} "
              f"{res['monthly_contribution_pct']:>+7.3f} "
              f"{res['cum_return_pct']:>+7.2f}")

    # Save results
    summary_out = {}
    for name, r in results.items():
        # Strip the non-serializable series
        r_copy = {k: v for k, v in r.items() if k != "series"}
        summary_out[name] = r_copy
        r["series"].to_csv(out_dir / f"series_{name}.csv", index=False)
    summary_out["_meta"] = {
        "start": args.start,
        "end": args.end,
        "symbols": args.symbols,
        "fee_bps": args.fee_bps,
        "max_gross": args.max_gross,
        "n_sessions": len(sessions),
        "periods_per_month": periods_per_month,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary_out, indent=2, default=str))
    print(f"\nResults written to {out_dir}")


if __name__ == "__main__":
    main()
