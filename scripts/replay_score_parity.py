#!/usr/bin/env python3
"""Replay live_trader scoring path against a target session and compare to log.

Verifies that the simulator's score_all_symbols path reproduces what
production logged for a given trading session. If scores match, sim/prod
parity is intact for the scoring stage. If not, there is a divergence in
features, model loading, ensemble blending, or filter order.

Usage
-----
  # Replay yesterday's pre-open session (uses live's Alpaca bar fetch):
  .venv/bin/python scripts/replay_score_parity.py \
      --models analysis/xgbnew_daily/alltrain_ensemble_gpu \
      --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
      --trade-log analysis/xgb_live_trade_log/2026-04-27.jsonl \
      --scored-event-index 1   # 0=first scored event in log, 1=second

  # Replay using the previous ensemble (train_end 2026-04-20):
  .venv/bin/python scripts/replay_score_parity.py \
      --models analysis/xgbnew_daily/alltrain_ensemble_gpu_prev_20260427T150729Z \
      --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
      --trade-log analysis/xgb_live_trade_log/2026-04-27.jsonl \
      --scored-event-index 0
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from xgbnew.live_trader import (  # noqa: E402
    _get_latest_bars,
    score_all_symbols,
)
from xgbnew.model import XGBStockModel  # noqa: E402

logger = logging.getLogger("replay")


def load_models(model_dir: Path) -> list[XGBStockModel]:
    pkls = sorted(model_dir.glob("alltrain_seed*.pkl"))
    if not pkls:
        raise FileNotFoundError(f"no alltrain_seed*.pkl found under {model_dir}")
    models = []
    for p in pkls:
        m = XGBStockModel.load(str(p))
        models.append(m)
    print(f"loaded {len(models)} models from {model_dir}")
    return models


def load_symbols(path: Path) -> list[str]:
    out = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(s)
    return out


def find_scored_event(log_path: Path, scored_index: int) -> dict:
    scored = []
    for line in log_path.read_text().splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        if row.get("event") == "scored":
            scored.append(row)
    if not scored:
        raise ValueError(f"no 'scored' event found in {log_path}")
    if scored_index >= len(scored):
        raise ValueError(
            f"requested scored event index {scored_index}, log has {len(scored)}"
        )
    return scored[scored_index]


def parse_ts_to_now(ts: str) -> datetime:
    """Live trader logs ts in ISO8601 with timezone — return UTC datetime."""
    return pd.Timestamp(ts).to_pydatetime().astimezone(timezone.utc)


def _short(score: float) -> str:
    return f"{score:.4f}"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--models", required=True, type=Path,
                   help="Directory containing alltrain_seed*.pkl")
    p.add_argument("--symbols-file", required=True, type=Path)
    p.add_argument("--trade-log", required=True, type=Path,
                   help="JSONL trade log produced by live_trader for the session")
    p.add_argument("--scored-event-index", type=int, default=0,
                   help="Which 'scored' event in the log to compare against (0-based)")
    p.add_argument("--top-k", type=int, default=20,
                   help="Compare top-K of replay against log top20")
    p.add_argument("--min-dollar-vol", type=float, default=50_000_000.0)
    p.add_argument("--min-vol-20d", type=float, default=0.12)
    p.add_argument("--max-vol-20d", type=float, default=0.0)
    p.add_argument("--max-ret-20d-rank-pct", type=float, default=1.0)
    p.add_argument("--min-ret-5d-rank-pct", type=float, default=0.0)
    p.add_argument("--data-root", type=Path, default=Path("trainingdata"))
    p.add_argument("--use-live-now", action="store_true",
                   help="Use datetime.now(UTC) instead of replaying the log's timestamp")
    p.add_argument("--no-fetch-live-bars", action="store_true",
                   help="Skip Alpaca bar fetch — pure-CSV replay (will fail freshness gate if CSVs stale)")
    p.add_argument("--include-today-bar", action="store_true",
                   help="Include the daily bar dated == today (for intraday rescore replays). "
                        "WARNING: replay sees the FINAL daily bar; live saw a partial mid-session bar.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    models = load_models(args.models)
    symbols = load_symbols(args.symbols_file)
    print(f"loaded {len(symbols)} symbols")

    log_event = find_scored_event(args.trade_log, args.scored_event_index)
    log_ts = log_event["ts"]
    print(f"target log event ts={log_ts}")

    if args.use_live_now:
        now = datetime.now(timezone.utc)
    else:
        now = parse_ts_to_now(log_ts)
    print(f"replay now={now.isoformat()}")

    if args.no_fetch_live_bars:
        live_bars = {}
        print("skipping Alpaca bar fetch — pure-CSV mode")
    else:
        print(f"fetching latest bars from Alpaca for {len(symbols)} symbols ...")
        live_bars = _get_latest_bars(symbols, n_days=20)
        print(f"received bars for {len(live_bars)}/{len(symbols)} symbols")

        # Drop bars dated after `now` so the replay sees the same panel
        # the live trader would have seen at session time. Without this,
        # Alpaca returns fresher bars than were available at the original
        # session, leaking lookahead and changing the feature snapshot.
        cutoff = pd.Timestamp(now).tz_convert("UTC") if pd.Timestamp(now).tzinfo else pd.Timestamp(now, tz="UTC")
        # The live session at session_start time hasn't yet seen today's bar.
        # Specifically, the freshness gate expects the latest *daily* bar to be
        # the previous trading day's close (since market hasn't closed yet).
        # So drop any bar with timestamp >= cutoff_date_midnight_UTC.
        cutoff_floor = cutoff.floor("D")
        if args.include_today_bar:
            # Allow the 2026-04-27 daily bar through (intraday rescore mode)
            keep_through = cutoff_floor + pd.Timedelta(days=1)
        else:
            keep_through = cutoff_floor
        truncated = 0
        empty = 0
        new_live = {}
        for sym, df in live_bars.items():
            ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            mask = ts < keep_through
            kept = df.loc[mask].reset_index(drop=True)
            if len(kept) == 0:
                empty += 1
                continue
            if len(kept) < len(df):
                truncated += 1
            new_live[sym] = kept
        live_bars = new_live
        print(
            f"truncated bars >= {cutoff_floor.date()} for replay realism: "
            f"truncated_syms={truncated}, dropped_syms_now_empty={empty}, "
            f"final_syms={len(live_bars)}"
        )

    print("scoring ...")
    scored = score_all_symbols(
        symbols=symbols,
        data_root=args.data_root,
        model=models,
        live_bars=live_bars,
        min_dollar_vol=args.min_dollar_vol,
        max_spread_bps=30.0,
        min_vol_20d=args.min_vol_20d,
        max_vol_20d=args.max_vol_20d,
        max_ret_20d_rank_pct=args.max_ret_20d_rank_pct,
        min_ret_5d_rank_pct=args.min_ret_5d_rank_pct,
        score_uncertainty_penalty=0.0,
        now=now,
    )

    if scored.empty:
        print("REPLAY produced 0 candidates — likely freshness gate rejected all")
        return 1

    print(f"\nreplay scored {len(scored)} candidates")
    print(f"replay top score: {scored.iloc[0]['symbol']} = {_short(scored.iloc[0]['score'])}")

    # Compare to log top20
    log_top20 = log_event.get("top20", [])
    if not log_top20:
        print("log event has no top20 — skip comparison")
        return 0

    print(f"\nLOG top score: {log_top20[0]['symbol']} = {_short(log_top20[0]['score'])}")

    # Build comparison table
    log_sym_score = {row["symbol"]: float(row["score"]) for row in log_top20}
    replay_sym_score = {
        scored.iloc[i]["symbol"]: float(scored.iloc[i]["score"])
        for i in range(min(args.top_k * 2, len(scored)))
    }

    # All log symbols should appear in replay; compare scores
    print(f"\n{'symbol':<8} {'log_score':>10} {'replay_score':>13} {'delta':>10} {'log_rank':>8} {'rep_rank':>8}")
    print("-" * 64)

    log_top_syms = [r["symbol"] for r in log_top20]
    replay_top_syms = [scored.iloc[i]["symbol"] for i in range(min(len(scored), args.top_k * 2))]

    max_abs_delta = 0.0
    n_compared = 0
    for sym in log_top_syms:
        log_s = log_sym_score[sym]
        rep_s = replay_sym_score.get(sym)
        if rep_s is None:
            print(f"{sym:<8} {log_s:>10.4f} {'MISSING':>13}")
            continue
        log_rank = log_top_syms.index(sym) + 1
        try:
            rep_rank = replay_top_syms.index(sym) + 1
        except ValueError:
            rep_rank = -1
        delta = rep_s - log_s
        max_abs_delta = max(max_abs_delta, abs(delta))
        n_compared += 1
        print(f"{sym:<8} {log_s:>10.4f} {rep_s:>13.4f} {delta:>+10.4f} {log_rank:>8d} {rep_rank:>8d}")

    print("-" * 64)
    print(f"compared {n_compared}/{len(log_top_syms)} log symbols; max |delta| = {max_abs_delta:.4f}")

    # Verdict
    if max_abs_delta < 1e-3:
        print("\nPARITY: BIT-IDENTICAL (max |delta| < 0.001)")
        return 0
    elif max_abs_delta < 1e-2:
        print("\nPARITY: CLOSE (max |delta| < 0.01) — feature drift or float-precision noise")
        return 0
    else:
        print(f"\nPARITY: DIVERGENCE — max |delta| = {max_abs_delta:.4f}")
        print("possible causes: wrong ensemble dir, stale data, feature recipe drift")
        return 2


if __name__ == "__main__":
    sys.exit(main())
