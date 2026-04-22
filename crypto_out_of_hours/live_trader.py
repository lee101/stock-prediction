"""Out-of-hours crypto live trader (additive with xgb-daily-trader-live).

SAFETY MODEL:
  - This trader must NEVER hold a crypto position while NYSE is open.
  - At every poll tick: query NYSE state via pandas_market_calendars.
      * If NYSE is OPEN now → ensure flat.
      * If NYSE is CLOSED and we have no position → maybe enter (signal gate).
      * If NYSE is CLOSED and we have a position → hold.
      * If NYSE OPEN in < 30min and we have a position → close (pre-open exit).
  - Separate Alpaca account (ALPACA_ACCOUNT_NAME=crypto-out-of-hours). No
    overlap with xgb-daily-trader-live lock.
  - Default gross_leverage=1.0, will respect `--max-gross` flag.

This file is a SKELETON — LIVE wiring needs:
  1. alpaca_wrapper crypto endpoints (already present via alpaca_wrapper).
  2. singleton lock via `src/alpaca_singleton.py` (if trading real money).
  3. ALPACA_ACCOUNT_NAME env var and API key file.
  4. Systemd / supervisor unit in deployments/crypto-out-of-hours-live/.

DO NOT RUN LIVE without:
  - Explicit user approval.
  - New singleton lock registered in LIVE_WRITER_UNITS (HARD RULE #2).
  - Separate Alpaca account / subaccount so death-spiral guard state is isolated.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "crypto_out_of_hours"))

from sessions import build_sessions  # noqa: E402
from ooh_backtest import (  # noqa: E402
    build_session_panel, apply_signal, DEFAULT_SYMBOLS,
)
from find_combined import combine_picks  # noqa: E402

log = logging.getLogger(__name__)

# Final validated config (see final_report.py):
WEEKEND_CFG = dict(sma_mult=1.0, mom7_min=0.0, mom_vs_sma_min=0.0,
                    vol_max=0.03, top_k=1, kinds={"weekend"})
WEEKDAY_CFG = dict(sma_mult=1.05, mom7_min=0.0, mom_vs_sma_min=0.0,
                   vol_max=0.02, top_k=2, kinds={"weekday", "holiday"})


@dataclass
class SessionState:
    start: pd.Timestamp
    end: pd.Timestamp
    kind: str
    symbols: list[str]
    weights: dict[str, float]
    entered_at: Optional[pd.Timestamp] = None
    entry_prices: Optional[dict[str, float]] = None


def compute_session_picks(
    symbols: list[str], session_start_utc: pd.Timestamp, session_end_utc: pd.Timestamp,
    kind: str, weekday_enabled: bool,
) -> tuple[list[str], dict[str, float]]:
    """Compute which symbols to buy at session start.

    Returns (pick_symbols, weight_map) where weight_map[symbol] sums to <= 1.0.
    """
    from sessions import Session
    session = Session(start=session_start_utc, end=session_end_utc,
                       kind=kind, duration_hours=(session_end_utc - session_start_utc).total_seconds() / 3600.0)
    # Build one-session panel
    panel = build_session_panel(symbols, [session])
    if panel.empty:
        return [], {}
    # Apply configs
    if kind == "weekend":
        picks = apply_signal(panel, **WEEKEND_CFG)
    elif weekday_enabled:
        # Weekday or holiday → weekday filter
        picks_w = apply_signal(panel, **WEEKEND_CFG)   # never fires on weekday kinds
        picks_d = apply_signal(panel, **WEEKDAY_CFG)
        picks = combine_picks(picks_w, picks_d)
    else:
        return [], {}  # weekday disabled
    if picks.empty:
        return [], {}
    chosen_syms = picks["symbol"].unique().tolist()
    n = len(chosen_syms)
    w = 1.0 / n if n > 0 else 0.0
    return chosen_syms, {s: w for s in chosen_syms}


def main():
    p = argparse.ArgumentParser(description="Out-of-hours crypto live trader")
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    p.add_argument("--max-gross", type=float, default=1.0)
    p.add_argument("--fee-bps", type=float, default=10.0)
    p.add_argument("--weekend-only", action="store_true",
                   help="Conservative mode: only trade Fri-close → Mon-open (skip weekday overnights)")
    p.add_argument("--poll-seconds", type=int, default=60)
    p.add_argument("--dry-run", action="store_true",
                   help="Compute intended orders but don't submit.")
    p.add_argument("--log-file", default=None)
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)sZ [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
        ] + ([logging.FileHandler(args.log_file)] if args.log_file else [])
    )
    log.info("Starting out-of-hours crypto trader")
    log.info(f"  symbols={args.symbols}")
    log.info(f"  max_gross={args.max_gross}")
    log.info(f"  weekend_only={args.weekend_only}")
    log.info(f"  dry_run={args.dry_run}")

    if not args.dry_run:
        log.error("LIVE mode not wired yet — operator must implement alpaca_wrapper integration.")
        log.error("Run with --dry-run to preview orders.")
        sys.exit(2)

    # Dry-run mode: compute the plan for the next 14 days and print
    now = pd.Timestamp.utcnow().tz_localize("UTC") if pd.Timestamp.utcnow().tzinfo is None else pd.Timestamp.utcnow()
    horizon = now + pd.Timedelta(days=14)
    sessions = build_sessions(now.strftime("%Y-%m-%d"), horizon.strftime("%Y-%m-%d"))
    for s in sessions:
        syms, weights = compute_session_picks(
            args.symbols, s.start, s.end, s.kind,
            weekday_enabled=not args.weekend_only,
        )
        if not syms:
            log.info(f"  [{s.kind}] {s.start} → {s.end}: no picks (flat)")
        else:
            tag = "WEEKEND" if s.kind == "weekend" else ("HOLIDAY" if s.kind == "holiday" else "WEEKDAY")
            log.info(f"  [{tag}] {s.start} → {s.end}: buy {syms} w={weights}")

    log.info("Dry-run complete. Review picks before wiring live trades.")


if __name__ == "__main__":
    main()
