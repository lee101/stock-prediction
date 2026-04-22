"""Compute out-of-hours crypto trading sessions using NYSE calendar.

A "session" is any contiguous [start_utc, end_utc) window when NYSE is
CLOSED. We trade crypto long-only during these windows; we are flat
whenever NYSE is OPEN, so this strategy is fully additive with
`xgb-daily-trader-live`.

Session boundaries:
  - session_start = NYSE market_close of day D (or the fixed 21:00 UTC
    previous-session equivalent at the start of the window).
  - session_end   = NYSE market_open of the NEXT trading day.

This naturally handles:
  - Weekday overnights (close 20:00/21:00 UTC → next-day 13:30/14:30 UTC ≈ 16.5h)
  - Weekends (Fri close → Mon open ≈ 64.5h)
  - Holidays (Thu close → Tuesday open when Friday is a holiday, etc.)
  - DST transitions (handled by pandas_market_calendars automatically)

We require at least 1 hour of gap between session end and the next BUY
to absorb slippage/latency.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import pandas_market_calendars as mcal


@dataclass
class Session:
    """One out-of-hours session."""
    start: pd.Timestamp   # UTC, NYSE market_close
    end: pd.Timestamp     # UTC, next trading day's NYSE market_open
    kind: str             # "weekday" | "weekend" | "holiday"
    duration_hours: float

    def as_dict(self) -> dict:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "kind": self.kind,
            "duration_hours": round(self.duration_hours, 3),
        }


def build_sessions(start_date: str, end_date: str) -> list[Session]:
    """Enumerate every out-of-hours window between start_date and end_date (UTC).

    The first session is the one that ends inside the window. We build pairs of
    consecutive NYSE trading days (D, D_next) and define the session as
    [close(D), open(D_next)).
    """
    nyse = mcal.get_calendar("NYSE")
    # Pad the schedule one day each side so we pick up sessions crossing the boundary.
    start_ts = pd.Timestamp(start_date, tz="UTC") - pd.Timedelta(days=3)
    end_ts = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=3)
    sched = nyse.schedule(start_date=start_ts.date(), end_date=end_ts.date())
    sessions: list[Session] = []
    day_list = list(sched.itertuples())
    window_start = pd.Timestamp(start_date, tz="UTC")
    window_end = pd.Timestamp(end_date, tz="UTC")
    for i in range(len(day_list) - 1):
        d0 = day_list[i]
        d1 = day_list[i + 1]
        session_start = pd.Timestamp(d0.market_close).tz_convert("UTC")
        session_end = pd.Timestamp(d1.market_open).tz_convert("UTC")
        if session_end <= window_start or session_start >= window_end:
            continue
        duration = (session_end - session_start).total_seconds() / 3600.0
        # Classify:
        gap_days = (d1.Index.date() - d0.Index.date()).days
        if gap_days == 1:
            kind = "weekday"
        elif gap_days == 3 and d0.Index.weekday() == 4:
            kind = "weekend"
        else:
            # > 3 days (long weekend / holiday) OR gap_days==3 on non-Friday
            kind = "holiday"
        sessions.append(Session(
            start=session_start,
            end=session_end,
            kind=kind,
            duration_hours=duration,
        ))
    return sessions


def sessions_to_df(sessions: Iterable[Session]) -> pd.DataFrame:
    rows = [s.as_dict() for s in sessions]
    df = pd.DataFrame(rows)
    df["start"] = pd.to_datetime(df["start"], utc=True)
    df["end"] = pd.to_datetime(df["end"], utc=True)
    return df


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2025-12-17")
    p.add_argument("--end", default="2026-04-15")
    args = p.parse_args()
    sessions = build_sessions(args.start, args.end)
    print(f"Total sessions in [{args.start}, {args.end}): {len(sessions)}")
    kinds = {}
    total_hours = 0.0
    for s in sessions:
        kinds[s.kind] = kinds.get(s.kind, 0) + 1
        total_hours += s.duration_hours
    print(f"By kind: {kinds}")
    print(f"Total out-of-hours hours: {total_hours:.1f}h")
    print(f"Avg per-session hours: {total_hours / len(sessions):.1f}h")
    print("\nFirst 5 sessions:")
    for s in sessions[:5]:
        print(f"  {s.kind:<8s} {s.start} → {s.end}  ({s.duration_hours:.1f}h)")
    print("\nLast 5 sessions:")
    for s in sessions[-5:]:
        print(f"  {s.kind:<8s} {s.start} → {s.end}  ({s.duration_hours:.1f}h)")
