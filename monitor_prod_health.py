#!/usr/bin/env python
"""Production health monitor for the unified orchestrator.

Checks:
  1. Process health  — is the orchestrator process running?
  2. Last cycle time — when did the last cycle complete?
  3. Error rate      — how many ERROR lines in the last hour of logs?
  4. Positions       — what does strategy_state/unified_state.json say?
  5. Fills           — any recent entries in strategy_state/fill_events.jsonl?
  6. Equity trend    — is equity up or down vs yesterday's snapshot?

Output: GREEN / YELLOW / RED status per check.

Usage:
  python monitor_prod_health.py
  python monitor_prod_health.py --log-path logs/unified_orchestrator.log
  python monitor_prod_health.py --state-dir /custom/strategy_state
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GREEN = "GREEN"
YELLOW = "YELLOW"
RED = "RED"

EQUITY_HISTORY_FILE_NAME = "equity_history.jsonl"

# How stale (in minutes) the last cycle can be before warning / alarm
CYCLE_WARN_MINUTES = 75    # slightly over 1 hour
CYCLE_RED_MINUTES = 130    # more than 2 hours

# Error-line thresholds in the last hour
ERROR_WARN_COUNT = 3
ERROR_RED_COUNT = 10

# Equity change thresholds (fraction)
EQUITY_WARN_DROP = 0.02   # 2 %
EQUITY_RED_DROP = 0.05    # 5 %


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _color(status: str) -> str:
    """Return an ANSI-colored status string for terminal output."""
    codes = {GREEN: "\033[92m", YELLOW: "\033[93m", RED: "\033[91m"}
    reset = "\033[0m"
    return f"{codes.get(status, '')}{status}{reset}"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Check 1 — Process health
# ---------------------------------------------------------------------------

def check_process() -> tuple[str, str]:
    """Return (status, detail) for orchestrator process liveness."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "unified_orchestrator.orchestrator"],
            capture_output=True, text=True, timeout=5,
        )
        pids = [p.strip() for p in result.stdout.strip().split("\n") if p.strip()]
        if pids:
            return GREEN, f"Running (PIDs: {', '.join(pids)})"
        return RED, "No orchestrator process found"
    except FileNotFoundError:
        return YELLOW, "pgrep not available on this system"
    except Exception as e:
        return YELLOW, f"Could not check process: {e}"


# ---------------------------------------------------------------------------
# Check 2 — Last cycle time
# ---------------------------------------------------------------------------

_CYCLE_PATTERN = re.compile(
    r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})"
    r".*(?:UNIFIED TRADING CYCLE|Cycle complete)"
)


def check_last_cycle(log_path: Path) -> tuple[str, str]:
    """Parse log for the most recent cycle timestamp."""
    if not log_path.exists():
        return YELLOW, f"Log file not found: {log_path}"

    last_ts: datetime | None = None
    try:
        # Read the tail of the log (last 500 lines) to find the latest cycle
        lines = _tail(log_path, 500)
        for line in reversed(lines):
            m = _CYCLE_PATTERN.search(line)
            if m:
                raw = m.group(1)
                try:
                    last_ts = datetime.fromisoformat(raw).replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
                if last_ts is not None:
                    break
    except Exception as e:
        return YELLOW, f"Error reading log: {e}"

    if last_ts is None:
        return YELLOW, "No cycle timestamps found in log"

    age_min = (_now_utc() - last_ts).total_seconds() / 60
    age_str = f"{age_min:.0f} min ago ({last_ts.strftime('%Y-%m-%d %H:%M UTC')})"

    if age_min > CYCLE_RED_MINUTES:
        return RED, f"Last cycle {age_str} — STALE"
    if age_min > CYCLE_WARN_MINUTES:
        return YELLOW, f"Last cycle {age_str} — slightly stale"
    return GREEN, f"Last cycle {age_str}"


def _tail(path: Path, n: int = 500) -> list[str]:
    """Return up to the last *n* lines of a file."""
    try:
        with open(path, "rb") as f:
            # Seek from end to efficiently grab tail
            f.seek(0, 2)
            size = f.tell()
            # Read last 512 KB at most
            chunk = min(size, 512 * 1024)
            f.seek(max(0, size - chunk))
            data = f.read().decode("utf-8", errors="replace")
        lines = data.splitlines()
        return lines[-n:]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Check 3 — Error rate
# ---------------------------------------------------------------------------

_ERROR_LINE = re.compile(r"\bERROR\b", re.IGNORECASE)
_TIMESTAMP_IN_LINE = re.compile(r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})")


def check_error_rate(log_path: Path) -> tuple[str, str]:
    """Count ERROR lines in the last hour of logs."""
    if not log_path.exists():
        return YELLOW, f"Log file not found: {log_path}"

    cutoff = _now_utc() - timedelta(hours=1)
    error_count = 0
    total_recent = 0
    sample_errors: list[str] = []

    try:
        lines = _tail(log_path, 2000)
        for line in lines:
            ts_match = _TIMESTAMP_IN_LINE.search(line)
            if ts_match:
                try:
                    ts = datetime.fromisoformat(ts_match.group(1)).replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
                if ts < cutoff:
                    continue
                total_recent += 1
                if _ERROR_LINE.search(line):
                    error_count += 1
                    if len(sample_errors) < 3:
                        sample_errors.append(line.strip()[:120])
    except Exception as e:
        return YELLOW, f"Error reading log: {e}"

    detail = f"{error_count} errors in last hour ({total_recent} lines scanned)"
    if sample_errors:
        detail += "\n      Latest: " + "\n              ".join(sample_errors)

    if error_count >= ERROR_RED_COUNT:
        return RED, detail
    if error_count >= ERROR_WARN_COUNT:
        return YELLOW, detail
    return GREEN, detail


# ---------------------------------------------------------------------------
# Check 4 — Positions
# ---------------------------------------------------------------------------

def check_positions(state_path: Path, state_data: dict | None = None) -> tuple[str, str]:
    """Read unified_state.json and summarise positions."""
    if state_data is None:
        if not state_path.exists():
            return YELLOW, f"State file not found: {state_path}"
        return RED, f"Cannot parse state file: {state_path}"

    data = state_data
    ts = data.get("timestamp", "unknown")
    regime = data.get("regime", "unknown")
    total = data.get("total_value", 0.0)
    cash = data.get("alpaca_cash", 0.0)

    alpaca_pos = data.get("alpaca_positions", {})
    binance_pos = data.get("binance_positions", {})

    pos_lines = []
    for sym, pos in {**alpaca_pos, **binance_pos}.items():
        qty = pos.get("qty", 0)
        price = pos.get("current_price", 0)
        mv = qty * price if price else 0
        if mv >= 1.0:
            pos_lines.append(f"{sym}: {qty} @ ${price:,.2f} (${mv:,.0f})")

    detail = (
        f"Snapshot {ts} | Regime={regime}\n"
        f"      Total: ${total:,.2f} | Cash: ${cash:,.2f}\n"
        f"      Positions: {len(pos_lines) if pos_lines else 'none with value >= $1'}"
    )
    if pos_lines:
        detail += "\n      " + "\n      ".join(pos_lines)

    # State age check
    try:
        state_ts = datetime.fromisoformat(ts)
        age_min = (_now_utc() - state_ts).total_seconds() / 60
        if age_min > CYCLE_RED_MINUTES:
            return YELLOW, detail + f"\n      (state is {age_min:.0f} min old)"
    except (ValueError, TypeError):
        pass

    return GREEN, detail


# ---------------------------------------------------------------------------
# Check 5 — Fills
# ---------------------------------------------------------------------------

def check_fills(fill_path: Path) -> tuple[str, str]:
    """Read fill_events.jsonl and report recent fills."""
    if not fill_path.exists():
        return YELLOW, f"Fill events file not found: {fill_path}"

    now = _now_utc()
    cutoff_24h = now - timedelta(hours=24)
    cutoff_1h = now - timedelta(hours=1)
    fills_24h: list[dict] = []
    fills_1h: list[dict] = []

    try:
        text = fill_path.read_text().strip()
        if not text:
            return YELLOW, "Fill events file is empty"
        for line in text.split("\n"):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                ts = datetime.fromisoformat(event["timestamp"])
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts >= cutoff_24h:
                    fills_24h.append(event)
                if ts >= cutoff_1h:
                    fills_1h.append(event)
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    except OSError as e:
        return RED, f"Cannot read fill events: {e}"

    detail = f"{len(fills_1h)} fills in last hour, {len(fills_24h)} in last 24h"
    if fills_1h:
        for f in fills_1h[-5:]:
            detail += (
                f"\n      {f.get('symbol','?')} {f.get('action','?')} "
                f"@ ${f.get('fill_price', '?')} qty={f.get('fill_qty', '?')} "
                f"({f.get('timestamp', '?')[:19]})"
            )

    if fills_24h:
        return GREEN, detail
    return YELLOW, detail + " (no fills in 24h)"


# ---------------------------------------------------------------------------
# Check 6 — Equity trend
# ---------------------------------------------------------------------------

def _load_equity_history(history_path: Path) -> list[dict]:
    """Load the equity history JSONL file."""
    if not history_path.exists():
        return []
    entries: list[dict] = []
    for line in history_path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def _save_equity_snapshot(history_path: Path, total_value: float) -> None:
    """Append the current equity reading to the history file.

    Trims entries older than 30 days to prevent unbounded growth.
    """
    history_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": _now_utc().isoformat(),
        "total_value": total_value,
    }
    with open(history_path, "a") as f:
        f.write(json.dumps(entry) + "\n")

    # Trim old entries (keep last 30 days)
    cutoff = _now_utc() - timedelta(days=30)
    try:
        entries = _load_equity_history(history_path)
        kept = []
        for e in entries:
            try:
                ts = datetime.fromisoformat(e["timestamp"])
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts >= cutoff:
                    kept.append(e)
            except (KeyError, ValueError):
                kept.append(e)  # keep unparseable entries to avoid data loss
        if len(kept) < len(entries):
            with open(history_path, "w") as f:
                for e in kept:
                    f.write(json.dumps(e) + "\n")
    except OSError:
        pass  # trim is best-effort


def check_equity(state_data: dict | None, state_dir: Path) -> tuple[str, str]:
    """Compare current equity vs the oldest snapshot from ~24h ago."""
    history_path = state_dir / EQUITY_HISTORY_FILE_NAME

    # Read current total_value from pre-parsed state
    current_value = 0.0
    if state_data is not None:
        try:
            current_value = float(state_data.get("total_value", 0))
        except (TypeError, ValueError):
            pass

    if current_value <= 0:
        return YELLOW, "Cannot determine current equity"

    # Save current reading
    _save_equity_snapshot(history_path, current_value)

    # Find the reading closest to 24 hours ago
    history = _load_equity_history(history_path)
    if len(history) < 2:
        return GREEN, f"Current equity: ${current_value:,.2f} (no prior snapshot to compare)"

    target_time = _now_utc() - timedelta(hours=24)
    best_entry: dict | None = None
    best_delta = float("inf")
    for entry in history:
        try:
            ts = datetime.fromisoformat(entry["timestamp"])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            delta = abs((ts - target_time).total_seconds())
            if delta < best_delta:
                best_delta = delta
                best_entry = entry
        except (KeyError, ValueError):
            continue

    if best_entry is None:
        return GREEN, f"Current equity: ${current_value:,.2f} (no usable prior snapshot)"

    prev_value = float(best_entry.get("total_value", 0))
    if prev_value <= 0:
        return YELLOW, f"Current equity: ${current_value:,.2f} (prior snapshot has zero value)"

    pct_change = (current_value - prev_value) / prev_value
    prev_ts_str = best_entry.get("timestamp", "?")[:19]
    detail = (
        f"${current_value:,.2f} now vs ${prev_value:,.2f} at {prev_ts_str} "
        f"({pct_change:+.2%})"
    )

    if pct_change <= -EQUITY_RED_DROP:
        return RED, detail
    if pct_change <= -EQUITY_WARN_DROP:
        return YELLOW, detail
    return GREEN, detail


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_all_checks(log_path: Path, state_dir: Path) -> list[tuple[str, str, str]]:
    """Run all checks and return [(name, status, detail), ...]."""
    state_path = state_dir / "unified_state.json"
    fill_path = state_dir / "fill_events.jsonl"

    # Read state file once and share across checks that need it
    state_data: dict | None = None
    if state_path.exists():
        try:
            state_data = json.loads(state_path.read_text())
        except (json.JSONDecodeError, OSError):
            state_data = None

    results = []
    results.append(("Process", *check_process()))
    results.append(("Last Cycle", *check_last_cycle(log_path)))
    results.append(("Error Rate", *check_error_rate(log_path)))
    results.append(("Positions", *check_positions(state_path, state_data)))
    results.append(("Fills", *check_fills(fill_path)))
    results.append(("Equity", *check_equity(state_data, state_dir)))
    return results


def print_report(results: list[tuple[str, str, str]]) -> None:
    """Pretty-print the health report to stdout."""
    print(f"\n{'=' * 70}")
    print(f"  ORCHESTRATOR HEALTH REPORT  —  {_now_utc().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'=' * 70}\n")

    for name, status, detail in results:
        tag = _color(status)
        print(f"  [{tag}]  {name}")
        for line in detail.split("\n"):
            print(f"      {line}")
        print()

    # Overall
    statuses = [s for _, s, _ in results]
    if RED in statuses:
        overall = RED
    elif YELLOW in statuses:
        overall = YELLOW
    else:
        overall = GREEN
    print(f"  Overall: {_color(overall)}")
    print(f"{'=' * 70}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Orchestrator production health monitor")
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("logs/unified_orchestrator.log"),
        help="Path to the orchestrator log file",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path("strategy_state"),
        help="Directory containing unified_state.json and fill_events.jsonl",
    )
    args = parser.parse_args(argv)

    results = run_all_checks(args.log_path, args.state_dir)
    print_report(results)

    # Exit code: 0 = all green, 1 = yellow, 2 = red
    statuses = [s for _, s, _ in results]
    if RED in statuses:
        return 2
    if YELLOW in statuses:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
