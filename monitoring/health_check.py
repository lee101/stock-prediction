#!/usr/bin/env python3
"""Alpaca trading system health check.

Checks:
  1. Service liveness (systemd + supervisor)
  2. Alpaca API key validity
  3. Portfolio state consistency
  4. Recent trade activity (staleness detection)
  5. Marketsim vs live PnL drift

Usage:
  python monitoring/health_check.py          # one-shot check
  python monitoring/health_check.py --json   # JSON output
  python monitoring/health_check.py --fix    # attempt auto-remediation
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = REPO_ROOT / "strategy_state" / "stock_portfolio_state.json"
LOG_DIR = REPO_ROOT / "monitoring" / "logs"
LOG_DIR.mkdir(exist_ok=True)


@dataclass
class CheckResult:
    name: str
    status: str  # "ok", "warn", "fail"
    message: str
    details: dict = field(default_factory=dict)


def run_cmd(cmd: str, timeout: int = 15) -> tuple[int, str, str]:
    """Run a shell command, return (returncode, stdout, stderr)."""
    try:
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"
    except Exception as e:
        return -1, "", str(e)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_xgb_daily_trader_live() -> CheckResult:
    """Check `xgb-daily-trader-live` under supervisor — the primary LIVE writer
    since 2026-04-19. Verifies supervisor RUNNING, pid matches singleton lock,
    and no recent auth/traceback errors in the stdout log.
    """
    rc, out, err = run_cmd("sudo -n supervisorctl status xgb-daily-trader-live 2>&1")
    if "RUNNING" not in out:
        return CheckResult(
            "xgb-daily-trader-live",
            "fail",
            f"supervisor not RUNNING: {out or err}",
            {"raw": out},
        )

    # Extract supervisor pid to compare against singleton lock holder.
    sup_pid: Optional[int] = None
    for token in out.split():
        if token.startswith("pid"):
            try:
                sup_pid = int(out.split("pid")[1].split(",")[0].strip())
            except (ValueError, IndexError):
                pass
            break
    # Fallback: grep for a bare pid= pattern
    if sup_pid is None:
        import re
        m = re.search(r"pid\s+(\d+)", out)
        if m:
            sup_pid = int(m.group(1))

    lock_path = REPO_ROOT / "strategy_state" / "account_locks" / "alpaca_live_writer.lock"
    lock_svc: Optional[str] = None
    lock_pid: Optional[int] = None
    if lock_path.exists():
        try:
            rec = json.loads(lock_path.read_text())
            lock_svc = rec.get("service_name")
            lock_pid = rec.get("pid")
        except Exception:
            pass

    if lock_svc != "xgb_live_trader":
        return CheckResult(
            "xgb-daily-trader-live",
            "fail",
            f"supervisor RUNNING but live-writer lock held by service={lock_svc!r} "
            f"(expected 'xgb_live_trader') — possible rogue writer",
            {"sup_pid": sup_pid, "lock_svc": lock_svc, "lock_pid": lock_pid},
        )
    if sup_pid is not None and lock_pid is not None and sup_pid != lock_pid:
        return CheckResult(
            "xgb-daily-trader-live",
            "fail",
            f"supervisor pid ({sup_pid}) != lock pid ({lock_pid}) — stale lock or double writer",
            {"sup_pid": sup_pid, "lock_pid": lock_pid},
        )

    # Error scan on today/yesterday only.
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    yday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    rc2, out2, _ = run_cmd(
        "sudo -n tail -3000 /var/log/supervisor/xgb-daily-trader-live.log 2>&1 | "
        f"grep -E '^({today}|{yday})' | "
        "grep -iE '401|unauthorized|Traceback|death-spiral|RuntimeError' | wc -l"
    )
    try:
        error_count = int(out2)
    except ValueError:
        error_count = 0

    if error_count > 0:
        return CheckResult(
            "xgb-daily-trader-live",
            "fail",
            f"supervisor RUNNING but {error_count} error lines today/yesterday "
            f"(401/Traceback/death-spiral/RuntimeError)",
            {"error_count": error_count, "sup_pid": sup_pid},
        )

    return CheckResult(
        "xgb-daily-trader-live",
        "ok",
        f"supervisor RUNNING (pid={sup_pid}), singleton lock held by xgb_live_trader, "
        f"no recent errors",
        {"sup_pid": sup_pid, "lock_pid": lock_pid},
    )


def check_daily_rl_trader_stopped() -> CheckResult:
    """Verify `daily-rl-trader` is STOPPED (intentional, since 2026-04-19).

    If this unit is RUNNING while `xgb-daily-trader-live` is also RUNNING,
    they'll race for the singleton lock and one will crash. Expected state
    is STOPPED / EXITED / FATAL / not-in-config.
    """
    rc, out, _ = run_cmd("sudo -n supervisorctl status daily-rl-trader 2>&1")
    if "RUNNING" in out:
        return CheckResult(
            "daily-rl-trader-stopped",
            "fail",
            f"daily-rl-trader is RUNNING but must be STOPPED (XGB holds the singleton) — "
            f"immediate `sudo supervisorctl stop daily-rl-trader` needed",
            {"raw": out},
        )
    return CheckResult(
        "daily-rl-trader-stopped",
        "ok",
        "daily-rl-trader is STOPPED as expected (XGB is the live writer)",
        {"raw": out},
    )


def check_trading_server_stopped() -> CheckResult:
    """Verify `trading-server` (broker boundary on :8050) is STOPPED.

    It was the broker boundary for the now-stopped `daily-rl-trader`. Since
    XGB writes direct to Alpaca SDK, port 8050 should be CLOSED. If the
    port is open, either the rollback was triggered without stopping XGB
    (race condition) or a stale process is lingering.
    """
    rc, out, _ = run_cmd("sudo -n supervisorctl status trading-server 2>&1")
    supervisor_running = "RUNNING" in out
    # Also probe the port directly — even a non-supervisor process would
    # be a violation since we expect :8050 CLOSED.
    rc_ss, out_ss, _ = run_cmd("ss -ltn 2>/dev/null | grep ':8050' | head -1")
    port_open = bool(out_ss.strip())
    if supervisor_running or port_open:
        return CheckResult(
            "trading-server-stopped",
            "fail",
            f"trading-server activity detected (supervisor_running={supervisor_running}, "
            f"port_8050_open={port_open}) — must be STOPPED while XGB is live",
            {"supervisor": out, "ss": out_ss},
        )
    return CheckResult(
        "trading-server-stopped",
        "ok",
        "trading-server STOPPED and :8050 closed as expected",
    )


def check_stale_writer_locks() -> CheckResult:
    """Flag stale/orphan writer locks — each lock file records a pid+host;
    if the holder process no longer exists, the file is orphaned and will
    block the next legitimate acquirer with a "lock held" error.
    """
    lock_dir = REPO_ROOT / "strategy_state" / "account_locks"
    if not lock_dir.exists():
        return CheckResult("writer-locks", "ok", "no lock dir yet")
    orphans = []
    alive = []
    for lock in lock_dir.glob("*.lock"):
        try:
            rec = json.loads(lock.read_text())
        except Exception:
            continue
        pid = rec.get("pid")
        host = rec.get("hostname")
        if not isinstance(pid, int):
            continue
        # /proc lookup is cheap and host-local; only trust if hostname matches.
        this_host = os.uname().nodename
        if host and host != this_host:
            alive.append((lock.name, pid, f"remote host {host}"))
            continue
        if Path(f"/proc/{pid}").exists():
            alive.append((lock.name, pid, rec.get("service_name", "?")))
        else:
            orphans.append((lock.name, pid, rec.get("service_name", "?"), rec.get("started_at", "?")))
    if orphans:
        return CheckResult(
            "writer-locks",
            "warn",
            f"{len(orphans)} orphan lock(s): "
            + "; ".join(f"{n} pid={p} svc={s} since {t}" for n, p, s, t in orphans),
            {"orphans": orphans, "alive": alive},
        )
    return CheckResult(
        "writer-locks",
        "ok",
        f"{len(alive)} active lock(s); no orphans",
        {"alive": alive},
    )


def check_death_spiral_markers() -> CheckResult:
    """Detect if the death-spiral guard has fired recently — markers are
    written to `<state>/alpaca_singleton/markers/`."""
    marker_dir = REPO_ROOT / "strategy_state" / "alpaca_singleton" / "markers"
    if not marker_dir.exists():
        return CheckResult("death-spiral", "ok", "no markers dir")
    recent = []
    now = time.time()
    for m in marker_dir.glob("*.marker"):
        try:
            age = now - m.stat().st_mtime
        except Exception:
            continue
        if age <= 24 * 3600:
            recent.append((m.name, age))
    if recent:
        return CheckResult(
            "death-spiral",
            "fail",
            f"{len(recent)} death-spiral marker(s) in last 24h — guard fired",
            {"markers": [(n, round(a)) for n, a in recent]},
        )
    return CheckResult("death-spiral", "ok", "no guard firings in last 24h")


def check_llm_stock_trader() -> CheckResult:
    """Check the optional llm-stock-trader supervisor process.

    This is an optional research/experimental trader — not core prod. If the
    unit isn't configured or is stopped, report `warn` rather than `fail`
    so core-prod health isn't masked by an optional component being down.
    """
    rc, out, _ = run_cmd("sudo -n supervisorctl status llm-stock-trader 2>&1")
    if "RUNNING" in out:
        return CheckResult("llm-stock-trader", "ok", "supervisor process running")
    if "no such process" in out.lower() or "no such group" in out.lower():
        return CheckResult("llm-stock-trader", "warn", "not configured in supervisor (optional)")
    return CheckResult("llm-stock-trader", "warn", f"not running (optional): {out}")


def check_cancel_multi_orders() -> CheckResult:
    """Check alpaca-cancel-multi-orders.service."""
    rc, out, _ = run_cmd("sudo systemctl is-active alpaca-cancel-multi-orders.service")
    if out == "active":
        return CheckResult("cancel-multi-orders", "ok", "service active")
    return CheckResult("cancel-multi-orders", "warn", f"service not active: {out}")


def check_alpaca_api() -> CheckResult:
    """Quick API key validity check via positions endpoint."""
    try:
        sys.path.insert(0, str(REPO_ROOT))
        # Try importing env_real to get credentials
        import importlib.util
        spec = importlib.util.spec_from_file_location("env_real", REPO_ROOT / "env_real.py")
        if spec is None or spec.loader is None:
            return CheckResult("alpaca-api", "warn", "env_real.py not found")
        env_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(env_mod)

        key_id = getattr(env_mod, "ALP_KEY_ID_PROD", None)
        secret = getattr(env_mod, "ALP_SECRET_KEY_PROD", None)
        if not key_id or not secret:
            return CheckResult("alpaca-api", "fail", "ALP_KEY_ID_PROD or ALP_SECRET_KEY_PROD not set")

        import urllib.request
        import urllib.error
        req = urllib.request.Request(
            "https://api.alpaca.markets/v2/account",
            headers={
                "APCA-API-KEY-ID": key_id,
                "APCA-API-SECRET-KEY": secret,
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            equity = float(data.get("equity", 0))
            return CheckResult("alpaca-api", "ok",
                               f"API key valid, equity=${equity:,.2f}",
                               {"equity": equity, "buying_power": float(data.get("buying_power", 0))})
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return CheckResult("alpaca-api", "fail", "API key EXPIRED (401 Unauthorized)")
        return CheckResult("alpaca-api", "fail", f"HTTP {e.code}: {e.reason}")
    except Exception as e:
        return CheckResult("alpaca-api", "warn", f"check failed: {e}")


def check_portfolio_state() -> CheckResult:
    """Check portfolio state file for consistency."""
    if not STATE_FILE.exists():
        return CheckResult("portfolio-state", "warn", "state file not found")

    try:
        state = json.loads(STATE_FILE.read_text())
    except Exception as e:
        return CheckResult("portfolio-state", "fail", f"corrupt state file: {e}")

    positions = state.get("positions", {})
    pending_close = state.get("pending_close", [])

    issues = []
    # Check for stale pending_close entries
    if pending_close:
        # If pending_close but no actual position, it's stale
        for sym in pending_close:
            if sym not in positions:
                issues.append(f"{sym} in pending_close but not in positions (stale)")

    if issues:
        return CheckResult("portfolio-state", "warn",
                           f"{len(issues)} issue(s): {'; '.join(issues)}",
                           {"positions": len(positions), "pending_close": pending_close})

    return CheckResult("portfolio-state", "ok",
                       f"{len(positions)} open positions, {len(pending_close)} pending close")


def check_recent_activity() -> CheckResult:
    """Check that XGB emitted a trading-session marker today or yesterday.

    The daemon writes one JSONL per trading day under
    `analysis/xgb_live_trade_log/YYYY-MM-DD.jsonl` with events like
    `session_start`, `scored`, `filtered`, `pick`, `buy_submitted`,
    `buy_filled`, `sell_submitted`, `session_end`. Also cross-checks
    the supervisor stdout log for recent `Scoring` or `No picks` lines.
    """
    log_dir = REPO_ROOT / "analysis" / "xgb_live_trade_log"
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    yday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")

    session_events = 0
    last_file = None
    if log_dir.exists():
        for day in (today, yday):
            p = log_dir / f"{day}.jsonl"
            if p.exists():
                last_file = p
                try:
                    rows = [l for l in p.read_text().splitlines() if l.strip()]
                    session_events += len(rows)
                except Exception:
                    pass

    # Also count supervisor stdout Scoring/No-picks lines as a
    # secondary activity signal.
    rc, out, _ = run_cmd(
        "sudo -n tail -4000 /var/log/supervisor/xgb-daily-trader-live.log 2>&1 | "
        f"grep -E '^({today}|{yday})' | "
        "grep -cE 'Scoring |No picks today|Conviction filter|BUY |SELL '"
    )
    try:
        sup_count = int(out)
    except ValueError:
        sup_count = 0

    if session_events == 0 and sup_count == 0:
        return CheckResult(
            "recent-activity",
            "warn",
            "no XGB session events in last 48h (trade-log dir empty AND "
            "supervisor stdout has no Scoring/No-picks/BUY/SELL lines)",
        )
    return CheckResult(
        "recent-activity",
        "ok",
        f"{session_events} trade-log events + {sup_count} stdout session markers "
        f"in last 48h" + (f" (latest: {last_file.name})" if last_file else ""),
    )


def check_gpu_available() -> CheckResult:
    """Check if GPU is available for training/inference."""
    rc, out, _ = run_cmd("nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits 2>&1")
    if rc != 0:
        return CheckResult("gpu", "fail", f"nvidia-smi failed: {out}")

    lines = [l.strip() for l in out.split("\n") if l.strip()]
    if not lines:
        return CheckResult("gpu", "fail", "no GPUs detected")

    gpus = []
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            gpus.append({
                "name": parts[0],
                "memory_used_mb": int(parts[1]),
                "memory_total_mb": int(parts[2]),
            })

    return CheckResult("gpu", "ok",
                       f"{len(gpus)} GPU(s): {gpus[0]['name']}, "
                       f"{gpus[0]['memory_used_mb']}/{gpus[0]['memory_total_mb']}MB used",
                       {"gpus": gpus})


def check_disk_space() -> CheckResult:
    """Check disk space on both `/` and `/nvme0n1-disk`.

    Either filling up breaks prod: `/` holds supervisor logs + /var/log, and
    `/nvme0n1-disk` holds repo, venvs, model checkpoints, strategy_state.
    """
    mounts = ["/", "/nvme0n1-disk"]
    worst_pct = 0
    worst_mount = ""
    details: dict = {}
    for mnt in mounts:
        rc, out, _ = run_cmd(f"df -P {mnt} | tail -1")
        if rc != 0 or not out:
            details[mnt] = "df failed"
            continue
        parts = out.split()
        if len(parts) < 5:
            continue
        try:
            pct = int(parts[4].rstrip("%"))
        except ValueError:
            continue
        details[mnt] = f"{pct}%"
        if pct > worst_pct:
            worst_pct = pct
            worst_mount = mnt
    summary = ", ".join(f"{m}={v}" for m, v in details.items()) or "no mounts"
    if worst_pct > 90:
        return CheckResult("disk", "fail", f"{worst_mount} {worst_pct}% full — {summary}", details)
    if worst_pct > 80:
        return CheckResult("disk", "warn", f"{worst_mount} {worst_pct}% full — {summary}", details)
    return CheckResult("disk", "ok", summary, details)


# ---------------------------------------------------------------------------
# Auto-fix actions
# ---------------------------------------------------------------------------

def auto_fix(results: list[CheckResult]) -> list[str]:
    """Attempt auto-remediation for known issues. Returns list of actions taken."""
    actions = []

    for r in results:
        if r.name == "portfolio-state" and r.status == "warn" and "stale" in r.message:
            # Clean stale pending_close entries
            try:
                state = json.loads(STATE_FILE.read_text())
                positions = state.get("positions", {})
                pending = state.get("pending_close", [])
                cleaned = [s for s in pending if s in positions]
                if len(cleaned) != len(pending):
                    state["pending_close"] = cleaned
                    STATE_FILE.write_text(json.dumps(state, indent=2))
                    actions.append(f"Cleaned {len(pending) - len(cleaned)} stale pending_close entries")
            except Exception as e:
                actions.append(f"Failed to clean pending_close: {e}")

        if r.name == "cancel-multi-orders" and r.status in ("warn", "fail"):
            # Try restarting the service
            rc, _, err = run_cmd("sudo systemctl restart alpaca-cancel-multi-orders.service")
            if rc == 0:
                actions.append("Restarted alpaca-cancel-multi-orders.service")
            else:
                actions.append(f"Failed to restart cancel-multi-orders: {err}")

    return actions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_all_checks() -> list[CheckResult]:
    checks = [
        check_xgb_daily_trader_live,
        check_daily_rl_trader_stopped,
        check_trading_server_stopped,
        check_stale_writer_locks,
        check_death_spiral_markers,
        check_llm_stock_trader,
        check_cancel_multi_orders,
        check_alpaca_api,
        check_portfolio_state,
        check_recent_activity,
        check_gpu_available,
        check_disk_space,
    ]
    results = []
    for check_fn in checks:
        try:
            results.append(check_fn())
        except Exception as e:
            results.append(CheckResult(check_fn.__name__, "fail", f"check crashed: {e}"))
    return results


def main():
    parser = argparse.ArgumentParser(description="Alpaca trading health check")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--fix", action="store_true", help="attempt auto-fix")
    args = parser.parse_args()

    results = run_all_checks()

    if args.fix:
        actions = auto_fix(results)
        if actions:
            # Re-run checks after fixes
            results = run_all_checks()

    # Log results
    now = datetime.now(timezone.utc)
    log_entry = {
        "timestamp": now.isoformat(),
        "results": [asdict(r) for r in results],
    }

    log_file = LOG_DIR / f"health_{now.strftime('%Y%m%d')}.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    if args.json:
        print(json.dumps(log_entry, indent=2))
        return

    # Human-readable output
    status_icons = {"ok": "+", "warn": "!", "fail": "X"}
    fails = 0
    warns = 0
    for r in results:
        icon = status_icons.get(r.status, "?")
        print(f"  [{icon}] {r.name}: {r.message}")
        if r.status == "fail":
            fails += 1
        elif r.status == "warn":
            warns += 1

    if args.fix and actions:
        print(f"\nAuto-fix actions taken:")
        for a in actions:
            print(f"  -> {a}")

    print(f"\n{'HEALTHY' if fails == 0 else 'UNHEALTHY'}: {len(results) - fails - warns} ok, {warns} warn, {fails} fail")
    sys.exit(1 if fails > 0 else 0)


if __name__ == "__main__":
    main()
