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
import hashlib
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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from unified_orchestrator.jsonl_utils import append_jsonl_row

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
    _rc, out, err = run_cmd("sudo -n supervisorctl status xgb-daily-trader-live 2>&1")
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
    _rc2, out2, _ = run_cmd(
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
    _rc, out, _ = run_cmd("sudo -n supervisorctl status daily-rl-trader 2>&1")
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
    _rc, out, _ = run_cmd("sudo -n supervisorctl status trading-server 2>&1")
    supervisor_running = "RUNNING" in out
    # Also probe the port directly — even a non-supervisor process would
    # be a violation since we expect :8050 CLOSED.
    _rc_ss, out_ss, _ = run_cmd("ss -ltn 2>/dev/null | grep ':8050' | head -1")
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
    _rc, out, _ = run_cmd("sudo -n supervisorctl status llm-stock-trader 2>&1")
    if "RUNNING" in out:
        return CheckResult("llm-stock-trader", "ok", "supervisor process running")
    if "no such process" in out.lower() or "no such group" in out.lower():
        return CheckResult("llm-stock-trader", "warn", "not configured in supervisor (optional)")
    return CheckResult("llm-stock-trader", "warn", f"not running (optional): {out}")


def check_cancel_multi_orders() -> CheckResult:
    """Check alpaca-cancel-multi-orders.service."""
    _rc, out, _ = run_cmd("sudo systemctl is-active alpaca-cancel-multi-orders.service")
    if out == "active":
        return CheckResult("cancel-multi-orders", "ok", "service active")
    return CheckResult("cancel-multi-orders", "warn", f"service not active: {out}")


def _load_secretbashrc_env() -> None:
    """Source ~/.secretbashrc via bash and copy emitted env vars into os.environ.

    env_real.py falls back to literal placeholder strings when ALP_KEY_ID_PROD /
    ALP_SECRET_KEY_PROD aren't set. When this script is invoked from cron or a
    shell that didn't source ~/.secretbashrc (which is where the real keys live
    on this box per deployments/xgb-daily-trader-live/launch.sh), those placeholders
    otherwise produce a false 401 "API key EXPIRED" alert. Load the file here.
    """
    rc = os.path.expanduser("~/.secretbashrc")
    if not os.path.isfile(rc):
        return
    try:
        out = subprocess.run(
            ["bash", "-c", f'set -a; source "{rc}" >/dev/null 2>&1; env'],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return
        for line in out.stdout.splitlines():
            if "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.startswith("ALP_") or k.startswith("APCA_"):
                os.environ.setdefault(k, v)
    except Exception:
        pass


def check_alpaca_api() -> CheckResult:
    """Quick API key validity check via positions endpoint."""
    try:
        sys.path.insert(0, str(REPO_ROOT))
        _load_secretbashrc_env()
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
        if key_id.startswith("alpaca-") or secret.startswith("alpaca-"):
            return CheckResult(
                "alpaca-api", "warn",
                "env_real using placeholder keys — source ~/.secretbashrc before calling, "
                "or export ALP_KEY_ID_PROD/ALP_SECRET_KEY_PROD",
            )

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
    _rc, out, _ = run_cmd(
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


def check_xgb_spy_provenance() -> CheckResult:
    """Fail if XGB live SPY risk-control decision events lack byte provenance."""
    analyzer = REPO_ROOT / "scripts" / "analyze_xgb_trade_log.py"
    rc, out, err = run_cmd(
        f"{sys.executable} {analyzer} --json --fail-on-spy-provenance-warning",
        timeout=30,
    )

    if rc == 0 and not out:
        return CheckResult(
            "xgb-spy-provenance",
            "ok",
            "no XGB trade-log files to inspect; recent-activity handles staleness",
            {"stderr": err},
        )

    payload: dict | None = None
    if out:
        try:
            payload = json.loads(out)
        except Exception as exc:
            return CheckResult(
                "xgb-spy-provenance",
                "warn",
                f"trade-log analyzer returned non-JSON output: {exc}",
                {"returncode": rc, "stdout": out[-2000:], "stderr": err[-2000:]},
            )

    if rc == 3:
        overall = (payload or {}).get("overall", {})
        sessions = overall.get("spy_provenance_warning_sessions", [])
        return CheckResult(
            "xgb-spy-provenance",
            "fail",
            f"SPY provenance warnings in {len(sessions)} session(s): "
            f"{', '.join(str(s) for s in sessions) or 'unknown'}",
            {"overall": overall},
        )

    if rc != 0:
        return CheckResult(
            "xgb-spy-provenance",
            "warn",
            f"trade-log analyzer failed rc={rc}: {err or out}",
            {"returncode": rc, "stdout": out[-2000:], "stderr": err[-2000:]},
        )

    overall = (payload or {}).get("overall", {})
    n_hashes = overall.get("n_spy_session_hashes", 0)
    n_sessions = overall.get("n_sessions", 0)
    return CheckResult(
        "xgb-spy-provenance",
        "ok",
        f"no SPY provenance warnings across {n_sessions} analyzed session(s); "
        f"{n_hashes} unique SPY session hash(es)",
        {"overall": overall},
    )


def _parse_current_status_line(text: str) -> tuple[dict[str, str], str | None]:
    fields: dict[str, str] = {}
    tokens = text.strip().split()
    if not tokens:
        return fields, "empty current status"
    for token in tokens[1:]:
        if "=" not in token:
            return fields, f"unexpected token {token!r}"
        key, _, value = token.partition("=")
        if not key:
            return fields, "empty current status field name"
        if key in fields:
            return fields, f"duplicate current status field {key!r}"
        fields[key] = value
    return fields, None


def _current_status_timestamp(text: str) -> datetime:
    timestamp = text.strip().split(maxsplit=1)[0] if text.strip() else ""
    if not timestamp:
        raise ValueError("missing current status timestamp")
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("invalid current status timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _hourly_audit_expected_now(now: float) -> bool:
    now_utc = datetime.fromtimestamp(now, tz=timezone.utc)
    # The hourly audit runs during the trading day (13:00-20:00 UTC). Keep
    # enforcing freshness through the 3h grace period so the 22:00 monitor run
    # catches a missed late-session audit instead of suppressing it as off-hours.
    return now_utc.weekday() < 5 and 13 <= now_utc.hour < 23


def _status_artifact_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return LOG_DIR / path


def _status_artifact_is_in_log_dir(path: Path) -> bool:
    try:
        return path.resolve().is_relative_to(LOG_DIR.resolve())
    except OSError:
        return False


def _expected_ok_status_artifact_names(name: str, fields: dict[str, str]) -> dict[str, str] | None:
    log_name = Path(fields.get("log", "")).name
    if name == "codex":
        if not log_name.startswith("codex_prod_") or _artifact_timestamp_from_log_name(log_name) is None:
            return None
        stem = log_name.removesuffix(".log")
        return {
            "log": log_name,
            "raw_log": f"{stem}.raw.jsonl",
            "last_msg": f"{stem}.last.txt",
        }
    if name == "hourly":
        if not log_name.startswith("hourly_prod_") or _artifact_timestamp_from_log_name(log_name) is None:
            return None
        stem = log_name.removesuffix(".log")
        return {
            "log": log_name,
            "raw_log": f"{stem}.raw.jsonl",
        }
    return None


def _artifact_timestamp_from_log_name(log_name: str) -> datetime | None:
    if log_name.startswith("monitor_") and log_name.endswith(".log"):
        stamp = log_name.removeprefix("monitor_").removesuffix(".log")
        fmt = "%Y%m%dT%H%M%S"
    elif log_name.startswith("codex_prod_") and log_name.endswith(".log"):
        stamp = log_name.removeprefix("codex_prod_").removesuffix(".log")
        fmt = "%Y%m%dT%H%M%SZ"
    elif log_name.startswith("hourly_prod_") and log_name.endswith(".log"):
        stamp = log_name.removeprefix("hourly_prod_").removesuffix(".log")
        fmt = "%Y%m%dT%H%M%SZ"
    else:
        return None
    try:
        return datetime.strptime(stamp, fmt).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _artifact_timestamp_errors(log_name: str, current_ts: datetime) -> list[str]:
    artifact_ts = _artifact_timestamp_from_log_name(log_name)
    if artifact_ts is None:
        return ["log:bad_timestamp"]
    age_s = (current_ts - artifact_ts).total_seconds()
    if age_s < -300:
        return ["log:future_timestamp"]
    if age_s > 2 * 3600:
        return ["log:stale_artifact_timestamp"]
    return []


def _is_sha256_hex(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdefABCDEF" for char in value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_hash_errors(field: str, path: Path, fields: dict[str, str]) -> list[str]:
    hash_field = f"{field}_sha256"
    expected = fields.get(hash_field, "")
    if not expected:
        return [f"{hash_field}:missing"]
    if not _is_sha256_hex(expected):
        return [f"{hash_field}:bad_hash"]
    try:
        actual = _file_sha256(path)
    except OSError:
        return [f"{hash_field}:unreadable"]
    if actual != expected.lower():
        return [f"{hash_field}:mismatch"]
    return []


def _invalid_ok_status_artifacts(
    name: str,
    fields: dict[str, str],
    current_ts: datetime,
) -> list[str]:
    required = ["log", "raw_log"]
    if name == "codex":
        required.append("last_msg")

    expected_names = _expected_ok_status_artifact_names(name, fields)
    if expected_names is None:
        return [f"{field}:bad_name" for field in required]

    invalid = _artifact_timestamp_errors(expected_names["log"], current_ts)
    for field in required:
        value = fields.get(field, "")
        if not value:
            invalid.append(f"{field}:missing")
            continue
        path = _status_artifact_path(value)
        if path.name != expected_names[field]:
            invalid.append(f"{field}:bad_name")
            continue
        if not path.exists():
            invalid.append(f"{field}:missing")
            continue
        if not _status_artifact_is_in_log_dir(path):
            invalid.append(f"{field}:outside_log_dir")
            continue
        try:
            if path.stat().st_size <= 0:
                invalid.append(f"{field}:empty")
                continue
        except OSError:
            invalid.append(f"{field}:unreadable")
            continue
        invalid.extend(_artifact_hash_errors(field, path, fields))
    return invalid


def check_scheduled_audit_status() -> CheckResult:
    """Check wrapper-owned current-status files for scheduled audit failures."""
    now = time.time()
    specs = [
        ("codex", LOG_DIR / "codex_current.log", 48 * 3600),
        (
            "hourly",
            LOG_DIR / "hourly_current.log",
            3 * 3600 if _hourly_audit_expected_now(now) else None,
        ),
    ]
    missing: list[str] = []
    failed: list[str] = []
    skipped: list[str] = []
    stale: list[str] = []
    ok: list[str] = []
    details: dict[str, dict[str, str] | str] = {}

    for name, path, stale_after_s in specs:
        if not path.exists():
            missing.append(name)
            details[name] = "missing"
            continue
        try:
            text = path.read_text().strip()
            if len(text.splitlines()) != 1:
                failed.append(f"{name}: malformed multi-line current status")
                details[name] = "malformed multi-line current status"
                continue
            fields, parse_error = _parse_current_status_line(text)
            if parse_error:
                failed.append(f"{name}: malformed current status ({parse_error})")
                details[name] = {
                    "parse_error": parse_error,
                }
                continue
            try:
                current_ts = _current_status_timestamp(text)
                age_s, age_source = now - current_ts.timestamp(), "line_timestamp"
            except ValueError as exc:
                failed.append(f"{name}: malformed current status ({exc})")
                details[name] = {
                    "parse_error": str(exc),
                }
                continue
        except Exception as exc:
            failed.append(f"{name}: unreadable ({exc})")
            details[name] = f"unreadable: {exc}"
            continue

        status = fields.get("status", "")
        rc = fields.get("rc", "")
        details[name] = {
            **fields,
            "age_s": str(round(age_s)),
            "age_source": age_source,
            "stale_after_s": str(stale_after_s) if stale_after_s is not None else "off_hours",
        }
        if not status or not rc:
            failed.append(f"{name}: malformed current status")
        elif age_s < -300:
            failed.append(f"{name}: future-dated current status")
        elif status in {"FAILED", "SETUP_FAILED"} or rc != "0":
            failed.append(f"{name}: status={status or '?'} rc={rc or '?'}")
        elif status in {"DRY_RUN", "SKIPPED_LOCK"}:
            skipped.append(name)
        elif status != "OK":
            failed.append(f"{name}: unknown status={status}")
        elif stale_after_s is not None and age_s > stale_after_s:
            stale.append(f"{name}: {round(age_s / 3600, 1)}h old")
        elif invalid_artifacts := _invalid_ok_status_artifacts(name, fields, current_ts):
            failed.append(
                f"{name}: OK current status invalid audit artifact(s): "
                + ", ".join(invalid_artifacts),
            )
        else:
            ok.append(f"{name}: status={status or '?'} rc={rc or '?'}")

    if failed:
        return CheckResult(
            "scheduled-audits",
            "fail",
            "; ".join(failed),
            details,
        )
    if stale:
        return CheckResult(
            "scheduled-audits",
            "fail",
            "stale scheduled audit status: " + "; ".join(stale),
            details,
        )
    if skipped:
        return CheckResult(
            "scheduled-audits",
            "warn",
            "scheduled audit skipped or dry-run did not run normally: "
            + ", ".join(skipped),
            details,
        )
    if missing:
        return CheckResult(
            "scheduled-audits",
            "warn",
            "missing scheduled audit status: " + ", ".join(missing),
            details,
        )
    return CheckResult(
        "scheduled-audits",
        "ok",
        "scheduled audit wrappers healthy: " + "; ".join(ok),
        details,
    )


def _invalid_monitor_status_artifacts(
    fields: dict[str, str],
    current_ts: datetime,
) -> list[str]:
    value = fields.get("log", "")
    if not value:
        return ["log:missing"]
    path = _status_artifact_path(value)
    if not (path.name.startswith("monitor_") and path.name.endswith(".log")):
        return ["log:bad_name"]
    if timestamp_errors := _artifact_timestamp_errors(path.name, current_ts):
        return timestamp_errors
    if not path.exists():
        return ["log:missing"]
    if not _status_artifact_is_in_log_dir(path):
        return ["log:outside_log_dir"]
    try:
        if path.stat().st_size <= 0:
            return ["log:empty"]
    except OSError:
        return ["log:unreadable"]
    return _artifact_hash_errors("log", path, fields)


def _monitor_status_contract_error(fields: dict[str, str]) -> str | None:
    required = ["status", "rc", "initial_rc", "final_rc", "agent_rc", "log"]
    missing = [field for field in required if not fields.get(field)]
    if missing:
        return "missing field(s): " + ", ".join(missing)

    for field in ["rc", "final_rc"]:
        if not fields[field].isdigit():
            return f"non-integer {field}={fields[field]}"
    if fields["initial_rc"] != "NA" and not fields["initial_rc"].isdigit():
        return f"non-integer initial_rc={fields['initial_rc']}"
    agent_rc = fields["agent_rc"]
    if agent_rc != "NA" and not agent_rc.isdigit():
        return f"non-integer agent_rc={agent_rc}"
    if fields["rc"] != fields["final_rc"]:
        return f"rc/final_rc mismatch: rc={fields['rc']} final_rc={fields['final_rc']}"

    status = fields["status"]
    initial_rc = fields["initial_rc"]
    final_rc = fields["final_rc"]
    if status == "OK":
        if initial_rc != "0" or final_rc != "0" or agent_rc != "NA":
            return "OK status must have initial_rc=0 final_rc=0 agent_rc=NA"
    elif status == "RECOVERED":
        if initial_rc == "0" or final_rc != "0" or agent_rc == "NA":
            return "RECOVERED status must have nonzero initial_rc, final_rc=0, numeric agent_rc"
    elif status == "STILL_UNHEALTHY":
        if initial_rc == "0" or final_rc == "0" or agent_rc == "NA":
            return (
                "STILL_UNHEALTHY status must have nonzero initial_rc, "
                "nonzero final_rc, numeric agent_rc"
            )
    elif status == "SETUP_FAILED":
        if initial_rc != "NA" or final_rc == "0" or agent_rc != "NA":
            return "SETUP_FAILED status must have initial_rc=NA, nonzero final_rc, agent_rc=NA"
    return None


def check_monitor_agent_status() -> CheckResult:
    """Check wrapper-owned current status for the remediation monitor itself."""
    if os.environ.get("HEALTH_CHECK_SKIP_MONITOR_CURRENT") == "1":
        return CheckResult(
            "monitor-agent-status",
            "ok",
            "skipped monitor current-status check in monitor-agent context",
        )

    path = LOG_DIR / "monitor_current.log"
    if not path.exists():
        return CheckResult(
            "monitor-agent-status",
            "warn",
            "missing monitor current status",
            {"monitor": "missing"},
        )

    now = time.time()
    try:
        text = path.read_text().strip()
        if len(text.splitlines()) != 1:
            return CheckResult(
                "monitor-agent-status",
                "fail",
                "monitor current status malformed: multi-line current status",
                {"monitor": "malformed multi-line current status"},
            )
        fields, parse_error = _parse_current_status_line(text)
        if parse_error:
            return CheckResult(
                "monitor-agent-status",
                "fail",
                f"monitor current status malformed: {parse_error}",
                {"parse_error": parse_error},
            )
        try:
            current_ts = _current_status_timestamp(text)
            age_s, age_source = now - current_ts.timestamp(), "line_timestamp"
        except ValueError as exc:
            return CheckResult(
                "monitor-agent-status",
                "fail",
                f"monitor current status malformed: {exc}",
                {"parse_error": str(exc)},
            )
    except Exception as exc:
        return CheckResult(
            "monitor-agent-status",
            "fail",
            f"monitor current status unreadable: {exc}",
            {"monitor": f"unreadable: {exc}"},
        )

    status = fields.get("status", "")
    rc = fields.get("rc", "")
    stale_after_s = 8 * 3600 if _hourly_audit_expected_now(now) else None
    details = {
        **fields,
        "age_s": str(round(age_s)),
        "age_source": age_source,
        "stale_after_s": str(stale_after_s) if stale_after_s is not None else "off_hours",
    }
    if contract_error := _monitor_status_contract_error(fields):
        return CheckResult(
            "monitor-agent-status",
            "fail",
            f"monitor current status malformed: {contract_error}",
            details,
        )
    if age_s < -300:
        return CheckResult(
            "monitor-agent-status",
            "fail",
            "monitor current status is future-dated",
            details,
        )
    if status not in {"OK", "RECOVERED", "STILL_UNHEALTHY", "SETUP_FAILED"}:
        return CheckResult(
            "monitor-agent-status",
            "fail",
            f"monitor current status unknown status={status}",
            details,
        )
    if status in {"SETUP_FAILED", "STILL_UNHEALTHY"} or rc != "0":
        return CheckResult(
            "monitor-agent-status",
            "fail",
            f"monitor current status={status} rc={rc}",
            details,
        )
    if stale_after_s is not None and age_s > stale_after_s:
        return CheckResult(
            "monitor-agent-status",
            "fail",
            f"stale monitor current status: {round(age_s / 3600, 1)}h old",
            details,
        )
    if invalid_artifacts := _invalid_monitor_status_artifacts(fields, current_ts):
        return CheckResult(
            "monitor-agent-status",
            "fail",
            "monitor current status invalid audit artifact(s): "
            + ", ".join(invalid_artifacts),
            details,
        )
    agent_rc = fields.get("agent_rc", "NA")
    if status == "RECOVERED" and agent_rc not in {"0", "NA"}:
        return CheckResult(
            "monitor-agent-status",
            "warn",
            f"monitor recovered but remediation agent exited rc={agent_rc}",
            details,
        )
    return CheckResult(
        "monitor-agent-status",
        "ok",
        f"monitor current status healthy: status={status} rc={rc}",
        details,
    )


def check_alpaca_monitor_timer() -> CheckResult:
    """Verify the systemd timer that runs this remediation monitor is active."""
    rc_active, active, err_active = run_cmd("systemctl is-active alpaca-monitor.timer")
    rc_enabled, enabled, err_enabled = run_cmd("systemctl is-enabled alpaca-monitor.timer")
    rc_failed, failed_state, err_failed = run_cmd("systemctl is-failed alpaca-monitor.service")

    details = {
        "active": active,
        "enabled": enabled,
        "service_failed_state": failed_state,
        "active_rc": rc_active,
        "enabled_rc": rc_enabled,
        "service_failed_rc": rc_failed,
        "active_error": err_active,
        "enabled_error": err_enabled,
        "service_failed_error": err_failed,
    }
    if active != "active":
        return CheckResult(
            "alpaca-monitor-timer",
            "fail",
            f"alpaca-monitor.timer is not active: {active or err_active or 'unknown'}",
            details,
        )
    if failed_state == "failed":
        return CheckResult(
            "alpaca-monitor-timer",
            "fail",
            "alpaca-monitor.service is failed; timer is active but remediation runs are failing",
            details,
        )
    if enabled not in {"enabled", "static"}:
        return CheckResult(
            "alpaca-monitor-timer",
            "warn",
            f"alpaca-monitor.timer is active but not enabled for restart persistence: "
            f"{enabled or err_enabled or 'unknown'}",
            details,
        )
    return CheckResult(
        "alpaca-monitor-timer",
        "ok",
        "alpaca-monitor.timer active and enabled",
        details,
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

        if r.name == "alpaca-monitor-timer" and r.status in ("warn", "fail"):
            actions.extend(_auto_fix_alpaca_monitor_timer(r))

    return actions


def _auto_fix_alpaca_monitor_timer(result: CheckResult) -> list[str]:
    """Repair only the monitor timer schedule; failed service runs need review."""
    actions = []
    if result.details.get("service_failed_state") == "failed":
        return [
            "Skipped alpaca-monitor.timer auto-fix: alpaca-monitor.service is failed",
        ]

    active = result.details.get("active")
    enabled = result.details.get("enabled")
    if not isinstance(active, str) or not isinstance(enabled, str):
        return [
            "Skipped alpaca-monitor.timer auto-fix: incomplete timer check details",
        ]

    if enabled not in {"enabled", "static"}:
        rc, _, err = run_cmd("sudo systemctl enable alpaca-monitor.timer")
        if rc == 0:
            actions.append("Enabled alpaca-monitor.timer")
        else:
            actions.append(f"Failed to enable alpaca-monitor.timer: {err}")
    if active != "active":
        rc, _, err = run_cmd("sudo systemctl start alpaca-monitor.timer")
        if rc == 0:
            actions.append("Started alpaca-monitor.timer")
        else:
            actions.append(f"Failed to start alpaca-monitor.timer: {err}")
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
        check_xgb_spy_provenance,
        check_scheduled_audit_status,
        check_monitor_agent_status,
        check_alpaca_monitor_timer,
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

    fails = sum(1 for r in results if r.status == "fail")
    warns = sum(1 for r in results if r.status == "warn")
    exit_code = 1 if fails > 0 else 0

    # Log results
    now = datetime.now(timezone.utc)
    log_entry = {
        "timestamp": now.isoformat(),
        "results": [asdict(r) for r in results],
    }

    log_file = LOG_DIR / f"health_{now.strftime('%Y%m%d')}.jsonl"
    append_jsonl_row(log_file, log_entry, default=str)

    if args.json:
        print(json.dumps(log_entry, indent=2))
        sys.exit(exit_code)

    # Human-readable output
    status_icons = {"ok": "+", "warn": "!", "fail": "X"}
    for r in results:
        icon = status_icons.get(r.status, "?")
        print(f"  [{icon}] {r.name}: {r.message}")

    if args.fix and actions:
        print(f"\nAuto-fix actions taken:")
        for a in actions:
            print(f"  -> {a}")

    print(
        f"\n{'HEALTHY' if fails == 0 else 'UNHEALTHY'}: "
        f"{len(results) - fails - warns} ok, {warns} warn, {fails} fail"
    )
    sys.exit(1 if fails > 0 else 0)


if __name__ == "__main__":
    main()
