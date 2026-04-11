#!/usr/bin/env python3
"""Preflight and optional apply helper for Alpaca live-service deploys.

This answers two different questions:
1. Should this service be restarted because config/code drifted?
2. Is it actually safe to restart it from the current checkout?

The main blocker this tries to make explicit is the dirty-tree problem:
if a live service runs directly from the repo checkout, a restart deploys
every dirty Python file in that checkout, not only the small fix you meant
to apply.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SERVICE_CONFIG_PATH = REPO_ROOT / "unified_orchestrator" / "service_config.json"


@dataclass(frozen=True)
class ServiceSpec:
    name: str
    manager: str
    actual_name: str
    config_path: Path
    repo_config_path: Path | None = None
    watched_repo_files: tuple[str, ...] = ()
    symbols_flag: str | None = None
    ownership_service_name: str | None = None


@dataclass
class GitStatusSummary:
    branch_line: str = ""
    dirty_paths: list[str] = field(default_factory=list)
    ahead: int = 0
    behind: int = 0


@dataclass
class ServiceReport:
    service: str
    manager: str
    pid: int | None
    running: bool
    process_start_utc: str | None
    runtime_cmd: str | None
    configured_cmd: str | None
    repo_configured_cmd: str | None
    runtime_symbols: list[str] = field(default_factory=list)
    configured_symbols: list[str] = field(default_factory=list)
    owned_symbols: list[str] = field(default_factory=list)
    runtime_symbols_outside_ownership: list[str] = field(default_factory=list)
    configured_symbols_outside_ownership: list[str] = field(default_factory=list)
    stale_files: list[str] = field(default_factory=list)
    restart_reasons: list[str] = field(default_factory=list)
    apply_blockers: list[str] = field(default_factory=list)
    safe_to_apply: bool = False


SPECS: dict[str, ServiceSpec] = {
    "unified-stock-trader": ServiceSpec(
        name="unified-stock-trader",
        manager="supervisor",
        actual_name="unified-stock-trader",
        config_path=Path("/etc/supervisor/conf.d/unified-stock-trader.conf"),
        repo_config_path=REPO_ROOT / "deployments" / "unified-stock-trader" / "supervisor.conf",
        watched_repo_files=(
            "unified_hourly_experiment/trade_unified_hourly_meta.py",
            "deployments/unified-stock-trader/supervisor.conf",
            "unified_orchestrator/service_config.json",
        ),
        symbols_flag="--stock-symbols",
        ownership_service_name="trade-unified-hourly-meta",
    ),
    "unified-orchestrator": ServiceSpec(
        name="unified-orchestrator",
        manager="systemd",
        actual_name="unified-orchestrator.service",
        config_path=Path("/etc/systemd/system/unified-orchestrator.service"),
        watched_repo_files=(
            "unified_orchestrator/orchestrator.py",
            "unified_orchestrator/service_config.json",
        ),
    ),
    "daily-rl-trader": ServiceSpec(
        name="daily-rl-trader",
        manager="systemd",
        actual_name="daily-rl-trader.service",
        config_path=Path("/etc/systemd/system/daily-rl-trader.service"),
        repo_config_path=REPO_ROOT / "systemd" / "daily-rl-trader.service",
        watched_repo_files=(
            "trade_daily_stock_prod.py",
            "pufferlib_market/inference.py",
            "pufferlib_market/inference_daily.py",
            "pufferlib_market/checkpoint_loader.py",
            "systemd/daily-rl-trader.service",
            "unified_orchestrator/service_config.json",
        ),
        ownership_service_name="daily-rl-trader",
    ),
    "llm-stock-trader": ServiceSpec(
        name="llm-stock-trader",
        manager="supervisor",
        actual_name="llm-stock-trader",
        config_path=Path("/etc/supervisor/conf.d/llm-stock-trader.conf"),
        repo_config_path=REPO_ROOT / "deployments" / "llm-stock-trader" / "launch.sh",
        watched_repo_files=(
            "unified_orchestrator/orchestrator.py",
            "deployments/llm-stock-trader/launch.sh",
            "unified_orchestrator/service_config.json",
            "llm_hourly_trader/providers.py",
            "llm_hourly_trader/gemini_wrapper.py",
        ),
        symbols_flag="--stock-symbols",
        ownership_service_name="llm-stock-trader",
    ),
}


def run_text(cmd: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def normalize_symbols(values: list[str] | tuple[str, ...] | set[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        sym = str(value).strip().upper()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
    return out


def extract_flag_csv_values(cmdline: str | None, flag: str) -> list[str]:
    if not cmdline:
        return []
    tokens = shlex.split(cmdline)
    values: list[str] = []
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token == flag and idx + 1 < len(tokens):
            idx += 1
            while idx < len(tokens) and not tokens[idx].startswith("--"):
                values.extend(tokens[idx].split(","))
                idx += 1
            continue
        if token.startswith(flag + "="):
            values.extend(token.split("=", 1)[1].split(","))
        idx += 1
    return normalize_symbols(values)


def parse_git_status_porcelain(text: str) -> GitStatusSummary:
    lines = [line.rstrip("\n") for line in text.splitlines()]
    summary = GitStatusSummary()
    for line in lines:
        if line.startswith("## "):
            summary.branch_line = line[3:]
            match = re.search(r"\[ahead (\d+)(?:, behind (\d+))?\]", line)
            if match:
                summary.ahead = int(match.group(1) or 0)
                summary.behind = int(match.group(2) or 0)
            else:
                match = re.search(r"\[behind (\d+)\]", line)
                if match:
                    summary.behind = int(match.group(1) or 0)
            continue
        if len(line) < 4:
            continue
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        summary.dirty_paths.append(path)
    return summary


def repo_relative_dirty_paths_outside_watchlist(
    dirty_paths: list[str],
    watched_repo_files: tuple[str, ...],
) -> list[str]:
    watched = {path.rstrip("/") for path in watched_repo_files}
    return sorted(path for path in dirty_paths if path.rstrip("/") not in watched)


def get_service_owned_symbols(service_name: str) -> list[str]:
    try:
        payload = json.loads(SERVICE_CONFIG_PATH.read_text())
    except Exception:
        return []
    service = payload.get(service_name) or {}
    return normalize_symbols(service.get("stock_symbols", []))


def symbols_outside_ownership(runtime_or_configured: list[str], owned_symbols: list[str]) -> list[str]:
    if not runtime_or_configured:
        return []
    owned = set(normalize_symbols(owned_symbols))
    return sorted(symbol for symbol in normalize_symbols(runtime_or_configured) if symbol not in owned)


def _read_runtime_cmd(pid: int | None) -> str | None:
    if not pid:
        return None
    path = Path("/proc") / str(pid) / "cmdline"
    if not path.exists():
        return None
    raw = path.read_bytes().replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
    return raw or None


def _read_process_start_utc(pid: int | None) -> str | None:
    if not pid:
        return None
    proc_dir = Path("/proc") / str(pid)
    if not proc_dir.exists():
        return None
    ts = proc_dir.stat().st_ctime
    return datetime.fromtimestamp(ts, tz=UTC).isoformat()


def files_newer_than_process(
    pid: int | None,
    watched_paths: list[Path],
) -> list[str]:
    if not pid:
        return []
    proc_dir = Path("/proc") / str(pid)
    if not proc_dir.exists():
        return []
    proc_start = proc_dir.stat().st_ctime
    stale: list[str] = []
    for path in watched_paths:
        if not path.exists():
            continue
        if path.stat().st_mtime > proc_start:
            stale.append(str(path))
    return sorted(stale)


def _parse_supervisor_pid(status_text: str) -> int | None:
    match = re.search(r"\bRUNNING\s+pid\s+(\d+)\b", status_text)
    return int(match.group(1)) if match else None


def get_supervisor_pid(program: str) -> int | None:
    try:
        output = run_text(["sudo", "supervisorctl", "status", program])
    except Exception:
        return None
    return _parse_supervisor_pid(output)


def get_systemd_pid(unit: str) -> int | None:
    try:
        output = run_text(["systemctl", "show", "-p", "MainPID", "--value", unit]).strip()
    except Exception:
        return None
    if not output or output == "0":
        return None
    return int(output)


def read_systemd_execstart(unit: str) -> str | None:
    try:
        output = run_text(["systemctl", "cat", unit])
    except Exception:
        return None
    return _extract_systemd_execstart(output)


def _extract_systemd_execstart(text: str) -> str | None:
    for line in text.splitlines():
        if line.startswith("ExecStart="):
            return line.split("=", 1)[1].strip()
    return None


def read_supervisor_command(conf_path: Path) -> str | None:
    try:
        text = conf_path.read_text()
    except Exception:
        return None
    for line in text.splitlines():
        if line.startswith("command="):
            return line.split("=", 1)[1].strip()
    return None


def read_repo_configured_command(spec: ServiceSpec) -> str | None:
    if spec.repo_config_path is None or not spec.repo_config_path.exists():
        return None
    try:
        text = spec.repo_config_path.read_text()
    except Exception:
        return None
    if spec.manager == "systemd":
        return _extract_systemd_execstart(text)
    return read_supervisor_command(spec.repo_config_path)


def build_service_report(spec: ServiceSpec, git_status: GitStatusSummary) -> ServiceReport:
    if spec.manager == "supervisor":
        pid = get_supervisor_pid(spec.actual_name)
        configured_cmd = read_supervisor_command(spec.config_path)
    else:
        pid = get_systemd_pid(spec.actual_name)
        configured_cmd = read_systemd_execstart(spec.actual_name)
    repo_configured_cmd = read_repo_configured_command(spec)

    runtime_cmd = _read_runtime_cmd(pid)
    process_start_utc = _read_process_start_utc(pid)
    watched_paths = [REPO_ROOT / path for path in spec.watched_repo_files]
    stale_files = files_newer_than_process(pid, [*watched_paths, spec.config_path])

    runtime_symbols = (
        extract_flag_csv_values(runtime_cmd, spec.symbols_flag)
        if spec.symbols_flag
        else []
    )
    configured_symbols = (
        extract_flag_csv_values(configured_cmd, spec.symbols_flag)
        if spec.symbols_flag
        else []
    )
    owned_symbols = (
        get_service_owned_symbols(spec.ownership_service_name)
        if spec.ownership_service_name
        else []
    )
    runtime_symbols_outside_ownership = symbols_outside_ownership(runtime_symbols, owned_symbols)
    configured_symbols_outside_ownership = symbols_outside_ownership(configured_symbols, owned_symbols)

    restart_reasons: list[str] = []
    if stale_files:
        restart_reasons.append("watched_files_newer_than_process")
    if runtime_cmd and configured_cmd and runtime_cmd != configured_cmd:
        restart_reasons.append("runtime_command_differs_from_config")
    if repo_configured_cmd and configured_cmd and repo_configured_cmd != configured_cmd:
        restart_reasons.append("installed_config_differs_from_repo")
    if configured_symbols_outside_ownership:
        restart_reasons.append("configured_symbols_do_not_match_service_ownership")
    if runtime_symbols_outside_ownership:
        restart_reasons.append("runtime_symbols_do_not_match_service_ownership")

    apply_blockers: list[str] = []
    dirty_outside_watchlist = repo_relative_dirty_paths_outside_watchlist(
        git_status.dirty_paths,
        spec.watched_repo_files,
    )
    if dirty_outside_watchlist:
        apply_blockers.append(
            f"dirty_repo_outside_watchlist:{len(dirty_outside_watchlist)}"
        )
    if git_status.behind > 0:
        apply_blockers.append(f"branch_behind_origin:{git_status.behind}")

    return ServiceReport(
        service=spec.name,
        manager=spec.manager,
        pid=pid,
        running=pid is not None,
        process_start_utc=process_start_utc,
        runtime_cmd=runtime_cmd,
        configured_cmd=configured_cmd,
        repo_configured_cmd=repo_configured_cmd,
        runtime_symbols=runtime_symbols,
        configured_symbols=configured_symbols,
        owned_symbols=owned_symbols,
        runtime_symbols_outside_ownership=runtime_symbols_outside_ownership,
        configured_symbols_outside_ownership=configured_symbols_outside_ownership,
        stale_files=stale_files,
        restart_reasons=restart_reasons,
        apply_blockers=apply_blockers,
        safe_to_apply=not apply_blockers,
    )


def apply_service(spec: ServiceSpec) -> None:
    if spec.manager == "supervisor":
        subprocess.run(["sudo", "supervisorctl", "reread"], check=True)
        subprocess.run(["sudo", "supervisorctl", "update", spec.actual_name], check=True)
        subprocess.run(["sudo", "supervisorctl", "restart", spec.actual_name], check=True)
        return
    subprocess.run(["sudo", "systemctl", "restart", spec.actual_name], check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--service",
        action="append",
        choices=sorted(SPECS.keys()),
        help=(
            "Service(s) to inspect. Defaults to daily-rl-trader + llm-stock-trader "
            "+ unified-stock-trader + unified-orchestrator."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Restart services if preflight says it is safe.",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Ignore dirty-tree blockers outside the watched-file list.",
    )
    return parser.parse_args()


def render_text(git_status: GitStatusSummary, reports: list[ServiceReport]) -> str:
    lines = [
        f"Branch: {git_status.branch_line}",
        f"Dirty paths: {len(git_status.dirty_paths)}",
    ]
    for report in reports:
        lines.append("")
        lines.append(f"[{report.service}]")
        lines.append(f"running: {report.running} pid={report.pid}")
        if report.process_start_utc:
            lines.append(f"process_start_utc: {report.process_start_utc}")
        if report.runtime_symbols:
            lines.append(f"runtime_symbols: {','.join(report.runtime_symbols)}")
        if report.configured_symbols:
            lines.append(f"configured_symbols: {','.join(report.configured_symbols)}")
        if report.owned_symbols:
            lines.append(f"owned_symbols: {','.join(report.owned_symbols)}")
        if report.repo_configured_cmd and report.repo_configured_cmd != report.configured_cmd:
            lines.append(f"repo_configured_cmd: {report.repo_configured_cmd}")
        if report.runtime_symbols_outside_ownership:
            lines.append(
                "runtime_symbols_outside_ownership: "
                + ",".join(report.runtime_symbols_outside_ownership)
            )
        if report.configured_symbols_outside_ownership:
            lines.append(
                "configured_symbols_outside_ownership: "
                + ",".join(report.configured_symbols_outside_ownership)
            )
        lines.append(
            "restart_reasons: "
            + (", ".join(report.restart_reasons) if report.restart_reasons else "none")
        )
        lines.append(
            "apply_blockers: "
            + (", ".join(report.apply_blockers) if report.apply_blockers else "none")
        )
        lines.append(f"safe_to_apply: {report.safe_to_apply}")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    targets = args.service or [
        "daily-rl-trader",
        "llm-stock-trader",
        "unified-stock-trader",
        "unified-orchestrator",
    ]
    git_output = run_text(["git", "status", "--porcelain=v1", "--branch"], cwd=REPO_ROOT)
    git_status = parse_git_status_porcelain(git_output)

    reports = [build_service_report(SPECS[name], git_status) for name in targets]
    if args.allow_dirty:
        for report in reports:
            report.apply_blockers = [
                item for item in report.apply_blockers if not item.startswith("dirty_repo_outside_watchlist:")
            ]
            report.safe_to_apply = not report.apply_blockers

    if args.apply:
        unsafe = [report.service for report in reports if not report.safe_to_apply]
        if unsafe:
            if args.json:
                payload = {
                    "git": asdict(git_status),
                    "reports": [asdict(report) for report in reports],
                    "apply_error": f"unsafe_to_apply:{','.join(unsafe)}",
                }
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                print(render_text(git_status, reports))
                print("")
                print(f"Refusing to apply: {', '.join(unsafe)}")
            return 2
        for name in targets:
            apply_service(SPECS[name])
        # Refresh after apply.
        reports = [build_service_report(SPECS[name], git_status) for name in targets]

    if args.json:
        payload = {
            "git": asdict(git_status),
            "reports": [asdict(report) for report in reports],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print(render_text(git_status, reports))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
