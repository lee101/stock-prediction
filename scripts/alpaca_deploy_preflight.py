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
import sys
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
    unmodeled_live_sidecar_flags: tuple[str, ...] = ()
    xgb_ensemble_dir: Path | None = None
    xgb_ensemble_seeds: tuple[int, ...] = ()
    xgb_ensemble_min_pkl_bytes: int = 100 * 1024


@dataclass
class GitStatusSummary:
    branch_line: str = ""
    dirty_paths: list[str] = field(default_factory=list)
    ahead: int = 0
    behind: int = 0


@dataclass
class XGBEnsembleValidationReport:
    validation_error: str | None = None
    model_paths: list[str] = field(default_factory=list)
    model_sha256: list[str] = field(default_factory=list)
    train_end: str | None = None
    model_paths_parse_error: str | None = None
    model_paths_missing: bool = False


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
    unmodeled_live_sidecars: list[str] = field(default_factory=list)
    live_sidecar_parse_error: str | None = None
    xgb_model_paths: list[str] = field(default_factory=list)
    xgb_model_sha256: list[str] = field(default_factory=list)
    xgb_ensemble_train_end: str | None = None
    xgb_model_paths_parse_error: str | None = None
    xgb_model_paths_missing: bool = False
    xgb_ensemble_validation_error: str | None = None
    stale_files: list[str] = field(default_factory=list)
    restart_reasons: list[str] = field(default_factory=list)
    apply_blockers: list[str] = field(default_factory=list)
    safe_to_apply: bool = False


SPECS: dict[str, ServiceSpec] = {
    "trading-server": ServiceSpec(
        name="trading-server",
        manager="supervisor",
        actual_name="trading-server",
        config_path=Path("/etc/supervisor/conf.d/trading-server.conf"),
        repo_config_path=REPO_ROOT / "deployments" / "trading-server" / "supervisor.conf",
        watched_repo_files=(
            "config/trading_server/accounts.json",
            "deployments/trading-server/launch.sh",
            "deployments/trading-server/supervisor.conf",
            "src/trading_server/client.py",
            "src/trading_server/server.py",
            "src/trading_server/settings.py",
        ),
    ),
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
        manager="supervisor",
        actual_name="daily-rl-trader",
        config_path=Path("/etc/supervisor/conf.d/daily-rl-trader.conf"),
        repo_config_path=REPO_ROOT / "deployments" / "daily-rl-trader" / "supervisor.conf",
        watched_repo_files=(
            "trade_daily_stock_prod.py",
            "pufferlib_market/inference.py",
            "pufferlib_market/inference_daily.py",
            "pufferlib_market/checkpoint_loader.py",
            "src/daily_stock_defaults.py",
            "config/trading_server/accounts.json",
            "deployments/daily-rl-trader/launch.sh",
            "deployments/daily-rl-trader/supervisor.conf",
        ),
    ),
    "xgb-daily-trader-live": ServiceSpec(
        name="xgb-daily-trader-live",
        manager="supervisor",
        actual_name="xgb-daily-trader-live",
        config_path=Path("/etc/supervisor/conf.d/xgb-daily-trader-live.conf"),
        repo_config_path=REPO_ROOT / "deployments" / "xgb-daily-trader-live" / "supervisor.conf",
        watched_repo_files=(
            "xgbnew/backtest.py",
            "xgbnew/dataset.py",
            "xgbnew/features.py",
            "xgbnew/live_trader.py",
            "xgbnew/model.py",
            "xgbnew/trade_log.py",
            "src/alpaca_singleton.py",
            "deployments/xgb-daily-trader-live/launch.sh",
            "deployments/xgb-daily-trader-live/supervisor.conf",
        ),
        unmodeled_live_sidecar_flags=("--crypto-weekend", "--eod-deleverage"),
        xgb_ensemble_dir=REPO_ROOT / "analysis" / "xgbnew_daily" / "alltrain_ensemble_gpu",
        xgb_ensemble_seeds=(0, 7, 42, 73, 197),
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


def _repo_python() -> str:
    for candidate in (
        REPO_ROOT / ".venv" / "bin" / "python",
        REPO_ROOT / ".venv313" / "bin" / "python",
        REPO_ROOT / ".venv312" / "bin" / "python",
    ):
        if candidate.exists():
            return str(candidate)
    return sys.executable


_SHELL_ASSIGN_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
_SHELL_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")


def _strip_inline_shell_comment(line: str) -> str:
    in_single = False
    in_double = False
    escaped = False
    out: list[str] = []
    for char in line:
        if escaped:
            out.append(char)
            escaped = False
            continue
        if char == "\\":
            out.append(char)
            escaped = True
            continue
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            break
        out.append(char)
    return "".join(out).strip()


def _read_launch_script_lines(script_path: Path | None) -> list[str]:
    if script_path is None or not script_path.exists():
        return []
    try:
        return script_path.read_text().splitlines()
    except Exception:
        return []


def _expand_shell_vars(value: str, variables: dict[str, str]) -> str:
    expanded = value
    for _ in range(8):
        next_value = _SHELL_VAR_RE.sub(
            lambda match: variables.get(match.group(1) or match.group(2), match.group(0)),
            expanded,
        )
        if next_value == expanded:
            break
        expanded = next_value
    return expanded


def _read_launch_script_assignments(script_path: Path | None) -> dict[str, str]:
    variables: dict[str, str] = {}
    for line in _read_launch_script_lines(script_path):
        stripped = _strip_inline_shell_comment(line)
        if not stripped or stripped.startswith(("if ", "fi", "then", "else", "exec ")):
            continue
        match = _SHELL_ASSIGN_RE.match(stripped)
        if not match:
            continue
        name, raw_value = match.groups()
        try:
            parts = shlex.split(raw_value, posix=True)
        except ValueError:
            continue
        if len(parts) != 1:
            continue
        variables[name] = _expand_shell_vars(parts[0], variables)
    return variables


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def extract_xgb_model_paths_from_command(command: str | None) -> tuple[list[Path], str | None]:
    launch_path = _extract_launch_script_path(command)
    command_to_parse = _read_launch_script_exec_command(launch_path) or command
    if not command_to_parse:
        return [], None
    variables = _read_launch_script_assignments(launch_path)
    command_to_parse = _expand_shell_vars(command_to_parse, variables)
    try:
        tokens = shlex.split(command_to_parse)
    except ValueError as exc:
        return [], str(exc)

    raw_values: list[str] = []
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token == "--model-paths" and idx + 1 < len(tokens):
            raw_values.extend(tokens[idx + 1].split(","))
            idx += 2
            continue
        if token.startswith("--model-paths="):
            raw_values.extend(token.split("=", 1)[1].split(","))
        idx += 1

    paths = [_resolve_repo_path(value.strip()) for value in raw_values if value.strip()]
    return paths, None


def _validator_json_details(stdout: str) -> tuple[list[str], list[str], str | None]:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"validator JSON invalid: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("validator JSON must be an object")
    raw_models = payload.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise ValueError("validator JSON missing non-empty models")
    paths: list[str] = []
    hashes: list[str] = []
    for raw_model in raw_models:
        if not isinstance(raw_model, dict):
            raise ValueError("validator JSON models contains non-object entry")
        path = raw_model.get("path")
        sha256 = raw_model.get("sha256")
        if not isinstance(path, str) or not path:
            raise ValueError("validator JSON model path invalid")
        if (
            not isinstance(sha256, str)
            or len(sha256.strip()) != 64
            or any(char not in "0123456789abcdefABCDEF" for char in sha256.strip())
        ):
            raise ValueError(f"validator JSON sha256 invalid for {path}")
        paths.append(str(_resolve_repo_path(path).resolve(strict=False)))
        hashes.append(sha256.strip().lower())
    train_end = payload.get("train_end")
    if train_end is not None and not isinstance(train_end, str):
        raise ValueError("validator JSON train_end invalid")
    return paths, hashes, train_end


def validate_xgb_ensemble_for_spec(
    spec: ServiceSpec,
    command: str | None = None,
) -> XGBEnsembleValidationReport:
    if spec.xgb_ensemble_dir is None:
        return XGBEnsembleValidationReport()
    script = REPO_ROOT / "scripts" / "validate_xgb_ensemble.py"
    if not script.exists():
        return XGBEnsembleValidationReport(validation_error=f"validator missing: {script}")
    seeds = ",".join(str(seed) for seed in spec.xgb_ensemble_seeds)
    model_paths, model_paths_parse_error = extract_xgb_model_paths_from_command(command)
    cmd = [
        _repo_python(),
        str(script),
        "--seeds",
        seeds,
        "--min-pkl-bytes",
        str(spec.xgb_ensemble_min_pkl_bytes),
        "--json",
    ]
    if model_paths_parse_error is not None:
        return XGBEnsembleValidationReport(model_paths_parse_error=model_paths_parse_error)
    if model_paths:
        cmd.extend([
            "--model-paths",
            ",".join(str(path) for path in model_paths),
            "--require-manifest",
        ])
    else:
        if command:
            return XGBEnsembleValidationReport(model_paths_missing=True)
        cmd.insert(2, str(spec.xgb_ensemble_dir))
    result = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        try:
            json_paths, json_hashes, train_end = _validator_json_details(result.stdout)
        except ValueError as exc:
            return XGBEnsembleValidationReport(validation_error=str(exc))
        return XGBEnsembleValidationReport(
            model_paths=json_paths,
            model_sha256=json_hashes,
            train_end=train_end,
        )
    output = "\n".join(
        line.strip()
        for line in (result.stdout, result.stderr)
        if line and line.strip()
    )
    return XGBEnsembleValidationReport(
        validation_error=output or f"validator exited {result.returncode}",
        model_paths=[str(path) for path in model_paths],
    )


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


def _extract_launch_script_path(command: str | None) -> Path | None:
    if not command:
        return None
    match = re.search(r"exec\s+((?:/[^'\" ]+|[^'\" ]+)?launch\.sh)\b", command)
    if not match:
        return None
    path = Path(match.group(1))
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _read_launch_script_exec_command(script_path: Path | None) -> str | None:
    if script_path is None or not script_path.exists():
        return None
    try:
        lines = script_path.read_text().splitlines()
    except Exception:
        return None

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("exec "):
            continue
        parts = [stripped[len("exec "):].rstrip("\\").strip()]
        while stripped.endswith("\\") and idx + 1 < len(lines):
            idx += 1
            stripped = lines[idx].strip()
            parts.append(stripped.rstrip("\\").strip())
        return " ".join(part for part in parts if part)
    return None


def scan_unmodeled_live_sidecars_from_command(
    command: str | None,
    flags: tuple[str, ...],
) -> tuple[list[str], str | None]:
    if not command or not flags:
        return [], None
    command_to_parse = _read_launch_script_exec_command(_extract_launch_script_path(command)) or command
    try:
        tokens = shlex.split(command_to_parse)
    except ValueError as exc:
        return [], str(exc)
    return [flag for flag in flags if flag in tokens], None


def unmodeled_live_sidecars_from_command(
    command: str | None,
    flags: tuple[str, ...],
) -> list[str]:
    sidecars, _parse_error = scan_unmodeled_live_sidecars_from_command(command, flags)
    return sidecars


def runtime_matches_configured_command(runtime_cmd: str | None, configured_cmd: str | None) -> bool:
    if not runtime_cmd or not configured_cmd:
        return False
    if runtime_cmd == configured_cmd:
        return True
    launch_exec = _read_launch_script_exec_command(_extract_launch_script_path(configured_cmd))
    if not launch_exec:
        return False
    variables = _read_launch_script_assignments(_extract_launch_script_path(configured_cmd))
    expanded_launch_exec = _expand_shell_vars(launch_exec, variables)
    try:
        return shlex.split(expanded_launch_exec) == shlex.split(runtime_cmd)
    except ValueError:
        return expanded_launch_exec == runtime_cmd


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
    unmodeled_live_sidecars, live_sidecar_parse_error = scan_unmodeled_live_sidecars_from_command(
        repo_configured_cmd or configured_cmd,
        spec.unmodeled_live_sidecar_flags,
    )
    xgb_validation = validate_xgb_ensemble_for_spec(spec, repo_configured_cmd or configured_cmd)

    restart_reasons: list[str] = []
    if stale_files:
        restart_reasons.append("watched_files_newer_than_process")
    if runtime_cmd and configured_cmd and not runtime_matches_configured_command(
        runtime_cmd,
        configured_cmd,
    ):
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
    if unmodeled_live_sidecars:
        apply_blockers.append(
            "unmodeled_live_sidecars:" + ",".join(unmodeled_live_sidecars)
        )
    if live_sidecar_parse_error is not None:
        apply_blockers.append("live_sidecar_parse_error")
    if xgb_validation.model_paths_parse_error is not None:
        apply_blockers.append("xgb_model_paths_parse_error")
    if xgb_validation.model_paths_missing:
        apply_blockers.append("xgb_model_paths_missing")
    if xgb_validation.validation_error is not None:
        apply_blockers.append("xgb_ensemble_validation_failed")

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
        unmodeled_live_sidecars=unmodeled_live_sidecars,
        live_sidecar_parse_error=live_sidecar_parse_error,
        xgb_model_paths=xgb_validation.model_paths,
        xgb_model_sha256=xgb_validation.model_sha256,
        xgb_ensemble_train_end=xgb_validation.train_end,
        xgb_model_paths_parse_error=xgb_validation.model_paths_parse_error,
        xgb_model_paths_missing=xgb_validation.model_paths_missing,
        xgb_ensemble_validation_error=xgb_validation.validation_error,
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
            "Service(s) to inspect. Defaults to trading-server + daily-rl-trader "
            "+ xgb-daily-trader-live + llm-stock-trader + unified-stock-trader "
            "+ unified-orchestrator."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Restart services if preflight says it is safe.",
    )
    parser.add_argument(
        "--fail-on-unsafe",
        action="store_true",
        help="Return non-zero when any inspected service is unsafe, without applying changes.",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Ignore dirty-tree blockers outside the watched-file list.",
    )
    parser.add_argument(
        "--allow-unmodeled-live-sidecars",
        action="store_true",
        help="Ignore apply blockers for live sidecar flags that are not represented "
             "in the service's promotion/evaluation artifact.",
    )
    parser.add_argument(
        "--allow-invalid-xgb-ensemble",
        action="store_true",
        help="Ignore XGB model-load validation failures. Break-glass only.",
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
        if report.unmodeled_live_sidecars:
            lines.append(
                "unmodeled_live_sidecars: "
                + ",".join(report.unmodeled_live_sidecars)
            )
        if report.live_sidecar_parse_error:
            lines.append(f"live_sidecar_parse_error: {report.live_sidecar_parse_error}")
        if report.xgb_model_paths:
            lines.append(f"xgb_model_paths: {','.join(report.xgb_model_paths)}")
        if report.xgb_model_sha256:
            lines.append(f"xgb_model_sha256: {','.join(report.xgb_model_sha256)}")
        if report.xgb_ensemble_train_end:
            lines.append(f"xgb_ensemble_train_end: {report.xgb_ensemble_train_end}")
        if report.xgb_model_paths_parse_error:
            lines.append(f"xgb_model_paths_parse_error: {report.xgb_model_paths_parse_error}")
        if report.xgb_model_paths_missing:
            lines.append("xgb_model_paths_missing: true")
        if report.xgb_ensemble_validation_error:
            lines.append(
                "xgb_ensemble_validation_error: "
                + report.xgb_ensemble_validation_error.replace("\n", " | ")
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
        "trading-server",
        "daily-rl-trader",
        "xgb-daily-trader-live",
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
    if args.allow_unmodeled_live_sidecars:
        for report in reports:
            report.apply_blockers = [
                item for item in report.apply_blockers if not item.startswith("unmodeled_live_sidecars:")
            ]
            report.safe_to_apply = not report.apply_blockers
    if args.allow_invalid_xgb_ensemble:
        for report in reports:
            report.apply_blockers = [
                item for item in report.apply_blockers
                if item not in {
                    "xgb_ensemble_validation_failed",
                    "xgb_model_paths_parse_error",
                    "xgb_model_paths_missing",
                }
            ]
            report.safe_to_apply = not report.apply_blockers

    unsafe = [report.service for report in reports if not report.safe_to_apply]

    if args.apply:
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
    elif args.fail_on_unsafe and unsafe:
        if args.json:
            payload = {
                "git": asdict(git_status),
                "reports": [asdict(report) for report in reports],
                "check_error": f"unsafe:{','.join(unsafe)}",
            }
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(render_text(git_status, reports))
            print("")
            print(f"Unsafe: {', '.join(unsafe)}")
        return 2

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
