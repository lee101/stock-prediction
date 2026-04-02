#!/usr/bin/env python3
"""Run the daily stock planner remotely on RunPod and return the JSON plan.

This wrapper is intentionally planning-only. It runs the existing daily stock
trader in `--dry-run --print-payload` mode on a remote pod, then prints the
resulting JSON locally for a separate local execution layer to consume.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import trade_daily_stock_prod as daily_stock
from src.runpod_client import PodConfig, RunPodClient, build_gpu_fallback_types, is_capacity_error

DEFAULT_REMOTE_WORKSPACE = "/workspace/stock-prediction"
DEFAULT_REMOTE_VENV = ".venv"
_SSH_OPTS = ["-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes"]


@dataclass(frozen=True)
class PlannerExecutionPlan:
    gpu_preferences: tuple[str, ...]
    checkpoint_local: Path
    checkpoint_remote: str
    data_source: str
    data_dir_local: Path | None
    data_dir_remote: str
    symbols_file_local: Path | None
    symbols_file_remote: str | None
    extra_checkpoints_local: tuple[Path, ...]
    extra_checkpoints_remote: tuple[str, ...]
    symbol_source: str
    symbols: tuple[str, ...]
    removed_duplicate_symbols: tuple[str, ...]
    ignored_symbol_inputs: tuple[str, ...]
    min_open_confidence: float
    min_open_value_estimate: float
    forward_env_names: tuple[str, ...]
    present_forward_env: tuple[str, ...]
    missing_forward_env: tuple[str, ...]
    planner_command_preview: str
    bootstrap_enabled: bool
    keep_pod: bool
    errors: tuple[str, ...]


def _resolve_local_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else (REPO / path).resolve()


def _relative_remote_path(local_path: Path, remote_workspace: str) -> str:
    try:
        rel = local_path.resolve().relative_to(REPO)
        return str(Path(remote_workspace) / rel)
    except ValueError:
        return str(Path(remote_workspace) / ".external" / local_path.name)


def _rsync_path(*, ssh_host: str, ssh_port: int, local_path: Path, remote_path: str, delete: bool = False) -> None:
    local_path = local_path.resolve()
    target = f"root@{ssh_host}:{remote_path}"
    parent_cmd = f"mkdir -p {shlex.quote(str(Path(remote_path).parent))}"
    mkdir_result = _ssh_run(
        ssh_host=ssh_host,
        ssh_port=ssh_port,
        remote_cmd=parent_cmd,
        capture_output=True,
    )
    if mkdir_result.returncode != 0:
        raise RuntimeError(
            f"Failed to create remote directory for {remote_path}:\n{mkdir_result.stderr}"
        )
    cmd = [
        "rsync",
        "-az",
        "-e",
        f"ssh {' '.join(_SSH_OPTS)} -p {ssh_port}",
    ]
    if delete:
        cmd.append("--delete")
    if local_path.is_dir():
        cmd.extend([f"{local_path}/", f"{target}/"])
    else:
        cmd.extend([str(local_path), target])
    _run_checked_subprocess(cmd, description=f"Failed to rsync {local_path} to {remote_path}")


def _rsync_repo(*, ssh_host: str, ssh_port: int, remote_workspace: str) -> None:
    cmd = [
        "rsync",
        "-az",
        "--delete",
        "--exclude",
        ".git/",
        "--exclude",
        "__pycache__/",
        "--exclude",
        ".venv*/",
        "--exclude",
        "trainingdata/",
        "--exclude",
        "*.pyc",
        "-e",
        f"ssh {' '.join(_SSH_OPTS)} -p {ssh_port}",
        f"{REPO}/",
        f"root@{ssh_host}:{remote_workspace}/",
    ]
    _run_checked_subprocess(cmd, description=f"Failed to rsync repository to {remote_workspace}")


def _run_checked_subprocess(cmd: list[str], *, description: str) -> None:
    result = subprocess.run(cmd, check=False, text=True, capture_output=True)
    if result.returncode == 0:
        return

    rendered_cmd = shlex.join(cmd)
    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    message = [
        f"{description} (exit {result.returncode})",
        f"command: {rendered_cmd}",
    ]
    if stdout:
        message.append(f"stdout:\n{stdout}")
    if stderr:
        message.append(f"stderr:\n{stderr}")
    raise RuntimeError("\n".join(message))


def _ssh_run(
    *,
    ssh_host: str,
    ssh_port: int,
    remote_cmd: str,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        "ssh",
        *_SSH_OPTS,
        "-p",
        str(ssh_port),
        f"root@{ssh_host}",
        remote_cmd,
    ]
    return subprocess.run(cmd, check=False, text=True, capture_output=capture_output)


def _remote_bootstrap_cmd(*, remote_workspace: str, remote_venv: str) -> str:
    return (
        f"mkdir -p {shlex.quote(remote_workspace)} && "
        f"cd {shlex.quote(remote_workspace)} && "
        f"python3 -m venv {shlex.quote(remote_venv)} && "
        f". {shlex.quote(remote_venv)}/bin/activate && "
        "python -m pip install -U pip setuptools wheel >/tmp/runpod_daily_pip.log 2>&1 && "
        "python -m pip install -e . >>/tmp/runpod_daily_pip.log 2>&1"
    )


def _extract_json(stdout: str) -> dict[str, object]:
    text = stdout.strip()
    if not text:
        raise RuntimeError("Remote planner returned empty stdout")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.rfind("\n{")
        if start >= 0:
            try:
                payload = json.loads(text[start + 1 :])
            except json.JSONDecodeError as exc:
                excerpt = text[-400:]
                raise RuntimeError(
                    "Remote planner did not emit valid JSON payload.\n"
                    f"Output excerpt:\n{excerpt}"
                ) from exc
        else:
            excerpt = text[-400:]
            raise RuntimeError(
                "Remote planner did not emit valid JSON payload.\n"
                f"Output excerpt:\n{excerpt}"
            ) from None

    if not isinstance(payload, dict):
        raise RuntimeError(
            "Remote planner JSON payload must be an object, "
            f"received {type(payload).__name__}"
        )
    return payload


def _split_gpu_fallbacks(raw_value: str | None) -> list[str] | None:
    if raw_value is None:
        return None
    return [token.strip() for token in raw_value.split(",") if token.strip()]


def _resolve_extra_checkpoint_inputs(args: argparse.Namespace) -> list[str] | None:
    if args.no_ensemble:
        return None
    if args.extra_checkpoints is None:
        return list(daily_stock.DEFAULT_EXTRA_CHECKPOINTS)
    return list(args.extra_checkpoints)


def _forward_env_assignments(
    names: list[str] | tuple[str, ...],
    *,
    include_values: bool,
) -> tuple[list[str], tuple[str, ...], tuple[str, ...]]:
    assignments: list[str] = []
    present: list[str] = []
    missing: list[str] = []
    for name in names:
        env_name = str(name).strip()
        if not env_name:
            continue
        value = os.getenv(env_name)
        if value is None or value == "":
            missing.append(env_name)
            continue
        present.append(env_name)
        if include_values:
            assignments.append(f"{env_name}={shlex.quote(value)}")
        else:
            assignments.append(f"{env_name}=${env_name}")
    return assignments, tuple(present), tuple(missing)


def _build_remote_planner_command(
    *,
    remote_workspace: str,
    remote_venv: str,
    data_source: str,
    data_dir_remote: str,
    symbols_file_remote: str | None,
    checkpoint_remote: str,
    extra_checkpoints_remote: tuple[str, ...],
    symbols: tuple[str, ...],
    min_open_confidence: float,
    min_open_value_estimate: float,
    env_assignments: list[str] | tuple[str, ...],
) -> str:
    planner_cmd = [
        f"cd {shlex.quote(remote_workspace)}",
        f". {shlex.quote(remote_venv)}/bin/activate",
    ]
    python_cmd_parts = list(env_assignments)
    python_cmd_parts.extend(
        [
            "python trade_daily_stock_prod.py",
            "--once",
            "--dry-run",
            "--print-payload",
            f"--data-source {shlex.quote(data_source)}",
            f"--data-dir {shlex.quote(data_dir_remote)}",
            f"--checkpoint {shlex.quote(checkpoint_remote)}",
            f"--min-open-confidence {min_open_confidence:g}",
            f"--min-open-value-estimate {min_open_value_estimate:g}",
        ]
    )
    if not extra_checkpoints_remote and data_source in {"local", "alpaca"}:
        python_cmd_parts.append("--no-ensemble")
    elif extra_checkpoints_remote:
        python_cmd_parts.append("--extra-checkpoints")
        python_cmd_parts.extend(shlex.quote(path) for path in extra_checkpoints_remote)
    if symbols_file_remote:
        python_cmd_parts.extend(["--symbols-file", shlex.quote(symbols_file_remote)])
    elif symbols:
        python_cmd_parts.append("--symbols")
        python_cmd_parts.extend(shlex.quote(symbol) for symbol in symbols)
    planner_cmd.append(" ".join(python_cmd_parts))
    return " && ".join(planner_cmd)


def _resolve_symbols_config(
    *,
    raw_symbols: tuple[str, ...],
    symbols_file_local: Path | None,
    errors: list[str],
) -> tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    symbol_source = "symbols_file" if symbols_file_local is not None else "cli"

    if symbols_file_local is not None:
        if not symbols_file_local.exists():
            return symbol_source, (), (), ()
        try:
            symbol_inputs = daily_stock._load_symbols_file(symbols_file_local)
        except (OSError, ValueError) as exc:
            errors.append(f"Invalid symbols file {symbols_file_local}: {exc}")
            return symbol_source, (), (), ()
    else:
        symbol_inputs = list(raw_symbols)

    try:
        normalized, removed_duplicates, ignored_inputs = daily_stock._normalize_symbols(symbol_inputs)
    except ValueError as exc:
        errors.append(f"Invalid symbols configuration: {exc}")
        return symbol_source, (), (), ()

    return (
        symbol_source,
        tuple(normalized),
        tuple(removed_duplicates),
        tuple(ignored_inputs),
    )


def _build_execution_plan(args: argparse.Namespace) -> PlannerExecutionPlan:
    gpu_fallbacks = _split_gpu_fallbacks(args.gpu_fallbacks)
    if gpu_fallbacks is None:
        gpu_preferences = tuple(build_gpu_fallback_types(args.gpu_type))
    else:
        gpu_preferences = tuple(build_gpu_fallback_types(args.gpu_type, gpu_fallbacks))

    checkpoint_local = _resolve_local_path(args.checkpoint)
    checkpoint_remote = _relative_remote_path(checkpoint_local, args.remote_workspace)

    if args.data_source == "local":
        data_dir_local: Path | None = _resolve_local_path(args.data_dir)
        data_dir_remote = _relative_remote_path(data_dir_local, args.remote_workspace)
    else:
        data_dir_local = None
        data_dir_remote = args.data_dir

    symbols_file_local = _resolve_local_path(args.symbols_file) if args.symbols_file else None
    symbols_file_remote = (
        _relative_remote_path(symbols_file_local, args.remote_workspace)
        if symbols_file_local is not None
        else None
    )

    extra_checkpoint_inputs = _resolve_extra_checkpoint_inputs(args)
    extra_checkpoints_local = tuple(
        _resolve_local_path(path_str) for path_str in (extra_checkpoint_inputs or [])
    )
    extra_checkpoints_remote = tuple(
        _relative_remote_path(path, args.remote_workspace) for path in extra_checkpoints_local
    )

    _, present_forward_env, missing_forward_env = _forward_env_assignments(
        tuple(args.forward_env),
        include_values=False,
    )

    errors: list[str] = []
    if not checkpoint_local.exists():
        errors.append(f"Checkpoint does not exist: {checkpoint_local}")
    if data_dir_local is not None and not data_dir_local.exists():
        errors.append(f"Local data directory does not exist: {data_dir_local}")
    if symbols_file_local is not None and not symbols_file_local.exists():
        errors.append(f"Symbols file does not exist: {symbols_file_local}")
    for path in extra_checkpoints_local:
        if not path.exists():
            errors.append(f"Extra checkpoint does not exist: {path}")

    (
        symbol_source,
        resolved_symbols,
        removed_duplicate_symbols,
        ignored_symbol_inputs,
    ) = _resolve_symbols_config(
        raw_symbols=tuple(args.symbols),
        symbols_file_local=symbols_file_local,
        errors=errors,
    )
    preview_env_assignments, _, _ = _forward_env_assignments(
        tuple(args.forward_env),
        include_values=False,
    )
    planner_command_preview = _build_remote_planner_command(
        remote_workspace=args.remote_workspace,
        remote_venv=args.remote_venv,
        data_source=args.data_source,
        data_dir_remote=data_dir_remote,
        symbols_file_remote=symbols_file_remote,
        checkpoint_remote=checkpoint_remote,
        extra_checkpoints_remote=extra_checkpoints_remote,
        symbols=resolved_symbols,
        min_open_confidence=args.min_open_confidence,
        min_open_value_estimate=args.min_open_value_estimate,
        env_assignments=preview_env_assignments,
    )

    return PlannerExecutionPlan(
        gpu_preferences=gpu_preferences,
        checkpoint_local=checkpoint_local,
        checkpoint_remote=checkpoint_remote,
        data_source=args.data_source,
        data_dir_local=data_dir_local,
        data_dir_remote=data_dir_remote,
        symbols_file_local=symbols_file_local,
        symbols_file_remote=symbols_file_remote,
        extra_checkpoints_local=extra_checkpoints_local,
        extra_checkpoints_remote=extra_checkpoints_remote,
        symbol_source=symbol_source,
        symbols=resolved_symbols,
        removed_duplicate_symbols=removed_duplicate_symbols,
        ignored_symbol_inputs=ignored_symbol_inputs,
        min_open_confidence=args.min_open_confidence,
        min_open_value_estimate=args.min_open_value_estimate,
        forward_env_names=tuple(args.forward_env),
        present_forward_env=present_forward_env,
        missing_forward_env=missing_forward_env,
        planner_command_preview=planner_command_preview,
        bootstrap_enabled=not args.skip_bootstrap,
        keep_pod=bool(args.keep_pod),
        errors=tuple(errors),
    )


def _print_dry_run_plan(plan: PlannerExecutionPlan) -> None:
    warnings: list[str] = []
    if plan.missing_forward_env:
        warnings.append(
            f"Forwarded environment variables not set: {', '.join(plan.missing_forward_env)}"
        )
    if plan.removed_duplicate_symbols:
        warnings.append(
            f"Removed duplicate symbols: {', '.join(plan.removed_duplicate_symbols)}"
        )
    if plan.ignored_symbol_inputs:
        warnings.append(
            f"Ignored symbol inputs: {', '.join(plan.ignored_symbol_inputs)}"
        )

    payload: dict[str, object] = {
        "ready": not plan.errors,
        "errors": list(plan.errors),
        "warnings": warnings,
        "gpu_preferences": list(plan.gpu_preferences),
        "checkpoint": {
            "local": str(plan.checkpoint_local),
            "remote": plan.checkpoint_remote,
        },
        "data_source": plan.data_source,
        "data_dir": {
            "local": None if plan.data_dir_local is None else str(plan.data_dir_local),
            "remote": plan.data_dir_remote,
        },
        "symbols_file": {
            "local": None if plan.symbols_file_local is None else str(plan.symbols_file_local),
            "remote": plan.symbols_file_remote,
        },
        "extra_checkpoints": [
            {"local": str(local_path), "remote": remote_path}
            for local_path, remote_path in zip(plan.extra_checkpoints_local, plan.extra_checkpoints_remote)
        ],
        "symbol_source": plan.symbol_source,
        "symbols": list(plan.symbols),
        "symbol_count": len(plan.symbols),
        "removed_duplicate_symbols": list(plan.removed_duplicate_symbols),
        "ignored_symbol_inputs": list(plan.ignored_symbol_inputs),
        "forward_env_present": list(plan.present_forward_env),
        "forward_env_missing": list(plan.missing_forward_env),
        "bootstrap_enabled": plan.bootstrap_enabled,
        "keep_pod": plan.keep_pod,
        "planner_command_preview": plan.planner_command_preview,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the daily stock planner remotely on RunPod")
    parser.add_argument("--gpu-type", default="4090")
    parser.add_argument("--gpu-fallbacks", default=None,
                        help="Comma-separated fallback GPU aliases/names (default: use client defaults)")
    parser.add_argument("--remote-workspace", default=DEFAULT_REMOTE_WORKSPACE)
    parser.add_argument("--remote-venv", default=DEFAULT_REMOTE_VENV)
    parser.add_argument("--checkpoint", default=daily_stock.DEFAULT_CHECKPOINT)
    parser.add_argument("--extra-checkpoints", nargs="*", default=None)
    parser.add_argument("--no-ensemble", action="store_true")
    parser.add_argument("--data-source", choices=["local", "alpaca"], default="local")
    parser.add_argument("--data-dir", default=daily_stock.DEFAULT_DATA_DIR)
    parser.add_argument("--symbols", nargs="+", default=list(daily_stock.DEFAULT_SYMBOLS))
    parser.add_argument("--symbols-file", default=None)
    parser.add_argument("--min-open-confidence", type=float, default=daily_stock.DEFAULT_MIN_OPEN_CONFIDENCE)
    parser.add_argument("--min-open-value-estimate", type=float, default=daily_stock.DEFAULT_MIN_OPEN_VALUE_ESTIMATE)
    parser.add_argument("--forward-env", nargs="*", default=[],
                        help="Environment variable names to forward to the remote planner command")
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--keep-pod", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the resolved remote execution plan and exit without creating a pod")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    plan = _build_execution_plan(args)
    if args.dry_run:
        _print_dry_run_plan(plan)
        if plan.errors:
            raise SystemExit(1)
        return
    if plan.errors:
        rendered = "\n".join(f"- {error}" for error in plan.errors)
        raise RuntimeError(f"Planner preflight failed:\n{rendered}")

    client = RunPodClient()

    pod = None
    last_error: Exception | None = None
    for gpu_name in plan.gpu_preferences:
        try:
            pod = client.create_pod(
                PodConfig(
                    name=f"daily-stock-plan-{int(time.time())}",
                    gpu_type=gpu_name,
                )
            )
            break
        except Exception as exc:
            last_error = exc
            if not is_capacity_error(exc):
                raise
    if pod is None:
        raise RuntimeError(f"Could not provision any requested RunPod GPU: {last_error}")

    terminate_pod = not args.keep_pod
    try:
        ready = client.wait_for_pod(pod.id, timeout=900, poll_interval=10)
        _rsync_repo(
            ssh_host=ready.ssh_host,
            ssh_port=ready.ssh_port,
            remote_workspace=args.remote_workspace,
        )

        _rsync_path(
            ssh_host=ready.ssh_host,
            ssh_port=ready.ssh_port,
            local_path=plan.checkpoint_local,
            remote_path=plan.checkpoint_remote,
        )
        if plan.data_dir_local is not None:
            _rsync_path(
                ssh_host=ready.ssh_host,
                ssh_port=ready.ssh_port,
                local_path=plan.data_dir_local,
                remote_path=plan.data_dir_remote,
            )
        if plan.symbols_file_local is not None and plan.symbols_file_remote is not None:
            _rsync_path(
                ssh_host=ready.ssh_host,
                ssh_port=ready.ssh_port,
                local_path=plan.symbols_file_local,
                remote_path=plan.symbols_file_remote,
            )
        for local_extra, remote_extra in zip(plan.extra_checkpoints_local, plan.extra_checkpoints_remote):
            _rsync_path(
                ssh_host=ready.ssh_host,
                ssh_port=ready.ssh_port,
                local_path=local_extra,
                remote_path=remote_extra,
            )

        if not args.skip_bootstrap:
            bootstrap = _ssh_run(
                ssh_host=ready.ssh_host,
                ssh_port=ready.ssh_port,
                remote_cmd=_remote_bootstrap_cmd(
                    remote_workspace=args.remote_workspace,
                    remote_venv=args.remote_venv,
                ),
                capture_output=True,
            )
            if bootstrap.returncode != 0:
                raise RuntimeError(
                    "Remote bootstrap failed:\n"
                    f"stdout:\n{bootstrap.stdout}\n\nstderr:\n{bootstrap.stderr}"
                )

        env_assignments, _, _ = _forward_env_assignments(
            plan.forward_env_names,
            include_values=True,
        )
        result = _ssh_run(
            ssh_host=ready.ssh_host,
            ssh_port=ready.ssh_port,
            remote_cmd=_build_remote_planner_command(
                remote_workspace=args.remote_workspace,
                remote_venv=args.remote_venv,
                data_source=plan.data_source,
                data_dir_remote=plan.data_dir_remote,
                symbols_file_remote=plan.symbols_file_remote,
                checkpoint_remote=plan.checkpoint_remote,
                extra_checkpoints_remote=plan.extra_checkpoints_remote,
                symbols=plan.symbols,
                min_open_confidence=plan.min_open_confidence,
                min_open_value_estimate=plan.min_open_value_estimate,
                env_assignments=env_assignments,
            ),
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Remote planner failed:\n"
                f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
            )

        payload = _extract_json(result.stdout)
        payload["runpod"] = {
            "pod_id": ready.id,
            "gpu_type": ready.gpu_type,
            "ssh_host": ready.ssh_host,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    finally:
        if terminate_pod and pod is not None:
            try:
                client.terminate_pod(pod.id)
            except Exception as exc:
                print(f"warning: failed to terminate pod {pod.id}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
