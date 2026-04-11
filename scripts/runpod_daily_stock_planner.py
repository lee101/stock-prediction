#!/usr/bin/env python3
# ruff: noqa: I001
"""Run the daily stock planner remotely on RunPod and return the JSON plan.

This wrapper is intentionally planning-only. It runs the existing daily stock
trader in `--dry-run --print-payload` mode on a remote pod, then prints the
resulting JSON locally for a separate local execution layer to consume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import sys
import time
from dataclasses import dataclass
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import src.runpod_client as _runpod_client  # noqa: E402
from src.daily_stock_defaults import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_DATA_DIR,
    DEFAULT_EXTRA_CHECKPOINTS,
    DEFAULT_MIN_OPEN_CONFIDENCE,
    DEFAULT_MIN_OPEN_VALUE_ESTIMATE,
    DEFAULT_SYMBOLS,
)
from src.runpod_remote_utils import (  # noqa: E402
    SSH_OPTIONS as _SSH_OPTS,
    run_checked_subprocess as _run_checked_subprocess,
    ssh_run as _ssh_run,
    tail_excerpt as _tail_excerpt,
)
from src.stock_symbol_inputs import load_symbols_file, normalize_symbols  # noqa: E402


DEFAULT_POD_READY_POLL_INTERVAL_SECONDS = _runpod_client.DEFAULT_POD_READY_POLL_INTERVAL_SECONDS
DEFAULT_POD_READY_TIMEOUT_SECONDS = _runpod_client.DEFAULT_POD_READY_TIMEOUT_SECONDS
Pod = _runpod_client.Pod
PodConfig = _runpod_client.PodConfig
RunPodClient = _runpod_client.RunPodClient

if hasattr(_runpod_client, "resolve_gpu_preferences"):
    resolve_gpu_preferences = _runpod_client.resolve_gpu_preferences
else:
    def resolve_gpu_preferences(
        primary: str,
        fallbacks: str | tuple[str, ...] | list[str] | None = None,
    ) -> tuple[str, ...]:
        parsed_fallbacks: tuple[str, ...] | list[str] | None
        if isinstance(fallbacks, str) or fallbacks is None:
            parsed_fallbacks = _runpod_client.parse_gpu_fallback_types(fallbacks)
        else:
            parsed_fallbacks = fallbacks
        return tuple(_runpod_client.build_gpu_fallback_types(primary, parsed_fallbacks))

DEFAULT_REMOTE_WORKSPACE = "/workspace/stock-prediction"
DEFAULT_REMOTE_VENV = ".venv"
_REPO_EXTERNAL_ROOTS = (REPO / ".pytest_tmp",)
_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ALPACA_PAPER_KEY_ENV_CANDIDATES = ("ALP_KEY_ID_PAPER",)
_ALPACA_PAPER_SECRET_ENV_CANDIDATES = ("ALP_SECRET_KEY_PAPER",)
_ALPACA_ANY_KEY_ENV_CANDIDATES = ("ALP_KEY_ID_PAPER", "ALP_KEY_ID")
_ALPACA_ANY_SECRET_ENV_CANDIDATES = ("ALP_SECRET_KEY_PAPER", "ALP_SECRET_KEY")


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
    auto_forward_env_names: tuple[str, ...]
    present_forward_env: tuple[str, ...]
    missing_forward_env: tuple[str, ...]
    setup_warnings: tuple[str, ...]
    planner_command_preview: str
    bootstrap_enabled: bool
    keep_pod: bool
    errors: tuple[str, ...]


def _collect_dry_run_warnings(plan: PlannerExecutionPlan) -> list[str]:
    warnings: list[str] = list(plan.setup_warnings)
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
    return warnings


def _resolve_local_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else (REPO / path).resolve()


def _external_remote_name(local_path: Path) -> str:
    resolved = local_path.resolve()
    digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:12]
    if resolved.suffix:
        return f"{resolved.stem}-{digest}{resolved.suffix}"
    return f"{resolved.name}-{digest}"


def _relative_remote_path(local_path: Path, remote_workspace: str) -> str:
    resolved = local_path.resolve()
    for external_root in _REPO_EXTERNAL_ROOTS:
        try:
            resolved.relative_to(external_root)
        except ValueError:
            continue
        return str(Path(remote_workspace) / ".external" / _external_remote_name(resolved))
    try:
        rel = resolved.relative_to(REPO)
        return str(Path(remote_workspace) / rel)
    except ValueError:
        return str(Path(remote_workspace) / ".external" / _external_remote_name(resolved))


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

def _build_stage_error(
    *,
    stage: str,
    ready,
    detail: str,
    plan: PlannerExecutionPlan | None = None,
    stdout: str = "",
    stderr: str = "",
) -> RuntimeError:
    message = [
        f"RunPod planner {stage} failed",
        f"pod_id: {ready.id}",
        f"gpu_type: {ready.gpu_type}",
        f"ssh_host: {ready.ssh_host}",
        f"ssh_port: {ready.ssh_port}",
        detail,
    ]
    if plan is not None:
        message.append(f"planner_command_preview: {plan.planner_command_preview}")
    stdout_excerpt = _tail_excerpt(stdout)
    stderr_excerpt = _tail_excerpt(stderr)
    if stdout_excerpt:
        message.append(f"stdout excerpt:\n{stdout_excerpt}")
    if stderr_excerpt:
        message.append(f"stderr excerpt:\n{stderr_excerpt}")
    return RuntimeError("\n".join(message))


def _resolve_extra_checkpoint_inputs(args: argparse.Namespace) -> list[str] | None:
    if args.no_ensemble:
        return None
    if args.extra_checkpoints is None:
        return list(DEFAULT_EXTRA_CHECKPOINTS)
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


def _normalize_forward_env_names(
    names: list[str] | tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    normalized: list[str] = []
    invalid: list[str] = []
    seen: set[str] = set()
    for name in names:
        env_name = str(name).strip()
        if not env_name:
            continue
        if not _ENV_VAR_NAME_RE.fullmatch(env_name):
            invalid.append(env_name)
            continue
        if env_name in seen:
            continue
        normalized.append(env_name)
        seen.add(env_name)
    return tuple(normalized), tuple(invalid)


def _auto_forward_env_names(
    *,
    data_source: str,
    explicit_names: tuple[str, ...],
) -> tuple[str, ...]:
    if data_source != "alpaca":
        return ()

    explicit = set(explicit_names)
    auto_forward: list[str] = []
    for candidates in (_ALPACA_PAPER_KEY_ENV_CANDIDATES, _ALPACA_PAPER_SECRET_ENV_CANDIDATES):
        if any(name in explicit for name in candidates):
            continue
        for candidate in candidates:
            value = os.getenv(candidate)
            if value is not None and value != "":
                auto_forward.append(candidate)
                break
    return tuple(auto_forward)


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
            symbol_inputs = load_symbols_file(symbols_file_local)
        except (OSError, ValueError) as exc:
            errors.append(f"Invalid symbols file {symbols_file_local}: {exc}")
            return symbol_source, (), (), ()
    else:
        symbol_inputs = list(raw_symbols)

    try:
        normalized, removed_duplicates, ignored_inputs = normalize_symbols(symbol_inputs)
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
    gpu_preferences = resolve_gpu_preferences(args.gpu_type, args.gpu_fallbacks)

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

    forward_env_names, invalid_forward_env_names = _normalize_forward_env_names(
        tuple(args.forward_env)
    )
    auto_forward_env_names = _auto_forward_env_names(
        data_source=args.data_source,
        explicit_names=forward_env_names,
    )
    effective_forward_env_names = tuple(dict.fromkeys([*forward_env_names, *auto_forward_env_names]))
    if invalid_forward_env_names:
        errors = [
            f"Invalid --forward-env names: {', '.join(invalid_forward_env_names)}"
        ]
    else:
        errors = []
    setup_warnings: list[str] = []
    if auto_forward_env_names:
        setup_warnings.append(
            "Automatically forwarding Alpaca paper credential env vars: "
            + ", ".join(auto_forward_env_names)
        )
    if args.data_source == "alpaca":
        effective_names = set(effective_forward_env_names)
        if not any(name in effective_names for name in _ALPACA_ANY_KEY_ENV_CANDIDATES) or not any(
            name in effective_names for name in _ALPACA_ANY_SECRET_ENV_CANDIDATES
        ):
            setup_warnings.append(
                "Alpaca paper credentials are not being forwarded. "
                "Remote planner may need ALP_KEY_ID_PAPER/ALP_SECRET_KEY_PAPER "
                "or ALP_KEY_ID/ALP_SECRET_KEY."
            )
        live_creds_present = any(os.getenv(name) for name in ("ALP_KEY_ID", "ALP_SECRET_KEY"))
        paper_creds_present = any(os.getenv(name) for name in ("ALP_KEY_ID_PAPER", "ALP_SECRET_KEY_PAPER"))
        if live_creds_present and not paper_creds_present and not any(
            name in effective_names for name in ("ALP_KEY_ID", "ALP_SECRET_KEY")
        ):
            setup_warnings.append(
                "Live Alpaca credentials detected but not auto-forwarded. "
                "Pass --forward-env ALP_KEY_ID ALP_SECRET_KEY explicitly if you intend "
                "to use live credentials on the remote pod."
            )

    _, present_forward_env, missing_forward_env = _forward_env_assignments(
        effective_forward_env_names,
        include_values=False,
    )
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
        effective_forward_env_names,
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
        forward_env_names=effective_forward_env_names,
        auto_forward_env_names=auto_forward_env_names,
        present_forward_env=present_forward_env,
        missing_forward_env=missing_forward_env,
        setup_warnings=tuple(setup_warnings),
        planner_command_preview=planner_command_preview,
        bootstrap_enabled=not args.skip_bootstrap,
        keep_pod=bool(args.keep_pod),
        errors=tuple(errors),
    )


def _build_dry_run_payload(plan: PlannerExecutionPlan) -> dict[str, object]:
    return {
        "ready": not plan.errors,
        "errors": list(plan.errors),
        "warnings": _collect_dry_run_warnings(plan),
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
        "auto_forward_env": list(plan.auto_forward_env_names),
        "forward_env_present": list(plan.present_forward_env),
        "forward_env_missing": list(plan.missing_forward_env),
        "bootstrap_enabled": plan.bootstrap_enabled,
        "keep_pod": plan.keep_pod,
        "planner_command_preview": plan.planner_command_preview,
    }


def _format_dry_run_summary(plan: PlannerExecutionPlan) -> str:
    status = "ready" if not plan.errors else "not ready"
    lines = [
        "RunPod Daily Planner",
        f"Status: {status}",
        f"GPU preferences: {', '.join(plan.gpu_preferences)}",
        f"Checkpoint: {plan.checkpoint_local} -> {plan.checkpoint_remote}",
        f"Data source: {plan.data_source}",
        f"Symbols: {len(plan.symbols)} via {plan.symbol_source}",
    ]
    if plan.symbols:
        lines.append(f"Resolved symbols: {', '.join(plan.symbols)}")
    if plan.present_forward_env or plan.missing_forward_env:
        lines.append(
            "Forward env: "
            f"{len(plan.present_forward_env)} present, {len(plan.missing_forward_env)} missing"
        )
    if plan.auto_forward_env_names:
        lines.append(f"Auto-forward env: {', '.join(plan.auto_forward_env_names)}")
    lines.append(f"Bootstrap: {'enabled' if plan.bootstrap_enabled else 'skipped'}")
    lines.append(f"Keep pod: {'yes' if plan.keep_pod else 'no'}")
    lines.append(f"Remote planner command: {plan.planner_command_preview}")
    if plan.errors:
        lines.append("Errors:")
        lines.extend(f"- {error}" for error in plan.errors)
    warnings = _collect_dry_run_warnings(plan)
    if warnings:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in warnings)
    if not plan.errors:
        lines.append("Next step: rerun without --dry-run to provision a pod and execute the planner.")
    return "\n".join(lines)


def _print_dry_run_plan(plan: PlannerExecutionPlan) -> None:
    print(json.dumps(_build_dry_run_payload(plan), indent=2, sort_keys=True))


def _iter_plan_sync_paths(plan: PlannerExecutionPlan) -> tuple[tuple[Path, str], ...]:
    sync_paths: list[tuple[Path, str]] = [
        (plan.checkpoint_local, plan.checkpoint_remote),
    ]
    if plan.data_dir_local is not None:
        sync_paths.append((plan.data_dir_local, plan.data_dir_remote))
    if plan.symbols_file_local is not None and plan.symbols_file_remote is not None:
        sync_paths.append((plan.symbols_file_local, plan.symbols_file_remote))
    sync_paths.extend(zip(plan.extra_checkpoints_local, plan.extra_checkpoints_remote))
    return tuple(sync_paths)


def _sync_execution_plan_inputs(
    *,
    ready: Pod,
    plan: PlannerExecutionPlan,
    remote_workspace: str,
) -> None:
    _rsync_repo(
        ssh_host=ready.ssh_host,
        ssh_port=ready.ssh_port,
        remote_workspace=remote_workspace,
    )
    for local_path, remote_path in _iter_plan_sync_paths(plan):
        _rsync_path(
            ssh_host=ready.ssh_host,
            ssh_port=ready.ssh_port,
            local_path=local_path,
            remote_path=remote_path,
        )


def _run_remote_bootstrap(
    *,
    ready: Pod,
    remote_workspace: str,
    remote_venv: str,
) -> None:
    bootstrap = _ssh_run(
        ssh_host=ready.ssh_host,
        ssh_port=ready.ssh_port,
        remote_cmd=_remote_bootstrap_cmd(
            remote_workspace=remote_workspace,
            remote_venv=remote_venv,
        ),
        capture_output=True,
    )
    if bootstrap.returncode != 0:
        raise _build_stage_error(
            stage="bootstrap",
            ready=ready,
            detail=(
                f"remote bootstrap command returned exit {bootstrap.returncode}. "
                "See /tmp/runpod_daily_pip.log on the pod for the full installer log."
            ),
            stdout=bootstrap.stdout,
            stderr=bootstrap.stderr,
        )


def _run_remote_planner(
    *,
    ready: Pod,
    plan: PlannerExecutionPlan,
    remote_workspace: str,
    remote_venv: str,
) -> dict[str, object]:
    env_assignments, _, _ = _forward_env_assignments(
        plan.forward_env_names,
        include_values=True,
    )
    result = _ssh_run(
        ssh_host=ready.ssh_host,
        ssh_port=ready.ssh_port,
        remote_cmd=_build_remote_planner_command(
            remote_workspace=remote_workspace,
            remote_venv=remote_venv,
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
        raise _build_stage_error(
            stage="planner command",
            ready=ready,
            plan=plan,
            detail=f"remote planner command returned exit {result.returncode}.",
            stdout=result.stdout,
            stderr=result.stderr,
        )

    try:
        return _extract_json(result.stdout)
    except RuntimeError as exc:
        raise _build_stage_error(
            stage="planner output parsing",
            ready=ready,
            plan=plan,
            detail=str(exc),
            stdout=result.stdout,
            stderr=result.stderr,
        ) from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the daily stock planner remotely on RunPod")
    parser.add_argument("--gpu-type", default="4090")
    parser.add_argument("--gpu-fallbacks", default=None,
                        help="Comma-separated fallback GPU aliases/names (default: use client defaults)")
    parser.add_argument("--remote-workspace", default=DEFAULT_REMOTE_WORKSPACE)
    parser.add_argument("--remote-venv", default=DEFAULT_REMOTE_VENV)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--extra-checkpoints", nargs="*", default=None)
    parser.add_argument("--no-ensemble", action="store_true")
    parser.add_argument("--data-source", choices=["local", "alpaca"], default="local")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--symbols-file", default=None)
    parser.add_argument("--min-open-confidence", type=float, default=DEFAULT_MIN_OPEN_CONFIDENCE)
    parser.add_argument("--min-open-value-estimate", type=float, default=DEFAULT_MIN_OPEN_VALUE_ESTIMATE)
    parser.add_argument("--forward-env", nargs="*", default=[],
                        help="Environment variable names to forward to the remote planner command")
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--keep-pod", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the resolved remote execution plan and exit without creating a pod")
    parser.add_argument(
        "--dry-run-text",
        action="store_true",
        help="When used with --dry-run, also print a human-readable summary to stderr.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    plan = _build_execution_plan(args)
    if args.dry_run:
        _print_dry_run_plan(plan)
        if args.dry_run_text:
            print(_format_dry_run_summary(plan), file=sys.stderr)
        if plan.errors:
            raise SystemExit(1)
        return
    if plan.errors:
        rendered = "\n".join(f"- {error}" for error in plan.errors)
        raise RuntimeError(f"Planner preflight failed:\n{rendered}")

    client = RunPodClient()

    ready = client.create_ready_pod_with_fallback(
        PodConfig(
            name=f"daily-stock-plan-{int(time.time())}",
            gpu_type=plan.gpu_preferences[0],
        ),
        plan.gpu_preferences,
        timeout=DEFAULT_POD_READY_TIMEOUT_SECONDS,
        poll_interval=DEFAULT_POD_READY_POLL_INTERVAL_SECONDS,
    )

    terminate_pod = not args.keep_pod
    try:
        _sync_execution_plan_inputs(
            ready=ready,
            plan=plan,
            remote_workspace=args.remote_workspace,
        )

        if not args.skip_bootstrap:
            _run_remote_bootstrap(
                ready=ready,
                remote_workspace=args.remote_workspace,
                remote_venv=args.remote_venv,
            )
        payload = _run_remote_planner(
            ready=ready,
            plan=plan,
            remote_workspace=args.remote_workspace,
            remote_venv=args.remote_venv,
        )
        payload["runpod"] = {
            "pod_id": ready.id,
            "gpu_type": ready.gpu_type,
            "ssh_host": ready.ssh_host,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    finally:
        if terminate_pod and ready is not None:
            try:
                client.terminate_pod(ready.id)
            except Exception as exc:
                print(f"warning: failed to terminate pod {ready.id}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
