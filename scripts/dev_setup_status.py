#!/usr/bin/env python3
"""Report the local development setup status for this repository."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


REPO = Path(__file__).resolve().parents[1]
DEFAULT_SECRETS_BASHRC = Path.home() / ".secretbashrc"
DEFAULT_ENV_NAMES: tuple[str, ...] = (
    ".venv313",
    ".venv",
    ".venv312",
    ".venv311",
    ".venvci",
)
REQUIRED_MODULES: tuple[str, ...] = (
    "numpy",
    "pandas",
    "requests",
    "pytest",
    "torch",
)
SetupProfile = Literal["dev", "alpaca", "runpod"]


@dataclass(frozen=True)
class VirtualEnvStatus:
    name: str
    path: str
    exists: bool
    python_path: str
    python_exists: bool
    ready: bool
    python_version: str | None
    missing_modules: tuple[str, ...]
    probe_error: str | None


@dataclass(frozen=True)
class SetupStatus:
    profile: SetupProfile
    repo_root: str
    active_python: str
    active_virtualenv: str | None
    uv_available: bool
    recommended_env: str | None
    recommended_env_ready: bool
    recommended_activate: str | None
    profile_activate: str | None
    bootstrap_command: str | None
    setup_check_command: str
    profile_ready: bool
    secrets_file: str
    secrets_file_exists: bool
    current_shell_secret_hints_present: bool
    runpod_api_key_present: bool
    envs: tuple[VirtualEnvStatus, ...]


def _venv_python_path(repo_root: Path, env_name: str) -> Path:
    return repo_root / env_name / "bin" / "python"


def _probe_virtualenv(repo_root: Path, env_name: str) -> VirtualEnvStatus:
    env_path = repo_root / env_name
    python_path = _venv_python_path(repo_root, env_name)
    if not env_path.exists():
        return VirtualEnvStatus(
            name=env_name,
            path=str(env_path),
            exists=False,
            python_path=str(python_path),
            python_exists=False,
            ready=False,
            python_version=None,
            missing_modules=(),
            probe_error=None,
        )
    if not python_path.exists():
        return VirtualEnvStatus(
            name=env_name,
            path=str(env_path),
            exists=True,
            python_path=str(python_path),
            python_exists=False,
            ready=False,
            python_version=None,
            missing_modules=(),
            probe_error="missing virtualenv python",
        )

    probe = subprocess.run(
        [
            str(python_path),
            "-c",
            (
                "import importlib.util, json, sys; "
                f"mods={REQUIRED_MODULES!r}; "
                "missing=[m for m in mods if importlib.util.find_spec(m) is None]; "
                'print(json.dumps({"python_version": sys.version.split()[0], "missing_modules": missing}))'
            ),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        return VirtualEnvStatus(
            name=env_name,
            path=str(env_path),
            exists=True,
            python_path=str(python_path),
            python_exists=True,
            ready=False,
            python_version=None,
            missing_modules=(),
            probe_error=(probe.stderr or probe.stdout or "probe failed").strip() or "probe failed",
        )

    payload = json.loads(probe.stdout)
    missing_modules = tuple(str(name) for name in payload.get("missing_modules", []))
    return VirtualEnvStatus(
        name=env_name,
        path=str(env_path),
        exists=True,
        python_path=str(python_path),
        python_exists=True,
        ready=not missing_modules,
        python_version=str(payload.get("python_version") or "").strip() or None,
        missing_modules=missing_modules,
        probe_error=None,
    )


def _active_virtualenv(repo_root: Path) -> str | None:
    virtual_env = os.environ.get("VIRTUAL_ENV")
    if not virtual_env:
        return None
    try:
        path = Path(virtual_env).resolve()
    except OSError:
        return virtual_env
    try:
        relative = path.relative_to(repo_root)
    except ValueError:
        return str(path)
    return str(relative)


def collect_setup_status(
    *,
    repo_root: Path = REPO,
    env_names: Sequence[str] = DEFAULT_ENV_NAMES,
    profile: SetupProfile = "dev",
) -> SetupStatus:
    envs = tuple(_probe_virtualenv(repo_root, env_name) for env_name in env_names)
    recommended = next((env for env in envs if env.ready), None)
    if recommended is None:
        recommended = next((env for env in envs if env.exists), None)

    recommended_name = recommended.name if recommended is not None else None
    recommended_activate = (
        f"source {recommended_name}/bin/activate"
        if recommended_name is not None
        else None
    )
    uv_bootstrap = (
        "uv"
        if shutil.which("uv") is not None
        else "python3 -m pip install --user uv && uv"
    )
    bootstrap_command = (
        f"{recommended_activate} && {uv_bootstrap} pip install -e '.[dev]'"
        if recommended_activate is not None
        else (
            f"{uv_bootstrap} venv .venv313 --python 3.13 && "
            "source .venv313/bin/activate && "
            f"{uv_bootstrap} pip install -e '.[dev]'"
        )
    )
    secrets_file = DEFAULT_SECRETS_BASHRC
    current_shell_secret_hints_present = any(
        str(os.environ.get(name, "")).strip()
        for name in ("ALP_KEY_ID", "ALP_KEY_ID_PROD", "APCA_API_KEY_ID", "APCA_API_SECRET_KEY")
    )
    runpod_api_key_present = bool(str(os.environ.get("RUNPOD_API_KEY", "")).strip())
    setup_check_command = "bash scripts/run_ci_locally.sh --dry-run --job lint"
    profile_ready = bool(recommended and recommended.ready and shutil.which("uv") is not None)
    profile_activate = recommended_activate
    if profile == "alpaca":
        setup_check_command = "python trade_daily_stock_prod.py --check-config --check-config-text"
        profile_ready = bool(profile_ready and secrets_file.exists())
        if recommended_activate is not None and secrets_file.exists():
            profile_activate = f"source {secrets_file} && {recommended_activate}"
    elif profile == "runpod":
        setup_check_command = 'python -c "from src.runpod_client import RunPodClient; RunPodClient()"'
        profile_ready = bool(profile_ready and runpod_api_key_present)
        if recommended_activate is not None and secrets_file.exists():
            profile_activate = f"source {secrets_file} && {recommended_activate}"
    return SetupStatus(
        profile=profile,
        repo_root=str(repo_root),
        active_python=sys.executable,
        active_virtualenv=_active_virtualenv(repo_root),
        uv_available=shutil.which("uv") is not None,
        recommended_env=recommended_name,
        recommended_env_ready=bool(recommended and recommended.ready),
        recommended_activate=recommended_activate,
        profile_activate=profile_activate,
        bootstrap_command=bootstrap_command,
        setup_check_command=setup_check_command,
        profile_ready=profile_ready,
        secrets_file=str(secrets_file),
        secrets_file_exists=secrets_file.exists(),
        current_shell_secret_hints_present=current_shell_secret_hints_present,
        runpod_api_key_present=runpod_api_key_present,
        envs=envs,
    )


def _render_env_line(env: VirtualEnvStatus) -> str:
    if not env.exists:
        return f"- {env.name}: missing"
    if not env.python_exists:
        return f"- {env.name}: broken ({env.probe_error})"
    if env.ready:
        version = f" (Python {env.python_version})" if env.python_version else ""
        return f"- {env.name}: ready{version}"
    if env.missing_modules:
        return f"- {env.name}: missing modules: {', '.join(env.missing_modules)}"
    return f"- {env.name}: not ready ({env.probe_error or 'unknown problem'})"


def render_setup_status(status: SetupStatus) -> str:
    lines = [
        "Development Setup Status",
        f"Profile: {status.profile}",
        f"Repo root: {status.repo_root}",
        f"Active python: {status.active_python}",
        f"Active virtualenv: {status.active_virtualenv or 'none'}",
        f"uv available: {'yes' if status.uv_available else 'no'}",
        f"Recommended env: {status.recommended_env or 'none'}",
        f"Profile ready: {'yes' if status.profile_ready else 'no'}",
    ]
    if status.recommended_activate:
        lines.append(f"Activate: {status.recommended_activate}")
    if status.profile_activate and status.profile_activate != status.recommended_activate:
        lines.append(f"Profile activate: {status.profile_activate}")
    if status.bootstrap_command:
        lines.append(f"Bootstrap: {status.bootstrap_command}")
    if status.profile == "alpaca":
        lines.append(f"Secrets file: {status.secrets_file} ({'present' if status.secrets_file_exists else 'missing'})")
        lines.append(
            "Current shell Alpaca hints: "
            f"{'present' if status.current_shell_secret_hints_present else 'missing'}"
        )
    elif status.profile == "runpod":
        lines.append(f"RUNPOD_API_KEY: {'present' if status.runpod_api_key_present else 'missing'}")
    lines.append(f"Quick check: {status.setup_check_command}")
    lines.append("Discovered environments:")
    lines.extend(_render_env_line(env) for env in status.envs)
    return "\n".join(lines)


def _is_status_ready(status: SetupStatus) -> bool:
    return bool(status.profile_ready)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Report local project setup status")
    parser.add_argument(
        "--profile",
        choices=("dev", "alpaca", "runpod"),
        default="dev",
        help="Tailor setup guidance to a workflow instead of only checking generic dev readiness.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the setup status as JSON instead of human-readable text.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero when no ready recommended environment is available.",
    )
    args = parser.parse_args(argv)

    status = collect_setup_status(profile=args.profile)
    if args.json:
        print(json.dumps(asdict(status), indent=2, sort_keys=True))
    else:
        print(render_setup_status(status))

    if args.check and not _is_status_ready(status):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
