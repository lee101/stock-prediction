from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts import dev_setup_status as status


def _env(
    name: str,
    *,
    exists: bool = True,
    ready: bool = True,
    missing_modules: tuple[str, ...] = (),
) -> status.VirtualEnvStatus:
    return status.VirtualEnvStatus(
        name=name,
        path=f"/repo/{name}",
        exists=exists,
        python_path=f"/repo/{name}/bin/python",
        python_exists=exists,
        ready=ready,
        python_version="3.13.9" if exists else None,
        missing_modules=missing_modules,
        probe_error=None if exists else "missing virtualenv python",
    )


def test_collect_setup_status_prefers_first_ready_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env_map = {
        ".venv313": _env(".venv313", ready=True),
        ".venv": _env(".venv", ready=True),
    }
    monkeypatch.setattr(status, "_probe_virtualenv", lambda repo_root, env_name: env_map[env_name])
    monkeypatch.setattr(status.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    monkeypatch.setattr(status, "DEFAULT_SECRETS_BASHRC", tmp_path / ".secretbashrc")
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)

    setup = status.collect_setup_status(repo_root=tmp_path, env_names=(".venv313", ".venv"))

    assert setup.recommended_env == ".venv313"
    assert setup.recommended_env_ready is True
    assert setup.recommended_activate == "source .venv313/bin/activate"
    assert setup.profile_activate == "source .venv313/bin/activate"


def test_collect_setup_status_falls_back_to_existing_not_ready_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env_map = {
        ".venv313": _env(".venv313", ready=False, missing_modules=("torch",)),
        ".venv": _env(".venv", exists=False, ready=False),
    }
    monkeypatch.setattr(status, "_probe_virtualenv", lambda repo_root, env_name: env_map[env_name])
    monkeypatch.setattr(status.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    monkeypatch.setattr(status, "DEFAULT_SECRETS_BASHRC", tmp_path / ".secretbashrc")

    setup = status.collect_setup_status(repo_root=tmp_path, env_names=(".venv313", ".venv"))

    assert setup.recommended_env == ".venv313"
    assert setup.recommended_env_ready is False
    assert "uv pip install -e '.[dev]'" in setup.bootstrap_command


def test_collect_setup_status_bootstrap_includes_uv_install_when_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env_map = {
        ".venv313": _env(".venv313", exists=False, ready=False),
    }
    monkeypatch.setattr(status, "_probe_virtualenv", lambda repo_root, env_name: env_map[env_name])
    monkeypatch.setattr(status.shutil, "which", lambda name: None)
    monkeypatch.setattr(status, "DEFAULT_SECRETS_BASHRC", tmp_path / ".secretbashrc")

    setup = status.collect_setup_status(repo_root=tmp_path, env_names=(".venv313",))

    assert "python3 -m pip install --user uv && uv venv .venv313 --python 3.13" in setup.bootstrap_command


def test_render_setup_status_includes_activate_and_env_lines() -> None:
    rendered = status.render_setup_status(
        status.SetupStatus(
            profile="dev",
            repo_root="/repo",
            active_python="/repo/.venv313/bin/python",
            active_virtualenv=".venv313",
            uv_available=True,
            recommended_env=".venv313",
            recommended_env_ready=True,
            recommended_activate="source .venv313/bin/activate",
            profile_activate="source .venv313/bin/activate",
            bootstrap_command="source .venv313/bin/activate && uv pip install -e '.[dev]'",
            setup_check_command="bash scripts/run_ci_locally.sh --dry-run --job lint",
            profile_ready=True,
            secrets_file="/home/me/.secretbashrc",
            secrets_file_exists=False,
            current_shell_secret_hints_present=False,
            runpod_api_key_present=False,
            envs=(
                _env(".venv313", ready=True),
                _env(".venv", ready=False, missing_modules=("pytest",)),
            ),
        )
    )

    assert "Profile: dev" in rendered
    assert "Recommended env: .venv313" in rendered
    assert "Activate: source .venv313/bin/activate" in rendered
    assert "- .venv313: ready (Python 3.13.9)" in rendered
    assert "- .venv: missing modules: pytest" in rendered


def test_collect_setup_status_alpaca_profile_requires_secret_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env_map = {".venv313": _env(".venv313", ready=True)}
    secrets_file = tmp_path / ".secretbashrc"
    monkeypatch.setattr(status, "_probe_virtualenv", lambda repo_root, env_name: env_map[env_name])
    monkeypatch.setattr(status.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    monkeypatch.setattr(status, "DEFAULT_SECRETS_BASHRC", secrets_file)
    monkeypatch.setenv("ALP_KEY_ID_PROD", "live-key")

    missing_setup = status.collect_setup_status(repo_root=tmp_path, env_names=(".venv313",), profile="alpaca")
    assert missing_setup.profile_ready is False
    assert missing_setup.setup_check_command == "python trade_daily_stock_prod.py --check-config --check-config-text"
    assert missing_setup.current_shell_secret_hints_present is True

    secrets_file.write_text("export ALP_KEY_ID_PROD=test\n", encoding="utf-8")
    ready_setup = status.collect_setup_status(repo_root=tmp_path, env_names=(".venv313",), profile="alpaca")
    assert ready_setup.profile_ready is True
    assert ready_setup.secrets_file_exists is True
    assert ready_setup.profile_activate == f"source {secrets_file} && source .venv313/bin/activate"


def test_collect_setup_status_runpod_profile_requires_api_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env_map = {".venv313": _env(".venv313", ready=True)}
    monkeypatch.setattr(status, "_probe_virtualenv", lambda repo_root, env_name: env_map[env_name])
    monkeypatch.setattr(status.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    monkeypatch.setattr(status, "DEFAULT_SECRETS_BASHRC", tmp_path / ".secretbashrc")
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)

    missing_setup = status.collect_setup_status(repo_root=tmp_path, env_names=(".venv313",), profile="runpod")
    assert missing_setup.profile_ready is False
    assert missing_setup.runpod_api_key_present is False
    assert missing_setup.setup_check_command == 'python -c "from src.runpod_client import RunPodClient; RunPodClient()"'

    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    (tmp_path / ".secretbashrc").write_text("export RUNPOD_API_KEY=test\n", encoding="utf-8")
    ready_setup = status.collect_setup_status(repo_root=tmp_path, env_names=(".venv313",), profile="runpod")
    assert ready_setup.profile_ready is True
    assert ready_setup.runpod_api_key_present is True
    assert ready_setup.profile_activate == f"source {tmp_path / '.secretbashrc'} && source .venv313/bin/activate"


def test_main_json_check_exits_nonzero_when_setup_not_ready(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(
        status,
        "collect_setup_status",
        lambda profile="dev": status.SetupStatus(
            profile=profile,
            repo_root="/repo",
            active_python="/usr/bin/python3",
            active_virtualenv=None,
            uv_available=False,
            recommended_env=".venv313",
            recommended_env_ready=False,
            recommended_activate="source .venv313/bin/activate",
            profile_activate="source .venv313/bin/activate",
            bootstrap_command="source .venv313/bin/activate && uv pip install -e '.[dev]'",
            setup_check_command="bash scripts/run_ci_locally.sh --dry-run --job lint",
            profile_ready=False,
            secrets_file="/home/me/.secretbashrc",
            secrets_file_exists=False,
            current_shell_secret_hints_present=False,
            runpod_api_key_present=False,
            envs=(_env(".venv313", ready=False, missing_modules=("torch",)),),
        ),
    )

    with pytest.raises(SystemExit, match="1"):
        status.main(["--json", "--check"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["recommended_env"] == ".venv313"
    assert payload["recommended_env_ready"] is False
