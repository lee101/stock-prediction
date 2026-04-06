from __future__ import annotations

from pathlib import Path

from src.server_settings_utils import (
    resolve_env_int,
    resolve_explicit_or_env_int,
    resolve_repo_relative_path,
)


def test_resolve_repo_relative_path_reports_explicit_source(tmp_path: Path) -> None:
    resolution = resolve_repo_relative_path(
        "config/accounts.json",
        repo_root=tmp_path,
        env_name="UNUSED_PATH_ENV",
        default_path=tmp_path / "default.json",
        explicit_label="registry_path",
    )

    assert resolution.path == tmp_path / "config" / "accounts.json"
    assert resolution.source == "explicit"
    assert resolution.detail == "explicit registry_path='config/accounts.json'"


def test_resolve_repo_relative_path_uses_env_when_present(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TEST_PATH_ENV", "runtime/accounts.json")

    resolution = resolve_repo_relative_path(
        None,
        repo_root=tmp_path,
        env_name="TEST_PATH_ENV",
        default_path=tmp_path / "default.json",
        explicit_label="registry_path",
    )

    assert resolution.path == tmp_path / "runtime" / "accounts.json"
    assert resolution.source == "env"
    assert resolution.detail == "TEST_PATH_ENV='runtime/accounts.json'"


def test_resolve_explicit_or_env_int_clamps_and_falls_back(monkeypatch) -> None:
    monkeypatch.setenv("TEST_INT_ENV", "not-a-number")

    assert resolve_env_int("TEST_INT_ENV", 7, minimum=1) == 7
    assert resolve_explicit_or_env_int(None, env_name="TEST_INT_ENV", default=9, minimum=1) == 9
    assert resolve_explicit_or_env_int(0, env_name="TEST_INT_ENV", default=9, minimum=3) == 3
