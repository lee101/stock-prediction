from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

import src.runpod_remote_utils as remote_utils


def test_ssh_cmd_builds_expected_invocation() -> None:
    cmd = remote_utils.ssh_cmd(
        ssh_host="1.2.3.4",
        ssh_port=22022,
        remote_cmd="echo hello",
    )

    assert cmd == [
        "ssh",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "BatchMode=yes",
        "-p",
        "22022",
        "root@1.2.3.4",
        "echo hello",
    ]


def test_render_subprocess_error_trims_stdout_and_stderr() -> None:
    result = SimpleNamespace(
        returncode=17,
        stdout="a" * 500,
        stderr="b" * 500,
    )

    error = remote_utils.render_subprocess_error(
        description="Remote command failed",
        cmd=["ssh", "root@1.2.3.4", "echo test"],
        result=result,
        excerpt_limit=40,
    )

    message = str(error)
    assert "Remote command failed (exit 17)" in message
    assert "command: ssh root@1.2.3.4 'echo test'" in message
    assert f"stdout excerpt:\n{'a' * 40}" in message
    assert f"stderr excerpt:\n{'b' * 40}" in message


def test_run_checked_subprocess_raises_rendered_runtime_error(monkeypatch) -> None:
    monkeypatch.setattr(
        remote_utils.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=23, stdout="partial", stderr="broken"),
    )

    with pytest.raises(RuntimeError, match="Failed to sync repository"):
        remote_utils.run_checked_subprocess(
            ["rsync", "-az", ".", "root@1.2.3.4:/workspace/app"],
            description="Failed to sync repository",
        )


def test_run_checked_subprocess_returns_completed_process_on_success(monkeypatch) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    result = SimpleNamespace(returncode=0, stdout="ok", stderr="")

    def _fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return result

    monkeypatch.setattr(remote_utils.subprocess, "run", _fake_run)

    completed = remote_utils.run_checked_subprocess(
        ["rsync", "-az", ".", "root@1.2.3.4:/workspace/app"],
        description="Failed to sync repository",
    )

    assert completed is result
    assert calls == [
        (
            (["rsync", "-az", ".", "root@1.2.3.4:/workspace/app"],),
            {"check": False, "text": True, "capture_output": True},
        )
    ]


def test_ssh_run_forwards_built_command_and_capture_output(monkeypatch) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    result = SimpleNamespace(returncode=0, stdout="remote ok", stderr="")

    def _fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return result

    monkeypatch.setattr(remote_utils.subprocess, "run", _fake_run)

    completed = remote_utils.ssh_run(
        ssh_host="1.2.3.4",
        ssh_port=22022,
        remote_cmd="echo hello",
        capture_output=True,
    )

    assert completed is result
    assert calls == [
        (
            (
                [
                    "ssh",
                    "-o",
                    "StrictHostKeyChecking=accept-new",
                    "-o",
                    "BatchMode=yes",
                    "-p",
                    "22022",
                    "root@1.2.3.4",
                    "echo hello",
                ],
            ),
            {"check": False, "text": True, "capture_output": True},
        )
    ]


def test_module_allows_explicit_ssh_host_key_override(monkeypatch) -> None:
    monkeypatch.setenv("RUNPOD_SSH_STRICT_HOST_KEY_CHECKING", "yes")

    reloaded = importlib.reload(remote_utils)
    try:
        assert reloaded.SSH_OPTIONS == (
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            "BatchMode=yes",
        )
    finally:
        monkeypatch.delenv("RUNPOD_SSH_STRICT_HOST_KEY_CHECKING", raising=False)
        importlib.reload(remote_utils)


def test_module_rejects_invalid_ssh_host_key_override(monkeypatch) -> None:
    monkeypatch.setenv("RUNPOD_SSH_STRICT_HOST_KEY_CHECKING", "definitely-not-valid")

    with pytest.raises(RuntimeError, match="RUNPOD_SSH_STRICT_HOST_KEY_CHECKING"):
        importlib.reload(remote_utils)

    monkeypatch.delenv("RUNPOD_SSH_STRICT_HOST_KEY_CHECKING", raising=False)
    importlib.reload(remote_utils)
