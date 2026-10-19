"""Tests for monitoring/hourly_prod_check.sh compatibility wrapper."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from textwrap import dedent


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "monitoring" / "hourly_prod_check.sh"


def _make_fake_repo(tmp_path: Path, delegate_exit: int = 0) -> Path:
    fake_repo = tmp_path / "repo"
    monitoring = fake_repo / "monitoring"
    monitoring.mkdir(parents=True)
    (monitoring / "codex_prod_check.sh").write_text(
        dedent(
            f"""\
            #!/usr/bin/env bash
            printf 'delegate_pwd=%s\\n' "$PWD"
            printf 'delegate_repo=%s\\n' "$REPO"
            exit {delegate_exit}
            """
        ),
        encoding="utf-8",
    )
    (monitoring / "codex_prod_check.sh").chmod(0o755)
    return fake_repo


def _run_script(fake_repo: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT)],
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "REPO": str(fake_repo)},
    )


def test_hourly_prod_check_shell_syntax_is_valid() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_hourly_prod_check_delegates_to_codex_prod_check() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "exec \"$REPO/monitoring/codex_prod_check.sh\"" in text
    assert "hourly_current.log" not in text
    assert "CLAUDE_BIN" not in text


def test_hourly_prod_check_runs_delegate_from_repo(tmp_path: Path) -> None:
    fake_repo = _make_fake_repo(tmp_path)

    proc = _run_script(fake_repo)

    assert proc.returncode == 0
    assert f"delegate_pwd={fake_repo}" in proc.stdout
    assert f"delegate_repo={fake_repo}" in proc.stdout


def test_hourly_prod_check_propagates_delegate_exit_code(tmp_path: Path) -> None:
    fake_repo = _make_fake_repo(tmp_path, delegate_exit=7)

    proc = _run_script(fake_repo)

    assert proc.returncode == 7
