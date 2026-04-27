"""Tests for monitoring/hourly_prod_check.sh."""
from __future__ import annotations

import hashlib
import os
import subprocess
import fcntl
from pathlib import Path
from textwrap import dedent


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "monitoring" / "hourly_prod_check.sh"
PROMPT = REPO / "monitoring" / "hourly_prod_check_prompt.md"


def _make_fake_repo(
    tmp_path: Path,
    claude_exit: int = 0,
    *,
    emit_output: bool = True,
) -> tuple[Path, Path]:
    fake_repo = tmp_path / "repo"
    monitoring = fake_repo / "monitoring"
    monitoring.mkdir(parents=True)
    (monitoring / "logs").mkdir()
    (monitoring / "hourly_prod_check_prompt.md").write_text("check hourly prod\n", encoding="utf-8")
    (monitoring / "filter_stream.py").write_text(
        dedent(
            """\
            import sys

            for line in sys.stdin:
                print("FILTERED:", line.strip(), flush=True)
            """
        ),
        encoding="utf-8",
    )
    fake_claude = tmp_path / "claude"
    fake_claude.write_text(
        dedent(
            f"""\
            #!/usr/bin/env bash
            if [ "{int(emit_output)}" -eq 1 ]; then
              echo '{{"type":"result","result":"fake hourly result"}}'
            fi
            exit {claude_exit}
            """
        ),
        encoding="utf-8",
    )
    fake_claude.chmod(0o755)
    return fake_repo, fake_claude


def _run_script(fake_repo: Path, fake_claude: Path) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "REPO": str(fake_repo),
        "CLAUDE_BIN": str(fake_claude),
    }
    return subprocess.run(
        ["bash", str(SCRIPT)],
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )


def _parse_current_status(line: str) -> dict[str, str]:
    fields = {}
    for token in line.strip().split()[1:]:
        key, _, value = token.partition("=")
        fields[key] = value
    return fields


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_hourly_prod_check_shell_syntax_is_valid() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_hourly_prod_check_writes_current_log_atomically() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    tmp_idx = text.index('tmp=$(mktemp "${CURRENT_LOG}.tmp.XXXXXX")')
    write_idx = text.index('> "$tmp"', tmp_idx)
    move_idx = text.index('mv "$tmp" "$CURRENT_LOG"', write_idx)

    assert tmp_idx < write_idx < move_idx


def test_hourly_prompt_keeps_current_status_wrapper_owned() -> None:
    text = PROMPT.read_text(encoding="utf-8")

    assert "Do not write `monitoring/logs/hourly_current.log`" in text
    assert 'BC="monitoring/logs/hourly_progress.log"' in text


def test_hourly_prod_check_updates_current_log_on_success(tmp_path: Path) -> None:
    fake_repo, fake_claude = _make_fake_repo(tmp_path, claude_exit=0)

    proc = _run_script(fake_repo, fake_claude)

    assert proc.returncode == 0
    current = (fake_repo / "monitoring" / "logs" / "hourly_current.log").read_text(
        encoding="utf-8",
    )
    assert "status=OK" in current
    assert "rc=0" in current
    assert "log=" in current
    assert "raw_log=" in current
    assert "log_sha256=" in current
    assert "raw_log_sha256=" in current
    assert "NA" not in current
    fields = _parse_current_status(current)
    assert fields["log_sha256"] == _sha256(Path(fields["log"]))
    assert fields["raw_log_sha256"] == _sha256(Path(fields["raw_log"]))


def test_hourly_prod_check_fails_when_success_lacks_raw_stream(tmp_path: Path) -> None:
    fake_repo, fake_claude = _make_fake_repo(tmp_path, claude_exit=0, emit_output=False)

    proc = _run_script(fake_repo, fake_claude)

    assert proc.returncode == 70
    current = (fake_repo / "monitoring" / "logs" / "hourly_current.log").read_text(
        encoding="utf-8",
    )
    assert "status=FAILED" in current
    assert "rc=70" in current
    assert "missing non-empty raw stream artifact" in proc.stdout


def test_hourly_prod_check_returns_failure_and_updates_current_log(tmp_path: Path) -> None:
    fake_repo, fake_claude = _make_fake_repo(tmp_path, claude_exit=9)

    proc = _run_script(fake_repo, fake_claude)

    assert proc.returncode == 9
    current = (fake_repo / "monitoring" / "logs" / "hourly_current.log").read_text(
        encoding="utf-8",
    )
    assert "status=FAILED" in current
    assert "rc=9" in current


def test_hourly_prod_check_updates_current_log_on_setup_failure(tmp_path: Path) -> None:
    fake_repo, _fake_claude = _make_fake_repo(tmp_path)

    proc = _run_script(fake_repo, tmp_path / "missing-claude")

    assert proc.returncode == 2
    current = (fake_repo / "monitoring" / "logs" / "hourly_current.log").read_text(
        encoding="utf-8",
    )
    assert "status=SETUP_FAILED" in current
    assert "rc=2" in current
    fields = _parse_current_status(current)
    assert fields["log_sha256"] == _sha256(Path(fields["log"]))
    assert "missing claude binary" in Path(fields["log"]).read_text(encoding="utf-8")


def test_hourly_prod_check_updates_current_log_when_lock_is_held(tmp_path: Path) -> None:
    fake_repo, fake_claude = _make_fake_repo(tmp_path)
    lock_path = fake_repo / "monitoring" / "logs" / ".hourly_prod_check.lock"

    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        proc = _run_script(fake_repo, fake_claude)
        fcntl.flock(lock_file, fcntl.LOCK_UN)

    assert proc.returncode == 0
    current = (fake_repo / "monitoring" / "logs" / "hourly_current.log").read_text(
        encoding="utf-8",
    )
    assert "status=SKIPPED_LOCK" in current
    assert "rc=0" in current
    fields = _parse_current_status(current)
    assert fields["log_sha256"] == _sha256(Path(fields["log"]))
    assert "previous run still holding lock" in Path(fields["log"]).read_text(
        encoding="utf-8",
    )
