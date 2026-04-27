"""Tests for monitoring/monitor_agent.sh."""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from textwrap import dedent


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "monitoring" / "monitor_agent.sh"


def _parse_current_status(line: str) -> dict[str, str]:
    fields = {}
    for token in line.strip().split()[1:]:
        key, _, value = token.partition("=")
        fields[key] = value
    return fields


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_monitor_agent_shell_syntax_is_valid() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_monitor_agent_supports_repo_and_claude_overrides() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'REPO="${REPO:-/nvme0n1-disk/code/stock-prediction}"' in text
    assert 'CLAUDE_BIN="${CLAUDE_BIN:-claude}"' in text
    assert 'cd "$REPO"' in text
    assert 'timeout 1800 "$CLAUDE_BIN"' in text


def test_monitor_agent_captures_health_exit_under_pipefail() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    health_idx = text.index("python monitoring/health_check.py --json")
    set_plus_idx = text.rindex("set +e", 0, health_idx)
    capture_idx = text.index("local exit_code=${PIPESTATUS[0]}", health_idx)
    set_minus_idx = text.index("set -e", capture_idx)
    return_idx = text.index('return "$exit_code"', set_minus_idx)
    call_idx = text.index("if run_health_check; then", return_idx)
    health_zero_idx = text.index("HEALTH_EXIT=0", call_idx)
    health_capture_idx = text.index("HEALTH_EXIT=$?", health_zero_idx)
    branch_idx = text.index('if [ "$HEALTH_EXIT" -ne 0 ]; then', health_capture_idx)

    assert set_plus_idx < health_idx < capture_idx < set_minus_idx < return_idx
    assert return_idx < call_idx < health_zero_idx < health_capture_idx < branch_idx


def test_monitor_agent_reruns_health_after_agent_and_exits_final_status() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'CURRENT_LOG="$LOG_DIR/monitor_current.log"' in text
    assert 'tmp=$(mktemp "${CURRENT_LOG}.tmp.XXXXXX")' in text
    assert 'mv "$tmp" "$CURRENT_LOG"' in text
    assert "HEALTH_CHECK_SKIP_MONITOR_CURRENT=1 python monitoring/health_check.py --json" in text

    branch_idx = text.index('if [ "$HEALTH_EXIT" -ne 0 ]; then')
    agent_idx = text.index('timeout 1800 "$CLAUDE_BIN"', branch_idx)
    agent_capture_idx = text.index("AGENT_EXIT=${PIPESTATUS[0]}", agent_idx)
    agent_log_idx = text.index("=== Agent exited with code $AGENT_EXIT ===", agent_capture_idx)
    post_check_idx = text.index("=== Post-agent health check", agent_log_idx)
    rerun_idx = text.index("if run_health_check; then", post_check_idx)
    final_zero_idx = text.index("FINAL_EXIT=0", rerun_idx)
    final_capture_idx = text.index("FINAL_EXIT=$?", final_zero_idx)
    still_unhealthy_idx = text.index("=== Still unhealthy after agent ===", final_capture_idx)
    complete_idx = text.index("=== Monitor complete", still_unhealthy_idx)
    exit_idx = text.index('exit "$FINAL_EXIT"', complete_idx)

    assert agent_idx < agent_capture_idx < agent_log_idx < post_check_idx
    assert post_check_idx < rerun_idx < final_zero_idx < final_capture_idx
    assert final_capture_idx < still_unhealthy_idx < complete_idx < exit_idx


def test_monitor_agent_unhealthy_path_returns_post_agent_status(tmp_path) -> None:
    """Exercise real Bash errexit/pipefail behavior with fake dependencies."""
    fake_repo = tmp_path / "repo"
    fake_repo.mkdir()
    (fake_repo / "monitoring").mkdir()
    (fake_repo / ".venv313" / "bin").mkdir(parents=True)
    (fake_repo / ".venv313" / "bin" / "activate").write_text(
        "export PATH=\"$PWD/fakebin:$PATH\"\n",
        encoding="utf-8",
    )
    fakebin = fake_repo / "fakebin"
    fakebin.mkdir()
    call_counter = fake_repo / "health_calls"
    fake_python = fakebin / "python"
    fake_python.write_text(
        dedent(
            f"""\
            #!/usr/bin/env bash
            calls_file={call_counter}
            calls=$(cat "$calls_file" 2>/dev/null || echo 0)
            calls=$((calls + 1))
            echo "$calls" > "$calls_file"
            echo "{{\\"health_call\\":$calls}}"
            if [ "$calls" -eq 1 ]; then
              exit 1
            fi
            exit 0
            """
        ),
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_claude = fakebin / "claude"
    fake_claude.write_text(
        "#!/usr/bin/env bash\necho fake claude remediation\nexit 0\n",
        encoding="utf-8",
    )
    fake_claude.chmod(0o755)

    script = fake_repo / "monitor_agent.sh"
    script.write_text(
        SCRIPT.read_text(encoding="utf-8").replace(
            "cd /nvme0n1-disk/code/stock-prediction",
            f"cd {fake_repo}",
        ),
        encoding="utf-8",
    )
    script.chmod(0o755)

    proc = subprocess.run(
        ["bash", str(script)],
        cwd=fake_repo,
        text=True,
        capture_output=True,
        check=False,
        env={"REPO": str(fake_repo), "CLAUDE_BIN": str(fake_claude)},
    )

    assert proc.returncode == 0
    assert call_counter.read_text(encoding="utf-8").strip() == "2"
    assert "=== Unhealthy" in proc.stdout
    assert "fake claude remediation" in proc.stdout
    assert "=== Agent exited with code 0 ===" in proc.stdout
    assert "=== Recovered after agent ===" in proc.stdout
    current = (fake_repo / "monitoring" / "logs" / "monitor_current.log").read_text(
        encoding="utf-8",
    )
    assert "status=RECOVERED" in current
    assert "rc=0" in current
    assert "initial_rc=1" in current
    assert "final_rc=0" in current
    assert "agent_rc=0" in current
    assert "log_sha256=" in current
    assert "NA" not in current
    assert f"log={fake_repo}/monitoring/logs/monitor_" in current
    fields = _parse_current_status(current)
    assert fields["log_sha256"] == _sha256(Path(fields["log"]))


def test_monitor_agent_logs_failed_agent_exit_but_uses_final_health_status(tmp_path) -> None:
    fake_repo = tmp_path / "repo"
    fake_repo.mkdir()
    (fake_repo / "monitoring").mkdir()
    (fake_repo / ".venv313" / "bin").mkdir(parents=True)
    (fake_repo / ".venv313" / "bin" / "activate").write_text(
        "export PATH=\"$PWD/fakebin:$PATH\"\n",
        encoding="utf-8",
    )
    fakebin = fake_repo / "fakebin"
    fakebin.mkdir()
    call_counter = fake_repo / "health_calls"
    fake_python = fakebin / "python"
    fake_python.write_text(
        dedent(
            f"""\
            #!/usr/bin/env bash
            calls_file={call_counter}
            calls=$(cat "$calls_file" 2>/dev/null || echo 0)
            calls=$((calls + 1))
            echo "$calls" > "$calls_file"
            echo "{{\\"health_call\\":$calls}}"
            if [ "$calls" -eq 1 ]; then
              exit 1
            fi
            exit 0
            """
        ),
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_claude = fakebin / "claude"
    fake_claude.write_text(
        "#!/usr/bin/env bash\necho fake claude failure\nexit 124\n",
        encoding="utf-8",
    )
    fake_claude.chmod(0o755)

    script = fake_repo / "monitor_agent.sh"
    script.write_text(SCRIPT.read_text(encoding="utf-8"), encoding="utf-8")
    script.chmod(0o755)

    proc = subprocess.run(
        ["bash", str(script)],
        cwd=fake_repo,
        text=True,
        capture_output=True,
        check=False,
        env={"REPO": str(fake_repo), "CLAUDE_BIN": str(fake_claude)},
    )

    assert proc.returncode == 0
    assert call_counter.read_text(encoding="utf-8").strip() == "2"
    assert "fake claude failure" in proc.stdout
    assert "=== Agent exited with code 124 ===" in proc.stdout
    assert "=== Recovered after agent ===" in proc.stdout
    current = (fake_repo / "monitoring" / "logs" / "monitor_current.log").read_text(
        encoding="utf-8",
    )
    assert "status=RECOVERED" in current
    assert "agent_rc=124" in current
    assert "log_sha256=" in current
    fields = _parse_current_status(current)
    assert fields["log_sha256"] == _sha256(Path(fields["log"]))


def test_monitor_agent_healthy_path_writes_current_status(tmp_path) -> None:
    fake_repo = tmp_path / "repo"
    fake_repo.mkdir()
    (fake_repo / "monitoring").mkdir()
    (fake_repo / ".venv313" / "bin").mkdir(parents=True)
    (fake_repo / ".venv313" / "bin" / "activate").write_text(
        "export PATH=\"$PWD/fakebin:$PATH\"\n",
        encoding="utf-8",
    )
    fakebin = fake_repo / "fakebin"
    fakebin.mkdir()
    fake_python = fakebin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\necho '{\"healthy\":true}'\nexit 0\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_claude = fakebin / "claude"
    fake_claude.write_text(
        "#!/usr/bin/env bash\necho should not run\nexit 1\n",
        encoding="utf-8",
    )
    fake_claude.chmod(0o755)

    script = fake_repo / "monitor_agent.sh"
    script.write_text(SCRIPT.read_text(encoding="utf-8"), encoding="utf-8")
    script.chmod(0o755)

    proc = subprocess.run(
        ["bash", str(script)],
        cwd=fake_repo,
        text=True,
        capture_output=True,
        check=False,
        env={"REPO": str(fake_repo), "CLAUDE_BIN": str(fake_claude)},
    )

    assert proc.returncode == 0
    assert "should not run" not in proc.stdout
    current = (fake_repo / "monitoring" / "logs" / "monitor_current.log").read_text(
        encoding="utf-8",
    )
    assert "status=OK" in current
    assert "rc=0" in current
    assert "initial_rc=0" in current
    assert "final_rc=0" in current
    assert "agent_rc=NA" in current
    assert "log_sha256=" in current
    fields = _parse_current_status(current)
    assert fields["log_sha256"] == _sha256(Path(fields["log"]))


def test_monitor_agent_setup_failure_writes_current_status(tmp_path) -> None:
    fake_repo = tmp_path / "repo"
    fake_repo.mkdir()
    (fake_repo / "monitoring").mkdir()

    script = fake_repo / "monitor_agent.sh"
    script.write_text(SCRIPT.read_text(encoding="utf-8"), encoding="utf-8")
    script.chmod(0o755)

    proc = subprocess.run(
        ["bash", str(script)],
        cwd=fake_repo,
        text=True,
        capture_output=True,
        check=False,
        env={"REPO": str(fake_repo), "CLAUDE_BIN": str(tmp_path / "missing-claude")},
    )

    assert proc.returncode == 2
    assert "Monitor setup failed" in proc.stdout
    current = (fake_repo / "monitoring" / "logs" / "monitor_current.log").read_text(
        encoding="utf-8",
    )
    assert "status=SETUP_FAILED" in current
    assert "rc=2" in current
    assert "initial_rc=NA" in current
    assert "final_rc=2" in current
    assert "agent_rc=NA" in current
    fields = _parse_current_status(current)
    assert fields["log_sha256"] == _sha256(Path(fields["log"]))
