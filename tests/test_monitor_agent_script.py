"""Tests for monitoring/monitor_agent.sh."""
from __future__ import annotations

import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "monitoring" / "monitor_agent.sh"


def test_monitor_agent_shell_syntax_is_valid() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_monitor_agent_captures_health_exit_under_pipefail() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    health_idx = text.index("python monitoring/health_check.py --json")
    set_plus_idx = text.rindex("set +e", 0, health_idx)
    capture_idx = text.index("HEALTH_EXIT=${PIPESTATUS[0]}", health_idx)
    set_minus_idx = text.index("set -e", capture_idx)
    branch_idx = text.index('if [ "$HEALTH_EXIT" -ne 0 ]; then', set_minus_idx)

    assert set_plus_idx < health_idx < capture_idx < set_minus_idx < branch_idx

