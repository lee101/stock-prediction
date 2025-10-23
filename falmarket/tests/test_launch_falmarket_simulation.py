from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "launch_falmarket_simulation.sh"


def _run_script(*args: str) -> list[str]:
    cmd = ["bash", str(SCRIPT_PATH), *args]
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    stdout = completed.stdout.strip()
    if not stdout:
        pytest.fail(f"Script produced no stdout. stderr={completed.stderr!r}")
    return stdout.splitlines()


def test_defaults_match_expected_payload():
    lines = _run_script("--dry-run", "--endpoint", "https://fal.run/fake-market")
    assert lines[0] == "POST https://fal.run/fake-market/api/simulate"
    payload = json.loads(lines[1])
    assert payload["symbols"] == ["AAPL", "MSFT", "NVDA"]
    assert payload["steps"] == 32
    assert payload["step_size"] == 1
    assert payload["initial_cash"] == pytest.approx(100000.0)
    assert payload["top_k"] == 4
    assert payload["kronos_only"] is False
    assert payload["compact_logs"] is True


def test_cli_overrides_apply_to_payload():
    lines = _run_script(
        "--dry-run",
        "--endpoint",
        "https://fal.run/custom",
        "--symbols",
        "TSLA, goog  ,",
        "--steps",
        "48",
        "--step-size",
        "4",
        "--initial-cash",
        "250000",
        "--top-k",
        "6",
        "--kronos-only",
        "--no-compact-logs",
    )
    assert lines[0] == "POST https://fal.run/custom/api/simulate"
    payload = json.loads(lines[1])
    assert payload["symbols"] == ["TSLA", "GOOG"]
    assert payload["steps"] == 48
    assert payload["step_size"] == 4
    assert payload["initial_cash"] == pytest.approx(250000.0)
    assert payload["top_k"] == 6
    assert payload["kronos_only"] is True
    assert payload["compact_logs"] is False
