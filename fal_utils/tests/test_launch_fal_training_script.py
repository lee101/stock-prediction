from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "launch_fal_training.sh"


def _run_script(*args: str) -> list[str]:
    cmd = ["bash", str(SCRIPT_PATH), *args]
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    stdout = completed.stdout.strip()
    if not stdout:
        pytest.fail(f"Script produced no stdout. stderr={completed.stderr!r}")
    return stdout.splitlines()


def test_train_mode_defaults_produce_expected_payload():
    lines = _run_script("--dry-run", "--endpoint", "https://fal.run/fake-app")
    assert lines[0] == "POST https://fal.run/fake-app/api/train"
    payload = json.loads(lines[1])
    assert payload["trainer"] == "hf"
    assert payload["do_sweeps"] is True
    assert payload["sweeps"]["parallel_trials"] == 2
    assert payload["symbols"] == ["AAPL", "MSFT", "NVDA", "BTCUSD", "ETHUSD"]
    assert payload["run_name"].startswith("faltrain_")


def test_simulate_mode_defaults_follow_cli_overrides():
    lines = _run_script(
        "--dry-run",
        "--endpoint",
        "https://fal.run/fake-sim",
        "--endpoint-path",
        "/api/simulate",
        "--symbols",
        "TSLA,GOOG",
        "--steps",
        "12",
    )
    assert lines[0] == "POST https://fal.run/fake-sim/api/simulate"
    payload = json.loads(lines[1])
    assert payload["symbols"] == ["TSLA", "GOOG"]
    assert payload["steps"] == 12
    assert payload["step_size"] == 6
    assert payload["initial_cash"] == pytest.approx(100000.0)
    assert payload["compact_logs"] is True
