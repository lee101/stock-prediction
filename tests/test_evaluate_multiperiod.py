"""Tests for multi-period evaluation."""
import json
from pathlib import Path
import subprocess
import sys

import pytest


REPO = Path(__file__).resolve().parents[1]
CHECKPOINT_CANDIDATES = [
    REPO / "pufferlib_market/checkpoints/autoresearch/longonly_forecast/best.pt",
    REPO / "pufferlib_market/checkpoints/autoresearch/slip_5bps/best.pt",
    REPO / "pufferlib_market/checkpoints/autoresearch_daily/slip_5bps/best.pt",
]
DATA_PATH = REPO / "pufferlib_market/data/crypto6_val.bin"


def _checkpoint_for_test() -> str:
    for candidate in CHECKPOINT_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    pytest.skip("no compatible multiperiod smoke-test checkpoint is present in this workspace")


def test_multiperiod_runs_on_deployed_model():
    """Smoke test: run multi-period eval on the deployed longonly_forecast checkpoint."""
    result = subprocess.run(
        [
            sys.executable, "-m", "pufferlib_market.evaluate_multiperiod",
            "--checkpoint", _checkpoint_for_test(),
            "--data-path", str(DATA_PATH),
            "--deterministic", "--disable-shorts",
            "--periods", "1d,7d",
            "--json",
        ],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    data = json.loads(result.stdout)
    assert len(data) == 1  # one checkpoint
    results = data[0]
    assert len(results) == 2  # 1d, 7d
    for r in results:
        assert "total_return" in r
        assert "sortino" in r
        assert "period" in r


def test_multiperiod_table_output():
    """Verify table output format."""
    result = subprocess.run(
        [
            sys.executable, "-m", "pufferlib_market.evaluate_multiperiod",
            "--checkpoint", _checkpoint_for_test(),
            "--data-path", str(DATA_PATH),
            "--deterministic", "--disable-shorts",
            "--periods", "1d",
        ],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0
    assert "Period" in result.stdout
    assert "Return%" in result.stdout
    assert "1d" in result.stdout


def test_multiperiod_custom_period():
    """Test custom period specification (e.g. 48h)."""
    result = subprocess.run(
        [
            sys.executable, "-m", "pufferlib_market.evaluate_multiperiod",
            "--checkpoint", _checkpoint_for_test(),
            "--data-path", str(DATA_PATH),
            "--deterministic", "--disable-shorts",
            "--periods", "48h",
            "--json",
        ],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data[0][0]["eval_hours"] == 48
