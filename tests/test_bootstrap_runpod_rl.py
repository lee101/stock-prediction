"""Tests for pufferlib_market.bootstrap_runpod_rl — H100 detection and training overrides."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from pufferlib_market.bootstrap_runpod_rl import (
    detect_h100,
    get_h100_training_overrides,
)


# ---------------------------------------------------------------------------
# detect_h100 — environment variable path
# ---------------------------------------------------------------------------

def test_detect_h100_true_via_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNPOD_GPU_TYPE", "NVIDIA H100 80GB HBM3")
    assert detect_h100() is True


def test_detect_h100_false_via_env_var_a100(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNPOD_GPU_TYPE", "NVIDIA A100 80GB PCIe")
    assert detect_h100() is False


def test_detect_h100_true_case_insensitive_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNPOD_GPU_TYPE", "nvidia h100 sxm")
    assert detect_h100() is True


# ---------------------------------------------------------------------------
# detect_h100 — nvidia-smi path (no env var set)
# ---------------------------------------------------------------------------

def test_detect_h100_true_via_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RUNPOD_GPU_TYPE", raising=False)

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "NVIDIA H100 80GB HBM3\n"

    with patch("subprocess.run", return_value=mock_result):
        assert detect_h100() is True


def test_detect_h100_false_via_nvidia_smi_a100(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RUNPOD_GPU_TYPE", raising=False)

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "NVIDIA A100 80GB PCIe\n"

    with patch("subprocess.run", return_value=mock_result):
        assert detect_h100() is False


def test_detect_h100_false_when_nvidia_smi_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RUNPOD_GPU_TYPE", raising=False)

    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""

    with patch("subprocess.run", return_value=mock_result):
        assert detect_h100() is False


def test_detect_h100_false_when_nvidia_smi_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """detect_h100 must not raise even when nvidia-smi is not installed."""
    monkeypatch.delenv("RUNPOD_GPU_TYPE", raising=False)

    with patch("subprocess.run", side_effect=FileNotFoundError("nvidia-smi not found")):
        assert detect_h100() is False


def test_detect_h100_false_when_nvidia_smi_empty_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RUNPOD_GPU_TYPE", raising=False)

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = ""

    with patch("subprocess.run", return_value=mock_result):
        assert detect_h100() is False


# ---------------------------------------------------------------------------
# get_h100_training_overrides
# ---------------------------------------------------------------------------

def test_get_h100_training_overrides_returns_list() -> None:
    overrides = get_h100_training_overrides()
    assert isinstance(overrides, list)
    assert len(overrides) > 0


def test_get_h100_training_overrides_includes_num_envs_256() -> None:
    overrides = get_h100_training_overrides()
    assert "--num-envs" in overrides
    idx = overrides.index("--num-envs")
    assert overrides[idx + 1] == "256"


def test_get_h100_training_overrides_includes_cuda_graph_ppo() -> None:
    overrides = get_h100_training_overrides()
    assert "--cuda-graph-ppo" in overrides


def test_get_h100_training_overrides_is_deterministic() -> None:
    """Calling twice returns the same overrides."""
    assert get_h100_training_overrides() == get_h100_training_overrides()
