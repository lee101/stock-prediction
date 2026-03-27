"""Tests for multi_period_eval() in evaluate_fast.py."""
import subprocess
import sys

import pytest


def test_multi_period_eval_structure():
    """Test that multi_period_eval raises on nonexistent paths, not AttributeError."""
    from pufferlib_market.evaluate_fast import multi_period_eval
    with pytest.raises((FileNotFoundError, RuntimeError, ValueError, OSError)):
        multi_period_eval("nonexistent.pt", "nonexistent.bin")


def test_multi_period_eval_importable():
    """Verify multi_period_eval is importable and has the right signature."""
    from pufferlib_market.evaluate_fast import multi_period_eval
    import inspect
    sig = inspect.signature(multi_period_eval)
    params = set(sig.parameters)
    assert "checkpoint_path" in params
    assert "data_path" in params
    assert "window_sizes" in params
    assert "n_windows_per_size" in params
    assert "smoothness_score" not in params  # it's a return key, not a param


def test_multi_period_eval_return_keys():
    """Verify multi_period_eval return dict has expected top-level keys."""
    from pufferlib_market.evaluate_fast import multi_period_eval
    import inspect
    # We can't run it without a real checkpoint, but we can check docstring/signature
    sig = inspect.signature(multi_period_eval)
    # Defaults are correct
    defaults = {k: v.default for k, v in sig.parameters.items() if v.default is not inspect.Parameter.empty}
    assert defaults["window_sizes"] == (5, 15, 30, 60, 90)
    assert defaults["n_windows_per_size"] == 8
    assert defaults["periods_per_year"] == 252.0


def test_cli_multi_windows_flag():
    """Verify --multi-windows flag appears in CLI help."""
    result = subprocess.run(
        [sys.executable, "-m", "pufferlib_market.evaluate_fast", "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"help failed: {result.stderr}"
    assert "--multi-windows" in result.stdout, "Missing --multi-windows flag in help"
    assert "--n-windows-per-size" in result.stdout, "Missing --n-windows-per-size flag in help"


def test_cli_multi_windows_error_on_bad_checkpoint():
    """--multi-windows with bad paths should exit non-zero, not crash with AttributeError."""
    result = subprocess.run(
        [
            sys.executable, "-m", "pufferlib_market.evaluate_fast",
            "--checkpoint", "nonexistent.pt",
            "--data-path", "nonexistent.bin",
            "--multi-windows", "5,15,30",
        ],
        capture_output=True, text=True, timeout=30,
    )
    # Should fail with a meaningful error, not a Python AttributeError traceback
    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "AttributeError" not in combined, f"Got unexpected AttributeError: {combined}"


def test_autoresearch_multi_period_eval_flag_in_help():
    """Verify --multi-period-eval flag appears in autoresearch_rl help."""
    result = subprocess.run(
        [sys.executable, "-m", "pufferlib_market.autoresearch_rl", "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"help failed: {result.stderr}"
    assert "--multi-period-eval" in result.stdout, "Missing --multi-period-eval in autoresearch_rl help"
    assert "--multi-period-windows" in result.stdout, "Missing --multi-period-windows in autoresearch_rl help"
    assert "--multi-period-n-per-size" in result.stdout, "Missing --multi-period-n-per-size in autoresearch_rl help"


def test_smooth_score_in_rank_metric_choices():
    """Verify replay- and smoothness-based metrics are valid --rank-metric choices."""
    result = subprocess.run(
        [sys.executable, "-m", "pufferlib_market.autoresearch_rl", "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert "smooth_score" in result.stdout, "smooth_score not in rank-metric choices"
    assert "replay_combo_score" in result.stdout, "replay_combo_score not in rank-metric choices"
