"""Tests for Qwen autoresearch filtering and remote launch plumbing."""

from pathlib import Path

from qwen_rl_trading.autoresearch_qwen import EXPERIMENTS, run_autoresearch
from qwen_rl_trading.launch_forever import run_qwen_batch_remote


def test_run_autoresearch_filters_requested_model_sizes(monkeypatch, tmp_path):
    seen_model_sizes: list[str] = []

    def fake_run_trial(trial_config, time_budget, checkpoint_root):
        seen_model_sizes.append(trial_config.model_size)
        return {
            "description": trial_config.description,
            "val_mean_reward": 1.0,
            "val_mean_sortino": 1.0,
            "val_mean_return": 0.1,
            "val_valid_json_pct": 1.0,
            "best_checkpoint": str(Path(checkpoint_root) / trial_config.description),
            "training_time_s": 1.0,
        }

    monkeypatch.setattr("qwen_rl_trading.autoresearch_qwen.run_trial", fake_run_trial)

    result = run_autoresearch(
        EXPERIMENTS,
        time_budget=1,
        max_trials=100,
        checkpoint_root=tmp_path / "checkpoints",
        leaderboard_path=tmp_path / "leaderboard.csv",
        model_sizes=["0.6B", "1.8B"],
    )

    assert seen_model_sizes
    assert set(seen_model_sizes) <= {"0.6B", "1.8B"}
    assert "3B" not in seen_model_sizes
    assert "7B" not in seen_model_sizes
    assert result["best_description"] is not None


def test_run_qwen_batch_remote_forwards_model_sizes(monkeypatch):
    captured: dict[str, object] = {}

    def fake_ssh_run(host, port, cmd, check=False):
        captured["host"] = host
        captured["port"] = port
        captured["cmd"] = cmd
        captured["check"] = check

    monkeypatch.setattr("qwen_rl_trading.launch_forever._ssh_run", fake_ssh_run)
    monkeypatch.setattr("qwen_rl_trading.launch_forever._scp_from_pod", lambda *args, **kwargs: None)
    monkeypatch.setattr("qwen_rl_trading.launch_forever._rsync_from_pod", lambda *args, **kwargs: None)

    run_qwen_batch_remote(
        "1.2.3.4",
        2222,
        "/workspace/stock-prediction",
        ["0.6B", "1.8B"],
        time_budget=600,
        max_trials=3,
    )

    assert captured["host"] == "1.2.3.4"
    assert captured["port"] == 2222
    assert captured["check"] is False
    assert "--model-sizes 0.6B,1.8B" in str(captured["cmd"])
