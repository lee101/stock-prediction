from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from pufferlib_market.autoresearch_rl import (
    TrialConfig,
    run_trial,
    summarize_replay_eval_payload,
    select_experiments,
    select_rank_score,
    summarize_holdout_payload,
    summarize_market_validation_payload,
)
from src.robust_trading_metrics import summarize_scenario_results


def test_summarize_holdout_payload_computes_robust_metrics() -> None:
    payload = {
        "summary": {
            "median_total_return": 0.04,
            "p10_total_return": -0.03,
            "median_sortino": 1.1,
            "p90_max_drawdown": 0.09,
        },
        "windows": [
            {
                "total_return": 0.10,
                "annualized_return": 0.80,
                "sortino": 1.8,
                "max_drawdown": 0.04,
                "num_trades": 4,
            },
            {
                "total_return": 0.03,
                "annualized_return": 0.20,
                "sortino": 0.9,
                "max_drawdown": 0.02,
                "num_trades": 3,
            },
            {
                "total_return": -0.06,
                "annualized_return": -0.50,
                "sortino": -0.4,
                "max_drawdown": 0.12,
                "num_trades": 2,
            },
        ],
    }

    summary = summarize_holdout_payload(payload)
    expected = summarize_scenario_results(
        [
            {
                "return_pct": 10.0,
                "annualized_return_pct": 80.0,
                "sortino": 1.8,
                "max_drawdown_pct": 4.0,
                "pnl_smoothness": 0.0,
                "trade_count": 4.0,
            },
            {
                "return_pct": 3.0,
                "annualized_return_pct": 20.0,
                "sortino": 0.9,
                "max_drawdown_pct": 2.0,
                "pnl_smoothness": 0.0,
                "trade_count": 3.0,
            },
            {
                "return_pct": -6.0,
                "annualized_return_pct": -50.0,
                "sortino": -0.4,
                "max_drawdown_pct": 12.0,
                "pnl_smoothness": 0.0,
                "trade_count": 2.0,
            },
        ]
    )

    assert summary["holdout_robust_score"] == pytest.approx(expected["robust_score"])
    assert summary["holdout_return_p25_pct"] == pytest.approx(expected["return_p25_pct"])
    assert summary["holdout_return_worst_pct"] == pytest.approx(expected["return_worst_pct"])
    assert summary["holdout_max_drawdown_worst_pct"] == pytest.approx(expected["max_drawdown_worst_pct"])
    assert summary["holdout_negative_return_rate"] == pytest.approx(expected["negative_return_rate"])
    assert summary["holdout_median_return_pct"] == pytest.approx(4.0)
    assert summary["holdout_p10_return_pct"] == pytest.approx(-3.0)
    assert summary["holdout_median_sortino"] == pytest.approx(1.1)
    assert summary["holdout_p90_max_drawdown_pct"] == pytest.approx(9.0)


def test_summarize_market_validation_payload_extracts_first_result() -> None:
    payload = [
        {
            "return_pct": 1.25,
            "sortino": 0.8,
            "max_drawdown_pct": 2.5,
            "trade_count": 7,
            "goodness_score": 3.6,
        }
    ]

    summary = summarize_market_validation_payload(payload)

    assert summary == {
        "market_return_pct": 1.25,
        "market_sortino": 0.8,
        "market_max_drawdown_pct": 2.5,
        "market_trade_count": 7.0,
        "market_goodness_score": 3.6,
    }


def test_summarize_replay_eval_payload_extracts_sections() -> None:
    payload = {
        "daily": {
            "total_return": 0.04,
            "sortino": 1.1,
            "max_drawdown": 0.08,
            "num_trades": 5,
        },
        "hourly_replay": {
            "total_return": 0.03,
            "sortino": 0.9,
            "max_drawdown": 0.12,
            "num_trades": 4,
            "num_orders": 9,
        },
        "hourly_policy": {
            "total_return": -0.06,
            "sortino": -0.4,
            "max_drawdown": 0.20,
            "num_trades": 8,
            "num_orders": 32,
        },
    }

    summary = summarize_replay_eval_payload(payload)

    assert summary == {
        "replay_daily_return_pct": 4.0,
        "replay_daily_sortino": 1.1,
        "replay_daily_max_drawdown_pct": 8.0,
        "replay_daily_trade_count": 5.0,
        "replay_hourly_return_pct": 3.0,
        "replay_hourly_sortino": 0.9,
        "replay_hourly_max_drawdown_pct": 12.0,
        "replay_hourly_trade_count": 4.0,
        "replay_hourly_order_count": 9.0,
        "replay_hourly_policy_return_pct": -6.0,
        "replay_hourly_policy_sortino": -0.4,
        "replay_hourly_policy_max_drawdown_pct": 20.0,
        "replay_hourly_policy_trade_count": 8.0,
        "replay_hourly_policy_order_count": 32.0,
    }


def test_select_rank_score_uses_expected_fallback_order() -> None:
    metrics = {
        "val_return": 0.04,
        "holdout_robust_score": 1.5,
        "market_goodness_score": 2.75,
    }

    assert select_rank_score(metrics, rank_metric="auto") == ("market_goodness_score", 2.75)
    assert select_rank_score(metrics, rank_metric="holdout_robust_score") == ("holdout_robust_score", 1.5)
    assert select_rank_score(metrics, rank_metric="val_return") == ("val_return", 0.04)
    assert select_rank_score({"replay_hourly_return_pct": 3.0}, rank_metric="auto") == ("replay_hourly_return_pct", 3.0)
    assert select_rank_score({"replay_hourly_policy_return_pct": -6.0}, rank_metric="replay_hourly_policy_return_pct") == (
        "replay_hourly_policy_return_pct",
        -6.0,
    )
    assert select_rank_score({"val_return": 0.01}, rank_metric="auto") == ("val_return", 0.01)
    assert select_rank_score({}, rank_metric="auto") == ("none", None)


def test_select_experiments_filters_by_description() -> None:
    exps = select_experiments(descriptions="ent_anneal,clip_vloss")
    assert [exp["description"] for exp in exps] == ["ent_anneal", "clip_vloss"]


def test_select_experiments_honors_start_offset_before_filter() -> None:
    exps = select_experiments(start_from=4, descriptions="clip_anneal,clip_vloss")
    assert [exp["description"] for exp in exps] == ["clip_anneal", "clip_vloss"]


def test_select_experiments_rejects_unknown_description() -> None:
    with pytest.raises(ValueError, match="Unknown experiment description"):
        select_experiments(descriptions="missing_trial")


def test_run_trial_passes_market_validation_decision_cadence(monkeypatch, tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "trial"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "best.pt").write_bytes(b"checkpoint")

    class _FakeStdout:
        def readline(self) -> bytes:
            return b""

    class _FakePopen:
        def __init__(self, *args, **kwargs) -> None:
            self.stdout = _FakeStdout()
            self.pid = 12345

        def poll(self) -> int:
            return 0

        def wait(self, timeout: float | None = None) -> int:
            return 0

    commands: list[list[str]] = []

    def _fake_run_capture(cmd: list[str], *, cwd: Path, timeout_s: int = 0) -> subprocess.CompletedProcess[str]:
        commands.append(cmd)
        if "pufferlib_market.evaluate" in cmd:
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=(
                    "Return: mean=0.10\n"
                    "Win rate: mean=0.55\n"
                    "Sortino: mean=1.20\n"
                    ">0: 10/10 (100.0%)\n"
                ),
                stderr="",
            )
        if "unified_orchestrator.market_validation" in cmd:
            out_idx = cmd.index("--write-json") + 1
            Path(cmd[out_idx]).write_text(
                json.dumps(
                    [
                        {
                            "return_pct": 2.0,
                            "sortino": 1.5,
                            "max_drawdown_pct": 1.0,
                            "trade_count": 3,
                            "goodness_score": 4.0,
                        }
                    ]
                )
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr("pufferlib_market.autoresearch_rl.subprocess.Popen", _FakePopen)
    monkeypatch.setattr("pufferlib_market.autoresearch_rl._run_capture", _fake_run_capture)

    result = run_trial(
        TrialConfig(description="test"),
        "train.bin",
        "val.bin",
        1,
        str(checkpoint_dir),
        market_validation_asset_class="crypto",
        market_validation_decision_cadence="daily",
    )

    market_cmd = next(cmd for cmd in commands if "unified_orchestrator.market_validation" in cmd)
    assert market_cmd[market_cmd.index("--decision-cadence") + 1] == "daily"
    assert result["market_goodness_score"] == pytest.approx(4.0)


def test_run_trial_collects_replay_eval_metrics(monkeypatch, tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "trial"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "best.pt").write_bytes(b"checkpoint")

    class _FakeStdout:
        def readline(self) -> bytes:
            return b""

    class _FakePopen:
        def __init__(self, *args, **kwargs) -> None:
            self.stdout = _FakeStdout()
            self.pid = 12345

        def poll(self) -> int:
            return 0

        def wait(self, timeout: float | None = None) -> int:
            return 0

    commands: list[list[str]] = []

    def _fake_run_capture(cmd: list[str], *, cwd: Path, timeout_s: int = 0) -> subprocess.CompletedProcess[str]:
        commands.append(cmd)
        if "pufferlib_market.evaluate" in cmd:
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=(
                    "Return: mean=0.10\n"
                    "Win rate: mean=0.55\n"
                    "Sortino: mean=1.20\n"
                    ">0: 10/10 (100.0%)\n"
                ),
                stderr="",
            )
        if "pufferlib_market.replay_eval" in cmd:
            out_idx = cmd.index("--output-json") + 1
            Path(cmd[out_idx]).write_text(
                json.dumps(
                    {
                        "daily": {"total_return": 0.04, "sortino": 1.1, "max_drawdown": 0.08, "num_trades": 5},
                        "hourly_replay": {
                            "total_return": 0.03,
                            "sortino": 0.9,
                            "max_drawdown": 0.12,
                            "num_trades": 4,
                            "num_orders": 9,
                        },
                        "hourly_policy": {
                            "total_return": -0.06,
                            "sortino": -0.4,
                            "max_drawdown": 0.20,
                            "num_trades": 8,
                            "num_orders": 32,
                        },
                    }
                )
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr("pufferlib_market.autoresearch_rl.subprocess.Popen", _FakePopen)
    monkeypatch.setattr("pufferlib_market.autoresearch_rl._run_capture", _fake_run_capture)

    result = run_trial(
        TrialConfig(description="test"),
        "train.bin",
        "val.bin",
        1,
        str(checkpoint_dir),
        replay_eval_hourly_root="trainingdatahourly",
        replay_eval_start_date="2025-06-01",
        replay_eval_end_date="2026-02-05",
        replay_eval_run_hourly_policy=True,
        rank_metric="replay_hourly_return_pct",
    )

    replay_cmd = next(cmd for cmd in commands if "pufferlib_market.replay_eval" in cmd)
    assert replay_cmd[replay_cmd.index("--hourly-data-root") + 1] == "trainingdatahourly"
    assert replay_cmd[replay_cmd.index("--fill-buffer-bps") + 1] == "5.0"
    assert "--run-hourly-policy" in replay_cmd
    assert result["replay_hourly_return_pct"] == pytest.approx(3.0)
    assert result["replay_hourly_policy_order_count"] == pytest.approx(32.0)
    assert result["rank_metric"] == "replay_hourly_return_pct"
    assert result["rank_score"] == pytest.approx(3.0)


def test_run_trial_surfaces_train_failure_when_no_checkpoint(monkeypatch, tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "trial"
    checkpoint_dir.mkdir()

    class _FakeStdout:
        def readline(self) -> bytes:
            return b""

        def read(self) -> bytes:
            return (
                b"market_data_load: cannot open missing.bin\n"
                b"RuntimeError: Failed to load market data from missing.bin\n"
            )

    class _FakePopen:
        def __init__(self, *args, **kwargs) -> None:
            self.stdout = _FakeStdout()
            self.pid = 12345

        def poll(self) -> int:
            return 1

        def wait(self, timeout: float | None = None) -> int:
            return 1

    monkeypatch.setattr("pufferlib_market.autoresearch_rl.subprocess.Popen", _FakePopen)

    result = run_trial(
        TrialConfig(description="broken"),
        "train.bin",
        "val.bin",
        1,
        str(checkpoint_dir),
    )

    assert result["train_steps"] == 0
    assert "train failed (exit 1)" in result["error"]
    assert "Failed to load market data" in result["error"]


def test_run_trial_passes_holdout_fill_buffer_bps(monkeypatch, tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "trial"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "best.pt").write_bytes(b"checkpoint")

    class _FakeStdout:
        def readline(self) -> bytes:
            return b""

    class _FakePopen:
        def __init__(self, *args, **kwargs) -> None:
            self.stdout = _FakeStdout()
            self.pid = 12345

        def poll(self) -> int:
            return 0

        def wait(self, timeout: float | None = None) -> int:
            return 0

    commands: list[list[str]] = []

    def _fake_run_capture(cmd: list[str], *, cwd: Path, timeout_s: int = 0) -> subprocess.CompletedProcess[str]:
        commands.append(cmd)
        if "pufferlib_market.evaluate" in cmd:
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=(
                    "Return: mean=0.10\n"
                    "Win rate: mean=0.55\n"
                    "Sortino: mean=1.20\n"
                    ">0: 10/10 (100.0%)\n"
                ),
                stderr="",
            )
        if "pufferlib_market.evaluate_holdout" in cmd:
            out_idx = cmd.index("--out") + 1
            Path(cmd[out_idx]).write_text(json.dumps({"summary": {}, "windows": []}))
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr("pufferlib_market.autoresearch_rl.subprocess.Popen", _FakePopen)
    monkeypatch.setattr("pufferlib_market.autoresearch_rl._run_capture", _fake_run_capture)

    run_trial(
        TrialConfig(description="test"),
        "train.bin",
        "val.bin",
        1,
        str(checkpoint_dir),
        holdout_data="holdout.bin",
        holdout_eval_steps=10,
        holdout_n_windows=3,
        holdout_fill_buffer_bps=7.5,
    )

    holdout_cmd = next(cmd for cmd in commands if "pufferlib_market.evaluate_holdout" in cmd)
    assert holdout_cmd[holdout_cmd.index("--fill-buffer-bps") + 1] == "7.5"


def test_run_trial_passes_risk_penalties_to_train_command(monkeypatch, tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "trial"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "best.pt").write_bytes(b"checkpoint")

    class _FakeStdout:
        def readline(self) -> bytes:
            return b""

    train_commands: list[list[str]] = []

    class _FakePopen:
        def __init__(self, cmd, *args, **kwargs) -> None:
            train_commands.append(list(cmd))
            self.stdout = _FakeStdout()
            self.pid = 12345

        def poll(self) -> int:
            return 0

        def wait(self, timeout: float | None = None) -> int:
            return 0

    def _fake_run_capture(cmd: list[str], *, cwd: Path, timeout_s: int = 0) -> subprocess.CompletedProcess[str]:
        if "pufferlib_market.evaluate" in cmd:
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=(
                    "Return: mean=0.10\n"
                    "Win rate: mean=0.55\n"
                    "Sortino: mean=1.20\n"
                    ">0: 10/10 (100.0%)\n"
                ),
                stderr="",
            )
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr("pufferlib_market.autoresearch_rl.subprocess.Popen", _FakePopen)
    monkeypatch.setattr("pufferlib_market.autoresearch_rl._run_capture", _fake_run_capture)

    run_trial(
        TrialConfig(
            description="risk_knobs",
            drawdown_penalty=0.02,
            downside_penalty=0.3,
            smooth_downside_penalty=0.2,
            smooth_downside_temperature=0.05,
            smoothness_penalty=0.01,
            advantage_norm="group_relative",
            group_relative_size=16,
            group_relative_mix=0.25,
            group_relative_clip=1.5,
        ),
        "train.bin",
        "val.bin",
        1,
        str(checkpoint_dir),
    )

    assert train_commands, "expected training command to be invoked"
    train_cmd = train_commands[0]
    assert train_cmd[train_cmd.index("--drawdown-penalty") + 1] == "0.02"
    assert train_cmd[train_cmd.index("--downside-penalty") + 1] == "0.3"
    assert train_cmd[train_cmd.index("--smooth-downside-penalty") + 1] == "0.2"
    assert train_cmd[train_cmd.index("--smooth-downside-temperature") + 1] == "0.05"
    assert train_cmd[train_cmd.index("--smoothness-penalty") + 1] == "0.01"
    assert train_cmd[train_cmd.index("--advantage-norm") + 1] == "group_relative"
    assert train_cmd[train_cmd.index("--group-relative-size") + 1] == "16"
    assert train_cmd[train_cmd.index("--group-relative-mix") + 1] == "0.25"
    assert train_cmd[train_cmd.index("--group-relative-clip") + 1] == "1.5"
