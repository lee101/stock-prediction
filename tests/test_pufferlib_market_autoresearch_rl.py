from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from pufferlib_market.autoresearch_rl import (
    EXPERIMENTS,
    TrialConfig,
    _read_checkpoint_global_step,
    _read_mktd_header,
    _termination_grace_timeout_s,
    annualize_total_return_pct,
    compute_eval_window_years,
    main,
    run_trial,
    summarize_replay_eval_payload,
    select_experiments,
    select_rank_score,
    summarize_holdout_payload,
    summarize_market_validation_payload,
)
from src.robust_trading_metrics import compute_replay_composite_score, summarize_scenario_results


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
            "pnl_smoothness": 0.001,
            "ulcer_index": 0.2,
            "goodness_score": 4.25,
        },
        "hourly_replay": {
            "total_return": 0.03,
            "sortino": 0.9,
            "max_drawdown": 0.12,
            "num_trades": 4,
            "num_orders": 9,
            "pnl_smoothness": 0.002,
            "ulcer_index": 0.35,
            "goodness_score": 3.75,
        },
        "hourly_policy": {
            "total_return": -0.06,
            "sortino": -0.4,
            "max_drawdown": 0.20,
            "num_trades": 8,
            "num_orders": 32,
            "pnl_smoothness": 0.004,
            "ulcer_index": 0.8,
            "goodness_score": -2.0,
        },
    }

    summary = summarize_replay_eval_payload(payload, replay_eval_years=1.0)

    assert summary["replay_daily_return_pct"] == pytest.approx(4.0)
    assert summary["replay_daily_annualized_return_pct"] == pytest.approx(4.0)
    assert summary["replay_daily_sortino"] == pytest.approx(1.1)
    assert summary["replay_daily_max_drawdown_pct"] == pytest.approx(8.0)
    assert summary["replay_daily_trade_count"] == pytest.approx(5.0)
    assert summary["replay_daily_pnl_smoothness"] == pytest.approx(0.001)
    assert summary["replay_daily_ulcer_index"] == pytest.approx(0.2)
    assert summary["replay_daily_goodness_score"] == pytest.approx(4.25)
    assert summary["replay_hourly_return_pct"] == pytest.approx(3.0)
    assert summary["replay_hourly_annualized_return_pct"] == pytest.approx(3.0)
    assert summary["replay_hourly_sortino"] == pytest.approx(0.9)
    assert summary["replay_hourly_max_drawdown_pct"] == pytest.approx(12.0)
    assert summary["replay_hourly_trade_count"] == pytest.approx(4.0)
    assert summary["replay_hourly_order_count"] == pytest.approx(9.0)
    assert summary["replay_hourly_pnl_smoothness"] == pytest.approx(0.002)
    assert summary["replay_hourly_ulcer_index"] == pytest.approx(0.35)
    assert summary["replay_hourly_goodness_score"] == pytest.approx(3.75)
    assert summary["replay_hourly_policy_return_pct"] == pytest.approx(-6.0)
    assert summary["replay_hourly_policy_annualized_return_pct"] == pytest.approx(-6.0)
    assert summary["replay_hourly_policy_sortino"] == pytest.approx(-0.4)
    assert summary["replay_hourly_policy_max_drawdown_pct"] == pytest.approx(20.0)
    assert summary["replay_hourly_policy_trade_count"] == pytest.approx(8.0)
    assert summary["replay_hourly_policy_order_count"] == pytest.approx(32.0)
    assert summary["replay_hourly_policy_pnl_smoothness"] == pytest.approx(0.004)
    assert summary["replay_hourly_policy_ulcer_index"] == pytest.approx(0.8)
    assert summary["replay_hourly_policy_goodness_score"] == pytest.approx(-2.0)
    assert summary["replay_combo_scenario_count"] == pytest.approx(4.0)
    assert "replay_combo_score" in summary

    expected_combo = compute_replay_composite_score(
        daily_return_pct=4.0,
        daily_annualized_return_pct=4.0,
        daily_sortino=1.1,
        daily_max_drawdown_pct=8.0,
        daily_pnl_smoothness=0.001,
        daily_trade_count=5.0,
        hourly_return_pct=3.0,
        hourly_annualized_return_pct=3.0,
        hourly_sortino=0.9,
        hourly_max_drawdown_pct=12.0,
        hourly_pnl_smoothness=0.002,
        hourly_trade_count=4.0,
        hourly_policy_return_pct=-6.0,
        hourly_policy_annualized_return_pct=-6.0,
        hourly_policy_sortino=-0.4,
        hourly_policy_max_drawdown_pct=20.0,
        hourly_policy_pnl_smoothness=0.004,
        hourly_policy_trade_count=8.0,
    )
    for key, value in expected_combo.items():
        assert summary[key] == pytest.approx(value)


def test_summarize_replay_eval_payload_extracts_robust_sections() -> None:
    payload = {
        "robust_start_summary": {
            "daily": {
                "median_total_return": 0.01,
                "worst_total_return": -0.02,
                "worst_sortino": -0.4,
                "worst_max_drawdown": 0.15,
            },
            "hourly_replay": {
                "median_total_return": 0.03,
                "worst_total_return": -0.04,
                "worst_sortino": -0.7,
                "worst_max_drawdown": 0.22,
            },
            "hourly_policy": {
                "median_total_return": -0.01,
                "worst_total_return": -0.08,
                "worst_sortino": -1.2,
                "worst_max_drawdown": 0.35,
            },
        }
    }

    summary = summarize_replay_eval_payload(payload, replay_eval_years=0.5)

    assert summary["replay_daily_robust_median_return_pct"] == pytest.approx(1.0)
    assert summary["replay_daily_robust_median_annualized_return_pct"] == pytest.approx(2.01, abs=0.01)
    assert summary["replay_daily_robust_worst_return_pct"] == pytest.approx(-2.0)
    assert summary["replay_daily_robust_worst_annualized_return_pct"] == pytest.approx(-3.96, abs=0.01)
    assert summary["replay_hourly_robust_worst_sortino"] == pytest.approx(-0.7)
    assert summary["replay_hourly_policy_robust_worst_max_drawdown_pct"] == pytest.approx(35.0)
    assert summary["replay_hourly_policy_robust_worst_annualized_return_pct"] == pytest.approx(-15.36, abs=0.01)


def test_compute_eval_window_years_handles_date_and_datetime_inputs() -> None:
    assert compute_eval_window_years("2025-06-01", "2026-02-05") == pytest.approx(250.0 / 365.0)
    assert compute_eval_window_years("2025-06-01T00:00:00+00:00", "2025-12-01T00:00:00+00:00") == pytest.approx(
        183.0 / 365.0
    )


def test_annualize_total_return_pct_handles_edge_cases() -> None:
    assert annualize_total_return_pct(25.0, 1.0) == pytest.approx(25.0)
    assert annualize_total_return_pct(-100.0, 0.5) is None
    assert annualize_total_return_pct(5.0, 0.0) is None


def test_termination_grace_timeout_scales_with_budget() -> None:
    assert _termination_grace_timeout_s(1) == 30
    assert _termination_grace_timeout_s(60) == 90
    assert _termination_grace_timeout_s(300) == 180


def test_select_rank_score_uses_expected_fallback_order() -> None:
    metrics = {
        "val_return": 0.04,
        "holdout_robust_score": 1.5,
        "market_goodness_score": 2.75,
        "replay_combo_score": 3.25,
    }

    assert select_rank_score(metrics, rank_metric="auto") == ("replay_combo_score", 3.25)
    assert select_rank_score(metrics, rank_metric="replay_combo_score") == ("replay_combo_score", 3.25)
    assert select_rank_score(metrics, rank_metric="holdout_robust_score") == ("holdout_robust_score", 1.5)
    assert select_rank_score(metrics, rank_metric="val_return") == ("val_return", 0.04)
    assert select_rank_score(
        {
            "replay_hourly_policy_robust_worst_return_pct": -1.0,
            "replay_hourly_robust_worst_return_pct": -2.0,
            "replay_hourly_return_pct": 3.0,
        },
        rank_metric="auto",
    ) == ("replay_hourly_policy_robust_worst_return_pct", -1.0)
    assert select_rank_score(
        {"replay_hourly_robust_worst_return_pct": -2.0, "replay_hourly_return_pct": 3.0},
        rank_metric="auto",
    ) == ("replay_hourly_robust_worst_return_pct", -2.0)
    assert select_rank_score({"replay_hourly_policy_return_pct": -6.0}, rank_metric="auto") == (
        "replay_hourly_policy_return_pct",
        -6.0,
    )
    assert select_rank_score({"replay_hourly_return_pct": 3.0}, rank_metric="auto") == ("replay_hourly_return_pct", 3.0)
    assert select_rank_score({"replay_hourly_policy_return_pct": -6.0}, rank_metric="replay_hourly_policy_return_pct") == (
        "replay_hourly_policy_return_pct",
        -6.0,
    )
    assert select_rank_score(
        {"replay_daily_annualized_return_pct": 21.5},
        rank_metric="replay_daily_annualized_return_pct",
    ) == ("replay_daily_annualized_return_pct", 21.5)
    assert select_rank_score(
        {"replay_hourly_policy_robust_worst_return_pct": -8.0},
        rank_metric="replay_hourly_policy_robust_worst_return_pct",
    ) == ("replay_hourly_policy_robust_worst_return_pct", -8.0)
    assert select_rank_score(
        {"replay_hourly_policy_robust_worst_annualized_return_pct": 88.0},
        rank_metric="replay_hourly_policy_robust_worst_annualized_return_pct",
    ) == ("replay_hourly_policy_robust_worst_annualized_return_pct", 88.0)
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


def test_experiments_have_unique_descriptions() -> None:
    descriptions = [cfg["description"] for cfg in EXPERIMENTS if not cfg["description"].startswith("random_")]
    assert len(descriptions) == len(set(descriptions))


def test_sortino_rc3_followup_variants_are_registered() -> None:
    exps = select_experiments(
        descriptions=",".join(
            [
                "sortino_rc3_tp08",
                "sortino_rc3_tp07",
                "sortino_rc3_tp09",
                "sortino_rc3_tp08_slip8",
                "sortino_rc3_tp08_wd01",
                "sortino_rc3_tp08_sm001",
                "sortino_rc3_tp08_dd002",
            ]
        )
    )

    got = {exp["description"] for exp in exps}
    assert got == {
        "sortino_rc3_tp08",
        "sortino_rc3_tp07",
        "sortino_rc3_tp09",
        "sortino_rc3_tp08_slip8",
        "sortino_rc3_tp08_wd01",
        "sortino_rc3_tp08_sm001",
        "sortino_rc3_tp08_dd002",
    }


def test_stability_long_run_variants_are_registered() -> None:
    exps = select_experiments(
        descriptions="stable_long_tp005_sds02,stable_long_tp005_sds02_bf16,stable_long_reg_combo_2"
    )

    got = {exp["description"] for exp in exps}
    assert got == {
        "stable_long_tp005_sds02",
        "stable_long_tp005_sds02_bf16",
        "stable_long_reg_combo_2",
    }
def test_main_list_experiments_respects_description_filter(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["autoresearch_rl", "--list-experiments", "--descriptions", "ent_anneal,clip_vloss"],
    )

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 0
    assert capsys.readouterr().out.strip().splitlines() == ["ent_anneal", "clip_vloss"]


def test_main_describe_experiments_prints_non_default_overrides(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["autoresearch_rl", "--describe-experiments", "--descriptions", "clip_vloss"],
    )

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "clip_vloss\n" in out
    assert "  clip_vloss=True" in out
    assert "  lr=" not in out
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
                        "robust_start_summary": {
                            "hourly_replay": {
                                "median_total_return": 0.01,
                                "worst_total_return": -0.04,
                                "worst_sortino": -0.7,
                                "worst_max_drawdown": 0.22,
                            },
                            "hourly_policy": {
                                "median_total_return": -0.02,
                                "worst_total_return": -0.08,
                                "worst_sortino": -1.1,
                                "worst_max_drawdown": 0.31,
                            },
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
        replay_eval_robust_start_states="flat,long:AAA:0.25",
        rank_metric="replay_hourly_policy_robust_worst_return_pct",
    )

    replay_cmd = next(cmd for cmd in commands if "pufferlib_market.replay_eval" in cmd)
    assert replay_cmd[replay_cmd.index("--hourly-data-root") + 1] == "trainingdatahourly"
    assert replay_cmd[replay_cmd.index("--fill-buffer-bps") + 1] == "5.0"
    assert replay_cmd[replay_cmd.index("--robust-start-states") + 1] == "flat,long:AAA:0.25"
    assert "--run-hourly-policy" in replay_cmd
    assert result["replay_hourly_return_pct"] == pytest.approx(3.0)
    assert result["replay_hourly_robust_worst_return_pct"] == pytest.approx(-4.0)
    assert result["replay_hourly_policy_robust_worst_return_pct"] == pytest.approx(-8.0)
    assert result["replay_hourly_policy_order_count"] == pytest.approx(32.0)
    assert result["rank_metric"] == "replay_hourly_policy_robust_worst_return_pct"
    assert result["rank_score"] == pytest.approx(-8.0)


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


def test_run_trial_falls_back_to_final_checkpoint(monkeypatch, tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "trial"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "final.pt").write_bytes(b"checkpoint")

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

    eval_commands: list[list[str]] = []

    def _fake_run_capture(cmd: list[str], *, cwd: Path, timeout_s: int = 0) -> subprocess.CompletedProcess[str]:
        if "pufferlib_market.evaluate" in cmd:
            eval_commands.append(cmd)
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout="Return: mean=0.10\nWin rate: mean=0.55\nSortino: mean=1.20\n>0: 10/10 (100.0%)\n",
                stderr="",
            )
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr("pufferlib_market.autoresearch_rl.subprocess.Popen", _FakePopen)
    monkeypatch.setattr("pufferlib_market.autoresearch_rl._run_capture", _fake_run_capture)

    result = run_trial(
        TrialConfig(description="final_only"),
        "train.bin",
        "val.bin",
        1,
        str(checkpoint_dir),
    )

    assert eval_commands, "expected validation eval to run from final.pt"
    eval_cmd = eval_commands[0]
    assert eval_cmd[eval_cmd.index("--checkpoint") + 1] == str(checkpoint_dir / "final.pt")
    assert result["val_return"] == pytest.approx(0.10)
    assert result["val_sortino"] == pytest.approx(1.20)


def test_run_trial_respects_eval_num_episodes(monkeypatch, tmp_path: Path) -> None:
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

    eval_commands: list[list[str]] = []

    def _fake_run_capture(cmd: list[str], *, cwd: Path, timeout_s: int = 0) -> subprocess.CompletedProcess[str]:
        if "pufferlib_market.evaluate" in cmd:
            eval_commands.append(cmd)
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout="Return: mean=0.10\nWin rate: mean=0.55\nSortino: mean=1.20\n>0: 10/10 (100.0%)\n",
                stderr="",
            )
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr("pufferlib_market.autoresearch_rl.subprocess.Popen", _FakePopen)
    monkeypatch.setattr("pufferlib_market.autoresearch_rl._run_capture", _fake_run_capture)

    run_trial(
        TrialConfig(description="eval_episodes", eval_num_episodes=7),
        "train.bin",
        "val.bin",
        1,
        str(checkpoint_dir),
    )

    assert eval_commands
    eval_cmd = eval_commands[0]
    assert eval_cmd[eval_cmd.index("--num-episodes") + 1] == "7"


def test_read_checkpoint_global_step_reads_saved_value(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"global_step": 12345}, checkpoint_path)

    assert _read_checkpoint_global_step(checkpoint_path) == 12345


def test_run_trial_caps_replay_eval_max_steps_to_data_length(monkeypatch, tmp_path: Path) -> None:
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
            Path(cmd[out_idx]).write_text(json.dumps({"hourly_replay": {"total_return": 0.01, "sortino": 0.2, "max_drawdown": 0.03}}))
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise AssertionError(f"Unexpected command: {cmd}")

    def _fake_read_mktd_header(path: str) -> tuple[int, int]:
        if path == "train.bin":
            return (12, 1000)
        if path == "val.bin":
            return (12, 21)
        return (12, 1000)

    monkeypatch.setattr("pufferlib_market.autoresearch_rl.subprocess.Popen", _FakePopen)
    monkeypatch.setattr("pufferlib_market.autoresearch_rl._run_capture", _fake_run_capture)
    monkeypatch.setattr("pufferlib_market.autoresearch_rl._read_mktd_header", _fake_read_mktd_header)

    run_trial(
        TrialConfig(description="test", max_steps=90),
        "train.bin",
        "val.bin",
        1,
        str(checkpoint_dir),
        replay_eval_hourly_root="trainingdatahourly",
        replay_eval_start_date="2025-06-01",
        replay_eval_end_date="2025-06-21",
    )

    replay_cmd = next(cmd for cmd in commands if "pufferlib_market.replay_eval" in cmd)
    assert replay_cmd[replay_cmd.index("--max-steps") + 1] == "20"


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


def test_main_creates_leaderboard_parent_directory(monkeypatch, tmp_path: Path) -> None:
    leaderboard = tmp_path / "nested" / "leaderboard.csv"
    checkpoint_root = tmp_path / "checkpoints"

    def _fake_run_trial(*args, **kwargs) -> dict[str, object]:
        return {
            "rank_metric": "val_return",
            "rank_score": 1.0,
            "val_return": 1.0,
            "replay_hourly_robust_worst_return_pct": -2.0,
        }

    monkeypatch.setattr("pufferlib_market.autoresearch_rl.run_trial", _fake_run_trial)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "autoresearch_rl.py",
            "--stocks12",
            "--time-budget",
            "1",
            "--max-trials",
            "1",
            "--leaderboard",
            str(leaderboard),
            "--checkpoint-root",
            str(checkpoint_root),
            "--descriptions",
            "stock_trade_pen_05",
        ],
    )

    main()

    assert leaderboard.exists()
    assert "replay_hourly_robust_worst_return_pct" in leaderboard.read_text()


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


def test_run_trial_caps_total_timesteps(monkeypatch, tmp_path: Path) -> None:
    import struct
    # Create a minimal fake MKTD header
    train_bin = tmp_path / "train.bin"
    num_symbols = 10
    num_timesteps = 100
    header = struct.pack("<4sIIIII", b"MKTD", 1, num_symbols, num_timesteps, 16, 0)
    header += b"\x00" * (64 - len(header))
    train_bin.write_bytes(header)

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
                cmd, 0,
                stdout="Return: mean=0.10\nWin rate: mean=0.55\nSortino: mean=1.20\n>0: 10/10 (100.0%)\n",
                stderr="",
            )
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr("pufferlib_market.autoresearch_rl.subprocess.Popen", _FakePopen)
    monkeypatch.setattr("pufferlib_market.autoresearch_rl._run_capture", _fake_run_capture)

    run_trial(
        TrialConfig(description="test_cap"),
        str(train_bin),
        "val.bin",
        1,
        str(checkpoint_dir),
        max_timesteps_per_sample=500,
    )

    assert train_commands
    train_cmd = train_commands[0]
    total_ts = int(train_cmd[train_cmd.index("--total-timesteps") + 1])
    # 10 symbols * 100 timesteps * 500 = 500,000
    assert total_ts == 500_000


def test_run_trial_result_includes_early_rejected(monkeypatch, tmp_path: Path) -> None:
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

    def _fake_run_capture(cmd: list[str], *, cwd: Path, timeout_s: int = 0) -> subprocess.CompletedProcess[str]:
        if "pufferlib_market.evaluate" in cmd:
            return subprocess.CompletedProcess(
                cmd, 0,
                stdout="Return: mean=0.10\nWin rate: mean=0.55\nSortino: mean=1.20\n>0: 10/10 (100.0%)\n",
                stderr="",
            )
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr("pufferlib_market.autoresearch_rl.subprocess.Popen", _FakePopen)
    monkeypatch.setattr("pufferlib_market.autoresearch_rl._run_capture", _fake_run_capture)

    result = run_trial(
        TrialConfig(description="test_rej"),
        "train.bin",
        "val.bin",
        1,
        str(checkpoint_dir),
    )

    assert "early_rejected" in result
    assert result["early_rejected"] is False


def test_run_trial_uses_baseline_val_return_floor_for_early_reject(monkeypatch, tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "trial"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "best.pt").write_bytes(b"checkpoint")

    class _FakeStdout:
        def readline(self) -> bytes:
            return b""

        def read(self) -> bytes:
            return b""

    class _FakePopen:
        def __init__(self, *args, **kwargs) -> None:
            self.stdout = _FakeStdout()
            self.pid = 12345

        def poll(self) -> int | None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            return 0

    eval_calls: list[list[str]] = []

    def _fake_run_capture(cmd: list[str], *, cwd: Path, timeout_s: int = 0) -> subprocess.CompletedProcess[str]:
        if "pufferlib_market.evaluate" in cmd:
            eval_calls.append(cmd)
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout="Return: mean=0.10\nWin rate: mean=0.55\nSortino: mean=0.40\n>0: 10/10 (100.0%)\n",
                stderr="",
            )
        raise AssertionError(f"Unexpected command: {cmd}")

    time_values = iter([0.0, 30.0, 30.0, 31.0])

    def _fake_time() -> float:
        try:
            return next(time_values)
        except StopIteration:
            return 31.0

    monkeypatch.setattr("pufferlib_market.autoresearch_rl.subprocess.Popen", _FakePopen)
    monkeypatch.setattr("pufferlib_market.autoresearch_rl._run_capture", _fake_run_capture)
    monkeypatch.setattr("pufferlib_market.autoresearch_rl.time.time", _fake_time)
    monkeypatch.setattr("pufferlib_market.autoresearch_rl.os.getpgid", lambda pid: pid)
    monkeypatch.setattr("pufferlib_market.autoresearch_rl.os.killpg", lambda pgid, sig: None)

    result = run_trial(
        TrialConfig(description="baseline_floor_val"),
        "train.bin",
        "val.bin",
        100,
        str(checkpoint_dir),
        use_poly_prune=False,
        baseline_val_return_floor=0.50,
    )

    assert result["early_rejected"] is True
    assert result["prune_reference_val_return"] == pytest.approx(0.50)
    assert len(eval_calls) >= 2


def test_run_trial_uses_baseline_combined_floor_for_poly_prune(monkeypatch, tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "trial"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "best.pt").write_bytes(b"checkpoint")

    class _FakeStdout:
        def readline(self) -> bytes:
            return b""

        def read(self) -> bytes:
            return b""

    class _FakePopen:
        def __init__(self, *args, **kwargs) -> None:
            self.stdout = _FakeStdout()
            self.pid = 12345

        def poll(self) -> int | None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            return 0

    quick_results = iter([(0.10, 0.10), (0.20, 0.20)])

    def _fake_run_capture(cmd: list[str], *, cwd: Path, timeout_s: int = 0) -> subprocess.CompletedProcess[str]:
        if "pufferlib_market.evaluate" in cmd:
            num_episodes = cmd[cmd.index("--num-episodes") + 1]
            if num_episodes == "30":
                ret, sortino = next(quick_results)
            else:
                ret, sortino = (0.10, 0.10)
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=(
                    f"Return: mean={ret:.2f}\n"
                    "Win rate: mean=0.55\n"
                    f"Sortino: mean={sortino:.2f}\n"
                    ">0: 10/10 (100.0%)\n"
                ),
                stderr="",
            )
        raise AssertionError(f"Unexpected command: {cmd}")

    time_values = iter([0.0, 30.0, 30.0, 60.0, 60.0, 61.0])

    def _fake_time() -> float:
        try:
            return next(time_values)
        except StopIteration:
            return 61.0

    monkeypatch.setattr("pufferlib_market.autoresearch_rl.subprocess.Popen", _FakePopen)
    monkeypatch.setattr("pufferlib_market.autoresearch_rl._run_capture", _fake_run_capture)
    monkeypatch.setattr("pufferlib_market.autoresearch_rl.time.time", _fake_time)
    monkeypatch.setattr("pufferlib_market.autoresearch_rl.os.getpgid", lambda pid: pid)
    monkeypatch.setattr("pufferlib_market.autoresearch_rl.os.killpg", lambda pgid, sig: None)

    result = run_trial(
        TrialConfig(description="baseline_floor_combined"),
        "train.bin",
        "val.bin",
        100,
        str(checkpoint_dir),
        baseline_combined_floor=1.0,
    )

    assert result["early_rejected"] is True
    assert result["prune_reference_combined"] == pytest.approx(1.0)
    assert result["poly_projected_final"] == pytest.approx(0.4)


def test_run_trial_passes_no_tf32_flag(monkeypatch, tmp_path: Path) -> None:
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
                cmd, 0,
                stdout="Return: mean=0.10\nWin rate: mean=0.55\nSortino: mean=1.20\n>0: 10/10 (100.0%)\n",
                stderr="",
            )
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr("pufferlib_market.autoresearch_rl.subprocess.Popen", _FakePopen)
    monkeypatch.setattr("pufferlib_market.autoresearch_rl._run_capture", _fake_run_capture)

    run_trial(
        TrialConfig(description="fidelity", no_tf32=True, use_bf16=False, no_cuda_graph=True),
        "train.bin",
        "val.bin",
        1,
        str(checkpoint_dir),
    )

    assert train_commands
    train_cmd = train_commands[0]
    assert "--no-tf32" in train_cmd
    assert "--use-bf16" not in train_cmd
    assert "--no-cuda-graph" in train_cmd


def test_run_trial_passes_stability_guard_flags(monkeypatch, tmp_path: Path) -> None:
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
                cmd, 0,
                stdout="Return: mean=0.10\nWin rate: mean=0.55\nSortino: mean=1.20\n>0: 10/10 (100.0%)\n",
                stderr="",
            )
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr("pufferlib_market.autoresearch_rl.subprocess.Popen", _FakePopen)
    monkeypatch.setattr("pufferlib_market.autoresearch_rl._run_capture", _fake_run_capture)

    run_trial(
        TrialConfig(
            description="stable_guards",
            grad_norm_warn_threshold=20.0,
            grad_norm_skip_threshold=200.0,
            unstable_update_patience=6,
            lr_backoff_factor=0.25,
            min_lr=5e-6,
        ),
        "train.bin",
        "val.bin",
        1,
        str(checkpoint_dir),
    )

    assert train_commands
    train_cmd = train_commands[0]
    assert train_cmd[train_cmd.index("--grad-norm-warn-threshold") + 1] == "20.0"
    assert train_cmd[train_cmd.index("--grad-norm-skip-threshold") + 1] == "200.0"
    assert train_cmd[train_cmd.index("--unstable-update-patience") + 1] == "6"
    assert train_cmd[train_cmd.index("--lr-backoff-factor") + 1] == "0.25"
    assert train_cmd[train_cmd.index("--min-lr") + 1] == "5e-06"


def test_run_trial_reuses_same_wandb_run_for_post_eval_summary(monkeypatch, tmp_path: Path) -> None:
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
                stdout="Return: mean=0.10\nWin rate: mean=0.55\nSortino: mean=1.20\n>0: 10/10 (100.0%)\n",
                stderr="",
            )
        raise AssertionError(f"Unexpected command: {cmd}")

    class _FakeSummaryRun:
        def __init__(self) -> None:
            self.summary: dict[str, object] = {}
            self.finished = False

        def finish(self) -> None:
            self.finished = True

    class _FakeWandbModule:
        def __init__(self) -> None:
            self.init_calls: list[dict[str, object]] = []
            self.runs: list[_FakeSummaryRun] = []

        def init(self, **kwargs):
            self.init_calls.append(dict(kwargs))
            run = _FakeSummaryRun()
            self.runs.append(run)
            return run

    fake_wandb = _FakeWandbModule()

    monkeypatch.setattr("pufferlib_market.autoresearch_rl.subprocess.Popen", _FakePopen)
    monkeypatch.setattr("pufferlib_market.autoresearch_rl._run_capture", _fake_run_capture)
    monkeypatch.setattr("pufferlib_market.autoresearch_rl._wandb_module", fake_wandb)
    monkeypatch.setattr("pufferlib_market.autoresearch_rl._make_wandb_run_id", lambda: "shared-run-123")

    result = run_trial(
        TrialConfig(description="wandb_shared_summary"),
        "train.bin",
        "val.bin",
        1,
        str(checkpoint_dir),
        wandb_project="stock",
        wandb_group="autoresearch_group",
    )

    assert train_commands
    train_cmd = train_commands[0]
    assert "--wandb-run-id" in train_cmd
    assert train_cmd[train_cmd.index("--wandb-run-id") + 1] == "shared-run-123"
    assert "--wandb-resume" in train_cmd
    assert train_cmd[train_cmd.index("--wandb-resume") + 1] == "allow"

    assert len(fake_wandb.init_calls) == 1
    init_kwargs = fake_wandb.init_calls[0]
    assert init_kwargs["id"] == "shared-run-123"
    assert init_kwargs["resume"] == "allow"
    assert init_kwargs["project"] == "stock"
    assert init_kwargs["group"] == "autoresearch_group"
    assert fake_wandb.runs[0].summary["val/return"] == pytest.approx(0.10)
    assert fake_wandb.runs[0].summary["trial/rank_score"] == pytest.approx(result["rank_score"])
    assert fake_wandb.runs[0].finished is True


def test_read_mktd_header_parses_real_file() -> None:
    import os
    path = "pufferlib_market/data/mixed23_latest_train_20260320.bin"
    if not os.path.exists(path):
        pytest.skip("mixed23 train data not available")
    ns, nt = _read_mktd_header(path)
    assert ns == 23
    assert nt > 0
