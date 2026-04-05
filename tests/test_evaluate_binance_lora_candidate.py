from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluate_binance_lora_candidate import (
    build_eval_command,
    build_remote_cache_command,
    build_rsync_remote_shell,
    build_ssh_command,
    load_candidate_config,
    parse_args,
    summarize_eval_windows,
)


def test_load_candidate_config_reads_symbol_and_model_id(tmp_path: Path) -> None:
    report_path = tmp_path / "candidate.json"
    report_path.write_text(
        json.dumps(
            {
                "symbol": "BNBUSDT",
                "output_dir": "chronos2_finetuned/example_run",
                "config": {
                    "context_length": 256,
                    "batch_size": 32,
                },
            }
        )
        + "\n"
    )

    candidate = load_candidate_config(report_path)

    assert candidate.symbol == "BNBUSDT"
    assert candidate.model_id == "chronos2_finetuned/example_run/finetuned-ckpt"
    assert candidate.context_length == 256
    assert candidate.batch_size == 32


def test_build_remote_cache_command_uses_explicit_candidate_runtime(tmp_path: Path) -> None:
    report_path = tmp_path / "candidate.json"
    report_path.write_text(
        json.dumps(
            {
                "symbol": "BNBUSDT",
                "output_dir": "chronos2_finetuned/example_run",
                "config": {
                    "context_length": 192,
                    "batch_size": 16,
                },
            }
        )
        + "\n"
    )
    candidate = load_candidate_config(report_path)

    command = build_remote_cache_command(
        remote_root=Path("/remote/repo"),
        remote_venv=".venv313",
        candidate=candidate,
        remote_cache_root="analysis/lora_candidate_eval/example",
        remote_data_root="trainingdatahourlybinance",
        horizons="1,24",
        lookback_hours=5000.0,
    )

    assert "--model-id chronos2_finetuned/example_run/finetuned-ckpt" in command
    assert "--context-hours 192" in command
    assert "--batch-size 16" in command
    assert "--forecast-cache-root analysis/lora_candidate_eval/example/forecast_cache" in command


def test_build_eval_command_passes_forecast_rule_tuning_args() -> None:
    cmd = build_eval_command(
        python_executable="python",
        baseline_symbols=["BTCUSD", "ethusd", "SolUsd"],
        add_symbol="bnbusd",
        end="2026-03-18",
        windows="120,60,30,7,1",
        signal_mode="forecast_rule",
        forecast_cache_root=Path("analysis/lora_candidate_eval/example/forecast_cache"),
        data_root="trainingdatahourly/crypto",
        model="gemini-2.5-flash",
        thinking="HIGH",
        rate_limit=0.2,
        forecast_rule_total_cost_bps=25.0,
        forecast_rule_min_reward_risk=1.5,
        add_symbol_forecast_rule_total_cost_bps=35.0,
        add_symbol_forecast_rule_min_reward_risk=6.0,
        add_symbol_max_pos=0.01,
    )

    assert cmd[:4] == ["python", "rl-trading-agent-binance/eval_new_symbol.py", "--symbols", "BTCUSD"]
    assert "--forecast-rule-total-cost-bps" in cmd
    assert "25.0" in cmd
    assert "--forecast-rule-min-reward-risk" in cmd
    assert "1.5" in cmd
    assert "--add-symbol-forecast-rule-total-cost-bps" in cmd
    assert "35.0" in cmd
    assert "--add-symbol-forecast-rule-min-reward-risk" in cmd
    assert "6.0" in cmd
    assert "--add-symbol-max-pos" in cmd
    assert "0.01" in cmd
    assert "--model" not in cmd
    assert "--thinking" not in cmd
    assert "--rate-limit" not in cmd


def test_build_eval_command_adds_gemini_runtime_args_only_for_gemini() -> None:
    cmd = build_eval_command(
        python_executable="python3",
        baseline_symbols=["BTCUSD", "ETHUSD", "SOLUSD"],
        add_symbol="BNBUSD",
        end="2026-03-18",
        windows="30,7,1",
        signal_mode="gemini",
        forecast_cache_root=Path("analysis/lora_candidate_eval/example/forecast_cache"),
        data_root="trainingdatahourly/crypto",
        model="gemini-3.1-flash-lite-preview",
        thinking="LOW",
        rate_limit=0.5,
        forecast_rule_total_cost_bps=20.0,
        forecast_rule_min_reward_risk=1.1,
        add_symbol_forecast_rule_total_cost_bps=None,
        add_symbol_forecast_rule_min_reward_risk=None,
        add_symbol_max_pos=None,
    )

    assert "--model" in cmd
    assert "gemini-3.1-flash-lite-preview" in cmd
    assert "--thinking" in cmd
    assert "LOW" in cmd
    assert "--rate-limit" in cmd
    assert "0.5" in cmd


def test_build_ssh_helpers_disable_muxing() -> None:
    ssh_cmd = build_ssh_command("administrator@93.127.141.100", "echo ok")
    rsync_shell = build_rsync_remote_shell()

    assert ssh_cmd[:7] == [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ControlMaster=no",
        "-o",
        "ControlPath=none",
    ]
    assert ssh_cmd[-2:] == ["administrator@93.127.141.100", "echo ok"]
    assert rsync_shell == "ssh -o StrictHostKeyChecking=no -o ControlMaster=no -o ControlPath=none"


def test_parse_args_exposes_model_default(tmp_path: Path) -> None:
    report_path = tmp_path / "candidate.json"
    report_path.write_text("{}\n")

    args = parse_args(["--report-path", str(report_path)])

    assert args.model == "gemini-3.1-flash-lite-preview"


def test_summarize_eval_windows_aggregates_deltas_and_verdicts() -> None:
    summary = summarize_eval_windows(
        {
            "windows": [
                {
                    "window": "30d",
                    "comparison": {
                        "return_delta": 2.0,
                        "sortino_delta": 0.5,
                        "max_dd_delta": -1.0,
                        "new_symbol_pnl": 12.0,
                        "verdict": "ACCEPT",
                    }
                },
                {
                    "window": "60d",
                    "comparison": {
                        "return_delta": -1.0,
                        "sortino_delta": -0.2,
                        "max_dd_delta": 0.5,
                        "new_symbol_pnl": 4.0,
                        "verdict": "REJECT",
                    }
                },
            ]
        }
    )

    assert summary["window_count"] == 2
    assert summary["accepted_window_count"] == 1
    assert summary["rejected_window_count"] == 1
    assert summary["all_windows_accept"] is False
    assert summary["mean_return_delta"] == 0.5
    assert summary["min_sortino_delta"] == -0.2
    assert summary["mean_new_symbol_pnl"] == 8.0
    assert summary["weighted_annualized_return_delta"] == pytest.approx(5.127279663740351)
    assert summary["weighted_annualized_new_symbol_pnl"] == pytest.approx(64.88888888888889)


def test_summarize_eval_windows_uses_window_dates_when_label_missing() -> None:
    summary = summarize_eval_windows(
        {
            "windows": [
                {
                    "start": "2026-01-01",
                    "end": "2026-01-31",
                    "comparison": {
                        "return_delta": 10.0,
                        "sortino_delta": 1.0,
                        "max_dd_delta": -2.0,
                        "new_symbol_pnl": 100.0,
                        "verdict": "ACCEPT",
                    },
                }
            ]
        }
    )

    assert summary["weighted_annualized_return_delta"] > 100.0
    assert summary["weighted_annualized_new_symbol_pnl"] == pytest.approx(1216.6666666666665)
