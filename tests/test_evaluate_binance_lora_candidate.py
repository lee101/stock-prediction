from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluate_binance_lora_candidate import (
    build_eval_command,
    build_remote_cache_command,
    load_candidate_config,
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
