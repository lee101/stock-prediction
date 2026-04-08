from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO / "scripts" / "evaluate_binance_hybrid_prod.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prod_eval = _load_module("evaluate_binance_hybrid_prod", SCRIPT_PATH)


def test_parse_args_defaults_require_runtime_guards() -> None:
    args = prod_eval.parse_args([])

    assert args.skip_runtime_audit is False
    assert args.require_runtime_match is True
    assert args.require_runtime_health is True


def test_parse_args_can_disable_runtime_guards() -> None:
    args = prod_eval.parse_args(["--no-require-runtime-match", "--no-require-runtime-health"])

    assert args.require_runtime_match is False
    assert args.require_runtime_health is False


def _make_runtime_audit_result(**overrides):
    payload = {
        "launch_script": "/tmp/launch.sh",
        "trace_dir": "/tmp/trace",
        "window_start": "2026-04-09T00:00:00+00:00",
        "window_end": "2026-04-09T01:00:00+00:00",
        "launch_checkpoint": "/tmp/current.pt",
        "launch_symbols": ["BTCUSD", "ETHUSD", "SOLUSD"],
        "launch_leverage": 0.5,
        "snapshot_count": 2,
        "healthy_completed_count": 2,
        "gemini_call_skipped_count": 0,
        "degraded_status_count": 0,
        "degraded_allocation_source_count": 0,
        "override_allocation_source_count": 0,
        "checkpoint_missing_count": 0,
        "checkpoint_mismatch_count": 0,
        "symbols_mismatch_count": 0,
        "leverage_mismatch_count": 0,
        "status_counts": {"completed": 2},
        "allocation_source_counts": {},
        "unexpected_symbol_activity_counts": {},
        "unexpected_order_symbol_counts": {},
        "degraded_examples": [],
    }
    payload.update(overrides)
    return prod_eval.BinanceHybridRuntimeAuditResult(**payload)


def _write_launch_script(path: Path, checkpoint: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "exec /tmp/.venv/bin/python -u \\",
                "  rl_trading_agent_binance/trade_binance_live.py \\",
                "  --live \\",
                "  --model gemini-3.1-flash-lite-preview \\",
                "  --symbols BTCUSD ETHUSD SOLUSD \\",
                "  --execution-mode margin \\",
                "  --leverage 0.5 \\",
                "  --interval 3600 \\",
                "  --fallback-mode chronos2 \\",
                f"  --rl-checkpoint {checkpoint} \\",
                '  "$@"',
            ]
        )
    )

def test_parse_launch_script_extracts_live_config(tmp_path: Path) -> None:
    launch = tmp_path / "launch.sh"
    checkpoint = tmp_path / "checkpoints" / "best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text("placeholder")
    launch.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "exec /tmp/.venv/bin/python -u \\",
                "  rl_trading_agent_binance/trade_binance_live.py \\",
                "  --live \\",
                "  --model gemini-3.1-flash-lite-preview \\",
                "  --symbols BTCUSD ETHUSD SOLUSD \\",
                "  --execution-mode margin \\",
                "  --leverage 0.5 \\",
                "  --interval 3600 \\",
                "  --fallback-mode chronos2 \\",
                f"  --rl-checkpoint {checkpoint} \\",
                '  "$@"',
            ]
        )
    )

    config = prod_eval.parse_launch_script(launch)

    assert config.python_bin == "/tmp/.venv/bin/python"
    assert config.trade_script == "rl_trading_agent_binance/trade_binance_live.py"
    assert config.model == "gemini-3.1-flash-lite-preview"
    assert config.symbols == ["BTCUSD", "ETHUSD", "SOLUSD"]
    assert config.execution_mode == "margin"
    assert config.leverage == 0.5
    assert config.interval == 3600
    assert config.fallback_mode == "chronos2"
    assert config.rl_checkpoint == str(checkpoint.resolve())


def test_parse_launch_script_allows_missing_rl_checkpoint_when_requested(tmp_path: Path) -> None:
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "exec /tmp/.venv/bin/python -u \\",
                "  rl_trading_agent_binance/trade_binance_live.py \\",
                "  --live \\",
                "  --model gemini-3.1-flash-lite-preview \\",
                "  --symbols BTCUSD ETHUSD SOLUSD \\",
                "  --execution-mode margin \\",
                "  --leverage 0.5 \\",
                "  --interval 3600 \\",
                "  --fallback-mode chronos2 \\",
                '  "$@"',
            ]
        )
    )

    config = prod_eval.parse_launch_script(launch, require_rl_checkpoint=False)

    assert config.symbols == ["BTCUSD", "ETHUSD", "SOLUSD"]
    assert config.rl_checkpoint is None


def test_resolve_target_launch_config_applies_symbol_override(tmp_path: Path) -> None:
    launch = tmp_path / "launch.sh"
    checkpoint = tmp_path / "checkpoints" / "best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text("placeholder")
    launch.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "exec /tmp/.venv/bin/python -u \\",
                "  rl_trading_agent_binance/trade_binance_live.py \\",
                "  --live \\",
                "  --model gemini-3.1-flash-lite-preview \\",
                "  --symbols BTCUSD ETHUSD SOLUSD \\",
                "  --execution-mode margin \\",
                "  --leverage 0.5 \\",
                "  --interval 3600 \\",
                "  --fallback-mode chronos2 \\",
                f"  --rl-checkpoint {checkpoint} \\",
                '  "$@"',
            ]
        )
    )

    config = prod_eval.resolve_target_launch_config(launch, symbols_override="btcusd, ethusd")

    assert config.symbols == ["BTCUSD", "ETHUSD"]
    assert config.leverage == 0.5
    assert config.rl_checkpoint == str(checkpoint.resolve())


def test_resolve_target_launch_config_applies_leverage_override(tmp_path: Path) -> None:
    launch = tmp_path / "launch.sh"
    checkpoint = tmp_path / "checkpoints" / "best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text("placeholder")
    launch.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "exec /tmp/.venv/bin/python -u \\",
                "  rl_trading_agent_binance/trade_binance_live.py \\",
                "  --live \\",
                "  --model gemini-3.1-flash-lite-preview \\",
                "  --symbols BTCUSD ETHUSD SOLUSD \\",
                "  --execution-mode margin \\",
                "  --leverage 0.5 \\",
                "  --interval 3600 \\",
                "  --fallback-mode chronos2 \\",
                f"  --rl-checkpoint {checkpoint} \\",
                '  "$@"',
            ]
        )
    )

    config = prod_eval.resolve_target_launch_config(launch, leverage_override=2.0)

    assert config.symbols == ["BTCUSD", "ETHUSD", "SOLUSD"]
    assert config.leverage == 2.0
    assert config.rl_checkpoint == str(checkpoint.resolve())


def test_build_holdout_command_applies_live_constraints(tmp_path: Path) -> None:
    launch_config = prod_eval.BinanceHybridLaunchConfig(
        launch_script="/tmp/launch.sh",
        python_bin="/tmp/.venv/bin/python",
        trade_script="rl_trading_agent_binance/trade_binance_live.py",
        model="gemini-3.1-flash-lite-preview",
        symbols=["BTCUSD", "ETHUSD", "SOLUSD"],
        execution_mode="margin",
        leverage=0.5,
        interval=3600,
        fallback_mode="chronos2",
        rl_checkpoint="/tmp/checkpoints/best.pt",
    )
    output_path = tmp_path / "result.json"

    command = prod_eval.build_holdout_command(
        launch_config,
        data_path="/tmp/data.bin",
        checkpoint="/tmp/checkpoints/candidate.pt",
        eval_hours=30,
        n_windows=50,
        seed=42,
        fee_rate=0.001,
        slippage_bps=5.0,
        fill_buffer_bps=5.0,
        decision_lag=2,
        periods_per_year=365.0,
        device="cpu",
        output_path=output_path,
        disable_shorts=True,
    )

    assert command[:3] == [sys.executable, "-m", "pufferlib_market.evaluate_holdout"]
    assert "--checkpoint" in command
    assert "/tmp/checkpoints/candidate.pt" in command
    assert "--max-leverage" in command
    assert "0.5" in command
    assert "--disable-shorts" in command
    tradable_index = command.index("--tradable-symbols")
    assert command[tradable_index + 1] == "BTCUSD,ETHUSD,SOLUSD"
    assert str(output_path) in command


def test_build_replay_command_applies_live_constraints(tmp_path: Path) -> None:
    launch_config = prod_eval.BinanceHybridLaunchConfig(
        launch_script="/tmp/launch.sh",
        python_bin="/tmp/.venv/bin/python",
        trade_script="rl_trading_agent_binance/trade_binance_live.py",
        model="gemini-3.1-flash-lite-preview",
        symbols=["BTCUSD", "ETHUSD", "SOLUSD"],
        execution_mode="margin",
        leverage=0.5,
        interval=3600,
        fallback_mode="chronos2",
        rl_checkpoint="/tmp/checkpoints/best.pt",
    )
    output_path = tmp_path / "replay.json"

    command = prod_eval.build_replay_command(
        launch_config,
        data_path="/tmp/data.bin",
        checkpoint="/tmp/checkpoints/candidate.pt",
        max_steps=30,
        fee_rate=0.001,
        fill_buffer_bps=5.0,
        daily_periods_per_year=365.0,
        hourly_periods_per_year=8760.0,
        hourly_data_root="/tmp/hourly",
        start_date="2025-06-01",
        end_date="2026-02-05",
        output_path=output_path,
        disable_shorts=True,
        robust_start_states="flat,long:BTCUSD:0.5",
        device="cpu",
    )

    assert command[:3] == [sys.executable, "-m", "pufferlib_market.replay_eval"]
    assert "--max-leverage" in command
    assert "0.5" in command
    assert "--disable-shorts" in command
    assert "--cpu" in command
    tradable_index = command.index("--tradable-symbols")
    assert command[tradable_index + 1] == "BTCUSD,ETHUSD,SOLUSD"
    robust_index = command.index("--robust-start-states")
    assert command[robust_index + 1] == "flat,long:BTCUSD:0.5"
    assert str(output_path) in command


def test_load_replay_summary_extracts_hourly_metrics(tmp_path: Path) -> None:
    replay_path = tmp_path / "replay.json"
    replay_path.write_text(
        json.dumps(
            {
                "daily": {"total_return": 0.1, "sortino": 1.5},
                "hourly_replay": {"total_return": 0.2, "sortino": 2.5, "goodness_score": 3.5},
                "robust_start_summary": {"hourly_replay": {"worst_total_return": -0.05}},
            }
        )
    )

    summary = prod_eval._load_replay_summary(replay_path)

    assert summary.daily_total_return == 0.1
    assert summary.daily_sortino == 1.5
    assert summary.hourly_total_return == 0.2
    assert summary.hourly_sortino == 2.5
    assert summary.hourly_goodness_score == 3.5
    assert summary.robust_worst_hourly_return == -0.05


def test_run_eval_writes_manifest_with_replay_results(tmp_path: Path, monkeypatch) -> None:
    launch = tmp_path / "launch.sh"
    checkpoint = tmp_path / "checkpoints" / "best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text("placeholder")
    launch.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "exec /tmp/.venv/bin/python -u \\",
                "  rl_trading_agent_binance/trade_binance_live.py \\",
                "  --live \\",
                "  --model gemini-3.1-flash-lite-preview \\",
                "  --symbols BTCUSD ETHUSD SOLUSD \\",
                "  --execution-mode margin \\",
                "  --leverage 0.5 \\",
                "  --interval 3600 \\",
                "  --fallback-mode chronos2 \\",
                f"  --rl-checkpoint {checkpoint} \\",
                '  "$@"',
            ]
        )
    )

    calls: list[list[str]] = []

    def _fake_run(command: list[str], **kwargs):
        calls.append(command)
        if "pufferlib_market.evaluate_holdout" in command:
            out_path = Path(command[command.index("--out") + 1])
            out_path.write_text(
                json.dumps(
                    {
                        "checkpoint": command[command.index("--checkpoint") + 1],
                        "summary": {
                            "median_total_return": 0.12,
                            "median_sortino": 1.4,
                            "median_max_drawdown": 0.08,
                            "p10_total_return": -0.03,
                        },
                    }
                )
            )
        elif "pufferlib_market.replay_eval" in command:
            out_path = Path(command[command.index("--output-json") + 1])
            out_path.write_text(
                json.dumps(
                    {
                        "daily": {"total_return": 0.05, "sortino": 0.9},
                        "hourly_replay": {"total_return": 0.07, "sortino": 1.2, "goodness_score": 2.3},
                        "robust_start_summary": {"hourly_replay": {"worst_total_return": -0.02}},
                    }
                )
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(prod_eval.subprocess, "run", _fake_run)
    monkeypatch.setattr(prod_eval, "audit_binance_hybrid_runtime", lambda **kwargs: _make_runtime_audit_result())

    args = prod_eval.parse_args(
        [
            "--launch-script",
            str(launch),
            "--data-path",
            "/tmp/data.bin",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert prod_eval.run_eval(args) == 0

    manifest = json.loads((tmp_path / "out" / "prod_launch_eval_manifest.json").read_text())
    assert isinstance(manifest["generated_at"], str)
    assert len(manifest["evaluations"]) == 1
    assert manifest["launch_config"]["symbols"] == ["BTCUSD", "ETHUSD", "SOLUSD"]
    assert manifest["launch_config"]["leverage"] == 0.5
    assert manifest["eval_config"]["data_path"] == str(Path("/tmp/data.bin").resolve())
    assert manifest["eval_config"]["slippage_bps"] == 5.0
    assert manifest["eval_config"]["skip_replay_eval"] is False
    evaluation = manifest["evaluations"][0]
    assert evaluation["median_total_return"] == 0.12
    assert evaluation["replay"]["hourly_total_return"] == 0.07
    assert evaluation["replay"]["hourly_goodness_score"] == 2.3
    assert evaluation["replay"]["robust_worst_hourly_return"] == -0.02
    assert any("pufferlib_market.evaluate_holdout" in command for command in calls)
    assert any("pufferlib_market.replay_eval" in command for command in calls)


def test_run_eval_writes_runtime_audit_into_manifest(tmp_path: Path, monkeypatch) -> None:
    launch = tmp_path / "launch.sh"
    checkpoint = tmp_path / "checkpoints" / "best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text("placeholder")
    _write_launch_script(launch, checkpoint)


    def _fake_run(command: list[str], **kwargs):
        if "pufferlib_market.evaluate_holdout" in command:
            out_path = Path(command[command.index("--out") + 1])
            out_path.write_text(
                json.dumps(
                    {
                        "checkpoint": command[command.index("--checkpoint") + 1],
                        "summary": {
                            "median_total_return": 0.12,
                            "median_sortino": 1.4,
                            "median_max_drawdown": 0.08,
                            "p10_total_return": -0.03,
                        },
                    }
                )
            )
        elif "pufferlib_market.replay_eval" in command:
            out_path = Path(command[command.index("--output-json") + 1])
            out_path.write_text(
                json.dumps(
                    {
                        "daily": {"total_return": 0.05, "sortino": 0.9},
                        "hourly_replay": {"total_return": 0.07, "sortino": 1.2, "goodness_score": 2.3},
                        "robust_start_summary": {"hourly_replay": {"worst_total_return": -0.02}},
                    }
                )
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(prod_eval.subprocess, "run", _fake_run)
    monkeypatch.setattr(prod_eval, "audit_binance_hybrid_runtime", lambda **kwargs: _make_runtime_audit_result())

    args = prod_eval.parse_args(
        [
            "--launch-script",
            str(launch),
            "--data-path",
            "/tmp/data.bin",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert prod_eval.run_eval(args) == 0

    manifest = json.loads((tmp_path / "out" / "prod_launch_eval_manifest.json").read_text())
    assert manifest["current_runtime_audit"]["snapshot_count"] == 2
    assert manifest["current_runtime_audit"]["healthy_completed_count"] == 2
    assert manifest["current_runtime_audit_issues"] == []
    assert manifest["current_runtime_health_issues"] == []
    assert manifest["eval_config"]["skip_runtime_audit"] is False
    assert manifest["eval_config"]["require_runtime_match"] is True
    assert manifest["eval_config"]["require_runtime_health"] is True


def test_run_eval_require_runtime_health_blocks_on_degraded_runtime(tmp_path: Path, monkeypatch, capsys) -> None:
    launch = tmp_path / "launch.sh"
    checkpoint = tmp_path / "checkpoints" / "best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text("placeholder")
    _write_launch_script(launch, checkpoint)

    calls: list[list[str]] = []

    def _unexpected_run(command: list[str], **kwargs):
        calls.append(command)
        raise AssertionError("subprocess.run should not be called when runtime health fails")

    monkeypatch.setattr(prod_eval.subprocess, "run", _unexpected_run)
    monkeypatch.setattr(
        prod_eval,
        "audit_binance_hybrid_runtime",
        lambda **kwargs: _make_runtime_audit_result(degraded_status_count=1),
    )

    args = prod_eval.parse_args(
        [
            "--launch-script",
            str(launch),
            "--data-path",
            "/tmp/data.bin",
            "--output-dir",
            str(tmp_path / "out"),
            "--require-runtime-health",
        ]
    )

    assert prod_eval.run_eval(args) == 2
    assert calls == []
    captured = capsys.readouterr()
    assert "current live runtime is too degraded to trust as a production baseline" in captured.err
    assert "degraded status cycles" in captured.err


def test_run_eval_require_runtime_match_blocks_on_launch_drift(tmp_path: Path, monkeypatch, capsys) -> None:
    launch = tmp_path / "launch.sh"
    checkpoint = tmp_path / "checkpoints" / "best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text("placeholder")
    _write_launch_script(launch, checkpoint)


    calls: list[list[str]] = []

    def _unexpected_run(command: list[str], **kwargs):
        calls.append(command)
        raise AssertionError("subprocess.run should not be called when runtime match fails")

    monkeypatch.setattr(prod_eval.subprocess, "run", _unexpected_run)
    monkeypatch.setattr(
        prod_eval,
        "audit_binance_hybrid_runtime",
        lambda **kwargs: _make_runtime_audit_result(checkpoint_mismatch_count=1),
    )

    args = prod_eval.parse_args(
        [
            "--launch-script",
            str(launch),
            "--data-path",
            "/tmp/data.bin",
            "--output-dir",
            str(tmp_path / "out"),
            "--require-runtime-match",
        ]
    )

    assert prod_eval.run_eval(args) == 2
    assert calls == []
    captured = capsys.readouterr()
    assert "current live runtime does not match launch config" in captured.err
    assert "launch checkpoint" in captured.err




def test_run_eval_with_symbols_override_writes_target_launch_symbols(tmp_path: Path, monkeypatch) -> None:
    launch = tmp_path / "launch.sh"
    checkpoint = tmp_path / "checkpoints" / "best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text("placeholder")
    launch.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "exec /tmp/.venv/bin/python -u \\",
                "  rl_trading_agent_binance/trade_binance_live.py \\",
                "  --live \\",
                "  --model gemini-3.1-flash-lite-preview \\",
                "  --symbols BTCUSD ETHUSD SOLUSD \\",
                "  --execution-mode margin \\",
                "  --leverage 0.5 \\",
                "  --interval 3600 \\",
                "  --fallback-mode chronos2 \\",
                f"  --rl-checkpoint {checkpoint} \\",
                '  "$@"',
            ]
        )
    )

    calls: list[list[str]] = []

    def _fake_run(command: list[str], **kwargs):
        calls.append(command)
        if "pufferlib_market.evaluate_holdout" in command:
            out_path = Path(command[command.index("--out") + 1])
            out_path.write_text(
                json.dumps(
                    {
                        "checkpoint": command[command.index("--checkpoint") + 1],
                        "summary": {
                            "median_total_return": 0.12,
                            "median_sortino": 1.4,
                            "median_max_drawdown": 0.08,
                            "p10_total_return": -0.03,
                        },
                    }
                )
            )
        elif "pufferlib_market.replay_eval" in command:
            out_path = Path(command[command.index("--output-json") + 1])
            out_path.write_text(
                json.dumps(
                    {
                        "daily": {"total_return": 0.05, "sortino": 0.9},
                        "hourly_replay": {"total_return": 0.07, "sortino": 1.2, "goodness_score": 2.3},
                        "robust_start_summary": {"hourly_replay": {"worst_total_return": -0.02}},
                    }
                )
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(prod_eval.subprocess, "run", _fake_run)
    monkeypatch.setattr(prod_eval, "audit_binance_hybrid_runtime", lambda **kwargs: _make_runtime_audit_result())

    args = prod_eval.parse_args(
        [
            "--launch-script",
            str(launch),
            "--data-path",
            "/tmp/data.bin",
            "--output-dir",
            str(tmp_path / "out"),
            "--symbols",
            "BTCUSD,ETHUSD",
        ]
    )

    assert prod_eval.run_eval(args) == 0

    manifest = json.loads((tmp_path / "out" / "prod_launch_eval_manifest.json").read_text())
    assert manifest["launch_config"]["symbols"] == ["BTCUSD", "ETHUSD"]
    holdout_commands = [command for command in calls if "pufferlib_market.evaluate_holdout" in command]
    replay_commands = [command for command in calls if "pufferlib_market.replay_eval" in command]
    assert holdout_commands
    assert replay_commands
    holdout_tradable_index = holdout_commands[0].index("--tradable-symbols")
    replay_tradable_index = replay_commands[0].index("--tradable-symbols")
    assert holdout_commands[0][holdout_tradable_index + 1] == "BTCUSD,ETHUSD"
    assert replay_commands[0][replay_tradable_index + 1] == "BTCUSD,ETHUSD"


def test_run_eval_with_leverage_override_writes_target_launch_leverage(tmp_path: Path, monkeypatch) -> None:
    launch = tmp_path / "launch.sh"
    checkpoint = tmp_path / "checkpoints" / "best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text("placeholder")
    launch.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "exec /tmp/.venv/bin/python -u \\",
                "  rl_trading_agent_binance/trade_binance_live.py \\",
                "  --live \\",
                "  --model gemini-3.1-flash-lite-preview \\",
                "  --symbols BTCUSD ETHUSD SOLUSD \\",
                "  --execution-mode margin \\",
                "  --leverage 0.5 \\",
                "  --interval 3600 \\",
                "  --fallback-mode chronos2 \\",
                f"  --rl-checkpoint {checkpoint} \\",
                '  "$@"',
            ]
        )
    )

    calls: list[list[str]] = []

    def _fake_run(command: list[str], **kwargs):
        calls.append(command)
        if "pufferlib_market.evaluate_holdout" in command:
            out_path = Path(command[command.index("--out") + 1])
            out_path.write_text(
                json.dumps(
                    {
                        "checkpoint": command[command.index("--checkpoint") + 1],
                        "summary": {
                            "median_total_return": 0.12,
                            "median_sortino": 1.4,
                            "median_max_drawdown": 0.08,
                            "p10_total_return": -0.03,
                        },
                    }
                )
            )
        elif "pufferlib_market.replay_eval" in command:
            out_path = Path(command[command.index("--output-json") + 1])
            out_path.write_text(
                json.dumps(
                    {
                        "daily": {"total_return": 0.05, "sortino": 0.9},
                        "hourly_replay": {"total_return": 0.07, "sortino": 1.2, "goodness_score": 2.3},
                        "robust_start_summary": {"hourly_replay": {"worst_total_return": -0.02}},
                    }
                )
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(prod_eval.subprocess, "run", _fake_run)
    monkeypatch.setattr(prod_eval, "audit_binance_hybrid_runtime", lambda **kwargs: _make_runtime_audit_result())

    args = prod_eval.parse_args(
        [
            "--launch-script",
            str(launch),
            "--data-path",
            "/tmp/data.bin",
            "--output-dir",
            str(tmp_path / "out"),
            "--leverage",
            "2.0",
        ]
    )

    assert prod_eval.run_eval(args) == 0

    manifest = json.loads((tmp_path / "out" / "prod_launch_eval_manifest.json").read_text())
    assert manifest["launch_config"]["leverage"] == 2.0
    holdout_commands = [command for command in calls if "pufferlib_market.evaluate_holdout" in command]
    replay_commands = [command for command in calls if "pufferlib_market.replay_eval" in command]
    assert holdout_commands
    assert replay_commands
    assert holdout_commands[0][holdout_commands[0].index("--max-leverage") + 1] == "2.0"
    assert replay_commands[0][replay_commands[0].index("--max-leverage") + 1] == "2.0"


def test_build_eval_plans_skips_replay_when_requested(tmp_path: Path) -> None:
    launch_config = prod_eval.BinanceHybridLaunchConfig(
        launch_script="/tmp/launch.sh",
        python_bin="/tmp/.venv/bin/python",
        trade_script="rl_trading_agent_binance/trade_binance_live.py",
        model="gemini-3.1-flash-lite-preview",
        symbols=["BTCUSD", "ETHUSD", "SOLUSD"],
        execution_mode="margin",
        leverage=0.5,
        interval=3600,
        fallback_mode="chronos2",
        rl_checkpoint="/tmp/checkpoints/best.pt",
    )
    args = prod_eval.parse_args(
        [
            "--data-path",
            "/tmp/data.bin",
            "--skip-replay-eval",
            "--candidate-checkpoint",
            "/tmp/checkpoints/candidate.pt",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    plans = prod_eval.build_eval_plans(launch_config, args, output_dir=tmp_path / "out")

    assert len(plans) == 2
    assert all(plan.replay_command is None for plan in plans)
    assert all(plan.replay_output_path is None for plan in plans)
    assert all("pufferlib_market.evaluate_holdout" in plan.holdout_command for plan in plans)


def test_build_eval_plans_with_gemini_only_launch_evaluates_candidates_only(tmp_path: Path) -> None:
    launch_config = prod_eval.BinanceHybridLaunchConfig(
        launch_script="/tmp/launch.sh",
        python_bin="/tmp/.venv/bin/python",
        trade_script="rl_trading_agent_binance/trade_binance_live.py",
        model="gemini-3.1-flash-lite-preview",
        symbols=["BTCUSD", "ETHUSD", "SOLUSD"],
        execution_mode="margin",
        leverage=0.5,
        interval=3600,
        fallback_mode="chronos2",
        rl_checkpoint=None,
    )
    args = prod_eval.parse_args(
        [
            "--data-path",
            "/tmp/data.bin",
            "--candidate-checkpoint",
            "/tmp/checkpoints/candidate_a.pt",
            "--candidate-checkpoint",
            "/tmp/checkpoints/candidate_b.pt",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    plans = prod_eval.build_eval_plans(launch_config, args, output_dir=tmp_path / "out")

    assert [plan.checkpoint for plan in plans] == [
        "/tmp/checkpoints/candidate_a.pt",
        "/tmp/checkpoints/candidate_b.pt",
    ]


def test_run_eval_returns_error_when_gemini_only_launch_has_no_candidates(tmp_path: Path) -> None:
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "exec /tmp/.venv/bin/python -u \\",
                "  rl_trading_agent_binance/trade_binance_live.py \\",
                "  --live \\",
                "  --model gemini-3.1-flash-lite-preview \\",
                "  --symbols BTCUSD ETHUSD SOLUSD \\",
                "  --execution-mode margin \\",
                "  --leverage 0.5 \\",
                "  --interval 3600 \\",
                "  --fallback-mode chronos2 \\",
                '  "$@"',
            ]
        )
    )
    args = prod_eval.parse_args(
        [
            "--launch-script",
            str(launch),
            "--data-path",
            "/tmp/data.bin",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert prod_eval.run_eval(args) == 2


def test_run_eval_skip_replay_writes_manifest_without_replay_results(tmp_path: Path, monkeypatch) -> None:
    launch = tmp_path / "launch.sh"
    checkpoint = tmp_path / "checkpoints" / "best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text("placeholder")
    launch.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "exec /tmp/.venv/bin/python -u \\",
                "  rl_trading_agent_binance/trade_binance_live.py \\",
                "  --live \\",
                "  --model gemini-3.1-flash-lite-preview \\",
                "  --symbols BTCUSD ETHUSD SOLUSD \\",
                "  --execution-mode margin \\",
                "  --leverage 0.5 \\",
                "  --interval 3600 \\",
                "  --fallback-mode chronos2 \\",
                f"  --rl-checkpoint {checkpoint} \\",
                '  "$@"',
            ]
        )
    )

    calls: list[list[str]] = []

    def _fake_run(command: list[str], **kwargs):
        calls.append(command)
        out_path = Path(command[command.index("--out") + 1])
        out_path.write_text(
            json.dumps(
                {
                    "checkpoint": command[command.index("--checkpoint") + 1],
                    "summary": {
                        "median_total_return": 0.04,
                        "median_sortino": 0.6,
                        "median_max_drawdown": 0.03,
                        "p10_total_return": -0.01,
                    },
                }
            )
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(prod_eval.subprocess, "run", _fake_run)
    monkeypatch.setattr(prod_eval, "audit_binance_hybrid_runtime", lambda **kwargs: _make_runtime_audit_result())

    args = prod_eval.parse_args(
        [
            "--launch-script",
            str(launch),
            "--data-path",
            "/tmp/data.bin",
            "--output-dir",
            str(tmp_path / "out"),
            "--skip-replay-eval",
        ]
    )

    assert prod_eval.run_eval(args) == 0

    manifest = json.loads((tmp_path / "out" / "prod_launch_eval_manifest.json").read_text())
    assert manifest["eval_config"]["skip_replay_eval"] is True
    assert manifest["evaluations"][0]["replay"] is None
    assert len(calls) == 1
    assert "pufferlib_market.evaluate_holdout" in calls[0]


def test_run_eval_dry_run_prints_plan_even_when_runtime_guard_fails(tmp_path: Path, monkeypatch, capsys) -> None:
    launch = tmp_path / "launch.sh"
    checkpoint = tmp_path / "checkpoints" / "best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text("placeholder")
    _write_launch_script(launch, checkpoint)

    calls: list[list[str]] = []

    def _unexpected_run(command: list[str], **kwargs):
        calls.append(command)
        raise AssertionError("subprocess.run should not be called during dry-run")

    monkeypatch.setattr(prod_eval.subprocess, "run", _unexpected_run)
    monkeypatch.setattr(
        prod_eval,
        "audit_binance_hybrid_runtime",
        lambda **kwargs: _make_runtime_audit_result(checkpoint_mismatch_count=1),
    )

    args = prod_eval.parse_args(
        [
            "--launch-script",
            str(launch),
            "--data-path",
            "/tmp/data.bin",
            "--output-dir",
            str(tmp_path / "out"),
            "--dry-run",
        ]
    )

    assert prod_eval.run_eval(args) == 0
    assert calls == []
    captured = capsys.readouterr()
    assert "current live runtime does not match launch config" in captured.err
    assert "DRY RUN -- no evaluations executed" in captured.out


def test_dry_run_prints_real_launch_command() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--dry-run"],
        capture_output=True,
        check=False,
        text=True,
        cwd=str(REPO),
    )

    assert result.returncode == 0, result.stderr
    combined = result.stdout + result.stderr
    assert "deployments/binance-hybrid-spot/launch.sh" in combined
    assert "--tradable-symbols BTCUSD,ETHUSD,SOLUSD,DOGEUSD,AAVEUSD,LINKUSD" in combined
    assert "--disable-shorts" in combined
    assert "robust_reg_tp005_dd002" in combined
    assert "pufferlib_market.replay_eval" in combined
    assert "--hourly-data-root trainingdatahourly" in combined
    assert "DRY RUN" in combined
