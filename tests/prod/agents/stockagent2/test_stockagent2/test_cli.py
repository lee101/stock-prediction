from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
import pytest

from stockagent.agentsimulator.data_models import TradingPlan
from stockagent.agentsimulator.simulator import SimulationResult
import stockagent2.cli as cli_module
from stockagent2.agentsimulator.runner import (
    PipelineMarketDataSummary,
    PipelineSimulationAttempt,
    PipelineSimulationConfig,
    PipelineSimulationResult,
    RunnerConfig,
)
from stockagent2.cli import main as cli_main


class _DummySimulator:
    def __init__(self) -> None:
        self.trade_log = [object(), object()]
        self.total_fees = 12.34
        self.equity_curve = [{"date": "2025-10-17", "equity": 101_250.0}]


def _fake_result() -> PipelineSimulationResult:
    simulation = SimulationResult(
        starting_cash=100_000.0,
        ending_cash=99_500.0,
        ending_equity=101_250.0,
        realized_pnl=900.0,
        unrealized_pnl=1_350.0,
        equity_curve=[{"date": "2025-10-17", "equity": 101_250.0}],
        trades=[{"symbol": "AAPL", "quantity": 10}],
        final_positions={"AAPL": {"quantity": 10, "avg_price": 100.0}},
        total_fees=12.34,
    )
    plan = TradingPlan(target_date=date(2025, 10, 17))
    return PipelineSimulationResult(
        simulator=_DummySimulator(),
        simulation=simulation,
        plans=(plan,),
        allocations=(),
        market_data_summary=PipelineMarketDataSummary(
            symbols_requested=("AAPL", "MSFT"),
            loaded_symbols=("AAPL",),
            empty_symbols=("MSFT",),
            bars_per_symbol={"AAPL": 3, "MSFT": 0},
            latest_bar_dates={"AAPL": "2025-10-17"},
            trading_day_count=3,
            first_trading_day="2025-10-15",
            last_trading_day="2025-10-17",
        ),
    )


def _fake_attempt(
    *,
    result: PipelineSimulationResult | None = None,
    failure_reason: str | None = None,
) -> PipelineSimulationAttempt:
    market_data_summary = (
        result.market_data_summary
        if result is not None
        else PipelineMarketDataSummary(
            symbols_requested=("AAPL", "MSFT"),
            loaded_symbols=("AAPL",),
            empty_symbols=("MSFT",),
            bars_per_symbol={"AAPL": 3, "MSFT": 0},
            latest_bar_dates={"AAPL": "2025-10-17"},
            trading_day_count=3,
            first_trading_day="2025-10-15",
            last_trading_day="2025-10-17",
        )
    )
    return PipelineSimulationAttempt(
        result=result,
        market_data_summary=market_data_summary,
        build_diagnostics=(),
        failure_reason=failure_reason,
    )


def test_pipeline_cli_defaults_paper(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    record: dict[str, object] = {}

    def fake_run_pipeline_simulation(*, runner_config, optimisation_config, pipeline_config, simulation_config):
        record["runner"] = runner_config
        record["optimisation"] = optimisation_config
        record["pipeline"] = pipeline_config
        record["simulation_config"] = simulation_config
        return _fake_attempt(result=_fake_result())

    monkeypatch.setattr("stockagent2.cli.run_pipeline_simulation_with_diagnostics", fake_run_pipeline_simulation)

    exit_code = cli_main(["pipeline-sim", "--symbols", "AAPL", "MSFT", "--summary-format", "json"])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert '"trading_mode": "paper"' in output

    runner = record["runner"]
    assert isinstance(runner, RunnerConfig)
    assert runner.symbols == ("AAPL", "MSFT")
    assert runner.allow_remote_data is False
    assert runner.use_fallback_data_dirs is True

    sim_cfg = record["simulation_config"]
    assert isinstance(sim_cfg, PipelineSimulationConfig)
    assert sim_cfg.symbols == ("AAPL", "MSFT")


def test_pipeline_cli_live_mode(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr("stockagent2.cli.run_pipeline_simulation_with_diagnostics", lambda **_: _fake_attempt(result=_fake_result()))

    exit_code = cli_main(["pipeline-sim", "--live"])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Trading mode: live" in output
    assert "Data coverage: 2025-10-15 to 2025-10-17 across 1/2 loaded symbols" in output
    assert "Empty symbols: MSFT" in output


def test_pipeline_cli_outputs_written(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("stockagent2.cli.run_pipeline_simulation_with_diagnostics", lambda **_: _fake_attempt(result=_fake_result()))

    summary_path = tmp_path / "summary.json"
    plans_path = tmp_path / "plans.json"
    trades_path = tmp_path / "trades.json"
    effective_args_path = tmp_path / "summary.effective_args.json"
    effective_args_txt_path = tmp_path / "summary.effective_args.txt"

    exit_code = cli_main(
        [
            "pipeline-sim",
            "--summary-format",
            "json",
            "--summary-output",
            summary_path.as_posix(),
            "--plans-output",
            plans_path.as_posix(),
            "--trades-output",
            trades_path.as_posix(),
            "--quiet",
        ]
    )
    assert exit_code == 0
    assert summary_path.exists()
    assert plans_path.exists()
    assert trades_path.exists()
    assert effective_args_path.exists()
    assert effective_args_txt_path.exists()
    report = summary_path.read_text(encoding="utf-8")
    assert '"effective_args_path"' in report
    assert '"market_data"' in report
    assert f'"plans_output_path": "{plans_path.as_posix()}"' in report
    assert f'"trades_output_path": "{trades_path.as_posix()}"' in report


def test_pipeline_cli_summary_output_auto_writes_plan_and_trade_sidecars(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("stockagent2.cli.run_pipeline_simulation_with_diagnostics", lambda **_: _fake_attempt(result=_fake_result()))

    summary_path = tmp_path / "summary.json"
    plans_path = tmp_path / "summary.plans.json"
    trades_path = tmp_path / "summary.trades.json"

    exit_code = cli_main(
        [
            "pipeline-sim",
            "--summary-format",
            "json",
            "--summary-output",
            summary_path.as_posix(),
            "--quiet",
        ]
    )

    assert exit_code == 0
    assert summary_path.exists()
    assert plans_path.exists()
    assert trades_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["plans_output_path"] == plans_path.as_posix()
    assert summary["trades_output_path"] == trades_path.as_posix()

    plans_payload = json.loads(plans_path.read_text(encoding="utf-8"))
    trades_payload = json.loads(trades_path.read_text(encoding="utf-8"))
    assert plans_payload and plans_payload[0]["target_date"] == "2025-10-17"
    assert trades_payload and trades_payload[0]["symbol"] == "AAPL"


def test_pipeline_cli_output_writer_is_atomic(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"

    cli_module._write_output(summary_path, '{"ok": true}\n')

    assert summary_path.read_text(encoding="utf-8") == '{"ok": true}\n'
    assert list(tmp_path.glob(".summary.json.tmp.*")) == []


def test_pipeline_cli_output_writer_cleans_temp_file_on_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary_path = tmp_path / "summary.json"
    captured: dict[str, Path] = {}

    def _raise_replace(src: str | bytes | os.PathLike[str] | os.PathLike[bytes], dst: str | bytes | os.PathLike[str] | os.PathLike[bytes]) -> None:
        captured["src"] = Path(src)
        captured["dst"] = Path(dst)
        raise OSError("disk full")

    monkeypatch.setattr(cli_module.os, "replace", _raise_replace)

    with pytest.raises(OSError, match="disk full"):
        cli_module._write_output(summary_path, '{"ok": true}\n')

    assert captured["dst"] == summary_path
    assert not summary_path.exists()
    assert not captured["src"].exists()
    assert list(tmp_path.glob(".summary.json.tmp.*")) == []


def test_pipeline_cli_handles_no_plans(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(
        "stockagent2.cli.run_pipeline_simulation_with_diagnostics",
        lambda **_: _fake_attempt(result=None, failure_reason="Pipeline simulation produced no trading plans."),
    )

    exit_code = cli_main(["pipeline-sim"])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Pipeline simulation produced no trading plans" in captured.err
    assert "Next steps:" in captured.err


def test_pipeline_cli_no_plans_writes_summary_and_effective_args(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "stockagent2.cli.run_pipeline_simulation_with_diagnostics",
        lambda **_: _fake_attempt(result=None, failure_reason="Pipeline simulation produced no trading plans."),
    )
    summary_path = tmp_path / "no-plans.json"

    exit_code = cli_main(
        [
            "pipeline-sim",
            "--summary-format",
            "json",
            "--summary-output",
            summary_path.as_posix(),
            "--quiet",
        ]
    )

    assert exit_code == 1
    assert capsys.readouterr().err == ""
    assert summary_path.exists()
    assert (tmp_path / "no-plans.effective_args.json").exists()
    assert (tmp_path / "no-plans.effective_args.txt").exists()

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["mode"] == "no-plans"
    assert payload["failure_reason"] == "Pipeline simulation produced no trading plans."
    assert payload["next_steps"]
    assert payload["market_data"]["empty_symbols"] == ["MSFT"]
    assert payload["effective_args_path"].endswith(".effective_args.json")
    assert payload["effective_args_cli_path"].endswith(".effective_args.txt")


def test_pipeline_cli_describe_run_skips_simulation(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(
        "stockagent2.cli.run_pipeline_simulation_with_diagnostics",
        lambda **_: (_ for _ in ()).throw(AssertionError("simulation should not run in describe mode")),
    )

    exit_code = cli_main(["pipeline-sim", "--describe-run", "--summary-format", "json", "--symbols", "AAPL", "MSFT"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert '"mode": "describe-run"' in output
    assert '"trading_mode": "paper"' in output
    assert '"symbols": [' in output


def test_pipeline_cli_supports_args_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(
        "stockagent2.cli.run_pipeline_simulation_with_diagnostics",
        lambda **_: (_ for _ in ()).throw(AssertionError("simulation should not run in describe mode")),
    )
    args_file = tmp_path / "pipeline.args"
    args_file.write_text(
        "\n".join(
            [
                "# stockagent2 pipeline preview",
                "pipeline-sim",
                "--describe-run",
                "--summary-format json",
                "--symbols AAPL MSFT",
                "--simulation-days 5",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = cli_main([f"@{args_file}"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert '"mode": "describe-run"' in output
    assert '"simulation_days": 5' in output


def test_pipeline_cli_describe_run_text_output_written(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "stockagent2.cli.run_pipeline_simulation_with_diagnostics",
        lambda **_: (_ for _ in ()).throw(AssertionError("simulation should not run in describe mode")),
    )
    summary_path = tmp_path / "describe.txt"

    exit_code = cli_main(
        [
            "pipeline-sim",
            "--describe-run",
            "--symbols",
            "AAPL",
            "MSFT",
            "--summary-output",
            summary_path.as_posix(),
            "--quiet",
        ]
    )

    assert exit_code == 0
    assert summary_path.exists()
    text = summary_path.read_text(encoding="utf-8")
    assert "Mode: describe-run" in text
    assert "Runner symbols: AAPL, MSFT" in text
    assert "Fallback data dirs: enabled" in text
    assert "History divisor: 4" in text
    assert "Secondary sample scale: 1.35" in text
    assert "Sample return clip: 0.25" in text
    assert "Data search order:" in text
    assert "Effective args file:" in text
    assert (tmp_path / "describe.effective_args.json").exists()
    assert (tmp_path / "describe.effective_args.txt").exists()


def test_pipeline_cli_runner_config_parses_false_bool_string(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    record: dict[str, object] = {}

    def fake_run_pipeline_simulation(*, runner_config, optimisation_config, pipeline_config, simulation_config):
        record["runner"] = runner_config
        return _fake_attempt(result=_fake_result())

    monkeypatch.setattr("stockagent2.cli.run_pipeline_simulation_with_diagnostics", fake_run_pipeline_simulation)
    runner_config = tmp_path / "runner.json"
    runner_config.write_text(
        '{"symbols": ["aapl", "msft"], "allow_remote_data": "false", "use_fallback_data_dirs": "false", "simulation_days": 7}',
        encoding="utf-8",
    )

    exit_code = cli_main(["pipeline-sim", "--runner-config", runner_config.as_posix(), "--summary-format", "json"])

    assert exit_code == 0
    assert '"trading_mode": "paper"' in capsys.readouterr().out
    runner = record["runner"]
    assert isinstance(runner, RunnerConfig)
    assert runner.allow_remote_data is False
    assert runner.use_fallback_data_dirs is False
    assert runner.symbols == ("AAPL", "MSFT")
    assert runner.simulation_days == 7


def test_pipeline_cli_describe_run_can_disable_fallback_dirs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "stockagent2.cli.run_pipeline_simulation_with_diagnostics",
        lambda **_: (_ for _ in ()).throw(AssertionError("simulation should not run in describe mode")),
    )
    local_data_dir = tmp_path / "cache"

    exit_code = cli_main(
        [
            "pipeline-sim",
            "--describe-run",
            "--summary-format",
            "json",
            "--local-data-dir",
            local_data_dir.as_posix(),
            "--no-fallback-data-dirs",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["runner"]["use_fallback_data_dirs"] is False
    assert payload["data_setup"]["use_fallback_data_dirs"] is False
    assert payload["data_setup"]["data_search_dirs"] == [local_data_dir.as_posix()]


def test_pipeline_cli_uses_environment_backed_runner_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    record: dict[str, object] = {}

    def fake_run_pipeline_simulation(*, runner_config, optimisation_config, pipeline_config, simulation_config):
        record["runner"] = runner_config
        return _fake_attempt(result=_fake_result())

    custom_dir = tmp_path / "env-cache"
    monkeypatch.setenv("STOCKAGENT_LOCAL_DATA_DIR", custom_dir.as_posix())
    monkeypatch.setenv("STOCKAGENT_USE_FALLBACK_DATA_DIRS", "false")
    monkeypatch.setattr("stockagent2.cli.run_pipeline_simulation_with_diagnostics", fake_run_pipeline_simulation)

    exit_code = cli_main(["pipeline-sim", "--summary-format", "json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    runner = record["runner"]
    assert isinstance(runner, RunnerConfig)
    assert runner.local_data_dir == custom_dir
    assert runner.use_fallback_data_dirs is False
    assert payload["runner"]["local_data_dir"] == custom_dir.as_posix()
    assert payload["runner"]["use_fallback_data_dirs"] is False
    assert payload["data_setup"]["data_search_dirs"] == [custom_dir.as_posix()]


def test_pipeline_cli_pipeline_config_rejects_invalid_bool_string(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pipeline_config = tmp_path / "pipeline.json"
    pipeline_config.write_text('{"apply_confidence_to_mu": "definitely"}', encoding="utf-8")

    exit_code = cli_main(["pipeline-sim", "--pipeline-config", pipeline_config.as_posix()])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "apply_confidence_to_mu must be a boolean-compatible value" in captured.err


def test_pipeline_cli_simulation_config_parses_heuristic_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    record: dict[str, object] = {}

    def fake_run_pipeline_simulation(*, runner_config, optimisation_config, pipeline_config, simulation_config):
        record["simulation_config"] = simulation_config
        return _fake_attempt(result=_fake_result())

    monkeypatch.setattr("stockagent2.cli.run_pipeline_simulation_with_diagnostics", fake_run_pipeline_simulation)
    simulation_config = tmp_path / "simulation.json"
    simulation_config.write_text(
        json.dumps(
            {
                "sample_count": 32,
                "history_min_period_divisor": 2,
                "secondary_sample_scale": 1.75,
                "sample_return_clip": 0.4,
                "min_view_half_life_days": 7,
                "max_view_half_life_days": 9,
                "rng_seed": 99,
            }
        ),
        encoding="utf-8",
    )

    exit_code = cli_main(["pipeline-sim", "--simulation-config", simulation_config.as_posix(), "--quiet"])

    assert exit_code == 0
    sim_cfg = record["simulation_config"]
    assert isinstance(sim_cfg, PipelineSimulationConfig)
    assert sim_cfg.sample_count == 32
    assert sim_cfg.history_min_period_divisor == 2
    assert sim_cfg.secondary_sample_scale == pytest.approx(1.75)
    assert sim_cfg.sample_return_clip == pytest.approx(0.4)
    assert sim_cfg.min_view_half_life_days == 7
    assert sim_cfg.max_view_half_life_days == 9
    assert sim_cfg.rng_seed == 99


def test_pipeline_cli_rejects_unsafe_symbol(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli_main(["pipeline-sim", "--describe-run", "--symbols", "../secrets"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Unsupported symbol" in captured.err


def test_pipeline_cli_keeps_summary_when_effective_args_write_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("stockagent2.cli.run_pipeline_simulation_with_diagnostics", lambda **_: _fake_attempt(result=_fake_result()))
    summary_path = tmp_path / "summary.json"
    real_write_json = __import__("stockagent2.cli", fromlist=["_write_json_output"])._write_json_output

    def _boom(path: Path, payload) -> None:
        if path.name.endswith(".effective_args.json"):
            raise OSError("disk full")
        real_write_json(path, payload)

    monkeypatch.setattr("stockagent2.cli._write_json_output", _boom)

    exit_code = cli_main(
        [
            "pipeline-sim",
            "--summary-format",
            "json",
            "--summary-output",
            summary_path.as_posix(),
            "--quiet",
        ]
    )

    assert exit_code == 0
    report = summary_path.read_text(encoding="utf-8")
    assert '"effective_args_warning"' in report
    assert "disk full" in report
