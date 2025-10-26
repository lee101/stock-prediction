from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from stockagent.agentsimulator.data_models import TradingPlan
from stockagent.agentsimulator.simulator import SimulationResult
from stockagent2.agentsimulator.runner import PipelineSimulationConfig, PipelineSimulationResult, RunnerConfig
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
    )


def test_pipeline_cli_defaults_paper(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    record: dict[str, object] = {}

    def fake_run_pipeline_simulation(*, runner_config, optimisation_config, pipeline_config, simulation_config):
        record["runner"] = runner_config
        record["optimisation"] = optimisation_config
        record["pipeline"] = pipeline_config
        record["simulation_config"] = simulation_config
        return _fake_result()

    monkeypatch.setattr("stockagent2.cli.run_pipeline_simulation", fake_run_pipeline_simulation)

    exit_code = cli_main(["pipeline-sim", "--symbols", "AAPL", "MSFT", "--summary-format", "json"])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert '"trading_mode": "paper"' in output

    runner = record["runner"]
    assert isinstance(runner, RunnerConfig)
    assert runner.symbols == ("AAPL", "MSFT")
    assert runner.allow_remote_data is False

    sim_cfg = record["simulation_config"]
    assert isinstance(sim_cfg, PipelineSimulationConfig)
    assert sim_cfg.symbols == ("AAPL", "MSFT")


def test_pipeline_cli_live_mode(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr("stockagent2.cli.run_pipeline_simulation", lambda **_: _fake_result())

    exit_code = cli_main(["pipeline-sim", "--live"])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Trading mode: live" in output


def test_pipeline_cli_outputs_written(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("stockagent2.cli.run_pipeline_simulation", lambda **_: _fake_result())

    summary_path = tmp_path / "summary.json"
    plans_path = tmp_path / "plans.json"
    trades_path = tmp_path / "trades.json"

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


def test_pipeline_cli_handles_no_plans(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr("stockagent2.cli.run_pipeline_simulation", lambda **_: None)

    exit_code = cli_main(["pipeline-sim"])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Pipeline simulation produced no trading plans" in captured.err
