from __future__ import annotations

import shutil
from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")

from marketsimulator.runner import simulate_strategy


@pytest.mark.integration
def test_simulation_runner_generates_report_and_graphs():
    output_dir = Path("testresults") / "pytest_run"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    report = simulate_strategy(
        symbols=["AAPL", "MSFT", "NVDA", "BTCUSD"],
        days=3,
        step_size=12,
        initial_cash=100_000.0,
        top_k=5,
        output_dir=output_dir,
    )

    summary_text = report.render_summary()
    assert "Simulation Summary" in summary_text
    assert report.daily_snapshots, "Expected snapshots to be recorded"
    assert len(report.daily_snapshots) == 6, "Expect open/close snapshots per day"
    assert report.trades_executed >= 0
    assert report.fees_paid >= 0

    assert output_dir.exists()
    pngs = list(output_dir.glob("*.png"))
    assert pngs, "Expected plot outputs in testresults/"
    day_pngs = sorted(output_dir.glob("day_*_equity.png"))
    assert len(day_pngs) == 3
    assert any("equity_curve" in p.name for p in pngs)
    assert any("symbol_contributions" in p.name for p in pngs)

    assert report.generated_files, "Report should track generated artifacts"
    assert set(report.generated_files) == set(pngs)

    prediction_files = list(Path("results").glob("predictions*.csv"))
    assert prediction_files, "Forecasting run should emit prediction CSVs"
