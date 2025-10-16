from __future__ import annotations

from datetime import datetime, timezone

from typer.testing import CliRunner

import stock_cli
from src.portfolio_risk import PortfolioSnapshotRecord
from stock.state_utils import ProbeStatus


def test_risk_text_cli(monkeypatch):
    runner = CliRunner()

    snapshots = [
        PortfolioSnapshotRecord(
            observed_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            portfolio_value=100_000.0,
            risk_threshold=0.5,
        ),
        PortfolioSnapshotRecord(
            observed_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
            portfolio_value=110_000.0,
            risk_threshold=0.6,
        ),
        PortfolioSnapshotRecord(
            observed_at=datetime(2025, 1, 3, tzinfo=timezone.utc),
            portfolio_value=120_000.0,
            risk_threshold=0.7,
        ),
    ]

    monkeypatch.setattr(stock_cli, "fetch_snapshots", lambda limit=None: snapshots)

    result = runner.invoke(stock_cli.app, ["risk-text", "--width", "5", "--limit", "3"])
    assert result.exit_code == 0
    assert "Portfolio Value (ASCII)" in result.stdout
    assert "Latest=$120,000.00" in result.stdout


def test_probe_status_cli(monkeypatch):
    runner = CliRunner()
    statuses = [
        ProbeStatus(
            symbol="AAPL",
            side="buy",
            pending_probe=False,
            probe_active=True,
            last_pnl=25.0,
            last_reason="take_profit",
            last_closed_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
            active_mode="probe",
            active_qty=1.5,
            active_opened_at=datetime(2025, 1, 3, tzinfo=timezone.utc),
            learning_updated_at=datetime(2025, 1, 4, tzinfo=timezone.utc),
        )
    ]

    monkeypatch.setattr(stock_cli, "collect_probe_statuses", lambda suffix=None: statuses)

    result = runner.invoke(stock_cli.app, ["probe-status", "--tz", "UTC"])
    assert result.exit_code == 0
    assert "AAPL" in result.stdout
    assert "take_profit" in result.stdout
