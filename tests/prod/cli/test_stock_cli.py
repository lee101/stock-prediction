from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

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


def test_format_strategy_profit_summary_highlight_selected():
    forecast = {
        "entry_takeprofit_profit": 0.051234,
        "maxdiffprofit_profit": 0.102345,
        "takeprofit_profit": -0.023456,
    }
    summary = stock_cli._format_strategy_profit_summary("maxdiff", forecast)
    assert summary == "profits entry=0.0512 maxdiff=0.1023* takeprofit=-0.0235"


def test_format_strategy_profit_summary_handles_missing():
    summary = stock_cli._format_strategy_profit_summary("simple", {})
    assert summary is None


def test_status_cli_live_portfolio_value(monkeypatch):
    runner = CliRunner()

    account = SimpleNamespace(
        equity="97659.92",
        last_equity="97448.9631540191",
        cash="1080.31",
        buying_power="11176.86",
        multiplier="2",
        status="ACTIVE",
    )

    positions = [
        SimpleNamespace(
            symbol="AAPL",
            side="long",
            qty="12",
            market_value="3101.4",
            unrealized_pl="96.36",
            current_price="258.45",
            last_trade_at=None,
        )
    ]

    snapshot = PortfolioSnapshotRecord(
        observed_at=datetime(2025, 10, 21, 20, 58, 17, tzinfo=timezone.utc),
        portfolio_value=0.0,
        risk_threshold=1.5,
    )

    monkeypatch.setattr(stock_cli, "get_leverage_settings", lambda: SimpleNamespace(max_gross_leverage=1.5))
    monkeypatch.setattr(stock_cli, "get_global_risk_threshold", lambda: 1.5)
    monkeypatch.setattr(stock_cli, "get_configured_max_risk_threshold", lambda: 1.5)
    monkeypatch.setattr(stock_cli, "fetch_latest_snapshot", lambda: snapshot)
    monkeypatch.setattr(stock_cli.alpaca_wrapper, "get_account", lambda: account)
    monkeypatch.setattr(stock_cli.alpaca_wrapper, "get_all_positions", lambda: positions)
    monkeypatch.setattr(stock_cli, "filter_to_realistic_positions", lambda items: list(items))
    monkeypatch.setattr(stock_cli.alpaca_wrapper, "get_orders", lambda: [])
    monkeypatch.setattr(stock_cli, "_load_active_trading_plan", lambda: [])
    monkeypatch.setattr(stock_cli, "_fetch_forecast_snapshot", lambda: ({}, None))
    monkeypatch.setattr(stock_cli, "_load_maxdiff_watchers", lambda: [])

    result = runner.invoke(stock_cli.app, ["status", "--tz", "US/Eastern"])
    assert result.exit_code == 0
    assert "Live Portfolio Value=$97,659.92" in result.stdout
    assert "Last Recorded Portfolio Value=$0.00" in result.stdout
