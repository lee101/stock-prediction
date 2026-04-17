from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO / "scripts" / "preview_daily_stock_open_plan.py"


def _load_module():
    spec = spec_from_file_location("preview_daily_stock_open_plan", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_preview_result_flat_signal_reports_no_order() -> None:
    mod = _load_module()
    payload = {
        "action": "flat",
        "symbol": None,
        "direction": None,
        "confidence": 0.0546,
        "value_estimate": -1.2208,
        "bars_fresh": True,
        "latest_bar_timestamp": "2026-04-14T04:00:00+00:00",
        "market_open": True,
        "bar_data_source": "alpaca",
        "quote_data_source": "alpaca",
        "execution_status": "no_action_flat_signal",
        "execution_skip_reason": "flat_signal",
        "execution_would_submit": False,
        "server_snapshot": {
            "cash": 28679.04,
            "equity": 28679.04,
            "buying_power": 28679.04,
            "positions": {},
        },
        "quotes": {},
    }

    result = mod.build_preview_result(
        payload,
        state_active_symbol=None,
        state_pending_close_symbol=None,
        allocation_pct=12.5,
        allocation_sizing_mode="static",
        min_open_confidence=0.05,
        forced_market_open=True,
        server_account="live_prod",
        server_bot_id="daily_stock_sortino_v1",
    )

    assert result["plan"]["summary"] == "No order; signal is flat"
    assert result["plan"]["would_submit"] is False
    rendered = mod.format_preview_text(result)
    assert "plan: No order; signal is flat" in rendered
    assert "safety: dry-run only" in rendered


def test_build_runtime_config_uses_valid_non_backtest_defaults() -> None:
    mod = _load_module()
    args = mod.parse_args([])
    config = mod._build_runtime_config(args)

    assert config.backtest is False
    assert config.backtest_days >= 1
    assert config.dry_run is True


def test_build_preview_result_open_leg_reports_qty_and_limit() -> None:
    mod = _load_module()
    payload = {
        "action": "buy",
        "symbol": "NVDA",
        "direction": "long",
        "confidence": 0.7,
        "value_estimate": 1.8,
        "allocation_fraction": 1.0,
        "bars_fresh": True,
        "latest_bar_timestamp": "2026-04-14T04:00:00+00:00",
        "market_open": True,
        "bar_data_source": "alpaca",
        "quote_data_source": "alpaca",
        "execution_status": "dry_run_would_execute",
        "execution_skip_reason": None,
        "execution_would_submit": True,
        "server_snapshot": {
            "cash": 1000.0,
            "equity": 1000.0,
            "buying_power": 1000.0,
            "positions": {},
        },
        "quotes": {"NVDA": 100.0},
    }

    result = mod.build_preview_result(
        payload,
        state_active_symbol=None,
        state_pending_close_symbol=None,
        allocation_pct=10.0,
        allocation_sizing_mode="static",
        min_open_confidence=0.05,
        forced_market_open=True,
        server_account="live_prod",
        server_bot_id="daily_stock_sortino_v1",
    )

    open_leg = result["plan"]["open_leg"]
    assert result["plan"]["summary"] == "Would open NVDA"
    assert result["plan"]["would_submit"] is True
    assert open_leg["symbol"] == "NVDA"
    assert open_leg["qty"] > 0.0
    assert open_leg["limit_price"] > open_leg["ref_price"]
    rendered = mod.format_preview_text(result)
    assert "open leg: NVDA" in rendered


def test_build_preview_result_rotate_reports_close_and_open() -> None:
    mod = _load_module()
    payload = {
        "action": "buy",
        "symbol": "NVDA",
        "direction": "long",
        "confidence": 0.7,
        "value_estimate": 1.8,
        "allocation_fraction": 1.0,
        "bars_fresh": True,
        "latest_bar_timestamp": "2026-04-14T04:00:00+00:00",
        "market_open": True,
        "bar_data_source": "alpaca",
        "quote_data_source": "alpaca",
        "execution_status": "dry_run_would_execute",
        "execution_skip_reason": None,
        "execution_would_submit": True,
        "server_snapshot": {
            "cash": 200.0,
            "equity": 1000.0,
            "buying_power": 200.0,
            "positions": {
                "AAPL": {
                    "qty": 2.0,
                    "avg_entry_price": 150.0,
                    "current_price": 160.0,
                }
            },
        },
        "quotes": {"NVDA": 100.0, "AAPL": 160.0},
    }

    result = mod.build_preview_result(
        payload,
        state_active_symbol="AAPL",
        state_pending_close_symbol=None,
        allocation_pct=10.0,
        allocation_sizing_mode="static",
        min_open_confidence=0.05,
        forced_market_open=True,
        server_account="live_prod",
        server_bot_id="daily_stock_sortino_v1",
    )

    assert result["plan"]["summary"] == "Would rotate AAPL -> NVDA"
    assert result["plan"]["close_leg"]["symbol"] == "AAPL"
    assert result["plan"]["open_leg"]["symbol"] == "NVDA"
    rendered = mod.format_preview_text(result)
    assert "close leg: AAPL" in rendered
    assert "open leg: NVDA" in rendered
