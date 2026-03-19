from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "rl-trading-agent-binance" / "eval_new_symbol.py"


def _load_module():
    module_name = "eval_new_symbol_testmod"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_load_forecast_at_uses_alias_and_as_of(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "h24" / "BNBUSDT.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-03-10T00:00:00Z"),
                "issued_at": pd.Timestamp("2026-03-09T00:00:00Z"),
                "symbol": "BNBUSDT",
                "predicted_close_p50": 600.0,
                "predicted_high_p50": 605.0,
                "predicted_low_p50": 595.0,
            },
            {
                "timestamp": pd.Timestamp("2026-03-12T00:00:00Z"),
                "issued_at": pd.Timestamp("2026-03-11T00:00:00Z"),
                "symbol": "BNBUSDT",
                "predicted_close_p50": 630.0,
                "predicted_high_p50": 635.0,
                "predicted_low_p50": 625.0,
            },
        ]
    )
    frame.to_parquet(path, index=False)

    forecast = module.load_forecast_at(
        "BNBUSD",
        pd.Timestamp("2026-03-10T12:00:00Z"),
        24,
        cache_root=tmp_path,
    )

    assert forecast is not None
    assert forecast["predicted_close_p50"] == 600.0


def test_load_forecast_at_falls_back_to_default_repo_cache(tmp_path: Path) -> None:
    module = _load_module()
    module.REPO = tmp_path

    default_path = tmp_path / "binanceneural" / "forecast_cache" / "h1" / "BTCUSD.parquet"
    default_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-03-18T01:00:00Z"),
                "issued_at": pd.Timestamp("2026-03-18T00:00:00Z"),
                "symbol": "BTCUSD",
                "predicted_close_p50": 91000.0,
                "predicted_high_p50": 91250.0,
                "predicted_low_p50": 90750.0,
            },
        ]
    ).to_parquet(default_path, index=False)

    experiment_root = tmp_path / "experiment_cache"
    experiment_root.mkdir(parents=True, exist_ok=True)

    forecast = module.load_forecast_at(
        "BTCUSD",
        pd.Timestamp("2026-03-18T00:30:00Z"),
        1,
        cache_root=experiment_root,
    )

    assert forecast is not None
    assert forecast["predicted_close_p50"] == 91000.0


def test_build_forecast_rule_signal_returns_long_when_edge_is_positive() -> None:
    module = _load_module()
    signal = module.build_forecast_rule_signal(
        symbol="BNBUSD",
        current_price=600.0,
        fc_1h={
            "predicted_close_p50": 606.0,
            "predicted_high_p50": 609.0,
            "predicted_low_p50": 599.0,
        },
        fc_24h={
            "predicted_close_p50": 612.0,
            "predicted_high_p50": 618.0,
            "predicted_low_p50": 598.0,
        },
    )

    assert signal["direction"] == "long"
    assert signal["buy_price"] > 0.0
    assert signal["sell_price"] > signal["buy_price"]
    assert signal["confidence"] > 0.0


def test_build_forecast_rule_signal_respects_stricter_thresholds() -> None:
    module = _load_module()
    signal = module.build_forecast_rule_signal(
        symbol="BNBUSD",
        current_price=600.0,
        fc_1h={
            "predicted_close_p50": 606.0,
            "predicted_high_p50": 609.0,
            "predicted_low_p50": 599.0,
        },
        fc_24h={
            "predicted_close_p50": 612.0,
            "predicted_high_p50": 618.0,
            "predicted_low_p50": 598.0,
        },
        total_cost=0.05,
        min_reward_risk=2.0,
    )

    assert signal["direction"] == "hold"
    assert signal["confidence"] == 0.0


def test_cache_namespace_changes_with_forecast_root(tmp_path: Path) -> None:
    module = _load_module()

    ns_a = module._cache_namespace(
        signal_mode="gemini",
        model="gemini-3.1-flash-lite-preview",
        thinking_level="HIGH",
        forecast_cache_root=tmp_path / "cache_a",
    )
    ns_b = module._cache_namespace(
        signal_mode="gemini",
        model="gemini-3.1-flash-lite-preview",
        thinking_level="HIGH",
        forecast_cache_root=tmp_path / "cache_b",
    )

    assert ns_a != ns_b
    assert module._cache_key("BNBUSD", "20260318_00", ns_a) != module._cache_key("BNBUSD", "20260318_00", ns_b)


def test_simulate_portfolio_respects_symbol_max_pos_override() -> None:
    module = _load_module()
    bars = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-03-18T00:00:00Z"),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1.0,
            },
            {
                "timestamp": pd.Timestamp("2026-03-18T01:00:00Z"),
                "open": 100.0,
                "high": 110.0,
                "low": 100.0,
                "close": 110.0,
                "volume": 1.0,
            },
        ]
    )
    signals = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-03-18T00:00:00Z"),
                "symbol": "BNBUSD",
                "direction": "long",
                "buy_price": 100.0,
                "sell_price": 110.0,
                "confidence": 1.0,
                "reasoning": "test",
            }
        ]
    )

    default_result = module.simulate_portfolio(
        {"BNBUSD": bars},
        {"BNBUSD": signals},
        initial_cash=10000.0,
        leverage=5.0,
        margin_rate_annual=0.0,
        margin_fee=0.0,
    )
    override_result = module.simulate_portfolio(
        {"BNBUSD": bars},
        {"BNBUSD": signals},
        initial_cash=10000.0,
        leverage=5.0,
        margin_rate_annual=0.0,
        margin_fee=0.0,
        symbol_max_pos_overrides={"BNBUSD": 0.01},
    )

    assert default_result["per_symbol_pnl"]["BNBUSD"] > override_result["per_symbol_pnl"]["BNBUSD"]
