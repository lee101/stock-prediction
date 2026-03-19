"""Tests for multi-horizon forecast expansion (h1, h4, h12, h24)."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "rl-trading-agent-binance"))

from binanceneural.data import build_default_feature_columns
from rl_trading_agent_binance_prompt import (
    FORECAST_MAE_1H,
    FORECAST_MAE_4H,
    FORECAST_MAE_12H,
    FORECAST_MAE_BY_HORIZON,
    build_live_prompt,
    build_live_prompt_freeform,
    load_latest_forecast,
)


def _make_history(n=50, base_price=85000.0):
    rows = []
    for i in range(n):
        p = base_price + i * 10
        rows.append({
            "timestamp": f"2026-03-19T{i%24:02d}:00:00",
            "open": p - 5, "high": p + 20, "low": p - 20, "close": p,
            "volume": 1000 + i,
        })
    return rows


def _make_forecast(price, delta_pct=0.5):
    d = price * delta_pct / 100
    return {
        "predicted_close_p50": price + d,
        "predicted_high_p50": price + 2 * d,
        "predicted_low_p50": price - d,
        "predicted_close_p90": price + 3 * d,
        "predicted_close_p10": price - 2 * d,
    }


class TestMAEDicts:
    def test_mae_4h_interpolated_above_1h(self):
        for sym in FORECAST_MAE_1H:
            assert FORECAST_MAE_4H[sym] > FORECAST_MAE_1H[sym]

    def test_mae_12h_interpolated_above_4h(self):
        for sym in FORECAST_MAE_4H:
            assert FORECAST_MAE_12H[sym] > FORECAST_MAE_4H[sym]

    def test_by_horizon_lookup(self):
        assert FORECAST_MAE_BY_HORIZON[1] is FORECAST_MAE_1H
        assert FORECAST_MAE_BY_HORIZON[4] is FORECAST_MAE_4H
        assert FORECAST_MAE_BY_HORIZON[12] is FORECAST_MAE_12H


class TestBuildLivePromptAllHorizons:
    def test_all_four_horizons(self):
        rows = _make_history()
        price = rows[-1]["close"]
        fc1 = _make_forecast(price, 0.3)
        fc4 = _make_forecast(price, 0.8)
        fc12 = _make_forecast(price, 1.5)
        fc24 = _make_forecast(price, 2.5)
        prompt = build_live_prompt(
            "BTCUSD", rows, price,
            fc_1h=fc1, fc_24h=fc24, fc_4h=fc4, fc_12h=fc12,
        )
        assert "1-hour ahead:" in prompt
        assert "4-hour ahead:" in prompt
        assert "12-hour ahead:" in prompt
        assert "24-hour ahead:" in prompt

    def test_ordering_in_prompt(self):
        rows = _make_history()
        price = rows[-1]["close"]
        fc1 = _make_forecast(price, 0.3)
        fc4 = _make_forecast(price, 0.8)
        fc12 = _make_forecast(price, 1.5)
        fc24 = _make_forecast(price, 2.5)
        prompt = build_live_prompt(
            "BTCUSD", rows, price,
            fc_1h=fc1, fc_24h=fc24, fc_4h=fc4, fc_12h=fc12,
        )
        idx1 = prompt.index("1-hour ahead:")
        idx4 = prompt.index("4-hour ahead:")
        idx12 = prompt.index("12-hour ahead:")
        idx24 = prompt.index("24-hour ahead:")
        assert idx1 < idx4 < idx12 < idx24

    def test_missing_4h_12h_graceful(self):
        rows = _make_history()
        price = rows[-1]["close"]
        fc1 = _make_forecast(price, 0.3)
        fc24 = _make_forecast(price, 2.5)
        prompt = build_live_prompt(
            "BTCUSD", rows, price,
            fc_1h=fc1, fc_24h=fc24,
        )
        assert "1-hour ahead:" in prompt
        assert "24-hour ahead:" in prompt
        assert "\n4-hour ahead:" not in prompt
        assert "\n12-hour ahead:" not in prompt

    def test_only_4h_present(self):
        rows = _make_history()
        price = rows[-1]["close"]
        fc4 = _make_forecast(price, 0.8)
        prompt = build_live_prompt(
            "BTCUSD", rows, price,
            fc_4h=fc4,
        )
        assert "4-hour ahead:" in prompt
        assert "CHRONOS2 ML FORECASTS:" in prompt

    def test_mae_per_horizon_in_prompt(self):
        rows = _make_history()
        price = rows[-1]["close"]
        fc1 = _make_forecast(price)
        fc4 = _make_forecast(price)
        fc12 = _make_forecast(price)
        prompt = build_live_prompt(
            "BTCUSD", rows, price,
            fc_1h=fc1, fc_4h=fc4, fc_12h=fc12,
        )
        assert "0.55%" in prompt  # 1h MAE for BTCUSD
        assert "0.83%" in prompt  # 4h MAE for BTCUSD
        assert "1.38%" in prompt  # 12h MAE for BTCUSD


class TestBuildLivePromptFreeformAllHorizons:
    def test_all_four_horizons(self):
        rows = _make_history()
        price = rows[-1]["close"]
        fc1 = _make_forecast(price, 0.3)
        fc4 = _make_forecast(price, 0.8)
        fc12 = _make_forecast(price, 1.5)
        fc24 = _make_forecast(price, 2.5)
        prompt = build_live_prompt_freeform(
            "BTCUSD", rows, price,
            fc_1h=fc1, fc_24h=fc24, fc_4h=fc4, fc_12h=fc12,
        )
        assert "1h ahead:" in prompt
        assert "4h ahead:" in prompt
        assert "12h ahead:" in prompt
        assert "24h ahead:" in prompt

    def test_missing_4h_12h_graceful(self):
        rows = _make_history()
        price = rows[-1]["close"]
        fc1 = _make_forecast(price, 0.3)
        fc24 = _make_forecast(price, 2.5)
        prompt = build_live_prompt_freeform(
            "BTCUSD", rows, price,
            fc_1h=fc1, fc_24h=fc24,
        )
        assert "1h ahead:" in prompt
        assert "24h ahead:" in prompt
        assert "\n4h ahead:" not in prompt
        assert "\n12h ahead:" not in prompt

    def test_forecast_error_4h_12h(self):
        rows = _make_history()
        price = rows[-1]["close"]
        fc4 = _make_forecast(price, 0.8)
        fc12 = _make_forecast(price, 1.5)
        prompt = build_live_prompt_freeform(
            "BTCUSD", rows, price,
            fc_4h=fc4, fc_12h=fc12,
            forecast_error_4h={"mae_pct": 1.2, "samples": 50},
            forecast_error_12h={"mae_pct": 2.8, "samples": 30},
        )
        assert "1.20%" in prompt
        assert "n=50" in prompt
        assert "2.80%" in prompt
        assert "n=30" in prompt


class TestFeatureColumnCount:
    def test_two_horizons_legacy(self):
        cols = build_default_feature_columns((1, 24))
        assert len(cols) == 10 + 3 * 2  # 10 base + 3 per horizon

    def test_four_horizons(self):
        cols = build_default_feature_columns((1, 4, 12, 24))
        assert len(cols) == 10 + 3 * 4  # 10 base + 3 per horizon

    def test_column_names(self):
        cols = build_default_feature_columns((1, 4, 12, 24))
        for h in [1, 4, 12, 24]:
            assert f"chronos_close_delta_h{h}" in cols
            assert f"chronos_high_delta_h{h}" in cols
            assert f"chronos_low_delta_h{h}" in cols


class TestLoadLatestForecastMissing:
    def test_missing_cache_returns_none(self, tmp_path):
        result = load_latest_forecast("BTCUSD", 4, cache_root=tmp_path)
        assert result is None

    def test_missing_h12_returns_none(self, tmp_path):
        result = load_latest_forecast("BTCUSD", 12, cache_root=tmp_path)
        assert result is None


class TestBackwardsCompatibility:
    def test_old_signature_works(self):
        rows = _make_history()
        price = rows[-1]["close"]
        fc1 = _make_forecast(price)
        fc24 = _make_forecast(price)
        prompt = build_live_prompt("BTCUSD", rows, price, fc1, fc24)
        assert "BTCUSD" in prompt
        assert "1-hour ahead:" in prompt

    def test_old_freeform_signature_works(self):
        rows = _make_history()
        price = rows[-1]["close"]
        fc1 = _make_forecast(price)
        fc24 = _make_forecast(price)
        prompt = build_live_prompt_freeform("BTCUSD", rows, price, fc1, fc24)
        assert "BTCUSD" in prompt
        assert "1h ahead:" in prompt
