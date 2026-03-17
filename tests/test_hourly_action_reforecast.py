from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from llm_hourly_trader.gemini_wrapper import TradePlan
from src.hourly_action_reforecast import (
    AMOUNT_REFORECAST_MODE,
    GEMINI_REFORECAST_MODE,
    HourlyActionReforecastConfig,
    apply_hourly_action_reforecasting,
    parse_reforecast_modes,
)


def _history_frame(
    symbol: str,
    *,
    weighted_delta: float,
    agreement: float = 0.9,
    confidence_h1: float = 2.0,
    confidence_h24: float = 2.5,
) -> pd.DataFrame:
    timestamps = pd.date_range("2026-03-01T00:00:00Z", periods=10, freq="h", tz="UTC")
    closes = np.linspace(95.0, 100.0, len(timestamps))
    predicted_close_h1 = closes * (1.0 + weighted_delta)
    predicted_close_h24 = closes * (1.0 + (weighted_delta * 1.5))
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": symbol,
            "open": closes - 0.5,
            "high": closes + 0.75,
            "low": closes - 0.75,
            "close": closes,
            "volume": np.linspace(1000.0, 2000.0, len(timestamps)),
            "predicted_close_p10_h1": predicted_close_h1 * 0.995,
            "predicted_close_p50_h1": predicted_close_h1,
            "predicted_close_p90_h1": predicted_close_h1 * 1.005,
            "predicted_high_p50_h1": predicted_close_h1 * 1.010,
            "predicted_low_p50_h1": predicted_close_h1 * 0.990,
            "predicted_close_p10_h24": predicted_close_h24 * 0.990,
            "predicted_close_p50_h24": predicted_close_h24,
            "predicted_close_p90_h24": predicted_close_h24 * 1.010,
            "predicted_high_p50_h24": predicted_close_h24 * 1.020,
            "predicted_low_p50_h24": predicted_close_h24 * 0.985,
            "forecast_agreement": agreement,
            "forecast_confidence_h1": confidence_h1,
            "forecast_confidence_h24": confidence_h24,
            "forecast_weighted_delta": weighted_delta,
            "chronos_close_delta_h1": weighted_delta,
            "chronos_close_delta_h24": weighted_delta * 1.2,
        }
    )
    return frame


def test_parse_reforecast_modes_normalizes_aliases_and_deduplicates() -> None:
    assert parse_reforecast_modes("baseline, amount, gemini+amount_reforecasting, amount") == [
        "baseline",
        "amount_reforecasting",
        "gemini+amount_reforecasting",
    ]


def test_amount_reforecasting_scales_to_gross_limit_and_favors_stronger_symbol() -> None:
    timestamp = pd.Timestamp("2026-03-01T09:00:00Z")
    actions = pd.DataFrame(
        [
            {
                "timestamp": timestamp,
                "symbol": "BTCUSD",
                "buy_price": 99.0,
                "sell_price": 104.0,
                "buy_amount": 60.0,
                "sell_amount": 5.0,
                "trade_amount": 60.0,
                "allocation_fraction": 0.60,
            },
            {
                "timestamp": timestamp,
                "symbol": "ETHUSD",
                "buy_price": 99.5,
                "sell_price": 101.5,
                "buy_amount": 60.0,
                "sell_amount": 5.0,
                "trade_amount": 60.0,
                "allocation_fraction": 0.60,
            },
        ]
    )
    history_by_symbol = {
        "BTCUSD": _history_frame("BTCUSD", weighted_delta=0.030, agreement=0.95, confidence_h1=4.0, confidence_h24=4.5),
        "ETHUSD": _history_frame("ETHUSD", weighted_delta=0.008, agreement=0.40, confidence_h1=1.2, confidence_h24=1.0),
    }

    result = apply_hourly_action_reforecasting(
        actions,
        history_by_symbol,
        config=HourlyActionReforecastConfig(
            mode=AMOUNT_REFORECAST_MODE,
            max_gross_allocation=0.40,
        ),
        allow_short=False,
    )

    by_symbol = result.set_index("symbol")
    assert by_symbol.loc["BTCUSD", "buy_amount"] > by_symbol.loc["ETHUSD", "buy_amount"]
    assert float(by_symbol["allocation_fraction"].sum()) <= 0.400000001
    assert by_symbol["trade_amount"].max() <= 100.0


def test_gemini_reforecasting_updates_prices(monkeypatch: pytest.MonkeyPatch) -> None:
    history_by_symbol = {"SOLUSD": _history_frame("SOLUSD", weighted_delta=0.015)}
    actions = pd.DataFrame(
        [
            {
                "timestamp": history_by_symbol["SOLUSD"]["timestamp"].iloc[-1],
                "symbol": "SOLUSD",
                "buy_price": 99.0,
                "sell_price": 101.0,
                "buy_amount": 55.0,
                "sell_amount": 0.0,
                "trade_amount": 55.0,
                "allocation_fraction": 0.55,
            }
        ]
    )

    monkeypatch.setattr(
        "src.hourly_action_reforecast.call_llm",
        lambda *args, **kwargs: TradePlan(
            direction="long",
            buy_price=98.25,
            sell_price=103.75,
            confidence=0.7,
            reasoning="test",
        ),
    )

    result = apply_hourly_action_reforecasting(
        actions,
        history_by_symbol,
        config=HourlyActionReforecastConfig(mode=GEMINI_REFORECAST_MODE),
        allow_short=False,
    )

    assert result.iloc[0]["buy_price"] == pytest.approx(98.25)
    assert result.iloc[0]["sell_price"] == pytest.approx(103.75)
    assert result.iloc[0]["buy_amount"] == pytest.approx(55.0)


def test_gemini_reforecasting_hold_suppresses_dominant_entry_side(monkeypatch: pytest.MonkeyPatch) -> None:
    history_by_symbol = {"LINKUSD": _history_frame("LINKUSD", weighted_delta=0.012)}
    actions = pd.DataFrame(
        [
            {
                "timestamp": history_by_symbol["LINKUSD"]["timestamp"].iloc[-1],
                "symbol": "LINKUSD",
                "buy_price": 99.0,
                "sell_price": 101.0,
                "buy_amount": 65.0,
                "sell_amount": 0.0,
                "trade_amount": 65.0,
                "allocation_fraction": 0.65,
            }
        ]
    )

    monkeypatch.setattr(
        "src.hourly_action_reforecast.call_llm",
        lambda *args, **kwargs: TradePlan(
            direction="hold",
            buy_price=0.0,
            sell_price=0.0,
            confidence=0.3,
            reasoning="test",
        ),
    )

    result = apply_hourly_action_reforecasting(
        actions,
        history_by_symbol,
        config=HourlyActionReforecastConfig(mode=GEMINI_REFORECAST_MODE),
        allow_short=False,
    )

    assert result.iloc[0]["buy_amount"] == 0.0
    assert result.iloc[0]["trade_amount"] == 0.0
    assert result.iloc[0]["allocation_fraction"] == 0.0
