"""Tests for the LLM forecast pipeline: loading, formatting, prompts, metrics."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_hourly_trader.experiment_runner import (
    _compute_metrics,
    get_forecast_at,
    load_bars,
    load_forecasts,
)
from llm_hourly_trader.gemini_wrapper import _format_forecasts, build_prompt

FORECAST_CACHE = Path(__file__).resolve().parent.parent / "binanceneural" / "forecast_cache"
DATA_DIR = Path(__file__).resolve().parent.parent / "trainingdatahourly" / "crypto"

# ---- fixtures ----

SAMPLE_FORECAST = {
    "predicted_close_p50": 70000.0,
    "predicted_close_p10": 69000.0,
    "predicted_close_p90": 71000.0,
    "predicted_high_p50": 70500.0,
    "predicted_low_p50": 69500.0,
}

SAMPLE_HISTORY = [
    {"timestamp": "2026-03-10T00:00", "open": 69000, "high": 69500, "low": 68800, "close": 69200, "volume": 100},
    {"timestamp": "2026-03-10T01:00", "open": 69200, "high": 69800, "low": 69100, "close": 69700, "volume": 120},
    {"timestamp": "2026-03-10T02:00", "open": 69700, "high": 70100, "low": 69600, "close": 70000, "volume": 110},
]


def _make_fc_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


# ---- unit tests: _format_forecasts ----

def test_format_forecasts_with_valid_data():
    out = _format_forecasts(SAMPLE_FORECAST, SAMPLE_FORECAST)
    assert "70000.00" in out
    assert "69000.00" in out
    assert "1h ahead" in out
    assert "24h ahead" in out


def test_format_forecasts_with_none():
    out = _format_forecasts(None, None)
    assert "no forecasts" in out.lower()


def test_format_forecasts_h1_only():
    out = _format_forecasts(SAMPLE_FORECAST, None)
    assert "1h ahead" in out
    assert "24h ahead" not in out


def test_format_forecasts_h24_only():
    out = _format_forecasts(None, SAMPLE_FORECAST)
    assert "24h ahead" in out
    assert "1h ahead" not in out


# ---- unit tests: build_prompt variants ----

@pytest.mark.parametrize("variant", [
    "default", "conservative", "aggressive",
    "position_context", "fractional", "anonymized",
    "no_forecast", "h1_only", "h24_only",
    "uncertainty_gated", "uncertainty_strict",
])
def test_build_prompt_all_variants(variant):
    prompt = build_prompt(
        symbol="BTCUSD",
        history_rows=SAMPLE_HISTORY,
        forecast_1h=SAMPLE_FORECAST,
        forecast_24h=SAMPLE_FORECAST,
        current_position="flat",
        cash=2000.0,
        equity=2000.0,
        allowed_directions=["long"],
        asset_class="crypto",
        maker_fee=0.001,
        variant=variant,
    )
    assert isinstance(prompt, str)
    assert len(prompt) > 100


def test_build_prompt_position_context_includes_pnl():
    prompt = build_prompt(
        symbol="BTCUSD",
        history_rows=SAMPLE_HISTORY,
        forecast_1h=SAMPLE_FORECAST,
        forecast_24h=SAMPLE_FORECAST,
        current_position="long 0.01 @ $69000",
        cash=1500.0,
        equity=2200.0,
        allowed_directions=["long"],
        asset_class="crypto",
        variant="position_context",
        position_info={"qty": 0.01, "entry_price": 69000.0, "held_hours": 3},
    )
    assert "69000" in prompt
    assert "CURRENT POSITION" in prompt
    assert "3 hours" in prompt or "3h" in prompt


def test_build_prompt_position_context_flat():
    prompt = build_prompt(
        symbol="BTCUSD",
        history_rows=SAMPLE_HISTORY,
        forecast_1h=SAMPLE_FORECAST,
        forecast_24h=SAMPLE_FORECAST,
        current_position="flat",
        cash=2000.0,
        equity=2000.0,
        allowed_directions=["long"],
        variant="position_context",
    )
    assert "Flat" in prompt


# ---- unit tests: _compute_metrics ----

def test_compute_metrics_known_curve():
    equity = np.array([1000, 1010, 1005, 1020, 1015, 1030])
    m = _compute_metrics(equity)
    assert abs(m["return_pct"] - 3.0) < 0.01
    assert m["max_dd_pct"] > 0
    assert m["sortino"] != 0  # 2 down-steps so downside_std > 0


def test_compute_metrics_flat_curve():
    equity = np.array([1000, 1000, 1000, 1000])
    m = _compute_metrics(equity)
    assert m["return_pct"] == 0
    assert m["sortino"] == 0
    assert m["max_dd_pct"] == 0


def test_compute_metrics_monotonic_up():
    equity = np.array([1000, 1010, 1020, 1030, 1040])
    m = _compute_metrics(equity)
    assert m["return_pct"] == pytest.approx(4.0, abs=0.01)
    assert m["max_dd_pct"] == 0.0
    assert m["sortino"] == 0  # no downside deviation


def test_compute_metrics_single_element():
    m = _compute_metrics(np.array([1000]))
    assert m["return_pct"] == 0


# ---- unit tests: get_forecast_at ----

def test_get_forecast_at_returns_latest_before_ts():
    rows = [
        {"timestamp": "2026-03-10T00:00", "predicted_close_p50": 100.0, "predicted_high_p50": 101.0, "predicted_low_p50": 99.0},
        {"timestamp": "2026-03-10T01:00", "predicted_close_p50": 200.0, "predicted_high_p50": 201.0, "predicted_low_p50": 199.0},
        {"timestamp": "2026-03-10T02:00", "predicted_close_p50": 300.0, "predicted_high_p50": 301.0, "predicted_low_p50": 299.0},
    ]
    df = _make_fc_df(rows)
    ts = pd.Timestamp("2026-03-10T01:30", tz="UTC")
    fc = get_forecast_at(df, ts)
    assert fc is not None
    assert fc["predicted_close_p50"] == 200.0


def test_get_forecast_at_exact_match():
    rows = [
        {"timestamp": "2026-03-10T01:00", "predicted_close_p50": 500.0, "predicted_high_p50": 501.0, "predicted_low_p50": 499.0},
    ]
    df = _make_fc_df(rows)
    ts = pd.Timestamp("2026-03-10T01:00", tz="UTC")
    fc = get_forecast_at(df, ts)
    assert fc["predicted_close_p50"] == 500.0


def test_get_forecast_at_before_all_returns_none():
    rows = [
        {"timestamp": "2026-03-10T05:00", "predicted_close_p50": 100.0, "predicted_high_p50": 101.0, "predicted_low_p50": 99.0},
    ]
    df = _make_fc_df(rows)
    ts = pd.Timestamp("2026-03-10T00:00", tz="UTC")
    assert get_forecast_at(df, ts) is None


def test_get_forecast_at_empty_df():
    assert get_forecast_at(pd.DataFrame(), pd.Timestamp("2026-03-10", tz="UTC")) is None


def test_get_forecast_at_skips_symbol_column():
    rows = [
        {"timestamp": "2026-03-10T01:00", "symbol": "BTCUSD", "predicted_close_p50": 70000.0},
    ]
    df = _make_fc_df(rows)
    fc = get_forecast_at(df, pd.Timestamp("2026-03-10T02:00", tz="UTC"))
    assert fc is not None
    assert "symbol" not in fc
    assert fc["predicted_close_p50"] == 70000.0


# ---- integration tests: real data ----

@pytest.mark.slow
def test_forecast_cache_h1_schema():
    path = FORECAST_CACHE / "h1" / "BTCUSD.parquet"
    if not path.exists():
        pytest.skip("h1/BTCUSD.parquet not found")
    df = pd.read_parquet(path)
    for col in ["timestamp", "predicted_close_p50", "predicted_close_p10", "predicted_close_p90",
                "predicted_high_p50", "predicted_low_p50"]:
        assert col in df.columns, f"missing column: {col}"
    assert len(df) > 1000


@pytest.mark.slow
def test_forecast_cache_h24_schema():
    path = FORECAST_CACHE / "h24" / "BTCUSD.parquet"
    if not path.exists():
        pytest.skip("h24/BTCUSD.parquet not found")
    df = pd.read_parquet(path)
    for col in ["timestamp", "predicted_close_p50", "predicted_close_p10", "predicted_close_p90"]:
        assert col in df.columns, f"missing column: {col}"
    assert len(df) > 1000


@pytest.mark.slow
def test_forecast_cache_recent():
    path = FORECAST_CACHE / "h1" / "BTCUSD.parquet"
    if not path.exists():
        pytest.skip("h1/BTCUSD.parquet not found")
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    last = df["timestamp"].max()
    now = pd.Timestamp.now(tz="UTC")
    assert (now - last).total_seconds() < 48 * 3600, f"last forecast is {last}, stale"


@pytest.mark.slow
@pytest.mark.parametrize("symbol", ["BTCUSD", "ETHUSD", "SOLUSD"])
@pytest.mark.parametrize("horizon", ["h1", "h24"])
def test_forecast_cache_all_symbols_present(symbol, horizon):
    path = FORECAST_CACHE / horizon / f"{symbol}.parquet"
    assert path.exists(), f"missing: {path}"
    df = pd.read_parquet(path)
    assert len(df) > 100, f"{symbol}/{horizon} has only {len(df)} rows"


@pytest.mark.slow
def test_load_bars_real_btc():
    df = load_bars("BTCUSD")
    assert not df.empty
    for col in ["timestamp", "open", "high", "low", "close"]:
        assert col in df.columns
    assert len(df) > 1000
