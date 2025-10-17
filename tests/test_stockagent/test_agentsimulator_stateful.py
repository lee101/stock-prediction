import json
from datetime import datetime, timezone, date
from pathlib import Path

import pandas as pd
import pytest

from stockagent.agentsimulator.market_data import MarketDataBundle, fetch_latest_ohlc
from stockagent.agentsimulator.prompt_builder import (
    build_daily_plan_prompt,
    dump_prompt_package,
    plan_response_schema,
)


def _sample_frame() -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=3, freq="D", tz="UTC")
    data = {
        "open": [100.0, 102.0, 103.0],
        "high": [100.0, 103.0, 104.0],
        "low": [100.0, 101.0, 102.0],
        "close": [100.0, 102.0, 104.0],
    }
    return pd.DataFrame(data, index=index)


def test_fetch_latest_ohlc_uses_local_cache(tmp_path: Path) -> None:
    df = _sample_frame().reset_index().rename(columns={"index": "timestamp"})
    csv_path = tmp_path / "AAPL_sample.csv"
    df.to_csv(csv_path, index=False)

    bundle = fetch_latest_ohlc(
        symbols=["AAPL"],
        lookback_days=2,
        as_of=datetime(2025, 1, 10, tzinfo=timezone.utc),
        local_data_dir=tmp_path,
    )

    bars = bundle.get_symbol_bars("AAPL")
    assert len(bars) == 2
    assert list(bars.index) == sorted(bars.index)
    trading_days = bundle.trading_days()
    assert len(trading_days) == len(bars)

    payload = bundle.to_payload()
    history = payload["AAPL"]
    assert len(history) == 2
    first = history[0]
    assert set(first.keys()) == {"timestamp", "open_pct", "high_pct", "low_pct", "close_pct"}
    assert first["open_pct"] == pytest.approx(0.0)
    last = history[-1]
    assert last["open_pct"] == pytest.approx((103.0 - 102.0) / 102.0)
    assert last["close_pct"] == pytest.approx((104.0 - 102.0) / 102.0)


def test_build_daily_plan_prompt_includes_account_percent_history() -> None:
    bundle = MarketDataBundle(
        bars={"AAPL": _sample_frame()},
        lookback_days=3,
        as_of=datetime(2025, 1, 4, tzinfo=timezone.utc),
    )
    account_payload = {
        "equity": 1_000_000.0,
        "cash": 500_000.0,
        "buying_power": 1_500_000.0,
        "timestamp": "2025-01-03T00:00:00+00:00",
        "positions": [],
    }
    target = date(2025, 1, 6)

    prompt, payload = build_daily_plan_prompt(
        market_data=bundle,
        account_payload=account_payload,
        target_date=target,
        symbols=["AAPL"],
        include_market_history=True,
    )

    assert "percent changes per symbol" in prompt
    assert "capital allocation" in prompt.lower()
    assert "capital_allocation_plan" in prompt
    assert "trainingdata/" in prompt
    assert str(bundle.lookback_days) in prompt
    assert payload["account"]["equity"] == account_payload["equity"]
    history = payload["market_data"]["AAPL"]
    assert len(history) == 3
    assert history[1]["close_pct"] == pytest.approx(0.02)


def test_dump_prompt_package_serializes_expected_payload() -> None:
    bundle = MarketDataBundle(
        bars={"AAPL": _sample_frame()},
        lookback_days=3,
        as_of=datetime(2025, 1, 4, tzinfo=timezone.utc),
    )
    package = dump_prompt_package(
        market_data=bundle,
        target_date=date(2025, 1, 6),
        include_market_history=True,
    )

    assert {"system_prompt", "user_prompt", "user_payload_json"} <= set(package.keys())
    payload = json.loads(package["user_payload_json"])
    assert "account" in payload
    assert "market_data" in payload
    assert payload["market_data"]["AAPL"][2]["high_pct"] == pytest.approx((104.0 - 102.0) / 102.0)

    schema = plan_response_schema()
    requirements = schema["properties"]["plan"]["properties"]["instructions"]["items"]["required"]
    assert {"symbol", "action", "quantity", "execution_session"} <= set(requirements)
