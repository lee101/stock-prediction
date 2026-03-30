import json
from datetime import datetime, timezone, date
from pathlib import Path

import pandas as pd
import pytest

from stockagent.agentsimulator.market_data import MarketDataBundle as StatefulMarketDataBundle
from stockagent.agentsimulator.prompt_builder import plan_response_schema as stateful_plan_response_schema
from stockagentindependant.agentsimulator import market_data as market_data_module
from stockagentindependant.agentsimulator.market_data import MarketDataBundle, fetch_latest_ohlc
from stockagentindependant.agentsimulator.prompt_builder import (
    build_daily_plan_prompt,
    dump_prompt_package,
    plan_response_schema,
)


def _sample_frame() -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=3, freq="D", tz="UTC")
    data = {
        "open": [50.0, 51.0, 52.0],
        "high": [50.0, 52.0, 53.0],
        "low": [50.0, 50.5, 51.0],
        "close": [50.0, 52.0, 54.0],
    }
    return pd.DataFrame(data, index=index)


def test_stateless_market_data_bundle_reuses_shared_contract() -> None:
    assert MarketDataBundle is StatefulMarketDataBundle


def test_stateless_prompt_schema_reuses_shared_contract() -> None:
    stateless_schema = plan_response_schema()
    stateful_schema = stateful_plan_response_schema()

    assert stateless_schema == stateful_schema
    assert stateless_schema is not stateful_schema


def test_fetch_latest_ohlc_stateless_local(tmp_path: Path) -> None:
    df = _sample_frame().reset_index().rename(columns={"index": "timestamp"})
    csv_path = tmp_path / "MSFT_sample.csv"
    df.to_csv(csv_path, index=False)

    bundle = fetch_latest_ohlc(
        symbols=["MSFT"],
        lookback_days=2,
        as_of=datetime(2025, 1, 10, tzinfo=timezone.utc),
        local_data_dir=tmp_path,
        allow_remote_download=False,
    )

    bars = bundle.get_symbol_bars("MSFT")
    assert len(bars) == 2
    history = bundle.to_payload()
    first = history["MSFT"][0]
    assert first["open_pct"] == pytest.approx(0.0)
    last = history["MSFT"][-1]
    assert last["high_pct"] == pytest.approx((53.0 - 52.0) / 52.0)
    assert last["close_pct"] == pytest.approx((54.0 - 52.0) / 52.0)


def test_fetch_latest_ohlc_stateless_rejects_unsafe_symbol(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported symbol"):
        fetch_latest_ohlc(
            symbols=["../secrets"],
            lookback_days=2,
            as_of=datetime(2025, 1, 10, tzinfo=timezone.utc),
            local_data_dir=tmp_path,
        )


def test_fetch_latest_ohlc_stateless_can_disable_fallback_dirs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary_dir = tmp_path / "primary"
    fallback_dir = tmp_path / "fallback"
    primary_dir.mkdir()
    fallback_dir.mkdir()

    frame = _sample_frame().reset_index().rename(columns={"index": "timestamp"})
    frame.to_csv(fallback_dir / "MSFT_only_in_fallback.csv", index=False)
    monkeypatch.setattr(market_data_module, "FALLBACK_DATA_DIRS", [fallback_dir])

    bundle = fetch_latest_ohlc(
        symbols=["MSFT"],
        lookback_days=3,
        as_of=datetime(2025, 1, 10, tzinfo=timezone.utc),
        local_data_dir=primary_dir,
        use_fallback_data_dirs=False,
    )

    assert bundle.get_symbol_bars("MSFT").empty


def test_build_daily_plan_prompt_stateless_payload() -> None:
    bundle = MarketDataBundle(
        bars={"MSFT": _sample_frame()},
        lookback_days=3,
        as_of=datetime(2025, 1, 4, tzinfo=timezone.utc),
    )
    prompt, payload = build_daily_plan_prompt(
        market_data=bundle,
        target_date=date(2025, 1, 7),
        symbols=["MSFT"],
        include_market_history=True,
    )

    assert "paper-trading benchmark" in prompt
    assert "percent changes per symbol" in prompt
    assert "capital allocation" in prompt.lower()
    assert "capital_allocation_plan" in prompt
    assert "trainingdata/" in prompt
    assert "market_data" in payload
    assert "account" not in payload
    history = payload["market_data"]["MSFT"]
    assert history[1]["high_pct"] == pytest.approx(0.04)


def test_dump_prompt_package_stateless_json() -> None:
    bundle = MarketDataBundle(
        bars={"MSFT": _sample_frame()},
        lookback_days=3,
        as_of=datetime(2025, 1, 4, tzinfo=timezone.utc),
    )
    package = dump_prompt_package(
        market_data=bundle,
        target_date=date(2025, 1, 7),
        include_market_history=True,
    )
    payload = json.loads(package["user_payload_json"])
    assert "market_data" in payload
    assert "account" not in payload
    assert payload["market_data"]["MSFT"][2]["close_pct"] == pytest.approx((54.0 - 52.0) / 52.0)

    schema = plan_response_schema()
    assert set(schema.get("required", [])) >= {"target_date", "instructions"}
    required_fields = set(schema["properties"]["instructions"]["items"].get("required", []))
    assert {
        "symbol",
        "action",
        "quantity",
        "execution_session",
        "entry_price",
        "exit_price",
        "exit_reason",
        "notes",
    } <= required_fields
    assert "notes" in required_fields
