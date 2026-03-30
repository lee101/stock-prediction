import os
import json
from datetime import datetime, timezone, date
from pathlib import Path

import pandas as pd
import pytest

from stockagent.constants import (
    DEFAULT_MAX_NEW_POSITION_EQUITY_FRACTION,
    DEFAULT_MAX_NEW_POSITION_NOTIONAL_FLOOR,
)
from stockagent import agentsimulator as stateful_agentsimulator
from stockagent.agentsimulator import local_market_data as shared_local_market_data
from stockagent.agentsimulator import market_data as market_data_module
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


def test_fetch_latest_ohlc_rejects_unsafe_symbol(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported symbol"):
        fetch_latest_ohlc(
            symbols=["../secrets"],
            lookback_days=2,
            as_of=datetime(2025, 1, 10, tzinfo=timezone.utc),
            local_data_dir=tmp_path,
        )


def test_find_latest_local_symbol_files_scans_directory_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    frame = _sample_frame().reset_index().rename(columns={"index": "timestamp"})
    old_path = tmp_path / "AAPL_old.csv"
    new_path = tmp_path / "AAPL_new.csv"
    frame.to_csv(old_path, index=False)
    frame.assign(close=[200.0, 201.0, 202.0]).to_csv(new_path, index=False)
    frame.assign(close=[300.0, 301.0, 302.0]).to_csv(tmp_path / "MSFT_sample.csv", index=False)
    old_stat = old_path.stat()
    os.utime(new_path, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns + 1))

    original_iterdir = Path.iterdir
    calls = 0

    def _counting_iterdir(self: Path):
        nonlocal calls
        if self == tmp_path:
            calls += 1
        yield from original_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", _counting_iterdir)

    indexed = shared_local_market_data.find_latest_local_symbol_files(
        symbols=["AAPL", "MSFT"],
        directories=[tmp_path],
    )

    assert calls == 1
    assert indexed["AAPL"].name == "AAPL_new.csv"
    assert indexed["MSFT"].name == "MSFT_sample.csv"


def test_fetch_latest_ohlc_respects_directory_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first_dir = tmp_path / "preferred"
    second_dir = tmp_path / "fallback"
    first_dir.mkdir()
    second_dir.mkdir()

    frame = _sample_frame().reset_index().rename(columns={"index": "timestamp"})
    frame.assign(close=[100.0, 101.0, 102.0]).to_csv(first_dir / "AAPL_old.csv", index=False)
    frame.assign(close=[500.0, 501.0, 502.0]).to_csv(second_dir / "AAPL_new.csv", index=False)
    monkeypatch.setattr(market_data_module, "FALLBACK_DATA_DIRS", [second_dir])

    bundle = fetch_latest_ohlc(
        symbols=["AAPL"],
        lookback_days=3,
        as_of=datetime(2025, 1, 10, tzinfo=timezone.utc),
        local_data_dir=first_dir,
    )

    bars = bundle.get_symbol_bars("AAPL")
    assert list(bars["close"]) == [100.0, 101.0, 102.0]


def test_fetch_latest_ohlc_can_disable_fallback_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    primary_dir = tmp_path / "primary"
    fallback_dir = tmp_path / "fallback"
    primary_dir.mkdir()
    fallback_dir.mkdir()

    frame = _sample_frame().reset_index().rename(columns={"index": "timestamp"})
    frame.to_csv(fallback_dir / "AAPL_only_in_fallback.csv", index=False)
    monkeypatch.setattr(market_data_module, "FALLBACK_DATA_DIRS", [fallback_dir])

    bundle = fetch_latest_ohlc(
        symbols=["AAPL"],
        lookback_days=3,
        as_of=datetime(2025, 1, 10, tzinfo=timezone.utc),
        local_data_dir=primary_dir,
        use_fallback_data_dirs=False,
    )

    assert bundle.get_symbol_bars("AAPL").empty


def test_fetch_latest_ohlc_uses_remote_download_when_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def _fake_download(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        seen["symbol"] = symbol
        seen["start"] = start
        seen["end"] = end
        return _sample_frame()

    monkeypatch.setattr(market_data_module, "_download_remote_bars", _fake_download)

    bundle = fetch_latest_ohlc(
        symbols=["AAPL"],
        lookback_days=2,
        as_of=datetime(2025, 1, 10, tzinfo=timezone.utc),
        local_data_dir=tmp_path,
        allow_remote_download=True,
        use_fallback_data_dirs=False,
    )

    assert seen["symbol"] == "AAPL"
    assert seen["start"] == datetime(2024, 12, 11, tzinfo=timezone.utc)
    assert seen["end"] == datetime(2025, 1, 10, tzinfo=timezone.utc)
    assert list(bundle.get_symbol_bars("AAPL")["close"]) == [102.0, 104.0]


def test_resolve_local_data_dirs_uses_environment_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    custom_dir = tmp_path / "custom"
    fallback_dir = tmp_path / "fallback"
    monkeypatch.setenv("STOCKAGENT_LOCAL_DATA_DIR", custom_dir.as_posix())
    monkeypatch.setenv("STOCKAGENT_USE_FALLBACK_DATA_DIRS", "false")
    monkeypatch.setattr(market_data_module, "FALLBACK_DATA_DIRS", [fallback_dir])

    resolved = market_data_module.resolve_local_data_dirs()

    assert resolved == [custom_dir]


def test_package_root_exports_shared_market_data_helpers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STOCKAGENT_LOCAL_DATA_DIR", tmp_path.as_posix())
    monkeypatch.setenv("STOCKAGENT_USE_FALLBACK_DATA_DIRS", "false")

    assert stateful_agentsimulator.normalize_market_symbol("brk/b") == "BRK-B"
    assert stateful_agentsimulator.default_local_data_dir() == tmp_path
    assert stateful_agentsimulator.default_use_fallback_data_dirs() is False
    assert stateful_agentsimulator.resolve_local_data_dirs() == [tmp_path]


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
    expected_notional = max(
        DEFAULT_MAX_NEW_POSITION_NOTIONAL_FLOOR,
        account_payload["equity"] * DEFAULT_MAX_NEW_POSITION_EQUITY_FRACTION,
    )
    assert f"${expected_notional:,.0f}" in prompt
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
    instructions_schema = schema["properties"]["instructions"]["items"]
    required_fields = set(instructions_schema.get("required", []))
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
    assert set(schema.get("required", [])) >= {"target_date", "instructions"}
