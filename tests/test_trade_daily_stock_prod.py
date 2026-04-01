from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType

import pandas as pd
import pytest
import torch

import trade_daily_stock_prod as daily_stock
from pufferlib_market.inference import Policy
from src.trading_server.server import TradingServerEngine


def _write_daily_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_policy_checkpoint(
    path: Path,
    *,
    num_symbols: int,
    num_actions: int,
    hot_action: int,
) -> None:
    obs_size = num_symbols * 16 + 5 + num_symbols
    model = Policy(obs_size, num_actions, hidden=32, num_blocks=1)
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        model.actor[2].bias[hot_action] = 10.0
    torch.save(
        {
            "model": model.state_dict(),
            "config": {"hidden_size": 32, "num_blocks": 1},
            "action_allocation_bins": 1,
            "action_level_bins": 1,
            "action_max_offset_bps": 0.0,
        },
        path,
    )


def test_load_local_daily_frames_aligns_common_dates(tmp_path: Path) -> None:
    _write_daily_csv(
        tmp_path / "AAPL.csv",
        [
            {"timestamp": "2026-01-01T00:00:00Z", "open": 10, "high": 11, "low": 9, "close": 10.5, "volume": 100},
            {"timestamp": "2026-01-02T00:00:00Z", "open": 11, "high": 12, "low": 10, "close": 11.5, "volume": 110},
            {"timestamp": "2026-01-03T00:00:00Z", "open": 12, "high": 13, "low": 11, "close": 12.5, "volume": 120},
        ],
    )
    _write_daily_csv(
        tmp_path / "MSFT.csv",
        [
            {"timestamp": "2026-01-02T00:00:00Z", "open": 21, "high": 22, "low": 20, "close": 21.5, "volume": 210},
            {"timestamp": "2026-01-03T00:00:00Z", "open": 22, "high": 23, "low": 21, "close": 22.5, "volume": 220},
            {"timestamp": "2026-01-04T00:00:00Z", "open": 23, "high": 24, "low": 22, "close": 23.5, "volume": 230},
        ],
    )

    frames = daily_stock.load_local_daily_frames(["AAPL", "MSFT"], data_dir=str(tmp_path), min_days=2)

    assert list(frames) == ["AAPL", "MSFT"]
    assert frames["AAPL"]["timestamp"].dt.strftime("%Y-%m-%d").tolist() == ["2026-01-02", "2026-01-03"]
    assert frames["MSFT"]["timestamp"].dt.strftime("%Y-%m-%d").tolist() == ["2026-01-02", "2026-01-03"]


def test_load_local_daily_frames_rejects_unsafe_symbol(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"Unsupported symbol: \.\./AAPL"):
        daily_stock.load_local_daily_frames(["../AAPL"], data_dir=str(tmp_path), min_days=1)


def test_frames_from_alpaca_bars_handles_single_symbol_without_multiindex() -> None:
    bars_df = pd.DataFrame(
        [
            {"timestamp": "2026-01-01T00:00:00Z", "open": 10, "high": 11, "low": 9, "close": 10.5, "volume": 100},
            {"timestamp": "2026-01-02T00:00:00Z", "open": 11, "high": 12, "low": 10, "close": 11.5, "volume": 110},
        ]
    )

    frames = daily_stock._frames_from_alpaca_bars(
        symbols=["AAPL"],
        bars_df=bars_df,
        now=datetime(2026, 1, 3, tzinfo=timezone.utc),
    )

    assert list(frames) == ["AAPL"]
    assert len(frames["AAPL"]) == 2


def test_load_alpaca_daily_frames_recovers_missing_symbol_with_single_symbol_retry(monkeypatch) -> None:
    fake_alpaca_data_enums = ModuleType("alpaca.data.enums")
    fake_alpaca_data_enums.DataFeed = SimpleNamespace(IEX="iex", SIP="sip")
    monkeypatch.setitem(sys.modules, "alpaca.data.enums", fake_alpaca_data_enums)

    aapl = daily_stock._normalize_daily_frame(
        pd.DataFrame(
            [
                {"timestamp": "2026-01-01T00:00:00Z", "open": 10, "high": 11, "low": 9, "close": 10.5, "volume": 100},
                {"timestamp": "2026-01-02T00:00:00Z", "open": 11, "high": 12, "low": 10, "close": 11.5, "volume": 110},
            ]
        )
    )
    msft = daily_stock._normalize_daily_frame(
        pd.DataFrame(
            [
                {"timestamp": "2026-01-01T00:00:00Z", "open": 20, "high": 21, "low": 19, "close": 20.5, "volume": 200},
                {"timestamp": "2026-01-02T00:00:00Z", "open": 21, "high": 22, "low": 20, "close": 21.5, "volume": 210},
            ]
        )
    )

    def _fake_request(*, symbols, feed, **_kwargs):
        key = (tuple(symbols), feed)
        if key == (("AAPL", "MSFT"), "iex"):
            return {"AAPL": aapl}
        if key == (("MSFT",), "iex"):
            return {"MSFT": msft}
        return {}

    monkeypatch.setattr(daily_stock, "_request_alpaca_daily_frames", _fake_request)

    frames = daily_stock.load_alpaca_daily_frames(
        ["AAPL", "MSFT"],
        paper=True,
        min_days=2,
        data_client=object(),
        now=datetime(2026, 1, 3, tzinfo=timezone.utc),
    )

    assert list(frames) == ["AAPL", "MSFT"]
    assert len(frames["AAPL"]) == 2
    assert len(frames["MSFT"]) == 2


def test_should_run_today_only_after_market_open_window() -> None:
    now = datetime(2026, 3, 16, 13, 40, tzinfo=timezone.utc)  # 09:40 ET

    assert daily_stock.should_run_today(now=now, is_market_open=True, last_run_date=None) is True
    assert daily_stock.should_run_today(now=now, is_market_open=False, last_run_date=None) is False
    assert daily_stock.should_run_today(
        now=now,
        is_market_open=True,
        last_run_date="2026-03-16",
    ) is False


def test_compute_target_qty_respects_buying_power_cap() -> None:
    account = SimpleNamespace(portfolio_value=10_000.0, buying_power=2_000.0)

    qty = daily_stock.compute_target_qty(account=account, price=100.0, allocation_pct=50.0)

    assert qty == 19.0


def test_build_signal_long_only_ensemble_masks_short_actions(tmp_path: Path) -> None:
    rows_aapl = []
    rows_msft = []
    base = pd.Timestamp("2025-09-01T00:00:00Z")
    for idx in range(130):
        ts = base + pd.Timedelta(days=idx)
        rows_aapl.append(
            {
                "timestamp": ts.isoformat(),
                "open": 100 + idx,
                "high": 101 + idx,
                "low": 99 + idx,
                "close": 100.5 + idx,
                "volume": 1_000 + idx,
            }
        )
        rows_msft.append(
            {
                "timestamp": ts.isoformat(),
                "open": 200 + idx,
                "high": 201 + idx,
                "low": 199 + idx,
                "close": 200.5 + idx,
                "volume": 2_000 + idx,
            }
        )

    frames = {
        "AAPL": pd.DataFrame(rows_aapl),
        "MSFT": pd.DataFrame(rows_msft),
    }
    primary_ckpt = tmp_path / "primary.pt"
    extra_ckpt = tmp_path / "extra.pt"
    _write_policy_checkpoint(primary_ckpt, num_symbols=2, num_actions=5, hot_action=4)
    _write_policy_checkpoint(extra_ckpt, num_symbols=2, num_actions=5, hot_action=4)

    signal, prices = daily_stock.build_signal(
        str(primary_ckpt),
        frames,
        extra_checkpoints=[str(extra_ckpt)],
    )

    assert signal.action == "flat"
    assert signal.symbol is None
    assert signal.direction is None
    assert prices == {"AAPL": 229.5, "MSFT": 329.5}


def test_append_signal_log_writes_jsonl(tmp_path: Path) -> None:
    log_path = tmp_path / "daily_stock_rl_signals.jsonl"

    daily_stock.append_signal_log({"symbol": "AAPL", "signal": "flat"}, path=log_path)

    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert rows == [{"signal": "flat", "symbol": "AAPL"}]


def test_main_uses_default_resolved_ensemble_checkpoints(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_run_once(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(daily_stock, "run_once", _fake_run_once)

    daily_stock.main(["--dry-run", "--data-source", "local", "--symbols", "AAPL"])

    resolved_defaults = [
        str((daily_stock.REPO / checkpoint).resolve())
        for checkpoint in daily_stock.DEFAULT_EXTRA_CHECKPOINTS
    ]

    assert captured["checkpoint"] == str((daily_stock.REPO / daily_stock.DEFAULT_CHECKPOINT).resolve())
    assert captured["extra_checkpoints"] == resolved_defaults
    assert len(resolved_defaults) == len(set(resolved_defaults))
    assert resolved_defaults[-1].endswith("s6758.pt")


def test_cli_runtime_config_exposes_derived_fields(monkeypatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    extra = tmp_path / "extra.pt"
    checkpoint.write_text("stub", encoding="utf-8")
    monkeypatch.setenv("TRADING_SERVER_URL", "http://server.internal:9000")

    config = daily_stock.CliRuntimeConfig(
        paper=False,
        symbols=["AAPL"],
        checkpoint=str(checkpoint),
        extra_checkpoints=[str(extra)],
        data_dir=str(tmp_path),
        data_source="local",
        allocation_pct=12.5,
        execution_backend="trading_server",
        server_account="acct-live",
        server_bot_id="bot-live",
        server_url=None,
        dry_run=True,
        backtest=False,
        backtest_days=60,
        backtest_starting_cash=25_000.0,
        daemon=False,
        compare_server_parity=False,
    )

    payload = config.to_runtime_payload()

    assert config.ensemble_enabled is True
    assert config.ensemble_size == 2
    assert config.account_mode == "live"
    assert config.run_mode == "once"
    assert config.checkpoint_paths == [str(checkpoint), str(extra)]
    assert config.missing_checkpoint_paths == [str(extra)]
    assert config.checkpoints_exist is False
    assert config.resolved_server_url == "http://server.internal:9000"
    assert payload["ensemble_size"] == 2
    assert payload["backtest_starting_cash"] == 25_000.0
    assert payload["missing_checkpoints"] == [str(extra)]


def test_parse_args_rejects_conflicting_account_mode_flags() -> None:
    with pytest.raises(SystemExit):
        daily_stock.parse_args(["--paper", "--live"])


@pytest.mark.parametrize(
    "argv",
    [
        ["--once", "--daemon"],
        ["--once", "--backtest"],
        ["--daemon", "--backtest"],
    ],
)
def test_parse_args_rejects_conflicting_run_mode_flags(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        daily_stock.parse_args(argv)


def test_main_print_config_outputs_resolved_runtime_config(capsys) -> None:
    checkpoint_path = (daily_stock.REPO / daily_stock.DEFAULT_CHECKPOINT).resolve()
    daily_stock.main(
        [
            "--print-config",
            "--live",
            "--no-ensemble",
            "--symbols",
            "aapl",
            "msft",
            "--checkpoint",
            daily_stock.DEFAULT_CHECKPOINT,
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert payload["account_mode"] == "live"
    assert payload["run_mode"] == "once"
    assert payload["ensemble_enabled"] is False
    assert payload["ensemble_size"] == 1
    assert payload["symbol_count"] == 2
    assert payload["symbols"] == ["AAPL", "MSFT"]
    assert payload["summary"] == "live once via alpaca using alpaca data on 2 symbols with 1 checkpoint"
    assert payload["extra_checkpoints"] is None
    assert payload["backtest_starting_cash"] == daily_stock.DEFAULT_BACKTEST_STARTING_CASH
    assert payload["checkpoint"] == str(checkpoint_path)
    assert "--once" in payload["run_command_preview"]
    assert "--live" in payload["run_command_preview"]
    assert "--dry-run" in payload["safe_command_preview"]
    assert payload["checkpoints_exist"] is checkpoint_path.exists()
    assert payload["missing_checkpoints"] == ([] if checkpoint_path.exists() else [str(checkpoint_path)])


def test_main_print_config_normalizes_duplicate_and_blank_symbols(capsys) -> None:
    daily_stock.main(
        [
            "--print-config",
            "--no-ensemble",
            "--symbols",
            "aapl",
            "AAPL",
            "   ",
            "msft",
            "aapl",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert payload["symbols"] == ["AAPL", "MSFT"]
    assert payload["symbol_count"] == 2
    assert payload["removed_duplicate_symbols"] == ["AAPL"]
    assert payload["ignored_symbol_inputs"] == ["<blank>"]
    assert "--symbols AAPL MSFT" in payload["run_command_preview"]


def test_main_logs_runtime_config(caplog, monkeypatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("stub", encoding="utf-8")

    captured: dict[str, object] = {}

    def _fake_run_once(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(daily_stock, "run_once", _fake_run_once)

    with caplog.at_level(logging.INFO, logger="daily_stock_rl"):
        daily_stock.main(
            [
                "--dry-run",
                "--no-ensemble",
                "--checkpoint",
                str(checkpoint),
                "--symbols",
                "AAPL",
            ]
        )

    runtime_records = [
        record.getMessage()
        for record in caplog.records
        if record.name == "daily_stock_rl" and record.getMessage().startswith("Runtime config: ")
    ]
    assert runtime_records
    assert '"account_mode": "paper"' in runtime_records[-1]
    assert '"symbols": ["AAPL"]' in runtime_records[-1]
    assert captured["checkpoint"] == str(checkpoint)


def test_main_rejects_all_blank_symbols() -> None:
    with pytest.raises(ValueError, match="No valid symbols configured after normalization"):
        daily_stock.main(
            [
                "--print-config",
                "--no-ensemble",
                "--symbols",
                " ",
                "   ",
            ]
        )


def test_main_rejects_unsafe_symbol_input() -> None:
    with pytest.raises(ValueError, match=r"Unsupported symbol: \.\./AAPL"):
        daily_stock.main(
            [
                "--print-config",
                "--no-ensemble",
                "--symbols",
                "../AAPL",
            ]
        )


def test_main_print_config_allows_standard_share_class_symbols(capsys) -> None:
    daily_stock.main(
        [
            "--print-config",
            "--no-ensemble",
            "--symbols",
            "brk.b",
            "bf-b",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert payload["symbols"] == ["BRK.B", "BF-B"]


def test_build_trading_client_uses_explicit_paper_credentials(monkeypatch) -> None:
    captured: dict[str, object] = {}

    fake_alpaca = ModuleType("alpaca")
    fake_alpaca_trading = ModuleType("alpaca.trading")
    fake_alpaca_trading_client = ModuleType("alpaca.trading.client")

    class FakeTradingClient:
        def __init__(self, key_id, secret, *, paper):
            captured["key_id"] = key_id
            captured["secret"] = secret
            captured["paper"] = paper

    fake_alpaca_trading_client.TradingClient = FakeTradingClient
    monkeypatch.setitem(sys.modules, "alpaca", fake_alpaca)
    monkeypatch.setitem(sys.modules, "alpaca.trading", fake_alpaca_trading)
    monkeypatch.setitem(sys.modules, "alpaca.trading.client", fake_alpaca_trading_client)

    fake_env_real = ModuleType("env_real")
    fake_env_real.ALP_KEY_ID_PAPER = "paper-key"
    fake_env_real.ALP_SECRET_KEY_PAPER = "paper-secret"
    fake_env_real.ALP_KEY_ID_PROD = "live-key"
    fake_env_real.ALP_SECRET_KEY_PROD = "live-secret"
    monkeypatch.setitem(sys.modules, "env_real", fake_env_real)

    daily_stock.build_trading_client(paper=True)

    assert captured == {
        "key_id": "paper-key",
        "secret": "paper-secret",
        "paper": True,
    }


def test_build_data_client_uses_explicit_live_credentials(monkeypatch) -> None:
    captured: dict[str, object] = {}

    fake_alpaca = ModuleType("alpaca")
    fake_alpaca_data = ModuleType("alpaca.data")

    class FakeStockHistoricalDataClient:
        def __init__(self, key_id, secret):
            captured["key_id"] = key_id
            captured["secret"] = secret

    fake_alpaca_data.StockHistoricalDataClient = FakeStockHistoricalDataClient
    monkeypatch.setitem(sys.modules, "alpaca", fake_alpaca)
    monkeypatch.setitem(sys.modules, "alpaca.data", fake_alpaca_data)

    fake_env_real = ModuleType("env_real")
    fake_env_real.ALP_KEY_ID_PAPER = "paper-key"
    fake_env_real.ALP_SECRET_KEY_PAPER = "paper-secret"
    fake_env_real.ALP_KEY_ID_PROD = "live-key"
    fake_env_real.ALP_SECRET_KEY_PROD = "live-secret"
    monkeypatch.setitem(sys.modules, "env_real", fake_env_real)

    daily_stock.build_data_client(paper=False)

    assert captured == {
        "key_id": "live-key",
        "secret": "live-secret",
    }


def test_build_trading_client_raises_clear_error_when_env_real_is_unavailable(monkeypatch) -> None:
    import importlib

    real_import_module = importlib.import_module

    def _fake_import_module(name: str):
        if name == "env_real":
            raise ModuleNotFoundError("env_real missing")
        return real_import_module(name)

    monkeypatch.setattr("importlib.import_module", _fake_import_module)

    with pytest.raises(RuntimeError, match="Unable to load env_real for Alpaca credentials"):
        daily_stock.build_trading_client(paper=True)


def test_load_latest_quotes_rejects_unsafe_symbol_before_client_build(monkeypatch) -> None:
    def _unexpected_client(*, paper: bool):
        raise AssertionError(f"unexpected client build for paper={paper}")

    monkeypatch.setattr(daily_stock, "build_data_client", _unexpected_client)

    with pytest.raises(ValueError, match=r"Unsupported symbol: \.\./AAPL"):
        daily_stock.load_latest_quotes_with_source(
            ["../AAPL"],
            paper=True,
            fallback_prices={},
        )


def test_execute_signal_logs_execution_safety_gate_reason(caplog) -> None:
    class FakeClient:
        def get_all_positions(self):
            return []

    signal = SimpleNamespace(symbol="AAPL", direction="long")

    with caplog.at_level(logging.WARNING, logger="daily_stock_rl"):
        executed = daily_stock.execute_signal(
            signal,
            client=FakeClient(),
            quotes={"AAPL": 100.0},
            state=daily_stock.StrategyState(),
            symbols=["AAPL"],
            allocation_pct=12.5,
            dry_run=True,
            allow_open=False,
            allow_open_reason="quote_source=close_fallback",
        )

    assert executed is False
    assert any(
        "execution safety gate is active (quote_source=close_fallback)" in record.getMessage()
        for record in caplog.records
    )


def test_execute_signal_with_trading_server_logs_execution_safety_gate_reason(caplog) -> None:
    class FakeServerClient:
        def get_account(self):
            return {"positions": {}, "account": {"cash": 10000.0}}

    signal = SimpleNamespace(symbol="AAPL", direction="long")

    with caplog.at_level(logging.WARNING, logger="daily_stock_rl"):
        executed = daily_stock.execute_signal_with_trading_server(
            signal,
            server_client=FakeServerClient(),
            quotes={"AAPL": 100.0},
            state=daily_stock.StrategyState(),
            symbols=["AAPL"],
            allocation_pct=12.5,
            dry_run=True,
            allow_open=False,
            allow_open_reason="quote_source=close_fallback",
        )

    assert executed is False
    assert any(
        "execution safety gate is active (quote_source=close_fallback)" in record.getMessage()
        for record in caplog.records
    )


def test_main_check_config_reports_ready_for_local_setup(tmp_path: Path, capsys) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("stub", encoding="utf-8")
    _write_daily_csv(
        tmp_path / "AAPL.csv",
        [
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "open": 10,
                "high": 11,
                "low": 9,
                "close": 10.5,
                "volume": 100,
            }
        ],
    )

    daily_stock.main(
        [
            "--check-config",
            "--dry-run",
            "--data-source",
            "local",
            "--symbols",
            "AAPL",
            "--checkpoint",
            str(checkpoint),
            "--no-ensemble",
            "--data-dir",
            str(tmp_path),
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert payload["ready"] is True
    assert payload["errors"] == []
    assert payload["warnings"] == []
    assert payload["local_data_required"] is True
    assert payload["data_dir_exists"] is True
    assert payload["missing_local_symbol_files"] == []
    assert payload["summary"] == "paper once via alpaca using local data on 1 symbol with 1 checkpoint"
    assert "--dry-run" in payload["run_command_preview"]
    assert payload["run_command_preview"] == payload["safe_command_preview"]


def test_main_check_config_warns_when_symbol_inputs_are_normalized(tmp_path: Path, capsys) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("stub", encoding="utf-8")
    _write_daily_csv(
        tmp_path / "AAPL.csv",
        [
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "open": 10,
                "high": 11,
                "low": 9,
                "close": 10.5,
                "volume": 100,
            }
        ],
    )

    daily_stock.main(
        [
            "--check-config",
            "--dry-run",
            "--data-source",
            "local",
            "--symbols",
            "aapl",
            "AAPL",
            " ",
            "--checkpoint",
            str(checkpoint),
            "--no-ensemble",
            "--data-dir",
            str(tmp_path),
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert payload["symbols"] == ["AAPL"]
    assert payload["removed_duplicate_symbols"] == ["AAPL"]
    assert payload["ignored_symbol_inputs"] == ["<blank>"]
    assert "--symbols AAPL" in payload["run_command_preview"]
    assert any("Removed duplicate symbol input" in warning for warning in payload["warnings"])
    assert any("Ignored blank symbol input" in warning for warning in payload["warnings"])


def test_main_check_config_reports_missing_live_alpaca_credentials(monkeypatch, tmp_path: Path, capsys) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("stub", encoding="utf-8")

    fake_env_real = ModuleType("env_real")
    fake_env_real.ALP_KEY_ID_PAPER = "paper-key"
    fake_env_real.ALP_SECRET_KEY_PAPER = "paper-secret"
    fake_env_real.ALP_KEY_ID_PROD = "alpaca-live-key-placeholder"
    fake_env_real.ALP_SECRET_KEY_PROD = "alpaca-live-secret-placeholder"
    monkeypatch.setitem(sys.modules, "env_real", fake_env_real)

    with pytest.raises(SystemExit) as exc_info:
        daily_stock.main(
            [
                "--check-config",
                "--live",
                "--checkpoint",
                str(checkpoint),
                "--no-ensemble",
            ]
        )

    payload = json.loads(capsys.readouterr().out)

    assert exc_info.value.code == 1
    assert payload["ready"] is False
    assert payload["alpaca_required"] is True
    assert payload["alpaca_credentials_configured"] is False
    assert payload["alpaca_missing_env"] == ["ALP_KEY_ID_PROD", "ALP_SECRET_KEY_PROD"]
    assert any("Missing Alpaca credential env var(s) for live mode" in error for error in payload["errors"])


def test_main_check_config_reports_env_real_import_failure(monkeypatch, tmp_path: Path, capsys) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("stub", encoding="utf-8")

    import importlib

    real_import_module = importlib.import_module

    def _fake_import_module(name: str):
        if name == "env_real":
            raise ModuleNotFoundError("env_real missing")
        return real_import_module(name)

    monkeypatch.setattr("importlib.import_module", _fake_import_module)

    with pytest.raises(SystemExit) as exc_info:
        daily_stock.main(
            [
                "--check-config",
                "--checkpoint",
                str(checkpoint),
                "--no-ensemble",
            ]
        )

    payload = json.loads(capsys.readouterr().out)

    assert exc_info.value.code == 1
    assert payload["ready"] is False
    assert payload["alpaca_required"] is True
    assert payload["alpaca_credentials_configured"] is False
    assert payload["alpaca_missing_env"] == []
    assert "Unable to load env_real for Alpaca credentials" in payload["alpaca_credentials_error"]
    assert any("Unable to load env_real for Alpaca credentials" in error for error in payload["errors"])


def test_main_check_config_warns_when_live_trading_server_uses_paper_default_account(
    tmp_path: Path,
    capsys,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("stub", encoding="utf-8")
    _write_daily_csv(
        tmp_path / "AAPL.csv",
        [
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "open": 10,
                "high": 11,
                "low": 9,
                "close": 10.5,
                "volume": 100,
            }
        ],
    )

    daily_stock.main(
        [
            "--check-config",
            "--live",
            "--execution-backend",
            "trading_server",
            "--data-source",
            "local",
            "--symbols",
            "AAPL",
            "--checkpoint",
            str(checkpoint),
            "--no-ensemble",
            "--data-dir",
            str(tmp_path),
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert payload["ready"] is True
    assert payload["trading_server_required"] is True
    assert payload["resolved_server_url"] == "http://127.0.0.1:8050"
    assert any("paper default server account" in warning for warning in payload["warnings"])


def test_main_fails_fast_when_checkpoint_is_missing(monkeypatch, tmp_path: Path) -> None:
    called = False

    def _fake_run_once(**kwargs):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(daily_stock, "run_once", _fake_run_once)

    missing = tmp_path / "missing.pt"

    with pytest.raises(FileNotFoundError, match="Missing checkpoint file"):
        daily_stock.main(
            [
                "--dry-run",
                "--data-source",
                "local",
                "--symbols",
                "AAPL",
                "--checkpoint",
                str(missing),
                "--no-ensemble",
            ]
        )

    assert called is False


def test_main_rejects_compare_server_parity_without_backtest(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("stub", encoding="utf-8")

    with pytest.raises(ValueError, match="--compare-server-parity requires --backtest"):
        daily_stock.main(
            [
                "--dry-run",
                "--data-source",
                "local",
                "--symbols",
                "AAPL",
                "--checkpoint",
                str(checkpoint),
                "--no-ensemble",
                "--compare-server-parity",
            ]
        )


def test_main_rejects_non_positive_backtest_days(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("stub", encoding="utf-8")

    with pytest.raises(ValueError, match="--backtest-days must be at least 1"):
        daily_stock.main(
            [
                "--backtest",
                "--backtest-days",
                "0",
                "--data-source",
                "local",
                "--symbols",
                "AAPL",
                "--checkpoint",
                str(checkpoint),
                "--no-ensemble",
            ]
        )


def test_main_rejects_non_positive_backtest_starting_cash(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("stub", encoding="utf-8")

    with pytest.raises(ValueError, match="--backtest-starting-cash must be positive"):
        daily_stock.main(
            [
                "--backtest",
                "--backtest-starting-cash",
                "0",
                "--data-source",
                "local",
                "--symbols",
                "AAPL",
                "--checkpoint",
                str(checkpoint),
                "--no-ensemble",
            ]
        )


def test_main_rejects_live_backtest_before_live_safety_checks(monkeypatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("stub", encoding="utf-8")

    def _unexpected_live_guard(*_args, **_kwargs):
        raise AssertionError("live safety guard should not run for backtests")

    monkeypatch.setattr(daily_stock, "require_explicit_live_trading_enable", _unexpected_live_guard)
    monkeypatch.setattr(daily_stock, "acquire_alpaca_account_lock", _unexpected_live_guard)

    with pytest.raises(ValueError, match="--backtest is local-only; omit --live"):
        daily_stock.main(
            [
                "--backtest",
                "--live",
                "--data-source",
                "local",
                "--symbols",
                "AAPL",
                "--checkpoint",
                str(checkpoint),
                "--no-ensemble",
            ]
        )


def test_main_passes_backtest_starting_cash_to_run_backtest(monkeypatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("stub", encoding="utf-8")
    captured: dict[str, object] = {}

    def _fake_run_backtest(**kwargs):
        captured.update(kwargs)
        return {"total_return": 0.0}

    monkeypatch.setattr(daily_stock, "run_backtest", _fake_run_backtest)

    daily_stock.main(
        [
            "--backtest",
            "--backtest-starting-cash",
            "25000",
            "--data-source",
            "local",
            "--symbols",
            "AAPL",
            "--checkpoint",
            str(checkpoint),
            "--no-ensemble",
        ]
    )

    assert captured["starting_cash"] == 25_000.0


@pytest.mark.parametrize(
    "runner",
    [daily_stock.run_backtest, daily_stock.run_backtest_via_trading_server],
)
def test_backtest_helpers_reject_non_positive_starting_cash(runner) -> None:
    with pytest.raises(ValueError, match="starting_cash must be positive"):
        runner(
            checkpoint="unused.pt",
            symbols=["AAPL"],
            data_dir="unused",
            days=20,
            starting_cash=0.0,
        )


class _FakeClient:
    def __init__(self, positions: list[object], *, portfolio_value: float = 10_000.0, buying_power: float = 10_000.0):
        self._positions = positions
        self._account = SimpleNamespace(portfolio_value=portfolio_value, buying_power=buying_power)

    def get_all_positions(self):
        return list(self._positions)

    def get_account(self):
        return self._account


def _write_server_registry(path: Path, *, account: str, bot_id: str, symbols: list[str], starting_cash: float = 10_000.0) -> None:
    path.write_text(
        json.dumps(
            {
                "accounts": {
                    account: {
                        "mode": "paper",
                        "allowed_bot_id": bot_id,
                        "starting_cash": starting_cash,
                        "sell_loss_cooldown_seconds": 0,
                        "min_sell_markup_pct": 0.0,
                        "symbols": symbols,
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def test_execute_signal_closes_managed_position_then_opens_new_one(monkeypatch) -> None:
    orders: list[tuple[str, float, str]] = []

    def _fake_submit_market_order(client, *, symbol: str, qty: float, side: str):
        orders.append((symbol, qty, side))
        return SimpleNamespace(id=f"{symbol}-{side}")

    def _fake_submit_limit_order(client, *, symbol: str, qty: float, side: str, limit_price: float):
        orders.append((symbol, qty, side))
        return SimpleNamespace(id=f"{symbol}-{side}")

    monkeypatch.setattr(daily_stock, "submit_market_order", _fake_submit_market_order)
    monkeypatch.setattr(daily_stock, "submit_limit_order", _fake_submit_limit_order)

    client = _FakeClient([SimpleNamespace(symbol="AAPL", qty="10", side="long")])
    state = daily_stock.StrategyState(active_symbol="AAPL", active_qty=10.0)
    signal = SimpleNamespace(symbol="MSFT", direction="long", action="long_MSFT")

    changed = daily_stock.execute_signal(
        signal,
        client=client,
        quotes={"AAPL": 100.0, "MSFT": 50.0},
        state=state,
        symbols=["AAPL", "MSFT"],
        allocation_pct=25.0,
        dry_run=False,
    )

    assert changed is True
    assert len(orders) == 2
    assert orders[0] == ("AAPL", 10.0, "sell")
    assert orders[1][0] == "MSFT"
    assert orders[1][2] == "buy"
    assert state.active_symbol == "MSFT"


def test_execute_signal_refuses_to_trade_with_unmanaged_position(monkeypatch) -> None:
    called = False

    def _fake_submit_market_order(client, *, symbol: str, qty: float, side: str):
        nonlocal called
        called = True
        return SimpleNamespace(id="unexpected")

    monkeypatch.setattr(daily_stock, "submit_market_order", _fake_submit_market_order)

    client = _FakeClient([SimpleNamespace(symbol="AAPL", qty="5", side="long")])
    state = daily_stock.StrategyState(active_symbol=None, active_qty=0.0)
    signal = SimpleNamespace(symbol="MSFT", direction="long", action="long_MSFT")

    changed = daily_stock.execute_signal(
        signal,
        client=client,
        quotes={"MSFT": 50.0},
        state=state,
        symbols=["AAPL", "MSFT"],
        allocation_pct=25.0,
        dry_run=False,
    )

    assert changed is False
    assert called is False
    assert state.active_symbol is None


def test_execute_signal_with_open_gate_only_closes_existing_position(monkeypatch) -> None:
    orders: list[tuple[str, float, str]] = []

    def _fake_submit_market_order(client, *, symbol: str, qty: float, side: str):
        orders.append((symbol, qty, side))
        return SimpleNamespace(id=f"{symbol}-{side}")

    def _fake_submit_limit_order(client, *, symbol: str, qty: float, side: str, limit_price: float):
        orders.append((symbol, qty, side))
        return SimpleNamespace(id=f"{symbol}-{side}")

    monkeypatch.setattr(daily_stock, "submit_market_order", _fake_submit_market_order)
    monkeypatch.setattr(daily_stock, "submit_limit_order", _fake_submit_limit_order)

    client = _FakeClient([SimpleNamespace(symbol="AAPL", qty="10", side="long")])
    state = daily_stock.StrategyState(active_symbol="AAPL", active_qty=10.0)
    signal = SimpleNamespace(symbol="MSFT", direction="long", action="long_MSFT")

    changed = daily_stock.execute_signal(
        signal,
        client=client,
        quotes={"AAPL": 100.0, "MSFT": 50.0},
        state=state,
        symbols=["AAPL", "MSFT"],
        allocation_pct=25.0,
        dry_run=False,
        allow_open=False,
    )

    assert changed is True
    assert len(orders) == 1
    assert orders[0] == ("AAPL", 10.0, "sell")
    assert state.active_symbol is None
    assert state.active_qty == 0.0


def test_execute_signal_with_trading_server_closes_managed_then_opens_new(tmp_path: Path) -> None:
    current_now = datetime(2026, 3, 16, 14, 0, tzinfo=timezone.utc)
    quotes = {
        "AAPL": {"symbol": "AAPL", "bid_price": 100.0, "ask_price": 100.0, "last_price": 100.0, "as_of": current_now.isoformat()},
        "MSFT": {"symbol": "MSFT", "bid_price": 50.0, "ask_price": 50.0, "last_price": 50.0, "as_of": current_now.isoformat()},
    }
    registry = tmp_path / "registry.json"
    _write_server_registry(registry, account="paper_sortino_daily", bot_id="daily_stock_sortino_v1", symbols=["AAPL", "MSFT"])
    engine = TradingServerEngine(
        registry_path=registry,
        state_dir=tmp_path / "state",
        quote_provider=lambda symbol: quotes[str(symbol).upper()],
        now_fn=lambda: current_now,
    )
    client = daily_stock.InMemoryTradingServerClient(
        engine=engine,
        account="paper_sortino_daily",
        bot_id="daily_stock_sortino_v1",
        execution_mode="paper",
        session_id="test-session",
    )
    client.claim_writer()
    client.refresh_prices(symbols=["AAPL", "MSFT"])
    client.submit_limit_order(symbol="AAPL", side="buy", qty=10.0, limit_price=100.0)

    state = daily_stock.StrategyState(active_symbol="AAPL", active_qty=10.0, entry_price=100.0, entry_date="2026-03-16")
    signal = SimpleNamespace(symbol="MSFT", direction="long", action="long_MSFT")

    changed = daily_stock.execute_signal_with_trading_server(
        signal,
        server_client=client,
        quotes={"AAPL": 100.0, "MSFT": 50.0},
        state=state,
        symbols=["AAPL", "MSFT"],
        allocation_pct=25.0,
        dry_run=False,
        now=current_now,
    )

    assert changed is True
    snapshot = client.get_account()
    assert "AAPL" not in snapshot["positions"]
    assert snapshot["positions"]["MSFT"]["qty"] == 50.0
    assert state.active_symbol == "MSFT"
    assert state.active_qty == 50.0

def test_adopt_existing_position_populates_state() -> None:
    state = daily_stock.StrategyState()
    adopted = daily_stock.adopt_existing_position(
        state=state,
        live_positions={
            "NVDA": SimpleNamespace(symbol="NVDA", qty="3", side="long", avg_entry_price="121.5"),
        },
        now=datetime(2026, 3, 16, 13, 40, tzinfo=timezone.utc),
    )

    assert adopted is True
    assert state.active_symbol == "NVDA"
    assert state.active_qty == 3.0
    assert state.entry_price == 121.5
    assert state.entry_date == "2026-03-16"


def test_load_latest_quotes_falls_back_to_close_prices() -> None:
    class _BrokenQuoteClient:
        def get_stock_latest_quote(self, request):
            raise RuntimeError("quotes unavailable")

    prices = daily_stock.load_latest_quotes(
        ["AAPL", "MSFT"],
        paper=True,
        fallback_prices={"AAPL": 101.0, "MSFT": 202.0},
        data_client=_BrokenQuoteClient(),
    )

    assert prices == {"AAPL": 101.0, "MSFT": 202.0}

    prices_with_source, source, source_by_symbol = daily_stock.load_latest_quotes_with_source(
        ["AAPL", "MSFT"],
        paper=True,
        fallback_prices={"AAPL": 101.0, "MSFT": 202.0},
        data_client=_BrokenQuoteClient(),
    )
    assert prices_with_source == {"AAPL": 101.0, "MSFT": 202.0}
    assert source == "close_fallback"
    assert source_by_symbol == {"AAPL": "close_fallback", "MSFT": "close_fallback"}


def test_load_latest_quotes_tracks_partial_symbol_fallbacks() -> None:
    class _QuoteClient:
        def get_stock_latest_quote(self, request):
            return {
                "AAPL": SimpleNamespace(ask_price=150.0, bid_price=149.5),
                "MSFT": SimpleNamespace(ask_price=0.0, bid_price=0.0),
            }

    prices, source, source_by_symbol = daily_stock.load_latest_quotes_with_source(
        ["AAPL", "MSFT"],
        paper=True,
        fallback_prices={"AAPL": 101.0, "MSFT": 202.0},
        data_client=_QuoteClient(),
    )

    assert prices == {"AAPL": 150.0, "MSFT": 202.0}
    assert source == "mixed_fallback"
    assert source_by_symbol == {"AAPL": "alpaca", "MSFT": "close_fallback"}


def test_bars_are_fresh_uses_age_gate() -> None:
    latest_bar = pd.Timestamp("2026-03-14T00:00:00Z")

    assert daily_stock.bars_are_fresh(
        latest_bar=latest_bar,
        now=datetime(2026, 3, 16, 13, 40, tzinfo=timezone.utc),
        max_age_days=5,
    ) is True
    assert daily_stock.bars_are_fresh(
        latest_bar=latest_bar,
        now=datetime(2026, 3, 25, 13, 40, tzinfo=timezone.utc),
        max_age_days=5,
    ) is False


def test_run_once_falls_back_to_local_daily_frames(monkeypatch, tmp_path: Path) -> None:
    rows = []
    for idx in range(130):
        ts = pd.Timestamp("2025-09-01T00:00:00Z") + pd.Timedelta(days=idx)
        rows.append(
            {
                "timestamp": ts.isoformat(),
                "open": 100 + idx,
                "high": 101 + idx,
                "low": 99 + idx,
                "close": 100.5 + idx,
                "volume": 1_000 + idx,
            }
        )
    _write_daily_csv(tmp_path / "AAPL.csv", rows)
    _write_daily_csv(tmp_path / "MSFT.csv", rows)

    monkeypatch.setattr(
        daily_stock,
        "load_alpaca_daily_frames",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("alpaca bars unavailable")),
    )
    monkeypatch.setattr(
        daily_stock,
        "build_trading_client",
        lambda paper: SimpleNamespace(
            get_all_positions=lambda: [],
            get_account=lambda: SimpleNamespace(cash=10_000.0, buying_power=10_000.0, portfolio_value=10_000.0),
        ),
    )
    monkeypatch.setattr(daily_stock, "build_data_client", lambda paper: object())
    monkeypatch.setattr(daily_stock, "execute_signal", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        daily_stock,
        "build_signal",
        lambda checkpoint, frames, portfolio=daily_stock.PortfolioContext(), device="cpu", extra_checkpoints=None: (
            SimpleNamespace(
                action="long_AAPL",
                symbol="AAPL",
                direction="long",
                confidence=0.75,
                value_estimate=1.25,
            ),
            {"AAPL": 123.0, "MSFT": 234.0},
        ),
    )
    monkeypatch.setattr(
        daily_stock,
        "load_latest_quotes_with_source",
        lambda symbols, paper, fallback_prices, data_client=None: (
            {"AAPL": 124.0, "MSFT": 235.0},
            "alpaca",
            {"AAPL": "alpaca", "MSFT": "alpaca"},
        ),
    )

    state_path = tmp_path / "state.json"
    payload = daily_stock.run_once(
        checkpoint="dummy.ckpt",
        symbols=["AAPL", "MSFT"],
        paper=True,
        allocation_pct=25.0,
        dry_run=True,
        data_source="alpaca",
        data_dir=str(tmp_path),
        state_path=state_path,
    )

    assert payload["bar_data_source"] == "local_fallback"
    assert payload["quote_data_source"] == "alpaca"
    assert payload["symbol"] == "AAPL"
    assert state_path.exists() is False


def test_run_once_dry_run_does_not_advance_state(monkeypatch, tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text(
        '{"active_symbol": null, "active_qty": 0.0, "last_run_date": null, "last_signal_action": null, "last_signal_timestamp": null, "last_order_id": null}'
    )

    monkeypatch.setattr(
        daily_stock,
        "build_signal",
        lambda checkpoint, frames, portfolio=daily_stock.PortfolioContext(), device="cpu", extra_checkpoints=None: (
            SimpleNamespace(
                action="flat",
                symbol=None,
                direction=None,
                confidence=0.5,
                value_estimate=0.0,
            ),
            {"AAPL": 123.0, "MSFT": 234.0},
        ),
    )
    monkeypatch.setattr(
        daily_stock,
        "load_local_daily_frames",
        lambda symbols, data_dir, min_days=120: {
            "AAPL": pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(["2026-03-13T00:00:00Z"]),
                    "open": [1.0],
                    "high": [1.0],
                    "low": [1.0],
                    "close": [1.0],
                    "volume": [1.0],
                }
            ),
            "MSFT": pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(["2026-03-13T00:00:00Z"]),
                    "open": [1.0],
                    "high": [1.0],
                    "low": [1.0],
                    "close": [1.0],
                    "volume": [1.0],
                }
            ),
        },
    )

    before = state_path.read_text()
    daily_stock.run_once(
        checkpoint="dummy.ckpt",
        symbols=["AAPL", "MSFT"],
        paper=True,
        allocation_pct=25.0,
        dry_run=True,
        data_source="local",
        data_dir=str(tmp_path),
        state_path=state_path,
    )
    after = state_path.read_text()

    assert after == before


def test_run_once_market_closed_does_not_advance_state(monkeypatch, tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text(
        '{"active_symbol": null, "active_qty": 0.0, "last_run_date": null, "last_signal_action": null, "last_signal_timestamp": null, "last_order_id": null}'
    )

    monkeypatch.setattr(
        daily_stock,
        "load_inference_frames",
        lambda symbols, paper, data_dir, now, data_client=None: (
            {
                "AAPL": pd.DataFrame(
                    {
                        "timestamp": pd.to_datetime(["2026-03-13T00:00:00Z"]),
                        "open": [1.0],
                        "high": [1.0],
                        "low": [1.0],
                        "close": [1.0],
                        "volume": [1.0],
                    }
                ),
                "MSFT": pd.DataFrame(
                    {
                        "timestamp": pd.to_datetime(["2026-03-13T00:00:00Z"]),
                        "open": [1.0],
                        "high": [1.0],
                        "low": [1.0],
                        "close": [1.0],
                        "volume": [1.0],
                    }
                ),
            },
            "alpaca",
        ),
    )
    monkeypatch.setattr(
        daily_stock,
        "build_trading_client",
        lambda paper: SimpleNamespace(
            get_all_positions=lambda: [],
            get_account=lambda: SimpleNamespace(cash=10_000.0, buying_power=10_000.0, portfolio_value=10_000.0),
            get_clock=lambda: SimpleNamespace(is_open=False),
        ),
    )
    monkeypatch.setattr(daily_stock, "build_data_client", lambda paper: object())
    monkeypatch.setattr(
        daily_stock,
        "load_latest_quotes_with_source",
        lambda symbols, paper, fallback_prices, data_client=None: (
            {"AAPL": 1.0, "MSFT": 1.0},
            "alpaca",
            {"AAPL": "alpaca", "MSFT": "alpaca"},
        ),
    )
    monkeypatch.setattr(
        daily_stock,
        "build_signal",
        lambda checkpoint, frames, portfolio=daily_stock.PortfolioContext(), device="cpu", extra_checkpoints=None: (
            SimpleNamespace(
                action="long_AAPL",
                symbol="AAPL",
                direction="long",
                confidence=0.5,
                value_estimate=0.0,
            ),
            {"AAPL": 1.0, "MSFT": 1.0},
        ),
    )
    execute_calls: list[object] = []
    monkeypatch.setattr(
        daily_stock,
        "execute_signal",
        lambda *args, **kwargs: execute_calls.append((args, kwargs)) or False,
    )

    before = state_path.read_text()
    daily_stock.run_once(
        checkpoint="dummy.ckpt",
        symbols=["AAPL", "MSFT"],
        paper=True,
        allocation_pct=25.0,
        dry_run=False,
        data_source="alpaca",
        data_dir=str(tmp_path),
        state_path=state_path,
    )
    after = state_path.read_text()

    assert execute_calls == []
    assert after == before


def test_run_backtest_via_trading_server_matches_legacy_single_symbol(tmp_path: Path) -> None:
    rows = []
    base = pd.Timestamp("2025-09-01T00:00:00Z")
    for idx in range(140):
        ts = base + pd.Timedelta(days=idx)
        rows.append(
            {
                "timestamp": ts.isoformat(),
                "open": 100 + idx,
                "high": 101 + idx,
                "low": 99 + idx,
                "close": 100.5 + idx,
                "volume": 1_000 + idx,
            }
        )
    _write_daily_csv(tmp_path / "AAPL.csv", rows)

    checkpoint = tmp_path / "always_long.pt"
    _write_policy_checkpoint(checkpoint, num_symbols=1, num_actions=2, hot_action=1)

    legacy = daily_stock.run_backtest(
        checkpoint=str(checkpoint),
        symbols=["AAPL"],
        data_dir=str(tmp_path),
        days=20,
        starting_cash=25_000.0,
    )
    server = daily_stock.run_backtest_via_trading_server(
        checkpoint=str(checkpoint),
        symbols=["AAPL"],
        data_dir=str(tmp_path),
        days=20,
        allocation_pct=100.0,
        starting_cash=25_000.0,
    )

    assert server["total_return"] == legacy["total_return"]
    assert server["annualized_return"] == legacy["annualized_return"]
    assert server["sortino"] == legacy["sortino"]
    assert server["max_drawdown"] == legacy["max_drawdown"]
