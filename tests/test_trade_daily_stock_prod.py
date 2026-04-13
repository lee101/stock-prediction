from __future__ import annotations

import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("fastapi")

import trade_daily_stock_prod as daily_stock
from pufferlib_market.inference import Policy
from pufferlib_market.train import TradingPolicy
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


def _write_mlp_policy_checkpoint(
    path: Path,
    *,
    num_symbols: int,
    num_actions: int,
    hot_action: int,
    use_encoder_norm: bool,
    activation: str = "relu",
) -> None:
    obs_size = num_symbols * 16 + 5 + num_symbols
    model = TradingPolicy(
        obs_size,
        num_actions,
        hidden=32,
        activation=activation,
        use_encoder_norm=use_encoder_norm,
    )
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        model.actor[2].bias[hot_action] = 10.0
    torch.save(
        {
            "model": model.state_dict(),
            "config": {"hidden_size": 32},
            "arch": "mlp_relu_sq" if activation == "relu_sq" else "mlp",
            "activation": activation,
            "use_encoder_norm": use_encoder_norm,
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


def test_save_and_load_state_use_shared_path_guard(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []
    guard_paths: list[Path] = []

    class _FakeGuard:
        def acquire_read(self) -> None:
            calls.append("acquire_read")

        def release_read(self) -> None:
            calls.append("release_read")

        def acquire_write(self) -> None:
            calls.append("acquire_write")

        def release_write(self) -> None:
            calls.append("release_write")

    guard = _FakeGuard()
    monkeypatch.setattr(
        daily_stock,
        "shared_path_guard",
        lambda path: guard_paths.append(Path(path)) or guard,
    )

    state_path = tmp_path / "state.json"
    state = daily_stock.StrategyState(active_symbol="AAPL", active_qty=10.0, entry_price=125.5)

    daily_stock.save_state(state, path=state_path)
    loaded = daily_stock.load_state(path=state_path)

    assert calls == ["acquire_write", "release_write", "acquire_read", "release_read"]
    assert guard_paths == [
        daily_stock._state_lock_path(state_path),
        daily_stock._state_lock_path(state_path),
    ]
    assert loaded.active_symbol == "AAPL"
    assert loaded.active_qty == 10.0
    assert loaded.entry_price == 125.5


def test_save_and_load_state_use_cross_process_lock_when_available(monkeypatch, tmp_path: Path) -> None:
    calls: list[object] = []

    class _FakeFcntl:
        LOCK_EX = "lock_ex"
        LOCK_SH = "lock_sh"
        LOCK_UN = "lock_un"

        @staticmethod
        def flock(_fileno: int, op: object) -> None:
            calls.append(op)

    monkeypatch.setattr(daily_stock, "_fcntl", _FakeFcntl)

    state_path = tmp_path / "state.json"
    daily_stock.save_state(daily_stock.StrategyState(active_symbol="AAPL"), path=state_path)
    loaded = daily_stock.load_state(path=state_path)

    assert loaded.active_symbol == "AAPL"
    assert calls == ["lock_ex", "lock_un", "lock_sh", "lock_un"]


def test_load_state_raises_on_malformed_json(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text("{not-json\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"Unreadable strategy state file .*state\.json") as exc_info:
        daily_stock.load_state(path=state_path)

    assert "Expecting property name enclosed in double quotes" in str(exc_info.value)


def test_load_state_raises_on_invalid_payload_shape(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text('{"active_qty": "abc"}', encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"Unreadable strategy state file .*state\.json") as exc_info:
        daily_stock.load_state(path=state_path)

    assert "could not convert string to float" in str(exc_info.value)


def test_load_state_rejects_symlinked_state_file(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text('{"active_symbol": "AAPL"}', encoding="utf-8")
    state_path = tmp_path / "state.json"
    state_path.symlink_to(target)

    with pytest.raises(RuntimeError, match=r"Unreadable strategy state file .*state\.json") as exc_info:
        daily_stock.load_state(path=state_path)

    assert "Unsafe strategy state file path is symlinked" in str(exc_info.value)


def test_load_state_rejects_broken_symlinked_state_file(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state_path.symlink_to(tmp_path / "missing-target.json")

    with pytest.raises(RuntimeError, match=r"Unreadable strategy state file .*state\.json") as exc_info:
        daily_stock.load_state(path=state_path)

    assert "Unsafe strategy state file path is symlinked" in str(exc_info.value)


def test_save_state_writes_via_temp_replace(monkeypatch, tmp_path: Path) -> None:
    replace_calls: dict[str, Path] = {}

    def _fake_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        src_path = Path(src)
        dst_path = Path(dst)
        replace_calls["src"] = src_path
        replace_calls["dst"] = dst_path
        dst_path.write_text(src_path.read_text(encoding="utf-8"), encoding="utf-8")
        src_path.unlink()

    monkeypatch.setattr(daily_stock.os, "replace", _fake_replace)

    state_path = tmp_path / "state.json"
    daily_stock.save_state(
        daily_stock.StrategyState(active_symbol="MSFT", active_qty=3.0),
        path=state_path,
    )

    assert replace_calls["dst"] == state_path
    assert replace_calls["src"] != state_path
    assert not replace_calls["src"].exists()
    assert json.loads(state_path.read_text(encoding="utf-8"))["active_symbol"] == "MSFT"


def test_save_state_rejects_symlinked_state_file(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    state_path = tmp_path / "state.json"
    state_path.symlink_to(target)

    with pytest.raises(OSError, match=r"Unsafe strategy state file path is symlinked: .*state\.json"):
        daily_stock.save_state(daily_stock.StrategyState(active_symbol="AAPL"), path=state_path)


def test_save_state_rejects_broken_symlinked_lock_file(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    lock_path = daily_stock._state_lock_path(state_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.symlink_to(tmp_path / "missing-state.lock")

    with pytest.raises(OSError, match=r"Unsafe strategy state lock file path is symlinked: .*state\.json\.lock"):
        daily_stock.save_state(daily_stock.StrategyState(active_symbol="AAPL"), path=state_path)


def test_load_state_rejects_symlinked_parent_directory(tmp_path: Path) -> None:
    target_dir = tmp_path / "real-state-dir"
    target_dir.mkdir()
    state_path = target_dir / "state.json"
    state_path.write_text('{"active_symbol": "AAPL"}', encoding="utf-8")
    symlink_dir = tmp_path / "linked-state-dir"
    symlink_dir.symlink_to(target_dir, target_is_directory=True)

    with pytest.raises(RuntimeError, match=r"Unreadable strategy state file .*state\.json") as exc_info:
        daily_stock.load_state(path=symlink_dir / "state.json")

    assert "Unsafe strategy state file parent directory is symlinked" in str(exc_info.value)


def test_save_state_rejects_symlinked_parent_directory(tmp_path: Path) -> None:
    target_dir = tmp_path / "real-state-dir"
    target_dir.mkdir()
    symlink_dir = tmp_path / "linked-state-dir"
    symlink_dir.symlink_to(target_dir, target_is_directory=True)

    with pytest.raises(OSError, match=r"Unsafe strategy state file parent directory is symlinked: .*linked-state-dir"):
        daily_stock.save_state(daily_stock.StrategyState(active_symbol="AAPL"), path=symlink_dir / "state.json")


def test_resolve_runtime_config_uses_symbols_file(tmp_path: Path) -> None:
    symbols_file = tmp_path / "symbols.txt"
    symbols_file.write_text("aapl, msft\n# ignore\nnvda\n", encoding="utf-8")

    args = daily_stock.parse_args(
        [
            "--checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--symbols",
            "TSLA",
            "--symbols-file",
            str(symbols_file),
        ]
    )
    config = daily_stock._resolve_runtime_config(args)

    assert config.symbols == ["AAPL", "MSFT", "NVDA"]
    assert config.symbols_file == str(symbols_file)
    assert config.symbol_source == "symbols_file"
    assert config.symbol_source_label == f"symbols file: {symbols_file}"
    assert config.symbol_preview == ["AAPL", "MSFT", "NVDA"]
    assert config.symbol_preview_text == "AAPL, MSFT, NVDA"
    assert "--symbols-file" in config.command_preview()
    assert "TSLA" not in config.command_preview()


def test_symbol_helper_wrappers_remain_available_for_compatibility(tmp_path: Path) -> None:
    symbols_file = tmp_path / "symbols.txt"
    symbols_file.write_text("aapl\nAAPL\n# ignore\nmsft\n", encoding="utf-8")

    assert daily_stock._normalize_stock_symbol(" brk.b ") == "BRK.B"
    assert daily_stock._load_symbols_file(symbols_file) == ["AAPL", "AAPL", "MSFT"]
    assert daily_stock._normalize_symbols(["aapl", "AAPL", " ", "msft"]) == (
        ["AAPL", "MSFT"],
        ["AAPL"],
        ["<blank>"],
    )


def test_resolve_runtime_config_preserves_multi_position_options(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_text("stub", encoding="utf-8")

    args = daily_stock.parse_args(
        [
            "--checkpoint",
            str(checkpoint),
            "--multi-position",
            "3",
            "--multi-position-min-prob-ratio",
            "0.55",
        ]
    )
    config = daily_stock._resolve_runtime_config(args)

    assert config.multi_position == 3
    assert config.multi_position_min_prob_ratio == pytest.approx(0.55)
    preview = config.command_preview()
    assert "--multi-position 3" in preview
    assert "--multi-position-min-prob-ratio 0.55" in preview


def test_resolve_runtime_config_preserves_allocation_sizing_mode(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_text("stub", encoding="utf-8")

    args = daily_stock.parse_args(
        [
            "--checkpoint",
            str(checkpoint),
            "--allocation-sizing-mode",
            "confidence_scaled",
        ]
    )
    config = daily_stock._resolve_runtime_config(args)

    assert config.allocation_sizing_mode == "confidence_scaled"
    assert "--allocation-sizing-mode confidence_scaled" in config.command_preview()


def test_command_preview_omits_extra_checkpoints_when_using_default_ensemble(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_text("stub", encoding="utf-8")

    args = daily_stock.parse_args(
        [
            "--checkpoint",
            str(checkpoint),
        ]
    )
    config = daily_stock._resolve_runtime_config(args)

    assert config.extra_checkpoints == daily_stock._resolved_default_extra_checkpoints()
    assert "--extra-checkpoints" not in config.command_preview()


def test_command_preview_supports_text_preflight_mode(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_text("stub", encoding="utf-8")

    args = daily_stock.parse_args(
        [
            "--checkpoint",
            str(checkpoint),
            "--data-source",
            "local",
        ]
    )
    config = daily_stock._resolve_runtime_config(args)

    preview = config.command_preview(check_config=True, check_config_text=True)

    assert "--check-config" in preview
    assert "--check-config-text" in preview
    assert "--once" in preview
    assert "--paper" in preview


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


def test_run_daemon_passes_multi_position_through_to_run_once(monkeypatch) -> None:
    class _StopDaemon(RuntimeError):
        pass

    captured: dict[str, object] = {}
    sleep_calls: list[float] = []

    monkeypatch.setattr(daily_stock, "load_state", lambda path=daily_stock.STATE_PATH: daily_stock.StrategyState())
    monkeypatch.setattr(
        daily_stock,
        "build_trading_client",
        lambda paper: SimpleNamespace(get_clock=lambda: SimpleNamespace(is_open=True)),
    )
    monkeypatch.setattr(daily_stock, "should_run_today", lambda **kwargs: True)

    def _fake_run_once(**kwargs):
        captured.update(kwargs)
        return {"executed": False}

    def _fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)
        raise _StopDaemon

    monkeypatch.setattr(daily_stock, "run_once", _fake_run_once)
    monkeypatch.setattr(daily_stock.time, "sleep", _fake_sleep)

    with pytest.raises(_StopDaemon):
        daily_stock.run_daemon(
            checkpoint="dummy.ckpt",
            symbols=["AAPL", "MSFT"],
            paper=True,
            allocation_pct=25.0,
            allocation_sizing_mode="static",
            dry_run=True,
            data_dir="trainingdata",
            extra_checkpoints=None,
            execution_backend="alpaca",
            multi_position=2,
            multi_position_min_prob_ratio=0.55,
        )

    assert captured["multi_position"] == 2
    assert captured["multi_position_min_prob_ratio"] == pytest.approx(0.55)
    assert captured["symbols"] == ["AAPL", "MSFT"]
    assert sleep_calls == [60.0]


def test_run_daemon_retries_when_state_load_fails(caplog, monkeypatch) -> None:
    class _StopDaemon(RuntimeError):
        pass

    sleep_calls: list[float] = []
    run_once_calls: list[object] = []

    monkeypatch.setattr(
        daily_stock,
        "load_state",
        lambda path=daily_stock.STATE_PATH: (_ for _ in ()).throw(RuntimeError("state corrupted")),
    )
    monkeypatch.setattr(daily_stock, "run_once", lambda **kwargs: run_once_calls.append(kwargs))

    def _fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)
        raise _StopDaemon

    monkeypatch.setattr(daily_stock.time, "sleep", _fake_sleep)

    with caplog.at_level(logging.WARNING, logger="daily_stock_rl"):
        with pytest.raises(_StopDaemon):
            daily_stock.run_daemon(
                checkpoint="dummy.ckpt",
                symbols=["AAPL"],
                paper=True,
                allocation_pct=25.0,
                allocation_sizing_mode="static",
                dry_run=True,
                data_dir="trainingdata",
            )

    assert run_once_calls == []
    assert sleep_calls == [daily_stock.DEFAULT_DAEMON_RETRY_SLEEP_SECONDS]
    assert any("State load failed (state corrupted); sleeping 5min" in record.getMessage() for record in caplog.records)
    daemon_warning_records = [
        record.getMessage()
        for record in caplog.records
        if record.name == "daily_stock_rl" and record.getMessage().startswith("Daemon warning: ")
    ]
    assert daemon_warning_records
    payload = json.loads(daemon_warning_records[-1].split("Daemon warning: ", 1)[1])
    assert payload["event"] == "daily_stock_daemon_warning"
    assert payload["stage"] == "load_state"
    assert payload["error_type"] == "RuntimeError"
    assert payload["error"] == "state corrupted"
    assert payload["sleep_seconds"] == daily_stock.DEFAULT_DAEMON_RETRY_SLEEP_SECONDS
    assert payload["state_path"] == str(daily_stock.STATE_PATH)
    assert payload["execution_backend"] == "alpaca"
    assert payload["account_mode"] == "paper"


def test_run_daemon_logs_structured_warning_for_clock_fallback_failure(caplog, monkeypatch) -> None:
    class _StopDaemon(RuntimeError):
        pass

    sleep_calls: list[float] = []
    clock_calls: list[bool] = []

    monkeypatch.setattr(daily_stock, "load_state", lambda path=daily_stock.STATE_PATH: daily_stock.StrategyState())

    def _fake_build_trading_client(*, paper: bool):
        clock_calls.append(paper)
        if not paper:
            return SimpleNamespace(get_clock=lambda: (_ for _ in ()).throw(RuntimeError("live clock down")))
        return SimpleNamespace(get_clock=lambda: (_ for _ in ()).throw(RuntimeError("paper clock down")))

    def _fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)
        raise _StopDaemon

    monkeypatch.setattr(daily_stock, "build_trading_client", _fake_build_trading_client)
    monkeypatch.setattr(daily_stock.time, "sleep", _fake_sleep)

    with caplog.at_level(logging.WARNING, logger="daily_stock_rl"):
        with pytest.raises(_StopDaemon):
            daily_stock.run_daemon(
                checkpoint="dummy.ckpt",
                symbols=["AAPL"],
                paper=False,
                allocation_pct=25.0,
                allocation_sizing_mode="static",
                dry_run=True,
                data_dir="trainingdata",
            )

    assert clock_calls == [False, True]
    assert sleep_calls == [daily_stock.DEFAULT_DAEMON_RETRY_SLEEP_SECONDS]
    daemon_warning_records = [
        record.getMessage()
        for record in caplog.records
        if record.name == "daily_stock_rl" and record.getMessage().startswith("Daemon warning: ")
    ]
    assert len(daemon_warning_records) == 2
    live_payload = json.loads(daemon_warning_records[0].split("Daemon warning: ", 1)[1])
    fallback_payload = json.loads(daemon_warning_records[1].split("Daemon warning: ", 1)[1])
    assert live_payload["stage"] == "clock_check_live"
    assert live_payload["retry_with_paper_clock"] is True
    assert live_payload["clock_source"] == "live"
    assert live_payload["error"] == "live clock down"
    assert "sleep_seconds" not in live_payload
    assert fallback_payload["stage"] == "clock_check_paper_fallback"
    assert fallback_payload["clock_source"] == "paper_fallback"
    assert fallback_payload["error"] == "paper clock down"
    assert fallback_payload["sleep_seconds"] == daily_stock.DEFAULT_DAEMON_RETRY_SLEEP_SECONDS
    assert fallback_payload["account_mode"] == "live"


def test_run_daemon_logs_structured_warning_for_paper_clock_failure(caplog, monkeypatch) -> None:
    class _StopDaemon(RuntimeError):
        pass

    sleep_calls: list[float] = []
    clock_calls: list[bool] = []

    monkeypatch.setattr(daily_stock, "load_state", lambda path=daily_stock.STATE_PATH: daily_stock.StrategyState())

    def _fake_build_trading_client(*, paper: bool):
        clock_calls.append(paper)
        return SimpleNamespace(get_clock=lambda: (_ for _ in ()).throw(RuntimeError("paper clock down")))

    def _fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)
        raise _StopDaemon

    monkeypatch.setattr(daily_stock, "build_trading_client", _fake_build_trading_client)
    monkeypatch.setattr(daily_stock.time, "sleep", _fake_sleep)

    with caplog.at_level(logging.WARNING, logger="daily_stock_rl"):
        with pytest.raises(_StopDaemon):
            daily_stock.run_daemon(
                checkpoint="dummy.ckpt",
                symbols=["AAPL"],
                paper=True,
                allocation_pct=25.0,
                allocation_sizing_mode="static",
                dry_run=True,
                data_dir="trainingdata",
            )

    assert clock_calls == [True]
    assert sleep_calls == [daily_stock.DEFAULT_DAEMON_RETRY_SLEEP_SECONDS]
    daemon_warning_records = [
        record.getMessage()
        for record in caplog.records
        if record.name == "daily_stock_rl" and record.getMessage().startswith("Daemon warning: ")
    ]
    assert len(daemon_warning_records) == 1
    payload = json.loads(daemon_warning_records[0].split("Daemon warning: ", 1)[1])
    assert payload["stage"] == "clock_check_paper"
    assert payload["clock_source"] == "paper"
    assert payload["error"] == "paper clock down"
    assert payload["sleep_seconds"] == daily_stock.DEFAULT_DAEMON_RETRY_SLEEP_SECONDS
    assert payload["account_mode"] == "paper"
    assert payload["execution_backend"] == "alpaca"


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


def test_build_signal_accepts_mixed_encoder_norm_mlp_ensemble(tmp_path: Path) -> None:
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
    primary_ckpt = tmp_path / "primary_old.pt"
    extra_ckpt = tmp_path / "extra_new.pt"
    _write_mlp_policy_checkpoint(
        primary_ckpt,
        num_symbols=2,
        num_actions=3,
        hot_action=1,
        use_encoder_norm=False,
    )
    _write_mlp_policy_checkpoint(
        extra_ckpt,
        num_symbols=2,
        num_actions=3,
        hot_action=2,
        use_encoder_norm=True,
    )

    signal, prices = daily_stock.build_signal(
        str(primary_ckpt),
        frames,
        extra_checkpoints=[str(extra_ckpt)],
    )

    assert signal.action in {"long_AAPL", "long_MSFT"}
    assert signal.direction == "long"
    assert prices == {"AAPL": 229.5, "MSFT": 329.5}


def test_load_bare_policy_preserves_mlp_relu_sq_activation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "relu_sq.pt"
    _write_mlp_policy_checkpoint(
        checkpoint,
        num_symbols=2,
        num_actions=3,
        hot_action=1,
        use_encoder_norm=True,
        activation="relu_sq",
    )

    obs_size = 2 * 16 + 5 + 2
    source = TradingPolicy(
        obs_size,
        3,
        hidden=32,
        activation="relu_sq",
        use_encoder_norm=True,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    source.load_state_dict(payload["model"], strict=True)
    source.eval()

    loaded = daily_stock._load_bare_policy(str(checkpoint), obs_size, 3, "cpu")
    obs = torch.randn(4, obs_size)
    with torch.inference_mode():
        expected_logits, expected_value = source(obs)
        actual_logits, actual_value = loaded(obs)

    assert getattr(loaded, "_activation_name", None) == "relu_sq"
    torch.testing.assert_close(actual_logits, expected_logits)
    torch.testing.assert_close(actual_value, expected_value)


def test_load_cached_daily_trader_reuses_template_for_unchanged_checkpoint(
    monkeypatch, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "cached_trader.pt"
    _write_policy_checkpoint(checkpoint, num_symbols=2, num_actions=5, hot_action=1)

    init_calls: list[tuple[str, str, tuple[str, ...], bool, bool]] = []
    real_trader_cls = daily_stock.DailyPPOTrader

    class _RecordingTrader(real_trader_cls):
        def __init__(
            self,
            checkpoint_path,
            device="cpu",
            long_only=False,
            symbols=None,
            allow_unsafe_checkpoint_loading=False,
        ):
            init_calls.append(
                (
                    str(checkpoint_path),
                    str(device),
                    tuple(symbols or ()),
                    bool(long_only),
                    bool(allow_unsafe_checkpoint_loading),
                )
            )
            super().__init__(
                checkpoint_path,
                device=device,
                long_only=long_only,
                symbols=symbols,
                allow_unsafe_checkpoint_loading=allow_unsafe_checkpoint_loading,
            )

    monkeypatch.setattr(daily_stock, "DailyPPOTrader", _RecordingTrader)
    daily_stock._DAILY_TRADER_CACHE.clear()
    try:
        first = daily_stock._load_cached_daily_trader(
            str(checkpoint),
            device="cpu",
            long_only=True,
            symbols=["AAPL", "MSFT"],
        )
        second = daily_stock._load_cached_daily_trader(
            str(checkpoint),
            device="cpu",
            long_only=True,
            symbols=["AAPL", "MSFT"],
        )
    finally:
        daily_stock._DAILY_TRADER_CACHE.clear()

    assert init_calls == [(str(checkpoint), "cpu", ("AAPL", "MSFT"), True, False)]
    assert first is not second
    assert first.policy is second.policy
    assert first.summary_dict() == second.summary_dict()
    first.cash = 123.0
    assert second.cash == 10000.0
    first.current_position = 1
    assert second.current_position is None
    first.SYMBOLS.append("NVDA")
    assert second.SYMBOLS == ["AAPL", "MSFT"]


def test_load_cached_daily_trader_evicts_older_entries_when_capacity_is_exceeded(
    monkeypatch, tmp_path: Path
) -> None:
    checkpoint_a = tmp_path / "a.pt"
    checkpoint_b = tmp_path / "b.pt"
    _write_policy_checkpoint(checkpoint_a, num_symbols=1, num_actions=2, hot_action=1)
    _write_policy_checkpoint(checkpoint_b, num_symbols=1, num_actions=2, hot_action=1)

    init_calls: list[tuple[str, str, tuple[str, ...], bool, bool]] = []
    real_trader_cls = daily_stock.DailyPPOTrader

    class _RecordingTrader(real_trader_cls):
        def __init__(
            self,
            checkpoint_path,
            device="cpu",
            long_only=False,
            symbols=None,
            allow_unsafe_checkpoint_loading=False,
        ):
            init_calls.append(
                (
                    str(checkpoint_path),
                    str(device),
                    tuple(symbols or ()),
                    bool(long_only),
                    bool(allow_unsafe_checkpoint_loading),
                )
            )
            super().__init__(
                checkpoint_path,
                device=device,
                long_only=long_only,
                symbols=symbols,
                allow_unsafe_checkpoint_loading=allow_unsafe_checkpoint_loading,
            )

    monkeypatch.setattr(daily_stock, "DailyPPOTrader", _RecordingTrader)
    monkeypatch.setattr(daily_stock, "_DAILY_TRADER_CACHE_MAX_ENTRIES", 1)
    daily_stock._DAILY_TRADER_CACHE.clear()
    try:
        daily_stock._load_cached_daily_trader(
            str(checkpoint_a),
            device="cpu",
            long_only=True,
            symbols=["AAPL"],
        )
        daily_stock._load_cached_daily_trader(
            str(checkpoint_b),
            device="cpu",
            long_only=True,
            symbols=["AAPL"],
        )
        daily_stock._load_cached_daily_trader(
            str(checkpoint_a),
            device="cpu",
            long_only=True,
            symbols=["AAPL"],
        )
    finally:
        daily_stock._DAILY_TRADER_CACHE.clear()

    assert init_calls == [
        (str(checkpoint_a), "cpu", ("AAPL",), True, False),
        (str(checkpoint_b), "cpu", ("AAPL",), True, False),
        (str(checkpoint_a), "cpu", ("AAPL",), True, False),
    ]


def test_load_cached_daily_trader_skips_cache_when_checkpoint_cannot_be_statted(
    monkeypatch, tmp_path: Path
) -> None:
    missing_checkpoint = tmp_path / "missing.pt"
    init_calls: list[tuple[str, str, tuple[str, ...], bool, bool]] = []

    class _FakeTrader:
        def __init__(
            self,
            checkpoint_path,
            device="cpu",
            long_only=False,
            symbols=None,
            allow_unsafe_checkpoint_loading=False,
        ):
            init_calls.append(
                (
                    str(checkpoint_path),
                    str(device),
                    tuple(symbols or ()),
                    bool(long_only),
                    bool(allow_unsafe_checkpoint_loading),
                )
            )
            self.checkpoint = checkpoint_path
            self.device = device
            self.long_only = long_only
            self.SYMBOLS = list(symbols or [])
            self.policy = object()
            self.cash = 10000.0

    monkeypatch.setattr(daily_stock, "DailyPPOTrader", _FakeTrader)
    daily_stock._DAILY_TRADER_CACHE.clear()
    try:
        first = daily_stock._load_cached_daily_trader(
            str(missing_checkpoint),
            device="cpu",
            long_only=True,
            symbols=["AAPL"],
        )
        second = daily_stock._load_cached_daily_trader(
            str(missing_checkpoint),
            device="cpu",
            long_only=True,
            symbols=["AAPL"],
        )
    finally:
        daily_stock._DAILY_TRADER_CACHE.clear()

    assert init_calls == [
        (str(missing_checkpoint), "cpu", ("AAPL",), True, False),
        (str(missing_checkpoint), "cpu", ("AAPL",), True, False),
    ]
    assert first is not second


def test_load_cached_daily_trader_coalesces_concurrent_cache_misses(
    monkeypatch, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "cached_trader.pt"
    _write_policy_checkpoint(checkpoint, num_symbols=2, num_actions=5, hot_action=1)

    started = Event()
    release = Event()
    init_calls: list[tuple[str, str, tuple[str, ...], bool, bool]] = []
    real_trader_cls = daily_stock.DailyPPOTrader

    class _RecordingTrader(real_trader_cls):
        def __init__(
            self,
            checkpoint_path,
            device="cpu",
            long_only=False,
            symbols=None,
            allow_unsafe_checkpoint_loading=False,
        ):
            init_calls.append(
                (
                    str(checkpoint_path),
                    str(device),
                    tuple(symbols or ()),
                    bool(long_only),
                    bool(allow_unsafe_checkpoint_loading),
                )
            )
            started.set()
            release.wait(timeout=5.0)
            super().__init__(
                checkpoint_path,
                device=device,
                long_only=long_only,
                symbols=symbols,
                allow_unsafe_checkpoint_loading=allow_unsafe_checkpoint_loading,
            )

    monkeypatch.setattr(daily_stock, "DailyPPOTrader", _RecordingTrader)
    daily_stock._DAILY_TRADER_CACHE.clear()
    daily_stock._DAILY_TRADER_CACHE_INFLIGHT.clear()
    results: list[object] = []

    def _worker() -> None:
        results.append(
            daily_stock._load_cached_daily_trader(
                str(checkpoint),
                device="cpu",
                long_only=True,
                symbols=["AAPL", "MSFT"],
            )
        )

    first = Thread(target=_worker)
    second = Thread(target=_worker)
    try:
        first.start()
        assert started.wait(timeout=5.0)
        second.start()
        time.sleep(0.1)
        assert len(init_calls) == 1
        release.set()
        first.join(timeout=5.0)
        second.join(timeout=5.0)
    finally:
        release.set()
        daily_stock._DAILY_TRADER_CACHE.clear()
        daily_stock._DAILY_TRADER_CACHE_INFLIGHT.clear()

    assert len(results) == 2
    assert len(init_calls) == 1


def test_load_bare_policy_reuses_cached_template_for_unchanged_checkpoint(
    monkeypatch, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "cached_policy.pt"
    _write_policy_checkpoint(checkpoint, num_symbols=2, num_actions=3, hot_action=1)
    obs_size = 2 * 16 + 5 + 2

    load_calls: list[dict[str, object]] = []
    real_loader = daily_stock.load_checkpoint_payload

    def _recording_loader(*args, **kwargs):
        load_calls.append(kwargs)
        return real_loader(*args, **kwargs)

    monkeypatch.setattr(daily_stock, "load_checkpoint_payload", _recording_loader)
    daily_stock._BARE_POLICY_CACHE.clear()
    try:
        first = daily_stock._load_bare_policy(str(checkpoint), obs_size, 3, "cpu")
        second = daily_stock._load_bare_policy(str(checkpoint), obs_size, 3, "cpu")
    finally:
        daily_stock._BARE_POLICY_CACHE.clear()

    assert load_calls == [{"map_location": "cpu", "allow_unsafe_checkpoint_loading": False}]
    assert first is second
    obs = torch.randn(4, obs_size)
    with torch.inference_mode():
        expected_logits, expected_value = first(obs)
        actual_logits, actual_value = second(obs)
    torch.testing.assert_close(actual_logits, expected_logits)
    torch.testing.assert_close(actual_value, expected_value)


def test_load_bare_policy_evicts_older_entries_when_capacity_is_exceeded(
    monkeypatch, tmp_path: Path
) -> None:
    checkpoint_a = tmp_path / "a.pt"
    checkpoint_b = tmp_path / "b.pt"
    _write_policy_checkpoint(checkpoint_a, num_symbols=1, num_actions=2, hot_action=1)
    _write_policy_checkpoint(checkpoint_b, num_symbols=1, num_actions=2, hot_action=1)
    obs_size = 1 * 16 + 5 + 1

    load_calls: list[Path] = []
    real_loader = daily_stock.load_checkpoint_payload

    def _recording_loader(*args, **kwargs):
        load_calls.append(Path(args[0]))
        return real_loader(*args, **kwargs)

    monkeypatch.setattr(daily_stock, "load_checkpoint_payload", _recording_loader)
    monkeypatch.setattr(daily_stock, "_BARE_POLICY_CACHE_MAX_ENTRIES", 1)
    daily_stock._BARE_POLICY_CACHE.clear()
    try:
        daily_stock._load_bare_policy(str(checkpoint_a), obs_size, 2, "cpu")
        daily_stock._load_bare_policy(str(checkpoint_b), obs_size, 2, "cpu")
        daily_stock._load_bare_policy(str(checkpoint_a), obs_size, 2, "cpu")
    finally:
        daily_stock._BARE_POLICY_CACHE.clear()

    assert load_calls == [checkpoint_a, checkpoint_b, checkpoint_a]


def test_load_bare_policy_coalesces_concurrent_cache_misses(
    monkeypatch, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "cached_policy.pt"
    _write_policy_checkpoint(checkpoint, num_symbols=2, num_actions=3, hot_action=1)
    obs_size = 2 * 16 + 5 + 2

    started = Event()
    release = Event()
    load_calls: list[dict[str, object]] = []
    real_loader = daily_stock.load_checkpoint_payload

    def _recording_loader(*args, **kwargs):
        load_calls.append(kwargs)
        started.set()
        release.wait(timeout=5.0)
        return real_loader(*args, **kwargs)

    monkeypatch.setattr(daily_stock, "load_checkpoint_payload", _recording_loader)
    daily_stock._BARE_POLICY_CACHE.clear()
    daily_stock._BARE_POLICY_CACHE_INFLIGHT.clear()
    results: list[object] = []

    def _worker() -> None:
        results.append(daily_stock._load_bare_policy(str(checkpoint), obs_size, 3, "cpu"))

    first = Thread(target=_worker)
    second = Thread(target=_worker)
    try:
        first.start()
        assert started.wait(timeout=5.0)
        second.start()
        time.sleep(0.1)
        assert len(load_calls) == 1
        release.set()
        first.join(timeout=5.0)
        second.join(timeout=5.0)
    finally:
        release.set()
        daily_stock._BARE_POLICY_CACHE.clear()
        daily_stock._BARE_POLICY_CACHE_INFLIGHT.clear()

    assert len(results) == 2
    assert len(load_calls) == 1


def test_ensemble_softmax_signal_averages_value_estimates() -> None:
    class _FakePrimary:
        device = "cpu"

        @staticmethod
        def build_observation(_features, _prices):
            return np.zeros(4, dtype=np.float32)

        @staticmethod
        def policy(obs_t):
            return torch.tensor([[0.0, 2.0]]), torch.tensor([[1.5]])

        @staticmethod
        def apply_action_constraints(logits):
            return logits

        @staticmethod
        def _decode_action(action, confidence, value_estimate):
            return SimpleNamespace(
                action=f"long_{action}",
                symbol="AAPL",
                direction="long",
                confidence=confidence,
                value_estimate=value_estimate,
            )

    extra_policies = [
        lambda obs_t: (torch.tensor([[0.0, 1.0]]), torch.tensor([[0.0]])),
        lambda obs_t: (torch.tensor([[0.0, 3.0]]), torch.tensor([[3.0]])),
    ]

    signal = daily_stock._ensemble_softmax_signal(
        _FakePrimary(),
        extra_policies,
        np.zeros((1, 16), dtype=np.float32),
        {"AAPL": 100.0},
    )

    assert signal.direction == "long"
    assert signal.value_estimate == pytest.approx(1.5)


def test_append_signal_log_writes_jsonl(tmp_path: Path) -> None:
    log_path = tmp_path / "daily_stock_rl_signals.jsonl"

    daily_stock.append_signal_log({"symbol": "AAPL", "signal": "flat"}, path=log_path)

    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert rows == [{"signal": "flat", "symbol": "AAPL"}]


def test_append_signal_log_uses_shared_path_guard(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []
    guard_paths: list[Path] = []

    class _FakeGuard:
        def acquire_write(self) -> None:
            calls.append("acquire_write")

        def release_write(self) -> None:
            calls.append("release_write")

    monkeypatch.setattr(
        daily_stock,
        "shared_path_guard",
        lambda path: guard_paths.append(Path(path)) or _FakeGuard(),
    )

    log_path = tmp_path / "daily_stock_rl_signals.jsonl"
    daily_stock.append_signal_log({"symbol": "AAPL", "signal": "flat"}, path=log_path)

    assert calls == ["acquire_write", "release_write"]
    assert guard_paths == [daily_stock._jsonl_log_lock_path(log_path)]


def test_append_signal_log_rejects_symlinked_path(tmp_path: Path) -> None:
    target = tmp_path / "target.jsonl"
    target.write_text("", encoding="utf-8")
    log_path = tmp_path / "daily_stock_rl_signals.jsonl"
    log_path.symlink_to(target)

    with pytest.raises(OSError, match=r"Unsafe signal log path is symlinked: .*signals\.jsonl"):
        daily_stock.append_signal_log({"symbol": "AAPL", "signal": "flat"}, path=log_path)


def test_append_signal_log_rejects_broken_symlinked_path(tmp_path: Path) -> None:
    log_path = tmp_path / "daily_stock_rl_signals.jsonl"
    log_path.symlink_to(tmp_path / "missing-target.jsonl")

    with pytest.raises(OSError, match=r"Unsafe signal log path is symlinked: .*signals\.jsonl"):
        daily_stock.append_signal_log({"symbol": "AAPL", "signal": "flat"}, path=log_path)


def test_append_run_event_log_writes_jsonl(tmp_path: Path) -> None:
    log_path = tmp_path / "daily_stock_rl_run_events.jsonl"

    daily_stock.append_run_event_log({"event": "daily_stock_run_once", "run_id": "abc123"}, path=log_path)

    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert rows == [{"event": "daily_stock_run_once", "run_id": "abc123"}]


def test_append_run_event_log_uses_shared_path_guard(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []
    guard_paths: list[Path] = []

    class _FakeGuard:
        def acquire_write(self) -> None:
            calls.append("acquire_write")

        def release_write(self) -> None:
            calls.append("release_write")

    monkeypatch.setattr(
        daily_stock,
        "shared_path_guard",
        lambda path: guard_paths.append(Path(path)) or _FakeGuard(),
    )

    log_path = tmp_path / "daily_stock_rl_run_events.jsonl"
    daily_stock.append_run_event_log({"event": "daily_stock_run_once", "run_id": "abc123"}, path=log_path)

    assert calls == ["acquire_write", "release_write"]
    assert guard_paths == [daily_stock._jsonl_log_lock_path(log_path)]


def test_append_run_event_log_rejects_symlinked_path(tmp_path: Path) -> None:
    target = tmp_path / "target.jsonl"
    target.write_text("", encoding="utf-8")
    log_path = tmp_path / "daily_stock_rl_run_events.jsonl"
    log_path.symlink_to(target)

    with pytest.raises(OSError, match=r"Unsafe run event log path is symlinked: .*run_events\.jsonl"):
        daily_stock.append_run_event_log({"event": "daily_stock_run_once", "run_id": "abc123"}, path=log_path)


def test_append_run_event_log_rejects_broken_symlinked_path(tmp_path: Path) -> None:
    log_path = tmp_path / "daily_stock_rl_run_events.jsonl"
    log_path.symlink_to(tmp_path / "missing-target.jsonl")

    with pytest.raises(OSError, match=r"Unsafe run event log path is symlinked: .*run_events\.jsonl"):
        daily_stock.append_run_event_log({"event": "daily_stock_run_once", "run_id": "abc123"}, path=log_path)


def test_append_run_event_log_rejects_broken_symlinked_lock_path(tmp_path: Path) -> None:
    log_path = tmp_path / "daily_stock_rl_run_events.jsonl"
    lock_path = daily_stock._jsonl_log_lock_path(log_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.symlink_to(tmp_path / "missing-run-event.lock")

    with pytest.raises(OSError, match=r"Unsafe run event log lock file path is symlinked: .*run_events\.jsonl\.lock"):
        daily_stock.append_run_event_log({"event": "daily_stock_run_once", "run_id": "abc123"}, path=log_path)


def test_append_signal_log_rejects_broken_symlinked_lock_path(tmp_path: Path) -> None:
    log_path = tmp_path / "daily_stock_rl_signals.jsonl"
    lock_path = daily_stock._jsonl_log_lock_path(log_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.symlink_to(tmp_path / "missing-signal.lock")

    with pytest.raises(OSError, match=r"Unsafe signal log lock file path is symlinked: .*signals\.jsonl\.lock"):
        daily_stock.append_signal_log({"symbol": "AAPL", "signal": "flat"}, path=log_path)


def test_append_signal_log_rejects_symlinked_parent_directory(tmp_path: Path) -> None:
    target_dir = tmp_path / "real-log-dir"
    target_dir.mkdir()
    symlink_dir = tmp_path / "linked-log-dir"
    symlink_dir.symlink_to(target_dir, target_is_directory=True)

    with pytest.raises(OSError, match=r"Unsafe signal log parent directory is symlinked: .*linked-log-dir"):
        daily_stock.append_signal_log({"symbol": "AAPL", "signal": "flat"}, path=symlink_dir / "signals.jsonl")


def test_append_run_event_log_rejects_symlinked_parent_directory(tmp_path: Path) -> None:
    target_dir = tmp_path / "real-log-dir"
    target_dir.mkdir()
    symlink_dir = tmp_path / "linked-log-dir"
    symlink_dir.symlink_to(target_dir, target_is_directory=True)

    with pytest.raises(OSError, match=r"Unsafe run event log parent directory is symlinked: .*linked-log-dir"):
        daily_stock.append_run_event_log({"event": "daily_stock_run_once"}, path=symlink_dir / "events.jsonl")


def test_main_uses_default_resolved_ensemble_checkpoints(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_run_once(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(daily_stock, "run_once", _fake_run_once)
    monkeypatch.setattr(
        daily_stock,
        "_checkpoint_load_diagnostics",
        lambda _config: {
            "ok": True,
            "error": None,
            "primary": {
                "arch": "mlp",
                "feature_schema": "legacy_prod",
                "feature_dimension": 8,
            },
            "extras": [{"path": "extra.pt", "class": "MLPPolicy"}],
        },
    )

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
    assert config.server_url_source == "env"
    assert config.resolved_server_url_details == {
        "host": "server.internal",
        "transport": "http",
        "scope": "remote",
        "security": "insecure_remote_http",
    }
    assert payload["ensemble_size"] == 2
    assert payload["backtest_starting_cash"] == 25_000.0
    assert payload["missing_checkpoints"] == [str(extra)]
    assert payload["configured_server_url"] is None
    assert payload["server_url"] == "http://server.internal:9000"
    assert payload["resolved_server_url"] == "http://server.internal:9000"
    assert payload["server_url_source"] == "env"
    assert payload["server_url_transport"] == "http"
    assert payload["server_url_scope"] == "remote"
    assert payload["server_url_security"] == "insecure_remote_http"


def test_ensemble_dir_auto_discovers_checkpoints(tmp_path) -> None:
    ens_dir = tmp_path / "ensemble"
    ens_dir.mkdir()
    for name in ["a.pt", "b.pt", "c.pt"]:
        (ens_dir / name).write_text("stub", encoding="utf-8")
    args = daily_stock.parse_args(
        ["--ensemble-dir", str(ens_dir), "--once"]
    )
    config = daily_stock._resolve_runtime_config(args)
    paths = config.checkpoint_paths
    assert len(paths) == 3
    assert all(p.endswith(".pt") for p in paths)


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
    assert payload["symbol_source"] == "cli"
    assert payload["symbol_source_label"] == "--symbols"
    assert payload["symbol_preview"] == ["AAPL", "MSFT"]
    assert payload["symbol_preview_text"] == "AAPL, MSFT"
    assert payload["summary"] == "live once via alpaca using alpaca data on 2 symbols with 1 checkpoint as single-position"
    assert payload["portfolio_mode"] is False
    assert payload["position_capacity"] == 1
    assert payload["strategy_mode"] == "single-position"
    assert payload["extra_checkpoints"] is None
    assert payload["backtest_starting_cash"] == daily_stock.DEFAULT_BACKTEST_STARTING_CASH
    assert payload["daily_frame_min_days"] == daily_stock.DEFAULT_DAILY_FRAME_MIN_DAYS
    assert (
        payload["alpaca_daily_history_lookback_days"]
        == daily_stock.DEFAULT_ALPACA_DAILY_HISTORY_LOOKBACK_DAYS
    )
    assert payload["bar_freshness_max_age_days"] == daily_stock.DEFAULT_BAR_FRESHNESS_MAX_AGE_DAYS
    assert payload["checkpoint"] == str(checkpoint_path)
    assert "--check-config" in payload["check_command_preview"]
    assert "--check-config-text" in payload["check_text_command_preview"]
    assert "--once" in payload["run_command_preview"]
    assert "--live" in payload["run_command_preview"]
    assert "--dry-run" in payload["safe_command_preview"]
    assert payload["checkpoints_exist"] is checkpoint_path.exists()
    assert payload["missing_checkpoints"] == ([] if checkpoint_path.exists() else [str(checkpoint_path)])


def test_main_print_config_reports_unsafe_checkpoint_loading_opt_in(tmp_path: Path, capsys) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("stub", encoding="utf-8")

    daily_stock.main(
        [
            "--print-config",
            "--no-ensemble",
            "--symbols",
            "AAPL",
            "--checkpoint",
            str(checkpoint),
            "--allow-unsafe-checkpoint-loading",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert payload["allow_unsafe_checkpoint_loading"] is True
    assert "--allow-unsafe-checkpoint-loading" in payload["run_command_preview"]
    assert "--allow-unsafe-checkpoint-loading" in payload["safe_command_preview"]


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


def test_main_print_config_reports_symbol_file_preview(tmp_path: Path, capsys) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("stub", encoding="utf-8")
    symbols_file = tmp_path / "symbols.txt"
    symbols_file.write_text(
        "aapl\nmsft\nnvda\namd\nmeta\ngoogl\n",
        encoding="utf-8",
    )

    daily_stock.main(
        [
            "--print-config",
            "--no-ensemble",
            "--checkpoint",
            str(checkpoint),
            "--symbols-file",
            str(symbols_file),
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert payload["symbol_source"] == "symbols_file"
    assert payload["symbol_source_label"] == f"symbols file: {symbols_file}"
    assert payload["symbol_preview"] == ["AAPL", "MSFT", "NVDA", "AMD", "META"]
    assert payload["symbol_preview_text"] == "AAPL, MSFT, NVDA, AMD, META (+1 more)"


def test_main_logs_runtime_config(caplog, monkeypatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("stub", encoding="utf-8")

    captured: dict[str, object] = {}

    def _fake_run_once(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(daily_stock, "run_once", _fake_run_once)
    monkeypatch.setattr(
        daily_stock,
        "_checkpoint_load_diagnostics",
        lambda _config: {
            "ok": True,
            "error": None,
            "primary": {
                "arch": "mlp",
                "feature_schema": "legacy_prod",
                "feature_dimension": 8,
            },
            "extras": [{"path": "extra.pt", "class": "MLPPolicy"}],
        },
    )

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
    assert '"checkpoint_feature_schema": "legacy_prod"' in runtime_records[-1]
    assert '"checkpoint_feature_dimension": 8' in runtime_records[-1]
    assert '"primary_checkpoint_arch": "mlp"' in runtime_records[-1]
    assert '"extra_checkpoint_classes": ["MLPPolicy"]' in runtime_records[-1]
    assert captured["checkpoint"] == str(checkpoint)


def test_runtime_log_payload_includes_local_data_health_and_warnings(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("stub", encoding="utf-8")
    data_root = tmp_path / "trainingdata"
    train_root = data_root / "train"
    train_root.mkdir(parents=True)
    _write_daily_csv(
        train_root / "AAPL.csv",
        [
            {
                "timestamp": "2026-04-07T00:00:00Z",
                "open": 10,
                "high": 11,
                "low": 9,
                "close": 10.5,
                "volume": 100,
            }
        ],
    )
    _write_daily_csv(
        train_root / "MSFT.csv",
        [
            {
                "timestamp": "2026-04-07T00:00:00Z",
                "open": 20,
                "high": 21,
                "low": 19,
                "close": 20.5,
                "volume": 200,
            }
        ],
    )
    config = daily_stock._resolve_runtime_config(
        daily_stock.parse_args(
            [
                "--backtest",
                "--no-ensemble",
                "--checkpoint",
                str(checkpoint),
                "--data-dir",
                str(data_root),
                "--symbols",
                "AAPL",
                "MSFT",
            ]
        )
    )

    payload = daily_stock._runtime_log_payload(
        config,
        preflight_payload={
            "local_data_status_counts": {
                "usable": 2,
                "stale": 1,
                "missing": 0,
                "invalid": 1,
            },
            "checkpoint_load": {
                "ok": True,
                "error": None,
                "primary": {
                    "arch": "mlp",
                    "feature_schema": "rsi_v5",
                    "feature_dimension": 9,
                },
                "extras": [{"path": "extra.pt", "class": "AltPolicy"}],
            },
            "usable_symbol_count": 2,
            "latest_local_data_date": "2026-04-08",
            "stale_symbol_data": {"MSFT": "2026-04-07"},
            "warnings": ["local data lags freshest date 2026-04-08"],
        },
    )

    assert payload["checkpoint_feature_schema"] == "rsi_v5"
    assert payload["checkpoint_feature_dimension"] == 9
    assert payload["primary_checkpoint_arch"] == "mlp"
    assert payload["extra_checkpoint_classes"] == ["AltPolicy"]
    assert payload["requested_local_data_dir"] == str(data_root.resolve())
    assert payload["usable_symbol_count"] == 2
    assert payload["latest_local_data_date"] == "2026-04-08"
    assert payload["resolved_local_data_dir"] == str(train_root.resolve())
    assert payload["resolved_local_data_dir_source"] == "nested_train"
    assert payload["stale_symbol_count"] == 1
    assert payload["missing_symbol_count"] == 0
    assert payload["invalid_symbol_count"] == 1
    assert payload["preflight_warnings"] == ["local data lags freshest date 2026-04-08"]


def test_runtime_log_payload_reuses_preflight_local_data_dir_context(
    monkeypatch,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("stub", encoding="utf-8")
    config = daily_stock._resolve_runtime_config(
        daily_stock.parse_args(
            [
                "--backtest",
                "--no-ensemble",
                "--checkpoint",
                str(checkpoint),
                "--data-dir",
                "daily_data",
                "--symbols",
                "AAPL",
            ]
        )
    )
    monkeypatch.setattr(
        daily_stock,
        "_local_data_dir_context",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not be called")),
    )

    payload = daily_stock._runtime_log_payload(
        config,
        preflight_payload={
            "requested_local_data_dir": str((tmp_path / "requested").resolve()),
            "resolved_local_data_dir": str((tmp_path / "resolved").resolve()),
            "resolved_local_data_dir_source": "requested",
        },
    )

    assert payload["data_dir"] == "daily_data"
    assert payload["requested_local_data_dir"] == str((tmp_path / "requested").resolve())
    assert payload["resolved_local_data_dir"] == str((tmp_path / "resolved").resolve())
    assert payload["resolved_local_data_dir_source"] == "requested"


def test_runtime_log_payload_falls_back_when_preflight_local_data_dir_context_is_invalid(
    monkeypatch,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("stub", encoding="utf-8")
    config = daily_stock._resolve_runtime_config(
        daily_stock.parse_args(
            [
                "--backtest",
                "--no-ensemble",
                "--checkpoint",
                str(checkpoint),
                "--data-dir",
                "daily_data",
                "--symbols",
                "AAPL",
            ]
        )
    )
    captured: dict[str, object] = {}

    def _fake_local_data_dir_context(data_dir: str, symbols: list[str]) -> daily_stock.LocalDataDirContext:
        captured["data_dir"] = data_dir
        captured["symbols"] = list(symbols)
        return {
            "requested_local_data_dir": str((tmp_path / "requested-fallback").resolve()),
            "resolved_local_data_dir": str((tmp_path / "resolved-fallback").resolve()),
            "resolved_local_data_dir_source": daily_stock.LocalDataDirResolutionSource.REQUESTED,
        }

    monkeypatch.setattr(daily_stock, "_local_data_dir_context", _fake_local_data_dir_context)

    payload = daily_stock._runtime_log_payload(
        config,
        preflight_payload={
            "requested_local_data_dir": str((tmp_path / "requested").resolve()),
            "resolved_local_data_dir": str((tmp_path / "resolved").resolve()),
            "resolved_local_data_dir_source": "bogus",
        },
    )

    assert captured == {"data_dir": "daily_data", "symbols": ["AAPL"]}
    assert payload["requested_local_data_dir"] == str((tmp_path / "requested-fallback").resolve())
    assert payload["resolved_local_data_dir"] == str((tmp_path / "resolved-fallback").resolve())
    assert payload["resolved_local_data_dir_source"] == "requested"


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


def test_main_check_config_reports_ready_for_local_setup(monkeypatch, tmp_path: Path, capsys) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("stub", encoding="utf-8")
    monkeypatch.setattr(
        daily_stock,
        "_checkpoint_load_diagnostics",
        lambda _config: {
            "ok": True,
            "error": None,
            "primary": {
                "arch": "mlp",
                "feature_schema": "legacy_prod",
                "feature_dimension": 8,
            },
            "extras": [{"path": "extra.pt", "class": "MLPPolicy"}],
        },
    )
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
    assert payload["next_steps"]
    assert payload["local_data_required"] is True
    assert payload["data_dir_exists"] is True
    assert payload["missing_local_symbol_files"] == []
    assert payload["usable_symbols"] == ["AAPL"]
    assert payload["usable_symbol_count"] == 1
    assert payload["latest_local_data_date"] == "2026-01-01"
    assert payload["oldest_local_data_date"] == "2026-01-01"
    assert payload["stale_symbol_data"] == {}
    assert payload["symbol_details"] == {
        "AAPL": {
            "status": "usable",
            "file_path": str((tmp_path / "AAPL.csv").resolve()),
            "local_data_date": "2026-01-01",
            "row_count": 1,
            "reason": None,
        }
    }
    assert payload["checkpoint_load"]["ok"] is True
    assert payload["summary"] == "paper once via alpaca using local data on 1 symbol with 1 checkpoint as single-position"
    assert payload["portfolio_mode"] is False
    assert payload["position_capacity"] == 1
    assert payload["strategy_mode"] == "single-position"
    assert payload["daily_frame_min_days"] == daily_stock.DEFAULT_DAILY_FRAME_MIN_DAYS
    assert (
        payload["alpaca_daily_history_lookback_days"]
        == daily_stock.DEFAULT_ALPACA_DAILY_HISTORY_LOOKBACK_DAYS
    )
    assert payload["bar_freshness_max_age_days"] == daily_stock.DEFAULT_BAR_FRESHNESS_MAX_AGE_DAYS
    assert "--check-config" in payload["check_command_preview"]
    assert "--check-config-text" in payload["check_text_command_preview"]
    assert "--dry-run" in payload["run_command_preview"]
    assert payload["run_command_preview"] == payload["safe_command_preview"]
    assert any("Run a safe dry run:" in step for step in payload["next_steps"])
    assert any("Re-run text preflight later:" in step for step in payload["next_steps"])


def test_main_check_config_text_emits_human_ready_summary(monkeypatch, tmp_path: Path, capsys) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("stub", encoding="utf-8")
    monkeypatch.setattr(
        daily_stock,
        "_checkpoint_load_diagnostics",
        lambda _config: {
            "ok": True,
            "error": None,
            "primary": {
                "arch": "mlp",
                "feature_schema": "legacy_prod",
                "feature_dimension": 8,
            },
            "extras": [{"path": "extra.pt", "class": "MLPPolicy"}],
        },
    )
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
            "--check-config-text",
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

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["ready"] is True
    assert "Daily stock RL setup is ready." in captured.err
    assert "Strategy mode: single-position" in captured.err
    assert "Position capacity: 1" in captured.err
    assert "Symbol source: --symbols" in captured.err
    assert "Symbols: AAPL" in captured.err
    assert f"Resolved local data dir: {tmp_path.resolve()}" in captured.err
    assert "Local data dir source: requested directory" in captured.err
    assert "Checkpoint load:" in captured.err
    assert "- primary: arch=mlp, schema=legacy_prod, feature_dim=8" in captured.err
    assert "- ensemble extras: MLPPolicy" in captured.err
    assert "Suggested commands:" in captured.err
    assert f"- dry run: {payload['safe_command_preview']}" in captured.err
    assert "Additional next steps:" not in captured.err


def test_main_check_config_summary_emits_compact_ready_line(monkeypatch, tmp_path: Path, capsys) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("stub", encoding="utf-8")
    monkeypatch.setattr(
        daily_stock,
        "_checkpoint_load_diagnostics",
        lambda _config: {
            "ok": True,
            "error": None,
            "primary": {
                "arch": "mlp",
                "feature_schema": "legacy_prod",
                "feature_dimension": 8,
            },
            "extras": [],
        },
    )
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
            "--check-config-summary",
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

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["ready"] is True
    assert captured.err.strip() == (
        "READY | single-position | symbols=1/1 usable | "
        "checkpoint=mlp schema=legacy_prod dim=8 | latest_local_data=2026-01-01"
    )


def test_main_check_config_summary_and_text_can_be_combined(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("stub", encoding="utf-8")
    monkeypatch.setattr(
        daily_stock,
        "_checkpoint_load_diagnostics",
        lambda _config: {
            "ok": True,
            "error": None,
            "primary": {
                "arch": "mlp",
                "feature_schema": "legacy_prod",
                "feature_dimension": 8,
            },
            "extras": [],
        },
    )
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
            "--check-config-summary",
            "--check-config-text",
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

    captured = capsys.readouterr()

    assert captured.err.startswith(
        "READY | single-position | symbols=1/1 usable | "
        "checkpoint=mlp schema=legacy_prod dim=8 | latest_local_data=2026-01-01\n"
    )
    assert "Daily stock RL setup is ready." in captured.err


def test_parse_args_rejects_check_config_text_without_check_config(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        daily_stock.parse_args(["--check-config-text"])

    captured = capsys.readouterr()

    assert exc_info.value.code == 2
    assert "--check-config-text requires --check-config" in captured.err


def test_parse_args_rejects_check_config_summary_without_check_config(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        daily_stock.parse_args(["--check-config-summary"])

    captured = capsys.readouterr()

    assert exc_info.value.code == 2
    assert "--check-config-summary requires --check-config" in captured.err


def test_main_print_config_reports_portfolio_strategy_mode(tmp_path: Path, capsys) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("stub", encoding="utf-8")

    daily_stock.main(
        [
            "--print-config",
            "--no-ensemble",
            "--symbols",
            "AAPL",
            "MSFT",
            "--checkpoint",
            str(checkpoint),
            "--multi-position",
            "3",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert payload["portfolio_mode"] is True
    assert payload["position_capacity"] == 3
    assert payload["strategy_mode"] == "portfolio (up to 3 positions)"
    assert payload["summary"] == (
        "paper once via alpaca using alpaca data on 2 symbols "
        "with 1 checkpoint as portfolio (up to 3 positions)"
    )


def test_main_check_config_text_reports_portfolio_strategy_mode(monkeypatch, tmp_path: Path, capsys) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("stub", encoding="utf-8")
    monkeypatch.setattr(
        daily_stock,
        "_checkpoint_load_diagnostics",
        lambda _config: {"ok": True, "error": None, "primary": {"arch": "mlp"}, "extras": []},
    )

    daily_stock.main(
        [
            "--check-config",
            "--check-config-text",
            "--dry-run",
            "--no-ensemble",
            "--symbols",
            "AAPL",
            "MSFT",
            "--checkpoint",
            str(checkpoint),
            "--multi-position",
            "3",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["ready"] is True
    assert payload["portfolio_mode"] is True
    assert "Strategy mode: portfolio (up to 3 positions)" in captured.err
    assert "Position capacity: 3" in captured.err


def test_main_check_config_reports_symbol_level_local_data_health(monkeypatch, tmp_path: Path, capsys) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("stub", encoding="utf-8")
    monkeypatch.setattr(
        daily_stock,
        "_checkpoint_load_diagnostics",
        lambda _config: {"ok": True, "error": None, "primary": {"arch": "mlp"}, "extras": []},
    )
    _write_daily_csv(
        tmp_path / "AAPL.csv",
        [
            {
                "timestamp": "2026-01-03T00:00:00Z",
                "open": 10,
                "high": 11,
                "low": 9,
                "close": 10.5,
                "volume": 100,
            }
        ],
    )
    _write_daily_csv(
        tmp_path / "MSFT.csv",
        [
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "open": 20,
                "high": 21,
                "low": 19,
                "close": 20.5,
                "volume": 200,
            }
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        daily_stock.main(
            [
                "--check-config",
                "--dry-run",
                "--data-source",
                "local",
                "--symbols",
                "AAPL",
                "MSFT",
                "NVDA",
                "--checkpoint",
                str(checkpoint),
                "--no-ensemble",
                "--data-dir",
                str(tmp_path),
            ]
        )

    payload = json.loads(capsys.readouterr().out)

    assert exc_info.value.code == 1
    assert payload["ready"] is False
    assert payload["usable_symbols"] == ["AAPL", "MSFT"]
    assert payload["usable_symbol_count"] == 2
    assert payload["latest_local_data_date"] == "2026-01-03"
    assert payload["oldest_local_data_date"] == "2026-01-01"
    assert payload["stale_symbol_data"] == {"MSFT": "2026-01-01"}
    assert payload["local_data_status_counts"] == {
        "usable": 1,
        "stale": 1,
        "missing": 1,
        "invalid": 0,
    }
    assert payload["missing_local_symbol_files"] == [str((tmp_path / "NVDA.csv").resolve())]
    assert payload["symbol_details"] == {
        "AAPL": {
            "status": "usable",
            "file_path": str((tmp_path / "AAPL.csv").resolve()),
            "local_data_date": "2026-01-03",
            "row_count": 1,
            "reason": None,
        },
        "MSFT": {
            "status": "stale",
            "file_path": str((tmp_path / "MSFT.csv").resolve()),
            "local_data_date": "2026-01-01",
            "row_count": 1,
            "reason": "local data lags freshest date 2026-01-03",
        },
        "NVDA": {
            "status": "missing",
            "file_path": str((tmp_path / "NVDA.csv").resolve()),
            "local_data_date": None,
            "row_count": None,
            "reason": "missing local daily CSV",
        },
    }
    assert any("Missing local daily data file(s):" in error for error in payload["errors"])
    assert any("local daily CSV data for 1 symbol lags freshest date 2026-01-03" in warning for warning in payload["warnings"])
    assert any("Refresh stale local daily CSVs for: MSFT." == step for step in payload["next_steps"])


def test_main_check_config_reports_unreadable_local_symbol_data(monkeypatch, tmp_path: Path, capsys) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("stub", encoding="utf-8")
    monkeypatch.setattr(
        daily_stock,
        "_checkpoint_load_diagnostics",
        lambda _config: {"ok": True, "error": None, "primary": {"arch": "mlp"}, "extras": []},
    )
    _write_daily_csv(
        tmp_path / "AAPL.csv",
        [
            {
                "timestamp": "2026-01-03T00:00:00Z",
                "open": 10,
                "high": 11,
                "low": 9,
                "close": 10.5,
                "volume": 100,
            }
        ],
    )
    (tmp_path / "MSFT.csv").write_text("bad,data\n1,2\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        daily_stock.main(
            [
                "--check-config",
                "--dry-run",
                "--data-source",
                "local",
                "--symbols",
                "AAPL",
                "MSFT",
                "--checkpoint",
                str(checkpoint),
                "--no-ensemble",
                "--data-dir",
                str(tmp_path),
            ]
        )

    payload = json.loads(capsys.readouterr().out)

    assert exc_info.value.code == 1
    assert payload["ready"] is False
    assert payload["usable_symbols"] == ["AAPL"]
    assert payload["usable_symbol_count"] == 1
    assert payload["latest_local_data_date"] == "2026-01-03"
    assert payload["oldest_local_data_date"] == "2026-01-03"
    assert payload["missing_local_symbol_files"] == []
    assert payload["stale_symbol_data"] == {}
    assert payload["local_data_status_counts"] == {
        "usable": 1,
        "stale": 0,
        "missing": 0,
        "invalid": 1,
    }
    assert payload["symbol_details"] == {
        "AAPL": {
            "status": "usable",
            "file_path": str((tmp_path / "AAPL.csv").resolve()),
            "local_data_date": "2026-01-03",
            "row_count": 1,
            "reason": None,
        },
        "MSFT": {
            "status": "invalid",
            "file_path": str((tmp_path / "MSFT.csv").resolve()),
            "local_data_date": None,
            "row_count": None,
            "reason": "Daily frame missing columns: timestamp/date + ['open', 'high', 'low', 'close', 'volume']",
        },
    }
    assert any("Unreadable local daily data file(s):" in error for error in payload["errors"])
    assert any("Repair unreadable local daily CSVs for: MSFT." == step for step in payload["next_steps"])


def test_main_check_config_reports_missing_local_data_directory(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("stub", encoding="utf-8")
    missing_dir = tmp_path / "missing_data"
    monkeypatch.setattr(
        daily_stock,
        "_checkpoint_load_diagnostics",
        lambda _config: {
            "ok": True,
            "error": None,
            "primary": {
                "arch": "mlp",
                "feature_schema": "legacy_prod",
                "feature_dimension": 8,
            },
            "extras": [],
        },
    )

    with pytest.raises(SystemExit) as exc_info:
        daily_stock.main(
            [
                "--check-config",
                "--check-config-summary",
                "--dry-run",
                "--data-source",
                "local",
                "--symbols",
                "AAPL",
                "MSFT",
                "--checkpoint",
                str(checkpoint),
                "--no-ensemble",
                "--data-dir",
                str(missing_dir),
            ]
        )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exc_info.value.code == 1
    assert payload["ready"] is False
    assert payload["data_dir_exists"] is False
    assert payload["resolved_local_data_dir"] == str(missing_dir.resolve())
    assert payload["resolved_local_data_dir_source"] == "requested"
    assert payload["usable_symbols"] == []
    assert payload["usable_symbol_count"] == 0
    assert payload["local_data_status_counts"] == {
        "usable": 0,
        "stale": 0,
        "missing": 2,
        "invalid": 0,
    }
    assert payload["missing_local_symbol_files"] == [
        str((missing_dir / "AAPL.csv").resolve()),
        str((missing_dir / "MSFT.csv").resolve()),
    ]
    assert payload["symbol_details"] == {
        "AAPL": {
            "status": "missing",
            "file_path": str((missing_dir / "AAPL.csv").resolve()),
            "local_data_date": None,
            "row_count": None,
            "reason": "missing local daily CSV",
        },
        "MSFT": {
            "status": "missing",
            "file_path": str((missing_dir / "MSFT.csv").resolve()),
            "local_data_date": None,
            "row_count": None,
            "reason": "missing local daily CSV",
        },
    }
    assert any(
        error == f"Local data directory does not exist: {missing_dir}"
        for error in payload["errors"]
    )
    assert any(
        step == "Add the missing local daily CSV files under the resolved data directory."
        for step in payload["next_steps"]
    )
    assert "local_issues=missing:2" in captured.err.strip()


def test_main_failure_summary_includes_invalid_local_data_reason(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("stub", encoding="utf-8")
    monkeypatch.setattr(
        daily_stock,
        "_checkpoint_load_diagnostics",
        lambda _config: {
            "ok": True,
            "error": None,
            "primary": {
                "arch": "mlp",
                "feature_schema": "legacy_prod",
                "feature_dimension": 8,
            },
            "extras": [],
        },
    )
    _write_daily_csv(
        tmp_path / "AAPL.csv",
        [
            {
                "timestamp": "2026-01-03T00:00:00Z",
                "open": 10,
                "high": 11,
                "low": 9,
                "close": 10.5,
                "volume": 100,
            }
        ],
    )
    (tmp_path / "MSFT.csv").write_text("timestamp,open\n2026-01-03T00:00:00Z,10\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        daily_stock.main(
            [
                "--dry-run",
                "--data-source",
                "local",
                "--symbols",
                "AAPL",
                "MSFT",
                "--checkpoint",
                str(checkpoint),
                "--no-ensemble",
                "--data-dir",
                str(tmp_path),
            ]
        )

    stderr = capsys.readouterr().err

    assert exc_info.value.code == 1
    assert (
        "- invalid symbols: MSFT (Daily frame missing columns: timestamp/date + "
        "['open', 'high', 'low', 'close', 'volume'])"
    ) in stderr


def test_build_local_symbol_details_uses_lightweight_metadata_inspection(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_daily_csv(
        tmp_path / "AAPL.csv",
        [
            {
                "timestamp": "2026-01-03T00:00:00Z",
                "open": 10,
                "high": 11,
                "low": 9,
                "close": 10.5,
                "volume": 100,
            },
            {
                "timestamp": "2026-01-04T00:00:00Z",
                "open": 10.5,
                "high": 11.5,
                "low": 9.5,
                "close": 11.0,
                "volume": 120,
            },
        ],
    )
    monkeypatch.setattr(
        daily_stock,
        "_normalize_daily_frame",
        lambda _frame: (_ for _ in ()).throw(AssertionError("preflight should not normalize full daily frames")),
    )

    details, usable_symbols, latest_local_data_date, oldest_local_data_date = daily_stock._build_local_symbol_details(
        symbols=["AAPL"],
        resolved_local_data_dir=tmp_path,
    )

    assert usable_symbols == ["AAPL"]
    assert latest_local_data_date == "2026-01-04"
    assert oldest_local_data_date == "2026-01-04"
    assert details["AAPL"] == {
        "status": "usable",
        "file_path": str((tmp_path / "AAPL.csv").resolve()),
        "local_data_date": "2026-01-04",
        "row_count": 2,
        "reason": None,
    }


def test_inspect_local_daily_symbol_file_reuses_cached_result(monkeypatch, tmp_path: Path) -> None:
    csv_path = tmp_path / "AAPL.csv"
    _write_daily_csv(
        csv_path,
        [
            {
                "timestamp": "2026-01-03T00:00:00Z",
                "open": 10,
                "high": 11,
                "low": 9,
                "close": 10.5,
                "volume": 100,
            },
            {
                "timestamp": "2026-01-04T00:00:00Z",
                "open": 10.5,
                "high": 11.5,
                "low": 9.5,
                "close": 11.0,
                "volume": 120,
            },
        ],
    )

    first = daily_stock._inspect_local_daily_symbol_file(csv_path)

    monkeypatch.setattr(
        daily_stock.pd,
        "read_csv",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("expected cached inspection reuse")),
    )

    second = daily_stock._inspect_local_daily_symbol_file(csv_path)

    assert second == first


def test_inspect_local_daily_symbol_file_coalesces_concurrent_cache_misses(
    monkeypatch,
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "AAPL.csv"
    _write_daily_csv(
        csv_path,
        [
            {
                "timestamp": "2026-01-03T00:00:00Z",
                "open": 10,
                "high": 11,
                "low": 9,
                "close": 10.5,
                "volume": 100,
            },
            {
                "timestamp": "2026-01-04T00:00:00Z",
                "open": 10.5,
                "high": 11.5,
                "low": 9.5,
                "close": 11.0,
                "volume": 120,
            },
        ],
    )

    original_read_csv = daily_stock.pd.read_csv
    started = Event()
    release = Event()
    full_read_calls = 0

    def _blocking_read_csv(*args, **kwargs):
        nonlocal full_read_calls
        frame = original_read_csv(*args, **kwargs)
        if kwargs.get("nrows") == 0:
            return frame
        full_read_calls += 1
        started.set()
        release.wait(timeout=5.0)
        return frame

    monkeypatch.setattr(daily_stock.pd, "read_csv", _blocking_read_csv)
    daily_stock._LOCAL_DAILY_SYMBOL_INSPECTION_CACHE.clear()
    daily_stock._LOCAL_DAILY_SYMBOL_INSPECTION_CACHE_INFLIGHT.clear()
    results: list[daily_stock.LocalDailySymbolInspection] = []

    def _worker() -> None:
        results.append(daily_stock._inspect_local_daily_symbol_file(csv_path))

    first = Thread(target=_worker)
    second = Thread(target=_worker)
    try:
        first.start()
        assert started.wait(timeout=5.0)
        second.start()
        time.sleep(0.1)
        assert full_read_calls == 1
        release.set()
        first.join(timeout=5.0)
        second.join(timeout=5.0)
    finally:
        release.set()
        daily_stock._LOCAL_DAILY_SYMBOL_INSPECTION_CACHE.clear()
        daily_stock._LOCAL_DAILY_SYMBOL_INSPECTION_CACHE_INFLIGHT.clear()

    assert len(results) == 2
    assert results[0] == results[1]
    assert full_read_calls == 1


def test_build_local_symbol_details_reuses_cached_symbol_inspection(
    monkeypatch,
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "AAPL.csv"
    _write_daily_csv(
        csv_path,
        [
            {
                "timestamp": "2026-01-03T00:00:00Z",
                "open": 10,
                "high": 11,
                "low": 9,
                "close": 10.5,
                "volume": 100,
            },
            {
                "timestamp": "2026-01-04T00:00:00Z",
                "open": 10.5,
                "high": 11.5,
                "low": 9.5,
                "close": 11.0,
                "volume": 120,
            },
        ],
    )

    first = daily_stock._build_local_symbol_details(
        symbols=["AAPL"],
        resolved_local_data_dir=tmp_path,
    )

    monkeypatch.setattr(
        daily_stock.pd,
        "read_csv",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("expected cached build_local_symbol_details reuse")),
    )

    second = daily_stock._build_local_symbol_details(
        symbols=["AAPL"],
        resolved_local_data_dir=tmp_path,
    )

    assert second == first


def test_inspect_local_daily_symbol_file_invalidates_cache_after_file_change(
    monkeypatch,
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "AAPL.csv"
    _write_daily_csv(
        csv_path,
        [
            {
                "timestamp": "2026-01-03T00:00:00Z",
                "open": 10,
                "high": 11,
                "low": 9,
                "close": 10.5,
                "volume": 100,
            }
        ],
    )

    first = daily_stock._inspect_local_daily_symbol_file(csv_path)
    original_read_csv = daily_stock.pd.read_csv
    read_calls = 0

    def _counting_read_csv(*args, **kwargs):
        nonlocal read_calls
        read_calls += 1
        return original_read_csv(*args, **kwargs)

    _write_daily_csv(
        csv_path,
        [
            {
                "timestamp": "2026-01-03T00:00:00Z",
                "open": 10,
                "high": 11,
                "low": 9,
                "close": 10.5,
                "volume": 100,
            },
            {
                "timestamp": "2026-01-04T00:00:00Z",
                "open": 10.5,
                "high": 11.5,
                "low": 9.5,
                "close": 11.0,
                "volume": 120,
            },
        ],
    )
    monkeypatch.setattr(daily_stock.pd, "read_csv", _counting_read_csv)

    second = daily_stock._inspect_local_daily_symbol_file(csv_path)

    assert first.row_count == 1
    assert second.row_count == 2
    assert read_calls >= 2


def test_inspect_local_daily_symbol_file_retries_when_file_changes_mid_inspection(
    monkeypatch,
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "AAPL.csv"
    original_rows = [
        {
            "timestamp": "2026-01-03T00:00:00Z",
            "open": 10,
            "high": 11,
            "low": 9,
            "close": 10.5,
            "volume": 100,
        }
    ]
    updated_rows = [
        *original_rows,
        {
            "timestamp": "2026-01-04T00:00:00Z",
            "open": 10.5,
            "high": 11.5,
            "low": 9.5,
            "close": 11.0,
            "volume": 120,
        },
    ]
    _write_daily_csv(csv_path, original_rows)

    original_read_csv = daily_stock.pd.read_csv
    full_read_calls = 0

    def _flaky_read_csv(*args, **kwargs):
        nonlocal full_read_calls
        frame = original_read_csv(*args, **kwargs)
        if kwargs.get("nrows") == 0:
            return frame
        full_read_calls += 1
        if full_read_calls == 1:
            _write_daily_csv(csv_path, updated_rows)
        return frame

    monkeypatch.setattr(daily_stock.pd, "read_csv", _flaky_read_csv)

    inspection = daily_stock._inspect_local_daily_symbol_file(csv_path)

    assert inspection.row_count == 2
    assert inspection.latest_timestamp.date().isoformat() == "2026-01-04"
    assert full_read_calls >= 2


def test_inspect_local_daily_symbol_file_raises_when_file_keeps_changing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "AAPL.csv"
    rows = [
        {
            "timestamp": "2026-01-03T00:00:00Z",
            "open": 10,
            "high": 11,
            "low": 9,
            "close": 10.5,
            "volume": 100,
        }
    ]
    _write_daily_csv(csv_path, rows)

    original_read_csv = daily_stock.pd.read_csv
    full_read_calls = 0

    def _unstable_read_csv(*args, **kwargs):
        nonlocal full_read_calls
        frame = original_read_csv(*args, **kwargs)
        if kwargs.get("nrows") == 0:
            return frame
        full_read_calls += 1
        csv_path.write_text(csv_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        return frame

    monkeypatch.setattr(daily_stock.pd, "read_csv", _unstable_read_csv)

    with pytest.raises(RuntimeError, match=r"Local daily CSV changed during inspection: .*AAPL\.csv"):
        daily_stock._inspect_local_daily_symbol_file(csv_path)

    assert full_read_calls == 2


def test_main_check_config_reports_daily_csv_with_no_valid_rows_as_invalid(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("stub", encoding="utf-8")
    monkeypatch.setattr(
        daily_stock,
        "_checkpoint_load_diagnostics",
        lambda _config: {
            "ok": True,
            "error": None,
            "primary": {
                "arch": "mlp",
                "feature_schema": "legacy_prod",
                "feature_dimension": 8,
            },
            "extras": [],
        },
    )
    _write_daily_csv(
        tmp_path / "AAPL.csv",
        [
            {
                "timestamp": "2026-01-03T00:00:00Z",
                "open": 10,
                "high": 11,
                "low": 9,
                "close": 10.5,
                "volume": 100,
            }
        ],
    )
    (tmp_path / "MSFT.csv").write_text(
        "\n".join(
            [
                "timestamp,open,high,low,close,volume",
                "not-a-time,10,11,9,10.5,100",
                "2026-01-04T00:00:00Z,10,11,9,not-a-number,100",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(SystemExit) as exc_info:
            daily_stock.main(
                [
                    "--check-config",
                    "--dry-run",
                    "--data-source",
                    "local",
                    "--symbols",
                    "AAPL",
                    "MSFT",
                    "--checkpoint",
                    str(checkpoint),
                    "--no-ensemble",
                    "--data-dir",
                    str(tmp_path),
                ]
            )

    payload = json.loads(capsys.readouterr().out)

    assert exc_info.value.code == 1
    assert payload["ready"] is False
    assert payload["usable_symbols"] == ["AAPL"]
    assert payload["symbol_details"]["MSFT"] == {
        "status": "invalid",
        "file_path": str((tmp_path / "MSFT.csv").resolve()),
        "local_data_date": None,
        "row_count": None,
        "reason": "Daily frame has no valid timestamped OHLCV rows",
    }
    assert any("Unreadable local daily data file(s):" in error for error in payload["errors"])
    assert any("Repair unreadable local daily CSVs for: MSFT." == step for step in payload["next_steps"])
    assert not [
        warning
        for warning in caught
        if "Could not infer format" in str(warning.message)
    ]


def test_main_check_config_warns_when_symbol_inputs_are_normalized(monkeypatch, tmp_path: Path, capsys) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("stub", encoding="utf-8")
    monkeypatch.setattr(
        daily_stock,
        "_checkpoint_load_diagnostics",
        lambda _config: {"ok": True, "error": None, "primary": {"arch": "mlp"}, "extras": []},
    )
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
    monkeypatch.setattr(
        daily_stock,
        "_checkpoint_load_diagnostics",
        lambda _config: {"ok": True, "error": None, "primary": {"arch": "mlp"}, "extras": []},
    )

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
    monkeypatch.setattr(
        daily_stock,
        "_checkpoint_load_diagnostics",
        lambda _config: {"ok": True, "error": None, "primary": {"arch": "mlp"}, "extras": []},
    )

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
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("stub", encoding="utf-8")
    monkeypatch.setattr(
        daily_stock,
        "_checkpoint_load_diagnostics",
        lambda _config: {"ok": True, "error": None, "primary": {"arch": "mlp"}, "extras": []},
    )
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
    assert payload["server_url_source"] == "default"
    assert payload["server_url_transport"] == "http"
    assert payload["server_url_scope"] == "loopback"
    assert payload["server_url_security"] == "loopback_http"
    assert any("built-in loopback default server URL" in warning for warning in payload["warnings"])
    assert any("paper default server account" in warning for warning in payload["warnings"])


def test_main_check_config_rejects_insecure_remote_live_trading_server_url(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("stub", encoding="utf-8")
    monkeypatch.setattr(
        daily_stock,
        "_checkpoint_load_diagnostics",
        lambda _config: {"ok": True, "error": None, "primary": {"arch": "mlp"}, "extras": []},
    )
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

    with pytest.raises(SystemExit) as exc_info:
        daily_stock.main(
            [
                "--check-config",
                "--live",
                "--execution-backend",
                "trading_server",
                "--server-url",
                "http://server.internal:8050",
                "--server-account",
                "live-account",
                "--server-bot-id",
                "daily-bot",
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

    assert exc_info.value.code == 1
    assert payload["ready"] is False
    assert payload["resolved_server_url"] == "http://server.internal:8050"
    assert payload["server_url_source"] == "cli"
    assert payload["server_url_transport"] == "http"
    assert payload["server_url_scope"] == "remote"
    assert payload["server_url_security"] == "insecure_remote_http"
    assert any(
        "Live trading_server requires an https URL unless it targets loopback" in error
        for error in payload["errors"]
    )
    assert any(
        "Update --server-url to an https:// trading server URL for live remote execution"
        in step
        for step in payload["next_steps"]
    )


def test_main_check_config_rejects_insecure_remote_env_live_trading_server_url(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("stub", encoding="utf-8")
    monkeypatch.setenv("TRADING_SERVER_URL", "http://server.internal:8050")
    monkeypatch.setattr(
        daily_stock,
        "_checkpoint_load_diagnostics",
        lambda _config: {"ok": True, "error": None, "primary": {"arch": "mlp"}, "extras": []},
    )
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

    with pytest.raises(SystemExit) as exc_info:
        daily_stock.main(
            [
                "--check-config",
                "--live",
                "--execution-backend",
                "trading_server",
                "--server-account",
                "live-account",
                "--server-bot-id",
                "daily-bot",
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

    assert exc_info.value.code == 1
    assert payload["server_url_source"] == "env"
    assert payload["resolved_server_url"] == "http://server.internal:8050"
    assert any(
        "Update TRADING_SERVER_URL to an https:// trading server URL for live remote execution"
        in step
        for step in payload["next_steps"]
    )


def test_checkpoint_load_diagnostics_accepts_mixed_encoder_norm_ensemble(tmp_path: Path) -> None:
    primary_ckpt = tmp_path / "primary_old.pt"
    extra_ckpt = tmp_path / "extra_new.pt"
    _write_mlp_policy_checkpoint(
        primary_ckpt,
        num_symbols=2,
        num_actions=3,
        hot_action=1,
        use_encoder_norm=False,
    )
    _write_mlp_policy_checkpoint(
        extra_ckpt,
        num_symbols=2,
        num_actions=3,
        hot_action=2,
        use_encoder_norm=True,
    )

    config = daily_stock.CliRuntimeConfig(
        paper=True,
        symbols=["AAPL", "MSFT"],
        checkpoint=str(primary_ckpt),
        extra_checkpoints=[str(extra_ckpt)],
        data_dir=str(tmp_path),
        data_source="local",
        allocation_pct=12.5,
        execution_backend="alpaca",
        server_account=daily_stock.DEFAULT_SERVER_PAPER_ACCOUNT,
        server_bot_id=daily_stock.DEFAULT_SERVER_PAPER_BOT_ID,
        server_url=None,
        dry_run=True,
        backtest=False,
        backtest_days=60,
        backtest_starting_cash=daily_stock.DEFAULT_BACKTEST_STARTING_CASH,
        daemon=False,
        compare_server_parity=False,
    )

    payload = daily_stock._checkpoint_load_diagnostics(config)

    assert payload["ok"] is True
    assert payload["error"] is None
    assert payload["primary"]["arch"] == "mlp"
    assert payload["primary"]["feature_schema"] == "rsi_v5"
    assert payload["primary"]["feature_dimension"] == daily_stock.daily_feature_dimension("rsi_v5")
    assert len(payload["extras"]) == 1


def test_main_check_config_reports_checkpoint_load_failure(monkeypatch, tmp_path: Path, capsys) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("stub", encoding="utf-8")
    monkeypatch.setattr(
        daily_stock,
        "_checkpoint_load_diagnostics",
        lambda _config: {"ok": False, "error": "RuntimeError: boom", "primary": None, "extras": []},
    )

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
    assert payload["checkpoint_load"]["ok"] is False
    assert any("Checkpoint load failed: RuntimeError: boom" in error for error in payload["errors"])


def test_main_check_config_warns_when_unsafe_checkpoint_loading_is_enabled(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("stub", encoding="utf-8")
    monkeypatch.setattr(
        daily_stock,
        "_checkpoint_load_diagnostics",
        lambda _config: {"ok": True, "error": None, "primary": {"arch": "mlp"}, "extras": []},
    )
    monkeypatch.setattr(daily_stock, "_resolve_alpaca_credentials", lambda *, paper: ("paper-key", "paper-secret"))

    daily_stock.main(
        [
            "--check-config",
            "--checkpoint",
            str(checkpoint),
            "--no-ensemble",
            "--symbols",
            "AAPL",
            "--allow-unsafe-checkpoint-loading",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert payload["ready"] is True
    assert payload["allow_unsafe_checkpoint_loading"] is True
    assert any("unsafe checkpoint loading enabled" in warning for warning in payload["warnings"])


def test_main_check_config_rejects_out_of_range_min_open_confidence(tmp_path: Path, capsys) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("stub", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        daily_stock.main(
            [
                "--check-config",
                "--checkpoint",
                str(checkpoint),
                "--no-ensemble",
                "--symbols",
                "AAPL",
                "--min-open-confidence",
                "1.5",
            ]
        )

    payload = json.loads(capsys.readouterr().out)

    assert exc_info.value.code == 1
    assert payload["ready"] is False
    assert any("--min-open-confidence must be between 0 and 1" == error for error in payload["errors"])


def test_main_fails_fast_with_preflight_message_when_checkpoint_is_missing(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    called = False

    def _fake_run_once(**kwargs):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(daily_stock, "run_once", _fake_run_once)

    missing = tmp_path / "missing.pt"

    with pytest.raises(SystemExit) as exc_info:
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

    stderr = capsys.readouterr().err

    assert exc_info.value.code == 1
    assert called is False
    assert "Daily stock RL setup is not ready." in stderr
    assert "Missing checkpoint file(s):" in stderr
    assert "--check-config" in stderr


def test_main_check_config_text_emits_human_failure_summary(tmp_path: Path, capsys) -> None:
    missing = tmp_path / "missing.pt"

    with pytest.raises(SystemExit) as exc_info:
        daily_stock.main(
            [
                "--check-config",
                "--check-config-text",
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

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exc_info.value.code == 1
    assert payload["ready"] is False
    assert "Daily stock RL setup is not ready." in captured.err
    assert "Missing checkpoint file(s):" in captured.err
    assert "--check-config" in captured.err


def test_main_check_config_text_reports_resolved_missing_local_data_dir(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("stub", encoding="utf-8")
    missing_dir = tmp_path / "missing_data"
    monkeypatch.setattr(
        daily_stock,
        "_checkpoint_load_diagnostics",
        lambda _config: {
            "ok": True,
            "error": None,
            "primary": {
                "arch": "mlp",
                "feature_schema": "legacy_prod",
                "feature_dimension": 8,
            },
            "extras": [],
        },
    )

    with pytest.raises(SystemExit) as exc_info:
        daily_stock.main(
            [
                "--check-config",
                "--check-config-text",
                "--dry-run",
                "--data-source",
                "local",
                "--symbols",
                "AAPL",
                "--checkpoint",
                str(checkpoint),
                "--no-ensemble",
                "--data-dir",
                str(missing_dir),
            ]
        )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exc_info.value.code == 1
    assert (
        f"Resolved local data dir: {missing_dir.resolve()} (missing)"
        in captured.err
    )
    assert "Local data dir source: requested directory" in captured.err
    assert payload["resolved_local_data_dir_source"] == "requested"


def test_main_check_config_text_reports_requested_and_resolved_local_data_dir(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("stub", encoding="utf-8")
    requested_dir = tmp_path / "daily_data"
    resolved_dir = requested_dir / "train"
    resolved_dir.mkdir(parents=True)
    _write_daily_csv(
        resolved_dir / "AAPL.csv",
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
    monkeypatch.setattr(
        daily_stock,
        "_checkpoint_load_diagnostics",
        lambda _config: {
            "ok": True,
            "error": None,
            "primary": {
                "arch": "mlp",
                "feature_schema": "legacy_prod",
                "feature_dimension": 8,
            },
            "extras": [],
        },
    )

    daily_stock.main(
        [
            "--check-config",
            "--check-config-text",
            "--dry-run",
            "--data-source",
            "local",
            "--symbols",
            "AAPL",
            "--checkpoint",
            str(checkpoint),
            "--no-ensemble",
            "--data-dir",
            str(requested_dir),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert f"Requested local data dir: {requested_dir.resolve()}" in captured.err
    assert f"Resolved local data dir: {resolved_dir.resolve()}" in captured.err
    assert "Local data dir source: auto-selected nested train/ directory" in captured.err
    assert payload["resolved_local_data_dir_source"] == "nested_train"


def test_main_check_config_uses_repo_relative_data_dir_outside_repo_cwd(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    data_root = repo_root / "daily_data"
    data_root.mkdir()
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("stub", encoding="utf-8")
    _write_daily_csv(
        data_root / "AAPL.csv",
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
    monkeypatch.setattr(daily_stock, "REPO", repo_root)
    monkeypatch.setattr(
        daily_stock,
        "_checkpoint_load_diagnostics",
        lambda _config: {
            "ok": True,
            "error": None,
            "primary": {
                "arch": "mlp",
                "feature_schema": "legacy_prod",
                "feature_dimension": 8,
            },
            "extras": [],
        },
    )
    outside_cwd = tmp_path / "outside"
    outside_cwd.mkdir()
    monkeypatch.chdir(outside_cwd)

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
            "daily_data",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert payload["ready"] is True
    assert payload["errors"] == []
    assert payload["data_dir_exists"] is True
    assert payload["requested_local_data_dir"] == str(data_root.resolve())
    assert payload["resolved_local_data_dir"] == str(data_root.resolve())
    assert payload["resolved_local_data_dir_source"] == "requested"
    assert payload["usable_symbols"] == ["AAPL"]
    assert payload["local_data_status_counts"] == {
        "usable": 1,
        "stale": 0,
        "missing": 0,
        "invalid": 0,
    }


def test_local_data_dir_context_reports_requested_and_nested_train_source(tmp_path: Path) -> None:
    requested_dir = tmp_path / "daily_data"
    resolved_dir = requested_dir / "train"
    resolved_dir.mkdir(parents=True)
    _write_daily_csv(
        resolved_dir / "AAPL.csv",
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

    context = daily_stock._local_data_dir_context(requested_dir, ["AAPL"])

    assert context["resolved_local_data_dir_source"] is daily_stock.LocalDataDirResolutionSource.NESTED_TRAIN
    assert context == {
        "requested_local_data_dir": str(requested_dir.resolve()),
        "resolved_local_data_dir": str(resolved_dir.resolve()),
        "resolved_local_data_dir_source": "nested_train",
    }


def test_coerce_local_data_dir_context_accepts_only_complete_payloads(tmp_path: Path) -> None:
    requested_dir = tmp_path / "requested"
    resolved_dir = tmp_path / "resolved"

    assert daily_stock._coerce_local_data_dir_context(
        {
            "requested_local_data_dir": str(requested_dir.resolve()),
            "resolved_local_data_dir": str(resolved_dir.resolve()),
            "resolved_local_data_dir_source": "requested",
        }
    ) == {
        "requested_local_data_dir": str(requested_dir.resolve()),
        "resolved_local_data_dir": str(resolved_dir.resolve()),
        "resolved_local_data_dir_source": daily_stock.LocalDataDirResolutionSource.REQUESTED,
    }
    assert daily_stock._coerce_local_data_dir_context(
        {
            "requested_local_data_dir": str(requested_dir.resolve()),
            "resolved_local_data_dir": str(resolved_dir.resolve()),
            "resolved_local_data_dir_source": "bogus",
        }
    ) is None
    assert daily_stock._coerce_local_data_dir_context(
        {
            "requested_local_data_dir": "",
            "resolved_local_data_dir": str(resolved_dir.resolve()),
            "resolved_local_data_dir_source": "requested",
        }
    ) is None


def test_main_check_config_text_includes_checkpoint_load_error_details(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("stub", encoding="utf-8")
    monkeypatch.setattr(
        daily_stock,
        "_checkpoint_load_diagnostics",
        lambda _config: {"ok": False, "error": "RuntimeError: boom", "primary": None, "extras": []},
    )

    with pytest.raises(SystemExit) as exc_info:
        daily_stock.main(
            [
                "--check-config",
                "--check-config-text",
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

    captured = capsys.readouterr()

    assert exc_info.value.code == 1
    assert "Checkpoint load:" in captured.err
    assert "- error: RuntimeError: boom" in captured.err


def test_main_check_config_summary_emits_compact_failure_line(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("stub", encoding="utf-8")
    monkeypatch.setattr(
        daily_stock,
        "_checkpoint_load_diagnostics",
        lambda _config: {"ok": False, "error": "RuntimeError: boom", "primary": None, "extras": []},
    )

    with pytest.raises(SystemExit) as exc_info:
        daily_stock.main(
            [
                "--check-config",
                "--check-config-summary",
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

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exc_info.value.code == 1
    assert payload["ready"] is False
    assert captured.err.strip() == (
        "NOT READY | single-position | symbols=0/1 usable | checkpoint=error | "
        "local_issues=missing:1 | "
        "errors=2 | first_error=Checkpoint load failed: RuntimeError: boom"
    )


def test_main_check_config_summary_includes_local_issue_counts(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("stub", encoding="utf-8")
    monkeypatch.setattr(
        daily_stock,
        "_checkpoint_load_diagnostics",
        lambda _config: {
            "ok": True,
            "error": None,
            "primary": {
                "arch": "mlp",
                "feature_schema": "legacy_prod",
                "feature_dimension": 8,
            },
            "extras": [],
        },
    )
    _write_daily_csv(
        tmp_path / "AAPL.csv",
        [
            {
                "timestamp": "2026-01-03T00:00:00Z",
                "open": 10,
                "high": 11,
                "low": 9,
                "close": 10.5,
                "volume": 100,
            }
        ],
    )
    _write_daily_csv(
        tmp_path / "MSFT.csv",
        [
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "open": 20,
                "high": 21,
                "low": 19,
                "close": 20.5,
                "volume": 200,
            }
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        daily_stock.main(
            [
                "--check-config",
                "--check-config-summary",
                "--dry-run",
                "--data-source",
                "local",
                "--symbols",
                "AAPL",
                "MSFT",
                "NVDA",
                "--checkpoint",
                str(checkpoint),
                "--no-ensemble",
                "--data-dir",
                str(tmp_path),
            ]
        )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exc_info.value.code == 1
    assert payload["local_data_status_counts"] == {
        "usable": 1,
        "stale": 1,
        "missing": 1,
        "invalid": 0,
    }
    assert "local_issues=stale:1,missing:1" in captured.err.strip()


def test_main_check_config_summary_squashes_multiline_error_text(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("stub", encoding="utf-8")
    monkeypatch.setattr(
        daily_stock,
        "_checkpoint_load_diagnostics",
        lambda _config: {
            "ok": False,
            "error": "RuntimeError: boom\nsecondary detail",
            "primary": None,
            "extras": [],
        },
    )

    with pytest.raises(SystemExit):
        daily_stock.main(
            [
                "--check-config",
                "--check-config-summary",
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

    stderr = capsys.readouterr().err.strip()

    assert "\n" not in stderr
    assert stderr.endswith(
        "first_error=Checkpoint load failed: RuntimeError: boom secondary detail"
    )


def test_main_formats_run_once_failure_as_operator_error(monkeypatch, tmp_path: Path, capsys) -> None:
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
    monkeypatch.setattr(
        daily_stock,
        "_checkpoint_load_diagnostics",
        lambda _config: {"ok": True, "error": None, "primary": {"arch": "mlp"}, "extras": []},
    )

    def _boom(**_kwargs):
        exc = RuntimeError("quote provider boom")
        exc.add_note("run_once stage: load_latest_quotes")
        exc.add_note("run_once context: bar_data_source=alpaca; quote_data_source=mixed_fallback")
        exc.add_note("quote provider returned malformed payload")
        raise exc

    monkeypatch.setattr(daily_stock, "run_once", _boom)

    with pytest.raises(SystemExit) as exc_info:
        daily_stock.main(
            [
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

    stderr = capsys.readouterr().err

    assert exc_info.value.code == 1
    assert "Daily stock RL run failed during load latest quotes." in stderr
    assert "Error: RuntimeError: quote provider boom" in stderr
    assert f"Resolved local data dir: {tmp_path.resolve()}" in stderr
    assert "Local data dir source: requested directory" in stderr
    assert "Latest context: bar_data_source=alpaca; quote_data_source=mixed_fallback" in stderr
    assert (
        "Check config: python trade_daily_stock_prod.py --check-config --check-config-text "
        "--once --paper --checkpoint "
        in stderr
    )
    assert "--check-config-text" in stderr
    assert "Try a safe dry run: python trade_daily_stock_prod.py --once --paper --dry-run" in stderr
    assert "Additional context:" in stderr
    assert "- quote provider returned malformed payload" in stderr


def test_main_formats_run_once_failure_even_when_local_data_dir_context_lookup_fails(
    monkeypatch,
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
    monkeypatch.setattr(
        daily_stock,
        "_checkpoint_load_diagnostics",
        lambda _config: {"ok": True, "error": None, "primary": {"arch": "mlp"}, "extras": []},
    )
    original_local_data_dir_context = daily_stock._local_data_dir_context

    def _boom(**_kwargs):
        exc = RuntimeError("quote provider boom")
        exc.add_note("run_once stage: load_latest_quotes")
        raise exc

    monkeypatch.setattr(daily_stock, "run_once", _boom)
    calls = {"count": 0}

    def _flaky_local_data_dir_context(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return original_local_data_dir_context(*args, **kwargs)
        raise RuntimeError("context lookup boom")

    monkeypatch.setattr(
        daily_stock,
        "_local_data_dir_context",
        _flaky_local_data_dir_context,
    )

    with pytest.raises(SystemExit) as exc_info:
        daily_stock.main(
            [
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

    stderr = capsys.readouterr().err

    assert exc_info.value.code == 1
    assert "Daily stock RL run failed during load latest quotes." in stderr
    assert "Error: RuntimeError: quote provider boom" in stderr
    assert "Local data dir context unavailable: RuntimeError: context lookup boom" in stderr
    assert "--check-config-text" in stderr


def test_main_fails_fast_with_local_data_health_summary(monkeypatch, tmp_path: Path, capsys) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("stub", encoding="utf-8")
    monkeypatch.setattr(
        daily_stock,
        "_checkpoint_load_diagnostics",
        lambda _config: {"ok": True, "error": None, "primary": {"arch": "mlp"}, "extras": []},
    )
    _write_daily_csv(
        tmp_path / "AAPL.csv",
        [
            {
                "timestamp": "2026-01-03T00:00:00Z",
                "open": 10,
                "high": 11,
                "low": 9,
                "close": 10.5,
                "volume": 100,
            }
        ],
    )
    _write_daily_csv(
        tmp_path / "MSFT.csv",
        [
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "open": 20,
                "high": 21,
                "low": 19,
                "close": 20.5,
                "volume": 200,
            }
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        daily_stock.main(
            [
                "--dry-run",
                "--data-source",
                "local",
                "--symbols",
                "AAPL",
                "MSFT",
                "NVDA",
                "--checkpoint",
                str(checkpoint),
                "--no-ensemble",
                "--data-dir",
                str(tmp_path),
            ]
        )

    stderr = capsys.readouterr().err

    assert exc_info.value.code == 1
    assert "Daily stock RL setup is not ready." in stderr
    assert "Local data health:" in stderr
    assert "- usable symbols: 2/3" in stderr
    assert "- latest local data date: 2026-01-03" in stderr
    assert "- stale symbols: MSFT (2026-01-01)" in stderr
    assert "- missing symbols: NVDA" in stderr
    assert "- Refresh stale local daily CSVs for: MSFT." in stderr
    assert "- Add the missing local daily CSV files under the resolved data directory." in stderr


def test_main_failure_summary_truncates_large_missing_symbol_lists(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("stub", encoding="utf-8")
    monkeypatch.setattr(
        daily_stock,
        "_checkpoint_load_diagnostics",
        lambda _config: {
            "ok": True,
            "error": None,
            "primary": {
                "arch": "mlp",
                "feature_schema": "legacy_prod",
                "feature_dimension": 8,
            },
            "extras": [],
        },
    )
    _write_daily_csv(
        tmp_path / "AAPL.csv",
        [
            {
                "timestamp": "2026-01-03T00:00:00Z",
                "open": 10,
                "high": 11,
                "low": 9,
                "close": 10.5,
                "volume": 100,
            }
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        daily_stock.main(
            [
                "--dry-run",
                "--data-source",
                "local",
                "--symbols",
                "AAPL",
                "MSFT",
                "NVDA",
                "GOOG",
                "META",
                "TSLA",
                "AMZN",
                "--checkpoint",
                str(checkpoint),
                "--no-ensemble",
                "--data-dir",
                str(tmp_path),
            ]
        )

    stderr = capsys.readouterr().err

    assert exc_info.value.code == 1
    assert (
        "- missing symbols: MSFT, NVDA, GOOG, META, TSLA (+1 more)"
    ) in stderr


def test_main_rejects_compare_server_parity_without_backtest(tmp_path: Path, capsys) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("stub", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
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

    stderr = capsys.readouterr().err
    assert exc_info.value.code == 1
    assert "--compare-server-parity requires --backtest" in stderr
    assert "Checkpoint load failed" not in stderr


def test_main_formats_trading_server_setup_details_for_insecure_remote_url(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("stub", encoding="utf-8")
    monkeypatch.setattr(
        daily_stock,
        "_checkpoint_load_diagnostics",
        lambda _config: {"ok": True, "error": None, "primary": {"arch": "mlp"}, "extras": []},
    )
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

    with pytest.raises(SystemExit) as exc_info:
        daily_stock.main(
            [
                "--live",
                "--execution-backend",
                "trading_server",
                "--server-url",
                "http://server.internal:8050",
                "--server-account",
                "live-account",
                "--server-bot-id",
                "daily-bot",
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

    stderr = capsys.readouterr().err

    assert exc_info.value.code == 1
    assert "Trading server config:" in stderr
    assert "- resolved server URL: http://server.internal:8050" in stderr
    assert "- server URL source: --server-url" in stderr
    assert "- server URL security: remote http rejected in live mode" in stderr
    assert (
        "- Update --server-url to an https:// trading server URL for live remote execution "
        "(current: http://server.internal:8050)."
    ) in stderr


def test_main_rejects_non_positive_backtest_days(tmp_path: Path, capsys) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("stub", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
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

    stderr = capsys.readouterr().err
    assert exc_info.value.code == 1
    assert "--backtest-days must be at least 1" in stderr
    assert "Checkpoint load failed" not in stderr


def test_main_rejects_non_positive_backtest_starting_cash(tmp_path: Path, capsys) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("stub", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
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

    stderr = capsys.readouterr().err
    assert exc_info.value.code == 1
    assert "--backtest-starting-cash must be positive" in stderr
    assert "Checkpoint load failed" not in stderr


def test_main_rejects_live_backtest_before_live_safety_checks(monkeypatch, tmp_path: Path, capsys) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("stub", encoding="utf-8")

    def _unexpected_live_guard(*_args, **_kwargs):
        raise AssertionError("live safety guard should not run for backtests")

    monkeypatch.setattr(daily_stock, "require_explicit_live_trading_enable", _unexpected_live_guard)
    monkeypatch.setattr(daily_stock, "acquire_alpaca_account_lock", _unexpected_live_guard)

    with pytest.raises(SystemExit) as exc_info:
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

    stderr = capsys.readouterr().err
    assert exc_info.value.code == 1
    assert "--backtest is local-only; omit --live" in stderr
    assert "Checkpoint load failed" not in stderr


def test_main_passes_backtest_starting_cash_to_run_backtest(monkeypatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("stub", encoding="utf-8")
    captured: dict[str, object] = {}

    def _fake_run_backtest(**kwargs):
        captured.update(kwargs)
        return {"total_return": 0.0}

    monkeypatch.setattr(daily_stock, "run_backtest", _fake_run_backtest)
    monkeypatch.setattr(
        daily_stock,
        "_checkpoint_load_diagnostics",
        lambda _config: {"ok": True, "error": None, "primary": {"arch": "mlp"}, "extras": []},
    )

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


def test_main_passes_backtest_buying_power_multiplier_to_run_backtest(monkeypatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("stub", encoding="utf-8")
    captured: dict[str, object] = {}

    def _fake_run_backtest(**kwargs):
        captured.update(kwargs)
        return {"total_return": 0.0}

    monkeypatch.setattr(daily_stock, "run_backtest", _fake_run_backtest)
    monkeypatch.setattr(
        daily_stock,
        "_checkpoint_load_diagnostics",
        lambda _config: {"ok": True, "error": None, "primary": {"arch": "mlp"}, "extras": []},
    )

    daily_stock.main(
        [
            "--backtest",
            "--backtest-buying-power-multiplier",
            "2",
            "--data-source",
            "local",
            "--symbols",
            "AAPL",
            "--checkpoint",
            str(checkpoint),
            "--no-ensemble",
        ]
    )

    assert captured["buying_power_multiplier"] == 2.0


def test_main_passes_backtest_offsets_to_run_backtest(monkeypatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("stub", encoding="utf-8")
    captured: dict[str, object] = {}

    def _fake_run_backtest(**kwargs):
        captured.update(kwargs)
        return {"total_return": 0.0}

    monkeypatch.setattr(daily_stock, "run_backtest", _fake_run_backtest)
    monkeypatch.setattr(
        daily_stock,
        "_checkpoint_load_diagnostics",
        lambda _config: {"ok": True, "error": None, "primary": {"arch": "mlp"}, "extras": []},
    )

    daily_stock.main(
        [
            "--backtest",
            "--backtest-entry-offset-bps",
            "-5",
            "--backtest-exit-offset-bps",
            "25",
            "--data-source",
            "local",
            "--symbols",
            "AAPL",
            "--checkpoint",
            str(checkpoint),
            "--no-ensemble",
        ]
    )

    assert captured["entry_offset_bps"] == -5.0
    assert captured["exit_offset_bps"] == 25.0


def test_compare_backtest_to_trading_server_passes_allocation_pct(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_run_backtest(**kwargs):
        captured["legacy"] = kwargs
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "trades": 0.0,
        }

    def _fake_run_backtest_via_trading_server(**kwargs):
        captured["server"] = kwargs
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "trades": 0.0,
        }

    monkeypatch.setattr(daily_stock, "run_backtest", _fake_run_backtest)
    monkeypatch.setattr(daily_stock, "run_backtest_via_trading_server", _fake_run_backtest_via_trading_server)

    daily_stock.compare_backtest_to_trading_server(
        checkpoint="model.pt",
        symbols=["AAPL"],
        data_dir="trainingdata",
        days=20,
        allocation_pct=12.5,
        allocation_sizing_mode="confidence_scaled",
        starting_cash=25_000.0,
        buying_power_multiplier=2.0,
    )

    assert captured["legacy"]["allocation_pct"] == 12.5
    assert captured["server"]["allocation_pct"] == 12.5
    assert captured["legacy"]["allocation_sizing_mode"] == "confidence_scaled"
    assert captured["server"]["allocation_sizing_mode"] == "confidence_scaled"
    assert captured["legacy"]["starting_cash"] == 25_000.0
    assert captured["server"]["starting_cash"] == 25_000.0
    assert captured["legacy"]["buying_power_multiplier"] == 2.0
    assert captured["server"]["buying_power_multiplier"] == 2.0


def test_compare_backtest_to_trading_server_passes_extra_checkpoints(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_run_backtest(**kwargs):
        captured["legacy"] = kwargs
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "trades": 0.0,
        }

    def _fake_run_backtest_via_trading_server(**kwargs):
        captured["server"] = kwargs
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "trades": 0.0,
        }

    monkeypatch.setattr(daily_stock, "run_backtest", _fake_run_backtest)
    monkeypatch.setattr(daily_stock, "run_backtest_via_trading_server", _fake_run_backtest_via_trading_server)

    extras = ["a.pt", "b.pt"]
    daily_stock.compare_backtest_to_trading_server(
        checkpoint="model.pt",
        symbols=["AAPL"],
        data_dir="trainingdata",
        days=20,
        extra_checkpoints=extras,
    )

    assert captured["legacy"]["extra_checkpoints"] == extras
    assert captured["server"]["extra_checkpoints"] == extras


def test_preflight_allows_multi_position_outside_backtest(monkeypatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("stub", encoding="utf-8")
    monkeypatch.setattr(
        daily_stock,
        "_checkpoint_load_diagnostics",
        lambda config: {
            "ok": True,
            "primary": {"path": config.checkpoint},
            "extras": [],
            "error": None,
        },
    )
    args = daily_stock.parse_args(
        [
            "--checkpoint",
            str(checkpoint),
            "--multi-position",
            "2",
            "--no-ensemble",
        ]
    )
    config = daily_stock._resolve_runtime_config(args)

    payload = daily_stock._preflight_config_payload(config)

    assert payload["ready"] is True
    assert "--multi-position is currently implemented for --backtest only" not in payload["errors"]


def test_preflight_rejects_negative_allocation_pct(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("stub", encoding="utf-8")
    args = daily_stock.parse_args(
        [
            "--checkpoint",
            str(checkpoint),
            "--allocation-pct",
            "-1",
            "--no-ensemble",
        ]
    )
    config = daily_stock._resolve_runtime_config(args)

    payload = daily_stock._preflight_config_payload(config)

    assert payload["ready"] is False
    assert "--allocation-pct must be >= 0" in payload["errors"]


def test_preflight_rejects_compare_server_parity_with_custom_backtest_offsets(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("stub", encoding="utf-8")
    args = daily_stock.parse_args(
        [
            "--checkpoint",
            str(checkpoint),
            "--backtest",
            "--compare-server-parity",
            "--backtest-entry-offset-bps",
            "-5",
            "--no-ensemble",
        ]
    )
    config = daily_stock._resolve_runtime_config(args)

    payload = daily_stock._preflight_config_payload(config)

    assert payload["ready"] is False
    assert "--compare-server-parity does not support custom backtest entry/exit offsets" in payload["errors"]


def test_preflight_allows_compare_server_parity_with_multi_position(
    monkeypatch, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("stub", encoding="utf-8")
    monkeypatch.setattr(
        daily_stock,
        "_checkpoint_load_diagnostics",
        lambda config: {
            "ok": True,
            "primary": {"path": config.checkpoint},
            "extras": [],
            "error": None,
        },
    )
    args = daily_stock.parse_args(
        [
            "--checkpoint",
            str(checkpoint),
            "--backtest",
            "--compare-server-parity",
            "--multi-position",
            "2",
            "--no-ensemble",
        ]
    )
    config = daily_stock._resolve_runtime_config(args)

    payload = daily_stock._preflight_config_payload(config)

    assert payload["ready"] is True
    assert payload["checkpoint_load"]["ok"] is True
    assert "--compare-server-parity does not support --multi-position" not in payload["errors"]


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


def test_run_backtest_rejects_non_positive_buying_power_multiplier() -> None:
    with pytest.raises(ValueError, match="buying_power_multiplier must be positive"):
        daily_stock.run_backtest(
            checkpoint="unused.pt",
            symbols=["AAPL"],
            data_dir="unused",
            days=20,
            buying_power_multiplier=0.0,
        )


def test_run_backtest_multi_position_rebalances_top_k_portfolio(monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01T00:00:00Z", periods=25, freq="D"),
            "open": np.linspace(100.0, 124.0, 25),
            "high": np.linspace(101.0, 125.0, 25),
            "low": np.linspace(99.0, 123.0, 25),
            "close": np.linspace(100.5, 124.5, 25),
            "volume": np.linspace(1_000.0, 1_024.0, 25),
        }
    )

    class _StaticPolicy:
        def __call__(self, obs_t):  # noqa: ANN001
            return (
                torch.tensor([[0.0, 4.0, 3.5]], dtype=torch.float32),
                torch.tensor([0.5], dtype=torch.float32),
            )

    class _FakeTrader:
        def __init__(self) -> None:
            self.device = "cpu"
            self.SYMBOLS = ["AAPL", "MSFT"]
            self.num_symbols = len(self.SYMBOLS)
            self.per_symbol_actions = 1
            self.obs_size = self.num_symbols * 16 + 5 + self.num_symbols
            self.num_actions = 1 + self.num_symbols
            self.policy = _StaticPolicy()
            self.cash = 0.0
            self.current_position = None
            self.position_qty = 0.0
            self.entry_price = 0.0

        def build_observation(self, features, prices):  # noqa: ANN001, ARG002
            return np.zeros((self.obs_size,), dtype=np.float32)

        def apply_action_constraints(self, logits):  # noqa: ANN001
            return logits

        def step_day(self):
            return None

    monkeypatch.setattr(
        daily_stock,
        "load_local_daily_frames",
        lambda symbols, data_dir, min_days=daily_stock.DEFAULT_DAILY_FRAME_MIN_DAYS: {
            "AAPL": frame.copy(),
            "MSFT": frame.copy(),
        },
    )
    monkeypatch.setattr(daily_stock, "_load_cached_daily_trader", lambda *args, **kwargs: _FakeTrader())

    results = daily_stock.run_backtest(
        checkpoint="unused.pt",
        symbols=["AAPL", "MSFT"],
        data_dir="unused",
        days=5,
        allocation_pct=100.0,
        multi_position=2,
    )

    assert results["trades"] == pytest.approx(10.0)
    assert results["total_return"] > 0.0


def test_run_backtest_uses_named_history_floor(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_load_local_daily_frames(symbols, data_dir, min_days=daily_stock.DEFAULT_DAILY_FRAME_MIN_DAYS):
        captured["symbols"] = list(symbols)
        captured["data_dir"] = data_dir
        captured["min_days"] = min_days
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01T00:00:00Z", periods=25, freq="D"),
                "open": np.linspace(100.0, 124.0, 25),
                "high": np.linspace(101.0, 125.0, 25),
                "low": np.linspace(99.0, 123.0, 25),
                "close": np.linspace(100.5, 124.5, 25),
                "volume": np.linspace(1_000.0, 1_024.0, 25),
            }
        )
        return {"AAPL": frame}

    class _FakeTrader:
        def __init__(
            self,
            checkpoint,
            device="cpu",
            long_only=True,
            symbols=None,
            allow_unsafe_checkpoint_loading=False,
        ):
            self.SYMBOLS = list(symbols or [])
            self.num_symbols = len(self.SYMBOLS)
            self.obs_size = 16 * max(1, self.num_symbols)
            self.num_actions = 1 + self.num_symbols
            self.cash = 0.0
            self.current_position = None
            self.position_qty = 0.0
            self.entry_price = 0.0

        def get_daily_signal(self, frames, prices):
            return SimpleNamespace(action="flat", symbol=None, direction=None, confidence=0.0, value_estimate=0.0)

        def update_state(self, action, price, symbol):
            return None

        def step_day(self):
            return None

    monkeypatch.setattr(daily_stock, "load_local_daily_frames", _fake_load_local_daily_frames)
    monkeypatch.setattr(daily_stock, "DailyPPOTrader", _FakeTrader)

    daily_stock.run_backtest(
        checkpoint="unused.pt",
        symbols=["AAPL"],
        data_dir="unused",
        days=20,
    )

    assert captured["symbols"] == ["AAPL"]
    assert captured["data_dir"] == "unused"
    assert captured["min_days"] == 20 + daily_stock.DEFAULT_DAILY_FRAME_MIN_DAYS


def test_run_backtest_single_position_precomputes_feature_history(monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01T00:00:00Z", periods=25, freq="D"),
            "open": np.linspace(100.0, 124.0, 25),
            "high": np.linspace(101.0, 125.0, 25),
            "low": np.linspace(99.0, 123.0, 25),
            "close": np.linspace(100.5, 124.5, 25),
            "volume": np.linspace(1_000.0, 1_024.0, 25),
        }
    )
    history_calls = {"count": 0}

    class _FakeTrader:
        def __init__(self) -> None:
            self.device = "cpu"
            self.SYMBOLS = ["AAPL", "MSFT"]
            self.num_symbols = len(self.SYMBOLS)
            self.num_actions = 1 + self.num_symbols
            self.cash = 0.0
            self.current_position = None
            self.position_qty = 0.0
            self.entry_price = 0.0
            self.hold_days = 0
            self.hold_hours = 0
            self.step = 0
            self.max_steps = 90

        def get_signal(self, features, prices):  # noqa: ANN001, ARG002
            assert features.shape == (self.num_symbols, 16)
            return SimpleNamespace(action="flat", symbol=None, direction=None, confidence=0.0, value_estimate=0.0)

        def step_day(self):
            return None

        def update_state(self, action, price, symbol):  # noqa: ANN001, ARG002
            return None

    monkeypatch.setattr(
        daily_stock,
        "load_local_daily_frames",
        lambda symbols, data_dir, min_days=daily_stock.DEFAULT_DAILY_FRAME_MIN_DAYS: {
            "AAPL": frame.copy(),
            "MSFT": frame.copy(),
        },
    )
    monkeypatch.setattr(daily_stock, "_load_cached_daily_trader", lambda *args, **kwargs: _FakeTrader())
    monkeypatch.setattr(
        daily_stock,
        "compute_daily_feature_history",
        lambda df: history_calls.__setitem__("count", history_calls["count"] + 1)
        or pd.DataFrame(np.tile(np.arange(16, dtype=np.float32), (len(df), 1))),
    )

    results = daily_stock.run_backtest(
        checkpoint="unused.pt",
        symbols=["AAPL", "MSFT"],
        data_dir="unused",
        days=5,
    )

    assert history_calls["count"] == 2
    assert results["trades"] == pytest.approx(0.0)
    assert results["total_return"] == pytest.approx(0.0)


def test_run_backtest_reports_full_loss_annualized_when_equity_turns_negative(monkeypatch, tmp_path: Path) -> None:
    rows = []
    base = pd.Timestamp("2025-09-01T00:00:00Z")
    for idx in range(140):
        ts = base + pd.Timedelta(days=idx)
        close = 100.0 if idx == 120 else 1.0
        rows.append(
            {
                "timestamp": ts.isoformat(),
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": 1_000 + idx,
            }
        )
    _write_daily_csv(tmp_path / "AAPL.csv", rows)

    class _FakeTrader:
        def __init__(
            self,
            checkpoint,
            device="cpu",
            long_only=True,
            symbols=None,
            allow_unsafe_checkpoint_loading=False,
        ):
            self.SYMBOLS = list(symbols or [])
            self.num_symbols = len(self.SYMBOLS)
            self.obs_size = 16 * max(1, self.num_symbols)
            self.num_actions = 1 + self.num_symbols
            self.cash = 0.0
            self.current_position = None
            self.position_qty = 0.0
            self.entry_price = 0.0
            self._calls = 0

        def get_daily_signal(self, frames, prices):
            self._calls += 1
            if self._calls == 1:
                return SimpleNamespace(action="buy", symbol="AAPL", direction="long", confidence=1.0, value_estimate=1.0)
            return SimpleNamespace(action="hold", symbol="AAPL", direction="long", confidence=1.0, value_estimate=1.0)

        def update_state(self, action, price, symbol):
            return None

        def step_day(self):
            return None

    monkeypatch.setattr(daily_stock, "DailyPPOTrader", _FakeTrader)

    results = daily_stock.run_backtest(
        checkpoint="unused.pt",
        symbols=["AAPL"],
        data_dir=str(tmp_path),
        days=20,
        allocation_pct=190.0,
        starting_cash=100.0,
        buying_power_multiplier=2.0,
    )

    assert results["total_return"] < -1.0
    assert results["annualized_return"] == -1.0


def test_run_backtest_applies_open_gate_to_low_confidence_signals(monkeypatch, tmp_path: Path) -> None:
    """Regression: backtest must honour min_open_confidence / min_open_value_estimate.

    Prior to the 2026-04-08 fix, ``run_backtest`` emitted open orders without
    calling the live open-gate, so threshold sweeps produced identical
    numbers and the production gate could not be tuned offline.
    """

    rows = []
    base = pd.Timestamp("2025-09-01T00:00:00Z")
    for idx in range(140):
        ts = base + pd.Timedelta(days=idx)
        close = 100.0 + idx
        rows.append(
            {
                "timestamp": ts.isoformat(),
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": 1_000 + idx,
            }
        )
    _write_daily_csv(tmp_path / "AAPL.csv", rows)

    class _FakeTrader:
        def __init__(
            self,
            checkpoint,
            device="cpu",
            long_only=True,
            symbols=None,
            allow_unsafe_checkpoint_loading=False,
        ):
            self.SYMBOLS = list(symbols or [])
            self.num_symbols = len(self.SYMBOLS)
            self.obs_size = 16 * max(1, self.num_symbols)
            self.num_actions = 1 + self.num_symbols
            self.cash = 0.0
            self.current_position = None
            self.position_qty = 0.0
            self.entry_price = 0.0

        def step_day(self):
            return None

        def update_state(self, action, price, symbol):
            return None

    def _fake_loader(checkpoint, *, device="cpu", long_only=True, symbols=None, allow_unsafe_checkpoint_loading=False):
        return _FakeTrader(checkpoint, symbols=symbols)

    monkeypatch.setattr(daily_stock, "DailyPPOTrader", _FakeTrader)
    monkeypatch.setattr(daily_stock, "_load_cached_daily_trader", _fake_loader)
    monkeypatch.setattr(
        daily_stock,
        "_trader_signal_from_features",
        lambda trader, *, features, prices, indexed, idx: SimpleNamespace(
            action="buy",
            symbol="AAPL",
            direction="long",
            confidence=0.10,
            value_estimate=0.25,
            allocation_pct=None,
            level_offset_bps=0.0,
        ),
    )
    monkeypatch.setattr(
        daily_stock,
        "_build_daily_feature_cube",
        lambda indexed, *, symbols, feature_schema: np.zeros((200, len(symbols), 16), dtype=np.float32),
    )

    blocked = daily_stock.run_backtest(
        checkpoint="unused.pt",
        symbols=["AAPL"],
        data_dir=str(tmp_path),
        days=10,
        allocation_pct=50.0,
        starting_cash=10_000.0,
        min_open_confidence=0.20,
        min_open_value_estimate=0.0,
    )
    allowed = daily_stock.run_backtest(
        checkpoint="unused.pt",
        symbols=["AAPL"],
        data_dir=str(tmp_path),
        days=10,
        allocation_pct=50.0,
        starting_cash=10_000.0,
        min_open_confidence=0.05,
        min_open_value_estimate=0.0,
    )
    value_blocked = daily_stock.run_backtest(
        checkpoint="unused.pt",
        symbols=["AAPL"],
        data_dir=str(tmp_path),
        days=10,
        allocation_pct=50.0,
        starting_cash=10_000.0,
        min_open_confidence=0.05,
        min_open_value_estimate=1.0,
    )

    # Blocked: every open is rejected by the gate, no PnL is realized.
    assert blocked["gate_blocked_opens"] >= 1.0
    assert blocked["total_return"] == pytest.approx(0.0)
    # Allowed: no gate blocks, position opens and rides the monotonic ramp.
    assert allowed["gate_blocked_opens"] == pytest.approx(0.0)
    assert allowed["total_return"] > 0.0
    # Value gate (min_open_value_estimate > signal.value_estimate) also bites.
    assert value_blocked["gate_blocked_opens"] >= 1.0
    assert value_blocked["total_return"] == pytest.approx(0.0)


def test_load_local_daily_frames_prefers_nested_train_subdir(tmp_path: Path) -> None:
    root = tmp_path / "trainingdata"
    nested = root / "train"
    root.mkdir(parents=True)
    nested.mkdir(parents=True)
    _write_daily_csv(
        root / "AAPL.csv",
        [
            {
                "timestamp": "2025-01-01T00:00:00Z",
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "volume": 1.0,
            }
        ],
    )
    _write_daily_csv(
        nested / "AAPL.csv",
        [
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "open": 2.0,
                "high": 2.0,
                "low": 2.0,
                "close": 2.0,
                "volume": 2.0,
            }
        ],
    )

    frames = daily_stock.load_local_daily_frames(["AAPL"], data_dir=str(root), min_days=1)
    assert float(frames["AAPL"]["close"].iloc[-1]) == 2.0


def test_run_backtest_passes_extra_checkpoints_to_preloaded_ensemble(monkeypatch, tmp_path: Path) -> None:
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

    captured: dict[str, object] = {"policies": []}

    class _FakeTrader:
        def __init__(
            self,
            checkpoint,
            device="cpu",
            long_only=True,
            symbols=None,
            allow_unsafe_checkpoint_loading=False,
        ):
            self.checkpoint = checkpoint
            self.device = device
            self.long_only = long_only
            self.SYMBOLS = list(symbols or [])
            self.num_symbols = len(self.SYMBOLS)
            self.obs_size = 16 * max(1, self.num_symbols)
            self.num_actions = 1 + self.num_symbols
            self.cash = 0.0
            self.current_position = None
            self.position_qty = 0.0
            self.entry_price = 0.0

        def get_daily_signal(self, frames, prices):
            return SimpleNamespace(action="flat", symbol=None, direction=None, confidence=0.0, value_estimate=0.0)

        def update_state(self, action, price, symbol):
            return None

        def step_day(self):
            return None

    monkeypatch.setattr(daily_stock, "DailyPPOTrader", _FakeTrader)
    monkeypatch.setattr(
        daily_stock,
        "_load_bare_policy",
        lambda path, obs_size, num_actions, device, allow_unsafe_checkpoint_loading=False: captured["policies"].append(
            (path, obs_size, num_actions, device)
        )
        or {"path": path},
    )
    monkeypatch.setattr(
        daily_stock,
        "_ensemble_softmax_signal",
        lambda trader, extra_policies, features, prices: SimpleNamespace(
            action="flat", symbol=None, direction=None, confidence=0.0, value_estimate=0.0
        ),
    )

    daily_stock.run_backtest(
        checkpoint="unused.pt",
        symbols=["AAPL"],
        data_dir=str(tmp_path),
        days=20,
        extra_checkpoints=["e1.pt", "e2.pt"],
    )

    assert [Path(item[0]).name for item in captured["policies"]] == ["e1.pt", "e2.pt"]


def test_run_backtest_ensemble_precomputes_feature_history_once_per_symbol(monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01T00:00:00Z", periods=25, freq="D"),
            "open": np.linspace(100.0, 124.0, 25),
            "high": np.linspace(101.0, 125.0, 25),
            "low": np.linspace(99.0, 123.0, 25),
            "close": np.linspace(100.5, 124.5, 25),
            "volume": np.linspace(1_000.0, 1_024.0, 25),
        }
    )
    history_calls = {"count": 0}
    ensemble_calls = {"count": 0}

    class _FakeTrader:
        def __init__(self) -> None:
            self.device = "cpu"
            self.SYMBOLS = ["AAPL", "MSFT"]
            self.num_symbols = len(self.SYMBOLS)
            self.obs_size = self.num_symbols * 16 + 5 + self.num_symbols
            self.num_actions = 1 + self.num_symbols
            self.cash = 0.0
            self.current_position = None
            self.position_qty = 0.0
            self.entry_price = 0.0

        def step_day(self):
            return None

        def update_state(self, action, price, symbol):  # noqa: ANN001, ARG002
            return None

    monkeypatch.setattr(
        daily_stock,
        "load_local_daily_frames",
        lambda symbols, data_dir, min_days=daily_stock.DEFAULT_DAILY_FRAME_MIN_DAYS: {
            "AAPL": frame.copy(),
            "MSFT": frame.copy(),
        },
    )
    monkeypatch.setattr(daily_stock, "_load_cached_daily_trader", lambda *args, **kwargs: _FakeTrader())
    monkeypatch.setattr(
        daily_stock,
        "compute_daily_feature_history",
        lambda df: history_calls.__setitem__("count", history_calls["count"] + 1)
        or pd.DataFrame(np.tile(np.arange(16, dtype=np.float32), (len(df), 1))),
    )
    monkeypatch.setattr(daily_stock, "_load_bare_policy", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        daily_stock,
        "_ensemble_softmax_signal",
        lambda trader, extra_policies, features, prices: (
            ensemble_calls.__setitem__("count", ensemble_calls["count"] + 1)
            or features.shape == (trader.num_symbols, 16)
        )
        and SimpleNamespace(action="flat", symbol=None, direction=None, confidence=0.0, value_estimate=0.0),
    )

    results = daily_stock.run_backtest(
        checkpoint="unused.pt",
        symbols=["AAPL", "MSFT"],
        data_dir="unused",
        days=5,
        extra_checkpoints=["e1.pt"],
    )

    assert history_calls["count"] == 2
    assert ensemble_calls["count"] == 5
    assert results["trades"] == pytest.approx(0.0)
    assert results["total_return"] == pytest.approx(0.0)


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


def test_execute_signal_closes_managed_position_and_waits_before_new_open(monkeypatch) -> None:
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
    assert orders == [("AAPL", 10.0, "sell")]
    assert state.active_symbol is None
    assert state.pending_close_symbol == "AAPL"
    assert state.pending_close_order_id == "AAPL-sell"


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


def test_execute_signal_skips_paper_market_fallback_without_quote(monkeypatch) -> None:
    called_market = False

    def _fake_submit_market_order(client, *, symbol: str, qty: float, side: str):
        nonlocal called_market
        called_market = True
        return SimpleNamespace(id=f"{symbol}-{side}")

    monkeypatch.setattr(daily_stock, "submit_market_order", _fake_submit_market_order)

    client = _FakeClient([])
    state = daily_stock.StrategyState(active_symbol=None, active_qty=0.0)
    signal = SimpleNamespace(symbol="MSFT", direction="long", action="long_MSFT")

    changed = daily_stock.execute_signal(
        signal,
        client=client,
        paper=True,
        quotes={"MSFT": 0.0},
        state=state,
        symbols=["MSFT"],
        allocation_pct=25.0,
        dry_run=False,
    )

    assert changed is False
    assert called_market is False
    assert state.active_symbol is None


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


def test_execute_signal_with_trading_server_rotation_close_does_not_force_loss_exit() -> None:
    submitted: list[dict[str, object]] = []

    class _FakeServerClient:
        def get_account(self):
            return {
                "cash": 5_000.0,
                "buying_power": 5_000.0,
                "positions": {
                    "AAPL": {"qty": 10.0, "avg_entry_price": 100.0, "current_price": 100.0},
                },
            }

        def refresh_prices(self, *, symbols):
            return {"accounts": []}

        def submit_limit_order(self, **kwargs):
            submitted.append(dict(kwargs))
            return {"order": {"id": "order-1"}}

    changed = daily_stock.execute_signal_with_trading_server(
        SimpleNamespace(symbol="MSFT", direction="long", action="long_MSFT"),
        server_client=_FakeServerClient(),
        quotes={"AAPL": 100.0, "MSFT": 50.0},
        state=daily_stock.StrategyState(active_symbol="AAPL", active_qty=10.0, entry_price=100.0),
        symbols=["AAPL", "MSFT"],
        allocation_pct=25.0,
        dry_run=False,
        now=datetime(2026, 3, 16, 14, 0, tzinfo=timezone.utc),
    )

    assert changed is True
    assert submitted[0] == {
        "symbol": "AAPL",
        "qty": 10.0,
        "side": "sell",
        "limit_price": 100.0,
        "metadata": {"strategy": "daily_stock_rl", "intent": "close_managed"},
    }


def test_execute_signal_with_trading_server_holds_position_when_loss_guard_blocks_close(caplog) -> None:
    submitted: list[dict[str, object]] = []

    class _FakeServerClient:
        def get_account(self):
            return {
                "cash": 5_000.0,
                "buying_power": 5_000.0,
                "positions": {
                    "AAPL": {"qty": 10.0, "avg_entry_price": 100.0, "current_price": 95.0},
                },
            }

        def refresh_prices(self, *, symbols):
            return {"accounts": []}

        def submit_limit_order(self, **kwargs):
            submitted.append(dict(kwargs))
            if kwargs["side"] == "sell":
                raise RuntimeError(
                    "400: sell rejected for AAPL: limit 95.0000 is below safety floor 100.0000. "
                    "Loss-taking exits require allow_loss_exit=true and a force_exit_reason."
                )
            return {"order": {"id": "order-2"}}

    state = daily_stock.StrategyState(active_symbol="AAPL", active_qty=10.0, entry_price=100.0)
    changed = daily_stock.execute_signal_with_trading_server(
        SimpleNamespace(symbol="MSFT", direction="long", action="long_MSFT"),
        server_client=_FakeServerClient(),
        quotes={"AAPL": 95.0, "MSFT": 50.0},
        state=state,
        symbols=["AAPL", "MSFT"],
        allocation_pct=25.0,
        dry_run=False,
        now=datetime(2026, 3, 16, 14, 0, tzinfo=timezone.utc),
    )

    assert changed is False
    assert len(submitted) == 1
    assert state.active_symbol == "AAPL"
    assert state.pending_close_symbol is None
    assert "Server rejected ordinary sell for AAPL" in caplog.text

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

    assert prices == {"AAPL": 149.75, "MSFT": 202.0}
    assert source == "mixed_fallback"
    assert source_by_symbol == {"AAPL": "alpaca", "MSFT": "close_fallback"}


def test_bars_are_fresh_uses_age_gate() -> None:
    latest_bar = pd.Timestamp("2026-03-14T00:00:00Z")

    assert daily_stock.bars_are_fresh(
        latest_bar=latest_bar,
        now=datetime(2026, 3, 16, 13, 40, tzinfo=timezone.utc),
        max_age_days=daily_stock.DEFAULT_BAR_FRESHNESS_MAX_AGE_DAYS,
    ) is True
    assert daily_stock.bars_are_fresh(
        latest_bar=latest_bar,
        now=datetime(2026, 3, 25, 13, 40, tzinfo=timezone.utc),
        max_age_days=daily_stock.DEFAULT_BAR_FRESHNESS_MAX_AGE_DAYS,
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
    monkeypatch.setattr(daily_stock, "execute_signal", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        daily_stock,
        "build_signal",
        lambda checkpoint,
        frames,
        portfolio=daily_stock.PortfolioContext(),
        device="cpu",
        extra_checkpoints=None,
        allow_unsafe_checkpoint_loading=False: (
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
    assert payload["execution_submitted"] is False
    assert payload["execution_would_submit"] is True
    assert payload["execution_status"] == "dry_run_would_execute"
    assert payload["execution_skip_reason"] is None
    assert payload["signal_log_written"] is True
    assert payload["signal_log_write_error"] is None
    assert state_path.exists() is False


def test_run_once_dry_run_does_not_advance_state(monkeypatch, tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text(
        '{"active_symbol": null, "active_qty": 0.0, "last_run_date": null, "last_signal_action": null, "last_signal_timestamp": null, "last_order_id": null}'
    )

    monkeypatch.setattr(
        daily_stock,
        "build_signal",
        lambda checkpoint,
        frames,
        portfolio=daily_stock.PortfolioContext(),
        device="cpu",
        extra_checkpoints=None,
        allow_unsafe_checkpoint_loading=False: (
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


def test_run_once_continues_when_signal_log_write_fails(caplog, monkeypatch, tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text(
        '{"active_symbol": null, "active_qty": 0.0, "last_run_date": null, "last_signal_action": null, "last_signal_timestamp": null, "last_order_id": null}',
        encoding="utf-8",
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
            )
        },
    )
    monkeypatch.setattr(
        daily_stock,
        "build_signal",
        lambda checkpoint,
        frames,
        portfolio=daily_stock.PortfolioContext(),
        device="cpu",
        extra_checkpoints=None,
        allow_unsafe_checkpoint_loading=False: (
            SimpleNamespace(
                action="flat",
                symbol=None,
                direction=None,
                confidence=0.5,
                value_estimate=0.0,
            ),
            {"AAPL": 123.0},
        ),
    )
    monkeypatch.setattr(
        daily_stock,
        "append_signal_log",
        lambda payload, path=daily_stock.SIGNAL_LOG_PATH: (_ for _ in ()).throw(OSError("disk full")),
    )

    with caplog.at_level(logging.INFO, logger="daily_stock_rl"):
        payload = daily_stock.run_once(
            checkpoint="dummy.ckpt",
            symbols=["AAPL"],
            paper=True,
            allocation_pct=25.0,
            dry_run=False,
            data_source="local",
            data_dir=str(tmp_path),
            state_path=state_path,
        )

    state_payload = json.loads(state_path.read_text(encoding="utf-8"))

    assert payload["signal_log_written"] is False
    assert payload["signal_log_write_error"] == "disk full"
    assert payload["state_advanced"] is True
    assert state_payload["last_signal_action"] == "flat"
    assert state_payload["last_run_date"] is not None
    assert any("Signal log write failed" in record.getMessage() for record in caplog.records)


def test_run_once_persists_structured_run_event(monkeypatch, tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text(
        '{"active_symbol": null, "active_qty": 0.0, "last_run_date": null, "last_signal_action": null, "last_signal_timestamp": null, "last_order_id": null}',
        encoding="utf-8",
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
            )
        },
    )
    monkeypatch.setattr(
        daily_stock,
        "build_signal",
        lambda checkpoint,
        frames,
        portfolio=daily_stock.PortfolioContext(),
        device="cpu",
        extra_checkpoints=None,
        allow_unsafe_checkpoint_loading=False: (
            SimpleNamespace(
                action="flat",
                symbol=None,
                direction=None,
                confidence=0.5,
                value_estimate=0.0,
            ),
            {"AAPL": 123.0},
        ),
    )
    monkeypatch.setattr(daily_stock, "append_signal_log", lambda payload, path=daily_stock.SIGNAL_LOG_PATH: None)
    run_events: list[dict[str, object]] = []
    monkeypatch.setattr(
        daily_stock,
        "append_run_event_log",
        lambda payload, path=daily_stock.RUN_EVENT_LOG_PATH: run_events.append(dict(payload)),
    )

    payload = daily_stock.run_once(
        checkpoint="dummy.ckpt",
        symbols=["AAPL"],
        paper=True,
        allocation_pct=25.0,
        dry_run=False,
        data_source="local",
        data_dir=str(tmp_path),
        state_path=state_path,
    )

    assert len(run_events) == 1
    event = run_events[0]
    assert event["event"] == "daily_stock_run_once"
    assert event["run_id"] == payload["run_id"]
    assert event["timestamp"] == payload["timestamp"]
    assert event["execution_status"] == "local_only"
    assert event["signal_log_written"] is True
    assert payload["run_event_log_written"] is True
    assert payload["run_event_log_write_error"] is None


def test_run_once_continues_when_run_event_log_write_fails(caplog, monkeypatch, tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text(
        '{"active_symbol": null, "active_qty": 0.0, "last_run_date": null, "last_signal_action": null, "last_signal_timestamp": null, "last_order_id": null}',
        encoding="utf-8",
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
            )
        },
    )
    monkeypatch.setattr(
        daily_stock,
        "build_signal",
        lambda checkpoint,
        frames,
        portfolio=daily_stock.PortfolioContext(),
        device="cpu",
        extra_checkpoints=None,
        allow_unsafe_checkpoint_loading=False: (
            SimpleNamespace(
                action="flat",
                symbol=None,
                direction=None,
                confidence=0.5,
                value_estimate=0.0,
            ),
            {"AAPL": 123.0},
        ),
    )
    monkeypatch.setattr(daily_stock, "append_signal_log", lambda payload, path=daily_stock.SIGNAL_LOG_PATH: None)
    monkeypatch.setattr(
        daily_stock,
        "append_run_event_log",
        lambda payload, path=daily_stock.RUN_EVENT_LOG_PATH: (_ for _ in ()).throw(OSError("readonly fs")),
    )

    with caplog.at_level(logging.INFO, logger="daily_stock_rl"):
        payload = daily_stock.run_once(
            checkpoint="dummy.ckpt",
            symbols=["AAPL"],
            paper=True,
            allocation_pct=25.0,
            dry_run=False,
            data_source="local",
            data_dir=str(tmp_path),
            state_path=state_path,
        )

    assert payload["run_id"]
    assert payload["run_event_log_written"] is False
    assert payload["run_event_log_write_error"] == "readonly fs"
    assert any("Run event log write failed" in record.getMessage() for record in caplog.records)
    summary_records = [
        record.getMessage()
        for record in caplog.records
        if record.name == "daily_stock_rl" and record.getMessage().startswith("Run summary: ")
    ]
    assert summary_records
    assert '"run_event_log_written": false' in summary_records[-1]
    assert '"run_event_log_write_error": "readonly fs"' in summary_records[-1]


def test_run_once_market_closed_does_not_advance_state(caplog, monkeypatch, tmp_path: Path) -> None:
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
        lambda checkpoint,
        frames,
        portfolio=daily_stock.PortfolioContext(),
        device="cpu",
        extra_checkpoints=None,
        allow_unsafe_checkpoint_loading=False: (
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
    with caplog.at_level(logging.INFO, logger="daily_stock_rl"):
        payload = daily_stock.run_once(
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
    assert payload["market_open"] is False
    assert payload["execution_submitted"] is False
    assert payload["execution_would_submit"] is False
    assert payload["execution_status"] == "skipped_market_closed"
    assert payload["execution_skip_reason"] == "market_closed"
    assert payload["state_advanced"] is False
    summary_records = [
        record.getMessage()
        for record in caplog.records
        if record.name == "daily_stock_rl" and record.getMessage().startswith("Run summary: ")
    ]
    assert summary_records
    assert '"event": "daily_stock_run_once"' in summary_records[-1]
    assert '"execution_status": "skipped_market_closed"' in summary_records[-1]


def test_run_once_logs_structured_failure_context_for_trading_server_startup(
    caplog,
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(daily_stock, "load_state", lambda path: daily_stock.StrategyState())
    monkeypatch.setattr(daily_stock, "build_data_client", lambda paper: object())
    monkeypatch.setattr(
        daily_stock,
        "build_trading_client",
        lambda paper: SimpleNamespace(get_clock=lambda: SimpleNamespace(is_open=True)),
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
                )
            },
            "alpaca",
        ),
    )
    monkeypatch.setattr(
        daily_stock,
        "load_latest_quotes_with_source",
        lambda symbols, paper, fallback_prices, data_client=None: (
            {"AAPL": 1.0},
            "alpaca",
            {"AAPL": "alpaca"},
        ),
    )
    monkeypatch.setattr(
        daily_stock,
        "build_server_client",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("server unavailable")),
    )

    with caplog.at_level(logging.ERROR, logger="daily_stock_rl"):
        with pytest.raises(RuntimeError, match="server unavailable") as exc_info:
            daily_stock.run_once(
                checkpoint="dummy.ckpt",
                symbols=["AAPL"],
                paper=True,
                allocation_pct=25.0,
                dry_run=False,
                data_source="alpaca",
                data_dir=str(tmp_path),
                execution_backend="trading_server",
                server_account="paper-account",
                server_bot_id="daily-bot",
                server_url="https://server.internal:8050",
            )

    failure_records = [
        record.getMessage()
        for record in caplog.records
        if record.name == "daily_stock_rl" and record.getMessage().startswith("Run failure: ")
    ]
    assert failure_records
    payload = json.loads(failure_records[-1].split("Run failure: ", 1)[1])
    assert payload["event"] == "daily_stock_run_once_failed"
    assert payload["stage"] == "build_trading_server_client"
    assert payload["error_type"] == "RuntimeError"
    assert payload["error"] == "server unavailable"
    assert payload["execution_backend"] == "trading_server"
    assert payload["server_url"] == "https://server.internal:8050"
    assert payload["observability"]["bar_data_source"] == "alpaca"
    assert payload["observability"]["quote_data_source"] == "alpaca"
    assert payload["observability"]["latest_bar_timestamp"] == "2026-03-13T00:00:00+00:00"
    assert "state_active_symbol" not in payload["observability"]
    notes = getattr(exc_info.value, "__notes__", [])
    assert "run_once stage: build_trading_server_client" in notes
    assert any(note.startswith("run_once context: ") for note in notes)


def test_run_once_logs_structured_failure_context_for_unreadable_state(
    caplog,
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text("{not-json\n", encoding="utf-8")

    with caplog.at_level(logging.ERROR, logger="daily_stock_rl"):
        with pytest.raises(RuntimeError, match=r"Unreadable strategy state file .*state\.json") as exc_info:
            daily_stock.run_once(
                checkpoint="dummy.ckpt",
                symbols=["AAPL"],
                paper=True,
                allocation_pct=25.0,
                dry_run=True,
                data_source="local",
                data_dir=str(tmp_path),
                state_path=state_path,
            )

    failure_records = [
        record.getMessage()
        for record in caplog.records
        if record.name == "daily_stock_rl" and record.getMessage().startswith("Run failure: ")
    ]
    assert failure_records
    payload = json.loads(failure_records[-1].split("Run failure: ", 1)[1])
    assert payload["stage"] == "load_state"
    assert payload["error_type"] == "RuntimeError"
    assert "Unreadable strategy state file" in payload["error"]
    assert payload["state_path"] == str(state_path)
    assert payload["requested_local_data_dir"] == str(tmp_path.resolve())
    assert payload["resolved_local_data_dir"] == str(tmp_path.resolve())
    assert payload["resolved_local_data_dir_source"] == "requested"
    notes = getattr(exc_info.value, "__notes__", [])
    assert "run_once stage: load_state" in notes
    assert not any(note.startswith("run_once context: ") for note in notes)


def test_run_once_failure_persists_structured_run_event(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(daily_stock, "load_state", lambda path: daily_stock.StrategyState())
    monkeypatch.setattr(daily_stock, "build_data_client", lambda paper: object())
    monkeypatch.setattr(
        daily_stock,
        "build_trading_client",
        lambda paper: SimpleNamespace(get_clock=lambda: SimpleNamespace(is_open=True)),
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
                )
            },
            "alpaca",
        ),
    )
    monkeypatch.setattr(
        daily_stock,
        "load_latest_quotes_with_source",
        lambda symbols, paper, fallback_prices, data_client=None: (
            {"AAPL": 1.0},
            "alpaca",
            {"AAPL": "alpaca"},
        ),
    )
    monkeypatch.setattr(
        daily_stock,
        "build_server_client",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("server unavailable")),
    )
    run_events: list[dict[str, object]] = []
    monkeypatch.setattr(
        daily_stock,
        "append_run_event_log",
        lambda payload, path=daily_stock.RUN_EVENT_LOG_PATH: run_events.append(dict(payload)),
    )

    with pytest.raises(RuntimeError, match="server unavailable"):
        daily_stock.run_once(
            checkpoint="dummy.ckpt",
            symbols=["AAPL"],
            paper=True,
            allocation_pct=25.0,
            dry_run=False,
            data_source="alpaca",
            data_dir=str(tmp_path),
            execution_backend="trading_server",
            server_account="paper-account",
            server_bot_id="daily-bot",
            server_url="https://server.internal:8050",
        )

    assert len(run_events) == 1
    event = run_events[0]
    assert event["event"] == "daily_stock_run_once_failed"
    assert event["stage"] == "build_trading_server_client"
    assert event["error"] == "server unavailable"
    assert event["run_id"]
    assert event["server_url"] == "https://server.internal:8050"


def test_run_once_blocks_new_entry_when_confidence_or_value_fail_gate(monkeypatch, tmp_path: Path) -> None:
    fresh_timestamp = pd.Timestamp.now(tz="UTC").normalize()

    monkeypatch.setattr(
        daily_stock,
        "load_inference_frames",
        lambda symbols, paper, data_dir, now, data_client=None: (
            {
                "AAPL": pd.DataFrame(
                    {
                        "timestamp": pd.to_datetime([fresh_timestamp]),
                        "open": [100.0],
                        "high": [101.0],
                        "low": [99.0],
                        "close": [100.5],
                        "volume": [1_000.0],
                    }
                )
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
            get_clock=lambda: SimpleNamespace(is_open=True),
        ),
    )
    monkeypatch.setattr(daily_stock, "build_data_client", lambda paper: object())
    monkeypatch.setattr(
        daily_stock,
        "load_latest_quotes_with_source",
        lambda symbols, paper, fallback_prices, data_client=None: (
            {"AAPL": 100.75},
            "alpaca",
            {"AAPL": "alpaca"},
        ),
    )
    monkeypatch.setattr(
        daily_stock,
        "build_signal",
        lambda checkpoint,
        frames,
        portfolio=daily_stock.PortfolioContext(),
        device="cpu",
        extra_checkpoints=None,
        allow_unsafe_checkpoint_loading=False: (
            SimpleNamespace(
                action="long_AAPL",
                symbol="AAPL",
                direction="long",
                confidence=0.11,
                value_estimate=-0.9,
            ),
            {"AAPL": 100.5},
        ),
    )
    execute_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        daily_stock,
        "execute_signal",
        lambda *args, **kwargs: execute_calls.append(kwargs) or False,
    )

    payload = daily_stock.run_once(
        checkpoint="dummy.ckpt",
        symbols=["AAPL"],
        paper=True,
        allocation_pct=25.0,
        dry_run=False,
        data_source="alpaca",
        data_dir=str(tmp_path),
        state_path=tmp_path / "state.json",
        min_open_confidence=0.2,
        min_open_value_estimate=0.0,
    )

    assert execute_calls and execute_calls[0]["allow_open"] is False
    assert "min_open_confidence" in execute_calls[0]["allow_open_reason"]
    assert "min_open_value_estimate" in execute_calls[0]["allow_open_reason"]
    assert payload["allow_open"] is False
    assert payload["allow_open_reasons"]
    assert payload["execution_status"] == "blocked_open_gate"
    assert payload["execution_skip_reason"] == "open_gate"


def test_run_once_dry_run_executes_multi_position_portfolio(monkeypatch, tmp_path: Path) -> None:
    fresh_timestamp = pd.Timestamp.now(tz="UTC").normalize()

    monkeypatch.setattr(
        daily_stock,
        "load_inference_frames",
        lambda symbols, paper, data_dir, now, data_client=None: (
            {
                "AAPL": pd.DataFrame(
                    {
                        "timestamp": pd.to_datetime([fresh_timestamp]),
                        "open": [100.0],
                        "high": [101.0],
                        "low": [99.0],
                        "close": [100.5],
                        "volume": [1_000.0],
                    }
                ),
                "MSFT": pd.DataFrame(
                    {
                        "timestamp": pd.to_datetime([fresh_timestamp]),
                        "open": [200.0],
                        "high": [201.0],
                        "low": [199.0],
                        "close": [200.5],
                        "volume": [2_000.0],
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
            get_account=lambda: SimpleNamespace(
                cash=10_000.0,
                buying_power=10_000.0,
                portfolio_value=10_000.0,
            ),
            get_clock=lambda: SimpleNamespace(is_open=True),
        ),
    )
    monkeypatch.setattr(daily_stock, "build_data_client", lambda paper: object())
    monkeypatch.setattr(
        daily_stock,
        "load_latest_quotes_with_source",
        lambda symbols, paper, fallback_prices, data_client=None: (
            {"AAPL": 100.75, "MSFT": 200.75},
            "alpaca",
            {"AAPL": "alpaca", "MSFT": "alpaca"},
        ),
    )
    monkeypatch.setattr(
        daily_stock,
        "_build_multi_position_signals",
        lambda *args, **kwargs: [
            daily_stock.TradingSignal("long_AAPL", "AAPL", "long", 0.9, 1.2, 0.6, 0.0),
            daily_stock.TradingSignal("long_MSFT", "MSFT", "long", 0.8, 1.1, 0.4, 0.0),
        ],
    )
    execute_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        daily_stock,
        "execute_multi_position_signals",
        lambda signals, **kwargs: execute_calls.append(
            {
                "signals": [signal.symbol for signal in signals],
                **kwargs,
            }
        ) or {"AAPL": 12.0, "MSFT": 5.0},
    )

    payload = daily_stock.run_once(
        checkpoint="dummy.ckpt",
        symbols=["AAPL", "MSFT"],
        paper=True,
        allocation_pct=25.0,
        dry_run=True,
        data_source="alpaca",
        data_dir=str(tmp_path),
        state_path=tmp_path / "state.json",
        multi_position=2,
    )

    assert execute_calls
    assert execute_calls[0]["signals"] == ["AAPL", "MSFT"]
    assert payload["portfolio_mode"] is True
    assert payload["portfolio_signal_count"] == 2
    assert [item["symbol"] for item in payload["portfolio_signals"]] == ["AAPL", "MSFT"]
    assert payload["held_positions"] == {"AAPL": 12.0, "MSFT": 5.0}
    assert payload["execution_status"] == "dry_run_would_execute"
    assert payload["execution_skip_reason"] is None


def test_run_once_multi_position_filters_blocked_signals_before_execution(
    monkeypatch,
    tmp_path: Path,
) -> None:
    fresh_timestamp = pd.Timestamp.now(tz="UTC").normalize()

    monkeypatch.setattr(
        daily_stock,
        "load_inference_frames",
        lambda symbols, paper, data_dir, now, data_client=None: (
            {
                "AAPL": pd.DataFrame(
                    {
                        "timestamp": pd.to_datetime([fresh_timestamp]),
                        "open": [100.0],
                        "high": [101.0],
                        "low": [99.0],
                        "close": [100.5],
                        "volume": [1_000.0],
                    }
                ),
                "MSFT": pd.DataFrame(
                    {
                        "timestamp": pd.to_datetime([fresh_timestamp]),
                        "open": [200.0],
                        "high": [201.0],
                        "low": [199.0],
                        "close": [200.5],
                        "volume": [2_000.0],
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
            get_account=lambda: SimpleNamespace(
                cash=10_000.0,
                buying_power=10_000.0,
                portfolio_value=10_000.0,
            ),
            get_clock=lambda: SimpleNamespace(is_open=True),
        ),
    )
    monkeypatch.setattr(daily_stock, "build_data_client", lambda paper: object())
    monkeypatch.setattr(
        daily_stock,
        "load_latest_quotes_with_source",
        lambda symbols, paper, fallback_prices, data_client=None: (
            {"AAPL": 100.75, "MSFT": 200.75},
            "alpaca",
            {"AAPL": "alpaca", "MSFT": "alpaca"},
        ),
    )
    monkeypatch.setattr(
        daily_stock,
        "_build_multi_position_signals",
        lambda *args, **kwargs: [
            daily_stock.TradingSignal("long_AAPL", "AAPL", "long", 0.1, 1.2, 0.6, 0.0),
            daily_stock.TradingSignal("long_MSFT", "MSFT", "long", 0.8, 1.1, 0.4, 0.0),
        ],
    )
    execute_calls: list[list[str]] = []
    monkeypatch.setattr(
        daily_stock,
        "execute_multi_position_signals",
        lambda signals, **kwargs: execute_calls.append([str(signal.symbol) for signal in signals]) or {"MSFT": 5.0},
    )

    payload = daily_stock.run_once(
        checkpoint="dummy.ckpt",
        symbols=["AAPL", "MSFT"],
        paper=True,
        allocation_pct=25.0,
        dry_run=True,
        data_source="alpaca",
        data_dir=str(tmp_path),
        state_path=tmp_path / "state.json",
        multi_position=2,
        min_open_confidence=0.2,
        min_open_value_estimate=0.0,
    )

    assert execute_calls == [["MSFT"]]
    assert payload["allow_open"] is True
    assert payload["blocked_portfolio_signals"] == [
        {
            "symbol": "AAPL",
            "reasons": ["confidence=0.1000 < min_open_confidence=0.2000"],
        }
    ]
    assert payload["execution_status"] == "dry_run_would_execute"


def test_run_once_dry_run_executes_multi_position_portfolio_via_trading_server(
    monkeypatch,
    tmp_path: Path,
) -> None:
    fresh_timestamp = pd.Timestamp.now(tz="UTC").normalize()

    monkeypatch.setattr(
        daily_stock,
        "load_inference_frames",
        lambda symbols, paper, data_dir, now, data_client=None: (
            {
                "AAPL": pd.DataFrame(
                    {
                        "timestamp": pd.to_datetime([fresh_timestamp]),
                        "open": [100.0],
                        "high": [101.0],
                        "low": [99.0],
                        "close": [100.5],
                        "volume": [1_000.0],
                    }
                ),
                "MSFT": pd.DataFrame(
                    {
                        "timestamp": pd.to_datetime([fresh_timestamp]),
                        "open": [200.0],
                        "high": [201.0],
                        "low": [199.0],
                        "close": [200.5],
                        "volume": [2_000.0],
                    }
                ),
            },
            "alpaca",
        ),
    )
    monkeypatch.setattr(
        daily_stock,
        "build_trading_client",
        lambda paper: SimpleNamespace(get_clock=lambda: SimpleNamespace(is_open=True)),
    )
    monkeypatch.setattr(daily_stock, "build_data_client", lambda paper: object())
    monkeypatch.setattr(
        daily_stock,
        "load_latest_quotes_with_source",
        lambda symbols, paper, fallback_prices, data_client=None: (
            {"AAPL": 100.75, "MSFT": 200.75},
            "alpaca",
            {"AAPL": "alpaca", "MSFT": "alpaca"},
        ),
    )
    monkeypatch.setattr(
        daily_stock,
        "_build_multi_position_signals",
        lambda *args, **kwargs: [
            daily_stock.TradingSignal("long_AAPL", "AAPL", "long", 0.9, 1.2, 0.6, 0.0),
            daily_stock.TradingSignal("long_MSFT", "MSFT", "long", 0.8, 1.1, 0.4, 0.0),
        ],
    )

    server_client = SimpleNamespace(
        get_account=lambda: {"cash": 10_000.0, "buying_power": 10_000.0, "positions": {}}
    )
    monkeypatch.setattr(daily_stock, "build_server_client", lambda **kwargs: server_client)

    execute_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        daily_stock,
        "execute_multi_position_signals_with_trading_server",
        lambda signals, **kwargs: execute_calls.append(
            {
                "signals": [signal.symbol for signal in signals],
                **kwargs,
            }
        ) or {"AAPL": 12.0, "MSFT": 5.0},
    )

    payload = daily_stock.run_once(
        checkpoint="dummy.ckpt",
        symbols=["AAPL", "MSFT"],
        paper=True,
        allocation_pct=25.0,
        dry_run=True,
        data_source="alpaca",
        data_dir=str(tmp_path),
        state_path=tmp_path / "state.json",
        execution_backend="trading_server",
        server_account="paper-account",
        server_bot_id="daily-bot",
        multi_position=2,
    )

    assert execute_calls
    assert execute_calls[0]["signals"] == ["AAPL", "MSFT"]
    assert execute_calls[0]["server_client"] is server_client
    assert payload["portfolio_mode"] is True
    assert payload["held_positions"] == {"AAPL": 12.0, "MSFT": 5.0}
    assert payload["server_account"] == "paper-account"
    assert payload["server_bot_id"] == "daily-bot"
    assert payload["server_snapshot"] == {"cash": 10_000.0, "buying_power": 10_000.0, "positions": {}}
    assert payload["execution_status"] == "dry_run_would_execute"


def test_run_once_multi_position_all_blocked_skips_execution(
    monkeypatch,
    tmp_path: Path,
) -> None:
    fresh_timestamp = pd.Timestamp.now(tz="UTC").normalize()

    monkeypatch.setattr(
        daily_stock,
        "load_inference_frames",
        lambda symbols, paper, data_dir, now, data_client=None: (
            {
                "AAPL": pd.DataFrame(
                    {
                        "timestamp": pd.to_datetime([fresh_timestamp]),
                        "open": [100.0],
                        "high": [101.0],
                        "low": [99.0],
                        "close": [100.5],
                        "volume": [1_000.0],
                    }
                ),
                "MSFT": pd.DataFrame(
                    {
                        "timestamp": pd.to_datetime([fresh_timestamp]),
                        "open": [200.0],
                        "high": [201.0],
                        "low": [199.0],
                        "close": [200.5],
                        "volume": [2_000.0],
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
            get_account=lambda: SimpleNamespace(
                cash=10_000.0,
                buying_power=10_000.0,
                portfolio_value=10_000.0,
            ),
            get_clock=lambda: SimpleNamespace(is_open=True),
        ),
    )
    monkeypatch.setattr(daily_stock, "build_data_client", lambda paper: object())
    monkeypatch.setattr(
        daily_stock,
        "load_latest_quotes_with_source",
        lambda symbols, paper, fallback_prices, data_client=None: (
            {"AAPL": 100.75, "MSFT": 200.75},
            "alpaca",
            {"AAPL": "alpaca", "MSFT": "alpaca"},
        ),
    )
    monkeypatch.setattr(
        daily_stock,
        "_build_multi_position_signals",
        lambda *args, **kwargs: [
            daily_stock.TradingSignal("long_AAPL", "AAPL", "long", 0.1, -1.2, 0.6, 0.0),
            daily_stock.TradingSignal("long_MSFT", "MSFT", "long", 0.15, -0.4, 0.4, 0.0),
        ],
    )
    execute_calls: list[list[str]] = []
    monkeypatch.setattr(
        daily_stock,
        "execute_multi_position_signals",
        lambda signals, **kwargs: execute_calls.append([str(signal.symbol) for signal in signals]) or {},
    )

    payload = daily_stock.run_once(
        checkpoint="dummy.ckpt",
        symbols=["AAPL", "MSFT"],
        paper=True,
        allocation_pct=25.0,
        dry_run=True,
        data_source="alpaca",
        data_dir=str(tmp_path),
        state_path=tmp_path / "state.json",
        multi_position=2,
        min_open_confidence=0.2,
        min_open_value_estimate=0.0,
    )

    assert execute_calls == [[]]
    assert payload["allow_open"] is False
    assert payload["allow_open_reason"] == (
        "AAPL: confidence=0.1000 < min_open_confidence=0.2000; "
        "value_estimate=-1.2000 < min_open_value_estimate=0.0000 | "
        "MSFT: confidence=0.1500 < min_open_confidence=0.2000; "
        "value_estimate=-0.4000 < min_open_value_estimate=0.0000"
    )
    assert payload["execution_status"] == "blocked_open_gate"
    assert payload["execution_skip_reason"] == "open_gate"


def test_main_print_payload_outputs_run_once_json(monkeypatch, capsys, tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("stub", encoding="utf-8")

    monkeypatch.setattr(
        daily_stock,
        "_checkpoint_load_diagnostics",
        lambda _config: {"ok": True, "error": None, "primary": {"arch": "mlp"}, "extras": []},
    )
    monkeypatch.setattr(
        daily_stock,
        "run_once",
        lambda **kwargs: {
            "action": "long_AAPL",
            "symbol": "AAPL",
            "allow_open": True,
            "min_open_confidence": kwargs["min_open_confidence"],
        },
    )

    daily_stock.main(
        [
            "--dry-run",
            "--no-ensemble",
            "--checkpoint",
            str(checkpoint),
            "--symbols",
            "AAPL",
            "--print-payload",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["symbol"] == "AAPL"
    assert payload["allow_open"] is True
    assert payload["min_open_confidence"] == pytest.approx(daily_stock.DEFAULT_MIN_OPEN_CONFIDENCE)


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
        allocation_pct=150.0,
        starting_cash=25_000.0,
        buying_power_multiplier=2.0,
    )
    server = daily_stock.run_backtest_via_trading_server(
        checkpoint=str(checkpoint),
        symbols=["AAPL"],
        data_dir=str(tmp_path),
        days=20,
        allocation_pct=150.0,
        starting_cash=25_000.0,
        buying_power_multiplier=2.0,
    )

    assert server["total_return"] == legacy["total_return"]
    assert server["annualized_return"] == legacy["annualized_return"]
    assert server["sortino"] == legacy["sortino"]
    assert server["max_drawdown"] == legacy["max_drawdown"]
    assert server["trades"] == legacy["trades"]
    assert server["orders"] == pytest.approx(1.0)


def test_run_backtest_via_trading_server_single_position_precomputes_feature_history(monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01T00:00:00Z", periods=25, freq="D"),
            "open": np.linspace(100.0, 124.0, 25),
            "high": np.linspace(101.0, 125.0, 25),
            "low": np.linspace(99.0, 123.0, 25),
            "close": np.linspace(100.5, 124.5, 25),
            "volume": np.linspace(1_000.0, 1_024.0, 25),
        }
    )
    history_calls = {"count": 0}

    class _FakeTrader:
        def __init__(self) -> None:
            self.device = "cpu"
            self.SYMBOLS = ["AAPL", "MSFT"]
            self.num_symbols = len(self.SYMBOLS)
            self.num_actions = 1 + self.num_symbols
            self.cash = 0.0
            self.current_position = None
            self.position_qty = 0.0
            self.entry_price = 0.0
            self.hold_days = 0
            self.hold_hours = 0
            self.step = 0
            self.max_steps = 90

        def get_signal(self, features, prices):  # noqa: ANN001, ARG002
            assert features.shape == (self.num_symbols, 16)
            return SimpleNamespace(action="flat", symbol=None, direction=None, confidence=0.0, value_estimate=0.0)

        def step_day(self):
            return None

    monkeypatch.setattr(
        daily_stock,
        "load_local_daily_frames",
        lambda symbols, data_dir, min_days=daily_stock.DEFAULT_DAILY_FRAME_MIN_DAYS: {
            "AAPL": frame.copy(),
            "MSFT": frame.copy(),
        },
    )
    monkeypatch.setattr(daily_stock, "_load_cached_daily_trader", lambda *args, **kwargs: _FakeTrader())
    monkeypatch.setattr(
        daily_stock,
        "compute_daily_feature_history",
        lambda df: history_calls.__setitem__("count", history_calls["count"] + 1)
        or pd.DataFrame(np.tile(np.arange(16, dtype=np.float32), (len(df), 1))),
    )
    monkeypatch.setattr(
        daily_stock,
        "build_signal",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("build_signal should not be used")),
    )

    results = daily_stock.run_backtest_via_trading_server(
        checkpoint="unused.pt",
        symbols=["AAPL", "MSFT"],
        data_dir="unused",
        days=5,
    )

    assert history_calls["count"] == 2
    assert results["trades"] == pytest.approx(0.0)
    assert results["orders"] == pytest.approx(0.0)
    assert results["total_return"] == pytest.approx(0.0)


def test_prepare_daily_backtest_data_precomputes_close_matrix_and_timestamps(monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01T00:00:00Z", periods=25, freq="D"),
            "open": np.linspace(100.0, 124.0, 25),
            "high": np.linspace(101.0, 125.0, 25),
            "low": np.linspace(99.0, 123.0, 25),
            "close": np.linspace(100.5, 124.5, 25),
            "volume": np.linspace(1_000.0, 1_024.0, 25),
        }
    )

    class _FakeTrader:
        def __init__(self) -> None:
            self.device = "cpu"
            self.SYMBOLS = ["AAPL", "MSFT"]
            self.num_symbols = len(self.SYMBOLS)
            self.obs_size = self.num_symbols * 16 + 5 + self.num_symbols
            self.num_actions = 1 + self.num_symbols

    monkeypatch.setattr(
        daily_stock,
        "load_local_daily_frames",
        lambda symbols, data_dir, min_days=daily_stock.DEFAULT_DAILY_FRAME_MIN_DAYS: {
            "AAPL": frame.copy(),
            "MSFT": frame.assign(close=frame["close"] + 10.0),
        },
    )
    monkeypatch.setattr(daily_stock, "_load_cached_daily_trader", lambda *args, **kwargs: _FakeTrader())
    monkeypatch.setattr(
        daily_stock,
        "compute_daily_feature_history",
        lambda df: pd.DataFrame(np.tile(np.arange(16, dtype=np.float32), (len(df), 1))),
    )

    prepared = daily_stock._prepare_daily_backtest_data(
        checkpoint="unused.pt",
        symbols=["AAPL", "MSFT"],
        data_dir="unused",
        days=5,
    )

    assert prepared.symbols == ("AAPL", "MSFT")
    assert prepared.close_matrix.shape == (25, 2)
    assert prepared.close_matrix[0].tolist() == pytest.approx([100.5, 110.5])
    assert prepared.timestamps[0] == datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert prepared.timestamps[-1] == datetime(2026, 1, 25, tzinfo=timezone.utc)


def test_run_backtest_via_trading_server_falls_back_to_build_signal_without_get_signal(monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01T00:00:00Z", periods=25, freq="D"),
            "open": np.linspace(100.0, 124.0, 25),
            "high": np.linspace(101.0, 125.0, 25),
            "low": np.linspace(99.0, 123.0, 25),
            "close": np.linspace(100.5, 124.5, 25),
            "volume": np.linspace(1_000.0, 1_024.0, 25),
        }
    )
    build_calls = {"count": 0}

    class _FakeTrader:
        def __init__(self) -> None:
            self.device = "cpu"
            self.SYMBOLS = ["AAPL", "MSFT"]
            self.num_symbols = len(self.SYMBOLS)
            self.num_actions = 1 + self.num_symbols
            self.obs_size = self.num_symbols * 16 + 5 + self.num_symbols
            self.cash = 0.0
            self.current_position = None
            self.position_qty = 0.0
            self.entry_price = 0.0
            self.hold_days = 0
            self.hold_hours = 0
            self.step = 0
            self.max_steps = 90

        def step_day(self):
            return None

    monkeypatch.setattr(
        daily_stock,
        "load_local_daily_frames",
        lambda symbols, data_dir, min_days=daily_stock.DEFAULT_DAILY_FRAME_MIN_DAYS: {
            "AAPL": frame.copy(),
            "MSFT": frame.copy(),
        },
    )
    monkeypatch.setattr(daily_stock, "_load_cached_daily_trader", lambda *args, **kwargs: _FakeTrader())
    monkeypatch.setattr(
        daily_stock,
        "build_signal",
        lambda *args, **kwargs: (
            build_calls.__setitem__("count", build_calls["count"] + 1)
            or SimpleNamespace(action="flat", symbol=None, direction=None, confidence=0.0, value_estimate=0.0),
            {"AAPL": 0.0, "MSFT": 0.0},
        ),
    )

    results = daily_stock.run_backtest_via_trading_server(
        checkpoint="unused.pt",
        symbols=["AAPL", "MSFT"],
        data_dir="unused",
        days=5,
    )

    assert build_calls["count"] == 5
    assert results["trades"] == pytest.approx(0.0)
    assert results["orders"] == pytest.approx(0.0)


def test_run_backtest_via_trading_server_ensemble_precomputes_feature_history_once_per_symbol(
    monkeypatch,
) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01T00:00:00Z", periods=25, freq="D"),
            "open": np.linspace(100.0, 124.0, 25),
            "high": np.linspace(101.0, 125.0, 25),
            "low": np.linspace(99.0, 123.0, 25),
            "close": np.linspace(100.5, 124.5, 25),
            "volume": np.linspace(1_000.0, 1_024.0, 25),
        }
    )
    history_calls = {"count": 0}
    ensemble_calls = {"count": 0}

    class _FakeTrader:
        def __init__(self) -> None:
            self.device = "cpu"
            self.SYMBOLS = ["AAPL", "MSFT"]
            self.num_symbols = len(self.SYMBOLS)
            self.obs_size = self.num_symbols * 16 + 5 + self.num_symbols
            self.num_actions = 1 + self.num_symbols
            self.cash = 0.0
            self.current_position = None
            self.position_qty = 0.0
            self.entry_price = 0.0
            self.hold_days = 0
            self.hold_hours = 0
            self.step = 0
            self.max_steps = 90

        def step_day(self):
            return None

    monkeypatch.setattr(
        daily_stock,
        "load_local_daily_frames",
        lambda symbols, data_dir, min_days=daily_stock.DEFAULT_DAILY_FRAME_MIN_DAYS: {
            "AAPL": frame.copy(),
            "MSFT": frame.copy(),
        },
    )
    monkeypatch.setattr(daily_stock, "_load_cached_daily_trader", lambda *args, **kwargs: _FakeTrader())
    monkeypatch.setattr(
        daily_stock,
        "compute_daily_feature_history",
        lambda df: history_calls.__setitem__("count", history_calls["count"] + 1)
        or pd.DataFrame(np.tile(np.arange(16, dtype=np.float32), (len(df), 1))),
    )
    monkeypatch.setattr(daily_stock, "_load_bare_policy", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        daily_stock,
        "_ensemble_softmax_signal",
        lambda trader, extra_policies, features, prices: (
            ensemble_calls.__setitem__("count", ensemble_calls["count"] + 1)
            or features.shape == (trader.num_symbols, 16)
        )
        and SimpleNamespace(action="flat", symbol=None, direction=None, confidence=0.0, value_estimate=0.0),
    )
    monkeypatch.setattr(
        daily_stock,
        "build_signal",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("build_signal should not be used")),
    )

    results = daily_stock.run_backtest_via_trading_server(
        checkpoint="unused.pt",
        symbols=["AAPL", "MSFT"],
        data_dir="unused",
        days=5,
        extra_checkpoints=["e1.pt"],
    )

    assert history_calls["count"] == 2
    assert ensemble_calls["count"] == 5
    assert results["trades"] == pytest.approx(0.0)
    assert results["orders"] == pytest.approx(0.0)
    assert results["total_return"] == pytest.approx(0.0)


def test_run_backtest_via_trading_server_matches_legacy_multi_position(monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01T00:00:00Z", periods=25, freq="D"),
            "open": np.linspace(100.0, 124.0, 25),
            "high": np.linspace(101.0, 125.0, 25),
            "low": np.linspace(99.0, 123.0, 25),
            "close": np.linspace(100.5, 124.5, 25),
            "volume": np.linspace(1_000.0, 1_024.0, 25),
        }
    )

    class _StaticPolicy:
        def __call__(self, obs_t):
            return (
                torch.tensor([[0.0, 4.0, 4.0, 3.5, 3.5]], dtype=torch.float32),
                torch.tensor([0.5], dtype=torch.float32),
            )

    class _FakeTrader:
        def __init__(self) -> None:
            self.device = "cpu"
            self.SYMBOLS = ["AAPL", "MSFT"]
            self.num_symbols = len(self.SYMBOLS)
            self.per_symbol_actions = 2
            self.obs_size = self.num_symbols * 16 + 5 + self.num_symbols
            self.num_actions = 1 + self.num_symbols * self.per_symbol_actions
            self.policy = _StaticPolicy()
            self.cash = 0.0
            self.current_position = None
            self.position_qty = 0.0
            self.entry_price = 0.0

        def build_observation(self, features, prices):
            del prices
            return np.asarray(features, dtype=np.float32)

        def apply_action_constraints(self, logits):
            return logits

        def step_day(self):
            return None

    monkeypatch.setattr(
        daily_stock,
        "load_local_daily_frames",
        lambda symbols, data_dir, min_days=daily_stock.DEFAULT_DAILY_FRAME_MIN_DAYS: {
            "AAPL": frame.copy(),
            "MSFT": frame.copy(),
        },
    )
    monkeypatch.setattr(daily_stock, "_load_cached_daily_trader", lambda *args, **kwargs: _FakeTrader())

    legacy = daily_stock.run_backtest(
        checkpoint="unused.pt",
        symbols=["AAPL", "MSFT"],
        data_dir="unused",
        days=5,
        allocation_pct=100.0,
        multi_position=2,
    )
    server = daily_stock.run_backtest_via_trading_server(
        checkpoint="unused.pt",
        symbols=["AAPL", "MSFT"],
        data_dir="unused",
        days=5,
        allocation_pct=100.0,
        multi_position=2,
    )

    assert server["total_return"] == pytest.approx(legacy["total_return"])
    assert server["annualized_return"] == pytest.approx(legacy["annualized_return"])
    assert server["sortino"] == pytest.approx(legacy["sortino"])
    assert server["max_drawdown"] == pytest.approx(legacy["max_drawdown"])
    assert server["trades"] == legacy["trades"]
    assert server["orders"] == pytest.approx(legacy["trades"])


def test_run_backtest_variant_matrix_via_trading_server_prepares_inputs_once(monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01T00:00:00Z", periods=25, freq="D"),
            "open": np.linspace(100.0, 124.0, 25),
            "high": np.linspace(101.0, 125.0, 25),
            "low": np.linspace(99.0, 123.0, 25),
            "close": np.linspace(100.5, 124.5, 25),
            "volume": np.linspace(1_000.0, 1_024.0, 25),
        }
    )
    history_calls = {"count": 0}

    class _FakeTrader:
        def __init__(self) -> None:
            self.device = "cpu"
            self.SYMBOLS = ["AAPL", "MSFT"]
            self.num_symbols = len(self.SYMBOLS)
            self.obs_size = self.num_symbols * 16 + 5 + self.num_symbols
            self.num_actions = 1 + self.num_symbols
            self.cash = 0.0
            self.current_position = None
            self.position_qty = 0.0
            self.entry_price = 0.0
            self.hold_days = 0
            self.hold_hours = 0
            self.step = 0
            self.max_steps = 90

        def step_day(self):
            return None

    monkeypatch.setattr(
        daily_stock,
        "load_local_daily_frames",
        lambda symbols, data_dir, min_days=daily_stock.DEFAULT_DAILY_FRAME_MIN_DAYS: {
            "AAPL": frame.copy(),
            "MSFT": frame.copy(),
        },
    )
    monkeypatch.setattr(daily_stock, "_load_cached_daily_trader", lambda *args, **kwargs: _FakeTrader())
    monkeypatch.setattr(
        daily_stock,
        "compute_daily_feature_history",
        lambda df: history_calls.__setitem__("count", history_calls["count"] + 1)
        or pd.DataFrame(np.tile(np.arange(16, dtype=np.float32), (len(df), 1))),
    )
    monkeypatch.setattr(
        daily_stock,
        "_run_backtest_via_trading_server_with_prepared_data",
        lambda **kwargs: {
            "total_return": float(kwargs["allocation_pct"]) / 100.0,
            "annualized_return": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "trades": 0.0,
            "orders": 0.0,
        },
    )

    results = daily_stock.run_backtest_variant_matrix_via_trading_server(
        checkpoint="unused.pt",
        symbols=["AAPL", "MSFT"],
        data_dir="unused",
        days=5,
        variants=[
            daily_stock.BacktestVariantSpec(name="a", allocation_pct=12.5),
            daily_stock.BacktestVariantSpec(name="b", allocation_pct=25.0, multi_position=2),
        ],
    )

    assert history_calls["count"] == 2
    assert [row["name"] for row in results] == ["a", "b"]
    assert results[0]["total_return"] == pytest.approx(0.125)
    assert results[1]["total_return"] == pytest.approx(0.25)


def test_run_backtest_multi_window_variant_matrix_via_trading_server_prepares_inputs_once(monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01T00:00:00Z", periods=40, freq="D"),
            "open": np.linspace(100.0, 139.0, 40),
            "high": np.linspace(101.0, 140.0, 40),
            "low": np.linspace(99.0, 138.0, 40),
            "close": np.linspace(100.5, 139.5, 40),
            "volume": np.linspace(1_000.0, 1_039.0, 40),
        }
    )
    history_calls = {"count": 0}

    class _FakeTrader:
        def __init__(self) -> None:
            self.device = "cpu"
            self.SYMBOLS = ["AAPL", "MSFT"]
            self.num_symbols = len(self.SYMBOLS)
            self.obs_size = self.num_symbols * 16 + 5 + self.num_symbols
            self.num_actions = 1 + self.num_symbols
            self.cash = 0.0
            self.current_position = None
            self.position_qty = 0.0
            self.entry_price = 0.0
            self.hold_days = 0
            self.hold_hours = 0
            self.step = 0
            self.max_steps = 90

        def step_day(self):
            return None

    monkeypatch.setattr(
        daily_stock,
        "load_local_daily_frames",
        lambda symbols, data_dir, min_days=daily_stock.DEFAULT_DAILY_FRAME_MIN_DAYS: {
            "AAPL": frame.copy(),
            "MSFT": frame.copy(),
        },
    )
    monkeypatch.setattr(daily_stock, "_load_cached_daily_trader", lambda *args, **kwargs: _FakeTrader())
    monkeypatch.setattr(
        daily_stock,
        "compute_daily_feature_history",
        lambda df: history_calls.__setitem__("count", history_calls["count"] + 1)
        or pd.DataFrame(np.tile(np.arange(16, dtype=np.float32), (len(df), 1))),
    )
    monkeypatch.setattr(
        daily_stock,
        "_run_backtest_variant_matrix_via_trading_server_with_prepared_data",
        lambda **kwargs: [
            {
                "name": "a",
                "allocation_pct": 12.5,
                "allocation_sizing_mode": "static",
                "multi_position": 0,
                "multi_position_min_prob_ratio": 0.3,
                "buying_power_multiplier": 1.0,
                "total_return": float(kwargs["days"]) / 100.0,
                "annualized_return": 0.0,
                "sortino": 0.0,
                "max_drawdown": 0.0,
                "trades": 0.0,
                "orders": 0.0,
            }
        ],
    )

    results = daily_stock.run_backtest_multi_window_variant_matrix_via_trading_server(
        checkpoint="unused.pt",
        symbols=["AAPL", "MSFT"],
        data_dir="unused",
        days_list=[5, 10, 5],
        variants=[daily_stock.BacktestVariantSpec(name="a", allocation_pct=12.5)],
    )

    assert history_calls["count"] == 2
    assert [row["days"] for row in results] == [5, 10]
    assert results[0]["results"][0]["total_return"] == pytest.approx(0.05)
    assert results[1]["results"][0]["total_return"] == pytest.approx(0.10)


def test_run_backtest_with_prepared_data_uses_precomputed_prices_and_timestamps() -> None:
    class _ExplodingSeries:
        @property
        def iloc(self):  # noqa: D401
            raise AssertionError("close prices should come from prepared.close_matrix")

    class _ExplodingFrame:
        @property
        def index(self):  # noqa: D401
            raise AssertionError("timestamps should come from prepared.timestamps")

        def __getitem__(self, key):  # noqa: ANN001
            if key == "close":
                return _ExplodingSeries()
            raise KeyError(key)

    class _FakeTrader:
        def __init__(self) -> None:
            self.device = "cpu"
            self.SYMBOLS = ["AAPL", "MSFT"]
            self.num_symbols = len(self.SYMBOLS)
            self.num_actions = 1 + self.num_symbols
            self.cash = 0.0
            self.current_position = None
            self.position_qty = 0.0
            self.entry_price = 0.0
            self.hold_days = 0
            self.hold_hours = 0
            self.step = 0
            self.max_steps = 90

        def get_signal(self, features, prices):  # noqa: ANN001, ARG002
            assert features.shape == (self.num_symbols, 16)
            return SimpleNamespace(action="flat", symbol=None, direction=None, confidence=0.0, value_estimate=0.0)

        def step_day(self):
            return None

    prepared = daily_stock.PreparedDailyBacktestData(
        feature_schema="rsi_v5",
        indexed={"AAPL": _ExplodingFrame(), "MSFT": _ExplodingFrame()},
        trader_template=_FakeTrader(),
        extra_policies=[],
        symbols=("AAPL", "MSFT"),
        feature_cube=np.zeros((4, 2, 16), dtype=np.float32),
        close_matrix=np.asarray(
            [
                [100.0, 200.0],
                [101.0, 201.0],
                [102.0, 202.0],
                [103.0, 203.0],
            ],
            dtype=np.float64,
        ),
        timestamps=(
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            datetime(2026, 1, 2, tzinfo=timezone.utc),
            datetime(2026, 1, 3, tzinfo=timezone.utc),
            datetime(2026, 1, 4, tzinfo=timezone.utc),
        ),
        start=1,
        min_len=4,
    )

    results = daily_stock._run_backtest_via_trading_server_with_prepared_data(
        prepared=prepared,
        symbols=["AAPL", "MSFT"],
        days=3,
    )

    assert results["trades"] == pytest.approx(0.0)
    assert results["orders"] == pytest.approx(0.0)


# ---- Tests for pending close reconciliation (BUG 3 fix) ----

class _FakePosition:
    """Minimal position stub for reconciliation tests."""
    def __init__(self, symbol: str, qty: float, avg_entry_price: float = 100.0):
        self.symbol = symbol
        self.qty = str(qty)
        self.qty_available = str(qty)
        self.avg_entry_price = str(avg_entry_price)
        self.side = "long"


def test_reconcile_pending_close_clears_when_position_gone():
    state = daily_stock.StrategyState(
        pending_close_symbol="AAPL",
        pending_close_order_id="order-123",
    )
    # No positions on broker => close filled
    daily_stock.reconcile_pending_close(state=state, live_positions={})
    assert state.pending_close_symbol is None
    assert state.pending_close_order_id is None


def test_reconcile_pending_close_readopts_when_still_open():
    state = daily_stock.StrategyState(
        pending_close_symbol="AAPL",
        pending_close_order_id="order-123",
    )
    pos = _FakePosition("AAPL", 10.0, avg_entry_price=150.0)
    live_positions = {"AAPL": pos}
    daily_stock.reconcile_pending_close(state=state, live_positions=live_positions)
    # Position still open => re-adopted
    assert state.active_symbol == "AAPL"
    assert state.active_qty == 10.0
    assert state.entry_price == 150.0
    assert state.pending_close_symbol is None


def test_reconcile_pending_close_noop_when_no_pending():
    state = daily_stock.StrategyState()
    daily_stock.reconcile_pending_close(state=state, live_positions={})
    assert state.pending_close_symbol is None
    assert state.active_symbol is None


def test_strategy_state_has_pending_close_fields():
    state = daily_stock.StrategyState()
    assert hasattr(state, "pending_close_symbol")
    assert hasattr(state, "pending_close_order_id")
    assert state.pending_close_symbol is None
    assert state.pending_close_order_id is None


# ---- Tests for RSI feature (duplicate trend_20d fix) ----

def test_daily_features_rsi_differs_from_return_20d():
    """RSI(14) at index 10 should differ from return_20d at index 2."""
    from pufferlib_market.inference_daily import compute_daily_features
    np.random.seed(42)
    n = 100
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": close + abs(np.random.randn(n) * 0.5),
        "low": close - abs(np.random.randn(n) * 0.5),
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })
    feat = compute_daily_features(df)
    assert feat.shape == (16,)
    assert np.all(np.isfinite(feat))
    # Index 2 = return_20d, Index 10 = rsi_14 (was duplicate trend_20d)
    assert feat[2] != feat[10], "RSI feature should differ from return_20d"


def test_daily_features_rsi_bounded():
    """RSI(14) should be in [-1, 1] range."""
    from pufferlib_market.inference_daily import compute_daily_features
    np.random.seed(123)
    n = 200
    close = 100 + np.cumsum(np.random.randn(n) * 2.0)
    close = np.maximum(close, 1.0)  # Prevent negative prices
    df = pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": close + abs(np.random.randn(n) * 1.0),
        "low": close - abs(np.random.randn(n) * 1.0),
        "close": close,
        "volume": np.random.randint(100, 50000, n).astype(float),
    })
    feat = compute_daily_features(df)
    rsi_val = feat[10]
    assert -1.0 <= rsi_val <= 1.0, f"RSI value {rsi_val} out of bounds"


# ---- Tests for multi-position top-K trading ----

def test_ensemble_top_k_signals_returns_multiple():
    class _StaticPolicy:
        def __init__(self, logits: list[float], value: float) -> None:
            self._logits = torch.tensor([logits], dtype=torch.float32)
            self._value = torch.tensor([value], dtype=torch.float32)

        def __call__(self, obs_t):  # noqa: ANN001
            return self._logits, self._value

    class _FakeTrader:
        SYMBOLS = ["AAPL", "MSFT", "NVDA"]

        def __init__(self) -> None:
            self.device = "cpu"
            self.num_symbols = len(self.SYMBOLS)
            self.per_symbol_actions = 2
            self.policy = _StaticPolicy(
                # flat, AAPL x2, MSFT x2, NVDA x2
                [0.0, 4.0, 4.0, 3.8, 3.8, -3.0, -3.0],
                value=0.75,
            )

        def build_observation(self, features, prices):  # noqa: ANN001, ARG002
            return np.asarray(features, dtype=np.float32)

        def apply_action_constraints(self, logits):  # noqa: ANN001
            return logits

    primary = _FakeTrader()
    extra = [
        _StaticPolicy([0.0, 3.7, 3.7, 3.6, 3.6, -2.0, -2.0], value=1.25),
    ]

    signals = daily_stock._ensemble_top_k_signals(
        primary,
        extra,
        features=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        prices={"AAPL": 100.0, "MSFT": 200.0, "NVDA": 300.0},
        k=2,
        min_prob_ratio=0.5,
    )

    assert [signal.symbol for signal in signals] == ["AAPL", "MSFT"]
    assert all(isinstance(signal, daily_stock.TradingSignal) for signal in signals)
    assert signals[0].direction == "long"
    assert signals[0].confidence > signals[1].confidence
    assert sum(signal.allocation_pct for signal in signals) == pytest.approx(1.0)
    assert signals[0].value_estimate == pytest.approx(1.0)


def test_execute_multi_position_signals_exists():
    """Test that execute_multi_position_signals function exists and has correct signature."""
    import inspect
    sig = inspect.signature(daily_stock.execute_multi_position_signals)
    params = list(sig.parameters.keys())
    assert 'signals' in params
    assert 'client' in params
    assert 'total_allocation_pct' in params
    assert 'dry_run' in params


def test_execute_multi_position_signals_closes_stale_rebalances_and_opens_new(monkeypatch) -> None:
    submitted: list[dict[str, float | str]] = []

    class _FakeClient:
        def get_account(self):
            return SimpleNamespace(equity=10_000.0, buying_power=10_000.0)

    live_positions = {
        "AAPL": SimpleNamespace(qty="5"),
        "MSFT": SimpleNamespace(qty="3"),
    }

    monkeypatch.setattr(daily_stock, "positions_by_symbol", lambda client, symbols: live_positions)
    monkeypatch.setattr(daily_stock, "_signed_position_qty", lambda pos: float(pos.qty))
    monkeypatch.setattr(
        daily_stock,
        "submit_limit_order",
        lambda client, symbol, qty, side, limit_price: submitted.append(
            {"symbol": symbol, "qty": qty, "side": side, "limit_price": limit_price}
        ),
    )

    signals = [
        daily_stock.TradingSignal("hold_aapl", "AAPL", "long", 0.9, 1.0, 0.6, 0.0),
        daily_stock.TradingSignal("open_nvda", "NVDA", "long", 0.8, 1.0, 0.4, 0.0),
    ]

    held = daily_stock.execute_multi_position_signals(
        signals,
        client=_FakeClient(),
        paper=True,
        quotes={"AAPL": 100.0, "MSFT": 200.0, "NVDA": 50.0},
        symbols=["AAPL", "MSFT", "NVDA"],
        total_allocation_pct=50.0,
        dry_run=False,
    )

    assert held == {"AAPL": 30.0, "NVDA": 40.0}
    assert submitted == [
        {"symbol": "MSFT", "qty": 3.0, "side": "sell", "limit_price": 200.0},
        {"symbol": "AAPL", "qty": 25.0, "side": "buy", "limit_price": 100.0},
        {"symbol": "NVDA", "qty": 40, "side": "buy", "limit_price": 50.0},
    ]


def test_execute_multi_position_signals_dry_run_skips_order_submission(monkeypatch) -> None:
    submitted: list[dict[str, float | str]] = []

    class _FakeClient:
        def get_account(self):
            return SimpleNamespace(equity=8_000.0, buying_power=8_000.0)

    monkeypatch.setattr(
        daily_stock,
        "positions_by_symbol",
        lambda client, symbols: {"MSFT": SimpleNamespace(qty="2")},
    )
    monkeypatch.setattr(daily_stock, "_signed_position_qty", lambda pos: float(pos.qty))
    monkeypatch.setattr(
        daily_stock,
        "submit_limit_order",
        lambda client, symbol, qty, side, limit_price: submitted.append(
            {"symbol": symbol, "qty": qty, "side": side, "limit_price": limit_price}
        ),
    )

    held = daily_stock.execute_multi_position_signals(
        [daily_stock.TradingSignal("long_nvda", "NVDA", "long", 0.7, 1.0, 1.0, 0.0)],
        client=_FakeClient(),
        paper=False,
        quotes={"MSFT": 200.0, "NVDA": 100.0},
        symbols=["MSFT", "NVDA"],
        total_allocation_pct=25.0,
        dry_run=True,
    )

    assert held == {"NVDA": pytest.approx(19.99)}
    assert submitted == []


def test_execute_multi_position_signals_respects_buying_power_cap(monkeypatch) -> None:
    class _FakeClient:
        def get_account(self):
            return SimpleNamespace(equity=10_000.0, buying_power=100.0)

    monkeypatch.setattr(daily_stock, "positions_by_symbol", lambda client, symbols: {})
    monkeypatch.setattr(daily_stock, "_signed_position_qty", lambda pos: float(pos.qty))

    held = daily_stock.execute_multi_position_signals(
        [daily_stock.TradingSignal("long_nvda", "NVDA", "long", 0.7, 1.0, 1.0, 0.0)],
        client=_FakeClient(),
        paper=True,
        quotes={"NVDA": 10.0},
        symbols=["NVDA"],
        total_allocation_pct=50.0,
        dry_run=True,
    )

    assert held == {"NVDA": pytest.approx(9.5)}


def test_execute_multi_position_signals_scales_shared_buying_power_budget(monkeypatch) -> None:
    class _FakeClient:
        def get_account(self):
            return SimpleNamespace(equity=10_000.0, buying_power=100.0)

    monkeypatch.setattr(daily_stock, "positions_by_symbol", lambda client, symbols: {})
    monkeypatch.setattr(daily_stock, "_signed_position_qty", lambda pos: float(pos.qty))

    held = daily_stock.execute_multi_position_signals(
        [
            daily_stock.TradingSignal("long_aapl", "AAPL", "long", 0.8, 1.0, 0.6, 0.0),
            daily_stock.TradingSignal("long_msft", "MSFT", "long", 0.7, 1.0, 0.4, 0.0),
        ],
        client=_FakeClient(),
        paper=True,
        quotes={"AAPL": 10.0, "MSFT": 10.0},
        symbols=["AAPL", "MSFT"],
        total_allocation_pct=50.0,
        dry_run=True,
    )

    assert held == {"AAPL": pytest.approx(5.7), "MSFT": pytest.approx(3.8)}


def test_execute_multi_position_signals_skips_malformed_allocation_fractions(monkeypatch, caplog) -> None:
    class _FakeClient:
        def get_account(self):
            return SimpleNamespace(equity=10_000.0, buying_power=10_000.0)

    monkeypatch.setattr(daily_stock, "positions_by_symbol", lambda client, symbols: {})
    monkeypatch.setattr(daily_stock, "_signed_position_qty", lambda pos: float(pos.qty))

    with caplog.at_level(logging.WARNING, logger="daily_stock_rl"):
        held = daily_stock.execute_multi_position_signals(
            [
                daily_stock.TradingSignal("bad_nan", "NVDA", "long", 0.7, 1.0, float("nan"), 0.0),
                daily_stock.TradingSignal("bad_neg", "AMD", "long", 0.7, 1.0, -0.25, 0.0),
                daily_stock.TradingSignal("good", "AAPL", "long", 0.7, 1.0, 0.5, 0.0),
            ],
            client=_FakeClient(),
            paper=True,
            quotes={"NVDA": 100.0, "AMD": 50.0, "AAPL": 20.0},
            symbols=["NVDA", "AMD", "AAPL"],
            total_allocation_pct=50.0,
            dry_run=True,
        )

    assert held == {"AAPL": pytest.approx(125.0)}
    assert "Skipping malformed portfolio signal allocation for NVDA" in caplog.text
    assert "Skipping malformed portfolio signal allocation for AMD" in caplog.text


def test_execute_multi_position_signals_with_trading_server_batches_refresh(monkeypatch) -> None:
    refresh_calls: list[list[str]] = []
    submitted: list[dict[str, object]] = []

    class _FakeServerClient:
        def get_account(self):
            return {
                "cash": 5_000.0,
                "buying_power": 5_000.0,
                "positions": {
                    "AAPL": {"qty": 5.0, "avg_entry_price": 100.0, "current_price": 100.0},
                },
            }

        def refresh_prices(self, *, symbols):
            refresh_calls.append(list(symbols))
            return {"accounts": []}

        def submit_limit_order(self, **kwargs):
            submitted.append(dict(kwargs))
            return {"order": {"id": f"order-{len(submitted)}"}}

    held = daily_stock.execute_multi_position_signals_with_trading_server(
        [
            daily_stock.TradingSignal("long_msft", "MSFT", "long", 0.7, 1.0, 1.0, 0.0),
        ],
        server_client=_FakeServerClient(),
        quotes={"AAPL": 100.0, "MSFT": 50.0},
        symbols=["AAPL", "MSFT"],
        total_allocation_pct=50.0,
        dry_run=False,
    )

    assert refresh_calls == [["AAPL", "MSFT"]]
    assert submitted == [
        {
            "symbol": "AAPL",
            "qty": 5.0,
            "side": "sell",
            "limit_price": 100.0,
            "metadata": {"strategy": "daily_stock_rl", "intent": "close_portfolio_position"},
        },
        {
            "symbol": "MSFT",
            "qty": pytest.approx(55.0),
            "side": "buy",
            "limit_price": pytest.approx(daily_stock._marketable_limit_price(50.0, "buy")),
            "metadata": {"strategy": "daily_stock_rl", "intent": "open_portfolio_position"},
        },
    ]
    assert held == {"MSFT": pytest.approx(55.0)}


def test_execute_multi_position_signals_with_trading_server_respects_buying_power_cap() -> None:
    class _FakeServerClient:
        def get_account(self):
            return {
                "cash": 10_000.0,
                "buying_power": 100.0,
                "positions": {},
            }

        def refresh_prices(self, *, symbols):
            raise AssertionError("dry-run should not refresh prices")

        def submit_limit_order(self, **kwargs):
            raise AssertionError("dry-run should not submit orders")

    held = daily_stock.execute_multi_position_signals_with_trading_server(
        [
            daily_stock.TradingSignal("long_nvda", "NVDA", "long", 0.7, 1.0, 1.0, 0.0),
        ],
        server_client=_FakeServerClient(),
        quotes={"NVDA": 10.0},
        symbols=["NVDA"],
        total_allocation_pct=50.0,
        dry_run=True,
    )

    assert held == {"NVDA": pytest.approx(9.5)}


def test_execute_multi_position_signals_with_trading_server_rebalances_existing_position() -> None:
    refresh_calls: list[list[str]] = []
    submitted: list[dict[str, object]] = []

    class _FakeServerClient:
        def get_account(self):
            return {
                "cash": 7_000.0,
                "buying_power": 7_000.0,
                "positions": {
                    "AAPL": {"qty": 30.0, "avg_entry_price": 100.0, "current_price": 100.0},
                },
            }

        def refresh_prices(self, *, symbols):
            refresh_calls.append(list(symbols))
            return {"accounts": []}

        def submit_limit_order(self, **kwargs):
            submitted.append(dict(kwargs))
            return {"order": {"id": f"order-{len(submitted)}"}}

    held = daily_stock.execute_multi_position_signals_with_trading_server(
        [
            daily_stock.TradingSignal("rebalance_aapl", "AAPL", "long", 0.8, 1.0, 0.25, 0.0),
            daily_stock.TradingSignal("open_msft", "MSFT", "long", 0.7, 1.0, 0.75, 0.0),
        ],
        server_client=_FakeServerClient(),
        quotes={"AAPL": 100.0, "MSFT": 50.0},
        symbols=["AAPL", "MSFT"],
        total_allocation_pct=50.0,
        dry_run=False,
    )

    assert refresh_calls == [["AAPL", "MSFT"]]
    assert held == {"AAPL": pytest.approx(12.5), "MSFT": pytest.approx(75.0)}
    assert submitted == [
        {
            "symbol": "AAPL",
            "qty": pytest.approx(17.5),
            "side": "sell",
            "limit_price": 100.0,
            "metadata": {"strategy": "daily_stock_rl", "intent": "rebalance_portfolio_position"},
        },
        {
            "symbol": "MSFT",
            "qty": pytest.approx(75.0),
            "side": "buy",
            "limit_price": pytest.approx(daily_stock._marketable_limit_price(50.0, "buy")),
            "metadata": {"strategy": "daily_stock_rl", "intent": "open_portfolio_position"},
        },
    ]


def test_execute_multi_position_signals_with_trading_server_keeps_position_when_loss_guard_blocks_trim(
    caplog,
) -> None:
    submitted: list[dict[str, object]] = []

    class _FakeServerClient:
        def get_account(self):
            return {
                "cash": 7_000.0,
                "buying_power": 7_000.0,
                "positions": {
                    "AAPL": {"qty": 30.0, "avg_entry_price": 100.0, "current_price": 95.0},
                },
            }

        def refresh_prices(self, *, symbols):
            return {"accounts": []}

        def submit_limit_order(self, **kwargs):
            submitted.append(dict(kwargs))
            if kwargs["side"] == "sell":
                raise RuntimeError(
                    "400: sell rejected for AAPL: limit 95.0000 is below safety floor 100.0000. "
                    "Loss-taking exits require allow_loss_exit=true and a force_exit_reason."
                )
            return {"order": {"id": f"order-{len(submitted)}"}}

    held = daily_stock.execute_multi_position_signals_with_trading_server(
        [
            daily_stock.TradingSignal("rebalance_aapl", "AAPL", "long", 0.8, 1.0, 0.25, 0.0),
            daily_stock.TradingSignal("open_msft", "MSFT", "long", 0.7, 1.0, 0.75, 0.0),
        ],
        server_client=_FakeServerClient(),
        quotes={"AAPL": 95.0, "MSFT": 50.0},
        symbols=["AAPL", "MSFT"],
        total_allocation_pct=50.0,
        dry_run=False,
    )

    assert held == {"AAPL": pytest.approx(30.0)}
    assert any(order["side"] == "sell" for order in submitted)
    assert not any(order["side"] == "buy" for order in submitted)
    assert "Server rejected ordinary sell for AAPL" in caplog.text
    assert "skipped 1 buy order(s) because loss guard kept existing positions in AAPL" in caplog.text


def test_execute_multi_position_signals_with_trading_server_scales_shared_buying_power_budget() -> None:
    class _FakeServerClient:
        def get_account(self):
            return {
                "cash": 10_000.0,
                "buying_power": 100.0,
                "positions": {},
            }

        def refresh_prices(self, *, symbols):
            raise AssertionError("dry-run should not refresh prices")

        def submit_limit_order(self, **kwargs):
            raise AssertionError("dry-run should not submit orders")

    held = daily_stock.execute_multi_position_signals_with_trading_server(
        [
            daily_stock.TradingSignal("long_aapl", "AAPL", "long", 0.8, 1.0, 0.6, 0.0),
            daily_stock.TradingSignal("long_msft", "MSFT", "long", 0.7, 1.0, 0.4, 0.0),
        ],
        server_client=_FakeServerClient(),
        quotes={"AAPL": 10.0, "MSFT": 10.0},
        symbols=["AAPL", "MSFT"],
        total_allocation_pct=50.0,
        dry_run=True,
    )

    assert held == {"AAPL": pytest.approx(5.7), "MSFT": pytest.approx(3.8)}


def test_execute_multi_position_signals_with_trading_server_skips_malformed_allocation_fractions(
    caplog,
) -> None:
    class _FakeServerClient:
        def get_account(self):
            return {
                "cash": 10_000.0,
                "buying_power": 10_000.0,
                "positions": {},
            }

        def refresh_prices(self, *, symbols):
            raise AssertionError("dry-run should not refresh prices")

        def submit_limit_order(self, **kwargs):
            raise AssertionError("dry-run should not submit orders")

    with caplog.at_level(logging.WARNING, logger="daily_stock_rl"):
        held = daily_stock.execute_multi_position_signals_with_trading_server(
            [
                daily_stock.TradingSignal("bad_nan", "NVDA", "long", 0.7, 1.0, float("nan"), 0.0),
                daily_stock.TradingSignal("bad_neg", "AMD", "long", 0.7, 1.0, -0.25, 0.0),
                daily_stock.TradingSignal("good", "AAPL", "long", 0.7, 1.0, 0.5, 0.0),
            ],
            server_client=_FakeServerClient(),
            quotes={"NVDA": 100.0, "AMD": 50.0, "AAPL": 20.0},
            symbols=["NVDA", "AMD", "AAPL"],
            total_allocation_pct=50.0,
            dry_run=True,
        )

    assert held == {"AAPL": pytest.approx(125.0)}
    assert "Skipping malformed server portfolio signal allocation for NVDA" in caplog.text
    assert "Skipping malformed server portfolio signal allocation for AMD" in caplog.text


def test_build_multi_position_signals_uses_latest_features(monkeypatch) -> None:
    class _FakeTrader:
        SYMBOLS = ["AAPL", "MSFT", "NVDA"]
        device = "cpu"
        obs_size = 3
        num_actions = 7
        num_symbols = 3
        per_symbol_actions = 2
        max_steps = 64
        cash = 0.0
        current_position = None
        position_qty = 0.0
        entry_price = 0.0

        @staticmethod
        def build_observation(features, prices):
            return np.array([features[0, 0], features[1, 0], features[2, 0]], dtype=np.float32)

        @staticmethod
        def policy(obs_t):
            return torch.tensor([[0.0, 3.7, 3.6, 3.6, 3.5, -2.0, -2.0]]), torch.tensor([[1.25]])

        @staticmethod
        def apply_action_constraints(logits):
            return logits

    monkeypatch.setattr(daily_stock, "_load_cached_daily_trader", lambda *args, **kwargs: _FakeTrader())
    monkeypatch.setattr(
        daily_stock,
        "compute_daily_features",
        lambda frame: np.full(16, float(frame["close"].iloc[-1]), dtype=np.float32),
    )

    frames = {
        "AAPL": pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2026-03-13T00:00:00Z"]),
                "open": [1.0],
                "high": [1.0],
                "low": [1.0],
                "close": [101.0],
                "volume": [1.0],
            }
        ),
        "MSFT": pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2026-03-13T00:00:00Z"]),
                "open": [1.0],
                "high": [1.0],
                "low": [1.0],
                "close": [202.0],
                "volume": [1.0],
            }
        ),
        "NVDA": pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2026-03-13T00:00:00Z"]),
                "open": [1.0],
                "high": [1.0],
                "low": [1.0],
                "close": [303.0],
                "volume": [1.0],
            }
        ),
    }

    signals = daily_stock._build_multi_position_signals(
        "dummy.ckpt",
        frames,
        quotes={"AAPL": 101.0, "MSFT": 202.0, "NVDA": 303.0},
        portfolio_value=12_500.0,
        multi_position=2,
        multi_position_min_prob_ratio=0.5,
    )

    assert [signal.symbol for signal in signals] == ["AAPL", "MSFT"]
    assert sum(signal.allocation_pct for signal in signals) == pytest.approx(1.0)
