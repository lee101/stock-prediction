from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
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
    monkeypatch.setattr(daily_stock, "shared_path_guard", lambda _path: guard)

    state_path = tmp_path / "state.json"
    state = daily_stock.StrategyState(active_symbol="AAPL", active_qty=10.0, entry_price=125.5)

    daily_stock.save_state(state, path=state_path)
    loaded = daily_stock.load_state(path=state_path)

    assert calls == ["acquire_write", "release_write", "acquire_read", "release_read"]
    assert loaded.active_symbol == "AAPL"
    assert loaded.active_qty == 10.0
    assert loaded.entry_price == 125.5


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
    assert "--symbols-file" in config.command_preview()
    assert "TSLA" not in config.command_preview()


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
    assert first.policy is not second.policy
    assert first.summary_dict() == second.summary_dict()
    first.cash = 123.0
    assert second.cash == 10000.0


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
    assert first is not second
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


def test_main_uses_default_resolved_ensemble_checkpoints(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_run_once(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(daily_stock, "run_once", _fake_run_once)
    monkeypatch.setattr(
        daily_stock,
        "_checkpoint_load_diagnostics",
        lambda _config: {"ok": True, "error": None, "primary": {"arch": "mlp"}, "extras": []},
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
    assert payload["daily_frame_min_days"] == daily_stock.DEFAULT_DAILY_FRAME_MIN_DAYS
    assert (
        payload["alpaca_daily_history_lookback_days"]
        == daily_stock.DEFAULT_ALPACA_DAILY_HISTORY_LOOKBACK_DAYS
    )
    assert payload["bar_freshness_max_age_days"] == daily_stock.DEFAULT_BAR_FRESHNESS_MAX_AGE_DAYS
    assert payload["checkpoint"] == str(checkpoint_path)
    assert "--check-config" in payload["check_command_preview"]
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
        lambda _config: {"ok": True, "error": None, "primary": {"arch": "mlp"}, "extras": []},
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


def test_main_check_config_reports_ready_for_local_setup(monkeypatch, tmp_path: Path, capsys) -> None:
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
    assert payload["summary"] == "paper once via alpaca using local data on 1 symbol with 1 checkpoint"
    assert payload["daily_frame_min_days"] == daily_stock.DEFAULT_DAILY_FRAME_MIN_DAYS
    assert (
        payload["alpaca_daily_history_lookback_days"]
        == daily_stock.DEFAULT_ALPACA_DAILY_HISTORY_LOOKBACK_DAYS
    )
    assert payload["bar_freshness_max_age_days"] == daily_stock.DEFAULT_BAR_FRESHNESS_MAX_AGE_DAYS
    assert "--check-config" in payload["check_command_preview"]
    assert "--dry-run" in payload["run_command_preview"]
    assert payload["run_command_preview"] == payload["safe_command_preview"]
    assert any("Run a safe dry run:" in step for step in payload["next_steps"])


def test_main_check_config_text_emits_human_ready_summary(monkeypatch, tmp_path: Path, capsys) -> None:
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
    assert "Suggested commands:" in captured.err
    assert f"- dry run: {payload['safe_command_preview']}" in captured.err
    assert "Additional next steps:" not in captured.err


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
    assert "Check config: python trade_daily_stock_prod.py --check-config" in stderr
    assert "Try a safe dry run: python trade_daily_stock_prod.py --once --paper --dry-run" in stderr
    assert "Additional context:" in stderr
    assert "- quote provider returned malformed payload" in stderr


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
        starting_cash=25_000.0,
        buying_power_multiplier=2.0,
    )

    assert captured["legacy"]["allocation_pct"] == 12.5
    assert captured["server"]["allocation_pct"] == 12.5
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

    with caplog.at_level(logging.WARNING, logger="daily_stock_rl"):
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
    assert '"event": "daily_stock_run_once_failed"' in failure_records[-1]
    assert '"stage": "build_trading_server_client"' in failure_records[-1]
    assert '"error_type": "RuntimeError"' in failure_records[-1]
    assert '"error": "server unavailable"' in failure_records[-1]
    assert '"execution_backend": "trading_server"' in failure_records[-1]
    assert '"server_url": "https://server.internal:8050"' in failure_records[-1]
    assert "run_once stage: build_trading_server_client" in getattr(exc_info.value, "__notes__", [])


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
