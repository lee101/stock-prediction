from __future__ import annotations

import json
import sys
import types
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import compare_rl_chronos2_marketsim as compare


def test_normalize_symbols_dedupes_uppercases_and_rejects_empty() -> None:
    assert compare._normalize_symbols([" aapl ", "MSFT", "aapl", "", " msft "]) == ("AAPL", "MSFT")

    with pytest.raises(ValueError, match="At least one symbol"):
        compare._normalize_symbols(["", "   "])


def test_resolve_date_window_uses_max_steps_when_start_date_omitted() -> None:
    args = SimpleNamespace(end_date="2026-04-08", start_date=None, max_steps=5)

    start_day, end_day, max_steps = compare._resolve_date_window(args)

    assert (start_day, end_day, max_steps) == (date(2026, 4, 3), date(2026, 4, 8), 5)


def test_resolve_date_window_uses_explicit_default_when_max_steps_omitted() -> None:
    args = SimpleNamespace(end_date="2026-04-08", start_date=None, max_steps=None)

    start_day, end_day, max_steps = compare._resolve_date_window(args)

    assert (start_day, end_day, max_steps) == (
        date(2025, 12, 9),
        date(2026, 4, 8),
        compare.DEFAULT_MAX_STEPS,
    )


def test_resolve_date_window_infers_steps_from_explicit_span_when_max_steps_omitted() -> None:
    args = SimpleNamespace(end_date="2026-04-08", start_date="2026-04-05", max_steps=None)

    start_day, end_day, max_steps = compare._resolve_date_window(args)

    assert (start_day, end_day, max_steps) == (date(2026, 4, 5), date(2026, 4, 8), 3)


def test_resolve_date_window_requires_max_steps_to_match_explicit_span() -> None:
    args = SimpleNamespace(end_date="2026-04-08", start_date="2026-04-05", max_steps=10)

    with pytest.raises(ValueError, match="span is 3 days but --max-steps was 10"):
        compare._resolve_date_window(args)


def test_loaded_symbols_splits_stock_and_crypto() -> None:
    stocks, crypto = compare._loaded_symbols(("AAPL", "ethusd", "MSFT", "BTCUSD"))
    assert stocks == ("AAPL", "MSFT")
    assert crypto == ("ETHUSD", "BTCUSD")


def test_chronos_history_start_date_requires_non_negative_warmup() -> None:
    assert compare._chronos_history_start_date(date(2026, 4, 8), 5) == date(2026, 4, 3)

    with pytest.raises(ValueError, match="chronos_warmup_days must be non-negative"):
        compare._chronos_history_start_date(date(2026, 4, 8), -1)


def test_symbol_label_truncates_and_counts_extra_symbols() -> None:
    label = compare._symbol_label(("AAPL", "MSFT", "BTCUSD", "ETHUSD"))
    assert label == "AAPL_MSFT_BTCUSD__PLUS1"


def test_resolve_output_dir_autogenerates_stable_path_and_avoids_overwrite(tmp_path) -> None:
    first = compare._resolve_output_dir(
        None,
        start_day=date(2026, 4, 5),
        end_day=date(2026, 4, 8),
        symbols=("AAPL", "MSFT", "BTCUSD", "ETHUSD"),
        max_steps=3,
        root=str(tmp_path),
    )
    assert first == tmp_path / "2026-04-05__2026-04-08__3d__AAPL_MSFT_BTCUSD__PLUS1"

    first.mkdir()
    second = compare._resolve_output_dir(
        None,
        start_day=date(2026, 4, 5),
        end_day=date(2026, 4, 8),
        symbols=("AAPL", "MSFT", "BTCUSD", "ETHUSD"),
        max_steps=3,
        root=str(tmp_path),
    )
    assert second == tmp_path / "2026-04-05__2026-04-08__3d__AAPL_MSFT_BTCUSD__PLUS1__run2"


def test_resolve_output_dir_respects_explicit_path(tmp_path) -> None:
    explicit = compare._resolve_output_dir(
        str(tmp_path / "custom-output"),
        start_day=date(2026, 4, 5),
        end_day=date(2026, 4, 8),
        symbols=("AAPL",),
        max_steps=3,
        root=str(tmp_path),
    )
    assert explicit == tmp_path / "custom-output"


def test_reserve_output_dir_claims_next_available_auto_directory(tmp_path) -> None:
    first = compare._reserve_output_dir(
        None,
        start_day=date(2026, 4, 5),
        end_day=date(2026, 4, 8),
        symbols=("AAPL", "MSFT", "BTCUSD", "ETHUSD"),
        max_steps=3,
        root=str(tmp_path),
    )
    second = compare._reserve_output_dir(
        None,
        start_day=date(2026, 4, 5),
        end_day=date(2026, 4, 8),
        symbols=("AAPL", "MSFT", "BTCUSD", "ETHUSD"),
        max_steps=3,
        root=str(tmp_path),
    )

    assert first == tmp_path / "2026-04-05__2026-04-08__3d__AAPL_MSFT_BTCUSD__PLUS1"
    assert second == tmp_path / "2026-04-05__2026-04-08__3d__AAPL_MSFT_BTCUSD__PLUS1__run2"
    assert first.is_dir()
    assert second.is_dir()


def test_reserve_output_dir_rejects_existing_explicit_path(tmp_path) -> None:
    existing = tmp_path / "custom-output"
    existing.mkdir()

    with pytest.raises(FileExistsError):
        compare._reserve_output_dir(
            str(existing),
            start_day=date(2026, 4, 5),
            end_day=date(2026, 4, 8),
            symbols=("AAPL",),
            max_steps=3,
            root=str(tmp_path),
        )


def test_reserve_output_dir_creates_explicit_path_when_missing(tmp_path) -> None:
    explicit = tmp_path / "custom-output"

    reserved = compare._reserve_output_dir(
        str(explicit),
        start_day=date(2026, 4, 5),
        end_day=date(2026, 4, 8),
        symbols=("AAPL",),
        max_steps=3,
        root=str(tmp_path),
    )

    assert reserved == explicit
    assert reserved.is_dir()


def test_should_render_rl_trace_only_when_visual_output_is_requested() -> None:
    assert compare._should_render_rl_trace(skip_video=False, skip_html=True) is True
    assert compare._should_render_rl_trace(skip_video=True, skip_html=False) is True
    assert compare._should_render_rl_trace(skip_video=True, skip_html=True) is False


def test_build_chronos_equity_series_uses_utc_dates() -> None:
    result = SimpleNamespace(
        daily_states=[
            SimpleNamespace(date="2026-04-07", total_equity=101_000.0),
            SimpleNamespace(date="2026-04-08", total_equity=102_500.0),
        ]
    )

    series = compare._build_chronos_equity_series(result)

    assert list(series.index) == [
        pd.Timestamp("2026-04-07T00:00:00Z"),
        pd.Timestamp("2026-04-08T00:00:00Z"),
    ]
    assert list(series.values) == [101_000.0, 102_500.0]
    assert series.name == "chronos2_equity"


def test_trace_from_daily_sim_records_entries_on_position_changes() -> None:
    data = SimpleNamespace(
        symbols=["AAPL", "MSFT"],
        prices=np.array(
            [
                [[100.0, 101.0, 99.0, 100.5], [200.0, 201.0, 199.0, 200.5]],
                [[101.0, 102.0, 100.0, 101.5], [201.0, 202.0, 200.0, 201.5]],
                [[102.0, 103.0, 101.0, 102.5], [202.0, 203.0, 201.0, 202.5]],
                [[103.0, 104.0, 102.0, 103.5], [203.0, 204.0, 202.0, 203.5]],
            ],
            dtype=np.float32,
        ),
    )
    result = SimpleNamespace(
        position_history=[
            SimpleNamespace(sym=0, is_short=False, qty=1.0, entry_price=100.5),
            SimpleNamespace(sym=0, is_short=False, qty=1.0, entry_price=100.5),
            SimpleNamespace(sym=1, is_short=True, qty=0.5, entry_price=202.5),
        ],
        equity_curve=[100_000.0, 100_500.0, 100_750.0, 101_000.0],
        actions=[1, 1, 2],
        evaluated_steps=3,
    )
    dates = pd.date_range("2026-04-06", periods=4, freq="D", tz="UTC")

    trace = compare._trace_from_daily_sim(data=data, result=result, dates=dates)

    assert trace.symbols == ["AAPL", "MSFT"]
    assert len(trace.frames) == 3
    assert trace.frames[0].orders and trace.frames[0].orders[0].sym == 0
    assert trace.frames[1].orders == []
    assert trace.frames[2].orders and trace.frames[2].orders[0].sym == 1
    assert trace.frames[2].position_is_short is True


def test_main_writes_failure_summary_on_runtime_error(monkeypatch, tmp_path, capsys) -> None:
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    monkeypatch.setattr(compare, "_reserve_output_dir", lambda *args, **kwargs: out_dir)

    def _boom(**kwargs):  # noqa: ANN003
        raise RuntimeError("sim exploded")

    monkeypatch.setattr(compare, "_run_comparison", _boom)

    rc = compare.main(["--end-date", "2026-04-08", "--symbols", "AAPL"])

    captured = capsys.readouterr()
    assert rc == 1
    assert "sim exploded" in captured.err
    payload = json.loads((out_dir / "failure.json").read_text(encoding="utf-8"))
    assert payload["status"] == "error"
    assert payload["error_type"] == "RuntimeError"
    assert payload["error"] == "sim exploded"
    assert payload["symbols"] == ["AAPL"]
    assert payload["output_dir"] == str(out_dir)
    assert payload["window"] == {
        "start_date": "2025-12-09",
        "end_date": "2026-04-08",
        "max_steps": 120,
    }


def test_safe_write_json_warns_and_returns_none_on_write_failure(monkeypatch, tmp_path, capsys) -> None:
    fake_reporting = types.SimpleNamespace(safe_write_summary_json=lambda path, payload: (None, "disk full"))
    monkeypatch.setitem(sys.modules, "binance_worksteal.reporting", fake_reporting)

    written = compare._safe_write_json(tmp_path / "summary.json", {"tool": "compare"})

    captured = capsys.readouterr()
    assert written is None
    assert "WARN: failed to write JSON" in captured.err
    assert "disk full" in captured.err


def test_main_prints_payload_and_returns_zero_on_success(monkeypatch, tmp_path, capsys) -> None:
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    monkeypatch.setattr(compare, "_reserve_output_dir", lambda *args, **kwargs: out_dir)
    monkeypatch.setattr(
        compare,
        "_run_comparison",
        lambda **kwargs: {"summary": {"status": "ok"}, "artifacts": {"summary_json": str(out_dir / "summary.json")}},
    )

    rc = compare.main(["--end-date", "2026-04-08", "--symbols", "AAPL"])

    captured = capsys.readouterr()
    assert rc == 0
    payload = json.loads(captured.out)
    assert payload["summary"] == {"status": "ok"}
    assert payload["artifacts"]["summary_json"] == str(out_dir / "summary.json")
    assert captured.err == ""


def test_main_returns_nonzero_on_invalid_window_before_output_dir_resolution(capsys) -> None:
    rc = compare.main(
        ["--end-date", "2026-04-08", "--start-date", "2026-04-05", "--max-steps", "10", "--symbols", "AAPL"]
    )

    captured = capsys.readouterr()
    assert rc == 1
    assert "span is 3 days but --max-steps was 10" in captured.err


def test_main_returns_nonzero_when_explicit_output_dir_already_exists(tmp_path, capsys) -> None:
    out_dir = tmp_path / "existing"
    out_dir.mkdir()

    rc = compare.main(
        [
            "--end-date",
            "2026-04-08",
            "--symbols",
            "AAPL",
            "--output-dir",
            str(out_dir),
        ]
    )

    captured = capsys.readouterr()
    assert rc == 1
    assert "File exists" in captured.err or "exists" in captured.err


def test_run_comparison_uses_explicit_chronos_warmup_days(monkeypatch, tmp_path) -> None:
    config_calls: list[dict[str, object]] = []
    loader_configs: list[object] = []
    backtester_calls: list[dict[str, object]] = []
    summary_writes: list[tuple[str, dict[str, object]]] = []
    plot_calls: list[dict[str, object]] = []

    fake_torch = types.SimpleNamespace(
        device=lambda name: name,
        cuda=types.SimpleNamespace(is_available=lambda: False),
    )

    class _FakeDataConfigLong:
        def __init__(self, **kwargs) -> None:
            config_calls.append(kwargs)
            self.__dict__.update(kwargs)

    class _FakeDailyDataLoader:
        def __init__(self, config) -> None:
            loader_configs.append(config)
            self.config = config

        def load_all_symbols(self) -> None:
            return None

    class _FakeRigorousBacktester:
        def __init__(self, data_loader, initial_capital: float, leverage: float) -> None:
            backtester_calls.append(
                {
                    "data_loader": data_loader,
                    "initial_capital": initial_capital,
                    "leverage": leverage,
                }
            )

        def run_chronos_full_v2(self, symbols, start_day, end_day, exit_mode, top_n):
            return SimpleNamespace(
                strategy_name="chronos2",
                total_return_pct=12.5,
                annualized_return_pct=18.0,
                sharpe_ratio=1.2,
                max_drawdown_pct=-6.0,
                win_rate=0.55,
                total_trades=7,
                daily_states=[
                    SimpleNamespace(date=str(start_day), total_equity=100_000.0),
                    SimpleNamespace(date=str(end_day), total_equity=101_250.0),
                ],
            )

    fake_pnl_module = types.SimpleNamespace(
        ExitMode=lambda value: value,
        RigorousBacktester=_FakeRigorousBacktester,
    )
    fake_config_module = types.SimpleNamespace(DataConfigLong=_FakeDataConfigLong)
    fake_data_module = types.SimpleNamespace(DailyDataLoader=_FakeDailyDataLoader)

    fake_loaded_policy = SimpleNamespace(
        policy=object(),
        arch="mlp",
        hidden_size=64,
        action_allocation_bins=3,
        action_level_bins=2,
        action_max_offset_bps=10.0,
    )
    fake_rl_data = SimpleNamespace(
        num_symbols=2,
        symbols=["AAPL", "BTCUSD"],
        features=np.zeros((4, 2, 16), dtype=np.float32),
        prices=np.array(
            [
                [[100.0, 101.0, 99.0, 100.5], [50.0, 51.0, 49.0, 50.5]],
                [[101.0, 102.0, 100.0, 101.5], [51.0, 52.0, 50.0, 51.5]],
                [[102.0, 103.0, 101.0, 102.5], [52.0, 53.0, 51.0, 52.5]],
                [[103.0, 104.0, 102.0, 103.5], [53.0, 54.0, 52.0, 53.5]],
            ],
            dtype=np.float32,
        ),
        num_timesteps=4,
    )
    fake_rl_result = SimpleNamespace(
        total_return=0.1,
        sortino=1.5,
        max_drawdown=-0.05,
        num_trades=3,
        win_rate=0.66,
        avg_hold_steps=2.0,
        equity_curve=[100_000.0, 100_500.0, 101_250.0],
        actions=[1, 0],
        position_history=[],
        evaluated_steps=2,
    )

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "marketsimlong.config", fake_config_module)
    monkeypatch.setitem(sys.modules, "marketsimlong.data", fake_data_module)
    monkeypatch.setitem(sys.modules, "pnlforecast.comparison_v2", fake_pnl_module)
    monkeypatch.setitem(
        sys.modules,
        "pufferlib_market.evaluate_multiperiod",
        types.SimpleNamespace(
            load_policy=lambda *args, **kwargs: fake_loaded_policy,
            make_policy_fn=lambda *args, **kwargs: "policy-fn",
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "pufferlib_market.hourly_replay",
        types.SimpleNamespace(
            read_mktd=lambda path: fake_rl_data,
            simulate_daily_policy=lambda *args, **kwargs: fake_rl_result,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "src.marketsim_video",
        types.SimpleNamespace(
            render_html_plotly=lambda *args, **kwargs: None,
            render_mp4=lambda *args, **kwargs: None,
            MarketsimTrace=object,
            OrderTick=object,
        ),
    )
    monkeypatch.setattr(compare, "_tail_for_window", lambda data, max_steps: data)
    monkeypatch.setattr(
        compare,
        "_plot_equity_curves",
        lambda **kwargs: plot_calls.append(kwargs) or kwargs["out_path"],
    )
    monkeypatch.setattr(
        compare,
        "_safe_write_json",
        lambda path, payload: summary_writes.append((str(path), payload)) or path,
    )
    monkeypatch.setattr(compare, "_trace_from_daily_sim", lambda **kwargs: None)

    args = SimpleNamespace(
        cpu=True,
        rl_checkpoint="checkpoint.pt",
        rl_data_path="data.bin",
        initial_capital=100_000.0,
        fee_rate=0.001,
        fill_buffer_bps=5.0,
        rl_max_leverage=2.0,
        periods_per_year=252.0,
        chronos_leverage=1.5,
        chronos_exit_mode="hold",
        chronos_top_n=2,
        chronos_warmup_days=45,
        skip_video=True,
        skip_html=True,
    )

    payload = compare._run_comparison(
        args=args,
        output_dir=tmp_path,
        start_day=date(2026, 4, 5),
        end_day=date(2026, 4, 8),
        max_steps=3,
        symbols=("AAPL", "BTCUSD"),
    )

    assert config_calls[0]["start_date"] == date(2026, 2, 19)
    assert config_calls[0]["end_date"] == date(2026, 4, 8)
    assert payload["summary"]["chronos2"]["warmup_days"] == 45
    assert backtester_calls[0]["initial_capital"] == 100_000.0
    assert backtester_calls[0]["leverage"] == 1.5
    assert plot_calls and plot_calls[0]["title"].startswith("RL vs Chronos2 Daily Marketsim")
    assert [Path(path).name for path, _ in summary_writes] == ["summary.json", "artifacts.json"]
