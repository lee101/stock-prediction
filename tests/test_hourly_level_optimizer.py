from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
from scripts import eval_hourly_level_optimizer

from src.tradinglib.hourly_level_optimizer import (
    HourlyLevelSearchConfig,
    optimize_levels_for_window,
    optimize_long_levels_for_window,
    optimize_short_levels_for_window,
    replay_long_levels_for_window,
    replay_short_levels_for_window,
    walk_forward_hourly_level_search,
)


def _bars(closes: list[float]) -> pd.DataFrame:
    ts0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    prev = closes[0]
    for i, close in enumerate(closes):
        # Every bar dips 1% below previous close and then can take +1% from
        # that entry. This gives the optimiser an obvious grid optimum.
        low = prev * 0.99
        high = low * 1.01
        rows.append(
            {
                "timestamp": ts0 + timedelta(hours=i),
                "open": prev,
                "high": high,
                "low": low,
                "close": close,
            }
        )
        prev = close
    return pd.DataFrame(rows)


def test_optimize_long_levels_finds_profitable_limit_pair():
    frame = _bars([100.0] * 16)
    cfg = HourlyLevelSearchConfig(
        entry_bps_grid=(50.0, 100.0, 150.0),
        take_profit_bps_grid=(50.0, 100.0, 150.0),
        fill_buffer_bps=0.0,
        fee_bps=0.0,
        max_hold_bars=4,
        min_train_trades=1,
    )

    result = optimize_long_levels_for_window(frame, cfg)

    assert result.entry_bps == 100.0
    assert result.take_profit_bps == 100.0
    assert result.train_trades > 0
    assert result.train_return_pct > 0
    assert result.train_win_rate_pct == pytest.approx(100.0)


def test_optimize_short_levels_finds_profitable_limit_pair():
    ts0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    prev = 100.0
    for i in range(16):
        entry = prev * 1.01
        rows.append(
            {
                "timestamp": ts0 + timedelta(hours=i),
                "open": prev,
                "high": entry,
                "low": entry * 0.99,
                "close": prev,
            }
        )
    frame = pd.DataFrame(rows)
    cfg = HourlyLevelSearchConfig(
        entry_bps_grid=(50.0, 100.0, 150.0),
        take_profit_bps_grid=(50.0, 100.0, 150.0),
        fill_buffer_bps=0.0,
        fee_bps=0.0,
        max_hold_bars=4,
        min_train_trades=1,
    )

    result = optimize_short_levels_for_window(frame, cfg, prev_close0=100.0)

    assert result.side == "short"
    assert result.entry_bps == 100.0
    assert result.take_profit_bps == 100.0
    assert result.train_return_pct > 0
    assert result.train_win_rate_pct == pytest.approx(100.0)


def test_replay_long_levels_uses_penetration_buffer():
    frame = pd.DataFrame(
        {
            "timestamp": [datetime(2026, 1, 1, tzinfo=timezone.utc)],
            "open": [100.0],
            "high": [100.5],
            "low": [99.0],
            "close": [100.0],
        }
    )
    no_buffer_ret, no_buffer_trades, _ = replay_long_levels_for_window(
        frame,
        100.0,
        50.0,
        HourlyLevelSearchConfig(fill_buffer_bps=0.0, fee_bps=0.0),
        prev_close0=100.0,
    )
    buffered_ret, buffered_trades, _ = replay_long_levels_for_window(
        frame,
        100.0,
        50.0,
        HourlyLevelSearchConfig(fill_buffer_bps=5.0, fee_bps=0.0),
        prev_close0=100.0,
    )

    assert no_buffer_trades == 1
    assert no_buffer_ret > 0
    assert buffered_trades == 0
    assert buffered_ret == 0.0


def test_replay_short_levels_uses_penetration_buffer():
    frame = pd.DataFrame(
        {
            "timestamp": [datetime(2026, 1, 1, tzinfo=timezone.utc)],
            "open": [100.0],
            "high": [101.0],
            "low": [100.0],
            "close": [100.0],
        }
    )
    no_buffer_ret, no_buffer_trades, _ = replay_short_levels_for_window(
        frame,
        100.0,
        50.0,
        HourlyLevelSearchConfig(fill_buffer_bps=0.0, fee_bps=0.0),
        prev_close0=100.0,
    )
    buffered_ret, buffered_trades, _ = replay_short_levels_for_window(
        frame,
        100.0,
        50.0,
        HourlyLevelSearchConfig(fill_buffer_bps=5.0, fee_bps=0.0),
        prev_close0=100.0,
    )

    assert no_buffer_trades == 1
    assert no_buffer_ret > 0
    assert buffered_trades == 0
    assert buffered_ret == 0.0


def test_walk_forward_reoptimizes_without_forward_lookahead():
    frame = _bars([100.0] * 32)
    cfg = HourlyLevelSearchConfig(
        lookback_bars=8,
        forward_bars=8,
        entry_bps_grid=(50.0, 100.0),
        take_profit_bps_grid=(50.0, 100.0),
        fill_buffer_bps=0.0,
        fee_bps=0.0,
        max_hold_bars=4,
    )

    result = walk_forward_hourly_level_search(frame, symbol="aapl", config=cfg)

    assert result.symbol == "AAPL"
    assert len(result.windows) == 3
    assert all(window.entry_bps == 100.0 for window in result.windows)
    assert result.total_return_pct > 0


def test_walk_forward_can_choose_short_side():
    ts0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(32):
        rows.append(
            {
                "timestamp": ts0 + timedelta(hours=i),
                "open": 100.0,
                "high": 101.0,
                "low": 99.99,
                "close": 100.0,
            }
        )
    cfg = HourlyLevelSearchConfig(
        lookback_bars=8,
        forward_bars=8,
        entry_bps_grid=(100.0,),
        take_profit_bps_grid=(100.0,),
        fill_buffer_bps=0.0,
        fee_bps=0.0,
        max_hold_bars=4,
        side_mode="both",
    )

    result = walk_forward_hourly_level_search(pd.DataFrame(rows), symbol="BTCUSDT", config=cfg)

    assert result.windows
    assert all(window.side == "short" for window in result.windows)
    assert result.total_return_pct > 0


def test_optimize_levels_honors_short_only_side_mode():
    frame = _bars([100.0] * 16)
    cfg = HourlyLevelSearchConfig(
        entry_bps_grid=(100.0,),
        take_profit_bps_grid=(100.0,),
        fill_buffer_bps=0.0,
        fee_bps=0.0,
        side_mode="short",
    )

    result = optimize_levels_for_window(frame, cfg)

    assert result.side == "short"


def test_walk_forward_can_skip_deployment_after_weak_training_window():
    frame = _bars([100.0] * 24)
    cfg = HourlyLevelSearchConfig(
        lookback_bars=8,
        forward_bars=8,
        entry_bps_grid=(100.0,),
        take_profit_bps_grid=(100.0,),
        fill_buffer_bps=0.0,
        fee_bps=0.0,
        min_deploy_train_return_pct=1_000.0,
    )

    result = walk_forward_hourly_level_search(frame, symbol="AAPL", config=cfg)

    assert result.windows
    assert all(window.forward_trades == 0 for window in result.windows)
    assert result.total_return_pct == 0.0


def test_replay_long_levels_stop_loss_caps_bad_forward_window():
    frame = pd.DataFrame(
        {
            "timestamp": [
                datetime(2026, 1, 1, tzinfo=timezone.utc),
                datetime(2026, 1, 1, 1, tzinfo=timezone.utc),
            ],
            "open": [100.0, 99.0],
            "high": [100.0, 99.0],
            "low": [99.0, 90.0],
            "close": [99.0, 90.0],
        }
    )

    uncapped_ret, _, _ = replay_long_levels_for_window(
        frame,
        100.0,
        100.0,
        HourlyLevelSearchConfig(fill_buffer_bps=0.0, fee_bps=0.0, max_hold_bars=8),
        prev_close0=100.0,
    )
    stopped_ret, _, _ = replay_long_levels_for_window(
        frame,
        100.0,
        100.0,
        HourlyLevelSearchConfig(
            fill_buffer_bps=0.0,
            fee_bps=0.0,
            max_hold_bars=8,
            stop_loss_bps=100.0,
        ),
        prev_close0=100.0,
    )

    assert uncapped_ret < stopped_ret
    assert stopped_ret == pytest.approx(-1.0)


def test_replay_long_levels_prefers_stop_loss_when_stop_and_take_profit_hit_same_bar():
    frame = pd.DataFrame(
        {
            "timestamp": [
                datetime(2026, 1, 1, tzinfo=timezone.utc),
                datetime(2026, 1, 1, 1, tzinfo=timezone.utc),
            ],
            "open": [100.0, 99.0],
            "high": [100.0, 101.0],
            "low": [99.0, 98.0],
            "close": [99.0, 100.0],
        }
    )

    ret, trades, win_rate = replay_long_levels_for_window(
        frame,
        100.0,
        100.0,
        HourlyLevelSearchConfig(
            fill_buffer_bps=0.0,
            fee_bps=0.0,
            max_hold_bars=8,
            stop_loss_bps=100.0,
        ),
        prev_close0=100.0,
    )

    assert trades == 1
    assert ret == pytest.approx(-1.0)
    assert win_rate == 0.0


def test_hourly_level_config_rejects_negative_costs():
    with pytest.raises(ValueError, match="fee_bps must be finite and non-negative"):
        walk_forward_hourly_level_search(
            _bars([100.0] * 16),
            config=HourlyLevelSearchConfig(fee_bps=-1.0),
        )


def test_eval_hourly_level_optimizer_rejects_invalid_config_before_reading_data(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["eval_hourly_level_optimizer.py", "--symbols", "AAPL", "--fill-buffer-bps", "-1"],
    )
    monkeypatch.setattr(
        eval_hourly_level_optimizer,
        "_find_hourly_csv",
        lambda *_: pytest.fail("invalid config should fail before looking up hourly CSV data"),
    )

    def fail_read_csv(_path):
        pytest.fail("invalid config should fail before reading hourly CSV data")

    monkeypatch.setattr(eval_hourly_level_optimizer.pd, "read_csv", fail_read_csv)

    assert eval_hourly_level_optimizer.main() == 2
    assert "fill_buffer_bps must be finite and non-negative" in capsys.readouterr().err


def test_eval_hourly_level_optimizer_rejects_malformed_grid_before_reading_data(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["eval_hourly_level_optimizer.py", "--symbols", "AAPL", "--entry-bps-grid", "5,nope,10"],
    )
    monkeypatch.setattr(
        eval_hourly_level_optimizer,
        "_find_hourly_csv",
        lambda *_: pytest.fail("invalid grid should fail before looking up hourly CSV data"),
    )
    monkeypatch.setattr(
        eval_hourly_level_optimizer.pd,
        "read_csv",
        lambda *_: pytest.fail("invalid grid should fail before reading hourly CSV data"),
    )

    assert eval_hourly_level_optimizer.main() == 2
    assert "grid contains non-numeric value: nope" in capsys.readouterr().err


def test_eval_hourly_level_optimizer_rejects_invalid_date_filter_before_reading_data(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["eval_hourly_level_optimizer.py", "--symbols", "AAPL", "--start", "not-a-date"],
    )
    monkeypatch.setattr(
        eval_hourly_level_optimizer,
        "_find_hourly_csv",
        lambda *_: pytest.fail("invalid date should fail before looking up hourly CSV data"),
    )
    monkeypatch.setattr(
        eval_hourly_level_optimizer.pd,
        "read_csv",
        lambda *_: pytest.fail("invalid date should fail before reading hourly CSV data"),
    )

    assert eval_hourly_level_optimizer.main() == 2
    assert "start must be a valid timestamp" in capsys.readouterr().err


def test_eval_hourly_level_optimizer_rejects_reversed_date_filter(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_hourly_level_optimizer.py",
            "--symbols",
            "AAPL",
            "--start",
            "2026-01-02",
            "--end",
            "2026-01-01",
        ],
    )
    monkeypatch.setattr(
        eval_hourly_level_optimizer,
        "_find_hourly_csv",
        lambda *_: pytest.fail("reversed dates should fail before looking up hourly CSV data"),
    )

    assert eval_hourly_level_optimizer.main() == 2
    assert "start must be <= end" in capsys.readouterr().err


def test_eval_hourly_level_optimizer_writes_report_atomically(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    stocks_dir = data_root / "stocks"
    stocks_dir.mkdir(parents=True)
    _bars([100.0] * 18).to_csv(stocks_dir / "AAPL.csv", index=False)
    writes = []

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_hourly_level_optimizer.py",
            "--data-root",
            str(data_root),
            "--symbols",
            "AAPL",
            "--lookback-bars",
            "8",
            "--forward-bars",
            "8",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    monkeypatch.setattr(
        eval_hourly_level_optimizer,
        "write_json_atomic",
        lambda path, payload, **kwargs: writes.append((path, payload, kwargs)),
    )

    assert eval_hourly_level_optimizer.main() == 0

    assert len(writes) == 1
    out_path, payload, kwargs = writes[0]
    assert out_path.parent == tmp_path / "out"
    assert payload["selection"] == {
        "data_root": str(data_root),
        "symbols": ["AAPL"],
        "max_symbols": 0,
        "start": None,
        "end": None,
    }
    assert payload["results"][0]["symbol"] == "AAPL"
    assert kwargs == {"default": str, "sort_keys": True}


def test_eval_hourly_level_optimizer_accepts_timezone_aware_date_filters(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    stocks_dir = data_root / "stocks"
    stocks_dir.mkdir(parents=True)
    _bars([100.0] * 24).to_csv(stocks_dir / "AAPL.csv", index=False)
    writes = []

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_hourly_level_optimizer.py",
            "--data-root",
            str(data_root),
            "--symbols",
            "AAPL",
            "--start",
            "2026-01-01T08:00:00Z",
            "--end",
            "2026-01-01T23:00:00+00:00",
            "--lookback-bars",
            "8",
            "--forward-bars",
            "8",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    monkeypatch.setattr(
        eval_hourly_level_optimizer,
        "write_json_atomic",
        lambda path, payload, **kwargs: writes.append((path, payload, kwargs)),
    )

    assert eval_hourly_level_optimizer.main() == 0

    assert len(writes) == 1
    payload = writes[0][1]
    assert payload["selection"]["start"] == "2026-01-01T08:00:00+00:00"
    assert payload["selection"]["end"] == "2026-01-01T23:00:00+00:00"
    assert payload["results"][0]["n_windows"] == 1
