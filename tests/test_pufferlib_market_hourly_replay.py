from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pufferlib_market.hourly_replay import (
    HourlyMarket,
    MktdData,
    replay_hourly_frozen_daily_actions,
    simulate_daily_policy,
    simulate_hourly_policy,
)
from pufferlib_market.intrabar_replay import HourlyOHLC, simulate_daily_policy_intrabar


def _single_symbol_daily_data(close: np.ndarray) -> MktdData:
    close = np.asarray(close, dtype=np.float32)
    features = np.zeros((len(close), 1, 16), dtype=np.float32)
    prices = np.zeros((len(close), 1, 5), dtype=np.float32)
    prices[:, 0, 0] = close
    prices[:, 0, 1] = close
    prices[:, 0, 2] = close
    prices[:, 0, 3] = close
    tradable = np.ones((len(close), 1), dtype=np.uint8)
    return MktdData(
        version=2,
        symbols=["AAA"],
        features=features,
        prices=prices,
        tradable=tradable,
    )


def test_hourly_replay_matches_daily_when_hourly_close_is_piecewise_constant():
    # 1 symbol, 3 days, 2-step episode (t=0->1, 1->2)
    symbols = ["AAA"]
    T = 3
    S = 1
    F = 16
    P = 5

    features = np.zeros((T, S, F), dtype=np.float32)
    prices = np.zeros((T, S, P), dtype=np.float32)
    prices[:, 0, 3] = np.array([100.0, 110.0, 121.0], dtype=np.float32)  # close
    prices[:, 0, 0] = prices[:, 0, 3]  # open
    prices[:, 0, 1] = prices[:, 0, 3]  # high
    prices[:, 0, 2] = prices[:, 0, 3]  # low
    tradable = np.ones((T, S), dtype=np.uint8)

    data = MktdData(version=2, symbols=symbols, features=features, prices=prices, tradable=tradable)

    # Policy always chooses "go long symbol 0" (action=1).
    daily = simulate_daily_policy(
        data,
        lambda obs: 1,
        max_steps=2,
        fee_rate=0.0,
        max_leverage=1.0,
        periods_per_year=365.0,
    )
    assert daily.total_return == pytest.approx(0.21, abs=1e-9)
    assert daily.num_trades == 1

    start_date = "2024-01-01"
    end_date = "2024-01-03"
    idx = pd.date_range(f"{start_date} 00:00", f"{end_date} 23:00", freq="h", tz="UTC")
    close = np.zeros((len(idx),), dtype=np.float64)

    # Piecewise-constant daily closes for each UTC day.
    close[idx.floor("D") == pd.Timestamp("2024-01-01", tz="UTC")] = 100.0
    close[idx.floor("D") == pd.Timestamp("2024-01-02", tz="UTC")] = 110.0
    close[idx.floor("D") == pd.Timestamp("2024-01-03", tz="UTC")] = 121.0

    market = HourlyMarket(index=idx, close={"AAA": close}, tradable={"AAA": np.ones_like(close, dtype=bool)})

    hourly = replay_hourly_frozen_daily_actions(
        data=data,
        actions=daily.actions,
        market=market,
        start_date=start_date,
        end_date=end_date,
        max_steps=2,
        fee_rate=0.0,
        max_leverage=1.0,
        periods_per_year=8760.0,
    )

    assert hourly.total_return == pytest.approx(0.21, abs=1e-9)
    assert hourly.num_trades == 1
    assert hourly.num_orders == 2  # open + terminal close
    assert hourly.max_drawdown == pytest.approx(0.0, abs=1e-12)
    assert hourly.equity_curve.shape == (len(idx),)


def test_simulate_daily_policy_allocation_bins_scale_position_size():
    symbols = ["AAA"]
    T = 3
    S = 1
    F = 16
    P = 5

    features = np.zeros((T, S, F), dtype=np.float32)
    prices = np.zeros((T, S, P), dtype=np.float32)
    prices[:, 0, 3] = np.array([100.0, 110.0, 121.0], dtype=np.float32)  # close
    prices[:, 0, 0] = prices[:, 0, 3]
    prices[:, 0, 1] = prices[:, 0, 3]
    prices[:, 0, 2] = prices[:, 0, 3]
    tradable = np.ones((T, S), dtype=np.uint8)
    data = MktdData(version=2, symbols=symbols, features=features, prices=prices, tradable=tradable)

    # alloc_bins=2 => action=1 is 50% long, action=2 is 100% long.
    res_half = simulate_daily_policy(
        data,
        lambda obs: 1,
        max_steps=2,
        fee_rate=0.0,
        max_leverage=1.0,
        periods_per_year=365.0,
        action_allocation_bins=2,
        action_level_bins=1,
    )
    res_full = simulate_daily_policy(
        data,
        lambda obs: 2,
        max_steps=2,
        fee_rate=0.0,
        max_leverage=1.0,
        periods_per_year=365.0,
        action_allocation_bins=2,
        action_level_bins=1,
    )

    assert res_half.total_return == pytest.approx(0.105, abs=1e-9)
    assert res_full.total_return == pytest.approx(0.21, abs=1e-9)
    assert res_full.total_return == pytest.approx(res_half.total_return * 2.0, abs=1e-9)


def test_hourly_replay_supports_allocation_bins_for_frozen_daily_actions() -> None:
    data = _single_symbol_daily_data(np.asarray([100.0, 110.0, 121.0], dtype=np.float32))
    market_index = pd.date_range("2024-01-01T00:00:00Z", "2024-01-03T23:00:00Z", freq="h", tz="UTC")
    market_close = np.zeros((len(market_index),), dtype=np.float64)
    market_close[market_index.floor("D") == pd.Timestamp("2024-01-01", tz="UTC")] = 100.0
    market_close[market_index.floor("D") == pd.Timestamp("2024-01-02", tz="UTC")] = 110.0
    market_close[market_index.floor("D") == pd.Timestamp("2024-01-03", tz="UTC")] = 121.0
    market = HourlyMarket(
        index=market_index,
        close={"AAA": market_close},
        tradable={"AAA": np.ones((len(market_index),), dtype=bool)},
    )

    hourly_half = replay_hourly_frozen_daily_actions(
        data=data,
        actions=np.asarray([1, 1], dtype=np.int32),
        market=market,
        start_date="2024-01-01",
        end_date="2024-01-03",
        max_steps=2,
        fee_rate=0.0,
        max_leverage=1.0,
        periods_per_year=8760.0,
        action_allocation_bins=2,
    )
    hourly_full = replay_hourly_frozen_daily_actions(
        data=data,
        actions=np.asarray([2, 2], dtype=np.int32),
        market=market,
        start_date="2024-01-01",
        end_date="2024-01-03",
        max_steps=2,
        fee_rate=0.0,
        max_leverage=1.0,
        periods_per_year=8760.0,
        action_allocation_bins=2,
    )

    assert hourly_half.total_return == pytest.approx(0.105, abs=1e-9)
    assert hourly_full.total_return == pytest.approx(0.21, abs=1e-9)


def test_simulate_hourly_policy_supports_allocation_bins() -> None:
    data = _single_symbol_daily_data(np.asarray([100.0, 110.0, 121.0], dtype=np.float32))
    market_index = pd.date_range("2024-01-01T00:00:00Z", "2024-01-03T23:00:00Z", freq="h", tz="UTC")
    market_close = np.zeros((len(market_index),), dtype=np.float64)
    market_close[market_index.floor("D") == pd.Timestamp("2024-01-01", tz="UTC")] = 100.0
    market_close[market_index.floor("D") == pd.Timestamp("2024-01-02", tz="UTC")] = 110.0
    market_close[market_index.floor("D") == pd.Timestamp("2024-01-03", tz="UTC")] = 121.0
    market = HourlyMarket(
        index=market_index,
        close={"AAA": market_close},
        tradable={"AAA": np.ones((len(market_index),), dtype=bool)},
    )

    hourly_half = simulate_hourly_policy(
        data=data,
        policy_fn=lambda obs: 1,
        market=market,
        start_date="2024-01-01",
        end_date="2024-01-03",
        max_steps_days=2,
        fee_rate=0.0,
        max_leverage=1.0,
        periods_per_year=8760.0,
        action_allocation_bins=2,
    )
    hourly_full = simulate_hourly_policy(
        data=data,
        policy_fn=lambda obs: 2,
        market=market,
        start_date="2024-01-01",
        end_date="2024-01-03",
        max_steps_days=2,
        fee_rate=0.0,
        max_leverage=1.0,
        periods_per_year=8760.0,
        action_allocation_bins=2,
    )

    assert hourly_half.total_return == pytest.approx(0.105, abs=1e-9)
    assert hourly_full.total_return == pytest.approx(0.21, abs=1e-9)


def test_simulate_daily_policy_intrabar_matches_simple_buy_and_hold() -> None:
    data = _single_symbol_daily_data(np.asarray([100.0, 110.0, 121.0], dtype=np.float32))
    market_index = pd.date_range("2024-01-01T00:00:00Z", "2024-01-03T23:00:00Z", freq="h", tz="UTC")
    market_close = np.zeros((len(market_index),), dtype=np.float64)
    market_close[market_index.floor("D") == pd.Timestamp("2024-01-01", tz="UTC")] = 100.0
    market_close[market_index.floor("D") == pd.Timestamp("2024-01-02", tz="UTC")] = 110.0
    market_close[market_index.floor("D") == pd.Timestamp("2024-01-03", tz="UTC")] = 121.0
    hourly = HourlyOHLC(
        index=market_index,
        symbols=["AAA"],
        open={"AAA": market_close.copy()},
        high={"AAA": market_close.copy()},
        low={"AAA": market_close.copy()},
        close={"AAA": market_close.copy()},
        tradable={"AAA": np.ones((len(market_index),), dtype=bool)},
    )

    calls = {"n": 0}

    def _policy_fn(obs: np.ndarray) -> int:
        calls["n"] += 1
        return 1

    result = simulate_daily_policy_intrabar(
        data=data,
        policy_fn=_policy_fn,
        hourly=hourly,
        start_date="2024-01-01",
        max_steps=2,
        fee_rate=0.0,
        fill_buffer_bps=0.0,
        max_leverage=1.0,
        trade_hour_mode="first_tradable",
    )

    assert result.actions.tolist() == [1, 1]
    assert result.total_return == pytest.approx(0.21, abs=1e-9)
    assert result.num_trades == 1
    assert calls["n"] == 2


def test_simulate_daily_policy_level_bins_require_fill():
    symbols = ["AAA"]
    T = 3
    S = 1
    F = 16
    P = 5

    features = np.zeros((T, S, F), dtype=np.float32)
    prices = np.zeros((T, S, P), dtype=np.float32)
    prices[:, 0, 3] = np.array([100.0, 100.0, 100.0], dtype=np.float32)  # close
    prices[:, 0, 0] = prices[:, 0, 3]
    prices[:, 0, 1] = prices[:, 0, 3]  # high
    prices[:, 0, 2] = prices[:, 0, 3]  # low
    tradable = np.ones((T, S), dtype=np.uint8)
    data = MktdData(version=2, symbols=symbols, features=features, prices=prices, tradable=tradable)

    # alloc_bins=1, level_bins=3 => action=1 is long with -max_offset_bps, which is below the bar range.
    res_no_fill = simulate_daily_policy(
        data,
        lambda obs: 1,
        max_steps=2,
        fee_rate=0.0,
        max_leverage=1.0,
        periods_per_year=365.0,
        action_allocation_bins=1,
        action_level_bins=3,
        action_max_offset_bps=100.0,  # +1%
    )

    assert res_no_fill.total_return == pytest.approx(0.0, abs=1e-12)
    assert res_no_fill.num_trades == 0


def test_simulate_daily_policy_fill_buffer_requires_trade_through_limit_long() -> None:
    symbols = ["AAA"]
    close = np.asarray([100.0, 110.0, 121.0], dtype=np.float32)
    low = np.asarray([99.98, 109.978, 120.978], dtype=np.float32)
    high = close.copy()

    features = np.zeros((3, 1, 16), dtype=np.float32)
    prices = np.zeros((3, 1, 5), dtype=np.float32)
    prices[:, 0, 0] = close
    prices[:, 0, 1] = high
    prices[:, 0, 2] = low
    prices[:, 0, 3] = close
    tradable = np.ones((3, 1), dtype=np.uint8)
    data = MktdData(version=2, symbols=symbols, features=features, prices=prices, tradable=tradable)

    res_touch = simulate_daily_policy(
        data,
        lambda obs: 1,
        max_steps=2,
        fee_rate=0.0,
        fill_buffer_bps=0.0,
        max_leverage=1.0,
        periods_per_year=365.0,
    )
    res_buffered = simulate_daily_policy(
        data,
        lambda obs: 1,
        max_steps=2,
        fee_rate=0.0,
        fill_buffer_bps=5.0,
        max_leverage=1.0,
        periods_per_year=365.0,
    )

    assert res_touch.total_return == pytest.approx(0.21, abs=1e-9)
    assert res_touch.num_trades == 1
    assert res_buffered.total_return == pytest.approx(0.0, abs=1e-12)
    assert res_buffered.num_trades == 0


def test_simulate_daily_policy_fill_buffer_requires_trade_through_limit_short() -> None:
    symbols = ["AAA"]
    close = np.asarray([100.0, 90.0, 81.0], dtype=np.float32)
    high = np.asarray([100.02, 90.018, 81.0162], dtype=np.float32)
    low = close.copy()

    features = np.zeros((3, 1, 16), dtype=np.float32)
    prices = np.zeros((3, 1, 5), dtype=np.float32)
    prices[:, 0, 0] = close
    prices[:, 0, 1] = high
    prices[:, 0, 2] = low
    prices[:, 0, 3] = close
    tradable = np.ones((3, 1), dtype=np.uint8)
    data = MktdData(version=2, symbols=symbols, features=features, prices=prices, tradable=tradable)

    res_touch = simulate_daily_policy(
        data,
        lambda obs: 2,
        max_steps=2,
        fee_rate=0.0,
        fill_buffer_bps=0.0,
        max_leverage=1.0,
        periods_per_year=365.0,
    )
    res_buffered = simulate_daily_policy(
        data,
        lambda obs: 2,
        max_steps=2,
        fee_rate=0.0,
        fill_buffer_bps=5.0,
        max_leverage=1.0,
        periods_per_year=365.0,
    )

    assert res_touch.total_return == pytest.approx(0.19, abs=1e-9)
    assert res_touch.num_trades == 1
    assert res_buffered.total_return == pytest.approx(0.0, abs=1e-12)
    assert res_buffered.num_trades == 0


def test_simulate_daily_policy_short_borrow_fee_hits_flat_short() -> None:
    symbols = ["AAA"]
    T = 3
    S = 1
    F = 16
    P = 5

    features = np.zeros((T, S, F), dtype=np.float32)
    prices = np.zeros((T, S, P), dtype=np.float32)
    prices[:, 0, 3] = np.array([100.0, 100.0, 100.0], dtype=np.float32)
    prices[:, 0, 0] = prices[:, 0, 3]
    prices[:, 0, 1] = prices[:, 0, 3]
    prices[:, 0, 2] = prices[:, 0, 3]
    tradable = np.ones((T, S), dtype=np.uint8)
    data = MktdData(version=2, symbols=symbols, features=features, prices=prices, tradable=tradable)

    apr = 0.10
    periods_per_year = 252.0
    expected_fee_per_step = 10_000.0 * apr / periods_per_year
    expected_total_return = -(2.0 * expected_fee_per_step) / 10_000.0

    res_short = simulate_daily_policy(
        data,
        lambda obs: 2,
        max_steps=2,
        fee_rate=0.0,
        max_leverage=1.0,
        periods_per_year=periods_per_year,
        short_borrow_apr=apr,
    )
    res_long = simulate_daily_policy(
        data,
        lambda obs: 1,
        max_steps=2,
        fee_rate=0.0,
        max_leverage=1.0,
        periods_per_year=periods_per_year,
        short_borrow_apr=apr,
    )

    assert res_long.total_return == pytest.approx(0.0, abs=1e-12)
    assert res_short.total_return == pytest.approx(expected_total_return, rel=1e-6, abs=1e-9)


def test_simulate_daily_policy_prints_early_exit_when_drawdown_exceeds_profit(
    capsys: pytest.CaptureFixture[str],
) -> None:
    close = np.asarray(
        [100.0, 104.0, 108.0, 112.0, 116.0, 120.0, 124.0, 128.0, 132.0, 120.0, 108.0, 96.0, 84.0, 72.0, 60.0, 58.0, 56.0, 54.0, 52.0, 50.0, 48.0],
        dtype=np.float32,
    )
    data = _single_symbol_daily_data(close)

    result = simulate_daily_policy(
        data,
        lambda obs: 1,
        max_steps=len(close) - 1,
        fee_rate=0.0,
        max_leverage=1.0,
        periods_per_year=365.0,
    )

    assert result.max_drawdown > 0.0
    assert "early stopping" in capsys.readouterr().out.lower()


def test_simulate_daily_policy_can_early_exit_on_max_drawdown_threshold(
    capsys: pytest.CaptureFixture[str],
) -> None:
    data = _single_symbol_daily_data(
        np.asarray(
            [100.0, 108.0, 116.0, 124.0, 132.0, 140.0, 138.0, 126.0, 114.0, 102.0, 90.0, 88.0, 86.0, 84.0, 82.0, 80.0, 78.0, 76.0, 74.0, 72.0, 70.0],
            dtype=np.float32,
        )
    )

    result = simulate_daily_policy(
        data,
        lambda obs: 1,
        max_steps=20,
        fee_rate=0.0,
        max_leverage=1.0,
        periods_per_year=365.0,
        enable_drawdown_profit_early_exit=False,
        early_exit_max_drawdown=0.20,
        drawdown_profit_early_exit_min_steps=6,
        drawdown_profit_early_exit_progress_fraction=0.5,
    )

    assert result.stopped_early is True
    assert result.evaluated_steps < 20
    assert "max drawdown" in result.stop_reason.lower()
    assert "max drawdown" in capsys.readouterr().out.lower()


def test_simulate_daily_policy_can_early_exit_on_running_sortino_threshold(
    capsys: pytest.CaptureFixture[str],
) -> None:
    data = _single_symbol_daily_data(
        np.asarray(
            [100.0, 102.0, 99.0, 101.0, 98.0, 100.0, 97.0, 99.0, 96.0, 98.0, 95.0, 97.0, 94.0, 96.0, 93.0, 95.0, 92.0, 94.0, 91.0, 93.0, 90.0],
            dtype=np.float32,
        )
    )

    result = simulate_daily_policy(
        data,
        lambda obs: 1,
        max_steps=20,
        fee_rate=0.0,
        max_leverage=1.0,
        periods_per_year=365.0,
        enable_drawdown_profit_early_exit=False,
        early_exit_min_sortino=0.0,
        drawdown_profit_early_exit_min_steps=6,
        drawdown_profit_early_exit_progress_fraction=0.5,
    )

    assert result.stopped_early is True
    assert result.evaluated_steps < 20
    assert "sortino" in result.stop_reason.lower()
    assert "sortino" in capsys.readouterr().out.lower()


def test_read_mktd_wide_universe(tmp_path):
    """read_mktd must accept num_symbols up to 1024 (regression for >64 limit)."""
    import struct
    from pathlib import Path
    from pufferlib_market.hourly_replay import read_mktd

    # Build a minimal valid MKTD binary with 73 symbols (wide73 use case).
    nts, nsym, nfeat, nprice = 10, 73, 16, 5
    hdr = struct.pack("<4sIIIII40s", b"MKTD", 2, nsym, nts, nfeat, nprice, b"\x00" * 40)
    sym_table = b"".join(
        (f"SYM{i}".encode() + b"\x00" * (16 - len(f"SYM{i}"))).ljust(16, b"\x00")
        for i in range(nsym)
    )
    feat = np.zeros((nts, nsym, nfeat), dtype=np.float32)
    price = np.ones((nts, nsym, nprice), dtype=np.float32)
    mask = np.ones((nts, nsym), dtype=np.uint8)

    p = tmp_path / "wide73.bin"
    with open(p, "wb") as f:
        f.write(hdr)
        f.write(sym_table)
        f.write(feat.tobytes())
        f.write(price.tobytes())
        f.write(mask.tobytes())

    data = read_mktd(p)
    assert data.num_symbols == 73
    assert data.num_timesteps == 10
    assert data.features.shape == (10, 73, 16)
