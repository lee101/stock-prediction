from __future__ import annotations

import builtins
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "benchmark_bitbankgo_stock_bridge.py"


spec = importlib.util.spec_from_file_location("benchmark_bitbankgo_stock_bridge", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def _make_hourly(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    ts = pd.date_range("2025-01-06 14:30:00", periods=n, freq="1h", tz="UTC")
    ret = rng.normal(0.0002, 0.004, size=n)
    close = 100.0 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * 1.001
    low = np.minimum(open_, close) * 0.999
    volume = rng.uniform(750_000.0, 1_250_000.0, size=n)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "symbol": "AAA",
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def test_bitbankgo_features_are_lagged_against_current_close() -> None:
    base = _make_hourly()
    pert = base.copy()
    pert.loc[80, "close"] *= 1.5
    pert.loc[80, "high"] = max(pert.loc[80, "high"], pert.loc[80, "close"])

    feat_a = mod.build_bitbankgo_hourly_features(base)
    feat_b = mod.build_bitbankgo_hourly_features(pert)
    row_a = feat_a.loc[80, mod.FEATURE_COLS].to_numpy(dtype=float)
    row_b = feat_b.loc[80, mod.FEATURE_COLS].to_numpy(dtype=float)
    mask = np.isfinite(row_a) & np.isfinite(row_b)

    assert mask.any()
    assert np.allclose(row_a[mask], row_b[mask])


def test_candidate_trades_respect_decision_lag_and_horizon() -> None:
    frame = mod.build_bitbankgo_hourly_features(_make_hourly())
    frame = frame.dropna(subset=mod.FEATURE_COLS + ["target_2h"]).reset_index(drop=True)
    preds = {2: np.full(len(frame), 0.01)}
    cfg = mod.SymbolConfig(
        symbol="AAA",
        horizon=2,
        threshold=0.001,
        mode="long_short",
        val_score=1.0,
        val_monthly_return=0.1,
        val_max_drawdown=0.0,
        val_sortino=1.0,
        val_trades=10,
        val_win_rate=0.8,
        val_long_trades=6,
        val_short_trades=4,
    )

    trades = mod.build_candidate_trades(frame, preds, [cfg], decision_lag=2)

    assert trades
    first = trades[0]
    assert first.side == 1
    assert first.entry_ts == frame.loc[2, "timestamp"]
    assert first.exit_ts == frame.loc[4, "timestamp"]
    assert first.edge > 0


def test_candidate_trades_include_short_side_when_enabled() -> None:
    frame = mod.build_bitbankgo_hourly_features(_make_hourly())
    frame = frame.dropna(subset=mod.FEATURE_COLS + ["target_2h"]).reset_index(drop=True)
    preds = {2: np.full(len(frame), -0.01)}
    cfg = mod.SymbolConfig(
        symbol="AAA",
        horizon=2,
        threshold=0.001,
        mode="long_short",
        val_score=1.0,
        val_monthly_return=0.1,
        val_max_drawdown=0.0,
        val_sortino=1.0,
        val_trades=10,
        val_win_rate=0.8,
        val_long_trades=5,
        val_short_trades=5,
    )

    trades = mod.build_candidate_trades(frame, preds, [cfg], decision_lag=2)

    assert trades
    assert trades[0].side == -1
    assert trades[0].edge > 0


def test_cross_sectional_candidates_demean_scores_into_longs_and_shorts() -> None:
    aaa = _make_hourly()
    bbb = _make_hourly()
    bbb["symbol"] = "BBB"
    bbb["close"] = bbb["close"] * 1.03
    bbb["open"] = bbb["open"] * 1.03
    bbb["high"] = bbb["high"] * 1.03
    bbb["low"] = bbb["low"] * 1.03
    frame = pd.concat(
        [
            mod.build_bitbankgo_hourly_features(aaa),
            mod.build_bitbankgo_hourly_features(bbb),
        ],
        ignore_index=True,
    )
    frame = frame.dropna(subset=mod.FEATURE_COLS + ["target_2h"]).reset_index(drop=True)
    preds = {2: np.where(frame["symbol"].to_numpy() == "AAA", 0.02, 0.01)}

    trades = mod.build_cross_sectional_candidate_trades(
        frame,
        preds,
        horizon=2,
        threshold=0.0,
        decision_lag=2,
        allow_shorts=True,
    )

    assert {trade.side for trade in trades} == {-1, 1}
    assert any(trade.symbol == "AAA" and trade.side == 1 for trade in trades)
    assert any(trade.symbol == "BBB" and trade.side == -1 for trade in trades)


def test_opportunistic_candidates_wait_for_limit_penetration() -> None:
    aaa = _make_hourly()
    bbb = _make_hourly()
    bbb["symbol"] = "BBB"
    frame = pd.concat(
        [
            mod.build_bitbankgo_hourly_features(aaa),
            mod.build_bitbankgo_hourly_features(bbb),
        ],
        ignore_index=True,
    )
    frame = frame.dropna(subset=mod.FEATURE_COLS + ["target_2h"]).reset_index(drop=True)
    aaa_mask = frame["symbol"] == "AAA"
    frame.loc[aaa_mask, ["actual_open", "actual_high", "actual_low", "actual_close"]] = [100.0, 100.2, 99.9, 100.0]
    first_aaa = frame.index[frame["symbol"] == "AAA"][0]
    signal_ts = frame.loc[first_aaa, "timestamp"]
    lagged_ts = frame.loc[first_aaa + 2, "timestamp"]
    fill_ts = frame.loc[first_aaa + 3, "timestamp"]
    frame.loc[first_aaa + 2, "actual_low"] = frame.loc[first_aaa + 2, "actual_open"] * 0.999
    frame.loc[first_aaa + 3, "actual_low"] = frame.loc[first_aaa + 2, "actual_open"] * 0.994
    preds = {2: np.where(frame["symbol"].to_numpy() == "AAA", 0.03, 0.01)}

    trades = mod.build_cross_sectional_candidate_trades(
        frame,
        preds,
        horizon=2,
        threshold=0.0,
        decision_lag=2,
        allow_shorts=False,
        opportunistic_watch_n=1,
        opportunistic_entry_discount_bps=50.0,
        opportunistic_watch_bars=3,
        fill_buffer_bps=5.0,
    )

    aaa_trade = next(t for t in trades if t.symbol == "AAA")
    assert aaa_trade.signal_ts == signal_ts
    assert aaa_trade.entry_ts != lagged_ts
    assert aaa_trade.entry_ts == fill_ts
    assert aaa_trade.entry_open == pytest.approx(frame.loc[first_aaa + 2, "actual_open"] * 0.995)


def test_opportunistic_candidates_do_not_fill_without_price_touch() -> None:
    aaa = _make_hourly()
    bbb = _make_hourly()
    bbb["symbol"] = "BBB"
    frame = pd.concat(
        [
            mod.build_bitbankgo_hourly_features(aaa),
            mod.build_bitbankgo_hourly_features(bbb),
        ],
        ignore_index=True,
    )
    frame = frame.dropna(subset=mod.FEATURE_COLS + ["target_2h"]).reset_index(drop=True)
    aaa_mask = frame["symbol"] == "AAA"
    frame.loc[aaa_mask, ["actual_open", "actual_high", "actual_low", "actual_close"]] = [100.0, 100.2, 99.9, 100.0]
    preds = {2: np.where(frame["symbol"].to_numpy() == "AAA", 0.03, 0.01)}

    trades = mod.build_cross_sectional_candidate_trades(
        frame,
        preds,
        horizon=2,
        threshold=0.0,
        decision_lag=2,
        allow_shorts=False,
        opportunistic_watch_n=1,
        opportunistic_entry_discount_bps=50.0,
        opportunistic_watch_bars=3,
        fill_buffer_bps=5.0,
    )

    assert all(t.symbol != "AAA" for t in trades)


def test_worst_slippage_reduces_window_return() -> None:
    start = pd.Timestamp("2025-01-01 14:30", tz="UTC")
    trades = [
        mod.CandidateTrade(
            symbol="AAA",
            side=1,
            horizon=1,
            threshold=0.0,
            signal_ts=start,
            entry_ts=start,
            exit_ts=start + pd.Timedelta(hours=1),
            edge=0.01,
            entry_open=100.0,
            exit_close=101.0,
        )
    ]
    windows = [(start, start + pd.Timedelta(hours=2))]

    clean = mod.simulate_candidate_windows(
        trades,
        windows,
        fee_rate=0.0,
        fill_buffer_bps=0.0,
        slippage_bps=0.0,
        leverage=1.0,
        window_bars=2,
    )[0]
    stressed = mod.simulate_candidate_windows(
        trades,
        windows,
        fee_rate=0.001,
        fill_buffer_bps=5.0,
        slippage_bps=20.0,
        leverage=1.0,
        window_bars=2,
    )[0]

    assert clean.total_return > stressed.total_return


def test_short_round_trip_profits_when_price_falls() -> None:
    ret = mod._round_trip_return(
        100.0,
        95.0,
        side=-1,
        fee_rate=0.0,
        fill_buffer_bps=0.0,
        slippage_bps=0.0,
        leverage=1.0,
        short_exposure_scale=0.85,
    )

    assert ret > 0
    assert ret < mod._round_trip_return(
        95.0,
        100.0,
        side=1,
        fee_rate=0.0,
        fill_buffer_bps=0.0,
        slippage_bps=0.0,
        leverage=1.0,
        short_exposure_scale=0.85,
    )


def test_short_round_trip_uses_entry_notional_and_both_fees() -> None:
    ret = mod._round_trip_return(
        100.0,
        90.0,
        side=-1,
        fee_rate=0.001,
        fill_buffer_bps=0.0,
        slippage_bps=0.0,
        leverage=1.0,
        short_exposure_scale=1.0,
    )

    assert ret == pytest.approx(0.0981)

    loss = mod._round_trip_return(
        100.0,
        110.0,
        side=-1,
        fee_rate=0.001,
        fill_buffer_bps=0.0,
        slippage_bps=0.0,
        leverage=1.0,
        short_exposure_scale=1.0,
    )

    assert loss == pytest.approx(-0.1021)


def test_candidate_trade_payload_is_json_ready() -> None:
    start = pd.Timestamp("2025-01-01 14:30", tz="UTC")
    trade = mod.CandidateTrade(
        symbol="AAA",
        side=-1,
        horizon=2,
        threshold=0.001,
        signal_ts=start,
        entry_ts=start + pd.Timedelta(hours=2),
        exit_ts=start + pd.Timedelta(hours=4),
        edge=0.01,
        entry_open=100.0,
        exit_close=98.0,
    )

    payload = mod.candidate_trade_to_dict(trade)

    assert payload["symbol"] == "AAA"
    assert payload["side"] == -1
    assert payload["entry_ts"] == "2025-01-01T16:30:00+00:00"


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["--symbols", ""], "symbols must not be empty"),
        (["--horizons", "0,2"], "horizons must contain only positive integers"),
        (["--horizons", "1,nope"], "horizons must be a comma-separated integer list"),
        (["--thresholds", "nan"], "thresholds must contain only finite values"),
        (["--slippage-bps-grid", "0,-5"], "slippage_bps_grid must contain only non-negative values"),
        (["--selection-slippage-bps", "inf"], "selection_slippage_bps must be finite and non-negative"),
        (["--decision-lag", "1"], "decision_lag must be >= 2"),
        (["--fee-rate", "nan"], "fee_rate must be finite and non-negative"),
        (["--fill-buffer-bps", "-1"], "fill_buffer_bps must be finite and non-negative"),
        (["--leverage", "0"], "leverage must be finite and positive"),
        (["--short-exposure-scale", "1.2"], "short_exposure_scale must be <= 1"),
        (["--target-short-fraction", "0.8"], "target_short_fraction must be between 0 and 0.5"),
        (["--total-short-fraction-cap", "1.2"], "total_short_fraction_cap must be between 0 and 1"),
        (["--window-bars", "0"], "window_bars must be positive"),
        (["--stride-bars", "0"], "stride_bars must be positive"),
        (["--max-positions", "0"], "max_positions must be positive"),
        (["--opportunistic-watch-n", "-1"], "opportunistic_watch_n must be finite and non-negative"),
        (["--opportunistic-entry-discount-bps", "-1"], "opportunistic_entry_discount_bps must be finite and non-negative"),
        (["--opportunistic-watch-bars", "0"], "opportunistic_watch_bars must be positive"),
        (["--train-end", "2026-01-01", "--val-end", "2025-01-01"], "date splits must satisfy"),
    ],
)
def test_invalid_configs_fail_before_xgboost_or_data(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    argv: list[str],
    expected: str,
) -> None:
    original_import = builtins.__import__

    def fail_xgboost_import(name, *args, **kwargs):
        if name == "xgboost":
            raise AssertionError("xgboost should not be imported for invalid configs")
        return original_import(name, *args, **kwargs)

    def fail_iter_symbol_files(*args, **kwargs):
        raise AssertionError("data discovery should not run for invalid configs")

    monkeypatch.setattr(builtins, "__import__", fail_xgboost_import)
    monkeypatch.setattr(mod, "_iter_symbol_files", fail_iter_symbol_files)

    rc = mod.main(["--data-root", str(tmp_path / "missing-data"), "--output", str(tmp_path / "out.json"), *argv])

    assert rc == 2
    assert expected in capsys.readouterr().err
    assert not (tmp_path / "out.json").exists()
