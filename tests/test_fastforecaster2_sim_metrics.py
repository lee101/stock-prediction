import numpy as np
import pandas as pd

from fastforecaster2.config import FastForecaster2Config
from fastforecaster2.trainer import FastForecaster2Trainer
from src.robust_trading_metrics import compute_trade_rate


def test_compute_equity_risk_metrics_empty_series():
    metrics = FastForecaster2Trainer._compute_equity_risk_metrics(pd.Series(dtype=float), initial_cash=10_000.0)
    assert metrics["sim_sortino"] == 0.0
    assert metrics["sim_mean_hourly_return"] == 0.0
    assert metrics["sim_max_drawdown"] == 0.0
    assert metrics["sim_annualized_volatility"] == 0.0
    assert metrics["sim_pnl_smoothness"] == 0.0
    assert metrics["sim_smoothness"] == 0.0
    assert metrics["sim_win_rate"] == 0.0


def test_compute_equity_risk_metrics_drawdown_and_win_rate():
    index = pd.date_range("2026-01-01", periods=6, freq="h", tz="UTC")
    equity = pd.Series([100.0, 110.0, 105.0, 120.0, 108.0, 118.0], index=index)
    metrics = FastForecaster2Trainer._compute_equity_risk_metrics(equity, initial_cash=10_000.0)

    assert 0.0 <= metrics["sim_max_drawdown"] <= 1.0
    assert metrics["sim_max_drawdown"] > 0.0
    assert 0.0 <= metrics["sim_win_rate"] <= 1.0
    assert metrics["sim_annualized_volatility"] >= 0.0
    assert metrics["sim_pnl_smoothness"] >= 0.0
    assert 0.0 <= metrics["sim_smoothness"] <= 1.0


def test_compute_equity_risk_metrics_flat_curve_has_zero_vol():
    index = pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC")
    equity = pd.Series(np.full(4, 100.0), index=index)
    metrics = FastForecaster2Trainer._compute_equity_risk_metrics(equity, initial_cash=10_000.0)

    assert metrics["sim_max_drawdown"] == 0.0
    assert metrics["sim_annualized_volatility"] == 0.0
    assert metrics["sim_pnl_smoothness"] == 0.0
    assert metrics["sim_smoothness"] == 1.0
    assert metrics["sim_win_rate"] == 0.0


def test_plan_market_sim_actions_transitions_without_pyramiding():
    timestamps = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")
    frame = pd.DataFrame(
        [
            {
                "timestamp": timestamps[0],
                "symbol": "AAA",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "predicted_return": 0.030,
                "realized_vol": 0.010,
                "smoothed_return": 0.030,
                "smoothed_score": 3.0,
            },
            {
                "timestamp": timestamps[0],
                "symbol": "BBB",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "predicted_return": 0.015,
                "realized_vol": 0.010,
                "smoothed_return": 0.015,
                "smoothed_score": 1.5,
            },
            {
                "timestamp": timestamps[1],
                "symbol": "AAA",
                "open": 101.0,
                "high": 102.0,
                "low": 100.0,
                "close": 101.0,
                "predicted_return": 0.020,
                "realized_vol": 0.010,
                "smoothed_return": 0.020,
                "smoothed_score": 2.0,
            },
            {
                "timestamp": timestamps[1],
                "symbol": "BBB",
                "open": 100.5,
                "high": 101.0,
                "low": 100.0,
                "close": 100.5,
                "predicted_return": 0.010,
                "realized_vol": 0.010,
                "smoothed_return": 0.010,
                "smoothed_score": 1.0,
            },
            {
                "timestamp": timestamps[2],
                "symbol": "AAA",
                "open": 99.0,
                "high": 99.5,
                "low": 98.0,
                "close": 99.0,
                "predicted_return": -0.005,
                "realized_vol": 0.010,
                "smoothed_return": -0.005,
                "smoothed_score": -0.5,
            },
            {
                "timestamp": timestamps[2],
                "symbol": "BBB",
                "open": 102.0,
                "high": 103.0,
                "low": 101.0,
                "close": 102.0,
                "predicted_return": 0.025,
                "realized_vol": 0.010,
                "smoothed_return": 0.025,
                "smoothed_score": 2.5,
            },
        ]
    )

    planned = FastForecaster2Trainer._plan_market_sim_actions_from_scores(
        frame,
        buy_threshold=0.01,
        sell_threshold=0.005,
        entry_score_threshold=0.0,
        top_k=1,
        max_trade_intensity=20.0,
        min_trade_intensity=2.0,
        vol_target=0.025,
        switch_score_gap=0.0,
        entry_buffer_bps=1.0,
        exit_buffer_bps=1.0,
    )

    aaa_t0 = planned[(planned["timestamp"] == timestamps[0]) & (planned["symbol"] == "AAA")].iloc[0]
    aaa_t1 = planned[(planned["timestamp"] == timestamps[1]) & (planned["symbol"] == "AAA")].iloc[0]
    aaa_t2 = planned[(planned["timestamp"] == timestamps[2]) & (planned["symbol"] == "AAA")].iloc[0]
    bbb_t2 = planned[(planned["timestamp"] == timestamps[2]) & (planned["symbol"] == "BBB")].iloc[0]

    assert aaa_t0["buy_amount"] > 0.0
    assert aaa_t1["buy_amount"] == 0.0
    assert aaa_t1["sell_amount"] == 0.0
    assert aaa_t2["sell_amount"] == 100.0
    assert bbb_t2["buy_amount"] > 0.0


def test_plan_market_sim_actions_switches_only_when_gap_is_large_enough():
    timestamps = pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC")
    frame = pd.DataFrame(
        [
            {
                "timestamp": timestamps[0],
                "symbol": "AAA",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "predicted_return": 0.025,
                "realized_vol": 0.010,
                "smoothed_return": 0.025,
                "smoothed_score": 0.0025,
            },
            {
                "timestamp": timestamps[0],
                "symbol": "BBB",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "predicted_return": 0.018,
                "realized_vol": 0.010,
                "smoothed_return": 0.018,
                "smoothed_score": 0.0018,
            },
            {
                "timestamp": timestamps[1],
                "symbol": "AAA",
                "open": 101.0,
                "high": 102.0,
                "low": 100.0,
                "close": 101.0,
                "predicted_return": 0.012,
                "realized_vol": 0.010,
                "smoothed_return": 0.012,
                "smoothed_score": 0.0012,
            },
            {
                "timestamp": timestamps[1],
                "symbol": "BBB",
                "open": 102.0,
                "high": 103.0,
                "low": 101.0,
                "close": 102.0,
                "predicted_return": 0.021,
                "realized_vol": 0.010,
                "smoothed_return": 0.021,
                "smoothed_score": 0.00125,
            },
        ]
    )

    sticky = FastForecaster2Trainer._plan_market_sim_actions_from_scores(
        frame,
        buy_threshold=0.01,
        sell_threshold=0.005,
        entry_score_threshold=0.0,
        top_k=1,
        max_trade_intensity=20.0,
        min_trade_intensity=2.0,
        vol_target=0.025,
        switch_score_gap=0.0001,
        entry_buffer_bps=1.0,
        exit_buffer_bps=1.0,
    )
    switching = FastForecaster2Trainer._plan_market_sim_actions_from_scores(
        frame,
        buy_threshold=0.01,
        sell_threshold=0.005,
        entry_score_threshold=0.0,
        top_k=1,
        max_trade_intensity=20.0,
        min_trade_intensity=2.0,
        vol_target=0.025,
        switch_score_gap=0.00001,
        entry_buffer_bps=1.0,
        exit_buffer_bps=1.0,
    )

    aaa_t1_sticky = sticky[(sticky["timestamp"] == timestamps[1]) & (sticky["symbol"] == "AAA")].iloc[0]
    bbb_t1_sticky = sticky[(sticky["timestamp"] == timestamps[1]) & (sticky["symbol"] == "BBB")].iloc[0]
    aaa_t1_switch = switching[(switching["timestamp"] == timestamps[1]) & (switching["symbol"] == "AAA")].iloc[0]
    bbb_t1_switch = switching[(switching["timestamp"] == timestamps[1]) & (switching["symbol"] == "BBB")].iloc[0]

    assert aaa_t1_sticky["sell_amount"] == 0.0
    assert bbb_t1_sticky["buy_amount"] == 0.0
    assert aaa_t1_switch["sell_amount"] == 100.0
    assert bbb_t1_switch["buy_amount"] > 0.0


def test_plan_market_sim_actions_requires_entry_score_threshold_for_new_positions():
    timestamps = pd.date_range("2026-01-01", periods=1, freq="h", tz="UTC")
    frame = pd.DataFrame(
        [
            {
                "timestamp": timestamps[0],
                "symbol": "AAA",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "predicted_return": 0.030,
                "realized_vol": 0.010,
                "smoothed_return": 0.030,
                "smoothed_score": 0.0012,
            },
            {
                "timestamp": timestamps[0],
                "symbol": "BBB",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "predicted_return": 0.025,
                "realized_vol": 0.010,
                "smoothed_return": 0.025,
                "smoothed_score": 0.0018,
            },
        ]
    )

    planned = FastForecaster2Trainer._plan_market_sim_actions_from_scores(
        frame,
        buy_threshold=0.01,
        sell_threshold=0.005,
        entry_score_threshold=0.0015,
        top_k=2,
        max_trade_intensity=20.0,
        min_trade_intensity=2.0,
        vol_target=0.025,
        switch_score_gap=0.0,
        entry_buffer_bps=1.0,
        exit_buffer_bps=1.0,
    )

    aaa = planned[planned["symbol"] == "AAA"].iloc[0]
    bbb = planned[planned["symbol"] == "BBB"].iloc[0]

    assert aaa["buy_amount"] == 0.0
    assert bbb["buy_amount"] > 0.0


def test_densify_market_sim_signal_frame_preserves_timestamp_and_forward_fills():
    timestamps = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")
    frame = pd.DataFrame(
        [
            {
                "timestamp": timestamps[0],
                "symbol": "AAA",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "predicted_return": 0.020,
                "realized_vol": 0.010,
                "signal_strength": 0.020,
                "raw_symbol_idx": 0.0,
            },
            {
                "timestamp": timestamps[2],
                "symbol": "AAA",
                "open": 102.0,
                "high": 103.0,
                "low": 101.0,
                "close": 102.0,
                "predicted_return": 0.010,
                "realized_vol": 0.012,
                "signal_strength": 0.010,
                "raw_symbol_idx": 0.0,
            },
            {
                "timestamp": timestamps[0],
                "symbol": "BBB",
                "open": 50.0,
                "high": 51.0,
                "low": 49.0,
                "close": 50.0,
                "predicted_return": 0.015,
                "realized_vol": 0.020,
                "signal_strength": 0.015,
                "raw_symbol_idx": 1.0,
            },
        ]
    )
    bars_frame = pd.DataFrame(
        [
            {
                "timestamp": ts,
                "symbol": symbol,
                "open": 100.0 if symbol == "AAA" else 50.0,
                "high": 101.0 if symbol == "AAA" else 51.0,
                "low": 99.0 if symbol == "AAA" else 49.0,
                "close": 100.0 if symbol == "AAA" else 50.0,
                "realized_vol": 0.010 if symbol == "AAA" else 0.020,
            }
            for symbol in ("AAA", "BBB")
            for ts in timestamps
        ]
    )

    dense = FastForecaster2Trainer._densify_market_sim_signal_frame(frame, bars_frame)

    assert "timestamp" in dense.columns
    assert len(dense[dense["symbol"] == "AAA"]) == 2
    assert len(dense[dense["symbol"] == "BBB"]) == 2
    bbb_last = dense[(dense["symbol"] == "BBB") & (dense["timestamp"] == timestamps[2])].iloc[0]
    assert bbb_last["predicted_return"] == 0.015


def _market_sim_test_config(tmp_path) -> FastForecaster2Config:
    cfg = FastForecaster2Config(
        output_dir=tmp_path / "artifacts",
        min_rows_per_symbol=400,
        lookback=128,
        horizon=16,
        use_market_sim_eval=True,
    )
    cfg.ensure_output_dirs()
    return cfg


def test_run_market_sim_eval_no_actionable_rows_records_period_coverage(tmp_path):
    timestamps = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")
    signal_frame = pd.DataFrame(
        [
            {"timestamp": ts, "symbol": "AAA", "buy_amount": 0.0, "sell_amount": 0.0}
            for ts in timestamps
        ]
    )
    trainer = FastForecaster2Trainer.__new__(FastForecaster2Trainer)
    trainer.config = _market_sim_test_config(tmp_path)
    trainer._build_market_sim_frames = lambda: (pd.DataFrame(), pd.DataFrame(), signal_frame)

    summary = trainer._run_market_sim_eval()

    assert summary["sim_periods"] == 3.0
    assert summary["sim_trade_rate"] == 0.0
    assert summary["sim_goodness_score"] > 0.0
    assert trainer.config.simulator_metrics_file.exists()


def test_run_market_sim_eval_uses_trade_rate_for_goodness_score(tmp_path, monkeypatch):
    timestamps = pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC")
    signal_frame = pd.DataFrame(
        [
            {"timestamp": timestamps[0], "symbol": "AAA", "buy_amount": 25.0, "sell_amount": 0.0},
            {"timestamp": timestamps[1], "symbol": "AAA", "buy_amount": 0.0, "sell_amount": 0.0},
            {"timestamp": timestamps[2], "symbol": "AAA", "buy_amount": 0.0, "sell_amount": 100.0},
            {"timestamp": timestamps[3], "symbol": "AAA", "buy_amount": 0.0, "sell_amount": 0.0},
        ]
    )
    trainer = FastForecaster2Trainer.__new__(FastForecaster2Trainer)
    trainer.config = _market_sim_test_config(tmp_path)
    trainer._build_market_sim_frames = lambda: (pd.DataFrame(), pd.DataFrame(), signal_frame)

    class _Result:
        combined_equity = pd.Series([10_000.0, 10_100.0, 10_050.0, 10_200.0], index=timestamps)
        per_symbol = {"AAA": type("_SymbolResult", (), {"trades": [object(), object()]})()}

    monkeypatch.setattr("fastforecaster2.trainer.run_shared_cash_simulation", lambda *args, **kwargs: _Result())

    summary = trainer._run_market_sim_eval()

    assert summary["sim_trades"] == 2.0
    assert summary["sim_periods"] == 4.0
    assert summary["sim_trade_rate"] == compute_trade_rate(2.0, 4.0)
    assert summary["sim_goodness_score"] != 0.0
    assert trainer.config.simulator_equity_file.exists()
