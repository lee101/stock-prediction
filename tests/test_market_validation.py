from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from newnanoalpacahourlyexp.marketsimulator.hourly_trader import (
    HourlyTraderMarketSimulator,
    HourlyTraderSimulationConfig,
)
from pufferlib_market.inference import Policy
from unified_orchestrator.market_validation import (
    ValidationConfig,
    _load_local_bars,
    _scaled_buy_amount_pct,
    build_market_sim_trade_plan,
    build_validation_frames,
)
from unified_orchestrator.rl_gemini_bridge import RLSignal


def _write_checkpoint(path: Path, *, obs_size: int = 39, num_actions: int = 5) -> None:
    model = Policy(obs_size=obs_size, num_actions=num_actions, hidden=16, num_blocks=1)
    payload = {
        "model": model.state_dict(),
        "config": {"hidden_size": 16, "num_blocks": 1},
        "action_allocation_bins": 1,
        "action_level_bins": 1,
    }
    torch.save(payload, path)


def _make_bars(symbol: str, *, periods: int = 80, start: str = "2026-01-01T00:00:00Z") -> pd.DataFrame:
    ts = pd.date_range(start, periods=periods, freq="h", tz="UTC")
    close = pd.Series([100.0 + idx * 0.25 for idx in range(periods)])
    return pd.DataFrame(
        {
            "timestamp": ts,
            "symbol": symbol,
            "open": close - 0.1,
            "high": close + 0.6,
            "low": close - 0.4,
            "close": close,
            "volume": 1000.0,
        }
    )


def test_load_local_bars_preserves_ohlc_from_pickled_stock_history(tmp_path: Path, monkeypatch) -> None:
    stocks_dir = tmp_path / "trainingdatahourly" / "stocks"
    stocks_dir.mkdir(parents=True)
    frame = _make_bars("AAPL", periods=4).set_index("timestamp")
    pd.to_pickle({"data": frame, "cutoff": None}, stocks_dir / "AAPL_hist.pkl")

    monkeypatch.setattr("unified_orchestrator.market_validation.REPO", tmp_path)
    loaded = _load_local_bars("AAPL", "stock")

    assert "open" in loaded.columns
    assert "timestamp" in loaded.columns
    assert loaded["open"].iloc[0] == frame["open"].iloc[0]


def test_scaled_buy_amount_pct_uses_equal_split_for_single_bin_crypto() -> None:
    amount = _scaled_buy_amount_pct(
        signal=RLSignal(
            symbol_idx=0,
            symbol_name="BTCUSD",
            direction="long",
            confidence=0.9,
            logit_gap=2.0,
            allocation_pct=1.0,
        ),
        spec_alloc_bins=1,
        long_signal_count=2,
        asset_class="crypto",
    )

    assert amount == 80.0


def test_build_market_sim_trade_plan_keeps_positive_levels_for_hold_signal() -> None:
    plan = build_market_sim_trade_plan(
        signal=RLSignal(
            symbol_idx=0,
            symbol_name="BTCUSD",
            direction="flat",
            confidence=0.2,
            logit_gap=0.0,
            allocation_pct=0.0,
        ),
        current_price=100.0,
        history_frame=_make_bars("BTCUSD", periods=72),
        forecast_1h={"predicted_close_p50": 100.8, "predicted_high_p50": 101.2},
        forecast_24h={"predicted_close_p50": 101.5, "predicted_close_p90": 102.0},
        asset_class="crypto",
        confidence_threshold=0.7,
    )

    assert plan.direction == "hold"
    assert plan.buy_price > 0.0
    assert plan.sell_price > plan.buy_price


def test_build_validation_frames_generates_portfolio_actions(monkeypatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "bridge.pt"
    _write_checkpoint(checkpoint)

    bars = {
        "BTCUSD": _make_bars("BTCUSD"),
        "ETHUSD": _make_bars("ETHUSD"),
    }

    monkeypatch.setattr(
        "unified_orchestrator.market_validation._read_trained_symbols_for_checkpoint",
        lambda checkpoint_path, expected_num_symbols: ["BTCUSD", "ETHUSD"],
    )
    monkeypatch.setattr(
        "unified_orchestrator.market_validation._load_local_bars",
        lambda symbol, asset_class: bars[symbol],
    )
    monkeypatch.setattr(
        "unified_orchestrator.market_validation._choose_forecast_cache_root",
        lambda symbols, candidates: None,
    )
    monkeypatch.setattr(
        "unified_orchestrator.market_validation._load_forecast_frames",
        lambda symbols, cache_root: ({}, {}),
    )
    monkeypatch.setattr(
        "unified_orchestrator.market_validation._build_crypto_rl_signal_map",
        lambda history_frames, bridge: {
            "BTCUSD": RLSignal(
                symbol_idx=0,
                symbol_name="BTCUSD",
                direction="long",
                confidence=0.95,
                logit_gap=3.0,
                allocation_pct=1.0,
            ),
            "ETHUSD": RLSignal(
                symbol_idx=1,
                symbol_name="ETHUSD",
                direction="flat",
                confidence=0.4,
                logit_gap=0.0,
                allocation_pct=0.0,
            ),
        },
    )

    config = ValidationConfig(asset_class="crypto", days=1)
    bars_df, actions_df, signal_universe, start_ts, end_ts = build_validation_frames(
        checkpoint_path=checkpoint,
        config=config,
        eval_symbols=["BTCUSD", "ETHUSD"],
    )

    assert signal_universe == ["BTCUSD", "ETHUSD"]
    assert start_ts < end_ts
    assert set(bars_df["symbol"]) == {"BTCUSD", "ETHUSD"}
    assert set(actions_df["symbol"]) == {"BTCUSD", "ETHUSD"}

    btc_rows = actions_df[actions_df["symbol"] == "BTCUSD"]
    eth_rows = actions_df[actions_df["symbol"] == "ETHUSD"]
    assert (btc_rows["buy_amount"] > 0.0).all()
    assert (btc_rows["sell_amount"] == 100.0).all()
    assert (btc_rows["sell_price"] > btc_rows["buy_price"]).all()

    assert (eth_rows["buy_amount"] == 0.0).all()
    assert (eth_rows["sell_amount"] == 100.0).all()
    assert (eth_rows["buy_price"] > 0.0).all()
    assert (eth_rows["sell_price"] > eth_rows["buy_price"]).all()


def test_build_validation_frames_daily_cadence_repeats_plan_within_day(monkeypatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "bridge.pt"
    _write_checkpoint(checkpoint)

    def _bars_for_two_days(symbol: str) -> pd.DataFrame:
        ts = pd.to_datetime(
            [
                "2026-01-01T00:00:00Z",
                "2026-01-01T01:00:00Z",
                "2026-01-01T02:00:00Z",
                "2026-01-02T00:00:00Z",
                "2026-01-02T01:00:00Z",
            ],
            utc=True,
        )
        close = pd.Series([100.0, 100.5, 101.0, 99.5, 99.0])
        return pd.DataFrame(
            {
                "timestamp": ts,
                "symbol": symbol,
                "open": close,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": 1000.0,
            }
        )

    bars = {"BTCUSD": _bars_for_two_days("BTCUSD")}

    monkeypatch.setattr(
        "unified_orchestrator.market_validation._read_trained_symbols_for_checkpoint",
        lambda checkpoint_path, expected_num_symbols: ["BTCUSD"],
    )
    monkeypatch.setattr(
        "unified_orchestrator.market_validation._load_local_bars",
        lambda symbol, asset_class: bars[symbol],
    )
    monkeypatch.setattr(
        "unified_orchestrator.market_validation._choose_forecast_cache_root",
        lambda symbols, candidates: None,
    )
    monkeypatch.setattr(
        "unified_orchestrator.market_validation._load_forecast_frames",
        lambda symbols, cache_root: ({}, {}),
    )

    def _signal_map(history_frames, bridge):
        latest_ts = pd.Timestamp(history_frames["BTCUSD"].iloc[-1]["timestamp"]).floor("D")
        if latest_ts == pd.Timestamp("2026-01-01T00:00:00Z"):
            return {
                "BTCUSD": RLSignal(
                    symbol_idx=0,
                    symbol_name="BTCUSD",
                    direction="long",
                    confidence=0.95,
                    logit_gap=3.0,
                    allocation_pct=1.0,
                )
            }
        return {
            "BTCUSD": RLSignal(
                symbol_idx=0,
                symbol_name="BTCUSD",
                direction="flat",
                confidence=0.2,
                logit_gap=0.0,
                allocation_pct=0.0,
            )
        }

    monkeypatch.setattr(
        "unified_orchestrator.market_validation._build_crypto_rl_signal_map",
        _signal_map,
    )

    config = ValidationConfig(asset_class="crypto", days=2, decision_cadence="daily")
    _bars_df, actions_df, _signal_universe, _start_ts, _end_ts = build_validation_frames(
        checkpoint_path=checkpoint,
        config=config,
        eval_symbols=["BTCUSD"],
    )

    day1 = actions_df[actions_df["timestamp"].dt.floor("D") == pd.Timestamp("2026-01-01T00:00:00Z")]
    day2 = actions_df[actions_df["timestamp"].dt.floor("D") == pd.Timestamp("2026-01-02T00:00:00Z")]

    assert len(day1) == 3
    assert len(day2) == 2
    assert day1["decision_timestamp"].nunique() == 1
    assert day2["decision_timestamp"].nunique() == 1
    assert (day1["buy_amount"] > 0.0).all()
    assert (day2["buy_amount"] == 0.0).all()
    assert day1["buy_price"].nunique() == 1
    assert day1["sell_price"].nunique() == 1


def test_hourly_trader_simulator_repeated_daily_plan_can_reenter_after_same_day_exit() -> None:
    ts0 = pd.Timestamp("2026-01-01T00:00:00Z")
    ts1 = ts0 + pd.Timedelta(hours=1)
    ts2 = ts1 + pd.Timedelta(hours=1)
    ts3 = ts2 + pd.Timedelta(hours=1)

    bars = pd.DataFrame(
        [
            {"timestamp": ts0, "symbol": "BTCUSD", "open": 104.0, "high": 105.0, "low": 103.5, "close": 104.5},
            {"timestamp": ts1, "symbol": "BTCUSD", "open": 100.2, "high": 100.4, "low": 99.5, "close": 100.0},
            {"timestamp": ts2, "symbol": "BTCUSD", "open": 104.2, "high": 105.2, "low": 103.8, "close": 104.8},
            {"timestamp": ts3, "symbol": "BTCUSD", "open": 100.1, "high": 100.3, "low": 99.4, "close": 100.0},
        ]
    )
    actions = pd.DataFrame(
        [
            {"timestamp": ts0, "symbol": "BTCUSD", "buy_price": 100.0, "sell_price": 105.0, "buy_amount": 100.0, "sell_amount": 100.0},
            {"timestamp": ts1, "symbol": "BTCUSD", "buy_price": 100.0, "sell_price": 105.0, "buy_amount": 100.0, "sell_amount": 100.0},
            {"timestamp": ts2, "symbol": "BTCUSD", "buy_price": 100.0, "sell_price": 105.0, "buy_amount": 100.0, "sell_amount": 100.0},
            {"timestamp": ts3, "symbol": "BTCUSD", "buy_price": 100.0, "sell_price": 105.0, "buy_amount": 100.0, "sell_amount": 100.0},
        ]
    )

    sim = HourlyTraderMarketSimulator(
        HourlyTraderSimulationConfig(
            initial_cash=1000.0,
            allocation_usd=1000.0,
            allocation_pct=None,
            decision_lag_bars=1,
            enforce_market_hours=False,
            fee_by_symbol={"BTCUSD": 0.0},
            always_full_exit=True,
            allow_position_adds=False,
        )
    )
    result = sim.run(bars, actions)

    assert [fill.side for fill in result.fills] == ["buy", "sell", "buy"]
    assert result.fills[0].timestamp == ts1
    assert result.fills[1].timestamp == ts2
    assert result.fills[2].timestamp == ts3
