from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

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
