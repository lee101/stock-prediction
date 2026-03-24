"""Tests for qwen_rl_trading.data_prompt."""
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from qwen_rl_trading.data_prompt import (
    SYMBOLS_30,
    MarketSnapshot,
    PromptDataset,
    _compute_features,
    build_chat_messages,
    format_prompt,
    load_symbol_data,
)


def _make_csv_dir(symbols=("BTCUSD", "ETHUSD"), n_hours=200):
    tmpdir = tempfile.mkdtemp()
    base_ts = pd.Timestamp("2025-01-01", tz="UTC")
    for sym in symbols:
        base_price = 67000 if sym == "BTCUSD" else 3500
        rows = []
        for h in range(n_hours):
            rows.append({
                "timestamp": (base_ts + pd.Timedelta(hours=h)).isoformat(),
                "open": base_price + h * 0.5,
                "high": base_price + h * 0.5 + 20,
                "low": base_price + h * 0.5 - 15,
                "close": base_price + h * 0.5 + 5,
                "volume": 100 + h,
                "trade_count": 50,
                "vwap": base_price + h * 0.5 + 2,
                "symbol": sym,
            })
        df = pd.DataFrame(rows)
        df.to_csv(Path(tmpdir) / f"{sym}.csv", index=False)
    return Path(tmpdir)


class TestComputeFeatures:
    def test_adds_columns(self):
        data_dir = _make_csv_dir(symbols=("BTCUSD",))
        df = load_symbol_data(data_dir, "BTCUSD")
        result = _compute_features(df)
        for col in ["return_1h", "return_4h", "return_24h", "volatility_24h", "range_pct", "volume_z"]:
            assert col in result.columns


class TestLoadSymbolData:
    def test_loads_csv(self):
        data_dir = _make_csv_dir(symbols=("BTCUSD",))
        df = load_symbol_data(data_dir, "BTCUSD")
        assert df is not None
        assert "timestamp" in df.columns
        assert "close" in df.columns
        assert len(df) == 200

    def test_missing_symbol(self):
        data_dir = _make_csv_dir(symbols=("BTCUSD",))
        assert load_symbol_data(data_dir, "DOESNOTEXIST") is None


class TestPromptDataset:
    def test_basic_construction(self):
        data_dir = _make_csv_dir(n_hours=200)
        ds = PromptDataset(data_dir, ["BTCUSD", "ETHUSD"], lookback=24, eval_horizon=24, stride=4)
        assert len(ds) > 0

    def test_getitem_returns_tuple(self):
        data_dir = _make_csv_dir(n_hours=200)
        ds = PromptDataset(data_dir, ["BTCUSD", "ETHUSD"], lookback=24, eval_horizon=24, stride=4)
        prompt, snapshot = ds[0]
        assert isinstance(prompt, str)
        assert isinstance(snapshot, MarketSnapshot)

    def test_snapshot_has_forward_bars(self):
        data_dir = _make_csv_dir(n_hours=200)
        ds = PromptDataset(data_dir, ["BTCUSD", "ETHUSD"], lookback=24, eval_horizon=24, stride=4)
        _, snapshot = ds[0]
        assert not snapshot.forward_bars.empty
        assert "timestamp" in snapshot.forward_bars.columns
        assert "symbol" in snapshot.forward_bars.columns

    def test_window_id_in_prompt(self):
        data_dir = _make_csv_dir(n_hours=200)
        ds = PromptDataset(data_dir, ["BTCUSD", "ETHUSD"], lookback=24, eval_horizon=24, stride=4)
        prompt, snapshot = ds[0]
        assert f"[window:{snapshot.window_id}]" in prompt

    def test_val_split(self):
        data_dir = _make_csv_dir(n_hours=300)
        train_ds = PromptDataset(data_dir, ["BTCUSD"], lookback=24, eval_horizon=24, stride=4,
                                  val_fraction=0.2, val_mode=False)
        val_ds = PromptDataset(data_dir, ["BTCUSD"], lookback=24, eval_horizon=24, stride=4,
                                val_fraction=0.2, val_mode=True)
        assert len(train_ds) > 0
        assert len(val_ds) > 0
        assert len(train_ds) > len(val_ds)


class TestFormatPrompt:
    def test_minimal(self):
        snap = MarketSnapshot(
            window_id="abc", timestamp=pd.Timestamp("2025-01-01", tz="UTC"),
            symbols=["BTCUSD"], features={"BTCUSD": {"price": 67000, "return_1h": 0.01, "return_4h": 0.02,
                                                       "return_24h": 0.05, "volatility_24h": 0.02,
                                                       "range_pct": 0.01, "volume_z": 0.5}},
            chronos_forecasts={}, forward_bars=pd.DataFrame(),
        )
        text = format_prompt(snap, variant="minimal")
        assert "BTCUSD" in text
        assert "p=67000" in text

    def test_detailed_has_vol(self):
        snap = MarketSnapshot(
            window_id="abc", timestamp=pd.Timestamp("2025-01-01", tz="UTC"),
            symbols=["BTCUSD"], features={"BTCUSD": {"price": 67000, "return_1h": 0.01, "return_4h": 0.02,
                                                       "return_24h": 0.05, "volatility_24h": 0.02,
                                                       "range_pct": 0.01, "volume_z": 0.5}},
            chronos_forecasts={}, forward_bars=pd.DataFrame(),
        )
        text = format_prompt(snap, variant="detailed")
        assert "vol24h" in text
        assert "vz=" in text

    def test_with_chronos2(self):
        snap = MarketSnapshot(
            window_id="abc", timestamp=pd.Timestamp("2025-01-01", tz="UTC"),
            symbols=["BTCUSD"], features={"BTCUSD": {"price": 67000, "return_1h": 0.01, "return_4h": 0.02,
                                                       "return_24h": 0.05, "volatility_24h": 0.02,
                                                       "range_pct": 0.01, "volume_z": 0.5}},
            chronos_forecasts={"BTCUSD": {"close_delta_h1": 0.003, "high_delta_h1": 0.005}},
            forward_bars=pd.DataFrame(),
        )
        text = format_prompt(snap, variant="with_chronos2")
        assert "close_delta_h1" in text


class TestBuildChatMessages:
    def test_structure(self):
        msgs = build_chat_messages("some prompt")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "some prompt"
