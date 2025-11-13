import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import types

import pandas as pd
import pytest
import torch

import backtest_test3_inline
import trade_stock_e2e

if "scripts.alpaca_cli" not in sys.modules:
    stub_cli = types.ModuleType("scripts.alpaca_cli")

    def _noop_set_strategy(symbol: str, strategy: str) -> None:
        return None

    stub_cli.set_strategy_for_symbol = _noop_set_strategy
    sys.modules["scripts.alpaca_cli"] = stub_cli

if not hasattr(backtest_test3_inline, "download_daily_stock_data"):
    module_path = Path(__file__).resolve().parents[3] / "backtest_test3_inline.py"
    spec = importlib.util.spec_from_file_location("backtest_test3_inline", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to locate backtest_test3_inline module for integration test")
    module = importlib.util.module_from_spec(spec)
    sys.modules["backtest_test3_inline"] = module
    spec.loader.exec_module(module)
    backtest_test3_inline = module  # type: ignore[assignment]
    trade_stock_e2e.backtest_forecasts = module.backtest_forecasts  # type: ignore[attr-defined]
    trade_stock_e2e.release_model_resources = module.release_model_resources  # type: ignore[attr-defined]


@pytest.mark.slow
def test_trade_stock_e2e_uses_chronos2_swept_configs(monkeypatch):
    """Full Chronos2 integration covering trade_stock_e2e + trainingdata MAE."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Chronos2 integration test")

    symbol = "AAPL"
    data_path = Path(__file__).resolve().parents[3] / "trainingdata" / f"{symbol}.csv"
    if not data_path.exists():
        pytest.skip(f"Missing training dataset for {symbol}")

    raw_df = pd.read_csv(data_path)
    if not {"open", "high", "low", "close"}.issubset({col.lower() for col in raw_df.columns}):
        pytest.skip(f"Training dataset {data_path} missing OHLC columns")

    ohlc_df = raw_df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "timestamp": "Timestamp",
        }
    )
    if "Timestamp" in ohlc_df.columns:
        ohlc_df["Timestamp"] = pd.to_datetime(ohlc_df["Timestamp"])
        ohlc_df = ohlc_df.set_index("Timestamp")
    else:
        ohlc_df.index = pd.date_range(start="2020-01-01", periods=len(ohlc_df), freq="D")
    ohlc_df = ohlc_df[["Open", "High", "Low", "Close"]]

    def _load_training_stock_data(current_time: str, symbols):
        assert symbols == [symbol], f"Unexpected symbol request: {symbols}"
        return ohlc_df.copy()

    monkeypatch.setattr(backtest_test3_inline, "download_daily_stock_data", _load_training_stock_data)
    monkeypatch.setattr(backtest_test3_inline, "fetch_spread", lambda _: 0.75)

    monkeypatch.setenv("ONLY_CHRONOS2", "1")
    monkeypatch.setenv("MARKETSIM_BACKTEST_SIMULATIONS", "3")
    monkeypatch.delenv("TESTING", raising=False)
    monkeypatch.delenv("MARKETSIM_ALLOW_MOCK_ANALYTICS", raising=False)

    monkeypatch.setattr(trade_stock_e2e, "is_nyse_trading_day_now", lambda: True)
    monkeypatch.setattr(trade_stock_e2e, "should_skip_closed_equity", lambda: False)
    monkeypatch.setattr(trade_stock_e2e, "get_bid", lambda _: float(ohlc_df["Close"].iloc[-1]))
    monkeypatch.setattr(trade_stock_e2e, "get_ask", lambda _: float(ohlc_df["Close"].iloc[-1]) + 0.01)
    monkeypatch.setattr(trade_stock_e2e, "download_exchange_latest_data", lambda *args, **kwargs: None)

    backtest_df = trade_stock_e2e.backtest_forecasts(symbol, num_simulations=3)
    assert not backtest_df.empty, "backtest_forecasts returned no rows"
    row = backtest_df.iloc[0]
    assert row.get("close_prediction_source") == "chronos2"

    preaug_path = Path("preaugstrategies") / "chronos2" / f"{symbol}.json"
    if not preaug_path.exists():
        pytest.skip(f"Missing pre-augmentation sweep for {symbol}")
    expected_preaug = json.loads(preaug_path.read_text()).get("best_strategy")
    assert row.get("chronos2_preaug_strategy") == expected_preaug
    assert row.get("chronos2_preaug_source") and preaug_path.as_posix() in row["chronos2_preaug_source"]

    hyper_path = Path("hyperparams") / "chronos2" / f"{symbol}.json"
    if not hyper_path.exists():
        pytest.skip(f"Missing hyperparameter sweep for {symbol}")
    hyper_config = json.loads(hyper_path.read_text()).get("config", {})
    assert row.get("chronos2_context_length") == hyper_config.get("context_length")
    assert row.get("chronos2_batch_size") == hyper_config.get("batch_size")

    close_mae = row.get("close_val_loss")
    assert close_mae is not None and math.isfinite(close_mae) and close_mae > 0
