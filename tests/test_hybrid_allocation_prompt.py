from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest


REPO = Path(__file__).resolve().parents[1]
RL_BINANCE_DIR = REPO / "rl-trading-agent-binance"


def _install_google_genai_stub() -> None:
    google_module = sys.modules.setdefault("google", types.ModuleType("google"))
    if hasattr(google_module, "genai") and "google.genai" in sys.modules:
        return

    class _Schema:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class _ThinkingConfig:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class _GenerateContentConfig:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class _Content:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class _Part:
        @staticmethod
        def from_text(*args, **kwargs):
            return {"args": args, "kwargs": kwargs}

    class _Type:
        OBJECT = "object"
        STRING = "string"

    types_namespace = types.SimpleNamespace(
        Schema=_Schema,
        Type=_Type,
        ThinkingConfig=_ThinkingConfig,
        GenerateContentConfig=_GenerateContentConfig,
        Content=_Content,
        Part=_Part,
    )
    genai_module = types.ModuleType("google.genai")
    genai_module.types = types_namespace
    google_module.genai = genai_module
    sys.modules["google.genai"] = genai_module


def _load_module(name: str, relative_path: str):
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    if str(RL_BINANCE_DIR) not in sys.path:
        sys.path.insert(0, str(RL_BINANCE_DIR))
    module_path = RL_BINANCE_DIR / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_install_google_genai_stub()
hybrid_prompt = _load_module("hybrid_prompt", "hybrid_prompt.py")
trade_binance_live = _load_module("trade_binance_live", "trade_binance_live.py")


def _sample_klines() -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=12, freq="h", tz="UTC")
    closes = [100.0 + i for i in range(12)]
    frame = pd.DataFrame(
        {
            "open": closes,
            "high": [c + 1.0 for c in closes],
            "low": [c - 1.0 for c in closes],
            "close": closes,
            "volume": [1000.0 + i for i in range(12)],
        },
        index=index,
    )
    frame.index.name = "timestamp"
    return frame


def test_parse_allocation_response_accepts_decorated_numbers_and_scales_total() -> None:
    text = json.dumps(
        {
            "btc_pct": "60%",
            "btc_entry": "$100,123.45",
            "btc_exit": "$101,500.00",
            "eth_pct": "50%",
            "eth_entry": "$2,450.50",
            "eth_exit": "$2,520.25",
            "doge_pct": "n/a",
            "doge_entry": "$0.00",
            "doge_exit": "$0.00",
            "aave_pct": "",
            "aave_entry": 0,
            "aave_exit": 0,
            "reasoning": "Concentrate in BTC and ETH.",
        }
    )

    plan = hybrid_prompt.parse_allocation_response(text)

    assert plan.allocations["BTCUSD"] == pytest.approx(60.0 * 100.0 / 110.0)
    assert plan.allocations["ETHUSD"] == pytest.approx(50.0 * 100.0 / 110.0)
    assert sum(plan.allocations.values()) == pytest.approx(100.0)
    assert plan.entry_prices["BTCUSD"] == pytest.approx(100123.45)
    assert plan.exit_prices["ETHUSD"] == pytest.approx(2520.25)
    assert "DOGEUSD" not in plan.allocations


def test_build_allocation_prompt_keeps_rl_probs_mapped_to_symbol() -> None:
    contexts = [
        hybrid_prompt.SymbolContext(symbol="BTCUSD", price=111.0, klines=_sample_klines()),
        hybrid_prompt.SymbolContext(symbol="DOGEUSD", price=0.42, klines=_sample_klines()),
    ]
    rl_signal = hybrid_prompt.RLSignal(
        action=3,
        action_name="LONG_DOGE",
        target_symbol="DOGEUSD",
        direction="long",
        logits=[0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 8.0, 0.0],
        value=1.23,
    )

    prompt = hybrid_prompt.build_allocation_prompt(
        contexts=contexts,
        rl_signal=rl_signal,
        portfolio_value=1000.0,
        cash_usd=1000.0,
        positions={},
    )

    probs = hybrid_prompt._softmax(rl_signal.logits)
    expected_doge_line = f"RL policy: long={probs[3]:.0%} | short={probs[7]:.0%}"
    expected_btc_line = f"RL policy: long={probs[1]:.0%} | short={probs[5]:.0%}"

    doge_block = prompt.split("--- DOGEUSD ---", 1)[1]
    btc_block = prompt.split("--- BTCUSD ---", 1)[1].split("--- DOGEUSD ---", 1)[0]

    assert expected_doge_line in doge_block
    assert expected_btc_line in btc_block


def test_allocation_plan_has_error_detects_parser_and_api_failures() -> None:
    assert trade_binance_live._allocation_plan_has_error(
        hybrid_prompt.AllocationPlan(reasoning="Failed to parse response")
    )
    assert trade_binance_live._allocation_plan_has_error(
        hybrid_prompt.AllocationPlan(reasoning="API error: rate limited")
    )
    assert not trade_binance_live._allocation_plan_has_error(
        hybrid_prompt.AllocationPlan(reasoning="Stay fully in cash while signals disagree.")
    )


def test_reserve_quote_balance_only_reduces_requested_asset() -> None:
    state = trade_binance_live.PortfolioState(fdusd_balance=100.0, usdt_balance=25.0)

    trade_binance_live._reserve_quote_balance(state, "FDUSD", 60.0)
    assert state.fdusd_balance == pytest.approx(40.0)
    assert state.usdt_balance == pytest.approx(25.0)

    trade_binance_live._reserve_quote_balance(state, "USDT", 30.0)
    assert state.fdusd_balance == pytest.approx(40.0)
    assert state.usdt_balance == pytest.approx(0.0)


def test_resolve_spot_leverage_clamps_anything_above_one() -> None:
    assert trade_binance_live._resolve_spot_leverage(0.5) == pytest.approx(0.5)
    assert trade_binance_live._resolve_spot_leverage(1.0) == pytest.approx(1.0)
    assert trade_binance_live._resolve_spot_leverage(5.0) == pytest.approx(1.0)
