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


def test_resolve_execution_mode_auto_uses_margin_only_for_real_leverage() -> None:
    assert trade_binance_live._resolve_execution_mode("auto", 1.0) == "spot"
    assert trade_binance_live._resolve_execution_mode("auto", 1.01) == "margin"
    assert trade_binance_live._resolve_execution_mode("margin", 1.0) == "margin"


def test_execution_pair_uses_usdt_pairs_for_margin_on_fdusd_symbols() -> None:
    btc = trade_binance_live.TRADING_SYMBOLS["BTCUSD"]
    sol = trade_binance_live.TRADING_SYMBOLS["SOLUSD"]

    assert trade_binance_live._execution_pair(btc, "spot") == "BTCFDUSD"
    assert trade_binance_live._execution_pair(btc, "margin") == "BTCUSDT"
    assert trade_binance_live._execution_pair(sol, "margin") == "SOLUSDT"


def test_normalize_live_trade_plan_sets_visible_exit_for_held_position() -> None:
    btc = trade_binance_live.TRADING_SYMBOLS["BTCUSD"]
    plan = trade_binance_live.TradePlan(
        direction="hold",
        buy_price=0.0,
        sell_price=0.0,
        confidence=0.85,
        reasoning="hold without target",
    )

    normalized = trade_binance_live._normalize_live_trade_plan(
        plan,
        btc,
        current_price=70537.44,
        execution_mode="margin",
        position_qty=0.01473671,
        position_entry_price=71000.0,
    )

    assert normalized.direction == "hold"
    assert normalized.sell_price == pytest.approx(71242.8144)


def test_normalize_live_trade_plan_sets_exit_for_long_entry_without_target() -> None:
    eth = trade_binance_live.TRADING_SYMBOLS["ETHUSD"]
    plan = trade_binance_live.TradePlan(
        direction="long",
        buy_price=2175.0,
        sell_price=0.0,
        confidence=0.65,
        reasoning="enter long",
    )

    normalized = trade_binance_live._normalize_live_trade_plan(
        plan,
        eth,
        current_price=2181.18,
        execution_mode="margin",
    )

    assert normalized.buy_price == pytest.approx(2175.0)
    assert normalized.sell_price > normalized.buy_price
    assert normalized.sell_price == pytest.approx(2202.9918)


@pytest.mark.parametrize(
    ("reasoning", "expected"),
    [
        ("API exhausted", True),
        ("codex API exhausted", True),
        ("API error: rate limited", True),
        ("No tool use in response", True),
        ("Could not parse response", True),
        ("Failed to parse response", True),
        ("All retries exhausted", True),
        ("Stay flat while forecasts disagree.", False),
    ],
)
def test_trade_plan_indicates_provider_failure(reasoning: str, expected: bool) -> None:
    plan = trade_binance_live.TradePlan(
        direction="hold",
        buy_price=0.0,
        sell_price=0.0,
        confidence=0.0,
        reasoning=reasoning,
    )

    assert trade_binance_live._trade_plan_indicates_provider_failure(plan) is expected


def test_get_hybrid_signal_falls_back_on_provider_error(monkeypatch: pytest.MonkeyPatch) -> None:
    prompt_module = _load_module("rl_trading_agent_binance_prompt", "rl_trading_agent_binance_prompt.py")
    btc = trade_binance_live.TRADING_SYMBOLS["BTCUSD"]
    expected = trade_binance_live.TradePlan(
        direction="long",
        buy_price=69900.0,
        sell_price=70500.0,
        confidence=0.55,
        reasoning="chronos2_fallback: test",
    )

    monkeypatch.setattr(trade_binance_live.binance_wrapper, "get_symbol_price", lambda symbol: 70000.0)
    monkeypatch.setattr(prompt_module, "load_latest_forecast", lambda symbol, horizon: {"predicted_close_p50": 70100.0})
    monkeypatch.setattr(prompt_module, "build_live_prompt", lambda *args, **kwargs: "prompt")
    monkeypatch.setattr(
        trade_binance_live,
        "call_llm",
        lambda *args, **kwargs: trade_binance_live.TradePlan(
            direction="hold",
            buy_price=0.0,
            sell_price=0.0,
            confidence=0.0,
            reasoning="API error: rate limited",
        ),
    )
    monkeypatch.setattr(trade_binance_live, "_chronos2_fallback_signal", lambda *args, **kwargs: expected)
    monkeypatch.setattr(trade_binance_live, "_normalize_live_trade_plan", lambda plan, *args, **kwargs: plan)

    plan = trade_binance_live.get_hybrid_signal(
        btc,
        model="gemini-3.1-flash-lite-preview",
        thinking_level="HIGH",
        position_qty=0.0,
        execution_mode="margin",
        prefetched_bars=[
            {
                "timestamp": f"2026-03-20T{hour:02d}:00:00Z",
                "open": 70000.0 + hour,
                "high": 70010.0 + hour,
                "low": 69990.0 + hour,
                "close": 70005.0 + hour,
                "volume": 1000.0 + hour,
            }
            for hour in range(24)
        ],
    )

    assert plan is expected


def test_quote_buying_power_only_counts_borrow_headroom_in_margin_mode() -> None:
    state = trade_binance_live.PortfolioState(
        fdusd_balance=50.0,
        usdt_balance=25.0,
        borrowable_quotes={"USDT": 125.0},
    )

    assert trade_binance_live._quote_buying_power(
        state,
        "USDT",
        execution_mode="spot",
        effective_leverage=5.0,
    ) == pytest.approx(25.0)
    assert trade_binance_live._quote_buying_power(
        state,
        "USDT",
        execution_mode="margin",
        effective_leverage=1.0,
    ) == pytest.approx(25.0)
    assert trade_binance_live._quote_buying_power(
        state,
        "USDT",
        execution_mode="margin",
        effective_leverage=5.0,
    ) == pytest.approx(150.0)


def test_get_portfolio_state_margin_uses_net_asset_for_locked_inventory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(trade_binance_live, "get_margin_free_balance", lambda asset: 1595.35 if asset == "USDT" else 0.0)
    monkeypatch.setattr(trade_binance_live, "get_margin_borrowed_balance", lambda asset: 0.0)
    monkeypatch.setattr(trade_binance_live, "get_max_borrowable", lambda asset: 10547.83 if asset == "USDT" else 0.0)
    monkeypatch.setattr(trade_binance_live, "get_margin_account", lambda: {"totalNetAssetOfBtc": "0.0374722"})
    monkeypatch.setattr(
        trade_binance_live.binance_wrapper,
        "get_symbol_price",
        lambda symbol: 70434.01 if symbol == "BTCUSDT" else 1.0,
    )

    balances = {
        "BTC": {"free": "0.00000671", "locked": "0.01473", "netAsset": "0.01473671"},
        "ETH": {"free": "0.00001189", "locked": "0", "netAsset": "0.00000842"},
    }
    monkeypatch.setattr(trade_binance_live, "get_margin_asset_balance", lambda asset: balances.get(asset, None))

    state = trade_binance_live.get_portfolio_state(execution_mode="margin")

    assert state.usdt_balance == pytest.approx(1595.35)
    assert state.borrowable_quotes["USDT"] == pytest.approx(10547.83)
    assert state.positions["BTC"] == pytest.approx(0.01473671)
    assert state.positions["ETH"] == pytest.approx(0.00000842)
    assert state.total_value_usd == pytest.approx(0.0374722 * 70434.01)


def test_reserve_buying_power_uses_free_quote_before_borrow_headroom() -> None:
    state = trade_binance_live.PortfolioState(
        usdt_balance=40.0,
        borrowable_quotes={"USDT": 120.0},
    )

    trade_binance_live._reserve_buying_power(
        state,
        "USDT",
        100.0,
        execution_mode="margin",
    )

    assert state.usdt_balance == pytest.approx(0.0)
    assert state.borrowable_quotes["USDT"] == pytest.approx(60.0)


def test_build_margin_capital_sync_plan_moves_base_assets_and_converts_fdusd_to_usdt() -> None:
    plan = trade_binance_live._build_margin_capital_sync_plan(
        spot_free={
            "USDT": 50.0,
            "FDUSD": 75.0,
            "BTC": 0.25,
            "ETH": 0.0,
            "SOL": 0.0,
            "DOGE": 0.0,
            "SUI": 0.0,
            "AAVE": 0.0,
        },
        margin_free={
            "USDT": 10.0,
            "FDUSD": 25.0,
            "BTC": 0.0,
            "ETH": 0.0,
            "SOL": 0.0,
            "DOGE": 0.0,
            "SUI": 0.0,
            "AAVE": 0.0,
        },
    )

    assert len(plan.spot_to_margin) == 1
    assert plan.spot_to_margin[0].asset == "BTC"
    assert plan.spot_to_margin[0].amount == pytest.approx(0.24999999)
    assert len(plan.margin_to_spot) == 1
    assert plan.margin_to_spot[0].asset == "FDUSD"
    assert plan.margin_to_spot[0].amount == pytest.approx(24.999999)
    assert plan.spot_fdusd_to_usdt == pytest.approx(99.999998)
    assert plan.transfer_all_spot_usdt_to_margin is True


def test_build_margin_capital_sync_plan_skips_small_fdusd_conversion() -> None:
    plan = trade_binance_live._build_margin_capital_sync_plan(
        spot_free={"USDT": 0.0, "FDUSD": 4.0},
        margin_free={"USDT": 0.0, "FDUSD": 3.0},
    )

    assert plan.margin_to_spot == ()
    assert plan.spot_fdusd_to_usdt == pytest.approx(0.0)
    assert plan.transfer_all_spot_usdt_to_margin is False


def test_dedupe_side_orders_skips_when_existing_sell_covers_desired_qty() -> None:
    open_orders = [
        {
            "symbol": "BTCUSDT",
            "side": "SELL",
            "origQty": "0.50",
            "executedQty": "0.00",
            "price": "71000.00",
            "orderId": 123,
        }
    ]

    remaining, skip = trade_binance_live._dedupe_side_orders(
        open_orders,
        symbol="BTCUSDT",
        side="SELL",
        execution_mode="margin",
        dry_run=True,
        desired_qty=0.49,
    )

    assert skip is True
    assert remaining == open_orders


def test_dedupe_side_orders_replaces_underfilled_buy_order() -> None:
    open_orders = [
        {
            "symbol": "ETHUSDT",
            "side": "BUY",
            "origQty": "0.10",
            "executedQty": "0.00",
            "price": "2000.00",
            "orderId": 456,
        }
    ]

    remaining, skip = trade_binance_live._dedupe_side_orders(
        open_orders,
        symbol="ETHUSDT",
        side="BUY",
        execution_mode="margin",
        dry_run=True,
        desired_notional=500.0,
    )

    assert skip is False
    assert remaining == []
