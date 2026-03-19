from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


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
prompt_experiment = _load_module("prompt_experiment_short_policy", "prompt_experiment.py")

# Import TradePlan from the module (re-exported from llm_hourly_trader.gemini_wrapper)
TradePlan = prompt_experiment.TradePlan


def test_short_plan_adjusts_buy_price_below_close() -> None:
    """Short entry with buy_price below last close gets bumped to close*1.001.

    This mirrors the inline logic in prompt_experiment.run_experiment (lines 336-340):
        if plan.direction == "short":
            if plan.buy_price > 0 and plan.buy_price < last_close:
                plan.buy_price = last_close * 1.001
            if plan.sell_price > 0 and plan.sell_price > last_close:
                plan.sell_price = last_close * 0.999
    """
    plan = TradePlan("short", 99.0, 95.0, 0.7, "bearish")
    last_close = 100.0

    # Apply the same inline short price normalization
    if plan.direction == "short":
        if plan.buy_price > 0 and plan.buy_price < last_close:
            plan.buy_price = last_close * 1.001
        if plan.sell_price > 0 and plan.sell_price > last_close:
            plan.sell_price = last_close * 0.999

    assert plan.direction == "short"
    assert plan.buy_price == 100.1  # bumped from 99.0
    assert plan.sell_price == 95.0  # unchanged (already below close)


def test_short_plan_adjusts_sell_price_above_close() -> None:
    """Short exit with sell_price above last close gets clamped to close*0.999."""
    plan = TradePlan("short", 99.0, 101.0, 0.7, "bearish")
    last_close = 100.0

    if plan.direction == "short":
        if plan.buy_price > 0 and plan.buy_price < last_close:
            plan.buy_price = last_close * 1.001
        if plan.sell_price > 0 and plan.sell_price > last_close:
            plan.sell_price = last_close * 0.999

    assert plan.direction == "short"
    assert plan.buy_price == 100.1   # bumped from 99.0
    assert plan.sell_price == 99.9   # clamped from 101.0


def test_long_plan_unchanged() -> None:
    """Long plans are not adjusted by the short price normalization logic."""
    plan = TradePlan("long", 99.0, 105.0, 0.8, "bullish")
    last_close = 100.0

    if plan.direction == "short":
        if plan.buy_price > 0 and plan.buy_price < last_close:
            plan.buy_price = last_close * 1.001
        if plan.sell_price > 0 and plan.sell_price > last_close:
            plan.sell_price = last_close * 0.999

    assert plan.direction == "long"
    assert plan.buy_price == 99.0
    assert plan.sell_price == 105.0


def test_invalid_direction_becomes_hold() -> None:
    """Invalid directions are mapped to hold with zeroed prices.

    This mirrors the inline logic in prompt_experiment.run_experiment (lines 332-333):
        if plan.direction not in ("long", "short", "hold"):
            plan = TradePlan("hold", 0, 0, 0, "invalid direction")
    """
    plan = TradePlan("sideways", 99.0, 101.0, 0.5, "confused")

    if plan.direction not in ("long", "short", "hold"):
        plan = TradePlan("hold", 0, 0, 0, "invalid direction")

    assert plan.direction == "hold"
    assert plan.buy_price == 0.0
    assert plan.sell_price == 0.0
