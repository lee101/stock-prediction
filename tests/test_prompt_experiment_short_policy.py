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


def test_symbol_config_for_short_policy_allows_crypto_shorts_when_enabled() -> None:
    cfg = prompt_experiment._symbol_config_for_short_policy(
        "BTCUSD",
        prompt_experiment.SHORT_POLICY_ALLOW,
    )

    assert cfg.allowed_directions == ["long", "short"]


def test_symbol_config_for_short_policy_filters_shorts_when_disabled() -> None:
    cfg = prompt_experiment._symbol_config_for_short_policy(
        "BTCUSD",
        prompt_experiment.SHORT_POLICY_FILTER,
    )

    assert cfg.allowed_directions == ["long"]


def test_normalize_plan_for_short_policy_filters_short_signal() -> None:
    plan = prompt_experiment.TradePlan("short", 99.0, 95.0, 0.7, "bearish")

    normalized, filtered = prompt_experiment._normalize_plan_for_short_policy(
        plan,
        last_close=100.0,
        short_policy=prompt_experiment.SHORT_POLICY_FILTER,
    )

    assert filtered is True
    assert normalized.direction == "hold"
    assert normalized.buy_price == 0.0
    assert normalized.sell_price == 0.0


def test_normalize_plan_for_short_policy_adjusts_short_prices() -> None:
    plan = prompt_experiment.TradePlan("short", 99.0, 101.0, 0.7, "bearish")

    normalized, filtered = prompt_experiment._normalize_plan_for_short_policy(
        plan,
        last_close=100.0,
        short_policy=prompt_experiment.SHORT_POLICY_ALLOW,
    )

    assert filtered is False
    assert normalized.direction == "short"
    assert normalized.buy_price == 100.1
    assert normalized.sell_price == 99.9
