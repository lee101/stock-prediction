"""Guard-kwargs wiring tests for the C trading env.

These verify that TradingEnvConfig forwards the prod-parity death-spiral
guard settings through to the C binding so new seeds can be trained with
guard-on dynamics (mirrors alpaca_singleton.guard_sell_against_death_spiral).
"""
from __future__ import annotations

import importlib
import sys
import types


def _fake_puffer_env(captured):
    fake_emulation = types.ModuleType("pufferlib.emulation")

    class FakePufferEnv:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_emulation.GymnasiumPufferEnv = FakePufferEnv
    return fake_emulation


def test_env_forwards_death_spiral_kwargs(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setitem(sys.modules, "gymnasium", types.ModuleType("gymnasium"))
    monkeypatch.setitem(sys.modules, "pufferlib.emulation", _fake_puffer_env(captured))

    import pufferlib_market.environment as environment
    environment = importlib.reload(environment)

    environment.TradingEnv(
        environment.TradingEnvConfig(
            num_symbols=3,
            death_spiral_tolerance_bps=50.0,
            death_spiral_overnight_tolerance_bps=500.0,
            death_spiral_stale_after_bars=8,
        )
    )

    assert captured["death_spiral_tolerance_bps"] == 50.0
    assert captured["death_spiral_overnight_tolerance_bps"] == 500.0
    assert captured["death_spiral_stale_after_bars"] == 8


def test_env_guard_defaults_off(monkeypatch) -> None:
    """Default tolerance_bps == 0.0 disables the guard (matches C env default)."""
    captured: dict[str, object] = {}
    monkeypatch.setitem(sys.modules, "gymnasium", types.ModuleType("gymnasium"))
    monkeypatch.setitem(sys.modules, "pufferlib.emulation", _fake_puffer_env(captured))

    import pufferlib_market.environment as environment
    environment = importlib.reload(environment)

    environment.TradingEnv(environment.TradingEnvConfig(num_symbols=3))

    assert captured["death_spiral_tolerance_bps"] == 0.0
    assert captured["death_spiral_overnight_tolerance_bps"] == 500.0
    assert captured["death_spiral_stale_after_bars"] == 8


def test_env_stale_bars_clamped_to_zero(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setitem(sys.modules, "gymnasium", types.ModuleType("gymnasium"))
    monkeypatch.setitem(sys.modules, "pufferlib.emulation", _fake_puffer_env(captured))

    import pufferlib_market.environment as environment
    environment = importlib.reload(environment)

    environment.TradingEnv(
        environment.TradingEnvConfig(
            num_symbols=3,
            death_spiral_stale_after_bars=-5,
        )
    )
    assert captured["death_spiral_stale_after_bars"] == 0
