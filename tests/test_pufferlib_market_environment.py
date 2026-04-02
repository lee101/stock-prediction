from __future__ import annotations

import importlib
import sys
import types


def test_trading_env_forwards_fill_params(monkeypatch) -> None:
    captured: dict[str, object] = {}

    fake_emulation = types.ModuleType("pufferlib.emulation")

    class FakePufferEnv:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_emulation.GymnasiumPufferEnv = FakePufferEnv
    monkeypatch.setitem(sys.modules, "gymnasium", types.ModuleType("gymnasium"))
    monkeypatch.setitem(sys.modules, "pufferlib.emulation", fake_emulation)

    import pufferlib_market.environment as environment

    environment = importlib.reload(environment)
    env = environment.TradingEnv(
        environment.TradingEnvConfig(
            fill_slippage_bps=7.5,
            fill_probability=0.8,
            num_symbols=3,
        )
    )

    assert env is not None
    assert captured["fill_slippage_bps"] == 7.5
    assert captured["fill_probability"] == 0.8
