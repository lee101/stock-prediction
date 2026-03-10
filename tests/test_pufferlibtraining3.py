import math
import sys
import types

import gymnasium as gym
import numpy as np
import pytest
import torch

from pufferlibtraining3.envs.market_env import MarketEnv, MarketEnvConfig
from pufferlibtraining3 import pufferrl


def _build_prices() -> torch.Tensor:
    # Columns: open, high, low, close
    data = torch.tensor(
        [
            [100.0, 101.0, 99.0, 100.5],
            [101.0, 102.5, 100.0, 101.8],
            [102.0, 104.5, 101.5, 103.7],
            [103.0, 105.5, 102.0, 104.2],
            [104.0, 105.9, 103.2, 104.7],
            [105.0, 106.1, 104.4, 105.5],
        ],
        dtype=torch.float32,
    )
    return data


def _build_long_prices() -> torch.Tensor:
    base = _build_prices()
    extra = torch.tensor(
        [
            [106.0, 107.4, 105.1, 106.8],
            [107.0, 108.2, 106.4, 107.6],
            [108.0, 109.5, 107.0, 108.7],
            [109.0, 110.3, 108.1, 109.4],
        ],
        dtype=torch.float32,
    )
    return torch.cat([base, extra], dim=0)


def test_market_env_maxdiff_fills_only_when_limit_touched():
    prices = _build_prices()
    cfg = MarketEnvConfig(
        mode="maxdiff",
        context_len=3,
        horizon=1,
        trading_fee=0.0005,
        slip_bps=1.5,
        maxdiff_limit_scale=0.05,
        maxdiff_deadband=0.01,
        seed=123,
        device="cpu",
    )
    env = MarketEnv(prices=prices, price_columns=("open", "high", "low", "close"), cfg=cfg)
    env.reset()

    action = np.array([3.0, 0.1], dtype=np.float32)
    _, reward, _, _, info = env.step(action)

    assert info["maxdiff_filled"] is True
    limit_price = info["limit_price"]
    expected_limit = 103.0 * (1.0 + math.tanh(0.1) * cfg.maxdiff_limit_scale)
    assert limit_price == pytest.approx(expected_limit, rel=1e-5)

    size = math.tanh(3.0)
    gross_return = (104.2 - expected_limit) / expected_limit
    gross = size * gross_return
    fee_rate = cfg.trading_fee
    slip_rate = cfg.slip_bps / 10_000.0
    total_cost = size * 2.0 * (fee_rate + slip_rate)
    expected_reward = gross - total_cost
    assert reward == pytest.approx(expected_reward, rel=1e-5, abs=1e-6)


def test_market_env_maxdiff_no_fill_without_cross():
    prices = _build_prices()
    cfg = MarketEnvConfig(
        mode="maxdiff",
        context_len=3,
        horizon=1,
        trading_fee=0.0005,
        slip_bps=1.5,
        maxdiff_limit_scale=0.05,
        maxdiff_deadband=0.01,
        seed=321,
        device="cpu",
    )
    env = MarketEnv(prices=prices, price_columns=("open", "high", "low", "close"), cfg=cfg)
    env.reset()

    action = np.array([3.0, 1.0], dtype=np.float32)  # limit well above day's high
    _, reward, _, _, info = env.step(action)

    assert info["maxdiff_filled"] is False
    assert reward == pytest.approx(0.0, abs=1e-9)


def test_market_env_reset_respects_explicit_start_and_episode_length():
    prices = _build_long_prices()
    cfg = MarketEnvConfig(
        context_len=2,
        horizon=1,
        start_index=3,
        episode_length=2,
        random_reset=False,
        device="cpu",
    )
    env = MarketEnv(prices=prices, price_columns=("open", "high", "low", "close"), cfg=cfg)

    obs, info = env.reset()

    assert obs.shape == (2, prices.shape[1] + 3)
    assert info["start_index"] == 3
    assert info["episode_end"] == 5

    _, _, terminated, _, _ = env.step(np.array([0.0], dtype=np.float32))
    assert terminated is False
    _, _, terminated, _, _ = env.step(np.array([0.0], dtype=np.float32))
    assert terminated is True


def test_market_env_random_reset_samples_valid_episode_window():
    prices = _build_long_prices()
    cfg = MarketEnvConfig(
        context_len=2,
        horizon=1,
        episode_length=2,
        random_reset=True,
        seed=7,
        device="cpu",
    )
    env = MarketEnv(prices=prices, price_columns=("open", "high", "low", "close"), cfg=cfg)

    starts = {env.reset()[1]["start_index"] for _ in range(12)}

    assert starts
    assert min(starts) >= cfg.context_len
    assert max(starts) <= prices.shape[0] - max(1, cfg.horizon) - cfg.episode_length
    assert len(starts) > 1


def test_pufferrl_build_configs_maps_cli_arguments():
    args = pufferrl.parse_args(
        [
            "--data-root",
            "trainingdata",
            "--symbol",
            "AAPL",
            "--mode",
            "open_close",
            "--is-crypto",
            "false",
            "--device",
            "cpu",
            "--num-envs",
            "4",
        ]
    )
    env_cfg, ppo_cfg, vec_cfg, device = pufferrl.build_configs(args)

    assert env_cfg.symbol == "AAPL"
    assert env_cfg.mode == "open_close"
    assert env_cfg.is_crypto is False
    assert env_cfg.data_root == "trainingdata"
    assert vec_cfg.num_envs == 4
    assert device.type == "cpu"


def test_import_pufferlib_module_prefers_repo_override(monkeypatch):
    override = "/tmp/pufferlib4"
    marker = object()
    calls: list[tuple[str, str | None]] = []
    original_path = list(sys.path)

    def fake_import(name: str):
        calls.append((name, sys.path[0] if sys.path else None))
        if name == "pufferlib.vector":
            return marker
        raise ModuleNotFoundError(name)

    monkeypatch.setenv("PUFFERLIB_REPO_ROOT", override)
    monkeypatch.setattr("importlib.import_module", fake_import)
    monkeypatch.delitem(sys.modules, "pufferlib.vector", raising=False)
    monkeypatch.setattr(sys, "path", [entry for entry in original_path if entry != override])

    module = pufferrl._import_pufferlib_module("pufferlib.vector")

    assert module is marker
    assert calls[0] == ("pufferlib.vector", override)
    assert sys.path[0] == override


def test_import_pufferlib_module_falls_back_to_python_pufferl(monkeypatch):
    incompatible = types.SimpleNamespace()

    class IncompatibleTrainer:
        def __init__(self, config, logger=None, verbose=True):
            del config, logger, verbose

    compatible = types.SimpleNamespace()

    class CompatibleTrainer:
        def __init__(self, config, vecenv, policy, logger=None, verbose=True):
            del config, vecenv, policy, logger, verbose

    incompatible.PuffeRL = IncompatibleTrainer
    compatible.PuffeRL = CompatibleTrainer
    calls: list[str] = []

    def fake_import(name: str):
        calls.append(name)
        if name == "pufferlib.pufferl":
            return incompatible
        if name == "pufferlib.python_pufferl":
            return compatible
        raise ModuleNotFoundError(name)

    monkeypatch.delenv("PUFFERLIB_PUFFERL_MODULE", raising=False)
    monkeypatch.delenv("PUFFERLIB_REPO_ROOT", raising=False)
    monkeypatch.setattr("importlib.import_module", fake_import)

    module = pufferrl._import_pufferlib_module("pufferlib.pufferl")

    assert module is compatible
    assert calls == ["pufferlib.pufferl", "pufferlib.python_pufferl"]


def test_import_pufferlib_module_clears_stale_cached_package_for_override(monkeypatch):
    stale_pkg = types.ModuleType("pufferlib")
    stale_pkg.__file__ = "/tmp/site-packages/pufferlib/__init__.py"
    stale_submodule = types.ModuleType("pufferlib.vector")
    stale_submodule.__file__ = "/tmp/site-packages/pufferlib/vector.py"
    marker = object()
    original_path = list(sys.path)
    calls: list[tuple[str, bool, bool]] = []

    def fake_import(name: str):
        calls.append((name, "pufferlib" in sys.modules, "pufferlib.vector" in sys.modules))
        if name == "pufferlib.vector":
            return marker
        raise ModuleNotFoundError(name)

    monkeypatch.setenv("PUFFERLIB_REPO_ROOT", "/tmp/pufferlib4")
    monkeypatch.setattr("importlib.import_module", fake_import)
    monkeypatch.setattr(sys, "path", list(original_path))
    monkeypatch.setitem(sys.modules, "pufferlib", stale_pkg)
    monkeypatch.setitem(sys.modules, "pufferlib.vector", stale_submodule)

    module = pufferrl._import_pufferlib_module("pufferlib.vector")

    assert module is marker
    assert calls[0] == ("pufferlib.vector", False, False)


def test_import_pufferlib_module_honors_explicit_pufferl_override(monkeypatch):
    compatible = types.SimpleNamespace()

    class CompatibleTrainer:
        def __init__(self, config, vecenv, policy, logger=None, verbose=True):
            del config, vecenv, policy, logger, verbose

    compatible.PuffeRL = CompatibleTrainer
    calls: list[str] = []

    def fake_import(name: str):
        calls.append(name)
        if name == "pufferlib.python_pufferl":
            return compatible
        raise ModuleNotFoundError(name)

    monkeypatch.setenv("PUFFERLIB_PUFFERL_MODULE", "pufferlib.python_pufferl")
    monkeypatch.delenv("PUFFERLIB_REPO_ROOT", raising=False)
    monkeypatch.setattr("importlib.import_module", fake_import)

    module = pufferrl._import_pufferlib_module("pufferlib.pufferl")

    assert module is compatible
    assert calls[0] == "pufferlib.python_pufferl"


def test_resolve_pufferlib_repo_root_prefers_pufferlib4():
    root = pufferrl._resolve_pufferlib_repo_root()
    assert root.name == "PufferLib4"


def test_load_pufferlib_base_config_falls_back_to_default():
    calls: list[str] = []

    class StubPuffer:
        @staticmethod
        def load_config(name: str):
            calls.append(name)
            if name == "trade_sim":
                raise RuntimeError("missing config")
            return {"train": {"seed": 1}}

    cfg, loaded = pufferrl._load_pufferlib_base_config(StubPuffer(), "trade_sim")

    assert loaded == "default"
    assert cfg == {"train": {"seed": 1}}
    assert calls == ["trade_sim", "default"]


def test_build_policy_config_uses_large_model_preset():
    args = pufferrl.parse_args(["--model-preset", "100m"])
    cfg = pufferrl._build_policy_config(args)

    assert cfg.hidden_size == 2048
    assert tuple(cfg.actor_layers) == (4096, 4096, 4096, 2048)
    assert tuple(cfg.critic_layers) == (4096, 4096, 4096, 2048)


def test_build_env_creator_uses_vector_seed(monkeypatch):
    captured: dict[str, int] = {}

    class CapturingEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self, cfg):
            super().__init__()
            captured["seed"] = int(cfg.seed)
            self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):  # type: ignore[override]
            del seed, options
            return np.zeros((1,), dtype=np.float32), {}

        def step(self, action):  # type: ignore[override]
            del action
            return np.zeros((1,), dtype=np.float32), 0.0, False, False, {}

    class FakePufferEnv:
        def __init__(self, env_creator, buf=None):
            del buf
            self.inner = env_creator()

    monkeypatch.setattr(pufferrl, "MarketEnv", CapturingEnv)
    monkeypatch.setattr(
        pufferrl,
        "_import_pufferlib_module",
        lambda name: types.SimpleNamespace(GymnasiumPufferEnv=FakePufferEnv),
    )

    collectors: list[pufferrl.MetricsCollector] = []
    cfg = MarketEnvConfig(context_len=3, horizon=1, seed=7, device="cpu")
    env_creator = pufferrl._build_env_creator(cfg, collectors, backend="python")
    wrapper = env_creator(seed=2025)

    assert isinstance(wrapper.inner, pufferrl.MetricsCollector)
    assert captured["seed"] == 2025
    assert len(collectors) == 1


def test_market_env_random_reset_samples_episode_windows():
    prices = torch.tensor(
        [
            [100.0 + idx, 101.0 + idx, 99.0 + idx, 100.5 + idx]
            for idx in range(12)
        ],
        dtype=torch.float32,
    )
    cfg = MarketEnvConfig(
        context_len=3,
        horizon=1,
        episode_length=2,
        random_reset=True,
        seed=123,
        device="cpu",
    )
    env = MarketEnv(prices=prices, price_columns=("open", "high", "low", "close"), cfg=cfg)

    starts: list[int] = []
    for _ in range(6):
        _, info = env.reset()
        starts.append(int(info["start_index"]))
        assert int(info["episode_end"]) - int(info["start_index"]) == 2

    assert len(set(starts)) > 1


def test_market_env_reset_honors_explicit_start_index():
    prices = torch.tensor(
        [
            [100.0 + idx, 101.0 + idx, 99.0 + idx, 100.5 + idx]
            for idx in range(10)
        ],
        dtype=torch.float32,
    )
    cfg = MarketEnvConfig(
        context_len=3,
        horizon=1,
        episode_length=2,
        device="cpu",
    )
    env = MarketEnv(prices=prices, price_columns=("open", "high", "low", "close"), cfg=cfg)

    _, info = env.reset(options={"start_index": 5})

    assert info["start_index"] == 5
    assert info["episode_end"] == 7

    _, _, terminated, truncated, _ = env.step(np.array([0.0], dtype=np.float32))
    assert terminated is False
    assert truncated is False
    _, _, terminated, truncated, _ = env.step(np.array([0.0], dtype=np.float32))
    assert terminated is True
    assert truncated is False
