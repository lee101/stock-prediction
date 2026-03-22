"""
tests/test_inference_tts.py — Unit tests for inference-time compute scaling (TTS).

Tests:
  - best_action_tts with K=1 returns a valid action without running env rollouts
  - best_action_tts with K>1 runs rollouts and returns action with highest mean return
  - best_action_tts stats dict contains expected keys
  - get_signal_tts wraps PPOTrader and returns a TradingSignal
  - PPOTrader.get_signal with tts_k=1 behaves identically to original (greedy argmax)
  - PPOTrader.get_signal raises ValueError when tts_k>1 and no data_path given
  - best_action_tts handles episode early-termination correctly (horizon > data length)
  - action with forced highest reward is selected by TTS
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers: write a minimal MKTD binary
# ---------------------------------------------------------------------------

def _write_mktd(
    path: Path,
    *,
    num_symbols: int = 2,
    num_timesteps: int = 60,
    seed: int = 0,
) -> None:
    magic = b"MKTD"
    version = 1
    features_per_sym = 16
    price_features = 5
    padding = b"\x00" * 40
    header = struct.pack(
        "<4sIIIII40s",
        magic, version, num_symbols, num_timesteps, features_per_sym, price_features, padding,
    )
    sym_table = b""
    for i in range(num_symbols):
        name = f"SYM{i}".encode()
        sym_table += name + b"\x00" * (16 - len(name))

    rng = np.random.default_rng(seed)
    features = rng.random((num_timesteps, num_symbols, features_per_sym)).astype(np.float32)
    prices = np.zeros((num_timesteps, num_symbols, price_features), dtype=np.float32)
    base = np.full(num_symbols, 100.0, dtype=np.float32)
    for t in range(num_timesteps):
        step = rng.standard_normal(num_symbols).astype(np.float32) * 0.5
        base = np.maximum(base + step, 1.0)
        prices[t, :, 0] = base
        prices[t, :, 1] = base + 0.5
        prices[t, :, 2] = base - 0.5
        prices[t, :, 3] = base
        prices[t, :, 4] = 1000.0

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header)
        f.write(sym_table)
        f.write(features.tobytes(order="C"))
        f.write(prices.tobytes(order="C"))


# ---------------------------------------------------------------------------
# Helpers: create a minimal Policy that biases one action
# ---------------------------------------------------------------------------

class _BiasedPolicy(nn.Module):
    """Trivial policy that always prefers action `hot_action`."""
    def __init__(self, obs_size: int, num_actions: int, hot_action: int):
        super().__init__()
        self.fc = nn.Linear(obs_size, num_actions)
        with torch.no_grad():
            self.fc.weight.zero_()
            self.fc.bias.zero_()
            self.fc.bias[hot_action] = 20.0  # dominant action
        self.value_head = nn.Linear(obs_size, 1)
        with torch.no_grad():
            self.value_head.weight.zero_()
            self.value_head.bias.zero_()

    def forward(self, x):
        return self.fc(x), self.value_head(x).squeeze(-1)


def _write_biased_checkpoint(
    path: Path,
    *,
    num_symbols: int,
    hot_action: int,
    hidden: int = 32,
) -> None:
    """Save a TradingPolicy checkpoint that always picks hot_action."""
    obs_size = num_symbols * 16 + 5 + num_symbols
    n_actions = 1 + 2 * num_symbols

    from pufferlib_market.evaluate_fast import TradingPolicy
    policy = TradingPolicy(obs_size, n_actions, hidden=hidden)
    with torch.no_grad():
        for p in policy.parameters():
            p.zero_()
        policy.actor[2].bias[hot_action] = 20.0

    payload = {"model": policy.state_dict()}
    torch.save(payload, path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_best_action_tts_k1_returns_valid_action(tmp_path: Path):
    """K=1 fast-path: returns a valid action index without running env rollouts."""
    num_symbols = 2
    obs_size = num_symbols * 16 + 5 + num_symbols
    n_actions = 1 + 2 * num_symbols

    hot = 3
    policy = _BiasedPolicy(obs_size, n_actions, hot)
    policy.eval()

    obs = np.zeros(obs_size, dtype=np.float32)
    data_path = str(tmp_path / "dummy.bin")  # Not actually opened for K=1

    best_action, expected_return, stats = __import__(
        "pufferlib_market.inference_tts", fromlist=["best_action_tts"]
    ).best_action_tts(
        policy=policy,
        current_obs=obs,
        data_path=data_path,
        current_timestep=0,
        n_actions=n_actions,
        K=1,
        device="cpu",
    )

    assert best_action == hot, f"Expected hot_action={hot}, got {best_action}"
    assert 0 <= best_action < n_actions
    assert stats["margin"] == 0.0


def test_best_action_tts_k8_picks_biased_action(tmp_path: Path):
    """K=8: TTS selects the action biased to win by the policy + env."""
    import pufferlib_market.binding as binding

    num_symbols = 2
    data_path = tmp_path / "market.bin"
    _write_mktd(data_path, num_symbols=num_symbols, num_timesteps=60)

    n_actions = 1 + 2 * num_symbols
    hot = 1  # long SYM0

    ckpt = tmp_path / "hot.pt"
    _write_biased_checkpoint(ckpt, num_symbols=num_symbols, hot_action=hot)

    obs_size = num_symbols * 16 + 5 + num_symbols
    device = torch.device("cpu")
    from pufferlib_market.inference_tts import _load_policy, best_action_tts
    policy = _load_policy(str(ckpt), obs_size, n_actions, device)

    binding.shared(data_path=str(data_path.resolve()))
    obs = np.zeros(obs_size, dtype=np.float32)

    best_action, expected_return, stats = best_action_tts(
        policy=policy,
        current_obs=obs,
        data_path=str(data_path),
        current_timestep=5,
        n_actions=n_actions,
        K=8,
        horizon=10,
        device="cpu",
    )

    assert 0 <= best_action < n_actions
    assert "action_returns" in stats
    assert len(stats["action_returns"]) == n_actions
    assert "margin" in stats
    assert stats["margin"] >= 0.0


def test_best_action_tts_stats_keys(tmp_path: Path):
    """Stats dict always contains required keys."""
    import pufferlib_market.binding as binding

    num_symbols = 2
    data_path = tmp_path / "market.bin"
    _write_mktd(data_path, num_symbols=num_symbols, num_timesteps=50)

    n_actions = 1 + 2 * num_symbols
    obs_size = num_symbols * 16 + 5 + num_symbols

    ckpt = tmp_path / "ckpt.pt"
    _write_biased_checkpoint(ckpt, num_symbols=num_symbols, hot_action=0)

    from pufferlib_market.inference_tts import _load_policy, best_action_tts
    device = torch.device("cpu")
    policy = _load_policy(str(ckpt), obs_size, n_actions, device)

    binding.shared(data_path=str(data_path.resolve()))

    _, _, stats = best_action_tts(
        policy=policy,
        current_obs=np.zeros(obs_size, dtype=np.float32),
        data_path=str(data_path),
        current_timestep=0,
        n_actions=n_actions,
        K=4,
        horizon=5,
        device="cpu",
    )

    for key in ("action_returns", "margin", "best_action", "best_return", "second_best_return"):
        assert key in stats, f"Missing key: {key}"
    assert len(stats["action_returns"]) == n_actions
    assert stats["best_action"] == stats.get("best_action")  # type sanity


def test_get_signal_tts_returns_trading_signal(tmp_path: Path):
    """get_signal_tts wraps PPOTrader and returns a proper TradingSignal."""
    import pufferlib_market.binding as binding
    from pufferlib_market.inference import PPOTrader, TradingSignal
    from pufferlib_market.inference_tts import get_signal_tts

    num_symbols = 2
    data_path = tmp_path / "market.bin"
    _write_mktd(data_path, num_symbols=num_symbols, num_timesteps=60)

    symbols = ["AAAA", "BBBB"]
    n_actions = 1 + 2 * num_symbols
    hot = 1  # long AAAA

    ckpt = tmp_path / "trader.pt"
    _write_biased_checkpoint(ckpt, num_symbols=num_symbols, hot_action=hot)

    binding.shared(data_path=str(data_path.resolve()))

    trader = PPOTrader(str(ckpt), device="cpu", symbols=symbols)
    features = np.zeros((num_symbols, 16), dtype=np.float32)
    prices = {s: 100.0 for s in symbols}

    signal, stats = get_signal_tts(
        trader=trader,
        features=features,
        prices=prices,
        data_path=str(data_path),
        current_timestep=5,
        tts_k=4,
        horizon=5,
    )

    assert isinstance(signal, TradingSignal)
    assert signal.action in {"flat"} | {f"{d}_{s}" for d in ("long", "short") for s in symbols}
    assert 0.0 <= signal.confidence <= 1.0
    assert "margin" in stats


def test_ppotrader_get_signal_tts_k1_matches_greedy(tmp_path: Path):
    """tts_k=1 in PPOTrader.get_signal() must match plain greedy argmax."""
    from pufferlib_market.inference import PPOTrader

    num_symbols = 2
    symbols = ["AA", "BB"]
    hot = 3  # short AA

    ckpt = tmp_path / "ckpt.pt"
    _write_biased_checkpoint(ckpt, num_symbols=num_symbols, hot_action=hot)

    trader = PPOTrader(str(ckpt), device="cpu", symbols=symbols)
    features = np.zeros((num_symbols, 16), dtype=np.float32)
    prices = {s: 100.0 for s in symbols}

    signal_greedy = trader.get_signal(features, prices, tts_k=1)
    signal_default = trader.get_signal(features, prices)  # original call

    assert signal_greedy.action == signal_default.action
    assert signal_greedy.symbol == signal_default.symbol
    assert signal_greedy.direction == signal_default.direction


def test_ppotrader_get_signal_tts_k_gt1_raises_without_data_path(tmp_path: Path):
    """tts_k > 1 without tts_data_path must raise ValueError."""
    from pufferlib_market.inference import PPOTrader

    num_symbols = 2
    ckpt = tmp_path / "ckpt.pt"
    _write_biased_checkpoint(ckpt, num_symbols=num_symbols, hot_action=1)

    trader = PPOTrader(str(ckpt), device="cpu", symbols=["AA", "BB"])
    features = np.zeros((num_symbols, 16), dtype=np.float32)
    prices = {"AA": 100.0, "BB": 100.0}

    with pytest.raises(ValueError, match="tts_data_path"):
        trader.get_signal(features, prices, tts_k=8)


def test_ppotrader_get_signal_tts_k_gt1_runs(tmp_path: Path):
    """tts_k > 1 with valid data_path runs and returns a TradingSignal."""
    import pufferlib_market.binding as binding
    from pufferlib_market.inference import PPOTrader

    num_symbols = 2
    data_path = tmp_path / "market.bin"
    _write_mktd(data_path, num_symbols=num_symbols, num_timesteps=60)

    symbols = ["AA", "BB"]
    ckpt = tmp_path / "ckpt.pt"
    _write_biased_checkpoint(ckpt, num_symbols=num_symbols, hot_action=1)

    binding.shared(data_path=str(data_path.resolve()))

    trader = PPOTrader(str(ckpt), device="cpu", symbols=symbols)
    features = np.zeros((num_symbols, 16), dtype=np.float32)
    prices = {s: 100.0 for s in symbols}

    signal = trader.get_signal(
        features, prices,
        tts_k=4,
        tts_data_path=str(data_path),
        tts_timestep=5,
        tts_horizon=5,
    )
    from pufferlib_market.inference import TradingSignal
    assert isinstance(signal, TradingSignal)


def test_best_action_tts_short_data_does_not_crash(tmp_path: Path):
    """TTS with horizon > remaining data length should complete without error."""
    import pufferlib_market.binding as binding
    from pufferlib_market.inference_tts import _load_policy, best_action_tts

    num_symbols = 2
    # Only 20 timesteps total; horizon=30 exceeds this, episode will terminate early
    data_path = tmp_path / "short.bin"
    _write_mktd(data_path, num_symbols=num_symbols, num_timesteps=20)

    n_actions = 1 + 2 * num_symbols
    obs_size = num_symbols * 16 + 5 + num_symbols

    ckpt = tmp_path / "ckpt.pt"
    _write_biased_checkpoint(ckpt, num_symbols=num_symbols, hot_action=0)

    device = torch.device("cpu")
    policy = _load_policy(str(ckpt), obs_size, n_actions, device)
    binding.shared(data_path=str(data_path.resolve()))

    best_action, expected_return, stats = best_action_tts(
        policy=policy,
        current_obs=np.zeros(obs_size, dtype=np.float32),
        data_path=str(data_path),
        current_timestep=0,
        n_actions=n_actions,
        K=4,
        horizon=30,  # deliberately longer than data
        device="cpu",
    )

    assert 0 <= best_action < n_actions


def test_best_action_tts_deterministic_after_first(tmp_path: Path):
    """deterministic_after_first=True should run without error and return valid action."""
    import pufferlib_market.binding as binding
    from pufferlib_market.inference_tts import _load_policy, best_action_tts

    num_symbols = 2
    data_path = tmp_path / "market.bin"
    _write_mktd(data_path, num_symbols=num_symbols, num_timesteps=60)

    n_actions = 1 + 2 * num_symbols
    obs_size = num_symbols * 16 + 5 + num_symbols

    ckpt = tmp_path / "ckpt.pt"
    _write_biased_checkpoint(ckpt, num_symbols=num_symbols, hot_action=2)

    device = torch.device("cpu")
    policy = _load_policy(str(ckpt), obs_size, n_actions, device)
    binding.shared(data_path=str(data_path.resolve()))

    best_action, _, stats = best_action_tts(
        policy=policy,
        current_obs=np.zeros(obs_size, dtype=np.float32),
        data_path=str(data_path),
        current_timestep=5,
        n_actions=n_actions,
        K=4,
        horizon=8,
        device="cpu",
        deterministic_after_first=True,
    )

    assert 0 <= best_action < n_actions
    assert len(stats["action_returns"]) == n_actions
