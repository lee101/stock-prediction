"""Tests for pufferlib portfolio-level model support in rl_signal.py."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rl_trading_agent_binance"))

import rl_signal as rl_signal_module
from rl_signal import (
    ACTION_NAMES,
    INITIAL_CASH,
    MIXED23_SYMBOLS,
    SYMBOLS,
    PortfolioSnapshot,
    RLSignal,
    RLSignalGenerator,
    TradingPolicy,
    _build_action_names,
    _has_obs_norm,
    _infer_hidden_size,
    _infer_num_actions,
    _infer_num_symbols,
    _infer_obs_size,
    _infer_symbols,
)


def test_legacy_constants():
    assert SYMBOLS == ("BTCUSD", "ETHUSD", "DOGEUSD", "AAVEUSD")
    assert len(ACTION_NAMES) == 9
    assert ACTION_NAMES[0] == "FLAT"


def test_mixed23_symbols():
    assert len(MIXED23_SYMBOLS) == 23
    assert MIXED23_SYMBOLS[0] == "AAPL"
    assert MIXED23_SYMBOLS[-1] == "XRPUSD"
    assert "BTCUSD" in MIXED23_SYMBOLS
    assert "ETHUSD" in MIXED23_SYMBOLS


def test_infer_num_symbols():
    assert _infer_num_symbols(73) == 4
    assert _infer_num_symbols(396) == 23


def test_infer_symbols():
    assert _infer_symbols(73) == SYMBOLS
    assert _infer_symbols(396) == MIXED23_SYMBOLS
    unknown = _infer_symbols(192)
    assert len(unknown) == 11
    assert unknown[0] == "SYM0"


def test_build_action_names():
    names = _build_action_names(("BTCUSD", "ETHUSD"))
    assert names == ["FLAT", "LONG_BTC", "LONG_ETH", "SHORT_BTC", "SHORT_ETH"]


def _make_checkpoint(obs_size, num_actions, hidden=256, with_obs_norm=False):
    policy = TradingPolicy(obs_size, num_actions, hidden=hidden, use_obs_norm=with_obs_norm)
    return {
        "model": policy.state_dict(),
        "update": 100,
        "global_step": 50000,
        "action_allocation_bins": 1,
        "action_level_bins": 1,
        "action_max_offset_bps": 0.0,
        "disable_shorts": False,
    }


def test_infer_from_state_dict_4sym():
    ckpt = _make_checkpoint(73, 9, hidden=1024)
    sd = ckpt["model"]
    assert _infer_obs_size(sd) == 73
    assert _infer_num_actions(sd) == 9
    assert _infer_hidden_size(sd) == 1024
    assert not _has_obs_norm(sd)


def test_infer_from_state_dict_23sym():
    ckpt = _make_checkpoint(396, 47, hidden=1024)
    sd = ckpt["model"]
    assert _infer_obs_size(sd) == 396
    assert _infer_num_actions(sd) == 47
    assert _infer_hidden_size(sd) == 1024


def test_infer_from_state_dict_with_obs_norm():
    ckpt = _make_checkpoint(73, 9, hidden=512, with_obs_norm=True)
    sd = ckpt["model"]
    assert _has_obs_norm(sd)
    assert _infer_obs_size(sd) == 73


def test_policy_forward_no_obs_norm():
    policy = TradingPolicy(396, 47, hidden=256, use_obs_norm=False)
    x = torch.randn(2, 396)
    logits, value = policy(x)
    assert logits.shape == (2, 47)
    assert value.shape == (2,)


def test_policy_forward_with_obs_norm():
    policy = TradingPolicy(73, 9, hidden=256, use_obs_norm=True)
    x = torch.randn(2, 73)
    logits, value = policy(x)
    assert logits.shape == (2, 9)
    assert value.shape == (2,)


def test_generator_load_4sym(tmp_path):
    ckpt = _make_checkpoint(73, 9, hidden=256, with_obs_norm=True)
    path = tmp_path / "test_4sym.pt"
    torch.save(ckpt, str(path))

    gen = RLSignalGenerator(path, forecast_cache_root=str(tmp_path / "fc"))
    assert gen.num_symbols == 4
    assert gen.obs_size == 73
    assert gen.num_actions == 9
    assert gen.symbols == SYMBOLS


def test_generator_load_23sym(tmp_path):
    ckpt = _make_checkpoint(396, 47, hidden=1024)
    path = tmp_path / "test_23sym.pt"
    torch.save(ckpt, str(path))

    gen = RLSignalGenerator(path, forecast_cache_root=str(tmp_path / "fc"))
    assert gen.num_symbols == 23
    assert gen.obs_size == 396
    assert gen.num_actions == 47
    assert gen.symbols == MIXED23_SYMBOLS


def test_generator_falls_back_to_cpu_on_auto_cuda_oom(tmp_path, monkeypatch):
    ckpt = _make_checkpoint(73, 9, hidden=256)
    path = tmp_path / "test_auto_fallback.pt"
    torch.save(ckpt, str(path))

    original_to = TradingPolicy.to
    calls: list[str] = []

    def flaky_to(self, device, *args, **kwargs):
        resolved = torch.device(device)
        calls.append(resolved.type)
        if resolved.type == "cuda":
            raise torch.OutOfMemoryError("CUDA out of memory")
        return original_to(self, device, *args, **kwargs)

    monkeypatch.setattr(rl_signal_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(TradingPolicy, "to", flaky_to)

    gen = RLSignalGenerator(path, forecast_cache_root=str(tmp_path / "fc"))

    assert calls[:2] == ["cuda", "cpu"]
    assert gen.device.type == "cpu"


def test_generator_get_signal_4sym(tmp_path):
    ckpt = _make_checkpoint(73, 9, hidden=256, with_obs_norm=True)
    path = tmp_path / "test.pt"
    torch.save(ckpt, str(path))

    gen = RLSignalGenerator(path, forecast_cache_root=str(tmp_path / "fc"))
    portfolio = PortfolioSnapshot(cash_usd=10000, total_value_usd=10000)
    sig = gen.get_signal(portfolio=portfolio, klines_map={})
    assert isinstance(sig, RLSignal)
    assert sig.direction in ("long", "flat")
    assert len(sig.logits) == 9


def test_generator_get_signal_23sym(tmp_path):
    ckpt = _make_checkpoint(396, 47, hidden=256)
    path = tmp_path / "test.pt"
    torch.save(ckpt, str(path))

    gen = RLSignalGenerator(path, forecast_cache_root=str(tmp_path / "fc"))
    portfolio = PortfolioSnapshot(cash_usd=10000, total_value_usd=10000)
    sig = gen.get_signal(portfolio=portfolio, klines_map={})
    assert isinstance(sig, RLSignal)
    assert sig.direction in ("long", "flat")
    assert len(sig.logits) == 47


def test_decode_action_4sym(tmp_path):
    ckpt = _make_checkpoint(73, 9, hidden=256)
    path = tmp_path / "test.pt"
    torch.save(ckpt, str(path))
    gen = RLSignalGenerator(path, forecast_cache_root=str(tmp_path / "fc"))

    assert gen._decode_action(0) == (None, "flat")
    assert gen._decode_action(1) == ("BTCUSD", "long")
    assert gen._decode_action(4) == ("AAVEUSD", "long")
    assert gen._decode_action(5) == ("BTCUSD", "short")
    assert gen._decode_action(8) == ("AAVEUSD", "short")


def test_decode_action_23sym(tmp_path):
    ckpt = _make_checkpoint(396, 47, hidden=256)
    path = tmp_path / "test.pt"
    torch.save(ckpt, str(path))
    gen = RLSignalGenerator(path, forecast_cache_root=str(tmp_path / "fc"))

    assert gen._decode_action(0) == (None, "flat")
    assert gen._decode_action(1) == ("AAPL", "long")
    assert gen._decode_action(12) == ("BTCUSD", "long")
    assert gen._decode_action(23) == ("XRPUSD", "long")
    assert gen._decode_action(24) == ("AAPL", "short")
    assert gen._decode_action(46) == ("XRPUSD", "short")


def test_obs_construction_23sym(tmp_path):
    ckpt = _make_checkpoint(396, 47, hidden=256)
    path = tmp_path / "test.pt"
    torch.save(ckpt, str(path))
    gen = RLSignalGenerator(path, forecast_cache_root=str(tmp_path / "fc"))

    portfolio = PortfolioSnapshot(
        cash_usd=5000,
        total_value_usd=10000,
        position_symbol="BTCUSD",
        position_value_usd=5000,
        hold_hours=3,
    )
    features = np.zeros((23, 16), dtype=np.float32)
    features[11, 0] = 0.5  # BTCUSD chronos delta
    obs = gen._build_obs(features, portfolio)

    assert obs.shape == (396,)
    # Market features
    assert obs[11 * 16] == 0.5
    # Portfolio state
    base = 23 * 16
    assert obs[base + 0] == pytest.approx(5000 / INITIAL_CASH)
    assert obs[base + 1] == pytest.approx(5000 / INITIAL_CASH)
    # Position encoding: BTCUSD is at index 11
    assert obs[base + 5 + 11] == 1.0
    # Other position slots are 0
    assert obs[base + 5 + 0] == 0.0


def test_real_checkpoint_if_available():
    """Load real ent_anneal checkpoint if present."""
    path = Path("pufferlib_market/checkpoints/mixed23_fresh_replay/ent_anneal/best.pt")
    if not path.exists():
        pytest.skip("Real checkpoint not available")
    gen = RLSignalGenerator(path, forecast_cache_root="binanceneural/forecast_cache")
    assert gen.num_symbols == 23
    assert gen.obs_size == 396
    assert gen.num_actions == 47
    assert gen.symbols == MIXED23_SYMBOLS

    portfolio = PortfolioSnapshot(cash_usd=10000, total_value_usd=10000)
    sig = gen.get_signal(portfolio=portfolio, klines_map={})
    assert isinstance(sig, RLSignal)
    assert len(sig.logits) == 47
