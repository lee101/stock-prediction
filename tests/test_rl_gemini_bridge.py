"""Tests for RL+Gemini hybrid bridge."""

from pathlib import Path

import numpy as np
import pytest
import torch

from pufferlib_market.inference import Policy

from unified_orchestrator.rl_gemini_bridge import (
    RLGeminiBridge,
    RLSignal,
    build_portfolio_observation,
    decode_rl_action,
    build_hybrid_prompt,
    _softmax,
)


def test_softmax():
    x = np.array([1.0, 2.0, 3.0])
    p = _softmax(x)
    assert abs(p.sum() - 1.0) < 1e-6
    assert p[2] > p[1] > p[0]


def test_decode_flat_logits():
    """When flat action has highest logit, all signals should be flat."""
    # 2 symbols, 1 alloc bin, 1 level bin -> num_actions = 1 + 2*2*1 = 5
    logits = np.array([10.0, -5.0, -5.0, -5.0, -5.0])  # flat dominates
    signals = decode_rl_action(
        logits, num_symbols=2, symbol_names=["BTC", "ETH"],
        alloc_bins=1, level_bins=1, top_k=2,
    )
    assert len(signals) == 2
    # All should be flat since flat logit dominates
    for s in signals:
        assert s.direction == "flat"


def test_decode_long_signal():
    """When a long action has highest logit, should detect long signal."""
    # 2 symbols: actions = [flat, BTC_long, ETH_long, BTC_short, ETH_short]
    logits = np.array([0.0, 10.0, -5.0, -5.0, -5.0])  # BTC long strong
    signals = decode_rl_action(
        logits, num_symbols=2, symbol_names=["BTC", "ETH"],
        alloc_bins=1, level_bins=1, top_k=2,
    )
    # BTC should be first with long direction
    btc_signal = next(s for s in signals if s.symbol_name == "BTC")
    assert btc_signal.direction == "long"
    assert btc_signal.confidence > 0.5
    assert btc_signal.logit_gap > 0


def test_decode_short_signal():
    """When a short action has highest logit, should detect short signal."""
    # 2 symbols: actions = [flat, BTC_long, ETH_long, BTC_short, ETH_short]
    logits = np.array([0.0, -5.0, -5.0, 10.0, -5.0])  # BTC short strong
    signals = decode_rl_action(
        logits, num_symbols=2, symbol_names=["BTC", "ETH"],
        alloc_bins=1, level_bins=1, top_k=2,
    )
    btc_signal = next(s for s in signals if s.symbol_name == "BTC")
    assert btc_signal.direction == "short"
    assert btc_signal.confidence > 0.5


def test_decode_multiple_alloc_bins():
    """Test with multiple allocation bins."""
    # 2 symbols, 2 alloc bins, 1 level bin
    # actions: flat, BTC_long_50%, BTC_long_100%, ETH_long_50%, ETH_long_100%,
    #          BTC_short_50%, BTC_short_100%, ETH_short_50%, ETH_short_100%
    logits = np.zeros(9)
    logits[2] = 5.0  # BTC long 100% alloc
    signals = decode_rl_action(
        logits, num_symbols=2, symbol_names=["BTC", "ETH"],
        alloc_bins=2, level_bins=1, top_k=2,
    )
    btc_signal = next(s for s in signals if s.symbol_name == "BTC")
    assert btc_signal.direction == "long"
    assert btc_signal.allocation_pct == 1.0  # 100% alloc


def test_build_hybrid_prompt():
    """Test prompt generation."""
    signal = RLSignal(
        symbol_idx=0, symbol_name="BTCUSD",
        direction="long", confidence=0.85,
        logit_gap=3.2, allocation_pct=1.0,
    )
    history = [
        {"timestamp": f"2026-03-12T{h:02d}:00:00Z",
         "open": 60000 + h * 10, "high": 60100 + h * 10,
         "low": 59900 + h * 10, "close": 60050 + h * 10, "volume": 1000}
        for h in range(24)
    ]
    prompt = build_hybrid_prompt(
        symbol="BTCUSD",
        rl_signal=signal,
        history_rows=history,
        current_price=60290.0,
    )
    assert "BTCUSD" in prompt
    assert "LONG" in prompt
    assert "85.0%" in prompt
    assert "60290" in prompt


def test_signals_sorted_by_confidence():
    """Signals should be sorted: non-flat first, then by confidence."""
    # BTC moderate long, ETH strong long
    logits = np.array([0.0, 3.0, 8.0, -5.0, -5.0])
    signals = decode_rl_action(
        logits, num_symbols=2, symbol_names=["BTC", "ETH"],
        alloc_bins=1, level_bins=1, top_k=2,
    )
    # ETH long (stronger) should come first
    non_flat = [s for s in signals if s.direction != "flat"]
    if len(non_flat) >= 2:
        assert non_flat[0].confidence >= non_flat[1].confidence


def test_build_portfolio_observation_encodes_position_state():
    features = np.zeros((2, 16), dtype=np.float32)
    obs = build_portfolio_observation(
        features,
        cash_ratio=0.8,
        position_value_ratio=-0.15,
        unrealized_pnl_ratio=0.03,
        hold_fraction=0.25,
        step_fraction=0.5,
        position_symbol_idx=1,
        position_direction="short",
    )

    assert obs.shape == (39,)
    base = 32
    assert obs[base + 0] == pytest.approx(0.8)
    assert obs[base + 1] == pytest.approx(-0.15)
    assert obs[base + 2] == pytest.approx(0.03)
    assert obs[base + 3] == pytest.approx(0.25)
    assert obs[base + 4] == pytest.approx(0.5)
    assert obs[base + 5] == pytest.approx(0.0)
    assert obs[base + 6] == pytest.approx(-1.0)


def test_bridge_infers_checkpoint_spec_from_residual_policy(tmp_path: Path):
    ckpt = tmp_path / "bridge.pt"
    model = Policy(obs_size=39, num_actions=13, hidden=32, num_blocks=2)
    payload = {
        "model": model.state_dict(),
        "config": {"num_blocks": 2},
        "action_allocation_bins": 2,
        "action_level_bins": 3,
        "action_max_offset_bps": 25.0,
        "disable_shorts": True,
    }
    torch.save(payload, ckpt)

    bridge = RLGeminiBridge(checkpoint_path=str(ckpt))
    spec = bridge.get_checkpoint_spec()

    assert spec.arch == "resmlp"
    assert spec.obs_size == 39
    assert spec.num_actions == 13
    assert spec.hidden_size == 32
    assert spec.num_blocks == 2
    assert spec.alloc_bins == 2
    assert spec.level_bins == 3
    assert spec.max_offset_bps == pytest.approx(25.0)
    assert spec.disable_shorts is True
