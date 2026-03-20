"""Tests for RL+Gemini hybrid bridge."""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from pufferlib_market.inference import Policy

from llm_hourly_trader.gemini_wrapper import TradePlan

from unified_orchestrator.rl_gemini_bridge import (
    RLGeminiBridge,
    RLSignal,
    build_portfolio_observation,
    decode_rl_action,
    build_hybrid_prompt,
    _softmax,
    _rl_only_plan,
    _hold_plan,
    _tag_plan_source,
    _SOURCE_GEMINI_RL,
    _SOURCE_RL_ONLY,
    _SOURCE_FALLBACK_HOLD,
    _RATE_LIMIT_BACKOFF_S,
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


def test_decode_level_bins_preserves_price_offset():
    logits = np.zeros(13)
    logits[6] = 5.0  # symbol 0, alloc_idx=1, level_idx=2 when alloc_bins=2 level_bins=3
    signals = decode_rl_action(
        logits,
        num_symbols=1,
        symbol_names=["BTC"],
        alloc_bins=2,
        level_bins=3,
        max_offset_bps=50.0,
        top_k=1,
    )

    assert signals[0].direction == "long"
    assert signals[0].allocation_pct == 1.0
    assert signals[0].level_offset_bps == pytest.approx(50.0)


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


# ─── New reliability & fallback tests ────────────────────────────────


def _make_signal(direction: str = "long", confidence: float = 0.8) -> RLSignal:
    """Build a minimal RLSignal for testing."""
    return RLSignal(
        symbol_idx=0,
        symbol_name="BTCUSD",
        direction=direction,
        confidence=confidence,
        logit_gap=2.5,
        allocation_pct=0.5,
    )


def _make_checkpoint(tmp_path: Path) -> Path:
    """Create a minimal valid checkpoint file for testing."""
    ckpt = tmp_path / "best.pt"
    model = Policy(obs_size=39, num_actions=13, hidden=32, num_blocks=2)
    payload = {
        "model": model.state_dict(),
        "config": {"num_blocks": 2},
        "action_allocation_bins": 2,
        "action_level_bins": 3,
        "action_max_offset_bps": 0.0,
        "disable_shorts": False,
    }
    torch.save(payload, ckpt)
    return ckpt


# ── _rl_only_plan helper ──────────────────────────────────────────────

def test_rl_only_plan_long_prices():
    sig = _make_signal("long", 0.75)
    plan = _rl_only_plan(sig, 100.0)
    assert plan.direction == "long"
    assert plan.buy_price == pytest.approx(100.0 * 0.998)
    assert plan.sell_price == pytest.approx(100.0 * 1.010)
    assert plan.confidence == pytest.approx(0.75 * 0.8)
    assert f"[{_SOURCE_RL_ONLY}]" in plan.reasoning


def test_rl_only_plan_short_prices():
    sig = _make_signal("short", 0.6)
    plan = _rl_only_plan(sig, 200.0)
    assert plan.direction == "short"
    assert plan.buy_price == pytest.approx(200.0 * 0.990)
    assert plan.sell_price == pytest.approx(200.0 * 1.002)
    assert plan.confidence == pytest.approx(0.6 * 0.8)


def test_rl_only_plan_flat_zero_prices():
    sig = _make_signal("flat", 0.5)
    plan = _rl_only_plan(sig, 100.0)
    assert plan.direction == "flat"
    assert plan.buy_price == 0.0
    assert plan.sell_price == 0.0


def test_rl_only_plan_includes_reason():
    sig = _make_signal("long")
    plan = _rl_only_plan(sig, 50.0, reason="timeout")
    assert "timeout" in plan.reasoning
    assert f"[{_SOURCE_RL_ONLY}]" in plan.reasoning


def test_rl_only_plan_allocation_scaled():
    # allocation_pct is 0-1 in RLSignal; _rl_only_plan should scale to 0-100
    sig = _make_signal("long", 0.9)
    sig = RLSignal(
        symbol_idx=0, symbol_name="BTC",
        direction="long", confidence=0.9,
        logit_gap=1.0, allocation_pct=0.4,
    )
    plan = _rl_only_plan(sig, 100.0)
    assert plan.allocation_pct == pytest.approx(40.0)


# ── _hold_plan helper ─────────────────────────────────────────────────

def test_hold_plan_structure():
    plan = _hold_plan(reason="checkpoint missing")
    assert plan.direction == "hold"
    assert plan.buy_price == 0.0
    assert plan.sell_price == 0.0
    assert plan.confidence == 0.0
    assert f"[{_SOURCE_FALLBACK_HOLD}]" in plan.reasoning
    assert "checkpoint missing" in plan.reasoning


def test_hold_plan_no_reason():
    plan = _hold_plan()
    assert plan.direction == "hold"
    assert f"[{_SOURCE_FALLBACK_HOLD}]" in plan.reasoning


# ── _tag_plan_source helper ────────────────────────────────────────────

def test_tag_plan_source_prepends_tag():
    original = TradePlan(direction="long", buy_price=100.0, sell_price=101.0,
                         confidence=0.8, reasoning="original reason")
    tagged = _tag_plan_source(original, _SOURCE_GEMINI_RL)
    assert tagged.reasoning.startswith(f"[{_SOURCE_GEMINI_RL}]")
    assert "original reason" in tagged.reasoning
    # Other fields preserved
    assert tagged.direction == "long"
    assert tagged.buy_price == 100.0
    assert tagged.confidence == 0.8


def test_tag_plan_source_empty_reasoning():
    original = TradePlan(direction="hold", buy_price=0.0, sell_price=0.0,
                         confidence=0.0, reasoning="")
    tagged = _tag_plan_source(original, _SOURCE_RL_ONLY)
    assert tagged.reasoning == f"[{_SOURCE_RL_ONLY}]"


# ── checkpoint not found ──────────────────────────────────────────────

def test_load_checkpoint_payload_missing_file_raises():
    bridge = RLGeminiBridge(checkpoint_path="/nonexistent/path/best.pt")
    with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
        bridge._load_checkpoint_payload()


# ── obs_size mismatch raises clearly ─────────────────────────────────

def test_load_policy_obs_size_mismatch_raises_with_message(tmp_path: Path):
    ckpt = _make_checkpoint(tmp_path)
    bridge = RLGeminiBridge(checkpoint_path=str(ckpt))
    # checkpoint expects obs_size=39; pass a wrong value
    with pytest.raises(ValueError, match="obs_size mismatch"):
        bridge._load_policy(obs_size=50, num_actions=13)


def test_load_policy_num_actions_mismatch_raises_with_message(tmp_path: Path):
    ckpt = _make_checkpoint(tmp_path)
    bridge = RLGeminiBridge(checkpoint_path=str(ckpt))
    with pytest.raises(ValueError, match="num_actions mismatch"):
        bridge._load_policy(obs_size=39, num_actions=99)


# ── generate_plans: Gemini failure → RL-only fallback (not None) ──────

def test_generate_plans_gemini_failure_returns_rl_only(tmp_path: Path):
    """When Gemini API call raises an exception, plan must not be None — use RL-only."""
    ckpt = _make_checkpoint(tmp_path)
    bridge = RLGeminiBridge(checkpoint_path=str(ckpt))

    signal = _make_signal("long", 0.8)

    def fake_call_llm(*args, **kwargs):
        raise RuntimeError("Simulated Gemini API error")

    with patch("unified_orchestrator.rl_gemini_bridge.build_hybrid_prompt",
               return_value="dummy prompt"):
        plan = bridge._call_llm_with_fallback(
            sym="BTCUSD",
            signal=signal,
            price=50000.0,
            prompt="dummy prompt",
            model="gemini-test",
            thinking_level="HIGH",
            call_llm=fake_call_llm,
        )

    assert plan is not None
    assert plan.direction == "long"
    assert f"[{_SOURCE_RL_ONLY}]" in plan.reasoning
    assert plan.buy_price > 0
    assert plan.sell_price > plan.buy_price


def test_generate_plans_gemini_failure_never_returns_none(tmp_path: Path):
    """generate_plans must never return None for a valid non-flat signal."""
    ckpt = _make_checkpoint(tmp_path)
    bridge = RLGeminiBridge(checkpoint_path=str(ckpt))

    signal = _make_signal("long", 0.8)

    def always_fail(*args, **kwargs):
        raise ConnectionError("network down")

    with patch("llm_hourly_trader.providers.call_llm", side_effect=always_fail):
        plans = bridge.generate_plans(
            rl_signals=[signal],
            price_histories={"BTCUSD": []},
            current_prices={"BTCUSD": 50000.0},
            dry_run=False,
        )

    assert "BTCUSD" in plans
    plan = plans["BTCUSD"]
    assert plan is not None
    assert plan.direction in ("long", "short", "hold")


# ── generate_plans: rate limit → backoff + retry ──────────────────────

def test_generate_plans_rate_limit_backoff_and_retry(tmp_path: Path):
    """On 429, bridge should sleep then retry once with simplified prompt."""
    call_count = [0]
    sleep_times = []

    def fake_call_llm(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("429 Too Many Requests")
        return TradePlan(
            direction="long", buy_price=49800.0, sell_price=50500.0,
            confidence=0.75, reasoning="retry succeeded",
        )

    def fake_sleep(seconds):
        sleep_times.append(seconds)

    signal = _make_signal("long", 0.8)
    ckpt = _make_checkpoint(tmp_path)
    bridge = RLGeminiBridge(checkpoint_path=str(ckpt))

    with patch("unified_orchestrator.rl_gemini_bridge.time.sleep", side_effect=fake_sleep):
        plan = bridge._call_llm_with_fallback(
            sym="BTCUSD",
            signal=signal,
            price=50000.0,
            prompt="prompt text",
            model="gemini-test",
            thinking_level="HIGH",
            call_llm=fake_call_llm,
        )

    # Should have called LLM twice
    assert call_count[0] == 2
    # Should have slept for the backoff period
    assert len(sleep_times) == 1
    assert sleep_times[0] == _RATE_LIMIT_BACKOFF_S
    # Should have succeeded on retry
    assert plan.direction == "long"
    assert f"[{_SOURCE_GEMINI_RL}]" in plan.reasoning


def test_generate_plans_rate_limit_both_fail_falls_back_to_rl(tmp_path: Path):
    """If both attempts fail with 429, should fall back to RL-only (not None)."""
    call_count = [0]

    def always_rate_limited(*args, **kwargs):
        call_count[0] += 1
        raise RuntimeError("429 rate limit exceeded")

    signal = _make_signal("long", 0.7)
    ckpt = _make_checkpoint(tmp_path)
    bridge = RLGeminiBridge(checkpoint_path=str(ckpt))

    with patch("unified_orchestrator.rl_gemini_bridge.time.sleep"):
        plan = bridge._call_llm_with_fallback(
            sym="BTCUSD",
            signal=signal,
            price=50000.0,
            prompt="prompt text",
            model="gemini-test",
            thinking_level="HIGH",
            call_llm=always_rate_limited,
        )

    assert call_count[0] == 2  # tried twice
    assert plan is not None
    assert f"[{_SOURCE_RL_ONLY}]" in plan.reasoning
    assert plan.direction == "long"


# ── generate_plans: successful Gemini → tagged gemini_rl ─────────────

def test_generate_plans_success_tags_gemini_rl(tmp_path: Path):
    """When Gemini succeeds, plan reasoning should be tagged [gemini_rl]."""
    def fake_call_llm(*args, **kwargs):
        return TradePlan(
            direction="long", buy_price=49800.0, sell_price=50500.0,
            confidence=0.85, reasoning="LLM reasoning",
        )

    signal = _make_signal("long", 0.8)
    ckpt = _make_checkpoint(tmp_path)
    bridge = RLGeminiBridge(checkpoint_path=str(ckpt))

    plan = bridge._call_llm_with_fallback(
        sym="BTCUSD",
        signal=signal,
        price=50000.0,
        prompt="prompt text",
        model="gemini-test",
        thinking_level="HIGH",
        call_llm=fake_call_llm,
    )

    assert f"[{_SOURCE_GEMINI_RL}]" in plan.reasoning
    assert "LLM reasoning" in plan.reasoning


# ── generate_plans dry_run returns RL-only plans ──────────────────────

def test_generate_plans_dry_run_returns_rl_only_tagged(tmp_path: Path):
    """In dry_run mode, plans must be returned with rl_only source tag."""
    ckpt = _make_checkpoint(tmp_path)
    bridge = RLGeminiBridge(checkpoint_path=str(ckpt))

    signal = _make_signal("long", 0.8)
    plans = bridge.generate_plans(
        rl_signals=[signal],
        price_histories={"BTCUSD": []},
        current_prices={"BTCUSD": 50000.0},
        dry_run=True,
    )

    assert "BTCUSD" in plans
    plan = plans["BTCUSD"]
    assert plan.direction == "long"
    assert f"[{_SOURCE_RL_ONLY}]" in plan.reasoning
    assert plan.buy_price == pytest.approx(50000.0 * 0.998)
    assert plan.sell_price == pytest.approx(50000.0 * 1.010)


# ── generate_plans: flat signals skipped, missing price skipped ───────

def test_generate_plans_skips_flat_signals(tmp_path: Path):
    """Flat signals should not produce a plan entry."""
    ckpt = _make_checkpoint(tmp_path)
    bridge = RLGeminiBridge(checkpoint_path=str(ckpt))

    flat_signal = _make_signal("flat", 0.1)
    plans = bridge.generate_plans(
        rl_signals=[flat_signal],
        price_histories={"BTCUSD": []},
        current_prices={"BTCUSD": 50000.0},
        dry_run=True,
    )
    assert "BTCUSD" not in plans


def test_generate_plans_skips_symbols_without_price(tmp_path: Path):
    ckpt = _make_checkpoint(tmp_path)
    bridge = RLGeminiBridge(checkpoint_path=str(ckpt))

    signal = _make_signal("long", 0.8)
    plans = bridge.generate_plans(
        rl_signals=[signal],
        price_histories={},
        current_prices={},  # BTCUSD price missing
        dry_run=True,
    )
    assert "BTCUSD" not in plans
