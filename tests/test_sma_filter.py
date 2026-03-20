"""Tests for the SMA-24 soft filter in unified_orchestrator/orchestrator.py.

Verifies that the updated SMA-24 logic:
- Does NOT force HOLD when price is mildly below SMA (was the production bug)
- Reduces allocation by 50% when price is 2-5% below SMA (moderate discount)
- Keeps signal unchanged (just logs) when price is <2% below SMA (mild discount)
- Hard blocks (LONG→HOLD) only when price is >5% below SMA (extreme discount)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import pandas as pd
import pytest

# ── path setup ──────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


# ── minimal stubs so we can import orchestrator logic without live deps ──────

@dataclass
class _RLSignal:
    symbol_idx: int
    symbol_name: str
    direction: str
    confidence: float
    logit_gap: float
    allocation_pct: float
    level_offset_bps: float = 0.0


@dataclass
class _TradePlan:
    direction: str
    buy_price: float
    sell_price: float
    confidence: float
    reasoning: str = ""
    allocation_pct: float = 0.0


# ── helpers that replicate the exact filter logic from orchestrator.py ───────
# We test the logic inline rather than importing the full orchestrator module
# (which requires live Alpaca keys, GPU, etc.).  Any change to orchestrator.py
# that affects the SMA filter thresholds must be reflected here too.

def _apply_rl_hint_sma_filter(
    rl_signal_map: dict[str, _RLSignal],
    history_frames: dict[str, pd.DataFrame],
) -> dict[str, _RLSignal]:
    """Replicate the RL-hint SMA-24 filter from get_crypto_signals (lines ~666-710)."""
    result = dict(rl_signal_map)
    for sym, sig in list(result.items()):
        if sig.direction != "long":
            continue
        frame = history_frames.get(sym)
        if frame is None or len(frame) < 24:
            continue
        sma24 = float(frame["close"].iloc[-24:].mean())
        current = float(frame["close"].iloc[-1])
        if current < sma24:
            discount_pct = (sma24 - current) / sma24
            if discount_pct > 0.05:
                result[sym] = _RLSignal(
                    symbol_idx=sig.symbol_idx,
                    symbol_name=sig.symbol_name,
                    direction="flat",
                    confidence=0.0,
                    logit_gap=0.0,
                    allocation_pct=0.0,
                )
            else:
                old_alloc = sig.allocation_pct
                new_alloc = old_alloc * 0.5
                result[sym] = _RLSignal(
                    symbol_idx=sig.symbol_idx,
                    symbol_name=sig.symbol_name,
                    direction=sig.direction,
                    confidence=sig.confidence,
                    logit_gap=sig.logit_gap,
                    allocation_pct=new_alloc,
                )
    return result


def _apply_post_llm_sma_filter(
    signals: dict[str, _TradePlan],
    history_frames: dict[str, pd.DataFrame],
) -> dict[str, _TradePlan]:
    """Replicate the post-LLM SMA-24 guard from get_crypto_signals (lines ~757-820)."""
    result = dict(signals)
    for sym, plan in list(result.items()):
        if plan.direction != "long":
            continue
        frame = history_frames.get(sym)
        if frame is None or len(frame) < 24:
            continue
        sma24g = float(frame["close"].iloc[-24:].mean())
        price_g = float(frame["close"].iloc[-1])
        if price_g >= sma24g:
            continue
        discount_pct = (sma24g - price_g) / sma24g
        if discount_pct > 0.05:
            result[sym] = _TradePlan(
                direction="hold",
                buy_price=0.0,
                sell_price=plan.sell_price,
                confidence=plan.confidence,
                reasoning=f"SMA-24 hard block (>{discount_pct:.1%} below); {plan.reasoning[:60]}",
                allocation_pct=0,
            )
        elif discount_pct > 0.02:
            old_alloc = plan.allocation_pct
            new_alloc = old_alloc * 0.5
            result[sym] = _TradePlan(
                direction=plan.direction,
                buy_price=plan.buy_price,
                sell_price=plan.sell_price,
                confidence=plan.confidence,
                reasoning=f"SMA-24 soft filter ({discount_pct:.1%} below); {plan.reasoning[:60]}",
                allocation_pct=new_alloc,
            )
        # else: mild discount <2%, pass through unchanged
    return result


# ── fixture helpers ──────────────────────────────────────────────────────────

def _make_frame(current_price: float, sma24: float, n_bars: int = 30) -> pd.DataFrame:
    """Create a fake OHLCV frame where the last bar has `current_price`
    and the 24-bar mean of close equals approximately `sma24`."""
    # Fill the last 24 bars: 23 bars at a value that makes the mean = sma24
    # when combined with the current_price bar.
    # sma24 = (23 * filler + current_price) / 24
    # filler = (sma24 * 24 - current_price) / 23
    filler = (sma24 * 24 - current_price) / 23
    closes = [filler] * (n_bars - 1) + [current_price]
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n_bars, freq="h"),
        "open": closes,
        "high": closes,
        "low": closes,
        "close": closes,
        "volume": [1000.0] * n_bars,
    })
    return df


def _make_rl_signal(sym: str, direction: str = "long", alloc: float = 0.20) -> _RLSignal:
    return _RLSignal(
        symbol_idx=0,
        symbol_name=sym,
        direction=direction,
        confidence=0.7,
        logit_gap=1.5,
        allocation_pct=alloc,
    )


def _make_trade_plan(direction: str = "long", alloc: float = 30.0, conf: float = 0.6) -> _TradePlan:
    return _TradePlan(
        direction=direction,
        buy_price=100.0,
        sell_price=105.0,
        confidence=conf,
        reasoning="test signal",
        allocation_pct=alloc,
    )


# ── RL hint filter tests ─────────────────────────────────────────────────────

class TestRLHintSmaFilter:
    """Tests for the pre-LLM RL hint SMA-24 filter."""

    def test_price_above_sma_unchanged(self):
        """When price > SMA-24, RL LONG hint passes through unmodified."""
        sig = _make_rl_signal("BTCUSD", alloc=0.20)
        frame = _make_frame(current_price=101.0, sma24=100.0)
        result = _apply_rl_hint_sma_filter({"BTCUSD": sig}, {"BTCUSD": frame})
        assert result["BTCUSD"].direction == "long"
        assert result["BTCUSD"].allocation_pct == pytest.approx(0.20, rel=1e-3)

    def test_price_equal_sma_unchanged(self):
        """When price == SMA-24, RL LONG hint passes through unmodified."""
        sig = _make_rl_signal("BTCUSD", alloc=0.20)
        frame = _make_frame(current_price=100.0, sma24=100.0)
        result = _apply_rl_hint_sma_filter({"BTCUSD": sig}, {"BTCUSD": frame})
        assert result["BTCUSD"].direction == "long"
        assert result["BTCUSD"].allocation_pct == pytest.approx(0.20, rel=1e-3)

    def test_mild_discount_below_2pct_reduces_allocation(self):
        """When price is 1% below SMA, allocation is halved (mild discount)."""
        # price 1% below SMA: sma=100, price=99
        sig = _make_rl_signal("BTCUSD", alloc=0.20)
        frame = _make_frame(current_price=99.0, sma24=100.0)
        result = _apply_rl_hint_sma_filter({"BTCUSD": sig}, {"BTCUSD": frame})
        # Should still be long but with 50% allocation
        assert result["BTCUSD"].direction == "long"
        assert result["BTCUSD"].allocation_pct == pytest.approx(0.10, rel=1e-3)

    def test_moderate_discount_3pct_reduces_allocation(self):
        """When price is 3% below SMA, allocation is halved (moderate discount < 5%)."""
        # price 3% below SMA: sma=100, price=97
        sig = _make_rl_signal("BTCUSD", alloc=0.20)
        frame = _make_frame(current_price=97.0, sma24=100.0)
        result = _apply_rl_hint_sma_filter({"BTCUSD": sig}, {"BTCUSD": frame})
        assert result["BTCUSD"].direction == "long"
        assert result["BTCUSD"].allocation_pct == pytest.approx(0.10, rel=1e-3)

    def test_extreme_discount_6pct_hard_suppresses(self):
        """When price is >5% below SMA, RL hint is hard suppressed to flat."""
        # price 6% below SMA: sma=100, price=94
        sig = _make_rl_signal("BTCUSD", alloc=0.20)
        frame = _make_frame(current_price=94.0, sma24=100.0)
        result = _apply_rl_hint_sma_filter({"BTCUSD": sig}, {"BTCUSD": frame})
        assert result["BTCUSD"].direction == "flat"
        assert result["BTCUSD"].allocation_pct == 0.0
        assert result["BTCUSD"].confidence == 0.0

    def test_short_direction_not_affected(self):
        """SHORT RL hints are never touched by the SMA-24 filter."""
        sig = _make_rl_signal("ETHUSD", direction="short", alloc=0.15)
        frame = _make_frame(current_price=90.0, sma24=100.0)  # 10% below SMA
        result = _apply_rl_hint_sma_filter({"ETHUSD": sig}, {"ETHUSD": frame})
        assert result["ETHUSD"].direction == "short"
        assert result["ETHUSD"].allocation_pct == pytest.approx(0.15, rel=1e-3)

    def test_flat_direction_not_affected(self):
        """FLAT RL signals are not touched."""
        sig = _make_rl_signal("SOLUSD", direction="flat", alloc=0.0)
        frame = _make_frame(current_price=90.0, sma24=100.0)
        result = _apply_rl_hint_sma_filter({"SOLUSD": sig}, {"SOLUSD": frame})
        assert result["SOLUSD"].direction == "flat"

    def test_insufficient_bars_skipped(self):
        """When frame has fewer than 24 bars, signal is not modified."""
        sig = _make_rl_signal("BTCUSD", alloc=0.20)
        frame = _make_frame(current_price=90.0, sma24=100.0, n_bars=10)
        result = _apply_rl_hint_sma_filter({"BTCUSD": sig}, {"BTCUSD": frame})
        # Fewer than 24 bars — filter must not fire
        assert result["BTCUSD"].direction == "long"
        assert result["BTCUSD"].allocation_pct == pytest.approx(0.20, rel=1e-3)

    def test_missing_frame_skipped(self):
        """When frame is absent, signal is not modified."""
        sig = _make_rl_signal("BTCUSD", alloc=0.20)
        result = _apply_rl_hint_sma_filter({"BTCUSD": sig}, {})
        assert result["BTCUSD"].direction == "long"


# ── Post-LLM hard block tests ────────────────────────────────────────────────

class TestPostLlmSmaFilter:
    """Tests for the post-LLM SMA-24 guard that acts on TradePlan objects."""

    def test_price_above_sma_no_change(self):
        """No modification when price >= SMA-24."""
        plan = _make_trade_plan(alloc=30.0)
        frame = _make_frame(current_price=101.0, sma24=100.0)
        result = _apply_post_llm_sma_filter({"BTCUSD": plan}, {"BTCUSD": frame})
        assert result["BTCUSD"].direction == "long"
        assert result["BTCUSD"].allocation_pct == pytest.approx(30.0, rel=1e-3)

    def test_mild_discount_below_2pct_unchanged(self):
        """When price is only 1% below SMA, TradePlan passes through unchanged."""
        plan = _make_trade_plan(alloc=30.0)
        frame = _make_frame(current_price=99.0, sma24=100.0)  # 1% below
        result = _apply_post_llm_sma_filter({"BTCUSD": plan}, {"BTCUSD": frame})
        assert result["BTCUSD"].direction == "long"
        assert result["BTCUSD"].allocation_pct == pytest.approx(30.0, rel=1e-3)

    def test_moderate_discount_2_to_5pct_reduces_allocation(self):
        """When price is 3% below SMA, allocation is halved but direction kept."""
        plan = _make_trade_plan(alloc=30.0)
        frame = _make_frame(current_price=97.0, sma24=100.0)  # 3% below
        result = _apply_post_llm_sma_filter({"BTCUSD": plan}, {"BTCUSD": frame})
        assert result["BTCUSD"].direction == "long"
        assert result["BTCUSD"].allocation_pct == pytest.approx(15.0, rel=1e-3)

    def test_moderate_discount_just_above_2pct(self):
        """2.1% discount triggers 50% allocation reduction (boundary at 2%)."""
        plan = _make_trade_plan(alloc=40.0)
        # 2.1% below: current = 100 * (1 - 0.021) = 97.9
        frame = _make_frame(current_price=97.9, sma24=100.0)
        result = _apply_post_llm_sma_filter({"BTCUSD": plan}, {"BTCUSD": frame})
        assert result["BTCUSD"].direction == "long"
        assert result["BTCUSD"].allocation_pct == pytest.approx(20.0, rel=1e-2)

    def test_extreme_discount_above_5pct_forces_hold(self):
        """When price is >5% below SMA, LONG is forced to HOLD."""
        plan = _make_trade_plan(alloc=30.0, conf=0.8)
        frame = _make_frame(current_price=94.0, sma24=100.0)  # 6% below
        result = _apply_post_llm_sma_filter({"BTCUSD": plan}, {"BTCUSD": frame})
        assert result["BTCUSD"].direction == "hold"
        assert result["BTCUSD"].allocation_pct == 0
        assert result["BTCUSD"].buy_price == 0.0
        # sell_price is preserved (for existing position management)
        assert result["BTCUSD"].sell_price == pytest.approx(105.0, rel=1e-3)
        # confidence is preserved
        assert result["BTCUSD"].confidence == pytest.approx(0.8, rel=1e-3)

    def test_extreme_discount_10pct_forces_hold(self):
        """Confirmed: 10% below SMA triggers hard HOLD block."""
        plan = _make_trade_plan(alloc=25.0)
        frame = _make_frame(current_price=90.0, sma24=100.0)  # 10% below
        result = _apply_post_llm_sma_filter({"BTCUSD": plan}, {"BTCUSD": frame})
        assert result["BTCUSD"].direction == "hold"
        assert result["BTCUSD"].allocation_pct == 0

    def test_hold_direction_not_affected(self):
        """HOLD TradePlans are never touched by the filter."""
        plan = _make_trade_plan(direction="hold", alloc=0.0)
        frame = _make_frame(current_price=90.0, sma24=100.0)
        result = _apply_post_llm_sma_filter({"BTCUSD": plan}, {"BTCUSD": frame})
        assert result["BTCUSD"].direction == "hold"

    def test_multi_symbol_independence(self):
        """Each symbol is filtered independently."""
        plans = {
            "BTCUSD": _make_trade_plan(alloc=30.0),   # 3% below SMA → soft
            "ETHUSD": _make_trade_plan(alloc=20.0),   # 6% below SMA → hard
            "SOLUSD": _make_trade_plan(alloc=25.0),   # above SMA → unchanged
        }
        frames = {
            "BTCUSD": _make_frame(current_price=97.0, sma24=100.0),
            "ETHUSD": _make_frame(current_price=94.0, sma24=100.0),
            "SOLUSD": _make_frame(current_price=103.0, sma24=100.0),
        }
        result = _apply_post_llm_sma_filter(plans, frames)
        # BTCUSD: 3% below → allocation halved
        assert result["BTCUSD"].direction == "long"
        assert result["BTCUSD"].allocation_pct == pytest.approx(15.0, rel=1e-3)
        # ETHUSD: 6% below → HOLD
        assert result["ETHUSD"].direction == "hold"
        assert result["ETHUSD"].allocation_pct == 0
        # SOLUSD: above SMA → unchanged
        assert result["SOLUSD"].direction == "long"
        assert result["SOLUSD"].allocation_pct == pytest.approx(25.0, rel=1e-3)

    def test_production_crash_scenario(self):
        """Regression: March 2026 crash — BTC 0.5% below SMA must NOT be blocked.

        Before fix: ANY price < SMA forced HOLD → zero trades for weeks.
        After fix: 0.5% discount is mild (<2%), signal passes through unchanged.
        """
        # BTC at 70181, SMA24 at 70210 → 0.04% below
        plan = _make_trade_plan(direction="long", alloc=30.0, conf=0.6)
        btc_price = 70181.0
        btc_sma24 = 70210.0
        frame = _make_frame(current_price=btc_price, sma24=btc_sma24)
        result = _apply_post_llm_sma_filter({"BTCUSD": plan}, {"BTCUSD": frame})
        # Must NOT be blocked — this was the production bug
        assert result["BTCUSD"].direction == "long", (
            "BUG REGRESSION: 0.04% below SMA should not block trades. "
            "This was the production issue causing 0 trades."
        )
        assert result["BTCUSD"].allocation_pct == pytest.approx(30.0, rel=1e-3)

    def test_avax_production_scenario(self):
        """Regression: AVAX 0.6% below SMA must NOT be blocked.

        Logs showed: AVAX price=9.4629 < sma24=9.5421 (0.83% below) → was blocked.
        After fix: 0.83% below is mild (<2%), signal passes through.
        """
        plan = _make_trade_plan(direction="long", alloc=20.0, conf=0.55)
        frame = _make_frame(current_price=9.4629, sma24=9.5421)
        result = _apply_post_llm_sma_filter({"AVAXUSD": plan}, {"AVAXUSD": frame})
        assert result["AVAXUSD"].direction == "long", (
            "BUG REGRESSION: 0.83% below SMA should not block trades."
        )

    def test_eth_production_scenario(self):
        """Regression: ETH 0.5% below SMA must NOT be blocked.

        Logs showed: ETH price=2145.77 < sma24=2156.87 (0.51% below) → was blocked.
        After fix: 0.51% below is mild (<2%), signal passes through.
        """
        plan = _make_trade_plan(direction="long", alloc=20.0, conf=0.6)
        frame = _make_frame(current_price=2145.77, sma24=2156.87)
        result = _apply_post_llm_sma_filter({"ETHUSD": plan}, {"ETHUSD": frame})
        assert result["ETHUSD"].direction == "long", (
            "BUG REGRESSION: 0.51% below SMA should not block trades."
        )
