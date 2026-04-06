"""Tests for RL signal action masking to restrict actions to tradable symbols."""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "rl_trading_agent_binance"))

from rl_signal import (
    FEATURES_PER_SYM,
    MIXED23_SYMBOLS,
    RLSignalGenerator,
    _build_action_names,
)


def _make_generator_stub():
    """Build a minimal RLSignalGenerator without loading a real checkpoint."""
    gen = object.__new__(RLSignalGenerator)
    gen.symbols = MIXED23_SYMBOLS
    gen.num_symbols = len(MIXED23_SYMBOLS)
    gen.features_per_sym = FEATURES_PER_SYM
    gen.action_allocation_bins = 1
    gen.action_level_bins = 1
    gen.per_symbol_actions = 1
    gen.disable_shorts = False
    gen.action_names = _build_action_names(MIXED23_SYMBOLS)
    gen.obs_size = gen.num_symbols * FEATURES_PER_SYM + 5 + gen.num_symbols
    gen.num_actions = 1 + 2 * gen.num_symbols  # 47
    gen._episode_step = 0
    gen.max_steps = 720
    gen.forecast_cache_root = None
    return gen


class TestActionToSymbol:
    def test_flat(self):
        gen = _make_generator_stub()
        assert gen._action_to_symbol(0) is None

    def test_long_first(self):
        gen = _make_generator_stub()
        assert gen._action_to_symbol(1) == "AAPL"

    def test_long_last(self):
        gen = _make_generator_stub()
        assert gen._action_to_symbol(23) == "XRPUSD"

    def test_short_first(self):
        gen = _make_generator_stub()
        assert gen._action_to_symbol(24) == "AAPL"

    def test_short_last(self):
        gen = _make_generator_stub()
        assert gen._action_to_symbol(46) == "XRPUSD"

    def test_out_of_range(self):
        gen = _make_generator_stub()
        assert gen._action_to_symbol(47) is None


class TestMaskLogits:
    def test_two_tradable_symbols(self):
        gen = _make_generator_stub()
        logits = np.ones(47, dtype=np.float32)
        masked = gen._mask_logits(logits, ["BTCUSD", "ETHUSD"])

        assert np.isfinite(masked[0]), "FLAT must never be masked"

        btc_idx = list(MIXED23_SYMBOLS).index("BTCUSD")
        eth_idx = list(MIXED23_SYMBOLS).index("ETHUSD")
        long_btc = 1 + btc_idx
        long_eth = 1 + eth_idx
        short_btc = 1 + gen.num_symbols + btc_idx
        short_eth = 1 + gen.num_symbols + eth_idx
        valid = {0, long_btc, long_eth, short_btc, short_eth}

        for i in range(47):
            if i in valid:
                assert np.isfinite(masked[i]), f"action {i} should be finite"
            else:
                assert masked[i] == -np.inf, f"action {i} should be -inf"

    def test_flat_never_masked(self):
        gen = _make_generator_stub()
        logits = np.ones(47, dtype=np.float32)
        masked = gen._mask_logits(logits, [])
        assert np.isfinite(masked[0])

    def test_all_symbols_tradable(self):
        gen = _make_generator_stub()
        logits = np.random.randn(47).astype(np.float32)
        masked = gen._mask_logits(logits, list(MIXED23_SYMBOLS))
        np.testing.assert_array_equal(masked, logits)

    def test_empty_tradable_only_flat_survives(self):
        gen = _make_generator_stub()
        logits = np.ones(47, dtype=np.float32)
        masked = gen._mask_logits(logits, [])
        assert np.isfinite(masked[0])
        assert np.all(masked[1:] == -np.inf)

    def test_does_not_mutate_input(self):
        gen = _make_generator_stub()
        logits = np.ones(47, dtype=np.float32)
        original = logits.copy()
        gen._mask_logits(logits, ["BTCUSD"])
        np.testing.assert_array_equal(logits, original)

    def test_argmax_returns_tradable_action(self):
        gen = _make_generator_stub()
        rng = np.random.default_rng(42)
        tradable = ["BTCUSD", "ETHUSD", "SOLUSD", "DOGEUSD", "AAVEUSD", "LINKUSD"]
        tradable_set = set(tradable)

        for _ in range(100):
            logits = rng.standard_normal(47).astype(np.float32)
            masked = gen._mask_logits(logits, tradable)
            action = int(masked.argmax())
            sym = gen._action_to_symbol(action)
            assert sym is None or sym in tradable_set, (
                f"argmax action {action} maps to {sym}, not in tradable set"
            )

    def test_six_crypto_masking(self):
        gen = _make_generator_stub()
        tradable = ["BTCUSD", "ETHUSD", "SOLUSD", "DOGEUSD", "AAVEUSD", "LINKUSD"]
        logits = np.zeros(47, dtype=np.float32)
        # Make UNI (action 20) the highest -- this is the bug scenario
        logits[20] = 10.0
        masked = gen._mask_logits(logits, tradable)
        action = int(masked.argmax())
        sym = gen._action_to_symbol(action)
        assert action != 20, "UNI action should be masked"
        assert sym is None or sym in set(tradable)

    def test_per_symbol_actions_gt_1(self):
        gen = _make_generator_stub()
        gen.per_symbol_actions = 2
        gen.num_actions = 1 + 2 * gen.num_symbols * 2  # 93
        logits = np.ones(gen.num_actions, dtype=np.float32)
        masked = gen._mask_logits(logits, ["BTCUSD"])
        btc_idx = list(MIXED23_SYMBOLS).index("BTCUSD")
        long_btc_0 = 1 + btc_idx * 2
        long_btc_1 = 1 + btc_idx * 2 + 1
        short_btc_0 = 1 + gen.num_symbols * 2 + btc_idx * 2
        short_btc_1 = 1 + gen.num_symbols * 2 + btc_idx * 2 + 1
        valid = {0, long_btc_0, long_btc_1, short_btc_0, short_btc_1}
        for i in range(gen.num_actions):
            if i in valid:
                assert np.isfinite(masked[i]), f"action {i} should be finite"
            else:
                assert masked[i] == -np.inf, f"action {i} should be -inf"


class TestMaskShorts:
    def test_shorts_masked(self):
        gen = _make_generator_stub()
        logits = np.ones(47, dtype=np.float32)
        masked = gen._mask_shorts(logits)
        S = gen.num_symbols
        assert np.isfinite(masked[0])
        for i in range(1, 1 + S):
            assert np.isfinite(masked[i]), f"long action {i} should be finite"
        for i in range(1 + S, 47):
            assert masked[i] == -np.inf, f"short action {i} should be -inf"

    def test_does_not_mutate_input(self):
        gen = _make_generator_stub()
        logits = np.ones(47, dtype=np.float32)
        original = logits.copy()
        gen._mask_shorts(logits)
        np.testing.assert_array_equal(logits, original)

    def test_flat_stays(self):
        gen = _make_generator_stub()
        logits = np.zeros(47, dtype=np.float32)
        logits[0] = 5.0
        masked = gen._mask_shorts(logits)
        assert masked.argmax() == 0

    def test_short_highest_becomes_long(self):
        gen = _make_generator_stub()
        logits = np.zeros(47, dtype=np.float32)
        logits[30] = 10.0  # SHORT action
        logits[5] = 3.0    # LONG action
        masked = gen._mask_shorts(logits)
        assert masked.argmax() == 5

    def test_per_symbol_actions_gt_1(self):
        gen = _make_generator_stub()
        gen.per_symbol_actions = 2
        gen.num_actions = 1 + 2 * gen.num_symbols * 2
        logits = np.ones(gen.num_actions, dtype=np.float32)
        masked = gen._mask_shorts(logits)
        S = gen.num_symbols
        psa = gen.per_symbol_actions
        short_start = 1 + S * psa
        for i in range(short_start):
            assert np.isfinite(masked[i])
        for i in range(short_start, gen.num_actions):
            assert masked[i] == -np.inf


class TestSignalMetadata:
    def test_get_signal_populates_confidence_gap_and_alloc_for_long(self):
        gen = _make_generator_stub()
        gen.symbols = ("BTCUSD", "ETHUSD")
        gen.num_symbols = 2
        gen.features_per_sym = FEATURES_PER_SYM
        gen.action_allocation_bins = 2
        gen.action_level_bins = 1
        gen.per_symbol_actions = 2
        gen.obs_size = gen.num_symbols * FEATURES_PER_SYM + 5 + gen.num_symbols
        gen.num_actions = 1 + 2 * gen.num_symbols * gen.per_symbol_actions

        logits = np.array([0.0, -2.0, 3.5, -3.0, -4.0, -6.0, -7.0, -8.0, -9.0], dtype=np.float32)

        class _Policy:
            def __call__(self, obs):
                import torch

                return torch.tensor([logits], dtype=torch.float32), torch.tensor([1.25], dtype=torch.float32)

        gen.policy = _Policy()
        gen.device = "cpu"
        signal = gen.get_signal(
            portfolio=type("P", (), {"cash_usd": 1.0, "position_value_usd": 0.0, "unrealized_pnl_usd": 0.0, "hold_hours": 0, "is_short": False, "position_symbol": None})(),
            klines_map={"BTCUSD": None, "ETHUSD": None},
            tradable_symbols=["BTCUSD", "ETHUSD"],
            spot_only=False,
        )

        assert signal.action_name == "LONG_BTC"
        assert signal.direction == "long"
        assert signal.confidence > 0.5
        assert signal.logit_gap > 0.0
        assert signal.allocation_pct == 1.0
        assert signal.value == 1.25

    def test_get_signal_masks_shorts_and_returns_flat_when_only_short_was_hot(self):
        gen = _make_generator_stub()
        gen.symbols = ("BTCUSD",)
        gen.num_symbols = 1
        gen.features_per_sym = FEATURES_PER_SYM
        gen.action_allocation_bins = 1
        gen.action_level_bins = 1
        gen.per_symbol_actions = 1
        gen.obs_size = gen.num_symbols * FEATURES_PER_SYM + 5 + gen.num_symbols
        gen.num_actions = 3

        logits = np.array([0.0, -1.0, 5.0], dtype=np.float32)

        class _Policy:
            def __call__(self, obs):
                import torch

                return torch.tensor([logits], dtype=torch.float32), torch.tensor([0.0], dtype=torch.float32)

        gen.policy = _Policy()
        gen.device = "cpu"
        signal = gen.get_signal(
            portfolio=type("P", (), {"cash_usd": 1.0, "position_value_usd": 0.0, "unrealized_pnl_usd": 0.0, "hold_hours": 0, "is_short": False, "position_symbol": None})(),
            klines_map={"BTCUSD": None},
            tradable_symbols=["BTCUSD"],
            spot_only=True,
        )

        assert signal.direction == "flat"
        assert signal.confidence >= 0.5
        assert signal.logit_gap == 0.0
        assert signal.allocation_pct == 0.0
