from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pufferlib_market.inference import PPOTrader, Policy
from pufferlib_market.inference_daily import DailyPPOTrader


def _write_checkpoint(
    path: Path,
    *,
    num_symbols: int,
    num_actions: int,
    hidden: int,
    num_blocks: int,
    hot_action: int,
    action_allocation_bins: int = 1,
    action_level_bins: int = 1,
    action_max_offset_bps: float = 0.0,
) -> None:
    obs_size = num_symbols * 16 + 5 + num_symbols
    model = Policy(obs_size, num_actions, hidden=hidden, num_blocks=num_blocks)
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        model.actor[2].bias[hot_action] = 10.0
    payload = {
        "model": model.state_dict(),
        "config": {"hidden_size": int(hidden), "num_blocks": int(num_blocks)},
        "action_allocation_bins": int(action_allocation_bins),
        "action_level_bins": int(action_level_bins),
        "action_max_offset_bps": float(action_max_offset_bps),
    }
    torch.save(payload, path)


def _write_mlp_checkpoint(
    path: Path,
    *,
    num_symbols: int,
    num_actions: int,
    hidden: int,
    hot_action: int,
    activation: str = "relu",
    use_encoder_norm: bool = False,
) -> None:
    from pufferlib_market.train import TradingPolicy

    obs_size = num_symbols * 16 + 5 + num_symbols
    model = TradingPolicy(
        obs_size,
        num_actions,
        hidden=hidden,
        activation=activation,
        use_encoder_norm=use_encoder_norm,
    )
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        model.actor[2].bias[hot_action] = 10.0
    payload = {
        "model": model.state_dict(),
        "arch": "mlp_relu_sq" if activation == "relu_sq" else "mlp",
        "activation": activation,
        "hidden_size": int(hidden),
        "use_encoder_norm": bool(use_encoder_norm),
    }
    torch.save(payload, path)


def test_ppotrader_respects_symbol_override_and_decodes_alloc_bins(tmp_path: Path):
    ckpt = tmp_path / "alloc_bins.pt"
    symbols = ["AAA", "BBB"]
    # 1 + 2 * 2 symbols * 3 alloc bins = 13 actions.
    _write_checkpoint(
        ckpt,
        num_symbols=2,
        num_actions=13,
        hidden=32,
        num_blocks=1,
        hot_action=6,  # long BBB with alloc_idx=2 => 100%
        action_allocation_bins=3,
        action_level_bins=1,
    )

    trader = PPOTrader(str(ckpt), device="cpu", symbols=symbols, long_only=False)
    features = np.zeros((2, 16), dtype=np.float32)
    prices = {"AAA": 10.0, "BBB": 11.0}
    signal = trader.get_signal(features, prices)

    assert trader.num_symbols == 2
    assert trader.obs_size == 39
    assert signal.symbol == "BBB"
    assert signal.direction == "long"
    assert signal.allocation_pct == 1.0
    assert signal.level_offset_bps == 0.0


def test_ppotrader_decodes_level_bins_offset(tmp_path: Path):
    ckpt = tmp_path / "level_bins.pt"
    symbols = ["AAA"]
    # alloc_bins=2, level_bins=3 => per_symbol_actions=6 => actions=13.
    # rem=5 => alloc_idx=1, level_idx=2 => action=1+5=6.
    _write_checkpoint(
        ckpt,
        num_symbols=1,
        num_actions=13,
        hidden=32,
        num_blocks=1,
        hot_action=6,
        action_allocation_bins=2,
        action_level_bins=3,
        action_max_offset_bps=50.0,
    )

    trader = PPOTrader(str(ckpt), device="cpu", symbols=symbols, long_only=False)
    features = np.zeros((1, 16), dtype=np.float32)
    prices = {"AAA": 10.0}
    signal = trader.get_signal(features, prices)

    assert signal.symbol == "AAA"
    assert signal.direction == "long"
    assert signal.allocation_pct == 1.0
    assert signal.level_offset_bps == 50.0


def test_ppotrader_long_only_masks_short_actions(tmp_path: Path):
    ckpt = tmp_path / "short_capable.pt"
    symbols = ["AAA", "BBB"]
    _write_checkpoint(
        ckpt,
        num_symbols=2,
        num_actions=5,
        hidden=32,
        num_blocks=1,
        hot_action=4,  # short BBB
    )

    trader = PPOTrader(str(ckpt), device="cpu", symbols=symbols, long_only=True)
    features = np.zeros((2, 16), dtype=np.float32)
    prices = {"AAA": 10.0, "BBB": 11.0}
    signal = trader.get_signal(features, prices)

    assert signal.action == "flat"
    assert signal.symbol is None
    assert signal.direction is None


def test_ppotrader_long_only_decode_handles_short_side_without_crash(tmp_path: Path):
    ckpt = tmp_path / "decode_short.pt"
    symbols = ["AAA", "BBB"]
    _write_checkpoint(
        ckpt,
        num_symbols=2,
        num_actions=5,
        hidden=32,
        num_blocks=1,
        hot_action=4,
    )

    trader = PPOTrader(str(ckpt), device="cpu", symbols=symbols, long_only=True)
    signal = trader._decode_action(4, 0.9, 1.2)

    assert signal.action == "short_BBB"
    assert signal.symbol == "BBB"
    assert signal.direction == "short"


def test_ppotrader_rejects_wrapped_checkpoint_with_invalid_model_payload(tmp_path: Path):
    ckpt = tmp_path / "invalid_model_payload.pt"
    torch.save({"model": {"arch": "mlp"}}, ckpt)

    with pytest.raises(KeyError, match="Checkpoint is missing a valid 'model' state_dict"):
        PPOTrader(str(ckpt), device="cpu", symbols=["AAA"])


def test_ppotrader_ignores_invalid_checkpoint_action_grid_metadata(tmp_path: Path):
    ckpt = tmp_path / "invalid_action_grid.pt"
    num_symbols = 1
    obs_size = num_symbols * 16 + 5 + num_symbols
    num_actions = 3
    model = Policy(obs_size, num_actions, hidden=32, num_blocks=1)
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        model.actor[2].bias[1] = 10.0
    torch.save(
        {
            "model": model.state_dict(),
            "config": {"hidden_size": 32, "num_blocks": 1},
            "action_allocation_bins": "oops",
            "action_level_bins": 2.5,
            "action_max_offset_bps": "bad",
        },
        ckpt,
    )

    trader = PPOTrader(str(ckpt), device="cpu", symbols=["AAA"], long_only=False)
    signal = trader.get_signal(np.zeros((1, 16), dtype=np.float32), {"AAA": 10.0})

    assert trader.action_allocation_bins == 1
    assert trader.action_level_bins == 1
    assert trader.action_max_offset_bps == pytest.approx(0.0)
    assert signal.symbol == "AAA"
    assert signal.direction == "long"
    assert signal.allocation_pct == pytest.approx(1.0)
    assert signal.level_offset_bps == pytest.approx(0.0)


def test_ppotrader_summary_dict_reports_resolved_config(tmp_path: Path):
    ckpt = tmp_path / "summary.pt"
    symbols = ["AAA", "BBB"]
    _write_checkpoint(
        ckpt,
        num_symbols=2,
        num_actions=13,
        hidden=32,
        num_blocks=1,
        hot_action=6,
        action_allocation_bins=3,
        action_level_bins=1,
        action_max_offset_bps=12.5,
    )

    trader = PPOTrader(str(ckpt), device="cpu", symbols=symbols, long_only=True)
    summary = trader.summary_dict()

    assert summary == {
        "checkpoint": str(ckpt),
        "device": "cpu",
        "arch": "resmlp",
        "hidden_size": 32,
        "num_actions": 13,
        "num_symbols": 2,
        "long_only": True,
        "action_allocation_bins": 3,
        "action_level_bins": 1,
        "action_max_offset_bps": pytest.approx(12.5),
        "max_steps": 720,
        "symbols": symbols,
    }


def test_ppotrader_init_is_quiet(tmp_path: Path, capsys):
    ckpt = tmp_path / "quiet.pt"
    _write_checkpoint(
        ckpt,
        num_symbols=1,
        num_actions=3,
        hidden=32,
        num_blocks=1,
        hot_action=1,
    )

    PPOTrader(str(ckpt), device="cpu", symbols=["AAA"])

    captured = capsys.readouterr()
    assert captured.out == ""


def test_ppotrader_loads_mlp_activation_metadata(tmp_path: Path):
    ckpt = tmp_path / "mlp_relu_sq.pt"
    _write_mlp_checkpoint(
        ckpt,
        num_symbols=1,
        num_actions=3,
        hidden=16,
        hot_action=1,
        activation="relu_sq",
        use_encoder_norm=True,
    )

    trader = PPOTrader(str(ckpt), device="cpu", symbols=["AAA"])

    assert trader.arch == "mlp_relu_sq"
    assert trader.hidden_size == 16
    assert getattr(trader.policy, "_activation_name", None) == "relu_sq"
    assert getattr(trader.policy, "_use_encoder_norm", False) is True


def test_main_prints_self_describing_json(monkeypatch, capsys):
    import pufferlib_market.inference as inference_mod

    fake_args = SimpleNamespace(checkpoint="fake.pt", device="cpu")
    fake_trader = SimpleNamespace(
        num_symbols=2,
        SYMBOLS=["AAA", "BBB"],
        summary_dict=lambda: {
            "checkpoint": "fake.pt",
            "device": "cpu",
            "arch": "resmlp",
            "hidden_size": 32,
            "num_actions": 13,
            "num_symbols": 2,
            "long_only": False,
            "action_allocation_bins": 3,
            "action_level_bins": 1,
            "action_max_offset_bps": 12.5,
            "max_steps": 720,
            "symbols": ["AAA", "BBB"],
        },
        get_signal=lambda features, prices: inference_mod.TradingSignal(
            action="long_BBB",
            symbol="BBB",
            direction="long",
            confidence=0.9,
            value_estimate=1.2,
            allocation_pct=1.0,
            level_offset_bps=0.0,
        ),
    )

    monkeypatch.setattr(inference_mod.argparse.ArgumentParser, "parse_args", lambda self: fake_args)
    monkeypatch.setattr(inference_mod, "PPOTrader", lambda checkpoint, device: fake_trader)

    inference_mod.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["model"]["arch"] == "resmlp"
    assert payload["model"]["symbols"] == ["AAA", "BBB"]
    assert payload["signal"] == {
        "action": "long_BBB",
        "symbol": "BBB",
        "direction": "long",
        "confidence": 0.9,
        "value_estimate": 1.2,
        "allocation_pct": 1.0,
        "level_offset_bps": 0.0,
    }


class TestBuildObservationUnrealizedPnl:
    """obs[base+2] must match C env: unrealised_pnl / INITIAL_CASH, not hardcoded 0."""

    def _make_trader(self, tmp_path, symbols=("AAA", "BBB")):
        ckpt = tmp_path / "ck.pt"
        num_sym = len(symbols)
        num_actions = 1 + num_sym
        _write_checkpoint(ckpt, num_symbols=num_sym, num_actions=num_actions, hidden=16, num_blocks=1, hot_action=0)
        trader = PPOTrader(str(ckpt), device="cpu", symbols=list(symbols))
        return trader

    def test_flat_position_zero_pnl(self, tmp_path):
        trader = self._make_trader(tmp_path)
        trader.cash = 10_000.0
        trader.current_position = None
        trader.position_qty = 0.0
        trader.entry_price = 0.0
        features = np.zeros((2, 16), dtype=np.float32)
        prices = {"AAA": 100.0, "BBB": 200.0}
        obs = trader.build_observation(features, prices)
        base = 2 * 16
        assert obs[base + 2] == 0.0

    def test_long_profitable_position(self, tmp_path):
        # Entry at 100, current price 110 → 10% up → obs[base+2] = 0.10
        trader = self._make_trader(tmp_path)
        entry_price = 100.0
        cur_price = 110.0
        trader.cash = 0.0
        trader.entry_price = entry_price
        trader.position_qty = 10_000.0 / entry_price  # 100 shares
        trader.current_position = 0  # long AAA
        features = np.zeros((2, 16), dtype=np.float32)
        prices = {"AAA": cur_price, "BBB": 200.0}
        obs = trader.build_observation(features, prices)
        base = 2 * 16
        expected_pnl = (10_000.0 / entry_price) * (cur_price - entry_price) / 10_000.0
        assert abs(obs[base + 2] - expected_pnl) < 1e-5
        assert obs[base + 2] > 0.0  # profitable → positive

    def test_long_losing_position(self, tmp_path):
        # Entry at 100, current price 90 → -10% → obs[base+2] = -0.10
        trader = self._make_trader(tmp_path)
        entry_price = 100.0
        cur_price = 90.0
        trader.cash = 0.0
        trader.entry_price = entry_price
        trader.position_qty = 10_000.0 / entry_price
        trader.current_position = 0  # long AAA
        features = np.zeros((2, 16), dtype=np.float32)
        prices = {"AAA": cur_price, "BBB": 200.0}
        obs = trader.build_observation(features, prices)
        base = 2 * 16
        expected_pnl = (10_000.0 / entry_price) * (cur_price - entry_price) / 10_000.0
        assert abs(obs[base + 2] - expected_pnl) < 1e-5
        assert obs[base + 2] < 0.0  # losing → negative

    def test_pnl_formula_matches_c_env(self, tmp_path):
        """Exact formula: qty*(cur-entry)/10000 = (cur-entry)/entry."""
        trader = self._make_trader(tmp_path)
        entry_price = 150.0
        cur_price = 157.5  # +5%
        trader.cash = 0.0
        trader.entry_price = entry_price
        trader.position_qty = 10_000.0 / entry_price
        trader.current_position = 0
        features = np.zeros((2, 16), dtype=np.float32)
        prices = {"AAA": cur_price, "BBB": 200.0}
        obs = trader.build_observation(features, prices)
        base = 2 * 16
        assert abs(obs[base + 2] - 0.05) < 1e-5  # exactly 5%


class TestDailyPPOTraderHoldDaysConvention:
    """DailyPPOTrader.step_day / update_state must match the C env hold_hours semantics.

    C env rule: build_observation() runs BEFORE each action; hold_hours++ only
    on a HOLD action.  So:
      - First obs while holding (day after buying) → hold_hours=0
      - After N HOLD actions → hold_hours=N
      - After closing → hold_hours=0

    Bugs fixed: step_day() used to increment hold_days on the same step as a
    buy (giving hold_hours=1 on day-1-after-buying) and never cleared a stale
    hold_days after a close.
    """

    def _make_daily_trader(self, tmp_path: Path) -> DailyPPOTrader:
        ckpt = tmp_path / "daily.pt"
        symbols = ["AAPL"]
        _write_checkpoint(ckpt, num_symbols=1, num_actions=2, hidden=16, num_blocks=1, hot_action=0)
        return DailyPPOTrader(str(ckpt), device="cpu", symbols=symbols)

    def test_hold_hours_zero_on_first_day_after_buying(self, tmp_path: Path):
        """Day-1-after-buying obs[base+3] must be 0/max_steps (not 1/max_steps)."""
        trader = self._make_daily_trader(tmp_path)
        # Day 0: flat
        assert trader.hold_hours == 0
        # BUY then advance one day
        trader.update_state(1, 100.0, "AAPL", qty=100.0)
        trader.step_day()
        # Day 1: first obs while holding
        assert trader.hold_hours == 0, (
            f"expected hold_hours=0 (C env convention) on day after buying, got {trader.hold_hours}"
        )

    def test_hold_hours_increments_correctly_while_holding(self, tmp_path: Path):
        """hold_hours must be 0 on day 1, 1 on day 2, 2 on day 3, ..."""
        trader = self._make_daily_trader(tmp_path)
        trader.update_state(1, 100.0, "AAPL", qty=100.0)
        for expected in range(5):
            trader.step_day()
            assert trader.hold_hours == expected, (
                f"day {expected+1}: expected hold_hours={expected}, got {trader.hold_hours}"
            )

    def test_hold_hours_resets_to_zero_after_close(self, tmp_path: Path):
        """After closing a 3-day position, the next obs must see hold_hours=0."""
        trader = self._make_daily_trader(tmp_path)
        trader.update_state(1, 100.0, "AAPL", qty=100.0)
        for _ in range(3):
            trader.step_day()
        assert trader.hold_hours == 2  # sanity: held 3 days → hold_hours=2

        # Close
        trader.update_state(0, 110.0, "")
        trader.step_day()
        assert trader.hold_hours == 0, (
            f"expected hold_hours=0 after closing, got {trader.hold_hours} (stale)"
        )
