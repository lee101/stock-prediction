from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from pufferlib_market.inference import PPOTrader, Policy


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
