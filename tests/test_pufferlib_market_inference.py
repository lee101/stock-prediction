from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
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
