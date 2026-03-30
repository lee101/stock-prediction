from __future__ import annotations

import struct
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from pufferlib_market import evaluate as eval_mod
from pufferlib_market.hourly_replay import MktdData
from pufferlib_market.train import GRUTradingPolicy


def _make_data(num_timesteps: int, num_symbols: int = 1) -> MktdData:
    features = np.zeros((num_timesteps, num_symbols, 16), dtype=np.float32)
    prices = np.ones((num_timesteps, num_symbols, 5), dtype=np.float32)
    tradable = np.ones((num_timesteps, num_symbols), dtype=np.uint8)
    return MktdData(
        version=2,
        symbols=[f"SYM{i}" for i in range(num_symbols)],
        features=features,
        prices=prices,
        tradable=tradable,
    )


def _write_fake_mktd_header(
    path: Path,
    *,
    num_symbols: int = 1,
    num_timesteps: int = 60,
    features_per_sym: int = 16,
) -> Path:
    header = struct.pack("<4sIIIII", b"MKTD", 2, num_symbols, num_timesteps, features_per_sym, 0)
    path.write_bytes(header + b"\x00" * (64 - len(header)))
    return path


def _fake_main_args(tmp_path: Path, **overrides) -> SimpleNamespace:
    values = dict(
        checkpoint="checkpoint.pt",
        data_path=str(_write_fake_mktd_header(tmp_path / "data.mktd")),
        max_steps=50,
        fee_rate=0.0,
        max_leverage=1.0,
        short_borrow_apr=0.0,
        periods_per_year=8760.0,
        action_allocation_bins=1,
        action_level_bins=1,
        action_max_offset_bps=0.0,
        fill_slippage_bps=0.0,
        num_envs=4,
        num_episodes=1,
        max_eval_steps=100,
        seed=123,
        deterministic=True,
        mode="random",
        extra_checkpoints=[],
        hidden_size=16,
        arch="mlp",
        cpu=True,
        max_hold_hours=0,
        disable_shorts=False,
        drawdown_profit_early_exit=True,
        drawdown_profit_early_exit_verbose=False,
        drawdown_profit_early_exit_min_steps=20,
        drawdown_profit_early_exit_progress_fraction=0.5,
        calmar=False,
        sliding=True,
        stride=25,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_load_policy_with_metadata_infers_effective_arch_and_hidden_size():
    obs_size = 21
    num_actions = 9
    source_policy = eval_mod.ResidualTradingPolicy(obs_size, num_actions, hidden=32, num_blocks=4)
    ckpt = {
        "model": source_policy.state_dict(),
        "arch": "resmlp",
        "hidden_size": 32,
    }

    loaded_policy, effective_arch, effective_hidden = eval_mod._load_policy_with_metadata(
        ckpt,
        obs_size,
        num_actions,
        hidden=16,
        arch="mlp",
        device=torch.device("cpu"),
    )

    assert isinstance(loaded_policy, eval_mod.ResidualTradingPolicy)
    assert effective_arch == "resmlp"
    assert effective_hidden == 32
    assert len(loaded_policy.blocks) == 4


def test_load_policy_with_metadata_preserves_checkpoint_declared_gru_arch():
    obs_size = 22
    num_actions = 3
    source_policy = GRUTradingPolicy(obs_size, num_actions, hidden=32)
    ckpt = {
        "model": source_policy.state_dict(),
        "arch": "gru",
        "hidden_size": 32,
    }

    loaded_policy, effective_arch, effective_hidden = eval_mod._load_policy_with_metadata(
        ckpt,
        obs_size,
        num_actions,
        hidden=16,
        arch="mlp",
        device=torch.device("cpu"),
    )

    assert effective_arch == "gru"
    assert effective_hidden == 32
    assert loaded_policy.__class__.__name__ == "_Wrapped"
    assert loaded_policy.inner.__class__.__name__ == "GRUTradingPolicy"


def test_main_rejects_checkpoint_without_model_state(tmp_path: Path):
    fake_args = _fake_main_args(tmp_path)
    fake_binding = SimpleNamespace(shared=lambda **_: None)

    with patch.object(eval_mod.argparse.ArgumentParser, "parse_args", return_value=fake_args), \
         patch.object(eval_mod.torch, "load", return_value={"update": 7}), \
         patch.dict(sys.modules, {"pufferlib_market.binding": fake_binding}):
        with pytest.raises(KeyError, match="missing a valid 'model' state_dict"):
            eval_mod.main()


def test_main_rejects_non_mapping_checkpoint_payload(tmp_path: Path):
    fake_args = _fake_main_args(tmp_path)
    fake_binding = SimpleNamespace(shared=lambda **_: None)

    with patch.object(eval_mod.argparse.ArgumentParser, "parse_args", return_value=fake_args), \
         patch.object(eval_mod.torch, "load", return_value=[]), \
         patch.dict(sys.modules, {"pufferlib_market.binding": fake_binding}):
        with pytest.raises(TypeError, match="must load to a mapping"):
            eval_mod.main()


def test_main_reports_effective_checkpoint_arch_and_hidden_size(tmp_path: Path, capsys):
    fake_args = _fake_main_args(tmp_path, arch="mlp", hidden_size=16)
    source_policy = eval_mod.ResidualTradingPolicy(22, 3, hidden=32, num_blocks=4)
    fake_binding = SimpleNamespace(shared=lambda **_: None)

    with patch.object(eval_mod.argparse.ArgumentParser, "parse_args", return_value=fake_args), \
         patch.object(eval_mod.torch, "load", return_value={"model": source_policy.state_dict(), "update": 9}), \
         patch.object(eval_mod, "read_mktd", return_value=_make_data(60)), \
         patch.object(eval_mod, "_build_policy_fn", return_value=lambda obs: 0), \
         patch.object(eval_mod, "sliding_window_eval", return_value=[]), \
         patch.object(eval_mod, "aggregate_sliding_results", return_value={"calmar": 1.23, "annualized_return": 0.0, "worst_max_drawdown": 0.0}), \
         patch.object(eval_mod, "print_sliding_results", return_value=None), \
         patch.dict(sys.modules, {"pufferlib_market.binding": fake_binding}):
        eval_mod.main()

    captured = capsys.readouterr()
    assert "CLI policy config: arch=mlp, hidden_size=16" in captured.out
    assert "Action grid: alloc_bins=1 level_bins=1 max_offset_bps=0.0" in captured.out
    assert "Runtime: device=cpu, deterministic=True" in captured.out
    assert f"Checkpoint file: {Path(fake_args.checkpoint).resolve()}" in captured.out
    assert "Effective checkpoint config: arch=resmlp, hidden_size=32" in captured.out
    assert "Checkpoint overrides CLI policy config: arch mlp -> resmlp, hidden_size 16 -> 32" in captured.out
    assert "Loaded checkpoint: update=9, train_best_return=?, arch=resmlp" in captured.out


def test_main_tolerates_extra_checkpoint_without_best_return(tmp_path: Path, capsys):
    fake_args = _fake_main_args(tmp_path, extra_checkpoints=["extra.pt"])
    base_policy = eval_mod.TradingPolicy(22, 3, hidden=16)
    extra_policy = eval_mod.TradingPolicy(22, 3, hidden=16)
    fake_binding = SimpleNamespace(shared=lambda **_: None)

    with patch.object(eval_mod.argparse.ArgumentParser, "parse_args", return_value=fake_args), \
         patch.object(
             eval_mod.torch,
             "load",
             side_effect=[
                 {"model": base_policy.state_dict(), "update": 4, "best_return": 0.1},
                 {"model": extra_policy.state_dict(), "update": 5},
             ],
         ), \
         patch.object(eval_mod, "read_mktd", return_value=_make_data(60)), \
         patch.object(eval_mod, "_build_policy_fn", return_value=lambda obs: 0), \
         patch.object(eval_mod, "sliding_window_eval", return_value=[]), \
         patch.object(eval_mod, "aggregate_sliding_results", return_value={"calmar": 1.23, "annualized_return": 0.0, "worst_max_drawdown": 0.0}), \
         patch.object(eval_mod, "print_sliding_results", return_value=None), \
         patch.dict(sys.modules, {"pufferlib_market.binding": fake_binding}):
        eval_mod.main()

    captured = capsys.readouterr()
    assert f"Checkpoint file: {Path(fake_args.checkpoint).resolve()}" in captured.out
    assert "Loaded checkpoint: update=4, train_best_return=0.1000, arch=mlp" in captured.out
    assert f"  + ensemble member file: {Path('extra.pt').resolve()}" in captured.out
    assert "  + ensemble member: update=5 train_best_return=? arch=mlp hidden_size=16" in captured.out


def test_main_rejects_extra_checkpoint_with_incompatible_action_grid(tmp_path: Path):
    fake_args = _fake_main_args(tmp_path, extra_checkpoints=["extra.pt"])
    base_policy = eval_mod.TradingPolicy(22, 3, hidden=16)
    extra_policy = eval_mod.TradingPolicy(22, 3, hidden=16)
    fake_binding = SimpleNamespace(shared=lambda **_: None)

    with patch.object(eval_mod.argparse.ArgumentParser, "parse_args", return_value=fake_args), \
         patch.object(
             eval_mod,
             "load_checkpoint_payload",
             side_effect=[
                 {"model": base_policy.state_dict(), "update": 4, "best_return": 0.1},
                 {
                     "model": extra_policy.state_dict(),
                     "update": 5,
                     "action_allocation_bins": 2,
                     "action_level_bins": 3,
                     "action_max_offset_bps": 12.5,
                 },
             ],
         ), \
         patch.dict(sys.modules, {"pufferlib_market.binding": fake_binding}):
        with pytest.raises(RuntimeError, match=f"Checkpoint {Path('extra.pt').resolve()} is incompatible with ensemble action grid") as excinfo:
            eval_mod.main()

    assert "expected alloc_bins=1 level_bins=1 max_offset_bps=0.0" in str(excinfo.value)
    assert "got alloc_bins=2 level_bins=3 max_offset_bps=12.5" in str(excinfo.value)


def test_main_wraps_extra_checkpoint_load_errors_with_checkpoint_path(tmp_path: Path):
    fake_args = _fake_main_args(tmp_path, extra_checkpoints=["missing-extra.pt"])
    base_policy = eval_mod.TradingPolicy(22, 3, hidden=16)
    fake_binding = SimpleNamespace(shared=lambda **_: None)

    with patch.object(eval_mod.argparse.ArgumentParser, "parse_args", return_value=fake_args), \
         patch.object(
             eval_mod,
             "load_checkpoint_payload",
             side_effect=[
                 {"model": base_policy.state_dict(), "update": 4, "best_return": 0.1},
                 RuntimeError(f"Failed to load checkpoint {Path('missing-extra.pt').resolve()}: no such file"),
             ],
         ), \
         patch.object(eval_mod, "read_mktd", return_value=_make_data(60)), \
         patch.dict(sys.modules, {"pufferlib_market.binding": fake_binding}):
        with pytest.raises(RuntimeError, match=f"Failed to load checkpoint {Path('missing-extra.pt').resolve()}"):
            eval_mod.main()


def test_main_reports_effective_config_for_extra_checkpoint_members(tmp_path: Path, capsys):
    fake_args = _fake_main_args(tmp_path, arch="mlp", hidden_size=16, extra_checkpoints=["extra.pt"])
    base_policy = eval_mod.TradingPolicy(22, 3, hidden=16)
    extra_policy = eval_mod.ResidualTradingPolicy(22, 3, hidden=32, num_blocks=4)
    fake_binding = SimpleNamespace(shared=lambda **_: None)

    with patch.object(eval_mod.argparse.ArgumentParser, "parse_args", return_value=fake_args), \
         patch.object(
             eval_mod.torch,
             "load",
             side_effect=[
                 {"model": base_policy.state_dict(), "update": 4, "best_return": 0.1},
                 {"model": extra_policy.state_dict(), "update": 5},
             ],
         ), \
         patch.object(eval_mod, "read_mktd", return_value=_make_data(60)), \
         patch.object(eval_mod, "_build_policy_fn", return_value=lambda obs: 0), \
         patch.object(eval_mod, "sliding_window_eval", return_value=[]), \
         patch.object(eval_mod, "aggregate_sliding_results", return_value={"calmar": 1.23, "annualized_return": 0.0, "worst_max_drawdown": 0.0}), \
         patch.object(eval_mod, "print_sliding_results", return_value=None), \
         patch.dict(sys.modules, {"pufferlib_market.binding": fake_binding}):
        eval_mod.main()

    captured = capsys.readouterr()
    assert f"  + ensemble member file: {Path('extra.pt').resolve()}" in captured.out
    assert "  + ensemble member: update=5 train_best_return=? arch=resmlp hidden_size=32" in captured.out
    assert "    Checkpoint overrides CLI policy config: arch mlp -> resmlp, hidden_size 16 -> 32" in captured.out
