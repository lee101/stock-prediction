from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from pufferlib_market import evaluate_tail as eval_mod
from pufferlib_market.evaluate_tail import TradingPolicy, load_policy
from pufferlib_market.hourly_replay import MktdData


@pytest.mark.unit
def test_load_policy_accepts_bare_state_dict_and_ignores_missing_encoder_norm_keys(tmp_path: Path) -> None:
    obs_size = 21
    hidden = 8
    policy = TradingPolicy(obs_size, 3, hidden=hidden)
    state_dict = policy.state_dict()
    state_dict.pop("encoder_norm.weight")
    state_dict.pop("encoder_norm.bias")
    checkpoint_path = tmp_path / "legacy_tail.pt"
    torch.save(state_dict, checkpoint_path)

    loaded = load_policy(
        checkpoint_path=checkpoint_path,
        obs_size=obs_size,
        num_symbols=1,
        arch="mlp",
        hidden_size=hidden,
        device=torch.device("cpu"),
    )

    assert isinstance(loaded.policy, TradingPolicy)
    assert loaded.action_allocation_bins == 1
    assert loaded.action_level_bins == 1
    assert loaded.action_max_offset_bps == 0.0
    assert loaded.policy._use_encoder_norm is False
    assert loaded.action_allocation_bins == 1
    assert loaded.action_level_bins == 1
    assert loaded.action_max_offset_bps == 0.0


@pytest.mark.unit
def test_load_policy_rejects_wrapped_checkpoint_with_invalid_model_payload(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "invalid_tail.pt"
    torch.save({"model": "bad-payload"}, checkpoint_path)

    with pytest.raises(KeyError, match="missing a valid 'model' state_dict"):
        load_policy(
            checkpoint_path=checkpoint_path,
            obs_size=21,
            num_symbols=1,
            arch="mlp",
            hidden_size=8,
            device=torch.device("cpu"),
        )


@pytest.mark.unit
def test_load_policy_ignores_invalid_action_grid_metadata(tmp_path: Path) -> None:
    obs_size = 21
    hidden = 8
    policy = TradingPolicy(obs_size, 3, hidden=hidden)
    checkpoint_path = tmp_path / "invalid_grid_tail.pt"
    torch.save(
        {
            "model": policy.state_dict(),
            "action_allocation_bins": "2.5",
            "action_level_bins": "nan",
        },
        checkpoint_path,
    )

    loaded = load_policy(
        checkpoint_path=checkpoint_path,
        obs_size=obs_size,
        num_symbols=1,
        arch="mlp",
        hidden_size=hidden,
        device=torch.device("cpu"),
    )

    assert isinstance(loaded.policy, TradingPolicy)


@pytest.mark.unit
def test_load_policy_accepts_action_grid_checkpoint(tmp_path: Path) -> None:
    obs_size = 21
    hidden = 8
    policy = TradingPolicy(obs_size, 5, hidden=hidden)
    checkpoint_path = tmp_path / "grid_tail.pt"
    torch.save(
        {
            "model": policy.state_dict(),
            "action_allocation_bins": 2,
            "action_level_bins": 1,
            "action_max_offset_bps": 12.5,
        },
        checkpoint_path,
    )

    loaded = load_policy(
        checkpoint_path=checkpoint_path,
        obs_size=obs_size,
        num_symbols=1,
        arch="mlp",
        hidden_size=hidden,
        device=torch.device("cpu"),
    )

    assert loaded.num_actions == 5
    assert loaded.per_symbol_actions == 2
    assert loaded.action_allocation_bins == 2
    assert loaded.action_level_bins == 1
    assert loaded.action_max_offset_bps == pytest.approx(12.5)



def _fake_main_args(tmp_path: Path, **overrides) -> SimpleNamespace:
    values = {
        "checkpoint": str(tmp_path / "checkpoint.pt"),
        "data_path": str(tmp_path / "data.mktd"),
        "eval_hours": 24,
        "fee_rate": 0.001,
        "fill_buffer_bps": 5.0,
        "max_leverage": 1.0,
        "periods_per_year": 8760.0,
        "short_borrow_apr": 0.0,
        "arch": "auto",
        "hidden_size": None,
        "disable_shorts": False,
        "shortable_symbols": None,
        "decision_lag": 0,
        "deterministic": True,
        "device": "cpu",
    }
    values.update(overrides)
    return SimpleNamespace(**values)



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


@pytest.mark.unit
def test_main_supports_legacy_bare_state_dict_checkpoint(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fake_args = _fake_main_args(tmp_path)
    source_policy = eval_mod.TradingPolicy(obs_size=22, num_actions=3, hidden=8)
    state_dict = source_policy.state_dict()
    state_dict.pop("encoder_norm.weight")
    state_dict.pop("encoder_norm.bias")
    fake_result = SimpleNamespace(
        total_return=0.12,
        sortino=1.25,
        max_drawdown=0.08,
        num_trades=4,
        win_rate=0.75,
        avg_hold_steps=3.5,
    )

    with (
        patch.object(eval_mod.argparse.ArgumentParser, "parse_args", return_value=fake_args),
        patch.object(eval_mod, "load_checkpoint_payload", return_value=state_dict),
        patch.object(eval_mod, "read_mktd", return_value=_make_data(40)),
        patch.object(eval_mod, "simulate_daily_policy", return_value=fake_result),
        patch.object(eval_mod, "annualize_total_return", return_value=0.34),
    ):
        eval_mod.main()

    out = json.loads(capsys.readouterr().out)
    assert out["arch"] == "mlp"
    assert out["hidden_size"] == 8
    assert out["action_allocation_bins"] == 1
    assert out["action_level_bins"] == 1
    assert out["action_max_offset_bps"] == pytest.approx(0.0)
    assert out["decision_lag"] == 0
    assert out["total_return"] == pytest.approx(0.12)
    assert out["annualized_return"] == pytest.approx(0.34)
    assert out["symbols"] == ["SYM0"]
    assert out["summary"]["total_return"] == pytest.approx(0.12)
    assert out["summary"]["annualized_return"] == pytest.approx(0.34)
    assert out["summary"]["num_trades"] == 4


@pytest.mark.unit
def test_main_forwards_action_grid_checkpoint_metadata(tmp_path: Path) -> None:
    fake_args = _fake_main_args(tmp_path)
    source_policy = eval_mod.TradingPolicy(obs_size=22, num_actions=5, hidden=8)
    captured: dict[str, object] = {}

    def _fake_simulate_daily_policy(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            total_return=0.0,
            sortino=0.0,
            max_drawdown=0.0,
            num_trades=0,
            win_rate=0.0,
            avg_hold_steps=0.0,
        )

    with (
        patch.object(eval_mod.argparse.ArgumentParser, "parse_args", return_value=fake_args),
        patch.object(
            eval_mod,
            "load_checkpoint_payload",
            return_value={
                "model": source_policy.state_dict(),
                "action_allocation_bins": 2,
                "action_level_bins": 1,
                "action_max_offset_bps": 7.5,
            },
        ),
        patch.object(eval_mod, "read_mktd", return_value=_make_data(40)),
        patch.object(eval_mod, "simulate_daily_policy", side_effect=_fake_simulate_daily_policy),
        patch.object(eval_mod, "annualize_total_return", return_value=0.0),
    ):
        eval_mod.main()

    assert captured["action_allocation_bins"] == 2
    assert captured["action_level_bins"] == 1
    assert captured["action_max_offset_bps"] == pytest.approx(7.5)
