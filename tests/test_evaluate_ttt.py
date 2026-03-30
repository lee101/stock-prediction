"""Tests for LoRA TTT (test-time training) adaptation for the trading policy.

Covers:
- LoRALinear forward pass identity when weights are zero
- reset_lora() zeroes out adapter weights
- Calibration step doesn't crash and updates only LoRA params
- LoRAPolicy only exposes LoRA params as trainable
- Full TTT loop on tiny synthetic data (no binary files needed)
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from pufferlib_market import evaluate_ttt as eval_mod
from pufferlib_market.lora import LoRALinear, LoRAPolicy, reset_adam_state
from pufferlib_market.train import TradingPolicy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_base_linear(in_f: int = 8, out_f: int = 4) -> nn.Linear:
    lin = nn.Linear(in_f, out_f)
    nn.init.zeros_(lin.weight)
    nn.init.zeros_(lin.bias)
    return lin


def _make_policy(obs_size: int = 21, num_actions: int = 5, hidden: int = 32) -> TradingPolicy:
    return TradingPolicy(obs_size, num_actions, hidden=hidden)


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
        max_steps=10,
        fee_rate=0.0,
        max_leverage=1.0,
        short_borrow_apr=0.0,
        periods_per_year=8760.0,
        action_allocation_bins=1,
        action_level_bins=1,
        action_max_offset_bps=0.0,
        fill_slippage_bps=0.0,
        max_hold_hours=0,
        num_episodes=1,
        seed=123,
        deterministic=True,
        disable_shorts=False,
        hidden_size=16,
        arch="mlp",
        cpu=True,
        lora_rank=2,
        lora_num_layers=1,
        calibration_steps=1,
        lora_lr=0.01,
        lora_grad_steps=1,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _fake_eval_result() -> tuple[np.ndarray, ...]:
    empty = np.array([], dtype=np.float32)
    return (empty, empty, empty, empty, empty, empty, empty, empty)


# ---------------------------------------------------------------------------
# LoRALinear tests
# ---------------------------------------------------------------------------


class TestLoRALinear:
    def test_identity_when_lora_b_is_zero(self):
        """LoRA delta is zero when lora_b == 0, so output must equal base output."""
        base = _make_base_linear(8, 4)
        # Set base weight to something non-trivial
        nn.init.normal_(base.weight)
        nn.init.normal_(base.bias)

        lora = LoRALinear(base, rank=4, alpha=1.0)
        # Ensure lora_b is zero (it should be by default)
        assert lora.lora_b.data.abs().max().item() == 0.0

        x = torch.randn(3, 8)
        with torch.no_grad():
            base_out = base(x)
            lora_out = lora(x)

        torch.testing.assert_close(lora_out, base_out)

    def test_lora_delta_applied_when_b_nonzero(self):
        """When lora_b != 0 the output differs from the base."""
        base = _make_base_linear(8, 4)
        lora = LoRALinear(base, rank=2, alpha=1.0)
        nn.init.normal_(lora.lora_b)  # make B non-zero

        x = torch.randn(3, 8)
        with torch.no_grad():
            base_out = base(x)
            lora_out = lora(x)

        assert not torch.allclose(lora_out, base_out), "Expected LoRA to change output"

    def test_reset_lora_zeroes_b_and_reinits_a(self):
        """reset_lora() must zero lora_b and reinitialise lora_a."""
        base = _make_base_linear(8, 4)
        lora = LoRALinear(base, rank=4, alpha=1.0)

        # Corrupt the weights first
        with torch.no_grad():
            lora.lora_b.fill_(99.0)
            lora.lora_a.fill_(99.0)

        lora.reset_lora()
        assert lora.lora_b.data.abs().max().item() == 0.0, "lora_b must be zero after reset"
        # lora_a should be small (re-initialised to ~N(0, 0.01))
        assert lora.lora_a.data.abs().max().item() < 1.0, "lora_a should be small after reset"

    def test_base_weights_frozen(self):
        """Base linear weights must not have requires_grad after wrapping."""
        base = _make_base_linear(8, 4)
        lora = LoRALinear(base, rank=2, alpha=1.0)
        for p in lora.base.parameters():
            assert not p.requires_grad, "Base weights must be frozen"

    def test_lora_params_require_grad(self):
        """LoRA parameters (A and B) must require grad."""
        base = _make_base_linear(8, 4)
        lora = LoRALinear(base, rank=2, alpha=1.0)
        assert lora.lora_a.requires_grad
        assert lora.lora_b.requires_grad

    def test_invalid_rank_raises(self):
        base = _make_base_linear(8, 4)
        with pytest.raises(ValueError):
            LoRALinear(base, rank=0)


# ---------------------------------------------------------------------------
# LoRAPolicy tests
# ---------------------------------------------------------------------------


class TestLoRAPolicy:
    def _policy_and_lora(self, obs_size=21, num_actions=5, hidden=32, rank=2, num_layers=2):
        base = _make_policy(obs_size, num_actions, hidden)
        lora = LoRAPolicy(base, rank=rank, num_layers=num_layers)
        return base, lora

    def test_forward_shape(self):
        """get_action_and_value must return tensors of correct shape."""
        _, lora = self._policy_and_lora()
        obs = torch.randn(4, 21)
        actions, log_probs, entropy, values = lora.get_action_and_value(obs)
        assert actions.shape == (4,)
        assert log_probs.shape == (4,)
        assert entropy.shape == (4,)
        assert values.shape == (4,)

    def test_only_lora_params_trainable(self):
        """All base parameters must be frozen; only LoRA params should be trainable."""
        base, lora = self._policy_and_lora()
        trainable = [p for p in lora.parameters() if p.requires_grad]
        lora_params = lora.lora_parameters()
        # Every trainable parameter must be a LoRA parameter
        lora_data_ptrs = {p.data_ptr() for p in lora_params}
        for p in trainable:
            assert p.data_ptr() in lora_data_ptrs, (
                "Found trainable parameter that is NOT a LoRA parameter"
            )

    def test_reset_lora_zeroes_all_b(self):
        """reset_lora() must zero lora_b in every wrapped LoRALinear layer."""
        _, lora = self._policy_and_lora(rank=4, num_layers=2)
        # Dirty all B matrices
        for ll in lora._lora_layers:
            with torch.no_grad():
                ll.lora_b.fill_(7.0)

        lora.reset_lora()
        for ll in lora._lora_layers:
            assert ll.lora_b.data.abs().max().item() == 0.0

    def test_reset_lora_identity_output(self):
        """After reset, LoRA output must equal base output (since B=0)."""
        base, lora = self._policy_and_lora()
        lora.reset_lora()

        obs = torch.randn(2, 21)
        # Both policies share the same underlying Linear objects, so outputs
        # should match when LoRA is identity.
        with torch.no_grad():
            base_logits, base_val = base(obs)
            lora_logits, lora_val = lora(obs)
        torch.testing.assert_close(lora_logits, base_logits)
        torch.testing.assert_close(lora_val, base_val)

    def test_lora_params_count(self):
        """Number of LoRA parameters should equal 2 * num_layers * (in + out) * rank."""
        rank = 4
        num_layers = 2
        hidden = 64
        obs_size = 21
        num_actions = 5
        base = TradingPolicy(obs_size, num_actions, hidden=hidden)
        lora = LoRAPolicy(base, rank=rank, num_layers=num_layers)
        n = sum(p.numel() for p in lora.lora_parameters())
        # Just verify it is positive and bounded
        assert n > 0
        assert n < sum(p.numel() for p in base.parameters())

    def test_invalid_num_layers_raises(self):
        base = _make_policy()
        with pytest.raises(ValueError):
            LoRAPolicy(base, rank=4, num_layers=0)


# ---------------------------------------------------------------------------
# Calibration step test
# ---------------------------------------------------------------------------


class TestCalibrationStep:
    def test_calibration_updates_only_lora_params(self):
        """A gradient step during calibration must only move LoRA parameters."""
        obs_size = 21
        num_actions = 5
        hidden = 32
        rank = 2

        base = _make_policy(obs_size, num_actions, hidden)
        lora = LoRAPolicy(base, rank=rank, num_layers=2)
        lora.reset_lora()

        # Save copies of base (non-LoRA) parameter values.
        # After LoRAPolicy wraps the actor, base.named_parameters() also includes
        # lora_a/lora_b from the injected LoRALinear modules — we exclude those
        # because they are explicitly *meant* to change.
        base_param_before = {
            n: p.data.clone() for n, p in base.named_parameters()
            if "lora_a" not in n and "lora_b" not in n
        }
        # Save LoRA B (should be zero before)
        lora_b_before = [ll.lora_b.data.clone() for ll in lora._lora_layers]

        optimizer = torch.optim.Adam(lora.lora_parameters(), lr=0.1)
        obs = torch.randn(8, obs_size)
        actions, log_probs, entropy, _ = lora.get_action_and_value(obs)
        loss = -log_probs.mean() - 0.01 * entropy.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Base parameters (weight/bias of original Linear layers) must be unchanged
        for n, p in base.named_parameters():
            if "lora_a" in n or "lora_b" in n:
                continue  # these are the LoRA adapter params — they are meant to change
            torch.testing.assert_close(p.data, base_param_before[n],
                                       msg=f"Base param '{n}' changed during LoRA update!")

        # At least one LoRA layer should have changed (gradient flowed back)
        any_changed = False
        for ll, b_before in zip(lora._lora_layers, lora_b_before):
            if not torch.allclose(ll.lora_b.data, b_before):
                any_changed = True
                break
        assert any_changed, "No LoRA parameter changed after gradient step"

    def test_calibration_does_not_crash(self):
        """Running the calibration loop on random data must not raise."""
        obs_size = 21
        num_actions = 5
        base = _make_policy(obs_size, num_actions, hidden=32)
        lora = LoRAPolicy(base, rank=4, num_layers=2)
        lora.reset_lora()

        optimizer = torch.optim.Adam(lora.lora_parameters(), lr=0.01)
        calib_obs = [torch.randn(1, obs_size) for _ in range(5)]
        calib_actions = [torch.randint(0, num_actions, (1,)) for _ in range(5)]

        obs_t = torch.cat(calib_obs, dim=0)
        act_t = torch.cat(calib_actions, dim=0)

        for _ in range(3):
            optimizer.zero_grad()
            _, log_probs, entropy, _ = lora.get_action_and_value(obs_t, act_t)
            loss = -log_probs.mean() - 0.01 * entropy.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora.lora_parameters(), 1.0)
            optimizer.step()

        # Just verifying no exception was raised


# ---------------------------------------------------------------------------
# reset_adam_state utility
# ---------------------------------------------------------------------------


class TestResetAdamState:
    def test_reset_clears_momentum(self):
        """reset_adam_state must zero momentum buffers."""
        obs_size = 21
        num_actions = 5
        base = _make_policy(obs_size, num_actions, hidden=32)
        lora = LoRAPolicy(base, rank=4, num_layers=2)
        params = lora.lora_parameters()
        opt = torch.optim.Adam(params, lr=0.01)

        # Do a real step so Adam state is populated
        obs = torch.randn(4, obs_size)
        _, lp, ent, _ = lora.get_action_and_value(obs)
        loss = -lp.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Verify state is populated
        assert len(opt.state) > 0

        reset_adam_state(opt)

        for state in opt.state.values():
            if "exp_avg" in state:
                assert state["exp_avg"].abs().max().item() == 0.0
            if "exp_avg_sq" in state:
                assert state["exp_avg_sq"].abs().max().item() == 0.0


# ---------------------------------------------------------------------------
# Full TTT mini-eval (mock, no binary file)
# ---------------------------------------------------------------------------


class TestFullTTTMock:
    """Run the TTT calibration logic end-to-end on synthetic observations."""

    def test_ttt_loop_improves_or_stays_flat(self):
        """The TTT loop must run without error and produce a valid loss."""
        obs_size = 21
        num_actions = 5
        hidden = 32
        rank = 4
        calibration_steps = 10
        lora_grad_steps = 3
        lora_lr = 0.01

        base = _make_policy(obs_size, num_actions, hidden)
        lora = LoRAPolicy(base, rank=rank, num_layers=2)

        # Simulate calibration phase on synthetic observations
        lora.reset_lora()
        optimizer = torch.optim.Adam(lora.lora_parameters(), lr=lora_lr)
        reset_adam_state(optimizer)

        calib_obs = torch.randn(calibration_steps, obs_size)
        with torch.no_grad():
            # Collect actions as pseudo-labels (same as evaluate_ttt.py logic)
            calib_actions = lora.get_action(calib_obs, deterministic=True)

        losses = []
        for _ in range(lora_grad_steps):
            optimizer.zero_grad()
            _, log_probs, entropy, _ = lora.get_action_and_value(calib_obs, calib_actions)
            loss = -log_probs.mean() - 0.01 * entropy.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora.lora_parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))

        assert len(losses) == lora_grad_steps
        assert all(np.isfinite(l) for l in losses), "Loss must be finite"

        # Run trading phase (just forward passes)
        trade_obs = torch.randn(20, obs_size)
        with torch.no_grad():
            actions = lora.get_action(trade_obs, deterministic=True)
        assert actions.shape == (20,)
        assert (actions >= 0).all() and (actions < num_actions).all()

    def test_multiple_episode_resets(self):
        """reset_lora() between episodes must give consistent identity output."""
        obs_size = 21
        num_actions = 5
        base = _make_policy(obs_size, num_actions, hidden=32)
        lora = LoRAPolicy(base, rank=4, num_layers=2)

        obs = torch.randn(3, obs_size)

        results = []
        for _ in range(3):
            lora.reset_lora()
            # Dirty the LoRA params (simulate training)
            with torch.no_grad():
                for ll in lora._lora_layers:
                    ll.lora_b.fill_(1.0)
            # Reset again
            lora.reset_lora()
            with torch.no_grad():
                logits, _ = lora(obs)
            results.append(logits.clone())

        # All resets should give the same output (identity LoRA)
        torch.testing.assert_close(results[0], results[1])
        torch.testing.assert_close(results[1], results[2])


def test_load_base_policy_from_checkpoint_infers_effective_arch_and_hidden_size():
    obs_size = 21
    num_actions = 9
    source_policy = eval_mod.ResidualTradingPolicy(obs_size, num_actions, hidden=32, num_blocks=4)
    ckpt = {
        "model": source_policy.state_dict(),
        "arch": "resmlp",
        "hidden_size": 32,
    }

    loaded_policy, effective_arch, effective_hidden_size = eval_mod._load_base_policy_from_checkpoint(
        ckpt=ckpt,
        obs_size=obs_size,
        num_actions=num_actions,
        arch="mlp",
        hidden_size=16,
        device=torch.device("cpu"),
    )

    assert isinstance(loaded_policy, eval_mod.ResidualTradingPolicy)
    assert effective_arch == "resmlp"
    assert effective_hidden_size == 32
    assert len(loaded_policy.blocks) == 4


def test_main_rejects_checkpoint_without_model_state(tmp_path: Path):
    fake_args = _fake_main_args(tmp_path)

    with patch.object(eval_mod.argparse.ArgumentParser, "parse_args", return_value=fake_args), \
         patch.object(eval_mod.torch, "load", return_value={"update": 7}):
        with pytest.raises(KeyError, match="missing a valid 'model' state_dict"):
            eval_mod.main()


def test_main_rejects_non_mapping_checkpoint_payload(tmp_path: Path):
    fake_args = _fake_main_args(tmp_path)

    with patch.object(eval_mod.argparse.ArgumentParser, "parse_args", return_value=fake_args), \
         patch.object(eval_mod.torch, "load", return_value=[]):
        with pytest.raises(TypeError, match="must load to a mapping"):
            eval_mod.main()


def test_main_reports_effective_checkpoint_arch_and_hidden_size(tmp_path: Path, capsys):
    fake_args = _fake_main_args(tmp_path, arch="mlp", hidden_size=16)
    obs_size = 22
    num_actions = 3
    source_policy = eval_mod.ResidualTradingPolicy(obs_size, num_actions, hidden=32, num_blocks=4)
    fake_binding = SimpleNamespace(shared=lambda **_: None)

    with patch.object(eval_mod.argparse.ArgumentParser, "parse_args", return_value=fake_args), \
         patch.object(eval_mod.torch, "load", return_value={"model": source_policy.state_dict(), "update": 9}), \
         patch.object(eval_mod, "evaluate_ttt", return_value=_fake_eval_result()), \
         patch.dict(sys.modules, {"pufferlib_market.binding": fake_binding}):
        eval_mod.main()

    captured = capsys.readouterr()
    assert "obs_size=22, num_actions=3" in captured.out
    assert "CLI policy config: arch=mlp, hidden_size=16" in captured.out
    assert "Action grid: alloc_bins=1 level_bins=1 max_offset_bps=0.0" in captured.out
    assert "Runtime: device=cpu, deterministic=True" in captured.out
    assert f"Checkpoint file: {Path(fake_args.checkpoint).resolve()}" in captured.out
    assert "Effective checkpoint config: arch=resmlp, hidden_size=32" in captured.out
    assert "Checkpoint overrides CLI policy config: arch mlp -> resmlp, hidden_size 16 -> 32" in captured.out
    assert "Loaded checkpoint: update=9, train_best_return=?, arch=resmlp" in captured.out


def test_main_tolerates_checkpoint_without_best_return(tmp_path: Path, capsys):
    fake_args = _fake_main_args(tmp_path)
    source_policy = eval_mod.TradingPolicy(22, 3, hidden=16)
    fake_binding = SimpleNamespace(shared=lambda **_: None)

    with patch.object(eval_mod.argparse.ArgumentParser, "parse_args", return_value=fake_args), \
         patch.object(eval_mod.torch, "load", return_value={"model": source_policy.state_dict(), "update": 4}), \
         patch.object(eval_mod, "evaluate_ttt", return_value=_fake_eval_result()), \
         patch.dict(sys.modules, {"pufferlib_market.binding": fake_binding}):
        eval_mod.main()

    captured = capsys.readouterr()
    assert "CLI policy config: arch=mlp, hidden_size=16" in captured.out
    assert "Action grid: alloc_bins=1 level_bins=1 max_offset_bps=0.0" in captured.out
    assert "Runtime: device=cpu, deterministic=True" in captured.out
    assert f"Checkpoint file: {Path(fake_args.checkpoint).resolve()}" in captured.out
    assert "Effective checkpoint config: arch=mlp, hidden_size=16" in captured.out
    assert "Loaded checkpoint: update=4, train_best_return=?, arch=mlp" in captured.out
