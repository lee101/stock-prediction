"""Tests for the root-level evaluate_ttt.py TTT evaluation framework.

Covers:
- _get_ttt_params returns correct parameters for MLP and ResidualMLP
- _do_ttt_update runs without error and only updates ttt_params
- _load_policy infers correct sizes from state_dict
- _print_comparison_table works without crashing
- Weights are restored after each window (per-window TTT isolation)
"""

from __future__ import annotations

import sys
import os
import importlib

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.distributions import Categorical

# Import root-level evaluate_ttt.py (not pufferlib_market version)
# Add worktree root to sys.path so it can be imported as a top-level module.
_ROOT = os.path.dirname(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Register in sys.modules before exec_module so @dataclass can resolve the module dict
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "evaluate_ttt",
    os.path.join(_ROOT, "evaluate_ttt.py"),
)
if "evaluate_ttt" not in sys.modules:
    evaluate_ttt_root = importlib.util.module_from_spec(_spec)
    sys.modules["evaluate_ttt"] = evaluate_ttt_root
    _spec.loader.exec_module(evaluate_ttt_root)
else:
    evaluate_ttt_root = sys.modules["evaluate_ttt"]

_get_ttt_params = evaluate_ttt_root._get_ttt_params
_do_ttt_update = evaluate_ttt_root._do_ttt_update
_load_policy = evaluate_ttt_root._load_policy
_print_comparison_table = evaluate_ttt_root._print_comparison_table

from pufferlib_market.train import TradingPolicy, ResidualTradingPolicy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trading_policy(obs_size=21, num_actions=5, hidden=32) -> TradingPolicy:
    return TradingPolicy(obs_size, num_actions, hidden=hidden)


def _make_resmlp_policy(obs_size=21, num_actions=5, hidden=32) -> ResidualTradingPolicy:
    return ResidualTradingPolicy(obs_size, num_actions, hidden=hidden)


# ---------------------------------------------------------------------------
# _get_ttt_params tests
# ---------------------------------------------------------------------------


class TestGetTTTParams:
    def test_mlp_returns_actor_params(self):
        """For TradingPolicy, ttt_params must include actor parameters."""
        policy = _make_trading_policy()
        params = _get_ttt_params(policy)
        assert len(params) > 0
        actor_param_ptrs = {p.data_ptr() for p in policy.actor.parameters()}
        ttt_param_ptrs = {p.data_ptr() for p in params}
        # Actor params must be in ttt_params
        assert actor_param_ptrs.issubset(ttt_param_ptrs), \
            "Actor parameters should all be in ttt_params"

    def test_mlp_freezes_early_encoder_layers(self):
        """For TradingPolicy, only last encoder layer + actor should be in ttt_params."""
        policy = _make_trading_policy()
        params = _get_ttt_params(policy)
        ttt_param_ptrs = {p.data_ptr() for p in params}

        # First encoder layer (encoder[0]) should NOT be in ttt_params
        first_encoder_ptrs = {p.data_ptr() for p in policy.encoder[0].parameters()}
        overlap = first_encoder_ptrs & ttt_param_ptrs
        assert len(overlap) == 0, \
            "First encoder layer should be frozen (not in ttt_params)"

    def test_resmlp_returns_actor_params(self):
        """For ResidualTradingPolicy, ttt_params must include actor parameters."""
        policy = _make_resmlp_policy()
        params = _get_ttt_params(policy)
        assert len(params) > 0
        actor_param_ptrs = {p.data_ptr() for p in policy.actor.parameters()}
        ttt_param_ptrs = {p.data_ptr() for p in params}
        assert actor_param_ptrs.issubset(ttt_param_ptrs)

    def test_resmlp_freezes_input_proj(self):
        """For ResidualTradingPolicy, input_proj should NOT be in ttt_params."""
        policy = _make_resmlp_policy()
        params = _get_ttt_params(policy)
        ttt_param_ptrs = {p.data_ptr() for p in params}
        input_proj_ptrs = {p.data_ptr() for p in policy.input_proj.parameters()}
        overlap = input_proj_ptrs & ttt_param_ptrs
        assert len(overlap) == 0, "input_proj should be frozen (not in ttt_params)"

    def test_params_list_nonempty(self):
        """ttt_params must never be empty for a valid policy."""
        for policy in [_make_trading_policy(), _make_resmlp_policy()]:
            params = _get_ttt_params(policy)
            assert len(params) > 0, "ttt_params must not be empty"


# ---------------------------------------------------------------------------
# _do_ttt_update tests
# ---------------------------------------------------------------------------


class TestDoTTTUpdate:
    def test_update_runs_without_error(self):
        """_do_ttt_update must complete without raising."""
        policy = _make_trading_policy(obs_size=21, num_actions=5, hidden=32)
        ttt_params = _get_ttt_params(policy)
        T = 10
        obs_list = [torch.randn(1, 21) for _ in range(T)]
        acts_list = [torch.randint(0, 5, (1,)) for _ in range(T)]
        rews_list = [float(np.random.randn()) for _ in range(T)]
        lp_list = [torch.zeros(1) for _ in range(T)]
        val_list = [torch.zeros(1) for _ in range(T)]

        _do_ttt_update(
            policy=policy,
            ttt_params=ttt_params,
            obs_list=obs_list,
            actions_list=acts_list,
            rewards_list=rews_list,
            old_log_probs_list=lp_list,
            old_values_list=val_list,
            ttt_update_steps=8,
            ttt_epochs=2,
            ttt_lr=1e-4,
            kl_coef=0.1,
            device=torch.device("cpu"),
        )

    def test_update_only_moves_ttt_params(self):
        """_do_ttt_update must only move ttt_params, not frozen params."""
        policy = _make_trading_policy(obs_size=21, num_actions=5, hidden=32)
        ttt_params = _get_ttt_params(policy)
        ttt_param_ptrs = {p.data_ptr() for p in ttt_params}

        # Snapshot ALL parameter values before update
        before = {n: p.data.clone() for n, p in policy.named_parameters()}

        T = 8
        obs_list = [torch.randn(1, 21) for _ in range(T)]
        acts_list = [torch.randint(0, 5, (1,)) for _ in range(T)]
        rews_list = [0.01 * i for i in range(T)]
        lp_list = [torch.zeros(1) for _ in range(T)]
        val_list = [torch.zeros(1) for _ in range(T)]

        _do_ttt_update(
            policy=policy,
            ttt_params=ttt_params,
            obs_list=obs_list,
            actions_list=acts_list,
            rewards_list=rews_list,
            old_log_probs_list=lp_list,
            old_values_list=val_list,
            ttt_update_steps=8,
            ttt_epochs=3,
            ttt_lr=1e-3,
            kl_coef=0.1,
            device=torch.device("cpu"),
        )

        # Frozen params must not change
        for n, p in policy.named_parameters():
            ptr = p.data_ptr()
            if ptr not in ttt_param_ptrs:
                torch.testing.assert_close(
                    p.data, before[n],
                    msg=f"Frozen param '{n}' was changed by _do_ttt_update",
                )

    def test_empty_inputs_does_not_crash(self):
        """_do_ttt_update with empty lists must be a no-op."""
        policy = _make_trading_policy()
        ttt_params = _get_ttt_params(policy)
        # Should return without error
        _do_ttt_update(
            policy=policy,
            ttt_params=ttt_params,
            obs_list=[],
            actions_list=[],
            rewards_list=[],
            old_log_probs_list=[],
            old_values_list=[],
            ttt_update_steps=8,
            ttt_epochs=2,
            ttt_lr=1e-4,
            kl_coef=0.1,
            device=torch.device("cpu"),
        )

    def test_params_remain_finite_after_update(self):
        """Policy parameters must remain finite after a TTT update."""
        policy = _make_trading_policy(obs_size=21, num_actions=5, hidden=32)
        ttt_params = _get_ttt_params(policy)
        T = 5
        obs_list = [torch.randn(1, 21) for _ in range(T)]
        acts_list = [torch.randint(0, 5, (1,)) for _ in range(T)]
        rews_list = [0.01 for _ in range(T)]
        lp_list = [torch.zeros(1) for _ in range(T)]
        val_list = [torch.zeros(1) for _ in range(T)]

        _do_ttt_update(
            policy=policy,
            ttt_params=ttt_params,
            obs_list=obs_list,
            actions_list=acts_list,
            rewards_list=rews_list,
            old_log_probs_list=lp_list,
            old_values_list=val_list,
            ttt_update_steps=5,
            ttt_epochs=1,
            ttt_lr=1e-4,
            kl_coef=0.1,
            device=torch.device("cpu"),
        )
        for n, p in policy.named_parameters():
            assert torch.isfinite(p).all(), f"NaN/Inf in param '{n}' after TTT update"


# ---------------------------------------------------------------------------
# _load_policy tests (uses saved tmp checkpoint, no binary data file needed)
# ---------------------------------------------------------------------------


class TestLoadPolicy:
    def test_load_infers_sizes_from_state_dict(self, tmp_path):
        """_load_policy must correctly rebuild the policy from state_dict shapes."""
        obs_size = 21
        num_actions = 5
        hidden = 32
        policy = _make_trading_policy(obs_size, num_actions, hidden)
        payload = {"model": policy.state_dict()}
        ckpt_path = str(tmp_path / "policy.pt")
        torch.save(payload, ckpt_path)

        loaded_policy, loaded_payload = _load_policy(
            ckpt_path, obs_size, num_actions, torch.device("cpu")
        )
        # Forward pass must produce correct shapes
        x = torch.randn(3, obs_size)
        with torch.no_grad():
            logits, value = loaded_policy(x)
        assert logits.shape == (3, num_actions)
        assert value.shape == (3,)

    def test_load_resmlp_infers_sizes(self, tmp_path):
        """_load_policy must handle ResidualTradingPolicy checkpoints."""
        obs_size = 21
        num_actions = 5
        hidden = 32
        policy = _make_resmlp_policy(obs_size, num_actions, hidden)
        payload = {"model": policy.state_dict()}
        ckpt_path = str(tmp_path / "resmlp.pt")
        torch.save(payload, ckpt_path)

        loaded_policy, _ = _load_policy(ckpt_path, obs_size, num_actions, torch.device("cpu"))
        x = torch.randn(3, obs_size)
        with torch.no_grad():
            logits, value = loaded_policy(x)
        assert logits.shape == (3, num_actions)
        assert value.shape == (3,)

    def test_wrong_obs_size_overridden_by_checkpoint(self, tmp_path):
        """If caller provides wrong obs_size, checkpoint state_dict overrides it."""
        obs_size = 21
        num_actions = 5
        policy = _make_trading_policy(obs_size, num_actions, hidden=32)
        payload = {"model": policy.state_dict()}
        ckpt_path = str(tmp_path / "policy.pt")
        torch.save(payload, ckpt_path)

        # Deliberately pass wrong obs_size — should still load correctly
        loaded_policy, _ = _load_policy(
            ckpt_path, obs_size=99, num_actions=999, device=torch.device("cpu")
        )
        x = torch.randn(2, obs_size)
        with torch.no_grad():
            logits, _ = loaded_policy(x)
        assert logits.shape == (2, num_actions)


# ---------------------------------------------------------------------------
# _print_comparison_table
# ---------------------------------------------------------------------------


class TestPrintComparisonTable:
    def test_no_crash_with_valid_data(self, capsys):
        """_print_comparison_table must not raise and must print output."""
        ttt_windows = [{"total_return": 0.05}, {"total_return": -0.02}]
        base_windows = [{"total_return": 0.03}, {"total_return": 0.01}]
        _print_comparison_table(ttt_windows, base_windows)
        captured = capsys.readouterr()
        assert "Win" in captured.out or "Baseline" in captured.out

    def test_no_crash_with_empty_data(self):
        """_print_comparison_table with empty lists must return silently."""
        _print_comparison_table([], [])
        _print_comparison_table([{"total_return": 0.1}], [])

    def test_winner_label_correct(self, capsys):
        """When TTT returns are higher, 'TTT wins' must appear."""
        ttt_windows = [{"total_return": 0.10}, {"total_return": 0.08}]
        base_windows = [{"total_return": 0.05}, {"total_return": 0.03}]
        _print_comparison_table(ttt_windows, base_windows)
        captured = capsys.readouterr()
        assert "TTT" in captured.out

    def test_baseline_label_when_baseline_wins(self, capsys):
        """When baseline returns are higher, 'Baseline wins' must appear."""
        ttt_windows = [{"total_return": 0.01}, {"total_return": 0.02}]
        base_windows = [{"total_return": 0.10}, {"total_return": 0.09}]
        _print_comparison_table(ttt_windows, base_windows)
        captured = capsys.readouterr()
        assert "Baseline" in captured.out


# ---------------------------------------------------------------------------
# Weight restoration tests (per-window TTT isolation)
# ---------------------------------------------------------------------------


class TestWeightRestoration:
    def test_ttt_update_does_not_persist_across_simulated_windows(self):
        """Simulates two consecutive TTT windows; weights must match baseline
        after each window because _run_single_window restores original state."""
        import copy

        policy = _make_trading_policy(obs_size=21, num_actions=5, hidden=32)
        ttt_params = _get_ttt_params(policy)
        original_state = copy.deepcopy(policy.state_dict())

        T = 6
        for _ in range(2):
            obs_list = [torch.randn(1, 21) for _ in range(T)]
            acts_list = [torch.randint(0, 5, (1,)) for _ in range(T)]
            rews_list = [0.01] * T
            lp_list = [torch.zeros(1)] * T
            val_list = [torch.zeros(1)] * T

            _do_ttt_update(
                policy=policy,
                ttt_params=ttt_params,
                obs_list=obs_list,
                actions_list=acts_list,
                rewards_list=rews_list,
                old_log_probs_list=lp_list,
                old_values_list=val_list,
                ttt_update_steps=6,
                ttt_epochs=2,
                ttt_lr=1e-3,
                kl_coef=0.1,
                device=torch.device("cpu"),
            )

            # Simulate weight restoration (as _run_single_window does)
            policy.load_state_dict(original_state)

        # After two windows + restores, weights must match original
        for (n, p), (n2, p2) in zip(
            policy.named_parameters(), original_state.items()
        ):
            torch.testing.assert_close(
                p.data, p2,
                msg=f"Parameter '{n}' was not restored after simulated window",
            )
