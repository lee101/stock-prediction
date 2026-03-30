from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from unittest.mock import patch

from pufferlib_market.checkpoint_loader import (
    build_action_grid_summary_line,
    build_cli_policy_config_line,
    build_ensemble_member_summary_lines,
    build_checkpoint_summary_lines,
    ensure_checkpoint_action_grid_compatible,
    build_runtime_summary_line,
    format_action_grid_override_note,
    format_checkpoint_override_note,
    format_best_return_label,
    load_checkpoint_payload,
    load_policy_from_checkpoint,
    resolve_checkpoint_action_grid_config,
    resolve_checkpoint_policy_details,
    resolve_checkpoint_policy_metadata,
)


class _FakeLoadResult:
    def __init__(self, missing=(), unexpected=()):
        self.missing_keys = list(missing)
        self.unexpected_keys = list(unexpected)


class _FakePolicy(nn.Module):
    def __init__(
        self,
        obs_size: int,
        num_actions: int,
        hidden_size: int,
        *,
        num_blocks: int | None = None,
        missing=(),
        unexpected=(),
    ):
        super().__init__()
        self.obs_size = obs_size
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self._use_encoder_norm = True
        self.obs_mean = None
        self.obs_std = None
        self.eval_called = False
        self.loaded_state_dict = None
        self._load_result = _FakeLoadResult(missing=missing, unexpected=unexpected)

    def load_state_dict(self, state_dict, strict=False):
        self.loaded_state_dict = state_dict
        return self._load_result

    def eval(self):
        self.eval_called = True
        return self


def test_format_best_return_label_handles_numeric_and_missing_values():
    assert format_best_return_label({"best_return": np.float32(1.25)}) == "1.2500"
    assert format_best_return_label({"best_return": np.float64(np.nan)}) == "?"
    assert format_best_return_label({"best_return": np.float64(np.inf)}) == "?"
    assert format_best_return_label({}) == "?"


def test_format_checkpoint_override_note_reports_only_changed_fields():
    assert format_checkpoint_override_note(
        requested_arch="mlp",
        requested_hidden_size=16,
        effective_arch="resmlp",
        effective_hidden_size=32,
    ) == "Checkpoint overrides CLI policy config: arch mlp -> resmlp, hidden_size 16 -> 32"
    assert format_checkpoint_override_note(
        requested_arch="resmlp",
        requested_hidden_size=32,
        effective_arch="resmlp",
        effective_hidden_size=32,
    ) is None


def test_build_cli_runtime_and_action_grid_summary_lines_keep_consistent_contract():
    assert build_cli_policy_config_line(arch="mlp", hidden_size=16) == "CLI policy config: arch=mlp, hidden_size=16"
    assert (
        build_action_grid_summary_line(
            action_allocation_bins=2,
            action_level_bins=3,
            action_max_offset_bps=12.5,
        )
        == "Action grid: alloc_bins=2 level_bins=3 max_offset_bps=12.5"
    )
    assert (
        build_action_grid_summary_line(
            action_allocation_bins=2,
            action_level_bins=3,
            action_max_offset_bps=0.04,
        )
        == "Action grid: alloc_bins=2 level_bins=3 max_offset_bps=0.04"
    )
    assert (
        build_runtime_summary_line(device=torch.device("cpu"), deterministic=False)
        == "Runtime: device=cpu, deterministic=False"
    )


def test_format_action_grid_override_note_reports_only_changed_fields():
    assert format_action_grid_override_note(
        requested_allocation_bins=1,
        requested_level_bins=1,
        requested_max_offset_bps=0.0,
        effective_allocation_bins=2,
        effective_level_bins=3,
        effective_max_offset_bps=12.5,
    ) == "Checkpoint overrides CLI action grid: alloc_bins 1 -> 2, level_bins 1 -> 3, max_offset_bps 0.0 -> 12.5"
    assert format_action_grid_override_note(
        requested_allocation_bins=2,
        requested_level_bins=3,
        requested_max_offset_bps=0.0,
        effective_allocation_bins=2,
        effective_level_bins=3,
        effective_max_offset_bps=0.04,
    ) == "Checkpoint overrides CLI action grid: max_offset_bps 0.0 -> 0.04"
    assert format_action_grid_override_note(
        requested_allocation_bins=2,
        requested_level_bins=3,
        requested_max_offset_bps=12.5,
        effective_allocation_bins=2,
        effective_level_bins=3,
        effective_max_offset_bps=12.5,
    ) is None


def test_build_checkpoint_summary_lines_keeps_consistent_cli_contract(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pt"

    assert build_checkpoint_summary_lines(
        ckpt={"update": 9},
        requested_arch="mlp",
        requested_hidden_size=16,
        effective_arch="resmlp",
        effective_hidden_size=32,
        checkpoint_path=checkpoint_path,
    ) == [
        f"Checkpoint file: {checkpoint_path.resolve()}",
        "Effective checkpoint config: arch=resmlp, hidden_size=32",
        "Checkpoint overrides CLI policy config: arch mlp -> resmlp, hidden_size 16 -> 32",
        "Loaded checkpoint: update=9, train_best_return=?, arch=resmlp",
    ]


def test_build_ensemble_member_summary_lines_keeps_consistent_cli_contract(tmp_path):
    checkpoint_path = tmp_path / "member.pt"

    assert build_ensemble_member_summary_lines(
        ckpt={"update": 5},
        requested_arch="mlp",
        requested_hidden_size=16,
        effective_arch="resmlp",
        effective_hidden_size=32,
        checkpoint_path=checkpoint_path,
    ) == [
        f"  + ensemble member file: {checkpoint_path.resolve()}",
        "  + ensemble member: update=5 train_best_return=? arch=resmlp hidden_size=32",
        "    Checkpoint overrides CLI policy config: arch mlp -> resmlp, hidden_size 16 -> 32",
    ]


def test_load_checkpoint_payload_wraps_path_and_original_error(tmp_path):
    checkpoint_path = tmp_path / "missing.pt"

    with patch("pufferlib_market.checkpoint_loader.torch.load", side_effect=FileNotFoundError("no such file")):
        with pytest.raises(RuntimeError, match=f"Failed to load checkpoint {checkpoint_path.resolve()}") as excinfo:
            load_checkpoint_payload(checkpoint_path, map_location="cpu")

    assert isinstance(excinfo.value.__cause__, FileNotFoundError)


def test_ensure_checkpoint_action_grid_compatible_rejects_mismatched_values(tmp_path):
    checkpoint_path = tmp_path / "extra.pt"

    with pytest.raises(RuntimeError, match=f"Checkpoint {checkpoint_path.resolve()} is incompatible with ensemble action grid") as excinfo:
        ensure_checkpoint_action_grid_compatible(
            {
                "action_allocation_bins": 2,
                "action_level_bins": 3,
                "action_max_offset_bps": 12.5,
            },
            checkpoint_path=checkpoint_path,
            expected_action_allocation_bins=1,
            expected_action_level_bins=1,
            expected_action_max_offset_bps=0.0,
        )

    assert "expected alloc_bins=1 level_bins=1 max_offset_bps=0.0" in str(excinfo.value)
    assert "got alloc_bins=2 level_bins=3 max_offset_bps=12.5" in str(excinfo.value)


def test_resolve_checkpoint_action_grid_config_uses_checkpoint_when_values_are_finite():
    assert resolve_checkpoint_action_grid_config(
        {
            "action_allocation_bins": "2",
            "action_level_bins": np.float64(3.0),
            "action_max_offset_bps": "12.5",
        },
        action_allocation_bins=1,
        action_level_bins=1,
        action_max_offset_bps=0.0,
    ) == (2, 3, 12.5)


def test_resolve_checkpoint_action_grid_config_ignores_invalid_checkpoint_values():
    assert resolve_checkpoint_action_grid_config(
        {
            "action_allocation_bins": np.float64(np.nan),
            "action_level_bins": -4,
            "action_max_offset_bps": "not-a-number",
        },
        action_allocation_bins=2,
        action_level_bins=3,
        action_max_offset_bps=5.0,
    ) == (2, 3, 5.0)


def test_resolve_checkpoint_action_grid_config_rejects_non_integral_bin_values():
    assert resolve_checkpoint_action_grid_config(
        {
            "action_allocation_bins": "2.5",
            "action_level_bins": 3.2,
            "action_max_offset_bps": 7.5,
        },
        action_allocation_bins=2,
        action_level_bins=3,
        action_max_offset_bps=5.0,
    ) == (2, 3, 7.5)


def test_resolve_checkpoint_policy_metadata_prefers_inferred_state_dict_shape():
    state_dict = {
        "input_proj.weight": torch.zeros(32, 21),
        "blocks.0.norm.weight": torch.ones(32),
    }

    resolved_state_dict, effective_arch, effective_hidden_size = resolve_checkpoint_policy_metadata(
        {"model": state_dict, "arch": "mlp", "hidden_size": 16},
        arch="mlp",
        hidden_size=16,
    )

    assert resolved_state_dict is state_dict
    assert effective_arch == "resmlp"
    assert effective_hidden_size == 32


def test_resolve_checkpoint_policy_metadata_preserves_checkpoint_declared_special_arch():
    state_dict = {"input_proj.weight": torch.zeros(64, 21)}

    resolved_state_dict, effective_arch, effective_hidden_size = resolve_checkpoint_policy_metadata(
        {"model": state_dict, "arch": "gru", "hidden_size": 64},
        arch="mlp",
        hidden_size=16,
    )

    assert resolved_state_dict is state_dict
    assert effective_arch == "gru"
    assert effective_hidden_size == 64


def test_resolve_checkpoint_policy_details_caches_resmlp_block_count():
    state_dict = {
        "input_proj.weight": torch.zeros(32, 21),
        "blocks.0.norm.weight": torch.ones(32),
        "blocks.3.norm.weight": torch.ones(32),
    }

    resolved = resolve_checkpoint_policy_details(
        {"model": state_dict, "arch": "mlp", "hidden_size": 16},
        arch="mlp",
        hidden_size=16,
    )

    assert resolved.effective_arch == "resmlp"
    assert resolved.effective_hidden_size == 32
    assert resolved.effective_resmlp_blocks == 4


def test_load_policy_from_checkpoint_prefers_inferred_resmlp_metadata():
    state_dict = {
        "input_proj.weight": torch.zeros(32, 21),
        "blocks.0.norm.weight": torch.ones(32),
        "blocks.3.norm.weight": torch.ones(32),
    }

    policy, effective_arch, effective_hidden_size = load_policy_from_checkpoint(
        ckpt={"model": state_dict, "arch": "mlp", "hidden_size": 16},
        obs_size=21,
        num_actions=9,
        arch="mlp",
        hidden_size=16,
        device=torch.device("cpu"),
        mlp_factory=lambda obs, acts, hidden: _FakePolicy(obs, acts, hidden),
        resmlp_factory=lambda obs, acts, hidden, num_blocks: _FakePolicy(
            obs, acts, hidden, num_blocks=num_blocks
        ),
    )

    assert effective_arch == "resmlp"
    assert effective_hidden_size == 32
    assert isinstance(policy, _FakePolicy)
    assert policy.hidden_size == 32
    assert policy.num_blocks == 4
    assert policy.eval_called is True


def test_load_policy_from_checkpoint_reuses_cached_resmlp_block_count():
    state_dict = {
        "input_proj.weight": torch.zeros(32, 21),
        "blocks.0.norm.weight": torch.ones(32),
        "blocks.3.norm.weight": torch.ones(32),
    }

    with patch("pufferlib_market.checkpoint_loader.infer_resmlp_blocks_from_state_dict", return_value=4) as infer_blocks:
        policy, effective_arch, effective_hidden_size = load_policy_from_checkpoint(
            ckpt={"model": state_dict, "arch": "mlp", "hidden_size": 16},
            obs_size=21,
            num_actions=9,
            arch="mlp",
            hidden_size=16,
            device=torch.device("cpu"),
            mlp_factory=lambda obs, acts, hidden: _FakePolicy(obs, acts, hidden),
            resmlp_factory=lambda obs, acts, hidden, num_blocks: _FakePolicy(
                obs, acts, hidden, num_blocks=num_blocks
            ),
        )

    assert effective_arch == "resmlp"
    assert effective_hidden_size == 32
    assert isinstance(policy, _FakePolicy)
    assert policy.num_blocks == 4
    assert infer_blocks.call_count == 1


def test_load_policy_from_checkpoint_uses_checkpoint_hidden_size_fallback():
    state_dict = {"unexpected.key": torch.tensor([1.0])}

    policy, effective_arch, effective_hidden_size = load_policy_from_checkpoint(
        ckpt={"model": state_dict, "arch": "mlp", "hidden_size": 64},
        obs_size=21,
        num_actions=9,
        arch="mlp",
        hidden_size=16,
        device=torch.device("cpu"),
        mlp_factory=lambda obs, acts, hidden: _FakePolicy(obs, acts, hidden),
        resmlp_factory=lambda obs, acts, hidden, num_blocks: _FakePolicy(
            obs, acts, hidden, num_blocks=num_blocks
        ),
    )

    assert effective_arch == "mlp"
    assert effective_hidden_size == 64
    assert policy.hidden_size == 64


def test_load_policy_from_checkpoint_ignores_non_finite_hidden_size_fallback():
    state_dict = {"unexpected.key": torch.tensor([1.0])}

    policy, effective_arch, effective_hidden_size = load_policy_from_checkpoint(
        ckpt={"model": state_dict, "arch": "mlp", "hidden_size": np.float64(np.nan)},
        obs_size=21,
        num_actions=9,
        arch="mlp",
        hidden_size=16,
        device=torch.device("cpu"),
        mlp_factory=lambda obs, acts, hidden: _FakePolicy(obs, acts, hidden),
        resmlp_factory=lambda obs, acts, hidden, num_blocks: _FakePolicy(
            obs, acts, hidden, num_blocks=num_blocks
        ),
    )

    assert effective_arch == "mlp"
    assert effective_hidden_size == 16
    assert policy.hidden_size == 16


def test_load_policy_from_checkpoint_ignores_non_integral_hidden_size_fallback():
    state_dict = {"unexpected.key": torch.tensor([1.0])}

    policy, effective_arch, effective_hidden_size = load_policy_from_checkpoint(
        ckpt={"model": state_dict, "arch": "mlp", "hidden_size": 16.5},
        obs_size=21,
        num_actions=9,
        arch="mlp",
        hidden_size=32,
        device=torch.device("cpu"),
        mlp_factory=lambda obs, acts, hidden: _FakePolicy(obs, acts, hidden),
        resmlp_factory=lambda obs, acts, hidden, num_blocks: _FakePolicy(
            obs, acts, hidden, num_blocks=num_blocks
        ),
    )

    assert effective_arch == "mlp"
    assert effective_hidden_size == 32
    assert policy.hidden_size == 32


def test_load_policy_from_checkpoint_populates_obs_buffers_and_ignores_encoder_norm_missing():
    state_dict = {
        "encoder.0.weight": torch.zeros(16, 21),
        "obs_mean": torch.ones(21),
        "obs_std": torch.full((21,), 2.0),
    }

    policy, effective_arch, effective_hidden_size = load_policy_from_checkpoint(
        ckpt={"model": state_dict},
        obs_size=21,
        num_actions=9,
        arch="mlp",
        hidden_size=16,
        device=torch.device("cpu"),
        mlp_factory=lambda obs, acts, hidden: _FakePolicy(
            obs,
            acts,
            hidden,
            missing=("obs_mean", "obs_std", "encoder_norm.weight", "encoder_norm.bias"),
        ),
        resmlp_factory=lambda obs, acts, hidden, num_blocks: _FakePolicy(
            obs, acts, hidden, num_blocks=num_blocks
        ),
    )

    assert effective_arch == "mlp"
    assert effective_hidden_size == 16
    assert policy._use_encoder_norm is False
    torch.testing.assert_close(policy.obs_mean, torch.ones(21))
    torch.testing.assert_close(policy.obs_std, torch.full((21,), 2.0))


def test_load_policy_from_checkpoint_raises_on_non_ignored_state_mismatch():
    state_dict = {"encoder.0.weight": torch.zeros(16, 21)}

    with pytest.raises(RuntimeError, match="Checkpoint architecture mismatch"):
        load_policy_from_checkpoint(
            ckpt={"model": state_dict},
            obs_size=21,
            num_actions=9,
            arch="mlp",
            hidden_size=16,
            device=torch.device("cpu"),
            mlp_factory=lambda obs, acts, hidden: _FakePolicy(
                obs,
                acts,
                hidden,
                missing=("actor.weight",),
            ),
            resmlp_factory=lambda obs, acts, hidden, num_blocks: _FakePolicy(
                obs, acts, hidden, num_blocks=num_blocks
            ),
        )
