from __future__ import annotations

import pickle
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

import numpy as np
import torch
from torch import nn


def load_checkpoint_payload(
    checkpoint_path: str | Path,
    *,
    map_location: torch.device | str | None = None,
) -> object:
    checkpoint_path = Path(checkpoint_path)
    try:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except (FileNotFoundError, IsADirectoryError, OSError, EOFError, RuntimeError, pickle.UnpicklingError) as exc:
        raise RuntimeError(f"Failed to load checkpoint {checkpoint_path.resolve()}: {exc}") from exc


def _looks_like_state_dict(value: object) -> bool:
    return isinstance(value, Mapping) and bool(value) and all(
        isinstance(key, str) and isinstance(tensor, torch.Tensor)
        for key, tensor in value.items()
    )


def extract_checkpoint_state_dict(ckpt: object) -> Mapping[str, object]:
    if isinstance(ckpt, Mapping) and "model" in ckpt:
        state_dict = ckpt.get("model")
        if _looks_like_state_dict(state_dict):
            return state_dict
        raise KeyError("Checkpoint is missing a valid 'model' state_dict")
    if _looks_like_state_dict(ckpt):
        return ckpt
    raise ValueError("Unsupported checkpoint format (expected state_dict or dict with 'model')")


def _coerce_finite_float(value: object) -> float | None:
    if isinstance(value, str):
        try:
            value_float = float(value.strip())
        except ValueError:
            return None
    elif isinstance(value, Real):
        value_float = float(value)
    else:
        return None
    if not np.isfinite(value_float):
        return None
    return value_float


def _coerce_finite_positive_int(value: object) -> int | None:
    value_float = _coerce_finite_float(value)
    if value_float is None or not value_float.is_integer():
        return None
    value_int = int(value_float)
    if value_int <= 0:
        return None
    return value_int


def _coerce_finite_nonnegative_float(value: object) -> float | None:
    value_float = _coerce_finite_float(value)
    if value_float is None or value_float < 0.0:
        return None
    return value_float


@dataclass(frozen=True)
class ResolvedCheckpointPolicyMetadata:
    state_dict: Mapping[str, object]
    effective_arch: str
    effective_hidden_size: int
    effective_resmlp_blocks: int | None = None


def resolve_checkpoint_action_grid_config(
    ckpt: object,
    *,
    action_allocation_bins: int,
    action_level_bins: int,
    action_max_offset_bps: float,
) -> tuple[int, int, float]:
    effective_allocation_bins = max(1, int(action_allocation_bins))
    effective_level_bins = max(1, int(action_level_bins))
    effective_max_offset_bps = max(0.0, float(action_max_offset_bps))

    if not isinstance(ckpt, Mapping):
        return effective_allocation_bins, effective_level_bins, effective_max_offset_bps

    checkpoint_allocation_bins = _coerce_finite_positive_int(ckpt.get("action_allocation_bins"))
    checkpoint_level_bins = _coerce_finite_positive_int(ckpt.get("action_level_bins"))
    checkpoint_max_offset_bps = _coerce_finite_nonnegative_float(ckpt.get("action_max_offset_bps"))

    if checkpoint_allocation_bins is not None:
        effective_allocation_bins = checkpoint_allocation_bins
    if checkpoint_level_bins is not None:
        effective_level_bins = checkpoint_level_bins
    if checkpoint_max_offset_bps is not None:
        effective_max_offset_bps = checkpoint_max_offset_bps

    return effective_allocation_bins, effective_level_bins, effective_max_offset_bps


def resolve_checkpoint_policy_details(
    ckpt: Mapping[str, object],
    *,
    arch: str,
    hidden_size: int,
) -> ResolvedCheckpointPolicyMetadata:
    if not isinstance(ckpt, Mapping):
        raise TypeError(f"Checkpoint must load to a mapping, got {type(ckpt).__name__}")
    state_dict = ckpt.get("model")
    if not isinstance(state_dict, Mapping):
        raise KeyError("Checkpoint is missing a valid 'model' state_dict")

    requested_arch = str(arch).strip().lower()
    checkpoint_arch = str(ckpt.get("arch") or requested_arch).strip().lower()
    base_arches = {"mlp", "resmlp"}
    supported_arches = base_arches | {"transformer", "gru", "depth_recurrence", "mlp_relu_sq"}
    effective_arch = checkpoint_arch if checkpoint_arch in supported_arches else requested_arch
    if effective_arch in base_arches:
        try:
            effective_arch = infer_arch_from_state_dict(state_dict)
        except ValueError:
            pass

    try:
        effective_hidden_size = infer_hidden_size_from_state_dict(state_dict, effective_arch)
    except Exception:
        checkpoint_hidden_size = _coerce_finite_positive_int(ckpt.get("hidden_size"))
        effective_hidden_size = checkpoint_hidden_size or int(hidden_size)

    effective_resmlp_blocks = None
    if effective_arch == "resmlp":
        effective_resmlp_blocks = infer_resmlp_blocks_from_state_dict(state_dict)

    return ResolvedCheckpointPolicyMetadata(
        state_dict=state_dict,
        effective_arch=effective_arch,
        effective_hidden_size=effective_hidden_size,
        effective_resmlp_blocks=effective_resmlp_blocks,
    )


def resolve_checkpoint_policy_metadata(
    ckpt: Mapping[str, object],
    *,
    arch: str,
    hidden_size: int,
) -> tuple[Mapping[str, object], str, int]:
    resolved = resolve_checkpoint_policy_details(
        ckpt,
        arch=arch,
        hidden_size=hidden_size,
    )
    return resolved.state_dict, resolved.effective_arch, resolved.effective_hidden_size


def infer_arch_from_state_dict(state_dict: Mapping[str, object]) -> str:
    if "input_proj.weight" in state_dict:
        return "resmlp"
    if "encoder.0.weight" in state_dict:
        return "mlp"
    for key in state_dict:
        key_str = str(key)
        if key_str.startswith(("input_proj.", "blocks.")):
            return "resmlp"
        if key_str.startswith("encoder."):
            return "mlp"
    raise ValueError("Could not infer policy architecture from checkpoint state_dict")


def infer_hidden_size_from_state_dict(state_dict: Mapping[str, object], arch: str) -> int:
    if arch == "resmlp":
        return int(state_dict["input_proj.weight"].shape[0])
    return int(state_dict["encoder.0.weight"].shape[0])


def infer_resmlp_blocks_from_state_dict(state_dict: Mapping[str, object]) -> int:
    block_idxs = [
        int(str(key).split(".")[1])
        for key in state_dict
        if str(key).startswith("blocks.") and str(key).split(".")[1].isdigit()
    ]
    return max(block_idxs) + 1 if block_idxs else 3


def format_best_return_label(ckpt: Mapping[str, object]) -> str:
    best_return = ckpt.get("best_return")
    if isinstance(best_return, (int, float, np.integer, np.floating)):
        best_return_float = float(best_return)
        if np.isfinite(best_return_float):
            return f"{best_return_float:.4f}"
    return "?"


def format_checkpoint_override_note(
    *,
    requested_arch: str,
    requested_hidden_size: int,
    effective_arch: str,
    effective_hidden_size: int,
) -> str | None:
    overrides: list[str] = []
    if str(requested_arch).strip().lower() != str(effective_arch).strip().lower():
        overrides.append(f"arch {requested_arch} -> {effective_arch}")
    if int(requested_hidden_size) != int(effective_hidden_size):
        overrides.append(f"hidden_size {int(requested_hidden_size)} -> {int(effective_hidden_size)}")
    if not overrides:
        return None
    return "Checkpoint overrides CLI policy config: " + ", ".join(overrides)


def build_cli_policy_config_line(*, arch: str, hidden_size: int) -> str:
    return f"CLI policy config: arch={arch}, hidden_size={int(hidden_size)}"


def _format_action_max_offset_bps(value: float) -> str:
    value_float = float(value)
    if value_float.is_integer():
        return f"{value_float:.1f}"
    return f"{value_float:.4f}".rstrip("0").rstrip(".")


def _format_action_grid_config(
    *,
    action_allocation_bins: int,
    action_level_bins: int,
    action_max_offset_bps: float,
) -> str:
    return (
        f"alloc_bins={int(action_allocation_bins)} "
        f"level_bins={int(action_level_bins)} "
        f"max_offset_bps={_format_action_max_offset_bps(float(action_max_offset_bps))}"
    )


def build_action_grid_summary_line(
    *,
    action_allocation_bins: int,
    action_level_bins: int,
    action_max_offset_bps: float,
) -> str:
    return "Action grid: " + _format_action_grid_config(
        action_allocation_bins=action_allocation_bins,
        action_level_bins=action_level_bins,
        action_max_offset_bps=action_max_offset_bps,
    )


def ensure_checkpoint_action_grid_compatible(
    ckpt: object,
    *,
    checkpoint_path: str | Path,
    expected_action_allocation_bins: int,
    expected_action_level_bins: int,
    expected_action_max_offset_bps: float,
) -> tuple[int, int, float]:
    expected_allocation_bins = max(1, int(expected_action_allocation_bins))
    expected_level_bins = max(1, int(expected_action_level_bins))
    expected_max_offset_bps = max(0.0, float(expected_action_max_offset_bps))
    effective_allocation_bins, effective_level_bins, effective_max_offset_bps = resolve_checkpoint_action_grid_config(
        ckpt,
        action_allocation_bins=expected_allocation_bins,
        action_level_bins=expected_level_bins,
        action_max_offset_bps=expected_max_offset_bps,
    )
    if (
        effective_allocation_bins,
        effective_level_bins,
        effective_max_offset_bps,
    ) != (
        expected_allocation_bins,
        expected_level_bins,
        expected_max_offset_bps,
    ):
        raise RuntimeError(
            "Checkpoint {} is incompatible with ensemble action grid: expected {}, got {}".format(
                Path(checkpoint_path).resolve(),
                _format_action_grid_config(
                    action_allocation_bins=expected_allocation_bins,
                    action_level_bins=expected_level_bins,
                    action_max_offset_bps=expected_max_offset_bps,
                ),
                _format_action_grid_config(
                    action_allocation_bins=effective_allocation_bins,
                    action_level_bins=effective_level_bins,
                    action_max_offset_bps=effective_max_offset_bps,
                ),
            )
        )
    return effective_allocation_bins, effective_level_bins, effective_max_offset_bps


def format_action_grid_override_note(
    *,
    requested_allocation_bins: int,
    requested_level_bins: int,
    requested_max_offset_bps: float,
    effective_allocation_bins: int,
    effective_level_bins: int,
    effective_max_offset_bps: float,
) -> str | None:
    overrides: list[str] = []
    if int(requested_allocation_bins) != int(effective_allocation_bins):
        overrides.append(f"alloc_bins {int(requested_allocation_bins)} -> {int(effective_allocation_bins)}")
    if int(requested_level_bins) != int(effective_level_bins):
        overrides.append(f"level_bins {int(requested_level_bins)} -> {int(effective_level_bins)}")
    if float(requested_max_offset_bps) != float(effective_max_offset_bps):
        overrides.append(
            f"max_offset_bps {_format_action_max_offset_bps(float(requested_max_offset_bps))} -> "
            f"{_format_action_max_offset_bps(float(effective_max_offset_bps))}"
        )
    if not overrides:
        return None
    return "Checkpoint overrides CLI action grid: " + ", ".join(overrides)


def build_runtime_summary_line(*, device: torch.device, deterministic: bool) -> str:
    return f"Runtime: device={device}, deterministic={deterministic}"


def build_checkpoint_summary_lines(
    *,
    ckpt: Mapping[str, object],
    requested_arch: str,
    requested_hidden_size: int,
    effective_arch: str,
    effective_hidden_size: int,
    checkpoint_path: str | Path | None = None,
) -> list[str]:
    lines: list[str] = []
    if checkpoint_path is not None:
        lines.append(f"Checkpoint file: {Path(checkpoint_path).resolve()}")
    lines.append(f"Effective checkpoint config: arch={effective_arch}, hidden_size={effective_hidden_size}")
    override_note = format_checkpoint_override_note(
        requested_arch=requested_arch,
        requested_hidden_size=requested_hidden_size,
        effective_arch=effective_arch,
        effective_hidden_size=effective_hidden_size,
    )
    if override_note is not None:
        lines.append(override_note)
    lines.append(
        "Loaded checkpoint: update={}, train_best_return={}, arch={}".format(
            ckpt.get("update", "?"),
            format_best_return_label(ckpt),
            effective_arch,
        )
    )
    return lines


def build_ensemble_member_summary_lines(
    *,
    ckpt: Mapping[str, object],
    requested_arch: str,
    requested_hidden_size: int,
    effective_arch: str,
    effective_hidden_size: int,
    checkpoint_path: str | Path,
) -> list[str]:
    lines = [f"  + ensemble member file: {Path(checkpoint_path).resolve()}"]
    lines.append(
        "  + ensemble member: update={} train_best_return={} arch={} hidden_size={}".format(
            ckpt.get("update", "?"),
            format_best_return_label(ckpt),
            effective_arch,
            effective_hidden_size,
        )
    )
    override_note = format_checkpoint_override_note(
        requested_arch=requested_arch,
        requested_hidden_size=requested_hidden_size,
        effective_arch=effective_arch,
        effective_hidden_size=effective_hidden_size,
    )
    if override_note is not None:
        lines.append(f"    {override_note}")
    return lines


def load_policy_from_checkpoint(
    *,
    ckpt: Mapping[str, object],
    obs_size: int,
    num_actions: int,
    arch: str,
    hidden_size: int,
    device: torch.device,
    mlp_factory: Callable[[int, int, int], nn.Module],
    resmlp_factory: Callable[[int, int, int, int], nn.Module],
) -> tuple[nn.Module, str, int]:
    resolved = resolve_checkpoint_policy_details(
        ckpt,
        arch=arch,
        hidden_size=hidden_size,
    )

    policy = load_policy_from_resolved_metadata(
        state_dict=resolved.state_dict,
        effective_arch=resolved.effective_arch,
        effective_hidden_size=resolved.effective_hidden_size,
        effective_resmlp_blocks=resolved.effective_resmlp_blocks,
        obs_size=obs_size,
        num_actions=num_actions,
        device=device,
        mlp_factory=mlp_factory,
        resmlp_factory=resmlp_factory,
    )

    return policy, resolved.effective_arch, resolved.effective_hidden_size


def load_policy_from_resolved_metadata(
    *,
    state_dict: Mapping[str, object],
    effective_arch: str,
    effective_hidden_size: int,
    effective_resmlp_blocks: int | None = None,
    obs_size: int,
    num_actions: int,
    device: torch.device,
    mlp_factory: Callable[[int, int, int], nn.Module],
    resmlp_factory: Callable[[int, int, int, int], nn.Module],
) -> nn.Module:
    """Instantiate and load a base MLP/ResMLP policy from already-resolved metadata."""

    if effective_arch == "resmlp":
        policy = resmlp_factory(
            obs_size,
            num_actions,
            effective_hidden_size,
            effective_resmlp_blocks
            if effective_resmlp_blocks is not None
            else infer_resmlp_blocks_from_state_dict(state_dict),
        ).to(device)
    elif effective_arch == "mlp":
        policy = mlp_factory(obs_size, num_actions, effective_hidden_size).to(device)
    else:
        raise ValueError(f"Unsupported checkpoint architecture: {effective_arch}")

    load_result = policy.load_state_dict(state_dict, strict=False)
    if hasattr(load_result, "missing_keys") and hasattr(load_result, "unexpected_keys"):
        missing = list(load_result.missing_keys)
        unexpected = list(load_result.unexpected_keys)
    elif isinstance(load_result, tuple) and len(load_result) == 2:
        missing, unexpected = load_result
    else:
        missing, unexpected = [], []

    if hasattr(policy, "_use_encoder_norm"):
        policy._use_encoder_norm = "encoder_norm.weight" not in missing
    ignored = {"obs_mean", "obs_std", "encoder_norm.weight", "encoder_norm.bias"}
    bad_missing = [key for key in missing if key not in ignored]
    bad_unexpected = [key for key in unexpected if key not in ignored]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            f"Checkpoint architecture mismatch - missing: {bad_missing}, unexpected: {bad_unexpected}"
        )

    for buf in ("obs_mean", "obs_std"):
        if buf in state_dict and state_dict[buf] is not None:
            setattr(policy, buf, state_dict[buf].to(device))

    policy.eval()
    return policy
