from __future__ import annotations

from dataclasses import asdict
from typing import Iterable, Tuple

import torch

from binanceneural.inference import generate_actions_from_frame, generate_latest_action
from binanceneural.model import BinancePolicyBase, build_policy, policy_config_from_payload
from binanceneural.data import FeatureNormalizer


def load_policy_checkpoint(
    checkpoint_path: str,
    *,
    device: torch.device | None = None,
) -> Tuple[BinancePolicyBase, FeatureNormalizer, Iterable[str], dict]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict", payload)
    feature_columns = payload.get("feature_columns") or []
    normalizer_payload = payload.get("normalizer") or {}
    normalizer = FeatureNormalizer.from_dict(normalizer_payload)

    cfg = payload.get("config") or {}
    if hasattr(cfg, "__dict__"):
        cfg = asdict(cfg)

    model = _build_policy(state_dict, cfg, len(feature_columns))
    if device is not None:
        model = model.to(device)
    return model, normalizer, feature_columns, cfg


def _build_policy(state_dict: dict, cfg: dict, input_dim: int) -> BinancePolicyBase:
    policy_cfg = policy_config_from_payload(cfg, input_dim=input_dim, state_dict=state_dict)
    model = build_policy(policy_cfg)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


__all__ = [
    "load_policy_checkpoint",
    "generate_actions_from_frame",
    "generate_latest_action",
]
