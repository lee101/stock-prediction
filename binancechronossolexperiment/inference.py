from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Tuple

import torch

from binanceneural.config import DatasetConfig
from binanceneural.data import BinanceHourlyDataModule
from binanceneural.inference import generate_actions_from_frame, generate_latest_action
from binanceneural.model import (
    BinancePolicyBase,
    align_state_dict_input_dim,
    build_policy,
    policy_config_from_payload,
)
from binanceneural.data import FeatureNormalizer

REPO = Path(__file__).resolve().parents[1]


def _resolve_recovery_path(path_value, *, default: Path) -> Path:
    if path_value in (None, ""):
        return default
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (REPO / path).resolve()


def _recover_dataset_artifacts(
    cfg: dict,
    *,
    data_root: str | Path | None = None,
    forecast_cache_root: str | Path | None = None,
) -> tuple[FeatureNormalizer, list[str]]:
    dataset_cfg = dict(cfg.get("dataset") or {})
    symbol = str(dataset_cfg.get("symbol") or cfg.get("symbol") or "")
    if not symbol:
        forecast_cfg = dict(cfg.get("forecast_config") or {})
        symbol = str(forecast_cfg.get("symbol") or "")
    if not symbol:
        raise KeyError("Checkpoint missing symbol metadata needed to recover feature metadata.")

    sequence_length = int(dataset_cfg.get("sequence_length") or cfg.get("sequence_length") or 72)
    dataset = DatasetConfig(
        symbol=symbol,
        data_root=_resolve_recovery_path(
            data_root if data_root is not None else dataset_cfg.get("data_root"),
            default=REPO / "trainingdatahourlybinance",
        ),
        forecast_cache_root=_resolve_recovery_path(
            forecast_cache_root if forecast_cache_root is not None else dataset_cfg.get("forecast_cache_root"),
            default=REPO / "binanceneural" / "forecast_cache",
        ),
        forecast_horizons=tuple(int(h) for h in dataset_cfg.get("forecast_horizons", (1,))),
        sequence_length=sequence_length,
        val_fraction=float(dataset_cfg.get("val_fraction", DatasetConfig.val_fraction)),
        min_history_hours=int(dataset_cfg.get("min_history_hours", DatasetConfig.min_history_hours)),
        max_feature_lookback_hours=int(
            dataset_cfg.get("max_feature_lookback_hours", DatasetConfig.max_feature_lookback_hours)
        ),
        feature_columns=dataset_cfg.get("feature_columns"),
        refresh_hours=int(dataset_cfg.get("refresh_hours", DatasetConfig.refresh_hours)),
        validation_days=int(dataset_cfg.get("validation_days", DatasetConfig.validation_days)),
        cache_only=bool(dataset_cfg.get("cache_only", True)),
    )
    dm = BinanceHourlyDataModule(dataset)
    return dm.normalizer, list(dm.feature_columns)


def load_policy_checkpoint(
    checkpoint_path: str,
    *,
    device: torch.device | None = None,
    data_root: str | Path | None = None,
    forecast_cache_root: str | Path | None = None,
) -> Tuple[BinancePolicyBase, FeatureNormalizer, Iterable[str], dict]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict", payload)
    cfg = payload.get("config") or {}
    if hasattr(cfg, "__dict__"):
        cfg = asdict(cfg)
    feature_columns = list(payload.get("feature_columns") or [])
    normalizer_payload = payload.get("normalizer") or {}
    if feature_columns and normalizer_payload:
        normalizer = FeatureNormalizer.from_dict(normalizer_payload)
    else:
        normalizer, feature_columns = _recover_dataset_artifacts(
            cfg,
            data_root=data_root,
            forecast_cache_root=forecast_cache_root,
        )

    model = _build_policy(state_dict, cfg, len(feature_columns))
    if device is not None:
        model = model.to(device)
    return model, normalizer, feature_columns, cfg


def _build_policy(state_dict: dict, cfg: dict, input_dim: int) -> BinancePolicyBase:
    state_dict = align_state_dict_input_dim(state_dict, input_dim=input_dim)
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
