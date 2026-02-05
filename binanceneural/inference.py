from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch

from src.torch_device_utils import require_cuda as require_cuda_device

from .data import FeatureNormalizer
from .model import BinancePolicyBase


def generate_actions_from_frame(
    *,
    model: BinancePolicyBase,
    frame: pd.DataFrame,
    feature_columns: Iterable[str],
    normalizer: FeatureNormalizer,
    sequence_length: int,
    horizon: int = 1,
    device: Optional[torch.device] = None,
    require_gpu: bool = False,
) -> pd.DataFrame:
    """Generate per-hour trading actions from a prepared feature frame."""

    if len(frame) < sequence_length:
        raise ValueError("Frame shorter than sequence length; cannot generate actions.")

    if require_gpu:
        if device is not None and device.type != "cuda":
            raise RuntimeError(f"GPU required for inference; received device={device}.")
        device = device or require_cuda_device("inference", allow_fallback=False)
    else:
        device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = model.to(device)
    model.eval()

    feature_cols = list(feature_columns)
    features = frame[feature_cols].to_numpy(dtype=np.float32)
    features = normalizer.transform(features)

    reference_close = frame["reference_close"].to_numpy(dtype=np.float32)
    high_col = f"predicted_high_p50_h{int(horizon)}"
    low_col = f"predicted_low_p50_h{int(horizon)}"
    if high_col not in frame.columns or low_col not in frame.columns:
        raise ValueError(f"Missing Chronos columns for horizon {horizon}h: {high_col}, {low_col}")
    chronos_high = frame[high_col].to_numpy(dtype=np.float32)
    chronos_low = frame[low_col].to_numpy(dtype=np.float32)

    actions = []
    with torch.no_grad():
        for idx in range(sequence_length - 1, len(frame)):
            start = idx - sequence_length + 1
            end = idx + 1
            seq = torch.from_numpy(features[start:end]).unsqueeze(0).to(device)
            ref = torch.from_numpy(reference_close[start:end]).unsqueeze(0).to(device)
            c_high = torch.from_numpy(chronos_high[start:end]).unsqueeze(0).to(device)
            c_low = torch.from_numpy(chronos_low[start:end]).unsqueeze(0).to(device)

            outputs = model(seq)
            decoded = model.decode_actions(
                outputs,
                reference_close=ref,
                chronos_high=c_high,
                chronos_low=c_low,
            )
            take = -1
            actions.append(
                {
                    "timestamp": frame["timestamp"].iloc[idx],
                    "symbol": frame["symbol"].iloc[idx],
                    "buy_price": float(decoded["buy_price"][0, take].cpu().item()),
                    "sell_price": float(decoded["sell_price"][0, take].cpu().item()),
                    "buy_amount": float(decoded["buy_amount"][0, take].cpu().item()),
                    "sell_amount": float(decoded["sell_amount"][0, take].cpu().item()),
                    "trade_amount": float(decoded["trade_amount"][0, take].cpu().item()),
                }
            )

    return pd.DataFrame(actions)


def generate_latest_action(
    *,
    model: BinancePolicyBase,
    frame: pd.DataFrame,
    feature_columns: Iterable[str],
    normalizer: FeatureNormalizer,
    sequence_length: int,
    horizon: int = 1,
    device: Optional[torch.device] = None,
    require_gpu: bool = False,
) -> dict:
    """Generate a single latest action from the most recent sequence window."""

    if len(frame) < sequence_length:
        raise ValueError("Frame shorter than sequence length; cannot generate latest action.")

    if require_gpu:
        if device is not None and device.type != "cuda":
            raise RuntimeError(f"GPU required for inference; received device={device}.")
        device = device or require_cuda_device("inference", allow_fallback=False)
    else:
        device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = model.to(device)
    model.eval()

    feature_cols = list(feature_columns)
    features = frame[feature_cols].to_numpy(dtype=np.float32)
    features = normalizer.transform(features)

    high_col = f"predicted_high_p50_h{int(horizon)}"
    low_col = f"predicted_low_p50_h{int(horizon)}"
    if high_col not in frame.columns or low_col not in frame.columns:
        raise ValueError(f"Missing Chronos columns for horizon {horizon}h: {high_col}, {low_col}")

    start = len(frame) - sequence_length
    end = len(frame)

    seq = torch.from_numpy(features[start:end]).unsqueeze(0).to(device)
    ref = torch.from_numpy(frame["reference_close"].to_numpy(dtype=np.float32)[start:end]).unsqueeze(0).to(device)
    c_high = torch.from_numpy(frame[high_col].to_numpy(dtype=np.float32)[start:end]).unsqueeze(0).to(device)
    c_low = torch.from_numpy(frame[low_col].to_numpy(dtype=np.float32)[start:end]).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(seq)
        decoded = model.decode_actions(
            outputs,
            reference_close=ref,
            chronos_high=c_high,
            chronos_low=c_low,
        )

    take = -1
    return {
        "timestamp": frame["timestamp"].iloc[-1],
        "symbol": frame["symbol"].iloc[-1],
        "buy_price": float(decoded["buy_price"][0, take].cpu().item()),
        "sell_price": float(decoded["sell_price"][0, take].cpu().item()),
        "buy_amount": float(decoded["buy_amount"][0, take].cpu().item()),
        "sell_amount": float(decoded["sell_amount"][0, take].cpu().item()),
        "trade_amount": float(decoded["trade_amount"][0, take].cpu().item()),
    }


__all__ = ["generate_actions_from_frame", "generate_latest_action"]
