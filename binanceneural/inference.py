from __future__ import annotations

from collections.abc import Iterable

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
    device: torch.device | None = None,
    require_gpu: bool = False,
    batch_size: int = 512,
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

    window_count = len(frame) - sequence_length + 1
    batch_size = window_count if batch_size <= 0 else max(1, int(batch_size))

    feature_tensor = torch.from_numpy(features).to(device)
    ref_tensor = torch.from_numpy(reference_close).to(device)
    high_tensor = torch.from_numpy(chronos_high).to(device)
    low_tensor = torch.from_numpy(chronos_low).to(device)

    feature_windows = feature_tensor.as_strided(
        (window_count, sequence_length, feature_tensor.shape[-1]),
        (feature_tensor.stride(0), feature_tensor.stride(0), feature_tensor.stride(1)),
    )
    ref_windows = ref_tensor.as_strided(
        (window_count, sequence_length),
        (ref_tensor.stride(0), ref_tensor.stride(0)),
    )
    high_windows = high_tensor.as_strided(
        (window_count, sequence_length),
        (high_tensor.stride(0), high_tensor.stride(0)),
    )
    low_windows = low_tensor.as_strided(
        (window_count, sequence_length),
        (low_tensor.stride(0), low_tensor.stride(0)),
    )

    timestamps = frame["timestamp"].iloc[sequence_length - 1 :].to_list()
    symbols = frame["symbol"].iloc[sequence_length - 1 :].to_list()
    actions = []
    with torch.inference_mode():
        for batch_start in range(0, window_count, batch_size):
            batch_end = min(batch_start + batch_size, window_count)
            seq = feature_windows[batch_start:batch_end]
            ref = ref_windows[batch_start:batch_end]
            c_high = high_windows[batch_start:batch_end]
            c_low = low_windows[batch_start:batch_end]
            outputs = model(seq)
            decoded = model.decode_actions(
                outputs,
                reference_close=ref,
                chronos_high=c_high,
                chronos_low=c_low,
            )
            decoded_cpu = {k: v.cpu() for k, v in decoded.items()}
            take = -1
            for batch_offset, row_idx in enumerate(range(batch_start, batch_end)):
                row = {
                    "timestamp": timestamps[row_idx],
                    "symbol": symbols[row_idx],
                    "buy_price": float(decoded_cpu["buy_price"][batch_offset, take].item()),
                    "sell_price": float(decoded_cpu["sell_price"][batch_offset, take].item()),
                    "buy_amount": float(decoded_cpu["buy_amount"][batch_offset, take].item()),
                    "sell_amount": float(decoded_cpu["sell_amount"][batch_offset, take].item()),
                    "trade_amount": float(decoded_cpu["trade_amount"][batch_offset, take].item()),
                }
                if "hold_hours" in decoded_cpu:
                    row["hold_hours"] = float(decoded_cpu["hold_hours"][batch_offset, take].item())
                if "allocation_fraction" in decoded_cpu:
                    row["allocation_fraction"] = float(decoded_cpu["allocation_fraction"][batch_offset, take].item())
                actions.append(row)

    return pd.DataFrame(actions)


def generate_latest_action(
    *,
    model: BinancePolicyBase,
    frame: pd.DataFrame,
    feature_columns: Iterable[str],
    normalizer: FeatureNormalizer,
    sequence_length: int,
    horizon: int = 1,
    device: torch.device | None = None,
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

    with torch.inference_mode():
        outputs = model(seq)
        decoded = model.decode_actions(
            outputs,
            reference_close=ref,
            chronos_high=c_high,
            chronos_low=c_low,
        )

    decoded_cpu = {k: v.cpu() for k, v in decoded.items()}
    take = -1
    close_col = f"predicted_close_p50_h{int(horizon)}"
    result = {
        "timestamp": frame["timestamp"].iloc[-1],
        "symbol": frame["symbol"].iloc[-1],
        "buy_price": float(decoded_cpu["buy_price"][0, take].item()),
        "sell_price": float(decoded_cpu["sell_price"][0, take].item()),
        "buy_amount": float(decoded_cpu["buy_amount"][0, take].item()),
        "sell_amount": float(decoded_cpu["sell_amount"][0, take].item()),
        "trade_amount": float(decoded_cpu["trade_amount"][0, take].item()),
        "predicted_high": float(frame[high_col].iloc[-1]),
        "predicted_low": float(frame[low_col].iloc[-1]),
        "predicted_close": float(frame[close_col].iloc[-1]) if close_col in frame.columns else 0.0,
    }
    if "hold_hours" in decoded_cpu:
        result["hold_hours"] = float(decoded_cpu["hold_hours"][0, take].item())
    if "allocation_fraction" in decoded_cpu:
        result["allocation_fraction"] = float(decoded_cpu["allocation_fraction"][0, take].item())
    return result


__all__ = ["generate_actions_from_frame", "generate_latest_action"]
