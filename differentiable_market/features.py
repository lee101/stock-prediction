from __future__ import annotations

import torch


def ohlc_to_features(ohlc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert OHLC data into model features and next-step log returns.

    Args:
        ohlc: Tensor shaped [T, A, 4] with columns (open, high, low, close)

    Returns:
        features: Tensor shaped [T-1, A, F=4]
        forward_returns: Tensor shaped [T-1, A]
    """
    if ohlc.ndim != 3 or ohlc.size(-1) != 4:
        raise ValueError(f"Expected [T, A, 4] tensor, got {tuple(ohlc.shape)}")

    O = ohlc[..., 0]
    H = ohlc[..., 1]
    L = ohlc[..., 2]
    C = ohlc[..., 3]

    prev_close = torch.cat([C[:1], C[:-1]], dim=0)
    eps = 1e-8

    features = torch.stack(
        [
            torch.log(torch.clamp(O / prev_close, min=eps)),
            torch.log(torch.clamp(H / O, min=eps)),
            torch.log(torch.clamp(L / O, min=eps)),
            torch.log(torch.clamp(C / O, min=eps)),
        ],
        dim=-1,
    )
    forward_returns = torch.log(torch.clamp(C[1:] / C[:-1], min=eps))
    return features[:-1], forward_returns

