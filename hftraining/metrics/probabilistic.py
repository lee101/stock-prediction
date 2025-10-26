from __future__ import annotations

import torch


def pinball_loss(
    y_true: torch.Tensor,
    q_pred: torch.Tensor,
    taus: torch.Tensor,
) -> torch.Tensor:
    """
    Vectorised pinball loss over a grid of quantiles.

    Args:
        y_true: Tensor of shape [B, H]
        q_pred: Tensor of shape [B, H, Q]
        taus:   Tensor of shape [Q] with monotonically increasing quantile levels
    """
    if y_true.dim() != 2:
        raise ValueError("y_true must have shape [batch, horizon]")
    if q_pred.dim() != 3:
        raise ValueError("q_pred must have shape [batch, horizon, n_quantiles]")
    if taus.dim() != 1:
        raise ValueError("taus must have shape [n_quantiles]")

    y = y_true.unsqueeze(-1)  # [B, H, 1]
    diff = y - q_pred
    tau = taus.view(1, 1, -1)
    return torch.maximum(tau * diff, (tau - 1.0) * diff)


def crps_from_quantiles(
    y_true: torch.Tensor,
    q_pred: torch.Tensor,
    taus: torch.Tensor,
) -> torch.Tensor:
    """
    Approximate the Continuous Ranked Probability Score (CRPS) given a set of
    quantile forecasts using the discrete trapezoidal rule.

    Returns a scalar tensor containing the mean CRPS across batch and horizon.
    """
    loss = pinball_loss(y_true, q_pred, taus)  # [B, H, Q]
    diffs = taus[1:] - taus[:-1]
    mids = 0.5 * (loss[..., 1:] + loss[..., :-1])
    crps = (mids * diffs).sum(dim=-1)  # [B, H]
    return crps.mean()
