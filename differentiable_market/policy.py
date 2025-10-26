from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class DirichletGRUPolicy(nn.Module):
    """
    Causal GRU encoder that produces Dirichlet concentration parameters.
    """

    def __init__(
        self,
        n_assets: int,
        feature_dim: int = 4,
        hidden_size: int = 1024,
        num_layers: int = 2,
        dropout: float = 0.0,
        gradient_checkpointing: bool = False,
        enable_shorting: bool = False,
        max_intraday_leverage: float = 1.0,
        max_overnight_leverage: float | None = None,
    ):
        super().__init__()
        self.n_assets = n_assets
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.gradient_checkpointing = gradient_checkpointing
        self.enable_shorting = enable_shorting

        intraday_cap = float(max(1.0, max_intraday_leverage))
        if max_overnight_leverage is None:
            overnight_cap = intraday_cap
        else:
            overnight_cap = float(max(0.0, max_overnight_leverage))
        if overnight_cap > intraday_cap:
            overnight_cap = intraday_cap
        self.max_intraday_leverage = intraday_cap
        self.max_overnight_leverage = overnight_cap

        head_dim = n_assets if not enable_shorting else n_assets * 2 + 1

        self.in_norm = nn.LayerNorm(n_assets * feature_dim)
        self.gru = nn.GRU(
            input_size=n_assets * feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, head_dim)
        self.softplus = nn.Softplus()
        self.alpha_bias = nn.Parameter(torch.ones(head_dim, dtype=torch.float32) * 1.1)

    def _gru_forward(self, x: Tensor) -> Tensor:
        out, _ = self.gru(x)
        return out

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor shaped [B, T, A, F]
        Returns:
            Dirichlet concentration parameters shaped [B, T, A]
        """
        if x.ndim != 4:
            raise ValueError(f"Expected input [B, T, A, F], got {tuple(x.shape)}")
        B, T, A, F = x.shape
        if A != self.n_assets or F != self.feature_dim:
            raise ValueError("Input asset/feature dims do not match policy configuration")

        flat = x.reshape(B, T, A * F)
        flat = flat.float()
        flat = self.in_norm(flat)
        if self.gradient_checkpointing and self.training:
            gru_out = torch.utils.checkpoint.checkpoint(self._gru_forward, flat, use_reentrant=False)
        else:
            gru_out = self._gru_forward(flat)
        logits = self.head(gru_out)
        alpha = self.softplus(logits.float()) + self.alpha_bias
        return alpha

    @staticmethod
    def _normalise(alpha: Tensor) -> Tensor:
        denom = alpha.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return alpha / denom

    def allocations_to_weights(self, allocations: Tensor) -> tuple[Tensor, Tensor]:
        """
        Convert Dirichlet allocations into intraday/overnight weight tensors.

        Args:
            allocations: Tensor shaped [B, T, D] with simplex-constrained rows.

        Returns:
            intraday_weights: [B, T, A] tensor used to compute rewards.
            overnight_weights: [B, T, A] tensor used as the next-step prior.
        """
        if not self.enable_shorting:
            weights = allocations
            return weights, weights

        B, T, D = allocations.shape
        A = self.n_assets
        if D != 2 * A + 1:
            raise ValueError(f"Expected allocation dimension {2 * A + 1}, got {D}")

        long_alloc = allocations[..., :A]
        short_alloc = allocations[..., A : 2 * A]
        reserve_alloc = allocations[..., 2 * A :]

        eps = 1e-8
        long_total = long_alloc.sum(dim=-1, keepdim=True)
        short_total = short_alloc.sum(dim=-1, keepdim=True)

        long_dir = torch.where(
            long_total > eps,
            long_alloc / long_total.clamp_min(eps),
            torch.zeros_like(long_alloc),
        )
        short_dir = torch.where(
            short_total > eps,
            short_alloc / short_total.clamp_min(eps),
            torch.zeros_like(short_alloc),
        )

        gross_long = long_total * self.max_intraday_leverage
        gross_short = short_total * self.max_intraday_leverage
        intraday = gross_long * long_dir - gross_short * short_dir

        gross_abs = intraday.abs().sum(dim=-1, keepdim=True).clamp_min(eps)
        overnight_cap = self.max_overnight_leverage
        if overnight_cap < self.max_intraday_leverage:
            scale = torch.minimum(torch.ones_like(gross_abs), overnight_cap / gross_abs)
            overnight = intraday * scale
        else:
            overnight = intraday

        # Ensure reserve mass only influences leverage magnitude; asserted for clarity.
        _ = reserve_alloc  # reserve intentionally unused beyond leverage scaling
        return intraday, overnight

    def decode_concentration(self, alpha: Tensor) -> tuple[Tensor, Tensor]:
        allocations = self._normalise(alpha)
        return self.allocations_to_weights(allocations)
