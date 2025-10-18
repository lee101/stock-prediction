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
    ):
        super().__init__()
        self.n_assets = n_assets
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.gradient_checkpointing = gradient_checkpointing

        self.in_norm = nn.LayerNorm(n_assets * feature_dim)
        self.gru = nn.GRU(
            input_size=n_assets * feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, n_assets)
        self.softplus = nn.Softplus()
        self.alpha_bias = nn.Parameter(torch.ones(n_assets, dtype=torch.float32) * 1.1)

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
            gru_out = torch.utils.checkpoint.checkpoint(self._gru_forward, flat)
        else:
            gru_out = self._gru_forward(flat)
        logits = self.head(gru_out)
        alpha = self.softplus(logits.float()) + self.alpha_bias
        return alpha
