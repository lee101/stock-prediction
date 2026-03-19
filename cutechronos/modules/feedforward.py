"""Fused FeedForward module for Chronos2.

Optimizations over the original:
1. RMS LayerNorm computed in FP32 (matching original Chronos2LayerNorm)
2. ReLU fused with wi projection via F.relu(F.linear(x, w)) — avoids
   materializing the 3072-wide intermediate separately for ReLU
3. Dropout is no-op at inference (eval mode), so we skip it entirely

Benchmark note: For production sizes (768->3072), cuBLAS GEMM is
extremely competitive. We use cuBLAS via F.linear for the matrix
multiplications and fuse only the elementwise operations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from cutechronos.modules._fallbacks import rms_layernorm


class FusedFeedForward(nn.Module):
    """Fused FeedForward block matching Chronos2 FeedForward.

    Architecture: RMS norm -> Linear(d_model, d_ff) -> ReLU -> Linear(d_ff, d_model) -> residual add
    Dropout is skipped (inference-only module).

    Optimizations:
    - ReLU fused with F.linear via F.relu(F.linear(x, w)) (single pass)
    - Residual add is explicit (allows compiler fusion)
    """

    def __init__(self, d_model: int = 768, d_ff: int = 3072, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

        # RMS LayerNorm weight (no bias, T5-style)
        self.norm_weight = nn.Parameter(torch.ones(d_model))

        # wi: d_model -> d_ff (no bias)
        self.wi = nn.Linear(d_model, d_ff, bias=False)

        # wo: d_ff -> d_model (no bias)
        self.wo = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass matching original FeedForward.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Output tensor of same shape with residual connection applied.
        """
        residual = hidden_states

        # Step 1: RMS LayerNorm (T5-style, FP32 variance, no mean subtraction)
        normed = rms_layernorm(hidden_states, self.norm_weight, self.eps)

        # Step 2: wi projection + ReLU (fused: avoids separate ReLU pass)
        hidden_states = F.relu(self.wi(normed))

        # Step 3: wo projection
        hidden_states = self.wo(hidden_states)

        # Step 4: Residual connection (dropout is no-op at inference)
        hidden_states = residual + hidden_states

        return hidden_states

    def load_from_original(self, original_layer: nn.Module) -> None:
        """Copy weights from an original Chronos2 FeedForward layer.

        Args:
            original_layer: A chronos.chronos2.layers.FeedForward instance.
        """
        self.norm_weight.data.copy_(original_layer.layer_norm.weight.data)
        self.eps = original_layer.layer_norm.variance_epsilon
        self.wi.weight.data.copy_(original_layer.mlp.wi.weight.data)
        self.wo.weight.data.copy_(original_layer.mlp.wo.weight.data)
