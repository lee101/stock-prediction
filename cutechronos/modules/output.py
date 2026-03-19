"""Fused output head for Chronos2 inference.

Replaces the original output pipeline:
    1. ResidualBlock (output_patch_embedding): (B, P_out, 768) -> (B, P_out, Q*P)
    2. rearrange "b n (q p) -> b q (n p)"
    3. inverse instance norm: sinh + unscale

with a single module that uses a Triton kernel for step 2+3.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Pure-PyTorch fallback for the fused rearrange + sinh + unscale
# ---------------------------------------------------------------------------

def _output_transform_fallback(
    x: torch.Tensor,
    loc: torch.Tensor,
    scale: torch.Tensor,
    num_quantiles: int = 21,
    patch_size: int = 16,
    use_arcsinh: bool = True,
) -> torch.Tensor:
    """Reference implementation of rearrange + sinh + unscale in PyTorch."""
    B, N, QP = x.shape
    Q = num_quantiles
    P = patch_size
    orig_dtype = x.dtype

    # Rearrange: (B, N, Q*P) -> (B, Q, N*P)
    x = x.view(B, N, Q, P).permute(0, 2, 1, 3).reshape(B, Q, N * P)

    # Inverse instance norm in FP32
    x = x.float()
    if use_arcsinh:
        x = torch.sinh(x)
    x = x * scale.unsqueeze(1) + loc.unsqueeze(1)

    return x.to(orig_dtype)


# ---------------------------------------------------------------------------
# Try importing the Triton kernel; fall back to PyTorch.
# ---------------------------------------------------------------------------

try:
    from cutechronos.triton_kernels.fused_output import fused_output_transform
    _fused_output_transform = fused_output_transform
except (ImportError, ModuleNotFoundError):
    _fused_output_transform = _output_transform_fallback


# ---------------------------------------------------------------------------
# FusedOutputHead
# ---------------------------------------------------------------------------

class FusedOutputHead(nn.Module):
    """Drop-in replacement for Chronos2 output_patch_embedding + inverse norm.

    Contains the ResidualBlock (3 linear layers + ReLU activation) and
    fuses the subsequent rearrange + sinh + unscale into a single Triton
    kernel call.

    Weight shapes are identical to the original ResidualBlock so that
    ``load_from_original`` can copy parameters directly.
    """

    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 3072,
        out_dim: int = 336,  # num_quantiles * patch_size = 21 * 16
        num_quantiles: int = 21,
        patch_size: int = 16,
    ):
        super().__init__()
        self.num_quantiles = num_quantiles
        self.patch_size = patch_size

        assert out_dim == num_quantiles * patch_size, (
            f"out_dim ({out_dim}) must equal num_quantiles * patch_size "
            f"({num_quantiles} * {patch_size} = {num_quantiles * patch_size})"
        )

        # ResidualBlock layers (bias=True, matching nn.Linear default)
        self.hidden_layer = nn.Linear(in_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, out_dim)
        self.residual_layer = nn.Linear(in_dim, out_dim)

    # ---- weight loading -------------------------------------------------

    def load_from_original(self, output_patch_embedding: nn.Module) -> None:
        """Copy weights from the original Chronos2 ``output_patch_embedding``.

        ``output_patch_embedding`` is expected to be a ResidualBlock with:
        - ``hidden_layer`` (nn.Linear: in_dim -> hidden_dim)
        - ``output_layer`` (nn.Linear: hidden_dim -> out_dim)
        - ``residual_layer`` (nn.Linear: in_dim -> out_dim)
        """
        with torch.no_grad():
            for name in ("hidden_layer", "output_layer", "residual_layer"):
                src = getattr(output_patch_embedding, name)
                dst = getattr(self, name)
                dst.weight.copy_(src.weight)
                if src.bias is not None:
                    dst.bias.copy_(src.bias)

    # ---- forward --------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        loc: torch.Tensor,
        scale: torch.Tensor,
        use_arcsinh: bool = True,
    ) -> torch.Tensor:
        """Forward pass: ResidualBlock + fused rearrange + inverse norm.

        Args:
            hidden_states: (B, P_out, d_model) encoder output for forecast patches
            loc: (B, 1) location from instance normalisation
            scale: (B, 1) scale from instance normalisation
            use_arcsinh: whether sinh should be applied (inverse of arcsinh)

        Returns:
            Quantile predictions of shape (B, num_quantiles, horizon)
            where horizon = P_out * patch_size.
        """
        # ResidualBlock: hidden + activation + output + residual skip
        hid = F.relu(F.linear(hidden_states, self.hidden_layer.weight, self.hidden_layer.bias))
        out = F.linear(hid, self.output_layer.weight, self.output_layer.bias)
        res = F.linear(hidden_states, self.residual_layer.weight, self.residual_layer.bias)
        x = out + res  # (B, P_out, Q*P)

        # Fused rearrange + sinh + unscale
        return _fused_output_transform(
            x,
            loc,
            scale,
            num_quantiles=self.num_quantiles,
            patch_size=self.patch_size,
            use_arcsinh=use_arcsinh,
        )
