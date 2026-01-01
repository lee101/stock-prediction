"""V5 Transformer model with NEPA-style portfolio latent prediction.

Key features:
- Sequence-of-latents: Autoregressively predict portfolio embeddings
- NEPA loss: Cosine similarity between predicted and target embeddings
- Atrous convolution: Long-range dependencies without full attention
- Multi-resolution: Aggregate across time scales
- Portfolio output: Target weights per asset with softmax normalization
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuraldailyv5.config import PolicyConfigV5


def rms_norm(x: torch.Tensor) -> torch.Tensor:
    """RMSNorm without learnable parameters."""
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary position embeddings."""
    x_r = x[..., ::2]
    x_i = x[..., 1::2]
    out_r = x_r * cos - x_i * sin
    out_i = x_r * sin + x_i * cos
    out = torch.stack([out_r, out_i], dim=-1)
    return out.flatten(-2)


def nepa_loss(
    h_in: torch.Tensor,
    h_out: torch.Tensor,
    shift: bool = True,
) -> torch.Tensor:
    """
    NEPA: Next-Embedding Prediction loss using cosine similarity.

    Args:
        h_in: (B, T, D) input hidden states (target)
        h_out: (B, T, D) output hidden states (prediction)
        shift: if True, compare h_out[:, :-1] with h_in[:, 1:]

    Returns:
        Scalar loss (negative cosine similarity)
    """
    # Detach target
    h_in = h_in.detach()

    if shift:
        p = h_out[:, :-1, :]  # Predict next
        z = h_in[:, 1:, :]    # Target is next hidden state
    else:
        p = h_out
        z = h_in

    # Normalize
    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1)

    # Negative cosine similarity
    loss = -(p * z).sum(dim=-1).mean()
    return loss


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE)."""

    def __init__(self, dim: int, max_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        cos = freqs.cos()[None, :, None, :]
        sin = freqs.sin()[None, :, None, :]
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)
        self.cached_len = seq_len

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cached_len:
            self._build_cache(seq_len)
        return self.cos_cached[:, :seq_len], self.sin_cached[:, :seq_len]


class PatchEmbedding(nn.Module):
    """Embed time series patches (Chronos-2 style)."""

    def __init__(self, config: PolicyConfigV5):
        super().__init__()
        self.patch_size = config.patch_size
        self.input_dim = config.input_dim

        patch_dim = config.patch_size * config.input_dim
        self.embed = nn.Linear(patch_dim, config.hidden_dim, bias=False)

        if config.patch_residual:
            self.residual = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim * 2, bias=False),
                nn.GELU(),
                nn.Linear(config.hidden_dim * 2, config.hidden_dim, bias=False),
            )
        else:
            self.residual = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        num_patches = T // self.patch_size
        x = x[:, :num_patches * self.patch_size]
        x = x.view(B, num_patches, self.patch_size * D)
        h = self.embed(x)
        if self.residual is not None:
            h = h + self.residual(h)
        return h


class AtrousConvBlock(nn.Module):
    """
    Atrous (dilated) convolution block for long-range dependencies.

    Uses multiple dilation rates to capture different time scales without
    the O(n^2) cost of full attention.
    """

    def __init__(self, config: PolicyConfigV5):
        super().__init__()
        self.rates = config.atrous_rates
        in_channels = config.hidden_dim
        out_channels = config.atrous_channels

        # One conv per dilation rate
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, kernel_size=3, dilation=rate, padding=rate, bias=False)
            for rate in self.rates
        ])

        # Combine all rates
        self.combine = nn.Linear(out_channels * len(self.rates), config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
        Returns:
            (batch, seq_len, hidden_dim)
        """
        # Conv1d expects (batch, channels, seq_len)
        x_conv = x.transpose(1, 2)

        # Apply each dilated conv
        outputs = []
        for conv in self.convs:
            out = conv(x_conv)
            out = F.relu(out).square()  # ReLU^2 activation (from nanochat)
            outputs.append(out)

        # Concatenate and project back
        combined = torch.cat(outputs, dim=1)  # (batch, out_channels * num_rates, seq_len)
        combined = combined.transpose(1, 2)   # (batch, seq_len, out_channels * num_rates)
        combined = self.combine(combined)
        combined = self.dropout(combined)

        return combined


class MultiQueryAttention(nn.Module):
    """Multi-Query Attention with QK normalization."""

    def __init__(self, config: PolicyConfigV5, layer_idx: int):
        super().__init__()
        self.n_head = config.num_heads
        self.n_kv_head = config.num_kv_heads
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(config.hidden_dim, config.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, config.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, config.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos_sin: Tuple[torch.Tensor, torch.Tensor],
        is_causal: bool = True,
    ) -> torch.Tensor:
        B, T, _ = x.size()
        cos, sin = cos_sin

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim)

        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # QK norm
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # GQA: expand kv heads if needed
        if self.n_kv_head != self.n_head:
            n_rep = self.n_head // self.n_kv_head
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        y = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=is_causal,
            dropout_p=self.dropout.p if self.training else 0.0,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.out_proj(y)


class MLP(nn.Module):
    """MLP with ReLU^2 activation (from nanochat)."""

    def __init__(self, config: PolicyConfigV5):
        super().__init__()
        self.fc = nn.Linear(config.hidden_dim, 4 * config.hidden_dim, bias=False)
        self.proj = nn.Linear(4 * config.hidden_dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = F.relu(x).square()  # ReLU^2
        x = self.dropout(x)
        return self.proj(x)


class TransformerBlock(nn.Module):
    """Transformer block with attention + optional atrous conv."""

    def __init__(self, config: PolicyConfigV5, layer_idx: int):
        super().__init__()
        self.time_attn = MultiQueryAttention(config, layer_idx)
        self.mlp = MLP(config)

        # V5: Optional atrous conv for long-range
        if config.use_atrous_conv and layer_idx % 2 == 0:
            self.atrous = AtrousConvBlock(config)
        else:
            self.atrous = None

    def forward(
        self,
        x: torch.Tensor,
        cos_sin: Tuple[torch.Tensor, torch.Tensor],
        is_causal: bool = True,
    ) -> torch.Tensor:
        # Attention
        x = x + self.time_attn(rms_norm(x), cos_sin, is_causal)

        # Optional atrous conv
        if self.atrous is not None:
            x = x + self.atrous(rms_norm(x))

        # MLP
        x = x + self.mlp(rms_norm(x))
        return x


class PortfolioLatentPredictor(nn.Module):
    """
    NEPA-style predictor for portfolio latents.

    Predicts the next latent in the sequence, which can be decoded
    to portfolio weights.
    """

    def __init__(self, config: PolicyConfigV5):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.latent_dim = config.latent_dim

        # Project from hidden to latent space
        self.to_latent = nn.Linear(config.hidden_dim, config.latent_dim, bias=False)

        # Predict next latent (NEPA-style)
        self.predict_next = nn.Sequential(
            nn.Linear(config.latent_dim, config.latent_dim * 2, bias=False),
            nn.GELU(),
            nn.Linear(config.latent_dim * 2, config.latent_dim, bias=False),
        )

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (batch, seq_len, hidden_dim)

        Returns:
            latents: (batch, seq_len, latent_dim) - current latents
            predicted: (batch, seq_len, latent_dim) - predicted next latents
        """
        latents = self.to_latent(h)
        predicted = self.predict_next(latents)
        return latents, predicted


class PortfolioHead(nn.Module):
    """
    Decode portfolio latents to asset weights.

    Outputs:
    - weights: (batch, num_assets) - target portfolio weights (sum to leverage limit)
    - volatility: (batch, num_assets) - predicted volatility per asset
    - confidence: (batch, 1) - overall allocation confidence
    """

    def __init__(self, config: PolicyConfigV5):
        super().__init__()
        self.config = config
        self.num_assets = config.num_assets

        # Project latent to per-asset logits
        self.weight_head = nn.Linear(config.latent_dim, config.num_assets, bias=False)

        if config.output_volatility:
            self.vol_head = nn.Linear(config.latent_dim, config.num_assets, bias=False)
        else:
            self.vol_head = None

        if config.output_confidence:
            self.conf_head = nn.Linear(config.latent_dim, 1, bias=False)
        else:
            self.conf_head = None

    def forward(
        self,
        latent: torch.Tensor,
        asset_class: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            latent: (batch, latent_dim) - pooled portfolio latent
            asset_class: (num_assets,) - 0 for equity, 1 for crypto

        Returns:
            Dict with portfolio outputs
        """
        config = self.config
        device = latent.device
        batch_size = latent.size(0)

        # Raw weight logits
        weight_logits = self.weight_head(latent)  # (batch, num_assets)

        # Softcap
        softcap = config.logits_softcap
        weight_logits = softcap * torch.tanh(weight_logits / softcap)

        # Apply softmax for normalized weights
        # Note: This gives weights that sum to 1.0
        # We'll scale by leverage limit during simulation
        weights = F.softmax(weight_logits, dim=-1)

        outputs = {
            "weight_logits": weight_logits,
            "weights": weights,
        }

        # Volatility prediction
        if self.vol_head is not None:
            vol_logits = self.vol_head(latent)
            # Volatility is always positive, use softplus
            volatility = F.softplus(vol_logits) * 0.01  # Scale to reasonable range
            outputs["volatility"] = volatility

        # Confidence prediction
        if self.conf_head is not None:
            conf_logits = self.conf_head(latent)
            confidence = torch.sigmoid(conf_logits)
            outputs["confidence"] = confidence

        return outputs


class MultiResolutionAggregator(nn.Module):
    """
    Aggregate predictions across multiple time resolutions.

    Uses trimmed mean for robust aggregation (from V4).
    """

    def __init__(self, config: PolicyConfigV5):
        super().__init__()
        self.scales = config.resolution_scales
        self.trim_fraction = config.trim_fraction

        # Project each resolution to common space
        self.projections = nn.ModuleList([
            nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
            for _ in self.scales
        ])

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (batch, seq_len, hidden_dim)

        Returns:
            (batch, hidden_dim) - aggregated representation
        """
        batch_size = h.size(0)
        device = h.device

        representations = []
        for scale, proj in zip(self.scales, self.projections):
            # Aggregate at this scale
            if scale == 1:
                # Use last position
                rep = h[:, -1, :]
            else:
                # Average last `scale` positions
                rep = h[:, -scale:, :].mean(dim=1)

            rep = proj(rep)
            representations.append(rep)

        # Stack: (batch, num_scales, hidden_dim)
        stacked = torch.stack(representations, dim=1)

        # Trimmed mean across scales
        if len(self.scales) >= 3:
            # Sort and trim
            sorted_vals, _ = torch.sort(stacked, dim=1)
            trim_n = max(1, int(len(self.scales) * self.trim_fraction))
            trimmed = sorted_vals[:, trim_n:-trim_n or None, :]
            aggregated = trimmed.mean(dim=1)
        else:
            aggregated = stacked.mean(dim=1)

        return aggregated


class PortfolioPolicyV5(nn.Module):
    """
    V5 Policy network with NEPA-style portfolio latent prediction.

    Key features:
    - Patching: Processes 5-day patches
    - NEPA: Predicts sequence of portfolio latents
    - Atrous: Long-range dependencies via dilated convolutions
    - Multi-resolution: Aggregates across time scales
    - Portfolio output: Normalized weights per asset
    """

    def __init__(self, config: PolicyConfigV5):
        super().__init__()
        self.config = config

        # Patch embedding
        if config.use_patch_embedding:
            self.patch_embed = PatchEmbedding(config)
        else:
            self.patch_embed = None
            self.embed = nn.Linear(config.input_dim, config.hidden_dim, bias=False)

        # Rotary embeddings
        self.rope = RotaryEmbedding(config.head_dim, max_len=config.max_len)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config, i) for i in range(config.num_layers)
        ])

        # V5: Portfolio latent predictor (NEPA-style)
        self.latent_predictor = PortfolioLatentPredictor(config)

        # V5: Multi-resolution aggregator
        if config.use_multi_resolution:
            self.multi_res = MultiResolutionAggregator(config)
        else:
            self.multi_res = None

        # V5: Portfolio output head
        self.portfolio_head = PortfolioHead(config)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                fan_out = module.weight.size(0)
                fan_in = module.weight.size(1)
                std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
                nn.init.normal_(module.weight, mean=0.0, std=std)

        # Small init for output heads
        nn.init.normal_(self.portfolio_head.weight_head.weight, mean=0.0, std=0.01)

        for block in self.blocks:
            nn.init.zeros_(block.time_attn.out_proj.weight)
            nn.init.zeros_(block.mlp.proj.weight)

    def forward(
        self,
        features: torch.Tensor,
        asset_class: Optional[torch.Tensor] = None,
        return_latents: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: (batch, seq_len, input_dim) - normalized features
            asset_class: (num_assets,) - asset class indicators
            return_latents: If True, return intermediate latents for NEPA loss

        Returns:
            Dict with portfolio outputs and optionally latents
        """
        # Patch embedding
        if self.patch_embed is not None:
            h = self.patch_embed(features)
        else:
            h = self.embed(features)

        h = rms_norm(h)

        # Get sequence length after patching
        T = h.size(1)
        cos_sin = self.rope(T)

        # Transformer blocks (causal attention for autoregressive)
        for block in self.blocks:
            h = block(h, cos_sin, is_causal=True)

        h = rms_norm(h)

        # V5: Portfolio latent prediction (NEPA-style)
        latents, predicted_latents = self.latent_predictor(h)

        # Pool representation
        if self.multi_res is not None:
            h_pooled = self.multi_res(h)
        else:
            h_pooled = h[:, -1, :]  # Last position

        # Get final latent for portfolio output
        final_latent = latents[:, -1, :]

        # Portfolio output
        outputs = self.portfolio_head(final_latent, asset_class)

        # Add NEPA loss components if requested
        if return_latents:
            outputs["latents"] = latents
            outputs["predicted_latents"] = predicted_latents
            # Compute NEPA loss
            outputs["nepa_loss"] = nepa_loss(latents, predicted_latents, shift=True)

        return outputs

    def compute_nepa_loss(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute NEPA loss from forward outputs."""
        if "nepa_loss" in outputs:
            return outputs["nepa_loss"]
        elif "latents" in outputs and "predicted_latents" in outputs:
            return nepa_loss(outputs["latents"], outputs["predicted_latents"], shift=True)
        else:
            return torch.tensor(0.0, device=next(self.parameters()).device)
