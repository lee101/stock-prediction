from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import torch
import torch.nn as nn
from torch.distributions import Independent, Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform


@dataclass(frozen=True)
class PolicyConfig:
    lookback: int
    num_assets: int
    feature_dim: int
    portfolio_dim: int
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    dropout: float = 0.1
    mlp_hidden: int = 256

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AttentionBlock(nn.Module):
    def __init__(self, *, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = self.attn_norm(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x, need_weights=True)
        x = x + attn_out
        return x + self.mlp(self.mlp_norm(x))


class RiskAwareActorCritic(nn.Module):
    def __init__(self, config: PolicyConfig) -> None:
        super().__init__()
        self.config = config
        self.seq_width = int(config.num_assets) * int(config.feature_dim)
        self.seq_size = int(config.lookback) * self.seq_width
        self.portfolio_offset = self.seq_size

        self.input_proj = nn.Linear(self.seq_width, int(config.d_model))
        self.pos_embedding = nn.Parameter(torch.zeros(1, int(config.lookback), int(config.d_model)))
        self.encoder = nn.ModuleList(
            [
                AttentionBlock(
                    d_model=int(config.d_model),
                    n_heads=int(config.n_heads),
                    dropout=float(config.dropout),
                )
                for _ in range(int(config.n_layers))
            ]
        )
        self.portfolio_proj = nn.Sequential(
            nn.Linear(int(config.portfolio_dim), int(config.d_model) // 2),
            nn.GELU(),
            nn.LayerNorm(int(config.d_model) // 2),
        )
        self.trunk = nn.Sequential(
            nn.Linear(int(config.d_model) + int(config.d_model) // 2, int(config.mlp_hidden)),
            nn.GELU(),
            nn.LayerNorm(int(config.mlp_hidden)),
            nn.Dropout(float(config.dropout)),
            nn.Linear(int(config.mlp_hidden), int(config.mlp_hidden)),
            nn.GELU(),
        )
        self.policy_head = nn.Linear(int(config.mlp_hidden), int(config.num_assets))
        self.value_head = nn.Linear(int(config.mlp_hidden), 1)
        self.log_std = nn.Parameter(torch.full((int(config.num_assets),), -1.0, dtype=torch.float32))

        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.zeros_(self.value_head.bias)

    def _split_observation(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq = obs[..., : self.portfolio_offset]
        portfolio = obs[..., self.portfolio_offset :]
        seq = seq.view(obs.shape[0], int(self.config.lookback), self.seq_width)
        return seq, portfolio

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        seq, portfolio = self._split_observation(obs)
        hidden = self.input_proj(seq) + self.pos_embedding
        encoded = hidden
        for block in self.encoder:
            encoded = block(encoded)
        seq_summary = encoded[:, -1]
        portfolio_summary = self.portfolio_proj(portfolio)
        return self.trunk(torch.cat([seq_summary, portfolio_summary], dim=-1))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self._encode(obs)
        mean = self.policy_head(hidden)
        value = self.value_head(hidden).squeeze(-1)
        return mean, value

    def distribution(self, obs: torch.Tensor) -> tuple[Independent, Independent, torch.Tensor]:
        mean, value = self.forward(obs)
        std = self.log_std.exp().expand_as(mean)
        base_dist = Independent(Normal(mean, std), 1)
        dist = Independent(
            TransformedDistribution(base_dist.base_dist, [TanhTransform(cache_size=1)]),
            1,
        )
        return dist, base_dist, value

    def act(self, obs: torch.Tensor, *, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, _, value = self.distribution(obs)
        if deterministic:
            mean, _ = self.forward(obs)
            action = torch.tanh(mean)
            log_prob = dist.log_prob(action)
            return action, log_prob, value
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def predict_deterministic(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, value = self.forward(obs)
        return torch.tanh(mean), value

    def evaluate_actions(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, base_dist, value = self.distribution(obs)
        log_prob = dist.log_prob(action)
        entropy = base_dist.entropy()
        return log_prob, entropy, value
