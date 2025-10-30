"""Lightweight actor-critic policy for MarketEnv."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


def _activation(name: str) -> nn.Module:
    key = name.lower()
    if key in {"relu"}:
        return nn.ReLU()
    if key in {"gelu"}:
        return nn.GELU()
    if key in {"silu", "swish"}:
        return nn.SiLU()
    if key in {"elu"}:
        return nn.ELU()
    raise ValueError(f"Unsupported activation '{name}'")


@dataclass
class PolicyConfig:
    hidden_size: int = 256
    actor_layers: Sequence[int] = field(default_factory=lambda: (256, 256))
    critic_layers: Sequence[int] = field(default_factory=lambda: (256, 256))
    activation: str = "swish"
    dropout_p: float = 0.0
    layer_norm: bool = True
    use_lstm: bool = False
    rnn_hidden_size: int = 256


def _mlp(  # pragma: no cover - exercised via policy tests
    input_dim: int,
    widths: Iterable[int],
    *,
    activation: nn.Module,
    layer_norm: bool,
    dropout_p: float,
) -> Tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    prev = input_dim
    for width in widths:
        linear = nn.Linear(prev, width)
        nn.init.orthogonal_(linear.weight, gain=np.sqrt(2))
        nn.init.zeros_(linear.bias)
        layers.append(linear)
        if layer_norm:
            layers.append(nn.LayerNorm(width))
        layers.append(activation.__class__())
        if dropout_p > 0.0:
            layers.append(nn.Dropout(dropout_p))
        prev = width
    return nn.Sequential(*layers), prev


class MarketPolicy(nn.Module):
    """Feed-forward actor-critic head compatible with PuffeRL."""

    def __init__(self, env, cfg: PolicyConfig):
        super().__init__()
        import pufferlib.spaces as pl_spaces  # local import to avoid eager dependency

        self.cfg = cfg
        obs_shape = env.single_observation_space.shape
        self.obs_dim = int(np.prod(obs_shape))

        self._is_multi_discrete = isinstance(env.single_action_space, pl_spaces.MultiDiscrete)
        self._is_discrete = isinstance(env.single_action_space, pl_spaces.Discrete)
        self._is_continuous = isinstance(env.single_action_space, pl_spaces.Box)
        if self._is_multi_discrete:
            self._action_dims = tuple(int(n) for n in env.single_action_space.nvec)
            self._action_total = sum(self._action_dims)
        elif self._is_continuous:
            self._action_total = int(np.prod(env.single_action_space.shape))
        else:
            self._action_total = int(env.single_action_space.n)

        activation = _activation(cfg.activation)
        encoder_layers = [nn.Linear(self.obs_dim, cfg.hidden_size)]
        nn.init.orthogonal_(encoder_layers[0].weight, gain=np.sqrt(2))
        nn.init.zeros_(encoder_layers[0].bias)
        if cfg.layer_norm:
            encoder_layers.append(nn.LayerNorm(cfg.hidden_size))
        encoder_layers.append(activation.__class__())
        if cfg.dropout_p > 0.0:
            encoder_layers.append(nn.Dropout(cfg.dropout_p))
        self.encoder = nn.Sequential(*encoder_layers)

        actor_tower, actor_dim = _mlp(
            cfg.hidden_size,
            cfg.actor_layers,
            activation=activation,
            layer_norm=cfg.layer_norm,
            dropout_p=cfg.dropout_p,
        )
        critic_tower, critic_dim = _mlp(
            cfg.hidden_size,
            cfg.critic_layers,
            activation=activation,
            layer_norm=cfg.layer_norm,
            dropout_p=cfg.dropout_p,
        )
        self.actor_tower = actor_tower
        self.critic_tower = critic_tower

        if self._is_multi_discrete:
            self.actor_head = nn.Linear(actor_dim, self._action_total)
            nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
            nn.init.zeros_(self.actor_head.bias)
        elif self._is_continuous:
            self.actor_mean = nn.Linear(actor_dim, self._action_total)
            nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
            nn.init.zeros_(self.actor_mean.bias)
            self.actor_logstd = nn.Parameter(torch.zeros(self._action_total))
        else:
            self.actor_head = nn.Linear(actor_dim, self._action_total)
            nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
            nn.init.zeros_(self.actor_head.bias)

        self.value_head = nn.Linear(critic_dim, 1)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

        self.hidden_size = cfg.hidden_size

    def forward_eval(self, observations: torch.Tensor, state=None):  # noqa: D401 - interface defined by PuffeRL
        batch = observations.shape[0]
        flat = observations.view(batch, -1)
        embedding = self.encoder(flat)
        actor_hidden = self.actor_tower(embedding)
        critic_hidden = self.critic_tower(embedding)

        if self._is_multi_discrete:
            logits = self.actor_head(actor_hidden).split(self._action_dims, dim=1)
        elif self._is_continuous:
            mean = self.actor_mean(actor_hidden)
            std = torch.exp(self.actor_logstd).expand_as(mean)
            logits = torch.distributions.Normal(mean, std)
        else:
            logits = self.actor_head(actor_hidden)

        values = self.value_head(critic_hidden)
        return logits, values

    def forward(self, observations: torch.Tensor, state=None):
        return self.forward_eval(observations, state=state)


__all__ = ["PolicyConfig", "MarketPolicy"]
