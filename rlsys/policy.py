"""Actor-critic policy network with modern PyTorch best practices."""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import torch
from torch import nn
from torch.distributions import Normal

from .config import PolicyConfig

_ACTIVATIONS: Dict[str, Callable[[], nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "elu": nn.ELU,
}


def _make_activation(name: str) -> nn.Module:
    try:
        return _ACTIVATIONS[name.lower()]()
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported activation: {name}") from exc


class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: Tuple[int, ...], activation: nn.Module, dropout: float) -> None:
        super().__init__()
        layers = []
        last_dim = input_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(last_dim, hidden))
            layers.append(nn.LayerNorm(hidden))
            layers.append(activation.__class__())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last_dim = hidden
        self.net = nn.Sequential(*layers)
        self.output_dim = last_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - thin wrapper
        return self.net(x)


class ActorCriticPolicy(nn.Module):
    """Gaussian actor-critic policy."""

    def __init__(self, observation_dim: int, config: PolicyConfig) -> None:
        super().__init__()
        activation = _make_activation(config.activation)
        hidden_sizes = tuple(config.hidden_sizes)
        if not hidden_sizes:
            raise ValueError("At least one hidden layer is required")

        if config.shared_backbone:
            self.backbone = _MLP(observation_dim, hidden_sizes, activation, config.dropout)
            actor_input_dim = self.backbone.output_dim
            critic_input_dim = self.backbone.output_dim
        else:
            self.backbone = None
            self.actor_body = _MLP(observation_dim, hidden_sizes, activation, config.dropout)
            self.critic_body = _MLP(observation_dim, hidden_sizes, activation, config.dropout)
            actor_input_dim = self.actor_body.output_dim
            critic_input_dim = self.critic_body.output_dim

        policy_layers = []
        last_dim = actor_input_dim
        for _ in range(config.policy_head_layers - 1):
            policy_layers.extend(
                [
                    nn.Linear(last_dim, last_dim),
                    nn.LayerNorm(last_dim),
                    activation.__class__(),
                ]
            )
        self.policy_head = nn.Sequential(*policy_layers, nn.Linear(last_dim, 1))

        value_layers = []
        last_dim_v = critic_input_dim
        for _ in range(config.value_head_layers - 1):
            value_layers.extend(
                [
                    nn.Linear(last_dim_v, last_dim_v),
                    nn.LayerNorm(last_dim_v),
                    activation.__class__(),
                ]
            )
        self.value_head = nn.Sequential(*value_layers, nn.Linear(last_dim_v, 1))

        self.log_std = nn.Parameter(torch.zeros(1))
        self.log_std_bounds = config.log_std_bounds

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _forward_backbone(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.backbone is not None:
            hidden = self.backbone(observation)
            return hidden, hidden
        actor_hidden = self.actor_body(observation)
        critic_hidden = self.critic_body(observation)
        return actor_hidden, critic_hidden

    def forward(self, observation: torch.Tensor) -> Tuple[Normal, torch.Tensor]:
        actor_hidden, critic_hidden = self._forward_backbone(observation)
        mean = self.policy_head(actor_hidden)
        value = self.value_head(critic_hidden)
        log_std = torch.clamp(self.log_std, *self.log_std_bounds)
        std = torch.exp(log_std)
        return Normal(mean, std), value

    def act(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, value = self.forward(observation)
        action = distribution.sample()
        log_prob = distribution.log_prob(action).sum(-1)
        return action, log_prob, value

    def evaluate_actions(
        self, observation: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, value = self.forward(observation)
        log_probs = distribution.log_prob(actions).sum(-1)
        entropy = distribution.entropy().sum(-1)
        return log_probs, entropy, value
