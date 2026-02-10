from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np


class TradingPolicy(nn.Module):
    def __init__(self, env, hidden_dim: int = 256):
        super().__init__()
        obs_dim = int(np.prod(env.single_observation_space.shape))
        n_actions = env.single_action_space.n

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, n_actions)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h)

    def forward_eval(self, x, state=None):
        h = self.shared(x)
        logits = self.actor(h)
        value = self.critic(h)
        return logits, value, state
