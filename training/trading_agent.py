import torch
import torch.nn as nn
from typing import Tuple, Optional
import numpy as np


class TradingAgent(nn.Module):
    def __init__(
        self,
        backbone_model=None,
        hidden_dim: int = 768,
        action_std_init: float = 0.5,
        use_pretrained_toto: bool = False
    ):
        super().__init__()
        
        if use_pretrained_toto:
            try:
                from toto.model.toto import Toto
                base = Toto.from_pretrained('Datadog/Toto-Open-Base-1.0')
                self.backbone = base.model
                hidden_dim = self.backbone.config.hidden_size if hasattr(self.backbone, 'config') else 768
            except ImportError:
                print("Toto not available, using provided backbone or creating simple MLP")
                self.backbone = backbone_model or self._create_simple_backbone(hidden_dim)
        else:
            self.backbone = backbone_model or self._create_simple_backbone(hidden_dim)
        
        self.hidden_dim = hidden_dim
        
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )
        
        self.action_var = nn.Parameter(torch.full((1,), action_std_init * action_std_init))
        
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def _create_simple_backbone(self, hidden_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(100, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if hasattr(self.backbone, '__call__'):
            features = self.backbone(state)
            if isinstance(features, (tuple, list)):
                features = features[0]
            if len(features.shape) > 2:
                features = features[:, -1, :]
        else:
            features = state
        
        action_mean = self.actor_mean(features)
        value = self.critic(features)
        
        return action_mean, value
    
    def act(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_mean, value = self.forward(state)
        
        if deterministic:
            action = action_mean
            action_logprob = torch.zeros_like(action)
        else:
            action_std = self.action_var.expand_as(action_mean).sqrt()
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
        
        action = torch.clamp(action, -1.0, 1.0)
        
        return action, action_logprob, value
    
    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_mean, value = self.forward(state)
        
        action_std = self.action_var.expand_as(action_mean).sqrt()
        dist = torch.distributions.Normal(action_mean, action_std)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        return action_logprobs, value, dist_entropy