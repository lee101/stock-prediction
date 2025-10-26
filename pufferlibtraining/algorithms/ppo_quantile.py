from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple

import torch
import torch.nn as nn

from pufferlibtraining.distributional.quantile import QuantileValueHead, quantile_huber_loss
from pufferlibtraining.utils.ppo_utils import compute_gae, minibatch_iterator


@dataclass
class PPOQuantileCfg:
    n_quantiles: int = 32
    kappa: float = 1.0
    tau_min: float = 0.01
    tau_max: float = 0.99
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    clip_coef: float = 0.2
    max_grad_norm: float = 1.0
    lr: float = 3e-4


class ActorCriticQuantile(nn.Module):
    """
    Actor-Critic module with a quantile-valued critic head.
    """

    def __init__(self, body: nn.Module, body_dim: int, n_actions: int, cfg: PPOQuantileCfg):
        super().__init__()
        self.body = body
        self.policy_head = nn.Linear(body_dim, n_actions)
        self.quantile_head = QuantileValueHead(body_dim, cfg.n_quantiles)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.body(obs)
        logits = self.policy_head(features)
        quantiles = self.quantile_head(features)
        return logits, quantiles, features


def ppo_update_quantile(
    model: ActorCriticQuantile,
    optimizer: torch.optim.Optimizer,
    batch,
    cfg: PPOQuantileCfg,
    device: torch.device,
) -> None:
    obs, actions, rewards, dones, bootstrap_value, logp_old = batch

    with torch.no_grad():
        logits, quantiles, _ = model(obs)
        baseline = quantiles.median(dim=-1).values.unsqueeze(-1)
    advantages, returns = compute_gae(rewards, dones, baseline, bootstrap_value)

    taus = torch.linspace(cfg.tau_min, cfg.tau_max, cfg.n_quantiles, device=device)

    for obs_b, act_b, adv_b, ret_b, logp_old_b in minibatch_iterator(
        obs, actions, advantages, returns, logp_old
    ):
        logits, quantiles, _ = model(obs_b)

        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(act_b)
        ratio = torch.exp(logp - logp_old_b)
        clipped = torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
        policy_loss = -torch.min(ratio * adv_b, clipped * adv_b).mean()
        entropy = dist.entropy().mean()

        value_loss = quantile_huber_loss(quantiles, ret_b.unsqueeze(-1), taus, kappa=cfg.kappa)

        loss = policy_loss - cfg.ent_coef * entropy + cfg.vf_coef * value_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()
