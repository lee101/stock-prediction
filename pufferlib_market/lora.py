"""LoRA (Low-Rank Adaptation) for trading policy test-time fine-tuning.

Inspired by cutellm's per-document LoRA TTT approach:
- Add low-rank adapters to the last K linear layers
- Freeze base model weights
- Fine-tune only LoRA adapters on recent market observations
- Use at evaluation time to adapt to current market regime

This helps the policy adapt to regime changes (e.g., high vs low volatility periods).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from pufferlib_market.train import _mask_short_logits


class LoRALinear(nn.Module):
    """Linear layer with a low-rank LoRA adapter.

    Output = base(x) + (x @ lora_a.T) @ lora_b.T * (alpha / rank)

    The B matrix is initialised to zero so the adapter is identity at init.
    The A matrix is initialised with small Gaussian noise.
    Base weights are frozen (requires_grad=False).
    """

    def __init__(self, base_linear: nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        self.rank = rank
        self.alpha = alpha

        # Freeze base weights — we never train them during TTT
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad_(False)

        in_features = base_linear.in_features
        out_features = base_linear.out_features

        # lora_a: (rank, in_features), lora_b: (out_features, rank)
        # Using small random A and zero B so the adapter starts as identity delta
        self.lora_a = nn.Parameter(
            torch.randn(rank, in_features) * 0.01
        )
        self.lora_b = nn.Parameter(
            torch.zeros(out_features, rank)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        # LoRA delta: x @ lora_a.T -> (batch, rank), then @ lora_b.T -> (batch, out)
        lora_delta = F.linear(F.linear(x, self.lora_a), self.lora_b)
        return base_out + lora_delta * (self.alpha / self.rank)

    def reset_lora(self) -> None:
        """Reset adapter to identity (zero B, re-init A with small noise)."""
        with torch.no_grad():
            self.lora_b.zero_()
            self.lora_a.normal_(0.0, 0.01)


def _collect_linear_layers(module: nn.Module) -> list[tuple[str, nn.Module, str, nn.Linear]]:
    """Return (parent_path, parent_module, attr_name, linear) for all Linear layers in DFS order."""
    results: list[tuple[str, nn.Module, str, nn.Linear]] = []
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            results.append(("", module, name, child))
        else:
            sub = _collect_linear_layers(child)
            for (path, parent, attr, lin) in sub:
                results.append((f"{name}.{path}" if path else name, parent, attr, lin))
    return results


def _find_actor_submodule(policy: nn.Module) -> nn.Module | None:
    """Return the actor submodule if the policy has one, otherwise None."""
    for name in ("actor", "policy_head", "policy"):
        sub = getattr(policy, name, None)
        if sub is not None and isinstance(sub, nn.Module):
            return sub
    return None


class LoRAPolicy(nn.Module):
    """Wraps a TradingPolicy adding LoRA adapters to the last ``num_layers`` Linear layers
    of the actor (policy head) submodule.

    Targeting the actor ensures gradients flow from the action distribution back through
    the LoRA parameters during calibration.  If no actor submodule is found, falls back
    to the last ``num_layers`` linear layers of the full policy.

    Base policy weights are frozen. Only LoRA parameters are trainable.
    Call ``reset_lora()`` at the start of each new episode to reset adapters.
    """

    def __init__(self, base_policy: nn.Module, rank: int = 4, num_layers: int = 2):
        super().__init__()
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")

        self.base_policy = base_policy
        self.rank = rank
        self.num_layers = num_layers

        # Freeze ALL base policy parameters first
        for p in self.base_policy.parameters():
            p.requires_grad_(False)

        # Prefer to inject LoRA into the actor head for proper gradient flow
        # (actor logits → log_prob → loss → LoRA params).
        actor_sub = _find_actor_submodule(self.base_policy)
        search_root = actor_sub if actor_sub is not None else self.base_policy

        all_linears = _collect_linear_layers(search_root)
        if not all_linears:
            # Fallback: search entire policy
            all_linears = _collect_linear_layers(self.base_policy)
        if not all_linears:
            raise ValueError("base_policy has no Linear layers to wrap with LoRA")

        # Replace the last `num_layers` linear layers in the search root with LoRALinear
        target_linears = all_linears[-num_layers:]
        self._lora_layers: list[LoRALinear] = []

        for (_path, parent, attr, lin) in target_linears:
            lora_lin = LoRALinear(lin, rank=rank, alpha=float(rank))
            setattr(parent, attr, lora_lin)
            self._lora_layers.append(lora_lin)

        # Register LoRA layers as submodules so their parameters are visible
        for i, ll in enumerate(self._lora_layers):
            self.add_module(f"_lora_{i}", ll)

    def lora_parameters(self) -> list[nn.Parameter]:
        """Return only the LoRA adapter parameters (for the optimizer)."""
        params: list[nn.Parameter] = []
        for ll in self._lora_layers:
            params.append(ll.lora_a)
            params.append(ll.lora_b)
        return params

    def reset_lora(self) -> None:
        """Reset all LoRA adapters to identity (call at start of each episode)."""
        for ll in self._lora_layers:
            ll.reset_lora()

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        *,
        disable_shorts: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward through base policy (with active LoRA layers), return PPO tuple."""
        logits, value = self.base_policy(obs)
        if disable_shorts:
            logits = _mask_short_logits(logits, logits.shape[-1])
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        disable_shorts: bool = False,
    ) -> torch.Tensor:
        """Forward through base policy (with active LoRA layers), return action."""
        logits, _ = self.base_policy(obs)
        if disable_shorts:
            logits = _mask_short_logits(logits, logits.shape[-1])
        if deterministic:
            return logits.argmax(dim=-1)
        return Categorical(logits=logits).sample()

    def forward(self, x: torch.Tensor):
        """Delegate forward to base policy (LoRA layers are active inside it)."""
        return self.base_policy(x)


def reset_adam_state(optimizer: torch.optim.Optimizer) -> None:
    """Reset Adam momentum buffers without recreating the optimizer."""
    for state in optimizer.state.values():
        if "step" in state:
            if torch.is_tensor(state["step"]):
                state["step"].zero_()
            else:
                state["step"] = 0
        if "exp_avg" in state:
            state["exp_avg"].zero_()
        if "exp_avg_sq" in state:
            state["exp_avg_sq"].zero_()
