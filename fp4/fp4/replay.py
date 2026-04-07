"""Circular GPU-resident replay buffer for off-policy RL (SAC).

All storage and sampling stays on the target device — no `.cpu()`, no `.item()`
calls on the hot path — so SAC can run fully GPU-resident.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch


class GPUReplayBuffer:
    """Fixed-size circular buffer. Stores flat per-env transitions.

    ``add`` accepts batched transitions shaped ``(N, ...)`` from a vector env;
    they are written into the ring contiguously (with wrap).
    ``sample`` draws ``batch_size`` uniform random rows using torch.randint on
    device, so no host sync is required.
    """

    def __init__(self, capacity: int, obs_dim: int, act_dim: int,
                 device: torch.device, n_costs: int = 0, dtype: torch.dtype = torch.float32):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.n_costs = int(n_costs)
        self.device = device
        self.dtype = dtype
        self.obs = torch.zeros(self.capacity, obs_dim, device=device, dtype=dtype)
        self.act = torch.zeros(self.capacity, act_dim, device=device, dtype=dtype)
        self.rew = torch.zeros(self.capacity, device=device, dtype=dtype)
        self.next_obs = torch.zeros(self.capacity, obs_dim, device=device, dtype=dtype)
        self.done = torch.zeros(self.capacity, device=device, dtype=dtype)
        if n_costs > 0:
            self.costs = torch.zeros(self.capacity, n_costs, device=device, dtype=dtype)
        else:
            self.costs = None
        # ``_ptr`` / ``_size`` are plain Python ints — safe: we never read
        # them inside a captured graph; they only control where to write.
        self._ptr = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    @property
    def size(self) -> int:
        return self._size

    def add(self, obs: torch.Tensor, act: torch.Tensor, rew: torch.Tensor,
            next_obs: torch.Tensor, done: torch.Tensor,
            costs: Optional[torch.Tensor] = None) -> None:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            act = act.unsqueeze(0)
            rew = rew.unsqueeze(0) if rew.dim() == 0 else rew
            next_obs = next_obs.unsqueeze(0)
            done = done.unsqueeze(0) if done.dim() == 0 else done
        n = obs.shape[0]
        if n == 0:
            return
        # Cast/move once.
        obs = obs.to(device=self.device, dtype=self.dtype, non_blocking=True)
        act = act.to(device=self.device, dtype=self.dtype, non_blocking=True)
        rew = rew.to(device=self.device, dtype=self.dtype, non_blocking=True).reshape(-1)
        next_obs = next_obs.to(device=self.device, dtype=self.dtype, non_blocking=True)
        done = done.to(device=self.device, dtype=self.dtype, non_blocking=True).reshape(-1)

        end = self._ptr + n
        if end <= self.capacity:
            sl = slice(self._ptr, end)
            self.obs[sl] = obs
            self.act[sl] = act
            self.rew[sl] = rew
            self.next_obs[sl] = next_obs
            self.done[sl] = done
            if self.costs is not None and costs is not None:
                self.costs[sl] = costs.to(device=self.device, dtype=self.dtype, non_blocking=True)
        else:
            first = self.capacity - self._ptr
            second = n - first
            self.obs[self._ptr:] = obs[:first]
            self.obs[:second] = obs[first:]
            self.act[self._ptr:] = act[:first]
            self.act[:second] = act[first:]
            self.rew[self._ptr:] = rew[:first]
            self.rew[:second] = rew[first:]
            self.next_obs[self._ptr:] = next_obs[:first]
            self.next_obs[:second] = next_obs[first:]
            self.done[self._ptr:] = done[:first]
            self.done[:second] = done[first:]
            if self.costs is not None and costs is not None:
                c = costs.to(device=self.device, dtype=self.dtype, non_blocking=True)
                self.costs[self._ptr:] = c[:first]
                self.costs[:second] = c[first:]
        self._ptr = (self._ptr + n) % self.capacity
        self._size = min(self._size + n, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        if self._size == 0:
            raise RuntimeError("sample() called on empty buffer")
        high = self._size
        idx = torch.randint(0, high, (int(batch_size),), device=self.device)
        out: Dict[str, torch.Tensor] = {
            "obs": self.obs.index_select(0, idx),
            "act": self.act.index_select(0, idx),
            "rew": self.rew.index_select(0, idx),
            "next_obs": self.next_obs.index_select(0, idx),
            "done": self.done.index_select(0, idx),
        }
        if self.costs is not None:
            out["costs"] = self.costs.index_select(0, idx)
        return out
