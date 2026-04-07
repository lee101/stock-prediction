"""GPU-resident vectorized env for fast-SPS RL.

Tries to wrap Unit C's `market_sim_py` bindings if importable; otherwise
falls back to a torch-only synthetic OHLC random-walk env. All tensors
returned live on the requested device — no .cpu()/.item() in the hot loop.

The synthetic env is intentionally cheap so that env_step + policy_forward
can be benchmarked end-to-end on GPU. Determinism is controlled by `seed`.

API (gym-ish, batched):
    env = GPUVecEnv(num_envs=N, obs_dim=D, act_dim=A, device='cuda', seed=0)
    obs = env.reset()                    # (N, D) tensor
    out = env.step(action)               # dict-of-tensors, all on device
        out['obs']      (N, D)
        out['reward']   (N,)
        out['done']     (N,) bool
        out['info_pnl'] (N,)             optional running pnl

No Python int/float syncs in the hot loop.
"""
from __future__ import annotations

from typing import Optional

import torch


def _try_import_market_sim_py():
    try:
        import market_sim_py  # noqa: F401
        return market_sim_py
    except Exception:
        return None


class SyntheticOHLCEnv:
    """Torch-only batched random-walk env. Pure tensor ops, GPU-resident.

    Observation = last `obs_dim` log-returns (rolling window). Action is a
    continuous (N, act_dim) tensor in [-1,1]; reward = action[...,0] * next_ret
    (a directional bet) minus a tiny fee. Episodes auto-reset every
    `episode_len` steps via masked reset (no Python branches per env).
    """

    def __init__(
        self,
        num_envs: int,
        obs_dim: int = 16,
        act_dim: int = 1,
        device: str | torch.device = "cuda",
        seed: int = 0,
        episode_len: int = 256,
        fee_bps: float = 1.0,
        vol: float = 0.01,
    ) -> None:
        self.num_envs = int(num_envs)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.device = torch.device(device)
        self.episode_len = int(episode_len)
        self.fee = float(fee_bps) * 1e-4
        self.vol = float(vol)
        # Default device generator is used for RNG so it's safe inside CUDA
        # graph capture (custom generators are not capture-mode aware as of
        # torch 2.9). Determinism is enforced by re-seeding on reset().
        self._seed = int(seed)
        # State buffers (persistent so CUDA graph capture sees stable addrs).
        self._returns = torch.zeros(self.num_envs, self.obs_dim, device=self.device)
        self._step = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._pnl = torch.zeros(self.num_envs, device=self.device)
        self._prev_action = torch.zeros(self.num_envs, self.act_dim, device=self.device)

    # ---- gym-ish API ----
    def reset(self) -> torch.Tensor:
        if self.device.type == "cuda":
            torch.cuda.manual_seed(self._seed)
        else:
            torch.manual_seed(self._seed)
        self._returns.normal_().mul_(self.vol)
        self._step.zero_()
        self._pnl.zero_()
        self._prev_action.zero_()
        return self._returns.clone()

    @torch.no_grad()
    def step(self, action: torch.Tensor) -> dict:
        # Generate next return then roll the window in-place.
        next_ret = torch.randn(self.num_envs, device=self.device).mul_(self.vol)
        # Roll window: shift left, append new return.
        self._returns[:, :-1] = self._returns[:, 1:].clone()
        self._returns[:, -1] = next_ret
        # Reward: directional bet on first action dim, fee on action change.
        direction = action[:, 0].clamp(-1.0, 1.0)
        reward = direction * next_ret
        delta = (action - self._prev_action).abs().sum(dim=-1)
        reward = reward - self.fee * delta
        self._pnl.add_(reward)
        self._prev_action.copy_(action)
        # Step counter + masked auto-reset.
        self._step.add_(1)
        done = self._step >= self.episode_len
        # Branchless masked reset (no host sync, graph-capture safe).
        mask = done.to(self._returns.dtype).unsqueeze(-1)
        new_window = torch.randn(
            self.num_envs, self.obs_dim, device=self.device
        ).mul_(self.vol)
        self._returns.mul_(1 - mask).add_(new_window * mask)
        not_done_i = (~done).to(self._step.dtype)
        self._step.mul_(not_done_i)
        self._pnl.mul_(1 - mask.squeeze(-1))
        self._prev_action.mul_(1 - mask)
        return {
            "obs": self._returns,
            "reward": reward,
            "done": done,
            "info_pnl": self._pnl,
        }


class _MarketSimPyWrapper:
    """Thin adapter so `market_sim_py.MarketEnvironment` looks like SyntheticOHLCEnv.

    This wrapper is intentionally minimal: it forwards reset/step and ensures
    outputs land on the requested device as torch tensors. If the binding
    cannot be constructed (missing data, version skew), we raise so the caller
    can fall back to the synthetic env.
    """

    def __init__(self, num_envs: int, device, seed: int, **kwargs) -> None:
        msp = _try_import_market_sim_py()
        if msp is None:
            raise RuntimeError("market_sim_py not importable")
        # Try a generic constructor; the real binding may evolve in Unit C.
        self._env = msp.MarketEnvironment(num_envs=num_envs, seed=seed, **kwargs)
        self.num_envs = num_envs
        self.device = torch.device(device)
        sample = self._env.reset()
        self.obs_dim = int(sample.shape[-1])
        self.act_dim = int(getattr(self._env, "action_dim", 1))

    def reset(self) -> torch.Tensor:
        obs = self._env.reset()
        return torch.as_tensor(obs, device=self.device)

    def step(self, action: torch.Tensor) -> dict:
        out = self._env.step(action)
        if isinstance(out, dict):
            return {k: torch.as_tensor(v, device=self.device) for k, v in out.items()}
        # tuple form (obs, reward, done, info)
        obs, reward, done, info = out
        return {
            "obs": torch.as_tensor(obs, device=self.device),
            "reward": torch.as_tensor(reward, device=self.device),
            "done": torch.as_tensor(done, device=self.device),
            "info_pnl": torch.as_tensor(info.get("pnl", reward * 0.0), device=self.device)
            if isinstance(info, dict) else torch.zeros_like(torch.as_tensor(reward)),
        }


def GPUVecEnv(
    num_envs: int = 256,
    obs_dim: int = 16,
    act_dim: int = 1,
    device: Optional[str | torch.device] = None,
    seed: int = 0,
    prefer_market_sim: bool = True,
    **kwargs,
):
    """Construct the best available GPU-resident vectorized env.

    Tries `market_sim_py` first (if `prefer_market_sim`), else falls back to
    `SyntheticOHLCEnv`. Always returns an object with `.reset()` and
    `.step(action)` returning dict-of-tensors on `device`.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if prefer_market_sim and _try_import_market_sim_py() is not None:
        try:
            return _MarketSimPyWrapper(num_envs=num_envs, device=device, seed=seed, **kwargs)
        except Exception:
            pass
    return SyntheticOHLCEnv(
        num_envs=num_envs,
        obs_dim=obs_dim,
        act_dim=act_dim,
        device=device,
        seed=seed,
    )
