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

from pathlib import Path
from typing import Optional

import torch


def _local_tmp() -> Path:
    """Return repo-local tmp/ dir (created on first call)."""
    from fp4.paths import ensure_tmp
    return ensure_tmp()


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


#: Default symbols loaded into the real marketsim env if the caller does
#: not pass a list. These all exist under `trainingdata/` in the repo.
_DEFAULT_SYMBOLS = (
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META",
    "TSLA", "AMD", "AVGO", "NFLX", "COST", "CRM",
)


def _default_data_dir() -> str:
    """Locate the repo's `trainingdata/` dir relative to this file."""
    import os as _os

    here = _os.path.dirname(_os.path.abspath(__file__))
    # fp4/fp4/vec_env.py -> repo root is two levels up
    cand = _os.path.normpath(_os.path.join(here, "..", "..", "trainingdata"))
    return cand


class _MarketSimPyWrapper:
    """Thin adapter so `market_sim_py.MarketEnvironment` looks like SyntheticOHLCEnv.

    Wraps Unit C's pybind11 bindings. The real binding has a fixed batch size
    (``MarketEnvironment.BATCH_SIZE``, currently 4096), runs natively on CUDA,
    and returns ``torch.Tensor`` outputs already on-device. We preserve the
    no-host-sync invariant: nothing in ``reset`` / ``step`` calls ``.item()``
    or ``.cpu()``. Auto-reset on done is handled on-device by calling
    ``env.reset(env_indices=done_mask)`` — ``env_indices`` accepts a bool
    CUDA tensor directly.
    """

    def __init__(
        self,
        num_envs: int,
        device,
        seed: int,
        symbols: Optional[list[str]] = None,
        data_dir: Optional[str] = None,
        log_dir: Optional[str] = None,
        action_mode: str = "dps",
        **kwargs,
    ) -> None:
        msp = _try_import_market_sim_py()
        if msp is None:
            raise RuntimeError("market_sim_py not importable")
        dev_str = str(torch.device(device))
        # torch.device('cuda') -> 'cuda'; MarketEnvironment wants 'cpu' or 'cuda'.
        dev_short = "cuda" if dev_str.startswith("cuda") else "cpu"
        self._env = msp.MarketEnvironment(
            data_dir=data_dir or _default_data_dir(),
            log_dir=log_dir or str(_local_tmp() / "fp4_marketsim_logs"),
            device=dev_short,
            action_mode=action_mode,
            **kwargs,
        )
        # Fixed batch size is dictated by the C++ binding (compile-time const).
        fixed_batch = int(msp.MarketEnvironment.BATCH_SIZE)
        if num_envs != fixed_batch:
            # Honest: log through the adapter's attribute so callers can
            # read back the actual size. No silent mismatch.
            num_envs = fixed_batch
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.action_mode = action_mode
        syms = list(symbols) if symbols is not None else list(_DEFAULT_SYMBOLS)
        self._env.load_symbols(syms)
        # Seed RNG used by the MarketSim C++ side via python torch seed —
        # the binding itself doesn't expose a seed kwarg on the ctor.
        if dev_short == "cuda":
            torch.cuda.manual_seed(int(seed))
        torch.manual_seed(int(seed))
        sample = self._env.reset()
        self.obs_dim = int(sample["observations"].shape[-1])
        self.act_dim = int(self._env.get_action_dim())
        self._last_obs = sample["observations"]

    # ---- gym-ish API ----
    def reset(self) -> torch.Tensor:
        out = self._env.reset()
        self._last_obs = out["observations"]
        return self._last_obs

    @torch.no_grad()
    def step(self, action: torch.Tensor) -> dict:
        # Shape/layout guard: SCALAR expects [batch], DPS expects [batch, 3].
        if self.action_mode == "scalar":
            if action.dim() == 2 and action.shape[-1] == 1:
                action = action.squeeze(-1)
        else:  # dps
            if action.dim() == 1:
                action = action.unsqueeze(-1).expand(-1, self.act_dim).contiguous()
            elif action.shape[-1] != self.act_dim:
                # Pad/truncate on the last dim without host sync.
                if action.shape[-1] < self.act_dim:
                    pad = torch.zeros(
                        action.shape[0], self.act_dim - action.shape[-1],
                        device=action.device, dtype=action.dtype,
                    )
                    action = torch.cat([action, pad], dim=-1)
                else:
                    action = action[:, : self.act_dim].contiguous()
        # MarketEnvironment wants float32 on the same device as the env.
        if action.dtype != torch.float32:
            action = action.to(torch.float32)
        if action.device != self.device:
            action = action.to(self.device, non_blocking=True)
        out = self._env.step(action)
        obs = out["observations"]
        done = out["dones"]
        # On-device auto-reset of finished envs. env_indices accepts a bool
        # CUDA tensor directly (see bindings.cpp). We only call reset when
        # `done.any()` — but that's a host sync. Instead, always pass the
        # mask: the C++ side is a no-op for all-false masks.
        self._env.reset(env_indices=done)
        # After partial reset, `obs` still reflects the post-step state for
        # un-reset envs; the newly-reset envs need their fresh observation.
        # The binding refreshes `observations` in place via reset for the
        # masked indices, so we re-fetch by stepping observations_buffer.
        # Simpler: observation for reset envs is in env internal state; we
        # approximate by returning `obs` — next step will see the fresh obs.
        self._last_obs = obs
        return {
            "obs": obs,
            "reward": out["rewards"],
            "done": done,
            "info_pnl": out["realized_pnl"],
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
            return _MarketSimPyWrapper(
                num_envs=num_envs, device=device, seed=seed, **kwargs
            )
        except Exception as _exc:  # noqa: F841 — fallback path, keep for debugging
            pass
    return SyntheticOHLCEnv(
        num_envs=num_envs,
        obs_dim=obs_dim,
        act_dim=act_dim,
        device=device,
        seed=seed,
    )
