"""Canonical env factory for fp4 trainers (PPO/SAC/QR-PPO).

Selects the highest-priority backend that can run on this box and returns a
uniform ``EnvHandle`` with the same surface every fp4 trainer expects:

    handle.reset() -> obs                # [N, obs_dim] float32 tensor
    handle.step(action) -> (obs, reward, done, cost_or_None)
    handle.obs_dim, handle.action_dim, handle.num_envs, handle.backend_name

Backends, in priority order:
  1. ``gpu_trading_env`` (P4-1 CUDA kernel) — preferred when CUDA + ext build
  2. ``market_sim_py`` SCALAR/DPS via ``fp4.vec_env.GPUVecEnv`` (Phase 2)
  3. ``StubVecEnv`` from ``fp4.trainer`` — synthetic, CPU-OK fallback

The fp4 trainers' policies come in two shapes:
  * ``ActorCritic`` (Unit A): emits a scalar/n-d "directional" action. The
    adapter for ``gpu_trading_env`` wraps it as a single-side market order at
    ``close ± buffer`` so the [B,4]=(p_bid,p_ask,q_bid,q_ask) contract is met
    without changing the trainer.
  * ``TwoLayerPolicy`` (P4-2): exposes ``to_quote_prices(layer_b, ref_px)``;
    the trainer is responsible for that conversion. When the trainer hands the
    adapter a 4-wide action we pass it through unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

_LAST_GTE_ERR: Optional[str] = None


# ---------------------------------------------------------------------------
# Uniform handle
# ---------------------------------------------------------------------------


@dataclass
class EnvHandle:
    backend_name: str
    num_envs: int
    obs_dim: int
    action_dim: int
    _reset_fn: Any
    _step_fn: Any

    # Aliases the existing trainers read.
    @property
    def act_dim(self) -> int:
        return self.action_dim

    def reset(self) -> torch.Tensor:
        return self._reset_fn()

    def step(self, action: torch.Tensor):
        return self._step_fn(action)


# ---------------------------------------------------------------------------
# Backend 1: gpu_trading_env (CUDA kernel)
# ---------------------------------------------------------------------------


def _try_gpu_trading_env(
    cfg: Dict[str, Any], num_envs: int, seed: int
) -> Optional[EnvHandle]:
    if not torch.cuda.is_available():
        return None
    try:
        import gpu_trading_env as gte
    except Exception:
        return None
    if gte._load_ext() is None:
        return None
    env_spec = cfg.get("env")
    params = {}
    ohlc_path = None
    bin_path = None
    if isinstance(env_spec, dict):
        params = {k: v for k, v in env_spec.items()
                  if k not in ("ohlc_path", "train_data", "val_data", "backend", "name")}
        ohlc_path = env_spec.get("ohlc_path")
        # Detect multi-symbol .bin data (stocks12_v5_rsi etc.).
        train_data = env_spec.get("train_data")
        if train_data and str(train_data).endswith(".bin"):
            from pathlib import Path
            # Resolve relative to repo root.
            p = Path(train_data)
            if not p.is_absolute():
                # fp4/fp4/env_adapter.py -> repo root is two parents up from fp4/fp4/
                repo = Path(__file__).resolve().parents[2]
                p = repo / train_data
            if p.exists():
                bin_path = str(p)
    global _LAST_GTE_ERR

    # --- Multi-symbol path: load .bin, present full obs/act shapes ---
    if bin_path is not None:
        try:
            inner = gte.make_multi_symbol(
                B=int(num_envs),
                bin_path=bin_path,
                params=params or None,
            )
        except Exception as exc:
            _LAST_GTE_ERR = f"{type(exc).__name__}: {exc}"
            return None

        device = inner.features.device
        obs_dim = inner.obs_dim
        action_dim = inner.action_dim

        def _reset_ms() -> torch.Tensor:
            inner.reset()
            return inner._obs().to(torch.float32)

        def _step_ms(action: torch.Tensor):
            obs, reward, done, cost = inner.step(action.to(device))
            return obs.to(torch.float32), reward, done.to(torch.float32), cost

        return EnvHandle(
            backend_name="gpu_trading_env_multi",
            num_envs=int(num_envs),
            obs_dim=obs_dim,
            action_dim=action_dim,
            _reset_fn=_reset_ms,
            _step_fn=_step_ms,
        )

    # --- Legacy single-instrument path ---
    try:
        inner = gte.make(
            B=int(num_envs),
            ohlc_path_or_tensor=ohlc_path,
            params=params or None,
            seed=int(seed),
        )
    except Exception as exc:
        _LAST_GTE_ERR = f"{type(exc).__name__}: {exc}"
        return None

    device = inner.ohlc.device
    # Observation: flatten the obs dict produced by EnvHandle._obs() into a
    # [N, 7] vector (open, high, low, close, equity, pos_qty, drawdown). The
    # cfg's nominal obs_dim is overridden by this real shape so the trainer
    # builds correctly-sized buffers.
    obs_dim = 7
    action_dim = 4

    def _flatten(obs_dict) -> torch.Tensor:
        bar = obs_dict["bar"]  # [N,4]
        eq = obs_dict["equity"].unsqueeze(-1)
        pq = obs_dict["pos_qty"].unsqueeze(-1)
        dd = obs_dict["drawdown"].unsqueeze(-1)
        return torch.cat([bar, eq, pq, dd], dim=-1).to(torch.float32)

    def _shim_action(action: torch.Tensor) -> torch.Tensor:
        # Trainers may hand us a [N,A] action where A != 4. Map it to a
        # single-side market order at the current close ± buffer. action[:,0]
        # > 0 -> buy, < 0 -> sell. |action[:,0]| sets size_frac in [0,1].
        if action.dim() == 1:
            action = action.unsqueeze(-1)
        if action.shape[-1] == 4:
            return action.to(torch.float32)
        ti = inner.state["t_idx"].clamp_max(inner.ohlc.size(0) - 1).long()
        close = inner.ohlc.index_select(0, ti)[:, 3]  # [N]
        sig = torch.tanh(action[:, 0].to(torch.float32))
        size_frac = sig.abs().clamp(0.0, 1.0)
        # Marketable quote: aggressive bid above close (buy) or aggressive ask
        # below close (sell). The kernel handles the cross. We always quote
        # both sides but starve the inactive side via q=0.
        buy = (sig > 0).to(torch.float32)
        sell = 1.0 - buy
        p_bid = close * (1.0 + 1e-3)
        p_ask = close * (1.0 - 1e-3)
        q_bid = size_frac * buy
        q_ask = size_frac * sell
        out = torch.stack([p_bid, p_ask, q_bid, q_ask], dim=-1)
        return out.contiguous()

    def _reset() -> torch.Tensor:
        inner.reset()
        return _flatten(inner._obs())

    def _step(action: torch.Tensor):
        a = _shim_action(action.to(device))
        obs_dict, reward, done, cost = inner.step(a)
        return _flatten(obs_dict), reward, done.to(torch.float32), cost

    return EnvHandle(
        backend_name="gpu_trading_env",
        num_envs=int(num_envs),
        obs_dim=obs_dim,
        action_dim=action_dim,
        _reset_fn=_reset,
        _step_fn=_step,
    )


# ---------------------------------------------------------------------------
# Backend 2: market_sim_py via fp4.vec_env.GPUVecEnv
# ---------------------------------------------------------------------------


def _try_market_sim_py(
    cfg: Dict[str, Any], num_envs: int, obs_dim: int, act_dim: int,
    device: torch.device, seed: int,
) -> Optional[EnvHandle]:
    try:
        from .vec_env import GPUVecEnv
    except Exception:
        return None
    env_spec = cfg.get("env")
    env_kwargs: Dict[str, Any] = {}
    if isinstance(env_spec, dict):
        if "action_mode" in env_spec:
            env_kwargs["action_mode"] = str(env_spec["action_mode"])
        if "symbols" in env_spec:
            env_kwargs["symbols"] = list(env_spec["symbols"])
    try:
        inner = GPUVecEnv(
            num_envs=num_envs, obs_dim=obs_dim, act_dim=act_dim,
            device=device, seed=seed, **env_kwargs,
        )
    except Exception:
        return None

    sample = inner.reset()
    cached = [sample if not isinstance(sample, dict)
              else (sample.get("obs") or sample.get("observations"))]

    real_obs_dim = int(cached[0].shape[-1])
    real_act_dim = int(getattr(inner, "act_dim", act_dim))
    real_n = int(getattr(inner, "num_envs", num_envs))

    def _reset() -> torch.Tensor:
        if cached[0] is not None:
            obs = cached[0]
            cached[0] = None
            return obs
        out = inner.reset()
        return out["obs"] if isinstance(out, dict) else out

    def _step(action: torch.Tensor):
        out = inner.step(action)
        if isinstance(out, dict):
            return (out["obs"], out["reward"],
                    out["done"].to(torch.float32), out.get("cost"))
        if len(out) == 3:
            obs, rew, done = out
            return obs, rew, done.to(torch.float32), None
        return out  # already 4-tuple

    return EnvHandle(
        backend_name="market_sim_py",
        num_envs=real_n,
        obs_dim=real_obs_dim,
        action_dim=real_act_dim,
        _reset_fn=_reset,
        _step_fn=_step,
    )


# ---------------------------------------------------------------------------
# Backend 3: stub
# ---------------------------------------------------------------------------


def _make_stub(
    cfg: Dict[str, Any], num_envs: int, obs_dim: int, act_dim: int,
    device: torch.device, seed: int,
) -> EnvHandle:
    from .trainer import StubVecEnv
    episode_len = int(cfg.get("episode_len", 256)) if isinstance(cfg, dict) else 256
    inner = StubVecEnv(num_envs, obs_dim, act_dim, episode_len, device, seed)

    def _reset() -> torch.Tensor:
        return inner.reset()

    def _step(action: torch.Tensor):
        obs, rew, done = inner.step(action)
        return obs, rew, done, None

    return EnvHandle(
        backend_name="stub",
        num_envs=int(num_envs),
        obs_dim=int(obs_dim),
        action_dim=int(act_dim),
        _reset_fn=_reset,
        _step_fn=_step,
    )


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def make_env(
    cfg: Dict[str, Any],
    num_envs: int = 8,
    obs_dim: int = 16,
    act_dim: int = 3,
    device: Optional[torch.device] = None,
    seed: int = 0,
) -> EnvHandle:
    """Build an env according to ``cfg['env']``.

    cfg['env'] may be:
      * 'auto'  (default)         — pick highest-priority available backend
      * 'gpu_trading_env'         — force the C kernel (raises if unavailable)
      * 'market_sim_py'           — force the marketsim binding
      * 'stub'                    — force the synthetic backend
      * dict {...}                — auto, but pass through env params
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = cfg or {}
    env_spec = cfg.get("env", "auto")
    if isinstance(env_spec, str):
        choice = env_spec
    elif isinstance(env_spec, dict):
        choice = str(env_spec.get("backend", "auto"))
    else:
        choice = "auto"

    if choice in ("gpu_trading_env", "auto"):
        h = _try_gpu_trading_env(cfg, num_envs, seed)
        if h is not None:
            return h
        if choice == "gpu_trading_env":
            raise RuntimeError(
                f"gpu_trading_env backend unavailable: {_LAST_GTE_ERR or 'no detail'}"
            )

    if choice in ("market_sim_py", "auto"):
        h = _try_market_sim_py(cfg, num_envs, obs_dim, act_dim, device, seed)
        if h is not None:
            return h
        if choice == "market_sim_py":
            raise RuntimeError("market_sim_py backend unavailable on this box")

    return _make_stub(cfg, num_envs, obs_dim, act_dim, device, seed)


__all__ = ["EnvHandle", "make_env"]
