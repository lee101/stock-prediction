"""Generic checkpoint -> C marketsim PnL/Sortino evaluator.

The per-trainer adapters in ``fp4/bench/adapters/`` and the fp4 trainer itself
save checkpoints in a variety of formats.  pufferlib_market.evaluate_fast can
only load checkpoints that exactly match its ``TradingPolicy`` /
``ResidualTradingPolicy`` architectures, so running the bench harness on an
hf_trainer or fp4 checkpoint raises a state-dict mismatch.

This module provides :func:`evaluate_policy_file` which:

1. Loads the checkpoint produced by any adapter.
2. Introspects the Linear-weight shapes to infer ``(obs_dim, act_dim, hidden)``
   and rebuild a plain MLP with matching shapes.
3. If the policy is compatible with the C marketsim for ``data_path`` (i.e.
   obs_dim == num_symbols*16 + 5 + num_symbols and act_dim == 1 + 2*num_symbols
   since we only support alloc_bins=level_bins=1 here), runs the
   :mod:`pufferlib_market.binding` C vec-env for ``n_windows`` rollouts at each
   slippage level and returns a dict keyed by slippage bps.
4. Otherwise returns ``{"status": "skip", "reason": ...}`` per slippage.

The summary metrics emitted match the fields consumed by
``fp4/bench/compare_trainers.py::_flatten_eval`` (``p10_return``,
``median_return``, ``max_drawdown``, ``sortino``) so no downstream changes are
needed.

The C env itself uses binary fills (``fill_temperature`` is a training-only
knob in the soft sim; the C env returns hard PnL) which is the production
ground truth per ``CLAUDE.md``.
"""
from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Checkpoint introspection
# ---------------------------------------------------------------------------

def _find_state_dict(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        if hasattr(payload, "state_dict"):
            return payload.state_dict()
        raise ValueError(f"unexpected checkpoint payload type: {type(payload)}")
    for key in ("model", "policy", "policy_state_dict", "state_dict", "net"):
        inner = payload.get(key)
        if isinstance(inner, dict) and inner:
            # Heuristic: assume it's a state_dict if its values look tensor-ish
            return inner
    # Already a state_dict?
    if all(hasattr(v, "shape") for v in payload.values() if v is not None):
        return payload
    raise ValueError(f"could not find a state_dict in checkpoint keys: {list(payload.keys())}")


def _linear_shapes(sd: Dict[str, Any]) -> List[Tuple[str, Tuple[int, int]]]:
    """Return [(key, (out, in))] for every 2-D weight tensor in the state_dict."""
    out: List[Tuple[str, Tuple[int, int]]] = []
    for k, v in sd.items():
        if v is None:
            continue
        shape = tuple(getattr(v, "shape", ()))
        if len(shape) == 2:
            out.append((k, (int(shape[0]), int(shape[1]))))
    return out


def _infer_dims(sd: Dict[str, Any]) -> Tuple[int, int, int]:
    """Infer (obs_dim, act_dim, hidden) from Linear weight shapes.

    Assumption: first Linear reads obs_dim, hidden == its out features,
    last Linear of a policy_head / actor writes act_dim.  We pick:
      - obs_dim = first linear's in_features
      - hidden  = first linear's out_features
      - act_dim = out_features of a layer whose key contains ``actor``,
        ``policy_head`` or ``pi_head``; fall back to the last linear.
    """
    shapes = _linear_shapes(sd)
    if not shapes:
        raise ValueError("state_dict contains no 2-D Linear weights")
    first_key, (hidden, obs_dim) = shapes[0]
    act_dim: int | None = None
    for key, (out, _in) in shapes:
        lk = key.lower()
        if any(tag in lk for tag in ("policy_head", "pi_head", "actor")):
            act_dim = out
    if act_dim is None:
        act_dim = shapes[-1][1][0]
    return int(obs_dim), int(act_dim), int(hidden)


# ---------------------------------------------------------------------------
# Generic MLP wrapper
# ---------------------------------------------------------------------------

def _build_compatible_policy(sd: Dict[str, Any], obs_dim: int, act_dim: int, hidden: int):
    """Build a tiny MLP that matches the Linear shapes present in ``sd``.

    We only reproduce enough of the graph to get a ``(logits, _)`` forward
    pass so :func:`_run_slippage_sweep` can argmax over logits.  Any critic /
    value head weights are ignored.
    """
    import torch
    from torch import nn

    shapes = _linear_shapes(sd)
    # Trunk = every Linear that doesn't look like an actor/critic head; but to
    # keep this robust against adapters we *only* hard-wire a 2-layer tanh
    # trunk (obs_dim -> hidden -> hidden) + a policy head (hidden -> act_dim).
    class PolicyShim(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.l1 = nn.Linear(obs_dim, hidden)
            self.l2 = nn.Linear(hidden, hidden)
            self.policy_head = nn.Linear(hidden, act_dim)

        def forward(self, x):
            h = torch.tanh(self.l1(x))
            h = torch.tanh(self.l2(h))
            return self.policy_head(h)

    policy = PolicyShim()

    # Try to map known key names onto our shim.  This is best-effort: if
    # nothing matches the shim remains at random init and we flag that in the
    # returned reason rather than silently producing noise.
    shim_sd = policy.state_dict()
    mapping_candidates = {
        "l1.weight": ["body.0.weight", "encoder.0.weight", "in_proj.weight"],
        "l1.bias":   ["body.0.bias",   "encoder.0.bias",   "in_proj.bias"],
        "l2.weight": ["body.2.weight", "encoder.2.weight", "hidden.weight"],
        "l2.bias":   ["body.2.bias",   "encoder.2.bias",   "hidden.bias"],
        "policy_head.weight": ["policy_head.weight", "pi_head.weight", "actor.0.weight", "actor.weight"],
        "policy_head.bias":   ["policy_head.bias",   "pi_head.bias",   "actor.0.bias",   "actor.bias"],
    }
    n_loaded = 0
    for dst, srcs in mapping_candidates.items():
        for s in srcs:
            if s in sd and tuple(sd[s].shape) == tuple(shim_sd[dst].shape):
                shim_sd[dst] = sd[s].detach().clone()
                n_loaded += 1
                break
    policy.load_state_dict(shim_sd)
    policy.eval()
    return policy, n_loaded


# ---------------------------------------------------------------------------
# C env rollout
# ---------------------------------------------------------------------------

def _read_header(data_path: Path) -> Tuple[int, int, int]:
    with open(data_path, "rb") as f:
        head = f.read(64)
    _, _, num_symbols, num_timesteps, features_per_sym, _ = struct.unpack("<4sIIIII", head[:24])
    if features_per_sym == 0:
        features_per_sym = 16
    return int(num_symbols), int(num_timesteps), int(features_per_sym)


def _summarise_windows(returns: List[float], sortinos: List[float], maxdds: List[float]) -> Dict[str, Any]:
    if not returns:
        return {"error": "no windows completed"}
    arr = np.asarray(returns, dtype=np.float64)
    return {
        "p10_return": float(np.percentile(arr, 10)),
        "median_return": float(np.percentile(arr, 50)),
        "p90_return": float(np.percentile(arr, 90)),
        "mean_return": float(arr.mean()),
        "sortino": float(np.median(sortinos)) if sortinos else 0.0,
        "max_drawdown": float(np.median(maxdds)) if maxdds else 0.0,
        "n_neg": int(int(np.sum(arr < 0.0))),
        "n_windows": int(arr.size),
    }


def _run_slippage_sweep(
    policy_callable: Callable,
    data_path: Path,
    *,
    slippages_bps: Iterable[float],
    n_windows: int,
    eval_hours: int,
    fee_rate: float,
    max_leverage: float,
    seed: int,
) -> Dict[str, Any]:
    import torch
    from pufferlib_market import binding

    num_symbols, num_timesteps, features_per_sym = _read_header(data_path)
    obs_size = num_symbols * features_per_sym + 5 + num_symbols

    binding.shared(data_path=str(data_path.resolve()))

    rng = np.random.default_rng(seed)
    window_len = eval_hours + 1
    max_offset = num_timesteps - window_len
    if max_offset < 0:
        return {"status": "skip", "reason": f"val data too short: {num_timesteps} < {window_len}"}
    starts = rng.choice(max_offset + 1, size=n_windows, replace=(max_offset + 1 < n_windows))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        policy_callable.to(device)  # type: ignore[attr-defined]
    except Exception:
        pass

    out_by_bps: Dict[str, Any] = {}
    for bps in slippages_bps:
        obs_bufs = np.zeros((n_windows, obs_size), dtype=np.float32)
        act_bufs = np.zeros((n_windows,), dtype=np.int32)
        rew_bufs = np.zeros((n_windows,), dtype=np.float32)
        term_bufs = np.zeros((n_windows,), dtype=np.uint8)
        trunc_bufs = np.zeros((n_windows,), dtype=np.uint8)

        vec_handle = binding.vec_init(
            obs_bufs, act_bufs, rew_bufs, term_bufs, trunc_bufs,
            n_windows, int(seed),
            max_steps=eval_hours,
            fee_rate=float(fee_rate),
            max_leverage=float(max_leverage),
            short_borrow_apr=0.0,
            periods_per_year=8760.0,
            fill_slippage_bps=float(bps),
            forced_offset=-1,
            action_allocation_bins=1,
            action_level_bins=1,
            enable_drawdown_profit_early_exit=True,
            drawdown_profit_early_exit_min_steps=20,
            drawdown_profit_early_exit_progress_fraction=0.5,
        )
        try:
            binding.vec_set_offsets(vec_handle, starts.astype(np.int32))
            binding.vec_reset(vec_handle, int(seed))

            completed: List[Dict[str, float]] = [None] * n_windows  # type: ignore
            active = np.ones(n_windows, dtype=bool)
            obs_cpu = torch.from_numpy(obs_bufs)

            for _step in range(eval_hours + 10):
                if not active.any():
                    break
                with torch.inference_mode():
                    obs_t = obs_cpu.to(device, non_blocking=True)
                    logits = policy_callable(obs_t)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    actions = logits.argmax(dim=-1)
                act_bufs[:] = actions.detach().cpu().numpy().astype(np.int32)
                binding.vec_step(vec_handle)
                for i in range(n_windows):
                    if not active[i]:
                        continue
                    if term_bufs[i] or trunc_bufs[i]:
                        env_handle = binding.vec_env_at(vec_handle, i)
                        env_data = binding.env_get(env_handle)
                        completed[i] = {
                            "total_return": float(env_data.get("total_return", 0.0)),
                            "sortino": float(env_data.get("sortino", 0.0)),
                            "max_drawdown": float(env_data.get("max_drawdown", 0.0)),
                        }
                        active[i] = False
        finally:
            try:
                binding.vec_close(vec_handle)
            except Exception:
                pass

        valid = [c for c in completed if c is not None]
        rets = [c["total_return"] for c in valid]
        sortinos = [c["sortino"] for c in valid]
        maxdds = [c["max_drawdown"] for c in valid]
        out_by_bps[str(int(bps))] = _summarise_windows(rets, sortinos, maxdds)
    return {"status": "ok", "by_slippage": out_by_bps}


# ---------------------------------------------------------------------------
# Same-backend fallback: evaluate a policy in the env it was trained in.
# ---------------------------------------------------------------------------


def _same_backend_eval(
    sd: Dict[str, Any],
    obs_dim: int,
    act_dim: int,
    hidden: int,
    cfg: Dict[str, Any],
    n_windows: int = 20,
    seed: int = 1337,
    *,
    video_out_dir: Optional[Path] = None,
    video_title: str = "fp4 eval",
) -> Dict[str, Any]:
    """Run a deterministic policy eval through ``fp4.env_adapter.make_env``.

    Returns a dict with the same shape as ``_run_slippage_sweep`` so the
    bench harness can flatten it without special-casing. Slippage levels are
    not honoured (the gpu_trading_env / synthetic backends ignore the
    marketsim slippage knob); we report a single ``by_slippage["0"]`` cell
    so the leaderboard plumbing keeps working.
    """
    try:
        import torch
        from fp4.env_adapter import make_env as _adapter_make_env
        from fp4.bench.eval_generic import _build_compatible_policy as _build_pi
    except Exception as exc:
        return {"status": "skip", "reason": f"same-backend eval import failed: {exc}"}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        handle = _adapter_make_env(
            cfg if isinstance(cfg, dict) else {},
            num_envs=128, obs_dim=obs_dim, act_dim=act_dim,
            device=device, seed=seed,
        )
    except Exception as exc:
        return {"status": "skip", "reason": f"same-backend make_env failed: {exc}"}

    real_obs_dim = int(handle.obs_dim)
    real_act_dim = int(handle.action_dim)
    if real_obs_dim != obs_dim:
        return {
            "status": "skip",
            "reason": f"same-backend env obs_dim={real_obs_dim} != policy obs_dim={obs_dim}",
        }

    policy, n_loaded = _build_pi(sd, obs_dim, act_dim, hidden)
    if n_loaded == 0:
        return {"status": "skip", "reason": "no recognisable policy weights for shim"}
    policy.to(device).eval()

    eval_steps = max(64, int(n_windows) * 32)

    # Opt-in: attach a ReplayRecorder so we can render an MP4 + HTML scrubber
    # of this exact eval rollout. Slow (~secs), so it's only wired when the
    # caller asks for it. The recorder captures env 0 of the batch.
    recorder = None
    if video_out_dir is not None:
        try:
            from fp4.fp4.replay_recorder import ReplayRecorder
            recorder = ReplayRecorder.attach(handle, max_steps=int(eval_steps), env_index=0)
        except Exception as _exc:
            import sys as _sys
            print(f"[eval_generic] ReplayRecorder attach failed: {_exc}", file=_sys.stderr)
            recorder = None

    obs = handle.reset().to(torch.float32).to(device)
    if recorder is not None:
        try:
            recorder.on_reset(obs)
        except Exception:
            recorder = None
    rewards_per_env = torch.zeros(int(handle.num_envs), device=device)
    peak_per_env = torch.zeros_like(rewards_per_env)
    drawdown_per_env = torch.zeros_like(rewards_per_env)
    sum_neg_sq = torch.zeros_like(rewards_per_env)
    sum_ret = torch.zeros_like(rewards_per_env)
    n_count = torch.zeros_like(rewards_per_env)

    with torch.no_grad():
        for _ in range(eval_steps):
            logits = policy(obs)
            # The shim outputs (..., act_dim). Trainer/SAC use continuous
            # actions; pick mean (== logits) and clamp to [-1, 1] so we don't
            # depend on knowing whether this came from PPO or QR-PPO.
            action = torch.tanh(logits)
            obs, reward, _done, _cost = handle.step(action)
            obs = obs.to(torch.float32)
            reward = reward.to(torch.float32)
            if recorder is not None:
                try:
                    recorder.on_step(action=action, obs=obs, reward=reward, done=_done)
                except Exception as _exc:
                    import sys as _sys
                    print(f"[eval_generic] recorder.on_step failed: {_exc}", file=_sys.stderr)
                    recorder = None
            rewards_per_env = rewards_per_env + reward
            peak_per_env = torch.maximum(peak_per_env, rewards_per_env)
            cur_dd = peak_per_env - rewards_per_env
            drawdown_per_env = torch.maximum(drawdown_per_env, cur_dd)
            sum_ret = sum_ret + reward
            sum_neg_sq = sum_neg_sq + torch.where(reward < 0, reward * reward, torch.zeros_like(reward))
            n_count = n_count + 1.0

    returns = rewards_per_env.detach().cpu().numpy().tolist()
    drawdowns = drawdown_per_env.detach().cpu().numpy().tolist()
    # Per-env Sortino (annualisation skipped — relative ranking is what we
    # care about for the leaderboard).
    eps = 1e-8
    sortinos: List[float] = []
    for s_ret, s_neg, n in zip(sum_ret.tolist(), sum_neg_sq.tolist(), n_count.tolist()):
        if n <= 1 or s_neg <= 0:
            sortinos.append(0.0)
            continue
        mean_r = s_ret / n
        downside = (s_neg / n) ** 0.5
        sortinos.append(mean_r / (downside + eps))

    summary = _summarise_windows(returns, sortinos, drawdowns)
    out: Dict[str, Any] = {
        "status": "ok",
        "backend": "same_as_train",
        "eval_env_backend": handle.backend_name,
        "n_envs": int(handle.num_envs),
        "n_steps": int(eval_steps),
        "n_loaded_weights": int(n_loaded),
        "by_slippage": {"0": {"summary": summary,
                              "p10_return": summary.get("p10_total_return", 0.0),
                              "median_return": summary.get("median_total_return", 0.0),
                              "max_drawdown": summary.get("median_max_drawdown", 0.0),
                              "sortino": summary.get("median_sortino", 0.0)}},
    }

    # ---- Render the captured trajectory into MP4 + HTML if requested ----
    if recorder is not None and video_out_dir is not None:
        try:
            from fp4.fp4.replay_recorder import trajectory_to_marketsim_trace
            from fp4.bench.render_replay import render_videos
            traj = recorder.trajectory()
            video_out_dir.mkdir(parents=True, exist_ok=True)
            produced = render_videos(
                traj, Path(video_out_dir),
                title=video_title, fps=8, num_pairs=1,
            )
            out["videos"] = produced
        except Exception as exc:
            import sys as _sys
            print(f"[eval_generic] video render failed: {type(exc).__name__}: {exc}",
                  file=_sys.stderr)
            out["videos"] = {"error": f"{type(exc).__name__}: {exc}"}

    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def evaluate_policy_file(
    ckpt_path: Path,
    cfg: Dict[str, Any],
    repo_root: Path,
) -> Dict[str, Any]:
    """Load a checkpoint and run the slippage sweep against the C marketsim."""
    try:
        import torch
    except Exception as exc:
        return {"status": "skip", "reason": f"torch missing: {exc}"}

    env_cfg = cfg.get("env", {})
    eval_cfg = cfg.get("eval", {})
    data_path = repo_root / env_cfg.get("val_data", "")
    if not data_path.exists():
        return {"status": "skip", "reason": f"val data missing: {data_path}"}

    try:
        payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except Exception as exc:
        return {"status": "skip", "reason": f"torch.load failed: {type(exc).__name__}: {exc}"}

    try:
        sd = _find_state_dict(payload)
    except Exception as exc:
        return {"status": "skip", "reason": f"{exc}"}

    try:
        obs_dim, act_dim, hidden = _infer_dims(sd)
    except Exception as exc:
        return {"status": "skip", "reason": f"could not infer policy dims: {exc}"}

    num_symbols, _, features_per_sym = _read_header(data_path)
    expected_obs = num_symbols * features_per_sym + 5 + num_symbols
    expected_act = 1 + 2 * num_symbols
    if obs_dim != expected_obs or act_dim != expected_act:
        # Same-backend fallback: when the trained policy doesn't fit the
        # pufferlib_market binary header, run a deterministic eval through
        # ``fp4.env_adapter.make_env`` (the very env the trainer used) and
        # report p10/sortino/max_dd from that. Without this fallback every
        # fp4 sweep cell that trains on ``gpu_trading_env`` (7-dim) gets
        # status="skip" and the leaderboard cannot rank it. The eval is
        # NOT directly comparable to the production marketsim numbers, but
        # it lets the sweep produce monotone-improving results we can
        # iterate on.
        video_out_dir: Optional[Path] = None
        video_title = ckpt_path.stem
        if bool(eval_cfg.get("video", False)):
            # Emit videos under models/artifacts/<ckpt_stem>/videos/ so the
            # existing src/artifacts_server.py serves them at /files/<stem>/videos/.
            artifacts_root = repo_root / "models" / "artifacts"
            video_out_dir = artifacts_root / ckpt_path.stem / "videos"
            video_title = f"fp4 {ckpt_path.stem}"
        same_backend = _same_backend_eval(
            sd, obs_dim, act_dim, hidden, cfg, n_windows=int(eval_cfg.get("n_windows", 20)),
            seed=int(eval_cfg.get("seed", 1337)),
            video_out_dir=video_out_dir,
            video_title=video_title,
        )
        same_backend["shape_mismatch"] = {
            "policy_obs_dim": obs_dim, "marketsim_obs_dim": expected_obs,
            "policy_act_dim": act_dim, "marketsim_act_dim": expected_act,
        }
        return same_backend

    policy, n_loaded = _build_compatible_policy(sd, obs_dim, act_dim, hidden)
    if n_loaded == 0:
        return {"status": "skip",
                "reason": "no recognisable weight keys were loaded into the eval shim"}

    return _run_slippage_sweep(
        policy,
        data_path,
        slippages_bps=tuple(eval_cfg.get("slippage_bps", (0, 5, 10, 20))),
        n_windows=int(eval_cfg.get("n_windows", 20)),
        eval_hours=int(eval_cfg.get("eval_hours", 720)),
        fee_rate=float(env_cfg.get("fee_rate", 0.001)),
        max_leverage=float(env_cfg.get("max_leverage_scalar_fallback", 1.5)),
        seed=int(eval_cfg.get("seed", 1337)),
    )
