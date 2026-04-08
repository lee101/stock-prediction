"""Per-step replay recorder for fp4 ``EnvHandle`` (P6-1).

Captures a single env lane (env_index=0 by default) of a vectorised
``EnvHandle`` into host-side ring buffers, suitable for feeding the existing
``pufferlib_market.intrabar_replay`` / ``src.marketsim_video`` rendering path
to produce an MP4 + HTML scrub.

Design goals (per the Phase 6 plan):

* **Zero impact on the hot loop SPS** — we never ``.cpu().tolist()`` inside
  the step. All copies use a dedicated CUDA stream (when available) and
  ``non_blocking=True`` into pre-allocated host pinned buffers. The only
  synchronisation point is ``trajectory()`` at the very end, which the caller
  invokes after the rollout finishes.
* **Tiny footprint** — only env 0 is recorded, and only the seven scalars the
  intrabar renderer actually needs: ref_px (ohlc[t_idx, 3]), equity, pos_qty,
  drawdown, reward, done, plus the raw action ``[p_bid, p_ask, q_bid, q_ask]``.
* **Works with any gpu_trading_env-backed ``EnvHandle``** — we reach into the
  underlying ``gte.EnvHandle`` via the closure stashed on the adapter's step
  function, or accept a raw ``gte.EnvHandle`` directly. Other backends fall
  back to an obs-only slice (no ref_px), which is enough for the test path.

The recorder is a simple context manager:

    rec = ReplayRecorder.attach(env_handle, max_steps=4096)
    obs = env_handle.reset()
    rec.on_reset(obs)
    for _ in range(max_steps):
        action = policy(obs)
        obs, reward, done, cost = env_handle.step(action)
        rec.on_step(action=action, obs=obs, reward=reward, done=done)
    traj = rec.trajectory()   # single sync point, returns numpy arrays
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Trajectory container (host-side numpy arrays, post-sync)
# ---------------------------------------------------------------------------


@dataclass
class ReplayTrajectory:
    """Host-side recorded trajectory for a single env lane.

    All arrays are length ``num_steps`` (<= ``max_steps``).
    """
    num_steps: int
    ref_px: np.ndarray          # [T] close of the current bar
    ref_ohlc: np.ndarray        # [T, 4] full OHLC of the current bar
    equity: np.ndarray          # [T]
    pos_qty: np.ndarray         # [T]
    drawdown: np.ndarray        # [T]
    reward: np.ndarray          # [T]
    done: np.ndarray            # [T] bool
    action: np.ndarray          # [T, 4]
    t_idx: np.ndarray           # [T] int32, bar index into the env's ohlc
    initial_equity: float = 10000.0
    backend_name: str = "unknown"

    def __len__(self) -> int:
        return int(self.num_steps)


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------


def _unwrap_gte(env_handle: Any) -> Optional[Any]:
    """Best-effort extraction of the underlying gpu_trading_env ``EnvHandle``.

    The fp4 env_adapter wraps the raw gte handle inside closures on
    ``_reset_fn`` / ``_step_fn``. We look for a ``__closure__`` cell named
    ``inner`` that exposes the attributes we need. Returns None if the
    backend isn't gpu_trading_env.
    """
    # Direct gte.EnvHandle
    if hasattr(env_handle, "ohlc") and hasattr(env_handle, "state"):
        return env_handle
    step_fn = getattr(env_handle, "_step_fn", None)
    if step_fn is None:
        return None
    closure = getattr(step_fn, "__closure__", None)
    freevars = getattr(getattr(step_fn, "__code__", None), "co_freevars", ())
    if not closure or not freevars:
        return None
    for name, cell in zip(freevars, closure):
        if name == "inner":
            val = cell.cell_contents
            if hasattr(val, "ohlc") and hasattr(val, "state"):
                return val
    return None


@dataclass
class ReplayRecorder:
    max_steps: int
    env_index: int = 0
    backend_name: str = "unknown"

    # Internal host pinned buffers.
    _ref_ohlc: torch.Tensor = field(init=False)
    _equity: torch.Tensor = field(init=False)
    _pos_qty: torch.Tensor = field(init=False)
    _drawdown: torch.Tensor = field(init=False)
    _reward: torch.Tensor = field(init=False)
    _done: torch.Tensor = field(init=False)
    _action: torch.Tensor = field(init=False)
    _t_idx: torch.Tensor = field(init=False)

    # Reference to the underlying gpu_trading_env handle when available.
    _gte: Optional[Any] = None
    _stream: Optional[Any] = None
    _cursor: int = 0
    _initial_equity: float = 10000.0

    @classmethod
    def attach(cls, env_handle: Any, max_steps: int, env_index: int = 0) -> "ReplayRecorder":
        backend = getattr(env_handle, "backend_name", "unknown")
        rec = cls(max_steps=int(max_steps), env_index=int(env_index), backend_name=str(backend))
        rec._init_buffers()
        rec._gte = _unwrap_gte(env_handle)
        # Dedicated copy stream (best-effort, only when CUDA is available).
        if torch.cuda.is_available():
            try:
                rec._stream = torch.cuda.Stream()
            except Exception:
                rec._stream = None
        return rec

    def _init_buffers(self) -> None:
        # pin_memory requires an initialised CUDA context. If CUDA is under
        # memory pressure from a sibling process, initialising the context
        # just to allocate pinned host memory would OOM. Try to pin; on any
        # failure, silently fall back to pageable host memory.
        pin = False
        if torch.cuda.is_available():
            try:
                torch.empty(1, pin_memory=True)
                pin = True
            except Exception:
                pin = False
        T = int(self.max_steps)
        kw = {"dtype": torch.float32, "pin_memory": pin}
        self._ref_ohlc = torch.zeros((T, 4), **kw)
        self._equity = torch.zeros(T, **kw)
        self._pos_qty = torch.zeros(T, **kw)
        self._drawdown = torch.zeros(T, **kw)
        self._reward = torch.zeros(T, **kw)
        self._action = torch.zeros((T, 4), **kw)
        self._done = torch.zeros(T, dtype=torch.int32, pin_memory=pin)
        self._t_idx = torch.zeros(T, dtype=torch.int32, pin_memory=pin)

    # ----- capture hooks ---------------------------------------------------

    def on_reset(self, obs: Any) -> None:
        """Record the initial equity snapshot for correct trace normalisation."""
        self._cursor = 0
        try:
            if torch.is_tensor(obs) and obs.dim() == 2 and obs.shape[-1] >= 7:
                self._initial_equity = float(obs[self.env_index, 4].item())
            elif isinstance(obs, dict) and "equity" in obs:
                self._initial_equity = float(obs["equity"][self.env_index].item())
        except Exception:
            pass

    def on_step(
        self,
        *,
        action: torch.Tensor,
        obs: Any,
        reward: Any,
        done: Any,
    ) -> None:
        if self._cursor >= self.max_steps:
            return
        i = self._cursor
        stream_ctx = (
            torch.cuda.stream(self._stream)
            if self._stream is not None and torch.cuda.is_available()
            else _NullCtx()
        )
        with stream_ctx:
            # Action is always [N, 4] after env_adapter shimming, but we are
            # defensive for callers that pass pre-shim actions.
            a = action.detach() if torch.is_tensor(action) else torch.as_tensor(action)
            if a.dim() == 1:
                a = a.unsqueeze(0)
            if a.shape[-1] != 4:
                # Pad/truncate so we always record a uniform [4] slot.
                pad = torch.zeros(a.shape[0], 4, device=a.device, dtype=a.dtype)
                pad[:, : min(a.shape[-1], 4)] = a[:, : min(a.shape[-1], 4)]
                a = pad
            self._action[i].copy_(a[self.env_index].float(), non_blocking=True)

            # equity / pos_qty / drawdown from obs (env_adapter flattens these
            # into a [N,7] obs for gpu_trading_env: open,high,low,close,eq,pq,dd).
            if torch.is_tensor(obs) and obs.dim() == 2 and obs.shape[-1] >= 7:
                o = obs[self.env_index].float()
                # bar=0..3 are OHLC of the *current* (pre-step) bar. The ref
                # price the renderer needs is the bar close.
                self._ref_ohlc[i].copy_(o[0:4], non_blocking=True)
                self._equity[i].copy_(o[4:5].squeeze(0), non_blocking=True)
                self._pos_qty[i].copy_(o[5:6].squeeze(0), non_blocking=True)
                self._drawdown[i].copy_(o[6:7].squeeze(0), non_blocking=True)
            elif isinstance(obs, dict):
                # Raw gte obs dict form.
                if "bar" in obs:
                    self._ref_ohlc[i].copy_(obs["bar"][self.env_index].float(), non_blocking=True)
                for key, buf in (
                    ("equity", self._equity),
                    ("pos_qty", self._pos_qty),
                    ("drawdown", self._drawdown),
                ):
                    if key in obs:
                        buf[i].copy_(obs[key][self.env_index].float(), non_blocking=True)

            if torch.is_tensor(reward):
                self._reward[i].copy_(reward[self.env_index].float(), non_blocking=True)
            if torch.is_tensor(done):
                d = done[self.env_index].to(torch.int32)
                self._done[i].copy_(d, non_blocking=True)

            # t_idx is only on the raw gte handle.
            if self._gte is not None:
                ti = self._gte.state["t_idx"][self.env_index].to(torch.int32)
                self._t_idx[i].copy_(ti, non_blocking=True)

        self._cursor += 1

    # ----- extraction ------------------------------------------------------

    def trajectory(self) -> ReplayTrajectory:
        """Synchronise copies and return a host-side trajectory snapshot."""
        if self._stream is not None and torch.cuda.is_available():
            torch.cuda.current_stream().wait_stream(self._stream)
            torch.cuda.synchronize()
        T = int(self._cursor)
        ref_ohlc = self._ref_ohlc[:T].cpu().numpy().astype(np.float32, copy=False)
        ref_px = ref_ohlc[:, 3].copy()  # close
        return ReplayTrajectory(
            num_steps=T,
            ref_px=ref_px,
            ref_ohlc=ref_ohlc,
            equity=self._equity[:T].cpu().numpy().astype(np.float32, copy=False),
            pos_qty=self._pos_qty[:T].cpu().numpy().astype(np.float32, copy=False),
            drawdown=self._drawdown[:T].cpu().numpy().astype(np.float32, copy=False),
            reward=self._reward[:T].cpu().numpy().astype(np.float32, copy=False),
            done=self._done[:T].cpu().numpy().astype(bool, copy=False),
            action=self._action[:T].cpu().numpy().astype(np.float32, copy=False),
            t_idx=self._t_idx[:T].cpu().numpy().astype(np.int32, copy=False),
            initial_equity=float(self._initial_equity),
            backend_name=str(self.backend_name),
        )

class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# ---------------------------------------------------------------------------
# Adapter: ReplayTrajectory -> MarketsimTrace (single synthetic symbol).
# ---------------------------------------------------------------------------


def trajectory_to_marketsim_trace(traj: ReplayTrajectory, symbol_name: str = "ENV0"):
    """Convert a recorded trajectory into a ``src.marketsim_video.MarketsimTrace``.

    We use a single synthetic symbol whose OHLC is the per-step bar the env
    was on. Fills are derived from action → pos_qty transitions. Equity is
    forwarded directly and rescaled by the recorded ``initial_equity`` so the
    renderer's default $10k display stays meaningful.
    """
    from src.marketsim_video import MarketsimTrace, OrderTick  # local import, optional dep

    T = int(traj.num_steps)
    closes = np.asarray(traj.ref_px, dtype=np.float32).reshape(T, 1)
    ohlc = np.asarray(traj.ref_ohlc, dtype=np.float32).reshape(T, 1, 4)
    trace = MarketsimTrace(symbols=[symbol_name], prices=closes, prices_ohlc=ohlc)

    prev_qty = 0.0
    for i in range(T):
        qty = float(traj.pos_qty[i])
        is_short = qty < 0.0
        pos_sym = 0 if abs(qty) > 1e-9 else -1
        orders: list = []
        if abs(qty - prev_qty) > 1e-9:
            # Some kind of fill at this bar.
            px = float(traj.ref_px[i])
            orders.append(OrderTick(sym=0, price=px, is_short=(qty < prev_qty)))
        prev_qty = qty
        eq = float(traj.equity[i])
        # Rescale so the renderer's $10k baseline matches.
        init_eq = max(float(traj.initial_equity), 1e-9)
        eq_norm = eq * 10000.0 / init_eq
        trace.record(
            step=i,
            action_id=int(np.argmax(np.abs(traj.action[i]))) if traj.action.shape[1] > 0 else 0,
            position_sym=pos_sym,
            position_is_short=is_short,
            equity=eq_norm,
            orders=orders,
        )
    return trace


__all__ = [
    "ReplayRecorder",
    "ReplayTrajectory",
    "trajectory_to_marketsim_trace",
]
