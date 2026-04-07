"""Thin Python wrapper around the SoA fused env_step CUDA kernel.

Public API::

    env = gpu_trading_env.make(B=1024, ohlc_path_or_tensor=None, params={...})
    obs, reward, done, cost = env.step(action)   # action: [B, 4] cuda float
    env.reset(mask=None)                         # mask: [B] bool, None = all

The environment is GPU-resident: all state tensors live on CUDA, and the
hot-loop performs zero host syncs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import torch


_EXT = None
_EXT_ERR: Optional[str] = None
_ATTEMPTED = False

_PKG_DIR = Path(__file__).resolve().parent
# csrc lives at gpu_trading_env/csrc (three levels up from this file).
_CSRC = (_PKG_DIR / ".." / ".." / "csrc").resolve()
if not _CSRC.exists():
    # Fallback: when installed as a wheel, csrc may be package-adjacent.
    _CSRC = (_PKG_DIR / "csrc").resolve()


def _ensure_cuda_home():
    import os
    cur = os.environ.get("CUDA_HOME", "")
    if cur and os.path.exists(os.path.join(cur, "bin", "nvcc")):
        return
    for cand in ("/usr/local/cuda", "/usr/local/cuda-13", "/usr/local/cuda-12.9",
                 "/usr/local/cuda-12.8", "/usr/local/cuda-12"):
        if os.path.exists(os.path.join(cand, "bin", "nvcc")):
            os.environ["CUDA_HOME"] = cand
            os.environ["CUDA_PATH"] = cand
            os.environ["PATH"] = os.path.join(cand, "bin") + os.pathsep + os.environ.get("PATH", "")
            return


def _load_ext():
    """JIT-compile the CUDA extension on first use. Cached after.

    Uses torch.utils.cpp_extension.load; non-fatal on non-CUDA boxes (returns
    None and records a skip reason).
    """
    global _EXT, _EXT_ERR, _ATTEMPTED
    if _ATTEMPTED:
        return _EXT
    _ATTEMPTED = True
    if not torch.cuda.is_available():
        _EXT_ERR = "no CUDA available"
        return None
    try:
        cap = torch.cuda.get_device_capability(0)
    except Exception as e:
        _EXT_ERR = f"cannot query GPU capability: {e}"
        return None
    try:
        import os
        _ensure_cuda_home()
        arch = f"{cap[0]}.{cap[1]}+PTX"
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", arch)
        # Force compiler temp files onto the package build dir (/tmp is
        # sandboxed on this box).
        build_tmp = (_PKG_DIR / ".." / ".." / "tmp").resolve()
        build_tmp.mkdir(parents=True, exist_ok=True)
        os.environ["TMPDIR"] = str(build_tmp)
        from torch.utils.cpp_extension import load
        sm = f"{cap[0]}{cap[1]}"
        _EXT = load(
            name="gpu_trading_env_C",
            sources=[str(_CSRC / "binding.cpp"), str(_CSRC / "env_step.cu")],
            extra_include_paths=[str(_CSRC)],
            extra_cuda_cflags=[
                "-O3", "--use_fast_math", "-lineinfo",
                f"-gencode=arch=compute_{sm},code=sm_{sm}",
                f"-gencode=arch=compute_{sm},code=compute_{sm}",
                "--expt-relaxed-constexpr",
            ],
            extra_cflags=["-O3", "-std=c++17"],
            verbose=False,
        )
    except Exception as e:  # pragma: no cover
        _EXT = None
        _EXT_ERR = f"JIT build failed: {type(e).__name__}: {e}"
    return _EXT


@dataclass
class EnvConfig:
    fee_bps: float = 10.0
    buffer_bps: float = 5.0
    max_quote_offset_bps: float = 50.0
    max_leverage: float = 5.0
    maint_margin: float = 0.0625
    liq_penalty: float = 0.10  # fraction of prior equity
    init_cash: float = 10_000.0
    episode_len: int = 2048

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "EnvConfig":
        if not d:
            return cls()
        fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in fields})


def _synthetic_ohlc(T: int, seed: int = 0, device: str = "cuda") -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    steps = torch.randn(T, generator=g) * 0.002
    close = 100.0 * torch.exp(torch.cumsum(steps, dim=0))
    rng = torch.rand(T, generator=g) * 0.003 + 0.001
    high = close * (1 + rng)
    low = close * (1 - rng)
    open_ = torch.roll(close, 1)
    open_[0] = close[0]
    ohlc = torch.stack([open_, high, low, close], dim=1).to(torch.float32)
    return ohlc.to(device)


def _load_ohlc(path: Union[str, Path]) -> torch.Tensor:
    path = Path(path)
    if path.suffix == ".pt":
        t = torch.load(path, map_location="cpu")
        if isinstance(t, dict):
            t = t.get("ohlc") or next(iter(t.values()))
    elif path.suffix in (".npy",):
        import numpy as np
        t = torch.from_numpy(np.load(path))
    else:
        import numpy as np
        arr = np.loadtxt(path, delimiter=",", dtype=np.float32)
        t = torch.from_numpy(arr)
    t = t.to(torch.float32).contiguous()
    if t.ndim != 2 or t.size(1) != 4:
        raise ValueError(f"OHLC must be [T, 4], got {tuple(t.shape)}")
    return t.cuda()


@dataclass
class EnvHandle:
    B: int
    cfg: EnvConfig
    ohlc: torch.Tensor
    state: dict = field(default_factory=dict)
    _reward: torch.Tensor = field(init=False)
    _cost: torch.Tensor = field(init=False)

    def __post_init__(self):
        dev = self.ohlc.device
        B = self.B
        self.state = {
            "pos_qty":      torch.zeros(B, device=dev, dtype=torch.float32),
            "pos_entry_px": torch.zeros(B, device=dev, dtype=torch.float32),
            "cash":         torch.full((B,), self.cfg.init_cash, device=dev, dtype=torch.float32),
            "equity":       torch.full((B,), self.cfg.init_cash, device=dev, dtype=torch.float32),
            "dd_peak":      torch.full((B,), self.cfg.init_cash, device=dev, dtype=torch.float32),
            "drawdown":     torch.zeros(B, device=dev, dtype=torch.float32),
            "t_idx":        torch.zeros(B, device=dev, dtype=torch.int32),
            "done":         torch.zeros(B, device=dev, dtype=torch.int32),
        }
        self._reward = torch.zeros(B, device=dev, dtype=torch.float32)
        self._cost = torch.zeros(B, 4, device=dev, dtype=torch.float32)

    def reset(self, mask: Optional[torch.Tensor] = None) -> None:
        if mask is None:
            mask_bool = torch.ones(self.B, dtype=torch.bool, device=self.ohlc.device)
        else:
            mask_bool = mask.to(torch.bool).to(self.ohlc.device)
        s = self.state
        idx = mask_bool
        s["pos_qty"][idx] = 0.0
        s["pos_entry_px"][idx] = 0.0
        s["cash"][idx] = self.cfg.init_cash
        s["equity"][idx] = self.cfg.init_cash
        s["dd_peak"][idx] = self.cfg.init_cash
        s["drawdown"][idx] = 0.0
        s["t_idx"][idx] = 0
        s["done"][idx] = 0

    def _obs(self) -> dict:
        # Lightweight obs dict view — trainers can build features from this.
        s = self.state
        ti = s["t_idx"].clamp_max(self.ohlc.size(0) - 1).long()
        bar = self.ohlc.index_select(0, ti)
        return {
            "bar": bar,
            "equity": s["equity"],
            "pos_qty": s["pos_qty"],
            "drawdown": s["drawdown"],
        }

    def step(self, action: torch.Tensor):
        ext = _load_ext()
        if ext is None:
            raise RuntimeError(_EXT_ERR or "gpu_trading_env._C not loaded")
        if action.dtype != torch.float32:
            action = action.to(torch.float32)
        if not action.is_contiguous():
            action = action.contiguous()
        if action.shape != (self.B, 4):
            raise ValueError(f"action must be [{self.B}, 4], got {tuple(action.shape)}")
        s = self.state
        ext.env_step(
            self.ohlc,
            s["pos_qty"], s["pos_entry_px"], s["cash"], s["equity"],
            s["dd_peak"], s["drawdown"], s["t_idx"], s["done"],
            action, self._reward, self._cost,
            float(self.cfg.fee_bps),
            float(self.cfg.buffer_bps),
            float(self.cfg.max_quote_offset_bps),
            float(self.cfg.max_leverage),
            float(self.cfg.maint_margin),
            float(self.cfg.liq_penalty),
            float(self.cfg.init_cash),
            int(self.ohlc.size(0)),
            int(self.cfg.episode_len),
        )
        return self._obs(), self._reward, s["done"], self._cost


def make(
    B: int,
    ohlc_path_or_tensor: Union[None, str, Path, torch.Tensor] = None,
    params: Optional[dict] = None,
    T_synth: int = 8192,
    seed: int = 0,
) -> EnvHandle:
    if not torch.cuda.is_available():
        raise RuntimeError("gpu_trading_env.make requires CUDA")
    cfg = EnvConfig.from_dict(params)
    if ohlc_path_or_tensor is None:
        ohlc = _synthetic_ohlc(T_synth, seed=seed, device="cuda")
    elif isinstance(ohlc_path_or_tensor, torch.Tensor):
        ohlc = ohlc_path_or_tensor.to(torch.float32).contiguous().cuda()
        if ohlc.ndim != 2 or ohlc.size(1) != 4:
            raise ValueError("ohlc tensor must be [T, 4]")
    else:
        ohlc = _load_ohlc(ohlc_path_or_tensor)
    return EnvHandle(B=B, cfg=cfg, ohlc=ohlc)


__all__ = ["make", "EnvConfig", "EnvHandle", "_load_ext", "_EXT_ERR"]
