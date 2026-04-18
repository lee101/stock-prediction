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
        # Force compiler temp files onto repo-local tmp/ (/tmp is sandboxed
        # on this box and fills up with CUDA artefacts).
        _repo_root = _PKG_DIR.parents[2]  # gpu_trading_env/python/gpu_trading_env -> repo
        build_tmp = _repo_root / "tmp" / "cuda_build"
        build_tmp.mkdir(parents=True, exist_ok=True)
        os.environ["TMPDIR"] = str(build_tmp)
        from torch.utils.cpp_extension import load
        sm = f"{cap[0]}{cap[1]}"
        _EXT = load(
            name="gpu_trading_env_C",
            sources=[str(_CSRC / "binding.cpp"), str(_CSRC / "env_step.cu"),
                     str(_CSRC / "env_step_multisym.cu"),
                     str(_CSRC / "env_step_portfolio.cu")],
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


# ---------------------------------------------------------------------------
# Multi-symbol handle: loads a pufferlib .bin, presents obs_dim=S*F+5+S,
# act_dim=1+2*S, delegates fill to the single-instrument CUDA kernel using
# an equal-weighted proxy price derived from the feature tensor.
# ---------------------------------------------------------------------------

def _load_bin_features(path: Union[str, Path]) -> tuple:
    """Load a pufferlib_market .bin file.

    Returns (features, num_symbols, num_timesteps, features_per_sym).
    features: [T, S*F] float32 on CPU.
    """
    import struct
    import numpy as np
    path = Path(path)
    with open(path, "rb") as f:
        header = f.read(64)
        magic, version, num_symbols, num_timesteps, features_per_sym, _ = struct.unpack(
            "<4sIIIII", header[:24]
        )
        if magic != b"MKTD":
            raise ValueError(f"Bad magic in .bin: {magic!r}, expected b'MKTD'")
        if features_per_sym == 0:
            features_per_sym = 16
        data = np.frombuffer(f.read(), dtype=np.float32)
    total = num_timesteps * num_symbols * features_per_sym
    if data.size < total:
        raise ValueError(
            f"bin data too short: got {data.size} floats, need {total} "
            f"(T={num_timesteps}, S={num_symbols}, F={features_per_sym})"
        )
    features = torch.from_numpy(data[:total].copy().reshape(num_timesteps, num_symbols * features_per_sym))
    return features, int(num_symbols), int(num_timesteps), int(features_per_sym)


def _build_proxy_ohlc(features: torch.Tensor, num_symbols: int,
                      features_per_sym: int) -> torch.Tensor:
    """Build a [T, 4] proxy OHLC from the multi-symbol feature tensor.

    Feature 0 of each symbol is used as a price proxy (typically close or
    a price-like feature). We average across symbols then build O/H/L/C
    with a small synthetic spread.
    """
    T = features.size(0)
    S = num_symbols
    F = features_per_sym
    feat_reshaped = features.reshape(T, S, F)
    prices = feat_reshaped[:, :, 0].abs().clamp(min=1e-4)
    avg_price = prices.mean(dim=1)  # [T]
    spread = 0.002
    close = avg_price
    open_ = torch.roll(close, 1)
    open_[0] = close[0]
    high = close * (1.0 + spread)
    low = close * (1.0 - spread)
    return torch.stack([open_, high, low, close], dim=1).to(torch.float32)


@dataclass
class MultiSymbolEnvHandle:
    """GPU-resident env wrapping the single-instrument CUDA kernel but
    presenting the full multi-symbol v5_rsi observation and action shapes.

    obs_dim = S*F + 5 + S  (market features + portfolio state + position encoding)
    act_dim = 1 + 2*S      (flat + S longs + S shorts)

    Internally, continuous actions [B, act_dim] are mapped to a single proxy
    instrument for fill simulation. This is a shape-matching v1.
    """
    B: int
    cfg: EnvConfig
    features: torch.Tensor     # [T, S*F] on CUDA
    ohlc_proxy: torch.Tensor   # [T, 4] on CUDA
    num_symbols: int
    features_per_sym: int
    state: dict = field(default_factory=dict)
    _reward: torch.Tensor = field(init=False)
    _cost: torch.Tensor = field(init=False)
    _sym_positions: torch.Tensor = field(init=False)  # [B, S]

    @property
    def obs_dim(self) -> int:
        return self.num_symbols * self.features_per_sym + 5 + self.num_symbols

    @property
    def action_dim(self) -> int:
        return 1 + 2 * self.num_symbols

    def __post_init__(self):
        dev = self.features.device
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
        self._sym_positions = torch.zeros(B, self.num_symbols, device=dev, dtype=torch.float32)

    def reset(self, mask: Optional[torch.Tensor] = None) -> None:
        if mask is None:
            mask_bool = torch.ones(self.B, dtype=torch.bool, device=self.features.device)
        else:
            mask_bool = mask.to(torch.bool).to(self.features.device)
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
        self._sym_positions[idx] = 0.0

    def _obs(self) -> torch.Tensor:
        """Build the full obs matching pufferlib_market format.

        Layout: [S*F market features | 5 portfolio state | S position encoding]
        """
        s = self.state
        T = self.features.size(0)
        ti = s["t_idx"].clamp(0, T - 1).long()
        feat = self.features.index_select(0, ti)  # [B, S*F]

        equity = s["equity"]
        cash = s["cash"]
        init = self.cfg.init_cash
        cash_frac = cash / (equity.abs() + 1e-6)
        equity_norm = (equity - init) / (init + 1e-6)
        drawdown = s["drawdown"]
        mid_px = self.ohlc_proxy.index_select(0, ti)[:, 3]
        pos_qty_norm = s["pos_qty"] / (equity.abs() / (mid_px + 1e-6) + 1e-6)
        gross_lev = (s["pos_qty"].abs() * mid_px) / (equity.abs() + 1e-6)
        portfolio = torch.stack([cash_frac, equity_norm, drawdown, pos_qty_norm, gross_lev], dim=-1)

        return torch.cat([feat, portfolio, self._sym_positions], dim=-1)

    def step(self, action: torch.Tensor):
        ext = _load_ext()
        if ext is None:
            raise RuntimeError(_EXT_ERR or "gpu_trading_env._C not loaded")

        S = self.num_symbols
        if action.dtype != torch.float32:
            action = action.to(torch.float32)
        if not action.is_contiguous():
            action = action.contiguous()

        kernel_action = self._action_to_kernel(action)

        s = self.state
        ext.env_step(
            self.ohlc_proxy,
            s["pos_qty"], s["pos_entry_px"], s["cash"], s["equity"],
            s["dd_peak"], s["drawdown"], s["t_idx"], s["done"],
            kernel_action, self._reward, self._cost,
            float(self.cfg.fee_bps),
            float(self.cfg.buffer_bps),
            float(self.cfg.max_quote_offset_bps),
            float(self.cfg.max_leverage),
            float(self.cfg.maint_margin),
            float(self.cfg.liq_penalty),
            float(self.cfg.init_cash),
            int(self.ohlc_proxy.size(0)),
            int(self.cfg.episode_len),
        )

        self._update_sym_positions(action)
        return self._obs(), self._reward, s["done"], self._cost

    def _action_to_kernel(self, action: torch.Tensor) -> torch.Tensor:
        """Map [B, 1+2S] continuous action to [B, 4] kernel action."""
        S = self.num_symbols
        dev = action.device
        expected = 1 + 2 * S

        if action.dim() == 1:
            idx = action.long()
            cont = torch.zeros(self.B, expected, device=dev, dtype=torch.float32)
            cont[idx == 0, 0] = 1.0
            for i in range(1, expected):
                m = idx == i
                if m.any():
                    cont[m, i] = 1.0
            action = cont

        if action.shape[-1] < expected:
            pad = torch.zeros(self.B, expected - action.shape[-1], device=dev, dtype=torch.float32)
            action = torch.cat([action, pad], dim=-1)
        elif action.shape[-1] > expected:
            action = action[:, :expected]

        flat_sig = torch.tanh(action[:, 0])
        long_sigs = torch.tanh(action[:, 1:S + 1])
        short_sigs = torch.tanh(action[:, S + 1:])

        net_buy = long_sigs.mean(dim=-1).clamp(0.0, 1.0)
        net_sell = short_sigs.mean(dim=-1).clamp(0.0, 1.0)

        go_flat = (flat_sig > 0.0).float()
        net_buy = net_buy * (1.0 - go_flat)
        net_sell = net_sell * (1.0 - go_flat)

        ti = self.state["t_idx"].clamp(0, self.ohlc_proxy.size(0) - 1).long()
        close = self.ohlc_proxy.index_select(0, ti)[:, 3]

        p_bid = close * (1.0 + 1e-3)
        p_ask = close * (1.0 - 1e-3)
        q_bid = net_buy
        q_ask = net_sell

        return torch.stack([p_bid, p_ask, q_bid, q_ask], dim=-1).contiguous()

    def _update_sym_positions(self, action: torch.Tensor) -> None:
        S = self.num_symbols
        if action.dim() == 1 or action.shape[-1] < 1 + 2 * S:
            return
        long_sigs = torch.tanh(action[:, 1:S + 1])
        short_sigs = torch.tanh(action[:, S + 1:2 * S + 1])
        pos = (long_sigs > 0.3).float() - (short_sigs > 0.3).float()
        done_mask = self.state["done"].bool()
        pos[done_mask] = 0.0
        self._sym_positions = pos


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


def make_multi_symbol(
    B: int,
    bin_path: Union[str, Path],
    params: Optional[dict] = None,
) -> MultiSymbolEnvHandle:
    """Create a multi-symbol env from a pufferlib_market .bin file.

    Returns a MultiSymbolEnvHandle with obs_dim = S*F + 5 + S and
    act_dim = 1 + 2*S, matching the production marketsim shapes.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("gpu_trading_env.make_multi_symbol requires CUDA")
    features, num_symbols, num_timesteps, features_per_sym = _load_bin_features(bin_path)
    features = features.to(torch.float32).contiguous().cuda()
    cfg = EnvConfig.from_dict(params)
    if cfg.episode_len > num_timesteps - 1:
        cfg.episode_len = num_timesteps - 1
    ohlc_proxy = _build_proxy_ohlc(features, num_symbols, features_per_sym).cuda()
    return MultiSymbolEnvHandle(
        B=B, cfg=cfg, features=features, ohlc_proxy=ohlc_proxy,
        num_symbols=num_symbols, features_per_sym=features_per_sym,
    )


@dataclass
class MultiSymEnv:
    """Proper multi-symbol GPU env.

    Data tapes (CUDA float32):
        prices   [T, S, 5]     (O, H, L, C, V)
        features [T, S, F]     (optional; obs passthrough)

    Action per-env: int32 in ``[0, 1 + 2*S)``.
        0           -> flat
        1..S        -> long  sym_i
        S+1..2S     -> short sym_i

    Reward: log(equity_t / equity_{t-1}).
    Single-position policy (one symbol at a time) — matches XGB top_n=1 and
    the current RL prod gate semantics.
    """
    B: int
    cfg: EnvConfig
    prices: torch.Tensor          # [T, S, 5]
    features: Optional[torch.Tensor] = None  # [T, S, F]
    state: dict = field(default_factory=dict)
    _reward: torch.Tensor = field(init=False)
    _cost: torch.Tensor = field(init=False)

    @property
    def T(self) -> int:
        return int(self.prices.size(0))

    @property
    def S(self) -> int:
        return int(self.prices.size(1))

    @property
    def action_dim(self) -> int:
        return 1 + 2 * self.S

    def __post_init__(self) -> None:
        dev = self.prices.device
        B = self.B
        ic = self.cfg.init_cash
        self.state = {
            "pos_sym":      torch.full((B,), -1, device=dev, dtype=torch.int32),
            "pos_side":     torch.zeros(B, device=dev, dtype=torch.int32),
            "pos_qty":      torch.zeros(B, device=dev, dtype=torch.float32),
            "pos_entry_px": torch.zeros(B, device=dev, dtype=torch.float32),
            "cash":         torch.full((B,), ic, device=dev, dtype=torch.float32),
            "equity":       torch.full((B,), ic, device=dev, dtype=torch.float32),
            "dd_peak":      torch.full((B,), ic, device=dev, dtype=torch.float32),
            "drawdown":     torch.zeros(B, device=dev, dtype=torch.float32),
            "t_idx":        torch.zeros(B, device=dev, dtype=torch.int32),
            "done":         torch.zeros(B, device=dev, dtype=torch.int32),
        }
        self._reward = torch.zeros(B, device=dev, dtype=torch.float32)
        self._cost = torch.zeros(B, 4, device=dev, dtype=torch.float32)

    def reset(self, mask: Optional[torch.Tensor] = None) -> None:
        dev = self.prices.device
        if mask is None:
            idx = torch.ones(self.B, dtype=torch.bool, device=dev)
        else:
            idx = mask.to(torch.bool).to(dev)
        s = self.state
        s["pos_sym"][idx] = -1
        s["pos_side"][idx] = 0
        s["pos_qty"][idx] = 0.0
        s["pos_entry_px"][idx] = 0.0
        s["cash"][idx] = self.cfg.init_cash
        s["equity"][idx] = self.cfg.init_cash
        s["dd_peak"][idx] = self.cfg.init_cash
        s["drawdown"][idx] = 0.0
        s["t_idx"][idx] = 0
        s["done"][idx] = 0

    def step(self, action: torch.Tensor):
        ext = _load_ext()
        if ext is None:
            raise RuntimeError(_EXT_ERR or "gpu_trading_env._C not loaded")
        if action.dtype != torch.int32:
            action = action.to(torch.int32)
        if not action.is_contiguous():
            action = action.contiguous()
        if action.shape != (self.B,):
            raise ValueError(f"action must be [{self.B}], got {tuple(action.shape)}")
        s = self.state
        ext.env_step_multisym(
            self.prices,
            s["pos_sym"], s["pos_side"], s["pos_qty"], s["pos_entry_px"],
            s["cash"], s["equity"], s["dd_peak"], s["drawdown"],
            s["t_idx"], s["done"], action, self._reward, self._cost,
            float(self.cfg.fee_bps),
            float(self.cfg.buffer_bps),
            float(self.cfg.max_leverage),
            float(self.cfg.maint_margin),
            float(self.cfg.liq_penalty),
            float(self.cfg.init_cash),
            int(self.T),
            int(self.S),
            int(self.cfg.episode_len),
        )
        return self._reward, s["done"], self._cost

    def obs(self) -> torch.Tensor:
        """Build the [B, S*F + 5 + S] obs (matches pufferlib_market layout).
        Callers with no features tape get a features-less [B, 5 + S] obs.
        """
        s = self.state
        ti = s["t_idx"].clamp(0, self.T - 1).long()
        parts: list[torch.Tensor] = []
        if self.features is not None:
            flat_feat = self.features.reshape(self.T, -1)
            parts.append(flat_feat.index_select(0, ti))  # [B, S*F]
        ic = self.cfg.init_cash
        equity = s["equity"]
        cash = s["cash"]
        cash_frac = cash / (equity.abs() + 1e-6)
        eq_norm = (equity - ic) / (ic + 1e-6)
        dd = s["drawdown"]
        pos_qty = s["pos_qty"]
        pos_side_f = s["pos_side"].to(torch.float32)
        portfolio = torch.stack([cash_frac, eq_norm, dd, pos_qty,
                                 pos_side_f], dim=-1)
        parts.append(portfolio)
        sym_one_hot = torch.zeros(self.B, self.S, device=self.prices.device,
                                  dtype=torch.float32)
        held = s["pos_sym"].clamp_min(0).long()
        has_pos = (s["pos_side"] != 0).unsqueeze(-1).to(torch.float32)
        sym_one_hot.scatter_(1, held.unsqueeze(-1), has_pos)
        parts.append(sym_one_hot)
        return torch.cat(parts, dim=-1)


@dataclass
class PortfolioBracketConfig:
    """Knobs for PortfolioBracketEnv (mirrors MultiSymBracketConfig in the
    numpy spec at pufferlib_cpp_market_sim/python/market_sim_py/multisym_bracket_ref.py).
    """
    fee_bps: float = 0.278
    fill_buffer_bps: float = 5.0
    max_leverage: float = 2.0
    annual_margin_rate: float = 0.0625
    trading_days_per_year: int = 252
    init_cash: float = 10_000.0
    episode_len: int = 252  # one trading year per episode

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "PortfolioBracketConfig":
        if not d:
            return cls()
        fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in fields})


@dataclass
class PortfolioBracketEnv:
    """Multi-symbol PORTFOLIO bracket env, GPU-resident.

    State per-env (CUDA float32):
        cash       [B]
        positions  [B, S]   (signed shares; neg = short)
        t_idx      [B] int32
        done       [B] int32

    Action per (env, sym): [B, S, 4]
        action[b, s, 0] = limit_buy_offset_bps
        action[b, s, 1] = limit_sell_offset_bps
        action[b, s, 2] = buy_qty_pct  (>= 0; >1 OK — leverage cap clips)
        action[b, s, 3] = sell_qty_pct (>= 0)

    Bar lookup: prev_close = prices[t-1, :, 3], bar_o/h/l/c = prices[t, :, 0..3].
    Episodes start at ``t_idx = 1`` so prev_close at t-1 is always valid.

    Auto-reset: completed envs (done=1) are reset at the START of the next
    step() call — same convention as MultiSymEnv.
    """
    B: int
    cfg: PortfolioBracketConfig
    prices: torch.Tensor                       # [T, S, 5] (O, H, L, C, V)
    tradable_tape: Optional[torch.Tensor] = None  # [T, S] uint8; None = all true
    features: Optional[torch.Tensor] = None    # [T, S, F] obs passthrough
    state: dict = field(default_factory=dict)
    _reward: torch.Tensor = field(init=False)
    _info: dict = field(init=False)
    _scratch: dict = field(init=False)

    @property
    def T(self) -> int:
        return int(self.prices.size(0))

    @property
    def S(self) -> int:
        return int(self.prices.size(1))

    @property
    def action_shape(self) -> tuple[int, int, int]:
        return (self.B, self.S, 4)

    def __post_init__(self) -> None:
        dev = self.prices.device
        B, S = self.B, self.S
        ic = self.cfg.init_cash
        self.state = {
            "cash":      torch.full((B,), ic, device=dev, dtype=torch.float32),
            "positions": torch.zeros((B, S), device=dev, dtype=torch.float32),
            "t_idx":     torch.ones(B, device=dev, dtype=torch.int32),
            "done":      torch.zeros(B, device=dev, dtype=torch.int32),
            "equity":    torch.full((B,), ic, device=dev, dtype=torch.float32),
        }
        self._reward = torch.zeros(B, device=dev, dtype=torch.float32)
        self._info = {
            "eq_prev":  torch.zeros(B, device=dev, dtype=torch.float32),
            "new_eq":   torch.zeros(B, device=dev, dtype=torch.float32),
            "fees":     torch.zeros(B, device=dev, dtype=torch.float32),
            "margin":   torch.zeros(B, device=dev, dtype=torch.float32),
            "borrowed": torch.zeros(B, device=dev, dtype=torch.float32),
        }
        # Pre-allocated scratch for the per-step gather + kernel I/O.
        self._scratch = {
            "cash_out": torch.empty(B, device=dev, dtype=torch.float32),
            "pos_out":  torch.empty((B, S), device=dev, dtype=torch.float32),
        }
        if self.tradable_tape is None:
            # Lazy-build a static all-true tape.
            self.tradable_tape = torch.ones((self.T, S), device=dev, dtype=torch.uint8)

    def reset(self, mask: Optional[torch.Tensor] = None) -> None:
        """Reset envs selected by ``mask`` (None = all). Always graph-safe:
        no host-syncing branches; uses ``torch.where`` so the topology is
        identical regardless of how many envs are flagged.
        """
        dev = self.prices.device
        s = self.state
        if mask is None:
            mask_b = torch.ones(self.B, dtype=torch.bool, device=dev)
        else:
            mask_b = mask.to(torch.bool).to(dev)
        ic = self.cfg.init_cash
        s["cash"].copy_(torch.where(mask_b, torch.full_like(s["cash"], ic), s["cash"]))
        s["positions"].copy_(torch.where(mask_b.unsqueeze(-1),
                                         torch.zeros_like(s["positions"]),
                                         s["positions"]))
        s["t_idx"].copy_(torch.where(mask_b, torch.ones_like(s["t_idx"]), s["t_idx"]))
        s["done"].copy_(torch.where(mask_b, torch.zeros_like(s["done"]), s["done"]))
        s["equity"].copy_(torch.where(mask_b, torch.full_like(s["equity"], ic), s["equity"]))

    def step(self, action: torch.Tensor):
        ext = _load_ext()
        if ext is None:
            raise RuntimeError(_EXT_ERR or "gpu_trading_env._C not loaded")
        if action.shape != self.action_shape:
            raise ValueError(
                f"action must be {self.action_shape}, got {tuple(action.shape)}"
            )
        if action.dtype != torch.float32:
            action = action.to(torch.float32)
        if not action.is_contiguous():
            action = action.contiguous()

        s = self.state
        # Auto-reset completed envs unconditionally — `done == 0` rows are a
        # no-op under torch.where, but the graph topology stays identical
        # so this is CUDA-Graph capturable.
        self.reset(s["done"].to(torch.bool))

        # Gather per-bar OHLC + tradable for each env's current t_idx.
        ti = s["t_idx"].long()
        # Clamp to [1, T-1] to keep prev (t-1) valid.
        ti = ti.clamp(1, self.T - 1)
        bar = self.prices.index_select(0, ti)            # [B, S, 5]
        prev = self.prices.index_select(0, ti - 1)       # [B, S, 5]
        prev_close = prev[..., 3].contiguous()
        bar_open   = bar[..., 0].contiguous()
        bar_high   = bar[..., 1].contiguous()
        bar_low    = bar[..., 2].contiguous()
        bar_close  = bar[..., 3].contiguous()
        tradable   = self.tradable_tape.index_select(0, ti).contiguous()

        cash_out = self._scratch["cash_out"]
        pos_out  = self._scratch["pos_out"]
        info = self._info
        ext.portfolio_bracket_step(
            s["cash"], s["positions"], action,
            prev_close, bar_open, bar_high, bar_low, bar_close, tradable,
            cash_out, pos_out, self._reward,
            info["eq_prev"], info["new_eq"], info["fees"],
            info["margin"], info["borrowed"],
            float(self.cfg.fee_bps),
            float(self.cfg.fill_buffer_bps),
            float(self.cfg.max_leverage),
            float(self.cfg.annual_margin_rate),
            int(self.cfg.trading_days_per_year),
            int(self.S),
        )

        # Commit state.
        s["cash"].copy_(cash_out)
        s["positions"].copy_(pos_out)
        s["equity"].copy_(info["new_eq"])

        # Advance time and mark done at end of episode / tape.
        new_t = ti + 1
        ep_done = (new_t - 1) >= self.cfg.episode_len  # episode_len bars consumed
        tape_done = new_t >= self.T - 1                # need t+1 in-bounds next step
        s["t_idx"].copy_(new_t.to(torch.int32))
        done = (ep_done | tape_done).to(torch.int32)
        s["done"].copy_(done)

        return self._reward, s["done"], info

    def obs(self) -> torch.Tensor:
        """Build a [B, F_obs] observation.

        Layout: [features[t, :, :] flat (S*F), portfolio (3 + S+S):
        cash_frac, eq_norm, drawdown, positions[B, S], unrealized_per_sym[B, S]].
        Without features tape: [B, 3 + 2S].
        """
        s = self.state
        ti = s["t_idx"].clamp(0, self.T - 1).long()
        parts: list[torch.Tensor] = []
        if self.features is not None:
            flat_feat = self.features.reshape(self.T, -1)
            parts.append(flat_feat.index_select(0, ti))   # [B, S*F]
        ic = self.cfg.init_cash
        equity = s["equity"]
        cash = s["cash"]
        cash_frac = cash / (equity.abs() + 1e-6)
        eq_norm = (equity - ic) / (ic + 1e-6)
        drawdown = torch.zeros_like(equity)  # not tracked in this env yet
        portfolio = torch.stack([cash_frac, eq_norm, drawdown], dim=-1)
        parts.append(portfolio)
        # Per-symbol position fraction of equity (mark-to-current-close).
        bar_close = self.prices.index_select(0, ti)[..., 3]  # [B, S]
        pos_value = s["positions"] * bar_close
        pos_frac = pos_value / (equity.abs().unsqueeze(-1) + 1e-6)
        parts.append(pos_frac)
        # Raw position shares (lightly normalized).
        parts.append(s["positions"] / (ic + 1e-6))
        return torch.cat(parts, dim=-1)


def make_portfolio_bracket(
    B: int,
    prices: torch.Tensor,
    tradable_tape: Optional[torch.Tensor] = None,
    features: Optional[torch.Tensor] = None,
    params: Optional[dict] = None,
) -> PortfolioBracketEnv:
    """Create a PortfolioBracketEnv with an in-memory [T, S, 5] price tape.

    ``tradable_tape`` (optional) [T, S] bool — defaults to all-true.
    ``features`` (optional) [T, S, F] for obs passthrough.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("gpu_trading_env.make_portfolio_bracket requires CUDA")
    if prices.dim() != 3 or prices.size(-1) != 5:
        raise ValueError(f"prices must be [T, S, 5], got {tuple(prices.shape)}")
    prices = prices.to(torch.float32).contiguous().cuda()
    if tradable_tape is not None:
        if tradable_tape.shape != prices.shape[:2]:
            raise ValueError(
                f"tradable_tape must be [T, S]={tuple(prices.shape[:2])}, "
                f"got {tuple(tradable_tape.shape)}"
            )
        tradable_tape = tradable_tape.to(torch.uint8).contiguous().cuda()
    if features is not None:
        features = features.to(torch.float32).contiguous().cuda()
    cfg = PortfolioBracketConfig.from_dict(params)
    if cfg.episode_len > prices.size(0) - 1:
        cfg.episode_len = prices.size(0) - 1
    return PortfolioBracketEnv(
        B=B, cfg=cfg, prices=prices, tradable_tape=tradable_tape, features=features,
    )


def make_multisym(
    B: int,
    prices: torch.Tensor,
    features: Optional[torch.Tensor] = None,
    params: Optional[dict] = None,
) -> MultiSymEnv:
    """Create a MultiSymEnv with an in-memory [T, S, 5] price tape.
    ``features`` (optional) is [T, S, F] and passed through for observation use.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("gpu_trading_env.make_multisym requires CUDA")
    if prices.dim() != 3 or prices.size(-1) != 5:
        raise ValueError(f"prices must be [T, S, 5], got {tuple(prices.shape)}")
    prices = prices.to(torch.float32).contiguous().cuda()
    if features is not None:
        features = features.to(torch.float32).contiguous().cuda()
    cfg = EnvConfig.from_dict(params)
    if cfg.episode_len > prices.size(0) - 1:
        cfg.episode_len = prices.size(0) - 1
    return MultiSymEnv(B=B, cfg=cfg, prices=prices, features=features)


__all__ = ["make", "make_multi_symbol", "make_multisym", "make_portfolio_bracket",
           "EnvConfig", "PortfolioBracketConfig",
           "EnvHandle", "MultiSymbolEnvHandle", "MultiSymEnv", "PortfolioBracketEnv",
           "_load_ext", "_EXT_ERR", "_load_bin_features"]
