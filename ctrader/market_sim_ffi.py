from __future__ import annotations

import ctypes
import os
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

try:
    import fcntl as _fcntl
except ImportError:
    _fcntl = None

import numpy as np


CTRADER_DIR = Path(__file__).resolve().parent
MAX_NATIVE_WEIGHT_SYMBOLS = 64
_LIBRARY_CACHE: dict[tuple[str, int], ctypes.CDLL] = {}


class WeightSimConfig(ctypes.Structure):
    _fields_ = [
        ("initial_cash", ctypes.c_double),
        ("max_gross_leverage", ctypes.c_double),
        ("fee_rate", ctypes.c_double),
        ("borrow_rate_per_period", ctypes.c_double),
        ("periods_per_year", ctypes.c_double),
        ("can_short", ctypes.c_int),
    ]


class WeightSimResult(ctypes.Structure):
    _fields_ = [
        ("total_return", ctypes.c_double),
        ("annualized_return", ctypes.c_double),
        ("sortino", ctypes.c_double),
        ("max_drawdown", ctypes.c_double),
        ("final_equity", ctypes.c_double),
        ("total_turnover", ctypes.c_double),
        ("total_fees", ctypes.c_double),
        ("total_borrow_cost", ctypes.c_double),
    ] 


class NativeWeightEnvConfig(ctypes.Structure):
    _fields_ = [
        ("lookback", ctypes.c_int),
        ("episode_steps", ctypes.c_int),
        ("reward_scale", ctypes.c_double),
    ]


class NativeWeightEnv(ctypes.Structure):
    _fields_ = [
        ("close", ctypes.POINTER(ctypes.c_double)),
        ("n_bars", ctypes.c_int),
        ("n_symbols", ctypes.c_int),
        ("env_cfg", NativeWeightEnvConfig),
        ("sim_cfg", WeightSimConfig),
        ("start_index", ctypes.c_int),
        ("t", ctypes.c_int),
        ("steps", ctypes.c_int),
        ("equity", ctypes.c_double),
        ("peak_equity", ctypes.c_double),
        ("recent_return", ctypes.c_double),
        ("total_turnover", ctypes.c_double),
        ("total_fees", ctypes.c_double),
        ("total_borrow_cost", ctypes.c_double),
        ("current_weights", ctypes.c_double * MAX_NATIVE_WEIGHT_SYMBOLS),
        ("equity_curve", ctypes.POINTER(ctypes.c_double)),
        ("returns", ctypes.POINTER(ctypes.c_double)),
    ]


class NativeWeightEnvStepInfo(ctypes.Structure):
    _fields_ = [
        ("reward", ctypes.c_double),
        ("turnover", ctypes.c_double),
        ("fees", ctypes.c_double),
        ("borrow_cost", ctypes.c_double),
        ("equity", ctypes.c_double),
        ("period_return", ctypes.c_double),
        ("done", ctypes.c_int),
        ("summary", WeightSimResult),
    ]


@dataclass(frozen=True)
class WeightSimSummary:
    total_return: float
    annualized_return: float
    sortino: float
    max_drawdown: float
    final_equity: float
    total_turnover: float
    total_fees: float
    total_borrow_cost: float

    @classmethod
    def from_c(cls, result: WeightSimResult) -> "WeightSimSummary":
        return cls(
            total_return=float(result.total_return),
            annualized_return=float(result.annualized_return),
            sortino=float(result.sortino),
            max_drawdown=float(result.max_drawdown),
            final_equity=float(result.final_equity),
            total_turnover=float(result.total_turnover),
            total_fees=float(result.total_fees),
            total_borrow_cost=float(result.total_borrow_cost),
        )


def _default_library_path() -> Path:
    return CTRADER_DIR / "libmarket_sim.so"


def _build_lock_path() -> Path:
    return CTRADER_DIR / ".build.lock"


def _needs_rebuild(lib_path: Path) -> bool:
    source_paths = [CTRADER_DIR / "market_sim.c", CTRADER_DIR / "market_sim.h"]
    if not lib_path.exists():
        return True
    return any(src.stat().st_mtime > lib_path.stat().st_mtime for src in source_paths)


@contextmanager
def _build_lock_context(lock_path: Path) -> Iterator[None]:
    with lock_path.open("a+b") as handle:
        if _fcntl is not None:
            _fcntl.flock(handle.fileno(), _fcntl.LOCK_EX)
        try:
            yield
        finally:
            if _fcntl is not None:
                _fcntl.flock(handle.fileno(), _fcntl.LOCK_UN)


def _validate_native_symbol_count(n_symbols: int) -> None:
    if n_symbols > MAX_NATIVE_WEIGHT_SYMBOLS:
        raise ValueError(
            "Native ctrader weight simulation supports at most "
            f"{MAX_NATIVE_WEIGHT_SYMBOLS} symbols; received {n_symbols}."
        )


def ensure_library_built(path: str | os.PathLike[str] | None = None) -> Path:
    lib_path = Path(path) if path is not None else _default_library_path()
    default_lib_path = _default_library_path()
    if lib_path.resolve() != default_lib_path.resolve():
        if not lib_path.exists():
            raise FileNotFoundError(f"Native ctrader library does not exist: {lib_path}")
        return lib_path

    if not _needs_rebuild(lib_path):
        return lib_path

    with _build_lock_context(_build_lock_path()):
        if not _needs_rebuild(lib_path):
            return lib_path
        try:
            subprocess.run(
                ["make", "libmarket_sim.so"],
                cwd=CTRADER_DIR,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            output = (exc.stdout or "").strip()
            message = [
                f"Failed to build native ctrader library at {lib_path}",
                "command: make libmarket_sim.so",
            ]
            if output:
                message.append(f"output:\n{output}")
            raise RuntimeError("\n".join(message)) from exc
    return lib_path


def _configure_library(lib: ctypes.CDLL) -> ctypes.CDLL:
    lib.simulate_target_weights.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(WeightSimConfig),
        ctypes.POINTER(WeightSimResult),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.simulate_target_weights.restype = None
    lib.compute_annualized_return.argtypes = [ctypes.c_double, ctypes.c_int, ctypes.c_double]
    lib.compute_annualized_return.restype = ctypes.c_double
    lib.weight_env_obs_dim.argtypes = [ctypes.POINTER(NativeWeightEnv)]
    lib.weight_env_obs_dim.restype = ctypes.c_int
    lib.weight_env_init.argtypes = [
        ctypes.POINTER(NativeWeightEnv),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(NativeWeightEnvConfig),
        ctypes.POINTER(WeightSimConfig),
    ]
    lib.weight_env_init.restype = ctypes.c_int
    lib.weight_env_free.argtypes = [ctypes.POINTER(NativeWeightEnv)]
    lib.weight_env_free.restype = None
    lib.weight_env_reset.argtypes = [ctypes.POINTER(NativeWeightEnv), ctypes.c_int]
    lib.weight_env_reset.restype = ctypes.c_int
    lib.weight_env_get_obs.argtypes = [
        ctypes.POINTER(NativeWeightEnv),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
    ]
    lib.weight_env_get_obs.restype = ctypes.c_int
    lib.weight_env_step.argtypes = [
        ctypes.POINTER(NativeWeightEnv),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.POINTER(NativeWeightEnvStepInfo),
    ]
    lib.weight_env_step.restype = ctypes.c_int
    return lib


def load_library(path: str | os.PathLike[str] | None = None) -> ctypes.CDLL:
    lib_path = ensure_library_built(path if path is not None else os.environ.get("CTRADER_SIM_LIB"))
    resolved_path = lib_path.resolve()
    cache_key = (str(resolved_path), resolved_path.stat().st_mtime_ns)
    cached = _LIBRARY_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        lib = _configure_library(ctypes.CDLL(str(resolved_path)))
    except OSError as exc:
        raise RuntimeError(
            f"Failed to load native ctrader library at {resolved_path}: {exc}"
        ) from exc
    stale_keys = [key for key in _LIBRARY_CACHE if key[0] == cache_key[0] and key != cache_key]
    for stale_key in stale_keys:
        del _LIBRARY_CACHE[stale_key]
    _LIBRARY_CACHE[cache_key] = lib
    return lib


def _as_config(config: WeightSimConfig | Mapping[str, float | int]) -> WeightSimConfig:
    if isinstance(config, WeightSimConfig):
        return config
    return WeightSimConfig(
        initial_cash=float(config.get("initial_cash", 10000.0)),
        max_gross_leverage=float(config.get("max_gross_leverage", 1.0)),
        fee_rate=float(config.get("fee_rate", 0.0)),
        borrow_rate_per_period=float(config.get("borrow_rate_per_period", 0.0)),
        periods_per_year=float(config.get("periods_per_year", 8760.0)),
        can_short=int(config.get("can_short", 0)),
    )


def simulate_target_weights(
    close: np.ndarray,
    target_weights: np.ndarray,
    config: WeightSimConfig | Mapping[str, float | int],
    library: ctypes.CDLL | None = None,
) -> tuple[WeightSimSummary, np.ndarray]:
    close_arr = np.ascontiguousarray(close, dtype=np.float64)
    weights_arr = np.ascontiguousarray(target_weights, dtype=np.float64)

    if close_arr.ndim != 2 or weights_arr.ndim != 2:
        raise ValueError("close and target_weights must both be 2D [n_bars, n_symbols]")
    if close_arr.shape != weights_arr.shape:
        raise ValueError("close and target_weights must have the same shape")

    n_bars, n_symbols = close_arr.shape
    _validate_native_symbol_count(int(n_symbols))
    lib = library or load_library()
    eq_curve = np.zeros(n_bars, dtype=np.float64)
    cfg = _as_config(config)
    result = WeightSimResult()

    lib.simulate_target_weights(
        int(n_bars),
        int(n_symbols),
        close_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        weights_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(cfg),
        ctypes.byref(result),
        eq_curve.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    return WeightSimSummary.from_c(result), eq_curve


class NativeWeightEnvHandle:
    def __init__(
        self,
        close: np.ndarray,
        env_config: NativeWeightEnvConfig,
        sim_config: WeightSimConfig | Mapping[str, float | int],
        library: ctypes.CDLL | None = None,
    ) -> None:
        self.close = np.ascontiguousarray(close, dtype=np.float64)
        if self.close.ndim != 2:
            raise ValueError("close must be 2D [n_bars, n_symbols]")
        _validate_native_symbol_count(int(self.close.shape[1]))
        self.lib = library or load_library()
        self.env = NativeWeightEnv()
        self.sim_config = _as_config(sim_config)
        rc = self.lib.weight_env_init(
            ctypes.byref(self.env),
            self.close.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            int(self.close.shape[0]),
            int(self.close.shape[1]),
            ctypes.byref(env_config),
            ctypes.byref(self.sim_config),
        )
        if rc != 0:
            raise RuntimeError(f"weight_env_init failed with code {rc}")

    def close_env(self) -> None:
        self.lib.weight_env_free(ctypes.byref(self.env))

    def __enter__(self) -> "NativeWeightEnvHandle":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close_env()

    def reset(self, start_index: int) -> None:
        rc = self.lib.weight_env_reset(ctypes.byref(self.env), int(start_index))
        if rc != 0:
            raise RuntimeError(f"weight_env_reset failed with code {rc}")

    def obs_dim(self) -> int:
        return int(self.lib.weight_env_obs_dim(ctypes.byref(self.env)))

    def get_obs(self) -> np.ndarray:
        obs = np.zeros(self.obs_dim(), dtype=np.float64)
        rc = self.lib.weight_env_get_obs(
            ctypes.byref(self.env),
            obs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            int(obs.size),
        )
        if rc != 0:
            raise RuntimeError(f"weight_env_get_obs failed with code {rc}")
        return obs

    def step(self, raw_scores: np.ndarray) -> NativeWeightEnvStepInfo:
        scores = np.ascontiguousarray(raw_scores, dtype=np.float64)
        info = NativeWeightEnvStepInfo()
        rc = self.lib.weight_env_step(
            ctypes.byref(self.env),
            scores.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            int(scores.size),
            ctypes.byref(info),
        )
        if rc != 0:
            raise RuntimeError(f"weight_env_step failed with code {rc}")
        return info
