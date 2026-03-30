from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


BINDING_IMPORT_ERROR: Exception | None = None

_HEADER_SIZE = 64
_SYMBOL_NAME_LEN = 16
_PRICE_FEATURES = 5
_CLOSE_IDX = 3


@dataclass(frozen=True)
class _SharedMarketData:
    symbols: list[str]
    features: np.ndarray
    prices: np.ndarray
    tradable: np.ndarray | None

    @property
    def num_symbols(self) -> int:
        return int(self.features.shape[1])

    @property
    def num_timesteps(self) -> int:
        return int(self.features.shape[0])

    @property
    def features_per_symbol(self) -> int:
        return int(self.features.shape[2])


@dataclass
class _EnvState:
    start_idx: int = 0
    timestep: int = 0
    step_count: int = 0
    current_action: int = 0
    hold_steps: int = 0
    total_hold_steps: int = 0
    num_trades: int = 0
    winning_trades: int = 0
    equity: float = 10_000.0
    peak_equity: float = 10_000.0
    max_drawdown: float = 0.0
    current_trade_return: float = 0.0
    rewards: list[float] = field(default_factory=list)
    done: bool = False
    log: dict[str, float | int] = field(default_factory=dict)


@dataclass
class _VecHandle:
    data: _SharedMarketData
    obs_bufs: np.ndarray
    act_bufs: np.ndarray
    rew_bufs: np.ndarray
    term_bufs: np.ndarray
    trunc_bufs: np.ndarray
    n_envs: int
    max_steps: int
    fee_rate: float
    short_borrow_apr: float
    periods_per_year: float
    fill_slippage_bps: float
    forced_offset: int
    action_allocation_bins: int
    action_level_bins: int
    offsets: np.ndarray
    envs: list[_EnvState]
    completed_logs: list[dict[str, float | int]] = field(default_factory=list)
    closed: bool = False


_SHARED_CACHE: dict[str, _SharedMarketData] = {}
_ACTIVE_DATA: _SharedMarketData | None = None


def _read_mktd(path: str | Path) -> _SharedMarketData:
    path = Path(path)
    with path.open("rb") as f:
        header = f.read(_HEADER_SIZE)
        if len(header) != _HEADER_SIZE:
            raise ValueError(f"Short MKTD header: {path}")
        if header[:4] != b"MKTD":
            raise ValueError(f"Bad MKTD magic in {path}: {header[:4]!r}")

        num_symbols = int.from_bytes(header[8:12], "little", signed=False)
        num_timesteps = int.from_bytes(header[12:16], "little", signed=False)
        features_per_symbol = int.from_bytes(header[16:20], "little", signed=False)
        price_features = int.from_bytes(header[20:24], "little", signed=False)

        if num_symbols <= 0 or num_timesteps <= 0 or features_per_symbol <= 0:
            raise ValueError(f"Invalid MKTD dimensions in {path}")
        if price_features != _PRICE_FEATURES:
            raise ValueError(f"Unsupported price feature count in {path}: {price_features}")

        raw_symbols = f.read(num_symbols * _SYMBOL_NAME_LEN)
        if len(raw_symbols) != num_symbols * _SYMBOL_NAME_LEN:
            raise ValueError(f"Short MKTD symbol table: {path}")
        symbols = []
        for idx in range(num_symbols):
            raw = raw_symbols[idx * _SYMBOL_NAME_LEN : (idx + 1) * _SYMBOL_NAME_LEN]
            symbols.append(raw.split(b"\x00", 1)[0].decode("ascii", errors="ignore").strip() or f"SYM{idx}")

        feature_count = num_timesteps * num_symbols * features_per_symbol
        features = np.fromfile(f, dtype=np.float32, count=feature_count)
        if features.size != feature_count:
            raise ValueError(f"Short MKTD features array in {path}")
        features = features.reshape((num_timesteps, num_symbols, features_per_symbol))

        price_count = num_timesteps * num_symbols * price_features
        prices = np.fromfile(f, dtype=np.float32, count=price_count)
        if prices.size != price_count:
            raise ValueError(f"Short MKTD price array in {path}")
        prices = prices.reshape((num_timesteps, num_symbols, price_features))

        remainder = f.read()
        tradable = None
        if remainder:
            mask_count = num_timesteps * num_symbols
            mask = np.frombuffer(remainder, dtype=np.uint8, count=mask_count)
            if mask.size == mask_count:
                tradable = mask.reshape((num_timesteps, num_symbols))

    return _SharedMarketData(symbols=symbols, features=features, prices=prices, tradable=tradable)


def _require_shared_data() -> _SharedMarketData:
    if _ACTIVE_DATA is None:
        raise RuntimeError("MarketData not loaded. Call binding.shared(data_path=...) first.")
    return _ACTIVE_DATA


def _decode_action(action: int, num_symbols: int, alloc_bins: int, level_bins: int) -> tuple[bool, int, float]:
    if action <= 0:
        return False, -1, 0.0
    alloc_bins = max(1, int(alloc_bins))
    level_bins = max(1, int(level_bins))
    per_symbol_actions = alloc_bins * level_bins
    action_idx = action - 1
    side_block = num_symbols * per_symbol_actions
    is_short = action_idx >= side_block
    if is_short:
        action_idx -= side_block
    sym_idx = action_idx // per_symbol_actions
    bucket = action_idx % per_symbol_actions
    alloc_idx = bucket // level_bins
    alloc = (alloc_idx + 1) / alloc_bins
    return is_short, sym_idx, float(alloc)


def _close_price(data: _SharedMarketData, timestep: int, sym_idx: int) -> float:
    t = int(np.clip(timestep, 0, data.num_timesteps - 1))
    return float(data.prices[t, sym_idx, _CLOSE_IDX])


def _is_tradable(data: _SharedMarketData, timestep: int, sym_idx: int) -> bool:
    if data.tradable is None:
        return True
    t = int(np.clip(timestep, 0, data.num_timesteps - 1))
    return bool(int(data.tradable[t, sym_idx]) != 0)


def _write_observation(handle: _VecHandle, env_idx: int) -> None:
    env = handle.envs[env_idx]
    obs = handle.obs_bufs[env_idx]
    obs.fill(0.0)
    if env.done:
        return

    data = handle.data
    timestep = int(np.clip(env.timestep, 0, data.num_timesteps - 1))
    feat_size = data.num_symbols * data.features_per_symbol
    obs[:feat_size] = data.features[timestep].reshape(-1)

    base = feat_size
    obs[base + 0] = env.equity / 10_000.0
    obs[base + 3] = env.hold_steps / max(1, handle.max_steps)
    obs[base + 4] = env.step_count / max(1, handle.max_steps)
    if env.current_action > 0:
        is_short, sym_idx, alloc = _decode_action(
            env.current_action,
            data.num_symbols,
            handle.action_allocation_bins,
            handle.action_level_bins,
        )
        if 0 <= sym_idx < data.num_symbols:
            obs[base + 1] = -alloc if is_short else alloc
            obs[base + 5 + sym_idx] = -alloc if is_short else alloc


def _finalize_env(handle: _VecHandle, env_idx: int) -> None:
    env = handle.envs[env_idx]
    if env.done and env.log:
        return
    if env.current_action > 0 and env.hold_steps > 0:
        env.total_hold_steps += env.hold_steps
        if env.current_trade_return > 0.0:
            env.winning_trades += 1
        env.hold_steps = 0
        env.current_trade_return = 0.0

    rewards = np.asarray(env.rewards, dtype=np.float64)
    if rewards.size:
        avg_reward = float(rewards.mean())
        downside = rewards[rewards < 0.0]
        if downside.size:
            downside_dev = float(np.sqrt(np.mean(np.square(downside))))
            sortino = float(avg_reward / downside_dev * np.sqrt(rewards.size)) if downside_dev > 0 else 0.0
        else:
            sortino = float(avg_reward * np.sqrt(rewards.size))
    else:
        sortino = 0.0

    env.log = {
        "total_return": float(env.equity / 10_000.0 - 1.0),
        "sortino": sortino,
        "max_drawdown": float(env.max_drawdown),
        "num_trades": int(env.num_trades),
        "win_rate": float(env.winning_trades / env.num_trades) if env.num_trades else 0.0,
        "avg_hold_hours": float(env.total_hold_steps / max(1, env.num_trades)) if env.num_trades else 0.0,
    }
    handle.completed_logs.append(dict(env.log))
    env.done = True
    _write_observation(handle, env_idx)


def shared(*, data_path: str) -> None:
    resolved = str(Path(data_path).resolve())
    data = _SHARED_CACHE.get(resolved)
    if data is None:
        data = _read_mktd(resolved)
        _SHARED_CACHE[resolved] = data
    globals()["_ACTIVE_DATA"] = data


def vec_init(
    obs_bufs: np.ndarray,
    act_bufs: np.ndarray,
    rew_bufs: np.ndarray,
    term_bufs: np.ndarray,
    trunc_bufs: np.ndarray,
    n_envs: int,
    seed: int,
    **kwargs,
):
    del seed
    data = _require_shared_data()
    forced_offset = int(kwargs.get("forced_offset", -1))
    max_steps = max(1, int(kwargs.get("max_steps", 1)))
    alloc_bins = max(1, int(kwargs.get("action_allocation_bins", 1)))
    level_bins = max(1, int(kwargs.get("action_level_bins", 1)))
    handle = _VecHandle(
        data=data,
        obs_bufs=obs_bufs,
        act_bufs=act_bufs,
        rew_bufs=rew_bufs,
        term_bufs=term_bufs,
        trunc_bufs=trunc_bufs,
        n_envs=int(n_envs),
        max_steps=max_steps,
        fee_rate=float(kwargs.get("fee_rate", 0.0)),
        short_borrow_apr=float(kwargs.get("short_borrow_apr", 0.0)),
        periods_per_year=float(kwargs.get("periods_per_year", 252.0)),
        fill_slippage_bps=float(kwargs.get("fill_slippage_bps", 0.0)),
        forced_offset=forced_offset,
        action_allocation_bins=alloc_bins,
        action_level_bins=level_bins,
        offsets=np.full(int(n_envs), forced_offset if forced_offset >= 0 else 0, dtype=np.int32),
        envs=[_EnvState() for _ in range(int(n_envs))],
    )
    return handle


def vec_set_offsets(handle: _VecHandle, offsets: np.ndarray) -> None:
    handle.offsets = np.asarray(offsets, dtype=np.int32).copy()


def vec_reset(handle: _VecHandle, seed: int) -> None:
    del seed
    handle.rew_bufs.fill(0.0)
    handle.term_bufs.fill(0)
    handle.trunc_bufs.fill(0)
    handle.completed_logs.clear()
    for idx, env in enumerate(handle.envs):
        start_idx = int(handle.offsets[idx]) if idx < handle.offsets.size else 0
        start_idx = int(np.clip(start_idx, 0, handle.data.num_timesteps - 1))
        env.start_idx = start_idx
        env.timestep = start_idx
        env.step_count = 0
        env.current_action = 0
        env.hold_steps = 0
        env.total_hold_steps = 0
        env.num_trades = 0
        env.winning_trades = 0
        env.equity = 10_000.0
        env.peak_equity = 10_000.0
        env.max_drawdown = 0.0
        env.current_trade_return = 0.0
        env.rewards.clear()
        env.done = False
        env.log = {}
        _write_observation(handle, idx)


def vec_step(handle: _VecHandle) -> None:
    if handle.closed:
        raise RuntimeError("Vector handle is closed")

    handle.rew_bufs.fill(0.0)
    handle.term_bufs.fill(0)
    handle.trunc_bufs.fill(0)

    data = handle.data
    for idx, env in enumerate(handle.envs):
        if env.done:
            handle.term_bufs[idx] = 1
            continue

        if env.timestep >= data.num_timesteps - 1 or env.step_count >= handle.max_steps:
            handle.term_bufs[idx] = 1
            _finalize_env(handle, idx)
            continue

        action = int(handle.act_bufs[idx])
        reward = 0.0
        if action > 0:
            is_short, sym_idx, alloc = _decode_action(
                action,
                data.num_symbols,
                handle.action_allocation_bins,
                handle.action_level_bins,
            )
            if 0 <= sym_idx < data.num_symbols and _is_tradable(data, env.timestep, sym_idx):
                cur_price = _close_price(data, env.timestep, sym_idx)
                next_price = _close_price(data, env.timestep + 1, sym_idx)
                price_return = 0.0 if cur_price <= 0 else (next_price - cur_price) / cur_price
                reward = (-price_return if is_short else price_return) * alloc
                reward -= handle.fee_rate * alloc
                reward -= handle.fill_slippage_bps * 1e-4 * alloc
                if is_short and handle.short_borrow_apr > 0.0 and handle.periods_per_year > 0.0:
                    reward -= handle.short_borrow_apr / handle.periods_per_year * alloc

            if action != env.current_action:
                if env.current_action > 0 and env.hold_steps > 0:
                    env.total_hold_steps += env.hold_steps
                    if env.current_trade_return > 0.0:
                        env.winning_trades += 1
                env.num_trades += 1
                env.hold_steps = 1
                env.current_trade_return = float(reward)
            else:
                env.hold_steps += 1
                env.current_trade_return += float(reward)
            env.current_action = action
        elif env.current_action > 0 and env.hold_steps > 0:
            env.total_hold_steps += env.hold_steps
            if env.current_trade_return > 0.0:
                env.winning_trades += 1
            env.hold_steps = 0
            env.current_action = 0
            env.current_trade_return = 0.0

        env.rewards.append(float(reward))
        env.equity *= 1.0 + float(reward)
        env.peak_equity = max(env.peak_equity, env.equity)
        if env.peak_equity > 0:
            drawdown = (env.peak_equity - env.equity) / env.peak_equity
            env.max_drawdown = max(env.max_drawdown, float(drawdown))

        env.timestep += 1
        env.step_count += 1
        handle.rew_bufs[idx] = float(reward)

        if env.timestep >= data.num_timesteps - 1 or env.step_count >= handle.max_steps:
            handle.term_bufs[idx] = 1
            _finalize_env(handle, idx)
        else:
            _write_observation(handle, idx)


def vec_close(handle: _VecHandle) -> None:
    handle.closed = True


def vec_env_at(handle: _VecHandle, idx: int):
    return handle, int(idx)


def env_get(env_handle) -> dict[str, float | int]:
    handle, idx = env_handle
    env = handle.envs[int(idx)]
    return dict(env.log)


def vec_log(handle: _VecHandle):
    if handle.completed_logs:
        return handle.completed_logs.pop(0)
    return None
