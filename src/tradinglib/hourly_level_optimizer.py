from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class HourlyLevelSearchConfig:
    """Config for rolling hourly limit-level optimisation.

    The optimiser fits entry/take-profit bps on a short trailing window and
    immediately replays the chosen levels on the next forward window.
    """

    lookback_bars: int = 48
    forward_bars: int = 48
    entry_bps_grid: tuple[float, ...] = (5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0)
    take_profit_bps_grid: tuple[float, ...] = (10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0)
    fill_buffer_bps: float = 5.0
    fee_bps: float = 10.0
    max_hold_bars: int = 12
    stop_loss_bps: float = 0.0
    min_train_trades: int = 1
    min_deploy_train_return_pct: float = -1e9
    min_deploy_train_win_rate_pct: float = 0.0
    side_mode: str = "long"
    close_open_positions: bool = True
    device: str = "cpu"


@dataclass(frozen=True)
class LevelGridResult:
    entry_bps: float
    take_profit_bps: float
    train_return_pct: float
    train_trades: int
    train_win_rate_pct: float
    side: str = "long"


@dataclass(frozen=True)
class WalkForwardLevelWindow:
    start_timestamp: pd.Timestamp
    end_timestamp: pd.Timestamp
    entry_bps: float
    take_profit_bps: float
    train_return_pct: float
    forward_return_pct: float
    forward_trades: int
    forward_win_rate_pct: float
    side: str = "long"


@dataclass(frozen=True)
class WalkForwardLevelResult:
    symbol: str
    windows: tuple[WalkForwardLevelWindow, ...]

    @property
    def total_return_pct(self) -> float:
        equity = 1.0
        for window in self.windows:
            equity *= 1.0 + window.forward_return_pct / 100.0
        return (equity - 1.0) * 100.0

    @property
    def median_window_return_pct(self) -> float:
        if not self.windows:
            return 0.0
        return float(np.median([w.forward_return_pct for w in self.windows]))


def _require_finite(name: str, value: float) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _require_finite_nonnegative(name: str, value: float) -> float:
    parsed = _require_finite(name, value)
    if parsed < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return parsed


def _require_nonnegative_int(name: str, value: int) -> int:
    finite = _require_finite(name, float(value))
    parsed = int(finite)
    if parsed < 0 or float(parsed) != finite:
        raise ValueError(f"{name} must be a non-negative integer")
    return parsed


def _require_positive_int(name: str, value: int) -> int:
    finite = _require_finite(name, float(value))
    parsed = int(finite)
    if parsed <= 0 or float(parsed) != finite:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def validate_hourly_level_search_config(config: HourlyLevelSearchConfig) -> None:
    """Reject configs that would produce misleading optimizer reports."""
    _require_positive_int("lookback_bars", config.lookback_bars)
    _require_positive_int("forward_bars", config.forward_bars)
    _require_positive_int("max_hold_bars", config.max_hold_bars)
    _require_nonnegative_int("min_train_trades", config.min_train_trades)
    _require_finite_nonnegative("fill_buffer_bps", config.fill_buffer_bps)
    _require_finite_nonnegative("fee_bps", config.fee_bps)
    _require_finite_nonnegative("stop_loss_bps", config.stop_loss_bps)
    _require_finite("min_deploy_train_return_pct", config.min_deploy_train_return_pct)
    if str(config.side_mode).lower() not in {"long", "short", "both"}:
        raise ValueError("side_mode must be one of: long, short, both")
    min_win_rate = _require_finite("min_deploy_train_win_rate_pct", config.min_deploy_train_win_rate_pct)
    if not 0.0 <= min_win_rate <= 100.0:
        raise ValueError("min_deploy_train_win_rate_pct must be between 0 and 100")
    _as_float_grid(config.entry_bps_grid, device=torch.device("cpu"))
    _as_float_grid(config.take_profit_bps_grid, device=torch.device("cpu"))


def _validate_ohlc_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "open", "high", "low", "close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"hourly frame missing columns: {sorted(missing)}")
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp")
    out = out.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    for col in ("open", "high", "low", "close"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out[(out["open"] > 0) & (out["high"] > 0) & (out["low"] > 0) & (out["close"] > 0)]
    return out.reset_index(drop=True)


def _as_float_grid(values: Sequence[float], *, device: torch.device) -> torch.Tensor:
    arr = torch.as_tensor([float(v) for v in values], dtype=torch.float32, device=device)
    if arr.numel() == 0:
        raise ValueError("grid must not be empty")
    if not torch.isfinite(arr).all() or bool((arr <= 0).any()):
        raise ValueError("grid values must be finite and positive")
    return arr / 10_000.0


@torch.no_grad()
def simulate_long_level_grid_torch(
    high: torch.Tensor,
    low: torch.Tensor,
    close: torch.Tensor,
    entry_bps_grid: Sequence[float],
    take_profit_bps_grid: Sequence[float],
    *,
    prev_close0: float | None = None,
    fill_buffer_bps: float = 5.0,
    fee_bps: float = 10.0,
    max_hold_bars: int = 12,
    stop_loss_bps: float = 0.0,
    close_open_positions: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replay a long limit-entry/take-profit strategy over all grid cells.

    Returns ``(return_fraction, trade_count, win_count)`` tensors shaped
    ``[len(entry_bps_grid), len(take_profit_bps_grid)]``.
    """
    if high.ndim != 1 or low.ndim != 1 or close.ndim != 1:
        raise ValueError("high, low and close must be one-dimensional tensors")
    if not (high.numel() == low.numel() == close.numel()):
        raise ValueError("high, low and close must have equal length")
    if high.numel() == 0:
        device = high.device
        shape = (len(entry_bps_grid), len(take_profit_bps_grid))
        return (
            torch.zeros(shape, dtype=torch.float32, device=device),
            torch.zeros(shape, dtype=torch.int64, device=device),
            torch.zeros(shape, dtype=torch.int64, device=device),
        )
    if max_hold_bars <= 0:
        raise ValueError("max_hold_bars must be positive")
    _require_finite_nonnegative("fill_buffer_bps", fill_buffer_bps)
    _require_finite_nonnegative("fee_bps", fee_bps)
    _require_finite_nonnegative("stop_loss_bps", stop_loss_bps)

    device = high.device
    entry_grid = _as_float_grid(entry_bps_grid, device=device)[:, None]
    take_grid = _as_float_grid(take_profit_bps_grid, device=device)[None, :]
    fee = float(fee_bps) / 10_000.0
    buffer = float(fill_buffer_bps) / 10_000.0
    stop_loss = max(float(stop_loss_bps), 0.0) / 10_000.0

    shape = (entry_grid.shape[0], take_grid.shape[1])
    in_trade = torch.zeros(shape, dtype=torch.bool, device=device)
    entry_price = torch.zeros(shape, dtype=torch.float32, device=device)
    held_bars = torch.zeros(shape, dtype=torch.int32, device=device)
    returns = torch.zeros(shape, dtype=torch.float32, device=device)
    trades = torch.zeros(shape, dtype=torch.int64, device=device)
    wins = torch.zeros(shape, dtype=torch.int64, device=device)

    prev_close = torch.as_tensor(
        float(prev_close0) if prev_close0 is not None else float(close[0].item()),
        dtype=torch.float32,
        device=device,
    )
    for i in range(high.numel()):
        hi = high[i]
        lo = low[i]
        cl = close[i]

        held_bars = torch.where(in_trade, held_bars + 1, held_bars)
        exit_limit = entry_price * (1.0 + take_grid)
        stop_limit = entry_price * (1.0 - stop_loss)
        hit_exit = in_trade & (hi >= exit_limit * (1.0 + buffer))
        hit_stop = in_trade & (stop_loss > 0.0) & (lo <= stop_limit * (1.0 - buffer))
        timed_out = in_trade & (held_bars >= int(max_hold_bars))
        exit_mask = hit_exit | hit_stop | timed_out
        # OHLC bars do not reveal intrabar ordering; take the conservative
        # outcome when take-profit and stop-loss are both touched.
        exit_price = torch.where(hit_stop, stop_limit, torch.where(hit_exit, exit_limit, cl))
        ret = exit_price * (1.0 - fee) / (entry_price.clamp_min(1e-12) * (1.0 + fee)) - 1.0
        returns = torch.where(exit_mask, returns + ret, returns)
        trades = torch.where(exit_mask, trades + 1, trades)
        wins = torch.where(exit_mask & (ret > 0), wins + 1, wins)
        in_trade = in_trade & ~exit_mask
        entry_price = torch.where(exit_mask, torch.zeros_like(entry_price), entry_price)
        held_bars = torch.where(exit_mask, torch.zeros_like(held_bars), held_bars)

        entry_limit = prev_close * (1.0 - entry_grid)
        enter = (~in_trade) & (~exit_mask) & (lo <= entry_limit * (1.0 - buffer))
        in_trade = in_trade | enter
        entry_price = torch.where(enter, entry_limit.expand(shape), entry_price)
        held_bars = torch.where(enter, torch.zeros_like(held_bars), held_bars)
        prev_close = cl

    if close_open_positions and bool(in_trade.any()):
        final_close = close[-1]
        ret = final_close * (1.0 - fee) / (entry_price.clamp_min(1e-12) * (1.0 + fee)) - 1.0
        returns = torch.where(in_trade, returns + ret, returns)
        trades = torch.where(in_trade, trades + 1, trades)
        wins = torch.where(in_trade & (ret > 0), wins + 1, wins)

    return returns, trades, wins


@torch.no_grad()
def simulate_short_level_grid_torch(
    high: torch.Tensor,
    low: torch.Tensor,
    close: torch.Tensor,
    entry_bps_grid: Sequence[float],
    take_profit_bps_grid: Sequence[float],
    *,
    prev_close0: float | None = None,
    fill_buffer_bps: float = 5.0,
    fee_bps: float = 10.0,
    max_hold_bars: int = 12,
    stop_loss_bps: float = 0.0,
    close_open_positions: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replay a short limit-entry/take-profit strategy over all grid cells.

    Short entries are explicit limit sells above the previous close; exits are
    buy limits below entry. Returns are expressed per unit notional.
    """
    if high.ndim != 1 or low.ndim != 1 or close.ndim != 1:
        raise ValueError("high, low and close must be one-dimensional tensors")
    if not (high.numel() == low.numel() == close.numel()):
        raise ValueError("high, low and close must have equal length")
    if high.numel() == 0:
        device = high.device
        shape = (len(entry_bps_grid), len(take_profit_bps_grid))
        return (
            torch.zeros(shape, dtype=torch.float32, device=device),
            torch.zeros(shape, dtype=torch.int64, device=device),
            torch.zeros(shape, dtype=torch.int64, device=device),
        )
    if max_hold_bars <= 0:
        raise ValueError("max_hold_bars must be positive")
    _require_finite_nonnegative("fill_buffer_bps", fill_buffer_bps)
    _require_finite_nonnegative("fee_bps", fee_bps)
    _require_finite_nonnegative("stop_loss_bps", stop_loss_bps)

    device = high.device
    entry_grid = _as_float_grid(entry_bps_grid, device=device)[:, None]
    take_grid = _as_float_grid(take_profit_bps_grid, device=device)[None, :]
    fee = float(fee_bps) / 10_000.0
    buffer = float(fill_buffer_bps) / 10_000.0
    stop_loss = max(float(stop_loss_bps), 0.0) / 10_000.0

    shape = (entry_grid.shape[0], take_grid.shape[1])
    in_trade = torch.zeros(shape, dtype=torch.bool, device=device)
    entry_price = torch.zeros(shape, dtype=torch.float32, device=device)
    held_bars = torch.zeros(shape, dtype=torch.int32, device=device)
    returns = torch.zeros(shape, dtype=torch.float32, device=device)
    trades = torch.zeros(shape, dtype=torch.int64, device=device)
    wins = torch.zeros(shape, dtype=torch.int64, device=device)

    prev_close = torch.as_tensor(
        float(prev_close0) if prev_close0 is not None else float(close[0].item()),
        dtype=torch.float32,
        device=device,
    )
    for i in range(high.numel()):
        hi = high[i]
        lo = low[i]
        cl = close[i]

        held_bars = torch.where(in_trade, held_bars + 1, held_bars)
        exit_limit = entry_price * (1.0 - take_grid)
        stop_limit = entry_price * (1.0 + stop_loss)
        hit_exit = in_trade & (lo <= exit_limit * (1.0 - buffer))
        hit_stop = in_trade & (stop_loss > 0.0) & (hi >= stop_limit * (1.0 + buffer))
        timed_out = in_trade & (held_bars >= int(max_hold_bars))
        exit_mask = hit_exit | hit_stop | timed_out
        # OHLC bars do not reveal intrabar ordering; take the conservative
        # outcome when take-profit and stop-loss are both touched.
        exit_price = torch.where(hit_stop, stop_limit, torch.where(hit_exit, exit_limit, cl))
        ret = entry_price * (1.0 - fee) / (exit_price.clamp_min(1e-12) * (1.0 + fee)) - 1.0
        returns = torch.where(exit_mask, returns + ret, returns)
        trades = torch.where(exit_mask, trades + 1, trades)
        wins = torch.where(exit_mask & (ret > 0), wins + 1, wins)
        in_trade = in_trade & ~exit_mask
        entry_price = torch.where(exit_mask, torch.zeros_like(entry_price), entry_price)
        held_bars = torch.where(exit_mask, torch.zeros_like(held_bars), held_bars)

        entry_limit = prev_close * (1.0 + entry_grid)
        enter = (~in_trade) & (~exit_mask) & (hi >= entry_limit * (1.0 + buffer))
        in_trade = in_trade | enter
        entry_price = torch.where(enter, entry_limit.expand(shape), entry_price)
        held_bars = torch.where(enter, torch.zeros_like(held_bars), held_bars)
        prev_close = cl

    if close_open_positions and bool(in_trade.any()):
        final_close = close[-1]
        ret = entry_price * (1.0 - fee) / (final_close.clamp_min(1e-12) * (1.0 + fee)) - 1.0
        returns = torch.where(in_trade, returns + ret, returns)
        trades = torch.where(in_trade, trades + 1, trades)
        wins = torch.where(in_trade & (ret > 0), wins + 1, wins)

    return returns, trades, wins


def _best_from_grid(
    ret: torch.Tensor,
    trades: torch.Tensor,
    wins: torch.Tensor,
    config: HourlyLevelSearchConfig,
    *,
    side: str,
) -> LevelGridResult:
    score = ret.clone()
    if config.min_train_trades > 0:
        score = torch.where(trades >= int(config.min_train_trades), score, torch.full_like(score, -1e9))
    flat_idx = int(torch.argmax(score).item())
    row = flat_idx // len(config.take_profit_bps_grid)
    col = flat_idx % len(config.take_profit_bps_grid)
    trade_count = int(trades[row, col].item())
    win_count = int(wins[row, col].item())
    win_rate = 0.0 if trade_count <= 0 else 100.0 * win_count / trade_count
    return LevelGridResult(
        entry_bps=float(config.entry_bps_grid[row]),
        take_profit_bps=float(config.take_profit_bps_grid[col]),
        train_return_pct=float(ret[row, col].item() * 100.0),
        train_trades=trade_count,
        train_win_rate_pct=float(win_rate),
        side=str(side),
    )


def optimize_long_levels_for_window(
    frame: pd.DataFrame,
    config: HourlyLevelSearchConfig | None = None,
    *,
    prev_close0: float | None = None,
) -> LevelGridResult:
    cfg = config or HourlyLevelSearchConfig()
    validate_hourly_level_search_config(cfg)
    clean = _validate_ohlc_frame(frame)
    if clean.empty:
        return LevelGridResult(0.0, 0.0, 0.0, 0, 0.0)
    device = torch.device(cfg.device if cfg.device == "cuda" and torch.cuda.is_available() else "cpu")
    high = torch.as_tensor(clean["high"].to_numpy(np.float32), device=device)
    low = torch.as_tensor(clean["low"].to_numpy(np.float32), device=device)
    close = torch.as_tensor(clean["close"].to_numpy(np.float32), device=device)
    ret, trades, wins = simulate_long_level_grid_torch(
        high,
        low,
        close,
        cfg.entry_bps_grid,
        cfg.take_profit_bps_grid,
        prev_close0=prev_close0,
        fill_buffer_bps=cfg.fill_buffer_bps,
        fee_bps=cfg.fee_bps,
        max_hold_bars=cfg.max_hold_bars,
        stop_loss_bps=cfg.stop_loss_bps,
        close_open_positions=cfg.close_open_positions,
    )
    return _best_from_grid(ret, trades, wins, cfg, side="long")


def optimize_short_levels_for_window(
    frame: pd.DataFrame,
    config: HourlyLevelSearchConfig | None = None,
    *,
    prev_close0: float | None = None,
) -> LevelGridResult:
    cfg = config or HourlyLevelSearchConfig()
    validate_hourly_level_search_config(cfg)
    clean = _validate_ohlc_frame(frame)
    if clean.empty:
        return LevelGridResult(0.0, 0.0, 0.0, 0, 0.0, side="short")
    device = torch.device(cfg.device if cfg.device == "cuda" and torch.cuda.is_available() else "cpu")
    high = torch.as_tensor(clean["high"].to_numpy(np.float32), device=device)
    low = torch.as_tensor(clean["low"].to_numpy(np.float32), device=device)
    close = torch.as_tensor(clean["close"].to_numpy(np.float32), device=device)
    ret, trades, wins = simulate_short_level_grid_torch(
        high,
        low,
        close,
        cfg.entry_bps_grid,
        cfg.take_profit_bps_grid,
        prev_close0=prev_close0,
        fill_buffer_bps=cfg.fill_buffer_bps,
        fee_bps=cfg.fee_bps,
        max_hold_bars=cfg.max_hold_bars,
        stop_loss_bps=cfg.stop_loss_bps,
        close_open_positions=cfg.close_open_positions,
    )
    return _best_from_grid(ret, trades, wins, cfg, side="short")


def optimize_levels_for_window(
    frame: pd.DataFrame,
    config: HourlyLevelSearchConfig | None = None,
    *,
    prev_close0: float | None = None,
) -> LevelGridResult:
    cfg = config or HourlyLevelSearchConfig()
    side_mode = str(cfg.side_mode).lower()
    if side_mode == "long":
        return optimize_long_levels_for_window(frame, cfg, prev_close0=prev_close0)
    if side_mode == "short":
        return optimize_short_levels_for_window(frame, cfg, prev_close0=prev_close0)
    long_result = optimize_long_levels_for_window(frame, cfg, prev_close0=prev_close0)
    short_result = optimize_short_levels_for_window(frame, cfg, prev_close0=prev_close0)
    return max((long_result, short_result), key=lambda result: result.train_return_pct)


def replay_long_levels_for_window(
    frame: pd.DataFrame,
    entry_bps: float,
    take_profit_bps: float,
    config: HourlyLevelSearchConfig | None = None,
    *,
    prev_close0: float | None = None,
) -> tuple[float, int, float]:
    cfg = config or HourlyLevelSearchConfig()
    validate_hourly_level_search_config(cfg)
    clean = _validate_ohlc_frame(frame)
    if clean.empty or entry_bps <= 0 or take_profit_bps <= 0:
        return 0.0, 0, 0.0
    device = torch.device(cfg.device if cfg.device == "cuda" and torch.cuda.is_available() else "cpu")
    high = torch.as_tensor(clean["high"].to_numpy(np.float32), device=device)
    low = torch.as_tensor(clean["low"].to_numpy(np.float32), device=device)
    close = torch.as_tensor(clean["close"].to_numpy(np.float32), device=device)
    ret, trades, wins = simulate_long_level_grid_torch(
        high,
        low,
        close,
        (float(entry_bps),),
        (float(take_profit_bps),),
        prev_close0=prev_close0,
        fill_buffer_bps=cfg.fill_buffer_bps,
        fee_bps=cfg.fee_bps,
        max_hold_bars=cfg.max_hold_bars,
        stop_loss_bps=cfg.stop_loss_bps,
        close_open_positions=cfg.close_open_positions,
    )
    trade_count = int(trades[0, 0].item())
    win_count = int(wins[0, 0].item())
    win_rate = 0.0 if trade_count <= 0 else 100.0 * win_count / trade_count
    return float(ret[0, 0].item() * 100.0), trade_count, float(win_rate)


def replay_short_levels_for_window(
    frame: pd.DataFrame,
    entry_bps: float,
    take_profit_bps: float,
    config: HourlyLevelSearchConfig | None = None,
    *,
    prev_close0: float | None = None,
) -> tuple[float, int, float]:
    cfg = config or HourlyLevelSearchConfig()
    validate_hourly_level_search_config(cfg)
    clean = _validate_ohlc_frame(frame)
    if clean.empty or entry_bps <= 0 or take_profit_bps <= 0:
        return 0.0, 0, 0.0
    device = torch.device(cfg.device if cfg.device == "cuda" and torch.cuda.is_available() else "cpu")
    high = torch.as_tensor(clean["high"].to_numpy(np.float32), device=device)
    low = torch.as_tensor(clean["low"].to_numpy(np.float32), device=device)
    close = torch.as_tensor(clean["close"].to_numpy(np.float32), device=device)
    ret, trades, wins = simulate_short_level_grid_torch(
        high,
        low,
        close,
        (float(entry_bps),),
        (float(take_profit_bps),),
        prev_close0=prev_close0,
        fill_buffer_bps=cfg.fill_buffer_bps,
        fee_bps=cfg.fee_bps,
        max_hold_bars=cfg.max_hold_bars,
        stop_loss_bps=cfg.stop_loss_bps,
        close_open_positions=cfg.close_open_positions,
    )
    trade_count = int(trades[0, 0].item())
    win_count = int(wins[0, 0].item())
    win_rate = 0.0 if trade_count <= 0 else 100.0 * win_count / trade_count
    return float(ret[0, 0].item() * 100.0), trade_count, float(win_rate)


def replay_levels_for_window(
    frame: pd.DataFrame,
    entry_bps: float,
    take_profit_bps: float,
    config: HourlyLevelSearchConfig | None = None,
    *,
    side: str,
    prev_close0: float | None = None,
) -> tuple[float, int, float]:
    if str(side).lower() == "short":
        return replay_short_levels_for_window(frame, entry_bps, take_profit_bps, config, prev_close0=prev_close0)
    return replay_long_levels_for_window(frame, entry_bps, take_profit_bps, config, prev_close0=prev_close0)


def walk_forward_hourly_level_search(
    frame: pd.DataFrame,
    *,
    symbol: str = "",
    config: HourlyLevelSearchConfig | None = None,
) -> WalkForwardLevelResult:
    cfg = config or HourlyLevelSearchConfig()
    validate_hourly_level_search_config(cfg)
    clean = _validate_ohlc_frame(frame)
    windows: list[WalkForwardLevelWindow] = []
    for start in range(int(cfg.lookback_bars), len(clean), int(cfg.forward_bars)):
        end = min(start + int(cfg.forward_bars), len(clean))
        if end <= start:
            break
        train = clean.iloc[start - int(cfg.lookback_bars) : start]
        forward = clean.iloc[start:end]
        train_prev = None
        if start - int(cfg.lookback_bars) > 0:
            train_prev = float(clean.iloc[start - int(cfg.lookback_bars) - 1]["close"])
        best = optimize_levels_for_window(train, cfg, prev_close0=train_prev)
        deploy = (
            best.train_trades >= int(cfg.min_train_trades)
            and best.train_return_pct >= float(cfg.min_deploy_train_return_pct)
            and best.train_win_rate_pct >= float(cfg.min_deploy_train_win_rate_pct)
        )
        if deploy:
            forward_ret, forward_trades, forward_win_rate = replay_levels_for_window(
                forward,
                best.entry_bps,
                best.take_profit_bps,
                cfg,
                side=best.side,
                prev_close0=float(train.iloc[-1]["close"]),
            )
        else:
            forward_ret, forward_trades, forward_win_rate = 0.0, 0, 0.0
        windows.append(
            WalkForwardLevelWindow(
                start_timestamp=pd.Timestamp(forward.iloc[0]["timestamp"]),
                end_timestamp=pd.Timestamp(forward.iloc[-1]["timestamp"]),
                entry_bps=best.entry_bps,
                take_profit_bps=best.take_profit_bps,
                train_return_pct=best.train_return_pct,
                forward_return_pct=forward_ret,
                forward_trades=forward_trades,
                forward_win_rate_pct=forward_win_rate,
                side=best.side,
            )
        )
    return WalkForwardLevelResult(symbol=str(symbol).upper(), windows=tuple(windows))


def summarize_walk_forward_results(results: Iterable[WalkForwardLevelResult]) -> dict[str, float]:
    rows = []
    for result in results:
        for window in result.windows:
            rows.append(window.forward_return_pct)
    if not rows:
        return {
            "n_windows": 0.0,
            "median_window_return_pct": 0.0,
            "mean_window_return_pct": 0.0,
            "p10_window_return_pct": 0.0,
        }
    arr = np.asarray(rows, dtype=np.float64)
    return {
        "n_windows": float(arr.size),
        "median_window_return_pct": float(np.median(arr)),
        "mean_window_return_pct": float(np.mean(arr)),
        "p10_window_return_pct": float(np.percentile(arr, 10)),
    }


__all__ = [
    "HourlyLevelSearchConfig",
    "LevelGridResult",
    "WalkForwardLevelResult",
    "WalkForwardLevelWindow",
    "optimize_levels_for_window",
    "optimize_long_levels_for_window",
    "optimize_short_levels_for_window",
    "replay_levels_for_window",
    "replay_long_levels_for_window",
    "replay_short_levels_for_window",
    "simulate_long_level_grid_torch",
    "simulate_short_level_grid_torch",
    "summarize_walk_forward_results",
    "validate_hourly_level_search_config",
    "walk_forward_hourly_level_search",
]
