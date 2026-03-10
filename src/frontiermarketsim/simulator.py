from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Mapping

import torch

from .data import SymbolDataset, align_symbol_lengths, load_symbol_datasets


@dataclass(slots=True)
class FrontierSimConfig:
    context_len: int = 128
    horizon: int = 1
    mode: str = "open_close"
    normalize_returns: bool = True
    seed: int = 1337
    device: str = "auto"
    trading_fee: float = 0.0005
    crypto_trading_fee: float = 0.0015
    slip_bps: float = 1.5
    annual_leverage_rate: float = 0.065
    intraday_leverage_max: float = 4.0
    overnight_leverage_max: float = 2.0


PolicyFn = Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and device.index is None and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return device


class FrontierMarketSimulator:
    """Large-scale market simulator with optional fast C++ backend.

    The simulator maps a bank of symbol OHLCV series into a batched environment.
    Each batch row behaves like an independent market episode.
    """

    def __init__(
        self,
        datasets: list[SymbolDataset],
        *,
        num_envs: int = 4096,
        cfg: FrontierSimConfig | None = None,
        use_fast_backend: bool = True,
    ) -> None:
        if not datasets:
            raise ValueError("datasets must contain at least one symbol.")
        self.cfg = cfg or FrontierSimConfig()
        self.device = _resolve_device(self.cfg.device)
        self.num_envs = int(num_envs)
        if self.num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}")

        aligned = align_symbol_lengths(datasets)
        symbol_bank = torch.stack([item.ohlcv for item in aligned], dim=0).to(torch.float32)
        is_crypto_bank = torch.tensor([item.is_crypto for item in aligned], dtype=torch.bool)
        self.symbols = [item.symbol for item in aligned]

        assignment = torch.arange(self.num_envs, dtype=torch.long) % symbol_bank.shape[0]
        self.env_symbol_indices = assignment
        self.env_symbols = [self.symbols[idx] for idx in assignment.tolist()]

        self.ohlcv = symbol_bank[assignment].to(self.device).contiguous()
        self.is_crypto = is_crypto_bank[assignment].to(self.device).contiguous()

        self.batch_size = int(self.ohlcv.shape[0])
        self.series_length = int(self.ohlcv.shape[1])
        self.feature_dim = int(self.ohlcv.shape[2])
        self.context_len = int(self.cfg.context_len)
        self.horizon = max(1, int(self.cfg.horizon))
        self.episode_end = self.series_length - self.horizon - 1

        if self.context_len <= 0:
            raise ValueError(f"context_len must be > 0, got {self.context_len}")
        if self.episode_end <= self.context_len:
            raise ValueError(
                "Insufficient history for requested context/horizon. "
                f"series_length={self.series_length}, context_len={self.context_len}, horizon={self.horizon}"
            )

        self._fast_backend_error: str | None = None
        self._fast_sim = None
        self.using_fast_backend = False
        if use_fast_backend:
            self._try_init_fast_backend()

        self._fee_rate = torch.where(
            self.is_crypto,
            torch.full((self.batch_size,), float(self.cfg.crypto_trading_fee), device=self.device),
            torch.full((self.batch_size,), float(self.cfg.trading_fee), device=self.device),
        )
        self._slip_rate = torch.full(
            (self.batch_size,),
            float(self.cfg.slip_bps) * 1e-4,
            device=self.device,
        )
        self._daily_financing = torch.full(
            (self.batch_size,),
            float(self.cfg.annual_leverage_rate) / 252.0,
            device=self.device,
        )
        self._daily_financing = torch.where(
            self.is_crypto,
            torch.zeros_like(self._daily_financing),
            self._daily_financing,
        )
        self._step_index = self.context_len
        self._position = torch.zeros((self.batch_size,), dtype=torch.float32, device=self.device)
        self._equity = torch.ones((self.batch_size,), dtype=torch.float32, device=self.device)

    @property
    def fast_backend_error(self) -> str | None:
        return self._fast_backend_error

    def _try_init_fast_backend(self) -> None:
        try:
            from fastmarketsim.config import build_sim_config
            from fastmarketsim.module import load_extension

            sim_cfg = build_sim_config(
                {
                    "context_len": self.context_len,
                    "horizon": self.horizon,
                    "mode": self.cfg.mode,
                    "normalize_returns": self.cfg.normalize_returns,
                    "seed": self.cfg.seed,
                    "trading_fee": self.cfg.trading_fee,
                    "crypto_trading_fee": self.cfg.crypto_trading_fee,
                    "slip_bps": self.cfg.slip_bps,
                    "annual_leverage_rate": self.cfg.annual_leverage_rate,
                    "intraday_leverage_max": self.cfg.intraday_leverage_max,
                    "overnight_leverage_max": self.cfg.overnight_leverage_max,
                }
            )
            extension = load_extension()
            self._fast_sim = extension.MarketSimulator(
                sim_cfg,
                self.ohlcv,
                self.is_crypto,
                str(self.device),
            )
            self.using_fast_backend = True
        except Exception as exc:  # pragma: no cover - depends on local C++ toolchain/runtime
            self._fast_backend_error = str(exc)
            self.using_fast_backend = False
            self._fast_sim = None

    def reset(self, *, start_step: int | None = None) -> torch.Tensor:
        start = self.context_len if start_step is None else int(start_step)
        if start < self.context_len:
            raise ValueError(f"start_step must be >= context_len ({self.context_len}), got {start}")
        if start >= self.episode_end:
            raise ValueError(f"start_step must be < episode_end ({self.episode_end}), got {start}")

        self._step_index = start
        self._position.zero_()
        self._equity.fill_(1.0)

        if self.using_fast_backend:
            return self._fast_sim.reset(start)
        return self._make_observation(start)

    def step(self, actions: torch.Tensor | list[float]) -> Mapping[str, torch.Tensor]:
        action_tensor = torch.as_tensor(actions, dtype=torch.float32, device=self.device).reshape(-1)
        if action_tensor.numel() != self.batch_size:
            raise ValueError(
                f"Expected {self.batch_size} actions, got {action_tensor.numel()}"
            )

        if self.using_fast_backend:
            result = self._fast_sim.step(action_tensor)
            self._step_index += 1
            return result
        return self._step_torch(action_tensor)

    def _make_observation(self, step_index: int) -> torch.Tensor:
        left = step_index - self.context_len
        return self.ohlcv[:, left:step_index, :]

    def _action_to_target(self, actions: torch.Tensor) -> torch.Tensor:
        unit = torch.tanh(actions)
        stock_target = unit * float(self.cfg.intraday_leverage_max)
        crypto_target = torch.clamp(unit, 0.0, 1.0)
        return torch.where(self.is_crypto, crypto_target, stock_target)

    def _step_torch(self, actions: torch.Tensor) -> Mapping[str, torch.Tensor]:
        t = self._step_index
        target = self._action_to_target(actions)
        dpos = target - self._position

        equity = self._equity
        trade_cost = dpos.abs() * (self._fee_rate + self._slip_rate) * equity

        financing_cost = torch.clamp(target.abs() - 1.0, min=0.0) * self._daily_financing * equity

        open_px = self.ohlcv[:, t, 0]
        close_idx = t + self.horizon - 1
        close_px = self.ohlcv[:, close_idx, 3]
        session_ret = (close_px - open_px) / torch.clamp(open_px, min=1e-6)
        gross = equity * target * session_ret

        overnight_cap = torch.where(
            self.is_crypto,
            torch.ones_like(target),
            torch.full_like(target, float(self.cfg.overnight_leverage_max)),
        )
        overnight_floor = torch.where(
            self.is_crypto,
            torch.zeros_like(target),
            -overnight_cap,
        )
        end_position = torch.minimum(torch.maximum(target, overnight_floor), overnight_cap)
        deleverage_notional = (target - end_position).abs()
        deleverage_cost = deleverage_notional * (self._fee_rate + self._slip_rate) * equity

        reward = gross - trade_cost - financing_cost - deleverage_cost
        self._equity = equity + reward
        self._position = end_position
        self._step_index += 1

        done_flag = bool(self._step_index >= self.episode_end)
        done = torch.full((self.batch_size,), done_flag, dtype=torch.bool, device=self.device)
        obs = self._make_observation(self._step_index)

        return {
            "obs": obs,
            "reward": reward,
            "done": done,
            "gross": gross,
            "trade_cost": trade_cost,
            "financing_cost": financing_cost,
            "deleverage_cost": deleverage_cost,
            "deleverage_notional": deleverage_notional,
            "position": end_position,
            "equity": self._equity,
        }

    def run_benchmark(
        self,
        *,
        num_steps: int,
        policy: PolicyFn | None = None,
    ) -> dict[str, float | int | str]:
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")

        policy_fn = policy or self._default_policy
        obs = self.reset()
        last_actions = torch.zeros((self.batch_size,), dtype=torch.float32, device=self.device)
        running_peak = torch.ones((self.batch_size,), dtype=torch.float32, device=self.device)
        max_drawdown = torch.zeros((self.batch_size,), dtype=torch.float32, device=self.device)
        latest_equity = torch.ones((self.batch_size,), dtype=torch.float32, device=self.device)
        reward_chunks: list[torch.Tensor] = []
        total_turnover = 0.0
        steps_executed = 0

        start = time.perf_counter()
        for step in range(num_steps):
            actions = policy_fn(obs, last_actions, step)
            result = self.step(actions)
            reward = result["reward"]
            equity = result["equity"].to(self.device)
            latest_equity = equity

            reward_chunks.append(reward.detach().to("cpu"))
            total_turnover += float((actions - last_actions).abs().mean().item())
            last_actions = actions

            running_peak = torch.maximum(running_peak, equity)
            drawdown = (running_peak - equity) / torch.clamp(running_peak, min=1e-6)
            max_drawdown = torch.maximum(max_drawdown, drawdown)

            obs = result["obs"]
            steps_executed += 1
            if bool(result["done"].all().item()):
                break

        elapsed = max(time.perf_counter() - start, 1e-9)
        rewards = torch.cat(reward_chunks) if reward_chunks else torch.zeros((0,), dtype=torch.float32)
        mean_reward = float(rewards.mean().item()) if rewards.numel() else 0.0
        reward_std = float(rewards.std(unbiased=False).item()) if rewards.numel() else 0.0
        sharpe_like = 0.0 if reward_std < 1e-12 else (mean_reward / reward_std) * math.sqrt(252.0)

        final_equity = latest_equity.detach().to("cpu")
        p01 = torch.quantile(final_equity, 0.01)
        p99 = torch.quantile(final_equity, 0.99)
        trimmed = torch.clamp(final_equity, min=float(p01.item()), max=float(p99.item()))
        env_steps = steps_executed * self.batch_size

        return {
            "backend": "fast" if self.using_fast_backend else "torch",
            "num_symbols": len(self.symbols),
            "num_envs": self.batch_size,
            "steps_requested": int(num_steps),
            "steps_executed": int(steps_executed),
            "env_steps": int(env_steps),
            "elapsed_sec": float(elapsed),
            "env_steps_per_sec": float(env_steps / elapsed),
            "mean_reward": mean_reward,
            "reward_std": reward_std,
            "sharpe_like": sharpe_like,
            "mean_turnover": float(total_turnover / max(1, steps_executed)),
            "final_equity_mean": float(final_equity.mean().item()),
            "final_equity_median": float(torch.median(final_equity).item()),
            "final_equity_trimmed_mean": float(trimmed.mean().item()),
            "final_equity_p05": float(torch.quantile(final_equity, 0.05).item()),
            "final_equity_p95": float(torch.quantile(final_equity, 0.95).item()),
            "max_drawdown_mean": float(max_drawdown.mean().item()),
            "fast_backend_error": self._fast_backend_error or "",
        }

    def _default_policy(self, obs: torch.Tensor, _last_actions: torch.Tensor, step: int) -> torch.Tensor:
        # Trend proxy: use the latest open/close drift from each environment.
        del step
        latest = obs[:, -1, :]
        open_px = latest[:, 0]
        close_px = latest[:, 3]
        momentum = (close_px - open_px) / torch.clamp(open_px, min=1e-6)
        noise = 0.05 * torch.randn_like(momentum)
        return torch.tanh(momentum * 12.0 + noise)


def build_frontier_simulator_from_data(
    data_root: str,
    *,
    symbols: list[str] | None,
    max_symbols: int,
    min_rows: int,
    num_envs: int,
    cfg: FrontierSimConfig,
    use_fast_backend: bool,
) -> FrontierMarketSimulator:
    datasets = load_symbol_datasets(
        data_root,
        symbols=symbols,
        max_symbols=max_symbols,
        min_rows=min_rows,
    )
    return FrontierMarketSimulator(
        datasets,
        num_envs=num_envs,
        cfg=cfg,
        use_fast_backend=use_fast_backend,
    )
