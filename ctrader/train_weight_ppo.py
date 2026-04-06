from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NotRequired, Sequence, TypedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions import Normal

from ctrader.market_sim_ffi import WeightSimConfig, simulate_target_weights
from pufferlib_market.metrics import annualize_total_return


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = REPO_ROOT / "trainingdatahourlybinance"


@dataclass(frozen=True)
class PPOTrainConfig:
    lookback: int = 48
    episode_steps: int = 168
    rollout_steps: int = 1024
    total_updates: int = 12
    hidden_dim: int = 256
    num_layers: int = 2
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    batch_size: int = 256
    ppo_epochs: int = 4
    reward_scale: float = 100.0
    fee_rate: float = 0.001
    borrow_rate_per_period: float = 0.0
    max_gross_leverage: float = 1.0
    can_short: bool = False
    initial_cash: float = 10000.0
    periods_per_year: float = 8760.0
    eval_every_updates: int = 2
    seed: int = 1337
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ContinuousWeightStepInfo(TypedDict):
    equity: float
    period_return: float
    fees: float
    borrow_cost: float
    turnover: float
    total_return: NotRequired[float]
    annualized_return: NotRequired[float]
    sortino: NotRequired[float]
    max_drawdown: NotRequired[float]
    weights: NotRequired[np.ndarray]
    equity_curve: NotRequired[np.ndarray]


class PPORolloutBatch(TypedDict):
    obs: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    rewards: np.ndarray
    values: np.ndarray
    dones: np.ndarray


class PPOUpdateStats(TypedDict):
    pg_loss: float
    vf_loss: float
    entropy: float
    grad_norm_mean: float
    grad_norm_max: float


class PPOHistoryRow(PPOUpdateStats):
    update: int
    eval_total_return: NotRequired[float]
    eval_annualized_return: NotRequired[float]
    eval_sortino: NotRequired[float]
    eval_max_drawdown: NotRequired[float]
    eval_turnover: NotRequired[float]
    eval_action_score_std: NotRequired[float]


class PPOTrainResult(TypedDict):
    config: dict[str, float | int | bool | str]
    best_eval: dict[str, float]
    history: list[PPOHistoryRow]

def load_close_matrix(
    symbols: Sequence[str],
    *,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    start: str | None = None,
    end: str | None = None,
    max_rows: int | None = None,
    min_rows: int = 3,
) -> pd.DataFrame:
    data_root = Path(data_root)
    frames: list[pd.Series] = []
    for symbol in symbols:
        path = data_root / f"{symbol}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing data file: {path}")
        frame = pd.read_csv(path, usecols=["timestamp", "close"])
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.dropna(subset=["close"]).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        series = frame.set_index("timestamp")["close"].astype(np.float64).rename(symbol)
        frames.append(series)

    close = pd.concat(frames, axis=1, join="inner").dropna().sort_index()
    if start is not None:
        close = close.loc[pd.Timestamp(start, tz="UTC"):]
    if end is not None:
        close = close.loc[:pd.Timestamp(end, tz="UTC")]
    if max_rows is not None and len(close) > max_rows:
        close = close.iloc[-max_rows:]
    if len(close) < min_rows:
        raise ValueError(f"Not enough aligned close rows after filtering: {len(close)} < {min_rows}.")
    return close


def split_close_frame(
    close: pd.DataFrame,
    *,
    train_fraction: float = 0.7,
    min_segment_rows: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.2 <= train_fraction < 0.95:
        raise ValueError("train_fraction must be in [0.2, 0.95).")
    split_idx = int(len(close) * train_fraction)
    train = close.iloc[:split_idx].copy()
    val = close.iloc[split_idx:].copy()
    if len(train) < min_segment_rows or len(val) < min_segment_rows:
        raise ValueError(
            f"Need at least {min_segment_rows} rows in both train and validation splits; "
            f"got train={len(train)} val={len(val)}."
        )
    return train, val


class ContinuousWeightEnv:
    def __init__(self, close: np.ndarray, config: PPOTrainConfig):
        if close.ndim != 2:
            raise ValueError("close must be 2D [n_bars, n_symbols].")
        if close.shape[0] <= config.lookback + 2:
            raise ValueError("close matrix is too short for the requested lookback.")
        self.close = np.asarray(close, dtype=np.float64)
        self.config = config
        self.num_symbols = int(self.close.shape[1])
        self.max_start = int(self.close.shape[0] - config.episode_steps - 2)
        self.rng = np.random.default_rng(config.seed)
        self.reset()

    @property
    def obs_dim(self) -> int:
        return self.config.lookback * self.num_symbols + self.num_symbols + 4

    def reset(self, start_index: int | None = None) -> np.ndarray:
        if start_index is None:
            if self.max_start <= self.config.lookback:
                start_index = self.config.lookback
            else:
                start_index = int(self.rng.integers(self.config.lookback, self.max_start))
        self.start_index = int(start_index)
        self.t = int(start_index)
        self.steps = 0
        self.weights = np.zeros(self.num_symbols, dtype=np.float64)
        self.equity = float(self.config.initial_cash)
        self.peak_equity = float(self.config.initial_cash)
        self.recent_return = 0.0
        self.returns: list[float] = []
        self.weight_history: list[np.ndarray] = []
        self.equity_curve = np.zeros(self.config.episode_steps + 1, dtype=np.float64)
        self.equity_curve[0] = self.equity
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        window = self.close[self.t - self.config.lookback : self.t]
        log_returns = np.diff(np.log(np.clip(window, 1e-12, None)), axis=0)
        if log_returns.shape[0] == 0:
            features = np.zeros((self.config.lookback, self.num_symbols), dtype=np.float32)
        else:
            padded = np.zeros((self.config.lookback, self.num_symbols), dtype=np.float32)
            padded[-log_returns.shape[0] :] = log_returns.astype(np.float32)
            features = padded
        drawdown = 1.0 - (self.equity / max(self.peak_equity, 1e-8))
        portfolio_state = np.array(
            [
                self.recent_return,
                drawdown,
                self.equity / max(self.config.initial_cash, 1e-8) - 1.0,
                self.steps / max(self.config.episode_steps, 1),
            ],
            dtype=np.float32,
        )
        obs = np.concatenate([features.reshape(-1), self.weights.astype(np.float32), portfolio_state])
        return obs.astype(np.float32)  # type: ignore[no-any-return]

    def scores_to_weights(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float64)
        if self.config.can_short:
            raw = np.tanh(scores)
            gross = np.sum(np.abs(raw))
            if gross > self.config.max_gross_leverage and gross > 0.0:
                raw = raw * (self.config.max_gross_leverage / gross)
            return raw  # type: ignore[no-any-return]

        shifted = scores - float(np.max(scores))
        exp_scores = np.exp(np.clip(shifted, -30.0, 30.0))
        total = float(np.sum(exp_scores))
        if total <= 0.0:
            weights = np.zeros_like(scores)
        else:
            weights = exp_scores / total
        return weights * self.config.max_gross_leverage

    def step(self, scores: np.ndarray) -> tuple[np.ndarray, float, bool, ContinuousWeightStepInfo]:
        weights = self.scores_to_weights(scores)
        prev_prices = self.close[self.t]
        next_prices = self.close[self.t + 1]
        turnover = float(np.sum(np.abs(weights - self.weights)))
        fees = self.equity * turnover * self.config.fee_rate
        long_exposure = float(np.sum(np.clip(weights, 0.0, None)))
        short_exposure = float(np.sum(np.clip(-weights, 0.0, None)))
        borrow_base = max(0.0, long_exposure - 1.0) + short_exposure
        borrow_cost = self.equity * borrow_base * self.config.borrow_rate_per_period
        gross_return = float(np.dot(weights, (next_prices - prev_prices) / np.clip(prev_prices, 1e-12, None)))
        new_equity = max(0.0, self.equity * (1.0 + gross_return) - fees - borrow_cost)
        period_return = new_equity / max(self.equity, 1e-8) - 1.0

        self.weights = weights
        self.equity = new_equity
        self.peak_equity = max(self.peak_equity, self.equity)
        self.recent_return = period_return
        self.returns.append(period_return)
        self.weight_history.append(weights.copy())
        self.steps += 1
        self.t += 1
        self.equity_curve[self.steps] = self.equity

        done = self.steps >= self.config.episode_steps or self.t >= self.close.shape[0] - 1
        reward = float(period_return * self.config.reward_scale)
        info: ContinuousWeightStepInfo = {
            "equity": self.equity,
            "period_return": period_return,
            "fees": fees,
            "borrow_cost": borrow_cost,
            "turnover": turnover,
        }
        if done:
            total_return = self.equity / self.config.initial_cash - 1.0
            returns_arr = np.asarray(self.returns, dtype=np.float64)
            neg = returns_arr[returns_arr < 0.0]
            downside = float(np.std(neg)) if neg.size else 0.0
            mean_ret = float(np.mean(returns_arr)) if returns_arr.size else 0.0
            sortino = mean_ret / max(downside, 1e-8) * math.sqrt(self.config.periods_per_year)
            drawdowns = 1.0 - self.equity_curve[: self.steps + 1] / np.maximum.accumulate(
                np.clip(self.equity_curve[: self.steps + 1], 1e-8, None)
            )
            info.update(
                {
                    "total_return": total_return,
                    "annualized_return": annualize_total_return(
                        total_return,
                        periods=max(self.steps, 1),
                        periods_per_year=self.config.periods_per_year,
                    ),
                    "sortino": sortino,
                    "max_drawdown": float(np.max(drawdowns)) if drawdowns.size else 0.0,
                    "weights": np.stack(self.weight_history, axis=0) if self.weight_history else np.zeros(
                        (0, self.num_symbols), dtype=np.float64
                    ),
                    "equity_curve": self.equity_curve[: self.steps + 1].copy(),
                }
            )
        return self._get_obs(), reward, done, info


class GaussianWeightPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()])
            in_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.encoder(obs)
        mean = self.mean_head(hidden)
        value = self.value_head(hidden).squeeze(-1)
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std, value

    def dist_and_value(self, obs: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        mean, std, value = self.forward(obs)
        return Normal(mean, std), value

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self.dist_and_value(obs)
        action = dist.mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self.dist_and_value(obs)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, value, entropy


@dataclass
class EvalSummary:
    total_return: float
    annualized_return: float
    sortino: float
    max_drawdown: float
    final_equity: float
    total_turnover: float
    total_fees: float
    total_borrow_cost: float
    action_score_std: float


class WeightPPOTrainer:
    def __init__(self, train_close: np.ndarray, val_close: np.ndarray, config: PPOTrainConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.env = ContinuousWeightEnv(train_close, config)
        self.val_close = np.asarray(val_close, dtype=np.float64)
        self.policy = GaussianWeightPolicy(
            obs_dim=self.env.obs_dim,
            action_dim=self.env.num_symbols,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        self.rng = np.random.default_rng(config.seed)

    def _collect_rollout(self) -> PPORolloutBatch:
        obs = self.env.reset()
        obs_buf, act_buf, logp_buf = [], [], []
        rew_buf, val_buf, done_buf = [], [], []
        grad_norms: list[float] = []

        for _ in range(self.config.rollout_steps):
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                action_t, logp_t, value_t = self.policy.act(obs_t, deterministic=False)
            action = action_t.squeeze(0).cpu().numpy()
            next_obs, reward, done, info = self.env.step(action)
            obs_buf.append(obs)
            act_buf.append(action)
            logp_buf.append(float(logp_t.item()))
            rew_buf.append(float(reward))
            val_buf.append(float(value_t.item()))
            done_buf.append(bool(done))
            obs = self.env.reset() if done else next_obs
            grad_norms.append(0.0)

        return {
            "obs": np.asarray(obs_buf, dtype=np.float32),
            "actions": np.asarray(act_buf, dtype=np.float32),
            "log_probs": np.asarray(logp_buf, dtype=np.float32),
            "rewards": np.asarray(rew_buf, dtype=np.float32),
            "values": np.asarray(val_buf, dtype=np.float32),
            "dones": np.asarray(done_buf, dtype=np.float32),
        }

    def _compute_gae(self, rewards: np.ndarray, values: np.ndarray, dones: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(len(rewards))):
            next_value = 0.0 if t == len(rewards) - 1 else float(values[t + 1])
            mask = 1.0 - float(dones[t])
            delta = float(rewards[t]) + self.config.gamma * next_value * mask - float(values[t])
            last_gae = delta + self.config.gamma * self.config.gae_lambda * mask * last_gae
            advantages[t] = last_gae
        returns = advantages + values
        return advantages, returns

    def _update(self, rollout: PPORolloutBatch) -> PPOUpdateStats:
        obs = torch.from_numpy(rollout["obs"]).float().to(self.device)
        actions = torch.from_numpy(rollout["actions"]).float().to(self.device)
        old_log_probs = torch.from_numpy(rollout["log_probs"]).float().to(self.device)
        advantages_np, returns_np = self._compute_gae(rollout["rewards"], rollout["values"], rollout["dones"])
        advantages = torch.from_numpy(advantages_np).float().to(self.device)
        returns = torch.from_numpy(returns_np).float().to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std().clamp_min(1e-8))

        pg_losses: list[float] = []
        vf_losses: list[float] = []
        entropies: list[float] = []
        grad_norms: list[float] = []

        for _ in range(self.config.ppo_epochs):
            indices = self.rng.permutation(len(obs))
            for start in range(0, len(obs), self.config.batch_size):
                batch_idx = indices[start : start + self.config.batch_size]
                batch_obs = obs[batch_idx]
                batch_actions = actions[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]

                log_prob, value, entropy = self.policy.evaluate_actions(batch_obs, batch_actions)
                ratio = torch.exp(log_prob - batch_old_log_probs)
                pg_loss_1 = -batch_advantages * ratio
                pg_loss_2 = -batch_advantages * torch.clamp(
                    ratio,
                    1.0 - self.config.clip_eps,
                    1.0 + self.config.clip_eps,
                )
                pg_loss = torch.maximum(pg_loss_1, pg_loss_2).mean()
                vf_loss = ((value - batch_returns) ** 2).mean()
                entropy_bonus = entropy.mean()
                loss = pg_loss + self.config.value_coef * vf_loss - self.config.entropy_coef * entropy_bonus

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = float(nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm))
                self.optimizer.step()

                pg_losses.append(float(pg_loss.item()))
                vf_losses.append(float(vf_loss.item()))
                entropies.append(float(entropy_bonus.item()))
                grad_norms.append(grad_norm)

        return {
            "pg_loss": float(np.mean(pg_losses)) if pg_losses else 0.0,
            "vf_loss": float(np.mean(vf_losses)) if vf_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else 0.0,
            "grad_norm_max": float(np.max(grad_norms)) if grad_norms else 0.0,
        }

    def evaluate_close(self, close: np.ndarray) -> EvalSummary:
        close_arr = np.asarray(close, dtype=np.float64)
        env = ContinuousWeightEnv(close_arr, self.config)
        obs = env.reset(start_index=self.config.lookback)
        raw_scores: list[np.ndarray] = []
        while True:
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                action_t, _, _ = self.policy.act(obs_t, deterministic=True)
            scores = action_t.squeeze(0).cpu().numpy().astype(np.float64)
            raw_scores.append(scores.copy())
            next_obs, _, done, info = env.step(scores)
            obs = next_obs
            if done:
                break

        weights = np.zeros_like(close_arr, dtype=np.float64)
        if raw_scores:
            target_weights = np.stack([env.scores_to_weights(scores) for scores in raw_scores], axis=0)
            start = env.start_index
            stop = start + target_weights.shape[0]
            weights[start:stop] = target_weights
        sim_cfg = WeightSimConfig(
            initial_cash=self.config.initial_cash,
            max_gross_leverage=self.config.max_gross_leverage,
            fee_rate=self.config.fee_rate,
            borrow_rate_per_period=self.config.borrow_rate_per_period,
            periods_per_year=self.config.periods_per_year,
            can_short=1 if self.config.can_short else 0,
        )
        sim_summary, _ = simulate_target_weights(close_arr, weights, sim_cfg)
        raw_action_std = float(np.std(np.asarray(raw_scores))) if raw_scores else 0.0
        return EvalSummary(
            total_return=sim_summary.total_return,
            annualized_return=sim_summary.annualized_return,
            sortino=sim_summary.sortino,
            max_drawdown=sim_summary.max_drawdown,
            final_equity=sim_summary.final_equity,
            total_turnover=sim_summary.total_turnover,
            total_fees=sim_summary.total_fees,
            total_borrow_cost=sim_summary.total_borrow_cost,
            action_score_std=raw_action_std,
        )

    def train(self) -> PPOTrainResult:
        best_eval: EvalSummary | None = None
        history: list[PPOHistoryRow] = []

        for update_idx in range(1, self.config.total_updates + 1):
            rollout = self._collect_rollout()
            train_stats = self._update(rollout)
            record: PPOHistoryRow = {
                "update": update_idx,
                "pg_loss": train_stats["pg_loss"],
                "vf_loss": train_stats["vf_loss"],
                "entropy": train_stats["entropy"],
                "grad_norm_mean": train_stats["grad_norm_mean"],
                "grad_norm_max": train_stats["grad_norm_max"],
            }
            if update_idx % self.config.eval_every_updates == 0 or update_idx == self.config.total_updates:
                eval_summary = self.evaluate_close(self.val_close)
                record.update(
                    {
                        "eval_total_return": eval_summary.total_return,
                        "eval_annualized_return": eval_summary.annualized_return,
                        "eval_sortino": eval_summary.sortino,
                        "eval_max_drawdown": eval_summary.max_drawdown,
                        "eval_turnover": eval_summary.total_turnover,
                        "eval_action_score_std": eval_summary.action_score_std,
                    }
                )
                if best_eval is None or eval_summary.annualized_return > best_eval.annualized_return:
                    best_eval = eval_summary
            history.append(record)

        if best_eval is None:
            best_eval = self.evaluate_close(self.val_close)

        return {
            "config": asdict(self.config),
            "best_eval": asdict(best_eval),
            "history": history,
        }


def run_training(
    close_frame: pd.DataFrame,
    config: PPOTrainConfig,
    *,
    train_fraction: float = 0.7,
) -> PPOTrainResult:
    min_rows = max(config.lookback + 2, config.lookback + config.episode_steps + 2)
    train_close, val_close = split_close_frame(
        close_frame,
        train_fraction=train_fraction,
        min_segment_rows=min_rows,
    )
    trainer = WeightPPOTrainer(train_close.to_numpy(dtype=np.float64), val_close.to_numpy(dtype=np.float64), config)
    return trainer.train()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a continuous target-weight PPO baseline on Binance close data.")
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols, e.g. LTCUSDT,ADAUSDT,BCHUSDT")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--lookback", type=int, default=48)
    parser.add_argument("--episode-steps", type=int, default=168)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--total-updates", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--max-gross-leverage", type=float, default=1.0)
    parser.add_argument("--borrow-rate-per-period", type=float, default=0.0)
    parser.add_argument(
        "--can-short",
        action="store_true",
        help="Allow signed target weights instead of long-only normalized weights.",
    )
    parser.add_argument("--periods-per-year", type=float, default=8760.0)
    parser.add_argument("--max-rows", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    symbols = [part.strip().upper() for part in args.symbols.split(",") if part.strip()]
    close = load_close_matrix(symbols, data_root=args.data_root, max_rows=args.max_rows)
    config = PPOTrainConfig(
        lookback=args.lookback,
        episode_steps=args.episode_steps,
        rollout_steps=args.rollout_steps,
        total_updates=args.total_updates,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        ppo_epochs=args.ppo_epochs,
        fee_rate=args.fee_rate,
        max_gross_leverage=args.max_gross_leverage,
        borrow_rate_per_period=args.borrow_rate_per_period,
        can_short=args.can_short,
        periods_per_year=args.periods_per_year,
        seed=args.seed,
        device=args.device,
    )
    result = run_training(close, config, train_fraction=args.train_fraction)
    payload = {"symbols": symbols, "rows": len(close), **result}
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
