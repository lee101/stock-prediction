from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch

from ..config import DataConfig, EnvironmentConfig, EvaluationConfig
from ..data import load_aligned_ohlc, split_train_eval
from ..env import DifferentiableMarketEnv, smooth_abs
from ..features import ohlc_to_features
from ..policy import DirichletGRUPolicy
from ..utils import ensure_dir
from ..differentiable_utils import augment_market_features


@dataclass(slots=True)
class WindowMetrics:
    start: int
    end: int
    objective: float
    mean_reward: float
    std_reward: float
    sharpe: float
    turnover: float
    cumulative_return: float
    max_drawdown: float


class DifferentiableMarketBacktester:
    def __init__(
        self,
        data_cfg: DataConfig,
        env_cfg: EnvironmentConfig,
        eval_cfg: EvaluationConfig,
        use_eval_split: bool = True,
        include_cash_override: bool | None = None,
    ):
        self.data_cfg = data_cfg
        self.env_cfg = env_cfg
        self.eval_cfg = eval_cfg
        self.use_eval_split = use_eval_split
        self._include_cash_override = include_cash_override

        ohlc_all, symbols, index = load_aligned_ohlc(data_cfg)
        self.symbols = symbols
        self.index = index
        if use_eval_split:
            train_tensor, eval_tensor = split_train_eval(ohlc_all)
            self.eval_start_idx = train_tensor.shape[0]
        else:
            eval_tensor = ohlc_all
            self.eval_start_idx = 0
        self.eval_tensor = eval_tensor

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = DifferentiableMarketEnv(env_cfg)

        features, returns = self._prepare_features(add_cash=data_cfg.include_cash, feature_cfg=None)
        self.eval_features = features
        self.eval_returns = returns
        self.asset_names = list(self.symbols) + (["CASH"] if data_cfg.include_cash else [])
        if self.asset_names:
            self.env.set_asset_universe(self.asset_names)
            self.env.reset()

    def run(self, checkpoint_path: Path) -> Dict[str, float]:
        payload = torch.load(checkpoint_path, map_location="cpu")
        data_cfg = payload["config"]["data"]
        # Basic validation to ensure compatibility
        if str(data_cfg["root"]) != str(self.data_cfg.root):
            print("Warning: checkpoint data root differs from current configuration.")

        ckpt_train_cfg = payload["config"].get("train", {})
        ckpt_data_cfg = payload["config"].get("data", {})
        include_cash_config = bool(ckpt_train_cfg.get("include_cash") or ckpt_data_cfg.get("include_cash"))
        if self._include_cash_override is not None:
            include_cash = self._include_cash_override
        else:
            include_cash = include_cash_config or self.data_cfg.include_cash

        self.eval_features, self.eval_returns = self._prepare_features(
            add_cash=include_cash,
            feature_cfg=ckpt_train_cfg,
        )
        self.asset_names = list(self.symbols) + (["CASH"] if include_cash else [])
        if self.asset_names:
            self.env.set_asset_universe(self.asset_names)
            self.env.reset()

        asset_count = self.eval_features.shape[1]
        feature_dim = self.eval_features.shape[-1]

        enable_shorting = bool(ckpt_train_cfg.get("enable_shorting", False))
        max_intraday = float(ckpt_train_cfg.get("max_intraday_leverage", self.env_cfg.max_intraday_leverage))
        max_overnight = float(ckpt_train_cfg.get("max_overnight_leverage", self.env_cfg.max_overnight_leverage))
        self.env_cfg.max_intraday_leverage = max_intraday
        self.env_cfg.max_overnight_leverage = max_overnight
        self._shorting_enabled = enable_shorting

        policy = DirichletGRUPolicy(
            n_assets=asset_count,
            feature_dim=feature_dim,
            gradient_checkpointing=False,
            enable_shorting=enable_shorting,
            max_intraday_leverage=max_intraday,
            max_overnight_leverage=max_overnight,
        ).to(self.device)
        policy.load_state_dict(payload["policy_state"])
        policy.eval()

        window_length = min(self.eval_cfg.window_length, self.eval_features.shape[0])
        if window_length <= 0:
            window_length = self.eval_features.shape[0]
        stride = max(1, self.eval_cfg.stride)

        metrics: List[WindowMetrics] = []
        trades_path = ensure_dir(self.eval_cfg.report_dir) / "trades.jsonl"
        trade_handle = trades_path.open("w", encoding="utf-8") if self.eval_cfg.store_trades else None

        with torch.inference_mode():
            for start in range(0, self.eval_features.shape[0] - window_length + 1, stride):
                end = start + window_length
                x_window = self.eval_features[start:end].unsqueeze(0)
                r_window = self.eval_returns[start:end]
                alpha = policy(x_window).float()
                intraday_seq, overnight_seq = policy.decode_concentration(alpha)
                window_metrics = self._simulate_window(
                    intraday_seq.squeeze(0),
                    r_window,
                    start,
                    end,
                    trade_handle,
                    overnight=overnight_seq.squeeze(0),
                )
                metrics.append(window_metrics)

        if trade_handle:
            trade_handle.close()

        aggregate = self._aggregate_metrics(metrics)
        report_dir = ensure_dir(self.eval_cfg.report_dir)
        (report_dir / "report.json").write_text(json.dumps(aggregate, indent=2))
        (report_dir / "windows.json").write_text(json.dumps([asdict(m) for m in metrics], indent=2))
        return aggregate

    def _prepare_features(self, add_cash: bool, feature_cfg: Dict | None) -> tuple[torch.Tensor, torch.Tensor]:
        features, returns = ohlc_to_features(self.eval_tensor, add_cash=add_cash)
        cfg = feature_cfg or {}
        features = augment_market_features(
            features,
            returns,
            use_taylor=bool(cfg.get("use_taylor_features", False)),
            taylor_order=int(cfg.get("taylor_order", 0) or 0),
            taylor_scale=float(cfg.get("taylor_scale", 32.0)),
            use_wavelet=bool(cfg.get("use_wavelet_features", False)),
            wavelet_levels=int(cfg.get("wavelet_levels", 0) or 0),
            padding_mode=str(cfg.get("wavelet_padding_mode", "reflect")),
        )
        return (
            features.to(self.device, non_blocking=True),
            returns.to(self.device, non_blocking=True),
        )

    def _simulate_window(
        self,
        intraday: torch.Tensor,
        returns: torch.Tensor,
        start: int,
        end: int,
        trade_handle,
        *,
        overnight: torch.Tensor | None = None,
    ) -> WindowMetrics:
        steps = intraday.shape[0]
        if overnight is None:
            overnight = intraday
        if getattr(self, "_shorting_enabled", False):
            w_prev = torch.zeros((intraday.shape[1],), device=intraday.device, dtype=torch.float32)
        else:
            w_prev = torch.full(
                (intraday.shape[1],),
                1.0 / intraday.shape[1],
                device=intraday.device,
                dtype=torch.float32,
            )
        rewards = []
        turnovers = []
        wealth = []
        gross_history = []
        overnight_history = []
        cumulative = torch.zeros((), dtype=intraday.dtype, device=intraday.device)
        for idx in range(steps):
            w_t = intraday[idx].to(torch.float32)
            r_next = returns[idx]
            reward = self.env.step(w_t, r_next, w_prev)
            rewards.append(reward)
            turnovers.append(smooth_abs(w_t - w_prev, self.env_cfg.smooth_abs_eps).sum())
            cumulative = cumulative + reward
            wealth.append(torch.exp(cumulative))
            gross_history.append(w_t.abs().sum())
            overnight_history.append(overnight[idx].abs().sum())
            if trade_handle is not None:
                timestamp_idx = self.eval_start_idx + start + idx + 1
                if timestamp_idx >= len(self.index):
                    raise IndexError(
                        f"Computed trade timestamp index {timestamp_idx} exceeds available history ({len(self.index)})"
                    )
                entry = {
                    "timestamp": str(self.index[timestamp_idx]),
                    "weights": w_t.tolist(),
                    "reward": reward.item(),
                    "gross_leverage": float(gross_history[-1].item()),
                    "overnight_leverage": float(overnight_history[-1].item()),
                }
                trade_handle.write(json.dumps(entry) + "\n")
            w_prev = overnight[idx].to(torch.float32)

        reward_tensor = torch.stack(rewards)
        turnover_tensor = torch.stack(turnovers)
        objective = self.env.aggregate_rewards(reward_tensor)
        mean_reward = reward_tensor.mean()
        std_reward = reward_tensor.std(unbiased=False).clamp_min(1e-8)
        sharpe = mean_reward / std_reward
        cumulative_return = torch.expm1(reward_tensor.sum()).item()

        wealth_tensor = torch.stack(wealth)
        roll, _ = torch.cummax(wealth_tensor, dim=0)
        drawdown = 1.0 - wealth_tensor / roll.clamp_min(1e-12)
        max_drawdown = float(drawdown.max().item())

        return WindowMetrics(
            start=start,
            end=end,
            objective=float(objective.item()),
            mean_reward=float(mean_reward.item()),
            std_reward=float(std_reward.item()),
            sharpe=float(sharpe.item()),
            turnover=float(turnover_tensor.mean().item()),
            cumulative_return=cumulative_return,
            max_drawdown=max_drawdown,
        )

    def _aggregate_metrics(self, metrics: Sequence[WindowMetrics]) -> Dict[str, float]:
        if not metrics:
            return {}
        mean = lambda key: sum(getattr(m, key) for m in metrics) / len(metrics)
        best_objective = max(metrics, key=lambda m: m.objective).objective
        worst_drawdown = max(metrics, key=lambda m: m.max_drawdown).max_drawdown
        return {
            "windows": len(metrics),
            "objective_mean": mean("objective"),
            "reward_mean": mean("mean_reward"),
            "reward_std": mean("std_reward"),
            "sharpe_mean": mean("sharpe"),
            "turnover_mean": mean("turnover"),
            "cumulative_return_mean": mean("cumulative_return"),
            "max_drawdown_worst": worst_drawdown,
            "objective_best": best_objective,
        }
