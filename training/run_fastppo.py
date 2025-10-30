from __future__ import annotations

import argparse
import json
import math
import csv
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

try:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting optional
    plt = None

import numpy as np
import pandas as pd
import torch
from gymnasium import ObservationWrapper, spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from fastmarketsim import FastMarketEnv
from pufferlibtraining3.envs.market_env import MarketEnv, MarketEnvConfig


@dataclass
class TrainingConfig:
    symbol: str = "AAPL"
    data_root: str = "trainingdata"
    context_len: int = 128
    horizon: int = 1
    total_timesteps: int = 32_768
    learning_rate: float = 3e-4
    gamma: float = 0.995
    num_envs: int = 4
    seed: int = 1337
    device: str = "cpu"
    log_json: str | None = None
    env_backend: str = "fast"
    plot: bool = False
    plot_path: str | None = None
    html_report: bool = False
    html_path: str | None = None
    sma_window: int = 32
    ema_window: int = 32
    downsample: int = 1
    evaluate: bool = True
    history_csv: str | None = None
    max_plot_points: int = 0


def _load_price_tensor(cfg: TrainingConfig) -> Tuple[torch.Tensor, Tuple[str, ...]]:
    root = Path(cfg.data_root).expanduser().resolve()
    csv_path = root / f"{cfg.symbol.upper()}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Unable to find data for symbol '{cfg.symbol}' at {csv_path}")
    frame = pd.read_csv(csv_path)
    frame.columns = [str(c).lower() for c in frame.columns]
    required = ["open", "high", "low", "close"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"CSV missing required columns {missing} for symbol {cfg.symbol}")
    float_cols = [
        col for col in frame.columns if col in required or pd.api.types.is_numeric_dtype(frame[col])
    ]
    values = frame[float_cols].to_numpy(dtype=np.float32)
    return torch.from_numpy(values).contiguous(), tuple(float_cols)


class FlattenObservation(ObservationWrapper):
    def __init__(self, env: FastMarketEnv):
        super().__init__(env)
        original = env.observation_space
        size = int(np.prod(original.shape))
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(size,),
            dtype=np.float32,
        )

    def observation(self, observation):
        return observation.reshape(-1)


def _make_env(prices: torch.Tensor, columns: Tuple[str, ...], base_cfg: TrainingConfig):
    cfg_dict: Dict[str, Any] = {
        "context_len": base_cfg.context_len,
        "horizon": base_cfg.horizon,
        "intraday_leverage_max": 4.0,
        "overnight_leverage_max": 2.0,
        "annual_leverage_rate": 0.0675,
        "trading_fee": 0.0005,
        "crypto_trading_fee": 0.0015,
        "slip_bps": 1.5,
        "is_crypto": False,
        "seed": base_cfg.seed,
    }
    backend = base_cfg.env_backend.lower()
    if backend == "fast":
        env = FastMarketEnv(prices=prices, cfg=cfg_dict, device=base_cfg.device)
    elif backend == "python":
        market_cfg = MarketEnvConfig(**cfg_dict)
        env = MarketEnv(prices=prices, price_columns=columns, cfg=market_cfg)
    else:
        raise ValueError(f"Unsupported env backend '{base_cfg.env_backend}'.")
    return env


def _dummy_env_factory(prices: torch.Tensor, columns: Tuple[str, ...], base_cfg: TrainingConfig):
    def _factory():
        env = _make_env(prices, columns, base_cfg)
        return FlattenObservation(env)

    return _factory


def _evaluate_policy(model: PPO, prices: torch.Tensor, columns: Tuple[str, ...], cfg: TrainingConfig) -> Dict[str, Any]:
    env = FlattenObservation(_make_env(prices, columns, cfg))
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    gross = 0.0
    trading = 0.0
    financing = 0.0
    deleverage_cost = 0.0
    steps = 0
    reward_trace: list[float] = []
    equity_trace: list[float] = []
    gross_trace: list[float] = []

    while not done and steps < (prices.shape[0] - cfg.context_len - 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        total_reward += float(reward)
        reward_trace.append(float(reward))
        gross += float(info.get("gross_pnl", 0.0))
        gross_trace.append(float(info.get("gross_pnl", 0.0)))
        trading += float(info.get("trading_cost", 0.0))
        financing += float(info.get("financing_cost", 0.0))
        deleverage_cost += float(info.get("deleverage_cost", 0.0))
        if "equity" in info:
            equity_trace.append(float(info["equity"]))
        steps += 1

    return {
        "total_reward": total_reward,
        "gross_pnl": gross,
        "trading_cost": trading,
        "financing_cost": financing,
        "deleverage_cost": deleverage_cost,
        "steps": float(steps),
        "reward_trace": reward_trace,
        "gross_trace": gross_trace,
        "equity_trace": equity_trace,
        "reward_stats": _reward_stats(reward_trace, cfg.sma_window, cfg.ema_window),
    }


def run_training(cfg: TrainingConfig) -> Tuple[PPO, Dict[str, Any]]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    prices, columns = _load_price_tensor(cfg)
    if prices.shape[0] <= cfg.context_len + cfg.horizon + 1:
        raise ValueError("Not enough timesteps in price data to satisfy context length.")

    env_fns = [_dummy_env_factory(prices, columns, cfg) for _ in range(cfg.num_envs)]
    vec_env = DummyVecEnv(env_fns)

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.context_len,
        batch_size=cfg.context_len,
        n_epochs=4,
        gamma=cfg.gamma,
        ent_coef=0.005,
        verbose=1,
        seed=cfg.seed,
        device=cfg.device,
    )
    model.learn(total_timesteps=cfg.total_timesteps, progress_bar=False)
    train_metrics = _extract_train_metrics(model)
    if cfg.evaluate:
        metrics = _evaluate_policy(model, prices, columns, cfg)
    else:
        metrics = _empty_metrics(cfg)
    metrics["train_metrics"] = train_metrics
    vec_env.close()
    return model, metrics


def _reward_stats(trace: list[float], sma_window: int, ema_window: int) -> Dict[str, Any]:
    if not trace:
        return {"mean": 0.0, "stdev": 0.0, "sma": 0.0, "ema": 0.0}
    arr = np.asarray(trace, dtype=np.float32)
    mean = float(arr.mean())
    stdev = float(arr.std())
    window = min(max(1, sma_window), arr.size)
    sma = float(arr[-window:].mean()) if window > 0 else mean
    ema_len = min(max(1, ema_window), arr.size)
    alpha = 2.0 / (ema_len + 1.0)
    ema = float(arr[0])
    for value in arr[1:]:
        ema = alpha * value + (1 - alpha) * ema
    return {"mean": mean, "stdev": stdev, "sma": sma, "ema": ema}


def _empty_metrics(cfg: TrainingConfig) -> Dict[str, Any]:
    return {
        "total_reward": 0.0,
        "gross_pnl": 0.0,
        "trading_cost": 0.0,
        "financing_cost": 0.0,
        "deleverage_cost": 0.0,
        "steps": 0.0,
        "reward_trace": [],
        "gross_trace": [],
        "equity_trace": [],
        "reward_stats": _reward_stats([], cfg.sma_window, cfg.ema_window),
        "train_metrics": {},
    }


def _json_default(obj: Any):
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.ndarray,)):
        return [float(x) for x in obj.tolist()]
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _extract_train_metrics(model: PPO) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    log_values = getattr(model.logger, "name_to_value", {})
    for key, value in log_values.items():
        if not key.startswith("train/"):
            continue
        clean_key = key.split("/", 1)[1]
        try:
            metrics[clean_key] = float(value)
        except (TypeError, ValueError):
            continue
    return metrics


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train PPO on the fast market simulator.")
    parser.add_argument("--symbol", type=str, default="AAPL")
    parser.add_argument("--data-root", type=str, default="trainingdata")
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--total-timesteps", type=int, default=32_768)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log-json", type=str, default=None)
    parser.add_argument("--env-backend", type=str, default="fast", choices=["fast", "python"], help="Select environment implementation")
    parser.add_argument("--plot", action="store_true", help="Generate reward/gross/equity trace plots (requires matplotlib)")
    parser.add_argument("--plot-path", type=str, default=None, help="Directory to store plots (defaults to log-json directory or ./results)")
    parser.add_argument("--sma-window", type=int, default=32, help="Window length for reward smoothing SMA")
    parser.add_argument("--ema-window", type=int, default=32, help="Window length for reward smoothing EMA")
    parser.add_argument("--downsample", type=int, default=1, help="Keep every Nth trace sample when plotting/HTML export")
    parser.add_argument("--max-plot-points", type=int, default=0, help="Auto-adjust downsampling to keep plots under this many points (0 disables)")
    parser.add_argument("--html-report", action="store_true", help="Generate an HTML report combining summary stats and the trace plot")
    parser.add_argument("--html-path", type=str, default=None, help="File path for the HTML report (defaults beside log-json)")
    parser.add_argument("--history-csv", type=str, default=None, help="Append run metrics to the specified CSV path")
    parser.add_argument("--no-eval", action="store_true", help="Skip post-training evaluation pass to save time")
    args = parser.parse_args()
    return TrainingConfig(
        symbol=args.symbol,
        data_root=args.data_root,
        context_len=args.context_len,
        horizon=args.horizon,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        num_envs=args.num_envs,
        seed=args.seed,
        device=args.device,
        log_json=args.log_json,
        env_backend=args.env_backend,
        plot=args.plot,
        plot_path=args.plot_path,
        sma_window=max(1, args.sma_window),
        ema_window=max(1, args.ema_window),
        downsample=max(1, args.downsample),
        max_plot_points=max(0, args.max_plot_points),
        html_report=args.html_report,
        html_path=args.html_path,
        evaluate=not args.no_eval,
        history_csv=args.history_csv,
    )


def main() -> None:
    cfg = parse_args()
    model, metrics = run_training(cfg)
    summary = {
        **metrics,
        "symbol": cfg.symbol.upper(),
        "total_timesteps": cfg.total_timesteps,
        "learning_rate": cfg.learning_rate,
        "gamma": cfg.gamma,
        "context_len": cfg.context_len,
        "horizon": cfg.horizon,
        "reward_stats": metrics.get("reward_stats", {}),
        "evaluation_skipped": not cfg.evaluate,
    }
    # reward_trace contains per-step rewards from the evaluation rollout.
    # reward_stats adds aggregate mean/stdev plus configurable SMA/EMA helpers.
    # train_metrics captures final PPO training-loop diagnostics (KL, losses, etc.).
    # equity_trace is populated when the environment reports equity in info dicts.
    # html_report writes a self-contained summary linking the PNG trace (if generated).
    # downsample allows keeping every Nth point when plotting/reporting to shrink large traces.
    # history_csv appends key metrics to a rolling CSV for long-term tracking.
    if cfg.log_json:
        path = Path(cfg.log_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, default=_json_default))
        print(f"[fastppo] wrote summary to {path}")
    else:
        print(json.dumps(summary, indent=2, default=_json_default))

    if cfg.history_csv:
        _append_history(Path(cfg.history_csv), summary)

    plot_path: Path | None = None
    if not cfg.evaluate:
        if cfg.plot:
            print("[fastppo] plot requested but evaluation skipped; nothing to plot.")
        if cfg.html_report:
            print("[fastppo] HTML report requested but evaluation skipped; nothing to report.")
        _ = model
        return

    ds = max(1, cfg.downsample)
    reward_raw = np.asarray(metrics["reward_trace"], dtype=np.float32)
    gross_raw = np.asarray(metrics["gross_trace"], dtype=np.float32)
    equity_raw = np.asarray(metrics["equity_trace"], dtype=np.float32) if metrics["equity_trace"] else np.array([])
    if cfg.max_plot_points and len(reward_raw) > cfg.max_plot_points:
        auto = int(np.ceil(len(reward_raw) / cfg.max_plot_points))
        ds = max(ds, auto)
    reward_trace = reward_raw[::ds]
    gross_trace = gross_raw[::ds]
    equity_trace = equity_raw[::ds] if equity_raw.size else []
    steps = np.arange(len(reward_trace)) * ds

    if cfg.plot:
        target_dir = Path(cfg.plot_path or (Path(cfg.log_json).parent if cfg.log_json else Path("results")))
        target_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        ax[0].plot(steps, reward_trace, label="Reward")
        stats = metrics.get("reward_stats", {})
        if stats and reward_trace.size > 1:
            plot_sma_window = min(len(reward_trace), max(1, int(np.ceil(cfg.sma_window / ds))))
            if plot_sma_window > 1:
                sma = np.convolve(reward_trace, np.ones(plot_sma_window) / plot_sma_window, mode="valid")
                ax[0].plot(steps[plot_sma_window - 1 :], sma, label=f"SMA({cfg.sma_window})", color="tab:orange")
            plot_ema_window = min(len(reward_trace), max(1, int(np.ceil(cfg.ema_window / ds))))
            if plot_ema_window > 1:
                alpha = 2.0 / (plot_ema_window + 1.0)
                ema_curve = np.empty_like(reward_trace)
                ema_curve[0] = reward_trace[0]
                for idx in range(1, len(reward_trace)):
                    ema_curve[idx] = alpha * reward_trace[idx] + (1 - alpha) * ema_curve[idx - 1]
                ax[0].plot(steps, ema_curve, label=f"EMA({cfg.ema_window})", color="tab:red", alpha=0.7)
        ax[0].set_ylabel("Reward")
        ax[0].grid(True, alpha=0.3)
        ax[0].legend()

        ax[1].plot(steps, gross_trace, label="Gross PnL", color="tab:orange")
        ax[1].set_ylabel("Gross PnL")
        ax[1].grid(True, alpha=0.3)
        ax[1].legend()

        if len(equity_trace):
            ax[2].plot(steps, equity_trace, label="Equity", color="tab:green")
            ax[2].set_ylabel("Equity")
        else:
            ax[2].plot(steps, np.cumsum(reward_trace), label="Cumulative Reward", color="tab:green")
            ax[2].set_ylabel("Cumulative Reward")
        ax[2].set_xlabel("Step")
        ax[2].grid(True, alpha=0.3)
        ax[2].legend()

        fig.tight_layout()
        plot_path = target_dir / f"{cfg.symbol.lower()}_fastppo_trace.png"
        fig.savefig(plot_path)
        plt.close(fig)
        print(f"[fastppo] wrote trace plot to {plot_path}")

    history_rows: list[dict[str, str]] = []
    if cfg.history_csv:
        csv_path = Path(cfg.history_csv)
        if csv_path.exists():
            with csv_path.open() as fh:
                reader = csv.DictReader(fh)
                history_rows = [row for row in reader][-5:]

    if cfg.html_report:
        report_path = Path(cfg.html_path or (Path(cfg.log_json).with_suffix(".html") if cfg.log_json else Path("results") / f"{cfg.symbol.lower()}_fastppo_report.html"))
        report_path.parent.mkdir(parents=True, exist_ok=True)
        plot_rel = plot_path.name if plot_path else None
        reward_stats = summary.get("reward_stats", {})
        html = [
            "<html><head><meta charset='utf-8'><title>FastPPO Trace Report</title>",
            "<style>body{font-family:Arial,sans-serif;margin:2rem;} table{border-collapse:collapse;} td,th{padding:0.4rem 0.8rem;border:1px solid #ccc;} h2{margin-top:2rem;}</style>",
            "</head><body>",
            f"<h1>FastPPO Summary – {cfg.symbol.upper()}</h1>",
            "<table>",
            "<tr><th>Metric</th><th>Value</th></tr>",
            f"<tr><td>Total Reward</td><td>{summary['total_reward']:.6f}</td></tr>",
            f"<tr><td>Gross PnL</td><td>{summary['gross_pnl']:.6f}</td></tr>",
            f"<tr><td>Trading Cost</td><td>{summary['trading_cost']:.6f}</td></tr>",
            f"<tr><td>Financing Cost</td><td>{summary['financing_cost']:.6f}</td></tr>",
            f"<tr><td>Steps</td><td>{summary['steps']:.0f}</td></tr>",
            "</table>",
        ]
        if reward_stats:
            html.extend([
                "<h2>Reward Statistics</h2>",
                "<table>",
                "<tr><th>Metric</th><th>Value</th></tr>",
                f"<tr><td>Mean</td><td>{reward_stats['mean']:.6e}</td></tr>",
                f"<tr><td>Std Dev</td><td>{reward_stats['stdev']:.6e}</td></tr>",
                f"<tr><td>SMA({cfg.sma_window})</td><td>{reward_stats['sma']:.6e}</td></tr>",
                f"<tr><td>EMA({cfg.ema_window})</td><td>{reward_stats['ema']:.6e}</td></tr>",
                "</table>",
            ])
        if history_rows:
            html.extend([
                "<h2>Recent Run History</h2>",
                "<table>",
                "<tr><th>Timestamp</th><th>Reward</th><th>Gross PnL</th><th>Train Loss</th><th>Approx KL</th></tr>",
            ])
            for row in history_rows:
                html.append(
                    f"<tr><td>{row.get('timestamp','')}</td><td>{row.get('reward','')}</td><td>{row.get('gross_pnl','')}</td><td>{row.get('train_loss','')}</td><td>{row.get('train_approx_kl','')}</td></tr>"
                )
            html.append("</table>")
        if plot_rel:
            html.extend([
                "<h2>Reward / PnL Trace</h2>",
                f"<img src='{plot_rel}' alt='trace plot' style='max-width:100%;height:auto;'>",
            ])
        html.append("</body></html>")
        report_path.write_text("\n".join(html))
        print(f"[fastppo] wrote HTML report to {report_path}")
    # Prevent linter from pruning the model variable prematurely during potential extensions.
    _ = model


def _append_history(csv_path: Path, summary: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    reward_stats = summary.get("reward_stats", {})
    train_metrics = summary.get("train_metrics", {})
    row = {
        "timestamp": summary.get("timestamp") or datetime.now(timezone.utc).isoformat(),
        "symbol": summary.get("symbol"),
        "total_timesteps": summary.get("total_timesteps"),
        "learning_rate": summary.get("learning_rate"),
        "gamma": summary.get("gamma"),
        "reward": summary.get("total_reward"),
        "gross_pnl": summary.get("gross_pnl"),
        "trading_cost": summary.get("trading_cost"),
        "steps": summary.get("steps"),
        "reward_mean": reward_stats.get("mean"),
        "reward_stdev": reward_stats.get("stdev"),
        "reward_sma": reward_stats.get("sma"),
        "reward_ema": reward_stats.get("ema"),
        "train_loss": train_metrics.get("loss"),
        "train_entropy": train_metrics.get("entropy_loss"),
        "train_value_loss": train_metrics.get("value_loss"),
        "train_policy_loss": train_metrics.get("policy_gradient_loss"),
        "train_approx_kl": train_metrics.get("approx_kl"),
        "train_clip_fraction": train_metrics.get("clip_fraction"),
        "train_explained_variance": train_metrics.get("explained_variance"),
    }

    existing_rows: list[dict[str, str]] = []
    existing_header: list[str] | None = None
    if csv_path.exists():
        with csv_path.open() as fh:
            reader = csv.DictReader(fh)
            existing_rows = list(reader)
            existing_header = reader.fieldnames

    fieldnames = list(row.keys())
    if existing_header and existing_header != fieldnames:
        for prev in existing_rows:
            for key in fieldnames:
                prev.setdefault(key, "")
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(existing_rows)

    with csv_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if not existing_rows and not existing_header:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()
