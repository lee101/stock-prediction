from __future__ import annotations

import argparse
import json
import math
import csv
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
    forecast_cache_root: str | None = None
    forecast_horizons: Tuple[int, ...] = ()
    correlated_symbols: Tuple[str, ...] = ()
    auto_correlated_count: int = 0
    correlation_lookback: int = 24 * 30
    correlation_min_abs: float = 0.25
    validation_interval_timesteps: int = 0


def _read_symbol_frame(root: Path, symbol: str) -> tuple[pd.DataFrame, str]:
    csv_path = root / f"{symbol.upper()}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Unable to find data for symbol '{symbol}' at {csv_path}")
    frame = pd.read_csv(csv_path)
    frame.columns = [str(c).lower() for c in frame.columns]
    required = ["open", "high", "low", "close"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"CSV missing required columns {missing} for symbol {symbol}")

    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        time_col = "timestamp"
    elif "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        frame = frame.rename(columns={"date": "timestamp"})
        time_col = "timestamp"
    else:
        frame["__row_id"] = np.arange(len(frame), dtype=np.int64)
        time_col = "__row_id"

    numeric_cols = [col for col in frame.columns if pd.api.types.is_numeric_dtype(frame[col])]
    keep_cols = []
    for col in ["open", "high", "low", "close", *numeric_cols]:
        if col not in keep_cols and col in frame.columns:
            keep_cols.append(col)
    if time_col not in keep_cols:
        keep_cols = [time_col, *keep_cols]
    return frame[keep_cols].copy(), time_col


def _symbol_name_token(symbol: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in symbol).strip("_")


def _select_correlated_symbols(cfg: TrainingConfig, root: Path) -> tuple[str, ...]:
    explicit: List[str] = [s.strip().upper() for s in cfg.correlated_symbols if s.strip()]
    selected: List[str] = []
    seen = {cfg.symbol.upper()}

    for symbol in explicit:
        if symbol not in seen:
            selected.append(symbol)
            seen.add(symbol)

    if cfg.auto_correlated_count <= 0:
        return tuple(selected)

    target_frame, time_col = _read_symbol_frame(root, cfg.symbol)
    target_close = pd.to_numeric(target_frame["close"], errors="coerce")
    target_returns = target_close.pct_change(fill_method=None)
    if cfg.correlation_lookback > 0 and len(target_returns) > cfg.correlation_lookback:
        target_returns = target_returns.iloc[-cfg.correlation_lookback :]
    target_returns = target_returns.replace([np.inf, -np.inf], np.nan).dropna()
    if target_returns.empty:
        return tuple(selected)

    scored: List[tuple[float, str]] = []
    for csv_path in sorted(root.glob("*.csv")):
        peer = csv_path.stem.upper()
        if peer in seen:
            continue
        try:
            peer_frame, peer_time_col = _read_symbol_frame(root, peer)
        except Exception:
            continue
        peer_close = pd.to_numeric(peer_frame["close"], errors="coerce")
        peer_returns = peer_close.pct_change(fill_method=None)
        if cfg.correlation_lookback > 0 and len(peer_returns) > cfg.correlation_lookback:
            peer_returns = peer_returns.iloc[-cfg.correlation_lookback :]
        peer_returns = peer_returns.replace([np.inf, -np.inf], np.nan)

        if time_col == "timestamp" and peer_time_col == "timestamp":
            left = pd.DataFrame({"timestamp": target_frame["timestamp"], "ret": target_returns})
            right = pd.DataFrame({"timestamp": peer_frame["timestamp"], "peer_ret": peer_returns})
            merged = left.merge(right, on="timestamp", how="inner").dropna()
            if merged.empty:
                continue
            corr = float(merged["ret"].corr(merged["peer_ret"]))
        else:
            min_len = min(len(target_returns), len(peer_returns))
            if min_len < 16:
                continue
            corr = float(np.corrcoef(target_returns.iloc[-min_len:], peer_returns.iloc[-min_len:])[0, 1])

        if not np.isfinite(corr):
            continue
        if abs(corr) < float(cfg.correlation_min_abs):
            continue
        scored.append((abs(corr), peer))

    scored.sort(key=lambda item: item[0], reverse=True)
    for _, peer in scored:
        if len(selected) >= len(explicit) + int(cfg.auto_correlated_count):
            break
        if peer not in seen:
            selected.append(peer)
            seen.add(peer)
    return tuple(selected)


def _merge_forecast_features(
    base: pd.DataFrame,
    *,
    symbol: str,
    time_col: str,
    forecast_cache_root: str | None,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    if forecast_cache_root is None or not horizons or time_col != "timestamp":
        return base

    merged = base.copy()
    cache_root = Path(forecast_cache_root).expanduser().resolve()
    for horizon in horizons:
        parquet_path = cache_root / f"h{int(horizon)}" / f"{symbol.upper()}.parquet"
        if not parquet_path.exists():
            continue
        forecast = pd.read_parquet(parquet_path)
        if "timestamp" not in forecast.columns:
            continue
        forecast["timestamp"] = pd.to_datetime(forecast["timestamp"], utc=True, errors="coerce")
        forecast = forecast.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        cols = {
            "predicted_close_p50": f"chronos_close_h{horizon}",
            "predicted_high_p50": f"chronos_high_h{horizon}",
            "predicted_low_p50": f"chronos_low_h{horizon}",
            "predicted_close_p10": f"chronos_close_p10_h{horizon}",
            "predicted_close_p90": f"chronos_close_p90_h{horizon}",
        }
        available = ["timestamp", *[src for src in cols if src in forecast.columns]]
        if len(available) <= 1:
            continue
        forecast = forecast[available].rename(columns={src: dst for src, dst in cols.items() if src in available})
        merged = merged.merge(forecast, on="timestamp", how="left")

        close = pd.to_numeric(merged["close"], errors="coerce").replace(0.0, np.nan)
        close_key = f"chronos_close_h{horizon}"
        high_key = f"chronos_high_h{horizon}"
        low_key = f"chronos_low_h{horizon}"
        p10_key = f"chronos_close_p10_h{horizon}"
        p90_key = f"chronos_close_p90_h{horizon}"

        if close_key in merged.columns:
            merged[f"chronos_ret_h{horizon}"] = (merged[close_key] - merged["close"]) / close
        if high_key in merged.columns and low_key in merged.columns:
            merged[f"chronos_spread_h{horizon}"] = (merged[high_key] - merged[low_key]) / close
        if p10_key in merged.columns and p90_key in merged.columns:
            merged[f"chronos_conf_h{horizon}"] = (merged[p90_key] - merged[p10_key]) / close

    return merged


def _merge_correlated_symbol_features(
    base: pd.DataFrame,
    *,
    time_col: str,
    root: Path,
    peers: tuple[str, ...],
) -> pd.DataFrame:
    merged = base.copy()
    for peer in peers:
        try:
            peer_frame, peer_time_col = _read_symbol_frame(root, peer)
        except Exception:
            continue
        peer_close = pd.to_numeric(peer_frame["close"], errors="coerce")
        token = _symbol_name_token(peer)
        peer_features = pd.DataFrame(
            {
                peer_time_col: peer_frame[peer_time_col],
                f"peer_{token}_ret1": peer_close.pct_change(fill_method=None),
                f"peer_{token}_ret6": peer_close.pct_change(periods=6, fill_method=None),
            }
        )

        if time_col == "timestamp" and peer_time_col == "timestamp":
            merged = merged.merge(peer_features, on="timestamp", how="left")
        elif time_col == "__row_id" and peer_time_col == "__row_id":
            merged = merged.merge(peer_features, on="__row_id", how="left")
        else:
            # Fall back to right-aligned row-wise merge when timestamps are unavailable.
            min_len = min(len(merged), len(peer_features))
            if min_len <= 0:
                continue
            for col in peer_features.columns:
                if col == peer_time_col:
                    continue
                merged[col] = np.nan
                merged.loc[merged.index[-min_len:], col] = peer_features[col].to_numpy()[-min_len:]
    return merged


def _load_price_tensor(cfg: TrainingConfig) -> Tuple[torch.Tensor, Tuple[str, ...]]:
    root = Path(cfg.data_root).expanduser().resolve()
    frame, time_col = _read_symbol_frame(root, cfg.symbol)
    peers = _select_correlated_symbols(cfg, root)
    frame = _merge_correlated_symbol_features(frame, time_col=time_col, root=root, peers=peers)
    frame = _merge_forecast_features(
        frame,
        symbol=cfg.symbol,
        time_col=time_col,
        forecast_cache_root=cfg.forecast_cache_root,
        horizons=cfg.forecast_horizons,
    )

    if time_col in frame.columns:
        frame = frame.sort_values(time_col).reset_index(drop=True)

    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.ffill().fillna(0.0)

    required = ["open", "high", "low", "close"]
    float_cols = []
    for col in required:
        if col in frame.columns and col not in float_cols:
            float_cols.append(col)
    for col in frame.columns:
        if col in float_cols:
            continue
        if pd.api.types.is_numeric_dtype(frame[col]):
            float_cols.append(col)
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
        "annual_leverage_rate": 0.065,
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

    start = time.time()
    validation_snapshots: list[Dict[str, Any]] = []
    if cfg.validation_interval_timesteps > 0 and cfg.evaluate:
        trained = 0
        interval = max(1, int(cfg.validation_interval_timesteps))
        while trained < cfg.total_timesteps:
            step = min(interval, int(cfg.total_timesteps - trained))
            model.learn(total_timesteps=step, progress_bar=False, reset_num_timesteps=False)
            trained += step
            snap = _evaluate_policy(model, prices, columns, cfg)
            snap["timesteps"] = int(trained)
            snap["train_metrics"] = _extract_train_metrics(model)
            validation_snapshots.append(snap)
        # Copy the last snapshot so attaching the full snapshots list below does
        # not create a self-reference cycle in the JSON summary payload.
        metrics = dict(validation_snapshots[-1]) if validation_snapshots else _empty_metrics(cfg)
    else:
        model.learn(total_timesteps=cfg.total_timesteps, progress_bar=False)
        if cfg.evaluate:
            metrics = _evaluate_policy(model, prices, columns, cfg)
        else:
            metrics = _empty_metrics(cfg)

    elapsed = max(1e-9, time.time() - start)
    train_metrics = _extract_train_metrics(model)
    metrics["training_seconds"] = float(elapsed)
    metrics["train_steps_per_second"] = float(cfg.total_timesteps / elapsed)
    metrics["validation_snapshots"] = validation_snapshots
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
    parser.add_argument(
        "--validation-interval-timesteps",
        type=int,
        default=0,
        help="If >0, train in chunks and run a discrete validation rollout after each chunk.",
    )
    parser.add_argument(
        "--forecast-cache-root",
        type=str,
        default=None,
        help="Optional Chronos2 cache root (expects h{N}/{SYMBOL}.parquet).",
    )
    parser.add_argument(
        "--forecast-horizons",
        type=str,
        default="",
        help="Comma-separated Chronos2 horizons to merge as RL features (e.g. 1,4,24).",
    )
    parser.add_argument(
        "--correlated-symbols",
        type=str,
        default="",
        help="Comma-separated peer symbols to append as correlation features.",
    )
    parser.add_argument(
        "--auto-correlated-count",
        type=int,
        default=0,
        help="Automatically add top-N correlated symbols from the data root.",
    )
    parser.add_argument(
        "--correlation-lookback",
        type=int,
        default=24 * 30,
        help="Lookback bars for automatic correlation ranking.",
    )
    parser.add_argument(
        "--correlation-min-abs",
        type=float,
        default=0.25,
        help="Minimum absolute correlation to include auto-selected peers.",
    )
    args = parser.parse_args()
    forecast_horizons = tuple(int(x.strip()) for x in args.forecast_horizons.split(",") if x.strip())
    correlated_symbols = tuple(str(x).strip().upper() for x in args.correlated_symbols.split(",") if x.strip())
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
        validation_interval_timesteps=max(0, int(args.validation_interval_timesteps)),
        forecast_cache_root=args.forecast_cache_root,
        forecast_horizons=forecast_horizons,
        correlated_symbols=correlated_symbols,
        auto_correlated_count=max(0, int(args.auto_correlated_count)),
        correlation_lookback=max(1, int(args.correlation_lookback)),
        correlation_min_abs=max(0.0, float(args.correlation_min_abs)),
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
        "forecast_horizons": list(cfg.forecast_horizons),
        "correlated_symbols": list(cfg.correlated_symbols),
        "auto_correlated_count": cfg.auto_correlated_count,
        "validation_interval_timesteps": cfg.validation_interval_timesteps,
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
