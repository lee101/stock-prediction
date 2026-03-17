#!/usr/bin/env python3
"""Benchmark HF and PufferLib checkpoints in the same MarketEnv simulator."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import torch

from hftraining.config import ExperimentConfig
from hftraining.data_utils import StockDataProcessor
from hftraining import hf_trainer as hf_trainer_module
from hftraining.hf_trainer import HFTrainingConfig, TransformerTradingModel
from pufferlibtraining3.envs.market_env import MarketEnv, MarketEnvConfig
from pufferlibtraining3.models import MarketPolicy, PolicyConfig


sys.modules.setdefault("hf_trainer", hf_trainer_module)


def _load_market_frame(
    data_root: str | Path,
    symbol: str,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    root = Path(data_root).expanduser()
    if root.is_file():
        candidates = [root]
    else:
        symbol_upper = symbol.upper()
        candidates = sorted(root.glob(f"**/{symbol_upper}.csv"))
        candidates.extend(sorted(root.glob(f"**/{symbol_upper}_*.csv")))
        if not candidates:
            candidates = [p for p in root.glob("**/*.csv") if symbol_upper.lower() in p.stem.lower()]
    if not candidates:
        raise FileNotFoundError(f"No CSV found for symbol '{symbol}' under '{root}'")

    frame = pd.read_csv(candidates[0]).copy()
    frame.columns = [str(col).lower() for col in frame.columns]

    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.sort_values("date").set_index("date")
    elif "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        frame = frame.sort_values("timestamp").set_index("timestamp")
    else:
        raise ValueError(f"CSV '{candidates[0]}' is missing a date/timestamp column")

    frame = frame.dropna(axis=0, how="any")
    if start_date is not None:
        frame = frame[frame.index >= pd.to_datetime(start_date)]
    if end_date is not None:
        frame = frame[frame.index <= pd.to_datetime(end_date)]
    if frame.empty:
        raise ValueError(f"No rows left for symbol '{symbol}' after date filtering")
    return frame


def _market_tensors_from_frame(
    frame: pd.DataFrame,
) -> tuple[torch.Tensor, Optional[torch.Tensor], tuple[str, ...]]:
    required = ["open", "high", "low", "close"]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing required price columns: {missing}")

    price_cols = ["open", "high", "low", "close"]
    if "volume" in frame.columns:
        price_cols.append("volume")

    prices = torch.from_numpy(frame[price_cols].to_numpy(dtype=np.float32))
    exog_cols = [
        col
        for col in frame.columns
        if col not in price_cols and pd.api.types.is_numeric_dtype(frame[col])
    ]
    exog = None
    if exog_cols:
        exog = torch.from_numpy(frame[exog_cols].to_numpy(dtype=np.float32))
    return prices, exog, tuple(price_cols)


def _estimate_periods_per_year(index: pd.Index, *, is_crypto: bool) -> float:
    if index is None:
        return 365.25 if is_crypto else 252.0
    if len(index) < 3 or not isinstance(index, pd.DatetimeIndex):
        return 365.25 if is_crypto else 252.0

    deltas = index.to_series().diff().dropna()
    if deltas.empty:
        return 365.25 if is_crypto else 252.0

    median_seconds = float(deltas.dt.total_seconds().median())
    if median_seconds <= 0:
        return 365.25 if is_crypto else 252.0

    if is_crypto:
        return (365.25 * 24.0 * 3600.0) / median_seconds

    one_day = 24.0 * 3600.0
    two_hours = 2.0 * 3600.0
    thirty_minutes = 30.0 * 60.0
    if median_seconds <= thirty_minutes:
        return 252.0 * 78.0
    if median_seconds <= two_hours:
        return 252.0 * 6.5
    if median_seconds <= one_day * 1.5:
        return 252.0
    return (365.25 * 24.0 * 3600.0) / median_seconds


def _annualize_total_return(total_return: float, periods: int, periods_per_year: float) -> float:
    if periods <= 0 or periods_per_year <= 0 or total_return <= -1.0:
        return 0.0
    growth = 1.0 + float(total_return)
    return float(growth ** (periods_per_year / float(periods)) - 1.0)


def _sharpe_ratio(step_returns: list[float], periods_per_year: float) -> float:
    if len(step_returns) < 2:
        return 0.0
    arr = np.asarray(step_returns, dtype=np.float64)
    denom = float(arr.std(ddof=0))
    if denom <= 1e-12:
        return 0.0
    return float(arr.mean() / denom * math.sqrt(periods_per_year))


def _sortino_ratio(step_returns: list[float], periods_per_year: float) -> float:
    if len(step_returns) < 2:
        return 0.0
    arr = np.asarray(step_returns, dtype=np.float64)
    downside = arr[arr < 0.0]
    if downside.size == 0:
        return 0.0
    downside_dev = float(np.sqrt(np.mean(np.square(downside))))
    if downside_dev <= 1e-12:
        return 0.0
    return float(arr.mean() / downside_dev * math.sqrt(periods_per_year))


def _max_drawdown(equity_curve: list[float]) -> float:
    if not equity_curve:
        return 0.0
    peak = float(equity_curve[0])
    worst = 0.0
    for equity in equity_curve:
        peak = max(peak, float(equity))
        if peak <= 0:
            continue
        drawdown = float(equity) / peak - 1.0
        worst = min(worst, drawdown)
    return float(worst)


def _coerce_hf_config(raw: Any) -> HFTrainingConfig:
    if isinstance(raw, HFTrainingConfig):
        return raw
    if isinstance(raw, dict):
        return HFTrainingConfig(**raw)
    raise TypeError(f"Unsupported HF config payload type: {type(raw)!r}")


@dataclass
class BenchmarkResult:
    framework: str
    symbol: str
    mode: str
    checkpoint: str
    start_date: str | None
    end_date: str | None
    periods_per_year: float
    steps: int
    final_equity: float
    total_return: float
    annualized_return: float
    sharpe: float
    sortino: float
    max_drawdown: float
    num_trade_steps: int
    signal_steps: int
    filled_trade_steps: int
    fill_rate: float
    total_trading_cost: float
    total_financing_cost: float
    total_deleverage_notional: float
    mean_abs_position: float
    turnover: float
    extra: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _run_market_env(
    env: MarketEnv,
    action_fn: Callable[[np.ndarray, MarketEnv], np.ndarray | int | float],
    *,
    checkpoint: str,
    framework: str,
    symbol: str,
    start_date: str | None,
    end_date: str | None,
    extra: Optional[dict[str, Any]] = None,
) -> BenchmarkResult:
    obs, _ = env.reset()
    done = False

    step_returns: list[float] = []
    equity_curve: list[float] = [float(env.equity.detach().cpu().item())]
    positions: list[float] = []

    total_trading_cost = 0.0
    total_financing_cost = 0.0
    total_deleverage_notional = 0.0
    num_trade_steps = 0
    filled_trade_steps = 0
    signal_steps = 0
    turnover = 0.0
    prev_position = 0.0

    while not done:
        current_position = float(env.position.detach().cpu().item())
        action = action_fn(obs, env)

        if env.cfg.mode.lower() == "maxdiff":
            action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
            if action_arr.size > 0 and abs(float(action_arr[0])) >= float(env.cfg.maxdiff_deadband):
                signal_steps += 1
        else:
            action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
            if action_arr.size > 0 and abs(float(action_arr[0])) > 1e-6:
                signal_steps += 1

        equity_before = float(env.equity.detach().cpu().item())
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)

        trading_cost = float(info.get("trading_cost", 0.0))
        financing_cost = float(info.get("financing_cost", 0.0))
        deleverage = float(info.get("deleverage_notional", 0.0))
        position = float(info.get("overnight_position", info.get("position", 0.0)))

        if trading_cost > 1e-12:
            num_trade_steps += 1
        if bool(info.get("maxdiff_filled", False)):
            filled_trade_steps += 1
        elif env.cfg.mode.lower() != "maxdiff" and trading_cost > 1e-12:
            filled_trade_steps += 1

        total_trading_cost += trading_cost
        total_financing_cost += financing_cost
        total_deleverage_notional += deleverage
        turnover += abs(position - prev_position)
        prev_position = position
        positions.append(abs(position))

        if equity_before > 1e-12:
            step_returns.append(float(reward) / equity_before)
        equity_curve.append(float(info.get("equity", equity_before + float(reward))))

    periods_per_year = _estimate_periods_per_year(env.date_index, is_crypto=bool(env.cfg.is_crypto))
    total_return = equity_curve[-1] - 1.0
    fill_rate = float(filled_trade_steps / signal_steps) if signal_steps > 0 else 0.0
    return BenchmarkResult(
        framework=framework,
        symbol=symbol,
        mode=env.cfg.mode,
        checkpoint=str(checkpoint),
        start_date=start_date,
        end_date=end_date,
        periods_per_year=float(periods_per_year),
        steps=len(step_returns),
        final_equity=float(equity_curve[-1]),
        total_return=float(total_return),
        annualized_return=_annualize_total_return(total_return, len(step_returns), periods_per_year),
        sharpe=_sharpe_ratio(step_returns, periods_per_year),
        sortino=_sortino_ratio(step_returns, periods_per_year),
        max_drawdown=_max_drawdown(equity_curve),
        num_trade_steps=int(num_trade_steps),
        signal_steps=int(signal_steps),
        filled_trade_steps=int(filled_trade_steps),
        fill_rate=float(fill_rate),
        total_trading_cost=float(total_trading_cost),
        total_financing_cost=float(total_financing_cost),
        total_deleverage_notional=float(total_deleverage_notional),
        mean_abs_position=float(np.mean(positions) if positions else 0.0),
        turnover=float(turnover),
        extra=extra or {},
    )


class HFEvaluator:
    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        processor_path: str | Path | None = None,
        config_json: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint_path)
        payload = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.config = _coerce_hf_config(payload["config"])
        state_dict = payload.get("model_state_dict", payload)
        inferred_input_dim = 0
        weight = state_dict.get("input_projection.weight") if isinstance(state_dict, dict) else None
        if weight is not None and hasattr(weight, "shape") and len(weight.shape) == 2:
            inferred_input_dim = int(weight.shape[1])
        self.input_dim = int(payload.get("input_dim") or inferred_input_dim or 0)
        if not self.input_dim:
            raise ValueError(f"HF checkpoint '{self.checkpoint_path}' is missing input_dim")

        self.model = TransformerTradingModel(self.config, input_dim=self.input_dim).to(self.device)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

        config_path = Path(config_json) if config_json else (self.checkpoint_path.parent / "config.json")
        self.experiment_config: Optional[ExperimentConfig] = None
        if config_path.exists():
            self.experiment_config = ExperimentConfig.load(str(config_path))

        use_toto_forecasts = bool(
            getattr(self.experiment_config.data, "use_toto_forecasts", False) if self.experiment_config else False
        )
        processor = StockDataProcessor(
            sequence_length=int(self.config.sequence_length),
            prediction_horizon=int(self.config.prediction_horizon),
            use_toto_forecasts=use_toto_forecasts,
        )
        resolved_processor = Path(processor_path) if processor_path else (self.checkpoint_path.parent / "data_processor.pkl")
        processor.load_scalers(str(resolved_processor))
        self.processor = processor

    def prepare_features(self, frame: pd.DataFrame, *, symbol: str) -> np.ndarray:
        prepared = frame.reset_index().rename(columns={frame.index.name or "index": "date"})
        features = self.processor.prepare_features(prepared, symbol=symbol)
        return self.processor.transform(features).astype(np.float32, copy=False)

    def action_fn(
        self,
        feature_matrix: np.ndarray,
        *,
        mode: str,
        action_mode: str,
    ) -> Callable[[np.ndarray, MarketEnv], np.ndarray]:
        seq_len = int(self.config.sequence_length)

        def _action(obs: np.ndarray, env: MarketEnv) -> np.ndarray:
            del obs
            end_idx = int(env.cursor)
            start_idx = end_idx - seq_len
            if start_idx < 0:
                raise ValueError("Environment cursor is earlier than HF sequence length")
            window = feature_matrix[start_idx:end_idx]
            inputs = torch.from_numpy(window).unsqueeze(0).to(self.device)
            attention_mask = torch.ones((1, seq_len), dtype=torch.float32, device=self.device)
            with torch.no_grad():
                outputs = self.model(inputs, attention_mask=attention_mask)
            allocation_raw = outputs["allocations"].reshape(-1)
            allocation = float(allocation_raw[0].detach().cpu().item()) if allocation_raw.numel() else 0.0
            action_label = int(torch.argmax(outputs["action_logits"], dim=-1).item())
            magnitude = min(1.0, abs(allocation))

            if action_mode == "alloc_only":
                signed = allocation
            elif action_mode == "alloc_gated":
                signed = 0.0 if action_label == 1 else allocation
            elif action_mode == "alloc_signed_by_logits":
                if action_label == 1:
                    signed = 0.0
                elif action_label == 0:
                    signed = magnitude
                else:
                    signed = -magnitude
            elif action_mode == "logits_only":
                signed = 1.0 if action_label == 0 else (-1.0 if action_label == 2 else 0.0)
            else:
                raise ValueError(f"Unsupported HF action mode '{action_mode}'")

            signed = float(max(-1.0, min(1.0, signed)))
            if mode.lower() == "maxdiff":
                return np.asarray([signed, min(1.0, abs(signed))], dtype=np.float32)
            return np.asarray([signed], dtype=np.float32)

        return _action


class PufferEvaluator:
    def __init__(
        self,
        *,
        summary_json: str | Path,
        checkpoint_path: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.summary_path = Path(summary_json)
        self.summary = json.loads(self.summary_path.read_text())
        self.env_config = MarketEnvConfig(**self.summary["env_config"])
        self.policy_config = PolicyConfig(**self.summary["policy_config"])
        resolved_checkpoint = checkpoint_path or self.summary.get("model_path")
        if not resolved_checkpoint:
            raise ValueError("Puffer summary is missing model_path and no checkpoint override was supplied")
        self.checkpoint_path = Path(resolved_checkpoint)

    def build_policy(self, env: MarketEnv) -> MarketPolicy:
        policy = MarketPolicy(env, self.policy_config).to(self.device)
        state_dict = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        policy.load_state_dict(state_dict, strict=False)
        policy.eval()
        return policy

    def action_fn(self, policy: MarketPolicy) -> Callable[[np.ndarray, MarketEnv], np.ndarray | int]:
        def _action(obs: np.ndarray, env: MarketEnv) -> np.ndarray | int:
            tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                logits, _ = policy.forward_eval(tensor)
            if isinstance(logits, torch.distributions.Distribution):
                return logits.mean[0].detach().cpu().numpy()
            if isinstance(logits, tuple):
                return int(torch.argmax(logits[0], dim=-1).item())
            if logits.ndim == 2 and logits.shape[-1] > 1:
                return int(torch.argmax(logits, dim=-1).item())
            return logits[0].detach().cpu().numpy()

        return _action


def _env_from_frame(
    frame: pd.DataFrame,
    *,
    context_len: int,
    mode: str,
    data_root: str | Path,
    symbol: str,
    start_date: str | None,
    end_date: str | None,
    trading_fee: float,
    crypto_fee: float,
    slip_bps: float,
    intraday_leverage: float,
    overnight_leverage: float,
    annual_leverage_rate: float,
    inv_penalty: float,
    is_crypto: bool,
) -> MarketEnv:
    prices, exog, price_columns = _market_tensors_from_frame(frame)
    cfg = MarketEnvConfig(
        context_len=int(context_len),
        mode=mode,
        data_root=str(data_root),
        symbol=symbol,
        trading_fee=float(trading_fee),
        crypto_trading_fee=float(crypto_fee),
        slip_bps=float(slip_bps),
        intraday_leverage_max=float(intraday_leverage),
        overnight_leverage_max=float(overnight_leverage),
        annual_leverage_rate=float(annual_leverage_rate),
        inv_penalty=float(inv_penalty),
        action_space="continuous",
        reward_scale=1.0,
        is_crypto=bool(is_crypto),
        device="cpu",
        start_date=start_date,
        end_date=end_date,
        random_reset=False,
    )
    env = MarketEnv(prices=prices, exog=exog, price_columns=price_columns, cfg=cfg)
    # Mirror the attributes PufferLib vector wrappers expose so standalone evals
    # can instantiate policies against the raw MarketEnv directly.
    env.date_index = frame.index.copy()
    env.single_observation_space = env.observation_space
    env.single_action_space = env.action_space
    return env


def evaluate_hf_checkpoint(
    *,
    checkpoint_path: str | Path,
    symbol: str,
    data_root: str | Path,
    mode: str,
    start_date: str | None = None,
    end_date: str | None = None,
    processor_path: str | Path | None = None,
    config_json: str | Path | None = None,
    action_mode: str = "alloc_only",
    device: str = "cpu",
    trading_fee: float = 0.0005,
    crypto_fee: float = 0.0015,
    slip_bps: float = 5.0,
    intraday_leverage: float = 4.0,
    overnight_leverage: float = 2.0,
    annual_leverage_rate: float = 0.065,
    inv_penalty: float = 0.0,
    is_crypto: bool = False,
) -> BenchmarkResult:
    evaluator = HFEvaluator(
        checkpoint_path=checkpoint_path,
        processor_path=processor_path,
        config_json=config_json,
        device=device,
    )
    frame = _load_market_frame(data_root, symbol, start_date=start_date, end_date=end_date)
    feature_matrix = evaluator.prepare_features(frame, symbol=symbol)
    env = _env_from_frame(
        frame,
        context_len=int(evaluator.config.sequence_length),
        mode=mode,
        data_root=data_root,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        trading_fee=trading_fee,
        crypto_fee=crypto_fee,
        slip_bps=slip_bps,
        intraday_leverage=intraday_leverage,
        overnight_leverage=overnight_leverage,
        annual_leverage_rate=annual_leverage_rate,
        inv_penalty=inv_penalty,
        is_crypto=is_crypto,
    )
    return _run_market_env(
        env,
        evaluator.action_fn(feature_matrix, mode=mode, action_mode=action_mode),
        checkpoint=str(checkpoint_path),
        framework="hf",
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        extra={
            "action_mode": action_mode,
            "sequence_length": int(evaluator.config.sequence_length),
            "prediction_horizon": int(evaluator.config.prediction_horizon),
        },
    )


def evaluate_puffer_checkpoint(
    *,
    summary_json: str | Path,
    symbol: str,
    data_root: str | Path,
    mode: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    checkpoint_path: str | Path | None = None,
    device: str = "cpu",
    trading_fee: float | None = None,
    crypto_fee: float | None = None,
    slip_bps: float | None = None,
    intraday_leverage: float | None = None,
    overnight_leverage: float | None = None,
    annual_leverage_rate: float | None = None,
    inv_penalty: float | None = None,
    is_crypto: bool | None = None,
) -> BenchmarkResult:
    evaluator = PufferEvaluator(summary_json=summary_json, checkpoint_path=checkpoint_path, device=device)
    frame = _load_market_frame(data_root, symbol, start_date=start_date, end_date=end_date)
    env_mode = mode or evaluator.env_config.mode
    env = _env_from_frame(
        frame,
        context_len=int(evaluator.env_config.context_len),
        mode=env_mode,
        data_root=data_root,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        trading_fee=float(trading_fee if trading_fee is not None else evaluator.env_config.trading_fee),
        crypto_fee=float(crypto_fee if crypto_fee is not None else evaluator.env_config.crypto_trading_fee),
        slip_bps=float(slip_bps if slip_bps is not None else evaluator.env_config.slip_bps),
        intraday_leverage=float(
            intraday_leverage if intraday_leverage is not None else evaluator.env_config.intraday_leverage_max
        ),
        overnight_leverage=float(
            overnight_leverage if overnight_leverage is not None else evaluator.env_config.overnight_leverage_max
        ),
        annual_leverage_rate=float(
            annual_leverage_rate if annual_leverage_rate is not None else evaluator.env_config.annual_leverage_rate
        ),
        inv_penalty=float(inv_penalty if inv_penalty is not None else evaluator.env_config.inv_penalty),
        is_crypto=bool(is_crypto if is_crypto is not None else evaluator.env_config.is_crypto),
    )
    policy = evaluator.build_policy(env)
    return _run_market_env(
        env,
        evaluator.action_fn(policy),
        checkpoint=str(evaluator.checkpoint_path),
        framework="pufferlib",
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        extra={
            "summary_json": str(summary_json),
            "context_len": int(evaluator.env_config.context_len),
            "model_preset_hidden_size": int(evaluator.policy_config.hidden_size),
        },
    )


def _write_json(payload: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark HF vs PufferLib checkpoints in MarketEnv")
    subparsers = parser.add_subparsers(dest="command", required=True)

    hf_parser = subparsers.add_parser("evaluate-hf", help="Evaluate one HF checkpoint")
    hf_parser.add_argument("--checkpoint", required=True)
    hf_parser.add_argument("--symbol", required=True)
    hf_parser.add_argument("--data-root", default="trainingdata")
    hf_parser.add_argument("--mode", default="open_close", choices=["open_close", "event", "maxdiff"])
    hf_parser.add_argument("--start-date", default=None)
    hf_parser.add_argument("--end-date", default=None)
    hf_parser.add_argument("--processor-path", default=None)
    hf_parser.add_argument("--config-json", default=None)
    hf_parser.add_argument(
        "--hf-action-mode",
        default="alloc_only",
        choices=["alloc_only", "alloc_gated", "alloc_signed_by_logits", "logits_only"],
    )
    hf_parser.add_argument("--device", default="cpu")
    hf_parser.add_argument("--trading-fee", type=float, default=0.0005)
    hf_parser.add_argument("--crypto-fee", type=float, default=0.0015)
    hf_parser.add_argument("--slip-bps", type=float, default=5.0)
    hf_parser.add_argument("--intraday-leverage", type=float, default=4.0)
    hf_parser.add_argument("--overnight-leverage", type=float, default=2.0)
    hf_parser.add_argument("--annual-leverage-rate", type=float, default=0.065)
    hf_parser.add_argument("--inv-penalty", type=float, default=0.0)
    hf_parser.add_argument("--is-crypto", action="store_true")
    hf_parser.add_argument("--output-json", default=None)

    puff_parser = subparsers.add_parser("evaluate-puffer", help="Evaluate one Puffer checkpoint")
    puff_parser.add_argument("--summary-json", required=True)
    puff_parser.add_argument("--checkpoint", default=None)
    puff_parser.add_argument("--symbol", required=True)
    puff_parser.add_argument("--data-root", default="trainingdata")
    puff_parser.add_argument("--mode", default=None, choices=["open_close", "event", "maxdiff"])
    puff_parser.add_argument("--start-date", default=None)
    puff_parser.add_argument("--end-date", default=None)
    puff_parser.add_argument("--device", default="cpu")
    puff_parser.add_argument("--trading-fee", type=float, default=None)
    puff_parser.add_argument("--crypto-fee", type=float, default=None)
    puff_parser.add_argument("--slip-bps", type=float, default=None)
    puff_parser.add_argument("--intraday-leverage", type=float, default=None)
    puff_parser.add_argument("--overnight-leverage", type=float, default=None)
    puff_parser.add_argument("--annual-leverage-rate", type=float, default=None)
    puff_parser.add_argument("--inv-penalty", type=float, default=None)
    puff_parser.add_argument("--is-crypto", action="store_true")
    puff_parser.add_argument("--output-json", default=None)

    compare_parser = subparsers.add_parser("compare", help="Evaluate both checkpoints under shared settings")
    compare_parser.add_argument("--symbol", required=True)
    compare_parser.add_argument("--data-root", default="trainingdata")
    compare_parser.add_argument("--mode", default="open_close", choices=["open_close", "event", "maxdiff"])
    compare_parser.add_argument("--start-date", default=None)
    compare_parser.add_argument("--end-date", default=None)
    compare_parser.add_argument("--device", default="cpu")
    compare_parser.add_argument("--trading-fee", type=float, default=0.0005)
    compare_parser.add_argument("--crypto-fee", type=float, default=0.0015)
    compare_parser.add_argument("--slip-bps", type=float, default=5.0)
    compare_parser.add_argument("--intraday-leverage", type=float, default=4.0)
    compare_parser.add_argument("--overnight-leverage", type=float, default=2.0)
    compare_parser.add_argument("--annual-leverage-rate", type=float, default=0.065)
    compare_parser.add_argument("--inv-penalty", type=float, default=0.0)
    compare_parser.add_argument("--is-crypto", action="store_true")
    compare_parser.add_argument("--hf-checkpoint", required=True)
    compare_parser.add_argument("--hf-processor-path", default=None)
    compare_parser.add_argument("--hf-config-json", default=None)
    compare_parser.add_argument(
        "--hf-action-mode",
        default="alloc_only",
        choices=["alloc_only", "alloc_gated", "alloc_signed_by_logits", "logits_only"],
    )
    compare_parser.add_argument("--puffer-summary-json", required=True)
    compare_parser.add_argument("--puffer-checkpoint", default=None)
    compare_parser.add_argument("--output-json", default=None)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "evaluate-hf":
        result = evaluate_hf_checkpoint(
            checkpoint_path=args.checkpoint,
            symbol=args.symbol,
            data_root=args.data_root,
            mode=args.mode,
            start_date=args.start_date,
            end_date=args.end_date,
            processor_path=args.processor_path,
            config_json=args.config_json,
            action_mode=args.hf_action_mode,
            device=args.device,
            trading_fee=args.trading_fee,
            crypto_fee=args.crypto_fee,
            slip_bps=args.slip_bps,
            intraday_leverage=args.intraday_leverage,
            overnight_leverage=args.overnight_leverage,
            annual_leverage_rate=args.annual_leverage_rate,
            inv_penalty=args.inv_penalty,
            is_crypto=args.is_crypto,
        )
        payload = result.to_dict()
    elif args.command == "evaluate-puffer":
        result = evaluate_puffer_checkpoint(
            summary_json=args.summary_json,
            checkpoint_path=args.checkpoint,
            symbol=args.symbol,
            data_root=args.data_root,
            mode=args.mode,
            start_date=args.start_date,
            end_date=args.end_date,
            device=args.device,
            trading_fee=args.trading_fee,
            crypto_fee=args.crypto_fee,
            slip_bps=args.slip_bps,
            intraday_leverage=args.intraday_leverage,
            overnight_leverage=args.overnight_leverage,
            annual_leverage_rate=args.annual_leverage_rate,
            inv_penalty=args.inv_penalty,
            is_crypto=args.is_crypto,
        )
        payload = result.to_dict()
    else:
        hf_result = evaluate_hf_checkpoint(
            checkpoint_path=args.hf_checkpoint,
            symbol=args.symbol,
            data_root=args.data_root,
            mode=args.mode,
            start_date=args.start_date,
            end_date=args.end_date,
            processor_path=args.hf_processor_path,
            config_json=args.hf_config_json,
            action_mode=args.hf_action_mode,
            device=args.device,
            trading_fee=args.trading_fee,
            crypto_fee=args.crypto_fee,
            slip_bps=args.slip_bps,
            intraday_leverage=args.intraday_leverage,
            overnight_leverage=args.overnight_leverage,
            annual_leverage_rate=args.annual_leverage_rate,
            inv_penalty=args.inv_penalty,
            is_crypto=args.is_crypto,
        )
        puffer_result = evaluate_puffer_checkpoint(
            summary_json=args.puffer_summary_json,
            checkpoint_path=args.puffer_checkpoint,
            symbol=args.symbol,
            data_root=args.data_root,
            mode=args.mode,
            start_date=args.start_date,
            end_date=args.end_date,
            device=args.device,
            trading_fee=args.trading_fee,
            crypto_fee=args.crypto_fee,
            slip_bps=args.slip_bps,
            intraday_leverage=args.intraday_leverage,
            overnight_leverage=args.overnight_leverage,
            annual_leverage_rate=args.annual_leverage_rate,
            inv_penalty=args.inv_penalty,
            is_crypto=args.is_crypto,
        )
        winner = "hf" if hf_result.sortino >= puffer_result.sortino else "pufferlib"
        payload = {
            "symbol": args.symbol,
            "mode": args.mode,
            "winner_by_sortino": winner,
            "hf": hf_result.to_dict(),
            "pufferlib": puffer_result.to_dict(),
            "delta": {
                "total_return": hf_result.total_return - puffer_result.total_return,
                "annualized_return": hf_result.annualized_return - puffer_result.annualized_return,
                "sortino": hf_result.sortino - puffer_result.sortino,
                "max_drawdown": hf_result.max_drawdown - puffer_result.max_drawdown,
                "fill_rate": hf_result.fill_rate - puffer_result.fill_rate,
            },
        }

    print(json.dumps(payload, indent=2, sort_keys=True))
    output_json = getattr(args, "output_json", None)
    if output_json:
        _write_json(payload, output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
