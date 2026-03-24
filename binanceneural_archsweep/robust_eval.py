"""Robust multi-scenario market simulator evaluation.

Runs each checkpoint through multiple validation scenarios:
- Different time periods (recent 30d, 60d, 90d)
- Different fee structures (0bps, 10bps, 15bps maker)
- Different fill slippage/buffer (0bps, 5bps, 10bps)
- Different decision lags (0, 1, 2 bars)
- Different initial cash levels (test scale invariance)
- Different intensity scales

Returns a composite robustness score.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from binanceneural.config import DatasetConfig, TrainingConfig
from binanceneural.data import BinanceHourlyDataModule
from binanceneural.inference import generate_actions_from_frame
from binanceneural.marketsimulator import BinanceMarketSimulator, SimulationConfig
from binanceneural.model import BinancePolicyBase, align_state_dict_input_dim, build_policy, policy_config_from_payload
from binanceneural.sweep import apply_action_overrides
from src.torch_load_utils import torch_load_compat

logger = logging.getLogger(__name__)

FEE_SCENARIOS = [0.0, 0.001, 0.0015]
FILL_BUFFER_BPS = [0.0, 5.0, 10.0]
DECISION_LAGS = [0, 1, 2]
INTENSITY_SCALES = [0.6, 0.8, 1.0, 1.2, 1.4]
INITIAL_CASH_LEVELS = [1000.0, 10000.0, 50000.0]
VAL_PERIOD_DAYS = [30, 60, 90]


@dataclass
class ScenarioResult:
    fee_rate: float
    fill_buffer_bps: float
    decision_lag: int
    intensity_scale: float
    initial_cash: float
    val_days: int
    total_return: float
    sortino: float
    max_drawdown: float
    num_trades: int
    win_rate: float


@dataclass
class RobustEvalResult:
    config_name: str
    symbol: str
    checkpoint_path: str
    scenarios: list[ScenarioResult] = field(default_factory=list)
    robust_score: float = 0.0
    mean_return: float = 0.0
    mean_sortino: float = 0.0
    worst_return: float = 0.0
    worst_sortino: float = 0.0
    p25_return: float = 0.0
    p25_sortino: float = 0.0
    profitable_pct: float = 0.0
    best_intensity: float = 1.0
    eval_seconds: float = 0.0


def load_model_from_checkpoint(
    checkpoint_path: Path,
    input_dim: int,
) -> BinancePolicyBase:
    payload = torch_load_compat(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict", payload)
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
    state_dict = align_state_dict_input_dim(state_dict, input_dim=input_dim)
    cfg = payload.get("config", {})
    if hasattr(cfg, "__dict__"):
        cfg = cfg.__dict__
    policy_cfg = policy_config_from_payload(cfg, input_dim=input_dim, state_dict=state_dict)
    model = build_policy(policy_cfg)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def robust_evaluate(
    config_name: str,
    checkpoint_path: Path,
    symbol: str = "BTCUSD",
    sequence_length: int = 96,
    horizon: int = 1,
    fee_scenarios: list[float] | None = None,
    fill_buffers: list[float] | None = None,
    lags: list[int] | None = None,
    intensities: list[float] | None = None,
    cash_levels: list[float] | None = None,
    val_periods: list[int] | None = None,
) -> RobustEvalResult:
    t0 = time.time()
    fee_scenarios = fee_scenarios or FEE_SCENARIOS
    fill_buffers = fill_buffers or FILL_BUFFER_BPS
    lags = lags or DECISION_LAGS
    intensities = intensities or INTENSITY_SCALES
    cash_levels = cash_levels or [10000.0]
    val_periods = val_periods or VAL_PERIOD_DAYS

    data_cfg = DatasetConfig(
        symbol=symbol,
        sequence_length=sequence_length,
        cache_only=True,
        forecast_horizons=(1, 24),
    )
    data = BinanceHourlyDataModule(data_cfg)

    model = load_model_from_checkpoint(checkpoint_path, len(data.feature_columns))

    val_frame = data.val_dataset.frame
    if val_frame is None or len(val_frame) == 0:
        logger.warning("No val data for %s", symbol)
        return RobustEvalResult(config_name=config_name, symbol=symbol, checkpoint_path=str(checkpoint_path))

    base_actions = generate_actions_from_frame(
        model=model,
        frame=val_frame,
        feature_columns=data.feature_columns,
        normalizer=data.normalizer,
        sequence_length=sequence_length,
        horizon=horizon,
    )

    total_val_hours = len(val_frame)
    scenarios = []

    for val_days in val_periods:
        val_hours = val_days * 24
        if val_hours > total_val_hours:
            val_hours = total_val_hours
        period_frame = val_frame.iloc[-val_hours:].copy()
        period_actions = base_actions.iloc[-val_hours:].copy()

        for fee in fee_scenarios:
            for buf_bps in fill_buffers:
                for lag in lags:
                    for intensity in intensities:
                        for cash in cash_levels:
                            adj_actions = apply_action_overrides(
                                period_actions,
                                intensity_scale=intensity,
                                price_offset_pct=0.0,
                            )
                            sim_cfg = SimulationConfig(
                                maker_fee=fee,
                                initial_cash=cash,
                                fill_buffer_bps=buf_bps,
                                decision_lag_bars=lag,
                            )
                            sim = BinanceMarketSimulator(sim_cfg)
                            try:
                                result = sim.run(period_frame, adj_actions)
                                metrics = result.metrics
                                trades = []
                                for sr in result.per_symbol.values():
                                    trades.extend(sr.trades)
                                n_trades = len(trades)
                                wins = sum(1 for t in trades if t.realized_pnl > 0)
                                wr = wins / n_trades if n_trades > 0 else 0.0
                            except Exception as e:
                                logger.warning("Sim failed: %s", e)
                                metrics = {"total_return": 0.0, "sortino": 0.0, "max_drawdown": 0.0}
                                n_trades = 0
                                wr = 0.0

                            scenarios.append(ScenarioResult(
                                fee_rate=fee,
                                fill_buffer_bps=buf_bps,
                                decision_lag=lag,
                                intensity_scale=intensity,
                                initial_cash=cash,
                                val_days=val_days,
                                total_return=float(metrics.get("total_return", 0.0)),
                                sortino=float(metrics.get("sortino", 0.0)),
                                max_drawdown=float(metrics.get("max_drawdown", 0.0)),
                                num_trades=n_trades,
                                win_rate=wr,
                            ))

    returns = np.array([s.total_return for s in scenarios])
    sortinos = np.array([s.sortino for s in scenarios])
    profitable = np.sum(returns > 0) / max(len(returns), 1)

    # Find best intensity (highest mean return across realistic scenarios)
    realistic = [s for s in scenarios if s.fee_rate >= 0.001 and s.fill_buffer_bps >= 5.0 and s.decision_lag >= 1]
    if realistic:
        by_intensity = {}
        for s in realistic:
            by_intensity.setdefault(s.intensity_scale, []).append(s.total_return)
        best_int = max(by_intensity, key=lambda k: np.mean(by_intensity[k]))
    else:
        best_int = 1.0

    # Composite robust score:
    # 40% mean return, 20% worst-case return, 20% mean sortino, 10% profitable%, 10% p25 return
    mean_ret = float(np.mean(returns)) if len(returns) > 0 else 0.0
    worst_ret = float(np.min(returns)) if len(returns) > 0 else 0.0
    p25_ret = float(np.percentile(returns, 25)) if len(returns) > 0 else 0.0
    mean_sort = float(np.mean(sortinos)) if len(sortinos) > 0 else 0.0
    worst_sort = float(np.min(sortinos)) if len(sortinos) > 0 else 0.0
    p25_sort = float(np.percentile(sortinos, 25)) if len(sortinos) > 0 else 0.0

    robust = (
        0.35 * mean_ret
        + 0.15 * worst_ret
        + 0.20 * (mean_sort / 100.0)
        + 0.10 * p25_ret
        + 0.10 * profitable
        + 0.10 * (p25_sort / 100.0)
    )

    elapsed = time.time() - t0
    return RobustEvalResult(
        config_name=config_name,
        symbol=symbol,
        checkpoint_path=str(checkpoint_path),
        scenarios=scenarios,
        robust_score=robust,
        mean_return=mean_ret,
        mean_sortino=mean_sort,
        worst_return=worst_ret,
        worst_sortino=worst_sort,
        p25_return=p25_ret,
        p25_sortino=p25_sort,
        profitable_pct=profitable * 100,
        best_intensity=best_int,
        eval_seconds=elapsed,
    )


def evaluate_multi_symbol(
    config_name: str,
    checkpoint_path: Path,
    symbols: list[str],
    **kwargs,
) -> dict[str, RobustEvalResult]:
    results = {}
    for sym in symbols:
        try:
            r = robust_evaluate(config_name, checkpoint_path, symbol=sym, **kwargs)
            results[sym] = r
            logger.info("%s/%s: robust=%.4f mean_ret=%.4f mean_sortino=%.2f profitable=%.1f%%",
                       config_name, sym, r.robust_score, r.mean_return, r.mean_sortino, r.profitable_pct)
        except Exception as e:
            logger.error("Failed eval %s/%s: %s", config_name, sym, e)
    return results
