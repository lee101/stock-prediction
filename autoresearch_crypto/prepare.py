"""Fixed evaluation harness for crypto RL autoresearch.

DO NOT MODIFY THIS FILE during autoresearch runs. All training experiments
must use this harness for consistent, comparable evaluation.
"""

from __future__ import annotations

import copy
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from binanceneural.config import DatasetConfig, PolicyConfig, TrainingConfig
from binanceneural.data import (
    BASE_FEATURES,
    BinanceHourlyDataModule,
    BinanceHourlyDataset,
    FeatureNormalizer,
    build_default_feature_columns,
)
from binanceneural.model import BinancePolicyBase, build_policy
from differentiable_loss_utils import (
    HOURLY_PERIODS_PER_YEAR,
    compute_loss_by_type,
    simulate_hourly_trades,
    simulate_hourly_trades_binary,
)
from src.robust_trading_metrics import (
    compute_max_drawdown,
    compute_pnl_smoothness,
    compute_return_series,
    summarize_scenario_results,
)

logger = logging.getLogger(__name__)

TIME_BUDGET = int(float(os.getenv("AUTORESEARCH_CRYPTO_TIME_BUDGET_SECONDS", "300")))

DEFAULT_SYMBOLS: Tuple[str, ...] = ("DOGEUSD", "AAVEUSD")
DEFAULT_EVAL_WINDOWS: Tuple[int, ...] = (72, 168, 336, 720)  # 3d, 7d, 14d, 30d
DEFAULT_HOLDOUT_DAYS = 30
DEFAULT_VAL_DAYS = 70
DEFAULT_SEQUENCE_LENGTH = 72
DEFAULT_FORECAST_HORIZONS: Tuple[int, ...] = (1,)


@dataclass(frozen=True)
class CryptoTaskConfig:
    symbols: Tuple[str, ...] = DEFAULT_SYMBOLS
    data_root: Path = Path("trainingdatahourly") / "crypto"
    forecast_cache_root: Path = Path("binanceneural") / "forecast_cache"
    forecast_horizons: Tuple[int, ...] = DEFAULT_FORECAST_HORIZONS
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH
    eval_windows: Tuple[int, ...] = DEFAULT_EVAL_WINDOWS
    holdout_days: int = DEFAULT_HOLDOUT_DAYS
    val_days: int = DEFAULT_VAL_DAYS
    initial_cash: float = 10_000.0
    maker_fee: float = 0.001
    max_leverage: float = 2.0
    margin_annual_rate: float = 0.0625
    fill_buffer_pct: float = 0.0005
    decision_lag_bars: int = 1
    max_hold_hours: int = 6
    batch_size: int = 16
    can_short: bool = False


@dataclass
class ScenarioData:
    name: str
    symbol: str
    window_hours: int
    features: np.ndarray  # (seq_len + window, n_features)
    opens: np.ndarray
    highs: np.ndarray
    lows: np.ndarray
    closes: np.ndarray
    reference_close: np.ndarray
    chronos_high: np.ndarray
    chronos_low: np.ndarray


@dataclass
class PreparedTask:
    config: CryptoTaskConfig
    feature_columns: Tuple[str, ...]
    normalizers: Dict[str, FeatureNormalizer]
    train_modules: Dict[str, BinanceHourlyDataModule]
    scenarios: List[ScenarioData]


def resolve_task_config(**kwargs: Any) -> CryptoTaskConfig:
    fields = {}
    for key in CryptoTaskConfig.__dataclass_fields__:
        if key in kwargs and kwargs[key] is not None:
            val = kwargs[key]
            if key == "symbols" and isinstance(val, str):
                val = tuple(s.strip() for s in val.split(","))
            if key == "eval_windows" and isinstance(val, str):
                val = tuple(int(x) for x in val.split(","))
            if key == "forecast_horizons" and isinstance(val, str):
                val = tuple(int(x) for x in val.split(","))
            if key in ("data_root", "forecast_cache_root"):
                val = Path(val)
            fields[key] = val
    return CryptoTaskConfig(**fields)


def prepare_task(config: CryptoTaskConfig) -> PreparedTask:
    feature_columns = tuple(build_default_feature_columns(config.forecast_horizons))
    train_modules: Dict[str, BinanceHourlyDataModule] = {}
    normalizers: Dict[str, FeatureNormalizer] = {}
    scenarios: List[ScenarioData] = []

    holdout_hours = config.holdout_days * 24
    val_hours = config.val_days * 24
    total_reserved = holdout_hours + val_hours

    for symbol in config.symbols:
        ds_config = DatasetConfig(
            symbol=symbol,
            data_root=config.data_root,
            forecast_cache_root=config.forecast_cache_root,
            forecast_horizons=config.forecast_horizons,
            sequence_length=config.sequence_length,
            validation_days=config.val_days + config.holdout_days,
            cache_only=True,
        )
        module = BinanceHourlyDataModule(ds_config)
        train_modules[symbol] = module
        normalizers[symbol] = module.normalizer

        full_frame = module.frame
        n = len(full_frame)
        holdout_start = n - holdout_hours
        feature_cols = list(feature_columns)
        raw_features = full_frame[feature_cols].to_numpy(dtype=np.float32)
        norm_features = module.normalizer.transform(raw_features)

        primary_h = config.forecast_horizons[0]
        opens_arr = full_frame["open"].to_numpy(dtype=np.float32)
        highs_arr = full_frame["high"].to_numpy(dtype=np.float32)
        lows_arr = full_frame["low"].to_numpy(dtype=np.float32)
        closes_arr = full_frame["close"].to_numpy(dtype=np.float32)
        ref_close_arr = full_frame["reference_close"].to_numpy(dtype=np.float32)
        ch_high_col = f"predicted_high_p50_h{primary_h}"
        ch_low_col = f"predicted_low_p50_h{primary_h}"
        ch_high_arr = full_frame[ch_high_col].to_numpy(dtype=np.float32)
        ch_low_arr = full_frame[ch_low_col].to_numpy(dtype=np.float32)

        for window in config.eval_windows:
            if window > holdout_hours:
                continue
            start = holdout_start - config.sequence_length
            end = holdout_start + window
            if start < 0 or end > n:
                continue
            scenarios.append(ScenarioData(
                name=f"{symbol}_{window}h",
                symbol=symbol,
                window_hours=window,
                features=norm_features[start:end],
                opens=opens_arr[start:end],
                highs=highs_arr[start:end],
                lows=lows_arr[start:end],
                closes=closes_arr[start:end],
                reference_close=ref_close_arr[start:end],
                chronos_high=ch_high_arr[start:end],
                chronos_low=ch_low_arr[start:end],
            ))

    logger.info("prepared %d scenarios for %d symbols", len(scenarios), len(config.symbols))
    return PreparedTask(
        config=config,
        feature_columns=feature_columns,
        normalizers=normalizers,
        train_modules=train_modules,
        scenarios=scenarios,
    )


def _run_scenario(
    model: BinancePolicyBase,
    scenario: ScenarioData,
    config: CryptoTaskConfig,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    seq = config.sequence_length
    total_len = len(scenario.features)
    eval_len = total_len - seq

    features_t = torch.from_numpy(scenario.features).unsqueeze(0).to(device)
    opens_t = torch.from_numpy(scenario.opens).to(device)
    highs_t = torch.from_numpy(scenario.highs).to(device)
    lows_t = torch.from_numpy(scenario.lows).to(device)
    closes_t = torch.from_numpy(scenario.closes).to(device)
    ref_close_t = torch.from_numpy(scenario.reference_close).to(device)
    ch_high_t = torch.from_numpy(scenario.chronos_high).to(device)
    ch_low_t = torch.from_numpy(scenario.chronos_low).to(device)

    buy_prices_list = []
    sell_prices_list = []
    buy_amounts_list = []
    sell_amounts_list = []

    with torch.no_grad():
        for i in range(eval_len):
            feat_window = features_t[:, i:i + seq, :]
            outputs = model(feat_window)
            ref_window = ref_close_t[i:i + seq].unsqueeze(0)
            ch_h_window = ch_high_t[i:i + seq].unsqueeze(0)
            ch_l_window = ch_low_t[i:i + seq].unsqueeze(0)
            actions = model.decode_actions(
                outputs,
                reference_close=ref_window,
                chronos_high=ch_h_window,
                chronos_low=ch_l_window,
            )
            buy_prices_list.append(actions["buy_price"][:, -1])
            sell_prices_list.append(actions["sell_price"][:, -1])
            buy_amounts_list.append(actions["buy_amount"][:, -1])
            sell_amounts_list.append(actions["sell_amount"][:, -1])

    bp = torch.stack(buy_prices_list, dim=-1)
    sp = torch.stack(sell_prices_list, dim=-1)
    ba = torch.stack(buy_amounts_list, dim=-1)
    sa = torch.stack(sell_amounts_list, dim=-1)
    if bp.ndim == 1:
        bp = bp.unsqueeze(0)
        sp = sp.unsqueeze(0)
        ba = ba.unsqueeze(0)
        sa = sa.unsqueeze(0)
    trade_int = torch.maximum(ba, sa)
    scale = float(model.trade_amount_scale)

    sim_highs = highs_t[seq:].unsqueeze(0)
    sim_lows = lows_t[seq:].unsqueeze(0)
    sim_closes = closes_t[seq:].unsqueeze(0)
    sim_opens = opens_t[seq:].unsqueeze(0)

    sim = simulate_hourly_trades_binary(
        highs=sim_highs,
        lows=sim_lows,
        closes=sim_closes,
        opens=sim_opens,
        buy_prices=bp,
        sell_prices=sp,
        trade_intensity=trade_int / scale,
        buy_trade_intensity=ba / scale,
        sell_trade_intensity=sa / scale,
        maker_fee=config.maker_fee,
        initial_cash=config.initial_cash,
        max_leverage=config.max_leverage,
        can_short=config.can_short,
        can_long=True,
        decision_lag_bars=config.decision_lag_bars,
        fill_buffer_pct=config.fill_buffer_pct,
        margin_annual_rate=config.margin_annual_rate,
    )

    equity = sim.portfolio_values.squeeze(0).cpu().numpy()
    if equity.size == 0:
        return {
            "name": scenario.name,
            "symbol": scenario.symbol,
            "window_hours": scenario.window_hours,
            "return_pct": 0.0,
            "sortino": 0.0,
            "max_drawdown_pct": 0.0,
            "pnl_smoothness": 0.0,
            "trade_count": 0,
        }

    initial = float(config.initial_cash)
    final = float(equity[-1])
    total_return_pct = (final - initial) / initial * 100.0

    returns = compute_return_series(equity)
    if len(returns) > 0:
        mean_r = float(np.mean(returns))
        ds = float(np.sqrt(np.mean(np.clip(-returns, 0, None) ** 2) + 1e-12))
        sortino = mean_r / ds * np.sqrt(HOURLY_PERIODS_PER_YEAR)
    else:
        sortino = 0.0

    max_dd = compute_max_drawdown(equity) * 100.0
    smoothness = compute_pnl_smoothness(returns)

    buys = sim.executed_buys.squeeze(0).cpu().numpy()
    sells = sim.executed_sells.squeeze(0).cpu().numpy()
    trade_count = int(np.sum(buys > 0)) + int(np.sum(sells > 0))

    return {
        "name": scenario.name,
        "symbol": scenario.symbol,
        "window_hours": scenario.window_hours,
        "return_pct": total_return_pct,
        "sortino": sortino,
        "max_drawdown_pct": max_dd,
        "pnl_smoothness": smoothness,
        "trade_count": trade_count,
    }


def evaluate_model(
    model: BinancePolicyBase,
    task: PreparedTask,
    *,
    device: torch.device,
) -> Dict[str, Any]:
    model.to(device)
    model.eval()
    scenario_results = []
    for scenario in task.scenarios:
        result = _run_scenario(model, scenario, task.config, device)
        scenario_results.append(result)
        logger.info(
            "%s: ret=%.2f%% sort=%.2f dd=%.2f%% trades=%d",
            result["name"],
            result["return_pct"],
            result["sortino"],
            result["max_drawdown_pct"],
            result["trade_count"],
        )

    if not scenario_results:
        return {"summary": {"robust_score": -999.0, "scenario_count": 0}, "scenarios": []}

    summary = summarize_scenario_results(scenario_results)
    summary["scenario_count"] = float(len(scenario_results))
    total_trades = sum(r["trade_count"] for r in scenario_results)
    summary["total_trade_count"] = float(total_trades)

    return {"summary": summary, "scenarios": scenario_results}


def print_metrics(
    summary: Dict[str, Any],
    *,
    val_loss: float = 0.0,
    val_sortino: float = 0.0,
    val_return_pct: float = 0.0,
    training_seconds: float = 0.0,
    total_seconds: float = 0.0,
    peak_vram_mb: float = 0.0,
    num_steps: int = 0,
    num_epochs: int = 0,
    symbols: str = "",
) -> None:
    print("---")
    print(f"robust_score:      {float(summary.get('robust_score', -999)):.6f}")
    print(f"val_loss:          {val_loss:.6f}")
    print(f"val_sortino:       {val_sortino:.4f}")
    print(f"val_return_pct:    {val_return_pct:.4f}")
    print(f"training_seconds:  {training_seconds:.1f}")
    print(f"total_seconds:     {total_seconds:.1f}")
    print(f"peak_vram_mb:      {peak_vram_mb:.1f}")
    print(f"scenario_count:    {int(summary.get('scenario_count', 0))}")
    print(f"total_trade_count: {int(summary.get('total_trade_count', 0))}")
    print(f"num_steps:         {num_steps}")
    print(f"num_epochs:        {num_epochs}")
    print(f"symbols:           {symbols}")


__all__ = [
    "CryptoTaskConfig",
    "PreparedTask",
    "ScenarioData",
    "TIME_BUDGET",
    "evaluate_model",
    "prepare_task",
    "print_metrics",
    "resolve_task_config",
]
