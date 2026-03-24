#!/usr/bin/env python3
"""Sweep per-symbol meta strategy selection over multiple stock strategies.

Each strategy is a checkpoint (or checkpoint directory). We generate actions for
every strategy, then build a meta action stream that picks one strategy per
symbol/day using trailing daily performance (previous-day winner by default).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
from loguru import logger

from binanceneural.config import DatasetConfig
from binanceneural.data import BinanceHourlyDataModule, FeatureNormalizer
from binanceneural.inference import generate_actions_from_frame
from binanceneural.model import build_policy, policy_config_from_payload
from src.hourly_trader_utils import normalize_entry_allocator_mode
from src.trade_directions import DEFAULT_ALPACA_LIVE8_STOCKS
from src.torch_load_utils import torch_load_compat
from unified_hourly_experiment.marketsimulator import PortfolioConfig, run_portfolio_simulation
from unified_hourly_experiment.meta_selector import (
    combine_actions_by_winners,
    daily_returns_from_equity,
    select_daily_winners,
)


@dataclass
class StrategyModel:
    name: str
    checkpoint_dir: Path
    checkpoint_name: str
    model: torch.nn.Module
    feature_columns: list[str]
    sequence_length: int
    normalizer: FeatureNormalizer | None
    horizons: list[int]


@dataclass(frozen=True)
class ExecutionScenario:
    label: str
    bar_margin: float
    entry_order_ttl_hours: int


def parse_csv_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in parse_csv_list(value)]


def parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in parse_csv_list(value)]


def _float_token(value: float) -> str:
    return str(float(value)).replace(".", "p")


def resolve_execution_scenarios(
    *,
    base_cfg: PortfolioConfig,
    bar_margins: Sequence[float] | None = None,
    entry_order_ttls: Sequence[int] | None = None,
) -> list[ExecutionScenario]:
    margin_values = list(bar_margins) if bar_margins else [float(base_cfg.bar_margin)]
    ttl_values = list(entry_order_ttls) if entry_order_ttls else [int(base_cfg.entry_order_ttl_hours)]

    unique_margins: list[float] = []
    seen_margins: set[float] = set()
    for raw in margin_values:
        value = float(raw)
        if value < 0:
            raise ValueError(f"execution bar margins must all be >= 0, got {margin_values}")
        if value not in seen_margins:
            seen_margins.add(value)
            unique_margins.append(value)

    unique_ttls: list[int] = []
    seen_ttls: set[int] = set()
    for raw in ttl_values:
        value = int(raw)
        if value < 0:
            raise ValueError(f"execution entry order TTLs must all be >= 0, got {ttl_values}")
        if value not in seen_ttls:
            seen_ttls.add(value)
            unique_ttls.append(value)

    scenarios: list[ExecutionScenario] = []
    for bar_margin in unique_margins:
        for ttl_hours in unique_ttls:
            scenarios.append(
                ExecutionScenario(
                    label=f"bm{_float_token(bar_margin)}_ttl{ttl_hours}",
                    bar_margin=float(bar_margin),
                    entry_order_ttl_hours=int(ttl_hours),
                )
            )
    return scenarios


def parse_sit_out_threshold_values(
    *,
    sit_out_if_negative: bool,
    sit_out_threshold: float,
    sit_out_thresholds: str,
) -> list[float | None]:
    """Resolve one or more sit-out thresholds, preserving legacy behavior."""
    if not sit_out_if_negative:
        return [None]
    raw = str(sit_out_thresholds).strip()
    if raw:
        values = parse_float_list(raw)
        if not values:
            raise ValueError("--sit-out-thresholds provided but no values parsed.")
        return [float(v) for v in values]
    return [float(sit_out_threshold)]


def parse_strategy_spec(spec: str) -> tuple[str, Path, int | None]:
    if "=" not in spec:
        raise ValueError(f"Invalid --strategy spec '{spec}'. Expected NAME=PATH.")
    name, path_spec = spec.split("=", 1)
    name = name.strip()
    path_spec = path_spec.strip()
    if not name:
        raise ValueError(f"Invalid --strategy spec '{spec}': empty strategy name.")

    epoch_override = None
    path_part = path_spec
    if ":" in path_spec:
        left, right = path_spec.rsplit(":", 1)
        if right.isdigit():
            path_part = left
            epoch_override = int(right)

    path = Path(path_part)
    if path.is_file():
        if path.suffix != ".pt":
            raise ValueError(f"Strategy path '{path}' is a file but not a .pt checkpoint.")
        checkpoint_dir = path.parent
        ckpt_name = path.stem
        if ckpt_name.startswith("epoch_"):
            inferred_epoch = int(ckpt_name.split("_")[1])
            if epoch_override is not None and epoch_override != inferred_epoch:
                raise ValueError(
                    f"Conflicting epoch for strategy '{name}': path has {inferred_epoch}, override has {epoch_override}."
                )
            epoch_override = inferred_epoch
        else:
            raise ValueError(f"Checkpoint file '{path}' is expected to be named epoch_XXX.pt.")
        return name, checkpoint_dir, epoch_override

    if not path.is_dir():
        raise ValueError(f"Strategy path does not exist: {path}")
    return name, path, epoch_override


def load_strategy_model(
    name: str,
    checkpoint_dir: Path,
    *,
    epoch: int | None,
    device: torch.device,
) -> StrategyModel:
    if epoch is not None:
        ckpt_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
        if not ckpt_path.exists():
            raise ValueError(f"Strategy '{name}': checkpoint not found: {ckpt_path}")
    else:
        checkpoints = sorted(checkpoint_dir.glob("epoch_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
        if not checkpoints:
            raise ValueError(f"Strategy '{name}': no epoch checkpoints found in {checkpoint_dir}")
        ckpt_path = checkpoints[-1]

    ckpt = torch_load_compat(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    config_path = checkpoint_dir / "config.json"
    meta_path = checkpoint_dir / "training_meta.json"
    if meta_path.exists():
        config = json.loads(meta_path.read_text())
    elif config_path.exists():
        config = json.loads(config_path.read_text())
    else:
        config = ckpt.get("config", {})

    feature_columns = config.get("feature_columns") or []
    if not feature_columns:
        raise ValueError(f"Strategy '{name}': missing feature_columns in config for {checkpoint_dir}")

    policy_cfg = policy_config_from_payload(config, input_dim=len(feature_columns), state_dict=state_dict)
    model = build_policy(policy_cfg)
    try:
        model.load_state_dict(state_dict, strict=False)
    except RuntimeError as exc:
        fallback_payload = _infer_policy_overrides_from_state_dict(config, state_dict)
        policy_cfg = policy_config_from_payload(
            fallback_payload,
            input_dim=len(feature_columns),
            state_dict=state_dict,
        )
        model = build_policy(policy_cfg)
        model.load_state_dict(state_dict, strict=False)
        logger.warning(
            "Strategy '{}' config mismatch detected; loaded with inferred dims from checkpoint ({})",
            name,
            exc,
        )
    model.eval().to(device)

    normalizer = None
    if config_path.exists():
        cfg = json.loads(config_path.read_text())
        if "normalizer" in cfg:
            normalizer = FeatureNormalizer.from_dict(cfg["normalizer"])

    horizons = sorted(
        {
            int(col.split("_h")[1])
            for col in feature_columns
            if "_h" in col and col.split("_h")[1].isdigit()
        }
    )
    if not horizons:
        horizons = [1]

    return StrategyModel(
        name=name,
        checkpoint_dir=checkpoint_dir,
        checkpoint_name=ckpt_path.name,
        model=model,
        feature_columns=feature_columns,
        sequence_length=int(config.get("sequence_length", 32)),
        normalizer=normalizer,
        horizons=horizons,
    )


def _infer_policy_overrides_from_state_dict(
    payload: dict,
    state_dict: dict,
) -> dict:
    inferred = dict(payload)

    embed_w = state_dict.get("embed.weight")
    if isinstance(embed_w, torch.Tensor) and embed_w.ndim == 2:
        hidden_dim = int(embed_w.shape[0])
        inferred["transformer_dim"] = hidden_dim
        inferred["hidden_dim"] = hidden_dim

    head_w = state_dict.get("head.weight")
    if isinstance(head_w, torch.Tensor) and head_w.ndim == 2:
        inferred["num_outputs"] = int(head_w.shape[0])

    pe = state_dict.get("pos_encoding.pe")
    if isinstance(pe, torch.Tensor) and pe.ndim >= 2:
        inferred["max_len"] = int(pe.shape[0])
        inferred["sequence_length"] = int(pe.shape[0])

    encoder_layer_indices = _extract_layer_indices(state_dict, prefix="encoder.layers.")
    if encoder_layer_indices:
        inferred["model_arch"] = "classic"
        inferred["transformer_layers"] = int(max(encoder_layer_indices) + 1)
        inferred["num_layers"] = int(max(encoder_layer_indices) + 1)

    block_indices = _extract_layer_indices(state_dict, prefix="blocks.")
    if block_indices:
        inferred["model_arch"] = "nano"
        inferred["transformer_layers"] = int(max(block_indices) + 1)
        inferred["num_layers"] = int(max(block_indices) + 1)

    return inferred


def _extract_layer_indices(state_dict: dict, prefix: str) -> list[int]:
    found = set()
    for key in state_dict:
        if not isinstance(key, str) or not key.startswith(prefix):
            continue
        tail = key[len(prefix) :]
        idx_str = tail.split(".", 1)[0]
        if idx_str.isdigit():
            found.add(int(idx_str))
    return sorted(found)


def build_data_cache(
    symbols: Sequence[str],
    strategies: Sequence[StrategyModel],
    *,
    data_root: Path,
    cache_root: Path,
) -> dict[tuple[str, tuple[int, ...], int], BinanceHourlyDataModule]:
    cache: dict[tuple[str, tuple[int, ...], int], BinanceHourlyDataModule] = {}
    for strategy in strategies:
        horizons_key = tuple(strategy.horizons)
        for symbol in symbols:
            key = (symbol, horizons_key, strategy.sequence_length)
            if key in cache:
                continue
            cfg = DatasetConfig(
                symbol=symbol,
                data_root=str(data_root),
                forecast_cache_root=str(cache_root),
                forecast_horizons=list(horizons_key),
                sequence_length=int(strategy.sequence_length),
                min_history_hours=100,
                validation_days=30,
                cache_only=True,
            )
            cache[key] = BinanceHourlyDataModule(cfg)
    return cache


def _file_signature(path: Path) -> dict[str, int | str | None]:
    if not path.exists():
        return {"path": str(path), "size": None, "mtime_ns": None}
    stat = path.stat()
    return {"path": str(path.resolve()), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def build_strategy_action_cache_key(
    *,
    strategy: StrategyModel,
    symbols: Sequence[str],
    data_cache: dict[tuple[str, tuple[int, ...], int], BinanceHourlyDataModule],
) -> str:
    horizons_key = tuple(strategy.horizons)
    checkpoint_path = strategy.checkpoint_dir / strategy.checkpoint_name
    config_path = strategy.checkpoint_dir / "config.json"
    meta_path = strategy.checkpoint_dir / "training_meta.json"

    data_signature = []
    for symbol in symbols:
        dm = data_cache[(symbol, horizons_key, strategy.sequence_length)]
        frame = dm.frame
        timestamps = pd.to_datetime(frame["timestamp"], utc=True)
        data_signature.append(
            {
                "symbol": str(symbol),
                "rows": int(len(frame)),
                "columns": [str(col) for col in frame.columns],
                "first_timestamp": timestamps.iloc[0].isoformat() if not timestamps.empty else None,
                "last_timestamp": timestamps.iloc[-1].isoformat() if not timestamps.empty else None,
            }
        )

    payload = {
        "checkpoint": _file_signature(checkpoint_path),
        "config": _file_signature(config_path),
        "meta": _file_signature(meta_path),
        "feature_columns": [str(col) for col in strategy.feature_columns],
        "sequence_length": int(strategy.sequence_length),
        "horizons": [int(h) for h in strategy.horizons],
        "symbols": [str(symbol) for symbol in symbols],
        "data_signature": data_signature,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def load_or_generate_strategy_frames(
    *,
    strategy: StrategyModel,
    symbols: Sequence[str],
    data_cache: dict[tuple[str, tuple[int, ...], int], BinanceHourlyDataModule],
    device: torch.device,
    action_cache_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_path: Path | None = None
    if action_cache_dir is not None:
        action_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = build_strategy_action_cache_key(strategy=strategy, symbols=symbols, data_cache=data_cache)
        cache_path = action_cache_dir / f"strategy_actions_{cache_key}.pkl"
        if cache_path.exists():
            cached = pd.read_pickle(cache_path)
            logger.info("Loaded action cache for {} -> {}", strategy.name, cache_path)
            return cached["actions"], cached["bars"]

    action_parts = []
    bar_parts = []
    horizons_key = tuple(strategy.horizons)
    for symbol in symbols:
        dm = data_cache[(symbol, horizons_key, strategy.sequence_length)]
        frame = dm.frame.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame["symbol"] = symbol
        bar_parts.append(frame)

        normalizer = strategy.normalizer if strategy.normalizer is not None else dm.normalizer
        actions = generate_actions_from_frame(
            model=strategy.model,
            frame=frame,
            feature_columns=strategy.feature_columns,
            normalizer=normalizer,
            sequence_length=int(strategy.sequence_length),
            horizon=1,
            device=device,
        )
        actions["timestamp"] = pd.to_datetime(actions["timestamp"], utc=True)
        actions["symbol"] = actions["symbol"].astype(str).str.upper()
        action_parts.append(actions)

    actions_df = pd.concat(action_parts, ignore_index=True)
    bars_df = pd.concat(bar_parts, ignore_index=True)

    if cache_path is not None:
        pd.to_pickle({"actions": actions_df, "bars": bars_df}, cache_path)
        logger.info("Saved action cache for {} -> {}", strategy.name, cache_path)

    return actions_df, bars_df


def join_common_keys(actions_by_strategy: dict[str, pd.DataFrame]) -> pd.DataFrame:
    key_df: pd.DataFrame | None = None
    for actions in actions_by_strategy.values():
        keys = actions[["timestamp", "symbol"]].copy()
        keys["timestamp"] = pd.to_datetime(keys["timestamp"], utc=True)
        keys["symbol"] = keys["symbol"].astype(str).str.upper()
        keys = keys.drop_duplicates()
        if key_df is None:
            key_df = keys
        else:
            key_df = key_df.merge(keys, on=["timestamp", "symbol"], how="inner")
    if key_df is None or key_df.empty:
        raise ValueError("No common timestamp/symbol keys across strategies.")
    return key_df.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def filter_to_common_keys(df: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["timestamp"] = pd.to_datetime(tmp["timestamp"], utc=True)
    tmp["symbol"] = tmp["symbol"].astype(str).str.upper()
    merged = tmp.merge(keys, on=["timestamp", "symbol"], how="inner")
    return merged.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def _with_execution_scenario(
    base_cfg: PortfolioConfig,
    scenario: ExecutionScenario,
) -> PortfolioConfig:
    return PortfolioConfig(
        **{
            **base_cfg.__dict__,
            "bar_margin": float(scenario.bar_margin),
            "entry_order_ttl_hours": int(scenario.entry_order_ttl_hours),
        }
    )


def evaluate_execution_scenarios(
    *,
    bars_eval: pd.DataFrame,
    actions_eval: pd.DataFrame,
    base_cfg: PortfolioConfig,
    execution_scenarios: Sequence[ExecutionScenario],
) -> dict[str, object]:
    scenario_rows: list[dict[str, float | int | str]] = []
    for scenario in execution_scenarios:
        sim = run_portfolio_simulation(
            bars_eval,
            actions_eval,
            _with_execution_scenario(base_cfg, scenario),
            horizon=1,
        )
        m = sim.metrics
        scenario_rows.append(
            {
                "label": scenario.label,
                "bar_margin": float(scenario.bar_margin),
                "entry_order_ttl_hours": int(scenario.entry_order_ttl_hours),
                "return_pct": float(m["total_return"] * 100.0),
                "annualized_return_pct": float(m.get("annualized_return", 0.0) * 100.0),
                "sortino": float(m["sortino"]),
                "max_drawdown_pct": float(m.get("max_drawdown", 0.0) * 100.0),
                "pnl_smoothness": float(m.get("pnl_smoothness", 0.0)),
                "ulcer_index": float(m.get("ulcer_index", 0.0)),
                "trade_rate": float(m.get("trade_rate", 0.0)),
                "goodness_score": float(m.get("goodness_score", 0.0)),
                "num_buys": int(m.get("num_buys", 0)),
                "num_sells": int(m.get("num_sells", 0)),
            }
        )

    returns = [float(row["return_pct"]) for row in scenario_rows]
    annualized_returns = [float(row["annualized_return_pct"]) for row in scenario_rows]
    sortinos = [float(row["sortino"]) for row in scenario_rows]
    drawdowns = [float(row["max_drawdown_pct"]) for row in scenario_rows]
    smoothness = [float(row["pnl_smoothness"]) for row in scenario_rows]
    ulcer_index = [float(row["ulcer_index"]) for row in scenario_rows]
    trade_rate = [float(row["trade_rate"]) for row in scenario_rows]
    goodness = [float(row["goodness_score"]) for row in scenario_rows]
    buys = [int(row["num_buys"]) for row in scenario_rows]
    sells = [int(row["num_sells"]) for row in scenario_rows]

    return {
        "return_pct": float(np.min(returns)),
        "annualized_return_pct": float(np.min(annualized_returns)),
        "sortino": float(np.min(sortinos)),
        "max_drawdown_pct": float(np.max(drawdowns)),
        "pnl_smoothness": float(np.max(smoothness)),
        "ulcer_index": float(np.max(ulcer_index)),
        "trade_rate": float(np.min(trade_rate)),
        "goodness_score": float(np.min(goodness)),
        "num_buys": int(np.min(buys)),
        "num_sells": int(np.min(sells)),
        "execution_scenario_count": int(len(scenario_rows)),
        "execution_scenarios": scenario_rows,
        "scenario_mean_return_pct": float(np.mean(returns)),
        "scenario_mean_annualized_return_pct": float(np.mean(annualized_returns)),
        "scenario_mean_sortino": float(np.mean(sortinos)),
        "scenario_mean_max_drawdown_pct": float(np.mean(drawdowns)),
        "scenario_mean_pnl_smoothness": float(np.mean(smoothness)),
        "scenario_mean_ulcer_index": float(np.mean(ulcer_index)),
        "scenario_mean_trade_rate": float(np.mean(trade_rate)),
        "scenario_mean_goodness_score": float(np.mean(goodness)),
        "scenario_mean_num_buys": float(np.mean(buys)),
        "scenario_mean_num_sells": float(np.mean(sells)),
    }


def build_meta_results(
    *,
    bars: pd.DataFrame,
    actions_by_strategy: dict[str, pd.DataFrame],
    daily_returns: dict[str, dict[str, pd.Series]],
    symbols: Sequence[str],
    lookback_days: int,
    metric: str,
    holdout_days: Sequence[int],
    base_cfg: PortfolioConfig,
    tie_break_order: Sequence[str],
    fallback_strategy: str,
    sit_out_threshold: float | None,
    selection_mode: str,
    switch_margin: float,
    min_score_gap: float,
    recency_halflife_days: float | None,
    execution_scenarios: Sequence[ExecutionScenario],
) -> tuple[list[dict], dict]:
    winners_by_symbol: dict[str, pd.Series] = {}
    template_actions = next(iter(actions_by_strategy.values()))
    template_actions = template_actions.copy()
    template_actions["timestamp"] = pd.to_datetime(template_actions["timestamp"], utc=True)
    template_actions["symbol"] = template_actions["symbol"].astype(str).str.upper()
    template_actions["day"] = template_actions["timestamp"].dt.floor("D")

    for symbol in symbols:
        returns_for_symbol = {
            strategy_name: daily_returns[strategy_name][symbol]
            for strategy_name in actions_by_strategy
        }
        winners = select_daily_winners(
            returns_for_symbol,
            lookback_days=lookback_days,
            metric=metric,
            fallback_strategy=fallback_strategy,
            tie_break_order=tie_break_order,
            require_full_window=True,
            sit_out_threshold=sit_out_threshold,
            selection_mode=selection_mode,
            switch_margin=switch_margin,
            min_score_gap=min_score_gap,
            recency_halflife_days=recency_halflife_days,
        )
        symbol_days = (
            template_actions[template_actions["symbol"] == symbol]["day"]
            .drop_duplicates()
            .sort_values()
        )
        if not symbol_days.empty:
            winners = winners.reindex(pd.DatetimeIndex(symbol_days), fill_value=fallback_strategy)
        winners_by_symbol[symbol] = winners

    meta_actions = combine_actions_by_winners(actions_by_strategy, winners_by_symbol)

    period_rows = []
    for days in holdout_days:
        if days > 0:
            cutoff = bars["timestamp"].max() - pd.Timedelta(days=int(days))
            bars_eval = bars[bars["timestamp"] >= cutoff].reset_index(drop=True)
            actions_eval = meta_actions[meta_actions["timestamp"] >= cutoff].reset_index(drop=True)
            period = f"{days}d"
        else:
            bars_eval = bars
            actions_eval = meta_actions
            period = "all"

        period_rows.append(
            {
                "period": period,
                "holdout_days": int(days),
                **evaluate_execution_scenarios(
                    bars_eval=bars_eval,
                    actions_eval=actions_eval,
                    base_cfg=base_cfg,
                    execution_scenarios=execution_scenarios,
                ),
            }
        )

    sortinos = [r["sortino"] for r in period_rows]
    returns = [r["return_pct"] for r in period_rows]
    annualized_returns = [float(r.get("annualized_return_pct", 0.0)) for r in period_rows]
    drawdowns = [r["max_drawdown_pct"] for r in period_rows]
    smoothness = [float(r.get("pnl_smoothness", 0.0)) for r in period_rows]
    ulcer_index = [float(r.get("ulcer_index", 0.0)) for r in period_rows]
    trade_rate = [float(r.get("trade_rate", 0.0)) for r in period_rows]
    goodness = [float(r.get("goodness_score", 0.0)) for r in period_rows]
    buys = [int(r.get("num_buys", 0)) for r in period_rows]
    scenario_mean_returns = [float(r.get("scenario_mean_return_pct", r["return_pct"])) for r in period_rows]
    scenario_mean_annualized_returns = [
        float(r.get("scenario_mean_annualized_return_pct", r.get("annualized_return_pct", 0.0)))
        for r in period_rows
    ]
    scenario_mean_sortinos = [float(r.get("scenario_mean_sortino", r["sortino"])) for r in period_rows]
    scenario_mean_drawdowns = [
        float(r.get("scenario_mean_max_drawdown_pct", r["max_drawdown_pct"]))
        for r in period_rows
    ]
    scenario_mean_smoothness = [
        float(r.get("scenario_mean_pnl_smoothness", r["pnl_smoothness"]))
        for r in period_rows
    ]
    scenario_mean_ulcer_index = [
        float(r.get("scenario_mean_ulcer_index", r["ulcer_index"]))
        for r in period_rows
    ]
    scenario_mean_trade_rate = [float(r.get("scenario_mean_trade_rate", r["trade_rate"])) for r in period_rows]
    scenario_mean_goodness = [
        float(r.get("scenario_mean_goodness_score", r["goodness_score"]))
        for r in period_rows
    ]
    scenario_mean_buys = [float(r.get("scenario_mean_num_buys", r["num_buys"])) for r in period_rows]
    summary = {
        "lookback_days": int(lookback_days),
        "metric": metric,
        "selection_mode": selection_mode,
        "switch_margin": float(switch_margin),
        "min_score_gap": float(min_score_gap),
        "recency_halflife_days": (float(recency_halflife_days) if recency_halflife_days is not None else None),
        "min_sortino": float(np.min(sortinos)),
        "mean_sortino": float(np.mean(sortinos)),
        "min_return_pct": float(np.min(returns)),
        "mean_return_pct": float(np.mean(returns)),
        "min_annualized_return_pct": float(np.min(annualized_returns)),
        "mean_annualized_return_pct": float(np.mean(annualized_returns)),
        "mean_max_drawdown_pct": float(np.mean(drawdowns)),
        "mean_pnl_smoothness": float(np.mean(smoothness)),
        "mean_ulcer_index": float(np.mean(ulcer_index)),
        "mean_trade_rate": float(np.mean(trade_rate)),
        "min_goodness_score": float(np.min(goodness)),
        "mean_goodness_score": float(np.mean(goodness)),
        "min_num_buys": int(np.min(buys)),
        "mean_num_buys": float(np.mean(buys)),
        "execution_scenario_count": int(period_rows[0].get("execution_scenario_count", 1)) if period_rows else 0,
        "mean_scenario_mean_return_pct": float(np.mean(scenario_mean_returns)),
        "mean_scenario_mean_annualized_return_pct": float(np.mean(scenario_mean_annualized_returns)),
        "mean_scenario_mean_sortino": float(np.mean(scenario_mean_sortinos)),
        "mean_scenario_mean_max_drawdown_pct": float(np.mean(scenario_mean_drawdowns)),
        "mean_scenario_mean_pnl_smoothness": float(np.mean(scenario_mean_smoothness)),
        "mean_scenario_mean_ulcer_index": float(np.mean(scenario_mean_ulcer_index)),
        "mean_scenario_mean_trade_rate": float(np.mean(scenario_mean_trade_rate)),
        "mean_scenario_mean_goodness_score": float(np.mean(scenario_mean_goodness)),
        "mean_scenario_mean_num_buys": float(np.mean(scenario_mean_buys)),
    }
    return period_rows, summary


def rank_key(summary: dict) -> tuple[float, float, float, float, float]:
    return (
        float(summary.get("min_goodness_score", summary["min_sortino"])),
        float(summary.get("mean_goodness_score", summary["mean_sortino"])),
        float(summary["min_sortino"]),
        float(summary["mean_sortino"]),
        float(summary["min_return_pct"]),
    )


def evaluate_strategy_baselines(
    *,
    bars: pd.DataFrame,
    actions_by_strategy: dict[str, pd.DataFrame],
    holdout_days: Sequence[int],
    base_cfg: PortfolioConfig,
    execution_scenarios: Sequence[ExecutionScenario],
) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    summaries: list[dict] = []
    for strategy_name, actions in actions_by_strategy.items():
        strategy_rows = []
        for days in holdout_days:
            if days > 0:
                cutoff = bars["timestamp"].max() - pd.Timedelta(days=int(days))
                bars_eval = bars[bars["timestamp"] >= cutoff].reset_index(drop=True)
                actions_eval = actions[actions["timestamp"] >= cutoff].reset_index(drop=True)
                period = f"{days}d"
            else:
                bars_eval = bars
                actions_eval = actions
                period = "all"
            row = {
                "strategy": strategy_name,
                "period": period,
                "holdout_days": int(days),
                **evaluate_execution_scenarios(
                    bars_eval=bars_eval,
                    actions_eval=actions_eval,
                    base_cfg=base_cfg,
                    execution_scenarios=execution_scenarios,
                ),
            }
            strategy_rows.append(row)
            rows.append(row)
        sortinos = [r["sortino"] for r in strategy_rows]
        returns = [r["return_pct"] for r in strategy_rows]
        annualized_returns = [float(r.get("annualized_return_pct", 0.0)) for r in strategy_rows]
        drawdowns = [r["max_drawdown_pct"] for r in strategy_rows]
        smoothness = [float(r.get("pnl_smoothness", 0.0)) for r in strategy_rows]
        ulcer_index = [float(r.get("ulcer_index", 0.0)) for r in strategy_rows]
        trade_rate = [float(r.get("trade_rate", 0.0)) for r in strategy_rows]
        goodness = [float(r.get("goodness_score", 0.0)) for r in strategy_rows]
        scenario_mean_returns = [float(r.get("scenario_mean_return_pct", r["return_pct"])) for r in strategy_rows]
        scenario_mean_annualized_returns = [
            float(r.get("scenario_mean_annualized_return_pct", r.get("annualized_return_pct", 0.0)))
            for r in strategy_rows
        ]
        scenario_mean_sortinos = [float(r.get("scenario_mean_sortino", r["sortino"])) for r in strategy_rows]
        scenario_mean_drawdowns = [
            float(r.get("scenario_mean_max_drawdown_pct", r["max_drawdown_pct"]))
            for r in strategy_rows
        ]
        scenario_mean_smoothness = [
            float(r.get("scenario_mean_pnl_smoothness", r["pnl_smoothness"]))
            for r in strategy_rows
        ]
        scenario_mean_ulcer_index = [
            float(r.get("scenario_mean_ulcer_index", r["ulcer_index"]))
            for r in strategy_rows
        ]
        scenario_mean_trade_rate = [float(r.get("scenario_mean_trade_rate", r["trade_rate"])) for r in strategy_rows]
        scenario_mean_goodness = [
            float(r.get("scenario_mean_goodness_score", r["goodness_score"]))
            for r in strategy_rows
        ]
        summaries.append(
            {
                "strategy": strategy_name,
                "min_sortino": float(np.min(sortinos)),
                "mean_sortino": float(np.mean(sortinos)),
                "min_return_pct": float(np.min(returns)),
                "mean_return_pct": float(np.mean(returns)),
                "min_annualized_return_pct": float(np.min(annualized_returns)),
                "mean_annualized_return_pct": float(np.mean(annualized_returns)),
                "mean_max_drawdown_pct": float(np.mean(drawdowns)),
                "mean_pnl_smoothness": float(np.mean(smoothness)),
                "mean_ulcer_index": float(np.mean(ulcer_index)),
                "mean_trade_rate": float(np.mean(trade_rate)),
                "min_goodness_score": float(np.min(goodness)),
                "mean_goodness_score": float(np.mean(goodness)),
                "execution_scenario_count": int(strategy_rows[0].get("execution_scenario_count", 1)) if strategy_rows else 0,
                "mean_scenario_mean_return_pct": float(np.mean(scenario_mean_returns)),
                "mean_scenario_mean_annualized_return_pct": float(np.mean(scenario_mean_annualized_returns)),
                "mean_scenario_mean_sortino": float(np.mean(scenario_mean_sortinos)),
                "mean_scenario_mean_max_drawdown_pct": float(np.mean(scenario_mean_drawdowns)),
                "mean_scenario_mean_pnl_smoothness": float(np.mean(scenario_mean_smoothness)),
                "mean_scenario_mean_ulcer_index": float(np.mean(scenario_mean_ulcer_index)),
                "mean_scenario_mean_trade_rate": float(np.mean(scenario_mean_trade_rate)),
                "mean_scenario_mean_goodness_score": float(np.mean(scenario_mean_goodness)),
            }
        )
    summaries.sort(key=rank_key, reverse=True)
    return rows, summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep per-symbol meta strategy selectors.")
    parser.add_argument(
        "--strategy",
        action="append",
        required=True,
        help="Strategy spec NAME=PATH or NAME=PATH:EPOCH. PATH can be checkpoint dir or epoch_XXX.pt.",
    )
    parser.add_argument("--symbols", default=",".join(DEFAULT_ALPACA_LIVE8_STOCKS))
    parser.add_argument("--data-root", type=Path, default=Path("trainingdatahourly/stocks"))
    parser.add_argument("--cache-root", type=Path, default=Path("unified_hourly_experiment/forecast_cache"))
    parser.add_argument(
        "--action-cache-dir",
        type=Path,
        default=None,
        help="Optional on-disk cache for generated per-strategy action frames across repeated sweeps.",
    )
    parser.add_argument("--metrics", default="return,sortino,calmar,sharpe")
    parser.add_argument("--selection-modes", default="winner")
    parser.add_argument("--switch-margins", default="0.0")
    parser.add_argument("--min-score-gaps", default="0.0")
    parser.add_argument("--lookback-days", default="1,2,3,5")
    parser.add_argument(
        "--recency-halflife-days",
        default="0.0",
        help="Comma-separated exponential recency half-life (in days) for meta scoring; <=0 disables weighting.",
    )
    parser.add_argument("--holdout-days", default="30,60,90")
    parser.add_argument("--initial-cash", type=float, default=10_000.0)
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--min-edge", type=float, default=0.0)
    parser.add_argument("--max-hold-hours", type=int, default=6)
    parser.add_argument("--trade-amount-scale", type=float, default=100.0)
    parser.add_argument("--entry-intensity-power", type=float, default=1.0)
    parser.add_argument("--entry-min-intensity-fraction", type=float, default=0.0)
    parser.add_argument("--long-intensity-multiplier", type=float, default=1.0)
    parser.add_argument("--short-intensity-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--entry-allocator-mode",
        type=str,
        default="legacy",
        choices=["legacy", "concentrated"],
        help="How to allocate capital across selected entries during portfolio simulation.",
    )
    parser.add_argument("--entry-allocator-edge-power", type=float, default=2.0)
    parser.add_argument("--entry-allocator-max-single-position-fraction", type=float, default=0.6)
    parser.add_argument("--entry-allocator-reserve-fraction", type=float, default=0.1)
    parser.add_argument("--min-buy-amount", type=float, default=0.0)
    parser.add_argument("--decision-lag-bars", type=int, default=1)
    parser.add_argument(
        "--entry-selection-mode",
        type=str,
        default="edge_rank",
        choices=["edge_rank", "first_trigger"],
        help="How the simulator prioritizes competing fillable entries.",
    )
    parser.add_argument(
        "--market-order-entry",
        action="store_true",
        help="Use market-order entry fill assumption in simulations.",
    )
    parser.add_argument("--bar-margin", type=float, default=0.0005)
    parser.add_argument(
        "--execution-bar-margins",
        default="",
        help="Optional comma-separated bar margins used for robustness validation; blank uses --bar-margin only.",
    )
    parser.add_argument(
        "--entry-order-ttl-hours",
        type=int,
        default=0,
        help="How many hourly bars to keep non-filled entry orders pending in simulator (0 disables).",
    )
    parser.add_argument(
        "--execution-entry-order-ttls",
        default="",
        help=(
            "Optional comma-separated entry-order TTL values used for robustness validation; "
            "blank uses --entry-order-ttl-hours only."
        ),
    )
    parser.add_argument("--leverage", type=float, default=2.0)
    parser.add_argument("--force-close-slippage", type=float, default=0.003)
    parser.add_argument("--no-int-qty", action="store_true")
    parser.add_argument("--margin-rate", type=float, default=0.0625)
    parser.add_argument(
        "--sim-backend",
        type=str,
        default="auto",
        choices=["python", "native", "auto"],
        help="Portfolio simulator backend for per-symbol and meta evaluation.",
    )
    parser.add_argument("--no-close-at-eod", action="store_true")
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--default-strategy", default="", help="Fallback strategy when lookback history is insufficient.")
    parser.add_argument("--tie-break-order", default="", help="Comma-separated strategy priority for ties.")
    parser.add_argument(
        "--sit-out-if-negative",
        action="store_true",
        help="When enabled, set symbol/day to cash (no entry) if best trailing score is below threshold.",
    )
    parser.add_argument(
        "--sit-out-threshold",
        type=float,
        default=0.0,
        help="Trailing score threshold used with --sit-out-if-negative (default: 0.0).",
    )
    parser.add_argument(
        "--sit-out-thresholds",
        default="",
        help=(
            "Optional comma-separated sit-out thresholds. "
            "When provided with --sit-out-if-negative, evaluates all listed thresholds in one run."
        ),
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if len(args.strategy) < 2:
        raise ValueError("At least two --strategy specs are required for meta selection.")

    symbols = [s.strip().upper() for s in parse_csv_list(args.symbols)]
    metrics = [m.lower() for m in parse_csv_list(args.metrics)]
    selection_modes = [m.lower() for m in parse_csv_list(args.selection_modes)]
    switch_margins = parse_float_list(args.switch_margins)
    min_score_gaps = parse_float_list(args.min_score_gaps)
    lookbacks = parse_int_list(args.lookback_days)
    recency_halflifes = parse_float_list(args.recency_halflife_days)
    holdouts = parse_int_list(args.holdout_days)
    sit_out_threshold_values = parse_sit_out_threshold_values(
        sit_out_if_negative=bool(args.sit_out_if_negative),
        sit_out_threshold=float(args.sit_out_threshold),
        sit_out_thresholds=args.sit_out_thresholds,
    )

    invalid_metrics = [
        m
        for m in metrics
        if m not in ("return", "sortino", "sharpe", "calmar", "omega", "gain_pain", "p10", "median", "goodness")
    ]
    if invalid_metrics:
        raise ValueError(f"Invalid metrics: {invalid_metrics}")
    invalid_modes = [m for m in selection_modes if m not in ("winner", "winner_cash", "sticky")]
    if invalid_modes:
        raise ValueError(f"Invalid selection modes: {invalid_modes}")
    if any(x < 0 for x in switch_margins):
        raise ValueError(f"switch-margins must all be >= 0, got {switch_margins}")
    if any(x < 0 for x in min_score_gaps):
        raise ValueError(f"min-score-gaps must all be >= 0, got {min_score_gaps}")
    if any(x <= 0 for x in lookbacks):
        raise ValueError(f"lookback-days must all be > 0, got {lookbacks}")
    if any(x < 0 for x in recency_halflifes):
        raise ValueError(f"recency-halflife-days must all be >= 0, got {recency_halflifes}")
    execution_bar_margins = (
        parse_float_list(args.execution_bar_margins)
        if str(args.execution_bar_margins).strip()
        else [float(args.bar_margin)]
    )
    execution_entry_order_ttls = (
        parse_int_list(args.execution_entry_order_ttls)
        if str(args.execution_entry_order_ttls).strip()
        else [int(args.entry_order_ttl_hours)]
    )
    if any(x < 0 for x in execution_bar_margins):
        raise ValueError(f"execution-bar-margins must all be >= 0, got {execution_bar_margins}")
    if any(x < 0 for x in execution_entry_order_ttls):
        raise ValueError(
            f"execution-entry-order-ttls must all be >= 0, got {execution_entry_order_ttls}"
        )
    if args.trade_amount_scale <= 0:
        raise ValueError("--trade-amount-scale must be > 0")
    if args.entry_intensity_power < 0:
        raise ValueError("--entry-intensity-power must be >= 0")
    if args.entry_min_intensity_fraction < 0:
        raise ValueError("--entry-min-intensity-fraction must be >= 0")
    if args.long_intensity_multiplier < 0:
        raise ValueError("--long-intensity-multiplier must be >= 0")
    if args.short_intensity_multiplier < 0:
        raise ValueError("--short-intensity-multiplier must be >= 0")
    if args.entry_allocator_edge_power < 0:
        raise ValueError("--entry-allocator-edge-power must be >= 0")
    if not 0 <= float(args.entry_allocator_max_single_position_fraction) <= 1:
        raise ValueError("--entry-allocator-max-single-position-fraction must be in [0, 1]")
    if not 0 <= float(args.entry_allocator_reserve_fraction) <= 1:
        raise ValueError("--entry-allocator-reserve-fraction must be in [0, 1]")
    if args.min_buy_amount < 0:
        raise ValueError("--min-buy-amount must be >= 0")

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but torch.cuda.is_available() is false.")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    strategy_specs = [parse_strategy_spec(spec) for spec in args.strategy]
    strategies = [
        load_strategy_model(name, path, epoch=epoch, device=device)
        for name, path, epoch in strategy_specs
    ]
    strategy_names = [s.name for s in strategies]
    if len(strategy_names) != len(set(strategy_names)):
        raise ValueError(f"Strategy names must be unique. Got {strategy_names}")

    tie_break_order = parse_csv_list(args.tie_break_order) if args.tie_break_order else strategy_names
    fallback_strategy = args.default_strategy.strip() if args.default_strategy else tie_break_order[0]
    if fallback_strategy not in strategy_names:
        raise ValueError(f"default-strategy '{fallback_strategy}' is not in strategy names {strategy_names}")

    logger.info("Loaded {} strategies on device {}", len(strategies), device)
    for s in strategies:
        logger.info("  {} -> {} ({})", s.name, s.checkpoint_dir, s.checkpoint_name)

    data_cache = build_data_cache(
        symbols,
        strategies,
        data_root=args.data_root,
        cache_root=args.cache_root,
    )

    actions_by_strategy: dict[str, pd.DataFrame] = {}
    bars_by_strategy: dict[str, pd.DataFrame] = {}
    for strategy in strategies:
        actions_df, bars_df = load_or_generate_strategy_frames(
            strategy=strategy,
            symbols=symbols,
            data_cache=data_cache,
            device=device,
            action_cache_dir=args.action_cache_dir,
        )
        actions_by_strategy[strategy.name] = actions_df
        bars_by_strategy[strategy.name] = bars_df

    common_keys = join_common_keys(actions_by_strategy)
    actions_by_strategy = {
        name: filter_to_common_keys(actions, common_keys)
        for name, actions in actions_by_strategy.items()
    }
    bars_full = filter_to_common_keys(bars_by_strategy[strategy_names[0]], common_keys)

    logger.info(
        "Aligned common data: {} rows, {} symbols, {} timestamps",
        len(common_keys),
        common_keys["symbol"].nunique(),
        common_keys["timestamp"].nunique(),
    )

    base_cfg = PortfolioConfig(
        initial_cash=args.initial_cash,
        max_positions=args.max_positions,
        min_edge=args.min_edge,
        max_hold_hours=args.max_hold_hours,
        enforce_market_hours=True,
        close_at_eod=not args.no_close_at_eod,
        symbols=symbols,
        trade_amount_scale=args.trade_amount_scale,
        entry_intensity_power=args.entry_intensity_power,
        entry_min_intensity_fraction=args.entry_min_intensity_fraction,
        long_intensity_multiplier=args.long_intensity_multiplier,
        short_intensity_multiplier=args.short_intensity_multiplier,
        entry_allocator_mode=normalize_entry_allocator_mode(args.entry_allocator_mode),
        entry_allocator_edge_power=args.entry_allocator_edge_power,
        entry_allocator_max_single_position_fraction=args.entry_allocator_max_single_position_fraction,
        entry_allocator_reserve_fraction=args.entry_allocator_reserve_fraction,
        min_buy_amount=args.min_buy_amount,
        decision_lag_bars=args.decision_lag_bars,
        entry_selection_mode=str(args.entry_selection_mode),
        market_order_entry=bool(args.market_order_entry),
        bar_margin=args.bar_margin,
        entry_order_ttl_hours=int(args.entry_order_ttl_hours),
        max_leverage=args.leverage,
        force_close_slippage=args.force_close_slippage,
        int_qty=not args.no_int_qty,
        fee_by_symbol=dict.fromkeys(symbols, args.fee_rate),
        margin_annual_rate=args.margin_rate,
        sim_backend=args.sim_backend,
    )
    execution_scenarios = resolve_execution_scenarios(
        base_cfg=base_cfg,
        bar_margins=execution_bar_margins,
        entry_order_ttls=execution_entry_order_ttls,
    )

    daily_returns: dict[str, dict[str, pd.Series]] = {name: {} for name in strategy_names}
    for strategy_name in strategy_names:
        actions = actions_by_strategy[strategy_name]
        for symbol in symbols:
            bars_sym = bars_full[bars_full["symbol"] == symbol].reset_index(drop=True)
            actions_sym = actions[actions["symbol"] == symbol].reset_index(drop=True)
            symbol_cfg = PortfolioConfig(**{**base_cfg.__dict__, "max_positions": 1, "symbols": [symbol]})
            sim = run_portfolio_simulation(bars_sym, actions_sym, symbol_cfg, horizon=1)
            daily_returns[strategy_name][symbol] = daily_returns_from_equity(sim.equity_curve)

    baseline_rows, baseline_summaries = evaluate_strategy_baselines(
        bars=bars_full,
        actions_by_strategy=actions_by_strategy,
        holdout_days=holdouts,
        base_cfg=base_cfg,
        execution_scenarios=execution_scenarios,
    )
    logger.info("Top single-strategy baselines on current holdouts:")
    for item in baseline_summaries[:5]:
        logger.info(
            "  {} -> min_sort={:.3f} mean_sort={:.3f} min_ret={:+.2f}% mean_ret={:+.2f}% min_ann={:+.2f}% mean_ann={:+.2f}% mean_dd={:.2f}%",
            item["strategy"],
            item["min_sortino"],
            item["mean_sortino"],
            item["min_return_pct"],
            item["mean_return_pct"],
            item.get("min_annualized_return_pct", 0.0),
            item.get("mean_annualized_return_pct", 0.0),
            item["mean_max_drawdown_pct"],
        )

    all_rows = []
    summaries = []
    for metric in metrics:
        for mode in selection_modes:
            for switch_margin in switch_margins:
                for min_score_gap in min_score_gaps:
                    for recency_halflife in recency_halflifes:
                        recency_value = float(recency_halflife)
                        recency_for_score = recency_value if recency_value > 0 else None
                        for sit_out_threshold in sit_out_threshold_values:
                            for lookback in lookbacks:
                                period_rows, summary = build_meta_results(
                                    bars=bars_full,
                                    actions_by_strategy=actions_by_strategy,
                                    daily_returns=daily_returns,
                                    symbols=symbols,
                                    lookback_days=int(lookback),
                                    metric=metric,
                                    holdout_days=holdouts,
                                    base_cfg=base_cfg,
                                    tie_break_order=tie_break_order,
                                    fallback_strategy=fallback_strategy,
                                    sit_out_threshold=sit_out_threshold,
                                    selection_mode=mode,
                                    switch_margin=float(switch_margin),
                                    min_score_gap=float(min_score_gap),
                                    recency_halflife_days=recency_for_score,
                                    execution_scenarios=execution_scenarios,
                                )
                                for row in period_rows:
                                    all_rows.append(
                                        {
                                            "metric": metric,
                                            "lookback_days": int(lookback),
                                            "selection_mode": mode,
                                            "switch_margin": float(switch_margin),
                                            "min_score_gap": float(min_score_gap),
                                            "recency_halflife_days": (
                                                recency_for_score if recency_for_score is not None else 0.0
                                            ),
                                            "sit_out_threshold": (
                                                float(sit_out_threshold)
                                                if sit_out_threshold is not None
                                                else None
                                            ),
                                            **row,
                                        }
                                    )
                                summary_row = {"metric": metric, "sit_out_threshold": sit_out_threshold, **summary}
                                summaries.append(summary_row)
                                logger.info(
                                    "meta {} mode={} sm={:.4f} mg={:.4f} hl={} th={} lb{} -> min_sort={:.3f} mean_sort={:.3f} min_ret={:+.2f}% mean_ret={:+.2f}% min_ann={:+.2f}% mean_ann={:+.2f}% mean_dd={:.2f}%",
                                    metric,
                                    mode,
                                    switch_margin,
                                    min_score_gap,
                                    (recency_for_score if recency_for_score is not None else 0.0),
                                    ("none" if sit_out_threshold is None else f"{float(sit_out_threshold):.4f}"),
                                    lookback,
                                    summary["min_sortino"],
                                    summary["mean_sortino"],
                                    summary["min_return_pct"],
                                    summary["mean_return_pct"],
                                    summary.get("min_annualized_return_pct", 0.0),
                                    summary.get("mean_annualized_return_pct", 0.0),
                                    summary["mean_max_drawdown_pct"],
                                )

    best = max(summaries, key=rank_key)
    logger.info(
        "BEST meta config: metric={} lookback={}d min_sort={:.3f} mean_sort={:.3f} min_ret={:+.2f}% mean_ret={:+.2f}% min_ann={:+.2f}% mean_ann={:+.2f}% mean_dd={:.2f}%",
        best["metric"],
        best["lookback_days"],
        best["min_sortino"],
        best["mean_sortino"],
        best["min_return_pct"],
        best["mean_return_pct"],
        best.get("min_annualized_return_pct", 0.0),
        best.get("mean_annualized_return_pct", 0.0),
        best["mean_max_drawdown_pct"],
    )

    if args.output is None:
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        out_dir = Path("experiments") / f"meta_portfolio_sweep_{stamp}"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / "meta_sweep_results.json"
    else:
        output_path = args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "strategies": [
            {
                "name": s.name,
                "checkpoint_dir": str(s.checkpoint_dir),
                "checkpoint": s.checkpoint_name,
                "sequence_length": int(s.sequence_length),
                "horizons": s.horizons,
            }
            for s in strategies
        ],
        "symbols": symbols,
        "config": {
            "action_cache_dir": (str(args.action_cache_dir) if args.action_cache_dir is not None else None),
            "metrics": metrics,
            "selection_modes": selection_modes,
            "switch_margins": switch_margins,
            "min_score_gaps": min_score_gaps,
            "lookback_days": lookbacks,
            "recency_halflife_days": recency_halflifes,
            "holdout_days": holdouts,
            "default_strategy": fallback_strategy,
            "tie_break_order": tie_break_order,
            "sit_out_if_negative": bool(args.sit_out_if_negative),
            "sit_out_threshold": float(args.sit_out_threshold),
            "sit_out_thresholds": [
                (float(v) if v is not None else None)
                for v in sit_out_threshold_values
            ],
            "portfolio": {
                "initial_cash": args.initial_cash,
                "max_positions": args.max_positions,
                "min_edge": args.min_edge,
                "max_hold_hours": args.max_hold_hours,
                "trade_amount_scale": args.trade_amount_scale,
                "entry_intensity_power": args.entry_intensity_power,
                "entry_min_intensity_fraction": args.entry_min_intensity_fraction,
                "long_intensity_multiplier": args.long_intensity_multiplier,
                "short_intensity_multiplier": args.short_intensity_multiplier,
                "entry_allocator_mode": normalize_entry_allocator_mode(args.entry_allocator_mode),
                "entry_allocator_edge_power": args.entry_allocator_edge_power,
                "entry_allocator_max_single_position_fraction": args.entry_allocator_max_single_position_fraction,
                "entry_allocator_reserve_fraction": args.entry_allocator_reserve_fraction,
                "min_buy_amount": args.min_buy_amount,
                "decision_lag_bars": args.decision_lag_bars,
                "entry_selection_mode": str(args.entry_selection_mode),
                "market_order_entry": bool(args.market_order_entry),
                "bar_margin": args.bar_margin,
                "entry_order_ttl_hours": int(args.entry_order_ttl_hours),
                "leverage": args.leverage,
                "fee_rate": args.fee_rate,
                "close_at_eod": not args.no_close_at_eod,
                "sim_backend": args.sim_backend,
            },
            "execution_validation": {
                "bar_margins": [float(s.bar_margin) for s in execution_scenarios],
                "entry_order_ttl_hours": [int(s.entry_order_ttl_hours) for s in execution_scenarios],
                "scenario_labels": [s.label for s in execution_scenarios],
                "scenario_count": len(execution_scenarios),
            },
        },
        "best": best,
        "strategy_baseline_summaries": baseline_summaries,
        "strategy_baseline_results": baseline_rows,
        "summaries": summaries,
        "results": all_rows,
    }

    output_path.write_text(json.dumps(payload, indent=2))
    logger.info("Saved meta sweep results to {}", output_path)


if __name__ == "__main__":
    main()
