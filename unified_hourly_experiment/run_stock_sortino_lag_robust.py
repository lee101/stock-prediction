#!/usr/bin/env python3
"""Train stock-only hourly policies and rank them by lag-robust risk-adjusted performance."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import torch
from loguru import logger

from binanceneural.config import DatasetConfig, PolicyConfig
from binanceneural.data import BinanceHourlyDataModule
from binanceneural.inference import generate_actions_from_frame
from binanceneural.model import build_policy
from src.robust_trading_metrics import (
    compute_max_drawdown,
    compute_pnl_smoothness_from_equity,
    summarize_lag_results,
)
from src.torch_load_utils import torch_load_compat
from unified_hourly_experiment.marketsimulator import PortfolioConfig, run_portfolio_simulation

DEFAULT_STOCKS = (
    "NVDA,MSFT,META,GOOG,NET,PLTR,NYT,YELP,DBX,TRIP,KIND,EBAY,MTCH,ANGI,Z,EXPE,BKNG,NWSA"
)

DEFAULT_CONFIGS: list[dict[str, Any]] = [
    {
        "name": "rw008_sm000_ft5e4_lagr012",
        "return_weight": 0.08,
        "smoothness_penalty": 0.00,
        "fill_temperature": 5e-4,
        "decision_lag_range": "0,1,2",
    },
    {
        "name": "rw012_sm003_ft5e4_lagr012",
        "return_weight": 0.12,
        "smoothness_penalty": 0.003,
        "fill_temperature": 5e-4,
        "decision_lag_range": "0,1,2",
    },
    {
        "name": "rw015_sm006_ft5e4_lagr0123",
        "return_weight": 0.15,
        "smoothness_penalty": 0.006,
        "fill_temperature": 5e-4,
        "decision_lag_range": "0,1,2,3",
    },
    {
        "name": "rw018_sm010_ft1e4_lagr0123",
        "return_weight": 0.18,
        "smoothness_penalty": 0.010,
        "fill_temperature": 1e-4,
        "decision_lag_range": "0,1,2,3",
    },
]


def parse_int_list(value: str) -> list[int]:
    values = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError(f"Expected at least one integer, got {value!r}")
    return values


def parse_symbols(value: str) -> list[str]:
    symbols = [token.strip().upper() for token in value.split(",") if token.strip()]
    if not symbols:
        raise ValueError("At least one symbol is required")
    return symbols


def infer_horizons(feature_columns: list[str], required_horizon: int) -> tuple[int, ...]:
    horizons: set[int] = set()
    for col in feature_columns:
        if "_h" not in col:
            continue
        suffix = col.rsplit("_h", 1)[-1]
        if suffix.isdigit():
            horizons.add(int(suffix))
    horizons.add(int(required_horizon))
    return tuple(sorted(horizons)) if horizons else (required_horizon, 24)


def load_configs(config_path: Path | None) -> list[dict[str, Any]]:
    if config_path is None:
        return [dict(row) for row in DEFAULT_CONFIGS]
    payload = json.loads(config_path.read_text())
    if not isinstance(payload, list):
        raise ValueError("--configs-json must point to a JSON list")
    out: list[dict[str, Any]] = []
    for row in payload:
        if not isinstance(row, dict) or "name" not in row:
            raise ValueError("Each config must be a JSON object with at least a 'name' field")
        out.append(dict(row))
    return out


def choose_best_epoch(checkpoint_dir: Path) -> int:
    meta_path = checkpoint_dir / "training_meta.json"
    if meta_path.exists():
        payload = json.loads(meta_path.read_text())
        best_epoch = payload.get("best_epoch")
        if isinstance(best_epoch, int) and best_epoch > 0:
            return best_epoch
        history = payload.get("history", [])
        if isinstance(history, list) and history:
            sorted_history = sorted(
                (row for row in history if isinstance(row, dict)),
                key=lambda row: float(row.get("val_sortino") or float("-inf")),
                reverse=True,
            )
            top = sorted_history[0] if sorted_history else None
            if top and int(top.get("epoch", 0)) > 0:
                return int(top["epoch"])

    checkpoints = sorted(checkpoint_dir.glob("epoch_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under {checkpoint_dir}")
    return int(checkpoints[-1].stem.split("_")[1])


def load_model_for_epoch(checkpoint_dir: Path, epoch: int, device: torch.device):
    config_path = checkpoint_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {checkpoint_dir}")
    config = json.loads(config_path.read_text())
    feature_columns = list(config.get("feature_columns", []))
    if not feature_columns:
        raise ValueError(f"No feature columns recorded in {config_path}")

    policy_cfg = PolicyConfig(
        input_dim=len(feature_columns),
        hidden_dim=int(config.get("transformer_dim", 128)),
        num_heads=int(config.get("transformer_heads", 4)),
        num_layers=int(config.get("transformer_layers", 3)),
        num_outputs=int(config.get("num_outputs", 4)),
        model_arch=str(config.get("model_arch", "gemma")),
        max_len=int(config.get("sequence_length", 32)),
    )
    model = build_policy(policy_cfg)

    ckpt_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
    payload = torch_load_compat(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict", payload)
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)
    return model, feature_columns, int(config.get("sequence_length", 32))


def prepare_holdout_data(
    symbols: list[str],
    *,
    data_root: Path,
    cache_root: Path,
    sequence_length: int,
    validation_days: int,
    horizons: tuple[int, ...],
) -> dict[str, dict[str, Any]]:
    by_symbol: dict[str, dict[str, Any]] = {}
    val_hours = max(24, int(validation_days) * 24)
    min_history = max(sequence_length + 24, 100)

    for symbol in symbols:
        cfg = DatasetConfig(
            symbol=symbol,
            data_root=str(data_root),
            forecast_cache_root=str(cache_root),
            forecast_horizons=horizons,
            sequence_length=sequence_length,
            min_history_hours=min_history,
            validation_days=validation_days,
            cache_only=True,
        )
        try:
            dm = BinanceHourlyDataModule(cfg)
        except Exception as exc:
            logger.warning("Skipping {}: {}", symbol, exc)
            continue
        frame = dm.frame.sort_values("timestamp").reset_index(drop=True)
        if len(frame) <= sequence_length:
            logger.warning("Skipping {}: insufficient rows ({})", symbol, len(frame))
            continue

        holdout_start_idx = max(sequence_length - 1, len(frame) - val_hours)
        holdout_start_idx = min(holdout_start_idx, len(frame) - 1)
        holdout_start_ts = frame["timestamp"].iloc[holdout_start_idx]
        inference_start_idx = max(0, holdout_start_idx - sequence_length + 1)
        inference_frame = frame.iloc[inference_start_idx:].reset_index(drop=True)

        by_symbol[symbol] = {
            "normalizer": dm.normalizer,
            "inference_frame": inference_frame,
            "holdout_start_ts": holdout_start_ts,
        }
    if not by_symbol:
        raise RuntimeError("No symbols with usable holdout data were loaded")
    return by_symbol


def build_holdout_bars_actions(
    model,
    feature_columns: list[str],
    *,
    sequence_length: int,
    horizon: int,
    symbol_data: dict[str, dict[str, Any]],
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    bars_parts: list[pd.DataFrame] = []
    action_parts: list[pd.DataFrame] = []
    used_symbols: list[str] = []

    for symbol, payload in symbol_data.items():
        frame = payload["inference_frame"]
        holdout_start_ts = payload["holdout_start_ts"]

        actions = generate_actions_from_frame(
            model=model,
            frame=frame,
            feature_columns=feature_columns,
            normalizer=payload["normalizer"],
            sequence_length=sequence_length,
            horizon=horizon,
            device=device,
        )
        actions = actions[actions["timestamp"] >= holdout_start_ts].copy()
        bars = frame[frame["timestamp"] >= holdout_start_ts].copy()

        if actions.empty or bars.empty:
            logger.warning("Skipping {} in eval: no holdout bars/actions after filtering", symbol)
            continue

        bars_parts.append(bars)
        action_parts.append(actions)
        used_symbols.append(symbol)

    if not bars_parts or not action_parts:
        raise RuntimeError("No bars/actions available for holdout evaluation")

    bars_df = pd.concat(bars_parts, ignore_index=True)
    actions_df = pd.concat(action_parts, ignore_index=True)
    return bars_df, actions_df, used_symbols


def evaluate_lag_sweep(
    bars: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    symbols: list[str],
    lags: list[int],
    initial_cash: float,
    max_positions: int,
    min_edge: float,
    max_hold_hours: int,
    trade_amount_scale: float,
    market_order_entry: bool,
    horizon: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    lag_results: list[dict[str, Any]] = []
    for lag in lags:
        cfg = PortfolioConfig(
            initial_cash=initial_cash,
            max_positions=max_positions,
            min_edge=min_edge,
            max_hold_hours=max_hold_hours,
            enforce_market_hours=True,
            close_at_eod=True,
            symbols=symbols,
            trade_amount_scale=trade_amount_scale,
            decision_lag_bars=lag,
            market_order_entry=market_order_entry,
        )
        sim = run_portfolio_simulation(bars, actions, cfg, horizon=horizon)
        equity = sim.equity_curve.to_numpy(dtype=float)
        result_row = {
            "lag_bars": int(lag),
            "return_pct": float(sim.metrics.get("total_return", 0.0) * 100.0),
            "sortino": float(sim.metrics.get("sortino", 0.0)),
            "max_drawdown_pct": float(compute_max_drawdown(equity) * 100.0),
            "pnl_smoothness": float(compute_pnl_smoothness_from_equity(equity)),
            "num_buys": int(sim.metrics.get("num_buys", 0)),
            "num_sells": int(sim.metrics.get("num_sells", 0)),
        }
        lag_results.append(result_row)
    summary = summarize_lag_results(lag_results)
    return lag_results, summary


def train_one_config(
    cfg: dict[str, Any],
    *,
    run_name: str,
    symbols_arg: str,
    args: argparse.Namespace,
) -> Path:
    checkpoint_dir = args.checkpoint_root / run_name
    existing = sorted(checkpoint_dir.glob("epoch_*.pt"))
    if args.reuse_checkpoints and existing:
        logger.info("Reusing existing checkpoint run {}", run_name)
        return checkpoint_dir

    train_cmd = [
        sys.executable,
        "unified_hourly_experiment/train_unified_policy.py",
        "--symbols",
        symbols_arg,
        "--crypto-symbols",
        "",
        "--stock-data-root",
        str(args.data_root),
        "--stock-cache-root",
        str(args.cache_root),
        "--epochs",
        str(int(cfg.get("epochs", args.epochs))),
        "--batch-size",
        str(int(cfg.get("batch_size", args.batch_size))),
        "--lr",
        str(float(cfg.get("lr", args.lr))),
        "--sequence-length",
        str(int(cfg.get("sequence_length", args.sequence_length))),
        "--hidden-dim",
        str(int(cfg.get("hidden_dim", args.hidden_dim))),
        "--num-layers",
        str(int(cfg.get("num_layers", args.num_layers))),
        "--num-heads",
        str(int(cfg.get("num_heads", args.num_heads))),
        "--return-weight",
        str(float(cfg.get("return_weight", args.return_weight))),
        "--smoothness-penalty",
        str(float(cfg.get("smoothness_penalty", args.smoothness_penalty))),
        "--fill-temperature",
        str(float(cfg.get("fill_temperature", args.fill_temperature))),
        "--decision-lag-bars",
        str(int(cfg.get("decision_lag_bars", args.decision_lag_bars))),
        "--decision-lag-range",
        str(cfg.get("decision_lag_range", args.decision_lag_range)),
        "--forecast-horizons",
        str(cfg.get("forecast_horizons", args.forecast_horizons)),
        "--checkpoint-root",
        str(args.checkpoint_root),
        "--checkpoint-name",
        run_name,
        "--seed",
        str(int(cfg.get("seed", args.seed))),
    ]
    if bool(cfg.get("market_order_entry", False)) or args.market_order_entry:
        train_cmd.append("--market-order-entry")
    if bool(cfg.get("no_compile", False)) or args.no_compile:
        train_cmd.append("--no-compile")

    logger.info("Training config {} as run {}", cfg["name"], run_name)
    logger.info("Command: {}", " ".join(train_cmd))
    completed = subprocess.run(train_cmd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Training failed for {cfg['name']} with exit code {completed.returncode}")
    return checkpoint_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default=DEFAULT_STOCKS)
    parser.add_argument("--configs-json", type=Path, default=None)
    parser.add_argument("--max-configs", type=int, default=0)
    parser.add_argument("--experiment-name", type=str, default="")
    parser.add_argument("--eval-lags", type=str, default="0,1,2,3")
    parser.add_argument("--validation-days", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--forecast-horizons", type=str, default="1,24")

    parser.add_argument("--data-root", type=Path, default=Path("trainingdatahourly/stocks"))
    parser.add_argument("--cache-root", type=Path, default=Path("unified_hourly_experiment/forecast_cache"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("unified_hourly_experiment/checkpoints"))

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--return-weight", type=float, default=0.08)
    parser.add_argument("--smoothness-penalty", type=float, default=0.0)
    parser.add_argument("--fill-temperature", type=float, default=5e-4)
    parser.add_argument("--decision-lag-bars", type=int, default=0)
    parser.add_argument("--decision-lag-range", type=str, default="0,1,2")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--market-order-entry", action="store_true")
    parser.add_argument("--reuse-checkpoints", action="store_true")

    parser.add_argument("--initial-cash", type=float, default=10_000.0)
    parser.add_argument("--max-positions", type=int, default=10)
    parser.add_argument("--min-edge", type=float, default=0.001)
    parser.add_argument("--max-hold-hours", type=int, default=4)
    parser.add_argument("--trade-amount-scale", type=float, default=100.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    symbols = parse_symbols(args.symbols)
    lag_values = parse_int_list(args.eval_lags)
    configs = load_configs(args.configs_json)
    if args.max_configs > 0:
        configs = configs[: args.max_configs]
    if not configs:
        raise ValueError("No configs to run")

    exp_name = args.experiment_name.strip() or datetime.now(timezone.utc).strftime(
        "stock_sortino_lag_robust_%Y%m%d_%H%M%S"
    )
    exp_dir = Path("experiments") / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    results_path = exp_dir / "results.json"
    summary_path = exp_dir / "summary.json"

    logger.info("Running {} configs on {} symbols", len(configs), len(symbols))
    logger.info("Experiment directory: {}", exp_dir)
    logger.info("Lag evaluation values: {}", lag_values)

    all_results: list[dict[str, Any]] = []
    holdout_cache: dict[tuple[int, tuple[int, ...]], dict[str, dict[str, Any]]] = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for idx, cfg in enumerate(configs, start=1):
        cfg_name = str(cfg["name"])
        run_name = f"{exp_name}_{cfg_name}"
        logger.info("[{}/{}] {}", idx, len(configs), cfg_name)

        checkpoint_dir = train_one_config(cfg, run_name=run_name, symbols_arg=",".join(symbols), args=args)
        best_epoch = choose_best_epoch(checkpoint_dir)
        logger.info("Selected epoch {} for {}", best_epoch, cfg_name)

        model, feature_columns, sequence_length = load_model_for_epoch(checkpoint_dir, best_epoch, device)
        horizons = infer_horizons(feature_columns, args.horizon)
        cache_key = (sequence_length, horizons)
        if cache_key not in holdout_cache:
            holdout_cache[cache_key] = prepare_holdout_data(
                symbols,
                data_root=args.data_root,
                cache_root=args.cache_root,
                sequence_length=sequence_length,
                validation_days=args.validation_days,
                horizons=horizons,
            )
        symbol_holdout = holdout_cache[cache_key]

        bars, actions, used_symbols = build_holdout_bars_actions(
            model,
            feature_columns,
            sequence_length=sequence_length,
            horizon=args.horizon,
            symbol_data=symbol_holdout,
            device=device,
        )
        lag_rows, lag_summary = evaluate_lag_sweep(
            bars,
            actions,
            symbols=used_symbols,
            lags=lag_values,
            initial_cash=args.initial_cash,
            max_positions=args.max_positions,
            min_edge=args.min_edge,
            max_hold_hours=args.max_hold_hours,
            trade_amount_scale=args.trade_amount_scale,
            market_order_entry=args.market_order_entry,
            horizon=args.horizon,
        )

        result_row = {
            "name": cfg_name,
            "run_name": run_name,
            "checkpoint_dir": str(checkpoint_dir),
            "best_epoch": best_epoch,
            "train_config": cfg,
            "used_symbols": used_symbols,
            "sequence_length": sequence_length,
            "feature_horizons": list(horizons),
            "lags": lag_rows,
            "summary": lag_summary,
        }
        all_results.append(result_row)
        results_path.write_text(json.dumps(all_results, indent=2))

        logger.info(
            "Config {} robust_score={:.4f} sortino_p10={:.4f} mean_ret={:+.2f}%",
            cfg_name,
            lag_summary["robust_score"],
            lag_summary["sortino_p10"],
            lag_summary["return_mean_pct"],
        )

    ranked = sorted(all_results, key=lambda row: row["summary"]["robust_score"], reverse=True)
    summary_payload = {
        "experiment_name": exp_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "symbols": symbols,
        "eval_lags": lag_values,
        "ranking": [
            {
                "name": row["name"],
                "run_name": row["run_name"],
                "robust_score": row["summary"]["robust_score"],
                "sortino_p10": row["summary"]["sortino_p10"],
                "sortino_mean": row["summary"]["sortino_mean"],
                "return_mean_pct": row["summary"]["return_mean_pct"],
                "max_drawdown_mean_pct": row["summary"]["max_drawdown_mean_pct"],
                "pnl_smoothness_mean": row["summary"]["pnl_smoothness_mean"],
            }
            for row in ranked
        ],
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2))

    logger.info("Saved results to {}", results_path)
    logger.info("Saved summary to {}", summary_path)
    if ranked:
        best = ranked[0]
        logger.success(
            "Best config {} | robust_score={:.4f} | sortino_p10={:.4f} | mean_return={:+.2f}%",
            best["name"],
            best["summary"]["robust_score"],
            best["summary"]["sortino_p10"],
            best["summary"]["return_mean_pct"],
        )


if __name__ == "__main__":
    main()
