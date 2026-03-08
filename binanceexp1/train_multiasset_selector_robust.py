from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

from binanceneural.config import TrainingConfig
from binanceneural.trainer import BinanceHourlyTrainer
from src.fees import get_fee_for_symbol
from src.torch_load_utils import torch_load_compat

from .config import DatasetConfig
from .data import BinanceExp1DataModule
from .search_checkpoint_sets_robust import _resolve_checkpoint_path

DEFAULT_SYMBOLS = ("BTCUSD", "ETHUSD", "SOLUSD")

DEFAULT_CONFIGS: list[dict[str, Any]] = [
    {
        "name": "mwdd_rw010_dd3_sm003_lagr012_buf10_fn01_ft12",
        "epochs": 12,
        "learning_rate": 1e-4,
        "weight_decay": 3e-4,
        "return_weight": 0.10,
        "smoothness_penalty": 0.003,
        "loss_type": "multiwindow_dd",
        "dd_penalty": 3.0,
        "multiwindow_fractions": "0.25,0.5,0.75,1.0",
        "multiwindow_aggregation": "minimax",
        "fill_temperature": 5e-4,
        "fill_buffer_pct": 0.0010,
        "decision_lag_bars": 1,
        "decision_lag_range": "0,1,2",
        "feature_noise_std": 0.01,
        "lr_schedule": "cosine",
        "lr_min_ratio": 0.10,
        "warmup_steps": 64,
        "validation_use_binary_fills": True,
    },
    {
        "name": "mwdd_rw012_dd5_sm006_lagr0123_buf15_fn02_ft12",
        "epochs": 12,
        "learning_rate": 1e-4,
        "weight_decay": 5e-4,
        "return_weight": 0.12,
        "smoothness_penalty": 0.006,
        "loss_type": "multiwindow_dd",
        "dd_penalty": 5.0,
        "multiwindow_fractions": "0.25,0.5,0.75,1.0",
        "multiwindow_aggregation": "minimax",
        "fill_temperature": 5e-4,
        "fill_buffer_pct": 0.0015,
        "decision_lag_bars": 1,
        "decision_lag_range": "0,1,2,3",
        "feature_noise_std": 0.02,
        "lr_schedule": "cosine",
        "lr_min_ratio": 0.05,
        "warmup_steps": 64,
        "validation_use_binary_fills": True,
    },
    {
        "name": "sortdd_rw009_dd4_sm004_lagr012_buf10_fn01_ft12",
        "epochs": 12,
        "learning_rate": 1e-4,
        "weight_decay": 3e-4,
        "return_weight": 0.09,
        "smoothness_penalty": 0.004,
        "loss_type": "sortino_dd",
        "dd_penalty": 4.0,
        "fill_temperature": 5e-4,
        "fill_buffer_pct": 0.0010,
        "decision_lag_bars": 1,
        "decision_lag_range": "0,1,2",
        "feature_noise_std": 0.01,
        "lr_schedule": "cosine",
        "lr_min_ratio": 0.10,
        "warmup_steps": 64,
        "validation_use_binary_fills": True,
    },
]


@dataclass(frozen=True)
class CandidateCheckpoint:
    path: Path
    score: float
    epoch: int
    run_name: str
    source: str


def parse_csv_symbols(raw: str) -> list[str]:
    symbols = [token.strip().upper() for token in raw.split(",") if token.strip()]
    if not symbols:
        raise ValueError("At least one symbol is required.")
    return symbols


def parse_csv_ints(raw: str) -> list[int]:
    values = [int(token.strip()) for token in raw.split(",") if token.strip()]
    if not values:
        raise ValueError("At least one integer value is required.")
    return values


def parse_symbol_map(raw: str | None, *, symbols: list[str]) -> dict[str, list[Path]]:
    result = {symbol: [] for symbol in symbols}
    if raw is None or not raw.strip():
        return result
    for spec in [token.strip() for token in raw.split(";") if token.strip()]:
        if "=" not in spec:
            raise ValueError(f"Expected SYMBOL=PATH|PATH mapping, got {spec!r}.")
        symbol_raw, values_raw = spec.split("=", 1)
        symbol = symbol_raw.strip().upper()
        if symbol not in result:
            raise ValueError(f"Unexpected symbol {symbol!r} in mapping.")
        paths = [_resolve_checkpoint_path(token.strip()) for token in values_raw.split("|") if token.strip()]
        if not paths:
            raise ValueError(f"No paths supplied for symbol {symbol!r}.")
        result[symbol].extend(paths)
    return result


def load_configs(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return [dict(row) for row in DEFAULT_CONFIGS]
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError("--configs-json must point to a JSON list.")
    configs: list[dict[str, Any]] = []
    for row in payload:
        if not isinstance(row, dict) or not row.get("name"):
            raise ValueError("Each config must be a JSON object with at least a non-empty 'name'.")
        configs.append(dict(row))
    return configs


def _checkpoint_metric(path: Path, metric_name: str) -> tuple[float, int]:
    payload = torch_load_compat(path, map_location="cpu", weights_only=False)
    metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
    epoch = int(payload.get("epoch", 0)) if isinstance(payload, dict) else 0
    metric_value = metrics.get(metric_name, float("-inf")) if isinstance(metrics, dict) else float("-inf")
    try:
        score = float(metric_value)
    except (TypeError, ValueError):
        score = float("-inf")
    return score, epoch


def collect_top_checkpoints(
    checkpoint_dir: Path,
    *,
    metric_name: str,
    top_k: int,
    run_name: str,
    source: str,
) -> list[CandidateCheckpoint]:
    checkpoint_paths = sorted(checkpoint_dir.glob("epoch_*.pt"))
    candidates: list[CandidateCheckpoint] = []
    for checkpoint_path in checkpoint_paths:
        score, epoch = _checkpoint_metric(checkpoint_path, metric_name)
        candidates.append(
            CandidateCheckpoint(
                path=checkpoint_path.resolve(),
                score=score,
                epoch=epoch,
                run_name=run_name,
                source=source,
            )
        )
    candidates.sort(key=lambda item: (item.score, item.epoch, item.path.name), reverse=True)
    return candidates[: max(1, int(top_k))]


def choose_symbol_candidates(
    candidate_groups: list[list[CandidateCheckpoint]],
    *,
    baseline_paths: list[Path],
    metric_name: str,
    max_candidates: int,
) -> list[CandidateCheckpoint]:
    merged: list[CandidateCheckpoint] = []
    seen: set[Path] = set()
    for group in candidate_groups:
        for candidate in group:
            if candidate.path in seen:
                continue
            seen.add(candidate.path)
            merged.append(candidate)
    for path in baseline_paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        score, epoch = _checkpoint_metric(resolved, metric_name)
        merged.append(
            CandidateCheckpoint(
                path=resolved,
                score=score,
                epoch=epoch,
                run_name=resolved.parent.name,
                source="baseline",
            )
        )
    merged.sort(key=lambda item: (item.score, item.epoch, item.path.name), reverse=True)
    return merged[: max(1, int(max_candidates))]


def build_candidate_spec(symbols: list[str], candidates_by_symbol: dict[str, list[CandidateCheckpoint]]) -> str:
    parts: list[str] = []
    for symbol in symbols:
        candidates = candidates_by_symbol.get(symbol, [])
        if not candidates:
            raise ValueError(f"No candidate checkpoints selected for {symbol}.")
        paths = "|".join(str(candidate.path) for candidate in candidates)
        parts.append(f"{symbol}={paths}")
    return ";".join(parts)


def _resolve_preload(
    symbol: str,
    config: dict[str, Any],
    preload_map: dict[str, list[Path]],
) -> Path | None:
    explicit = str(config.get("preload_checkpoint", "")).strip()
    if explicit:
        return _resolve_checkpoint_path(explicit)
    symbol_paths = preload_map.get(symbol, [])
    return symbol_paths[0] if symbol_paths else None


def train_one_run(
    *,
    symbol: str,
    seed: int,
    config: dict[str, Any],
    args: argparse.Namespace,
    preload_map: dict[str, list[Path]],
) -> tuple[Path, list[CandidateCheckpoint]]:
    timestamp = args.timestamp
    run_name = (
        f"{args.run_prefix}_{symbol.lower()}_{config['name']}_seed{seed}_{timestamp}"
        if args.timestamp_runs
        else f"{args.run_prefix}_{symbol.lower()}_{config['name']}_seed{seed}"
    )
    checkpoint_dir = args.checkpoint_root / run_name
    existing = sorted(checkpoint_dir.glob("epoch_*.pt"))
    if args.reuse_checkpoints and existing:
        logger.info("Reusing existing run {}", run_name)
        return checkpoint_dir, collect_top_checkpoints(
            checkpoint_dir,
            metric_name=args.checkpoint_metric,
            top_k=args.top_epochs_per_run,
            run_name=run_name,
            source="reused",
        )

    logger.info("Training {} seed={} config={}", symbol, seed, config["name"])
    preload_checkpoint = _resolve_preload(symbol, config, preload_map)
    if preload_checkpoint is not None:
        logger.info("Preloading {} from {}", symbol, preload_checkpoint)

    data = BinanceExp1DataModule(
        DatasetConfig(
            symbol=symbol,
            data_root=args.data_root,
            forecast_cache_root=args.forecast_cache_root,
            forecast_horizons=tuple(args.forecast_horizons),
            sequence_length=int(config.get("sequence_length", args.sequence_length)),
            validation_days=float(config.get("validation_days", args.validation_days)),
            cache_only=bool(args.cache_only),
        )
    )
    training_cfg = TrainingConfig(
        epochs=int(config.get("epochs", args.epochs)),
        batch_size=int(config.get("batch_size", args.batch_size)),
        sequence_length=int(config.get("sequence_length", args.sequence_length)),
        learning_rate=float(config.get("learning_rate", args.learning_rate)),
        weight_decay=float(config.get("weight_decay", args.weight_decay)),
        maker_fee=float(config.get("maker_fee", get_fee_for_symbol(symbol))),
        periods_per_year=float(config.get("periods_per_year", args.periods_per_year)),
        return_weight=float(config.get("return_weight", args.return_weight)),
        smoothness_penalty=float(config.get("smoothness_penalty", args.smoothness_penalty)),
        fill_temperature=float(config.get("fill_temperature", args.fill_temperature)),
        decision_lag_bars=int(config.get("decision_lag_bars", args.decision_lag_bars)),
        decision_lag_range=str(config.get("decision_lag_range", args.decision_lag_range)),
        fill_buffer_pct=float(config.get("fill_buffer_pct", args.fill_buffer_pct)),
        loss_type=str(config.get("loss_type", args.loss_type)),
        dd_penalty=float(config.get("dd_penalty", args.dd_penalty)),
        multiwindow_fractions=str(config.get("multiwindow_fractions", args.multiwindow_fractions)),
        multiwindow_aggregation=str(config.get("multiwindow_aggregation", args.multiwindow_aggregation)),
        feature_noise_std=float(config.get("feature_noise_std", args.feature_noise_std)),
        transformer_dim=int(config.get("transformer_dim", args.transformer_dim)),
        transformer_layers=int(config.get("transformer_layers", args.transformer_layers)),
        transformer_heads=int(config.get("transformer_heads", args.transformer_heads)),
        transformer_dropout=float(config.get("transformer_dropout", args.transformer_dropout)),
        model_arch=str(config.get("model_arch", args.model_arch)),
        num_memory_tokens=int(config.get("num_memory_tokens", args.num_memory_tokens)),
        dilated_strides=str(config.get("dilated_strides", args.dilated_strides)),
        lr_schedule=str(config.get("lr_schedule", args.lr_schedule)),
        lr_warmdown_ratio=float(config.get("lr_warmdown_ratio", args.lr_warmdown_ratio)),
        lr_min_ratio=float(config.get("lr_min_ratio", args.lr_min_ratio)),
        warmup_steps=int(config.get("warmup_steps", args.warmup_steps)),
        validation_use_binary_fills=bool(
            config.get("validation_use_binary_fills", args.validation_use_binary_fills)
        ),
        seed=seed,
        run_name=run_name,
        checkpoint_root=args.checkpoint_root,
        log_dir=args.log_dir,
        preload_checkpoint_path=str(preload_checkpoint) if preload_checkpoint is not None else None,
        use_compile=not bool(args.no_compile),
        dry_train_steps=args.dry_train_steps,
        num_workers=int(args.num_workers),
        top_k_checkpoints=max(50, int(config.get("epochs", args.epochs))),
    )
    trainer = BinanceHourlyTrainer(training_cfg, data)
    artifacts = trainer.train()
    if artifacts.best_checkpoint is None:
        raise RuntimeError(f"Training did not produce a checkpoint for {run_name}.")

    return checkpoint_dir, collect_top_checkpoints(
        checkpoint_dir,
        metric_name=args.checkpoint_metric,
        top_k=args.top_epochs_per_run,
        run_name=run_name,
        source="trained",
    )


def build_search_command(
    *,
    args: argparse.Namespace,
    candidate_spec: str,
    output_dir: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "binanceexp1.search_checkpoint_sets_robust",
        "--symbols",
        ",".join(args.symbols),
        "--candidate-checkpoints",
        candidate_spec,
        "--sequence-length",
        str(args.sequence_length),
        "--forecast-horizons",
        ",".join(str(value) for value in args.forecast_horizons),
        "--data-root",
        str(args.data_root),
        "--forecast-cache-root",
        str(args.forecast_cache_root),
        "--validation-days",
        str(float(args.validation_days)),
        "--window-hours",
        args.search_window_hours,
        "--initial-cash",
        str(float(args.initial_cash)),
        "--seed-position-fraction",
        str(float(args.seed_position_fraction)),
        "--default-intensity",
        str(float(args.default_intensity)),
        "--default-offset",
        str(float(args.default_offset)),
        "--min-edge",
        str(float(args.min_edge)),
        "--risk-weight",
        str(float(args.risk_weight)),
        "--edge-mode",
        str(args.edge_mode),
        "--max-hold-hours",
        str(int(args.max_hold_hours)),
        "--decision-lag-bars",
        str(int(args.search_decision_lag_bars)),
        "--fill-buffer-bps",
        str(float(args.search_fill_buffer_bps)),
        "--max-volume-fraction",
        str(float(args.max_volume_fraction)),
        "--max-concurrent-positions",
        str(int(args.max_concurrent_positions)),
        "--sortino-clip",
        str(float(args.sortino_clip)),
        "--min-trade-count-mean",
        str(float(args.min_trade_count_mean)),
        "--output-dir",
        str(output_dir),
    ]
    if args.cache_only:
        cmd.append("--cache-only")
    if args.realistic_selection:
        cmd.append("--realistic-selection")
    if args.require_all_positive:
        cmd.append("--require-all-positive")
    if args.offset_map:
        cmd.extend(["--offset-map", args.offset_map])
    if args.intensity_map:
        cmd.extend(["--intensity-map", args.intensity_map])
    if args.work_steal:
        cmd.append("--work-steal")
        cmd.extend(["--work-steal-min-profit-pct", str(float(args.work_steal_min_profit_pct))])
        cmd.extend(["--work-steal-min-edge", str(float(args.work_steal_min_edge))])
        cmd.extend(["--work-steal-edge-margin", str(float(args.work_steal_edge_margin))])
    return cmd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train robustness-focused BTC/ETH/SOL checkpoint candidates and run the robust combo search."
    )
    parser.add_argument("--symbols", default="BTCUSD,ETHUSD,SOLUSD")
    parser.add_argument("--seeds", default="42,2024")
    parser.add_argument("--configs-json", type=Path, default=None)
    parser.add_argument("--max-configs", type=int, default=0)
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--run-prefix", default="selector_robust")
    parser.add_argument("--checkpoint-root", type=Path, default=Path("binanceneural/checkpoints"))
    parser.add_argument("--log-dir", type=Path, default=Path("tensorboard_logs/binanceneural"))
    parser.add_argument("--data-root", type=Path, default=Path(DatasetConfig().data_root))
    parser.add_argument("--forecast-cache-root", type=Path, default=Path(DatasetConfig().forecast_cache_root))
    parser.add_argument("--forecast-horizons", default="1,24")
    parser.add_argument("--validation-days", type=float, default=30.0)
    parser.add_argument("--sequence-length", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--return-weight", type=float, default=0.10)
    parser.add_argument("--smoothness-penalty", type=float, default=0.003)
    parser.add_argument("--loss-type", default="multiwindow_dd")
    parser.add_argument("--dd-penalty", type=float, default=3.0)
    parser.add_argument("--multiwindow-fractions", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--multiwindow-aggregation", default="minimax")
    parser.add_argument("--fill-temperature", type=float, default=5e-4)
    parser.add_argument("--fill-buffer-pct", type=float, default=0.0010)
    parser.add_argument("--decision-lag-bars", type=int, default=1)
    parser.add_argument("--decision-lag-range", default="0,1,2")
    parser.add_argument("--feature-noise-std", type=float, default=0.01)
    parser.add_argument("--transformer-dim", type=int, default=256)
    parser.add_argument("--transformer-layers", type=int, default=4)
    parser.add_argument("--transformer-heads", type=int, default=8)
    parser.add_argument("--transformer-dropout", type=float, default=0.1)
    parser.add_argument("--model-arch", default="classic")
    parser.add_argument("--num-memory-tokens", type=int, default=0)
    parser.add_argument("--dilated-strides", default="")
    parser.add_argument("--lr-schedule", default="cosine")
    parser.add_argument("--lr-warmdown-ratio", type=float, default=0.5)
    parser.add_argument("--lr-min-ratio", type=float, default=0.10)
    parser.add_argument("--warmup-steps", type=int, default=64)
    parser.add_argument("--periods-per-year", type=float, default=24.0 * 365.0)
    parser.add_argument("--validation-use-binary-fills", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--reuse-checkpoints", action="store_true")
    parser.add_argument("--dry-train-steps", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-metric", default="score")
    parser.add_argument("--top-epochs-per-run", type=int, default=2)
    parser.add_argument("--max-candidates-per-symbol", type=int, default=6)
    parser.add_argument(
        "--preload-checkpoints",
        default=None,
        help="Optional semicolon-separated SYMBOL=PATH|PATH mapping. First path per symbol is used for preload.",
    )
    parser.add_argument(
        "--baseline-candidates",
        default=None,
        help="Optional semicolon-separated SYMBOL=PATH|PATH mapping added directly to the robust search pool.",
    )
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--search-window-hours", default="336")
    parser.add_argument("--initial-cash", type=float, default=10_000.0)
    parser.add_argument("--seed-position-fraction", type=float, default=1.0)
    parser.add_argument("--default-intensity", type=float, default=6.0)
    parser.add_argument("--default-offset", type=float, default=0.0)
    parser.add_argument("--intensity-map", default=None)
    parser.add_argument("--offset-map", default="ETHUSD=0.0003,SOLUSD=0.0005")
    parser.add_argument("--min-edge", type=float, default=0.0015)
    parser.add_argument("--risk-weight", type=float, default=0.25)
    parser.add_argument("--edge-mode", default="high_low", choices=["high_low", "high", "close"])
    parser.add_argument("--max-hold-hours", type=int, default=6)
    parser.add_argument("--search-decision-lag-bars", type=int, default=2)
    parser.add_argument("--search-fill-buffer-bps", type=float, default=20.0)
    parser.add_argument("--max-volume-fraction", type=float, default=0.1)
    parser.add_argument("--max-concurrent-positions", type=int, default=1)
    parser.add_argument("--sortino-clip", type=float, default=10.0)
    parser.add_argument("--min-trade-count-mean", type=float, default=6.0)
    parser.add_argument("--realistic-selection", action="store_true")
    parser.add_argument("--require-all-positive", action="store_true")
    parser.add_argument("--work-steal", action="store_true")
    parser.add_argument("--work-steal-min-profit-pct", type=float, default=0.001)
    parser.add_argument("--work-steal-min-edge", type=float, default=0.005)
    parser.add_argument("--work-steal-edge-margin", type=float, default=0.0)
    parser.add_argument("--search-output-dir", type=Path, default=None)
    parser.add_argument("--no-run-search", action="store_true")
    parser.add_argument("--timestamp-runs", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.symbols = parse_csv_symbols(args.symbols)
    args.seeds = parse_csv_ints(args.seeds)
    args.forecast_horizons = tuple(parse_csv_ints(args.forecast_horizons))
    args.timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    configs = load_configs(args.configs_json)
    if args.max_configs > 0:
        configs = configs[: int(args.max_configs)]
    if not configs:
        raise ValueError("No training configs selected.")

    experiment_name = args.experiment_name or f"binance_selector_robust_train_{args.timestamp}"
    experiment_dir = Path("experiments") / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_root.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    preload_map = parse_symbol_map(args.preload_checkpoints, symbols=args.symbols)
    baseline_map = parse_symbol_map(args.baseline_candidates, symbols=args.symbols)
    all_runs: list[dict[str, Any]] = []
    candidate_groups_by_symbol: dict[str, list[list[CandidateCheckpoint]]] = {symbol: [] for symbol in args.symbols}

    run_counter = 0
    for symbol in args.symbols:
        for config in configs:
            for seed in args.seeds:
                run_counter += 1
                if args.max_runs > 0 and run_counter > int(args.max_runs):
                    break
                checkpoint_dir, top_candidates = train_one_run(
                    symbol=symbol,
                    seed=int(seed),
                    config=config,
                    args=args,
                    preload_map=preload_map,
                )
                candidate_groups_by_symbol[symbol].append(top_candidates)
                all_runs.append(
                    {
                        "symbol": symbol,
                        "seed": int(seed),
                        "config_name": str(config["name"]),
                        "checkpoint_dir": str(checkpoint_dir),
                        "top_candidates": [
                            {
                                "path": str(candidate.path),
                                "score": candidate.score,
                                "epoch": candidate.epoch,
                                "source": candidate.source,
                            }
                            for candidate in top_candidates
                        ],
                    }
                )
            if args.max_runs > 0 and run_counter >= int(args.max_runs):
                break
        if args.max_runs > 0 and run_counter >= int(args.max_runs):
            break

    selected_candidates: dict[str, list[CandidateCheckpoint]] = {}
    for symbol in args.symbols:
        selected_candidates[symbol] = choose_symbol_candidates(
            candidate_groups_by_symbol[symbol],
            baseline_paths=baseline_map.get(symbol, []),
            metric_name=args.checkpoint_metric,
            max_candidates=int(args.max_candidates_per_symbol),
        )

    candidate_spec = build_candidate_spec(args.symbols, selected_candidates)
    (experiment_dir / "candidate_checkpoints.txt").write_text(candidate_spec + "\n")
    manifest = {
        "experiment_name": experiment_name,
        "timestamp": args.timestamp,
        "symbols": args.symbols,
        "seeds": args.seeds,
        "configs": configs,
        "runs": all_runs,
        "selected_candidates": {
            symbol: [
                {
                    "path": str(candidate.path),
                    "score": candidate.score,
                    "epoch": candidate.epoch,
                    "run_name": candidate.run_name,
                    "source": candidate.source,
                }
                for candidate in candidates
            ]
            for symbol, candidates in selected_candidates.items()
        },
        "candidate_spec": candidate_spec,
    }
    (experiment_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if args.no_run_search:
        logger.info("Training complete; robust search skipped.")
        return

    search_output_dir = args.search_output_dir or (experiment_dir / "search")
    cmd = build_search_command(args=args, candidate_spec=candidate_spec, output_dir=search_output_dir)
    logger.info("Running robust combo search: {}", " ".join(cmd))
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Robust combo search failed with exit code {completed.returncode}.")

    ranking_path = search_output_dir / "ranking.csv"
    manifest["search_output_dir"] = str(search_output_dir)
    manifest["search_ranking_csv"] = str(ranking_path)
    if ranking_path.exists():
        import pandas as pd

        ranking = pd.read_csv(ranking_path)
        if not ranking.empty:
            best = ranking.iloc[0].to_dict()
            manifest["best_combo"] = best
            logger.info(
                "Best combo {} | score={:.3f} | worst_ret={:+.2f}% | mean_ret={:+.2f}% | worst_dd={:.2f}%",
                best.get("combo", ""),
                float(best.get("selection_score", 0.0)),
                float(best.get("return_worst_pct", 0.0)),
                float(best.get("return_mean_pct", 0.0)),
                float(best.get("max_drawdown_worst_pct", 0.0)),
            )
    (experiment_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
