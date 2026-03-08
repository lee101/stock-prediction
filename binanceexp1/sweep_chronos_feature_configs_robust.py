from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from .joint_chronos_forecast_cache import build_joint_forecast_cache


@dataclass(frozen=True)
class ChronosFeatureConfig:
    name: str
    forecast_horizons: tuple[int, ...]
    context_hours: int
    batch_size: int = 128
    force_multivariate: bool | None = None
    force_cross_learning: bool | None = None
    force_multiscale: bool | None = None
    skip_rates: tuple[int, ...] = (1,)
    aggregation_method: str | None = None
    grouped_joint_cache: bool = False
    use_time_covariates: bool = False
    description: str = ""


def parse_csv_list(raw: str) -> list[str]:
    return [token.strip() for token in str(raw).split(",") if token.strip()]


def parse_csv_ints(raw: str) -> tuple[int, ...]:
    values = tuple(int(token) for token in parse_csv_list(raw))
    if not values:
        raise ValueError(f"Expected at least one integer in {raw!r}.")
    return values


def parse_symbols(raw: str) -> list[str]:
    symbols = [token.upper() for token in parse_csv_list(raw)]
    if not symbols:
        raise ValueError("At least one symbol is required.")
    return symbols


def _default_feature_configs() -> list[ChronosFeatureConfig]:
    return [
        ChronosFeatureConfig(
            name="baseline_h1_24_ctx336",
            forecast_horizons=(1, 24),
            context_hours=24 * 14,
            batch_size=128,
            description="Current baseline-style Chronos feature mix.",
        ),
        ChronosFeatureConfig(
            name="h6_only_ctx1024",
            forecast_horizons=(6,),
            context_hours=1024,
            batch_size=64,
            description="Single 6h forecast horizon with longer context.",
        ),
        ChronosFeatureConfig(
            name="h1_6_24_ctx1024",
            forecast_horizons=(1, 6, 24),
            context_hours=1024,
            batch_size=64,
            description="Blend 1h, 6h, and 24h horizons with longer context.",
        ),
        ChronosFeatureConfig(
            name="h1_6_24_ctx1024_multivar",
            forecast_horizons=(1, 6, 24),
            context_hours=1024,
            batch_size=64,
            force_multivariate=True,
            description="Longer context plus forced OHLC multivariate Chronos inference.",
        ),
        ChronosFeatureConfig(
            name="h1_24_ctx1024_ms124_weighted",
            forecast_horizons=(1, 24),
            context_hours=1024,
            batch_size=64,
            force_multiscale=True,
            skip_rates=(1, 2, 4),
            aggregation_method="weighted",
            description="Longer context with multi-scale Chronos aggregation.",
        ),
        ChronosFeatureConfig(
            name="h1_6_24_joint_ctx1024_cov",
            forecast_horizons=(1, 6, 24),
            context_hours=1024,
            batch_size=96,
            force_cross_learning=True,
            grouped_joint_cache=True,
            use_time_covariates=True,
            description="Cross-symbol joint Chronos cache generation with time covariates.",
        ),
    ]


def load_feature_configs(path: Path | None) -> list[ChronosFeatureConfig]:
    if path is None:
        return _default_feature_configs()
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError("--feature-configs-json must point to a JSON list.")
    configs: list[ChronosFeatureConfig] = []
    for row in payload:
        if not isinstance(row, dict) or not str(row.get("name", "")).strip():
            raise ValueError("Each feature config must be a JSON object with a non-empty 'name'.")
        horizons = row.get("forecast_horizons")
        if horizons is None:
            raise ValueError(f"Feature config {row.get('name')!r} is missing forecast_horizons.")
        config = ChronosFeatureConfig(
            name=str(row["name"]),
            forecast_horizons=tuple(int(value) for value in horizons),
            context_hours=int(row.get("context_hours", 24 * 14)),
            batch_size=int(row.get("batch_size", 128)),
            force_multivariate=row.get("force_multivariate"),
            force_cross_learning=row.get("force_cross_learning"),
            force_multiscale=row.get("force_multiscale"),
            skip_rates=tuple(int(value) for value in row.get("skip_rates", [1])),
            aggregation_method=str(row["aggregation_method"]) if row.get("aggregation_method") else None,
            grouped_joint_cache=bool(row.get("grouped_joint_cache", False)),
            use_time_covariates=bool(row.get("use_time_covariates", False)),
            description=str(row.get("description", "")),
        )
        configs.append(config)
    return configs


def build_env_overrides(config: ChronosFeatureConfig) -> dict[str, str]:
    env: dict[str, str] = {
        "CHRONOS2_CONTEXT_HOURS": str(int(config.context_hours)),
        "CHRONOS2_CONTEXT_LENGTH": str(int(config.context_hours)),
        "CHRONOS2_BATCH_SIZE": str(int(config.batch_size)),
    }
    if config.force_multivariate is not None:
        env["CHRONOS2_FORCE_MULTIVARIATE"] = "1" if config.force_multivariate else "0"
    if config.force_cross_learning is not None:
        env["CHRONOS2_FORCE_CROSS_LEARNING"] = "1" if config.force_cross_learning else "0"
    if config.force_multiscale is not None:
        env["CHRONOS2_FORCE_MULTISCALE"] = "1" if config.force_multiscale else "0"
    if config.skip_rates:
        env["CHRONOS2_SKIP_RATES"] = ",".join(str(int(value)) for value in config.skip_rates)
    if config.aggregation_method:
        env["CHRONOS2_AGGREGATION_METHOD"] = str(config.aggregation_method)
    return env


def build_train_command(
    *,
    args: argparse.Namespace,
    config: ChronosFeatureConfig,
    feature_experiment_name: str,
    forecast_cache_root: Path,
    cache_only: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "binanceexp1.train_multiasset_selector_robust",
        "--symbols",
        ",".join(args.symbols),
        "--seeds",
        args.seeds,
        "--forecast-horizons",
        ",".join(str(value) for value in config.forecast_horizons),
        "--data-root",
        str(args.data_root),
        "--forecast-cache-root",
        str(forecast_cache_root),
        "--validation-days",
        str(float(args.validation_days)),
        "--sequence-length",
        str(int(args.sequence_length)),
        "--experiment-name",
        feature_experiment_name,
        "--search-window-hours",
        str(args.search_window_hours),
        "--max-configs",
        str(int(args.max_train_configs)),
        "--top-epochs-per-run",
        str(int(args.top_epochs_per_run)),
        "--max-candidates-per-symbol",
        str(int(args.max_candidates_per_symbol)),
        "--min-trade-count-mean",
        str(float(args.min_trade_count_mean)),
    ]
    if args.training_configs_json is not None:
        cmd.extend(["--configs-json", str(args.training_configs_json)])
    if args.preload_checkpoints:
        cmd.extend(["--preload-checkpoints", args.preload_checkpoints])
    if args.baseline_candidates:
        cmd.extend(["--baseline-candidates", args.baseline_candidates])
    if args.offset_map:
        cmd.extend(["--offset-map", args.offset_map])
    if args.intensity_map:
        cmd.extend(["--intensity-map", args.intensity_map])
    if args.realistic_selection:
        cmd.append("--realistic-selection")
    if args.require_all_positive:
        cmd.append("--require-all-positive")
    if args.work_steal:
        cmd.append("--work-steal")
    if args.no_compile:
        cmd.append("--no-compile")
    if args.reuse_checkpoints:
        cmd.append("--reuse-checkpoints")
    if args.validation_use_binary_fills:
        cmd.append("--validation-use-binary-fills")
    if cache_only:
        cmd.append("--cache-only")
    if args.dry_train_steps is not None:
        cmd.extend(["--dry-train-steps", str(int(args.dry_train_steps))])
    if args.epochs is not None:
        cmd.extend(["--epochs", str(int(args.epochs))])
    if args.batch_size is not None:
        cmd.extend(["--batch-size", str(int(args.batch_size))])
    if args.learning_rate is not None:
        cmd.extend(["--learning-rate", str(float(args.learning_rate))])
    if args.run_prefix:
        cmd.extend(["--run-prefix", str(args.run_prefix)])
    return cmd


def seed_forecast_cache(
    *,
    seed_root: Path,
    dest_root: Path,
    symbols: list[str],
    horizons: tuple[int, ...],
) -> int:
    copied = 0
    for horizon in horizons:
        for symbol in symbols:
            source = Path(seed_root) / f"h{int(horizon)}" / f"{symbol.upper()}.parquet"
            if not source.exists():
                continue
            target = Path(dest_root) / f"h{int(horizon)}" / f"{symbol.upper()}.parquet"
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied += 1
    return copied


def has_complete_forecast_cache(
    *,
    cache_root: Path,
    symbols: list[str],
    horizons: tuple[int, ...],
) -> bool:
    for horizon in horizons:
        for symbol in symbols:
            path = Path(cache_root) / f"h{int(horizon)}" / f"{symbol.upper()}.parquet"
            if not path.exists():
                return False
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep Chronos feature configurations through the robust BTC/ETH/SOL selector training pipeline."
    )
    parser.add_argument("--symbols", default="BTCUSD,ETHUSD,SOLUSD")
    parser.add_argument("--feature-configs-json", type=Path, default=None)
    parser.add_argument("--max-feature-configs", type=int, default=0)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--data-root", type=Path, default=Path("trainingdatahourly") / "crypto")
    parser.add_argument("--validation-days", type=float, default=30.0)
    parser.add_argument("--sequence-length", type=int, default=96)
    parser.add_argument("--training-configs-json", type=Path, default=None)
    parser.add_argument("--max-train-configs", type=int, default=1)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--preload-checkpoints", default=None)
    parser.add_argument("--baseline-candidates", default=None)
    parser.add_argument("--search-window-hours", default="336")
    parser.add_argument("--top-epochs-per-run", type=int, default=2)
    parser.add_argument("--max-candidates-per-symbol", type=int, default=6)
    parser.add_argument("--min-trade-count-mean", type=float, default=6.0)
    parser.add_argument("--offset-map", default="ETHUSD=0.0003,SOLUSD=0.0005")
    parser.add_argument("--intensity-map", default=None)
    parser.add_argument("--realistic-selection", action="store_true")
    parser.add_argument("--require-all-positive", action="store_true")
    parser.add_argument("--work-steal", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--reuse-checkpoints", action="store_true")
    parser.add_argument("--validation-use-binary-fills", action="store_true")
    parser.add_argument("--dry-train-steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--run-prefix", default="chronos_feature_sweep")
    parser.add_argument("--force-rebuild-joint-cache", action="store_true")
    parser.add_argument("--seed-forecast-cache-root", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.symbols = parse_symbols(args.symbols)

    configs = load_feature_configs(args.feature_configs_json)
    if args.max_feature_configs > 0:
        configs = configs[: int(args.max_feature_configs)]
    if not configs:
        raise ValueError("No Chronos feature configurations selected.")

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    experiment_name = args.experiment_name or f"binance_selector_chronos_feature_sweep_{timestamp}"
    experiment_dir = Path("experiments") / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []

    for config in configs:
        feature_experiment_name = f"{experiment_name}/{config.name}"
        forecast_cache_root = experiment_dir / "forecast_cache" / config.name
        forecast_cache_root.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update(build_env_overrides(config))
        cache_only = bool(config.grouped_joint_cache)

        if args.seed_forecast_cache_root is not None:
            copied = seed_forecast_cache(
                seed_root=Path(args.seed_forecast_cache_root),
                dest_root=forecast_cache_root,
                symbols=args.symbols,
                horizons=config.forecast_horizons,
            )
            if copied:
                logger.info("Seeded {} forecast cache files for {}", copied, config.name)
            if has_complete_forecast_cache(
                cache_root=forecast_cache_root,
                symbols=args.symbols,
                horizons=config.forecast_horizons,
            ):
                cache_only = True

        joint_cache_summary: dict[str, dict[str, int]] | None = None
        if config.grouped_joint_cache:
            logger.info(
                "Building grouped Chronos caches for {} with horizons {}",
                config.name,
                config.forecast_horizons,
            )
            joint_cache_summary = build_joint_forecast_cache(
                symbols=args.symbols,
                data_root=Path(args.data_root),
                cache_root=forecast_cache_root,
                horizons=config.forecast_horizons,
                context_hours=int(config.context_hours),
                batch_size=int(config.batch_size),
                use_cross_learning=bool(config.force_cross_learning),
                use_time_covariates=bool(config.use_time_covariates),
                force_rebuild=bool(args.force_rebuild_joint_cache),
            )

        cmd = build_train_command(
            args=args,
            config=config,
            feature_experiment_name=feature_experiment_name,
            forecast_cache_root=forecast_cache_root,
            cache_only=cache_only,
        )
        logger.info("Running Chronos feature config {}: {}", config.name, " ".join(cmd))
        completed = subprocess.run(cmd, check=False, env=env)

        feature_dir = Path("experiments") / feature_experiment_name
        ranking_path = feature_dir / "search" / "ranking.csv"
        scenarios_path = feature_dir / "search" / "scenarios.csv"
        feature_manifest_path = feature_dir / "manifest.json"
        best_metrics: dict[str, Any] = {}
        if ranking_path.exists():
            ranking = pd.read_csv(ranking_path)
            if not ranking.empty:
                best_metrics = ranking.iloc[0].to_dict()
        row = {
            "name": config.name,
            "forecast_horizons": ",".join(str(value) for value in config.forecast_horizons),
            "context_hours": int(config.context_hours),
            "batch_size": int(config.batch_size),
            "force_multivariate": config.force_multivariate,
            "force_cross_learning": config.force_cross_learning,
            "force_multiscale": config.force_multiscale,
            "skip_rates": ",".join(str(int(value)) for value in config.skip_rates),
            "aggregation_method": config.aggregation_method or "",
            "grouped_joint_cache": bool(config.grouped_joint_cache),
            "use_time_covariates": bool(config.use_time_covariates),
            "returncode": int(completed.returncode),
            "feature_dir": str(feature_dir),
            "ranking_csv": str(ranking_path),
            "scenarios_csv": str(scenarios_path),
            **best_metrics,
        }
        summary_rows.append(row)
        run_rows.append(
            {
                "config": asdict(config),
                "command": cmd,
                "env_overrides": build_env_overrides(config),
                "returncode": int(completed.returncode),
                "feature_manifest": str(feature_manifest_path),
                "joint_cache_summary": joint_cache_summary,
            }
        )

        if completed.returncode != 0:
            logger.warning("Chronos feature config {} failed with exit code {}", config.name, completed.returncode)

    ranking_out = experiment_dir / "ranking.csv"
    runs_out = experiment_dir / "manifest.json"
    ranking_df = pd.DataFrame(summary_rows)
    if not ranking_df.empty:
        sort_columns = [column for column in ("selection_score", "return_worst_pct", "return_mean_pct") if column in ranking_df.columns]
        if sort_columns:
            ranking_df = ranking_df.sort_values(sort_columns, ascending=[False] * len(sort_columns)).reset_index(drop=True)
        ranking_df.to_csv(ranking_out, index=False)

    runs_out.write_text(
        json.dumps(
            {
                "experiment_name": experiment_name,
                "timestamp": timestamp,
                "symbols": args.symbols,
                "runs": run_rows,
                "ranking_csv": str(ranking_out),
            },
            indent=2,
        )
    )
    if not ranking_df.empty:
        best = ranking_df.iloc[0].to_dict()
        logger.info(
            "Best Chronos feature config {} | score={} | worst_ret={} | mean_ret={}",
            best.get("name"),
            best.get("selection_score"),
            best.get("return_worst_pct"),
            best.get("return_mean_pct"),
        )


if __name__ == "__main__":
    main()
