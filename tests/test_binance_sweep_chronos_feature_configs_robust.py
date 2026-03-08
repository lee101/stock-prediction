from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from binanceexp1.sweep_chronos_feature_configs_robust import (
    ChronosFeatureConfig,
    build_env_overrides,
    build_train_command,
    has_complete_forecast_cache,
    load_feature_configs,
    seed_forecast_cache,
)


def test_load_feature_configs_reads_json(tmp_path: Path) -> None:
    path = tmp_path / "feature_configs.json"
    path.write_text(
        json.dumps(
            [
                {
                    "name": "h1_6_24_joint",
                    "forecast_horizons": [1, 6, 24],
                    "context_hours": 1024,
                    "batch_size": 64,
                    "force_cross_learning": True,
                    "grouped_joint_cache": True,
                    "use_time_covariates": True,
                }
            ]
        )
    )

    configs = load_feature_configs(path)

    assert len(configs) == 1
    assert configs[0].forecast_horizons == (1, 6, 24)
    assert configs[0].grouped_joint_cache is True
    assert configs[0].use_time_covariates is True


def test_build_env_overrides_formats_chronos_knobs() -> None:
    env = build_env_overrides(
        ChronosFeatureConfig(
            name="ms",
            forecast_horizons=(1, 24),
            context_hours=1024,
            batch_size=16,
            force_multivariate=True,
            force_cross_learning=False,
            force_multiscale=True,
            skip_rates=(1, 2, 4),
            aggregation_method="weighted",
        )
    )

    assert env["CHRONOS2_CONTEXT_HOURS"] == "1024"
    assert env["CHRONOS2_CONTEXT_LENGTH"] == "1024"
    assert env["CHRONOS2_BATCH_SIZE"] == "16"
    assert env["CHRONOS2_FORCE_MULTIVARIATE"] == "1"
    assert env["CHRONOS2_FORCE_CROSS_LEARNING"] == "0"
    assert env["CHRONOS2_FORCE_MULTISCALE"] == "1"
    assert env["CHRONOS2_SKIP_RATES"] == "1,2,4"
    assert env["CHRONOS2_AGGREGATION_METHOD"] == "weighted"


def test_build_train_command_uses_feature_cache_root() -> None:
    args = SimpleNamespace(
        symbols=["BTCUSD", "ETHUSD", "SOLUSD"],
        seeds="42",
        data_root=Path("trainingdatahourly/crypto"),
        validation_days=30.0,
        sequence_length=96,
        search_window_hours="336",
        max_train_configs=1,
        top_epochs_per_run=2,
        max_candidates_per_symbol=6,
        min_trade_count_mean=6.0,
        training_configs_json=None,
        preload_checkpoints="BTCUSD=a.pt;ETHUSD=b.pt;SOLUSD=c.pt",
        baseline_candidates="BTCUSD=a.pt;ETHUSD=b.pt;SOLUSD=c.pt",
        offset_map=None,
        intensity_map=None,
        realistic_selection=True,
        require_all_positive=False,
        work_steal=False,
        no_compile=True,
        reuse_checkpoints=True,
        validation_use_binary_fills=True,
        dry_train_steps=8,
        epochs=None,
        batch_size=None,
        learning_rate=None,
        run_prefix="chronos_feature_sweep",
    )

    cmd = build_train_command(
        args=args,
        config=ChronosFeatureConfig(
            name="joint",
            forecast_horizons=(1, 6, 24),
            context_hours=1024,
            grouped_joint_cache=True,
        ),
        feature_experiment_name="exp/joint",
        forecast_cache_root=Path("experiments/exp/cache/joint"),
        cache_only=True,
    )

    joined = " ".join(cmd)
    assert "--forecast-cache-root experiments/exp/cache/joint" in joined
    assert "--forecast-horizons 1,6,24" in joined
    assert "--cache-only" in joined
    assert "--realistic-selection" in joined


def test_seed_forecast_cache_copies_matching_symbol_horizon_files(tmp_path: Path) -> None:
    seed_root = tmp_path / "seed"
    dest_root = tmp_path / "dest"
    (seed_root / "h1").mkdir(parents=True)
    (seed_root / "h24").mkdir(parents=True)
    (seed_root / "h1" / "BTCUSD.parquet").write_text("btc")
    (seed_root / "h24" / "BTCUSD.parquet").write_text("btc24")

    copied = seed_forecast_cache(
        seed_root=seed_root,
        dest_root=dest_root,
        symbols=["BTCUSD", "ETHUSD"],
        horizons=(1, 24),
    )

    assert copied == 2
    assert (dest_root / "h1" / "BTCUSD.parquet").exists()
    assert (dest_root / "h24" / "BTCUSD.parquet").exists()
    assert has_complete_forecast_cache(
        cache_root=dest_root,
        symbols=["BTCUSD", "ETHUSD"],
        horizons=(1, 24),
    ) is False
