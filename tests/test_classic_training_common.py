from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import unified_hourly_experiment.classic_training_common as classic_training_common
from unified_hourly_experiment.jax_classic_defaults import compute_jax_classic_min_history_hours


def _build_args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        data_root=tmp_path / "data",
        cache_root=tmp_path / "cache",
        checkpoint_root=tmp_path / "ckpts",
        log_dir=tmp_path / "logs",
        run_name="classic_shared",
        epochs=3,
        batch_size=4,
        sequence_length=48,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        learning_rate=1e-4,
        weight_decay=0.05,
        grad_clip=0.5,
        return_weight=0.2,
        smoothness_penalty=0.01,
        maker_fee=0.001,
        fill_temperature=5e-4,
        validation_days=30,
        cache_only=False,
        seed=123,
        max_hold_hours=5.0,
        max_leverage=2.0,
        margin_annual_rate=0.0625,
        decision_lag_bars=1,
        market_order_entry=True,
        fill_buffer_pct=0.0005,
        preload=None,
        dry_train_steps=2,
    )


def test_parse_horizons_rejects_empty_tokens() -> None:
    try:
        classic_training_common.parse_horizons(" , ")
    except ValueError as exc:
        assert "At least one forecast horizon" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected ValueError for empty forecast horizons")


def test_build_classic_dataset_config_uses_shared_history_formula(tmp_path: Path) -> None:
    args = _build_args(tmp_path)

    dataset_cfg = classic_training_common.build_classic_dataset_config(
        args,
        symbols=["AAPL", "QQQ"],
        horizons=(1, 4),
    )

    assert dataset_cfg.symbol == "AAPL"
    assert dataset_cfg.forecast_horizons == (1, 4)
    assert dataset_cfg.cache_only is False
    assert dataset_cfg.min_history_hours == compute_jax_classic_min_history_hours(48, 30)


def test_build_classic_training_config_applies_backend_and_overrides(tmp_path: Path) -> None:
    args = _build_args(tmp_path)

    train_cfg = classic_training_common.build_classic_training_config(
        args,
        backend="jax_classic",
        checkpoint_root=tmp_path / "out" / "checkpoints",
        log_dir=tmp_path / "out" / "logs",
        run_name="compare_jax_classic_seed7",
        extra_kwargs={
            "seed": 7,
            "use_compile": False,
            "wandb_mode": "disabled",
            "wandb_log_metrics": False,
        },
    )

    assert train_cfg.trainer_backend == "jax_classic"
    assert train_cfg.model_arch == "classic"
    assert train_cfg.seed == 7
    assert train_cfg.use_compile is False
    assert train_cfg.market_order_entry is True
    assert train_cfg.checkpoint_root == tmp_path / "out" / "checkpoints"
    assert train_cfg.log_dir == tmp_path / "out" / "logs"


def test_build_classic_data_module_uses_explicit_constraints(tmp_path: Path) -> None:
    args = _build_args(tmp_path)
    captured: dict[str, object] = {}

    class _FakeDataModule:
        def __init__(self, *, symbols, config, directional_constraints) -> None:
            captured["symbols"] = list(symbols)
            captured["config"] = config
            captured["directional_constraints"] = dict(directional_constraints)

    dataset_cfg, data_module = classic_training_common.build_classic_data_module(
        args,
        symbols=["AAPL", "QQQ"],
        horizons=(1, 4),
        data_module_cls=_FakeDataModule,
        directional_constraints={"AAPL": (1.0, 0.0), "QQQ": (1.0, 1.0)},
    )

    assert captured["symbols"] == ["AAPL", "QQQ"]
    assert captured["config"] is dataset_cfg
    assert captured["directional_constraints"] == {"AAPL": (1.0, 0.0), "QQQ": (1.0, 1.0)}
    assert data_module is not None


def test_build_classic_data_module_adds_forecast_refresh_hint_on_cache_only_failure(tmp_path: Path) -> None:
    args = _build_args(tmp_path)
    args.cache_only = True

    class _BoomModule:
        def __init__(self, **_kwargs) -> None:
            raise RuntimeError("missing forecast cache")

    try:
        classic_training_common.build_classic_data_module(
            args,
            symbols=["AAPL", "QQQ"],
            horizons=(1, 4),
            data_module_cls=_BoomModule,
        )
    except RuntimeError as exc:
        assert "missing forecast cache" in str(exc)
        assert "--allow-forecast-refresh" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected RuntimeError with forecast refresh hint")


def test_render_classic_run_plan_summary_includes_compare_fields() -> None:
    summary = classic_training_common.render_classic_run_plan_summary(
        {
            "symbols": ["AAPL", "QQQ"],
            "backends": ["torch", "jax_classic"],
            "seeds": [7, 9],
            "forecast_horizons": [1, 4],
            "cache_only": True,
            "output_dir": "/tmp/out",
            "training": {
                "epochs": 2,
                "dry_train_steps": 3,
                "batch_size": 4,
                "sequence_length": 48,
                "validation_days": 30,
            },
        },
        title="Backend Compare Plan",
    )

    assert "Backend Compare Plan" in summary
    assert "Symbols: AAPL,QQQ" in summary
    assert "Backends: torch,jax_classic" in summary
    assert "Seeds: 7,9" in summary
    assert "Forecast horizons: 1,4" in summary
    assert "Output dir: /tmp/out" in summary
    assert "Training: epochs=2, dry_train_steps=3, batch_size=4, sequence_length=48, validation_days=30" in summary


def test_write_effective_args_artifacts_writes_json_and_rerun_file(tmp_path: Path) -> None:
    parser = classic_training_common.ArgsFileParser(fromfile_prefix_chars="@")
    parser.add_argument("--symbols", default="aapl,qqq")
    parser.add_argument("--cache-only", dest="cache_only", action="store_true", default=True)
    parser.add_argument("--allow-forecast-refresh", dest="cache_only", action="store_false")
    parser.add_argument("--epochs", type=int, default=3)

    args = parser.parse_args(["--symbols", "aapl,dbx", "--allow-forecast-refresh", "--epochs", "5"])

    json_path, txt_path = classic_training_common.write_effective_args_artifacts(
        parser,
        args,
        tmp_path,
        module_name="unified_hourly_experiment.train_jax_classic",
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    rerun_text = txt_path.read_text(encoding="utf-8")

    assert payload["symbols"] == "aapl,dbx"
    assert payload["cache_only"] is False
    assert payload["epochs"] == 5
    assert "python -m unified_hourly_experiment.train_jax_classic @effective_args.txt" in rerun_text
    assert "--symbols aapl,dbx" in rerun_text
    assert "--allow-forecast-refresh" in rerun_text
