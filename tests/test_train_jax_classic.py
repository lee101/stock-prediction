from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import unified_hourly_experiment.train_jax_classic as train_jax_classic
from unified_hourly_experiment.jax_classic_defaults import (
    DEFAULT_JAX_CLASSIC_SYMBOLS_CSV,
    JAX_CLASSIC_DEFAULT_BATCH_SIZE,
    JAX_CLASSIC_DEFAULT_NUM_OUTPUTS,
    JAX_CLASSIC_DEFAULT_SEQUENCE_LENGTH,
    JAX_CLASSIC_DEFAULT_TRAIN_EPOCHS,
    compute_jax_classic_min_history_hours,
)


def test_build_directional_constraints_uses_default_symbol_groups() -> None:
    constraints = train_jax_classic.build_directional_constraints(["AAPL", "DBX", "QQQ"])

    assert constraints["AAPL"] == (1.0, 0.0)
    assert constraints["DBX"] == (0.0, 1.0)
    assert constraints["QQQ"] == (1.0, 1.0)


def test_parse_args_uses_shared_default_symbols(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["train_jax_classic.py"])
    args = train_jax_classic.parse_args()
    assert args.symbols == DEFAULT_JAX_CLASSIC_SYMBOLS_CSV


def test_parse_args_uses_shared_training_defaults(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["train_jax_classic.py"])
    args = train_jax_classic.parse_args()

    assert args.epochs == JAX_CLASSIC_DEFAULT_TRAIN_EPOCHS
    assert args.batch_size == JAX_CLASSIC_DEFAULT_BATCH_SIZE
    assert args.sequence_length == JAX_CLASSIC_DEFAULT_SEQUENCE_LENGTH
    assert args.num_outputs == JAX_CLASSIC_DEFAULT_NUM_OUTPUTS


def test_parse_args_supports_argfiles_and_forecast_refresh(tmp_path: Path) -> None:
    args_file = tmp_path / "jax_args.txt"
    args_file.write_text(
        "\n".join(
            [
                "# Comment line should be ignored",
                "--symbols aapl,dbx",
                "--allow-forecast-refresh",
                "--describe-run",
            ]
        ),
        encoding="utf-8",
    )

    args = train_jax_classic.parse_args([f"@{args_file}"])

    assert args.symbols == "aapl,dbx"
    assert args.cache_only is False
    assert args.describe_run is True


def test_build_run_plan_reflects_resolved_training_defaults(tmp_path: Path) -> None:
    args = SimpleNamespace(
        symbols="aapl,dbx,qqq",
        data_root=tmp_path / "data",
        cache_root=tmp_path / "cache",
        checkpoint_root=tmp_path / "ckpts",
        log_dir=tmp_path / "logs",
        run_name="jax_plan",
        epochs=3,
        batch_size=4,
        sequence_length=48,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        num_outputs=4,
        learning_rate=1e-4,
        weight_decay=0.05,
        grad_clip=0.5,
        return_weight=0.2,
        smoothness_penalty=0.01,
        maker_fee=0.001,
        fill_temperature=5e-4,
        validation_days=30,
        forecast_horizons="1,4",
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
        wandb_project="proj",
        wandb_entity="entity",
        wandb_group="group",
        wandb_tags="jax,stocks",
        wandb_notes="notes",
        wandb_mode="offline",
        wandb_log_metrics=True,
        describe_run=False,
    )

    plan = train_jax_classic.build_run_plan(args)

    assert plan["symbols"] == ["AAPL", "DBX", "QQQ"]
    assert plan["forecast_horizons"] == [1, 4]
    assert plan["cache_only"] is False
    assert plan["training"]["min_history_hours"] == compute_jax_classic_min_history_hours(48, 30)
    assert plan["training"]["dry_train_steps"] == 2


def test_main_rejects_invalid_symbol_before_loading_data(monkeypatch, tmp_path: Path) -> None:
    args = SimpleNamespace(
        symbols="../../etc/passwd",
        data_root=tmp_path / "data",
        cache_root=tmp_path / "cache",
        checkpoint_root=tmp_path / "ckpts",
        log_dir=tmp_path / "logs",
        run_name="jax_test_run",
        epochs=3,
        batch_size=4,
        sequence_length=48,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        num_outputs=4,
        learning_rate=1e-4,
        weight_decay=0.05,
        grad_clip=0.5,
        return_weight=0.2,
        smoothness_penalty=0.01,
        maker_fee=0.001,
        fill_temperature=5e-4,
        validation_days=30,
        forecast_horizons="1,4",
        cache_only=True,
        seed=123,
        max_hold_hours=5.0,
        max_leverage=2.0,
        margin_annual_rate=0.0625,
        decision_lag_bars=1,
        market_order_entry=True,
        fill_buffer_pct=0.0005,
        preload=None,
        dry_train_steps=2,
        wandb_project="proj",
        wandb_entity="entity",
        wandb_group="group",
        wandb_tags="jax,stocks",
        wandb_notes="notes",
        wandb_mode="offline",
        wandb_log_metrics=True,
        describe_run=False,
    )

    def _unexpected(*args, **kwargs):
        raise AssertionError("Invalid symbol input should fail before touching the data module")

    monkeypatch.setattr(train_jax_classic, "parse_args", lambda: args)
    monkeypatch.setattr(train_jax_classic, "MultiSymbolDataModule", _unexpected)

    try:
        train_jax_classic.main()
    except SystemExit as exc:
        assert "Unsupported symbol" in str(exc)
        assert str(exc).startswith("Plan error:")
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected SystemExit for invalid symbol input")


def test_main_builds_jax_classic_trainer_via_factory(monkeypatch, tmp_path: Path, capsys) -> None:
    captured: dict[str, object] = {}

    class _FakeDataModule:
        def __init__(self, *, symbols, config, directional_constraints) -> None:
            captured["symbols"] = list(symbols)
            captured["dataset_config"] = config
            captured["directional_constraints"] = dict(directional_constraints)

    class _FakeTrainer:
        def train(self):
            return SimpleNamespace(
                best_checkpoint=Path("/tmp/jax-best.flax"),
                stop_reason=None,
                history=[],
            )

    def _fake_build_trainer(config, data_module):
        captured["train_config"] = config
        captured["data_module"] = data_module
        return _FakeTrainer()

    args = SimpleNamespace(
        symbols="aapl,dbx,qqq",
        data_root=tmp_path / "data",
        cache_root=tmp_path / "cache",
        checkpoint_root=tmp_path / "ckpts",
        log_dir=tmp_path / "logs",
        run_name="jax_test_run",
        epochs=3,
        batch_size=4,
        sequence_length=48,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        num_outputs=4,
        learning_rate=1e-4,
        weight_decay=0.05,
        grad_clip=0.5,
        return_weight=0.2,
        smoothness_penalty=0.01,
        maker_fee=0.001,
        fill_temperature=5e-4,
        validation_days=30,
        forecast_horizons="1,4",
        seed=123,
        max_hold_hours=5.0,
        max_leverage=2.0,
        margin_annual_rate=0.0625,
        decision_lag_bars=1,
        market_order_entry=True,
        fill_buffer_pct=0.0005,
        preload=None,
        dry_train_steps=2,
        wandb_project="proj",
        wandb_entity="entity",
        wandb_group="group",
        wandb_tags="jax,stocks",
        wandb_notes="notes",
        wandb_mode="offline",
        wandb_log_metrics=True,
        cache_only=True,
        describe_run=False,
    )

    monkeypatch.setattr(train_jax_classic, "parse_args", lambda: args)
    monkeypatch.setattr(train_jax_classic, "MultiSymbolDataModule", _FakeDataModule)
    monkeypatch.setattr(train_jax_classic, "build_trainer", _fake_build_trainer)

    train_jax_classic.main()
    stdout = capsys.readouterr().out

    train_cfg = captured["train_config"]
    dataset_cfg = captured["dataset_config"]
    directional_constraints = captured["directional_constraints"]

    assert captured["symbols"] == ["AAPL", "DBX", "QQQ"]
    assert train_cfg.trainer_backend == "jax_classic"
    assert train_cfg.model_arch == "classic"
    assert train_cfg.market_order_entry is True
    assert dataset_cfg.cache_only is True
    assert dataset_cfg.forecast_horizons == (1, 4)
    assert dataset_cfg.min_history_hours == compute_jax_classic_min_history_hours(48, 30)
    assert directional_constraints["AAPL"] == (1.0, 0.0)
    assert directional_constraints["DBX"] == (0.0, 1.0)
    assert directional_constraints["QQQ"] == (1.0, 1.0)
    assert "JAX Classic Training Plan" in stdout
    assert "Symbols: AAPL,DBX,QQQ" in stdout
    assert "Rerun with: python -m unified_hourly_experiment.train_jax_classic" in stdout
    run_dir = args.checkpoint_root / "jax_test_run"
    effective_args = json.loads((run_dir / "effective_args.json").read_text(encoding="utf-8"))
    effective_args_txt = (run_dir / "effective_args.txt").read_text(encoding="utf-8")
    assert effective_args["symbols"] == "aapl,dbx,qqq"
    assert "--symbols aapl,dbx,qqq" in effective_args_txt


def test_main_continues_when_effective_args_write_fails(monkeypatch, tmp_path: Path, capsys) -> None:
    captured: dict[str, object] = {}

    class _FakeDataModule:
        def __init__(self, *, symbols, config, directional_constraints) -> None:
            captured["symbols"] = list(symbols)

    class _FakeTrainer:
        def train(self):
            captured["train_called"] = True
            return SimpleNamespace(
                best_checkpoint=Path("/tmp/jax-best.flax"),
                stop_reason=None,
                history=[],
            )

    def _fake_build_trainer(config, data_module):
        captured["train_config"] = config
        return _FakeTrainer()

    def _boom(*_args, **_kwargs):
        raise OSError("disk full")

    args = SimpleNamespace(
        symbols="aapl,dbx,qqq",
        data_root=tmp_path / "data",
        cache_root=tmp_path / "cache",
        checkpoint_root=tmp_path / "ckpts",
        log_dir=tmp_path / "logs",
        run_name="jax_test_run",
        epochs=3,
        batch_size=4,
        sequence_length=48,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        num_outputs=4,
        learning_rate=1e-4,
        weight_decay=0.05,
        grad_clip=0.5,
        return_weight=0.2,
        smoothness_penalty=0.01,
        maker_fee=0.001,
        fill_temperature=5e-4,
        validation_days=30,
        forecast_horizons="1,4",
        seed=123,
        max_hold_hours=5.0,
        max_leverage=2.0,
        margin_annual_rate=0.0625,
        decision_lag_bars=1,
        market_order_entry=True,
        fill_buffer_pct=0.0005,
        preload=None,
        dry_train_steps=2,
        wandb_project="proj",
        wandb_entity="entity",
        wandb_group="group",
        wandb_tags="jax,stocks",
        wandb_notes="notes",
        wandb_mode="offline",
        wandb_log_metrics=True,
        cache_only=True,
        describe_run=False,
    )

    monkeypatch.setattr(train_jax_classic, "parse_args", lambda: args)
    monkeypatch.setattr(train_jax_classic, "MultiSymbolDataModule", _FakeDataModule)
    monkeypatch.setattr(train_jax_classic, "build_trainer", _fake_build_trainer)
    monkeypatch.setattr(train_jax_classic, "write_effective_args_artifacts", _boom)

    train_jax_classic.main()
    stdout, stderr = capsys.readouterr()

    assert captured["symbols"] == ["AAPL", "DBX", "QQQ"]
    assert captured["train_called"] is True
    assert "Failed to write effective args artifacts" in stderr
    assert "Best checkpoint: /tmp/jax-best.flax" in stdout
    assert "Rerun with:" not in stdout


def test_main_describe_run_prints_plan_without_loading_data(monkeypatch, tmp_path: Path, capsys) -> None:
    args = SimpleNamespace(
        symbols="aapl,dbx",
        data_root=tmp_path / "data",
        cache_root=tmp_path / "cache",
        checkpoint_root=tmp_path / "ckpts",
        log_dir=tmp_path / "logs",
        run_name="jax_describe",
        epochs=3,
        batch_size=4,
        sequence_length=48,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        num_outputs=4,
        learning_rate=1e-4,
        weight_decay=0.05,
        grad_clip=0.5,
        return_weight=0.2,
        smoothness_penalty=0.01,
        maker_fee=0.001,
        fill_temperature=5e-4,
        validation_days=30,
        forecast_horizons="1,4",
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
        wandb_project="proj",
        wandb_entity="entity",
        wandb_group="group",
        wandb_tags="jax,stocks",
        wandb_notes="notes",
        wandb_mode="offline",
        wandb_log_metrics=True,
        describe_run=True,
    )

    def _unexpected(*args, **kwargs):
        raise AssertionError("Describe run should not touch the data module")

    monkeypatch.setattr(train_jax_classic, "parse_args", lambda: args)
    monkeypatch.setattr(train_jax_classic, "MultiSymbolDataModule", _unexpected)

    train_jax_classic.main()

    printed = json.loads(capsys.readouterr().out)
    assert printed["symbols"] == ["AAPL", "DBX"]
    assert printed["cache_only"] is False
    assert printed["training"]["min_history_hours"] == compute_jax_classic_min_history_hours(48, 30)


def test_main_adds_forecast_refresh_hint_on_cache_only_failure(monkeypatch, tmp_path: Path) -> None:
    args = SimpleNamespace(
        symbols="aapl,dbx",
        data_root=tmp_path / "data",
        cache_root=tmp_path / "cache",
        checkpoint_root=tmp_path / "ckpts",
        log_dir=tmp_path / "logs",
        run_name="jax_cache_only",
        epochs=3,
        batch_size=4,
        sequence_length=48,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        num_outputs=4,
        learning_rate=1e-4,
        weight_decay=0.05,
        grad_clip=0.5,
        return_weight=0.2,
        smoothness_penalty=0.01,
        maker_fee=0.001,
        fill_temperature=5e-4,
        validation_days=30,
        forecast_horizons="1,4",
        cache_only=True,
        seed=123,
        max_hold_hours=5.0,
        max_leverage=2.0,
        margin_annual_rate=0.0625,
        decision_lag_bars=1,
        market_order_entry=True,
        fill_buffer_pct=0.0005,
        preload=None,
        dry_train_steps=2,
        wandb_project="proj",
        wandb_entity="entity",
        wandb_group="group",
        wandb_tags="jax,stocks",
        wandb_notes="notes",
        wandb_mode="offline",
        wandb_log_metrics=True,
        describe_run=False,
    )

    monkeypatch.setattr(train_jax_classic, "parse_args", lambda: args)

    def _raise_missing_cache(*args, **kwargs):
        raise FileNotFoundError("missing forecast cache")

    monkeypatch.setattr(train_jax_classic, "MultiSymbolDataModule", _raise_missing_cache)

    try:
        train_jax_classic.main()
    except RuntimeError as exc:
        assert "missing forecast cache" in str(exc)
        assert "--allow-forecast-refresh" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected RuntimeError with forecast refresh hint")
