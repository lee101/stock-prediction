from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import unified_hourly_experiment.compare_trainer_backends as compare_trainer_backends
from unified_hourly_experiment.jax_classic_defaults import (
    JAX_CLASSIC_COMPARE_DEFAULT_DRY_TRAIN_STEPS,
    JAX_CLASSIC_COMPARE_DEFAULT_EPOCHS,
    JAX_CLASSIC_COMPARE_DEFAULT_MAX_RUNS,
)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_parse_backends_rejects_unknown_backend() -> None:
    try:
        compare_trainer_backends.parse_backends("torch,mystery")
    except ValueError as exc:
        assert "Unsupported trainer backend" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected ValueError for unsupported backend")


def test_parse_symbols_rejects_empty_input() -> None:
    try:
        compare_trainer_backends.parse_symbols(" , ")
    except ValueError as exc:
        assert "At least one symbol is required" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected ValueError for empty symbol input")


def test_parse_symbols_rejects_path_like_input() -> None:
    try:
        compare_trainer_backends.parse_symbols("../../etc/passwd")
    except ValueError as exc:
        assert "Unsupported symbol" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected ValueError for path-like symbol input")


def test_render_report_marks_winner() -> None:
    report = {
        "symbols": ["AAPL", "QQQ"],
        "forecast_horizons": [1, 4],
        "dry_train_steps": 2,
        "cache_only": True,
        "run_plan": {
            "output_dir": "/tmp/backend-compare",
            "seeds": [41, 42],
            "total_runs": 4,
        },
        "effective_args_path": "/tmp/backend-compare/effective_args.json",
        "run_events_path": "/tmp/backend-compare/run_events.jsonl",
        "plan_error": None,
        "backend_summary": [
            {
                "backend": "torch",
                "run_count": 2,
                "ok_runs": 2,
                "error_runs": 0,
                "success_rate": 1.0,
                "best_val_score_mean": 0.9,
                "best_val_score_std": 0.05,
                "stability_adjusted_score": 0.85,
                "best_val_sortino_mean": 1.1,
                "best_val_return_mean": 0.3,
                "duration_sec_mean": 1.4,
            }
        ],
        "winner_backend": "torch",
        "recommended_backend": "torch",
        "results": [
            {
                "backend": "torch",
                "seed": 41,
                "status": "ok",
                "duration_sec": 1.25,
                "best_val_score": 0.9,
                "best_val_sortino": 1.1,
                "best_val_return": 0.3,
                "stop_reason": None,
            },
            {
                "backend": "jax_classic",
                "seed": 42,
                "status": "ok",
                "duration_sec": 1.75,
                "best_val_score": 0.7,
                "best_val_sortino": 0.8,
                "best_val_return": 0.2,
                "stop_reason": "dry_train_steps=2 reached",
            },
        ],
    }
    rendered = compare_trainer_backends.render_markdown_report(report)
    assert "Winner by `best_val_score`: `torch`" in rendered
    assert "| torch | 41 | ok | 1.250 | 0.900000 | 1.100000 | 0.300000 |  |" in rendered
    assert "dry_train_steps=2 reached" in rendered
    assert "Effective args" in rendered
    assert "Run events" in rendered
    assert "## Aggregated Summary" in rendered
    assert "- Seeds: `41,42`" in rendered
    assert "- Planned runs: `4`" in rendered
    assert "Recommended backend by reliability: `torch`" in rendered


def test_argfile_support_ignores_comments(tmp_path: Path) -> None:
    args_file = tmp_path / "compare_args.txt"
    args_file.write_text(
        "--symbols aapl,qqq\n"
        "# keep the smoke run short\n"
        "--forecast-horizons 1,4\n"
        "--backends torch,jax_classic\n",
        encoding="utf-8",
    )

    parser = compare_trainer_backends.build_arg_parser()
    args = parser.parse_args([f"@{args_file}"])

    assert args.symbols == "aapl,qqq"
    assert args.forecast_horizons == "1,4"
    assert args.backends == "torch,jax_classic"


def test_build_arg_parser_uses_shared_compare_defaults() -> None:
    args = compare_trainer_backends.build_arg_parser().parse_args([])

    assert args.epochs == JAX_CLASSIC_COMPARE_DEFAULT_EPOCHS
    assert args.dry_train_steps == JAX_CLASSIC_COMPARE_DEFAULT_DRY_TRAIN_STEPS
    assert args.max_runs == JAX_CLASSIC_COMPARE_DEFAULT_MAX_RUNS


def test_build_run_plan_summarizes_resolved_args(tmp_path: Path) -> None:
    args = SimpleNamespace(
        symbols="aapl,qqq",
        backends="torch,jax_classic",
        output_dir=tmp_path / "out",
        forecast_horizons="1,4",
        cache_only=True,
        preload=tmp_path / "preload.ckpt",
        seeds="7,9",
        seed=42,
        max_runs=8,
        allow_large_run=False,
        epochs=2,
        dry_train_steps=3,
        batch_size=4,
        sequence_length=48,
        validation_days=30,
        learning_rate=1e-4,
        weight_decay=0.05,
        grad_clip=0.5,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        use_compile=True,
    )

    plan = compare_trainer_backends.build_run_plan(args)

    assert plan["symbols"] == ["AAPL", "QQQ"]
    assert plan["backends"] == ["torch", "jax_classic"]
    assert plan["forecast_horizons"] == [1, 4]
    assert plan["backend_count"] == 2
    assert plan["seed_count"] == 2
    assert plan["seeds"] == [7, 9]
    assert plan["total_runs"] == 4
    assert plan["plan_error"] is None
    assert plan["effective_args_path"].endswith("effective_args.json")
    assert plan["training"]["use_compile"] is True
    assert plan["preload"] == str(tmp_path / "preload.ckpt")


def test_parse_seeds_defaults_and_dedupes() -> None:
    assert compare_trainer_backends.parse_seeds(None, fallback_seed=42) == [42]
    assert compare_trainer_backends.parse_seeds("7,7,9", fallback_seed=42) == [7, 9]


def test_build_run_plan_marks_oversized_run(tmp_path: Path) -> None:
    args = SimpleNamespace(
        symbols="aapl,qqq",
        backends="torch,jax_classic",
        output_dir=tmp_path / "out",
        forecast_horizons="1,4",
        cache_only=True,
        preload=None,
        seeds="1,2,3",
        seed=42,
        max_runs=4,
        allow_large_run=False,
        epochs=2,
        dry_train_steps=3,
        batch_size=4,
        sequence_length=48,
        validation_days=30,
        learning_rate=1e-4,
        weight_decay=0.05,
        grad_clip=0.5,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        use_compile=True,
    )

    plan = compare_trainer_backends.build_run_plan(args)

    assert plan["total_runs"] == 6
    assert "exceeds max_runs=4" in plan["plan_error"]


def test_build_run_plan_rejects_non_positive_max_runs(tmp_path: Path) -> None:
    args = SimpleNamespace(
        symbols="aapl",
        backends="torch",
        output_dir=tmp_path / "out",
        forecast_horizons="1",
        cache_only=True,
        preload=None,
        seeds=None,
        seed=42,
        max_runs=0,
        allow_large_run=False,
        epochs=2,
        dry_train_steps=3,
        batch_size=4,
        sequence_length=48,
        validation_days=30,
        learning_rate=1e-4,
        weight_decay=0.05,
        grad_clip=0.5,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        use_compile=True,
    )

    try:
        compare_trainer_backends.build_run_plan(args)
    except ValueError as exc:
        assert "max_runs must be positive" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected ValueError for non-positive max_runs")


def test_summarize_backend_groups_aggregates_ok_and_error_runs() -> None:
    summary = compare_trainer_backends.summarize_backend_groups(
        [
            {"backend": "torch", "seed": 1, "status": "ok", "duration_sec": 1.0, "best_val_score": 0.6, "best_val_sortino": 0.8, "best_val_return": 0.1},
            {"backend": "torch", "seed": 2, "status": "ok", "duration_sec": 3.0, "best_val_score": 1.0, "best_val_sortino": 1.2, "best_val_return": 0.3},
            {"backend": "jax_classic", "seed": 1, "status": "error", "duration_sec": 2.0},
        ]
    )

    assert summary[0]["backend"] == "jax_classic"
    assert summary[0]["run_count"] == 1
    assert summary[0]["ok_runs"] == 0
    assert summary[0]["error_runs"] == 1
    assert summary[0]["success_rate"] == 0.0
    assert summary[0]["best_val_score_mean"] is None
    assert summary[1]["backend"] == "torch"
    assert summary[1]["run_count"] == 2
    assert summary[1]["ok_runs"] == 2
    assert summary[1]["error_runs"] == 0
    assert summary[1]["success_rate"] == 1.0
    assert summary[1]["best_val_score_mean"] == 0.8
    assert summary[1]["best_val_score_std"] == 0.2
    assert summary[1]["stability_adjusted_score"] == pytest.approx(0.6)


def test_recommend_backend_prefers_reliability_over_single_higher_mean() -> None:
    backend_summary = compare_trainer_backends.summarize_backend_groups(
        [
            {"backend": "torch", "seed": 1, "status": "ok", "duration_sec": 1.0, "best_val_score": 1.2, "best_val_sortino": 1.4, "best_val_return": 0.3},
            {"backend": "torch", "seed": 2, "status": "error", "duration_sec": 1.1, "error": "boom", "traceback": "trace"},
            {"backend": "jax_classic", "seed": 1, "status": "ok", "duration_sec": 1.5, "best_val_score": 1.05, "best_val_sortino": 1.1, "best_val_return": 0.25},
            {"backend": "jax_classic", "seed": 2, "status": "ok", "duration_sec": 1.4, "best_val_score": 1.04, "best_val_sortino": 1.08, "best_val_return": 0.24},
        ]
    )

    assert compare_trainer_backends.recommend_backend(backend_summary) == "jax_classic"


def test_main_writes_backend_report(monkeypatch, tmp_path: Path, capsys) -> None:
    captured: dict[str, object] = {}

    class _FakeDataModule:
        def __init__(self, *, symbols, config, directional_constraints) -> None:
            captured["symbols"] = list(symbols)
            captured["dataset_config"] = config
            captured["directional_constraints"] = dict(directional_constraints)

    class _FakeTrainer:
        def __init__(self, backend: str) -> None:
            self._backend = backend

        def train(self):
            if self._backend == "torch":
                history = [
                    SimpleNamespace(epoch=1, train_loss=1.2, val_score=0.4, val_sortino=0.5, val_return=0.1),
                    SimpleNamespace(epoch=2, train_loss=0.9, val_score=0.7, val_sortino=0.8, val_return=0.2),
                ]
                return SimpleNamespace(
                    history=history,
                    best_checkpoint=Path("/tmp/torch-best.pt"),
                    stop_reason=None,
                )
            history = [
                SimpleNamespace(epoch=1, train_loss=1.1, val_score=0.3, val_sortino=0.4, val_return=0.05),
            ]
            return SimpleNamespace(
                history=history,
                best_checkpoint=Path("/tmp/jax-best.flax"),
                stop_reason="dry_train_steps=2 reached",
            )

    def _fake_build_trainer(config, data_module):
        captured.setdefault("train_backends", []).append(config.trainer_backend)
        captured.setdefault("use_compile", {})[config.trainer_backend] = config.use_compile
        return _FakeTrainer(config.trainer_backend)

    args = SimpleNamespace(
        symbols="aapl,qqq",
        backends="torch,jax_classic",
        data_root=tmp_path / "data",
        cache_root=tmp_path / "cache",
        output_dir=tmp_path / "out",
        epochs=2,
        dry_train_steps=2,
        cache_only=True,
        batch_size=4,
        sequence_length=48,
        validation_days=30,
        forecast_horizons="1,4",
        learning_rate=1e-4,
        weight_decay=0.05,
        grad_clip=0.5,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        return_weight=0.2,
        smoothness_penalty=0.01,
        maker_fee=0.001,
        fill_temperature=5e-4,
        max_hold_hours=5.0,
        max_leverage=2.0,
        margin_annual_rate=0.0625,
        decision_lag_bars=1,
        market_order_entry=True,
        fill_buffer_pct=0.0005,
        seed=123,
        seeds=None,
        max_runs=8,
        allow_large_run=False,
        use_compile=True,
        preload=None,
    )

    monkeypatch.setattr(compare_trainer_backends, "parse_args", lambda: args)
    monkeypatch.setattr(compare_trainer_backends, "MultiSymbolDataModule", _FakeDataModule)
    monkeypatch.setattr(compare_trainer_backends, "build_trainer", _fake_build_trainer)

    compare_trainer_backends.main()
    stdout = capsys.readouterr().out

    report = json.loads((args.output_dir / "report.json").read_text(encoding="utf-8"))
    markdown = (args.output_dir / "report.md").read_text(encoding="utf-8")
    effective_args = json.loads((args.output_dir / "effective_args.json").read_text(encoding="utf-8"))
    effective_args_txt = (args.output_dir / "effective_args.txt").read_text(encoding="utf-8")
    events = _read_jsonl(args.output_dir / "run_events.jsonl")

    assert captured["symbols"] == ["AAPL", "QQQ"]
    assert captured["train_backends"] == ["torch", "jax_classic"]
    assert captured["use_compile"] == {"torch": True, "jax_classic": False}
    assert report["winner_backend"] == "torch"
    assert report["recommended_backend"] == "torch"
    assert report["cache_only"] is True
    assert report["run_plan"]["backend_count"] == 2
    assert report["run_plan"]["seed_count"] == 1
    assert report["run_plan"]["total_runs"] == 2
    assert report["effective_args_path"].endswith("effective_args.json")
    assert report["effective_args_cli_path"].endswith("effective_args.txt")
    assert report["run_events_path"].endswith("run_events.jsonl")
    assert [item["backend"] for item in report["results"]] == ["torch", "jax_classic"]
    assert [item["seed"] for item in report["results"]] == [123, 123]
    assert "Winner by `best_val_score`: `torch`" in markdown
    assert "Recommended backend by reliability: `torch`" in markdown
    assert effective_args["symbols"] == "aapl,qqq"
    assert "Effective args" in markdown
    assert "Rerun args file" in markdown
    assert "Run events" in markdown
    assert "--backends torch,jax_classic" in effective_args_txt
    assert [event["event_type"] for event in events] == [
        "run_start",
        "dataset_prepare_start",
        "dataset_ready",
        "backend_seed_start",
        "backend_seed_complete",
        "backend_seed_start",
        "backend_seed_complete",
        "run_complete",
    ]
    assert "Backend Compare Plan" in stdout
    assert "Backends: torch,jax_classic" in stdout
    assert "Seeds: 123" in stdout
    assert "Rerun with: python -m unified_hourly_experiment.compare_trainer_backends" in stdout


def test_run_backend_comparison_aggregates_multiple_seeds(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class _FakeDataModule:
        def __init__(self, *, symbols, config, directional_constraints) -> None:
            captured["symbols"] = list(symbols)

    class _FakeTrainer:
        def __init__(self, backend: str, seed: int) -> None:
            self._backend = backend
            self._seed = seed

        def train(self):
            base = 1.0 if self._backend == "torch" else 0.7
            score = base + (0.01 * self._seed)
            return SimpleNamespace(
                history=[SimpleNamespace(epoch=1, train_loss=1.0, val_score=score, val_sortino=score + 0.1, val_return=score - 0.2)],
                best_checkpoint=Path(f"/tmp/{self._backend}-{self._seed}.ckpt"),
                stop_reason=None,
            )

    def _fake_build_trainer(config, data_module):
        captured.setdefault("seen", []).append((config.trainer_backend, config.seed, config.run_name))
        return _FakeTrainer(config.trainer_backend, config.seed)

    args = SimpleNamespace(
        symbols="aapl,qqq",
        backends="torch,jax_classic",
        data_root=tmp_path / "data",
        cache_root=tmp_path / "cache",
        output_dir=tmp_path / "out",
        epochs=2,
        dry_train_steps=2,
        cache_only=True,
        batch_size=4,
        sequence_length=48,
        validation_days=30,
        forecast_horizons="1,4",
        learning_rate=1e-4,
        weight_decay=0.05,
        grad_clip=0.5,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        return_weight=0.2,
        smoothness_penalty=0.01,
        maker_fee=0.001,
        fill_temperature=5e-4,
        max_hold_hours=5.0,
        max_leverage=2.0,
        margin_annual_rate=0.0625,
        decision_lag_bars=1,
        market_order_entry=True,
        fill_buffer_pct=0.0005,
        seed=123,
        seeds="3,5",
        max_runs=8,
        allow_large_run=False,
        use_compile=True,
        preload=None,
    )

    monkeypatch.setattr(compare_trainer_backends, "MultiSymbolDataModule", _FakeDataModule)
    monkeypatch.setattr(compare_trainer_backends, "build_trainer", _fake_build_trainer)

    report = compare_trainer_backends.run_backend_comparison(args)

    assert len(report["results"]) == 4
    assert report["run_plan"]["seeds"] == [3, 5]
    assert report["run_plan"]["total_runs"] == 4
    assert report["winner_backend"] == "torch"
    assert report["recommended_backend"] == "torch"
    assert captured["seen"] == [
        ("torch", 3, "compare_torch_seed3"),
        ("torch", 5, "compare_torch_seed5"),
        ("jax_classic", 3, "compare_jax_classic_seed3"),
        ("jax_classic", 5, "compare_jax_classic_seed5"),
    ]
    torch_summary = next(item for item in report["backend_summary"] if item["backend"] == "torch")
    assert torch_summary["ok_runs"] == 2
    assert torch_summary["success_rate"] == 1.0
    assert torch_summary["best_val_score_mean"] == 1.04


def test_run_backend_comparison_records_backend_seed_error(monkeypatch, tmp_path: Path) -> None:
    class _FakeDataModule:
        def __init__(self, *, symbols, config, directional_constraints) -> None:
            pass

    class _FakeTrainer:
        def __init__(self, backend: str, seed: int) -> None:
            self._backend = backend
            self._seed = seed

        def train(self):
            if self._backend == "jax_classic" and self._seed == 5:
                raise RuntimeError("boom")
            score = 1.0 + (0.01 * self._seed)
            return SimpleNamespace(
                history=[SimpleNamespace(epoch=1, train_loss=1.0, val_score=score, val_sortino=score + 0.1, val_return=score - 0.2)],
                best_checkpoint=Path(f"/tmp/{self._backend}-{self._seed}.ckpt"),
                stop_reason=None,
            )

    def _fake_build_trainer(config, data_module):
        return _FakeTrainer(config.trainer_backend, config.seed)

    args = SimpleNamespace(
        symbols="aapl,qqq",
        backends="torch,jax_classic",
        data_root=tmp_path / "data",
        cache_root=tmp_path / "cache",
        output_dir=tmp_path / "out",
        epochs=2,
        dry_train_steps=2,
        cache_only=True,
        batch_size=4,
        sequence_length=48,
        validation_days=30,
        forecast_horizons="1,4",
        learning_rate=1e-4,
        weight_decay=0.05,
        grad_clip=0.5,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        return_weight=0.2,
        smoothness_penalty=0.01,
        maker_fee=0.001,
        fill_temperature=5e-4,
        max_hold_hours=5.0,
        max_leverage=2.0,
        margin_annual_rate=0.0625,
        decision_lag_bars=1,
        market_order_entry=True,
        fill_buffer_pct=0.0005,
        seed=123,
        seeds="3,5",
        max_runs=8,
        allow_large_run=False,
        use_compile=True,
        preload=None,
    )

    monkeypatch.setattr(compare_trainer_backends, "MultiSymbolDataModule", _FakeDataModule)
    monkeypatch.setattr(compare_trainer_backends, "build_trainer", _fake_build_trainer)

    report = compare_trainer_backends.run_backend_comparison(args)
    events = _read_jsonl(tmp_path / "out" / "run_events.jsonl")

    assert len(report["results"]) == 4
    error_result = next(item for item in report["results"] if item["backend"] == "jax_classic" and item["seed"] == 5)
    assert error_result["status"] == "error"
    assert error_result["error"] == "boom"
    assert any(event["event_type"] == "backend_seed_error" and event["backend"] == "jax_classic" and event["seed"] == 5 for event in events)
    assert events[-1]["event_type"] == "run_complete"


def test_main_writes_plan_error_report_before_training(monkeypatch, tmp_path: Path) -> None:
    args = SimpleNamespace(
        symbols="aapl,qqq",
        backends="torch,jax_classic",
        data_root=tmp_path / "data",
        cache_root=tmp_path / "cache",
        output_dir=tmp_path / "out",
        epochs=2,
        dry_train_steps=2,
        cache_only=True,
        batch_size=4,
        sequence_length=48,
        validation_days=30,
        forecast_horizons="1,4",
        learning_rate=1e-4,
        weight_decay=0.05,
        grad_clip=0.5,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        return_weight=0.2,
        smoothness_penalty=0.01,
        maker_fee=0.001,
        fill_temperature=5e-4,
        max_hold_hours=5.0,
        max_leverage=2.0,
        margin_annual_rate=0.0625,
        decision_lag_bars=1,
        market_order_entry=True,
        fill_buffer_pct=0.0005,
        seed=123,
        seeds="1,2,3",
        max_runs=4,
        allow_large_run=False,
        use_compile=False,
        preload=None,
    )

    def _unexpected(*args, **kwargs):
        raise AssertionError("Data/training path should not run when plan is over budget")

    monkeypatch.setattr(compare_trainer_backends, "parse_args", lambda: args)
    monkeypatch.setattr(compare_trainer_backends, "MultiSymbolDataModule", _unexpected)

    try:
        compare_trainer_backends.main()
    except SystemExit as exc:
        assert exc.code == 1
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected compare tool to exit 1 on plan error")

    report = json.loads((args.output_dir / "report.json").read_text(encoding="utf-8"))
    markdown = (args.output_dir / "report.md").read_text(encoding="utf-8")
    events = _read_jsonl(args.output_dir / "run_events.jsonl")
    assert report["results"] == []
    assert "exceeds max_runs=4" in report["plan_error"]
    assert "## Plan Error" in markdown
    assert [event["event_type"] for event in events] == ["run_start", "plan_error"]


def test_main_writes_invalid_input_report_before_training(monkeypatch, tmp_path: Path) -> None:
    args = SimpleNamespace(
        symbols="../../etc/passwd",
        backends="torch,jax_classic",
        data_root=tmp_path / "data",
        cache_root=tmp_path / "cache",
        output_dir=tmp_path / "out",
        epochs=2,
        dry_train_steps=2,
        cache_only=True,
        batch_size=4,
        sequence_length=48,
        validation_days=30,
        forecast_horizons="1,4",
        learning_rate=1e-4,
        weight_decay=0.05,
        grad_clip=0.5,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        return_weight=0.2,
        smoothness_penalty=0.01,
        maker_fee=0.001,
        fill_temperature=5e-4,
        max_hold_hours=5.0,
        max_leverage=2.0,
        margin_annual_rate=0.0625,
        decision_lag_bars=1,
        market_order_entry=True,
        fill_buffer_pct=0.0005,
        seed=123,
        seeds=None,
        max_runs=8,
        allow_large_run=False,
        use_compile=False,
        preload=None,
        describe_run=False,
    )

    def _unexpected(*args, **kwargs):
        raise AssertionError("Comparison path should not run when CLI input is invalid")

    monkeypatch.setattr(compare_trainer_backends, "parse_args", lambda: args)
    monkeypatch.setattr(compare_trainer_backends, "run_backend_comparison", _unexpected)

    try:
        compare_trainer_backends.main()
    except SystemExit as exc:
        assert exc.code == 1
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected compare tool to exit 1 on invalid input")

    report = json.loads((args.output_dir / "report.json").read_text(encoding="utf-8"))
    markdown = (args.output_dir / "report.md").read_text(encoding="utf-8")
    events = _read_jsonl(args.output_dir / "run_events.jsonl")
    assert report["results"] == []
    assert "Unsupported symbol" in report["plan_error"]
    assert report["run_plan"]["symbols"] == []
    assert "## Plan Error" in markdown
    assert [event["event_type"] for event in events] == ["run_start", "plan_error"]


def test_main_writes_dataset_error_report(monkeypatch, tmp_path: Path) -> None:
    args = SimpleNamespace(
        symbols="aapl,qqq",
        backends="torch,jax_classic",
        data_root=tmp_path / "data",
        cache_root=tmp_path / "cache",
        output_dir=tmp_path / "out",
        epochs=2,
        dry_train_steps=2,
        cache_only=True,
        batch_size=4,
        sequence_length=48,
        validation_days=30,
        forecast_horizons="1,4",
        learning_rate=1e-4,
        weight_decay=0.05,
        grad_clip=0.5,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        return_weight=0.2,
        smoothness_penalty=0.01,
        maker_fee=0.001,
        fill_temperature=5e-4,
        max_hold_hours=5.0,
        max_leverage=2.0,
        margin_annual_rate=0.0625,
        decision_lag_bars=1,
        market_order_entry=True,
        fill_buffer_pct=0.0005,
        seed=123,
        seeds=None,
        max_runs=8,
        allow_large_run=False,
        use_compile=False,
        preload=None,
    )

    def _boom(*args, **kwargs):
        raise RuntimeError("missing cache")

    monkeypatch.setattr(compare_trainer_backends, "parse_args", lambda: args)
    monkeypatch.setattr(compare_trainer_backends, "MultiSymbolDataModule", _boom)

    try:
        compare_trainer_backends.main()
    except SystemExit as exc:
        assert exc.code == 1
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected compare tool to exit 1 on dataset error")

    report = json.loads((args.output_dir / "report.json").read_text(encoding="utf-8"))
    markdown = (args.output_dir / "report.md").read_text(encoding="utf-8")
    events = _read_jsonl(args.output_dir / "run_events.jsonl")
    assert report["results"] == []
    assert "Use --allow-forecast-refresh" in report["dataset_error"]
    assert "## Dataset Error" in markdown
    assert report["effective_args_path"].endswith("effective_args.json")
    assert [event["event_type"] for event in events] == ["run_start", "dataset_prepare_start", "dataset_error"]


def test_main_keeps_report_when_effective_args_write_fails(monkeypatch, tmp_path: Path, capsys) -> None:
    args = SimpleNamespace(
        symbols="aapl,qqq",
        backends="torch,jax_classic",
        data_root=tmp_path / "data",
        cache_root=tmp_path / "cache",
        output_dir=tmp_path / "out",
        epochs=2,
        dry_train_steps=2,
        cache_only=True,
        batch_size=4,
        sequence_length=48,
        validation_days=30,
        forecast_horizons="1,4",
        learning_rate=1e-4,
        weight_decay=0.05,
        grad_clip=0.5,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        return_weight=0.2,
        smoothness_penalty=0.01,
        maker_fee=0.001,
        fill_temperature=5e-4,
        max_hold_hours=5.0,
        max_leverage=2.0,
        margin_annual_rate=0.0625,
        decision_lag_bars=1,
        market_order_entry=True,
        fill_buffer_pct=0.0005,
        seed=123,
        seeds=None,
        max_runs=8,
        allow_large_run=False,
        use_compile=False,
        preload=None,
        describe_run=False,
    )

    def _fake_run_backend_comparison(_args):
        return {
            "run_plan": compare_trainer_backends.build_run_plan(args),
            "run_id": "test-run",
            "run_events_path": str(args.output_dir / "run_events.jsonl"),
            "symbols": ["AAPL", "QQQ"],
            "forecast_horizons": [1, 4],
            "dry_train_steps": 2,
            "cache_only": True,
            "plan_error": None,
            "results": [],
            "backend_summary": [],
            "winner_backend": None,
            "recommended_backend": None,
        }

    def _boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(compare_trainer_backends, "parse_args", lambda: args)
    monkeypatch.setattr(compare_trainer_backends, "run_backend_comparison", _fake_run_backend_comparison)
    monkeypatch.setattr(compare_trainer_backends, "write_effective_args_artifacts", _boom)

    compare_trainer_backends.main()

    report = json.loads((args.output_dir / "report.json").read_text(encoding="utf-8"))
    markdown = (args.output_dir / "report.md").read_text(encoding="utf-8")
    stdout = capsys.readouterr()
    assert "effective_args_path" not in report
    assert "Failed to write effective args artifacts" in report["effective_args_warning"]
    assert "Effective args warning" in markdown
    assert "Failed to write effective args artifacts" in stdout.err
    assert f"Wrote {args.output_dir / 'report.json'}" in stdout.out
    assert f"Wrote {args.output_dir / 'report.md'}" in stdout.out
    assert "Rerun with:" not in stdout.out


def test_main_describe_run_prints_plan_and_skips_training(monkeypatch, tmp_path: Path, capsys) -> None:
    args = SimpleNamespace(
        symbols="aapl,qqq",
        backends="torch,jax_classic",
        data_root=tmp_path / "data",
        cache_root=tmp_path / "cache",
        output_dir=tmp_path / "out",
        epochs=2,
        dry_train_steps=2,
        cache_only=True,
        batch_size=4,
        sequence_length=48,
        validation_days=30,
        forecast_horizons="1,4",
        learning_rate=1e-4,
        weight_decay=0.05,
        grad_clip=0.5,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        return_weight=0.2,
        smoothness_penalty=0.01,
        maker_fee=0.001,
        fill_temperature=5e-4,
        max_hold_hours=5.0,
        max_leverage=2.0,
        margin_annual_rate=0.0625,
        decision_lag_bars=1,
        market_order_entry=True,
        fill_buffer_pct=0.0005,
        seed=123,
        seeds=None,
        max_runs=8,
        allow_large_run=False,
        use_compile=False,
        preload=None,
        describe_run=True,
    )

    def _unexpected(*args, **kwargs):
        raise AssertionError("Training path should not run during --describe-run")

    monkeypatch.setattr(compare_trainer_backends, "parse_args", lambda: args)
    monkeypatch.setattr(compare_trainer_backends, "run_backend_comparison", _unexpected)

    compare_trainer_backends.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["symbols"] == ["AAPL", "QQQ"]
    assert payload["backends"] == ["torch", "jax_classic"]
