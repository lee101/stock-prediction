from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
import torch.nn as nn

import autoresearch_stock.train as train_mod


def test_resolve_autoresearch_training_device_rejects_cpu_request(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    with pytest.raises(RuntimeError, match="requested_device='cpu'") as excinfo:
        train_mod.resolve_autoresearch_training_device("cpu")
    assert "cuda_device_count=2" in str(excinfo.value)


def test_resolve_autoresearch_training_device_rejects_missing_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)

    with pytest.raises(RuntimeError, match="cuda_available=False") as excinfo:
        train_mod.resolve_autoresearch_training_device("auto")
    assert "cuda_device_count=0" in str(excinfo.value)


def test_resolve_autoresearch_training_device_reports_non_cuda_resolution(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(train_mod, "resolve_runtime_device", lambda _requested: torch.device("cpu"))

    with pytest.raises(RuntimeError, match="resolved_device=cpu") as excinfo:
        train_mod.resolve_autoresearch_training_device("auto")
    assert "requested_device='auto'" in str(excinfo.value)


def test_build_auto_lr_cache_key_changes_with_network_size(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device=None: "UnitTestGPU")
    monkeypatch.setattr(train_mod, "_cuda_total_memory_gb", lambda _device: 24.0)
    device = torch.device("cuda")

    key_small = train_mod.build_auto_lr_cache_key(
        frequency="hourly",
        device=device,
        feature_dim=32,
        symbol_count=4,
        hidden_size=128,
        num_layers=2,
        symbol_embedding_dim=16,
        context_blocks=1,
        batch_size=64,
    )
    key_large = train_mod.build_auto_lr_cache_key(
        frequency="hourly",
        device=device,
        feature_dim=32,
        symbol_count=4,
        hidden_size=256,
        num_layers=2,
        symbol_embedding_dim=16,
        context_blocks=1,
        batch_size=64,
    )

    assert key_small != key_large


def test_resolve_learning_rate_prefers_cached_value(tmp_path: Path) -> None:
    cache_path = tmp_path / "auto_lr.json"
    cache_key = "planner-key"
    train_mod.write_auto_lr_cache(cache_path, cache_key, 7.5e-4)

    def _unexpected_tuner() -> float:
        raise AssertionError("tuner should not run when cache is warm")

    learning_rate, source = train_mod.resolve_learning_rate(
        requested_lr=None,
        disable_auto_lr_find=False,
        cache_path=str(cache_path),
        cache_key=cache_key,
        tuner=_unexpected_tuner,
    )

    assert learning_rate == pytest.approx(7.5e-4)
    assert source == "cache"


def test_resolve_learning_rate_writes_auto_tuned_value(tmp_path: Path) -> None:
    cache_path = tmp_path / "auto_lr.json"
    cache_key = "planner-key"

    learning_rate, source = train_mod.resolve_learning_rate(
        requested_lr=None,
        disable_auto_lr_find=False,
        cache_path=str(cache_path),
        cache_key=cache_key,
        tuner=lambda: 5e-4,
    )

    assert learning_rate == pytest.approx(5e-4)
    assert source == "auto"
    assert train_mod.load_auto_lr_cache(cache_path)[cache_key] == pytest.approx(5e-4)


def test_execution_modifier_tuning_is_opt_in() -> None:
    assert train_mod.resolve_execution_modifier_tuning_enabled(requested=False, disabled=False) is False
    assert train_mod.resolve_execution_modifier_tuning_enabled(requested=True, disabled=False) is True
    assert train_mod.resolve_execution_modifier_tuning_enabled(requested=True, disabled=True) is False


def test_default_autoresearch_checkpoint_root_prefers_env(monkeypatch, tmp_path: Path) -> None:
    override = tmp_path / "autoresearch_ckpts"
    monkeypatch.setenv("AUTORESEARCH_STOCK_CHECKPOINT_ROOT", str(override))

    assert train_mod.default_autoresearch_checkpoint_root() == override
    assert train_mod.resolve_autoresearch_checkpoint_dir(frequency="hourly", checkpoint_dir=None) == override / "hourly"


def test_save_autoresearch_checkpoint_keeps_top_k_and_aliases(tmp_path: Path) -> None:
    model = nn.Linear(4, 2)
    planner_cfg = train_mod.PlannerConfig(hidden_size=16, num_layers=2, batch_size=8, eval_batch_size=8)
    task_config = SimpleNamespace(
        frequency="hourly",
        data_root=tmp_path / "data",
        recent_data_root=None,
        symbols=("AAPL", "MSFT"),
        sequence_length=8,
        hold_bars=3,
        eval_windows=(8, 16),
        recent_overlay_bars=0,
        initial_cash=10_000.0,
        max_positions=2,
        max_volume_fraction=0.01,
        min_edge_bps=4.0,
        entry_slippage_bps=1.0,
        exit_slippage_bps=1.0,
        decision_lag_bars=1,
        allow_short=True,
        close_at_session_end=True,
        spread_lookback_days=14,
        periods_per_year=252.0 * 7.0,
        dashboard_db_path=tmp_path / "metrics.db",
    )
    task = SimpleNamespace(
        config=task_config,
        train_features=torch.zeros(12, 8, 4).numpy(),
        symbol_to_id={"AAPL": 0, "MSFT": 1},
    )
    checkpoint_dir = tmp_path / "ckpts"

    saved_paths: list[Path] = []
    for score in (1.0, 3.0, 2.0):
        checkpoint_path, best_path = train_mod.save_autoresearch_checkpoint(
            checkpoint_dir=checkpoint_dir,
            frequency="hourly",
            model=model,
            planner_cfg=planner_cfg,
            task=task,
            summary={"robust_score": score, "scenario_count": 3.0, "total_trade_count": 10.0},
            val_loss=0.1,
            model_parameters=10,
            step_count=5,
            training_seconds=1.0,
            total_seconds=2.0,
            peak_vram_mb=3.0,
            runtime_device="cuda",
            auto_cpu_fallback_used=False,
            best_modifiers=train_mod.ExecutionModifierSet(),
            learning_rate_source="auto",
            execution_modifier_tuning_enabled=False,
            top_k_checkpoints=2,
        )
        saved_paths.append(checkpoint_path)
        assert best_path is not None

    manifest = json.loads((checkpoint_dir / ".topk_manifest.json").read_text(encoding="utf-8"))
    manifest_paths = [Path(entry["path"]) for entry in manifest]

    assert len(manifest_paths) == 2
    assert saved_paths[0] not in manifest_paths
    assert saved_paths[1] in manifest_paths
    assert saved_paths[2] in manifest_paths
    assert (checkpoint_dir / "latest.pt").exists()
    assert (checkpoint_dir / "latest_summary.json").exists()
    assert (checkpoint_dir / "best.pt").exists()

    best_payload = torch.load(checkpoint_dir / "best.pt", map_location="cpu", weights_only=False)
    latest_summary = json.loads((checkpoint_dir / "latest_summary.json").read_text(encoding="utf-8"))

    assert best_payload["metadata"]["summary"]["robust_score"] == pytest.approx(3.0)
    assert latest_summary["summary"]["robust_score"] == pytest.approx(2.0)
    assert latest_summary["runtime_device"] == "cuda"
    assert latest_summary["auto_cpu_fallback_used"] is False


def test_auto_find_learning_rate_retries_probe_on_cpu_after_cuda_failure(monkeypatch) -> None:
    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(2, 3)

        def predict_with_plan(self, features, symbol_ids):
            logits = self.linear(features[:, -1, :])
            return logits, logits, logits[:, :1], logits

    first_attempt = {"done": False}
    seen_devices: list[str] = []

    def _fake_objective(model, batch, *, device, **kwargs):
        seen_devices.append(device.type)
        if device.type == "cuda" and not first_attempt["done"]:
            first_attempt["done"] = True
            raise RuntimeError("CUDA out of memory during probe")
        return (next(model.parameters()) ** 2).sum()

    monkeypatch.setattr(train_mod, "_planner_objective", _fake_objective)
    monkeypatch.setattr(train_mod, "load_planner_model", lambda **kwargs: (TinyModel(), torch.device("cuda")))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    batch = {
        "features": torch.zeros((2, 3, 2), dtype=torch.float32),
        "targets": torch.zeros((2, 3), dtype=torch.float32),
        "symbol_ids": torch.zeros((2,), dtype=torch.long),
        "weights": torch.ones((2,), dtype=torch.float32),
        "plan_labels": torch.zeros((2,), dtype=torch.long),
        "margin_targets": torch.zeros((2,), dtype=torch.float32),
        "budget_labels": torch.zeros((2,), dtype=torch.long),
    }

    best_lr = train_mod.auto_find_learning_rate(
        model_factory=TinyModel,
        base_state=TinyModel().state_dict(),
        train_loader=[batch],
        requested_device="auto",
        device=torch.device("cuda"),
        ambiguity_floor=1e-4,
        plan_loss_weight=0.2,
        margin_loss_weight=0.1,
        budget_loss_weight=0.1,
        plan_class_weights=torch.ones((3,), dtype=torch.float32),
        budget_class_weights=torch.ones((3,), dtype=torch.float32),
        weight_decay=0.0,
        candidate_lrs=(1e-3,),
        max_batches=1,
    )

    assert best_lr == pytest.approx(1e-3)
    assert seen_devices == ["cuda", "cpu"]


def test_run_with_auto_cpu_fallback_retries_operation_on_cpu(monkeypatch) -> None:
    seen_devices: list[str] = []
    seen_fallback_flags: list[bool] = []

    def _operation(device: torch.device, fallback_used: bool) -> str:
        seen_devices.append(device.type)
        seen_fallback_flags.append(fallback_used)
        if device.type == "cuda":
            raise RuntimeError("CUDA out of memory during train step")
        return "ok"

    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    with pytest.warns(RuntimeWarning, match="falling back to CPU"):
        result = train_mod.run_with_auto_cpu_fallback(
            requested_device="auto",
            device=torch.device("cuda"),
            context="unit-test",
            operation=_operation,
        )

    assert result.value == "ok"
    assert result.final_device == torch.device("cpu")
    assert result.fallback_used is True
    assert seen_devices == ["cuda", "cpu"]
    assert seen_fallback_flags == [False, True]


def test_run_with_auto_cpu_fallback_does_not_mark_initial_cpu_execution_as_fallback() -> None:
    seen_fallback_flags: list[bool] = []

    def _operation(device: torch.device, fallback_used: bool) -> str:
        assert device == torch.device("cpu")
        seen_fallback_flags.append(fallback_used)
        return "cpu-ok"

    result = train_mod.run_with_auto_cpu_fallback(
        requested_device="auto",
        device=torch.device("cpu"),
        context="unit-test",
        operation=_operation,
    )

    assert result.value == "cpu-ok"
    assert result.final_device == torch.device("cpu")
    assert result.fallback_used is False
    assert seen_fallback_flags == [False]


def test_main_surfaces_helpful_missing_dataset_error(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "hourly"
    data_root.mkdir()
    pd = pytest.importorskip("pandas")
    pd.DataFrame(
        [
            {
                "timestamp": "2024-01-02T14:30:00Z",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1_000_000.0,
                "vwap": 100.25,
                "symbol": "DBX",
            }
        ]
    ).to_csv(data_root / "DBX.csv", index=False)

    monkeypatch.setattr(train_mod, "resolve_autoresearch_training_device", lambda _device: torch.device("cpu"))
    monkeypatch.setattr(train_mod, "seed_everything", lambda *args, **kwargs: None)

    with pytest.raises(FileNotFoundError, match=r"Missing dataset for AAPL") as excinfo:
        train_mod.main(
            [
                "--frequency",
                "hourly",
                "--data-root",
                str(data_root),
                "--symbols",
                "AAPL",
                "--sequence-length",
                "8",
                "--hold-bars",
                "3",
                "--eval-windows",
                "8",
                "--disable-auto-lr-find",
                "--dashboard-db",
                str(tmp_path / "missing.db"),
            ]
        )

    message = str(excinfo.value)
    assert "DBX.csv" in message
    assert "Point --data-root at a directory containing per-symbol CSVs" in message


def test_main_rejects_path_like_symbol_input(tmp_path: Path) -> None:
    data_root = tmp_path / "hourly"
    data_root.mkdir()

    with pytest.raises(ValueError, match=r"Unsupported symbol"):
        train_mod.main(
            [
                "--frequency",
                "hourly",
                "--data-root",
                str(data_root),
                "--symbols",
                "../secret",
                "--check-inputs",
            ]
        )


def test_main_rejects_invalid_numeric_config_before_cuda(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "hourly"
    data_root.mkdir()

    monkeypatch.setattr(
        train_mod,
        "resolve_autoresearch_training_device",
        lambda _device: (_ for _ in ()).throw(AssertionError("CUDA path should not run for invalid config")),
    )

    with pytest.raises(ValueError, match=r"max_volume_fraction must be > 0 and <= 1"):
        train_mod.main(
            [
                "--frequency",
                "hourly",
                "--data-root",
                str(data_root),
                "--symbols",
                "AAPL",
                "--max-volume-fraction",
                "0",
                "--check-inputs",
            ]
        )


def test_main_rejects_conflicting_check_input_modes(capsys) -> None:
    with pytest.raises(SystemExit, match="2"):
        train_mod.main(["--check-inputs", "--check-inputs-text"])

    assert "not allowed with argument" in capsys.readouterr().err


def test_main_help_documents_default_data_roots(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        train_mod.main(["--help"])

    assert excinfo.value.code == 0
    output = capsys.readouterr().out
    assert "trainingdatahourly/stocks" in output
    assert "trainingdata for daily" in output
    assert "built-in" in output and "hourly 8-symbol set" in output


def test_main_check_inputs_reports_invalid_worker_env_cleanly(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    data_root = tmp_path / "hourly"
    data_root.mkdir()
    pd.DataFrame(
        [
            {
                "timestamp": "2024-01-02T14:30:00Z",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1_000_000.0,
                "vwap": 100.25,
                "symbol": "AAPL",
            }
        ]
    ).to_csv(data_root / "AAPL.csv", index=False)
    pd.DataFrame(
        [
            {
                "timestamp": "2024-01-02T14:30:00Z",
                "open": 200.0,
                "high": 201.0,
                "low": 199.0,
                "close": 200.5,
                "volume": 2_000_000.0,
                "vwap": 200.25,
                "symbol": "MSFT",
            }
        ]
    ).to_csv(data_root / "MSFT.csv", index=False)
    monkeypatch.setenv("AUTORESEARCH_STOCK_INPUT_CHECK_WORKERS", "not-an-int")
    monkeypatch.setattr(
        train_mod,
        "resolve_autoresearch_training_device",
        lambda _device: (_ for _ in ()).throw(AssertionError("CUDA path should not run under --check-inputs")),
    )

    exit_code = train_mod.main(
        [
            "--frequency",
            "hourly",
            "--data-root",
            str(data_root),
            "--symbols",
            "AAPL,MSFT",
            "--check-inputs",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "Autoresearch stock input check failed:" in captured.err
    assert "AUTORESEARCH_STOCK_INPUT_CHECK_WORKERS" in captured.err
    assert "positive integer" in captured.err


def test_main_check_inputs_reports_ready_symbols_without_touching_cuda(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    data_root = tmp_path / "hourly"
    data_root.mkdir()
    pd.DataFrame(
        [
            {
                "timestamp": "2024-01-02T14:30:00Z",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1_000_000.0,
                "vwap": 100.25,
                "symbol": "AAPL",
            }
        ]
    ).to_csv(data_root / "AAPL.csv", index=False)

    monkeypatch.setattr(
        train_mod,
        "resolve_autoresearch_training_device",
        lambda _device: (_ for _ in ()).throw(AssertionError("CUDA path should not run under --check-inputs")),
    )

    exit_code = train_mod.main(
        [
            "--frequency",
            "hourly",
            "--data-root",
            str(data_root),
            "--symbols",
            "AAPL",
            "--check-inputs",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["all_symbols_ready"] is True
    assert payload["worker_count"] >= 1
    assert payload["worker_source"] in {"serial", "auto", "env"}
    assert payload["status_counts"] == {
        "ready": 1,
        "missing": 0,
        "invalid": 0,
        "partial": 0,
        "other": 0,
    }
    assert payload["symbols"][0]["symbol"] == "AAPL"
    assert payload["symbols"][0]["status"] == "ready"


def test_main_check_inputs_returns_nonzero_for_missing_symbols(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    data_root = tmp_path / "hourly"
    data_root.mkdir()
    pd.DataFrame(
        [
            {
                "timestamp": "2024-01-02T14:30:00Z",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1_000_000.0,
                "vwap": 100.25,
                "symbol": "DBX",
            }
        ]
    ).to_csv(data_root / "DBX.csv", index=False)

    monkeypatch.setattr(
        train_mod,
        "resolve_autoresearch_training_device",
        lambda _device: (_ for _ in ()).throw(AssertionError("CUDA path should not run under --check-inputs")),
    )

    exit_code = train_mod.main(
        [
            "--frequency",
            "hourly",
            "--data-root",
            str(data_root),
            "--symbols",
            "AAPL",
            "--check-inputs",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["all_symbols_ready"] is False
    assert payload["data_root_preview"].endswith("DBX.csv")
    assert payload["symbols"][0]["status"] == "missing"


def test_main_check_inputs_reports_invalid_csv_without_touching_cuda(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    data_root = tmp_path / "hourly"
    data_root.mkdir()
    (data_root / "AAPL.csv").write_text(
        "timestamp,open,close\n2024-01-02T14:30:00Z,100,100.5\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        train_mod,
        "resolve_autoresearch_training_device",
        lambda _device: (_ for _ in ()).throw(AssertionError("CUDA path should not run under --check-inputs")),
    )

    exit_code = train_mod.main(
        [
            "--frequency",
            "hourly",
            "--data-root",
            str(data_root),
            "--symbols",
            "AAPL",
            "--check-inputs",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["all_symbols_ready"] is False
    assert payload["symbols"][0]["status"] == "invalid"
    assert "missing required columns" in str(payload["symbols"][0]["primary_error"])


def test_main_check_inputs_returns_nonzero_for_invalid_recent_overlay(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    data_root = tmp_path / "hourly"
    recent_root = tmp_path / "recent"
    data_root.mkdir()
    recent_root.mkdir()
    pd.DataFrame(
        [
            {
                "timestamp": "2024-01-02T14:30:00Z",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1_000_000.0,
                "vwap": 100.25,
                "symbol": "AAPL",
            }
        ]
    ).to_csv(data_root / "AAPL.csv", index=False)
    (recent_root / "AAPL.csv").write_text(
        "timestamp,open,close\n2024-01-02T14:30:00Z,100,100.5\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        train_mod,
        "resolve_autoresearch_training_device",
        lambda _device: (_ for _ in ()).throw(AssertionError("CUDA path should not run under --check-inputs")),
    )

    exit_code = train_mod.main(
        [
            "--frequency",
            "hourly",
            "--data-root",
            str(data_root),
            "--recent-data-root",
            str(recent_root),
            "--symbols",
            "AAPL",
            "--check-inputs",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["all_symbols_ready"] is False
    assert payload["status_counts"] == {
        "ready": 0,
        "missing": 0,
        "invalid": 0,
        "partial": 1,
        "other": 0,
    }
    assert payload["symbols"][0]["status"] == "partial"
    assert "missing required columns" in str(payload["symbols"][0]["recent_error"])


def test_main_check_inputs_text_reports_human_readable_summary(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    data_root = tmp_path / "hourly"
    data_root.mkdir()
    pd.DataFrame(
        [
            {
                "timestamp": "2024-01-02T14:30:00Z",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1_000_000.0,
                "vwap": 100.25,
                "symbol": "AAPL",
            }
        ]
    ).to_csv(data_root / "AAPL.csv", index=False)

    monkeypatch.setattr(
        train_mod,
        "resolve_autoresearch_training_device",
        lambda _device: (_ for _ in ()).throw(AssertionError("CUDA path should not run under --check-inputs-text")),
    )

    exit_code = train_mod.main(
        [
            "--frequency",
            "hourly",
            "--data-root",
            str(data_root),
            "--symbols",
            "AAPL",
            "--check-inputs-text",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Autoresearch Stock Input Check" in output
    assert "Input check workers: 1 (serial)" in output
    assert "Ready symbols: 1/1" in output
    assert "All symbols ready: yes" in output
    assert "Summary: ready=1, missing=0, invalid=0, partial=0, other=0" in output
    assert "Next step: rerun without --check-inputs-text to start training." in output
    assert "Suggested command:" in output
    assert (
        f"{sys.executable} -m autoresearch_stock.train --frequency hourly --data-root {data_root} --symbols AAPL"
        in output
    )
    assert (
        f"{sys.executable} -m autoresearch_stock.train --frequency hourly --data-root {data_root} --symbols AAPL --check-inputs-text"
        not in output
    )
    assert "- AAPL: ready, rows=1" in output


def test_main_check_inputs_text_surfaces_next_step_for_missing_symbols(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    data_root = tmp_path / "hourly"
    data_root.mkdir()
    pd.DataFrame(
        [
            {
                "timestamp": "2024-01-02T14:30:00Z",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1_000_000.0,
                "vwap": 100.25,
                "symbol": "DBX",
            }
        ]
    ).to_csv(data_root / "DBX.csv", index=False)

    monkeypatch.setattr(
        train_mod,
        "resolve_autoresearch_training_device",
        lambda _device: (_ for _ in ()).throw(AssertionError("CUDA path should not run under --check-inputs-text")),
    )

    exit_code = train_mod.main(
        [
            "--frequency",
            "hourly",
            "--data-root",
            str(data_root),
            "--symbols",
            "AAPL",
            "--check-inputs-text",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "Ready symbols: 0/1" in output
    assert "Summary: ready=0, missing=1, invalid=0, partial=0, other=0" in output
    assert "Next step: fix the symbols above, or rerun with --check-inputs for JSON output." in output
    assert "Suggested JSON command:" in output
    assert (
        f"{sys.executable} -m autoresearch_stock.train --frequency hourly --data-root {data_root} --symbols AAPL --check-inputs"
        in output
    )
    assert (
        f"{sys.executable} -m autoresearch_stock.train --frequency hourly --data-root {data_root} --symbols AAPL --check-inputs-text"
        not in output
    )
    assert "- AAPL: missing, rows=0" in output
