from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import autoresearch_stock.train as train_mod


def test_resolve_autoresearch_training_device_rejects_cpu_request(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    with pytest.raises(RuntimeError, match="requires CUDA"):
        train_mod.resolve_autoresearch_training_device("cpu")


def test_resolve_autoresearch_training_device_rejects_missing_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="requires CUDA"):
        train_mod.resolve_autoresearch_training_device("auto")


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
