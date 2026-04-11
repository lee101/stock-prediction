from __future__ import annotations

import concurrent.futures
import csv
import importlib.util
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.binance_hybrid_eval_defaults import build_expected_prod_eval_config


REPO = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO / "scripts" / "search_binance_prod_config_grid.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


grid = _load_module("search_binance_prod_config_grid", SCRIPT_PATH)


def _write_launch(path: Path, checkpoint: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "exec /tmp/.venv/bin/python -u \\",
                "  rl_trading_agent_binance/trade_binance_live.py \\",
                "  --live \\",
                "  --model gemini-3.1-flash-lite-preview \\",
                "  --symbols BTCUSD ETHUSD SOLUSD \\",
                "  --execution-mode margin \\",
                "  --leverage 0.5 \\",
                "  --interval 3600 \\",
                "  --fallback-mode chronos2 \\",
                f"  --rl-checkpoint {checkpoint} \\",
                '  "$@"',
            ]
        )
    )


def _write_manifest(
    path: Path,
    current_checkpoint: Path,
    candidate_checkpoint: Path | None,
    *,
    symbols: list[str],
    leverage: float,
    candidate_goodness: float | None = None,
    launch_goodness: float = -1.0,
    generated_at: str | None = None,
) -> None:
    evaluations = [
        {
            "checkpoint": str(current_checkpoint.resolve()),
            "median_total_return": -0.03,
            "median_sortino": -0.5,
            "replay": {
                "hourly_goodness_score": launch_goodness,
                "hourly_total_return": -0.02,
                "hourly_sortino": -0.4,
            },
        }
    ]
    if candidate_checkpoint is not None and candidate_goodness is not None:
        evaluations.append(
            {
                "checkpoint": str(candidate_checkpoint.resolve()),
                "median_total_return": 0.02,
                "median_sortino": 0.7,
                "replay": {
                    "hourly_goodness_score": candidate_goodness,
                    "hourly_total_return": 0.05,
                    "hourly_sortino": 0.9,
                },
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "generated_at": generated_at or datetime.now(UTC).isoformat(),
                "launch_config": {
                    "launch_script": "/tmp/launch.sh",
                    "python_bin": "/tmp/.venv/bin/python",
                    "trade_script": "rl_trading_agent_binance/trade_binance_live.py",
                    "model": "gemini-3.1-flash-lite-preview",
                    "symbols": symbols,
                    "execution_mode": "margin",
                    "leverage": leverage,
                    "interval": 3600,
                    "fallback_mode": "chronos2",
                    "rl_checkpoint": str(current_checkpoint.resolve()),
                },
                "eval_config": build_expected_prod_eval_config(),
                "current_runtime_audit": None,
                "current_runtime_audit_issues": [],
                "current_runtime_health_issues": [],
                "evaluations": evaluations,
            }
        )
    )


def _write_same_checkpoint_multi_target_manifest(
    path: Path,
    checkpoint: Path,
    *,
    symbols: list[str],
    leverage: float,
    launch_goodness: float,
    running_symbols: list[str],
    running_leverage: float,
    running_goodness: float,
    generated_at: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "generated_at": generated_at or datetime.now(UTC).isoformat(),
                "launch_config": {
                    "launch_script": "/tmp/launch.sh",
                    "python_bin": "/tmp/.venv/bin/python",
                    "trade_script": "rl_trading_agent_binance/trade_binance_live.py",
                    "model": "gemini-3.1-flash-lite-preview",
                    "symbols": symbols,
                    "execution_mode": "margin",
                    "leverage": leverage,
                    "interval": 3600,
                    "fallback_mode": "chronos2",
                    "rl_checkpoint": str(checkpoint.resolve()),
                },
                "eval_config": build_expected_prod_eval_config(),
                "evaluation_targets": [
                    {
                        "label": "running_hybrid",
                        "config": {
                            "launch_script": "pid=1712424",
                            "python_bin": "/tmp/.venv/bin/python",
                            "trade_script": "rl_trading_agent_binance/trade_binance_live.py",
                            "model": "gemini-3.1-flash-lite-preview",
                            "symbols": running_symbols,
                            "execution_mode": "margin",
                            "leverage": running_leverage,
                            "interval": 3600,
                            "fallback_mode": "chronos2",
                            "rl_checkpoint": str(checkpoint.resolve()),
                        },
                        "checkpoints": [str(checkpoint.resolve())],
                    },
                    {
                        "label": "launch_target",
                        "config": {
                            "launch_script": "/tmp/launch.sh",
                            "python_bin": "/tmp/.venv/bin/python",
                            "trade_script": "rl_trading_agent_binance/trade_binance_live.py",
                            "model": "gemini-3.1-flash-lite-preview",
                            "symbols": symbols,
                            "execution_mode": "margin",
                            "leverage": leverage,
                            "interval": 3600,
                            "fallback_mode": "chronos2",
                            "rl_checkpoint": str(checkpoint.resolve()),
                        },
                        "checkpoints": [str(checkpoint.resolve())],
                    },
                ],
                "current_running_hybrid_config": {
                    "launch_script": "pid=1712424",
                    "python_bin": "/tmp/.venv/bin/python",
                    "trade_script": "rl_trading_agent_binance/trade_binance_live.py",
                    "model": "gemini-3.1-flash-lite-preview",
                    "symbols": running_symbols,
                    "execution_mode": "margin",
                    "leverage": running_leverage,
                    "interval": 3600,
                    "fallback_mode": "chronos2",
                    "rl_checkpoint": str(checkpoint.resolve()),
                },
                "current_machine_audit_issues": [
                    "running hybrid process does not match launch config: symbols, leverage"
                ],
                "current_runtime_audit_issues": [
                    "24 recent live snapshot(s) disagree with the launch symbols"
                ],
                "current_runtime_health_issues": [],
                "evaluations": [
                    {
                        "target_label": "running_hybrid",
                        "symbols": running_symbols,
                        "leverage": running_leverage,
                        "checkpoint": str(checkpoint.resolve()),
                        "median_total_return": 0.01,
                        "median_sortino": 0.2,
                        "replay": {
                            "hourly_goodness_score": running_goodness,
                            "hourly_total_return": 0.02,
                            "hourly_sortino": 0.4,
                        },
                    },
                    {
                        "target_label": "launch_target",
                        "symbols": symbols,
                        "leverage": leverage,
                        "checkpoint": str(checkpoint.resolve()),
                        "median_total_return": 0.03,
                        "median_sortino": 0.8,
                        "replay": {
                            "hourly_goodness_score": launch_goodness,
                            "hourly_total_return": 0.05,
                            "hourly_sortino": 1.1,
                        },
                    },
                ],
            }
        )
    )


def _write_running_hybrid_drift_manifest(
    path: Path,
    launch_checkpoint: Path,
    running_checkpoint: Path,
    candidate_checkpoint: Path,
    *,
    symbols: list[str],
    leverage: float,
    running_goodness: float,
    candidate_goodness: float,
    generated_at: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "generated_at": generated_at or datetime.now(UTC).isoformat(),
                "launch_config": {
                    "launch_script": "/tmp/launch.sh",
                    "python_bin": "/tmp/.venv/bin/python",
                    "trade_script": "rl_trading_agent_binance/trade_binance_live.py",
                    "model": "gemini-3.1-flash-lite-preview",
                    "symbols": symbols,
                    "execution_mode": "margin",
                    "leverage": leverage,
                    "interval": 3600,
                    "fallback_mode": "chronos2",
                    "rl_checkpoint": str(launch_checkpoint.resolve()),
                },
                "eval_config": build_expected_prod_eval_config(),
                "evaluation_targets": [
                    {
                        "label": "launch_target",
                        "config": {
                            "launch_script": "/tmp/launch.sh",
                            "python_bin": "/tmp/.venv/bin/python",
                            "trade_script": "rl_trading_agent_binance/trade_binance_live.py",
                            "model": "gemini-3.1-flash-lite-preview",
                            "symbols": symbols,
                            "execution_mode": "margin",
                            "leverage": leverage,
                            "interval": 3600,
                            "fallback_mode": "chronos2",
                            "rl_checkpoint": str(launch_checkpoint.resolve()),
                        },
                        "checkpoints": [
                            str(launch_checkpoint.resolve()),
                            str(candidate_checkpoint.resolve()),
                        ],
                    },
                    {
                        "label": "running_hybrid",
                        "config": {
                            "launch_script": "pid=1712424",
                            "python_bin": "/tmp/.venv/bin/python",
                            "trade_script": "rl_trading_agent_binance/trade_binance_live.py",
                            "model": "gemini-3.1-flash-lite-preview",
                            "symbols": symbols,
                            "execution_mode": "margin",
                            "leverage": leverage,
                            "interval": 3600,
                            "fallback_mode": "chronos2",
                            "rl_checkpoint": str(running_checkpoint.resolve()),
                        },
                        "checkpoints": [str(running_checkpoint.resolve())],
                    },
                ],
                "current_running_hybrid_config": {
                    "launch_script": "pid=1712424",
                    "python_bin": "/tmp/.venv/bin/python",
                    "trade_script": "rl_trading_agent_binance/trade_binance_live.py",
                    "model": "gemini-3.1-flash-lite-preview",
                    "symbols": symbols,
                    "execution_mode": "margin",
                    "leverage": leverage,
                    "interval": 3600,
                    "fallback_mode": "chronos2",
                    "rl_checkpoint": str(running_checkpoint.resolve()),
                },
                "current_machine_audit_issues": [
                    "running hybrid process does not match launch config: rl_checkpoint"
                ],
                "current_runtime_audit_issues": [
                    "24 recent live snapshot(s) disagree with the launch checkpoint"
                ],
                "current_runtime_health_issues": [],
                "evaluations": [
                    {
                        "target_label": "launch_target",
                        "symbols": symbols,
                        "leverage": leverage,
                        "checkpoint": str(launch_checkpoint.resolve()),
                        "median_total_return": -0.02,
                        "median_sortino": -0.5,
                        "replay": {
                            "hourly_goodness_score": -1.0,
                            "hourly_total_return": -0.03,
                            "hourly_sortino": -0.4,
                        },
                    },
                    {
                        "target_label": "running_hybrid",
                        "symbols": symbols,
                        "leverage": leverage,
                        "checkpoint": str(running_checkpoint.resolve()),
                        "median_total_return": 0.03,
                        "median_sortino": 0.9,
                        "replay": {
                            "hourly_goodness_score": running_goodness,
                            "hourly_total_return": 0.05,
                            "hourly_sortino": 1.1,
                        },
                    },
                    {
                        "target_label": "launch_target",
                        "symbols": symbols,
                        "leverage": leverage,
                        "checkpoint": str(candidate_checkpoint.resolve()),
                        "median_total_return": 0.01,
                        "median_sortino": 0.2,
                        "replay": {
                            "hourly_goodness_score": candidate_goodness,
                            "hourly_total_return": 0.04,
                            "hourly_sortino": 0.8,
                        },
                    },
                ],
            }
        )
    )




def test_build_config_variants_dedupes_normalized_variants(tmp_path: Path) -> None:
    launch = tmp_path / "launch.sh"
    checkpoint = tmp_path / "current.pt"
    checkpoint.write_text("x")
    _write_launch(launch, checkpoint)

    variants = grid.build_config_variants(
        launch_script=launch,
        symbol_set_specs=["btcusd ethusd", "BTCUSD,ETHUSD", "BTCUSD ETHUSD SOLUSD"],
        symbol_subset_sizes=[],
        leverage_options=[0.5, 2.0, 2.0],
        output_root=tmp_path / "out",
    )

    assert [variant.symbols for variant in variants] == [
        ("BTCUSD", "ETHUSD"),
        ("BTCUSD", "ETHUSD"),
        ("BTCUSD", "ETHUSD", "SOLUSD"),
        ("BTCUSD", "ETHUSD", "SOLUSD"),
    ]
    assert [variant.leverage for variant in variants] == [0.5, 2.0, 0.5, 2.0]
    assert len({variant.slug for variant in variants}) == 4


def test_build_config_variants_generates_launch_symbol_subsets(tmp_path: Path) -> None:
    launch = tmp_path / "launch.sh"
    checkpoint = tmp_path / "current.pt"
    checkpoint.write_text("x")
    _write_launch(launch, checkpoint)

    variants = grid.build_config_variants(
        launch_script=launch,
        symbol_set_specs=[],
        symbol_subset_sizes=[2],
        leverage_options=[0.5],
        output_root=tmp_path / "out",
    )

    assert [variant.symbols for variant in variants] == [
        ("BTCUSD", "ETHUSD"),
        ("BTCUSD", "SOLUSD"),
        ("ETHUSD", "SOLUSD"),
    ]


def test_select_best_rows_can_include_launch_rows_when_requested() -> None:
    rows = [
        grid.ConfigSearchRow(
            config_slug="cfg",
            symbols="BTCUSD ETHUSD",
            leverage=0.5,
            checkpoint="/tmp/current.pt",
            is_launch_checkpoint=True,
            metric_name="replay.hourly_goodness_score",
            metric_value=10.0,
            median_total_return=0.3,
            median_sortino=1.5,
            replay_hourly_goodness_score=10.0,
            manifest_path="/tmp/manifest.json",
        ),
        grid.ConfigSearchRow(
            config_slug="cfg",
            symbols="BTCUSD ETHUSD",
            leverage=0.5,
            checkpoint="/tmp/candidate.pt",
            is_launch_checkpoint=False,
            metric_name="replay.hourly_goodness_score",
            metric_value=1.0,
            median_total_return=0.1,
            median_sortino=0.5,
            replay_hourly_goodness_score=1.0,
            manifest_path="/tmp/manifest.json",
        ),
    ]

    best_rows = grid.select_best_rows(
        rows,
        candidate_checkpoints_requested=True,
        include_launch_checkpoint=True,
    )

    assert len(best_rows) == 1
    assert best_rows[0].checkpoint == "/tmp/current.pt"



def test_select_best_rows_omits_plain_live_baseline_when_candidates_requested() -> None:
    rows = [
        grid.ConfigSearchRow(
            config_slug="cfg",
            symbols="BTCUSD ETHUSD",
            leverage=0.5,
            checkpoint="/tmp/current.pt",
            is_launch_checkpoint=True,
            metric_name="replay.hourly_goodness_score",
            metric_value=1.0,
            median_total_return=0.01,
            median_sortino=0.3,
            replay_hourly_goodness_score=1.0,
            manifest_path="/tmp/manifest.json",
            gate_metric_name="replay.hourly_goodness_score",
            gate_current_metric=1.0,
            gate_candidate_metric=1.0,
            gate_allowed=True,
            gate_reason="candidate already matches current live checkpoint",
        ),
    ]

    best_rows = grid.select_best_rows(rows, candidate_checkpoints_requested=True)

    assert best_rows == []


def test_select_best_rows_keeps_same_checkpoint_config_change_when_candidates_requested() -> None:
    rows = [
        grid.ConfigSearchRow(
            config_slug="cfg",
            symbols="BTCUSD ETHUSD",
            leverage=2.0,
            checkpoint="/tmp/current.pt",
            is_launch_checkpoint=True,
            metric_name="replay.hourly_goodness_score",
            metric_value=1.5,
            median_total_return=0.04,
            median_sortino=1.3,
            replay_hourly_goodness_score=1.5,
            manifest_path="/tmp/manifest.json",
            gate_metric_name="replay.hourly_goodness_score",
            gate_current_metric=0.6,
            gate_candidate_metric=1.5,
            gate_allowed=True,
            gate_reason="candidate passes deploy gate",
        ),
        grid.ConfigSearchRow(
            config_slug="cfg",
            symbols="BTCUSD ETHUSD",
            leverage=2.0,
            checkpoint="/tmp/candidate.pt",
            is_launch_checkpoint=False,
            metric_name="replay.hourly_goodness_score",
            metric_value=0.8,
            median_total_return=0.02,
            median_sortino=0.7,
            replay_hourly_goodness_score=0.8,
            manifest_path="/tmp/manifest.json",
            gate_metric_name="replay.hourly_goodness_score",
            gate_current_metric=0.6,
            gate_candidate_metric=0.8,
            gate_allowed=True,
            gate_reason="candidate passes deploy gate",
        ),
    ]

    best_rows = grid.select_best_rows(rows, candidate_checkpoints_requested=True)

    assert len(best_rows) == 1
    assert best_rows[0].checkpoint == "/tmp/current.pt"


def test_select_best_rows_prefers_candidate_over_launch_rows() -> None:
    rows = [
        grid.ConfigSearchRow(
            config_slug="cfg",
            symbols="BTCUSD ETHUSD",
            leverage=0.5,
            checkpoint="/tmp/current.pt",
            is_launch_checkpoint=True,
            metric_name="replay.hourly_goodness_score",
            metric_value=10.0,
            median_total_return=-0.1,
            median_sortino=-0.5,
            replay_hourly_goodness_score=10.0,
            manifest_path="/tmp/manifest.json",
        ),
        grid.ConfigSearchRow(
            config_slug="cfg",
            symbols="BTCUSD ETHUSD",
            leverage=0.5,
            checkpoint="/tmp/candidate.pt",
            is_launch_checkpoint=False,
            metric_name="replay.hourly_goodness_score",
            metric_value=1.0,
            median_total_return=0.1,
            median_sortino=0.5,
            replay_hourly_goodness_score=1.0,
            manifest_path="/tmp/manifest.json",
        ),
    ]

    best_rows = grid.select_best_rows(rows, candidate_checkpoints_requested=True)

    assert len(best_rows) == 1
    assert best_rows[0].checkpoint == "/tmp/candidate.pt"


def test_load_manifest_rows_prefers_launch_target_evaluation_when_checkpoint_is_reused(tmp_path: Path) -> None:
    checkpoint = tmp_path / "current.pt"
    checkpoint.write_text("x")
    manifest = tmp_path / "prod_launch_eval_manifest.json"
    _write_same_checkpoint_multi_target_manifest(
        manifest,
        checkpoint,
        symbols=["BTCUSD", "ETHUSD", "SOLUSD"],
        leverage=0.5,
        launch_goodness=1.2,
        running_symbols=["BTCUSD", "ETHUSD"],
        running_leverage=1.5,
        running_goodness=0.4,
    )

    rows = grid.load_manifest_rows(manifest)

    assert len(rows) == 1
    assert rows[0].checkpoint == str(checkpoint.resolve())
    assert rows[0].metric_name == "replay.hourly_goodness_score"
    assert rows[0].metric_value == pytest.approx(1.2)
    assert rows[0].symbols == "BTCUSD ETHUSD SOLUSD"
    assert rows[0].leverage == pytest.approx(0.5)



def test_load_manifest_rows_reuses_single_payload_for_gate_annotation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launch = tmp_path / "launch.sh"
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    candidate_checkpoint = tmp_path / "candidate.pt"
    candidate_checkpoint.write_text("y")
    _write_launch(launch, current_checkpoint)
    launch = tmp_path / "launch.sh"
    _write_launch(launch, current_checkpoint)
    manifest = tmp_path / "prod_launch_eval_manifest.json"
    _write_manifest(
        manifest,
        current_checkpoint,
        candidate_checkpoint,
        symbols=["BTCUSD", "ETHUSD", "SOLUSD"],
        leverage=0.5,
        candidate_goodness=2.0,
    )

    payload_ids: list[int] = []
    seen_launch_cfg = []
    seen_target_cfg = []
    real_helper = grid.gate_deploy_candidate_from_payload

    def _recording_helper(*, payload, launch_cfg=None, target_launch_cfg=None, **kwargs):
        payload_ids.append(id(payload))
        seen_launch_cfg.append(launch_cfg)
        seen_target_cfg.append(target_launch_cfg)
        return real_helper(
            payload=payload,
            launch_cfg=launch_cfg,
            target_launch_cfg=target_launch_cfg,
            **kwargs,
        )

    monkeypatch.setattr(grid, "gate_deploy_candidate_from_payload", _recording_helper)

    rows = grid.load_manifest_rows(manifest, launch_script=launch)

    assert len(rows) == 2
    assert len(set(payload_ids)) == 1
    assert all(cfg is not None for cfg in seen_launch_cfg)
    assert all(cfg is not None for cfg in seen_target_cfg)


def test_load_manifest_rows_passes_require_process_isolation_to_gate_annotation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    candidate_checkpoint = tmp_path / "candidate.pt"
    candidate_checkpoint.write_text("y")
    launch = tmp_path / "launch.sh"
    _write_launch(launch, current_checkpoint)
    manifest = tmp_path / "prod_launch_eval_manifest.json"
    _write_manifest(
        manifest,
        current_checkpoint,
        candidate_checkpoint,
        symbols=["BTCUSD", "ETHUSD", "SOLUSD"],
        leverage=0.5,
        candidate_goodness=2.0,
    )

    seen_require_process_isolation: list[bool] = []

    def _fake_gate_deploy_candidate_from_payload(**kwargs):
        seen_require_process_isolation.append(bool(kwargs.get("require_process_isolation")))
        return type(
            "GateResultStub",
            (),
            {
                "allowed": not kwargs.get("require_process_isolation"),
                "reason": "ok",
                "launch_script": str(Path(kwargs["launch_script"]).resolve()),
                "current_checkpoint": str(current_checkpoint.resolve()),
                "candidate_checkpoint": str(Path(kwargs["candidate_checkpoint"]).resolve()),
                "manifest_path": str(manifest.resolve()),
                "metric_name": "replay.hourly_goodness_score",
                "current_metric": -1.0,
                "candidate_metric": 2.0,
                "current_target_label": "launch_target",
                "current_symbols": "BTCUSD ETHUSD SOLUSD",
                "current_leverage": 0.5,
            },
        )()

    monkeypatch.setattr(grid, "audit_binance_hybrid_machine_state", lambda launch_script: object())
    monkeypatch.setattr(grid, "build_machine_deploy_preflight_reason", lambda audit_result: None)
    monkeypatch.setattr(grid, "gate_deploy_candidate_from_payload", _fake_gate_deploy_candidate_from_payload)

    rows = grid.load_manifest_rows(manifest, launch_script=launch, require_process_isolation=True)

    assert len(rows) == 2
    assert seen_require_process_isolation == [True, True]
    assert all(row.gate_allowed is False for row in rows)



def test_load_manifest_rows_reuses_single_process_isolation_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    candidate_checkpoint = tmp_path / "candidate.pt"
    candidate_checkpoint.write_text("y")
    launch = tmp_path / "launch.sh"
    _write_launch(launch, current_checkpoint)
    manifest = tmp_path / "prod_launch_eval_manifest.json"
    _write_manifest(
        manifest,
        current_checkpoint,
        candidate_checkpoint,
        symbols=["BTCUSD", "ETHUSD", "SOLUSD"],
        leverage=0.5,
        candidate_goodness=2.0,
    )

    audit_calls: list[str] = []
    build_calls: list[object] = []
    seen_checked: list[bool] = []
    seen_reasons: list[str | None] = []

    def _fake_audit(launch_script):
        audit_calls.append(str(Path(launch_script).resolve()))
        return object()

    def _fake_build(audit_result):
        build_calls.append(audit_result)
        return "binance live process isolation failed: blocked for test"

    def _fake_gate_deploy_candidate_from_payload(**kwargs):
        seen_checked.append(bool(kwargs.get("process_isolation_preflight_checked")))
        seen_reasons.append(kwargs.get("process_isolation_preflight_reason"))
        return type(
            "GateResultStub",
            (),
            {
                "allowed": False,
                "reason": kwargs.get("process_isolation_preflight_reason"),
                "launch_script": str(Path(kwargs["launch_script"]).resolve()),
                "current_checkpoint": str(current_checkpoint.resolve()),
                "candidate_checkpoint": str(Path(kwargs["candidate_checkpoint"]).resolve()),
                "manifest_path": str(manifest.resolve()),
                "metric_name": "replay.hourly_goodness_score",
                "current_metric": -1.0,
                "candidate_metric": 2.0,
                "current_target_label": "launch_target",
                "current_symbols": "BTCUSD ETHUSD SOLUSD",
                "current_leverage": 0.5,
            },
        )()

    monkeypatch.setattr(grid, "audit_binance_hybrid_machine_state", _fake_audit)
    monkeypatch.setattr(grid, "build_machine_deploy_preflight_reason", _fake_build)
    monkeypatch.setattr(grid, "gate_deploy_candidate_from_payload", _fake_gate_deploy_candidate_from_payload)

    rows = grid.load_manifest_rows(manifest, launch_script=launch, require_process_isolation=True)

    assert len(rows) == 2
    assert audit_calls == [str(launch.resolve())]
    assert len(build_calls) == 1
    assert seen_checked == [True, True]
    assert seen_reasons == [
        "binance live process isolation failed: blocked for test",
        "binance live process isolation failed: blocked for test",
    ]
    assert all(row.gate_allowed is False for row in rows)


def test_load_manifest_rows_can_use_precomputed_process_isolation_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    candidate_checkpoint = tmp_path / "candidate.pt"
    candidate_checkpoint.write_text("y")
    launch = tmp_path / "launch.sh"
    _write_launch(launch, current_checkpoint)
    manifest = tmp_path / "prod_launch_eval_manifest.json"
    _write_manifest(
        manifest,
        current_checkpoint,
        candidate_checkpoint,
        symbols=["BTCUSD", "ETHUSD", "SOLUSD"],
        leverage=0.5,
        candidate_goodness=2.0,
    )

    seen_checked: list[bool] = []
    seen_reasons: list[str | None] = []

    def _unexpected(*_args, **_kwargs):
        raise AssertionError("machine audit should not run when preflight is provided")

    def _fake_gate_deploy_candidate_from_payload(**kwargs):
        seen_checked.append(bool(kwargs.get("process_isolation_preflight_checked")))
        seen_reasons.append(kwargs.get("process_isolation_preflight_reason"))
        return type(
            "GateResultStub",
            (),
            {
                "allowed": False,
                "reason": kwargs.get("process_isolation_preflight_reason"),
                "launch_script": str(Path(kwargs["launch_script"]).resolve()),
                "current_checkpoint": str(current_checkpoint.resolve()),
                "candidate_checkpoint": str(Path(kwargs["candidate_checkpoint"]).resolve()),
                "manifest_path": str(manifest.resolve()),
                "metric_name": "replay.hourly_goodness_score",
                "current_metric": -1.0,
                "candidate_metric": 2.0,
                "current_target_label": "launch_target",
                "current_symbols": "BTCUSD ETHUSD SOLUSD",
                "current_leverage": 0.5,
            },
        )()

    monkeypatch.setattr(grid, "audit_binance_hybrid_machine_state", _unexpected)
    monkeypatch.setattr(grid, "build_machine_deploy_preflight_reason", _unexpected)
    monkeypatch.setattr(grid, "gate_deploy_candidate_from_payload", _fake_gate_deploy_candidate_from_payload)

    rows = grid.load_manifest_rows(
        manifest,
        launch_script=launch,
        require_process_isolation=True,
        process_isolation_preflight_checked=True,
        process_isolation_preflight_reason="blocked by shared preflight",
    )

    assert len(rows) == 2
    assert seen_checked == [True, True]
    assert seen_reasons == ["blocked by shared preflight", "blocked by shared preflight"]
    assert all(row.gate_allowed is False for row in rows)


def test_load_manifest_rows_reuses_single_prepared_gate_payload_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    candidate_checkpoint = tmp_path / "candidate.pt"
    candidate_checkpoint.write_text("y")
    launch = tmp_path / "launch.sh"
    _write_launch(launch, current_checkpoint)
    manifest = tmp_path / "prod_launch_eval_manifest.json"
    _write_manifest(
        manifest,
        current_checkpoint,
        candidate_checkpoint,
        symbols=["BTCUSD", "ETHUSD", "SOLUSD"],
        leverage=0.5,
        candidate_goodness=2.0,
    )

    prepared_contexts = []
    seen_context_ids: list[int] = []
    real_prepare = grid.prepare_gate_payload_context
    real_helper = grid.gate_deploy_candidate_from_payload

    def _recording_prepare(*args, **kwargs):
        context = real_prepare(*args, **kwargs)
        prepared_contexts.append(context)
        return context

    def _recording_helper(*, prepared_payload_context=None, **kwargs):
        seen_context_ids.append(id(prepared_payload_context))
        return real_helper(prepared_payload_context=prepared_payload_context, **kwargs)

    monkeypatch.setattr(grid, "prepare_gate_payload_context", _recording_prepare)
    monkeypatch.setattr(grid, "gate_deploy_candidate_from_payload", _recording_helper)

    rows = grid.load_manifest_rows(manifest, launch_script=launch)

    assert len(rows) == 2
    assert len(prepared_contexts) == 1
    assert seen_context_ids == [id(prepared_contexts[0]), id(prepared_contexts[0])]



def test_merge_manifest_payloads_preserves_same_checkpoint_target_variants() -> None:
    checkpoint = Path("/tmp/current.pt")
    candidate = Path("/tmp/candidate.pt")
    existing_payload = {
        "evaluations": [
            {
                "target_label": "running_hybrid",
                "symbols": ["BTCUSD", "ETHUSD"],
                "leverage": 1.5,
                "checkpoint": str(checkpoint),
                "replay": {"hourly_goodness_score": 0.4},
            }
        ]
    }
    incremental_payload = {
        "evaluations": [
            {
                "target_label": "launch_target",
                "symbols": ["BTCUSD", "ETHUSD", "SOLUSD"],
                "leverage": 0.5,
                "checkpoint": str(checkpoint),
                "replay": {"hourly_goodness_score": 1.2},
            },
            {
                "checkpoint": str(candidate),
                "replay": {"hourly_goodness_score": 2.0},
            },
        ]
    }

    merged = grid._merge_manifest_payloads(existing_payload, incremental_payload)

    identities = {
        (
            evaluation.get("checkpoint"),
            evaluation.get("target_label"),
            tuple(evaluation.get("symbols", [])),
            evaluation.get("leverage"),
        )
        for evaluation in merged["evaluations"]
    }
    assert identities == {
        (str(checkpoint), "running_hybrid", ("BTCUSD", "ETHUSD"), 1.5),
        (str(checkpoint), "launch_target", ("BTCUSD", "ETHUSD", "SOLUSD"), 0.5),
        (str(candidate), None, (), None),
    }



def test_run_grid_search_reruns_when_manifest_only_has_running_hybrid_eval_for_launch_checkpoint_same_checkpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    launch = tmp_path / "launch.sh"
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    _write_launch(launch, current_checkpoint)

    output_root = tmp_path / "grid"
    variant = grid.build_config_variants(
        launch_script=launch,
        symbol_set_specs=[],
        symbol_subset_sizes=[],
        leverage_options=[],
        output_root=output_root,
    )[0]
    manifest_path = Path(variant.output_dir) / "prod_launch_eval_manifest.json"
    _write_same_checkpoint_multi_target_manifest(
        manifest_path,
        current_checkpoint,
        symbols=list(variant.symbols),
        leverage=variant.leverage,
        launch_goodness=1.2,
        running_symbols=["BTCUSD", "ETHUSD"],
        running_leverage=1.5,
        running_goodness=0.4,
    )
    payload = json.loads(manifest_path.read_text())
    payload["evaluations"] = [
        evaluation
        for evaluation in payload["evaluations"]
        if evaluation.get("target_label") == "running_hybrid"
    ]
    manifest_path.write_text(json.dumps(payload))

    run_calls: list[list[str]] = []

    def _fake_run(cmd, cwd, capture_output, text, check):
        run_calls.append(cmd)
        _write_same_checkpoint_multi_target_manifest(
            manifest_path,
            current_checkpoint,
            symbols=list(variant.symbols),
            leverage=variant.leverage,
            launch_goodness=1.2,
            running_symbols=["BTCUSD", "ETHUSD"],
            running_leverage=1.5,
            running_goodness=0.4,
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(grid.subprocess, "run", _fake_run)

    args = grid.parse_args(
        [
            "--launch-script",
            str(launch),
            "--output-dir",
            str(output_root),
        ]
    )

    assert grid.run_grid_search(args) == 0
    assert len(run_calls) == 1
    all_rows = list(csv.DictReader((output_root / "all_results.csv").open()))
    assert len(all_rows) == 1
    assert all_rows[0]["checkpoint"] == str(current_checkpoint.resolve())
    assert float(all_rows[0]["metric_value"]) == pytest.approx(1.2)


def test_run_grid_search_writes_ranked_summary(tmp_path: Path, monkeypatch) -> None:
    launch = tmp_path / "launch.sh"
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    candidate_checkpoint = tmp_path / "candidate.pt"
    candidate_checkpoint.write_text("y")
    _write_launch(launch, current_checkpoint)

    def _fake_run(cmd, cwd, capture_output, text, check):
        assert check is False
        output_dir = Path(cmd[cmd.index("--output-dir") + 1])
        symbols = cmd[cmd.index("--symbols") + 1].split(",")
        leverage = float(cmd[cmd.index("--leverage") + 1])
        candidate_goodness = 1.0 + leverage
        if symbols == ["BTCUSD", "ETHUSD"]:
            candidate_goodness += 0.5
        _write_manifest(
            output_dir / "prod_launch_eval_manifest.json",
            current_checkpoint,
            candidate_checkpoint,
            symbols=symbols,
            leverage=leverage,
            candidate_goodness=candidate_goodness,
            launch_goodness=10.0,
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(grid.subprocess, "run", _fake_run)

    args = grid.parse_args(
        [
            "--launch-script",
            str(launch),
            "--output-dir",
            str(tmp_path / "grid"),
            "--symbols-set",
            "BTCUSD ETHUSD",
            "--symbols-set",
            "BTCUSD ETHUSD SOLUSD",
            "--leverage-option",
            "0.5",
            "--leverage-option",
            "2.0",
            "--candidate-checkpoint",
            str(candidate_checkpoint),
        ]
    )

    assert grid.run_grid_search(args) == 0

    best_rows_path = tmp_path / "grid" / "best_by_config.csv"
    rows = list(csv.DictReader(best_rows_path.open()))
    assert len(rows) == 4
    assert rows[0]["symbols"] == "BTCUSD ETHUSD"
    assert rows[0]["checkpoint"] == str(current_checkpoint.resolve())
    assert rows[0]["is_launch_checkpoint"] == "True"
    assert float(rows[0]["metric_value"]) == pytest.approx(10.0)
    assert rows[0]["gate_allowed"] == "True"


def test_run_grid_search_can_select_launch_checkpoint_config_when_requested(tmp_path: Path, monkeypatch) -> None:
    launch = tmp_path / "launch.sh"
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    candidate_checkpoint = tmp_path / "candidate.pt"
    candidate_checkpoint.write_text("y")
    _write_launch(launch, current_checkpoint)

    def _fake_run(cmd, cwd, capture_output, text, check):
        assert check is False
        output_dir = Path(cmd[cmd.index("--output-dir") + 1])
        symbols = cmd[cmd.index("--symbols") + 1].split(",")
        leverage = float(cmd[cmd.index("--leverage") + 1])
        if symbols == ["BTCUSD", "ETHUSD"]:
            _write_manifest(
                output_dir / "prod_launch_eval_manifest.json",
                current_checkpoint,
                candidate_checkpoint,
                symbols=symbols,
                leverage=leverage,
                candidate_goodness=0.8,
                launch_goodness=1.4,
            )
        else:
            _write_manifest(
                output_dir / "prod_launch_eval_manifest.json",
                current_checkpoint,
                candidate_checkpoint,
                symbols=symbols,
                leverage=leverage,
                candidate_goodness=0.5,
                launch_goodness=0.4,
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(grid.subprocess, "run", _fake_run)

    args = grid.parse_args(
        [
            "--launch-script",
            str(launch),
            "--output-dir",
            str(tmp_path / "grid"),
            "--symbols-set",
            "BTCUSD ETHUSD",
            "--candidate-checkpoint",
            str(candidate_checkpoint),
            "--include-launch-checkpoint",
        ]
    )

    assert grid.run_grid_search(args) == 0

    best_rows = list(csv.DictReader((tmp_path / "grid" / "best_by_config.csv").open()))
    assert best_rows[0]["checkpoint"] == str(current_checkpoint.resolve())
    assert best_rows[0]["symbols"] == "BTCUSD ETHUSD"
    assert best_rows[0]["is_launch_checkpoint"] == "True"



def test_run_grid_search_can_select_current_checkpoint_config_change_without_include_launch_checkpoint(tmp_path: Path, monkeypatch) -> None:
    launch = tmp_path / "launch.sh"
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    candidate_checkpoint = tmp_path / "candidate.pt"
    candidate_checkpoint.write_text("y")
    _write_launch(launch, current_checkpoint)

    def _fake_run(cmd, cwd, capture_output, text, check):
        assert check is False
        output_dir = Path(cmd[cmd.index("--output-dir") + 1])
        symbols = cmd[cmd.index("--symbols") + 1].split(",")
        leverage = float(cmd[cmd.index("--leverage") + 1])
        if symbols == ["BTCUSD", "ETHUSD"]:
            _write_manifest(
                output_dir / "prod_launch_eval_manifest.json",
                current_checkpoint,
                candidate_checkpoint,
                symbols=symbols,
                leverage=leverage,
                candidate_goodness=0.8,
                launch_goodness=1.4,
            )
        else:
            _write_manifest(
                output_dir / "prod_launch_eval_manifest.json",
                current_checkpoint,
                candidate_checkpoint,
                symbols=symbols,
                leverage=leverage,
                candidate_goodness=0.5,
                launch_goodness=0.4,
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(grid.subprocess, "run", _fake_run)

    args = grid.parse_args(
        [
            "--launch-script",
            str(launch),
            "--output-dir",
            str(tmp_path / "grid"),
            "--symbols-set",
            "BTCUSD ETHUSD",
            "--candidate-checkpoint",
            str(candidate_checkpoint),
        ]
    )

    assert grid.run_grid_search(args) == 0

    best_rows = list(csv.DictReader((tmp_path / "grid" / "best_by_config.csv").open()))
    assert best_rows[0]["checkpoint"] == str(current_checkpoint.resolve())
    assert best_rows[0]["symbols"] == "BTCUSD ETHUSD"
    assert best_rows[0]["is_launch_checkpoint"] == "True"


def test_run_grid_search_fails_when_only_plain_live_baseline_rows_remain(tmp_path: Path, monkeypatch, capsys) -> None:
    launch = tmp_path / "launch.sh"
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    _write_launch(launch, current_checkpoint)

    def _fake_run(cmd, cwd, capture_output, text, check):
        assert check is False
        output_dir = Path(cmd[cmd.index("--output-dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "prod_launch_eval_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "launch_config": {
                        "rl_checkpoint": str(current_checkpoint.resolve()),
                        "symbols": ["BTCUSD", "ETHUSD", "SOLUSD"],
                        "leverage": 0.5,
                    },
                    "generated_at": datetime.now(UTC).isoformat(),
                    "expected_prod_eval_config": build_expected_prod_eval_config(),
                    "evaluations": [
                        {
                            "checkpoint": str(current_checkpoint.resolve()),
                            "median_total_return": 0.01,
                            "median_sortino": 0.3,
                            "replay": {
                                "hourly_goodness_score": 1.0,
                                "hourly_total_return": 0.01,
                                "hourly_sortino": 0.3,
                            },
                        }
                    ],
                }
            )
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(grid.subprocess, "run", _fake_run)

    args = grid.parse_args(
        [
            "--launch-script",
            str(launch),
            "--output-dir",
            str(tmp_path / "grid"),
            "--symbols-set",
            "BTCUSD ETHUSD SOLUSD",
            "--candidate-checkpoint",
            str(current_checkpoint),
        ]
    )

    assert grid.run_grid_search(args) == 2

    captured = capsys.readouterr()
    assert "no actionable production configs found" in captured.err
    assert not (tmp_path / "grid" / "best_by_config.csv").exists()


def test_run_grid_search_prefers_deployable_same_checkpoint_config_change_over_weaker_candidate(tmp_path: Path, monkeypatch) -> None:
    launch = tmp_path / "launch.sh"
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    candidate_checkpoint = tmp_path / "candidate.pt"
    candidate_checkpoint.write_text("y")
    _write_launch(launch, current_checkpoint)

    def _fake_run(cmd, cwd, capture_output, text, check):
        assert check is False
        output_dir = Path(cmd[cmd.index("--output-dir") + 1])
        symbols = cmd[cmd.index("--symbols") + 1].split(",")
        leverage = float(cmd[cmd.index("--leverage") + 1])
        if symbols == ["BTCUSD", "ETHUSD"]:
            _write_manifest(
                output_dir / "prod_launch_eval_manifest.json",
                current_checkpoint,
                candidate_checkpoint,
                symbols=symbols,
                leverage=leverage,
                candidate_goodness=0.8,
                launch_goodness=1.0,
            )
        else:
            _write_manifest(
                output_dir / "prod_launch_eval_manifest.json",
                current_checkpoint,
                candidate_checkpoint,
                symbols=symbols,
                leverage=leverage,
                candidate_goodness=0.4,
                launch_goodness=-0.5,
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(grid.subprocess, "run", _fake_run)

    args = grid.parse_args(
        [
            "--launch-script",
            str(launch),
            "--output-dir",
            str(tmp_path / "grid"),
            "--symbols-set",
            "BTCUSD ETHUSD",
            "--symbols-set",
            "BTCUSD ETHUSD SOLUSD",
            "--candidate-checkpoint",
            str(candidate_checkpoint),
        ]
    )

    assert grid.run_grid_search(args) == 0

    best_rows = list(csv.DictReader((tmp_path / "grid" / "best_by_config.csv").open()))
    assert best_rows[0]["symbols"] == "BTCUSD ETHUSD"
    assert best_rows[0]["checkpoint"] == str(current_checkpoint.resolve())
    assert best_rows[0]["is_launch_checkpoint"] == "True"
    assert best_rows[0]["gate_allowed"] == "True"
    assert best_rows[0]["gate_candidate_metric"] == "1.0"


def test_load_requested_manifest_rows_preserves_running_hybrid_baseline_for_gate_annotations(
    tmp_path: Path,
) -> None:
    launch = tmp_path / "launch.sh"
    launch_checkpoint = tmp_path / "launch.pt"
    launch_checkpoint.write_text("launch")
    running_checkpoint = tmp_path / "running.pt"
    running_checkpoint.write_text("running")
    candidate_checkpoint = tmp_path / "candidate.pt"
    candidate_checkpoint.write_text("candidate")
    _write_launch(launch, launch_checkpoint)

    output_root = tmp_path / "grid"
    variant = grid.build_config_variants(
        launch_script=launch,
        symbol_set_specs=["BTCUSD ETHUSD SOLUSD"],
        symbol_subset_sizes=[],
        leverage_options=[0.5],
        output_root=output_root,
    )[0]
    manifest_path = Path(variant.output_dir) / "prod_launch_eval_manifest.json"
    _write_running_hybrid_drift_manifest(
        manifest_path,
        launch_checkpoint,
        running_checkpoint,
        candidate_checkpoint,
        symbols=list(variant.symbols),
        leverage=variant.leverage,
        running_goodness=1.5,
        candidate_goodness=2.0,
    )

    rows = grid._load_requested_manifest_rows(
        manifest_path,
        variant=variant,
        launch_script=launch,
        launch_config=grid.parse_launch_script(launch, require_rl_checkpoint=False),
        candidate_checkpoints=[candidate_checkpoint],
    )

    requested_manifest = Path(rows[0].manifest_path)
    filtered_payload = json.loads(requested_manifest.read_text())
    filtered_checkpoints = {evaluation["checkpoint"] for evaluation in filtered_payload["evaluations"]}
    assert filtered_checkpoints == {
        str(launch_checkpoint.resolve()),
        str(running_checkpoint.resolve()),
        str(candidate_checkpoint.resolve()),
    }
    candidate_row = next(row for row in rows if row.checkpoint == str(candidate_checkpoint.resolve()))
    assert candidate_row.gate_allowed is True
    assert candidate_row.gate_current_target_label == "running_hybrid"
    assert candidate_row.gate_current_symbols == "BTCUSD ETHUSD SOLUSD"
    assert candidate_row.gate_current_leverage == pytest.approx(0.5)
    assert candidate_row.gate_current_metric == pytest.approx(1.5)
    assert candidate_row.gate_candidate_metric == pytest.approx(2.0)


def test_run_grid_search_reuses_compatible_manifest_without_subprocess(tmp_path: Path, monkeypatch, capsys) -> None:
    launch = tmp_path / "launch.sh"
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    candidate_checkpoint = tmp_path / "candidate.pt"
    candidate_checkpoint.write_text("y")
    _write_launch(launch, current_checkpoint)

    output_root = tmp_path / "grid"
    variant = grid.build_config_variants(
        launch_script=launch,
        symbol_set_specs=["BTCUSD ETHUSD"],
        symbol_subset_sizes=[],
        leverage_options=[2.0],
        output_root=output_root,
    )[0]
    _write_manifest(
        Path(variant.output_dir) / "prod_launch_eval_manifest.json",
        current_checkpoint,
        candidate_checkpoint,
        symbols=list(variant.symbols),
        leverage=variant.leverage,
        candidate_goodness=3.0,
    )

    def _unexpected_run(*_args, **_kwargs):
        raise AssertionError("subprocess.run should not be called when a reusable manifest exists")

    monkeypatch.setattr(grid.subprocess, "run", _unexpected_run)

    args = grid.parse_args(
        [
            "--launch-script",
            str(launch),
            "--output-dir",
            str(output_root),
            "--symbols-set",
            "BTCUSD ETHUSD",
            "--leverage-option",
            "2.0",
            "--candidate-checkpoint",
            str(candidate_checkpoint),
        ]
    )

    assert grid.run_grid_search(args) == 0
    captured = capsys.readouterr()
    assert "Reused manifests: full=1/1 partial=0/1" in captured.out


def test_run_grid_search_budget_accounts_for_reusable_manifests(tmp_path: Path, monkeypatch, capsys) -> None:
    launch = tmp_path / "launch.sh"
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    candidate_a = tmp_path / "candidate_a.pt"
    candidate_b = tmp_path / "candidate_b.pt"
    candidate_a.write_text("y")
    candidate_b.write_text("z")
    _write_launch(launch, current_checkpoint)

    output_root = tmp_path / "grid"
    variants = grid.build_config_variants(
        launch_script=launch,
        symbol_set_specs=["BTCUSD ETHUSD", "BTCUSD ETHUSD SOLUSD"],
        symbol_subset_sizes=[],
        leverage_options=[0.5],
        output_root=output_root,
    )
    for variant in variants:
        _write_manifest(
            Path(variant.output_dir) / "prod_launch_eval_manifest.json",
            current_checkpoint,
            candidate_a,
            symbols=list(variant.symbols),
            leverage=variant.leverage,
            candidate_goodness=1.0,
        )

    calls: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    def _fake_run(cmd, cwd, capture_output, text, check):
        assert check is False
        output_dir = Path(cmd[cmd.index("--output-dir") + 1])
        symbols = tuple(cmd[cmd.index("--symbols") + 1].split(","))
        checkpoint_args = tuple(Path(cmd[index + 1]).resolve().name for index, token in enumerate(cmd) if token == "--candidate-checkpoint")
        calls.append((symbols, checkpoint_args))
        assert checkpoint_args == (candidate_b.name,)
        _write_manifest(
            output_dir / "prod_launch_eval_manifest.json",
            current_checkpoint,
            candidate_b,
            symbols=list(symbols),
            leverage=0.5,
            candidate_goodness=2.0,
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(grid.subprocess, "run", _fake_run)

    args = grid.parse_args(
        [
            "--launch-script",
            str(launch),
            "--output-dir",
            str(output_root),
            "--symbols-set",
            "BTCUSD ETHUSD",
            "--symbols-set",
            "BTCUSD ETHUSD SOLUSD",
            "--candidate-checkpoint",
            str(candidate_a),
            "--candidate-checkpoint",
            str(candidate_b),
            "--max-checkpoint-config-evals",
            "4",
        ]
    )

    assert grid.run_grid_search(args) == 0
    assert len(calls) == 2
    captured = capsys.readouterr()
    assert "pruning to 1 variant(s)" not in captured.err
    best_rows = list(csv.DictReader((output_root / "best_by_config.csv").open()))
    assert len(best_rows) == 2


def test_run_grid_search_no_reuse_budget_ignores_cached_manifests(tmp_path: Path, monkeypatch, capsys) -> None:
    launch = tmp_path / "launch.sh"
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    candidate_checkpoint = tmp_path / "candidate.pt"
    candidate_checkpoint.write_text("y")
    _write_launch(launch, current_checkpoint)

    output_root = tmp_path / "grid"
    variants = grid.build_config_variants(
        launch_script=launch,
        symbol_set_specs=["BTCUSD ETHUSD", "ETHUSD SOLUSD"],
        symbol_subset_sizes=[],
        leverage_options=[2.0],
        output_root=output_root,
    )
    for variant in variants:
        _write_manifest(
            Path(variant.output_dir) / "prod_launch_eval_manifest.json",
            current_checkpoint,
            candidate_checkpoint,
            symbols=list(variant.symbols),
            leverage=variant.leverage,
            candidate_goodness=1.0,
        )

    call_order: list[tuple[str, ...]] = []

    def _fake_run(cmd, cwd, capture_output, text, check):
        assert check is False
        output_dir = Path(cmd[cmd.index("--output-dir") + 1])
        symbols = tuple(cmd[cmd.index("--symbols") + 1].split(","))
        leverage = float(cmd[cmd.index("--leverage") + 1])
        call_order.append(symbols)
        _write_manifest(
            output_dir / "prod_launch_eval_manifest.json",
            current_checkpoint,
            candidate_checkpoint,
            symbols=list(symbols),
            leverage=leverage,
            candidate_goodness=2.0,
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(grid.subprocess, "run", _fake_run)

    args = grid.parse_args(
        [
            "--launch-script",
            str(launch),
            "--output-dir",
            str(output_root),
            "--symbols-set",
            "BTCUSD ETHUSD",
            "--symbols-set",
            "ETHUSD SOLUSD",
            "--leverage-option",
            "2.0",
            "--candidate-checkpoint",
            str(candidate_checkpoint),
            "--max-checkpoint-config-evals",
            "2",
            "--no-reuse-manifests",
        ]
    )

    assert grid.run_grid_search(args) == 0
    captured = capsys.readouterr()
    assert "pruning to 1 variant(s)" in captured.err
    assert len(call_order) == 1


def test_run_grid_search_reuses_single_process_isolation_preflight_across_variants(tmp_path: Path, monkeypatch) -> None:
    launch = tmp_path / "launch.sh"
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    candidate_checkpoint = tmp_path / "candidate.pt"
    candidate_checkpoint.write_text("y")
    _write_launch(launch, current_checkpoint)

    output_root = tmp_path / "grid"
    audit_calls: list[str] = []
    build_calls: list[object] = []

    def _fake_audit(launch_script):
        audit_calls.append(str(Path(launch_script).resolve()))
        return object()

    def _fake_build(audit_result):
        build_calls.append(audit_result)

    def _fake_run(cmd, cwd, capture_output, text, check):
        assert check is False
        output_dir = Path(cmd[cmd.index("--output-dir") + 1])
        symbols = cmd[cmd.index("--symbols") + 1].split(",")
        leverage = float(cmd[cmd.index("--leverage") + 1])
        _write_manifest(
            output_dir / "prod_launch_eval_manifest.json",
            current_checkpoint,
            candidate_checkpoint,
            symbols=list(symbols),
            leverage=leverage,
            candidate_goodness=1.0,
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(grid, "audit_binance_hybrid_machine_state", _fake_audit)
    monkeypatch.setattr(grid, "build_machine_deploy_preflight_reason", _fake_build)
    monkeypatch.setattr(grid.subprocess, "run", _fake_run)

    args = grid.parse_args(
        [
            "--launch-script",
            str(launch),
            "--output-dir",
            str(output_root),
            "--symbols-set",
            "BTCUSD ETHUSD",
            "--symbols-set",
            "ETHUSD SOLUSD",
            "--leverage-option",
            "2.0",
            "--candidate-checkpoint",
            str(candidate_checkpoint),
            "--jobs",
            "1",
            "--max-checkpoint-config-evals",
            "0",
            "--no-reuse-manifests",
            "--require-process-isolation",
        ]
    )

    assert grid.run_grid_search(args) == 0
    assert audit_calls == [str(launch.resolve())]
    assert len(build_calls) == 1


def test_run_grid_search_executes_lower_pending_cost_variants_first(tmp_path: Path, monkeypatch) -> None:
    launch = tmp_path / "launch.sh"
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    candidate_checkpoint = tmp_path / "candidate.pt"
    candidate_checkpoint.write_text("y")
    _write_launch(launch, current_checkpoint)

    output_root = tmp_path / "grid"
    call_order: list[tuple[str, ...]] = []
    pending_costs = {
        ("BTCUSD", "ETHUSD", "SOLUSD"): 3,
        ("BTCUSD", "ETHUSD"): 1,
    }

    monkeypatch.setattr(
        grid,
        "order_config_variants_by_pending_eval_cost",
        lambda variants, **_kwargs: sorted(variants, key=lambda variant: pending_costs[variant.symbols]),
    )

    def _fake_run(cmd, cwd, capture_output, text, check):
        assert check is False
        output_dir = Path(cmd[cmd.index("--output-dir") + 1])
        symbols = tuple(cmd[cmd.index("--symbols") + 1].split(","))
        leverage = float(cmd[cmd.index("--leverage") + 1])
        call_order.append(symbols)
        _write_manifest(
            output_dir / "prod_launch_eval_manifest.json",
            current_checkpoint,
            candidate_checkpoint,
            symbols=list(symbols),
            leverage=leverage,
            candidate_goodness=1.0,
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(grid.subprocess, "run", _fake_run)

    args = grid.parse_args(
        [
            "--launch-script",
            str(launch),
            "--output-dir",
            str(output_root),
            "--symbols-set",
            "BTCUSD ETHUSD SOLUSD",
            "--symbols-set",
            "BTCUSD ETHUSD",
            "--leverage-option",
            "2.0",
            "--candidate-checkpoint",
            str(candidate_checkpoint),
            "--jobs",
            "1",
            "--max-checkpoint-config-evals",
            "0",
            "--no-reuse-manifests",
        ]
    )

    assert grid.run_grid_search(args) == 0
    assert call_order == [
        ("BTCUSD", "ETHUSD"),
        ("BTCUSD", "ETHUSD", "SOLUSD"),
    ]



def test_run_grid_search_executes_launch_variant_first_when_pending(tmp_path: Path, monkeypatch) -> None:
    launch = tmp_path / "launch.sh"
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    candidate_checkpoint = tmp_path / "candidate.pt"
    candidate_checkpoint.write_text("y")
    _write_launch(launch, current_checkpoint)

    output_root = tmp_path / "grid"
    call_order: list[tuple[str, ...]] = []
    pending_costs = {
        ("BTCUSD", "ETHUSD"): 1,
        ("BTCUSD", "ETHUSD", "SOLUSD"): 3,
    }

    def _order_variants(variants, **_kwargs):
        return sorted(
            variants,
            key=lambda variant: (
                0 if variant.symbols == ("BTCUSD", "ETHUSD", "SOLUSD") else 1,
                pending_costs[variant.symbols],
            ),
        )

    monkeypatch.setattr(grid, "order_config_variants_by_pending_eval_cost", _order_variants)

    def _fake_run(cmd, cwd, capture_output, text, check):
        assert check is False
        output_dir = Path(cmd[cmd.index("--output-dir") + 1])
        symbols = tuple(cmd[cmd.index("--symbols") + 1].split(","))
        leverage = float(cmd[cmd.index("--leverage") + 1])
        call_order.append(symbols)
        _write_manifest(
            output_dir / "prod_launch_eval_manifest.json",
            current_checkpoint,
            candidate_checkpoint,
            symbols=list(symbols),
            leverage=leverage,
            candidate_goodness=1.0,
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(grid.subprocess, "run", _fake_run)

    args = grid.parse_args(
        [
            "--launch-script",
            str(launch),
            "--output-dir",
            str(output_root),
            "--symbols-set",
            "BTCUSD ETHUSD",
            "--candidate-checkpoint",
            str(candidate_checkpoint),
            "--jobs",
            "1",
            "--max-checkpoint-config-evals",
            "0",
            "--include-launch-checkpoint",
            "--no-reuse-manifests",
        ]
    )

    assert grid.run_grid_search(args) == 0
    assert call_order[0] == ("BTCUSD", "ETHUSD", "SOLUSD")


def test_run_grid_search_reuse_ignores_extra_manifest_candidates(tmp_path: Path, monkeypatch) -> None:
    launch = tmp_path / "launch.sh"
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    requested_candidate = tmp_path / "candidate_requested.pt"
    extra_candidate = tmp_path / "candidate_extra.pt"
    requested_candidate.write_text("y")
    extra_candidate.write_text("z")
    _write_launch(launch, current_checkpoint)

    output_root = tmp_path / "grid"
    variant = grid.build_config_variants(
        launch_script=launch,
        symbol_set_specs=["BTCUSD ETHUSD"],
        symbol_subset_sizes=[],
        leverage_options=[2.0],
        output_root=output_root,
    )[0]
    manifest_path = Path(variant.output_dir) / "prod_launch_eval_manifest.json"
    _write_manifest(
        manifest_path,
        current_checkpoint,
        requested_candidate,
        symbols=list(variant.symbols),
        leverage=variant.leverage,
        candidate_goodness=1.0,
    )
    payload = json.loads(manifest_path.read_text())
    payload["evaluations"].append(
        {
            "checkpoint": str(extra_candidate.resolve()),
            "median_total_return": 0.9,
            "median_sortino": 2.5,
            "replay": {
                "hourly_goodness_score": 9.0,
                "hourly_total_return": 0.9,
                "hourly_sortino": 2.5,
            },
        }
    )
    manifest_path.write_text(json.dumps(payload))

    def _unexpected_run(*_args, **_kwargs):
        raise AssertionError("subprocess.run should not be called when a reusable manifest exists")

    monkeypatch.setattr(grid.subprocess, "run", _unexpected_run)

    args = grid.parse_args(
        [
            "--launch-script",
            str(launch),
            "--output-dir",
            str(output_root),
            "--symbols-set",
            "BTCUSD ETHUSD",
            "--leverage-option",
            "2.0",
            "--candidate-checkpoint",
            str(requested_candidate),
        ]
    )

    assert grid.run_grid_search(args) == 0
    best_rows = list(csv.DictReader((output_root / "best_by_config.csv").open()))
    assert best_rows[0]["checkpoint"] == str(requested_candidate.resolve())
    filtered_manifest_path = Path(best_rows[0]["manifest_path"])
    assert filtered_manifest_path == grid._requested_manifest_path(
        variant,
        launch_config=grid.parse_launch_script(launch, require_rl_checkpoint=False),
        candidate_checkpoints=[requested_candidate],
    )
    filtered_payload = json.loads(filtered_manifest_path.read_text())
    filtered_checkpoints = {
        evaluation["checkpoint"]
        for evaluation in filtered_payload["evaluations"]
    }
    assert filtered_checkpoints == {
        str(current_checkpoint.resolve()),
        str(requested_candidate.resolve()),
    }
    all_rows = list(csv.DictReader((output_root / "all_results.csv").open()))
    checkpoints = {row["checkpoint"] for row in all_rows}
    assert str(extra_candidate.resolve()) not in checkpoints


def test_run_grid_search_requested_manifest_path_is_stable_per_candidate_set(tmp_path: Path, monkeypatch) -> None:
    launch = tmp_path / "launch.sh"
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    candidate_a = tmp_path / "candidate_a.pt"
    candidate_b = tmp_path / "candidate_b.pt"
    candidate_a.write_text("a")
    candidate_b.write_text("b")
    _write_launch(launch, current_checkpoint)

    output_root = tmp_path / "grid"

    def _fake_run(cmd, cwd, capture_output, text, check):
        output_dir = Path(cmd[cmd.index("--output-dir") + 1])
        symbols = cmd[cmd.index("--symbols") + 1].split(",")
        leverage = float(cmd[cmd.index("--leverage") + 1])
        candidate_flags = [cmd[index + 1] for index, token in enumerate(cmd) if token == "--candidate-checkpoint"]
        candidate_path = Path(candidate_flags[0]) if candidate_flags else None
        candidate_goodness = 2.0 if candidate_path == candidate_a.resolve() else 3.0
        _write_manifest(
            output_dir / "prod_launch_eval_manifest.json",
            current_checkpoint,
            candidate_path,
            symbols=symbols,
            leverage=leverage,
            candidate_goodness=candidate_goodness,
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(grid.subprocess, "run", _fake_run)

    args_a = grid.parse_args(
        [
            "--launch-script",
            str(launch),
            "--output-dir",
            str(output_root),
            "--symbols-set",
            "BTCUSD ETHUSD",
            "--leverage-option",
            "2.0",
            "--candidate-checkpoint",
            str(candidate_a),
            "--no-reuse-manifests",
        ]
    )
    assert grid.run_grid_search(args_a) == 0
    best_rows_a = list(csv.DictReader((output_root / "best_by_config.csv").open()))
    manifest_a = Path(best_rows_a[0]["manifest_path"])

    args_b = grid.parse_args(
        [
            "--launch-script",
            str(launch),
            "--output-dir",
            str(output_root),
            "--symbols-set",
            "BTCUSD ETHUSD",
            "--leverage-option",
            "2.0",
            "--candidate-checkpoint",
            str(candidate_b),
            "--no-reuse-manifests",
        ]
    )
    assert grid.run_grid_search(args_b) == 0
    best_rows_b = list(csv.DictReader((output_root / "best_by_config.csv").open()))
    manifest_b = Path(best_rows_b[0]["manifest_path"])

    assert manifest_a != manifest_b
    payload_a = json.loads(manifest_a.read_text())
    payload_b = json.loads(manifest_b.read_text())
    checkpoints_a = {evaluation["checkpoint"] for evaluation in payload_a["evaluations"]}
    checkpoints_b = {evaluation["checkpoint"] for evaluation in payload_b["evaluations"]}
    assert checkpoints_a == {str(current_checkpoint.resolve()), str(candidate_a.resolve())}
    assert checkpoints_b == {str(current_checkpoint.resolve()), str(candidate_b.resolve())}


def test_run_grid_search_no_reuse_manifests_forces_subprocess(tmp_path: Path, monkeypatch) -> None:
    launch = tmp_path / "launch.sh"
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    candidate_checkpoint = tmp_path / "candidate.pt"
    candidate_checkpoint.write_text("y")
    _write_launch(launch, current_checkpoint)

    output_root = tmp_path / "grid"
    variant = grid.build_config_variants(
        launch_script=launch,
        symbol_set_specs=["BTCUSD ETHUSD"],
        symbol_subset_sizes=[],
        leverage_options=[2.0],
        output_root=output_root,
    )[0]
    _write_manifest(
        Path(variant.output_dir) / "prod_launch_eval_manifest.json",
        current_checkpoint,
        candidate_checkpoint,
        symbols=list(variant.symbols),
        leverage=variant.leverage,
        candidate_goodness=3.0,
    )

    run_calls: list[list[str]] = []

    def _fake_run(cmd, cwd, capture_output, text, check):
        run_calls.append(cmd)
        _write_manifest(
            Path(variant.output_dir) / "prod_launch_eval_manifest.json",
            current_checkpoint,
            candidate_checkpoint,
            symbols=list(variant.symbols),
            leverage=variant.leverage,
            candidate_goodness=3.5,
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(grid.subprocess, "run", _fake_run)

    args = grid.parse_args(
        [
            "--launch-script",
            str(launch),
            "--output-dir",
            str(output_root),
            "--symbols-set",
            "BTCUSD ETHUSD",
            "--leverage-option",
            "2.0",
            "--candidate-checkpoint",
            str(candidate_checkpoint),
            "--no-reuse-manifests",
        ]
    )

    assert grid.run_grid_search(args) == 0
    assert len(run_calls) == 1


def test_load_reusable_manifest_plan_skips_row_loading_for_partial_reuse(tmp_path: Path, monkeypatch) -> None:
    launch = tmp_path / "launch.sh"
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    candidate_a = tmp_path / "candidate_a.pt"
    candidate_b = tmp_path / "candidate_b.pt"
    candidate_a.write_text("y")
    candidate_b.write_text("z")
    _write_launch(launch, current_checkpoint)

    output_root = tmp_path / "grid"
    variant = grid.build_config_variants(
        launch_script=launch,
        symbol_set_specs=["BTCUSD ETHUSD"],
        symbol_subset_sizes=[],
        leverage_options=[2.0],
        output_root=output_root,
    )[0]
    manifest_path = Path(variant.output_dir) / "prod_launch_eval_manifest.json"
    _write_manifest(
        manifest_path,
        current_checkpoint,
        candidate_a,
        symbols=list(variant.symbols),
        leverage=variant.leverage,
        candidate_goodness=2.0,
    )

    monkeypatch.setattr(grid, "manifest_matches_deploy_config", lambda *args, **kwargs: (True, None))

    def _unexpected(*args, **kwargs):
        raise AssertionError("partial-reuse plan should not load rows before merge")

    monkeypatch.setattr(grid, "_load_requested_manifest_rows", _unexpected)

    launch_config = grid.parse_launch_script(launch, require_rl_checkpoint=False)
    plan = grid._load_reusable_manifest_plan(
        variant,
        launch_script=launch,
        launch_config=launch_config,
        candidate_checkpoints=[candidate_a, candidate_b],
        max_manifest_age_hours=24.0,
    )

    assert plan is not None
    assert plan.manifest_path == manifest_path
    assert plan.rows == []
    assert plan.missing_requested_checkpoints == (str(candidate_b.resolve()),)



def test_load_reusable_manifest_plan_loads_rows_when_manifest_is_complete(tmp_path: Path, monkeypatch) -> None:
    launch = tmp_path / "launch.sh"
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    candidate_checkpoint = tmp_path / "candidate.pt"
    candidate_checkpoint.write_text("y")
    _write_launch(launch, current_checkpoint)

    output_root = tmp_path / "grid"
    variant = grid.build_config_variants(
        launch_script=launch,
        symbol_set_specs=["BTCUSD ETHUSD"],
        symbol_subset_sizes=[],
        leverage_options=[2.0],
        output_root=output_root,
    )[0]
    manifest_path = Path(variant.output_dir) / "prod_launch_eval_manifest.json"
    _write_manifest(
        manifest_path,
        current_checkpoint,
        candidate_checkpoint,
        symbols=list(variant.symbols),
        leverage=variant.leverage,
        candidate_goodness=2.0,
    )

    monkeypatch.setattr(grid, "manifest_matches_deploy_config", lambda *args, **kwargs: (True, None))
    fake_row = grid.ConfigSearchRow(
        config_slug=variant.slug,
        symbols=" ".join(variant.symbols),
        leverage=variant.leverage,
        checkpoint=str(candidate_checkpoint.resolve()),
        is_launch_checkpoint=False,
        metric_name="replay.hourly_goodness_score",
        metric_value=2.0,
        median_total_return=0.02,
        median_sortino=0.7,
        replay_hourly_goodness_score=2.0,
        manifest_path=str(manifest_path.resolve()),
    )
    seen: list[tuple[list[str | Path], bool]] = []

    def _fake_load_requested(*args, candidate_checkpoints, require_process_isolation=False, **kwargs):
        seen.append((list(candidate_checkpoints), require_process_isolation))
        return [fake_row]

    monkeypatch.setattr(grid, "_load_requested_manifest_rows", _fake_load_requested)

    launch_config = grid.parse_launch_script(launch, require_rl_checkpoint=False)
    plan = grid._load_reusable_manifest_plan(
        variant,
        launch_script=launch,
        launch_config=launch_config,
        candidate_checkpoints=[candidate_checkpoint],
        max_manifest_age_hours=24.0,
        require_process_isolation=True,
    )

    assert plan is not None
    assert plan.rows == [fake_row]
    assert plan.missing_requested_checkpoints == ()
    assert seen == [([candidate_checkpoint], True)]



def test_run_grid_search_fails_when_fresh_variant_manifest_omits_requested_candidate(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    launch = tmp_path / "launch.sh"
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    candidate_checkpoint = tmp_path / "candidate.pt"
    candidate_checkpoint.write_text("y")
    _write_launch(launch, current_checkpoint)

    output_root = tmp_path / "grid"
    variant = grid.build_config_variants(
        launch_script=launch,
        symbol_set_specs=["BTCUSD ETHUSD"],
        symbol_subset_sizes=[],
        leverage_options=[2.0],
        output_root=output_root,
    )[0]

    def _fake_run(cmd, cwd, capture_output, text, check):
        _write_manifest(
            Path(variant.output_dir) / "prod_launch_eval_manifest.json",
            current_checkpoint,
            None,
            symbols=list(variant.symbols),
            leverage=variant.leverage,
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(grid.subprocess, "run", _fake_run)

    args = grid.parse_args(
        [
            "--launch-script",
            str(launch),
            "--output-dir",
            str(output_root),
            "--symbols-set",
            "BTCUSD ETHUSD",
            "--leverage-option",
            "2.0",
            "--candidate-checkpoint",
            str(candidate_checkpoint),
            "--no-reuse-manifests",
        ]
    )

    assert grid.run_grid_search(args) == 2
    captured = capsys.readouterr()
    assert "requested checkpoint evaluations missing from manifest" in captured.err
    assert str(candidate_checkpoint.resolve()) in captured.err


def test_run_grid_search_incrementally_reuses_manifest_when_candidate_checkpoint_is_missing(tmp_path: Path, monkeypatch) -> None:
    launch = tmp_path / "launch.sh"
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    candidate_a = tmp_path / "candidate_a.pt"
    candidate_b = tmp_path / "candidate_b.pt"
    candidate_a.write_text("y")
    candidate_b.write_text("z")
    _write_launch(launch, current_checkpoint)

    output_root = tmp_path / "grid"
    variant = grid.build_config_variants(
        launch_script=launch,
        symbol_set_specs=["BTCUSD ETHUSD"],
        symbol_subset_sizes=[],
        leverage_options=[2.0],
        output_root=output_root,
    )[0]
    _write_manifest(
        Path(variant.output_dir) / "prod_launch_eval_manifest.json",
        current_checkpoint,
        candidate_a,
        symbols=list(variant.symbols),
        leverage=variant.leverage,
        candidate_goodness=2.0,
    )

    run_calls: list[list[str]] = []

    def _fake_run(cmd, cwd, capture_output, text, check):
        run_calls.append(cmd)
        _write_manifest(
            Path(variant.output_dir) / "prod_launch_eval_manifest.json",
            current_checkpoint,
            candidate_b,
            symbols=list(variant.symbols),
            leverage=variant.leverage,
            candidate_goodness=3.0,
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(grid.subprocess, "run", _fake_run)

    args = grid.parse_args(
        [
            "--launch-script",
            str(launch),
            "--output-dir",
            str(output_root),
            "--symbols-set",
            "BTCUSD ETHUSD",
            "--leverage-option",
            "2.0",
            "--candidate-checkpoint",
            str(candidate_a),
            "--candidate-checkpoint",
            str(candidate_b),
        ]
    )

    assert grid.run_grid_search(args) == 0
    assert len(run_calls) == 1
    called_cmd = run_calls[0]
    assert called_cmd.count("--candidate-checkpoint") == 1
    assert str(candidate_b.resolve()) in called_cmd
    assert str(candidate_a.resolve()) not in called_cmd

    all_rows = list(csv.DictReader((output_root / "all_results.csv").open()))
    checkpoints = {row["checkpoint"] for row in all_rows}
    assert checkpoints == {
        str(current_checkpoint.resolve()),
        str(candidate_a.resolve()),
        str(candidate_b.resolve()),
    }


def test_run_grid_search_fails_when_incremental_reuse_still_misses_requested_candidate(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    launch = tmp_path / "launch.sh"
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    candidate_a = tmp_path / "candidate_a.pt"
    candidate_b = tmp_path / "candidate_b.pt"
    candidate_a.write_text("y")
    candidate_b.write_text("z")
    _write_launch(launch, current_checkpoint)

    output_root = tmp_path / "grid"
    variant = grid.build_config_variants(
        launch_script=launch,
        symbol_set_specs=["BTCUSD ETHUSD"],
        symbol_subset_sizes=[],
        leverage_options=[2.0],
        output_root=output_root,
    )[0]
    _write_manifest(
        Path(variant.output_dir) / "prod_launch_eval_manifest.json",
        current_checkpoint,
        candidate_a,
        symbols=list(variant.symbols),
        leverage=variant.leverage,
        candidate_goodness=2.0,
    )

    run_calls: list[list[str]] = []

    def _fake_run(cmd, cwd, capture_output, text, check):
        run_calls.append(cmd)
        _write_manifest(
            Path(variant.output_dir) / "prod_launch_eval_manifest.json",
            current_checkpoint,
            None,
            symbols=list(variant.symbols),
            leverage=variant.leverage,
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(grid.subprocess, "run", _fake_run)

    args = grid.parse_args(
        [
            "--launch-script",
            str(launch),
            "--output-dir",
            str(output_root),
            "--symbols-set",
            "BTCUSD ETHUSD",
            "--leverage-option",
            "2.0",
            "--candidate-checkpoint",
            str(candidate_a),
            "--candidate-checkpoint",
            str(candidate_b),
        ]
    )

    assert grid.run_grid_search(args) == 2
    assert len(run_calls) == 1
    captured = capsys.readouterr()
    assert "requested checkpoint evaluations missing from manifest" in captured.err
    assert str(candidate_b.resolve()) in captured.err


def test_run_grid_search_parallel_fails_when_incremental_reuse_still_misses_requested_candidate(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    launch = tmp_path / "launch.sh"
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    candidate_a = tmp_path / "candidate_a.pt"
    candidate_b = tmp_path / "candidate_b.pt"
    candidate_a.write_text("y")
    candidate_b.write_text("z")
    _write_launch(launch, current_checkpoint)

    output_root = tmp_path / "grid"
    variant = grid.build_config_variants(
        launch_script=launch,
        symbol_set_specs=["BTCUSD ETHUSD"],
        symbol_subset_sizes=[],
        leverage_options=[2.0],
        output_root=output_root,
    )[0]
    _write_manifest(
        Path(variant.output_dir) / "prod_launch_eval_manifest.json",
        current_checkpoint,
        candidate_a,
        symbols=list(variant.symbols),
        leverage=variant.leverage,
        candidate_goodness=2.0,
    )

    class FakeExecutor:
        def __init__(self, *, max_workers: int):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            future = concurrent.futures.Future()
            future.set_result(fn(*args, **kwargs))
            return future

    run_calls: list[list[str]] = []

    def _fake_run(cmd, cwd, capture_output, text, check):
        run_calls.append(cmd)
        _write_manifest(
            Path(variant.output_dir) / "prod_launch_eval_manifest.json",
            current_checkpoint,
            None,
            symbols=list(variant.symbols),
            leverage=variant.leverage,
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(grid.subprocess, "run", _fake_run)
    monkeypatch.setattr(grid.concurrent.futures, "ThreadPoolExecutor", FakeExecutor)

    args = grid.parse_args(
        [
            "--launch-script",
            str(launch),
            "--output-dir",
            str(output_root),
            "--symbols-set",
            "BTCUSD ETHUSD",
            "--leverage-option",
            "2.0",
            "--candidate-checkpoint",
            str(candidate_a),
            "--candidate-checkpoint",
            str(candidate_b),
            "--jobs",
            "2",
        ]
    )

    assert grid.run_grid_search(args) == 2
    assert len(run_calls) == 1
    captured = capsys.readouterr()
    assert "requested checkpoint evaluations missing from manifest" in captured.err
    assert str(candidate_b.resolve()) in captured.err


def test_run_grid_search_reruns_when_manifest_is_stale(tmp_path: Path, monkeypatch) -> None:
    launch = tmp_path / "launch.sh"
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    candidate_checkpoint = tmp_path / "candidate.pt"
    candidate_checkpoint.write_text("y")
    _write_launch(launch, current_checkpoint)

    output_root = tmp_path / "grid"
    variant = grid.build_config_variants(
        launch_script=launch,
        symbol_set_specs=["BTCUSD ETHUSD"],
        symbol_subset_sizes=[],
        leverage_options=[2.0],
        output_root=output_root,
    )[0]
    _write_manifest(
        Path(variant.output_dir) / "prod_launch_eval_manifest.json",
        current_checkpoint,
        candidate_checkpoint,
        symbols=list(variant.symbols),
        leverage=variant.leverage,
        candidate_goodness=3.0,
        generated_at="2020-01-01T00:00:00+00:00",
    )

    run_calls: list[list[str]] = []

    def _fake_run(cmd, cwd, capture_output, text, check):
        run_calls.append(cmd)
        _write_manifest(
            Path(variant.output_dir) / "prod_launch_eval_manifest.json",
            current_checkpoint,
            candidate_checkpoint,
            symbols=list(variant.symbols),
            leverage=variant.leverage,
            candidate_goodness=3.2,
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(grid.subprocess, "run", _fake_run)

    args = grid.parse_args(
        [
            "--launch-script",
            str(launch),
            "--output-dir",
            str(output_root),
            "--symbols-set",
            "BTCUSD ETHUSD",
            "--leverage-option",
            "2.0",
            "--candidate-checkpoint",
            str(candidate_checkpoint),
            "--manifest-max-age-hours",
            "1",
        ]
    )

    assert grid.run_grid_search(args) == 0
    assert len(run_calls) == 1


def test_run_grid_search_parallel_jobs(tmp_path: Path, monkeypatch) -> None:
    launch = tmp_path / "launch.sh"
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    candidate_checkpoint = tmp_path / "candidate.pt"
    candidate_checkpoint.write_text("y")
    _write_launch(launch, current_checkpoint)

    seen_workers: list[int] = []

    class FakeExecutor:
        def __init__(self, *, max_workers: int):
            seen_workers.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            future = concurrent.futures.Future()
            future.set_result(fn(*args, **kwargs))
            return future

    def _fake_run(cmd, cwd, capture_output, text, check):
        assert check is False
        output_dir = Path(cmd[cmd.index("--output-dir") + 1])
        symbols = cmd[cmd.index("--symbols") + 1].split(",")
        leverage = float(cmd[cmd.index("--leverage") + 1])
        _write_manifest(
            output_dir / "prod_launch_eval_manifest.json",
            current_checkpoint,
            candidate_checkpoint,
            symbols=symbols,
            leverage=leverage,
            candidate_goodness=1.0 + leverage,
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(grid.subprocess, "run", _fake_run)
    monkeypatch.setattr(grid.concurrent.futures, "ThreadPoolExecutor", FakeExecutor)

    args = grid.parse_args(
        [
            "--launch-script",
            str(launch),
            "--output-dir",
            str(tmp_path / "grid"),
            "--symbols-set",
            "BTCUSD ETHUSD",
            "--symbols-set",
            "BTCUSD ETHUSD SOLUSD",
            "--leverage-option",
            "0.5",
            "--leverage-option",
            "2.0",
            "--candidate-checkpoint",
            str(candidate_checkpoint),
            "--jobs",
            "3",
        ]
    )

    assert grid.run_grid_search(args) == 0
    assert seen_workers == [3]


def test_run_grid_search_rejects_jobs_below_one(tmp_path: Path, capsys) -> None:
    launch = tmp_path / "launch.sh"
    checkpoint = tmp_path / "current.pt"
    checkpoint.write_text("x")
    _write_launch(launch, checkpoint)

    args = grid.parse_args(
        [
            "--launch-script",
            str(launch),
            "--jobs",
            "0",
        ]
    )

    assert grid.run_grid_search(args) == 2
    captured = capsys.readouterr()
    assert "--jobs must be at least 1" in captured.err


def test_run_grid_search_reuses_single_pending_eval_plan_for_budgeting_and_ordering(tmp_path: Path, monkeypatch) -> None:
    launch = tmp_path / "launch.sh"
    checkpoint = tmp_path / "current.pt"
    checkpoint.write_text("x")
    _write_launch(launch, checkpoint)

    pending_plan = object()
    observed_pending_plan_args: list[object] = []

    def _fake_build_plan(variants, **kwargs):
        assert len(variants) == 2
        return pending_plan

    def _fake_estimate(variants, **kwargs):
        observed_pending_plan_args.append(kwargs.get("pending_eval_plan"))
        return 2

    def _fake_limit(variants, **kwargs):
        observed_pending_plan_args.append(kwargs.get("pending_eval_plan"))
        return list(variants)

    def _fake_order(variants, **kwargs):
        observed_pending_plan_args.append(kwargs.get("pending_eval_plan"))
        return list(reversed(variants))

    monkeypatch.setattr(grid, "build_pending_checkpoint_eval_plan", _fake_build_plan)
    monkeypatch.setattr(grid, "estimate_pending_checkpoint_config_evals", _fake_estimate)
    monkeypatch.setattr(grid, "limit_config_variants_by_pending_eval_budget", _fake_limit)
    monkeypatch.setattr(grid, "order_config_variants_by_pending_eval_cost", _fake_order)

    args = grid.parse_args(
        [
            "--launch-script",
            str(launch),
            "--output-dir",
            str(tmp_path / "grid"),
            "--symbols-set",
            "BTCUSD ETHUSD",
            "--symbols-set",
            "BTCUSD ETHUSD SOLUSD",
            "--leverage-option",
            "0.5",
            "--max-checkpoint-config-evals",
            "4",
            "--dry-run",
        ]
    )

    assert grid.run_grid_search(args) == 0
    assert observed_pending_plan_args == [pending_plan, pending_plan, pending_plan, pending_plan]



def test_run_grid_search_dry_run_prints_commands(tmp_path: Path, capsys, monkeypatch) -> None:
    launch = tmp_path / "launch.sh"
    checkpoint = tmp_path / "current.pt"
    checkpoint.write_text("x")
    _write_launch(launch, checkpoint)

    def _order_variants(variants, **_kwargs):
        return sorted(variants, key=lambda variant: 0 if variant.symbols == ("BTCUSD", "ETHUSD") else 1)

    monkeypatch.setattr(grid, "order_config_variants_by_pending_eval_cost", _order_variants)

    args = grid.parse_args(
        [
            "--launch-script",
            str(launch),
            "--output-dir",
            str(tmp_path / "grid"),
            "--symbols-set",
            "BTCUSD ETHUSD SOLUSD",
            "--symbols-set",
            "BTCUSD ETHUSD",
            "--leverage-option",
            "2.0",
            "--dry-run",
        ]
    )

    assert grid.run_grid_search(args) == 0
    captured = capsys.readouterr()
    assert "DRY RUN -- no evaluations executed" in captured.out
    assert captured.out.index("[1/2] symbols=BTCUSD ETHUSD leverage=2") < captured.out.index("[2/2] symbols=BTCUSD ETHUSD SOLUSD leverage=2")
    assert "--symbols BTCUSD,ETHUSD --leverage 2.0" in captured.out


def test_run_grid_search_rejects_invalid_symbol_subset_size(tmp_path: Path, capsys) -> None:
    launch = tmp_path / "launch.sh"
    checkpoint = tmp_path / "current.pt"
    checkpoint.write_text("x")
    _write_launch(launch, checkpoint)

    args = grid.parse_args(
        [
            "--launch-script",
            str(launch),
            "--output-dir",
            str(tmp_path / "grid"),
            "--symbols-subset-size",
            "4",
        ]
    )

    rc = grid.run_grid_search(args)

    assert rc == 2
    assert "exceeds launch symbol count" in capsys.readouterr().err


def test_run_grid_search_prunes_excessive_checkpoint_config_eval_count(tmp_path: Path, capsys, monkeypatch) -> None:
    launch = tmp_path / "launch.sh"
    checkpoint = tmp_path / "current.pt"
    checkpoint.write_text("x")
    candidate_a = tmp_path / "candidate_a.pt"
    candidate_b = tmp_path / "candidate_b.pt"
    candidate_a.write_text("a")
    candidate_b.write_text("b")
    _write_launch(launch, checkpoint)

    calls: list[tuple[tuple[str, ...], float]] = []

    def _fake_run(cmd, cwd, capture_output, text, check):
        output_dir = Path(cmd[cmd.index("--output-dir") + 1])
        symbols = tuple(cmd[cmd.index("--symbols") + 1].split(","))
        leverage = float(cmd[cmd.index("--leverage") + 1])
        calls.append((symbols, leverage))
        manifest_path = output_dir / "prod_launch_eval_manifest.json"
        _write_manifest(
            manifest_path,
            checkpoint,
            candidate_a,
            symbols=list(symbols),
            leverage=leverage,
            candidate_goodness=1.0 + leverage,
        )
        payload = json.loads(manifest_path.read_text())
        payload["evaluations"].append(
            {
                "checkpoint": str(candidate_b.resolve()),
                "median_total_return": 0.03,
                "median_sortino": 0.8,
                "replay": {
                    "hourly_goodness_score": 1.5 + leverage,
                    "hourly_total_return": 0.06,
                    "hourly_sortino": 1.0,
                },
            }
        )
        manifest_path.write_text(json.dumps(payload))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(grid.subprocess, "run", _fake_run)

    args = grid.parse_args(
        [
            "--launch-script",
            str(launch),
            "--output-dir",
            str(tmp_path / "grid"),
            "--symbols-subset-size",
            "2",
            "--candidate-checkpoint",
            str(candidate_a),
            "--candidate-checkpoint",
            str(candidate_b),
            "--max-variants",
            "10",
            "--max-checkpoint-config-evals",
            "8",
        ]
    )

    rc = grid.run_grid_search(args)

    assert rc == 0
    assert len(calls) == 2
    assert "pruning to 2 variant(s)" in capsys.readouterr().err


def test_run_grid_search_prunes_excessive_variant_count(tmp_path: Path, capsys, monkeypatch) -> None:
    launch = tmp_path / "launch.sh"
    checkpoint = tmp_path / "current.pt"
    checkpoint.write_text("x")
    candidate = tmp_path / "candidate.pt"
    candidate.write_text("y")
    _write_launch(launch, checkpoint)

    calls: list[tuple[tuple[str, ...], float]] = []

    def _fake_run(cmd, cwd, capture_output, text, check):
        output_dir = Path(cmd[cmd.index("--output-dir") + 1])
        symbols = tuple(cmd[cmd.index("--symbols") + 1].split(","))
        leverage = float(cmd[cmd.index("--leverage") + 1])
        calls.append((symbols, leverage))
        _write_manifest(
            output_dir / "prod_launch_eval_manifest.json",
            checkpoint,
            candidate,
            symbols=list(symbols),
            leverage=leverage,
            candidate_goodness=1.0 + leverage,
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(grid.subprocess, "run", _fake_run)

    args = grid.parse_args(
        [
            "--launch-script",
            str(launch),
            "--output-dir",
            str(tmp_path / "grid"),
            "--symbols-subset-size",
            "2",
            "--leverage-option",
            "0.5",
            "--leverage-option",
            "2.0",
            "--max-variants",
            "5",
            "--candidate-checkpoint",
            str(candidate),
        ]
    )

    rc = grid.run_grid_search(args)

    assert rc == 0
    assert len(calls) == 5
    assert "pruning to 5 variant(s)" in capsys.readouterr().err


def test_run_grid_search_variant_offset_rotates_pruned_variants_and_keeps_launch_variant(tmp_path: Path, monkeypatch) -> None:
    launch = tmp_path / "launch.sh"
    checkpoint = tmp_path / "current.pt"
    checkpoint.write_text("x")
    candidate = tmp_path / "candidate.pt"
    candidate.write_text("y")
    _write_launch(launch, checkpoint)

    calls: list[tuple[tuple[str, ...], float]] = []

    def _fake_run(cmd, cwd, capture_output, text, check):
        output_dir = Path(cmd[cmd.index("--output-dir") + 1])
        symbols = tuple(cmd[cmd.index("--symbols") + 1].split(","))
        leverage = float(cmd[cmd.index("--leverage") + 1])
        calls.append((symbols, leverage))
        _write_manifest(
            output_dir / "prod_launch_eval_manifest.json",
            checkpoint,
            candidate,
            symbols=list(symbols),
            leverage=leverage,
            candidate_goodness=1.0 + leverage,
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(grid.subprocess, "run", _fake_run)

    args = grid.parse_args(
        [
            "--launch-script",
            str(launch),
            "--output-dir",
            str(tmp_path / "grid"),
            "--symbols-subset-size",
            "2",
            "--candidate-checkpoint",
            str(candidate),
            "--include-launch-checkpoint",
            "--max-variants",
            "2",
            "--variant-offset",
            "1",
        ]
    )

    assert grid.run_grid_search(args) == 0
    assert calls == [
        (("BTCUSD", "ETHUSD", "SOLUSD"), 0.5),
        (("BTCUSD", "SOLUSD"), 0.5),
    ]


def test_run_grid_search_rejects_negative_variant_offset(tmp_path: Path, capsys) -> None:
    launch = tmp_path / "launch.sh"
    checkpoint = tmp_path / "current.pt"
    checkpoint.write_text("x")
    _write_launch(launch, checkpoint)

    args = grid.parse_args(
        [
            "--launch-script",
            str(launch),
            "--output-dir",
            str(tmp_path / "grid"),
            "--variant-offset",
            "-1",
        ]
    )

    rc = grid.run_grid_search(args)

    assert rc == 2
    assert "--variant-offset must be >= 0" in capsys.readouterr().err


def test_run_grid_search_rejects_negative_max_variants(tmp_path: Path, capsys) -> None:
    launch = tmp_path / "launch.sh"
    checkpoint = tmp_path / "current.pt"
    checkpoint.write_text("x")
    _write_launch(launch, checkpoint)

    args = grid.parse_args(
        [
            "--launch-script",
            str(launch),
            "--output-dir",
            str(tmp_path / "grid"),
            "--max-variants",
            "-1",
        ]
    )

    rc = grid.run_grid_search(args)

    assert rc == 2
    assert "--max-variants must be >= 0" in capsys.readouterr().err


def test_run_grid_search_max_checkpoint_config_evals_zero_disables_cap(tmp_path: Path, monkeypatch) -> None:
    launch = tmp_path / "launch.sh"
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    candidate_checkpoint = tmp_path / "candidate.pt"
    candidate_checkpoint.write_text("y")
    _write_launch(launch, current_checkpoint)

    def _fake_run(cmd, cwd, capture_output, text, check):
        output_dir = Path(cmd[cmd.index("--output-dir") + 1])
        symbols = cmd[cmd.index("--symbols") + 1].split(",")
        leverage = float(cmd[cmd.index("--leverage") + 1])
        _write_manifest(
            output_dir / "prod_launch_eval_manifest.json",
            current_checkpoint,
            candidate_checkpoint,
            symbols=symbols,
            leverage=leverage,
            candidate_goodness=1.0 + leverage,
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(grid.subprocess, "run", _fake_run)

    args = grid.parse_args(
        [
            "--launch-script",
            str(launch),
            "--output-dir",
            str(tmp_path / "grid"),
            "--symbols-subset-size",
            "2",
            "--candidate-checkpoint",
            str(candidate_checkpoint),
            "--max-checkpoint-config-evals",
            "0",
        ]
    )

    assert grid.run_grid_search(args) == 0


def test_run_grid_search_max_variants_zero_disables_cap(tmp_path: Path, monkeypatch) -> None:
    launch = tmp_path / "launch.sh"
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    candidate_checkpoint = tmp_path / "candidate.pt"
    candidate_checkpoint.write_text("y")
    _write_launch(launch, current_checkpoint)

    def _fake_run(cmd, cwd, capture_output, text, check):
        output_dir = Path(cmd[cmd.index("--output-dir") + 1])
        symbols = cmd[cmd.index("--symbols") + 1].split(",")
        leverage = float(cmd[cmd.index("--leverage") + 1])
        _write_manifest(
            output_dir / "prod_launch_eval_manifest.json",
            current_checkpoint,
            candidate_checkpoint,
            symbols=symbols,
            leverage=leverage,
            candidate_goodness=1.0 + leverage,
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(grid.subprocess, "run", _fake_run)

    args = grid.parse_args(
        [
            "--launch-script",
            str(launch),
            "--output-dir",
            str(tmp_path / "grid"),
            "--symbols-subset-size",
            "2",
            "--leverage-option",
            "0.5",
            "--leverage-option",
            "2.0",
            "--max-variants",
            "0",
            "--candidate-checkpoint",
            str(candidate_checkpoint),
        ]
    )

    assert grid.run_grid_search(args) == 0
