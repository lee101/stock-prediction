from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from src.binance_hybrid_eval_defaults import build_expected_prod_eval_config


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "deploy_crypto_model.sh"


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


def _write_launch_without_rl(path: Path) -> None:
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
                '  "$@"',
            ]
        )
    )


def _write_manifest(
    path: Path,
    current_checkpoint: Path,
    candidate_checkpoint: Path,
    *,
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
                    "symbols": ["BTCUSD", "ETHUSD", "SOLUSD"],
                    "execution_mode": "margin",
                    "leverage": 0.5,
                    "interval": 3600,
                    "fallback_mode": "chronos2",
                    "rl_checkpoint": str(current_checkpoint.resolve()),
                },
                "eval_config": build_expected_prod_eval_config(),
                "evaluations": [
                    {
                        "checkpoint": str(current_checkpoint.resolve()),
                        "median_total_return": -0.02,
                        "median_sortino": -0.5,
                        "replay": {"hourly_goodness_score": -1.0},
                    },
                    {
                        "checkpoint": str(candidate_checkpoint.resolve()),
                        "median_total_return": 0.01,
                        "median_sortino": 0.2,
                        "replay": {"hourly_goodness_score": candidate_goodness},
                    },
                ]
            }
        )
    )


def _write_grid_best_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "config_slug",
        "symbols",
        "leverage",
        "checkpoint",
        "is_launch_checkpoint",
        "metric_name",
        "metric_value",
        "median_total_return",
        "median_sortino",
        "replay_hourly_goodness_score",
        "manifest_path",
        "gate_metric_name",
        "gate_current_metric",
        "gate_candidate_metric",
        "gate_allowed",
        "gate_reason",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(0o755)


def test_deploy_crypto_model_dry_run_blocks_stale_manifest(tmp_path: Path) -> None:
    launch = tmp_path / "launch.sh"
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    candidate_checkpoint = tmp_path / "candidate.pt"
    candidate_checkpoint.write_text("y")
    _write_launch(launch, current_checkpoint)
    manifest = tmp_path / "prod_launch_eval_manifest.json"
    _write_manifest(
        manifest,
        current_checkpoint,
        candidate_checkpoint,
        candidate_goodness=2.0,
        generated_at="2020-01-01T00:00:00+00:00",
    )

    env = os.environ.copy()
    env["LAUNCH"] = str(launch)
    env["LOG"] = str(tmp_path / "supervisor.log")
    env["SUPERVISOR_PROG"] = "test-binance-hybrid-spot"
    env["PYTHON_BIN"] = sys.executable

    result = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run", "--manifest-path", str(manifest), str(candidate_checkpoint)],
        capture_output=True,
        check=False,
        text=True,
        cwd=str(REPO),
        env=env,
    )

    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "Deployment blocked by production-eval gate." in combined
    assert "older than maximum allowed age" in combined


def test_deploy_crypto_model_dry_run_blocks_manifest_with_runtime_health_issues(tmp_path: Path) -> None:
    launch = tmp_path / "launch.sh"
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    candidate_checkpoint = tmp_path / "candidate.pt"
    candidate_checkpoint.write_text("y")
    _write_launch(launch, current_checkpoint)
    manifest = tmp_path / "prod_launch_eval_manifest.json"
    _write_manifest(manifest, current_checkpoint, candidate_checkpoint, candidate_goodness=2.0)
    payload = json.loads(manifest.read_text())
    payload["current_runtime_health_issues"] = ["recent live runtime has too many degraded status cycles (1 > 0)"]
    manifest.write_text(json.dumps(payload))

    env = os.environ.copy()
    env["LAUNCH"] = str(launch)
    env["LOG"] = str(tmp_path / "supervisor.log")
    env["SUPERVISOR_PROG"] = "test-binance-hybrid-spot"
    env["PYTHON_BIN"] = sys.executable

    result = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run", "--manifest-path", str(manifest), str(candidate_checkpoint)],
        capture_output=True,
        check=False,
        text=True,
        cwd=str(REPO),
        env=env,
    )

    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "Deployment blocked by production-eval gate." in combined
    assert "manifest baseline runtime health issues are present" in combined


def test_deploy_crypto_model_dry_run_blocks_worse_candidate(tmp_path: Path) -> None:
    launch = tmp_path / "launch.sh"
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    candidate_checkpoint = tmp_path / "candidate.pt"
    candidate_checkpoint.write_text("y")
    _write_launch(launch, current_checkpoint)
    manifest = tmp_path / "prod_launch_eval_manifest.json"
    _write_manifest(manifest, current_checkpoint, candidate_checkpoint, candidate_goodness=-2.0)

    env = os.environ.copy()
    env["LAUNCH"] = str(launch)
    env["LOG"] = str(tmp_path / "supervisor.log")
    env["SUPERVISOR_PROG"] = "test-binance-hybrid-spot"
    env["PYTHON_BIN"] = sys.executable

    result = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run", "--manifest-path", str(manifest), str(candidate_checkpoint)],
        capture_output=True,
        check=False,
        text=True,
        cwd=str(REPO),
        env=env,
    )

    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "Deployment blocked by production-eval gate." in combined
    assert "Decision: BLOCK" in combined


def test_deploy_crypto_model_dry_run_blocks_negative_candidate_when_live_is_gemini_only(tmp_path: Path) -> None:
    launch = tmp_path / "launch.sh"
    _write_launch_without_rl(launch)
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    candidate_checkpoint = tmp_path / "candidate.pt"
    candidate_checkpoint.write_text("y")
    manifest = tmp_path / "prod_launch_eval_manifest.json"
    _write_manifest(manifest, current_checkpoint, candidate_checkpoint, candidate_goodness=-0.25)

    env = os.environ.copy()
    env["LAUNCH"] = str(launch)
    env["LOG"] = str(tmp_path / "supervisor.log")
    env["SUPERVISOR_PROG"] = "test-binance-hybrid-spot"
    env["PYTHON_BIN"] = sys.executable

    result = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run", "--manifest-path", str(manifest), str(candidate_checkpoint)],
        capture_output=True,
        check=False,
        text=True,
        cwd=str(REPO),
        env=env,
    )

    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "Deployment blocked by production-eval gate." in combined
    assert "candidate does not clear minimum production-faithful metric required to re-enable RL" in combined
    assert "Current:" in combined
    assert "Decision: BLOCK" in combined


def test_deploy_crypto_model_dry_run_force_allows_override(tmp_path: Path) -> None:
    launch = tmp_path / "launch.sh"
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    candidate_checkpoint = tmp_path / "candidate.pt"
    candidate_checkpoint.write_text("y")
    _write_launch(launch, current_checkpoint)
    manifest = tmp_path / "prod_launch_eval_manifest.json"
    _write_manifest(manifest, current_checkpoint, candidate_checkpoint, candidate_goodness=-2.0)

    env = os.environ.copy()
    env["LAUNCH"] = str(launch)
    env["LOG"] = str(tmp_path / "supervisor.log")
    env["SUPERVISOR_PROG"] = "test-binance-hybrid-spot"
    env["PYTHON_BIN"] = sys.executable

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--dry-run",
            "--force",
            "--manifest-path",
            str(manifest),
            str(candidate_checkpoint),
        ],
        capture_output=True,
        check=False,
        text=True,
        cwd=str(REPO),
        env=env,
    )

    combined = result.stdout + result.stderr
    assert result.returncode == 0
    assert "WARNING: overriding production-eval gate with --force" in combined
    assert "[dry-run] Would update existing --rl-checkpoint" in combined


def test_deploy_crypto_model_dry_run_blocks_non_best_manifest_candidate(tmp_path: Path) -> None:
    launch = tmp_path / "launch.sh"
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    candidate_checkpoint = tmp_path / "candidate.pt"
    candidate_checkpoint.write_text("y")
    best_checkpoint = tmp_path / "best.pt"
    best_checkpoint.write_text("z")
    _write_launch(launch, current_checkpoint)
    manifest = tmp_path / "prod_launch_eval_manifest.json"
    _write_manifest(manifest, current_checkpoint, candidate_checkpoint, candidate_goodness=2.0)
    payload = json.loads(manifest.read_text())
    payload["evaluations"].append(
        {
            "checkpoint": str(best_checkpoint.resolve()),
            "median_total_return": 0.02,
            "median_sortino": 0.4,
            "replay": {"hourly_goodness_score": 3.0},
        }
    )
    manifest.write_text(json.dumps(payload))

    env = os.environ.copy()
    env["LAUNCH"] = str(launch)
    env["LOG"] = str(tmp_path / "supervisor.log")
    env["SUPERVISOR_PROG"] = "test-binance-hybrid-spot"
    env["PYTHON_BIN"] = sys.executable

    result = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run", "--manifest-path", str(manifest), str(candidate_checkpoint)],
        capture_output=True,
        check=False,
        text=True,
        cwd=str(REPO),
        env=env,
    )

    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "Deployment blocked by production-eval gate." in combined
    assert "candidate is not best production-evaluated checkpoint in manifest" in combined
    assert str(best_checkpoint.resolve()) in combined


def test_deploy_crypto_model_dry_run_manifest_best_blocks_stale_manifest(tmp_path: Path) -> None:
    launch = tmp_path / "launch.sh"
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    candidate_checkpoint = tmp_path / "candidate.pt"
    candidate_checkpoint.write_text("y")
    _write_launch(launch, current_checkpoint)
    manifest = tmp_path / "prod_launch_eval_manifest.json"
    _write_manifest(
        manifest,
        current_checkpoint,
        candidate_checkpoint,
        candidate_goodness=2.0,
        generated_at="2020-01-01T00:00:00+00:00",
    )

    env = os.environ.copy()
    env["LAUNCH"] = str(launch)
    env["LOG"] = str(tmp_path / "supervisor.log")
    env["SUPERVISOR_PROG"] = "test-binance-hybrid-spot"
    env["PYTHON_BIN"] = sys.executable

    result = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run", "--manifest-path", str(manifest), "--manifest-best"],
        capture_output=True,
        check=False,
        text=True,
        cwd=str(REPO),
        env=env,
    )

    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "ERROR: unable to resolve best checkpoint from production manifest" in combined
    assert "older than maximum allowed age" in combined


def test_deploy_crypto_model_dry_run_grid_best_uses_selected_config(tmp_path: Path) -> None:
    launch = tmp_path / "launch.sh"
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    candidate_checkpoint = tmp_path / "candidate.pt"
    candidate_checkpoint.write_text("y")
    _write_launch(launch, current_checkpoint)
    manifest = tmp_path / "grid" / "variant" / "prod_launch_eval_manifest.json"
    _write_manifest(manifest, current_checkpoint, candidate_checkpoint, candidate_goodness=2.5)
    payload = json.loads(manifest.read_text())
    payload["launch_config"]["symbols"] = ["BTCUSD", "ETHUSD"]
    payload["launch_config"]["leverage"] = 2.0
    manifest.write_text(json.dumps(payload))
    grid_rows = tmp_path / "grid" / "best_by_config.csv"
    _write_grid_best_rows(
        grid_rows,
        [
            {
                "config_slug": "allowed",
                "symbols": "BTCUSD ETHUSD",
                "leverage": 2.0,
                "checkpoint": str(candidate_checkpoint.resolve()),
                "is_launch_checkpoint": False,
                "metric_name": "replay.hourly_goodness_score",
                "metric_value": 2.5,
                "median_total_return": 0.03,
                "median_sortino": 0.7,
                "replay_hourly_goodness_score": 2.5,
                "manifest_path": str(manifest.resolve()),
                "gate_metric_name": "replay.hourly_goodness_score",
                "gate_current_metric": -1.0,
                "gate_candidate_metric": 2.5,
                "gate_allowed": True,
                "gate_reason": "candidate passes deploy gate",
            },
        ],
    )

    env = os.environ.copy()
    env["LAUNCH"] = str(launch)
    env["LOG"] = str(tmp_path / "supervisor.log")
    env["SUPERVISOR_PROG"] = "test-binance-hybrid-spot"
    env["PYTHON_BIN"] = sys.executable

    result = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run", "--grid-best", str(grid_rows)],
        capture_output=True,
        check=False,
        text=True,
        cwd=str(REPO),
        env=env,
    )

    combined = result.stdout + result.stderr
    assert result.returncode == 0
    assert f"Resolved grid-best checkpoint: {candidate_checkpoint.resolve()}" in combined
    assert "Resolved grid-best symbols: BTCUSD ETHUSD" in combined
    assert "Resolved grid-best leverage: 2" in combined
    assert f"Resolved grid-best manifest: {manifest.resolve()}" in combined
    assert f"[dry-run] Would update existing --rl-checkpoint to: {candidate_checkpoint.resolve()}" in combined
    assert "[dry-run] Would update --symbols to: BTCUSD ETHUSD" in combined
    assert "[dry-run] Would update --leverage to: 2" in combined


def test_deploy_crypto_model_grid_best_rejects_manual_symbol_override(tmp_path: Path) -> None:
    launch = tmp_path / "launch.sh"
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    _write_launch(launch, current_checkpoint)
    grid_rows = tmp_path / "grid" / "best_by_config.csv"
    _write_grid_best_rows(grid_rows, [])

    env = os.environ.copy()
    env["LAUNCH"] = str(launch)
    env["LOG"] = str(tmp_path / "supervisor.log")
    env["SUPERVISOR_PROG"] = "test-binance-hybrid-spot"
    env["PYTHON_BIN"] = sys.executable

    result = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run", "--grid-best", str(grid_rows), "--symbols", "BTCUSD ETHUSD"],
        capture_output=True,
        check=False,
        text=True,
        cwd=str(REPO),
        env=env,
    )

    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "ERROR: --grid-best cannot be combined with symbol overrides" in combined


def test_deploy_crypto_model_dry_run_manifest_best_uses_best_checkpoint(tmp_path: Path) -> None:
    launch = tmp_path / "launch.sh"
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    candidate_checkpoint = tmp_path / "candidate.pt"
    candidate_checkpoint.write_text("y")
    best_checkpoint = tmp_path / "best.pt"
    best_checkpoint.write_text("z")
    _write_launch(launch, current_checkpoint)
    manifest = tmp_path / "prod_launch_eval_manifest.json"
    _write_manifest(manifest, current_checkpoint, candidate_checkpoint, candidate_goodness=2.0)
    payload = json.loads(manifest.read_text())
    payload["evaluations"].append(
        {
            "checkpoint": str(best_checkpoint.resolve()),
            "median_total_return": 0.02,
            "median_sortino": 0.4,
            "replay": {"hourly_goodness_score": 3.0},
        }
    )
    manifest.write_text(json.dumps(payload))

    env = os.environ.copy()
    env["LAUNCH"] = str(launch)
    env["LOG"] = str(tmp_path / "supervisor.log")
    env["SUPERVISOR_PROG"] = "test-binance-hybrid-spot"
    env["PYTHON_BIN"] = sys.executable

    result = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run", "--manifest-path", str(manifest), "--manifest-best"],
        capture_output=True,
        check=False,
        text=True,
        cwd=str(REPO),
        env=env,
    )

    combined = result.stdout + result.stderr
    assert result.returncode == 0
    assert f"Resolved manifest-best checkpoint: {best_checkpoint.resolve()}" in combined
    assert f"[dry-run] Would update existing --rl-checkpoint to: {best_checkpoint.resolve()}" in combined


def test_deploy_crypto_model_dry_run_manifest_best_with_symbols_option(tmp_path: Path) -> None:
    launch = tmp_path / "launch.sh"
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    candidate_checkpoint = tmp_path / "candidate.pt"
    candidate_checkpoint.write_text("y")
    best_checkpoint = tmp_path / "best.pt"
    best_checkpoint.write_text("z")
    _write_launch(launch, current_checkpoint)
    manifest = tmp_path / "prod_launch_eval_manifest.json"
    _write_manifest(manifest, current_checkpoint, candidate_checkpoint, candidate_goodness=2.0)
    payload = json.loads(manifest.read_text())
    payload["launch_config"]["symbols"] = ["BTCUSD", "ETHUSD"]
    payload["evaluations"].append(
        {
            "checkpoint": str(best_checkpoint.resolve()),
            "median_total_return": 0.02,
            "median_sortino": 0.4,
            "replay": {"hourly_goodness_score": 3.0},
        }
    )
    manifest.write_text(json.dumps(payload))

    env = os.environ.copy()
    env["LAUNCH"] = str(launch)
    env["LOG"] = str(tmp_path / "supervisor.log")
    env["SUPERVISOR_PROG"] = "test-binance-hybrid-spot"
    env["PYTHON_BIN"] = sys.executable

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--dry-run",
            "--manifest-path",
            str(manifest),
            "--manifest-best",
            "--symbols",
            "BTCUSD,ETHUSD",
        ],
        capture_output=True,
        check=False,
        text=True,
        cwd=str(REPO),
        env=env,
    )

    combined = result.stdout + result.stderr
    assert result.returncode == 0
    assert f"Resolved manifest-best checkpoint: {best_checkpoint.resolve()}" in combined
    assert f"[dry-run] Would update existing --rl-checkpoint to: {best_checkpoint.resolve()}" in combined
    assert "[dry-run] Would update --symbols to: BTCUSD ETHUSD" in combined


def test_deploy_crypto_model_dry_run_blocks_symbol_override_without_matching_manifest(tmp_path: Path) -> None:
    launch = tmp_path / "launch.sh"
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    candidate_checkpoint = tmp_path / "candidate.pt"
    candidate_checkpoint.write_text("y")
    _write_launch(launch, current_checkpoint)
    manifest = tmp_path / "prod_launch_eval_manifest.json"
    _write_manifest(manifest, current_checkpoint, candidate_checkpoint, candidate_goodness=2.0)

    env = os.environ.copy()
    env["LAUNCH"] = str(launch)
    env["LOG"] = str(tmp_path / "supervisor.log")
    env["SUPERVISOR_PROG"] = "test-binance-hybrid-spot"
    env["PYTHON_BIN"] = sys.executable

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--dry-run",
            "--manifest-path",
            str(manifest),
            str(candidate_checkpoint),
            "BTCUSD",
            "ETHUSD",
        ],
        capture_output=True,
        check=False,
        text=True,
        cwd=str(REPO),
        env=env,
    )

    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "Deployment blocked by production-eval gate." in combined
    assert "manifest launch config mismatch for symbols" in combined


def test_deploy_crypto_model_dry_run_blocks_current_checkpoint_when_symbol_override_changes_target_config(tmp_path: Path) -> None:
    launch = tmp_path / "launch.sh"
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    candidate_checkpoint = tmp_path / "candidate.pt"
    candidate_checkpoint.write_text("y")
    _write_launch(launch, current_checkpoint)
    manifest = tmp_path / "prod_launch_eval_manifest.json"
    _write_manifest(manifest, current_checkpoint, candidate_checkpoint, candidate_goodness=2.0)

    env = os.environ.copy()
    env["LAUNCH"] = str(launch)
    env["LOG"] = str(tmp_path / "supervisor.log")
    env["SUPERVISOR_PROG"] = "test-binance-hybrid-spot"
    env["PYTHON_BIN"] = sys.executable

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--dry-run",
            "--manifest-path",
            str(manifest),
            str(current_checkpoint),
            "--symbols",
            "BTCUSD,ETHUSD",
        ],
        capture_output=True,
        check=False,
        text=True,
        cwd=str(REPO),
        env=env,
    )

    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "Deployment blocked by production-eval gate." in combined
    assert "manifest launch config mismatch for symbols" in combined


def test_deploy_crypto_model_dry_run_blocks_equal_scoring_candidate(tmp_path: Path) -> None:
    launch = tmp_path / "launch.sh"
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    candidate_checkpoint = tmp_path / "candidate.pt"
    candidate_checkpoint.write_text("y")
    _write_launch(launch, current_checkpoint)
    manifest = tmp_path / "prod_launch_eval_manifest.json"
    _write_manifest(manifest, current_checkpoint, candidate_checkpoint, candidate_goodness=-1.0)

    env = os.environ.copy()
    env["LAUNCH"] = str(launch)
    env["LOG"] = str(tmp_path / "supervisor.log")
    env["SUPERVISOR_PROG"] = "test-binance-hybrid-spot"
    env["PYTHON_BIN"] = sys.executable

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--dry-run",
            "--manifest-path",
            str(manifest),
            str(candidate_checkpoint),
        ],
        capture_output=True,
        check=False,
        text=True,
        cwd=str(REPO),
        env=env,
    )

    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "Deployment blocked by production-eval gate." in combined
    assert "candidate does not beat current live checkpoint" in combined


def test_deploy_crypto_model_dry_run_blocks_less_bad_but_still_negative_candidate(tmp_path: Path) -> None:
    launch = tmp_path / "launch.sh"
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    candidate_checkpoint = tmp_path / "candidate.pt"
    candidate_checkpoint.write_text("y")
    _write_launch(launch, current_checkpoint)
    manifest = tmp_path / "prod_launch_eval_manifest.json"
    _write_manifest(manifest, current_checkpoint, candidate_checkpoint, candidate_goodness=-0.25)

    env = os.environ.copy()
    env["LAUNCH"] = str(launch)
    env["LOG"] = str(tmp_path / "supervisor.log")
    env["SUPERVISOR_PROG"] = "test-binance-hybrid-spot"
    env["PYTHON_BIN"] = sys.executable

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--dry-run",
            "--manifest-path",
            str(manifest),
            str(candidate_checkpoint),
        ],
        capture_output=True,
        check=False,
        text=True,
        cwd=str(REPO),
        env=env,
    )

    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "Deployment blocked by production-eval gate." in combined
    assert "candidate does not clear minimum production-faithful metric required to replace current live checkpoint" in combined


def test_deploy_crypto_model_dry_run_blocks_current_checkpoint_when_leverage_override_changes_target_config(tmp_path: Path) -> None:
    launch = tmp_path / "launch.sh"
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    candidate_checkpoint = tmp_path / "candidate.pt"
    candidate_checkpoint.write_text("y")
    _write_launch(launch, current_checkpoint)
    manifest = tmp_path / "prod_launch_eval_manifest.json"
    _write_manifest(manifest, current_checkpoint, candidate_checkpoint, candidate_goodness=2.0)

    env = os.environ.copy()
    env["LAUNCH"] = str(launch)
    env["LOG"] = str(tmp_path / "supervisor.log")
    env["SUPERVISOR_PROG"] = "test-binance-hybrid-spot"
    env["PYTHON_BIN"] = sys.executable

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--dry-run",
            "--manifest-path",
            str(manifest),
            str(current_checkpoint),
            "--leverage",
            "2.0",
        ],
        capture_output=True,
        check=False,
        text=True,
        cwd=str(REPO),
        env=env,
    )

    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "Deployment blocked by production-eval gate." in combined
    assert "manifest launch config mismatch for leverage" in combined


def test_deploy_crypto_model_rejects_invalid_leverage_override_even_with_force(tmp_path: Path) -> None:
    launch = tmp_path / "launch.sh"
    checkpoint = tmp_path / "current.pt"
    checkpoint.write_text("x")
    _write_launch(launch, checkpoint)

    env = os.environ.copy()
    env["LAUNCH"] = str(launch)
    env["LOG"] = str(tmp_path / "supervisor.log")
    env["SUPERVISOR_PROG"] = "test-binance-hybrid-spot"
    env["PYTHON_BIN"] = sys.executable

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--dry-run",
            "--force",
            str(checkpoint),
            "--leverage",
            "not-a-number",
        ],
        capture_output=True,
        check=False,
        text=True,
        cwd=str(REPO),
        env=env,
    )

    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "ERROR: invalid leverage override: not-a-number" in combined


def test_deploy_crypto_model_live_run_verifies_restart_state(tmp_path: Path) -> None:
    launch = tmp_path / "launch.sh"
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    candidate_checkpoint = tmp_path / "candidate.pt"
    candidate_checkpoint.write_text("y")
    _write_launch(launch, current_checkpoint)
    manifest = tmp_path / "prod_launch_eval_manifest.json"
    _write_manifest(manifest, current_checkpoint, candidate_checkpoint, candidate_goodness=2.0)
    log_path = tmp_path / "supervisor.log"
    log_path.write_text(
        f"Hybrid mode: RL={candidate_checkpoint.resolve()} + Gemini=gemini-3.1-flash-lite-preview\n"
    )

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "sudo",
        '#!/usr/bin/env bash\nif [[ "$1" == "-S" ]]; then shift; fi\nexec "$@"\n',
    )
    _write_executable(
        fake_bin / "supervisorctl",
        "#!/usr/bin/env bash\nexit 0\n",
    )
    _write_executable(
        fake_bin / "sleep",
        "#!/usr/bin/env bash\nexit 0\n",
    )

    env = os.environ.copy()
    env["LAUNCH"] = str(launch)
    env["LOG"] = str(log_path)
    env["SUPERVISOR_PROG"] = "test-binance-hybrid-spot"
    env["PYTHON_BIN"] = sys.executable
    env["TRACE_DIR"] = str(tmp_path / "trace")
    env["PATH"] = f"{fake_bin}:{env['PATH']}"

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--manifest-path",
            str(manifest),
            str(candidate_checkpoint),
        ],
        capture_output=True,
        check=False,
        text=True,
        cwd=str(REPO),
        env=env,
    )

    combined = result.stdout + result.stderr
    assert result.returncode == 0
    assert "Verifying deploy..." in combined
    assert "Decision: ALLOW -- deploy verification passed" in combined
    assert str(candidate_checkpoint.resolve()) in launch.read_text()


def test_deploy_crypto_model_waits_for_healthy_live_snapshot_when_requested(tmp_path: Path) -> None:
    launch = tmp_path / "launch.sh"
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    candidate_checkpoint = tmp_path / "candidate.pt"
    candidate_checkpoint.write_text("y")
    _write_launch(launch, current_checkpoint)
    manifest = tmp_path / "prod_launch_eval_manifest.json"
    _write_manifest(manifest, current_checkpoint, candidate_checkpoint, candidate_goodness=2.0)
    log_path = tmp_path / "supervisor.log"

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "sudo",
        '#!/usr/bin/env bash\nif [[ "$1" == "-S" ]]; then shift; fi\nexec "$@"\n',
    )
    _write_executable(
        fake_bin / "supervisorctl",
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'launch="${LAUNCH:?}"',
                'log="${LOG:?}"',
                'trace_dir="${TRACE_DIR:?}"',
                'checkpoint="$(grep -oE -- \'--rl-checkpoint [^ \\\\\\\\]*\' "$launch" | awk \'{print $2}\' | head -n 1)"',
                'ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"',
                'printf "Hybrid mode: RL=%s + Gemini=gemini-3.1-flash-lite-preview\\n" "$checkpoint" > "$log"',
                'mkdir -p "$trace_dir"',
                'python - "$trace_dir" "$ts" "$checkpoint" <<\'PY\'',
                "import json",
                "import sys",
                "from pathlib import Path",
                "trace_dir = Path(sys.argv[1])",
                "ts = sys.argv[2]",
                "checkpoint = sys.argv[3]",
                "path = trace_dir / f\"hybrid-cycle_{ts[:10].replace('-', '')}.jsonl\"",
                "path.write_text(json.dumps({",
                '    "event": "cycle_snapshot",',
                '    "cycle_started_at": ts,',
                '    "mode": "live",',
                '    "cycle_id": "c1",',
                '    "status": "completed",',
                '    "rl_checkpoint": checkpoint,',
                '    "requested_tradable_symbols": ["BTCUSD", "ETHUSD", "SOLUSD"],',
                '    "requested_leverage": 0.5,',
                '}) + "\\n", encoding="utf-8")',
                "PY",
                "exit 0",
                "",
            ]
        ),
    )
    _write_executable(
        fake_bin / "sleep",
        "#!/usr/bin/env bash\nexit 0\n",
    )

    env = os.environ.copy()
    env["LAUNCH"] = str(launch)
    env["LOG"] = str(log_path)
    env["SUPERVISOR_PROG"] = "test-binance-hybrid-spot"
    env["PYTHON_BIN"] = sys.executable
    env["TRACE_DIR"] = str(tmp_path / "trace")
    env["PATH"] = f"{fake_bin}:{env['PATH']}"

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--manifest-path",
            str(manifest),
            "--wait-for-live-cycle-seconds",
            "60",
            str(candidate_checkpoint),
        ],
        capture_output=True,
        check=False,
        text=True,
        cwd=str(REPO),
        env=env,
    )

    combined = result.stdout + result.stderr
    assert result.returncode == 0
    assert "Snap St:  completed" in combined
    assert "Decision: ALLOW -- deploy verification passed" in combined


def test_deploy_crypto_model_rolls_back_when_waited_live_snapshot_has_unexpected_order_symbol(tmp_path: Path) -> None:
    launch = tmp_path / "launch.sh"
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    candidate_checkpoint = tmp_path / "candidate.pt"
    candidate_checkpoint.write_text("y")
    _write_launch(launch, current_checkpoint)
    manifest = tmp_path / "prod_launch_eval_manifest.json"
    _write_manifest(manifest, current_checkpoint, candidate_checkpoint, candidate_goodness=2.0)
    log_path = tmp_path / "supervisor.log"
    call_count_path = tmp_path / "restart_count.txt"

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "sudo",
        '#!/usr/bin/env bash\nif [[ "$1" == "-S" ]]; then shift; fi\nexec "$@"\n',
    )
    _write_executable(
        fake_bin / "supervisorctl",
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'count_file="${SUPERVISOR_CALL_COUNT:?}"',
                'launch="${LAUNCH:?}"',
                'log="${LOG:?}"',
                'trace_dir="${TRACE_DIR:?}"',
                "count=0",
                'if [[ -f "$count_file" ]]; then count="$(cat "$count_file")"; fi',
                "count=$((count + 1))",
                'printf "%s" "$count" > "$count_file"',
                'checkpoint="$(grep -oE -- \'--rl-checkpoint [^ \\\\\\\\]*\' "$launch" | awk \'{print $2}\' | head -n 1)"',
                'printf "Hybrid mode: RL=%s + Gemini=gemini-3.1-flash-lite-preview\\n" "$checkpoint" > "$log"',
                'ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"',
                'mkdir -p "$trace_dir"',
                'python - "$trace_dir" "$ts" "$checkpoint" "$count" <<\'PY\'',
                "import json",
                "import sys",
                "from pathlib import Path",
                "trace_dir = Path(sys.argv[1])",
                "ts = sys.argv[2]",
                "checkpoint = sys.argv[3]",
                "count = int(sys.argv[4])",
                "path = trace_dir / f\"hybrid-cycle_{ts[:10].replace('-', '')}.jsonl\"",
                "snapshot = {",
                '    "event": "cycle_snapshot",',
                '    "cycle_started_at": ts,',
                '    "mode": "live",',
                '    "cycle_id": f"c{count}",',
                '    "status": "completed",',
                '    "rl_checkpoint": checkpoint,',
                '    "requested_tradable_symbols": ["BTCUSD", "ETHUSD", "SOLUSD"],',
                '    "requested_leverage": 0.5,',
                '    "symbols_detail": [{"symbol": "BTCUSD", "market_symbol": "BTCUSDT", "allocation_pct": 50.0, "current_qty": 0.1, "current_value": 100.0, "target_value": 150.0, "actions": []}],',
                '    "orders": {"placed": [{"symbol": "BTCUSDT", "side": "BUY"}]},',
                "}",
                "if count == 1:",
                '    snapshot["orders"]["placed"].append({"symbol": "DOTUSDT", "side": "SELL"})',
                'path.write_text(json.dumps(snapshot) + "\\n", encoding="utf-8")',
                "PY",
                "exit 0",
                "",
            ]
        ),
    )
    _write_executable(fake_bin / "sleep", "#!/usr/bin/env bash\nexit 0\n")

    env = os.environ.copy()
    env["LAUNCH"] = str(launch)
    env["LOG"] = str(log_path)
    env["SUPERVISOR_PROG"] = "test-binance-hybrid-spot"
    env["PYTHON_BIN"] = sys.executable
    env["TRACE_DIR"] = str(tmp_path / "trace")
    env["SUPERVISOR_CALL_COUNT"] = str(call_count_path)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--manifest-path",
            str(manifest),
            "--wait-for-live-cycle-seconds",
            "60",
            str(candidate_checkpoint),
        ],
        capture_output=True,
        check=False,
        text=True,
        cwd=str(REPO),
        env=env,
    )

    combined = result.stdout + result.stderr
    assert result.returncode == 1
    assert "unexpected order symbols: DOTUSDT" in combined
    assert "Rolling back launch.sh" in combined
    assert launch.read_text().find(str(current_checkpoint.resolve())) != -1
    assert call_count_path.read_text() == "2"


def test_deploy_crypto_model_rolls_back_when_live_snapshot_omits_checkpoint(tmp_path: Path) -> None:
    launch = tmp_path / "launch.sh"
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    candidate_checkpoint = tmp_path / "candidate.pt"
    candidate_checkpoint.write_text("y")
    _write_launch(launch, current_checkpoint)
    manifest = tmp_path / "prod_launch_eval_manifest.json"
    _write_manifest(manifest, current_checkpoint, candidate_checkpoint, candidate_goodness=2.0)
    log_path = tmp_path / "supervisor.log"
    call_count_path = tmp_path / "restart_count.txt"

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "sudo",
        '#!/usr/bin/env bash\nif [[ "$1" == "-S" ]]; then shift; fi\nexec "$@"\n',
    )
    _write_executable(
        fake_bin / "supervisorctl",
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'count_file="${SUPERVISOR_CALL_COUNT:?}"',
                'launch="${LAUNCH:?}"',
                'log="${LOG:?}"',
                'trace_dir="${TRACE_DIR:?}"',
                "count=0",
                'if [[ -f "$count_file" ]]; then count="$(cat "$count_file")"; fi',
                "count=$((count + 1))",
                'printf "%s" "$count" > "$count_file"',
                'checkpoint="$(grep -oE -- \'--rl-checkpoint [^ \\\\\\\\]*\' "$launch" | awk \'{print $2}\' | head -n 1)"',
                'printf "Hybrid mode: RL=%s + Gemini=gemini-3.1-flash-lite-preview\\n" "$checkpoint" > "$log"',
                'ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"',
                'mkdir -p "$trace_dir"',
                'python - "$trace_dir" "$ts" "$count" "$checkpoint" <<\'PY\'',
                "import json",
                "import sys",
                "from pathlib import Path",
                "trace_dir = Path(sys.argv[1])",
                "ts = sys.argv[2]",
                "count = int(sys.argv[3])",
                "checkpoint = sys.argv[4]",
                "path = trace_dir / f\"hybrid-cycle_{ts[:10].replace('-', '')}.jsonl\"",
                "snapshot = {",
                '    "event": "cycle_snapshot",',
                '    "cycle_started_at": ts,',
                '    "mode": "live",',
                '    "cycle_id": f"c{count}",',
                '    "status": "completed",',
                '    "requested_tradable_symbols": ["BTCUSD", "ETHUSD", "SOLUSD"],',
                '    "requested_leverage": 0.5,',
                "}",
                "if count > 1:",
                '    snapshot["rl_checkpoint"] = checkpoint',
                'path.write_text(json.dumps(snapshot) + "\\n", encoding="utf-8")',
                "PY",
                "exit 0",
                "",
            ]
        ),
    )
    _write_executable(fake_bin / "sleep", "#!/usr/bin/env bash\nexit 0\n")

    env = os.environ.copy()
    env["LAUNCH"] = str(launch)
    env["LOG"] = str(log_path)
    env["SUPERVISOR_PROG"] = "test-binance-hybrid-spot"
    env["PYTHON_BIN"] = sys.executable
    env["TRACE_DIR"] = str(tmp_path / "trace")
    env["SUPERVISOR_CALL_COUNT"] = str(call_count_path)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--manifest-path",
            str(manifest),
            "--wait-for-live-cycle-seconds",
            "60",
            str(candidate_checkpoint),
        ],
        capture_output=True,
        check=False,
        text=True,
        cwd=str(REPO),
        env=env,
    )

    combined = result.stdout + result.stderr
    assert result.returncode == 1
    assert "latest live snapshot does not record rl_checkpoint" in combined
    assert "Rolling back launch.sh" in combined
    assert launch.read_text().find(str(current_checkpoint.resolve())) != -1
    assert call_count_path.read_text() == "2"


def test_deploy_crypto_model_requires_multiple_healthy_live_cycles_when_requested(tmp_path: Path) -> None:
    launch = tmp_path / "launch.sh"
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    candidate_checkpoint = tmp_path / "candidate.pt"
    candidate_checkpoint.write_text("y")
    _write_launch(launch, current_checkpoint)
    manifest = tmp_path / "prod_launch_eval_manifest.json"
    _write_manifest(manifest, current_checkpoint, candidate_checkpoint, candidate_goodness=2.0)
    log_path = tmp_path / "supervisor.log"

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "sudo",
        '#!/usr/bin/env bash\nif [[ "$1" == "-S" ]]; then shift; fi\nexec "$@"\n',
    )
    _write_executable(
        fake_bin / "supervisorctl",
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'launch="${LAUNCH:?}"',
                'log="${LOG:?}"',
                'trace_dir="${TRACE_DIR:?}"',
                'checkpoint="$(grep -oE -- \'--rl-checkpoint [^ \\\\\\\\]*\' "$launch" | awk \'{print $2}\' | head -n 1)"',
                'ts1="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"',
                'ts2="$(date -u -d "$ts1 + 1 hour" +"%Y-%m-%dT%H:%M:%SZ")"',
                'printf "Hybrid mode: RL=%s + Gemini=gemini-3.1-flash-lite-preview\\n" "$checkpoint" > "$log"',
                'mkdir -p "$trace_dir"',
                'python - "$trace_dir" "$ts1" "$ts2" "$checkpoint" <<\'PY\'',
                "import json",
                "import sys",
                "from pathlib import Path",
                "trace_dir = Path(sys.argv[1])",
                "ts1 = sys.argv[2]",
                "ts2 = sys.argv[3]",
                "checkpoint = sys.argv[4]",
                "path = trace_dir / f\"hybrid-cycle_{ts1[:10].replace('-', '')}.jsonl\"",
                "rows = [",
                "    {",
                '        "event": "cycle_snapshot",',
                '        "cycle_started_at": ts1,',
                '        "mode": "live",',
                '        "cycle_id": "c1",',
                '        "status": "completed",',
                '        "rl_checkpoint": checkpoint,',
                '        "requested_tradable_symbols": ["BTCUSD", "ETHUSD", "SOLUSD"],',
                '        "requested_leverage": 0.5,',
                "    },",
                "    {",
                '        "event": "cycle_snapshot",',
                '        "cycle_started_at": ts2,',
                '        "mode": "live",',
                '        "cycle_id": "c2",',
                '        "status": "completed",',
                '        "rl_checkpoint": checkpoint,',
                '        "requested_tradable_symbols": ["BTCUSD", "ETHUSD", "SOLUSD"],',
                '        "requested_leverage": 0.5,',
                "    },",
                "]",
                'path.write_text("".join(json.dumps(row) + "\\n" for row in rows), encoding="utf-8")',
                "PY",
                "exit 0",
                "",
            ]
        ),
    )
    _write_executable(
        fake_bin / "sleep",
        "#!/usr/bin/env bash\nexit 0\n",
    )

    env = os.environ.copy()
    env["LAUNCH"] = str(launch)
    env["LOG"] = str(log_path)
    env["SUPERVISOR_PROG"] = "test-binance-hybrid-spot"
    env["PYTHON_BIN"] = sys.executable
    env["TRACE_DIR"] = str(tmp_path / "trace")
    env["PATH"] = f"{fake_bin}:{env['PATH']}"

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--manifest-path",
            str(manifest),
            "--wait-for-live-cycle-seconds",
            "60",
            "--min-healthy-live-cycles",
            "2",
            str(candidate_checkpoint),
        ],
        capture_output=True,
        check=False,
        text=True,
        cwd=str(REPO),
        env=env,
    )

    combined = result.stdout + result.stderr
    assert result.returncode == 0
    assert "Healthy:  2/2" in combined
    assert "Decision: ALLOW -- deploy verification passed" in combined


def test_deploy_crypto_model_rolls_back_when_waited_live_snapshot_is_unhealthy(tmp_path: Path) -> None:
    launch = tmp_path / "launch.sh"
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    candidate_checkpoint = tmp_path / "candidate.pt"
    candidate_checkpoint.write_text("y")
    _write_launch(launch, current_checkpoint)
    manifest = tmp_path / "prod_launch_eval_manifest.json"
    _write_manifest(manifest, current_checkpoint, candidate_checkpoint, candidate_goodness=2.0)
    log_path = tmp_path / "supervisor.log"
    call_count_path = tmp_path / "restart_count.txt"

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "sudo",
        '#!/usr/bin/env bash\nif [[ "$1" == "-S" ]]; then shift; fi\nexec "$@"\n',
    )
    _write_executable(
        fake_bin / "supervisorctl",
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'count_file="${SUPERVISOR_CALL_COUNT:?}"',
                'launch="${LAUNCH:?}"',
                'log="${LOG:?}"',
                'trace_dir="${TRACE_DIR:?}"',
                "count=0",
                'if [[ -f "$count_file" ]]; then count="$(cat "$count_file")"; fi',
                "count=$((count + 1))",
                'printf "%s" "$count" > "$count_file"',
                'checkpoint="$(grep -oE -- \'--rl-checkpoint [^ \\\\\\\\]*\' "$launch" | awk \'{print $2}\' | head -n 1)"',
                'printf "Hybrid mode: RL=%s + Gemini=gemini-3.1-flash-lite-preview\\n" "$checkpoint" > "$log"',
                'if [[ "$count" -eq 1 ]]; then',
                '  ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"',
                '  mkdir -p "$trace_dir"',
                '  python - "$trace_dir" "$ts" "$checkpoint" <<\'PY\'',
                "import json",
                "import sys",
                "from pathlib import Path",
                "trace_dir = Path(sys.argv[1])",
                "ts = sys.argv[2]",
                "checkpoint = sys.argv[3]",
                "path = trace_dir / f\"hybrid-cycle_{ts[:10].replace('-', '')}.jsonl\"",
                "path.write_text(json.dumps({",
                '    "event": "cycle_snapshot",',
                '    "cycle_started_at": ts,',
                '    "mode": "live",',
                '    "cycle_id": "c1",',
                '    "status": "context_error",',
                '    "error": "missing cache",',
                '    "rl_checkpoint": checkpoint,',
                '    "requested_tradable_symbols": ["BTCUSD", "ETHUSD", "SOLUSD"],',
                '    "requested_leverage": 0.5,',
                '}) + "\\n", encoding="utf-8")',
                "PY",
                "fi",
                "exit 0",
                "",
            ]
        ),
    )
    _write_executable(
        fake_bin / "sleep",
        "#!/usr/bin/env bash\nexit 0\n",
    )

    env = os.environ.copy()
    env["LAUNCH"] = str(launch)
    env["LOG"] = str(log_path)
    env["SUPERVISOR_PROG"] = "test-binance-hybrid-spot"
    env["PYTHON_BIN"] = sys.executable
    env["TRACE_DIR"] = str(tmp_path / "trace")
    env["SUPERVISOR_CALL_COUNT"] = str(call_count_path)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--manifest-path",
            str(manifest),
            "--wait-for-live-cycle-seconds",
            "60",
            str(candidate_checkpoint),
        ],
        capture_output=True,
        check=False,
        text=True,
        cwd=str(REPO),
        env=env,
    )

    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "latest live snapshot status is unhealthy: context_error" in combined
    assert "Rolling back launch.sh" in combined
    assert call_count_path.read_text() == "2"
    assert str(current_checkpoint.resolve()) in launch.read_text()
    assert str(candidate_checkpoint.resolve()) not in launch.read_text()


def test_deploy_crypto_model_rolls_back_when_waited_live_snapshot_uses_rl_only_fallback(tmp_path: Path) -> None:
    launch = tmp_path / "launch.sh"
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    candidate_checkpoint = tmp_path / "candidate.pt"
    candidate_checkpoint.write_text("y")
    _write_launch(launch, current_checkpoint)
    manifest = tmp_path / "prod_launch_eval_manifest.json"
    _write_manifest(manifest, current_checkpoint, candidate_checkpoint, candidate_goodness=2.0)
    log_path = tmp_path / "supervisor.log"
    call_count_path = tmp_path / "restart_count.txt"

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "sudo",
        '#!/usr/bin/env bash\nif [[ "$1" == "-S" ]]; then shift; fi\nexec "$@"\n',
    )
    _write_executable(
        fake_bin / "supervisorctl",
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'count_file="${SUPERVISOR_CALL_COUNT:?}"',
                'launch="${LAUNCH:?}"',
                'log="${LOG:?}"',
                'trace_dir="${TRACE_DIR:?}"',
                "count=0",
                'if [[ -f "$count_file" ]]; then count="$(cat "$count_file")"; fi',
                "count=$((count + 1))",
                'printf "%s" "$count" > "$count_file"',
                'checkpoint="$(grep -oE -- \'--rl-checkpoint [^ \\\\\\\\]*\' "$launch" | awk \'{print $2}\' | head -n 1)"',
                'printf "Hybrid mode: RL=%s + Gemini=gemini-3.1-flash-lite-preview\\n" "$checkpoint" > "$log"',
                'if [[ "$count" -eq 1 ]]; then',
                '  ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"',
                '  mkdir -p "$trace_dir"',
                '  python - "$trace_dir" "$ts" "$checkpoint" <<\'PY\'',
                "import json",
                "import sys",
                "from pathlib import Path",
                "trace_dir = Path(sys.argv[1])",
                "ts = sys.argv[2]",
                "checkpoint = sys.argv[3]",
                "path = trace_dir / f\"hybrid-cycle_{ts[:10].replace('-', '')}.jsonl\"",
                "path.write_text(json.dumps({",
                '    "event": "cycle_snapshot",',
                '    "cycle_started_at": ts,',
                '    "mode": "live",',
                '    "cycle_id": "c1",',
                '    "status": "completed",',
                '    "allocation_source": "rl_only_fallback",',
                '    "allocation_plan": {"reasoning": "API error: 403 PERMISSION_DENIED"},',
                '    "rl_checkpoint": checkpoint,',
                '    "requested_tradable_symbols": ["BTCUSD", "ETHUSD", "SOLUSD"],',
                '    "requested_leverage": 0.5,',
                '}) + "\\n", encoding="utf-8")',
                "PY",
                "fi",
                "exit 0",
                "",
            ]
        ),
    )
    _write_executable(
        fake_bin / "sleep",
        "#!/usr/bin/env bash\nexit 0\n",
    )

    env = os.environ.copy()
    env["LAUNCH"] = str(launch)
    env["LOG"] = str(log_path)
    env["SUPERVISOR_PROG"] = "test-binance-hybrid-spot"
    env["PYTHON_BIN"] = sys.executable
    env["TRACE_DIR"] = str(tmp_path / "trace")
    env["SUPERVISOR_CALL_COUNT"] = str(call_count_path)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--manifest-path",
            str(manifest),
            "--wait-for-live-cycle-seconds",
            "60",
            str(candidate_checkpoint),
        ],
        capture_output=True,
        check=False,
        text=True,
        cwd=str(REPO),
        env=env,
    )

    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "Gemini fallback allocation source: rl_only_fallback" in combined
    assert "Rolling back launch.sh" in combined
    assert call_count_path.read_text() == "2"
    assert str(current_checkpoint.resolve()) in launch.read_text()
    assert str(candidate_checkpoint.resolve()) not in launch.read_text()


def test_deploy_crypto_model_rolls_back_when_verification_fails(tmp_path: Path) -> None:
    launch = tmp_path / "launch.sh"
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    candidate_checkpoint = tmp_path / "candidate.pt"
    candidate_checkpoint.write_text("y")
    _write_launch(launch, current_checkpoint)
    manifest = tmp_path / "prod_launch_eval_manifest.json"
    _write_manifest(manifest, current_checkpoint, candidate_checkpoint, candidate_goodness=2.0)
    log_path = tmp_path / "supervisor.log"
    call_count_path = tmp_path / "restart_count.txt"

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "sudo",
        '#!/usr/bin/env bash\nif [[ "$1" == "-S" ]]; then shift; fi\nexec "$@"\n',
    )
    _write_executable(
        fake_bin / "supervisorctl",
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'count_file="${SUPERVISOR_CALL_COUNT:?}"',
                'launch="${LAUNCH:?}"',
                'log="${LOG:?}"',
                "count=0",
                'if [[ -f "$count_file" ]]; then count="$(cat "$count_file")"; fi',
                "count=$((count + 1))",
                'printf "%s" "$count" > "$count_file"',
                'if [[ "$count" -eq 1 ]]; then',
                '  printf "broken startup\\n" > "$log"',
                "else",
                '  checkpoint="$(grep -oE -- \'--rl-checkpoint [^ \\\\\\\\]*\' "$launch" | awk \'{print $2}\' | head -n 1)"',
                '  printf "Hybrid mode: RL=%s + Gemini=gemini-3.1-flash-lite-preview\\n" "$checkpoint" > "$log"',
                "fi",
                "exit 0",
                "",
            ]
        ),
    )
    _write_executable(
        fake_bin / "sleep",
        "#!/usr/bin/env bash\nexit 0\n",
    )

    env = os.environ.copy()
    env["LAUNCH"] = str(launch)
    env["LOG"] = str(log_path)
    env["SUPERVISOR_PROG"] = "test-binance-hybrid-spot"
    env["PYTHON_BIN"] = sys.executable
    env["TRACE_DIR"] = str(tmp_path / "trace")
    env["SUPERVISOR_CALL_COUNT"] = str(call_count_path)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--manifest-path",
            str(manifest),
            str(candidate_checkpoint),
        ],
        capture_output=True,
        check=False,
        text=True,
        cwd=str(REPO),
        env=env,
    )

    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "Rolling back launch.sh" in combined
    assert "Verifying rollback..." in combined
    assert call_count_path.read_text() == "2"
    assert str(current_checkpoint.resolve()) in launch.read_text()
    assert str(candidate_checkpoint.resolve()) not in launch.read_text()


def test_deploy_crypto_model_verifies_existing_symbols_when_not_overridden(tmp_path: Path) -> None:
    launch = tmp_path / "launch.sh"
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    candidate_checkpoint = tmp_path / "candidate.pt"
    candidate_checkpoint.write_text("y")
    _write_launch(launch, current_checkpoint)
    manifest = tmp_path / "prod_launch_eval_manifest.json"
    _write_manifest(manifest, current_checkpoint, candidate_checkpoint, candidate_goodness=2.0)
    log_path = tmp_path / "supervisor.log"
    call_count_path = tmp_path / "restart_count.txt"

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "sudo",
        '#!/usr/bin/env bash\nif [[ "$1" == "-S" ]]; then shift; fi\nexec "$@"\n',
    )
    _write_executable(
        fake_bin / "supervisorctl",
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'count_file="${SUPERVISOR_CALL_COUNT:?}"',
                'launch="${LAUNCH:?}"',
                'log="${LOG:?}"',
                "count=0",
                'if [[ -f "$count_file" ]]; then count="$(cat "$count_file")"; fi',
                "count=$((count + 1))",
                'printf "%s" "$count" > "$count_file"',
                'checkpoint="$(grep -oE -- \'--rl-checkpoint [^ \\\\\\\\]*\' "$launch" | awk \'{print $2}\' | head -n 1)"',
                'if [[ "$count" -eq 1 ]]; then',
                "  python - \"$launch\" <<'PY'",
                "from pathlib import Path",
                "import sys",
                "path = Path(sys.argv[1])",
                "text = path.read_text()",
                "text = text.replace('--symbols BTCUSD ETHUSD SOLUSD \\\\\\n', '--symbols BTCUSD ETHUSD \\\\\\n')",
                "path.write_text(text)",
                "PY",
                "fi",
                'printf "Hybrid mode: RL=%s + Gemini=gemini-3.1-flash-lite-preview\\n" "$checkpoint" > "$log"',
                "exit 0",
                "",
            ]
        ),
    )
    _write_executable(
        fake_bin / "sleep",
        "#!/usr/bin/env bash\nexit 0\n",
    )

    env = os.environ.copy()
    env["LAUNCH"] = str(launch)
    env["LOG"] = str(log_path)
    env["SUPERVISOR_PROG"] = "test-binance-hybrid-spot"
    env["PYTHON_BIN"] = sys.executable
    env["TRACE_DIR"] = str(tmp_path / "trace")
    env["SUPERVISOR_CALL_COUNT"] = str(call_count_path)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--manifest-path",
            str(manifest),
            str(candidate_checkpoint),
        ],
        capture_output=True,
        check=False,
        text=True,
        cwd=str(REPO),
        env=env,
    )

    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "launch symbols do not match expected symbols" in combined
    assert "Rolling back launch.sh" in combined
    assert call_count_path.read_text() == "2"
    assert "--symbols BTCUSD ETHUSD SOLUSD \\" in launch.read_text()


def test_deploy_crypto_model_remove_rl_verifies_gemini_only_restart(tmp_path: Path) -> None:
    launch = tmp_path / "launch.sh"
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    _write_launch(launch, current_checkpoint)
    log_path = tmp_path / "supervisor.log"

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "sudo",
        '#!/usr/bin/env bash\nif [[ "$1" == "-S" ]]; then shift; fi\nexec "$@"\n',
    )
    _write_executable(
        fake_bin / "supervisorctl",
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'launch="${LAUNCH:?}"',
                'log="${LOG:?}"',
                'checkpoint="$(grep -oE -- \'--rl-checkpoint [^ \\\\\\\\]*\' "$launch" | awk \'{print $2}\' | head -n 1)"',
                'if [[ -n "$checkpoint" ]]; then',
                '  printf "Hybrid mode: RL=%s + Gemini=gemini-3.1-flash-lite-preview\\n" "$checkpoint" > "$log"',
                "else",
                '  printf "Hybrid mode: RL=disabled + Gemini=gemini-3.1-flash-lite-preview\\n" > "$log"',
                "fi",
                "exit 0",
                "",
            ]
        ),
    )
    _write_executable(
        fake_bin / "sleep",
        "#!/usr/bin/env bash\nexit 0\n",
    )

    env = os.environ.copy()
    env["LAUNCH"] = str(launch)
    env["LOG"] = str(log_path)
    env["SUPERVISOR_PROG"] = "test-binance-hybrid-spot"
    env["PYTHON_BIN"] = sys.executable
    env["TRACE_DIR"] = str(tmp_path / "trace")
    env["PATH"] = f"{fake_bin}:{env['PATH']}"

    result = subprocess.run(
        ["bash", str(SCRIPT), "--remove-rl"],
        capture_output=True,
        check=False,
        text=True,
        cwd=str(REPO),
        env=env,
    )

    combined = result.stdout + result.stderr
    assert result.returncode == 0
    assert "Verifying deploy..." in combined
    assert "Expected: RL disabled" in combined
    assert "Decision: ALLOW -- deploy verification passed" in combined
    assert "--rl-checkpoint" not in launch.read_text()


def test_deploy_crypto_model_remove_rl_updates_symbols_when_overridden(tmp_path: Path) -> None:
    launch = tmp_path / "launch.sh"
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    _write_launch(launch, current_checkpoint)
    log_path = tmp_path / "supervisor.log"

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "sudo",
        '#!/usr/bin/env bash\nif [[ "$1" == "-S" ]]; then shift; fi\nexec "$@"\n',
    )
    _write_executable(
        fake_bin / "supervisorctl",
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'launch="${LAUNCH:?}"',
                'log="${LOG:?}"',
                'checkpoint="$(grep -oE -- \'--rl-checkpoint [^ \\\\\\\\]*\' "$launch" | awk \'{print $2}\' | head -n 1)"',
                'if [[ -n "$checkpoint" ]]; then',
                '  printf "Hybrid mode: RL=%s + Gemini=gemini-3.1-flash-lite-preview\\n" "$checkpoint" > "$log"',
                "else",
                '  printf "Hybrid mode: RL=disabled + Gemini=gemini-3.1-flash-lite-preview\\n" > "$log"',
                "fi",
                "exit 0",
                "",
            ]
        ),
    )
    _write_executable(
        fake_bin / "sleep",
        "#!/usr/bin/env bash\nexit 0\n",
    )

    env = os.environ.copy()
    env["LAUNCH"] = str(launch)
    env["LOG"] = str(log_path)
    env["SUPERVISOR_PROG"] = "test-binance-hybrid-spot"
    env["PYTHON_BIN"] = sys.executable
    env["TRACE_DIR"] = str(tmp_path / "trace")
    env["PATH"] = f"{fake_bin}:{env['PATH']}"

    result = subprocess.run(
        ["bash", str(SCRIPT), "--remove-rl", "--symbols", "BTCUSD,ETHUSD"],
        capture_output=True,
        check=False,
        text=True,
        cwd=str(REPO),
        env=env,
    )

    combined = result.stdout + result.stderr
    launch_text = launch.read_text()
    assert result.returncode == 0
    assert "Verifying deploy..." in combined
    assert "Decision: ALLOW -- deploy verification passed" in combined
    assert "--rl-checkpoint" not in launch_text
    assert "--symbols BTCUSD ETHUSD \\" in launch_text


def test_deploy_crypto_model_remove_rl_rolls_back_on_verification_failure(tmp_path: Path) -> None:
    launch = tmp_path / "launch.sh"
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    _write_launch(launch, current_checkpoint)
    log_path = tmp_path / "supervisor.log"
    call_count_path = tmp_path / "restart_count.txt"

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "sudo",
        '#!/usr/bin/env bash\nif [[ "$1" == "-S" ]]; then shift; fi\nexec "$@"\n',
    )
    _write_executable(
        fake_bin / "supervisorctl",
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'count_file="${SUPERVISOR_CALL_COUNT:?}"',
                'launch="${LAUNCH:?}"',
                'log="${LOG:?}"',
                "count=0",
                'if [[ -f "$count_file" ]]; then count="$(cat "$count_file")"; fi',
                "count=$((count + 1))",
                'printf "%s" "$count" > "$count_file"',
                'checkpoint="$(grep -oE -- \'--rl-checkpoint [^ \\\\\\\\]*\' "$launch" | awk \'{print $2}\' | head -n 1)"',
                'if [[ "$count" -eq 1 ]]; then',
                '  printf "Hybrid mode: RL=stale + Gemini=gemini-3.1-flash-lite-preview\\n" > "$log"',
                'elif [[ -n "$checkpoint" ]]; then',
                '  printf "Hybrid mode: RL=%s + Gemini=gemini-3.1-flash-lite-preview\\n" "$checkpoint" > "$log"',
                "else",
                '  printf "Hybrid mode: RL=disabled + Gemini=gemini-3.1-flash-lite-preview\\n" > "$log"',
                "fi",
                "exit 0",
                "",
            ]
        ),
    )
    _write_executable(
        fake_bin / "sleep",
        "#!/usr/bin/env bash\nexit 0\n",
    )

    env = os.environ.copy()
    env["LAUNCH"] = str(launch)
    env["LOG"] = str(log_path)
    env["SUPERVISOR_PROG"] = "test-binance-hybrid-spot"
    env["PYTHON_BIN"] = sys.executable
    env["TRACE_DIR"] = str(tmp_path / "trace")
    env["SUPERVISOR_CALL_COUNT"] = str(call_count_path)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"

    result = subprocess.run(
        ["bash", str(SCRIPT), "--remove-rl"],
        capture_output=True,
        check=False,
        text=True,
        cwd=str(REPO),
        env=env,
    )

    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "post-remove-rl verification failed" in combined
    assert "Verifying rollback..." in combined
    assert call_count_path.read_text() == "2"
    assert str(current_checkpoint.resolve()) in launch.read_text()


def test_deploy_crypto_model_redeploys_from_gemini_only_launch_and_rolls_back_cleanly(tmp_path: Path) -> None:
    launch = tmp_path / "launch.sh"
    _write_launch_without_rl(launch)
    current_checkpoint = tmp_path / "current.pt"
    current_checkpoint.write_text("x")
    candidate_checkpoint = tmp_path / "candidate.pt"
    candidate_checkpoint.write_text("y")
    manifest = tmp_path / "prod_launch_eval_manifest.json"
    _write_manifest(manifest, current_checkpoint, candidate_checkpoint, candidate_goodness=2.0)
    log_path = tmp_path / "supervisor.log"
    call_count_path = tmp_path / "restart_count.txt"

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "sudo",
        '#!/usr/bin/env bash\nif [[ "$1" == "-S" ]]; then shift; fi\nexec "$@"\n',
    )
    _write_executable(
        fake_bin / "supervisorctl",
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'count_file="${SUPERVISOR_CALL_COUNT:?}"',
                'launch="${LAUNCH:?}"',
                'log="${LOG:?}"',
                "count=0",
                'if [[ -f "$count_file" ]]; then count="$(cat "$count_file")"; fi',
                "count=$((count + 1))",
                'printf "%s" "$count" > "$count_file"',
                'checkpoint="$(grep -oE -- \'--rl-checkpoint [^ \\\\\\\\]*\' "$launch" | awk \'{print $2}\' | head -n 1)"',
                'if [[ "$count" -eq 1 ]]; then',
                '  printf "broken startup\\n" > "$log"',
                'elif [[ -n "$checkpoint" ]]; then',
                '  printf "Hybrid mode: RL=%s + Gemini=gemini-3.1-flash-lite-preview\\n" "$checkpoint" > "$log"',
                "else",
                '  printf "Hybrid mode: RL=disabled + Gemini=gemini-3.1-flash-lite-preview\\n" > "$log"',
                "fi",
                "exit 0",
                "",
            ]
        ),
    )
    _write_executable(
        fake_bin / "sleep",
        "#!/usr/bin/env bash\nexit 0\n",
    )

    env = os.environ.copy()
    env["LAUNCH"] = str(launch)
    env["LOG"] = str(log_path)
    env["SUPERVISOR_PROG"] = "test-binance-hybrid-spot"
    env["PYTHON_BIN"] = sys.executable
    env["TRACE_DIR"] = str(tmp_path / "trace")
    env["SUPERVISOR_CALL_COUNT"] = str(call_count_path)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--manifest-path",
            str(manifest),
            str(candidate_checkpoint),
        ],
        capture_output=True,
        check=False,
        text=True,
        cwd=str(REPO),
        env=env,
    )

    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "Rolling back launch.sh" in combined
    assert "Verifying rollback..." in combined
    assert "Expected: RL disabled" in combined
    assert call_count_path.read_text() == "2"
    assert "--rl-checkpoint" not in launch.read_text()
