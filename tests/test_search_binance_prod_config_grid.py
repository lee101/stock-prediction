from __future__ import annotations

import concurrent.futures
import csv
import importlib.util
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

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
    assert float(rows[0]["leverage"]) == 2.0
    assert float(rows[0]["metric_value"]) == 3.5
    assert rows[0]["checkpoint"] == str(candidate_checkpoint.resolve())
    assert rows[0]["gate_allowed"] == "False"


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


def test_run_grid_search_prefers_deployable_config_over_higher_blocked_metric(tmp_path: Path, monkeypatch) -> None:
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
    assert best_rows[0]["symbols"] == "BTCUSD ETHUSD SOLUSD"
    assert best_rows[0]["gate_allowed"] == "True"
    assert best_rows[0]["gate_candidate_metric"] == "0.4"


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


def test_run_grid_search_dry_run_prints_commands(tmp_path: Path, capsys) -> None:
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
        _write_manifest(
            output_dir / "prod_launch_eval_manifest.json",
            checkpoint,
            candidate_a,
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
        (("BTCUSD", "SOLUSD"), 0.5),
        (("BTCUSD", "ETHUSD", "SOLUSD"), 0.5),
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
