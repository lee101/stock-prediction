#!/usr/bin/env python3
# ruff: noqa: E402
"""Binance crypto autoresearch forever loop.

Spins up RunPod 4090s (or runs locally), trains models on both daily and hourly
Binance data, evaluates against production baseline, saves top-K checkpoints
and logs to R2, and writes results to numbered progress files.

Usage:
    # Local GPU (auto-detect):
    python scripts/binance_autoresearch_forever.py

    # RunPod 4090:
    python scripts/binance_autoresearch_forever.py --gpu-type 4090

    # Auto-deploy production-approved winners:
    python scripts/binance_autoresearch_forever.py --auto-deploy

    # Validate deploy path without touching supervisor:
    python scripts/binance_autoresearch_forever.py --auto-deploy --deploy-dry-run --once

    # Dry run:
    python scripts/binance_autoresearch_forever.py --dry-run

    # Single round (no forever loop):
    python scripts/binance_autoresearch_forever.py --once
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import logging
import math
import os
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.binance_deploy_gate import (
    GateResult,
    gate_deploy_candidate,
    load_launch_checkpoint_evaluation,
    production_metric_sort_key,
)
from src.binance_hybrid_launch import (
    DEFAULT_LAUNCH_SCRIPT,
    parse_launch_script,
    parse_symbols_override,
    resolve_launch_eval_constraints,
)
from src.binance_prod_config_search import (
    DEFAULT_MAX_CHECKPOINT_CONFIG_EVALS,
    DEFAULT_MAX_VARIANTS,
    build_config_variants,
    limit_config_variants,
    limit_ranked_candidates,
    max_candidate_checkpoints_for_budget,
)
from src.checkpoint_manager import TopKCheckpointManager
from src.r2_client import R2Client
from src.runpod_client import (
    DEFAULT_GPU_FALLBACKS,
    GPU_ALIASES,
    HOURLY_RATES,
    PodConfig,
    RunPodClient,
    build_gpu_fallback_types,
    is_capacity_error,
    parse_gpu_fallback_types,
    resolve_gpu_type,
)


log = logging.getLogger("binance_forever")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
DEFAULT_PROD_EVAL_TOP_K = 3
DEFAULT_PROD_CONFIG_MAX_VARIANTS = DEFAULT_MAX_VARIANTS
DEFAULT_PROD_CONFIG_MAX_CHECKPOINT_EVALS = DEFAULT_MAX_CHECKPOINT_CONFIG_EVALS

# ---------------------------------------------------------------------------
# Data tracks
# ---------------------------------------------------------------------------

@dataclass
class DataTrack:
    name: str
    train: str
    val: str
    periods_per_year: float
    max_steps: int
    fee_rate: float = 0.001
    description: str = ""
    focused_descriptions: str = ""  # comma-separated descriptions for focused sweeps

CRYPTO_DAILY = DataTrack(
    name="crypto_daily",
    train="pufferlib_market/data/crypto29_daily_train.bin",
    val="pufferlib_market/data/crypto29_daily_val.bin",
    periods_per_year=365.0,
    max_steps=90,
    description="29-symbol Binance daily bars",
)

CRYPTO_HOURLY = DataTrack(
    name="crypto_hourly",
    train="pufferlib_market/data/crypto34_hourly_train.bin",
    val="pufferlib_market/data/crypto34_hourly_val.bin",
    periods_per_year=8760.0,
    max_steps=720,
    description="34-symbol Binance hourly bars",
    focused_descriptions=",".join(
        f"c34h_{v}_s{s}"
        for v in [
            "tp01_slip5_wd01", "tp01_slip5_wd05",
            "tp03_slip5_wd01", "tp03_slip5_wd05",
            "tp05_slip8_wd01", "tp05_slip8_wd05",
        ]
        for s in [7, 19, 33, 42, 80, 99]
    ),
)

MIXED_DAILY = DataTrack(
    name="mixed_daily",
    train="pufferlib_market/data/mixed40_daily_train.bin",
    val="pufferlib_market/data/mixed40_daily_val.bin",
    periods_per_year=365.0,
    max_steps=90,
    description="40-symbol mixed daily (crypto+stocks)",
)

MIXED_HOURLY = DataTrack(
    name="mixed_hourly",
    train="pufferlib_market/data/mixed23_latest_train_20260320.bin",
    val="pufferlib_market/data/mixed23_latest_val_20250922_20260320.bin",
    periods_per_year=8760.0,
    max_steps=720,
    description="23-symbol mixed hourly",
)

ALL_TRACKS = [CRYPTO_DAILY, CRYPTO_HOURLY, MIXED_DAILY, MIXED_HOURLY]

# ---------------------------------------------------------------------------
# Production baseline
# ---------------------------------------------------------------------------

@dataclass
class ProductionBaseline:
    val_return: float | None = None
    val_sortino: float | None = None
    win_rate: float | None = None
    description: str = "current live checkpoint"
    checkpoint: str = ""
    source: str = "unknown"
    manifest_path: str | None = None
    production_metric_name: str | None = None
    production_metric_value: float | None = None
    symbols: str = ""
    leverage: float | None = None

    def combined_score(self) -> float:
        return 0.5 * (self.val_return or 0.0) + 0.5 * (self.val_sortino or 0.0)

    def to_dict(self) -> dict:
        return {
            "val_return": self.val_return,
            "val_sortino": self.val_sortino,
            "win_rate": self.win_rate,
            "combined_score": self.combined_score(),
            "description": self.description,
            "checkpoint": self.checkpoint,
            "source": self.source,
            "manifest_path": self.manifest_path,
            "production_metric_name": self.production_metric_name,
            "production_metric_value": self.production_metric_value,
            "symbols": self.symbols,
            "leverage": self.leverage,
        }

# ---------------------------------------------------------------------------
# Trial result
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    description: str
    track: str
    val_return: float | None = None
    val_sortino: float | None = None
    win_rate: float | None = None
    rank_metric: str | None = None
    rank_score: float | None = None
    holdout_robust_score: float | None = None
    best_checkpoint: str | None = None
    production_manifest_path: str | None = None
    production_metric_name: str | None = None
    production_metric_current: float | None = None
    production_metric_candidate: float | None = None
    production_gate_reason: str | None = None
    production_gate_allowed: bool | None = None
    production_symbols_override: str = ""
    production_leverage_override: float | None = None
    deploy_attempted: bool = False
    deploy_succeeded: bool | None = None
    deploy_dry_run: bool = False
    deploy_message: str | None = None
    training_time_s: float = 0.0
    gpu_type: str = ""
    config: dict = field(default_factory=dict)
    early_stopped: bool = False
    error: str | None = None

    def combined_score(self) -> float:
        r = self.val_return or 0.0
        s = self.val_sortino or 0.0
        return 0.5 * r + 0.5 * s

    def beats_baseline(self, baseline: ProductionBaseline) -> bool:
        if self.production_gate_reason is None:
            if self.val_return is None or self.val_sortino is None:
                return False
            return self.combined_score() > baseline.combined_score()

        if self.production_gate_allowed is not True:
            return False
        if self.production_gate_reason == "candidate already matches current live checkpoint":
            return False
        if baseline.checkpoint and self.best_checkpoint:
            baseline_checkpoint = str(Path(baseline.checkpoint).expanduser().resolve(strict=False))
            candidate_checkpoint = str(Path(self.best_checkpoint).expanduser().resolve(strict=False))
            if candidate_checkpoint == baseline_checkpoint:
                return (
                    _trial_result_changes_production_config(self, baseline)
                    and self.production_metric_name is not None
                    and self.production_metric_candidate is not None
                )
        return self.production_metric_name is not None and self.production_metric_candidate is not None


@dataclass(frozen=True)
class ProductionConfigSelection:
    checkpoint: str
    manifest_path: str
    symbols_override: str
    leverage_override: float
    gate_result: GateResult
    val_return: float | None = None
    val_sortino: float | None = None


def _normalized_symbol_text(raw: str) -> str:
    return " ".join(parse_symbols_override(raw) or [])


def _trial_result_changes_production_config(result: TrialResult, baseline: ProductionBaseline) -> bool:
    result_symbols = _normalized_symbol_text(result.production_symbols_override)
    baseline_symbols = _normalized_symbol_text(baseline.symbols)
    symbols_changed = bool(result_symbols) and result_symbols != baseline_symbols
    leverage_changed = (
        result.production_leverage_override is not None
        and baseline.leverage is not None
        and float(result.production_leverage_override) != float(baseline.leverage)
    )
    return symbols_changed or leverage_changed


def _checkpoint_label(checkpoint: str | Path) -> str:
    checkpoint_path = Path(checkpoint)
    parent = checkpoint_path.parent.name.strip()
    if parent:
        return parent
    return checkpoint_path.stem or str(checkpoint_path)


def _baseline_from_launch(
    launch_script: str | Path = DEFAULT_LAUNCH_SCRIPT,
    *,
    source: str = "live_launch",
) -> ProductionBaseline:
    launch_cfg = parse_launch_script(launch_script, require_rl_checkpoint=False)
    if not launch_cfg.rl_checkpoint:
        return ProductionBaseline(
            description="Gemini-only live launch",
            checkpoint="",
            source=source,
            symbols=" ".join(launch_cfg.symbols),
            leverage=float(launch_cfg.leverage),
        )
    return ProductionBaseline(
        description=f"{_checkpoint_label(launch_cfg.rl_checkpoint)} (live launch)",
        checkpoint=str(launch_cfg.rl_checkpoint),
        source=source,
        symbols=" ".join(launch_cfg.symbols),
        leverage=float(launch_cfg.leverage),
    )


def evaluate_candidate_vs_production(
    candidate_checkpoint: str | Path,
    *,
    run_id: str,
    launch_script: str | Path = DEFAULT_LAUNCH_SCRIPT,
    output_root: str | Path | None = None,
    python_bin: str | Path = sys.executable,
    symbols_override: str = "",
    leverage_override: float | None = None,
    require_runtime_match: bool = True,
    require_runtime_health: bool = True,
) -> tuple[ProductionBaseline, GateResult]:
    baseline, gate_results = evaluate_candidates_vs_production(
        [candidate_checkpoint],
        run_id=run_id,
        launch_script=launch_script,
        output_root=output_root,
        python_bin=python_bin,
        symbols_override=symbols_override,
        leverage_override=leverage_override,
        require_runtime_match=require_runtime_match,
        require_runtime_health=require_runtime_health,
    )
    normalized = str(Path(candidate_checkpoint).expanduser().resolve(strict=False))
    return baseline, gate_results[normalized]


def deploy_candidate_checkpoint(
    candidate_checkpoint: str | Path,
    *,
    manifest_path: str | Path,
    launch_script: str | Path = DEFAULT_LAUNCH_SCRIPT,
    deploy_script: str | Path | None = None,
    python_bin: str | Path = sys.executable,
    dry_run: bool = False,
    symbols_override: str = "",
    leverage_override: float | None = None,
    wait_for_live_cycle_seconds: float | None = None,
    min_healthy_live_cycles: int | None = None,
) -> tuple[bool, str]:
    script_path = Path(deploy_script) if deploy_script is not None else REPO / "scripts" / "deploy_crypto_model.sh"
    cmd = ["bash", str(script_path)]
    if dry_run:
        cmd.append("--dry-run")
    cmd += [
        "--manifest-path",
        str(manifest_path),
        str(candidate_checkpoint),
    ]
    if symbols_override.strip():
        cmd.extend(["--symbols", symbols_override.strip()])
    if leverage_override is not None:
        cmd.extend(["--leverage", str(float(leverage_override))])
    if wait_for_live_cycle_seconds is not None:
        cmd.extend(["--wait-for-live-cycle-seconds", str(float(wait_for_live_cycle_seconds))])
    if min_healthy_live_cycles is not None:
        cmd.extend(["--min-healthy-live-cycles", str(int(min_healthy_live_cycles))])
    env = os.environ.copy()
    env["LAUNCH"] = str(Path(launch_script).resolve())
    env["PYTHON_BIN"] = str(python_bin)
    completed = subprocess.run(
        cmd,
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    output = "\n".join(part for part in (completed.stdout.strip(), completed.stderr.strip()) if part).strip()
    if completed.returncode == 0:
        return True, output or ("deployment dry-run succeeded" if dry_run else "deployment succeeded")
    return False, output or f"deployment failed with returncode={completed.returncode}"


def evaluate_candidates_vs_production(
    candidate_checkpoints: list[str | Path],
    *,
    run_id: str,
    launch_script: str | Path = DEFAULT_LAUNCH_SCRIPT,
    output_root: str | Path | None = None,
    python_bin: str | Path = sys.executable,
    symbols_override: str = "",
    leverage_override: float | None = None,
    require_runtime_match: bool = True,
    require_runtime_health: bool = True,
) -> tuple[ProductionBaseline, dict[str, GateResult]]:
    normalized_candidates: list[str] = []
    seen: set[str] = set()
    for checkpoint in candidate_checkpoints:
        normalized = str(Path(checkpoint).expanduser().resolve(strict=False))
        if normalized in seen:
            continue
        seen.add(normalized)
        normalized_candidates.append(normalized)
    if not normalized_candidates:
        raise ValueError("at least one candidate checkpoint is required")

    output_root_path = Path(output_root) if output_root is not None else REPO / "analysis"
    output_dir = output_root_path / f"{run_id}_prod_eval"
    cmd = [
        str(python_bin),
        str(REPO / "scripts" / "evaluate_binance_hybrid_prod.py"),
        "--launch-script",
        str(launch_script),
        "--output-dir",
        str(output_dir),
    ]
    if require_runtime_match:
        cmd.append("--require-runtime-match")
    else:
        cmd.append("--no-require-runtime-match")
    if require_runtime_health:
        cmd.append("--require-runtime-health")
    else:
        cmd.append("--no-require-runtime-health")
    if symbols_override.strip():
        cmd.extend(["--symbols", symbols_override.strip()])
    if leverage_override is not None:
        cmd.extend(["--leverage", str(float(leverage_override))])
    for checkpoint in normalized_candidates:
        cmd.extend(["--candidate-checkpoint", checkpoint])

    completed = subprocess.run(
        cmd,
        cwd=str(REPO),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "production comparison failed: "
            f"returncode={completed.returncode} stdout={completed.stdout.strip()} stderr={completed.stderr.strip()}"
        )

    manifest_path = output_dir / "prod_launch_eval_manifest.json"
    gate_results = {
        checkpoint: gate_deploy_candidate(
            candidate_checkpoint=checkpoint,
            launch_script=launch_script,
            manifest_path=manifest_path,
            symbols_override=symbols_override or None,
            leverage_override=leverage_override,
        )
        for checkpoint in normalized_candidates
    }
    baseline = _baseline_from_launch(launch_script, source="production_eval_manifest")
    baseline.manifest_path = str(manifest_path)
    manifest_eval = load_launch_checkpoint_evaluation(
        launch_script=launch_script,
        manifest_path=manifest_path,
        symbols_override=symbols_override or None,
        leverage_override=leverage_override,
    )
    if manifest_eval is not None:
        _manifest_path, evaluation = manifest_eval
        baseline.val_return = _safe_float(evaluation.get("median_total_return"))
        baseline.val_sortino = _safe_float(evaluation.get("median_sortino"))
    for gate_result in gate_results.values():
        if gate_result.metric_name is not None and gate_result.current_metric is not None:
            baseline.production_metric_name = gate_result.metric_name
            baseline.production_metric_value = gate_result.current_metric
            break
    return baseline, gate_results


def _merge_prod_symbol_sets(symbol_specs: list[str]) -> str:
    ordered_symbols: list[str] = []
    seen_symbols: set[str] = set()
    for spec in symbol_specs:
        for symbol in parse_symbols_override(spec) or []:
            if symbol in seen_symbols:
                continue
            seen_symbols.add(symbol)
            ordered_symbols.append(symbol)
    return ",".join(ordered_symbols)


def evaluate_candidates_across_prod_configs(
    candidate_checkpoints: list[str | Path],
    *,
    run_id: str,
    launch_script: str | Path = DEFAULT_LAUNCH_SCRIPT,
    output_root: str | Path | None = None,
    python_bin: str | Path = sys.executable,
    symbol_sets: list[str] | None = None,
    symbol_subset_sizes: list[int] | None = None,
    leverage_options: list[float] | None = None,
    jobs: int = 1,
    max_variants: int = DEFAULT_PROD_CONFIG_MAX_VARIANTS,
    max_checkpoint_config_evals: int = DEFAULT_PROD_CONFIG_MAX_CHECKPOINT_EVALS,
    include_launch_checkpoint: bool = True,
    variant_offset: int = 0,
) -> tuple[ProductionBaseline, ProductionConfigSelection]:
    normalized_candidates: list[str] = []
    seen: set[str] = set()
    for checkpoint in candidate_checkpoints:
        normalized = str(Path(checkpoint).expanduser().resolve(strict=False))
        if normalized in seen:
            continue
        seen.add(normalized)
        normalized_candidates.append(normalized)
    if not normalized_candidates:
        raise ValueError("at least one candidate checkpoint is required")
    if jobs < 1:
        raise ValueError("jobs must be at least 1")
    if max_variants < 0:
        raise ValueError("max_variants must be >= 0")
    if max_checkpoint_config_evals < 0:
        raise ValueError("max_checkpoint_config_evals must be >= 0")
    if variant_offset < 0:
        raise ValueError("variant_offset must be >= 0")

    output_root_path = Path(output_root) if output_root is not None else REPO / "analysis"
    output_dir = output_root_path / f"{run_id}_prod_config_grid"
    cmd = [
        str(python_bin),
        str(REPO / "scripts" / "search_binance_prod_config_grid.py"),
        "--launch-script",
        str(launch_script),
        "--output-dir",
        str(output_dir),
        "--python-bin",
        str(python_bin),
        "--top-k",
        "1",
        "--jobs",
        str(int(jobs)),
        "--max-variants",
        str(int(max_variants)),
        "--max-checkpoint-config-evals",
        str(int(max_checkpoint_config_evals)),
        "--variant-offset",
        str(int(variant_offset)),
    ]
    if include_launch_checkpoint:
        cmd.append("--include-launch-checkpoint")
    for symbol_set in symbol_sets or []:
        if str(symbol_set).strip():
            cmd.extend(["--symbols-set", str(symbol_set).strip()])
    for subset_size in symbol_subset_sizes or []:
        cmd.extend(["--symbols-subset-size", str(int(subset_size))])
    for leverage in leverage_options or []:
        cmd.extend(["--leverage-option", str(float(leverage))])
    for checkpoint in normalized_candidates:
        cmd.extend(["--candidate-checkpoint", checkpoint])

    completed = subprocess.run(
        cmd,
        cwd=str(REPO),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "production config grid search failed: "
            f"returncode={completed.returncode} stdout={completed.stdout.strip()} stderr={completed.stderr.strip()}"
        )

    best_rows_path = output_dir / "best_by_config.csv"
    if not best_rows_path.exists():
        raise RuntimeError(f"production config grid search missing best_by_config.csv: {best_rows_path}")
    with best_rows_path.open() as handle:
        best_rows = list(csv.DictReader(handle))
    if not best_rows:
        raise RuntimeError("production config grid search produced no ranked rows")

    all_rows_path = output_dir / "all_results.csv"
    all_rows: list[dict[str, str]] = []
    if all_rows_path.exists():
        with all_rows_path.open() as handle:
            all_rows = list(csv.DictReader(handle))

    best_row = best_rows[0]
    selected_checkpoint = str(best_row.get("checkpoint") or "").strip()
    selected_manifest_path = str(best_row.get("manifest_path") or "").strip()
    selected_symbols = str(best_row.get("symbols") or "").strip()
    leverage_value = _safe_float(best_row.get("leverage"))
    if not selected_checkpoint:
        raise RuntimeError("production config grid search did not return a checkpoint")
    if not selected_manifest_path:
        raise RuntimeError("production config grid search did not return a manifest path")
    if leverage_value is None:
        raise RuntimeError("production config grid search did not return a leverage value")

    gate_result = gate_deploy_candidate(
        candidate_checkpoint=selected_checkpoint,
        launch_script=launch_script,
        manifest_path=selected_manifest_path,
        symbols_override=selected_symbols or None,
        leverage_override=leverage_value,
    )
    baseline = _baseline_from_launch(launch_script, source="production_eval_manifest")
    launch_cfg = parse_launch_script(launch_script, require_rl_checkpoint=False)
    launch_checkpoint_norm = (
        str(Path(launch_cfg.rl_checkpoint).expanduser().resolve(strict=False))
        if launch_cfg.rl_checkpoint
        else ""
    )
    launch_symbols = " ".join(launch_cfg.symbols)
    launch_leverage = float(launch_cfg.leverage)
    live_row = next(
        (
            row
            for row in all_rows
            if str(row.get("checkpoint") or "").strip() == launch_checkpoint_norm
            and _normalized_symbol_text(str(row.get("symbols") or "")) == _normalized_symbol_text(launch_symbols)
            and _safe_float(row.get("leverage")) == launch_leverage
        ),
        None,
    )
    if live_row is not None:
        baseline.manifest_path = str(live_row.get("manifest_path") or "") or selected_manifest_path
        baseline.val_return = _safe_float(live_row.get("median_total_return"))
        baseline.val_sortino = _safe_float(live_row.get("median_sortino"))
        baseline.production_metric_name = str(live_row.get("gate_metric_name") or live_row.get("metric_name") or "") or None
        baseline.production_metric_value = _safe_float(live_row.get("gate_candidate_metric"))
        if baseline.production_metric_value is None:
            baseline.production_metric_value = _safe_float(live_row.get("metric_value"))
    else:
        baseline.manifest_path = selected_manifest_path
        manifest_eval = load_launch_checkpoint_evaluation(
            launch_script=launch_script,
            manifest_path=selected_manifest_path,
            symbols_override=selected_symbols or None,
            leverage_override=leverage_value,
        )
        if manifest_eval is not None:
            _manifest_path, evaluation = manifest_eval
            baseline.val_return = _safe_float(evaluation.get("median_total_return"))
            baseline.val_sortino = _safe_float(evaluation.get("median_sortino"))
        if gate_result.metric_name is not None and gate_result.current_metric is not None:
            baseline.production_metric_name = gate_result.metric_name
            baseline.production_metric_value = gate_result.current_metric
    return baseline, ProductionConfigSelection(
        checkpoint=selected_checkpoint,
        manifest_path=selected_manifest_path,
        symbols_override=selected_symbols,
        leverage_override=leverage_value,
        gate_result=gate_result,
        val_return=_safe_float(best_row.get("median_total_return")),
        val_sortino=_safe_float(best_row.get("median_sortino")),
    )

# ---------------------------------------------------------------------------
# Progress file management
# ---------------------------------------------------------------------------

def _find_next_progress_number() -> int:
    existing = list(REPO.glob("binanceprogress*.md"))
    max_num = 0
    for p in existing:
        name = p.stem
        if name == "binanceprogress_failed":
            continue
        digits = "".join(c for c in name.replace("binanceprogress", "") if c.isdigit())
        if digits:
            max_num = max(max_num, int(digits))
    return max_num + 1


def write_success_progress(result: TrialResult, baseline: ProductionBaseline, round_num: int) -> Path:
    num = _find_next_progress_number()
    path = REPO / f"binanceprogress{num}.md"
    now = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M UTC")
    content = f"""# Binance Autoresearch Success #{num} ({now})

## Round {round_num} -- {result.description}

**Track**: {result.track}
**GPU**: {result.gpu_type}
**Training time**: {result.training_time_s:.0f}s
**Early stopped**: {result.early_stopped}

## Results

| Metric | Trial | Production Baseline |
|--------|-------|-------------------|
| Return | {_fmt(result.val_return)} | {_fmt(baseline.val_return)} |
| Sortino | {_fmt(result.val_sortino)} | {_fmt(baseline.val_sortino)} |
| Win Rate | {_fmt(result.win_rate)} | {_fmt(baseline.win_rate)} |
| Combined | {result.combined_score():.4f} | {baseline.combined_score():.4f} |
| Rank Score | {_fmt(result.rank_score)} | -- |
| Holdout Robust | {_fmt(result.holdout_robust_score)} | -- |

**BEATS PRODUCTION**: YES

"""
    if result.production_metric_name and result.production_metric_candidate is not None:
        content += f"""
## Production-Faithful Comparison

**Metric**: `{result.production_metric_name}`
**Candidate**: {_fmt(result.production_metric_candidate)}
**Live**: {_fmt(baseline.production_metric_value)}
**Manifest**: `{result.production_manifest_path or baseline.manifest_path or 'N/A'}`
**Gate**: {result.production_gate_reason or 'candidate passes live comparison'}

"""
    if result.deploy_attempted:
        content += f"""
## Deployment

**Mode**: {'dry-run' if result.deploy_dry_run else 'live'}
**Status**: {'success' if result.deploy_succeeded else 'failed'}
**Details**:
```
{result.deploy_message or 'N/A'}
```

"""
    content += f"""## Checkpoint
`{result.best_checkpoint or 'N/A'}`

## Config
```json
{json.dumps(result.config, indent=2)}
```
"""
    path.write_text(content)
    log.info("wrote success progress: %s", path)
    return path


def append_failure_progress(result: TrialResult, baseline: ProductionBaseline, round_num: int) -> Path:
    path = REPO / "binanceprogress_failed.md"
    now = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M UTC")
    entry = f"""
---

## Round {round_num} -- {result.description} ({now})

**Track**: {result.track} | **GPU**: {result.gpu_type} | **Time**: {result.training_time_s:.0f}s | **Early stopped**: {result.early_stopped}

| Metric | Trial | Baseline |
|--------|-------|----------|
| Return | {_fmt(result.val_return)} | {_fmt(baseline.val_return)} |
| Sortino | {_fmt(result.val_sortino)} | {_fmt(baseline.val_sortino)} |
| Combined | {result.combined_score():.4f} | {baseline.combined_score():.4f} |

"""
    if result.production_gate_reason:
        entry += (
            f"Production comparison: `{result.production_metric_name or 'N/A'}` "
            f"candidate={_fmt(result.production_metric_candidate)} "
            f"live={_fmt(baseline.production_metric_value)} "
            f"reason={result.production_gate_reason}\n\n"
        )
    if result.deploy_attempted:
        entry += (
            f"Deployment: mode={'dry-run' if result.deploy_dry_run else 'live'} "
            f"status={'success' if result.deploy_succeeded else 'failed'} "
            f"details={result.deploy_message or 'N/A'}\n\n"
        )
    if result.error:
        entry += f"**Error**: {result.error}\n\n"
    if result.config:
        entry += f"Config: `{json.dumps(result.config)}`\n"

    if not path.exists():
        header = "# Binance Autoresearch -- Failed Trials\n\nTrials that did not beat the production baseline.\n"
        path.write_text(header + entry)
    else:
        with open(path, "a") as f:
            f.write(entry)
    return path


def _fmt(v: float | None) -> str:
    if v is None:
        return "N/A"
    return f"{v:+.4f}"

# ---------------------------------------------------------------------------
# R2 sync
# ---------------------------------------------------------------------------

def sync_to_r2(checkpoint_dir: Path, log_file: Path, run_id: str) -> None:
    try:
        client = R2Client()
        prefix = f"binance_autoresearch/{run_id}"

        if checkpoint_dir.exists():
            keys = client.sync_dir_to_r2(str(checkpoint_dir), f"{prefix}/checkpoints", skip_existing=True)
            log.info("R2: uploaded %d checkpoint files", len(keys))

        if log_file.exists():
            client.upload_file(str(log_file), f"{prefix}/trial_log.jsonl")
            log.info("R2: uploaded trial log")

    except Exception as e:
        log.warning("R2 sync failed: %s", e)

# ---------------------------------------------------------------------------
# SSH helpers for RunPod
# ---------------------------------------------------------------------------

_SSH_OPTS = ["-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes"]


def _ssh_run(host: str, port: int, cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    full = ["ssh", *_SSH_OPTS, "-p", str(port), f"root@{host}", cmd]
    return subprocess.run(full, check=check, capture_output=True, text=True)


def _rsync_to_pod(host: str, port: int, local_dir: Path, remote_dir: str) -> None:
    cmd = [
        "rsync", "-az", "--delete",
        "--exclude", "__pycache__/", "--exclude", ".git/",
        "--exclude", "pufferlib_market/data/", "--exclude", "pufferlib_market/checkpoints/",
        "--exclude", ".venv*/", "--exclude", "*.pyc",
        "-e", f"ssh {' '.join(_SSH_OPTS)} -p {port}",
        f"{local_dir}/", f"root@{host}:{remote_dir}/",
    ]
    subprocess.run(cmd, check=True)


def _rsync_data(host: str, port: int, local_path: Path, remote_dir: str) -> None:
    rel = local_path.relative_to(REPO) if local_path.is_relative_to(REPO) else Path(local_path.name)
    remote_parent = str(Path(remote_dir) / rel.parent)
    _ssh_run(host, port, f"mkdir -p {remote_parent}")
    cmd = [
        "rsync", "-az",
        "-e", f"ssh {' '.join(_SSH_OPTS)} -p {port}",
        str(local_path), f"root@{host}:{remote_dir}/{rel}",
    ]
    subprocess.run(cmd, check=True)


def _rsync_from_pod(host: str, port: int, remote_path: str, local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "rsync", "-az",
        "-e", f"ssh {' '.join(_SSH_OPTS)} -p {port}",
        f"root@{host}:{remote_path}/", f"{local_dir}/",
    ]
    subprocess.run(cmd, check=False)


def _bootstrap_pod(host: str, port: int, remote_dir: str) -> None:
    bootstrap = (
        f"set -euo pipefail && cd {remote_dir} && "
        f"pip install uv -q && "
        f"uv venv .venv313 --python python3.13 2>/dev/null || uv venv .venv313 && "
        f"source .venv313/bin/activate && "
        f"uv pip install -e . -q && "
        f"{{ [ -d PufferLib ] && uv pip install -e PufferLib/ -q || true; }} && "
        f"cd pufferlib_market && python setup.py build_ext --inplace -q && cd .."
    )
    _ssh_run(host, port, bootstrap)

# ---------------------------------------------------------------------------
# Run a single autoresearch batch on a pod or locally
# ---------------------------------------------------------------------------

def run_autoresearch_batch(
    track: DataTrack,
    *,
    time_budget: int = 300,
    max_trials: int = 15,
    run_id: str = "",
    gpu_type: str = "",
    ssh_host: str = "",
    ssh_port: int = 0,
    remote_dir: str = "/workspace/stock-prediction",
    local: bool = False,
    descriptions: str = "",
    holdout_max_leverage: float = 1.0,
    eval_tradable_symbols: str = "",
    eval_disable_shorts: bool = False,
) -> tuple[Path, Path]:
    """Run autoresearch_rl and return (leaderboard_path, checkpoint_dir)."""
    if not run_id:
        run_id = f"binance_forever_{track.name}_{time.strftime('%Y%m%d_%H%M%S')}"

    leaderboard = REPO / "analysis" / f"{run_id}_leaderboard.csv"
    checkpoint_dir = REPO / "pufferlib_market" / "checkpoints" / run_id

    cmd_parts = [
        sys.executable, "-u", "-m", "pufferlib_market.autoresearch_rl",
        "--train-data", track.train,
        "--val-data", track.val,
        "--time-budget", str(time_budget),
        "--max-trials", str(max_trials),
        "--periods-per-year", str(track.periods_per_year),
        "--max-steps-override", str(track.max_steps),
        "--fee-rate-override", str(track.fee_rate),
        "--leaderboard", str(leaderboard),
        "--checkpoint-root", str(checkpoint_dir),
        "--holdout-n-windows", "20",
        "--holdout-fill-buffer-bps", "5",
        "--holdout-max-leverage", str(float(holdout_max_leverage)),
        "--poly-prune",
    ]
    if descriptions:
        cmd_parts += ["--descriptions", descriptions]
    if eval_tradable_symbols:
        cmd_parts += ["--eval-tradable-symbols", eval_tradable_symbols]
    if eval_disable_shorts:
        cmd_parts.append("--eval-disable-shorts")

    if local:
        log.info("running locally: %s", run_id)
        leaderboard.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(cmd_parts, cwd=str(REPO), check=False)
    else:
        remote_lb = f"pufferlib_market/{run_id}_leaderboard.csv"
        remote_ckpt = f"pufferlib_market/checkpoints/{run_id}"
        remote_cmd_parts = [
            "python", "-u", "-m", "pufferlib_market.autoresearch_rl",
            "--train-data", track.train,
            "--val-data", track.val,
            "--time-budget", str(time_budget),
            "--max-trials", str(max_trials),
            "--periods-per-year", str(track.periods_per_year),
            "--max-steps-override", str(track.max_steps),
            "--fee-rate-override", str(track.fee_rate),
            "--leaderboard", remote_lb,
            "--checkpoint-root", remote_ckpt,
            "--holdout-n-windows", "20",
            "--holdout-fill-buffer-bps", "5",
            "--holdout-max-leverage", str(float(holdout_max_leverage)),
            "--poly-prune",
        ]
        if descriptions:
            remote_cmd_parts += ["--descriptions", descriptions]
        if eval_tradable_symbols:
            remote_cmd_parts += ["--eval-tradable-symbols", eval_tradable_symbols]
        if eval_disable_shorts:
            remote_cmd_parts.append("--eval-disable-shorts")

        wandb_key = os.environ.get("WANDB_API_KEY", "")
        wandb_export = f"export WANDB_API_KEY={wandb_key} && " if wandb_key else ""
        shell_cmd = (
            f"cd {remote_dir} && source .venv313/bin/activate && "
            f"export PYTHONPATH={remote_dir}:${{PYTHONPATH:-}} && "
            f"{wandb_export}"
            f"{' '.join(remote_cmd_parts)}"
        )
        log.info("running on pod %s:%d: %s", ssh_host, ssh_port, run_id)
        _ssh_run(ssh_host, ssh_port, shell_cmd, check=False)

        # download results
        leaderboard.parent.mkdir(parents=True, exist_ok=True)
        _scp_from_pod(ssh_host, ssh_port, f"{remote_dir}/{remote_lb}", leaderboard)
        _rsync_from_pod(ssh_host, ssh_port, f"{remote_dir}/{remote_ckpt}", checkpoint_dir)

    return leaderboard, checkpoint_dir


def _scp_from_pod(host: str, port: int, remote_path: str, local_path: Path) -> bool:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["scp", *_SSH_OPTS, "-P", str(port), f"root@{host}:{remote_path}", str(local_path)]
    return subprocess.run(cmd, check=False).returncode == 0

# ---------------------------------------------------------------------------
# Read leaderboard and find best result
# ---------------------------------------------------------------------------

def _leaderboard_score(row: dict) -> float:
    for metric_field in ("rank_score", "val_return", "val_sortino"):
        value = _safe_float(row.get(metric_field))
        if value is not None:
            return value
    return -float("inf")


def _result_from_leaderboard_row(row: dict, checkpoint_dir: Path) -> TrialResult:
    result = TrialResult(description=row.get("description", row.get("model", "unknown")), track="")
    result.rank_metric = row.get("rank_metric") or None
    result.rank_score = _safe_float(row.get("rank_score"))
    result.val_return = _safe_float(row.get("val_return"))
    result.val_sortino = _safe_float(row.get("val_sortino"))
    result.win_rate = _safe_float(row.get("val_wr"))
    result.holdout_robust_score = _safe_float(row.get("holdout_robust_score"))

    ckpt_path = row.get("checkpoint_path", "")
    if ckpt_path and Path(ckpt_path).exists():
        result.best_checkpoint = ckpt_path
        return result

    desc = result.description
    candidates = list(checkpoint_dir.glob(f"*{desc}*/best.pt"))
    if candidates:
        result.best_checkpoint = str(candidates[0])
        return result

    pts = sorted(checkpoint_dir.rglob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if pts:
        result.best_checkpoint = str(pts[0])
    return result


def read_top_from_leaderboard(leaderboard: Path, checkpoint_dir: Path, *, limit: int = 1) -> list[TrialResult]:
    if not leaderboard.exists():
        result = TrialResult(description="none", track="")
        result.error = "leaderboard not found"
        return [result]

    try:
        with open(leaderboard) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as e:
        result = TrialResult(description="none", track="")
        result.error = str(e)
        return [result]

    if not rows:
        result = TrialResult(description="none", track="")
        result.error = "empty leaderboard"
        return [result]

    rows.sort(key=_leaderboard_score, reverse=True)
    return [_result_from_leaderboard_row(row, checkpoint_dir) for row in rows[: max(1, int(limit))]]


def read_best_from_leaderboard(leaderboard: Path, checkpoint_dir: Path) -> TrialResult:
    return read_top_from_leaderboard(leaderboard, checkpoint_dir, limit=1)[0]


def _selection_sort_key(result: TrialResult, baseline: ProductionBaseline) -> tuple[float, float, int, float, float]:
    improved = 1.0 if result.beats_baseline(baseline) else 0.0
    gate_allowed = 1.0 if result.production_gate_allowed is True else 0.0
    metric_priority, production_metric = production_metric_sort_key(
        result.production_metric_name,
        result.production_metric_candidate,
    )
    rank_score = result.rank_score
    if rank_score is None:
        rank_score = result.combined_score()
    return improved, gate_allowed, metric_priority, production_metric, rank_score


def select_result_for_production(results: list[TrialResult], baseline: ProductionBaseline) -> TrialResult:
    if not results:
        return TrialResult(description="none", track="", error="no leaderboard candidates")
    return max(results, key=lambda result: _selection_sort_key(result, baseline))


def _safe_float(v) -> float | None:
    if v in (None, "", "None", "nan"):
        return None
    try:
        f = float(v)
        if math.isnan(f):
            return None
        return f
    except (TypeError, ValueError):
        return None

# ---------------------------------------------------------------------------
# Pod lifecycle
# ---------------------------------------------------------------------------

class PodManager:
    def __init__(self, gpu_type: str = "4090", gpu_fallback_types: list[str] | None = None):
        self.gpu_type = gpu_type
        self.gpu_fallback_types = None if gpu_fallback_types is None else list(gpu_fallback_types)
        self.client: RunPodClient | None = None
        self.pod_id: str | None = None
        self.ssh_host: str = ""
        self.ssh_port: int = 0
        self._bootstrapped: bool = False
        self.active_gpu_type: str = resolve_gpu_type(gpu_type)

    def ensure_ready(self) -> tuple[str, int]:
        if self.ssh_host and self.ssh_port:
            try:
                _ssh_run(self.ssh_host, self.ssh_port, "echo ok")
                return self.ssh_host, self.ssh_port
            except Exception:
                log.warning("pod connection lost, reprovisioning")
                self._bootstrapped = False

        if self.client is None:
            self.client = RunPodClient()

        if self.pod_id:
            try:
                self.client.terminate_pod(self.pod_id)
            except Exception:
                pass

        candidates = build_gpu_fallback_types(self.gpu_type, self.gpu_fallback_types)
        last_error: Exception | None = None
        for idx, candidate in enumerate(candidates):
            name = f"binance-forever-{time.strftime('%H%M%S')}"
            log.info("provisioning pod: %s (%s)", name, candidate)
            try:
                config = PodConfig(name=name, gpu_type=candidate)
                pod = self.client.create_pod(config)
                self.pod_id = pod.id

                pod = self.client.wait_for_pod(pod.id)
                self.ssh_host = pod.ssh_host
                self.ssh_port = pod.ssh_port
                self.active_gpu_type = pod.gpu_type or candidate
                if idx > 0:
                    log.warning(
                        "requested %s unavailable, fell back to %s",
                        resolve_gpu_type(self.gpu_type),
                        self.active_gpu_type,
                    )
                log.info("pod ready: %s:%d", self.ssh_host, self.ssh_port)
                return self.ssh_host, self.ssh_port
            except Exception as e:
                last_error = e
                if self.pod_id:
                    try:
                        self.client.terminate_pod(self.pod_id)
                    except Exception:
                        pass
                    self.pod_id = None
                if is_capacity_error(e) and idx < len(candidates) - 1:
                    log.warning("pod provisioning failed for %s: %s", candidate, e)
                    continue
                raise

        if last_error is not None:
            raise last_error
        raise RuntimeError("RunPod provisioning failed without an exception")

    def bootstrap(self, remote_dir: str = "/workspace/stock-prediction") -> None:
        if self._bootstrapped:
            return
        log.info("syncing code to pod")
        _rsync_to_pod(self.ssh_host, self.ssh_port, REPO, remote_dir)
        log.info("bootstrapping pod")
        _bootstrap_pod(self.ssh_host, self.ssh_port, remote_dir)
        self._bootstrapped = True

    def upload_data(self, track: DataTrack, remote_dir: str = "/workspace/stock-prediction") -> None:
        for data_rel in (track.train, track.val):
            local = REPO / data_rel
            if local.exists():
                log.info("uploading %s", data_rel)
                _rsync_data(self.ssh_host, self.ssh_port, local, remote_dir)
            else:
                log.warning("data file not found: %s", local)

    def terminate(self) -> None:
        if self.client and self.pod_id:
            try:
                self.client.terminate_pod(self.pod_id)
                log.info("pod terminated: %s", self.pod_id)
            except Exception as e:
                log.warning("failed to terminate pod: %s", e)
            self.pod_id = None
            self.ssh_host = ""
            self.ssh_port = 0
            self._bootstrapped = False

    def hourly_rate(self) -> float:
        resolved = self.active_gpu_type or resolve_gpu_type(self.gpu_type)
        return HOURLY_RATES.get(resolved, 0.0)

# ---------------------------------------------------------------------------
# Main forever loop
# ---------------------------------------------------------------------------

def run_forever(
    *,
    gpu_type: str = "4090",
    gpu_fallback_types: list[str] | None = None,
    time_budget: int = 300,
    max_trials_per_round: int = 15,
    tracks: list[DataTrack] | None = None,
    local: bool = False,
    once: bool = False,
    dry_run: bool = False,
    descriptions: str = "",
    remote_dir: str = "/workspace/stock-prediction",
    r2_sync: bool = True,
    prod_launch_script: str = str(DEFAULT_LAUNCH_SCRIPT),
    prod_symbols: str = "",
    prod_symbol_sets: list[str] | None = None,
    prod_symbol_subset_sizes: list[int] | None = None,
    prod_leverage: float | None = None,
    prod_leverage_options: list[float] | None = None,
    prod_config_jobs: int = 1,
    prod_config_max_variants: int = DEFAULT_PROD_CONFIG_MAX_VARIANTS,
    prod_config_max_checkpoint_evals: int = DEFAULT_PROD_CONFIG_MAX_CHECKPOINT_EVALS,
    prod_config_variant_offset: int = 0,
    auto_deploy: bool = False,
    deploy_dry_run: bool = False,
    deploy_wait_for_live_cycle_seconds: float | None = None,
    deploy_min_healthy_live_cycles: int | None = None,
    prod_eval_top_k: int = DEFAULT_PROD_EVAL_TOP_K,
):
    if tracks is None:
        tracks = [CRYPTO_DAILY, CRYPTO_HOURLY]
    normalized_prod_symbol_sets = [str(symbol_set).strip() for symbol_set in (prod_symbol_sets or []) if str(symbol_set).strip()]
    normalized_prod_symbol_subset_sizes = [int(subset_size) for subset_size in (prod_symbol_subset_sizes or [])]
    if any(subset_size < 1 for subset_size in normalized_prod_symbol_subset_sizes):
        raise ValueError("prod symbol subset sizes must be at least 1")
    normalized_prod_leverage_options = [float(leverage) for leverage in (prod_leverage_options or [])]
    if prod_config_variant_offset < 0:
        raise ValueError("prod config variant offset must be >= 0")
    grid_eval_symbols = prod_symbols.strip()
    if not grid_eval_symbols and normalized_prod_symbol_sets:
        grid_eval_symbols = _merge_prod_symbol_sets(normalized_prod_symbol_sets)
    grid_eval_leverage = prod_leverage
    if grid_eval_leverage is None and normalized_prod_leverage_options:
        grid_eval_leverage = max(normalized_prod_leverage_options)
    eval_constraints = resolve_launch_eval_constraints(
        prod_launch_script=prod_launch_script,
        eval_max_leverage=grid_eval_leverage,
        eval_tradable_symbols=grid_eval_symbols,
    )

    log_file = REPO / "analysis" / "binance_autoresearch_forever_log.jsonl"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    pod_mgr: PodManager | None = None
    if not local:
        pod_mgr = PodManager(gpu_type=gpu_type, gpu_fallback_types=gpu_fallback_types)

    round_num = 0
    track_idx = 0
    _shutdown = False

    def _handle_signal(signum, frame):
        nonlocal _shutdown
        log.info("shutdown signal received, finishing current round")
        _shutdown = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    if dry_run:
        _print_plan(
            gpu_type,
            gpu_fallback_types,
            time_budget,
            max_trials_per_round,
            tracks,
            local,
            prod_launch_script,
            eval_constraints.max_leverage,
            eval_constraints.tradable_symbols_csv,
            eval_constraints.disable_shorts,
            normalized_prod_symbol_sets,
            normalized_prod_symbol_subset_sizes,
            normalized_prod_leverage_options,
            prod_config_jobs,
            prod_config_max_variants,
            prod_config_max_checkpoint_evals,
            prod_config_variant_offset,
            auto_deploy,
            deploy_dry_run,
            deploy_wait_for_live_cycle_seconds,
            deploy_min_healthy_live_cycles,
            prod_eval_top_k,
        )
        return

    try:
        while not _shutdown:
            round_num += 1
            track = tracks[track_idx % len(tracks)]
            track_idx += 1
            run_id = f"binance_forever_{track.name}_{time.strftime('%Y%m%d_%H%M%S')}"

            log.info("=== round %d: %s (%s) ===", round_num, track.name, track.description)

            t0 = time.time()
            ssh_host, ssh_port = "", 0
            baseline = _baseline_from_launch(prod_launch_script)

            try:
                if pod_mgr:
                    ssh_host, ssh_port = pod_mgr.ensure_ready()
                    pod_mgr.bootstrap(remote_dir)
                    pod_mgr.upload_data(track, remote_dir)

                round_descriptions = descriptions or track.focused_descriptions
                leaderboard, checkpoint_dir = run_autoresearch_batch(
                    track,
                    time_budget=time_budget,
                    max_trials=max_trials_per_round,
                    run_id=run_id,
                    gpu_type=pod_mgr.active_gpu_type if pod_mgr else "local",
                    ssh_host=ssh_host,
                    ssh_port=ssh_port,
                    remote_dir=remote_dir,
                    local=local,
                    descriptions=round_descriptions,
                    holdout_max_leverage=eval_constraints.max_leverage,
                    eval_tradable_symbols=eval_constraints.tradable_symbols_csv,
                    eval_disable_shorts=eval_constraints.disable_shorts,
                )

                training_time = time.time() - t0
                candidate_results = read_top_from_leaderboard(
                    leaderboard,
                    checkpoint_dir,
                    limit=prod_eval_top_k,
                )
                for candidate_result in candidate_results:
                    candidate_result.track = track.name
                    candidate_result.gpu_type = pod_mgr.active_gpu_type if pod_mgr else "local"
                    candidate_result.training_time_s = training_time

                result = candidate_results[0]
                candidate_checkpoints = [
                    candidate_result.best_checkpoint
                    for candidate_result in candidate_results
                    if candidate_result.best_checkpoint
                ]
                if candidate_checkpoints:
                    try:
                        if normalized_prod_symbol_sets or normalized_prod_symbol_subset_sizes or normalized_prod_leverage_options:
                            grid_variant_offset = prod_config_variant_offset + (round_num - 1)
                            launch_cfg = parse_launch_script(prod_launch_script, require_rl_checkpoint=False)
                            planned_variants = build_config_variants(
                                launch_script=prod_launch_script,
                                symbol_set_specs=normalized_prod_symbol_sets,
                                symbol_subset_sizes=normalized_prod_symbol_subset_sizes,
                                leverage_options=normalized_prod_leverage_options,
                                output_root=REPO / "analysis" / "_prod_config_budget_probe",
                                include_launch_variant=True,
                            )
                            planned_variants = limit_config_variants(
                                planned_variants,
                                launch_config=launch_cfg,
                                max_variants=None if prod_config_max_variants == 0 else prod_config_max_variants,
                                variant_offset=grid_variant_offset,
                            )
                            budgeted_max_candidates = max_candidate_checkpoints_for_budget(
                                launch_cfg,
                                variant_count=len(planned_variants),
                                max_checkpoint_config_evals=prod_config_max_checkpoint_evals,
                            )
                            if budgeted_max_candidates is not None:
                                if budgeted_max_candidates < 1:
                                    raise RuntimeError(
                                        "production config search budget is too small to compare even one candidate across the requested configs"
                                    )
                                if len(candidate_checkpoints) > budgeted_max_candidates:
                                    candidate_rotation_offset = round_num - 1
                                    log.warning(
                                        "trimming production config search candidates from %d to %d to respect checkpoint-config eval budget (leader preserved, rotation offset=%d)",
                                        len(candidate_checkpoints),
                                        budgeted_max_candidates,
                                        candidate_rotation_offset,
                                    )
                                    candidate_results = limit_ranked_candidates(
                                        candidate_results,
                                        max_candidates=budgeted_max_candidates,
                                        offset=candidate_rotation_offset,
                                    )
                                    candidate_checkpoints = [
                                        candidate_result.best_checkpoint
                                        for candidate_result in candidate_results
                                        if candidate_result.best_checkpoint
                                    ]
                                    result = candidate_results[0]
                            baseline, config_selection = evaluate_candidates_across_prod_configs(
                                candidate_checkpoints,
                                run_id=run_id,
                                launch_script=prod_launch_script,
                                symbol_sets=normalized_prod_symbol_sets,
                                symbol_subset_sizes=normalized_prod_symbol_subset_sizes,
                                leverage_options=normalized_prod_leverage_options,
                                jobs=prod_config_jobs,
                                max_variants=prod_config_max_variants,
                                max_checkpoint_config_evals=prod_config_max_checkpoint_evals,
                                variant_offset=grid_variant_offset,
                            )
                            selected_checkpoint = config_selection.checkpoint
                            selected_result = next(
                                (
                                    candidate_result
                                    for candidate_result in candidate_results
                                    if candidate_result.best_checkpoint
                                    and str(Path(candidate_result.best_checkpoint).expanduser().resolve(strict=False)) == selected_checkpoint
                                ),
                                None,
                            )
                            if selected_result is None:
                                if baseline.checkpoint and selected_checkpoint == str(Path(baseline.checkpoint).expanduser().resolve(strict=False)):
                                    selected_result = TrialResult(
                                        description="current live checkpoint under searched production config",
                                        track=track.name,
                                        val_return=config_selection.val_return,
                                        val_sortino=config_selection.val_sortino,
                                        best_checkpoint=baseline.checkpoint,
                                        gpu_type=pod_mgr.active_gpu_type if pod_mgr else "local",
                                        training_time_s=training_time,
                                    )
                                else:
                                    raise RuntimeError(
                                        "production config grid selected a checkpoint that was not present in the candidate set"
                                    )
                            selected_result.production_manifest_path = config_selection.manifest_path
                            selected_result.production_metric_name = config_selection.gate_result.metric_name
                            selected_result.production_symbols_override = config_selection.symbols_override
                            selected_result.production_leverage_override = config_selection.leverage_override
                            selected_result.production_metric_current = config_selection.gate_result.current_metric
                            if (
                                baseline.checkpoint
                                and selected_result.best_checkpoint
                                and str(Path(selected_result.best_checkpoint).expanduser().resolve(strict=False))
                                == str(Path(baseline.checkpoint).expanduser().resolve(strict=False))
                                and _trial_result_changes_production_config(selected_result, baseline)
                                and baseline.production_metric_value is not None
                            ):
                                selected_result.production_metric_current = baseline.production_metric_value
                            selected_result.production_metric_candidate = config_selection.gate_result.candidate_metric
                            selected_result.production_gate_reason = config_selection.gate_result.reason
                            selected_result.production_gate_allowed = config_selection.gate_result.allowed
                            result = selected_result
                        else:
                            prod_eval_kwargs: dict[str, object] = {}
                            if prod_symbols.strip():
                                prod_eval_kwargs["symbols_override"] = prod_symbols
                            if prod_leverage is not None:
                                prod_eval_kwargs["leverage_override"] = prod_leverage
                            baseline, gate_results = evaluate_candidates_vs_production(
                                candidate_checkpoints,
                                run_id=run_id,
                                launch_script=prod_launch_script,
                                **prod_eval_kwargs,
                            )
                            for candidate_result in candidate_results:
                                if not candidate_result.best_checkpoint:
                                    continue
                                gate_result = gate_results.get(
                                    str(Path(candidate_result.best_checkpoint).expanduser().resolve(strict=False))
                                )
                                if gate_result is None:
                                    continue
                                candidate_result.production_manifest_path = gate_result.manifest_path
                                candidate_result.production_metric_name = gate_result.metric_name
                                candidate_result.production_metric_current = gate_result.current_metric
                                candidate_result.production_metric_candidate = gate_result.candidate_metric
                                candidate_result.production_gate_reason = gate_result.reason
                                candidate_result.production_gate_allowed = gate_result.allowed
                            result = select_result_for_production(candidate_results, baseline)
                            if prod_symbols.strip():
                                result.production_symbols_override = prod_symbols.strip()
                            if prod_leverage is not None:
                                result.production_leverage_override = float(prod_leverage)
                    except Exception as e:
                        comparison_error = str(e)
                        for candidate_result in candidate_results:
                            candidate_result.production_gate_reason = comparison_error
                            candidate_result.production_gate_allowed = False
                            candidate_result.error = (
                                comparison_error
                                if not candidate_result.error
                                else f"{candidate_result.error}; {comparison_error}"
                            )
                        result = candidate_results[0]
                else:
                    result = select_result_for_production(candidate_results, baseline)

                improved = result.beats_baseline(baseline)
                if improved and auto_deploy:
                    if result.best_checkpoint and result.production_manifest_path:
                        result.deploy_attempted = True
                        result.deploy_dry_run = deploy_dry_run
                        deploy_kwargs: dict[str, object] = {}
                        if result.production_symbols_override.strip():
                            deploy_kwargs["symbols_override"] = result.production_symbols_override.strip()
                        elif prod_symbols.strip():
                            deploy_kwargs["symbols_override"] = prod_symbols
                        selected_leverage_override = (
                            result.production_leverage_override
                            if result.production_leverage_override is not None
                            else prod_leverage
                        )
                        if selected_leverage_override is not None:
                            deploy_kwargs["leverage_override"] = selected_leverage_override
                        if deploy_wait_for_live_cycle_seconds is not None:
                            deploy_kwargs["wait_for_live_cycle_seconds"] = deploy_wait_for_live_cycle_seconds
                        if deploy_min_healthy_live_cycles is not None:
                            deploy_kwargs["min_healthy_live_cycles"] = deploy_min_healthy_live_cycles
                        deploy_ok, deploy_message = deploy_candidate_checkpoint(
                            result.best_checkpoint,
                            manifest_path=result.production_manifest_path,
                            launch_script=prod_launch_script,
                            dry_run=deploy_dry_run,
                            **deploy_kwargs,
                        )
                        result.deploy_succeeded = deploy_ok
                        result.deploy_message = deploy_message
                    else:
                        result.deploy_attempted = True
                        result.deploy_dry_run = deploy_dry_run
                        result.deploy_succeeded = False
                        result.deploy_message = "missing checkpoint or production manifest for deploy"
                        log.warning("skipping auto-deploy: %s", result.deploy_message)
                round_succeeded = improved and (not auto_deploy or result.deploy_succeeded is True)

                # log result
                log_entry = {
                    "round": round_num,
                    "run_id": run_id,
                    "track": track.name,
                    "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
                    "description": result.description,
                    "rank_metric": result.rank_metric,
                    "rank_score": result.rank_score,
                    "val_return": result.val_return,
                    "val_sortino": result.val_sortino,
                    "win_rate": result.win_rate,
                    "holdout_robust_score": result.holdout_robust_score,
                    "combined_score": result.combined_score(),
                    "baseline_combined": baseline.combined_score(),
                    "baseline_checkpoint": baseline.checkpoint,
                    "baseline_source": baseline.source,
                    "production_metric_name": result.production_metric_name or baseline.production_metric_name,
                    "production_metric_live": baseline.production_metric_value,
                    "production_metric_candidate": result.production_metric_candidate,
                    "production_manifest_path": result.production_manifest_path or baseline.manifest_path,
                    "production_gate_reason": result.production_gate_reason,
                    "production_symbols_override": result.production_symbols_override or None,
                    "production_leverage_override": result.production_leverage_override,
                    "beats_baseline": improved,
                    "round_succeeded": round_succeeded,
                    "deploy_attempted": result.deploy_attempted,
                    "deploy_succeeded": result.deploy_succeeded,
                    "deploy_dry_run": result.deploy_dry_run,
                    "deploy_message": result.deploy_message,
                    "training_time_s": training_time,
                    "checkpoint": result.best_checkpoint,
                    "error": result.error,
                }
                with open(log_file, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

                if round_succeeded:
                    if result.production_metric_name:
                        log.info(
                            "NEW BEST: %s %s=%+.4f > live=%+.4f",
                            result.description,
                            result.production_metric_name,
                            result.production_metric_candidate or 0.0,
                            baseline.production_metric_value or 0.0,
                        )
                    else:
                        log.info(
                            "NEW BEST: %s combined=%.4f > baseline=%.4f",
                            result.description,
                            result.combined_score(),
                            baseline.combined_score(),
                        )
                    if result.deploy_attempted:
                        if result.deploy_succeeded:
                            log.info("auto-deploy %s: %s", "dry-run" if result.deploy_dry_run else "succeeded", result.deploy_message)
                        else:
                            log.warning("auto-deploy failed: %s", result.deploy_message)
                    write_success_progress(result, baseline, round_num)
                else:
                    if improved and auto_deploy:
                        log.warning(
                            "live improvement found but auto-deploy failed: %s",
                            result.deploy_message or "unknown deploy failure",
                        )
                    elif result.production_metric_name:
                        log.info(
                            "no live improvement: %s %s=%+.4f <= live=%+.4f",
                            result.description,
                            result.production_metric_name,
                            result.production_metric_candidate or 0.0,
                            baseline.production_metric_value or 0.0,
                        )
                    else:
                        log.info(
                            "no improvement: %s combined=%.4f <= baseline=%.4f",
                            result.description,
                            result.combined_score(),
                            baseline.combined_score(),
                        )
                    append_failure_progress(result, baseline, round_num)

                # R2 sync
                if r2_sync and os.environ.get("R2_ENDPOINT"):
                    sync_to_r2(checkpoint_dir, log_file, run_id)

                # top-K checkpoint management
                if result.best_checkpoint and result.val_sortino is not None:
                    try:
                        r2_prefix = f"binance_autoresearch/{run_id}" if r2_sync else None
                        mgr = TopKCheckpointManager(
                            checkpoint_dir, max_keep=5, mode="max", r2_prefix=r2_prefix,
                        )
                        mgr.register(result.best_checkpoint, result.val_sortino)
                    except Exception as e:
                        log.warning("checkpoint manager error: %s", e)

            except Exception as e:
                log.error("round %d failed: %s", round_num, e)
                traceback.print_exc()
                result = TrialResult(
                    description=f"round_{round_num}_error",
                    track=track.name,
                    error=str(e),
                    training_time_s=time.time() - t0,
                    gpu_type=pod_mgr.active_gpu_type if pod_mgr else "local",
                )
                append_failure_progress(result, baseline, round_num)

            if once:
                log.info("--once mode, exiting after round %d", round_num)
                break

            if _shutdown:
                break

            # brief pause between rounds
            log.info("round %d complete, next round in 10s", round_num)
            time.sleep(10)

    finally:
        if pod_mgr:
            pod_mgr.terminate()
        log.info("forever loop stopped after %d rounds", round_num)


def _print_plan(
    gpu_type: str,
    gpu_fallback_types: list[str] | None,
    time_budget: int,
    max_trials: int,
    tracks: list[DataTrack],
    local: bool,
    prod_launch_script: str,
    holdout_max_leverage: float,
    eval_tradable_symbols: str,
    eval_disable_shorts: bool,
    prod_symbol_sets: list[str],
    prod_symbol_subset_sizes: list[int],
    prod_leverage_options: list[float],
    prod_config_jobs: int,
    prod_config_max_variants: int,
    prod_config_max_checkpoint_evals: int,
    prod_config_variant_offset: int,
    auto_deploy: bool,
    deploy_dry_run: bool,
    deploy_wait_for_live_cycle_seconds: float | None,
    deploy_min_healthy_live_cycles: int | None,
    prod_eval_top_k: int,
):
    resolved = resolve_gpu_type(gpu_type)
    rate = HOURLY_RATES.get(resolved, 0.0)
    fallback_chain = build_gpu_fallback_types(gpu_type, gpu_fallback_types)
    round_time_min = (max_trials * time_budget + 1800) / 60  # +30min overhead

    print("Binance Autoresearch Forever -- Dry Run Plan")
    print(f"  Mode: {'LOCAL' if local else 'RunPod'}")
    print(f"  GPU: {resolved} @ ${rate:.2f}/hr")
    if not local:
        print(f"  GPU fallback order: {', '.join(fallback_chain)}")
    print(f"  Time budget: {time_budget}s per trial")
    print(f"  Max trials per round: {max_trials}")
    print(f"  Est. round time: {round_time_min:.0f} min")
    print(f"  Est. cost per round: ${rate * round_time_min / 60:.2f}")
    print(f"  Live launch: {prod_launch_script}")
    print(f"  Eval leverage: {holdout_max_leverage}")
    print(f"  Eval symbols: {eval_tradable_symbols or '(full track universe)'}")
    print(f"  Eval shorts: {'disabled' if eval_disable_shorts else 'enabled'}")
    if prod_symbol_sets or prod_symbol_subset_sizes or prod_leverage_options:
        print(f"  Prod config search: {len(prod_symbol_sets) or 1} symbol-set option(s) x {len(prod_leverage_options) or 1} leverage option(s)")
        if prod_symbol_subset_sizes:
            print(f"  Prod subset sizes: {', '.join(str(size) for size in prod_symbol_subset_sizes)}")
        print(f"  Prod config jobs: {prod_config_jobs}")
        print(f"  Prod config max variants: {prod_config_max_variants}")
        print(f"  Prod config max checkpoint-config evals: {prod_config_max_checkpoint_evals}")
        print(f"  Prod config variant offset base: {prod_config_variant_offset}")
    print(f"  Prod eval top-K: {prod_eval_top_k}")
    print(f"  Auto deploy: {'enabled' if auto_deploy else 'disabled'}")
    if auto_deploy:
        print(f"  Deploy mode: {'dry-run' if deploy_dry_run else 'live'}")
        if deploy_wait_for_live_cycle_seconds is not None:
            print(f"  Deploy live-cycle wait: {deploy_wait_for_live_cycle_seconds:.0f}s")
        if deploy_min_healthy_live_cycles is not None:
            print(f"  Deploy min healthy cycles: {deploy_min_healthy_live_cycles}")
    print()
    print("  Tracks (alternating):")
    for t in tracks:
        exists_train = "OK" if (REPO / t.train).exists() else "MISSING"
        exists_val = "OK" if (REPO / t.val).exists() else "MISSING"
        print(f"    {t.name}: {t.description}")
        print(f"      train: {t.train} [{exists_train}]")
        print(f"      val:   {t.val} [{exists_val}]")
        print(f"      periods/yr: {t.periods_per_year}, max_steps: {t.max_steps}, fee: {t.fee_rate}")
    print()
    print("  Workflow per round:")
    print("    1. Provision/reuse RunPod pod" if not local else "    1. Use local GPU")
    print("    2. rsync code + data")
    print("    3. autoresearch_rl (N trials with poly early stopping)")
    print("    4. Download leaderboard + checkpoints")
    if prod_symbol_sets or prod_symbol_subset_sizes or prod_leverage_options:
        print(f"    5. Run live-like prod config search for the top {prod_eval_top_k} checkpoint(s) and compare vs current launch")
    else:
        print(f"    5. Run live-like prod eval for the top {prod_eval_top_k} checkpoint(s) and compare vs current launch")
    print(f"    6. {'Deploy gated winner to production' if auto_deploy else 'Write to binanceprogress{N}.md (success) or binanceprogress_failed.md'}")
    if auto_deploy:
        print("    7. Write deployment status to progress/log files")
        print("    8. Sync top-K checkpoints + logs to R2")
        print("    9. Loop forever (alternating tracks)")
    else:
        print("    7. Sync top-K checkpoints + logs to R2")
        print("    8. Loop forever (alternating tracks)")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Binance crypto autoresearch forever loop")
    p.add_argument("--gpu-type", default="4090", choices=["", *sorted(GPU_ALIASES.keys())],
                   help="RunPod GPU type (default: 4090)")
    p.add_argument(
        "--gpu-fallback-types",
        default="",
        help=(
            "comma-separated RunPod fallback GPU aliases "
            f"(default: {','.join(DEFAULT_GPU_FALLBACKS)}; use 'none' to disable)"
        ),
    )
    p.add_argument("--time-budget", type=int, default=300, help="seconds per trial")
    p.add_argument("--max-trials", type=int, default=15, help="trials per round")
    p.add_argument("--local", action="store_true", help="run on local GPU")
    p.add_argument("--once", action="store_true", help="run one round then exit")
    p.add_argument("--dry-run", action="store_true", help="print plan and exit")
    p.add_argument("--tracks", default="crypto_daily,crypto_hourly",
                   help="comma-separated track names: crypto_daily, crypto_hourly, mixed_daily, mixed_hourly")
    p.add_argument("--descriptions", default="", help="comma-separated experiment descriptions to run")
    p.add_argument("--no-r2", action="store_true", help="disable R2 sync")
    p.add_argument("--remote-dir", default="/workspace/stock-prediction")
    p.add_argument("--prod-launch-script", default=str(DEFAULT_LAUNCH_SCRIPT))
    p.add_argument("--prod-symbols", default="", help="override target production symbols for eval and auto-deploy")
    p.add_argument("--prod-symbols-set", action="append", default=[], help="repeatable target production symbol set for config search")
    p.add_argument("--prod-symbols-subset-size", action="append", type=int, default=[], help="repeatable exact subset size to search from the live production symbol universe")
    p.add_argument("--prod-leverage", type=float, default=None, help="override target production leverage for eval and auto-deploy")
    p.add_argument("--prod-leverage-option", action="append", type=float, default=[], help="repeatable target production leverage for config search")
    p.add_argument("--prod-config-jobs", type=int, default=1, help="parallel jobs for multi-config production comparison")
    p.add_argument("--prod-config-max-variants", type=int, default=DEFAULT_PROD_CONFIG_MAX_VARIANTS, help="maximum number of production config variants to search; use 0 to disable")
    p.add_argument("--prod-config-max-checkpoint-evals", type=int, default=DEFAULT_PROD_CONFIG_MAX_CHECKPOINT_EVALS, help="maximum total checkpoint-config evaluations in production config search; use 0 to disable")
    p.add_argument("--prod-config-variant-offset", type=int, default=0, help="base offset into the production config grid; the forever loop adds the round number to rotate capped searches")
    p.add_argument("--auto-deploy", action="store_true", help="deploy production-approved winners automatically")
    p.add_argument("--deploy-dry-run", action="store_true", help="with --auto-deploy, validate deploys without changing launch/supervisor")
    p.add_argument("--deploy-wait-for-live-cycle-seconds", type=float, default=None, help="with --auto-deploy, wait up to N seconds for a healthy post-deploy live cycle")
    p.add_argument("--deploy-min-healthy-live-cycles", type=int, default=None, help="with --auto-deploy, require at least N healthy live cycles after deploy")
    p.add_argument("--prod-eval-top-k", type=int, default=DEFAULT_PROD_EVAL_TOP_K, help="number of leaderboard candidates to compare against production per round")
    return p.parse_args()


def main():
    args = parse_args()

    track_map = {t.name: t for t in ALL_TRACKS}
    tracks = []
    for raw_track_name in args.tracks.split(","):
        track_name = raw_track_name.strip()
        if track_name in track_map:
            tracks.append(track_map[track_name])
        else:
            log.error("unknown track: %s (available: %s)", track_name, ", ".join(track_map.keys()))
            sys.exit(1)

    run_forever(
        gpu_type=args.gpu_type,
        gpu_fallback_types=parse_gpu_fallback_types(args.gpu_fallback_types),
        time_budget=args.time_budget,
        max_trials_per_round=args.max_trials,
        tracks=tracks,
        local=args.local,
        once=args.once,
        dry_run=args.dry_run,
        descriptions=args.descriptions,
        remote_dir=args.remote_dir,
        r2_sync=not args.no_r2,
        prod_launch_script=args.prod_launch_script,
        prod_symbols=args.prod_symbols,
        prod_symbol_sets=list(args.prod_symbols_set),
        prod_symbol_subset_sizes=list(args.prod_symbols_subset_size),
        prod_leverage=args.prod_leverage,
        prod_leverage_options=list(args.prod_leverage_option),
        prod_config_jobs=max(1, int(args.prod_config_jobs)),
        prod_config_max_variants=int(args.prod_config_max_variants),
        prod_config_max_checkpoint_evals=int(args.prod_config_max_checkpoint_evals),
        prod_config_variant_offset=int(args.prod_config_variant_offset),
        auto_deploy=args.auto_deploy,
        deploy_dry_run=args.deploy_dry_run,
        deploy_wait_for_live_cycle_seconds=args.deploy_wait_for_live_cycle_seconds,
        deploy_min_healthy_live_cycles=args.deploy_min_healthy_live_cycles,
        prod_eval_top_k=max(1, int(args.prod_eval_top_k)),
    )


if __name__ == "__main__":
    main()
