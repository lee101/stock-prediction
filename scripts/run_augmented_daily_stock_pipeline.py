#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import struct
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pufferlib_market.export_data_daily import MAGIC, PRICE_FEATURES, export_binary as export_daily_binary
from pufferlib_market.hourly_replay import MktdData, read_mktd
from scripts.build_stock_shifted_daily_from_hourly import build_shifted_daily_dataset
from src.remote_training_pipeline import compute_daily_overlap_bounds, compute_hourly_overlap_bounds
from src.stock_symbol_inputs import load_symbols_file, normalize_symbols
from wandboard import WandBoardLogger

SYM_NAME_LEN = 16
HEADER_SIZE = 64


@dataclass(frozen=True)
class DailyTrainValWindow:
    earliest_common: str
    latest_common: str
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    val_days: int
    gap_days: int


def _resolve_symbols(values: Sequence[str] | None, *, symbols_file: str | None, hourly_root: Path) -> list[str]:
    if symbols_file:
        return load_symbols_file(symbols_file)
    raw: list[str] = []
    for value in values or ():
        raw.extend(part.strip() for part in str(value).split(",") if part.strip())
    if raw:
        normalized, _removed, _ignored = normalize_symbols(raw)
        return normalized
    discovered = sorted(path.stem.upper() for path in hourly_root.glob("*.csv"))
    if not discovered:
        raise ValueError(f"No hourly CSVs found under {hourly_root}")
    return discovered


def _to_utc_day(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.floor("D")


def build_daily_train_val_window(
    *,
    symbols: Sequence[str],
    daily_root: Path,
    hourly_root: Path | None,
    train_start: str | None,
    train_end: str | None,
    val_start: str | None,
    val_end: str | None,
    val_days: int,
    gap_days: int,
) -> DailyTrainValWindow:
    earliest_common_raw, latest_common_raw = compute_daily_overlap_bounds(symbols=symbols, data_root=daily_root)
    earliest_common = _to_utc_day(earliest_common_raw)
    latest_common = _to_utc_day(latest_common_raw)
    if hourly_root is not None:
        hourly_earliest_raw, hourly_latest_raw = compute_hourly_overlap_bounds(symbols=symbols, data_root=hourly_root)
        earliest_common = max(earliest_common, _to_utc_day(hourly_earliest_raw))
        latest_common = min(latest_common, _to_utc_day(hourly_latest_raw))

    resolved_val_end = _to_utc_day(val_end) if val_end else latest_common
    resolved_val_start = _to_utc_day(val_start) if val_start else (resolved_val_end - pd.Timedelta(days=int(val_days) - 1))
    resolved_train_start = _to_utc_day(train_start) if train_start else earliest_common
    resolved_train_end = _to_utc_day(train_end) if train_end else (resolved_val_start - pd.Timedelta(days=int(gap_days) + 1))

    if resolved_val_end > latest_common:
        raise ValueError(f"val_end {resolved_val_end.date()} exceeds latest common {latest_common.date()}")
    if resolved_val_start < earliest_common:
        raise ValueError(f"val_start {resolved_val_start.date()} predates earliest common {earliest_common.date()}")
    if resolved_train_start < earliest_common:
        raise ValueError(f"train_start {resolved_train_start.date()} predates earliest common {earliest_common.date()}")
    if resolved_train_end < resolved_train_start:
        raise ValueError(f"train_end {resolved_train_end.date()} is before train_start {resolved_train_start.date()}")
    if resolved_train_end >= resolved_val_start:
        raise ValueError("Training window overlaps validation window")

    return DailyTrainValWindow(
        earliest_common=earliest_common.isoformat(),
        latest_common=latest_common.isoformat(),
        train_start=resolved_train_start.isoformat(),
        train_end=resolved_train_end.isoformat(),
        val_start=resolved_val_start.isoformat(),
        val_end=resolved_val_end.isoformat(),
        val_days=int((resolved_val_end - resolved_val_start).days) + 1,
        gap_days=int(gap_days),
    )


def _generate_wandb_run_id() -> str:
    try:
        import wandb  # type: ignore

        util = getattr(wandb, "util", None)
        if util is not None and hasattr(util, "generate_id"):
            return str(util.generate_id())
    except Exception:
        pass
    return uuid.uuid4().hex[:8]


def write_mktd(path: Path, data: MktdData) -> None:
    num_timesteps, num_symbols, features_per_sym = data.features.shape
    if data.prices.shape != (num_timesteps, num_symbols, PRICE_FEATURES):
        raise ValueError(f"Unexpected price tensor shape {data.prices.shape}")
    if data.tradable is None:
        raise ValueError("Tradable mask is required for MKTD v2+ writes")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        header = struct.pack(
            "<4sIIIII40s",
            MAGIC,
            int(data.version),
            int(num_symbols),
            int(num_timesteps),
            int(features_per_sym),
            int(PRICE_FEATURES),
            b"\x00" * 40,
        )
        handle.write(header)
        for symbol in data.symbols:
            raw = str(symbol).encode("ascii", errors="ignore")[:15]
            handle.write(raw + b"\x00" * (SYM_NAME_LEN - len(raw)))
        handle.write(data.features.astype("float32", copy=False).tobytes(order="C"))
        handle.write(data.prices.astype("float32", copy=False).tobytes(order="C"))
        handle.write(data.tradable.astype("uint8", copy=False).tobytes(order="C"))


def concat_mktd_files(*, input_paths: Sequence[Path], output_path: Path) -> MktdData:
    payloads = [read_mktd(path) for path in input_paths]
    if not payloads:
        raise ValueError("No MKTD files to concatenate")
    first = payloads[0]
    for payload in payloads[1:]:
        if payload.symbols != first.symbols:
            raise ValueError("Cannot concatenate MKTD files with different symbol tables")
        if payload.features.shape[1:] != first.features.shape[1:]:
            raise ValueError("Cannot concatenate MKTD files with different feature widths")
        if payload.prices.shape[1:] != first.prices.shape[1:]:
            raise ValueError("Cannot concatenate MKTD files with different price widths")
        if payload.version != first.version:
            raise ValueError("Cannot concatenate MKTD files with different versions")
    merged = MktdData(
        version=int(first.version),
        symbols=list(first.symbols),
        features=np.concatenate([payload.features for payload in payloads], axis=0).astype("float32", copy=False),
        prices=np.concatenate([payload.prices for payload in payloads], axis=0).astype("float32", copy=False),
        tradable=np.concatenate([payload.tradable for payload in payloads], axis=0).astype("uint8", copy=False),
    )
    write_mktd(output_path, merged)
    return merged


def build_train_command(
    *,
    train_data: Path,
    val_data: Path,
    checkpoint_dir: Path,
    wandb_project: str | None,
    wandb_entity: str | None,
    wandb_group: str | None,
    wandb_run_name: str,
    wandb_run_id: str | None,
    wandb_mode: str,
    extra_args: Sequence[str],
) -> list[str]:
    extra_args_list = [str(arg) for arg in extra_args]

    def _has_flag(flag: str) -> bool:
        return flag in extra_args_list

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "pufferlib_market.train",
        "--data-path",
        str(train_data),
        "--val-data-path",
        str(val_data),
        "--checkpoint-dir",
        str(checkpoint_dir),
    ]
    if not _has_flag("--hidden-size"):
        cmd.extend(["--hidden-size", "1024"])
    if not _has_flag("--trade-penalty"):
        cmd.extend(["--trade-penalty", "0.05"])
    if not _has_flag("--fill-slippage-bps"):
        cmd.extend(["--fill-slippage-bps", "0"])
    if not _has_flag("--weight-decay"):
        cmd.extend(["--weight-decay", "0.0"])
    if not _has_flag("--ent-coef"):
        cmd.extend(["--ent-coef", "0.05"])
    if not _has_flag("--periods-per-year"):
        cmd.extend(["--periods-per-year", "252"])
    if not _has_flag("--max-steps"):
        cmd.extend(["--max-steps", "252"])
    if not _has_flag("--lr-schedule") and not _has_flag("--anneal-lr"):
        cmd.extend(["--lr-schedule", "cosine"])
    if not _has_flag("--obs-norm"):
        cmd.append("--obs-norm")
    if not _has_flag("--use-bf16"):
        cmd.append("--use-bf16")
    if not _has_flag("--cuda-graph-ppo") and not _has_flag("--no-cuda-graph"):
        cmd.append("--cuda-graph-ppo")
    if wandb_project:
        cmd.extend(["--wandb-project", wandb_project, "--wandb-run-name", wandb_run_name, "--wandb-mode", wandb_mode])
        if wandb_entity:
            cmd.extend(["--wandb-entity", wandb_entity])
        if wandb_group:
            cmd.extend(["--wandb-group", wandb_group])
        if wandb_run_id:
            cmd.extend(["--wandb-run-id", wandb_run_id, "--wandb-resume", "allow"])
    cmd.extend(extra_args_list)
    return cmd


def build_eval_command(
    *,
    checkpoint: Path,
    val_data: Path,
    out_dir: Path,
    hourly_data_root: Path | None,
    daily_start_date: str | None,
    extra_args: Sequence[str],
) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        "scripts/eval_100d.py",
        "--checkpoint",
        str(checkpoint),
        "--val-data",
        str(val_data),
        "--out-dir",
        str(out_dir),
        "--fail-fast-max-dd",
        "0.20",
        "--monthly-target",
        "0.27",
    ]
    if hourly_data_root is not None and daily_start_date is not None:
        cmd.extend(
            [
                "--execution-granularity",
                "hourly_intrabar",
                "--hourly-data-root",
                str(hourly_data_root),
                "--daily-start-date",
                str(daily_start_date),
            ]
        )
    cmd.extend(str(arg) for arg in extra_args)
    return cmd


def _run_subprocess(cmd: Sequence[str], *, log_path: Path, allowed_returncodes: Sequence[int] = (0,)) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(shlex.quote(str(part)) for part in cmd) + "\n\n")
        handle.flush()
        proc = subprocess.run(list(cmd), cwd=str(REPO), check=False, stdout=handle, stderr=subprocess.STDOUT, text=True)
    if int(proc.returncode) not in {int(code) for code in allowed_returncodes}:
        raise subprocess.CalledProcessError(proc.returncode, proc.args)
    return int(proc.returncode)


def _load_eval_summary(eval_out_dir: Path, *, checkpoint_path: Path) -> dict[str, Any]:
    payload_path = eval_out_dir / f"{checkpoint_path.stem}_eval100d.json"
    if not payload_path.exists():
        raise FileNotFoundError(f"Missing eval payload: {payload_path}")
    return json.loads(payload_path.read_text(encoding="utf-8"))


def _resolve_eval_checkpoint(checkpoint_dir: Path) -> Path:
    for candidate_name in ("val_best.pt", "best.pt", "final.pt"):
        candidate = checkpoint_dir / candidate_name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir} (expected val_best.pt, best.pt, or final.pt)")


def _log_stage(
    *,
    run_name: str,
    log_dir: Path,
    wandb_project: str | None,
    wandb_entity: str | None,
    wandb_mode: str,
    wandb_run_id: str | None,
    hparams: dict[str, Any],
    metrics: dict[str, Any],
    text_name: str | None = None,
    text_body: str | None = None,
) -> None:
    with WandBoardLogger(
        run_name=run_name,
        run_id=wandb_run_id,
        resume="allow" if wandb_run_id else None,
        project=wandb_project,
        entity=wandb_entity,
        mode=wandb_mode,
        config=hparams,
        log_dir=log_dir,
    ) as logger:
        logger.log_hparams(hparams, metrics, table_name="c_pipeline")
        logger.log(metrics)
        if text_name and text_body:
            logger.log_text(text_name, text_body)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="End-to-end augmented daily stock RL pipeline.")
    parser.add_argument("--run-name", default=None, help="Pipeline run name.")
    parser.add_argument("--symbols", nargs="*", default=None, help="Optional symbol list.")
    parser.add_argument("--symbols-file", default=None, help="Optional newline-delimited symbol file.")
    parser.add_argument("--hourly-root", type=Path, default=Path("trainingdatahourly/stocks"))
    parser.add_argument("--daily-root", type=Path, default=Path("trainingdata"))
    parser.add_argument("--offsets", default="0,1,2,3")
    parser.add_argument("--augment-mode", choices=("combined", "per_offset"), default="combined")
    parser.add_argument("--bars-per-session", type=int, default=None)
    parser.add_argument("--separator-days", type=int, default=14)
    parser.add_argument("--work-root", type=Path, default=Path("analysis/augmented_daily_stock_runs"))
    parser.add_argument("--data-output-root", type=Path, default=Path("pufferlib_market/data/augmented_daily_stock"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("pufferlib_market/checkpoints/augmented_daily_stock"))
    parser.add_argument("--train-start", default=None)
    parser.add_argument("--train-end", default=None)
    parser.add_argument("--val-start", default=None)
    parser.add_argument("--val-end", default=None)
    parser.add_argument("--val-days", type=int, default=180)
    parser.add_argument("--gap-days", type=int, default=5)
    parser.add_argument("--min-train-days", type=int, default=200)
    parser.add_argument("--min-val-days", type=int, default=60)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default="c_augmented_daily_stock")
    parser.add_argument("--wandb-mode", default="online")
    parser.add_argument("--prepare-only", action="store_true", help="Build shifted data and MKTD bins, then stop before training.")
    parser.add_argument("--skip-eval-100d", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args, extra_train_args = parser.parse_known_args(list(argv) if argv is not None else None)

    offsets = sorted({int(part.strip()) for part in str(args.offsets).split(",") if part.strip()})
    if not offsets:
        raise ValueError("At least one offset is required")

    symbols = _resolve_symbols(args.symbols, symbols_file=args.symbols_file, hourly_root=Path(args.hourly_root))
    run_name = args.run_name or f"aug_daily_{int(time.time())}"
    run_root = Path(args.work_root) / run_name
    stage_log_dir = run_root / "wandboard"
    shifted_root = run_root / "shifted_daily_train"
    eval_out_dir = run_root / "eval100d"
    train_log_path = run_root / "train.log"
    eval_log_path = run_root / "eval100d.log"
    manifest_path = run_root / "run_manifest.json"
    data_output_root = Path(args.data_output_root)
    checkpoint_dir = Path(args.checkpoint_root) / run_name
    train_bin = data_output_root / f"{run_name}_train.bin"
    val_bin = data_output_root / f"{run_name}_val.bin"
    wandb_run_id = _generate_wandb_run_id() if args.wandb_project else None

    window = build_daily_train_val_window(
        symbols=symbols,
        daily_root=Path(args.daily_root),
        hourly_root=Path(args.hourly_root),
        train_start=args.train_start,
        train_end=args.train_end,
        val_start=args.val_start,
        val_end=args.val_end,
        val_days=int(args.val_days),
        gap_days=int(args.gap_days),
    )

    manifest: dict[str, Any] = {
        "run_name": run_name,
        "symbols": list(symbols),
        "offsets": offsets,
        "augment_mode": str(args.augment_mode),
        "window": asdict(window),
        "paths": {
            "run_root": str(run_root),
            "shifted_root": str(shifted_root),
            "train_bin": str(train_bin),
            "val_bin": str(val_bin),
            "checkpoint_dir": str(checkpoint_dir),
            "eval_out_dir": str(eval_out_dir),
        },
        "wandb": {
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "group": args.wandb_group,
            "mode": args.wandb_mode,
            "run_id": wandb_run_id,
        },
        "commands": {},
        "status": "starting",
    }
    run_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(f"{json.dumps(manifest, indent=2, sort_keys=True)}\n", encoding="utf-8")
    train_cmd = build_train_command(
        train_data=train_bin,
        val_data=val_bin,
        checkpoint_dir=checkpoint_dir,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        wandb_run_name=run_name,
        wandb_run_id=wandb_run_id,
        wandb_mode=args.wandb_mode,
        extra_args=extra_train_args,
    )
    manifest["commands"]["train"] = train_cmd

    if args.dry_run:
        manifest["status"] = "dry_run"
        manifest["shift_manifest"] = {"status": "skipped_dry_run"}
        manifest_path.write_text(f"{json.dumps(manifest, indent=2, sort_keys=True)}\n", encoding="utf-8")
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    shift_manifest = build_shifted_daily_dataset(
        hourly_root=Path(args.hourly_root),
        output_root=shifted_root,
        symbols=symbols,
        offsets=offsets,
        mode=str(args.augment_mode),
        bars_per_session=args.bars_per_session,
        separator_days=int(args.separator_days),
        start_date=window.train_start,
        end_date=window.train_end,
        force=True,
    )

    _log_stage(
        run_name=run_name,
        log_dir=stage_log_dir,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
        wandb_run_id=wandb_run_id,
        hparams={
            "symbols": ",".join(symbols),
            "offsets": ",".join(str(v) for v in offsets),
            "augment_mode": args.augment_mode,
            "val_days": int(args.val_days),
        },
        metrics={
            "build/symbol_count": len(symbols),
            "build/offset_count": len(offsets),
            "build/train_calendar_days": int((_to_utc_day(window.train_end) - _to_utc_day(window.train_start)).days) + 1,
            "build/val_calendar_days": int(window.val_days),
        },
        text_name="build/shift_manifest",
        text_body=json.dumps(shift_manifest, indent=2, sort_keys=True),
    )

    if args.augment_mode == "combined":
        train_row_counts = [
            int((shift_manifest.get("symbols") or {}).get(symbol, {}).get("combined_rows", 0))
            for symbol in symbols
        ]
    else:
        train_row_counts = [
            min(
                int(value)
                for value in ((shift_manifest.get("symbols") or {}).get(symbol, {}).get("offset_rows") or {}).values()
            )
            for symbol in symbols
            if ((shift_manifest.get("symbols") or {}).get(symbol, {}).get("offset_rows") or {})
        ]
    if not train_row_counts:
        raise RuntimeError("Shift manifest did not report any train row counts")
    effective_min_train_days = min(int(args.min_train_days), min(train_row_counts))
    if args.augment_mode == "combined":
        export_daily_binary(
            symbols=symbols,
            data_root=shifted_root,
            output_path=train_bin,
            min_days=effective_min_train_days,
        )
    else:
        per_offset_bins: list[Path] = []
        for offset in offsets:
            offset_dir = shifted_root / f"offset_{int(offset)}"
            if not offset_dir.exists():
                continue
            offset_bin = data_output_root / f"{run_name}_offset_{int(offset)}.bin"
            export_daily_binary(
                symbols=symbols,
                data_root=offset_dir,
                output_path=offset_bin,
                min_days=max(5, effective_min_train_days // max(1, len(offsets))),
            )
            per_offset_bins.append(offset_bin)
        if not per_offset_bins:
            raise RuntimeError("No per-offset bins were generated")
        concat_mktd_files(input_paths=per_offset_bins, output_path=train_bin)

    effective_min_val_days = min(int(args.min_val_days), int(window.val_days))
    export_daily_binary(
        symbols=symbols,
        data_root=Path(args.daily_root),
        output_path=val_bin,
        start_date=window.val_start,
        end_date=window.val_end,
        min_days=effective_min_val_days,
    )

    if args.prepare_only:
        manifest["status"] = "prepared"
        manifest["shift_manifest"] = shift_manifest
        manifest_path.write_text(f"{json.dumps(manifest, indent=2, sort_keys=True)}\n", encoding="utf-8")
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    _run_subprocess(train_cmd, log_path=train_log_path)
    manifest["status"] = "trained"

    if not args.skip_eval_100d:
        checkpoint_path = _resolve_eval_checkpoint(checkpoint_dir)
        eval_cmd = build_eval_command(
            checkpoint=checkpoint_path,
            val_data=val_bin,
            out_dir=eval_out_dir,
            hourly_data_root=Path(args.hourly_root),
            daily_start_date=window.val_start,
            extra_args=(),
        )
        manifest["commands"]["eval_100d"] = eval_cmd
        eval_rc = _run_subprocess(eval_cmd, log_path=eval_log_path, allowed_returncodes=(0, 3))
        eval_payload = _load_eval_summary(eval_out_dir, checkpoint_path=checkpoint_path)
        manifest["eval_100d"] = eval_payload
        manifest["eval_100d_returncode"] = int(eval_rc)
        aggregate = eval_payload.get("aggregate", {})
        _log_stage(
            run_name=run_name,
            log_dir=stage_log_dir,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_mode=args.wandb_mode,
            wandb_run_id=wandb_run_id,
            hparams={"symbols": ",".join(symbols), "offsets": ",".join(str(v) for v in offsets)},
            metrics={
                "eval100d/worst_slip_monthly": float(aggregate.get("worst_slip_monthly", 0.0)),
                "eval100d/slippage_cells": len((aggregate.get("by_slippage") or {})),
            },
            text_name="eval100d/payload",
            text_body=json.dumps(eval_payload, indent=2, sort_keys=True),
        )

    manifest["status"] = "completed"
    manifest["shift_manifest"] = shift_manifest
    manifest_path.write_text(f"{json.dumps(manifest, indent=2, sort_keys=True)}\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
