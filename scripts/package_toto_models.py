#!/usr/bin/env python3
"""
Generate ready-to-sync Toto model artifacts for each dtype and compilation mode.

The script:
  * ensures ModelCacheManager has cached checkpoints for compiled (`toto`) and
    uncompiled (`toto_uncompiled`) namespaces;
  * copies the resulting trees into `data/models/toto/{compiled,uncompiled}/...`;
  * writes a manifest describing available variants;
  * optionally syncs the exported folder to an S3-compatible endpoint (e.g. R2).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import torch

from src.models.model_cache import ModelCacheManager
from src.models.toto_wrapper import TotoPipeline

MODEL_ID = "Datadog/Toto-Open-Base-1.0"
MODEL_DIRNAME = MODEL_ID.replace("/", "-")
REPO_ROOT = Path(__file__).resolve().parents[1]
COMPILED_ROOT = REPO_ROOT / "compiled_models"
EXPORT_ROOT_DEFAULT = REPO_ROOT / "data" / "models" / "toto"
VARIANT_CONFIG = {
    "compiled": ("toto", True),
    "uncompiled": ("toto_uncompiled", False),
}
DTYPE_MAP: Dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "fp16": torch.float16,
}
if hasattr(torch, "bfloat16"):
    DTYPE_MAP["bf16"] = torch.bfloat16


def _ensure_cached_models(namespace: str, *, compile_model: bool, device: str) -> None:
    manager = ModelCacheManager(namespace)
    for token, dtype in DTYPE_MAP.items():
        metadata = manager.load_metadata(MODEL_ID, token)
        needs_refresh = not metadata or bool(metadata.get("compile_model")) != compile_model
        if metadata and not needs_refresh:
            recorded_version = metadata.get("torch_version")
            if recorded_version != torch.__version__:
                needs_refresh = True
        if not needs_refresh:
            continue
        if metadata:
            manager.reset_cache(MODEL_ID, token)
        TotoPipeline.from_pretrained(  # noqa: F841
            model_id=MODEL_ID,
            device_map=device,
            torch_dtype=dtype,
            compile_model=compile_model,
            compile_mode="max-autotune",
            compile_backend="inductor" if compile_model else None,
            warmup_sequence=256 if compile_model else 128,
            cache_policy="prefer",
            force_refresh=True,
            cache_manager=manager,
        ).unload()


def _copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing source cache at {src}")
    shutil.copytree(src, dst, dirs_exist_ok=True)


def _write_manifest(export_root: Path, *, metadata_payload: Dict[str, Dict[str, str]]) -> None:
    manifest_path = export_root / "manifest.json"
    serialisable: Dict[str, object] = {
        "model_id": MODEL_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "variants": {},
    }
    for variant, dtype_map in metadata_payload.items():
        entries: Dict[str, object] = {}
        for dtype_token, metadata_text in dtype_map.items():
            entries[dtype_token] = {
                "metadata": json.loads(metadata_text),
                "relative_path": f"{variant}/{MODEL_DIRNAME}/{dtype_token}",
            }
        serialisable["variants"][variant] = entries
    export_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(serialisable, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sync_to_remote(export_root: Path, bucket: str, endpoint: str, remote_prefix: str) -> None:
    dest_uri = f"s3://{bucket.strip('/')}/{remote_prefix.strip('/')}/"
    cmd = [
        "aws",
        "s3",
        "sync",
        str(export_root),
        dest_uri,
        "--endpoint-url",
        endpoint,
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Package Toto model caches for distribution.")
    parser.add_argument("--dest-root", type=Path, default=EXPORT_ROOT_DEFAULT, help="Export directory for packaged models.")
    parser.add_argument("--device", default="auto", help="Specific device for loading models (default: auto-detect).")
    parser.add_argument("--upload", action="store_true", help="Upload packaged models to the configured R2/S3 endpoint.")
    parser.add_argument("--remote-prefix", default="models/stock/toto", help="Remote key prefix when uploading.")
    parser.add_argument("--bucket", default=os.getenv("R2_BUCKET", "models"), help="Target bucket (defaults to R2_BUCKET env).")
    parser.add_argument("--endpoint", default=os.getenv("R2_ENDPOINT"), help="S3-compatible endpoint URL.")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    for variant, (namespace, compile_flag) in VARIANT_CONFIG.items():
        _ensure_cached_models(namespace, compile_model=compile_flag, device=device)

    export_root = args.dest_root.resolve()
    for variant, (namespace, _) in VARIANT_CONFIG.items():
        src = COMPILED_ROOT / namespace / MODEL_DIRNAME
        dst = export_root / variant / MODEL_DIRNAME
        _copy_tree(src, dst)

    metadata_payload = {}
    for variant, (namespace, _) in VARIANT_CONFIG.items():
        metadata_payload[variant] = {}
        for dtype_token in DTYPE_MAP.keys():
            metadata_path = COMPILED_ROOT / namespace / MODEL_DIRNAME / dtype_token / "metadata.json"
            if metadata_path.exists():
                metadata_payload[variant][dtype_token] = metadata_path.read_text(encoding="utf-8")
    _write_manifest(export_root, metadata_payload=metadata_payload)

    if args.upload:
        if not args.endpoint:
            raise RuntimeError("Endpoint URL required for upload (set --endpoint or R2_ENDPOINT).")
        _sync_to_remote(export_root, args.bucket, args.endpoint, args.remote_prefix)


if __name__ == "__main__":
    main()
