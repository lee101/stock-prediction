"""Bootstrap script for RunPod RL training pods.

Run this on the pod before starting training. It:
1. Downloads training data from R2
2. Optionally downloads a checkpoint to resume from
3. Configures wandb credentials from env vars
4. Registers atexit handler to upload results + top-5 checkpoints to R2
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# R2 client — prefer the canonical src implementation; fall back to inline
# when src is not installed on the pod.
# ---------------------------------------------------------------------------

try:
    from src.r2_client import R2Client, get_r2_client  # type: ignore[import]
except ImportError:
    import boto3  # type: ignore[import]

    class R2Client:  # type: ignore[no-redef]
        """Minimal S3-compatible client for Cloudflare R2."""

        def __init__(
            self,
            endpoint: Optional[str] = None,
            bucket: Optional[str] = None,
            access_key: Optional[str] = None,
            secret_key: Optional[str] = None,
        ) -> None:
            self.endpoint = endpoint or os.environ.get("R2_ENDPOINT", "")
            self.bucket = bucket or os.environ.get("R2_BUCKET", "")
            _access = access_key or os.environ.get("R2_ACCESS_KEY", "")
            _secret = secret_key or os.environ.get("R2_SECRET_KEY", "")
            if not self.endpoint:
                raise ValueError("R2_ENDPOINT is required")
            if not self.bucket:
                raise ValueError("R2_BUCKET is required")
            self._s3 = boto3.client(
                "s3",
                endpoint_url=self.endpoint,
                aws_access_key_id=_access or None,
                aws_secret_access_key=_secret or None,
            )

        def upload_file(self, local_path: str | Path, r2_key: str) -> None:
            self._s3.upload_file(str(local_path), self.bucket, r2_key)

        def download_file(self, r2_key: str, local_path: str | Path) -> None:
            dest = Path(local_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            self._s3.download_file(self.bucket, r2_key, str(dest))

    def get_r2_client() -> R2Client:  # type: ignore[misc]
        return R2Client()


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def setup_wandb_credentials(api_key: str) -> None:
    """Write wandb API key to ~/.netrc for non-interactive auth."""
    netrc_path = Path.home() / ".netrc"
    existing = netrc_path.read_text() if netrc_path.exists() else ""
    if "api.wandb.ai" not in existing:
        with netrc_path.open("a") as fh:
            fh.write(f"machine api.wandb.ai login user password {api_key}\n")
        netrc_path.chmod(0o600)
    os.environ["WANDB_API_KEY"] = api_key
    print(f"wandb credentials written to {netrc_path}")


def download_training_data(
    r2_client: R2Client,
    data_files: list[str],
    local_data_dir: Path,
) -> None:
    """Pull .bin files from R2 training_data/ prefix."""
    local_data_dir.mkdir(parents=True, exist_ok=True)
    for filename in data_files:
        key = f"training_data/{filename}"
        dest = local_data_dir / filename
        if dest.exists():
            print(f"  [skip] {filename} already present at {dest}")
            continue
        print(f"  Downloading {key} -> {dest}")
        r2_client.download_file(key, dest)
        print(f"  Done ({dest.stat().st_size:,} bytes)")


def download_checkpoint_for_resume(
    r2_client: R2Client,
    run_id: str,
    local_ckpt_dir: Path,
) -> Optional[Path]:
    """Download best.pt from R2 rl_checkpoints/{run_id}/ for resume training.

    Returns the local path if the checkpoint was found and downloaded, else None.
    """
    key = f"rl_checkpoints/{run_id}/best.pt"
    dest = local_ckpt_dir / "best.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        print(f"  Downloading resume checkpoint {key} -> {dest}")
        r2_client.download_file(key, dest)
        print(f"  Done ({dest.stat().st_size:,} bytes)")
        return dest
    except Exception as exc:
        print(f"  No resume checkpoint found at {key}: {exc}")
        return None


def _upload_checkpoint_dir(
    r2_client: R2Client,
    run_id: str,
    checkpoint_dir: Path,
) -> None:
    """Upload top-5 checkpoints from checkpoint_dir to R2.

    Reads .topk_manifest.json (list of {path, metric} dicts written by
    TopKCheckpointManager). Falls back to best.pt when no manifest exists.
    """
    r2_prefix = f"rl_checkpoints/{run_id}/{checkpoint_dir.name}"
    manifest_path = checkpoint_dir / ".topk_manifest.json"

    top_entries: list[dict] = []
    if manifest_path.exists():
        try:
            raw = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError):
            raw = []
        if isinstance(raw, list):
            valid = [
                e for e in raw
                if isinstance(e, dict) and e.get("path") and Path(e["path"]).exists()
            ]
            valid.sort(key=lambda e: float(e.get("metric", 0.0) or 0.0), reverse=True)
            top_entries = valid[:5]

    if not top_entries:
        best_pt = checkpoint_dir / "best.pt"
        if best_pt.exists():
            top_entries = [{"path": str(best_pt), "metric": 0.0}]

    for entry in top_entries:
        local_path = Path(entry["path"])
        if not local_path.exists():
            print(f"  [warn] checkpoint missing, skipping: {local_path}")
            continue
        r2_key = f"{r2_prefix}/{local_path.name}"
        print(f"  Uploading {local_path.name} -> {r2_key}")
        r2_client.upload_file(local_path, r2_key)

    if manifest_path.exists():
        manifest_key = f"{r2_prefix}/.topk_manifest.json"
        print(f"  Uploading manifest -> {manifest_key}")
        r2_client.upload_file(manifest_path, manifest_key)


def register_exit_handler(
    r2_client: R2Client,
    run_id: str,
    checkpoint_dirs: list[Path],
    log_files: list[Path],
) -> None:
    """Register atexit to upload results to R2 when training finishes."""

    def _on_exit() -> None:
        print("Uploading results to R2...")

        for log_file in log_files:
            if not log_file.exists():
                print(f"  [warn] log file not found, skipping: {log_file}")
                continue
            r2_key = f"rl_runs/{run_id}/{log_file.name}"
            print(f"  Uploading {log_file.name} -> {r2_key}")
            try:
                r2_client.upload_file(log_file, r2_key)
            except Exception as exc:
                print(f"  [warn] failed to upload {log_file}: {exc}")

        for ckpt_dir in checkpoint_dirs:
            if not ckpt_dir.exists():
                print(f"  [warn] checkpoint dir not found, skipping: {ckpt_dir}")
                continue
            try:
                _upload_checkpoint_dir(r2_client, run_id, ckpt_dir)
            except Exception as exc:
                print(f"  [warn] failed to upload checkpoints from {ckpt_dir}: {exc}")

        results = {
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checkpoint_dirs": [str(d) for d in checkpoint_dirs],
            "log_files": [str(f) for f in log_files],
        }
        results_key = f"rl_runs/{run_id}/results.json"
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as tmp:
                json.dump(results, tmp, indent=2)
                tmp_path = Path(tmp.name)
            print(f"  Uploading results summary -> {results_key}")
            r2_client.upload_file(tmp_path, results_key)
        except Exception as exc:
            print(f"  [warn] failed to upload results.json: {exc}")
        finally:
            tmp_path.unlink(missing_ok=True)

        print("Upload complete.")

    atexit.register(_on_exit)
    print(f"Exit handler registered for run {run_id}")


def _parse_csv_paths(raw: str, base_dir: Optional[Path] = None) -> list[Path]:
    """Split comma-separated path string, resolving relative paths against base_dir."""
    paths: list[Path] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        p = Path(token)
        if base_dir is not None and not p.is_absolute():
            p = base_dir / p
        paths.append(p)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap RunPod RL training pod")
    parser.add_argument("--run-id", required=True, help="Unique identifier for this training run")
    parser.add_argument(
        "--data-files",
        default="",
        help="Comma-separated .bin filenames to download from R2 training_data/",
    )
    parser.add_argument(
        "--checkpoint-run-id",
        default="",
        help="Run ID to resume from (downloads rl_checkpoints/{id}/best.pt)",
    )
    parser.add_argument(
        "--remote-dir",
        default="/workspace/stock-prediction",
        help="Repo root on the pod",
    )
    parser.add_argument(
        "--checkpoint-dirs",
        default="pufferlib_market/checkpoints",
        help="Comma-separated dirs (relative to --remote-dir) to upload on exit",
    )
    parser.add_argument(
        "--log-files",
        default="",
        help="Comma-separated log files (relative to --remote-dir) to upload on exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without executing",
    )
    args = parser.parse_args()

    if args.dry_run:
        print(f"[dry-run] Would download data: {args.data_files}")
        print("[dry-run] Would set up wandb from WANDB_API_KEY env var")
        print(f"[dry-run] Would register exit handler for: {args.checkpoint_dirs}")
        if args.checkpoint_run_id:
            print(f"[dry-run] Would download resume checkpoint from run: {args.checkpoint_run_id}")
        return

    remote_dir = Path(args.remote_dir)
    local_data_dir = remote_dir / "pufferlib_market" / "data"

    # 1. Build R2 client from env vars
    r2_client = get_r2_client()
    print(f"R2 client ready (bucket={r2_client.bucket})")

    # 2. Download training data
    data_files = [f.strip() for f in args.data_files.split(",") if f.strip()]
    if data_files:
        print(f"Downloading {len(data_files)} data file(s) to {local_data_dir} ...")
        download_training_data(r2_client, data_files, local_data_dir)
    else:
        print("No data files specified; skipping download.")

    # 3. Optionally download resume checkpoint
    if args.checkpoint_run_id:
        resume_ckpt_dir = remote_dir / "pufferlib_market" / "checkpoints" / args.checkpoint_run_id
        resume_path = download_checkpoint_for_resume(r2_client, args.checkpoint_run_id, resume_ckpt_dir)
        if resume_path:
            print(f"Resume checkpoint available at: {resume_path}")

    # 4. Configure wandb if key set
    wandb_key = os.environ.get("WANDB_API_KEY", "")
    if wandb_key:
        setup_wandb_credentials(wandb_key)
    else:
        print("WANDB_API_KEY not set; skipping wandb credential setup.")

    # 5. Register exit handler
    ckpt_dirs = _parse_csv_paths(args.checkpoint_dirs, base_dir=remote_dir)
    log_files = _parse_csv_paths(args.log_files, base_dir=remote_dir)
    register_exit_handler(r2_client, args.run_id, ckpt_dirs, log_files)

    # 6. Summary
    print()
    print("=== Bootstrap complete ===")
    print(f"  run_id:          {args.run_id}")
    print(f"  remote_dir:      {remote_dir}")
    print(f"  data_files:      {data_files or '(none)'}")
    print(f"  checkpoint_dirs: {[str(d) for d in ckpt_dirs]}")
    print(f"  log_files:       {[str(f) for f in log_files]}")
    print(f"  wandb:           {'configured' if wandb_key else 'not configured'}")
    print()
    print("Training environment is ready. Start training now.")


if __name__ == "__main__":
    main()
