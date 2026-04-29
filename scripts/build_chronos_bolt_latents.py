#!/usr/bin/env python3
"""Generate Chronos-Bolt encoder latents for the XGB daily universe.

For every (symbol, trading-day D) we feed the previous ``context_length``
log-close values (strictly < D, no lookahead) into the Chronos-Bolt T5
encoder, mean-pool the per-patch embeddings into a single 768-D vector,
PCA-reduce across (symbol, date) to ``--n-components`` dims, and write
the result to a parquet file with one row per (symbol, date).

This is the "produce daily-bar embeddings/latents per symbol"
entrypoint that the XGB pipeline can consume as new features.

Usage::

    .venv/bin/python -m scripts.build_chronos_bolt_latents \
        --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
        --data-root trainingdata \
        --start-date 2019-09-01 \
        --end-date  2026-04-30 \
        --context-length 256 \
        --batch-size 1024 \
        --pool last \
        --n-components 32 \
        --output analysis/foundation_model_features/chronos_bolt_base_latents.parquet

Notes
-----
* ``--pool last`` (default) takes the embedding at the last patch position
  — it carries the most up-to-date temporal context. ``--pool mean``
  averages across all patch positions including the [REG] token.
* ``--start-date`` should be set so that we have at least ``context_length``
  prior trading days available; days with insufficient context are skipped.
* The script saves a JSON sidecar manifest with the requested model
  revision, resolved checkpoint commit hash when available, pooling, PCA
  components/explained-variance, and run timing -- required for
  reproducibility.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
CHRONOS_SRC = REPO / "chronos-forecasting" / "src"
if str(CHRONOS_SRC) not in sys.path:
    sys.path.insert(0, str(CHRONOS_SRC))


logger = logging.getLogger("build_chronos_bolt_latents")


def _rel_to_repo(p: Path) -> str:
    """Best-effort relative path from REPO; falls back to absolute string."""
    try:
        return str(Path(p).resolve().relative_to(REPO))
    except Exception:
        return str(Path(p).resolve())


def _load_symbols(path: Path) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for line in path.read_text().splitlines():
        s = line.strip().upper()
        if not s or s.startswith("#"):
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _load_csv(symbol: str, data_root: Path) -> pd.DataFrame | None:
    """Mirror xgbnew/dataset._load_symbol_csv but only return what we need."""
    for sub in ("", "stocks", "train"):
        p = (data_root / sub / f"{symbol}.csv") if sub else (data_root / f"{symbol}.csv")
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            return None
        df.columns = df.columns.str.strip().str.lower()
        if "close" not in df.columns:
            return None
        ts_col = next(
            (c for c in ("timestamp", "date") if c in df.columns), df.columns[0]
        )
        df["timestamp"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        df = df.drop_duplicates(subset=["timestamp"], keep="last")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"])
        df = df[df["close"] > 0]
        df["date"] = df["timestamp"].dt.date
        df = df.drop_duplicates(subset=["date"], keep="last")
        if len(df) < 50:
            return None
        return df[["date", "close"]].reset_index(drop=True)
    return None


def _build_index_table(
    symbols: list[str],
    data_root: Path,
    *,
    start_date: date,
    end_date: date,
    context_length: int,
) -> tuple[list[tuple[str, date]], dict[str, np.ndarray], dict[str, dict[date, int]]]:
    """For each symbol, collect (sym, day) pairs with at least ``context_length``
    prior closes available.

    Returns the eligible pair list, per-symbol close arrays, and per-symbol
    date-to-row indices derived from the same CSV read. Keeping the date index
    from this pass avoids re-reading large universes before embedding.

    All closes are converted to log-returns inside ``_make_context``; here we
    just keep raw closes for window slicing.
    """
    sym_closes: dict[str, np.ndarray] = {}
    sym_date_indices: dict[str, dict[date, int]] = {}
    pairs: list[tuple[str, date]] = []
    for sym in symbols:
        df = _load_csv(sym, data_root)
        if df is None:
            continue
        closes = df["close"].to_numpy(dtype=np.float64)
        days = df["date"].to_list()
        sym_closes[sym] = closes
        sym_date_indices[sym] = {d: i for i, d in enumerate(days)}
        for i, d in enumerate(days):
            if d < start_date or d > end_date:
                continue
            if i < context_length:
                continue
            pairs.append((sym, d))
    return pairs, sym_closes, sym_date_indices


def _make_context(
    closes: np.ndarray, idx: int, context_length: int, return_log: bool
) -> np.ndarray:
    """Slice context window of ``context_length`` ending at ``idx-1`` (no lookahead)."""
    start = idx - context_length
    win = closes[start:idx]  # length = context_length
    assert len(win) == context_length, (len(win), context_length, idx, len(closes))
    if return_log:
        # Convert to log-return series. log(close[i]/close[i-1]) — drop the
        # first element so length stays the same we keep ``len(win)``.
        # We left-pad with 0.0 so the model still sees ``context_length``.
        logc = np.log(np.maximum(win, 1e-9))
        rets = np.zeros_like(logc)
        rets[1:] = np.diff(logc)
        return rets
    return win


def _embed_batch(
    pipe,
    contexts: np.ndarray,  # shape (B, T)
    pool: str,
) -> np.ndarray:
    """Run pipe.embed on a batch and pool down to (B, D)."""
    ctx_t = torch.from_numpy(contexts).to(dtype=torch.float32)
    with torch.inference_mode():
        emb, _locscale = pipe.embed(ctx_t)  # (B, num_patches+1, d_model)
    emb = emb.float()  # cast bf16 → f32 on cpu
    if pool == "last":
        out = emb[:, -1, :]  # most recent patch
    elif pool == "mean":
        out = emb.mean(dim=1)
    elif pool == "first":
        out = emb[:, 0, :]  # [REG] token
    else:
        raise ValueError(f"unknown pool: {pool}")
    return out.numpy().astype(np.float32)


def _load_chronos_pipeline(
    *,
    model_id: str,
    device: str,
    torch_dtype: str,
    model_revision: str | None,
):
    from chronos import BaseChronosPipeline  # noqa: PLC0415

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    kwargs = {
        "device_map": device,
        "dtype": dtype_map[torch_dtype],
    }
    if model_revision:
        kwargs["revision"] = model_revision
    return BaseChronosPipeline.from_pretrained(model_id, **kwargs)


def _extract_model_commit_hash(pipe) -> str | None:
    """Best-effort Hugging Face commit hash extraction from common config slots."""
    candidates = [
        pipe,
        getattr(pipe, "config", None),
        getattr(pipe, "model", None),
        getattr(getattr(pipe, "model", None), "config", None),
        getattr(pipe, "tokenizer", None),
        getattr(getattr(pipe, "tokenizer", None), "init_kwargs", None),
    ]
    for obj in candidates:
        if obj is None:
            continue
        if isinstance(obj, dict):
            values = (obj.get("_commit_hash"), obj.get("commit_hash"))
        else:
            values = (getattr(obj, "_commit_hash", None), getattr(obj, "commit_hash", None))
        for value in values:
            if value:
                return str(value)
    return None


def _atomic_path_write(path: Path, write_fn) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        delete=False,
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=path.suffix,
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        write_fn(tmp_path)
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    def write(tmp_path: Path) -> None:
        df.to_parquet(tmp_path, index=False, compression="zstd")

    _atomic_path_write(path, write)


def _write_npz_atomic(path: Path, **arrays) -> None:
    def write(tmp_path: Path) -> None:
        with tmp_path.open("wb") as f:
            np.savez_compressed(f, **arrays)

    _atomic_path_write(path, write)


def _write_json_atomic(path: Path, payload: dict) -> None:
    def write(tmp_path: Path) -> None:
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _atomic_path_write(path, write)


def _validate_run_config(args) -> tuple[date, date]:
    try:
        start_date = date.fromisoformat(args.start_date)
        end_date = date.fromisoformat(args.end_date)
    except ValueError as exc:
        raise ValueError("start-date and end-date must be ISO dates") from exc
    if start_date > end_date:
        raise ValueError("start-date must be <= end-date")
    if args.context_length <= 0:
        raise ValueError("context-length must be > 0")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be > 0")
    if args.n_components < 0:
        raise ValueError("n-components must be >= 0")
    if args.limit_symbols < 0:
        raise ValueError("limit-symbols must be >= 0")
    return start_date, end_date


def _validate_pca_components(*, n_components: int, embed_dim: int, n_rows: int) -> None:
    if n_components <= 0:
        return
    if n_components > embed_dim:
        raise ValueError(
            f"n-components must be <= raw embedding dim ({embed_dim}); "
            "use 0 to keep raw embeddings"
        )
    if n_components > n_rows:
        raise ValueError(f"n-components must be <= number of latent rows ({n_rows})")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--symbols-file", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=REPO / "trainingdata")
    p.add_argument("--start-date", type=str, default="2019-09-01")
    p.add_argument("--end-date", type=str, default="2026-12-31")
    p.add_argument("--model-id", default="amazon/chronos-bolt-base")
    p.add_argument(
        "--model-revision",
        default=None,
        help="Optional immutable Hugging Face revision/commit to pin model loading.",
    )
    p.add_argument(
        "--device",
        default="cuda",
        help='"cuda" or "cpu"; chronos-bolt-base ~205MB, runs fine on either.',
    )
    p.add_argument(
        "--torch-dtype",
        default="bfloat16",
        choices=["bfloat16", "float32", "float16"],
    )
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument(
        "--pool",
        default="last",
        choices=["last", "mean", "first"],
        help="Patch-pooling strategy.",
    )
    p.add_argument(
        "--input-mode",
        default="raw_close",
        choices=["raw_close", "log_returns"],
        help="What to feed Chronos. raw_close passes raw price levels; "
        "Chronos-Bolt internally instance-normalises so this is fine.",
    )
    p.add_argument(
        "--n-components",
        type=int,
        default=32,
        help="PCA components after pooling. 0 = keep raw 768-dim (large).",
    )
    p.add_argument(
        "--limit-symbols",
        type=int,
        default=0,
        help="Process only this many symbols (debug / smoke test).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=REPO
        / "analysis/foundation_model_features/chronos_bolt_base_latents.parquet",
    )
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        start_date, end_date = _validate_run_config(args)
    except ValueError as exc:
        print(f"chronos-latents: {exc}", file=sys.stderr)
        return 2

    syms = _load_symbols(args.symbols_file)
    if args.limit_symbols:
        syms = syms[: args.limit_symbols]
    print(f"[chronos-latents] symbols: {len(syms)}", flush=True)

    pairs, sym_closes, sym_dates_idx = _build_index_table(
        syms,
        args.data_root,
        start_date=start_date,
        end_date=end_date,
        context_length=args.context_length,
    )
    print(
        f"[chronos-latents] (sym,day) pairs to embed: {len(pairs):,} | "
        f"covering {len(sym_closes)} symbols with data",
        flush=True,
    )

    if not pairs:
        print("[chronos-latents] no eligible (sym,day) pairs — abort", flush=True)
        return 1

    # Load model
    print(
        f"[chronos-latents] loading model {args.model_id} on {args.device} "
        f"dtype={args.torch_dtype}",
        flush=True,
    )
    pipe = _load_chronos_pipeline(
        model_id=args.model_id,
        device=args.device,
        torch_dtype=args.torch_dtype,
        model_revision=args.model_revision,
    )
    pipe_name = type(pipe).__name__
    model_commit_hash = _extract_model_commit_hash(pipe)

    # Probe embedding dim
    ctx0 = np.zeros((1, args.context_length), dtype=np.float32)
    e0 = _embed_batch(pipe, ctx0, args.pool)
    embed_dim = int(e0.shape[1])
    try:
        _validate_pca_components(
            n_components=args.n_components,
            embed_dim=embed_dim,
            n_rows=len(pairs),
        )
    except ValueError as exc:
        print(f"chronos-latents: {exc}", file=sys.stderr)
        return 2
    print(f"[chronos-latents] {pipe_name} embed dim={embed_dim}", flush=True)

    # Iterate in batches over the pairs list, building tensors row-by-row.
    B = int(args.batch_size)
    T = int(args.context_length)
    all_lat: list[np.ndarray] = []
    all_keys: list[tuple[str, date]] = []
    use_log = args.input_mode == "log_returns"

    t_start = time.perf_counter()
    n_done = 0
    for batch_start in range(0, len(pairs), B):
        chunk = pairs[batch_start : batch_start + B]
        ctx_arr = np.empty((len(chunk), T), dtype=np.float32)
        for j, (sym, d) in enumerate(chunk):
            i = sym_dates_idx[sym][d]
            ctx_arr[j] = _make_context(
                sym_closes[sym], i, T, return_log=use_log
            ).astype(np.float32)
        emb = _embed_batch(pipe, ctx_arr, args.pool)
        all_lat.append(emb)
        all_keys.extend(chunk)
        n_done += len(chunk)
        if (batch_start // B) % 25 == 0 or n_done == len(pairs):
            dt = time.perf_counter() - t_start
            rate = n_done / max(dt, 1e-6)
            eta = (len(pairs) - n_done) / max(rate, 1e-6)
            print(
                f"[chronos-latents] {n_done:>7}/{len(pairs):,}"
                f"  ({100*n_done/len(pairs):5.1f}%)"
                f"  rate={rate:6.0f}/s"
                f"  eta={eta/60:5.1f}m",
                flush=True,
            )

    raw = np.vstack(all_lat)  # (N, embed_dim)
    print(
        f"[chronos-latents] raw latents shape={raw.shape}  bytes={raw.nbytes/1e6:.1f} MB",
        flush=True,
    )

    # PCA-reduce
    pca_explained = None
    pca_components_path = None
    if args.n_components and args.n_components > 0 and args.n_components < embed_dim:
        from sklearn.decomposition import PCA  # noqa: PLC0415

        # Centered PCA. Fit on a subsample for speed (large N).
        n_fit = min(200_000, raw.shape[0])
        rng = np.random.default_rng(42)
        idx = rng.choice(raw.shape[0], size=n_fit, replace=False)
        print(
            f"[chronos-latents] fitting PCA(n_components={args.n_components}) "
            f"on {n_fit:,} rows of {raw.shape[1]}-D…",
            flush=True,
        )
        pca = PCA(n_components=args.n_components, random_state=42, svd_solver="auto")
        pca.fit(raw[idx])
        latents = pca.transform(raw).astype(np.float32)
        pca_explained = float(pca.explained_variance_ratio_.sum())
        # Save PCA components alongside parquet for predict-time symmetry.
        pca_components_path = args.output.with_suffix(".pca.npz")
        _write_npz_atomic(
            pca_components_path,
            mean=pca.mean_.astype(np.float32),
            components=pca.components_.astype(np.float32),
            explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
        )
        print(
            f"[chronos-latents] PCA fit: explained_var={pca_explained:.3f} → "
            f"{pca_components_path}",
            flush=True,
        )
    else:
        latents = raw.astype(np.float32)

    # Build a DataFrame with symbol, date, latent_0..K
    syms_col = [k[0] for k in all_keys]
    dates_col = [k[1] for k in all_keys]
    cols = {
        "symbol": syms_col,
        "date": dates_col,
    }
    K = latents.shape[1]
    for k in range(K):
        cols[f"latent_{k}"] = latents[:, k]
    df = pd.DataFrame(cols)
    _write_parquet_atomic(df, args.output)
    print(
        f"[chronos-latents] wrote {args.output}  rows={len(df):,}  K={K}  "
        f"size={args.output.stat().st_size/1e6:.1f}MB",
        flush=True,
    )

    # Sidecar manifest
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "model_commit_hash": model_commit_hash,
        "pipeline": pipe_name,
        "device": args.device,
        "torch_dtype": args.torch_dtype,
        "context_length": int(args.context_length),
        "input_mode": args.input_mode,
        "pool": args.pool,
        "raw_embed_dim": int(embed_dim),
        "n_components_after_pca": int(K),
        "pca_explained_variance_ratio": pca_explained,
        "pca_components_path": (
            _rel_to_repo(pca_components_path) if pca_components_path else None
        ),
        "n_rows": len(df),
        "n_symbols_with_data": int(df["symbol"].nunique()),
        "n_unique_dates": int(df["date"].nunique()),
        "start_date": str(start_date),
        "end_date": str(end_date),
        "symbols_file": _rel_to_repo(args.symbols_file),
        "data_root": _rel_to_repo(args.data_root),
        "wall_clock_seconds": round(time.perf_counter() - t_start, 1),
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    _write_json_atomic(manifest_path, manifest)
    print(f"[chronos-latents] manifest → {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
