#!/usr/bin/env python3
"""Generate Chronos-2 encoder latents for the XGB / RL daily universe.

Mirrors ``scripts.build_chronos_bolt_latents`` for the newer
``amazon/chronos-2`` model. For every (symbol, trading-day D) we feed the
previous ``context_length`` log-close values (strictly < D, no lookahead)
into the Chronos-2 encoder, pool the per-patch hidden states into a single
``d_model``-D vector, PCA-reduce across (symbol, date) to ``--n-components``
dims, and write the result to a parquet file with one row per (sym, date).

Chronos-2 differs from Chronos-Bolt in two ways relevant here:

1. The encoder is a T5-style Chronos2Encoder; the public pipeline does not
   expose ``embed()``. We capture the encoder's last hidden state via a
   forward hook on ``model.encoder``. The captured tensor has shape
   ``(B, num_context_patches + use_reg_token + num_output_patches, d_model)``.
   We discard the trailing future patches and pool the remainder.
2. Context can grow up to 8192 bars (vs 512 in Bolt-base) — useful for
   long daily histories. Default here is still 256 to match the existing
   Bolt parquet for A/B testing.

Pooling strategies
------------------
* ``--pool reg``     (default) take the [REG] token at index ``num_context_patches``.
                     This is a learned global summary, analogous to a [CLS] token,
                     and is what the decoder reads when generating future patches.
* ``--pool last``    take the last context patch (most recent slot).
* ``--pool mean``    mean over context patches (REG excluded).

Usage
-----

    .venv/bin/python -m scripts.build_chronos2_latents \
        --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
        --data-root trainingdata \
        --start-date 2019-09-01 \
        --end-date  2026-04-30 \
        --context-length 256 \
        --batch-size 256 \
        --pool reg \
        --n-components 32 \
        --output analysis/foundation_model_features/chronos2_base_latents.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
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


# Reuse the well-tested helpers from the Bolt builder verbatim. Only the
# embedding step and the pipeline loader change for Chronos-2.
from scripts.build_chronos_bolt_latents import (  # noqa: E402
    _atomic_path_write,
    _build_index_table,
    _extract_model_commit_hash,
    _load_csv,
    _load_symbols,
    _make_context,
    _rel_to_repo,
    _validate_pca_components,
    _validate_run_config,
    _write_json_atomic,
    _write_npz_atomic,
    _write_parquet_atomic,
)


logger = logging.getLogger("build_chronos2_latents")


def _load_chronos2_pipeline(
    *,
    model_id: str,
    device: str,
    torch_dtype: str,
    model_revision: str | None,
):
    """Load a Chronos2Pipeline with the requested device / dtype / revision.

    Returns the pipeline; callers should access ``pipeline.model`` for the
    encoder hook below.
    """
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


def _embed_batch_chronos2(
    pipe,
    contexts: np.ndarray,  # (B, T) raw closes (model handles instance norm)
    pool: str,
) -> np.ndarray:
    """Run the Chronos-2 encoder on a batch and pool to (B, d_model).

    We register a one-shot forward hook on ``pipe.model.encoder`` to capture
    the last hidden state (the encoder's full output before the decoder /
    quantile head touches it). The model is then called with
    ``num_output_patches=1`` so we only pay the encoder cost for one trailing
    future slot, which we slice away.

    Pooling positions in the captured tensor::

        [ context_patches | REG | future_patch ]
          0 ... NCP-1       NCP    NCP+1

    """
    model = pipe.model
    use_reg_token = bool(getattr(model.chronos_config, "use_reg_token", False))

    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output) -> None:
        # Chronos2EncoderOutput[0] is last_hidden_state. ModelOutput supports indexing.
        captured["hidden_states"] = output[0]

    handle = model.encoder.register_forward_hook(hook)
    try:
        ctx_t = torch.from_numpy(contexts).to(
            device=model.device, dtype=torch.float32
        )
        with torch.inference_mode():
            _ = model(context=ctx_t, num_output_patches=1)
    finally:
        handle.remove()

    hidden = captured.get("hidden_states")
    if hidden is None:
        raise RuntimeError(
            "Chronos2 encoder forward hook did not fire — model.encoder may have "
            "been replaced or compiled in a way that strips hooks."
        )

    # Drop the trailing num_output_patches=1 slot.
    hidden = hidden[:, :-1, :]
    if use_reg_token:
        ctx_part = hidden[:, :-1, :]  # context patches only
        reg_token = hidden[:, -1:, :]  # the [REG] summary
    else:
        ctx_part = hidden
        reg_token = None

    if pool == "reg":
        if reg_token is None:
            raise ValueError("pool='reg' requires the model to have use_reg_token=True")
        out = reg_token.squeeze(1)
    elif pool == "last":
        out = ctx_part[:, -1, :]
    elif pool == "mean":
        out = ctx_part.mean(dim=1)
    else:
        raise ValueError(f"unknown pool: {pool}")

    return out.float().cpu().numpy().astype(np.float32)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--symbols-file", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=REPO / "trainingdata")
    p.add_argument("--start-date", type=str, default="2019-09-01")
    p.add_argument("--end-date", type=str, default="2026-12-31")
    p.add_argument("--model-id", default="amazon/chronos-2")
    p.add_argument(
        "--model-revision",
        default=None,
        help="Optional immutable Hugging Face revision/commit to pin model loading.",
    )
    p.add_argument(
        "--device",
        default="cuda",
        help='"cuda" or "cpu"; chronos-2 base ~290MB, runs comfortably on GPU.',
    )
    p.add_argument(
        "--torch-dtype",
        default="bfloat16",
        choices=["bfloat16", "float32", "float16"],
    )
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument(
        "--pool",
        default="reg",
        choices=["reg", "last", "mean"],
        help="Encoder pooling strategy. 'reg' uses the [REG] summary token.",
    )
    p.add_argument(
        "--input-mode",
        default="raw_close",
        choices=["raw_close", "log_returns"],
        help="What to feed Chronos-2. raw_close passes raw price levels; "
        "the model's instance_norm handles scaling internally.",
    )
    p.add_argument(
        "--n-components",
        type=int,
        default=32,
        help="PCA components after pooling. 0 = keep raw d_model dims.",
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
        / "analysis/foundation_model_features/chronos2_base_latents.parquet",
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
        print(f"chronos2-latents: {exc}", file=sys.stderr)
        return 2

    syms = _load_symbols(args.symbols_file)
    if args.limit_symbols:
        syms = syms[: args.limit_symbols]
    print(f"[chronos2-latents] symbols: {len(syms)}", flush=True)

    pairs, sym_closes, sym_dates_idx = _build_index_table(
        syms,
        args.data_root,
        start_date=start_date,
        end_date=end_date,
        context_length=args.context_length,
    )
    print(
        f"[chronos2-latents] (sym,day) pairs to embed: {len(pairs):,} | "
        f"covering {len(sym_closes)} symbols with data",
        flush=True,
    )

    if not pairs:
        print("[chronos2-latents] no eligible (sym,day) pairs — abort", flush=True)
        return 1

    print(
        f"[chronos2-latents] loading model {args.model_id} on {args.device} "
        f"dtype={args.torch_dtype}",
        flush=True,
    )
    pipe = _load_chronos2_pipeline(
        model_id=args.model_id,
        device=args.device,
        torch_dtype=args.torch_dtype,
        model_revision=args.model_revision,
    )
    pipe_name = type(pipe).__name__
    model_commit_hash = _extract_model_commit_hash(pipe)

    # Probe embedding dim with a tiny zero context.
    ctx0 = np.zeros((1, args.context_length), dtype=np.float32)
    e0 = _embed_batch_chronos2(pipe, ctx0, args.pool)
    embed_dim = int(e0.shape[1])
    try:
        _validate_pca_components(
            n_components=args.n_components,
            embed_dim=embed_dim,
            n_rows=len(pairs),
        )
    except ValueError as exc:
        print(f"chronos2-latents: {exc}", file=sys.stderr)
        return 2
    print(f"[chronos2-latents] {pipe_name} encoder dim={embed_dim}", flush=True)

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
        emb = _embed_batch_chronos2(pipe, ctx_arr, args.pool)
        all_lat.append(emb)
        all_keys.extend(chunk)
        n_done += len(chunk)
        if (batch_start // B) % 25 == 0 or n_done == len(pairs):
            dt = time.perf_counter() - t_start
            rate = n_done / max(dt, 1e-6)
            eta = (len(pairs) - n_done) / max(rate, 1e-6)
            print(
                f"[chronos2-latents] {n_done:>7}/{len(pairs):,}"
                f"  ({100*n_done/len(pairs):5.1f}%)"
                f"  rate={rate:6.0f}/s"
                f"  eta={eta/60:5.1f}m",
                flush=True,
            )

    raw = np.vstack(all_lat)  # (N, embed_dim)
    print(
        f"[chronos2-latents] raw latents shape={raw.shape}  bytes={raw.nbytes/1e6:.1f} MB",
        flush=True,
    )

    pca_explained = None
    pca_components_path = None
    if args.n_components and args.n_components > 0 and args.n_components < embed_dim:
        from sklearn.decomposition import PCA  # noqa: PLC0415

        n_fit = min(200_000, raw.shape[0])
        rng = np.random.default_rng(42)
        idx = rng.choice(raw.shape[0], size=n_fit, replace=False)
        print(
            f"[chronos2-latents] fitting PCA(n_components={args.n_components}) "
            f"on {n_fit:,} rows of {raw.shape[1]}-D…",
            flush=True,
        )
        pca = PCA(n_components=args.n_components, random_state=42, svd_solver="auto")
        pca.fit(raw[idx])
        latents = pca.transform(raw).astype(np.float32)
        pca_explained = float(pca.explained_variance_ratio_.sum())
        pca_components_path = args.output.with_suffix(".pca.npz")
        _write_npz_atomic(
            pca_components_path,
            mean=pca.mean_.astype(np.float32),
            components=pca.components_.astype(np.float32),
            explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
        )
        print(
            f"[chronos2-latents] PCA fit: explained_var={pca_explained:.3f} → "
            f"{pca_components_path}",
            flush=True,
        )
    else:
        latents = raw.astype(np.float32)

    syms_col = [k[0] for k in all_keys]
    dates_col = [k[1] for k in all_keys]
    cols: dict[str, object] = {
        "symbol": syms_col,
        "date": dates_col,
    }
    K = latents.shape[1]
    for k in range(K):
        cols[f"latent_{k}"] = latents[:, k]
    df = pd.DataFrame(cols)
    _write_parquet_atomic(df, args.output)
    print(
        f"[chronos2-latents] wrote {args.output}  rows={len(df):,}  K={K}  "
        f"size={args.output.stat().st_size/1e6:.1f}MB",
        flush=True,
    )

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
    print(f"[chronos2-latents] manifest → {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
