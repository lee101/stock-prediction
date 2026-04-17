"""Diversity pre-screen for new screened32 ensemble candidates.

CALIBRATION NOTE (2026-04-17): The simple corr/jaccard thresholds do NOT
reliably distinguish marginal-pass (AD_s4: 14m med +0.09%) from clear-reject
(AD_s9: 14m med −0.99%) candidates. Both score corr~0.29, jaccard~0.18 on
this ensemble. Mechanism: with 14 members averaged, even a strong-standalone
candidate dilutes existing decisive 13-vote consensus on borderline argmax
flips. The deploy-gate-quality pre-screen would need to actually run the
full 14m ensemble (which is the 14m gate itself).

This script remains useful as DIAGNOSTIC output: per-window correlation,
neg-window overlap, and unique loss windows tell you HOW a candidate
relates to the baseline. Do not use the binary verdict as a hard filter —
always run the full 14m gate before deciding.

Use cases that still work:
- "Is this candidate doing something completely different?" (corr near 0)
- "Does this candidate share the baseline's worst windows?" (overlap)
- "Where exactly does this candidate fail?" (per-window returns dump)

Usage::

    python scripts/screened32_diversity_screen.py \\
        --candidate-checkpoint pufferlib_market/checkpoints/screened32_sweep/AD/s9/best.pt \\
        --val-data pufferlib_market/data/screened32_single_offset_val_full.bin

The first run caches baseline per-window returns at
docs/diversity_screen/baseline_returns.json so subsequent candidates only
need to compute themselves (~2 min). Force rebuild with --rebuild-baseline.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pufferlib_market.evaluate_holdout import _slice_window  # noqa: E402
from pufferlib_market.hourly_replay import read_mktd, simulate_daily_policy  # noqa: E402
from scripts.screened32_realism_gate import (  # noqa: E402
    _build_ensemble_policy_fn,
)
from src.daily_stock_defaults import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_EXTRA_CHECKPOINTS,
)


def compute_window_returns(
    *,
    data,
    checkpoints: Sequence[Path],
    num_symbols: int,
    features_per_sym: int,
    decision_lag: int,
    disable_shorts: bool,
    deterministic: bool,
    device: torch.device,
    fill_buffer_bps: float,
    max_leverage: float,
    fee_rate: float,
    slippage_bps: float,
    window_days: int,
    start_indices: Sequence[int],
    label: str = "",
) -> list[float]:
    policy_fn, reset_buffer, head = _build_ensemble_policy_fn(
        checkpoints=checkpoints,
        num_symbols=num_symbols,
        features_per_sym=features_per_sym,
        decision_lag=decision_lag,
        disable_shorts=disable_shorts,
        device=device,
        deterministic=deterministic,
    )
    rets: list[float] = []
    n = len(start_indices)
    for i, start in enumerate(start_indices):
        window = _slice_window(data, start=int(start), steps=int(window_days))
        reset_buffer()
        result = simulate_daily_policy(
            window,
            policy_fn,
            max_steps=int(window_days),
            fee_rate=float(fee_rate),
            slippage_bps=float(slippage_bps),
            max_leverage=float(max_leverage),
            periods_per_year=365.0,
            fill_buffer_bps=float(fill_buffer_bps),
            action_allocation_bins=int(head.action_allocation_bins),
            action_level_bins=int(head.action_level_bins),
            action_max_offset_bps=float(head.action_max_offset_bps),
            enable_drawdown_profit_early_exit=False,
        )
        rets.append(float(result.total_return))
        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"  [{label}] {i + 1}/{n} windows", flush=True)
    return rets


def _pearson(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    sd_a = arr_a.std()
    sd_b = arr_b.std()
    if sd_a == 0 or sd_b == 0:
        return 0.0
    return float(np.corrcoef(arr_a, arr_b)[0, 1])


def _neg_jaccard(a: list[float], b: list[float]) -> tuple[float, list[int], list[int], list[int]]:
    neg_a = {i for i, v in enumerate(a) if v < 0}
    neg_b = {i for i, v in enumerate(b) if v < 0}
    union = neg_a | neg_b
    inter = neg_a & neg_b
    only_a = sorted(neg_a - neg_b)
    only_b = sorted(neg_b - neg_a)
    overlap = sorted(inter)
    if not union:
        return 1.0, overlap, only_a, only_b
    return len(inter) / len(union), overlap, only_a, only_b


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate-checkpoint", required=True)
    ap.add_argument("--val-data", default="pufferlib_market/data/screened32_single_offset_val_full.bin")
    ap.add_argument("--window-days", type=int, default=50)
    ap.add_argument("--fill-buffer-bps", type=float, default=5.0)
    ap.add_argument("--max-leverage", type=float, default=1.0)
    ap.add_argument("--fee-rate", type=float, default=0.001)
    ap.add_argument("--slippage-bps", type=float, default=5.0)
    ap.add_argument("--decision-lag", type=int, default=2)
    ap.add_argument("--disable-shorts", action="store_true", default=True)
    ap.add_argument("--no-disable-shorts", dest="disable_shorts", action="store_false")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default="docs/diversity_screen")
    ap.add_argument("--rebuild-baseline", action="store_true",
                    help="Force re-compute the baseline 13-model per-window returns.")
    ap.add_argument(
        "--corr-threshold",
        type=float,
        default=0.50,
        help="PASS if pearson(candidate_returns, baseline_returns) < this.",
    )
    ap.add_argument(
        "--jaccard-threshold",
        type=float,
        default=0.40,
        help="PASS if jaccard(candidate_neg_starts, baseline_neg_starts) < this.",
    )
    args = ap.parse_args(argv)

    val_path = Path(args.val_data).resolve()
    if not val_path.exists():
        print(f"diversity_screen: val data not found: {val_path}", file=sys.stderr)
        return 2

    cand_path = Path(args.candidate_checkpoint)
    if not cand_path.is_absolute():
        cand_path = REPO / cand_path
    if not cand_path.exists():
        print(f"diversity_screen: candidate not found: {cand_path}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    data = read_mktd(val_path)
    num_symbols = int(data.num_symbols)
    features_per_sym = int(data.features.shape[2])
    window_len = int(args.window_days) + 1
    if window_len > data.num_timesteps:
        print(
            f"diversity_screen: val too short for window_days={args.window_days} "
            f"(T={data.num_timesteps})",
            file=sys.stderr,
        )
        return 2
    candidate_count = data.num_timesteps - window_len + 1
    start_indices = list(range(candidate_count))
    device = torch.device(args.device)

    base_ckpts = [Path(DEFAULT_CHECKPOINT), *(Path(p) for p in DEFAULT_EXTRA_CHECKPOINTS)]
    abs_base = [REPO / c for c in base_ckpts]

    cache_path = out_dir / f"baseline_returns_{val_path.stem}_w{args.window_days}_fb{int(args.fill_buffer_bps)}_lev{args.max_leverage}_lag{args.decision_lag}.json"
    if cache_path.exists() and not args.rebuild_baseline:
        print(f"loading cached baseline returns from {cache_path}")
        baseline_payload = json.loads(cache_path.read_text())
        baseline_rets = baseline_payload["window_returns"]
        if len(baseline_rets) != len(start_indices):
            print(
                f"cached baseline length {len(baseline_rets)} != expected "
                f"{len(start_indices)}; re-computing",
            )
            baseline_rets = None
    else:
        baseline_rets = None

    if baseline_rets is None:
        print(f"computing baseline 13-model per-window returns ({len(start_indices)} windows)...")
        baseline_rets = compute_window_returns(
            data=data,
            checkpoints=abs_base,
            num_symbols=num_symbols,
            features_per_sym=features_per_sym,
            decision_lag=int(args.decision_lag),
            disable_shorts=bool(args.disable_shorts),
            deterministic=True,
            device=device,
            fill_buffer_bps=float(args.fill_buffer_bps),
            max_leverage=float(args.max_leverage),
            fee_rate=float(args.fee_rate),
            slippage_bps=float(args.slippage_bps),
            window_days=int(args.window_days),
            start_indices=start_indices,
            label="baseline",
        )
        cache_path.write_text(json.dumps({
            "checkpoints": [str(c) for c in base_ckpts],
            "window_returns": baseline_rets,
            "window_start_indices": start_indices,
            "fill_buffer_bps": float(args.fill_buffer_bps),
            "max_leverage": float(args.max_leverage),
            "decision_lag": int(args.decision_lag),
            "fee_rate": float(args.fee_rate),
            "slippage_bps": float(args.slippage_bps),
            "window_days": int(args.window_days),
        }, indent=2))
        print(f"cached baseline returns to {cache_path}")

    print(f"computing candidate per-window returns: {cand_path.name}")
    cand_rets = compute_window_returns(
        data=data,
        checkpoints=[cand_path],
        num_symbols=num_symbols,
        features_per_sym=features_per_sym,
        decision_lag=int(args.decision_lag),
        disable_shorts=bool(args.disable_shorts),
        deterministic=True,
        device=device,
        fill_buffer_bps=float(args.fill_buffer_bps),
        max_leverage=float(args.max_leverage),
        fee_rate=float(args.fee_rate),
        slippage_bps=float(args.slippage_bps),
        window_days=int(args.window_days),
        start_indices=start_indices,
        label="candidate",
    )

    corr = _pearson(cand_rets, baseline_rets)
    # _neg_jaccard returns only_a, only_b in the order args were passed.
    jaccard, overlap, only_cand, only_baseline = _neg_jaccard(cand_rets, baseline_rets)

    base_med = float(np.percentile(baseline_rets, 50))
    cand_med = float(np.percentile(cand_rets, 50))
    base_neg = sum(1 for v in baseline_rets if v < 0)
    cand_neg = sum(1 for v in cand_rets if v < 0)

    pass_corr = corr < float(args.corr_threshold)
    pass_jacc = jaccard < float(args.jaccard_threshold)
    # The simple verdict is not reliable (see module docstring). Report it
    # but mark as advisory.
    verdict = "ADVISORY-PASS" if (pass_corr and pass_jacc) else "ADVISORY-FAIL"

    print()
    print(f"# Diversity screen: {cand_path.name}")
    print()
    print(f"baseline 13-model: med={base_med:+.4f}  n_neg={base_neg}/{len(baseline_rets)}")
    print(f"candidate alone:   med={cand_med:+.4f}  n_neg={cand_neg}/{len(cand_rets)}")
    print()
    print(f"per-window pearson correlation: {corr:+.3f}  ({'PASS' if pass_corr else 'FAIL'} < {args.corr_threshold})")
    print(f"neg-window jaccard:             {jaccard:.3f}  ({'PASS' if pass_jacc else 'FAIL'} < {args.jaccard_threshold})")
    print(f"  overlap (both lose):    {overlap}")
    print(f"  baseline-only neg:      {only_baseline}")
    print(f"  candidate-only neg:     {only_cand}")
    print()
    print(
        f"VERDICT: {verdict}  (advisory only — see module docstring; "
        f"always run full 14m gate before deciding)"
    )
    print()

    out_path = out_dir / f"{cand_path.stem}_diversity.json"
    out_path.write_text(json.dumps({
        "candidate": str(cand_path),
        "val_data": str(val_path),
        "window_days": int(args.window_days),
        "fill_buffer_bps": float(args.fill_buffer_bps),
        "max_leverage": float(args.max_leverage),
        "decision_lag": int(args.decision_lag),
        "baseline_median_total": base_med,
        "candidate_median_total": cand_med,
        "baseline_neg_count": base_neg,
        "candidate_neg_count": cand_neg,
        "pearson_correlation": corr,
        "neg_jaccard": jaccard,
        "neg_overlap_indices": overlap,
        "baseline_only_neg_indices": only_baseline,
        "candidate_only_neg_indices": only_cand,
        "candidate_window_returns": cand_rets,
        "verdict": verdict,
        "corr_threshold": float(args.corr_threshold),
        "jaccard_threshold": float(args.jaccard_threshold),
    }, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
