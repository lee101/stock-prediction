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
import math
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
from xgbnew.artifacts import write_json_atomic  # noqa: E402
from xgbnew.cli_realism import validate_nonnegative_realism_args  # noqa: E402

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


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate-checkpoint", required=True)
    ap.add_argument("--val-data", default="pufferlib_market/data/screened32_single_offset_val_full.bin")
    ap.add_argument("--window-days", type=int, default=50)
    ap.add_argument("--fill-buffer-bps", type=float, default=5.0)
    ap.add_argument("--max-leverage", type=float, default=1.0)
    ap.add_argument("--fee-rate", type=float, default=0.001)
    ap.add_argument("--slippage-bps", type=float, default=5.0)
    ap.add_argument("--decision-lag", type=int, default=2)
    ap.add_argument(
        "--allow-low-lag-diagnostics",
        action="store_true",
        help="Allow lag 0/1 diagnostic runs; not production-realistic.",
    )
    ap.add_argument("--disable-shorts", action="store_true", default=True)
    ap.add_argument("--no-disable-shorts", dest="disable_shorts", action="store_false")
    ap.add_argument("--device", default=None, help="Torch device. Defaults to cuda when available, else cpu.")
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
    return ap


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def validate_args(args: argparse.Namespace) -> list[str]:
    failures = validate_nonnegative_realism_args(
        args,
        fields=(
            ("fee_rate", "fee_rate"),
            ("fill_buffer_bps", "fill_buffer_bps"),
            ("slippage_bps", "slippage_bps"),
        ),
    )
    try:
        max_leverage = float(args.max_leverage)
    except (TypeError, ValueError):
        failures.append("max_leverage must be finite and positive")
    else:
        if not math.isfinite(max_leverage) or max_leverage <= 0.0:
            failures.append("max_leverage must be finite and positive")
    if int(args.window_days) <= 0:
        failures.append("window_days must be positive")
    if int(args.decision_lag) < 0:
        failures.append("decision_lag must be non-negative")
    elif int(args.decision_lag) < 2 and not bool(args.allow_low_lag_diagnostics):
        failures.append("decision_lag below 2 requires --allow-low-lag-diagnostics")
    for attr, label in (
        ("corr_threshold", "corr_threshold"),
        ("jaccard_threshold", "jaccard_threshold"),
    ):
        value = getattr(args, attr, None)
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            failures.append(f"{label} must be finite")
            continue
        if not math.isfinite(value_f):
            failures.append(f"{label} must be finite")
    return failures


def _float_matches(payload: dict, key: str, expected: float) -> bool:
    try:
        observed = float(payload[key])
    except (KeyError, TypeError, ValueError):
        return False
    return math.isfinite(observed) and abs(observed - float(expected)) <= 1e-12


def baseline_cache_matches_config(
    payload: dict,
    *,
    args: argparse.Namespace,
    base_ckpts: Sequence[Path],
    start_indices: Sequence[int],
) -> bool:
    if payload.get("window_start_indices") != list(start_indices):
        return False
    if payload.get("checkpoints") != [str(c) for c in base_ckpts]:
        return False
    if bool(payload.get("disable_shorts")) != bool(args.disable_shorts):
        return False
    if int(payload.get("decision_lag", -1)) != int(args.decision_lag):
        return False
    if int(payload.get("window_days", -1)) != int(args.window_days):
        return False
    for key, expected in (
        ("fill_buffer_bps", float(args.fill_buffer_bps)),
        ("max_leverage", float(args.max_leverage)),
        ("fee_rate", float(args.fee_rate)),
        ("slippage_bps", float(args.slippage_bps)),
    ):
        if not _float_matches(payload, key, expected):
            return False
    returns = payload.get("window_returns")
    return isinstance(returns, list) and len(returns) == len(start_indices)


def build_baseline_cache_payload(
    *,
    args: argparse.Namespace,
    base_ckpts: Sequence[Path],
    baseline_rets: list[float],
    start_indices: Sequence[int],
) -> dict:
    return {
        "checkpoints": [str(c) for c in base_ckpts],
        "window_returns": baseline_rets,
        "window_start_indices": list(start_indices),
        "fill_buffer_bps": float(args.fill_buffer_bps),
        "max_leverage": float(args.max_leverage),
        "decision_lag": int(args.decision_lag),
        "fee_rate": float(args.fee_rate),
        "slippage_bps": float(args.slippage_bps),
        "disable_shorts": bool(args.disable_shorts),
        "window_days": int(args.window_days),
    }


def build_result_payload(
    *,
    args: argparse.Namespace,
    cand_path: Path,
    val_path: Path,
    base_med: float,
    cand_med: float,
    base_neg: int,
    cand_neg: int,
    corr: float,
    jaccard: float,
    overlap: list[int],
    only_baseline: list[int],
    only_cand: list[int],
    cand_rets: list[float],
    verdict: str,
) -> dict:
    return {
        "candidate": str(cand_path),
        "val_data": str(val_path),
        "window_days": int(args.window_days),
        "fill_buffer_bps": float(args.fill_buffer_bps),
        "max_leverage": float(args.max_leverage),
        "decision_lag": int(args.decision_lag),
        "fee_rate": float(args.fee_rate),
        "slippage_bps": float(args.slippage_bps),
        "disable_shorts": bool(args.disable_shorts),
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
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    validation_failures = validate_args(args)
    if validation_failures:
        for failure in validation_failures:
            print(f"diversity_screen: {failure}", file=sys.stderr)
        return 2

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
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    base_ckpts = [Path(DEFAULT_CHECKPOINT), *(Path(p) for p in DEFAULT_EXTRA_CHECKPOINTS)]
    abs_base = [REPO / c for c in base_ckpts]

    cache_path = out_dir / (
        f"baseline_returns_{val_path.stem}_w{args.window_days}_fb{int(args.fill_buffer_bps)}"
        f"_lev{args.max_leverage}_lag{args.decision_lag}.json"
    )
    if cache_path.exists() and not args.rebuild_baseline:
        print(f"loading cached baseline returns from {cache_path}")
        baseline_payload = json.loads(cache_path.read_text())
        if baseline_cache_matches_config(
            baseline_payload,
            args=args,
            base_ckpts=base_ckpts,
            start_indices=start_indices,
        ):
            baseline_rets = baseline_payload["window_returns"]
        else:
            print("cached baseline config does not match requested run; re-computing")
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
        write_json_atomic(
            cache_path,
            build_baseline_cache_payload(
                args=args,
                base_ckpts=base_ckpts,
                baseline_rets=baseline_rets,
                start_indices=start_indices,
            ),
        )
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
    print(
        f"neg-window jaccard:             {jaccard:.3f}  "
        f"({'PASS' if pass_jacc else 'FAIL'} < {args.jaccard_threshold})"
    )
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
    write_json_atomic(
        out_path,
        build_result_payload(
            args=args,
            cand_path=cand_path,
            val_path=val_path,
            base_med=base_med,
            cand_med=cand_med,
            base_neg=base_neg,
            cand_neg=cand_neg,
            corr=corr,
            jaccard=jaccard,
            overlap=overlap,
            only_baseline=only_baseline,
            only_cand=only_cand,
            cand_rets=cand_rets,
            verdict=verdict,
        ),
    )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
