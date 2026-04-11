"""A/B benchmark: Gemini re-planner ON vs OFF.

Runs the same trainer twice per seed -- once with ``cfg['gemini_replanner']
= False`` (arm A, baseline) and once with ``= True`` (arm B).  If the live
Gemini planner (``fp4.fp4.gemini_planner``) isn't importable yet, arm B
records a skip but the report still generates.  If ``--live-llm`` is not
passed, arm B is forced to use a stub planner regardless of availability
so we never spend API quota from the A/B harness.

Computes monthly metrics via ``fp4.bench.monthly_metrics`` and writes
``fp4/bench/results/gemini_ab_<date>.md`` with:
  - per-arm/seed table
  - mean delta (B - A)
  - recommendation:
      * if B shows >= +5% mean monthly AND non-worse max_dd: recommend ON
      * else: recommend OFF (default) and note the API spend saving

CLI::

    python fp4/bench/bench_gemini_ab.py --steps 50000 --seeds 0,1 [--live-llm]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from fp4.bench.bench_trading import DEFAULT_CFG, RESULTS_DIR, _load_cfg, run_one  # noqa: E402
from fp4.bench.monthly_metrics import MONTHLY_TARGET, compute_monthly_metrics  # noqa: E402


def _planner_available() -> tuple[bool, str]:
    """Graceful check for P6-3's planner module.  Returns (available, reason)."""
    try:
        from fp4.fp4 import gemini_planner  # type: ignore  # noqa: F401
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _synthetic_equity_curve(seed: int, n_months: int = 6, bump: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic synthetic equity curve used when no real trainer curve
    is produced.  Keeps the A/B harness runnable end-to-end in CI even
    without a full training run.

    ``bump`` adds a per-month drift (e.g. +0.02 for arm B when testing
    the harness path under --live-llm with a stub planner).
    """
    g = torch.Generator().manual_seed(int(seed))
    samples_per_month = 24
    T = n_months * samples_per_month + 1
    # Random per-month returns around a 10% mean.
    base = 0.10 + bump
    month_rets = (0.10 * torch.randn(n_months, generator=g)) + base
    eq = [1.0]
    y, m = 2025, 1
    ts = [int(_dt.datetime(y, m, 1, tzinfo=_dt.timezone.utc).timestamp())]
    for i, r in enumerate(month_rets.tolist()):
        start = eq[-1]
        end = start * (1.0 + r)
        nm = m + 1
        ny = y
        if nm > 12:
            nm, ny = 1, y + 1
        start_ts = int(_dt.datetime(y, m, 1, tzinfo=_dt.timezone.utc).timestamp())
        end_ts = int(_dt.datetime(ny, nm, 1, tzinfo=_dt.timezone.utc).timestamp())
        for k in range(1, samples_per_month + 1):
            frac = k / samples_per_month
            eq.append(start + (end - start) * frac)
            ts.append(int(start_ts + (end_ts - start_ts) * frac))
        y, m = ny, nm
    return torch.tensor(eq).unsqueeze(0), torch.tensor(ts, dtype=torch.int64)


def _extract_curve(rec: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Attempt to extract (equity, timestamps) tensors from a trainer record.

    We look in a few places trainers might stash them; return None if not found.
    """
    train = rec.get("train") or {}
    for holder in (train, rec.get("eval") or {}, train.get("trainer_output") or {}):
        if not isinstance(holder, dict):
            continue
        eq = holder.get("equity_curve")
        ts = holder.get("timestamps")
        if eq is not None and ts is not None:
            try:
                return torch.as_tensor(eq, dtype=torch.float64), torch.as_tensor(ts, dtype=torch.int64)
            except Exception:
                continue
    return None


def _run_arm(trainer: str, cfg_path: Path, steps: int, seed: int,
             gemini_on: bool, live_llm: bool) -> dict[str, Any]:
    """Run one arm.  Loads cfg, sets ``gemini_replanner``, runs the trainer,
    then computes monthly metrics from whatever curve is available (falling
    back to a deterministic synthetic curve so the harness is testable
    without a full training run)."""
    cfg = _load_cfg(cfg_path)
    cfg["gemini_replanner"] = bool(gemini_on)
    cfg["gemini_live_llm"] = bool(live_llm)

    arm_name = "B_gemini_on" if gemini_on else "A_gemini_off"
    rec: dict[str, Any] = {"arm": arm_name, "seed": seed, "gemini_on": gemini_on}

    if gemini_on:
        avail, why = _planner_available()
        if not avail:
            rec["planner_available"] = False
            rec["skip_reason"] = f"gemini_planner not importable ({why}); P6-3 not yet committed"
        else:
            rec["planner_available"] = True
            if not live_llm:
                rec["planner_mode"] = "stub (no --live-llm)"
            else:
                rec["planner_mode"] = "live"
    else:
        rec["planner_available"] = True
        rec["planner_mode"] = "disabled"

    # Run the underlying trainer (smoke-ish; the fp4 trainer may skip if not built).
    try:
        t0 = time.perf_counter()
        train_rec = run_one(trainer, cfg_path, steps, seed, smoke=True)
        rec["wall_sec"] = time.perf_counter() - t0
        rec["trainer_status"] = train_rec.get("status", "unknown")
        rec["result_path"] = train_rec.get("result_path")
        curve = _extract_curve(train_rec)
    except Exception as exc:
        rec["trainer_status"] = "error"
        rec["trainer_error"] = f"{type(exc).__name__}: {exc}"
        curve = None

    if curve is None:
        # Fall back to deterministic synthetic curve. Arm B gets a small
        # deterministic bump only if stub/live planner path is active, so the
        # A/B report exercises the full comparison code path.
        bump = 0.0
        if gemini_on and rec.get("planner_available"):
            bump = 0.01  # 1% per month bump so we exercise the recommendation logic
        eq, ts = _synthetic_equity_curve(seed, n_months=6, bump=bump)
        rec["curve_source"] = "synthetic_fallback"
    else:
        eq, ts = curve
        rec["curve_source"] = "trainer"

    metrics = compute_monthly_metrics(eq, ts)
    rec["metrics"] = metrics
    return rec


def _format_row(r: dict[str, Any]) -> str:
    m = r.get("metrics") or {}
    return (
        f"| {r['arm']} | {r['seed']} | {r.get('curve_source','?')} | "
        f"{m.get('n_months',0)} | {m.get('mean_monthly',0.0):+.4f} | "
        f"{m.get('p10_monthly',0.0):+.4f} | {m.get('max_dd_monthly',0.0):+.4f} | "
        f"{m.get('hit_27pct',0.0):.3f} |"
    )


def _recommendation(a_rows: list[dict[str, Any]], b_rows: list[dict[str, Any]]) -> tuple[str, dict[str, float]]:
    def _agg(rows: list[dict[str, Any]], key: str) -> float:
        vals = [r["metrics"].get(key, 0.0) for r in rows if r.get("metrics")]
        return float(sum(vals) / max(1, len(vals)))

    a_mean = _agg(a_rows, "mean_monthly")
    b_mean = _agg(b_rows, "mean_monthly")
    a_dd = _agg(a_rows, "max_dd_monthly")
    b_dd = _agg(b_rows, "max_dd_monthly")
    delta_mean = b_mean - a_mean
    # "non-worse max_dd" means b_dd (a negative number) >= a_dd - tol.
    dd_non_worse = b_dd >= a_dd - 1e-6
    improve_5pct = delta_mean >= 0.05

    summary = {
        "a_mean_monthly": a_mean,
        "b_mean_monthly": b_mean,
        "delta_mean_monthly": delta_mean,
        "a_max_dd_monthly": a_dd,
        "b_max_dd_monthly": b_dd,
    }

    b_skipped = all(r.get("trainer_status") == "skip" or r.get("skip_reason") for r in b_rows) and all(
        r.get("curve_source") != "trainer" for r in b_rows
    )
    # Only meaningful if arm B actually ran.
    if not any(r.get("metrics", {}).get("n_months", 0) > 0 for r in b_rows):
        return ("INCONCLUSIVE: arm B produced no valid monthly metrics.", summary)

    if improve_5pct and dd_non_worse:
        verdict = (
            f"RECOMMEND gemini_replanner=True for production. "
            f"Arm B mean monthly {b_mean:+.4f} vs arm A {a_mean:+.4f} "
            f"(delta {delta_mean:+.4f} >= +0.05), "
            f"max_dd {b_dd:+.4f} non-worse than {a_dd:+.4f}."
        )
    else:
        reason = []
        if not improve_5pct:
            reason.append(f"delta {delta_mean:+.4f} < +0.05")
        if not dd_non_worse:
            reason.append(f"max_dd {b_dd:+.4f} worse than {a_dd:+.4f}")
        verdict = (
            f"RECOMMEND gemini_replanner=False (default). "
            f"{'; '.join(reason)}. "
            f"Save Gemini API spend in marketsim runs."
        )
    if b_skipped:
        verdict = (
            "NOTE: arm B used synthetic/stub fallback because P6-3 gemini_planner "
            "was not yet committed. " + verdict
        )
    return verdict, summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--trainer", default="fp4")
    p.add_argument("--config", default=str(DEFAULT_CFG))
    p.add_argument("--steps", type=int, default=50_000)
    p.add_argument("--seeds", default="0,1", help="comma-separated seed list")
    p.add_argument("--live-llm", action="store_true",
                   help="allow arm B to actually hit the Gemini API (costs money)")
    args = p.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    cfg_path = Path(args.config)
    all_rows: list[dict[str, Any]] = []
    a_rows: list[dict[str, Any]] = []
    b_rows: list[dict[str, Any]] = []
    for seed in seeds:
        a = _run_arm(args.trainer, cfg_path, args.steps, seed, gemini_on=False, live_llm=False)
        b = _run_arm(args.trainer, cfg_path, args.steps, seed, gemini_on=True, live_llm=args.live_llm)
        all_rows += [a, b]
        a_rows.append(a)
        b_rows.append(b)

    verdict, summary = _recommendation(a_rows, b_rows)

    date = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"gemini_ab_{date}.md"
    lines: list[str] = []
    lines.append(f"# Gemini A/B Benchmark -- {date}")
    lines.append("")
    lines.append(f"- trainer: `{args.trainer}`")
    lines.append(f"- config: `{cfg_path}`")
    lines.append(f"- steps: {args.steps}")
    lines.append(f"- seeds: {seeds}")
    lines.append(f"- live_llm: {args.live_llm}")
    lines.append(f"- monthly_target: {MONTHLY_TARGET:.2f}")
    lines.append("")
    lines.append("## Per-arm/seed metrics")
    lines.append("")
    lines.append("| arm | seed | curve | n_months | mean_monthly | p10_monthly | max_dd_monthly | hit_27pct |")
    lines.append("|-----|------|-------|----------|--------------|-------------|----------------|-----------|")
    for r in all_rows:
        lines.append(_format_row(r))
    lines.append("")
    lines.append("## Aggregate summary")
    lines.append("")
    for k, v in summary.items():
        lines.append(f"- **{k}**: {v:+.6f}")
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    lines.append(verdict)
    lines.append("")
    lines.append("## Raw records")
    lines.append("")
    lines.append("```json")
    # Strip non-serialisable entries (tensors already converted in compute_monthly_metrics).
    serialisable = []
    for r in all_rows:
        s = {k: v for k, v in r.items() if k != "trainer_output"}
        serialisable.append(s)
    lines.append(json.dumps(serialisable, indent=2, default=str))
    lines.append("```")
    out_path.write_text("\n".join(lines))
    print(f"wrote {out_path}")
    print(verdict)
    print("\nSample rows:")
    for r in all_rows[:2]:
        print(_format_row(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
