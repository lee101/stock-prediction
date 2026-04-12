"""End-to-end 100-day unseen-data eval harness.

Loads an fp4 / pufferlib_market checkpoint, runs it through the C marketsim
on rolling windows of N unseen days each, aggregates per-window
returns/sortino/max_dd, estimates annualised monthly return, and asserts
the configured target (default: 27%/month).

Emits two sibling artifacts next to the checkpoint:

  - ``<ckpt_stem>_eval100d.json`` — raw per-slippage summaries + aggregate
  - ``<ckpt_stem>_eval100d.md``   — one-page markdown for the leaderboard

Usage::

    python scripts/eval_100d.py \
        --checkpoint pufferlib_market/checkpoints/stocks12_v5_rsi/tp05_s42/best.pt \
        --val-data pufferlib_market/data/stocks12_daily_v5_rsi_val.bin \
        --n-windows 30 --window-days 100 --monthly-target 0.27

The C marketsim is the ground truth (binary fills, realistic fees +
slippage). For policies that don't fit the marketsim shape (e.g. fp4
trainers on ``gpu_trading_env``) the script falls back to
``fp4.bench.eval_generic._same_backend_eval`` so the pipeline still
produces ranked output.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _monthly_from_total(total_return: float, window_days: int,
                        trading_days_per_month: float = 21.0) -> float:
    """Annualise a total window return into a per-month compound rate.

    monthly = (1 + total) ** (trading_days_per_month / window_days) - 1
    """
    if window_days <= 0:
        return 0.0
    try:
        return math.expm1(math.log1p(float(total_return)) * (trading_days_per_month / float(window_days)))
    except Exception:
        return 0.0


def _aggregate(summaries_by_bps: Dict[str, Any], window_days: int) -> Dict[str, Any]:
    """Flatten {bps: {...}} into a single aggregate dict.

    For each slippage cell we compute the implied median monthly return
    from ``median_return`` (a per-window total), then return the minimum
    across slippages (conservative: "does the worst slip still meet target?").
    """
    out: Dict[str, Any] = {"by_slippage": {}, "worst_slip_monthly": None}
    worst: float | None = None
    for bps, cell in summaries_by_bps.items():
        # Both _run_slippage_sweep and _same_backend_eval emit slightly
        # different shapes; normalise here.
        summary = cell.get("summary") or cell
        median_ret = float(summary.get("median_return", 0.0))
        p10_ret = float(summary.get("p10_return", 0.0))
        mean_ret = float(summary.get("mean_return", 0.0))
        median_sortino = float(summary.get("sortino", 0.0))
        median_dd = float(summary.get("max_drawdown", 0.0))
        n = int(summary.get("n_windows", 0))
        n_neg = int(summary.get("n_neg", 0))
        monthly = _monthly_from_total(median_ret, window_days)
        p10_monthly = _monthly_from_total(p10_ret, window_days)
        cell_out = {
            "median_total_return": median_ret,
            "p10_total_return": p10_ret,
            "mean_total_return": mean_ret,
            "median_monthly_return": monthly,
            "p10_monthly_return": p10_monthly,
            "median_sortino": median_sortino,
            "median_max_drawdown": median_dd,
            "n_windows": n,
            "n_negative_windows": n_neg,
        }
        out["by_slippage"][str(bps)] = cell_out
        if worst is None or monthly < worst:
            worst = monthly
    out["worst_slip_monthly"] = float(worst if worst is not None else 0.0)
    return out


def _render_md(
    ckpt: Path,
    aggregate: Dict[str, Any],
    window_days: int,
    n_windows: int,
    target_monthly: float,
    eval_result: Dict[str, Any],
) -> str:
    worst_m = float(aggregate.get("worst_slip_monthly", 0.0))
    ok = "PASS" if worst_m >= target_monthly else "FAIL"
    lines: List[str] = []
    lines.append(f"# 100d unseen-data eval — `{ckpt.name}`")
    lines.append("")
    lines.append(f"- **status**: {ok}  ({worst_m * 100:.2f}%/month vs target {target_monthly * 100:.2f}%/month)")
    lines.append(f"- windows: {n_windows} × {window_days}d  (total {n_windows * window_days}d unseen)")
    lines.append(f"- backend: {eval_result.get('backend', 'pufferlib_market')}")
    if 'shape_mismatch' in eval_result:
        lines.append(f"- shape_mismatch: `{eval_result['shape_mismatch']}`")
    if 'videos' in eval_result and isinstance(eval_result['videos'], dict):
        vids = eval_result['videos']
        if 'mp4' in vids:
            lines.append(f"- video: `{vids['mp4']}`")
        if 'html' in vids:
            lines.append(f"- scrubber: `{vids['html']}`")
    lines.append("")
    lines.append("| slip_bps | median total | median monthly | p10 total | p10 monthly | sortino | max dd | n_neg |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for bps, cell in sorted(aggregate["by_slippage"].items(), key=lambda kv: int(kv[0])):
        lines.append(
            f"| {bps} "
            f"| {cell['median_total_return'] * 100:+.2f}% "
            f"| {cell['median_monthly_return'] * 100:+.2f}% "
            f"| {cell['p10_total_return'] * 100:+.2f}% "
            f"| {cell['p10_monthly_return'] * 100:+.2f}% "
            f"| {cell['median_sortino']:.2f} "
            f"| {cell['median_max_drawdown'] * 100:.2f}% "
            f"| {cell['n_negative_windows']}/{cell['n_windows']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _slice_mktd_window(data, *, start: int, steps: int):
    from pufferlib_market.hourly_replay import MktdData

    end = int(start) + int(steps) + 1
    if start < 0 or end > data.num_timesteps:
        raise ValueError(f"Invalid MKTD slice start={start} steps={steps} for T={data.num_timesteps}")
    tradable = None if data.tradable is None else data.tradable[start:end].copy()
    return MktdData(
        version=int(data.version),
        symbols=list(data.symbols),
        features=data.features[start:end].copy(),
        prices=data.prices[start:end].copy(),
        tradable=tradable,
    )


def _summarise_window_metrics(rows: List[Dict[str, float]]) -> Dict[str, Any]:
    if not rows:
        return {"error": "no windows completed"}
    returns = np.asarray([float(r["total_return"]) for r in rows], dtype=np.float64)
    sortinos = np.asarray([float(r["sortino"]) for r in rows], dtype=np.float64)
    maxdds = np.asarray([float(r["max_drawdown"]) for r in rows], dtype=np.float64)
    return {
        "p10_return": float(np.percentile(returns, 10)),
        "median_return": float(np.percentile(returns, 50)),
        "p90_return": float(np.percentile(returns, 90)),
        "mean_return": float(np.mean(returns)),
        "sortino": float(np.median(sortinos)),
        "max_drawdown": float(np.median(maxdds)),
        "n_neg": int(np.sum(returns < 0.0)),
        "n_windows": int(returns.size),
    }


def _evaluate_intrabar_hourly(
    *,
    checkpoint: Path,
    val_data: Path,
    hourly_data_root: Path,
    daily_start_date: str,
    n_windows: int,
    window_days: int,
    slippages: List[int],
    fee_rate: float,
    max_leverage: float,
    seed: int,
    fill_buffer_bps: float,
    max_hold_hours: int,
    stop_loss_pct: float,
    take_profit_pct: float,
    trade_hour_mode: str,
    fail_fast: bool,
    fail_fast_max_dd: float,
    fail_fast_min_completed: int,
):
    import torch

    from pufferlib_market.evaluate_multiperiod import load_policy, make_policy_fn
    from pufferlib_market.hourly_replay import read_mktd
    from pufferlib_market.intrabar_replay import load_hourly_ohlc, simulate_daily_policy_intrabar

    data = read_mktd(val_data)
    num_symbols = data.num_symbols
    features_per_sym = int(data.features.shape[2])
    loaded = load_policy(
        str(checkpoint),
        num_symbols,
        arch="auto",
        hidden_size=None,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        features_per_sym=features_per_sym,
    )

    full_start_day = np.datetime64(str(daily_start_date))
    full_end_day = full_start_day + np.timedelta64(data.num_timesteps - 1, "D")
    hourly = load_hourly_ohlc(
        data.symbols,
        hourly_data_root,
        start=f"{str(full_start_day)} 00:00",
        end=f"{str(full_end_day)} 23:00",
    )

    window_len = int(window_days) + 1
    max_offset = data.num_timesteps - window_len
    if max_offset < 0:
        return {"status": "skip", "reason": f"val data too short: {data.num_timesteps} < {window_len}"}

    rng = np.random.default_rng(int(seed))
    starts = rng.choice(max_offset + 1, size=int(n_windows), replace=(max_offset + 1 < int(n_windows)))
    by_slippage: Dict[str, Any] = {}

    for bps in slippages:
        cell_rows: List[Dict[str, float]] = []
        failed_fast_reason: str | None = None
        for idx, start in enumerate(starts.tolist()):
            window_data = _slice_mktd_window(data, start=int(start), steps=int(window_days))
            policy_fn = make_policy_fn(
                loaded.policy,
                num_symbols=num_symbols,
                deterministic=True,
                decision_lag=2,
                device=next(loaded.policy.parameters()).device,
            )
            window_start_day = full_start_day + np.timedelta64(int(start), "D")
            result = simulate_daily_policy_intrabar(
                data=window_data,
                policy_fn=policy_fn,
                hourly=hourly,
                start_date=str(window_start_day),
                max_steps=int(window_days),
                fee_rate=float(fee_rate) + float(bps) / 10_000.0,
                fill_buffer_bps=float(fill_buffer_bps),
                max_leverage=float(max_leverage),
                stop_loss_pct=(float(stop_loss_pct) if stop_loss_pct > 0.0 else None),
                take_profit_pct=(float(take_profit_pct) if take_profit_pct > 0.0 else None),
                max_hold_hours=(int(max_hold_hours) if max_hold_hours > 0 else None),
                periods_per_year=8760.0,
                action_allocation_bins=loaded.action_allocation_bins,
                action_level_bins=loaded.action_level_bins,
                action_max_offset_bps=loaded.action_max_offset_bps,
                trade_hour_mode=trade_hour_mode,
            )
            rec = {
                "total_return": float(result.total_return),
                "sortino": float(result.sortino),
                "max_drawdown": float(result.max_drawdown),
            }
            cell_rows.append(rec)
            if fail_fast:
                if rec["max_drawdown"] > float(fail_fast_max_dd):
                    failed_fast_reason = f"window {idx} max_drawdown={rec['max_drawdown']:.3f} > {float(fail_fast_max_dd):.3f}"
                    break
                if rec["total_return"] < 0.0 and len(cell_rows) >= int(fail_fast_min_completed):
                    failed_fast_reason = f"window {idx} total_return={rec['total_return']:.4f} < 0 after {len(cell_rows)} completed"
                    break

        cell = _summarise_window_metrics(cell_rows)
        if failed_fast_reason is not None:
            cell["failed_fast"] = True
            cell["failed_reason"] = failed_fast_reason
            cell["n_completed_before_bail"] = int(len(cell_rows))
            by_slippage[str(int(bps))] = cell
            return {
                "status": "failed_fast",
                "backend": "pufferlib_market_intrabar_hourly",
                "failed_reason": failed_fast_reason,
                "by_slippage": by_slippage,
            }
        by_slippage[str(int(bps))] = cell

    return {"status": "ok", "backend": "pufferlib_market_intrabar_hourly", "by_slippage": by_slippage}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--val-data", required=True,
                    help="Path to pufferlib_market .bin. For fp4 checkpoints that "
                         "don't match this shape the script falls back to "
                         "same-backend eval and keeps going.")
    ap.add_argument("--n-windows", type=int, default=30)
    ap.add_argument("--window-days", type=int, default=100)
    ap.add_argument("--slippage-bps", default="0,5,10,20",
                    help="Comma-separated slippage levels in bps")
    ap.add_argument("--fee-rate", type=float, default=0.001)
    ap.add_argument("--max-leverage", type=float, default=1.5)
    ap.add_argument(
        "--execution-granularity",
        choices=["daily", "hourly_intrabar"],
        default="daily",
        help="Use the default C daily marketsim or replay daily decisions through hourly OHLC execution.",
    )
    ap.add_argument("--hourly-data-root", default=None,
                    help="Required for --execution-granularity hourly_intrabar. Root with stocks/ and/or crypto/ hourly CSVs.")
    ap.add_argument("--daily-start-date", default=None,
                    help="Required for --execution-granularity hourly_intrabar. UTC start date of row 0 in the MKTD file.")
    ap.add_argument("--hourly-fill-buffer-bps", type=float, default=5.0,
                    help="Hourly limit fill-through buffer for hourly_intrabar mode.")
    ap.add_argument("--hourly-max-hold-hours", type=int, default=0,
                    help="Optional hourly max-hold guard in hourly_intrabar mode.")
    ap.add_argument("--hourly-stop-loss-pct", type=float, default=0.0,
                    help="Optional intrabar stop-loss fraction in hourly_intrabar mode.")
    ap.add_argument("--hourly-take-profit-pct", type=float, default=0.0,
                    help="Optional intrabar take-profit fraction in hourly_intrabar mode.")
    ap.add_argument("--hourly-trade-hour-mode", choices=["first_tradable", "last_tradable"], default="first_tradable",
                    help="When to query the daily policy inside each day for hourly_intrabar mode.")
    ap.add_argument("--monthly-target", type=float, default=0.27,
                    help="Minimum acceptable median monthly return (worst slip).")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--video", action="store_true",
                    help="Render an MP4+HTML of the rollout when running the "
                         "same-backend fallback.")
    ap.add_argument("--no-fail-fast", dest="fail_fast", action="store_false",
                    help="Disable the early-exit on >max-dd or negative-window "
                         "checks (default: enabled).")
    ap.set_defaults(fail_fast=True)
    ap.add_argument("--fail-fast-max-dd", type=float, default=0.20,
                    help="Bail any sweep cell whose worst window drawdown "
                         "exceeds this fraction (default 0.20 = 20%%).")
    ap.add_argument("--fail-fast-min-completed", type=int, default=3,
                    help="Min completed windows before the negative-window "
                         "check fires (default 3).")
    ap.add_argument("--out-dir", default=None,
                    help="Where to write the JSON + MD. Defaults to the ckpt dir.")
    args = ap.parse_args(argv)

    ckpt = Path(args.checkpoint).resolve()
    val = Path(args.val_data).resolve()
    if not ckpt.exists():
        print(f"eval_100d: checkpoint not found: {ckpt}", file=sys.stderr)
        return 2
    if not val.exists():
        print(f"eval_100d: val data not found: {val}", file=sys.stderr)
        return 2
    slippages = [int(x) for x in args.slippage_bps.split(",") if x.strip()]

    # Build a cfg shaped like fp4_ppo_stocks12.yaml so evaluate_policy_file
    # reads the knobs it already knows.
    cfg = {
        "env": {
            "val_data": str(val.relative_to(REPO)) if val.is_relative_to(REPO) else str(val),
            "fee_rate": float(args.fee_rate),
            "max_leverage_scalar_fallback": float(args.max_leverage),
        },
        "eval": {
            "slippage_bps": slippages,
            "n_windows": int(args.n_windows),
            "eval_hours": int(args.window_days),
            "seed": int(args.seed),
            "video": bool(args.video),
            "fail_fast": bool(args.fail_fast),
            "fail_fast_max_dd": float(args.fail_fast_max_dd),
            "fail_fast_min_completed": int(args.fail_fast_min_completed),
        },
    }

    if args.execution_granularity == "hourly_intrabar":
        if not args.hourly_data_root:
            print("eval_100d: --hourly-data-root is required for hourly_intrabar", file=sys.stderr)
            return 2
        if not args.daily_start_date:
            print("eval_100d: --daily-start-date is required for hourly_intrabar", file=sys.stderr)
            return 2
        result = _evaluate_intrabar_hourly(
            checkpoint=ckpt,
            val_data=val,
            hourly_data_root=Path(args.hourly_data_root).resolve(),
            daily_start_date=str(args.daily_start_date),
            n_windows=int(args.n_windows),
            window_days=int(args.window_days),
            slippages=slippages,
            fee_rate=float(args.fee_rate),
            max_leverage=float(args.max_leverage),
            seed=int(args.seed),
            fill_buffer_bps=float(args.hourly_fill_buffer_bps),
            max_hold_hours=int(args.hourly_max_hold_hours),
            stop_loss_pct=float(args.hourly_stop_loss_pct),
            take_profit_pct=float(args.hourly_take_profit_pct),
            trade_hour_mode=str(args.hourly_trade_hour_mode),
            fail_fast=bool(args.fail_fast),
            fail_fast_max_dd=float(args.fail_fast_max_dd),
            fail_fast_min_completed=int(args.fail_fast_min_completed),
        )
    else:
        from fp4.bench.eval_generic import evaluate_policy_file
        result = evaluate_policy_file(ckpt, cfg, REPO)

    if result.get("status") not in ("ok", "failed_fast"):
        print(f"eval_100d: evaluate_policy_file returned status={result.get('status')}: "
              f"{result.get('reason', '<no reason>')}", file=sys.stderr)
        out_dir = Path(args.out_dir) if args.out_dir else ckpt.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{ckpt.stem}_eval100d.json").write_text(json.dumps(result, indent=2, default=str))
        return 1
    if result.get("status") == "failed_fast":
        # Still emit JSON + a short MD so the leaderboard can record the dud
        # without spending more time. No videos rendered downstream either —
        # the same-backend path already skips them on failed_fast.
        out_dir = Path(args.out_dir) if args.out_dir else ckpt.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        bail_md = (
            f"# 100d unseen-data eval — `{ckpt.name}`\n\n"
            f"- **status**: FAILED_FAST  ({result.get('failed_reason', '<no reason>')})\n"
            f"- backend: {result.get('backend', 'pufferlib_market')}\n"
        )
        (out_dir / f"{ckpt.stem}_eval100d.md").write_text(bail_md)
        (out_dir / f"{ckpt.stem}_eval100d.json").write_text(json.dumps({
            "checkpoint": str(ckpt), "val_data": str(val), "raw": result,
            "monthly_target": float(args.monthly_target),
            "n_windows": int(args.n_windows), "window_days": int(args.window_days),
            "slippage_bps": slippages,
        }, indent=2, default=str))
        print(bail_md)
        return 3

    by_slip = result.get("by_slippage", {})
    aggregate = _aggregate(by_slip, window_days=int(args.window_days))
    out_dir = Path(args.out_dir) if args.out_dir else ckpt.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    full = {
        "checkpoint": str(ckpt),
        "val_data": str(val),
        "n_windows": int(args.n_windows),
        "window_days": int(args.window_days),
        "slippage_bps": slippages,
        "monthly_target": float(args.monthly_target),
        "raw": result,
        "aggregate": aggregate,
    }
    (out_dir / f"{ckpt.stem}_eval100d.json").write_text(json.dumps(full, indent=2, default=str))
    md = _render_md(
        ckpt=ckpt, aggregate=aggregate,
        window_days=int(args.window_days), n_windows=int(args.n_windows),
        target_monthly=float(args.monthly_target),
        eval_result=result,
    )
    (out_dir / f"{ckpt.stem}_eval100d.md").write_text(md)
    print(md)

    worst_m = float(aggregate.get("worst_slip_monthly", 0.0))
    return 0 if worst_m >= float(args.monthly_target) else 3


if __name__ == "__main__":
    raise SystemExit(main())
