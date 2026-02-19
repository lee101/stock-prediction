#!/usr/bin/env python3
"""Sweep smoothness/risk penalty configs to find smooth PnL with low DD."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from .run_cuda_ppo_residual_sweep import ResidualSweepConfig, run_sweep


CONFIGS = [
    # (name, downside_pen, pnl_smooth_pen, cap_smooth_pen, cap_floor, max_cap_change, total_steps)
    ("smooth_dd01_ps1e4", 0.1, 1e-4, 0.0, 0.0, None, 60_000),
    ("smooth_dd05_ps1e4", 0.5, 1e-4, 0.0, 0.0, None, 60_000),
    ("smooth_dd1_ps1e3", 1.0, 1e-3, 0.0, 0.0, None, 60_000),
    ("smooth_dd01_ps1e3_cs1e3", 0.1, 1e-3, 1e-3, 0.0, None, 60_000),
    ("smooth_dd05_ps5e4_cap03", 0.5, 5e-4, 0.0, 0.3, 0.1, 60_000),
    ("smooth_dd1_ps1e3_cap05", 1.0, 1e-3, 0.0, 0.5, 0.05, 60_000),
    ("smooth_dd2_ps5e3", 2.0, 5e-3, 0.0, 0.0, None, 60_000),
    ("smooth_dd01_ps0_cap05_cs1e3", 0.1, 0.0, 1e-3, 0.5, 0.1, 60_000),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="1337,2024")
    parser.add_argument("--total-timesteps", type=int, default=60_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--baseline-checkpoint", default="binanceleveragesui/checkpoints/lev5x_rw0.012_s1337/policy_checkpoint.pt")
    parser.add_argument("--configs", default="all", help="all or comma-sep indices")
    args = parser.parse_args()

    configs_to_run = CONFIGS
    if args.configs != "all":
        indices = [int(i) for i in args.configs.split(",")]
        configs_to_run = [CONFIGS[i] for i in indices]

    results = {}
    for name, dd_pen, ps_pen, cs_pen, cap_floor, max_cap_change, steps in configs_to_run:
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"  dd_pen={dd_pen}, ps_pen={ps_pen}, cs_pen={cs_pen}, cap_floor={cap_floor}, max_cap_change={max_cap_change}")
        print(f"{'='*60}")

        cfg = ResidualSweepConfig(
            downside_penalty=dd_pen,
            pnl_smoothness_penalty=ps_pen,
            leverage_cap_smoothness_penalty=cs_pen,
            cap_floor_ratio=cap_floor,
            max_cap_change_per_step=max_cap_change,
            total_timesteps=args.total_timesteps or steps,
            seeds=args.seeds,
            device=args.device,
            baseline_checkpoint=args.baseline_checkpoint,
            output=f"binanceleveragesui_rlcuda/results_smoothness_{name}.json",
            artifacts_root=f"binanceleveragesui_rlcuda/artifacts_smoothness_{name}",
            drawdown_weight_for_rank=0.5,
            sortino_weight_for_rank=0.001,
        )

        try:
            summary = run_sweep(cfg)
            out_path = Path(cfg.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(summary, indent=2))

            best = summary["best"]
            rl5 = best["rl_metrics"]["lev_5.0x"]
            bl5 = best["baseline_metrics"]["lev_5.0x"]
            results[name] = {
                "seed": best["seed"],
                "rl_5x_return": rl5["total_return"],
                "rl_5x_sortino": rl5["sortino"],
                "rl_5x_dd": rl5["max_drawdown"],
                "rl_5x_step_return_std": rl5.get("step_return_std", 0),
                "rl_5x_avg_cap": rl5.get("avg_cap_ratio", 0),
                "bl_5x_return": bl5["total_return"],
                "bl_5x_dd": bl5["max_drawdown"],
            }
            print(f"  BEST: seed={best['seed']} ret={rl5['total_return']:.3f} sort={rl5['sortino']:.0f} dd={rl5['max_drawdown']:.4f} step_std={rl5.get('step_return_std',0):.6f}")
        except Exception as e:
            print(f"  FAILED: {e}")
            results[name] = {"error": str(e)}

    summary_path = Path("binanceleveragesui_rlcuda/smoothness_sweep_summary.json")
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\n\nSummary saved: {summary_path}")
    print("\n=== RESULTS ===")
    for name, r in results.items():
        if "error" in r:
            print(f"  {name}: ERROR - {r['error']}")
        else:
            print(f"  {name}: ret={r['rl_5x_return']:.3f} sort={r['rl_5x_sortino']:.0f} dd={r['rl_5x_dd']:.4f} step_std={r['rl_5x_step_return_std']:.6f} cap={r['rl_5x_avg_cap']:.3f}")


if __name__ == "__main__":
    main()
