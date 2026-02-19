#!/usr/bin/env python3
"""Fine-grained sweep around best smoothness config (dd=0.1, ps=0, cap=0.5, cs=1e-3)."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from .run_cuda_ppo_residual_sweep import ResidualSweepConfig, run_sweep

CONFIGS = [
    # Vary cap_floor around 0.5
    ("cap04_cs1e3", 0.1, 0.0, 1e-3, 0.4, 0.1, 60_000),
    ("cap06_cs1e3", 0.1, 0.0, 1e-3, 0.6, 0.1, 60_000),
    ("cap07_cs1e3", 0.1, 0.0, 1e-3, 0.7, 0.1, 60_000),
    # Vary max_cap_change
    ("cap05_mc005", 0.1, 0.0, 1e-3, 0.5, 0.05, 60_000),
    ("cap05_mc02", 0.1, 0.0, 1e-3, 0.5, 0.2, 60_000),
    ("cap05_mcNone", 0.1, 0.0, 1e-3, 0.5, None, 60_000),
    # Vary cs_pen
    ("cap05_cs5e4", 0.1, 0.0, 5e-4, 0.5, 0.1, 60_000),
    ("cap05_cs5e3", 0.1, 0.0, 5e-3, 0.5, 0.1, 60_000),
    # Add small pnl smoothness
    ("cap05_ps1e4_cs1e3", 0.1, 1e-4, 1e-3, 0.5, 0.1, 60_000),
    # Higher dd penalty
    ("dd05_cap05_cs1e3", 0.5, 0.0, 1e-3, 0.5, 0.1, 60_000),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="1337,2024")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--baseline-checkpoint", default="binanceleveragesui/checkpoints/lev5x_rw0.012_s1337/policy_checkpoint.pt")
    parser.add_argument("--configs", default="all")
    args = parser.parse_args()

    configs_to_run = CONFIGS
    if args.configs != "all":
        indices = [int(i) for i in args.configs.split(",")]
        configs_to_run = [CONFIGS[i] for i in indices]

    results = {}
    for name, dd_pen, ps_pen, cs_pen, cap_floor, max_cap_change, steps in configs_to_run:
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"  dd={dd_pen}, ps={ps_pen}, cs={cs_pen}, cap_floor={cap_floor}, mc={max_cap_change}")
        print(f"{'='*60}")

        cfg = ResidualSweepConfig(
            downside_penalty=dd_pen, pnl_smoothness_penalty=ps_pen,
            leverage_cap_smoothness_penalty=cs_pen, cap_floor_ratio=cap_floor,
            max_cap_change_per_step=max_cap_change,
            total_timesteps=steps, seeds=args.seeds, device=args.device,
            baseline_checkpoint=args.baseline_checkpoint,
            output=f"binanceleveragesui_rlcuda/results_sweep2_{name}.json",
            artifacts_root=f"binanceleveragesui_rlcuda/artifacts_sweep2_{name}",
            drawdown_weight_for_rank=0.5, sortino_weight_for_rank=0.001,
        )

        try:
            summary = run_sweep(cfg)
            Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
            Path(cfg.output).write_text(json.dumps(summary, indent=2))
            best = summary["best"]
            rl5 = best["rl_metrics"]["lev_5.0x"]
            results[name] = {
                "seed": best["seed"], "ret": rl5["total_return"],
                "sortino": rl5["sortino"], "dd": rl5["max_drawdown"],
                "cap": rl5.get("avg_cap_ratio", 0),
            }
            print(f"  BEST: seed={best['seed']} ret={rl5['total_return']:.3f} sort={rl5['sortino']:.0f} dd={rl5['max_drawdown']:.4f}")
        except Exception as e:
            print(f"  FAILED: {e}")
            results[name] = {"error": str(e)}

    out = Path("binanceleveragesui_rlcuda/sweep2_summary.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSummary: {out}")
    print("\n=== RESULTS ===")
    for name, r in results.items():
        if "error" in r:
            print(f"  {name}: ERROR - {r['error']}")
        else:
            print(f"  {name}: ret={r['ret']:.3f} sort={r['sortino']:.0f} dd={r['dd']:.4f} cap={r['cap']:.3f}")


if __name__ == "__main__":
    main()
